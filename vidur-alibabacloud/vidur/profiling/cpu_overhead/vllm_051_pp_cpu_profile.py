"""Profile real vLLM 0.5.1 async PP execution with separate phase labels."""
import asyncio
import contextvars
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import ray
from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.async_llm_engine import _AsyncLLMEngine, AsyncLLMEngine
from vllm.executor.ray_gpu_executor import RayGPUExecutorAsync
from vllm.executor.ray_utils import initialize_ray_cluster

from pp_profile_contract import aggregate_step, request_key, CPU_METRICS, bind_worker_keys


class ProfilingRayExecutor(RayGPUExecutorAsync):
    def _run_workers(self, method, *args, **kwargs):
        if method == "init_worker":
            self.driver_worker.execute_method("__setattr__", "worker_module_name",
                                              "vllm_051_pp_worker")
            ray.get([w.execute_method.remote("__setattr__", "worker_module_name",
                                            "vllm_051_pp_worker") for w in self.workers])
        return super()._run_workers(method, *args, **kwargs)


def mean(values):
    return statistics.fmean(values)


async def profile(args):
    config = EngineArgs(
        model=args.model, tokenizer=args.tokenizer or args.model,
        trust_remote_code=args.trust_remote_code, revision=args.revision,
        tokenizer_revision=args.tokenizer_revision,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        distributed_executor_backend="ray", dtype=args.dtype,
        load_format=args.load_format, seed=args.seed,
        max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager, disable_log_stats=True,
        disable_custom_all_reduce=True,
    ).create_engine_config()
    initialize_ray_cluster(config.parallel_config)
    serving = getattr(args, "profile_serving_loop", False)
    outer = None
    if serving:
        outer = AsyncLLMEngine(worker_use_ray=True, engine_use_ray=False,
            **config.to_dict(), executor_class=ProfilingRayExecutor,
            log_stats=False, log_requests=False)
        engine = outer.engine
    else:
        engine = _AsyncLLMEngine(**config.to_dict(), executor_class=ProfilingRayExecutor,
                                 log_stats=False)
    executor = engine.model_executor
    environment = executor._run_workers("pp_profile_environment")
    executor._run_workers("pp_profile_reserve_events",
                          max(2, args.repetitions) * args.decode_tokens + 16)
    node_aliases = {}
    rank_nodes = [node_aliases.setdefault(e["node_id"], len(node_aliases))
                  for e in sorted(environment, key=lambda e: e["rank"])]
    if len(node_aliases) > 1 and not args.network_transport:
        raise ValueError("Cross-node profiling requires --network-transport verified from NCCL logs")
    step_context = contextvars.ContextVar("cpu_profile_step", default=None)
    steps, step_observations = [], []
    last_step = {}
    enabled = False

    for index, scheduler in enumerate(engine.scheduler):
        original = scheduler.schedule
        def schedule(original=original, index=index):
            start = time.perf_counter_ns()
            result = original()
            current = step_context.get()
            metadata = result[0]
            if metadata and current is not None:
                current["request_ids"] = tuple(m.request_id for m in metadata)
            if metadata and current is not None:
                current["phase"] = "prefill" if metadata[0].is_prompt else "decode"
            if enabled and metadata:
                current.update()
                phase = "prefill" if metadata[0].is_prompt else "decode"
                if any(m.is_prompt != metadata[0].is_prompt for m in metadata):
                    raise RuntimeError("Mixed prefill/decode profiling is unsupported")
                current.update(phase=phase, batch_size=len(metadata),
                               prefill_tokens_per_request=(
                                   sum(m.token_chunk_size for m in metadata) / len(metadata)
                                   if phase == "prefill" else 0.0),
                               schedule_ms=(time.perf_counter_ns()-start)*1e-6)
            return result
        scheduler.schedule = schedule

    original_execute = executor.execute_model_async
    async def execute(req):
        if serving and req and req.seq_group_metadata_list:
            idle.clear()
        start = time.perf_counter_ns()
        result = await original_execute(req)
        current = step_context.get()
        if enabled and req and req.seq_group_metadata_list:
            current.update(key=request_key(req),
                           executor_ms=(time.perf_counter_ns()-start)*1e-6)
        return result
    executor.execute_model_async = execute
    original_process = engine._process_model_outputs
    def process_outputs(*a, **kw):
        start = time.perf_counter_ns()
        result = original_process(*a, **kw)
        current = step_context.get()
        if enabled and current is not None and "key" in current:
            current["process_outputs_ms"] = (time.perf_counter_ns()-start)*1e-6
            steps.append(current)
        return result
    engine._process_model_outputs = process_outputs

    original_step = engine.step_async
    async def timed_step(virtual_engine):
        current = {}
        token = step_context.set(current)
        start = time.perf_counter_ns()
        try:
            result = await original_step(virtual_engine)
        finally:
            step_context.reset(token)
        end = time.perf_counter_ns()
        if result:
            step_ms = (end - start) * 1e-6
            step_observations.append(dict(ms=step_ms, phase=current["phase"], request_ids=current.get("request_ids", ())))
            ids = current.get("request_ids", ())
            gap = 0.0
            previous = last_step.get(virtual_engine)
            if serving and previous and set(ids) & set(previous[1]):
                gap = (start - previous[0]) * 1e-6
            last_step[virtual_engine] = (end, ids)
            if enabled:
                remainder = step_ms - sum(current[k] for k in (
                    "executor_ms", "schedule_ms", "process_outputs_ms"))
                if remainder < -0.01:
                    raise RuntimeError("Driver timing regions overlap")
                current["engine_step_ms"] = step_ms + gap
                current["engine_bookkeeping_ms"] = max(0.0, remainder) + gap
                current["serving_loop_gap_ms"] = gap
        return result
    engine.step_async = timed_step
    if serving:
        original_add_request = engine._add_processed_request
        def pinned_add_request(*a, **kw):
            schedulers = engine.scheduler
            try:
                engine.scheduler = schedulers[:1]
                return original_add_request(*a, **kw)
            finally:
                engine.scheduler = schedulers
        engine._add_processed_request = pinned_add_request

    idle = asyncio.Event()
    if serving:
        original_stop = engine.stop_remote_worker_execution_loop_async
        async def mark_idle():
            await original_stop()
            idle.set()
        engine.stop_remote_worker_execution_loop_async = mark_idle

    request_id = 0
    async def run_batch(batch_size, decode_tokens):
        nonlocal request_id
        observations_start = len(step_observations)
        outputs = []
        params = SamplingParams(temperature=0, ignore_eos=True, max_tokens=decode_tokens)
        if serving:
            idle.clear()
            async def consume(request):
                async for result in outer.generate(
                    {"prompt_token_ids": [42] * args.prompt_tokens}, params, request):
                    outputs.append(result)
            request_ids = [str(request_id + i) for i in range(batch_size)]
            request_id += batch_size
            await asyncio.gather(*(consume(request) for request in request_ids))
            await idle.wait()
        else:
            schedulers = engine.scheduler
            try:
                engine.scheduler = schedulers[:1]
                for _ in range(batch_size):
                    engine.add_request(str(request_id),
                        {"prompt_token_ids": [42] * args.prompt_tokens}, params)
                    request_id += 1
            finally:
                engine.scheduler = schedulers
            while engine.has_unfinished_requests():
                outputs.extend(await engine.step_async(0))
        times = step_observations[observations_start:]
        finished = {o.request_id: o for o in outputs if o.finished}
        if len(finished) != batch_size or any(
            len(o.outputs[0].token_ids) != decode_tokens for o in finished.values()):
            raise RuntimeError("Output token count / completed batch mismatch")
        if not serving:
            await engine.stop_remote_worker_execution_loop_async()
        return times

    rows, audits = [], []
    try:
        for batch_size in args.batch_sizes:
            await run_batch(batch_size, args.warmup_decode_tokens)
            executor._run_workers("pp_profile_set_enabled", False)
            baseline = []
            for _ in range(max(2, args.repetitions)):
                baseline.extend(await run_batch(batch_size, args.decode_tokens))
            if args.trace_output_dir:
                executor._run_workers("pp_profile_set_enabled", True)
                executor._run_workers("pp_profile_trace_begin", args.trace_output_dir + f"/b{batch_size}")
                trace_times = await run_batch(batch_size, args.decode_tokens)
                paths = executor._run_workers("pp_profile_trace_end")
                executor._run_workers("pp_profile_set_enabled", False)
                post_times = []
                for _ in range(2): post_times.extend(await run_batch(batch_size, args.decode_tokens))
                Path(args.trace_output_dir).mkdir(parents=True, exist_ok=True)
                Path(args.trace_output_dir + f"/b{batch_size}/audit.json").write_text(json.dumps(dict(before_ms=baseline, trace_ms=trace_times, after_ms=post_times, paths=paths), indent=2))
            steps.clear()
            executor._run_workers("pp_profile_set_enabled", True)
            enabled = True
            measured = []
            for _ in range(args.repetitions):
                measured.extend(await run_batch(batch_size, args.decode_tokens))
            enabled = False
            worker_rows = executor._run_workers("pp_profile_get_records")
            executor._run_workers("pp_profile_set_enabled", False)
            bind_worker_keys(worker_rows, args.tensor_parallel_size, args.pipeline_parallel_size)
            by_key = defaultdict(list)
            for records in worker_rows:
                for record in records:
                    by_key[record["key"]].append(record)
            phase_values = defaultdict(list)
            for step in steps:
                if step["batch_size"] != batch_size:
                    raise RuntimeError(f"Requested batch={batch_size}, observed={step['batch_size']}; "
                                       "raise token budget / KV capacity")
                values = aggregate_step(step, by_key.pop(step["key"]),
                                        args.tensor_parallel_size, args.pipeline_parallel_size)
                values.update(step)
                phase_values[step["phase"]].append(values)
            if by_key:
                raise RuntimeError("Unmatched worker timing records")
            for phase in ["prefill", "decode"]:
                samples = phase_values[phase]
                if not samples:
                    raise RuntimeError(f"No {phase} samples")
                row = dict(model_name=args.model_name_for_vidur or args.model,
                           batch_size=batch_size, tensor_parallel_degree=args.tensor_parallel_size,
                           pipeline_parallel_degree=args.pipeline_parallel_size,
                           phase=phase, prefill_tokens_per_request=(args.prompt_tokens
                                                                     if phase == "prefill" else 0),
                           profile_schema_version=5 if serving else 4, vllm_version="0.5.1",
                           profile_loop_mode="serving" if serving else "direct",
                           rank_node_map=json.dumps(rank_nodes, separators=(",", ":")),
                           num_nodes=len(node_aliases),
                           network_transport=args.network_transport if len(node_aliases) > 1 else "local",
                           event_loop_impl=type(asyncio.get_running_loop()).__module__,
                           executor_backend=type(executor).__name__,
                           requested_executor_backend="ray", enforce_eager=args.enforce_eager,
                           prompt_tokens=args.prompt_tokens, decode_tokens=args.decode_tokens,
                           repetitions=args.repetitions, num_steps=len(samples),
                           observed_batch_size_min=batch_size, observed_batch_size_max=batch_size)
                for metric in (*CPU_METRICS, "pp_handoff_e2e", "model_execution_e2e",
                               *[f"pp_handoff_boundary_{i}" for i in range(args.pipeline_parallel_size - 1)],
                               "graph_input_staging_e2e",
                               *[f"graph_input_staging_stage_{i}" for i in range(args.pipeline_parallel_size)]):
                    row[metric+"_mean"] = mean(s[metric] for s in samples)
                    row[metric+"_median"] = statistics.median(s[metric] for s in samples)
                row["step_e2e_mean"] = mean(
                    s["engine_step_ms"] for s in samples)
                rows.append(row)
            audits.append(dict(batch_size=batch_size, driver_steps=list(steps),
                               worker_steps=worker_rows,
                               uninstrumented_step_mean_ms=mean(x["ms"] for x in baseline),
                               instrumented_step_mean_ms=mean(x["ms"] for x in measured),
                               baseline_by_phase={p: mean(x["ms"] for x in baseline if x["phase"] == p) for p in ["prefill", "decode"]},
                               instrumented_by_phase={p: mean(x["ms"] for x in measured if x["phase"] == p) for p in ["prefill", "decode"]}))
            post_baseline = []
            for _ in range(max(2, args.repetitions)):
                post_baseline.extend(await run_batch(batch_size, args.decode_tokens))
            audits[-1]["post_baseline_by_phase"] = {
                p: mean(x["ms"] for x in post_baseline if x["phase"] == p)
                for p in ["prefill", "decode"]}
            print("PP_PROFILE", json.dumps(rows[-2:]), flush=True)
    finally:
        if outer and outer._background_loop_unshielded:
            task = outer._background_loop_unshielded
            for callback, _ in list(task._callbacks or []):
                if getattr(getattr(callback, "func", None), "__name__", "") == "_log_task_completion":
                    task.remove_done_callback(callback)
            task.cancel()
            try:
                await outer._background_loop_unshielded
            except asyncio.CancelledError:
                pass
        if not serving:
            await engine.stop_remote_worker_execution_loop_async()
        executor.shutdown()
        engine.model_executor = None
        ray.shutdown()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    from vllm_051_cpu_overhead import _write_rows
    _write_rows(output, rows, args.append)
    output.with_suffix(".audit.json").write_text(json.dumps(audits, indent=2))
    output.with_name(f"{output.stem}.tp{args.tensor_parallel_size}.pp{args.pipeline_parallel_size}.metadata.json").write_text(
        json.dumps(dict(arguments=vars(args), profile_schema_version=5 if serving else 4,
                        profile_loop_mode="serving" if serving else "direct",
                        sampler_semantics="forward-end to post-logits/sample CUDA event; no per-step synchronization",
                        pp_handoff_semantics="sum of max forward-end to tensor_dict-send-complete stream intervals per boundary",
                        ray_comm_semantics="executor residual excluding forward GPU, first prepare, last sampling, and handoff",
                        phase_separated=True, timing_units="milliseconds",
                        worker_environment=environment, rank_node_map=rank_nodes,
                        graph_input_staging_semantics="CUDA interval from runner entry to pre-replay stream event, excluded from forward and executor residual",
                        engine_bookkeeping_semantics="step_async regions outside schedule/execute/process plus serving-loop between-step interval when enabled",
                        baseline_audits=audits and [{k:v for k,v in a.items() if k not in ["driver_steps","worker_steps"]}
                                                   for a in audits]), indent=2))


def run(args):
    if getattr(args, "profile_serving_loop", False):
        # Match uvicorn's auto event-loop selection for the API serving path.
        try:
            import uvloop
        except ImportError:
            pass
        else:
            uvloop.install()
    asyncio.run(profile(args))
