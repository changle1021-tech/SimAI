"""Profile real vLLM 0.5.1 async PP execution with separate phase labels."""
import asyncio
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import ray
from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.async_llm_engine import _AsyncLLMEngine
from vllm.executor.ray_gpu_executor import RayGPUExecutorAsync
from vllm.executor.ray_utils import initialize_ray_cluster

from pp_profile_contract import aggregate_step, request_key, CPU_METRICS


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
    engine = _AsyncLLMEngine(**config.to_dict(), executor_class=ProfilingRayExecutor,
                             log_stats=False)
    executor = engine.model_executor
    current, steps = {}, []
    enabled = False

    for index, scheduler in enumerate(engine.scheduler):
        original = scheduler.schedule
        def schedule(original=original, index=index):
            start = time.perf_counter_ns()
            result = original()
            if enabled:
                current.clear()
                metadata = result[0]
                if not metadata:
                    raise RuntimeError("Unexpected empty step in fixed-batch profiling")
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
        start = time.perf_counter_ns()
        result = await original_execute(req)
        if enabled:
            current.update(key=request_key(req),
                           executor_ms=(time.perf_counter_ns()-start)*1e-6)
        return result
    executor.execute_model_async = execute
    original_process = engine._process_model_outputs
    def process_outputs(*a, **kw):
        start = time.perf_counter_ns()
        result = original_process(*a, **kw)
        if enabled:
            current["process_outputs_ms"] = (time.perf_counter_ns()-start)*1e-6
            steps.append(dict(current))
        return result
    engine._process_model_outputs = process_outputs

    request_id = 0
    async def run_batch(batch_size, decode_tokens):
        nonlocal request_id
        # The unit being measured is one batch, not PP virtual-engine concurrency.
        # Place it entirely into VE0; restore the full scheduler list before running.
        schedulers = engine.scheduler
        try:
            engine.scheduler = schedulers[:1]
            for _ in range(batch_size):
                engine.add_request(str(request_id),
                    {"prompt_token_ids": [42] * args.prompt_tokens},
                    SamplingParams(temperature=0, ignore_eos=True, max_tokens=decode_tokens))
                request_id += 1
        finally:
            engine.scheduler = schedulers
        times = []
        outputs = []
        while engine.has_unfinished_requests():
            start = time.perf_counter_ns()
            outputs.extend(await engine.step_async(0))
            times.append((time.perf_counter_ns()-start)*1e-6)
        finished = {o.request_id: o for o in outputs if o.finished}
        if len(finished) != batch_size or any(
            len(o.outputs[0].token_ids) != decode_tokens for o in finished.values()):
            raise RuntimeError("Output token count / completed batch mismatch")
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
            steps.clear()
            executor._run_workers("pp_profile_set_enabled", True)
            enabled = True
            measured = []
            for _ in range(args.repetitions):
                measured.extend(await run_batch(batch_size, args.decode_tokens))
            enabled = False
            worker_rows = executor._run_workers("pp_profile_get_records")
            executor._run_workers("pp_profile_set_enabled", False)
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
                           profile_schema_version=2, vllm_version="0.5.1",
                           executor_backend=type(executor).__name__,
                           requested_executor_backend="ray", enforce_eager=args.enforce_eager,
                           prompt_tokens=args.prompt_tokens, decode_tokens=args.decode_tokens,
                           repetitions=args.repetitions, num_steps=len(samples),
                           observed_batch_size_min=batch_size, observed_batch_size_max=batch_size)
                for metric in (*CPU_METRICS, "pp_handoff_e2e", "model_execution_e2e"):
                    row[metric+"_mean"] = mean(s[metric] for s in samples)
                    row[metric+"_median"] = statistics.median(s[metric] for s in samples)
                row["step_e2e_mean"] = mean(
                    s["executor_ms"]+s["schedule_ms"]+s["process_outputs_ms"] for s in samples)
                rows.append(row)
            audits.append(dict(batch_size=batch_size, driver_steps=list(steps),
                               worker_steps=worker_rows,
                               uninstrumented_step_mean_ms=mean(baseline),
                               instrumented_step_mean_ms=mean(measured)))
            print("PP_PROFILE", json.dumps(rows[-2:]), flush=True)
    finally:
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
        json.dumps(dict(arguments=vars(args), profile_schema_version=2,
                        sampler_semantics="forward-end to post-logits/sample CUDA event; no per-step synchronization",
                        pp_handoff_semantics="sum of max forward-end to tensor_dict-send-complete stream intervals per boundary",
                        ray_comm_semantics="executor residual excluding forward GPU, first prepare, last sampling, and handoff",
                        phase_separated=True, timing_units="milliseconds",
                        baseline_audits=audits and [{k:v for k,v in a.items() if k not in ["driver_steps","worker_steps"]}
                                                   for a in audits]), indent=2))


def run(args):
    asyncio.run(profile(args))
