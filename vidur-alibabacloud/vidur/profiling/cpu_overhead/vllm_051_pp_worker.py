"""vLLM worker used only by the PP CPU profiler, never by serving."""
import time
import os

import torch
from vllm.distributed import get_pp_group,get_tp_group
from vllm.worker.worker import Worker as BaseWorker

from pp_profile_contract import request_key


class Worker(BaseWorker):
    def pp_profile_environment(self):
        import ray
        return dict(rank=self.rank, node_id=ray.get_runtime_context().get_node_id(),
                    device_name=torch.cuda.get_device_name(),
                    monotonic_clock_domain=open('/proc/sys/kernel/random/boot_id').read().strip(),
                    tensor_parallel_size=self.parallel_config.tensor_parallel_size,
                    pipeline_parallel_size=self.parallel_config.pipeline_parallel_size)

    def pp_profile_reserve_events(self, capacity):
        # Initialize CUDA handles before any measured serving step. Event
        # creation and first record can otherwise compete with the engine loop.
        self._pp_event_pool = [tuple(torch.cuda.Event(enable_timing=True)
                                    for _ in range(5)) for _ in range(capacity)]
        for events in self._pp_event_pool:
            for event in events:
                event.record()
        torch.cuda.synchronize()
        self._pp_event_index = 0
        return dict(rank=self.rank, event_capacity=capacity)

    def pp_profile_trace_begin(self, directory):
        os.makedirs(directory, exist_ok=True)
        self._pp_trace_path = directory + f"/rank{self.rank}.trace.json"
        self._pp_trace = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
        self._pp_trace.__enter__()
        self._pp_trace_enabled = True
        return dict(rank=self.rank, path=self._pp_trace_path)

    def pp_profile_trace_end(self):
        self._pp_trace_enabled = False
        torch.cuda.synchronize()
        self._pp_trace.__exit__(None, None, None)
        self._pp_trace.export_chrome_trace(self._pp_trace_path)
        return dict(rank=self.rank, path=self._pp_trace_path)

    def pp_profile_set_enabled(self, enabled):
        self._pp_profile_enabled = enabled
        self._pp_profile_rows = []
        self._pp_profile_current = None
        self._pp_event_index = 0
        if getattr(self, "_pp_profile_installed", False):
            return
        self._pp_profile_installed = True
        runner = self.model_runner
        from eager_dispatch_trace import install_llama_dispatch_trace
        # Leaf CPU timestamps are collected only in eager prefill. Decode
        # graph replay retains its existing pooled GPU event measurements.
        install_llama_dispatch_trace(self, runner.model)

        def cpu_wrap(obj, name, metric, begin_sampler=False, end_sampler=False):
            original = getattr(obj, name)
            def timed(*args, **kwargs):
                row = self._pp_profile_current
                if row is None:
                    return original(*args, **kwargs)
                trace = getattr(self, "_pp_trace_enabled", False) and name in ("compute_logits", "sample", "send_tensor_dict")
                if trace: torch.cuda._sleep(1)
                start = time.perf_counter_ns()
                cpu_start = time.thread_time_ns()
                if begin_sampler:
                    row["_sample_start"] = start
                try:
                    return original(*args, **kwargs)
                finally:
                    if trace: torch.cuda._sleep(1)
                    end_ns = time.perf_counter_ns()
                    cpu_ms = (time.thread_time_ns() - cpu_start) * 1e-6
                    elapsed = (end_ns - start) * 1e-6
                    row.setdefault("host_regions", []).append(dict(
                        name=name,start_ns=start,end_ns=end_ns,wall_ms=elapsed,
                        thread_cpu_ms=cpu_ms))
                    row[metric] = row.get(metric, 0.0) + elapsed
                    if end_sampler:
                        row["sampler_wall_ms"] = (time.perf_counter_ns() -
                                                  row.pop("_sample_start")) * 1e-6
                        event = row["_events"][3]
                        event.record()
                        row["_sampler_end"] = event
                    if metric == "send_ms":
                        event = row["_events"][4]
                        event.record()
                        row["_send_end"] = event
            setattr(obj, name, timed)

        def forward_wrap(obj):
            original = obj.forward
            def timed(*args, **kwargs):
                row = self._pp_profile_current
                if row is None:
                    return original(*args, **kwargs)
                start, end = row["_events"][:2]
                trace = getattr(self, "_pp_trace_enabled", False)
                if trace: torch.cuda._sleep(1)
                host_start=time.perf_counter_ns();cpu_start=time.thread_time_ns()
                metadata=kwargs.get('attn_metadata')
                capture=(obj is runner.model and metadata is not None and
                         metadata.prefill_metadata is not None)
                if capture:
                    row['_capture_cpu_dispatch']=True
                    row['cpu_forward_start_ns']=host_start
                start.record()
                try:
                    result = original(*args, **kwargs)
                finally:
                    if capture:
                        row['_capture_cpu_dispatch']=False
                        row['cpu_forward_end_ns']=time.perf_counter_ns()
                end.record()
                host_end=time.perf_counter_ns()
                row.setdefault("host_regions",[]).append(dict(
                    name="forward_graph" if hasattr(obj,"graph") else "forward_eager",
                    start_ns=host_start,end_ns=host_end,wall_ms=(host_end-host_start)*1e-6,
                    thread_cpu_ms=(time.thread_time_ns()-cpu_start)*1e-6))
                if trace: torch.cuda._sleep(1)
                # Read events only after requests finish. Synchronizing here
                # prevents CPU metadata and GPU compute from overlapping and
                # materially perturbs the very PP path being measured.
                row["_forward_start"], row["_forward_end"] = start, end
                return result
            obj.forward = timed

        cpu_wrap(runner, "prepare_model_input", "prepare_ms")
        cpu_wrap(runner.model, "compute_logits", "logits_host_ms", begin_sampler=True)
        cpu_wrap(runner.model, "sample", "sample_host_ms", end_sampler=True)
        forward_wrap(runner.model)
        for graphs in runner.graph_runners:
            for graph in graphs.values():
                original_replay = graph.graph.replay
                def replay(original_replay=original_replay):
                    row = self._pp_profile_current
                    if row is not None:
                        event = row["_events"][2]
                        event.record()
                        row["_graph_compute_start"] = event
                    return original_replay()
                graph.graph.replay = replay
                forward_wrap(graph)
        original_runner_execute = runner.execute_model
        def follower_execute(*a, **kw):
            if self.is_driver_worker or not self._pp_profile_enabled:
                return original_runner_execute(*a, **kw)
            row = self._pp_new_row(None)
            self._pp_profile_current = row
            start = time.perf_counter_ns()
            cpu_start = time.thread_time_ns()
            row.update(worker_start_ns=start,worker_timing_scope='model_runner')
            try:
                return original_runner_execute(*a, **kw)
            finally:
                end = time.perf_counter_ns()
                row.update(worker_end_ns=end,worker_wall_ms=(end-start)*1e-6,
                           worker_thread_cpu_ms=(time.thread_time_ns()-cpu_start)*1e-6)
                self._pp_profile_rows.append(row)
                self._pp_profile_current = None
        runner.execute_model = follower_execute
        cpu_wrap(runner, "execute_model", "model_runner_host_ms")
        cpu_wrap(self, "prepare_worker_input", "worker_input_ms")
        cpu_wrap(self, "execute_worker", "cache_operations_ms")
        cpu_wrap(get_tp_group(), "broadcast_tensor_dict", "tp_metadata_broadcast_ms")

        group = get_pp_group()
        cpu_wrap(group, "send_tensor_dict", "send_ms")
        cpu_wrap(group, "recv_tensor_dict", "recv_wait_ms")

    def pp_profile_get_records(self):
        records = []
        for row in getattr(self, "_pp_profile_rows", []):
            start, end = row["_forward_start"], row["_forward_end"]
            end.synchronize()
            result = {k:v for k,v in row.items() if not k.startswith("_")}
            result["send_host_wall_ms"] = row["send_ms"]
            result["forward_gpu_ms"] = start.elapsed_time(end)
            result["graph_input_staging_ms"] = 0.0
            if "_graph_compute_start" in row:
                split = row["_graph_compute_start"]
                result["graph_input_staging_ms"] = start.elapsed_time(split)
                result["forward_gpu_ms"] = split.elapsed_time(end)
            for event_name, metric in [("_sampler_end", "sampler_ms"),
                                       ("_send_end", "send_ms")]:
                if event_name in row:
                    row[event_name].synchronize()
                    # This interval starts when forward actually completes,
                    # so CPU work overlapped with forward is counted once.
                    result[metric] = end.elapsed_time(row[event_name])
            records.append(result)
        return records

    def _pp_new_row(self, key):
        row = dict(key=key, rank=self.rank, step_index=self._pp_event_index,
                   forward_gpu_ms=0.0, prepare_ms=0.0, sampler_ms=0.0,
                   send_ms=0.0, recv_wait_ms=0.0)
        if self._pp_event_index >= len(self._pp_event_pool):
            raise RuntimeError("CUDA event pool exhausted; increase profiling reserve")
        row["_events"] = self._pp_event_pool[self._pp_event_index]
        self._pp_event_index += 1
        return row

    def execute_model(self, execute_model_req=None):
        if (execute_model_req is None or not getattr(self, "_pp_profile_enabled", False)
                or not execute_model_req.seq_group_metadata_list):
            return super().execute_model(execute_model_req)
        row = self._pp_new_row(request_key(execute_model_req))
        self._pp_profile_current = row
        start = time.perf_counter_ns()
        cpu_start = time.thread_time_ns()
        row.update(worker_start_ns=start,worker_timing_scope='worker')
        try:
            return super().execute_model(execute_model_req)
        finally:
            end = time.perf_counter_ns()
            row.update(worker_end_ns=end,worker_wall_ms=(end-start)*1e-6,
                       worker_thread_cpu_ms=(time.thread_time_ns()-cpu_start)*1e-6)
            self._pp_profile_rows.append(row)
            self._pp_profile_current = None
