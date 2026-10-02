"""vLLM worker used only by the PP CPU profiler, never by serving."""
import time

import torch
from vllm.distributed import get_pp_group
from vllm.worker.worker import Worker as BaseWorker

from pp_profile_contract import request_key


class Worker(BaseWorker):
    def pp_profile_set_enabled(self, enabled):
        self._pp_profile_enabled = enabled
        self._pp_profile_rows = []
        self._pp_profile_current = None
        if getattr(self, "_pp_profile_installed", False):
            return
        self._pp_profile_installed = True
        runner = self.model_runner

        def cpu_wrap(obj, name, metric, begin_sampler=False, end_sampler=False):
            original = getattr(obj, name)
            def timed(*args, **kwargs):
                row = self._pp_profile_current
                if row is None:
                    return original(*args, **kwargs)
                start = time.perf_counter_ns()
                if begin_sampler:
                    row["_sample_start"] = start
                try:
                    return original(*args, **kwargs)
                finally:
                    elapsed = (time.perf_counter_ns() - start) * 1e-6
                    row[metric] = row.get(metric, 0.0) + elapsed
                    if end_sampler:
                        row["sampler_wall_ms"] = (time.perf_counter_ns() -
                                                  row.pop("_sample_start")) * 1e-6
                        event = torch.cuda.Event(enable_timing=True)
                        event.record()
                        row["_sampler_end"] = event
                    if metric == "send_ms":
                        event = torch.cuda.Event(enable_timing=True)
                        event.record()
                        row["_send_end"] = event
            setattr(obj, name, timed)

        def forward_wrap(obj):
            original = obj.forward
            def timed(*args, **kwargs):
                row = self._pp_profile_current
                if row is None:
                    return original(*args, **kwargs)
                start, end = (torch.cuda.Event(enable_timing=True),
                              torch.cuda.Event(enable_timing=True))
                start.record()
                result = original(*args, **kwargs)
                end.record()
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
                forward_wrap(graph)
        group = get_pp_group()
        cpu_wrap(group, "send_tensor_dict", "send_ms")
        cpu_wrap(group, "recv_tensor_dict", "recv_wait_ms")

    def pp_profile_get_records(self):
        records = []
        for row in getattr(self, "_pp_profile_rows", []):
            start, end = row["_forward_start"], row["_forward_end"]
            end.synchronize()
            result = {k:v for k,v in row.items() if not k.startswith("_")}
            result["forward_gpu_ms"] = start.elapsed_time(end)
            for event_name, metric in [("_sampler_end", "sampler_ms"),
                                       ("_send_end", "send_ms")]:
                if event_name in row:
                    row[event_name].synchronize()
                    # This interval starts when forward actually completes,
                    # so CPU work overlapped with forward is counted once.
                    result[metric] = end.elapsed_time(row[event_name])
            records.append(result)
        return records

    def execute_model(self, execute_model_req=None):
        if (execute_model_req is None or not getattr(self, "_pp_profile_enabled", False)
                or not execute_model_req.seq_group_metadata_list):
            return super().execute_model(execute_model_req)
        row = dict(key=request_key(execute_model_req), rank=self.rank,
                   forward_gpu_ms=0.0, prepare_ms=0.0, sampler_ms=0.0,
                   send_ms=0.0, recv_wait_ms=0.0)
        self._pp_profile_current = row
        start = time.perf_counter_ns()
        try:
            return super().execute_model(execute_model_req)
        finally:
            row["worker_wall_ms"] = (time.perf_counter_ns() - start) * 1e-6
            self._pp_profile_rows.append(row)
            self._pp_profile_current = None
