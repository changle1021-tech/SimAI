"""Measure one async API batch's coarse critical path without per-step sync.

GPU clock anchors are taken only outside requests. Model GPU intervals end
before sampler GPU work. Prefix/sampler/return tails are named wall intervals,
not pure CPU or network times. Raw endpoints and clock uncertainty are retained.
"""
import contextvars
import functools
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import time
from importlib.metadata import version
import torch
from vllm.worker.worker import Worker
from vllm.executor.ray_utils import RayWorkerWrapper

STATE = {'active': False, 'steps': [], 'windows': 0}
CURRENT = contextvars.ContextVar('path_step', default=None)

class PathWorker(Worker):
    def load_model(self):
        result = super().load_model()
        self.path_active = False
        self.path_rows = []
        self.path_current = None
        runner = self.model_runner
        original = runner.execute_model

        @functools.wraps(original)
        def execute(model_input, *args, **kwargs):
            if not self.path_active:
                return original(model_input, *args, **kwargs)
            index = len(self.path_rows)
            if index >= len(self.path_events):
                raise RuntimeError('Preallocated event capacity exhausted')
            metadata = model_input.attn_metadata
            queries = model_input.query_lens
            lengths = model_input.seq_lens
            batch_size = len(queries) if queries is not None else None
            row = {'ordinal': index, 'real_batch_size': batch_size,
                   'query_lens': queries,
                   'seq_lens': lengths[:batch_size] if lengths is not None and batch_size is not None else lengths,
                   'forward_tokens': int(model_input.input_tokens.shape[0]),
                   'num_prefill_tokens': metadata.num_prefill_tokens,
                   'cuda_graph': bool(metadata.decode_metadata is not None and metadata.decode_metadata.use_cuda_graph),
                   'host_enter_ns': time.perf_counter_ns(),
                   'start': self.path_events[index][0],
                   'end': self.path_events[index][1], 'end_recorded': False}
            self.path_current = row
            row['start'].record()
            try:
                return original(model_input, *args, **kwargs)
            finally:
                if not row['end_recorded']:
                    row['host_model_end_ns'] = time.perf_counter_ns()
                    row['end'].record()
                    row['end_recorded'] = True
                row['host_exit_ns'] = time.perf_counter_ns()
                self.path_rows.append(row)
                self.path_current = None
        runner.execute_model = execute
        original_sample = runner.model.sample

        @functools.wraps(original_sample)
        def sample(*args, **kwargs):
            row = self.path_current
            if row is not None:
                row['host_sampler_start_ns'] = time.perf_counter_ns()
                row['host_model_end_ns'] = row['host_sampler_start_ns']
                row['end'].record()
                row['end_recorded'] = True
            try:
                return original_sample(*args, **kwargs)
            finally:
                if row is not None:
                    row['host_sampler_end_ns'] = time.perf_counter_ns()
        runner.model.sample = sample
        return result

    def path_anchor(self, event):
        before = time.perf_counter_ns()
        event.record()
        event.synchronize()
        after = time.perf_counter_ns()
        return {'before_ns': before, 'after_ns': after}

    def path_begin(self):
        if self.path_active:
            raise RuntimeError('Path probe already active')
        self.path_rows = []
        # Events are allocated and initialized outside the measured workload.
        self.path_events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                            for _ in range(4096)]
        self.path_anchor_start = torch.cuda.Event(enable_timing=True)
        self.path_anchor_end = torch.cuda.Event(enable_timing=True)
        for pair in self.path_events:
            pair[0].record(); pair[1].record()
        self.path_anchor_start.record(); self.path_anchor_end.record()
        torch.cuda.synchronize()
        self.path_start_bounds = self.path_anchor(self.path_anchor_start)
        self.path_active = True
        return {'rank': self.rank, 'clock_anchor': self.path_start_bounds}

    def path_finish(self):
        self.path_active = False
        end_bounds = self.path_anchor(self.path_anchor_end)
        # Both anchors now complete, and so do all earlier model markers.
        anchor_gpu_ms = self.path_anchor_start.elapsed_time(self.path_anchor_end)
        rows = []
        for row in self.path_rows:
            start, end = row.pop('start'), row.pop('end')
            row.pop('end_recorded')
            row['gpu_start_since_anchor_ms'] = self.path_anchor_start.elapsed_time(start)
            row['gpu_end_since_anchor_ms'] = self.path_anchor_start.elapsed_time(end)
            row['model_gpu_interval_ms'] = start.elapsed_time(end)
            rows.append(row)
        from vllm.distributed import get_tp_group, get_pp_group
        tp, pp = get_tp_group(), get_pp_group()
        return {'rank': self.rank, 'tp_rank': tp.rank_in_group, 'tp_size': tp.world_size,
                'pp_stage': pp.rank_in_group, 'pp_size': pp.world_size,
                'model_class': type(self.model_runner.model).__module__+'.'+type(self.model_runner.model).__name__,
                'hf_config': self.model_config.hf_config.to_dict(), 'dtype': str(self.model_config.dtype),
                'anchor_start': self.path_start_bounds, 'anchor_end': end_bounds,
                'anchor_gpu_ms': anchor_gpu_ms, 'samples': rows}

class PathRayWorker(RayWorkerWrapper):
    def __init__(self, *args, **kwargs):
        kwargs['worker_module_name'] = 'async_path_server'
        kwargs['worker_class_name'] = 'PathWorker'
        super().__init__(*args, **kwargs)

def main():
    if version('vllm').split('+')[0] != '0.5.1':
        raise RuntimeError('This probe targets vLLM 0.5.1')
    index = sys.argv.index('--path-output')
    root = Path(sys.argv[index+1]); del sys.argv[index:index+2]
    root.mkdir(parents=True, exist_ok=False)
    (root/'server_pid.txt').write_text(str(os.getpid()))
    (root/'state.json').write_text(json.dumps({'status': 'starting'}))
    import vllm.executor.ray_gpu_executor as executor_module
    from vllm.engine.async_llm_engine import _AsyncLLMEngine
    from vllm.core.scheduler import Scheduler
    executor_module.RayWorkerWrapper = PathRayWorker
    engines = []
    original_init = _AsyncLLMEngine.__init__
    original_step = _AsyncLLMEngine.step_async
    original_schedule = Scheduler.schedule

    def schedule(self, *args, **kwargs):
        row = CURRENT.get()
        if row is None:
            return original_schedule(self, *args, **kwargs)
        row['schedule_start_ns'] = time.perf_counter_ns()
        result = original_schedule(self, *args, **kwargs)
        row['schedule_end_ns'] = time.perf_counter_ns()
        metadata, outputs = result
        row['batch_size'] = len(outputs.scheduled_seq_groups)
        row['num_prefill_groups'] = outputs.num_prefill_groups
        row['request_ids'] = [m.request_id for m in metadata]
        return result

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        engines.append(self)
        original_execute = self.model_executor.execute_model_async
        async def execute(*args, **kwargs):
            row = CURRENT.get()
            if row is None:
                return await original_execute(*args, **kwargs)
            row['executor_start_ns'] = time.perf_counter_ns()
            result = await original_execute(*args, **kwargs)
            row['executor_end_ns'] = time.perf_counter_ns()
            return result
        self.model_executor.execute_model_async = execute

    async def step(self, virtual_engine):
        if not STATE['active']:
            return await original_step(self, virtual_engine)
        row = {'virtual_engine': virtual_engine, 'step_start_ns': time.perf_counter_ns()}
        token = CURRENT.set(row)
        try:
            result = await original_step(self, virtual_engine)
            row['step_end_ns'] = time.perf_counter_ns()
        finally:
            CURRENT.reset(token)
        STATE['steps'].append(row)
        return result

    Scheduler.schedule = schedule
    _AsyncLLMEngine.__init__ = initialize
    _AsyncLLMEngine.step_async = step

    def toggle(signum, frame):
        if len(engines) != 1:
            raise RuntimeError('Expected one async engine')
        executor = engines[0].model_executor
        if not STATE['active']:
            STATE['steps'] = []
            anchors = executor._run_workers('path_begin')
            STATE['active'] = True
            (root/'active.json').write_text(json.dumps(anchors))
            (root/'state.json').write_text(json.dumps({'status': 'collecting', 'window': STATE['windows']}))
        else:
            STATE['active'] = False
            ranks = executor._run_workers('path_finish')
            window = STATE['windows']; STATE['windows'] += 1
            payload = {'argv': sys.argv, 'versions': {k: version(k) for k in ('vllm','torch','ray')},
                       'scope': 'CPU perf_counter endpoints and per-rank CUDA model intervals before sampler, tied by two outside-workload GPU clock anchor brackets. No per-step forced synchronization.',
                       'steps': STATE['steps'], 'ranks': ranks}
            (root/f'window_{window}.json').write_text(json.dumps(payload, indent=2))
            (root/'state.json').write_text(json.dumps({'status': 'complete', 'window': window, 'scope': 'raw collection only; timing validation separate'}))
        print('PATH_STATE', STATE['active'], STATE['windows'], flush=True)
    signal.signal(signal.SIGUSR1, toggle)
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')

if __name__ == '__main__':
    main()
