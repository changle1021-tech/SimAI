"""Temporary vLLM worker extension; the installed vLLM source is unchanged."""
import collections
import gzip
import json
import pathlib
import time

import torch


ROOT = pathlib.Path('/home/turbo_ops/changle/Files/analysis/qwen80b_alignment_20261016')


class _TimedModelProxy:
    def __init__(self, model, worker):
        self._model, self._worker = model, worker

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __call__(self, *args, **kwargs):
        from vllm.forward_context import get_forward_context
        context = get_forward_context()
        metadata = context.dp_metadata
        sizes = getattr(metadata, 'num_tokens_across_dp_cpu', None)
        if isinstance(sizes, torch.Tensor):
            sizes = sizes.tolist() if sizes.device.type == 'cpu' else None
        extra = dict(dp_sizes=list(sizes) if sizes is not None else None,
                     cudagraph_runtime_mode=str(getattr(context, 'cudagraph_runtime_mode', 'unknown')))
        if getattr(self._worker, '_q80_current_iteration', None) is None:
            self._worker._q80_last_execute_info = None
        return self._worker._q80_timed_call('model', self._model, args, kwargs, extra)


class AuditWorker:
    def _q80_timed_call(self, kind, fn, args, kwargs, extra=None):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        cpu_start = time.perf_counter_ns()
        start.record()
        try:
            return fn(*args, **kwargs)
        finally:
            end.record()
            row = dict(kind=kind, start=start, end=end,
                       cpu_start_ns=cpu_start, cpu_end_ns=time.perf_counter_ns(),
                       iteration=getattr(self, '_q80_current_iteration', None))
            if extra:
                row.update(extra)
            self._q80_timer_rows.append(row)

    def q80_start_timers(self):
        self._q80_timer_rows = []
        self._q80_current_iteration = None
        self._q80_timer_origin = torch.cuda.Event(enable_timing=True)
        self._q80_timer_origin.record()
        runner = self.model_runner
        self._q80_original_model = runner.model
        self._q80_original_execute = runner.execute_model
        self._q80_original_sample = runner.sample_tokens
        runner.model = _TimedModelProxy(runner.model, self)

        def execute(scheduler_output, *args, **kwargs):
            counts = dict(scheduler_output.num_scheduled_tokens)
            self._q80_current_iteration = dict(request_tokens=counts,
                                               total_tokens=sum(counts.values())) if counts else None
            self._q80_last_execute_info = self._q80_current_iteration
            try:
                return self._q80_timed_call('execute', self._q80_original_execute,
                                            (scheduler_output, *args), kwargs)
            finally:
                self._q80_current_iteration = None

        def sample(*args, **kwargs):
            self._q80_current_iteration = self._q80_last_execute_info
            try:
                return self._q80_timed_call('sample', self._q80_original_sample, args, kwargs)
            finally:
                self._q80_current_iteration = None
                self._q80_last_execute_info = None

        runner.execute_model, runner.sample_tokens = execute, sample
        return dict(dp_rank=self.vllm_config.parallel_config.data_parallel_rank)

    def q80_stop_timers(self, label):
        runner = self.model_runner
        runner.model = self._q80_original_model
        runner.execute_model = self._q80_original_execute
        runner.sample_tokens = self._q80_original_sample
        torch.cuda.synchronize()
        rows = []
        for row in self._q80_timer_rows:
            start, end = row.pop('start'), row.pop('end')
            row.update(start_ms=self._q80_timer_origin.elapsed_time(start),
                       end_ms=self._q80_timer_origin.elapsed_time(end),
                       device_elapsed_ms=start.elapsed_time(end),
                       cpu_elapsed_ms=(row['cpu_end_ns']-row['cpu_start_ns'])/1e6)
            rows.append(row)
        rank = self.vllm_config.parallel_config.data_parallel_rank
        path = ROOT/'timers'/label/f'dp{rank}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(info=self.q80_info(), rows=rows)))
        return dict(dp_rank=rank, rows=len(rows), path=str(path))

    def q80_info(self):
        p = self.vllm_config.parallel_config
        return dict(dp_rank=p.data_parallel_rank, local_rank=self.local_rank,
                    device=str(self.device), dtype=str(self.vllm_config.model_config.dtype),
                    modules=dict(collections.Counter(type(m).__name__ for m in self.get_model().modules())))

    def q80_start_trace(self, label):
        if self.profiler is not None:
            raise RuntimeError('Profiler already configured')
        self._q80_label = label
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False)
        self.profiler.start()
        return self.q80_info()

    def q80_stop_trace(self):
        self.profiler.stop()
        rank = self.vllm_config.parallel_config.data_parallel_rank
        directory = ROOT/'traces'/self._q80_label
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/f'dp{rank}.json'
        self.profiler.export_chrome_trace(str(path))
        with path.open('rb') as src, gzip.open(str(path)+'.gz', 'wb') as dst:
            while block := src.read(1024*1024):
                dst.write(block)
        path.unlink()
        self.profiler = None
        return dict(dp_rank=rank, path=str(path)+'.gz')

    def q80_start_events(self):
        if getattr(self, '_q80_handles', None):
            raise RuntimeError('Event hooks already active')
        self._q80_rows, self._q80_handles = [], []
        self._q80_origin = torch.cuda.Event(enable_timing=True)
        self._q80_origin.record()
        stacks = collections.defaultdict(list)
        targets = dict(Qwen3NextGatedDeltaNet='gdn', Qwen3NextAttention='attention',
                       Qwen3NextSparseMoeBlock='moe', Qwen3NextMLP='shared_expert',
                       VocabParallelEmbedding='embedding', LogitsProcessor='logits')
        for name, module in self.get_model().named_modules():
            category = targets.get(type(module).__name__)
            if name.endswith('.shared_expert'):
                category = 'shared_expert'
            if name.endswith(('input_layernorm', 'post_attention_layernorm')):
                category = 'layernorm'
            if category is None:
                continue

            def before(mod, args, kwargs, name=name, category=category):
                tensor = kwargs.get('hidden_states')
                if tensor is None:
                    tensor = next((v for v in args if isinstance(v, torch.Tensor)), None)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                row = dict(name=name, category=category,
                           shape=list(tensor.shape) if tensor is not None else None,
                           start=start, end=end, cpu_start_ns=time.perf_counter_ns())
                start.record()
                stacks[id(mod)].append(row)

            def after(mod, args, kwargs, output):
                row = stacks[id(mod)].pop()
                row['end'].record()
                row['cpu_elapsed_ms'] = (time.perf_counter_ns()-row.pop('cpu_start_ns'))/1e6
                self._q80_rows.append(row)

            self._q80_handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
            self._q80_handles.append(module.register_forward_hook(after, with_kwargs=True))
        return dict(dp_rank=self.vllm_config.parallel_config.data_parallel_rank,
                    instrumented_modules=len(self._q80_handles)//2)

    def q80_stop_events(self, label):
        for handle in self._q80_handles:
            handle.remove()
        self._q80_handles = []
        torch.cuda.synchronize()
        rows = []
        for row in self._q80_rows:
            start, end = row.pop('start'), row.pop('end')
            row.update(start_ms=self._q80_origin.elapsed_time(start),
                       end_ms=self._q80_origin.elapsed_time(end),
                       device_elapsed_ms=start.elapsed_time(end))
            rows.append(row)
        rank = self.vllm_config.parallel_config.data_parallel_rank
        path = ROOT/'events'/label/f'dp{rank}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(info=self.q80_info(), rows=rows)))
        return dict(dp_rank=rank, samples=len(rows), path=str(path))
