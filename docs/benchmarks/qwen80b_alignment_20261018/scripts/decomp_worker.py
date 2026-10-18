import functools,time,torch,json
from alignment_worker import AlignmentWorker
from qwen_audit_worker import ROOT
class DecompWorker(AlignmentWorker):
 def q80_start_decomposition(self,gpu_events=False):
  self.q80_start_components(gpu_events)
  from torch._inductor.codecache import CompiledFxGraph
  original=CompiledFxGraph.__call__;worker=self;self._fx_original=original
  @functools.wraps(original)
  def call(graph,*args,**kwargs):
   iteration=getattr(worker,'_q80_current_iteration',None)
   if not iteration or iteration.get('total_tokens',0)<=1:return original(graph,*args,**kwargs)
   start_cpu=time.perf_counter_ns();start=end=None
   if gpu_events:
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
   try:return original(graph,*args,**kwargs)
   finally:
    if end is not None:end.record()
    fn=getattr(graph,'current_callable',None);code=getattr(fn,'__code__',None)
    worker._align_rows.append(dict(kind='compiled_fx',layer=str(getattr(graph,'cache_key','unknown')),filename=code.co_filename if code else str(type(fn)),cpu_start_ns=start_cpu,cpu_end_ns=time.perf_counter_ns(),start=start,end=end,iteration=iteration))
  CompiledFxGraph.__call__=call
  return dict(dp_rank=self.vllm_config.parallel_config.data_parallel_rank)
 def q80_stop_decomposition(self,label):
  from torch._inductor.codecache import CompiledFxGraph
  CompiledFxGraph.__call__=self._fx_original
  return self.q80_stop_components(label)
