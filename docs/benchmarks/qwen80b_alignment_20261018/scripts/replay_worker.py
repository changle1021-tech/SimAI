import copy,json,time,torch,types,sys
from cache_audit_worker import CacheAuditWorker
from alignment_worker import AlignmentWorker
from qwen_audit_worker import ROOT
class ReplayWorker(AlignmentWorker,CacheAuditWorker):
 def q80_start_replay_capture(self):
  import vllm.model_executor.models.qwen3_next as model
  from vllm.forward_context import get_forward_context
  original=model.Qwen3NextGatedDeltaNet._forward_core;self._replay_original=original;self._replay_saved=None;worker=self
  def clone(v):
   if isinstance(v,torch.Tensor):return v.detach().clone()
   if isinstance(v,dict):return {k:clone(x) for k,x in v.items()}
   if isinstance(v,list):return [clone(x) for x in v]
   if isinstance(v,tuple):return tuple(clone(x) for x in v)
   return v
  def wrapped(obj,*aa,**kw):
   ctx=get_forward_context();md=ctx.attn_metadata.get(obj.prefix) if isinstance(ctx.attn_metadata,dict) else None
   if md is not None and md.num_prefills and worker._replay_saved is None:
    metadata=copy.copy(md)
    for k,v in vars(md).items():setattr(metadata,k,clone(v))
    context=copy.copy(ctx);context.attn_metadata={obj.prefix:metadata}
    worker._replay_saved=(obj,clone(aa),clone(kw),context)
    from torch.overrides import _get_current_function_mode_stack
    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
    worker._replay_mode=dict(default_device=str(torch.get_default_device()),threads=torch.get_num_threads(),function_modes=[str(type(m)) for m in _get_current_function_mode_stack()],dispatch_modes=[str(type(m)) for m in _get_current_dispatch_mode_stack()])
   return original(obj,*aa,**kw)
  model.Qwen3NextGatedDeltaNet._forward_core=wrapped
  return dict(dp_rank=self.vllm_config.parallel_config.data_parallel_rank)
 def q80_replay_saved(self):
  import vllm.model_executor.models.qwen3_next as model
  model.Qwen3NextGatedDeltaNet._forward_core=self._replay_original
  if self._replay_saved is None:return dict(saved=False)
  sys.path.insert(0,str(ROOT/'source/SimAI/aicb'))
  from workload_generator.mocked_model.inference.qwen3_next_projections import benchmark_gpu_region
  from workload_generator.mocked_model.inference.qwen3_next_attention import benchmark_gdn_native_core
  from vllm.forward_context import override_forward_context
  from vllm.model_executor.layers.fla.ops import index
  obj,aa,kw,ctx=self._replay_saved
  mixed=aa[0] if aa else kw['mixed_qkv'];tokens=mixed.shape[0]
  rank=self.vllm_config.parallel_config.data_parallel_rank;p=ROOT/'service_replay_v2'/f'dp{rank}_p{tokens}.json';p.parent.mkdir(exist_ok=True)
  result=dict(saved=True,mode=self._replay_mode,prefix=obj.prefix,ssm_shape=list(obj.kv_cache[0][1].shape))
  def clear():
   for name in ['prepare_lens','prepare_chunk_indices','prepare_chunk_offsets']:
    fn=getattr(index,name)
    for n,c in zip(fn.__code__.co_freevars,fn.__closure__ or ()):
     if n=='cache_entries':c.cell_contents=[]
  with torch.inference_mode(),override_forward_context(ctx):
   def call():return self._replay_original(obj,*aa,**kw)
   result['inputs']={k:dict(shape=list(v.shape),stride=list(v.stride()),dtype=str(v.dtype)) for k,v in kw.items() if isinstance(v,torch.Tensor)}
   result['warm_ms']=[benchmark_gpu_region(call,warmup=3,samples=15,timing_mode='eager')*1e3 for _ in range(3)];p.write_text(json.dumps(result,indent=2))
   def custom_op():
    return torch.ops.vllm.gdn_attention_core(kw['mixed_qkv'],kw['b'],kw['a'],kw['core_attn_out'],obj.prefix)
   result['custom_op_ms']=[benchmark_gpu_region(custom_op,warmup=3,samples=15,timing_mode='eager')*1e3 for _ in range(3)];p.write_text(json.dumps(result,indent=2))
   def cold():clear();return call()
   result['metadata_cold_ms']=[benchmark_gpu_region(cold,warmup=3,samples=15,timing_mode='eager')*1e3 for _ in range(3)];p.write_text(json.dumps(result,indent=2))
  args=types.SimpleNamespace(**json.loads((ROOT/'source/SimAI/aicb/scripts/inference_configs/qwen3_next_default.json').read_text()))
  args.tensor_model_parallel_size=1;args.phase='prefill';args.context_length=0
  result['standalone_helper_ms']=[benchmark_gdn_native_core(args,'prefill',tokens)*1e3 for _ in range(3)]
  p.write_text(json.dumps(result,indent=2))
  self._replay_saved=None
  return result
