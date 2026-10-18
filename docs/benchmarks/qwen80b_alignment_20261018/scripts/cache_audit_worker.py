import functools,time,json,pathlib,torch
from qwen_audit_worker import AuditWorker,ROOT
class CacheAuditWorker(AuditWorker):
 def q80_start_cache_audit(self):
  import vllm.model_executor.models.qwen3_next as model
  from vllm.model_executor.layers.fla.ops import index
  from vllm.forward_context import get_forward_context
  self._cache_rows=[];self._cache_original=model.Qwen3NextGatedDeltaNet._forward_core;w=self
  def entries(fn):
   return inspect_closure(fn).get('cache_entries',[])
  def inspect_closure(fn):
   return dict(zip(fn.__code__.co_freevars,[c.cell_contents for c in (fn.__closure__ or ())]))
  def snapshot(cu):
   result={}
   for name in ['prepare_lens','prepare_chunk_indices','prepare_chunk_offsets']:
    fn=getattr(index,name);ee=entries(fn)
    result[name]={'entries':len(ee),'matching_tensor_entries':sum(e[0][0] is cu for e in ee)}
   return result
  @functools.wraps(self._cache_original)
  def wrapped(obj,*aa,**kw):
   ctx=get_forward_context();md=ctx.attn_metadata.get(obj.prefix) if isinstance(ctx.attn_metadata,dict) else None
   if md is None or not md.num_prefills:return w._cache_original(obj,*aa,**kw)
   cu=md.non_spec_query_start_loc
   before=snapshot(cu);t=time.perf_counter_ns()
   result=w._cache_original(obj,*aa,**kw)
   w._cache_rows.append(dict(layer=obj.prefix,cu_object=id(cu),cu_ptr=cu.data_ptr(),cu_shape=list(cu.shape),cpu_ms=(time.perf_counter_ns()-t)/1e6,before=before,after=snapshot(cu),conv_stride=list(obj.kv_cache[ctx.virtual_engine][0].stride()),ssm_shape=list(obj.kv_cache[ctx.virtual_engine][1].shape)))
   return result
  model.Qwen3NextGatedDeltaNet._forward_core=wrapped
  return dict(dp_rank=self.vllm_config.parallel_config.data_parallel_rank)
 def q80_stop_cache_audit(self):
  import vllm.model_executor.models.qwen3_next as model
  model.Qwen3NextGatedDeltaNet._forward_core=self._cache_original
  rank=self.vllm_config.parallel_config.data_parallel_rank;p=ROOT/'cache_audit'/f'dp{rank}.json';p.parent.mkdir(exist_ok=True)
  p.write_text(json.dumps(self._cache_rows));return dict(rows=len(self._cache_rows),path=str(p))
