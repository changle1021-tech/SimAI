import copy,functools,json,torch,statistics as st
from continuous_moe_worker import ContinuousMoEWorker
from qwen_audit_worker import ROOT
class NativeChainWorker(ContinuousMoEWorker):
 def q80_start_chain_capture(self,tokens):
  result=self.q80_start_moe_capture(tokens)
  from vllm.model_executor.models.qwen3_next import Qwen3NextGatedDeltaNet
  from vllm.forward_context import get_forward_context
  self._gdn_saved={};self._gdn_original=Qwen3NextGatedDeltaNet._forward_core;worker=self
  @functools.wraps(self._gdn_original)
  def call(obj,mixed_qkv,b,a,core_attn_out):
   ctx=get_forward_context()
   md=ctx.attn_metadata.get(obj.prefix) if isinstance(ctx.attn_metadata,dict) else None
   inputs=(mixed_qkv,b,a,core_attn_out)
   if md is not None and mixed_qkv.shape[0]==tokens and md.num_prefill_tokens and obj.prefix not in worker._gdn_saved:
    ctx=copy.copy(ctx);ctx.attn_metadata=copy.deepcopy(ctx.attn_metadata)
    def clone(x):
     y=torch.empty_strided(x.shape,x.stride(),device=x.device,dtype=x.dtype);y.copy_(x);return y
    worker._gdn_saved[obj.prefix]=(obj,tuple(clone(x) for x in inputs),{},ctx)
   return worker._gdn_original(obj,*inputs)
  Qwen3NextGatedDeltaNet._forward_core=call
  return result
 def q80_stop_chain_capture(self):
  from vllm.model_executor.models.qwen3_next import Qwen3NextGatedDeltaNet
  Qwen3NextGatedDeltaNet._forward_core=self._gdn_original
  result=self.q80_stop_moe_capture();result['gdn_layers']=len(self._gdn_saved);return result
 def q80_benchmark_native_gdn_chain(self):
  import torch.distributed as dist
  from vllm.distributed import get_ep_group
  from vllm.forward_context import override_forward_context
  rank=self.vllm_config.parallel_config.data_parallel_rank;saved=list(self._gdn_saved.values());assert len(saved) in (0,36)
  def run():
   for obj,args,kwargs,ctx in saved:
    with override_forward_context(ctx):obj._forward_core(*args,**kwargs)
  flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8);values=[]
  with torch.inference_mode():
   for _ in range(3):run()
   for _ in range(20):
    flush.zero_();dist.barrier(group=get_ep_group().device_group);torch.cuda.synchronize()
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record();run();end.record();end.synchronize();values.append(start.elapsed_time(end))
   t=torch.tensor(values,device='cuda',dtype=torch.float64);dist.all_reduce(t,op=dist.ReduceOp.MAX,group=get_ep_group().device_group)
  d=dict(rank=rank,tokens=self._moe_tokens,layers=len(saved),local_ms=values,max_rank_ms=t.cpu().tolist(),median_ms=st.median(t.cpu().tolist()))
  p=ROOT/'native_chain'/f'gdn_dp{rank}_p{self._moe_tokens}.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(d,indent=2));self._gdn_saved={}
  return {k:v for k,v in d.items() if not isinstance(v,list)}
