import copy,functools,json,time,pathlib,torch
from qwen_audit_worker import AuditWorker,ROOT
class NativeMoEReplayWorker(AuditWorker):
 def q80_start_moe_capture(self,tokens):
  from vllm.model_executor.layers.fused_moe.layer import FusedMoE
  from vllm.forward_context import get_forward_context
  self._moe_saved={};self._moe_tokens=tokens;self._moe_original=FusedMoE.forward_impl;worker=self
  def clone(v):
   if isinstance(v,torch.Tensor):return v.detach().clone()
   if isinstance(v,dict):return {k:clone(x) for k,x in v.items()}
   if isinstance(v,list):return [clone(x) for x in v]
   if isinstance(v,tuple):return tuple(clone(x) for x in v)
   return v
  @functools.wraps(worker._moe_original)
  def call(obj,hidden_states,router_logits):
   if hidden_states.shape[0]==tokens and obj.layer_name not in worker._moe_saved:
    ctx=copy.copy(get_forward_context())
    if ctx.dp_metadata is not None:
     md=copy.copy(ctx.dp_metadata)
     for k,v in vars(md).items():setattr(md,k,clone(v))
     ctx.dp_metadata=md
    worker._moe_saved[obj.layer_name]=(obj,hidden_states.detach().clone(),router_logits.detach().clone(),ctx)
   return worker._moe_original(obj,hidden_states,router_logits)
  FusedMoE.forward_impl=call
  return dict(dp_rank=self.vllm_config.parallel_config.data_parallel_rank,tokens=tokens)
 def q80_stop_moe_capture(self):
  from vllm.model_executor.layers.fused_moe.layer import FusedMoE
  FusedMoE.forward_impl=self._moe_original
  return dict(dp_rank=self.vllm_config.parallel_config.data_parallel_rank,tokens=self._moe_tokens,layers=len(self._moe_saved))
 def q80_benchmark_native_moe(self):
  import statistics as st,torch.distributed as dist
  from vllm.forward_context import override_forward_context
  from vllm.distributed import get_ep_group
  assert len(self._moe_saved)==48,len(self._moe_saved)
  rank=self.vllm_config.parallel_config.data_parallel_rank;tokens=self._moe_tokens
  p=ROOT/'native_moe_replay'/f'dp{rank}_p{tokens}.json';p.parent.mkdir(exist_ok=True)
  flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
  start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  rows=[]
  for name in sorted(self._moe_saved,key=lambda n:int(n.split('.')[2])):
   obj,x,logits,ctx=self._moe_saved[name]
   with torch.inference_mode(),override_forward_context(ctx):
    def run():
     out=torch.ops.vllm.moe_forward_shared(x,logits,name)
     return out[0]+out[1]
    for _ in range(3):run()
    torch.cuda.synchronize();values=[]
    for _ in range(20):
     flush.zero_();dist.barrier(group=get_ep_group().device_group);torch.cuda.synchronize()
     start.record();run();end.record();end.synchronize();values.append(start.elapsed_time(end))
    tensor=torch.tensor(values,device='cuda',dtype=torch.float64)
    dist.all_reduce(tensor,op=dist.ReduceOp.MAX,group=get_ep_group().device_group)
    critical=tensor.cpu().tolist()
    rows.append(dict(layer=name,ep_rank=rank,local_ms=values,max_rank_ms=critical,median_ms=st.median(critical)))
    p.write_text(json.dumps(dict(tokens=tokens,rows=rows),indent=2))
  self._moe_saved={}
  return dict(dp_rank=rank,tokens=tokens,layers=len(rows),total_ms=sum(x['median_ms'] for x in rows),path=str(p))
