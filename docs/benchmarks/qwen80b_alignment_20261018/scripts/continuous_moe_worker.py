import torch,json,statistics as st
from native_moe_replay_worker import NativeMoEReplayWorker
from qwen_audit_worker import ROOT
class ContinuousMoEWorker(NativeMoEReplayWorker):
 def q80_benchmark_continuous_moe(self):
  import torch.distributed as dist
  from vllm.forward_context import override_forward_context
  from vllm.distributed import get_ep_group
  assert len(self._moe_saved)==48
  rank=self.vllm_config.parallel_config.data_parallel_rank
  saved=[self._moe_saved[n] for n in sorted(self._moe_saved,key=lambda n:int(n.split('.')[2]))]
  def run(add_output):
   for obj,x,logits,ctx in saved:
    with override_forward_context(ctx):
     out=torch.ops.vllm.moe_forward_shared(x,logits,obj.layer_name)
     if add_output:out[0]+out[1]
  values={};flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
  with torch.inference_mode():
   for add in (False,True):
    for _ in range(3):run(add)
    times=[]
    for _ in range(20):
     flush.zero_();dist.barrier(group=get_ep_group().device_group);torch.cuda.synchronize()
     start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
     start.record();run(add);end.record();end.synchronize();times.append(start.elapsed_time(end))
    t=torch.tensor(times,device='cuda',dtype=torch.float64);dist.all_reduce(t,op=dist.ReduceOp.MAX,group=get_ep_group().device_group)
    values['with_output_add' if add else 'without_output_add']=dict(local_ms=times,max_rank_ms=t.cpu().tolist(),median_ms=st.median(t.cpu().tolist()))
  d=dict(rank=rank,tokens=self._moe_tokens,profiles=values)
  p=ROOT/'continuous_moe'/f'dp{rank}_p{self._moe_tokens}.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(d,indent=2));self._moe_saved={}
  return dict(rank=rank,tokens=self._moe_tokens,median_ms={k:v['median_ms'] for k,v in values.items()})
