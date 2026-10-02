"""Native vLLM NCCL GPU primitives on the actual Ray rank placement."""
import argparse,csv,json,os,statistics,time
from pathlib import Path
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
p=argparse.ArgumentParser();p.add_argument('--rank-node-map',required=True);p.add_argument('--tensor-width',type=int,default=4096);p.add_argument('--tokens',type=int,nargs='+',default=[1,2,4,8,16,32,64,128,256,512,1024,2048,4096]);p.add_argument('--network-transport',required=True);p.add_argument('--output',required=True);p.add_argument('--master-port',type=int,default=29599);p.add_argument('--repetitions',type=int,default=10);a=p.parse_args()
@ray.remote(num_gpus=1,num_cpus=1)
class Rank:
 def initialize(self,rank,world,endpoint):
  import torch
  from vllm.distributed import init_distributed_environment,initialize_model_parallel,get_tp_group
  from vllm.distributed.parallel_state import set_custom_all_reduce
  torch.cuda.set_device(0);set_custom_all_reduce(False)
  init_distributed_environment(world_size=world,rank=rank,distributed_init_method=endpoint,local_rank=0,backend='nccl')
  initialize_model_parallel(tensor_model_parallel_size=world,pipeline_model_parallel_size=1)
  self.rank=rank;self.world=world;self.group=get_tp_group()
  return dict(rank=rank,device_name=torch.cuda.get_device_name(),node_id=ray.get_runtime_context().get_node_id())
 def measure(self,tokens,width,mode,repetitions):
  import torch
  group=self.group;x=torch.zeros((tokens,width),device='cuda',dtype=torch.float16);count=64
  for _ in range(3):group.all_reduce(x)
  torch.cuda.synchronize()
  graph=None
  if mode=='cuda_graph':
   with group.graph_capture() as capture:
    for _ in range(3):group.all_reduce(x)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=capture.stream):
     for _ in range(count):group.all_reduce(x)
   forward=graph.replay
  else:
   def forward():
    for _ in range(count):group.all_reduce(x)
  pairs=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(repetitions)]
  for start,end in pairs:start.record();end.record()
  torch.cuda.synchronize()
  # These 64-operation spans amortize the graph replay call and event overhead.
  for _ in range(2):forward()
  torch.cuda.synchronize()
  for start,end in pairs:start.record();forward();end.record()
  torch.cuda.synchronize()
  return [start.elapsed_time(end)/count for start,end in pairs]
 def close(self):
  from vllm.distributed import destroy_model_parallel,destroy_distributed_environment
  destroy_model_parallel();destroy_distributed_environment()

def main():
 address=os.environ.get('RAY_ADDRESS')
 if address:ray.init(address=address,ignore_reinit_error=True)
 else:ray.init(ignore_reinit_error=True,num_cpus=8,object_store_memory=1024**3)
 nodes=sorted([n for n in ray.nodes() if n['Alive'] and n['Resources'].get('GPU',0)>0],key=lambda n:n['NodeManagerAddress'])
 layout=json.loads(a.rank_node_map);world=len(layout)
 if world<2 or max(layout)>=len(nodes):raise ValueError('Requested native collective rank placement is unavailable')
 for index in set(layout):
  if layout.count(index)>nodes[index]['Resources']['GPU']:raise ValueError('Insufficient GPUs for requested placement')
 ranks=[Rank.options(scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[node]['NodeID'],soft=False)).remote() for node in layout]
 endpoint=f"tcp://{nodes[layout[0]]['NodeManagerAddress']}:{a.master_port}"
 try:
  environment=ray.get([rank.initialize.remote(i,world,endpoint) for i,rank in enumerate(ranks)])
  rows=[]
  for tokens in a.tokens:
   for mode in ['eager','cuda_graph']:
    values=ray.get([rank.measure.remote(tokens,a.tensor_width,mode,a.repetitions) for rank in ranks]);span=[max(v) for v in zip(*values)]
    row=dict(profile_kind='native_collective_v1',num_workers=world,devices_per_node=max(layout.count(i) for i in set(layout)),rank_node_map=json.dumps(layout,separators=(',',':')),network_transport=a.network_transport,collective='all_reduce',execution_mode=mode,dtype='float16',size=tokens*a.tensor_width*2,device_name=environment[0]['device_name'],vllm_version='0.5.1',timing_semantics='native collective GPU span divided by 64 dependent operations; eager and CUDA graph measured separately')
    for stat,value in [('mean',statistics.mean(span)),('median',statistics.median(span)),('min',min(span)),('max',max(span)),('std',statistics.pstdev(span))]:row[f'time_stats.all_reduce.{stat}']=value
    rows.append(row);print(json.dumps(row),flush=True)
  out=Path(a.output)
  with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  out.with_suffix('.metadata.json').write_text(json.dumps(dict(arguments=vars(a),environment=environment,rank_layout=layout),indent=2))
 finally:
  ray.get([r.close.remote() for r in ranks]);[ray.kill(r) for r in ranks];ray.shutdown()
if __name__=='__main__':main()
