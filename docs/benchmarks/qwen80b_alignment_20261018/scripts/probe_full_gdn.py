import pathlib,time,os,json,types,traceback,statistics as st
r=pathlib.Path(__file__).resolve().parent
while not (r/'compiled_decomposition.done').exists():time.sleep(1)
os.environ['CUDA_VISIBLE_DEVICES']='4';os.environ['OMP_NUM_THREADS']='1'
import torch
from workload_generator.mocked_model.inference import qwen3_next_attention as helper
from workload_generator.mocked_model.inference.qwen3_next_projections import benchmark_gpu_region
from vllm.forward_context import get_forward_context,override_forward_context
from vllm.model_executor.layers.fla.ops import RMSNormGated
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
args=types.SimpleNamespace(**json.loads((r/'source/SimAI/aicb/scripts/inference_configs/qwen3_next_default.json').read_text()));args.tensor_model_parallel_size=1;args.phase='prefill';args.context_length=0
class Linear(torch.nn.Module):
 def __init__(self,k,n):
  super().__init__();self.weight=torch.nn.Parameter(torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.02)
 def forward(self,x):return torch.nn.functional.linear(x,self.weight),None
rows=[]
for tokens in (256,512):
 row=dict(tokens=tokens)
 try:
  box=[];original=helper.benchmark_component
  def capture(args,fn):box.append((fn,get_forward_context()));return 0.
  helper.benchmark_component=capture
  try:helper.benchmark_gdn_native_core(args,'prefill',tokens)
  finally:helper.benchmark_component=original
  core_fn,ctx=box[0];obj=ctx.no_compile_layers['aicb.gdn']
  obj.in_proj_qkvz=Linear(2048,12288);obj.in_proj_ba=Linear(2048,64);obj.out_proj=Linear(4096,2048)
  obj.norm=RMSNormGated(128,eps=1e-6,group_size=None,norm_before_gate=True,device='cuda',dtype=torch.bfloat16)
  x=torch.randn(tokens,2048,device='cuda',dtype=torch.bfloat16);residual=torch.randn_like(x);output=torch.empty_like(x)
  pre=GemmaRMSNorm(2048,1e-6).to(device='cuda',dtype=torch.bfloat16)
  post=GemmaRMSNorm(2048,1e-6).to(device='cuda',dtype=torch.bfloat16)
  def run():
   hidden,res=pre.forward_cuda(x,residual)
   obj.forward(hidden,output)
   return post.forward_cuda(output,res)
  with torch.inference_mode(),override_forward_context(ctx):
   row['core_only_ms']=[benchmark_gpu_region(core_fn,timing_mode='eager',samples=20)*1e3 for _ in range(3)]
   row['full_eager_ms']=[benchmark_gpu_region(run,timing_mode='eager',samples=20)*1e3 for _ in range(3)]
   try:
    compiled=torch.compile(run,fullgraph=True,dynamic=True)
    row['full_compiled_ms']=[benchmark_gpu_region(compiled,timing_mode='eager',samples=20)*1e3 for _ in range(3)]
   except Exception:row['compile_error']=traceback.format_exc()
 except Exception:row['error']=traceback.format_exc()
 rows.append(row);(r/'full_gdn_probe.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
(r/'full_gdn_probe.done').write_text('0');print('COMPLETE',flush=True)
