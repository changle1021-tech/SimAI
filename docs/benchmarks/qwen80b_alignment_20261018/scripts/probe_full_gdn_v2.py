import pathlib,time,os,json,types,traceback,statistics as st
r=pathlib.Path(__file__).resolve().parent
while not (r/'native_moe_replay.done').exists():time.sleep(1)
os.environ['CUDA_VISIBLE_DEVICES']='4';os.environ['OMP_NUM_THREADS']='1';os.environ['TORCHINDUCTOR_COMPILE_THREADS']='2'
import torch
from workload_generator.mocked_model.inference import qwen3_next_attention as helper
from workload_generator.mocked_model.inference.qwen3_next_projections import benchmark_gpu_region
from vllm.forward_context import get_forward_context,override_forward_context
from vllm.model_executor.layers.layernorm import RMSNormGated
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
  row['norm_default_dispatch']=obj.norm._forward_method.__name__
  obj.norm._forward_method=obj.norm.forward_native
  x=torch.randn(tokens,2048,device='cuda',dtype=torch.bfloat16);residual=torch.randn_like(x);output=torch.empty_like(x)
  pre=GemmaRMSNorm(2048,1e-6).to(device='cuda',dtype=torch.bfloat16)
  post=GemmaRMSNorm(2048,1e-6).to(device='cuda',dtype=torch.bfloat16)
  def run():
   hidden,res=pre.forward_native(x,residual)
   obj.forward(hidden,output)
   return post.forward_native(output,res)
  with torch.inference_mode(),override_forward_context(ctx):
   row['core_only_ms']=[benchmark_gpu_region(core_fn,timing_mode='eager',samples=20)*1e3 for _ in range(3)]
   original_cache=obj.kv_cache
   row['cache_layouts']=[]
   for pages,padded in [(32,False),(32,True),(2090,True)]:
    if padded:
     page_elements=544*2048//2
     pool=torch.zeros(pages,page_elements,device='cuda',dtype=torch.bfloat16)
     conv=pool.as_strided((pages,3,8192),(page_elements,8192,1))
     state=pool.as_strided((pages,32,128,128),(page_elements,16384,128,1),storage_offset=3*8192)
    else:
     conv=torch.zeros(pages,3,8192,device='cuda',dtype=torch.bfloat16)
     state=torch.zeros(pages,32,128,128,device='cuda',dtype=torch.bfloat16)
    obj.kv_cache=[(conv,state)]
    md=ctx.attn_metadata['aicb.gdn'];md.non_spec_state_indices_tensor.fill_(pages-1)
    value=[benchmark_gpu_region(core_fn,timing_mode='eager',samples=20)*1e3 for _ in range(3)]
    row['cache_layouts'].append(dict(pages=pages,padded=padded,core_ms=value,ssm_stride=list(state.stride())))
   obj.kv_cache=original_cache;md.non_spec_state_indices_tensor.zero_()
   row['full_eager_ms']=[benchmark_gpu_region(run,timing_mode='eager',samples=20)*1e3 for _ in range(3)]
   try:
    compiled=torch.compile(run,fullgraph=True,dynamic=True)
    row['full_compiled_ms']=[benchmark_gpu_region(compiled,timing_mode='eager',samples=20)*1e3 for _ in range(3)]
   except Exception:row['compile_error']=traceback.format_exc()
   try:
    def before_core():
     hidden,res=pre.forward_native(x,residual)
     qkv,_=obj.in_proj_qkvz(hidden);ba,_=obj.in_proj_ba(hidden)
     q,k,v,z,b,a=obj.fix_query_key_value_ordering(qkv,ba)
     mixed=torch.cat((q.flatten(1),k.flatten(1),v.flatten(1)),dim=-1)
     core=torch.zeros(tokens,32,128,device='cuda',dtype=torch.bfloat16)
     return mixed,b,a,core,z,res
    def after_core(core,z,res):
     attn=obj.norm(core.reshape(-1,128),z.reshape(-1,128)).reshape(tokens,4096)
     out,_=obj.out_proj(attn)
     return post.forward_native(out,res)
    before=torch.compile(before_core,fullgraph=True,dynamic=True)
    after=torch.compile(after_core,fullgraph=True,dynamic=True)
    def split_compiled():
     mixed,b,a,core,z,res=before()
     torch.ops.vllm.gdn_attention_core(mixed,b,a,core,obj.prefix)
     return after(core,z,res)
    row['split_compiled_ms']=[benchmark_gpu_region(split_compiled,timing_mode='eager',samples=20)*1e3 for _ in range(3)]
   except Exception:row['split_compile_error']=traceback.format_exc()
 except Exception:row['error']=traceback.format_exc()
 rows.append(row);(r/'full_gdn_probe_v2.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
(r/'full_gdn_probe_v2.done').write_text('0');print('COMPLETE',flush=True)
