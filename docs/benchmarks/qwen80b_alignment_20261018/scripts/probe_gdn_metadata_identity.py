import pathlib,json,types,copy,time,os,statistics as st
r=pathlib.Path(__file__).resolve().parent
while not (r/'phase_profile_v5.done').exists():time.sleep(1)
assert (r/'phase_profile_v5.done').read_text()=='0'
os.environ.update(CUDA_VISIBLE_DEVICES='5',OMP_NUM_THREADS='1')
import torch
from vllm.forward_context import override_forward_context
from workload_generator.mocked_model.inference.qwen3_next_attention import _make_gdn_native_core
args=types.SimpleNamespace(**json.loads((r/'source/SimAI/aicb/scripts/inference_configs/qwen3_next_default.json').read_text()));args.tensor_model_parallel_size=1;args.phase='prefill';args.context_length=0
rows=[]
for tokens in (256,512):
 saved=[_make_gdn_native_core(args,'prefill',tokens) for _ in range(36)]
 common=saved[0][1].attn_metadata['aicb.gdn'].non_spec_query_start_loc
 def reset(shared):
  cu=torch.tensor([0,tokens],device='cuda',dtype=torch.int32)
  for obj,ctx,inputs in saved:ctx.attn_metadata[obj.prefix].non_spec_query_start_loc=cu if shared else cu.clone()
 def run():
  for obj,ctx,inputs in saved:
   with override_forward_context(ctx):torch.ops.vllm.gdn_attention_core(*inputs,obj.prefix)
 flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
 with torch.inference_mode():
  for label,shared,fresh in [('shared_warm',True,False),('shared_fresh_per_chunk',True,True),('cloned_per_layer',False,False)]:
   reset(shared)
   for _ in range(3):run()
   values=[]
   for _ in range(15):
    if fresh:reset(shared)
    flush.zero_();torch.cuda.synchronize();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();run();b.record();b.synchronize();values.append(a.elapsed_time(b))
   row=dict(tokens=tokens,mode=label,median_ms=st.median(values),samples_ms=values);rows.append(row);print(json.dumps(row),flush=True)
 (r/'gdn_metadata_identity.json').write_text(json.dumps(rows,indent=2))
(r/'gdn_metadata_identity.done').write_text('0')
