import pathlib,json,types,statistics as st,os
os.environ.update(CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='2')
import torch
from workload_generator.mocked_model.inference import qwen3_next_attention as h
from workload_generator.mocked_model.inference.qwen3_next_projections import benchmark_gpu_region
r=pathlib.Path(__file__).resolve().parent
args=types.SimpleNamespace(**json.loads((r/'source/SimAI/aicb/scripts/inference_configs/qwen3_next_default.json').read_text()));args.tensor_model_parallel_size=1;args.phase='prefill';args.context_length=0
rows=[]
for n in (256,512):
 row=dict(tokens=n)
 def measure(args,fn):
  row['isolated_ms']=[benchmark_gpu_region(fn,samples=20,timing_mode='eager')*1e3 for _ in range(3)]
  for repeat in (1,8,32):
   times=[]
   for _ in range(15):
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8);flush.zero_();torch.cuda.synchronize()
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):fn()
    end.record();end.synchronize();times.append(start.elapsed_time(end)/repeat)
   row[f'repeat{repeat}_ms']=times
  return 0.
 h.benchmark_component=measure;h.benchmark_gdn_compiled_chain(args,n);rows.append(row)
 print(json.dumps({k:st.median(v) if isinstance(v,list) else v for k,v in row.items()}),flush=True)
 (r/'gdn_continuous.json').write_text(json.dumps(rows,indent=2))
(r/'gdn_continuous.done').write_text('0')
