import os,pathlib,json,types,time,subprocess,concurrent.futures,traceback
r=pathlib.Path(__file__).resolve().parent
# This driver is also a worker to keep CUDA processes independent.
if 'Q80_EP_RANK' not in os.environ:
 while not (r/'comm_profile_v2.done').exists():time.sleep(1)
 out=r/'moe_critical';out.mkdir(exist_ok=True)
 def worker(rank):
  env=dict(os.environ,PYTHONPATH=str(r/'source/SimAI/aicb'),Q80_EP_RANK=str(rank),CUDA_VISIBLE_DEVICES=str(4+rank),OMP_NUM_THREADS='1',CUDA_HOME='/usr/local/cuda-12.8',PATH='/usr/local/cuda-12.8/bin:'+os.environ['PATH'])
  with (out/f'ep{rank}.log').open('w') as log:
   return subprocess.run([str(r/'env/bin/python'),'-u',__file__],cwd=r/'source/SimAI/aicb',env=env,stdout=log,stderr=subprocess.STDOUT).returncode
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:codes=list(ex.map(worker,range(4)))
 (r/'moe_critical.done').write_text(json.dumps(codes));print('COMPLETE',codes,flush=True)
else:
 import torch
 from workload_generator.mocked_model.inference.qwen3_next_moe import benchmark_moe_critical_compute
 rank=int(os.environ['Q80_EP_RANK']);out=r/'moe_critical';corpus=r/'route_corpus/shared_on_p512_o256'
 records=[]
 for p in sorted(corpus.glob('dp*.pt')):records+=torch.load(p,weights_only=False)
 args=types.SimpleNamespace(**json.loads((r/'source/SimAI/aicb/scripts/inference_configs/qwen3_next_default.json').read_text()))
 args.tensor_model_parallel_size=1;args.pipeline_model_parallel=1;args.world_size=4;args.expert_model_parallel_size=4
 results=[]
 for phase,tokens in [('prefill',256),('prefill',512),('decode',1)]:
  args.phase=phase
  selected=[x for x in records if x['phase']==phase and (phase=='prefill' or x['step'] in (1,64,255))]
  for i,row in enumerate(selected):
   ids=row['ids']
   if phase=='prefill' and tokens!=512:ids=ids.reshape(4,512,10)[:,:tokens].reshape(-1,10).contiguous()
   try:
    sec=benchmark_moe_critical_compute(args,tokens,ids,rank,samples=12)
    item={k:row[k] for k in ['phase','layer','request','step']};item.update(tokens=tokens,ep_rank=rank,compute_ms=sec*1e3);results.append(item)
    (out/f'ep{rank}.json').write_text(json.dumps(results))
    if i%48==0:print(json.dumps(item),flush=True)
   except Exception:
    print(traceback.format_exc(),flush=True);raise
 print('COMPLETE',rank,len(results),flush=True)
