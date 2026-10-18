import pathlib,subprocess,os,json,concurrent.futures,time
r=pathlib.Path(__file__).resolve().parent;a=r/'source/SimAI/aicb'
while not (r/'service_replay_v2.done').exists():time.sleep(1)
status=json.loads((r/'service_replay_v2.done').read_text())
# A diagnostic RPC failure does not invalidate the independent operator grid.
out=r/'phase_profile_v3';out.mkdir(exist_ok=True)
work=[('prefill',256,0),('prefill',512,0),('prefill',512,512),('prefill',512,8192)]+[('decode',n,0) for n in (256,512,1024,2048,4096,8192,8446)]
def worker(gpu,jobs):
 rows=[]
 for phase,n,ctx in jobs:
  cmd=[str(r/'env/bin/python'),'-u','-m','workload_generator.Vidur_workload_generator','Qwen3-Next-80B','scripts/inference_configs/qwen3_next_default.json','--seq_length',str(n),'--micro_batch','1','--world_size','4','--tensor_model_parallel_size','1','--expert_model_parallel_size','4','--context_length',str(ctx),'--moe_compute_profile',str(r/'moe_compute_profile.json'),'--aiob_enable','--phase',phase,'--result_dir',str(out/'workload')]
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',CUDA_HOME='/usr/local/cuda-12.8',PATH='/usr/local/cuda-12.8/bin:'+os.environ['PATH'])
  log=out/f'{phase}_{n}_{ctx}.log'
  with log.open('w') as f:p=subprocess.run(cmd,cwd=a,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
  row=dict(phase=phase,seq=n,context=ctx,gpu=gpu,returncode=p.returncode);rows.append(row);print(json.dumps(row),flush=True)
  if p.returncode:print(log.read_text()[-4000:],flush=True)
 return rows
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 rows=[row for rr in ex.map(lambda pair:worker(*pair),[(4+i,work[i::3]) for i in range(3)]) for row in rr]
(out/'summary.json').write_text(json.dumps(rows,indent=2));(r/'phase_profile_v3.done').write_text('0' if all(x['returncode']==0 for x in rows) else '1')
