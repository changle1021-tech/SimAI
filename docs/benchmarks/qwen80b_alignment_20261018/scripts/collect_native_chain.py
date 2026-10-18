import json
from collect_runtime_audit import rpc,ROOT,URL
from benchmark_single_request_absolute import request
out=ROOT/'native_chain';out.mkdir(exist_ok=True);full=json.loads((ROOT/'baseline/prompt_8192.json').read_text());rows=[]
def run(n,label):
 q=request(URL,'qwen80b-audit',full[:n],64,42);q['label']=label;rows.append(q);(out/'requests.json').write_text(json.dumps(rows));print(json.dumps({k:v for k,v in q.items() if k not in ('stream_events','usage')}),flush=True)
for n in (256,512):
 for _ in range(3):run(n,'warmup')
 for _ in range(5):run(n,'control')
 rpc('q80_start_timers')
 try:
  for _ in range(3):run(n,'boundary')
 finally:rpc('q80_stop_timers',f'chain_p{n}_o64')
 rpc('q80_start_chain_capture',n);run(n,'capture');captured=rpc('q80_stop_chain_capture');print('CAPTURE',json.dumps(captured),flush=True)
 print('GDN_PROFILE',json.dumps(rpc('q80_benchmark_native_gdn_chain')),flush=True)
 print('MOE_PROFILE',json.dumps(rpc('q80_benchmark_continuous_moe')),flush=True)
 for _ in range(3):run(n,'post_control')
for _ in range(3):run(8192,'long_control')
print('COMPLETE',flush=True)
