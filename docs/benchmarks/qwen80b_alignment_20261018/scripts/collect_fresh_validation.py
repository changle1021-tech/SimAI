import pathlib,json,random,time
from collect_runtime_audit import rpc,ROOT,URL
from benchmark_single_request_absolute import request
out=ROOT/'fresh_validation_20261018';out.mkdir(exist_ok=True);rows=[]
full=json.loads((ROOT/'baseline/prompt_8192.json').read_text())
cases=[(512,256),(2048,256),(8192,256),(256,64),(1024,128),(4096,128),(1536,192),(3072,192)]
def run(n,m,label):
 row=request(URL,'qwen80b-audit',full[:n],m,42);row.update(label=label);rows.append(row)
 (out/'requests.json').write_text(json.dumps(rows));print(json.dumps({k:v for k,v in row.items() if k not in ('usage','stream_events')}),flush=True)
for n,m in cases:
 (out/f'prompt_{n}.json').write_text(json.dumps(full[:n]))
 for _ in range(3):run(n,m,'warmup')
 for _ in range(5):run(n,m,'control')
order=cases*3;random.Random(1018).shuffle(order)
for n,m in order:run(n,m,'interleaved_control')
for n,m in cases:
 rpc('q80_start_timers')
 try:
  for _ in range(3):run(n,m,'boundary')
 finally:rpc('q80_stop_timers',f'fresh_p{n}_o{m}')
(out/'workers.json').write_text(json.dumps(rpc('q80_info'),indent=2));print('COMPLETE',flush=True)
