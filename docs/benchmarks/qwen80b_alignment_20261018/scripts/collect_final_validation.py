import pathlib,json
from collect_runtime_audit import rpc,ROOT,URL
from benchmark_single_request import request
out=ROOT/'final_validation';out.mkdir(exist_ok=True);rows=[]
full=json.loads((ROOT/'baseline/prompt_8192.json').read_text())
for n,m in [(512,256),(2048,256),(8192,256),(256,64),(1024,128),(4096,128)]:
 ids=json.loads((ROOT/'baseline'/f'prompt_{n}.json').read_text()) if n in (512,2048,8192) else full[:n]
 (out/f'prompt_{n}.json').write_text(json.dumps(ids))
 for label,count in [('warmup',3),('validation',5)]:
  for i in range(count):
   row=request(URL,'qwen80b-audit',ids,m,42);row.update(label=label);rows.append(row)
   (out/'requests.json').write_text(json.dumps(rows));print(json.dumps({k:v for k,v in row.items() if k not in ('usage','stream_events')}),flush=True)
# Calibration of cache misses is performed after uninstrumented validation.
rpc('q80_start_cache_audit')
try:
 for n in (512,512,2048):
  ids=json.loads((ROOT/'baseline'/f'prompt_{n}.json').read_text());request(URL,'qwen80b-audit',ids,16,42)
finally:rpc('q80_stop_cache_audit')
(out/'workers.json').write_text(json.dumps(rpc('q80_info'),indent=2))
print('COMPLETE',flush=True)
