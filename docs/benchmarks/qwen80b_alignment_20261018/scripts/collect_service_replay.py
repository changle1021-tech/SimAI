import pathlib,json
from collect_runtime_audit import rpc,ROOT,URL
from benchmark_single_request import request
out=ROOT/'service_replay';out.mkdir(exist_ok=True);rows=[]
full=json.loads((ROOT/'baseline/prompt_8192.json').read_text())
for n in (256,512,1024,2048,8192):
 ids=full[:n]
 for _ in range(2):request(URL,'qwen80b-audit',ids,64,42)
 rpc('q80_start_timers')
 try:
  for _ in range(3):rows.append(request(URL,'qwen80b-audit',ids,64,42))
 finally:rpc('q80_stop_timers',f'boundary_p{n}_o64')
 (out/'boundary_requests.json').write_text(json.dumps(rows))
rpc('q80_start_replay_capture')
request(URL,'qwen80b-audit',full[:512],16,42)
replay=rpc('q80_replay_saved');(out/'replay_results.json').write_text(json.dumps(replay,indent=2));print(json.dumps(replay),flush=True)
print('COMPLETE',flush=True)
