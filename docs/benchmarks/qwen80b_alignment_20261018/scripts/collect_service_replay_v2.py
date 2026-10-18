import pathlib,json
from collect_runtime_audit import rpc,ROOT,URL
from benchmark_single_request import request
out=ROOT/'service_replay_v2';out.mkdir(exist_ok=True);rows=[]
full=json.loads((ROOT/'baseline/prompt_8192.json').read_text())
for n in (256,512):
 ids=full[:n]
 for _ in range(3):request(URL,'qwen80b-audit',ids,16,42)
 rpc('q80_start_components',False)
 try:
  for _ in range(3):rows.append(request(URL,'qwen80b-audit',ids,16,42))
 finally:rpc('q80_stop_components',f'microcpu_p{n}_o16')
 rpc('q80_start_replay_capture')
 request(URL,'qwen80b-audit',ids,16,42)
 replay=rpc('q80_replay_saved');(out/f'replay_p{n}.json').write_text(json.dumps(replay,indent=2));print(json.dumps(replay),flush=True)
(out/'requests.json').write_text(json.dumps(rows,indent=2))
print('COMPLETE',flush=True)
