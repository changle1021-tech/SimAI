import json
from collect_runtime_audit import rpc,ROOT,URL
from benchmark_single_request import request
out=ROOT/'native_moe_replay';out.mkdir(exist_ok=True);full=json.loads((ROOT/'baseline/prompt_8192.json').read_text());rows=[]
for n in (256,512):
 for _ in range(3):request(URL,'qwen80b-audit',full[:n],16,42)
 rpc('q80_start_moe_capture',n)
 request(URL,'qwen80b-audit',full[:n],16,42)
 captured=rpc('q80_stop_moe_capture');print('CAPTURE',json.dumps(captured),flush=True)
 assert all(x['layers']==48 for x in captured['results']),captured
 result=rpc('q80_benchmark_native_moe');print('PROFILE',json.dumps(result),flush=True);rows.append(result)
 (out/'summary.json').write_text(json.dumps(rows,indent=2))
print('COMPLETE',flush=True)
