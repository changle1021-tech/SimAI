import json
from collect_runtime_audit import rpc,ROOT,URL
from benchmark_single_request import request
out=ROOT/'compiled_decomposition';out.mkdir(exist_ok=True);rows=[]
full=json.loads((ROOT/'baseline/prompt_8192.json').read_text())
def run(n,label):
 row=request(URL,'qwen80b-audit',full[:n],16,42);row['label']=label;rows.append(row);(out/'requests.json').write_text(json.dumps(rows));print(json.dumps({k:v for k,v in row.items() if k not in ('usage','stream_events')}),flush=True)
for n in (256,512,2048):
 for _ in range(3):run(n,'warmup')
 for _ in range(3):run(n,'control')
 for gpu in (False,True):
  rpc('q80_start_decomposition',gpu)
  try:
   for _ in range(3 if not gpu else 1):run(n,'gpu_components' if gpu else 'cpu_components')
  finally:rpc('q80_stop_decomposition',f'fx_{"gpu" if gpu else "cpu"}_p{n}_o16')
 for _ in range(2):run(n,'post_control')
# A heavy trace is for kernel counts/dependencies, not timing calibration.
rpc('q80_start_trace','native_p256_o16')
try:run(256,'kernel_trace')
finally:rpc('q80_stop_trace')
print('COMPLETE',flush=True)
