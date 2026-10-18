import json,pathlib,urllib.request,time,traceback
from benchmark_single_request import request
from collect_runtime_audit import rpc,ROOT,URL
out=ROOT/'component_requests';out.mkdir(exist_ok=True)
rows=[]
def run(n,label,m=256):
    ids=json.loads((ROOT/'baseline'/f'prompt_{n}.json').read_text())
    row=request(URL,'qwen80b-audit',ids,m,42);row['label']=label;rows.append(row)
    (out/'requests.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps({k:v for k,v in row.items() if k not in ['stream_events','usage']}),flush=True)
(out/'workers.json').write_text(json.dumps(rpc('q80_alignment_info'),indent=2))
for n in [512,2048,8192]:
    for _ in range(3):run(n,'warmup')
    for _ in range(5):run(n,'control')
    rpc('q80_start_components',False)
    try:
        for _ in range(3):run(n,'cpu_components')
    finally:rpc('q80_stop_components',f'cpu_p{n}_o256')
    if n==512:
        rpc('q80_start_components',True)
        try:run(n,'gpu_components')
        finally:rpc('q80_stop_components',f'gpu_p{n}_o256')
    for _ in range(2):run(n,'post_control')
print('COMPLETE',flush=True)
