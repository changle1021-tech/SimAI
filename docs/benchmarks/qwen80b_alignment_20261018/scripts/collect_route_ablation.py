import pathlib,json,os
from collect_runtime_audit import rpc,ROOT,URL
from benchmark_single_request import request
variant=os.environ['ALIGN_VARIANT'];out=ROOT/'route_ablation'/variant;out.mkdir(parents=True,exist_ok=True)
(out/'workers.json').write_text(json.dumps(rpc('q80_runtime_config'),indent=2));rows=[]
def run(n,m,label):
    ids=json.loads((ROOT/'baseline'/f'prompt_{n}.json').read_text());row=request(URL,'qwen80b-audit',ids,m,42);row['label']=label;rows.append(row);(out/'requests.json').write_text(json.dumps(rows,indent=2));print(json.dumps({k:v for k,v in row.items() if k not in ('usage','stream_events')}),flush=True)
for n in [512,2048,8192]:
    for _ in range(3):run(n,256,'warmup')
    for _ in range(5):run(n,256,'control')
    if variant=='shared_on':
        rpc('q80_start_routes')
        try:
            for _ in range(2):run(n,256,'route_capture')
        finally:rpc('q80_stop_routes',f'{variant}_p{n}_o256')
        for _ in range(2):run(n,256,'post_control')
print('COMPLETE',flush=True)
