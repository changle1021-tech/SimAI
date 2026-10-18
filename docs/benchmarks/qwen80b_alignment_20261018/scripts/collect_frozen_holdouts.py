import pathlib,json,random,subprocess,os,csv,statistics as st
from collect_runtime_audit import ROOT,URL
from benchmark_single_request_absolute import request
out=ROOT/'frozen_holdouts';out.mkdir(exist_ok=True);full=json.loads((ROOT/'baseline/prompt_8192.json').read_text());rows=[]
cases=[(2560,96),(6144,96)]
def run(n,m,label):
 q=request(URL,'qwen80b-audit',full[:n],m,42);q['label']=label;rows.append(q);(out/'requests.json').write_text(json.dumps(rows));print(json.dumps({k:v for k,v in q.items() if k not in ('stream_events','usage')}),flush=True)
for n,m in cases:
 for _ in range(3):run(n,m,'warmup')
order=cases*8;random.Random(101804).shuffle(order)
for n,m in order:run(n,m,'control')
results=[];template=json.loads((ROOT/'sim_alignment_v5/p512_o256/command.json').read_text())
for n,m in cases:
 dest=out/f'sim_p{n}_o{m}';cmd=template[:]
 for flag,value in (('--fixed_request_length_generator_config_prefill_tokens',n),('--fixed_request_length_generator_config_decode_tokens',m),('--metrics_config_output_dir',dest)):
  cmd[cmd.index(flag)+1]=str(value)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',QWEN_AUDIT_ROWS=str(dest/'iterations.jsonl'))
 dest.mkdir(exist_ok=True);(dest/'command.json').write_text(json.dumps(cmd,indent=2))
 with (dest/'run.log').open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
 assert p.returncode==0,(dest/'run.log').read_text()[-3000:]
 metrics=list(dest.rglob('request_metrics.csv'));assert len(metrics)==1
 row=list(csv.DictReader(metrics[0].open()))[0];pred=dict(ttft_ms=float(row['prefill_e2e_time'])*1e3,e2e_ms=float(row['request_e2e_time'])*1e3);pred['tpot_ms']=(pred['e2e_ms']-pred['ttft_ms'])/(m-1)
 vv=[q for q in rows if q['label']=='control' and q['input_tokens']==n];entry=dict(input_tokens=n,output_tokens=m,samples=len(vv),metrics={})
 for metric in ('ttft_ms','tpot_ms','e2e_ms'):
  actual=st.median(q[metric] for q in vv);entry['metrics'][metric]=dict(actual=actual,predicted=pred[metric],error_pct=(pred[metric]/actual-1)*100,range=[min(q[metric] for q in vv),max(q[metric] for q in vv)])
 results.append(entry);(out/'summary.json').write_text(json.dumps(results,indent=2));print('VALIDATION',json.dumps(entry),flush=True)
print('COMPLETE',flush=True)
