import pathlib,json,collections,statistics as st
r=pathlib.Path(__file__).resolve().parents[1];allout=[]
for n in (256,512,2048):
 for mode in ('cpu','gpu'):
  samples=[]
  for p in (r/'components'/f'fx_{mode}_p{n}_o16').glob('dp*.json'):
   rows=[x for x in json.loads(p.read_text())['rows'] if x.get('iteration') and x['iteration']['total_tokens']>1]
   by_req=collections.defaultdict(list)
   for x in rows:by_req[next(iter(x['iteration']['request_tokens']))].append(x)
   for req,rr in by_req.items():
    chunks=n//512 if n>=512 else 1
    ss=collections.defaultdict(float);cc=collections.Counter()
    for x in rr:
     if x['kind'] in ('compiled_fx','gdn_core_chain','moe','flash_attention'):
      ss[x['kind']+'_cpu_ms']+=x['cpu_ms'];cc[x['kind']]+=1
      if 'gpu_ms' in x:ss[x['kind']+'_gpu_ms']+=x['gpu_ms']
    outercpu=outergpu=0;patterns=collections.Counter()
    for fx in [x for x in rr if x['kind']=='compiled_fx']:
     children=[x for x in rr if x['kind'] in ('gdn_core_chain','moe','flash_attention') and fx['cpu_start_ns']<=x['cpu_start_ns'] and x['cpu_end_ns']<=fx['cpu_end_ns']]
     patterns[tuple(sorted(x['kind'] for x in children))]+=1
     outercpu+=fx['cpu_ms']-sum(x['cpu_ms'] for x in children)
     if 'gpu_ms' in fx:outergpu+=fx['gpu_ms']-sum(x.get('gpu_ms',0) for x in children)
    ss['compiled_outer_exclusive_cpu_ms']=outercpu;ss['compiled_outer_exclusive_gpu_ms']=outergpu
    samples.append({k:v/chunks for k,v in ss.items()});
   if mode=='gpu' and by_req:print('FX patterns',n,dict(patterns))
  keys=set().union(*(x.keys() for x in samples));out=dict(tokens=n,mode=mode,requests=len(samples),per_chunk_median={k:st.median(x[k] for x in samples) for k in keys});allout.append(out);print(json.dumps(out,indent=2))
(r/'compiled_decomposition_summary.json').write_text(json.dumps(allout,indent=2))
