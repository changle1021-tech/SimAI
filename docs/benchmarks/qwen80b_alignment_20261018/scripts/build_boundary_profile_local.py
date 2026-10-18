import pathlib,json,statistics as st,collections,hashlib
r=pathlib.Path(__file__).resolve().parents[1]
groups={};sources={}
for p in sorted((r/'timers').glob('boundary_p*/dp*.json')):
 sources[str(p.relative_to(r))]=hashlib.sha256(p.read_bytes()).hexdigest()
 for row in json.loads(p.read_text())['rows']:
  it=row.get('iteration')
  if not it or not it.get('request_tokens'):continue
  assert len(it['request_tokens'])==1
  request=next(iter(it['request_tokens']));phase='prefill' if it['total_tokens']>1 else 'decode'
  key=(str(p.relative_to(r)),request,phase,it['total_tokens'] if phase=='prefill' else 1)
  entry=groups.setdefault(key,{'sums':collections.defaultdict(float),'counts':collections.Counter()})
  entry['sums'][row['kind']]+=row['device_elapsed_ms'];entry['counts'][row['kind']]+=1
points=collections.defaultdict(list)
for (file,req,phase,tokens),d in groups.items():
 s,c=d['sums'],d['counts']
 assert c['execute']==c['model']==c['sample'],(file,req,c)
 points[(phase,tokens)].append(((s['execute']-s['model'])/c['execute'],s['sample']/c['sample']))
out=dict(schema='qwen_worker_perimeter_v1',scope='execute_minus_model_plus_sample_gpu',configuration=dict(model='qwen3-next-80B',world_size=4,tp=1,pp=1,ep=4,batch_size=1),phases={},source_sha256=sources,notes=['No component wrappers in these measurements. Outer model/execute/sample events only.','Inclusive bookkeeping wall time contains GPU waits; it is not added.'])
for (phase,tokens),v in sorted(points.items()):
 row=dict(tokens=tokens,execute_minus_model_ms=st.median(a for a,b in v),sample_gpu_ms=st.median(b for a,b in v),samples=len(v),execute_minus_model_range_ms=[min(a for a,b in v),max(a for a,b in v)])
 out['phases'].setdefault(phase,[]).append(row);print(phase,row)
(r/'cpu_boundary_profile.json').write_text(json.dumps(out,indent=2))
