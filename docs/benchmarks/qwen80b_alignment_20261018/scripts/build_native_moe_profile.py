import pathlib,json,hashlib,sys,math
r=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(r/'source/SimAI/aicb'))
from workload_generator.mocked_model.inference.qwen3_next_moe import critical_profile_identity,native_moe_runtime_identity
out=json.loads((r/'moe_compute_profile.json').read_text());assert out['producer_sha256']==critical_profile_identity()
out['schema']='qwen_moe_critical_compute_v2';out['time_scope_by_phase']={'decode':'compute_only','prefill':'compute_and_ep_comm'}
out['native_runtime_source_sha256']=native_moe_runtime_identity();out['native_replay_producer_sha256']=hashlib.sha256((r/'native_moe_replay_worker.py').read_bytes()).hexdigest();out['native_raw_sha256']={}
for n in (256,512):
 rank_rows=[]
 for rank in range(4):
  p=r/f'native_moe_replay/dp{rank}_p{n}.json';out['native_raw_sha256'][str(p.relative_to(r))]=hashlib.sha256(p.read_bytes()).hexdigest()
  d=json.loads(p.read_text());assert d['tokens']==n and len(d['rows'])==48
  rank_rows.append(d['rows'])
 for a,b in zip(rank_rows[0],rank_rows[1]):assert a['max_rank_ms']==b['max_rank_ms']
 points={str(int(x['layer'].split('.')[2])):x['median_ms'] for x in rank_rows[0]}
 assert set(points)==set(map(str,range(48))) and all(math.isfinite(v) and v>0 for v in points.values())
 out['phases']['prefill'][str(n)]=points
 print('NATIVE_PREFILL',n,sum(points.values()))
(r/'moe_profile_native_v2.json').write_text(json.dumps(out,indent=2))
