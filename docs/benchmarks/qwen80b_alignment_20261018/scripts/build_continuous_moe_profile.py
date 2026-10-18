import pathlib,json,hashlib,sys,math,statistics as st
r=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(r/'source/SimAI/aicb'))
from workload_generator.mocked_model.inference.qwen3_next_moe import critical_profile_identity,native_moe_runtime_identity
out=json.loads((r/'moe_profile_native_v2.json').read_text());assert out['producer_sha256']==critical_profile_identity()
out['native_runtime_source_sha256']=native_moe_runtime_identity()
out['native_measurement_scope']='continuous_48_layer_moe_chain'
out['layer_allocation']='isolated_layer_proportions_normalized_to_directly_measured_48_layer_total'
out['router_linear_included_by_phase']={'decode':True,'prefill':False}
out['native_replay_sources_sha256']={name:hashlib.sha256((r/name).read_bytes()).hexdigest() for name in ('continuous_moe_worker.py','native_chain_worker.py','native_moe_replay_worker.py','collect_native_chain.py')}
out['native_raw_sha256']={};out['aggregate_prefill_moe_ms']={};out['aggregation_factors']={}
for n in (256,512):
 values=[]
 for rank in range(4):
  p=r/f'continuous_moe/dp{rank}_p{n}.json';d=json.loads(p.read_text());assert d['tokens']==n and d['rank']==rank
  out['native_raw_sha256'][str(p.relative_to(r))]=hashlib.sha256(p.read_bytes()).hexdigest()
  values.append(d['profiles']['with_output_add']['max_rank_ms'])
 assert all(v==values[0] for v in values), 'Ranks must agree on max-rank samples'
 total=st.median(values[0]);assert math.isfinite(total) and total>0
 points=out['phases']['prefill'][str(n)];factor=total/sum(points.values())
 out['phases']['prefill'][str(n)]={k:v*factor for k,v in points.items()}
 out['aggregate_prefill_moe_ms'][str(n)]=total;out['aggregation_factors'][str(n)]=factor
 assert abs(sum(out['phases']['prefill'][str(n)].values())-total)<1e-8
 print('CONTINUOUS_MOE',n,total,'allocation_factor',factor,flush=True)
out['notes']=['Prefill aggregate measured directly over 48 real-weight native MoE calls; includes EP communication, shared stream dependencies and final output add.','Per-layer CSV allocation uses isolated layer proportions. The aggregate is measured; this allocation is not a fresh per-layer measurement.','External router linear is not in native FusedMoE replay and is profiled separately by AICB.','Decode retains independently measured real-route compute-only costs, with separate communication. No client E2E values enter model calibration.']
(r/'moe_profile_native_v3.json').write_text(json.dumps(out,indent=2))
