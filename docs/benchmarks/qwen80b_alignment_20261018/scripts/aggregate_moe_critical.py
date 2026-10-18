import pathlib,json,statistics as st,hashlib,sys
r=pathlib.Path(__file__).resolve().parent;a=r/'source/SimAI/aicb';sys.path.insert(0,str(a))
from workload_generator.mocked_model.inference.qwen3_next_moe import critical_profile_identity
rows=[row for rank in range(4) for row in json.loads((r/f'moe_critical/ep{rank}.json').read_text())]
config=json.loads((a/'scripts/inference_configs/qwen3_next_default.json').read_text())
by_sample={}
for x in rows:
 k=(x['phase'],x['tokens'],x['layer'],x['request'],x['step']);by_sample.setdefault(k,[]).append(x)
by_layer={}
for (phase,tokens,layer,request,step),rr in by_sample.items():
 assert {x['ep_rank'] for x in rr}=={0,1,2,3}
 key=(phase,tokens,int(layer.split('.')[2]));by_layer.setdefault(key,[]).append(max(x['compute_ms'] for x in rr))
out={'schema':'qwen_moe_critical_compute_v1','producer_sha256':critical_profile_identity(),
 'configuration':dict(world_size=4,tp=1,pp=1,ep=4,hidden_size=config['hidden_size'],num_experts=config['num_experts'],topk=config['num_experts_per_tok'],intermediate_size=config['moe_intermediate_size'],shared_intermediate_size=config['shared_expert_intermediate_size'],precision='bf16'),
 'measurement':'Per sample max of four independent EP shard measurements, then mean per layer. Real router IDs; native stream dependencies; no AG/RS.',
 'corpus_sha256':{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((r/'route_corpus/shared_on_p512_o256').glob('*.pt'))},'phases':{},'variability':{}}
for (phase,tokens,layer),values in by_layer.items():
 out['phases'].setdefault(phase,{}).setdefault(str(tokens),{})[str(layer)]=st.mean(values)
 out['variability'].setdefault(phase,{}).setdefault(str(tokens),{})[str(layer)]=dict(count=len(values),min_ms=min(values),max_ms=max(values),stdev_ms=st.stdev(values) if len(values)>1 else 0)
(r/'moe_compute_profile.json').write_text(json.dumps(out,indent=2))
for phase,points in out['phases'].items():
 for tokens,layers in points.items():print('MOE',phase,tokens,'total_ms',sum(layers.values()),'range_layer_ms',min(layers.values()),max(layers.values()))
