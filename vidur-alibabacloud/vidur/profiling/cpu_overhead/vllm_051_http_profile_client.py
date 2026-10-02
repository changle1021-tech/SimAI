import argparse,asyncio,csv,json,statistics,time
from pathlib import Path
import aiohttp
from eager_dispatch_trace import dispatch_program, mean_program
p=argparse.ArgumentParser();p.add_argument('--server-output',required=True);p.add_argument('--model-name',required=True);p.add_argument('--model-name-for-vidur',required=True);p.add_argument('--tp',type=int,required=True);p.add_argument('--pp',type=int,required=True);p.add_argument('--batch-sizes',type=int,nargs='+',default=[1,2,3,4]);p.add_argument('--repetitions',type=int,default=5);p.add_argument('--network-transport',default='local');p.add_argument('--output',required=True);p.add_argument('--prompt-tokens',type=int,default=512);p.add_argument('--decode-tokens',type=int,default=50);p.add_argument('--enforce-eager',action='store_true');p.add_argument('--instrumented-warmup',type=int,default=10);p.add_argument('--warmup-repetitions',type=int,default=10);p.add_argument('--diagnostic-request-interval',type=float,default=0);a=p.parse_args();OUT=Path(a.server_output)
async def main():
 rows=[];audits=[];last_start=[None]
 if a.instrumented_warmup<1 or a.warmup_repetitions<1:raise ValueError('Profile warmups must be positive')
 if a.diagnostic_request_interval<0:raise ValueError('Diagnostic interval cannot be negative')
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
  async def control(endpoint,**kw):
   async with session.post('http://127.0.0.1:8089/diagnostic/profile/'+endpoint,params=kw) as r:
    if r.status!=200:raise RuntimeError(await r.text())
    return await r.json()
  async def batch(b):
   if last_start[0] is not None:
    await asyncio.sleep(max(0,a.diagnostic_request_interval-(time.perf_counter()-last_start[0])))
   last_start[0]=time.perf_counter()
   tokens=[1]+[42]*(a.prompt_tokens-1);payload=dict(model=a.model_name,prompt=tokens if b==1 else [tokens]*b,max_tokens=a.decode_tokens,ignore_eos=True,temperature=0,stream=True,stream_options={'include_usage':True,'continuous_usage_stats':True});begin=time.perf_counter();usage_by_choice={}
   async with session.post('http://127.0.0.1:8089/v1/completions',json=payload) as r:
    if r.status!=200:raise RuntimeError(await r.text())
    async for line in r.content:
     if line.startswith(b'data: ') and line.strip()!=b'data: [DONE]':
      event=json.loads(line[6:])
      if event.get('usage'):
       for choice in event.get('choices',[]):usage_by_choice[choice['index']]=event['usage']
   if set(usage_by_choice)!=set(range(b)) or any(x['prompt_tokens']!=a.prompt_tokens or x['completion_tokens']!=a.decode_tokens for x in usage_by_choice.values()):raise RuntimeError(f'Invalid per-choice token counts: {usage_by_choice}')
   return (time.perf_counter()-begin)*1000
  for b in a.batch_sizes:
   for _ in range(a.warmup_repetitions):await batch(b)
   await control('audit',label=f'warm_b{b}')
   before=[await batch(b) for _ in range(a.repetitions)]
   await control('audit',label=f'before_b{b}')
   async def instrumented_request(label):
    # Reuse a small event set only after this request and its GPU work finish.
    # Collection and disk writes happen after HTTP response completion.
    await control('start',event_capacity=a.decode_tokens+16)
    elapsed=await batch(b)
    await control('collect',label=label)
    return elapsed,json.loads((OUT/label/'components.json').read_text())
   priming=[]
   for request_index in range(a.instrumented_warmup):
    elapsed,_=await instrumented_request(f'priming_b{b}_request_{request_index}')
    priming.append(elapsed)
   measured=[];combined=None
   for request_index in range(a.repetitions):
    elapsed,record=await instrumented_request(f'b{b}_request_{request_index}')
    measured.append(elapsed)
    if combined is None:combined=record
    else:
     if combined['environment']!=record['environment']:raise RuntimeError('Worker environment changed within collection')
     for name in ['dispatch_intervals','aggregates','observations']:combined[name].extend(record[name])
     for destination,source in zip(combined['worker_rows'],record['worker_rows']):destination.extend(source)
   target=OUT/f'b{b}';target.mkdir(exist_ok=False)
   (target/'components.json').write_text(json.dumps(combined))
   after=[await batch(b) for _ in range(a.repetitions)]
   await control('audit',label=f'after_b{b}')
   audits.append(dict(batch_size=b,baseline_e2e_ms=before,instrumented_priming_e2e_ms=priming,instrumented_e2e_ms=measured,post_e2e_ms=after))
   Path(a.output).with_suffix('.audit.json').write_text(json.dumps(audits,indent=2))
   data=json.loads((OUT/f'b{b}/components.json').read_text());env=data['environment'];aliases={};nodes=[aliases.setdefault(x['node_id'],len(aliases)) for x in sorted(env,key=lambda x:x['rank'])]
   if len(aliases)>1 and a.network_transport=='local':raise RuntimeError('Cross-node profiles require a verified socket/ib transport descriptor')
   for phase in ['prefill','decode']:
    samples=[r for r in data['aggregates'] if r['phase']==phase]
    if not samples or {r['batch_size'] for r in samples}!={b}:raise RuntimeError('Uncontrolled HTTP batch or phase')
    protocol=all('pipeline_result_return' in r for r in samples)
    row=dict(model_name=a.model_name_for_vidur,batch_size=b,tensor_parallel_degree=a.tp,pipeline_parallel_degree=a.pp,phase=phase,prefill_tokens_per_request=a.prompt_tokens if phase=='prefill' else 0,profile_schema_version=8 if protocol else 6,profile_loop_mode='http_serving_spaced_diagnostic' if a.diagnostic_request_interval else 'http_serving',request_interval_seconds=a.diagnostic_request_interval,rank_node_map=json.dumps(nodes,separators=(',',':')),network_transport=a.network_transport,num_nodes=len(aliases),enforce_eager=a.enforce_eager,event_loop_impl='uvloop',repetitions=a.repetitions,num_steps=len(samples),prompt_tokens=a.prompt_tokens,decode_tokens=a.decode_tokens)
    names=['schedule','prepare_inputs_e2e','sampler_e2e','process_model_outputs','ray_comm_time','engine_bookkeeping','pp_handoff_e2e','model_execution_e2e','graph_input_staging_e2e']+[f'pp_handoff_boundary_{i}' for i in range(a.pp-1)]+[f'graph_input_staging_stage_{i}' for i in range(a.pp)]
    if protocol:
     names+=['pipeline_result_return','executor_pre_dispatch']+[f'pipeline_{kind}_stage_{i}' for kind in ['input_ready','input_work','input_dispatch','post_receive'] for i in range(a.pp)]
     keys={r['key'] for r in samples}
     for stage in range(a.pp):
      records=[r for rank_rows in data['worker_rows'] for r in rank_rows if r['rank']==stage*a.tp and r['key'] in keys]
      row[f'eager_dispatch_program_stage_{stage}']=json.dumps(mean_program([dispatch_program(r) for r in records]),separators=(',',':')) if phase=='prefill' else ''
    for k in names:row[k+'_mean']=statistics.mean(r[k] for r in samples);row[k+'_median']=statistics.median(r[k] for r in samples)
    row['step_e2e_mean']=statistics.mean(r['engine_step_ms'] for r in samples);rows.append(row)
   print('HTTP_PROFILE',json.dumps(rows[-2:]),flush=True)
 out=Path(a.output)
 with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 out.with_suffix('.audit.json').write_text(json.dumps(audits,indent=2))
 out.with_suffix('.metadata.json').write_text(json.dumps(dict(arguments=vars(a),measurement='CPU, protocol and graph staging primitives in native streaming HTTP serving; GPU DAG span independently profiled; no client E2E used as prediction target',controlled_active_virtual_engines=1,event_handles_initialized_before_measurement=True,event_pool_capacity=a.decode_tokens+16,event_handles_reused_only_after_request_completion=True,per_request_collection_outside_measurement=True),indent=2))
asyncio.run(main())
