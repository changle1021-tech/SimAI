"""Recompute v3 perimeter medians from raw client/worker timestamps."""
import argparse,collections,hashlib,json,statistics as st
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();r=args.root
requests=json.loads((r/'fresh_validation_20261018/requests.json').read_text())
cal={q['response_id']:q for q in requests if q['label']=='boundary' and q['input_tokens'] in (256,512,2048,8192)}
points=collections.defaultdict(list);front=[];pref_gaps=[];decode_gaps=[];sources={}
for file in (r/'timers').glob('fresh_p*/dp*.json'):
 grouped=collections.defaultdict(lambda:collections.defaultdict(list))
 for row in json.loads(file.read_text())['rows']:
  iteration=row.get('iteration')
  if not iteration or not iteration.get('request_tokens'):continue
  request=next(iter(iteration['request_tokens'])).rsplit('-',1)[0]
  if request in cal:grouped[request][row['kind']].append(row)
 if not grouped:continue
 sources[str(file.relative_to(r))]=hashlib.sha256(file.read_bytes()).hexdigest()
 for request,g in grouped.items():
  ex,sa,mo=[sorted(g[k],key=lambda x:x['cpu_start_ns']) for k in ('execute','sample','model')]
  assert len(ex)==len(sa)==len(mo)
  context=0;decode=[];pref=[]
  for e,s,m in zip(ex,sa,mo):
   n=e['iteration']['total_tokens'];assert n==s['iteration']['total_tokens']==m['iteration']['total_tokens']
   net=(s['cpu_end_ns']-e['cpu_start_ns'])/1e6-m['device_elapsed_ms'];assert net>=0
   if n==1:decode.append(net)
   else:
    points[('prefill_first' if context==0 else 'prefill_next',n)].append(net);context+=n;pref.append((e,s))
  points[('decode',1)].append(st.median(decode))
  q=cal[request]
  front.append((pref[0][0]['cpu_start_ns']-q['client_start_ns']+q['first_token_ns']-pref[-1][1]['cpu_end_ns'])/1e6)
  pref_gaps += [(e['cpu_start_ns']-s['cpu_end_ns'])/1e6 for (_,s),(e,_) in zip(pref,pref[1:])]
  decode_gaps.append(st.median((e['cpu_start_ns']-s['cpu_end_ns'])/1e6 for s,e in zip(sa,ex[1:]) if e['iteration']['total_tokens']==1))
profile=json.loads((r/'cpu_boundary_profile_v3.json').read_text());assert profile['source_sha256']==sources
for (phase,tokens),values in points.items():
 row=next(x for x in profile['phases'][phase] if x['tokens']==tokens)
 assert row['critical_perimeter_ms']==st.median(values),(phase,tokens)
 assert row['samples']==len(values)
for key,values in [('request_io_ms',front),('inter_chunk_gap_ms',pref_gaps),('inter_decode_gap_ms',decode_gaps)]:
 assert profile[key]==st.median(values),key
print('Verified all CPU perimeter medians and raw source hashes; no E2E residual used.')
