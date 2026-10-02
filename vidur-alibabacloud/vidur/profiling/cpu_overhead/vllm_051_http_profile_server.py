"""Diagnostic API serving path; collect named regions without per-step CUDA waits."""
import asyncio,atexit,contextvars,json,os,runpy,sys,time
from pathlib import Path
from collections import defaultdict
idx=sys.argv.index('--diagnostic-output');OUT=Path(sys.argv[idx+1]);del sys.argv[idx:idx+2]
OUT.mkdir(parents=True,exist_ok=False);(OUT/'server_pid.txt').write_text(str(os.getpid()))
from pp_profile_contract import request_key,aggregate_step,bind_worker_keys
from dispatch_profile_contract import trace_driver_tasks,task_dispatch_intervals,pipeline_protocol_primitives
from vllm_051_pp_cpu_profile import ProfilingRayExecutor
import vllm.executor.ray_gpu_executor as raymodule
from vllm.engine.async_llm_engine import _AsyncLLMEngine,AsyncLLMEngine
from vllm.core.scheduler import Scheduler
from fastapi import FastAPI
executors=[];steps=[];observations=[];last_step={};current=contextvars.ContextVar('api_component_step',default=None);enabled=False;idle=asyncio.Event()
original_stop=_AsyncLLMEngine.stop_remote_worker_execution_loop_async
async def mark_idle(self):
 await original_stop(self);idle.set()
_AsyncLLMEngine.stop_remote_worker_execution_loop_async=mark_idle
class DiagnosticExecutor(ProfilingRayExecutor):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);executors.append(self)
 async def _driver_execute_model_async(self,req=None):
  row=current.get()
  if enabled and row is not None and req and req.seq_group_metadata_list:
   return await trace_driver_tasks(self,req,row)
  return await super()._driver_execute_model_async(req)
 async def execute_model_async(self,req):
  if req and req.seq_group_metadata_list:idle.clear()
  begin=time.perf_counter_ns();result=await super().execute_model_async(req)
  row=current.get()
  if row is not None and req and req.seq_group_metadata_list:
   row.update(key=request_key(req),executor_begin_ns=begin,executor_end_ns=time.perf_counter_ns(),executor_ms=(time.perf_counter_ns()-begin)/1e6)
  return result
raymodule.RayGPUExecutorAsync=DiagnosticExecutor
original_schedule=Scheduler.schedule
original_process=_AsyncLLMEngine._process_model_outputs
original_step=_AsyncLLMEngine.step_async

def schedule(self,*a,**kw):
 begin=time.perf_counter_ns();result=original_schedule(self,*a,**kw);end=time.perf_counter_ns();row=current.get()
 if row is not None:
  metadata,scheduled=result
  if metadata:
   row.update(schedule_begin_ns=begin,schedule_end_ns=end,schedule_ms=(end-begin)/1e6,phase='prefill' if metadata[0].is_prompt else 'decode',
    batch_size=len(metadata),prefill_tokens_per_request=sum(m.token_chunk_size for m in metadata)/len(metadata) if metadata[0].is_prompt else 0,
    request_ids=[m.request_id for m in metadata])
 return result

def process(self,*a,**kw):
 begin=time.perf_counter_ns();result=original_process(self,*a,**kw);row=current.get()
 if row is not None:row.update(process_begin_ns=begin,process_end_ns=time.perf_counter_ns(),process_outputs_ms=(time.perf_counter_ns()-begin)/1e6)
 return result

async def step(self,ve):
 row=dict(virtual_engine=ve);token=current.set(row);start=time.perf_counter_ns()
 try:result=await original_step(self,ve)
 finally:current.reset(token)
 end=time.perf_counter_ns();row.update(start_ns=start,end_ns=end,engine_step_ms=(end-start)/1e6)
 if 'key' in row:
  ids=row['request_ids'];previous=last_step.get(ve)
  gap=(start-previous[0])/1e6 if previous and set(ids)&set(previous[1]) else 0
  last_step[ve]=(end,ids)
  observations.append(dict(phase=row['phase'],ms=row['engine_step_ms'],instrumented=enabled))
  if enabled:
   row['engine_step_ms']+=gap
   row['engine_bookkeeping_ms']=gap+sum((b-a)*1e-6 for a,b in [
       (start,row['schedule_begin_ns']),
       (row['schedule_end_ns'],row['executor_begin_ns']),
       (row['executor_end_ns'],row['process_begin_ns']),
       (row['process_end_ns'],end)])
   steps.append(row)
 return result
Scheduler.schedule=schedule;_AsyncLLMEngine._process_model_outputs=process;_AsyncLLMEngine.step_async=step

def persist(collect_workers=False):
 payload=dict(steps=steps,measurement='named regions in actual streaming API; no per-step CUDA synchronization')
 if collect_workers:
  ex=executors[-1];workers=ex._run_workers('pp_profile_get_records');environment=ex._run_workers('pp_profile_environment')
  (OUT/'worker_records.json').write_text(json.dumps(dict(workers=workers,environment=environment),indent=2))
  by_key=defaultdict(list)
  for rank_rows in workers:
   for r in rank_rows:by_key[r['key']].append(r)
  tp=ex.parallel_config.tensor_parallel_size;pp=ex.parallel_config.pipeline_parallel_size
  aggregates=[]
  for s in steps:
   timing=aggregate_step(s,by_key.pop(s['key']),tp,pp);aggregates.append(dict(s,**timing))
  if by_key:raise RuntimeError('Unmatched GPU records')
  payload.update(worker_rows=workers,environment=environment,aggregates=aggregates)
 (OUT/'components.json').write_text(json.dumps(payload,indent=2));return dict(steps=len(steps),collected_workers=collect_workers)
async def start_profile(event_capacity: int = 2048):
 global enabled
 if enabled:raise RuntimeError('Already profiling')
 await idle.wait()
 steps.clear();last_step.clear()
 ex=executors[-1]
 if event_capacity<1:raise ValueError('Invalid event capacity')
 if not getattr(ex,'_events_reserved',False):
  ex._run_workers('pp_profile_reserve_events',event_capacity);ex._events_reserved=event_capacity
 elif ex._events_reserved!=event_capacity:
  raise ValueError('Profile event capacity is immutable within one server')
 ex._run_workers('pp_profile_set_enabled',True);enabled=True
 return dict(profiling=True)
async def collect_profile(label: str):
 global enabled
 if not label.replace('_','').isalnum():raise ValueError('Invalid profile label')
 enabled=False
 await idle.wait()
 target=OUT/label;target.mkdir(exist_ok=False)
 ex=executors[-1];workers=ex._run_workers('pp_profile_get_records');environment=ex._run_workers('pp_profile_environment')
 by_key=defaultdict(list)
 for rank_rows in workers:
  for r in rank_rows:by_key[r['key']].append(r)
 tp=ex.parallel_config.tensor_parallel_size;pp=ex.parallel_config.pipeline_parallel_size
 bind_worker_keys(workers,tp,pp)
 by_key=defaultdict(list)
 for rank_rows in workers:
  for r in rank_rows:by_key[r['key']].append(r)
 aggregates=[]
 for row in steps:
  records=by_key.pop(row['key'])
  named_protocol=len({item['monotonic_clock_domain'] for item in environment})==1
  timing=aggregate_step(row,records,tp,pp,legacy_residual=not named_protocol)
  if named_protocol:
   timing.update(pipeline_protocol_primitives(row,records,tp,pp,environment))
   # The protocol DAG replaces the old executor residual in the predictor.
   timing['ray_comm_time']=0.0
  aggregates.append(dict(row,**timing))
 dispatch=[]
 if len({item['monotonic_clock_domain'] for item in environment})==1:
  indexed=defaultdict(list)
  for records in workers:
   for record in records:indexed[record['key']].append(record)
  dispatch=[dict(key=row['key'],phase=row['phase'],stages=task_dispatch_intervals(row,indexed[row['key']],tp,pp,environment)) for row in steps]
 if by_key:raise RuntimeError('Unmatched GPU records')
 (target/'components.json').write_text(json.dumps(dict(dispatch_intervals=dispatch,aggregates=aggregates,worker_rows=workers,environment=environment,observations=list(observations)),indent=2))
 ex._run_workers('pp_profile_set_enabled',False)
 steps.clear();observations.clear();last_step.clear()
 return dict(label=label,steps=len(aggregates))
async def audit_observations(label: str):
 if not label.replace('_','').isalnum():raise ValueError('Invalid audit label')
 (OUT/(label+'.json')).write_text(json.dumps(list(observations),indent=2));observations.clear()
 return dict(label=label)

original_add=_AsyncLLMEngine._add_processed_request
def add_request(self,*a,**kw):
 schedulers=self.scheduler
 try:
  self.scheduler=schedulers[:1]
  return original_add(self,*a,**kw)
 finally:self.scheduler=schedulers
_AsyncLLMEngine._add_processed_request=add_request
original_init=FastAPI.__init__
def app_init(self,*a,**kw):
 original_init(self,*a,**kw)
 self.add_api_route('/diagnostic/profile/start',start_profile,methods=['POST'])
 self.add_api_route('/diagnostic/profile/collect',collect_profile,methods=['POST'])
 self.add_api_route('/diagnostic/profile/audit',audit_observations,methods=['POST'])
FastAPI.__init__=app_init
atexit.register(lambda:(OUT/'exit_observations.json').write_text(json.dumps(observations)))
runpy.run_module('vllm.entrypoints.openai.api_server',run_name='__main__')
