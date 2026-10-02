"""Task dispatch intervals in the native vLLM 0.5.1 Ray executor.

These timestamps describe named transitions, not executor/E2E residuals.
No CUDA synchronization or additional Ray payload is introduced.
"""
import asyncio
import time


async def trace_driver_tasks(executor, request, row):
    """Preserve native per-stage locks, task concurrency and final-stage output."""
    if executor.pp_locks is None:
        executor.pp_locks=[asyncio.Lock() for _ in range(executor.parallel_config.pipeline_parallel_size)]
    row['driver_tasks_begin_ns']=time.perf_counter_ns()
    traces=[]

    async def run_task(task,stage):
        trace=dict(stage=stage,task_started_ns=time.perf_counter_ns())
        traces.append(trace)
        async with executor.pp_locks[stage]:
            trace['lock_acquired_ns']=time.perf_counter_ns()
            trace['call_begin_ns']=time.perf_counter_ns()
            operation=task('execute_model',request)
            trace['call_return_ns']=time.perf_counter_ns()
            result=await operation
            trace['result_return_ns']=time.perf_counter_ns()
        trace['task_finished_ns']=time.perf_counter_ns()
        return result

    tasks=[asyncio.create_task(run_task(executor.driver_exec_method,0))]
    for stage,worker in enumerate(executor.tp_driver_workers,start=1):
        tasks.append(asyncio.create_task(run_task(worker.execute_method.remote,stage)))
    row['driver_tasks_created_ns']=time.perf_counter_ns()
    results=await asyncio.gather(*tasks)
    row.update(driver_join_complete_ns=time.perf_counter_ns(),executor_tasks=traces)
    return results[-1]


def task_dispatch_intervals(step,workers,tp,pp,environment):
    """Read observable dispatch/return intervals on a shared monotonic clock.

    Stage-to-stage GPU and CPU overlap remains separate. These intervals must
    not be summed across stages or mislabeled as serial critical-path costs.
    """
    if (len(environment)!=tp*pp or
            any(not item.get('monotonic_clock_domain') for item in environment) or
            len({item['monotonic_clock_domain'] for item in environment})!=1):
        raise ValueError('Dispatch intervals need a shared monotonic clock domain')
    by_rank={item['rank']:item for item in workers}
    tasks={item['stage']:item for item in step['executor_tasks']}
    if len(workers)!=tp*pp or set(by_rank)!=set(range(tp*pp)):
        raise ValueError('Missing or duplicate worker rank timestamps')
    if len(step['executor_tasks'])!=pp or set(tasks)!=set(range(pp)):
        raise ValueError('Missing or duplicate pipeline driver task timestamps')
    result=[]
    for stage in range(pp):
        worker=by_rank[stage*tp];task=tasks[stage]
        if worker['worker_timing_scope']!='worker':
            raise ValueError('Pipeline driver requires complete Worker timing scope')
        if worker['key']!=step['key']:
            raise ValueError('Dispatch/worker request keys differ')
        endpoints=[step['driver_tasks_begin_ns'],task['task_started_ns'],
                   task['lock_acquired_ns'],task['call_begin_ns'],
                   worker['worker_start_ns'],worker['worker_end_ns'],
                   task['result_return_ns'],step['driver_join_complete_ns']]
        if any(a>b for a,b in zip(endpoints,endpoints[1:])):
            raise ValueError('Dispatch timestamps contradict native task ordering')
        def elapsed(a,b):return (b-a)*1e-6
        result.append(dict(stage=stage,
            task_schedule_ms=elapsed(step['driver_tasks_begin_ns'],task['task_started_ns']),
            lock_wait_ms=elapsed(task['task_started_ns'],task['lock_acquired_ns']),
            worker_ingress_ms=elapsed(task['call_begin_ns'],worker['worker_start_ns']),
            worker_return_ms=elapsed(worker['worker_end_ns'],task['result_return_ns']),
            join_wait_ms=elapsed(task['result_return_ns'],step['driver_join_complete_ns']),
            submission_host_ms=elapsed(task['call_begin_ns'],task['call_return_ns'])))
    return result


def pipeline_protocol_primitives(step, workers, tp, pp, environment):
    """Named pipeline input/return paths; GPU-dependent receive waits excluded.

    All stage tasks start concurrently. Their metadata preparation is a
    parallel predecessor of activation reception, not serial per-stage cost.
    """
    intervals=task_dispatch_intervals(step,workers,tp,pp,environment)
    by_rank={w['rank']:w for w in workers}
    result={}
    for stage in range(pp):
        worker=by_rank[stage*tp]
        regions=worker['host_regions']
        forwards=[r for r in regions if r['name'] in ('forward_eager','forward_graph')]
        if len(forwards)!=1:
            raise ValueError('Expected one native forward per worker step')
        forward=forwards[0]
        if stage:
            receive=[r for r in regions if r['name']=='recv_tensor_dict']
            if len(receive)!=1:
                raise ValueError('Missing pipeline receive region')
            ready=receive[0]['start_ns']
            post_receive=(forward['start_ns']-receive[0]['end_ns'])*1e-6
        else:
            ready=forward['start_ns']
            post_receive=0.0
        prefix=(ready-worker['worker_start_ns'])*1e-6
        if min(prefix,post_receive)<0:
            raise ValueError('Worker input timestamps contradict protocol order')
        interval=intervals[stage]
        result[f'pipeline_input_work_stage_{stage}']=prefix
        result[f'pipeline_input_dispatch_stage_{stage}']=(
            interval['task_schedule_ms']+interval['worker_ingress_ms'])
        result[f'pipeline_input_ready_stage_{stage}']=(
            interval['task_schedule_ms']+interval['worker_ingress_ms']+prefix)
        result[f'pipeline_post_receive_stage_{stage}']=post_receive
    last=by_rank[(pp-1)*tp]
    samples=[r for r in last['host_regions'] if r['name']=='sample']
    if len(samples)!=1:
        raise ValueError('Missing native final-stage sampler region')
    exit_ms=(last['worker_end_ns']-samples[0]['end_ns'])*1e-6
    if exit_ms<0:
        raise ValueError('Result exit precedes sampling')
    result['pipeline_result_return']=(exit_ms+intervals[-1]['worker_return_ms']+
                                       intervals[-1]['join_wait_ms'])
    result['executor_pre_dispatch']=(step['driver_tasks_begin_ns']-step['executor_begin_ns'])*1e-6
    if result['executor_pre_dispatch']<0:
        raise ValueError('Task dispatch precedes executor')
    return result
