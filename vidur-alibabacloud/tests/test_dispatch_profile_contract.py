"""Native task ordering and direct timestamp measurements, without GPU imports."""
import asyncio,importlib.util,unittest
from pathlib import Path
from types import SimpleNamespace as NS

path=Path(__file__).resolve().parents[1]/'vidur/profiling/cpu_overhead/dispatch_profile_contract.py'
spec=importlib.util.spec_from_file_location('dispatch_contract',path)
contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)

class DispatchTests(unittest.TestCase):
    def test_stages_execute_concurrently_and_return_last_stage_result(self):
        async def scenario():
            entered=[];all_entered=asyncio.Event()
            async def task(stage,method,request):
                self.assertEqual(method,'execute_model');self.assertEqual(request,'request')
                entered.append(stage)
                if len(entered)==3:all_entered.set()
                await all_entered.wait()
                return f'stage-{stage}'
            executor=NS(pp_locks=None,parallel_config=NS(pipeline_parallel_size=3),
                driver_exec_method=lambda *a:task(0,*a),tp_driver_workers=[
                    NS(execute_method=NS(remote=lambda *a,s=s:task(s,*a))) for s in [1,2]])
            row={};result=await asyncio.wait_for(contract.trace_driver_tasks(executor,'request',row),1)
            self.assertEqual(result,'stage-2');self.assertEqual(set(entered),{0,1,2})
            self.assertEqual({t['stage'] for t in row['executor_tasks']},{0,1,2})
        asyncio.run(scenario())

    def test_stage_locks_preserve_exclusion_across_virtual_engines(self):
        async def scenario():
            active=[0,0];calls=[]
            async def task(stage,method,request):
                active[stage]+=1;self.assertEqual(active[stage],1)
                calls.append((stage,request));await asyncio.sleep(0);active[stage]-=1
                return request
            executor=NS(pp_locks=None,parallel_config=NS(pipeline_parallel_size=2),
                driver_exec_method=lambda *a:task(0,*a),
                tp_driver_workers=[NS(execute_method=NS(remote=lambda *a:task(1,*a)))])
            self.assertEqual(await asyncio.gather(
                contract.trace_driver_tasks(executor,'a',{}),
                contract.trace_driver_tasks(executor,'b',{})),['a','b'])
            self.assertEqual(len(calls),4);self.assertEqual(active,[0,0])
        asyncio.run(scenario())

    def fixture(self):
        stage=dict(stage=0,task_started_ns=100,lock_acquired_ns=200,
            call_begin_ns=300,call_return_ns=350,result_return_ns=900)
        worker=dict(rank=0,key='key',worker_start_ns=500,worker_end_ns=800,worker_timing_scope='worker')
        step=dict(key='key',driver_tasks_begin_ns=0,driver_join_complete_ns=1000,executor_tasks=[stage])
        return step,[worker],[dict(monotonic_clock_domain='same-host')]

    def test_named_endpoints_measure_ingress_return_and_join_separately(self):
        step,workers,environment=self.fixture()
        result=contract.task_dispatch_intervals(step,workers,1,1,environment)[0]
        self.assertAlmostEqual(result['worker_ingress_ms'],.0002)
        self.assertAlmostEqual(result['worker_return_ms'],.0001)
        self.assertAlmostEqual(result['join_wait_ms'],.0001)
        self.assertAlmostEqual(result['submission_host_ms'],.00005)

    def test_cross_host_clocks_and_invalid_order_are_rejected(self):
        step,workers,environment=self.fixture()
        workers[0]['worker_start_ns']=299
        with self.assertRaisesRegex(ValueError,'ordering'):
            contract.task_dispatch_intervals(step,workers,1,1,environment)
        step,workers,environment=self.fixture();environment[0]['monotonic_clock_domain']=''
        with self.assertRaisesRegex(ValueError,'clock domain'):
            contract.task_dispatch_intervals(step,workers,1,1,environment)
        with self.assertRaisesRegex(ValueError,'clock domain'):
            contract.task_dispatch_intervals(step,workers,1,2,[dict(monotonic_clock_domain='a'),dict(monotonic_clock_domain='b')])

    def test_duplicate_task_records_are_rejected(self):
        step,workers,environment=self.fixture();step['executor_tasks']*=2
        with self.assertRaisesRegex(ValueError,'duplicate'):
            contract.task_dispatch_intervals(step,workers,1,1,environment)

if __name__=='__main__':unittest.main()
