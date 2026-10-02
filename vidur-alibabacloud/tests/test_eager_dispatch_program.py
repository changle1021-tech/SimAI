"""Validate CPU/GPU overlap and input dependencies using physical timelines."""
import ast
import importlib.util
import unittest
import time
from pathlib import Path
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parents[1]
def load(relative):
    path=ROOT/relative
    spec=importlib.util.spec_from_file_location(path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
program=load('vidur/utils/eager_dispatch_program.py')
trace=load('vidur/profiling/cpu_overhead/eager_dispatch_trace.py')

class EagerDispatchTests(unittest.TestCase):
    def frames(self,calls):
        return dict(frames=[dict(operation=str(i),layer=0,gap_ms=0,call_ms=x)
                            for i,x in enumerate(calls)],tail_ms=0)
    def test_host_submissions_overlap_previous_gpu_work(self):
        # CPU submissions finish at 1,2; GPU completes at 6,11.
        self.assertEqual(program.replay_program(self.frames([1,1]),lambda *_:5),11)
    def test_gpu_waits_when_cpu_producer_is_late(self):
        # CPU submissions finish at 1,11; GPU completes at 3,13.
        self.assertEqual(program.replay_program(self.frames([1,10]),lambda *_:2),13)
    def test_interpolation_is_bounded_and_preserves_operator_layout(self):
        points={(1,256):self.frames([1]),(1,1024):self.frames([4])}
        self.assertEqual(program.interpolate_program(points,1,512)['frames'][0]['call_ms'],2)
        with self.assertRaisesRegex(ValueError,'cover'):
            program.interpolate_program(points,2,512)
        with self.assertRaisesRegex(ValueError,'cover'):
            program.interpolate_program(points,1,2048)
        points[(1,1024)]=self.frames([2,3])
        with self.assertRaisesRegex(ValueError,'layouts'):
            program.interpolate_program(points,1,512)
    def test_leaf_trace_rejects_nested_or_overlapping_regions(self):
        row=dict(cpu_forward_start_ns=0,cpu_forward_end_ns=100,
                 cpu_dispatch_frames=[dict(operation='qkv',layer=0,start_ns=20,end_ns=50,thread_cpu_ms=0)])
        result=trace.dispatch_program(row)
        self.assertAlmostEqual(result['frames'][0]['gap_ms'],.00002)
        row['cpu_dispatch_frames'].append(dict(operation='rope',layer=0,start_ns=40,end_ns=70))
        with self.assertRaisesRegex(ValueError,'overlap'):
            trace.dispatch_program(row)
    def test_cached_rotary_module_is_hooked_once_with_the_actual_calling_layer(self):
        class Leaf:
            def forward(self):return None
            def __call__(self):return self.forward()
        shared_rope=Leaf()
        class Layer(Leaf):
            def __init__(self):
                self.input_layernorm=Leaf();self.post_attention_layernorm=Leaf()
                self.self_attn=NS(qkv_proj=Leaf(),rotary_emb=shared_rope,attn=Leaf(),o_proj=Leaf())
                self.mlp=NS(gate_up_proj=Leaf(),act_fn=Leaf(),down_proj=Leaf())
            def forward(self):
                for leaf in [self.input_layernorm,self.self_attn.qkv_proj,self.self_attn.rotary_emb,
                             self.self_attn.attn,self.self_attn.o_proj,self.post_attention_layernorm,
                             self.mlp.gate_up_proj,self.mlp.act_fn,self.mlp.down_proj]:leaf()
        body=NS(layers=[Layer(),Layer()],embed_tokens=Leaf(),norm=Leaf())
        row=dict(_capture_cpu_dispatch=True,cpu_forward_start_ns=time.perf_counter_ns())
        worker=NS(rank=0,parallel_config=NS(tensor_parallel_size=1,pipeline_parallel_size=1),_pp_profile_current=row)
        trace.install_llama_dispatch_trace(worker,NS(model=body))
        body.embed_tokens()
        for layer in body.layers:layer()
        body.norm();row['cpu_forward_end_ns']=time.perf_counter_ns()
        result=trace.dispatch_program(row)
        self.assertEqual(len(result['frames']),20)
        self.assertEqual([f['layer'] for f in result['frames'] if f['operation']=='rope'],[0,1])

class PipelineInputTests(unittest.TestCase):
    def method(self):
        path=ROOT/'vidur/execution_time_predictor/sklearn_execution_time_predictor.py'
        node=next(n for n in ast.walk(ast.parse(path.read_text()))
                  if isinstance(n,ast.FunctionDef) and n.name=='_get_pipeline_protocol_time')
        env=dict(Batch=object)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
        return env[node.name]
    def run_case(self,remote_input_work,available=0):
        values=dict(executor_pre_dispatch=.2,pipeline_input_dispatch_stage_0=.3,
                    pipeline_input_dispatch_stage_1=.5,pipeline_input_work_stage_0=1,
                    pipeline_input_work_stage_1=remote_input_work,
                    pipeline_post_receive_stage_1=.1,pipeline_result_return=.4)
        predictor=NS(_config=NS(skip_cpu_overhead_modeling=False),_pipeline_protocol_model_names=['present'],
                     _replica_config=NS(num_pipeline_stages=2),_get_schedule_time=lambda _:0,
                     _get_engine_bookkeeping_time=lambda _:0,_get_cpu_profile_prediction=lambda name,_:values[name],
                     _pipeline_worker_available_at=[0,available])
        batch=NS();fn=self.method()
        first=fn(predictor,batch,0,0)
        last=fn(predictor,batch,1,.01)
        return first,last
    def test_remote_input_preparation_overlaps_first_stage_compute(self):
        first,last=self.run_case(1)
        self.assertAlmostEqual(first,1.5)
        self.assertAlmostEqual(last,.5)
    def test_slow_remote_input_is_a_real_dependency_stall(self):
        first,last=self.run_case(12)
        self.assertAlmostEqual(first,1.5)
        self.assertAlmostEqual(last,3.2)
    def test_previous_stage_worker_lock_delays_input_preparation(self):
        _,last=self.run_case(1,available=.02)
        self.assertAlmostEqual(last,11.5)

if __name__=='__main__':unittest.main()
