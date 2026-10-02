import ast
import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
import sys
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace as NS
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'vidur/execution_time_predictor/sklearn_execution_time_predictor.py'

def method(name):
    tree=ast.parse(P.read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    env=dict(pd=pd,json=json,hashlib=hashlib,Dict=dict,List=list,Any=object,BaseEstimator=object,Batch=object,logger=NS(info=lambda *_:None))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(P),'exec'),env)
    return env[name]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
layout=load('layout',ROOT/'vidur/utils/runtime_layout.py')
geometry=load('geometry',ROOT/'vidur/profiling/cpu_overhead/native_attention_geometry.py')
contract=load('contract',ROOT/'vidur/profiling/cpu_overhead/pp_profile_contract.py')
collective=load('collective',ROOT/'vidur/utils/native_collective_profile.py')
graph=load('vidur.utils.decoder_graph_profile',ROOT/'vidur/utils/decoder_graph_profile.py')

class RuntimeGeometryTests(unittest.TestCase):
    def test_graph_padding_matches_special_small_batches(self):
        self.assertEqual([layout.graph_batch_size(b) for b in [1,2,3,4,5,8,9,16,17]],[1,2,4,4,8,8,16,16,24])
    def test_gqa_and_mqa_keep_replicated_kv_heads(self):
        self.assertEqual(geometry.local_attention_heads(32,8,2),(16,4))
        self.assertEqual(geometry.local_attention_heads(32,1,4),(8,1))
        self.assertEqual(geometry.local_attention_heads(32,8,16),(2,1))
        with self.assertRaises(ValueError):geometry.local_attention_heads(32,3,8)
    def test_uneven_pipeline_preserves_every_layer(self):
        bounds=[layout.pipeline_layer_bounds(32,i,3) for i in range(3)]
        self.assertEqual(bounds,[(0,10),(10,20),(20,32)])
        self.assertEqual(sum(b-a for a,b in bounds),32)
    def test_parameter_budget_uses_largest_dense_stage_and_explicit_stage(self):
        source=ROOT/'vidur/utils/param_counter.py'
        tree=ast.parse(source.read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='get_num_parameters_per_device')
        env={'pipeline_layer_bounds':layout.pipeline_layer_bounds}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),env)
        obj=NS(_replica_config=NS(model_name='dense',num_pipeline_stages=3),
               _model_config=NS(num_layers=32),get_num_parameters_per_layer=lambda:100)
        self.assertEqual(env['get_num_parameters_per_device'](obj),1200)
        obj._pipeline_stage_id=0
        self.assertEqual(env['get_num_parameters_per_device'](obj),1000)
    def test_rank_placement_keeps_cross_node_tp_and_pp_distinct(self):
        self.assertEqual(layout.configured_rank_nodes(4,8,'[7,7,9,9]'),[0,0,1,1])
        self.assertEqual(layout.configured_rank_nodes(4,8,'[7,9,7,9]'),[0,1,0,1])
        self.assertEqual(layout.configured_rank_nodes(16,8),[0]*8+[1]*8)
        with self.assertRaises(ValueError):layout.configured_rank_nodes(2,8,'[0]')
        with self.assertRaises(ValueError):layout.configured_rank_nodes(3,2,'[0,0,0]')
    def test_rope_uses_measurement_and_non_rope_model_does_not(self):
        fn=method('_get_attention_rope_execution_time')
        obj=NS(_config=NS(backend='vidur'),_model_config=NS(rope_theta=10000),_predictions={'attn_rope':{(1,):.007}},_get_compute_tokens=lambda _:1)
        self.assertEqual(fn(obj,object()),.007)
        obj._model_config.rope_theta=None;self.assertEqual(fn(obj,object()),0)
    def test_endpoint_operations_count_once_for_pp1_pp2_pp4(self):
        fn=method('_get_model_boundary_time')
        for pp in [1,2,4]:
            obj=NS(_config=NS(backend='vidur'),_replica_config=NS(num_pipeline_stages=pp),_get_compute_tokens=lambda _:1,
                   _predictions={n:{(1,):v} for n,v in [('emb',1),('first_layernorm',2),('input_layernorm',.5),('final_layernorm',3)]})
            self.assertEqual(sum(fn(obj,object(),s) for s in range(pp)),5.5)
    def test_graph_staging_excluded_from_ray_and_gpu_compute(self):
        w=[dict(key='k',rank=i,forward_gpu_ms=f,graph_input_staging_ms=g,prepare_ms=p,sampler_ms=a,send_ms=s)
           for i,f,g,p,a,s in [(0,20,3,5,0,7),(1,30,6,8,11,0)]]
        step=dict(key='k',executor_ms=100,schedule_ms=3,process_outputs_ms=4,engine_bookkeeping_ms=2)
        r=contract.aggregate_step(step,w,1,2)
        self.assertEqual(r['model_execution_e2e'],50);self.assertEqual(r['graph_input_staging_e2e'],9);self.assertEqual(r['ray_comm_time'],18)
        total=sum(r[k] for k in ['model_execution_e2e','graph_input_staging_e2e','pp_handoff_e2e',*contract.CPU_METRICS])
        self.assertEqual(total,109)
    def test_heterogeneous_pp_boundaries_are_not_averaged(self):
        fn=method('_get_pipeline_parallel_communication_time')
        values={'pp_handoff_boundary_0':.02,'pp_handoff_boundary_1':.5}
        obj=NS(_has_pp_handoff_profile=True,_pp_handoff_model_names=list(values),_get_cpu_profile_prediction=lambda n,_:values[n])
        self.assertEqual(fn(obj,object(),0),.02);self.assertEqual(fn(obj,object(),1),.5)

class FollowerTimingTests(unittest.TestCase):
    def test_tp_broadcast_followers_match_controlled_driver_sequence(self):
        rows=[[dict(rank=r,step_index=i,key=f'k{i}' if r%2==0 else None)
               for i in range(3)] for r in range(4)]
        contract.bind_worker_keys(rows,2,2)
        self.assertTrue(all([x['key'] for x in group]==['k0','k1','k2'] for group in rows))
        rows[1].pop()
        with self.assertRaisesRegex(ValueError,'step counts'):contract.bind_worker_keys(rows,2,2)

class ProfileIsolationTests(unittest.TestCase):
    def cpu_rows(self):
        return pd.DataFrame([dict(model_name='x',batch_size=1,tensor_parallel_degree=1,pipeline_parallel_degree=2,
            rank_node_map=placement,network_transport='socket' if placement=='[0,1]' else 'local',phase=phase,prefill_tokens_per_request=512 if phase=='prefill' else 0,
            profile_schema_version=schema,pp_handoff_e2e_mean=1,engine_bookkeeping_mean=value)
            for placement,value,schema in [('[0,0]',5,2),('[0,0]',7,4),('[0,1]',99,4)] for phase in ['prefill','decode']])
    def cpu_obj(self,frame,nodes):
        return NS(_config=NS(network_transport='socket',execution_loop_mode='direct',decode_attention_execution_mode='cuda_graph'),_rank_node_map=nodes,_replica_config=NS(num_pipeline_stages=2,tensor_parallel_size=1),_model_config=NS(get_name=lambda:'x'),_read_input_file=lambda _:frame.copy())
    def test_cpu_timing_schema_and_placement_are_not_mixed(self):
        fn=method('_load_cpu_overhead_df');obj=self.cpu_obj(self.cpu_rows(),[0,0]);out=fn(obj,'unused')
        self.assertEqual(set(out.engine_bookkeeping_mean),{7});self.assertEqual(len(out),2)
        obj=self.cpu_obj(self.cpu_rows(),[0,1]);self.assertEqual(set(fn(obj,'unused').engine_bookkeeping_mean),{99})
    def test_all_reduce_uses_each_tp_group_placement(self):
        fn=method('_load_all_reduce_df')
        frame=pd.DataFrame([dict(num_workers=2,devices_per_node=d,collective='all_reduce',value=v) for d,v in [(2,1),(1,8)]])
        obj=NS(_replica_config=NS(tensor_parallel_size=2),_rank_node_map=[0,0,0,1],_read_input_file=lambda _:frame.copy())
        self.assertEqual(fn(obj,'x',0).value.tolist(),[1])
        obj._config=NS(network_transport='socket')
        with self.assertRaisesRegex(ValueError,'verified network_transport'):fn(obj,'x',1)
        obj._read_input_file=lambda _:frame.assign(rank_node_map=['[0,0]','[0,1]'],network_transport=['local','socket'])
        self.assertEqual(fn(obj,'x',1).value.tolist(),[8])
        obj._read_input_file=lambda _:frame.copy()
        obj._replica_config.tensor_parallel_size=3;obj._rank_node_map=[0,0,1]
        with self.assertRaisesRegex(ValueError,'Uneven'):fn(obj,'x',0)
    def test_native_collective_timings_are_not_mixed_with_legacy(self):
        fn=method('_load_all_reduce_df')
        frame=pd.DataFrame([dict(num_workers=2,devices_per_node=2,collective='all_reduce',profile_kind=k)
                            for k in ['native_collective_v1',None]])
        obj=NS(_replica_config=NS(tensor_parallel_size=2),_rank_node_map=[0,0],_read_input_file=lambda _:frame)
        with self.assertRaisesRegex(ValueError,'legacy timing semantics'):fn(obj,'x')
    def test_cpu_profile_file_changes_do_not_invalidate_gpu_cost_models(self):
        fn=method('_get_model_hash')
        def obj(path):return NS(_rank_node_map=[0,0],_replica_config=NS(num_pipeline_stages=2),to_dict=lambda:{'cpu_overhead_input_file':path,'compute_input_file':'same'})
        self.assertEqual(fn(obj('one'),'attn_pre_proj'),fn(obj('two'),'attn_pre_proj'))
        self.assertNotEqual(fn(obj('one'),'schedule'),fn(obj('two'),'schedule'))
    def test_exact_native_cpu_observations_are_not_regressed_across_phases(self):
        fn=method('_get_cpu_profile_prediction')
        obj=NS(_cpu_profile_feature_cols=['batch_size','prefill_tokens_per_request'],_cpu_observed_points={'schedule':{(1,0):1,(1,512):100}})
        self.assertEqual(fn(obj,'schedule',NS(size=1,num_prefill_tokens=0)),1)
        self.assertEqual(fn(obj,'schedule',NS(size=1,num_prefill_tokens=512)),100)
    def test_serving_profile_is_not_mixed_with_inner_step_data(self):
        fn=method('_load_cpu_overhead_df');frame=self.cpu_rows();frame['profile_loop_mode']='direct'
        added=frame.copy();added['profile_loop_mode']='serving';added['engine_bookkeeping_mean']=123
        obj=self.cpu_obj(pd.concat([frame,added]),[0,0]);obj._config=NS(execution_loop_mode='serving',network_transport='socket')
        self.assertEqual(set(fn(obj,'x').engine_bookkeeping_mean),{123})
    def test_cpu_regressions_do_not_mix_prefill_with_decode(self):
        fn=method('_train_cpu_overhead_models')
        frame=pd.DataFrame([dict(batch_size=b,phase=p,prefill_tokens_per_request=512 if p=='prefill' else 0) for b in [1,2] for p in ['prefill','decode']])
        seen={}
        def train(model_name,df,**kw):seen[model_name]=set(df.phase);return object()
        obj=NS(_config=NS(skip_cpu_overhead_modeling=False),_get_cpu_overhead_df_with_derived_features=lambda x:x,
            _load_cpu_overhead_df=lambda _:frame,_cpu_overhead_input_file='x',_has_pp_handoff_profile=False,
            _has_engine_bookkeeping_profile=False,_graph_staging_model_names=[],_cpu_profile_feature_cols=['batch_size','prefill_tokens_per_request'],_train_model=train)
        fn(obj)
        self.assertEqual(seen['schedule_phase_prefill'],{'prefill'})
        self.assertEqual(seen['schedule_phase_decode'],{'decode'})
    def test_cross_node_rejects_table_with_unknown_placement(self):
        fn=method('_load_cpu_overhead_df');obj=self.cpu_obj(self.cpu_rows().drop(columns=['rank_node_map']),[0,1])
        with self.assertRaisesRegex(ValueError,'Cross-node'):fn(obj,'unused')
    def test_gpu_cache_does_not_depend_on_cpu_rank_placement(self):
        fn=method('_get_model_hash')
        def obj(nodes):return NS(_config=NS(network_transport='socket',execution_loop_mode='direct',decode_attention_execution_mode='cuda_graph'),_rank_node_map=nodes,_replica_config=NS(num_pipeline_stages=2),to_dict=lambda:{'same':'cfg'})
        self.assertNotEqual(fn(obj([0,0]),'ray_comm_time'),fn(obj([0,1]),'ray_comm_time'))
        self.assertEqual(fn(obj([0,0]),'attn_pre_proj'),fn(obj([0,1]),'attn_pre_proj'))
    def test_attention_capacity_matches_runtime_not_active_context(self):
        fn=method('_load_attention_df')
        common=dict(n_embd=4096,n_q_head=32,n_kv_head=32,block_size=16,num_tensor_parallel_workers=1,batch_size=1,
                    decode_layout='cuda_graph',physical_batch_size=1)
        rows=[dict(common,is_prefill=False,decode_block_table_capacity=c,kv_cache_size=512) for c in [528,4096]]
        rows.append(dict(common,is_prefill=True,decode_block_table_capacity=0,kv_cache_size=0))
        obj=NS(_block_size=16,_replica_config=NS(tensor_parallel_size=1),_config=NS(decode_attention_execution_mode='cuda_graph',cuda_graph_max_seq_len=4096,prediction_max_tokens_per_request=4096),
               _model_config=NS(embedding_dim=4096,num_q_heads=32,num_kv_heads=32,max_position_embeddings=4096))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'a.csv';pd.DataFrame(rows).to_csv(path,index=False);out=fn(obj,str(path))
            self.assertEqual(list(out[~out.is_prefill].decode_block_table_capacity),[4096])
            obj._config.cuda_graph_max_seq_len=1024
            with self.assertRaisesRegex(ValueError,'No attention profile'):fn(obj,str(path))

class NativeCollectiveTests(unittest.TestCase):
    def test_collective_mode_and_message_coverage_are_required(self):
        rows=[dict(dtype='float16',execution_mode=mode,size=size,**{'time_stats.all_reduce.mean':value})
              for mode,offset in [('eager',10),('cuda_graph',0)] for size,value in [(8192,1+offset),(32768,4+offset)]]
        profile=collective.NativeCollectiveProfile(rows,4096)
        self.assertEqual(profile.predict(2,False),2)
        self.assertEqual(profile.predict(2,True),12)
        with self.assertRaisesRegex(ValueError,'coverage'):profile.predict(5,False)
        with self.assertRaisesRegex(ValueError,'execution modes'):collective.NativeCollectiveProfile(rows[:2],4096)
        qualified=[dict(r,device_name='NVIDIA H100 80GB HBM3') for r in rows]
        self.assertEqual(collective.NativeCollectiveProfile(qualified,4096,device='h100').predict(2,False),2)
        with self.assertRaises(ValueError):collective.NativeCollectiveProfile(qualified,4096,device='a100')

class DecoderGraphTests(unittest.TestCase):
    def rows(self):
        return [dict(model_name='x',num_tensor_parallel_workers=1,num_layers=16,batch_size=1,physical_batch_size=1,
            kv_cache_size=kv,decode_block_table_capacity=4096,block_size=16,execution_mode='cuda_graph',phase='decode',dtype='float16',
            n_embd=4096,n_q_head=32,n_kv_head=32,n_expanded_embd=11008,**{'time_stats.decoder_graph.median':v}) for kv,v in [(512,3),(576,4)]]
    def test_gpu_dag_interpolation_rejects_unprofiled_layers_and_layouts(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'graph.csv';pd.DataFrame(self.rows()).to_csv(path,index=False)
            shape=dict(n_embd=4096,n_q_head=32,n_kv_head=32,n_expanded_embd=11008)
            with patch.dict(sys.modules, {'vidur.utils.runtime_layout':layout}):
                table=graph.DecoderGraphProfile(path,'x',shape,1,4096,16)
            self.assertEqual(table.predict(16,1,544),3.5)
            with self.assertRaisesRegex(ValueError,'coverage'):table.predict(12,1,544)
            with self.assertRaisesRegex(ValueError,'coverage'):table.predict(16,3,544)
            with self.assertRaisesRegex(ValueError,'matches'):graph.DecoderGraphProfile(path,'x',shape,1,2048,16)
    def test_combined_tp_graph_rejects_mixed_scope_and_unknown_placement(self):
        rows=[dict(row,includes_tp_collectives='true',rank_node_map='[0]',network_transport='local',tp_collective_backend='nccl') for row in self.rows()]
        shape=dict(n_embd=4096,n_q_head=32,n_kv_head=32,n_expanded_embd=11008)
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {'vidur.utils.runtime_layout':layout}):
            path=Path(folder)/'graph.csv';pd.DataFrame(rows).to_csv(path,index=False)
            table=graph.DecoderGraphProfile(path,'x',shape,1,4096,16)
            self.assertTrue(table.includes_tp_collectives)
            rows[0]['includes_tp_collectives']='false';pd.DataFrame(rows).to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'cannot mix'):graph.DecoderGraphProfile(path,'x',shape,1,4096,16)
            rows[0]['includes_tp_collectives']='true';rows[0]['rank_node_map']='[0,1]';pd.DataFrame(rows).to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'placement'):graph.DecoderGraphProfile(path,'x',shape,1,4096,16)
    def test_combined_tp_graph_charges_communication_once_only_in_decode(self):
        fn=method('_get_tensor_parallel_communication_time')
        obj=NS(_decoder_graph_profile=NS(includes_tp_collectives=True),
               _replica_config=NS(tensor_parallel_size=2),_rank_node_map=[0,0,0,0],
               _native_tp_collectives={0:NS(predict=lambda *_:.024)},
               _config=NS(decode_attention_execution_mode='cuda_graph'))
        self.assertEqual(fn(obj,NS(num_prefill_tokens=0),0),0)
        obj._get_compute_tokens=lambda _:512
        self.assertEqual(fn(obj,NS(num_prefill_tokens=0),0,include_boundary=True),.024)
        self.assertEqual(fn(obj,NS(num_prefill_tokens=512),0),.024)
        obj._rank_node_map=[0,1,0,0]
        with self.assertRaisesRegex(ValueError,'placement'):fn(obj,NS(num_prefill_tokens=0),0)
    def test_gpu_dag_replaces_operator_sum_and_keeps_communication(self):
        path=ROOT/'vidur/entities/execution_time.py';tree=ast.parse(path.read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='model_time');node.decorator_list=[]
        env={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
        obj=NS(_decoder_graph_execution_time=3,_tensor_parallel_communication_time=.1,_num_layers_per_pipeline_stage=16,
            pipeline_parallel_communication_time=.2,_model_boundary_time=.3)
        self.assertAlmostEqual(env['model_time'](obj),.0067)

if __name__=='__main__':unittest.main()
