"""PP-aware selection, phase lookup, and non-duplicated handoff regression tests."""
import ast
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def methods(names):
    path = ROOT / 'vidur/execution_time_predictor/sklearn_execution_time_predictor.py'
    tree = ast.parse(path.read_text())
    selected = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    env = dict(pd=pd, Dict=Dict, List=List, Any=Any, BaseEstimator=object,
               Batch=object, hashlib=hashlib, json=json)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), env)
    return env


class ProfileSelectionTests(unittest.TestCase):
    def predictor(self, frame, pp):
        obj = SimpleNamespace(
            _replica_config=SimpleNamespace(num_pipeline_stages=pp, tensor_parallel_size=1),
            _model_config=SimpleNamespace(get_name=lambda:'llama'),
            _read_input_file=lambda _:frame.copy())
        obj.load = lambda: methods(['_load_cpu_overhead_df'])['_load_cpu_overhead_df'](obj,'unused')
        return obj

    def rows(self):
        return pd.DataFrame([
            dict(model_name='llama',tensor_parallel_degree=1,pipeline_parallel_degree=pp,
                 phase=phase,prefill_tokens_per_request=512 if phase=='prefill' else 0,
                 pp_handoff_e2e_mean=value)
            for pp,value in [(1,0),(2,2),(4,4)] for phase in ['prefill','decode']])

    def test_pp_selects_only_its_own_samples(self):
        obj=self.predictor(self.rows(),2)
        selected=obj.load()
        self.assertEqual(set(selected.pipeline_parallel_degree),{2})
        self.assertTrue(obj._has_pp_handoff_profile)
        self.assertEqual(obj._cpu_profile_feature_cols,
                         ['batch_size','prefill_tokens_per_request'])

    def test_legacy_rows_are_pp1_only(self):
        frame=pd.DataFrame([dict(model_name='llama',tensor_parallel_degree=1,batch_size=1)])
        self.assertEqual(len(self.predictor(frame,1).load()),1)
        with self.assertRaisesRegex(ValueError,'legacy'):
            self.predictor(frame,2).load()

    def test_missing_pp_does_not_fall_back(self):
        with self.assertRaisesRegex(ValueError,'No CPU profile'):
            self.predictor(self.rows(),3).load()

    def test_incomplete_phase_table_rejected(self):
        with self.assertRaisesRegex(ValueError,'both prefill and decode'):
            self.predictor(self.rows().query("phase == 'decode'"),2).load()

    def test_same_batch_prefill_and_decode_use_different_keys(self):
        fn=methods(['_get_cpu_profile_prediction'])['_get_cpu_profile_prediction']
        obj=SimpleNamespace(_cpu_profile_feature_cols=['batch_size','prefill_tokens_per_request'],
                            _predictions={'schedule':{(1,0):1,(1,512):9}})
        self.assertEqual(fn(obj,'schedule',SimpleNamespace(size=1,num_prefill_tokens=0)),1)
        self.assertEqual(fn(obj,'schedule',SimpleNamespace(size=1,num_prefill_tokens=512)),9)

    def test_full_handoff_replaces_bare_send_recv_and_is_added_once(self):
        fn=methods(['_get_pipeline_parallel_communication_time'])['_get_pipeline_parallel_communication_time']
        obj=SimpleNamespace(_has_pp_handoff_profile=True,
                            _replica_config=SimpleNamespace(num_pipeline_stages=4),
                            _get_cpu_profile_prediction=lambda *_:6,
                            _predictions={'send_recv':{(8,):999}})
        batch=SimpleNamespace(_total_num_tokens_rounded=8)
        self.assertEqual(fn(obj,batch)*3,6)

    def test_cpu_cache_separates_pp_but_gpu_cache_is_stable(self):
        fn=methods(['_get_model_hash'])['_get_model_hash']
        def obj(pp):return SimpleNamespace(to_dict=lambda:{'same':'config'},
                            _replica_config=SimpleNamespace(num_pipeline_stages=pp))
        self.assertNotEqual(fn(obj(1),'schedule'),fn(obj(2),'schedule'))
        self.assertEqual(fn(obj(1),'attn_pre_proj'),fn(obj(2),'attn_pre_proj'))


class CsvMigrationTests(unittest.TestCase):
    def test_append_preserves_legacy_rows_and_marks_them_pp1(self):
        path=ROOT/'vidur/profiling/cpu_overhead/vllm_051_cpu_overhead.py'
        tree=ast.parse(path.read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_write_rows')
        env=dict(Path=Path,List=List,Dict=Dict,Any=Any,csv=csv)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'cpu.csv'
            output.write_text('model_name,batch_size,tensor_parallel_degree\nllama,1,1\n')
            env['_write_rows'](output,[dict(model_name='llama',batch_size=1,
                tensor_parallel_degree=1,pipeline_parallel_degree=2,phase='decode')],True)
            with output.open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),2)
            self.assertEqual([r['pipeline_parallel_degree'] for r in rows],['1','2'])


if __name__=='__main__':unittest.main()
