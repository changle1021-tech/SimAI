"""Tests for send/recv collective profiling defaults and training targets.

These tests extract the relevant source definitions so importing optional GPU
profiling and simulation dependencies is not required.
"""

from __future__ import annotations

import ast
import enum
import hashlib
import importlib.util
import pickle
import statistics
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "vidur" / "profiling" / "collectives"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMM_PROFILE = load_module(PROFILE / "vllm_051_comm_profile.py", "send_recv_comm_profile")


def source_definitions(path, names, dependencies=None):
    """Compile just named definitions, supplying lightweight test doubles."""
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign):
            targets = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if targets & set(names):
                selected.append(node)
    env = dict(dependencies or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), env)
    return SimpleNamespace(**env)


class RoundTimer:
    def __init__(self, *args, **kwargs):
        pass


class ProfilingRoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        singleton_path = ROOT / "vidur" / "profiling" / "utils" / "singleton.py"
        singleton_defs = source_definitions(singleton_path, ["Singleton"])
        store_defs = source_definitions(
            ROOT / "vidur" / "profiling" / "common" / "timer_stats_store.py",
            ["TimerStatsStore"],
            {"np": __import__("numpy"), "Singleton": singleton_defs.Singleton,
             "ProfileMethod": enum.Enum("ProfileMethod", {
                 "CUDA_EVENT": "cuda_event", "KINETO": "kineto",
                 "PERF_COUNTER": "perf_counter", "RECORD_FUNCTION": "record_function"})},
        )
        cls.wrapper_defs = source_definitions(
            PROFILE / "collectives_wrapper.py",
            ["ACTIVE_STEPS", "SEND_RECV_PROFILE_ROUNDS", "resolve_num_profile_rounds",
             "CollectiveWrapper"],
            {"np": SimpleNamespace(median=statistics.median),
             "torch": SimpleNamespace(), "GraphedCollective": lambda *args, **kwargs: object(),
             "CudaTimer": RoundTimer, "TimerStatsStore": store_defs.TimerStatsStore,
             "DISABLE_GRAPH": True, "GRAPH_DISABLED_STEPS": 10,
             "WARMUP_STEPS": 1},
        )

    def test_round_defaults_and_overrides(self):
        resolve = self.wrapper_defs.resolve_num_profile_rounds
        self.assertEqual(resolve("send_recv"), 60)
        self.assertEqual(resolve("all_reduce"), self.wrapper_defs.ACTIVE_STEPS)
        self.assertEqual(resolve("broadcast"), self.wrapper_defs.ACTIVE_STEPS)
        self.assertEqual(resolve("send_recv", 7), 7)
        self.assertEqual(resolve("all_reduce", 9), 9)
        for collective in ("send_recv", "all_reduce"):
            with self.subTest(collective=collective), self.assertRaises(ValueError):
                resolve(collective, 0)

    def test_default_send_recv_profile_runs_sixty_rounds_and_reports_mean(self):
        wrapper = self.wrapper_defs.CollectiveWrapper(
            rank=0, num_workers=2, comm_id=1, size=4, collective="send_recv",
            devices_per_node=2, max_devices_per_node=2,
        )
        timings = iter([*range(1, 60), 1000])
        calls = []

        def run_round():
            value = next(timings)
            calls.append(value)
            wrapper.timer_stats_store.record_time("send_recv", float(value))

        wrapper._run_collective = run_round
        result = wrapper.profile()
        self.assertEqual(len(calls), 60)
        self.assertEqual(result["time_stats"]["send_recv"]["mean"], statistics.fmean([*range(1, 60), 1000]))
        self.assertEqual(result["time_stats"]["send_recv"]["median"], 30.5)

    def test_custom_round_count_reaches_profile_loop(self):
        wrapper = self.wrapper_defs.CollectiveWrapper(
            rank=0, num_workers=2, comm_id=1, size=4, collective="send_recv",
            devices_per_node=2, max_devices_per_node=2, num_profile_rounds=4,
        )
        calls = []
        def run_round():
            value = len(calls) + 1
            calls.append(value)
            wrapper.timer_stats_store.record_time("send_recv", float(value))
        wrapper._run_collective = run_round
        wrapper.profile()
        self.assertEqual(len(calls), 4)


class VllmContractTests(unittest.TestCase):
    def test_contract_uses_collective_default_and_reports_mean_aggregation(self):
        send_recv = COMM_PROFILE.contract(ROOT, collective="send_recv")
        send_recv_override = COMM_PROFILE.contract(ROOT, 11, collective="send_recv")
        overridden = COMM_PROFILE.contract(ROOT, 11)
        default_contract = COMM_PROFILE.contract(ROOT)
        default_rounds = COMM_PROFILE.load_grid(ROOT)[1].ACTIVE_STEPS
        self.assertEqual(send_recv["active_steps"], 60)
        self.assertEqual(send_recv_override["active_steps"], 11)
        self.assertEqual(overridden["active_steps"], 11)
        self.assertEqual(default_contract["active_steps"], default_rounds)
        self.assertEqual(overridden["active_steps"], 11)
        self.assertEqual(send_recv["rounds_aggregation"], "arithmetic mean of round timings")


class PredictorTargetTests(unittest.TestCase):
    def test_send_recv_training_uses_mean_target(self):
        path = ROOT / "vidur" / "execution_time_predictor" / "sklearn_execution_time_predictor.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        cls = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef)
                   and node.name == "SklearnExecutionTimePredictor")
        method = next(node for node in cls.body
                      if isinstance(node, ast.FunctionDef)
                      and node.name == "_train_compute_models")
        # Execute the real method body against a recorder to avoid importing
        # sklearn, pandas, torch, Ray, or the full simulator configuration.
        method.decorator_list = []
        env = {"Dict": dict, "BaseEstimator": object,
               "logger": SimpleNamespace(debug=lambda *args, **kwargs: None)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), env)

        class PredictorHarness:
            _replica_config = SimpleNamespace(num_pipeline_stages=2, tensor_parallel_size=1)
            _compute_input_file = "unused"
            _attention_input_file = "unused"
            _send_recv_input_file = "unused"

            def __init__(self):
                self.targets = {}

            def _load_compute_df(self, _): return [1]
            def _get_compute_df_with_derived_features(self, df): return df
            def _train_model(self, model_name, target_col, **kwargs):
                self.targets[model_name] = target_col
                return object()
            def _load_attention_df(self, _): return object()
            def _get_attention_df_with_derived_features(self, df): return df
            def _load_send_recv_df(self, _): return object()
            def _get_send_recv_df_with_derived_features(self, df): return df

        harness = PredictorHarness()
        env["_train_compute_models"](harness)
        self.assertEqual(harness.targets["send_recv"], "time_stats.send_recv.mean")

    @classmethod
    def setUpClass(cls):
        cls.predictor_path = ROOT / "vidur" / "execution_time_predictor" / "sklearn_execution_time_predictor.py"
        tree = ast.parse(cls.predictor_path.read_text(), filename=str(cls.predictor_path))
        cls.predictor_class = next(node for node in tree.body
                                   if isinstance(node, ast.ClassDef)
                                   and node.name == "SklearnExecutionTimePredictor")

    def extract_method(self, method_name, env):
        method = next(node for node in self.predictor_class.body
                      if isinstance(node, ast.FunctionDef) and node.name == method_name)
        method.decorator_list = []
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(self.predictor_path), "exec"), env)
        return env[method_name]

    def test_training_hash_includes_target_column(self):
        method = self.extract_method("_get_model_hash", {"hashlib": hashlib})

        class Frame:
            def to_json(self): return '{"same": "training rows"}'

        harness = SimpleNamespace(to_dict=lambda: {"model": "test"})
        base = method(harness, "send_recv", Frame())
        mean_hash = method(harness, "send_recv", Frame(), "time_stats.send_recv.mean")
        median_hash = method(harness, "send_recv", Frame(), "time_stats.send_recv.median")
        self.assertNotEqual(base, mean_hash)
        self.assertNotEqual(mean_hash, median_hash)

    def test_prediction_cache_key_includes_training_hash_and_inputs(self):
        method = self.extract_method(
            "_get_model_prediction",
            {"Dict": dict, "Tuple": tuple, "BaseEstimator": object,
             "pd": SimpleNamespace(DataFrame=object), "hashlib": hashlib,
             "pickle": pickle, "logger": SimpleNamespace(info=lambda *a, **k: None)},
        )

        class Frame:
            def __init__(self, values): self.values = [values]
            def copy(self): return Frame(self.values[0])
            def to_json(self): return repr(self.values)
            def __setitem__(self, key, value): pass
            def to_csv(self, *args, **kwargs): pass

        class Model:
            def predict(self, _): return [1.0]

        class Harness:
            _cache_dir = "/tmp"

            def __init__(self, training_hash):
                self._model_training_hashes = {"send_recv": training_hash}
                self.cache_keys = []
            def _load_model_predication_cache(self, name, model_hash):
                self.cache_keys.append(model_hash)
                return {}
            def _store_model_predication_cache(self, *args): pass

        harness = Harness("training-a")
        method(harness, "send_recv", Model(), Frame([16]))
        method(harness, "send_recv", Model(), Frame([16]))
        method(harness, "send_recv", Model(), Frame([32]))
        self.assertEqual(harness.cache_keys[0], harness.cache_keys[1])
        self.assertNotEqual(harness.cache_keys[0], harness.cache_keys[2])

        other_training = Harness("training-b")
        method(other_training, "send_recv", Model(), Frame([16]))
        self.assertNotEqual(harness.cache_keys[0], other_training.cache_keys[0])

    def test_random_forest_training_uses_mean_target_and_reuses_training_cache(self):
        method_names = {
            "_get_model_hash", "_load_model_from_cache", "_store_model_in_cache",
            "_store_training_prediction_data", "_train_model",
            "_load_model_predication_cache", "_store_model_predication_cache",
            "_get_model_prediction",
        }
        methods = [node for node in self.predictor_class.body
                   if isinstance(node, ast.FunctionDef) and node.name in method_names]
        for method in methods:
            method.decorator_list = []
        harness_node = ast.ClassDef(name="PredictorHarness", bases=[], keywords=[],
                                    body=methods, decorator_list=[])
        harness_node = ast.fix_missing_locations(harness_node)

        class DummyLock:
            def __init__(self, *args, **kwargs): pass
            def read_lock(self): return self
            def write_lock(self): return self
            def __enter__(self): return self
            def __exit__(self, *args): return False

        env = {
            "BaseEstimator": object, "Dict": dict, "Tuple": tuple, "List": list,
            "pd": SimpleNamespace(DataFrame=pd.DataFrame), "hashlib": hashlib,
            "os": __import__("os"), "pickle": pickle,
            "InterProcessReaderWriterLock": DummyLock, "GridSearchCV": GridSearchCV,
            "logger": SimpleNamespace(debug=lambda *a, **k: None,
                                       info=lambda *a, **k: None),
        }
        exec(compile(ast.Module(body=[harness_node], type_ignores=[]),
                     str(self.predictor_path), "exec"), env)
        Harness = env["PredictorHarness"]

        class RandomForestHarness(Harness):
            def __init__(self, cache_dir):
                self._cache_dir = cache_dir
                self._config = SimpleNamespace(no_cache=False, k_fold_cv_splits=2,
                                               num_training_job_threads=1)
                self._model_training_hashes = {}
                self.fit_calls = 0

            def to_dict(self): return {"test_model": "rf"}
            def _get_estimator(self):
                self.fit_calls += 1
                return RandomForestRegressor(random_state=7)
            def _get_grid_search_params(self):
                return {"n_estimators": [4], "max_depth": [2]}
            def _get_scorer(self):
                return make_scorer(mean_squared_error, greater_is_better=False)

        with tempfile.TemporaryDirectory() as cache_dir:
            harness = RandomForestHarness(cache_dir)
            frame = pd.DataFrame({
                "num_tokens": [1, 2, 3, 4],
                "time_stats.send_recv.mean": [10.0, 20.0, 30.0, 40.0],
            })
            kwargs = {"model_name": "send_recv", "feature_cols": ["num_tokens"],
                      "target_col": "time_stats.send_recv.mean"}
            first_model = harness._train_model(df=frame, **kwargs)
            first_hash = harness._model_training_hashes["send_recv"]
            first_predictions = harness._get_model_prediction(
                "send_recv", first_model, pd.DataFrame({"num_tokens": [1, 4]}))
            training_csv = pd.read_csv(Path(cache_dir) /
                                       f"send_recv_{first_hash}_training_predictions.csv")
            self.assertEqual(training_csv["time_stats.send_recv.mean"].tolist(),
                             [10.0, 20.0, 30.0, 40.0])

            changed = frame.copy()
            changed["time_stats.send_recv.mean"] += 100.0
            second_model = harness._train_model(df=changed, **kwargs)
            second_hash = harness._model_training_hashes["send_recv"]
            second_predictions = harness._get_model_prediction(
                "send_recv", second_model, pd.DataFrame({"num_tokens": [1, 4]}))
            self.assertNotEqual(first_hash, second_hash)
            self.assertNotEqual(first_predictions, second_predictions)
            self.assertEqual(harness.fit_calls, 2)

            reused_model = harness._train_model(df=changed.copy(), **kwargs)
            self.assertEqual(harness._model_training_hashes["send_recv"], second_hash)
            self.assertEqual(harness.fit_calls, 2)
            self.assertEqual(
                harness._get_model_prediction(
                    "send_recv", reused_model, pd.DataFrame({"num_tokens": [1, 4]})),
                second_predictions,
            )


if __name__ == "__main__":
    unittest.main()
