"""CPU-only wiring checks for Qwen AICB paths in ExecutionTime."""
import ast
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import Mock, patch
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict
from collections import OrderedDict
from enum import Enum


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vidur" / "entities" / "execution_time.py"
PROFILE_SOURCE = ROOT / "vidur" / "entities" / "qwen_aicb_profiles.py"
AIOB_WORKLOAD_SOURCE = ROOT.parent / "aicb" / "workload_generator" / "Vidur_workload_generator.py"
METHODS = (
    "_load_aicb_data",
    "_get_mlp_layer_execution_time_from_aicb",
    "_get_moe_layer_execution_time_from_aicb",
    "_get_cpu_overhead",
    "_get_moe_comm_time",
)


def load_methods():
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ExecutionTime")
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    assert len(nodes) == len(METHODS)
    holder = ast.ClassDef(name="Harness", bases=[], keywords=[], body=nodes, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[holder], type_ignores=[]))
    spec = importlib.util.spec_from_file_location("qwen_aicb_profiles_for_wiring", str(PROFILE_SOURCE))
    profiles = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profiles)
    env = {"QwenAICBProfiles": profiles.QwenAICBProfiles,
           "RuntimeBoundaryProfile": profiles.RuntimeBoundaryProfile,
           "RuntimeMoECommunication": profiles.RuntimeMoECommunication,
           "_AICB_ROOT": "/unused/aicb/root", "Dict": Dict}
    exec(compile(module, str(SOURCE), "exec"), env)
    return env["Harness"], env


class CacheMustNotBeUsed(object):
    def get(self, *args, **kwargs):
        raise AssertionError("Qwen must bypass the legacy global AICB cache")


class FakeQwenProfiles(object):
    instances = []

    def __init__(self, root, profile_dir):
        self.root = root
        self.profile_dir = profile_dir
        self.calls = []
        self.instances.append(self)

    def load(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"loaded": True}


class WorkloadLayerType(Enum):
    ATTENTION = "attention"
    MOE = "moe"


def load_workload_method():
    tree = ast.parse(AIOB_WORKLOAD_SOURCE.read_text(), filename=str(AIOB_WORKLOAD_SOURCE))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VidurWorkload")
    method = next(n for n in cls.body
                  if isinstance(n, ast.FunctionDef) and n.name == "workload_generate_aiob")

    def layer_type(name):
        if "attention" in name:
            return WorkloadLayerType.ATTENTION
        if "moe" in name:
            return WorkloadLayerType.MOE
        return None

    env = {
        "OrderedDict": OrderedDict,
        "LayerType": WorkloadLayerType,
        "_get_model_details": lambda model: model.layers,
        "_extract_layer_type": layer_type,
        "_get_aiob_compute_time": lambda cache, phase, name: cache["forward"][name],
        "_get_layer_comm_size": lambda layer_type, tp, ep: 0,
    }
    holder = ast.ClassDef(name="WorkloadHarness", bases=[], keywords=[], body=[method], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[holder], type_ignores=[]))
    exec(compile(module, str(AIOB_WORKLOAD_SOURCE), "exec"), env)
    return env["WorkloadHarness"]


class QwenAICBWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Harness, cls.env = load_methods()

    def setUp(self):
        FakeQwenProfiles.instances = []
        self.obj = self.Harness()
        self.obj._replica_config = SimpleNamespace(
            model_name="Qwen3-Next-80B", context_len=37)
        self.obj._config = SimpleNamespace(aicb_profile_dir="/profiles")
        self.obj._aicb_data = None
        self.obj._get_aicb_params = lambda: (
            "Qwen3-Next-80B", None, 1, 1, 4, 4, 1, 128, "prefill")
        self.env["_GLOBAL_AICB_CACHE"] = CacheMustNotBeUsed()
        self.env["QwenAICBProfiles"] = FakeQwenProfiles

    def test_qwen_loader_calls_profile_store_with_context_and_bypasses_global_cache(self):
        result = self.obj._load_aicb_data()
        self.assertEqual(result, {"loaded": True})
        self.assertEqual(len(FakeQwenProfiles.instances), 1)
        store = FakeQwenProfiles.instances[0]
        self.assertEqual((store.root, store.profile_dir), ("/unused/aicb/root", "/profiles"))
        self.assertEqual(store.calls, [((4, 1, 1, 4, 1, 128, "prefill", 37), {})])

    def write_json(self, value):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "profile.json"
        path.write_text(json.dumps(value))
        return str(path)

    @staticmethod
    def boundary_profile():
        return {
            "schema": "qwen_worker_critical_perimeter_v2",
            "scope": "host_iteration_wall_minus_model_gpu",
            "configuration": {"model": "qwen3-next-80B", "world_size": 4,
                              "tp": 1, "pp": 1, "ep": 4, "batch_size": 1},
            "phases": {"prefill": [{"tokens": 128,
                                     "critical_perimeter_ms": 2.5}]}}

    def test_cpu_overhead_uses_only_measured_perimeter_in_milliseconds(self):
        path = self.write_json(self.boundary_profile())
        self.obj._config = SimpleNamespace(backend="aicb", skip_cpu_overhead_modeling=False,
                                           aicb_cpu_overhead_profile=path)
        self.obj._replica_config.model_name = "qwen3-next-80B"
        self.obj._schedule_time = 1e8
        self.obj._sampler_e2e_time = 2e8
        self.obj._prepare_inputs_e2e_time = 3e8
        self.obj._process_model_outputs_time = 4e8
        self.obj._ray_comm_time = 5e8
        self.assertEqual(self.obj._get_cpu_overhead(), 2.5)

    def test_cpu_overhead_disabled_is_zero_and_enabled_without_profile_errors(self):
        self.obj._config = SimpleNamespace(backend="aicb", skip_cpu_overhead_modeling=True,
                                           aicb_cpu_overhead_profile=None)
        self.obj._replica_config.model_name = "qwen3-next-80B"
        self.assertEqual(self.obj._get_cpu_overhead(), 0.0)
        self.obj._config.skip_cpu_overhead_modeling = False
        with self.assertRaisesRegex(ValueError, "Qwen CPU modeling requires"):
            self.obj._get_cpu_overhead()

    def test_qwen_mlp_component_is_zero_without_loading_profiles(self):
        self.obj._replica_config.model_name = "qwen3-next-80B"
        self.assertEqual(self.obj._get_mlp_layer_execution_time_from_aicb(7), 0.0)
        self.assertIsNone(self.obj._aicb_data)
        self.assertEqual(FakeQwenProfiles.instances, [])

    def test_moe_profile_with_embedded_ep_communication_skips_communication_estimate(self):
        self.obj._aicb_data = {0: {"moe": {"comp_time": 3000000.0,
                                             "comm_size": 123456.0,
                                             "includes_ep_comm": 1.0}}}
        self.obj._get_moe_comm_time = Mock(
            side_effect=AssertionError("profile time already includes EP communication"))
        self.assertAlmostEqual(self.obj._get_moe_layer_execution_time_from_aicb(0), 0.003)
        self.obj._get_moe_comm_time.assert_not_called()

    def test_qwen_moe_communication_uses_runtime_measurement_and_converts_ms_to_seconds(self):
        data = {"world_size": 4, "dtype": "bfloat16", "profiles": [
            {"mode": "eager", "active_tokens": 128, "sizes": [128] * 4, "median_ms": 1.0},
            {"mode": "cuda_graph", "active_tokens": 1, "sizes": [1] * 4, "median_ms": 2.75}]}
        path = self.write_json(data)
        self.obj._config = SimpleNamespace(aicb_moe_comm_profile=path)
        self.obj._replica_config.model_name = "qwen3-next-80B"
        # Decode communication is keyed by batch size, and uses CUDA graph data.
        self.obj._get_aicb_params = lambda: (
            "Qwen3-Next-80B", None, 1, 1, 4, 4, 1, 128, "decode")
        self.assertAlmostEqual(self.obj._get_moe_comm_time(999999), 0.00275)

    def test_moe_csv_retains_external_router_and_accepts_filtered_zero(self):
        module_name = "workload_generator.mocked_model.inference.qwen3_next_moe"
        profile = SimpleNamespace(load_critical_moe_profile=lambda args: {0: 2.0})
        for router in (None, 13.0):
            with self.subTest(router_ns=router), patch.dict(sys.modules, {module_name: profile}):
                obj = load_workload_method()()
                obj.model = SimpleNamespace(layers=[SimpleNamespace(layer_id=0, layer_name="layer0_moe")])
                obj.args = SimpleNamespace(model_name="Qwen3-Next-80B", phase="prefill",
                                           full_attention_interval=1, moe_compute_profile="measured.json")
                obj.compute_cache = {"forward": {"layer0_moe": 1}, "moe_norm": 11.0}
                if router is not None:
                    obj.compute_cache["moe_route"] = router
                obj.tp = 1
                obj.get_comm_size = lambda: (0, 0, 0)
                obj.workload_generate_aiob()
                self.assertEqual(obj.vidur_layers[0][2], 2000011.0 + (router or 0))

    def test_prefill_gdn_norms_are_subtracted_once_while_full_attention_norms_remain(self):
        Harness = load_workload_method()
        names = ["layer0_attention_gdn", "layer0_moe_gdn",
                 "layer1_attention_full", "layer1_moe_full",
                 "layer2_attention_gdn", "layer2_moe_gdn"]
        layers = [SimpleNamespace(layer_id=i // 2, layer_name=name)
                  for i, name in enumerate(names)]
        values = dict(zip(names, (100.0, 200.0, 300.0, 400.0, 500.0, 600.0)))
        obj = Harness()
        obj.model = SimpleNamespace(layers=layers)
        obj.args = SimpleNamespace(model_name="Qwen3-Next-80B", phase="prefill",
                                   full_attention_interval=2, moe_compute_profile=None)
        obj.compute_cache = {"forward": values, "attention_norm": 7.0, "moe_norm": 11.0}
        obj.vidur_layers = []
        obj.tp = 1
        obj.get_comm_size = lambda: (0, 0, 0)
        obj.workload_generate_aiob()
        costs = {(row[0], row[1].value): row[2] for row in obj.vidur_layers}
        self.assertEqual(costs, {
            (0, "attention"): 93.0,
            (0, "moe"): 189.0,
            (1, "attention"): 300.0,
            (1, "moe"): 400.0,
            (2, "attention"): 493.0,
            (2, "moe"): 589.0,
        })


if __name__ == "__main__":
    unittest.main()
