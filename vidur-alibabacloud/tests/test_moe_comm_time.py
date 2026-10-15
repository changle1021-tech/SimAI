"""Regression tests for topology-aware MoE EP communication time.

Extract the production methods from execution_time.py so the test runs without
importing the simulator, CUDA, torch, or Ray.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vidur" / "entities" / "execution_time.py"
METHODS = (
    "_get_moe_layer_execution_time_from_aicb",
    "_is_moe_comm_cross_node",
    "_get_moe_comm_time",
)


def load_methods():
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ExecutionTime")
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    assert len(nodes) == len(METHODS)
    holder = ast.ClassDef(name="Harness", bases=[], keywords=[], body=nodes, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[holder], type_ignores=[]))
    env = {"logger": SimpleNamespace(warning=lambda *a, **k: None)}
    exec(compile(module, str(SOURCE), "exec"), env)
    return env["Harness"]


GBPS = 1024 ** 3 / 8


class MoECommunicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Harness = load_methods()

    def make(self, ep, gpus_per_node=None, devices_per_node=8, intra=0.0, inter=0.0,
             comp_ns=0.0, comm_bytes=0.0):
        obj = self.Harness()
        obj._replica_config = SimpleNamespace(
            expert_model_parallel_size=ep,
            num_gpus_per_node=gpus_per_node,
            node_config=SimpleNamespace(num_devices_per_node=devices_per_node),
            nvlink_bandwidth=3600,
            rdma_bandwidth=800,
            moe_comm_intra_node_latency_us=intra,
            moe_comm_inter_node_latency_us=inter,
        )
        obj._aicb_data = {0: {"moe": {"comp_time": comp_ns, "comm_size": comm_bytes}}}
        return obj

    def test_intra_node_uses_nvlink_bandwidth(self):
        obj = self.make(ep=4)
        self.assertFalse(obj._is_moe_comm_cross_node())
        self.assertAlmostEqual(obj._get_moe_comm_time(3600 * GBPS), 1.0)

    def test_cross_node_uses_rdma_bandwidth(self):
        obj = self.make(ep=16)
        self.assertTrue(obj._is_moe_comm_cross_node())
        self.assertAlmostEqual(obj._get_moe_comm_time(800 * GBPS), 1.0)

    def test_explicit_gpus_per_node_overrides_sku(self):
        obj = self.make(ep=4, gpus_per_node=2)
        self.assertTrue(obj._is_moe_comm_cross_node())

    def test_fixed_latency_is_added_per_layer(self):
        obj = self.make(ep=4, intra=23.5, inter=50.0)
        self.assertAlmostEqual(obj._get_moe_comm_time(0.0), 0.0)
        self.assertAlmostEqual(obj._get_moe_comm_time(1.0), 23.5e-6 + 1.0 / (3600 * GBPS))
        obj = self.make(ep=16, intra=23.5, inter=50.0)
        self.assertAlmostEqual(obj._get_moe_comm_time(1.0), 50.0e-6 + 1.0 / (800 * GBPS))

    def test_layer_time_is_compute_plus_communication(self):
        obj = self.make(ep=4, intra=10.0, comp_ns=2000.0, comm_bytes=3600 * GBPS * 1e-6)
        self.assertAlmostEqual(obj._get_moe_layer_execution_time_from_aicb(0), 2e-6 + 10e-6 + 1e-6)

    def test_source_has_no_phase_hardcoded_bandwidth(self):
        self.assertNotIn('replica_stage = "prefill"', SOURCE.read_text())


if __name__ == "__main__":
    unittest.main()
