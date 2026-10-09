"""Tests for rank-to-node placement and TP/PP communication topology."""

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLACEMENT_MODULE = ROOT / "vidur" / "config" / "replica_placement.py"


def load_replica_placement():
    # Load the module directly so these tests do not import Vidur's optional
    # runtime dependencies. Register it first because dataclasses consult
    # sys.modules while resolving annotations.
    module_name = "vidur_replica_placement_test_module"
    spec = importlib.util.spec_from_file_location(module_name, PLACEMENT_MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load replica placement module: {PLACEMENT_MODULE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ReplicaPlacement = load_replica_placement().ReplicaPlacement


class ReplicaPlacementTests(unittest.TestCase):
    def test_two_nodes_two_gpus_tp2_pp2(self):
        placement = ReplicaPlacement(
            tensor_parallel_size=2,
            num_pipeline_stages=2,
            num_nodes=2,
            gpus_per_node=2,
        )

        self.assertEqual(placement.world_size, 4)
        self.assertEqual(placement.tp_groups, ((0, 1), (2, 3)))
        self.assertEqual(placement.pp_groups, ((0, 2), (1, 3)))
        self.assertEqual(
            [placement.rank_location(rank) for rank in range(4)],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
        )
        self.assertEqual(placement.tp_devices_per_node(0), 2)
        self.assertEqual(placement.pp_devices_per_node(0), 1)
        with self.assertRaises(ValueError):
            placement.pp_devices_per_node(1)

    def test_two_nodes_four_gpus_tp2_pp4_pipeline_edges(self):
        placement = ReplicaPlacement(
            tensor_parallel_size=2,
            num_pipeline_stages=4,
            num_nodes=2,
            gpus_per_node=4,
        )

        self.assertEqual(placement.world_size, 8)
        self.assertEqual(placement.tp_groups, ((0, 1), (2, 3), (4, 5), (6, 7)))
        self.assertEqual(placement.pp_groups, ((0, 2, 4, 6), (1, 3, 5, 7)))
        self.assertEqual(
            [placement.pp_devices_per_node(stage) for stage in range(3)],
            [2, 1, 2],
        )
        with self.assertRaises(ValueError):
            placement.pp_devices_per_node(3)

    def test_two_nodes_four_gpus_tp8_pp1_cross_node_tensor_parallel(self):
        placement = ReplicaPlacement(
            tensor_parallel_size=8,
            num_pipeline_stages=1,
            num_nodes=2,
            gpus_per_node=4,
        )

        self.assertEqual(placement.world_size, 8)
        self.assertEqual(placement.tp_groups, (tuple(range(8)),))
        self.assertEqual(placement.tp_devices_per_node(0), 4)
        with self.assertRaises(ValueError):
            placement.pp_devices_per_node(0)

    def test_default_eight_gpu_inference_layout(self):
        placement = ReplicaPlacement(
            tensor_parallel_size=1,
            num_pipeline_stages=1,
            num_nodes=1,
            gpus_per_node=8,
        )

        self.assertEqual(placement.world_size, 1)
        self.assertEqual(placement.rank_location(0), (0, 0))
        self.assertEqual(placement.tp_groups, ((0,),))
        self.assertEqual(placement.pp_groups, ((0,),))
        self.assertEqual(placement.tp_devices_per_node(0), 1)
        with self.assertRaises(ValueError):
            placement.pp_devices_per_node(0)

    def test_tp4_pp2_three_gpus_per_node_detects_nonuniform_tp_placement(self):
        placement = ReplicaPlacement(
            tensor_parallel_size=4,
            num_pipeline_stages=2,
            num_nodes=3,
            gpus_per_node=3,
        )

        self.assertEqual(placement.world_size, 8)
        self.assertEqual(placement.tp_groups, ((0, 1, 2, 3), (4, 5, 6, 7)))
        self.assertEqual(placement.rank_location(3), (1, 0))
        # Stage 0 occupies 3+1 GPUs across its participating nodes, while
        # stage 1 occupies 2+2. One CSV scalar cannot represent stage 0.
        with self.assertRaises(ValueError):
            placement.tp_devices_per_node(0)
        self.assertEqual(placement.tp_devices_per_node(1), 2)

    def test_insufficient_gpu_capacity_is_rejected(self):
        with self.assertRaises(ValueError):
            ReplicaPlacement(
                tensor_parallel_size=2,
                num_pipeline_stages=2,
                num_nodes=1,
                gpus_per_node=3,
            )

    def test_rank_and_pipeline_stage_bounds_are_checked(self):
        placement = ReplicaPlacement(
            tensor_parallel_size=2,
            num_pipeline_stages=2,
            num_nodes=2,
            gpus_per_node=2,
        )

        for rank in (-1, placement.world_size):
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                placement.rank_location(rank)
        for stage in (-1, placement.num_pipeline_stages):
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                placement.tp_devices_per_node(stage)
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                placement.pp_devices_per_node(stage)

    def test_topology_dimensions_must_be_positive_integers(self):
        valid = {
            "tensor_parallel_size": 2,
            "num_pipeline_stages": 2,
            "num_nodes": 2,
            "gpus_per_node": 2,
        }
        for field in valid:
            for value in (0, -1, 1.5, True):
                arguments = dict(valid)
                arguments[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    ReplicaPlacement(**arguments)


if __name__ == "__main__":
    unittest.main()
