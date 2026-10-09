"""Exercise CLI configuration and topology-specific communication lookups."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from vidur.config import ReplicaConfig, SimulationConfig
from vidur.config.utils import dataclass_to_dict
from vidur.execution_time_predictor.communication_time_predictor import TPTimePredictor
from vidur.execution_time_predictor.random_forrest_execution_time_predictor import RandomForrestExecutionTimePredictor


class NodeLayoutProfileTests(unittest.TestCase):
    def config(self, tp=2, pp=2, nodes=2, gpus=2):
        return ReplicaConfig(device="h100", network_device="h100_dgx",
                             tensor_parallel_size=tp, num_pipeline_stages=pp,
                             num_nodes=nodes, num_gpus_per_node=gpus)

    def predictor(self, config):
        predictor = object.__new__(RandomForrestExecutionTimePredictor)
        predictor._replica_config = config
        predictor._config = SimpleNamespace(nccl_cpu_launch_overhead_ms=0,
                                           nccl_cpu_skew_overhead_per_device_ms=0)
        return predictor

    def test_cli_parses_both_new_flags(self):
        with tempfile.TemporaryDirectory() as output, patch("sys.argv", ["vidur",
                "--replica_config_network_device", "h100_dgx",
                "--replica_config_tensor_parallel_size", "2",
                "--replica_config_num_pipeline_stages", "2",
                "--replica_config_num_nodes", "2",
                "--replica_config_num_gpus_per_node", "2",
                "--metrics_config_output_dir", output]):
            config = SimulationConfig.create_from_cli_args()
            rc = config.cluster_config.replica_config
            self.assertEqual(rc.placement.rank_location(2), (1, 0))
            stored = json.loads((Path(config.metrics_config.output_dir) / "config.json").read_text())
            self.assertEqual(stored["cluster_config"]["replica_config"]["num_nodes"], 2)
            self.assertEqual(stored["cluster_config"]["replica_config"]["num_gpus_per_node"], 2)

    def test_defaults_and_inferred_node_count(self):
        rc = ReplicaConfig(network_device="h100_dgx", tensor_parallel_size=4)
        self.assertEqual((rc.placement.num_nodes, rc.placement.gpus_per_node), (1, 8))
        rc = ReplicaConfig(network_device="h100_dgx", tensor_parallel_size=2,
                           num_pipeline_stages=3, num_gpus_per_node=2)
        self.assertEqual(rc.placement.num_nodes, 3)
        json.dumps(dataclass_to_dict(rc))

    def test_invalid_capacity_and_values(self):
        for overrides in ({"num_nodes": 1}, {"num_nodes": 0}, {"num_gpus_per_node": 0},
                          {"num_gpus_per_node": 9}, {"tensor_parallel_size": -1}):
            values = dict(device="h100", network_device="h100_dgx", tensor_parallel_size=2,
                          num_pipeline_stages=2, num_nodes=2, num_gpus_per_node=2)
            values.update(overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                ReplicaConfig(**values)

    def test_profile_rows_select_actual_tp_occupancy_and_pp_edge(self):
        predictor = self.predictor(self.config(tp=4, pp=1, nodes=2, gpus=2))
        rows = pd.DataFrame([
            dict(collective="all_reduce", num_workers=4, devices_per_node=4, marker="local"),
            dict(collective="all_reduce", num_workers=4, devices_per_node=2, marker="remote"),
            dict(collective="send_recv", num_workers=2, devices_per_node=1, marker="remote"),
            dict(collective="send_recv", num_workers=2, devices_per_node=2, marker="local"),
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "profiles.csv")
            rows.to_csv(path, index=False)
            self.assertEqual(predictor._load_all_reduce_df(path)["marker"].tolist(), ["remote"])
            predictor._replica_config = self.config(tp=2, pp=4, gpus=4)
            self.assertEqual(predictor._load_send_recv_df(path)["marker"].tolist(), ["local"])
            self.assertEqual(predictor._load_send_recv_df(path, 1)["marker"].tolist(), ["remote"])
            with self.assertRaisesRegex(ValueError, "Missing all_reduce profile"):
                predictor._load_all_reduce_df(path, 3)

    def test_mixed_pp_edges_use_separate_prediction_tables(self):
        predictor = self.predictor(self.config(tp=2, pp=4, gpus=4))
        self.assertEqual(predictor._communication_profiles("send_recv"),
                         {"send_recv_dpn1": 1, "send_recv_dpn2": 2})
        predictor._predictions = {"send_recv_dpn1": {(8,): 7.0}, "send_recv_dpn2": {(8,): 1.0}}
        batch = SimpleNamespace(_total_num_tokens_rounded=8)
        self.assertEqual([predictor._get_pipeline_parallel_communication_time(batch, stage)
                          for stage in range(3)], [1.0, 7.0, 1.0])

    def test_mixed_tp_edges_use_separate_prediction_tables(self):
        predictor = self.predictor(self.config(tp=2, pp=3, gpus=3))
        predictor._predictions = {"all_reduce_dpn1": {(8,): 9.0}, "all_reduce_dpn2": {(8,): 2.0}}
        batch = SimpleNamespace(_total_num_tokens_rounded=8)
        self.assertEqual([predictor._get_tensor_parallel_communication_time(batch, stage)
                          for stage in range(3)], [2.0, 9.0, 2.0])

    def test_single_shape_keeps_existing_prediction_names(self):
        predictor = self.predictor(self.config())
        self.assertEqual(predictor._communication_profiles("all_reduce"), {"all_reduce": 2})
        self.assertEqual(predictor._communication_profiles("send_recv"), {"send_recv": 1})

    def test_simai_cache_isolated_by_layout_and_topology_checked(self):
        predictor = object.__new__(TPTimePredictor)
        predictor.replica_config = self.config()
        predictor.num_layers_per_pp_stage = 16
        original = predictor._placement_cache_key()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topology"
            path.write_text("6 2 1 1 8\n")
            predictor._validate_topology_placement(path)
            predictor.replica_config = self.config(nodes=1, gpus=4)
            self.assertNotEqual(predictor._placement_cache_key(), original)
            with self.assertRaisesRegex(ValueError, "differs"):
                predictor._validate_topology_placement(path)


if __name__ == "__main__":
    unittest.main()
