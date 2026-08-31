"""Regression tests for SimAI tensor-parallel communication prediction."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidur.execution_time_predictor.communication_time_predictor import (
    TPTimePredictor,
)


class _WorkloadStub:
    def flush(self):
        pass


def _make_cached_predictor(
    latency_ms: float = 0.013, custom_allreduce: bool = False
) -> TPTimePredictor:
    predictor = TPTimePredictor.__new__(TPTimePredictor)
    predictor.hidden_size = 8192
    predictor.tensor_size = 2
    predictor.workload = _WorkloadStub()
    predictor.predictor_config = SimpleNamespace(
        nccl_cpu_launch_overhead_ms=0.02,
        nccl_cpu_skew_overhead_per_device_ms=0.01,
        simai_sglang_custom_allreduce=custom_allreduce,
        simai_sglang_custom_allreduce_max_bytes=64 * 1024,
        simai_sglang_custom_allreduce_latency_ms=0.0125,
    )
    predictor.replica_config = SimpleNamespace(
        tensor_parallel_size=4,
        network_device="h100_dgx",
    )
    predictor.cache = {(8192, 1, 2): latency_ms}
    return predictor


class TestTPTimePredictor(unittest.TestCase):
    def test_decode_all_reduce_uses_actual_token_count(self):
        predictor = _make_cached_predictor()
        batch = SimpleNamespace(total_num_tokens=1, _total_num_tokens_rounded=8)

        num_tokens, num_bytes, cache_key = predictor._get_all_reduce_shape(batch)

        self.assertEqual(num_tokens, 1)
        self.assertEqual(num_bytes, 16 * 1024)
        self.assertEqual(cache_key, (8192, 1, 2))

    def test_cached_simai_latency_does_not_add_cpu_overhead(self):
        batch = SimpleNamespace(total_num_tokens=1, _total_num_tokens_rounded=8)
        method_names = (
            "get_execution_time",
            "get_execution_time_by_simai_analytical",
        )

        for method_name in method_names:
            with self.subTest(method_name=method_name):
                predictor = _make_cached_predictor(latency_ms=0.013)
                latency = getattr(predictor, method_name)(batch)
                self.assertAlmostEqual(latency, 0.013)

    def test_small_decode_collective_uses_sglang_custom_allreduce(self):
        batch = SimpleNamespace(total_num_tokens=1, _total_num_tokens_rounded=8)
        method_names = (
            "get_execution_time",
            "get_execution_time_by_simai_analytical",
        )

        for method_name in method_names:
            with self.subTest(method_name=method_name):
                predictor = _make_cached_predictor(
                    latency_ms=0.036793,
                    custom_allreduce=True,
                )
                latency = getattr(predictor, method_name)(batch)
                self.assertAlmostEqual(latency, 0.0125)

    def test_large_collective_keeps_simai_latency(self):
        predictor = _make_cached_predictor(
            latency_ms=0.148723,
            custom_allreduce=True,
        )
        predictor.cache = {(8192, 2048, 2): 0.148723}
        batch = SimpleNamespace(
            total_num_tokens=2048,
            _total_num_tokens_rounded=2048,
        )

        latency = predictor.get_execution_time(batch)

        self.assertAlmostEqual(latency, 0.148723)


if __name__ == "__main__":
    unittest.main()
