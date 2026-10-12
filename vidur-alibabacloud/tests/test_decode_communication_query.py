"""Regression tests for phase-aware tensor-parallel communication queries.

This test extracts the production predictor method from its source file so it
can run without importing the simulator, CUDA, torch, or Ray.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
PREDICTOR_PATH = (
    ROOT
    / "vidur"
    / "execution_time_predictor"
    / "sklearn_execution_time_predictor.py"
)


def extract_communication_method():
    tree = ast.parse(PREDICTOR_PATH.read_text(), filename=str(PREDICTOR_PATH))
    predictor_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "SklearnExecutionTimePredictor"
    )
    method = next(
        node
        for node in predictor_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_get_tensor_parallel_communication_time"
    )
    method.decorator_list = []
    env = {"Batch": object}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(PREDICTOR_PATH), "exec"), env)
    return env[method.name]


class PredictionTable:
    def __init__(self, stage):
        self.stage = stage
        self.queries = []

    def __getitem__(self, key):
        self.queries.append(key)
        return 100.0 * self.stage + float(key[0])


class DecodeCommunicationQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predict = staticmethod(extract_communication_method())

    def make_harness(self, stages=(0, 1, 2)):
        tables = {
            "all_reduce_stage_{}".format(stage): PredictionTable(stage)
            for stage in stages
        }
        selected_models = []

        def communication_model_name(model_name, pipeline_stage):
            name = "{}_stage_{}".format(model_name, pipeline_stage)
            selected_models.append(name)
            return name

        harness = SimpleNamespace(
            _predictions=tables,
            _config=SimpleNamespace(
                nccl_cpu_launch_overhead_ms=1.25,
                nccl_cpu_skew_overhead_per_device_ms=0.5,
            ),
            _replica_config=SimpleNamespace(tensor_parallel_size=4),
            _communication_model_name=communication_model_name,
        )
        return harness, tables, selected_models

    @staticmethod
    def make_batch(num_tokens, prefill_tokens=0, decode_tokens=None):
        total_tokens = sum(num_tokens)
        if decode_tokens is None:
            decode_tokens = total_tokens - prefill_tokens
        return SimpleNamespace(
            num_tokens=list(num_tokens),
            num_prefill_tokens=prefill_tokens,
            num_decode_tokens=decode_tokens,
            _total_num_tokens_rounded=((total_tokens + 7) // 8) * 8,
        )

    def call_and_check_unchanged(self, harness, batch, pipeline_stage):
        original_tokens = list(batch.num_tokens)
        original_rounded = batch._total_num_tokens_rounded
        result = self.predict(harness, batch, pipeline_stage)
        self.assertEqual(batch.num_tokens, original_tokens)
        self.assertEqual(batch._total_num_tokens_rounded, original_rounded)
        return result

    def test_pure_decode_uses_cuda_graph_token_boundaries(self):
        cases = (
            (1, 1),
            (2, 2),
            (3, 4),
            (4, 4),
            (5, 8),
            (6, 8),
            (7, 8),
            (8, 8),
            (9, 16),
            (16, 16),
            (17, 24),
            (24, 24),
            (25, 32),
        )
        for token_count, expected_query in cases:
            with self.subTest(token_count=token_count):
                harness, tables, selected = self.make_harness()
                batch = self.make_batch([1] * token_count, prefill_tokens=0)
                result = self.call_and_check_unchanged(harness, batch, 0)
                self.assertEqual(selected, ["all_reduce_stage_0"])
                self.assertEqual(tables["all_reduce_stage_0"].queries, [(expected_query,)])
                expected = expected_query + 1.25 + 0.5 * (4 ** 1.25)
                self.assertAlmostEqual(result, expected)

    def test_prefill_keeps_rounded_query(self):
        for token_count, expected_query in ((44, 48), (1, 8)):
            with self.subTest(token_count=token_count):
                harness, tables, selected = self.make_harness()
                batch = self.make_batch(
                    [token_count], prefill_tokens=token_count, decode_tokens=0
                )
                self.call_and_check_unchanged(harness, batch, 1)
                self.assertEqual(selected, ["all_reduce_stage_1"])
                self.assertEqual(tables["all_reduce_stage_1"].queries, [(expected_query,)])

    def test_mixed_batch_keeps_rounded_query(self):
        harness, tables, selected = self.make_harness()
        batch = self.make_batch([43, 1], prefill_tokens=43, decode_tokens=1)
        self.call_and_check_unchanged(harness, batch, 2)
        self.assertEqual(selected, ["all_reduce_stage_2"])
        self.assertEqual(tables["all_reduce_stage_2"].queries, [(48,)])

    def test_pipeline_stage_selects_its_own_prediction_cache(self):
        harness, tables, selected = self.make_harness(stages=(0, 2))
        batch = self.make_batch([1, 1, 1], prefill_tokens=0)
        stage_zero = self.call_and_check_unchanged(harness, batch, 0)
        stage_two = self.call_and_check_unchanged(harness, batch, 2)
        self.assertEqual(selected, ["all_reduce_stage_0", "all_reduce_stage_2"])
        self.assertEqual(tables["all_reduce_stage_0"].queries, [(4,)])
        self.assertEqual(tables["all_reduce_stage_2"].queries, [(4,)])
        self.assertAlmostEqual(stage_two - stage_zero, 200.0)


if __name__ == "__main__":
    unittest.main()
