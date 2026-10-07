"""Deterministic tests for the explicit async execution profile input.

These tests validate CSV selection and interpolation behavior only. They do
not represent a measured performance acceptance test.
"""
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


IMPLEMENTATION = Path(__file__).parents[1] / 'vidur' / 'execution_time_predictor' / 'async_execution_profile.py'
SPEC = importlib.util.spec_from_file_location('async_execution_profile_under_test', IMPLEMENTATION)
PROFILE_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE_MODULE)
AsyncExecutionProfile = PROFILE_MODULE.AsyncExecutionProfile
COMPONENTS = PROFILE_MODULE.COMPONENTS
SCOPE = PROFILE_MODULE.SCOPE


MODEL = SimpleNamespace(
    num_layers=60, embedding_dim=5120, mlp_hidden_dim=13824,
    num_q_heads=40, num_kv_heads=8, vocab_size=92544,
    get_name=lambda: 'fixture/model',
)
REPLICA_TP4 = SimpleNamespace(tensor_parallel_size=4, num_pipeline_stages=1)


def make_row(phase='decode', batch_size=1, axis=10, costs=None, **overrides):
    # Six separate measurements: model GPU plus five host-side wall costs.
    if costs is None:
        costs = {
            'schedule_ms': 1.0 + axis / 100,
            'input_prefix_ms': 2.0 + axis / 100,
            'model_gpu_ms': 3.0 + axis / 10,
            'post_model_sampler_ms': 4.0 + axis / 100,
            'executor_return_ms': 5.0 + axis / 100,
            'output_processing_ms': 6.0 + axis / 100,
        }
    row = {
        'timing_scope': SCOPE,
        'profile_group': 'fixture-v1',
        'model_name': 'fixture/model',
        'num_layers': 60,
        'n_embd': 5120,
        'n_expanded_embd': 13824,
        'n_q_head': 40,
        'n_kv_head': 8,
        'vocab_size': 92544,
        'num_tensor_parallel_workers': 4,
        'num_pipeline_stages': 1,
        'phase': phase,
        'batch_size': batch_size,
        'prefill_tokens': axis if phase == 'prefill' else 0,
        'sequence_length': axis if phase == 'decode' else axis / batch_size,
        'forward_tokens': batch_size if phase == 'decode' else axis,
        'cuda_graph': 'true' if phase == 'decode' else 'false',
        **costs,
    }
    row.update(overrides)
    return row


class ProfileFixture(unittest.TestCase):
    def write_profile(self, rows):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / 'profile.csv'
        fields = list(rows[0])
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def load(self, rows, model=MODEL, replica=REPLICA_TP4):
        return AsyncExecutionProfile(self.write_profile(rows), model, replica)

    @staticmethod
    def decode_batch(seq_lengths, rounded8=999):
        size = len(seq_lengths)
        return SimpleNamespace(
            size=size, num_prefill_tokens=0, num_decode_tokens=size,
            requests=[SimpleNamespace(num_processed_tokens=x) for x in seq_lengths],
            _total_num_tokens_rounded8=rounded8,
        )

    def test_duplicate_shape_samples_use_mean_for_each_component(self):
        first = {c: i + 1.0 for i, c in enumerate(COMPONENTS)}
        second = {c: i + 3.0 for i, c in enumerate(COMPONENTS)}
        profile = self.load([
            make_row(axis=10, costs=first), make_row(axis=10, costs=second),
        ])
        prediction = profile.predict(self.decode_batch([10]))
        for i, component in enumerate(COMPONENTS):
            self.assertEqual(prediction[component], i + 2.0)

    def test_interpolates_within_measured_range_and_ignores_rounded8(self):
        low = {c: float(i + 1) for i, c in enumerate(COMPONENTS)}
        high = {c: float(i + 11) for i, c in enumerate(COMPONENTS)}
        decoy = {c: float(i + 101) for i, c in enumerate(COMPONENTS)}
        profile = self.load([
            make_row(axis=10, costs=low), make_row(axis=20, costs=high),
            make_row(axis=64, costs=decoy),
        ])
        prediction = profile.predict(self.decode_batch([15], rounded8=64))
        for i, component in enumerate(COMPONENTS):
            self.assertEqual(prediction[component], i + 6.0)

    def test_unmeasured_batch_size_fails(self):
        profile = self.load([make_row(batch_size=1, axis=10)])
        with self.assertRaisesRegex(ValueError, 'Unmeasured phase/batch size'):
            profile.predict(self.decode_batch([10, 10]))

    def test_outside_sequence_range_fails(self):
        profile = self.load([make_row(axis=10), make_row(axis=20)])
        with self.assertRaisesRegex(ValueError, 'outside measured profile range'):
            profile.predict(self.decode_batch([21]))

    def test_chunked_or_cached_prefill_fails(self):
        profile = self.load([make_row(phase='prefill', axis=64)])
        batch = SimpleNamespace(size=1, num_prefill_tokens=64, num_decode_tokens=0,
                                requests=[SimpleNamespace(num_processed_tokens=16)])
        with self.assertRaisesRegex(ValueError, 'cached/chunked prefill'):
            profile.predict(batch)

    def test_mixed_prefill_decode_batch_fails(self):
        profile = self.load([make_row(axis=10)])
        batch = SimpleNamespace(size=1, num_prefill_tokens=10, num_decode_tokens=1,
                                requests=[SimpleNamespace(num_processed_tokens=10)])
        with self.assertRaisesRegex(ValueError, 'mixed prefill/decode'):
            profile.predict(batch)

    def test_model_structure_mismatch_fails(self):
        wrong_model = SimpleNamespace(**vars(MODEL))
        wrong_model.num_layers = 32
        with self.assertRaisesRegex(ValueError, 'No measured profile matches'):
            self.load([make_row(axis=10)], model=wrong_model)

    def test_tensor_parallel_mismatch_fails(self):
        replica_tp2 = SimpleNamespace(tensor_parallel_size=2, num_pipeline_stages=1)
        with self.assertRaisesRegex(ValueError, 'No measured profile matches'):
            self.load([make_row(axis=10)], replica=replica_tp2)

    def test_negative_component_fails(self):
        costs = {c: i + 1.0 for i, c in enumerate(COMPONENTS)}
        costs['schedule_ms'] = -0.1
        with self.assertRaisesRegex(ValueError, 'Negative/nonfinite'):
            self.load([make_row(axis=10, costs=costs)])

    def test_mixed_scope_fails(self):
        rows = [make_row(axis=10), make_row(axis=20, timing_scope='other_scope')]
        with self.assertRaisesRegex(ValueError, 'Mixed timing scopes'):
            self.load(rows)


if __name__ == '__main__':
    unittest.main()
