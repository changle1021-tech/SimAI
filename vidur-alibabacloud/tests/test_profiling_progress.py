import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vidur.profiling.attention import main as attention
from vidur.profiling.mlp import main as mlp
from vidur.profiling.common.progress import collect_profile_results


class Progress:
    def __init__(self):
        self.n = 0
        self.labels = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, count):
        self.n += count

    def set_description_str(self, label):
        self.labels.append(label)

    def set_postfix_str(self, label, **kwargs):
        self.labels.append(label)


class ProgressTests(unittest.TestCase):
    def test_ray_reports_completed_tasks_and_preserves_csv_order(self):
        progress = Progress()
        completed = []

        class Ray:
            def wait(self, pending, num_returns):
                self_check.assertEqual(progress.n, len(completed))
                ref = pending[-1]
                return [ref], pending[:-1]

            def get(self, ref):
                self_check.assertEqual(progress.n, len(completed))
                completed.append(ref)
                return ref * 10

        self_check = self
        rows = collect_profile_results(Ray(), [1, 2, 3], [16, 32, 64], progress, lambda n: f"tokens={n}")
        self.assertEqual(completed, [3, 2, 1])
        self.assertEqual(rows, [10, 20, 30])
        self.assertEqual(progress.n, 3)
        self.assertEqual(progress.labels[0], "done tokens=64")

    def test_failed_ray_result_does_not_advance_progress(self):
        progress = Progress()
        ray = SimpleNamespace(wait=lambda pending, num_returns: (pending, []))
        def fail(ref):
            raise RuntimeError("worker failed")
        ray.get = fail
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            collect_profile_results(ray, [1], [16], progress, str)
        self.assertEqual(progress.n, 0)

    def test_mlp_total_excludes_unsupported_tp_and_advances_after_profile(self):
        progress = Progress()
        calls = []
        class Worker:
            def __init__(self, config, tp):
                self.tp = tp

            def profile(self, tokens):
                self_check.assertEqual(progress.n, len(calls))
                calls.append((self.tp, tokens))
                return {"tokens": tokens}

        self_check = self
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(disable_ray=True, num_gpus=1, max_tokens=2, output_dir=tmp,
                                   models=["microsoft/phi-2", "meta-llama/Llama-2-7b-hf"],
                                   num_tensor_parallel_workers=[1, 2])
            with patch.object(mlp, "parse_args", return_value=args), \
                 patch.object(mlp, "get_num_tokens_to_profile", return_value=[2, 1]), \
                 patch.object(mlp, "MlpWrapper", Worker), \
                 patch.object(mlp.torch.cuda, "empty_cache"), \
                 patch.object(mlp, "tqdm", return_value=progress) as bar:
                mlp.main()
                self.assertEqual(bar.call_args.kwargs["total"], 6)
        self.assertEqual(progress.n, 6)
        self.assertEqual(len(calls), 6)

    def test_attention_total_uses_memory_filtered_shapes(self):
        progress = Progress()
        calls = []
        def item(size):
            return SimpleNamespace(is_prefill=False, batch_size=1, prefill_chunk_size=0,
                kv_cache_size=size, is_valid=lambda limit: True,
                is_under_memory_limit=lambda capacity: size <= capacity)
        inputs = [item(8), item(32)]

        class Worker:
            def __init__(self, *args):
                pass

            def profile(self, shape):
                self_check.assertEqual(progress.n, len(calls))
                calls.append(shape.kv_cache_size)
                return {"kv_cache_size": shape.kv_cache_size}

        self_check = self
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(disable_ray=True, num_gpus=1, output_dir=tmp,
                models=["meta-llama/Llama-2-7b-hf"], num_tensor_parallel_workers=[1, 2],
                max_seq_len=64, max_model_len=64, min_batch_size=1, max_batch_size=1,
                block_size=16, profile_only_prefill=False, profile_only_decode=False)
            with patch.object(attention, "parse_args", return_value=args), \
                 patch.object(attention, "get_attention_input_combinations", return_value=inputs), \
                 patch.object(attention, "get_max_num_blocks", return_value=1), \
                 patch.object(attention, "AttentionWrapper", Worker), \
                 patch.object(attention.torch.cuda, "empty_cache"), \
                 patch.object(attention, "tqdm", return_value=progress) as bar:
                attention.main()
                self.assertEqual(bar.call_args.kwargs["total"], 2)
        self.assertEqual(progress.n, 2)
        self.assertEqual(calls, [8, 8])


if __name__ == "__main__":
    unittest.main()
