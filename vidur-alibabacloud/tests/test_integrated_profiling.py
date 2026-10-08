import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from vidur.config import RandomForrestExecutionTimePredictorConfig
from vidur.execution_time_predictor.sklearn_execution_time_predictor import SklearnExecutionTimePredictor as Predictor
from vidur.profiling.attention.main import DEFAULT_MODELS as ATTENTION_MODELS, parse_args as attention_args
from vidur.profiling.mlp.main import DEFAULT_MODELS as MLP_MODELS, parse_args as mlp_args
from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.common.gpu_kernel_profiler import graph_batch_size, kernel_time_ms


class IntegratedProfilingTests(unittest.TestCase):
    def test_original_model_families_and_tp_shapes(self):
        self.assertEqual(set(ATTENTION_MODELS), set(MLP_MODELS))
        self.assertEqual(len(MLP_MODELS), 8)
        for name in MLP_MODELS:
            model = ModelConfig.from_model_name(name)
            for tp in ([1] if model.no_tensor_parallel else [1, 2, 4, 8]):
                model.validate_parallelism(tp)
                parallel = SimpleNamespace(tensor_parallel_size=tp)
                self.assertGreater(model.get_num_q_heads(parallel), 0)
                self.assertGreater(model.get_num_kv_heads(parallel), 0)
        phi = ModelConfig.from_model_name("microsoft/phi-2")
        self.assertEqual(phi.rotary_dim, 32)
        self.assertEqual(phi.norm, "layer_norm")
        self.assertFalse(phi.use_gated_mlp)
        with self.assertRaises(ValueError):
            phi.validate_parallelism(2)
        gqa = ModelConfig.from_model_name("meta-llama/Meta-Llama-3-8B")
        self.assertEqual(gqa.get_num_kv_heads(SimpleNamespace(tensor_parallel_size=16)), 1)

    def test_no_legacy_switches_or_separate_profile_parameters(self):
        a, m = attention_args([]), mlp_args([])
        self.assertFalse(hasattr(a, "attention_backend"))
        self.assertFalse(hasattr(m, "profile_method"))
        config = RandomForrestExecutionTimePredictorConfig()
        for key in ("vllm_attention_profile_file", "vllm_attention_decode_capture_limit", "skip_rope_modeling"):
            self.assertFalse(hasattr(config, key))

    def test_kernel_sum_excludes_parents_and_conditioning(self):
        def event(name, device, value):
            return SimpleNamespace(name=name, device_type=device, self_cuda_time_total=value)
        events = [event("aten::linear", "DeviceType.CPU", 200),
                  event("gemm", "DeviceType.CUDA", 100),
                  event("cache_read_reduce", "DeviceType.CUDA", 80),
                  event("FillFunctor<long>", "DeviceType.CUDA", 5),
                  event("bias", "DeviceType.CUDA", 10)]
        self.assertAlmostEqual(kernel_time_ms(events, {"cache_read_reduce"}), 0.110)

    def harness(self):
        model = SimpleNamespace(embedding_dim=4096, num_q_heads=32, num_kv_heads=32)
        return SimpleNamespace(_model_config=model, _replica_config=SimpleNamespace(tensor_parallel_size=1),
                               _block_size=16)

    def frame(self, **changes):
        row = {"n_embd": 4096, "n_q_head": 32, "n_kv_head": 32, "block_size": 16,
               "num_tensor_parallel_workers": 1, "max_model_len": 4096, "batch_size": 3,
               "prefill_chunk_size": 0, "kv_cache_size": 512, "forward_tokens": 4,
               "block_table_width": 256, "cuda_graph": True,
               "profile_method": "vllm_cuda_graph_kernel_sum_v2",
               "time_stats.attn_kv_cache_save.median": 0.003}
        row.update(changes)
        return pd.DataFrame([row])

    def load(self, frame):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attention.csv"
            frame.to_csv(path, index=False)
            return Predictor._load_attention_df(self.harness(), path)

    def test_standard_csv_requires_real_kv_cost_and_graph_shape(self):
        self.assertEqual(len(self.load(self.frame())), 1)
        for value in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                self.load(self.frame(**{"time_stats.attn_kv_cache_save.median": value}))
        with self.assertRaises(ValueError):
            self.load(self.frame().drop(columns=["time_stats.attn_kv_cache_save.median"]))
        for changes in (dict(forward_tokens=3), dict(block_table_width=32), dict(cuda_graph=False)):
            with self.assertRaises(ValueError):
                self.load(self.frame(**changes))

    def test_cache_training_uses_padded_tokens(self):
        derived = Predictor._get_attention_df_with_derived_features(self.harness(), self.frame())
        self.assertEqual(derived["num_tokens"].iloc[0], 4)

    def test_projection_rope_and_kv_lookup_use_same_native_shape(self):
        p = SimpleNamespace(_config=SimpleNamespace(backend="vidur"))
        p._compute_tokens = lambda batch: Predictor._compute_tokens(p, batch)
        p._predictions = {k: {(4,): 0.004} for k in ("attn_pre_proj", "attn_rope", "attn_kv_cache_save")}
        b = SimpleNamespace(num_tokens=[1, 1, 1], num_prefill_tokens=0)
        self.assertEqual(p._compute_tokens(b), graph_batch_size(3))
        for method in (Predictor._get_attention_layer_pre_proj_execution_time,
                       Predictor._get_attention_rope_execution_time,
                       Predictor._get_attention_kv_cache_save_execution_time):
            self.assertEqual(method(p, b), 0.004)
        b.num_tokens, b.num_prefill_tokens = [513], 513
        self.assertEqual(p._compute_tokens(b), 513)


if __name__ == "__main__":
    unittest.main()
