import torch

from vidur.profiling.common.gpu_kernel_profiler import GPUKernelProfiler, PROFILE_METHOD
from vidur.profiling.mlp.mlp_impl import GPTModel


class MlpWrapper:
    def __init__(self, model_config, num_tensor_parallel_workers):
        self.config = model_config
        self.tp = num_tensor_parallel_workers
        self.model = GPTModel(model_config, self.tp).eval()
        self.profiler = GPUKernelProfiler()

    @torch.inference_mode()
    def profile(self, num_tokens):
        c = self.config
        stats = {name: self.profiler.measure(operation)
                 for name, operation in self.model.operations(num_tokens).items()}
        return {"time_stats": stats, "n_head": c.num_q_heads, "n_kv_head": c.num_kv_heads,
                "n_embd": c.embedding_dim, "n_expanded_embd": c.mlp_hidden_dim,
                "vocab_size": c.vocab_size, "use_gated_mlp": c.use_gated_mlp,
                "num_tokens": num_tokens, "num_tensor_parallel_workers": self.tp,
                "rope_theta": c.rope_theta, "rotary_dim": c.rotary_dim,
                "profile_method": PROFILE_METHOD, "vllm_version": self.profiler.vllm_version}
