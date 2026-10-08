"""Native GPU operators used by the single MLP profiling implementation."""
from types import SimpleNamespace

import torch
import torch.nn.functional as F


class GPTModel(torch.nn.Module):
    def __init__(self, config, world_size):
        super().__init__()
        from vllm.model_executor.layers.activation import SiluAndMul
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.layers.rotary_embedding import get_rope

        config.validate_parallelism(world_size)
        self.config = config
        head = config.get_head_size()
        parallel = SimpleNamespace(tensor_parallel_size=world_size)
        self.q_size = config.get_num_q_heads(parallel) * head
        self.kv_size = config.get_num_kv_heads(parallel) * head
        self.up_size = config.mlp_hidden_dim // world_size
        self.vocab_size = (config.vocab_size + world_size - 1) // world_size
        def weight(shape):
            return torch.nn.Parameter(torch.randn(shape, dtype=torch.float16, device="cuda") * 0.01,
                                      requires_grad=False)
        self.embedding = weight((self.vocab_size, config.embedding_dim))
        self.qkv = weight((self.q_size + 2*self.kv_size, config.embedding_dim))
        self.out = weight((config.embedding_dim, self.q_size))
        self.up = weight((self.up_size * (2 if config.use_gated_mlp else 1), config.embedding_dim))
        self.down = weight((config.embedding_dim, self.up_size))
        self.qkv_bias = weight((self.qkv.shape[0],)) if config.use_bias or config.use_qkv_bias else None
        self.out_bias = weight((config.embedding_dim,)) if config.use_bias else None
        self.up_bias = weight((self.up.shape[0],)) if config.use_bias else None
        self.down_bias = weight((config.embedding_dim,)) if config.use_bias else None
        norm = lambda: (RMSNorm(config.embedding_dim) if config.norm == "rms_norm" else
                        torch.nn.LayerNorm(config.embedding_dim)).to(device="cuda", dtype=torch.float16)
        self.input_norm = norm()
        self.post_norm = norm() if config.post_attn_norm else None
        self.activation = SiluAndMul() if config.use_gated_mlp else torch.nn.GELU()
        self.rotary = None
        if config.rope_theta is not None:
            self.rotary = get_rope(head, config.rotary_dim, config.max_position_embeddings,
                config.rope_theta, is_neox_style=config.is_neox_style,
                rope_scaling=config.rope_scaling, dtype=torch.float16).cuda()

    def operations(self, num_tokens):
        c = self.config
        x = torch.randn((num_tokens, c.embedding_dim), dtype=torch.float16, device="cuda")
        ids = torch.randint(self.vocab_size, (num_tokens,), dtype=torch.long, device="cuda")
        positions = torch.arange(num_tokens, device="cuda") % c.max_position_embeddings
        qkv = F.linear(x, self.qkv, self.qkv_bias)
        q, k, _ = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        projected = torch.randn((num_tokens, self.q_size), dtype=x.dtype, device=x.device)
        up = F.linear(x, self.up, self.up_bias)
        activated = self.activation(up)
        residual = torch.randn_like(x)
        return {"emb": lambda: F.embedding(ids, self.embedding),
                "input_layernorm": lambda: self.input_norm(x),
                "attn_pre_proj": lambda: F.linear(x, self.qkv, self.qkv_bias),
                "attn_rope": (lambda: self.rotary(positions, q, k)) if self.rotary is not None else None,
                "attn_post_proj": lambda: F.linear(projected, self.out, self.out_bias),
                "post_attention_layernorm": (lambda: self.post_norm(x)) if self.post_norm is not None else None,
                "mlp_up_proj": lambda: F.linear(x, self.up, self.up_bias),
                "mlp_act": lambda: self.activation(up),
                "mlp_down_proj": lambda: F.linear(activated, self.down, self.down_bias),
                "add": lambda: x + residual}
