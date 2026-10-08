"""Native paged attention and KV-write profiling for the existing model registry."""
import math

import torch

from vidur.profiling.common.gpu_kernel_profiler import GPUKernelProfiler, PROFILE_METHOD, graph_batch_size


class AttentionWrapper:
    def __init__(self, model_config, parallel_config, max_num_blocks, max_model_len,
                 block_size, dtype=torch.float16):
        self.model = model_config
        self.parallel = parallel_config
        self.max_num_blocks = max_num_blocks
        self.max_model_len = max_model_len
        self.block_size = block_size
        self.dtype = dtype
        self.q_heads = model_config.get_num_q_heads(parallel_config)
        self.kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head = model_config.get_head_size()
        self.profiler = GPUKernelProfiler()

    @torch.inference_mode()
    def profile(self, attention_input):
        from vllm import _custom_ops as ops
        from vllm.attention.backends.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

        item = attention_input
        if not item.is_valid(self.max_model_len):
            raise ValueError("Attention input exceeds the model/capture limit")
        bs, past = item.batch_size, item.kv_cache_size
        query_len = item.prefill_chunk_size if item.is_prefill else 1
        total = past + query_len
        tokens = bs * query_len if item.is_prefill else graph_batch_size(bs)
        blocks_per_seq = math.ceil(total / self.block_size)
        num_blocks = bs * blocks_per_seq
        if num_blocks > self.max_num_blocks:
            raise ValueError("Attention input exceeds available KV blocks")
        packed = torch.randn((tokens, (self.q_heads + 2 * self.kv_heads) * self.head),
                             dtype=self.dtype, device="cuda")
        q, k, v = packed.split([self.q_heads*self.head, self.kv_heads*self.head,
                               self.kv_heads*self.head], dim=-1)
        q = q.view(tokens, self.q_heads, self.head)
        k = k.view(tokens, self.kv_heads, self.head)
        v = v.view(tokens, self.kv_heads, self.head)
        cache = torch.randn((2, num_blocks, self.block_size, self.kv_heads, self.head),
                            dtype=self.dtype, device="cuda")
        width = blocks_per_seq if item.is_prefill else math.ceil(self.max_model_len / self.block_size)
        table = torch.zeros((bs if item.is_prefill else tokens, width), dtype=torch.int32, device="cuda")
        table[:bs, :blocks_per_seq] = torch.arange(num_blocks, dtype=torch.int32, device="cuda").view(bs, blocks_per_seq)
        slots = [i*blocks_per_seq*self.block_size + past+j for i in range(bs) for j in range(query_len)]
        slots += [-1] * (tokens - len(slots))
        slots = torch.tensor(slots, dtype=torch.long, device="cuda")
        seq = torch.tensor([total]*bs + ([1]*(tokens-bs) if not item.is_prefill else []),
                           dtype=torch.int32, device="cuda")
        cu_q = torch.arange(bs+1, dtype=torch.int32, device="cuda") * query_len
        cu_k = torch.arange(bs+1, dtype=torch.int32, device="cuda") * total

        def save_cache():
            ops.reshape_and_cache_flash(k, v, cache[0], cache[1], slots, "auto")

        save_cache()

        def attention():
            if not item.is_prefill:
                return flash_attn_with_kvcache(q.unsqueeze(1), cache[0], cache[1],
                    block_table=table, cache_seqlens=seq, softmax_scale=self.head**-0.5, causal=True)
            if past:
                return flash_attn_varlen_func(q=q, k=cache[0], v=cache[1],
                    cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=query_len,
                    max_seqlen_k=total, block_table=table, softmax_scale=self.head**-0.5, causal=True)
            return flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                max_seqlen_q=query_len, max_seqlen_k=total, softmax_scale=self.head**-0.5, causal=True)

        stats = {"attn_kv_cache_save": self.profiler.measure(save_cache)}
        stats["attn_prefill" if item.is_prefill else "attn_decode"] = self.profiler.measure(attention)
        return {"time_stats": stats, "n_embd": self.model.embedding_dim,
                "n_q_head": self.model.num_q_heads, "n_kv_head": self.model.num_kv_heads,
                "block_size": self.block_size, "num_tensor_parallel_workers": self.parallel.tensor_parallel_size,
                "max_model_len": self.max_model_len, "batch_size": bs,
                "prefill_chunk_size": item.prefill_chunk_size, "kv_cache_size": past,
                "is_prefill": item.is_prefill, "attention_backend": "vllm_flash_attention",
                "profile_method": PROFILE_METHOD, "vllm_version": self.profiler.vllm_version, "forward_tokens": tokens,
                "block_table_width": width, "cuda_graph": not item.is_prefill}
