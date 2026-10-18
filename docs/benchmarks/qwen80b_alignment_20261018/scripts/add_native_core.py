from pathlib import Path
p=Path('work/moe_alignment_20261016/source/aicb/workload_generator/mocked_model/inference/qwen3_next_attention.py')
s=p.read_text();s+='''

def benchmark_gdn_native_core(args, phase, tokens):
    """Profile vLLM's complete GDN custom op, including cache state copies.

    The kernel-only chunk helper omits Q/K/V contiguous copies, initial-state
    gathering/zeroing, cache writeback and output copies. Prefill executes this
    Python custom-op body eagerly; decode captures the identical body in Graph.
    """
    import torch
    from vllm.model_executor.models.qwen3_next import Qwen3NextGatedDeltaNet
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.forward_context import ForwardContext, override_forward_context

    _ensure_triton_allocator()
    tp = args.tensor_model_parallel_size
    obj = Qwen3NextGatedDeltaNet.__new__(Qwen3NextGatedDeltaNet)
    torch.nn.Module.__init__(obj)
    obj.tp_size = tp
    obj.num_k_heads, obj.num_v_heads = args.linear_num_key_heads, args.linear_num_value_heads
    obj.head_k_dim, obj.head_v_dim = args.linear_key_head_dim, args.linear_value_head_dim
    obj.key_dim = obj.num_k_heads * obj.head_k_dim
    obj.value_dim = obj.num_v_heads * obj.head_v_dim
    obj.prefix = "aicb.gdn"
    obj.activation = "silu"
    dim = gdn_conv_dim(args)
    obj.conv1d = torch.nn.Conv1d(dim, dim, args.linear_conv_kernel_dim,
                               groups=dim, bias=False, device="cuda", dtype=torch.bfloat16)
    heads = obj.num_v_heads // tp
    obj.A_log = torch.nn.Parameter(torch.zeros(heads, device="cuda", dtype=torch.float32))
    obj.dt_bias = torch.nn.Parameter(torch.ones(heads, device="cuda", dtype=torch.float32))
    sequences = tokens if phase == "decode" else 1
    obj.kv_cache = [(torch.zeros(sequences, args.linear_conv_kernel_dim-1, dim,
                                device="cuda", dtype=torch.bfloat16),
                     torch.zeros(sequences, heads, obj.head_k_dim, obj.head_v_dim,
                                 device="cuda", dtype=torch.bfloat16))]
    conv = conv_prefill_metadata(tokens)
    metadata = GDNAttentionMetadata(
        num_prefills=int(phase == "prefill"), num_prefill_tokens=tokens if phase == "prefill" else 0,
        num_decodes=sequences if phase == "decode" else 0,
        num_decode_tokens=tokens if phase == "decode" else 0,
        num_spec_decodes=0, num_spec_decode_tokens=0, num_actual_tokens=tokens,
        has_initial_state=torch.full((sequences,), bool(getattr(args,"context_length",0)), device="cuda",dtype=torch.bool),
        non_spec_query_start_loc=(torch.arange(tokens+1,device="cuda",dtype=torch.int32) if phase=="decode"
                                  else torch.tensor([0,tokens],device="cuda",dtype=torch.int32)),
        non_spec_state_indices_tensor=torch.arange(sequences,device="cuda",dtype=torch.int32),
        nums_dict=conv.nums_dict, batch_ptr=conv.batch_ptr,
        token_chunk_offset_ptr=conv.token_chunk_offset_ptr)
    context = ForwardContext(no_compile_layers={obj.prefix:obj},
                             attn_metadata={obj.prefix:metadata},virtual_engine=0)
    mixed = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(tokens, heads, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    output = torch.zeros(tokens,heads,obj.head_v_dim,device="cuda",dtype=torch.bfloat16)
    with torch.inference_mode(), override_forward_context(context):
        return benchmark_component(args, lambda: obj._forward_core(mixed,b,a,output))
''';p.write_text(s)
p=Path('work/moe_alignment_20261016/source/aicb/workload_generator/mocked_model/inference/AiobQwen3Next.py');s=p.read_text().replace('benchmark_gdn_conv, benchmark_gdn_core, benchmark_kv_cache_save, benchmark_flash_attention,','benchmark_gdn_conv, benchmark_gdn_core, benchmark_kv_cache_save, benchmark_flash_attention, benchmark_gdn_native_core,')
s=s.replace('attn_causal_conv = self._causal_conv(m)','attn_causal_conv = 0.0  # included in the native custom-op region')
s=s.replace('gdn_gating = self._fused_gdn_gating(m)','gdn_gating = 0.0  # native custom op computes both g and beta')
s=s.replace('gdn_core = self._fused_gdn_core()','gdn_core = benchmark_gdn_native_core(self.args, phase, m)')
p.write_text(s)
