from pathlib import Path
root=Path('work/moe_alignment_20261016/source/aicb')
p=root/'workload_generator/mocked_model/inference/qwen3_next_attention.py';s=p.read_text();s=s.replace('def benchmark_gdn_native_core(args, phase, tokens):','def _make_gdn_native_core(args, phase, tokens):',1)
needle='''    with torch.inference_mode(), override_forward_context(context):
        return benchmark_component(args, lambda: obj._forward_core(mixed,b,a,output))
'''
assert s.count(needle)==1;s=s.replace(needle,'''    return obj, context, (mixed, b, a, output)


def benchmark_gdn_native_core(args, phase, tokens):
    import torch
    from vllm.forward_context import override_forward_context
    obj, context, inputs = _make_gdn_native_core(args, phase, tokens)
    with torch.inference_mode(), override_forward_context(context):
        return benchmark_component(args, lambda: obj._forward_core(*inputs))


def benchmark_gdn_compiled_chain(args, tokens):
    """Prefill's two compiled regions around the native GDN custom op.

    Include QKV ordering/materialization, zeroing the output, native gated RMS
    norm, projections and both decoder norms. Isolated GEMMs omit this launch
    and allocation work. Compilation/warmup are outside the timed callback.
    """
    import torch
    from vllm.forward_context import override_forward_context
    from vllm.model_executor.layers.layernorm import RMSNormGated, GemmaRMSNorm

    class TupleLinear(torch.nn.Module):
        def __init__(self, inputs, outputs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(outputs, inputs, device="cuda", dtype=torch.bfloat16)*.02,
                                             requires_grad=False)
        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight), None

    obj, context, _ = _make_gdn_native_core(args, "prefill", tokens)
    shapes = projection_shapes(args)
    obj.in_proj_qkvz = TupleLinear(args.hidden_size, shapes.gdn_qkvz_size)
    obj.in_proj_ba = TupleLinear(args.hidden_size, shapes.gdn_ba_size)
    obj.out_proj = TupleLinear(shapes.gdn_value_size, args.hidden_size)
    obj.norm = RMSNormGated(args.linear_value_head_dim, eps=args.rms_norm_eps,
                           group_size=None, norm_before_gate=True,
                           device="cuda", dtype=torch.bfloat16)
    # vLLM compiles the native expression into its surrounding FX region.
    # The FLA autograd wrapper is a different path and cannot be traced here.
    obj.norm._forward_method = obj.norm.forward_native
    pre = GemmaRMSNorm(args.hidden_size, args.rms_norm_eps).to(device="cuda", dtype=torch.bfloat16)
    post = GemmaRMSNorm(args.hidden_size, args.rms_norm_eps).to(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(tokens, args.hidden_size, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    heads = args.linear_num_value_heads // args.tensor_model_parallel_size

    def before_core():
        hidden, res = pre.forward_native(x, residual)
        qkv, _ = obj.in_proj_qkvz(hidden)
        ba, _ = obj.in_proj_ba(hidden)
        q, k, v, z, b, a = obj.fix_query_key_value_ordering(qkv, ba)
        mixed = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1)
        core = torch.zeros(tokens, heads, args.linear_value_head_dim, device="cuda", dtype=torch.bfloat16)
        return mixed, b, a, core, z, res

    def after_core(core, z, res):
        attn = obj.norm(core.reshape(-1, args.linear_value_head_dim),
                        z.reshape(-1, args.linear_value_head_dim)).reshape(tokens, shapes.gdn_value_size)
        out, _ = obj.out_proj(attn)
        return post.forward_native(out, res)

    with torch.inference_mode(), override_forward_context(context):
        before = torch.compile(before_core, fullgraph=True, dynamic=True)
        after = torch.compile(after_core, fullgraph=True, dynamic=True)
        def run():
            mixed, b, a, core, z, res = before()
            torch.ops.vllm.gdn_attention_core(mixed, b, a, core, obj.prefix)
            return after(core, z, res)
        return benchmark_component(args, run)
''');p.write_text(s)
p=root/'workload_generator/mocked_model/inference/AiobQwen3Next.py';s=p.read_text().replace('benchmark_flash_attention, benchmark_gdn_native_core,','benchmark_flash_attention, benchmark_gdn_native_core, benchmark_gdn_compiled_chain,')
needle='''        # 1. qkv
        qkv_quant = self._attn_quant(True, m)
''';pos=s.index(needle);s=s[:pos]+'''        if phase == InferencePhase.PREFILL.value:
            return 0.0, 0.0, benchmark_gdn_compiled_chain(self.args, m), 0.0

'''+s[pos:];p.write_text(s)
p=root/'workload_generator/Vidur_workload_generator.py';s=p.read_text();needle='''        # Comm part
        tp_comm_size, ep_dispatch_size, ep_combine_size = self.get_comm_size()
''';repl='''        if self.args.model_name == "Qwen3-Next-80B" and self.args.phase == "prefill":
            # The compiled GDN chain already includes both decoder norms.
            for row in self.vidur_layers:
                is_gdn = (row[0]+1) % self.args.full_attention_interval != 0
                if is_gdn:
                    if row[1].value == "attention":
                        row[2] -= self.compute_cache["attention_norm"]
                    elif row[1].value == "moe":
                        row[2] -= self.compute_cache["moe_norm"]

'''+needle;assert s.count(needle)==1;s=s.replace(needle,repl);p.write_text(s)
