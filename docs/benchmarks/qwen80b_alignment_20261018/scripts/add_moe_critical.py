from pathlib import Path
p=Path('work/moe_alignment_20261016/source/aicb/workload_generator/mocked_model/inference/qwen3_next_moe.py')
s=p.read_text();s+='''

def benchmark_moe_critical_compute(args, local_tokens, route_ids, ep_rank, samples=12):
    """Replay one layer's measured routes and native shared-stream dependency.

    Returns compute seconds for this EP shard. Communication is measured by a
    separate four-rank AG/RS benchmark; max across EP shards is taken per layer,
    then samples are averaged. Never sum the four ranks' elapsed times.
    """
    import torch
    import torch.nn.functional as F
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts, fused_topk
    from workload_generator.mocked_model.inference.qwen3_next_projections import benchmark_gpu_region

    dp, ep, local_experts = moe_parallel_layout(args)
    if args.tensor_model_parallel_size != 1:
        raise ValueError("Measured MoE critical compute currently requires TP1")
    tokens = gathered_tokens(local_tokens, dp)
    hidden, inter, topk = args.hidden_size, args.moe_intermediate_size, args.num_experts_per_tok
    route_ids = route_ids.to(device="cuda", dtype=torch.int64)
    if tuple(route_ids.shape) != (tokens, topk):
        raise ValueError("Routing corpus shape differs from padded DP token layout")
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    local_x = x[:local_tokens]
    w1 = torch.randn(local_experts,2*inter,hidden,device="cuda",dtype=torch.bfloat16)*.02
    w2 = torch.randn(local_experts,hidden,inter,device="cuda",dtype=torch.bfloat16)*.02
    router_w = torch.randn(args.num_experts,hidden,device="cuda",dtype=torch.bfloat16)*.02
    logits = torch.full((tokens,args.num_experts),-10.,device="cuda",dtype=torch.bfloat16)
    logits.scatter_(1,route_ids,10.)
    expert_map = torch.full((args.num_experts,),-1,device="cuda",dtype=torch.int32)
    offset = ep_rank*local_experts
    expert_map[offset:offset+local_experts] = torch.arange(local_experts,device="cuda",dtype=torch.int32)
    shared_inter = args.shared_expert_intermediate_size
    up_w = torch.randn(2*shared_inter,hidden,device="cuda",dtype=torch.bfloat16)*.02
    down_w = torch.randn(hidden,shared_inter,device="cuda",dtype=torch.bfloat16)*.02
    gate_w = torch.randn(1,hidden,device="cuda",dtype=torch.bfloat16)*.02
    overlap = local_tokens <= getattr(args,"shared_expert_stream_token_threshold",256)
    stream = torch.cuda.Stream() if overlap else None
    def shared(z):
        up = F.linear(z,up_w)
        act = torch.empty(local_tokens,shared_inter,device="cuda",dtype=torch.bfloat16)
        torch.ops._C.silu_and_mul(act,up)
        out = F.linear(act,down_w)
        return torch.sigmoid(F.linear(z,gate_w))*out
    def run():
        if overlap:
            z = local_x.clone()
            z.record_stream(stream)
            stream.wait_stream(torch.cuda.current_stream())
        F.linear(local_x,router_w)
        weights, ids, _ = fused_topk(x,logits,topk,renormalize=True)
        routed = fused_experts(x,w1,w2,weights,ids,global_num_experts=args.num_experts,expert_map=expert_map)
        if overlap:
            with torch.cuda.stream(stream):out = shared(z)
            torch.cuda.current_stream().wait_stream(stream)
        else:
            out = shared(local_x)
        # Reduction/scatter is excluded. The final local add is included.
        out + local_x
        return routed
    with torch.inference_mode():
        return benchmark_gpu_region(run,warmup=3,samples=samples,
                                    timing_mode="eager" if args.phase=="prefill" else "cuda_graph")
''';p.write_text(s)
