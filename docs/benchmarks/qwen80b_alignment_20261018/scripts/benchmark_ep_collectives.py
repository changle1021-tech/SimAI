"""Four-rank EP communication microbenchmark using vLLM GroupCoordinator.

Run with torchrun, CUDA_VISIBLE_DEVICES=4,5,6,7. Measures both inactive-rank
sizes=0 and dummy-rank sizes=1; these are distinct scenarios, not assumptions.
"""
import argparse
import json
import os
import pathlib
import statistics

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import GroupCoordinator


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--fully-padded-only', action='store_true')
    args = p.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    group = GroupCoordinator([list(range(4))], rank, 'nccl', True, group_name='q80_audit_ep')
    results = []
    for m in (1, 256, 512):
        for padding in ((m,) if args.fully_padded_only else (0, 1)):
            sizes = [m, padding, padding, padding]
            x = torch.ones(sizes[rank], 2048, device='cuda', dtype=torch.bfloat16)
            logits = torch.ones(sizes[rank], 512, device='cuda', dtype=torch.bfloat16)

            def call():
                hidden, router = group.all_gatherv([x, logits], dim=0, sizes=sizes)
                return group.reduce_scatterv(hidden, dim=0, sizes=sizes)

            for _ in range(20):
                output = call()
            torch.cuda.synchronize()
            for mode in ('eager', 'cuda_graph'):
                graph = None
                if mode == 'cuda_graph':
                    graph = torch.cuda.CUDAGraph()
                    with group.graph_capture() as context:
                        with torch.cuda.graph(graph, stream=context.stream):
                            output = call()
                    torch.cuda.synchronize()
                fn = graph.replay if graph else call
                samples = []
                for _ in range(7):
                    dist.barrier()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(50):
                        output = fn()
                    end.record()
                    end.synchronize()
                    value = torch.tensor(start.elapsed_time(end)/50, device='cuda')
                    dist.all_reduce(value, op=dist.ReduceOp.MAX)
                    samples.append(value.item())
                row = dict(active_tokens=m, sizes=sizes, mode=mode,
                           median_ms=statistics.median(samples), samples_ms=samples)
                results.append(row)
                if rank == 0:
                    path = pathlib.Path(args.out)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(dict(
                        world_size=4, dtype='bfloat16',
                        method='GroupCoordinator.all_gatherv(hidden,router) + reduce_scatterv(hidden); max rank CUDA events; 7x50 calls',
                        notes=['Microbenchmark has no attention or expert compute.',
                               'Graph/eager measurements include their respective host launch gaps.',
                               'Do not add different ranks to obtain request latency.'],
                        profiles=results), indent=2))
                    print(json.dumps(row), flush=True)
                del graph
    dist.barrier()
    group.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
