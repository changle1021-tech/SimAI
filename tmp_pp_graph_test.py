import argparse
import os

import torch
import torch.distributed as dist


def exchange(tensor: torch.Tensor, variant: str) -> None:
    rank = dist.get_rank()
    peer = 1 - rank
    if variant == "blocking":
        if rank == 0:
            dist.send(tensor, peer)
        else:
            dist.recv(tensor, peer)
    elif variant == "nonblocking":
        work = dist.isend(tensor, peer) if rank == 0 else dist.irecv(tensor, peer)
        work.wait()
    elif variant == "batch":
        op = dist.isend if rank == 0 else dist.irecv
        works = dist.batch_isend_irecv([dist.P2POp(op, tensor, peer)])
        for work in works:
            work.wait()
    else:
        raise ValueError(variant)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=["blocking", "nonblocking", "batch"])
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    rank = dist.get_rank()
    tensor = torch.full((1024,), float(rank + 1), device="cuda")

    for _ in range(3):
        exchange(tensor, args.variant)
    torch.cuda.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        exchange(tensor, args.variant)
    graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    print(f"rank={rank} variant={args.variant} value={tensor[0].item()} ok", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
