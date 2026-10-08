"""One profiling implementation shared by attention and MLP collectors."""
import statistics

import torch


PROFILE_METHOD = "vllm_cuda_graph_kernel_sum_v2"


def graph_batch_size(size):
    if size <= 2:
        return size
    if size <= 4:
        return 4
    return (size + 7) // 8 * 8


def kernel_time_ms(events, excluded_kernel_names=()):
    """Do not add CPU parent times to their CUDA child kernel times."""
    total = 0.0
    for event in events:
        if str(event.device_type) != "DeviceType.CUDA":
            continue
        if event.name in excluded_kernel_names:
            continue
        if "FillFunctor<long>" in event.name:
            continue  # PyTorch graph-replay RNG bookkeeping, outside the operator.
        total += event.self_cuda_time_total * 1e-3
    return total


class GPUKernelProfiler:
    def __init__(self, rounds=5, replays=10):
        import vllm

        if vllm.__version__.split("+")[0] != "0.5.1":
            raise ValueError("The integrated native profiler requires vLLM 0.5.1")
        self.vllm_version = vllm.__version__
        self.rounds = rounds
        self.replays = replays
        # Read a buffer larger than L2. Writing it would leave dirty cache lines
        # whose later writeback contaminates the following memory-bound GEMM.
        self.eviction = torch.randn(32 * 1024 * 1024, dtype=torch.float32, device="cuda")
        self.eviction.sum()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as profiler:
            self.eviction.sum()
            torch.cuda.synchronize()
        self.conditioning_kernels = {e.name for e in profiler.events()
                                     if str(e.device_type) == "DeviceType.CUDA"}
        if not self.conditioning_kernels:
            raise RuntimeError("Cache-conditioning kernels were not captured")

    @torch.inference_mode()
    def measure(self, operation):
        if operation is None:
            return dict(min=0.0, max=0.0, mean=0.0, median=0.0, std=0.0, samples=[])
        for _ in range(5):
            output = operation()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = operation()
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        samples = []
        kernel_names = set()
        for _ in range(self.rounds):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                for _ in range(self.replays):
                    self.eviction.sum()
                    graph.replay()
                torch.cuda.synchronize()
            value = kernel_time_ms(profiler.events(), self.conditioning_kernels) / self.replays
            kernel_names.update(e.name for e in profiler.events()
                                if str(e.device_type) == "DeviceType.CUDA"
                                and e.name not in self.conditioning_kernels
                                and "FillFunctor<long>" not in e.name)
            if value <= 0:
                raise RuntimeError("No CUDA kernel duration captured for an active operator")
            samples.append(value)
        del graph, output
        return dict(min=min(samples), max=max(samples), mean=statistics.fmean(samples),
                    median=statistics.median(samples), std=statistics.pstdev(samples),
                    samples=samples, kernels=sorted(kernel_names))
