#!/usr/bin/env python3
"""Shared helpers for standalone vLLM 0.5.1 operator profilers."""

from __future__ import annotations

import csv
import math
import os
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

STATS = ("min", "max", "mean", "median", "std")
MLP_METADATA = ("n_head", "n_kv_head", "n_embd", "n_expanded_embd",
                "vocab_size", "use_gated_mlp", "num_tokens",
                "num_tensor_parallel_workers")
ATTENTION_METADATA = ("n_embd", "n_q_head", "n_kv_head", "block_size",
                      "num_tensor_parallel_workers", "max_model_len", "batch_size",
                      "prefill_chunk_size", "kv_cache_size", "is_prefill",
                      "attention_backend")
MLP_OPERATIONS = ("emb", "input_layernorm", "attn_pre_proj", "attn_rope",
                  "attn_post_proj", "post_attention_layernorm", "mlp_up_proj",
                  "mlp_act", "mlp_down_proj", "add")
# Both layouts occur in Vidur's existing profiling data. The legacy
# FlashAttention files do not have an independent cache-write target.
ATTENTION_OPERATIONS = ("attn_input_reshape", "attn_decode",
                        "attn_output_reshape", "attn_prefill")
ATTENTION_OPERATIONS_WITH_KV = ("attn_input_reshape", "attn_kv_cache_save",
                                "attn_prefill", "attn_decode", "attn_output_reshape")


def csv_schema(kind: str, reference: str | None = None) -> List[str]:
    if kind not in ("mlp", "attention"):
        raise ValueError(f"Unknown profiling CSV kind: {kind}")
    metadata = MLP_METADATA if kind == "mlp" else ATTENTION_METADATA
    operations = MLP_OPERATIONS if kind == "mlp" else ATTENTION_OPERATIONS
    if reference:
        with open(reference, newline="") as stream:
            fields = next(csv.reader(stream), [])
        if len(fields) != len(set(fields)) or not fields:
            raise ValueError("Reference CSV must have a nonempty, unique header")
        missing = set(metadata) - set(fields)
        allowed = set(metadata) | {
            f"time_stats.{name}.{stat}"
            for name in (MLP_OPERATIONS if kind == "mlp" else ATTENTION_OPERATIONS_WITH_KV)
            for stat in STATS
        }
        if missing or set(fields) - allowed:
            raise ValueError(f"Incompatible {kind} reference CSV header: "
                             f"missing={sorted(missing)}, unknown={sorted(set(fields) - allowed)}")
        for name in (MLP_OPERATIONS if kind == "mlp" else ATTENTION_OPERATIONS_WITH_KV):
            group = {f"time_stats.{name}.{stat}" for stat in STATS}
            if set(fields) & group and not group <= set(fields):
                raise ValueError(f"Reference CSV contains incomplete statistics for {name}")
        required = ("emb", "attn_pre_proj", "attn_post_proj", "mlp_up_proj",
                    "mlp_act", "mlp_down_proj") if kind == "mlp" else ()
        for name in required:
            if f"time_stats.{name}.median" not in fields:
                raise ValueError(f"Reference CSV is missing {name}.median")
        return fields
    return [f"time_stats.{name}.{stat}" for name in operations for stat in STATS] + list(metadata)


def resolve_tp_sizes(sizes: Sequence[int] | None, reference: str | None) -> List[int]:
    if sizes is None and reference:
        with open(reference, newline="") as stream:
            sizes = list(dict.fromkeys(int(row["num_tensor_parallel_workers"])
                                       for row in csv.DictReader(stream)))
    if sizes is None:
        sizes = [1, 2, 4]
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("Tensor-parallel sizes must be positive")
    return list(dict.fromkeys(sizes))


def read_reference(path: str, sizes: Sequence[int], tp: int | None = None) -> List[dict]:
    with open(path, newline="") as stream:
        return [row for row in csv.DictReader(stream)
                if int(row["num_tensor_parallel_workers"]) in sizes
                and (tp is None or int(row["num_tensor_parallel_workers"]) == tp)]


def check_metadata(actual: dict, reference_rows: Sequence[dict], keys: Sequence[str]) -> None:
    """Reject a file that Vidur's model/TP filters would discard."""
    for row in reference_rows:
        for key in keys:
            expected = row[key]
            equal = (str(actual[key]).lower() == expected.lower()
                     if isinstance(actual[key], bool) else int(actual[key]) == int(expected))
            if not equal:
                raise ValueError(f"Reference CSV {key}={expected} does not match "
                                 f"profiled model {key}={actual[key]}")


def align_reference_rows(rows: Sequence[dict], reference: str, sizes: Sequence[int],
                         kind: str, phase: bool | None = None) -> List[dict]:
    """Validate full metadata/coverage and restore global reference row order."""
    metadata = MLP_METADATA if kind == "mlp" else ATTENTION_METADATA

    def key(row):
        return tuple(str(row[name]).lower() for name in metadata)

    remaining = defaultdict(deque)
    for row in rows:
        remaining[key(row)].append(row)
    ordered = []
    for row in read_reference(reference, sizes):
        if kind == "attention" and phase is not None:
            if (row["is_prefill"].lower() == "true") != phase:
                continue
        identity = key(row)
        if not remaining[identity]:
            raise ValueError(f"Generated {kind} CSV is missing reference parameters: {identity}")
        ordered.append(remaining[identity].popleft())
    if any(remaining.values()):
        raise ValueError(f"Generated {kind} CSV contains rows absent from the reference")
    return ordered


def check_vllm_version() -> None:
    import vllm

    if vllm.__version__ != "0.5.1":
        raise RuntimeError(f"These profilers require vLLM 0.5.1, found {vllm.__version__}")


def init_tp() -> tuple[int, int, int]:
    import torch
    from vllm.distributed import (init_distributed_environment,
                                  initialize_model_parallel)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    init_distributed_environment(world_size=world_size,
                                 rank=rank,
                                 local_rank=local_rank)
    initialize_model_parallel(tensor_model_parallel_size=world_size,
                              pipeline_model_parallel_size=1)
    return rank, local_rank, world_size


def close_tp() -> None:
    import torch
    from vllm.distributed import destroy_model_parallel

    destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def gpu_times(fn: Callable[[], object], warmup: int, repetitions: int) -> List[float]:
    """Sum CUDA kernel durations per call, as Vidur's Kineto tracer does."""
    import torch
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as profiler:
            for _ in range(repetitions):
                with torch.profiler.record_function("vidur_vllm051_operation"):
                    fn()
        torch.cuda.synchronize()
    # Kineto exposes both a CPU annotation and its CUDA mirror. FlashAttention
    # decode's C++ launcher is not correlated to the CPU annotation, while the
    # CUDA mirror still carries the kernel duration for the marked call.
    events = [event for event in profiler.events()
              if event.name == "vidur_vllm051_operation"
              and event.device_type == torch.autograd.DeviceType.CUDA]
    timings = [event.cuda_time_total * 1e-3 for event in events]
    if len(timings) != repetitions:
        raise RuntimeError(f"Expected {repetitions} CUDA measurements, got "
                           f"{len(timings)}")
    if any(not math.isfinite(value) or value <= 0 for value in timings):
        raise RuntimeError(f"Kineto did not capture CUDA kernels for every "
                           f"call: {timings}")
    values = torch.tensor(timings,
                          device="cuda", dtype=torch.float64)
    if torch.distributed.get_world_size() > 1:
        # A TP step finishes when its slowest rank has finished.
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)
    return values.cpu().tolist()


def put_stats(row: dict, name: str, values: Sequence[float]) -> None:
    if not values or any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"Invalid timings for {name}: {values}")
    prefix = f"time_stats.{name}."
    row[prefix + "min"] = min(values)
    row[prefix + "max"] = max(values)
    row[prefix + "mean"] = statistics.fmean(values)
    row[prefix + "median"] = statistics.median(values)
    row[prefix + "std"] = statistics.pstdev(values)


def write_csv(path: str, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    if not rows:
        raise RuntimeError("No profiling rows were produced")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    # Projection onto a reference header deliberately removes metrics that
    # its version of Vidur does not expose. Missing phase-specific fields
    # remain empty, as with pandas.json_normalize in the original profiler.
    rows = [{key: row.get(key, "") for key in fieldnames} for row in rows]
    temporary = target.with_name(target.name + ".partial")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)


def run_all_tp(script: str, args: Sequence[str], sizes: Iterable[int],
               output: str, fieldnames: Sequence[str] | None = None,
               reference: str | None = None, kind: str | None = None,
               phase: bool | None = None) -> None:
    import torch
    sizes = list(dict.fromkeys(sizes))
    if not sizes or any(tp < 1 for tp in sizes):
        raise ValueError("Tensor-parallel sizes must be positive")
    available = torch.cuda.device_count()
    if max(sizes) > available:
        raise ValueError(f"Requested TP={max(sizes)} but only {available} GPUs are visible")
    rows: List[dict] = []
    with tempfile.TemporaryDirectory(prefix="vllm051_operator_profile_") as temp:
        for tp in sizes:
            part = str(Path(temp) / f"tp{tp}.csv")
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                       "--nnodes=1", f"--nproc_per_node={tp}", script, "--worker",
                       "--part-output", part, *args]
            print(f"Profiling TP={tp}", flush=True)
            subprocess.run(command, check=True)
            with open(part, newline="") as stream:
                rows.extend(csv.DictReader(stream))
    if reference:
        rows = align_reference_rows(rows, reference, sizes, kind, phase)
    write_csv(output, rows, fieldnames)
    print(f"Wrote {len(rows)} rows to {output}", flush=True)
