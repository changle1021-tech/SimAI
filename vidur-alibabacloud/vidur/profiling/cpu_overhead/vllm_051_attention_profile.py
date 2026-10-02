#!/usr/bin/env python3
"""Profile vLLM 0.5.1 FlashAttention kernels into Vidur's attention.csv schema.

This measures the actual vLLM FlashAttention prefill/decode calls and KV
cache writes. Legacy CSVs include writes in the attention target; schemas
with attn_kv_cache_save profile them separately, without double counting.
It does not include QKV/output projections; those are in mlp.csv.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import List, Sequence, Tuple

from vllm_051_profile_common import (check_metadata, check_vllm_version, close_tp,
                                     csv_schema, gpu_times, init_tp, put_stats,
                                     read_reference, resolve_tp_sizes, run_all_tp, write_csv)


from native_attention_geometry import local_attention_heads

DEFAULT_OUTPUT = "/root/changle/Files/vidur_vllm051_profiles/attention.csv"


def prefill_grid(max_seq_len: int) -> List[int]:
    grid = (list(range(64, 129, 16)) + list(range(128, 1025, 32))
            + list(range(1024, 4097, 64)) + list(range(4096, 16385, 128))
            + list(range(16384, 65537, 256)))
    result = []
    for value in grid:
        if value > max_seq_len:
            break
        result.append(value)
    return result


def kv_grid(max_seq_len: int) -> List[int]:
    grid = (list(range(0, 1025, 32)) + list(range(1024, 4097, 64))
            + list(range(4096, 65537, 256)))
    result = []
    for value in grid:
        if value >= max_seq_len:
            break
        result.append(value)
    return result


def batch_grid(min_batch_size: int, max_batch_size: int) -> List[int]:
    grid = list(range(1, 129)) + list(range(128, 1025, 8))
    return [value for value in grid
            if min_batch_size <= value <= max_batch_size]


def cases(args: argparse.Namespace, tp: int | None = None) -> List[Tuple[int, int, int, bool]]:
    if args.cases_from_csv:
        result = []
        for row in read_reference(args.cases_from_csv, args.tensor_parallel_sizes, tp):
            if int(row["block_size"]) != args.block_size:
                raise ValueError("Input CSV block_size differs from --block-size")
            if int(row["max_model_len"]) != args.max_model_len:
                raise ValueError("Input CSV max_model_len differs from --max-model-len")
            prefill_text = row["is_prefill"].lower()
            if prefill_text not in ("true", "false"):
                raise ValueError(f"Invalid is_prefill value: {row['is_prefill']}")
            is_prefill = prefill_text == "true"
            if ((args.only_decode and is_prefill)
                    or (args.only_prefill and not is_prefill)):
                continue
            result.append((int(row["prefill_chunk_size"]),
                           int(row["kv_cache_size"]), int(row["batch_size"]), is_prefill))
        invalid = [case for case in result if not valid_case(args, case)]
        if invalid:
            raise ValueError(f"{len(invalid)} input CSV cases exceed the length or "
                             f"memory limits; first invalid case: {invalid[0]}")
        # Keep repeated source rows: the original profiler also measured
        # duplicate grid-boundary cases independently.
        return result

    chunks = args.prefill_chunk_sizes or prefill_grid(args.max_seq_len)
    kv_sizes = args.kv_cache_sizes if args.kv_cache_sizes is not None else kv_grid(args.max_seq_len)
    batches = args.batch_sizes or batch_grid(args.min_batch_size,
                                              args.max_batch_size)
    result = []
    if not args.only_decode:
        for chunk in chunks:
            # Vidur samples chunked prefill at integer multiples of the
            # chunk length, rather than crossing the decode KV length grid.
            chunk_kv_sizes = (kv_sizes if args.kv_cache_sizes is not None else
                              range(0, (args.max_seq_len // chunk) * chunk, chunk))
            for kv in chunk_kv_sizes:
                if kv + chunk <= args.max_seq_len:
                    result.append((chunk, kv, 1, True))
        # Full prefill includes short lengths (32/64/96...) that are not all
        # in the chunk grid. Keep duplicates exactly as the original does.
        if args.prefill_chunk_sizes is None and args.kv_cache_sizes is None:
            result.extend((length, 0, 1, True) for length in kv_sizes if length > 0)
    if not args.only_prefill:
        for kv in kv_sizes:
            if kv < 1:
                continue
            for batch in batches:
                result.append((0, kv, batch, False))
    return [case for case in result if valid_case(args, case)]


def valid_case(args: argparse.Namespace, case: Tuple[int, int, int, bool]) -> bool:
    chunk, kv, batch, prefill = case
    if kv < 0 or batch < 1 or chunk < 0:
        return False
    if prefill and (batch != 1 or chunk < 1):
        return False
    if not prefill and (chunk != 0 or kv < 1):
        return False
    return ((chunk + kv if prefill else kv + 1) <= args.max_model_len
            and (batch if prefill or args.decode_layout == "eager" else graph_batch_size(batch)) * ((kv + chunk) if prefill else
                          args.decode_block_table_capacity if args.decode_layout == "cuda_graph" else kv + 1) <= args.max_kv_tokens)


def reference_defaults(args: argparse.Namespace) -> None:
    references = (read_reference(args.cases_from_csv, args.tensor_parallel_sizes)
                  if args.cases_from_csv else [])
    for option, default in (("max_model_len", 4096), ("block_size", 16)):
        if getattr(args, option) is None:
            values = {int(row[option]) for row in references}
            if len(values) > 1:
                raise ValueError(f"Reference CSV contains multiple {option} values")
            setattr(args, option, next(iter(values)) if values else default)
    if args.max_kv_tokens is None:
        args.max_kv_tokens = max((int(row["batch_size"]) *
                                  (int(row["kv_cache_size"]) +
                                   (int(row["prefill_chunk_size"]) if row["is_prefill"].lower() == "true" else 1))
                                  for row in references), default=524288)


def graph_batch_size(size):
    if size <= 2: return size
    if size <= 4: return 4
    return math.ceil(size / 8) * 8


def execution_fields(args):
    fields = csv_schema("attention", args.cases_from_csv)
    for name in ["attn_kv_cache_save"]:
        for stat in ["min", "max", "mean", "median", "std"]:
            col = f"time_stats.{name}.{stat}"
            if col not in fields: fields.append(col)
    return fields + ["profile_schema_version", "decode_layout", "decode_block_table_capacity", "physical_batch_size", "timing_semantics"]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/mnt/data02/000000/model/Llama-2-7b-hf")
    p.add_argument("--tensor-parallel-sizes", "--num_tensor_parallel_workers",
                   "--tensor_parallel_sizes", type=int, nargs="+",
                   help="TP sizes (all reference TP sizes, or 1 2 4 without a reference)")
    p.add_argument("--max-model-len", "--max_model_len", type=int,
                   help="Model length (reference value, or 4096 without a reference)")
    p.add_argument("--max-seq-len", "--max_seq_len", type=int, default=4096)
    p.add_argument("--min-batch-size", "--min_batch_size", type=int, default=1)
    p.add_argument("--max-batch-size", "--max_batch_size", type=int, default=128)
    p.add_argument("--batch-sizes", "--batch_sizes", type=int, nargs="+")
    p.add_argument("--prefill-chunk-sizes", "--prefill_chunk_sizes", type=int, nargs="+")
    p.add_argument("--kv-cache-sizes", "--kv_cache_sizes", type=int, nargs="+")
    p.add_argument("--cases-from-csv", "--cases_from_csv", "--reference-csv", "--reference_csv",
                   dest="cases_from_csv", help="Replay old CSV rows and preserve its exact header")
    p.add_argument("--only-decode", "--profile_only_decode", action="store_true")
    p.add_argument("--only-prefill", "--profile_only_prefill", action="store_true")
    p.add_argument("--block-size", "--block_size", type=int,
                   help="Block size (reference value, or 16 without a reference)")
    p.add_argument("--max-kv-tokens", "--max_kv_tokens", type=int,
                   help="Memory limit (reference maximum, or 524288 without a reference)")
    p.add_argument("--decode-layout", choices=("eager", "cuda_graph"), default="cuda_graph")
    p.add_argument("--decode-block-table-capacity", type=int, default=None, help="CUDA graph block table token capacity, independent of active context length")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--part-output", help=argparse.SUPPRESS)
    return p


def profile_case(args: argparse.Namespace, config, tp: int,
                 case: Tuple[int, int, int, bool], cache: torch.Tensor) -> dict:
    import torch
    from vllm import _custom_ops as ops
    from vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    chunk, kv_size, batch, is_prefill = case
    block_size = args.block_size
    heads, kv_heads = local_attention_heads(config.num_attention_heads,
        getattr(config, "num_key_value_heads", config.num_attention_heads), tp)
    head_size = config.hidden_size // config.num_attention_heads
    token_count = chunk if is_prefill else batch
    physical_batch = batch if is_prefill or args.decode_layout == "eager" else graph_batch_size(batch)
    blocks_per_seq = math.ceil(((kv_size + chunk) if is_prefill else
                               args.decode_block_table_capacity if args.decode_layout == "cuda_graph" else kv_size + 1) / block_size)
    num_blocks = physical_batch * blocks_per_seq
    key_cache, value_cache = cache[0, :num_blocks], cache[1, :num_blocks]
    q = torch.randn((chunk if is_prefill else physical_batch, heads, head_size), device="cuda", dtype=torch.float16)
    k = torch.randn((token_count, kv_heads, head_size), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    if is_prefill:
        slots = torch.arange(kv_size, kv_size + chunk, device="cuda",
                             dtype=torch.long)
        block_table = torch.arange(blocks_per_seq, device="cuda",
                                   dtype=torch.int32).view(1, -1)
    else:
        slots = torch.arange(batch, device="cuda", dtype=torch.long)
        slots = slots * (blocks_per_seq * block_size) + kv_size
        block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32)
        block_table = block_table.view(physical_batch, blocks_per_seq)

    def save_kv():
        return ops.reshape_and_cache_flash(k, v, key_cache, value_cache,
                                           slots, "auto")

    save_kv()
    torch.cuda.synchronize()
    scale = head_size ** -0.5
    if is_prefill and kv_size == 0:
        offsets = torch.tensor([0, chunk], device="cuda", dtype=torch.int32)

        def attention():
            return flash_attn_varlen_func(
                q=q, k=k, v=v, cu_seqlens_q=offsets, cu_seqlens_k=offsets,
                max_seqlen_q=chunk, max_seqlen_k=chunk,
                softmax_scale=scale, causal=True)
    elif is_prefill:
        q_offsets = torch.tensor([0, chunk], device="cuda", dtype=torch.int32)
        seq_offsets = torch.tensor([0, kv_size + chunk], device="cuda",
                                   dtype=torch.int32)

        def attention():
            return flash_attn_varlen_func(
                q=q, k=key_cache, v=value_cache,
                cu_seqlens_q=q_offsets, cu_seqlens_k=seq_offsets,
                max_seqlen_q=chunk, max_seqlen_k=kv_size + chunk,
                softmax_scale=scale, causal=True, block_table=block_table)
    else:
        cache_lengths = torch.ones((physical_batch,), device="cuda", dtype=torch.int32)
        cache_lengths[:batch] = kv_size + 1

        def attention():
            return flash_attn_with_kvcache(
                q.unsqueeze(1), key_cache, value_cache,
                block_table=block_table, cache_seqlens=cache_lengths,
                softmax_scale=scale, causal=True).squeeze(1)

    row = {
        "n_embd": config.hidden_size,
        "n_q_head": config.num_attention_heads,
        "n_kv_head": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "block_size": block_size,
        "num_tensor_parallel_workers": tp,
        "max_model_len": args.max_model_len,
        "batch_size": batch,
        "prefill_chunk_size": chunk,
        "kv_cache_size": kv_size,
        "is_prefill": is_prefill,
        "attention_backend": "AttentionBackend.FLASH_ATTENTION",
        "profile_schema_version": 3,
        "physical_batch_size": batch if is_prefill else physical_batch,
        "decode_layout": args.decode_layout if not is_prefill else "eager",
        "decode_block_table_capacity": args.decode_block_table_capacity if not is_prefill and args.decode_layout == "cuda_graph" else 0,
        "timing_semantics": "attention kernels and KV write measured independently",
    }
    name = "attn_prefill" if is_prefill else "attn_decode"
    fields = args.csv_fields
    if "time_stats.attn_kv_cache_save.median" in fields:
        put_stats(row, "attn_kv_cache_save", gpu_times(save_kv, args.warmup, args.repetitions))
        put_stats(row, name, gpu_times(attention, args.warmup, args.repetitions))
    else:
        # The legacy reader supplies zero for the absent cache-write target.
        # Measure both kernels together so replacing that CSV counts writes
        # once, even though the old schema has no independent save column.
        def attention_with_cache():
            save_kv()
            return attention()

        put_stats(row, name, gpu_times(attention_with_cache, args.warmup, args.repetitions))
    # The corresponding vLLM views do not launch CUDA kernels. Keep these
    # targets explicit so old/new CSV schemas and timing boundaries align.
    put_stats(row, "attn_input_reshape", [0.0] * args.repetitions)
    put_stats(row, "attn_output_reshape", [0.0] * args.repetitions)
    return row


def worker(args: argparse.Namespace, combinations: Sequence[tuple]) -> None:
    import torch
    from transformers import AutoConfig
    from vllm.attention.selector import get_attn_backend
    rank, _, tp = init_tp()
    try:
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
        if not all(hasattr(config, name) for name in ("hidden_size", "num_attention_heads")):
            raise ValueError("FlashAttention profiling requires explicit head geometry")
        if getattr(config, "sliding_window", None) or getattr(config, "alibi", False):
            raise ValueError("Windowed/ALiBi attention requires its own backend adapter")
        references = (read_reference(args.cases_from_csv, args.tensor_parallel_sizes, tp)
                      if args.cases_from_csv else [])
        check_metadata({"n_embd": config.hidden_size,
                        "n_q_head": config.num_attention_heads,
                        "n_kv_head": getattr(config, "num_key_value_heads", config.num_attention_heads)},
                       references, ("n_embd", "n_q_head", "n_kv_head"))
        if references:
            labels = {row["attention_backend"] for row in references}
            if not labels <= {"AttentionBackend.FLASH_ATTENTION", "FLASH_ATTENTION", "flash-attn"}:
                raise ValueError("Reference CSV is not FlashAttention; this profiler "
                                 "cannot reproduce FlashInfer/backend-specific measurements")
        selected_references = [reference for reference in references
                               if (reference["is_prefill"].lower() == "true" and not args.only_decode)
                               or (reference["is_prefill"].lower() == "false" and not args.only_prefill)]
        head_size = config.hidden_size // config.num_attention_heads
        kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        local_heads, local_kv = local_attention_heads(config.num_attention_heads, kv_heads, tp)
        backend = get_attn_backend(local_heads, head_size,
                                   local_kv, None, torch.float16, "auto",
                                   args.block_size)
        if backend.get_name() != "flash-attn":
            raise RuntimeError(f"Expected vLLM FlashAttention, got {backend.get_name()}")
        max_blocks = max((case[2] if case[3] or args.decode_layout == "eager" else graph_batch_size(case[2])) * math.ceil(
            ((case[1] + case[0]) if case[3] else
             args.decode_block_table_capacity if args.decode_layout == "cuda_graph" else case[1] + 1) / args.block_size)
                         for case in combinations)
        cache = torch.empty((2, max_blocks, args.block_size, local_kv,
                             head_size), device="cuda", dtype=torch.float16)
        cache.normal_()
        rows = []
        for index, case in enumerate(combinations, 1):
            row = profile_case(args, config, tp, case, cache)
            if references:
                # Replay rows one-for-one, including enum/string formatting.
                row["attention_backend"] = selected_references[index - 1]["attention_backend"]
            rows.append(row)
            if rank == 0 and (index == 1 or index % 100 == 0
                              or index == len(combinations)):
                print(f"TP={tp} attention case {index}/{len(combinations)}: {case}",
                      flush=True)
        if rank == 0:
            write_csv(args.part_output, rows, execution_fields(args))
    finally:
        close_tp()


def main() -> None:
    args = parser().parse_args()
    args.tensor_parallel_sizes = resolve_tp_sizes(args.tensor_parallel_sizes, args.cases_from_csv)
    reference_defaults(args)
    if args.decode_block_table_capacity is None:
        args.decode_block_table_capacity = args.max_model_len
    if args.decode_layout == "cuda_graph" and args.decode_block_table_capacity < args.max_model_len:
        raise ValueError("Set max-model-len to the covered graph sequence limit")
    check_vllm_version()
    if args.only_decode and args.only_prefill:
        raise ValueError("Choose at most one of --only-decode and --only-prefill")
    if args.max_seq_len < 2 or args.max_model_len < 2 or args.block_size < 1:
        raise ValueError("Sequence lengths and block size must be positive")
    if args.block_size % 16:
        raise ValueError("vLLM FlashAttention block-size must be a multiple of 16")
    if args.min_batch_size < 1 or args.max_batch_size < args.min_batch_size or args.max_kv_tokens < 1:
        raise ValueError("Invalid batch size or KV token limits")
    for option in ("batch_sizes", "prefill_chunk_sizes", "kv_cache_sizes"):
        values = getattr(args, option)
        if values is not None and any(value < (0 if option == "kv_cache_sizes" else 1) for value in values):
            raise ValueError(f"Invalid {option} values")
    if args.cases_from_csv and any(getattr(args, option) is not None for option in
                                   ("batch_sizes", "prefill_chunk_sizes", "kv_cache_sizes")):
        raise ValueError("Reference CSV replay cannot be combined with explicit case grids")
    fields = execution_fields(args)
    args.csv_fields = fields
    if args.warmup < 0 or args.repetitions < 2:
        raise ValueError("warmup >= 0 and repetitions >= 2 are required")
    if args.worker:
        combinations = cases(args, tp=int(os.environ["WORLD_SIZE"]))
        if not combinations:
            raise ValueError("No attention cases match this TP size")
        worker(args, combinations)
        return
    for tp in args.tensor_parallel_sizes:
        if not cases(args, tp=tp):
            raise ValueError(f"No attention cases match TP={tp}")
    forwarded = ["--model", args.model, "--tensor-parallel-sizes",
                 *map(str, args.tensor_parallel_sizes),
                 "--max-model-len", str(args.max_model_len),
                 "--max-seq-len", str(args.max_seq_len), "--min-batch-size",
                 str(args.min_batch_size), "--max-batch-size", str(args.max_batch_size),
                 "--block-size", str(args.block_size), "--max-kv-tokens",
                 str(args.max_kv_tokens), "--warmup", str(args.warmup),
                 "--repetitions", str(args.repetitions),
                 "--decode-layout", args.decode_layout,
                 "--decode-block-table-capacity", str(args.decode_block_table_capacity)]
    for option in ("batch_sizes", "prefill_chunk_sizes", "kv_cache_sizes"):
        value = getattr(args, option)
        if value:
            forwarded += ["--" + option.replace("_", "-"), *map(str, value)]
    if args.only_decode:
        forwarded.append("--only-decode")
    if args.only_prefill:
        forwarded.append("--only-prefill")
    if args.cases_from_csv:
        forwarded += ["--cases-from-csv", args.cases_from_csv]
    run_all_tp(str(Path(__file__).resolve()), forwarded,
               args.tensor_parallel_sizes, args.output, fields, args.cases_from_csv, "attention",
               True if args.only_prefill else False if args.only_decode else None)


if __name__ == "__main__":
    main()
