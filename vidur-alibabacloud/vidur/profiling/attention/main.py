import argparse
import datetime
import json
import os
from types import SimpleNamespace

import pandas as pd
import torch

from vidur.profiling.attention.attention_wrapper import AttentionWrapper
from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.utils import get_attention_input_combinations, get_max_num_blocks


DEFAULT_MODELS = ["microsoft/phi-2", "internlm/internlm-20b", "Qwen/Qwen-72B",
                  "meta-llama/Llama-2-7b-hf", "codellama/CodeLlama-34b-Instruct-hf",
                  "meta-llama/Llama-2-70b-hf", "meta-llama/Meta-Llama-3-8B",
                  "meta-llama/Meta-Llama-3-70B"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Native attention and KV-write profiling")
    parser.add_argument("--disable_ray", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--output_dir", default="profiling_outputs")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--num_tensor_parallel_workers", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Native model/capture limit; decode block tables are padded to this length")
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--min_batch_size", type=int, default=1)
    parser.add_argument("--max_batch_size", type=int, default=128)
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--profile_only_decode", action="store_true")
    phase.add_argument("--profile_only_prefill", action="store_true")
    parser.add_argument("--block_size", type=int, default=16)
    return parser.parse_args(argv)


def profile_model(args, model, tp, combinations):
    config = ModelConfig.from_model_name(model)
    config.validate_parallelism(tp)
    parallel = SimpleNamespace(tensor_parallel_size=tp, pipeline_parallel_size=1)
    max_blocks = get_max_num_blocks(config, parallel, args.block_size, torch.float16)
    combinations = [x for x in combinations if x.is_valid(args.max_model_len)
                    and x.is_under_memory_limit(max_blocks * args.block_size)]
    constructor = (config, parallel, max_blocks, args.max_model_len, args.block_size, torch.float16)
    if args.disable_ray:
        worker = AttentionWrapper(*constructor)
        rows = [worker.profile(x) for x in combinations]
        del worker
        torch.cuda.empty_cache()
        return rows
    import ray
    actor = ray.remote(num_cpus=1, num_gpus=1)(AttentionWrapper)
    workers = [actor.options(runtime_env={"env_vars": {"KINETO_LOG_LEVEL": "5"}}).remote(*constructor)
               for _ in range(args.num_gpus)]
    rows = []
    try:
        for i in range(0, len(combinations), len(workers)):
            rows.extend(ray.get([worker.profile.remote(item) for worker, item
                                 in zip(workers, combinations[i:i+len(workers)])]))
    finally:
        for worker in workers:
            ray.kill(worker)
    return rows


def main():
    args = parse_args()
    if min(args.num_gpus, args.min_batch_size, args.block_size, args.max_seq_len) < 1:
        raise ValueError("Profiling dimensions must be positive")
    if args.max_batch_size < args.min_batch_size or args.max_seq_len > args.max_model_len:
        raise ValueError("Invalid batch or context range")
    if not args.disable_ray and args.num_gpus > torch.cuda.device_count():
        raise ValueError("num_gpus exceeds the GPUs visible to this profiling process")
    out = os.path.join(args.output_dir, "attention", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"))
    os.makedirs(out, exist_ok=False)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    combinations = get_attention_input_combinations(args.max_seq_len, args.min_batch_size,
        args.max_batch_size, args.profile_only_prefill, args.profile_only_decode)
    for model in args.models:
        config = ModelConfig.from_model_name(model)
        rows = []
        for tp in args.num_tensor_parallel_workers:
            if config.no_tensor_parallel and tp != 1:
                continue
            rows.extend(profile_model(args, model, tp, combinations))
        if not rows:
            raise ValueError("No valid profiling combinations for " + model)
        frame = pd.json_normalize(rows)
        path = os.path.join(out, model, "attention.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        frame.to_csv(path, index=False)
        print(path, flush=True)


if __name__ == "__main__":
    main()
