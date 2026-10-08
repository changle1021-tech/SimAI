import argparse
import datetime
import json
import os

import pandas as pd
import torch
from tqdm import tqdm

from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.common.progress import collect_profile_results
from vidur.profiling.mlp.mlp_wrapper import MlpWrapper
from vidur.profiling.utils import get_num_tokens_to_profile


DEFAULT_MODELS = ["microsoft/phi-2", "internlm/internlm-20b", "Qwen/Qwen-72B",
                  "meta-llama/Llama-2-7b-hf", "codellama/CodeLlama-34b-Instruct-hf",
                  "meta-llama/Llama-2-70b-hf", "meta-llama/Meta-Llama-3-8B",
                  "meta-llama/Meta-Llama-3-70B"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Native MLP, projection, norm and RoPE profiling")
    parser.add_argument("--disable_ray", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--output_dir", default="profiling_outputs")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--num_tensor_parallel_workers", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--max_tokens", type=int, default=4096)
    return parser.parse_args(argv)


def profile_model(args, model, tokens, progress=None):
    config = ModelConfig.from_model_name(model)
    rows = []
    for tp in args.num_tensor_parallel_workers:
        if config.no_tensor_parallel and tp != 1:
            continue
        config.validate_parallelism(tp)
        if progress is not None:
            progress.set_description_str(f"MLP {model} TP={tp}")
            progress.set_postfix_str("initializing workers")
        if args.disable_ray:
            worker = MlpWrapper(config, tp)
            for n in tokens:
                if progress is not None:
                    progress.set_postfix_str(f"tokens={n}")
                rows.append(worker.profile(n))
                if progress is not None:
                    progress.update(1)
            del worker
            torch.cuda.empty_cache()
        else:
            import ray
            actor = ray.remote(num_cpus=1, num_gpus=1)(MlpWrapper)
            workers = [actor.options(runtime_env={"env_vars": {"KINETO_LOG_LEVEL": "5"}}).remote(config, tp)
                       for _ in range(args.num_gpus)]
            try:
                for i in range(0, len(tokens), len(workers)):
                    inputs = tokens[i:i+len(workers)]
                    if progress is not None:
                        progress.set_postfix_str(f"running {len(inputs)} shapes")
                    refs = [worker.profile.remote(n) for worker, n in zip(workers, inputs)]
                    rows.extend(collect_profile_results(
                        ray, refs, inputs, progress, lambda n: f"tokens={n}"))
            finally:
                for worker in workers:
                    ray.kill(worker)
    return pd.json_normalize(rows)


def main():
    args = parse_args()
    if args.num_gpus < 1 or args.max_tokens < 1:
        raise ValueError("GPU and token counts must be positive")
    if not args.disable_ray and args.num_gpus > torch.cuda.device_count():
        raise ValueError("num_gpus exceeds the GPUs visible to this profiling process")
    out = os.path.join(args.output_dir, "mlp", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"))
    os.makedirs(out, exist_ok=False)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    tokens = get_num_tokens_to_profile(args.max_tokens)
    total = 0
    for model in args.models:
        config = ModelConfig.from_model_name(model)
        for tp in args.num_tensor_parallel_workers:
            if config.no_tensor_parallel and tp != 1:
                continue
            config.validate_parallelism(tp)
            total += len(tokens)
    with tqdm(total=total, desc="MLP profiling", unit="shape", dynamic_ncols=True) as progress:
        for model in args.models:
            frame = profile_model(args, model, tokens, progress)
            if frame.empty:
                raise ValueError("No valid profiling combinations for " + model)
            path = os.path.join(out, model, "mlp.csv")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            frame.to_csv(path, index=False)
            tqdm.write(path)


if __name__ == "__main__":
    main()
