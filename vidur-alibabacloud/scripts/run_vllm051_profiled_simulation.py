#!/usr/bin/env python3
"""Export decode predictions or replay an immutable arrival trace using saved config.

Only diagnostic outputs are written; active profiling inputs are never replaced.
Run from the Vidur repository in its existing Python environment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import is_dataclass
from pathlib import Path


def restore_config(saved, output_dir, trace=None):
    from vidur.config import SimulationConfig
    from vidur.config.base_poly_config import BasePolyConfig
    from vidur.config.flat_dataclass import create_flat_dataclass
    from vidur.config.utils import get_all_subclasses, is_subclass
    flat_class = create_flat_dataclass(SimulationConfig)
    flat = flat_class()

    def apply(cls, data):
        for prefixed, original, field_type in flat_class.dataclass_args[cls]:
            if original not in data:
                continue
            value = data[original]
            if is_subclass(field_type, BasePolyConfig):
                name = value["name"]
                matches = [s for s in get_all_subclasses(field_type)
                           if str(s.get_type()) == name]
                if len(matches) != 1:
                    raise ValueError(f"Cannot restore {original}={name}")
                setattr(flat, original + "_type", name)
                apply(matches[0], value)
            elif is_dataclass(field_type):
                apply(field_type, value)
            else:
                setattr(flat, prefixed, value)

    if trace:
        saved = dict(saved)
        saved["request_generator_config"] = {
            "name": "trace_replay", "seed": saved.get("seed", 36),
            "trace_file": str(trace), "prefill_scale_factor": 1.0,
            "decode_scale_factor": 1.0, "time_scale_factor": 1.0,
            "max_tokens": 4096,
        }
    apply(SimulationConfig, saved)
    flat.metrics_config_output_dir = str(output_dir)
    config = flat.reconstruct_original_dataclass()
    config.__flat_config__ = flat
    return config


def export_matrix(config, args):
    import statistics
    from vidur.entities.batch import Batch
    from vidur.entities.request import Request, RequestType
    from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
    r = config.cluster_config.replica_config
    s = config.cluster_config.replica_scheduler_config
    e = config.execution_time_predictor_config
    predictor = ExecutionTimePredictorRegistry.get(
        e.get_type(), predictor_config=e, replica_config=r,
        replica_scheduler_config=s, metrics_config=config.metrics_config,
        simulation_config=config)
    rows = []
    for prompt in args.prompt_tokens:
        for batch_size in args.batch_sizes:
            steps = []
            for step in range(1, args.decode_tokens):
                requests = [Request(0, prompt, args.decode_tokens,
                                    num_processed_tokens=prompt + step)
                            for _ in range(batch_size)]
                for request in requests:
                    request._is_prefill_complete = True
                    request.request_type = RequestType.DECODE
                batch = Batch(0, requests, [1] * batch_size)
                timing = predictor.get_execution_time(batch, 0)
                steps.append({
                    "prompt_tokens": prompt, "batch_size": batch_size,
                    "decode_step": step, "total_ms": timing.total_time * 1000,
                    "model_ms": timing.model_time_ms,
                    "cpu_model_ms": timing.total_time * 1000 - timing.model_time_ms,
                    "schedule_ms": timing._schedule_time,
                    "sampler_e2e_ms": timing._sampler_e2e_time,
                    "prepare_inputs_e2e_ms": timing._prepare_inputs_e2e_time,
                    "process_outputs_ms": timing._process_model_outputs_time,
                    "ray_comm_ms": timing._ray_comm_time,
                })
            row = {"prompt_tokens": prompt, "batch_size": batch_size}
            for key in steps[0]:
                if key.endswith("_ms"):
                    row[key] = statistics.fmean(x[key] for x in steps)
            rows.append(row)
    output = Path(args.output_dir)
    with (output / "vidur_matrix.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    inputs = {}
    for name in ("compute", "attention", "all_reduce", "cpu_overhead"):
        path = Path(getattr(predictor, "_" + name + "_input_file"))
        inputs[name] = {"path": str(path.resolve()),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (output / "vidur_inputs.json").write_text(json.dumps(inputs, indent=2))
    print(json.dumps(rows), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mode", choices=("matrix", "replay", "simulate"), default="matrix")
    p.add_argument("--trace", type=Path)
    p.add_argument("--prompt-tokens", nargs="+", type=int, default=[128, 512, 3072])
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--decode-tokens", type=int, default=50)
    args = p.parse_args()
    if args.mode == "replay" and args.trace is None:
        p.error("replay requires an immutable trace")
    if args.decode_tokens < 2 or min(args.batch_sizes + args.prompt_tokens) < 1:
        p.error("need positive batch/length and at least two output tokens")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    saved = json.loads(args.config.read_text())
    config = restore_config(saved, args.output_dir / "runs", args.trace)
    if args.mode == "matrix":
        export_matrix(config, args)
    else:
        from vidur.simulator import Simulator
        from vidur.utils.random import set_seeds
        set_seeds(config.seed)
        Simulator(config).run()
        (args.output_dir / "run_path.txt").write_text(config.metrics_config.output_dir)
        print("RESULT", config.metrics_config.output_dir, flush=True)


if __name__ == "__main__":
    main()
