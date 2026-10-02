#!/usr/bin/env python3
"""Print a compact summary for the newest completed Vidur result."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from statistics import fmean


RUN_DIR_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})-(\d{6})$")
QPS_IN_NAME_RE = re.compile(r"(?:^|[-_])qps(?P<qps>\d+(?:\.\d+)?)", re.IGNORECASE)
RESULT_ROOT_MARKERS = ("benchmark", "result", "run", "output", "simulator")
PRUNED_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules"}


def run_datetime(metrics_file: Path) -> datetime | None:
    """Read Vidur's timestamped output-directory name."""
    for parent in metrics_file.parents:
        match = RUN_DIR_RE.fullmatch(parent.name)
        if not match:
            continue
        value = f"{match.group(1)}_{match.group(2)}-{match.group(3)}"
        parsed = datetime.strptime(value, "%Y-%m-%d_%H-%M-%S-%f")
        return parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return None


def candidate_metric_files(base_dir: Path) -> list[Path]:
    """Scan result-like top-level directories without walking source/data trees."""
    candidates: set[Path] = set()
    direct = base_dir / "request_metrics.csv"
    if direct.is_file():
        candidates.add(direct)

    try:
        children = list(base_dir.iterdir())
    except OSError as exc:
        raise RuntimeError(f"cannot scan {base_dir}: {exc}") from exc

    roots = [
        child
        for child in children
        if child.is_dir()
        and any(marker in child.name.lower() for marker in RESULT_ROOT_MARKERS)
    ]
    for root in roots:
        for current, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in PRUNED_DIRS and not d.startswith(".")]
            if "request_metrics.csv" in files:
                candidates.add(Path(current) / "request_metrics.csv")
    return list(candidates)


def sort_timestamp(path: Path) -> float:
    parsed = run_datetime(path)
    if parsed is not None:
        return parsed.timestamp()
    return path.stat().st_mtime


def number(row: dict[str, str], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def load_completed_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    completed = []
    for row in rows:
        required = (
            number(row, "completed_at"),
            number(row, "request_e2e_time"),
            number(row, "prefill_e2e_time"),
            number(row, "request_num_prefill_tokens"),
            number(row, "request_num_decode_tokens"),
        )
        if all(math.isfinite(value) for value in required) and required[0] >= 0:
            completed.append(row)
    return completed


def percentile(values: list[float], percent: float) -> float:
    """NumPy-compatible linear percentile for a one-dimensional sample."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def metric_line(label: str, values_ms: list[float]) -> str:
    return (
        f"{label}: mean={fmean(values_ms):.3f} ms, "
        f"P50={percentile(values_ms, 50):.3f}, "
        f"P90={percentile(values_ms, 90):.3f}, "
        f"P95={percentile(values_ms, 95):.3f}"
    )


def token_summary(values: list[int]) -> str:
    total = sum(values)
    if len(set(values)) == 1:
        return f"{values[0]}/request (total={total})"
    return (
        f"mean={fmean(values):.1f}/request, min={min(values)}, "
        f"max={max(values)}, total={total}"
    )


def load_config(metrics_file: Path) -> dict:
    config_file = metrics_file.parent / "config.json"
    if not config_file.is_file():
        return {}
    try:
        with config_file.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {config_file}: {exc}") from exc
    if not isinstance(config, dict):
        raise RuntimeError(f"invalid config object in {config_file}")
    return config


def directory_qps(metrics_file: Path) -> float | None:
    for parent in metrics_file.parents:
        match = QPS_IN_NAME_RE.search(parent.name)
        if match:
            return float(match.group("qps"))
    return None


def nested(config: dict, *keys: str):
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def summarize(metrics_file: Path, base_dir: Path) -> None:
    rows = load_completed_rows(metrics_file)
    if not rows:
        raise RuntimeError(f"no completed requests in {metrics_file}")
    config = load_config(metrics_file)

    ttft_ms = [number(row, "prefill_e2e_time") * 1000.0 for row in rows]
    e2e_ms = [number(row, "request_e2e_time") * 1000.0 for row in rows]
    input_tokens = [int(number(row, "request_num_prefill_tokens")) for row in rows]
    output_tokens = [int(number(row, "request_num_decode_tokens")) for row in rows]

    # Match vllm_benchmark_client.py exactly: the first output token is included
    # in TTFT, so TPOT averages the remaining output_tokens - 1 intervals.
    tpot_ms = []
    for ttft, e2e, output_size in zip(ttft_ms, e2e_ms, output_tokens):
        if output_size > 1:
            tpot_ms.append((e2e - ttft) / (output_size - 1))
        elif output_size == 1:
            tpot_ms.append(0.0)
        else:
            raise RuntimeError("output token count must be positive")

    # Vidur's timestamps are measured from simulator time zero.
    total_runtime_ms = max(number(row, "completed_at") for row in rows) * 1000.0
    started_at = run_datetime(metrics_file)
    if started_at is None:
        started_at = datetime.fromtimestamp(metrics_file.stat().st_mtime).astimezone()
    completed_at = started_at + timedelta(milliseconds=total_runtime_ms)

    try:
        result_dir = metrics_file.parent.relative_to(base_dir)
    except ValueError:
        result_dir = metrics_file.parent

    configured_qps = nested(
        config,
        "request_generator_config",
        "interval_generator_config",
        "qps",
    )
    configured_requests = nested(config, "request_generator_config", "num_requests")
    tensor_parallel = nested(
        config, "cluster_config", "replica_config", "tensor_parallel_size"
    )
    pipeline_parallel = nested(
        config, "cluster_config", "replica_config", "num_pipeline_stages"
    )
    nccl_launch_ms = nested(
        config, "execution_time_predictor_config", "nccl_cpu_launch_overhead_ms"
    )
    skip_cpu_overhead = nested(
        config, "execution_time_predictor_config", "skip_cpu_overhead_modeling"
    )
    qps_from_directory = directory_qps(metrics_file)

    print("========== Results ==========")
    print(f"Result directory    : {result_dir}")
    print(f"Run started at      : {started_at.isoformat(sep=' ', timespec='microseconds')}")
    print(f"Run completed at    : {completed_at.isoformat(sep=' ', timespec='microseconds')}")
    if configured_qps is not None:
        print(f"Configured QPS      : {configured_qps:g}")
    if configured_requests is not None:
        print(f"Configured requests : {configured_requests}")
    if tensor_parallel is not None and pipeline_parallel is not None:
        print(f"Parallelism         : TP={tensor_parallel}, PP={pipeline_parallel}")
    if nccl_launch_ms is not None:
        print(f"NCCL CPU launch     : {nccl_launch_ms:g} ms")
    if skip_cpu_overhead is not None:
        state = "disabled" if skip_cpu_overhead else "enabled"
        print(f"CPU overhead model : {state}")
    print(f"Input tokens        : {token_summary(input_tokens)}")
    print(f"Output tokens       : {token_summary(output_tokens)}")
    print(f"Successful requests : {len(rows)}")
    print(f"Total runtime       : {total_runtime_ms:.3f} ms")
    print(metric_line("TTFT", ttft_ms))
    print(metric_line("TPOT", tpot_ms))
    print(metric_line("E2E", e2e_ms))
    if (
        configured_qps is not None
        and qps_from_directory is not None
        and not math.isclose(float(configured_qps), qps_from_directory)
    ):
        print(
            "WARNING: directory QPS "
            f"({qps_from_directory:g}) != config.json QPS ({configured_qps:g}); "
            "the config.json value was used."
        )


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    candidates = candidate_metric_files(base_dir)
    if not candidates:
        print(f"No Vidur request_metrics.csv found under {base_dir}", file=sys.stderr)
        return 1

    errors = []
    for metrics_file in sorted(candidates, key=sort_timestamp, reverse=True):
        try:
            rows = load_completed_rows(metrics_file)
            if not rows:
                errors.append(f"{metrics_file}: no completed requests")
                continue
            summarize(metrics_file, base_dir)
            return 0
        except (OSError, csv.Error, RuntimeError) as exc:
            errors.append(str(exc))

    print("No complete Vidur result could be read.", file=sys.stderr)
    for error in errors[:5]:
        print(f"  - {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
