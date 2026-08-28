#!/usr/bin/env python3
"""统计各场景每条有向链路超过带宽利用率阈值的次数和时间占比。"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_THRESHOLDS = (90.0, 95.0, 99.0)
DEFAULT_TOPOLOGY_NAME = "Spectrum-X_32g_2gps_100Gbps_A100"
DEFAULT_BW_RELATIVE_PATH = Path("astra-sim/simulation/llama_hpn7_bw.txt")


@dataclass(frozen=True)
class DirectedLink:
    link_id: int
    node_id: int
    port_id: int
    peer_node_id: int
    peer_port_id: int
    capacity_gbps: float

    @property
    def path(self) -> str:
        return (
            f"{self.node_id}:{self.port_id}->"
            f"{self.peer_node_id}:{self.peer_port_id}"
        )


@dataclass
class LinkStats:
    sample_rows: int
    bandwidth_sum_gbps: float
    max_bandwidth_gbps: float
    threshold_counts: Dict[float, int]


@dataclass(frozen=True)
class ScenarioResult:
    scenario: str
    topology: Path
    bw_file: Path
    interval_us: int
    monitor_start_ns: int
    observation_end_ns: int
    total_samples: int
    links: List[DirectedLink]
    stats: Dict[Tuple[int, int], LinkStats]
    unmatched_bw_rows: int


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "读取 results 下各场景的 bw 文件和拓扑文件，统计每条有向链路"
            "带宽利用率超过 90%、95%、99% 的次数、持续时间和占比。"
        )
    )
    parser.add_argument(
        "scenarios",
        nargs="*",
        help="要分析的场景名；不指定时自动分析所有包含 bw 和拓扑文件的场景",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="results 目录，默认是脚本所在目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="汇总 CSV；默认写入 <results-dir>/bw_utilization_thresholds.csv",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="利用率阈值（百分比），默认：90 95 99",
    )
    parser.add_argument(
        "--topology-name",
        default=DEFAULT_TOPOLOGY_NAME,
        help=f"场景目录中的拓扑文件名，默认：{DEFAULT_TOPOLOGY_NAME}",
    )
    parser.add_argument(
        "--bw-relative-path",
        type=Path,
        default=DEFAULT_BW_RELATIVE_PATH,
        help=f"场景目录内的 bw 相对路径，默认：{DEFAULT_BW_RELATIVE_PATH}",
    )
    args = parser.parse_args(argv)

    thresholds = sorted(set(args.thresholds))
    if not thresholds or any(value < 0 for value in thresholds):
        parser.error("阈值必须是非负数")
    args.thresholds = thresholds
    return args


def parse_data_rate_gbps(value: str) -> float:
    match = re.fullmatch(
        r"\s*([0-9]+(?:\.[0-9]+)?)\s*([kKmMgGtT]?)bps\s*", value
    )
    if not match:
        raise ValueError(f"无法解析链路速率: {value!r}")
    amount = float(match.group(1))
    prefix = match.group(2).lower()
    scale_to_gbps = {
        "": 1e-9,
        "k": 1e-6,
        "m": 1e-3,
        "g": 1.0,
        "t": 1e3,
    }
    return amount * scale_to_gbps[prefix]


def parse_topology(path: Path) -> List[DirectedLink]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) < 3:
        raise ValueError(f"拓扑文件内容不足: {path}")

    header = lines[0].split()
    if len(header) < 5:
        raise ValueError(f"拓扑首行格式异常: {path}: {lines[0]}")
    try:
        link_count = int(header[4])
    except ValueError as exc:
        raise ValueError(f"拓扑首行无法取得链路数量: {path}") from exc

    link_lines = lines[2 : 2 + link_count]
    if len(link_lines) != link_count:
        raise ValueError(
            f"拓扑声明 {link_count} 条链路，实际只读取到 {len(link_lines)} 条: {path}"
        )

    next_port: Dict[int, int] = defaultdict(int)
    directed_links: List[DirectedLink] = []
    for link_id, line in enumerate(link_lines, start=1):
        fields = line.split()
        if len(fields) < 3:
            raise ValueError(f"拓扑链路行格式异常: {path}: {line}")
        try:
            src = int(fields[0])
            dst = int(fields[1])
        except ValueError as exc:
            raise ValueError(f"拓扑节点编号格式异常: {path}: {line}") from exc
        capacity_gbps = parse_data_rate_gbps(fields[2])
        if capacity_gbps <= 0:
            raise ValueError(f"链路速率必须大于 0: {path}: {line}")

        next_port[src] += 1
        src_port = next_port[src]
        next_port[dst] += 1
        dst_port = next_port[dst]

        directed_links.append(
            DirectedLink(
                link_id=link_id,
                node_id=src,
                port_id=src_port,
                peer_node_id=dst,
                peer_port_id=dst_port,
                capacity_gbps=capacity_gbps,
            )
        )
        directed_links.append(
            DirectedLink(
                link_id=link_id,
                node_id=dst,
                port_id=dst_port,
                peer_node_id=src,
                peer_port_id=src_port,
                capacity_gbps=capacity_gbps,
            )
        )
    return directed_links


def read_monitor_config(path: Path) -> Tuple[Optional[int], int]:
    interval_us: Optional[int] = None
    monitor_start_us = 0
    if not path.is_file():
        return interval_us, monitor_start_us

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        if fields[0] == "BW_MON_INTERVAL":
            interval_us = int(fields[1])
        elif fields[0] == "MON_START":
            monitor_start_us = int(fields[1])
    return interval_us, monitor_start_us


def infer_interval_us(timestamps_ns: Iterable[int]) -> int:
    ordered = sorted(set(timestamps_ns))
    if len(ordered) < 2:
        raise ValueError("bw 时间点不足，且 SimAI.conf 未提供 BW_MON_INTERVAL")
    gcd_ns = 0
    for previous, current in zip(ordered, ordered[1:]):
        difference = current - previous
        if difference > 0:
            gcd_ns = math.gcd(gcd_ns, difference)
    if gcd_ns <= 0 or gcd_ns % 1000 != 0:
        raise ValueError("无法从 bw 时间戳推断微秒单位的采样间隔")
    return gcd_ns // 1000


def read_bw_rows(path: Path) -> List[Tuple[int, int, int, float]]:
    rows: List[Tuple[int, int, int, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"bw 文件为空: {path}")
        normalized_header = [field.strip() for field in header]
        expected = ["time", "node_id", "port_id", "bandwidth"]
        if normalized_header[:4] != expected:
            raise ValueError(
                f"bw 表头异常，期望 {expected}，实际 {normalized_header}: {path}"
            )
        for line_number, fields in enumerate(reader, start=2):
            if not fields or all(not field.strip() for field in fields):
                continue
            if len(fields) < 4:
                raise ValueError(f"bw 第 {line_number} 行字段不足: {path}")
            try:
                rows.append(
                    (
                        int(fields[0].strip()),
                        int(fields[1].strip()),
                        int(fields[2].strip()),
                        float(fields[3].strip()),
                    )
                )
            except ValueError as exc:
                raise ValueError(f"bw 第 {line_number} 行数值格式异常: {path}") from exc
    if not rows:
        raise ValueError(f"bw 文件没有采样数据: {path}")
    return rows


def analyze_scenario(
    scenario_dir: Path,
    topology: Path,
    bw_file: Path,
    thresholds: Sequence[float],
) -> ScenarioResult:
    links = parse_topology(topology)
    link_by_port = {(link.node_id, link.port_id): link for link in links}
    bw_rows = read_bw_rows(bw_file)

    configured_interval_us, monitor_start_us = read_monitor_config(
        scenario_dir / "SimAI.conf"
    )
    interval_us = configured_interval_us or infer_interval_us(
        timestamp for timestamp, _, _, _ in bw_rows
    )
    if interval_us <= 0:
        raise ValueError(f"BW_MON_INTERVAL 必须大于 0: {scenario_dir}")

    interval_ns = interval_us * 1000
    monitor_start_ns = monitor_start_us * 1000
    observation_end_ns = max(timestamp for timestamp, _, _, _ in bw_rows)
    observed_ns = observation_end_ns - monitor_start_ns
    total_samples = max(1, math.ceil(observed_ns / interval_ns))

    stats: Dict[Tuple[int, int], LinkStats] = {
        key: LinkStats(
            sample_rows=0,
            bandwidth_sum_gbps=0.0,
            max_bandwidth_gbps=0.0,
            threshold_counts={threshold: 0 for threshold in thresholds},
        )
        for key in link_by_port
    }
    unmatched_bw_rows = 0
    seen_samples: Set[Tuple[int, int, int]] = set()

    for timestamp, node_id, port_id, bandwidth_gbps in bw_rows:
        key = (node_id, port_id)
        link = link_by_port.get(key)
        if link is None:
            unmatched_bw_rows += 1
            continue
        sample_key = (timestamp, node_id, port_id)
        if sample_key in seen_samples:
            raise ValueError(
                f"bw 中存在重复采样: time={timestamp}, node={node_id}, port={port_id}"
            )
        seen_samples.add(sample_key)

        item = stats[key]
        item.sample_rows += 1
        item.bandwidth_sum_gbps += bandwidth_gbps
        item.max_bandwidth_gbps = max(item.max_bandwidth_gbps, bandwidth_gbps)
        utilization_pct = bandwidth_gbps / link.capacity_gbps * 100.0
        for threshold in thresholds:
            if utilization_pct > threshold:
                item.threshold_counts[threshold] += 1

    return ScenarioResult(
        scenario=scenario_dir.name,
        topology=topology,
        bw_file=bw_file,
        interval_us=interval_us,
        monitor_start_ns=monitor_start_ns,
        observation_end_ns=observation_end_ns,
        total_samples=total_samples,
        links=links,
        stats=stats,
        unmatched_bw_rows=unmatched_bw_rows,
    )


def threshold_label(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")


def write_results(
    output: Path,
    results: Sequence[ScenarioResult],
    thresholds: Sequence[float],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    base_header = [
        "scenario",
        "link_id",
        "directional_path",
        "node_id",
        "port_id",
        "peer_node_id",
        "peer_port_id",
        "capacity_gbps",
        "sample_interval_us",
        "observation_start_ns",
        "observation_end_ns",
        "total_sample_count",
        "nonzero_sample_count",
        "average_bandwidth_gbps",
        "average_utilization_pct",
        "max_bandwidth_gbps",
        "max_utilization_pct",
    ]
    threshold_header: List[str] = []
    for threshold in thresholds:
        label = threshold_label(threshold)
        threshold_header.extend(
            [
                f"over_{label}_count",
                f"over_{label}_duration_us",
                f"over_{label}_ratio_pct",
            ]
        )

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(base_header + threshold_header)
        for result in sorted(results, key=lambda item: item.scenario):
            for link in sorted(
                result.links,
                key=lambda item: (item.link_id, item.node_id, item.port_id),
            ):
                item = result.stats[(link.node_id, link.port_id)]
                average_bandwidth = item.bandwidth_sum_gbps / result.total_samples
                average_utilization = average_bandwidth / link.capacity_gbps * 100.0
                max_utilization = (
                    item.max_bandwidth_gbps / link.capacity_gbps * 100.0
                )
                row: List[object] = [
                    result.scenario,
                    link.link_id,
                    link.path,
                    link.node_id,
                    link.port_id,
                    link.peer_node_id,
                    link.peer_port_id,
                    f"{link.capacity_gbps:.6f}",
                    result.interval_us,
                    result.monitor_start_ns,
                    result.observation_end_ns,
                    result.total_samples,
                    item.sample_rows,
                    f"{average_bandwidth:.6f}",
                    f"{average_utilization:.6f}",
                    f"{item.max_bandwidth_gbps:.6f}",
                    f"{max_utilization:.6f}",
                ]
                for threshold in thresholds:
                    count = item.threshold_counts[threshold]
                    ratio_pct = count / result.total_samples * 100.0
                    row.extend(
                        [
                            count,
                            count * result.interval_us,
                            f"{ratio_pct:.6f}",
                        ]
                    )
                writer.writerow(row)


def discover_scenarios(
    results_dir: Path,
    requested: Sequence[str],
    topology_name: str,
    bw_relative_path: Path,
) -> Tuple[List[Tuple[Path, Path, Path]], List[str]]:
    if requested:
        scenario_dirs = [results_dir / name for name in requested]
    else:
        scenario_dirs = sorted(
            (path for path in results_dir.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )

    found: List[Tuple[Path, Path, Path]] = []
    skipped: List[str] = []
    for scenario_dir in scenario_dirs:
        if not scenario_dir.is_dir():
            skipped.append(f"{scenario_dir.name}: 场景目录不存在")
            continue
        topology = scenario_dir / topology_name
        bw_file = scenario_dir / bw_relative_path
        missing = []
        if not topology.is_file():
            missing.append(f"拓扑 {topology.name}")
        if not bw_file.is_file():
            missing.append(f"bw {bw_relative_path}")
        if missing:
            skipped.append(f"{scenario_dir.name}: 缺少 " + "、".join(missing))
            continue
        found.append((scenario_dir, topology, bw_file))
    return found, skipped


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        print(f"错误: results 目录不存在: {results_dir}", file=sys.stderr)
        return 2

    discovered, skipped = discover_scenarios(
        results_dir,
        args.scenarios,
        args.topology_name,
        args.bw_relative_path,
    )
    if not discovered:
        print("错误: 没有找到同时包含 bw 和拓扑文件的场景", file=sys.stderr)
        for message in skipped:
            print(f"- {message}", file=sys.stderr)
        return 2

    results: List[ScenarioResult] = []
    failed = False
    for scenario_dir, topology, bw_file in discovered:
        try:
            result = analyze_scenario(
                scenario_dir, topology, bw_file, args.thresholds
            )
        except (OSError, ValueError) as exc:
            print(f"错误: {scenario_dir.name}: {exc}", file=sys.stderr)
            failed = True
            continue
        results.append(result)
        print(
            f"{result.scenario}: {len(result.links)} 条有向链路，"
            f"{result.total_samples} 个采样周期，间隔 {result.interval_us} us"
        )
        if result.unmatched_bw_rows:
            print(
                f"  警告: {result.unmatched_bw_rows} 行 bw 无法映射到拓扑端口，已忽略",
                file=sys.stderr,
            )

    if not results:
        return 1

    output = (args.output or results_dir / "bw_utilization_thresholds.csv").resolve()
    write_results(output, results, args.thresholds)
    print(f"汇总结果: {output}")
    print(
        "占比分母为从 MON_START 到 bw 最后时间点的全部采样周期；"
        "bw 未记录的端口/时刻按 0 Gbps 处理。"
    )
    if skipped and not args.scenarios:
        print(f"自动扫描时跳过 {len(skipped)} 个不完整目录。")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
