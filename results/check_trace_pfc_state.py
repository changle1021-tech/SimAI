#!/usr/bin/env python3
"""记录每个 rank、端口和优先级队列的 PFC 状态变化。"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from trace_events import TraceEvent, TraceEventReader


@dataclass(frozen=True)
class PfcQueueKey:
    rank: int
    interface: int
    queue_index: int


@dataclass(frozen=True)
class PfcStateChange:
    event_time: str
    event_time_value: int
    line_number: int
    key: PfcQueueKey
    is_paused: bool
    pause_time: int
    advertised_queue_length: int
    node_port_queue: str
    src_ip: str
    dst_ip: str

    @property
    def state(self) -> str:
        return "PAUSED" if self.is_paused else "RESUMED"


class PfcStateAnalyzer:
    """消费 PFC Recv 事件，只保存真正改变接收端状态的事件。"""

    def __init__(self, rank_filter: Optional[Set[int]]) -> None:
        self.rank_filter = rank_filter
        self.states: Dict[PfcQueueKey, bool] = {}
        self.changes: List[PfcStateChange] = []
        self.pfc_receive_events = 0
        self.unchanged_events = 0
        self.malformed_pfc_events = 0

    def consume(self, event: TraceEvent) -> None:
        # Enqu/Dequ 描述 PFC 包在链路上的传输；只有 Recv 才改变接收端口状态。
        if event.packet_type != "P" or event.action != "Recv":
            return
        if self.rank_filter is not None and event.event_rank not in self.rank_filter:
            return

        self.pfc_receive_events += 1
        try:
            interface = int(event.port_queue.split(":", 1)[0])
            pause_time = int(event.details[0])
            advertised_queue_length = int(event.details[1])
            queue_index = int(event.details[2])
        except (IndexError, ValueError):
            self.malformed_pfc_events += 1
            return

        key = PfcQueueKey(
            rank=event.event_rank,
            interface=interface,
            queue_index=queue_index,
        )
        is_paused = pause_time > 0
        # 仿真开始时队列默认为未暂停；相同状态的重复 PFC 不生成记录。
        if self.states.get(key, False) == is_paused:
            self.unchanged_events += 1
            return

        self.states[key] = is_paused
        self.changes.append(
            PfcStateChange(
                event_time=event.timestamp,
                event_time_value=event.timestamp_value,
                line_number=event.line_number,
                key=key,
                is_paused=is_paused,
                pause_time=pause_time,
                advertised_queue_length=advertised_queue_length,
                node_port_queue=event.port_queue,
                src_ip=event.src_ip,
                dst_ip=event.dst_ip,
            )
        )

    @property
    def paused_at_trace_end(self) -> List[PfcQueueKey]:
        return sorted(key for key, is_paused in self.states.items() if is_paused)


def sorted_changes(analyzer: PfcStateAnalyzer) -> List[PfcStateChange]:
    changes = list(analyzer.changes)
    changes.sort(
        key=lambda change: (
            change.event_time_value,
            change.line_number,
            change.key.rank,
            change.key.interface,
            change.key.queue_index,
        )
    )
    return changes


def write_csv(output: Path, changes: List[PfcStateChange]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "event_time",
                "rank",
                "interface",
                "queue_index",
                "state",
                "is_paused",
                "pause_time",
                "advertised_queue_length",
                "node_port_queue",
                "src_ip",
                "dst_ip",
                "trace_line",
            ]
        )
        for change in changes:
            writer.writerow(
                [
                    change.event_time,
                    change.key.rank,
                    change.key.interface,
                    change.key.queue_index,
                    change.state,
                    int(change.is_paused),
                    change.pause_time,
                    change.advertised_queue_length,
                    change.node_port_queue,
                    change.src_ip,
                    change.dst_ip,
                    change.line_number,
                ]
            )


def print_report(
    trace: Path,
    output: Path,
    selected_ranks: List[int],
    analyzer: PfcStateAnalyzer,
    changes: List[PfcStateChange],
    malformed_lines: int,
    max_display: int,
) -> None:
    print("PFC 状态变化分析")
    print(f"Trace: {trace}")
    print(f"rank 范围: {','.join(str(rank) for rank in selected_ranks)}")

    changes_by_rank: Dict[int, int] = defaultdict(int)
    for change in changes:
        changes_by_rank[change.key.rank] += 1
    for rank in selected_ranks:
        print(f"rank {rank}: PFC 状态变化 {changes_by_rank[rank]}")

    print(f"PFC Recv 事件: {analyzer.pfc_receive_events}")
    print(f"合计 PFC 状态变化: {len(changes)}")
    if analyzer.unchanged_events:
        print(f"提示: 跳过 {analyzer.unchanged_events} 个未改变状态的重复 PFC 事件。")
    if analyzer.malformed_pfc_events:
        print(f"提示: 跳过 {analyzer.malformed_pfc_events} 个字段异常的 PFC 事件。")
    if malformed_lines:
        print(f"提示: 跳过 {malformed_lines} 行无法解析的非空记录。")
    if analyzer.paused_at_trace_end:
        queues = ", ".join(
            f"rank={key.rank}/if={key.interface}/q={key.queue_index}"
            for key in analyzer.paused_at_trace_end
        )
        print(f"提示: trace 结束时仍处于暂停状态: {queues}")

    shown = changes if max_display == 0 else changes[:max_display]
    if shown:
        print("\n按事件时间排序的 PFC 状态变化：")
        for change in shown:
            print(
                f"- t={change.event_time} rank={change.key.rank} "
                f"if={change.key.interface} q={change.key.queue_index} "
                f"state={change.state}"
            )
        if len(shown) < len(changes):
            print(f"... 还有 {len(changes) - len(shown)} 条，完整内容见 CSV。")

    print(f"\nPFC 状态变化 CSV 已写入: {output}")
    print("event_time 的单位与 trace 第一列一致。")


def finish_analysis(
    trace: Path,
    output: Path,
    selected_ranks: List[int],
    analyzer: PfcStateAnalyzer,
    malformed_lines: int,
    max_display: int,
) -> int:
    changes = sorted_changes(analyzer)
    write_csv(output, changes)
    print_report(
        trace,
        output,
        selected_ranks,
        analyzer,
        changes,
        malformed_lines,
        max_display,
    )
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "读取 PFC Recv 事件，在每个 rank/端口/优先级队列的暂停状态"
            "发生变化时写入 CSV；第一列为事件时间。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  %(prog)s trace.txt\n"
            "  %(prog)s trace.txt 6 10\n"
            "  %(prog)s trace.txt --output trace_pfc_state.csv"
        ),
    )
    parser.add_argument("trace", type=Path, help="待分析的 trace.txt 路径")
    parser.add_argument(
        "ranks",
        type=int,
        nargs="*",
        metavar="RANK",
        help="可选的接收端 rank；不指定时分析全部 rank",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV 输出路径；默认在 trace 旁生成 <trace名>_pfc_state.csv",
    )
    parser.add_argument(
        "--max-display",
        type=int,
        default=20,
        help="终端最多显示多少条状态变化；0 表示全部显示",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    total_start = time.perf_counter()
    args = parse_args(argv)
    if args.max_display < 0:
        print("错误: --max-display 不能小于 0。", file=sys.stderr)
        return 2
    if not args.trace.is_file():
        print(f"错误: trace 文件不存在: {args.trace}", file=sys.stderr)
        return 2
    if any(rank < 0 for rank in args.ranks):
        print("错误: rank 不能为负数。", file=sys.stderr)
        return 2

    requested_ranks = sorted(set(args.ranks))
    rank_filter = set(requested_ranks) if requested_ranks else None
    analyzer = PfcStateAnalyzer(rank_filter)
    reader = TraceEventReader(args.trace)
    parse_start = time.perf_counter()
    for event in reader:
        analyzer.consume(event)
    parse_seconds = time.perf_counter() - parse_start

    if not reader.seen_ranks:
        print("错误: trace 中没有找到任何 n:<rank> 事件。", file=sys.stderr)
        return 2
    missing_ranks = sorted(set(requested_ranks) - reader.seen_ranks)
    if missing_ranks:
        print(f"错误: trace 中没有 rank 事件: {missing_ranks}", file=sys.stderr)
        return 2

    output = args.output or args.trace.with_name(f"{args.trace.stem}_pfc_state.csv")
    status = finish_analysis(
        trace=args.trace,
        output=output,
        selected_ranks=requested_ranks or sorted(reader.seen_ranks),
        analyzer=analyzer,
        malformed_lines=reader.malformed_lines,
        max_display=args.max_display,
    )
    print(f"共享事件解析耗时: {parse_seconds:.3f} 秒")
    print(f"PFC 状态变化分析总耗时: {time.perf_counter() - total_start:.3f} 秒")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
