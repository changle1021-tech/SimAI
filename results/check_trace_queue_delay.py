#!/usr/bin/env python3
"""按 rank 计算同一数据包从 Enqu 到 Dequ 的本地排队时延。"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

from trace_events import TraceEvent, TraceEventReader


@dataclass(frozen=True)
class QueuePacketKey:
    rank: int
    port_queue: str
    src_ip: str
    dst_ip: str
    src_port: str
    dst_port: str
    packet_type: str
    details: Tuple[str, ...]


@dataclass(frozen=True)
class EnqueueEvent:
    timestamp: str
    timestamp_value: int
    line_number: int
    queue_length: str
    ecn: str


@dataclass(frozen=True)
class QueueDelaySample:
    key: QueuePacketKey
    enqueue: EnqueueEvent
    dequeue: TraceEvent

    @property
    def queue_delay(self) -> int:
        return self.dequeue.timestamp_value - self.enqueue.timestamp_value


class QueueDelayAnalyzer:
    """逐事件匹配每个 rank 本地队列中的 Enqu/Dequ。"""

    def __init__(self, rank_filter: Optional[Set[int]]) -> None:
        self.rank_filter = rank_filter
        self.pending: Dict[QueuePacketKey, Deque[EnqueueEvent]] = defaultdict(deque)
        self.samples: List[QueueDelaySample] = []
        self.enqueue_events = 0
        self.dequeue_events = 0
        self.unmatched_dequeues = 0

    def consume(self, event: TraceEvent) -> None:
        # PFC 状态由专用分析器处理，避免控制包混入数据包排队时延。
        if event.packet_type == "P":
            return
        if event.action not in {"Enqu", "Dequ"}:
            return
        if self.rank_filter is not None and event.event_rank not in self.rank_filter:
            return

        key = QueuePacketKey(
            rank=event.event_rank,
            port_queue=event.port_queue,
            src_ip=event.src_ip,
            dst_ip=event.dst_ip,
            src_port=event.src_port,
            dst_port=event.dst_port,
            packet_type=event.packet_type,
            # ECN 和队列长度可能在排队期间变化，不属于包身份。
            details=event.details,
        )
        if event.action == "Enqu":
            self.pending[key].append(
                EnqueueEvent(
                    timestamp=event.timestamp,
                    timestamp_value=event.timestamp_value,
                    line_number=event.line_number,
                    queue_length=event.queue_length,
                    ecn=event.ecn,
                )
            )
            self.enqueue_events += 1
            return

        self.dequeue_events += 1
        pending = self.pending.get(key)
        if not pending:
            # 例如源端直接出现 Dequ、trace 从队列中途开始，均没有可计算时延的 Enqu。
            self.unmatched_dequeues += 1
            return
        enqueue = pending.popleft()
        if not pending:
            del self.pending[key]
        self.samples.append(
            QueueDelaySample(key=key, enqueue=enqueue, dequeue=event)
        )

    @property
    def unmatched_enqueues(self) -> int:
        return sum(len(events) for events in self.pending.values())


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "匹配每个 rank 上同一包的 Enqu/Dequ，计算本地排队时延，"
            "并按出队时间写入同一个 CSV。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  %(prog)s trace.txt\n"
            "  %(prog)s trace.txt 0 24\n"
            "  %(prog)s trace.txt 0 24 --output trace_queue_delay.csv"
        ),
    )
    parser.add_argument("trace", type=Path, help="待分析的 trace.txt 路径")
    parser.add_argument(
        "ranks",
        type=int,
        nargs="*",
        metavar="RANK",
        help="可选的事件所在 rank；不指定时分析全部 rank",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV 输出路径；默认在 trace 旁生成 <trace名>_queue_delay.csv",
    )
    parser.add_argument(
        "--max-display",
        type=int,
        default=20,
        help="终端最多显示多少条排队时延；0 表示全部显示",
    )
    return parser.parse_args(argv)


def sorted_samples(analyzer: QueueDelayAnalyzer) -> List[QueueDelaySample]:
    samples = list(analyzer.samples)
    samples.sort(
        key=lambda sample: (
            sample.dequeue.timestamp_value,
            sample.dequeue.line_number,
            sample.key.rank,
            sample.enqueue.line_number,
        )
    )
    return samples


def write_csv(output: Path, samples: List[QueueDelaySample]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dequeue_time",
                "rank",
                "queue_delay",
                "enqueue_time",
                "node_port_queue",
                "packet_type",
                "src_rank",
                "dst_rank",
                "src_ip",
                "dst_ip",
                "src_port",
                "dst_port",
                "packet_fields",
                "queue_length_at_enqueue",
                "queue_length_at_dequeue",
                "ecn_at_enqueue",
                "ecn_at_dequeue",
                "enqueue_line",
                "dequeue_line",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    sample.dequeue.timestamp,
                    sample.key.rank,
                    sample.queue_delay,
                    sample.enqueue.timestamp,
                    sample.key.port_queue,
                    sample.key.packet_type,
                    sample.dequeue.src_rank,
                    sample.dequeue.dst_rank,
                    sample.key.src_ip,
                    sample.key.dst_ip,
                    sample.key.src_port,
                    sample.key.dst_port,
                    " ".join(sample.key.details),
                    sample.enqueue.queue_length,
                    sample.dequeue.queue_length,
                    sample.enqueue.ecn,
                    sample.dequeue.ecn,
                    sample.enqueue.line_number,
                    sample.dequeue.line_number,
                ]
            )


def print_report(
    trace: Path,
    output: Path,
    selected_ranks: List[int],
    analyzer: QueueDelayAnalyzer,
    samples: List[QueueDelaySample],
    malformed_lines: int,
    max_display: int,
) -> None:
    print("排队时延分析")
    print(f"Trace: {trace}")
    print(f"rank 范围: {','.join(str(rank) for rank in selected_ranks)}")

    sample_count_by_rank: Dict[int, int] = defaultdict(int)
    for sample in samples:
        sample_count_by_rank[sample.key.rank] += 1
    for rank in selected_ranks:
        print(f"rank {rank}: 排队时延样本 {sample_count_by_rank[rank]}")

    print(f"合计排队时延样本: {len(samples)}")
    if analyzer.unmatched_dequeues:
        print(
            f"提示: {analyzer.unmatched_dequeues} 个 Dequ 没有更早的对应 Enqu，"
            "未生成排队时延。"
        )
    if analyzer.unmatched_enqueues:
        print(
            f"提示: {analyzer.unmatched_enqueues} 个 Enqu 在 trace 结束前没有对应 Dequ。"
        )
    if malformed_lines:
        print(f"提示: 跳过 {malformed_lines} 行无法解析的非空记录。")

    shown = samples if max_display == 0 else samples[:max_display]
    if shown:
        print("\n按出队时间排序的排队时延：")
        for sample in shown:
            print(
                f"- dequ_t={sample.dequeue.timestamp} rank={sample.key.rank} "
                f"queue_delay={sample.queue_delay} type={sample.key.packet_type}"
            )
        if len(shown) < len(samples):
            print(f"... 还有 {len(samples) - len(shown)} 条，完整内容见 CSV。")

    print(f"\n排队时延 CSV 已写入: {output}")
    print("时间和排队时延的单位与 trace 第一列一致。")


def finish_analysis(
    trace: Path,
    output: Path,
    selected_ranks: List[int],
    analyzer: QueueDelayAnalyzer,
    malformed_lines: int,
    max_display: int,
) -> int:
    samples = sorted_samples(analyzer)
    write_csv(output, samples)
    print_report(
        trace,
        output,
        selected_ranks,
        analyzer,
        samples,
        malformed_lines,
        max_display,
    )
    return 0


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
    analyzer = QueueDelayAnalyzer(rank_filter)
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

    output = args.output or args.trace.with_name(
        f"{args.trace.stem}_queue_delay.csv"
    )
    status = finish_analysis(
        trace=args.trace,
        output=output,
        selected_ranks=requested_ranks or sorted(reader.seen_ranks),
        analyzer=analyzer,
        malformed_lines=reader.malformed_lines,
        max_display=args.max_display,
    )
    print(f"共享事件解析耗时: {parse_seconds:.3f} 秒")
    print(f"排队时延分析总耗时: {time.perf_counter() - total_start:.3f} 秒")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
