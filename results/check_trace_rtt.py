#!/usr/bin/env python3
"""根据数据包的发送事件和返回的 ACK 计算各 rank 的 RTT。"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from trace_events import TraceEvent, TraceEventReader


PAYLOAD_PATTERN = re.compile(r"\((\d+)\)$")


@dataclass(frozen=True)
class FlowKey:
    """始终使用原始数据包的方向描述一条流。"""

    sender_rank: int
    receiver_rank: int
    sender_ip: str
    receiver_ip: str
    sender_port: str
    receiver_port: str
    priority_group: str


@dataclass(frozen=True)
class DataSend:
    timestamp: str
    timestamp_value: int
    line_number: int
    sequence: int
    payload_bytes: int

    @property
    def end_sequence(self) -> int:
        return self.sequence + self.payload_bytes


@dataclass(frozen=True)
class AckReceive:
    timestamp: str
    timestamp_value: int
    line_number: int
    ack_sequence: int


@dataclass
class FlowHistory:
    sends: List[DataSend] = field(default_factory=list)
    acks: List[AckReceive] = field(default_factory=list)


@dataclass(frozen=True)
class RttSample:
    flow: FlowKey
    send: DataSend
    ack: AckReceive
    retransmission_count: int

    @property
    def rtt(self) -> int:
        return self.ack.timestamp_value - self.send.timestamp_value


@dataclass(frozen=True)
class ScanStats:
    malformed_lines: int
    data_send_events: int
    ack_receive_events: int


Histories = Dict[FlowKey, FlowHistory]


class RttAnalyzer:
    """消费结构化事件并累计 RTT 匹配所需的数据。"""

    def __init__(self, sender_filter: Optional[Set[int]]) -> None:
        self.sender_filter = sender_filter
        self.histories: Histories = defaultdict(FlowHistory)
        self.data_send_events = 0
        self.ack_receive_events = 0

    def consume(self, event: TraceEvent) -> None:
        if (
            event.packet_type == "U"
            and event.action == "Dequ"
            and event.event_rank == event.src_rank
        ):
            if (
                self.sender_filter is not None
                and event.src_rank not in self.sender_filter
            ):
                return
            try:
                sequence = int(event.details[0])
                priority_group = event.details[2]
                payload_bytes = parse_payload_bytes(event.details[3])
            except (IndexError, ValueError):
                return

            key = FlowKey(
                sender_rank=event.src_rank,
                receiver_rank=event.dst_rank,
                sender_ip=event.src_ip,
                receiver_ip=event.dst_ip,
                sender_port=event.src_port,
                receiver_port=event.dst_port,
                priority_group=priority_group,
            )
            self.histories[key].sends.append(
                DataSend(
                    timestamp=event.timestamp,
                    timestamp_value=event.timestamp_value,
                    line_number=event.line_number,
                    sequence=sequence,
                    payload_bytes=payload_bytes,
                )
            )
            self.data_send_events += 1
            return

        if not (
            event.packet_type == "A"
            and event.action == "Recv"
            and event.event_rank == event.dst_rank
        ):
            return

        original_sender_rank = event.dst_rank
        if (
            self.sender_filter is not None
            and original_sender_rank not in self.sender_filter
        ):
            return
        try:
            priority_group = event.details[1]
            ack_sequence = int(event.details[2])
        except (IndexError, ValueError):
            return

        key = FlowKey(
            sender_rank=original_sender_rank,
            receiver_rank=event.src_rank,
            sender_ip=event.dst_ip,
            receiver_ip=event.src_ip,
            sender_port=event.dst_port,
            receiver_port=event.src_port,
            priority_group=priority_group,
        )
        self.histories[key].acks.append(
            AckReceive(
                timestamp=event.timestamp,
                timestamp_value=event.timestamp_value,
                line_number=event.line_number,
                ack_sequence=ack_sequence,
            )
        )
        self.ack_receive_events += 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按 rank 计算数据包 Dequ 到对应 ACK Recv 的 RTT，"
            "并按 ACK 接收时间写入同一个 CSV。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  %(prog)s trace.txt\n"
            "  %(prog)s trace.txt 0 24\n"
            "  %(prog)s trace.txt 0 24 --output trace_rtt.csv"
        ),
    )
    parser.add_argument("trace", type=Path, help="待分析的 trace.txt 路径")
    parser.add_argument(
        "ranks",
        type=int,
        nargs="*",
        metavar="RANK",
        help="可选的发送方 rank；不指定时分析全部 rank",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV 输出路径；默认在 trace 旁生成 <trace名>_rtt.csv",
    )
    parser.add_argument(
        "--max-display",
        type=int,
        default=20,
        help="终端最多显示多少条 RTT；0 表示全部显示",
    )
    return parser.parse_args(argv)


def address_rank(address: str) -> int:
    """解析形如 0b001801 的地址；末尾两位是主机地址，前面是 rank。"""
    value = address.lower()
    if not value.startswith("0b") or len(value) <= 4:
        raise ValueError(address)
    return int(value[2:-2], 16)


def discover_ranks(trace: Path) -> Set[int]:
    ranks: Set[int] = set()
    with trace.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.split(maxsplit=2)
            if len(fields) < 2 or not fields[1].startswith("n:"):
                continue
            try:
                ranks.add(int(fields[1][2:]))
            except ValueError:
                continue
    return ranks


def parse_payload_bytes(size_field: str) -> int:
    """从 trace 的 ``总字节数(负载字节数)`` 字段读取负载长度。"""
    match = PAYLOAD_PATTERN.search(size_field)
    if match is None:
        raise ValueError(size_field)
    return int(match.group(1))


def scan_trace(
    trace: Path, sender_filter: Optional[Set[int]] = None
) -> Tuple[Histories, ScanStats]:
    """收集数据发送和返回到原发送方的 ACK 接收事件。"""
    histories: Histories = defaultdict(FlowHistory)
    malformed_lines = 0
    data_send_events = 0
    ack_receive_events = 0

    with trace.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if len(fields) < 11:
                if fields:
                    malformed_lines += 1
                continue

            node_token = fields[1]
            if not node_token.startswith("n:"):
                malformed_lines += 1
                continue

            try:
                event_rank = int(node_token[2:])
                timestamp_value = int(fields[0])
                src_rank = address_rank(fields[6])
                dst_rank = address_rank(fields[7])
            except (IndexError, ValueError):
                malformed_lines += 1
                continue

            action = fields[4]
            packet_type = fields[10]

            if packet_type == "U" and action == "Dequ" and event_rank == src_rank:
                if sender_filter is not None and src_rank not in sender_filter:
                    continue
                try:
                    sequence = int(fields[11])
                    priority_group = fields[13]
                    payload_bytes = parse_payload_bytes(fields[14])
                except (IndexError, ValueError):
                    malformed_lines += 1
                    continue

                key = FlowKey(
                    sender_rank=src_rank,
                    receiver_rank=dst_rank,
                    sender_ip=fields[6].lower(),
                    receiver_ip=fields[7].lower(),
                    sender_port=fields[8],
                    receiver_port=fields[9],
                    priority_group=priority_group,
                )
                histories[key].sends.append(
                    DataSend(
                        timestamp=fields[0],
                        timestamp_value=timestamp_value,
                        line_number=line_number,
                        sequence=sequence,
                        payload_bytes=payload_bytes,
                    )
                )
                data_send_events += 1

            elif packet_type == "A" and action == "Recv" and event_rank == dst_rank:
                # ACK 的地址、端口方向与它确认的数据包相反。
                original_sender_rank = dst_rank
                if (
                    sender_filter is not None
                    and original_sender_rank not in sender_filter
                ):
                    continue
                try:
                    priority_group = fields[12]
                    ack_sequence = int(fields[13])
                except (IndexError, ValueError):
                    malformed_lines += 1
                    continue

                key = FlowKey(
                    sender_rank=original_sender_rank,
                    receiver_rank=src_rank,
                    sender_ip=fields[7].lower(),
                    receiver_ip=fields[6].lower(),
                    sender_port=fields[9],
                    receiver_port=fields[8],
                    priority_group=priority_group,
                )
                histories[key].acks.append(
                    AckReceive(
                        timestamp=fields[0],
                        timestamp_value=timestamp_value,
                        line_number=line_number,
                        ack_sequence=ack_sequence,
                    )
                )
                ack_receive_events += 1

    return dict(histories), ScanStats(
        malformed_lines=malformed_lines,
        data_send_events=data_send_events,
        ack_receive_events=ack_receive_events,
    )


def calculate_rtt_samples(histories: Histories) -> Tuple[List[RttSample], int]:
    """把累计 ACK 与此前尚未确认的数据包匹配。"""
    samples: List[RttSample] = []
    duplicate_or_unmatched_acks = 0

    for flow, history in histories.items():
        timeline = []
        for send in history.sends:
            timeline.append((send.timestamp_value, send.line_number, 0, send))
        for ack in history.acks:
            timeline.append((ack.timestamp_value, ack.line_number, 1, ack))
        timeline.sort(key=lambda item: (item[0], item[1], item[2]))

        pending_by_end_sequence: Dict[int, List[DataSend]] = defaultdict(list)
        highest_ack_sequence = -1

        for _, _, event_kind, event in timeline:
            if event_kind == 0:
                send = event
                assert isinstance(send, DataSend)
                # trace 截断后可能出现已经被更早 ACK 覆盖的旧重传。
                if send.end_sequence > highest_ack_sequence:
                    pending_by_end_sequence[send.end_sequence].append(send)
                continue

            ack = event
            assert isinstance(ack, AckReceive)
            acknowledged_ends = sorted(
                end_sequence
                for end_sequence in pending_by_end_sequence
                if end_sequence <= ack.ack_sequence
            )
            if not acknowledged_ends:
                duplicate_or_unmatched_acks += 1
                highest_ack_sequence = max(highest_ack_sequence, ack.ack_sequence)
                continue

            for end_sequence in acknowledged_ends:
                attempts = pending_by_end_sequence.pop(end_sequence)
                # 一个 ACK 无法区分同一序号的多次发送；与丢包脚本一致，
                # 使用 ACK 前最近一次发送作为成功的发送尝试。
                send = attempts[-1]
                samples.append(
                    RttSample(
                        flow=flow,
                        send=send,
                        ack=ack,
                        retransmission_count=len(attempts) - 1,
                    )
                )
            highest_ack_sequence = max(highest_ack_sequence, ack.ack_sequence)

    samples.sort(
        key=lambda sample: (
            sample.ack.timestamp_value,
            sample.ack.line_number,
            sample.flow.sender_rank,
            sample.send.line_number,
        )
    )
    return samples, duplicate_or_unmatched_acks


def write_csv(output: Path, samples: List[RttSample]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "ack_receive_time",
                "rank",
                "rtt",
                "send_time",
                "receiver_rank",
                "sender_ip",
                "receiver_ip",
                "sender_port",
                "receiver_port",
                "priority_group",
                "packet_sequence",
                "packet_end_sequence",
                "ack_sequence",
                "payload_bytes",
                "retransmission_count",
                "send_line",
                "ack_receive_line",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    sample.ack.timestamp,
                    sample.flow.sender_rank,
                    sample.rtt,
                    sample.send.timestamp,
                    sample.flow.receiver_rank,
                    sample.flow.sender_ip,
                    sample.flow.receiver_ip,
                    sample.flow.sender_port,
                    sample.flow.receiver_port,
                    sample.flow.priority_group,
                    sample.send.sequence,
                    sample.send.end_sequence,
                    sample.ack.ack_sequence,
                    sample.send.payload_bytes,
                    sample.retransmission_count,
                    sample.send.line_number,
                    sample.ack.line_number,
                ]
            )


def print_report(
    trace: Path,
    output: Path,
    selected_senders: List[int],
    histories: Histories,
    samples: List[RttSample],
    stats: ScanStats,
    duplicate_or_unmatched_acks: int,
    max_display: int,
) -> None:
    print("RTT 分析")
    print(f"Trace: {trace}")
    print(f"发送方范围: {','.join(str(rank) for rank in selected_senders)}")

    send_count_by_rank: Dict[int, int] = defaultdict(int)
    sample_count_by_rank: Dict[int, int] = defaultdict(int)
    for flow, history in histories.items():
        send_count_by_rank[flow.sender_rank] += len(history.sends)
    for sample in samples:
        sample_count_by_rank[sample.flow.sender_rank] += 1

    for rank in selected_senders:
        print(
            f"rank {rank}: 数据发送事件 {send_count_by_rank[rank]}, "
            f"RTT 样本 {sample_count_by_rank[rank]}"
        )

    print(f"合计 RTT 样本: {len(samples)}")
    if duplicate_or_unmatched_acks:
        print(
            f"提示: {duplicate_or_unmatched_acks} 个 ACK 没有产生新的 RTT 样本"
            "（可能是重复 ACK、trace 被截断或发送事件不在筛选范围内）。"
        )
    if stats.malformed_lines:
        print(f"提示: 跳过 {stats.malformed_lines} 行无法解析的非空记录。")

    shown = samples if max_display == 0 else samples[:max_display]
    if shown:
        print("\n按 ACK 接收时间排序的 RTT：")
        for sample in shown:
            print(
                f"- ack_t={sample.ack.timestamp} rank={sample.flow.sender_rank} "
                f"rtt={sample.rtt} seq={sample.send.sequence} "
                f"ack={sample.ack.ack_sequence}"
            )
        if len(shown) < len(samples):
            print(f"... 还有 {len(samples) - len(shown)} 条，完整内容见 CSV。")

    print(f"\nRTT CSV 已写入: {output}")
    print("时间和 RTT 的单位与 trace 第一列一致。")


def finish_analysis(
    trace: Path,
    output: Path,
    selected_senders: List[int],
    analyzer: RttAnalyzer,
    malformed_lines: int,
    max_display: int,
) -> int:
    samples, duplicate_or_unmatched_acks = calculate_rtt_samples(
        dict(analyzer.histories)
    )
    write_csv(output, samples)
    stats = ScanStats(
        malformed_lines=malformed_lines,
        data_send_events=analyzer.data_send_events,
        ack_receive_events=analyzer.ack_receive_events,
    )
    print_report(
        trace,
        output,
        selected_senders,
        dict(analyzer.histories),
        samples,
        stats,
        duplicate_or_unmatched_acks,
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
    sender_filter = set(requested_ranks) if requested_ranks else None
    analyzer = RttAnalyzer(sender_filter)
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

    selected_senders = requested_ranks or sorted(reader.seen_ranks)
    output = args.output or args.trace.with_name(f"{args.trace.stem}_rtt.csv")
    status = finish_analysis(
        trace=args.trace,
        output=output,
        selected_senders=selected_senders,
        analyzer=analyzer,
        malformed_lines=reader.malformed_lines,
        max_display=args.max_display,
    )
    print(f"共享事件解析耗时: {parse_seconds:.3f} 秒")
    print(f"RTT 分析总耗时: {time.perf_counter() - total_start:.3f} 秒")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
