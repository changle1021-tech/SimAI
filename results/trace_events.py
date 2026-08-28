#!/usr/bin/env python3
"""trace 文本的共享流式事件解析器。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Set, Tuple


@dataclass(frozen=True)
class TraceEvent:
    timestamp: str
    timestamp_value: int
    line_number: int
    event_rank: int
    port_queue: str
    queue_length: str
    action: str
    ecn: str
    src_rank: Optional[int]
    dst_rank: Optional[int]
    src_ip: str
    dst_ip: str
    src_port: str
    dst_port: str
    packet_type: str
    details: Tuple[str, ...]


def address_rank(address: str) -> int:
    """解析形如 0b001801 的地址；末尾两位是主机地址，前面是 rank。"""
    value = address.lower()
    if not value.startswith("0b") or len(value) <= 4:
        raise ValueError(address)
    return int(value[2:-2], 16)


class TraceEventReader:
    """逐行解析 trace；迭代结束后提供 rank 与异常行统计。"""

    def __init__(self, trace: Path) -> None:
        self.trace = trace
        self.seen_ranks: Set[int] = set()
        self.malformed_lines = 0
        self.parsed_events = 0

    def __iter__(self) -> Iterator[TraceEvent]:
        with self.trace.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                fields = line.split()
                if len(fields) < 11:
                    if fields:
                        self.malformed_lines += 1
                        self._remember_rank(fields)
                    continue

                node_token = fields[1]
                if not node_token.startswith("n:"):
                    self.malformed_lines += 1
                    continue
                try:
                    event_rank = int(node_token[2:])
                    timestamp_value = int(fields[0])
                except (IndexError, ValueError):
                    self.malformed_lines += 1
                    continue

                # PFC 行没有四层端口，格式为：
                # ... sip dip P pause_time advertised_qlen queue_index size
                is_pfc = len(fields) >= 13 and fields[8] == "P"
                if is_pfc:
                    src_rank = None
                    dst_rank = None
                    src_port = ""
                    dst_port = ""
                    packet_type = "P"
                    details = tuple(fields[9:])
                else:
                    try:
                        src_rank = address_rank(fields[6])
                        dst_rank = address_rank(fields[7])
                    except (IndexError, ValueError):
                        self.malformed_lines += 1
                        continue
                    src_port = fields[8]
                    dst_port = fields[9]
                    packet_type = fields[10]
                    details = tuple(fields[11:])

                self.seen_ranks.add(event_rank)
                self.parsed_events += 1
                yield TraceEvent(
                    timestamp=fields[0],
                    timestamp_value=timestamp_value,
                    line_number=line_number,
                    event_rank=event_rank,
                    port_queue=fields[2],
                    queue_length=fields[3],
                    action=fields[4],
                    ecn=fields[5],
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                    src_ip=fields[6].lower(),
                    dst_ip=fields[7].lower(),
                    src_port=src_port,
                    dst_port=dst_port,
                    packet_type=packet_type,
                    details=details,
                )

    def _remember_rank(self, fields: list[str]) -> None:
        if len(fields) < 2 or not fields[1].startswith("n:"):
            return
        try:
            self.seen_ranks.add(int(fields[1][2:]))
        except ValueError:
            pass
