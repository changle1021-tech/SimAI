#!/usr/bin/env python3
"""解析 trace，并完成丢包、RTT、排队时延与 PFC 状态分析。"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set

import check_trace_packet_loss
import check_trace_pfc_state
import check_trace_queue_delay
import check_trace_rtt
from trace_events import TraceEvent, TraceEventReader


DEFAULT_QUEUE_SIZE = 16
DEFAULT_BATCH_SIZE = 2048
DEFAULT_TRACE_NAME = "trace.txt"


@dataclass(frozen=True)
class AnalysisContext:
    seen_ranks: Set[int]
    selected_senders: List[int]
    malformed_lines: int


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "解析 trace，分析丢包、RTT 和排队时延，和PFC状态变化。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  %(prog)s trace.txt\n"
            "  %(prog)s trace.txt 0 24\n"
            "  %(prog)s Megatron --workload "
            "A100-Megatron-world_size32-tp4-pp2-ep2-gbs4-mbs1-seq512-"
            "MOE-True-GEMM-False-flash_attn-False\n"
            "  %(prog)s results/Megatron/<负载名>/\n"
            "  %(prog)s trace.txt --loss-output loss.csv --rtt-output rtt.csv "
            "--queue-delay-output queue.csv --pfc-state-output pfc.csv"
        ),
    )
    parser.add_argument(
        "trace",
        type=Path,
        help=(
            "待分析的 trace.txt、包含 trace.txt 的结果目录，或 results 下的场景名"
        ),
    )
    parser.add_argument(
        "ranks",
        type=int,
        nargs="*",
        metavar="RANK",
        help="可选的发送方 rank；不指定时分析全部 rank",
    )
    parser.add_argument("--loss-output", type=Path, help="丢包 CSV 输出路径")
    parser.add_argument("--rtt-output", type=Path, help="RTT CSV 输出路径")
    parser.add_argument(
        "--queue-delay-output", type=Path, help="排队时延 CSV 输出路径"
    )
    parser.add_argument(
        "--pfc-state-output", type=Path, help="PFC 状态变化 CSV 输出路径"
    )
    parser.add_argument(
        "--packet-types",
        help="丢包检查只核对这些包类型，逗号分隔；默认核对全部类型",
    )
    parser.add_argument(
        "--max-display",
        type=int,
        default=20,
        help="四项分析各自在终端最多显示多少条记录；0 表示全部显示",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=DEFAULT_QUEUE_SIZE,
        help=f"每个分析进程的批次队列容量；默认 {DEFAULT_QUEUE_SIZE}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"每次跨进程传递的事件数；默认 {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--fail-on-loss",
        action="store_true",
        help="发现丢包时在全部分析完成后返回退出码 1",
    )
    parser.add_argument(
        "--workload",
        help=(
            "负载名；指定后，第一个位置参数按场景名解析，读取 "
            "<results-dir>/<场景名>/<负载名>/trace.txt"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="results 目录；默认是本脚本所在目录",
    )
    return parser.parse_args(argv)


def resolve_trace_path(
    target: Path, results_dir: Path, workload: Optional[str]
) -> Path:
    """兼容直接路径、结果目录以及新的场景/负载目录结构。"""
    target = target.expanduser()
    results_dir = results_dir.expanduser()

    if workload is not None:
        if (
            target.is_absolute()
            or len(target.parts) != 1
            or target.name in {"", ".", ".."}
        ):
            raise ValueError("使用 --workload 时，第一个位置参数必须是场景名")
        workload_path = Path(workload)
        if (
            workload_path.is_absolute()
            or len(workload_path.parts) != 1
            or workload_path.name in {"", ".", ".."}
        ):
            raise ValueError("--workload 必须是负载名，不能包含路径")
        return results_dir / target.name / workload / DEFAULT_TRACE_NAME

    if target.is_file():
        return target
    if target.is_dir():
        direct_trace = target / DEFAULT_TRACE_NAME
        if direct_trace.is_file():
            return direct_trace

    results_target = results_dir / target
    if results_target.is_file():
        return results_target
    if results_target.is_dir():
        direct_trace = results_target / DEFAULT_TRACE_NAME
        if direct_trace.is_file():
            return direct_trace

        workload_traces = sorted(results_target.glob(f"*/{DEFAULT_TRACE_NAME}"))
        if len(workload_traces) == 1:
            return workload_traces[0]
        if len(workload_traces) > 1:
            workloads = "\n".join(
                f"  {trace.parent.name}" for trace in workload_traces
            )
            raise ValueError(
                f"场景 {target} 下有多个负载，请使用 --workload 指定：\n{workloads}"
            )

    return target


def loss_worker(
    event_queue: mp.Queue,
    result_queue: mp.Queue,
    sender_filter: Optional[Set[int]],
    packet_types: Optional[Set[str]],
    trace: Path,
    output: Path,
    max_display: int,
    fail_on_loss: bool,
) -> None:
    try:
        analyzer = check_trace_packet_loss.PacketLossAnalyzer(
            sender_filter, packet_types
        )
        while True:
            item = event_queue.get()
            if item is None:
                return
            if isinstance(item, AnalysisContext):
                context = item
                break
            for event in item:
                analyzer.consume(event)

        print("=== 丢包检查（子进程） ===")
        status = check_trace_packet_loss.finish_analysis(
            trace=trace,
            output=output,
            seen_ranks=context.seen_ranks,
            selected_senders=context.selected_senders,
            analyzer=analyzer,
            malformed_lines=context.malformed_lines,
            max_display=max_display,
            fail_on_loss=fail_on_loss,
        )
        result_queue.put(("loss", status, ""))
    except BaseException:
        result_queue.put(("loss", 2, traceback.format_exc()))


def rtt_worker(
    event_queue: mp.Queue,
    result_queue: mp.Queue,
    sender_filter: Optional[Set[int]],
    trace: Path,
    output: Path,
    max_display: int,
) -> None:
    try:
        analyzer = check_trace_rtt.RttAnalyzer(sender_filter)
        while True:
            item = event_queue.get()
            if item is None:
                return
            if isinstance(item, AnalysisContext):
                context = item
                break
            for event in item:
                analyzer.consume(event)

        print("=== RTT 分析（子进程） ===")
        status = check_trace_rtt.finish_analysis(
            trace=trace,
            output=output,
            selected_senders=context.selected_senders,
            analyzer=analyzer,
            malformed_lines=context.malformed_lines,
            max_display=max_display,
        )
        result_queue.put(("rtt", status, ""))
    except BaseException:
        result_queue.put(("rtt", 2, traceback.format_exc()))


def queue_delay_worker(
    event_queue: mp.Queue,
    result_queue: mp.Queue,
    rank_filter: Optional[Set[int]],
    trace: Path,
    output: Path,
    max_display: int,
) -> None:
    try:
        analyzer = check_trace_queue_delay.QueueDelayAnalyzer(rank_filter)
        while True:
            item = event_queue.get()
            if item is None:
                return
            if isinstance(item, AnalysisContext):
                context = item
                break
            for event in item:
                analyzer.consume(event)

        print("=== 排队时延分析（子进程） ===")
        status = check_trace_queue_delay.finish_analysis(
            trace=trace,
            output=output,
            selected_ranks=context.selected_senders,
            analyzer=analyzer,
            malformed_lines=context.malformed_lines,
            max_display=max_display,
        )
        result_queue.put(("queue_delay", status, ""))
    except BaseException:
        result_queue.put(("queue_delay", 2, traceback.format_exc()))


def put_checked(target_queue: mp.Queue, item: object, process: mp.Process) -> None:
    """带子进程存活检查地写入有界队列，避免子进程异常后永久阻塞。"""
    while True:
        if not process.is_alive():
            raise RuntimeError(f"分析子进程 {process.name} 已异常退出")
        try:
            target_queue.put(item, timeout=1.0)
            return
        except queue.Full:
            continue


def stop_processes(processes: Sequence[mp.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.max_display < 0:
        print("错误: --max-display 不能小于 0。", file=sys.stderr)
        return 2
    if args.queue_size <= 0:
        print("错误: --queue-size 必须大于 0。", file=sys.stderr)
        return 2
    if args.batch_size <= 0:
        print("错误: --batch-size 必须大于 0。", file=sys.stderr)
        return 2
    try:
        args.trace = resolve_trace_path(
            args.trace, args.results_dir, args.workload
        )
    except ValueError as error:
        print(f"错误: {error}", file=sys.stderr)
        return 2
    if not args.trace.is_file():
        print(f"错误: trace 文件不存在: {args.trace}", file=sys.stderr)
        return 2
    if any(rank < 0 for rank in args.ranks):
        print("错误: rank 不能为负数。", file=sys.stderr)
        return 2
    try:
        packet_types = check_trace_packet_loss.parse_packet_types(
            args.packet_types
        )
    except ValueError as error:
        print(f"错误: {error}", file=sys.stderr)
        return 2

    requested_ranks = sorted(set(args.ranks))
    sender_filter = set(requested_ranks) if requested_ranks else None
    loss_output = args.loss_output or args.trace.with_name(
        f"{args.trace.stem}_packet_loss.csv"
    )
    rtt_output = args.rtt_output or args.trace.with_name(
        f"{args.trace.stem}_rtt.csv"
    )
    queue_delay_output = args.queue_delay_output or args.trace.with_name(
        f"{args.trace.stem}_queue_delay.csv"
    )
    pfc_state_output = args.pfc_state_output or args.trace.with_name(
        f"{args.trace.stem}_pfc_state.csv"
    )
    pfc_analyzer = check_trace_pfc_state.PfcStateAnalyzer(sender_filter)

    context = mp.get_context("spawn")
    loss_queue = context.Queue(maxsize=args.queue_size)
    rtt_queue = context.Queue(maxsize=args.queue_size)
    queue_delay_queue = context.Queue(maxsize=args.queue_size)
    result_queue = context.Queue()
    loss_process = context.Process(
        name="packet-loss-analyzer",
        target=loss_worker,
        args=(
            loss_queue,
            result_queue,
            sender_filter,
            packet_types,
            args.trace,
            loss_output,
            args.max_display,
            args.fail_on_loss,
        ),
    )
    rtt_process = context.Process(
        name="rtt-analyzer",
        target=rtt_worker,
        args=(
            rtt_queue,
            result_queue,
            sender_filter,
            args.trace,
            rtt_output,
            args.max_display,
        ),
    )
    queue_delay_process = context.Process(
        name="queue-delay-analyzer",
        target=queue_delay_worker,
        args=(
            queue_delay_queue,
            result_queue,
            sender_filter,
            args.trace,
            queue_delay_output,
            args.max_display,
        ),
    )
    processes = [loss_process, rtt_process, queue_delay_process]
    for process in processes:
        process.start()

    total_start = time.perf_counter()
    reader = TraceEventReader(args.trace)
    try:
        batch = []
        for event in reader:
            pfc_analyzer.consume(event)
            batch.append(event)
            if len(batch) >= args.batch_size:
                put_checked(loss_queue, batch, loss_process)
                put_checked(rtt_queue, batch, rtt_process)
                put_checked(queue_delay_queue, batch, queue_delay_process)
                batch = []
        if batch:
            put_checked(loss_queue, batch, loss_process)
            put_checked(rtt_queue, batch, rtt_process)
            put_checked(queue_delay_queue, batch, queue_delay_process)

        if not reader.seen_ranks:
            print("错误: trace 中没有找到任何 n:<rank> 事件。", file=sys.stderr)
            stop_processes(processes)
            return 2
        missing_ranks = sorted(set(requested_ranks) - reader.seen_ranks)
        if missing_ranks:
            print(f"错误: trace 中没有 rank 事件: {missing_ranks}", file=sys.stderr)
            stop_processes(processes)
            return 2

        analysis_context = AnalysisContext(
            seen_ranks=reader.seen_ranks,
            selected_senders=requested_ranks or sorted(reader.seen_ranks),
            malformed_lines=reader.malformed_lines,
        )
        put_checked(loss_queue, analysis_context, loss_process)
        put_checked(rtt_queue, analysis_context, rtt_process)
        put_checked(queue_delay_queue, analysis_context, queue_delay_process)

        print("=== PFC 状态变化分析（主进程） ===")
        pfc_status = check_trace_pfc_state.finish_analysis(
            trace=args.trace,
            output=pfc_state_output,
            selected_ranks=analysis_context.selected_senders,
            analyzer=pfc_analyzer,
            malformed_lines=reader.malformed_lines,
            max_display=args.max_display,
        )

        results = {}
        while len(results) < 3:
            try:
                name, status, error = result_queue.get(timeout=1.0)
                results[name] = status
                if error:
                    print(error, file=sys.stderr)
            except queue.Empty:
                if not any(process.is_alive() for process in processes):
                    break
        for process in processes:
            process.join()

        if len(results) != 3:
            print("错误: 至少一个分析子进程未返回结果。", file=sys.stderr)
            return 2
        print(f"\n单次解析及并行分析总耗时: {time.perf_counter() - total_start:.3f} 秒")
        if (
            results["loss"] == 2
            or results["rtt"] == 2
            or results["queue_delay"] == 2
            or pfc_status == 2
        ):
            return 2
        return results["loss"]
    except BaseException:
        stop_processes(processes)
        raise
    finally:
        loss_queue.close()
        rtt_queue.close()
        queue_delay_queue.close()
        result_queue.close()


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
