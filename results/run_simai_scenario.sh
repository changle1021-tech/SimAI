#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
用法:
  ./run_simai_scenario.sh <场景名> [负载名]

示例:
  ./run_simai_scenario.sh SingleLinkError
  ./run_simai_scenario.sh Normal
  ./run_simai_scenario.sh Normal my_workload

场景名对应本脚本所在 results 目录下的 <场景名> 子目录。
负载名可选，不要携带 .txt 后缀。默认为：
  A100-NORMAL-world_size32-tp4-pp2-ep2-gbs4-mbs1-seq512-MOE-True-GEMM-False-flash_attn-False
负载文件需位于对应的场景目录中。
脚本会在场景目录下创建与负载同名的结果目录。
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage >&2
    exit 2
fi

case "$1" in
    -h|--help)
        usage
        exit 0
        ;;
esac

scenario=$1
workload_arg="${2:-A100-RDMA_TP4_PP2_EP2_DP4_2GPU_PER_NODE-world_size32-tp4-pp2-ep2-gbs4-mbs1-seq2048-MOE-True-GEMM-False-flash_attn-False}"
if [[ ! "$scenario" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "错误: 场景名只能包含字母、数字、下划线和连字符: $scenario" >&2
    exit 2
fi

if [[ -z "$workload_arg" || "$workload_arg" == */* ]]; then
    echo "错误: 负载参数必须是场景目录中的文件名，不能包含路径: $workload_arg" >&2
    exit 2
fi

if [[ "$workload_arg" == *.txt ]]; then
    echo "错误: 负载名不要携带 .txt 后缀: $workload_arg" >&2
    exit 2
fi
workload_stem=$workload_arg
workload_name="${workload_stem}.txt"
if [[ -z "$workload_stem" || "$workload_stem" == "." || "$workload_stem" == ".." ]]; then
    echo "错误: 无效的负载名: $workload_arg" >&2
    exit 2
fi

# 脚本可以从任意工作目录启动；所有路径都基于脚本自身位置计算。
results_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "$results_dir/.." && pwd)
result_dir="$results_dir/$scenario"
run_dir="$result_dir/$workload_stem"
parameter_dir="$run_dir/parameters"

topology_name="Spectrum-X_32g_2gps_100Gbps_A100"
simulator="$project_dir/bin/SimAI_simulator"
trace_reader_dir="$project_dir/ns-3-alibabacloud/analysis"
trace_reader="$trace_reader_dir/trace_reader"
binary_trace="$run_dir/astra-sim/simulation/llama_hpn7_mix.tr"
text_trace="$run_dir/trace.txt"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "错误: 缺少文件: $1" >&2
        exit 2
    fi
}

require_executable() {
    if [[ ! -x "$1" ]]; then
        echo "错误: 文件不存在或不可执行: $1" >&2
        exit 2
    fi
}

if [[ ! -d "$result_dir" ]]; then
    echo "错误: 场景目录不存在: $result_dir" >&2
    echo "可用场景目录：" >&2
    find "$results_dir" -mindepth 1 -maxdepth 1 -type d -printf '  %f\n' | sort >&2
    exit 2
fi

require_executable "$simulator"
require_executable "$trace_reader"
require_file "$result_dir/$workload_name"
require_file "$result_dir/$topology_name"
require_file "$result_dir/SimAI.conf"

echo "[1/4] 准备结果目录和输入参数"
mkdir -p "$parameter_dir"
cp -f -- "$result_dir/$workload_name" "$parameter_dir/"
cp -f -- "$result_dir/$topology_name" "$parameter_dir/"
cp -f -- "$result_dir/SimAI.conf" "$parameter_dir/"
rm -f /etc/astra-sim/SimAI.log

echo "[2/4] 运行场景: $scenario"
cd "$project_dir"
AS_SEND_LAT=3 AS_NVLS_ENABLE=1 AS_LOG_LEVEL=DEBUG \
    "$simulator" \
    -t 1 \
    -w "$result_dir/$workload_name" \
    -n "$result_dir/$topology_name" \
    -c "$result_dir/SimAI.conf"

require_file "$project_dir/ncclFlowModel_EndToEnd.csv"
require_file "$project_dir/ncclFlowModel_test1_dimension_utilization_0.csv"
if [[ ! -d /etc/astra-sim ]]; then
    echo "错误: 模拟结束后未找到 /etc/astra-sim" >&2
    exit 1
fi

echo "[3/4] 保存 CSV 和 astra-sim 输出到 $run_dir"
cp -f "$project_dir/ncclFlowModel_EndToEnd.csv" "$run_dir/"
cp -f "$project_dir/ncclFlowModel_test1_dimension_utilization_0.csv" "$run_dir/"
cp -r /etc/astra-sim "$run_dir/"

require_file "$binary_trace"

echo "[4/4] 解析二进制 trace"
cd "$trace_reader_dir"
"$trace_reader" "$binary_trace" > "$text_trace"

echo
echo "完成: $scenario"
echo "  Workload: $result_dir/$workload_name"
echo "  参数目录: $parameter_dir"
echo "  结果目录: $run_dir"
echo "  文本 trace: $text_trace"
