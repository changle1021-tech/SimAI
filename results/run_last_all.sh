#!/usr/bin/env bash

set -Eeuo pipefail

runner="./results/run_simai_inference_with_topology.sh"
topology_4g="results/1600gbps/DCN+SingleToR_4g_4gps_400Gbps_H100"
topology_8g="results/1600gbps/DCN+SingleToR_8g_4gps_400Gbps_H100"


# world_size8 workloads: use the 8g topology.
for workload in results/1600gbps/Qwen3-Next-80B-world_size8-*.txt; do
    "$runner" -n "$topology_8g" -w "$workload"
done

echo "全部 28 个负载已逐个执行完成。"
