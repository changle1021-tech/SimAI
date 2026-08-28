#!/bin/sh

# Run every SimAI inference workload in this 2Nodes scenario.
# Cross-node TP8 is not supported by the current NVLS flow-model generator, so
# NVLS is disabled by default. Set AS_NVLS_ENABLE=1 explicitly to override it.
# Usage: ./results/2Nodes/run_all_workloads.sh [TOPOLOGY_FILE]

set -u

SCENARIO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCENARIO_DIR/../.." && pwd)
RUNNER=${SIMAI_RUNNER:-"$REPO_ROOT/results/run_simai_inference_with_topology.sh"}
TOPOLOGY=${1:-"$SCENARIO_DIR/DCN+SingleToR_4g_2gps_400Gbps_H100"}
NVLS_ENABLE=${AS_NVLS_ENABLE:-0}

if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [TOPOLOGY_FILE]" >&2
  exit 2
fi

case $NVLS_ENABLE in
  0|1) ;;
  *)
    echo "AS_NVLS_ENABLE must be 0 or 1: $NVLS_ENABLE" >&2
    exit 2
    ;;
esac

case $TOPOLOGY in
  /*) ;;
  *) TOPOLOGY="$REPO_ROOT/$TOPOLOGY" ;;
esac

if [ ! -f "$RUNNER" ]; then
  echo "SimAI runner not found: $RUNNER" >&2
  echo "Set SIMAI_RUNNER to override its location." >&2
  exit 2
fi

if [ ! -f "$TOPOLOGY" ]; then
  echo "Topology file not found: $TOPOLOGY" >&2
  exit 2
fi

set -- "$SCENARIO_DIR"/*.txt
if [ ! -e "$1" ]; then
  echo "No workload .txt files found in: $SCENARIO_DIR" >&2
  exit 2
fi

total=$#
passed=0
failed=0
index=0

cd "$REPO_ROOT" || exit 2

echo "AS_NVLS_ENABLE=$NVLS_ENABLE"

for workload do
  index=$((index + 1))
  echo "[$index/$total] Running: $(basename -- "$workload")"
  if AS_NVLS_ENABLE="$NVLS_ENABLE" "$RUNNER" -n "$TOPOLOGY" -w "$workload"; then
    passed=$((passed + 1))
  else
    status=$?
    failed=$((failed + 1))
    echo "[$index/$total] FAILED (exit $status): $(basename -- "$workload")" >&2
  fi
done

echo "Completed: total=$total passed=$passed failed=$failed"

if [ "$failed" -ne 0 ]; then
  exit 1
fi
