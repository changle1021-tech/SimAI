#!/usr/bin/env bash
set -euo pipefail
QWEN_PROFILE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QWEN_REPO_ROOT="$(git -C "$QWEN_PROFILE_ROOT" rev-parse --show-toplevel)"
cd "$QWEN_REPO_ROOT/vidur-alibabacloud"
export CUDA_VISIBLE_DEVICES=''
exec "${VIDUR_PYTHON:-python}" -m vidur.main \
  --no-metrics_config_store_plots \
  --no-metrics_config_enable_chrome_trace \
  --replica_config_network_device h100_dgx \
  --replica_config_device h100 \
  --replica_config_nvlink_bandwidth 3600 \
  --poisson_request_interval_generator_config_qps 512 \
  --synthetic_request_generator_config_num_requests 1 \
  --length_generator_config_type fixed \
  --fixed_request_length_generator_config_max_tokens 32768 \
  --interval_generator_config_type poisson \
  --cluster_config_num_replicas 4 \
  --replica_config_pd_node_ratio 1 \
  --global_scheduler_config_type lor \
  --replica_scheduler_config_type sarathi \
  --sarathi_scheduler_config_chunk_size 512 \
  --replica_config_model_name qwen3-next-80B \
  --replica_config_tensor_parallel_size 1 \
  --replica_config_num_pipeline_stages 1 \
  --random_forrest_execution_time_predictor_config_backend aicb \
  --random_forrest_execution_time_predictor_config_prediction_max_batch_size 1 \
  --random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 9216 \
  --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 512 \
  --no-random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling \
  --random_forrest_execution_time_predictor_config_aicb_profile_dir "$QWEN_PROFILE_ROOT/phase_profile_v5/workload" \
  --random_forrest_execution_time_predictor_config_aicb_cpu_overhead_profile "$QWEN_PROFILE_ROOT/cpu_boundary_profile_v3.json" \
  --random_forrest_execution_time_predictor_config_aicb_moe_comm_profile "$QWEN_PROFILE_ROOT/comm_profile_v2.json" \
  --fixed_request_length_generator_config_prefill_tokens 512 \
  --fixed_request_length_generator_config_decode_tokens 256 \
  --metrics_config_output_dir "${QWEN_ALIGNMENT_OUTPUT_DIR:-$QWEN_REPO_ROOT/vidur-alibabacloud/benchmark_runs/qwen80b_alignment_20261018}"
