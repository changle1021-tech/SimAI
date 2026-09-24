import subprocess

for prefill_tokens in range(512, 16384 + 1, 512):
    print("=" * 80)
    print(f"Running prefill_tokens = {prefill_tokens}")
    print("=" * 80)

    command = f"""
cd /home/turbo_ops/changle/SimAI/vidur-alibabacloud

CACHE_DIR="./data/aicb_workload/cache"

mkdir -p "${{CACHE_DIR}}"
find "${{CACHE_DIR}}" \
  -maxdepth 1 \
  -type f \
  -name 'aicb-*.json' \
  -delete

CUDA_VISIBLE_DEVICES=6 python -m vidur.main \
  --replica_config_network_device h100_dgx \
  --replica_config_device h100 \
  --replica_config_nvlink_bandwidth 3600 \
  --replica_config_rdma_bandwidth 400 \
  --replica_config_pd_p2p_comm_bandwidth 400 \
  --poisson_request_interval_generator_config_qps 512 \
  --synthetic_request_generator_config_num_requests 1 \
  --length_generator_config_type fixed \
  --fixed_request_length_generator_config_prefill_tokens {prefill_tokens} \
  --fixed_request_length_generator_config_decode_tokens 256 \
  --fixed_request_length_generator_config_max_tokens 32768 \
  --interval_generator_config_type poisson \
  --cluster_config_num_replicas 4 \
  --replica_config_pd_node_ratio 1 \
  --global_scheduler_config_type lor \
  --replica_scheduler_config_type vllm \
  --vllm_scheduler_config_max_tokens_in_batch 32768 \
  --vllm_scheduler_config_block_size 32 \
  --replica_config_model_name qwen3-next-80B \
  --replica_config_tensor_parallel_size 1 \
  --replica_config_num_pipeline_stages 1 \
  --replica_config_memory_margin_fraction 0.15 \
  --random_forrest_execution_time_predictor_config_backend aicb \
  --random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 32768 \
  --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 32768
"""

    subprocess.run(
        ["bash", "-c", command],
        check=True
    )

print("=" * 80)
print("All runs completed: 512 -> 16384, step = 512")
print("=" * 80)
