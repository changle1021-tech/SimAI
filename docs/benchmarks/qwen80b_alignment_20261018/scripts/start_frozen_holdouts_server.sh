#!/usr/bin/env bash
set -eu
export PYTHONPATH=/home/turbo_ops/changle/Files/analysis/qwen80b_alignment_20261016
export VLLM_SERVER_DEV_MODE=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
exec /home/turbo_ops/miniconda3/envs/vllm011/bin/python -u -m vllm.entrypoints.openai.api_server \
  --model /mnt/data02/000000/model/Qwen3-Next-80B-A3B-Instruct \
  --served-model-name qwen80b-audit \
  --dtype bfloat16 --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
  --max-model-len 9216 --max-num-batched-tokens 512 --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 --no-enable-prefix-caching \
  --port 18180 --host 127.0.0.1 --seed 42 --disable-log-requests
