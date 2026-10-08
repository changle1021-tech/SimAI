# 新版 MLP / Attention 采集命令

分支：`codex/pp-rollback-20261002`。以下为 Linux/Bash 命令，除 Docker 示例外，在仓库的 `vidur-alibabacloud` 目录执行。

## 1. 采集环境

使用独立的 GPU Python 环境，或已经包含 vLLM 0.5.1 的容器：

```bash
python -m pip install -r requirements-profiling.txt
python -m pip install -e . --no-deps
```

CPU 仿真继续使用原来的 Vidur 环境；只有 GPU 采集需要 vLLM。模型按现有 `vidur/config/model_config.py` 注册表选择，不需要下载模型权重。

## 2. 单卡：全部默认模型和 TP 分片

GPU 编号替换为实际用于采集的卡。`--disable_ray` 使同一个采集实现直接在当前进程执行。

```bash
CUDA_VISIBLE_DEVICES=0 python -m vidur.profiling.mlp.main \
  --disable_ray \
  --num_tensor_parallel_workers 1 2 4 8 \
  --max_tokens 4096 \
  --output_dir profiling_outputs

CUDA_VISIBLE_DEVICES=0 python -m vidur.profiling.attention.main \
  --disable_ray \
  --num_tensor_parallel_workers 1 2 4 8 \
  --max_model_len 4096 \
  --max_seq_len 4096 \
  --min_batch_size 1 \
  --max_batch_size 8 \
  --block_size 16 \
  --output_dir profiling_outputs
```

不写 `--models` 时，两者都采集以下默认模型：

```text
microsoft/phi-2
internlm/internlm-20b
Qwen/Qwen-72B
meta-llama/Llama-2-7b-hf
codellama/CodeLlama-34b-Instruct-hf
meta-llama/Llama-2-70b-hf
meta-llama/Meta-Llama-3-8B
meta-llama/Meta-Llama-3-70B
```

Phi-2 保留模型配置中的非 TP 限制，自动跳过 TP>1。这里的 TP 参数决定单卡分片的算子形状，不会执行多卡 AllReduce。

## 3. 指定模型或只采集某个 TP

```bash
CUDA_VISIBLE_DEVICES=0 python -m vidur.profiling.mlp.main \
  --disable_ray \
  --models meta-llama/Llama-2-7b-hf internlm/internlm-20b \
  --num_tensor_parallel_workers 1 \
  --max_tokens 4096

CUDA_VISIBLE_DEVICES=0 python -m vidur.profiling.attention.main \
  --disable_ray \
  --models meta-llama/Llama-2-7b-hf internlm/internlm-20b \
  --num_tensor_parallel_workers 1 \
  --max_model_len 4096 --max_seq_len 4096 \
  --max_batch_size 8 --block_size 16
```

## 4. Attention 只采集 prefill 或 decode

这两个开关互斥。完整仿真输入表需要包含两种阶段；单独采集用于分项分析或之后合并。

```bash
CUDA_VISIBLE_DEVICES=0 python -m vidur.profiling.attention.main \
  --disable_ray --models meta-llama/Llama-2-7b-hf \
  --num_tensor_parallel_workers 1 \
  --max_model_len 4096 --max_seq_len 4096 \
  --profile_only_prefill

CUDA_VISIBLE_DEVICES=0 python -m vidur.profiling.attention.main \
  --disable_ray --models meta-llama/Llama-2-7b-hf \
  --num_tensor_parallel_workers 1 \
  --max_model_len 4096 --max_seq_len 4096 \
  --min_batch_size 1 --max_batch_size 8 \
  --profile_only_decode
```

## 5. Ray 多 GPU 并行采集

不传 `--disable_ray`，并让 `--num_gpus` 与可见的 GPU 数量匹配。例如：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vidur.profiling.mlp.main \
  --num_gpus 4 --models meta-llama/Llama-2-7b-hf internlm/internlm-20b \
  --num_tensor_parallel_workers 1 2 4 8 --max_tokens 4096

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vidur.profiling.attention.main \
  --num_gpus 4 --models meta-llama/Llama-2-7b-hf internlm/internlm-20b \
  --num_tensor_parallel_workers 1 2 4 8 \
  --max_model_len 4096 --max_seq_len 4096 --max_batch_size 8
```

## 6. sim-101 上用现有容器采集

使用已验证的镜像，不依赖宿主机是否安装 vLLM：

```bash
SRC=/home/turbo_ops/changle/SimAI-pp-rollback-fix-20261008/vidur-alibabacloud
OUT=/home/turbo_ops/changle/Files/profiling_outputs
GPU_ID=5
mkdir -p "$OUT"

sudo docker run --rm --gpus "device=$GPU_ID" --ipc host \
  -e CUDA_VISIBLE_DEVICES=0 -e PYTHONPATH=/source -e KINETO_LOG_LEVEL=5 \
  -v "$SRC:/source:ro" -v "$OUT:/output" \
  --entrypoint python3 docker.io/library/vllm:v0.5.1-trace \
  -m vidur.profiling.mlp.main \
  --disable_ray --models meta-llama/Llama-2-7b-hf internlm/internlm-20b \
  --num_tensor_parallel_workers 1 2 4 8 --max_tokens 4096 --output_dir /output

sudo docker run --rm --gpus "device=$GPU_ID" --ipc host \
  -e CUDA_VISIBLE_DEVICES=0 -e PYTHONPATH=/source -e KINETO_LOG_LEVEL=5 \
  -v "$SRC:/source:ro" -v "$OUT:/output" \
  --entrypoint python3 docker.io/library/vllm:v0.5.1-trace \
  -m vidur.profiling.attention.main \
  --disable_ray --models meta-llama/Llama-2-7b-hf internlm/internlm-20b \
  --num_tensor_parallel_workers 1 2 4 8 \
  --max_model_len 4096 --max_seq_len 4096 --max_batch_size 8 \
  --block_size 16 --output_dir /output
```

## 7. 安装采集结果到原有路径

输出结构仍然是：

```text
profiling_outputs/mlp/<时间戳>/<模型名>/mlp.csv
profiling_outputs/attention/<时间戳>/<模型名>/attention.csv
```

将下面两个运行目录替换为实际输出目录：

```bash
MLP_RUN="profiling_outputs/mlp/实际时间戳"
ATTENTION_RUN="profiling_outputs/attention/实际时间戳"
MODEL="meta-llama/Llama-2-7b-hf"
DEST="data/profiling/compute/h100/$MODEL"

mkdir -p "$DEST"
cp "$MLP_RUN/$MODEL/mlp.csv" "$DEST/mlp.csv"
cp "$ATTENTION_RUN/$MODEL/attention.csv" "$DEST/attention.csv"
```

仿真器默认读取这两个路径，不需要新的 profile 参数。也可使用原有的 `--random_forrest_execution_time_predictor_config_compute_input_file` 和 `--random_forrest_execution_time_predictor_config_attention_input_file` 指向其他位置。

## 8. 用实际到达轨迹做 TP1/PP1 仿真

下面是 sim-101 上本次对应的模型、调度和 CPU 配置。先完成上一步，将新表装入当前仓库的数据目录。`TRACE` 可替换为自己的实际请求轨迹。

```bash
TRACE=/home/turbo_ops/changle/Files/analysis/tp_pp_alignment_20261007/tp_pp1_paired_v16/7b_tp1_pp1_before_r0_b1_p512/arrival_trace.csv
CPU_PROFILE=/home/turbo_ops/changle/Files/vidur_cpu_overhead/cpu_overheads.csv

python -m vidur.main \
  --seed 36 \
  --replica_config_model_name meta-llama/Llama-2-7b-hf \
  --replica_config_device h100 --replica_config_network_device h100_dgx \
  --replica_config_tensor_parallel_size 1 \
  --replica_config_num_pipeline_stages 1 --replica_config_pd_node_ratio 1 \
  --cluster_config_num_replicas 1 \
  --global_scheduler_config_type round_robin \
  --replica_scheduler_config_type vllm \
  --vllm_scheduler_config_batch_size_cap 8 \
  --vllm_scheduler_config_block_size 16 \
  --vllm_scheduler_config_max_tokens_in_batch 4096 \
  --request_generator_config_type trace_replay \
  --trace_request_generator_config_trace_file "$TRACE" \
  --trace_request_generator_config_max_tokens 4096 \
  --random_forrest_execution_time_predictor_config_backend vidur \
  --random_forrest_execution_time_predictor_config_cpu_overhead_input_file "$CPU_PROFILE" \
  --no-random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling \
  --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 4096 \
  --random_forrest_execution_time_predictor_config_prediction_max_batch_size 8 \
  --random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 4096 \
  --metrics_config_cache_dir .cache/vidur \
  --metrics_config_output_dir simulator_output \
  --no-metrics_config_store_plots --no-metrics_config_enable_chrome_trace
```

## 9. 帮助和测试

```bash
python -m vidur.profiling.mlp.main --help
python -m vidur.profiling.attention.main --help
python -m unittest discover -s tests -v
```

### 参数表

| 范围 | 参数 | 默认值 |
|---|---|---|
| 两者 | `--models` | 上述 8 个模型 |
| 两者 | `--num_tensor_parallel_workers` | `1 2 4 8` |
| 两者 | `--num_gpus` | `8`，Ray 使用的可见 GPU 数 |
| 两者 | `--disable_ray` | 不指定时用 Ray |
| 两者 | `--output_dir` | `profiling_outputs` |
| MLP | `--max_tokens` | `4096` |
| Attention | `--max_model_len` | `4096`，同时决定 decode 图的 block-table 上限 |
| Attention | `--max_seq_len` | `4096`，有效输入长度的采样上界 |
| Attention | `--min_batch_size` / `--max_batch_size` | `1` / `128`，控制 decode batch 范围 |
| Attention | `--block_size` | `16` |
| Attention | `--profile_only_prefill` / `--profile_only_decode` | 默认同时采集，两者互斥 |

旧 `--attention_backend`、`--profile_method` 选择器已删除。独立的 `vllm_attention_profile_file`、`vllm_attention_decode_capture_limit` 和 RoPE opt-out 参数也不存在。新的 KV 写入及图形状信息由标准 CSV 携带，旧的缺列 attention 表需要重新采集。
