# Sarathi-Serve CUDA Graph 修改文档

目标：只修改 sarathi-Serve，使其真实性能尽量接近 vidur 仿真结果。

## 1. 根因定位

vidur 用 GPU-only 的执行时间模型（TP2 显式 `--random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling`），
而 sarathi-Serve 是真实执行，每步有大量 Python 侧 kernel launch 开销。

用 CUDA Event + `perf_counter` 对 `ModelRunner.run` 实测（TP2, Llama-2-7B, prefill 128）：

| 路径 | CPU enqueue (launch) | GPU 时间 |
|---|---|---|
| prefill（eager，改前） | **16.4 ms** | ~8.7 ms |
| decode（已有 cudagraph） | 0.25 ms | 5.1 ms |

结论：prefill ttft 的差距（19ms vs vidur 8.35ms）几乎全部来自 CPU launch overhead。
`gpu_model_ms` 里也包含 GPU 空泡，因为 CPU 供不上。

## 2. 修改的文件与内容

所有修改位于容器内 `/workspace/sarathi-serve`（仓库分支 `vidur`）。

### 2.0 已有的 decode CUDA Graph（改动前镜像里就有）
- `sarathi/config/config.py`: `enable_decode_cuda_graph`, `decode_cuda_graph_batch_size(s)`
- `model_runner.py`: `capture_decode_cuda_graph` / `_replay_decode_cuda_graph`
- `flashinfer_attention_wrapper.py`: `init_decode_cuda_graphs` / `can_use_decode_cuda_graph` + 固定 FlashInfer buffer
- `base_worker.py`: KV cache 分配后 capture

### 2.1 prefill CUDA Graph（本次新增，已备份镜像 `sarathi:vidur-cudagraph-prefill-graph`）
- `config.py`: `enable_prefill_cuda_graph`, `prefill_cuda_graph_chunk_sizes`, `prefill_cuda_graph_batch_sizes`
- `model_runner.py`:
  - `_get_prefill_cuda_graph_shapes()`：按 (num_seqs, chunk_len) 生成形状，受 `scheduler.chunk_size` 约束
  - `capture_prefill_cuda_graph(gpu_cache)`：按形状构造 dummy batch（多个 prefill 序列，各自独立 KV block），
    warmup 后 `torch.cuda.graph` capture；静态 input/output 按 key 保存
  - `_replay_prefill_cuda_graph(key, tokens, positions)`：拷贝输入后 `graph.replay()`
  - `run()`：纯 prefill 批量命中 key 时走 replay
- `flashinfer_attention_wrapper.py`:
  - `init_prefill_cuda_graphs(shapes, max_model_len)`：每个 key 建 `use_cuda_graph=True` 的 prefill wrapper + 固定 buffer
  - `get_prefill_cuda_graph_key()`：batch 全是 prompt、chunk 长度一致且 key 已 capture 时返回 key
  - `begin_forward` 选择 active prefill wrapper，并把 append buffer 指向固定 buffer
  - `forward`/`end_forward` 使用 `_active_prefill_wrapper`
- `base_worker.py`: `capture_prefill_cuda_graph(self.gpu_cache)`

### 2.2 多 batch decode
直接使用已有字段：`--worker_config_decode_cuda_graph_batch_sizes 1 2 3 4 5 6 7 8`。

### 2.3 实验性：自适应 universal CUDA Graph（未完成/不安全，默认关闭）
- `config.py`: `enable_universal_cuda_graph`, `universal_cuda_graph_max_seqs`
- `flashinfer_attention_wrapper.py`: `init_adaptive_cuda_graphs` / `get_adaptive_cuda_graph_key`
- `model_runner.py`: `capture_adaptive_cuda_graphs` / `_verify_adaptive_cuda_graphs` / `_replay_adaptive_cuda_graph`
- 发现：FlashInfer 的 cudagraph kernel 对 token 数、序列数有“部分”自适应（离线自检 `np4->np1`、`chunk128` vs `chunk512` 均 `diff=0`），
  但真实混合 shape 下会触发 `illegal memory access`（`flashinfer/attention/prefill.cuh` line 2344），
  且 decode wrapper 会断言 batch size 固定。因此该路径默认关闭，不要在生产开启。

## 3. 正确性验证

所有已启用路径都在 capture 后与 eager 前向逐位比对：
- prefill graph，多序列 (1~4)、多 context（ctx=128/256/512/1024）：`max_abs_diff = 0.000000`
- 说明固定 buffer + 每次 replay 前 `begin_forward` 更新 plan 的机制是正确的

## 4. 结果

单请求（TP2, prefill=128, decode=50, 单流）：

| 指标 | vidur | sarathi 改前 | sarathi + prefill graph |
|---|---|---|---|
| ttft | 8.35 ms | 19.0 ms | ~13.0 ms |
| tpot | 6.91 ms | 7.01 ms | ~7.0 ms |
| e2e | 354 ms | 370 ms | ~368 ms |

prefill CPU enqueue：16.4ms → 0.89ms；prefill GPU 8.7ms ≈ vidur 8.35ms。

吞吐（TP2, 256 请求, cap 8, 均为 SARATHI）：

| 场景 | 指标 | vidur | sarathi（纯 prefill/decode graph） |
|---|---|---|---|
| 128/qps10 | ttft / tpot / e2e | 12.1 / 6.98 / 360.8 | 25.8 / 8.33 / 442.3 |
| 512/qps7 | ttft / tpot / e2e | 23.6 / 7.30 / 388.3 | 45.7 / 9.04 / 497.6 |
| 3072/qps4.4 | ttft / tpot / e2e | 118.7 / 9.64 / 600.7 | 241.0 / 15.16 / 999.0 |

## 5. 吞吐仍有差距的原因

Sarathi 的 `SarathiScheduler._schedule` 会把 **decode token 和 prefill chunk 放进同一批**
（先加 running decode，`num_batched_tokens += 1`；再加 prefill，`chunk = chunk_size - num_batched_tokens`）。
因此混合批 / 非整数 chunk 频繁出现，而纯 prefill / 纯 decode 的固定 shape graph 覆盖不到，只能回退 eager，
CPU launch overhead 再次出现（prefill 越大越明显）。

要彻底消除，需要覆盖混合批，但：
- 精确形状组合爆炸；
- “万能自适应 graph” 在真实混合 shape 下触发 FlashInfer illegal memory access，不可用。

## 6. 使用命令

单流 / 纯 prefill 场景（已接近 vidur）：
```
--worker_config_enable_decode_cuda_graph True \
--worker_config_decode_cuda_graph_batch_sizes 1 2 3 4 5 6 7 8 \
--worker_config_enable_prefill_cuda_graph True \
--worker_config_prefill_cuda_graph_chunk_sizes 128 512 \
--worker_config_prefill_cuda_graph_batch_sizes 1 2 3 4
```
注意：需要给 graph 留显存，`gpu_memory_utilization` 建议 0.6~0.65。

## 7. 镜像
- `sarathi:vidur-cudagraph-decode-graph` — 仅 decode graph
- `sarathi:vidur-cudagraph-prefill-graph` — decode + prefill graph（推荐）
- `sarathi:wip-adaptive` — 含未完成的自适应 graph（实验，勿用）

## 8. 环境注意
- sim-101 的 `sarathi` 容器被固定到宿主机 GPU 0-1，且与其他作业共享；GPU 常被占用导致 OOM。
  测试时在空闲 GPU 上重建容器（`--gpus '"device=4,5"'`）即可。
- benchmark 结束时会因容器缺 Chrome 在 `metric_store.plot()` 报错（rc=1），但 `sequence_metrics.csv` 已写出。
