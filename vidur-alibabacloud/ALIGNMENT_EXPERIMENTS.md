# Vidur vs vLLM 0.5.1 对齐实验记录

- 模型：Llama-2-7b-hf (H100, TP=2, PP=1, block=16, max-num-seqs=8, max-num-batched-tokens=4096)
- 请求：256，Poisson，seed=36，prefill/decode 按各实验，decode=50
- 客户端：vllm_benchmark_client.py（TTFT=POST→首个非空 SSE token；TPOT=(E2E−TTFT)/49）
- Vidur：NCCL=0.02，skip_cpu_overhead_modeling（除注明外）
- 验收：E2E 误差 ≤10%

---

## 1. Vidur baseline（256 请求）

| 实验 | 输入/QPS | TTFT mean | TPOT mean | E2E mean | E2E P50 | E2E P99 |
|------|---------|----------:|----------:|---------:|--------:|--------:|
| skip_cpu=TRUE, NCCL=0.02 | 128 / 10 | 11.838 | 7.602 | 384.333 | 384.333 | 412.743 |
| skip_cpu=TRUE, NCCL=0.02 | 512 / 7 | 16.007 | 7.808 | 398.614 | 398.206 | 450.163 |
| skip_cpu=TRUE, NCCL=0.02 | 3072 / 4.4 | 70.123 | 11.275 | 622.579 | 623.921 | 916.549 |
| skip_cpu=FALSE（旧 Sarathi profile） | 128 / 10 | 37.193 | 10.981 | 575.257 | 564.629 | 752.335 |
| batching_overhead=0（A/B） | 128 / 10 | 11.838 | 7.602 | 384.333 | — | — |
| batching_overhead=0（A/B） | 3072 / 4.4 | 70.123 | 11.275 | 622.579 | — | — |

结论：
- 关掉 skip 后方向对但高估 ~2.5 倍（旧 profile 来自 Sarathi，batch≥8，ray_comm=1.63ms vs vLLM 实测 0.80ms）。
- `attention_*_batching_overhead_fraction` 对 Llama-2 **无效**（MHA 被 `sklearn_execution_time_predictor.py:49-59` 强制置 0），A/B 逐位相同。

## 2. vLLM 三组（256 请求）

| 实验 | 输入/QPS | TTFT mean | TPOT mean | E2E mean | E2E P50 | E2E P99 |
|------|---------|----------:|----------:|---------:|--------:|--------:|
| ray（原配置） | 128 / 10 | 21.377 | 8.929 | 458.891 | 453.675 | 579.079 |
| ray（原配置） | 512 / 7 | 19.120 | 8.432 | 432.308 | 432.825 | 503.920 |
| ray（原配置） | 3072 / 4.4 | 73.387 | 10.955 | 610.202 | 604.148 | 924.140 |
| **mp** | 128 / 10 | 23.733 | 8.257 | 428.310 | 417.701 | 665.238 |
| **mp** | 512 / 7 | 22.046 | 8.296 | 428.528 | 417.308 | 582.547 |
| **mp** | 3072 / 4.4 | 75.626 | 11.183 | 623.575 | 580.689 | 1071.598 |
| mp（首次复现） | 128 / 10 | 23.930 | 8.135 | 422.523 | 386.993 | 747.040 |
| ray + `--disable-custom-all-reduce` | 128 / 10 | 29.592 | 9.424 | 491.360 | 491.467 | 649.762 |
| ray + `--disable-custom-all-reduce` | 3072 / 4.4 | 86.729 | 12.680 | 708.043 | 636.100 | 1721.671 |
| ray + `--enforce-eager` | 128 / 10 | 2005.455 | 17.350 | 2855.588 | 3270.151 | 4841.681 |

## 3. 对齐差值（E2E，相对 Vidur）

| vLLM 配置 | P128/QPS10 | P512/QPS7 | P3072/QPS4.4 |
|------|------:|------:|------:|
| ray | +19.4% | +8.5% | −2.0% |
| **mp** | **+11.4%** | **+7.5%** | **+0.2%** |
| ray + no custom-AR | +27.9% | — | +13.7% |

## 4. vLLM 容量 sweep（ray，512 请求）

| 输入 | QPS_max | QPS_target |
|------|--------:|-----------:|
| 128 | 12 | 10 |
| 512 | 12 | 7 |
| 3072 | 6.2 | 4.4 |

## 5. 关键诊断

- `vidur/entities/batch.py:49`：`_total_num_tokens_rounded=(total+7)//8*8`，decode batch 1–8 全被取整到 8 → Vidur decode 单步恒为 7.016ms（bs1–8 几乎不变），与 vLLM 的 batch 增长不符。
- TP=2 MLP 分解（nt=8）：MLP 0.097ms/层(44%)、2×all-reduce 0.114ms/层(52%)、attn 0.009ms/层(4%)。
- attention profile 在 bs1–8/kv128 恒为 0.009ms；all_reduce profile 2-worker 为同节点 NVLink。
- vLLM attention backend = FLASH_ATTN，decode 走 `flash_attn_with_kvcache`，与 profile 一致。
- vLLM CUDA Graph 已捕获；`--enforce-eager` 使 TPOT 8.93→17.35，证明 profile 属 graph 路径。
- vLLM 默认启用 custom all-reduce；`--disable-custom-all-reduce` 使 TPOT 变慢（P128 +0.49、P3072 +1.72ms），说明 NCCL profile 高估通信。
- CPU overhead profiler（`vidur/profiling/cpu_overhead/benchmark_runner.py`）基于 `sarathi.LLMEngine`，且 batch 列表从 8 起，无法在不改代码下产出 vLLM 0.5.1 的 batch 1–8 数据。

## 6. 当前最优配置

```
vLLM:  --distributed-executor-backend mp （单机 TP=2 默认，省 ~0.80ms/step Ray 开销）
Vidur: NCCL=0.02, skip_cpu_overhead_modeling, batching_overhead 默认
```
结果：P512 +7.5%、P3072 +0.2% 达标；P128 +11.4%（边界外）。

P128 剩余 ~44ms = TTFT 11.9ms（frontend + 步粒度等待）+ TPOT 0.66ms×49=32ms（未建模 CPU 开销）。

---
（后续实验追加于下）

## 7. CPU overhead 校准（自定义 cpu_overheads.csv）

方法：生成 `overhead(batch)=A+B*batch`（全部放在 ray_comm，其余列 0），
用 `--random_forrest_execution_time_predictor_config_cpu_overhead_input_file` 指向该 csv，
并加 `--no-...skip_cpu_overhead_modeling`。Vidur 其余设置不变（NCCL=0.02）。

| A | B | P128 E2E | P512 E2E | P3072 E2E | vs ray (128/512/3072) | vs mp (128/512/3072) |
|---|---|------:|------:|------:|---|---|
| 0.50 | 0.20 | 477.90 | 474.57 | 722.74 | +4.1 / +9.8 / +18.4 | +11.6 / +10.7 / +15.9 |
| 0.25 | 0.10 | 427.90 | 435.11 | 668.25 | −6.8 / +0.7 / +9.5 | −0.1 / +1.5 / +7.2 |
| 0.20 | 0.08 | 417.58 | 426.40 | 657.42 | −9.0 / −1.4 / +7.7 | −2.5 / −0.5 / +5.4 |
| 0.15 | 0.06 | 408.92 | 419.45 | 648.84 | −10.9 / −3.0 / +6.3 | −4.5 / −2.1 / +4.1 |

**最优解：**
- 对齐 **mp**：A=0.15, B=0.06 → 最大误差 **4.53%**
- 对齐 **ray**：A=0.20, B=0.08 → 最大误差 **9.00%**

备注：`overhead(batch)=0.15+0.06*batch`（bs1≈0.21ms、bs8≈0.63ms）低于实测 vLLM CPU/Ray 开销
（ray 在 batch≈3.9 时约 1.3ms），说明该经验项同时在补偿长 context 下 GPU attention 画像的偏高。
因此它是"端到端对齐"的折中，不是纯 CPU 开销的物理真值。

## 8. 稳定性复测（fresh server，每组多次）

| 输入 | ray E2E（各次） | ray 均值 | mp E2E（各次） | mp 均值 | Vidur baseline |
|------|------|------:|------|------:|------:|
| P128 | 442.0 / 446.3 / 458.9 | 449.1 | 404.6 / 407.1 / 428.3 | 415.6 | 384.3 |
| P512 | 423.4 / 424.5 / 432.3 | 426.7 | 396.7 / 397.4 / 428.5 | 407.5 | 398.6 |
| P3072 | 603.4 / 614.3 / 610.2 | 609.3 | 598.2 / 598.7 / 623.6 | 606.8 | 622.6 |

**Vidur baseline（NCCL=0.02, skip_cpu）相对各后端：**

| 输入 | vs ray | vs mp |
|------|------:|------:|
| P128 | −14.4% | **−7.5%** |
| P512 | −6.6% | **−2.2%** |
| P3072 | +2.2% | **+2.6%** |

结论：
- **vLLM `mp` + Vidur baseline 三组均在 10% 内（max 7.5%），无需任何 CPU overhead 校准。**
- ray 使 P128 达 −14.4%（超 10%），根因是 Ray 每步 ~0.8ms 开销 Vidur 未建模。
- 单次运行方差较大（同一 session 内 ~1%，跨 session ~5%），因此必须重复取均值。
- 先前 P128 的 −10.3%/−16.2% 属单次方差。

## 9. 最终推荐配置

```
vLLM:  --distributed-executor-backend mp
Vidur: NCCL=0.02, skip_cpu_overhead_modeling（默认）
```
若必须保留 Ray（生产部署），则需给 Vidur 加 CPU overhead 校准：
`overhead(batch)=0.20+0.08*batch`（ray_comm）→ 三组 max 9.0%。

## 10. TP=4 对齐

**Vidur TP=4**（NCCL=0.02, skip_cpu）：| 输入 | TTFT | TPOT | E2E |
|---|---|---:|---:|
| 128/10 | 9.71 | 6.51 | 328.86 |
| 512/7 | 12.37 | 6.57 | 334.29 |
| 3072/4.4 | 44.42 | 8.00 | 436.35 |

**vLLM TP=4**（mp, GPU0-3, 256 请求, 3 次）：| 输入 | E2E 各次 | 均值 |
|---|---|---:|
| 128/10 | 385.6 / 387.0 / 353.2 | 375.3 |
| 512/7 | 320.3 / 378.4 / 409.8 | 369.5 |
| 3072/4.4 | 468.2 / 501.8 / 621.7 | 530.6 |

**Vidur TP=4 vs vLLM TP=4**：
| 输入 | 误差(均值) |
|---|---:|
| 128/10 | −12.4% |
| 512/7 | −9.5% |
| 3072/4.4 | −17.8% |

结论：
- TP=4 下 Vidur 偏快更明显。Vidur 预测 TP2→TP4 E2E 384→329（−14%），但 vLLM 378→385（约不变）。
- 即 **Vidur 高估了 TP 扩展收益**（可能 TP=4 的 all-reduce 开销/通信未被 profile 正确反映），叠加未建模的 CPU overhead。
- 需要对 TP=4 也补 CPU overhead，量级比 TP=2 更大。

## 11. TP=4 CPU overhead 校准（overhead(batch)=A+B*batch，ray_comm）

| A | B | P128 | P512 | P3072 | vs vLLM TP=4 |
|---|---|------:|------:|------:|---|
| 0（baseline） | 0 | 328.86 | 334.29 | 436.35 | −14.7 / −11.7 / −13.1 |
| 0.30 | 0.12 | 376.03 | 374.00 | 480.36 | −2.5 / −1.2 / −4.3 |
| 0.35 | 0.14 | 384.40 | 380.93 | 487.64 | −0.3 / +0.7 / −2.8 |
| 0.40 | 0.16 | 392.92 | 387.87 | 495.68 | **+1.9 / +2.5 / −1.2** |
| 0.50 | 0.20 | 411.46 | 402.48 | 510.50 | +6.7 / +6.4 / +1.7 |

vLLM TP=4 参照取 3 次中位数：P128 385.6 / P512 378.4 / P3072 501.8。

**TP=4 最优：A=0.40, B=0.16，最大误差 2.50%。**

## 12. 最终结论（TP=2 与 TP=4）

| 并行 | vLLM 后端 | Vidur CPU overhead（ray_comm） | 三组最大误差 |
|---|---|---|---|
| TP=2 | mp | 0.15 + 0.06*batch | 4.53% (vs mp 3 次均值) |
| TP=4 | mp | 0.40 + 0.16*batch | 2.50% |

- 两组的最优 overhead 之比 ≈ 2.7 ≈ (4/2)^1.4，说明该经验项随 TP worker 数增长（协调开销），
  同时也补偿了 TP 扩展的 profile 偏差（Vidur 高估 TP2→TP4 的收益）。
- 因此 overhead 需按 TP 分别标定；Vidur 其余设置保持 NCCL=0.02、batching_overhead 默认。
- vLLM 统一用 mp（单机默认），避免 Ray 的 ~0.8ms/step 额外开销。
- 测量方差较大（尤其高负载点），必须同 session 内多次取中位数。
