# H100 原生采集结果：2026-10-08

本次保存的是通过新版标准 MLP / Attention 入口完成的两组采集结果，CSV 按原始字节提交，未调整测量数值。

## 覆盖范围

- 模型：`meta-llama/Llama-2-7b-hf`、`internlm/internlm-20b`。
- GPU / dtype：H100 / FP16；vLLM 0.5.1。
- TP：**1、2、4**。本组数据不包含 TP8。
- MLP：`max_tokens=4096`。
- Attention：`max_model_len=4096`、`max_seq_len=4096`、`block_size=16`、decode batch 范围 1–8，同时采集 prefill/decode。
- 运行方式：Ray，4 个 GPU worker。
- 计时方法：`vllm_cuda_graph_kernel_sum_v2`。数据包含分项统计、逐轮样本和 kernel 名称。

四张表放在相邻的标准路径中，仿真器可以通过原有默认输入路径读取：

```text
data/profiling/compute/h100/meta-llama/Llama-2-7b-hf/mlp.csv
data/profiling/compute/h100/meta-llama/Llama-2-7b-hf/attention.csv
data/profiling/compute/h100/internlm/internlm-20b/mlp.csv
data/profiling/compute/h100/internlm/internlm-20b/attention.csv
```

| 模型 | 类型 | 行数 | SHA256 |
|---|---|---:|---|
| meta-llama/Llama-2-7b-hf | mlp | 783 | 70565d7ff2d220d14d077d480767a5b1d065c0605208ddade05cd7b4825e8a74 |
| meta-llama/Llama-2-7b-hf | attention | 3876 | 97f2ff85e6d474049c94b8f1722147a597137c23bf5d6963e98409eb53a96393 |
| internlm/internlm-20b | mlp | 783 | 84e427cc675da5e6036caa1fe41194ef674b562ed3c340ef978fbb84f25b3e91 |
| internlm/internlm-20b | attention | 3876 | f2c484791f96606bae480075e616375aceed73b8c9bb978adab60cdec0361b29 |

`mlp_config.json` 和 `attention_config.json` 是采集时保存的原始参数。`manifest.json` 记录来源、哈希、每个 TP 的行数，以及按采样网格核对的完整性。

## 新旧 Attention 对照

`attention_comparison.csv` 对照旧表与新表在 B1、prefill512 或 KV512 下的同名 Attention 中位数字段，覆盖可匹配的 TP1/2/4。重复形状行取其中位数字段的算术平均。

新表的 KV 写入成本独立列出，未混入 Attention 对比列。旧表与新表的 runtime、图形状和缓存条件并不相同，因此这里是数值对照，不是等价性证明，也不能代替真实 TP4 服务的准确性验证。旧表来源及哈希保存在 manifest 中。

## 附带的已完成仿真结果

`benchmark_tp1_pp1/` 保存一次 Llama-2-7B TP1/PP1 回放的配置、请求指标、batch 指标和汇总：3 个请求，输入512、输出50。

TPOT 按 `decode_time / (output_tokens - 1)` 计算；prefill completion 不等同于客户端 TTFT。它是仿真输出，不是原生 vLLM 实测结果。只有到达时间和 token 长度与指标一致时，才同时保存到达轨迹快照。

原配置中的 CPU overhead 表为外部输入，原始路径保留在配置中。本目录不宣称已完成其他模型或 TP4 的端到端准确性验收。
