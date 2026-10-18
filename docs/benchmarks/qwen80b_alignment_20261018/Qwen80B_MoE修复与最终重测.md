# Qwen80B：MoE 分支修复与单请求最终重测

原 MoE 分支的修复方向正确，但完整生成链路、计时方式、运行时算子边界、通信及 CPU 聚合仍有问题。本次已在隔离分支修复并重测。10 档单请求（含冻结后2档新长度）的最大绝对误差为：TTFT **4.98%**、TPOT **1.97%**、E2E **1.93%**。这是本机、固定请求语料和下述配置的实测结果，不是其他硬件、版本或请求分布的保证。

范围：sim-101 GPU 4–7，H100，Qwen3-Next-80B-A3B-Instruct BF16，TP1/DP4/EP4/PP1；一个 active 请求，chunk512，prefix cache 关闭。服务为 **vLLM 0.11.2 / torch 2.9.0+cu128**，没有把这些结果称为 vLLM 0.27.1 的验证。

## 最终端到端结果

每档 3 次 warmup、5 次顺序对照，再加 3 次打乱顺序对照；表中实测取 8 次无桩请求的中位数。另有每档 3 次边界桩，仅用于分项。TPOT=(E2E−TTFT)/(输出 token−1)。输入为固定 token IDs，输出长度强制固定。原始记录见 [requests.json](fresh_validation_20261018/requests.json)，完整误差与最小/最大值见 [comparison_v5.json](comparison_v5.json)。

| 输入→输出 | vLLM TTFT | Vidur TTFT | 误差 | vLLM TPOT | Vidur TPOT | 误差 | E2E 误差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256→64 | 113.747 | 114.311 | +0.50% | 9.1024 | 9.2720 | +1.86% | +1.68% |
| 512→256 | 105.940 | 108.147 | +2.08% | 9.1423 | 9.2769 | +1.47% | +1.53% |
| 1024→128 | 204.663 | 201.275 | -1.66% | 9.1282 | 9.2877 | +1.75% | +1.34% |
| 1536→192 | 304.458 | 294.428 | -3.29% | 9.1494 | 9.2978 | +1.62% | +0.92% |
| 2048→256 | 389.230 | 387.608 | -0.42% | 9.1551 | 9.3073 | +1.66% | +1.46% |
| 3072→192 | 601.004 | 574.044 | -4.49% | 9.2046 | 9.3208 | +1.26% | -0.18% |
| 4096→128 | 791.537 | 760.584 | -3.91% | 9.2132 | 9.3339 | +1.31% | -0.94% |
| 8192→256 | 1505.657 | 1507.776 | +0.14% | 9.1863 | 9.3676 | +1.97% | +1.31% |

单位均为 ms。1024/1536/3072/4096 请求未用于 CPU profile 校准，但这些长度曾用于诊断，因此不把它们称为完全盲测。源码/profile 冻结后的新长度复核单列在后面。

## 原因、证据与修复

| 问题 | 打桩/检查证据 | 本次处理 |
|---|---|---|
| 默认 device 回归 | 原提交完整 prefill/decode 7/7 生成失败：CPU A_log/dt_bias 进入 Triton | 移除全局 CPU device 切换，GDN 参数显式 CUDA；最终 11/11 profile 点生成成功 |
| Graph 内 event 偏置 | 空 callback 也约 3.04µs；add 从 5.76µs 对照到冷缓存摊销 1.643µs | event 移到 replay 外，32 次回放与 flush-only 基线作差；每个 callback 前仍刷 L2 |
| prefill 被当作 Graph 内核和零散 GEMM | 原独立相加漏掉 eager launch gap、QKV 重排/物化、cache writeback、norm/编译区域边界 | prefill 使用 eager stream region；GDN 测完整编译链，CSV 对两个 norm 去重 |
| attention 路径不同 | 原表是 FlashInfer；服务是 FA3；另有 partial RoPE、Q/K norm 和 hybrid block 差异 | 改为 native paged FA3、KV 写入、partial_rotary_factor=.25、Q/K norm 无 residual、BF16 hybrid block544 |
| shared expert 串行相加 | 服务消融关闭 shared stream，decode 每 token 约增加 0.59–0.63ms | MoE 按真实路由和依赖测关键路径，保留 shared/routed 重叠 |
| BF16 EP 流量公式不对应实际后端 | 运行时 AG(hidden,router)+RS(hidden)，原公式按 DeepEP FP8/top-k 展开 | 修正 BF16 字节公式；四 rank、相同 padding 的通信独立实测 |
| prefill MoE 计算和通信耦合、逐层同步计时偏差 | 隔离 48 层合计 pre256=46.690ms、pre512=50.637ms；本次连续执行合计44.935/39.583ms | 采用直接测得的连续48层 MoE 总时间；已含通信时不再另加 EP |
| native MoE replay 不含外部 router linear | replay 的入口已拿到 router_logits；原 CSV 覆写丢掉 router | profile 明确 router 是否已包含；未包含时测 local-token router 并加一次；compute-only profile 不重复添加 |
| CPU/外围经验常数和等待重叠 | 原 CPU 五项相加3.08ms/iter；inclusive bookkeeping含GPU等待 | 用同一迭代的 host wall−model GPU elapsed，按首个/后续prefill和decode分别校准；请求IO和迭代间隙单列 |
| 缓存/插值掩盖新 profile | 旧全局/持久 cache 不检查源码和 CSV 身份，结果受历史端点影响 | Qwen绕过旧cache，检查dtype、形状、源文件/CSV/MoE profile SHA；按已测网格插值，越界报错 |
| 不存在的 dense MLP | 48层每层1µs fallback形成额外0.048ms | Qwen dense MLP项为0，shared/routed都归MoE |

原 MoE 分支已正确落地的部分包括投影形状、默认 BF16、BF16 routed fused experts、prefill conv/chunk GDN、shared expert 内容和机内链路选择；上述问题是这些改动之外的完整链路与时间聚合差异。

连续 MoE 总时间是测量值；为兼容逐层 CSV，用原独立逐层占比分配这个总量。**分配后的逐层数值不是一次新的逐层实测**。没有用客户端 E2E 残差拟合 MoE 或添加总时长常数。隔离与连续的差值包含执行顺序、同步和CPU状态差异，不能全部归因于一种同步API。不同实验轮次的 eager 时间有波动，第一次连续 MoE 测量的原始记录也保留在 continuous_moe_first_pass/，未冒充最终那轮数据。

GDN 原型对照在512-token时，native core约0.7392ms/层，包含投影、native gated norm和两个decoder norm的split compiled chain中位数约1.3022ms/层；差值在36个GDN层合计约20.3ms。这是组件边界对照，不是把全部服务差值归因于某一个kernel，原始样本见 [full_gdn_probe_v2.json](full_gdn_probe_v2.json)。

## 时间组成对应的源码入口

| 部分 | 文件与函数 |
|---|---|
| Attention/GDN采样 | `AiobQwen3Next.py:Qwen3NextModel.forward`，`qwen3_next_attention.py:benchmark_gdn_compiled_chain/benchmark_gdn_native_core/benchmark_flash_attention` |
| MoE采样/标定 | `qwen3_next_moe.py:benchmark_moe_critical_compute/load_critical_moe_profile/critical_profile_includes_router`；prefill连续native标定为 `continuous_moe_worker.py:q80_benchmark_continuous_moe` |
| CSV聚合与去重 | `utils/utils.py:extract_inference_averages`；`Vidur_workload_generator.py:VidurWorkload.workload_generate_aiob` |
| CSV身份校验/上下文插值 | `vidur/entities/qwen_aicb_profiles.py:QwenAICBProfiles._read/load` |
| 模型区域与通信 | `vidur/entities/execution_time.py:ExecutionTime._get_moe_layer_execution_time_from_aicb/model_time` |
| CPU/请求外围 | `qwen_aicb_profiles.py:RuntimeBoundaryProfile.estimate`；由 `ExecutionTime._get_cpu_overhead` 调用 |
| 最终迭代时间 | `ExecutionTime.total_time = model_time + _get_cpu_overhead()*1e-3`，模型项为秒，外围profile为ms |

相关文件在 [AICB源码快照](source/aicb/) 与 [Vidur源码快照](source/vidur-alibabacloud/)。

## CPU 开关具体加了什么

这里的外围成本是关键路径 wall time，包含输入准备、采样/LM head 等 model.forward 之外的 GPU 工作、等待/IPC；不是“CPU忙碌时间”。model CUDA event 区间也包含区域内CPU发射空隙，不是纯 kernel busy time。model、execute、sample 互相嵌套，不能把 inclusive 时间直接相加。

| 项目 | ms | 应用次数 |
|---|---:|---|
| 首个 prefill512 worker 净外围 | 8.7159 | 每请求一次 |
| 后续 prefill512 worker 净外围 | 1.9817 | 后续每chunk |
| 请求 admission+首token返回 | 5.8607 | context=0一次 |
| prefill chunk间隙 | 0.1299 | 后续每chunk |
| decode worker净外围 | 1.3284 | 每decode迭代 |
| decode迭代间隙 | 0.1164 | 每decode迭代 |

CPU profile 仅用输入256/512/2048/8192的边界桩校准；client/worker用同一机器的monotonic时间戳和request ID关联。字段与原始文件哈希见 [cpu_boundary_profile_v3.json](cpu_boundary_profile_v3.json)。可用 [verify_request_perimeter_v3.py](scripts/verify_request_perimeter_v3.py) 从原始时间戳重新计算并核对全部中位数。旧v2仅保留复现能力，最终使用v3。

## 模型内部与观察器检查

不是只检查E2E。下面把Vidur模型区域与同一边界的vLLM model事件对照。边界桩每档只有3次，且存在观察扰动/机器波动，不能视为零误差真值。

| 输入 | vLLM model中位数 ms | Vidur model ms | 差异 |
|---|---:|---:|---:|
| 256 | 101.103 | 99.623 | -1.46% |
| 512 | 92.993 | 93.570 | +0.62% |
| 1024 | 197.902 | 184.586 | -6.73% |
| 1536 | 286.405 | 275.628 | -3.76% |
| 2048 | 375.184 | 366.696 | -2.26% |
| 3072 | 566.739 | 548.909 | -3.15% |
| 4096 | 736.731 | 731.226 | -0.75% |
| 8192 | 1474.289 | 1461.525 | -0.87% |

**1024档内部模型仍低估6.73%**；该档轻量桩TTFT相对无桩中位数增加5.57%。因此不能宣称每一内部项也都已到5%以内。8档轻量桩TTFT相对无桩的变化约−3.81%到+5.57%，TPOT约+0.40%到+1.51%。这些原始内部差异保留，不用一个CPU常数把它们抹掉。decode模型部分约高估1%–1.3%，端到端TPOT仍约高1.3%–2%。

额外发现并排除了无效诊断：某次GDN replay每层deep-copy元数据，破坏了实际每chunk共享的cu_seqlens对象身份，导致FLA的8项identity cache反复miss。对照probe中36层shared_warm与cloned_per_layer约24ms对35–36ms。实际缓存审计是每chunk一次miss、随后35次hit。这组被破坏的GDN replay **没有进入校准**，见 [rejected_gdn_diagnostic.json](rejected_gdn_diagnostic.json)、[gdn_metadata_identity.json](gdn_metadata_identity.json)。重型 profiler 也仅用于依赖/调用结构检查，不用其绝对时间回填基线。

直接执行未加仿真观察器的 `python -m vidur.main` 后，512→256的TTFT/E2E与记录分项的wrapper逐数相同，见 [observer_simulation_equivalence.json](observer_simulation_equivalence.json)。

## 源码、测试与复现

- AICB: `db9353e6e924422baa02962aaceacdbde24e8309`，分支 `codex/qwen80b-runtime-profile-20261016`。
- SimAI/Vidur: `e4a84a4e33e014c4ac0ad2c05ee5691238869c13`，分支 `codex/qwen80b-aicb-alignment-20261016`，包含上述AICB gitlink。
- 原 MoE 基线：SimAI d5e67bf9443e98196a5ad0d73dcc93d0f8fe0035；AICB 75a142824258ff6e3c2c3138db0536cf8f8ebad4。修复在隔离工作树中，补丁分别为 [aicb_final.patch](aicb_final.patch)、[simai_final.patch](simai_final.patch)。
- sim-101实际环境：AICB 35/35、Vidur 67/67 测试通过；最终GPU profile 11/11生成成功；8个仿真case全部成功。测试日志与 [tests_final.json](tests_final.json) 一并保存。
- 远端完整工作树：`/home/turbo_ops/changle/Files/analysis/qwen80b_alignment_20261016/source/SimAI`；本地 source/ 为本次相关源码快照。
- CPU开启的原生Vidur命令：[reproduce_vidur_p512_o256_cpu.sh](reproduce_vidur_p512_o256_cpu.sh)。实际vLLM启动：[start_frozen_holdouts_server.sh](scripts/start_frozen_holdouts_server.sh)。profile生成：[phase_profile_v5.py](scripts/phase_profile_v5.py)。
- CPU开关为 `--no-random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling`；另外显式传 `aicb_profile_dir`、`aicb_cpu_overhead_profile` 和 `aicb_moe_comm_profile`。命令中保留了实际配置类前缀。

## 源码/profile冻结后的新长度验证

模型源码、11份CSV/sidecar与CPU profile先冻结并记录哈希，再启动无worker扩展、无桩的原生vLLM服务器。新长度2560/6144、输出96，此前未参与任何校准或验证；每档3次warmup后打乱顺序各测8次。其后没有调整参数。

| 输入→输出 | vLLM TTFT / Vidur | TTFT误差 | vLLM TPOT / Vidur | TPOT误差 | E2E误差 |
|---|---:|---:|---:|---:|---:|
| 2560→96 | 505.992 / 480.813 | -4.98% | 9.2400 / 9.3131 | +0.79% | -1.12% |
| 6144→96 | 1183.452 / 1133.973 | -4.18% | 9.2658 / 9.3493 | +0.90% | -1.93% |

原始客户端记录、命令与完整范围在 [frozen_holdouts/](frozen_holdouts/)。冻结前后哈希、源码提交和GPU清理状态见 [final_receipt.json](final_receipt.json)。

本结论限于上述配置和测量范围。任意长度、更多并发、其他路由语料、TP/PP配置、GPU或vLLM版本需要新profile与验证；当前实现对不在已测网格内的查询会报错。
