# TP 时间对齐首个修复：配对 GPU／wall 区间

2026-10-07。依据用户指定的 SIMULATOR_ALIGNMENT_RULES.md。当前交付仅覆盖纯 TP4、单请求低并发、输入512／输出50；完整 TP／PP 路线尚未完成。

旧预测器将历史单算子／通信表组合为模型成本，再加入同步引擎 CPU 表。目标服务为 vLLM 0.5.1 的异步 Ray API，prefill 普通执行、decode CUDA graph。历史表采集条件未知；本次没有替换旧表、删除旧 skew 参数或以最终误差拟合补偿值。

新采样在同一真实 API 步骤中记录本地 driver GPU 模型区间，结束点在 sampler 之前；CPU／executor 端点与 GPU 事件通过请求外时钟锚点对齐。调度、模型开始前的准备／dispatch／staging、模型结束后的 sampling 尾段、executor 返回等待及输出处理分别计一次。GPU 区间包含实际模型入口、层内计算／通信、logits、enqueue 空挡与 rank 等待；sampling GPU 工作在尾段中。各段不是“纯 CPU／网络时间”。

20B TP4 decode 采样均值：GPU模型8.395ms、其余关键路径区间合计2.044ms，整步10.439ms。sampler完整墙钟8.410ms，而模型GPU结束后的sampling尾段0.534ms，证明完整sampler墙钟包含模型GPU等待，不能再整体叠加GPU模型区间。时钟对齐不确定度0.030ms。7B对应整步8.140ms、不确定度0.050ms。分项账闭合只用于核对边界，不是准确性证明；实测验收另用无仪器控制请求。

候选新增显式 `async_execution_input_file`，读取同一CSV中的GPU及五个wall区间。模型结构／TP／时序范围必须匹配；按实际batch及序列长度的样本均值、范围内线性插值预测，超出已测形状报错。GPU模型整体区间计一次，不重复叠加旧层内通信／skew估计；旧参数在默认路径保持原样。该粗粒度路径不提供逐算子的独立费用估计。

严格匹配发现7B默认模拟配置vocab=32768，实际HF配置vocab=32000。因此新路径另有显式 `async_execution_vocab_size=32000`，只用于与实测输入验证匹配；原7B配置、性能表、缓存键保持原状。这是实际模型结构参数，不是误差补偿。

## 同条件前后对照

真实服务参数、请求、EOS、QPS0.3、GPU4–7、actual arrival trace均保持。每模型before/after各3请求，合并全部6条无仪器正式请求；插桩profile请求单列且不进入验收。

| 模型／TP | 实测 E2E均值ms | 旧路径预测ms／误差 | 新路径预测ms／误差 |
|---|---:|---:|---:|
| Llama-2-7B TP4 | 448.644 | 376.200／−16.147% | 420.787／−6.209% |
| InternLM-20B TP4 | 551.111 | 690.850／+25.356% | 556.909／+1.052% |

这两组低并发负载达到±10%标准，不能据此称为通用建模已经正确。TTFT仍是模拟prefill completion代理，与客户端首个非空输出不同。入口等待及各卡区间仍只能按其定义解释，不能用类别max之和解释整步。

实测输入版本：`async_path_v2/async_execution_low_512_v1.csv`，300正式step样本，SHA256 `895c33096e28bb79390cfe759a1b02d30ea1a18486ac6054cfabaee0a0afe1d2`。原始window、命令、软件／HF配置、脚本与输入hash、硬件快照及clock bracket在metadata与原目录保留。客户端E2E／TPOT不作为预测输入成本的训练目标。旧表和已保存实验结果未覆盖。

## 验证与边界

确定性测试验证样本均值、插值、结构／TP／scope拒绝、超出范围拒绝；时序自测验证GPU／sampler等待不会重复累计、拒绝宽时钟锚点及乱序端点，也覆盖长uptime浮点精度。源代码和回放修复分支实际模块路径由运行时断言核对。

首次回放因cwd优先加载原仓库而失败；第二次包根断言因namespace package失败；第三次被词表结构校验拒绝。失败记录均保留。成功回放版本为服务器 `async_replay_v4`，原命令未使用-O。

原默认路径36条固定长度TP2／TP4回归已通过，与修复前预测的最大差值为0.0ms；结果在 `async_replay_v4/legacy_regression.json`。已有达标的随机7B及20B TP2仍需按要求完成相关回归记录。其他输入、并发、随机长度以及PP／混合并行尚未由这份新表覆盖，不能外推；跨节点不属于本任务。后续只补与本次时间修复直接相关的采样和验证。

服务器备份分支：`codex/alignment-backup-20261007`，提交52d87b5。修复worktree：`/home/turbo_ops/changle/SimAI-alignment-20261007`，分支`codex/tp-pp-alignment-20261007`。主仓库既有改动保留。
