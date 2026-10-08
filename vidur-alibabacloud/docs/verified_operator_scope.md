# 逐算子预测分支的可用范围

本分支基于 `codex/pp-rollback-20261002`，基线提交为
`445717400a8c43e42ee1a54bafc2987bf7d40663`。最终预测继续使用逐算子计算、通信及已有 CPU 开销组合。

## 已包含的可用实现

以下代码已存在于基线中，本次没有重复修改或将其列为新增精度收益：

- `collectives_wrapper.py`：send/recv 默认采集 60 轮，支持显式正整数轮数覆盖；保留逐轮代表值及 min/max/mean/median/std。
- `benchmark_runner.py` 与 `main.py`：`--num_profile_rounds` 从 CLI 传到实际采集循环。
- `sklearn_execution_time_predictor.py`：send/recv 随机森林使用 `time_stats.send_recv.mean`。训练缓存包含标签列与数据内容，预测缓存包含训练身份和预测输入。
- `vllm_051_comm_profile.py`：vLLM 0.5.1 通信采集适配器，记录执行模式与轮次元数据，导出 Vidur 兼容 CSV。

这里的均值是每轮 NCCL 事件中位数的跨轮算术平均值，不是所有单次调用的平均值。两份采集入口的统计策略不同：基线 `collectives.main` 对各 rank 的逐轮值取最大值；vLLM 适配器保存 rank 0 的统计。已有 CSV 不因这份说明而被替换，两个口径不能混称。

## 本次提交的范围

本次只增加本说明，不改变运行时预测代码、配置、缓存或性能表。

未纳入整个 GPU 模型区间替代逐算子预测的候选，也未纳入相关外层候选。PP 张量数量／padding 候选虽有原生形状核验，仍缺完整逐算子 E2E 验收，因此本次没有纳入。PP virtual engine 调度差异仍是定位结果，没有完成修复。

60 轮新 send/recv 表的列名及单位可被当前预测器读取，但它仅覆盖单机双卡 decode 图模式；本分支不将其直接覆盖到旧表，也不将其适用性推广到 prefill 或跨节点。

## 验证及精度边界

在该分支的完整基线源码上运行：

```bash
python -m unittest discover -s tests -p test_send_recv_profiling.py -v
```

2026-10-08 在 sim-101 上运行上述命令，8 项测试全部通过。测试产生的既有 pickle 文件句柄 ResourceWarning 已保留，不影响测试终态。

该测试覆盖轮数默认值及传递、均值训练标签、标签与数据变更后的缓存失效，以及随机森林重新训练／缓存复用。

这份提交没有新增运行时修复，不报告新增 E2E 精度收益。整体 GPU 区间候选的低并发精度改善不属于本分支成果。PP 和 20B TP4 随机负载的整体对齐仍未完成。
