# vLLM 0.5.1 通信 profiling

`vllm_051_comm_profile.py` 在 vLLM 0.5.1 容器内运行。采样网格、worker 布局、有效性过滤、buffer 大小、轮数、计时聚合和 CSV 列定义都从旁边现有 Vidur 源码加载；不要手工改一份参数副本。size 输入单位是元素，CSV 的 `size` 字段按 FP16 的每元素 2 字节写成 bytes。默认 size 网格沿用 Vidur，包括其中的重复点。

所有命令假定宿主机 SimAI checkout 已挂载到容器的 `/root/changle/SimAI`，输出路径挂载到 `/root/changle/Files`。推荐显式指定 `--mode decode`：`--mode` 默认值也是 `decode`，并支持 `prefill`、`decode`、`both`。

## all_reduce

建议显式指定本机可用的 1、2、4 卡布局：

```bash
sudo nerdctl --namespace changle exec -it vllm051 python3 /root/changle/SimAI/vidur-alibabacloud/vidur/profiling/collectives/vllm_051_comm_profile.py --mode decode --collective all_reduce --num_workers_per_node_combinations 1 2 4 --output_dir /root/changle/Files/vidur_vllm051_comm
```

默认 worker-per-node 列表仍是 `1 2 4 8`。在每节点只有 4 张可见 GPU 的集群上，布局 8 会被记录为不可用并跳过；显式写 `1 2 4` 可避免该提示。

## send_recv

`send_recv` 只支持两 worker，使用每节点 2 卡布局：

```bash
sudo nerdctl --namespace changle exec -it vllm051 python3 /root/changle/SimAI/vidur-alibabacloud/vidur/profiling/collectives/vllm_051_comm_profile.py --mode decode --collective send_recv --num_workers_per_node_combinations 2 --num_profile_rounds 60 --output_dir /root/changle/Files/vidur_vllm051_comm
```

## 只查看计划

`--plan-only` 只加载 Vidur 参数并打印合同、网格摘要和源码哈希，不启动 Ray 或 GPU：

```bash
sudo nerdctl --namespace changle exec -it vllm051 python3 /root/changle/SimAI/vidur-alibabacloud/vidur/profiling/collectives/vllm_051_comm_profile.py --mode decode --collective all_reduce --num_workers_per_node_combinations 1 2 4 --output_dir /root/changle/Files/vidur_vllm051_comm --plan-only
```

## 接入 Vidur

一次运行会为所选 collective 写入 `<output_dir>/collective/<时间戳>/<collective>.csv` 和对应的 `<collective>.metadata.json`，例如 `all_reduce.csv`、`all_reduce.metadata.json`。CSV 保持 Vidur 的 `time_stats.*` 与结果字段，并保留 pandas 默认 index 列。使用容器中可访问的 CSV 路径，将以下参数追加到原仿真命令（将时间戳替换成实际目录名）：

替换表严格采用 Vidur 原生 CSV 列顺序：匿名 index 列、`time_stats.<collective>.min/max/mean/median/std`、`rank`、`num_workers`、`size`、`collective`、`devices_per_node`、`max_devices_per_node`。`size` 为 bytes，时间为毫秒；网格顺序和重复采样点保持不变。`round_times`、运行审计和执行模式只写 metadata，不增加 CSV 列。格式兼容并不代表测量数值相同：原生 `DISABLE_GRAPH=True` 使用 eager，而 `--mode decode` 测量 PyNccl CUDA graph replay；对比数值时需使用相同执行模式、GPU、通信布局和统计口径。

```bash
--random_forrest_execution_time_predictor_config_backend vidur \
--random_forrest_execution_time_predictor_config_all_reduce_input_file "/root/changle/Files/vidur_vllm051_comm/collective/<时间戳>/all_reduce.csv"
```

`PP > 1` 时还需追加 send/recv 表：

```bash
--random_forrest_execution_time_predictor_config_send_recv_input_file "/root/changle/Files/vidur_vllm051_comm/collective/<send_recv时间戳>/send_recv.csv"
```

路径必须能被运行 Vidur 的环境访问；例如容器内 `/root/changle/Files` 对应宿主机 `/home/turbo_ops/changle/Files`。`--mode both` 会测量 prefill 与 decode，但供原版 Vidur 使用的单一 CSV 全部取 decode 结果，保留 Vidur 原有网格点（包括重复点）。metadata 记录 `profiled_modes`、`csv_mode`、`mode_row_counts` 和 `mode_row_audits`。因此 prefill/decode 在原版 Vidur 中统一使用 decode 标定值，单表不分别表示两种模式。

## 采集约定

prefill 使用 eager，每轮 11 次 collective 调用；decode 使用 CUDA graph，建图前预热 5 次、捕获 3 次调用，每轮 replay 一次（一次 replay 执行图内 3 次调用）。`send_recv` 默认采集 60 个独立计时轮次；其余 collective 默认沿用 `collectives_wrapper.py` 的 `ACTIVE_STEPS`（sim-101 当前为 60）。两阶段只改变执行模式，因此不能把它们描述为执行次数相同。

`collectives.main` 和 `vllm_051_comm_profile.py` 都支持可选参数 `--num_profile_rounds`。省略参数时 `send_recv` 使用 60 轮，其余 collective 使用 `ACTIVE_STEPS`；指定正整数可覆盖每个 size/layout 各自的计时轮数，例如：

```bash
sudo nerdctl --namespace changle exec -it vllm051 python3 /root/changle/SimAI/vidur-alibabacloud/vidur/profiling/collectives/vllm_051_comm_profile.py --mode decode --collective all_reduce --num_workers_per_node_combinations 1 2 4 --num_profile_rounds 30 --output_dir /root/changle/Files/vidur_vllm051_comm
```

该参数只控制每个 size/layout 的有效计时轮数，不改变每轮 eager 的 11 次调用、decode CUDA graph 内的 3 次调用，也不改变建图前 5 次 warmup。使用 `--plan-only` 时，计划摘要会显示生效的 `active_steps`（例如 30）；运行生成的 metadata 也会记录实际轮数。

每轮先对 NCCL 事件耗时取中位数，再汇总全部轮次，输出 `min/max/mean/median/std`（毫秒）。`time_stats.send_recv.mean` 是 60 个轮次代表耗时的算术平均值，Vidur 的 send/recv 随机森林使用此列作为训练标签；60 轮不会展开成 60 行训练样本。轮数由参数覆盖后，均值按实际轮数计算。sim-101 的 all-reduce 保留现有的 60 轮采集与 `time_stats.all_reduce.mean` 训练标签；原生采集入口保留每轮跨 rank 取最大耗时后再汇总的逻辑。

metadata 的 `round_aggregation` 记录轮内计时口径；send/recv 的 `rounds_aggregation` 记录跨轮算术平均值口径，`training_target` 标明训练标签。每行 audit 的 `statistics_samples` 记录实际计时轮数。训练及预测缓存关联训练数据与目标列，更换采集数据或统计口径后自动生成新的缓存。旧 CSV 的 mean 列仍可读取，但需重新采集才能得到 60 轮均值。

全程关闭 vLLM custom all-reduce。FP16、rank 0 结果策略和 Kineto NCCL 事件计时沿用 Vidur。vLLM 0.5.1 没有对应 PyNccl 原语的 `all_gather`、`broadcast`、`reduce_scatter`、`all_to_all` 使用其 `device_group` 上的 torch/NCCL 集合通信；metadata 将其标为 torch/NCCL group 实现，不称作原生 PyNccl。all-reduce 和 send/recv 的 prefill 走 torch/NCCL，decode 走 vLLM PyNccl 的 CUDA Graph replay；PyNccl 不可用时 decode 会报错。
