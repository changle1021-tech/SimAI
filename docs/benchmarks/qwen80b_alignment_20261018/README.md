# Qwen80B AICB/Vidur 对齐与验证材料

本目录包含修复源码对应的校准表、CPU/通信 profile、原始计时、对照数据、脚本、测试结果及报告。详细结果见 [Qwen80B_MoE修复与最终重测.md](Qwen80B_MoE修复与最终重测.md)。

实测环境：sim-101 H100 GPU 4–7，Qwen3-Next-80B-A3B BF16，TP1/DP4/EP4/PP1，单请求，chunk512，关闭 prefix cache。vLLM 实测版本为 **0.11.2**；这些结果不构成 vLLM 0.27.1 的验证。

10 档单请求（含源码/profile 冻结后的 2 档新长度）最大绝对误差：TTFT 4.98%、TPOT 1.97%、E2E 1.93%。分项计时限制和探针扰动见报告。

验证时的源码提交：

- SimAI/Vidur：[e4a84a4e33e014c4ac0ad2c05ee5691238869c13](https://github.com/changle1021-tech/SimAI/commit/e4a84a4e33e014c4ac0ad2c05ee5691238869c13)
- AICB：[db9353e6e924422baa02962aaceacdbde24e8309](https://github.com/changle1021-tech/aicb/commit/db9353e6e924422baa02962aaceacdbde24e8309)

本次文档发布提交仅新增本目录，不修改已经验证的生产源码与 profile。`SHA256.json` 保留冻结时的 296 个文件摘要；它不包括自身及发布时新增的 README 和便携启动脚本。

## CPU 仿真复现

先检出本分支并初始化 AICB 子模块，在已经安装仓库依赖的 Vidur Python 环境中执行：

```bash
git submodule update --init aicb
VIDUR_PYTHON=/path/to/vidur/python bash docs/benchmarks/qwen80b_alignment_20261018/run_vidur_cpu.sh
```

该脚本开启 CPU 开销建模，使用本目录的 phase_profile_v5、cpu_boundary_profile_v3 和 comm_profile_v2，模拟 512 输入 / 256 输出的单请求，无需 GPU。可通过 `QWEN_ALIGNMENT_OUTPUT_DIR` 指定输出目录。参考 TTFT 为 108.14672365771482 ms，E2E 为 2473.768407092126 ms。

`reproduce_vidur_p512_o256_cpu.sh` 是实测时使用的原始命令，保留了原机器绝对路径；`run_vidur_cpu.sh` 按检出目录解析路径。原始 GPU 采样脚本和路径记录属于实验溯源材料，换机器后需配置模型、运行时和路径。
