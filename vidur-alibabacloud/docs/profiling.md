# How to add a new model to the simulator?

For complete current commands, including the sim-101 Docker environment, see [profiling_commands.md](profiling_commands.md).

## Structure of Profiling data

The profiling data is stored in the `data/profiling` directory. The profiling data is stored in CSV format. The profiling data is stored in the following format:

```yaml
    profiling/
        - compute
            - a100
                - codellama/CodeLlama-34b-Instruct-hf/
                    - mlp.csv
                    - attention.csv
                - internlm/internlm-20b/
                    - mlp.csv
                    - attention.csv
            - h100
                - meta-llama/Llama-2-70b-hf/
                    - mlp.csv
                    - attention.csv
        - network
            - a100_pair_nvlink
                - allreduce.csv
                - send_recv.csv
            - h100_dgx
                - allreduce.csv
                - send_recv.csv
```

For compute profiling, only the SKU matters not the network configuration of the node. So, we don't discriminate between `a100_pair_nvlink` (Azure Standard_NC96ads_A100_v4 with 4 A100s) and `a100_dgx` (A100 DGX with 8 A100s), the same compute data is used in both in the folder called `a100`.
For network profiling, the network configuration of the node matters. So, we have separate folders for `a100_pair_nvlink` and `a100_dgx`. One example is that TP4 is different in these. In `a100_pair_nvlink`, there are two pairs connected by NVLink but between these pairs is a relatively slower link, but in `a100_dgx` where all 8 GPUs are connected by NVLink.

## Adding a new model

We need actual GPUs to get profiling data for a new model. Once the profiling is done, simulations can be run on CPUs only.

1. Use a CUDA environment with **vLLM 0.5.1**, for example the matching vLLM container.
    - Install compute-profiling dependencies with `python -m pip install -r requirements-profiling.txt`.
    - Install this repository with `python -m pip install -e . --no-deps`.
    - CPU-only simulation does not import vLLM; vLLM is needed for GPU data collection.
1. Register a model configuration in `vidur/config/model_config.py`, following the existing `BaseModelConfig` classes.
    - Set heads, KV heads, hidden/intermediate sizes, bias, activation, norm and positional-encoding settings from the model's Hugging Face configuration.
    - The collectors use the same `ModelConfig.from_model_name()` registry. There is no Llama-only model-type check.
    - Phi-2's partial RoPE and non-gated GELU/LayerNorm, GQA, KV-head replication and model-specific TP restrictions are honored.
    - The default model set includes Phi-2, InternLM-20B, Qwen-72B, CodeLlama-34B, Llama-2-7B/70B and Llama-3-8B/70B.
1. MLP and attention each have one native implementation. There is no separate attention-profile simulator path or old-implementation switch.
1. For compute profiling (mlp and attention), 1 GPU is enough even for tensor parallel degrees greater than 1. So `num_gpus` set to 1 is sufficient albeit slower for profiling.
1. Now we need to do the MLP profiling:

    ```bash
        python vidur/profiling/mlp/main.py \
        --models codellama/CodeLlama-34b-Instruct-hf \
        --num_gpus 4
    ```

    - Run `python vidur/profiling/mlp/main.py --help` for more options.
    - Copy the CSV file from `profiling_outputs/mlp/<timestamp>/codellama/CodeLlama-34b-Instruct-hf/mlp.csv` to `data/profiling/compute/a100/codellama/CodeLlama-34b-Instruct-hf/mlp.csv`.
1. Now we need to do the attention profiling:

    ```bash
        python vidur/profiling/attention/main.py \
        --models codellama/CodeLlama-34b-Instruct-hf \
        --num_gpus 4
    ```

    - Run `python vidur/profiling/attention/main.py --help` for more options.
    - Copy the CSV file from `profiling_outputs/attention/<timestamp>/codellama/CodeLlama-34b-Instruct-hf/attention.csv` to `data/profiling/compute/a100/codellama/CodeLlama-34b-Instruct-hf/attention.csv`.
    - Note that we are using `a100` as the device name. If you are using `h100` or some other device, then you need to create a new folder for that device in `data/profiling/compute` and copy the CSV files there.

### Single-GPU collection and timing scope

For an isolated GPU, `--disable_ray` runs the same collector in the current process:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vidur.profiling.mlp.main \
  --disable_ray --models meta-llama/Llama-2-7b-hf internlm/internlm-20b \
  --num_tensor_parallel_workers 1 2 4 8 --max_tokens 4096
CUDA_VISIBLE_DEVICES=0 python -m vidur.profiling.attention.main \
  --disable_ray --models meta-llama/Llama-2-7b-hf internlm/internlm-20b \
  --num_tensor_parallel_workers 1 2 4 8 --max_model_len 4096 \
  --max_seq_len 4096 --max_batch_size 8
```

`--max_model_len` is also the native decode graph capture limit. Decode block tables are padded to that limit, and batch sizes use vLLM's graph buckets (1, 2, 4, then multiples of 8). `--max_seq_len` selects the effective input-length range to sample. Set the capture limit to the one used by the native service.

The shared profiler captures the native GPU operations and sums **CUDA child-kernel durations only** using PyTorch/Kineto. Each operator has five measured rounds of ten graph replays, following warmup. Before each replay, a 128MiB buffer is read to condition L2 without dirty writeback traffic. These separately identified conditioning kernels and PyTorch graph bookkeeping are excluded. CPU parent events are not added to CUDA children. Per-round samples and min/max/mean/median/std are written to CSV. No endpoint-latency residual is used as a cost.

* `mlp.csv` contains projections, MLP operations, norms, residual add, embedding and `time_stats.attn_rope.*`. RoPE uses the configured rotary dimension, style and scaling.
* `attention.csv` contains `time_stats.attn_kv_cache_save.*` and prefill/decode attention, including split-KV combine. It supports first and cached/chunked prefill, packed QKV, and disjoint physical KV blocks for requests.
* Attention data records `forward_tokens`, `block_table_width`, `cuda_graph`, `max_model_len`, `profile_method` and vLLM version. One model/TP/block configuration must have one capture limit per input table.

The simulator consumes these tables through the existing `compute_input_file` and `attention_input_file` settings (or their standard `data/profiling/compute/{DEVICE}/{MODEL}/` paths). There are no `vllm_attention_profile_file`, `vllm_attention_decode_capture_limit`, or RoPE opt-out parameters. Missing KV-write measurements are rejected rather than replaced with zero. Regenerate old attention tables with the integrated collector.

TP values here select the **single-shard compute shape**; collectives are still profiled separately below. Operator timing excludes scheduling, sampling, logits and inter-kernel/engine gaps, so operator-level fixes alone do not guarantee endpoint TTFT accuracy.

### Validation on H100, 2026-10-08

The integrated attention and MLP entry points completed GPU smoke tests for all eight default models. Llama-2-7B shard profiling also completed at TP2/4/8. Serial execution, Ray execution, cached prefill and B3-to-4 decode graph padding were exercised. All 15 unit tests passed, including the existing communication-profile tests.

Independent kernel data was collected and replayed against the saved three-request native trace (Llama-2-7B, FP16, TP1/PP1, B1, input512/output50):

| Metric | Native vLLM | Original prediction | Integrated tables |
|---|---:|---:|---:|
| E2E, ms | 406.869 | 361.050 | 386.916 |
| TPOT, ms (49 decode intervals) | 7.893 | 7.060 | 7.602 |

The new E2E error is approximately -4.90% and TPOT error -3.68%. Predicted prefill completion is 14.407 ms versus client TTFT 20.125 ms; the endpoint/launch/outer-engine difference remains outside this operator-only repair. These results do not assert end-to-end accuracy for every profiled model.

During validation, a write-based cache-conditioning experiment inflated the B1 sum of four linear operators to 6.019 ms per 32 layers. Reading the conditioning buffer instead measured 4.938 ms; the separately traced native linear-kernel sum was 5.022 ms. The write-based experiment was rejected, rather than subtracting a fitted latency constant. Only the read-based implementation is retained.

Launches, CSVs (including per-round samples and kernel names), and replay results are retained on sim-101 at `/home/turbo_ops/changle/Files/analysis/integrated_profiling_20261008_172545/`. Copy the newly collected `mlp.csv` and `attention.csv` to the standard data paths before running the simulator; old attention tables without measured KV writes and execution shapes cannot be used.

## Network (Collectives) profiling

Network profiling is not dependent on the model 🎉. So, we can use the same network profiling data for all models. However, we need to ensure that the network profiling data is available for the node configuration we are using. If not, then we need to profile the network for the device. 1.

For network profiling, the node setup i.e. type of connectivity between the gpus matter. This is why we have the concept of `network_device`. The network_device is an informal name for the network configuration of the node. Eg: `a100_pair_nvlink`, `a100_dgx`, `h100_dgx` etc.
    1. For tensor parallelism, 4 GPUs are needed for TP4 and 8 GPUs are needed for TP8 etc.
    2. For pipeline parallelism across nodes, 2 nodes are needed to profile the link between the nodes.

Currently available data include:

- `a100_pair_nvlink`: Azure Standard_NC96ads_A100_v4 with 4 80GB A100 PCIe GPUs with pair-wise NVLINK connectivity.
- `h100_pair_nvlink`: Azure internal VM with 4 80GB H100 NVL GPUs with pair-wise NVLINK connectivity.
- `a100_dgx`: A100 DGX with 8 80GB A100s.
- `h100_dgx`: H100 DGX with 8 H100s.

### Steps to profile:

1. Clone this (`vidur`) repo and create a Python virtual environment as in [Setup](README.md).
1. Setup a ray cluster:
    1. Tensor parallelism is typically done on a single node so we don't need a multi-node cluster.
    1. However, pipeline parallelism is typically done across multiple nodes so we need at least 2 nodes there.
    1. Run `ray start --head` from the root node.
    1. Run `ray start --address <head-node-ip>:<head-node-port>` from the other nodes. The other nodes also need to have the same git commit checked out.
1. Run the following command to profile for the `all_reduce` operation, (sufficient for TP):

    ```bash
        python vidur/profiling/collectives/main.py \
        --num_workers_per_node_combinations 1,2,4,8 \
        --collective all_reduce
    ```

    - One may need to adjust `--num_workers_per_node_combinations` depending on the number of GPUs in the node eg. `--num_workers_per_node_combinations 1,2,4` for Azure Standard_NC96ads_A100_v4 node.
    - Copy the CSV file from `profiling_outputs/collectives/<timestamp>/all_reduce.csv` to `data/profiling/network/{network_device}/allreduce.csv`.
    - `network_device` is an informal name for the network configuration of the node. Eg: `a100_pair_nvlink`, `a100_dgx`, `h100_dgx` etc.
    - Run `python vidur/profiling/collectives/main.py --help` for more options.
1. Run the following command to profile for the `send_recv` operation, (required only for PP):

    ```bash
        python vidur/profiling/collectives/main.py \
        --num_workers_per_node_combinations 1 2 \
        --collective send_recv \
        --num_profile_rounds 60
    ```

    - Typically, PP is done across nodes so `num_workers_per_node_combinations` should be the same as the number of GPUs available in one node. Profiling `num_workers_per_node_combinations` less than the number of GPUs in the node to have PP inside a node. This can be useful when each gpu is not connected to every other gpu using the same high speed link.
    - Copy the CSV file from `profiling_outputs/collectives/<timestamp>/send_recv.csv` to `data/profiling/network/{network_device}/send_recv.csv`.
    - `network_device` is an informal name for the network configuration of the node. Eg: `a100_pair_nvlink`, `a100_dgx`, `h100_dgx` etc.

    - `send_recv` defaults to 60 profiling rounds per size/layout. An explicit positive `--num_profile_rounds` overrides this default. Each round takes the median NCCL event duration; `time_stats.send_recv.mean` averages those round timings in milliseconds and is the send/recv random forest training target. Reprofile existing tables to obtain a 60-round average. Other collectives retain their existing defaults and training targets.

## CPU Overhead Profiling

These include implementation overheads like scheduling time, sampling time, detokenization etc. For better fidelity, these should also be profiled. However, they tie the simulator closely to the implementation eg. `vLLM`. Scripts are available [here](vidur/profiling/cpu_overhead/) but not documented yet. These scripts follow a similar pattern to the compute and network profiling scripts.
