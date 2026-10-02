# vLLM 0.5.1 PP CPU profiling

`vidur/profiling/cpu_overhead/vllm_051_cpu_overhead.py` accepts
`--tensor-parallel-size` and `--pipeline-parallel-size`. PP1 retains its existing
synchronous collection path. PP > 1 uses vLLM's asynchronous Ray executor and
collects separate prefill/decode rows. One fixed batch is assigned to VE0;
virtual-engine concurrency is not used to pretend that two smaller batches are
one large batch.

```bash
python3 /vllm-workspace/vidur_cpu_overhead/vllm_051_cpu_overhead.py \
  --model /mnt/data02/000000/model/Llama-2-7b-hf-bin-export \
  --model-name-for-vidur meta-llama/Llama-2-7b-hf \
  --tensor-parallel-size 1 --pipeline-parallel-size 2 \
  --executor-backend ray --load-format auto \
  --batch-sizes 1 2 3 4 5 6 7 8 \
  --prompt-tokens 512 --decode-tokens 50 --repetitions 5 \
  --max-model-len 4096 --max-num-seqs 8 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.2 \
  --output /root/changle/Files/vidur_cpu_overhead/cpu_overheads.csv \
  --append
```

Run each TP/PP combination separately; visible GPUs must cover TP × PP.
Appending upgrades legacy rows with `pipeline_parallel_degree=1` and retains
their values. Additional PP rows may share the same CSV. The four profiler
modules must remain in the same directory; the entry point exports that path
for Ray actors automatically.

CPU table selection uses model, TP and PP. Legacy tables without a PP column
are accepted only for PP1. A missing PP profile raises a descriptive error
instead of silently using PP1. The phase-aware table uses batch size and
prefill tokens per request as prediction features; more prompt lengths can be
collected with repeated `--prompt-tokens ... --append` runs.

The prefill token budget and KV capacity must accommodate the requested fixed
batch. For example, batch=256 and prompt=256 requires at least 65536 prefill
tokens in the token budget. The PP profiler rejects mismatched observed batch
sizes rather than labeling partial batches as the requested full batch.

## Timing boundaries

CUDA events are read after all requests complete. There is no per-step GPU
synchronization inserted between model compute and metadata transmission.
Separate uninstrumented repetitions quantify instrumentation perturbation.

- `model_execution_e2e`: sum of maximum forward GPU interval in each PP stage.
- `pp_handoff_e2e`: sum of maximum interval from forward completion to
  tensor_dict send completion at each non-final boundary. This contains the
  non-overlapped part of metadata, both tensors and their synchronization.
- `sampler_e2e`: final stage interval from forward completion to completion of
  compute_logits and sample. Previous model GPU time is not counted twice.
- `prepare_inputs_e2e`: first driver input preparation.
- `ray_comm_time`: executor wall time minus forward, first prepare, final
  sampling and PP handoff; includes control/return and other critical-path work.

Worker receive duration is audit-only: it includes waiting for upstream model
compute and must not be treated as additional communication time.
The handoff profile replaces bare send_recv prediction. It is excluded from
the executor residual, so the two terms are added once. For PP > 2 the aggregate
handoff is spread across the PP-1 boundaries; heterogeneous boundary placement
and concurrent virtual engines need additional validation.

The existing GPU operator predictor is not replaced by these CPU measurements.
Remaining GPU compute or HTTP service gaps can therefore remain after this fix.
