"""Non-overlapping timings for a single batch traversing PP stages."""

CPU_METRICS = (
    "schedule", "prepare_inputs_e2e", "sampler_e2e",
    "process_model_outputs", "ray_comm_time",
)


def aggregate_step(step, workers, tp, pp):
    if len(workers) != tp * pp:
        raise ValueError("Missing worker records for a profiled step")
    by_rank = {w["rank"]: w for w in workers}
    if set(by_rank) != set(range(tp * pp)):
        raise ValueError("Missing or duplicate worker ranks")
    if any(w["key"] != step["key"] for w in workers):
        raise ValueError("Worker records belong to a different step")
    forward = sum(max(by_rank[s * tp + t]["forward_gpu_ms"]
                      for t in range(tp)) for s in range(pp))
    handoff = sum(max(by_rank[s * tp + t]["send_ms"]
                      for t in range(tp)) for s in range(pp - 1))
    prepare = by_rank[0]["prepare_ms"]
    sample = by_rank[(pp - 1) * tp]["sampler_ms"]
    residual = step["executor_ms"] - forward - handoff - prepare - sample
    if residual < -0.05:
        raise ValueError(f"Overlapping timing regions: residual={residual:.6f} ms")
    return {
        "schedule": step["schedule_ms"],
        "prepare_inputs_e2e": prepare,
        "sampler_e2e": sample,
        "process_model_outputs": step["process_outputs_ms"],
        "ray_comm_time": max(0.0, residual),
        "pp_handoff_e2e": handoff,
        "model_execution_e2e": forward,
    }


def request_key(req):
    groups = []
    for group in req.seq_group_metadata_list:
        seqs = tuple(sorted((sid, data.get_len())
                            for sid, data in group.seq_data.items()))
        groups.append((group.request_id, group.token_chunk_size, seqs))
    return repr((req.virtual_engine, tuple(groups)))
