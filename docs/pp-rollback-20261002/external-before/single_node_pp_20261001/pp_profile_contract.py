"""Non-overlapping timings for a single batch traversing PP stages."""

CPU_METRICS = (
    "schedule", "prepare_inputs_e2e", "sampler_e2e",
    "process_model_outputs", "ray_comm_time", "engine_bookkeeping",
)


def aggregate_step(step, workers, tp, pp, legacy_residual=True):
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
    staging = sum(max(by_rank[s * tp + t].get("graph_input_staging_ms", 0.0)
                      for t in range(tp)) for s in range(pp))
    prepare = by_rank[0]["prepare_ms"]
    sample = by_rank[(pp - 1) * tp]["sampler_ms"]
    residual = (step["executor_ms"] - forward - staging - handoff - prepare - sample
                if legacy_residual else 0.0)
    if legacy_residual and residual < -0.05:
        raise ValueError(f"Overlapping timing regions: residual={residual:.6f} ms")
    result = {
        "schedule": step["schedule_ms"],
        "prepare_inputs_e2e": prepare,
        "sampler_e2e": sample,
        "process_model_outputs": step["process_outputs_ms"],
        "ray_comm_time": max(0.0, residual),
        "engine_bookkeeping": step.get("engine_bookkeeping_ms", 0.0),
        "pp_handoff_e2e": handoff,
        "model_execution_e2e": forward,
        "graph_input_staging_e2e": staging,
    }
    for stage in range(pp):
        result[f"graph_input_staging_stage_{stage}"] = max(
            by_rank[stage * tp + t].get("graph_input_staging_ms", 0.0) for t in range(tp))
    for boundary in range(pp - 1):
        result[f"pp_handoff_boundary_{boundary}"] = max(
            by_rank[boundary * tp + t]["send_ms"] for t in range(tp))
    return result


def request_key(req):
    groups = []
    for group in req.seq_group_metadata_list:
        seqs = tuple(sorted((sid, data.get_len())
                            for sid, data in group.seq_data.items()))
        groups.append((group.request_id, group.token_chunk_size, seqs))
    return repr((req.virtual_engine, tuple(groups)))


def bind_worker_keys(worker_rows, tp, pp):
    """Match controlled single-VE steps across TP metadata broadcasts.

    Non-driver TP workers receive tensors rather than ExecuteModelRequest.
    Their local ordered step index must agree with the stage driver. No extra
    profiling payload or synchronization is inserted into the native protocol.
    """
    by_rank = {records[0]['rank']: records for records in worker_rows if records}
    if set(by_rank) != set(range(tp * pp)):
        raise ValueError('Missing rank timing records')
    for stage in range(pp):
        driver = by_rank[stage * tp]
        for rank in range(stage * tp, (stage + 1) * tp):
            records = by_rank[rank]
            if len(records) != len(driver):
                raise ValueError('TP follower and driver step counts differ')
            for index, (record, source) in enumerate(zip(records, driver)):
                if record.get('step_index', index) != index:
                    raise ValueError('TP worker steps are out of order')
                if record['key'] is None:
                    record['key'] = source['key']
                if record['key'] != source['key']:
                    raise ValueError('TP worker key does not match its stage driver')
    return worker_rows
