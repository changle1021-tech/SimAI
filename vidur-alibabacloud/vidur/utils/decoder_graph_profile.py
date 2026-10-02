"""Measured native GPU DAG primitives, with interpolation inside profiled coverage."""
import bisect
import csv
import math
import statistics


class DecoderGraphProfile:
    def __init__(self, path, model_name, shape, tp, capacity, block_size, dtype="float16", device=None):
        self.values = {}
        samples = {}
        collective_modes = set()
        with open(path, newline="") as stream:
            for row in csv.DictReader(stream):
                if device and device.lower() not in row.get("device_name", "").lower():
                    continue
                if row.get("model_name") != model_name:
                    continue
                if any(int(row[k]) != v for k, v in shape.items()):
                    continue
                if (int(row["num_tensor_parallel_workers"]) != tp
                        or int(row["decode_block_table_capacity"]) != capacity
                        or int(row["block_size"]) != block_size
                        or row["dtype"] != dtype or row["execution_mode"] != "cuda_graph"
                        or row["phase"] != "decode"):
                    continue
                value_mode = row.get("includes_tp_collectives", "false").strip().lower()
                if value_mode not in ("true", "false", ""):
                    raise ValueError("Invalid decoder graph collective scope")
                includes_tp = value_mode == "true"
                if includes_tp:
                    import json
                    nodes = json.loads(row.get("rank_node_map", "null"))
                    if (nodes != [0] * tp or row.get("network_transport") != "local"
                            or row.get("tp_collective_backend") != "nccl"):
                        raise ValueError("Decoder graph TP scope needs verified single-node placement")
                collective_modes.add(includes_tp)
                layers, batch, kv = (int(row[k]) for k in ("num_layers", "batch_size", "kv_cache_size"))
                from .runtime_layout import graph_batch_size
                if int(row["physical_batch_size"]) != graph_batch_size(batch):
                    raise ValueError("Decoder graph physical batch does not match native graph padding")
                value = float(row["time_stats.decoder_graph.median"])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("Invalid native decoder GPU span")
                samples.setdefault((layers, batch, kv), []).append(value)
        if len(collective_modes) > 1:
            raise ValueError("Decoder graph profiles cannot mix included and excluded TP collectives")
        self.includes_tp_collectives = collective_modes == {True}
        for key, values in samples.items():
            self.values[key] = statistics.mean(values)
        if not self.values:
            raise ValueError("No native decoder graph profile matches the model and physical runtime descriptors")

    def predict(self, layers, batch, kv):
        grid = sorted(k[2] for k in self.values if k[:2] == (layers, batch))
        if not grid or kv < grid[0] or kv > grid[-1]:
            raise ValueError(f"Native decoder GPU profile lacks coverage for layers={layers}, batch={batch}, KV={kv}")
        index = bisect.bisect_left(grid, kv)
        if grid[index] == kv:
            return self.values[(layers, batch, grid[index])]
        lo, hi = grid[index-1], grid[index]
        fraction = (kv - lo) / (hi - lo)
        return (1-fraction) * self.values[(layers, batch, lo)] + fraction * self.values[(layers, batch, hi)]
