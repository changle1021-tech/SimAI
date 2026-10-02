"""Physical execution descriptors; none are fitted to request latency."""
import json
import math


def graph_batch_size(size):
    if size < 1:
        raise ValueError("Batch size must be positive")
    if size <= 2:
        return size
    if size <= 4:
        return 4
    return math.ceil(size / 8) * 8


def canonical_rank_nodes(nodes):
    aliases = {}
    return [aliases.setdefault(node, len(aliases)) for node in nodes]


def configured_rank_nodes(world_size, devices_per_node, encoded=None):
    if world_size < 1 or devices_per_node < 1:
        raise ValueError("GPU world size and node capacity must be positive")
    if encoded is None:
        return [rank // devices_per_node for rank in range(world_size)]
    nodes = json.loads(encoded) if isinstance(encoded, str) else list(encoded)
    if len(nodes) != world_size or any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in nodes):
        raise ValueError("rank_node_map must contain one non-negative node index per rank")
    result = canonical_rank_nodes(nodes)
    if any(result.count(node) > devices_per_node for node in set(result)):
        raise ValueError("rank_node_map exceeds the configured GPUs per node")
    return result


def pipeline_layer_bounds(num_layers, pp_rank, pp_size):
    """vLLM 0.5.1 partitioning: the final stage owns remainder layers."""
    if num_layers < 1 or pp_size < 1 or not 0 <= pp_rank < pp_size:
        raise ValueError("Invalid layer count or pipeline rank")
    base = num_layers // pp_size
    start = pp_rank * base
    return start, num_layers if pp_rank == pp_size - 1 else start + base
