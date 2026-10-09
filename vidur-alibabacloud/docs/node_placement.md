# Configurable TP/PP node placement

Vidur accepts two optional flags for each model replica:

- `--replica_config_num_nodes`: available physical nodes.
- `--replica_config_num_gpus_per_node`: GPUs usable on each node, up to the network SKU's capacity (8 for `h100_dgx`).

For example, append these flags to your existing Vidur command to place TP=2, PP=2 on two H100 nodes using two GPUs per node:

```bash
--replica_config_device h100 \
--replica_config_network_device h100_dgx \
--replica_config_num_nodes 2 \
--replica_config_num_gpus_per_node 2 \
--replica_config_tensor_parallel_size 2 \
--replica_config_num_pipeline_stages 2
```

TP and PP remain explicit. Model world size is `TP * PP`, and must not exceed `num_nodes * num_gpus_per_node`. Omitting GPUs per node uses the SKU capacity; omitting nodes infers `ceil(TP * PP / GPUs_per_node)`. Extra capacity is unused. Nodes are specified per replica; this does not introduce vLLM data parallelism. For a single vLLM 0.5.1 engine, use the existing default `--cluster_config_num_replicas 1`.

## Rank placement

The group definitions follow [vLLM 0.5.1 initialize_model_parallel](https://github.com/vllm-project/vllm/blob/v0.5.1/vllm/distributed/parallel_state.py):

```text
rank = pp_rank * TP + tp_rank
node_id, local_rank = divmod(rank, GPUs_per_node)
```

TP groups contain consecutive ranks. PP groups contain ranks with the same `tp_rank`, separated by TP. Vidur packs ranks in node order. vLLM's Ray executor assigns ranks through placement group bundles; Ray does not guarantee this physical order for every cluster. Arrange the real workers in the same node order when comparing runs.

| Nodes × GPUs/node | TP × PP | TP groups | PP groups | PP edges |
| --- | --- | --- | --- | --- |
| 2 × 2 | 2 × 2 | (0,1), (2,3) | (0,2), (1,3) | inter-node |
| 2 × 4 | 2 × 4 | (0,1), (2,3), (4,5), (6,7) | (0,2,4,6), (1,3,5,7) | intra, inter, intra |
| 2 × 4 | 8 × 1 | (0,1,2,3,4,5,6,7) | singleton groups | none |

## Communication profiles and SimAI

TP all-reduce selects CSV rows matching `num_workers=TP` and the number of participating TP GPUs per node. PP selects two-worker send/recv rows with `devices_per_node=2` for intra-node edges or `1` for inter-node edges. Each outgoing PP edge uses its own placement; mixed layouts train separate models. If an edge has both intra-node and inter-node lanes, its completion uses the inter-node profile. Missing measurements raise an error identifying the required profile, rather than using a different layout's timings.

The current all-reduce CSV schema represents uniform GPUs per participating node. Uneven TP splits, such as 3+1, need a richer profiling schema and are rejected by the CSV predictor. Choose a GPUs-per-node value dividing TP or a multiple of TP for uniform TP groups.

The analytical backend passes the configured value to `-g_p_s`. The simulation backend still requires an explicit topology file; when either new flag is provided, it checks that the topology has the same GPUs per server and enough GPUs for TP × PP. The flags do not generate a physical network topology. Communication caches include TP, PP, node count and GPUs per node so results for different placements do not collide.
