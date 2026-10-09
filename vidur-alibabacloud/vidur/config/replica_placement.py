"""Node-packed placement with the TP/PP rank groups of vLLM 0.5.1."""

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class ReplicaPlacement:
    tensor_parallel_size: int
    num_pipeline_stages: int
    num_nodes: int
    gpus_per_node: int

    def __post_init__(self):
        for name in ("tensor_parallel_size", "num_pipeline_stages", "num_nodes", "gpus_per_node"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.num_nodes * self.gpus_per_node < self.world_size:
            raise ValueError(
                f"TP * PP = {self.world_size} exceeds node capacity "
                f"({self.num_nodes} * {self.gpus_per_node})"
            )

    @property
    def world_size(self):
        return self.tensor_parallel_size * self.num_pipeline_stages

    def rank_location(self, rank):
        if type(rank) is not int or not 0 <= rank < self.world_size:
            raise ValueError(f"Invalid rank {rank!r} for world_size={self.world_size}")
        return divmod(rank, self.gpus_per_node)

    @property
    def tp_groups(self):
        tp = self.tensor_parallel_size
        return tuple(tuple(range(stage * tp, (stage + 1) * tp))
                     for stage in range(self.num_pipeline_stages))

    @property
    def pp_groups(self):
        return tuple(tuple(range(lane, self.world_size, self.tensor_parallel_size))
                     for lane in range(self.tensor_parallel_size))

    def _validate_stage(self, pipeline_stage):
        if type(pipeline_stage) is not int or not 0 <= pipeline_stage < self.num_pipeline_stages:
            raise ValueError(f"Invalid pipeline stage {pipeline_stage!r}")

    def tp_devices_per_node(self, pipeline_stage):
        """Return the profile's participating GPUs per node, not SKU capacity."""
        self._validate_stage(pipeline_stage)
        counts = Counter(self.rank_location(rank)[0] for rank in self.tp_groups[pipeline_stage])
        if len(set(counts.values())) != 1:
            raise ValueError(
                f"TP stage {pipeline_stage} has uneven GPUs per node {tuple(counts.values())}; "
                "the all_reduce CSV schema requires a uniform devices_per_node. "
                "Choose GPUs per node dividing TP, or a multiple of TP."
            )
        return next(iter(counts.values()))

    def pp_devices_per_node(self, pipeline_stage):
        """Select inter-node (1) or intra-node (2) send/recv profiling."""
        self._validate_stage(pipeline_stage)
        if pipeline_stage == self.num_pipeline_stages - 1:
            raise ValueError("The last pipeline stage has no outgoing PP edge")
        tp = self.tensor_parallel_size
        for lane in range(tp):
            source = pipeline_stage * tp + lane
            if self.rank_location(source)[0] != self.rank_location(source + tp)[0]:
                return 1
        return 2
