/*
 * Copyright (c) 2024, Alibaba Group;
 * Licensed under the Apache License, Version 2.0.
 */
#ifndef __PARALLEL_GROUP_POLICY_H__
#define __PARALLEL_GROUP_POLICY_H__

#include <vector>

namespace MockNccl {

struct ParallelGroups {
  std::vector<std::vector<int>> tensor_parallel;
  std::vector<std::vector<int>> data_parallel;
  std::vector<std::vector<int>> pipeline_parallel;
  std::vector<std::vector<int>> expert_parallel;
};

// Match vLLM's DP x PP x TP rank layout:
//   rank = ((dp_rank * pipeline_parallel_size) + pp_rank) *
//          tensor_parallel_size + tp_rank
// Expert parallelism is wide EP: one TP x DP group per pipeline stage.
ParallelGroups build_vllm_parallel_groups(
    int world_size,
    int tensor_parallel_size,
    int data_parallel_size,
    int pipeline_parallel_size);

}  // namespace MockNccl

#endif
