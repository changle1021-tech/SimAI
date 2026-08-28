/*
 * Copyright (c) 2024, Alibaba Group;
 * Licensed under the Apache License, Version 2.0.
 */
#include "astra-sim/system/ParallelGroupPolicy.h"

#include <stdexcept>

namespace MockNccl {

ParallelGroups build_vllm_parallel_groups(
    int world_size,
    int tensor_parallel_size,
    int data_parallel_size,
    int pipeline_parallel_size) {
  if (world_size <= 0 || tensor_parallel_size <= 0 ||
      data_parallel_size <= 0 || pipeline_parallel_size <= 0) {
    throw std::invalid_argument("parallel sizes must be positive");
  }
  if (tensor_parallel_size * data_parallel_size *
          pipeline_parallel_size !=
      world_size) {
    throw std::invalid_argument("TP * DP * PP must equal world size");
  }

  ParallelGroups groups;

  // TP: contiguous ranks within one DP replica and one pipeline stage.
  for (int dp = 0; dp < data_parallel_size; ++dp) {
    for (int pp = 0; pp < pipeline_parallel_size; ++pp) {
      std::vector<int> ranks;
      for (int tp = 0; tp < tensor_parallel_size; ++tp) {
        ranks.push_back(
            ((dp * pipeline_parallel_size) + pp) * tensor_parallel_size + tp);
      }
      groups.tensor_parallel.push_back(ranks);
    }
  }

  // DP: same PP stage and TP lane across data-parallel replicas.
  for (int pp = 0; pp < pipeline_parallel_size; ++pp) {
    for (int tp = 0; tp < tensor_parallel_size; ++tp) {
      std::vector<int> ranks;
      for (int dp = 0; dp < data_parallel_size; ++dp) {
        ranks.push_back(
            ((dp * pipeline_parallel_size) + pp) * tensor_parallel_size + tp);
      }
      groups.data_parallel.push_back(ranks);
    }
  }

  // PP: same DP replica and TP lane across pipeline stages.
  for (int dp = 0; dp < data_parallel_size; ++dp) {
    for (int tp = 0; tp < tensor_parallel_size; ++tp) {
      std::vector<int> ranks;
      for (int pp = 0; pp < pipeline_parallel_size; ++pp) {
        ranks.push_back(
            ((dp * pipeline_parallel_size) + pp) * tensor_parallel_size + tp);
      }
      groups.pipeline_parallel.push_back(ranks);
    }
  }

  // vLLM wide EP: all TP and DP ranks in the same pipeline stage.
  for (int pp = 0; pp < pipeline_parallel_size; ++pp) {
    std::vector<int> ranks;
    for (int dp = 0; dp < data_parallel_size; ++dp) {
      for (int tp = 0; tp < tensor_parallel_size; ++tp) {
        ranks.push_back(
            ((dp * pipeline_parallel_size) + pp) * tensor_parallel_size + tp);
      }
    }
    groups.expert_parallel.push_back(ranks);
  }

  return groups;
}

}  // namespace MockNccl
