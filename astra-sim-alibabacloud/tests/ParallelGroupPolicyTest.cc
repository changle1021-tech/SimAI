#include "astra-sim/system/ParallelGroupPolicy.h"

#include <cassert>
#include <stdexcept>
#include <vector>

using MockNccl::ParallelGroups;

int main() {
  // world=16, TP=2, PP=2, DP=4. EP size must be TP * DP = 8.
  const ParallelGroups groups =
      MockNccl::build_vllm_parallel_groups(16, 2, 4, 2);

  assert(groups.tensor_parallel.size() == 8);
  assert((groups.tensor_parallel[0] == std::vector<int>{0, 1}));
  assert((groups.tensor_parallel[1] == std::vector<int>{2, 3}));
  assert((groups.tensor_parallel[7] == std::vector<int>{14, 15}));

  assert(groups.data_parallel.size() == 4);
  assert((groups.data_parallel[0] == std::vector<int>{0, 4, 8, 12}));
  assert((groups.data_parallel[3] == std::vector<int>{3, 7, 11, 15}));

  assert(groups.pipeline_parallel.size() == 8);
  assert((groups.pipeline_parallel[0] == std::vector<int>{0, 2}));
  assert((groups.pipeline_parallel[7] == std::vector<int>{13, 15}));

  assert(groups.expert_parallel.size() == 2);
  assert((groups.expert_parallel[0] ==
          std::vector<int>{0, 1, 4, 5, 8, 9, 12, 13}));
  assert((groups.expert_parallel[1] ==
          std::vector<int>{2, 3, 6, 7, 10, 11, 14, 15}));

  bool rejected = false;
  try {
    MockNccl::build_vllm_parallel_groups(16, 2, 3, 2);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
  return 0;
}
