"""Contract tests for the PP/TP CPU overhead aggregation helper."""

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / 'vidur/profiling/cpu_overhead/pp_profile_contract.py'
_SPEC = importlib.util.spec_from_file_location("pp_profile_contract", MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load contract module at {MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
aggregate_step = _MODULE.aggregate_step


def worker(key, rank, forward, prepare, sampler, send):
    return {
        "key": key,
        "rank": rank,
        "forward_gpu_ms": forward,
        "prepare_ms": prepare,
        "sampler_ms": sampler,
        "send_ms": send,
    }


def step(key="step-0", executor=100):
    return {
        "key": key,
        "phase": "decode",
        "batch_size": 1,
        "prefill_tokens_per_request": 0,
        "executor_ms": executor,
        "schedule_ms": 3,
        "process_outputs_ms": 4,
    }


class AggregateStepContractTests(unittest.TestCase):
    def test_tp1_pp2_expected_aggregation(self):
        step_row = step()
        workers = [
            worker("step-0", 0, forward=20, prepare=5, sampler=99, send=7),
            worker("step-0", 1, forward=30, prepare=8, sampler=11, send=9),
        ]

        result = aggregate_step(step_row, workers, tp=1, pp=2)

        self.assertEqual(result["schedule"], 3)
        self.assertEqual(result["prepare_inputs_e2e"], 5)
        self.assertEqual(result["sampler_e2e"], 11)
        self.assertEqual(result["process_model_outputs"], 4)
        self.assertEqual(result["ray_comm_time"], 27)
        self.assertEqual(result["pp_handoff_e2e"], 7)

    def test_tp2_pp2_uses_max_within_each_pipeline_stage(self):
        step_row = step()
        workers = [
            worker("step-0", 0, forward=20, prepare=5, sampler=99, send=7),
            worker("step-0", 1, forward=25, prepare=6, sampler=98, send=8),
            worker("step-0", 2, forward=30, prepare=8, sampler=11, send=9),
            worker("step-0", 3, forward=35, prepare=9, sampler=12, send=10),
        ]

        result = aggregate_step(step_row, workers, tp=2, pp=2)

        # TP ranks are synchronized, so each stage contributes its slowest
        # rank: max(20, 25) + max(30, 35).
        self.assertEqual(result["model_execution_e2e"], 60)
        self.assertEqual(result["prepare_inputs_e2e"], 5)
        self.assertEqual(result["sampler_e2e"], 11)
        # Stage 0 is non-final and uses max(send_ms ranks 0, 1) = 8 ms.
        self.assertEqual(result["pp_handoff_e2e"], 8)
        self.assertEqual(result["ray_comm_time"], 16)

    def test_missing_rank_raises_value_error(self):
        step_row = step()
        with self.assertRaises(ValueError):
            aggregate_step(
                step_row,
                [worker("step-0", 0, 20, 5, 99, 7)],
                tp=1,
                pp=2,
            )

    def test_duplicate_rank_raises_value_error(self):
        step_row = step()
        with self.assertRaises(ValueError):
            aggregate_step(
                step_row,
                [worker("step-0", 0, 20, 5, 99, 7), worker("step-0", 0, 30, 8, 11, 9)],
                tp=1,
                pp=2,
            )

    def test_materially_negative_residual_raises_value_error(self):
        step_row = step(executor=1)
        workers = [
            worker("step-0", 0, forward=80, prepare=50, sampler=20, send=10),
            worker("step-0", 1, forward=80, prepare=50, sampler=20, send=10),
        ]
        with self.assertRaises(ValueError):
            aggregate_step(step_row, workers, tp=1, pp=2)

    def test_worker_key_mismatch_raises_value_error(self):
        step_row = step()
        workers = [
            worker("step-0", 0, forward=20, prepare=5, sampler=99, send=7),
            worker("other-step", 1, forward=30, prepare=8, sampler=11, send=9),
        ]
        with self.assertRaises(ValueError):
            aggregate_step(step_row, workers, tp=1, pp=2)


if __name__ == "__main__":
    unittest.main()
