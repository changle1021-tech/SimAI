"""CPU-only validation tests for the standalone Qwen profile readers."""
import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "vidur/entities/qwen_aicb_profiles.py"
_spec = importlib.util.spec_from_file_location("qwen_aicb_profiles", str(MODULE_PATH))
_profiles = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_profiles)
QwenAICBProfiles = _profiles.QwenAICBProfiles
RuntimeBoundaryProfile = _profiles.RuntimeBoundaryProfile
RuntimeMoECommunication = _profiles.RuntimeMoECommunication


class ProfileFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "aicb"
        producer = self.root / "workload_generator/mocked_model/inference"
        producer.mkdir(parents=True)
        for name in ("qwen3_next_projections.py", "AiobQwen3Next.py",
                     "qwen3_next_attention.py", "qwen3_next_moe.py"):
            (producer / name).write_text("# dummy " + name)
        self.directory = self.root / "results/workload"
        self.directory.mkdir(parents=True)

    def write_profile(self, seq, phase="decode", context=0, value=None,
                      invocation_update=None, metadata_update=None,
                      payload_update=None, filename=None):
        invocation = dict(model_name="Qwen3-Next-80B", world_size=4,
                          tensor_model_parallel_size=1, pipeline_model_parallel=1,
                          expert_model_parallel_size=4, micro_batch=1,
                          phase=phase, seq_length=seq, context_length=context)
        invocation.update(invocation_update or {})
        value = float(seq + 2 * context) if value is None else value
        rows = ["layer_id\tlayer_name\tcomp_time\tcomm_size"]
        for layer in range(48):
            for component in ("attention", "moe"):
                rows.append("{}\t{}\t{}\t{}".format(layer, component, value, value + 1))
        payload = ("\n".join(rows) + "\n").encode()
        if payload_update:
            payload = payload_update(payload)
        if filename is None:
            filename = ("vidur-Qwen3-Next-80B-world_size4-tp1-pp1-ep4-bs1-seq{}-{}-{}.csv"
                        .format(seq, context, phase) if phase == "prefill" else
                        "vidur-Qwen3-Next-80B-world_size4-tp1-pp1-ep4-bs1-seq{}-{}.csv".format(seq, phase))
        path = self.directory / filename
        path.write_bytes(payload)
        metadata = dict(projection_profile_version=3, scope="qwen3_next_components",
                        source_sha256=_profiles.producer_identity(self.root),
                        profile_sha256=hashlib.sha256(payload).hexdigest(),
                        precision="bf16", invocation=invocation)
        metadata.update(metadata_update or {})
        Path(str(path) + ".projections.json").write_text(json.dumps(metadata))
        return path

    def write_json(self, data):
        path = Path(self.temp.name) / "runtime.json"
        path.write_text(json.dumps(data))
        return path


class QwenAICBProfileTests(ProfileFixtures):
    def test_metadata_source_version_scope_precision_payload_and_invocation_are_checked(self):
        variants = [
            ({"source_sha256": "wrong"}, {}, "source changed"),
            ({"projection_profile_version": 2}, {}, "Legacy"),
            ({"scope": "all_layers"}, {}, "Legacy"),
            ({"precision": "fp8"}, {}, "FP8"),
            ({"profile_sha256": "wrong"}, {}, "hash differs"),
            ({}, {"world_size": 8}, "mismatch"),
        ]
        for metadata, invocation, message in variants:
            with self.subTest(metadata=metadata, invocation=invocation):
                self.directory.mkdir(parents=True, exist_ok=True)
                self.write_profile(16, metadata_update=metadata,
                                   invocation_update=invocation)
                with self.assertRaisesRegex(ValueError, message):
                    QwenAICBProfiles(self.root).load(4, 1, 1, 4, 1, 16, "decode")
                for path in self.directory.iterdir():
                    path.unlink()

    def test_csv_requires_all_48_layers_two_components_and_finite_positive_compute(self):
        cases = [
            (lambda b: b.replace(b"0\tmoe", b"0\tattention", 1), "Duplicate"),
            (lambda b: b.replace(b"47\tmoe\t16.0\t17.0\n", b""), "48"),
            (lambda b: b.replace(b"0\tattention\t16.0\t17.0", b"0\tattention\tnan\t17.0"), "Nonfinite"),
            (lambda b: b.replace(b"0\tattention\t16.0\t17.0", b"0\tattention\t-1\t17.0"), "Nonfinite"),
            (lambda b: b.replace(b"0\tattention\t16.0\t17.0", b"0\tattention\t0\t17.0"), "zero compute"),
        ]
        for mutate, message in cases:
            with self.subTest(message=message):
                payload_update = lambda b, mutate=mutate: mutate(b)
                self.write_profile(16, payload_update=payload_update)
                with self.assertRaisesRegex(ValueError, message):
                    QwenAICBProfiles(self.root).load(4, 1, 1, 4, 1, 16, "decode")
                for path in self.directory.iterdir():
                    path.unlink()

    def test_decode_exact_points_and_interpolation_ignore_request_history(self):
        self.write_profile(10, value=10)
        self.write_profile(20, value=20)
        profiles = QwenAICBProfiles(self.root)
        exact = profiles.load(4, 1, 1, 4, 1, 10, "decode")
        self.assertEqual(exact[0]["attention"]["comp_time"], 10)
        self.assertEqual(profiles.load(4, 1, 1, 4, 1, 20, "decode")[0]["moe"]["comp_time"], 20)
        interpolated = profiles.load(4, 1, 1, 4, 1, 15, "decode")
        fresh = QwenAICBProfiles(self.root).load(4, 1, 1, 4, 1, 15, "decode")
        self.assertEqual(interpolated, fresh)
        self.assertEqual(interpolated[0]["attention"]["comp_time"], 15)

    def test_prefill_interpolates_context_then_chunk_size(self):
        for seq in (10, 20):
            for context in (0, 100):
                self.write_profile(seq, phase="prefill", context=context)
        result = QwenAICBProfiles(self.root).load(4, 1, 1, 4, 1, 15, "prefill", context=25)
        self.assertEqual(result[0]["attention"]["comp_time"], 65)

    def test_moe_time_scope_sets_ep_communication_flag_and_rejects_unknown_scope(self):
        self.write_profile(16, metadata_update={"moe_time_scope": "compute_only"})
        data = QwenAICBProfiles(self.root).load(4, 1, 1, 4, 1, 16, "decode")
        self.assertEqual(data[0]["moe"]["includes_ep_comm"], 0.0)
        for path in self.directory.iterdir():
            path.unlink()

        self.write_profile(16, metadata_update={"moe_time_scope": "compute_and_ep_comm"})
        data = QwenAICBProfiles(self.root).load(4, 1, 1, 4, 1, 16, "decode")
        self.assertEqual(data[0]["moe"]["includes_ep_comm"], 1.0)
        for path in self.directory.iterdir():
            path.unlink()

        self.write_profile(16, metadata_update={"moe_time_scope": "unknown"})
        with self.assertRaisesRegex(ValueError, "Unknown Qwen MoE timing scope"):
            QwenAICBProfiles(self.root).load(4, 1, 1, 4, 1, 16, "decode")

    def test_decode_interpolation_rejects_mixed_moe_timing_scopes(self):
        self.write_profile(10, metadata_update={"moe_time_scope": "compute_only"})
        self.write_profile(20, metadata_update={"moe_time_scope": "compute_and_ep_comm"})
        with self.assertRaisesRegex(ValueError, "different MoE timing scopes"):
            QwenAICBProfiles(self.root).load(4, 1, 1, 4, 1, 15, "decode")

    def test_out_of_range_never_uses_nearest_point_and_batch_pp_are_rejected(self):
        self.write_profile(10)
        self.write_profile(20)
        profiles = QwenAICBProfiles(self.root)
        for query in (9, 21):
            with self.subTest(query=query), self.assertRaisesRegex(ValueError, "outside measured range"):
                profiles.load(4, 1, 1, 4, 1, query, "decode")
        with self.assertRaisesRegex(ValueError, "batch1 and PP1"):
            profiles.load(4, 1, 2, 4, 1, 15, "decode")
        with self.assertRaisesRegex(ValueError, "batch1 and PP1"):
            profiles.load(4, 1, 1, 4, 2, 15, "decode")

    def test_prefill_context_out_of_range_is_rejected(self):
        for context in (10, 20):
            self.write_profile(16, phase="prefill", context=context)
        with self.assertRaisesRegex(ValueError, "outside measured range"):
            QwenAICBProfiles(self.root).load(4, 1, 1, 4, 1, 16, "prefill", context=5)


class RuntimeProfileTests(ProfileFixtures):
    def boundary_data(self):
        return dict(schema="qwen_worker_critical_perimeter_v2",
                    scope="host_iteration_wall_minus_model_gpu",
                    configuration=dict(model="qwen3-next-80B", world_size=4, tp=1,
                                       pp=1, ep=4, batch_size=1),
                    phases={"decode": [
                        dict(tokens=1, critical_perimeter_ms=1.5),
                        dict(tokens=3, critical_perimeter_ms=3.5)],
                            "prefill": [
                        dict(tokens=1, critical_perimeter_ms=11.0),
                        dict(tokens=3, critical_perimeter_ms=31.0)]})

    def request_boundary_data(self):
        data = self.boundary_data()
        data.update(schema="qwen_worker_request_perimeter_v3",
                    scope="host_iteration_wall_minus_model_gpu_plus_request_io",
                    request_io_ms=3.0, inter_chunk_gap_ms=2.0, inter_decode_gap_ms=0.5)
        data["phases"] = {
            "prefill_first": [dict(tokens=8, critical_perimeter_ms=4.0),
                              dict(tokens=16, critical_perimeter_ms=6.0)],
            "prefill_next": [dict(tokens=8, critical_perimeter_ms=6.0)],
            "decode": [dict(tokens=1, critical_perimeter_ms=9.0)],
        }
        return data

    def test_request_perimeter_adds_io_once_and_separate_iteration_gaps(self):
        profile = RuntimeBoundaryProfile(self.write_json(self.request_boundary_data()))
        args = (4, 1, 1, 4)
        self.assertEqual(profile.estimate("prefill", 16, *args, context=0), 9.0)
        self.assertEqual(profile.estimate("prefill", 8, *args, context=25), 8.0)
        self.assertEqual(profile.estimate("decode", 1, *args, context=0), 9.5)

    def test_request_perimeter_rejects_negative_io_gap_and_context(self):
        args = (4, 1, 1, 4)
        data = self.request_boundary_data()
        data["request_io_ms"] = -0.1
        profile = RuntimeBoundaryProfile(self.write_json(data))
        with self.assertRaisesRegex(ValueError, "Invalid request IO or inter-iteration time"):
            profile.estimate("prefill", 8, *args, context=0)

        data = self.request_boundary_data()
        data["inter_chunk_gap_ms"] = -0.1
        profile = RuntimeBoundaryProfile(self.write_json(data))
        with self.assertRaisesRegex(ValueError, "Invalid request IO or inter-iteration time"):
            profile.estimate("prefill", 8, *args, context=25)

        profile = RuntimeBoundaryProfile(self.write_json(self.request_boundary_data()))
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            profile.estimate("prefill", 8, *args, context=-1)

    def test_worker_boundary_validates_schema_scope_configuration_and_phase_interpolates(self):
        data = self.boundary_data()
        profile = RuntimeBoundaryProfile(self.write_json(data))
        self.assertEqual(profile.estimate("decode", 2, 4, 1, 1, 4), 2.5)
        self.assertEqual(profile.estimate("prefill", 2, 4, 1, 1, 4), 21.0)
        for update, message in (({"schema": "qwen_worker_perimeter_v1"}, "Unsupported"),
                                ({"scope": "model_only"}, "boundary")):
            invalid = dict(data)
            invalid.update(update)
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, message):
                RuntimeBoundaryProfile(self.write_json(invalid))
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            profile.estimate("decode", 2, 8, 1, 1, 4)

    def test_worker_boundary_rejects_invalid_times_and_unmeasured_range(self):
        data = self.boundary_data()
        data["phases"]["decode"][0]["critical_perimeter_ms"] = float("nan")
        profile = RuntimeBoundaryProfile(self.write_json(data))
        with self.assertRaisesRegex(ValueError, "Invalid worker perimeter time"):
            profile.estimate("decode", 1, 4, 1, 1, 4)
        data["phases"]["decode"][0]["critical_perimeter_ms"] = 1.5
        profile = RuntimeBoundaryProfile(self.write_json(data))
        with self.assertRaisesRegex(ValueError, "outside measured range"):
            profile.estimate("decode", 4, 4, 1, 1, 4)

    def test_moe_communication_uses_phase_modes_interpolates_and_validates_padded_profiles(self):
        data = dict(world_size=4, dtype="bfloat16", profiles=[])
        for mode, timing in (("eager", 1.0), ("cuda_graph", 5.0)):
            for tokens, value in ((2, timing), (4, timing + 2)):
                data["profiles"].append(dict(mode=mode, active_tokens=tokens,
                                             sizes=[tokens] * 4, median_ms=value))
        profile = RuntimeMoECommunication(self.write_json(data))
        args = (4, 1, 1, 4)
        self.assertEqual(profile.estimate("prefill", 3, *args), 2.0)
        self.assertEqual(profile.estimate("decode", 3, *args), 6.0)
        with self.assertRaisesRegex(ValueError, "WS4 TP1 PP1 EP4"):
            profile.estimate("decode", 3, 8, 1, 1, 4)
        data["profiles"].append(dict(mode="cuda_graph", active_tokens=8,
                                     sizes=[8, 8, 7, 8], median_ms=1.0))
        profile = RuntimeMoECommunication(self.write_json(data))
        with self.assertRaisesRegex(ValueError, "outside measured range"):
            profile.estimate("decode", 8, *args)
        data["dtype"] = "fp8"
        with self.assertRaisesRegex(ValueError, "four-rank BF16"):
            RuntimeMoECommunication(self.write_json(data))

    def test_moe_communication_rejects_invalid_selected_timing_and_out_of_range(self):
        data = dict(world_size=4, dtype="bfloat16", profiles=[
            dict(mode="eager", active_tokens=2, sizes=[2] * 4, median_ms=1.0),
            dict(mode="eager", active_tokens=4, sizes=[4] * 4, median_ms=0.0)])
        profile = RuntimeMoECommunication(self.write_json(data))
        with self.assertRaisesRegex(ValueError, "Invalid MoE communication timing"):
            profile.estimate("prefill", 2, 4, 1, 1, 4)
        data["profiles"][1]["median_ms"] = 3.0
        profile = RuntimeMoECommunication(self.write_json(data))
        with self.assertRaisesRegex(ValueError, "outside measured range"):
            profile.estimate("prefill", 1, 4, 1, 1, 4)


if __name__ == "__main__":
    unittest.main()
