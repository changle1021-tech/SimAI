"""Ensure vLLM communication profiles export Vidur-compatible CSV tables."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCRIPT = ROOT / "vidur" / "profiling" / "collectives" / "vllm_051_comm_profile.py"
REFERENCE_DIR = ROOT / "data" / "profiling" / "network" / "h100_dgx"


def load_profile_script():
    spec = importlib.util.spec_from_file_location("collectives_csv_profile", PROFILE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROFILE = load_profile_script()


class CollectivesCsvSchemaTests(unittest.TestCase):
    def test_export_matches_vidur_schema_and_preserves_order_and_measurements(self):
        for collective in ("send_recv", "all_reduce"):
            with self.subTest(collective=collective), tempfile.TemporaryDirectory() as tmp:
                reference = REFERENCE_DIR / f"{collective}.csv"
                with reference.open(newline="") as stream:
                    expected_header = next(csv.reader(stream))

                operation = collective
                rows = []
                for index, size_bytes in enumerate((1024, 2048, 1024)):
                    rows.append({
                        "time_stats": {
                            operation: {"min": 1.1 + index, "max": 1.3 + index,
                                       "mean": 1.2 + index, "median": 1.25 + index,
                                       "std": 0.1 + index},
                        },
                        "rank": 0,
                        "num_workers": 2 if collective == "send_recv" else 8,
                        "size": size_bytes,
                        "collective": collective,
                        "devices_per_node": 1 if collective == "send_recv" else 8,
                        "max_devices_per_node": 8,
                        "round_times": [1.1 + index, 1.3 + index],
                        "unexpected_debug_value": f"extra-{index}",
                        "_audit": {"mode": "decode", "sample": index},
                    })

                output = Path(tmp)
                PROFILE.write_results(output, collective, rows, {"fixture": True})
                result_path = output / f"{collective}.csv"
                with result_path.open(newline="") as stream:
                    reader = csv.reader(stream)
                    actual_header = next(reader)
                    values = list(csv.DictReader(stream, fieldnames=actual_header))

                self.assertEqual(actual_header, expected_header)
                self.assertEqual(actual_header[0], "")
                self.assertNotIn("round_times", actual_header)
                self.assertNotIn("unexpected_debug_value", actual_header)
                self.assertEqual([row[actual_header[0]] for row in values], ["0", "1", "2"])
                self.assertEqual([int(row["size"]) for row in values], [1024, 2048, 1024])
                self.assertEqual([float(row[f"time_stats.{operation}.mean"]) for row in values],
                                 [1.2, 2.2, 3.2])
                self.assertEqual([float(row[f"time_stats.{operation}.median"]) for row in values],
                                 [1.25, 2.25, 3.25])

                metadata = json.loads((output / f"{collective}.metadata.json").read_text())
                self.assertTrue(self._contains_key(metadata, "round_times"))

    @classmethod
    def _contains_key(cls, value, key):
        if isinstance(value, dict):
            return key in value or any(cls._contains_key(item, key) for item in value.values())
        if isinstance(value, list):
            return any(cls._contains_key(item, key) for item in value)
        return False


if __name__ == "__main__":
    unittest.main()
