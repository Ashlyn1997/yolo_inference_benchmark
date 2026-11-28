from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.plot_t4_output_buffer_results import EVENT_INTERVALS, WALL_FIELDS, verify_event_csv


class OutputBufferArchiveTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "cuda_event_samples.csv"
        self.fields = [*WALL_FIELDS, *EVENT_INTERVALS]
        self.rows = [{"iteration": i, **{key: i + 1.0 for key in self.fields}} for i in range(2)]
        self.means = {key: 1.5 for key in self.fields}

    def verify(self):
        with self.path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        return verify_event_csv(self.path, 2, self.means)

    def test_event_and_host_means_are_recomputed(self):
        self.assertEqual(self.verify(), 2)

    def test_modified_event_timing_is_rejected(self):
        self.rows[1]["d2h_event_ms"] = 20
        with self.assertRaisesRegex(ValueError, "d2h_event_ms"):
            self.verify()

    def test_modified_host_timing_is_rejected(self):
        self.rows[1]["host_output_call_wall_ms"] = 20
        with self.assertRaisesRegex(ValueError, "host_output_call_wall_ms"):
            self.verify()

    def test_missing_rows_or_duplicate_iterations_are_rejected(self):
        self.rows[1]["iteration"] = 0
        with self.assertRaisesRegex(ValueError, "unordered"):
            self.verify()
        self.rows.pop()
        with self.assertRaisesRegex(ValueError, "row count"):
            self.verify()

    def test_missing_schema_fields_are_rejected(self):
        del self.means["d2h_event_ms"]
        with self.assertRaisesRegex(ValueError, "summary fields"):
            self.verify()

    def test_negative_or_nonfinite_event_values_are_rejected(self):
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.rows[1]["execution_event_ms"] = value
                with self.assertRaisesRegex(ValueError, "invalid Event timing"):
                    self.verify()


if __name__ == "__main__":
    unittest.main()
