from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from yolo_benchmark.environment import file_metadata


class EnvironmentTest(unittest.TestCase):
    def test_file_metadata_records_size_and_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.engine"
            path.write_bytes(b"t4-engine")
            metadata = file_metadata(path)

        self.assertEqual(metadata["size_bytes"], 9)
        self.assertEqual(metadata["sha256"], hashlib.sha256(b"t4-engine").hexdigest())
        self.assertTrue(str(metadata["path"]).endswith("model.engine"))
