from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

from scripts.profile_t4_preprocess import ProfiledPreprocess, STAGES, summarize_samples


class PreprocessProfilingTest(unittest.TestCase):
    def test_initial_call_is_excluded_and_outer_samples_are_aligned(self):
        samples = [
            {stage: value for stage in STAGES}
            for value in (1000.0, 2.0, 4.0)
        ]
        means, rows = summarize_samples(
            samples,
            [SimpleNamespace(preprocess_ms=3.0), SimpleNamespace(preprocess_ms=7.0)],
            2,
        )
        self.assertEqual(means["body_total_ms"], 3.0)
        self.assertEqual(means["outside_body_ms"], 2.0)
        self.assertEqual(rows[0]["iteration"], 0)
        self.assertEqual(rows[1]["outside_body_ms"], 3.0)

    def test_missing_or_extra_samples_are_rejected(self):
        for samples, raw in (([], []), ([{}], []), ([{}, {}], [])):
            with self.assertRaisesRegex(ValueError, "exactly runs"):
                summarize_samples(samples, raw, 1)

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) is not None for name in ("numpy", "cv2", "torch")),
        "CPU pipeline dependencies not installed",
    )
    def test_instrumentation_preserves_pixels_and_letterbox_metadata(self):
        import numpy as np
        from yolo_benchmark.pipeline import preprocess

        rng = np.random.default_rng(123)
        images = [
            rng.integers(0, 256, shape, dtype=np.uint8)
            for shape in ((37, 91, 3), (84, 54, 3), (64, 64, 3))
        ]
        expected, expected_metas = preprocess(images, (64, 64))
        profiler = ProfiledPreprocess()
        actual, metas = profiler(images, (64, 64))
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(metas, expected_metas)
        self.assertEqual(len(profiler.samples), 1)
