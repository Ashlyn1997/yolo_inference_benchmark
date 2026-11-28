from __future__ import annotations

import importlib.util
import csv
import io
import json
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.check_t4_output_correctness import OwnedOutput, check_sequence, image_groups, select_images


class OutputCorrectnessUtilityTest(unittest.TestCase):
    def test_owned_wrapper_copies_mutable_borrowed_storage(self):
        buffer = [0, 0]

        def infer(values):
            buffer[:] = values
            return buffer

        owned = OwnedOutput(infer)
        first = owned([1, 2])
        second = owned([3, 4])
        self.assertEqual(first, [1, 2])
        self.assertEqual(second, [3, 4])
        self.assertIsNot(first, buffer)
        self.assertIsNot(first, second)

    def test_final_batch_is_padded_without_reordering_real_images(self):
        groups = list(image_groups(["a", "b", "c", "d", "e"], 4))
        self.assertEqual(groups, [(["a", "b", "c", "d"], 4), (["e", "e", "e", "e"], 1)])
        self.assertEqual(list(image_groups([], 4)), [])
        with self.assertRaisesRegex(ValueError, "positive"):
            list(image_groups(["a"], 0))

    def test_image_selection_is_sorted_and_never_cycles(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for name in ("z.jpg", "a.png", "b.JPG", "ignored.txt"):
                (source / name).write_bytes(b"fixture")
            self.assertEqual([path.name for path in select_images(source, 2)], ["a.png", "b.JPG"])
            with self.assertRaisesRegex(ValueError, "Requested 4.*found 3"):
                select_images(source, 4)
            with self.assertRaisesRegex(ValueError, "directory"):
                select_images(source / "z.jpg", 2)


@unittest.skipUnless(importlib.util.find_spec("numpy") is not None, "NumPy is not installed")
class OutputCorrectnessSequenceTest(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np

    def batches(self):
        result = []
        for ids, count in image_groups(list(range(5)), 2):
            prediction = self.np.asarray([[[i + 1, i + 2, i + 3, i + 4, .9, 0]] for i in ids], dtype="float32")
            identities = [{"image_id": f"{i}.jpg", "decoded_sha256": f"hash-{i}"} for i in ids[:count]]
            result.append((prediction, [None] * 2, identities))
        return result

    def borrower(self):
        buffer = None

        def infer(batch):
            nonlocal buffer
            if buffer is None:
                buffer = self.np.empty_like(batch)
            self.np.copyto(buffer, batch)
            return buffer
        return infer

    @staticmethod
    def decode(prediction, metas):
        return [[SimpleNamespace(box=sample[0, :4], score=sample[0, 4], class_id=int(sample[0, 5]))]
                for sample in prediction]

    def test_continuous_images_output_lifetimes_and_padding(self):
        saved = []
        summary, trace, images = check_sequence(
            lambda batch: batch.copy(), self.borrower(), self.batches(), self.decode,
            on_batch=lambda row, image_rows: saved.append((row, image_rows)),
        )
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["images_checked"], 5)
        self.assertEqual(summary["batches_checked"], 3)
        self.assertEqual(summary["padded_images"], 1)
        self.assertTrue(summary["borrowed_contents_changed_observed"])
        self.assertTrue(summary["first_owned_and_reference_survived_later_calls"])
        self.assertEqual(summary["owned_max_abs_error"], 0)
        self.assertEqual([row["image_id"] for row in images], [f"{i}.jpg" for i in range(5)])
        self.assertEqual(images[-1]["batch_slot"], 0)
        self.assertEqual(images[-1]["owned"][0]["box"], [5, 6, 7, 8])
        self.assertEqual(len(saved), 3)
        self.assertEqual(trace[-1]["real_images"], 1)

    def test_identical_predictions_do_not_falsely_fail_the_lifetime_check(self):
        batches = self.batches()
        for batch, _, _ in batches:
            batch[:] = batches[0][0][0]
        summary, _, _ = check_sequence(lambda batch: batch.copy(), self.borrower(), batches, self.decode)
        self.assertTrue(summary["passed"])
        self.assertFalse(summary["borrowed_contents_changed_observed"])

    def test_swapped_batch_slots_are_detected(self):
        infer = self.borrower()
        with self.assertRaisesRegex(AssertionError, "Batch 0 correctness"):
            check_sequence(lambda batch: batch.copy(), lambda batch: infer(batch[::-1]), self.batches(), self.decode)

    def test_owned_output_that_still_aliases_is_rejected(self):
        with patch("scripts.check_t4_output_correctness.OwnedOutput", side_effect=lambda infer: infer):
            with self.assertRaisesRegex(AssertionError, "Owned output aliases"):
                check_sequence(lambda batch: batch.copy(), self.borrower(), self.batches(), self.decode)

    def test_borrowed_path_that_does_not_reuse_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "did not reuse"):
            check_sequence(lambda batch: batch.copy(), lambda batch: batch.copy(), self.batches(), self.decode)

    def test_reference_output_that_is_not_independent_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "Independent outputs alias"):
            check_sequence(self.borrower(), self.borrower(), self.batches(), self.decode)

    def test_duplicate_image_ids_are_rejected(self):
        batches = self.batches()
        batches[1][2][0]["image_id"] = "0.jpg"
        with self.assertRaisesRegex(AssertionError, "Duplicate image ID"):
            check_sequence(lambda batch: batch.copy(), self.borrower(), batches, self.decode)

    def test_nonfinite_predictions_are_rejected(self):
        batches = self.batches()
        batches[0][0][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(AssertionError, "nonfinite"):
            check_sequence(lambda batch: batch.copy(), self.borrower(), batches, self.decode)

    def test_postprocessed_detection_mismatch_is_rejected(self):
        calls = 0

        def decode(prediction, metas):
            nonlocal calls
            calls += 1
            output = self.decode(prediction, metas)
            if calls == 2:
                output[0] = []
            return output

        with self.assertRaisesRegex(AssertionError, "Detection count differs"):
            check_sequence(lambda batch: batch.copy(), self.borrower(), self.batches(), decode)

    def test_repeated_image_contents_cannot_claim_diverse_input_coverage(self):
        batches = self.batches()
        for _, _, images in batches:
            for image in images:
                image["decoded_sha256"] = "same"
        with self.assertRaisesRegex(AssertionError, "two different decoded"):
            check_sequence(lambda batch: batch.copy(), self.borrower(), batches, self.decode)

    def run_mock_cli(self, root, *, fail=False):
        from scripts.check_t4_output_correctness import main
        source = root / "images"
        source.mkdir()
        for index in range(3):
            (source / f"{index}.jpg").write_bytes(bytes([index]))
        engine = root / "model.engine"
        engine.write_bytes(b"fake engine")
        output = root / "output"
        cv2 = types.ModuleType("cv2")
        cv2.__version__ = "mock"
        cv2.getNumThreads = lambda: 1
        cv2.imread = lambda path: self.np.full((2, 2, 3), int(Path(path).stem), dtype="uint8")
        torch = types.ModuleType("torch")
        torch.device = lambda value: SimpleNamespace(type="cuda", index=0)
        torch.cuda = SimpleNamespace(is_available=lambda: True, set_device=lambda device: None)
        torch.get_num_threads = lambda: 2
        backends = types.ModuleType("yolo_benchmark.backends")
        backends.TensorRTBackend = lambda *args, **kwargs: SimpleNamespace(infer=lambda batch: batch.copy())
        pipeline = types.ModuleType("yolo_benchmark.pipeline")
        pipeline.preprocess = lambda images, size: (
            self.np.asarray([[[int(image[0, 0, 0]) + 1, 2, 3, 4, .9, 0]] for image in images], dtype="float32"),
            [None] * len(images),
        )
        pipeline.postprocess = lambda prediction, metas, **kwargs: self.decode(prediction, metas)
        argv = ["checker", "--source", str(source), "--engine", str(engine), "--batch", "2",
                "--images", "3", "--output-dir", str(output)]
        infer = (lambda batch: self.np.zeros_like(batch)) if fail else self.borrower()
        with patch("sys.argv", argv), patch.dict("sys.modules", {
            "cv2": cv2, "torch": torch, "yolo_benchmark.backends": backends, "yolo_benchmark.pipeline": pipeline,
        }), patch("yolo_benchmark.environment.collect_environment", return_value={"gpu": "Tesla T4"}), patch(
            "scripts.check_t4_output_correctness.OutputBufferInfer", return_value=infer
        ), redirect_stdout(io.StringIO()):
            if fail:
                with self.assertRaisesRegex(SystemExit, "CORRECTNESS FAILED"):
                    main()
            else:
                main()
        return output

    def test_cli_saves_per_image_results_and_passed_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.run_mock_cli(Path(directory))
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["images_checked"], 3)
            self.assertEqual(summary["padded_images"], 1)
            predictions = [json.loads(line) for line in (output / "image_results.jsonl").read_text().splitlines()]
            self.assertEqual([row["image_id"] for row in predictions], ["0.jpg", "1.jpg", "2.jpg"])
            self.assertTrue((output / "provenance.json").is_file())
            with (output / "batch_checks.csv").open() as file:
                self.assertEqual(len(list(csv.DictReader(file))), 2)

    def test_cli_failure_preserves_error_and_does_not_claim_success(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.run_mock_cli(Path(directory), fail=True)
            summary = json.loads((output / "summary.json").read_text())
            self.assertFalse(summary["passed"])
            self.assertEqual(summary["images_completed"], 0)
            self.assertIn("correctness check failed", summary["error"])
            self.assertTrue((output / "failure.txt").is_file())


if __name__ == "__main__":
    unittest.main()
