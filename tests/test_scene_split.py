import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.scene_split import split_scenes


class SceneSplitTests(unittest.TestCase):
    def rows(self, count):
        return [{"scene_id": f"scene-{i}", "positive_blind_labels": i % 2} for i in range(count)]

    def test_all_scenes_are_assigned_once_even_for_tiny_datasets(self):
        for size in range(30):
            with self.subTest(size=size):
                result = split_scenes(self.rows(size))
                partitions = [result[key] for key in ("train", "val", "test")]
                assigned = [scene for part in partitions for scene in part]
                self.assertEqual(len(assigned), size)
                self.assertEqual(set(assigned), {row["scene_id"] for row in self.rows(size)})
                if size >= 3:
                    self.assertTrue(all(partitions))

    def test_same_seed_reproduces_split_without_mutating_input(self):
        rows = self.rows(30)
        original = json.dumps(rows)
        self.assertEqual(split_scenes(rows), split_scenes(rows))
        self.assertEqual(json.dumps(rows), original)
        self.assertNotEqual(split_scenes(rows, seed=7), split_scenes(rows, seed=8))

    def test_invalid_ratios_are_rejected(self):
        for ratios in [(-0.1, 0.5, 0.6), (float("nan"), 0.1, 0.9), (float("inf"), 0.1, 0.9), (0.7, 0.2, 0.2), (0, 0.5, 0.5)]:
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                split_scenes(self.rows(5), train_ratio=ratios[0], val_ratio=ratios[1], test_ratio=ratios[2])

    def test_duplicate_scene_is_rejected_to_prevent_leakage(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            split_scenes(self.rows(3) + self.rows(1))

    def test_invalid_label_counts_and_ids_are_rejected(self):
        for count in [-1, 0.5, True, "2", None]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                split_scenes([{"scene_id": "s", "positive_blind_labels": count}])
        for scene in [None, 12, "", " "]:
            with self.subTest(scene=scene), self.assertRaises(ValueError):
                split_scenes([{"scene_id": scene, "positive_blind_labels": 0}])

    def test_cli_error_does_not_create_output(self):
        script = Path(__file__).resolve().parents[1] / "scripts/create_scene_split.py"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "input.json").write_text(json.dumps(self.rows(3) + self.rows(1)))
            result = subprocess.run(
                [sys.executable, str(script), "--summary", str(root / "input.json"), "--output", str(root / "split.json")],
                cwd=root, capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate", result.stderr)
            self.assertFalse((root / "split.json").exists())
