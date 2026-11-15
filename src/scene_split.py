"""Deterministic scene-level splitting, independent of training dependencies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import random


def ensure_minimum_nonempty_splits(split: dict[str, list[str]]) -> None:
    """Populate all partitions when at least three scenes are available."""
    def move_one(source_keys: list[str], target_key: str) -> None:
        for source_key in source_keys:
            if len(split[source_key]) > 1:
                split[target_key].append(split[source_key].pop())
                return

    if not split["train"]:
        move_one(["val", "test"], "train")
    if not split["val"]:
        move_one(["test", "train"], "val")
    if not split["test"]:
        move_one(["val", "train"], "test")


def split_scenes(
    summary: Sequence[Mapping[str, object]],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 7,
) -> dict[str, object]:
    """Split unique scenes without leakage, preserving the original allocation.

    Ratios are positive, finite and sum to one. Tiny datasets may have empty
    partitions; no scene is duplicated to manufacture an evaluation set.
    """
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(not math.isfinite(value) or value <= 0 for value in ratios):
        raise ValueError("train/val/test ratios must be finite and positive")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")
    positive: list[str] = []
    negative: list[str] = []
    seen: set[str] = set()
    for row in summary:
        scene = row.get("scene_id")
        count = row.get("positive_blind_labels")
        if not isinstance(scene, str) or not scene.strip():
            raise ValueError("scene_id must be a non-empty string")
        if scene in seen:
            raise ValueError(f"duplicate scene_id: {scene}")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"positive_blind_labels must be a non-negative integer: {scene}")
        seen.add(scene)
        (positive if count > 0 else negative).append(scene)

    rng = random.Random(seed)
    rng.shuffle(positive)
    rng.shuffle(negative)
    split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for scenes in (positive, negative):
        train_count = max(1, round(len(scenes) * train_ratio)) if len(scenes) >= 3 else max(0, len(scenes) - 2)
        remaining = len(scenes) - train_count
        val_count = max(1, round(len(scenes) * val_ratio)) if remaining >= 2 else max(0, remaining - 1)
        split["train"].extend(scenes[:train_count])
        split["val"].extend(scenes[train_count:train_count + val_count])
        split["test"].extend(scenes[train_count + val_count:])
    ensure_minimum_nonempty_splits(split)
    return {
        **{key: sorted(value) for key, value in split.items()},
        "metadata": {
            "seed": seed, "train_ratio": train_ratio, "val_ratio": val_ratio,
            "test_ratio": test_ratio, "positive_scene_count": len(positive),
            "negative_scene_count": len(negative), "split_unit": "scene",
        },
    }
