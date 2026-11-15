#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scene_split import split_scenes


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a stratified scene-level train/val/test split.")
    parser.add_argument("--summary", required=True, help="preprocess_summary.json from graph preprocessing")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    try:
        summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
        if not isinstance(summary, list) or not all(isinstance(row, dict) for row in summary):
            raise ValueError("summary must be an array of scene records")
        split = split_scenes(
            summary, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
            test_ratio=args.test_ratio, seed=args.seed,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(
        f"[OK] {output}: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}, "
        f"positive_scenes={split['metadata']['positive_scene_count']}, negative_scenes={split['metadata']['negative_scene_count']}"
    )


if __name__ == "__main__":
    main()
