"""Fail CI when the latest real-PDF evaluation misses configured thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT
from src.evaluation.real_runner import check_thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=ROOT / "data/evaluation/results/real/latest.json")
    parser.add_argument("--thresholds", type=Path, default=ROOT / "data/evaluation/thresholds.json")
    args = parser.parse_args()
    try:
        result = json.loads(args.result.read_text(encoding="utf-8"))
        thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"error: could not read threshold input: {exc}\n")
    failures = check_thresholds(result, thresholds)
    if failures:
        print("Threshold check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Threshold check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())