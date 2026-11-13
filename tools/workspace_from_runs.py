"""Suggest per-arm workspace boxes for the action filter from recorded runs.

After a few direct or passthrough runs on the real robot, every request's
measured end-effector state is in runs/*/events.jsonl. This prints the
min/max of each arm's position, padded by a margin, as YAML you can paste
into config (filter.workspace).

    python tools/workspace_from_runs.py runs/ --margin 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_dir")
    ap.add_argument("--margin", type=float, default=0.05, help="metres added on every side")
    args = ap.parse_args()

    points: dict[str, list] = {"left_ee": [], "right_ee": []}
    for events in Path(args.runs_dir).glob("*/events.jsonl"):
        for line in events.read_text().splitlines():
            ev = json.loads(line)
            if ev.get("event") == "request":
                for arm in points:
                    points[arm].append(ev["state"][arm]["position"])

    print("filter:\n  workspace:")
    for arm, pts in points.items():
        if not pts:
            print(f"    # {arm}: no data")
            continue
        arr = np.asarray(pts)
        lo = (arr.min(axis=0) - args.margin).round(3).tolist()
        hi = (arr.max(axis=0) + args.margin).round(3).tolist()
        print(f"    {arm}: {{min: {lo}, max: {hi}}}   # from {len(arr)} samples")


if __name__ == "__main__":
    main()
