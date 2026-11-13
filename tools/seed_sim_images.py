"""Render labeled sim scenes into images/ for tools/eval_verifier.py.

Layout (the same one real phone photos use):

    images/<stage_id>/{yes,no}/<sample>/{front,left_wrist,right_wrist}.jpg

`release_retract` exists in all three cases with different questions, so its
folders are named <task_id>.release_retract instead.
Sample names starting with hard_ are hard negatives.

    python tools/seed_sim_images.py [--out images] [--clean]
"""

from __future__ import annotations

import argparse
import base64
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.camera import render_views  # noqa: E402
from sim.scenes import Scene, all_scenes  # noqa: E402
from sim.world import STAGE_TRUTH  # noqa: E402


def folder_name(scene: Scene) -> str:
    return f"{scene.task_id}.{scene.stage}" if scene.stage == "release_retract" else scene.stage


def truth_conflicts(scenes: list[Scene]) -> list[str]:
    """Scenes whose label disagrees with the sim's own ground truth (a bug in scenes.py)."""
    bad = []
    for sc in scenes:
        truth = STAGE_TRUTH.get(sc.stage)
        if sc.stage == "release_retract" or truth is None:
            continue  # release questions also mention the seated part; checked by hand
        if truth(sc.world) != (sc.label == "yes"):
            bad.append(f"{sc.task_id}/{sc.stage}/{sc.name}: labeled {sc.label}, sim truth {truth(sc.world)}")
    return bad


def write_scene(root: Path, scene: Scene) -> None:
    folder = root / folder_name(scene) / scene.label / scene.name
    folder.mkdir(parents=True, exist_ok=True)
    for cam, b64 in render_views(scene.world).items():
        (folder / f"{cam}.jpg").write_bytes(base64.b64decode(b64))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="images")
    ap.add_argument("--clean", action="store_true", help="delete the output folder first")
    args = ap.parse_args()
    root = Path(args.out)
    if args.clean and root.exists():
        shutil.rmtree(root)
    scenes = all_scenes()
    bad = truth_conflicts(scenes)
    if bad:
        print("LABEL CONFLICTS:\n  " + "\n  ".join(bad))
        return 1
    for sc in scenes:
        write_scene(root, sc)
    counts = Counter((folder_name(s), s.label) for s in scenes)
    print(f"wrote {len(scenes)} scenes to {root}/")
    for (stage, label), n in sorted(counts.items()):
        print(f"  {stage:48s} {label:3s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
