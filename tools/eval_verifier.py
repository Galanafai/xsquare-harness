"""Measure how well the vision verifier answers the done-questions.

Runs every stage question from config/tasks/*.yaml against labeled images:

    images/<stage_id>/{yes,no}/<sample>/{front,left_wrist,right_wrist}.jpg

(if two cases ever share a stage id, name the folder <task_id>.<stage_id>). Sample names starting with "hard_"
are hard negatives. Make sim images with tools/seed_sim_images.py; real phone
photos use the same layout.

Two commands:

    python tools/eval_verifier.py run --label base
    python tools/eval_verifier.py run --label noreason --extra-body '{"reasoning_effort": "none"}'
    python tools/eval_verifier.py run --label front_left --cameras front,left_wrist
    python tools/eval_verifier.py report                  # compare all labels

Results are appended to runs/eval/<label>.jsonl, so a run can be stopped and
resumed (finished samples are skipped). Calls are throttled (default 5 per
minute, the Groq free tier). HTTP 429 is backed off and retried: it means
"slow down", not a bug. "yes" when the label is "no" is the FALSE-YES, the
dangerous error: it advances the supervisor past a stage that is not done.

Needs AGENT_CHAT_URL, AGENT_MODEL, AGENT_API_KEY in the environment. The key
is never printed or written.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import dataclasses
import glob
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xharness.config import load_config  # noqa: E402
from xharness.tasks import Task, load_task  # noqa: E402
from xharness.verifier import Verifier  # noqa: E402

CAMERAS = ("front", "left_wrist", "right_wrist")
OUT_DIR = Path("runs/eval")


@dataclasses.dataclass
class Sample:
    task: Task
    stage_id: str
    question: str
    label: str  # yes | no
    name: str
    folder: Path

    @property
    def qid(self) -> str:
        return f"{self.task.task_id}.{self.stage_id}"

    @property
    def hard(self) -> bool:
        return self.name.startswith("hard_")


# ------------------------------------------------------------------ loading


def stage_folder(root: Path, task: Task, stage_id: str, tasks: list[Task]) -> Path | None:
    """images/<task_id>.<stage_id> if present, else images/<stage_id> when the id is unique."""
    qualified = root / f"{task.task_id}.{stage_id}"
    if qualified.is_dir():
        return qualified
    shared = sum(1 for t in tasks for s in t.stages if s.id == stage_id) > 1
    bare = root / stage_id
    return bare if bare.is_dir() and not shared else None


def pick(names: list[str], count: int) -> list[str]:
    """First `count` names, but make sure hard negatives are included."""
    easy = [n for n in names if not n.startswith("hard_")]
    hard = [n for n in names if n.startswith("hard_")]
    chosen: list[str] = []
    while len(chosen) < count and (easy or hard):
        for pool in (easy, hard):  # alternate easy / hard
            if pool and len(chosen) < count:
                chosen.append(pool.pop(0))
    return chosen


def load_samples(root: Path, tasks: list[Task], per_class: int, stages: set[str]) -> list[Sample]:
    out: list[Sample] = []
    for task in tasks:
        for stage in task.stages:
            if stages and stage.id not in stages:
                continue
            folder = stage_folder(root, task, stage.id, tasks)
            if folder is None:
                print(f"skip {task.task_id}.{stage.id}: no image folder", file=sys.stderr)
                continue
            for label in ("yes", "no"):
                base = folder / label
                names = sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
                for name in pick(names, per_class):
                    out.append(Sample(task, stage.id, stage.done.question, label, name, base / name))
    return out


def request_for(sample: Sample) -> dict:
    images = {}
    for cam in CAMERAS:
        path = sample.folder / f"{cam}.jpg"
        images[cam] = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"observation": {"images": images}}


# ------------------------------------------------------------------ running


def done_keys(path: Path) -> set[tuple]:
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if not row.get("error", "").startswith("http_429"):
                keys.add((row["qid"], row["sample"], row["label"]))
    return keys


TOKENS_PER_IMAGE = 1700  # measured on Groq qwen: ~1.8k for one image plus text, whatever the pixel size
TOKENS_TEXT = 250


class Throttle:
    """Keep calls under a requests-per-minute AND an input-tokens-per-minute budget.

    Groq's free tier allows 7000 input tokens per minute and an image costs
    about 1.7k tokens no matter how small it is, so three cameras allow only
    about one call per minute, far below the 5 calls per minute we assumed.
    """

    def __init__(self, rpm: float, itpm: float, tokens_per_call: int):
        self.rpm, self.itpm, self.tokens = rpm, itpm, tokens_per_call
        self.sent: list[tuple[float, int]] = []

    async def wait(self) -> None:
        while True:
            now = time.monotonic()
            self.sent = [(t, n) for t, n in self.sent if now - t < 61.0]
            calls_ok = len(self.sent) < self.rpm
            tokens_ok = sum(n for _, n in self.sent) + self.tokens <= self.itpm
            if (calls_ok and tokens_ok) or not self.sent:
                self.sent.append((now, self.tokens))
                return
            await asyncio.sleep(1.0)


async def ask(verifier: Verifier, sample: Sample, retries: int = 4):
    """One verifier call. A 429 backs off (15, 30, 60, 60 s) and retries."""
    delay = 15.0
    for attempt in range(retries + 1):
        verdict = await verifier.check(sample.qid, sample.question, request_for(sample), sample.task.context)
        if not verdict.error.startswith("http_429") or attempt == retries:
            return verdict, attempt
        print(f"    429, backing off {delay:.0f}s", flush=True)
        await asyncio.sleep(delay)
        delay = min(delay * 2, 60.0)
    raise AssertionError("unreachable")


async def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    for cam in cameras:
        if cam not in CAMERAS:
            raise SystemExit(f"unknown camera {cam!r}")
    cfg.verifier = dataclasses.replace(
        cfg.verifier, cameras=cameras, extra_body=json.loads(args.extra_body), timeout_s=args.timeout
    )
    verifier = Verifier(cfg.verifier)
    if not verifier.available:
        raise SystemExit("verifier not configured: set AGENT_CHAT_URL and AGENT_MODEL")
    tasks = [load_task(p) for p in sorted(glob.glob("config/tasks/*.yaml"))]
    samples = load_samples(Path(args.images), tasks, args.per_class, set(args.stages))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.label}.jsonl"
    finished = done_keys(out_path)
    todo = [s for s in samples if (s.qid, s.name, s.label) not in finished]
    print(f"[{args.label}] model={cfg.verifier.model} cameras={cameras} extra_body={cfg.verifier.extra_body} "
          f"{len(todo)} calls to run ({len(samples) - len(todo)} already done)", flush=True)
    throttle = Throttle(args.rpm, args.itpm, tokens_per_call=TOKENS_PER_IMAGE * len(cameras) + TOKENS_TEXT)
    for i, sample in enumerate(todo, 1):
        await throttle.wait()
        verdict, retries = await ask(verifier, sample)
        row = {
            "label_run": args.label, "qid": sample.qid, "stage": sample.stage_id, "sample": sample.name,
            "label": sample.label, "hard": sample.hard, "answer": verdict.answer, "reason": verdict.reason,
            "latency_ms": verdict.latency_ms, "error": verdict.error, "thinking": verdict.thinking,
            "completion_tokens": verdict.completion_tokens, "retries_429": retries,
            "cameras": cameras, "extra_body": cfg.verifier.extra_body,
        }
        with out_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        ok = "ok " if verdict.answer == sample.label else ("FALSE-YES" if verdict.answer == "yes" else "miss")
        print(f"  {i}/{len(todo)} {sample.qid}/{sample.label}/{sample.name}: {verdict.answer} {ok} "
              f"{verdict.latency_ms:.0f}ms {verdict.error}", flush=True)
    return 0


# ------------------------------------------------------------------ report


def load_rows(labels: list[str]) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for path in sorted(OUT_DIR.glob("*.jsonl")):
        if labels and path.stem not in labels:
            continue
        runs[path.stem] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return runs


def summarize(rows: list[dict]) -> dict:
    """Metrics for any list of rows (one question or a whole run)."""
    n = len(rows)
    no_rows = [r for r in rows if r["label"] == "no"]
    lat = [r["latency_ms"] for r in rows if not r["error"]]
    toks = [r["completion_tokens"] for r in rows if r["completion_tokens"]]
    return {
        "n": n,
        "acc": sum(r["answer"] == r["label"] for r in rows) / n if n else 0.0,
        "false_yes": sum(r["answer"] == "yes" for r in no_rows),
        "n_no": len(no_rows),
        "unsure": sum(r["answer"] == "unsure" for r in rows) / n if n else 0.0,
        "errors": sum(bool(r["error"]) for r in rows),
        "med_ms": statistics.median(lat) if lat else 0.0,
        "max_ms": max(lat) if lat else 0.0,
        "think": sum(r["thinking"] for r in rows),
        "tokens": statistics.mean(toks) if toks else 0.0,
    }


def fmt(s: dict) -> str:
    return (f"{s['n']:3d}  {s['acc']*100:5.0f}%  {s['false_yes']:2d}/{s['n_no']:<2d} "
            f"{s['unsure']*100:5.0f}%  {s['errors']:3d}  {s['med_ms']:6.0f} {s['max_ms']:6.0f}  "
            f"{s['think']:3d} {s['tokens']:5.0f}")


HEADER = "  n    acc  FALSE-YES unsure  err  med_ms max_ms  thk  toks"


def report(args: argparse.Namespace) -> int:
    runs = load_rows(args.labels)
    if not runs:
        raise SystemExit(f"no results in {OUT_DIR}/")
    print("=== Overall per run (acc = answer equals label; FALSE-YES = said yes when label is no) ===")
    print(f"{'run':16s}{HEADER}")
    for label, rows in runs.items():
        print(f"{label:16s}{fmt(summarize(rows))}")
    print("\n=== Hard negatives only ===")
    print(f"{'run':16s}{HEADER}")
    for label, rows in runs.items():
        hard = [r for r in rows if r["hard"]]
        if hard:
            print(f"{label:16s}{fmt(summarize(hard))}")
    for label, rows in runs.items():
        print(f"\n=== Per question: {label} ===")
        print(f"{'question':48s}{HEADER}")
        for qid in sorted({r["qid"] for r in rows}):
            print(f"{qid:48s}{fmt(summarize([r for r in rows if r['qid'] == qid]))}")
        bad = [r for r in rows if r["label"] == "no" and r["answer"] == "yes"]
        for r in bad:
            print(f"  FALSE-YES {r['qid']}/{r['sample']}: {r['reason']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="ask the verifier about every labeled sample")
    r.add_argument("--label", required=True, help="name of this configuration, e.g. base, noreason")
    r.add_argument("--images", default="images")
    r.add_argument("--config", default=None, help="harness YAML (default: config/harness.local.yaml if present)")
    r.add_argument("--per-class", type=int, default=2, help="samples per yes and per no class, per question")
    r.add_argument("--cameras", default=",".join(CAMERAS))
    r.add_argument("--extra-body", default="{}", help="JSON merged into the request body")
    r.add_argument("--rpm", type=float, default=5.0, help="max calls per minute")
    r.add_argument("--itpm", type=float, default=6500, help="max estimated input tokens per minute (limit 7000)")
    r.add_argument("--timeout", type=float, default=20.0, help="per-call timeout (s); looser than production")
    r.add_argument("--stages", nargs="*", default=[], help="only these stage ids")
    p = sub.add_parser("report", help="compare finished runs")
    p.add_argument("labels", nargs="*")
    args = ap.parse_args()
    if args.cmd == "run":
        if args.config is None and Path("config/harness.local.yaml").exists():
            args.config = "config/harness.local.yaml"
        return asyncio.run(run(args))
    return report(args)


if __name__ == "__main__":
    raise SystemExit(main())
