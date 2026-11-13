"""Offline experiments against the hosted Wall-X endpoint (no robot involved).

    python tools/wallx_experiments.py all            # a, b and c, writes docs/WALLX_EXPERIMENTS.md
    python tools/wallx_experiments.py a|b|c          # one experiment, prints a markdown section

  a  determinism: the same observation 5 times
  b  phrasing: 3 phrasings of one instruction, 3 times each
  c  control_context sweep: omitted, all-null, then ONE slot changed at a time

Inputs are a rendered sim scene (case 1) plus the API Tutorial example poses, so
the actions are NOT a meaningful plan; the experiments measure how sensitive the
server is to noise, wording and control_context, not task skill. Calls are spaced
(default 2 s). Auth: WALLX_HEADERS_FILE or probe_out/wallx-headers.json (never
printed). Raw chunks are saved to runs/wallx_exp/ (git-ignored).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from websockets.sync.client import connect

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xharness.protocol import MAX_FRAME_BYTES, is_error_response, pack, unpack  # noqa: E402

ENDPOINT = "wss://maas-api.x2robot.com/v1/vla/wallx"
EMB = "quanta_x1_pro/gripper-g"
STATE = {
    "left_ee": {"position": [0.2905, 0.0224, 0.3754], "rotation_rpy": [0.2968, -0.3565, -0.1049], "gripper": 0.107},
    "right_ee": {"position": [0.2825, -0.0510, 0.3906], "rotation_rpy": [0.0270, -0.5094, -0.3716], "gripper": 0.119},
}
SLOTS = ("eff_delay_ms", "lpf_T_ms", "grip_delay_ms", "grip_gain", "grip_offset", "z_offset_mm", "rot_offset_deg")
PHRASINGS = [
    "Pick up the orange slider.",
    "Grasp the orange block and lift it.",
    "Take hold of the orange slider and raise it off the table.",
]
OUT = Path("runs/wallx_exp")


class Client:
    """One Wall-X connection, rebuilt after any failure. Spaces its calls."""

    def __init__(self, gap_s: float):
        path = os.path.expanduser(os.environ.get("WALLX_HEADERS_FILE", "probe_out/wallx-headers.json"))
        self.headers = json.loads(Path(path).read_text())
        self.gap_s, self.ws, self.last = gap_s, None, 0.0
        self.log: list[dict] = []

    def _open(self) -> None:
        self.ws = connect(ENDPOINT, additional_headers=self.headers, proxy=None, max_size=MAX_FRAME_BYTES,
                          compression=None, ping_interval=None, open_timeout=15)
        self.ws.recv(timeout=15)  # handshake

    def call(self, request: dict) -> tuple[dict, float]:
        wait = self.last + self.gap_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        for attempt in (0, 1):
            try:
                if self.ws is None:
                    self._open()
                started = time.monotonic()
                self.ws.send(pack(request))
                reply = unpack(self.ws.recv(timeout=30))
                self.last = time.monotonic()
                return reply, round((self.last - started) * 1000)
            except Exception:  # noqa: BLE001 - reconnect once, then give up
                self.ws = None
                if attempt:
                    raise
        raise AssertionError

    def close(self) -> None:
        if self.ws is not None:
            self.ws.close()


def base_request(instruction: str, control_context=None) -> dict:
    from sim.camera import render_views
    from sim.world import World

    req = {
        "embodiment": EMB,
        "normalization": "provided",
        "instruction": instruction,
        "observation": {"images": render_views(World(case=1)), "state": json.loads(json.dumps(STATE))},
    }
    if control_context is not None:
        req["control_context"] = control_context
    return req


def chunk(reply: dict) -> dict[str, np.ndarray]:
    """(32, 7) per arm: x y z (m), roll pitch yaw (rad), gripper."""
    out = {}
    for arm in ("left_ee", "right_ee"):
        a = reply["actions"][arm]
        out[arm] = np.hstack([np.asarray(a["position"]), np.asarray(a["rotation_rpy"]),
                              np.asarray(a["gripper"])[:, None]])
    return out


def dist_mm(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(mean over rows, max over rows) of the position distance in mm."""
    d = np.linalg.norm(a[:, :3] - b[:, :3], axis=1) * 1000
    return float(d.mean()), float(d.max())


def pair_stats(chunks_a: list, chunks_b: list | None = None) -> dict:
    """Pairwise position distances (mm) per arm within one list, or between two lists."""
    pairs = (list(itertools.combinations(chunks_a, 2)) if chunks_b is None
             else list(itertools.product(chunks_a, chunks_b)))
    stats = {}
    for arm in ("left_ee", "right_ee"):
        d = [dist_mm(x[arm], y[arm]) for x, y in pairs]
        stats[arm] = (float(np.mean([m for m, _ in d])), float(np.max([x for _, x in d])))
    return stats


def fmt_pair(stats: dict) -> str:
    return " | ".join(f"{stats[a][0]:.2f} | {stats[a][1]:.2f}" for a in ("left_ee", "right_ee"))


def checked(client: Client, request: dict, label: str, raw: list) -> dict | None:
    reply, ms = client.call(request)
    if is_error_response(reply):
        print(f"  {label}: ERROR {str(reply.get('error'))[:160]}", flush=True)
        raw.append({"label": label, "error": str(reply.get("error"))[:300], "ms": ms})
        return None
    c = chunk(reply)
    raw.append({"label": label, "ms": ms, "left": c["left_ee"].tolist(), "right": c["right_ee"].tolist(),
                "ctx": reply.get("control_context")})
    print(f"  {label}: {ms} ms", flush=True)
    return c


# ------------------------------------------------------------------ experiments


def exp_a(client: Client, raw: list) -> tuple[str, list]:
    chunks = [checked(client, base_request("Pick up the orange slider."), f"a{i}", raw) for i in range(5)]
    chunks = [c for c in chunks if c]
    s = pair_stats(chunks)
    g = max(float(np.ptp(np.stack([c[a][:, 6] for c in chunks]), axis=0).max()) for a in ("left_ee", "right_ee"))
    rot = max(float(np.degrees(np.ptp(np.stack([c[a][:, 3:6] for c in chunks]), axis=0)).max())
              for a in ("left_ee", "right_ee"))
    text = ("## a. Determinism: same observation, 5 calls\n\n"
            "Pairwise distance between chunks, position only, in mm (mean over the 32 rows, and the worst row).\n\n"
            "| pairs | left mean | left max | right mean | right max |\n|---|---|---|---|---|\n"
            f"| 10 pairs | {fmt_pair(s)} |\n\n"
            f"Largest per-row range over the 5 chunks: rotation {rot:.3f} deg, gripper {g:.5f}.\n")
    return text, chunks


def exp_b(client: Client, raw: list, noise: list) -> str:
    groups = []
    for p, phrasing in enumerate(PHRASINGS):
        cs = [checked(client, base_request(phrasing), f"b{p}_{i}", raw) for i in range(3)]
        groups.append([c for c in cs if c])
    within_all = {arm: (float(np.mean([pair_stats(g)[arm][0] for g in groups])),
                        float(np.max([pair_stats(g)[arm][1] for g in groups]))) for arm in ("left_ee", "right_ee")}
    rows = []
    for (i, gi), (j, gj) in itertools.combinations(enumerate(groups), 2):
        rows.append(f"| {i} vs {j} | {fmt_pair(pair_stats(gi, gj))} |")
    noise_s = pair_stats(noise)
    return ("## b. Phrasing: 3 phrasings x 3 calls\n\n"
            + "".join(f"- phrasing {i}: \"{p}\"\n" for i, p in enumerate(PHRASINGS))
            + "\nPosition distance in mm (mean over rows | worst row).\n\n"
            "| comparison | left mean | left max | right mean | right max |\n|---|---|---|---|---|\n"
            f"| noise from (a), same wording | {fmt_pair(noise_s)} |\n"
            f"| within one phrasing, this run | {fmt_pair(within_all)} |\n" + "\n".join(rows) + "\n")


def context_variants() -> list[tuple[str, dict | None, int]]:
    """(name, control_context, repeats). Slot values are NORMALIZED numbers in [-1, 1].

    The first sweep sent physical units (+-5 mm, 50 ms, ...) and the server answered
    "control_context.left must be null or a number in [-1, 1]"; it also said only
    eff_delay_ms, grip_delay_ms and z_offset_mm are trained parameters. So this sweep
    uses [-1, 1] and only those three slots.
    """

    def one(slot: str, value, arm: str = "left") -> dict:
        ctx = {"left": dict.fromkeys(SLOTS), "right": dict.fromkeys(SLOTS)}
        ctx[arm][slot] = value
        return ctx

    variants: list[tuple[str, dict | None, int]] = [
        ("omitted", None, 4),
        ("all_null", {"left": dict.fromkeys(SLOTS), "right": dict.fromkeys(SLOTS)}, 6)]
    for slot, values in [("z_offset_mm", (-1.0, -0.5, 0.5, 1.0)), ("eff_delay_ms", (-1.0, 1.0)),
                         ("grip_delay_ms", (-1.0, 1.0))]:
        for v in values:
            variants.append((f"{slot}={v:g} (left only)", one(slot, v), 4))
    return variants


def exp_c(client: Client, raw: list) -> str:
    results: dict[str, list] = {}
    for name, ctx, reps in context_variants():
        cs = [checked(client, base_request("Pick up the orange slider.", ctx), f"c_{name}_{i}", raw) for i in range(reps)]
        results[name] = [c for c in cs if c]
    ref = results["all_null"]

    def mean_z(c, arm):
        return float(c[arm][:, 2].mean() * 1000)

    sd = {arm: float(np.std([mean_z(c, arm) for c in ref], ddof=1)) for arm in ("left_ee", "right_ee")}
    lines = ["| variant | n | left dz mean +- s.e. (mm) | right dz mean +- s.e. (mm, control) | left grip delta | left dx, dy (mm) |",
             "|---|---|---|---|---|---|"]
    for name, cs in results.items():
        if not cs:
            lines.append(f"| {name} | 0 | rejected | | | |")
            continue
        cell = []
        for arm in ("left_ee", "right_ee"):
            ref_z = float(np.mean([mean_z(c, arm) for c in ref]))
            zs = [mean_z(c, arm) - ref_z for c in cs]
            se = float(np.std(zs, ddof=1) / np.sqrt(len(zs))) if len(zs) > 1 else float("nan")
            cell.append(f"{np.mean(zs):+.1f} +- {se:.1f}")
        lm, rm = np.mean([c["left_ee"] for c in ref], axis=0), np.mean([c["left_ee"] for c in cs], axis=0)
        dxy = (rm[:, :2] - lm[:, :2]).mean(axis=0) * 1000
        gr = float(np.mean([c["left_ee"][:, 6].mean() for c in cs]) - np.mean([c["left_ee"][:, 6].mean() for c in ref]))
        lines.append(f"| {name} | {len(cs)} | {cell[0]} | {cell[1]} | {gr:+.4f} | {dxy[0]:+.1f}, {dxy[1]:+.1f} |")
    return ("## c. control_context sweep (normalized slot values)\n\n"
            "Baseline = field omitted (server startup context) and all-null (0 slots set). Every other row sets ONE\n"
            "slot on the LEFT arm only; the right arm is the control and should not move. dz is the mean over the\n"
            "32 rows of the z difference to the mean of the all-null calls.\n\n"
            f"Noise floor, spread of the per-chunk mean z across {len(ref)} all-null calls: "
            f"left {sd['left_ee']:.1f} mm, right {sd['right_ee']:.1f} mm (1 sd).\n\n" + "\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["a", "b", "c", "all"])
    ap.add_argument("--gap", type=float, default=2.0, help="seconds between calls")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    client = Client(args.gap)
    raw: list[dict] = []
    sections: list[str] = []
    try:
        noise: list = []
        if args.which in ("a", "b", "all"):
            text, noise = exp_a(client, raw)
            if args.which in ("a", "all"):
                sections.append(text)
        if args.which in ("b", "all"):
            sections.append(exp_b(client, raw, noise))
        if args.which in ("c", "all"):
            sections.append(exp_c(client, raw))
    finally:
        client.close()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"raw_{args.which}_{int(time.time())}.json").write_text(json.dumps(raw))
    body = "\n".join(sections)
    if args.out:
        Path(args.out).write_text(body)
    print("\n" + body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
