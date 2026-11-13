"""Probe the real Wall-X endpoint and write a report of how it actually behaves.

Run this from YOUR laptop before the event (organizers said pre-event cloud
access is allowed). It answers the questions the docs leave open:

  1. Which auth header format works?          (docs never name the header)
  2. What does the handshake contain?
  3. How long is the first (warm-up) call, and normal calls?
  4. Exact response layout: keys, dtypes, shapes, extra fields.
     The raw reply is saved so our HOLD chunks can copy its exact format.
  5. How each documented error looks, and whether the connection survives.
  6. Does the instruction text change the output at all?

Usage
-----
    export WALLX_API_KEY='sk-...'          # from the API Tutorial; never commit it
    python tools/probe_wallx.py                       # synthetic camera images
    python tools/probe_wallx.py --images photos/      # front.jpg left_wrist.jpg right_wrist.jpg

Or with a headers file you already have:
    python tools/probe_wallx.py --headers-file ~/.config/xharness/wallx-headers.json

Output: probe_out/report.md, probe_out/response_first.msgpack, and (if auth
discovery found a format) probe_out/wallx-headers.json with chmod 600.
probe_out/ is git-ignored. Synthetic images make the ACTIONS meaningless;
this tool tests the protocol, not task skill.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import stat
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.sync.client import connect

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xharness.protocol import CAMERAS, MAX_FRAME_BYTES, pack, unpack  # noqa: E402

DEFAULT_ENDPOINT = "wss://maas-api.x2robot.com/v1/vla/wallx"
EMB = "quanta_x1_pro/gripper-g"
# State from the API Tutorial example (a plausible desktop pose).
EXAMPLE_STATE = {
    "left_ee": {"position": [0.2905, 0.0224, 0.3754], "rotation_rpy": [0.2968, -0.3565, -0.1049], "gripper": 0.107},
    "right_ee": {"position": [0.2825, -0.0510, 0.3906], "rotation_rpy": [0.0270, -0.5094, -0.3716], "gripper": 0.119},
}


def auth_variants(key: str) -> list[tuple[str, dict, str]]:
    """(label, headers, url_suffix) to try, most likely first. Values never printed."""
    return [
        ("Authorization: Bearer <key>", {"Authorization": f"Bearer {key}"}, ""),
        ("Authorization: <key>", {"Authorization": key}, ""),
        ("X-API-Key: <key>", {"X-API-Key": key}, ""),
        ("api-key: <key>", {"api-key": key}, ""),
        ("query ?api_key=<key>", {}, f"?api_key={key}"),
        ("query ?token=<key>", {}, f"?token={key}"),
    ]


def open_ws(url: str, headers: dict):
    return connect(url, additional_headers=headers, open_timeout=20, max_size=MAX_FRAME_BYTES,
                   compression=None, proxy=None)


def load_images(folder: str | None) -> dict[str, str]:
    if folder:
        return {cam: base64.b64encode((Path(folder) / f"{cam}.jpg").read_bytes()).decode() for cam in CAMERAS}
    from sim.camera import render_views
    from sim.world import World

    return render_views(World(case=3))


def request(images, instruction, **overrides) -> dict:
    req = {
        "embodiment": EMB,
        "normalization": "provided",
        "instruction": instruction,
        "observation": {"images": dict(images), "state": json.loads(json.dumps(EXAMPLE_STATE))},
    }
    req.update(overrides)
    return req


def describe(value, depth=0) -> str:
    """Compact type/shape/dtype description of a decoded msgpack value."""
    pad = "  " * depth
    if isinstance(value, dict):
        return "\n".join(f"{pad}{k}: " + (("\n" + describe(v, depth + 1)) if isinstance(v, dict) else describe(v))
                         for k, v in value.items())
    if isinstance(value, np.ndarray):
        return f"ndarray shape={value.shape} dtype={value.dtype}"
    if isinstance(value, list):
        arr = np.asarray(value)
        return f"list len={len(value)} (as array shape={arr.shape}, dtype={arr.dtype})"
    return f"{type(value).__name__} = {str(value)[:80]!r}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default=os.environ.get("WALLX_ENDPOINT", DEFAULT_ENDPOINT))
    ap.add_argument("--headers-file", default=os.environ.get("WALLX_HEADERS_FILE", ""))
    ap.add_argument("--key-env", default="WALLX_API_KEY")
    ap.add_argument("--images", help="folder with front.jpg, left_wrist.jpg, right_wrist.jpg")
    ap.add_argument("--requests", type=int, default=5, help="steady-state requests after warm-up")
    ap.add_argument("--out", default="probe_out")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    lines: list[str] = [f"# Wall-X probe report ({time.strftime('%Y-%m-%d %H:%M:%S')})", f"endpoint: {args.endpoint}", ""]

    def say(text=""):
        print(text, flush=True)
        lines.append(text)

    # ---------------------------------------------------------------- 1. auth
    say("## 1. Authentication")
    headers, url = {}, args.endpoint
    if args.headers_file:
        headers = json.loads(Path(os.path.expanduser(args.headers_file)).read_text())
        say(f"using headers file with header names: {sorted(headers)}")
    else:
        key = os.environ.get(args.key_env, "")
        if not key:
            say(f"No headers file and ${args.key_env} is empty. Set one of them and re-run.")
            return 2
        found = None
        for label, hdrs, suffix in auth_variants(key):
            try:
                with open_ws(args.endpoint + suffix, hdrs) as ws:
                    unpack(ws.recv(timeout=20))
                say(f"- {label}: ACCEPTED")
                found = (hdrs, suffix)
                break
            except InvalidStatus as err:
                say(f"- {label}: rejected (HTTP {err.response.status_code})")
            except Exception as err:  # noqa: BLE001
                detail = str(err).replace(key, "<key>")[:100]  # never print the key
                say(f"- {label}: failed ({type(err).__name__}: {detail})")
        if not found:
            say("\nNo auth format worked. Ask organizers for the exact header name. Stopping.")
            (out / "report.md").write_text("\n".join(lines))
            return 1
        headers, suffix = found
        url = args.endpoint + suffix
        if headers:
            path = out / "wallx-headers.json"
            path.write_text(json.dumps(headers))
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            say(f"saved working headers to {path} (chmod 600). Use it as WALLX_HEADERS_FILE.")

    images = load_images(args.images)
    say(f"images: {'from ' + args.images if args.images else 'SYNTHETIC (actions will be meaningless)'}")

    with open_ws(url, headers) as ws:
        # ------------------------------------------------------------ 2. handshake
        say("\n## 2. Handshake")
        t0 = time.monotonic()
        meta = unpack(ws.recv(timeout=30))
        say(f"received in {1000 * (time.monotonic() - t0):.0f} ms")
        say(describe(meta))

        # ------------------------------------------------------------ 3. latency + 4. layout
        say("\n## 3. Latency")
        timings, replies = [], []
        for i in range(1 + args.requests):
            t0 = time.monotonic()
            ws.send(pack(request(images, "Pick up the teal button.")))
            raw = ws.recv(timeout=120 if i == 0 else 60)
            timings.append(1000 * (time.monotonic() - t0))
            replies.append(raw)
        say(f"first call (warm-up): {timings[0]:.0f} ms")
        if len(timings) > 1:
            steady = timings[1:]
            say(f"steady calls: median {statistics.median(steady):.0f} ms, max {max(steady):.0f} ms (n={len(steady)})")

        (out / "response_first.msgpack").write_bytes(replies[0] if isinstance(replies[0], bytes) else b"")
        first = unpack(replies[-1])
        say("\n## 4. Response layout (last steady reply)")
        say(describe(first))
        if isinstance(first, dict) and "actions" in first:
            for arm in ("left_ee", "right_ee"):
                pos = np.asarray(first["actions"][arm]["position"], float)
                grip = np.asarray(first["actions"][arm]["gripper"], float)
                cur = np.asarray(EXAMPLE_STATE[arm]["position"])
                say(f"- {arm}: rows={len(pos)}, row31==row30: {np.allclose(pos[-1], pos[-2])}, "
                    f"row0 - state = {np.round(pos[0] - cur, 4).tolist()} m, "
                    f"max row step = {np.linalg.norm(np.diff(pos, axis=0), axis=1).max():.4f} m, "
                    f"gripper {grip.min():.3f}..{grip.max():.3f}")
            extra = sorted(set(first) - {"embodiment", "actions"})
            say(f"- extra top-level fields: {extra or 'none'}")

        # ------------------------------------------------------------ 5. errors
        say("\n## 5. Error behavior (does the connection survive?)")
        probes = {
            "capital G embodiment": request(images, "x", embodiment="quanta_x1_pro/gripper-G"),
            "missing front image": _without_front(request(images, "Pick up the teal button.")),
            "wrong normalization": request(images, "Pick up the teal button.", normalization="dataset"),
            "extra model_version field": request(images, "Pick up the teal button.", model_version="wall-oss-1.0"),
        }
        for label, req in probes.items():
            try:
                ws.send(pack(req))
                reply = unpack(ws.recv(timeout=60))
                kind = f"error {reply['error']}" if isinstance(reply, dict) and "error" in reply else "ACCEPTED (actions)"
                ws.send(pack(request(images, "Pick up the teal button.")))
                alive = "actions" in unpack(ws.recv(timeout=60))
                say(f"- {label}: {str(kind)[:160]} | connection still usable: {alive}")
            except ConnectionClosed as err:
                say(f"- {label}: CONNECTION CLOSED (code {err.rcvd.code if err.rcvd else '?'})")
                break

    # ---------------------------------------------------------------- 6. instruction sensitivity
    say("\n## 6. Does the instruction change the output? (same images and state)")
    with open_ws(url, headers) as ws:
        unpack(ws.recv(timeout=30))
        ends = {}
        for text in ("Pick up the teal button.", "Pick up the orange slider.", "Move both arms up."):
            ws.send(pack(request(images, text)))
            reply = unpack(ws.recv(timeout=120))
            if "actions" in reply:
                ends[text] = np.asarray(reply["actions"]["right_ee"]["position"], float)[30]
                say(f"- {text!r}: right arm row 30 = {np.round(ends[text], 4).tolist()}")
        if len(ends) == 3:
            spread = max(np.linalg.norm(a - b) for a in ends.values() for b in ends.values())
            say(f"max difference between instructions: {spread * 1000:.1f} mm")

    (out / "report.md").write_text("\n".join(lines))
    print(f"\nreport written to {out / 'report.md'}. Paste it back to Claude or the team.")
    return 0


def _without_front(req: dict) -> dict:
    req["observation"]["images"].pop("front")
    return req


if __name__ == "__main__":
    sys.exit(main())
