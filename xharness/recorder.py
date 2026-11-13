"""Write everything that happens to disk, one folder per robot session.

runs/<timestamp>_<goal-slug>/
    events.jsonl         one JSON line per event (request, decision, verify, chunk, error)
    frames/000012_front.jpg  camera frames, so verifier prompts can be replayed offline
    requests/000012.msgpack  (optional) raw observation for exact replay

Never written: API keys, handshake headers, full HTTP bodies.
"""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import RecorderConfig
from .protocol import CAMERAS, pack


def _slug(text: str, limit: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit] or "session"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.round(5).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class SessionRecorder:
    def __init__(self, cfg: RecorderConfig, label: str = "session", save_raw: bool = False):
        self.cfg = cfg
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(cfg.dir) / f"{stamp}_{_slug(label)}"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "frames").mkdir(exist_ok=True)
        self.save_raw = save_raw
        if save_raw:
            (self.dir / "requests").mkdir(exist_ok=True)
        self._events = (self.dir / "events.jsonl").open("a", buffering=1)

    def event(self, kind: str, **fields: Any) -> None:
        record = {"t": round(time.time(), 3), "event": kind, **_jsonable(fields)}
        self._events.write(json.dumps(record) + "\n")

    def frames(self, index: int, request: dict) -> None:
        if not self.cfg.save_frames or index % max(1, self.cfg.frame_every):
            return
        images = request.get("observation", {}).get("images", {})
        for cam in CAMERAS:
            data = images.get(cam)
            if not isinstance(data, str):
                continue
            if data.startswith("data:"):
                data = data.split(",", 1)[1]
            try:
                (self.dir / "frames" / f"{index:06d}_{cam}.jpg").write_bytes(base64.b64decode(data))
            except ValueError:
                pass
        if self.save_raw:
            (self.dir / "requests" / f"{index:06d}.msgpack").write_bytes(pack(request))

    def close(self) -> None:
        self._events.close()
