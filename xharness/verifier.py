"""Ask a vision model narrow yes/no questions about the current scene.

Design choice: the vision model never *plans*. The supervisor owns the plan
(a fixed list of stages). The model only answers questions like "Is the purple
pin held by the right gripper?" with yes / no / unsure. Narrow questions
are faster, cheaper, easier to test offline, and fail safely.

Robustness rules (each one fixes a real failure of the example harness):
* Any error (timeout, HTTP 4xx/5xx, bad JSON, empty reply) -> "unsure".
  It never raises into the relay, so it can never close the robot session.
* Replies wrapped in ```json fences, or with prose around the JSON, are
  accepted. The example harness crashed on those.
* Images are downscaled before upload to keep latency down.

Works with any OpenAI-compatible /v1/chat/completions endpoint that accepts
image_url content parts.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from PIL import Image

from .config import VerifierConfig

ANSWERS = ("yes", "no", "unsure")
USER_AGENT = "xsquare-harness/0.1 (+https://github.com/Galanafai/xsquare-harness)"

SYSTEM_PROMPT = """You check the state of a robot tabletop task from camera images.
Cameras: "front" is a fixed camera facing the robot. It is MIRRORED: the robot's
RIGHT arm appears on the LEFT side of the front image. "left_wrist" and
"right_wrist" are mounted on each gripper.
Answer only the question asked, about what is visible right now.
If the view is blocked or you cannot tell, answer "unsure". Never guess "yes".
Reply with JSON only: {"answer": "yes" | "no" | "unsure", "reason": "<one short sentence>"}"""


@dataclass
class Verdict:
    answer: str  # yes | no | unsure
    reason: str
    latency_ms: float
    error: str = ""  # non-empty when we fell back to "unsure" because of a failure
    # Diagnostics for tools/eval_verifier.py. The supervisor ignores them.
    thinking: bool = False  # reply carried reasoning text (think tags or a reasoning field)
    completion_tokens: int = 0

    @property
    def yes(self) -> bool:
        return self.answer == "yes"


def extract_json(text: str) -> dict | None:
    """Find the first JSON object in a model reply, tolerating fences and prose."""
    if not text:
        return None
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    # Last resort: first {...} block that parses.
    for match in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            value = json.loads(match.group(0))
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None


def normalize_answer(value: object) -> str:
    answer = str(value or "").strip().lower()
    if answer in ("true", "y"):
        return "yes"
    if answer in ("false", "n"):
        return "no"
    return answer if answer in ANSWERS else "unsure"


def downscale_jpeg_b64(image_b64: str, max_side: int) -> str:
    """Shrink a base64 JPEG so its longest side is max_side pixels."""
    if image_b64.startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    raw = base64.b64decode(image_b64)
    with Image.open(io.BytesIO(raw)) as img:
        img = img.convert("RGB")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=85)
    return base64.b64encode(out.getvalue()).decode("ascii")


class Verifier:
    def __init__(self, cfg: VerifierConfig):
        self.cfg = cfg
        self.api_key = os.environ.get(cfg.api_key_env, "")

    @property
    def available(self) -> bool:
        return bool(self.cfg.enabled and self.cfg.chat_url and self.cfg.model)

    async def check(self, check_id: str, question: str, request: dict, context: str = "") -> Verdict:
        """Ask one question about the scene in `request` (a public_v2 observation)."""
        started = time.monotonic()
        if not self.available:
            return Verdict("unsure", "verifier disabled or not configured", 0.0, error="disabled")
        try:
            body = self._build_body(check_id, question, request, context)
            text, meta = await asyncio.wait_for(
                asyncio.to_thread(self._post, body), timeout=self.cfg.timeout_s
            )
            parsed = extract_json(text)
            if parsed is None:
                return Verdict("unsure", "unparseable reply", _ms(started),
                               error=f"unparseable: {text[:120]!r}", **meta)
            return Verdict(normalize_answer(parsed.get("answer")), str(parsed.get("reason", ""))[:300],
                           _ms(started), **meta)
        except asyncio.TimeoutError:
            return Verdict("unsure", "timeout", _ms(started), error="timeout")
        except urllib.error.HTTPError as err:
            # Keep a short slice of the error body: it says WHY (bad model id,
            # rate limit, blocked client). It never contains our API key.
            detail = _error_body(err)
            return Verdict("unsure", f"HTTP {err.code}", _ms(started), error=f"http_{err.code}: {detail}")
        except Exception as err:  # noqa: BLE001 - verifier must never raise
            return Verdict("unsure", type(err).__name__, _ms(started), error=type(err).__name__)

    # -- internals ----------------------------------------------------------

    def _build_body(self, check_id: str, question: str, request: dict, context: str) -> dict:
        text = f"Check ID: {check_id}\nQuestion: {question}"
        if context:
            text = f"Scene context: {context}\n{text}"
        content: list[dict] = [{"type": "text", "text": text}]
        images = request["observation"]["images"]
        for cam in self.cfg.cameras:
            small = downscale_jpeg_b64(images[cam], self.cfg.image_max_side)
            content.append({"type": "text", "text": f"camera: {cam}"})
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + small}})
        body = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        }
        body.update(self.cfg.extra_body)
        return body

    def _post(self, body: dict) -> tuple[str, dict]:
        # Some providers sit behind Cloudflare, which rejects Python's default
        # "Python-urllib/3.x" user agent with HTTP 403 (seen on Groq). Name ourselves.
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(self.cfg.chat_url, json.dumps(body).encode(), headers=headers)
        # The socket timeout is a backstop; asyncio.wait_for enforces the real one.
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s + 2) as resp:
            result = json.load(resp)
        message = result["choices"][0]["message"]
        text = message.get("content") or ""
        thinking = bool(message.get("reasoning") or message.get("reasoning_content") or "<think>" in text)
        tokens = int((result.get("usage") or {}).get("completion_tokens") or 0)
        return text, {"thinking": thinking, "completion_tokens": tokens}


def _error_body(err: urllib.error.HTTPError, limit: int = 200) -> str:
    try:
        return err.read().decode("utf-8", "replace")[:limit]
    except Exception:  # noqa: BLE001
        return ""


def _ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000, 1)
