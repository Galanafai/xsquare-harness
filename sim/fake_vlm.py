"""A stand-in vision model with an OpenAI-compatible /v1/chat/completions API.

It answers the supervisor's yes/no questions from the fake world's ground
truth (sim/world.py STAGE_TRUTH), looked up by the "Check ID:" line the
verifier puts in every prompt. It can also misbehave on purpose, the way
real hosted models do:

    fenced     reply wrapped in ```json fences       (crashed the example harness)
    prose      JSON buried in a sentence
    http500    server error
    slow       sleeps past the verifier timeout
    wrong      flips the true answer (a confident false positive)

Pass a probability for each in `faults`, e.g. {"fenced": 0.3, "wrong": 0.05}.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .world import STAGE_TRUTH, World


class FakeVLM:
    def __init__(self, world: World, faults: dict[str, float] | None = None, slow_s: float = 15.0, seed: int = 0):
        self.world = world
        self.faults = faults or {}
        self.slow_s = slow_s
        self.rng = random.Random(seed)
        self.calls = 0
        self.server: ThreadingHTTPServer | None = None

    def answer_for(self, check_id: str) -> str:
        stage = check_id.split(".")[-1]
        truth = STAGE_TRUTH.get(stage)
        if truth is None:
            return "unsure"
        return "yes" if truth(self.world) else "no"

    def reply_text(self, prompt_text: str) -> tuple[int, str]:
        """Return (http_status, content) for one request."""
        self.calls += 1
        match = re.search(r"Check ID:\s*(\S+)", prompt_text)
        answer = self.answer_for(match.group(1)) if match else "unsure"
        roll = self.rng.random

        if roll() < self.faults.get("http500", 0):
            return 500, ""
        if roll() < self.faults.get("slow", 0):
            time.sleep(self.slow_s)
        if roll() < self.faults.get("wrong", 0) and answer in ("yes", "no"):
            answer = "no" if answer == "yes" else "yes"
        body = json.dumps({"answer": answer, "reason": "fake oracle"})
        if roll() < self.faults.get("fenced", 0):
            body = f"```json\n{body}\n```"
        elif roll() < self.faults.get("prose", 0):
            body = f"Looking at the images, here is my answer: {body} Hope that helps."
        return 200, body

    def start(self, host: str = "127.0.0.1", port: int = 0) -> str:
        vlm = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                text = " ".join(
                    part.get("text", "")
                    for msg in body.get("messages", [])
                    for part in (msg["content"] if isinstance(msg["content"], list) else [{"text": msg["content"]}])
                )
                status, content = vlm.reply_text(text)
                payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer((host, port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://{host}:{self.server.server_port}/v1/chat/completions"

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
