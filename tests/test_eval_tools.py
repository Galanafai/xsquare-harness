"""Tests for the verifier evaluation tooling (no network)."""

import asyncio
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ev = _load("eval_verifier", "tools/eval_verifier.py")
seed = _load("seed_sim_images", "tools/seed_sim_images.py")


def test_sim_scene_labels_agree_with_sim_truth():
    from sim.scenes import all_scenes

    scenes = all_scenes()
    assert seed.truth_conflicts(scenes) == []
    # every stage has both classes and at least one hard negative
    by_folder = {}
    for sc in scenes:
        by_folder.setdefault(seed.folder_name(sc), []).append(sc)
    for folder, group in by_folder.items():
        assert {s.label for s in group} == {"yes", "no"}, folder
        assert any(s.name.startswith("hard_") for s in group if s.label == "no"), folder


def test_pick_mixes_easy_and_hard_negatives():
    names = ["hard_a", "hard_b", "no_easy", "no_other"]
    assert ev.pick(names, 2) == ["no_easy", "hard_a"]
    assert ev.pick(["x"], 3) == ["x"]


def test_summarize_counts_false_yes_and_unsure():
    base = {"error": "", "thinking": False, "completion_tokens": 30, "latency_ms": 1000.0}
    rows = [
        dict(base, label="yes", answer="yes"),
        dict(base, label="no", answer="yes"),
        dict(base, label="no", answer="unsure"),
        dict(base, label="no", answer="no", latency_ms=3000.0),
    ]
    s = ev.summarize(rows)
    assert (s["n"], s["false_yes"], s["n_no"]) == (4, 1, 3)
    assert s["acc"] == 0.5 and s["unsure"] == 0.25
    assert s["max_ms"] == 3000.0


def test_throttle_waits_when_token_budget_is_full():
    async def go():
        th = ev.Throttle(rpm=100, itpm=1000, tokens_per_call=600)
        await th.wait()
        started = time.monotonic()
        # Fake the first call as 59 s old so the second needs only ~2 s.
        th.sent = [(started - 59.0, 600)]
        await th.wait()
        return time.monotonic() - started

    waited = asyncio.run(go())
    assert 1.0 < waited < 4.0


def test_verifier_reports_thinking_and_tokens(monkeypatch):
    from xharness.config import VerifierConfig
    from xharness.verifier import Verifier

    v = Verifier(VerifierConfig(chat_url="http://x", model="m"))
    monkeypatch.setattr(v, "_post", lambda body: ('{"answer": "yes", "reason": "r"}', {"thinking": True, "completion_tokens": 42}))
    request = {"observation": {"images": {}}}
    monkeypatch.setattr(v, "_build_body", lambda *a: {})
    verdict = asyncio.run(v.check("id", "q", request))
    assert verdict.yes and verdict.thinking and verdict.completion_tokens == 42
