"""End-to-end tests: fake robot -> relay -> fake Wall-X / fake vision model.

These are the tests that matter most before Saturday. Each one rehearses a
failure that would hurt in a no-touch official run.
"""

from __future__ import annotations

import asyncio

from sim.fake_wallx import FaultPlan
from sim.run import VLM_FAULTS, run_case


def run(**kwargs):
    return asyncio.run(run_case(verbose=False, max_requests=kwargs.pop("max_requests", 120), **kwargs))


def no_false_positives(result):
    return not any(row["flag"].startswith("FALSE") for row in result["stages"])


def test_clean_case1_completes_then_holds():
    r = run(case=1)
    assert r["task_status"] == "done"
    assert no_false_positives(r)
    assert r["hold_chunks"] > 0  # after done, the robot keeps receiving HOLD


def test_wallx_faults_never_reach_the_robot_as_bad_chunks():
    r = run(case=1, wallx_faults=FaultPlan({2: "nan", 3: "bad_shape", 4: "jump", 5: "error:missing_field"}))
    assert r["rejected_chunks"] == 0  # NaN / bad shape replaced with HOLD
    assert r["error_replies"] == 1  # documented error forwarded unchanged
    assert r["task_status"] == "done"


def test_wallx_crash_and_timeout_do_not_drop_the_robot():
    r = run(case=1, wallx_faults=FaultPlan({3: "close1011", 6: "text_frame", 9: "delay:3"}))
    assert r["reconnects"] == 0  # the robot never saw a disconnect
    assert r["task_status"] == "done"


def test_client_reconnect_keeps_progress():
    r = run(case=3, drop_connection=(8, 20, 40), max_requests=150)
    assert r["reconnects"] == 3
    assert no_false_positives(r)
    # Stages done before a reconnect are not repeated: grasp_pin appears once.
    assert [row["stage"] for row in r["stages"]].count("grasp_pin") == 1


def test_messy_vision_model_is_tolerated():
    r = run(case=2, vlm_faults=VLM_FAULTS["messy"], seed=1)
    assert r["task_status"] == "done"
    assert no_false_positives(r)


def test_prewarm_sends_one_synthetic_request_and_never_reaches_the_robot():
    from sim.fake_wallx import FakeWallX
    from sim.world import World
    from xharness.config import load_config
    from xharness.relay import Relay
    from xharness.supervisor import Supervisor
    from xharness.verifier import Verifier

    async def go():
        wallx = FakeWallX(World(case=1))
        server = await wallx.serve()
        cfg = load_config(None)
        cfg.wallx.endpoint = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        cfg.wallx.headers_file = ""
        cfg.recorder.dir = __import__("tempfile").mkdtemp()
        relay = Relay(cfg, Supervisor([], cfg.supervisor, Verifier(cfg.verifier)), verbose=False)
        result = await relay.prewarm()
        server.close()
        await server.wait_closed()
        return wallx, result

    wallx, result = asyncio.run(go())
    assert result["ok"] and result["latency_ms"] >= 0
    assert wallx.requests == 1 and wallx.instructions == ["Stay still."]


def test_prewarm_failure_is_swallowed():
    from xharness.config import load_config
    from xharness.relay import Relay
    from xharness.supervisor import Supervisor
    from xharness.verifier import Verifier

    cfg = load_config(None)
    cfg.wallx.endpoint = "ws://127.0.0.1:1"  # nothing listens here
    cfg.wallx.open_timeout_s = 1
    cfg.recorder.dir = __import__("tempfile").mkdtemp()
    relay = Relay(cfg, Supervisor([], cfg.supervisor, Verifier(cfg.verifier)), verbose=False)
    result = asyncio.run(relay.prewarm())
    assert result["ok"] is False and "error" in result


def test_all_three_cases_finish_done_with_the_press_release_seated_pattern():
    for case in (2, 3):  # case 1 is covered above
        r = run(case=case, max_requests=200)
        assert r["task_status"] == "done", case
        assert no_false_positives(r), case
        stages = [row["stage"] for row in r["stages"]]
        assert any(s.startswith("seated_") for s in stages), case
        assert all(row["supervisor"] == "done" for row in r["stages"]), case
