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
    # Stages done before a reconnect are not repeated: grasp_teal appears once.
    assert [row["stage"] for row in r["stages"]].count("grasp_teal") == 1


def test_messy_vision_model_is_tolerated():
    r = run(case=2, vlm_faults=VLM_FAULTS["messy"], seed=1)
    assert r["task_status"] == "done"
    assert no_false_positives(r)
