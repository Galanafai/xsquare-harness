"""Fast unit tests: protocol, verifier parsing, action filter, supervisor logic."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from xharness.action_filter import ActionFilter
from xharness.config import ArmBox, FilterConfig, SupervisorConfig
from xharness.protocol import (
    ACTION_ROWS,
    ProtocolError,
    build_response,
    hold_actions,
    pack,
    read_state,
    unpack,
    validate_request,
    validate_response,
)
from xharness.supervisor import ACT, HOLD, PASS, Supervisor
from xharness.tasks import DoneCheck, Stage, Task
from xharness.verifier import Verdict, extract_json, normalize_answer

EMB = "quanta_x1_pro/gripper-g"


def make_request(goal="Press the two buttons into the toy.", left=(0.3, 0.2, 0.2), right=(0.3, -0.2, 0.2), grip=1.5):
    return {
        "embodiment": EMB,
        "normalization": "provided",
        "instruction": goal,
        "observation": {
            "images": {c: "aW1n" for c in ("front", "left_wrist", "right_wrist")},
            "state": {
                "left_ee": {"position": list(left), "rotation_rpy": [0, 0, 0], "gripper": grip},
                "right_ee": {"position": list(right), "rotation_rpy": [0, 0, 0], "gripper": grip},
            },
        },
    }


# ----------------------------------------------------------------- protocol


def test_pack_roundtrip_and_text_frame_rejected():
    assert unpack(pack({"a": np.arange(3)}))["a"].tolist() == [0, 1, 2]
    with pytest.raises(ProtocolError):
        unpack("Traceback ...")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(instruction=""),
        lambda r: r.update(embodiment="quanta_x1_pro/gripper-G"),  # capital G fails on the server too
        lambda r: r.update(use_rtc=True),
        lambda r: r["observation"]["images"].pop("front"),
        lambda r: r["observation"]["state"]["left_ee"].update(gripper=float("nan")),
        lambda r: r["observation"]["state"]["right_ee"].update(position=[0, 0]),
    ],
)
def test_validate_request_rejects(mutate):
    r = make_request()
    mutate(r)
    with pytest.raises(ProtocolError):
        validate_request(r)


def test_hold_response_is_valid_and_stationary():
    state = read_state(make_request())
    resp = build_response(EMB, hold_actions(state))
    actions = validate_response(unpack(pack(resp)), EMB)
    assert actions["left_ee"]["position"].shape == (ACTION_ROWS, 3)
    assert np.allclose(actions["left_ee"]["position"], [0.3, 0.2, 0.2])


# ----------------------------------------------------------------- verifier parsing


@pytest.mark.parametrize(
    "text,answer",
    [
        ('{"answer": "yes", "reason": "ok"}', "yes"),
        ('```json\n{"answer": "no", "reason": "x"}\n```', "no"),
        ('Sure! Here you go: {"answer": "unsure", "reason": "blocked"} hope it helps', "unsure"),
        ("I think yes", None),
    ],
)
def test_extract_json_tolerates_fences_and_prose(text, answer):
    parsed = extract_json(text)
    assert (parsed or {}).get("answer") == answer


def test_normalize_answer():
    assert normalize_answer("YES") == "yes"
    assert normalize_answer(True) == "yes"
    assert normalize_answer("maybe") == "unsure"


# ----------------------------------------------------------------- action filter


def chunk(start, end, grip=1.0):
    t = np.linspace(0, 1, ACTION_ROWS)[:, None]
    pos = np.asarray(start) + t * (np.asarray(end) - np.asarray(start))
    return {"position": pos, "rotation_rpy": np.zeros((ACTION_ROWS, 3)), "gripper": np.full(ACTION_ROWS, grip)}


def test_filter_accepts_normal_chunk_and_flags_jump_and_box():
    state = read_state(make_request())
    good = build_response(EMB, {"left_ee": chunk((0.3, 0.2, 0.2), (0.32, 0.2, 0.15)),
                                "right_ee": chunk((0.3, -0.2, 0.2), (0.3, -0.2, 0.2))})
    f = ActionFilter(FilterConfig())
    assert f.check(good, EMB, state).ok

    jump = build_response(EMB, {"left_ee": chunk((0.3, 0.2, 0.6), (0.3, 0.2, 0.6)),
                                "right_ee": chunk((0.3, -0.2, 0.2), (0.3, -0.2, 0.2))})
    result = f.check(jump, EMB, state)
    assert not result.ok and any("row 0" in v for v in result.violations)

    boxed = ActionFilter(FilterConfig(workspace={"left_ee": ArmBox([0, 0, 0.18], [1, 1, 1]),
                                                 "right_ee": ArmBox()}))
    assert any("workspace" in v for v in boxed.check(good, EMB, state).violations)


def test_filter_rejects_nan_even_in_log_mode():
    state = read_state(make_request())
    bad = chunk((0.3, 0.2, 0.2), (0.3, 0.2, 0.2))
    bad["position"][3, 1] = np.nan
    resp = build_response(EMB, {"left_ee": bad, "right_ee": chunk((0.3, -0.2, 0.2), (0.3, -0.2, 0.2))})
    result = ActionFilter(FilterConfig(mode="log")).check(resp, EMB, state)
    assert not result.ok and result.violations[0].startswith("protocol")


def test_freeze_idle_arm_overwrites_other_arm():
    state = read_state(make_request())
    resp = build_response(EMB, {"left_ee": chunk((0.3, 0.2, 0.2), (0.35, 0.2, 0.2)),
                                "right_ee": chunk((0.3, -0.2, 0.2), (0.35, -0.2, 0.2))})
    result = ActionFilter(FilterConfig(freeze_idle_arm=True)).check(resp, EMB, state, active_arm="left_ee")
    assert np.allclose(result.modified_actions["right_ee"]["position"], [0.3, -0.2, 0.2])
    assert not np.allclose(result.modified_actions["left_ee"]["position"][-1], [0.3, 0.2, 0.2])


# ----------------------------------------------------------------- supervisor


class ScriptedVerifier:
    """Answers from a list, then 'no' forever."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    async def check(self, check_id, question, request, context=""):
        self.calls += 1
        return Verdict(self.answers.pop(0) if self.answers else "no", "scripted", 1.0)


def two_stage_task():
    return Task(
        task_id="t",
        goal="Press the two buttons into the toy.",
        stages=[
            Stage("grasp", ["Pick up the teal button."], DoneCheck("held?", confirmations=2), budget_requests=4, max_retries=1),
            Stage("press", ["Press it.", "Push it."], DoneCheck("seated?", confirmations=1), budget_requests=4, max_retries=0),
        ],
    )


def run(coro):
    return asyncio.run(coro)


def make_supervisor(tmp_path, answers, **cfg):
    cfg = SupervisorConfig(verify_every=1, state_file=str(tmp_path / "s.json"), **cfg)
    return Supervisor([two_stage_task()], cfg, ScriptedVerifier(answers))


def test_unknown_goal_passes_through(tmp_path):
    sup = make_supervisor(tmp_path, [])
    req = make_request(goal="Something else")
    assert run(sup.decide(req, read_state(req))).kind == PASS


def test_debounce_unsure_does_not_count_and_no_resets(tmp_path):
    sup = make_supervisor(tmp_path, ["yes", "unsure", "no", "yes", "yes", "yes"])
    req = make_request()
    st = read_state(req)
    stages = [run(sup.decide(req, st)).stage_id for _ in range(6)]
    # yes(1) unsure(1) no(0) yes(1) yes(2 -> advance) then press: yes(1/1 -> done)
    assert stages[:4] == ["grasp"] * 4
    assert stages[4] == "press"
    assert run(sup.decide(req, st)).kind == HOLD  # task done -> HOLD forever


def test_retries_rotate_phrasing_then_fail_to_hold(tmp_path):
    sup = make_supervisor(tmp_path, ["yes", "yes"])  # grasp done fast, press never
    req = make_request()
    st = read_state(req)
    decisions = [run(sup.decide(req, st)) for _ in range(8)]
    press = [d.instruction for d in decisions if d.stage_id == "press" and d.kind == ACT]
    assert press and press[0] == "Press it."
    assert decisions[-1].kind == HOLD and "retries" in decisions[-1].reason


def test_progress_survives_restart(tmp_path):
    sup = make_supervisor(tmp_path, ["yes", "yes"])
    req = make_request()
    st = read_state(req)
    for _ in range(3):
        run(sup.decide(req, st))
    # New supervisor = harness restarted or client reconnected: same file.
    sup2 = make_supervisor(tmp_path, [])
    assert run(sup2.decide(req, st)).stage_id == "press"


def test_gripper_holding_requires_stall(tmp_path):
    task = Task("t", "g", [Stage("grasp", ["Pick."], DoneCheck("held?", confirmations=1, gripper_holding="left_ee"))])
    cfg = SupervisorConfig(verify_every=1, state_file=str(tmp_path / "s.json"), grasp_band=[0.08, 1.3], stall_margin=0.05)
    verifier = ScriptedVerifier(["yes"] * 5)
    sup = Supervisor([task], cfg, verifier)
    # Commanded fully closed (0.0) and measured 0.02: fingers closed on nothing.
    sup.note_commanded_gripper("g", {"left_ee": 0.0, "right_ee": 1.5})
    empty = make_request(goal="g", grip=0.02)
    assert run(sup.decide(empty, read_state(empty))).stage_id == "grasp"
    assert verifier.calls == 0  # cheap gripper check failed: no vision call spent
    # Measured 0.30 while commanded 0.0: stalled on a part -> ask vision -> done.
    held = make_request(goal="g", grip=0.30)
    assert run(sup.decide(held, read_state(held))).kind == HOLD
