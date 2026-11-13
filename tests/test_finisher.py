"""Finisher tests: detection, servo, press, aborts, loop-back, and full sim runs."""

from __future__ import annotations

import asyncio
import dataclasses

import numpy as np

from sim.camera import render_views
from sim.run import run_case
from sim.scenes import _hold, _prepare
from xharness.config import FilterConfig, FinisherConfig, SupervisorConfig, ArmBox
from xharness.finisher import Finisher, detect
from xharness.protocol import read_state
from xharness.supervisor import FINISH, HOLD, Supervisor
from xharness.tasks import DoneCheck, Stage, Task, load_task
from xharness.verifier import Verdict


def observation(world):
    return {"embodiment": "quanta_x1_pro/gripper-g", "instruction": "g",
            "observation": {"images": render_views(world), "state": world.state()}}


def drive(world, part, arm, offset, cfg=None, max_steps=150, hook=None, after_hold=None):
    """Run a Finisher directly against the fake world. Returns (last result, finisher)."""
    slot = world.free_slot_for(part)
    _hold(world, arm, part, slot.xy + np.array(offset), slot.seat_z + 0.03)
    if after_hold:
        after_hold(world)
    fin = Finisher(cfg or FinisherConfig(), FilterConfig(), part, None)
    for _ in range(max_steps):
        req = observation(world)
        res = fin.step(req, read_state(req), {arm: 0.0})
        if hook:
            hook(fin)
        if res.status != "running":
            return res, fin
        world.execute(res.actions)
    return res, fin


def test_detect_finds_held_part_and_round_hole():
    world = _prepare(1, 0)
    slot = world.slots["handle_slot"]
    _hold(world, "left_ee", "slider", slot.xy + [0.008, 0.006], slot.seat_z + 0.03)
    det = detect(render_views(world)["left_wrist"], "orange", FinisherConfig())
    assert det.confidence > 0.8 and det.target_px is not None
    assert abs(det.part_px[0] - 320) < 3 and abs(det.part_px[1] - 240) < 3  # held part sits at the image center


def test_finisher_seats_each_part_from_5_to_15_mm_off():
    rng = np.random.default_rng(3)
    cases = [(1, "slider", (), "left_ee"), (2, "shell", (), "left_ee"),
             (3, "button_a", (), "right_ee"), (3, "button_b", ("button_a",), "right_ee")]
    for case, part, before, arm in cases:
        for _ in range(3):
            r, th = rng.uniform(0.005, 0.015), rng.uniform(0, 2 * np.pi)
            world = _prepare(case, int(rng.integers(100)), before)
            res, _ = drive(world, part, arm, (r * np.cos(th), r * np.sin(th)))
            assert res.status == "done", (part, r, res.reason)
            assert world.seated(part), (part, r)


def test_wrong_sign_jacobian_aborts_to_hold():
    """With Broyden off, a flipped Jacobian drives the error up; two rises in a row abort."""
    world = _prepare(1, 0)
    cfg = FinisherConfig(broyden=False)

    def flip(fin):
        if fin.phase == "servo" and not getattr(fin, "_flipped", False):
            fin.J, fin._flipped = -fin.J, True

    res, _ = drive(world, "slider", "left_ee", (0.010, 0.004), cfg, hook=flip)
    assert res.status == "aborted" and "grew" in res.reason
    assert not world.seated("slider")


def test_missing_hole_aborts():
    world = _prepare(1, 0)
    res, _ = drive(world, "slider", "left_ee", (0.010, 0.0),
                   after_hold=lambda w: [setattr(sl, "filled_by", "x") for sl in w.slots.values()])  # no hole is drawn any more
    assert res.status == "aborted" and "detection lost" in res.reason


def test_chunks_that_fail_the_action_filter_abort():
    world = _prepare(1, 0)
    slot = world.slots["handle_slot"]
    _hold(world, "left_ee", "slider", slot.xy + [0.01, 0.0], slot.seat_z + 0.03)
    tiny = {arm: ArmBox(min=[0.0, 0.0, 0.0], max=[0.1, 0.1, 0.1]) for arm in ("left_ee", "right_ee")}
    fin = Finisher(FinisherConfig(), FilterConfig(workspace=tiny), "slider", None)
    req = observation(world)
    res = fin.step(req, read_state(req), {"left_ee": 0.0})
    assert res.status == "aborted" and "action filter" in res.reason and res.actions is None


def test_press_stops_at_the_hard_floor():
    world = _prepare(1, 0)
    cfg = FinisherConfig(max_press_depth_mm=5.0)  # floor far above the seat: never makes contact
    res, fin = drive(world, "slider", "left_ee", (0.002, 0.0), cfg)
    assert res.status == "done" and not world.seated("slider")
    assert fin.press_z0 - world.arms["left_ee"].pos[2] < 0.0  # lifted back above where it started


def finisher_task():
    return Task("t", "g", [
        Stage("grasp_x", ["Pick up."], DoneCheck("held?", 1), budget_requests=4, max_retries=0),
        Stage("seated_x", [], DoneCheck("seated?", 1), budget_requests=2, max_retries=0, hold=True,
              on_fail_goto="grasp_x", max_loops=1),
    ])


class Scripted:
    def __init__(self, answers):
        self.answers = list(answers)

    async def check(self, *a, **k):
        return Verdict(self.answers.pop(0) if self.answers else "no", "scripted", 1.0)


def test_failed_seated_check_loops_back_once_then_fails(tmp_path):
    from tests.test_units import make_request

    cfg = SupervisorConfig(verify_every=1, state_file=str(tmp_path / "s.json"))
    fin_cfg = FinisherConfig(enabled=True)
    sup = Supervisor([finisher_task()], cfg, Scripted(["yes"] + ["no"] * 40 + ["yes"]), fin_cfg)
    req = make_request(goal="g")
    st = read_state(req)
    stages = []
    for _ in range(40):
        d = asyncio.run(sup.decide(req, st))
        stages.append(d.stage_id)
        if sup.runs["g"].status != "running":
            break
    run = sup.runs["g"]
    assert run.status == "failed" and run.loops == {"seated_x": 1}
    assert stages.count("grasp_x") >= 2  # it went back to grasp once


def test_loop_back_is_ignored_when_the_finisher_is_off(tmp_path):
    from tests.test_units import make_request

    cfg = SupervisorConfig(verify_every=1, state_file=str(tmp_path / "s.json"))
    sup = Supervisor([finisher_task()], cfg, Scripted(["yes"] + ["no"] * 40))
    req = make_request(goal="g")
    for _ in range(30):
        asyncio.run(sup.decide(req, read_state(req)))
    assert sup.runs["g"].status == "failed" and sup.runs["g"].loops == {}


def test_yaml_stage_lists_per_mode():
    import glob

    for path in glob.glob("config/tasks/d2_*.yaml"):
        off = [s.id for s in load_task(path, False).stages]
        on = [s.id for s in load_task(path, True).stages]
        assert not any(i.startswith("finish_") for i in off), path
        assert not any(i.startswith(("push_", "release_")) for i in on), path
        stages = load_task(path, True).stages
        for i, s in enumerate(stages):
            if s.type == "finisher":
                assert stages[i - 1].id.startswith("align_") and stages[i + 1].hold, path
                assert s.on_fail_goto == stages[i - 1].id


def test_full_sim_runs_with_the_finisher_when_wallx_leaves_parts_off():
    for case in (1, 2, 3):
        r = asyncio.run(run_case(case, verbose=False, max_requests=300, place_offset_mm=(5, 15), seed=1,
                                 cfg=_finisher_cfg()))
        assert r["task_status"] == "done", (case, r["stages"])
        assert all(row["supervisor"] == "done" for row in r["stages"]), case


def test_without_the_finisher_the_same_offsets_fail():
    r = asyncio.run(run_case(1, verbose=False, max_requests=150, place_offset_mm=(12, 15), seed=1))
    assert r["task_status"] != "done"


def _finisher_cfg():
    from xharness.config import load_config

    cfg = load_config(None)
    cfg.finisher = dataclasses.replace(cfg.finisher, enabled=True)
    return cfg
