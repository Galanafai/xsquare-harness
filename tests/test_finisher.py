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


def drive(world, part, arm, offset, cfg=None, max_steps=200, hook=None, after_hold=None, seated_z="world"):
    """Run a Finisher directly against the fake world. Returns (last result, finisher)."""
    slot = world.free_slot_for(part)
    _hold(world, arm, part, slot.xy + np.array(offset), slot.seat_z + 0.03)
    if after_hold:
        after_hold(world)
    z = world.seated_gripper_z(part) if seated_z == "world" else seated_z
    fin = Finisher(cfg or FinisherConfig(), FilterConfig(), part, None, z)
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
    fin = Finisher(FinisherConfig(), FilterConfig(workspace=tiny), "slider", None, world.seated_gripper_z("slider"))
    req = observation(world)
    res = fin.step(req, read_state(req), {"left_ee": 0.0})
    assert res.status == "aborted" and "action filter" in res.reason and res.actions is None


def test_stopping_on_first_contact_does_not_complete_the_snap_but_the_new_logic_does():
    # Old logic: step down until the arm stops following, then stop. The sim part resists 1.5 mm above its seat.
    world = _prepare(1, 0)
    slot = world.slots["handle_slot"]
    _hold(world, "left_ee", "slider", slot.xy, slot.seat_z + 0.02)
    arm, z = world.arms["left_ee"], world.arms["left_ee"].pos[2]
    for _ in range(40):
        cmd = z - 0.001
        pos = np.tile(arm.pos, (32, 1))
        pos[:, 2] = cmd
        actions = {a: {"position": np.tile(world.arms[a].pos, (32, 1)), "rotation_rpy": np.tile(world.arms[a].rpy, (32, 1)),
                       "gripper": np.full(32, 0.0)} for a in world.arms}
        actions["left_ee"]["position"] = pos
        world.execute(actions)
        if arm.pos[2] - cmd > 0.0005:  # first contact: the arm stopped following
            break
        z = cmd
    assert not world.seated("slider")
    assert abs(world.parts["slider"].pos[2] - (slot.seat_z + 0.0015)) < 1e-4  # resting on the snap

    world = _prepare(1, 0)
    res, _ = drive(world, "slider", "left_ee", (0.0, 0.0))
    assert res.status == "done" and world.seated("slider")


def test_each_press_target_is_sent_twice_and_descent_is_fast_then_slow():
    world = _prepare(1, 0)
    sent = []

    def spy(fin):
        if fin.phase == "press" and fin.pending:
            sent.append((fin.pending["cmd"], fin.pending["sends"]))

    res, fin = drive(world, "slider", "left_ee", (0.003, 0.0), hook=spy)
    assert res.status == "done"
    cmds = [c for c, _ in sent]
    assert [n for _, n in sent][:4] == [1, 2, 1, 2]  # every target twice
    distinct = list(dict.fromkeys(cmds))
    steps = -np.diff(distinct) * 1000
    seated = world.seated_gripper_z("slider")
    fast = [d for d, c in zip(steps, distinct) if c - seated > 0.003]
    assert max(steps) <= 5.0 + 1e-6 and fast and all(abs(d - 5.0) < 1e-6 for d in fast[:-1])
    assert abs(steps[-1] - 1.0) < 1.01 and abs(distinct[-1] - (seated - 0.0005)) < 1e-9  # ends at seated_z - overtravel


def test_jam_above_seated_z_lifts_keeps_the_part_and_aborts_back_to_align():
    world = _prepare(1, 0)

    def tiny_tolerance(w):
        w.slots["handle_slot"].tol_m = 1e-6  # even an aligned part now jams on the rim

    res, fin = drive(world, "slider", "left_ee", (0.0, 0.0), after_hold=tiny_tolerance)
    assert res.status == "aborted" and "jam" in res.reason
    assert world.arms["left_ee"].holding == "slider"  # never released on a jam
    assert not any("released" in e for e in world.events)
    rim_z = world.slots["handle_slot"].seat_z + 0.01 + world.parts["slider"].size[2] / 2
    assert world.arms["left_ee"].pos[2] > rim_z + 0.009  # lifted about 10 mm


def test_unset_seated_z_aborts_before_any_motion():
    world = _prepare(1, 0)
    slot = world.slots["handle_slot"]
    _hold(world, "left_ee", "slider", slot.xy + [0.01, 0.0], slot.seat_z + 0.03)
    fin = Finisher(FinisherConfig(), FilterConfig(), "slider", None, None)
    req = observation(world)
    res = fin.step(req, read_state(req), {"left_ee": 0.0})
    assert res.status == "aborted" and "seated_z" in res.reason and res.actions is None


def test_wrong_seated_z_is_caught_as_a_jam_or_the_hard_floor():
    # seated_z 6 mm lower than reality: the arm bottoms out far above the "seat" = jam, never a release
    world = _prepare(1, 0)
    res, _ = drive(world, "slider", "left_ee", (0.0, 0.0), seated_z=world.seated_gripper_z("slider") - 0.006)
    assert res.status == "aborted" and "jam" in res.reason
    assert world.arms["left_ee"].holding == "slider"  # not released
    # seated_z raised mid-press so the arm is below the absolute floor: abort
    world = _prepare(1, 0)

    def raise_seated_z(fin):
        if fin.phase == "press" and fin.pending and fin.pending["cmd"] < fin.seated_z + 0.004:
            fin.seated_z += 0.02

    res, _ = drive(world, "slider", "left_ee", (0.0, 0.0), hook=raise_seated_z)
    assert res.status == "aborted" and "hard floor" in res.reason


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
