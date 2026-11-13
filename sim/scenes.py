"""Labeled still scenes for testing the vision verifier (not the robot loop).

For every stage question in config/tasks/*.yaml this builds World states where
the stage is truly done ("yes") and truly not done ("no"), including hard
negatives that look almost right. Used by tools/seed_sim_images.py.

Everything here is FAKE geometry from sim/world.py. Real phone photos of the
printed parts replace these renders; the folder layout stays the same.

Sample names starting with "hard_" are the hard negatives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xharness.parts import GRIPPER

from .world import World

OPEN = GRIPPER.max_command


@dataclass
class Scene:
    stage: str  # stage id from the task YAML
    task_id: str
    label: str  # "yes" | "no"
    name: str  # sample folder name
    world: World


def _hold(world: World, arm: str, part_name: str, xy, bottom_z: float) -> None:
    """Put `part_name` in `arm`'s fingers with its bottom face at bottom_z."""
    part, a = world.parts[part_name], world.arms[arm]
    a.pos = np.array([xy[0], xy[1], bottom_z + part.size[2] / 2])
    a.holding, part.held_by = part_name, arm
    a.gripper = GRIPPER.command_for_opening(part.grasp_width_mm)
    part.pos = np.array([xy[0], xy[1], bottom_z])


def _near(world: World, arm: str, part_name: str) -> None:
    """Open fingers straddling a part that is still on the table."""
    part = world.parts[part_name]
    world.arms[arm].pos = np.array([part.pos[0], part.pos[1], part.size[2] / 2])
    world.arms[arm].gripper = OPEN


def _touch_closed(world: World, arm: str, part_name: str) -> None:
    """Fingers closed on a part that has NOT been lifted off the table."""
    part = world.parts[part_name]
    _hold(world, arm, part_name, part.pos[:2], 0.0)


def _offset(xy, dx: float, dy: float):
    return np.array([xy[0] + dx, xy[1] + dy])


def _prepare(case: int, seed: int, seated_first: tuple[str, ...] = ()) -> World:
    world = World(case, seed=seed)
    for name in seated_first:  # earlier buttons of case 3 are already in
        slot = world.free_slot_for(name)
        world._seat(name, slot.name)
    return world


def _part_stage_scenes(task_id, case, part, slot_name, ids, arm_pair, radius, seated_first=(), seed0=0):
    """grasp / align / press scenes for one part. `ids` = (grasp, align, press)."""
    grasp_id, align_id, press_id = ids
    out: list[Scene] = []

    def add(stage, label, name, world):
        out.append(Scene(stage, task_id, label, name, world))

    def fresh(seed):
        return _prepare(case, seed, seated_first)

    # --- grasp: lifted off the table in the fingers
    for i, arm in enumerate(arm_pair):
        w = fresh(seed0 + i)
        start = w.parts[part].pos[:2].copy()
        _hold(w, arm, part, start, 0.10 + 0.02 * i)
        add(grasp_id, "yes", f"yes_lifted_{arm[:1]}", w)
    w = fresh(seed0 + 3)
    add(grasp_id, "no", "no_on_table", w)
    w = fresh(seed0 + 4)
    _near(w, arm_pair[0], part)
    add(grasp_id, "no", "hard_fingers_open_around_part", w)
    w = fresh(seed0 + 5)
    _touch_closed(w, arm_pair[1], part)
    add(grasp_id, "no", "hard_gripped_but_on_table", w)

    # --- align: held right above the slot
    slot = fresh(0).slots[slot_name]
    slot_xy = slot.xy
    for i, arm in enumerate(arm_pair):
        w = fresh(seed0 + 10 + i)
        _hold(w, arm, part, _offset(slot_xy, 0.006 * (i + 1), -0.004 * i), slot.seat_z + 0.03 + 0.01 * i)
        add(align_id, "yes", f"yes_above_{arm[:1]}", w)
    w = fresh(seed0 + 13)
    _hold(w, arm_pair[0], part, w.parts[part].pos[:2], 0.12)
    add(align_id, "no", "no_lifted_far_away", w)
    w = fresh(seed0 + 14)
    _hold(w, arm_pair[0], part, _offset(slot_xy, 0.0, 3 * radius), slot.seat_z + 0.03)
    add(align_id, "no", "hard_beside_slot", w)
    w = fresh(seed0 + 15)
    _hold(w, arm_pair[1], part, _offset(slot_xy, 0.0, 0.0), slot.seat_z + 0.16)
    add(align_id, "no", "hard_too_high", w)

    # --- press / seat
    for i, arm in enumerate(arm_pair):
        w = fresh(seed0 + 20 + i)
        w._seat(part, w.free_slot_for(part).name)
        if i:
            w.arms[arm].pos = w.arms[arm].pos + [0.05, 0.1, 0.1]
        add(press_id, "yes", f"yes_seated_{arm[:1]}", w)
    w = fresh(seed0 + 23)
    add(press_id, "no", "no_on_table", w)
    w = fresh(seed0 + 24)
    _hold(w, arm_pair[0], part, slot_xy, slot.seat_z + 0.012)
    add(press_id, "no", "hard_resting_on_slot", w)
    w = fresh(seed0 + 25)
    _hold(w, arm_pair[1], part, _offset(slot_xy, 0.0, slot.tol_m + 0.006), slot.seat_z + 0.01)
    add(press_id, "no", "hard_jammed_on_rim", w)
    return out


def _release_scenes(task_id, case, last_part, arm_pair, seated_before=()):
    """release_retract: arms open and away, the final part still seated."""
    out: list[Scene] = []

    def fresh(seed):
        return _prepare(case, seed, seated_before)

    for i, arm in enumerate(arm_pair):
        w = fresh(30 + i)
        w._seat(last_part, w.free_slot_for(last_part).name)
        if i:
            w.arms[arm].pos = w.arms[arm].pos + [0.05, 0.1, 0.1]
        out.append(Scene("release_retract", task_id, "yes", f"yes_open_clear_{arm[:1]}", w))
    # Hard: gripper still closed on the part, resting on its slot.
    w = fresh(33)
    slot = w.free_slot_for(last_part)
    _hold(w, arm_pair[0], last_part, slot.xy, slot.seat_z + 0.012)
    out.append(Scene("release_retract", task_id, "no", "hard_still_holding_part", w))
    # Hard: arms open, but the part was dropped on the table instead of seated.
    out.append(Scene("release_retract", task_id, "no", "hard_part_left_on_table", fresh(34)))
    return out


def all_scenes() -> list[Scene]:
    """Scenes for every stage of the three Demo 2 cases."""
    s: list[Scene] = []
    s += _part_stage_scenes("d2_case1_insert_slider", 1, "slider", "handle_slot",
                            ("grasp_slider", "align_slider", "insert_slider"), ("left_ee", "right_ee"), 0.03)
    s += _release_scenes("d2_case1_insert_slider", 1, "slider", ("left_ee", "right_ee"))
    s += _part_stage_scenes("d2_case2_snap_shell", 2, "shell", "shell_seat",
                            ("grasp_shell", "align_shell", "press_shell"), ("left_ee", "right_ee"), 0.03, seed0=100)
    s += _release_scenes("d2_case2_snap_shell", 2, "shell", ("left_ee", "right_ee"))
    s += _part_stage_scenes("d2_case3_press_buttons", 3, "button_a", "hole_a",
                            ("grasp_teal", "align_teal", "press_teal"), ("right_ee", "left_ee"), 0.02, seed0=200)
    s += _part_stage_scenes("d2_case3_press_buttons", 3, "button_b", "hole_b",
                            ("grasp_purple", "align_purple", "press_purple"), ("right_ee", "left_ee"), 0.02,
                            seated_first=("button_a",), seed0=300)
    s += _release_scenes("d2_case3_press_buttons", 3, "button_b", ("right_ee", "left_ee"),
                        seated_before=("button_a",))
    return s
