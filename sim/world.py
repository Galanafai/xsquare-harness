"""A tiny kinematic tabletop world for testing the harness without a robot.

This is NOT a physics simulator. It exists to exercise harness logic
(stages, retries, done detection, HOLD, faults) against something that
behaves like the task, using the real Demo 2 part dimensions.

What it models:
* Two arms whose fingertip point follows the commanded end-effector position.
* Grasping: closing the gripper near a part attaches it, if the part fits
  between the fingers. The gripper reading then STALLS at the part width,
  which is the signal the supervisor uses to tell "holding" from "empty".
* Insertion: a held part pressed down onto its slot seats only if it is
  within the slot's xy tolerance. Outside the tolerance it jams on top.
  Tolerances are deliberately tight for the buttons (the K = 4.5 step).

What it does not model: contact forces, friction, tipping, joint limits,
real camera optics. Treat all numbers here as fake unless marked measured.

Frame: robot base frame, metres. Table surface at z = 0 (FAKE; the real
table height relative to the base is unknown).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from xharness.parts import GRIPPER, PARTS
from xharness.protocol import ARMS, ACTION_ROWS

# Start poses are the example state from the API Tutorial (real numbers, in the
# robot base frame, so Wall-X sees a plausible pose). Everything else in this
# file (table at z = 0, part and holder positions, the camera) is FAKE and only
# chosen to keep parts within the 649 mm reach of those poses.
HOME = {
    "left_ee": np.array([0.2905, 0.0224, 0.3754]),
    "right_ee": np.array([0.2825, -0.0510, 0.3906]),
}
HOME_RPY = {
    "left_ee": np.array([0.2968, -0.3565, -0.1049]),
    "right_ee": np.array([0.0270, -0.5094, -0.3716]),
}


@dataclass
class SimPart:
    name: str
    size: np.ndarray  # metres (x, y, z)
    color: str
    pos: np.ndarray  # centre of the bottom face, metres
    fixed: bool = False  # sits in the holder; cannot be picked
    seated_in: str | None = None
    held_by: str | None = None

    @property
    def top_z(self) -> float:
        return float(self.pos[2] + self.size[2])

    @property
    def grasp_width_mm(self) -> float:
        return float(min(self.size[0], self.size[1]) * 1000)


@dataclass
class Slot:
    """Where a part goes. seat_z is where the part's bottom ends up when seated."""

    name: str
    accepts: tuple[str, ...]
    xy: np.ndarray
    seat_z: float
    tol_m: float  # xy capture tolerance (chamfer lead-in included)
    requires: str | None = None  # another slot that must be filled first
    filled_by: str | None = None


@dataclass
class Arm:
    pos: np.ndarray
    rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gripper: float = GRIPPER.max_command  # measured reading
    holding: str | None = None


class World:
    def __init__(self, case: int, seed: int = 0, tolerance_scale: float = 1.0):
        self.case = case
        self.rng = np.random.default_rng(seed)
        self.arms = {arm: Arm(pos=HOME[arm].copy(), rpy=HOME_RPY[arm].copy()) for arm in ARMS}
        self.parts: dict[str, SimPart] = {}
        self.slots: dict[str, Slot] = {}
        self.events: list[str] = []
        self._build(case, tolerance_scale)

    def log(self, text: str) -> None:
        """Record a world event, collapsing immediate repeats ('... x3')."""
        if self.events and self.events[-1].split(" x")[0] == text:
            head, _, count = self.events[-1].partition(" x")
            self.events[-1] = f"{head} x{int(count or 1) + 1}"
        else:
            self.events.append(text)

    # ------------------------------------------------------------ scene setup

    def _add_part(self, name: str, xy, z: float = 0.0, fixed: bool = False) -> SimPart:
        spec = PARTS[name]
        part = SimPart(name, np.array(spec.size_mm) / 1000.0, spec.color,
                       np.array([xy[0], xy[1], z], dtype=float), fixed=fixed)
        self.parts[name] = part
        return part

    def _build(self, case: int, tol: float) -> None:
        holder = np.array([0.40, 0.0])  # FAKE position
        handle = self._add_part("handle", holder, z=0.0, fixed=True)
        jitter = self.rng.uniform(-0.02, 0.02, size=2)  # position generalization
        top = handle.top_z
        self.slots["handle_slot"] = Slot("handle_slot", ("slider",), holder.copy(), top - 0.015, 0.004 * tol)
        self.slots["shell_seat"] = Slot("shell_seat", ("shell",), holder.copy(), top - 0.008, 0.005 * tol,
                                        requires="handle_slot")
        hole_z = top + 0.0235 - 0.010
        self.slots["hole_a"] = Slot("hole_a", ("button_a", "button_b"), holder + [0.015, 0.0], hole_z, 0.002 * tol,
                                    requires="shell_seat")
        self.slots["hole_b"] = Slot("hole_b", ("button_a", "button_b"), holder + [-0.015, 0.0], hole_z, 0.002 * tol,
                                    requires="shell_seat")

        if case == 1:
            self._add_part("slider", np.array([0.33, 0.09]) + jitter)
        elif case == 2:
            self._seat("slider", "handle_slot")
            self._add_part("shell", np.array([0.33, 0.09]) + jitter)
        elif case == 3:
            self._seat("slider", "handle_slot")
            self._seat("shell", "shell_seat")
            self._add_part("button_a", np.array([0.32, 0.07]) + jitter)
            self._add_part("button_b", np.array([0.32, 0.12]) + jitter)
        else:
            raise ValueError("case must be 1, 2 or 3")

    def _seat(self, name: str, slot_name: str) -> None:
        slot = self.slots[slot_name]
        part = self.parts.get(name) or self._add_part(name, slot.xy)
        part.pos = np.array([slot.xy[0], slot.xy[1], slot.seat_z])
        part.seated_in, part.fixed = slot_name, True
        slot.filled_by = name

    # ------------------------------------------------------------ execution

    def execute(self, actions: dict, rows: int = ACTION_ROWS - 1) -> None:
        """Play rows 0..30 of a chunk (row 31 repeats row 30)."""
        for i in range(rows):
            for arm in ARMS:
                self._step_arm(arm, actions[arm]["position"][i], actions[arm]["rotation_rpy"][i],
                               float(actions[arm]["gripper"][i]))

    def _step_arm(self, arm_name: str, target, rpy, grip_cmd: float) -> None:
        arm = self.arms[arm_name]
        target = np.asarray(target, dtype=float).copy()
        target[2] = max(target[2], 0.002)  # cannot go through the table
        held = self.parts.get(arm.holding) if arm.holding else None

        if held is not None:
            target = self._constrain_held(arm_name, held, target)
            if arm.holding is None:  # it just got seated: the slot holds it now
                held = None
        arm.pos, arm.rpy = target, np.asarray(rpy, dtype=float)

        opening = GRIPPER.opening_mm(grip_cmd)
        if held is not None:
            if opening > held.grasp_width_mm + 2.0:
                self._release(arm_name, held)
                arm.gripper = grip_cmd
            else:
                # Fingers are blocked by the part: the reading stalls at its width.
                arm.gripper = GRIPPER.command_for_opening(held.grasp_width_mm)
                held.pos = np.array([arm.pos[0], arm.pos[1], arm.pos[2] - held.size[2] / 2])
            return

        part = self._graspable_part_near(arm.pos)
        if part is not None and opening <= part.grasp_width_mm and arm.gripper > grip_cmd:
            arm.holding, part.held_by = part.name, arm_name
            arm.gripper = GRIPPER.command_for_opening(part.grasp_width_mm)
            self.log(f"{arm_name} grasped {part.name}")
        else:
            arm.gripper = grip_cmd

    def _graspable_part_near(self, tip: np.ndarray) -> SimPart | None:
        for part in self.parts.values():
            if part.fixed or part.held_by:
                continue
            xy_err = np.linalg.norm(tip[:2] - part.pos[:2])
            in_height = part.pos[2] + 0.002 <= tip[2] <= part.top_z + 0.004
            if xy_err <= 0.008 and in_height:  # fingers must straddle the part
                return part
        return None

    def _constrain_held(self, arm_name: str, part: SimPart, target: np.ndarray) -> np.ndarray:
        """Pressing a held part down onto a slot: seat it or jam on top."""
        for slot in self.slots.values():
            if part.name not in slot.accepts or slot.filled_by:
                continue
            if slot.requires and not self.slots[slot.requires].filled_by:
                continue
            xy_err = float(np.linalg.norm(target[:2] - slot.xy))
            bottom = target[2] - part.size[2] / 2
            if xy_err > 0.03 or bottom > slot.seat_z + 0.02:
                continue  # not near this slot
            if xy_err <= slot.tol_m and bottom <= slot.seat_z:
                part.pos = np.array([slot.xy[0], slot.xy[1], slot.seat_z])
                part.seated_in, part.fixed, part.held_by = slot.name, True, None
                slot.filled_by = part.name
                self.arms[arm_name].holding = None
                self.log(f"{part.name} seated in {slot.name} (xy error {xy_err*1000:.1f} mm)")
            elif bottom <= slot.seat_z + 0.01 and xy_err > slot.tol_m:
                target[2] = slot.seat_z + 0.01 + part.size[2] / 2  # jams on the rim
                self.log(f"{part.name} jammed on {slot.name} rim (off by > {slot.tol_m*1000:.0f} mm)")
        return target

    def _release(self, arm_name: str, part: SimPart) -> None:
        self.arms[arm_name].holding = None
        part.held_by = None
        if part.seated_in is None:
            part.pos = np.array([part.pos[0], part.pos[1], 0.0])  # falls to the table
            self.log(f"{part.name} released on the table")

    # ------------------------------------------------------------ queries

    def state(self) -> dict:
        """public_v2 observation.state for both arms."""
        return {
            arm: {
                "position": self.arms[arm].pos.round(5).tolist(),
                "rotation_rpy": self.arms[arm].rpy.round(5).tolist(),
                "gripper": round(float(self.arms[arm].gripper), 5),
            }
            for arm in ARMS
        }

    def is_held(self, name: str) -> bool:
        part = self.parts.get(name)
        return bool(part and part.held_by and part.pos[2] > 0.01)

    def held_above(self, name: str, slot_names: list[str], radius: float) -> bool:
        part = self.parts.get(name)
        if not part or not part.held_by:
            return False
        for s in slot_names:
            slot = self.slots[s]
            if slot.filled_by:
                continue
            if np.linalg.norm(part.pos[:2] - slot.xy) <= radius and part.pos[2] - slot.seat_z < 0.06:
                return True
        return False

    def seated(self, name: str) -> bool:
        part = self.parts.get(name)
        return bool(part and part.seated_in)

    def arms_clear(self) -> bool:
        return all(a.holding is None and a.gripper > GRIPPER.max_command * 0.6 for a in self.arms.values())

    def free_slot_for(self, name: str) -> Slot | None:
        for slot in self.slots.values():
            if name in slot.accepts and not slot.filled_by and (not slot.requires or self.slots[slot.requires].filled_by):
                return slot
        return None


# Ground-truth answers for each stage id used in config/tasks/*.yaml.
# The fake vision model uses these; a real one has to look at images.
STAGE_TRUTH = {
    "grasp_slider": lambda w: w.is_held("slider"),
    "align_slider": lambda w: w.held_above("slider", ["handle_slot"], 0.03),
    # push_*: the sim seats a part the moment a press lands inside the slot
    # tolerance, so "pushed down onto it" is true exactly when it is seated.
    # (Real push questions are looser: contact, no gap, hole hidden by the gripper.)
    "push_slider": lambda w: w.seated("slider"),
    "release_slider": lambda w: w.arms_clear(),
    "seated_slider": lambda w: w.seated("slider"),
    "grasp_shell": lambda w: w.is_held("shell"),
    "align_shell": lambda w: w.held_above("shell", ["shell_seat"], 0.03),
    "push_shell": lambda w: w.seated("shell"),
    "release_shell": lambda w: w.arms_clear(),
    "seated_shell": lambda w: w.seated("shell"),
    "grasp_teal": lambda w: w.is_held("button_a"),
    "align_teal": lambda w: w.held_above("button_a", ["hole_a", "hole_b"], 0.02),
    "push_teal": lambda w: w.seated("button_a"),
    "release_teal": lambda w: w.arms_clear(),
    "seated_teal": lambda w: w.seated("button_a"),
    "grasp_purple": lambda w: w.is_held("button_b"),
    "align_purple": lambda w: w.held_above("button_b", ["hole_a", "hole_b"], 0.02),
    "push_purple": lambda w: w.seated("button_b"),
    "release_purple": lambda w: w.arms_clear(),
    "seated_both": lambda w: w.seated("button_a") and w.seated("button_b"),
}
