"""Check every Wall-X action chunk before the robot executes it.

Wall-X returns 32 absolute end-effector poses per arm. The robot client
executes them without any collision checking we know of. This module is
the last software gate between the model and the hardware.

Checks (each one produces a readable violation string):
1. Shape, finiteness, embodiment echo (protocol.validate_response).
2. Row 0 is not a big jump away from where the arm actually is.
3. Consecutive rows do not move faster than a sane limit.
4. Every row stays inside a per-arm workspace box (measure on site).
5. Gripper values stay inside the documented range.

Modes (config filter.mode):
* off      -> forward the original bytes, no checks.
* log      -> check, record violations, still forward unchanged.
* enforce  -> on any violation, the relay sends HOLD instead.

Optionally freeze the idle arm: Wall-X always moves BOTH arms. When a stage
declares an active arm, the other arm can be overwritten with HOLD rows so
it stops wandering. Whether organizers allow modifying actions is an open
question: ask before enabling enforce or freeze_idle_arm in official runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import FilterConfig
from .protocol import ARMS, GRIPPER_RANGE, ArmState, ProtocolError, validate_response


@dataclass
class FilterResult:
    ok: bool
    violations: list[str] = field(default_factory=list)
    # The actions to send if we modified them (freeze idle arm); None = unchanged.
    modified_actions: dict | None = None
    # Summary numbers for the recorder (how far/fast this chunk wants to go).
    stats: dict = field(default_factory=dict)


class ActionFilter:
    def __init__(self, cfg: FilterConfig):
        self.cfg = cfg

    def check(
        self,
        response: dict,
        embodiment: str,
        state: dict[str, ArmState],
        active_arm: str | None = None,
    ) -> FilterResult:
        if self.cfg.mode == "off":
            return FilterResult(ok=True)

        try:
            actions = validate_response(response, embodiment)
        except ProtocolError as err:
            # A malformed chunk is never safe to forward, in any mode.
            return FilterResult(ok=False, violations=[f"protocol: {err}"])

        violations: list[str] = []
        stats: dict = {}
        lo_g, hi_g = GRIPPER_RANGE[embodiment]

        for arm in ARMS:
            pos = actions[arm]["position"]
            rot = actions[arm]["rotation_rpy"]
            grip = actions[arm]["gripper"]

            first_jump = float(np.linalg.norm(pos[0] - state[arm].position))
            steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
            rot_steps = rotation_steps(rot)
            travel = float(np.linalg.norm(pos[-1] - state[arm].position))
            stats[arm] = {
                "first_jump_m": round(first_jump, 4),
                "max_step_m": round(float(steps.max()), 4),
                "travel_m": round(travel, 4),
                "gripper_end": round(float(grip[-1]), 3),
            }

            if first_jump > self.cfg.max_first_row_jump_m:
                violations.append(f"{arm}: row 0 is {first_jump:.3f} m from current pose")
            if steps.max() > self.cfg.max_row_step_m:
                violations.append(f"{arm}: step {steps.max():.3f} m between rows exceeds limit")
            if rot_steps.max() > self.cfg.max_row_rot_step_rad:
                violations.append(f"{arm}: rotation step {rot_steps.max():.3f} rad exceeds limit")

            box = self.cfg.workspace.get(arm)
            if box is not None:
                below = pos < np.asarray(box.min)
                above = pos > np.asarray(box.max)
                if below.any() or above.any():
                    rows = sorted(set(np.where(below | above)[0].tolist()))
                    violations.append(f"{arm}: rows {rows[:5]} leave the workspace box")

            if grip.min() < lo_g - 1e-3 or grip.max() > hi_g + 1e-3:
                violations.append(
                    f"{arm}: gripper {grip.min():.3f}..{grip.max():.3f} outside [{lo_g}, {hi_g}]"
                )

        modified = None
        if self.cfg.freeze_idle_arm and active_arm in ARMS:
            idle = "left_ee" if active_arm == "right_ee" else "right_ee"
            modified = {arm: dict(actions[arm]) for arm in ARMS}
            modified[idle] = {
                "position": np.tile(state[idle].position, (len(actions[idle]["position"]), 1)),
                "rotation_rpy": np.tile(state[idle].rotation_rpy, (len(actions[idle]["position"]), 1)),
                "gripper": np.full(len(actions[idle]["gripper"]), state[idle].gripper),
            }

        return FilterResult(ok=not violations, violations=violations, modified_actions=modified, stats=stats)


def rpy_to_matrices(rpy: np.ndarray) -> np.ndarray:
    """Rotation matrices for rows of (roll, pitch, yaw), R = Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = np.asarray(rpy, dtype=float).T
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.stack(
        [
            np.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], axis=-1),
            np.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], axis=-1),
            np.stack([-sp, cp * sr, cp * cr], axis=-1),
        ],
        axis=-2,
    )


def rotation_steps(rpy: np.ndarray) -> np.ndarray:
    """Geodesic angle (rad) between consecutive rows of RPY angles.

    Differencing roll, pitch and yaw separately is wrong near pitch = +-90
    degrees (gimbal lock): roll and yaw swing wildly while the orientation
    barely moves. The angle of the relative rotation matrix has no such trap.
    """
    mats = rpy_to_matrices(rpy)
    rel = np.einsum("nji,njk->nik", mats[:-1], mats[1:])  # R_i^T R_{i+1}
    cos_angle = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    skew = np.stack([rel[:, 2, 1] - rel[:, 1, 2], rel[:, 0, 2] - rel[:, 2, 0], rel[:, 1, 0] - rel[:, 0, 1]], axis=1)
    return np.arctan2(np.linalg.norm(skew, axis=1) / 2.0, cos_angle)  # stable for tiny and ~pi angles
