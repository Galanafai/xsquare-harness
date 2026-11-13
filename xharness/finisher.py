"""Training-free final alignment and press controller (the "finisher").

Wall-X is random (65 mm between calls on the same input, docs/WALLX_EXPERIMENTS.md)
so it cannot be asked for a 0.5 mm press. The finisher replaces it for the last
centimeter only:

  1. DETECT   held part (HSV color) and target hole or slot (round dark blob)
              in the wrist camera of the arm that holds the part.
  2. SERVO    look-then-move, no calibration: one ~3 mm exploratory move per
              axis gives a 2x2 image Jacobian (refined by Broyden updates),
              then a few corrections (gain 0.6, each axis capped at 3 mm).
              z and rotation stay frozen. Aborts if the error grows twice
              in a row or detection confidence drops.
  3. PRESS    to an ABSOLUTE target, the stage's seated_z minus a small overtravel:
              fast steps until 3 mm above seated_z, then 1 mm steps. Each target is
              sent twice and the lag (measured z above commanded z) is read on the
              second, so a partly executed chunk is not mistaken for resistance.
              Resistance far above seated_z is a JAM: lift 10 mm, keep the part
              gripped, abort back to align. Resistance in the last couple of mm is
              the snap, so the press continues to the target. Without seated_z the
              finisher aborts before moving: it never presses blind.
  4. RELEASE  open the gripper, then LIFT 3 cm. The supervisor then runs its
              normal seated vision check.

Safety rules:
* Every chunk is checked by action_filter.py (forced to `enforce`, whatever the
  global filter mode). A violation aborts the finisher; the relay answers HOLD.
* Any doubt (low confidence, arm not moving, bad Jacobian) aborts. The relay
  then keeps the arms still; the supervisor decides about retries.
* All numbers in FinisherConfig are PLACEHOLDERS to measure on site.
* OFF by default (finisher.enabled). It is harness-generated arm motion, which
  the organizers approved on Nov 12 (docs/ORGANIZER_QUESTIONS.md #13). Off until
  seated_z and the other placeholders are measured on site.

The wrist camera is assumed to look along the gripper axis, so a part held in
the fingers sits at a fixed place in the image and the hole moves as the arm
moves. `target_offset_px` is the image offset at which an aligned hole appears
(MEASURE ON SITE). Known real-world risk: the held part may hide the hole once
aligned; the sim draws holes on top of parts, a simplification.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field, replace

import numpy as np
from PIL import Image

from .action_filter import ActionFilter
from .config import FilterConfig, FinisherConfig
from .parts import PARTS
from .protocol import ARMS, ACTION_ROWS, build_response, hold_actions

# Hue ranges in degrees for the planned part colors (xharness/parts.py).
# PLACEHOLDERS: update to match the real prints and lighting.
HUE_RANGES = {
    "orange": (10.0, 45.0),
    "green": (85.0, 150.0),
    "blue": (195.0, 240.0),
    "purple": (255.0, 305.0),
}
MIN_SATURATION, MIN_VALUE = 0.35, 0.30
WRIST = {"left_ee": "left_wrist", "right_ee": "right_wrist"}


# --------------------------------------------------------------- detection


@dataclass
class Detection:
    part_px: tuple[float, float] | None = None
    target_px: tuple[float, float] | None = None
    confidence: float = 0.0
    part_width_px: float = 0.0
    reason: str = ""


def decode_rgb(image_b64: str) -> np.ndarray:
    if image_b64.startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(image_b64))) as img:
        return np.asarray(img.convert("RGB"))


def components(mask: np.ndarray, limit: int = 60000) -> list[np.ndarray]:
    """4-connected components of a boolean mask, as arrays of (row, col)."""
    todo = mask.copy()
    out = []
    h, w = mask.shape
    for r0, c0 in zip(*np.nonzero(todo)):
        if not todo[r0, c0]:
            continue
        stack, pts = [(r0, c0)], []
        todo[r0, c0] = False
        while stack and len(pts) < limit:
            r, c = stack.pop()
            pts.append((r, c))
            for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= rr < h and 0 <= cc < w and todo[rr, cc]:
                    todo[rr, cc] = False
                    stack.append((rr, cc))
        out.append(np.asarray(pts))
    return out


def _part_blob(rgb: np.ndarray, color: str) -> tuple[tuple[float, float], float, float] | None:
    """Center (bounding-box center of the part-colored pixels, x y), area, min side in px."""
    lo, hi = HUE_RANGES[color]
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=float)
    hue = hsv[..., 0] * 360.0 / 255.0
    mask = (hue >= lo) & (hue <= hi) & (hsv[..., 1] >= MIN_SATURATION * 255) & (hsv[..., 2] >= MIN_VALUE * 255)
    blobs = [b for b in components(mask) if len(b) >= 8]
    if not blobs:
        return None
    pts = max(blobs, key=len)
    (r0, c0), (r1, c1) = pts.min(axis=0), pts.max(axis=0)
    return ((c0 + c1) / 2.0, (r0 + r1) / 2.0), float(len(pts)), float(min(r1 - r0 + 1, c1 - c0 + 1))


def _round_dark_blobs(rgb: np.ndarray, value_max: float) -> list[tuple[tuple[float, float], float, float]]:
    """Round dark blobs: (centroid x y, area, roundness 0..1). Shape, not just darkness:
    a filled circle covers pi/4 of its bounding box with an aspect ratio near 1."""
    value = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=float)[..., 2] / 255.0
    out = []
    for pts in components(value < value_max):
        if len(pts) < 12:
            continue
        (r0, c0), (r1, c1) = pts.min(axis=0), pts.max(axis=0)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        fill = len(pts) / float(h * w)
        aspect = min(h, w) / float(max(h, w))
        roundness = max(0.0, 1.0 - abs(fill - np.pi / 4) / 0.25) * max(0.0, 1.0 - (1.0 - aspect) / 0.4)
        if roundness > 0.0:
            cy, cx = pts.mean(axis=0)
            out.append(((float(cx), float(cy)), float(len(pts)), float(roundness)))
    return out


def detect(image_b64: str, color: str, cfg: FinisherConfig) -> Detection:
    """Find the held part and the nearest round dark target in one wrist image."""
    rgb = decode_rgb(image_b64)
    part = _part_blob(rgb, color)
    if part is None:
        return Detection(reason="part not found")
    part_px, area, width_px = part
    blobs = _round_dark_blobs(rgb, cfg.dark_value_max)
    if not blobs:
        return Detection(part_px=part_px, part_width_px=width_px, reason="no round dark target")
    # Nearest to the part: with two holes (case 3) either empty one is acceptable.
    target_px, _, roundness = min(blobs, key=lambda b: np.hypot(b[0][0] - part_px[0], b[0][1] - part_px[1]))
    confidence = min(roundness, min(1.0, area / 20.0))
    return Detection(part_px, target_px, confidence, width_px)


# ---------------------------------------------------------------- controller


@dataclass
class StepResult:
    status: str  # running | done | aborted
    actions: dict | None = None
    reason: str = ""
    info: dict = field(default_factory=dict)  # for the recorder


class Finisher:
    """One finishing attempt for one held part. Create a new one per attempt."""

    def __init__(self, cfg: FinisherConfig, filter_cfg: FilterConfig, part: str, arm: str | None = None,
                 seated_z: float | None = None):
        self.cfg = cfg
        self.seated_z = seated_z  # metres, gripper z when fully seated (MEASURE ON SITE)
        # Our own chunks are ALWAYS enforced, whatever the global filter mode.
        self.filter = ActionFilter(replace(filter_cfg, mode="enforce", freeze_idle_arm=False))
        self.part_key = part
        self.color = PARTS[part].color
        self.width_mm = PARTS[part].top_width_mm  # what the wrist camera measures across the part
        self.arm = arm
        self.phase = "look"
        self.J: np.ndarray | None = None  # px per mm, 2x2
        self.prev_e: np.ndarray | None = None
        self.prev_pos: np.ndarray | None = None
        self.prev_norm: float | None = None
        self.grow = 0
        self.converged = 0
        self.corrections = 0
        self.pending: dict | None = None  # press target in flight: {cmd, sends}
        self.jam_z0 = 0.0
        self.release_requests = self.open_count = 0
        self.lift_z0 = 0.0
        self.last_move = False

    # -- public --------------------------------------------------------------

    def step(self, request: dict, state: dict, grip_by_arm: dict | None = None) -> StepResult:
        """One request. `grip_by_arm` = last gripper values Wall-X commanded (keeps the part gripped)."""
        try:
            result = self._step(request, state, grip_by_arm or {})
        except Exception as err:  # noqa: BLE001 - a finisher bug must end in HOLD, never in a crash
            return StepResult("aborted", reason=f"finisher error: {type(err).__name__}: {err}")
        if result.actions is not None:
            violations = self._violations(result.actions, request, state)
            if violations:
                return StepResult("aborted", reason="action filter: " + "; ".join(violations), info=result.info)
        return result

    # -- phases --------------------------------------------------------------

    def _step(self, request: dict, state: dict, grip_by_arm: dict) -> StepResult:
        if self.seated_z is None:
            return self._abort("seated_z is not set for this stage: refusing to press blind (measure it on site)")
        if self.arm is None:
            self.arm = self._pick_arm(request)
            if self.arm is None:
                return self._abort("part not visible in either wrist camera")
        pos = np.asarray(state[self.arm].position, float)
        grip = float(grip_by_arm.get(self.arm, self.cfg.grip_command))
        if self.phase in ("look", "explore_x", "explore_y", "servo"):
            return self._vision_step(request, state, pos, grip)
        if self.phase == "press":
            return self._press_step(state, pos, grip)
        if self.phase == "jam_lift":
            return self._jam_lift_step(state, pos, grip)
        if self.phase == "release":
            return self._release_step(state, pos)
        return self._lift_step(state, pos)

    def _vision_step(self, request, state, pos, grip) -> StepResult:
        det = detect(request["observation"]["images"][WRIST[self.arm]], self.color, self.cfg)
        if det.target_px is None or det.confidence < self.cfg.min_confidence:
            return self._abort(f"detection lost ({det.reason or 'confidence %.2f' % det.confidence})")
        e = np.asarray(det.target_px, float) - np.asarray(det.part_px, float) - np.asarray(self.cfg.target_offset_px, float)
        info = {"phase": self.phase, "e_px": e.round(2).tolist(), "confidence": round(det.confidence, 2)}
        explore = self.cfg.explore_mm

        if self.phase == "look":
            self.phase = "explore_x"
            self._remember(e, pos)
            return self._move(state, pos, [explore, 0.0], grip, info)

        moved = (pos[:2] - self.prev_pos[:2]) * 1000.0  # measured, mm
        d_e = e - self.prev_e
        if self.phase == "explore_x":
            if abs(moved[0]) < 0.5 * explore:
                return self._abort("arm did not follow the exploratory x move")
            self.J = np.zeros((2, 2))
            self.J[:, 0] = d_e / moved[0]
            self.phase = "explore_y"
            self._remember(e, pos)
            return self._move(state, pos, [0.0, explore], grip, info)
        if self.phase == "explore_y":
            if abs(moved[1]) < 0.5 * explore:
                return self._abort("arm did not follow the exploratory y move")
            self.J[:, 1] = d_e / moved[1]
            bad = self._jacobian_problem(det)
            if bad:
                return self._abort(bad)
            self.phase = "servo"
            self.last_move = False  # the J columns already include this move
        elif self.cfg.broyden and self.last_move and float(moved @ moved) > 0.04:
            self.J = self.J + np.outer(d_e - self.J @ moved, moved) / float(moved @ moved)
            bad = self._jacobian_problem(det, check_scale=False)
            if bad:
                return self._abort(bad)

        norm = float(np.linalg.norm(e))
        if self.prev_norm is not None and self.last_move:
            self.grow = self.grow + 1 if norm > self.prev_norm + 0.3 else 0
            if self.grow >= 2:
                return self._abort(f"image error grew twice in a row ({self.prev_norm:.1f} -> {norm:.1f} px)")
        err_mm = np.linalg.solve(self.J, e)
        info.update(e_mm=round(float(np.linalg.norm(err_mm)), 2), corrections=self.corrections)
        self.converged = self.converged + 1 if np.linalg.norm(err_mm) < self.cfg.converge_mm else 0
        if self.converged >= 2:
            self.phase = "press"
            return self._press_step(state, pos, grip)
        if self.corrections >= self.cfg.max_corrections and self.converged == 0:
            return self._abort(f"not converged after {self.corrections} corrections ({np.linalg.norm(err_mm):.2f} mm)")
        move = np.clip(-self.cfg.gain * err_mm, -self.cfg.max_step_mm, self.cfg.max_step_mm)
        self.corrections += 1
        self._remember(e, pos)
        self.prev_norm = norm
        self.last_move = True
        return self._move(state, pos, move.tolist(), grip, info, count=False)

    def _press_step(self, state, pos, grip) -> StepResult:
        cfg, z = self.cfg, float(pos[2])
        seated = float(self.seated_z)
        target = seated - cfg.overtravel_mm / 1000.0
        floor = target - cfg.hard_floor_mm / 1000.0  # absolute
        info = {"phase": "press", "above_seated_mm": round((z - seated) * 1000, 2)}
        if z < floor - 1e-6:
            return self._abort(f"arm is below the hard floor ({(z - floor) * 1000:.1f} mm): seated_z is wrong")
        pend = self.pending
        if pend and pend["sends"] == 1:  # settle: send the same target again, read the lag after it
            pend["sends"] = 2
            return StepResult("running", self._chunk(state, np.array([pos[0], pos[1], pend["cmd"]]), grip), info=info)
        if pend:  # second send executed: z is settled against the command
            lag = (z - pend["cmd"]) * 1000.0
            above = (z - seated) * 1000.0
            info.update(lag_mm=round(lag, 2))
            at_target = pend["cmd"] <= target + 1e-9
            if at_target and above <= cfg.jam_zone_mm:
                return self._start_release(state, pos, info, "seated")
            if (at_target or lag > cfg.contact_margin_mm) and above > cfg.jam_zone_mm:
                self.phase, self.jam_z0 = "jam_lift", z
                info["jam"] = f"resistance {above:.1f} mm above seated_z"
                return self._jam_lift_step(state, pos, grip, info)
        cur = pend["cmd"] if pend else z
        if cur - seated > cfg.slow_zone_mm / 1000.0 + 1e-6:
            nxt = max(cur - cfg.fast_step_mm / 1000.0, seated + cfg.slow_zone_mm / 1000.0)
        else:
            nxt = max(cur - cfg.press_step_mm / 1000.0, target)
        self.pending = {"cmd": max(nxt, floor), "sends": 1}
        return StepResult("running", self._chunk(state, np.array([pos[0], pos[1], self.pending["cmd"]]), grip), info=info)

    def _jam_lift_step(self, state, pos, grip, info=None) -> StepResult:
        """A jam: lift a little with the part still gripped, then abort so the supervisor re-aligns."""
        info = info or {"phase": "jam_lift"}
        lifted = (float(pos[2]) - self.jam_z0) * 1000.0
        if lifted >= self.cfg.jam_lift_mm - 0.5:
            return self._abort(f"jam above seated_z; lifted {lifted:.1f} mm, part still gripped")
        top = self.jam_z0 + self.cfg.jam_lift_mm / 1000.0
        target = np.array([pos[0], pos[1], min(pos[2] + self.cfg.lift_step_mm / 1000.0, top)])
        return StepResult("running", self._chunk(state, target, grip), info=info)

    def _start_release(self, state, pos, info, why) -> StepResult:
        self.phase = "release"
        info["pressed_until"] = why
        return self._release_step(state, pos, info)

    def _release_step(self, state, pos, info=None) -> StepResult:
        info = info or {}
        info["phase"] = "release"
        self.release_requests += 1
        reading = float(state[self.arm].gripper)
        self.open_count = self.open_count + 1 if reading >= self.cfg.open_above else 0
        if self.open_count >= 2:
            self.phase, self.lift_z0 = "lift", float(pos[2])
            return self._lift_step(state, pos)
        if self.release_requests > 10:
            return self._abort("gripper did not open")
        return StepResult("running", self._chunk(state, pos, self.cfg.open_command), info=info)

    def _lift_step(self, state, pos) -> StepResult:
        lifted = (float(pos[2]) - self.lift_z0) * 1000.0
        if lifted >= self.cfg.lift_mm - 0.5:
            return StepResult("done", info={"phase": "lift", "lifted_mm": round(lifted, 1)})
        target = np.array([pos[0], pos[1], min(pos[2] + self.cfg.lift_step_mm / 1000.0, self.lift_z0 + self.cfg.lift_mm / 1000.0)])
        return StepResult("running", self._chunk(state, target, self.cfg.open_command), info={"phase": "lift", "lifted_mm": round(lifted, 1)})

    # -- helpers -------------------------------------------------------------

    def _pick_arm(self, request: dict) -> str | None:
        """The arm whose wrist camera sees the biggest blob of the part color."""
        best, best_area = None, 0.0
        for arm in ARMS:
            blob = _part_blob(decode_rgb(request["observation"]["images"][WRIST[arm]]), self.color)
            if blob is not None and blob[1] > best_area:
                best, best_area = arm, blob[1]
        return best

    def _remember(self, e: np.ndarray, pos: np.ndarray) -> None:
        self.prev_e, self.prev_pos = e.copy(), pos.copy()

    def _jacobian_problem(self, det: Detection, check_scale: bool = True) -> str:
        J = self.J
        if not np.all(np.isfinite(J)) or abs(np.linalg.det(J)) < 1e-3 or np.linalg.cond(J) > 50.0:
            return f"image Jacobian unusable (cond {np.linalg.cond(J):.0f})"
        if check_scale and det.part_width_px > 0:
            expected = det.part_width_px / self.width_mm  # px per mm at the part's depth
            for col in np.linalg.norm(J, axis=0):
                ratio = col / expected
                if not (1.0 / self.cfg.scale_tolerance <= ratio <= self.cfg.scale_tolerance):
                    return f"Jacobian scale {col:.2f} px/mm does not match part width ({expected:.2f} px/mm)"
        return ""

    def _move(self, state, pos, delta_mm, grip, info, count=True) -> StepResult:
        target = np.array([pos[0] + delta_mm[0] / 1000.0, pos[1] + delta_mm[1] / 1000.0, pos[2]])
        info["move_mm"] = [round(float(v), 2) for v in delta_mm]
        return StepResult("running", self._chunk(state, target, grip), info=info)

    def _chunk(self, state, target: np.ndarray, grip: float) -> dict:
        """32 rows: the idle arm holds; the active arm ramps to `target`, rotation frozen."""
        actions = hold_actions(state)
        a = actions[self.arm]
        start = np.asarray(state[self.arm].position, float)
        t = np.clip(np.arange(ACTION_ROWS) / max(1, self.cfg.ramp_rows), 0.0, 1.0)[:, None]
        a["position"] = start + t * (target - start)
        a["gripper"] = np.full(ACTION_ROWS, float(grip))
        return actions

    def _violations(self, actions: dict, request: dict, state: dict) -> list[str]:
        response = build_response(request["embodiment"], actions)
        return self.filter.check(response, request["embodiment"], state, self.arm).violations

    def _abort(self, reason: str) -> StepResult:
        self.phase = "aborted"
        return StepResult("aborted", reason=reason)
