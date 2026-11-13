"""The WALLX public_v2 wire protocol, in one place.

Everything that touches the wire (the robot client, Wall-X, our relay, the
fake robot, the probe tool) imports from here, so the contract lives in a
single file.

Source of truth: "API Tutorial & User Guide 2610" and the Hackathon
Participant Guide. Anything marked UNVERIFIED below has not yet been checked
against the live endpoint; run tools/probe_wallx.py to confirm it.

Wire summary
------------
* WebSocket, binary frames, MessagePack + msgpack-numpy.
* Server sends one handshake (metadata dict) first.
* Then strictly: client sends one request, server sends one response.
* Request:  {embodiment, normalization, instruction, observation, [control_context]}
* Response: {embodiment, actions: {left_ee, right_ee}} or {error: ...}
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import msgpack
import msgpack_numpy
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARMS = ("left_ee", "right_ee")
CAMERAS = ("front", "left_wrist", "right_wrist")
POSE_FIELDS = ("position", "rotation_rpy")

# Number of rows Wall-X returns per arm. Row 31 repeats row 30.
ACTION_ROWS = 32

# Nominal time between model rows (API Tutorial). The sync client may play
# rows at a different rate (control_dt * interpolate_multiplier = 40 ms).
MODEL_ROW_DT_S = 0.0337

# Gripper command range per gripper variant (API Tutorial). The unit is very
# likely the drive-joint angle in radians, NOT millimetres (see X2Real
# gripper_convention.py). Opening in mm must be measured on the robot.
GRIPPER_RANGE = {
    "quanta_x1_pro": (0.0, 1.526),
    "quanta_x1_pro/gripper-g": (0.0, 1.526),
    "quanta_x1_pro/gripper-h": (0.0, 5.5),
}
EMBODIMENTS = tuple(GRIPPER_RANGE)

# Fields the sync harness must never send (they belong to RTC mode).
RTC_FIELDS = ("use_rtc", "prev_action_chunk", "inference_delay")

# Largest frame we accept (three JPEGs fit easily).
MAX_FRAME_BYTES = 32 * 1024 * 1024

# Keys of the optional `control_context` request field, per arm. Read from the
# live handshake on 2026-11-12. Each value is a number, or null = masked.
# Omitting the whole field makes the server use its startup context (6 of 7
# slots set per arm); sending all-null sets 0 slots. The relay forwards the
# client's field untouched and never judges it.
CONTROL_CONTEXT_PARAMS = (
    "eff_delay_ms", "lpf_T_ms", "grip_delay_ms", "grip_gain",
    "grip_offset", "z_offset_mm", "rot_offset_deg",
)

# Error codes documented in the API Tutorial.
KNOWN_ERROR_CODES = (
    "unknown_embodiment",
    "missing_field",
    "image_decode_failed",
    "mixed_schema",
    "normalization_conflict",
    "rtc_delay_exceeds_training",
)


class ProtocolError(ValueError):
    """A message does not match the public_v2 contract."""


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def pack(value: Any) -> bytes:
    """Encode a Python object as a binary MessagePack frame (numpy-aware)."""
    return msgpack.packb(value, default=msgpack_numpy.encode, use_bin_type=True)


def unpack(frame: Any) -> Any:
    """Decode a binary frame. Text frames are a protocol violation.

    The open-source Wall-X server sends a *text* traceback just before
    closing with code 1011, so a str here usually means "server crashed".
    """
    if isinstance(frame, str):
        raise ProtocolError(f"text frame received (server crash?): {frame[:200]!r}")
    if not isinstance(frame, (bytes, bytearray)):
        raise ProtocolError(f"expected binary frame, got {type(frame).__name__}")
    return msgpack.unpackb(frame, raw=False, object_hook=msgpack_numpy.decode)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def as_vec(value: Any, length: int, name: str) -> np.ndarray:
    """Convert to a finite float vector of a fixed length or raise."""
    arr = np.asarray(value, dtype=float)
    if arr.shape != (length,):
        raise ProtocolError(f"{name}: expected shape ({length},), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ProtocolError(f"{name}: contains NaN or inf")
    return arr


def as_scalar(value: Any, name: str) -> float:
    arr = np.asarray(value, dtype=float)
    if arr.shape != ():
        raise ProtocolError(f"{name}: expected scalar, got shape {arr.shape}")
    if not math.isfinite(float(arr)):
        raise ProtocolError(f"{name}: not finite")
    return float(arr)


@dataclass(frozen=True)
class ArmState:
    """One arm's measured state: absolute pose in the robot base frame."""

    position: np.ndarray  # metres, shape (3,)
    rotation_rpy: np.ndarray  # radians, shape (3,), R = Rz(yaw) Ry(pitch) Rx(roll)
    gripper: float  # gripper units (see GRIPPER_RANGE)


def read_state(observation_request: dict) -> dict[str, ArmState]:
    """Pull both arms' state out of a request. Raises ProtocolError."""
    state = observation_request["observation"]["state"]
    out = {}
    for arm in ARMS:
        s = state[arm]
        out[arm] = ArmState(
            position=as_vec(s["position"], 3, f"{arm}.position"),
            rotation_rpy=as_vec(s["rotation_rpy"], 3, f"{arm}.rotation_rpy"),
            gripper=as_scalar(s["gripper"], f"{arm}.gripper"),
        )
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_request(request: Any) -> None:
    """Check a client->model request. Raises ProtocolError with a clear reason."""
    if not isinstance(request, dict):
        raise ProtocolError("request is not a dict")
    if request.get("normalization") != "provided":
        raise ProtocolError("normalization must be 'provided'")
    if request.get("embodiment") not in EMBODIMENTS:
        raise ProtocolError(f"unsupported embodiment {request.get('embodiment')!r}")
    for field in RTC_FIELDS:
        if field in request:
            raise ProtocolError(f"sync harness must not send RTC field {field!r}")
    instruction = request.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ProtocolError("instruction must be a non-empty string")
    images = request.get("observation", {}).get("images", {})
    for cam in CAMERAS:
        if not isinstance(images.get(cam), str) or not images[cam]:
            raise ProtocolError(f"missing JPEG base64 image {cam!r}")
    read_state(request)  # raises on bad state


def validate_control_context(request: dict) -> None:
    """Mimic the real server's check of the optional control_context field.

    Used by the fake Wall-X only. NOT called by the relay: that field belongs
    to the organizers' client and only the real server should reject it.
    """
    ctx = request.get("control_context")
    if ctx is None:
        return
    if not isinstance(ctx, dict):
        raise ProtocolError("control_context must be an object")
    for arm in ("left", "right"):
        block = ctx.get(arm)
        if not isinstance(block, dict):
            raise ProtocolError(f"control_context.{arm} must be an object")
        if set(block) != set(CONTROL_CONTEXT_PARAMS):
            raise ProtocolError(f"control_context.{arm} must contain exactly {list(CONTROL_CONTEXT_PARAMS)}")


def is_error_response(response: Any) -> bool:
    return isinstance(response, dict) and "error" in response


def error_code(response: dict) -> str:
    """The error may be a string or {code, message} (API Tutorial)."""
    err = response.get("error")
    if isinstance(err, dict):
        return str(err.get("code", "unknown"))
    return str(err)


def validate_response(response: Any, embodiment: str) -> dict[str, dict[str, np.ndarray]]:
    """Check a model->client action response and return it as numpy arrays.

    Returns {arm: {"position": (32,3), "rotation_rpy": (32,3), "gripper": (32,)}}.
    """
    if not isinstance(response, dict):
        raise ProtocolError("response is not a dict")
    if is_error_response(response):
        raise ProtocolError(f"error response: {error_code(response)}")
    if response.get("embodiment") != embodiment:
        raise ProtocolError(f"embodiment echo mismatch: {response.get('embodiment')!r} != {embodiment!r}")
    actions = response.get("actions")
    if not isinstance(actions, dict):
        raise ProtocolError("missing 'actions'")
    out: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARMS:
        block = actions.get(arm)
        if not isinstance(block, dict):
            raise ProtocolError(f"missing actions.{arm}")
        pos = np.asarray(block.get("position"), dtype=float)
        rot = np.asarray(block.get("rotation_rpy"), dtype=float)
        grip = np.asarray(block.get("gripper"), dtype=float)
        if pos.shape != (ACTION_ROWS, 3) or rot.shape != (ACTION_ROWS, 3) or grip.shape != (ACTION_ROWS,):
            raise ProtocolError(
                f"actions.{arm}: bad shapes position{pos.shape} rotation{rot.shape} gripper{grip.shape}"
            )
        for name, arr in (("position", pos), ("rotation_rpy", rot), ("gripper", grip)):
            if not np.isfinite(arr).all():
                raise ProtocolError(f"actions.{arm}.{name} contains NaN or inf")
        out[arm] = {"position": pos, "rotation_rpy": rot, "gripper": grip}
    return out


# ---------------------------------------------------------------------------
# Building responses ourselves (HOLD chunks)
# ---------------------------------------------------------------------------


def build_response(embodiment: str, actions: dict[str, dict[str, np.ndarray]]) -> dict:
    """Wrap per-arm arrays in the public_v2 response envelope.

    Matches the real service as measured by tools/probe_wallx.py (2026-11-12):
    top-level `embodiment`, `normalization: "provided"` and `actions`, with
    position and rotation_rpy as plain nested lists of floats (32 x 3) and
    gripper as a plain list (32). The real reply uses plain msgpack lists,
    NOT msgpack-numpy ext arrays, so we convert with tolist(). Wall-X's extra
    diagnostic fields (server_timing, usage, rtc, ...) are left out.
    UNVERIFIED on hardware: that the robot client accepts a reply without them.
    """
    return {
        "embodiment": embodiment,
        "normalization": "provided",
        "actions": {
            arm: {
                "position": np.asarray(actions[arm]["position"], dtype=np.float64).tolist(),
                "rotation_rpy": np.asarray(actions[arm]["rotation_rpy"], dtype=np.float64).tolist(),
                "gripper": np.asarray(actions[arm]["gripper"], dtype=np.float64).tolist(),
            }
            for arm in ARMS
        },
    }


def hold_actions(state: dict[str, ArmState]) -> dict[str, dict[str, np.ndarray]]:
    """32 rows that repeat the current measured pose: 'stay where you are'."""
    return {
        arm: {
            "position": np.tile(state[arm].position, (ACTION_ROWS, 1)),
            "rotation_rpy": np.tile(state[arm].rotation_rpy, (ACTION_ROWS, 1)),
            "gripper": np.full(ACTION_ROWS, state[arm].gripper),
        }
        for arm in ARMS
    }
