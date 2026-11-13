"""Harness configuration: YAML file + environment variables for secrets.

Secrets never live in YAML. They come from environment variables:

    WALLX_ENDPOINT        overrides wallx.endpoint
    WALLX_HEADERS_FILE    JSON file with the handshake headers (chmod 600)
    AGENT_CHAT_URL        full /v1/chat/completions URL of the vision model
    AGENT_MODEL           vision model name
    AGENT_API_KEY         key for the vision model service

Every field has a default so a minimal YAML works. See
config/harness.example.yaml for the annotated version.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class WallXConfig:
    endpoint: str = "wss://maas-api.x2robot.com/v1/vla/wallx"
    headers_file: str = ""  # path to JSON headers; env WALLX_HEADERS_FILE wins
    open_timeout_s: float = 15.0
    # Measured warm-up 4.0 s (probe 2026-11-12, n=1).
    first_request_timeout_s: float = 20.0
    # Measured steady median 0.61 s, max 0.67 s (n=5). Venue network will be worse.
    request_timeout_s: float = 8.0
    reconnect_attempts: int = 1  # per failed request; we never replay actions


@dataclass
class ListenConfig:
    host: str = "0.0.0.0"  # must be the LAN IP the robot can reach, never 127.0.0.1
    port: int = 8001


@dataclass
class VerifierConfig:
    enabled: bool = True
    chat_url: str = ""  # env AGENT_CHAT_URL wins
    model: str = ""  # env AGENT_MODEL wins
    api_key_env: str = "AGENT_API_KEY"
    timeout_s: float = 8.0
    image_max_side: int = 512  # downscale before upload to cut latency
    cameras: list[str] = field(default_factory=lambda: ["front", "left_wrist", "right_wrist"])
    temperature: float = 0.0
    # Our reply is a tiny JSON ({"answer", "reason"}). Providers like Groq
    # reserve max_tokens against an output-tokens-per-minute budget (free
    # tier: 1000 OTPM on qwen/qwen3.8-27b), so leaving it unset fails with 429.
    max_tokens: int = 150
    # Provider-specific fields merged into the request body, e.g. to turn off
    # a model's thinking output: {"reasoning_effort": "none"} on Groq.
    extra_body: dict = field(default_factory=dict)


@dataclass
class ArmBox:
    min: list[float] = field(default_factory=lambda: [-1.0, -1.0, -1.0])
    max: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass
class FilterConfig:
    # off: no checks. log: check and record, forward unchanged.
    # enforce: replace a violating chunk with HOLD.
    # Default is enforce: forwarding a 40 cm jump is worse than holding still.
    # Still confirm with organizers that the harness may replace chunks.
    mode: str = "enforce"
    # PLACEHOLDERS: measure real limits on site from recorded states.
    workspace: dict[str, ArmBox] = field(
        default_factory=lambda: {"left_ee": ArmBox(), "right_ee": ArmBox()}
    )
    max_row_step_m: float = 0.05  # ~1.5 m/s at 33.7 ms rows; arm max is 2.5 m/s
    max_first_row_jump_m: float = 0.10  # row 0 vs current measured pose
    max_row_rot_step_rad: float = 0.35
    freeze_idle_arm: bool = False  # overwrite the non-active arm with HOLD rows


@dataclass
class SupervisorConfig:
    verify_every: int = 2  # ask the vision model every N requests within a stage
    still_threshold_m: float = 0.005  # EE moved less than this between requests = still
    # MEASURE ON SITE: gripper reading when closed on nothing vs on a part.
    # A reading inside this band after a close command = "something is held".
    grasp_band: list[float] = field(default_factory=lambda: [0.08, 1.30])
    # MEASURE ON SITE: gripper reading above this = open (part released).
    open_above: float = 1.0
    # Measured gripper must be at least this much MORE open than the last
    # commanded value to count as "stalled on a part" (fingers blocked).
    stall_margin: float = 0.05
    state_file: str = "runs/supervisor_state.json"
    unknown_goal: str = "passthrough"  # passthrough | hold


@dataclass
class HoldConfig:
    on_done: bool = True  # after the last stage: HOLD instead of more Wall-X calls
    on_upstream_failure: bool = True  # Wall-X timeout or disconnect: HOLD this request
    on_task_failed: bool = True  # stage retries exhausted: HOLD


@dataclass
class RecorderConfig:
    dir: str = "runs"
    save_frames: bool = True
    frame_every: int = 1  # save camera frames every N requests


@dataclass
class FinisherConfig:
    """Training-free final alignment and press controller (xharness/finisher.py).

    OFF by default. It generates arm motion itself instead of Wall-X, which
    organizers have not yet approved (docs/ORGANIZER_QUESTIONS.md #13).
    Every number below is a PLACEHOLDER until measured on the real robot.
    """

    enabled: bool = False
    # --- look-then-move servo (wrist camera, z and rotation frozen)
    explore_mm: float = 3.0  # exploratory move per axis to estimate the 2x2 image Jacobian
    gain: float = 0.6  # 0.5 to 0.7
    max_step_mm: float = 3.0  # per axis, per correction
    max_corrections: int = 8
    converge_mm: float = 0.3  # converged = 2 consecutive frames under this
    broyden: bool = True  # refine the Jacobian after every move
    min_confidence: float = 0.5  # below this the finisher aborts to HOLD
    scale_tolerance: float = 2.5  # Jacobian px/mm must be within this factor of part width px / mm
    target_offset_px: list = field(default_factory=lambda: [0.0, 0.0])  # MEASURE: where the aligned hole appears
    # --- image detection
    dark_value_max: float = 0.22  # HSV value below which a pixel can be hole shadow (0 to 1)
    # --- press. The target is ABSOLUTE: the stage's seated_z (gripper z with the part fully
    # seated, MEASURE ON SITE by hand-placing a seated part and reading observation.state)
    # minus overtravel. Without seated_z the finisher never presses.
    overtravel_mm: float = 0.5  # command this far below seated_z so the snap completes. PLACEHOLDER
    hard_floor_mm: float = 1.0  # absolute floor = press target minus this; never commanded, abort if the arm is lower
    fast_step_mm: float = 5.0  # descent step until slow_zone_mm above seated_z. PLACEHOLDER
    # Must start ABOVE the part's first contact (resting on the hole, before the snap), or a fast
    # step can drive up to fast_step_mm into a jam. PLACEHOLDER: first contact z minus seated_z, plus 2 mm.
    slow_zone_mm: float = 3.0
    press_step_mm: float = 1.0  # descent step inside the slow zone
    contact_margin_mm: float = 0.5  # settled lag (measured z above commanded) that means resistance. PLACEHOLDER
    # Resistance while MORE than this above seated_z is a jam (part on the rim, misaligned), not the snap.
    # Must exceed the snap resistance offset (sim: 1.5 mm). The brief said 1 mm; 2 mm is needed for the sim snap.
    jam_zone_mm: float = 2.0
    jam_lift_mm: float = 10.0  # on a jam: lift this much (gripper stays closed), then abort back to align
    # --- release and lift
    grip_command: float = 0.0  # gripper command that keeps the part gripped (used if Wall-X never set one). PLACEHOLDER
    open_command: float = 1.526  # command that opens the gripper
    open_above: float = 1.0  # measured reading that counts as open. PLACEHOLDER
    lift_mm: float = 30.0
    lift_step_mm: float = 6.0
    ramp_rows: int = 8  # rows over which a chunk reaches its target; the rest hold it


@dataclass
class HarnessConfig:
    wallx: WallXConfig = field(default_factory=WallXConfig)
    listen: ListenConfig = field(default_factory=ListenConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)
    hold: HoldConfig = field(default_factory=HoldConfig)
    recorder: RecorderConfig = field(default_factory=RecorderConfig)
    finisher: FinisherConfig = field(default_factory=FinisherConfig)
    task_files: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _build(cls: type, data: Any) -> Any:
    """Recursively turn plain dicts into the dataclasses above."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ValueError(f"{cls.__name__}: expected a mapping, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        f = known[name]
        default = f.default_factory() if callable(f.default_factory) else f.default  # type: ignore[misc]
        if is_dataclass(default):
            kwargs[name] = _build(type(default), value)
        elif name == "workspace":
            kwargs[name] = {arm: _build(ArmBox, box) for arm, box in value.items()}
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None) -> HarnessConfig:
    """Load YAML (optional) then apply environment overrides."""
    data = {}
    if path:
        data = yaml.safe_load(Path(path).read_text()) or {}
    cfg: HarnessConfig = _build(HarnessConfig, data)
    if os.environ.get("WALLX_ENDPOINT"):
        cfg.wallx.endpoint = os.environ["WALLX_ENDPOINT"]
    if os.environ.get("WALLX_HEADERS_FILE"):
        cfg.wallx.headers_file = os.environ["WALLX_HEADERS_FILE"]
    if os.environ.get("AGENT_CHAT_URL"):
        cfg.verifier.chat_url = os.environ["AGENT_CHAT_URL"]
    if os.environ.get("AGENT_MODEL"):
        cfg.verifier.model = os.environ["AGENT_MODEL"]
    if cfg.filter.mode not in ("off", "log", "enforce"):
        raise ValueError("filter.mode must be off, log or enforce")
    if cfg.supervisor.unknown_goal not in ("passthrough", "hold"):
        raise ValueError("supervisor.unknown_goal must be passthrough or hold")
    return cfg


def load_headers(cfg: WallXConfig) -> dict[str, str]:
    """Read handshake headers from a private JSON file. Never logged."""
    if not cfg.headers_file:
        return {}
    path = Path(os.path.expanduser(cfg.headers_file))
    headers = json.loads(path.read_text())
    if not isinstance(headers, dict) or not all(isinstance(v, str) for v in headers.values()):
        raise ValueError(f"{path}: must be a JSON object of string headers")
    return headers
