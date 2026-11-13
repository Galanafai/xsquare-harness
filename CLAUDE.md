# CLAUDE.md: xsquare-harness

## What this project is

Our team's (Yanzhou Pan, Kuangyuan Sun, Galanafai Windross) agent harness for the
BRIGHT 2026 Embodied Agent Hackathon, run by X Square Robot.
Event: Sat Nov 14, 10:00 to Sun Nov 15, 20:00 PT, 170 St Germain Ave, San Francisco.

We are targeting **Demo 2: SnapFit Carrot**, assembling a 3D-printed snap-fit fidget
toy (handle, slider, shell, two buttons) with a dual-arm robot. Fallback demo if
Demo 2 proves infeasible on site: Demo 5 MedSort (K = 2.3).

## Competition rules that shape the code

- Final score = (sum of the 6 B-board case scores, 10 points each) x K. Demo 2 has K = 4.5.
- The A board is practice: 3 official runs plus unlimited self-tests, not scored.
  The B board is a blind test of ONE demo with ONE attempt; it decides the ranking.
  We may switch between our confirmed demos on site.
- Each case is limited to 8 minutes. Unfinished steps score 0.
- **Fully autonomous:** once a run starts, nobody may touch a keyboard, controller
  or phone. LAN traffic is monitored and covert teleoperation disqualifies us.
  The harness must detect completion itself and then keep the arms still.
- No actions that could damage the robot or overload motors. Pressing snap fits
  is where this risk lives.
- Organizers provide no props. We bring all printed parts and the holder.
- Organizer confirmations so far: we may use the cloud Wall-X endpoint before the
  event, commercial models, and fine-tuned open-source models. X2Real is their
  "simulation guide".

## The platform (tags: [doc] = organizer docs, [code] = seen in source, [unverified])

- Robot: X Square desktop leader-follower kit, two 6-DoF arms, G grippers.
  Embodiment string: `quanta_x1_pro/gripper-g` (lowercase; capital G fails). [doc]
- Reach 649 mm per arm, repeatability +-1 mm, max EE speed 2.5 m/s. Gripper
  stroke, force and finger size are NOT published: measure on site. [doc]
- Cameras: `front` (fixed, faces the robot, MIRRORED: the robot's right arm appears
  on the left of the image), `left_wrist`, `right_wrist`. [doc]
- The organizers' robot client (x2robot_client) owns observation capture, IK,
  interpolation and execution. We must NOT replace it or start another control
  process. [doc]
- Rules claim joint current/torque and RGB-D depth APIs exist, but the harness
  protocol carries neither. How to read them is an open organizer question.
- WebUI at `http://<robot IP>/`. Task JSONs go in `/home/xr/diversity/local_<id>.json`
  via scp. WebUI model overrides in `/home/xr/diversity/local/state/model_overrides.json`
  BEAT the task JSON; "Restore Default" can wipe API keys. [doc]

## The Wall-X protocol (public_v2), see xharness/protocol.py

- WebSocket, binary MessagePack + msgpack-numpy. Server sends a handshake dict
  first (`model: wall-x`, `model_version: 2610-dualhead-v2` per docs), then strictly
  one request, one response. [doc]
- Endpoint: `wss://maas-api.x2robot.com/v1/vla/wallx`. Auth = handshake HTTP
  header `Authorization: Bearer <key>` (measured 2026-11-12 with `tools/probe_wallx.py`). [measured]
- Request: `embodiment`, `normalization: "provided"`, non-empty string
  `instruction`, `observation.images.{front,left_wrist,right_wrist}` (JPEG base64,
  all three), `observation.state.{left_ee,right_ee}` with only `position[3]` (m),
  `rotation_rpy[3]` (rad, R = Rz(yaw) Ry(pitch) Rx(roll)) and scalar `gripper`.
  Absolute values in the robot base frame. Preserve client extension fields like
  `control_context`. No RTC fields in sync mode. [doc]
- Measured latency (2026-11-12, home network, n=5 steady): first call after idle 4.0 s (warm-up, not per connection),
  steady median 0.61 s, max 0.67 s. Server-side inference is about 85 ms; the rest is
  network. Relay timeouts: 20 s first call, 8 s steady. A request with a changed
  `control_context` took 3.2 s once. [measured]
- Real replies use plain msgpack lists (not msgpack-numpy arrays) and carry extra
  fields: `normalization`, `server_timing`, `usage`, `valid_rows`, `rtc`, `grip_check`,
  `control_context`. Our HOLD chunks copy the plain-list layout plus `normalization`. [measured]
- `control_context` (optional request field) must be an object per arm with exactly the
  keys eff_delay_ms, lpf_T_ms, grip_delay_ms, grip_gain, grip_offset, z_offset_mm,
  rot_offset_deg (null = masked). Omitted = server startup context (6 of 7 slots set);
  all-null sets 0 slots. The relay forwards the client's field untouched. [measured]
- Errors seen live: unknown_embodiment, missing_field, control_context_invalid. All keep
  the connection open. [measured]
- Response: `embodiment` echo + `actions.{left_ee,right_ee}.{position[32,3],
  rotation_rpy[32,3], gripper[32]}`. Row 31 repeats row 30. Rows 33.7 ms apart.
  BOTH arms always move. [doc]
- Errors: validation errors return `{"error": ...}` (string or {code, message}) and
  keep the connection open. Server crashes close with 1011 (the open-source server
  sends a text traceback first). [doc][code]
- Gripper G range `[0, 1.526]`. The unit is very likely drive-joint radians, not
  mm (X2Real gripper_convention.py). [code][unverified on hardware]
- Sync client config: action_horizon 32, control_dt 0.005, interpolate_multiplier 8
  (suggests 40 ms per row on hardware vs 33.7 ms model rows). How many rows the
  client executes per request is unknown. [doc][unverified]
- The API Tutorial PDF contains the competition key. Never copy it anywhere.

## Architecture

robot client -> `xharness/relay.py` -> Wall-X; replies -> `action_filter.py` -> robot.

- `protocol.py`: wire contract, validation, HOLD chunk builder.
- `relay.py`: WebSocket server. NEVER closes the robot connection because of our
  own or Wall-X errors (the organizers' example did; the client reconnects and the
  task restarts mid-motion). Wall-X down or slow -> HOLD. After any Wall-X timeout
  the connection is discarded (a late reply would pair with the next request).
  Forwards Wall-X bytes unchanged unless we deliberately replace them. Never
  caches or replays actions.
- `supervisor.py`: fixed stage list per case (from `config/tasks/*.yaml`). Advances
  on 2 consecutive "yes" from the verifier (plus optional gripper-stall and
  stillness checks). "unsure" never counts. Budget per attempt, retries rotate
  instruction phrasings, exhausted retries -> task failed -> HOLD. After the last
  stage: HOLD forever. Progress keyed by goal text, saved to
  `runs/supervisor_state.json`, survives reconnects and restarts.
- `verifier.py`: narrow yes/no questions to any OpenAI-compatible vision model
  (AGENT_CHAT_URL, AGENT_MODEL, AGENT_API_KEY). Any failure -> "unsure", never
  raises. Tolerates fenced or prose-wrapped JSON.
- `action_filter.py`: shape/NaN checks, row-0 jump, per-row step, workspace box,
  gripper range. Default `enforce` = unsafe chunk replaced with HOLD. Optional
  freeze of the idle arm (off by default).
- `recorder.py`: `runs/<time>_<session>/events.jsonl` + camera frames.
- `tasks.py`, `parts.py`, `config.py`: task YAML format, Demo 2 parts, config + env.
- Why fixed stages instead of an LLM planner: no-touch runs need predictable,
  readable, offline-testable behavior.

## Demo 2 specifics

- Parts (STL bounding boxes, mm): handle 70x40x20, solid slider 60x25.1x25.1,
  shell 70x33x23.5, buttons 10.7x10.7x12.1 (x2). Source: MakerWorld "Fidget gravity
  knife, carrot kid, little radish" (plates 1, 2, 4, 6). Colors in parts.py and
  the task YAMLs are planned colors: update to match the prints.
- K = 4.5 comes from F4 Precision_Insert: snap depth +-0.5 mm, axial press, hole
  hidden by the gripper. Other scored actions: C1 hold 3.9, C3 mate shell 3.3,
  G1 grasp button 2.5, G1 grasp part 2.3, F2 slider cycle 2.3, P2 place 1.3.
- A-board cases visible so far: 1 insert slider, 2 snap on shell, 3 press buttons.
  Cases 4 to 6 were cut off in our export. Each case starts pre-assembled up to its step.
- Zero-shot Wall-X is weakest at precision insertion. Expect grasp and align stages
  to work far more often than press stages. A passive holder with chamfers
  (pending organizer approval) is the biggest physical help.
- `config/webui/local_d2_snapfit.json` is the WebUI task file. Each case's
  `instruction` MUST equal the matching task YAML's `goal`.

## Placeholders that must be measured on site (don't invent values)

- Gripper opening in mm vs command, fingertip size (`xharness/parts.py`).
- Gripper reading when closed on a part vs on nothing (`supervisor.grasp_band`,
  `open_above`, `stall_margin`).
- Workspace boxes (`filter.workspace`; use `tools/workspace_from_runs.py`).
- Whether the client accepts our HOLD chunks (`python -m xharness --hold-only`,
  arms clear of everything).

## Simulation

- `sim/` is a kinematic fake (real part sizes, tight insertion tolerances, fake
  camera renders, fault injection). It tests harness logic, not whether Wall-X can
  do the task. `python -m sim.run --case 3 --faults standard --vlm-faults hostile`.
- Do NOT install Isaac Sim or X2Real for this event. X2Real needs an RTX GPU and
  uses a different wire format (`views`/`state`, init-relative poses), so it can't
  talk to the hosted endpoint without an adapter. Treat it as reference only
  (useful facts: `manaenv/assets/robots/ex001_6r.py`, `manaenv/policy/gripper_convention.py`).

## Commands

    python -m pytest -q                     # 28 tests, about 6 s, no network
    python -m xharness --check              # config + tasks + warnings
    python -m xharness --host <LAN IP>      # run the harness (never 127.0.0.1 on site)
    python -m xharness --hold-only          # every reply is HOLD
    python -m xharness --reset-state        # forget progress between practice runs
    python -m sim.run --case 1|2|3 [--faults standard] [--vlm-faults messy|hostile] [--drop-connection N] [--real-wallx]
    python tools/probe_wallx.py             # needs WALLX_API_KEY or WALLX_HEADERS_FILE

## Rules for Claude

- Never send motion commands to the robot or start any process on it that
  controls the arms. All motion goes robot client -> harness -> Wall-X.
- Never interact with the robot, the WebUI or the harness during an official run.
- Never print, log, or commit keys or header files. Secrets live in env vars only.
  Check `git diff --cached` before every commit.
- Run `python -m pytest -q` before every commit. Add a test for every bug fixed.
- Keep code readable: short functions, docstrings that explain why, plain names.
- No em dashes or double hyphens in docs or comments.
- Stop and ask before changing stage logic in `supervisor.py`, the task YAMLs,
  filter defaults, or anything that touches the real robot.
- Don't present placeholder numbers as measured. Mark anything unverified.
- Open organizer questions are in `docs/ORGANIZER_QUESTIONS.md`; add new ones there.
