# Difficulties we will probably hit, and what handles each one

Ranked by how likely they are to cost us points. "Handled in" points at the
code; "Still open" is what code cannot fix.

## Facts from the rules that shape everything

* **Final score = (sum of the 6 B-board case scores) x K.** Demo 2 has
  K = 4.5, so 1 point on Demo 2 is worth about 2 points on Demo 5 (K = 2.3).
* **B board: one demo, one attempt, blind.** We may switch between our
  confirmed demos on site if one proves unsuitable.
* **No touching any keyboard, controller or phone once a run starts.**
  LAN traffic is monitored. The harness must finish and hold on its own.
* **8 minutes per case.** Unfinished steps score 0.
* **No motor overload or hardware-damaging actions.** Pressing snap fits
  hard against a fixed part is exactly where this risk lives.
* The rules say the platform has **joint current/torque APIs and RGB-D
  cameras**. The harness protocol (public_v2) carries neither. Ask how to
  read them (see ORGANIZER_QUESTIONS.md).

## 1. Zero-shot Wall-X may not manage precision insertion

The K-setting step (button snap, +-0.5 mm, axial, hole hidden by the
gripper) is the hardest class of task for any zero-shot VLA.

* Handled in: `config/tasks/*.yaml` split each case into grasp / align /
  press stages with retries, so grasp and align still earn partial credit.
  `sim/` uses tight tolerances so jams and retries get exercised.
* Still open: measure real success rates per stage on day one (10 trials
  each). Physical help: an approved holder with chamfered lead-ins, buttons
  presented upright, the loosest fit we can justify.

## 2. A harness failure becomes a robot failure

The example harness closed the connection on any agent error; the client
reconnects and the task restarts mid-motion.

* Handled in: `relay.py` (never closes on our errors; HOLD instead),
  `verifier.py` (any failure -> "unsure", never raises), `supervisor.py`
  (progress saved by goal, survives reconnects and restarts).
* Tested by: `tests/test_end_to_end.py` (Wall-X crash, timeout, text frame,
  client reconnects, messy vision replies).

## 3. Declaring "done" too early (or never)

A false "done" in a no-touch run means the robot stops before finishing.
A missed "done" means wasted minutes and maybe undoing the work.

* Handled in: `supervisor.py` needs 2 consecutive "yes"; "unsure" never
  counts; optional gripper-stall and stillness checks; after the last stage
  the robot only ever receives HOLD.
* Tested by: a 60-run sweep with a vision model that lies 10% of the time
  produced zero premature advances (`sim/run.py` flags FALSE POSITIVE).
* Still open: real vision-model accuracy on real camera frames. Replay
  recorded frames from `runs/*/frames/` to tune questions.

## 4. Grasping 10.7 mm buttons

Smaller than most fingertips, easy to knock over, and the gripper hides
the hole once it holds one.

* Handled in: `parts.py` graspability report; stage questions ask about
  "held and lifted", not just "touching".
* Still open: gripper mm-per-unit and fingertip size (placeholders now).

## 5. Left / right confusion

The front camera is mirrored: the robot's right arm appears on the left.

* Handled in: verifier system prompt states this; instructions name parts
  by color, never by side.

## 6. Bad action chunks

NaN, wrong shapes, 40 cm jumps, leaving the table area.

* Handled in: `action_filter.py` (default `enforce`: replaced with HOLD).
* Still open: real workspace boxes (`tools/workspace_from_runs.py`);
  organizer OK for replacing chunks.

## 7. Wall-X slow, down, or the network flaky

* Handled in: `relay.py` uses a longer timeout for the warm-up call, throws
  away a connection after any timeout (a late reply would otherwise be
  paired with the next request), retries once, then HOLD. The robot keeps
  asking and resumes when Wall-X is back.
* Still open: venue network. Use wired Ethernet and a fixed LAN IP.

## 8. The idle arm wanders

Wall-X always moves both arms.

* Handled in: `filter.freeze_idle_arm` + `active_arm` per stage (off by
  default; needs organizer OK).

## 9. Config drift on the robot side

The WebUI's `model_overrides.json` beats the task JSON; "Restore Default"
can wipe keys; the gripper suffix must be lowercase and match the robot.

* Handled in: `config/webui/local_d2_snapfit.json` as the single source;
  the recorder logs the handshake and every embodiment echo.
* Still open: the RUNBOOK checklist, every time.

## 10. Latency eats the 8 minutes

Each vision check is seconds; the robot waits during it (sync mode).

* Handled in: `verify_every`, image downscaling, cheap gripper check
  before any vision call.
* Measure: `elapsed_ms` per request in `runs/*/events.jsonl`, and the probe
  report's warm-up and steady latencies.
