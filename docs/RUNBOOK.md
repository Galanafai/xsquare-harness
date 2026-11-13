# On-site runbook (Nov 14 to 15, 170 St Germain Ave)

Print this. Tick each line. The rule that changes everything: **once a run
starts, nobody touches a keyboard, controller or phone.**

## Before leaving home

- [ ] Printed parts in every color, plus spares (buttons break and roll away)
- [ ] Holder / fixture, tape for marking positions
- [ ] Laptop charger, USB Ethernet adapter, Ethernet cable
- [ ] `python -m pytest -q` passes on the laptop you bring
- [ ] `tools/probe_wallx.py` report saved; `WALLX_HEADERS_FILE` works
- [ ] Vision model key works (`AGENT_CHAT_URL`, `AGENT_MODEL`, `AGENT_API_KEY`)

## Hour 1: baseline without our harness

- [ ] WebUI at `http://<robot IP>/`, "Local Environment", three camera feeds live
- [ ] Find the physical e-stop and the WebUI stop control
- [ ] Ask at check-in: the **local on-site Wall-X URL** and auth (set `wallx.url`), wired LAN
      or Wi-Fi isolation, and from the FAE: what trips a protective stop, max safe press force
- [ ] **Measured or commanded?** Close the gripper on a part: does `state.gripper` stop at the
      part's width (measured) or reach the command (commanded)? Same for z: a slow 1 mm
      finisher press onto a rigid block. Finisher jam detection needs measured values.
- [ ] `scp config/webui/local_d2_snapfit.json xr@<robot IP>:/home/xr/diversity/`
- [ ] Edit Model Configuration: Wall-X address directly, EndPose, sync,
      embodiment `quanta_x1_pro/gripper-g` (lowercase)
- [ ] Run case 1 once with a fixed instruction. Note what happened.

- [ ] **Finisher inputs (only if harness-generated motion is approved).** (a) Hand-place each
      part fully seated, read the gripper z in `observation.state` from a recorded request, and
      write it as `seated_z` (metres) on that finisher stage in `config/tasks/*.yaml`; without it
      the finisher refuses to press. (b) Wrist photos with a held button over the hole at 0, 2
      and 5 mm offsets: is the hole visible, and where does it appear when aligned
      (`finisher.target_offset_px`)? The sim draws holes on top of parts; a real part may hide them.
      (c) Also read the gripper z with the part resting aligned on top of its hole, before the
      snap (first contact). Set `finisher.slow_zone_mm` to (first contact z minus seated_z) plus
      2 mm, so the 5 mm fast steps never reach the part's first contact. (d) Hand-press a part
      and note how far above the seat the snap resists; `finisher.jam_zone_mm` must be larger.
- [ ] **Determinism on real frames.** Take ONE real observation (the three live camera
      frames and state saved from a run's `runs/<session>/` folder, or capture one), send
      the same request to Wall-X twice (more is better) and record how far apart the
      chunks are in mm (mean and worst row over the 32 rows, per arm).
      `tools/wallx_experiments.py a` does this with a sim scene; point its `base_request` at
      the real frames. Offline result on a sim scene was 65 mm mean (worst row 179 mm). If real
      frames are also about 65 mm, a single chunk is a random draw and retries will not repeat
      a motion: plan around closed-loop checks, not instruction wording.

## Hour 1 to 2: measure what the code marks as placeholder

- [ ] Gripper: calipers at commands 0, 0.5, 1.0, 1.526 -> `xharness/parts.py`
- [ ] Close on a button vs on nothing; read gripper values in the run log
      -> `supervisor.grasp_band`, `open_above`
- [ ] After a few runs: `python tools/workspace_from_runs.py runs/` -> `filter.workspace`

## Switch to the harness

- [ ] `ip addr` / `ipconfig`: your LAN IP the robot can reach
- [ ] `python -m xharness --config config/harness.local.yaml --host <LAN IP> --check`
- [ ] Start it without `--check`; WebUI model address `ws://<LAN IP>:8001`
- [ ] First run: watch for `[0001] ACT ... forwarded` lines in the terminal
- [ ] HOLD test, arms clear of everything: `python -m xharness --hold-only --host <LAN IP>`,
      start a run, confirm the arms do not move at all. If they move or the
      client errors, HOLD chunks are not accepted: tell Claude / the team.

## Skill measurement (evidence for the demo decision)

For each stage, 10 trials, write the success count:

| stage | successes / 10 |
|---|---|
| grasp slider | |
| insert slider | |
| grasp shell | |
| snap shell | |
| grasp button | |
| press button | |

## Between runs

- [ ] `python -m xharness --reset-state` (otherwise the supervisor resumes the old run)
- [ ] Same scene setup marks, same prompt version, note harness git commit

## Official run

- [ ] Harness running, terminal visible, nobody touching anything
- [ ] Robot ends in HOLD after the last stage; organizers end the run
