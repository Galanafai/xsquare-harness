# Proposal: gripper-first verification (not implemented)

Status: proposal for the team to approve. Changing stage logic needs sign-off
(CLAUDE.md), so nothing here is built yet.

## Why

The vision model is the slowest, most rate-limited part of the loop. Measured
on Groq `qwen/qwen3.8-27b`, free tier (2026-11-12): 7000 input tokens per
minute, and every image costs about 1.7k tokens whatever its pixel size
(a 128 px image counted the same as a 512 px one). Three cameras is about
5.3k tokens per call, so the free tier allows roughly ONE vision call per
minute, not six. An 8 minute case then affords about 8 calls in total.

Vision calls per case today (sim, fake Wall-X that behaves well, one call
every 2nd request; real Wall-X will need more):

| case | vision calls | of which grasp or release stages |
|---|---|---|
| 1 insert slider | 11 | 5 (grasp 4, release 1) = 45% |
| 2 snap shell | 11 | 5 = 45% |
| 3 press buttons | 24 | 8 (grasp 4 + 3, release 1) = 33% |

Hard floor from the YAMLs (2 confirmations per stage, 1 for release): 7, 7 and
13 calls. Worst case with all retries used is 50 to 100 per case.

So even the optimistic sim numbers do not fit a free tier in 8 minutes, and
removing grasp and release calls saves one third to one half of them.

## The idea

Two kinds of stage question:

* Mechanical state, readable from the robot state we already receive on every
  request: "is something gripped" and "has the gripper opened and moved away".
* Geometric state, needs eyes: "is the part lined up over the slot",
  "is it seated flush".

Decide the first kind from `observation.state` only. Vision answers the second
kind only.

### Grasp stage done when, for N consecutive requests (N = 3 suggested)

1. The holding arm's gripper reading is inside `grasp_band` AND at least
   `stall_margin` above the last commanded gripper value (fingers blocked by
   something). The supervisor already tracks the last commanded value
   (`note_commanded_gripper`) and `_gripper_condition` already computes this.
2. The arm is still (`still_threshold_m`), so the part is not mid-swing.
3. The arm's z is higher than at the start of the stage by a lift margin
   (to separate "closed on the part and lifted" from "closed on the part still
   on the table"). The margin needs on-site tuning.

### Release stage done when, for N consecutive requests

1. Both gripper readings are above `open_above`.
2. The arm that held the part has moved up and away by a margin from where it
   released.

### Vision stays for

align_* (is it above the slot), insert, press and seat questions, where the
answer is geometry that the robot state cannot give.

## Config sketch

    done:
      mode: gripper            # default stays vision, so nothing changes unless a YAML opts in
      gripper_holding: left_ee
      confirmations: 3
      lift_m: 0.03             # PLACEHOLDER, measure on site

`_check_done` gets one branch: `mode == "gripper"` returns a Verdict built
from the state, with `latency_ms = 0` and reason "gripper stall", and never
calls the verifier. The sim already models the stall, so it can be tested
offline.

## Risks

* False "grasped": fingers stall on the wrong thing (the holder, the other
  part, a part edge) or the part slips out later. Mitigation: the next stage
  asks vision whether the part is above the slot, so a missed grasp costs one
  stage budget rather than a wrong completion. Option (needs approval): while
  in align or press, if the stall signal disappears for N requests, send the
  task back to the grasp stage.
* Everything depends on `state.gripper` being the MEASURED follower position.
  If it is the commanded echo (organizer question 6), the stall signal does not
  exist and this mode is useless. Check this first on site, from the first
  recorded run: the gripper should read above the command while holding a part.
* `grasp_band`, `open_above`, `stall_margin` and `lift_m` are all unmeasured.
* Fully vision-free grasp decisions are only as good as those thresholds.
  A cheap safety net: keep ONE vision confirmation after the gripper signal
  (confirmations: 1) instead of two. That still saves about half of the grasp
  calls.

## Estimated saving

Sim, per case: 5 of 11 calls (case 1 and 2), 8 of 24 (case 3), i.e. 33 to 45%.
With the "one vision confirmation after the gripper signal" variant, about
half of that. Counting only the floor of the YAML (7, 7, 13): 3, 3 and 5 calls
fewer, so 4 left in cases 1 and 2 and 8 in case 3. Adding fewer cameras for
the remaining calls (front plus one wrist, about 3.7k tokens) or a paid tier
is still needed to get the remaining calls to fit in 8 minutes.
