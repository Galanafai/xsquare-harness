# Questions for the organizers (contact@x2robot.com)

Answered (Nov 11 email): cloud Wall-X may be used before the event;
commercial models and fine-tuned open-source models are allowed; X2Real is
the simulation guide. The rules doc answers scoring (sum x K) and that we
may switch between confirmed demos on site.

## Answered in the organizers' reply of Nov 12 (22:31 PT)

Their answers, paraphrased, with what each means for the harness.

| # | Question | Answer | What it means for us |
|---|---|---|---|
| 13 | May the harness send its own arm motion for the final insertion instead of forwarding Wall-X chunks? | **Yes**, "in accordance with the competition rules". | The finisher is allowed. Track A of [D2_PLAN](D2_PLAN.md) is on. Our motion still goes as replies through their client; we start no other control process. |
| 1 | May the harness replace or modify Wall-X chunks (HOLD, safety)? | Not answered directly. They said we may freely switch the instruction we send. | Covered by the answer to 13: if our own motion is allowed, HOLD chunks are too. Confirm in person if anything looks off. |
| 2 | Who ends a run? | The judges stop the process. | HOLD forever after the last stage is the right behavior. |
| 3 | Current/torque and depth? | Allowed, but we must modify the client side ourselves; no official guidance. | Not pursued: modifying their client is a big unguided bet. The finisher detects contact from pose lag instead. |
| 4 | Passive holder approved? | Show them photos first. | Send photos of the prints and holder when available. |
| 5 | What does the client execute per request? | A new request starts only after the arm finishes the current action. | The state we receive is after the chunk finished, so the finisher's lag reading is meaningful. The partial-execution worry is mostly gone (the double send can stay as cheap insurance). The arm idles while we wait for a reply. |
| 6 | Is `state.gripper` measured or commanded; mm per unit? | Measure on site. | Still unknown; see open question 15. |
| 7 | On-site Wall-X endpoint? | An API will be deployed locally on site. | Get the local URL and auth at check-in, set `wallx.url`. Latency should drop; the cold-start question (12) mostly stops mattering. |
| 9 | Protective stop and reset? | An FAE will help reset it. | Ask the FAE what trips it, before the first press test. |
| 10, 14 | `control_context`? | Don't worry about it. | Matches our finding: we forward it untouched and never set it. |
| 11 | Does the client accept replies without the diagnostic fields? | The client mainly reads the EE coordinates. | HOLD replies are probably fine. Still run the `--hold-only` test in Saturday hour 1. |

## Still open, most important first

15. **Are the EE pose and gripper values in `observation.state` measured or
    commanded?** The finisher's contact and jam detection need the measured pose.
    If they are commanded, lag is always zero: the absolute floor still bounds the
    press, but jams go undetected. Measure on site: close the gripper on a part (a
    measured reading stops at the part's width), and command a slow 1 mm press onto
    a rigid block (a measured z stops at the surface).
16. **Is there a maximum safe press depth or force** for the arms? Ask the FAE on
    site, together with what trips a protective stop.
4. **Holder approval:** pending photos.
8. **Network:** wired LAN or Wi-Fi with client isolation? Ask at check-in, along
   with the local Wall-X URL and whether the same key works there.
17. **New prop, new cases?** Our Demo 2 prop changed to four parts (cover, main body, core,
    one snap pin; possibly at 200% size) instead of the shell, slider and two buttons. Do
    the A/B-board cases and the scored actions (and K) get updated to match? Send photos.
