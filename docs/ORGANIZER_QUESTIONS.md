# Questions for the organizers (contact@x2robot.com)

Already answered (Nov 11 email): cloud Wall-X may be used before the event;
commercial models and fine-tuned open-source models are allowed; X2Real is
the simulation guide. The rules doc answers scoring (sum x K) and that we
may switch between confirmed demos on site.

Still open, most important first:

1. **May the harness replace or modify Wall-X action chunks?** We use this
   for safety (replace NaN / huge jumps with a stay-still chunk), to hold
   still after the task is done, and optionally to freeze the idle arm.
2. **Once a run starts nobody may touch a keyboard. Who ends the run, and
   how?** After the last step our harness keeps the arms still. Is that the
   expected behavior, and do the judges press End in the WebUI?
3. **The rules say the platform exposes joint current/torque and RGB-D
   depth. How do we read them?** The harness protocol (public_v2) only has
   EE pose, gripper and three JPEGs. Is there a read-only API or topic on
   the robot we may use?
4. **Is the passive holder for Demo 2 approved**, and may it be taped or
   clamped to the table at a fixed position?
5. **In sync mode, how many of the 32 rows does the client execute per
   request, at what period, and is the arm still while it waits?**
6. **Is `state.gripper` the measured follower position or the commanded
   value?** What opening in mm do 0 and 1.526 correspond to?
7. Answered by our probe: auth is `Authorization: Bearer <key>`. Still open:
   is there an on-site Wall-X endpoint if the cloud is slow? Steady latency
   from home was 0.61 s per request; what will it be from the venue?
8. Robot-to-laptop network: wired LAN? Any client isolation on Wi-Fi?
9. Does the arm protective-stop on contact or overcurrent, and how is it
   reset?
10. **What `control_context` does the robot client send, and should we leave it
    alone?** The server wants an object with 7 named slots per arm. Omitted, it
    uses a startup context (6 slots set). Sending all-null sets 0 slots and one
    such call took 3.2 s. We forward whatever the client sends.
11. **Does the client accept a reply that lacks Wall-X's diagnostic fields**
    (`server_timing`, `usage`, `rtc`, ...)? Our HOLD chunks contain only
    `embodiment`, `normalization` and `actions`.
12. Partly answered by our probe (2026-11-12): warm-up is NOT per connection.
    A second connection opened while the first was live took 322 ms on its
    first call, and the relay's fresh connection in the sim run took 292 ms.
    The only slow call (4.0 s) was the first call after the endpoint had been
    idle. Still open: does the server go cold after idling, and for how long?
    If so, the client should send a throwaway request before the official run.
