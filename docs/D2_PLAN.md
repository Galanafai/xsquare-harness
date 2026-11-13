# Demo 2 (SnapFit Carrot): why it's hard, why it's doable, one possible answer

Status, Nov 12: analysis and a proposal for the team to discuss, not a decision.
Nothing in this doc is built yet. The demo choice is still a team call; this
doc only covers "if we pick D2, how do we finish it?"

## Short version

- Wall-X is good at reaching, grasping and carrying. It is bad at the last
  half millimeter, and no prompt fixes that.
- So don't ask it for the last half millimeter. Let Wall-X bring the part
  close, then let the harness do the final alignment and press itself,
  using the wrist camera.
- That plan depends on one organizer answer: **may the harness send its own
  arm motion, instead of only forwarding Wall-X's?** Without it, D2 is mostly
  partial credit.

## What we discovered

1. **The precision gap is real, and X Square's own paper shows it.** In the
   Wall-OSS-0.5 report, zero-shot sorting scores 96 to 100, but charger
   plugging (the closest task to our snap fit) scores 2 to 9. Even fine-tuned
   it only reaches 24 to 35. The authors say failures happen in the "final
   precise execution", after a good approach and grasp
   ([Wall-OSS-0.5](https://arxiv.org/pdf/2605.30877v2)).
2. **Wall-X is random from call to call.** We sent the same observation 5
   times. The chunks differed by 65 mm on average, 179 mm worst case
   ([WALLX_EXPERIMENTS](WALLX_EXPERIMENTS.md)). The snap needs about 0.5 mm.
   Caveat: sim-rendered input, so real frames may be calmer. We measure that
   Saturday.
3. **Wording doesn't steer it finely.** Different phrasings differed no more
   than the same phrasing called twice. "Move 1 mm left" is not a usable
   command.
4. **The tuning knobs are a dead end.** `control_context` (the server's offset
   slots) moved nothing above the noise.
5. **Vision models are bad judges of contact.** On an assembly benchmark no
   model beat 0.60 accuracy at "did it succeed", and they lean toward saying
   yes ([FailBench](https://arxiv.org/pdf/2609.03611)). Also, the gripper
   hides the hole while pressing. So "seated?" is only asked after the gripper
   lets go and lifts away (already in the task files).
6. **Fine-tuning would help, but it's too big a bet here.** Even multi-task
   fine-tuning only got charger plugging to about a third, and it costs
   data collection plus overnight training we don't have.

## Why D2 is so hard

Three things stack up at the final press:

- **Tolerance:** the button seats within about 0.5 mm, axially. The arm's own
  repeatability is about 1 mm.
- **Blindness:** the gripper covers the hole exactly when precision matters,
  and there is no force sensor in the API.
- **Safety:** pressing a snap fit is where motors get overloaded, and the
  rules forbid that. Wall-X's chunks are absolute poses with no depth limit.

Everything before the press (grasp, carry, rough align) is the kind of task
Wall-X already does well.

## Why it's still doable

- **K = 4.5 is the highest of our demos,** and partial steps score. A case
  where grasp and align work but the press fails still earns points.
- **The hard part is small and well understood.** It's the last centimeter of
  motion, the same in all three cases: put a part over a hole and push it in.
  One small controller covers all of them.
- **There's a training-free method with real precedent.** Uncalibrated
  look-then-move visual servoing has put pegs into holes with 0.5 mm clearance
  in 20 of 20 trials ([Huang et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5017361)).
  That was on better hardware than ours, so it's precedent, not a promise.
- **Our parts are easy to see.** Bright, known colors and known sizes (the
  button is 10.7 mm wide), so simple color and shape detection should work
  without a learned model. Unverified until we have real photos.

## One possible answer: Wall-X carries, a small "finisher" finishes

1. **Hand-off.** Wall-X grasps the part and brings it above the target. The
   vision check confirms "above the toy, within a few cm".
2. **Align with the wrist camera.** The harness finds the held part (color) and
   the hole or slot (shape). It makes one tiny test move per axis (about 3 mm)
   to learn how the image moves, so no camera calibration is needed. Then it
   nudges 2 to 3 mm at a time until they line up, usually 3 to 8 nudges.
3. **Stepped press.** Lower 1 mm per step to a hard depth limit. If the arm
   stops reaching the height we commanded, that's contact, so stop. The depth
   limit protects the motors.
4. **Release, lift, check.** Open, lift 3 cm, ask "seated flush?" now that the
   hole is visible.
5. **Retry or move on.** Bounded retries, then hold still. Earlier steps keep
   their points.

Everything the finisher sends still goes through the existing safety filter.
It would sit behind a config flag, off by default, so the Wall-X-only path
stays available.

## Two tracks, decided by the organizers

| | "Harness may send its own motion" | "Only Wall-X motion" |
|---|---|---|
| Grasp and carry | Wall-X | Wall-X |
| Final align | Finisher, wrist camera | Wall-X, with retries |
| Press | Finisher, stepped, depth-capped | Wall-X, then release and check |
| Realistic outcome | Real shot at full D2 cases | Mostly partial credit |

Ask at the Friday 4 PM fireside chat, and again at check-in Saturday. This is
[ORGANIZER_QUESTIONS](ORGANIZER_QUESTIONS.md) question 13.

## Possible schedule

- **Friday:** build the finisher in the fake world, flag off. The fake wrist
  camera already renders the colored parts, so the loop can be tested.
- **Saturday, first hour:** hold-still test, then a 3 mm harness-move test (does
  the robot execute our own motion?), then measure the gripper and press height.
- **After that:** practice runs on the A board, tuning only from recordings.

## Risks, honestly

- **Tolerance stack:** 1 mm repeatability against a 0.5 mm snap. If the hole
  has any lead-in taper, the finisher can probably make it. If it's a straight
  interference fit, it may jam anyway. Hand-pressing a printed part tells us
  which, as soon as someone has one.
- **The organizers may say no** to harness-generated motion. Then D2 is a
  partial-credit play, and other demos should be compared on that basis.
- **Unknown press force** for our prints. The depth limit must be set from a
  measurement, not a guess.
- **Measured vs commanded pose:** contact detection assumes the robot reports
  where the arm actually is. Unverified (question 15).
- **Integration time:** this is the biggest piece of new code on the list. If
  it isn't working in sim by Friday night, it shouldn't go on the robot.

## Open questions for the team

1. Do we want D2 enough to build the finisher Friday, before the organizers
   answer?
2. If they say no, do we still run D2 for partial credit, or switch?
3. Who brings the printed parts, and can someone hand-press a button before
   Saturday?
