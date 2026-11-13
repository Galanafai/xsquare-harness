# Offline Wall-X experiments (2026-11-12)

Endpoint `wss://maas-api.x2robot.com/v1/vla/wallx`, model `2610-dualhead-v2`, no robot involved.
Script: `tools/wallx_experiments.py` (calls spaced 2 s, about 60 calls in total). Raw chunks are in
`runs/wallx_exp/` (git-ignored).

**Caveat that applies to everything below:** the input is a rendered sim scene plus the API
Tutorial example poses, so the model sees a meaningless scene. Real scenes may behave
differently (unverified). Latency was 0.3 to 2.0 s per call, 3 calls above 1.2 s.

## Decision (team, 2026-11-12)

**control_context is dropped as a lever.** Sweeping the three accepted slots produced no effect
above the noise, and we cannot convert mm to its [-1, 1] encoding anyway. The relay still forwards the
client's `control_context` field untouched; we never set it ourselves.

## Summary

1. **Wall-X is strongly stochastic.** The same observation gave chunks that differ by 65 mm on
   average (worst row 179 mm) between any two calls. Any single chunk is one random draw.
2. **Phrasing is not distinguishable from noise** in this setup: different wording differs by
   65 to 93 mm, and the same wording by 65 to 85 mm.
3. **control_context:** slot values are normalized numbers in [-1, 1] (not mm or ms), and only
   `eff_delay_ms`, `grip_delay_ms`, `z_offset_mm` are trained. With the noise above, no
   slot change moved z by more than the noise. The "+5 mm / -5 mm" test cannot be run literally:
   the server rejects any value outside [-1, 1].
4. Consequence for the harness: chunk-to-chunk noise of this size makes a 0.5 mm press
   impossible to ask of Wall-X zero-shot, and a retry loop will not repeat the same motion.
   Closed-loop checks (the seated stage) matter more than instruction wording.

## a. Determinism: same observation, 5 calls

Pairwise distance between chunks, position only, in mm (mean over the 32 rows, and the worst row).

| pairs | left mean | left max | right mean | right max |
|---|---|---|---|---|
| 10 pairs | 65.32 | 179.10 | 65.73 | 152.56 |

Largest per-row range over the 5 chunks: rotation 29.196 deg, gripper 0.04153.

## b. Phrasing: 3 phrasings x 3 calls

- phrasing 0: "Pick up the orange slider."
- phrasing 1: "Grasp the orange block and lift it."
- phrasing 2: "Take hold of the orange slider and raise it off the table."

Position distance in mm (mean over rows | worst row).

| comparison | left mean | left max | right mean | right max |
|---|---|---|---|---|
| noise from (a), same wording | 65.32 | 179.10 | 65.73 | 152.56 |
| within one phrasing, this run | 84.77 | 200.66 | 80.44 | 196.49 |
| 0 vs 1 | 86.52 | 206.94 | 82.42 | 218.38 |
| 0 vs 2 | 92.77 | 281.81 | 86.44 | 227.52 |
| 1 vs 2 | 81.82 | 234.42 | 64.54 | 158.00 |


## c. control_context: what the server accepts

First sweep sent physical units. Replies (connection stayed open each time):

| sent | reply |
|---|---|
| `z_offset_mm` 5 or -5, `eff_delay_ms` 50 or 150, `grip_delay_ms` 100, `grip_gain` 1.2 | `control_context.left must be null or a number in [-1, 1]` |
| `grip_gain` 0.8, `grip_offset` +-0.05 | `grip_gain is not a trained parameter (kept: [eff_delay_ms, grip_delay_ms, z_offset_mm])`, same for `grip_offset` |
| `lpf_T_ms`, `rot_offset_deg` (not shown in the log) | rejected, same family of errors |
| `z_offset_mm` 0, `grip_delay_ms` 0 | accepted |

The startup context in the handshake is also in this [-1, 1] encoding (left: eff_delay 0.674,
grip_delay 0.337, z_offset 0.339; right: 0.450, -0.112, 0.289; other slots null). The handshake
says the encoding is `clip((v - center) / scale, -3, 3) / 3`, but not the centers or scales, so
we cannot convert mm to a slot value (organizer question 14).

## c2. control_context sweep with normalized values

Baseline = field omitted (server startup context) and all-null (0 slots set). Every other row sets ONE
slot on the LEFT arm only; the right arm is the control and should not move. dz is the mean over the
32 rows of the z difference to the mean of the all-null calls.

Noise floor, spread of the per-chunk mean z across 6 all-null calls: left 69.0 mm, right 63.0 mm (1 sd).

| variant | n | left dz mean +- s.e. (mm) | right dz mean +- s.e. (mm, control) | left grip delta | left dx, dy (mm) |
|---|---|---|---|---|---|
| omitted | 4 | +22.9 +- 11.0 | +13.3 +- 14.5 | -0.0162 | +12.2, +2.4 |
| all_null | 6 | -0.0 +- 28.2 | -0.0 +- 25.7 | +0.0000 | +0.0, +0.0 |
| z_offset_mm=-1 (left only) | 4 | +10.3 +- 4.7 | -5.2 +- 11.0 | -0.0151 | +41.3, -5.0 |
| z_offset_mm=-0.5 (left only) | 4 | +15.9 +- 11.7 | -1.4 +- 8.6 | -0.0199 | +56.6, -8.2 |
| z_offset_mm=0.5 (left only) | 4 | -12.3 +- 23.0 | -26.0 +- 26.3 | -0.0088 | -7.7, +1.5 |
| z_offset_mm=1 (left only) | 4 | +34.0 +- 18.2 | +13.6 +- 20.0 | -0.0012 | -65.3, +4.5 |
| eff_delay_ms=-1 (left only) | 4 | +18.9 +- 12.9 | -4.8 +- 22.6 | -0.0159 | +2.1, +2.0 |
| eff_delay_ms=1 (left only) | 4 | -1.3 +- 41.1 | -14.3 +- 41.2 | -0.0214 | +8.9, +0.8 |
| grip_delay_ms=-1 (left only) | 4 | -19.8 +- 46.0 | -24.9 +- 36.6 | +0.0075 | +2.7, -14.1 |
| grip_delay_ms=1 (left only) | 4 | +20.3 +- 9.8 | +4.0 +- 15.2 | +0.0036 | -27.2, -12.5 |
0

Reading: the noise (sd about 65 mm on the per-chunk mean z) is far larger than any z effect
(the largest, +34 mm, has a standard error of 18 mm). The right arm, which should be
unaffected, moves by similar amounts, which is the noise again. The only pattern that may be real is
the x shift under `z_offset_mm` (+41, +57, -8, -65 mm from -1 to +1), but it is not monotone in
the middle and has no error bar here. Inconclusive. To measure a real effect we would need many more
calls per setting or a scene where Wall-X is less random.
