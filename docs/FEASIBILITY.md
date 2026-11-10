# FEASIBILITY.md

Phase 0 findings for `blindside`. Date: 10 November 2026.
Source: `github.com/SkillCorner/opendata` at commit 4340d27, inspected directly.

Scope of this check: repository structure, dynamic events of all 20 matches, and the committed pose sample
(`data/bodypose/sample_1925299_phase406.jsonl.gz`, 309 pose frames, 12.3 seconds, one Brisbane Roar quick break).
The two full pose archives were **not** inspected. Everything marked "sample" below rests on 12 seconds of one match
and must be re-measured on the full files (section 7).

## 1. Data inventory (verified)

| Item | Finding |
|---|---|
| Matches in `data/matches/` | 20 |
| Matches with dynamic events | 20 of 20, 322 columns each, EPV and pressure columns present in all |
| Matches with phases of play | 20 of 20 |
| Matches with body pose | 2 (1925299 Brisbane Roar v Perth Glory, 1996435 Sydney FC v Adelaide United) |
| Pose archive size | 623 MB and 644 MB zipped, about 3.3 GB each as JSON |
| Tracking files | Git LFS, 10 fps |

Consequence: the XY-only baseline can be trained on 18 matches that are never touched by the pose analysis.

## 2. Body pose schema (verified)

- One JSON record per frame at 25 fps. `pose_frame = 2.5 * tracking_frame`.
- `player_data[i]` has `player_id`, `x`, `y`, `is_detected`, `joints`.
- `joints[name] = {"xyz": [x, y, z], "p90_mae_cm": float}`. `joints` is `null` when pose was not resolved.
- 29 joints: nose, neck, lEye, rEye, lEar, rEar, lShoulder, rShoulder, lElbow, rElbow, lWrist, rWrist, lThumb, rThumb,
  lPinky, rPinky, midHip, lHip, rHip, lKnee, rKnee, lAnkle, rAnkle, lHeel, rHeel, lBigToe, rBigToe, lSmallToe, rSmallToe.
- Usable axes: shoulders (lShoulder, rShoulder), hips (lHip, rHip), head (lEar, rEar, with nose and eyes as alternatives).
- Every frame lists 32 players (both squads). In the sample 50% of player-frames carry joints.
  The SkillCorner README states about one third across a full match.

## 3. On-ball engagements (verified, all 20 matches)

| | Value |
|---|---|
| Engagements per match | 754 to 1025 |
| Pose match 1925299 | 936 (pressure 281, recovery_press 224, other 193, pressing 176, counter_press 62) |
| Pose match 1996435 | 1025 |
| Total in the two pose matches | 1961 |
| Total in the 18 other matches | about 15,600 |

Link to the pressed possession: `associated_player_possession_event_id`, `player_in_possession_id`,
`associated_player_possession_frame_start`, `associated_player_possession_end_type`.

Outcome candidates already in the data: `force_backward`, `beaten_by_possession`, `beaten_by_movement`,
`stop_possession_danger`, `reduce_possession_danger`, `xloss_player_possession_*`, and on the possession row
`end_type` (in 1925299: pass 912, possession_loss 74, shot 30, clearance 24, foul_suffered 11) and `possession_epv_total` (95% populated).

XY baseline features already in the data: `interplayer_distance_start`, `angle_of_engagement`, `goal_side_start`, `speed_avg`,
`speed_difference`, `pressing_chain`, `simultaneous_defensive_engagement_same_target`, plus `overall_pressure_start` on the possession row (89% populated).

Caveat: several engagements can target one possession. Statistics must be clustered by possession.

## 4. Joint error and angular uncertainty (sample)

Median `p90_mae_cm` per joint: neck 5.0, shoulders 5.6 to 5.7, ears 6.7 to 6.8, hips 7.3 to 7.8, knees 9.6 to 9.9,
ankles 12.5, feet and hands 14.7 to 16.5.

Monte Carlo propagation (200 draws, isotropic Gaussian, sigma = p90 / 2.5, joints independent):

| Axis | Median width | Angular SD, median | Angular SD, 90th percentile | Share with SD below 20 degrees |
|---|---|---|---|---|
| Shoulders | 0.37 m | 5.0 deg | 6.5 deg | 99% |
| Hips | 0.22 m | 12.2 deg | 17.3 deg | 95% |
| Ears (head) | 0.18 m | 12.5 deg | 18.4 deg | 92% |

Agreement between axes (absolute difference, median): shoulders versus hips 12.2 degrees, shoulders versus ears 15.3 degrees.

Interpretation:
- Torso orientation from shoulders is precise under this error model.
- Head orientation is about 2.5 times noisier because the baseline is half as wide.
- The typical head-versus-torso difference (15 degrees) is of the same size as the head noise (12.5 degrees).
  A single-frame head turn relative to the torso is therefore barely distinguishable from noise.
  Any scanning-style metric needs temporal smoothing and an explicit uncertainty gate.

Assumptions that are not verified: Gaussian shape, the 2.5 conversion factor, independence of the two joints,
and the calibration of `p90_mae_cm` itself. There is no ground truth in the dataset. Correlated joint errors would make the true angular noise smaller.

SkillCorner's own gate (`src/features/pose_orientation.py`: error at most 15 cm, width 0.15 to 0.60 m) passes 98.7% of sample poses.
A hard gate therefore removes almost nothing and hides the 2.5-fold difference between torso and head. This is the gap `blindside` fills.

## 5. Consistency check without ground truth (sample)

Hip facing direction versus XY velocity heading. Convention: left-to-right axis rotated by +90 degrees.

| Speed band | n | Median abs. difference, hips | Shoulders | Share above 90 deg |
|---|---|---|---|---|
| 0 to 2 m/s | 1609 | 34.3 deg | 35.6 deg | 14% |
| 2 to 4 m/s | 2118 | 14.0 deg | 18.9 deg | 10% |
| 4 to 6 m/s | 903 | 11.4 deg | 14.3 deg | 3% |
| above 6 m/s | 176 | 9.1 deg | 9.1 deg | 0% |

The difference shrinks monotonically with speed and no front/back flips occur at sprint speed.
The facing convention is correct and the orientation carries real signal. At low speed players face away from their movement direction, as expected.

## 6. Blind-side feasibility

- Engagements inside the sample: 2. Both have pose for presser and carrier at engagement start. This is anecdotal.
- With a head angular SD of about 12 degrees and a visual field half-angle near 100 degrees, the probability that a presser is outside
  the carrier's field of view will be close to 0 or 1 for most geometries and uncertain only near the boundary.
  A probabilistic blind-side label is therefore workable. This is an expectation, not a measurement.

## 7. Open: full-match coverage (to be measured)

Not yet known: the share of the 1961 engagements with pose for both players at engagement start and during the approach.
Players near the ball are in frame more often than the one-third average suggests, so the usable share should be well above one third.
At one third the count would still be about 650, which clears the gate of 300.

## 8. Recommendation

**Go**, conditional on section 7. Sample size is not the binding constraint. The binding constraints are the unverified error model
and the fact that pose exists for only two matches, which limits any claim to these two fixtures.

Changes to the plan that follow from the data:
1. Report torso and head separately. Torso is the robust channel, head is the informative but noisy one.
2. Use SkillCorner's own outcome flags as outcome candidates in the pre-registration. They are documented and judges know them.
3. Train the XY baseline on the 18 non-pose matches.
4. Position the method against SkillCorner's hard-gate helper: same inputs, uncertainty instead of a threshold.
