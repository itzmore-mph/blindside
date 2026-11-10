# FEASIBILITY.md

Phase 0 findings for `blindside`. Date: 10 November 2026.
Source: `github.com/SkillCorner/opendata` at commit 4340d27, inspected directly.

Scope of sections 1 to 6: repository structure, dynamic events of all 20 matches, and the pose sample committed in the
opendata repo (`data/bodypose/sample_1925299_phase406.jsonl.gz`, 309 pose frames, 12.3 seconds, one Brisbane Roar quick break).
Everything marked "sample" rests on 12 seconds of one match.
Section 7 was added after reading both full pose archives. It re-measures coverage and joint error on the full files.
The angular uncertainty (section 4) and the velocity check (section 5) are still sample-only and are re-measured in Phase 1.

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
- Every in-period frame lists the full squads: 32 players in 1925299, 30 in 1996435. In the sample 50% of player-frames carry joints.
  Across the full matches the share is 31.4% and 36.4% (section 7.3).
- Full files only: each match has 47 frames outside the periods (`period` is null) with an empty `player_data` list.
  Pose frames start at 0. Period 1 starts at pose frame 25, which is tracking frame 10.
- Full files only: the zip archives carry a macOS metadata member (`__MACOSX/._{match_id}.jsonl`) next to the data.
  A reader must select the real `.jsonl` member explicitly.
- Full files only: every resolved pose has all 29 joints (0 incomplete poses in 3.1 million).

### Event coordinates (verified on both pose matches)

- Dynamic event coordinates are direction-normalised: the acting team always attacks towards +x.
  For rows with `attacking_side == right_to_left`, pitch coordinates are `(-x, -y)` of the event coordinates.
  After this mirroring, the presser's event position matches the tracking position in the pose file to within 1.3 cm for all 1961 engagements.
- `attacking_side` is the direction of the acting player's own team (100% of 10,393 event rows, checked against `home_team_side` in match.json).
- `third_start` follows the same normalisation. On an on-ball engagement, `defensive_third` is the presser's own third.
- `player_in_possession_x_start` and `player_in_possession_y_start` are empty on on-ball engagement rows.
  The carrier position at engagement start must come from tracking or from the pose file.

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

## 7. Full-match coverage (measured on both full pose archives)

Question: for how many of the 1961 on-ball engagements does body pose exist for presser and carrier,
at engagement start and during the approach? Reproduce with `uv run python scripts/coverage.py --write-docs`.
The block below is generated by that script.

<!-- coverage:start (generated by scripts/coverage.py, do not edit by hand) -->
### 7.1 Sources and definitions

- Pose: `raw/1925299.jsonl.zip` (sha256 `0415665098d5...`); `raw/1996435.jsonl.zip` (sha256 `64bcab2be977...`) from `huggingface.co/datasets/SkillCorner/opendata-bodypose` at revision `a62e1ec`, read in full with `blindside.data.iter_pose_frames`.
- Engagements: `{match_id}_dynamic_events.csv` rows with `event_type == on_ball_engagement`, opendata commit `4340d27`. Presser = `player_id`, carrier = `player_in_possession_id`.
- All numbers in this section are produced by `scripts/coverage.py`.
- **Has pose**: `joints` is not null for that player in that pose frame.
- **Target pose frame**: `2.5 * frame_start - 25 * offset_s`. A player has pose "at" a time if any pose frame within plus or minus 2 frames (0.08 s) has joints.
- **Both**: at least one pose frame in that window in which presser and carrier have joints simultaneously. This is the definition used for the gate.
- A target before the start of the period counts as no pose.
- Definitions were fixed in the script before the full files were read. No outcome column was loaded.

### 7.2 Checks

- Match 1925299: frames, frames with pose, player-frames, player-frames with pose and joint observations are identical to the counts on the Hugging Face dataset card.
- Match 1925299: 1,540,715 poses, 0 of them with fewer or more than 29 joints. Pose frames 0 to 153,251. Runtime of the pass 82 s.
- Match 1996435: 1,571,830 poses, 0 of them with fewer or more than 29 joints. Pose frames 0 to 144,051. Runtime of the pass 85 s.
- Frame alignment and event coordinates: the presser's `x_start`, `y_start` were compared with the tracking position stored in the pose file at `2.5 * frame_start` (n = 1,961). As given, 46.9% lie within 0.5 m. After mirroring both axes for rows with `attacking_side == right_to_left`, the median distance is 0.000 m and the maximum 0.013 m. Event coordinates are therefore direction-normalised (the acting team attacks towards +x), and `pose_frame = 2.5 * tracking_frame` holds on the full files.
- Carrier identity: `interplayer_distance_start` versus the presser-to-carrier distance from the pose-file positions (n = 1,961): median absolute difference 0.00 m, 95th percentile 0.01 m.
- `third_start` equals the third of the normalised `x_start` in 100.0% of engagements (pitch length from match.json). Thirds are therefore relative to the pressing team's own attacking direction: `defensive_third` is the presser's own third.

### 7.3 Share of player-frames with joints

| Match | Period | Frames | Frames with any pose | Player-frames | With joints | Share | Share among `is_detected` |
|---|---|---|---|---|---|---|---|
| 1925299 | 1 | 75,578 | 71.4% | 2,418,496 | 738,555 | 30.5% | 99.3% |
| 1925299 | 2 | 77,627 | 68.0% | 2,484,064 | 802,160 | 32.3% | 99.2% |
| 1925299 | outside periods | 47 | 0.0% | 0 | 0 | n/a | n/a |
| **1925299** | **all** | **153,252** | **69.7%** | **4,902,560** | **1,540,715** | **31.4%** | **99.3%** |
| 1996435 | 1 | 70,028 | 70.9% | 2,100,840 | 819,897 | 39.0% | 99.2% |
| 1996435 | 2 | 73,977 | 62.1% | 2,219,310 | 751,933 | 33.9% | 99.4% |
| 1996435 | outside periods | 47 | 0.0% | 0 | 0 | n/a | n/a |
| **1996435** | **all** | **144,052** | **66.4%** | **4,320,150** | **1,571,830** | **36.4%** | **99.3%** |

Player-frames count every listed player, including substitutes and players off camera. The last column restricts the denominator to `is_detected` players.

### 7.4 Engagements with pose at `frame_start`

|  | Engagements | Presser has pose | Carrier has pose | Both |
|---|---|---|---|---|
| 1925299 | 936 | 906 (96.8%) | 904 (96.6%) | 892 (95.3%) |
| 1996435 | 1,025 | 993 (96.9%) | 995 (97.1%) | 987 (96.3%) |
| **Total** | **1,961** | **1,899 (96.8%)** | **1,899 (96.8%)** | **1,879 (95.8%)** |

By `event_subtype`, both matches pooled:

|  | Engagements | Presser has pose | Carrier has pose | Both |
|---|---|---|---|---|
| pressure | 507 | 494 (97.4%) | 497 (98.0%) | 492 (97.0%) |
| recovery_press | 473 | 459 (97.0%) | 458 (96.8%) | 452 (95.6%) |
| pressing | 444 | 425 (95.7%) | 425 (95.7%) | 420 (94.6%) |
| other | 373 | 361 (96.8%) | 361 (96.8%) | 358 (96.0%) |
| counter_press | 164 | 160 (97.6%) | 158 (96.3%) | 157 (95.7%) |

By `third_start`, both matches pooled:

|  | Engagements | Presser has pose | Carrier has pose | Both |
|---|---|---|---|---|
| defensive_third | 738 | 725 (98.2%) | 719 (97.4%) | 717 (97.2%) |
| middle_third | 829 | 788 (95.1%) | 794 (95.8%) | 776 (93.6%) |
| attacking_third | 394 | 386 (98.0%) | 386 (98.0%) | 386 (98.0%) |

Sensitivity of the "both" count at `frame_start`, both matches pooled:

| Definition | Engagements |
|---|---|
| Both in the same pose frame, window plus or minus 2 (gate definition) | 1,879 (95.8%) |
| Each somewhere in the window, not necessarily the same frame | 1,879 (95.8%) |
| Both in the nearest pose frame only (no window) | 1,870 (95.4%) |
| Gate definition, and all six shoulder, hip and ear joints of both players at most 15 cm | 1,727 (88.1%) |

### 7.5 Coverage during the approach

| Time | Presser has pose | Carrier has pose | Both, pooled | Both, 1925299 | Both, 1996435 |
|---|---|---|---|---|---|
| `frame_start` | 1,899 (96.8%) | 1,899 (96.8%) | 1,879 (95.8%) | 892 (95.3%) | 987 (96.3%) |
| minus 0.5 s | 1,864 (95.1%) | 1,860 (94.8%) | 1,829 (93.3%) | 860 (91.9%) | 969 (94.5%) |
| minus 1 s | 1,822 (92.9%) | 1,803 (91.9%) | 1,762 (89.9%) | 820 (87.6%) | 942 (91.9%) |
| **Both at all three times** |  |  | **1,748 (89.1%)** | 813 (86.9%) | 935 (91.2%) |

Denominator in every row: all 1,961 engagements.

### 7.6 `p90_mae_cm` on the full matches versus the sample

| Joint | Sample median (section 4) | Median, 1925299 | Median, 1996435 | 10th pct, pooled | Median, pooled | 90th pct, pooled | Share above 15 cm |
|---|---|---|---|---|---|---|---|
| lShoulder | 5.6 to 5.7 | 5.7 | 5.6 | 5.4 | 5.6 | 8.2 | 1.8% |
| rShoulder | 5.6 to 5.7 | 5.7 | 5.7 | 5.4 | 5.7 | 8.1 | 1.8% |
| lHip | 7.3 to 7.8 | 7.3 | 7.3 | 6.5 | 7.3 | 11.4 | 2.8% |
| rHip | 7.3 to 7.8 | 7.2 | 7.2 | 6.4 | 7.2 | 11.5 | 2.5% |
| lEar | 6.7 to 6.8 | 6.8 | 6.8 | 6.0 | 6.8 | 10.9 | 3.7% |
| rEar | 6.7 to 6.8 | 6.7 | 6.7 | 5.7 | 6.7 | 10.9 | 3.6% |

Values in cm. Pooled medians of the other joints: nose 7.3, neck 5.0, lEye 6.9, rEye 6.8, lElbow 8.3, rElbow 8.4, lWrist 13.0, rWrist 13.3, lThumb 15.9, rThumb 16.2, lPinky 17.5, rPinky 17.2, midHip 10.3, lKnee 10.1, rKnee 10.1, lAnkle 12.7, rAnkle 12.7, lHeel 14.9, rHeel 15.1, lBigToe 15.3, rBigToe 15.4, lSmallToe 15.2, rSmallToe 15.2.

### 7.7 Usable engagements and gate

- **Usable engagements (both players with pose at `frame_start`): 1,879 of 1,961 (95.8%)**, 1925299: 892, 1996435: 987. They belong to 1,384 distinct possessions, which is the number of clusters for any bootstrap.
- Stricter count for the approach feature (both players with pose at `frame_start`, minus 0.5 s and minus 1.0 s): 1,748 engagements in 1,291 possessions. This also clears the gate.
- Gate from docs/PLAN.md: at least roughly 300 engagements with pose for both players across both matches. **Result: Go.**
<!-- coverage:end -->

## 8. Recommendation

**Go.** The gate is passed: 1,879 of 1,961 engagements (95.8%) have pose for presser and carrier at engagement start,
against a gate of roughly 300. The cut rule does not apply and Phase 4 stays in scope.

What section 7 shows:
- Pose is missing almost only when a player is not detected. 99.3% of `is_detected` player-frames carry joints.
  The overall share of 31 to 36% reflects players off camera, and engagement participants are on camera.
- Coverage is similar across engagement subtypes (94.6% to 97.0%) and pitch thirds (93.6% to 98.0%).
- Coverage decays moderately before the engagement: 93.3% at minus 0.5 s, 89.9% at minus 1.0 s.
  1,748 engagements (89.1%) are covered at all three time points.
- The per-joint error on the full matches matches the sample: medians for shoulders, hips and ears agree to 0.1 cm.
- The effective sample for inference is the number of possessions, not engagements: 1,384 clusters (1,291 for the strict count).

What it does not show:
- Whether the 4% of engagements without pose differ systematically in outcome. Outcome columns were not read in this phase.
- Whether the angular uncertainty of section 4 holds on the full matches. Only the joint error was re-measured, not the angles.

Sample size is not the binding constraint. The binding constraints are the unverified error model
and the fact that pose exists for only two matches, which limits any claim to these two fixtures.

Changes to the plan that follow from the data:
1. Report torso and head separately. Torso is the robust channel, head is the informative but noisy one.
2. Use SkillCorner's own outcome flags as outcome candidates in the pre-registration. They are documented and judges know them.
3. Train the XY baseline on the 18 non-pose matches.
4. Position the method against SkillCorner's hard-gate helper: same inputs, uncertainty instead of a threshold.
5. Mirror event coordinates for `right_to_left` rows before joining events with pose or tracking (section 2).
6. Decide in the pre-registration whether inclusion requires pose at `frame_start` only (1,879) or during the whole approach (1,748).
