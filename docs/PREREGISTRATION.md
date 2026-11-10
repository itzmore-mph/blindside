# PREREGISTRATION.md

**Status: approved by Moritz on 10 November 2026 and binding from the commit that adds this file.**
It was committed before any outcome was modelled with pose features and before `p_blind_side` was computed for any engagement.
It is not changed afterwards. Any later deviation is listed in section 12 with its date and reason.

Decisions taken by Moritz at the review on 10 November 2026: possession loss as the primary outcome (section 4),
`scikit-learn` as a new dependency (section 7), a tracking downloader inside the package (section 5).

## 1. Question and hypothesis

Does body orientation from broadcast pose explain the outcome of an on-ball engagement beyond what XY tracking already explains?

- **H1 (incremental value).** Adding the pose features of section 6 to the XY baseline of section 5 lowers the out-of-sample
  log loss for the primary outcome on the two pose matches.
- **H2 (direction, descriptive).** Engagements with a high probability that the presser is outside the carrier's visual field
  (`p_blind_side`) end in the primary outcome more often than engagements with a low one.

A null result is a valid result and is reported as such.

## 2. What has been looked at before this document

Stated so that a reader can judge how blind the design is.

- Phases 0 to 2 read no outcome column of the two pose matches, with one exception: docs/FEASIBILITY.md section 3 lists the marginal
  counts of possession `end_type` for match 1925299 (pass 912, possession_loss 74, shot 30, clearance 24, foul_suffered 11).
- While drafting, marginal outcome rates were computed on the **18 matches without body pose** only (section 4). They hold no pose.
- No outcome has been tabulated or plotted against any pose feature, in any match. `p_blind_side` has not been computed for any engagement.
- The reliability work (docs/RELIABILITY.md) used poses of all players in all frames and no event column.

## 3. Unit of analysis and inclusion

**Unit:** one on-ball engagement, a row of `{match_id}_dynamic_events.csv` with `event_type == "on_ball_engagement"`.
Presser = `player_id`, carrier = `player_in_possession_id`.

**Reference time t0:** the engagement's `frame_start` (10 fps tracking clock, pose frame `2.5 * frame_start`).
Every feature uses information up to and including t0. Section 5 names the one exception.

**Test set (pose matches 1925299 and 1996435).** An engagement is included if all of the following hold:
1. Presser and carrier both have joints in the same pose frame within plus or minus 2 pose frames of t0.
   This is the gate definition of docs/FEASIBILITY.md section 7.1 and gives 1,879 of 1,961 engagements in 1,384 possessions.
2. The primary outcome is defined (`associated_player_possession_end_type` is not `unknown`).
3. Presser and carrier have a tracking position at t0 and at t0 minus 0.5 s (needed for the closing speed).

No engagement is excluded for high angular uncertainty in the primary analysis. Uncertainty enters through `p_blind_side` itself.
The usability subset is a pre-registered robustness check (section 10).

**Training set for the XY baseline:** all on-ball engagements of the 18 matches without pose that meet criteria 2 and 3
(15,484 engagements in 11,499 possessions before those criteria).

**Clusters:** the pressed possession, `(match_id, associated_player_possession_event_id)`. 28% of possessions carry more than one engagement.

**Known property, kept on purpose:** in the 18 training matches 57% of engagements start before the pressed possession starts
(median 0.2 s earlier). The "carrier" is then a receiver who is being closed down. These engagements stay in,
because pressure that arrives with the ball is where a blind side matters most. Section 10 reports both groups separately.

## 4. Primary outcome

**Primary outcome: possession loss of the pressed possession**, `associated_player_possession_end_type == "possession_loss"`,
coded 1, every other defined end type coded 0. Decided by Moritz on 10 November 2026, before any outcome was set against a pose feature.

Two candidates were weighed. The comparison is kept as the record of that choice.

Marginal rates on the 18 non-pose matches (engagement level). Expected events = rate times 1,879 and is only a planning number.

| Candidate | Definition | Rate | By match | Expected events in the test set |
|---|---|---|---|---|
| **A. Possession loss** | `associated_player_possession_end_type == "possession_loss"`: the carrier loses the ball | 13.8% | 9.9% to 19.3% | about 260 |
| B. Forced backward | SkillCorner flag `force_backward == True` | 5.9% | not computed | about 110 |

| | A. Possession loss | B. Forced backward |
|---|---|---|
| Football meaning | The press won the ball or made the carrier lose it. Every coach reads it the same way | The press took away the forward option. A real defensive success that does not need a turnover |
| Link to the blind side | Direct: a carrier who does not see the presser is dispossessed | Unclear direction: a carrier who does not see the presser may simply play on forward |
| Objectivity | Event fact | Model-derived flag, definition not published in the open data |
| Power | About 2.3 times the events of B | About 110 events for four pose features is thin |
| Weakness | A property of the possession: simultaneous engagements share the label (handled by clustering). Losses also happen for reasons unrelated to this presser | Rare. Attributed to the single engagement, which is cleaner |

Not proposed as primary:
- `possession_epv_total` (continuous, 98.6% populated): more statistical power, but a model output, noisy around zero (median 0.000)
  and hard to explain in one sentence of a 1000 word README. Secondary.
- Turnover in the wide sense (possession loss or unsuccessful pass, 27.6%): the most events, but a failed pass is one step further
  from the presser. Secondary.
- `beaten_by_possession` (0.9%), `stop_possession_danger` (1.3%), `reduce_possession_danger` (3.4%): too rare.

**Secondary outcomes, fixed now, reported as secondary:** wide turnover, `force_backward` (candidate B),
`possession_epv_total`. No claim of the submission rests on them.

**Minimum events.** If the test set holds fewer than 100 events of the primary outcome, the incremental test is reported
as underpowered and the blind-side metric is presented descriptively, as CLAUDE.md foresees.

## 5. XY baseline features

Everything below is available without body pose. Event coordinates are mirrored for `attacking_side == right_to_left`
before any join (docs/FEASIBILITY.md section 2).

From the engagement row, all measured at engagement start:

| Feature | Source column |
|---|---|
| Distance presser to carrier | `interplayer_distance_start` |
| Presser goal side of the carrier | `goal_side_start` |
| Pitch zone | `third_start`, `channel_start`, `penalty_area_start` |
| Carrier distance and angle to goal | `player_targeted_distance_to_goal_start`, `player_targeted_angle_to_goal_start` |
| Carrier distance to the last defensive line | `delta_to_last_defensive_line_start` |
| Defenders and team mates within 5 m of the carrier | `n_player_targeted_opponents_within_5m_start`, `n_player_targeted_teammates_within_5m_start` |
| Engagement type | `event_subtype` |
| Defensive phase and game state | `team_out_of_possession_phase_type`, `game_state` |

From tracking, positions up to t0 only:

| Feature | Definition |
|---|---|
| Closing speed | Decrease of the presser to carrier distance over the 0.5 s before t0, in m/s |
| Presser speed, carrier speed | Central XY speed, smoothed as in `blindside.geometry.kinematics`, evaluated with data up to t0 |
| Approach angle to goal | Angle at the carrier between the direction to the presser and the direction to the goal the carrier attacks |
| **Approach angle to the carrier's heading** | Angle between the carrier's direction of travel and the direction to the presser. This is the XY stand-in for "from behind" |
| Defenders within 10 m of the carrier | Count of opponents of the carrier within 10 m at t0 |

From the pressed possession row:

| Feature | Source column |
|---|---|
| SkillCorner pressure label | `overall_pressure_start` (2.6% missing) |
| SkillCorner reception difficulty | `reception_difficulty_start` (4.6% missing) |

Why the heading feature is in the baseline: a moving player faces roughly where they run (docs/RELIABILITY.md section 2.4).
Without it the pose features would get credit for information that XY already holds. With it, the pose features can only add
what the body shows and the movement does not.

**The one exception to "up to t0".** The two SkillCorner labels are measured at possession start, which lies after t0 for 57% of
engagements (median 0.2 s). They are kept because docs/PLAN.md names them and because the leak favours the baseline,
which makes the test for pose harder, not easier. Section 10 repeats the comparison without them.

**Excluded because they are measured during or after the engagement:** `speed_avg`, `speed_difference`, `angle_of_engagement`,
`trajectory_angle`, `interplayer_distance_end`, `interplayer_distance_min`, every `_end` and `_gain` column, `duration`,
`pressing_chain` and its fields, `consecutive_on_ball_engagements`, `simultaneous_defensive_engagement_same_target`,
`end_type`, `lead_to_shot`, `lead_to_goal`, `team_possession_loss_in_phase`, all `xloss_` and `xshot_` columns and every outcome flag.

**Tracking source.** The five tracking features need the 10 fps tracking of all 20 matches (about 2 GB, Git LFS in the opendata repo).
A tracking downloader in `blindside.data` fetches the LFS files over HTTPS and checks size and sha256 against the pointer files,
the same pattern as the pose downloader. The tracking of the
pose matches is used for the test set too, not the positions inside the pose files, so that training and test features come from one source.

## 6. Pose features

Four features, all at t0, each computed twice: under the lower and under the upper bound of the angular uncertainty
(`bound_correlation(segment, "lower" | "upper")`, docs/RELIABILITY.md section 1).

| Feature | Definition |
|---|---|
| `p_blind_side_head` | Share of Monte Carlo samples of the carrier's head direction for which the presser lies more than 100 degrees off that direction |
| `p_blind_side_torso` | The same with the carrier's torso direction |
| `presser_alignment` | Circular mean over samples of the absolute angle between the presser's hip direction and the direction to the carrier, in degrees |
| `carrier_openness` | Mean over samples of the cosine of the angle between the carrier's torso direction and the carrier's attacking direction |

Fixed parameters: field of view half-angle 100 degrees, 200 Monte Carlo draws, sigma = p90 / 2.5, seed 0, pose frame nearest to
`2.5 * frame_start` in which both players have joints (within plus or minus 2). Positions are taken as exact. Only the orientation is sampled.
`p_blind_side_head` is the headline feature for H2.

## 7. Model class

**XY baseline.** L2-regularised logistic regression on the features of section 5. Numeric features standardised with training statistics,
categorical features one-hot encoded, missing numeric values imputed with the training median plus a missing indicator,
missing categories as their own level. The penalty strength is chosen by grouped 5-fold cross-validation (groups = matches)
on the 18 training matches, from ten values log-spaced between 0.001 and 100, by log loss. The fitted model is then frozen.
Its output for a test engagement is the baseline logit `z_xy`.

**Comparison on the pose matches.** Two logistic regressions, fitted on the same folds:
- Model 0: outcome on `z_xy` (intercept and slope). This recalibrates the baseline to the two pose matches.
- Model 1: outcome on `z_xy` plus the four pose features, same penalty family, penalty applied to the pose coefficients only,
  strength chosen inside each training fold by grouped cross-validation.

Model 0 exists so that Model 1 cannot win by recalibration alone.

A logistic model is chosen over gradient boosting because the test set has a few hundred events and four added features:
the increment should be readable as four coefficients. A gradient-boosted baseline is a robustness check (section 10).

**Dependency.** The models use `scikit-learn` (BSD 3-Clause, https://scikit-learn.org), added for this purpose.

## 8. Evaluation metric

- **Primary:** difference in mean log loss, Model 0 minus Model 1, on out-of-fold predictions for the test set. Positive means pose helps.
- Secondary: difference in ROC AUC and in Brier score, and the calibration slope of each model.
- For the baseline alone (Phase 3 report): log loss, AUC and Brier score of the frozen model on the two pose matches,
  against the constant-rate prediction.

## 9. Resampling scheme

- **Out-of-fold predictions:** grouped 10-fold cross-validation on the pooled test set, groups = possessions, stratified by match,
  repeated 20 times with seeds 0 to 19. The prediction of an engagement is the mean over repeats.
- **Confidence interval:** paired cluster bootstrap. 2,000 resamples of possessions with replacement, stratified by match.
  In each resample the metric difference is recomputed from the fixed out-of-fold predictions. Percentile 95% interval.
  This interval covers sampling of possessions. It does not cover the variability of refitting, and the write-up says so.
- **Baseline report (Phase 3):** the same cluster bootstrap for the frozen baseline's metrics, plus leave-one-match-out
  cross-validation over the 18 training matches as the estimate of its match-to-match spread.

**Decision rule.** H1 counts as supported only if the 95% interval of the log loss difference lies above zero
**under both uncertainty bounds**. If it does so under one bound only, the result is reported as dependent on the error model.
Otherwise the result is "no detectable increment", reported with its interval. No p-values are reported.

## 10. Pre-registered robustness checks

Each is reported next to the primary result, whatever it shows. None can replace it.

1. Leave-one-match-out on the pose matches: fit Models 0 and 1 on one, evaluate on the other, both directions.
2. Usability subset: only engagements in which the carrier's head and torso upper-bound SD and the presser's hip upper-bound SD are below 20 degrees.
3. Baseline without the two SkillCorner labels (the timing exception of section 5).
4. Field of view half-angle 90 and 110 degrees.
5. Sigma factor 2.0 and 3.0 for the upper bound.
6. Engagements that start before the possession starts versus at or after it.
7. Gradient-boosted baseline (`HistGradientBoostingClassifier`, default settings) in place of the logistic baseline.
8. Distance from the camera touchline added as a covariate to Model 0 and Model 1 (docs/RELIABILITY.md section 3.5).

## 11. Descriptive analysis for H2

Rate of the primary outcome in three bins of `p_blind_side_head`: below 0.1 (presser in view), 0.1 to 0.9 (uncertain), above 0.9 (blind side),
with cluster bootstrap 95% intervals, under both bounds, with the bin sizes. The same table for `p_blind_side_torso`.
The bin edges are fixed here. If a bin holds fewer than 50 engagements it is reported with its count and without an interval.

Everything else is exploratory and goes into a section with that title.

## 12. Deviations

No deviation from the design so far.

Clarifications of wording, recorded on 10 November 2026 while implementing the XY baseline and before any pose feature was computed.
None of them changes a feature set, the outcome, the model class, the metric or the resampling.

1. **Speeds (section 5).** "Central XY speed evaluated with data up to t0" is implemented as the mean velocity over the 0.5 s
   that end at t0 (position at t0 minus position 0.5 s earlier). A central difference centred on t0 would need positions after t0.
   The carrier's direction of travel uses the same displacement.
2. **Penalty grid (section 7).** The ten log-spaced values between 0.001 and 100 are values of scikit-learn's `C`,
   the inverse of the penalty strength.
3. **Defenders within 10 m (section 5).** The count includes the presser.
4. **Inclusion criterion 3.** The two positions must lie in the same period.
5. **Pose window (section 3).** "Within plus or minus 2 pose frames" means pose frames at most 2 away from `2.5 * frame_start`,
   which is a half frame for odd tracking frames. This reproduces the 1,879 engagements of docs/FEASIBILITY.md section 7.

Observation after the baseline was fitted (docs/BASELINE.md section 2): `team_out_of_possession_phase_type` may be assigned
with hindsight by the data provider. It stays in the baseline as registered. If it leaks, it favours the baseline.
