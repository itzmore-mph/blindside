# PLAN.md

Phase plan with gates and ready-to-paste Claude Code prompts.
Budget assumption: 8 to 10 hours per week, about 45 to 50 hours in total.

| Phase | Dates | Hours | Gate |
|---|---|---|---|
| 0 Feasibility | 10 to 15 Nov | 3 | Passed on 10 Nov: 1,879 of 1,961 engagements have pose for both players (docs/FEASIBILITY.md section 7). Manual tasks still open |
| 1 Orientation and uncertainty | 16 to 22 Nov | 10 | Passed on 10 Nov: 100 tests green, velocity check plotted (docs/FEASIBILITY.md section 9) |
| 2 Reliability map | 23 to 29 Nov | 8 | Passed on 10 Nov: composite figure `docs/figures/reliability_composite.png`, findings in docs/RELIABILITY.md section 3 |
| 3 Engagements and baseline | 30 Nov to 6 Dec | 10 | Passed on 10 Nov: pre-registration committed (fe7ad6e), XY baseline scored (docs/BASELINE.md) |
| 4 Incremental value | 7 to 11 Dec | 6 | Passed on 11 Nov: no detectable increment, log loss difference minus 0.0002 under both bounds, 95% intervals within minus 0.0004 to 0.0000 (docs/INCREMENTAL.md) |
| 5 Packaging | 12 to 16 Dec | 8 | Engineering passed on 11 Nov: clean-clone run reproduces both README figures byte-identically, README drafted. Video and submission open (manual tasks) |
| Buffer | 17 to 18 Dec | 0 | Submit on 17 Dec |

Cut rule: if phase 0 or phase 3 shows too few usable engagements, drop phase 4 and submit deliverables 1 and 2
with the blind-side metric shown descriptively on case examples.

---

## Phase 0: Feasibility (gate: go / no-go)

Manual tasks for Moritz, status on 10 November 2026:
- Discord question 1, dynamic events for all matches: **answered by the data.** All 20 matches have dynamic events (docs/FEASIBILITY.md section 1).
- Discord question 2, multi-GB download in the full path: **treated as allowed, not explicitly confirmed.** The rules require a repository
  without the data that works out of the box, so a download step is implied. Moritz found nothing to the contrary in the rules or on Discord.
  The fast path on the sample phase stays as the safeguard.
- Discord question 3, figures inside the video: **not confirmed.** The two-figure limit is stated for the README only.
- AI assistance: **treated as allowed, not explicitly confirmed.** The rules do not address it and Moritz found nothing to the contrary on Discord.
  It is disclosed through the commit trailer.
- Employer permissions (side-activity and IP clauses): **settled.** Checked by Moritz on 10 November 2026: no clause applies.
  The project is done in private time on private equipment, with open data only. The rules make the participant responsible for employer permissions.
- Paris on 17 February 2027 in person: **feasible**, with annual leave. Remote talks are not accepted.
- Solo or pair: **solo.**

Status: done. Schema, event structure, joint error, the sample-based checks and the full-match coverage count
are in docs/FEASIBILITY.md. Result: go, the cut rule does not apply.

Prompt:

```
Read CLAUDE.md and docs/FEASIBILITY.md. Scaffold the repo (uv, pyproject, ruff, pytest, src layout, MIT LICENSE.md, .gitignore with data/).
Clone github.com/SkillCorner/opendata into ../opendata as a read-only reference.

Implement only src/blindside/data: a downloader for the two pose archives from Hugging Face (cache in data/raw, verify sha256 against
MANIFEST.json, resume on failure) and a streaming reader that never holds more than one frame in memory.

Then write scripts/coverage.py and append its results to docs/FEASIBILITY.md section 7:
1. Share of player-frames with joints, per match and per period.
2. For every on-ball engagement in both pose matches: does the presser have pose at frame_start, does the carrier, do both
   (allow a window of plus or minus 2 pose frames). Break down by event_subtype and by pitch third.
3. The same for engagement start minus 0.5 s and minus 1.0 s, because the blind-side feature needs the approach, not only the contact.
4. Distribution of p90_mae_cm for shoulders, hips and ears across both full matches, compared with the sample values in section 4.

End with the number of usable engagements and a go / no-go against the gate. State every number's source.
```

Gate: at least roughly 300 engagements with pose for both players across both matches. Expected from the sample: well above that. Below it, apply the cut rule.

## Phase 1: Orientation and uncertainty

```
Read CLAUDE.md and docs/FEASIBILITY.md.

Implement src/blindside/geometry:
- circular.py: circular mean, circular std, angular difference, with tests.
- orientation.py: torso, hip and head facing angle in the pitch plane from joint arrays, vectorised over frames and players.
  Resolve front/back ambiguity explicitly.
- uncertainty.py: Monte Carlo propagation of the per-joint error into angular uncertainty using the error model in CLAUDE.md,
  with the sigma conversion factor as a parameter.

Tests: synthetic skeletons with known facing angles, including rotated, mirrored and noisy cases.
Verify that the Monte Carlo output matches an analytic small-angle approximation for a simple two-joint case.

Then build a cached per-frame orientation table (parquet, git ignored) for the sample phase first,
full matches second. Reproduce the sample numbers in docs/FEASIBILITY.md section 4 and 5 as a regression test. Report runtime and memory for one full match.
```

Gate: tests green. Plot of hip orientation minus velocity heading as a function of speed looks sensible
(tight around zero at sprint speed). If not, debug the axis conventions before moving on.

Status: done. Results and the figure are in docs/FEASIBILITY.md section 9. Above 6 m/s the hip facing lies within 90 degrees
of the direction of travel in 99.7% of poses, with a median absolute difference of 12 degrees and no bias.
One full match takes about 105 s and 1.0 GB peak memory (86 s before the lower-bound columns were added). Open point for Phase 2: the Monte Carlo SD is about 3.5 times larger
than the frame-to-frame change of the hip angle allows for independent noise (section 9.7 and 9.8).

## Phase 2: Reliability map

```
Read CLAUDE.md. Using the cached orientation tables for both pose matches, build src/blindside/analysis/reliability.py and produce:
1. Angular uncertainty by pitch zone (mplsoccer heatmap), separately for torso, hips and head.
2. Angular uncertainty versus player speed and versus distance from the main camera touchline.
3. Hip orientation versus velocity heading agreement by speed band, as the no-ground-truth consistency check.
4. Share of player-frames passing a usability threshold (for example circular std below 20 degrees), per joint group,
   plus the sensitivity of that share to the sigma conversion factor (2.0, 2.5, 3.0).
5. Misalignment between XY position and pose centroid, as a distribution.

Write a short findings section in docs/RELIABILITY.md: what is reliable, what is not, and what an analyst should
therefore not build on this data. Label assumptions.
```

Gate: one composite reliability figure that is a candidate for README figure 1. All other plots go to docs/RELIABILITY.md.

Additional task: bracket the angular uncertainty instead of reporting one number.

1. Lower bound (jitter): for each segment, estimate high-frequency noise as the residual from a local
   quadratic fit over a short window (test 0.2 s and 0.4 s), on the unwrapped angle. Convert to an SD.
2. Upper bound: the Monte Carlo SD under independence (existing).
3. Report the autocorrelation of the jitter residuals at lags of 1, 5 and 25 frames, and the left/right
   correlation that would reconcile both bounds, using the existing correlation parameter.
4. Break the bracket down by segment, speed band and distance to the camera.
5. State clearly that the true angular error lies between the bounds and cannot be identified without ground truth.

The camera side is not given in the data. Infer it from image_corners_projection and verify the inference
on a few frames before using it. If it cannot be inferred reliably, drop the camera-distance breakdown and say so.

Carry both bounds forward: every downstream quantity that uses angular uncertainty (in particular p_blind_side
in Phase 4) is computed under the lower and the upper bound.

Status of the additional task: done on 10 Nov, docs/RELIABILITY.md section 1. Bracket of the angular SD (lower to upper):
torso 1.4 to 5.1 degrees, hips 2.0 to 12.0, head 2.0 to 13.2. Camera side inferred (negative y in both matches) and verified.
The orientation table carries `{segment}_sd_deg` (upper) and `{segment}_sd_low_deg` (lower).

Status of the main task: done on 10 Nov, docs/RELIABILITY.md sections 2 and 3, `scripts/reliability_map.py`.
Usable share at 20 degrees under the upper bound: torso 99.2%, hips 93.7%, head 87.3% (72 to 93% across sigma factors), above 99.8% under the lower bound.
The published joint error is flat across the pitch while the jitter rises by up to 55% towards the far touchline.
XY to pose offset: median 6.8 cm. The composite figure is the candidate for README figure 1.

Open points carried into Phase 3:
- `../opendata` holds the tracking files as Git LFS pointers and `git lfs` is not installed. The XY baseline features
  "defenders within 5 and 10 meters" and "closing speed" need tracking for all 20 matches (about 2 GB).
- `speed_avg`, `angle_of_engagement` and `interplayer_distance` (without `_start`) are measured over the engagement, not at its start.
  The pre-registration must restrict the baseline to `_start` columns or to values computed from tracking up to `frame_start`.

## Phase 3: Engagements and XY baseline

```
Read CLAUDE.md and docs/FEASIBILITY.md.
First, draft docs/PREREGISTRATION.md and stop for my review. It must fix: unit of analysis (one on-ball engagement),
inclusion criteria, the primary outcome, the XY-only baseline feature set, the pose feature set, the model class,
the evaluation metric and the resampling scheme. Propose two candidate primary outcomes with pros and cons
(for example SkillCorner's own force_backward flag, possession loss of the associated possession, or possession_epv_total)
and recommend one. Do not look at outcome distributions conditional on pose features while drafting.

After I approve: implement src/blindside/features (engagement extraction, XY baseline features: distance, closing speed,
approach angle, defenders within 5 and 10 meters, pitch zone, SkillCorner pressure and difficulty labels at start).
Fit the XY baseline on the 18 matches without body pose (about 15,600 engagements) and evaluate it on the two pose matches,
so the baseline is as strong as the data allows and the pose matches stay untouched for the incremental test.
Report out-of-sample performance with cluster bootstrap confidence intervals and leave-one-match-out.
```

Gate: pre-registration committed before any outcome modelling with pose features.

Status: done on 10 Nov. Primary outcome: possession loss of the pressed possession. Test set: 1,875 engagements in 1,382 possessions,
217 events, so the cut rule does not apply. XY baseline on the pose matches: log loss 0.310 against 0.361 for the constant prediction
(difference minus 0.050, 95% interval minus 0.071 to minus 0.030), ROC AUC 0.76. Tracking is fetched by `python -m blindside.data.tracking`.
New dependency: scikit-learn. Phase 4 starts from `data/cache/baseline/test_predictions.parquet` and docs/BASELINE.md section 3.

## Phase 4: Incremental value of pose

```
Read CLAUDE.md and docs/PREREGISTRATION.md. Implement the pose features:
- p_blind_side: probability that the presser lies outside the carrier's visual field at engagement start,
  computed from the Monte Carlo head and torso orientation samples (field of view half-angle as a parameter, default 100 degrees).
  Compute it twice, under the lower and the upper bound of the angular uncertainty
  (`blindside.geometry.uncertainty.bound_correlation`, docs/RELIABILITY.md section 1), and report both.
- presser body alignment: angle between presser hip orientation and the direction to the carrier, with uncertainty.
- carrier openness: carrier torso orientation relative to the attacking direction.

Run exactly the pre-registered comparison: baseline versus baseline plus pose features. Report the performance difference
with a paired cluster bootstrap confidence interval, and the descriptive effect of p_blind_side on the primary outcome in bins.
Run the comparison under both uncertainty bounds. A result counts as robust only if it holds under both.
Repeat using only engagements that pass the usability threshold. Report all results including a null result.
Anything beyond the pre-registration goes into a clearly labelled exploratory section.
```

Gate: one honest headline sentence, supported by the interval. A null result is acceptable and is reported as such.

Status: done on 11 Nov. Null result, reported as pre-registered. Model 1 (baseline plus four pose features) does not beat Model 0
(recalibrated baseline) under either uncertainty bound: log loss difference minus 0.0002 (95% intervals within minus 0.0004 to 0.0000).
None of the ten robustness checks and neither binary secondary outcome shows an increment. Descriptively, possession loss is
13.2% in the blind bin of `p_blind_side_head` against 10.8% in the in-view bin (lower bound), plus 2.4 points (minus 1.2 to plus 5.8),
and plus 1.1 (minus 2.2 to plus 4.4) after the XY baseline. Three additions to the pre-registration were committed on 11 Nov
before any pose feature was computed (section 12). Deliverable 3 is implemented and geometrically verified, not validated
as an explanatory variable. Reproduce with `uv run python scripts/incremental_value.py --write-docs`.

Exploratory addition on 11 Nov, no outcome used (docs/INCREMENTAL.md section 1.10): the XY proxy for the blind side
(presser more than 100 degrees off the carrier's direction of travel) against the pose label (`p_blind_side` above 0.5).
The proxy calls 45.5% of engagements blind, head pose 20.5%. Cohen's kappa is 0.28 (0.24 to 0.31) overall and rises with
carrier speed from 0.09 below 2 m/s to 0.58 above 6 m/s (torso label: 0.10 to 0.77). This became the lead of the submission.

Open points carried into Phase 5:
- README figure 2 is not drawn yet. Decided on 11 Nov: Cohen's kappa of the XY proxy against the pose label by carrier speed
  band, plus the share of engagements called blind by each label (docs/INCREMENTAL.md section 1.10). It replaces the earlier
  candidate, possession loss per bin of `p_blind_side_head` (section 1.6), which stays in docs/.
- The README leads with who counts as pressed from the blind side (pose against the XY proxy, by carrier speed), reports the
  pre-registered null plainly with its interval and resolution, and presents the frozen design as ready to rerun on more matches.
  The reliability work is what makes it credible.
- Figure 2 needs the engagements of both pose matches. The fast path runs on one sample phase of 12.3 s and cannot compute it.
  Open decision for Phase 5: what `blindside reproduce --fast` draws for figure 2. No derived data table may be committed.
- `load_tables` and `inclusion_flow` live in `scripts/xy_baseline.py` and are imported by `scripts/incremental_value.py`.
  Move them into the package when the CLI is built.

## Phase 5: Packaging

```
Read CLAUDE.md. Implement the CLI: `blindside reproduce --fast` (sample phase, under five minutes) and `blindside reproduce`
(full download with cache and progress, checksum verification, clear error messages on network failure). Both regenerate the two README figures.

Draw README figure 2 with the style module, from `blindside.analysis.agreement.agreement_table` (docs/INCREMENTAL.md section 1.10):
left panel Cohen's kappa of the XY proxy against the pose label per carrier speed band, head and torso, with the cluster
bootstrap intervals. Right panel the share of engagements called blind by each label (pose head, pose torso, XY proxy)
per speed band. No outcome column in the figure. Figure 1 is the reliability composite (candidate). Before building the fast path,
ask Moritz what it draws for figure 2: the sample phase cannot compute it (docs/PLAN.md, open points of Phase 4).

Draft README.md: at most 1000 words, exactly two figures. Structure mapped to the judging criteria, in this order:
1. The defensive question and who it helps: who counts as pressed from the blind side. XY tracking alone calls about twice
   as many engagements blind as body pose does, and agrees with pose beyond chance only when the carrier runs.
2. Method in plain language: orientation with an uncertainty bracket, the reliability map, `p_blind_side` under both bounds.
3. The pre-registered test, stated plainly as a null result: log loss difference with its interval, and the resolution
   (what effect size 217 events could not have told from zero). No wording that suggests blind-side pressure predicts
   possession loss. Kappa is agreement, not accuracy: say that it cannot tell which label is right.
4. The frozen design as the reusable result: pre-registration, baseline, features and one script, ready to rerun unchanged
   when more matches with body pose are published.
5. Limitations, how to reproduce, how to reuse the package (10-line example), third-party libraries and licences.
Take every number from docs/INCREMENTAL.md and CLAUDE.md. Label section 1.10 results as exploratory.
Add a word-count check to CI that fails above 1000 words or more than two figures or tables.
Add LICENSE.md (MIT). Add a GitHub Actions workflow that runs tests and the fast path on a clean runner.

Then run a clean-clone test in a fresh directory on a machine state without caches and fix everything that breaks.
```

Status: engineering done on 11 Nov, manual tasks open.
- Decisions by Moritz on 11 Nov: `reproduce --fast` writes stamped stand-ins to `outputs/fast/` (the reliability composite on the
  sample and a geometry panel of its two engagements). The full `reproduce` draws both figures from the two pose matches,
  `--with-test` adds the baseline on 18 matches and the pre-registered comparison.
- A clean clone has no `../opendata`: events, match metadata and the sample phase are now downloaded at the pinned commit and
  verified by sha256 (`blindside.data.opendata`). Without this the package did not run out of the box.
- `load_tables`, `inclusion_flow`, `attach_pose`, `compute` and `proxy_agreement` moved to `blindside.analysis.pipeline`.
  Rerunning the three doc scripts afterwards left `docs/` byte-identical.
- Clean-clone test (fresh directory, no opendata clone, no data, fresh uv cache, macOS): 222 tests pass and 5 skip (they need
  cached full-match tables or a local sample), `reproduce --fast` takes 50 s including a 97 MB download, `reproduce` takes
  13 minutes including 1.4 GB of downloads (orientation tables 107 and 110 s). Both figures are byte-identical to the committed ones.
  `reproduce --with-test` on the cached data takes 170 s and prints minus 0.0002 (minus 0.0004 to 0.0000) under both bounds.
- README: 900 words by the conservative count of `scripts/check_readme.py`, two figures, no table.
- Not yet verified: the GitHub Actions workflow has not run, the branch is not pushed.

Manual tasks for Moritz: review the README draft, add the video link to it, record the one-minute video (script: problem 10 s, pose against XY by carrier speed 20 s,
the null result and its resolution 15 s, rerun and reuse 15 s), upload to YouTube,
make the repo public, submit on Pretalx on 17 Dec.
