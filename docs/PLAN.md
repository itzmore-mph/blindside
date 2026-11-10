# PLAN.md

Phase plan with gates and ready-to-paste Claude Code prompts.
Budget assumption: 8 to 10 hours per week, about 45 to 50 hours in total.

| Phase | Dates | Hours | Gate |
|---|---|---|---|
| 0 Feasibility | 10 to 15 Nov | 3 | Passed on 10 Nov: 1,879 of 1,961 engagements have pose for both players (docs/FEASIBILITY.md section 7). Manual tasks still open |
| 1 Orientation and uncertainty | 16 to 22 Nov | 10 | Unit tests green, velocity consistency check plotted |
| 2 Reliability map | 23 to 29 Nov | 8 | One composite reliability figure |
| 3 Engagements and baseline | 30 Nov to 6 Dec | 10 | Pre-registration committed, XY baseline scored |
| 4 Incremental value | 7 to 11 Dec | 6 | Result with confidence intervals, whatever it shows |
| 5 Packaging | 12 to 16 Dec | 8 | Clean-clone run passes, README and video done |
| Buffer | 17 to 18 Dec | 0 | Submit on 17 Dec |

Cut rule: if phase 0 or phase 3 shows too few usable engagements, drop phase 4 and submit deliverables 1 and 2
with the blind-side metric shown descriptively on case examples.

---

## Phase 0: Feasibility (gate: go / no-go)

Manual tasks for Moritz:
- Join the Discord. Ask: (1) Do the 10 new tracking matches also have dynamic events, and where are they hosted?
  (2) Does "work out of the box" tolerate a multi-GB download from Hugging Face in the full path if a fast path on the sample exists?
  (3) Do figures inside the one-minute video count toward the two-figure limit? (Presumably not, confirm.)
- Check the Publicis employment contract for side-activity and IP clauses before publishing the repo.
- Confirm in-person availability for Paris on 17 February 2027. Remote talks are not accepted.
- Decide solo or pair before phase 1.

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

## Phase 4: Incremental value of pose

```
Read CLAUDE.md and docs/PREREGISTRATION.md. Implement the pose features:
- p_blind_side: probability that the presser lies outside the carrier's visual field at engagement start,
  computed from the Monte Carlo head and torso orientation samples (field of view half-angle as a parameter, default 100 degrees).
- presser body alignment: angle between presser hip orientation and the direction to the carrier, with uncertainty.
- carrier openness: carrier torso orientation relative to the attacking direction.

Run exactly the pre-registered comparison: baseline versus baseline plus pose features. Report the performance difference
with a paired cluster bootstrap confidence interval, and the descriptive effect of p_blind_side on the primary outcome in bins.
Repeat using only engagements that pass the usability threshold. Report all results including a null result.
Anything beyond the pre-registration goes into a clearly labelled exploratory section.
```

Gate: one honest headline sentence, supported by the interval. A null result is acceptable and is reported as such.

## Phase 5: Packaging

```
Read CLAUDE.md. Implement the CLI: `blindside reproduce --fast` (sample phase, under five minutes) and `blindside reproduce`
(full download with cache and progress, checksum verification, clear error messages on network failure). Both regenerate the two README figures.

Draft README.md: at most 1000 words, exactly two figures. Structure mapped to the judging criteria:
the defensive question and who it helps, method in plain language, the headline result with its confidence interval,
limitations, how to reproduce, how to reuse the package (10-line example), third-party libraries and licences.
Add a word-count check to CI that fails above 1000 words or more than two figures or tables.
Add LICENSE.md (MIT). Add a GitHub Actions workflow that runs tests and the fast path on a clean runner.

Then run a clean-clone test in a fresh directory on a machine state without caches and fix everything that breaks.
```

Manual tasks for Moritz: record the one-minute video (script: problem 10 s, idea 15 s, result 20 s, reuse 15 s), upload to YouTube,
make the repo public, submit on Pretalx on 17 Dec.
