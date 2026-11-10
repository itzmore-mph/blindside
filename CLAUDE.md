# CLAUDE.md

Project context for Claude Code. Read this file fully before any task.

## Project

Name: **blindside**. Blind-side pressing from broadcast body pose, with measurement uncertainty.

Entry for the SkillCorner X PySport Analytics Cup 2.0 (football challenge).
Submission deadline: 18 December 2026. Internal deadline: 16 December 2026.

### Research question

How much pose-derived defensive signal survives the measurement error of broadcast body pose,
and does it add explanatory value beyond XY tracking?

### Three deliverables, in priority order

1. **Orientation with uncertainty.** Torso, hip and head orientation per player and frame from the
   29-joint body pose, with the per-joint predicted error propagated into an angular uncertainty
   (Monte Carlo). Output: angle plus circular standard deviation, not a point estimate.
2. **Reliability map.** Where and when is pose-derived orientation usable? Uncertainty broken down by
   pitch zone, player speed, distance to the broadcast camera side, and joint group. Includes a
   no-ground-truth consistency check: at high running speed, hip orientation should align with the
   XY velocity heading.
3. **Applied metric: blind-side pressure.** For each on-ball engagement, the probability that the
   presser approaches from outside the ball carrier's visual field. Test whether this adds explanatory
   value over an XY-only baseline for a pre-registered outcome.

Framing for the submission: the football question leads, the reliability work is the method that makes the answer trustworthy.
Headline: "Pressure the ball carrier cannot see: measuring blind-side pressing from broadcast body pose, with honest uncertainty."
Deliverables 1 and 2 are the methodological backbone and the reusable package. Deliverable 3 is what a coach cares about.

If the sample is too small for the incremental value test, keep deliverable 3 as a descriptive metric with uncertainty
and state clearly that it is implemented but not validated.

## Official rules (Analytics Cup 2.0, confirmed)

Football theme: **Defensive Positioning**. "How can tracking and contextual data help us better understand the way players and teams defend?"

Submission consists of exactly three things:
1. A YouTube video of at most one minute, voiced by the participant.
2. A public GitHub repository with all code and resources **except the data**, open-source licence, with a `LICENSE.md` file (not `LICENSE`).
3. A `README.md` of at most 1000 words with at most **two figures or tables in total**, including instructions to reproduce the results in a clean environment.

Disqualification triggers that matter for engineering:
- Work that does not run out of the box.
- Any additional data, including video or tracking data from outside the competition.
- Proprietary software in the reproduction path.
- Resubmitted or lightly adapted earlier projects. All analysis code is written new for this entry.
  Third-party libraries are allowed and must be listed in the README with their licences.

Judging, 20% each: Relevance, Methodology, Originality, Communication, Open-source potential.

Consequences for how we build:
- The README is the submission. Every analysis must earn a place in 1000 words and two figures. Everything else goes to `docs/`.
- One command reproduces both README figures from a clean clone: `uv run blindside reproduce`.
  A fast path on the committed sample phase must finish in under five minutes. The full path downloads and caches the pose files.
- The package must be installable and reusable on its own (`pip install .`), with a documented public API,
  because the winner's grant funds conversion into a package or a pull request to an existing library.
- Do not commit data or derived data tables.

## Data sources

- Tracking, dynamic events, phases of play, match metadata: `github.com/SkillCorner/opendata`, folder `data/matches/{match_id}/`.
  20 matches, **all 20 have dynamic events** (322 columns) and phases of play. Tracking `.jsonl` files are Git LFS.
- Body pose for matches 1925299 and 1996435: `huggingface.co/datasets/SkillCorner/opendata-bodypose`, files `raw/{match_id}.jsonl.zip`.
  Download size about 620 MB each (3.3 GB uncompressed JSON). Sizes and sha256 are in `data/bodypose/MANIFEST.json` **of the opendata repo**.
  Our copy with a pinned Hugging Face revision is `src/blindside/data/pose_manifest.json`. Downloads go to `data/raw/bodypose/` (git ignored).
- One sample phase committed **in the opendata repo, not in ours**: `data/bodypose/sample_1925299_phase406.jsonl.gz` (309 pose frames, 12.3 s, 1.8 MB).
- Local reference clone of opendata: `../opendata` (commit 4340d27, LFS tracking files not pulled). Override with `BLINDSIDE_OPENDATA_DIR`.
- Verified findings from the data: `docs/FEASIBILITY.md`. Read it before writing loaders.
- SkillCorner ships reference code in `src/data/pose_loading.py` and `src/features/pose_orientation.py` (MIT).
  Read it for format details, then write our own implementation. Disclose it in the README as prior art:
  their helper gates shoulder orientation with a hard 15 cm threshold, ours propagates the error into an angular uncertainty.

### Known data facts (verified, see docs/FEASIBILITY.md)

- Tracking and events are 10 fps, body pose is 25 fps. `pose_frame = 2.5 * tracking_frame`. Every 5th pose frame aligns exactly.
- Coordinates in meters, origin at pitch centre, x is the long side.
- Pose record: `player_data[i].joints[name] = {"xyz": [x, y, z], "p90_mae_cm": float}`. `joints` is null when pose was not resolved.
- 29 joints: nose, neck, lEye, rEye, lEar, rEar, lShoulder, rShoulder, lElbow, rElbow, lWrist, rWrist, lThumb, rThumb, lPinky, rPinky,
  midHip, lHip, rHip, lKnee, rKnee, lAnkle, rAnkle, lHeel, rHeel, lBigToe, rBigToe, lSmallToe, rSmallToe.
- Facing direction convention that passed the velocity check: left-to-right axis rotated by +90 degrees.
- Across a full match 31.4% (1925299) and 36.4% (1996435) of player-frames carry joints. Among `is_detected` players it is 99.3%:
  pose is missing almost only when the player is off camera. 95.8% of on-ball engagements (1,879 of 1,961) have pose for presser
  and carrier at `frame_start`, 89.1% (1,748) also at minus 0.5 s and minus 1.0 s. Source: docs/FEASIBILITY.md section 7.
- The pose zip archives contain a macOS metadata member (`__MACOSX/._*.jsonl`). 47 frames per match lie outside the periods with empty `player_data`.
- The z-axis is accurate relative to the player's centroid, not in the pitch coordinate system. Use x and y only for ground-plane headings.
- XY tracking and body pose are generated separately. Minor misalignment is expected.
- About 97% of player identities are correct. Raw speeds need smoothing.
- Dynamic event x/y are not scaled to a standard pitch size.
- Dynamic event x/y are direction-normalised: the acting team attacks towards +x. For rows with `attacking_side == right_to_left`,
  pitch coordinates are `(-x, -y)`. Mirror before joining events with pose or tracking. `third_start` follows the same normalisation.
- EPV and pressure columns are populated on `player_possession` rows only.
- On-ball engagements link to the pressed possession via `associated_player_possession_event_id` and `player_in_possession_id`.
  They already carry outcome flags (`force_backward`, `beaten_by_possession`, `stop_possession_danger`, `reduce_possession_danger`)
  and XY context (`interplayer_distance_start`, `angle_of_engagement`, `goal_side_start`, `speed_avg`).
  Several engagements can target the same possession. Always cluster by possession.

## Method conventions

- **Error model.** Treat the per-joint error as an isotropic 3D Gaussian. A 90% radius r90 corresponds to
  sigma = r90 / 2.50 (square root of the 0.90 quantile of chi-square with 3 degrees of freedom).
  This is an assumption. State it in the README and run a sensitivity check with sigma = r90 / 2.0 and r90 / 3.0.
- **Orientation.** Torso facing = ground-plane normal of lShoulder to rShoulder, hips = normal of lHip to rHip,
  head = normal of lEar to rEar (compare against the ear-midpoint to nose vector and keep the more stable one).
  The errors of two joints of one player are probably correlated. Independence is our assumption and it overstates angular noise if they are. Say so.
- **Baseline widths matter.** Shoulder width is about 0.37 m, hip width 0.22 m, ear width 0.18 m. The same joint error therefore gives
  roughly 5 degrees of angular noise for the torso and 12 degrees for hips and head. Head-versus-torso differences have a median of 15 degrees,
  which is close to the head noise floor. Never interpret a head turn without its uncertainty.
- **Monte Carlo.** Sample joints N times (default 200), recompute the angle, report circular mean and circular standard deviation.
  Vectorise with NumPy. No per-frame Python loops over samples.
- **Circular statistics everywhere.** Never average angles arithmetically.
- **Validation without ground truth.** Hip orientation versus velocity heading at speeds above a threshold,
  temporal smoothness, left/right symmetry checks. Report results as distributions, not single numbers.
- **Statistics.** Only two pose matches exist. Report effect sizes with bootstrap confidence intervals,
  clustered by possession. Use leave-one-match-out as a robustness check. No significance claims that the sample cannot support.
- **Pre-registration.** The primary outcome and the baseline feature set for deliverable 3 are fixed in
  `docs/PREREGISTRATION.md` before any outcome analysis is run. Do not change them afterwards. Exploratory analyses are labelled as such.
- **No leakage.** Features use information up to the engagement start only. Outcomes come from after it.

## Engineering conventions

- Python 3.11+, managed with `uv`. All dependencies in `pyproject.toml`.
- Package code in `src/blindside/`, tests in `tests/`, notebooks only for presentation.
- Use polars or pandas consistently (default: pandas, pyarrow backend). Cache intermediate tables as parquet in `data/cache/` (git ignored).
- Stream the large pose JSON files. Never load a full 3.3 GB file into memory.
- Type hints, docstrings, `ruff` for linting, `pytest` for tests. Every geometric function gets a unit test with a synthetic skeleton.
- All code, comments and docs in English. No em dashes in any text.
- Plots with matplotlib and mplsoccer. One consistent style module in `src/blindside/viz/style.py`.
- Small, reviewable commits. Never commit data files or credentials.
- Every commit message ends with the trailer `Co-Authored-By: Claude <noreply@anthropic.com>`.
  Project decision by Moritz (10 November 2026) to disclose AI assistance. The published rules neither require nor forbid it.
  This overrides the global no-trailer preference for this repository only.

## Repository layout

```
src/blindside/
  data/        # download, caching, loaders for tracking, events, pose
  geometry/    # orientation, circular stats, error propagation
  features/    # engagement extraction, XY baseline features, pose features
  analysis/    # reliability map, incremental value test
  viz/         # plotting helpers
  cli.py       # `blindside reproduce [--fast]`
tests/
docs/
  PLAN.md
  PREREGISTRATION.md
  FEASIBILITY.md
  RELIABILITY.md
notebooks/     # optional walkthrough, not part of the judged submission
README.md      # the judged document: max 1000 words, max two figures or tables
LICENSE.md
```

## How to work

- Follow `docs/PLAN.md` phase by phase. Do not start a phase before the previous gate is passed.
- When a data fact is unclear, inspect the data and write the finding to `docs/FEASIBILITY.md`. Do not guess.
- When an assumption is needed, state it, mark it as an assumption, and add a sensitivity check.
- Distinguish in all write-ups between implemented and validated, implemented but not validated, and proposed.
- Ask before adding a dependency, changing the pre-registered design, or cutting scope.
