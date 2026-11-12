# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Project context for Claude Code. Read this file fully before any task.

## Project

Name: **blindside**. Blind-side pressing from broadcast body pose, with measurement uncertainty.

Entry for the SkillCorner X PySport Analytics Cup 2.0 (football challenge).
Submission deadline: 18 December 2026. Internal deadline: 16 December 2026.
Finalists announced: 8 January 2027. Finals: 17 February 2027 (Europe, Paris), 24 February 2027 (US, Boston).

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

Framing for the submission (updated 11 November 2026, after Phase 4): the football question leads, the reliability work is
the method that makes the answer trustworthy. The submission
1. leads with **who counts as pressed from the blind side**: the pose label against the XY proxy (presser more than 100 degrees
   off the carrier's direction of travel), by carrier speed. XY alone calls 45.5% of engagements blind, head pose 20.5%,
   and the two agree beyond chance only when the carrier runs (docs/INCREMENTAL.md section 1.10).
2. states the pre-registered null plainly, with its interval and its resolution: no detectable increment for possession loss,
   and an effect up to an odds ratio of about 1.5 per standard deviation could not have been told from zero with 217 events.
3. presents the frozen design (pre-registration, baseline, features, script) as ready to rerun unchanged on more matches.

Headline: "Who is pressed from the blind side? Broadcast body pose against the XY proxy, with honest uncertainty."
Deliverables 1 and 2 are the methodological backbone and the reusable package. Deliverable 3 is what a coach cares about.
Never write that blind-side pressure causes or predicts possession loss. The test found no increment.

If the sample is too small for the incremental value test, keep deliverable 3 as a descriptive metric with uncertainty
and state clearly that it is implemented but not validated.

## Official rules (Analytics Cup 2.0, confirmed)

Football theme: **Defensive Positioning**. "How can tracking and contextual data help us better understand the way players and teams defend?"

Submission consists of exactly three things:
1. A YouTube video of at most three minutes (raised from one minute, update received 11 November 2026), voiced by the participant.
   Covers: what we did, why we chose this problem, how we approached and solved it, implications and open-source potential.
2. A public GitHub repository with all code and resources **except the data**, open-source licence, with a `LICENSE.md` file (not `LICENSE`).
3. A `README.md` of at most 1000 words with at most **two figures or tables in total**, including instructions to reproduce the results in a clean environment.
   Must be written in our own words and aligned with the video's content (same update).

Disqualification triggers that matter for engineering:
- Work that does not run out of the box.
- Any additional data, including video or tracking data from outside the competition.
- Proprietary software in the reproduction path.
- Resubmitted or lightly adapted earlier projects. All analysis code is written new for this entry.
  Third-party libraries are allowed and must be listed in the README with their licences.

Judging, 20% each: Relevance, Methodology, Originality, Communication, Open-source potential.

Consequences for how we build:
- The README is the submission. Every analysis must earn a place in 1000 words and two figures. Everything else goes to `docs/`.
- README figure 1: the reliability composite (`docs/figures/reliability_composite.png`).
  README figure 2 (`docs/figures/proxy_agreement.png`): Cohen's kappa of the XY proxy against the pose label by carrier speed band, plus the share of engagements called blind by each label (pose head, pose torso, XY proxy). No outcome in it.
- One command reproduces both README figures from a clean clone: `uv run blindside reproduce`. It needs the two pose matches only
  (1.4 GB download). `--with-test` also refits the baseline on the other 18 matches and reruns the pre-registered comparison.
- `uv run blindside reproduce --fast` runs on the sample phase and must finish in under five minutes. The sample holds two engagements,
  so it cannot compute figure 2. The fast path writes stamped stand-ins to `outputs/fast/`
  (git ignored), the reliability composite on the sample and a geometry panel of the two engagements, and never touches `docs/figures/`.
- `README.md` limits (1000 words, two figures or tables) are not checked automatically: `scripts/check_readme.py` was removed on
  12 November 2026. Count by hand after every README edit (conservative count: every whitespace-separated token).
  `tests/test_readme.py` does check that the README embeds exactly two images, that its relative links exist, and that its
  one Python block prints `1.00` and `0.73`. Changing the snippet or adding an image fails the suite.
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
  A clean clone has no such directory: events, match metadata and the sample phase are then downloaded from `raw.githubusercontent.com`
  at commit 4340d27 into `data/raw/opendata/` and verified against `src/blindside/data/opendata_manifest.json`
  (`blindside.data.opendata.opendata_file`). An explicit `BLINDSIDE_OPENDATA_DIR` is never filled by a download.
- Tracking is downloaded by `python -m blindside.data.tracking` from the GitHub LFS media endpoint at commit 4340d27 into
  `data/raw/tracking/` (20 files, 1.8 GB, sha256 from `src/blindside/data/tracking_manifest.json`). `git-lfs` is not needed.
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
- XY tracking and body pose are generated separately. Minor misalignment is expected. Measured: the hip midpoint lies 6.8 cm (median)
  and 22 cm (99th percentile) from the tracking position (docs/RELIABILITY.md section 2.6).
- The main camera stands behind the touchline at negative y in both pose matches, at the halfway line.
  Inferred from `image_corners_projection` (docs/RELIABILITY.md section 1.4), not given in the data.
- The published `p90_mae_cm` does not grow with distance from the camera, the frame-to-frame jitter does
  (head: 1.8 degrees next to the camera, 2.8 at the far touchline). Report the distance from the camera touchline next to pose features.
- About 97% of player identities are correct. Raw speeds need smoothing.
- Dynamic event x/y are not scaled to a standard pitch size.
- Dynamic event x/y are direction-normalised: the acting team attacks towards +x. For rows with `attacking_side == right_to_left`,
  pitch coordinates are `(-x, -y)`. Mirror before joining events with pose or tracking. `third_start` follows the same normalisation.
- EPV and pressure columns are populated on `player_possession` rows only.
- On-ball engagements link to the pressed possession via `associated_player_possession_event_id` and `player_in_possession_id`.
  They already carry outcome flags (`force_backward`, `beaten_by_possession`, `stop_possession_danger`, `reduce_possession_danger`)
  and XY context (`interplayer_distance_start`, `angle_of_engagement`, `goal_side_start`, `speed_avg`).
  Several engagements can target the same possession. Always cluster by possession.
- 57% of on-ball engagements start before the pressed possession starts (median 0.2 s): the carrier is then still a receiver.
- Primary outcome (docs/PREREGISTRATION.md): possession loss of the pressed possession. 217 events in 1,875 included test engagements.
- Phase 4 result (docs/INCREMENTAL.md): no detectable increment of the four pose features over the XY baseline, under both
  uncertainty bounds (log loss difference minus 0.0002, 95% intervals within minus 0.0004 to 0.0000). Possession loss is 13.2% in the blind bin
  of `p_blind_side_head` against 10.8% in view (lower bound), plus 2.4 points (minus 1.2 to plus 5.8). Deliverable 3 is implemented, not validated.
- Resolution of the Phase 4 test (exploratory, docs/INCREMENTAL.md section 1.9): the unpenalised coefficient of `p_blind_side_head` is
  plus 0.11 per standard deviation (minus 0.18 to plus 0.39, odds ratio 0.84 to 1.47, lower bound). Without a penalty the pose features
  make the out-of-sample log loss worse by 0.002, so the null is not produced by the shrinkage.
- XY proxy against the pose label (exploratory, no outcome used, docs/INCREMENTAL.md section 1.10). Proxy: `approach_angle_heading_deg`
  above 100 degrees. Pose label: `p_blind_side` above 0.5. Head: agreement 65.8% (63.5 to 68.0), Cohen's kappa 0.28 (0.24 to 0.31).
  Torso: kappa 0.35 (0.31 to 0.39). The proxy calls 45.5% of engagements blind, head pose 20.5%, torso pose 23.2%.
  Recall 77%, precision 35% (head).
- Kappa by carrier speed, head label: 0.09 below 2 m/s (665 engagements), 0.31 at 2 to 4 (727), 0.54 at 4 to 6 (370),
  0.58 above 6 m/s (113). Torso label: 0.10, 0.41, 0.66, 0.77. Below 2 m/s the proxy is close to chance, and 35% of engagements are there.
- A label cut at `p_blind_side` 0.5 is almost the point estimate: both bounds give the same agreement table to within 0.2 points.
  The bounds matter for who lands in the uncertain bin (56 engagements under the lower bound, 342 under the upper, head), not for this cut.
- Low agreement cannot say which label is right. Without ground truth, "pose carries orientation that XY lacks" and "one label is noisy"
  are not separable. Say so wherever kappa is shown.

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
- **Uncertainty is a bracket.** The Monte Carlo SD under independence is the upper bound, the frame-to-frame jitter of the angle
  is the lower bound (docs/RELIABILITY.md section 1). The true angular error lies between them and cannot be identified without
  ground truth. Every downstream quantity that uses angular uncertainty, in particular `p_blind_side`, is computed under both bounds
  (`bound_correlation(segment, "lower" | "upper")`, table columns `{segment}_sd_low_deg` and `{segment}_sd_deg`).
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
- Package code in `src/blindside/`, tests in `tests/`.
- Use polars or pandas consistently (default: pandas, pyarrow backend). Cache intermediate tables as parquet in `data/cache/` (git ignored).
- Stream the large pose JSON files. Never load a full 3.3 GB file into memory.
- Type hints, docstrings, `ruff` for linting, `pytest` for tests. Every geometric function gets a unit test with a synthetic skeleton.
- All code, comments and docs in English. No em dashes in any text.
- Plots with matplotlib and mplsoccer. One consistent style module in `src/blindside/viz/style.py`.
- Small, reviewable commits. Never commit data files or credentials.
- Every commit message ends with the trailer `Co-Authored-By: Claude <noreply@anthropic.com>` to disclose AI assistance.
## Repository layout

```
src/blindside/
  data/        # download, caching, loaders for tracking, events, pose
  geometry/    # orientation, circular stats, error propagation
  features/    # engagement extraction, XY baseline features, pose features
  analysis/    # reliability map, incremental value test, pipeline.py (the steps shared by scripts and CLI)
  viz/         # plotting helpers
  cli.py       # `blindside reproduce [--fast | --with-test] [--with-scene]`
scripts/       # write the generated blocks of docs/*.md and the pitch figures outside the README
examples/      # quickstart.py: the package on the sample phase, run in CI
               # walkthrough.ipynb: the same on the sample phase, committed with outputs (jupyter is not a dependency)
.github/workflows/ci.yml  # lint, tests, fast path on a clean runner
tests/
docs/
  PLAN.md
  PREREGISTRATION.md
  FEASIBILITY.md
  RELIABILITY.md
  BASELINE.md
  INCREMENTAL.md
  EXPLORATORY_DYNAMIC_EVENTS.md   # not pre-registered: possession labels, defending phases, pitch case study, XY helper
  TACTICAL_INSIGHTS.md            # not pre-registered, descriptive: pressers, carriers, zones, sample size for a rerun
README.md      # the judged document: max 1000 words, max two figures or tables
LICENSE.md
```

## Commands

```
uv sync                             # install, including the dev group (pytest, ruff, scipy)
uv run pytest                       # full test suite
uv run pytest tests/test_orientation.py          # one file
uv run pytest tests/test_orientation.py::test_x  # one test
uv run ruff check .                 # lint
uv run ruff format --check .        # format check (CI fails on unformatted code)
uv run blindside reproduce --fast   # sample phase only, under 5 min, writes outputs/fast/ (git ignored)
uv run blindside reproduce          # full run, both README figures, ~13 min, downloads 1.4 GB
uv run blindside reproduce --with-test  # also refits baseline on 18 matches, reruns the pre-registered test, ~3 GB
uv run blindside reproduce --with-scene # also docs/figures/match_scene_cone.png (linked from the README, not embedded)
uv run python scripts/tactical_case_study.py                    # docs/figures/tactical_case_study.png (video, not README)
uv run python scripts/plot_scene_example.py                     # docs/figures/match_scene_cone.png on its own
uv run python scripts/tactical_insights.py --write-docs         # generated block of docs/TACTICAL_INSIGHTS.md
uv run --with nbconvert --with ipykernel jupyter nbconvert --to notebook --execute --inplace examples/walkthrough.ipynb
uv run python scripts/exploratory_dynamic_events.py --write-docs  # generated block of docs/EXPLORATORY_DYNAMIC_EVENTS.md
```

CI (`.github/workflows/ci.yml`) runs lint, `pytest`, `blindside reproduce --fast` and `examples/quickstart.py` on a clean
runner (Python 3.11 and 3.13), plus a separate `pip install .` job that checks the package and CLI work outside uv
and outside the repository. There is no local data directory in CI: the fast path downloads everything it needs.

## Architecture: how a run flows

`src/blindside/analysis/pipeline.py` is the shared spine between `scripts/*.py` (which render `docs/*.md`) and
`cli.py` (which renders the two README figures). Both call into the same functions, so the reported numbers and the
regenerated figures never diverge. Rough data flow for `blindside reproduce`:

1. **`data/`** downloads and verifies pose archives (Hugging Face) and tracking (`opendata` LFS media), or reads
   them from `BLINDSIDE_OPENDATA_DIR` / `../opendata` if present. `pose.py` streams the pose JSONL; nothing loads a
   full match into memory at once.
2. **`features/orientation_table.py`** turns a streamed pose file into a cached per-frame, per-player parquet table
   (`data/cache/`) via `geometry/orientation.py` (facing angles) and `geometry/uncertainty.py` (Monte Carlo angular
   SD under both the lower/jitter and upper/independent-joint bounds).
3. **`analysis/reliability.py`** and **`analysis/camera.py`** turn that table into the reliability-map inputs
   (pitch zone, speed band, camera-side distance) consumed by `viz/reliability.py` for figure 1.
4. **`features/engagements.py`** and **`features/xy_baseline.py`** build the on-ball-engagement table with the
   pre-registered inclusion criteria and the XY-only baseline features; **`features/pose_features.py`** attaches
   `p_blind_side` and friends for engagements with pose at both ends.
5. **`analysis/baseline.py`** (fit, cluster bootstrap) and **`analysis/incremental.py`** (out-of-fold log loss,
   calibration, robustness checks) implement the frozen pre-registered comparison; **`analysis/agreement.py`**
   implements the exploratory XY-proxy-versus-pose-label kappa behind figure 2 (`viz/agreement.py`).
6. `cli.py` orchestrates steps 1 to 5 end to end for `reproduce`/`reproduce --fast`/`reproduce --with-test`
   (`--with-scene` adds the pitch scene to a full run and is rejected with `--fast`); the
   `scripts/` files call the same `pipeline.py` functions to regenerate the numbers embedded in `docs/*.md`.

Outside the judged path (added 12 November 2026, all exploratory): `features/xy_proxy.py` (`xy_blind_side_proxy`,
`classify_xy_reliability`: unreliable below 2 m/s, reliable from 4 m/s, for matches without pose),
`analysis/dynamic_events.py` and `analysis/case_study.py` with `viz/case_study.py`. The README is organised by the four
video pillars: what we did, why, how, implications. The case study figure says "the labels disagree", never "XY is wrong".

Also outside the judged path (added 12 November 2026, descriptive, no outcome read): `analysis/insights.py`
(`confident_blind`: `p_blind_side` above 0.8 under both bounds, `group_profile`, `required_events`) and `analysis/scene.py`
with `viz/case_study.plot_scene_cone`. `match_scene_cone.png` is a third figure and must stay a link in the README:
embedding it would break the two-figure limit. Re-execute `examples/walkthrough.ipynb` after changing a function it calls
(`tests/test_notebook.py` only checks that outputs are stored). The sample size numbers in the README (about 700 possession
losses for an odds ratio of 1.25 at 80% power) are a planning approximation from the standard error in docs/INCREMENTAL.md 1.9.

When changing a geometry or feature function, check both call sites (a `scripts/` file and `cli.py`/`pipeline.py`)
so the README figures and the `docs/` write-ups stay consistent.

## How to work

- `docs/PLAN.md` holds the phases and their gates. All of them are done (Phase 5 closed on 12 November 2026).
  Open: the video, its link in the README, the submission. New work is exploratory and stays outside the judged path.
- Run commands from the repository root. Downloads and caches resolve against the working directory (`./data`, `../opendata`,
  see `data/paths.py`), so a run from another folder downloads everything again into that folder. Override with `BLINDSIDE_DATA_DIR`.
- When a data fact is unclear, inspect the data and write the finding to `docs/FEASIBILITY.md`. Do not guess.
- When an assumption is needed, state it, mark it as an assumption, and add a sensitivity check.
- Distinguish in all write-ups between implemented and validated, implemented but not validated, and proposed.
- Ask before adding a dependency, changing the pre-registered design, or cutting scope.
