# CLAUDE.md

Project context for Claude Code. Read this file fully before any task.

## Project

Working title: **blindside: reliability-aware body orientation for pressing analysis**

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

- Tracking, dynamic events, phases of play, match metadata: `github.com/SkillCorner/opendata`, folder `data/matches/{match_id}/`
- Body pose (two matches, 1925299 and 1996435, about 3.3 GB JSON each): `huggingface.co/datasets/SkillCorner/opendata-bodypose`
- One committed sample phase of play: `data/bodypose/` in the opendata repo
- Schema of the body pose files: `data/bodypose/README.md` in the opendata repo. **Read it before writing any loader. Do not assume joint names, units or axis conventions.**
- Reference loaders: `src/` in the opendata repo. Read them for format details, then write our own code.

### Known data facts (from the opendata README)

- Tracking is 10 fps, body pose is 25 fps with XY interpolated to 25 fps.
- Coordinates in meters, origin at pitch centre, x is the long side.
- Body pose exists only for detected players (`is_detected`).
- The z-axis is accurate relative to the player's centroid, not in the pitch coordinate system.
  Consequence: use joint positions relative to the player, avoid absolute heights.
- XY tracking and body pose are generated separately. Minor misalignment is expected. Quantify it, do not ignore it.
- Each joint has a predicted error in centimeters, defined so that 90% of estimates lie within that
  radius of the true joint position, relative to the player's pose.
- About 97% of player identities are correct. Raw speeds need smoothing.
- Dynamic event x/y are not scaled to a standard pitch size.
- EPV and pressure columns are populated on `player_possession` rows only.

## Method conventions

- **Error model.** Treat the per-joint error as an isotropic 3D Gaussian. A 90% radius r90 corresponds to
  sigma = r90 / 2.50 (square root of the 0.90 quantile of chi-square with 3 degrees of freedom).
  This is an assumption. State it in the notebook and run a sensitivity check with sigma = r90 / 2.0 and r90 / 3.0.
- **Orientation.** Torso facing = horizontal normal of the shoulder line, hips = horizontal normal of the hip line,
  head = derived from head joints as available in the schema. Resolve the front/back ambiguity explicitly and test it.
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

## Repository layout

```
src/blindside/
  data/        # download, caching, loaders for tracking, events, pose
  geometry/    # orientation, circular stats, error propagation
  features/    # engagement extraction, XY baseline features, pose features
  analysis/    # reliability map, incremental value test
  viz/         # plotting helpers
tests/
docs/
  PLAN.md
  PREREGISTRATION.md
  FEASIBILITY.md
  cli.py       # `blindside reproduce [--fast]`
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
