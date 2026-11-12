"""Two engagements on the pitch: where the XY proxy and the pose label part, and where they meet.

Draws ``docs/figures/tactical_case_study.png`` for the video walkthrough and
docs/EXPLORATORY_DYNAMIC_EVENTS.md. It is not a README figure. The engagements are picked by
the rule of :func:`blindside.analysis.case_study.case_study_candidates`: the first candidate
of each kind. No outcome column enters the choice or the drawing.

Run from the repository root: ``uv run python scripts/tactical_case_study.py``.
Needs the body pose and the tracking of the two pose matches, as ``blindside reproduce`` does.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from blindside.analysis.case_study import AGREE_BLIND, DISAGREE, case_study_candidates
from blindside.analysis.pipeline import (
    attach_pose,
    engagement_geometry,
    inclusion_flow,
    load_test_table,
)
from blindside.data.events import load_dynamic_events, load_match_metadata
from blindside.data.tracking import cached_positions, position_grid
from blindside.features.pose_features import BOUNDS, FOV_HALF_ANGLE_DEG, pose_feature_table
from blindside.viz.case_study import plot_tactical_case_study

FIGURE = Path("docs/figures/tactical_case_study.png")
DPI = 150
PANELS = {
    DISAGREE: "A  Slow carrier, {speed:.1f} m/s: the labels disagree",
    AGREE_BLIND: "B  Running carrier, {speed:.1f} m/s: the labels agree",
}


def other_players(row: pd.Series) -> pd.DataFrame:
    """Everyone but presser and carrier at the engagement start, from the 10 fps tracking."""
    match_id = int(row["match_id"])
    grid = position_grid(cached_positions(match_id))
    xy = grid.xy[int(grid.row(int(row["frame_start"])))].astype(float)
    team_of = {int(p["id"]): int(p["team_id"]) for p in load_match_metadata(match_id)["players"]}
    table = pd.DataFrame({"player_id": grid.players, "x": xy[:, 0], "y": xy[:, 1]})
    table["pressing_team"] = table["player_id"].map(team_of) == int(row["presser_team_id"])
    involved = table["player_id"].isin([int(row["presser_id"]), int(row["carrier_id"])])
    return table[~involved & table["x"].notna()].reset_index(drop=True)


def source_line(row: dict[str, object]) -> str:
    """Teams, clock time and match id of one engagement."""
    match_id = int(row["match_id"])
    metadata = load_match_metadata(match_id)
    events = load_dynamic_events(match_id, columns=["event_id", "time_start", "period"])
    event = events.loc[events["event_id"] == row["event_id"]].iloc[0]
    teams = f"{metadata['home_team']['short_name']} against {metadata['away_team']['short_name']}"
    return f"{teams}, period {int(event['period'])} at {event['time_start']} (match {match_id})"


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=FIGURE)
    args = parser.parse_args(argv)

    test, _ = inclusion_flow(load_test_table(), with_pose=True)
    test, poses = attach_pose(test)
    features = {bound: pose_feature_table(test, poses, bound=bound) for bound in BOUNDS}
    candidates = case_study_candidates(test, features)

    rows, others, pitch_dims = [], [], []
    for kind, title in PANELS.items():
        print(f"{kind}: {len(candidates[kind]):,} candidates of {len(test):,} engagements")
        chosen = candidates[kind].iloc[[0]]
        match_id = int(chosen["match_id"].iloc[0])
        geometry = engagement_geometry(
            chosen, poses[poses["match_id"] == match_id], features,
            position_grid(cached_positions(match_id)),
        ).iloc[0].to_dict()  # fmt: skip
        engagement = chosen.iloc[0]
        geometry["match_id"], geometry["event_id"] = match_id, engagement["event_id"]
        geometry["title"] = title.format(speed=float(engagement["carrier_speed"]))
        geometry["source"] = source_line(geometry)
        print(
            f"  {geometry['source']}, event {engagement['event_id']}: proxy angle "
            f"{geometry['approach_angle_heading_deg']:.0f} deg, p_blind_side_head "
            f"{geometry['p_head_lower']:.2f} / {geometry['p_head_upper']:.2f}, head SD "
            f"{geometry['head_sd_lower_deg']:.1f} / {geometry['head_sd_upper_deg']:.1f} deg"
        )
        rows.append(geometry)
        others.append(other_players(engagement))
        metadata = load_match_metadata(match_id)
        pitch_dims.append((float(metadata["pitch_length"]), float(metadata["pitch_width"])))

    caption = (
        "Each panel is the clearest of its kind by a fixed rule that reads no outcome, not a "
        "typical engagement.\nThe labels can disagree. Without ground truth for orientation, "
        "a panel cannot show which label is right."
    )
    figure = plot_tactical_case_study(
        pd.DataFrame(rows), others, pitch_dims, half_angle_deg=FOV_HALF_ANGLE_DEG, caption=caption
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=DPI, bbox_inches=None)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
