"""Use the package on your own pose file: orientation with both bounds, then ``p_blind_side``.

Runs on the 12.3 s sample phase of SkillCorner's open data (match 1925299), which is
downloaded on first use together with the events and metadata of that match (about 6 MB,
verified by sha256). No tracking and no full pose archive are needed, and no outcome column
is read. To use another pose file, replace ``sample_pose_file()`` and ``MATCH_ID``.

Run from the repository root: ``uv run python examples/quickstart.py``.

The sample holds two on-ball engagements. It shows how the package is called. It does not
reproduce a result: for those, run ``blindside reproduce``.
"""

from __future__ import annotations

import pandas as pd

from blindside import (
    build_orientation_table,
    classify_xy_reliability,
    pose_feature_table,
    sample_pose_file,
)
from blindside.features import read_engagement_poses, select_pose_frames
from blindside.features.engagements import engagement_table

MATCH_ID = 1925299
SEGMENTS = ("torso", "hips", "head")
BOUNDS = ("lower", "upper")


def orientation_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Median angular uncertainty per body segment under the lower and the upper bound."""
    return pd.DataFrame(
        {
            "poses": [int(table[f"{segment}_deg"].notna().sum()) for segment in SEGMENTS],
            "sd_lower_deg": [table[f"{segment}_sd_low_deg"].median() for segment in SEGMENTS],
            "sd_upper_deg": [table[f"{segment}_sd_deg"].median() for segment in SEGMENTS],
        },
        index=list(SEGMENTS),
    ).round(1)


def blind_side_table(pose_file: str, table: pd.DataFrame) -> pd.DataFrame:
    """``p_blind_side`` under both bounds for the on-ball engagements inside the pose file."""
    engagements = engagement_table(MATCH_ID, with_outcomes=False)
    engagements["pose_frame"] = select_pose_frames(engagements, table[["frame", "player_id"]])
    engagements = engagements[engagements["pose_frame"] >= 0].reset_index(drop=True)
    frames = engagements["pose_frame"].tolist()
    keys = set(zip(frames, engagements["presser_id"].tolist(), strict=True)) | set(
        zip(frames, engagements["carrier_id"].tolist(), strict=True)
    )
    poses = read_engagement_poses(pose_file, keys, MATCH_ID)

    # The carrier's speed at the pose frame, from the positions in the pose file.
    speed = table.set_index(["frame", "player_id"])["speed"].reindex(
        pd.MultiIndex.from_arrays([engagements["pose_frame"], engagements["carrier_id"]])
    )
    out = engagements[["pose_frame", "presser_id", "carrier_id"]].copy()
    out["carrier_speed_ms"] = speed.to_numpy(dtype=float).round(1)
    for bound in BOUNDS:  # the true angular error lies between the two bounds
        features = pose_feature_table(engagements, poses, bound=bound)
        out[f"p_head_{bound}"] = features["p_blind_side_head"].round(3)
        out[f"p_torso_{bound}"] = features["p_blind_side_torso"].round(3)
    # On a match without body pose only an XY proxy exists. Can it be trusted at this speed?
    out["xy_proxy_would_be"] = classify_xy_reliability(out["carrier_speed_ms"])
    return out


def main() -> int:
    """Print both tables for the sample phase."""
    pose_file = sample_pose_file()
    table = build_orientation_table(pose_file, MATCH_ID)
    print(f"{len(table):,} poses in {table['frame'].nunique()} frames of {pose_file.name}\n")
    print("Median angular uncertainty in degrees, lower (jitter) and upper (Monte Carlo) bound:")
    print(orientation_summary(table).to_string(), "\n")
    print("p_blind_side of the on-ball engagements in the clip, under both bounds:")
    print(blind_side_table(str(pose_file), table).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
