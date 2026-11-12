"""One slow-carrier press on the pitch where the XY proxy and the head pose label disagree.

Draws ``docs/figures/match_scene_cone.png``: the carrier's direction of travel with the
field the XY proxy assumes, the head facing with the visual field and both bounds of its
angular error, the runs of presser and carrier before the engagement start, and a box with
the exact values. It is not one of the two README figures.

The engagement is the clearest candidate of the fixed rule in
:func:`blindside.analysis.case_study.case_study_candidates` (carrier at 1 to 2 m/s, XY proxy
blind, head and torso ``p_blind_side`` below 0.1 under both bounds). No outcome column
enters the choice or the drawing. The figure shows that the labels disagree. It cannot show
which label is right: there is no ground truth for orientation.

Run from the repository root: ``uv run python scripts/plot_scene_example.py``, or
``uv run blindside reproduce --with-scene``. Needs the body pose and the tracking of the two
pose matches, as ``blindside reproduce`` does.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from blindside.analysis.pipeline import attach_pose, inclusion_flow, load_test_table
from blindside.analysis.scene import scene_example
from blindside.cli import DPI, SCENE_CAPTION, SCENE_FIGURE
from blindside.features.pose_features import BOUNDS, FOV_HALF_ANGLE_DEG, pose_feature_table
from blindside.viz.case_study import plot_scene_cone

FIGURE = Path("docs/figures") / SCENE_FIGURE


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=FIGURE)
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="candidate of the selection rule to draw, 0 is the clearest (the default)",
    )
    args = parser.parse_args(argv)

    test, _ = inclusion_flow(load_test_table(), with_pose=True)
    test, poses = attach_pose(test)
    features = {bound: pose_feature_table(test, poses, bound=bound) for bound in BOUNDS}
    scene = scene_example(test, poses, features, rank=args.rank)
    row = scene.geometry
    print(f"{scene.n_candidates:,} candidates of {len(test):,} engagements, rank {args.rank}")
    print(
        f"{row['source']}, event {row['event_id']}: carrier at {row['carrier_speed']:.1f} m/s, "
        f"proxy angle {row['approach_angle_heading_deg']:.0f} deg, p_blind_side_head "
        f"{row['p_head_lower']:.2f} / {row['p_head_upper']:.2f}, head SD "
        f"{row['head_sd_lower_deg']:.1f} / {row['head_sd_upper_deg']:.1f} deg"
    )
    figure = plot_scene_cone(
        row,
        scene.others,
        scene.pitch_dims,
        scene.paths,
        half_angle_deg=FOV_HALF_ANGLE_DEG,
        caption=SCENE_CAPTION,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=DPI, bbox_inches=None)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
