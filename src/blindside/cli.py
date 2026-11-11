"""Command line interface: ``blindside reproduce``.

``blindside reproduce``
    Downloads the body pose and the tracking of the two pose matches (about 1.4 GB, cached
    and verified), builds the orientation tables and regenerates both README figures.
``blindside reproduce --with-test``
    Also downloads the tracking of the other 18 matches (about 1.6 GB), refits the XY
    baseline and reruns the pre-registered comparison of docs/PREREGISTRATION.md.
``blindside reproduce --fast``
    Runs the same code on the committed 12.3 s sample phase in under five minutes. The sample
    holds two engagements, so the figures it writes are stand-ins, stamped as such, and
    never overwrite the README figures.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from blindside import __version__
from blindside.data.download import DownloadError

FIGURE_1 = "reliability_composite.png"
FIGURE_2 = "proxy_agreement.png"
FAST_FIGURE_1 = "sample_reliability_composite.png"
FAST_FIGURE_2 = "sample_engagement_geometry.png"
DEFAULT_OUTPUT = Path("docs/figures")
FAST_OUTPUT = Path("outputs/fast")
#: Pitch zones of the sample phase need far fewer poses than those of a full match.
FAST_MIN_ZONE_POSES = 20
FAST_STAMP = "SAMPLE PHASE (12.3 s of match 1925299), a smoke test, not the README figure."
DPI = 150


def _say(message: str) -> None:
    print(message, flush=True)


def _save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI)
    _say(f"wrote {path}")


def _build_orientation_table(match_id: int) -> float:
    """Build the default orientation table of one pose match. Returns the seconds it took."""
    from blindside.analysis.pipeline import orientation_table_path

    start = time.perf_counter()
    orientation_table_path(match_id)
    return time.perf_counter() - start


def _agreement_lines(agreement: pd.DataFrame, bound: str = "lower") -> list[str]:
    """The numbers of README figure 2 as text."""
    lines = [f"XY proxy against the pose label ({bound} bound), exploratory, no outcome used:"]
    for feature, rows in agreement[agreement["bound"] == bound].groupby("feature", sort=False):
        lines.append(f"  {feature}")
        for row in rows.to_dict("records"):
            lines.append(
                f"    {row['group']:<28} n = {row['n']:>5,}"
                f"   pose blind {100 * row['pose_blind']:4.1f}%"
                f"   proxy blind {100 * row['xy_blind']:4.1f}%   kappa {row['kappa']:+.2f}"
                f" ({row['kappa_low']:+.2f} to {row['kappa_high']:+.2f})"
            )
    return lines


def _test_lines(results: dict[str, object]) -> list[str]:
    """Headline of the pre-registered comparison as text."""
    from blindside.analysis.pipeline import decision

    rows: pd.DataFrame = results["primary_rows"]  # type: ignore[assignment]
    lines = [
        "Pre-registered comparison, Model 0 (recalibrated XY baseline) against Model 1 "
        "(plus four pose features):",
        f"  {results['n']:,} engagements, {results['events']:,} possession losses",
    ]
    for row in rows.to_dict("records"):
        lines.append(
            f"  {row['bound']} bound: log loss {row['loss0']:.4f} against {row['loss1']:.4f}, "
            f"Model 0 minus Model 1 {row['gain']:+.4f} ({row['low']:+.4f} to {row['high']:+.4f})"
        )
    lines.append(f"  {decision(rows)}")
    return lines


def reproduce_full(output_dir: Path, with_test: bool = False) -> None:
    """Regenerate both README figures from the two pose matches."""
    from blindside.analysis.baseline import N_BOOTSTRAP, cluster_bootstrap_weights
    from blindside.analysis.pipeline import (
        attach_pose,
        compute,
        inclusion_flow,
        load_tables,
        load_test_table,
        pose_match_orientation_tables,
        proxy_agreement,
        reliability_inputs,
    )
    from blindside.data.download import download_pose_archive, print_progress
    from blindside.data.manifest import pose_match_ids
    from blindside.data.tracking import download_tracking, tracking_match_ids
    from blindside.features.pose_features import BOUNDS, pose_feature_table
    from blindside.viz.agreement import plot_proxy_agreement
    from blindside.viz.reliability import plot_reliability_composite

    pose_ids = pose_match_ids()
    _say("1/4 downloads (verified files are skipped, interrupted downloads resume)")
    for match_id in pose_ids:
        download_pose_archive(match_id, progress=print_progress(f"  pose {match_id}"))
    for match_id in tracking_match_ids() if with_test else pose_ids:
        download_tracking(match_id, progress=print_progress(f"  tracking {match_id}"))

    _say("2/4 orientation tables (about two minutes per match on first use)")
    with ProcessPoolExecutor(max_workers=len(pose_ids)) as pool:
        for match_id, seconds in zip(
            pose_ids, pool.map(_build_orientation_table, pose_ids), strict=True
        ):
            _say(f"  {match_id}: ready" + (f" after {seconds:.0f} s" if seconds > 1 else ""))

    _say("3/4 figure 1, reliability of pose-derived orientation")
    table, pitch_dims, camera_side = pose_match_orientation_tables()
    inputs = reliability_inputs(table, pitch_dims)
    caption = (
        f"Reliability of pose-derived orientation, {len(table):,} poses from "
        f"{table['match_id'].nunique()} matches. No ground truth exists: "
        "the true angular error lies between the two bounds."
    )
    _save(
        plot_reliability_composite(**inputs.composite(), camera_side=camera_side, caption=caption),
        output_dir / FIGURE_1,
    )
    del table, inputs

    _say("4/4 figure 2, XY proxy against the pose blind-side label")
    if with_test:
        train_all, test_all = load_tables()
        train, _ = inclusion_flow(train_all, with_pose=False)
    else:
        test_all = load_test_table()
    test, flow = inclusion_flow(test_all, with_pose=True)
    test, poses = attach_pose(test)
    if with_test:
        results = compute(train, test, poses)
        agreement: pd.DataFrame = results["proxy_agreement"]  # type: ignore[assignment]
    else:
        features = {bound: pose_feature_table(test, poses, bound=bound) for bound in BOUNDS}
        weights = cluster_bootstrap_weights(
            test["cluster"].to_numpy(), test["match_id"].to_numpy(), N_BOOTSTRAP, seed=0
        )
        agreement = proxy_agreement(test, features, weights)
    caption = (
        f"{len(test):,} on-ball engagements in {test['cluster'].nunique():,} pressed possessions, "
        f"{test['match_id'].nunique()} matches. Pose label: p_blind_side above 0.5. XY proxy: "
        "presser more than 100 degrees off the carrier's direction of travel.\n"
        "Exploratory. Kappa measures agreement, not accuracy: without ground truth it cannot "
        "say which label is right."
    )
    _save(plot_proxy_agreement(agreement, caption=caption), output_dir / FIGURE_2)
    _say("")
    _say(flow.to_string(index=False))
    _say("")
    _say("\n".join(_agreement_lines(agreement)))
    if with_test:
        _say("")
        _say("\n".join(_test_lines(results)))


def reproduce_fast(output_dir: Path) -> None:
    """Run the pipeline on the sample phase and draw stamped stand-ins for both figures."""
    from blindside.analysis.camera import estimate_camera, read_corners
    from blindside.analysis.pipeline import engagement_geometry, reliability_inputs
    from blindside.data.download import print_progress
    from blindside.data.events import load_match_metadata
    from blindside.data.manifest import POSE_FPS
    from blindside.data.opendata import sample_pose_file
    from blindside.data.paths import SAMPLE_MATCH_ID
    from blindside.data.tracking import cached_positions, download_tracking, position_grid
    from blindside.features.orientation_table import cached_orientation_table
    from blindside.features.pose_features import (
        BOUNDS,
        FOV_HALF_ANGLE_DEG,
        pose_feature_table,
        read_engagement_poses,
        select_pose_frames,
    )
    from blindside.features.xy_baseline import baseline_table
    from blindside.viz.agreement import plot_engagement_geometry
    from blindside.viz.reliability import plot_reliability_composite

    match_id = SAMPLE_MATCH_ID
    _say("1/3 sample phase and orientation table")
    sample = sample_pose_file()
    table = cached_orientation_table(sample, match_id)
    metadata = load_match_metadata(match_id)
    pitch_dims = {match_id: (float(metadata["pitch_length"]), float(metadata["pitch_width"]))}
    _say(f"  {len(table):,} poses in {table['frame'].nunique()} frames")

    _say("2/3 stand-in for figure 1, reliability on the sample phase")
    _, periods, corners = read_corners(sample, step=1)
    camera = estimate_camera(corners, periods, match_id) if len(corners) else None
    camera_side = camera.side if camera is not None and camera.reliable else None
    inputs = reliability_inputs(table, pitch_dims, min_zone_poses=FAST_MIN_ZONE_POSES)
    caption = f"{FAST_STAMP} {len(table):,} poses. The README figure uses both full matches."
    _save(
        plot_reliability_composite(**inputs.composite(), camera_side=camera_side, caption=caption),
        output_dir / FAST_FIGURE_1,
    )

    _say("3/3 stand-in for figure 2, both blind-side labels on the engagements of the sample")
    download_tracking(match_id, progress=print_progress(f"  tracking {match_id}"))
    engagements = baseline_table(match_id, with_outcomes=False)
    keys = table[["frame", "player_id"]]
    engagements["pose_frame"] = select_pose_frames(engagements, keys)
    engagements = engagements[
        (engagements["pose_frame"] >= 0) & engagements["has_tracking"]
    ].reset_index(drop=True)
    wanted = set(zip(engagements["pose_frame"].tolist(), engagements["presser_id"].tolist(),
                     strict=True)) | set(
        zip(engagements["pose_frame"].tolist(), engagements["carrier_id"].tolist(), strict=True)
    )  # fmt: skip
    poses = read_engagement_poses(sample, wanted, match_id)
    features = {bound: pose_feature_table(engagements, poses, bound=bound) for bound in BOUNDS}
    geometry = engagement_geometry(
        engagements, poses, features, position_grid(cached_positions(match_id))
    )
    geometry["time_s"] = (engagements["pose_frame"] - int(table["frame"].min())) / POSE_FPS
    caption = (
        f"{FAST_STAMP} {len(geometry)} on-ball engagements.\nREADME figure 2 compares the two "
        "labels over 1,875 engagements and needs the full run: blindside reproduce"
    )
    _save(
        plot_engagement_geometry(geometry, half_angle_deg=FOV_HALF_ANGLE_DEG, caption=caption),
        output_dir / FAST_FIGURE_2,
    )
    for row in geometry.to_dict("records"):
        proxy = "blind" if row["approach_angle_heading_deg"] > FOV_HALF_ANGLE_DEG else "in view"
        _say(
            f"  engagement at {row['time_s']:.1f} s: p_blind_side under the lower / upper bound, "
            f"head {row['p_head_lower']:.2f} / {row['p_head_upper']:.2f}, "
            f"torso {row['p_torso_lower']:.2f} / {row['p_torso_upper']:.2f}. XY proxy: {proxy}"
        )
    _say("The fast path is a smoke test. Run `blindside reproduce` for the README figures.")


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the ``blindside`` command."""
    parser = argparse.ArgumentParser(
        prog="blindside",
        description="Blind-side pressing from broadcast body pose, with measurement uncertainty.",
    )
    parser.add_argument("--version", action="version", version=f"blindside {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    reproduce = commands.add_parser(
        "reproduce",
        help="regenerate the two README figures",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = reproduce.add_mutually_exclusive_group()
    mode.add_argument(
        "--fast",
        action="store_true",
        help="sample phase only, under five minutes, writes stamped stand-ins",
    )
    mode.add_argument(
        "--with-test",
        action="store_true",
        help="also refit the XY baseline on 18 matches and rerun the pre-registered test",
    )
    reproduce.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"figure directory, default {DEFAULT_OUTPUT} ({FAST_OUTPUT} with --fast)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``blindside`` command. Returns the exit code."""
    args = build_parser().parse_args(argv)
    start = time.perf_counter()
    try:
        if args.fast:
            reproduce_fast(args.output_dir or FAST_OUTPUT)
        else:
            reproduce_full(args.output_dir or DEFAULT_OUTPUT, with_test=args.with_test)
    except DownloadError as error:
        print(
            f"error: {error}\nCheck the network connection and run the command again. "
            "Verified files are kept and interrupted downloads resume.",
            file=sys.stderr,
        )
        return 1
    except (FileNotFoundError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    _say(f"done in {time.perf_counter() - start:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
