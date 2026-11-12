"""Run paths for the scene drawing. A synthetic position grid only."""

import numpy as np
import pandas as pd

from blindside.analysis.scene import run_path
from blindside.data.tracking import position_grid


def _grid():
    frames = np.arange(100, 160)
    positions = pd.DataFrame(
        {
            "frame": frames,
            "player_id": 7,
            "x": np.linspace(0.0, 5.9, 60),
            "y": 1.0,
            "period": np.where(frames < 130, 1, 2),
        }
    )
    return position_grid(positions)


def test_run_path_ends_at_the_engagement_start_and_never_looks_ahead():
    path = run_path(_grid(), 125, 7, seconds=2.0)
    assert path.shape == (21, 2)
    assert path[-1, 0] == np.float32(2.5) and path[0, 0] == np.float32(0.5)
    assert (np.diff(path[:, 0]) > 0).all()


def test_run_path_stays_inside_the_period_and_the_grid():
    # Frame 135 is in period 2, which starts at frame 130: six frames remain.
    assert len(run_path(_grid(), 135, 7, seconds=2.0)) == 6
    # The grid starts at frame 100.
    assert len(run_path(_grid(), 105, 7, seconds=2.0)) == 6
    assert len(run_path(_grid(), 125, 99, seconds=2.0)) == 0
