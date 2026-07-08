"""Tests for the subsequence builder and its temporal-contiguity logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from voice_clustering.config import SubsequenceConfig
from voice_clustering.features.subsequences import (
    aggregate_subject,
    aggregated_feature_names,
    valid_starts,
)

CFG = SubsequenceConfig(
    length=20, stride=10, nominal_delta_ms=50.0, tolerance_ms=25.0, max_gap_ms=125.0, max_gaps=1
)
FEATURES = ["cppall", "freq"]


def _ts_ns(n: int, step_ms: float = 50.0, start: int = 0) -> np.ndarray:
    return (start + np.arange(n) * step_ms * 1e6).astype(np.int64)


def test_valid_starts_contiguous_block():
    # 40 evenly spaced windows, stride 10 -> starts at 0,10,20 (span of 20 fits).
    starts = valid_starts(_ts_ns(40), CFG)
    assert starts.tolist() == [0, 10, 20]


def test_valid_starts_rejects_hard_break():
    ts = _ts_ns(20)
    ts[10:] += int(500 * 1e6)  # a 0.5 s gap in the middle = hard break
    assert valid_starts(ts, CFG).size == 0


def test_valid_starts_tolerates_single_gap():
    ts = _ts_ns(20)
    ts[10:] += int(50 * 1e6)  # one extra 50 ms -> one 100 ms step, tolerated
    assert valid_starts(ts, CFG).tolist() == [0]


def test_too_short_block_yields_nothing():
    assert valid_starts(_ts_ns(19), CFG).size == 0


def test_aggregate_subject_shapes_and_stats():
    n = 40
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(_ts_ns(n), unit="ns"),
            "subject_id": "PF001",
            "week": "Pre",
            "date": "20120101",
            "cppall": np.arange(n, dtype=float),
            "freq": np.full(n, 200.0),
        }
    )
    agg = aggregate_subject(df, FEATURES, CFG)
    assert agg is not None
    assert len(agg) == 3  # starts 0,10,20
    assert set(aggregated_feature_names(FEATURES)) <= set(agg.columns)
    # freq is constant -> std 0, median 200 for every subsequence
    assert np.allclose(agg["freq_std"], 0.0)
    assert np.allclose(agg["freq_med"], 200.0)
    # first subsequence spans windows 0..19 -> median of arange(20) == 9.5
    assert np.isclose(agg["cppall_med"].iloc[0], 9.5)


def test_subsequences_never_cross_days():
    # Two days of 20 windows each; a subsequence must not straddle the boundary.
    day1 = _ts_ns(20, start=0)
    day2 = _ts_ns(20, start=int(10 * 3600 * 1e9))  # 10 h later, different date
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(np.concatenate([day1, day2]), unit="ns"),
            "subject_id": "PF001",
            "week": "Pre",
            "date": ["20120101"] * 20 + ["20120102"] * 20,
            "cppall": np.arange(40, dtype=float),
            "freq": np.full(40, 200.0),
        }
    )
    agg = aggregate_subject(df, FEATURES, CFG)
    assert len(agg) == 2  # one per day, none crossing the boundary
    assert set(agg["date"]) == {"20120101", "20120102"}
