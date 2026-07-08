"""Build 1-second subsequences from 50 ms windows (thesis section 3.2.2).

Refactored from ``build_subsequences_noSing_IBIF.py``. A subsequence is a sliding
window of ``length`` atomic windows (default 20 = 1 s) with a ``stride`` hop
(default 10 = 0.5 s, 50% overlap). Subsequences never cross a subject or a day and
must be temporally contiguous, tolerating at most ``max_gaps`` single missing
windows. Each subsequence is summarised by the per-feature median and standard
deviation, capturing both the central value and the local variability.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from voice_clustering.config import SubsequenceConfig

logger = logging.getLogger(__name__)


def valid_starts(ts_ns: np.ndarray, cfg: SubsequenceConfig) -> np.ndarray:
    """Return start indices of valid subsequences within one contiguous block.

    A start is valid if its ``length``-window span has zero hard breaks and at most
    ``max_gaps`` tolerated single-window gaps, judged from the inter-window deltas.

    Args:
        ts_ns: Timestamps (int64 nanoseconds) of a single (subject, date) block,
            already sorted ascending.
        cfg: Subsequence parameters.

    Returns:
        A 1-D int64 array of local start indices (possibly empty).
    """
    n = len(ts_ns)
    length, stride = cfg.length, cfg.stride
    if n < length:
        return np.empty(0, dtype=np.int64)

    deltas_ms = np.diff(ts_ns) / 1e6
    starts = np.arange(0, n - length + 1, stride)
    if starts.size == 0:
        return np.empty(0, dtype=np.int64)

    hi = cfg.nominal_delta_ms + cfg.tolerance_ms
    lo = cfg.nominal_delta_ms - cfg.tolerance_ms
    is_gap = ((deltas_ms > hi) & (deltas_ms <= cfg.max_gap_ms)).astype(np.int32)
    is_break = ((deltas_ms > cfg.max_gap_ms) | (deltas_ms < lo)).astype(np.int32)

    # Prefix sums let us count gaps/breaks inside any window in O(1).
    cs_gap = np.concatenate(([0], np.cumsum(is_gap)))
    cs_break = np.concatenate(([0], np.cumsum(is_break)))
    win_gap = cs_gap[starts + (length - 1)] - cs_gap[starts]
    win_break = cs_break[starts + (length - 1)] - cs_break[starts]

    valid = (win_break == 0) & (win_gap <= cfg.max_gaps)
    return starts[valid]


def aggregate_subject(
    df: pd.DataFrame, features: list[str], cfg: SubsequenceConfig
) -> pd.DataFrame | None:
    """Aggregate one subject's windows into 1-second subsequences.

    Args:
        df: Rows for a single subject (must contain ``ts``, ``subject_id``,
            ``week``, ``date`` and every column in ``features``).
        features: Feature columns to summarise (median + std per subsequence).
        cfg: Subsequence parameters.

    Returns:
        One row per subsequence with ``{feature}_med`` / ``{feature}_std`` columns
        plus metadata, or None if the subject yields no valid subsequences.
    """
    df = df.sort_values(["date", "ts"]).reset_index(drop=True)
    ts_ns = df["ts"].to_numpy(dtype="datetime64[ns]").astype("int64")
    subject = df["subject_id"].iloc[0]

    starts_by_day = df.groupby("date", sort=False, observed=True).indices
    global_starts: list[np.ndarray] = []
    for idx in starts_by_day.values():
        if len(idx) < cfg.length:
            continue
        base = idx[0]  # contiguous block after the sort
        local = valid_starts(ts_ns[base : base + len(idx)], cfg)
        if local.size:
            global_starts.append(base + local)

    if not global_starts:
        return None

    gstarts = np.concatenate(global_starts)
    n_subseq = gstarts.size
    length = cfg.length

    offsets = np.arange(length, dtype=np.int64)
    expanded_idx = (gstarts[:, None] + offsets[None, :]).reshape(-1)
    expanded = df.iloc[expanded_idx].reset_index(drop=True)

    ts_starts = ts_ns[gstarts].astype("datetime64[ns]")
    ids = [f"{subject}_{pd.Timestamp(t).strftime('%Y%m%dT%H%M%S%f')[:-3]}" for t in ts_starts]

    values = expanded[features].to_numpy(dtype=np.float64).reshape(n_subseq, length, len(features))
    median = np.nanmedian(values, axis=1)
    std = np.nanstd(values, axis=1, ddof=0)

    head = expanded.iloc[::length].reset_index(drop=True)
    tail = expanded.iloc[length - 1 :: length].reset_index(drop=True)
    agg = pd.DataFrame(
        {
            "subseq_id": ids,
            "subject_id": head["subject_id"],
            "week": head["week"],
            "date": head["date"],
            "ts_start": head["ts"],
            "ts_end": tail["ts"],
        }
    )
    for i, feature in enumerate(features):
        agg[f"{feature}_med"] = median[:, i]
        agg[f"{feature}_std"] = std[:, i]
    return agg


def build_subsequences(
    df: pd.DataFrame, features: list[str], cfg: SubsequenceConfig
) -> pd.DataFrame:
    """Build subsequences for every subject in ``df`` and concatenate the result.

    Args:
        df: Master-table rows (any number of subjects).
        features: Feature columns to summarise.
        cfg: Subsequence parameters.

    Returns:
        Concatenated per-subsequence aggregate table (empty if none are valid).
    """
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df = df.assign(ts=pd.to_datetime(df["ts"]))

    parts: list[pd.DataFrame] = []
    for subject, sub_df in df.groupby("subject_id", sort=True, observed=True):
        agg = aggregate_subject(sub_df, features, cfg)
        if agg is not None:
            parts.append(agg)
            logger.info("%s: %d subsequences", subject, len(agg))

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def aggregated_feature_names(features: list[str]) -> list[str]:
    """Return the ``{feature}_med`` / ``{feature}_std`` column names for ``features``."""
    return [f"{f}_{stat}" for f in features for stat in ("med", "std")]
