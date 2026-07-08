"""Row-level filters and balanced sampling shared by the pipeline stages."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.utils import shuffle

logger = logging.getLogger(__name__)


def drop_singing(df: pd.DataFrame) -> pd.DataFrame:
    """Remove windows flagged as singing (the "noSing" variant).

    Works whether the singing flag is the raw ``voiced_singing`` column or the
    ``isSinging`` marker; if neither exists the frame is returned unchanged.
    """
    if "voiced_singing" in df.columns:
        return df[df["voiced_singing"] != 1].reset_index(drop=True)
    if "isSinging" in df.columns:
        return df[df["isSinging"] != 1].reset_index(drop=True)
    logger.warning("No singing flag found; returning frame unchanged.")
    return df


def balanced_sample(
    df: pd.DataFrame,
    n_per_group: int,
    *,
    group_col: str = "week",
    seed: int = 42,
    replace: bool = False,
) -> pd.DataFrame:
    """Draw a class-balanced, shuffled sample (``n_per_group`` rows per group).

    This is the time-balanced subsampling (stratified by clinical condition) used
    to fit UMAP/HDBSCAN and the hyperparameter search without loading millions of
    rows onto the GPU.

    Args:
        df: Source frame.
        n_per_group: Rows to draw from each group. Groups smaller than this are
            taken whole (unless ``replace=True``).
        group_col: Column defining the groups.
        seed: Random seed.
        replace: Sample with replacement (allows exceeding a group's size).

    Returns:
        The shuffled, balanced sample with a reset index.
    """
    parts = []
    for _, group in df.groupby(group_col, observed=True):
        n = n_per_group if replace else min(n_per_group, len(group))
        parts.append(group.sample(n=n, random_state=seed, replace=replace))
    sample = pd.concat(parts).reset_index(drop=True)
    sample = shuffle(sample, random_state=seed).reset_index(drop=True)
    logger.info("Balanced sample: %d rows across %d groups.", len(sample), df[group_col].nunique())
    return sample
