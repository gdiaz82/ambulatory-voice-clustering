"""Spatial and temporal preservation metrics for the UMAP embedding.

Quantifies how faithfully the low-dimensional projection preserves the structure of
the original feature space (thesis section 3.5):

Spatial
    * ``trustworthiness`` — penalises false neighbours introduced by the projection.
    * ``continuity``      — penalises true neighbours lost in the projection.
    * ``normalized_stress`` — global distance distortion.

Temporal (for the time-ordered voice series)
    * ``temporal_stress`` — how proportional consecutive-step displacements are
      between the original and projected spaces.
    * ``temporal_pearson`` / ``temporal_spearman`` / ``temporal_kendall`` — Shepard-style
      monotonicity between real and projected consecutive-step changes.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.manifold import trustworthiness as _trustworthiness
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)


@dataclass
class PreservationMetrics:
    trustworthiness: float
    continuity: float
    normalized_stress: float
    temporal_stress: float
    temporal_pearson: float
    temporal_spearman: float
    temporal_kendall: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def continuity(x_high: np.ndarray, x_low: np.ndarray, n_neighbors: int = 10) -> float:
    """Continuity: the complement of trustworthiness (true neighbours retained).

    Computed as trustworthiness with the two spaces swapped, which is the standard
    equivalence.
    """
    return float(_trustworthiness(x_low, x_high, n_neighbors=n_neighbors))


def normalized_stress(x_high: np.ndarray, x_low: np.ndarray) -> float:
    """Kruskal-style normalized stress between high- and low-dim pairwise distances."""
    d_high = pairwise_distances(x_high)
    d_low = pairwise_distances(x_low)
    iu = np.triu_indices_from(d_high, k=1)
    dh, dl = d_high[iu], d_low[iu]
    denom = np.sum(dh**2)
    if denom == 0:
        return float("nan")
    return float(np.sqrt(np.sum((dh - dl) ** 2) / denom))


def temporal_metrics(x_high: np.ndarray, x_low: np.ndarray) -> dict[str, float]:
    """Consecutive-step (temporal) stress and Shepard correlations.

    Args:
        x_high: Time-ordered original features, shape (n, d_high).
        x_low: Time-ordered embedding, shape (n, d_low). Rows must correspond to the
            same, chronologically ordered points as ``x_high``.

    Returns:
        Dict with ``temporal_stress`` and the three temporal correlations.
    """
    step_high = np.linalg.norm(np.diff(x_high, axis=0), axis=1)
    step_low = np.linalg.norm(np.diff(x_low, axis=0), axis=1)
    if step_high.size < 2:
        nan = float("nan")
        return {
            "temporal_stress": nan,
            "temporal_pearson": nan,
            "temporal_spearman": nan,
            "temporal_kendall": nan,
        }

    denom = np.sum(step_high**2)
    stress = float(np.sqrt(np.sum((step_high - step_low) ** 2) / denom)) if denom else float("nan")
    return {
        "temporal_stress": stress,
        "temporal_pearson": float(pearsonr(step_high, step_low)[0]),
        "temporal_spearman": float(spearmanr(step_high, step_low)[0]),
        "temporal_kendall": float(kendalltau(step_high, step_low)[0]),
    }


def compute_preservation(
    x_high: np.ndarray,
    x_low: np.ndarray,
    n_neighbors: int = 10,
    sample_size: int | None = 5000,
    seed: int = 42,
) -> PreservationMetrics:
    """Compute all preservation metrics, subsampling for the O(n^2) spatial ones.

    Rows are assumed to be in chronological order (required for temporal metrics).
    Subsampling preserves order so temporal steps stay meaningful.

    Args:
        x_high: Original standardized features, shape (n, d_high).
        x_low: UMAP embedding, shape (n, d_low).
        n_neighbors: Neighbourhood size for trustworthiness/continuity.
        sample_size: Cap on rows used (distance matrices are quadratic).
        seed: Random seed for subsampling.

    Returns:
        A :class:`PreservationMetrics`.
    """
    if sample_size is not None and len(x_high) > sample_size:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(x_high), size=sample_size, replace=False))
        x_high, x_low = x_high[idx], x_low[idx]

    logger.info("Computing preservation metrics on %d points.", len(x_high))
    temporal = temporal_metrics(x_high, x_low)
    return PreservationMetrics(
        trustworthiness=float(_trustworthiness(x_high, x_low, n_neighbors=n_neighbors)),
        continuity=continuity(x_high, x_low, n_neighbors=n_neighbors),
        normalized_stress=normalized_stress(x_high, x_low),
        **temporal,
    )
