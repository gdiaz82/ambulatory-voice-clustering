"""Internal cluster-validity metrics for label-free evaluation.

There is no per-window ground truth (only subject-level clinical labels), so the
clustering is judged with internal indices, exactly as in the thesis:

* **Silhouette** — cohesion vs. separation (higher is better); assumes convex clusters.
* **Davies-Bouldin** — within/between dispersion ratio (lower is better).
* **DBCV** — density-based validity, appropriate for HDBSCAN's irregular clusters
  (higher is better).

Each function degrades gracefully to ``nan`` instead of raising when its
preconditions (>=2 clusters, >=2 points per cluster) are not met — the guarded
behaviour of the original ``safe_validity_index`` / ``calculate_metrics`` helpers.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score

logger = logging.getLogger(__name__)


@dataclass
class ClusterMetrics:
    """Container for the three internal metrics plus descriptive counts."""

    n_clusters: int
    noise_ratio: float
    silhouette: float
    davies_bouldin: float
    dbcv: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def dbcv_score(x: np.ndarray, labels: np.ndarray, metric: str = "euclidean") -> float:
    """Density-Based Clustering Validation, guarded against degenerate inputs.

    Returns ``nan`` if there are fewer than two clusters or any cluster has fewer
    than two points, or if the underlying computation fails.
    """
    try:
        from hdbscan.validity import validity_index
    except ImportError:
        logger.warning("hdbscan not installed; DBCV unavailable.")
        return float("nan")

    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")
    counts = np.bincount(labels[labels >= 0])
    if np.any(counts[counts > 0] < 2):
        return float("nan")
    try:
        return float(validity_index(x.astype(np.float64), labels, metric=metric))
    except Exception as exc:  # noqa: BLE001 - metric is best-effort
        logger.warning("DBCV computation failed: %s", exc)
        return float("nan")


def compute_metrics(x: np.ndarray, labels: np.ndarray) -> ClusterMetrics:
    """Compute all internal metrics on already noise-free points.

    Args:
        x: Embedding of the points to score (noise excluded), shape (n, d).
        labels: Their cluster labels (no ``-1`` expected here).

    Returns:
        A :class:`ClusterMetrics`; unavailable metrics are ``nan``.
    """
    unique = np.unique(labels)
    n_clusters = len(unique)
    metrics = ClusterMetrics(
        n_clusters=n_clusters,
        noise_ratio=float("nan"),
        silhouette=float("nan"),
        davies_bouldin=float("nan"),
        dbcv=float("nan"),
    )
    if n_clusters < 2 or len(x) < 10:
        logger.warning(
            "Too few clusters/points for metrics (clusters=%d, n=%d).", n_clusters, len(x)
        )
        return metrics

    try:
        metrics.silhouette = float(silhouette_score(x, labels))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Silhouette failed: %s", exc)
    try:
        metrics.davies_bouldin = float(davies_bouldin_score(x, labels))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Davies-Bouldin failed: %s", exc)

    metrics.dbcv = dbcv_score(x, labels)
    return metrics


def score_clustering(
    embedding: np.ndarray, labels: np.ndarray, sample_size: int | None = None, seed: int = 42
) -> ClusterMetrics:
    """Score a full clustering, excluding noise and optionally subsampling.

    Args:
        embedding: Embedding of every point, shape (n, d).
        labels: Cluster labels including ``-1`` for noise.
        sample_size: If set, stratified-subsample the non-noise points to this many
            rows before computing the (costly) metrics.
        seed: Random seed for subsampling.

    Returns:
        A :class:`ClusterMetrics` with ``noise_ratio`` populated from the full labels.
    """
    noise_ratio = float(np.mean(labels == -1)) if labels.size else 0.0
    mask = labels != -1
    x_clu, y_clu = embedding[mask], labels[mask]

    if sample_size is not None and len(x_clu) > sample_size:
        from sklearn.model_selection import train_test_split

        x_clu, _, y_clu, _ = train_test_split(
            x_clu, y_clu, train_size=sample_size, stratify=y_clu, random_state=seed
        )

    metrics = compute_metrics(x_clu, y_clu)
    metrics.noise_ratio = noise_ratio
    return metrics
