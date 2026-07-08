"""Figures for interpreting the clustering.

Refactored from ``analysis_boxplots*.py`` and ``visualization.py``. Selects a
non-interactive Matplotlib backend so it runs headless (CI, servers), but uses
``force=False`` so importing this module inside a notebook does not clobber the
kernel's inline backend.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

# Qualitative, colour-blind-friendly palette (cluster 0, 1, 2, ...).
CLUSTER_COLORS = ["#F39C12", "#8E44AD", "#1ABC9C", "#E74C3C", "#2980B9", "#7F8C8D"]

FEATURE_LABELS = {
    "zcrall": "ZCR (Zero-Crossing Rate)",
    "normpeakall": "Normalized Peak",
    "spectralTiltall": "Spectral Tilt",
    "LHratioall": "L/H Ratio",
    "level": "Level (dB SPL)",
    "cppall": "CPP (dB)",
    "freq": "F0 (Hz)",
    "H1H2all": "H1-H2 (dB)",
}


def _color(cluster: int) -> str:
    return CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]


def plot_embedding(
    x: np.ndarray, y: np.ndarray, labels: np.ndarray, out_path: Path, max_points: int = 200_000
) -> Path:
    """Scatter a 2-D embedding coloured by cluster label.

    Args:
        x, y: Embedding coordinates (e.g. ``umap_0``, ``umap_1``).
        labels: Cluster labels (``-1`` = noise, drawn light grey).
        out_path: Destination PNG path (parents created).
        max_points: Randomly subsample above this many points for a readable plot.

    Returns:
        The path written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(x) > max_points:
        idx = np.random.default_rng(42).choice(len(x), size=max_points, replace=False)
        x, y, labels = x[idx], y[idx], labels[idx]

    fig, ax = plt.subplots(figsize=(8, 7))
    noise = labels == -1
    if noise.any():
        ax.scatter(x[noise], y[noise], s=2, c="#D5D8DC", alpha=0.4, label="noise")
    for cluster in sorted(set(labels[~noise])):
        m = labels == cluster
        ax.scatter(x[m], y[m], s=2, c=_color(int(cluster)), alpha=0.6, label=f"cluster {cluster}")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Latent voice space (UMAP + HDBSCAN)")
    ax.legend(markerscale=4, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote embedding figure -> %s", out_path)
    return out_path


def plot_feature_boxplots(
    df: pd.DataFrame, features: list[str], out_path: Path, label_col: str = "cluster"
) -> Path:
    """Boxplot each feature's distribution across clusters, one panel per feature.

    Args:
        df: Frame with ``features`` columns and a cluster ``label_col``.
        features: Feature columns to plot.
        out_path: Destination PNG path (parents created).
        label_col: Column holding cluster labels.

    Returns:
        The path written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clusters = sorted(c for c in df[label_col].unique() if c != -1)

    n = len(features)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feature in zip(axes, features, strict=False):
        data = [df.loc[df[label_col] == c, feature].to_numpy() for c in clusters]
        bp = ax.boxplot(
            data, patch_artist=True, showfliers=False, tick_labels=[str(c) for c in clusters]
        )
        for patch, cluster in zip(bp["boxes"], clusters, strict=True):
            patch.set_facecolor(_color(int(cluster)))
            patch.set_alpha(0.7)
        ax.set_title(FEATURE_LABELS.get(feature, feature), fontsize=10)
        ax.set_xlabel("cluster")
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Feature distributions by cluster", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote boxplot figure -> %s", out_path)
    return out_path
