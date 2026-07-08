"""Tests for the guarded internal-metric helpers."""

from __future__ import annotations

import numpy as np

from voice_clustering.evaluation.metrics import compute_metrics, dbcv_score, score_clustering


def _two_blobs(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.normal(-5, 0.3, size=(n, 2))
    b = rng.normal(5, 0.3, size=(n, 2))
    x = np.vstack([a, b])
    y = np.array([0] * n + [1] * n)
    return x, y


def test_metrics_on_well_separated_blobs():
    x, y = _two_blobs()
    m = compute_metrics(x, y)
    assert m.n_clusters == 2
    assert m.silhouette > 0.8  # very well separated
    assert m.davies_bouldin < 0.5  # low is good
    assert not np.isnan(m.dbcv)


def test_single_cluster_returns_nan_metrics():
    x = np.random.default_rng(0).normal(size=(50, 2))
    m = compute_metrics(x, np.zeros(50, dtype=int))
    assert m.n_clusters == 1
    assert np.isnan(m.silhouette)
    assert np.isnan(m.dbcv)


def test_dbcv_guards_tiny_clusters():
    x = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0]])
    labels = np.array([0, 0, 1])  # cluster 1 has a single point
    assert np.isnan(dbcv_score(x, labels))


def test_score_clustering_excludes_noise_and_reports_ratio():
    x, y = _two_blobs(n=100)
    labels = y.copy()
    labels[:20] = -1  # mark 20 points as noise
    m = score_clustering(x, labels, sample_size=None)
    assert np.isclose(m.noise_ratio, 20 / len(labels))
    assert m.n_clusters == 2
