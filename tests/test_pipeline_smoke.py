"""End-to-end smoke test on synthetic data (CPU backend).

Exercises the full modelling path — standardize -> UMAP -> HDBSCAN -> metrics — on a
small planted-structure dataset, so a regression anywhere in the chain is caught.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voice_clustering.config import load_config
from voice_clustering.evaluation.metrics import score_clustering
from voice_clustering.models.pipeline import ClusteringPipeline

pytest.importorskip("umap")
pytest.importorskip("hdbscan")

FEATURES = ["f0", "f1", "f2"]


def _planted(n_per: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    blobs = []
    for center in ([-6, -6, -6], [0, 0, 0], [6, 6, 6]):
        blobs.append(rng.normal(center, 0.4, size=(n_per, 3)))
    x = np.vstack(blobs)
    df = pd.DataFrame(x, columns=FEATURES)
    df["week"] = rng.choice(["Pre", "Post", "Control"], size=len(df))
    return df


def test_pipeline_recovers_three_clusters():
    cfg = load_config(env={"AVC_BACKEND": "cpu"})
    # Fit on all rows (tiny dataset) with sensible small-data HDBSCAN settings.
    cfg = cfg.with_backend("cpu")
    from dataclasses import replace

    cfg = replace(
        cfg,
        model=replace(
            cfg.model,
            fit_sample_per_group=None,
            umap=replace(cfg.model.umap, n_components=2, n_neighbors=15),
            hdbscan=replace(cfg.model.hdbscan, min_cluster_size=30, min_samples=5),
        ),
    )
    df = _planted()
    result = ClusteringPipeline(cfg, FEATURES).run(df)

    assert result.n_clusters >= 2
    assert len(result.frame) == len(df)
    assert "cluster" in result.frame.columns

    metrics = score_clustering(result.embedding, result.labels, sample_size=None)
    assert metrics.silhouette > 0.3
