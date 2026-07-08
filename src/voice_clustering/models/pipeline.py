"""End-to-end UMAP -> HDBSCAN clustering pipeline.

Orchestrates the modelling stage described in thesis section 3.3:

1. Standardize features (z-score) using stats fit on a balanced sample.
2. Fit UMAP on that sample; project the full dataset in batches.
3. Fit HDBSCAN on the sample's embedding.
4. Assign labels to the full dataset via batched approximate prediction.

The result is a copy of the input frame with ``umap_0..umap_k`` and ``cluster``
columns appended.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from voice_clustering.config import Config
from voice_clustering.features.filters import balanced_sample
from voice_clustering.models import clusterer, reducer
from voice_clustering.models.backend import Backend, get_backend

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """Output of :meth:`ClusteringPipeline.run`."""

    frame: pd.DataFrame  # input rows + umap_* + cluster columns
    embedding: np.ndarray  # full UMAP embedding
    labels: np.ndarray  # full cluster labels (-1 = noise)
    n_clusters: int
    noise_ratio: float


class ClusteringPipeline:
    """Configurable UMAP + HDBSCAN pipeline over a feature matrix."""

    def __init__(self, config: Config, features: list[str], backend: Backend | None = None):
        self.config = config
        self.features = features
        self.backend = backend or get_backend(config.backend)

    def run(self, df: pd.DataFrame, group_col: str = "week") -> ClusteringResult:
        """Fit on a balanced sample and label every row of ``df``.

        Args:
            df: Feature table containing at least ``self.features`` and ``group_col``.
            group_col: Column used to balance the fitting sample (clinical condition).

        Returns:
            A :class:`ClusteringResult`.
        """
        model_cfg = self.config.model
        x_all = df[self.features].to_numpy(dtype=np.float64)

        if model_cfg.fit_sample_per_group:
            sample = balanced_sample(
                df,
                model_cfg.fit_sample_per_group,
                group_col=group_col,
                seed=self.config.seed,
            )
            x_sample = sample[self.features].to_numpy(dtype=np.float64)
        else:
            x_sample = x_all

        logger.info("Standardizing (%d sample rows, %d total rows).", len(x_sample), len(x_all))
        x_sample_scaled, x_all_scaled = self.backend.standardize(x_sample, x_all)

        logger.info("Fitting UMAP on the balanced sample.")
        umap_model = self.backend.fit_umap(x_sample_scaled, model_cfg.umap, self.config.seed)
        sample_embedding = self.backend.umap_transform(umap_model, x_sample_scaled)
        embedding = reducer.batched_transform(
            self.backend, umap_model, x_all_scaled, model_cfg.transform_batch_size
        )

        logger.info("Fitting HDBSCAN on the sample embedding.")
        hdb_model = self.backend.fit_hdbscan(sample_embedding, model_cfg.hdbscan)
        labels = clusterer.batched_predict(
            self.backend, hdb_model, embedding, model_cfg.transform_batch_size
        )

        n_clusters = int(labels.max() + 1) if labels.size else 0
        noise_ratio = float(np.mean(labels == -1)) if labels.size else 0.0
        logger.info("Found %d clusters; noise ratio %.4f.", n_clusters, noise_ratio)

        out = df.copy()
        for i in range(embedding.shape[1]):
            out[f"umap_{i}"] = embedding[:, i]
        out["cluster"] = labels

        return ClusteringResult(out, embedding, labels, n_clusters, noise_ratio)
