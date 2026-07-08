"""Batched UMAP projection.

UMAP is fit on a class-balanced sample, then used to transform the full dataset in
chunks so the GPU (or RAM) never holds all rows at once — the "fit on sample,
transform in batches" pattern from the thesis.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from voice_clustering.models.backend import Backend

logger = logging.getLogger(__name__)


def batched_transform(backend: Backend, model: Any, x: np.ndarray, batch_size: int) -> np.ndarray:
    """Project ``x`` through a fitted UMAP model in row batches.

    Args:
        backend: Active compute backend.
        model: A fitted UMAP model (from ``backend.fit_umap``).
        x: Scaled feature matrix, shape (n_samples, n_features).
        batch_size: Rows per batch.

    Returns:
        The stacked embedding, shape (n_samples, n_components).
    """
    n = x.shape[0]
    n_batches = int(np.ceil(n / batch_size))
    chunks: list[np.ndarray] = []
    for i in range(n_batches):
        start, end = i * batch_size, min((i + 1) * batch_size, n)
        chunks.append(backend.umap_transform(model, x[start:end]))
        backend.free_memory()
        logger.info("UMAP transform batch %d/%d", i + 1, n_batches)
    return np.vstack(chunks)
