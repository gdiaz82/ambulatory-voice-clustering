"""Batched HDBSCAN label assignment via approximate prediction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from voice_clustering.models.backend import Backend

logger = logging.getLogger(__name__)


def batched_predict(
    backend: Backend, model: Any, embedding: np.ndarray, batch_size: int
) -> np.ndarray:
    """Assign HDBSCAN labels to a full embedding in row batches.

    Args:
        backend: Active compute backend.
        model: A fitted HDBSCAN model with prediction data (``backend.fit_hdbscan``).
        embedding: UMAP embedding of every row, shape (n_samples, n_components).
        batch_size: Rows per batch.

    Returns:
        Integer labels, shape (n_samples,); ``-1`` marks noise.
    """
    n = embedding.shape[0]
    n_batches = int(np.ceil(n / batch_size))
    chunks: list[np.ndarray] = []
    for i in range(n_batches):
        start, end = i * batch_size, min((i + 1) * batch_size, n)
        chunks.append(backend.hdbscan_predict(model, embedding[start:end]))
        backend.free_memory()
        logger.info("HDBSCAN predict batch %d/%d", i + 1, n_batches)
    return np.concatenate(chunks)
