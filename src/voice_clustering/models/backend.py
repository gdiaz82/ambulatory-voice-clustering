"""Compute backend abstraction: run the same pipeline on CPU or GPU.

The research code hard-wired NVIDIA RAPIDS (``cuml``/``cudf``/``cupy``), so it only
ran on a CUDA machine. This module hides that behind a small interface with two
implementations:

* :class:`CPUBackend`   — scikit-learn ``StandardScaler``, ``umap-learn`` and ``hdbscan``.
* :class:`GPUBackend`   — cuML ``StandardScaler``/``UMAP``/``HDBSCAN`` on CuPy arrays.

Both expose identical methods returning NumPy arrays, so the reducer, clusterer and
metrics code is written once and is backend-independent. Use :func:`get_backend`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from voice_clustering.config import HDBSCANParams, UMAPParams
from voice_clustering.utils.gpu import resolve_backend

logger = logging.getLogger(__name__)


class Backend(ABC):
    """Backend interface. All array inputs/outputs are NumPy on the caller's side."""

    name: str

    @abstractmethod
    def standardize(self, x_train: np.ndarray, x_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Z-score using stats fit on ``x_train``; return scaled (train, all)."""

    @abstractmethod
    def fit_umap(self, x: np.ndarray, params: UMAPParams, seed: int) -> Any:
        """Fit a UMAP reducer on ``x`` and return the fitted model."""

    @abstractmethod
    def umap_transform(self, model: Any, x: np.ndarray) -> np.ndarray:
        """Project ``x`` with a fitted UMAP model; return a NumPy embedding."""

    @abstractmethod
    def fit_hdbscan(self, embedding: np.ndarray, params: HDBSCANParams) -> Any:
        """Fit HDBSCAN (with prediction data enabled) and return the model."""

    @abstractmethod
    def hdbscan_labels(self, model: Any) -> np.ndarray:
        """Return the training labels of a fitted HDBSCAN model as NumPy int."""

    @abstractmethod
    def hdbscan_predict(self, model: Any, embedding: np.ndarray) -> np.ndarray:
        """Assign labels to unseen points via approximate prediction (NumPy int)."""

    def free_memory(self) -> None:  # noqa: B027 - intentional no-op default (GPU backend overrides)
        """Release backend memory pools. No-op on CPU; overridden by the GPU backend."""


class CPUBackend(Backend):
    """scikit-learn / umap-learn / hdbscan implementation."""

    name = "cpu"

    def standardize(self, x_train, x_all):
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(x_train)
        return scaler.transform(x_train), scaler.transform(x_all)

    def fit_umap(self, x, params, seed):
        import umap

        model = umap.UMAP(
            n_neighbors=params.n_neighbors,
            min_dist=params.min_dist,
            n_components=params.n_components,
            random_state=seed,
        )
        model.fit(x)
        return model

    def umap_transform(self, model, x):
        return np.asarray(model.transform(x))

    def fit_hdbscan(self, embedding, params):
        import hdbscan

        model = hdbscan.HDBSCAN(
            min_cluster_size=params.min_cluster_size,
            min_samples=params.min_samples,
            prediction_data=True,
        )
        model.fit(embedding)
        return model

    def hdbscan_labels(self, model):
        return np.asarray(model.labels_, dtype=np.int64)

    def hdbscan_predict(self, model, embedding):
        import hdbscan

        labels, _ = hdbscan.approximate_predict(model, embedding)
        return np.asarray(labels, dtype=np.int64)


class GPUBackend(Backend):
    """NVIDIA RAPIDS (cuML) implementation. Inputs/outputs stay NumPy at the edges."""

    name = "gpu"

    def _to_gpu(self, x: np.ndarray):
        import cupy as cp

        return cp.asarray(x)

    def standardize(self, x_train, x_all):
        import cupy as cp
        from cuml.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(self._to_gpu(x_train))
        scaled_train = cp.asnumpy(scaler.transform(self._to_gpu(x_train)))
        scaled_all = cp.asnumpy(scaler.transform(self._to_gpu(x_all)))
        return scaled_train, scaled_all

    def fit_umap(self, x, params, seed):
        from cuml.manifold.umap import UMAP

        model = UMAP(
            n_neighbors=params.n_neighbors,
            min_dist=params.min_dist,
            n_components=params.n_components,
            random_state=seed,
        )
        model.fit(self._to_gpu(x))
        return model

    def umap_transform(self, model, x):
        import cupy as cp

        return cp.asnumpy(model.transform(self._to_gpu(x)))

    def fit_hdbscan(self, embedding, params):
        from cuml.cluster import HDBSCAN

        model = HDBSCAN(
            min_cluster_size=params.min_cluster_size,
            min_samples=params.min_samples,
            prediction_data=True,
        )
        model.fit(self._to_gpu(embedding))
        return model

    def hdbscan_labels(self, model):
        import cupy as cp

        return cp.asnumpy(model.labels_).astype(np.int64)

    def hdbscan_predict(self, model, embedding):
        import cupy as cp
        from cuml.cluster.hdbscan import approximate_predict

        labels, _ = approximate_predict(model, self._to_gpu(embedding))
        return cp.asnumpy(labels).astype(np.int64)

    def free_memory(self):
        from voice_clustering.utils.gpu import free_gpu_memory

        with free_gpu_memory():
            pass


def get_backend(requested: str = "auto") -> Backend:
    """Return a concrete backend for the requested mode ("auto"/"cpu"/"gpu")."""
    resolved = resolve_backend(requested)
    backend: Backend = GPUBackend() if resolved == "gpu" else CPUBackend()
    logger.info("Using %s backend.", backend.name.upper())
    return backend
