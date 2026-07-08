"""GPU detection and memory helpers.

Centralises the RAPIDS/cuML availability check and the CuPy memory-pool cleanup
pattern that was copy-pasted across every research script.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def gpu_available() -> bool:
    """Return True if cuML and a usable CUDA device are importable.

    The result is cached; the check is cheap but should be stable within a process.
    """
    try:
        import cuml  # noqa: F401
        import cupy as cp  # noqa: F401

        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:  # ImportError, CUDARuntimeError, ...
        return False


def resolve_backend(requested: str) -> str:
    """Map a requested backend ("auto"/"gpu"/"cpu") to a concrete "gpu"/"cpu".

    Raises:
        RuntimeError: If "gpu" is requested but no GPU/cuML is available.
    """
    requested = requested.lower()
    if requested == "cpu":
        return "cpu"
    if requested == "gpu":
        if not gpu_available():
            raise RuntimeError("backend='gpu' requested but cuML/CUDA is not available.")
        return "gpu"
    if requested == "auto":
        backend = "gpu" if gpu_available() else "cpu"
        logger.info("Auto-selected '%s' backend.", backend)
        return backend
    raise ValueError(f"Unknown backend {requested!r}; use 'cpu', 'gpu' or 'auto'.")


@contextmanager
def free_gpu_memory() -> Iterator[None]:
    """Free CuPy's default and pinned memory pools on exit (no-op without CuPy)."""
    try:
        yield
    finally:
        try:
            import cupy as cp

            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
