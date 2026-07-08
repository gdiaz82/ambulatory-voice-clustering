"""Hyperparameter search over UMAP + HDBSCAN.

Unifies the ~10 near-duplicate ``optuna_search_*`` / ``grid_search`` scripts into a
single, resumable searcher driven by config. Both a deterministic grid and an
Optuna study share one objective: fit UMAP+HDBSCAN on a balanced sample and score
the resulting clustering with the internal metrics.

Results are written incrementally to ``<outputs>/search_results.csv`` and completed
combinations are skipped on restart, so a run is safe to interrupt.
"""

from __future__ import annotations

import logging
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from voice_clustering.config import Config, HDBSCANParams, UMAPParams
from voice_clustering.evaluation.metrics import score_clustering
from voice_clustering.features.filters import balanced_sample
from voice_clustering.models.backend import Backend, get_backend

logger = logging.getLogger(__name__)

_PARAM_COLS = ["n_components", "n_neighbors", "min_dist", "min_cluster_size", "min_samples"]


def _evaluate(backend: Backend, x_scaled: np.ndarray, params: dict, config: Config) -> dict:
    """Fit UMAP+HDBSCAN for one parameter set and return a result row."""
    umap_params = UMAPParams(
        n_components=params["n_components"],
        n_neighbors=params["n_neighbors"],
        min_dist=params["min_dist"],
    )
    hdb_params = HDBSCANParams(
        min_cluster_size=params["min_cluster_size"],
        min_samples=params["min_samples"],
    )

    umap_model = backend.fit_umap(x_scaled, umap_params, config.seed)
    embedding = backend.umap_transform(umap_model, x_scaled)
    hdb_model = backend.fit_hdbscan(embedding, hdb_params)
    labels = backend.hdbscan_labels(hdb_model)
    backend.free_memory()

    metrics = score_clustering(
        embedding, labels, sample_size=config.search.metric_sample_size, seed=config.seed
    )
    return {**params, **metrics.as_dict()}


def _prepare_sample(
    config: Config, df: pd.DataFrame, features: list[str], backend: Backend
) -> np.ndarray:
    sample = balanced_sample(df, config.search.sample_per_group, seed=config.seed)
    x = sample[features].to_numpy(dtype=np.float64)
    x_scaled, _ = backend.standardize(x, x)
    return x_scaled


def _load_done(results_file: Path) -> tuple[set[tuple], list[dict]]:
    if not results_file.exists():
        return set(), []
    prev = pd.read_csv(results_file)
    done = set(zip(*(prev[c] for c in _PARAM_COLS), strict=True))
    return done, prev.to_dict("records")


def run_grid_search(config: Config, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Exhaustive grid search, resumable via the on-disk results CSV.

    Args:
        config: Pipeline configuration (``config.search.grid`` defines the grid).
        df: Feature table to sample from.
        features: Feature columns to cluster on.

    Returns:
        The full results table (previously completed rows included).
    """
    backend = get_backend(config.backend)
    x_scaled = _prepare_sample(config, df, features, backend)

    grid = config.search.grid
    combos = list(product(*(grid[c] for c in _PARAM_COLS)))
    results_file = config.paths.outputs_dir / "search_results.csv"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    done, results = _load_done(results_file)

    for values in combos:
        params = dict(zip(_PARAM_COLS, values, strict=True))
        key = tuple(values)
        if key in done:
            logger.info("Skipping already-scored combo %s", params)
            continue
        logger.info("Scoring %s", params)
        row = _evaluate(backend, x_scaled, params, config)
        results.append(row)
        done.add(key)
        pd.DataFrame(results).to_csv(results_file, index=False)
        logger.info(
            "  silhouette=%.4f dbcv=%.4f db=%.4f noise=%.4f clusters=%d",
            row["silhouette"],
            row["dbcv"],
            row["davies_bouldin"],
            row["noise_ratio"],
            row["n_clusters"],
        )

    return pd.DataFrame(results)


def run_optuna_search(config: Config, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Optuna search maximising DBCV over the same parameter space.

    Requires the optional ``optuna`` dependency (``pip install .[search]``).
    """
    import optuna

    backend = get_backend(config.backend)
    x_scaled = _prepare_sample(config, df, features, backend)
    grid = config.search.grid
    rows: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_components": trial.suggest_categorical("n_components", grid["n_components"]),
            "n_neighbors": trial.suggest_categorical("n_neighbors", grid["n_neighbors"]),
            "min_dist": trial.suggest_categorical("min_dist", grid["min_dist"]),
            "min_cluster_size": trial.suggest_categorical(
                "min_cluster_size", grid["min_cluster_size"]
            ),
            "min_samples": trial.suggest_categorical("min_samples", grid["min_samples"]),
        }
        row = _evaluate(backend, x_scaled, params, config)
        rows.append(row)
        dbcv = row["dbcv"]
        return -1.0 if np.isnan(dbcv) else dbcv

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config.search.optuna_trials)

    results = pd.DataFrame(rows)
    results_file = config.paths.outputs_dir / "search_results_optuna.csv"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_file, index=False)
    logger.info("Best trial: %s (DBCV=%.4f)", study.best_params, study.best_value)
    return results


def run_search(config: Config, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Dispatch to grid or Optuna search based on ``config.search.method``."""
    if config.search.method == "optuna":
        return run_optuna_search(config, df, features)
    return run_grid_search(config, df, features)
