"""Command-line entry points for each pipeline stage.

Installed as console scripts (see ``pyproject.toml``):

    avc-ingest    raw .mat            -> data/interim/master.parquet
    avc-features  master.parquet      -> data/processed/features.parquet
    avc-cluster   features.parquet    -> data/processed/clustered.parquet (+ figures)
    avc-validate  clustered.parquet   -> outputs/ metrics & preservation, or a search

Each stage reads ``config/default.yaml`` (override with ``--config``) and honours
``AVC_*`` environment variables.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from voice_clustering.config import Config, load_config
from voice_clustering.utils.logging import configure_logging

logger = logging.getLogger("voice_clustering.cli")


def _base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None, help="Path to a YAML config overriding defaults.")
    parser.add_argument(
        "--backend", default=None, choices=["cpu", "gpu", "auto"], help="Compute backend."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    return parser


def _setup(args: argparse.Namespace) -> Config:
    configure_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    config = load_config(args.config)
    if args.backend:
        config = config.with_backend(args.backend)
    return config


def _cluster_features(config: Config) -> list[str]:
    """The feature columns to cluster on, depending on the configured scale."""
    from voice_clustering.features.subsequences import aggregated_feature_names

    if config.features_engineering.scale == "subsequence":
        return aggregated_feature_names(config.acoustic_features)
    return config.acoustic_features


# ─────────────────────────────────────────────────────────────────────────────
# Stages
# ─────────────────────────────────────────────────────────────────────────────
def ingest_main() -> None:
    args = _base_parser("Ingest raw .mat files into the master parquet.").parse_args()
    config = _setup(args)
    from voice_clustering.data.ingest import build_master_parquet

    build_master_parquet(config)


def features_main() -> None:
    args = _base_parser("Build the model-ready feature table from the master parquet.").parse_args()
    config = _setup(args)
    from voice_clustering.features import filters
    from voice_clustering.features.subsequences import build_subsequences

    df = pd.read_parquet(config.paths.master_parquet)
    if config.ingest.drop_singing:
        df = filters.drop_singing(df)

    out_path = config.paths.processed_dir / "features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if config.features_engineering.scale == "subsequence":
        result = build_subsequences(
            df, config.acoustic_features, config.features_engineering.subsequence
        )
        logger.info("Built %d subsequences.", len(result))
    else:
        result = df
    result.to_parquet(out_path, index=False)
    logger.info("Wrote features -> %s", out_path)


def cluster_main() -> None:
    args = _base_parser("Fit UMAP + HDBSCAN and label the feature table.").parse_args()
    config = _setup(args)
    from voice_clustering.models.pipeline import ClusteringPipeline
    from voice_clustering.visualization.plots import plot_embedding

    df = pd.read_parquet(config.paths.processed_dir / "features.parquet")
    features = _cluster_features(config)

    result = ClusteringPipeline(config, features).run(df)
    out_path = config.paths.processed_dir / "clustered.parquet"
    result.frame.to_parquet(out_path, index=False)
    logger.info("Wrote clustered table -> %s", out_path)

    plot_embedding(
        result.embedding[:, 0],
        result.embedding[:, 1],
        result.labels,
        config.paths.outputs_dir / "figures" / "embedding.png",
    )


def validate_main() -> None:
    parser = _base_parser("Score the clustering, or run a hyperparameter search.")
    parser.add_argument(
        "--search", action="store_true", help="Run the hyperparameter search instead of scoring."
    )
    args = parser.parse_args()
    config = _setup(args)
    features = _cluster_features(config)

    if args.search:
        from voice_clustering.evaluation.search import run_search

        df = pd.read_parquet(config.paths.processed_dir / "features.parquet")
        run_search(config, df, features)
        return

    from voice_clustering.evaluation.metrics import score_clustering
    from voice_clustering.evaluation.preservation import compute_preservation
    from voice_clustering.visualization.plots import plot_feature_boxplots

    df = pd.read_parquet(config.paths.processed_dir / "clustered.parquet")
    umap_cols = [c for c in df.columns if c.startswith("umap_")]
    embedding = df[umap_cols].to_numpy(dtype=np.float64)
    labels = df["cluster"].to_numpy()

    metrics = score_clustering(
        embedding, labels, sample_size=config.search.metric_sample_size, seed=config.seed
    )
    logger.info("Cluster metrics: %s", metrics.as_dict())

    x_high = df.sort_values("ts_start" if "ts_start" in df else "ts")[features].to_numpy(
        dtype=np.float64
    )
    preservation = compute_preservation(
        x_high, embedding, sample_size=config.preservation_sample_size, seed=config.seed
    )
    logger.info("Preservation metrics: %s", preservation.as_dict())

    out_dir = config.paths.outputs_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**metrics.as_dict(), **preservation.as_dict()}]).to_csv(
        out_dir / "validation_metrics.csv", index=False
    )
    plottable = [f for f in features if f in df.columns]
    plot_feature_boxplots(df, plottable, out_dir / "figures" / "boxplots.png")
