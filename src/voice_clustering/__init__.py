"""Unsupervised dimensionality reduction + clustering for ambulatory voice signals.

The package implements the thesis pipeline:

    raw .mat  ->  master parquet  ->  (sub)sequences  ->  UMAP  ->  HDBSCAN  ->  metrics

See ``voice_clustering.config.Config`` for the entry point and ``README.md`` for the
scientific background.
"""

from voice_clustering.config import Config, load_config

__all__ = ["Config", "load_config"]
__version__ = "0.1.0"
