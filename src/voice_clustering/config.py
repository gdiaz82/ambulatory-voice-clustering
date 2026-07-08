"""Typed configuration loading.

Configuration is layered, from lowest to highest precedence:

1. ``config/default.yaml`` (checked into the repo).
2. An optional user YAML passed explicitly (``load_config(path)``).
3. Environment variables (``AVC_*``), useful for paths and secrets in CI/CD.

The result is a frozen, fully typed :class:`Config` dataclass so the rest of the
codebase never touches raw dictionaries or ``os.environ``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    outputs_dir: Path

    @property
    def master_parquet(self) -> Path:
        return self.interim_dir / "master.parquet"


@dataclass(frozen=True)
class UMAPParams:
    n_components: int = 8
    n_neighbors: int = 50
    min_dist: float = 0.05


@dataclass(frozen=True)
class HDBSCANParams:
    min_cluster_size: int = 250
    min_samples: int = 100


@dataclass(frozen=True)
class ModelConfig:
    umap: UMAPParams = field(default_factory=UMAPParams)
    hdbscan: HDBSCANParams = field(default_factory=HDBSCANParams)
    fit_sample_per_group: int | None = 250_000
    transform_batch_size: int = 1_000_000


@dataclass(frozen=True)
class SubsequenceConfig:
    length: int = 20
    stride: int = 10
    nominal_delta_ms: float = 50.0
    tolerance_ms: float = 25.0
    max_gap_ms: float = 125.0
    max_gaps: int = 1


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    scale: str = "window"  # "window" | "subsequence"
    subsequence: SubsequenceConfig = field(default_factory=SubsequenceConfig)


@dataclass(frozen=True)
class IngestConfig:
    max_days_per_week: int = 7
    require_recording_on: bool = True
    drop_singing: bool = False


@dataclass(frozen=True)
class SearchConfig:
    method: str = "grid"
    sample_per_group: int = 200_000
    metric_sample_size: int = 10_000
    optuna_trials: int = 50
    grid: dict[str, list[Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    seed: int
    backend: str
    paths: Paths
    acoustic_features: list[str]
    ingest: IngestConfig
    features_engineering: FeatureEngineeringConfig
    model: ModelConfig
    search: SearchConfig
    preservation_sample_size: int

    def with_backend(self, backend: str) -> Config:
        """Return a copy with a different compute backend (cpu/gpu/auto)."""
        return replace(self, backend=backend)


def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    path: str | Path | None = None,
    *,
    root: Path = PROJECT_ROOT,
    env: dict[str, str] | None = None,
) -> Config:
    """Load, merge and validate configuration into a typed :class:`Config`.

    Args:
        path: Optional user YAML overriding ``config/default.yaml``.
        root: Project root used to resolve relative paths (injectable for tests).
        env: Environment mapping (defaults to ``os.environ``; injectable for tests).

    Returns:
        A frozen, fully populated :class:`Config`.
    """
    env = os.environ if env is None else env

    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    if path is not None:
        with open(path, encoding="utf-8") as fh:
            raw = _deep_merge(raw, yaml.safe_load(fh) or {})

    paths_raw = raw["paths"]
    paths = Paths(
        raw_dir=_resolve(root, env.get("AVC_RAW_DIR", paths_raw["raw_dir"])),
        interim_dir=_resolve(root, env.get("AVC_INTERIM_DIR", paths_raw["interim_dir"])),
        processed_dir=_resolve(root, env.get("AVC_PROCESSED_DIR", paths_raw["processed_dir"])),
        outputs_dir=_resolve(root, env.get("AVC_OUTPUTS_DIR", paths_raw["outputs_dir"])),
    )

    fe_raw = raw["features_engineering"]
    model_raw = raw["model"]
    search_raw = raw["search"]

    return Config(
        seed=int(env.get("AVC_SEED", raw["seed"])),
        backend=env.get("AVC_BACKEND", raw["backend"]),
        paths=paths,
        acoustic_features=list(raw["features"]["acoustic"]),
        ingest=IngestConfig(**raw["ingest"]),
        features_engineering=FeatureEngineeringConfig(
            scale=fe_raw["scale"],
            subsequence=SubsequenceConfig(**fe_raw["subsequence"]),
        ),
        model=ModelConfig(
            umap=UMAPParams(**model_raw["umap"]),
            hdbscan=HDBSCANParams(**model_raw["hdbscan"]),
            fit_sample_per_group=model_raw["fit_sample_per_group"],
            transform_batch_size=model_raw["transform_batch_size"],
        ),
        search=SearchConfig(
            method=search_raw["method"],
            sample_per_group=search_raw["sample_per_group"],
            metric_sample_size=search_raw["metric_sample_size"],
            optuna_trials=search_raw["optuna_trials"],
            grid=search_raw["grid"],
        ),
        preservation_sample_size=raw["evaluation"]["preservation_sample_size"],
    )
