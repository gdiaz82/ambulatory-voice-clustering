"""Tests for layered configuration loading and env-var overrides."""

from __future__ import annotations

from pathlib import Path

from voice_clustering.config import load_config


def test_defaults_load():
    cfg = load_config(env={})
    assert cfg.seed == 42
    assert cfg.acoustic_features[0] == "zcrall"
    assert len(cfg.acoustic_features) == 8
    assert cfg.model.umap.n_components == 9  # thesis optimum


def test_env_overrides_paths_and_backend(tmp_path: Path):
    env = {"AVC_RAW_DIR": str(tmp_path / "raw"), "AVC_BACKEND": "cpu", "AVC_SEED": "7"}
    cfg = load_config(env=env)
    assert cfg.paths.raw_dir == tmp_path / "raw"
    assert cfg.backend == "cpu"
    assert cfg.seed == 7


def test_relative_paths_resolved_against_root(tmp_path: Path):
    cfg = load_config(root=tmp_path, env={})
    assert cfg.paths.master_parquet == tmp_path / "data" / "interim" / "master.parquet"


def test_with_backend_returns_copy():
    cfg = load_config(env={})
    other = cfg.with_backend("cpu")
    assert other.backend == "cpu"
    assert other is not cfg
