#!/usr/bin/env python
"""Generate a tiny synthetic dataset so the pipeline is runnable without MGH data.

The real ambulatory voice recordings are private clinical data (~35 GB) and are not
distributed with this repo. This script fabricates a small, schema-compatible master
parquet with a few subjects, days and planted cluster structure, so a reviewer can
run ``avc-features`` / ``avc-cluster`` / ``avc-validate`` end-to-end in seconds.

    python scripts/make_sample_dataset.py

Writes ``data/interim/master.parquet``.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from voice_clustering.config import load_config
from voice_clustering.data import schema


def _synth_subject(
    subject_id: str, week: str, date: str, n: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Three latent phonation modes with distinct acoustic centroids + noise."""
    centroids = {
        0: dict(
            zcrall=0.05,
            normpeakall=0.8,
            spectralTiltall=-12,
            LHratioall=20,
            level=70,
            cppall=18,
            freq=200,
            H1H2all=6,
        ),
        1: dict(
            zcrall=0.15,
            normpeakall=0.5,
            spectralTiltall=-8,
            LHratioall=12,
            level=80,
            cppall=12,
            freq=350,
            H1H2all=2,
        ),
        2: dict(
            zcrall=0.02,
            normpeakall=0.9,
            spectralTiltall=-16,
            LHratioall=28,
            level=60,
            cppall=22,
            freq=160,
            H1H2all=9,
        ),
    }
    modes = rng.choice(list(centroids), size=n, p=[0.6, 0.3, 0.1])
    cols: dict[str, np.ndarray] = {}
    for feat in schema.ACOUSTIC_FEATURES:
        base = np.array([centroids[m][feat] for m in modes], dtype=float)
        scale = 0.08 * (abs(base).mean() + 1e-6)
        cols[feat] = base + rng.normal(0, scale, size=n)

    start = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:]} 09:00:00")
    ts = start + pd.to_timedelta(np.arange(n) * 50, unit="ms")

    df = pd.DataFrame(cols)
    df["ts"] = ts
    df["subject_id"] = subject_id
    df["week"] = week
    df["date"] = date
    # Remaining master columns are filled with neutral placeholders.
    for extra in ("periodicity", "dBcms2", *schema.IBIF_FEATURES, "isSinging", "saturationCount"):
        df[extra] = rng.normal(0, 1, size=n) if extra in schema.IBIF_FEATURES else 0.0
    for flag in ("breathGroup", "recordingOn", "voiced_speech"):
        df[flag] = 1
    df["voiced_singing"] = 0
    return df[schema.MASTER_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic master parquet.")
    parser.add_argument("--rows-per-day", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config()
    rng = np.random.default_rng(args.seed)

    plan = [
        ("PF001", "Pre", ["20120101", "20120102"]),
        ("PF001", "Post", ["20120701", "20120702"]),
        ("NF001", "Control", ["20120101", "20120102"]),
        ("NF002", "Control", ["20120103", "20120104"]),
    ]
    parts = [
        _synth_subject(sid, week, date, args.rows_per_day, rng)
        for sid, week, dates in plan
        for date in dates
    ]
    df = pd.concat(parts, ignore_index=True)

    out = config.paths.master_parquet
    out.parent.mkdir(parents=True, exist_ok=True)
    df.astype({"date": "string"}).to_parquet(out, index=False)
    print(f"Wrote {len(df):,} synthetic rows -> {out}")


if __name__ == "__main__":
    main()
