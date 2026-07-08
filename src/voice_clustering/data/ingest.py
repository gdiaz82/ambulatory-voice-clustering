"""Build the master parquet from raw MGH .mat files (MATLAB v7.3 / HDF5).

Refactored from the research script ``build_master_parquet.py``. Responsibilities:

* Walk ``<raw_dir>/<subject>/<week>/*.mat`` and read per-window acoustic + IBIF fields.
* Derive ``subject_id`` (person) and ``week`` (clinical condition) from the layout.
* Keep only voiced windows recorded during acquisition
  (``recordingOn == 1`` and (``voiced_speech`` or ``voiced_singing``)).
* Skip files without the ``IBIF`` group (mirrors the reference dataset).
* Drop rows with NaNs in the acoustic or IBIF columns.
* Stream one row-group per file so ~16M rows never sit in RAM at once.

Files are otherwise identical in schema to the reference ``all_data_with_metadata``
parquet used in the thesis.
"""

from __future__ import annotations

import glob
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from voice_clustering.config import Config
from voice_clustering.data import schema

logger = logging.getLogger(__name__)

_DATE_RE = re.compile(r"_(\d{8})\.mat$")


@dataclass
class IngestReport:
    """Summary statistics collected during ingestion (for logging and tests)."""

    rows_raw: int = 0
    rows_voiced: int = 0
    rows_final: int = 0
    rows_per_week: dict[str, int] = field(default_factory=dict)
    subject_weeks: dict[str, set[str]] = field(default_factory=dict)
    skipped_no_ibif: list[str] = field(default_factory=list)


def _date_str(path: str) -> str:
    match = _DATE_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"No YYYYMMDD date found in filename: {path}")
    return match.group(1)


def _list_days(week_dir: Path, max_days: int) -> list[str]:
    """Return the chronologically first ``max_days`` .mat files in a week folder."""
    files = [f for f in glob.glob(str(week_dir / "*.mat")) if not f.endswith("Zone.Identifier")]
    files.sort(key=_date_str)  # YYYYMMDD sorts chronologically as a string
    return files[:max_days]


def _read_vec(node: h5py.Group, key: str, n: int) -> np.ndarray:
    """Read a 1-D vector from an HDF5 node, or an all-NaN vector if it is absent."""
    if key in node:
        return np.asarray(node[key][()]).reshape(-1)
    return np.full(n, np.nan)


def _load_file(path: str) -> pd.DataFrame | None:
    """Load every window of one .mat file, or None if it lacks the IBIF group."""
    with h5py.File(path, "r") as handle:
        if "IBIF" not in handle:
            return None
        n = handle["zcrall"].shape[1]
        cols = {k: _read_vec(handle, k, n) for k in schema.TOP_LEVEL_FIELDS}
        ibif = handle["IBIF"]
        for k in schema.IBIF_FEATURES:
            cols[k] = _read_vec(ibif, k, n)
        ts_raw = _read_vec(handle, "timestamps", n)

    lengths = [n, *(len(v) for v in cols.values()), len(ts_raw)]
    m = min(lengths)
    if any(length != m for length in lengths):
        logger.warning("Unequal vector lengths in %s; truncating to %d.", path, m)
        ts_raw = ts_raw[:m]
        cols = {k: v[:m] for k, v in cols.items()}

    ts = pd.to_datetime((ts_raw - schema.MAT_DATENUM_EPOCH) * 86400.0, unit="s", errors="coerce")
    df = pd.DataFrame(cols)
    df["ts"] = ts
    return df


def _voiced_mask(df: pd.DataFrame, require_recording_on: bool) -> pd.Series:
    voiced = (df["voiced_speech"] == 1) | (df["voiced_singing"] == 1)
    if require_recording_on:
        return (df["recordingOn"] == 1) & voiced
    return voiced


def build_master_parquet(config: Config) -> IngestReport:
    """Ingest all raw .mat files into a single master parquet.

    Args:
        config: Pipeline configuration (paths, ingest options, feature list).

    Returns:
        An :class:`IngestReport` with row counts and per-subject coverage.

    Raises:
        FileNotFoundError: If the raw data directory does not exist.
    """
    raw_dir = config.paths.raw_dir
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    out_path = config.paths.master_parquet
    out_path.parent.mkdir(parents=True, exist_ok=True)

    need_non_nan = schema.ACOUSTIC_FEATURES + schema.IBIF_FEATURES
    dtype_cast = {
        "breathGroup": "uint8",
        "recordingOn": "uint8",
        "voiced_singing": "uint8",
        "voiced_speech": "uint8",
        "isSinging": "float64",
        "saturationCount": "float64",
        "date": "string",
    }

    report = IngestReport()
    subjects = sorted(d.name for d in raw_dir.iterdir() if d.is_dir())
    logger.info("Found %d subject folders under %s.", len(subjects), raw_dir)

    with pq.ParquetWriter(out_path, schema.MASTER_SCHEMA, compression="snappy") as writer:
        for subject in subjects:
            subject_dir = raw_dir / subject
            week_folders = sorted(w.name for w in subject_dir.iterdir() if w.is_dir())
            for week_folder in week_folders:
                week = schema.week_label(subject, week_folder)
                for path in _list_days(subject_dir / week_folder, config.ingest.max_days_per_week):
                    df = _load_file(path)
                    if df is None:
                        report.skipped_no_ibif.append(path)
                        continue

                    df["subject_id"] = subject
                    df["week"] = week
                    df["date"] = _date_str(path)
                    df = df[schema.MASTER_COLUMNS]
                    report.rows_raw += len(df)

                    df = df[_voiced_mask(df, config.ingest.require_recording_on)]
                    if config.ingest.drop_singing:
                        df = df[df["voiced_singing"] != 1]
                    report.rows_voiced += len(df)

                    df = df.dropna(subset=need_non_nan)
                    report.rows_final += len(df)
                    if len(df):
                        report.subject_weeks.setdefault(subject, set()).add(week)
                        report.rows_per_week[week] = report.rows_per_week.get(week, 0) + len(df)

                    df = df.astype(dtype_cast)
                    table = pa.Table.from_pandas(
                        df, schema=schema.MASTER_SCHEMA, preserve_index=False
                    )
                    writer.write_table(table)

                logger.info(
                    "%s/%s (%s) -> %d rows so far", subject, week_folder, week, report.rows_final
                )

    _log_report(report, out_path)
    return report


def _log_report(report: IngestReport, out_path: Path) -> None:
    logger.info("=" * 60)
    logger.info("Raw windows read (files with IBIF): %d", report.rows_raw)
    logger.info("Voiced windows kept:                %d", report.rows_voiced)
    logger.info("Final rows (post-cleaning):         %d", report.rows_final)
    for week in ("Pre", "Post", "Control"):
        logger.info("  %-8s: %d", week, report.rows_per_week.get(week, 0))
    logger.info("Unique subjects: %d", len(report.subject_weeks))
    if report.skipped_no_ibif:
        logger.info("Skipped %d files without an IBIF group.", len(report.skipped_no_ibif))
    logger.info("Wrote master parquet -> %s", out_path)
