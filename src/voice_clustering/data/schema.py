"""Canonical column names, feature groups and the master-parquet Arrow schema.

Centralising this here means the ingestion writer, the feature stage and the tests
all agree on exactly one source of truth for the data contract.
"""

from __future__ import annotations

import pyarrow as pa

# ─── Metadata columns (per window) ───────────────────────────────────────────
META_COLS: list[str] = ["ts", "subject_id", "week", "date"]

# ─── The 8 acoustic features used by the clustering pipeline ─────────────────
ACOUSTIC_FEATURES: list[str] = [
    "zcrall",
    "normpeakall",
    "spectralTiltall",
    "LHratioall",
    "level",
    "cppall",
    "freq",
    "H1H2all",
]

# ─── Aerodynamic (IBIF) features — optional, used by extended experiments ────
IBIF_FEATURES: list[str] = ["acflow", "mfdr", "oq", "naq", "h1h2", "voicedRMS"]

# ─── Per-window fields read from the top level of each .mat file ─────────────
# (Includes the flags needed for the voice-activity filter.)
TOP_LEVEL_FIELDS: list[str] = [
    "cppall",
    "zcrall",
    "normpeakall",
    "spectralTiltall",
    "LHratioall",
    "H1H2all",
    "periodicity",
    "level",
    "freq",
    "dBcms2",
    "breathGroup",
    "isSinging",
    "recordingOn",
    "saturationCount",
    "voiced_singing",
    "voiced_speech",
]

# Full column order of the master parquet.
MASTER_COLUMNS: list[str] = [
    *META_COLS,
    "cppall",
    "zcrall",
    "normpeakall",
    "spectralTiltall",
    "LHratioall",
    "H1H2all",
    "periodicity",
    "level",
    "freq",
    "dBcms2",
    *IBIF_FEATURES,
    "breathGroup",
    "isSinging",
    "recordingOn",
    "saturationCount",
    "voiced_singing",
    "voiced_speech",
]

# MATLAB datenum for the Unix epoch (1970-01-01). Used to convert `timestamps`.
MAT_DATENUM_EPOCH: int = 719529

# Arrow schema of the master parquet (explicit dtypes for reproducible round-trips).
MASTER_SCHEMA = pa.schema(
    [
        ("ts", pa.timestamp("ns")),
        ("subject_id", pa.string()),
        ("week", pa.string()),
        ("date", pa.string()),
        ("cppall", pa.float64()),
        ("zcrall", pa.float64()),
        ("normpeakall", pa.float64()),
        ("spectralTiltall", pa.float64()),
        ("LHratioall", pa.float64()),
        ("H1H2all", pa.float64()),
        ("periodicity", pa.float64()),
        ("level", pa.float64()),
        ("freq", pa.float64()),
        ("dBcms2", pa.float64()),
        ("acflow", pa.float64()),
        ("mfdr", pa.float64()),
        ("oq", pa.float64()),
        ("naq", pa.float64()),
        ("h1h2", pa.float64()),
        ("voicedRMS", pa.float64()),
        ("breathGroup", pa.uint8()),
        ("isSinging", pa.float64()),
        ("recordingOn", pa.uint8()),
        ("saturationCount", pa.float64()),
        ("voiced_singing", pa.uint8()),
        ("voiced_speech", pa.uint8()),
    ]
)


def week_label(subject_id: str, week_folder: str) -> str:
    """Map a (subject, week-folder) pair to a clinical condition label.

    NF* subjects are healthy controls; PF* patients have a pre-therapy week (W1)
    and a post-therapy week (W2).

    Raises:
        ValueError: If the week folder is not recognised for a patient.
    """
    if subject_id.startswith("NF"):
        return "Control"
    if week_folder == "W1":
        return "Pre"
    if week_folder == "W2":
        return "Post"
    raise ValueError(f"Unexpected week folder {week_folder!r} for subject {subject_id!r}.")
