# Data schema & layout

The raw dataset (private clinical recordings from Massachusetts General Hospital) is
**not** included in this repository. This document specifies the contract the code
expects so the pipeline can be reproduced on the real data, and so the synthetic
generator (`scripts/make_sample_dataset.py`) stays compatible.

## Directory layout (raw)

```
data/raw/
├── PF001/            # patient (phonotraumatic vocal hyperfunction)
│   ├── W1/           # pre-therapy week
│   │   ├── PF001_20120101.mat
│   │   └── ...
│   └── W2/           # post-therapy week
├── NF001/            # matched healthy control
│   └── W1/
└── ...
```

- `PF*` = patient, `NF*` = control. The letter after (`F`) is sex (all female in this cohort).
- `.mat` files are MATLAB v7.3 (HDF5). Each holds per-window acoustic fields at the top
  level and an `IBIF` group with the aerodynamic (glottal-airflow) estimates.
- Windows are 50 ms, non-overlapping, sampled from an 11,025 Hz neck-accelerometer signal.

## Condition mapping

| Folder / prefix | `week` label |
|-----------------|--------------|
| `PF*/W1`        | `Pre`        |
| `PF*/W2`        | `Post`       |
| `NF*/*`         | `Control`    |

## Master parquet (`data/interim/master.parquet`)

One row per retained 50 ms window. Full column list and Arrow dtypes are defined in
[`src/voice_clustering/data/schema.py`](../src/voice_clustering/data/schema.py).

**Acoustic features used for clustering (8):**
`zcrall`, `normpeakall`, `spectralTiltall`, `LHratioall`, `level`, `cppall`, `freq`, `H1H2all`.

**Retention rules** (applied during ingestion):
- keep only `recordingOn == 1` and (`voiced_speech == 1` or `voiced_singing == 1`);
- drop rows with NaNs in the 8 acoustic features or the 6 IBIF columns;
- keep at most the first 7 days per week;
- files lacking an `IBIF` group are skipped.

## Processed tables (`data/processed/`)

- `features.parquet` — model-ready table. At `scale: window` it is the master rows;
  at `scale: subsequence` it is one row per 1 s subsequence with `{feature}_med` /
  `{feature}_std` columns.
- `clustered.parquet` — `features.parquet` plus `umap_0..k` and `cluster` columns.
