# Ambulatory Voice Clustering

Unsupervised discovery of latent phonation patterns in **ambulatory voice signals**
captured by a neck-surface accelerometer, using **UMAP** dimensionality reduction
followed by **HDBSCAN** density-based clustering — scaled to tens of millions of
50 ms windows, and runnable on **CPU or GPU** from the same code.

> Productionised from my undergraduate thesis (*Reducción de dimensión y clustering
> no supervisado para el análisis de señales de voz ambulatorias captadas por
> acelerómetro*, Universidad Técnica Federico Santa María) and its follow-up work.

---

## Why this problem is hard

Voice disorders caused by **vocal hyperfunction** (chronic, inefficient use of the
laryngeal musculature) lack stable objective markers, and vocal behaviour varies
enormously across a normal day. A neck accelerometer worn during daily life records
phonation continuously (11,025 Hz), producing **millions of windows per subject with
no per-window labels** — only a subject-level clinical condition is known.

This makes it a genuinely **unsupervised** problem: there is no ground truth to train
against, so the pipeline must (a) find structure that is stable and interpretable, and
(b) be validated with **internal** metrics rather than accuracy.

## What the pipeline finds

On 13 patient/control pairs from Massachusetts General Hospital (~16M windows), the
tuned pipeline recovers a **stable three-cluster latent structure**:

| Cluster | Interpretation |
|--------|-----------------|
| 0 | Baseline everyday **modal phonation** |
| 1 | **Singing** voice |
| 2 | Minority cluster with abnormally high glottal airflow (likely modelling artifacts) |

**Best configuration** (`n_components=9, n_neighbors=50, min_dist=0.05`,
`min_cluster_size`/`min_samples` tuned): **DBCV 0.856**, **Silhouette 0.671**,
3 clusters, <0.001% noise — a dense, separable latent space.

---

## Architecture

```
 raw .mat (HDF5)        master.parquet         features.parquet        clustered.parquet
 per-subject/week   →   one row / 50 ms    →   windows OR 1 s      →    + umap_0..k
 (acoustic + IBIF)      voiced, cleaned        subsequences             + cluster label
        │                     │                      │                        │
   data/ingest.py      features/subsequences.py  models/pipeline.py    evaluation/*.py
   (schema-checked,    (contiguity-aware         (StandardScaler →      (Silhouette, DBCV,
    streamed writes)    median+std agg)           UMAP → HDBSCAN,        Davies-Bouldin,
                                                  fit-on-sample /        spatial+temporal
                                                  transform-in-batches)  preservation)
```

Every stage is a small, tested module behind a CLI. The **compute backend is
abstracted** (`models/backend.py`): with NVIDIA RAPIDS present the pipeline runs UMAP
and HDBSCAN on the GPU (cuML/cuDF/CuPy); otherwise it falls back to `umap-learn` +
`hdbscan` on CPU. The same commands work in both cases.

## Quickstart

No GPU and no clinical data required — a synthetic, schema-compatible dataset lets you
run the whole thing in under a minute:

```bash
pip install -e ".[search,dev]"     # or: make install

make pipeline                      # sample → features → cluster → validate (synthetic)
# equivalently:
python scripts/make_sample_dataset.py
python scripts/run_features.py
python scripts/run_clustering.py
python scripts/run_validation.py

make test                          # run the test suite
```

Outputs land in `outputs/` (metrics CSV) and `outputs/figures/` (UMAP scatter,
per-feature cluster boxplots).

### On the real data

Point the pipeline at the raw MGH dataset (layout in [`data/SCHEMA.md`](data/SCHEMA.md))
and run the ingest stage first:

```bash
export AVC_RAW_DIR=/path/to/mgh/data
python scripts/run_ingest.py           # raw .mat → data/interim/master.parquet
python scripts/run_features.py
python scripts/run_clustering.py --backend gpu     # auto-detected if omitted
python scripts/run_validation.py --search          # UMAP+HDBSCAN hyperparameter search
```

## Configuration

All behaviour is driven by [`config/default.yaml`](config/default.yaml) — features,
sampling sizes, UMAP/HDBSCAN parameters, the analysis scale (`window` vs 1 s
`subsequence`), and the search grid. Override it three ways, in increasing precedence:

1. edit / copy the YAML and pass `--config my.yaml`;
2. set `AVC_*` environment variables (see [`.env.example`](.env.example));
3. pass CLI flags (`--backend`, `--search`, `--verbose`).

The config is parsed into a **frozen, typed `Config` dataclass** (`config.py`), so no
module reads raw dicts or `os.environ`.

## Repository layout

```
src/voice_clustering/
├── config.py              # typed, layered configuration
├── data/                  # ingest.py (.mat → parquet), schema.py (data contract)
├── features/              # subsequences.py (1 s aggregation), filters.py (sampling/QC)
├── models/                # backend.py (CPU/GPU), reducer.py, clusterer.py, pipeline.py
├── evaluation/            # metrics.py, search.py (grid + Optuna), preservation.py
├── visualization/         # plots.py (embedding scatter, cluster boxplots)
├── utils/                 # logging.py, gpu.py
└── cli.py                 # avc-ingest / avc-features / avc-cluster / avc-validate
scripts/                   # thin runnable wrappers + synthetic-data generator
tests/                     # pytest suite (config, subsequences, metrics, e2e smoke)
config/  data/  notebooks/
```

## Key design decisions

- **Fit on a balanced sample, transform in batches.** UMAP and HDBSCAN are fit on a
  time-balanced sample (stratified by clinical condition), then labels/embeddings are
  extended to the full dataset via `transform` / `approximate_predict` in row batches.
  This keeps GPU/RAM bounded on datasets too large to fit at once.
- **Internal validation only.** With no per-window ground truth, clustering quality is
  judged with Silhouette, Davies-Bouldin and — because HDBSCAN produces irregular,
  density-based clusters — **DBCV**, complemented by spatial (trustworthiness,
  continuity, stress) and **temporal** (temporal stress, Shepard correlations)
  preservation metrics for the dimensionality reduction.
- **One parametrised pipeline, not many scripts.** The research code had ~50
  near-duplicate scripts (10+ search variants, 11+ validation variants). Here the
  variants — `noSing`, IBIF features, window vs subsequence scale — are **config
  flags**, not copies.
- **Backend abstraction over a hard GPU dependency.** The original code imported cuML
  at module top level and only ran on CUDA; the abstraction makes it reproducible
  anywhere while preserving the GPU fast path.

## Development

```bash
make lint        # ruff
make format      # black + ruff --fix
make test        # pytest
```

## Data & ethics

The clinical recordings are **private** and are not distributed here; only the data
*contract* ([`data/SCHEMA.md`](data/SCHEMA.md)) and a synthetic generator are included.
The accelerometer signal is not filtered by the vocal tract, which mitigates
speech-intelligibility privacy concerns, but the dataset remains protected human-subject
data.

## License

MIT — see `pyproject.toml`.
