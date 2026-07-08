# Notebooks

Exploratory and reporting notebooks. These are **not** the source of truth — the
pipeline lives in `src/voice_clustering/`. The notebooks import that package and stay
thin, so logic remains tested and reusable.

Both notebooks are committed **already executed on the synthetic dataset**
(`scripts/make_sample_dataset.py`), so their figures and outputs render directly on
GitHub without running anything. The analysis structure mirrors the thesis; only the
numbers differ from the real MGH data.

- **`01_exploration.ipynb`** — master table overview: coverage per clinical condition,
  acoustic-feature distributions, and the feature correlation matrix.
- **`02_embedding_analysis.ipynb`** — the UMAP + HDBSCAN result: cluster sizes and noise,
  the 2-D latent space coloured by cluster, internal validity metrics (Silhouette /
  Davies-Bouldin / DBCV), and per-feature cluster profiles.

## Re-running them

```bash
pip install -e ".[notebooks]"

# regenerate the inputs the notebooks read
python scripts/make_sample_dataset.py    # or run_ingest.py on the real data
python scripts/run_features.py
python scripts/run_clustering.py

jupyter notebook notebooks/              # or: jupyter lab
```

To re-execute headlessly and refresh the committed outputs:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```
