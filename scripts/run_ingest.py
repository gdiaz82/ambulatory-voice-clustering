#!/usr/bin/env python
"""Thin wrapper: build the master parquet from raw .mat files.

Equivalent to the ``avc-ingest`` console script; kept so the repo is runnable with
``python scripts/run_ingest.py`` without installing it.
"""

from voice_clustering.cli import ingest_main

if __name__ == "__main__":
    ingest_main()
