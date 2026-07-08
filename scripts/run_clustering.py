#!/usr/bin/env python
"""Thin wrapper: fit UMAP + HDBSCAN and label the data. See ``avc-cluster``."""

from voice_clustering.cli import cluster_main

if __name__ == "__main__":
    cluster_main()
