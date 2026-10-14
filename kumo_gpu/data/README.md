# kumo_gpu/data — configuration and metadata only

This folder contains **no competition data**. It holds the small, aggregate inputs the Kumo workflow needs:

- `km_features_412.txt` — the list of feature columns offered to the Kumo members.
- `km_importance_412_*.tsv` — per-fold (and per-airport) gain importance of those features, used to choose each member's column subset (aggregate scores, one row per feature).
- `km_drift20.txt` — the 20 feature columns with the largest 2025→2026 distribution shift (summary statistics only).
- `MANIFEST_*.tsv`, `*.md5` — which source files the GPU workflow stages and their checksums; the files themselves are built locally from the official data (see `docs/REPRODUCE.md`).

The folder name is kept because the Kumo scripts reference these paths directly.
