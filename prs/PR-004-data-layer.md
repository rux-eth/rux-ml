# PR-004: Data layer

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time).

## Research findings

_To be populated by `PROCEDURE-pr-research.md`._

---

## Scope

Implement the data layer per D3 / D9 / D14 — Polars/Parquet ingest, splits, content-addressed dataset versioning, and the `data` CLI verb group.

- `src/rux_ml/data/loaders.py` — `load_parquet(path: Path) -> pl.LazyFrame` (lazy by default), `materialize(lf, engine="streaming") -> pl.DataFrame` for the small-data path
- `src/rux_ml/data/splits.py` — `train_val_test_split(df, target_col, ratios, seed)` with deterministic splits via `np.random.SeedSequence` (full per-component seeds land in PR-013)
- `src/rux_ml/data/versioning.py` — composite `data_hash`:
  - `bytes_hash = sha256(sorted(per_partition_xxhash_bytes(p) for p in parquet_files))`
  - `logical_hash = sha256(canonical_serialization_of_sorted_columns(df))`
  - `compute_data_hash(path) -> {"bytes_hash": ..., "logical_hash": ..., "row_count": ..., "schema": ...}`
  - `write_manifest(name, version_id, hashes, source_path, **meta)` → `data/manifests/<name>/<version_id>.json`
  - `cas_hardlink(source_path, hashes) -> Path` → places content into `data/cas/<bytes_hash[:2]>/<bytes_hash>/...`
- `src/rux_ml/data/data_iter.py` — XGBoost `DataIter` for `ExtMemQuantileDMatrix` reading per-partition Parquet via Polars (used by PR-006 training layer; included here so the data layer owns it)
- CLI bodies (`src/rux_ml/cli/data.py`):
  - `rux-ml data hash <path>` — print composite hash JSON
  - `rux-ml data version <name> <path>` — snapshot into CAS, write manifest, print version_id
  - `rux-ml data list [--name X]` — list versioned datasets from manifests dir
- `configs/base.toml` extended with `[data]` section: `cas_root` (default `data/cas`), `manifests_root` (default `data/manifests`), `gpu_in_memory_x_gb_max = 18`
- Tests: synthesize a tiny parquet, hash it, snapshot it, list it; assert hash stability across re-snapshot; assert hash CHANGES when row order differs but `logical_hash` does NOT (layout-invariance per D9 conflict notes)

NOT in scope: feature engineering (PR-005), training (PR-006).

## Dependencies

PR-003.

## Architecture section implemented

`docs/ARCHITECTURE.md` → "Data" component row, "Storage" (Data CAS row), "Decision Rules: XGBoost ingest path", "Reproducibility Architecture" point 5.

## Verification criteria

- [ ] `rux-ml data hash <parquet>` produces deterministic JSON with `bytes_hash`, `logical_hash`, `row_count`, `schema`
- [ ] Re-snapshot of identical data yields identical `bytes_hash`
- [ ] Two parquets with same logical content but different row order yield same `logical_hash`, different `bytes_hash`
- [ ] `data version` writes a manifest and creates a hardlink in CAS
- [ ] `data list` reads manifests
- [ ] `train_val_test_split` is deterministic given the same seed
- [ ] `DataIter` yields batches that XGBoost's `ExtMemQuantileDMatrix` can consume (smoke test only — full training in PR-006)
- [ ] xxhash is the bytes hasher; not Rust (per D13 "use mature Python bindings first")

## Research backing

Tier 1:

- D3: [Polars benchmarks](https://pola.rs/posts/benchmarks/), [XGBoost ExtMem tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html), [NVIDIA Polars+XGBoost](https://developer.nvidia.com/blog/training-xgboost-models-with-gpu-accelerated-polars-dataframes/)
- D9: [Arrow #40202 layout-fragile bytes hash](https://github.com/apache/arrow/issues/40202), [DVC internal files](https://dvc.org/doc/user-guide/project-structure/internal-files); composite recommended because no canonical recipe exists
- D14: source data outside repo, CAS in-repo gitignored

State assessment must verify Polars' per-partition reading API and XGBoost's `DataIter` callback signature are unchanged.

## Notes

- The "no canonical Parquet-directory hashing recipe" conflict from D9 means this implementation is the project's *own* convention. Document it in `docs/ARCHITECTURE.md` Storage section if any detail changes during implementation.
- Hardlinks (not copies) keep CAS storage cheap. On filesystems without hardlink support, fall back to copy with a warning.
