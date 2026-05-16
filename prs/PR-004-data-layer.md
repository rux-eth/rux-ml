# PR-004: Data layer

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time).

## Research findings

### State Assessment (2026-05-16)

**Current state of the codebase**:

- `dev` at commit `fe90aab` (PR-003 merged: CLI skeleton; 61 tests passing)
- `src/rux_ml/` has `config/`, `cli/`, `_internal/` (`hashing.py` with `xxh3_64_*`, `canonical_json`, `sha256_canonical`); **no `data/` subpackage yet**
- `pyproject.toml` declares `xxhash>=3.7` (added by PR-002); polars + xgboost NOT yet in runtime deps — PR-004 must add `polars>=1.40` and `xgboost>=3.1`
- `configs/base.toml` has a `[data]` section header (placeholder per PR-001 + PR-002); PR-004 adds defaults
- `src/rux_ml/cli/data.py` has no-op `hash`, `version`, `list` subcommands — PR-004 fills them
- `src/rux_ml/config/data.py` already defines `DataConfig` with `source_path`, `target_column`, `cas_root`, `manifests_root`, `gpu_in_memory_x_gb_max`, `split_ratios` — PR-004 consumes this schema

**Assumptions at PR draft time** (drafted 2026-05-15 from D3, D9, D14):

- Polars (`pl.scan_parquet` / `pl.read_parquet`) as the canonical reader, including hive-partitioned directories
- `pl.Expr.hash` for sorted-canonical-column-projection logical hashing
- `xxhash` for streaming bytes hashing (PR-002 added `xxhash>=3.7`)
- SHA-256 for the composite manifest hash combine
- XGBoost `DataIter` subclass with `next(self, input_data) -> bool` + `reset(self)` for `ExtMemQuantileDMatrix`
- `ExtMemQuantileDMatrix(..., cache_host_ratio=...)` for the GPU out-of-core path with host-RAM cache
- `pathlib.Path.hardlink_to` for the CAS
- `LazyFrame.collect(engine="streaming")` for larger-than-RAM batch materialization
- `LazyFrame.collect_schema()` for zero-materialization schema inspection

**Verification against current authoritative sources**:

| Item | Status | Source |
|---|---|---|
| Polars 1.40.x; `read_parquet` / `scan_parquet` stable | **STILL CURRENT** | [polars on PyPI](https://pypi.org/project/polars/) |
| `scan_parquet(dir, hive_partitioning=...)` partition discovery | **STILL CURRENT** | [Polars GitHub releases](https://github.com/pola-rs/polars/releases) |
| `pl.Expr.hash(seed=...)` for column hashing | **STILL CURRENT** | [polars.Expr.hash](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.hash.html) |
| XGBoost `DataIter` (`next(input_data)`, `reset`, `super().__init__(cache_prefix=...)`) | **STILL CURRENT** | [external_memory.html](https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html) |
| `ExtMemQuantileDMatrix(..., cache_host_ratio=...)` (added 3.1.0; auto-estimated when `None`) | **STILL CURRENT** | [XGBoost python_api](https://xgboost.readthedocs.io/en/stable/python/python_api.html) |
| `xxhash` 3.7.0 (released 2026-04-25); use `xxh3_64` (already exposed via `_internal/hashing.py`) | **STILL CURRENT** | [xxhash on PyPI](https://pypi.org/project/xxhash/) |
| SHA-256 vs BLAKE3 for manifest combine | **STILL CURRENT** — SHA-256 remains the universal default; combine step is kilobytes of JSON, not a hot path | [SHA-256 alternatives 2025](https://devtoolspro.org/articles/sha256-alternatives-faster-hash-functions-2025/), [Kerkour: hash 2030](https://kerkour.com/fast-secure-hash-function-sha256-sha512-sha3-blake3) |
| `pathlib.Path.hardlink_to(target)` (3.10+; arg order reversed vs `os.link`) | **STILL CURRENT** — **doc note: same-filesystem requirement** | [pathlib.Path.hardlink_to](https://docs.python.org/3/library/pathlib.html#pathlib.Path.hardlink_to) |
| `LazyFrame.collect(engine="streaming")` for larger-than-RAM | **STILL CURRENT** | [Polars streaming guide](https://docs.pola.rs/user-guide/concepts/streaming/) |
| `LazyFrame.collect_schema()` zero-materialization schema | **STILL CURRENT** | [polars.LazyFrame.collect_schema](https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.collect_schema.html) |

**Stale assumptions**: None.

**New constraints learned from prior PRs or codebase evolution**:

- `DataConfig` (PR-002) already pins the schema (`cas_root`, `manifests_root`, `split_ratios`, `gpu_in_memory_x_gb_max`) — PR-004 consumes it, doesn't define new fields.
- `configs/base.toml` has an empty `[data]` table; PR-004 may add concrete defaults if they need to override the Pydantic field defaults (currently they match — likely no edit needed beyond a comment).
- `_internal/hashing.py` (PR-002) already exposes `xxh3_64_bytes`, `xxh3_64_file`, `canonical_json`, `sha256_canonical`. PR-004 uses these — no duplication.
- PR-003 CLI's `data.py` already wires the subcommand surface with `get_options(ctx)`. PR-004 replaces the `not_implemented` bodies with real implementations using the resolved `RuxMLConfig`.

**Documentation note from state assessment** (worth capturing in PR + ARCHITECTURE.md if applicable):

- **`Path.hardlink_to` requires source and destination on the same filesystem.** The CAS dir (`data/cas/`) must live on the same mount as wherever the source Parquet is read from, or the hardlink fails with `OSError`. PR-004 should fall back to a `shutil.copy2` with a warning when `os.link` raises `OSError(errno.EXDEV)` ("Invalid cross-device link"), and document this in the CLI help text + `docs/ARCHITECTURE.md` Storage section if needed.

**Synthesis Outcome: CONFIRM** (with one cross-filesystem doc note).

### Synthesis (2026-05-16)

**Outcome:** Confirm.

**Changes to this PR from research:**
- Add **`polars>=1.40`** and **`xgboost>=3.1`** as runtime dependencies in `pyproject.toml` (planned; just locking the exact pins).
- Implement an **`EXDEV` cross-device fallback** in the CAS path (`shutil.copy2` + warning) — derived from the documentation note above.

**Changes to ARCHITECTURE.md:** Add a one-sentence note in the "Storage" section that `data/cas/` should live on the same mount as the source Parquet for hardlink-based CAS to be efficient; otherwise the workbench falls back to copies and emits a warning.

**Changes to CONSTRAINTS.md / CONVENTIONS.md / CLAUDE.md:** None.

**New PRs that must come first:** None.

**Research-backed details now locked in this PR:**
- Polars 1.40.x + `scan_parquet` (lazy) / `read_parquet` (eager) with hive-partition discovery
- `pl.Expr.hash` over a sorted canonical column projection for `logical_hash` (D9)
- `xxh3_64` via `_internal/hashing.py` for `bytes_hash` component
- SHA-256 for the composite manifest combine (kilobytes-of-JSON, not hot)
- XGBoost 3.x `DataIter` subclass; `ExtMemQuantileDMatrix(cache_host_ratio=…)` for GPU out-of-core
- `pathlib.Path.hardlink_to(target)` with EXDEV → copy fallback
- `LazyFrame.collect(engine="streaming")` for larger-than-RAM batches
- `LazyFrame.collect_schema()` for zero-materialization schema access

### Gate Check

- Premise still valid: ✓ (no drift, no architectural rework)
- No prerequisite PRs surfaced: ✓
- User approved updated spec: ✓ (2026-05-16)
- Implementation cleared: ✓ (2026-05-16)

### Implementation notes (2026-05-16)

Three small bugs surfaced during the test pass; each is documented here for future maintainers:

1. **`_logical_hash` was scanning `files[0].parent` when only one file was passed**, which silently mixed in unrelated Parquet files under the same directory. Fixed to pass the explicit file list to `pl.scan_parquet`. Tests verifying single-file vs partitioned-directory logical-hash equality caught this.
2. **`ParquetDataIter.__init__` validated `files` *before* calling `super().__init__()`**, which left `xgboost.DataIter.__del__` to crash with `AttributeError` on a missing `_temporary_data` attribute when the ValueError raised. Fixed by calling `super().__init__` first.
3. **Two CLI tests in `tests/cli/` referenced obsolete "not yet implemented" behavior for `data` verbs.** Removed those entries from `test_subcommands.py` (now covered by `test_data_subcommands.py`) and switched `test_callback_populates_global_options` to invoke `tune start` (still a no-op until PR-007).

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
