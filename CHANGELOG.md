# Changelog

All notable user-facing changes to `rux-ml`. Format: [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning policy: see [`docs/VERSIONING.md`](docs/VERSIONING.md).

## [Unreleased]

### Added
- LightGBM Trainer family. Install via `uv sync --extra lightgbm`. Selected per problem via `[training]` `kind = "lightgbm"`. CPU-only in v0.1 (per Q-GPU research: LightGBM-GPU is 8-28x slower than XGBoost-GPU on workbench-scale data; install path via uv is non-trivial). Categorical features integrate cleanly with the existing `_ColumnRouter` (no pipeline changes); reference HPO search-space template at `configs/search_spaces/lightgbm.toml` (PR-018).

### Changed
- Adopt vibe-rails hybrid docs-versioning layout: temporal docs (`DESIGN-log`, `RESEARCH-BACKLOG`, `ROADMAP`) move under `docs/0.0/`; flat SSOT files (`ARCHITECTURE`, `CONSTRAINTS`, `CONVENTIONS`) stay at `docs/`. Add `docs/VERSIONING.md`, `docs/DEPLOYMENT.md`, `/CHANGELOG.md`. Add `Landed-in:` header to every numbered PR file. Add `scripts/rewrite_doc_refs.py` migrator + `make rewrite-doc-refs` targets. Append Per-Phase Approval Gate (NON-NEGOTIABLE) to `docs/CONSTRAINTS.md` — Claude halts at every phase boundary of every multi-phase procedure (PR-016).
- Trainer registry refactor: `src/rux_ml/training/` is now multi-family. XGBoost integration moved into `src/rux_ml/training/xgboost/` subpackage. `TrainingConfig` is a Pydantic v2 discriminated union over per-family variants with `kind` discriminator. `TRAINER_FAMILIES` registry dict (B-explicit, HF transformers `MODEL_MAPPING_NAMES` pattern). New parametrized conformance test `tests/training/test_registry_conformance.py` is the stale-path tripwire (PR-017).
- Sentinel `PASSTHROUGH_TO_XGB_CATEGORICAL` renamed to `PASSTHROUGH_NATIVE_CATEGORICAL` — same sentinel now serves XGBoost AND LightGBM AND future CatBoost (PR-018 Q-Cat).

## [0.0.1] - 2026-05-16

### Added
- Initial v0 workbench: end-to-end **data → features → training → tuning → run logging → registry promotion** lifecycle. Single CLI (`rux-ml`) over the Python package (`rux_ml`). XGBoost-first via GPU (`device="cuda"`); three-tier TOML config (`base.toml` → `problems/<n>.toml` → `studies/<n>.toml`) with `CLI > env > .env > study > problem > base` override precedence. Optuna sequential trials with subprocess-per-trial spawn isolation, SQLite study, K-fold CV-mean objective, WilcoxonPruner, TPE sampler. Five `Splitter` strategies (`KFold` / `StratifiedKFold` / `TimeSeriesSplit` / `GroupKFold` / `CombinatorialPurgedCV`). Two-file model bundle (`pipeline.skops` + `model.ubj` + Pydantic-validated `manifest.json`); atomic `champion.json` rewrite. Per-trial provenance triple in `user_attrs` (`data_hash`, per-layer/root config hashes, `git_sha`, `entropy_hex`, `image_digest`, library/CUDA/driver versions, `omp_threads`, `peak_rss_mb`). Memory watchdog with `MemoryPressureError` → `optuna.TrialPruned`. Tolerance-based golden regression tests (CPU bit-exact contract). Container image with NVIDIA CUDA 12.4.1 base, GPU passthrough via Compose, `mem_limit: 32g`, image-digest pinning (PR-001 through PR-015).

[Unreleased]: https://github.com/rux-eth/rux-ml/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/rux-eth/rux-ml/releases/tag/v0.0.1
