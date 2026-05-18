# Roadmap

Ordered list of PRs. Each PR is a single, reviewable change. Dependencies flow downward — no PR should be started before its predecessors are merged.

Full PR descriptions live in `prs/`. This file is the index.

**Status legend:** `[ ]` pending | `[x]` merged | `[~]` in progress

---

## Phase A: Foundation

| PR | Description | Status | Depends on |
|----|-------------|--------|------------|
| [PR-001](../../prs/PR-001-project-scaffold.md) | Project scaffold — `pyproject.toml` + uv + ruff + basedpyright + Makefile + `.gitignore` + `src/rux_ml/__init__.py` (`__version__`) + `crates/.gitkeep` | `[x]` | — |
| [PR-002](../../prs/PR-002-config-skeleton.md) | Config layer skeleton — Pydantic-settings + TOML loader + per-layer config models + `base.toml` + composition (CLI > env > study > problem > base) | `[x]` | PR-001 |
| [PR-003](../../prs/PR-003-cli-skeleton.md) | CLI skeleton — Typer entry point + nested verb groups (`data`, `train`, `tune`, `runs`, `registry`) wired as no-op subcommands; `rux-ml --help` and `--version` work | `[x]` | PR-002 |

## Phase B: Data & features

| PR | Description | Status | Depends on |
|----|-------------|--------|------------|
| [PR-004](../../prs/PR-004-data-layer.md) | Data layer — Polars/Parquet loaders + train/val/test splits with seed + composite `data_hash` (xxhash bytes + logical) + manifest read/write + CAS hardlinks; CLI: `rux-ml data {hash, version, list}` | `[x]` | PR-003 |
| [PR-005](../../prs/PR-005-features-layer.md) | Features layer — sklearn `Pipeline` + custom `_ColumnRouter` + Polars `FunctionTransformer` wrapper + `make_features(cfg)` factory + categorical encoding decision rule | `[x]` | PR-004 |

## Phase C: Training

| PR | Description | Status | Depends on |
|----|-------------|--------|------------|
| [PR-006](../../prs/PR-006-training-layer.md) | Training layer — `Trainer` `typing.Protocol` + `make_trainer(cfg)` factory + sklearn estimator wrapper for `XGBClassifier`/`XGBRegressor` + metric registry + ingest-path decision rule (QuantileDMatrix vs ExtMemQuantileDMatrix); CLI: `rux-ml train` for a single baseline run | `[x]` | PR-005 |

## Phase C.5: CV strategy (Tier-2 — blocks all HPO work)

| PR | Description | Status | Depends on |
|----|-------------|--------|------------|
| [PR-015](../../prs/PR-015-cv-strategy.md) | **Tier-2** CV strategy — `Splitter` `typing.Protocol` + research-backed library choice across `KFold` / `StratifiedKFold` / `TimeSeriesSplit` / `GroupKFold` / walk-forward / CPCV; rewrites `src/rux_ml/data/splits.py` to consume a Splitter; per-strategy leakage tests + parallelism notes; updates `docs/ARCHITECTURE.md` + `docs/CONVENTIONS.md` | `[x]` | PR-004, PR-005 |

## Phase D: Tuning

| PR | Description | Status | Depends on |
|----|-------------|--------|------------|
| [PR-007](../../prs/PR-007-optuna-basics.md) | Optuna basics — study create/load + functional `objective(trial, base_cfg)` + `SearchSpec` walker (TOML-declared search space) + sequential trials in-process + SQLite `RDBStorage` + K-fold CV-mean objective + WilcoxonPruner default; CLI: `rux-ml tune {start, resume, status, retry-trial}`. **Consumes the Splitter from PR-015** | `[x]` | PR-006, PR-015 |
| [PR-008](../../prs/PR-008-subprocess-isolation.md) | Subprocess-per-trial isolation — `python -m rux_ml._internal.trial_runner` entry point + `subprocess.run` dispatcher with spawn semantics + parent/child coordination via shared SQLite + override JSON passing; CLI: `tuning.trial_isolation` config knob | `[x]` | PR-007 |

## Phase E: Run logging & registry

| PR | Description | Status | Depends on |
|----|-------------|--------|------------|
| [PR-009](../../prs/PR-009-run-logging.md) | Run logging schema — canonical `user_attrs` (provenance triple, environment, peak_rss) + `ask`/`tell` wrapper for one-off runs + query helpers; CLI: `rux-ml runs {list, show, compare}` | `[x]` | PR-008 |
| [PR-010](../../prs/PR-010-model-registry.md) | Model registry — bundle write/read (`pipeline.skops` + `model.ubj`) + Pydantic-validated `manifest.json` + atomic `champion.json` rewrite + thin `load_model(problem, version="champion")` loader; CLI: `rux-ml registry {promote, list, rollback}` | `[x]` | PR-009 |

## Phase F: Memory & ops

| PR | Description | Status | Depends on |
|----|-------------|--------|------------|
| [PR-011](../../prs/PR-011-memory-and-threading.md) | Memory & threading — `psutil` watchdog + `MemoryPressureError` → `optuna.TrialPruned` + thread-pinning env vars (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `POLARS_MAX_THREADS`); per-trial `peak_rss_mb` recorded in `user_attrs` | `[x]` | PR-008 |
| [PR-012](../../prs/PR-012-container.md) | Container — `Dockerfile` from `nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:<digest>` + uv + maturin install + non-root user + `docker-compose.yml` with GPU reservation block + `mem_limit: 32g` + volume mounts | `[x]` | PR-001 |

## Phase G: Reproducibility hardening

| PR | Description | Status | Depends on |
|----|-------------|--------|------------|
| [PR-013](../../prs/PR-013-seed-management.md) | Seed management — `SeedSequence(entropy)` per run + `.spawn()` for `{split, cv, sampler, xgb}` seeds + entropy persistence in `user_attrs` + version logging (`xgboost_version`, `cuda_runtime_version`, `image_digest`) | `[x]` | PR-009, PR-015 |
| [PR-014](../../prs/PR-014-golden-tests.md) | Golden regression test infrastructure — tolerance-based fixtures + `@pytest.mark.golden` marker + a single end-to-end golden test on a tiny fixed-seed dataset (XGBoost) using `np.testing.assert_allclose` + AUC tolerance bands | `[x]` | PR-010, PR-013 |

## Phase H: Post-v0 template migration

| PR | Description | Status | Depends on |
|----|-------------|--------|------------|
| [PR-016](../../prs/PR-016-template-migration.md) | Vibe-rails hybrid docs-versioning migration — move `DESIGN-log` / `RESEARCH-BACKLOG` / `ROADMAP` into `docs/0.0/`; add `docs/VERSIONING.md` + `/CHANGELOG.md` + `docs/DEPLOYMENT.md`; append Per-Phase Approval Gate (NON-NEGOTIABLE) to `docs/CONSTRAINTS.md`; add `Landed-in:` header to every numbered PR file + `prs/PR-TEMPLATE.md`; ship `scripts/rewrite_doc_refs.py` (Python port of the canonical TS migrator); tag `v0.0.1` at merge | `[x]` | — |

---

## Notes

- Each PR must satisfy the verification criteria in `docs/CONSTRAINTS.md` and pass `PROCEDURE-pr-research.md` before implementation begins
- No PR proceeds without user review and approval
- Full PR descriptions in `prs/` directory
- Phase F (`PR-011`, `PR-012`) is parallelizable with the late stages of Phase D/E once PR-008 lands — note the `Depends on` column carefully
- **PR-015 (CV strategy) is Tier-2** — needs full `PROCEDURE-pr-research.md` Phase 3 web research before implementation. Blocks PR-007 because the Optuna `objective` consumes the Splitter. Also blocks PR-013 because the per-component seed `spawn` set includes a `cv_seed` whose meaning depends on which Splitter strategies exist.
- Custom Rust crates do **not** appear in this roadmap by design (per D13). The first Rust crate is added when a profile shows a Python hot path consuming >5 % of a real workbench task — a future PR-XX created at that point.
