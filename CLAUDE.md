# CLAUDE.md

<!-- STATUS: initialized -->

## What Is rux-ml

`rux-ml` is a personal ML research workbench for the full lifecycle of tabular gradient-boosted models — primarily XGBoost, with a contract that admits LightGBM, CatBoost, and other sklearn-compatible models later. It is operated via a single CLI (`rux-ml`) and a Python package (`rux_ml`); there is no web UI and no long-running server.

The workbench supports the loop **data → features → training → tuning → run logging → registry promotion**, with reproducibility (seed, config, data, code, environment) recorded per trial. Hyperparameter sweeps (Optuna) and one-off baseline trainings are first-class peers — both are recorded as Optuna trials in a shared SQLite study, with artifacts content-addressed via `optuna.artifacts`. Promoted models are bundled to a filesystem registry (`pipeline.skops` + `model.ubj` + Pydantic-validated manifest).

Single-user, single-machine: RTX 4090 (24 GB VRAM), i9-13900K (24 threads), 36 GB DDR5 RAM. **System RAM is the binding constraint.** XGBoost runs GPU-first via `device="cuda"`. HPO trials run **sequentially** with `n_jobs=1` (single GPU saturated by XGBoost-internal parallelism).

## Build & Test Commands

The project uses **uv** for Python dependency management and **maturin** for the future Rust+PyO3 components (none at v0). All commands operate on the containerized runtime by default; use the bare-metal commands when iterating fast on Python.

| Task | Command |
|---|---|
| Install / sync deps | `uv sync` |
| Run the CLI | `uv run rux-ml --help` |
| Run tests (default — fast) | `uv run pytest` (excludes `gpu`, `slow`, `golden` markers) |
| Run GPU tests | `uv run pytest -m gpu` |
| Run golden regression tests | `make test-golden` |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` |
| Type check | `uv run basedpyright src/` |
| All the above | `make test` |
| Build container | `make docker-build` |
| Run CLI in container | `docker compose run rux-ml --help` |
| Build first Rust crate (when it exists) | `uv run maturin develop --uv` |

Test markers are registered in `pyproject.toml` (`gpu`, `slow`, `golden`, `integration`); default `addopts` excludes the slow/gpu/golden ones.

## Architecture

Layered Python package under `src/rux_ml/` with subpackages per concern (`config/`, `data/`, `features/`, `training/`, `tuning/`, `runs/`, `registry/`, `cli/`, `_internal/`). Cargo workspace at root with flat `crates/` is the *deferred* layout — no `Cargo.toml` at v0; first crate materializes the workspace per D13.

Hybrid functional + OO architecture: layers expose factory functions (`make_data(cfg)`, `make_features(cfg)`, `make_trainer(cfg)`); Optuna's objective is functional; `Trainer` is sklearn-class OO inside. Subprocess-per-trial isolation via `subprocess.run` of `python -m rux_ml._internal.trial_runner` (spawn semantics — CUDA + fork is broken).

Configuration: three-tier layered TOML (`base.toml` → `problems/<n>.toml` → `studies/<n>.toml`) loaded via pydantic-settings v2 with `deep_merge=True`. Override precedence: CLI > env (`RUXML_*`) > .env > study > problem > base > defaults.

**See `docs/ARCHITECTURE.md` for the full design**, including data flow diagram, decision rules (XGBoost ingest path, categorical encoding, memory-pressure response, trial isolation, sampler/pruner choice), key abstractions (`Trainer` Protocol, `RuxMLConfig`, `SearchSpec`, model bundle + manifest), and storage layout.

## Implementation Status

See `docs/ROADMAP.md` for the ordered PR plan (PR-001 through PR-014, organized into Phases A–G). Full PR descriptions in `prs/`. Per-PR research status in `docs/RESEARCH-BACKLOG.md`.

No code yet. Phase 4 (this doc set) just landed; Phase 5 (implementation) begins with PR-001 once `PROCEDURE-pr-research.md` Phase 1 (State Assessment) completes.

## Ongoing Behavior (MANDATORY)

- **Every PR runs `PROCEDURE-pr-research.md` before implementation.** No exceptions. All v0 PRs are Tier-1 (research-backed at design time) per `docs/RESEARCH-BACKLOG.md`; Phase 1 (State Assessment) is required to catch drift, Phases 2-4 may be light if no drift is found.
- **Research findings travel with the PR** — appended to the PR file's `## Research findings` section. Do not discard.
- **State drifts.** Even research-backed decisions need Phase 1 state assessment before implementation. Project staleness threshold is **60 days** (per `docs/CONSTRAINTS.md`).
- **Doc updates ride with code.** When a PR changes architectural behavior, `docs/ARCHITECTURE.md` (and `docs/CONSTRAINTS.md` / `docs/CONVENTIONS.md` if relevant) update in the same commit.
- **No phantom implementations.** Every PR includes a test exercising actual behavior end-to-end through the layer it touches.
- **All configurable values from config files.** Zero hardcoded parameters for behavior that might change. TOML knobs are the only acceptable source for tunable values.

## Design References

- `docs/ARCHITECTURE.md` — canonical architecture reference (data flow, components, decision rules, key abstractions, storage)
- `docs/CONSTRAINTS.md` — hard rules (no UI, single-GPU sequential, CUDA+fork forbidden, tolerance-not-hash for goldens, two-file model bundle, container digest pinning, per-trial provenance triple, reuse over reinvent)
- `docs/CONVENTIONS.md` — soft patterns (directory naming, module dependency rules, public API discipline, version-string format, test markers, logging, configuration, Rust+PyO3 conventions)
- `docs/ROADMAP.md` — PR index (PR-001 through PR-014) with phases and dependencies
- `docs/RESEARCH-BACKLOG.md` — per-PR research status + drift watch
- `prs/` — full PR descriptions (start new PRs from `prs/PR-TEMPLATE.md`)
- `docs/DESIGN-log.md` — full design conversation log (D1–D17 with research trail, BEST-GUESS items acknowledged, conflicts flagged)
- `PROCEDURE-design-planning.md` — how to run design sessions (with integrated research rounds)
- `PROCEDURE-pr-research.md` — mandatory research procedure before every PR implementation
- `PROCEDURE-code-audit.md` — post-design-session code audit

## Constraints

Non-negotiables from `docs/CONSTRAINTS.md`:

- **No web UI / no server** — workbench is CLI + Python imports only
- **Single-GPU, sequential trials** (`n_jobs=1`) — XGBoost saturates the 4090
- **CUDA + `fork` is forbidden** — subprocess-per-trial uses spawn semantics
- **Tolerance-based golden tests only** — never exact hashes (XGBoost GPU `hist` is not bit-exact across hardware)
- **Zero hardcoded parameters** — all configurable values in TOML
- **Per-trial provenance triple** required in Optuna `user_attrs`: `data_hash`, per-layer + root config hashes, `git_sha`, `entropy_hex`, `image_digest`, library/CUDA/driver versions, `omp_threads`, `peak_rss_mb`. Promotion to the registry rejects trials missing any field.
- **Reuse over reinvent** — custom code (Python or Rust) only when no mature library covers the need; first custom Rust crate lands only on a profiled Python hot path > 5 % of a real workbench task
- **No phantom implementations** — every PR has a real test exercising actual behavior
- **Two-file model bundle** (`pipeline.skops` + `model.ubj` + Pydantic-validated `manifest.json`) — combined pickles forbidden
- **Container digest pinning** by `@sha256:...` (never tag); recorded per-trial in `image_digest`
