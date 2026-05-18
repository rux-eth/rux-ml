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

See `docs/0.1/ROADMAP.md` for the ordered PR plan of the active version. Full PR descriptions in `prs/`. Per-PR research status in `docs/0.1/RESEARCH-BACKLOG.md`. The v0 roadmap is frozen at `docs/0.0/ROADMAP.md`. <!-- rewrite-doc-refs:skip-line -->

No code yet. Phase 4 (this doc set) just landed; Phase 5 (implementation) begins with PR-001 once `PROCEDURE-pr-research.md` Phase 1 (State Assessment) completes.

## Ongoing Behavior (MANDATORY)

- **Every PR runs `PROCEDURE-pr-research.md` before implementation.** No exceptions. All v0 PRs were Tier-1 (research-backed at design time) per `docs/0.0/RESEARCH-BACKLOG.md`; v0.1 PRs are Tier-2 per `docs/0.1/RESEARCH-BACKLOG.md`. Phase 1 (State Assessment) is required to catch drift, Phases 2-4 may be light if no drift is found. <!-- rewrite-doc-refs:skip-line -->
- **Honor the Per-Phase Approval Gate** in any multi-phase procedure (`PROCEDURE-pr-research.md`, `PROCEDURE-design-planning.md`) — see `docs/CONSTRAINTS.md`. Default behavior is to halt at every phase boundary and request explicit approval before advancing.
- **Research findings travel with the PR** — appended to the PR file's `## Research findings` section. Do not discard.
- **State drifts.** Even research-backed decisions need Phase 1 state assessment before implementation. Project staleness threshold is **60 days** (per `docs/CONSTRAINTS.md`).
- **Doc updates ride with code.** When a PR changes architectural behavior, `docs/ARCHITECTURE.md` (and `docs/CONSTRAINTS.md` / `docs/CONVENTIONS.md` if relevant) update in the same commit. User-facing changes get a `[Unreleased]` entry in `/CHANGELOG.md` per `docs/VERSIONING.md` §6.
- **Run the version-cut ritual** when minor/major bumps land — `PROCEDURE-design-planning.md` from Phase 1 first, then snapshot temporal docs into a new `docs/<x.y>/` dir, then run `scripts/rewrite_doc_refs.py`. See `docs/VERSIONING.md` §2.
- **No phantom implementations.** Every PR includes a test exercising actual behavior end-to-end through the layer it touches.
- **All configurable values from config files.** Zero hardcoded parameters for behavior that might change. TOML knobs are the only acceptable source for tunable values.

## Design References

- `docs/ARCHITECTURE.md` — canonical architecture reference (data flow, components, decision rules, key abstractions, storage)
- `docs/CONSTRAINTS.md` — hard rules (no UI, single-GPU sequential, CUDA+fork forbidden, tolerance-not-hash for goldens, two-file model bundle, container digest pinning, per-trial provenance triple, reuse over reinvent, per-phase approval gate)
- `docs/CONVENTIONS.md` — soft patterns (directory naming, module dependency rules, public API discipline, version-string format, test markers, logging, configuration, Rust+PyO3 conventions)
- `docs/VERSIONING.md` — versioning policy + bump rules + changelog format + hybrid docs-versioning layout (flat, meta-rule)
- `docs/DEPLOYMENT.md` — container build + image-digest capture + smoke test + rebuild triggers (flat, SSOT)
- `/CHANGELOG.md` — user-facing changelog (Keep-a-Changelog 1.1.0)
- `docs/0.1/ROADMAP.md` — active PR index with phases and dependencies (v0 roadmap frozen at `docs/0.0/ROADMAP.md`) <!-- rewrite-doc-refs:skip-line -->
- `docs/0.1/RESEARCH-BACKLOG.md` — per-PR research status + drift watch (v0 backlog frozen at `docs/0.0/RESEARCH-BACKLOG.md`) <!-- rewrite-doc-refs:skip-line -->
- `prs/` — full PR descriptions (start new PRs from `prs/PR-TEMPLATE.md`)
- `docs/0.1/DESIGN-log.md` — active design conversation log with research trail (BEST-GUESS items acknowledged, conflicts flagged); v0 design history (D1–D17) frozen at `docs/0.0/DESIGN-log.md` <!-- rewrite-doc-refs:skip-line -->
- `PROCEDURE-design-planning.md` — how to run design sessions (with integrated research rounds)
- `PROCEDURE-pr-research.md` — mandatory research procedure before every PR implementation
- `PROCEDURE-code-audit.md` — post-design-session code audit
- `scripts/rewrite_doc_refs.py` — version-cut cross-reference migrator (run at every minor/major bump per `docs/VERSIONING.md` §5)

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
