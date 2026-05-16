# Research Backlog

Index of PRs with research status. Every PR runs `PROCEDURE-pr-research.md` before implementation — this document tracks the state of each PR's research and flags drift risk per the time-decay policy.

**Tiers:**
- **Tier 1** — Design-time research exists. Phase 1 (State Assessment) required before implementation; Phases 2-4 may be light if no drift found.
- **Tier 2** — Design-time research is partial or absent. Full procedure required before the PR can be written in final form.

**Status legend:**
- `design-research ✓` — research done at design time
- `design-research ~` — partial design research (some aspects covered, others not)
- `design-research ✗` — no design-time research
- `state-assessed YYYY-MM-DD` — Phase 1 of `PROCEDURE-pr-research.md` completed
- `fully-researched YYYY-MM-DD` — all 5 phases of `PROCEDURE-pr-research.md` completed
- `implementation-cleared YYYY-MM-DD` — Phase 5 Gate Check passed

---

## Tier 1 — Implementation-Ready (pending state assessment)

All 14 v0 PRs are Tier 1 — every architectural choice was research-backed during the 2026-05-14 → 2026-05-15 design session (D1–D17). Each PR's `## Research backing` section lists the relevant decisions and primary citations.

| PR | Title | Design research | State assessed | Implementation cleared |
|----|-------|-----------------|----------------|------------------------|
| [PR-001](../prs/PR-001-project-scaffold.md) | Project scaffold | `design-research ✓` (D1, D12, D14, D15) | `state-assessed 2026-05-15` (2 drifts → amended) | `implementation-cleared 2026-05-15` |
| [PR-002](../prs/PR-002-config-skeleton.md) | Config layer skeleton | `design-research ✓` (D2, D16, D17) | `state-assessed 2026-05-15` (zero drift) | `implementation-cleared 2026-05-15` |
| [PR-003](../prs/PR-003-cli-skeleton.md) | CLI skeleton | `design-research ✓` (D11, D15) | `state-assessed 2026-05-16` (zero substantive drift) | `implementation-cleared 2026-05-16` |
| [PR-004](../prs/PR-004-data-layer.md) | Data layer | `design-research ✓` (D3, D9, D14) | — | — |
| [PR-005](../prs/PR-005-features-layer.md) | Features layer | `design-research ✓` (D4, D12) | — | — |
| [PR-006](../prs/PR-006-training-layer.md) | Training layer | `design-research ✓` (D5, D3, D7) | — | — |
| [PR-007](../prs/PR-007-optuna-basics.md) | Optuna basics | `design-research ✓` (D6, D16) | — | — |
| [PR-008](../prs/PR-008-subprocess-isolation.md) | Subprocess-per-trial isolation | `design-research ✓` (D10, D16) | — | — |
| [PR-009](../prs/PR-009-run-logging.md) | Run logging | `design-research ✓` (D7, D11) | — | — |
| [PR-010](../prs/PR-010-model-registry.md) | Model registry | `design-research ✓` (D8) | — | — |
| [PR-011](../prs/PR-011-memory-and-threading.md) | Memory & threading | `design-research ✓` (D10) | — | — |
| [PR-012](../prs/PR-012-container.md) | Container | `design-research ✓` (D1, D9, D10) | — | — |
| [PR-013](../prs/PR-013-seed-management.md) | Seed management | `design-research ✓` (D9) | — | — |
| [PR-014](../prs/PR-014-golden-tests.md) | Golden regression test infrastructure | `design-research ✓` (D12) | — | — |

## Tier 2 — Research-Pending

None at v0. New PRs created later (e.g., a PR adding the first Rust crate per D13's profile-driven trigger) start as Tier 2 unless they explicitly inherit prior research.

---

## Drift Watch

Per the time-decay policy in `PROCEDURE-pr-research.md`, any PR marked `fully-researched` or `state-assessed` more than the project's staleness threshold before implementation begins must re-run Phase 1 (State Assessment).

**Project staleness threshold:** **60 days** (recorded in `docs/CONSTRAINTS.md`; BEST-GUESS, user-acknowledged).

**Currently watching:** all 14 v0 PRs. Design-time research was completed 2026-05-14 → 2026-05-15. Each PR's state assessment must complete by **2026-07-14** to avoid re-research; PRs implemented after that date must re-run Phase 1.

### Per-PR drift-risk notes

These are the PR-specific things state assessment should verify (highlights only — each PR file's `Research backing` section has the full list):

- **PR-001** — `pyproject.toml` `[build-system]` + uv `[tool.uv]` schema may evolve; verify before committing.
- **PR-002** — `pydantic-settings v2` `TomlConfigSettingsSource(deep_merge=True)` API; `settings_customise_sources` hook signature.
- **PR-003** — Typer's `add_typer` API; shell-completion install behavior.
- **PR-004** — Polars per-partition reading API; XGBoost `DataIter` callback signature; `xxhash` PyPI maintenance.
- **PR-005** — sklearn `set_output("polars")` behavior; `category_encoders.NestedCVWrapper` import path.
- **PR-006** — XGBoost sklearn wrapper still exposes `device`, `enable_categorical`, `eval_set`, `callbacks`, `best_iteration_`.
- **PR-007** — Optuna 4.x sampler/pruner/callback APIs; `XGBoostPruningCallback` import path.
- **PR-008** — XGBoost ≥ 2.x still requires `spawn` for CUDA across subprocesses; SQLite locking under serialized-write multi-process; `optuna.load_study` from a child process behavior.
- **PR-009** — `study.ask()` + `study.tell()` API; `trials_dataframe()` includes `user_attrs` columns.
- **PR-010** — XGBoost `Booster.save_model("...ubj")` and `load_model(...)` unchanged; `skops.io.dump` / `load` API current; no new sklearn version-incompatibility warnings.
- **PR-011** — `psutil.Process.memory_info().rss` still canonical; `threadpoolctl` cross-runtime limitation status (if it has been resolved, simplify).
- **PR-012** — Look up the current digest of `nvidia/cuda:12.4.1-devel-ubuntu22.04` (NVIDIA may have rebuilt the image); verify XGBoost CI is still on CUDA 12.4 (or update with documented justification); NVIDIA Container Toolkit installation steps for any 2026 changes.
- **PR-013** — `numpy.random.SeedSequence.spawn` API unchanged; XGBoost `random_state` still flows through the sklearn wrapper; GPU determinism behavior has not regressed.
- **PR-014** — `np.testing.assert_allclose` behavior on `atol`/`rtol` unchanged; pinned XGBoost / sklearn / Polars versions in the fixture manifest still installable.

---

## How to update this document

- After Phase 4 (Docs) of a design session, add each new PR as a row with its design-research status (already done for PR-001 through PR-014).
- After running `PROCEDURE-pr-research.md` Phase 1 for a PR, update the `State assessed` column with the date.
- After completing all 5 phases, update `Implementation cleared` with the date.
- If state assessment surfaces drift that blocks implementation, move the PR back to Tier 2 and document required research.
- New PRs added to the roadmap start as Tier 2 unless they explicitly inherit prior research.
