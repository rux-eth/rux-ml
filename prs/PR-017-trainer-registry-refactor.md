# PR-017: Trainer registry refactor

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

**This PR is Tier-2** per `docs/0.1/RESEARCH-BACKLOG.md` and memory `feedback_tier_inheritance`. The architectural pattern is research-backed by the 2026-05-18 design session (`docs/0.1/DESIGN-log.md` Q1–Q6), but per-instance integration details remain open. All 5 phases of `PROCEDURE-pr-research.md` must run.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

_To be populated by `PROCEDURE-pr-research.md`. Do not begin implementation until this section exists with completed findings from all required phases._

**Open research questions** (must be resolved before implementation):

1. **A1 — `RuxMLConfig.training` shape.** Pydantic v2 discriminated-union (`family: Literal["xgboost"]` + `params: XGBoostConfig`, dispatched via `Discriminator`) vs registry-of-blocks (`training.xgboost: XGBoostConfig | None` with mutual-exclusion validation). Success criteria: pick the shape that gives type-safe validation, clear error messages on misuse, and the lowest migration cost from existing v0 single-family blocks. Cite Pydantic v2 official docs.

2. **Conformance-test design.** `typing.Protocol` with `@runtime_checkable` only checks method *names*, not full signatures. The conformance test needs to assert each registered family's factory returns an object that quacks like `Trainer` in a meaningful way. Success criteria: a test that fails loudly if a registered family fails to expose `fit` / `predict` with the expected signatures. Sklearn's `parametrize_with_checks` (`docs/0.1/DESIGN-log.md` Q4 citation) is the reference precedent.

3. **Manifest-schema migration.** Existing v0 bundles in the registry (if any) may carry an implicit `family: "xgboost"` or no family field at all. Success criteria: a documented back-compat path — either auto-infer family for legacy bundles, or require an explicit migration before loading. Decide whether legacy bundles fail-fast or auto-upgrade.

4. **A5 — `docs/VERSIONING.md §1` amendments.** Codify two updates in this PR's commit:
   - Add "new built-in `Solver` family" alongside the existing "new built-in `Trainer` or `Splitter` family" in the MINOR triggers list.
   - Add a sentence under §1 documenting that removal of a registered family is MINOR after a one-release `FutureWarning` window per `docs/0.1/DESIGN-log.md` Q6.

---

## Scope

Refactor the existing single-family Trainer integration into a registry-driven multi-family pattern, with XGBoost as the first (and currently only) registered family. No new families land in this PR — that's PR-018+.

Concretely:

- **Move XGBoost integration into a subpackage:**
  - `src/rux_ml/training/xgboost/` (new subpackage)
    - `__init__.py` — exposes `make_trainer` factory + family-specific helpers
    - `factory.py` — the XGBoost-specific factory function moved from `src/rux_ml/training/factory.py`
    - `config.py` — Pydantic schema for XGBoost-specific config block (co-located per Q5)
  - Keep `src/rux_ml/training/protocol.py` (the `Trainer` Protocol — unchanged from v0/D5)
- **Add the registry dict:**
  - `src/rux_ml/training/__init__.py` declares `TRAINER_FAMILIES: dict[str, Callable[[RuxMLConfig], Trainer]] = {"xgboost": _make_xgboost_trainer}` (B-explicit per Q4)
  - Top-level `make_trainer(cfg)` dispatches via `TRAINER_FAMILIES[cfg.training.family](cfg)`
- **Update `RuxMLConfig`:**
  - Resolves A1 — shape decided in Phase 2 research
  - The existing `cfg.training` block grows a `family` field (or equivalent per A1 resolution)
- **Restructure `pyproject.toml` extras:**
  - Add `[project.optional-dependencies]` `xgboost = ["xgboost>=2.1"]`
  - Move `xgboost` from `[project.dependencies]` if currently required → required-via-extras
  - The lazy-import + clear-error-on-missing pattern from `docs/0.1/DESIGN-log.md` Q3
- **Add the Protocol-conformance test:**
  - `tests/training/test_registry_conformance.py` — iterates `TRAINER_FAMILIES`, instantiates each with a minimal valid config, asserts `Trainer` Protocol conformance
- **Amend `docs/VERSIONING.md §1`** per A5 above (same commit).
- **Amend `docs/ARCHITECTURE.md`** "Key Abstractions" → `Trainer` Protocol (per D5) section: replace the single-family `make_trainer` description with the registry pattern. Same commit.
- **Amend `docs/CONVENTIONS.md`** — codify subpackage-per-family layout, B-explicit registry pattern, Pydantic schema co-location. Same commit.

Out of scope: new families (PR-018/019), Solver layer (PR-020), CLI changes beyond what registry dispatch requires, search-space TOML restructuring (defer per-family search-space placement until PR-018 codifies the convention).

## Dependencies

None at the PR level — operates on existing v0 code. Requires v0.0.1 tag present (`ec306c6`) which is the pre-cut baseline.

## Architecture section implemented

`docs/ARCHITECTURE.md` "Key Abstractions" → `Trainer` Protocol (per D5) — replaces the single-family factory description with the registry pattern. Adds the dispatch shape and the conformance-test guarantee.

`docs/0.1/DESIGN-log.md` Q1 (two-Protocol pattern, registry-unified), Q2 (subpackage per family), Q3 (optional extras + lazy imports), Q4 (B-explicit registry + conformance test), Q5 (config schema co-located).

## Verification criteria

Populated after research. Initial sketch (refine in Phase 2-3):

- [ ] `src/rux_ml/training/xgboost/` subpackage exists with `__init__.py`, `factory.py`, `config.py`
- [ ] `src/rux_ml/training/__init__.py` exports `TRAINER_FAMILIES` dict with `"xgboost"` entry
- [ ] `make_trainer(cfg)` dispatches via the registry; raises clear error on unknown family name
- [ ] `RuxMLConfig.training` carries the family selector per A1 resolution
- [ ] `pyproject.toml` declares `[xgboost]` optional extra; `uv sync --extra xgboost` works
- [ ] Factory imports `xgboost` lazily; running without the `[xgboost]` extra installed gives a clear `ImportError` on family instantiation (not at module-load time)
- [ ] `tests/training/test_registry_conformance.py` iterates `TRAINER_FAMILIES` and asserts each family conforms to `Trainer` Protocol
- [ ] All existing v0 tests pass (`make test`)
- [ ] `docs/VERSIONING.md §1` carries the Solver-family + family-removal-as-MINOR amendments
- [ ] `docs/ARCHITECTURE.md` describes the registry pattern (not the single-family factory)
- [ ] `docs/CONVENTIONS.md` codifies subpackage-per-family + B-explicit registry + Pydantic schema co-location
- [ ] `docs/0.1/ROADMAP.md` PR-017 row flipped `[ ]` → `[x]` in this PR's commit (per memory `feedback_roadmap_flip_in_pr`)
- [ ] `docs/0.1/RESEARCH-BACKLOG.md` PR-017 row marked `fully-researched YYYY-MM-DD` + `implementation-cleared YYYY-MM-DD`

## Research backing

Architectural pattern locked by `docs/0.1/DESIGN-log.md` Q1–Q5 (2026-05-18 design session, 6-agent parallel research round + 1 sequential round). Reputable sources cited per question. Key references:

- Q4 B-explicit registry pattern: HF transformers [`MODEL_MAPPING_NAMES @ v5.8.1`](https://github.com/huggingface/transformers/blob/v5.8.1/src/transformers/models/auto/modeling_auto.py)
- Q4 conformance test: scikit-learn [`parametrize_with_checks @ 1.5.0`](https://github.com/scikit-learn/scikit-learn/blob/1.5.0/sklearn/tests/test_common.py)
- Q3 optional extras + lazy imports: XGBoost [`compat.py @ v2.1.3`](https://github.com/dmlc/xgboost/blob/v2.1.3/python-package/xgboost/compat.py), Optuna [`_LazyImport @ v4.8.0`](https://github.com/optuna/optuna/blob/v4.8.0/optuna/_imports.py)
- Q2 subpackage-per-family: HF transformers `src/transformers/models/<family>/` 200+ families

Phase 2-3 research of `PROCEDURE-pr-research.md` must resolve the four open research questions above with reputable-source citations.

## Notes

- This is the **foundation PR** for v0.1's Phase I. PR-018, PR-019, PR-020 all depend on this PR landing first.
- Care required around the **manifest-back-compat** question (#3 above) — existing v0 bundles in any registry (filesystem or remote) must remain loadable, OR a migration path must be clearly documented. This is the most architecturally risky aspect of this PR.
- The **conformance test** is the stale-path tripwire that makes the registry-dict approach safe. If the test passes trivially (e.g., empty dict or no actual `fit/predict` assertion), the safety net is gone. Phase 2-3 research must produce a real conformance test, not a phantom.
- Per `One PR, One Thing`, this PR is just the refactor — no new families. PR-018 ships LightGBM as the first real exercise of the new pattern; that's the "second user" stress test for the architecture.
