# PR-017: Trainer registry refactor

**Landed-in:** v0.1.0

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

**This PR is Tier-2** per `docs/0.1/RESEARCH-BACKLOG.md` and memory `feedback_tier_inheritance`. The architectural pattern is research-backed by the 2026-05-18 design session (`docs/0.1/DESIGN-log.md` Q1–Q6), but per-instance integration details remain open. All 5 phases of `PROCEDURE-pr-research.md` must run.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

`PROCEDURE-pr-research.md` 5-phase Tier-2 procedure completed 2026-05-18.

### State Assessment (2026-05-18) — Phase 1

**Current state:**
- Branch `pr-017/trainer-registry-refactor` at `dev` HEAD `47b6295` (PR #21 merge). No drift since Phase 1.
- Training layer flat at `src/rux_ml/training/` (5 files): `__init__.py`, `factory.py`, `ingest.py`, `metrics.py`, `protocol.py`. No subpackage layout yet.
- `TrainingConfig.kind: Literal["xgboost", "lightgbm", "catboost", "sklearn"]` already declared in v0 (PR-006). PR-017 reframes this from "if/elif dispatch" to "discriminated union + registry dispatch" without changing the field name.
- `xgboost` is a hard `[project.dependencies]` entry (`pyproject.toml:19`). `factory.py:17` + `ingest.py:23` both import xgboost at module top.
- `ModelManifest` (`src/rux_ml/registry/manifest.py`) carries `training_cfg_hash` but no explicit `family` field — family is implicit via the stored config.
- `canonical_json` in `_internal/hashing.py:47` uses `sort_keys=True`, so field declaration order does NOT affect `training_cfg_hash`. Critical for back-compat.
- Two existing in-tree discriminator precedents: `cv.py:94` `Field(discriminator="kind")` and `tuning.py:33` `Field(discriminator="type")`.

**Stale assumptions** (none severe enough to loop back to design-planning):
- "Restructure pyproject.toml extras to `[xgboost]`" — Q-Dep research recommended dropping this. xgboost stays hard.
- "Manifest-schema migration" — less severe than assumed; manifest doesn't carry an explicit `family` field.

**New constraints surfaced:**
- xgboost-hard-vs-optional became a new must-answer (Q-Dep).
- `model_kwargs: dict[str, Any]` is currently XGBoost-specific; its multi-family shape needs research (Q-MK).

### Research Questions — Phase 2

Five must-answer questions; one nice-to-have (Q-A5 doc-only).

| Q | Bundle | Decided in |
|---|---|---|
| Q-A1 | Pydantic shape: discriminated union vs registry-of-blocks | Round A (bundled with A1b + MK) |
| Q-A1b | Discriminator field name: `kind` vs `family` | Round A |
| Q-MK | `model_kwargs` shape post-refactor | Round A |
| Q-CT | `typing.Protocol` conformance-test design | Round B |
| Q-Dep | xgboost hard-required vs `[xgboost]` extra | Round C |
| Q-A5 | `docs/VERSIONING.md §1` amendments | Doc-only, no research |

### Phase 3 — Research findings

Three parallel research agents dispatched. All findings cite primary sources at pinned commits/tags.

**Q-A1 + Q-A1b + Q-MK (bundled):** **Discriminated union with `kind`, shared `TrainingBase` base class, per-variant typed params, `model_kwargs` retained ONLY on variants whose upstream `__init__` accepts `**kwargs`.** PROVEN / CONVENTION.

- Pydantic v2 canonical idiom: [`Field(discriminator='<name>')`](https://docs.pydantic.dev/latest/concepts/unions/) over `Union[...]` with `Literal[...]` per variant.
- Production precedent: DataRobot `syftr` ([`syftr/configuration.py:374-533` @ `7174a26`](https://github.com/datarobot/syftr/blob/7174a26/syftr/configuration.py)) — closest analogue (multi-backend LLM config with shared `LLMConfig(BaseModel)` + provider discriminator).
- In-tree precedent (decisive): `src/rux_ml/config/cv.py:94` already uses `Field(discriminator="kind")`. Renaming to `family` would break every v0 `training_cfg_hash`.
- `model_kwargs` rule is forced by upstream constructor signatures:
  - XGBoost [`v2.1.3 sklearn.py:691-735`](https://github.com/dmlc/xgboost/blob/v2.1.3/python-package/xgboost/sklearn.py) — `**kwargs: Any` (supported).
  - LightGBM [`v4.5.0 sklearn.py:485-507`](https://github.com/microsoft/LightGBM/blob/v4.5.0/python-package/lightgbm/sklearn.py) — `**kwargs: Any` (supported).
  - CatBoost [`v1.2.7 core.py:5002-5123`](https://github.com/catboost/catboost/blob/v1.2.7/catboost/python-package/catboost/core.py) — explicit ~120-param signature, **no `**kwargs`**. Generic `model_kwargs` would raise `TypeError` on unknown keys. Variants for CatBoost OMIT `model_kwargs` (PR-019).

**Q-CT (conformance test):** **NO `@runtime_checkable`. Use behavioral parametrize-over-registry test pattern (sklearn `parametrize_with_checks` + Optuna `pytest_samplers.py`).** PROVEN.

- [CPython 3.12 typing docs](https://docs.python.org/3.12/library/typing.html#typing.Protocol): `@runtime_checkable` "will check only the presence of the required methods or attributes, **not their type signatures or types**."
- [PEP 544](https://peps.python.org/pep-0544/): "*There is no intent to provide sophisticated runtime instance and class checks against protocol classes.*"
- Reference precedents: sklearn [`parametrize_with_checks @ 1.5.0`](https://github.com/scikit-learn/scikit-learn/blob/1.5.0/sklearn/utils/estimator_checks.py); Optuna [`pytest_samplers.py @ v4.8.0`](https://github.com/optuna/optuna/blob/v4.8.0/optuna/testing/pytest_samplers.py).
- Test asserts: factory returns non-None; `fit` + `predict` callable; end-to-end `make_classification(50, 4)` smoke; `fit` returns self (sklearn convention); `predict` returns finite array of correct shape.

**Q-Dep (xgboost dep strategy):** **Keep xgboost HARD-required. Only siblings are optional extras (added by PR-018/019).** CONVENTION + scope reduction.

- Closest analog: [`lightning-hydra-template`](https://github.com/ashleve/lightning-hydra-template/blob/main/setup.py) — single-user research template with one primary backend (torch + lightning hard, no backend extras).
- Symmetric pattern (Optuna [`v4.8.0 pyproject.toml`](https://github.com/optuna/optuna/blob/v4.8.0/pyproject.toml), transformers `setup.py`) is for libraries serving competing user populations; rux-ml has one user with a stated primary.
- Migration cost of symmetric: 5 source files + 6 test files + container path + golden tests. Zero upside realized — nobody will `uv sync` rux-ml without XGBoost. **Scope reduction**: PR-017 does NOT touch `pyproject.toml`.

**Q6 / family removal (decided in v0.1 design session; codified by PR-017):** Removal = MINOR after one-release `FutureWarning`. Anchored on sklearn / NumPy NEP 23 / PyTorch core (4-of-5 pre-1.0 convention).

### Phase 4 — Locked decisions

| # | Decision | Source(s) |
|---|---|---|
| Q-A1 + A1b | Discriminated union with shared `TrainingBase`, `kind` discriminator, per-variant typed params | Pydantic docs + syftr + in-tree `cv.py` |
| Q-MK | `model_kwargs` retained on XGBoost variant only (LightGBM in PR-018 also gets it; CatBoost in PR-019 does NOT) | Upstream constructor signatures |
| Q-CT | Static Protocol (no `@runtime_checkable`) + behavioral parametrize-over-registry test, `make_classification(50, 4)` smoke | PEP 544 + 3.12 typing docs + sklearn + Optuna |
| Q-Dep | xgboost hard-required; **PR-017 pyproject.toml UNTOUCHED** | lightning-hydra-template + migration cost analysis |
| Q-A5 | Amend `docs/VERSIONING.md §1` — add `Solver` to MINOR triggers + family-removal-as-MINOR rule | v0.1 DESIGN-log Q1 + Q6 |

**Gate Check (Phase 5) sub-decisions resolved during implementation:**
- `ingest.py` placement: moved to `src/rux_ml/training/xgboost/ingest.py`; re-exported from `rux_ml.training` for back-compat (cli/train.py + tuning/objective.py unchanged).
- `TrainingBase` location: `src/rux_ml/training/base.py` (neutral module, defines Pydantic `BaseModel` with inlined strict `model_config` — does NOT inherit from `rux_ml.config._strict_model.StrictModel` to avoid a circular import via `rux_ml.config.__init__`).
- `kind` discriminator declared on each variant (not on `TrainingBase`) to avoid `reportIncompatibleVariableOverride` — the dispatcher takes the union `TrainingConfig` so basedpyright sees the discriminator field on every narrowed branch.
- TOMLs require explicit `kind = "xgboost"` under `[training]` (Pydantic discriminator dispatch runs before field defaults — same pattern as `[cv]`). Updated `configs/base.toml` + every test TOML fixture.

### Verification artifacts

- `make test` → **305 passed**, 0 failed, 15 deselected (gpu/slow/golden/docker). Was 301 pre-refactor; added 4 new tests (3 in `test_registry_conformance.py` parametrized + new factory dispatcher tests).
- `uv run basedpyright src/` → **0 errors, 0 warnings, 0 notes**.
- `uv run ruff check .` → **All checks passed**.
- Idempotency of XGBoost back-compat: `training_cfg_hash` is preserved for v0 trials because (a) field set unchanged, (b) defaults unchanged, (c) `canonical_json` sorts keys (line 47, `sort_keys=True`).

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
