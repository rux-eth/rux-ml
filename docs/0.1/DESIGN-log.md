# Design Log — v0.1

Design decisions and rationale from planning sessions for the v0.1 cut. Append new sessions below.

The canonical record of v0.0 design (D1–D17 + PR-015 + PR-016) lives in [`docs/0.0/DESIGN-log.md`](../0.0/DESIGN-log.md). Decisions and constraints from v0.0 remain in force unless an entry in this log explicitly supersedes them.

---

## Session: 2026-05-18 — Multi-family extensibility (Q1–Q6)

### Context

Surfaced by the user immediately after the v0.0.1 tag: *"i was thinking wed have a branch per model... i want to avoid having hundreds of thousands of lines of code for multiple models in one branch when a lot of the logic isnt tied together or shared."* Phase 1 (Idea) clarified that "branch" was a loose term — the user wanted whatever's conventional for **localized, low-overhead extensibility** when adding/swapping/removing estimator-shaped families. The scope of "family" was explicitly broadened beyond ML models to include **non-ML estimators like QP solvers and optimizers**.

The decision to plan this as a v0.1 cut followed from `docs/VERSIONING.md §1`: introducing a new `Solver` factory + new built-in `Trainer` families + new optional-extras grouping all trip MINOR triggers, and `docs/VERSIONING.md §2` mandates `PROCEDURE-design-planning.md` from Phase 1 before any v0.1 implementation PR lands.

This session ran the **Per-Phase Approval Gate** (introduced by PR-016) from the start — Claude halted at every phase boundary for explicit user approval. This is the first design session under the gate.

Run on `dev` at `ec306c6` (PR-016 merge). v0.0.1 tagged at `ec306c6` per `docs/VERSIONING.md §7`. Tests: 301 passed, 0 failed.

### Carry-forward from v0.0

User priorities from v0.0 remain in order (no changes):

1. Optimization (time + memory; RAM-bound at 36 GB)
2. Reuse over reinvent — battle-tested libraries; custom code only when no mature library covers the need
3. Convention over novelty
4. Reproducibility — every run traceable
5. Flexibility — XGBoost today, other model families later

v0.0 constraints continue to apply: no web UI / no server, single-GPU sequential trials, CUDA + `fork` forbidden, tolerance-based golden tests only, zero hardcoded parameters, per-trial provenance triple, reuse over reinvent, no phantom implementations, two-file model bundle, container digest pinning, Per-Phase Approval Gate.

### Decisions

Each decision is preceded by a focused research round (parallel where independent, sequential where dependent), with findings labeled PROVEN / CONVENTION / BEST-GUESS and citations recorded inline. The user approved every lean post-research. Below are the locked decisions.

**Q1 — Abstraction surface: two parallel `typing.Protocol`s, no shared parent (PROVEN)**

The existing `Trainer` Protocol (v0, D5) covers "learns from data" — `fit(X, y) → predict`. To admit QP solvers and other non-ML estimators, introduce a parallel `Solver` Protocol covering "solves constrained problem" — `solve(problem) → result`. The two Protocols share no parent class. What they share is the **factory pattern** — `make_trainer(cfg)` / `make_solver(cfg)` each dispatch on a string family name (`cfg.training.family = "xgboost"`, `cfg.solving.family = "osqp"`).

- **Optuna `BaseSampler` vs `BasePruner`** are two independent ABCs with no shared parent ([`optuna/samplers/_base.py @ v4.8.0`](https://github.com/optuna/optuna/blob/v4.8.0/optuna/samplers/_base.py), [`optuna/pruners/_base.py @ v4.8.0`](https://github.com/optuna/optuna/blob/v4.8.0/optuna/pruners/_base.py)). PROVEN precedent for side-by-side abstractions over a shared lifecycle.
- **cvxpy / pyomo / scipy.optimize** solver-side data flow is fundamentally `Problem(obj, constraints).solve()` — there is no `X, y`, no fitted reusable artifact. Forcing a shared `fit(X, y) → predict` Protocol would either reduce to a contract so generic it carries no meaning, or force solvers to fake a tabular shape that doesn't match how cvxpy/pyomo users actually express problems. The shared abstraction across solver libraries is **the registry mechanism itself** (string → callable), not a common runtime contract.

Stay consistent with D5 — both Protocols use `typing.Protocol` (not `abc.ABC`); this is a surface preference, not a research-backed decision.

**Q2 — Physical layout: subpackage per family (PROVEN)**

Each estimator family lives as a subpackage — `src/rux_ml/training/<family>/` (and future `src/rux_ml/solving/<family>/`) — with a thin `__init__.py` exposing the family's factory function. No plugin entry-points; no third-party-style discovery mechanism.

- **Hugging Face transformers** (`src/transformers/models/<family>/` with `configuration_<family>.py` + `modeling_<family>.py` + `tokenization_<family>.py`) — 200+ families using this exact pattern. PROVEN at scale.
- **mmengine convention**: `<scope>.models.<family>` ([`mmengine/registry/registry.py @ v0.10.7`](https://github.com/open-mmlab/mmengine/blob/v0.10.7/mmengine/registry/registry.py) docstring lines 38–69).
- **lightning-hydra-template** uses `src/models/components/` even at N=1 — convention isn't overkill at small scale.

**Q3 — Dependencies: PEP 631 optional extras + lazy imports (PROVEN)**

Each family's backend Python dependency (`xgboost`, `lightgbm`, `catboost`, `osqp`, etc.) lives in `[project.optional-dependencies]` of `pyproject.toml`. Factory imports of family backends are wrapped in `try/except ImportError`; an unwired family raises a clear runtime error on instantiation rather than crashing at workbench startup.

- **XGBoost** `python-package/compat.py @ v2.1.3` — try/except on sklearn import; `XGBModelBase = sklearn.base.BaseEstimator` when present, `object` stub otherwise. scikit-learn + pandas are PEP 631 extras, not hard deps.
- **LightGBM** `python-package/lightgbm/compat.py @ v4.5.0` — identical pattern, `_LGBM*Base` stubs.
- **Optuna** `_LazyImport` shim on `GPSampler` (scipy + torch) ([`optuna/samplers/_gp.py @ v4.8.0`](https://github.com/optuna/optuna/blob/v4.8.0/optuna/samplers/_gp.py)); scikit-learn, scipy, torch, cmaes, botorch all optional extras.
- **HF transformers** `setup.py @ v5.8.1` — extras as capability buckets (`[torch]`, `[vision]`, `[audio]`, `[sentencepiece]`).
- **cvxpy** `pyproject.toml @ v1.5.0` — 4 conic solvers required (`osqp`, `ecos`, `clarabel`, `scs`), 10+ optional extras (`mosek`, `gurobi`, `xpress`, `cplex`, `glpk`, `highs`, …).

Convergent across 5/6 surveyed projects. CatBoost is the outlier (pandas required, sklearn never imported).

**Q4 — Stale-path prevention: B-explicit registry dict + Protocol-conformance test (PROVEN)**

Each layer's registry is a plain `dict[str, Callable]` declared in the layer's `__init__.py`:

```python
# src/rux_ml/training/__init__.py
TRAINER_FAMILIES: dict[str, Callable[[RuxMLConfig], Trainer]] = {
    "xgboost": _make_xgboost_trainer,
    # entries added here as families land
}
```

A single Protocol-conformance test iterates each registry, instantiates each registered family with a minimal valid config, and asserts the runtime contract. Adding a family = add one dict line + add the subpackage + green test. Removing a family = delete the dict line + delete the subpackage + green test (any orphan reference in source / config / tests fails the test loudly).

The **conformance-test half** is universal across all surveyed projects:

- **scikit-learn**: `parametrize_with_checks` over `all_estimators()` ([`sklearn/tests/test_common.py @ 1.5.0`](https://github.com/scikit-learn/scikit-learn/blob/1.5.0/sklearn/tests/test_common.py))
- **Optuna**: `pytest_samplers.py` shared parametrized test suite reused across all built-ins ([`optuna/testing/pytest_samplers.py @ v4.8.0`](https://github.com/optuna/optuna/blob/v4.8.0/optuna/testing/pytest_samplers.py))

The **registry-dict half** has two convergent sub-flavors in the wild:

- **B-explicit** (HF transformers `MODEL_MAPPING_NAMES` — hand-maintained OrderedDict, [`src/transformers/models/auto/modeling_auto.py @ v5.8.1`](https://github.com/huggingface/transformers/blob/v5.8.1/src/transformers/models/auto/modeling_auto.py))
- **B-decorator** (mmengine `@MODELS.register_module()`; AllenNLP `@BaseClass.register("name")`; pyomo `@SolverFactory.register('glpk')`)

`rux-ml` chose **B-explicit** for code-style preference: statically analyzable, no decoration order concerns, hand-maintained registry is leaner at 5–10 families than at 200+.

Sklearn-style `pkgutil.walk_packages` discovery + base-class filter is a viable alternative ([`sklearn/utils/discovery.py @ 1.5.0`](https://github.com/scikit-learn/scikit-learn/blob/1.5.0/sklearn/utils/discovery.py)) but was rejected because `rux-ml`'s config-driven shape (`cfg.training.family = "xgboost"`) treats the string AS the registry key — Pattern B is more natural.

**Entry-points rejected for first-party extensibility.** Zero surveyed first-party projects use `[project.entry-points]` for in-tree families. pytest's own built-ins are a hardcoded tuple ([`_pytest/config/__init__.py @ 8.3.3`](https://github.com/pytest-dev/pytest/blob/8.3.3/src/_pytest/config/__init__.py) lines 253–287); `pytest11` is only for `pytest-cov`-style external packages. `rux-ml` has no third-party plugin ecosystem to support and never will under the No Web UI / No Server constraint.

**Q5 — Config layout: hybrid (PROVEN)**

Three layers:

- **Parameter schema co-located with family code** — `src/rux_ml/training/<family>/config.py` declares the family's Pydantic schema. Convention precedent: HF transformers `configuration_<family>.py` co-located with `modeling_<family>.py`.
- **Hyperparam search-space TOMLs centralized** — `configs/search_spaces/<family>.toml`. Convention precedent: lightning-hydra-template `configs/hparams_search/` is centralized.
- **Per-problem defaults user-facing** — `configs/problems/<n>.toml` keeps the existing v0 layout. Convention precedent: lightning-hydra-template `configs/experiment/*.yaml` centralized + user-facing.

No surveyed project unifies all three layers in one location. The hybrid is the convergent answer.

**Q6 — Family lifecycle: removal = MINOR after one-release `FutureWarning` window (PROVEN)**

Removing a family is a MINOR bump, not MAJOR. The deprecation window: announce in `0.y` via a `FutureWarning` emitted from the family's factory and from `load_model()`, then remove in `0.(y+1)` or later. `manifest.json` invalidation of bundles tagged with the removed family is accepted as a documented side effect of the deprecation period — recorded in the CHANGELOG, not the version number.

- **scikit-learn**: 2 MINOR releases, `FutureWarning`, removal in MINOR. Example: `RandomizedLasso` / `RandomizedLogisticRegression` deprecated in 0.19, removed in 0.21 — both MINOR releases ([scikit-learn deprecation policy](https://scikit-learn.org/stable/developers/contributing.html#deprecation), [whats_new 0.19](https://scikit-learn.org/0.19/whats_new.html), [issue #8995](https://github.com/scikit-learn/scikit-learn/issues/8995)).
- **NumPy NEP 23**: "*at least 2 releases assuming the current 6-monthly release cycle*"; removal in "*any minor, but not bugfix, release*". `DeprecationWarning` default.
- **PyTorch core**: 2 releases + 180 nightlies; removal in MINOR for stable tier ([PyTorch Python Frontend BC/FC Policy](https://github.com/pytorch/pytorch/wiki/PyTorch's-Python-Frontend-Backward-and-Forward-Compatibility-Policy)).
- **pandas** is the outlier (removal in MAJOR only, [pandas version policy](https://pandas.pydata.org/docs/development/policies.html#version-policy)), but pandas only adopted the MAJOR-only rule *at* 1.0.0; pandas pre-1.0 was looser.

4-of-5 pre-1.0 convention is "deprecate in MINOR, remove in next MINOR after a warning window." For a single-user workbench where the `FutureWarning` reaches the only user reliably, this is the right shape. Rejects the pandas-strict alternative as overkill.

`docs/VERSIONING.md §1` needs amending to record this rule — work that rides with PR-017 per the deferred-research map below.

### Deferred research (resolves per-PR, not in this session)

Per Phase 3 convergence, six surviving ambiguities are deferred to be researched by the PR that touches each one. This is intentional — the design session settled the **architectural pattern**, not the **per-family instantiation details**. Each PR runs the full `PROCEDURE-pr-research.md` 5-phase procedure under its Tier-2 designation.

| Ambiguity | Research lands in | What's unsettled |
|---|---|---|
| A1 — `RuxMLConfig.training` shape | PR-017 | Pydantic discriminated-union vs registry-of-blocks tradeoff. Type-safety + validation + error-message + migration cost. |
| A2 — `FutureWarning` placement | First family-deprecation PR (post-v0.1) | Factory-time vs config-validate-time vs `load_model`-time. Likely all three with a canonical placement. |
| A3 — Optional-extras naming convention | PR-018 | Per-family extras named after the family (`[lightgbm]`) vs after the backend (`[scikit-learn-lightgbm]`). Codifies into `CONVENTIONS.md` in PR-018's commit. |
| A4 — CLI surface for `Solver` | PR-020 | `rux-ml solve` verb shape — what subcommands (`solve start`, `solve resume`, `solve status`)? What's the analog of `rux-ml train`? |
| A5 — `VERSIONING.md §1` amendments | PR-017 | Add `Solver` to MINOR triggers; codify family-removal-as-MINOR per Q6. Doc-only write-down ride along with PR-017. |
| A6 — `rewrite_doc_refs.py` versioned→versioned mapping | PR-021 | Active forward-pointing refs `docs/0.0/X.md → docs/0.1/X.md` vs historical citations (e.g. "D5 in `docs/0.0/DESIGN-log.md`") that should stay pinned. Opt-out list vs smarter regex. |

### v0.1 implementation plan

Five PRs, all Tier-2 (per memory `feedback_tier_inheritance` + the per-family research surface). Details in `docs/0.1/ROADMAP.md`; per-PR scope in `prs/PR-NNN-*.md`.

| PR | Title | Depends on |
|---|---|---|
| PR-017 | Trainer registry refactor | — |
| PR-018 | LightGBM Trainer family | PR-017 |
| PR-019 | CatBoost Trainer family | PR-017 |
| PR-020 | Solver Protocol + first Solver family | PR-017 |
| PR-021 | v0.1.0 version cut + rewrite script versioned→versioned | PR-017, PR-018, PR-019, PR-020 |

PRs 018 + 019 are parallelizable post-017. PR-020 can be drafted in parallel but blocks on its own Tier-2 research before implementation.

### Process notes

- This session ran the Per-Phase Approval Gate from Phase 1 onward. Every phase boundary halted for explicit user approval — total of ~10 approval gates from Phase 1 idea-restate through Phase 4 entry.
- Research was anchored on six parallel agents (Round A + B) plus one sequential agent (Round C). All six found primary-source citations matching the procedure's reputable-sources criteria.
- Per memory `feedback_research_discipline`, each pre-research lean was explicitly labeled INTUITION and tested against the research; user approved all leans after the research backed them.
- Per memory `feedback_tier_inheritance`, all five v0.1 PRs are Tier-2 (not Tier-1) — the architectural pattern is research-backed by this session, but per-family integration details are not.
