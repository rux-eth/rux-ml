# PR-062: The CPU thread budget reaches XGBoost (`memory.omp_threads` → `n_jobs` / `nthread`)

<!-- Written 2026-10-16 from the program lead's draft (program `docs/0.5/evidence/ruxml-track/PR-062-threads-draft-20261015.md`): Phase 1 (State Assessment) by one read-only agent, Phases 2 – 4 light by the lead; the build go-list approved by the operator 2026-10-16 ≈ 08:49 CDT ("all leans approved, go on all"). -->
<!-- Paths are relative to this repo at `dev` `814cc15` unless marked "program" (the rux-capital program repo). Claim labels: proven / convention / best-guess. -->

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

### State Assessment (2026-10-15)

**Current state**:

- **(proven) Repo and number.** `dev` is at `814cc15`. The last PR file is `prs/PR-061-c6-pin-committed-bytes.md`, and 056 is taken by the prior-art branch `pr-056/promote-pin-threads` (on origin, `87d84f1`), so this PR is **PR-062**.
- **(proven) The thread decision belongs to the XGBoost factory, and today the factory passes no `n_jobs`.** `_xgb_kwargs` builds `base` with device, tree_method, enable_categorical, learning_rate, max_depth, n_estimators, subsample, colsample_bytree, early_stopping_rounds and eval_metric. `n_jobs` appears only when `cfg.model_kwargs` carries it, because `base.update(cfg.model_kwargs)` runs last. The factory receives `XGBoostTraining` and never `cfg.memory`. *Evidence:* `src/rux_ml/training/xgboost/factory.py:47-62` (`:61` the update), `:65` `make_xgboost_trainer(cfg: XGBoostTraining, *, seed)`; `src/rux_ml/training/factory.py:30,59` `make_trainer(cfg: TrainingConfig, *, seed)`.
- **(proven) Every fit path runs at the OpenMP default thread count:**

| # | Fit path | How the trainer is built | Thread count today | Evidence |
|---|---|---|---|---|
| 1 | `rux-ml train`, sklearn wrapper (the default) | `make_trainer(cfg.training, seed=bag.xgb_seed)`, no thread argument | booster `nthread` 0 = the OpenMP default | `cli/train.py:481`; read back at `:498` via `booster_threads_device` (`training/xgboost/batches.py:343-347`) |
| 2 | `rux-ml train`, `use_native=True` | `XGBoostNativeAdapter(memory_cfg=cfg.memory)`; `memory_cfg` is used only for `cache_host_ratio` | `_booster_params` sets no `nthread` (model_kwargs last-wins at `:130`) | `cli/train.py:444-453`; `training/xgboost/native_adapter.py:118-131`, `:168-169` |
| 3 | the `[m9]` per-day batch fit (in `train` and in the promote re-fit) | `fit_batches` calls `make_xgboost_trainer(training, seed=seed)` directly, not through `make_trainer`; params from `get_xgb_params()` with None dropped | `nthread=model.n_jobs` (None) to `QuantileDMatrix` and `BatchFit` | `cli/train.py:404,654`; `batches.py:228-233, 245-251, 266`; `registry/promote.py:224-225` |
| 4 | `rux-ml tune` | the objective calls `make_trainer(cfg.training, seed=bag.xgb_seed)` per fold, no threads | in-process mode pins after xgboost is loaded (`tuning/objective.py:40` imports it at module load); the default mode is `subprocess` | `tuning/objective.py:40, :248`; `cli/tune.py:96-101`; `config/tuning.py:62` |
| 5 | in-process `rux-ml train`: xgboost loads before the pin | `rux_ml.cli` imports every verb module at load; the config holding `memory.omp_threads` is known only after Typer parses `--config/--problem/--set` inside `run_command` | **measured on the desktop:** `OMP_NUM_THREADS` = 24 but `omp_get_max_threads()` = 32 after the pin; a cpu M6 fit stalled at 16.81 s | `cli/__init__.py:26`; `pyproject.toml:36`; `cli/train.py:28`, `:706-717` (cfg `:708-713`, pin `:717`); program `prs/PR-082-probe-runner-statistic-null-holm-freeze.md:1205-1210` (R3-a) |
| 6 | `registry promote` | in-memory re-fit `make_trainer(cfg.training, seed=bag.xgb_seed)`; the `[m9]` re-fit through `fit_batches`; promote never calls `pin_threads` | the re-fit records the booster's `nthread` into the manifest's `refit` block | `registry/promote.py:246`, `:219-220`, `:180-196`; `pin_threads` only in `cli/solve.py:170`, `cli/train.py:717`, `cli/tune.py:99`, `_internal/trial_runner.py:99`; `prs/PR-055-m9-oos-export-batch-promote.md:95` |

  Path 3 matters for the fix: a factory default reaches it automatically, but only if the default does not depend on an argument that only `make_trainer` passes.
- **(proven, import chain / best-guess, runtime value) New, wider than the brief: the subprocess trial child also loads xgboost before it pins.** `trial_runner` imports `rux_ml.config` before `pin_threads`. `rux_ml.config` imports `config/training.py`, which imports `rux_ml.training.base`, which runs `rux_ml/training/__init__.py`; that imports `make_trainer` and the xgboost package, whose factory runs `from xgboost import XGBClassifier, XGBRegressor`. The comment *"Config import is light (pydantic + tomllib only)"* is false. `omp_get_max_threads()` inside the child was **not measured**; under R3-a's mechanism it is expected to be 32. *Evidence:* `_internal/trial_runner.py:86-87` before `:97-99`; `config/__init__.py:23`; `config/training.py:27`; `training/__init__.py:35,47`; `training/xgboost/__init__.py:15`; `training/xgboost/factory.py:17`.
- **(proven) `runs learning-curve` runs no fit.** It reads `fold_meta.json` from COMPLETE trials, whose fits came from `train`, so it is covered by whatever `train` does. *Evidence:* `cli/runs.py:142-175`, `:182-223`.
- **(proven) The CatBoost precedent reads the env var, not `cfg.memory`.** Its factory reads `OMP_NUM_THREADS` from `os.environ` when the estimator is built and falls back to -1 (all cores) when it is unset. That works for CatBoost because it is TBB and the env var is read after the pin. Its docstrings say it *"passes cfg.memory.omp_threads explicitly"* — doc drift from the code. *Evidence:* `training/catboost/factory.py:21-24` (docstring), `:197-216` (`_resolve_thread_count`), `:236-237`; `tests/training/test_catboost_smoke.py:192-205`.
- **(proven) The PR-011 thread tests are a blind instrument.** They assert `os.environ` values only, never `omp_get_max_threads()` and never a booster's `nthread`, which is why the defect was never caught. *Evidence:* `tests/_internal/test_env.py:15-44`; `tests/_internal/test_trial_runner.py:52-70`.
- **(proven, import order / best-guess, runtime no-op) The CPU bit-exact tests are carried by `n_jobs=1`, not by the env.** The golden and determinism tests set `OMP_NUM_THREADS=1` in an autouse fixture after their module already imported `rux_ml.training` (and with it xgboost), so the env pin is expected to be a no-op. The effective pin is `model_kwargs={"n_jobs": 1}`, which wins in the factory, so a factory default leaves these tests untouched. *Evidence:* `tests/integration/test_determinism_cpu.py:22` (import) vs `:35` (setenv), `:70` (`n_jobs` 1); `tests/golden/conftest.py:90-99`; `tests/golden/test_xgb_baseline.py:170`; `factory.py:61`.
- **(proven) Tests that pin a non-default count do it through `model_kwargs`, and no test asserts a default fit's `nthread` == 0.** `--set training.model_kwargs.n_jobs=3` gives `booster_nthread` 3 and `fold_meta` booster `{nthread 3, device cpu}`. The promote-manifest test compares the re-fit's `nthread` to the trial's, both from the same code, so both sides move together. The manifest-field tests construct `nthread=0` literally. *Evidence:* `tests/cli/test_m9_batch_train_cli.py:69-73`; `tests/cli/test_m9_promote_manifest_refit.py:70-71`; `tests/registry/test_manifest_refit_fields.py:59,110,115`.
- **(proven) No rux-ml config sets `n_jobs` or `nthread`,** so a factory default applies to every configured problem. `memory.omp_threads` defaults to 24. *Evidence:* `configs/` grep → only a comment at `configs/base.toml:70`; `config/memory.py:12`.
- **(proven) The design always meant XGBoost `nthread=24` as an explicit allocation.** Option A1 below implements the documented D10 row; it adds no new design. *Evidence:* `docs/ARCHITECTURE.md:621`; `docs/0.0/DESIGN-log.md:102`; contrast `docs/CONVENTIONS.md:456`.
- **(proven) Where the thread count is recorded:** `TrialAttrs.booster_nthread` and `fold_meta` `booster.nthread` (0 today = "OpenMP default"), the manifest's `refit.nthread`, and `TrialAttrs.omp_threads`. `omp_threads` holds the intended budget (24), not what ran (32). *Evidence:* `runs/attrs.py:90-95, :168`; `_internal/env.py:146,162`; `registry/promote.py:189-196`.

**Assumptions at PR draft time** (no PR file existed; these are the guarantees the repo documents and the brief carried):

- PR-008 / PR-011: `pin_threads` sets the env before the heavy imports on every verb, so libgomp sizes its pool from `OMP_NUM_THREADS` (`_internal/env.py:5-25`; `README.md:629`).
- `docs/CONVENTIONS.md:452-460`: XGBoost honors `OMP_NUM_THREADS` through the env var, "no factory action" (LightGBM has the same row).
- `_internal/trial_runner.py:86`: the config import is light and safe before env pinning.
- `docs/CONVENTIONS.md:359`: the CPU bit-exact contract is `OMP_NUM_THREADS=1`.
- The brief (program PR-082 R3-a) located the defect in the in-process paths; the default subprocess trial child was not named.

**Stale assumptions**:

- **(proven)** "Pin before heavy imports" holds on **no** path: not `train`, not in-process `tune`, not the subprocess child (the import chain above; the comment at `trial_runner.py:86` is false).
- **(proven)** The CONVENTIONS XGBoost row is stale: an env pin after import cannot reach libgomp (R3-a measured 24 set vs 32 effective). `README.md:629` states the same broken guarantee.
- **(proven / best-guess)** `CONVENTIONS.md:359` probably mis-attributes the bit-exact contract: in-test the env pin comes after the import, and `n_jobs=1` in `model_kwargs` carries it (import order proven, runtime best-guess).
- **(best-guess)** By the same mechanism `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` are probably no-ops too (xgboost imports numpy, so BLAS loads before the pin). `POLARS_MAX_THREADS` may still work if Polars builds its pool lazily. Not measured; out of this PR's XGBoost scope but the same root cause. Program PR-027 R4 (`prs/PR-027-*.md:799`) saw promote's pre-build memory fall 7.1 → 5.6 GiB with `pin_threads`, consistent with a lazily sized pool reacting (unattributed).
- **(proven)** CatBoost's docstrings (`catboost/factory.py:21-24`) describe an explicit `cfg.memory.omp_threads` transport the code (`:197-216`) does not have.

**New constraints**:

- **(proven) `root_cfg_hash` and `training_cfg_hash` must not move** — PR-082 freezes on the configs' hashes. Both hash the config dumps with nothing elided in the memory or training layers, so any config-side pin (`memory.omp_threads` or `training.model_kwargs.n_jobs`) moves them; a code-only change does not. *Evidence:* `config/root.py:60-71` (`_HASH_ELIDED_FIELDS`: training `set()`, memory `set()`); program PR-082:1211-1216 (R3-b measured `9e74ad95…` vs `894f…`).
- **(convention) `model_kwargs` must keep winning on key collisions.** A factory `n_jobs` default goes into `base` before `base.update(cfg.model_kwargs)`, and the native `params` before `params.update(...)`, so the `n_jobs=1` golden / determinism pins and the `n_jobs=3` A11 test are unchanged. *Evidence:* `factory.py:36-38, 61`; `native_adapter.py:115-116, 130`; `training/xgboost/config.py:46-48`.
- **(proven) These must not move:** the c6 pin (committed parquet bytes and the data-hash digests; no fit is involved), golden 15 / 15 (fitted at `n_jobs=1` through `model_kwargs`), and the OOS export bytes (the export carries no thread field, so its bytes follow the predictions). *Evidence:* `tests/cli/test_data_hash_per_file_c6.py:1-10, 27-72`; `tests/golden/test_xgb_baseline.py:170`; `src/rux_ml/runs` grep for `nthread` → `attrs.py` only.
- **(convention) No hardcoded thread value.** The source is `cfg.memory.omp_threads`, never a literal 24 or 8. The operator's "8" ruling belongs to PR-082's harness, not to rux-ml's default (`~/.claude/CLAUDE.md` §Code Quality; rux-ml CONSTRAINTS; program PR-082:1481-1486).
- **(convention) Process.** Phase 1 before any code; a failing test first, then the fix; the per-phase gate; docs updated in the same commit; small PRs — an import-order fix is its own PR (rux-ml `CLAUDE.md:52-53`; program memory "Smaller PRs").
- **(proven) Whichever transport XGBoost uses, CONVENTIONS must name it** (`docs/CONVENTIONS.md:450-460`; the CatBoost precedent `catboost/factory.py:197-237`).
- **(proven) This fix does not block PR-082.** The harness pins `nthread` as an xgboost param in its own probe config, and the real cells run on cuda (program PR-082:1481-1486).

**Prior-art audit**:

- **(proven)** Branch `pr-056/promote-pin-threads` `87d84f1` (2026-09-28: +4 lines in `cli/registry.py`, +47 in a test) is unmerged and still on origin. It calls `pin_threads` in promote — an env pin after import, so it cannot reach libgomp. PR-027 R4 found the model file byte-identical pinned vs unpinned, which is consistent with the pin being ineffective; it does not prove thread-insensitivity. Not adopted, per program PR-027:799 (a memory question). **Do not resurrect it.**
- **(proven)** The PR-011 tests assert the env, not the effect — the new tests must read the booster (`booster_threads_device`) or `omp_get_max_threads()`.
- **(proven)** PR-055 put the re-fit's `nthread` into the manifest (`prs/PR-055-m9-oos-export-batch-promote.md:95`); that field moves under A1.

**Fix options** (hash and output consequences spelled out):

| Option | What changes | `root_cfg_hash` / `training_cfg_hash` | Recorded fields / outputs | Tests |
|---|---|---|---|---|
| **A1 — pass the budget explicitly** (Round 3's lean) | `make_xgboost_trainer`, and `make_trainer` passing it through, gain a thread argument sourced from `cfg.memory.omp_threads` at four sites: `cli/train.py:481`, `tuning/objective.py:248`, `registry/promote.py:246`, `training/xgboost/batches.py:228` (`fit_batches` has `cfg`). It goes into `base["n_jobs"]` before the `model_kwargs` update. The native adapter sets `params["nthread"] = memory_cfg.omp_threads` before its `model_kwargs` update (it already holds `memory_cfg`). CatBoost / LightGBM factories accept and ignore the argument, or CatBoost switches from the env read to it. | **No move** (no config field or value changes; proven from `root.py:60-71`; R3-b: a code fix that keeps 24 moves the commit, not the hashes — best-guess per PR-082:1215). The rux-ml `git_sha` moves. | `fold_meta` `booster.nthread`, `TrialAttrs.booster_nthread`, manifest `refit.nthread` go **0 → 24** (or whatever `memory.omp_threads` is) — proven by code path. Predictions expected bit-equal: proven at nthread 8–32 for cpu M6 fits (Q16); best-guess for the `[m9]` regression batch path and for cuda. `model.ubj` bytes expected unchanged (nthread is a config parameter, `save_config`, not model state — best-guess, unverified). OOS export bytes follow the predictions (best-guess). | New, failing first: (1) default `make_xgboost_trainer` with memory `omp_threads=7` → `get_params()['n_jobs']==7` and, after a tiny fit, `booster_threads_device()['nthread']==7`; (2) `model_kwargs` `n_jobs=1` still wins; (3) the native adapter's params carry `nthread` from memory; (4) a default `train` CLI fit records `booster_nthread == cfg.memory.omp_threads`. Unchanged: golden, `test_determinism_cpu`, the A11 `n_jobs=3` test, the c6 pin. Changed by design: `test_m9_promote_manifest_refit` (both sides move). |
| **A2 — CatBoost's env read, no signature change** | The XGBoost factory and native adapter read `OMP_NUM_THREADS` from `os.environ` at build time and set `n_jobs` / `nthread` from it before the `model_kwargs` update; unset → set nothing (today's behaviour). | **No move** (proven: no config change). | As A1 wherever the env var is set (train, in-process tune, the subprocess child all pin before they fit). **But promote never pins**, so promote re-fits still run at 32 threads and record `nthread` 0 against the trial's 24 unless promote also pins. Predictions expected bit-equal (best-guess beyond Q16). | As A1 with monkeypatched `OMP_NUM_THREADS`, plus "env unset leaves `n_jobs` unset". The golden autouse setenv OMP=1 would then reach the factory too (`n_jobs=1` by both routes; harmless). |
| **B — pin before xgboost loads** | Make every module-level xgboost import lazy (`training/__init__.py:35,47` → `xgboost/__init__.py:15-21` → `factory.py:17`, `batches.py:40`, `ingest.py:30`, `native_adapter.py:42`, `tuning/objective.py:40`, `data/data_iter.py:13`, `registry/bundle.py:23`, `scorer.py:48`, `registry/__init__.py:31`); reorder `trial_runner` so nothing heavy loads before `:99`; give the CLI a pre-parse step that reads `--config/--problem/--set` and pins before `rux_ml.cli` imports the verbs (or re-execs). | **No move** (proven: no config change). | Recorded `booster.nthread` **stays 0**; the OpenMP pool becomes 24 through the env var. Predictions expected bit-equal (best-guess). Also fixes BLAS / MKL if they are broken too (best-guess). Does not protect callers that import xgboost themselves first (an embedding harness, pytest modules). | A subprocess-level test that `omp_get_max_threads() == memory.omp_threads` after the pin with xgboost not in `sys.modules` before it (via ctypes on xgboost's vendored libgomp, or a fit's timing — feasibility best-guess). Wide import churn risks cycles (`training/factory.py:47` already breaks one). |
| **A1 + a follow-up PR for B** (or a narrow B-lite: only `trial_runner`'s config import chain + the false comment) | A1 is PR-062; the import-order and BLAS findings become a stub, per the smaller-PRs ruling. | **No move** for either part (proven: code only). | As A1 for PR-062; the follow-up changes no recorded field. | PR-062: the A1 tests. Follow-up: the `omp_get_max_threads` subprocess test plus the BLAS check. |
| **E — config-only pin (REJECTED)** | Set `training.model_kwargs.n_jobs` in the problem configs, or change `memory.omp_threads`. | **YES — `root_cfg_hash` moves, and `training_cfg_hash` for `model_kwargs`** (proven, R3-b: `9e74ad95…` vs `894f…`). | Same predictions (best-guess), but every config's identity changes before PR-082's freeze. | n/a |

### Path-tier checkpoint

**Tier-1, light path (proposed; best-guess)** — if the scope is A1 only. The defect is proven on the in-process paths (R3-a measured it), the change is one factory, one adapter and four call sites, and the intent is already documented (`docs/ARCHITECTURE.md:621`, D10). No premise of v0.5 changes, so there is no loop to design-planning. Against it: rux-ml `CLAUDE.md:52` makes post-v0 PRs Tier-2 by default, and `PROCEDURE-pr-research.md` has no tier-criteria text (grep found none). **Re-tier to Tier-2 if B (import order, BLAS) is folded in.** The operator's call at this gate.

### Gate leans (one decision each)

1. **Tier.** *Lean: Tier-1, light path, scope A1 only.* The two desktop probes (must-answer 2 and 3 below) run before the build as the light Phase 3, inside one heavy-lock window. *Alternative:* Tier-2 by `CLAUDE.md:52`'s default — all five phases before the PR is final. *Hashes:* none either way.
2. **How the budget reaches XGBoost (the transport).** *Lean: A1 — an explicit argument from `cfg.memory`* (a signature change at `make_trainer`, `make_xgboost_trainer` and the four call sites; the native adapter uses the `memory_cfg` it already holds). It covers promote and `fit_batches` with no pin. *Alternatives:* (a) **A2** — copy CatBoost's env read, no signature change; it inherits promote's missing pin, so re-fits stay at 32 threads and record `nthread` 0 against the trial's 24; (b) **B** — fix the import order so the env pin works; the recorded `nthread` stays 0 and the pool becomes 24. *Hashes:* `root_cfg_hash` / `training_cfg_hash` move under none of A1 / A2 / B; only the rux-ml `git_sha` moves. *Recorded fields:* A1 moves `booster_nthread`, `fold_meta` `booster.nthread` and `refit.nthread` 0 → 24 everywhere; A2 moves them except on promote; B moves none.
3. **What happens to the import-order defect (the broken "pin before heavy imports").** *Lean: a separate follow-up PR stub* (B, or B-lite: `trial_runner`'s config import chain and the false `:86` comment), carrying the BLAS / MKL question. *Alternatives:* (a) fold B-lite into PR-062 (small, but it adds a second root cause to a one-cause PR); (b) fold full B in (re-tier to Tier-2; wide import churn). *Hashes:* none under any choice.
4. **The default `memory.omp_threads`.** *Lean: keep 24 in PR-062* (code only; no hash moves). Q16 measured 8 as faster (median 0.186 s vs 0.241 s at 24) and both stall-free (0 in 12). *Alternative:* change the default to 8 — **moves `root_cfg_hash` for every config** (the memory layer is hashed with nothing elided); if wanted, its own config PR after PR-082's freeze.
5. **CatBoost's transport.** *Lean: CatBoost moves to the same explicit argument,* so there is one transport and its docstring drift (`catboost/factory.py:21-24` vs `:210`) closes. LightGBM accepts and ignores the argument (its lazy import is after the pin; best-guess it is not exposed). *Alternative:* CatBoost and LightGBM accept and ignore it; CatBoost keeps its env read and the docstring is corrected instead. *Hashes:* none. *Behaviour:* under the lean, a CatBoost fit with `OMP_NUM_THREADS` unset would get `memory.omp_threads` instead of today's -1 fallback (inference from `:197-216`; unchecked whether CatBoost records its thread count anywhere).
6. **The predict side** (`registry score`, a loaded bundle Booster with `nthread` 0 predicts at 32 threads). *Lean: out of scope unless measured; recorded as a note.* *Alternative:* `booster.set_param({"nthread": …})` on load at `registry/scorer.py:48` / `registry/bundle.py:23,51`, with one desktop timing of 12 predicts at 32 vs 24 threads first. *Hashes:* none.
7. **A trial recorded before the fix, promoted after it** (trial `nthread` 0 vs re-fit 24). *Lean: accept it and say so in the CHANGELOG line;* promote does not refuse on it today (no `src` code compares the two). *Alternative:* none needed in rux-ml; the program-side consumers are must-answer 6 (the lead's grep). *Hashes:* none.
8. **Merge order against PR-082's freeze.** *Lean: not tied — PR-062 does not block PR-082* (the harness pins `nthread` itself; the real cells run on cuda). The operator orders the merge. *Alternatives:* merge before the freeze (the frozen rux-ml commit carries the fix; recorded `nthread` reads 24) or after (the frozen commit is unchanged). *Hashes:* the configs' hashes are the same either way; only the recorded rux-ml commit differs.

### Must-answer (open at the gate)

1. **Transport** — gate lean 2 (decided from the call sites already read; not research).
2. **Bytes at nthread 0 vs `memory.omp_threads`.** Are `model.ubj` and `oos_rows.parquet` byte-identical between nthread 0 (32 threads) and nthread = `memory.omp_threads`, on cpu **and** cuda, for the in-memory classifier and the `[m9]` regression batch path? Best-guess today beyond Q16's cpu M6 predictions. *How:* one desktop run ≤ 5 min under the mkdir heavy lock — fit the c6 set at `n_jobs` unset vs 24 vs 8 through `rux-ml train` on both devices; compare sha256 of `model.ubj` and the oos export; diff `save_config`'s `generic_param`.
3. **The subprocess child.** Is xgboost loaded before the pin inside the trial child too, giving `omp_get_max_threads()` = 32? It sizes the follow-up (gate lean 3). *How:* one desktop probe ≤ 1 min — import `rux_ml.config` as `trial_runner` does, print `'xgboost' in sys.modules`, call `pin_threads`, read `omp_get_max_threads` via the vendored libgomp. The static chain is already proven.
4. **The default** — gate lean 4 (operator ruling).
5. **The predict side** — gate lean 6.
6. **Program-side consumers.** Does PR-082's freeze, the harness mirror or its export check read `fold_meta` `booster.nthread` or `TrialAttrs.booster_nthread`? A1 moves them 0 → 24 with no hash move; a program check pinned to 0 would break. *How:* the lead greps the harness repo and program PR-082 for `booster_nthread` and `"nthread"` (outside rux-ml).
   **ANSWERED by the lead 2026-10-15 (grep, harness `52763ed` src / scripts / tests; program scripts, PR-082 – PR-084):** no consumer
   compares rux-ml's recorded nthread. The harness's M9 serving model sets its own `nthread` on load, "whatever was saved"
   (`m9predicted/model.py:170`, `:216`; `tests/test_pr028b_adapter.py:273`). PR-082's D6 was folded at its Phase 4 to compare `device`
   only and RECORD both nthread values, because its mirror runs cpu nthread 8 (PR-082 ⚑3). So A1's 0 → 24 breaks nothing program-side.
7. **Tier** — gate lean 1.
8. **Docs in the same commit** — listed in Scope 5.

### Scope (Phase 2, light)

<!-- 2026-10-15 ≈ 21:12 CDT. The operator's "go" (≈ 20:56 CDT) approved gate leans 1 – 8 as stated: Tier-1 light; A1; the default stays 24; CatBoost takes the same argument; the import-order defect becomes a follow-up stub; the predict side is out of scope; a pre-fix trial vs a post-fix re-fit is accepted with a CHANGELOG line; the merge is not tied to PR-082. -->

| # | Must-answer | Status | Closed by |
|---|---|---|---|
| 1 | Transport | **closed: A1** | gate lean 2 (the go) |
| 2 | Bytes at nthread 0 vs `memory.omp_threads` | **closed** (proven on the probe shapes) | Phase 3 probe 1 |
| 3 | The subprocess child | **closed: yes, 32** (proven) | Phase 3 probe 2 |
| 4 | The default | **closed: stays 24** | gate lean 4 (the go) |
| 5 | The predict side | **closed: out of scope, kept as a note** | gate lean 6 (the go) |
| 6 | Program-side consumers | **closed: none** | the lead's grep (above) |
| 7 | Tier | **closed: Tier-1 light** | gate lean 1 (the go) |
| 8 | Docs in the same commit | **carried to the build** | Scope 5, Verification 11 |

Leans 5 (CatBoost) and 7 (a trial before the fix vs a re-fit after it) were also closed by the go. Questions 2 and 3 are independent and ran in one heavy-lock window.

**Excluded from this round (nice-to-have):**
- the import order and the BLAS / MKL / Polars pins, which go to the follow-up stub;
- the predict side's timing (lean 6);
- the stall rate at nthread 0 vs 24, carried to the build's performance budget (Synthesis, change 3);
- byte-neutrality on the real v0.5 tables. The cells run on cuda under the harness's own `nthread`.

### Findings (Phase 3, light: the two desktop probes)

**Window.**
- **Machine and checkout.** The desktop (`rux@100.90.42.41`); rux-ml `~/projects/rux-ml` at `814cc15`. It has one untracked file, `receipts/holdout_score_crypto_breakout_h3_2026-05-22.json`, which was left alone. Its `.venv` runs Python 3.12 and xgboost 3.2.0. `os.cpu_count()` = 32.
- **Pre-checks: clean.** No lock, no qBittorrent, the GPU idle (0 %, 1,273 MiB), load 0.02.
- **The heavy lock.** Taken by mkdir at **21:08:32 CDT** and released by the trap at **21:09:12 CDT** (40 s). One read-only `fold_meta` read came after the release.
- **The scratch dir.** `~/pr062-probe/`, kept, ≈ 17M by du.
- **What the probes ran on.** Made-up tables only; the problems `pr062probe_clf` and `pr062probe_m9reg`; a temp Optuna storage, artifacts, cas and manifests root per run, all inside the scratch dir. The scripts were validated on the Mac first (cpu).
- **Raw outputs** are in program `docs/0.5/evidence/ruxml-track/pr062-probes-20261015/`: `desktop-session.log`, `child_*.json`, `bytes.json` / `bytes.stderr`, `fold_meta_timings_cpu.txt` and both scripts.

**Probe 1: must-answer 2 (the bytes).**

*Method.*
- **How it runs.** `rux-ml train` runs in-process through Typer's `CliRunner`. xgboost loads at `rux_ml.cli` import, as it does in the real CLI. `omp_get_max_threads()` = 32 before and after the runs, and `OMP_NUM_THREADS` = "24" after `run_command`'s pin. `tuning.entropy` is pinned.
- **Thread settings.** `n_jobs` unset (nthread 0); unset again (a determinism control); 24; 8. Each is set with `--set training.model_kwargs.n_jobs`, the same kwarg A1 writes into `base`.
- **Devices.** cpu and cuda.
- **The two paths:**
  - **clf** — an `[m9]` in-memory XGBClassifier (logloss, the PR-060 export path): 20,000 rows × 8 features, 100 trees, depth 6.
  - **m9reg** — an `[m9]` mae regression: 5 day files × 10,000 rows, through `fit_batches` / `QuantileDMatrix`.
- **Capturing the model.** Wrappers on `train.make_trainer` / `train.fit_batches` capture the booster; no rux-ml file is changed. It is saved with `Booster.save_model(model.ubj)`, as `registry/bundle.py:51` does.

*Results.*
- **(proven, on these shapes) The thread count moves no bytes.** For each path and device, all four runs have the same `model.ubj` sha256, the same `oos_rows.parquet` sha256 (`oos_export_sha256`) and the same told value (repr):

| Path | Device | `model.ubj` (sha256 prefix, bytes) | OOS export | Told value | Equal across unset / repeat / 24 / 8 |
|---|---|---|---|---|---|
| clf | cpu | `52c8c60c39d3`, 405,527 | `39ba2eef53f5` | 0.2774132449442908 | yes |
| clf | cuda | `570b78a9ec27`, 405,319 | `44b67b9d142a` | 0.27927047571694563 | yes |
| m9reg | cpu | `03b51decbe54`, 473,084 | `15fb87acde83` | 0.583322804636058 | yes |
| m9reg | cuda | `a9ae1eb44f37`, 473,084 | `0ba7771d0569` | 0.5833228045059449 | yes |

- **(proven) In `save_config`'s `generic_param`, only `n_jobs` and `nthread` differ** ("0" → "24" / "8"). In xgboost 3.2.0 the booster's `nthread` is a config parameter, not model state. The recorded `booster_nthread` is 0 / 0 / 24 / 8.
- **(proven) A config-side pin moves both hashes.** The `n_jobs` runs moved `training_cfg_hash` and `root_cfg_hash`. Example, clf cpu: `c1326f61…` / `bb5fac6a…` with `n_jobs` unset vs `a6b93f0f…` / `29e1ce4b…` at 24. This re-proves R3-b, so option E stays rejected. A1's "no hash move" holds by construction until Verification 7 checks it.
- **(proven) `TrialAttrs.omp_threads` recorded 24 in every run while the pool was 32.** R3-a reproduces through `rux-ml train` in-process.
- **(proven, one event) A stall reproduced through `rux-ml train` at nthread 0.** The second m9reg cpu run with `n_jobs` unset took 27.54 s wall. Of that, `boost` (the `xgb.train` call) took **26.866 s**, against 0.070 – 0.074 s in the other three m9reg cpu runs (`fold_meta_timings_cpu.txt`).
  - Stall count by setting: 1 of 2 at nthread 0 on the batch path; 0 of 2 at nthread 0 on clf; 0 of 1 each at 24 and 8.
  - **(best-guess)** The 32-thread pool is the cause; R3-a's 16.81 s stall was also at 32. These counts are far too small to rank 0 vs 24 vs 8.
- **(best-guess) Not measured beyond these shapes:** the real v0.5 tables, larger trees or more rows, other xgboost versions.

**Probe 2: must-answer 3 (the subprocess child).**

*Method.* Three fresh interpreters, with `OMP_NUM_THREADS` unset unless stated:
- **unset:** import xgboost, then fit.
- **trial_order:** replays `python -m rux_ml._internal.trial_runner`'s order — the module's own imports, `from rux_ml.config import RuxMLConfig` (`:87`), the config, `pin_threads` (`:97-99`) — then fits.
- **control:** `OMP_NUM_THREADS=24` is set before the interpreter starts.

In each, `omp_get_max_threads()` is read through ctypes on every libgomp in `/proc/self/maps`, and the thread count from `/proc/self/task`.

*Results.*
- **(proven) Yes: the trial child loads xgboost before it pins.**
  - After the trial_runner module import alone: neither xgboost nor numpy is loaded, and no libgomp is mapped.
  - After `from rux_ml.config import RuxMLConfig`: xgboost and numpy are in `sys.modules`, and libgomp is mapped.
  - After `pin_threads`: `OMP_NUM_THREADS` = "24" but **`omp_get_max_threads()` = 32**.
  - Control (the env set before start): 24. Baseline (unset): 32.
- **(proven) A default fit's thread team follows the pool.** A default fit added 34 threads in unset (64 → 98) and in trial_order (65 → 99), and 26 in the control (48 → 74). The difference of 8 is 32 − 24.
- **(proven) The process has one libgomp: scikit-learn's vendored copy** (`site-packages/scikit_learn.libs/libgomp-e985bcbb.so.1.0.0`), in all three modes.
  - xgboost's pool is that library's. That only one libgomp is mapped is proven; how the loader picks it is a best-guess.
  - The ctypes read works on Linux, so the follow-up's effect test is feasible there. macOS uses a different OpenMP runtime (not checked).
- **(best-guess) The BLAS pin is probably a no-op in the child too.**
  - Before any fit, the trial_order process held 65 threads against 64 unset. Yet `pin_threads` had set `OPENBLAS_NUM_THREADS=1`, after numpy was already loaded.
  - In the control, `OMP_NUM_THREADS=24` alone gave 48.
  - The threads were not attributed per library. This goes to the follow-up stub.

### Synthesis (Phase 4, light)

**Outcome: Confirm.** The probes support A1 as scoped.
- **What A1 visibly changes.** The thread count moves no model bytes, no export bytes and no told value, on cpu or cuda, on either path. What A1 visibly changes is the recorded `nthread` (0 → `memory.omp_threads`) and the commit, as the CHANGELOG line (lean 7) states.
- **Must-answer 3 widens the follow-up, not PR-062.** A1 reaches the subprocess child's fits as well, because the trainer carries `n_jobs` explicitly whatever the pool size. Tune's objective (`tuning/objective.py:248`) is one of A1's four sites.

**Changes to this PR** (leans, approved by the build go):
1. **Verification 8 becomes a before/after re-run of probe 1.**
   - What runs: the same script on the branch head and on `dev` `814cc15`, cpu + cuda, both paths. The branch runs at its default with no `--set`; `dev` runs with `n_jobs` unset.
   - Pass: `model.ubj`, the OOS export and the told value are equal, and the branch records `booster_nthread` 24.
   - Any byte difference goes to the lead before the merge.
2. **A new effect check beside it (desktop, Linux).** In a fresh interpreter with `OMP_NUM_THREADS` unset, a default fit on the branch must add a 24-thread team (≈ +26 threads on this box, the control's figure), not a 32-thread one (≈ +34). This reads the fix by its effect, not by a parameter (the PR-011 lesson). The result is recorded; a miss is reported.
3. **The performance budget names its fit shape and gains a comparison arm.**
   - The arm: 12 default cpu fits on the `pr062probe_m9reg` shape on the branch (nthread 24) and 12 on `dev` (nthread 0, a 32-thread pool).
   - What is reported for each: the stall count (a stall = `boost` > 10× the run's median) and the `boost` median. Reported, not gated: 12 fits cannot rank a rare event, and probe 1 saw 1 stall in 2 at nthread 0.
   - A stall at 24 is reported to the lead, because it would reopen gate lean 4 with data.
4. **Unchanged:** the Scope, the tests (Verification 1 – 5), and Verification 6, 7 and 9 – 12.

**Changes to ARCHITECTURE.md / CONSTRAINTS.md:** none. A1 implements `docs/ARCHITECTURE.md:621` as written; the CONVENTIONS edits are already in Scope 5.

**New PRs that must come first:** none.

**The follow-up stub (lean 3).** A rux-ml PR file with its procedure pending and its research not started. It covers:
- the import order: the trial child's `rux_ml.config` chain (proven here, with the numbers above) and the CLI's verb imports;
- the false `trial_runner.py:86` comment;
- the BLAS / MKL / Polars pins (best-guess no-ops);
- the stall rate.

**Research-backed details now locked:**
- A1 (lean 2);
- `memory.omp_threads` is the source and the default stays 24 (lean 4);
- CatBoost takes the same argument; LightGBM accepts and ignores it (lean 5);
- no byte move from the thread count on cpu / cuda × the in-memory classifier / the `[m9]` batch path (probe 1, on the shapes above);
- today the subprocess child runs at 32 (probe 2).

**The build go-list (rux-ml PR-062, Phase 5 gate):**

*"Go" approves:*
1. The rux-ml PR file, written from this draft (the Research findings and the three changes above), and a branch `pr-062` off `dev` `814cc15` in its own worktree.
2. Tests first (Verification 1 – 5; test (4) must go red on `dev` by assertion). Then A1, the native adapter and CatBoost / LightGBM, with the docs in the same commit (Scope 5 and the CHANGELOG line).
3. The checks:
   - the Mac checks (9);
   - the desktop suite diffed against `dev` (10);
   - the hash lists (7);
   - one mkdir heavy-lock window for 8, the effect check and the budget arm, clear of the 23:10 / 01:30 CDT timers.
4. The follow-up stub file (doc only).
5. Pushing the `pr-062` branch for CI. No merge.

*"Go" does not cover:*
- the merge to rux-ml `dev` (the operator's word; not tied to PR-082, lean 8);
- any config change (the default 24; `n_jobs` in any config);
- the predict side;
- the follow-up's research or build;
- deleting the prior-art branch `pr-056/promote-pin-threads`;
- deleting `~/pr062-probe/` (≈ 17M by du; kept).

*Estimate:*
- **Tokens:** one builder ≈ 68k + 3k × ~90 calls + a ~30k pack ≈ 0.37M, plus the lead's review and desktop window ≈ 0.1M — **≈ 0.45 – 0.55M** in all, inside the 750k per-run cap.
- **Desktop time:** ≈ 10 – 15 min under the lock. The suite run dominates and is not measured here (best-guess). Probe 1 took 37 s; the budget arm is 24 fits, plus up to ≈ 27 s for each stall on the `dev` side.
- **Spend:** $0.

---

## Scope

1. **The factory takes the budget (A1).** `make_xgboost_trainer` gains a thread argument; `make_trainer` passes it through; the value is `cfg.memory.omp_threads` at `cli/train.py:481`, `tuning/objective.py:248`, `registry/promote.py:246` and `training/xgboost/batches.py:228`. It is written to `base["n_jobs"]` **before** `base.update(cfg.model_kwargs)`, so `model_kwargs` still wins.
2. **The native adapter** sets `params["nthread"] = memory_cfg.omp_threads` before its `model_kwargs` update (`native_adapter.py:118-131`).
3. **CatBoost / LightGBM** take the argument (gate lean 5: CatBoost uses it; LightGBM accepts and ignores it).
4. **Tests first** (Verification 1 – 4), then the change.
5. **Docs in the same commit:** `docs/CONVENTIONS.md:450-460` (the XGBoost row → the factory passes `n_jobs`; name the transport), `:359` (attribute bit-exactness to `n_jobs=1`), `src/rux_ml/_internal/env.py:1-31` (the guarantee as it now stands), `README.md:629`, the CatBoost docstring (`catboost/factory.py:21-24`), and a CHANGELOG `[Unreleased]` line naming the recorded-field change 0 → `memory.omp_threads` (gate lean 7). `_internal/trial_runner.py:86` only if B-lite is folded in (gate lean 3).

### Not in scope

- The import order (option B / B-lite), the false `trial_runner.py:86` comment and the BLAS / MKL / Polars pins — a follow-up stub (gate lean 3).
- Any config change: `memory.omp_threads`' default, `training.model_kwargs.n_jobs` in any config (option E) — both move `root_cfg_hash`.
- The predict side (gate lean 6).
- The prior-art branch `pr-056/promote-pin-threads` (not adopted, program PR-027:799).
- Any promote refusal on a trial / re-fit `nthread` mismatch.

## Dependencies

None open in rux-ml: `dev` `814cc15` carries PR-060 and PR-061. Program side: must-answer 6 (the lead's consumer grep) before the merge. PR-082 is not blocked by this PR and does not block it (program PR-082:1481-1486); the merge order is gate lean 8, the operator's word.

## Architecture section implemented

`docs/ARCHITECTURE.md:621` — the D10 thread allocation ("XGBoost `nthread=24`"), recorded at `docs/0.0/DESIGN-log.md:102`. `docs/CONVENTIONS.md:450-460` (the per-family threading table) is corrected to match.

## Verification criteria

- [x] **1. Test first, failing on `dev` by assertion.** Test (4): a default `rux-ml train` CLI fit records `booster_nthread == cfg.memory.omp_threads` (shaped like `tests/cli/test_m9_batch_train_cli.py:69-73`). On `dev` `814cc15` it fails on the assertion (`booster_nthread` 0 ≠ `memory.omp_threads`), not on an import or collection error; after the change it passes. *Evidence:* both `pytest -rA` outputs with their commits. (Test (1) below calls the new argument, so on `dev` it fails with a signature error; the assertion-level red is test (4).)
- [x] **2. The factory.** Test (1): `make_xgboost_trainer` built with memory `omp_threads=7` has `get_params()['n_jobs'] == 7`, and after a tiny fit `booster_threads_device()['nthread'] == 7`.
- [x] **3. `model_kwargs` still wins.** Test (2): `model_kwargs` `n_jobs=1` with memory `omp_threads=7` gives `n_jobs` 1 and booster `nthread` 1.
- [x] **4. The native adapter.** Test (3): the native adapter's params carry `nthread` from `memory_cfg.omp_threads`, and a `model_kwargs` `nthread` still wins.
- [x] **5. The `[m9]` batch path and promote.** A default `[m9]` batch fit records `booster_nthread == cfg.memory.omp_threads`, and `test_m9_promote_manifest_refit` passes with both the trial and the re-fit at that value (changed by design). *Evidence:* the test output.
- [x] **6. Unchanged by construction:** golden 15 / 15, `tests/integration/test_determinism_cpu.py`, the A11 `n_jobs=3` test, the c6 pin (`tests/cli/test_data_hash_per_file_c6.py`). *Evidence:* the junit `testcase` entries by name.
- [x] **7. No hash moves.** `root_cfg_hash` and `training_cfg_hash` for every config under `configs/` are equal on `dev` and on the branch. *Evidence:* both lists, diffed.
- [x] **8. Bytes, before / after (must-answer 2; Synthesis change 1).** On the desktop, under the mkdir heavy lock: the probe of program `docs/0.5/evidence/ruxml-track/pr062-probes-20261015/probe_bytes.py` re-run on the branch head (its default, no `--set`) and on `dev` `814cc15` (`n_jobs` unset), cpu + cuda, the in-memory classifier and the `[m9]` regression batch path. Pass: `model.ubj`, the OOS export (sha256) and the told value equal branch vs `dev`, and the branch records `booster_nthread` 24. Any byte difference is reported to the lead before the merge — never waived silently. *Evidence:* a record under program `docs/0.5/evidence/ruxml-track/`.
- [x] **8b. The effect (Synthesis change 2; Linux).** In a fresh interpreter with `OMP_NUM_THREADS` unset, a default fit on the branch adds a 24-thread team (≈ +26 threads on the desktop, probe 2's control figure), not a 32-thread one (≈ +34). Recorded; a miss is reported.
- [x] **9. Mac:** `uv run pytest`, `ruff check --no-cache .`, `ruff format --check src tests`, `basedpyright src/` clean.
- [x] **10. Desktop:** `uv run --frozen --all-extras pytest -n 8` diffed against `dev` — no new failure.
- [x] **11. Docs in the same commit** (Scope 5). *Evidence:* the commit's diff.
- [x] **12. Program-side consumers checked** (must-answer 6) by the lead before the merge.

## Performance budget

Not a per-event path: the change sets one parameter per fit. The effect it must show is the fit's thread count.

- **Baseline (program PR-082):** on the desktop, `OMP_NUM_THREADS` 24 with `omp_get_max_threads()` 32 after the pin, and a cpu M6 fit stalled at **16.81 s** (R3-a, :1205-1210). Q16 (i9-13900K, fits only): **0 stalls in 12** at both 24 and 8, medians **0.241 s** at 24 and **0.186 s** at 8 (:1462-1477).
- **Budget (lean; Synthesis change 3):** a default cpu fit through `rux-ml train` runs at `memory.omp_threads` (booster `nthread` = 24 at the default). The fit shape is named: the `pr062probe_m9reg` shape of probe 1 (an `[m9]` mae regression, 5 day files × 10,000 rows, through `fit_batches`).
- **The comparison arm:** 12 default cpu fits on that shape on the branch (nthread 24) and 12 on `dev` (nthread 0, a 32-thread pool). Reported for each: the stall count (a stall = `boost` > 10× the run's median) and the `boost` median. Reported, not gated — 12 fits cannot rank a rare event (probe 1 saw 1 stall in 2 at nthread 0). A stall at 24 is reported to the lead first: it would reopen gate lean 4 with data.
- **Closed by** the desktop measurement in Verification 8's window. A miss is reported, never waived silently.

## Research backing

- Phase 1 (this assessment), one read-only agent, 19 tool calls; by the cost model ≈ 1 × (68k + 3k × 19) ≈ 0.13M (the actual is the lead's).
- Program `prs/PR-082-probe-runner-statistic-null-holm-freeze.md`: R3-a (:1205-1210, the env pin after import), R3-b (:1211-1216, the config hashes), Q16 (:1462-1477, stalls and medians), :1479-1486 (cuda not exposed; the harness's own `nthread`).
- Program `prs/PR-027-*.md:799` (R4; the prior-art branch not adopted).
- `docs/ARCHITECTURE.md:621`; `docs/0.0/DESIGN-log.md:102` (D10).

## Notes

- **(best-guess)** BLAS / MKL pins are probably no-ops by the same mechanism; Polars may still honor its env var if its pool is lazy. The follow-up stub carries this (gate lean 3).
- **(best-guess)** LightGBM imports lazily after the pin, so its libgomp probably honors the env var; the CONVENTIONS row is unverified.
- **(best-guess)** On cuda, `n_jobs` affects only host-side work (e.g. DMatrix construction), so cuda predictions are expected unchanged; the only visible change is the recorded `nthread` 0 → 24. Whether PR-082's freeze or export reads `fold_meta` `booster.nthread` is must-answer 6.
- **(proven)** `TrialAttrs.omp_threads` records the intended budget, not what ran; after A1 the booster's `nthread` records what the booster was given.
- **(convention)** rux-ml's default 24 is not the fastest measured setting (Q16: 8 faster); changing it moves every `root_cfg_hash` and is a separate config PR (gate lean 4).
- **(best-guess)** The predict side runs at 32 threads; its exposure to the stall is unmeasured (gate lean 6).

## Build status (2026-10-16)

<!-- The builder, on the operator's build go of 2026-10-16 ≈ 08:49 CDT. Branch `pr-062-threads` (off `dev` `814cc15`), worktree `~/projects/rux-ml-pr-062`; NOT pushed (the operator's word "push"), NOT merged. Evidence: program `docs/0.5/evidence/ruxml-track/pr062-build-20261016/` (`pr062-build-summary.json`, `mac/`, `desktop/`, `scripts/`). -->

**Commits:** `5e4c5f2` (this file + the tests first, red on `dev`'s code), `d07450e` (A1 + the docs + the CHANGELOG line), `c2096e5` (the PR-063 stub), and this status commit.

- **1. Red first (by assertion).** On `dev` `814cc15`'s code with the new tests: 13 failed / 5 passed. Every CLI path fails on the assertion `booster_nthread` / `refit.nthread` 0 == 24 (train sklearn + native, tune in-process + subprocess, the in-memory promote re-fit, the `[m9]` batch fit, the `[m9]` promote re-fit); the factory / adapter tests fail on the missing `threads` argument (`TypeError`) and the adapter's missing `nthread` key (`KeyError`), as Verification 1 expected. After the change: 55 passed in the targeted set (`mac/red_tests_on_dev_code.txt`, `mac/green_targeted.txt`).
- **2 – 5.** `tests/training/test_thread_budget.py` (8 tests: the factory and `make_trainer` at `omp_threads=7`, booster `nthread` 7; no budget → `n_jobs` unset; `model_kwargs` `n_jobs=1` wins; the native adapter's `nthread` from `memory_cfg` and a `model_kwargs` `nthread` winning; CatBoost `thread_count` from the argument, the env no longer; LightGBM unchanged params) and `tests/cli/test_thread_budget_cli.py` (7 tests, above); `test_m9_promote_manifest_refit` gains `refit.nthread == attrs.omp_threads`; `test_catboost_smoke`'s env test now reads the argument (changed by design, gate lean 5).
- **6.** Desktop junit: golden 15 / 15 on both trees, `test_determinism_cpu` (2), the A11 `n_jobs=3` test, the c6 pin (5) passed on both.
- **7. No hash moves.** `root_cfg_hash` + `training_cfg_hash` over 35 layer combinations of `configs/` (base, × 4 problems, × 6 studies; 30 hashed, the same 5 combinations refused by validation on both sides): the two lists are equal, `diff` empty (`mac/hashes_branch.tsv`, `mac/hashes_dev.tsv`, `mac/hashes_diff.txt`). `configs/` is unchanged (0 diff lines).
- **8. Bytes, before / after (the desktop).** Probe 1 re-run on the branch (default, repeat, `n_jobs` 8) and on `dev` (unset), cpu + cuda, both paths: `model.ubj`, the OOS export and the told value equal branch vs `dev` on all 12 rows; the branch records `booster_nthread` 24 (8 at `--set n_jobs=8`), `dev` 0; `generic_param` differs only in `n_jobs` / `nthread`; the default rows' config hashes equal. The `model.ubj` sha256 prefixes are probe 1's (clf cpu `52c8c60c39d3`, clf cuda `570b78a9ec27`, m9reg cpu `03b51decbe54`, m9reg cuda `a9ae1eb44f37`). No byte difference to report.
- **8b. The effect.** Fresh interpreter, `OMP_NUM_THREADS` unset, the trial child's order, a default fit through `make_trainer`: the branch adds **26** threads (65 → 91, booster `nthread` 24), `dev` adds **34** (65 → 99, `nthread` 0). `omp_get_max_threads()` is 32 on both (the pool is still sized before the pin — PR-063).
- **9. Mac.** `uv run pytest` 800 passed / 3 skipped (exit 0); `ruff check --no-cache .`, `ruff format --check src tests`, `basedpyright src/` (0 errors) — each gated on its bare exit status.
- **10. Desktop suite** (`uv run --frozen --all-extras pytest -n 8`): branch 798 passed / 5 skipped / 0 failed in 878.8 s; `dev` 784 / 5 / 0 in 1,100.8 s. Diff: no new failure, no status change on any common test, +15 new tests, 1 renamed, the same 5 skips.
- **11.** The docs ride in `d07450e` (CONVENTIONS ×4, `env.py`, README, the CatBoost docstrings, the CHANGELOG line, the backlog row).
- **12.** Answered by the program lead 2026-10-15 (must-answer 6: no consumer compares rux-ml's recorded `nthread`).

**Performance budget — closed.** The arm (12 default cpu fits of the `pr062probe_m9reg` shape per tree, `boost` = the `xgb.train` call):

| Tree | nthread | `boost` median | min – max | Stalls (rule: > 10× the run's median) | > 10× the branch median |
|---|---|---|---|---|---|
| branch | 24 | **0.0718 s** | 0.0595 – 0.0766 s | **0 / 12** | 0 / 12 |
| `dev` | 0 (a 32-thread pool) | **27.24 s** | 3.24 – 32.44 s | 0 / 12 (its own median is stalled) | **12 / 12** |

No stall at 24, so gate lean 4 is not reopened. At nthread 0 every `dev` fit of this shape ran slow today (probe 1 saw this shape at 0.070 – 0.074 s when not stalled, and 1 stall in 2). The model bytes are equal across all 24 fits (`03b51decbe54`). Reported, not gated; the stall's mechanism is PR-063's.

**Deviations.**
- The heavy lock was held in four segments, not one: the `dev` arm hit its 210 s timeout after 7 of 12 fits (every fit stalled, ≈ 27 s each), and the first foreground branch suite hit a 450 s timeout at ≈ 85 % (the desktop suite takes ≈ 15 – 18 min; the Mac's 33 s). Each time the trap released the lock; it was re-taken within a minute; the `dev` arm was completed with 5 more fits in a separate scratch dir, and both suites re-ran in the background. Lock segments (CDT): 09:03:27 – 09:07:56, 09:08:44 – 09:18:46, 09:19:37 – 09:52:39, 09:52:54 – 09:53:00.
- The branch is `pr-062-threads` (the draft's go-list named `pr-062`). Pushing it (the draft's go-list item 5) waits on the operator's word "push".
- During both desktop suites the load average reached ≈ 135 – 195 (8 xdist workers, each fit a 24- or 32-thread team on 32 logical CPUs). Not gated; recorded for PR-063.
