# PR-060: the in-memory classification path writes the per-row OOS export and runs val-less

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**Tier: Tier-1 (approved at the Gate Check below).** This PR is a track of the rux-capital program lead. Its research is program PR-082's Phase 1 (`prs/PR-082-probe-runner-statistic-null-holm-freeze.md`, `wf_b8bc6040-dc2`; the lens file `pr082-p1-results.json`). The operator approved that gate on 2026-10-14 ("go"): **lean (1)** says route (b): rux-ml's in-memory path writes the per-row OOS export (`predict_proba[:, 1]`) through a rux-ml PR under rux-ml's own process, before the tag. **Lean (2)** says val = 0 runs through a val-less mode in that PR, and the told value is the test score. **Lean (3)** says `early_stopping_rounds = 0` goes in every toml, with a pinning test. The merge belongs to the program lead and must land on `dev` before the v0.5 freeze tag.

## Research findings

### State Assessment (2026-10-14)

Baseline: rux-ml `dev` @ `09b9bd5` (the PR #64 merge of PR-059). There is no drift since the program's lens reads. On 2026-10-01 the desktop default suite there gave 2 failed / 755 passed / 3 skipped. The 2 failures are the C10 digest pins `tests/cli/test_data_hash_per_file_c6.py::…[fill]` and `[walk]` (program `docs/0.3/evidence/close-out/c1-ruxml-20261001.json:50-58`).

**Current state**:
- **The export is wired only to the batch branch.** `run_command` *always* passes `export_dir=Path(export_tmp)` to `_fit_and_score` (`cli/train.py:585, :599-601`). The batch branch forwards it (`:264-269` → `:488-497`). The in-memory branch (`:270-360`) never reads it.
- **Downstream of `_fit_and_score`, the plumbing is already route-agnostic.** `run_command` reads `fold_extras['oos_export']` (`:635`), uploads the file (`:636-643`) and records `oos_export_sha256` (`:434-436, :624`). promote copies the sha into the manifest (`registry/promote.py:435`). Part (1) needs no `run_command` change.
- **`OosExportWriter` assumes nothing about batch or regression in its code** (`runs/oos_export.py:75-172`):
  - It needs non-empty, non-clashing `key_columns` (`:83-93`).
  - It casts `prediction` to Float32 and `realized` to Float64 (`:136-147`).
  - Its first `write` fixes the schema (`:149-151`), and `close()` refuses an empty file (`:159-161`).
  - Only the docstring states a batch contract: "per-day batch fit" (`:3-4`) and `prediction` = "XGBModel.predict's value up to the best iteration" (`:17-18`).
  - There is no key-uniqueness check.
- **The in-memory branch already has every input the writer needs.** The splits carry the key columns before the target is stripped (`:271, :279-280`). The transformed frames exist (`x_val_pd` `:339`; `x_pd` in `_oos_record` `:197`). The fitted `XGBClassifier` exposes `predict_proba`, whose `[:, 1]` is the array `_logloss` scores (`training/metrics.py:61-81`). So the exported p̂ and the logged logloss come from identical probabilities.
- **`_oos_record` (`:174-214`)** scores on rows with a non-NaN label (`:198-203`). Its `model.predict` call, which returns class labels for a classifier, runs only under honesty or calibration (`:204-213`). The probe cells set neither.
- **val = 0 is accepted up to the fit:**
  - `validate_ratios` checks only the key set, non-negative values and a sum of 1 (`data/partitions.py:70-79`; `config/data.py:52-55`).
  - On a `time_ordered` `[m9]` split, `val_first_stamp == test_first_stamp` (`partitions.py:406-415`). VAL is empty, and there is **one** embargo band before test (`:301-307`).
  - `make_splits`, the leakage audit and `_label_diagnostics` all tolerate an empty val (`data/splits.py:225-230`; `data/leakage.py:45-58, :72-77, :138-145`; `cli/train.py:107-129`).
  - What fails is `XGBClassifier.fit(eval_set=[(empty, empty)])` (`:340-345`) and the val score (`:346`). The exact raise site was not executed (best-guess).
- **The told value is the val score.** The code calls `run.tell(score)` (`:664`), and `metrics.json` comes from `build_metrics_dict(metric, [score])` (`:644`). `fold_meta.json` = fold_idx / row_count / fit_seconds / timestamp + `**fold_extras` (`:645-663`). The test score already exists as `fold_extras['oos']['score']` (`:352-353` → `:203`).
- **The three M9 configs never reach this code.** `m9_walk_bp` (mae), `m9_markout_bp` (mae) and `m9_fill_frac` (brier) are xgboost regression configs, so `uses_batch_fit` is True (`training/xgboost/batches.py:64-72`). They fit through `_fit_and_score_batches` (`cli/train.py:439-530`) / `fit_batches` (`batches.py:205-260`) and promote's `_refit_batches` (`promote.py:210-211, :247-290`).
- **promote's in-memory re-fit is a near-copy of the train branch.** It always passes `eval_set=[(x_val_t, y_val)]` and records `val_rows` (`promote.py:212-243`). It does not re-run the export; it copies the trial's sha (`:435`) and `oos` (`:331-341`). `rux-ml train` writes no model file (no `save_model` in `cli/train.py`).
- **The native branch (`use_native = True`) already accepts `eval_set=None`.** Early stopping is added only when the value is not None and the evals are non-empty (`training/xgboost/native_adapter.py:176-216`). That branch still scores on val (`cli/train.py:320`). The cells use the sklearn wrapper (`use_native = False`).
- **Early stopping at the pinned xgboost 3.2.0** (`uv.lock:1925-1926`; floor `xgboost>=3.1`, `pyproject.toml:19`):
  - The only gate is `if early_stopping_rounds:`, so `0` is off.
  - The EarlyStopping callback raises "Must have at least 1 validation dataset for early stopping." when it has no evals (`xgboost/callback.py:458`).
  - rux-ml's default is `early_stopping_rounds = 50` (`training/base.py:57`).
  - Versions on the Mac `.venv`: xgboost 3.2.0, scikit-learn 1.8.0, CPython 3.12.15. On the desktop (2026-10-01, `uv run --frozen`): xgboost 3.2.0.
- **Tests that pin these paths:**
  - `tests/cli/test_m9_batches.py::_in_memory` (`:66-90`) is a self-contained *copy* of the pre-PR-054 in-memory fit. The bit-identity test (`:116-121`) cannot move.
  - `tests/cli/test_train_subcommand.py:396-414` runs an `[m9]` + auc in-memory fit with `row_key_columns = [x1, x2]` (`:302`). It asserts exactly fit + `eval_set` (`:410, :414`), and it would newly write and upload an export.
  - `tests/golden/test_xgb_baseline.py` references `_fit_and_score`, which pins its return-tuple shape.
  - No test uses `early_stopping_rounds = 0` (they use None or 2: `tests/training/test_factory.py:61, :87, :132-142`; `test_native_adapter.py:47`).
  - No test fits with val = 0.
  - No test covers a classification `[m9]` fit.
- **Prior art** (`git log` of `cli/train.py`, `runs/oos_export.py`): the M9 series, each Tier-1 on a program track. PR-041 (configs), PR-042 (split / embargo), PR-044 (gates reader), PR-045 (leakage audit), PR-054 (batch fit), PR-055 (the export), PR-058 (honesty forms), PR-059 (manifest re-fit fields).

**Assumptions at PR draft time** (the program's approved leans and texts):
- "The in-memory path writes the export when `export_dir` is given."
- The cells' val plan is justified by the batch path, which "always builds the val matrix" (program page `docs/0.5/PREREG-alpha-probe.md:232-234`, citing `batches.py:240-242`).
- `early_stopping_rounds = 0` is not "off" and needs a "0 → None" code change (program `docs/0.5/ROADMAP.md:31`; page :240-242 and §12 item 1, :345-347).
- One rux-ml row covers both track items (the digests + the early-stopping item), sized at ≈ 0.3–0.4M (ROADMAP:31).
- The equality-test reference is `batches.py:228-252` (program ACCEPTANCE C2 :69-73; ROADMAP:27; PR-082 §Scope 3).
- `auc` is reported beside logloss (page :240).

**Stale assumptions** (where current state disagrees):
- **"When `export_dir` is given" is always true** (`cli/train.py:600`). The export therefore switches on for *every* `[m9]` in-memory fit. Such a fit with empty `row_key_columns` (the default `[]`, `config/m9.py:63`; required only for `split_kind = random`, `:59-62`) would be newly refused by the writer (`oos_export.py:84-89`), where today it trains. Not read: whether the `[m9]` fixtures at `test_train_subcommand.py:447, :477` carry keys.
- **The cells take the in-memory path, not the batch path**, so the page's batch-path val reasoning (:232-234) does not apply. The equality-test reference is `cli/train.py:279-360` with the feature Pipeline, not `batches.py:228-252`. These are program texts and belong to the lead.
- **`0` is already off at 3.2.0.** No code change is needed, only the pinning test (lean 3 already replaced the "0 → None" change). The program texts at ROADMAP:31 and page :240-242 / :345-347 are the lead's to correct at the freeze clean-up.
- **rux-ml records one metric per fit** (`cli/train.py:644, :666`). AUC is not available unless the harness computes it from the export or the page drops the clause.
- **The export's `prediction` would take a new meaning under the same column name.** Schema v1 says `XGBModel.predict`; the probe needs P(class 1).
- **The told value's meaning changes in val-less mode.** CHANGELOG:8 (PR-059) says the manifest's `metric_value` "stays the Optuna value, the VAL score for an [m9] fit".
- **A val-less trial cannot be promoted as written**, because promote's in-memory re-fit always builds `eval_set` from val (`promote.py:226-231`). Program PR-084 needs "the model file's sha256 reproduced Mac = desktop" on the same route.
- **The ROADMAP:31 size covered the digests + the early-stopping item only.** The approved PR-060 scope is larger, so the estimate must be re-stated.
- *(Cosmetic, not this PR's)*: rux-ml `CLAUDE.md:46-48, :69-72` still points at `docs/0.1` and says "No code yet".

**New constraints**:
- **The M9 regression route stays byte-for-byte.** All new code goes below `cli/train.py:269`'s and `promote.py:211`'s batch dispatch. The following stay untouched:
  - `uses_batch_fit` and `fit_batches`' classification / val = 0 refusals (`batches.py:64-72, :209-216`);
  - `_fit_and_score_batches` / `_oos_record_batches` / `_test_sinks` / `_export_sink` (`cli/train.py:363-530`);
  - `predict_fold` and `_xgb_kwargs` (`factory.py:47-62`);
  - `OosExportWriter`'s on-disk bytes for regression files (metadata JSON :98-109, `OOS_EXPORT_SCHEMA_VERSION` :54, column order :97, casts :145-146);
  - `_refit_batches`.

  Any of these moving changes every M9 export's sha256 and the recorded `oos_export_sha256`.
- **Non-`[m9]` in-memory fits stay unchanged**, including the golden baseline. Both new branches are gated on `cfg.m9 is not None`. With val > 0, the current `eval_set` fit and val score run unchanged, and `_fit_and_score`'s 5-tuple return is kept.
- **No new config field.** The val-less mode keys on the existing `split_ratios['val'] == 0`, and early stopping on the existing `early_stopping_rounds`. A new field could move every recorded config hash if defaults are serialised. That was not verified (`runs/attrs.py` `TrialAttrs.from_cfg` was not read), so avoiding a new field is the safe side.
- **Val-less mode requires early stopping off.** At the default 50, xgboost raises (callback.py:458) instead of rux-ml refusing by name. The mode refuses `early_stopping_rounds ∉ {None, 0}` by name, before the fit.
- **The export is refused before the fit, never after**, as on the batch path (`cli/train.py:488`). Writer construction errors map to `typer.BadParameter`, exit 2 (`:496-497`).
- **The pinning test asserts behaviour, not a version string**, because the floor `>=3.1` admits versions the gate was not checked on.
- **No phantom implementation.** Each item gets an end-to-end `rux-ml train` test on a made-up classification `[m9]` table: no real data and no cell fit (`CLAUDE.md:58, :89`).
- **Goldens are tolerance-based**, because GPU hist is not bit-exact (`factory.py:40-45`). `device = cuda` tests carry the `gpu` marker, and default-suite tests run on CPU.
- **Docs ride in the same commit** (`CLAUDE.md:56`): the CHANGELOG `[Unreleased]` entry with both meaning changes named, the `oos_export.py` docstring, the ARCHITECTURE export section if it names the batch path only, and a `docs/0.3/RESEARCH-BACKLOG.md` row. Track PRs are recorded there (:50-53), not in the 0.3 ROADMAP.
- **rux-ml has no hosted CI** (no `.github`). Verification is local on the Mac, then on the desktop diffed against `dev`. Program C12 does not cover rux-ml commits.
- **Merge order against the freeze.** The freeze records rux-ml's commit (program PR-082:419-421), and any change after the tag is a new trial (page :7). PR-060 and any digests PR must land on `dev` before the tag.

**Downstream contracts**:

| Consumer | Contract | Satisfiable |
|---|---|---|
| Program PR-082: the per-row export | `oos_rows.parquet` for every test row (and val rows when val > 0), holding `partition`, `row_key_columns` (`stamp_ms`, `coin`[, `event_id`]), `prediction` Float32 = `predict_proba[:, 1]`, `realized` Float64 = the binary target (NaN kept). Its sha256 goes in fold_meta `oos_export` and in trial attr `oos_export_sha256`; it is uploaded as an artifact. | **yes**, after Scope (1). The writer and the upload exist. The realised excess return is not in the export, so PR-082 joins by key. |
| Program PR-082 / PR-083: the val-less cell fits (page val = 0.00) | A completed trial with no `eval_set` and no early stopping; told value = the test score; fold_meta `oos` written; one embargo band; prefix fits via `--set data.train_prefix_frac` readable. | **yes**, after Scope (2). Today the fit raises at `cli/train.py:340-346`. |
| Program PR-082: no validation-driven early stopping | `early_stopping_rounds = 0` is off at the installed xgboost, and val = 0 with a non-zero value is refused by name. | **yes**. 0 is already off at 3.2.0. The PR adds the pinning test and the named refusal. |
| The harness train-mask rule (`oos_export.py:25-32`) | Stays exact with val = 0. | **yes**. `val_first_stamp == test_first_stamp`, and `split_record` writes both cuts (`partitions.py:303, :414-415, :521-523`). |
| Program ACCEPTANCE C2: the null fit's equality test | The in-memory fit's parameters and Pipeline are unchanged by the PR. | **yes**, if the fit call is untouched. A test pins `predict_proba` identical with and without `export_dir`. C2's text is the lead's. |
| Program ACCEPTANCE C2: "no fit of any cell before the tag" | A record that shows rux-ml fits. | **no**, not through PR-060. This belongs to PR-082 (lean 5: harness trial rows AND rux-ml's Optuna storage). |
| Program PR-083 / the freeze record | The rux-ml commit recorded at the freeze is the run's commit. | **unclear**. Yes only if PR-060 and the digests PR are both on `dev` before the tag. |
| rux-ml promote / program PR-084 (the "at most one trained" cell) | Re-fit a val-less in-memory trial on the same route and write a model file. | **unclear**. promote always passes `eval_set=val` (`promote.py:226-231`). This depends on Gate decision 6. The sha / `oos` copying works unchanged. |
| The M9 batch problems and their promote re-fits (program v0.3 PR-027's floor comparison) | Outputs, exports and sha256s byte-for-byte unchanged; schema v1's regression meaning kept. | **yes**, if the change stays below the dispatch, the regression bytes are pinned, and no config field is added. |
| `rux-ml runs` learning curve / any reader of `trial.value` (`cli/runs.py:142-156`) | Tells a test-told trial from a val-told one. | **only with a marker** (Gate decision 5). |
| Program ACCEPTANCE C5 / the page's (0, 1) check and M5 LL | `prediction` strictly inside (0, 1). | **partly**. Float32 can round P to exactly 0 or 1. The clip rule is the page's (lean 10), and PR-060 keeps Float32 and does not clip. |

### Path-tier checkpoint

**Tier-1** (proposed). Drift was found, but it does not change the premise. The stale items are a scope clarification (the trigger is always on), program texts the lead owns, and a promote gap. None reverses route (b) or lean (2) / (3), so there is no loop to design-planning. Every open question can be answered from the repo and the pinned xgboost 3.2.0 source, so there is no web round. The one library fact (0 is off; no evals means a raise) is already proven from source. Precedent: PR-059 (Tier-1 on a program track, Phases 2–4 light).

**Caveat.** The decision count below is ten. The program's smaller-PRs ruling says to split above a handful. Decision 1 does that split for the digests. Decisions 2–10 each have a lean and are one-line changes, so they are presented together. If the lead wants fewer, decision 6 (promote) is the one to carve out.

### Gate leans (one decision each)

1. **One rux-ml PR or two?** *Lean: two.* PR-060 is this route PR (export, val-less mode, pinning test; Tier-1). The digests go in a separate PR (provisionally PR-061, Tier-2: the cause of the c6 failures is uninvestigated, and the files are disjoint). Both merge to `dev` before the tag. If the digests PR is ready first, land it first so PR-060's desktop suite starts clean; otherwise PR-060's gate diffs against `dev`'s 2 c6 failures. *Alternative:* one PR for both (ROADMAP:31 as written).
2. **When does the in-memory export switch on?** *Lean: always, for every `[m9]` in-memory fit*, mirroring the batch path. A keyless `[m9]` fit is refused by name before the fit, and any keyless fixture is updated. *Alternatives:* an explicit opt-in; or silently skip when there are no keys (rejected in the lens: a missing export looks like a run that exported nothing).
3. **What switches on val-less mode?** *Lean: the configured `split_ratios['val'] == 0`* (an existing field, so no hash risk). It requires `[m9]` and `early_stopping_rounds ∈ {None, 0}`, each refused by name otherwise, before any fit. A val > 0 config whose val comes out empty keeps today's failure. *Alternative:* an empty val partition at run time (rejected: it would silently turn a mis-sized config test-told). *Open within the lean:* whether the refusal can sit at config load, before a trial row exists (lens 2), or only at the top of the fit (lens 1). Read where the root config sees both `[data]` and `[training]`.
4. **What does the export carry for each kind of model?** *Lean:* a binary classifier exports `predict_proba[:, 1]`. A regression in-memory `[m9]` fit exports `predict()`. More than two classes is refused by name before the fit. *Alternative:* classification only, with regression in-memory `[m9]` refused.
5. **How does a reader tell the new meanings apart?** *Lean, two parts.* (a) In val-less mode only, one fold_meta marker (`fold_extras['told_partition'] = 'test'`, the name a lean). Existing fold_meta stays byte-identical. (b) A metadata key in the file's `rux_ml.oos_export` naming the prediction as P(class 1), written **only** for probability exports. The regression file's bytes are pinned unchanged by a test. *Alternative for (b):* the writer unchanged, docstring only, with the meaning recorded only in fold_meta and the CHANGELOG. This is lens 1's stricter reading, since any writer change is a byte risk. The file then cannot be told apart on its own.
6. **Does PR-060 touch promote's in-memory re-fit?** The lenses disagree. Lens 1 leans to mirror the val-less branch there. Lens 2 leans to leave promote alone, because PR-084 is conditional. *Writer's lean: the middle option*, a by-name refusal of a val-less trial in promote (a few lines plus one test). The re-fit branch is then left to a rux-ml follow-up that PR-084 opens only if a cell passes. *Alternatives:* mirror the branch now (lens 1); leave promote untouched and note the gap (lens 2; promote then fails inside xgboost, not by name).
7. **Does val-less mode also cover the native branch (`use_native = True`)?** *Lean:* one shared val-less helper for both branches if it adds under ≈ 10 lines; otherwise refuse native by name. *Alternative:* refuse native by name outright (the cells don't use it).
8. **What does the pinning test assert?** *Lean: behaviour, on CPU, in the default suite.*
   - A made-up table where val loss rises, so `early_stopping_rounds = 2` or `3` provably stops early. This is the non-vacuity control; reuse the `tests/training/test_factory.py:132-142` setup.
   - With `early_stopping_rounds = 0` through rux-ml's own factory and a non-empty `eval_set`: `num_boosted_rounds() == n_estimators`, `best_iteration` absent, and `predict_proba` equal to the full-range prediction.
   - `xgboost.__version__` goes in the assert message, never asserted.

   *Optional:* the same check on the native adapter and the batch path, since all three gate on `if early_stopping_rounds:`.
9. **Who checks that export keys are unique?** *Lean:* PR-082's reader asserts uniqueness of (`stamp_ms`, `coin`[, `event_id`]) and row-count equality, and rux-ml's writer is unchanged. *Alternative:* a refusal on rux-ml's classification path only.
10. **AUC and a single-class partition: in scope?** *Lean: neither.* The harness computes AUC from the export. The page's "auc reported beside it" clause is the lead's to re-word. A single-class test partition would crash `_logloss`, which passes no `labels` (`metrics.py:79-81`); in val-less mode that partition becomes the told value. It is named here and checked by PR-082's dry count (per-cell label variety before the freeze). *Alternative:* add `labels=[0, 1]` to `_logloss`. That is new scope, and it would change `_logloss` for every caller.

**Budget.** To be re-stated at the gate with `agents × (68k + 3k × calls)`. ROADMAP:31's ≈ 0.3–0.4M covered a smaller scope.


### Gate Check (2026-10-14)

**Approved** by the program operator's "go" (2026-10-14 ≈ 09:50 CDT) on the program lead's go-list, which stated this State Assessment, the tier, the scope and the ten leans above. **Tier-1, cleared after Phase 1**: no premise change, every open question answered from this repo and the pinned xgboost 3.2.0 source, so Phases 2–4 are not run (precedent PR-059).

Ruled, each at its lean:

1. **Two rux-ml PRs.** This PR is the route (export, val-less mode, pinning test). The C10 digests go in a separate rux-ml PR. Both land on `dev` before the v0.5 freeze tag.
2. **The export is always on for an `[m9]` in-memory fit**, mirroring the batch path. A keyless `[m9]` in-memory fit is refused by name before the fit.
3. **The val-less mode keys on the configured `data.split_ratios.val == 0`** (no new config field). It requires `[m9]` and `training.early_stopping_rounds ∈ {None, 0}`, each refused by name, before any fit. The fit has no `eval_set`, early stopping is off, and the told value is the test score.
4. **The prediction:** `predict_proba[:, 1]` for a binary classifier, `predict()` for an in-memory regression; more than two classes is refused by name before the fit.
5. **The markers:** (a) `fold_meta` `told_partition = "test"` in val-less mode only; (b) a `prediction` key in the file's `rux_ml.oos_export` metadata for probability exports only. A regression export's bytes are pinned unchanged.
6. **promote refuses a val-less trial by name.** The val-less re-fit is a rux-ml follow-up, opened by program PR-084 only if a cell passes.
7. **The native branch** (`use_native = True`) takes the val-less mode through the shared helper if it costs ≤ ≈ 10 lines, else it is refused by name.
8. **The pinning test is behavioural**: `early_stopping_rounds = 0` is off at the installed xgboost (all rounds boosted, no `best_iteration`), with a firing control at 2, on CPU, the version in the message only.
9. Export key uniqueness is program PR-082's reader's check; the writer is unchanged.
10. AUC and the single-class logloss case are out of scope.

**Budget (re-stated):** one builder, capped at 70 tool calls.

---

## Scope

1. **The export on the in-memory path** (`cli/train.py:270-360`, below the batch dispatch at `:264-269`). When `cfg.m9` is set:
   - Before the fit, build `OosExportWriter(export_dir / OOS_EXPORT_FILENAME, key_columns = m9.row_key_columns, target_column, split_definition = fold_extras['split_definition'])`, mirroring `:488-497`, with construction errors raised as exit 2.
   - After the fit, write `val` (when non-empty), then `test`: keys = `splits[p].select(row_key_columns)`, `prediction` per Gate decision 4, and `realized` = the target cast to Float64, NaN kept.
   - Set `fold_extras['oos_export'] = writer.close()`.
2. **The val-less mode** (same branch), keyed on `split_ratios['val'] == 0` (decision 3). It is refused by name unless `[m9]` is set and `early_stopping_rounds ∈ {None, 0}`. The mode:
   - fits without `eval_set`;
   - skips the val score;
   - tells `fold_extras['oos']['score']` (refused by name if absent, i.e. `n_scored = 0`);
   - writes the marker (decision 5a).
3. **The pinning test** for `early_stopping_rounds = 0` (decision 8), in `tests/training/test_factory.py` beside `:132-142`.
4. **promote** (`registry/promote.py:212-243`), per decision 6.
5. **`runs/oos_export.py`**: the docstring widened from "per-day batch fit" to either path, with the probability meaning (`:3-4, :17-18`), plus decision 5(b) if taken. **Docs**: CHANGELOG `[Unreleased]` naming both meaning changes, the ARCHITECTURE export section if it names the batch path only, and a `docs/0.3/RESEARCH-BACKLOG.md` row.

**Not in scope:**
- the digests (decision 1);
- AUC and the single-class logloss case (decision 10);
- an export key-uniqueness check (decision 9);
- any change to the batch path, `_xgb_kwargs`, the regression export bytes or any M9 config;
- the program texts (C2, ROADMAP:27 / :31, page :232-234 / :240-242 / §12), which are the lead's.

## Dependencies

None open in rux-ml. PR-054 (the batch fit), PR-055 (the export) and PR-059 are merged in `dev` @ `09b9bd5`. Program side: PR-082's Phase 1 gate leans (1)–(3), approved 2026-10-14. Merge ordering against the digests PR and the v0.5 freeze tag: see decision 1 and the last constraint.

## Architecture section implemented

The OOS export section of `docs/ARCHITECTURE.md`, extended from the batch path to the in-memory path. The section was not read in this assessment; locate it at Phase 2.

## Verification criteria

- [ ] The new tests fail on `09b9bd5`'s `src` and pass after.
- [ ] **Export, end to end.** `rux-ml train` on a made-up classification `[m9]` table (CPU, no real data) writes `oos_rows.parquet` with `partition`, the row keys, `prediction` Float32 = `predict_proba[:, 1]` in [0, 1] and `realized` Float64 (NaN kept). Its sha256 is in fold_meta `oos_export` and in trial attr `oos_export_sha256`, and the file is uploaded. The exported p̂ equals the probabilities the logloss scored.
- [ ] **Val-less, end to end.** With `split_ratios.val = 0.00` and `early_stopping_rounds = 0`, the trial completes. A spy on `fit` sees no `eval_set`. The told value, `metrics.json` and stdout all equal fold_meta `oos.score`. The marker is present. The train mask `stamp + embargo < val_first_stamp` (= `test_first_stamp`) marks train exactly.
- [ ] **Named refusals, each before the fit:**
  - val = 0 with `early_stopping_rounds ∉ {None, 0}`;
  - val = 0 without `[m9]`;
  - an `[m9]` in-memory fit with no `row_key_columns` (decision 2);
  - a more-than-two-class classifier (decision 4);
  - `n_scored = 0` in val-less mode;
  - promote on a val-less trial (decision 6 as ruled).
- [ ] **Val > 0 unchanged.** `test_train_subcommand.py:396-414` still sees fit + `eval_set`. `predict_proba` is identical with and without `export_dir` (the C2 pin).
- [ ] **Non-`[m9]` unchanged.** `tests/golden/test_xgb_baseline.py` passes, and `_fit_and_score` still returns the 5-tuple.
- [ ] **M9 byte-for-byte.** A regression `[m9]` export's bytes / sha256 and every M9 config's `root_cfg_hash` are pinned unchanged. `test_m9_batches`' bit-identity test passes with its assertions unchanged.
- [ ] **The pinning test** (decision 8) passes in the default CPU suite, with its non-vacuity control.
- [ ] **Mac:** `uv run pytest`, `ruff check --no-cache .`, `ruff format --check src tests` and `basedpyright src/` are clean (synced `--all-extras`).
- [ ] **Desktop:**
  - `uv run --frozen python -c 'import xgboost'` reports 3.2.0 (= `uv.lock:1926`);
  - `uv run --frozen --all-extras pytest -n 8` is **diffed against `dev`**, not reported as a bare count (the 2 c6 failures are pre-existing unless the digests PR landed first);
  - `gpu`-marked tests run for `device = cuda`.
- [ ] Docs in the same commit: CHANGELOG, the `oos_export.py` docstring, ARCHITECTURE, and the RESEARCH-BACKLOG row.
- [ ] The program lead's merge to `dev`, before the v0.5 freeze tag.

## Research backing

- Program PR-082 §State Assessment and §Gate (Phase 1), approved 2026-10-14, leans (1)–(3) (`prs/PR-082-probe-runner-statistic-null-holm-freeze.md:376-383`).
- The lens file `pr082-p1-results.json` (lens 0: the in-memory path as the C2 reference; lens 1: the freeze records rux-ml's commit; lens 2: `early_stopping_rounds = 0` is off at 3.2.0, `training.py:189-190`, `callback.py:458`).
- This PR's two Phase 1 lenses, re-reading rux-ml `dev` @ `09b9bd5`.

## Notes

- **For the program lead, not this PR.** These texts need correcting before the tag:
  - C2's equality reference (`batches.py:228-252` → `cli/train.py:279-360` with the Pipeline);
  - ROADMAP:27 and :31 ("0 → None"; split into two rows; the estimate);
  - page :232-234 (the batch-path val reasoning), :240-242 and §12 items 1–2 (`early_stopping_rounds = 0`, pinned);
  - page :240 (AUC).
- **Row order.** In-memory exports keep the loaded frame's order (`data/splits.py:218-223`); batch exports go in day-file order. Consumers join by key, never by position. The train-mask rule is key-based, so this is harmless.
- **promote and GPU.** With early stopping off, `eval_set` should not change the trees, so a promote without `eval_set` would reproduce the trial's booster on CPU. This is library knowledge, not executed. GPU hist is near-deterministic only.
- **Float32 at the edges.** P(class 1) can round to exactly 0.0 or 1.0. Whether any row does is not measured. The clip rule is the page's (program lean 10).
- **Pre-existing, not this PR's:** the 2 c6 digest failures on the desktop. rux-ml `CLAUDE.md` still points at `docs/0.1`.
