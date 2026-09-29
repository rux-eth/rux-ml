# PR-058: the M9 honesty records carry program PR-027 A7's forms

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**Tier: Tier-1 proposed (operator's call).** R-14 track, on the rux-capital program lead's instruction of 2026-09-29: program PR-027 P2 found that rux-ml `48ffd4e` does not implement two of A7's honesty forms, so P2 computed them outside rux-ml from the exported predictions ("Finding for P3", program PR-027 "P2 PREPARED"). This PR makes rux-ml's records carry them so that P3 can read them from rux-ml. The research is program PR-027 §Findings Phase 3 **Q5** (a) / (b) / (c) with R2-c's measured moments, and Phase 4 amendment **A7**, operator-approved 2026-09-28 ≈ 11:02 CDT ("go on all": A1–A12, program PR-027 :785). The merge is operator-gated.

## Research findings

### State Assessment (2026-09-29)

Baselines: rux-ml `dev` @ `48ffd4e` (PR-057 merged). The default suite at `48ffd4e` has 731 tests. In a fresh worktree synced without the extras, 716 pass and the 15 LightGBM / CatBoost smoke and conformance tests fail on `ModuleNotFoundError`. That is environment only: `uv sync --all-extras` restores them.

**A7's wording** (program `prs/PR-027-stage3-learnability-stage4-selection.md` :690-691): *"Honesty per target: fill is gated on Brier, with a signed calibration bias per (p, Q, h) bucket reported (walk-forward drift of 3 pp measured). Markout is tested as the EV deducts it, `max(0, m̂) < r`. Q5c: keep the signed definition, no re-signing (L6 stands)."* Its leans:
- Q5 (a), :519-520: "a signed calibration bias `mean(pf̂) − mean(pf)` per (p, Q, h) bucket is reported beside it, not gated — gating it would need a new signed key".
- Q5 (b), :521-524: "test what the EV deducts — the prediction clipped at zero (`max(0, m̂)`, `shell/ev.py:17-18`) against the signed realized markout … The model still trains on the signed label."

**Current state (read):**
- `training/honesty.py::signed_error_honesty` (PR-044) tests the prediction as fitted. Criterion (i) is `mean(p >= r)` and criterion (ii) is `(mean(p) − mean(r)) / |mean(r)|`. There is no clipped form. Its docstring flagged the (ii) aggregation for the program lead, and A7 ruled it: "keep the signed definition".
- `cli/train.py::_oos_record` (in memory) and `_oos_record_batches` (PR-054's per-day batch path, which every M9 regression fit takes) run the test on `[m9] signed_error_honesty` problems (markout and walk). The fill (`signed_error_honesty = false`) records Brier and nothing else.
- PR-055's export already carries per test row the keys `p_bp` / `q_usd` / `h_ms`, the prediction and the realized value. Program P2's `p2_fit.py::export_stats` (program `3c38e8a`, ≈ :90-119) computed A7's forms from that export: `pc = p.clip(lower_bound=0)`, `mean(pc >= r)`, `(mean(pc) − mean(r)) / |mean(r)|`; fill `mean_pred − mean_realized` overall and per bucket, plus the max-|bias| bucket.

**Drift:** none against A7. The build forced these decisions (leans are flagged in §Notes):
1. **Which form decides the verdict.** A7 rules the markout test's form (:690-691), so `m9_markout_bp` decides on `clipped_at_zero`. A7 writes the event of criterion (i) out explicitly. For criterion (ii), this PR applies the same named form, reading "tested as the EV deducts it" as covering the whole test (**lean**). Both forms are always reported.
2. **The walk:** A7 names no clip for it, so the walk keeps PR-044's signed form, reported beside the clipped one (**lean**).
3. **The bucket:** one exact (`p_bp`, `q_usd`, `h_ms`) cell as the rows carry it, with no binning. This is how P2 grouped the rows (20 buckets). It is set in the TOML, not in code (**lean**).
4. **No silent change:** `honesty_verdict_form` defaults to `signed`, so a config that does not set it keeps every verdict value. Only the markout config changes, explicitly, citing A7.
5. **No row:** the calibration record reports `None`, never a number, and does not refuse, because it is reported and never gated. The honesty test refuses as before.
6. **Not a validator:** "bucket columns ⊆ row keys" is not enforced at config load. Enforcing it turned PR-055's row-keys refusal (exit 2, before the fit) into a config error instead. A config test pins the relation for the repo's problems.

### Phases 2–4 (light)

- **No web round.** The definitions are the program's: A7, and the harness EV's `max(0, ·)` as program PR-027 cites it. P2's `export_stats` is the reference implementation.
- **Cross-check against the reference.** P2's `export_stats` (program `3c38e8a`) and this PR's functions were run on one synthetic 400k-row export in P2's schema (`partition` Categorical, the three keys, `prediction` f32, `realized` f64, 1 % of rows without a realized value).
  - Markout: the no-under-deduction fraction is exactly equal in both forms (0.5075713576256492 signed, 0.5122597914194086 clipped). The over-deduction agrees to within 1.3e-15 in both forms.
  - Fill: the overall bias, all 20 buckets and the max bucket (key and bias) agree to within 1.7e-15, with the accumulator fed in 37 uneven batches.
- **Budget (Mac, measured).**
  - The calibration accumulator took 0.22 s in all for 19,968,000 rows in 156 batches (1.4 ms per batch), and the record 1.5 ms. Memory is per-bucket sums only: O(batches × buckets).
  - The two-form honesty test took 0.12 s on 20M rows. Its one extra transient is the clipped float64 array of the test rows, ≈ 8 B/row.
  - No extra column is read: the fill's bucket columns are features and row keys already in `predict_fold`'s read.

### Synthesis / Gate

**Outcome: Confirm (A7 as approved; the leans above).** **Gate:** track-autonomous under program PR-027's approved build gate; the operator's approval is pending at merge.

---

## Scope

- **`training/honesty.py`**:
  - `signed_error_honesty(..., verdict_form=)` scores two forms of the prediction and reports both under `forms`: `signed` and `clipped_at_zero`. Each form carries `no_underdeduct_frac`, `overdeduct_rel`, `mean_pred_bp`, `passes_*` and `honest`. The top-level verdict keys (PR-044's names) carry the named form's values, and `verdict_form` names it. The record adds `mean_realized_bp`. MAE and the mean signed error stay the fitted prediction's.
  - New `CalibrationBias`: `add(bucket_frame, pred, realized)` per batch and `record()`, which gives `n`, `n_realized_missing`, `mean_pred`, `mean_realized`, `signed_bias`, `bucket_columns`, `n_buckets`, `buckets` (sorted) and `max_abs_bucket_bias`.
- **`config/m9.py`**:
  - New `HonestyForm`, and new `[m9]` keys `honesty_verdict_form` (default `signed`), `calibration_bias` (default false) and `calibration_bucket_columns`.
  - It refuses a non-signed form without the test, bucket columns without `calibration_bias`, and duplicate or empty bucket names.
- **`cli/train.py`**:
  - `_oos_record` and `_oos_record_batches` run the verdict in the configured form and add `oos.calibration`.
  - The batch path feeds `CalibrationBias` from `predict_fold`'s sink beside the PR-055 export (`_test_sinks`). Each gets its own columns of the one read, so the export's rows are unchanged.
- **Problem configs**:
  - `m9_fill_frac.toml`: `calibration_bias = true` over `["p_bp", "q_usd", "h_ms"]`.
  - `m9_markout_bp.toml`: `honesty_verdict_form = "clipped_at_zero"`.
  - `m9_walk_bp.toml`: `honesty_verdict_form = "signed"`, set explicitly.
- **Tests**:
  - `tests/training/test_honesty.py`: 9 new tests, hand-computed. They cover both forms, the verdict following the named form, an all-negative prediction, a zero mean realized cost, an unknown form, the bias over two batches, rows and buckets without a realized value, no bucket columns, no row, and refusals.
  - `tests/config/test_m9_problem_configs.py`: 4 new tests (the repo problems' A7 forms, and the config refusals).
  - New `tests/cli/test_m9_a7_honesty_forms.py`: 4 tests. Fill and markout fits run end to end, with the record held to an independent computation from the fit's own export. The in-memory `_oos_record` is checked for both.
- **Docs**: `docs/ARCHITECTURE.md` (§ M9 honesty metric: the A7 forms), `CHANGELOG.md`, `docs/0.3/RESEARCH-BACKLOG.md`.

## Dependencies

None open: PR-044 (the honesty test), PR-054 (the batch path) and PR-055 (the export and its sink) are merged in `dev` @ `48ffd4e`.

## Architecture section implemented

`docs/ARCHITECTURE.md` § "M9 honesty metric and the signed keys": the new bullet "The A7 honesty forms".

## Verification criteria

- [x] Every honesty record reports both forms and names the verdict's form. The default (`signed`) keeps PR-044's verdict values (hand-computed unit test).
- [x] The markout verdict is the clipped form, per A7 (config test, plus an end-to-end fit whose record equals `mean(max(0, p) >= r)` recomputed from its own export).
- [x] The fill record carries the signed calibration bias overall and per (`p_bp`, `q_usd`, `h_ms`) bucket, with the max-|bias| bucket. It equals an independent group-by over the fit's exported test rows (end to end, batch path) and hand values (in-memory path).
- [x] Edge cases, all hand-computed: an all-negative prediction (clip to 0; a favourable mean flips the over-deduction to +1.0), a zero mean realized cost (`None`, never a pass, in both forms), rows and buckets without a realized value (dropped, counted, absent), no bucket columns (overall only), and no row (`None`).
- [x] PR-055's export is unchanged with the calibration sink on (`tests/cli/test_m9_oos_export.py` and `test_m9_promote_batches.py` pass unmodified).
- [x] The new tests failed before the code (8 failed + 1 collection error: `CalibrationBias` did not exist) and pass after.
- [x] Mac: `uv run pytest` gives **748 passed, 3 skipped, 0 failed**. `ruff check`, `ruff format --check src tests` and `basedpyright src/` are clean.
- [x] Cross-checked against program P2's reference implementation on a synthetic export (§Research findings).
- [ ] Desktop: `make test-golden`. It is unaffected by construction (`golden_v1` has no `[m9]` layer, so its hash and record are untouched), but it is to be run at the next desktop slot.
- [ ] Desktop: the same statistics on P2's real exports (steps in §Notes), after program PR-025's chain releases the heavy lock.
- [ ] The operator's approval of the merge.

## Research backing

Program PR-027 §Findings Phase 3 Q5 (a)–(c) with R2-c's moments (:640-648), Phase 4 A7 (:690-691), the build-gate approval (:785), and P2's finding and records (:813, :819). The reference implementation is program `3c38e8a` `measurements/pr027/p2/p2_fit.py`. ACCEPTANCE C9 asks for "the honesty verdict per target per regime against the floor's". This PR records the verdict's form beside the verdict. The floor comparison stays with the program (PR-026).

## Notes

- **Leans for the program lead (the operator's through the lead):**
  - (1) The verdict's form applies to both criteria. A7 writes out only the event of criterion (i).
  - (2) The walk stays `signed`.
  - (3) A bucket is the exact (p, Q, h) cell, with no binning.
  - (4) The calibration record reports `None` rather than refusing when no row has a realized value.

  Each is one TOML value or one line of code to change.
- **Hash:** the three M9 problems' `root_cfg_hash` change, because the new fields are in the hashed `[m9]` layer (as with PR-054's `row_key_columns`). Non-M9 hashes are unchanged. P2's 18 fits at `48ffd4e` carry the old hash and records without A7's forms. For P3 to read the forms from rux-ml, those fits are re-run at this PR's code (P2 measured 22.0 min of fit wall), or P3 keeps P2's export-derived numbers. That is the lead's call.
- **Desktop steps (later; nothing ran on the desktop).** Do these after program PR-025's chain releases `~/rumpy-heavy.lock`. Take the lock by the mkdir protocol only, never `flock`, and release it after.
  1. `ssh rux@100.90.42.41` (or `rux@10.0.0.238`).
  2. `cd ~/projects/rux-ml && git fetch origin && git worktree add ~/projects/rux-ml-pr058 origin/pr-058/a7-honesty-forms && cd ~/projects/rux-ml-pr058 && uv sync --all-extras`.
  3. `make test-golden`. The expected result is a pass with no fixture change.
  4. For the time_block f1.0 markout fit and the fill fit, take the `oos_rows.parquet` each P2 record names. Read the test rows with a realized value (`partition == "test"`, `realized` neither null nor NaN):
     - markout: `signed_error_honesty(pred, real, …, verdict_form="clipped_at_zero")["forms"]`;
     - fill: `CalibrationBias(["p_bp", "q_usd", "h_ms"])`, then `.add(frame, pred, real)`, then `.record()`.

     The thresholds do not enter the compared statistics.
  5. Compare with `~/pr027-p2/summary.md`:
     - markout: nud 0.5183 signed / 0.5213 clipped; over-deduction +0.068 / +0.208;
     - fill: bias +0.0088; max bucket +0.0217 at (100, 3000, 14400000) over 20 buckets.
- **Known limitation, inherited:** the in-memory record scores `model.predict`, as PR-044's honesty did. Every M9 problem is an XGBoost regression, so it takes the batch path. An `[m9]` classification fit, which no problem is, would get labels, not probabilities.
- **Pre-existing, not this PR's:** `ruff format --check .` flags `scripts/calibrate_pr025.py` on `dev`. It is untouched here.
