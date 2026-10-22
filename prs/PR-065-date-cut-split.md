# PR-065: The `[m9]` time-block split with the cut given as a date

<!-- Number: next free at origin/dev d51e89f — PR-064 is the ExtMem eval-set stub (branch pr-064-extmem-eval-stub, 9f24fe6);
     no prs/ file, local or remote branch, or doc names PR-065 (checked 2026-10-22). Confirmed at the lead's gate. -->

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**Tier (lean): Tier-1.** The rule is fixed by the consumer (program v0.7 D63 as amended; the harness's
`probe/partition.py` `partition_at_cut`, program PR-102) and Phase 1 shows rux-ml already applies it once the cut is
an input (below); what is left is config shape — operator decisions, not external research. Phases 2–4 light.

## Research findings

### State Assessment (2026-10-22)

Run as a track of the rumpy program (program branch `track/ruxml-date-cut`, the operator's go ≈ 12:18 CDT,
item 10 of v0.7's Stage 1 go-list), under rux-ml's own procedure. Base: `origin/dev` = `d51e89f` (unchanged by the
fetch). Labels: **proven** (read in code or executed), **convention** (repo precedent), **best-guess**.

**Why this PR (the program's ask).** Program v0.7 fits two things through rux-ml under a calendar split — the feature
probe (program PR-110) and, conditionally, one training (program PR-111): train = rows whose label ends before
2025-11-26, test = rows stamped on or after it, in blocks of 28 days. rux-ml's split takes only fractions
(`data.split_ratios`), so landing the cut on a date means choosing `split_ratios.train` per cell so that the stamp at
sorted position `int(n · train)` is the date — fragile (float64 `int(n · frac)` rounding, ties on the boundary stamp,
re-chosen per cell and per row filter). The program asks for the cut as a DATE, row-identical to the harness's
`partition_at_cut` (harness `b440fb0`): train = valid & `stamp + h·DAY_MS < cut`; test = valid & `stamp ≥ cut`; ties
to test; an event's stamp = `(day + 1) · DAY_MS`.

**Current state:**
- **The fraction split is the only cut source (proven).** `config/data.py:65` `split_kind: Literal["random",
  "time_ordered", "symbol_holdout"]`; `split_ratios` (default 0.7 / 0.15 / 0.15); `split_embargo` (int, the time
  column's own units, `ge=0`); `train_prefix_frac`. No date- or stamp-valued cut field anywhere in `src/`.
- **An `[m9]` split already IS a stamp-predicate rule (proven).** Since PR-054, `make_splits` with `[m9]` set runs
  `rux_ml.data.partitions.plan_partitions` over the frame as one batch; `_plan_time` turns the ratios into two stamps
  (`val_first_stamp` = the stamp at sorted position `int(n·train)`, `test_first_stamp` = at `int(n·(train+val))`) and
  `PartitionPlan._time_folds` assigns: train = `s + emb < t_val`; val = `s ≥ t_val & s + emb < t_test`; test =
  `s ≥ t_test`; the rest `PURGED`. Boundary ties go to the later partition (`split_record` `cuts.boundary_ties =
  "later_partition"`). The per-day batch fit (`cli/train.py:616`) and promote's re-fit (`registry/promote.py:293`) use
  the same plan, so every consumer shares the membership by construction.
- **Val-less mode (proven; PR-060).** `split_ratios.val == 0` → `t_val == t_test`, val is empty by the inequality, no
  `eval_set`, early stopping must be off (`cli/train.py:249` `_val_less` refuses otherwise), `[m9]` only.
- **The embargo floor (proven; PR-042).** `root.py:199-210`: an `[m9]` `time_ordered` problem needs `m9.h_max_ms`
  and `data.split_embargo ≥ h_max_ms` ("a label reaches the split"); `_plan_time` refuses a missing embargo and
  "split embargo … purged every train row" when `int(n·train) ≥ 1` but no train row survives.
- **The record and its readers key on `time_ordered` (proven).** `split_record` writes `cuts.val_first_stamp /
  test_first_stamp`; the OOS export's train mask (`runs/oos_export.py:35-40`, the rule the harness recomputes) is
  `stamp + split_embargo < cuts.val_first_stamp`; `leakage.assert_regime_clean("time_ordered", …)` checks stamp
  separation within `h_max_ms`; `registry/scorer.py:204` uses seed 0 for `time_ordered`; `runs/learning_curve.py` and
  `cli/runs.py` read `split_kind` / `split_definition.kind`.
- **Hash stability for new optional fields (proven; convention from PR-042).** `root.py:82-86`
  `_HASH_OPTIONAL_FIELDS["data"]` drops listed fields from the hashed dump while `None`, so existing `data_cfg_hash` /
  `root_cfg_hash` pins do not move.
- **The harness mirror (proven, harness `b440fb0`).** `probe/partition.py` `partition` = rux-ml's val-less `[m9]`
  time-block partition, row-identical at `814cc15` (program PR-082: 352 checks, 0 mismatches, 32 / 32 refusals on a
  committed fixture, `tests/fixtures/pr082/`, generated by rux-ml's own `make_splits`); `partition_at_cut` (program
  PR-102) = the same rule with the cut given, no prefix, refusing `partition_embargo_purged_train` only when valid rows
  lie below the cut but the embargo keeps none — an empty side below the cut is returned empty, not refused.
- **Executed check (proven, 2026-10-22, synthetic data only).** rux-ml's existing `PartitionPlan.assign` at `d51e89f`,
  with `val_first_stamp = test_first_stamp = cut` and `embargo = h · 86,400,000`, against the harness's
  `partition_at_cut` loaded from the harness checkout: **400 sets, 0 mismatches** in train / test positions —
  607 rows tied on the cut (all test), 595 rows at exactly `cut − h·DAY` (all purged), h ∈ {0, 1, 7, 14}, day-stamped
  and intra-day stamps, ~15 % null targets; on the 31 sets the harness refuses, rux-ml's train is empty too.
  Script: the program track's scratchpad `datecut_identity.py` (not committed; the PR's tests replace it).

**Assumptions at PR draft time** (the program's ROADMAP row, 2026-10-21):
- rux-ml needs a new split mode with its own rule; the seam is `make_splits`.
- The cut is "given as a date".

**Stale assumptions:**
1. **No new rule is needed — only a new source for the cut.** The `[m9]` fold rule at a given cut is already
   row-identical to the harness (executed check above). The PR is a config field + `_plan_time` taking the cut from it
   + validators + the record + tests; `_time_folds`, the per-day batch path, promote, the OOS export mask and the
   leakage audit stay as they are.
2. **The seam is `partitions.plan_partitions`, not `make_splits`** — since PR-054 `make_splits`' `[m9]` branch
   delegates to it; the legacy in-memory `temporal_train_val_test_split` (non-`[m9]` problems) is not on v0.7's path.
3. **"A date" needs a unit contract.** rux-ml's time column has no declared unit (the embargo and `h_max_ms` are "the
   column's own units — epoch ms for an Int64 stamp"); a date becomes a stamp only under that contract
   (2025-11-26 → 1,764,115,200,000 ms at UTC midnight).

**New constraints** (from prior PRs — the prior-art audit, `git log` on `partitions.py`, `splits.py`,
`config/data.py`, `config/root.py`: PR-024, PR-041, PR-042 ×2, PR-043, PR-046, PR-054):
- Ties go to the later partition (PR-054); the strict `<` in `s + emb < cut` (a train row whose label ends exactly at
  the cut is purged) is the shared rule — row identity requires keeping it.
- `split_embargo ≥ m9.h_max_ms` stays (PR-042); for identity with the harness the cell's config must set
  `split_embargo = h · 86,400,000` exactly (h 7 → 604,800,000; h 14 → 1,209,600,000) — the harness's config to write.
- Identity also needs `m9.row_filter_non_null` = exactly the target column (the harness's `target_valid` is the
  target alone) and the set's time column = the harness's event stamp `(day + 1)·DAY_MS` — both the harness's side.
- A new optional `data` field joins `_HASH_OPTIONAL_FIELDS` (PR-042), with a test that existing hash pins hold.
- The plan-over-day-batches = plan-over-frame equivalence test (PR-054) extends to the new mode.
- Val-less refusals (PR-060) are unchanged and apply.
- No change to `docs/0.3/DESIGN-log.md` decisions (nothing there on a date cut; no drift). Staleness threshold 60 days
  (`docs/CONSTRAINTS.md`): this assessment is good to 2026-12-21.

**Performance budget (the program's Build to Budget, stated before the build).** The per-row path (`assign`) is
unchanged — vectorised NumPy, no per-row Python. `_plan_time` in date mode drops the sorted-position lookup and keeps
one key-column pass (the "rows below the cut" check for the refusal). Budget: plan + assign ≤ the fraction mode's on
the same set, and ≤ 1 s for a 1,000,000-row synthetic set on the Mac (v0.7's largest cell is ≈ 27–28 k rows);
closing measurement in the PR, and on the first real cell set at program PR-110's sample.

**Phase 1 gate — decisions for the operator (each a lean with basis and confidence):**
1. **Shape — an optional field on `time_ordered`, not a new `split_kind`** (lean). Basis: every `time_ordered` branch
   (embargo floor, leakage audit, scorer seed, OOS mask, learning curve, runs listing) then works unchanged; a new kind
   re-opens ~8 sites. Confidence high (proven by reading + the executed check).
2. **Config type — a TOML date, `data.split_cut_date = 2025-11-26`** (lean), converted to UTC-midnight epoch ms for an
   Int64 epoch-ms time column; the split record writes the date and the stamp. Alternative: an int `split_cut_ms` like
   `split_embargo`. Basis: the operator's rule is a date; a hand-typed 13-digit stamp is the error-prone form. Confidence
   medium (rux-ml cannot check the column's unit — the same contract `h_max_ms` already relies on).
3. **Val-less only** (lean): the date mode requires `split_ratios.val == 0` and refuses `val > 0` by name; a val date
   is added when a consumer needs one. Basis: the harness partition and v0.7's probe are val-less. Confidence medium —
   **open:** program PR-111's training must also be val-less (early stopping off) for this to cover it.
4. **The ratios in date mode** (lean): still validated (sum to 1), not used for the cut; the record says
   `cuts.source = "date"`. Confidence medium-high.
5. **Refusals** (lean): the purged-train refusal exactly as today ("split embargo … purged every train row") when rows
   lie below the cut but none survive; plus an empty-train refusal by name when no row lies below the cut (the harness
   returns that case empty for its dry count; a fit cannot run on it). Confidence medium-high.
6. **Scope** (lean): `[m9]` problems only; refused with a non-`time_ordered` kind, without `[m9]`, and with
   `train_prefix_frac` (the harness's `partition_at_cut` has no prefix; a later consumer lifts it with its own test).
   Confidence medium.
7. **Row identity with the harness — two sides** (lean): rux-ml's tests (date mode = fraction mode where the cut is the
   fraction-chosen `test_first_stamp`; batches = frame; ties, the `cut − h·DAY` row, every refusal; hash pins) plus a
   committed fixture in PR-082's pattern — made-up parquet tables + rux-ml's date-mode output at the merged commit as
   `partition_at_cut_expected.json` — for a harness test to replay through `partition_at_cut`. The harness test is a
   harness change: **its owner is the program lead's call** (lean: program PR-110's build, which needs it first).
   Confidence high on the method (precedent), open on the owner.

**Size (re-floored):** Tier-1; Phases 2–4 light in the main loop ≈ 0.05–0.1M; build one agent ≈ 0.2M (68k + 3k × ~35
calls + a ~30k pack); review ≈ 0.1M — **≈ 0.35–0.5M, $0, no desktop run**; plus ≈ 0.1M for the harness-side fixture
replay in whichever harness PR owns it.

**Exit:** stale assumptions flagged, new constraints documented; the premise holds (smaller than drafted). HALT at the
Phase 1 gate.


### Research Questions (Phase 2, light — 2026-10-22)

Run as program track `track/ruxml-pr065-s2` after the operator's go (≈ 12:39 CDT 2026-10-22, the program's Stage 1
combined gate: every Phase 1 lean above approved as written). Tier-1 in rux-ml's sense; no drift since Phase 1
(`origin/dev` still `d51e89f`), so Phases 2 – 4 run light in the main loop with no external sources — the rule is fixed
by its consumer (the harness's `partition_at_cut`), not by a library choice.

**Must-answer:**
1. Shape, type, val, ratios, refusals, scope, identity — **answered by the gate** (leans 1 – 7, approved values).
2. A TOML *datetime* (or an int) given where the date belongs: truncate or refuse? — success: no input moves the cut
   silently.
3. Where the refusals live so that no route reaches the plan unchecked — success: every route that builds a config
   re-runs them, or the plan refuses too.
4. Does anything change for a fraction-mode config? — success: `data_cfg_hash` / `root_cfg_hash` pins and the
   fraction-mode split record unchanged.

**Dependencies:** none (2 – 4 independent of each other; all depend on the gate's leans).

**Explicitly excluded** (as Phase 1): a val date; a prefix with a date cut; non-`[m9]` problems and the legacy
in-memory splitter; a declared unit for the time column; the harness-side replay test (program PR-110's build, the
lead's call).

### Research (Phase 3, light — executed in the main loop, 2026-10-22)

- **Q2 (proven, executed in this worktree's venv, pydantic 2):** a lax `datetime.date` field accepts and *truncates*
  more than a date. `tomllib` reads `2025-11-26` as a `date`, but `2025-11-26T00:00:00-06:00` as an offset `datetime`,
  which pydantic turns into the date 2025-11-26 → the UTC-midnight cut, six hours before the Central midnight the
  author wrote; the ints `1764115200000` and `1764115200` are also accepted (as Unix ms / s) → 2025-11-26. A datetime
  with a non-zero time is refused (`date_from_datetime_inexact`). **Decision:** the field takes only a `date` (TOML
  date) or an ISO `YYYY-MM-DD` string (the env / dot-path override routes deliver strings); a `datetime`, an int or any
  other string is refused by name. This implements gate 2 ("a TOML date"); no new value.
- **Q3 (proven, read):** config overlays reach the plan through `RuxMLConfig.model_validate` (`registry/promote.py:167`,
  `tuning/objective.py:140`), which re-runs the root validators; only a direct attribute assignment on a built config
  skips the cross-field checks. **Decision:** the refusals are root validators at load (gates 3, 6) and the plan
  re-checks `val == 0` in date mode, so a mutated config cannot get a silent val-less split.
- **Q4 (proven by construction, pinned by test):** the field joins `_HASH_OPTIONAL_FIELDS["data"]` (PR-042's rule); the
  split record gains its date keys (`cuts.source`, `cuts.cut_date`, `cuts.cut_stamp`) only in date mode, so a
  fraction-mode record and every existing hash pin are untouched.
- **Found while reading (proven):** `m9.h_max_ms` is `gt=0` and `split_embargo >= h_max_ms` for every `[m9]`
  time-block problem, so rux-ml runs h ≥ 1 ms only; the harness's `partition_at_cut` also takes `h_days = 0`. v0.7's
  cells are h 7 / 14, so nothing is lost; the fixture covers h ∈ {1, 7, 14}.

### Synthesis (Phase 4, light)

**Outcome:** Confirm — the build is the approved leans; Phase 3 adds only input-type refusals inside gate 2's contract.

**Changes to this PR from research:** the date field refuses a `datetime`, an int and a non-ISO string by name (Q2);
the plan re-checks val-less in date mode (Q3).

**Changes to ARCHITECTURE.md:** the split section gains the date mode (in the build commit).

**Changes to CONSTRAINTS.md:** none.

**New PRs that must come first:** none.

**Research-backed details now locked:** `data.split_cut_date` (a TOML date → `(date − 1970-01-01).days ·
86,400,000` ms; 2025-11-26 → 1,764,115,200,000); the record writes the date and the stamp with `cuts.source = "date"`;
val-less only; `time_ordered` + `[m9]` only; no `train_prefix_frac`; the purged-train refusal unchanged plus an
empty-train refusal by name; the ratios sum-checked and unused for the cut.

### Gate Check (Phase 5)

- Premise still valid: ✓ (no drift; `origin/dev` = `d51e89f`)
- No prerequisite PRs surfaced: ✓
- User approved updated spec: ✓ (2026-10-22 ≈ 12:39 CDT — the operator's "go" on the rumpy program's Stage 1 combined
  gate approves every Phase 1 lean above as written; Phase 3's input-type refusals sit inside lean 2)
- Implementation cleared

---

## Scope

- `data.split_cut_date` (per gate 2) on `DataConfig`, in `_HASH_OPTIONAL_FIELDS["data"]`; validators per gates 3, 5, 6.
- `partitions._plan_time`: with the field set, `val_first_stamp = test_first_stamp = ` the cut stamp; the refusals.
- `split_record` / `split_definition`: the cut's source and date.
- Tests (gate 7, rux-ml side) and the committed fixture for the harness replay.
- Docs: `docs/ARCHITECTURE.md` (the split section, ~lines 232–255), `CHANGELOG.md` `[Unreleased]`, `configs/` example
  if one is added.

## Dependencies

None in rux-ml (base `d51e89f`). Consumers: program PR-110 (the probe) and PR-111 (the training), which land after it.

## Architecture section implemented

`docs/ARCHITECTURE.md` — the one-off split / `[m9]` partition rule (`time_ordered`, `split_embargo`, the partition plan).

## Verification criteria

- [x] Date mode = fraction mode, row for row, wherever the date equals the fraction-chosen `test_first_stamp`
  (`test_date_mode_equals_fraction_mode_at_the_fraction_chosen_cut`: train 0.3 / 0.5 / 0.7 / 0.85 × h 1 / 7 — plan,
  assign, tally, `make_splits` partitions, and the record minus its three date keys).
- [x] Rows tied on the cut are test; a row at `cut − split_embargo` is purged; null targets are in no partition
  (`test_ties_go_to_test_the_cut_minus_h_row_is_purged_and_nulls_are_nowhere`, h 1 / 7 / 14).
- [x] The per-day batch plan equals the frame plan in date mode (60 day batches; plan, tally, record, and
  `split_definition` of the in-memory split).
- [x] Each refusal fires by name — val > 0 (at load and, for a config mutated after load, in the plan), kind
  `random` / `symbol_holdout`, no `[m9]`, `train_prefix_frac`, the purged-train refusal kept, the new empty-train
  refusal (no row below the cut; an empty frame), and a datetime / int / non-ISO string for the date; the ratios still
  sum-checked. Existing `data_cfg_hash` / `root_cfg_hash` pins unchanged (the PR-040 pins re-asserted in the new module;
  `tests/config/` all green).
- [ ] The committed fixture reproduces through the harness's `partition_at_cut` — **the harness-side test is its
  owner's PR** (the lead's lean: program PR-110's build). A scratch replay (not committed; the harness checkout read
  only, `probe/partition.py` loaded by path as in Phase 1) agreed on all **81 cases, 0 mismatches**: 47 fitted (20 of
  them with an empty test), 22 purged-train refusals = the harness's `partition_embargo_purged_train`, 12 empty-train
  refusals = the harness's empty train.
- [x] Performance budget met and measured (below).

## Build status (2026-10-22 ≈ 12:59 CDT, branch `track/date-cut-split`)

Built at the approved leans, tests first: the new module `tests/data/test_m9_date_cut.py` ran **37 failed** before the
build (the field did not exist) and **37 passed** after.

- `src/rux_ml/config/data.py`: `split_cut_date: date | None` (before-validator: a `date` or an ISO `YYYY-MM-DD`
  string only), `split_cut_stamp` (property, not a field — the dump and the hash carry only the date),
  `refuse_val_with_cut()`.
- `src/rux_ml/config/root.py`: `split_cut_date` in `_HASH_OPTIONAL_FIELDS["data"]`; the root validator
  `_validate_split_cut_date` (kind, `[m9]`, prefix, val).
- `src/rux_ml/data/partitions.py`: `_plan_time` → `_plan_cut` in date mode (val = test first stamp = the cut; the rows
  below the cut returned as the purged-train check's n; the empty-train refusal); `PartitionPlan.cut_date`;
  `split_record`'s `cuts` gains `source` / `cut_date` / `cut_stamp` in date mode only. `_time_folds`, the per-day batch
  path, promote, the OOS export mask and the leakage audit unchanged.
- `tests/data/pr065_fixture.py` + `tests/fixtures/pr065/` (six made-up tables in 2020 — event stamps, intra-day, ties
  on the cut and on `cut − h·DAY ± 1`, n 1 / 2 / 3 — y null-only, no NaN, since pyarrow's `is_valid` counts NaN valid;
  81 cases × h 1 / 7 / 14; `_contract` key states the replay rule). Regenerate: `uv run python -m
  tests.data.pr065_fixture`.
- `docs/ARCHITECTURE.md`: the date-given cut bullet after the val-less mode.

**Suite (the Mac, this worktree, `--all-extras`):** 849 collected → **844 passed, 5 skipped, 0 failed** (in
selections: config / data / features / solving / _internal 447 + 2 s; cli 168 + 1 s; registry / runs / training /
tuning / integration / container / golden / smoke 229 + 2 s). Without the extras the 17 catboost / lightgbm tests fail
on import only (env, not this PR). `ruff check --no-cache .` clean; `ruff format --check` flags only
`scripts/calibrate_pr025.py` (untouched, as on `dev`); `basedpyright src/` 0 errors with the extras.

**CI:** rux-ml has **no GitHub Actions workflow** (`gh api repos/rux-eth/rux-ml/actions/workflows` → `total_count 0`;
no `.github/`), so there is no CI run to watch; the local suite above is the record (as PR-063's Mac + desktop runs).

**Performance (the budget stated at Phase 1; Mac, 1,000,000 synthetic rows over 200 days, h 7, median of 7):** plan +
assign, whole frame — fraction 0.010 s, **date 0.009 s**; 200 day batches — fraction 0.236 s, **date 0.231 s**. Met:
date ≤ fraction and ≤ 1 s. (Script in the program track's scratch folder, not committed.) The first real-cell
measurement stays with program PR-110's sample.

**Found (proven, flag for the consumers):** a date cut is val-less, so it runs on the in-memory `[m9]` fit only — the
per-day batch fit (an `[m9]` XGBoost *regression* target, `uses_batch_fit`) refuses an empty val (`fit_batches`:
"empty partition: train … rows, val 0 rows"). **Program PR-111 is covered only if its training is val-less (early
stopping off) AND on the in-memory route** (not an `[m9]` XGBoost regression target). Flag carried to the lead.

**Held for the lead (the track's standing rule: no CHANGELOG edit), the `[Unreleased]` entry to add at the merge:**

> ### Added
> - `data.split_cut_date` (PR-065): the `[m9]` time-block cut given as a date — its UTC-midnight epoch-ms stamp
>   replaces the fraction-chosen cut of a val-less `time_ordered` split (train `t + split_embargo < cut`, test
>   `t >= cut`); refused with `split_ratios.val > 0`, another split kind, no `[m9]`, or `train_prefix_frac`, and when
>   no row lies below the cut. The split record adds `cuts.source = "date"`, `cuts.cut_date`, `cuts.cut_stamp`.
>   Hash-neutral while unset.

## Research backing

This file's State Assessment; program `prs/PR-102-stage-a-dry-count-new-extract.md` (gap A), program
`docs/0.7/ROADMAP.md` (the rux-ml date-cut row), harness `b440fb0` `src/rumpy_harness/probe/partition.py` and
`tests/test_pr102_stage_a_calendar.py`, `tests/test_pr082_mirror_partition.py`.

## Notes

- Out of scope: a val date; a prefix with a date cut; the legacy non-`[m9]` `temporal_train_val_test_split`; any
  harness change.
- PR-064 (ExtMem eval-set) is independent: a val-less fit passes no eval set.
