# PR-053: rux-ml computes its logical data hash one file at a time

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**Tier: Tier-1 proposed (operator's call).** R-14 track, program PR-027 build step **R1** on the rux-capital program lead's instruction of 2026-09-28. The research is program PR-027 amendment **A2**, operator-approved 2026-09-28 ("go on all", the Phase 4 → 5 gate). The merge is operator-gated.

## Research findings

### State Assessment (2026-09-28)

Baselines: rux-ml `dev` @ `9e3f0c3`, Polars 1.40.1, NumPy 2.4.5. Program PR-027 §Findings, Phase 3 Round 2b (`measurements/pr027/r2b/records/q3_hash_stream.json`, desktop).

- **The finding (proven on the real set, program side).** `_logical_hash` (`data/versioning.py`) hands Polars every file in one scan, hashes each row as `struct(sorted cols).hash(seed=0)`, sorts with the streaming engine, and SHA-256s the sorted u64 bytes.
  - On the 156-day fill view (95,361,520 rows, 156 files) this peaked at **18.0 GiB**, about 90 % of a fit's 20 GiB scope before any fit.
  - It leaves about 400 B per row resident. `data_hashes` runs at `cli/train.py:333`, before the fit.
  - The same digest recomputed file by file, concatenated and sorted, was **equal** at **7.3 GiB** (51 s vs 10 s).
- **Callers (read).** Every consumer goes through `compute_data_hash` → `_logical_hash`, so changing that one function changes every path:
  - `runs.provenance.data_hashes` (train `cli/train.py:333`, tune `tuning/objective.py:313`, promote's re-fit check `registry/promote.py:191`);
  - `data.bridge.training_set_sidecar` (`data bridge`);
  - `cli/data.py` (`data hash`);
  - `versioning.snapshot`.
- **Drift found while probing (Mac, Polars 1.40.1): a naive per-file scan is NOT equivalent.** One scan over every file checks each later file against the first file's schema.
  - It **refuses** a missing column (`ColumnNotFoundError`), an extra column, and any dtype difference (`SchemaError`). The dtype cases probed: Int32/Int64, Float32/Float64, String/Categorical, Null/Int64, time unit, another time zone, Enum categories, Enum/Categorical, List inner, struct inner and extra fields, String/Binary, Date/Datetime, Decimal, Int64/UInt64, Int64/Float64.
  - It **accepts** a different column order, a different struct field order, and a tz-naive datetime under a tz-aware first file.
  - Hashing each file on its own would accept every one of these silently.
  - **Scanning each file with the first file's schema (`scan_parquet(f, schema=first)`) reproduces all 19 probed outcomes exactly**: the same digest where one scan accepts, the same exception class where it refuses. So the per-file path hands this check to Polars rather than re-implementing it.
- **Categorical row hashes are string-based, not code-based** (probed), so the digest does not depend on the order categories were first seen in the process.
- **Hive.** A list of files (or one file) disables Polars' hive inference, so hive-style directories add no partition columns in either path (tested).
- **Recorded values.** `tests/golden/fixtures/golden_v1/manifest.json`'s `data_logical_hash` (`98f8c686…`) is reproduced by the unchanged code, and by the new code.

### Synthesis / Gate

**Outcome: Confirm, with one decision the probes forced (the `schema=` scan).** No sidecar, trial or bridge value changes. The digest is bit-identical, which is proven on the fixtures here and on the real 156-day view by the program's probe. The same proof on PR-024's final set is the deferred script below. Gate: track-autonomous; operator approval pending at merge.

---

## Scope

- `rux_ml.data.versioning._logical_hash`, per file:
  1. Read the first file's schema.
  2. Take each file's row count from its footer.
  3. Preallocate one u64 array.
  4. Hash each file on its own, scanned with the first file's schema: only its hash column is collected, and a schema refusal names the file.
  5. Sort the array in place and SHA-256 it through a `memoryview`, with no `tobytes()` copy.
  6. Free the array before returning.

  A file whose hashed row count differs from its footer is refused.
- `_logical_hash_whole_source`: the pre-PR-053 body, verbatim, kept as the reference for the equality tests and the deferred real-set check. It is on no runtime path.
- `scripts/check_logical_hash_per_file.py`: the deferred real-set check (program PR-027 R1's proof on PR-024's final set). Not run here.
- Tests:
  - `tests/data/test_logical_hash_per_file.py`: equality in 10 layouts over an all-dtype frame; 12 later-file schema variants; layout invariance; the duplicate multiset; the golden and pinned digests; the one-file-at-a-time memory bound; array release.
  - `tests/cli/test_data_hash_per_file_c6.py`: `c6_set` `fill/` and `walk/`, through `data_hashes` and `training_set_sidecar`, pinned.
  - `tests/data/test_check_logical_hash_script.py`: the deferred script end to end on a small harness-shaped set (exit 0 / 1 / 2).
- Docs: `docs/ARCHITECTURE.md` (§Reproducibility, data), `CHANGELOG.md`, `docs/0.3/RESEARCH-BACKLOG.md`.

## Dependencies

None (on `dev` @ `9e3f0c3`, after PR-052).

## Architecture section implemented

`docs/ARCHITECTURE.md` §Reproducibility Architecture item 5 (Data): how `logical_hash` is computed and what it holds in memory.

## Verification criteria

- [x] **Failing first.** On `dev` @ `9e3f0c3`, the new tests gave **20 failed, 7 passed**.
  - `test_polars_is_handed_one_file_at_a_time` failed on the behaviour itself: `AssertionError: a scan named several files: [[…/a.parquet, …b.parquet, …c.parquet]]`.
  - The equality and refusal tests failed on the then-missing reference.
  - The 7 that passed are the locks that must hold both before and after: the golden and pinned digests, layout invariance, the duplicate multiset, no files, and array release.
- [x] **Bit-identity:**
  - digest, row count and schema are equal to the whole-source reference in 10 layouts: one file; three files; a shuffled uneven split; per-file column order; an empty file first, in the middle, or alone; duplicates across files; a hive layout; nested directories;
  - the frame covers every dtype family: Int8 extremes, UInt64 max, NaN / ±0.0 / ±inf / subnormal, Float32, nulls, Boolean, unicode and empty strings, Categorical, Enum, Date, Datetime(ms, UTC), Duration, List, Struct, Binary;
  - equal on `c6_set` `fill/` and `walk/`;
  - the golden manifest's recorded `data_logical_hash` / `data_bytes_hash` and four digests pinned from the old code (Polars 1.40.1) are reproduced.
- [x] **Refusals.** For 12 later-file schema variants, the per-file path accepts exactly what one scan over every file accepts (with an equal digest) and refuses the rest with the same exception class, naming the file. A Null-typed column followed by a typed one is refused; an empty file list is refused.
- [x] **Memory bound (structural).** No Polars scan names more than one file, and no collected frame holds more than one file's rows. The row-hash array is traced at ≥ 8 B per row while hashing and is gone on return (< 1 B per row left). rux-ml had no RSS-test pattern, and an RSS bound in a unit test is platform- and allocator-dependent, so the bound is asserted on its cause. Mac measurement, informational (24 files, 9.6M rows × 43 columns): whole-source peak 3.9 GiB, per-file 1.0–1.2 GiB, at 1.4 s vs 1.8 s.
- [x] **The paths.** `data_hashes` (train, tune, promote) and `training_set_sidecar` (`data bridge`) both return the reference digest on `c6_set`.
- [ ] **Real set (DEFERRED; the program lead runs it on the desktop).** `scripts/check_logical_hash_per_file.py <set>` on PR-024's final set. Expected:
  - fill about 50 s at about 7.3 GiB, and the reference about 10 s at about 18 GiB at 156 days;
  - walk about a fifth of each;
  - about 1.5 min in all.

  Tested end to end on a small set (exit 0 when equal, 1 when the listed `data_hash` differs, 2 with no subtrees). Smoked on the Mac on a 9.6M-row synthetic set: equal, per-file 1.8 s at 1.17 GiB peak vs whole-source 0.7 s at 3.9 GiB.
- [x] Default suite, ruff check, `ruff format --check` on the touched files, basedpyright `src/`: see the commit message.

## Research backing

Program PR-027 §Findings Phase 3 Round 2b ("The hash, per file") and Phase 4 A2. The probes are in the State Assessment above.

## Notes — for the program lead

- **Wall.** The per-file path is slower. Measured on the desktop it takes about 51 s against 10 s at 156 days: at about 1.1 effective cores it is bound to one file's hash. The trade was accepted in A2 for the 10.7 GiB it frees. Hashing several files at once would need a new TOML knob and costs memory, so it is not built.
- **What the memory figure covers.** The bound is on what the hash itself holds: one u64 per row plus one file's working set. How many freed pages the allocator (jemalloc, on Linux) keeps resident after the call is not controllable from Python. The deferred script records the RSS left after the call returns (`rss_after_return_mib`) so the real set shows it.
- **Running the reference.** On a set much longer than 156 days, the whole-source child can pass a 20 GiB scope (about 200 B per row). The script prints that estimate first. A child killed by the scope is recorded as a failure, and `--skip-whole-source` then compares with the manifest's recorded `data_hash` alone.
