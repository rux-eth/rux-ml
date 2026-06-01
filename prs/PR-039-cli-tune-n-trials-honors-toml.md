# PR-039: `rux-ml tune {start,resume}` honor `[tuning] n_trials` from TOML

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-debug-pr.md` has been completed in full and its output appended to the `## Research findings` section below.

Debug-PR (always research-required; no Tier-1 light path). Two mandatory user signoffs:

- After **Phase 4 — Test Design**
- After **Phase 6 — Fix Design**

Skipping the debug-PR procedure or either signoff is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md` and the failing-test-first gate in `PROCEDURE-debug-pr.md`.

## Status

Debug-PR

## Problem

`rux-ml tune start` and `rux-ml tune resume` silently ignore the `[tuning] n_trials` value in the active layered TOML config and always run the Typer flag's default of `50` trials when `--n-trials` is not passed on the CLI.

Observed 2026-05-23 on the workbench during a 10-trial HPO calibration: a study TOML authored with `n_trials = 10` produced a banner reading `isolation: subprocess    n_trials: 50` and the parent loop ran 50 trials until killed. Every HPO sweep in rux-ml's history that relied on the TOML's `n_trials` silently ran 50 trials instead. Sweeps launched with `--n-trials N` explicitly (e.g. the PR-031 baseline) were unaffected; the bug only bites when the user expects the TOML to win.

## Bugs (current vs expected behavior)

### Bug 1a — `rux-ml tune start` ignores `[tuning] n_trials` in study TOML

**Current behavior**

`src/rux_ml/cli/tune.py:123-143`: the `start` Typer command declares `n_trials: Annotated[int, typer.Option("--n-trials", "-n", min=1)] = 50` at `:133`. When the user invokes `rux-ml ... tune start` without passing `--n-trials`, Typer supplies the parameter's default (`50`) as the bound argument. The function body uses that bound `n_trials` directly (banner at `:142`; loop driver at `:143`). The active `RuxMLConfig`'s `cfg.tuning.n_trials` (loaded from `[tuning] n_trials = N` in the layered TOML overlay; e.g. `configs/studies/crypto_breakout_h3_baseline.toml:19` sets `n_trials = 10`) is never consulted.

Empirical evidence (2026-05-23 workbench run): a study TOML authored with `n_trials = 10`; invocation `rux-ml --problem crypto_breakout_h3 --study crypto_breakout_h3_hpo tune start` emitted banner `isolation: subprocess    n_trials: 50` and the parent loop ran 50 iterations until killed. Banner reproduces the symptom.

**Expected behavior**

When the user does NOT pass `--n-trials` on the CLI, `tune start` honors `cfg.tuning.n_trials` from the layered TOML (study > problem > base, per the documented override precedence `CLI > env > .env > study > problem > base > defaults`). When the user DOES pass `--n-trials N`, the CLI value wins. The banner line MUST report the effective `n_trials` actually about to drive the trial loop.

Acceptable solution shapes (deferred to Phase 6 research):

- Change the Typer option's default to `None`; body resolves `effective = n_trials if n_trials is not None else cfg.tuning.n_trials`.
- Remove the `--n-trials` flag entirely and force TOML-only.
- Add an explicit sentinel (e.g. `-1`) that triggers TOML fallback.

Observable acceptance criteria:

- A `CliRunner.invoke(app, [..., "tune", "start", "..."])` against a config whose `[tuning] n_trials = 2` runs exactly 2 trials AND the banner reports `n_trials: 2`.
- A `CliRunner.invoke(app, [..., "tune", "start", "...", "--n-trials", "3"])` against the same config runs 3 trials AND banner reports `n_trials: 3` (regression guard: explicit CLI flag still overrides TOML).

### Bug 1b — `rux-ml tune resume` ignores `[tuning] n_trials` in study TOML

**Current behavior**

Same mechanism as Bug 1a, second site. `src/rux_ml/cli/tune.py:153-173`: `resume` declares `n_trials: Annotated[int, typer.Option("--n-trials", "-n", min=1)] = 50` at `:157`; body uses the bound value at the banner `:163` and the loop driver `:165`. No reference to `cfg.tuning.n_trials`.

Not independently reproduced on workbench in the 2026-05-23 session (only `start` was exercised), but the code shape is identical to Bug 1a; reproducibility is expected to be analogous.

**Expected behavior**

Same contract as Bug 1a, applied to `resume`: TOML default honored when `--n-trials` is absent, CLI flag wins when present, banner reports the effective count.

Acceptable solution shapes: whichever shape Phase 6 selects for Bug 1a applies symmetrically to Bug 1b.

Observable acceptance criteria:

- A `CliRunner.invoke(app, [..., "tune", "resume", "..."])` against a config whose `[tuning] n_trials = 2`, after a one-trial `start` precursor seeds the study, runs exactly 2 additional trials AND the banner reports `n_trials: 2`.
- A `CliRunner.invoke(app, [..., "tune", "resume", "...", "--n-trials", "3"])` against the same config runs 3 trials AND banner reports `n_trials: 3` (regression guard).

**Cross-coupling**: 1a and 1b are atomic siblings — same defect class, two call sites, independently testable. Treated as sub-bugs of "Bug 1": one Phase 6 fix shape, two Phase 4 test designs, two Phase 5 RED confirmations. Independent verifiability per `PROCEDURE-debug-pr.md` Phase 0 "Atomic granularity rule" is preserved.

## Dependencies

None. PR-038 (procedure sync) merged at `ba30b15` and is the gating procedural prerequisite; satisfied.

## Out of scope

- The 2026-05-23 abandoned workbench study `crypto_breakout_h3_crypto_breakout_h3_hpo_20260523T144713Z` in `studies/studies.db` (leave it; not a baseline).
- Calibration branch `calibration/crypto-h3-hpo-9dim` work (held until PR-039 merges).
- Any other CLI flag's interaction with its TOML counterpart (only `--n-trials` is in scope here; Phase 6's convention research may inform a future audit).
- ROADMAP row: no active `docs/<x.y>/ROADMAP.md` for v0.4; "no row to flip" semantic preserved per PR-028/29/37/38 precedent.

## Research findings

_To be populated by `PROCEDURE-debug-pr.md`. Do not begin implementation until this section exists with completed findings from all required phases (with the two mandatory user signoffs after Phase 4 + Phase 6)._

### Phase 1 — Expanded State Assessment (2026-05-31)

**Files in bug spec**: `src/rux_ml/cli/tune.py` (both bug sites in the same module).

**Import closure (≥1 hop from `cli/tune.py`)**:

- `rux_ml._internal.env.pin_threads` — env-var pinning before in-process objective
- `rux_ml._internal.seeds.make_seed_bag` — study-level sampler seed
- `rux_ml.cli._shared.get_options` + `GlobalOptions` — root-callback payload (config path, problem, study, overrides dict)
- `rux_ml.config.RuxMLConfig` — `from_layers(...)` produces the resolved `cfg.tuning.n_trials` from TOML overlay
- `rux_ml.runs.study_name` — name-template substitution
- `rux_ml.training.optuna_direction` — direction enum from metric
- `rux_ml.tuning.{build_objective, create_or_load, make_pruner, make_sampler, run_subprocess_trial}` — the trial machinery

**Common imports across the 1a + 1b sites**: identical — both `start` and `resume` are in the same module and share the entire import set. The shared trial driver is `_run_trials(...)` at `cli/tune.py:84-120`, which correctly forwards whatever integer it receives (both the subprocess loop at `:105` and the in-process `study.optimize(..., n_trials=n_trials)` at `:101`). The defect is upstream of `_run_trials`: it lives in the Typer-flag binding + the function bodies that never consult `cfg.tuning.n_trials`.

**Full `n_trials` surface in production code** (`grep -rn "n_trials" src/rux_ml/`):

| Location | Role | Bug-relevant? |
|---|---|---|
| `config/tuning.py:55` | `TuningConfig.n_trials: int = 50` (Pydantic default) | Yes — this default is what should win when no CLI flag and no TOML override is present, but currently never gets a chance |
| `cli/tune.py:84-120` | `_run_trials(n_trials)` shared driver | No — correctly forwards its argument |
| `cli/tune.py:133` | `start` Typer flag default `= 50` | **YES — Bug 1a originating defect** |
| `cli/tune.py:142` | `start` banner | Reports the bound (buggy) value |
| `cli/tune.py:143` | `start` → `_run_trials(...)` | Forwards the bound (buggy) value |
| `cli/tune.py:157` | `resume` Typer flag default `= 50` | **YES — Bug 1b originating defect** |
| `cli/tune.py:163` | `resume` banner | Reports the bound (buggy) value |
| `cli/tune.py:165` | `resume` → `_run_trials(...)` | Forwards the bound (buggy) value |
| `cli/tune.py:229` | `retry_trial` calls `study.optimize(objective, n_trials=1)` | No — hardcoded `1` is intentional (single-trial replay of one prior failed trial); not a configurability bug |
| `_internal/trial_runner.py:129` | Child runs `study.optimize(objective, n_trials=1)` | No — one trial per subprocess child is by-design per D10/D16 |

The defect is fully localized to `cli/tune.py:133` and `:157`. No transitive import is implicated.

**Recent commits filter (since `v0.3.0` tag `66cd42c`)**: ZERO. `git log 66cd42c..HEAD -- src/rux_ml/cli/tune.py` returns empty. The bug is not the result of a recent change.

All-time commits touching `cli/tune.py`: PR-003 (`fb3a01f`, CLI skeleton), PR-007 (`e1b4ec5`, Tier-2 Optuna basics — first appearance of an `n_trials` Typer flag in this file), PR-008 (`5768b26`, subprocess split — likely the commit that duplicated the flag default into the `resume` site), PR-011 (`19fef2f`, memory & threading), PR-013 (`519803c`, seeds). `config/tuning.py:55`'s `n_trials: int = 50` predates all CLI work (PR-002). The bug has shipped in every release from v0.1.0 onward; it is longstanding, not regressed.

**Override-precedence contract** (`docs/ARCHITECTURE.md:476-484`): `CLI > env > .env > study > problem > base > Pydantic defaults`. The documented "CLI" channel is **the `--set <dot-path>=<value>` mechanism in `cli/_shared.py:39-63`**. Implicit Typer-flag defaults are NOT a documented override channel — yet `--n-trials`'s default of `50` is currently leaking into that channel via Typer's "always supply the default when the flag is absent" semantics. The bug is therefore a contract violation: an implicit value is being treated as if the user passed `--set tuning.n_trials=50` explicitly.

**Empirical baseline (persisted telemetry)**:

| Bug | Status | Persisted-run count | Window | Density | Notes |
|---|---|---|---|---|---|
| 1a | not-telemetry-observable | N/A | — | — | CLI option-parsing bug; reproduction in Phase 3 via `CliRunner` |
| 1b | not-telemetry-observable | N/A | — | — | Same class as 1a; reproduction in Phase 3 |

Scan command: `N/A — CLI option-parsing bugs produce no persisted-telemetry signal per DEBUG-PR-STANDARDS.md Phase 1 status taxonomy.` (Side note: the workbench's `studies/*.db` would only show "every recent HPO study has exactly 50 completed trials unless the user explicitly passed `--n-trials N`" — which is symptomatic but not diagnostic; the bug is in option parsing, not in the persisted state.)

**Approach decision: parallel handling of Bug 1a + Bug 1b.** Rationale:

- Same defect class (Typer flag default `= 50` at adjacent sites)
- Same fix shape (whatever Phase 6 lands at site 1a is mechanically applied at 1b)
- Same `_run_trials` shared driver downstream — neither bug interacts with anything the other doesn't
- Both bug sites are in one file → one commit per phase, not two

The 1a/1b atomic split is preserved at the **test-design + RED-confirmation level** (Phase 4 + Phase 5 produce two tests each — one TOML-honored, one CLI-overrides; four total). The fix in Phase 7 lands as a single coordinated change to `cli/tune.py`.

**Test fixture readiness**: `tests/cli/test_tune_subcommands.py:16-76` (`tune_workdir`) already writes a `base.toml` with `[tuning] n_trials = 2` and exercises the `start` + `resume` CLI surface end-to-end. Every existing `tune_workdir`-based test (`test_tune_start_*`, `test_tune_resume_*`, etc., lines 83-260) passes `--n-trials N` explicitly on the CLI, which masks the bug — none of them tests the TOML-honoring path. This is the natural test bed for the four new Phase-5 RED tests; no fixture changes needed.

**Exit criteria**: every bug from Phase 0 has a cited candidate root-cause location (Bug 1a → `cli/tune.py:133`; Bug 1b → `cli/tune.py:157`). User approval requested before Phase 2.
### Phase 2 — Hypothesis Assembly (2026-05-31)

**Bug 1a hypotheses (ranked)**:

1. **H1a-1 — Typer always supplies the parameter default when the flag is absent.** Per `cli/tune.py:133`, `n_trials: Annotated[int, typer.Option(...)] = 50`; when the user invokes `tune start` without `--n-trials`, Typer binds `n_trials = 50` in the function's namespace. The body at `:142-143` uses that bound integer directly and never consults `cfg.tuning.n_trials`. Confirm via dev-mode `CliRunner` repro: invoke `tune start <name>` against a fixture whose `[tuning] n_trials = 2`, with NO `--n-trials` flag → observe banner `n_trials: 50` and 50-trial loop kickoff. Refute: banner reads `n_trials: 2` or the loop runs 2 trials. **Highest likelihood given Phase 1 evidence.**

2. **H1a-2 — `RuxMLConfig.from_layers` silently drops `[tuning] n_trials` during overlay.** Less likely; would imply a broader config-loading bug affecting ALL `cfg.tuning.*` fields, which Phase 1's grep on the production source contradicts (other `cfg.tuning.*` reads — `sampler`, `pruner`, `trial_isolation`, `entropy` at `cli/tune.py:141, 142, 73` — work correctly in shipped runs). Confirm: instrument a `CliRunner` test to inspect `cfg.tuning.n_trials` immediately after `_load_cfg(ctx)` returns — should be `2` (TOML value) if the overlay works, `50` (Pydantic default) if it doesn't. Refute (expected): cfg.tuning.n_trials == 2 → the config loads correctly; the body just never reads it; confirms H1a-1.

3. **H1a-3 — Closure / late-binding capture in the f-string banner.** Implausible: `cli/tune.py:142` uses local-parameter interpolation, no closure capture. Already refuted by Phase 1 code read. Listed for procedural completeness.

**Bug 1b hypotheses (ranked)**: structurally identical to Bug 1a applied to `cli/tune.py:157` and the `resume` body at `:163-165`.

1. **H1b-1 — Typer always supplies the parameter default when the flag is absent** (the `resume` site). Confirm via parallel `CliRunner` repro: seed an existing study via a one-trial `start` precursor, then invoke `tune resume <name>` against the same fixture with NO `--n-trials` flag → observe banner `n_trials: 50` and 50-additional-trial loop kickoff. Refute: banner reads `n_trials: 2` or the loop runs 2 trials.

2. **H1b-2 / H1b-3** — same as H1a-2 / H1a-3, identical refutation paths. Listed for symmetry.

**Dependencies**: H1a-1 and H1b-1 are independent (separate Typer commands, separate function bodies, separate parameter bindings — Typer parses each command's options in isolation). Confirming one does not confirm the other; Phase 3 will run both `CliRunner` repros. The two sites are linked only by code shape and by the shared `_run_trials` driver (which is not implicated).

**Exit criteria**: every bug has ≥1 hypothesis with a defined confirm/refute observation. User approval requested before Phase 3.
### Phase 3 — Bug Confirmation (2026-05-31)

**Web evidence — Typer/Click default-binding semantics (Q1)**:

When a Typer option declares a default (e.g. `param: Annotated[int, typer.Option(...)] = 50`) and the user omits the flag, the parameter is bound to the default value inside the function body — and from the value alone, the body **cannot** distinguish "user explicitly passed `--flag 50`" from "user did not pass `--flag` at all". The only supported distinction mechanism is `click.Context.get_parameter_source()`, which returns a `ParameterSource` enum (`COMMANDLINE` / `ENVIRONMENT` / `DEFAULT_MAP` / `DEFAULT` / `PROMPT`). Standard "did the user pass it?" workaround per Typer's own tutorial: declare `Optional[T] = None` and sentinel-check inside the body.

Sources:
- Click — Advanced Patterns (`get_parameter_source` API + `ParameterSource` enum): <https://click.palletsprojects.com/en/stable/advanced/>
- pallets/click DeepWiki — Value Resolution and Defaults (confirms source-tracking lives on `Context`, not on the value): <https://deepwiki.com/pallets/click/3.4-value-resolution-and-defaults>
- Typer — Optional CLI Arguments tutorial (canonical `Optional[T] = None` + sentinel pattern): <https://typer.tiangolo.com/tutorial/arguments/optional/>

*Disconfirming evidence searched*: looked for a Typer-level `parameter_source` shortcut or a "was-set" attribute on `typer.Option` — none exists; Typer defers entirely to Click's `Context`. Also looked for known unreliability in `get_parameter_source` — one caveat: it returns `None` if `ctx.params` is directly written (Click Advanced).

**Web evidence — canonical "config-file default + CLI override" pattern (Q2)**:

Two pattern shapes surface in the Typer/Click ecosystem:

1. **Click `Context.default_map`** (framework-canonical for config-defaults): an eager `--config` callback parses the config file and assigns the parsed dict to `ctx.default_map` BEFORE per-parameter parsing. Click's documented precedence then takes over: `CLI > env > default_map > Click default`. Sources:
   - Click — Advanced Patterns / Overriding Defaults: <https://click.palletsprojects.com/en/stable/advanced/>
   - Knowledge Bits — *Setting Default Option Values from Config Files with Click*: <https://jwodder.github.io/kbits/posts/click-config/>
   - `typer-config` (`@use_toml_config` wires TOML → `default_map`): <https://pypi.org/project/typer-config/> + <https://maxb2.github.io/typer-config/latest/api/>
   - `click-extra` — Configuration: <https://kdeldycke.github.io/click-extra/config.html>
   - `click-contrib/click-configfile`: <https://github.com/click-contrib/click-configfile>

2. **`default=None` + in-body fallback** (documented workaround): when the application also needs to know *whether* the user passed the flag, declare `Optional[T] = None` and resolve `effective = cli_value if cli_value is not None else config_value`. Same sources as Q1 — Typer's optional-arguments tutorial documents this as the canonical "was-set?" pattern.

*Disconfirming evidence searched* for `default_map` pitfalls: (i) `multiple=True` defaults must be list/tuple, not string ([jwodder]); (ii) `show_default=True` doesn't reflect `default_map` for flag options with secondary opts (Click #2632); (iii) `BoolParamType` historically `AttributeError`'d when fed via `default_map` (Click #1567). None contradict `default_map` as canonical — all are edge-case bugs. **Both patterns are framework-blessed; the choice is a Phase 6 fix-design call, not a Phase 3 mechanism question.**

**Empirical evidence (dev-mode repro, local Mac)**:

Scratch test: `/tmp/pr039_phase3_repro.py` (not committed). Mimics `tests/cli/test_tune_subcommands.py`'s `tune_workdir` fixture shape: synthetic 200-row Parquet + `base.toml` with `[tuning] n_trials = 2`, `trial_isolation = "in_process"`, `device = "cpu"`. Three tests:

1. **`test_h1a2_refute_cfg_loads_n_trials_from_toml`** — invokes `RuxMLConfig.from_layers(workdir / "base.toml")` directly, asserts `cfg.tuning.n_trials == 2`. **PASSED**. H1a-2 / H1b-2 **REFUTED**: the config overlay correctly loads the TOML's `n_trials = 2`. The bug is NOT in `from_layers`.

2. **`test_h1a1_confirm_tune_start_banner_reads_50_despite_toml_2`** — `CliRunner.invoke(app, ["--config", ..., "tune", "start", "smoke_h1a1"])` (NO `--n-trials` flag). **PASSED** (asserting bug present). Captured banner: `isolation: in_process    n_trials: 50`. Final `optuna.load_study(...).trials` count: **50**. H1a-1 **CONFIRMED**: Typer binds the parameter default `50` when the flag is absent; the body uses it directly; the loop runs 50 iterations despite the TOML asking for 2.

3. **`test_h1b1_confirm_tune_resume_banner_reads_50_despite_toml_2`** — `CliRunner.invoke(app, ..., "tune", "resume", "smoke_h1b1")` (NO `--n-trials` flag, after a 1-trial seeded study). **PASSED** (asserting bug present). Captured banner: `resuming study smoke_h1b1 (1 existing trials)\nisolation: in_process    n_trials: 50`. Final study trial count: **51** (1 seed + 50 resume). H1b-1 **CONFIRMED**: identical mechanism at the `resume` site.

Total wall time: 13.3 s on local Mac (M4 Max, CPU mode). No instrumentation added to production code; scratch lives in `/tmp/` and is not committed.

**Bug 1a — confirmed mechanism**

- *Hypothesis tested*: H1a-1 (Typer always supplies the parameter default when the flag is absent)
- *End-to-end mechanism*: User invokes `rux-ml ... tune start <name>` without `--n-trials` → Typer's option parser reaches `cli/tune.py:133` (`n_trials: Annotated[int, typer.Option(...)] = 50`), sees no CLI value, and binds `n_trials = 50` in the `start` function's namespace (per Click `get_parameter_source` semantics: source becomes `ParameterSource.DEFAULT`) → `cli/tune.py:142` interpolates the bound `50` into the banner → `cli/tune.py:143` forwards `50` to `_run_trials(...)` → the subprocess loop at `cli/tune.py:105` (or the in-process `study.optimize(..., n_trials=50)` at `:101`) executes 50 iterations. The correctly-loaded `cfg.tuning.n_trials = 2` (verified at Phase 3 step 1) is never consulted by the `start` body.
- *Status*: **confirmed** (originating defect: `src/rux_ml/cli/tune.py:133` — `n_trials: Annotated[int, typer.Option("--n-trials", "-n", min=1)] = 50`)

**Bug 1b — confirmed mechanism**

- *Hypothesis tested*: H1b-1 (parallel to H1a-1, second site)
- *End-to-end mechanism*: structurally identical to Bug 1a, at the `resume` function. User invokes `rux-ml ... tune resume <name>` without `--n-trials` → Typer parser at `cli/tune.py:157` binds `n_trials = 50` → `:163` banner reports `50` → `:164-166` `_run_trials(...)` loops 50 additional trials regardless of `cfg.tuning.n_trials`.
- *Status*: **confirmed** (originating defect: `src/rux_ml/cli/tune.py:157` — `n_trials: Annotated[int, typer.Option("--n-trials", "-n", min=1)] = 50`)

**Isolation observation**: the bug is independent of `cfg.tuning.trial_isolation` mode — `_run_trials` consumes the buggy `n_trials` argument in both the `in_process` branch (line `:101`, the path Phase 3's repro exercised) and the `subprocess` branch (line `:105`, the workbench-observed path in the 2026-05-23 reproduction). The dev-mode in-process repro is sufficient evidence for the subprocess path because `_run_trials` is the common downstream consumer.

**Exit criteria**: every bug has `Status: confirmed` with file:line and empirical evidence. User approval requested before Phase 4 (Test Design).
### Phase 4 — Test Design (2026-05-31) ⚠️ USER SIGNOFF REQUIRED

**Cross-cutting design decisions** (apply to all four tests):

- *Test type*: **integration**. All four are `CliRunner.invoke(app, ...)`-driven tests at the canonical rux-ml CLI integration surface (`tests/cli/`). Both the Typer binding AND the `cfg.tuning.n_trials` read path must be exercised end-to-end; mocking Typer would defeat the test. Per `docs/DEBUG-PR-STANDARDS.md` Phase 4: "for rux-ml CLI bugs the typer `CliRunner`-driven tests in `tests/cli/` are the canonical integration surface."
- *Fixture*: existing `tune_workdir` from `tests/cli/test_tune_subcommands.py:16-76`. It already writes `[tuning] n_trials = 2` (line 52), `device = "cpu"` (line 43), and a tiny 200-row synthetic Parquet — exactly the shape the bug needs. **No fixture changes**.
- *Helpers*: existing `_argv(workdir, *args)` at `test_tune_subcommands.py:79-80`; existing `runner` fixture from `tests/cli/conftest.py:10-11`.
- *Isolation mode*: each test will force `--set tuning.trial_isolation="in_process"` for speed (3 trials × 200 rows × 3 folds × 8 estimators completes in <1 s per test on the local Mac; subprocess mode would add ~10 s per test for child-spawn overhead × 4 tests = ~40 s tax with no additional bug-coverage value). The bug is mode-independent per Phase 3's isolation observation; existing subprocess-mode tests (`test_tune_resume_appends_trials` at `:149`, etc.) already cover the subprocess dispatch path and will continue to pass after the fix because they all pass `--n-trials` explicitly.
- *Assertions per test*: BOTH the banner line AND the final `study.trials` count. The banner alone could pass even if the loop dispatch bug remained; the trial count alone could pass even if the banner was inconsistent. Two assertions catch divergence at either point in the data flow.
- *Test placement*: append to `tests/cli/test_tune_subcommands.py` (existing module already exercises `tune start` + `tune resume` + the `tune_workdir` fixture; co-located tests preserve discoverability).

---

**Bug 1a — TOML-honored test**

*Test name*: `test_tune_start_honors_tuning_n_trials_from_toml`

*Setup*: `tune_workdir` fixture (TOML has `[tuning] n_trials = 2`); force `tuning.trial_isolation="in_process"` via `--set`.

*Trigger*: `runner.invoke(app, _argv(workdir, "--set", "tuning.trial_isolation=\"in_process\"", "tune", "start", "smoke_toml_honored"))` — note the absence of `--n-trials` on the CLI.

*Assertion*:
1. `result.exit_code == 0` (sanity)
2. `"n_trials: 2"` appears in `result.stdout` (banner reflects TOML value)
3. `optuna.load_study(study_name="smoke_toml_honored", storage=<workdir>/studies/studies.db).trials` has length `== 2` (loop honored the TOML value)

*Predicted current behavior (current/unfixed code)*: test FAILS. Banner reads `n_trials: 50` (Phase 3 confirmed mechanism — Typer binds the parameter default at `cli/tune.py:133`), and the in-process `study.optimize(..., n_trials=50)` runs 50 trials. Both assertions 2 and 3 fail; the test thereby proves Bug 1a exists in shipped code.

*Predicted post-fix behavior*: test PASSES. Whichever fix shape Phase 6 lands (Click `default_map` injection or `Optional[int] = None` + in-body fallback), the body resolves `effective_n_trials = 2` from `cfg.tuning.n_trials`; banner interpolates `2`; loop runs 2 trials.

---

**Bug 1a — CLI-override regression guard**

*Test name*: `test_tune_start_cli_flag_overrides_toml_n_trials`

*Setup*: same as above.

*Trigger*: `runner.invoke(app, _argv(workdir, "--set", "tuning.trial_isolation=\"in_process\"", "tune", "start", "smoke_cli_override", "--n-trials", "3"))` — explicit `--n-trials 3` against TOML's `n_trials = 2`.

*Assertion*:
1. `result.exit_code == 0`
2. `"n_trials: 3"` appears in `result.stdout` (banner reflects CLI value, not TOML)
3. `len(study.trials) == 3` (loop honored CLI value)

*Predicted current behavior*: test PASSES today (current behavior coincidentally satisfies the contract because the Typer default and the explicit flag both bind via the same channel — the bug masks itself when the user passes the flag). This is a **regression guard**, not a RED test for the bug — Phase 5 will document it as "passes today; protects the override-precedence contract from being broken by the Phase 6 fix" per `PROCEDURE-debug-pr.md`'s acceptance of pre-existing-pass guards. (See note below.)

*Predicted post-fix behavior*: test PASSES. CLI value wins per documented precedence `CLI > … > TOML`.

---

**Bug 1b — TOML-honored test**

*Test name*: `test_tune_resume_honors_tuning_n_trials_from_toml`

*Setup*: `tune_workdir` fixture; seed an existing study via a precursor `runner.invoke(app, _argv(workdir, "--set", "tuning.trial_isolation=\"in_process\"", "tune", "start", "smoke_resume_toml", "--n-trials", "1"))` — explicit `--n-trials 1` to keep the seed deterministic and isolate the bug under test to the `resume` call.

*Trigger*: `runner.invoke(app, _argv(workdir, "--set", "tuning.trial_isolation=\"in_process\"", "tune", "resume", "smoke_resume_toml"))` — no `--n-trials` flag.

*Assertion*:
1. `result.exit_code == 0`
2. `"n_trials: 2"` appears in `result.stdout` (banner reflects TOML value)
3. `len(study.trials) == 3` (1 seed + 2 resume — loop honored TOML's `n_trials = 2`)

*Predicted current behavior*: test FAILS. Banner reads `n_trials: 50`; loop runs 50 additional trials; `len(study.trials) == 51`. Proves Bug 1b exists.

*Predicted post-fix behavior*: test PASSES.

---

**Bug 1b — CLI-override regression guard**

*Test name*: `test_tune_resume_cli_flag_overrides_toml_n_trials`

*Setup*: same seed precursor as the Bug 1b TOML-honored test (one-trial start with explicit `--n-trials 1`).

*Trigger*: `runner.invoke(app, _argv(workdir, "--set", "tuning.trial_isolation=\"in_process\"", "tune", "resume", "smoke_resume_cli_override", "--n-trials", "3"))` — explicit `--n-trials 3` against TOML's `n_trials = 2`.

*Assertion*:
1. `result.exit_code == 0`
2. `"n_trials: 3"` appears in `result.stdout`
3. `len(study.trials) == 4` (1 seed + 3 resume — loop honored CLI value)

*Predicted current behavior*: PASSES today (same reasoning as the Bug 1a regression guard).

*Predicted post-fix behavior*: PASSES.

---

**Note on the "test that passes today" anti-pattern from `DEBUG-PR-STANDARDS.md`**:

`docs/DEBUG-PR-STANDARDS.md` ("Anti-patterns" §) flags "test that passes against unfixed code" as a hard violation when the test is meant to prove a bug. The two regression-guard tests are NOT meant to prove the bug — they're meant to lock in the contract that the fix must preserve (i.e., they protect the `CLI > TOML` half of the override precedence chain from being broken by a Phase 6 fix that overcorrects to "TOML always wins"). Without these guards, a fix that defaults to `cfg.tuning.n_trials` and forgets to respect explicit `--n-trials` would slip silently. Both guards map 1:1 to the "explicit CLI wins" half of each bug's expected-behavior contract in `## Bugs`.

This is consistent with the procedure: each test maps to a bug (1:1 from the verification-criteria checklist), the RED tests prove the current defect, the guards lock the regression surface. Total tests: **4** (2 RED + 2 guards), exactly matching the 4 verification-criteria checkboxes already in `## Verification criteria`.

**Test→Verification-criteria mapping**:

| Test name | Maps to verification criterion |
|---|---|
| `test_tune_start_honors_tuning_n_trials_from_toml` | Bug 1a TOML-honored |
| `test_tune_start_cli_flag_overrides_toml_n_trials` | Bug 1a regression guard |
| `test_tune_resume_honors_tuning_n_trials_from_toml` | Bug 1b TOML-honored |
| `test_tune_resume_cli_flag_overrides_toml_n_trials` | Bug 1b regression guard |

**Exit criteria**: ⚠️ **USER SIGNOFF REQUIRED** on this test design before Phase 5 (test implementation + failure confirmation). After signoff: implement exactly as designed, no deviations without re-opening Phase 4 and re-acquiring signoff.
### Phase 5 — Test Implementation + Failure Confirmation (2026-05-31)

Four tests implemented exactly as designed in Phase 4 and appended to `tests/cli/test_tune_subcommands.py` (before the trailing `@pytest.mark.gpu` block, preserving the GPU-tests-last convention). No deviations from Phase 4 design.

Run: `uv run pytest tests/cli/test_tune_subcommands.py -k "n_trials_from_toml or cli_flag_overrides_toml" -v`. Wall time: 14.4 s on local Mac (M4 Max, CPU mode). Final result: **2 failed, 2 passed** — matches Phase 4 predictions exactly.

**Bug 1a — test results**

| Test | Predicted | Actual | Match |
|---|---|---|---|
| `test_tune_start_honors_tuning_n_trials_from_toml` | FAIL (banner = 50; trials = 50) | **FAILED** | ✓ |
| `test_tune_start_cli_flag_overrides_toml_n_trials` | PASS today (regression guard) | **PASSED** | ✓ |

*Failure excerpt* (`test_tune_start_honors_tuning_n_trials_from_toml`):

```
AssertionError: banner must report effective n_trials=2 (TOML value); got stdout:
assert 'n_trials: 2' in 'study:    smoke_toml_honored\n... isolation: in_process    n_trials: 50\n...
best value (auc): 0.964725\nbest trial:  #39\nbest params: ...\n'
```

The fixture's `[tuning] n_trials = 2` was ignored; the banner emitted `n_trials: 50` and the in-process `study.optimize(..., n_trials=50)` ran 50 trials (best trial `#39` proves more than 2 trials executed). Failure mode matches Phase 3's confirmed mechanism for Bug 1a (`cli/tune.py:133` Typer default leaks into the bound parameter).

**Bug 1b — test results**

| Test | Predicted | Actual | Match |
|---|---|---|---|
| `test_tune_resume_honors_tuning_n_trials_from_toml` | FAIL (banner = 50; trials = 51) | **FAILED** | ✓ |
| `test_tune_resume_cli_flag_overrides_toml_n_trials` | PASS today (regression guard) | **PASSED** | ✓ |

*Failure excerpt* (`test_tune_resume_honors_tuning_n_trials_from_toml`):

```
AssertionError: banner must report effective n_trials=2 (TOML value); got stdout:
assert 'n_trials: 2' in 'resuming study smoke_resume_toml (1 existing trials)\nisolation: in_process    n_trials: 50\n\ntotal trials: 51 (best auc=0.964725)\n'
```

The `resume` banner emitted `n_trials: 50` despite the TOML's `n_trials = 2`; the loop ran 50 additional trials, giving `total trials: 51` (1 seed + 50 resume). Failure mode matches Phase 3's confirmed mechanism for Bug 1b (`cli/tune.py:157` Typer default leaks into the bound parameter).

**Discrepancies**: none. Predicted failure mode (banner mismatch first, trial-count mismatch second) matched exactly. No loopback to Phase 4 or Phase 0 required.

**Exit criteria**: every RED test fails in its predicted way; both regression guards pass against unfixed code as predicted. Tests committed to branch `pr-039/cli-tune-n-trials-honors-toml` alongside the PR stub.
### Phase 6 — Fix Design (2026-05-31) ⚠️ USER SIGNOFF REQUIRED

**Bug 1 (1a + 1b — symmetric fix at two sites) — proposed fix**

*Convention research* (cited from Phase 3, refined here):

Two framework-blessed shapes exist for "Typer/Click CLI flag with config-file default":

- **Shape A — Click `Context.default_map`** (canonical for whole-app config-defaults): an eager `--config` callback parses the TOML and assigns the parsed dict to `ctx.default_map`; Click's documented precedence then applies (CLI > env > `default_map` > Click `default`) automatically across all options. Sources: Click Advanced Patterns (`Context.default_map`) <https://click.palletsprojects.com/en/stable/advanced/>; `typer-config`'s `@use_toml_config` decorator <https://pypi.org/project/typer-config/>; `click-extra`'s Configuration mechanism <https://kdeldycke.github.io/click-extra/config.html>; Knowledge Bits — *Setting Default Option Values from Config Files with Click* <https://jwodder.github.io/kbits/posts/click-config/>.

- **Shape B — `Optional[T] = None` + in-body sentinel fallback** (canonical "did the user pass it?" pattern, applicable to per-option config-fallback): declare `n_trials: Annotated[int | None, typer.Option(...)] = None`, then in the body `effective_n_trials = n_trials if n_trials is not None else cfg.tuning.n_trials`. Sources: Typer — Optional CLI Arguments tutorial <https://typer.tiangolo.com/tutorial/arguments/optional/>; Click — Advanced Patterns (`Context.get_parameter_source`) <https://click.palletsprojects.com/en/stable/advanced/>; pallets/click DeepWiki — Value Resolution and Defaults <https://deepwiki.com/pallets/click/3.4-value-resolution-and-defaults>.

*Project precedent*:

- No architecture-reference repo is identified in `CLAUDE.md` for rux-ml. The named-arch-ref half of Phase 6 is **N/A**.
- **In-repo precedent strongly favors Shape B.** `src/rux_ml/cli/data.py:67` (`name: Annotated[str | None, typer.Option("--name", ...)] = None`) and `src/rux_ml/cli/runs.py:35` (`study: Annotated[str | None, typer.Option("--study", ...)] = None`) are the only existing Typer flags in the codebase that use the `Optional[T] = None` + sentinel pattern. Both rely on the sentinel for filter/optional behavior, not for TOML fallback specifically — but the pattern (`| None = None` + sentinel-check in the body) is the same shape Shape B requires.
- **No `default_map` / `get_parameter_source` / `ParameterSource` usage anywhere in `src/rux_ml/` or `tests/`.** Adopting Shape A would be a from-scratch addition: a new eager root callback, a new TOML-parse-into-dict step before Typer dispatch (separate from `RuxMLConfig.from_layers`'s existing layered overlay), and a new layer of coupling between Click's `Context` and rux-ml's Pydantic config object. The infrastructure cost is non-trivial.
- **All other ~20 Typer options in `src/rux_ml/cli/`** use bare defaults and have no TOML counterpart that should leak into them (`--verbose`, `--dry-run`, `--problem`, `--study`, `--trial`, `--to`, `--version`, etc.). The TWO bug sites at `cli/tune.py:133` + `:157` are the ONLY Typer options in the codebase that need TOML-fallback behavior today. Generalizing to `default_map` would be YAGNI.
- `cli/tune.py` already reads `cfg.tuning.*` heavily (`sampler` at `:77`, `pruner` at `:78`, `trial_isolation` at `:96, 142, 163`, `entropy` at `:73`). Adding `cfg.tuning.n_trials` to the body fits the existing pattern.

*Disconfirming evidence searched* for Shape B:

- Help-text wart: `--help` currently shows `default: 50`; after Shape B's `= None` change, `--help` would show `default: None` unless the help string is updated to clarify "defaults to `[tuning] n_trials` in active TOML (Pydantic default: 50)". Fix in the same commit by adding explicit `help=...` text on the Typer option. **Mitigated, not a blocker.**
- "Pattern (a) is described in the Typer docs as a 'workaround' specifically when you need to know whether the user passed the flag" — but the codebase's PR-039 use case is exactly that ("was the flag passed?" → "fall back to TOML if not"). So it's the canonical fit, not a workaround.
- A `[tuning] n_trials = 50` TOML value (coinciding with the old Pydantic default) becomes indistinguishable from "no TOML override" — true but harmless: the user's intent (`n_trials = 50`) is honored either way; the bug only bites when TOML asks for `≠ 50`.

*Recommended fix* (Shape B):

Two coordinated edits to `src/rux_ml/cli/tune.py`:

1. **Line 133** (`start` signature): change
   ```python
   n_trials: Annotated[int, typer.Option("--n-trials", "-n", min=1)] = 50,
   ```
   to
   ```python
   n_trials: Annotated[
       int | None,
       typer.Option(
           "--n-trials", "-n", min=1,
           help="Number of trials; defaults to `[tuning] n_trials` in TOML.",
       ),
   ] = None,
   ```

2. **Body of `start`** (between lines 137 and 142): insert sentinel resolution
   ```python
   effective_n_trials = n_trials if n_trials is not None else cfg.tuning.n_trials
   ```
   then replace every `n_trials` reference in the body (banner at `:142`, `_run_trials(...)` call at `:143`) with `effective_n_trials`.

3. **Lines 157 + body of `resume`**: structurally identical change at the second site.

*Files affected*: `src/rux_ml/cli/tune.py` only.

*Status*: **convention** (Shape B is framework-blessed by Typer's own optional-arguments tutorial AND is already in-repo at `cli/data.py:67` + `cli/runs.py:35`). Not BGGC — both the convention and the project-precedent dimensions have ≥1 cited working example.

*Risks accepted*:

- Two adjacent Typer options now permit `None` as input. Typer's `min=1` constraint still applies to non-`None` integer inputs (verified: Click's `IntRange(min=1)` doesn't apply to `None`); explicit `--n-trials 0` still raises as before.
- Future PRs that add new TOML-coupled CLI flags (e.g., `--sampler`, `--pruner`) will follow the same shape per the drafted CONVENTION below. Generalizing to `default_map` is deferred until ≥3 such flags exist (YAGNI threshold; if the rule becomes load-bearing, it gets infrastructure-fied).

**Rollback decisions** (`## Code changes since baseline` is N/A — clean branch, only the Phase 5 RED tests commit at `9255208` lives on it; no speculative commits to roll back):

| Commit | Decision | Citation |
|---|---|---|
| `9255208` (Phase 5 RED tests + PR stub) | **keep** — turns green in Phase 7 once the fix lands | Phase 5 design + Phase 6 fix shape both reference these tests; they're the load-bearing safety net |

**CONSTRAINTS.md / CONVENTIONS.md amendments needed**:

Drafted text for `docs/CONVENTIONS.md`, appended to the "Configuration" section (currently at `docs/CONVENTIONS.md:48` per Phase 1's read). Lands in the Phase 7 commit alongside the fix.

```markdown
- **CLI flags with TOML counterparts use `Optional[T] = None` + in-body fallback, NOT bare Typer defaults.** A Typer option declared as `param: Annotated[T, typer.Option(...)] = <default>` always binds `<default>` into the function namespace when the user omits the flag (Typer / Click documented behavior — see Click `Context.get_parameter_source`). If the function body also has a TOML-loaded counterpart in `cfg.X.Y`, the bare Typer default silently overrides the TOML value and breaks the documented override precedence chain (`CLI > env > .env > study > problem > base > Pydantic defaults` per `docs/ARCHITECTURE.md` §Configuration). Canonical shape:

    ```python
    param: Annotated[T | None, typer.Option("--param", help="…; defaults to [section] param in TOML.")] = None,
    # in body:
    effective_param = param if param is not None else cfg.section.param
    ```

    Precedents: `src/rux_ml/cli/data.py:67`, `src/rux_ml/cli/runs.py:35`, `src/rux_ml/cli/tune.py` start/resume `--n-trials` (PR-039). For users wanting fine-grained per-field CLI overrides without a dedicated flag, the canonical mechanism remains `--set <dot-path>=<value>` (`cli/_shared.py:39-63`). The `default_map` infrastructure pattern (Click Advanced Patterns) is deferred until ≥3 such flags exist; current usage (1 flag, 2 sites) does not justify it (YAGNI per `docs/CONSTRAINTS.md` "Reuse over reinvent").
```

No `docs/CONSTRAINTS.md` amendment proposed — the rule is a soft pattern (escapable via `--set`), not a hard constraint. Recommendation level: CONVENTION, not CONSTRAINT.

**Exit criteria**: ⚠️ **USER SIGNOFF REQUIRED** on:

1. Fix shape selection (Shape B — `Optional[int] = None` + sentinel fallback) at both sites
2. Help-text update wording
3. The drafted `docs/CONVENTIONS.md` amendment
4. The rollback decisions (trivially: keep `9255208`)

After signoff: implement exactly as designed in Phase 7. No deviations without re-opening Phase 6 and re-acquiring signoff. Failed-attempt protocol applies if the fix doesn't turn the Phase-5 tests green.
### Phase 7 — Implementation + Verification (2026-05-31)

**Implementation summary**

Files modified:

- `src/rux_ml/cli/tune.py` — two symmetric edits per Phase 6 Shape B:
  - **`start` (lines 132-148)**: signature `n_trials` changed from `Annotated[int, typer.Option("--n-trials", "-n", min=1)] = 50` to `Annotated[int | None, typer.Option("--n-trials", "-n", min=1, help="Number of trials; defaults to `[tuning] n_trials` in active TOML.")] = None`. Body resolves `effective_n_trials = n_trials if n_trials is not None else cfg.tuning.n_trials` immediately after `_open_study`. Banner at the n_trials echo line now reads `effective_n_trials`; `_run_trials(...)` call passes `effective_n_trials`.
  - **`resume` (lines 160-178)**: structurally identical change. Help text adjusted to "Number of additional trials" to reflect resume semantics.
- `docs/CONVENTIONS.md` — new paragraph appended to §Configuration Conventions (after the `Hash elision:` paragraph at line 155, before the `---` separator). Codifies the `Optional[T] = None` + in-body fallback rule for CLI flags with TOML counterparts; lists the three in-repo precedents (`cli/data.py:67`, `cli/runs.py:35`, the two new `cli/tune.py` sites); documents the `--set <dot-path>=<value>` escape hatch; defers Click `Context.default_map` infrastructure to the ≥3-flag YAGNI threshold per `docs/CONSTRAINTS.md` "Reuse over reinvent".
- `CHANGELOG.md` — `[Unreleased] ### Fixed` entry for PR-039 (inserted ahead of the PR-032 docstring fix entry to preserve reverse-chronological order within the section).
- `tests/cli/test_tune_subcommands.py` — three docstring summary lines tightened to satisfy ruff E501 (100-char cap); test bodies + assertions unchanged from the Phase 5 commit.

Files reverted (per Phase 6 rollback decisions): none. The only commit on this branch is `9255208` (Phase 5 RED tests + PR stub), kept per Phase 6.

`CONSTRAINTS.md` amendments committed: none (Phase 6 deliberately recommended CONVENTION, not CONSTRAINT).

ROADMAP row flipped: **N/A — no active version-scoped ROADMAP** (post-v0.3.0 cut; no v0.4 ROADMAP exists yet). "No row to flip" semantic preserves the `feedback_roadmap_flip_in_pr` streak per the PR-028 / PR-029 / PR-037 / PR-038 precedent chain.

**Test results**:

- Bug 1a: `test_tune_start_honors_tuning_n_trials_from_toml` — was failing in Phase 5 (banner: `n_trials: 50`, study trials: 50); now **PASSING** (banner: `n_trials: 2`, study trials: 2). ✓
- Bug 1a regression guard: `test_tune_start_cli_flag_overrides_toml_n_trials` — was passing in Phase 5 (locks `CLI > TOML` precedence); still **PASSING**. ✓
- Bug 1b: `test_tune_resume_honors_tuning_n_trials_from_toml` — was failing in Phase 5 (banner: `n_trials: 50`, total: 51); now **PASSING** (banner: `n_trials: 2`, total: 3). ✓
- Bug 1b regression guard: `test_tune_resume_cli_flag_overrides_toml_n_trials` — was passing; still **PASSING**. ✓

Run: `uv run pytest tests/cli/test_tune_subcommands.py -k "n_trials_from_toml or cli_flag_overrides_toml" -v` → **4 passed in 4.26 s** (10× speedup vs Phase 5's 14.4 s because the test loops now run 2-3 trials instead of 50; this is itself evidence the fix landed).

No failed-attempt loop required. Phase 6 design implemented as-designed on first attempt.

**Other gates** (project-mandated per `CLAUDE.md` "Build & Test Commands"):

- `uv run pytest` (default suite — excludes `gpu`/`slow`/`golden` markers): **407 passed, 1 skipped, 15 deselected in 33.23 s**. Net +4 from pre-PR baseline (403 → 407), matching the 4 new tests added in Phase 5. No regressions.
- `uv run ruff check .`: **All checks passed!** (after tightening 3 docstring summary lines that initially exceeded the 100-char cap — content equivalent, just line-broken differently)
- `uv run basedpyright src/`: **0 errors, 0 warnings, 0 notes**
- `make test-golden`: **not run** — the touched surface (`cli/tune.py` option-parsing only) does not affect the golden contract (XGBoost regression / tolerance comparison surface); explicit decision per `CLAUDE.md` "Build & Test Commands" guidance that goldens are required only when the touched surface affects them. The fix changes only CLI option binding and a single banner string; XGBoost behavior, training paths, scoring, and bundle layout are all unchanged.
- **Workbench SSH end-to-end run**: **not run** — PR-039 is CPU / option-parsing only; the local Mac `CliRunner` integration tests exercise the full code path. Per `feedback_local_mac_compute_default`, workbench is reserved for CUDA-only gates. The pending `calibration/crypto-h3-hpo-9dim` work (held until this PR merges) is the canonical real-data validation of the fix and will run on the workbench when it frees up.

**PR link**: <https://github.com/rux-eth/rux-ml/pull/45>

**Exit criteria**: all bugs verified green via Phase-5 tests; all project gates pass; PR link delivered.

## Verification criteria

_Populated post-Phase 4. Each criterion maps 1:1 to a bug above and to a Phase 4 test._

- [x] Bug 1a: `tune start` invoked without `--n-trials` against `[tuning] n_trials = 2` runs exactly 2 trials and banner reports `n_trials: 2`. Test: `tests/cli/test_tune_subcommands.py::test_tune_start_honors_tuning_n_trials_from_toml`.
- [x] Bug 1a regression guard: `tune start --n-trials 3` against `[tuning] n_trials = 2` runs 3 trials and banner reports `n_trials: 3`. Test: `tests/cli/test_tune_subcommands.py::test_tune_start_cli_flag_overrides_toml_n_trials`.
- [x] Bug 1b: `tune resume` invoked without `--n-trials` against `[tuning] n_trials = 2` runs exactly 2 additional trials and banner reports `n_trials: 2`. Test: `tests/cli/test_tune_subcommands.py::test_tune_resume_honors_tuning_n_trials_from_toml`.
- [x] Bug 1b regression guard: `tune resume --n-trials 3` against `[tuning] n_trials = 2` runs 3 trials and banner reports `n_trials: 3`. Test: `tests/cli/test_tune_subcommands.py::test_tune_resume_cli_flag_overrides_toml_n_trials`.

## Verification run

Run date: **2026-05-31** (local Mac M4 Max, CPU mode per `feedback_local_mac_compute_default`).

- `uv run pytest` (default suite — excludes `gpu`/`slow`/`golden` markers): **407 passed, 1 skipped, 15 deselected in 33.23 s** (+4 from pre-PR 403; matches the four new tests added in Phase 5; no regressions).
- `uv run pytest tests/cli/test_tune_subcommands.py -k "n_trials_from_toml or cli_flag_overrides_toml"` (PR-039 targeted): **4 passed in 4.26 s** (all four green — 2 RED tests turned green by the fix; 2 regression guards stayed green).
- `uv run ruff check .`: **All checks passed!**
- `uv run basedpyright src/`: **0 errors, 0 warnings, 0 notes**.
- `make test-golden`: **not run** (touched surface — CLI option binding + a single banner string — does not affect the golden contract; XGBoost behavior, training paths, scoring, bundle layout all unchanged).
- Workbench SSH end-to-end: **not run** (PR-039 is CPU / option-parsing only; the local-Mac integration tests exercise the full code path. The pending `calibration/crypto-h3-hpo-9dim` work is the canonical real-data validation and runs on the workbench when it frees up per `feedback_local_mac_compute_default`).
