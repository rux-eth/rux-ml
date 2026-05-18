# PR-003: CLI skeleton

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time).

## Research findings

### State Assessment (2026-05-16)

**Current state of the codebase**:

- `dev` at commit `1aed7c9` (PR-002 merged: config layer + hashing + 26 tests)
- `src/rux_ml/` has `__init__.py`, `py.typed`, `_internal/` (hashing.py), `config/` (8 modules including SearchSpec); no `cli/` subpackage yet
- `pyproject.toml` already declares `typer>=0.12`; installed version is **0.25.1**
- `[project.scripts]` deliberately deferred from PR-001 to this PR — clean slate for the entry point
- No `__main__.py` yet
- `tests/config/` has 26 passing tests from PR-002; no CLI tests yet

**Assumptions at PR draft time** (drafted 2026-05-15 from D11, D15):

- Typer as the CLI framework with `app.add_typer(sub_app, name="...")` for nested groups
- `@app.callback()` for global flags (`--config`, `--problem`, `--study`, `--verbose`, `--dry-run`, `--profile`)
- Dot-path overrides (`--training.lr=0.05`) — Typer has no native support, plan was a `--set key=value` list translated to the `overrides` mapping accepted by `RuxMLConfig.from_layers` (PR-002 confirmed this signature)
- `__main__.py` invoking `cli.app()`
- `[project.scripts]` entry point: `rux-ml = "rux_ml.cli:app"`
- Shell completion install via `rux-ml --install-completion`
- `typer.testing.CliRunner` for tests

**Verification against current authoritative sources**:

| Item | Status | Source |
|---|---|---|
| Typer 0.25.x active and Click-based | **STILL CURRENT** | [Typer PyPI](https://pypi.org/project/typer/) |
| `app.add_typer(sub_app, name="...")` pattern | **STILL CURRENT** | [add-typer tutorial](https://typer.tiangolo.com/tutorial/subcommands/add-typer/) |
| `@app.callback()` for global flags | **STILL CURRENT** | [callback tutorial](https://typer.tiangolo.com/tutorial/commands/callback/) |
| `--install-completion` (shell autodetected) | **DRIFT FLAGGED (cosmetic)** | [package tutorial](https://typer.tiangolo.com/tutorial/package/) — modern Typer auto-detects shell; explicit `bash\|zsh\|fish` positional optional |
| `typer.testing.CliRunner` invocation API | **STILL CURRENT** | [testing tutorial](https://typer.tiangolo.com/tutorial/testing/) |
| `[project.scripts]` → Typer instance, no wrapper | **STILL CURRENT** | [package tutorial](https://typer.tiangolo.com/tutorial/package/) |
| Dot-path overrides via collected `--set key=value` list | **STILL CURRENT (no native support)** | [typer-config](https://pypi.org/project/typer-config/), [typer #86](https://github.com/fastapi/typer/issues/86) |
| `Annotated[Type, typer.Option(...)]` preferred over legacy syntax | **STILL CURRENT (preferred)** | [optional args tutorial](https://typer.tiangolo.com/tutorial/arguments/optional/) |
| Click ≥ 8.2.1 pinned by Typer 0.25.1, no caveats | **STILL CURRENT** | [Typer pyproject.toml](https://github.com/fastapi/typer/blob/master/pyproject.toml) |
| `rich` help formatting on by default | **STILL CURRENT** | [commands/help tutorial](https://typer.tiangolo.com/tutorial/commands/help/) |

**Stale assumptions**: None of substance. One cosmetic-only drift: `--install-completion` now auto-detects shell.

**New constraints learned from prior PRs or codebase evolution**:

- PR-002 locked `RuxMLConfig.from_layers(base, *, problem, study, problems_dir, studies_dir, overrides)` as the loader signature. PR-003's CLI must translate `--set key=value` flags + `--config` + `--problem` + `--study` into this exact call shape.
- `extra="forbid"` is enforced on every Pydantic model, so PR-003's `--set` translator must use only valid dot-paths into the schema; the loader will surface typos as `ValidationError`.

**Synthesis Outcome: CONFIRM**

Zero substantive drift. One cosmetic detail (`--install-completion` no longer requires a shell positional) — apply at implementation time. No amendments needed.

### Synthesis (2026-05-16)

**Outcome:** Confirm.

**Changes to this PR from research:** Drop the `bash|zsh|fish` positional from the `--install-completion` invocation in docs/help text. Cosmetic only.

**Changes to ARCHITECTURE.md / CONSTRAINTS.md / CONVENTIONS.md / CLAUDE.md:** None.

**New PRs that must come first:** None.

**Research-backed details now locked in this PR:**
- Typer 0.25.x with `Annotated[Type, typer.Option(...)]` syntax everywhere
- `app.add_typer(sub_app, name="...")` for `data`, `train`, `tune`, `runs`, `registry`
- `@app.callback()` on the root app for global flags
- `--set key=value` repeatable option → list of strings → translated to `overrides={...}` passed to `RuxMLConfig.from_layers`
- `[project.scripts]`: `rux-ml = "rux_ml.cli:app"`
- `typer.testing.CliRunner` for tests
- `__main__.py` exposing `python -m rux_ml`

### Gate Check

- Premise still valid: ✓ (no drift, no architectural rework)
- No prerequisite PRs surfaced: ✓
- User approved updated spec: ✓ (2026-05-16)
- Implementation cleared: ✓ (2026-05-16)

### Implementation notes (2026-05-16)

Two minor tidyings surfaced during implementation, neither research-driven:

1. **Dropped `--profile` global flag.** The original PR-003 plan listed `--profile <name>` ("selects a named profile section in TOML"), but D17 explicitly rejected single-file profile sections in favor of the three-tier `base/problems/<n>/studies/<n>` file layout. `--profile` would be redundant with `--problem` / `--study` and confusing. Not present in the implementation.
2. **Per-file ruff ignore for `src/rux_ml/cli/**/*.py`** disables TC001/TC002/TC003 for the CLI subpackage. Reason: Typer uses `inspect.signature(..., eval_str=True)` to introspect parameter annotations at registration time; moving `typer` or `pathlib.Path` into a `TYPE_CHECKING` block makes string annotations like `"typer.Context"` unevaluable at runtime, causing `NameError` and breaking CLI parameter binding. This is a Typer architectural constraint, not a project-style choice.

Both notes mirrored in the commit message and `docs/CONVENTIONS.md` (CLI ruff exception added under Code Style).

---

## Scope

Wire the Typer CLI per D11 / D15 with all verb groups present as no-op subcommands. The CLI must show its full surface area so users can navigate it; subcommand bodies land in their respective layer PRs.

- `src/rux_ml/cli/__init__.py` instantiating the root `app = Typer(...)` and `app.add_typer(...)` for each verb group
- `src/rux_ml/cli/{data,train,tune,runs,registry}.py` each defining a `Typer()` sub-app with subcommands per `docs/ARCHITECTURE.md` Data Flow + the CLI surface in D11. Each subcommand body just prints `"<command>: not yet implemented (PR-XXX)"` referencing the PR that will implement it.
- Global flags: `--config` (default `configs/base.toml`), `--problem`, `--study`, `--<dot.path>=<value>` overrides, `--verbose`, `--dry-run`, `--profile` (named profile section in TOML)
- `src/rux_ml/__main__.py` calling `cli.app()`
- `pyproject.toml` `[project.scripts]` exposing `rux-ml = "rux_ml.cli:app"`
- Shell completion install via `rux-ml --install-completion`
- Tests: `typer.testing.CliRunner` invokes `rux-ml --help`, `rux-ml --version`, `rux-ml data --help`, every group's `--help`, and at least one no-op subcommand to confirm the wiring

NOT in scope: any subcommand actually doing work.

## Dependencies

PR-002.

## Architecture section implemented

`docs/ARCHITECTURE.md` → "CLI" component row, "Data Flow" (the CLI is the entry point), and `docs/0.0/DESIGN-log.md` D11.

## Verification criteria

- [ ] `rux-ml --version` prints the package version
- [ ] `rux-ml --help` shows all five verb groups (`data`, `train`, `tune`, `runs`, `registry`)
- [ ] Each verb group `--help` shows its subcommands per `docs/ARCHITECTURE.md`
- [ ] Each no-op subcommand exits 0 with a "not yet implemented (PR-XXX)" message
- [ ] `--config`, `--problem`, `--study`, `--<dot.path>=<value>` flags accepted (validation deferred to layer PRs that need them)
- [ ] Shell completion install command works (`--install-completion bash`)
- [ ] Tests via `CliRunner` cover all of the above

## Research backing

Tier 1:

- D11: [pydantic-settings CLI](https://docs.pydantic.dev/latest/concepts/pydantic_settings/), [vLLM CLI](https://docs.vllm.ai/en/latest/cli/), [Optuna CLI](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/004_cli.html), [Typer add_typer](https://typer.tiangolo.com/tutorial/subcommands/add-typer/)
- D15: `cli/` subpackage with one module per verb group is convention when verb-group count > ~3

State assessment must verify Typer's `add_typer` pattern is unchanged and shell completion install works in the chosen Typer version.

## Notes

- Don't pull `pydantic-settings v2 CliApp` into this PR — D11 explicitly chose Typer + pydantic-settings as the more battle-tested combo. CliApp can be revisited later.
- The "not yet implemented" message must include the PR number that will land it, so users (and future Claude) can trace work clearly.
