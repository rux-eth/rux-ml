# PR-003: CLI skeleton

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time).

## Research findings

_To be populated by `PROCEDURE-pr-research.md`._

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

`docs/ARCHITECTURE.md` → "CLI" component row, "Data Flow" (the CLI is the entry point), and `docs/DESIGN-log.md` D11.

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
