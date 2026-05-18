# PR-002: Config layer skeleton

**Landed-in:** v0.0.1

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

## Research findings

### State Assessment (2026-05-15)

**Current state of the codebase**:

- `dev` at commit `c39f617` (PR-001 merged: scaffold + uv_build + ruff + basedpyright + pytest)
- `src/rux_ml/` exists with `__init__.py` (curated `__all__`) + `py.typed`; **no subpackages yet** — `src/rux_ml/config/` and `src/rux_ml/_internal/` will be created by this PR
- `pyproject.toml` already includes `pydantic>=2.6` and `pydantic-settings>=2.6` as runtime deps
- `configs/` has only the README (per PR-001); no `base.toml` yet
- `tests/` has the smoke tests from PR-001; no config-layer tests yet

**Assumptions at PR draft time** (drafted 2026-05-15 from D2, D16, D17):

- `pydantic-settings v2` with `TomlConfigSettingsSource(deep_merge=True)` and a list-valued `toml_file=[...]`
- `settings_customise_sources` returns a tuple ordered (highest → lowest priority): CLI, init, env, dotenv, toml, file_secrets
- `CliSettingsSource` for dot-path overrides (`--training.lr=0.01`)
- Env nesting via `env_nested_delimiter="__"`; prefix `RUXML_`
- `extra="forbid"` on every model
- `model_copy(update=overrides, deep=True)` for D16's trial config derivation
- `Annotated[Union[FloatSpec, IntSpec, CatSpec], Field(discriminator="type")]` for `SearchSpec`
- stdlib `tomllib` for parsing (Python ≥ 3.11 / 3.12 in this project)
- `xxhash` Python binding for fast non-cryptographic hashing in `_internal/hashing.py`

**Verification against current authoritative sources**:

| Item | Status | Source |
|---|---|---|
| `pydantic-settings v2` package status | **STILL CURRENT** | [pydantic-settings docs](https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/) |
| `TomlConfigSettingsSource(toml_file=[...], deep_merge=True)` | **STILL CURRENT** | [source](https://github.com/pydantic/pydantic-settings/blob/main/pydantic_settings/sources/providers/toml.py) |
| `settings_customise_sources` 5-arg signature | **STILL CURRENT** | [pydantic-settings docs](https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/) |
| `CliSettingsSource` dot-path overrides | **STILL CURRENT** | [pydantic-settings docs](https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/) |
| `env_prefix` + `env_nested_delimiter="__"` | **STILL CURRENT** | [pydantic-settings docs](https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/) |
| `extra="forbid"` on `BaseSettings` | **STILL CURRENT** (in fact default for BaseSettings) | [pydantic-settings docs](https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/) |
| `model_copy(update=..., deep=True)` | **STILL CURRENT** | [Pydantic models docs](https://pydantic.dev/docs/validation/latest/concepts/models/) |
| stdlib `tomllib` (no `tomli` needed on 3.11+) | **STILL CURRENT** | [pydantic-settings TOML source](https://github.com/pydantic/pydantic-settings/blob/main/pydantic_settings/sources/providers/toml.py) |
| `xxhash` Python package maintenance | **STILL CURRENT** (v3.7.0 released 2026-04-25, supports Python 3.8–3.14 incl. free-threading) | [xxhash on PyPI](https://pypi.org/project/xxhash/) |
| `Annotated[Union[...], Field(discriminator="type")]` | **STILL CURRENT** | [Pydantic unions docs](https://pydantic.dev/docs/validation/latest/concepts/unions/) |

**Stale assumptions**: None.

**New constraints learned from prior PRs or codebase evolution**:

- `pyproject.toml` uses PEP 735 `[dependency-groups]` (set in PR-001 after the deprecation warning fix). Any new dev deps added by PR-002 (e.g. `xxhash`) go under `[project.dependencies]` if used at runtime in `_internal/hashing.py`, or `[dependency-groups] dev` if test-only.
- `xxhash` will be a runtime dep (used by `data/versioning.py` in PR-004 too) — declare it in `[project.dependencies]`.

**Synthesis Outcome: CONFIRM**

Zero drift across all 10 verification items. The original PR-002 scope, verification criteria, and architectural assumptions are all current as of 2026-05-15. No amendments needed; proceed to implementation.

### Synthesis (2026-05-15)

**Outcome:** Confirm.

**Changes to this PR from research:** None to scope; one trivial addition — `xxhash>=3.7` declared as a runtime dependency in `pyproject.toml` (rather than a dev dep) since it's used by `_internal/hashing.py` in this PR and will be reused by PR-004.

**Changes to ARCHITECTURE.md / CONSTRAINTS.md / CONVENTIONS.md / CLAUDE.md:** None.

**New PRs that must come first:** None.

**Research-backed details now locked in this PR:**
- `pydantic-settings v2` `TomlConfigSettingsSource(toml_file=[...], deep_merge=True)`
- `settings_customise_sources` 5-arg signature returning tuple ordered highest → lowest
- `CliSettingsSource` for dot-path overrides; env via `env_nested_delimiter="__"` + `RUXML_` prefix
- `Annotated[Union[FloatSpec, IntSpec, CatSpec], Field(discriminator="type")]` for `SearchSpec`
- stdlib `tomllib`; no `tomli` extra needed on Python 3.12
- `xxhash>=3.7` as runtime dep

### Gate Check

- Premise still valid: ✓ (no drift, no architectural rework)
- No prerequisite PRs surfaced: ✓
- User approved updated spec: ✓ (2026-05-15)
- Implementation cleared: ✓ (2026-05-15)

### Implementation note (2026-05-15)

One API ergonomics tweak surfaced during implementation, not a research-driven amendment: `from_layers` now takes `overrides: Mapping[str, Any] | None = None` (a single keyword param) instead of `**cli_overrides` (variadic kwargs). Reason: dot-path keys like `"training.learning_rate"` confused basedpyright's strict checker, which tried to match each key against the function's positional parameters. Type-safe call site is `RuxMLConfig.from_layers(base, overrides={"training.lr": 0.05})`. The PR-003 CLI will translate `--training.lr=0.05` flags into the overrides mapping.

---

## Scope

Lay down the configuration architecture from D2 + D17 — schema in code, layered TOML on disk, override precedence wired up.

- `src/rux_ml/config/__init__.py` re-exports `RuxMLConfig` + each per-layer config via `__all__`
- `src/rux_ml/config/root.py` with `RuxMLConfig(BaseSettings)`, `settings_customise_sources` returning `(cli, init, env, dotenv, toml, file_secrets)` ordered, env prefix `RUXML_`, `extra="forbid"`, hash-elision constant `_HASH_ELIDED_FIELDS`
- `src/rux_ml/config/{data,features,training,tuning,runs,registry,memory}.py` each with a Pydantic model containing only the fields required by later PRs (placeholder defaults marked with TOML knob comments)
- `src/rux_ml/config/tuning.py` includes `SearchSpec` tagged-union (`FloatSpec` / `IntSpec` / `CatSpec`) per D16
- TOML loader entry point: `RuxMLConfig.from_layers(base, problem=None, study=None, **cli_overrides)` resolving the file list passed to `TomlConfigSettingsSource(deep_merge=True)`
- `configs/base.toml` with a minimal complete config (every layer has the fields needed by the smoke tests below)
- `configs/README.md` documenting the three-tier composition + override precedence
- Hashing helpers in `src/rux_ml/_internal/hashing.py` (uses `xxhash` for bytes; canonical JSON via `json.dumps(sort_keys=True)`); helpers `cfg_hash(cfg, exclude=...)` and `layer_cfg_hash(cfg, layer)` returning hex strings
- Tests: load `configs/base.toml`, validate, assert overrides via env + dict CLI win in priority order; assert `extra="forbid"` raises on unknown key; assert per-layer + root hashes are stable across reloads of identical TOML

NOT in scope: data/features/training implementations (later PRs), CLI subcommands (PR-003), search-space walker (PR-007).

## Dependencies

PR-001.

## Architecture section implemented

`docs/ARCHITECTURE.md` → "Configuration Architecture", "Key Abstractions: `RuxMLConfig` and `SearchSpec`", and the `_HASH_ELIDED_FIELDS` discussion.

## Verification criteria

- [ ] `RuxMLConfig.from_layers("configs/base.toml")` returns a fully-validated config
- [ ] Layered loading: `from_layers(base, problem="dummy", study="dummy")` overlays in order
- [ ] CLI overrides beat env vars; env vars beat study TOML; study beats problem; problem beats base
- [ ] Unknown TOML keys raise `pydantic.ValidationError` (`extra="forbid"`)
- [ ] `RUXML_TRAINING__LEARNING_RATE=0.01` env override works
- [ ] `cfg_hash(cfg)` is deterministic across reloads; differs when a non-elided field changes; identical when only an elided field (path, timestamp) changes
- [ ] `layer_cfg_hash(cfg, "training")` differs only when the `training` layer changes
- [ ] `SearchSpec` discriminated union round-trips through TOML

## Research backing

Tier 1:

- D2: [pydantic-settings docs](https://docs.pydantic.dev/latest/concepts/pydantic_settings/), conflict re Hydra stagnation
- D16: [Pydantic frozen `model_copy`](https://github.com/pydantic/pydantic/discussions/4250), [Hydra Optuna Sweeper precedent](https://hydra.cc/docs/plugins/optuna_sweeper/)
- D17: [pydantic-settings configuration files](https://deepwiki.com/pydantic/pydantic-settings/3.2-configuration-files), [DVC repro](https://dvc.org/doc/command-reference/repro), [MLflow datasets](https://mlflow.org/docs/latest/python_api/mlflow.data.html)

State assessment must verify the `TomlConfigSettingsSource(deep_merge=True)` API is still present and the source-priority hook signature has not changed.

## Notes

- Per D17, lists in TOML *replace* on overlay; document in `configs/README.md` so users don't expect concatenation.
- `data_cfg_hash` in this PR is the **config layer** hash; `data_hash` (dataset content) lands in PR-004. Naming is intentional per `docs/ARCHITECTURE.md` "Cross-Decision Consistency Notes".
