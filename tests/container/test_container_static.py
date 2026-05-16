"""Static regression gates for the container artifacts (per PR-012).

These tests always run — they protect against regressions like dropping the
``@sha256:`` pin, reverting to the obsolete ``gpus: all`` Compose syntax, or
shipping a ``.dockerignore`` that leaks runtime directories into the build
context. They read the files from disk; no Docker daemon required.

The build + run smoke tests live in ``test_container_smoke.py`` and are
gated behind ``RUXML_RUN_DOCKER_TESTS`` + a working ``docker`` binary.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"
COMPOSE = REPO_ROOT / "docker-compose.yml"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"


# ---------------------------------------------------------------------------
# Dockerfile — digest pinning + system-dep hygiene
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def test_dockerfile_exists() -> None:
    assert DOCKERFILE.is_file(), f"missing {DOCKERFILE}"


def test_dockerfile_base_image_is_digest_pinned(dockerfile_text: str) -> None:
    """`docs/CONSTRAINTS.md` "Container Digest Pinning" requires `@sha256:...`
    on every base image — tags can be rebuilt or removed."""
    from_lines = [line for line in dockerfile_text.splitlines() if line.strip().startswith("FROM ")]
    assert from_lines, "no FROM line in Dockerfile"
    for line in from_lines:
        assert "@sha256:" in line, f"FROM not digest-pinned: {line!r}"
        # 64-hex digest after sha256:
        assert re.search(r"@sha256:[0-9a-f]{64}\b", line), (
            f"sha256 digest is not 64 hex chars: {line!r}"
        )


def test_dockerfile_base_image_is_nvidia_cuda(dockerfile_text: str) -> None:
    """D1 fixed the base as nvidia/cuda:12.4.1-devel-ubuntu22.04. Changing this
    is a Phase-3 research event, not a silent edit."""
    assert "nvidia/cuda:12.4.1-devel-ubuntu22.04" in dockerfile_text


def test_dockerfile_uv_image_is_digest_pinned(dockerfile_text: str) -> None:
    """The uv copy-in stage must also be digest-pinned (same constraint).

    Skips comment lines that mention the image purely for documentation —
    only actual COPY directives count.
    """
    uv_directives = [
        line
        for line in dockerfile_text.splitlines()
        if "ghcr.io/astral-sh/uv" in line and not line.lstrip().startswith("#")
    ]
    assert uv_directives, "no COPY --from=ghcr.io/astral-sh/uv directive in Dockerfile"
    for line in uv_directives:
        assert "@sha256:" in line, f"uv image not digest-pinned: {line!r}"
        assert re.search(r"@sha256:[0-9a-f]{64}\b", line), (
            f"uv image sha256 digest is not 64 hex chars: {line!r}"
        )


def test_dockerfile_does_not_apt_install_xxhash(dockerfile_text: str) -> None:
    """The Python `xxhash` wheel bundles its C extension — no system pkg
    needed. Phase-1 finding (PR-012): drop the apt dep to keep the image
    surface lean."""
    # apt-install lines (one combined RUN block in our Dockerfile) must
    # not contain a bare ``xxhash`` token.
    pattern = re.compile(r"\bxxhash\b")
    apt_section = False
    for line in dockerfile_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("RUN apt-get"):
            apt_section = True
        if apt_section:
            assert not pattern.search(line), f"apt installs `xxhash`: {line!r}"
            if not stripped.endswith("\\") and not stripped.startswith("&&"):
                # End of the multi-line RUN block.
                apt_section = False


def test_dockerfile_runs_as_non_root_user(dockerfile_text: str) -> None:
    """Volume-mounted runtime dirs (./studies, ./registry, ...) round-trip
    to the host as uid 1000 — drop privileges in-image."""
    assert "USER rux" in dockerfile_text
    assert "useradd" in dockerfile_text
    assert "--uid 1000" in dockerfile_text


def test_dockerfile_uses_tini_as_entrypoint(dockerfile_text: str) -> None:
    """tini PID 1 forwards SIGTERM/SIGINT to rux-ml and its subprocess-per-trial
    children (PR-008 spawn-semantics requirement)."""
    assert "/usr/bin/tini" in dockerfile_text
    # ENTRYPOINT must invoke tini explicitly, not rely on PATH.
    entrypoint_lines = [
        line for line in dockerfile_text.splitlines() if line.strip().startswith("ENTRYPOINT")
    ]
    assert entrypoint_lines, "no ENTRYPOINT in Dockerfile"
    assert any("tini" in line for line in entrypoint_lines)


def test_dockerfile_uses_uv_sync_locked(dockerfile_text: str) -> None:
    """PR-012 AMEND 2: --locked supersedes --frozen in current Astral docs."""
    assert "uv sync --locked" in dockerfile_text
    assert "uv sync --frozen" not in dockerfile_text


# ---------------------------------------------------------------------------
# docker-compose.yml — GPU reservation + memory caps
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compose_text() -> str:
    return COMPOSE.read_text(encoding="utf-8")


def test_compose_exists() -> None:
    assert COMPOSE.is_file(), f"missing {COMPOSE}"


def test_compose_uses_deploy_resources_reservation_devices(compose_text: str) -> None:
    """PR-012 AMEND 1: the legacy ``gpus: all`` Compose service key is no
    longer documented in Docker docs (2026). Canonical GPU passthrough lives
    under ``deploy.resources.reservations.devices``."""
    assert "deploy:" in compose_text
    assert "resources:" in compose_text
    assert "reservations:" in compose_text
    assert "devices:" in compose_text
    assert "driver: nvidia" in compose_text
    assert "capabilities: [gpu]" in compose_text or "capabilities:\n" in compose_text


def test_compose_does_not_use_obsolete_gpus_all_key(compose_text: str) -> None:
    """Regression gate against reverting AMEND 1."""
    for line in compose_text.splitlines():
        stripped = line.strip()
        # `gpus: all` as a service-level key is the obsolete pattern.
        if stripped.startswith("gpus:"):
            pytest.fail(f"obsolete `gpus:` service key present: {line!r}")


def test_compose_has_memory_caps(compose_text: str) -> None:
    """D10 memory budget: 32 GB hard cap + 28 GB soft cap. Watchdog threshold
    in MemoryConfig is 28 GB — the two must match."""
    assert "mem_limit: 32g" in compose_text
    assert "mem_reservation: 28g" in compose_text


def test_compose_sets_workbench_home(compose_text: str) -> None:
    """The container exposes WORKBENCH_HOME=/workbench so future PRs that
    consume CONVENTIONS.md's path-root convention work out of the box."""
    assert "WORKBENCH_HOME" in compose_text
    assert "/workbench" in compose_text


def test_compose_mounts_runtime_dirs(compose_text: str) -> None:
    """Trials, registry artifacts, and logs must persist outside the
    container — every mutable runtime path is bind-mounted from host."""
    for path in ("./configs", "./studies", "./registry", "./data", "./logs"):
        assert path in compose_text, f"missing host bind mount for {path}"


# ---------------------------------------------------------------------------
# .dockerignore — build context hygiene
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dockerignore_text() -> str:
    return DOCKERIGNORE.read_text(encoding="utf-8")


def test_dockerignore_exists() -> None:
    assert DOCKERIGNORE.is_file(), f"missing {DOCKERIGNORE}"


def test_dockerignore_excludes_runtime_dirs(dockerignore_text: str) -> None:
    """Anchored entries (leading slash) must exclude top-level workbench
    runtime dirs from the build context — they're host bind mounts, never
    baked into the image."""
    entries = {line.strip() for line in dockerignore_text.splitlines() if line.strip()}
    for path in ("/studies", "/registry", "/data", "/logs"):
        assert path in entries, f".dockerignore must exclude {path}"


def test_dockerignore_excludes_venv_and_caches(dockerignore_text: str) -> None:
    """A stale local .venv leaking into the build context would bloat the
    image by hundreds of megabytes and conflict with the in-image venv."""
    entries = {line.strip() for line in dockerignore_text.splitlines() if line.strip()}
    for entry in (".venv", ".pytest_cache", ".ruff_cache", ".basedpyright_cache", ".hypothesis"):
        assert entry in entries, f".dockerignore must exclude {entry}"


def test_dockerignore_excludes_git_metadata(dockerignore_text: str) -> None:
    """No .git in the image — git_sha is computed at host-side at trial time
    and recorded in TrialAttrs.git_sha, not derived from in-image state."""
    entries = {line.strip() for line in dockerignore_text.splitlines() if line.strip()}
    assert ".git" in entries
