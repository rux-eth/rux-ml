"""End-to-end container smoke tests (per PR-012).

Gated behind ``RUXML_RUN_DOCKER_TESTS=1`` and a working ``docker`` binary so
``uv run pytest`` stays green on a Mac dev machine. On the Linux workbench:

    RUXML_RUN_DOCKER_TESTS=1 uv run pytest -m docker

This is the only place the actual image build is exercised. The
``test_container_static.py`` regression gates run unconditionally and protect
the Dockerfile / Compose / .dockerignore from regressions without needing
Docker.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.docker


_SKIP_REASON = (
    "docker smoke tests skipped — set RUXML_RUN_DOCKER_TESTS=1 and ensure "
    "`docker` is on PATH (run on the Linux workbench, not the Mac dev side)"
)


def _docker_tests_enabled() -> bool:
    return os.environ.get("RUXML_RUN_DOCKER_TESTS") == "1" and shutil.which("docker") is not None


@pytest.fixture(scope="session")
def built_image() -> str:
    """Build rux-ml:local once per session and return the image digest.

    Skips the whole module's tests when Docker isn't available — gating
    happens here so basedpyright doesn't flag an unused fixture.
    """
    if not _docker_tests_enabled():
        pytest.skip(_SKIP_REASON)
    result = subprocess.run(
        ["make", "docker-build"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"`make docker-build` failed (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    digest_file = REPO_ROOT / ".docker-image-digest"
    assert digest_file.is_file(), "docker-build did not produce .docker-image-digest"
    digest = digest_file.read_text(encoding="utf-8").strip()
    assert digest.startswith("sha256:"), f"unexpected digest format: {digest!r}"
    return digest


def test_image_builds_and_records_digest(built_image: str) -> None:
    assert built_image.startswith("sha256:")
    assert len(built_image) == len("sha256:") + 64  # 64 hex chars


def test_image_prints_version(built_image: str) -> None:
    _ = built_image
    result = subprocess.run(
        ["docker", "compose", "run", "--rm", "rux-ml", "--version"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"--version failed (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "0." in result.stdout, f"version output looks wrong: {result.stdout!r}"


def test_image_help_shows_all_verb_groups(built_image: str) -> None:
    _ = built_image
    result = subprocess.run(
        ["docker", "compose", "run", "--rm", "rux-ml", "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
    # Per CONVENTIONS.md "CLI verbs": data, train, tune, runs, registry.
    for verb in ("data", "train", "tune", "runs", "registry"):
        assert verb in result.stdout, (
            f"verb group {verb!r} missing from --help output:\n{result.stdout}"
        )


def test_image_memory_cap_is_32gb(built_image: str) -> None:
    """The cgroup-enforced hard cap must read as 32 GB from inside the
    container — D10 reproducibility requirement."""
    _ = built_image
    result = subprocess.run(
        [
            "docker",
            "compose",
            "run",
            "--rm",
            "--entrypoint",
            "cat",
            "rux-ml",
            "/sys/fs/cgroup/memory.max",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"failed to read memory.max:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # 32 GiB = 34_359_738_368 bytes; Docker's `mem_limit: 32g` is interpreted as GiB.
    memory_max_bytes = int(result.stdout.strip())
    expected = 32 * 1024**3
    assert memory_max_bytes == expected, (
        f"memory.max = {memory_max_bytes} bytes, expected {expected} (32 GiB)"
    )


def test_image_python_is_312(built_image: str) -> None:
    """The uv-managed Python in the venv must be 3.12.x per pyproject.toml
    `requires-python = ">=3.12"`."""
    _ = built_image
    result = subprocess.run(
        [
            "docker",
            "compose",
            "run",
            "--rm",
            "--entrypoint",
            "/workbench/.venv/bin/python",
            "rux-ml",
            "--version",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    # `python --version` writes to stdout on Python 3.4+.
    output = result.stdout or result.stderr
    assert "Python 3.12" in output, f"expected Python 3.12.x, got: {output!r}"
