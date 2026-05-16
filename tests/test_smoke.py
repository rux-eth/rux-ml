"""Smoke tests asserting the package is importable and exposes its version."""

import rux_ml


def test_package_imports() -> None:
    assert rux_ml is not None


def test_version_is_a_nonempty_string() -> None:
    assert isinstance(rux_ml.__version__, str)
    assert rux_ml.__version__ != ""


def test_public_api_includes_version() -> None:
    assert "__version__" in rux_ml.__all__
