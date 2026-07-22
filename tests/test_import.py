"""Tests for the initial package structure."""

from importlib import import_module


def test_bih_guide_can_be_imported() -> None:
    """The project package should be importable after editable installation."""
    package = import_module("bih_guide")

    assert package.__name__ == "bih_guide"
