"""Test the package itself."""

import importlib.metadata

import potamides as ptd


def test_version():
    assert importlib.metadata.version("potamides") == ptd.__version__
