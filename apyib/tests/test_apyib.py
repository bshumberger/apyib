"""
Unit and regression test for the apyib package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import apyib


def test_apyib_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "apyib" in sys.modules
