"""Shared fixtures — deterministic synthetic book-tax-difference panels with a *known* planted
return effect, so tests never touch the network. The only thing the tercile long-short can harvest
(low-BTD-minus-high-BTD forward-return predictability) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from book_tax_diff import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No effect (BTD carries no return info). The tercile long-short must not manufacture t."""
    return data.synthetic_panel(n_names=42, n_years=16, edge=0.0, seed=856)


@pytest.fixture
def planted_panel():
    """A strong planted Hanlon effect: low-BTD names drift up, high-BTD names drift down."""
    return data.synthetic_panel(n_names=42, n_years=16, edge=0.35, seed=856)
