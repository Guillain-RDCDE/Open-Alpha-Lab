"""Shared fixtures — the deterministic synthetic predictable series and its random-walk null, no network.

Kept small (short series, small net, few walk-forward folds) so the whole suite runs in well under a
minute on CI."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
# This dir too: the test module does `from conftest import FAST, SEED, WF`, which under
# pytest's importlib import-mode is a plain import and needs tests/ on sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from black_box import data

SEED = 39

# Small, fast knobs reused across tests — a net capable enough to overfit in-sample (the trap)
# yet quick (few folds, short series). The whole suite runs in well under a minute.
FAST = dict(hidden=(16, 8), max_iter=250)
WF = dict(min_train=400, step=200, **FAST)


@pytest.fixture
def signal():
    """A predictable series strong enough to recover OOS on a short sample (and inflate in-sample)."""
    return data.synthetic_predictable(predictable_strength=0.015, n_bars=1200, seed=SEED)


@pytest.fixture
def null():
    return data.synthetic_predictable(predictable_strength=0.0, n_bars=1200, seed=SEED)


@pytest.fixture
def signal_close(signal):
    return signal[0]["close"]


@pytest.fixture
def null_close(null):
    return null[0]["close"]
