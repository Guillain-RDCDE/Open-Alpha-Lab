"""Shared fixtures for Study 197 (Dividend-Payout-Ratio).

The synthetic tape is fully deterministic and offline: tests only touch
``_cache/shiller_sp500.parquet`` in the real-data tests, which are guarded with
``@requires_cache`` so they skip cleanly on CI runners where the cache is absent.
"""

import os
import sys

import pytest

# Make the study package importable from tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dividend_payout_ratio import data  # noqa: E402

# ---------------------------------------------------------------------------
# Cache guard — real data tests SKIP if the shared Shiller cache is absent.
# Defined here so pytest picks it up automatically; test files also define it
# locally (same condition) so the decorator is self-contained in each module.
# ---------------------------------------------------------------------------
_SHILLER_PATH = data.SHILLER_CACHE

requires_cache = pytest.mark.skipif(
    not os.path.exists(_SHILLER_PATH),
    reason="Shiller cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Synthetic-tape fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def null_tape():
    """No payout signal (payout_signal=0): the null — slope should be ~0."""
    df, truth = data.synthetic_monthly(n_years=100, payout_signal=0.0, seed=197)
    return df, truth


@pytest.fixture
def signal_tape():
    """Positive payout signal planted (payout_signal=0.10): slope should be detectable."""
    df, truth = data.synthetic_monthly(n_years=100, payout_signal=0.10, seed=197)
    return df, truth
