"""The synthetic panel is deterministic; the long-high-yield hedge pays when the premium is planted;
the null shows no edge; the hedge is long minus short; the net buyback yield is sign-flipped issuance;
and real-tape tests are skipped when the EDGAR cache is absent (offline / CI)."""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from shareholder_yield import data, strategy as st

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SH_CACHE = os.path.join(_REPO_ROOT, "_cache",
                          "_edgar_WeightedAverageNumberOfDilutedSharesOutstanding.parquet")
_RET_CACHE = os.path.join(_REPO_ROOT, "_cache", "_edgar_yrret.parquet")
_CACHE_PRESENT = os.path.exists(_SH_CACHE) and os.path.exists(_RET_CACHE)


# ---------------------------------------------------------------------------
# Offline / synthetic tests
# ---------------------------------------------------------------------------

def test_world_deterministic(yield_world):
    """Same seed always produces the same panel."""
    sig, fwd, truth = yield_world
    sig2, fwd2, _ = data.synthetic_panel(yield_premium=0.06, seed=233)
    assert np.allclose(sig.to_numpy(), sig2.to_numpy())
    assert np.allclose(fwd.to_numpy(), fwd2.to_numpy())
    assert truth.has_premium


def test_yield_world_hedge_pays(yield_world):
    """When the premium is planted, the long-high-yield hedge earns a positive mean."""
    sig, fwd, _ = yield_world
    h = st.quantile_hedge(sig, fwd, long_high=True)
    s = st.summary(h["hedge"])
    assert s["mean"] > 0.0, f"Expected positive hedge mean, got {s['mean']:.4f}"
    assert s["sharpe"] > 0.3, f"Expected Sharpe > 0.3, got {s['sharpe']:.3f}"


def test_null_world_no_hedge(null_world):
    """Without a planted premium, the hedge mean is negligible."""
    sig, fwd, _ = null_world
    s = st.summary(st.quantile_hedge(sig, fwd, long_high=True)["hedge"])
    assert abs(s["mean"]) < 0.04, f"Null mean too large: {s['mean']:.4f}"


def test_hedge_is_long_minus_short(yield_world):
    """The 'hedge' column equals long leg minus short leg."""
    sig, fwd, _ = yield_world
    h = st.quantile_hedge(sig, fwd, long_high=True)
    assert np.allclose(h["hedge"], h["high"] - h["low"])


def test_net_buyback_yield_sign():
    """Net buyback yield = -(share change); a share decrease is positive buyback yield."""
    shares = pd.DataFrame(
        {"A": [100.0, 90.0, 99.0]},
        index=pd.Index([2010, 2011, 2012], name="year"),
    )
    bby = data._net_buyback_yield(shares)
    # 2011: shares fell 10% → buyback yield = +0.10
    assert np.isclose(bby.loc[2011, "A"], 0.10), f"got {bby.loc[2011, 'A']}"
    # 2012: shares rose 10% → buyback yield = -0.10
    assert np.isclose(bby.loc[2012, "A"], -0.10), f"got {bby.loc[2012, 'A']}"


# ---------------------------------------------------------------------------
# Real-tape test (skipped on CI when cache absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _CACHE_PRESENT, reason="real-tape EDGAR cache absent offline/CI")
def test_real_panel_loads():
    """The real panel loads with the expected shape and the hedge is computable."""
    total_yield, fwd, _ = data.fetch_panel()
    assert not total_yield.empty, "total_yield should not be empty with cache present"
    assert not fwd.empty, "fwd should not be empty with cache present"
    assert total_yield.shape[1] > 50, "expected many tickers"
    h = st.quantile_hedge(total_yield, fwd, q=0.2, long_high=True)
    assert len(h) >= 10, f"expected at least 10 years in hedge, got {len(h)}"
    s = st.summary(h["hedge"])
    assert np.isfinite(s["tstat"]), "t-stat should be finite"
