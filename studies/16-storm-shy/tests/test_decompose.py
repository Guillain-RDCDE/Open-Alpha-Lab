"""The inference earns the stamps and bounds them: a significant Moreira–Muir spanning alpha and a
bootstrap-positive Sharpe gain on the clustered tape (Signal REAL), both absent on the flat null —
and a certainty-equivalent that stays positive but smaller than the headline (the honest catch)."""

import numpy as np
import pandas as pd

from storm_shy import decompose


# --- the HAC-OLS helper recovers known coefficients --------------------------

def test_ols_nw_recovers_coefficients():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4000)
    y = 0.5 + 2.0 * x + 0.1 * rng.standard_normal(4000)
    out = decompose._ols_nw(y, x)
    assert abs(out["alpha"] - 0.5) < 0.01
    assert abs(out["beta"] - 2.0) < 0.01
    assert out["beta_t"] > 10                       # a real slope is wildly significant


# --- Moreira–Muir spanning alpha --------------------------------------------

def test_spanning_alpha_is_real_with_regime(regime_returns):
    reg = decompose.spanning_alpha(regime_returns, cost_bps=1.0)
    assert reg["alpha"] > 0
    assert reg["alpha_t"] > 2.0                      # HAC-significant: expands the MV frontier

def test_spanning_alpha_vanishes_when_flat(flat_returns):
    reg = decompose.spanning_alpha(flat_returns, cost_bps=1.0)
    assert reg["alpha_t"] < 2.0


# --- bootstrap CI on the Sharpe gain ----------------------------------------

def test_sharpe_gain_ci_excludes_zero_with_regime(regime_returns):
    bs = decompose.sharpe_gain_bootstrap(regime_returns, n_boot=800, seed=0, cost_bps=1.0)
    assert bs["sharpe_gain"] > 0
    assert bs["ci_low"] > 0                          # whole interval above zero
    assert bs["frac_negative"] < 0.05


# --- the honest counter: CRRA certainty-equivalent at matched risk ----------

def test_certainty_equivalent_positive_with_regime(regime_returns):
    """At matched vol a risk-averse CRRA investor is still genuinely better off — the overlay's gain
    survives the honest utility test on a clean clustered tape."""
    ce = decompose.certainty_equivalent(regime_returns, gamma=5.0, cost_bps=1.0)
    assert ce["ce_gain_pct"] > 0

def test_certainty_equivalent_flat_is_negligible(flat_returns):
    ce = decompose.certainty_equivalent(flat_returns, gamma=5.0, cost_bps=1.0)
    assert abs(ce["ce_gain_pct"]) < 1.0


def test_equal_risk_return_positive_with_regime(regime_returns):
    er = decompose.equal_risk_return(regime_returns, cost_bps=1.0)
    assert er["excess_cagr_pct"] > 0
    assert er["managed_maxdd"] > er["buy_hold_maxdd"]     # shallower drawdown at equal risk
