"""Tests for the statistics helpers (bootstrap Sharpe, beta decomposition)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from overnight import decompose, diagnostics, stats


def test_bootstrap_ci_brackets_point_estimate():
    dec = decompose.decompose(
        diagnostics.synthetic_ohlc(overnight_bias_bps=3, intraday_bias_bps=-1, seed=0)
    )
    res = stats.sharpe_ci_bootstrap(dec["r_overnight"], n_boot=500, seed=1)
    assert res["ci_low"] <= res["sharpe"] <= res["ci_high"]
    assert 0.0 <= res["frac_negative"] <= 1.0


def test_strong_positive_sharpe_rarely_negative_in_bootstrap():
    # A clearly positive-drift series should almost never bootstrap negative.
    dec = decompose.decompose(
        diagnostics.synthetic_ohlc(
            overnight_bias_bps=8, intraday_bias_bps=0, daily_vol_bps=60, seed=2, n_days=252 * 20
        )
    )
    res = stats.sharpe_ci_bootstrap(dec["r_overnight"], n_boot=500, seed=3)
    assert res["sharpe"] > 0
    assert res["frac_negative"] < 0.05


def test_beta_decomposition_keys_and_identity():
    dec = decompose.decompose(diagnostics.synthetic_ohlc(seed=4))
    d = stats.beta_decomposition(dec, leg="overnight")
    # mean leg ~= alpha_daily + beta_contrib (decomposition identity in bps)
    recomposed = d["alpha_daily_bps"] + d["beta_contrib_bps"]
    assert abs(recomposed - d["mean_leg_bps"]) < 1e-6
    assert 0.0 <= d["r_squared"] <= 1.0


def test_beta_positive_when_overnight_carries_market_risk():
    # Most daily noise lives in the night leg -> overnight beta to market > 0.
    dec = decompose.decompose(
        diagnostics.synthetic_ohlc(night_share_of_vol=0.9, seed=5)
    )
    d = stats.beta_decomposition(dec, leg="overnight")
    assert d["beta"] > 0.5
