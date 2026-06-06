"""Tests for the core decomposition and cost model.

The decomposition identity is the foundation of the whole project, so it gets
the most scrutiny: if (1+r_on)(1+r_id) != (1+r_cc) the rest is meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab import backtest, decompose, diagnostics


def _toy_ohlc() -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=6)
    return pd.DataFrame(
        {
            "Open": [100, 101, 103, 102, 104, 107],
            "High": [102, 104, 104, 105, 108, 109],
            "Low": [99, 100, 101, 101, 103, 106],
            "Close": [101, 102, 101, 104, 106, 108],
        },
        index=idx,
    )


def test_identity_holds_on_toy_data():
    dec = decompose.decompose(_toy_ohlc())
    assert decompose.check_identity(dec) < 1e-12


def test_identity_holds_on_synthetic():
    ohlc = diagnostics.synthetic_ohlc(n_days=2000, seed=7)
    dec = decompose.decompose(ohlc)
    assert decompose.check_identity(dec) < 1e-10


def test_overnight_times_intraday_equals_close_close_per_row():
    dec = decompose.decompose(_toy_ohlc())
    lhs = (1 + dec["r_overnight"]) * (1 + dec["r_intraday"])
    rhs = 1 + dec["r_close_close"]
    np.testing.assert_allclose(lhs.values, rhs.values, rtol=0, atol=1e-12)


def test_first_row_dropped_no_prior_close():
    ohlc = _toy_ohlc()
    dec = decompose.decompose(ohlc)
    assert len(dec) == len(ohlc) - 1
    assert dec.index[0] == ohlc.index[1]


def test_case_insensitive_columns():
    ohlc = _toy_ohlc().rename(columns=str.lower)
    dec = decompose.decompose(ohlc)
    assert decompose.check_identity(dec) < 1e-12


def test_missing_column_raises():
    bad = _toy_ohlc().drop(columns=["Open"])
    with pytest.raises(KeyError):
        decompose.decompose(bad)


def test_summary_shape_and_keys():
    dec = decompose.decompose(diagnostics.synthetic_ohlc(n_days=1000, seed=3))
    s = decompose.summary(dec)
    assert list(s.index) == ["overnight", "intraday", "close_close"]
    assert {"cum_return", "mean_bps_day", "vol_ann", "sharpe", "n_days"} <= set(s.columns)


def test_synthetic_overnight_beats_intraday_by_construction():
    # +3 bps night, -1 bp day -> overnight cumulative must exceed intraday.
    dec = decompose.decompose(
        diagnostics.synthetic_ohlc(overnight_bias_bps=3, intraday_bias_bps=-1, seed=0)
    )
    s = decompose.summary(dec)
    assert s.loc["overnight", "cum_return"] > s.loc["intraday", "cum_return"]


def test_compounding_table_matches_closed_form():
    t = diagnostics.compounding_table(bias_bps=(10,), horizons_years=(10,))
    expected = (1 + 10e-4) ** (252 * 10) - 1
    assert abs(t.loc["10 bps/night", "~10y"] - expected) < 1e-6


def test_artifact_moves_return_into_overnight():
    flat = diagnostics.synthetic_ohlc(overnight_bias_bps=0, intraday_bias_bps=0, seed=1)
    clean = decompose.decompose(flat)
    dirty = decompose.decompose(diagnostics.inject_split_artifact(flat, factor=1.5))
    assert dirty["cum_overnight"].iloc[-1] > clean["cum_overnight"].iloc[-1]
    assert len(diagnostics.flag_suspicious_returns(dirty)) >= 3


def test_cost_sweep_is_monotonically_worse():
    dec = decompose.decompose(
        diagnostics.synthetic_ohlc(overnight_bias_bps=3, intraday_bias_bps=-1, seed=0)
    )
    sweep = backtest.cost_sweep(dec, roundtrip_bps=(0, 2, 5, 10))
    cagr = sweep["cagr_net"].values
    assert all(cagr[i] >= cagr[i + 1] for i in range(len(cagr) - 1))


def test_breakeven_equals_mean_overnight_bps():
    dec = decompose.decompose(diagnostics.synthetic_ohlc(seed=0))
    be = backtest.breakeven_cost_bps(dec)
    assert abs(be - dec["r_overnight"].mean() * 1e4) < 1e-9
