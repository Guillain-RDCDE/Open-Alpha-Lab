"""The cross-asset panel must summarise the decomposition correctly for each asset."""

import numpy as np
import pandas as pd

from paper_prophet import extension


def _prices(seed, n=300, drift=0.0003, vol=0.01):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=n)
    p = 100.0 * np.exp(np.cumsum(rng.normal(drift, vol, size=n)))
    return pd.Series(p, index=idx, name="Close")


def test_run_asset_keys_and_identities():
    row = extension.run_asset("TEST", _prices(0))
    expected = {
        "ticker", "n_days", "hit_rate_pct", "dir_hac_t", "sharpe_stack", "sharpe_voltgt",
        "sharpe_buyhold", "arima_increment", "voltgt_lift", "leverage_corr",
    }
    assert expected <= set(row)
    # voltgt_lift and arima_increment are exact differences of the reported Sharpes.
    assert np.isclose(row["voltgt_lift"], row["sharpe_voltgt"] - row["sharpe_buyhold"], atol=1e-9)
    assert np.isclose(row["arima_increment"], row["sharpe_stack"] - row["sharpe_voltgt"], atol=1e-9)
    assert 0.0 <= row["hit_rate_pct"] <= 100.0


def test_run_panel_one_row_per_asset():
    panel = extension.run_panel({"A": _prices(1), "B": _prices(2)})
    assert list(panel.index) == ["A", "B"]
    assert panel.loc["A", "n_days"] > 0
