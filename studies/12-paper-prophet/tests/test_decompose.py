"""The decomposition arithmetic must be exactly right — it carries the whole verdict."""

import numpy as np
import pandas as pd

from paper_prophet import decompose
from paper_prophet.stack import generate_signals


def test_directional_edge_keys_and_ranges(wf):
    de = decompose.directional_edge(wf)
    assert 0.0 <= de["hit_rate"] <= 1.0
    assert de["n_graded"] <= len(wf.frame)
    assert np.isfinite(de["dir_edge_hac_t"])


def test_sharpe_decomposition_identity(wf):
    sd = decompose.sharpe_decomposition(wf)
    # the four legs are all finite Sharpes
    for k in ("sharpe_stack", "sharpe_voltgt", "sharpe_flat_forecast", "sharpe_buyhold"):
        assert np.isfinite(sd[k])
    # the ARIMA increment is exactly stack-Sharpe minus vol-targeting-Sharpe (definitional)
    assert np.isclose(
        sd["arima_increment_sharpe"], sd["sharpe_stack"] - sd["sharpe_voltgt"], atol=1e-9
    )


def test_voltgt_equals_buyhold_when_sizing_inert():
    # On a calm window where sigma<1 throughout, size==1, so vol-targeting == buy&hold and
    # the full stack == the flat forecast. This is the article's cap making sizing a no-op.
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=300)
    r = pd.Series(rng.normal(0.0, 0.3, size=len(idx)), index=idx)  # very low vol -> size caps at 1
    wf = generate_signals(r, lookback=252, max_steps=30)
    if (wf.frame["size"] == 1.0).all():
        sd = decompose.sharpe_decomposition(wf)
        assert np.isclose(sd["sharpe_voltgt"], sd["sharpe_buyhold"], atol=1e-9)
        assert np.isclose(sd["sharpe_stack"], sd["sharpe_flat_forecast"], atol=1e-9)


def test_cost_sweep_monotonic_in_spread(wf):
    cs = decompose.cost_sweep(wf)
    # higher spread can only lower the stack's net Sharpe (turnover is charged, never rebated).
    nets = cs["sharpe_stack_net"].to_numpy()
    assert np.all(np.diff(nets) <= 1e-9)


def test_win_rate_is_a_rate(wf):
    wr = decompose.win_rate(wf)
    assert 0.0 <= wr["win_rate"] <= 1.0
    assert wr["n_active"] <= len(wf.frame)
