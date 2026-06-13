"""Strategy-engine tests for Study 89 (Turn-of-the-Month), including the spine tests."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from turn_of_the_month import strategy  # noqa: E402


def _bdays(n=600, start="2000-01-03"):
    return pd.bdate_range(start=start, periods=n)


def test_tom_mask_is_boolean_minority():
    idx = _bdays()
    mask = strategy.tom_mask(idx, last=1, first=3)
    assert mask.dtype == bool
    # ~4 of ~22 business days per month are flagged.
    assert 0.10 < mask.mean() < 0.30


def test_tom_mask_flags_month_edges():
    idx = _bdays()
    mask = strategy.tom_mask(idx, last=1, first=3)
    tdom = strategy.trading_day_of_month(idx)
    # The first trading day of each month is always in the window (it is +1).
    assert mask[tdom == 1].all()


def test_tom_position_is_binary_and_unlagged():
    idx = _bdays()
    pos = strategy.tom_position(idx, last=1, first=3)
    mask = strategy.tom_mask(idx, last=1, first=3)
    assert set(pd.unique(pos)) <= {0.0, 1.0}
    # Calendar-known -> position equals the mask exactly, no shift.
    assert np.array_equal(pos.to_numpy(), mask.astype(float).to_numpy())


def test_window_contrast_share_and_keys():
    idx = _bdays(800)
    close = pd.Series(100.0 * np.exp(np.cumsum(np.full(len(idx), 0.0003))), index=idx)
    wc = strategy.window_contrast(close, last=1, first=3)
    for k in ("tom_mean_bps", "rest_mean_bps", "diff_bps", "t_diff",
              "share_of_total_logret", "frac_tom_days"):
        assert k in wc
    assert 0.0 < wc["frac_tom_days"] < 1.0


def test_costs_reduce_return(planted_tom):
    close = planted_tom[0]["close"]
    pos = strategy.tom_position(close.index)
    free = strategy.backtest(close, pos, cost_bps=0.0)
    paid = strategy.backtest(close, pos, cost_bps=20.0)
    assert paid["final"] < free["final"]


def test_timer_holds_less_time_than_buy_hold(planted_tom):
    close = planted_tom[0]["close"]
    pos = strategy.tom_position(close.index)
    timer = strategy.backtest(close, pos, cost_bps=5.0)
    bh = strategy.buy_and_hold(close)
    assert timer["time_in_market"] < bh["time_in_market"]
    assert bh["time_in_market"] == 1.0


def test_hac_tstat_detects_constant_mean():
    # A constant positive series has an arbitrarily large t (zero variance guard aside).
    x = np.full(500, 0.001)
    t = strategy.hac_tstat(x)
    assert np.isnan(t) or t > 10  # se=0 path returns nan; otherwise huge


# --- Spine tests: the harness must bank a planted TOM and ignore a flat null ----------
def test_spine_planted_tom_is_detected(planted_tom):
    """With a planted +10 bps TOM bump, the TOM-minus-rest premium must be strongly
    significant under HAC and capture far more than its day-share of total return."""
    close = planted_tom[0]["close"]
    wc = strategy.window_contrast(close, last=1, first=3)
    assert wc["t_diff"] > 2.0
    assert wc["diff_bps"] > 5.0
    assert wc["share_of_total_logret"] > 2.0 * wc["frac_tom_days"]


def test_spine_flat_null_is_not_detected(null_walk):
    """On a flat i.i.d. walk there is no calendar cluster — the TOM-minus-rest premium
    must NOT clear the HAC significance bar (no spurious detection)."""
    close = null_walk[0]["close"]
    wc = strategy.window_contrast(close, last=1, first=3)
    assert abs(wc["t_diff"]) < 2.0


def test_subperiod_contrast_runs(planted_tom):
    close = planted_tom[0]["close"]
    sp = strategy.subperiod_contrast(close, split="2008-01-01", n_boot=300)
    assert 0.0 <= sp["p_delta"] <= 1.0
    assert sp["n_early"] > 0 and sp["n_late"] > 0
