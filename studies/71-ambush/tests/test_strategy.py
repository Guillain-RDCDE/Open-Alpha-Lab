"""The book's accounting: one lag, costs on turnover, the stop, the 1% budget."""

import numpy as np
import pandas as pd
import pytest

from ambush import signals, strategy, synth


def _tape(n=400, seed=3):
    return synth.synthetic_tape(n_days=n, plant_bps_per_signal=0.0, seed=seed)


def test_one_lag_no_lookahead():
    spy, vix = _tape()
    rf = synth.flat_rf(spy.index)
    led = strategy.book(spy, vix, rf, k=2)
    conf = signals.confluence(spy, vix)
    target = (conf["count"] >= 2).astype(float) * strategy.sizing(spy["Close"].pct_change())
    # the position held on day t is exactly yesterday's target — never today's
    pd.testing.assert_series_equal(
        led["pos"], target.shift(1).iloc[1:], check_names=False
    )


def test_future_data_cannot_change_todays_position():
    spy, vix = _tape()
    rf = synth.flat_rf(spy.index)
    cut = 300
    full = strategy.book(spy, vix, rf, k=2)
    spy2 = spy.copy()
    spy2.iloc[cut:] *= 1.5  # rewrite the future
    tampered = strategy.book(spy2, vix, rf, k=2)
    pd.testing.assert_series_equal(full["pos"].iloc[: cut - 1], tampered["pos"].iloc[: cut - 1])


def test_costs_charged_on_each_leg():
    spy, vix = _tape()
    rf = synth.flat_rf(spy.index)
    led = strategy.book(spy, vix, rf, k=2, spread_bps=10.0)
    led0 = strategy.book(spy, vix, rf, k=2, spread_bps=0.0)
    assert led0["cost"].sum() == 0
    held = led["pos"] > 0
    entries = (held & ~held.shift(1, fill_value=False)).sum()
    assert led["cost"].sum() > 0 and entries > 0
    # gross identical, net differs by exactly the cost column
    pd.testing.assert_series_equal(led["gross"], led0["gross"], check_names=False)
    assert led["net_excess"].sum() == pytest.approx((led0["net_excess"] - led["cost"]).sum())


def test_stop_caps_the_day_at_one_percent_of_nav():
    # hand-built crash bar: armed day (low close, red, VIX spike), then a -10% day
    idx = pd.bdate_range("2024-01-02", periods=40)
    spy = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0}, index=idx)
    rng = np.random.default_rng(0)
    spy["Close"] += rng.normal(0, 0.3, len(spy))  # give the vol estimator something
    spy["High"] = spy[["Open", "Close"]].max(axis=1) + 1.0
    spy["Low"] = spy[["Open", "Close"]].min(axis=1) - 1.0
    t = 30
    spy.iloc[t] = [100.0, 100.5, 95.0, 95.5]  # low close, red
    spy.iloc[t + 1] = [98.0, 98.5, 85.0, 86.0]  # the next-day crash, no gap at the open
    vix = pd.Series(20.0, index=idx)
    vix.iloc[t] = 30.0
    rf = synth.flat_rf(idx)
    led = strategy.book(spy, vix, rf, k=3, spread_bps=0.0, financing_spread=0.0)
    day = led.iloc[t]  # book() drops day 0, so ledger row t == calendar day t+1
    assert day["pos"] > 0 and day["stopped"]
    assert day["gross"] == pytest.approx(-strategy.DAILY_RISK)
    nostop = strategy.book(spy, vix, rf, k=3, spread_bps=0.0, financing_spread=0.0, use_stop=False)
    assert nostop.iloc[t]["gross"] < day["gross"]  # without the stop the crash lands in full


def test_sizing_respects_budget_and_cap():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 300),
                  index=pd.bdate_range("2023-01-02", periods=300))
    w = strategy.sizing(r)
    sig = r.rolling(strategy.VOL_WINDOW).std(ddof=1)
    ok = sig.notna() & (w > 0)
    # the 1%/day budget: a 2-sigma down day never costs more than 1% of NAV
    assert (w[ok] * strategy.RISK_Z * sig[ok] <= strategy.DAILY_RISK + 1e-12).all()
    assert (w <= strategy.MAX_LEVERAGE).all()
    assert (w.iloc[: strategy.VOL_WINDOW - 1] == 0).all()  # no trade before the estimator exists


def test_financing_charged_only_on_held_nights():
    spy, vix = _tape()
    rf = synth.flat_rf(spy.index, ann_rate=0.05)
    led = strategy.book(spy, vix, rf, k=2, financing_spread=0.025)
    flat = led["pos"] == 0
    assert (led.loc[flat, "fin"] == 0).all()
    held = led["pos"] > 0
    expected = led.loc[held, "pos"] * (0.05 / 252 + 0.025 / 252)
    pd.testing.assert_series_equal(led.loc[held, "fin"], expected, check_names=False)


def test_lift_table_shares_sum_to_one():
    spy, vix = _tape(800)
    lift = strategy.lift_table(spy, vix)
    assert lift["share"].sum() == pytest.approx(1.0)
    assert lift["n"].sum() == len(spy) - 1  # the last day has no next-day return
