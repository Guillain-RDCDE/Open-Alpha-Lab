"""Offline, seeded tests for Study 797 (FX Value / PPP).

The synthetic world is deterministic; the value sort banks a planted PPP reversion and
stays flat on the null; the inference primitives behave; costs and borrow only reduce
the net; the signal is point-in-time (no look-ahead). All on the seeded panel — no
network, no real data.
"""
import numpy as np

from fx_value import data, strategy as st


def test_world_deterministic(planted_world):
    a = planted_world
    b = data.synthetic_world(value_strength=0.15, seed=797)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_value_sort_pays_when_reversion_real(planted_world):
    d = st.synthetic_detect(planted_world)
    assert d["sharpe"] > 0.5          # a real planted reversion is harvestable
    assert d["t_nw"] > 2.0            # and clears the bar on the control


def test_null_world_no_edge(null_world):
    d = st.synthetic_detect(null_world)
    assert abs(d["t_nw"]) < 2.5       # the null must not systematically fire


def test_signal_is_point_in_time():
    """value_signal at time t uses only the trailing window ending at t — dropping all
    rows after t must not change the signal value at t."""
    w = data.synthetic_world(value_strength=0.1, seed=1)
    full = st.value_signal(w)
    cut = st.value_signal(w.iloc[:150])
    common = full.index.intersection(cut.index)
    a = full.loc[common].to_numpy()
    b = cut.loc[common].to_numpy()
    m = np.isfinite(a) & np.isfinite(b)
    assert np.allclose(a[m], b[m])


def test_weights_dollar_neutral(planted_world):
    sig = st.value_signal(planted_world)
    w = st.weights_panel(sig).dropna(how="all")
    active = w[w.abs().sum(axis=1) > 0]
    assert np.allclose(active.sum(axis=1).to_numpy(), 0.0, atol=1e-9)   # sum w = 0
    assert np.allclose(active.abs().sum(axis=1).to_numpy(), 1.0, atol=1e-9)  # gross = 1


def test_costs_and_borrow_reduce_net(planted_world):
    sig = st.value_signal(planted_world)
    rets = planted_world.diff()
    gross = st.headline_stats(st.portfolio_returns(sig, rets))["ann_pct"]
    net0 = st.timer_stats(sig, rets, cost_bps=0.0, borrow_bps_ann=0.0)["net_ann_pct"]
    net_costed = st.timer_stats(sig, rets, cost_bps=10.0, borrow_bps_ann=100.0)["net_ann_pct"]
    assert abs(net0 - gross) < 1e-6          # zero cost, zero borrow == gross
    assert net_costed < net0                 # frictions only subtract


def test_newey_west_matches_ols_at_zero_lag():
    rng = np.random.default_rng(0)
    x = rng.normal(0.01, 0.05, 400)
    t_ols = st.one_sample_t(x)               # ddof=1 SE
    t_nw0 = st.newey_west_t(x, lags=0)       # /n SE — differs only by sqrt((n-1)/n)
    assert abs(t_ols - t_nw0) / abs(t_ols) < 0.01   # agree up to the finite-sample scaling


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(60, 100)
    assert lo < 0.60 < hi
