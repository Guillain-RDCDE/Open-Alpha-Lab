"""Strategy tests for Study 1014 — what perfect macro foresight is worth.

Synthetic-first: every piece of machinery is shown to bank a planted lead and to stay quiet
on the matched null. A handful of gated checks then pin the real-tape headline.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from clairvoyance import data, strategy as st  # noqa: E402

needs_real = pytest.mark.skipif(not data.have_real(),
                                reason="bundled arch/statsmodels tapes unavailable")


# --------------------------------------------------------------------------- #
# Small statistics
# --------------------------------------------------------------------------- #
def test_ann_sharpe_is_quarterly_annualised():
    r = np.array([0.02, 0.00, 0.04, 0.02] * 10)
    assert st.ann_sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * 2.0)
    assert np.isnan(st.ann_sharpe([0.01, 0.01, 0.01]))


def test_ols_hac_recovers_a_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 0.5 * x + rng.normal(size=500)
    d = st.ols_hac(y, x)
    assert d["slope"] == pytest.approx(0.5, abs=0.1)
    assert d["t"] > 5
    assert st.ols_hac(y[:10], x[:10])["n"] == 10 and np.isnan(st.ols_hac(y[:10], x[:10])["t"])


def test_hac_t_is_sized_on_the_null():
    s = st.lead_lag_size(n_seeds=120)
    assert s["reject_rate"] < 0.12


# --------------------------------------------------------------------------- #
# Lead-lag
# --------------------------------------------------------------------------- #
def test_lead_lag_finds_the_planted_lead(planted):
    L = st.lead_lag(planted, "g")
    assert L["corr"].idxmax() == 1
    assert L.loc[1, "t"] > 4
    assert abs(L.loc[0, "t"]) < 3


def test_lead_lag_is_quiet_on_the_null(null_panel):
    L = st.lead_lag(null_panel, "g")
    assert (L["t"].abs() < 3).all()


# --------------------------------------------------------------------------- #
# Oracle construction — the one execution lag
# --------------------------------------------------------------------------- #
def test_oracle_threshold_uses_only_the_past():
    idx = pd.date_range("1950-03-31", periods=40, freq="QE")
    p = pd.DataFrame({"g": np.arange(40, dtype=float), "ex": 0.0}, index=idx)
    pos = st.oracle_position(p, "g", 0, warmup=5)
    # g rises every quarter, so x_t is always above the median of x_{<=t-1}.
    assert (pos.dropna() == 1.0).all()
    assert pos.iloc[:5].isna().all()
    # Changing a FUTURE value never changes today's threshold or position at h <= 0.
    p2 = p.copy()
    p2.iloc[30:, 0] = -100.0
    a = st.oracle_position(p, "g", -1, warmup=5).iloc[:30]
    b = st.oracle_position(p2, "g", -1, warmup=5).iloc[:30]
    pd.testing.assert_series_equal(a, b)


def test_oracle_horizon_shifts_the_revealed_value():
    idx = pd.date_range("1950-03-31", periods=60, freq="QE")
    rng = np.random.default_rng(3)
    p = pd.DataFrame({"g": rng.normal(size=60), "ex": 0.0}, index=idx)
    p0 = st.oracle_position(p, "g", 0, warmup=10)
    p1 = st.oracle_position(p, "g", 1, warmup=10)
    assert p1.iloc[-1:].isna().all()          # h=+1 needs the quarter after the last
    assert not p0.equals(p1)


def test_bad_variables_are_inverted():
    idx = pd.date_range("1950-03-31", periods=40, freq="QE")
    p = pd.DataFrame({"du": np.arange(40, dtype=float), "ex": 0.0}, index=idx)
    # unemployment rising every quarter -> always "bad" -> always in bills.
    assert (st.oracle_position(p, "du", 0, warmup=5).dropna() == 0.0).all()


def test_combined_is_a_majority_vote(planted):
    c = st.combined_position(planted, 1).dropna()
    parts = pd.concat([st.oracle_position(planted, v, 1) for v in st.COMBINED],
                      axis=1).loc[c.index]
    assert ((parts.sum(axis=1) >= 2).astype(float) == c).all()


def test_book_charges_costs_one_way_on_traded_nav():
    idx = pd.date_range("2000-03-31", periods=4, freq="QE")
    pos = pd.Series([1.0, 0.0, 1.0, 1.0], index=idx)
    ex = pd.Series([0.05, 0.05, 0.05, 0.05], index=idx)
    r = st.book(pos, ex, cost_bps=10.0)
    assert np.allclose(r.to_numpy(), [0.05 - 0.001, -0.001, 0.05 - 0.001, 0.05])


def test_market_oracle_is_the_ceiling(planted):
    P = st.all_positions(planted)
    win = st.common_window(P)
    sub = planted.loc[win]
    ceil = st.ann_sharpe(st.book(st.market_oracle(sub), sub["ex"]))
    for c in P.columns:
        assert st.ann_sharpe(st.book(P.loc[win, c], sub["ex"])) < ceil


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def test_bootstrap_of_identical_series_is_null():
    rng = np.random.default_rng(0)
    a = rng.normal(0.01, 0.08, 200)
    bt = st.sharpe_diff_bootstrap(a, a, n_boot=200)
    assert bt["dsharpe"] == 0.0 and bt["p"] >= 0.5


def test_bootstrap_is_deterministic():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0.02, 0.08, 150), rng.normal(0.01, 0.08, 150)
    assert st.sharpe_diff_bootstrap(a, b, n_boot=200) == st.sharpe_diff_bootstrap(a, b, n_boot=200)


def test_oracle_beats_buy_and_hold_on_the_planted_world(planted):
    pos = st.oracle_position(planted, "g", 1).dropna()
    e = st.evaluate(pos, planted, n_boot=300)
    assert e["dsharpe"] > 0.2
    assert st.passes(e)


def test_same_quarter_oracle_is_worth_little_on_the_planted_world(planted):
    """The study's mechanism, planted: returns price NEXT quarter, so h=0 knowledge is stale."""
    pos0 = st.oracle_position(planted, "g", 0).dropna()
    pos1 = st.oracle_position(planted, "g", 1).dropna()
    win = pos0.index.intersection(pos1.index)
    e0 = st.evaluate(pos0.loc[win], planted, n_boot=200, with_rotation=False)
    e1 = st.evaluate(pos1.loc[win], planted, n_boot=200, with_rotation=False)
    assert e1["sharpe"] > e0["sharpe"] + 0.2


def test_oracle_is_quiet_on_the_null(null_panel):
    pos = st.oracle_position(null_panel, "g", 1).dropna()
    e = st.evaluate(pos, null_panel, n_boot=300)
    assert not st.passes(e)


def test_rotation_placebo_is_exhaustive_and_bounded(planted):
    pos = st.oracle_position(planted, "g", 1).dropna()
    rt = st.rotation_test(pos, planted["ex"].loc[pos.index])
    assert rt["n_rot"] == len(pos) - 2 * st.MIN_SHIFT + 1
    assert 0 < rt["p"] < 0.05


def test_synthetic_power_fires_on_planted_and_not_on_null():
    sp = st.synthetic_power(seeds=range(1014, 1018), n_boot=150)
    assert sp[sp["signal_strength"] == 1.0]["passes"].mean() >= 0.75
    assert sp[sp["signal_strength"] == 0.0]["passes"].mean() <= 0.25


def test_horizon_table_covers_every_cell_on_one_window(planted):
    T, win = st.horizon_table(planted, n_boot=100, with_rotation=False)
    assert len(T) == 5 * len(st.HORIZONS)
    assert T["n"].nunique() == 1 and T["n"].iloc[0] == len(win)
    assert T["bh_sharpe"].nunique() == 1


def test_ceiling_ladder_is_monotone(planted):
    P = st.all_positions(planted)
    lad = st.ceiling_ladder(planted, st.common_window(P))
    assert lad["sharpe"].is_monotonic_decreasing


# --------------------------------------------------------------------------- #
# Real tape (gated) — pins the headline the docs quote
# --------------------------------------------------------------------------- #
@needs_real
def test_real_stocks_lead_gdp(real_panel):
    L = st.lead_lag(real_panel, "g")
    assert L.loc[1, "t"] > 2 and L.loc[2, "t"] > 2
    assert L.loc[1, "corr"] > L.loc[0, "corr"]


@needs_real
def test_real_same_quarter_gdp_oracle_does_not_beat_buy_and_hold(real_panel):
    P = st.all_positions(real_panel)
    win = st.common_window(P)
    sub = real_panel.loc[win]
    bh = st.ann_sharpe(sub["ex"])
    g0 = st.ann_sharpe(st.book(P.loc[win, "g@+0"], sub["ex"]))
    g1 = st.ann_sharpe(st.book(P.loc[win, "g@+1"], sub["ex"]))
    assert g0 < bh < g1


# --------------------------------------------------------------------------- #
# The verdict rule — both directions
# --------------------------------------------------------------------------- #
def _e(sharpe=0.3, bh=0.25, boot_p=0.4, rot_p=0.2, capture=0.03):
    return {"sharpe": sharpe, "bh_sharpe": bh, "dsharpe": sharpe - bh, "boot_p": boot_p,
            "rot_p": rot_p, "capture": capture, "hit_rate": 0.5}


def _h(**over):
    h = {"head": _e(), "lead": _e(0.47, boot_p=0.09, rot_p=0.01),
         "real": _e(0.48, boot_p=0.08, rot_p=0.01), "strict": _e(0.13, boot_p=0.8),
         "g_lead": _e(0.58, boot_p=0.02, rot_p=0.006), "ceiling": 1.5, "cost_bps": 10.0,
         "ll_g": {0: {"corr": 0.07, "t": 0.8}, 1: {"corr": 0.27, "t": 3.6},
                  2: {"corr": 0.34, "t": 4.4}}}
    h.update(over)
    return h


def test_verdict_signal_all_branches():
    assert st.verdict(_h())["signal"] == "None"
    assert st.verdict(_h(head=_e(0.6, boot_p=0.01, rot_p=0.01)))["signal"] == "Real"
    assert st.verdict(_h(lead=_e(0.6, boot_p=0.01, rot_p=0.01)))["signal"] == "Mixed"
    assert st.verdict(_h(head=_e(0.4, boot_p=0.08, rot_p=0.3)))["signal"] == "Weak"
    assert st.verdict(_h(head=_e(0.2, boot_p=0.01, rot_p=0.01)))["signal"] == "None"


def test_verdict_tradability_never_investable():
    assert st.verdict(_h())["trad"] == "Fragile"
    assert st.verdict(_h(real=_e(0.9, boot_p=0.0, rot_p=0.0)))["trad"] == "Fragile"
    assert st.verdict(_h(real=_e(0.2, boot_p=0.6)))["trad"] == "Mirage"
    assert st.verdict(_h(real=_e(0.4, boot_p=0.2)))["trad"] == "Mirage"


def test_verdict_third_axis_both_directions():
    assert st.verdict(_h())["third"] == "Confirmed"
    flat = {0: {"corr": 0.3, "t": 3.0}, 1: {"corr": 0.1, "t": 1.0}, 2: {"corr": 0.05, "t": 0.5}}
    assert st.verdict(_h(ll_g=flat))["third"] == "Not supported"


def test_verdict_prose_and_keys():
    v = st.verdict(_h())
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "third", "third_why",
                      "one_sentence"}
    assert "already priced" in v["signal_why"]
    assert "Fragile" in v["trad_why"]
    assert "one more quarter of lag" in v["one_sentence"]
