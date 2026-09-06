"""Strategy tests for Study 972 — identities on the null, planted yields on the signal."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from adj_mode import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The null: no dividends, no difference
# --------------------------------------------------------------------------- #
def test_the_two_panels_are_identical_without_dividends(no_yield):
    tr, px, _ = no_yield
    assert np.allclose(tr.to_numpy(), px.to_numpy())
    y = st.yield_table(tr, px)
    assert (y["implied_yield"].abs() < 1e-12).all()


def test_rankings_agree_perfectly_without_dividends(no_yield):
    tr, px, _ = no_yield
    r = st.ranking_table(tr, px)
    assert (r["spearman"] > 0.999).all()
    assert (r["pair_flips"] == 0).all()


# --------------------------------------------------------------------------- #
# The planted world
# --------------------------------------------------------------------------- #
def test_implied_yield_recovers_the_planted_yield(planted):
    """Recovered, with a small upward bias that is itself arithmetic, not error.

    A yield subtracted from a *daily* return compounds differently from one subtracted from
    an annual one, so the CAGR gap runs slightly above the planted yield (roughly by a factor
    of one plus the growth rate). The tolerance below admits that and nothing else.
    """
    tr, px, truth = planted
    y = st.yield_table(tr, px)
    for i, planted_y in enumerate(truth["yields"]):
        got = y.loc[f"A{i}", "implied_yield"]
        assert got == pytest.approx(planted_y, abs=0.002, rel=0.20)
        assert got >= planted_y - 1e-9      # the bias has a sign, and this is it


def test_volatility_is_almost_untouched(planted):
    tr, px, _ = planted
    y = st.yield_table(tr, px)
    assert (y["vol_tr"] - y["vol_px"]).abs().max() < 0.01


def test_price_only_ranking_is_a_yield_tilted_ranking():
    """Equal total returns by construction, so a price ranking must order by yield.

    The mechanism is measured on a deliberately clean panel — wide yield dispersion, modest
    idiosyncratic noise — because on the default fixture a 6% yield spread sits inside an 11%
    idiosyncratic vol and the tilt, while present, is drowned. That is a statement about
    signal-to-noise, not about the mechanism, and the *real* tape is where its size is
    measured.
    """
    tr, px, truth = data.synthetic_pair(n_years=25, vol_ann=0.06,
                                        yields=np.linspace(0.0, 0.15, 6), seed=972)
    mean_rank = st.trailing_return(px).dropna().rank(axis=1).mean()
    ys = pd.Series(truth["yields"], index=tr.columns)
    assert mean_rank.idxmin() == ys.idxmax()
    assert mean_rank.corr(ys) < -0.8


def test_the_tilt_shrinks_as_noise_grows():
    """The same mechanism, quantified: more idiosyncratic vol, weaker yield ordering."""
    corrs = {}
    for vol in (0.06, 0.30):
        tr, px, truth = data.synthetic_pair(n_years=25, vol_ann=vol,
                                            yields=np.linspace(0.0, 0.15, 6), seed=972)
        mr = st.trailing_return(px).dropna().rank(axis=1).mean()
        corrs[vol] = mr.corr(pd.Series(truth["yields"], index=tr.columns))
    assert corrs[0.06] < corrs[0.30] < 0.0


def test_rankings_disagree_when_yields_are_dispersed(planted):
    tr, px, _ = planted
    r = st.ranking_table(tr, px)
    assert r["spearman"].mean() < 0.99
    assert r["flip_share"].mean() > 0.0
    assert not r["same_top"].all()


def test_cagr_matches_a_hand_computation():
    """CAGR is measured over CALENDAR years, so a 252-bday year is ~1.0 calendar years."""
    idx = pd.bdate_range("2010-01-04", periods=252 * 4)
    p = pd.Series(100 * 1.10 ** (np.arange(len(idx)) / 252), index=idx)
    years = (idx[-1] - idx[0]).days / 365.25
    expected = (p.iloc[-1] / p.iloc[0]) ** (1 / years) - 1
    assert st.cagr(p) == pytest.approx(expected)
    assert st.cagr(p) == pytest.approx(0.10, rel=0.06)


def test_trailing_return_skips_the_last_month(planted):
    tr, _, _ = planted
    s = st.trailing_return(tr, lookback=252, skip=21)
    col = tr.columns[0]
    i = 400
    expected = tr[col].iloc[i - 21] / tr[col].iloc[i - 252] - 1
    assert s[col].iloc[i] == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# The backtest
# --------------------------------------------------------------------------- #
def test_backtest_scores_both_arms_on_the_same_panel(planted):
    """The comparison must isolate SELECTION, not dividend income."""
    tr, px, _ = planted
    a = st.momentum_backtest(tr, tr, top_k=2)
    b = st.momentum_backtest(px, tr, top_k=2)
    assert a["returns"].index.equals(b["returns"].index)
    # identical signals would give identical books; different signals must not
    assert not np.allclose(a["returns"].fillna(0), b["returns"].fillna(0))


def test_backtest_has_execution_lag_and_costs(planted):
    tr, _, _ = planted
    free = st.momentum_backtest(tr, tr, cost_bps=0.0)
    paid = st.momentum_backtest(tr, tr, cost_bps=50.0)
    assert paid["cagr"] < free["cagr"]
    assert free["turnover_ann"] > 0
    # a position can only be held from the session AFTER the signal date
    w = free["weights"]
    assert w.iloc[0].sum() == 0


def test_backtest_is_identical_across_panels_when_there_are_no_dividends(no_yield):
    tr, px, _ = no_yield
    a = st.momentum_backtest(tr, tr, top_k=2)
    b = st.momentum_backtest(px, tr, top_k=2)
    assert np.allclose(a["returns"].fillna(0), b["returns"].fillna(0))


def test_holding_yield_measures_the_tilt(planted):
    tr, px, truth = planted
    ys = pd.Series(truth["yields"], index=tr.columns)
    on_tr = st.momentum_backtest(tr, tr, top_k=2)
    on_px = st.momentum_backtest(px, tr, top_k=2)
    y_tr = st.holding_yield(on_tr["weights"], ys)
    y_px = st.holding_yield(on_px["weights"], ys)
    assert y_px < y_tr           # ranking on price avoids the payers


def test_risk_table_reports_both_conventions(planted):
    tr, px, _ = planted
    r = st.risk_table(tr, px)
    # The zero-yield asset is identical under both conventions; every payer must improve.
    assert (r["sharpe_tr"] >= r["sharpe_px"] - 1e-12).all()
    assert (r["sharpe_gap"] > 0).sum() >= len(r) - 1
    assert set(["sharpe_gap", "maxdd_gap"]).issubset(r.columns)


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"max_implied_yield": 0.055, "min_implied_yield": 0.004, "max_yield_ticker": "HYG",
         "min_yield_ticker": "QQQ", "max_share_of_return": 0.75, "max_vol_gap": 0.003,
         "max_sharpe_gap": 0.31, "mean_flip_share": 0.18, "same_top_share": 0.7,
         "momentum_cagr_tr": 0.09, "momentum_cagr_px": 0.07, "momentum_cagr_gap": 0.02,
         "momentum_sharpe_tr": 0.6, "momentum_sharpe_px": 0.45, "momentum_sharpe_gap": 0.15,
         "yield_tilt": -0.012}
    h.update(over)
    return h


def test_verdict_signal_ladder():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(max_implied_yield=0.01))["signal"] == "Weak"
    assert st.verdict(_headline(max_implied_yield=0.001))["signal"] == "None"


def test_verdict_usefulness_ladder():
    assert st.verdict(_headline())["trad"] == "Useful"
    quiet = _headline(mean_flip_share=0.02, momentum_cagr_gap=0.001, momentum_sharpe_gap=0.0)
    assert st.verdict(quiet)["trad"] == "Fragile"
    assert st.verdict(dict(quiet, max_implied_yield=0.001))["trad"] == "Mirage"


def test_verdict_prose_quotes_the_numbers():
    v = st.verdict(_headline(max_implied_yield=0.062))
    assert "6.20%" in v["signal_why"] or "6.2%" in v["signal_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
