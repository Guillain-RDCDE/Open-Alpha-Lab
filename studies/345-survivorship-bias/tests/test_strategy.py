"""The engine invariants, and the study's two spines:
(1) with NO deaths the FULL and SURVIVORS reads are IDENTICAL (the harness invents no gap);
(2) with a realistic death rate, deleting the dead names MANUFACTURES a significant edge
    out of a panel with no true signal — survivorship bias confirmed."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from survivorship_bias import data, strategy as st  # noqa: E402


# ---- signal & weights ------------------------------------------------------
def test_weights_long_only_sum_to_one(null_panel):
    rets, _, _ = null_panel
    sig = st.momentum_signal(rets, lookback=12)
    w = st.cross_section_weights(sig.iloc[100], frac=0.2, long_short=False)
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w >= 0).all()
    assert (w > 0).sum() == max(1, int(round(0.2 * sig.iloc[100].dropna().shape[0])))


def test_weights_long_short_dollar_neutral(null_panel):
    rets, _, _ = null_panel
    sig = st.momentum_signal(rets, lookback=12)
    w = st.cross_section_weights(sig.iloc[100], frac=0.2, long_short=True)
    assert abs(w.sum()) < 1e-9            # net zero
    assert abs(w[w > 0].sum() - 1.0) < 1e-9
    assert abs(w[w < 0].sum() + 1.0) < 1e-9


def test_signal_excludes_delisted_names(biased_panel):
    """A delisted name has NaN return → NaN signal → never ranked that month."""
    rets, alive, truth = biased_panel
    sig = st.momentum_signal(rets, lookback=12)
    dead = truth["doomed_cols"][0]
    # After the death month the signal for that name is NaN.
    last_alive = alive[dead][alive[dead]].index[-1]
    after = sig[dead].loc[sig.index > last_alive]
    assert after.isna().all()


# ---- engine invariants -----------------------------------------------------
def test_costs_monotonically_lower_returns(biased_panel):
    rets, alive, _ = biased_panel
    surv = data.survivors_only(rets, alive)
    means = [st.backtest(surv, cost_bps=c, long_short=False, direction=-1).mean()
             for c in (0.0, 25.0, 50.0)]
    assert means[0] > means[1] > means[2]


def test_one_lag_first_period_is_nan(null_panel):
    """The first month carries no position (signal needs a prior close) → NaN, not a peek."""
    rets, _, _ = null_panel
    r = st.backtest(rets, long_short=False, direction=-1)
    assert np.isnan(r.iloc[0])


def test_backtest_reproducible(biased_panel):
    rets, alive, _ = biased_panel
    surv = data.survivors_only(rets, alive)
    a = st.backtest(surv, long_short=False, direction=-1)
    b = st.backtest(surv, long_short=False, direction=-1)
    assert np.allclose(a.dropna().to_numpy(), b.dropna().to_numpy())


# ---- the theses ------------------------------------------------------------
def test_no_deaths_means_identical_reads(null_panel):
    """Spine #1 — the control: with no delisting, FULL and SURVIVORS are the SAME panel,
    so the survivorship delta is exactly zero. The harness manufactures no gap."""
    rets, alive, _ = null_panel
    surv = data.survivors_only(rets, alive)
    res = st.compare_views(rets, surv, long_short=False, direction=-1, cost_bps=0.0)
    assert abs(res["delta_sharpe"]) < 1e-9
    assert abs(res["delta_mean_ann"]) < 1e-9
    assert np.isclose(res["test_full"]["tstat"], res["test_survivors"]["tstat"])


def test_deletion_manufactures_a_significant_edge(biased_panel):
    """Spine #2 — the CONFIRMED axis: with a realistic death rate and NO true signal, the
    honest FULL read is insignificant while the survivors-only read clears t=2 — the edge
    is manufactured purely by deleting the firms that died."""
    rets, alive, _ = biased_panel
    surv = data.survivors_only(rets, alive)
    res = st.compare_views(rets, surv, long_short=False, direction=-1, cost_bps=0.0)
    # Survivors look better than full on every honest metric.
    assert res["delta_sharpe"] > 0.15
    assert res["delta_mean_ann"] > 0.0
    # The honest tape has no significant edge; the biased tape does.
    assert abs(res["test_full"]["tstat"]) < 2.0
    assert res["test_survivors"]["tstat"] > 2.0


def test_bias_grows_with_death_rate():
    """The survivorship delta is monotone in how many firms die."""
    deltas = []
    for dr in (0.0, 0.15, 0.30, 0.45):
        rets, alive, _ = data.synthetic_panel(n_periods=240, n_assets=200,
                                              death_rate=dr, signal_strength=0.0, seed=345)
        surv = data.survivors_only(rets, alive)
        res = st.compare_views(rets, surv, long_short=False, direction=-1, cost_bps=0.0)
        deltas.append(res["delta_sharpe"])
    assert deltas[0] == pytest.approx(0.0, abs=1e-9)
    assert deltas[1] < deltas[2] < deltas[3]


def test_summarize_keys(null_panel):
    rets, _, _ = null_panel
    s = st.summarize(st.backtest(rets, long_short=False, direction=-1))
    assert set(s) == {"n", "cagr", "sharpe", "max_dd", "mean_ann"}


def test_hac_and_bootstrap_agree_on_sign(biased_panel):
    rets, alive, _ = biased_panel
    surv = data.survivors_only(rets, alive)
    r = st.backtest(surv, long_short=False, direction=-1)
    lo, hi = st.block_bootstrap_ci(r, n_boot=1000, seed=1)
    assert st.hac_tstat(r)["mean"] > 0
    assert (lo + hi) / 2 > 0
