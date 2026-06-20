"""The Benford engine: the law itself, the leading-digit extractor, the goodness-of-fit
statistics, and the two spines — (1) the harness discriminates a Benford tape from a
non-Benford one (positive + anti control), and (2) the forensic 'deviation predicts
returns' claim has no signal on a tape with no planted relationship."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benford_law import data, strategy as st  # noqa: E402

SHARED_PANEL = os.path.abspath(os.path.join(data.SHARED_CACHE, "daily_panel.parquet"))


# ---- the law -------------------------------------------------------------
def test_benford_pmf_is_a_distribution():
    pmf = st.benford_pmf()
    assert pmf.shape == (9,)
    assert np.isclose(pmf.sum(), 1.0)
    assert np.isclose(pmf[0], np.log10(2.0))           # P(1) ~ 0.301
    assert pmf[0] > pmf[-1]                              # monotone decreasing


def test_leading_digits_basic():
    vals = np.array([1.0, 23.0, 0.045, 900.0, 7.7, -5.0])
    lead = st.leading_digits(vals)
    # 1->1, 23->2, 0.045->4, 900->9, 7.7->7, -5->5
    assert list(lead) == [1, 2, 4, 9, 7, 5]


def test_leading_digits_drops_nonpositive_and_nan():
    vals = np.array([0.0, np.nan, -0.0, 5.0])
    assert list(st.leading_digits(vals)) == [5]


def test_digit_frequencies_sum_to_one(benford_tape):
    s, _ = benford_tape
    f = st.digit_frequencies(s)
    assert np.isclose(f.sum(), 1.0)
    assert (f >= 0).all()


# ---- spine #1: the harness discriminates ----------------------------------
def test_benford_tape_conforms_better_than_nonbenford(benford_tape, nonbenford_tape):
    """A wide-range walk fits Benford far better than a range-bound price — both MAD & chi2."""
    sb, _ = benford_tape
    sn, _ = nonbenford_tape
    assert st.mad(sb) < st.mad(sn)
    assert st.chi_square(sb) < st.chi_square(sn)


def test_benford_tape_passes_nigrini_marginal_threshold(benford_tape):
    """The positive control's MAD should be at least 'marginal' (< nonconformity 0.015 is
    ideal; we require it to sit well below the range-bound control and under ~0.02)."""
    s, _ = benford_tape
    assert st.mad(s) < 0.02


def test_nonbenford_tape_is_flagged(nonbenford_tape):
    """A price living in [40, 60] has digits dominated by 4s and 5s — far from Benford."""
    s, _ = nonbenford_tape
    f = st.digit_frequencies(s)
    assert f[3] + f[4] > 0.9          # digits 4 and 5 dominate
    assert st.mad(s) > 0.05            # gross nonconformity


# ---- the rolling deviation engine -----------------------------------------
def test_rolling_deviation_matches_pointwise(benford_tape):
    """The vectorised rolling score equals the direct windowed statistic."""
    s, _ = benford_tape
    d = st.rolling_benford_deviation(s, window=252, statistic="mad")
    i = 4000
    direct = st.mad(s.to_numpy()[i - 251 : i + 1])
    assert np.isclose(d.iloc[i], direct)
    dc = st.rolling_benford_deviation(s, window=252, statistic="chi2")
    assert np.isclose(dc.iloc[i], st.chi_square(s.to_numpy()[i - 251 : i + 1]))


def test_rolling_deviation_warmup_is_nan(benford_tape):
    s, _ = benford_tape
    d = st.rolling_benford_deviation(s, window=252)
    assert d.iloc[:251].isna().all()
    assert np.isfinite(d.iloc[251])


# ---- inference helpers ----------------------------------------------------
def test_hac_tstat_zero_mean_noise_is_small():
    rng = np.random.default_rng(0)
    t = st.hac_tstat(rng.normal(0, 1, 2000))
    assert abs(t) < 3.0


def test_hac_tstat_strong_mean_is_significant():
    rng = np.random.default_rng(0)
    t = st.hac_tstat(rng.normal(0.5, 1, 2000))
    assert t > 5.0


def test_block_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(0.3, 1.0, 1500)
    lo, hi = st.block_bootstrap_ci(x, block=6, n_boot=500)
    assert lo < x.mean() < hi


# ---- spine #2: no planted relationship => no forensic signal ---------------
def test_forensic_backtest_null_on_unrelated_panel():
    """Build a panel where each name's price path is independent of any future return —
    the Benford deviation cannot predict returns, so |t| stays below the inference bar."""
    rng = np.random.default_rng(328)
    dates = data._decorative_index(2000)
    cols = {}
    for j in range(30):
        lr = rng.normal(0.0003, 0.02, len(dates))
        cols[f"N{j}"] = 20.0 * np.exp(np.cumsum(lr))
    prices = pd.DataFrame(cols, index=dates)
    res = st.forensic_backtest(prices, window=252, rebal=42, quantile=0.3, cost_bps=0.0)
    assert res["n_periods"] > 0
    assert abs(res["tstat"]) < 2.0          # no edge on an unrelated panel


def test_forensic_backtest_cost_lowers_return():
    rng = np.random.default_rng(5)
    dates = data._decorative_index(1500)
    prices = pd.DataFrame(
        {f"N{j}": 30.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, len(dates))))
         for j in range(20)},
        index=dates,
    )
    gross = st.forensic_backtest(prices, window=252, rebal=42, cost_bps=0.0)["mean_period"]
    net = st.forensic_backtest(prices, window=252, rebal=42, cost_bps=20.0)["mean_period"]
    assert net < gross


# ---- real panel (cache-gated: skips offline) ------------------------------
@pytest.mark.skipif(not os.path.exists(SHARED_PANEL), reason="offline CI: shared panel absent")
def test_real_panel_loads_and_runs_small():
    panel = pd.read_parquet(SHARED_PANEL)
    rets = panel.dropna(axis=1, thresh=int(len(panel) * 0.9)).iloc[:1500, :15]
    prices = (1 + rets.fillna(0)).cumprod() * 100
    res = st.forensic_backtest(prices, window=252, rebal=63, quantile=0.3)
    assert res["n_periods"] > 0
    assert np.isfinite(res["tstat"])
