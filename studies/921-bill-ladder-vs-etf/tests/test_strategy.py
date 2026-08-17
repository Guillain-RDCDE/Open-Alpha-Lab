"""Ladder mechanics, inference primitives, and the study's spine — all offline/synthetic.

The spine: on a world where the ETF charges a *known* fee the ladder recovers a gap of
about that size; on a world where the ETF is free the gap vanishes. Around that sit the
mechanical invariants — the discount/BEY conversion round-trips, exactly one execution lag,
frictions only ever reduce the ladder's return, and the annualisation is honest.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bill_ladder import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Rate conversion
# --------------------------------------------------------------------------- #
def test_discount_to_bey_exceeds_the_discount_quote():
    d = pd.Series([0.5, 1.0, 3.7, 5.0], index=pd.bdate_range("2020-01-01", periods=4))
    bey = st.discount_to_bey(d)
    assert (bey.to_numpy() > d.to_numpy() / 100.0).all()


def test_discount_to_bey_magnitude_is_textbook():
    """A 3.70% discount on a 91-day bill is ~3.787% bond-equivalent (about +9 bps)."""
    d = pd.Series([3.70], index=pd.bdate_range("2020-01-01", periods=1))
    assert st.discount_to_bey(d).iloc[0] == pytest.approx(0.037868, abs=2e-5)


def test_conversion_gap_widens_with_the_rate_level():
    d = pd.Series([0.5, 5.0], index=pd.bdate_range("2020-01-01", periods=2))
    gap = st.discount_to_bey(d).to_numpy() - d.to_numpy() / 100.0
    assert gap[1] > gap[0] > 0


def test_raw_basis_is_the_identity():
    d = pd.Series([1.0, 2.0], index=pd.bdate_range("2020-01-01", periods=2))
    assert np.allclose(st.rate_to_yield(d, basis="raw").to_numpy(), [0.01, 0.02])


def test_unknown_basis_raises():
    d = pd.Series([1.0], index=pd.bdate_range("2020-01-01", periods=1))
    with pytest.raises(ValueError):
        st.rate_to_yield(d, basis="nonsense")


def test_synthetic_irx_round_trips_through_the_conversion():
    """data.synthetic_daily builds irx by inverting discount_to_bey — check the loop closes."""
    frame, truth = data.synthetic_daily(seed=921)
    bey = st.discount_to_bey(frame["irx"])
    assert bey.mean() == pytest.approx(truth["mean_rate"], abs=2e-4)


# --------------------------------------------------------------------------- #
# Ladder mechanics
# --------------------------------------------------------------------------- #
def test_ladder_columns_and_warmup(fee_world):
    frame, _ = fee_world
    lad = st.ladder_returns(frame["irx"])
    assert {"rate", "roll", "r_gross", "r_ladder"}.issubset(lad.columns)
    assert not lad.isna().any().any()
    # the first ~91 calendar days are consumed building the 13 rungs
    assert (lad.index[0] - frame.index[0]).days >= 84


def test_ladder_on_a_constant_rate_accrues_that_rate():
    idx = pd.bdate_range("2020-01-01", periods=800)
    rate = pd.Series(4.0, index=idx)            # 4% discount, flat forever
    lad = st.ladder_returns(rate)
    expected = float(st.discount_to_bey(rate).iloc[0])
    assert lad["rate"].std() < 1e-12
    assert lad["rate"].iloc[-1] == pytest.approx(expected, abs=1e-12)
    # the ladder reinvests daily, so the realised CAGR is the *compounded* BEY
    realised = (1.0 + lad["r_ladder"]).prod() ** (365.25 / (lad.index[-1] - lad.index[0]).days) - 1
    assert realised == pytest.approx(float(np.expm1(expected)), abs=2e-4)


def test_ladder_rate_is_a_trailing_average_so_it_lags_a_jump():
    """A step in the quote must feed through gradually — 13 rungs, one replaced per week."""
    idx = pd.bdate_range("2020-01-01", periods=600)
    rate = pd.Series(np.where(np.arange(600) < 300, 1.0, 5.0), index=idx, dtype=float)
    lad = st.ladder_returns(rate)
    after = lad["rate"].loc[lad.index > idx[300]]
    assert after.iloc[0] < after.iloc[-1]           # it climbs
    assert after.iloc[0] < 0.05 and after.iloc[-1] > 0.045   # from ~1% toward ~5%


def test_exactly_one_execution_lag(fee_world):
    """Perturbing the tail of the quote must not change any earlier ladder rate."""
    frame, _ = fee_world
    rate = frame["irx"].copy()
    base = st.ladder_returns(rate)
    bumped = rate.copy()
    cut = rate.index[3000]
    bumped.loc[bumped.index > cut] *= 2.0
    pert = st.ladder_returns(bumped)
    common = base.index[base.index <= cut]
    assert np.allclose(base.loc[common, "rate"].to_numpy(),
                       pert.loc[common, "rate"].to_numpy())


def test_lag_is_one_not_zero():
    """A rung bought on a roll day must carry *yesterday's* quote, never today's."""
    idx = pd.bdate_range("2020-01-01", periods=400)
    rate = pd.Series(2.0, index=idx)
    rate.iloc[:1] = 2.0
    # make one single day's quote wildly different and check the ladder never uses it
    # on that same day
    rate.iloc[200] = 40.0
    lad = st.ladder_returns(rate)
    day = idx[200]
    if day in lad.index:
        assert lad.loc[day, "rate"] < 0.06     # today's spike cannot be in today's rate


def test_roll_count_matches_the_schedule(fee_world):
    frame, _ = fee_world
    lad = st.ladder_returns(frame["irx"], n_rungs=13)
    span_days = (lad.index[-1] - lad.index[0]).days
    assert abs(lad["roll"].sum() - span_days / 7.0) < 25    # ~one roll a week


def test_more_rungs_means_more_rolls(fee_world):
    frame, _ = fee_world
    counts = [st.ladder_returns(frame["irx"], n_rungs=n)["roll"].sum() for n in (4, 13, 26)]
    assert counts[0] < counts[1] < counts[2]


def test_frictions_monotonically_reduce_the_ladder(fee_world):
    frame, _ = fee_world
    means = [st.ladder_returns(frame["irx"], cost_bps=c)["r_ladder"].mean()
             for c in (0.0, 2.0, 10.0)]
    assert means[0] > means[1] > means[2]
    idles = [st.ladder_returns(frame["irx"], idle_days=d)["r_ladder"].mean()
             for d in (0.0, 2.0, 5.0)]
    assert idles[0] > idles[1] > idles[2]


def test_zero_friction_leaves_gross_untouched(fee_world):
    frame, _ = fee_world
    lad = st.ladder_returns(frame["irx"], cost_bps=0.0, idle_days=0.0)
    assert np.allclose(lad["r_gross"].to_numpy(), lad["r_ladder"].to_numpy())


def test_weekend_gaps_accrue_three_days():
    """Accrual is actual/365 on calendar days, so a Friday->Monday step pays 3x."""
    idx = pd.bdate_range("2020-01-01", periods=500)
    lad = st.ladder_returns(pd.Series(4.0, index=idx))
    dow = lad.index.dayofweek
    mon = lad["r_ladder"][dow == 0].mean()
    tue = lad["r_ladder"][dow == 1].mean()
    assert mon == pytest.approx(3.0 * tue, rel=0.05)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    x = 0.001 + rng.normal(0, 0.01, 5000)
    assert st.newey_west_t(x) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_welch_and_one_sample_finite(fee_world):
    frame, _ = fee_world
    res = st.race(frame["irx"], frame["etf"])
    assert np.isfinite(res["t_hac"]) and np.isfinite(res["t_naive"])
    assert np.isfinite(st.welch_t(res["frame"]["r_ladder"].to_numpy(),
                                  res["frame"]["r_etf"].to_numpy()))
    assert np.isfinite(st.one_sample_t(res["diff"].to_numpy()))


def test_wilson_interval_brackets_point():
    lo, hi = st.wilson_interval(60, 100)
    assert lo < 0.60 < hi


def test_annualisation_honours_an_explicit_obs_per_year():
    idx = pd.bdate_range("2020-01-01", periods=504)
    d = pd.Series(1e-5, index=idx)
    assert st.annualise_gap_bps(d, obs_per_year=252.0) == pytest.approx(25.2, rel=1e-6)
    # a masked subset annualised on its own span would understate; on the parent it does not
    sub = d.iloc[::4]
    assert st.annualise_gap_bps(sub, obs_per_year=252.0) == pytest.approx(25.2, rel=1e-6)
    assert st.annualise_gap_bps(sub) < 10.0


def test_bootstrap_ci_brackets_point(fee_world):
    frame, _ = fee_world
    res = st.race(frame["irx"], frame["etf"])
    ci = st.bootstrap_gap_ci(res["diff"], n_boot=400, seed=921)
    assert ci["ci_low"] <= ci["gap_bps"] <= ci["ci_high"]


def test_bootstrap_handles_a_tiny_series():
    d = pd.Series([1e-5] * 5, index=pd.bdate_range("2020-01-01", periods=5))
    ci = st.bootstrap_gap_ci(d)
    assert np.isnan(ci["gap_bps"])


# --------------------------------------------------------------------------- #
# The study's spine — machinery recovers a planted fee and is quiet on the null
# --------------------------------------------------------------------------- #
def test_planted_fee_is_recovered(fee_world):
    frame, truth = fee_world
    d = st.synthetic_detect(frame)
    assert abs(d["gap_bps"] - truth["fee_bps_effective"]) < 3.0
    assert d["t_hac"] > 2.0


def test_null_shows_no_gap(free_world):
    frame, _ = free_world
    d = st.synthetic_detect(frame)
    assert abs(d["gap_bps"]) < 4.0
    assert abs(d["t_hac"]) < 2.0


def test_null_across_seeds_is_centred():
    gaps = np.array([
        st.synthetic_detect(data.synthetic_daily(signal_strength=0.0, seed=921 + s)[0])["gap_bps"]
        for s in range(6)
    ])
    assert abs(gaps.mean()) < 2.0
    assert (np.abs(gaps) >= 4.0).sum() == 0


def test_half_fee_recovers_half_the_gap():
    frame, truth = data.synthetic_daily(signal_strength=0.5, seed=921)
    d = st.synthetic_detect(frame)
    assert abs(d["gap_bps"] - truth["fee_bps_effective"]) < 3.0


def test_friction_can_eat_the_planted_fee(fee_world):
    """Charge enough per roll and even a genuine fee advantage disappears."""
    frame, _ = fee_world
    rows = st.friction_sweep(frame["irx"], frame["etf"], cost_grid=(0.0, 5.0))
    assert rows[0]["gap_bps"] > rows[1]["gap_bps"]
    assert rows[1]["gap_bps"] < 0.0


def test_fee_attribution_residual_is_small_on_the_planted_world(fee_world):
    frame, truth = fee_world
    a = st.fee_attribution(frame["irx"], frame["etf"], truth["fee_bps"])
    assert abs(a["residual_bps"]) < 3.0


def test_ladder_vol_is_far_below_the_etf_vol(fee_world):
    """The amortised-cost artefact: it must be visible, and it must not be read as safety."""
    frame, _ = fee_world
    d = st.synthetic_detect(frame)
    assert d["vol_ladder"] < d["vol_etf"]


def test_era_cut_partitions_the_sample(fee_world):
    frame, _ = fee_world
    res = st.race(frame["irx"], frame["etf"])
    eras = st.era_cut(frame["irx"], frame["etf"], edges=("2012-01-01", "2019-01-01"))
    live = [e for e in eras.values() if e is not None]
    assert len(live) == 3
    assert sum(e["n_days"] for e in live) == res["n_days"]
    for e in live:
        assert np.isfinite(e["gap_bps"])


def test_rate_regime_cut_returns_both_sides(fee_world):
    frame, _ = fee_world
    out = st.rate_regime_cut(frame["irx"], frame["etf"], threshold_pct=2.5)
    assert out["zero_rate"] is not None and out["normal_rate"] is not None
    assert out["zero_rate"]["mean_rate_pct"] < out["normal_rate"]["mean_rate_pct"]


def test_basis_check_raw_is_the_conservative_read(fee_world):
    frame, _ = fee_world
    b = st.basis_check(frame["irx"], frame["etf"])
    assert b["raw"]["gap_bps"] < b["discount"]["gap_bps"]


def test_rung_count_does_not_move_the_answer(fee_world):
    frame, _ = fee_world
    rows = st.rung_check(frame["irx"], frame["etf"])
    gaps = [r["gap_bps"] for r in rows]
    assert max(gaps) - min(gaps) < 2.0


# --------------------------------------------------------------------------- #
# The inference audit — the knob-free arbiter and the disclosed knobs
#
# This study's HAC t is LARGER than its naive t, so the audit machinery that
# justifies that has to be tested as hard as the ladder itself.
# --------------------------------------------------------------------------- #
def test_acf1_detects_bid_offer_bounce():
    # a pure first-difference of white noise has lag-1 autocorrelation near -0.5:
    # that is exactly the Roll (1984) bounce signature the study leans on
    rng = np.random.default_rng(921)
    e = rng.normal(size=4000)
    bounce = e[1:] - e[:-1]
    idx = pd.bdate_range("2010-01-01", periods=len(bounce))
    assert st.acf1(pd.Series(bounce, index=idx)) < -0.35
    # an i.i.d. series has none
    assert abs(st.acf1(pd.Series(e[:3999], index=idx))) < 0.06


def test_acf1_handles_degenerate_input():
    assert np.isnan(st.acf1(pd.Series([1.0])))
    assert np.isnan(st.acf1(pd.Series([2.0, 2.0, 2.0, 2.0])))


def test_nonoverlap_t_recovers_significance_hidden_by_bounce():
    """A drift buried under bid-offer bounce: naive daily t misses it, period sums find it.

    This is the study's central inference claim in miniature. The planted drift is real; the
    daily t is deflated by the bounce noise; summing into non-overlapping months lets the
    bounce telescope and exposes the drift with no bandwidth anywhere.
    """
    rng = np.random.default_rng(921)
    n = 3000
    idx = pd.bdate_range("2010-01-01", periods=n)
    noise = rng.normal(0.0, 3e-4, n)
    drift = 5e-6
    x = pd.Series(drift + noise - np.concatenate([[0.0], noise[:-1]]), index=idx)
    naive = st.one_sample_t(x.to_numpy())
    monthly = st.nonoverlap_t(x, freq="M")
    assert abs(naive) < 2.0                      # the bounce hides it daily
    assert monthly["t"] > 3.0                    # the knob-free test finds it
    assert monthly["n_periods"] < n              # genuinely aggregated


def test_nonoverlap_t_is_quiet_on_a_true_null():
    rng = np.random.default_rng(921)
    idx = pd.bdate_range("2010-01-01", periods=3000)
    x = pd.Series(rng.normal(0.0, 3e-4, 3000), index=idx)
    assert abs(st.nonoverlap_t(x, freq="M")["t"]) < 2.0


def test_nonoverlap_t_period_counts_are_ordered():
    idx = pd.bdate_range("2010-01-01", periods=1500)
    x = pd.Series(np.linspace(1e-6, 2e-6, 1500), index=idx)
    w = st.nonoverlap_t(x, "W")["n_periods"]
    m = st.nonoverlap_t(x, "M")["n_periods"]
    q = st.nonoverlap_t(x, "Q")["n_periods"]
    assert w > m > q >= 1


def test_nonoverlap_t_conserves_the_total():
    """Aggregation must move no return: sum of period sums == sum of daily values."""
    idx = pd.bdate_range("2010-01-01", periods=900)
    x = pd.Series(np.random.default_rng(921).normal(1e-5, 1e-4, 900), index=idx)
    for f in ("W", "M", "Q"):
        g = x.groupby(pd.PeriodIndex(x.index, freq=f)).sum()
        assert g.sum() == pytest.approx(x.sum(), rel=1e-12)


def test_nonoverlap_t_handles_degenerate_input():
    out = st.nonoverlap_t(pd.Series([1.0], index=pd.bdate_range("2020-01-01", periods=1)))
    assert np.isnan(out["t"]) and out["n_periods"] == 0


def test_horizon_check_agrees_with_hac_on_the_planted_world(fee_world):
    frame, truth = fee_world
    res = st.race(frame["irx"], frame["etf"])
    rows = st.horizon_check(res["diff"])
    assert [r["freq"] for r in rows] == ["W", "M", "Q"]
    # the planted fee is real, so every knob-free horizon must see it, as HAC does
    assert res["t_hac"] > 2.0
    assert all(r["t"] > 2.0 for r in rows)


def test_horizon_check_is_quiet_on_the_null(free_world):
    frame, _ = free_world
    res = st.race(frame["irx"], frame["etf"])
    assert all(abs(r["t"]) < 2.5 for r in st.horizon_check(res["diff"]))


def test_hac_bandwidth_scan_reports_naive_first_and_all_bandwidths(fee_world):
    frame, _ = fee_world
    res = st.race(frame["irx"], frame["etf"])
    rows = st.hac_bandwidth_scan(res["diff"])
    assert rows[0]["lags"] == 0 and rows[0]["label"] == "naive (i.i.d.)"
    assert rows[0]["t"] == pytest.approx(res["t_naive"], rel=1e-9)
    assert all(np.isfinite(r["t"]) for r in rows)
    # under bounce, widening the HAC window must not *lower* the t back to the naive one
    assert rows[-1]["t"] > rows[0]["t"]


def test_race_exposes_the_bounce_diagnostic(fee_world):
    frame, _ = fee_world
    res = st.race(frame["irx"], frame["etf"])
    # the synthetic ETF is built with an explicit bounce term, so acf1 must be negative
    assert res["acf1"] < -0.2
    assert res["t_hac"] > res["t_naive"]


def test_era_cut_carries_the_knob_free_t(fee_world):
    frame, _ = fee_world
    eras = st.era_cut(frame["irx"], frame["etf"], edges=("2012-01-01", "2018-01-01"))
    live = [e for e in eras.values() if e is not None]
    assert live and all(np.isfinite(e["t_month"]) for e in live)
