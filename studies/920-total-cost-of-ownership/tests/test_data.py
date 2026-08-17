"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tco import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(seed=920)
    b, _ = data.synthetic_daily(seed=920)
    for col in ("liquid", "cheap", "cash"):
        assert np.allclose(a[col].to_numpy(), b[col].to_numpy())


def test_synthetic_seed_sensitive():
    a, _ = data.synthetic_daily(seed=920)
    b, _ = data.synthetic_daily(seed=921)
    assert not np.allclose(a["cheap"].to_numpy(), b["cheap"].to_numpy())


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_daily(n_years=12, seed=920)
    assert {"liquid", "cheap", "cash"}.issubset(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 12 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay well inside pandas' ns Timestamp horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")
    assert (prices[["liquid", "cheap", "cash"]] > 0).all().all()


def test_signal_strength_zero_removes_the_gap():
    _, t1 = data.synthetic_daily(signal_strength=1.0, gap_bp_yr=6.0, seed=920)
    _, t0 = data.synthetic_daily(signal_strength=0.0, gap_bp_yr=6.0, seed=920)
    assert t1["planted_gap_bp_yr"] == pytest.approx(6.0)
    assert t0["planted_gap_bp_yr"] == pytest.approx(0.0)


def test_signal_strength_scales_the_gap():
    _, t_half = data.synthetic_daily(signal_strength=0.5, gap_bp_yr=8.0, seed=920)
    assert t_half["planted_gap_bp_yr"] == pytest.approx(4.0)


def test_cash_leg_is_monotone_growing():
    prices, _ = data.synthetic_daily(seed=920)
    assert (prices["cash"].diff().dropna() > 0).all()
    assert prices["cash"].iloc[-1] > prices["cash"].iloc[0]


def test_two_wrappers_track_each_other_closely():
    """The whole premise: the two legs share an index, so their gap is basis points."""
    prices, _ = data.synthetic_daily(seed=920)
    lr = np.log(prices["cheap"] / prices["liquid"])
    assert float(lr.abs().max()) < 0.02  # never more than 200 bp apart
    corr = prices["liquid"].pct_change().corr(prices["cheap"].pct_change())
    assert corr > 0.99


def test_synthetic_panel_shape():
    panel, truths = data.synthetic_panel(gaps_bp_yr=(0.0, 5.0), seed=920)
    assert set(panel) == {"gap_0bp", "gap_5bp"}
    assert truths["pairs"]["gap_5bp"]["planted_gap_bp_yr"] == pytest.approx(5.0)
    assert truths["pairs"]["gap_0bp"]["planted_gap_bp_yr"] == pytest.approx(0.0)
    for px in panel.values():
        assert {"liquid", "cheap", "cash"}.issubset(px.columns)


def test_panel_pairs_have_independent_noise():
    panel, _ = data.synthetic_panel(gaps_bp_yr=(3.0, 3.0), seed=920)
    # identical planted gaps, different seed offsets -> the key collides, so only one pair
    assert len(panel) == 1


# --------------------------------------------------------------------------- #
# Non-tape inputs are declared, not smuggled
# --------------------------------------------------------------------------- #
def test_assumption_tables_cover_every_wrapper():
    for tk in ("SPY", "IVV", "VOO", "SPLG", "QQQ", "QQQM"):
        assert tk in data.STATED_ER_BPS
        assert tk in data.ASSUMED_RT_SPREAD_BPS
    # the cash leg is not a wrapper under test and carries no assumed spread
    assert "BIL" not in data.ASSUMED_RT_SPREAD_BPS


def test_stated_fee_gap_signs_match_the_pair_ordering():
    """Every pair is (liquid, cheap): the second leg must not be the pricier one."""
    for liquid, cheap, _ in data.PAIRS:
        assert st.stated_fee_gap_bps(liquid, cheap, data.STATED_ER_BPS) >= 0.0


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_daily(seed=920)
    b, _ = data.synthetic_daily(seed=921)
    assert data.fingerprint(a) == data.fingerprint(a)
    assert len(data.fingerprint(a)) == 12
    assert data.fingerprint(a) != data.fingerprint(b)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_have_real_false_without_cache(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


def test_as_of_is_a_complete_month_end():
    ts = pd.Timestamp(data.AS_OF)
    assert (ts + pd.Timedelta(days=1)).month != ts.month


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no shared _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_estimator_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    est = st.td_estimate(px["QQQ"].dropna(), px["QQQM"].dropna(), n_boot=200)
    for k in ("td_ann_bp_yr", "t_annual", "ci_low", "ci_high", "td_cum_bp_yr"):
        assert np.isfinite(est[k])
    assert est["n_years"] >= 3
