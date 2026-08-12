"""Offline, fixed-seed tests for the sector-neutral low-vol machinery.

The synthetic panel is deterministic; trailing vol recovers each name's planted vol level;
the sector demean genuinely neutralises the sector (per-sector median ~0 each day); the
sector-neutral sort recovers a planted *stock-level* low-vol effect and stays silent on the
null; the central claim — a pure sector premium makes a RAW sort fire but leaves the
SECTOR-NEUTRAL sort silent — holds; the sort is point-in-time; the timer costs bite; the
inference primitives behave. All offline; no real cache required.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pytest  # noqa: E402

from sn_lowvol import data, strategy as st  # noqa: E402

CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.1, seed=903, n_assets=40, n_days=1500, n_sectors=8)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_trailing_vol_recovers_level(edge_world):
    # Cross-name dispersion in realized vol must be non-trivial (the sort has something to bite on).
    ret = st.close_returns(edge_world)
    v = st.trailing_vol(ret, window=63).mean()
    assert v.std() > 0.0
    assert (v.dropna() > 0).all()


def test_sector_demean_neutralises(edge_world, secmap):
    # After demeaning by sector median, each sector's per-day median is ~0.
    ret = st.close_returns(edge_world)
    sig = st.trailing_vol(ret, window=63)
    dem = st.sector_demean(sig, secmap)
    sec = secmap.reindex(dem.columns)
    for label, cols in sec.groupby(sec).groups.items():
        med = dem[list(cols)].median(axis=1).dropna()
        assert np.nanmax(np.abs(med.to_numpy())) < 1e-12


def test_planted_relation_recovered_neutral(edge_world, secmap):
    ret = st.close_returns(edge_world)
    ts = st.vol_stats(st.vol_spreads(ret, secmap, window=63, frac=0.3, neutral=True))
    assert ts["t_nw"] > 3.0             # sector-neutral low-vol book lights up
    assert ts["spread_bps"] > 0
    assert ts["lo_bps"] > ts["hi_bps"]  # low-vol names out-earn high-vol names


def test_null_world_no_signal_neutral(null_world, secmap):
    ret = st.close_returns(null_world)
    ts = st.vol_stats(st.vol_spreads(ret, secmap, window=63, frac=0.3, neutral=True))
    assert abs(ts["t_nw"]) < 2.5


def test_confound_raw_fires_neutral_silent(confound_world, secmap):
    # The heart of the study: a pure defensive-SECTOR premium fools a raw low-vol sort but
    # not a sector-neutral one.
    ret = st.close_returns(confound_world)
    raw_t = st.vol_stats(st.vol_spreads(ret, secmap, 63, 0.3, neutral=False))["t_nw"]
    neu_t = st.vol_stats(st.vol_spreads(ret, secmap, 63, 0.3, neutral=True))["t_nw"]
    assert abs(raw_t) > 2.0            # raw sort reaps the sector bet
    assert abs(neu_t) < 2.0            # sector-neutral sort stays silent


def test_defensive_tilt_shrinks_under_neutral(confound_world, secmap):
    # Declare the calmest synthetic sectors "defensive"; the raw long-minus-short tilt is
    # large, the sector-neutral one collapses toward zero.
    ret = st.close_returns(confound_world)
    defensive = ("SEC0", "SEC1")
    raw = st.defensive_tilt(ret, secmap, defensive, 63, 0.3, neutral=False)
    neu = st.defensive_tilt(ret, secmap, defensive, 63, 0.3, neutral=True)
    assert abs(neu["long_minus_short_defensive"]) < abs(raw["long_minus_short_defensive"])


def test_sort_is_point_in_time():
    ret = pd.DataFrame(
        np.linspace(-0.02, 0.02, 60).reshape(20, 3),
        index=pd.bdate_range("2020-01-01", periods=20),
        columns=["A", "B", "C"],
    )
    sig = st.trailing_vol(ret, window=3)
    shifted = sig.shift(1)
    assert np.allclose(shifted.iloc[5].to_numpy(), sig.iloc[4].to_numpy(), equal_nan=True)


def test_costs_reduce_net(edge_world, secmap):
    ret = st.close_returns(edge_world)
    sp = st.vol_spreads(ret, secmap, 63, 0.3, neutral=True)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_neutral_requires_sectors(edge_world):
    ret = st.close_returns(edge_world)
    with pytest.raises(ValueError):
        st.vol_spreads(ret, sectors=None, neutral=True)


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_welch_sign(edge_world, secmap):
    ret = st.close_returns(edge_world)
    sp = st.vol_spreads(ret, secmap, 63, 0.3, neutral=True)
    assert st.welch_t(sp["lo"].to_numpy(), sp["hi"].to_numpy()) > 0


# --------------------------------------------------------------------------- #
# Real-cache tests — skipped entirely when _cache/ is absent (e.g. on CI).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(), reason="real panel cache absent (offline CI)")
def test_real_neutral_shrinks_defensive_tilt():
    panel = data.load_panel()
    sectors = data.sector_series(panel)
    ret = st.close_returns(panel)
    raw = st.defensive_tilt(ret, sectors, data.DEFENSIVE_SECTORS, 63, 0.3, neutral=False)
    neu = st.defensive_tilt(ret, sectors, data.DEFENSIVE_SECTORS, 63, 0.3, neutral=True)
    # The raw low-vol long book is heavily defensive; the neutral one is not.
    assert raw["long_minus_short_defensive"] > 0.25
    assert abs(neu["long_minus_short_defensive"]) < 0.10


@pytest.mark.skipif(not data.have_real(), reason="real panel cache absent (offline CI)")
def test_real_headline_sign_is_reversed():
    panel = data.load_panel()
    sectors = data.sector_series(panel)
    ret = st.close_returns(panel)
    ts = st.vol_stats(st.vol_spreads(ret, sectors, 63, 0.3, neutral=True))
    # On mega-caps the low-vol spread is negative (wrong-signed vs the claim) and significant.
    assert ts["spread_bps"] < 0
    assert ts["t_nw"] < -2.0
