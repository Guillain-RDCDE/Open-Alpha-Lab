"""Data-layer tests — the hardcoded IPO list, synthetic determinism, and a skipif smoke test.

Everything except the final smoke test is offline and cache-free: the suite is green on a
fresh checkout where ``studies/_cache`` does not exist.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cef_ipo import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The hardcoded IPO list is itself an input — check its integrity
# --------------------------------------------------------------------------- #
def test_ipo_list_shape_and_uniqueness():
    tickers = [t for t, _, _, _ in data.CEF_IPOS]
    assert len(tickers) == len(set(tickers)) >= 20
    assert set(tickers) == set(data.FUND_TICKERS)


def test_offering_prices_are_the_documented_syndicate_prices():
    """Every offer is $20 or $25 — the ASSUMPTION named in the README."""
    offers = {p for _, p, _, _ in data.CEF_IPOS}
    assert offers.issubset({20.0, 25.0})


def test_every_fund_maps_to_a_downloaded_benchmark():
    for tk, _, _, bench in data.CEF_IPOS:
        assert bench in data.BENCH_TICKERS
        assert bench in data.TICKERS


def test_alt_mappings_cover_every_fund():
    for name, mp in data.ALT_MAPPINGS.items():
        assert set(mp) == set(data.FUND_TICKERS), name
        assert set(mp.values()).issubset(set(data.BENCH_TICKERS)), name


def test_vintages_span_more_than_one_market_episode():
    """A single-vintage list would just be one macro episode wearing 28 tickers."""
    assert len(data.LOAD_BAND_PCT) == 3
    assert min(data.LOAD_BAND_PCT) >= 1.0 and max(data.LOAD_BAND_PCT) <= 10.0


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_daily_is_deterministic():
    a, _ = data.synthetic_daily(seed=931)
    b, _ = data.synthetic_daily(seed=931)
    assert np.allclose(a["fund"].to_numpy(), b["fund"].to_numpy())
    assert np.allclose(a["bench"].to_numpy(), b["bench"].to_numpy())


def test_synthetic_daily_shape_and_columns():
    px, truth = data.synthetic_daily(n_days=500, seed=931)
    assert {"fund", "bench"}.issubset(px.columns)
    assert isinstance(px.index, pd.DatetimeIndex)
    assert len(px) == truth["n_days"] == 500
    # OOB-safe: the synthetic index must stay inside pandas' nanosecond horizon.
    assert px.index[-1] < pd.Timestamp("2262-01-01")


def test_signal_strength_scales_the_planted_slide():
    _, t1 = data.synthetic_daily(signal_strength=1.0, seed=931)
    _, t0 = data.synthetic_daily(signal_strength=0.0, seed=931)
    assert t1["planted_slide"] > 0.05
    assert t0["planted_slide"] == 0.0


def test_synthetic_panel_is_staggered_and_deterministic():
    p1, truth = data.synthetic_panel(n_funds=6, seed=931)
    p2, _ = data.synthetic_panel(n_funds=6, seed=931)
    assert len(p1) == truth["n_funds"] == 6
    starts = [px.index[0] for px in p1.values()]
    assert starts == sorted(starts) and starts[0] < starts[-1]
    for k in p1:
        assert np.allclose(p1[k]["fund"].to_numpy(), p2[k]["fund"].to_numpy())


def test_synthetic_panel_seeds_differ_per_fund():
    panel, _ = data.synthetic_panel(n_funds=4, seed=931)
    keys = list(panel)
    a = panel[keys[0]]["bench"].pct_change().dropna().to_numpy()
    b = panel[keys[1]]["bench"].pct_change().dropna().to_numpy()
    n = min(len(a), len(b))
    assert not np.allclose(a[:n], b[:n])


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_daily(seed=931)
    b, _ = data.synthetic_daily(seed=932)
    fp = data.fingerprint(a)
    assert fp == data.fingerprint(a) and len(fp) == 12
    assert fp != data.fingerprint(b)


def test_fingerprint_is_invariant_to_yahoo_rescaling():
    """Yahoo rescales a whole adjusted-close history on every distribution.

    That multiplies every level by a constant, changes no return and no number in this
    study — so the data stamp must not churn for it.
    """
    a, _ = data.synthetic_daily(seed=931)
    assert data.fingerprint(a) == data.fingerprint(a * 1.0731)


def test_synthetic_panel_can_plant_leverage():
    """The beta knob has to reach the panel — it is what the beta control is tested on."""
    lev, truth = data.synthetic_panel(n_funds=3, beta=1.6, signal_strength=0.0, seed=931)
    flat, _ = data.synthetic_panel(n_funds=3, beta=1.0, signal_strength=0.0, seed=931)
    assert truth["beta"] == 1.6
    for k in lev:
        rl = lev[k]["fund"].pct_change(fill_method=None).dropna().std()
        rf = flat[k]["fund"].pct_change(fill_method=None).dropna().std()
        assert rl > rf


# --------------------------------------------------------------------------- #
# Offline loaders refuse to invent data
# --------------------------------------------------------------------------- #
def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_load_raw_closes_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_raw_closes(cache_dir=str(tmp_path))


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no shared _cache present (fresh checkout / CI) — "
                           "the synthetic tests cover the logic")
def test_real_cache_event_study_runs():
    px = data.load_prices()
    raw = data.load_raw_closes()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    tbl = st.build_table(px, data.CEF_IPOS, raw=raw)
    assert len(tbl) == len(data.CEF_IPOS)
    assert np.isfinite(tbl["abn_12m"].to_numpy(dtype=float)).all()
    # The offering-price ASSUMPTION must be corroborated by the tape: the day-one close
    # of a syndicate-supported CEF IPO sits within a few percent of the offering price.
    assert (tbl["day1_gap_pct"].abs() <= 8.0).all()
