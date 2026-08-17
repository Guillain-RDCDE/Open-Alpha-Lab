"""Data-layer contracts — offline only. Nothing here touches the network or the cache.

The real tape is exercised through ``have_real`` / ``_cache_path`` only (pure path logic);
every behavioural test runs on the deterministic synthetic panel, so the suite is green on
a fresh checkout with no ``studies/_cache`` present.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from two_ladders import data  # noqa: E402


# --------------------------------------------------------------------------- #
# Static declarations
# --------------------------------------------------------------------------- #
def test_pairs_are_well_formed():
    assert len(data.PAIRS) == 8
    labels = [lbl for lbl, _, _, _ in data.PAIRS]
    assert len(set(labels)) == len(labels)
    for lbl, pref, bond, sector in data.PAIRS:
        assert lbl and pref and bond and sector
        assert pref != bond
    # No ticker is reused across the two ladders.
    assert not set(data.PREF_TICKERS) & set(data.BOND_TICKERS)


def test_every_pair_declares_what_its_two_rungs_actually_are():
    """The framing 'junior perpetual preferred vs senior dated note' must not be assumed.

    Each pair carries an explicit instrument description and a qualitative seniority-step
    size, because in most of this panel the step is one rung or less.
    """
    labels = [lbl for lbl, _, _, _ in data.PAIRS]
    assert set(data.RUNG_STRUCTURE) == set(labels)
    for lbl, (pref_desc, bond_desc, step) in data.RUNG_STRUCTURE.items():
        assert pref_desc and bond_desc
        assert step in {"narrow", "wide"}
    # The textbook case is the minority, and it is exactly the distressed pair set.
    wide = {k for k, v in data.RUNG_STRUCTURE.items() if v[2] == "wide"}
    assert wide == set(data.DISTRESSED)
    assert len(wide) < len(labels) / 2


def test_distressed_names_are_real_panel_members():
    labels = [lbl for lbl, _, _, _ in data.PAIRS]
    for name in data.DISTRESSED:
        assert name in labels


def test_ticker_universe_covers_every_leg_plus_bench_and_cash():
    for tk in data.PREF_TICKERS + data.BOND_TICKERS + data.BENCH + (data.CASH,):
        assert tk in data.TICKERS


def test_asof_is_a_complete_month_end():
    ts = pd.Timestamp(data.AS_OF)
    assert ts == ts + pd.offsets.MonthEnd(0)
    assert pd.Timestamp(data.PANEL_START) < ts


def test_cache_path_sanitises_and_is_shared():
    p = data._cache_path("CMS-PB", "/tmp/cache")
    assert p.endswith("prices_CMS-PB_1d.parquet")
    assert data._cache_path("^IRX", "/tmp/cache").endswith("prices_IRX_1d.parquet")
    # The cache is the SHARED desk cache, two levels up from the package.
    assert os.path.basename(data.DEFAULT_CACHE) == "_cache"
    assert os.path.basename(os.path.dirname(data.DEFAULT_CACHE)) == "studies"


def test_have_real_is_false_on_an_empty_directory(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


def test_load_prices_raises_without_cache(tmp_path):
    try:
        data.load_prices(cache_dir=str(tmp_path))
    except FileNotFoundError as exc:
        assert "fetch()" in str(exc)
    else:  # pragma: no cover - would mean the offline guard is broken
        raise AssertionError("load_prices must refuse to run without a cache")


def test_data_module_does_not_import_yfinance_at_module_level():
    src = open(data.__file__, encoding="utf-8").read()
    head = src.split("def fetch(")[0]
    assert "import yfinance" not in head


# --------------------------------------------------------------------------- #
# Synthetic panel
# --------------------------------------------------------------------------- #
def test_synthetic_shapes_and_alignment(planted):
    pref, bond, cash, truth = planted
    assert pref.shape == bond.shape
    assert list(pref.columns) == list(bond.columns)
    assert pref.index.equals(bond.index) and pref.index.equals(cash.index)
    assert pref.shape[1] == truth["n_issuers"] == 8
    assert len(pref) == truth["n_days"]
    assert not pref.isna().any().any() and not bond.isna().any().any()


def test_synthetic_is_deterministic():
    a = data.synthetic_panel(seed=933)[0]
    b = data.synthetic_panel(seed=933)[0]
    c = data.synthetic_panel(seed=934)[0]
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_signal_strength_scales_the_planted_premium():
    _, _, _, t1 = data.synthetic_panel(signal_strength=1.0, seed=933)
    _, _, _, th = data.synthetic_panel(signal_strength=0.5, seed=933)
    _, _, _, t0 = data.synthetic_panel(signal_strength=0.0, seed=933)
    assert t1["planted_premium_ann"] > th["planted_premium_ann"] > t0["planted_premium_ann"]
    assert t0["planted_premium_ann"] == 0.0


def test_planted_premium_is_in_the_mean_difference(planted):
    """The planted carry must actually be in the tape, not just in the truth dict."""
    pref, bond, _, truth = planted
    realised = float((pref - bond).mean().mean() * data.TRADING_DAYS_PER_YEAR)
    assert abs(realised - truth["planted_premium_ann"]) < 0.03


def test_null_difference_is_centred_on_zero(null_panel):
    pref, bond, _, _ = null_panel
    realised = float((pref - bond).mean().mean() * data.TRADING_DAYS_PER_YEAR)
    assert abs(realised) < 0.03


def test_junior_leg_is_the_more_volatile_rung(planted):
    """The preferred sits lower in the stack, so it must load harder on the credit factor."""
    pref, bond, _, truth = planted
    assert truth["pref_beta"] > truth["bond_beta"]
    assert pref.std().mean() > bond.std().mean()


def test_legs_of_one_issuer_are_correlated(planted):
    """Same balance sheet -> the two rungs must share the issuer's credit factor."""
    pref, bond, _, _ = planted
    corrs = [pref[c].corr(bond[c]) for c in pref.columns]
    assert min(corrs) > 0.4


def test_cash_leg_is_positive_and_flat(planted):
    _, _, cash, truth = planted
    assert (cash > 0).all()
    assert abs(cash.mean() * data.TRADING_DAYS_PER_YEAR - truth["cash_rate_ann"]) < 1e-9


def test_synthetic_daily_wrapper_is_a_single_pair():
    pref, bond, cash, truth = data.synthetic_daily(seed=933)
    assert truth["n_issuers"] == 1
    assert pref.shape[1] == bond.shape[1] == 1
    assert len(cash) == len(pref)


def test_panel_returns_aligns_and_labels(planted):
    """panel_returns must relabel by issuer and keep only fully-populated rows."""
    idx = pd.bdate_range("2022-02-01", periods=300)
    frame = pd.DataFrame(
        {tk: 100.0 * np.cumprod(1 + np.full(len(idx), 0.0002)) for tk in data.TICKERS},
        index=idx,
    )
    frame.loc[frame.index[:20], data.PAIRS[0][1]] = np.nan  # a late-listing leg
    pref, bond = data.panel_returns(frame, start="2022-02-01", asof="2026-06-30")
    assert list(pref.columns) == [lbl for lbl, _, _, _ in data.PAIRS]
    assert list(pref.columns) == list(bond.columns)
    assert len(pref) == len(idx) - 20 - 1  # 20 NaN rows dropped, 1 lost to pct_change
    assert not pref.isna().any().any()


def test_fingerprint_is_stable_and_content_sensitive(planted):
    pref = planted[0]
    assert data.fingerprint(pref) == data.fingerprint(pref.copy())
    bumped = pref.copy()
    bumped.iloc[0, 0] += 0.01
    assert data.fingerprint(pref) != data.fingerprint(bumped)
