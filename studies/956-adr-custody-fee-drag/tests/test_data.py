"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adr_drag import data, strategy as st  # noqa: E402

COLS = {"adr_tr", "adr_px", "loc_tr", "loc_px", "fx", "loc_tr_usd", "loc_px_usd"}


# --------------------------------------------------------------------------- #
# The pair table itself
# --------------------------------------------------------------------------- #
def test_pairs_are_well_formed():
    assert len(data.PAIRS) >= 10
    for p in data.PAIRS:
        assert set(p) == {"adr", "local", "fx", "fx_invert", "ccy", "country", "wht", "name"}
        assert 0.0 <= p["wht"] < 0.5
        assert p["fx"].endswith("=X")
    assert len(set(p["adr"] for p in data.PAIRS)) == len(data.PAIRS)
    # Every UK pair carries a zero rate: the UK levies no dividend withholding tax.
    for p in data.PAIRS:
        if p["country"] == "United Kingdom":
            assert p["wht"] == 0.0


def test_pair_by_adr_roundtrip():
    for p in data.PAIRS:
        assert data.pair_by_adr(p["adr"])["local"] == p["local"]
    with pytest.raises(KeyError):
        data.pair_by_adr("NOPE")


def test_cache_filename_convention():
    assert data._safe("GBPUSD=X") == "GBPUSDX"
    assert data._safe("SHEL.L") == "SHEL-L"
    assert data._safe("NOVO-B.CO") == "NOVO-B-CO"
    assert data._cache_path("TM", "/c", "tr").endswith("prices_TM_1d.parquet")
    assert data._cache_path("TM", "/c", "px").endswith("praw_TM_1d.parquet")


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_pair(seed=956)
    b, _ = data.synthetic_pair(seed=956)
    for c in ("adr_tr", "adr_px", "loc_tr", "fx"):
        assert np.allclose(a[c].to_numpy(), b[c].to_numpy())


def test_synthetic_shape_and_columns(planted):
    df, truth = planted
    assert COLS.issubset(df.columns)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) == truth["n_days"] == truth["n_years"] * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay inside pandas' ns horizon.
    assert df.index[-1] < pd.Timestamp("2262-01-01")
    assert (df[["adr_tr", "adr_px", "loc_tr", "loc_px", "fx"]] > 0).all().all()


def test_synthetic_seed_sensitivity():
    a, _ = data.synthetic_pair(seed=956)
    b, _ = data.synthetic_pair(seed=957)
    assert not np.allclose(a["adr_px"].to_numpy(), b["adr_px"].to_numpy())


def test_total_return_dominates_price_only(planted):
    """Both legs must accumulate a distribution yield: TR/price rises through the sample."""
    df, _ = planted
    for tr, px in (("adr_tr", "adr_px"), ("loc_tr", "loc_px")):
        ratio = (df[tr] / df[px]).to_numpy()
        assert ratio[-1] > ratio[0]


def test_signal_strength_zero_kills_the_fee():
    _, t1 = data.synthetic_pair(signal_strength=1.0, seed=956)
    _, t0 = data.synthetic_pair(signal_strength=0.0, seed=956)
    assert t1["total_drag_per_year"] > 0 and t1["planted_gap_per_year"] > 0
    assert t0["total_drag_per_year"] == 0.0
    assert abs(t0["planted_gap_per_year"]) < 1e-12


def test_fee_lands_on_the_income_leg_not_the_price_leg(planted, null_pair):
    """The custody fee is netted from the dividend, so the price ratio must be unaffected."""
    d1, _ = planted
    d0, _ = null_pair
    x1 = np.log(d1["adr_px"]) - np.log(d1["loc_px_usd"])
    x0 = np.log(d0["adr_px"]) - np.log(d0["loc_px_usd"])
    assert np.allclose(x1.to_numpy(), x0.to_numpy())


def test_ratio_break_is_a_step(broken_pair):
    df, truth = broken_pair
    x = np.log(df["adr_px"]) - np.log(df["loc_px_usd"])
    jump = float(x.iloc[len(df) // 2] - x.iloc[len(df) // 2 - 1])
    assert abs(jump - truth["ratio_break"]) < 0.05


def test_synthetic_panel_shape(planted_panel):
    frames, truth = planted_panel
    assert len(frames) == truth["n_names"] == 10
    assert all(COLS.issubset(df.columns) for df in frames.values())
    keys = list(frames)
    assert not np.allclose(frames[keys[0]]["adr_px"].to_numpy(),
                           frames[keys[1]]["adr_px"].to_numpy())


def test_clean_mask_flags_a_planted_bad_print(planted):
    df, _ = planted
    x = np.log(df["adr_px"]) - np.log(df["loc_px_usd"])
    x = x.copy()
    x.iloc[500] += 0.9  # a corrupt close
    m = data.clean_mask(x)
    assert bool(m.iloc[500])
    assert m.sum() <= 3


def test_fingerprint_stable_and_sensitive(planted):
    df, _ = planted
    fp = data.fingerprint(df)
    assert fp == data.fingerprint(df) and len(fp) == 12
    other, _ = data.synthetic_pair(seed=957)
    assert fp != data.fingerprint(other)


# --------------------------------------------------------------------------- #
# The offline loaders never reach the network
# --------------------------------------------------------------------------- #
def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(["TM"], cache_dir=str(tmp_path))


def test_have_real_false_on_empty_cache(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no shared _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_panel_runs():
    frames = {p["adr"]: data.load_pair(p) for p in data.PAIRS}
    whts = {p["adr"]: p["wht"] for p in data.PAIRS}
    for df in frames.values():
        assert df.index[-1] <= pd.Timestamp(data.AS_OF)
    kept, report = st.screen_frames(frames)
    assert 0 < len(kept) < len(frames)          # the LSE names must fail the screen
    tbl = st.panel_table(kept, whts)
    for c in ("drag_total", "income_gap", "price_drift"):
        assert np.isfinite(pd.to_numeric(tbl[c])).all()
