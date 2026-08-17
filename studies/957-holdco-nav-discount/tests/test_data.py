"""Data-layer tests — synthetic determinism (offline) plus a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holdco_nav import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(seed=957, n_years=8)
    b, _ = data.synthetic_daily(seed=957, n_years=8)
    for col in ("stake", "holdco", "discount", "cash"):
        assert np.allclose(a[col].to_numpy(), b[col].to_numpy())


def test_synthetic_seed_sensitive():
    a, _ = data.synthetic_daily(seed=957, n_years=8)
    b, _ = data.synthetic_daily(seed=958, n_years=8)
    assert not np.allclose(a["discount"].to_numpy(), b["discount"].to_numpy())


def test_synthetic_shape_and_columns(one_pair):
    frame, truth = one_pair
    assert {"stake", "holdco", "nav", "discount", "cash"}.issubset(frame.columns)
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert len(frame) == truth["n_days"]
    # OOB-safe: the synthetic index must stay well inside pandas' ns Timestamp horizon.
    assert frame.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_discount_is_bounded_and_identity_holds(one_pair):
    frame, _ = one_pair
    d = frame["discount"]
    assert d.between(0.0, 1.0).all()
    # holdco = nav * (1 - d) must hold exactly, by construction
    assert np.allclose(frame["holdco"].to_numpy(),
                       (frame["nav"] * (1.0 - d)).to_numpy())


def test_null_discount_is_a_martingale():
    """At signal_strength=0 the discount must have no systematic drift (across seeds)."""
    drifts = []
    for s in range(8):
        f, _ = data.synthetic_daily(signal_strength=0.0, seed=957 + 13 * s, n_years=12)
        drifts.append(float(f["discount"].iloc[-1] - f["discount"].iloc[0]))
    drifts = np.array(drifts)
    assert abs(drifts.mean()) < 0.5 * drifts.std(ddof=1) + 0.05


def test_planted_discount_reverts_more_than_null():
    """The OU pull must make the planted discount visibly stickier around its mean."""
    ou, _ = data.synthetic_daily(signal_strength=1.0, seed=957, n_years=12)
    rw, _ = data.synthetic_daily(signal_strength=0.0, seed=957, n_years=12)
    assert st.halflife(ou["discount"]) < st.halflife(rw["discount"])


def test_panel_shape_and_determinism():
    a, ta = data.synthetic_panel(n_names=4, seed=957, n_years=8)
    b, _ = data.synthetic_panel(n_names=4, seed=957, n_years=8)
    assert len(a) == 4 and ta["n_names"] == 4
    for k in a:
        assert np.allclose(a[k]["discount"].to_numpy(), b[k]["discount"].to_numpy())
    # independent names: no two share a discount path
    keys = list(a)
    assert not np.allclose(a[keys[0]]["discount"].to_numpy(),
                           a[keys[1]]["discount"].to_numpy())


def test_cash_index_from_irx_is_monotone():
    irx = pd.Series(np.full(500, 4.0),
                    index=pd.bdate_range("2020-01-01", periods=500))
    cash = data.cash_index_from_irx(irx)
    assert (cash.diff().dropna() > 0).all()
    assert 1.07 < cash.iloc[-1] < 1.09    # ~4%/yr over ~2 years


def test_stakes_table_is_well_formed():
    """The NAV assumptions are a declared table: every entry must be complete and sane."""
    for name, cfg in data.STAKES.items():
        assert cfg["k"] > 0, name
        assert 0.0 < cfg["stake_share_nav"] <= 1.0, name
        assert cfg["holdco"] != cfg["stake"], name
        assert pd.Timestamp(cfg["start"]) < pd.Timestamp(data.AS_OF), name
        if cfg.get("end"):
            assert pd.Timestamp(cfg["end"]) > pd.Timestamp(cfg["start"]), name
        assert isinstance(cfg["note"], str) and len(cfg["note"]) > 20, name


def test_every_truncation_is_declared_both_ways():
    """A truncation must appear in the note AND in ``end`` — never one without the other.

    This is the invariant an audit caught: Liberty Broadband's post-2024 gap was documented
    as a merger spread in the prose while the code still ran the series to the as-of date.
    """
    for name, cfg in data.STAKES.items():
        says = "TRUNCATED" in cfg["note"]
        does = bool(cfg.get("end"))
        assert says == does, f"{name}: note says TRUNCATED={says} but end={cfg.get('end')!r}"
        if does:
            assert pd.Timestamp(cfg["end"]) < pd.Timestamp(data.AS_OF), name


def test_fingerprint_stable_and_sensitive(one_pair):
    frame, _ = one_pair
    fp = data.fingerprint(frame[["stake", "holdco"]])
    assert fp == data.fingerprint(frame[["stake", "holdco"]]) and len(fp) == 12
    other, _ = data.synthetic_daily(seed=958, n_years=8)
    assert fp != data.fingerprint(other[["stake", "holdco"]])


def test_load_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        data.load_price_only(cache_dir=str(tmp_path))


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the cache is absent (CI safe)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no real _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_panel_builds():
    px = data.load_price_only()
    tr = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    panel = st.build_panel(px, tr, data.STAKES, safe=data._safe)
    assert len(panel) == len(data.STAKES)
    for name, f in panel.items():
        assert {"d", "z", "h", "r_holdco", "r_stake"}.issubset(f.columns)
        assert len(f) > 500, name
        assert np.isfinite(f["d"]).all()
