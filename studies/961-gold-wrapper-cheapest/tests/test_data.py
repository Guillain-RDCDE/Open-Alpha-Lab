"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from which_gold import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(seed=961)
    b, _ = data.synthetic_panel(seed=961)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_shape_and_columns(planted):
    prices, truth = planted
    assert truth["n_funds"] == 5
    assert set(truth["fund_cols"]) | {"bench", "cash"} == set(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"]
    # OOB-safe: the synthetic index must stay inside pandas' ns horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_cash_is_monotone_growing(planted):
    prices, _ = planted
    assert (prices["cash"].diff().dropna() > 0).all()
    assert prices["cash"].iloc[-1] > prices["cash"].iloc[0]


def test_signal_strength_zero_collapses_the_fee_ladder():
    _, t1 = data.synthetic_panel(signal_strength=1.0, seed=961)
    _, t0 = data.synthetic_panel(signal_strength=0.0, seed=961)
    assert t0["fee_spread_bps"] == pytest.approx(0.0, abs=1e-9)
    assert t1["fee_spread_bps"] > 25.0
    # the *published* sheet is unchanged in both worlds — that is the point of the null
    assert t0["fees_bps"] == t1["fees_bps"]


def test_planted_fees_are_actually_charged(planted):
    """Each planted wrapper's endpoint drift vs the bullion truth ~ minus its fee."""
    prices, truth = planted
    for col, fee in zip(truth["fund_cols"], truth["fees_eff_bps"]):
        td = st.td_endpoint(prices[col], prices["bench"])
        assert td == pytest.approx(-fee, abs=8.0)


def test_fingerprint_stable_and_sensitive(planted):
    prices, _ = planted
    fp1 = data.fingerprint(prices)
    other, _ = data.synthetic_panel(seed=962)
    assert fp1 == data.fingerprint(prices) and len(fp1) == 12
    assert fp1 != data.fingerprint(other)


# --------------------------------------------------------------------------- #
# The fee ASSUMPTION layer
# --------------------------------------------------------------------------- #
def test_fee_sheet_covers_the_cohort():
    assert set(data.FEE_BPS) == set(data.COHORT)
    assert min(data.FEE_BPS, key=data.FEE_BPS.get) == "GLDM"
    assert max(data.FEE_BPS, key=data.FEE_BPS.get) == "GLD"


def test_blended_fee_sits_between_the_two_levels():
    """GLDM's blended fee over the full window is between its 18 bp and 10 bp levels."""
    b = data.blended_fee_bps("GLDM", "2018-06-26", "2026-06-30")
    assert 10.0 < b < 18.0
    # a wrapper with no recorded change blends to its flat sheet fee
    assert data.blended_fee_bps("GLD", "2018-06-26", "2026-06-30") == pytest.approx(40.0)
    # over a window entirely after the cut, the blend collapses to the current fee
    assert data.blended_fee_bps("GLDM", "2021-01-01", "2026-06-30") == pytest.approx(10.0)


def test_fee_vector_shapes():
    flat = data.fee_vector()
    blended = data.fee_vector(blended_over=("2018-06-26", "2026-06-30"))
    assert set(flat) == set(blended) == set(data.COHORT)
    assert blended["GLDM"] > flat["GLDM"]


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_load_dollar_volume_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_dollar_volume(cache_dir=str(tmp_path))


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no shared _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_pair_spread_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    coh = data.cohort_frame(px)
    ps = st.pair_spread(coh, "GLDM", "GLD")
    for k in ("spread_bpy", "tstat", "daily_bpy", "ci_low", "ci_high"):
        assert np.isfinite(ps[k])
    # the five wrappers hold the same metal: pairwise daily noise is basis points, not points
    assert ps["sd_daily_bp"] < 50.0
