"""Data-layer tests — synthetic determinism and the payout reconstruction (all offline).

The one real-tape test is skipped cleanly when ``studies/_cache`` is absent, so the suite is
green on a fresh checkout with no network and no cache.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dist_illusion import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism, shape, portability
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, ta = data.synthetic_panel(seed=946)
    b, tb = data.synthetic_panel(seed=946)
    assert np.allclose(a["total"].to_numpy(), b["total"].to_numpy())
    assert np.allclose(a["price"].to_numpy(), b["price"].to_numpy())
    assert ta["planted_slope_per_sd"] == tb["planted_slope_per_sd"]


def test_synthetic_seed_changes_the_world():
    a, _ = data.synthetic_panel(seed=946)
    b, _ = data.synthetic_panel(seed=947)
    assert not np.allclose(a["total"].to_numpy(), b["total"].to_numpy())


def test_synthetic_shape_and_keys():
    panel, truth = data.synthetic_panel(n_funds=10, n_months=120, seed=946)
    for key in ("total", "price", "dist", "dist_rate", "cash", "bench", "funds"):
        assert key in panel
    assert panel["total"].shape == (120, 10)
    assert list(panel["total"].columns) == panel["funds"]
    assert truth["n_funds"] == 10 and truth["n_months"] == 120
    assert isinstance(panel["total"].index, pd.DatetimeIndex)
    # OOB-safe: the period_range index must stay well inside pandas' ns horizon.
    assert panel["total"].index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_reconstruction_identity_is_exact(null_panel):
    """price = (1+total)/(1+dist) - 1 by construction, so the gap recovers the payout."""
    panel, _ = null_panel
    implied = (1.0 + panel["total"]) / (1.0 + panel["price"]) - 1.0
    assert np.allclose(implied.to_numpy(), panel["dist"].to_numpy(), atol=1e-12)


def test_synthetic_payout_is_positive_and_ordered(null_panel):
    panel, truth = null_panel
    assert (panel["dist"].to_numpy() >= 0).all()
    mean_rate = panel["dist_rate"].mean()
    # Fund 0 has the lowest planted yield, the last fund the highest.
    assert mean_rate.iloc[0] < mean_rate.iloc[-1]
    assert truth["yields"][0] < truth["yields"][-1]


def test_dist_rate_is_trailing_only(null_panel):
    """Perturbing the future must not change any earlier trailing distribution rate."""
    panel, _ = null_panel
    base = panel["dist_rate"].copy()
    d2 = panel["dist"].copy()
    d2.iloc[100:] *= 5.0
    perturbed = np.exp(np.log1p(d2).rolling(12).sum()) - 1.0
    assert np.allclose(base.iloc[:100].to_numpy(), perturbed.iloc[:100].to_numpy(),
                       equal_nan=True)


def test_dist_rate_warmup_is_nan(null_panel):
    panel, _ = null_panel
    assert panel["dist_rate"].iloc[:11].isna().all().all()
    assert panel["dist_rate"].iloc[11:].notna().all().all()


def test_signal_strength_moves_the_planted_slope():
    _, t0 = data.synthetic_panel(signal_strength=0.0, seed=946)
    _, t1 = data.synthetic_panel(signal_strength=1.0, seed=946)
    assert t0["planted_slope_per_sd"] == 0.0
    assert t1["planted_slope_per_sd"] > 0.0


def test_beta_slope_tilts_betas():
    _, flat = data.synthetic_panel(beta_slope=0.0, seed=946)
    _, tilt = data.synthetic_panel(beta_slope=0.5, seed=946)
    assert np.allclose(flat["betas"], 1.0)
    assert tilt["betas"][0] > tilt["betas"][-1]  # high-yield funds carry less market


# --------------------------------------------------------------------------- #
# The monthly panel builder (exercised on synthetic daily tapes, no cache needed)
# --------------------------------------------------------------------------- #
def _fake_tapes(n_days=900, seed=3):
    """Two tiny daily tapes (total-return and price-only) with a known payout."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-02", periods=n_days, name="date")
    cols = {}
    price_cols = {}
    for i, name in enumerate(["A", "B", "C", "D", "E", "F", "G", data.BENCH, data.CASH]):
        r = rng.normal(0.0003, 0.008, n_days)
        tr = 100 * np.exp(np.cumsum(r))
        drag = (0.002 + 0.010 * i) / 252.0          # a per-fund payout drag on the price leg
        pr = tr * np.exp(-drag * np.arange(n_days))
        cols[name] = tr
        price_cols[name] = pr
    return (pd.DataFrame(cols, index=idx), pd.DataFrame(price_cols, index=idx))


def test_monthly_panel_shapes_and_payout_ordering():
    tr, pr = _fake_tapes()
    panel = data.monthly_panel(tr, pr, funds=["A", "B", "C", "D", "E", "F", "G"])
    assert panel["funds"] == ["A", "B", "C", "D", "E", "F", "G"]
    assert panel["cash"] is not None and panel["bench"] is not None
    rate = panel["dist_rate"].mean()
    assert rate["A"] < rate["G"]          # planted payout drag increases across the funds
    implied = (1.0 + panel["total"]) / (1.0 + panel["price"]) - 1.0
    assert np.allclose(implied.dropna().to_numpy(), panel["dist"].dropna().to_numpy())


def test_guard_masks_corporate_action_months():
    tr, pr = _fake_tapes()
    # Inject an unadjusted 1-for-2 reverse split into BOTH tapes on the same day.
    tr = tr.copy(); pr = pr.copy()
    tr.loc[tr.index[400]:, "A"] *= 2.0
    pr.loc[pr.index[400]:, "A"] *= 2.0
    guarded = data.monthly_panel(tr, pr, funds=["A", "B", "C"], guard=0.50)
    raw = data.monthly_panel(tr, pr, funds=["A", "B", "C"], guard=None)
    assert raw["total"]["A"].abs().max() > 0.5
    assert guarded["total"]["A"].abs().max() <= 0.5
    assert guarded["total"]["A"].isna().sum() > raw["total"]["A"].isna().sum()
    # The guard masks the return legs but must not touch the other funds.
    assert guarded["total"]["B"].isna().sum() == raw["total"]["B"].isna().sum()


def test_guard_none_keeps_everything():
    tr, pr = _fake_tapes()
    panel = data.monthly_panel(tr, pr, funds=["A", "B", "C"], guard=None)
    assert panel["guard"] is None
    assert panel["total"].iloc[1:].notna().all().all()


# --------------------------------------------------------------------------- #
# Cache contract and fingerprint
# --------------------------------------------------------------------------- #
def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_load_prices_rejects_bad_kind(tmp_path):
    with pytest.raises(ValueError):
        data.load_prices(kind="nav", cache_dir=str(tmp_path))


def test_fingerprint_stable_and_sensitive():
    tr, _ = _fake_tapes()
    fp = data.fingerprint(tr)
    assert fp == data.fingerprint(tr) and len(fp) == 12
    tr2 = tr.copy()
    tr2.iloc[10, 0] += 1.0
    assert fp != data.fingerprint(tr2)


def test_returns_fingerprint_survives_a_refetch_rescale():
    """The data stamp must reproduce when the feed back-adjusts the whole history.

    ``auto_adjust=True`` rescales every past close each time a new distribution lands, so a
    *level* fingerprint drifts without a single return changing. The stamp printed by
    verify.py fingerprints returns instead, which is invariant to that rescale — and still
    moves when one observation actually changes.
    """
    tr, _ = _fake_tapes()
    rescaled = tr * 1.0173          # exactly what a re-fetch does to the TR tape
    assert data.fingerprint(rescaled) != data.fingerprint(tr)
    assert data.returns_fingerprint(rescaled) == data.returns_fingerprint(tr)
    bumped = tr.copy()
    bumped.iloc[100, 0] *= 1.05
    assert data.returns_fingerprint(bumped) != data.returns_fingerprint(tr)


def test_price_leg_is_an_identity_not_a_second_experiment():
    """hml_price = hml_total - hml_payout, by construction of the payout proxy.

    The study leans on this: the "NAV erosion" t is the payout-persistence t carried over
    once the total leg contributes nothing, so it must never be sold as independent
    evidence. Asserted here so nobody can quietly re-read it as one.
    """
    tr, pr = _fake_tapes()
    panel = data.monthly_panel(tr, pr, funds=["A", "B", "C", "D", "E", "F", "G"])
    legs = st.sorted_legs(panel, min_funds=6)
    implied = legs["hml"] - legs["hml_d"]
    assert abs(implied.mean() - legs["hml_p"].mean()) < 2e-5      # < 0.2 bps/month
    assert np.corrcoef(legs["hml_p"], implied)[0, 1] > 0.999


def test_universe_constants_are_consistent():
    assert set(data.CORE_FUNDS).issubset(set(data.WIDE_FUNDS))
    assert data.BENCH not in data.FUNDS and data.CASH not in data.FUNDS
    assert set(data.TICKERS) == set(data.FUNDS) | {data.BENCH, data.CASH}


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no shared _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_pipeline_runs():
    tr = data.load_prices(kind="tr")
    pr = data.load_prices(kind="pr")
    assert tr.index[-1] <= pd.Timestamp(data.AS_OF)
    panel = data.monthly_panel(tr, pr)
    legs = st.sorted_legs(panel)
    assert len(legs) > 100
    # The high-payout leg really does pay more than the low-payout leg.
    assert legs["dhi"].mean() > legs["dlo"].mean()
    assert np.isfinite(st.giveback_ratio(legs))
