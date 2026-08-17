"""Data-layer tests — the dividend reconstruction and the synthetic generators.

Everything here is offline and cache-free: the synthetic tapes carry a *known*
dividend stream, so the reconstruction can be scored against the truth without any
network. The one real-cache test is skipped cleanly when ``studies/_cache`` is absent
(a fresh CI checkout).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drip_sweep import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_years=8, seed=939)
    b, _ = data.synthetic_daily(n_years=8, seed=939)
    for col in ("close", "dividend", "cash"):
        assert np.allclose(a[col].to_numpy(), b[col].to_numpy())


def test_synthetic_shape_and_columns():
    frame, truth = data.synthetic_daily(n_years=10, seed=939)
    assert {"close", "dividend", "cash"}.issubset(frame.columns)
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert len(frame) == truth["n_days"] == 10 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay inside pandas' ns horizon.
    assert frame.index[-1] < pd.Timestamp("2262-01-01")
    assert (frame["close"] > 0).all()


def test_synthetic_pays_roughly_the_planted_yield():
    frame, truth = data.synthetic_daily(n_years=10, div_yield_ann=0.06, seed=939)
    n_events = int((frame["dividend"] > 0).sum())
    assert n_events == pytest.approx(10 * truth["payments_per_year"], abs=2)
    # Per-event cash is the planted quarterly slice of the pre-drop price.
    ex = frame["dividend"] > 0
    q = frame.loc[ex, "dividend"] / (frame.loc[ex, "close"] + frame.loc[ex, "dividend"])
    assert q.max() - q.min() < 1e-9
    assert float(q.iloc[0]) == pytest.approx(0.06 / 4, rel=1e-6)


def test_synthetic_cash_leg_is_monotone():
    frame, truth = data.synthetic_daily(n_years=8, seed=939)
    assert (frame["cash"].diff().dropna() > 0).all()
    grew = frame["cash"].iloc[-1] / frame["cash"].iloc[0]
    n_steps = truth["n_days"] - 1
    assert grew == pytest.approx((1 + truth["cash_rate_ann"]) ** (n_steps / 252), rel=1e-9)


def test_signal_strength_scales_the_planted_premium():
    _, t1 = data.synthetic_daily(n_years=8, signal_strength=1.0, seed=939)
    _, th = data.synthetic_daily(n_years=8, signal_strength=0.5, seed=939)
    _, t0 = data.synthetic_daily(n_years=8, signal_strength=0.0, seed=939)
    assert t0["effective_premium"] == 0.0
    assert th["effective_premium"] == pytest.approx(t1["effective_premium"] / 2)


def test_null_tape_grows_like_cash():
    """At signal_strength=0 the fund's TOTAL return must equal the cash rate.

    Run at zero vol so the check is exact arithmetic rather than a one-draw
    realisation: the diffusion carries the whole total-return drift and the ex-date
    drops are handed back as cash, so a zero-lag DRIP must compound at the cash rate.
    """
    frame, truth = data.synthetic_daily(n_years=20, signal_strength=0.0,
                                        vol_ann=0.0, seed=939)
    flat = pd.Series(1.0, index=frame.index)
    sim = st.simulate(frame["close"], frame["dividend"], flat,
                      policy="drip", pay_lag_days=0, cost_bps=0.0)
    years = (len(frame) - 1) / 252
    cagr_fund = (sim["wealth"].iloc[-1] / 10_000.0) ** (1 / years) - 1
    # The generator uses a continuous drift, the cash leg a discrete one: 4.5 bp apart.
    assert cagr_fund == pytest.approx(truth["cash_rate_ann"], abs=0.001)


# --------------------------------------------------------------------------- #
# The dividend reconstruction — scored against a known truth
# --------------------------------------------------------------------------- #
def _total_return_index(frame):
    """Build the total-return leg the reconstruction is supposed to invert."""
    r = (frame["close"] + frame["dividend"]) / frame["close"].shift(1) - 1.0
    return (1.0 + r.fillna(0.0)).cumprod() * 100.0


def test_reconstruction_recovers_the_planted_dividends():
    frame, truth = data.synthetic_daily(n_years=12, seed=939)
    tr = _total_return_index(frame)
    rec = data.reconstruct_dividends(frame["close"], tr)
    truth_div = frame["dividend"]
    assert int((rec > 0).sum()) == int((truth_div > 0).sum()) == truth["n_events"]
    assert np.allclose(rec.to_numpy(), truth_div.to_numpy(), rtol=1e-6, atol=1e-8)


def test_reconstruction_is_silent_on_a_non_paying_tape():
    frame, _ = data.synthetic_daily(n_years=8, div_yield_ann=0.0, seed=939)
    tr = _total_return_index(frame)
    rec = data.reconstruct_dividends(frame["close"], tr)
    assert int((rec > 0).sum()) == 0


def test_reconstruction_check_scores_itself():
    frame, truth = data.synthetic_daily(n_years=10, seed=939)
    tr = _total_return_index(frame)
    chk = data.dividend_reconstruction_check(frame["close"], tr, frame["dividend"])
    assert chk["n_events_matched"] == truth["n_events"]
    assert chk["ratio_rec_over_rep"] == pytest.approx(1.0, abs=1e-4)
    assert chk["event_corr"] > 0.999


def test_noise_threshold_rejects_float_dust():
    """A 0.1 bp implied amount is rounding dust, not a distribution."""
    idx = pd.bdate_range("2020-01-02", periods=300)
    price = pd.Series(np.linspace(100.0, 130.0, 300), index=idx)
    tr = price * (1.0 + 1e-5)   # a constant 0.1 bp scale gap -> no ex-date anywhere
    rec = data.reconstruct_dividends(price, tr)
    assert float(rec.abs().max()) == 0.0


# --------------------------------------------------------------------------- #
# Panel generator
# --------------------------------------------------------------------------- #
def test_panel_shapes_and_yields():
    frames, truth = data.synthetic_panel(n_years=8, seed=939)
    assert set(frames) == set(data.PAYERS)
    for tk, y in zip(truth["tickers"], truth["div_yields"]):
        assert truth["per_ticker"][tk]["div_yield_ann"] == y
        assert {"close", "dividend", "cash"}.issubset(frames[tk].columns)


# --------------------------------------------------------------------------- #
# Cache contract
# --------------------------------------------------------------------------- #
def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_load_distributions_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_distributions(cache_dir=str(tmp_path))


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_daily(n_years=6, seed=939)
    b, _ = data.synthetic_daily(n_years=6, seed=940)
    fp = data.fingerprint(a)
    assert fp == data.fingerprint(a) and len(fp) == 12
    assert fp != data.fingerprint(b)


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when studies/_cache is absent (CI safe)
# --------------------------------------------------------------------------- #
CACHE = data.DEFAULT_CACHE


@pytest.mark.skipif(not data.have_real(cache_dir=CACHE),
                    reason="no real _cache present (offline / CI) - synthetic tests cover the logic")
def test_real_cache_reconstruction_matches_reported():
    tr = data.load_prices()
    dist = data.load_distributions()
    assert tr.index[-1] <= pd.Timestamp(data.AS_OF)
    for tk in data.PAYERS:
        chk = data.dividend_reconstruction_check(
            dist[tk]["close"].dropna(), tr[tk].dropna(), dist[tk]["dividend"])
        assert chk["n_events_matched"] == chk["n_events_reported"]
        assert 0.99 < chk["ratio_rec_over_rep"] < 1.01
