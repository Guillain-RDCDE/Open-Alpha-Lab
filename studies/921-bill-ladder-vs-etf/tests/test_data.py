"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bill_ladder import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(seed=921)
    b, _ = data.synthetic_daily(seed=921)
    assert np.allclose(a["irx"].to_numpy(), b["irx"].to_numpy())
    assert np.allclose(a["etf"].to_numpy(), b["etf"].to_numpy())


def test_synthetic_shape_and_columns():
    frame, truth = data.synthetic_daily(n_years=10, seed=921)
    assert {"irx", "etf"}.issubset(frame.columns)
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert len(frame) == truth["n_days"] == 10 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay inside pandas' ns horizon.
    assert frame.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_rate_is_positive_and_mean_reverts():
    frame, truth = data.synthetic_daily(seed=921)
    assert (frame["irx"] > 0).all()
    # the OU path should sit near its long-run level over 20 years
    assert abs(truth["mean_rate"] - truth["theta"]) < 0.01


def test_signal_strength_scales_the_planted_fee():
    _, t1 = data.synthetic_daily(signal_strength=1.0, seed=921)
    _, t0 = data.synthetic_daily(signal_strength=0.0, seed=921)
    _, th = data.synthetic_daily(signal_strength=0.5, seed=921)
    assert t0["fee_bps_effective"] == 0.0
    assert t1["fee_bps_effective"] == pytest.approx(13.5)
    assert th["fee_bps_effective"] == pytest.approx(6.75)


def test_synthetic_etf_is_a_total_return_index():
    frame, _ = data.synthetic_daily(seed=921)
    # a cash index grinds upward; it must never be non-positive
    assert (frame["etf"] > 0).all()
    assert frame["etf"].iloc[-1] > frame["etf"].iloc[0]


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_daily(seed=921)
    fp1 = data.fingerprint(a)
    fp2 = data.fingerprint(a)
    b, _ = data.synthetic_daily(seed=922)
    assert fp1 == fp2 and len(fp1) == 12
    assert fp1 != data.fingerprint(b)


def test_expense_ratios_are_labelled_proxies():
    # they are reference stickers, not tape — sanity-bound them so a typo cannot
    # silently distort the fee attribution
    for tk, er in data.EXPENSE_RATIO_BPS.items():
        assert tk in data.ETFS
        assert 0.0 < er < 50.0


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
CACHE = data.DEFAULT_CACHE


@pytest.mark.skipif(not data.have_real(cache_dir=CACHE),
                    reason="no shared _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_race_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    res = st.race(px["^IRX"].dropna(), px["BIL"].dropna())
    for k in ("gap_bps", "t_hac", "cagr_ladder", "cagr_etf"):
        assert np.isfinite(res[k])
    # a bill ladder and a bill ETF must live in the same postcode
    assert abs(res["cagr_ladder"] - res["cagr_etf"]) < 0.005
