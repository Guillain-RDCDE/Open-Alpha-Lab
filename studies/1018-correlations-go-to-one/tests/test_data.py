"""Data-layer tests for Study 1018 — synthetic determinism offline, pinned tapes gated."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from goestoone import data  # noqa: E402


def test_synthetic_panel_is_deterministic():
    a, ma, _ = data.synthetic_panel(n_assets=5, n_years=3, seed=1018)
    b, mb, _ = data.synthetic_panel(n_assets=5, n_years=3, seed=1018)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert np.allclose(ma.to_numpy(), mb.to_numpy())


def test_synthetic_panel_seed_sensitive():
    a, _, _ = data.synthetic_panel(n_assets=5, n_years=3, seed=1018)
    b, _, _ = data.synthetic_panel(n_assets=5, n_years=3, seed=1019)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_panel_shape_index_and_market():
    R, m, truth = data.synthetic_panel(n_assets=7, n_years=4, seed=1018)
    assert R.shape == (4 * data.TRADING_DAYS, 7)
    assert isinstance(R.index, pd.DatetimeIndex) and R.index.is_monotonic_increasing
    assert R.index[-1] < pd.Timestamp("2262-01-01")
    assert np.allclose(m.to_numpy(), R.mean(axis=1).to_numpy())
    assert truth["n_days"] == len(R)


def test_signal_strength_scales_the_planted_jump():
    _, _, t1 = data.synthetic_panel(signal_strength=1.0, n_years=3)
    _, _, th = data.synthetic_panel(signal_strength=0.5, n_years=3)
    _, _, t0 = data.synthetic_panel(signal_strength=0.0, n_years=3)
    assert t0["planted_jump"] == 0.0
    assert t1["planted_jump"] > th["planted_jump"] > 0.0
    assert t0["rho"].nunique() == 1                       # null: correlation constant
    assert t1["rho"].max() == pytest.approx(t1["rho_crisis"])
    assert t1["crisis"].mean() == pytest.approx(0.10, abs=0.01)


def test_null_has_constant_true_correlation_but_volatility_clusters():
    R, _, t = data.synthetic_panel(n_assets=6, n_years=10, signal_strength=0.0, seed=3)
    sig = t["sigma"]
    assert sig.max() / sig.min() > 5                      # volatility genuinely moves
    # realised correlation in the latent crisis state matches the calm state (truly constant)
    C_hi = np.corrcoef(R[t["crisis"]].T.to_numpy())
    C_lo = np.corrcoef(R[~t["crisis"]].T.to_numpy())
    iu = np.triu_indices(6, 1)
    assert abs(C_hi[iu].mean() - C_lo[iu].mean()) < 0.05


def test_planted_correlation_is_recovered_in_the_latent_state():
    R, _, t = data.synthetic_panel(n_assets=6, n_years=12, signal_strength=1.0, seed=3)
    C_hi = np.corrcoef(R[t["crisis"]].T.to_numpy())
    iu = np.triu_indices(6, 1)
    assert C_hi[iu].mean() == pytest.approx(t["rho_crisis"], abs=0.06)


def test_invalid_correlation_is_refused():
    with pytest.raises(ValueError):
        data.synthetic_panel(rho0=0.8, crisis_jump=0.3, n_years=1)


def test_fingerprint_stable_and_sensitive():
    a, _, _ = data.synthetic_panel(n_assets=4, n_years=2, seed=1)
    b, _, _ = data.synthetic_panel(n_assets=4, n_years=2, seed=2)
    assert data.fingerprint(a) == data.fingerprint(a) and len(data.fingerprint(a)) == 12
    assert data.fingerprint(a) != data.fingerprint(b)


def test_as_of_is_a_full_month_inside_the_tape():
    asof = pd.Timestamp(data.AS_OF)
    assert asof == asof + pd.offsets.MonthEnd(0)          # a month end
    assert asof < pd.Timestamp("2022-12-28")              # the tape's last, partial, month dropped
    assert len(data.TICKERS) == 20 == len(set(data.TICKERS))


def test_have_real_false_on_empty_cache(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


needs_real = pytest.mark.skipif(not data.have_real(),
                                reason="skfolio tapes not cached — synthetic tests cover logic")


@needs_real
def test_real_tapes_load_aligned_and_pinned():
    R, m = data.load_returns()
    assert list(R.columns) == list(data.TICKERS)
    assert R.index.equals(m.index)
    assert R.index[-1] <= pd.Timestamp(data.AS_OF)
    assert R.index[-1] >= pd.Timestamp(data.AS_OF) - pd.Timedelta(days=5)
    assert R.index[0] < pd.Timestamp("1990-01-10")
    assert not R.isna().any().any() and not m.isna().any()
    assert R.index.is_monotonic_increasing and not R.index.has_duplicates
    # daily returns of large caps: sane magnitudes
    assert R.abs().max().max() < 1.0 and m.abs().max() < 0.25


@needs_real
def test_real_fingerprints_are_stable():
    px = data.load_prices()
    assert data.fingerprint(px) == data.fingerprint(data.load_prices())
    assert len(px) > 8000
