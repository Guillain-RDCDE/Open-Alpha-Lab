"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from advance_decline import data  # noqa: E402


def test_synthetic_shape(null_tape):
    spy, ad, truth = null_tape
    assert len(spy) == truth["n_days"]
    assert len(ad)  == truth["n_days"]
    assert list(spy.columns[:4]) == ["open", "high", "low", "close"]


def test_synthetic_ohlc_sanity(null_tape):
    spy, ad, truth = null_tape
    assert (spy["high"] >= spy[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (spy["low"]  <= spy[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (spy["close"] > 0).all()


def test_synthetic_is_deterministic():
    _, spy_a, _ = data.synthetic_daily(n_stocks=50, n_days=252, seed=7)
    _, spy_b, _ = data.synthetic_daily(n_stocks=50, n_days=252, seed=7)
    assert np.allclose(spy_a["close"].to_numpy(), spy_b["close"].to_numpy())
    _, spy_c, _ = data.synthetic_daily(n_stocks=50, n_days=252, seed=8)
    assert not np.allclose(spy_a["close"].to_numpy(), spy_c["close"].to_numpy())


def test_ad_line_is_cumsum(null_tape):
    spy, ad, truth = null_tape
    # A/D line embedded in synthetic spy matches build_ad_line on panel
    panel, spy2, _ = data.synthetic_daily(
        n_stocks=truth["n_stocks"], n_days=truth["n_days"],
        divergence=0.0, seed=truth["seed"]
    )
    ad_rebuilt = data.build_ad_line(panel)
    assert np.allclose(ad.to_numpy(), ad_rebuilt.to_numpy())


def test_divergence_creates_ad_lag(null_tape, divergence_tape):
    """With planted divergence, the A/D line grows more slowly than the index."""
    spy_n, ad_n, _ = null_tape
    spy_d, ad_d, truth = divergence_tape

    # Normalise both A/D lines to their starting value for comparison
    ad_n_norm = ad_n - ad_n.iloc[0]
    ad_d_norm = ad_d - ad_d.iloc[0]

    # The divergence tape should have a *lower* cumulative A/D than the null tape
    # (since we dragged down 60% of stocks in divergence windows)
    assert float(ad_d_norm.iloc[-1]) < float(ad_n_norm.iloc[-1])


def test_fetch_spy_cache_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_spy(fetch=False, cache_dir=str(tmp_path))


def test_fetch_panel_requires_opt_in(tmp_path):
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    from quantlab.hf_data import SurvivorshipBiasError
    with pytest.raises(SurvivorshipBiasError):
        data.fetch_panel(fetch=False, cache_dir=str(tmp_path),
                         allow_survivorship_bias=False)


def test_fingerprint_stable_and_sensitive(null_tape):
    spy, _, _ = null_tape
    assert data.fingerprint(spy) == data.fingerprint(spy)
    _, spy2, _ = data.synthetic_daily(n_stocks=50, n_days=100, seed=99)
    assert data.fingerprint(spy) != data.fingerprint(spy2)
