"""Data-layer tests — synthetic determinism (offline) + skipif real-cache smoke tests.

Every test here runs on a fresh checkout with NO ``studies/_cache`` present: the
generators are the only tape. The two real-cache tests skip cleanly when the cache is
absent, which is the CI case.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from withholding import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_panel_is_deterministic():
    tr1, px1, _ = data.synthetic_panel(seed=916)
    tr2, px2, _ = data.synthetic_panel(seed=916)
    assert np.allclose(tr1.to_numpy(), tr2.to_numpy())
    assert np.allclose(px1.to_numpy(), px2.to_numpy())


def test_synthetic_panel_seed_changes_the_tape():
    tr1, _, _ = data.synthetic_panel(seed=916)
    tr2, _, _ = data.synthetic_panel(seed=917)
    assert not np.allclose(tr1.to_numpy(), tr2.to_numpy())


def test_synthetic_panel_shape_and_columns(planted):
    tr, px, truth = planted
    assert list(tr.columns) == list(px.columns)
    assert {"broad", "bench_a", "bench_b", "bench_c"}.issubset(tr.columns)
    assert isinstance(tr.index, pd.DatetimeIndex)
    assert len(tr) == len(px) == truth["n_days"]
    # OOB-safe: the synthetic index must stay well inside pandas' ns Timestamp horizon.
    assert tr.index[-1] < pd.Timestamp("2262-01-01")


def test_total_return_leg_dominates_price_only_leg(planted):
    """A dividend-paying fund's total-return index must outgrow its price-only index."""
    tr, px, _ = planted
    for c in tr.columns:
        assert tr[c].iloc[-1] / tr[c].iloc[0] > px[c].iloc[-1] / px[c].iloc[0]


def test_synthetic_daily_view_matches_panel():
    one, truth = data.synthetic_daily(seed=916)
    tr, px, _ = data.synthetic_panel(seed=916)
    assert {"tr", "px"}.issubset(one.columns)
    assert np.allclose(one["tr"].to_numpy(), tr["broad"].to_numpy())
    assert np.allclose(one["px"].to_numpy(), px["broad"].to_numpy())
    assert truth["seed"] == 916


def test_signal_strength_scales_the_planted_gap():
    _, _, t1 = data.synthetic_panel(signal_strength=1.0, seed=916)
    _, _, t5 = data.synthetic_panel(signal_strength=0.5, seed=916)
    _, _, t0 = data.synthetic_panel(signal_strength=0.0, seed=916)
    assert t0["true_gap_bp"] == 0.0
    assert t1["true_gap_bp"] > t5["true_gap_bp"] > 0.0
    assert abs(t1["true_gap_bp"] - 2 * t5["true_gap_bp"]) < 1e-6


def test_prices_are_positive_and_finite(planted):
    tr, px, _ = planted
    assert np.isfinite(tr.to_numpy()).all() and (tr.to_numpy() > 0).all()
    assert np.isfinite(px.to_numpy()).all() and (px.to_numpy() > 0).all()


def test_fingerprint_stable_and_sensitive(planted):
    tr, _, _ = planted
    fp1 = data.fingerprint(tr)
    assert fp1 == data.fingerprint(tr) and len(fp1) == 12
    tr2, _, _ = data.synthetic_panel(seed=917)
    assert fp1 != data.fingerprint(tr2)


# --------------------------------------------------------------------------- #
# The labelled non-tape assumptions must be present and sane
# --------------------------------------------------------------------------- #
def test_assumption_tables_are_complete():
    for tk in data.TICKERS:
        assert tk in data.EXPENSE_RATIO
        assert 0.0 <= data.EXPENSE_RATIO[tk] < 0.02
    assert abs(sum(data.BLEND_WEIGHTS.values()) - 1.0) < 1e-9
    assert set(data.BLEND_WEIGHTS) == set(data.SINGLE)
    assert 0.0 < data.ASSUMED_EFFECTIVE_WITHHOLDING < 0.35


def test_as_of_is_a_complete_month_end():
    ts = pd.Timestamp(data.AS_OF)
    assert ts == ts + pd.offsets.MonthEnd(0)


# --------------------------------------------------------------------------- #
# Cache contract (offline)
# --------------------------------------------------------------------------- #
def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_load_distributions_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_distributions(cache_dir=str(tmp_path))


def test_have_real_is_false_on_an_empty_dir(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


# --------------------------------------------------------------------------- #
# Real-cache smoke tests — skipped entirely when the shared cache is absent
# --------------------------------------------------------------------------- #
CACHE = data.DEFAULT_CACHE


@pytest.mark.skipif(not data.have_real(cache_dir=CACHE),
                    reason="no shared _cache present (fresh checkout / CI) — synthetic tests cover the logic")
def test_real_cache_respects_asof():
    tr = data.load_prices()
    assert tr.index[-1] <= pd.Timestamp(data.AS_OF)
    dist = data.load_distributions()
    for tk, df in dist.items():
        assert {"close", "dividend", "capital_gains"}.issubset(df.columns)
        assert df.index[-1] <= pd.Timestamp(data.AS_OF)


@pytest.mark.skipif(not data.have_real(cache_dir=CACHE),
                    reason="no shared _cache present (fresh checkout / CI) — synthetic tests cover the logic")
def test_real_ruler_matches_declared_cash_dividends():
    """The differenced yield must reproduce the funds' actual distributions to ~1 bp."""
    tr = data.load_prices()
    dist = data.load_distributions()
    for tk in data.BROAD:
        px = dist[tk]["close"]
        cc = st.distribution_cross_check(tr[tk].dropna(), px.dropna(), dist[tk]["dividend"])
        assert abs(cc["residual_bp"]) < 2.0
        assert cc["n_ex_dates"] > 10
