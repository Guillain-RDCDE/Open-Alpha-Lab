"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lending_offset import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(seed=914)
    b, _ = data.synthetic_daily(seed=914)
    for col in ("fund_a", "fund_b", "cash"):
        assert np.allclose(a[col].to_numpy(), b[col].to_numpy())


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_daily(n_years=15, seed=914)
    assert {"fund_a", "fund_b", "cash"}.issubset(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 15 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay inside pandas' ns Timestamp horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_seed_sensitivity():
    a, _ = data.synthetic_daily(seed=914)
    b, _ = data.synthetic_daily(seed=915)
    assert not np.allclose(a["fund_a"].to_numpy(), b["fund_a"].to_numpy())


def test_synthetic_cash_is_monotone_growing():
    prices, _ = data.synthetic_daily(seed=914)
    assert (prices["cash"].diff().dropna() > 0).all()
    assert prices["cash"].iloc[-1] > prices["cash"].iloc[0]


def test_signal_strength_scales_the_planted_effect():
    _, t1 = data.synthetic_daily(signal_strength=1.0, seed=914)
    _, t_half = data.synthetic_daily(signal_strength=0.5, seed=914)
    _, t0 = data.synthetic_daily(signal_strength=0.0, seed=914)
    assert t1["planted_residual_bp"] == pytest.approx(60.0)
    assert t_half["planted_residual_bp"] == pytest.approx(30.0)
    assert t0["planted_residual_bp"] == pytest.approx(0.0)


def test_both_funds_share_the_index_path():
    """The two legs must be near-perfectly correlated — they track the same basket."""
    prices, _ = data.synthetic_daily(seed=914)
    lr = np.log(prices[["fund_a", "fund_b"]]).diff().dropna()
    assert lr.corr().iloc[0, 1] > 0.99


def test_synthetic_panel_shape():
    frames, truth = data.synthetic_panel(n_pairs=4, n_years=10, seed=914)
    assert len(frames) == 4 == truth["n_pairs"]
    for px in frames.values():
        assert {"fund_a", "fund_b", "cash"}.issubset(px.columns)
        assert len(px) == 10 * data.TRADING_DAYS_PER_YEAR


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_daily(seed=914)
    fp1 = data.fingerprint(a)
    assert fp1 == data.fingerprint(a) and len(fp1) == 12
    b, _ = data.synthetic_daily(seed=915)
    assert fp1 != data.fingerprint(b)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_er_history_band_covers_every_pair_and_is_monotone():
    """(launch, mid, today) must be ordered and defined for every ticker in a pair.

    The band encodes the study's binding assumption: fees were CUT, never raised, so
    launch >= mid >= today for every fund. If a future edit breaks that ordering, the
    fee-history scenarios stop bracketing the headline and the sign claim goes unguarded.
    """
    for tk in data.TICKERS:
        assert tk in data.ER_HISTORY_BAND, tk
        launch, mid, today = data.ER_HISTORY_BAND[tk]
        assert launch >= mid >= today > 0.0, tk
        assert data.EXPENSE_RATIOS_EARLY[tk] == launch
        assert data.EXPENSE_RATIOS_TIMEAVG[tk] == mid
        # "today" in the band must be the headline assumption actually used.
        assert data.EXPENSE_RATIOS[tk] == pytest.approx(today)


def test_todays_fees_bias_the_residual_upward_on_the_dear_pairs():
    """The one-sidedness the study corrects for, asserted rather than asserted-in-prose.

    Using today's ERs instead of the early ones shifts the residual by ``cut_b - cut_a``.
    On the four pairs whose legacy leg ``a`` is the dearer one, the modern leg ``b`` was
    cut harder, so today's fees push the residual UP — the direction that makes a lending
    offset look absent. VEA/IEFA is the documented exception (its legacy leg is the cheap
    one) and is checked to be the *only* one, so the caveat is neither over- nor
    under-stated.
    """
    exceptions = []
    for a, b, _klass, _note in data.PAIRS:
        cut_a = data.ER_HISTORY_BAND[a][0] - data.ER_HISTORY_BAND[a][2]
        cut_b = data.ER_HISTORY_BAND[b][0] - data.ER_HISTORY_BAND[b][2]
        if data.EXPENSE_RATIOS[a] > data.EXPENSE_RATIOS[b]:
            assert cut_b > cut_a, (a, b, cut_a, cut_b)   # upward bias on today's fees
        else:
            exceptions.append((a, b))
    assert exceptions == [("VEA", "IEFA")]


def test_pairs_and_expense_ratios_are_consistent():
    """Every ticker named in a pair must carry a (PROXY) expense ratio."""
    for a, b, klass, note in data.PAIRS:
        assert a in data.EXPENSE_RATIOS and b in data.EXPENSE_RATIOS
        assert a in data.TICKERS and b in data.TICKERS
        assert isinstance(klass, str) and len(note) > 20


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
CACHE = data.DEFAULT_CACHE


@pytest.mark.skipif(not data.have_real(cache_dir=CACHE),
                    reason="no real _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_residual_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    tbl = st.residual_table(px, data.PAIRS, data.EXPENSE_RATIOS, freq="M", n_boot=200)
    assert len(tbl) == len(data.PAIRS)
    for col in ("residual_bp", "t_hac", "te_pct", "noise_floor_bp"):
        assert np.isfinite(tbl[col]).all()
    # Same-class trackers must be near-identical exposures.
    assert (tbl["beta"].between(0.9, 1.1)).all()
