"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test.

Nothing here touches the network, and nothing here requires ``studies/_cache`` to exist:
the one test that reads the real tape is skipped whole when the cache is absent, which is
the state of a fresh CI checkout.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fee_war import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, ta = data.synthetic_panel(seed=959)
    b, tb = data.synthetic_panel(seed=959)
    assert list(a.columns) == list(b.columns)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert ta["fees_bps"] == tb["fees_bps"]


def test_synthetic_seed_changes_the_tape():
    a, _ = data.synthetic_panel(seed=959)
    b, _ = data.synthetic_panel(seed=960)
    assert not np.allclose(a["F00"].to_numpy(), b["F00"].to_numpy())


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_panel(n_years=2.5, seed=959)
    assert {"bench", "cash"}.issubset(prices.columns)
    assert truth["fund_cols"] == [c for c in prices.columns if c.startswith("F")]
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"]
    assert prices.notna().all().all()
    # OOB-safe: the synthetic index must stay well inside pandas' ns Timestamp horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_cash_is_monotone_growing():
    prices, truth = data.synthetic_panel(seed=959)
    assert (prices["cash"].diff().dropna() > 0).all()
    assert prices["cash"].iloc[-1] > prices["cash"].iloc[0]


def test_signal_strength_zero_flattens_the_fee_ladder():
    _, t1 = data.synthetic_panel(signal_strength=1.0, seed=959)
    _, t0 = data.synthetic_panel(signal_strength=0.0, seed=959)
    assert t1["fee_spread_bps"] > 100.0
    assert t0["fee_spread_bps"] == pytest.approx(0.0, abs=1e-9)
    # The PUBLISHED sheet is untouched by the knob — that is the point of the null.
    assert t0["fees_bps"] == t1["fees_bps"]


def test_planted_fee_shows_up_as_a_drag(fee_ladder):
    """The dearest planted wrapper must end below the cheapest one, by roughly its fee."""
    prices, truth = fee_ladder
    funds = truth["fund_cols"]
    fees = np.asarray(truth["fees_eff_bps"])
    cheap, dear = funds[int(fees.argmin())], funds[int(fees.argmax())]
    years = truth["n_years"]
    gap_bpy = (np.log(prices[cheap].iloc[-1] / prices[cheap].iloc[0])
               - np.log(prices[dear].iloc[-1] / prices[dear].iloc[0])) / years * 1e4
    planted = fees.max() - fees.min()
    assert gap_bpy > 0
    assert abs(gap_bpy - planted) < 40.0     # premium noise at both anchors, nothing more


def test_clock_stub_is_large_against_bench_and_absent_between_funds(fee_ladder):
    """The planted measurement problem: the stub dominates fund-vs-bench, cancels fund-vs-fund."""
    prices, truth = fee_ladder
    d = np.log(prices[truth["fund_cols"] + ["bench"]]).diff().dropna()
    vs_bench = float((d["F00"] - d["bench"]).std(ddof=1)) * 1e4
    vs_peer = float((d["F00"] - d["F01"]).std(ddof=1)) * 1e4
    assert vs_bench > 5.0 * vs_peer


def test_synthetic_daily_is_the_two_fund_wrapper():
    prices, truth = data.synthetic_daily(seed=959)
    assert truth["n_funds"] == 2
    assert truth["fees_bps"] == [20.0, 150.0]


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_panel(seed=959)
    b, _ = data.synthetic_panel(seed=960)
    fp = data.fingerprint(a)
    assert fp == data.fingerprint(a) and len(fp) == 12
    assert fp != data.fingerprint(b)


# --------------------------------------------------------------------------- #
# The fee schedule is an ASSUMPTION — test the arithmetic that blends it, not the tape
# --------------------------------------------------------------------------- #
def test_blended_fee_is_between_waiver_and_headline():
    for tk, (wfee, wend) in data.WAIVER.items():
        b = data.blended_fee_bps(tk, "2024-01-11", "2026-06-30")
        lo, hi = min(wfee, data.FEE_BPS[tk]), max(wfee, data.FEE_BPS[tk])
        assert lo - 1e-9 <= b <= hi + 1e-9


def test_blended_fee_collapses_to_headline_after_the_waiver():
    b = data.blended_fee_bps("IBIT", "2025-06-01", "2026-06-30")
    assert b == pytest.approx(data.FEE_BPS["IBIT"])


def test_gbtc_is_the_priciest_and_was_never_waived():
    assert max(data.FEE_BPS[t] for t in data.COHORT) == data.FEE_BPS["GBTC"]
    assert data.WAIVER["GBTC"][0] == data.FEE_BPS["GBTC"]


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_have_real_is_false_on_an_empty_cache(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when _cache/ is absent (CI safe)
# --------------------------------------------------------------------------- #
CACHE = data.DEFAULT_CACHE


@pytest.mark.skipif(not data.have_real(cache_dir=CACHE),
                    reason="no real _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_pair_spread_runs():
    px = data.cohort_frame(data.load_prices())
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    assert len(px) > 400
    hd = st.pair_spread(px, "IBIT", "GBTC")
    for k in ("spread_bpy", "tstat", "ci_low", "ci_high"):
        assert np.isfinite(hd[k])
