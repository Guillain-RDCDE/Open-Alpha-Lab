"""Data-layer tests — synthetic determinism and the hardcoded holdings lists (offline).

Nothing here touches the network or the shared cache except the final smoke test, which
skips itself cleanly when the cache is absent (a fresh CI checkout).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diy_sector import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The hardcoded holdings lists (labelled PROXY inputs)
# --------------------------------------------------------------------------- #
def test_every_fund_has_both_lists():
    for fund, spec in data.SECTORS.items():
        assert len(spec["top_2026"]) == 10, fund
        assert len(spec["top_2011"]) == 10, fund
        assert 0.0 < spec["expense_ratio"] < 0.01, fund


def test_holdings_lists_have_no_duplicates():
    for fund, spec in data.SECTORS.items():
        cap = [t for t, _ in spec["top_2026"]]
        assert len(set(cap)) == len(cap), fund
        assert len(set(spec["top_2011"])) == len(spec["top_2011"]), fund


def test_cap_weights_are_descending_and_positive():
    """The published list is ordered by weight — a silent reorder would change every
    top-3 and top-5 basket in the study."""
    for fund, spec in data.SECTORS.items():
        w = [x for _, x in spec["top_2026"]]
        assert all(a >= b for a, b in zip(w, w[1:])), fund
        assert all(x > 0 for x in w), fund


def test_basket_weights_sum_to_one_and_respect_depth():
    for fund in data.FUNDS:
        for scheme in ("cap2026", "eq2026", "eq2011"):
            for n in (3, 5, 10):
                w = data.basket_weights(fund, n, scheme)
                assert len(w) == n
                assert abs(float(w.sum()) - 1.0) < 1e-12


def test_equal_weight_scheme_is_actually_equal():
    w = data.basket_weights("XLF", 5, "eq2011")
    assert np.allclose(w.to_numpy(), 0.2)


def test_cap_weight_scheme_is_not_equal():
    w = data.basket_weights("XLK", 5, "cap2026")
    assert w.std(ddof=0) > 0.01


def test_eq2026_identifies_the_look_ahead_one_factor_at_a_time():
    """``cap2026`` vs ``eq2011`` changes the names AND the weighting. ``eq2026`` is the
    control that holds the weighting fixed, so the decomposition attributes only the
    membership vintage to hindsight. It must share ``cap2026``'s names and ``eq2011``'s
    equal weighting — otherwise the study's headline premium is confounded."""
    for fund in data.FUNDS:
        cap = data.basket_weights(fund, 10, "cap2026")
        eqn = data.basket_weights(fund, 10, "eq2026")
        old = data.basket_weights(fund, 10, "eq2011")
        assert list(eqn.index) == list(cap.index)          # same names as the hindsight list
        assert np.allclose(eqn.to_numpy(), 0.1)            # same weighting as the control
        assert cap.std(ddof=0) > eqn.std(ddof=0)           # ... and genuinely different weights
        # The two lists must really differ, or there would be nothing to identify.
        assert set(old.index) != set(cap.index), fund


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        data.basket_weights("XLK", 3, "whatever")


def test_all_tickers_covers_funds_cash_and_names():
    tk = set(data.TICKERS)
    assert set(data.FUNDS).issubset(tk)
    assert data.CASH in tk
    for fund, spec in data.SECTORS.items():
        assert {t for t, _ in spec["top_2026"]}.issubset(tk)
        assert set(spec["top_2011"]).issubset(tk)


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(n_years=3, seed=962)
    b, _ = data.synthetic_panel(n_years=3, seed=962)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_panel(n_years=5, n_names=7, seed=962)
    assert "fund" in prices.columns and "cash" in prices.columns
    assert len([c for c in prices.columns if c.startswith("name_")]) == 7
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 5 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay inside pandas' ns Timestamp horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_cash_is_monotone_growing():
    prices, _ = data.synthetic_panel(n_years=3, seed=962)
    assert (prices["cash"].diff().dropna() > 0).all()


def test_synthetic_weights_sum_to_one():
    _, truth = data.synthetic_panel(n_years=3, seed=962)
    assert abs(float(truth["weights"].sum()) - 1.0) < 1e-12


def test_signal_strength_scales_the_planted_fee():
    _, t1 = data.synthetic_panel(n_years=3, signal_strength=1.0, seed=962)
    _, t_half = data.synthetic_panel(n_years=3, signal_strength=0.5, seed=962)
    _, t0 = data.synthetic_panel(n_years=3, signal_strength=0.0, seed=962)
    assert t0["planted_fee_ann"] == 0.0
    assert abs(t_half["planted_fee_ann"] - t1["planted_fee_ann"] / 2.0) < 1e-15
    assert t1["planted_fee_ann"] == t1["fee_ann"]


def test_fee_only_moves_the_fund_leg():
    """Switching the fee on must change the fund and nothing else — otherwise the
    'planted effect' would be leaking into the constituents."""
    p1, _ = data.synthetic_panel(n_years=3, signal_strength=1.0, seed=962)
    p0, _ = data.synthetic_panel(n_years=3, signal_strength=0.0, seed=962)
    names = [c for c in p1.columns if c.startswith("name_")]
    assert np.allclose(p1[names].to_numpy(), p0[names].to_numpy())
    assert p1["fund"].iloc[-1] < p0["fund"].iloc[-1]


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_panel(n_years=3, seed=962)
    b, _ = data.synthetic_panel(n_years=3, seed=963)
    assert data.fingerprint(a) == data.fingerprint(a)
    assert len(data.fingerprint(a)) == 12
    assert data.fingerprint(a) != data.fingerprint(b)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_synthetic_daily_is_an_alias():
    a, _ = data.synthetic_daily(n_years=2, seed=962)
    b, _ = data.synthetic_panel(n_years=2, seed=962)
    assert np.allclose(a.to_numpy(), b.to_numpy())


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no shared _cache present (fresh checkout / CI) — the "
                           "synthetic tests cover the logic")
def test_real_cache_race_runs():
    px = data.load_prices()
    assert px.index[0] >= pd.Timestamp(data.SAMPLE_START)
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    w = data.basket_weights("XLF", 10, "eq2011")
    res = st.race(px, "XLF", w, px[data.CASH].dropna(), cost_bps=5.0)
    for k in ("ann_gap", "t_diff", "tracking_error", "corr"):
        assert np.isfinite(res[k])
    # A ten-name basket of a sector's biggest banks tracks its own sector fund closely.
    assert res["corr"] > 0.85


@pytest.mark.skipif(not data.have_real(),
                    reason="no shared _cache present (fresh checkout / CI) — the "
                           "synthetic tests cover the logic")
def test_real_cache_decomposition_is_additive():
    """The three baskets must line up so that (vintage) + (weighting) reproduces the raw
    hindsight-minus-contemporaneous difference the README quotes."""
    px = data.load_prices()
    cash = px[data.CASH].dropna()
    gaps = {}
    for scheme in ("eq2011", "eq2026", "cap2026"):
        d = pd.concat(
            {f: st.race(px, f, data.basket_weights(f, 10, scheme), cash,
                        cost_bps=5.0)["diff"] for f in data.FUNDS},
            axis=1,
        ).mean(axis=1).dropna()
        gaps[scheme] = (1.0 + d.mean()) ** st.TRADING_DAYS - 1.0
    vintage = gaps["eq2026"] - gaps["eq2011"]
    weighting = gaps["cap2026"] - gaps["eq2026"]
    raw = gaps["cap2026"] - gaps["eq2011"]
    assert abs((vintage + weighting) - raw) < 1e-12
    # The look-ahead is the larger term, but it is NOT the whole of the raw difference.
    assert vintage > 0.0
    assert abs(weighting) > 0.002
