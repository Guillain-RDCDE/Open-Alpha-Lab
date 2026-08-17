"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trust_yield import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The hardcoded list itself
# --------------------------------------------------------------------------- #
def test_spac_list_is_well_formed():
    assert len(data.SPACS) >= 25
    tickers = [t for t, _, _ in data.SPACS]
    assert len(set(tickers)) == len(tickers)
    for tk, sym, close in data.SPACS:
        assert tk.isupper() and sym.isupper()
        d = pd.Timestamp(close)
        assert pd.Timestamp("2020-01-01") <= d <= pd.Timestamp(data.AS_OF)


def test_deal_close_dates_span_both_rate_regimes():
    years = sorted({pd.Timestamp(c).year for _, _, c in data.SPACS})
    assert 2021 in years and 2022 in years and max(years) >= 2023


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, ca, ma, _ = data.synthetic_panel(seed=932)
    b, cb, mb, _ = data.synthetic_panel(seed=932)
    assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)
    assert np.allclose(ca.to_numpy(), cb.to_numpy())
    assert np.allclose(ma["payoff"].to_numpy(dtype=float), mb["payoff"].to_numpy(dtype=float))


def test_synthetic_shape_and_columns(put_binds):
    prices, cash, meta, truth = put_binds
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices.columns) == truth["n_spacs"] == meta.shape[0]
    assert {"start", "deadline", "payoff"}.issubset(meta.columns)
    assert len(prices) == len(cash) == truth["n_days"]
    # OOB-safe: the synthetic index must stay inside pandas' ns horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_quotes_live_near_ten(put_binds):
    prices, _, _, _ = put_binds
    v = prices.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    assert 8.0 < np.median(v) < 11.0
    assert v.min() > 5.0


def test_synthetic_cash_is_monotone_growing(put_binds):
    _, cash, _, _ = put_binds
    assert (cash.diff().dropna() > 0).all()
    assert cash.iloc[-1] > cash.iloc[0]


def test_null_payoff_is_the_last_quote_not_the_trust(put_is_fiction):
    prices, _, meta, _ = put_is_fiction
    for tk in meta.index:
        s = prices[tk].dropna()
        assert np.isclose(float(s.iloc[-1]), float(meta.loc[tk, "payoff"]))


def test_planted_payoff_is_the_accrued_trust(put_binds):
    prices, cash, meta, truth = put_binds
    for tk in meta.index[:5]:
        t0, t1 = meta.loc[tk, "start"], meta.loc[tk, "deadline"]
        expect = truth["trust0"] * float(cash.loc[t1] / cash.loc[t0])
        assert abs(float(meta.loc[tk, "payoff"]) - expect) < 1e-6


def test_synthetic_daily_is_single_name():
    prices, _, meta, _ = data.synthetic_daily(seed=932)
    assert prices.shape[1] == 1 and len(meta) == 1


# --------------------------------------------------------------------------- #
# Trust path, fingerprint, cleaning, cache guards
# --------------------------------------------------------------------------- #
def test_trust_path_accrues_at_the_cash_leg(put_binds):
    _, cash, _, _ = put_binds
    tp = st.trust_path(cash, cash.index[0], cash.index[-1], trust0=10.0)
    assert abs(float(tp.iloc[0]) - 10.0) < 1e-9
    assert float(tp.iloc[-1]) > float(tp.iloc[0])
    ratio = float(tp.iloc[-1] / tp.iloc[0]) / float(cash.iloc[-1] / cash.iloc[0])
    assert abs(ratio - 1.0) < 1e-9


def test_trust_fee_drag_reduces_the_path(put_binds):
    _, cash, _, _ = put_binds
    a = st.trust_path(cash, cash.index[0], cash.index[-1], fee_bps=0.0)
    b = st.trust_path(cash, cash.index[0], cash.index[-1], fee_bps=25.0)
    assert float(b.iloc[-1]) < float(a.iloc[-1])


def test_clean_quotes_only_touches_the_pre_deal_window():
    idx = pd.bdate_range("2021-01-04", periods=400)
    df = pd.DataFrame({"AAA": np.full(400, 9.9)}, index=idx)
    spacs = (("AAA", "AAA", "2022-01-04"),)
    df.iloc[50, 0] = 0.5      # a bad print inside the pre-deal window
    df.iloc[380, 0] = 0.5     # a real post-deal collapse
    out, counts = data.clean_quotes(df, spacs=spacs, buffer_days=30)
    assert counts == {"AAA": 1}
    assert abs(float(out.iloc[50, 0]) - 9.9) < 1e-9
    assert abs(float(out.iloc[380, 0]) - 0.5) < 1e-9


def test_fingerprint_stable_and_sensitive(put_binds):
    prices, _, _, _ = put_binds
    other, _, _, _ = data.synthetic_panel(seed=933)
    fp = data.fingerprint(prices)
    assert fp == data.fingerprint(prices) and len(fp) == 12
    assert fp != data.fingerprint(other)


def test_loaders_raise_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_spac_closes(cache_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        data.load_cash(cache_dir=str(tmp_path))


def test_have_real_is_false_on_an_empty_cache(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no shared _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_windows_are_sane():
    raw = data.load_spac_closes()
    assert raw.index[-1] <= pd.Timestamp(data.AS_OF)
    px, _counts = data.clean_quotes(raw)
    win = st.spac_windows(px)
    assert len(win) >= 25
    # After the redemption buffer, every pre-deal window must end at a plausible quote
    # for a share redeemable at ~$10 — no sub-trust wreckage, which is the whole point
    # of exiting before the vote rather than at the close.
    assert (win["px_last"] > 9.0).all()
    assert (win["deadline"] < win["deal_close"]).all()
