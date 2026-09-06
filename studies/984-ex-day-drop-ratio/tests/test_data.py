"""Data-layer tests for Study 984 — synthetic determinism offline, cache-gated on tape."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exday import data  # noqa: E402


def test_synthetic_panel_is_deterministic():
    a = data.synthetic_panel(n=600, n_tickers=3, seed=984)
    b = data.synthetic_panel(n=600, n_tickers=3, seed=984)
    assert a.keys() == b.keys()
    for k in a:
        assert np.allclose(a[k].to_numpy(), b[k].to_numpy())


def test_synthetic_panel_seed_sensitive():
    a = data.synthetic_panel(n=600, n_tickers=3, seed=984)
    b = data.synthetic_panel(n=600, n_tickers=3, seed=985)
    assert not np.allclose(a["SIM0"]["close"].to_numpy(), b["SIM0"]["close"].to_numpy())


def test_synthetic_panel_shape_and_index():
    bars = data.synthetic_panel(n=800, n_tickers=5)
    assert set(bars) == {f"SIM{k}" for k in range(5)} | {"MKT"}
    for k, b in bars.items():
        assert len(b) == 800
        assert list(b.columns) == ["close", "dividend"]
        assert isinstance(b.index, pd.DatetimeIndex)
        assert (b["close"] > 0).all()
        assert (b["dividend"] >= 0).all()
        assert b.index[-1] < pd.Timestamp("2262-01-01")


def test_the_market_leg_pays_no_dividend():
    bars = data.synthetic_panel(n=500, n_tickers=2)
    assert (bars["MKT"]["dividend"] == 0).all()


def test_every_payer_pays_roughly_quarterly():
    bars = data.synthetic_panel(n=2520, n_tickers=4)
    for k in ("SIM0", "SIM3"):
        d = bars[k]["dividend"]
        ex = d[d > 0]
        assert 30 <= len(ex) <= 50            # ten years, four payments a year
        assert (ex.index.to_series().diff().dropna().dt.days.median()) > 60


def test_drop_fraction_is_the_knob_that_matters():
    """The generator's contract: the price gives up exactly ``drop_fraction`` of the cash."""
    for frac in (0.0, 0.5, 1.0):
        bars = data.synthetic_panel(n=1200, n_tickers=1, drop_fraction=frac,
                                    daily_vol=1e-9, market_beta=0.0)
        b = bars["SIM0"]
        d = b["dividend"]
        ex = d[d > 0].index
        pos = {t: i for i, t in enumerate(b.index)}
        ratios = [(b["close"].iloc[pos[t] - 1] - b["close"].iloc[pos[t]]) / d.loc[t]
                  for t in ex if pos[t] > 0]
        assert np.mean(ratios) == pytest.approx(frac, abs=0.01)


def test_quarterly_yield_controls_the_dividend_size():
    small = data.synthetic_panel(n=1000, n_tickers=1, quarterly_yield=0.002)["SIM0"]
    large = data.synthetic_panel(n=1000, n_tickers=1, quarterly_yield=0.02)["SIM0"]
    ratio = (large["dividend"].sum() / large["close"].mean()) / \
            (small["dividend"].sum() / small["close"].mean())
    assert 5 < ratio < 20


def test_daily_vol_is_the_noise_knob():
    quiet = data.synthetic_panel(n=1500, n_tickers=1, daily_vol=0.004)["SIM0"]
    loud = data.synthetic_panel(n=1500, n_tickers=1, daily_vol=0.02)["SIM0"]
    assert loud["close"].pct_change().std() > quiet["close"].pct_change().std() * 2


def test_fingerprint_stable_and_sensitive():
    a = data.synthetic_panel(n=400, n_tickers=3, seed=984)["SIM0"]
    b = data.synthetic_panel(n=400, n_tickers=3, seed=985)["SIM0"]
    fp = data.fingerprint(a)
    assert fp == data.fingerprint(a) and len(fp) == 12
    assert fp != data.fingerprint(b)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_load_bars_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_bars(cache_dir=str(tmp_path))


def test_have_real_is_false_on_empty_dir(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


def test_the_cache_prefix_is_not_the_desks_shared_one(tmp_path):
    """These bars are deliberately dividend-UNadjusted; they must not shadow ``prices_``."""
    p = data._cache_path("KO", str(tmp_path))
    assert os.path.basename(p).startswith("exday_")
    assert "prices_" not in p


def test_universe_is_declared():
    assert len(data.TICKERS) == len(set(data.TICKERS)) >= 2
    assert data.MARKET in data.TICKERS
    assert data.MARKET not in data.PAYERS
    assert pd.Timestamp(data.AS_OF) > pd.Timestamp(data.START)


@pytest.mark.skipif(not data.have_real(),
                    reason="no shared _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_loads_and_is_pinned():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    assert px.index.is_monotonic_increasing and not px.index.has_duplicates
    assert (px.dropna(how="all") > 0).any().all()


@pytest.mark.skipif(not data.have_real(),
                    reason="no shared _cache present (offline / CI)")
def test_the_real_tape_is_NOT_dividend_adjusted():
    """The load-bearing property of this data layer, checked against the tape itself.

    A dividend-adjusted close would show no systematic drop on ex-days. This asserts the
    opposite: pooled across every payer, the average ex-day return is materially more negative
    than the average non-ex-day return.
    """
    bars = data.load_bars()
    ex_rets, other_rets = [], []
    for tk in data.PAYERS:
        b = bars[tk]
        r = b["close"].pct_change()
        is_ex = b["dividend"] > 0
        ex_rets.append(r[is_ex].dropna())
        other_rets.append(r[~is_ex].dropna())
    ex = pd.concat(ex_rets)
    other = pd.concat(other_rets)
    assert len(ex) > 100
    assert ex.mean() < other.mean()


@pytest.mark.skipif(not data.have_real(),
                    reason="no shared _cache present (offline / CI)")
def test_the_dividend_feed_passes_its_own_audit():
    sr = data.sanity_report()
    assert (sr["ex_days"] > 20).all()
    assert (sr.loc[list(data.PAYERS), "median_yield"] > 0.0005).all()
    assert (sr.loc[list(data.PAYERS), "median_yield"] < 0.05).all()
