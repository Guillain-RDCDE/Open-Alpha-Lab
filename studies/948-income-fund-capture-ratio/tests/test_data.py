"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capture_ratio import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic panel — determinism, shape, and the planted truth
# --------------------------------------------------------------------------- #
def test_panel_is_deterministic():
    a, _ = data.synthetic_panel(seed=948)
    b, _ = data.synthetic_panel(seed=948)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c, _ = data.synthetic_panel(seed=949)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_panel_shape_and_columns(planted_panel):
    rets, truth = planted_panel
    assert {"BENCH", "CASH"}.issubset(rets.columns)
    assert set(truth["fund_names"]).issubset(rets.columns)
    assert isinstance(rets.index, pd.DatetimeIndex)
    assert len(rets) == truth["n_months"] == 240
    assert not rets.isna().any().any()


def test_panel_index_is_month_end_and_oob_safe(planted_panel):
    rets, _ = planted_panel
    # period_range -> to_timestamp keeps us far inside pandas' ns Timestamp horizon.
    assert rets.index[-1] < pd.Timestamp("2200-01-01")
    assert (rets.index == rets.index.to_period("M").to_timestamp(how="end").normalize()).all()


def test_signal_strength_zero_kills_every_planted_spread():
    _, t1 = data.synthetic_panel(signal_strength=1.0, seed=948)
    _, t0 = data.synthetic_panel(signal_strength=0.0, seed=948)
    assert all(abs(v["true_spread"]) < 1e-12 for v in t0["funds"].values())
    assert max(abs(v["true_spread"]) for v in t1["funds"].values()) > 0.1


def test_planted_truth_matches_its_own_definition(planted_panel):
    _, truth = planted_panel
    for v in truth["funds"].values():
        assert abs((v["b_up"] - v["b_dn"]) - v["true_spread"]) < 1e-12
        assert abs(0.5 * (v["b_up"] + v["b_dn"]) - v["beta"]) < 1e-12


def test_cash_leg_is_flat_and_positive(planted_panel):
    rets, truth = planted_panel
    assert (rets["CASH"] > 0).all()
    assert rets["CASH"].std() < 1e-12
    grew = float((1.0 + rets["CASH"]).prod())
    assert grew > 1.0


# --------------------------------------------------------------------------- #
# Synthetic daily tape + the daily -> monthly path
# --------------------------------------------------------------------------- #
def test_daily_tape_shape(daily_tape):
    px, truth = daily_tape
    assert {"FUND", "BENCH", "CASH"}.issubset(px.columns)
    assert len(px) == truth["n_days"]
    assert (px > 0).all().all()
    assert px.index[-1] < pd.Timestamp("2200-01-01")


def test_to_monthly_returns_drops_the_partial_month(daily_tape):
    px, _ = daily_tape
    asof = str((px.index[-1] - pd.offsets.MonthEnd(1)).date())
    m = data.to_monthly_returns(px, asof=asof)
    assert m.index[-1] <= pd.Timestamp(asof)
    # every surviving label is a month end
    assert (m.index == m.index.to_period("M").to_timestamp(how="end").normalize()).all()
    assert m.iloc[0].isna().all()          # first month has no prior level to diff against


def test_monthly_returns_recover_the_daily_compounding(daily_tape):
    px, _ = daily_tape
    levels = px["BENCH"].resample("ME").last()
    m = data.to_monthly_returns(px, asof=str(px.index[-1].date())).dropna()
    total_monthly = float((1.0 + m["BENCH"]).prod())
    # Compounding the surviving monthly returns must reproduce the level ratio between
    # the first and the last *complete* month end (the partial trailing month is dropped).
    assert abs(total_monthly - levels.loc[m.index[-1]] / levels.iloc[0]) < 1e-8


def test_daily_tape_carries_the_planted_kink(daily_tape):
    px, truth = daily_tape
    m = data.to_monthly_returns(px, asof=str(px.index[-1].date())).dropna()
    cap = st.capture_ratios(m["FUND"], m["BENCH"])
    assert cap["spread"] > 0.05          # the planted convexity is visible after resampling
    assert truth["true_spread"] > 0


# --------------------------------------------------------------------------- #
# Fingerprint & the offline cache contract
# --------------------------------------------------------------------------- #
def test_fingerprint_stable_and_sensitive(planted_panel):
    rets, _ = planted_panel
    fp1, fp2 = data.fingerprint(rets), data.fingerprint(rets)
    other, _ = data.synthetic_panel(seed=949)
    assert fp1 == fp2 and len(fp1) == 12
    assert fp1 != data.fingerprint(other)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


# --------------------------------------------------------------------------- #
# Unadjusted-corporate-action screen (the NUSI 2025-02-18 defect, synthetically)
# --------------------------------------------------------------------------- #
def test_clean_tape_has_no_split_artifacts(daily_tape):
    px, _ = daily_tape
    assert data.split_artifacts(px) == []
    fixed, log = data.repair_split_artifacts(px)
    assert log == []
    assert np.allclose(fixed.to_numpy(), px.to_numpy())


@pytest.mark.parametrize("factor", [2.0, 0.5, 3.0, 1.0 / 3.0])
def test_planted_share_count_event_is_detected_and_undone(daily_tape, factor):
    """Plant a share-count event mid-tape; the screen must restore the original returns."""
    px, _ = daily_tape
    cut = px.index[len(px) // 2]
    dirty = px.copy()
    # Prices from ``cut`` on are re-based by ``factor``: exactly what an unadjusted split
    # looks like in a vendor's "adjusted" tape.
    dirty.loc[dirty.index >= cut, "FUND"] = dirty.loc[dirty.index >= cut, "FUND"] * factor

    log = data.split_artifacts(dirty)
    assert len(log) == 1
    assert log[0]["ticker"] == "FUND" and log[0]["date"] == cut
    assert log[0]["repairable"] and abs(log[0]["factor"] - factor) < 1e-9

    fixed, _ = data.repair_split_artifacts(dirty)
    r_fixed = fixed["FUND"].pct_change().dropna()
    r_true = px["FUND"].pct_change().dropna()
    assert np.allclose(r_fixed.to_numpy(), r_true.to_numpy(), atol=1e-12)


def test_genuine_crash_is_not_smoothed_away(daily_tape):
    """A −41.7% one-day move is a return, not a share count: leave it alone."""
    px, _ = daily_tape
    cut = px.index[len(px) // 2]
    dirty = px.copy()
    dirty.loc[dirty.index >= cut, "FUND"] = dirty.loc[dirty.index >= cut, "FUND"] * 0.583
    log = data.split_artifacts(dirty)
    assert len(log) == 1 and not log[0]["repairable"]
    fixed, _ = data.repair_split_artifacts(dirty)
    assert np.allclose(fixed.to_numpy(), dirty.to_numpy())


def test_split_factor_rejects_ordinary_moves():
    for ordinary in (1.07, 0.62, 0.583, 1.42, 2.2, 0.45):
        assert not np.isfinite(data._split_factor(ordinary)), ordinary
    assert data._split_factor(2.0) == 2.0
    assert abs(data._split_factor(0.3335) - 1.0 / 3.0) < 1e-9
    assert abs(data._split_factor(1.4995) - 1.5) < 1e-12
    # a split day carrying its own genuine 2% move is still recognised
    assert data._split_factor(2.0 * 1.02) == 2.0


def test_split_repair_does_not_touch_the_returns_before_or_after(daily_tape):
    """Only the event day's return changes; every other day is bit-for-bit identical."""
    px, _ = daily_tape
    cut = px.index[2 * len(px) // 3]
    dirty = px.copy()
    dirty.loc[dirty.index >= cut, "BENCH"] = dirty.loc[dirty.index >= cut, "BENCH"] * 2.0
    fixed, _ = data.repair_split_artifacts(dirty)
    r_dirty = dirty["BENCH"].pct_change()
    r_fixed = fixed["BENCH"].pct_change()
    off = r_dirty.index != cut
    assert np.allclose(r_dirty[off].dropna().to_numpy(), r_fixed[off].dropna().to_numpy())
    assert r_dirty.loc[cut] > 0.9 and abs(r_fixed.loc[cut]) < 0.1


def test_bench_map_is_complete_and_uses_known_benchmarks():
    assert set(data.FUND_BENCH) == set(data.FUNDS)
    assert set(data.FUND_BENCH.values()) <= set(data.BENCHES)
    assert set(data.OPTION_WRITERS) <= set(data.FUNDS)


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no shared _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_scorecard_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    rets = data.to_monthly_returns(px)
    sc = st.scorecard(rets, {"QYLD": "QQQ", "SCHD": "SPY"}, n_boot=200)
    assert len(sc) == 2
    for col in ("up", "down", "spread", "convexity", "t_convexity", "beta"):
        assert np.isfinite(sc[col]).all()
    # every income fund is a lower-beta version of its index
    assert (sc["beta"] < 1.0).all()
