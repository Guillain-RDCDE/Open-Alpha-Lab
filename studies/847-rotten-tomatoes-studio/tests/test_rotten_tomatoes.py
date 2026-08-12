"""Fully-offline, deterministic tests for Study 847 — Rotten-Tomatoes -> Studio.

No network: everything runs on the seeded synthetic world and on pure functions. The one
real-tape test is skipped when the git-ignored ``_cache/`` is absent (offline CI).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from rotten_tomatoes import data as dt, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The curated table
# --------------------------------------------------------------------------- #
def test_film_table_shape_and_tiers():
    films = dt.film_table()
    assert len(films) == 40
    assert set(films["tier"]) == {"fresh", "rotten"}
    # only clearly-fresh (>=75) or clearly-rotten (<50) — no mixed 50-74 titles
    for r in films.itertuples(index=False):
        if r.tier == "fresh":
            assert r.rt >= dt.FRESH_MIN
        else:
            assert r.rt < dt.ROTTEN_MAX
    # every studio ticker is one of the six named distributors
    assert set(films["studio"]) <= set(dt.STUDIOS)
    # dated within the window where every ticker (incl. WBD from 2022-04-11) traded
    assert films["date"].min() >= pd.Timestamp("2022-04-01")
    assert films["date"].max() <= pd.Timestamp(dt.AS_OF)


def test_table_has_both_tiers_per_analysis():
    films = dt.film_table()
    assert (films["tier"] == "fresh").sum() >= 15
    assert (films["tier"] == "rotten").sum() >= 15


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_one_sample_t_matches_hand_calc():
    x = np.array([-0.01, -0.02, 0.03, 0.00, -0.015])
    r = st.one_sample_t(x)
    assert r["n"] == 5
    assert abs(r["mean"] - x.mean()) < 1e-12
    se = x.std(ddof=1) / np.sqrt(5)
    assert abs(r["t"] - x.mean() / se) < 1e-9


def test_welch_t_sign_and_zero():
    a = np.array([0.02, 0.03, 0.025, 0.028])
    b = np.array([-0.01, -0.02, -0.015, -0.012])
    assert st.welch_t(a, b) > 0            # a clearly above b
    assert abs(st.welch_t(a, a)) < 1e-9    # identical -> 0


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=5) - st.one_sample_t(x)["t"]) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(14, 19)
    assert lo < 14 / 19 < hi
    assert 0.0 <= lo <= hi <= 1.0


# --------------------------------------------------------------------------- #
# Synthetic world — determinism, null quiet, planted edge recovered
# --------------------------------------------------------------------------- #
def test_world_deterministic(edge_world):
    s2, b2, e2 = dt.synthetic_world(edge=0.004, seed=847)
    studio, bench, events = edge_world
    assert np.allclose(studio.to_numpy(), s2.to_numpy())
    assert np.allclose(bench.to_numpy(), b2.to_numpy())
    assert events == e2
    # balanced fresh/rotten pseudo-events
    tiers = [t for _, t in events]
    assert tiers.count("fresh") == tiers.count("rotten")


def test_synthetic_null_is_quiet():
    # edge=0 -> the tier-conditioned gap detector should not fire; mean |t| small
    ts = np.array([st.synthetic_detect(edge=0.0, seed=847 + s)["gap_welch_t"] for s in range(20)])
    assert abs(ts.mean()) < 0.75
    # honest small-sample FPR bound at ~20 events/tier
    assert (np.abs(ts) >= 2).sum() <= 3


def test_synthetic_planted_edge_lights_up_monotonically():
    # a larger planted tier edge drives the fresh-minus-rotten gap MORE POSITIVE
    g0 = st.synthetic_detect(edge=0.0, seed=847)["gap_bps"]
    g1 = st.synthetic_detect(edge=0.004, seed=847)["gap_bps"]
    g2 = st.synthetic_detect(edge=0.008, seed=847)["gap_bps"]
    assert g0 < g1 < g2
    # a clearly-planted edge is detected across seeds
    ts = np.array([st.synthetic_detect(edge=0.004, seed=847 + s)["gap_welch_t"] for s in range(20)])
    assert ts.mean() > 2.0


def test_synthetic_planted_edge_signs():
    d = st.synthetic_detect(edge=0.006, seed=847)
    assert d["fresh_bps"] > 0        # fresh pseudo-events drift up
    assert d["rotten_bps"] < 0       # rotten pseudo-events drift down


# --------------------------------------------------------------------------- #
# Event-table mechanics on a tiny synthetic price panel (offline, no cache)
# --------------------------------------------------------------------------- #
def _toy_prices(seed: int = 1) -> dict[str, pd.Series]:
    """A tiny, valid per-ticker price panel so build_event_table runs offline."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-06-01", periods=1150)   # lookback before the first film, like the real cache
    out = {}
    for t in dt.all_tickers():
        out[t] = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    return out


def test_build_event_table_includes_and_is_market_adjusted():
    prices = _toy_prices()
    ev = st.build_event_table(prices)
    assert len(ev) == 40
    # films dated 2022-04 .. 2025-06 all fall inside a 900-bday tape from 2022-04-01
    assert int(ev["included"].sum()) == 40
    for c in ("ow_car", "fw_car", "full_car", "anchor_date"):
        assert c in ev.columns
    # full_car ~ ow_car + fw_car (contiguous windows [0..1]+[2..6]=[0..6])
    inc = ev[ev["included"]]
    diff = (inc["full_car"] - (inc["ow_car"] + inc["fw_car"])).abs().max()
    assert diff < 1e-9


def test_tier_stats_and_permutation_placebo_on_toy():
    prices = _toy_prices()
    ev = st.build_event_table(prices)
    ts = st.tier_stats(ev, col="fw_car")
    assert ts["n_fresh"] + ts["n_rotten"] == 40
    # random price panel -> no real tier gap; placebo p should be unremarkable (not in a tail)
    perm = st.permutation_placebo(ev, col="fw_car", n_seeds=4, n_draws_per_seed=500)
    assert 0.02 < perm["p_value"] < 0.98


def test_market_adjusted_ar_is_demeaned():
    prices = _toy_prices()
    ar = st.market_adjusted_ar(prices["DIS"], prices["SPY"])
    assert abs(np.nanmean(ar.to_numpy())) < 1e-12


def test_timer_costs_reduce_net():
    prices = _toy_prices()
    ev = st.build_event_table(prices)
    gross = st.timer_stats(ev, col="fw_car", cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(ev, col="fw_car", cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


# --------------------------------------------------------------------------- #
# Real-tape smoke test — skipped when the git-ignored cache is absent (offline CI)
# --------------------------------------------------------------------------- #
_CACHE_PRESENT = dt.have_real()


@pytest.mark.skipif(not _CACHE_PRESENT, reason="real cache absent offline CI")
def test_real_tape_builds_full_event_table():
    prices = dt.load_real()
    ev = st.build_event_table(prices)
    assert int(ev["included"].sum()) == 40      # full coverage on the real tape
    ts = st.tier_stats(ev, col="ow_car")
    # the opening-weekend gap is the clean, direct window — it is NOT significant
    assert abs(ts["gap_welch_t"]) < 2.0
