"""Tests for the strategy / event-study layer of Study 298 (Swiftonomics).

All tests are offline and deterministic -- they use the synthetic generator and
the hardcoded event table only. The spine of the suite: the engine reads ~zero on
the null tape and lights up on a planted positive control.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swiftonomics import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# Returns and event mapping
# ---------------------------------------------------------------------------
def test_daily_returns_columns(synthetic_null):
    prices, _, _ = synthetic_null
    rets = st.daily_returns(prices)
    assert list(rets.columns) == [data.STOCK, data.MARKET]
    assert len(rets) == len(prices) - 1


def test_map_events_to_trading_days(eras_events):
    prices, _, _ = data.synthetic_panel(seed=298)
    mapped = st.map_events_to_trading_days(eras_events, st.daily_returns(prices).index)
    assert "event_pos" in mapped.columns
    assert "event_day" in mapped.columns
    # Mapped trading days are within the price index
    assert mapped["event_day"].isin(st.daily_returns(prices).index).all()


def test_map_events_weekend_maps_forward():
    import pandas as pd
    ti = pd.bdate_range("2023-01-02", periods=10)
    ev = pd.DataFrame({"date": [pd.Timestamp("2023-01-07")],  # a Saturday
                       "label": ["x"], "kind": ["announce"]})
    mapped = st.map_events_to_trading_days(ev, ti)
    assert mapped.iloc[0]["event_day"] >= pd.Timestamp("2023-01-09")  # next Monday


# ---------------------------------------------------------------------------
# event_study: output structure and invariants
# ---------------------------------------------------------------------------
def test_event_study_keys(synthetic_null):
    prices, events, _ = synthetic_null
    study = st.event_study(prices, events, pre=1, post=3)
    for k in ("car", "ar_matrix", "aar", "caar", "betas", "offsets", "n_events"):
        assert k in study


def test_event_study_shapes(synthetic_null):
    prices, events, _ = synthetic_null
    pre, post = 1, 3
    study = st.event_study(prices, events, pre=pre, post=post)
    width = pre + post + 1
    assert study["ar_matrix"].shape == (study["n_events"], width)
    assert study["aar"].shape == (width,)
    assert study["caar"].shape == (width,)
    assert len(study["offsets"]) == width


def test_event_study_recovers_beta():
    """The estimated betas should be near the planted beta on a clean panel."""
    prices, events, truth = data.synthetic_panel(beta=1.30, bump_bps=0.0, seed=298)
    study = st.event_study(prices, events, pre=1, post=3)
    assert abs(np.median(study["betas"]) - truth["beta"]) < 0.20


def test_caar_is_cumsum_of_aar(synthetic_null):
    prices, events, _ = synthetic_null
    study = st.event_study(prices, events, pre=1, post=3)
    assert np.allclose(study["caar"], np.cumsum(study["aar"]))


# ---------------------------------------------------------------------------
# car_inference
# ---------------------------------------------------------------------------
def test_car_inference_keys(synthetic_null):
    prices, events, _ = synthetic_null
    study = st.event_study(prices, events)
    inf = st.car_inference(study)
    for k in ("n_events", "mean_car_bps", "car_t", "car_p", "hit_rate", "aar_hac_t"):
        assert k in inf


def test_car_p_is_probability(synthetic_null):
    prices, events, _ = synthetic_null
    study = st.event_study(prices, events)
    inf = st.car_inference(study)
    assert 0.0 <= inf["car_p"] <= 1.0
    assert 0.0 <= inf["hit_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Null vs planted signal: the spine of the study
# ---------------------------------------------------------------------------
def test_null_has_no_signal():
    """On the null tape (bump=0), the mean CAR t-stat should be small."""
    prices, events, _ = data.synthetic_panel(
        n_days=900, n_events=40, bump_bps=0.0, seed=298)
    study = st.event_study(prices, events, pre=1, post=3)
    inf = st.car_inference(study)
    assert abs(inf["car_t"]) < 2.0


def test_planted_signal_detectable():
    """A large planted abnormal return is recovered with the right sign and significance."""
    prices, events, _ = data.synthetic_panel(
        n_days=900, n_events=40, bump_bps=400.0, seed=298)
    study = st.event_study(prices, events, pre=1, post=3)
    inf = st.car_inference(study)
    assert inf["mean_car_bps"] > 0
    assert inf["car_t"] > 2.0
    assert inf["hit_rate"] > 0.5


def test_event_study_no_lookahead_clean_window():
    """Estimation window ends strictly before the event window (no overlap)."""
    # An event very early in the sample (no room for estimation) must be skipped.
    prices, _, _ = data.synthetic_panel(seed=298)
    import pandas as pd
    early = pd.DataFrame({"date": [prices.index[3]], "label": ["early"], "kind": ["announce"]})
    study = st.event_study(prices, early, pre=1, post=3, est_window=120, gap=5)
    assert study["n_events"] == 0


# ---------------------------------------------------------------------------
# Placebo
# ---------------------------------------------------------------------------
def test_placebo_distribution_shape(synthetic_null):
    prices, _, _ = synthetic_null
    plc = st.placebo_distribution(prices, n_events=16, n_draws=200, seed=298)
    assert plc["mean_cars"].shape == (200,)


def test_placebo_deterministic(synthetic_null):
    prices, _, _ = synthetic_null
    a = st.placebo_distribution(prices, n_events=16, n_draws=100, seed=7)
    b = st.placebo_distribution(prices, n_events=16, n_draws=100, seed=7)
    assert np.allclose(a["mean_cars"], b["mean_cars"])


def test_placebo_pvalue_in_range(synthetic_null):
    prices, _, _ = synthetic_null
    plc = st.placebo_distribution(prices, n_events=16, n_draws=300, seed=298)
    p = st.placebo_pvalue(0.0, plc["mean_cars"])
    assert 0.0 <= p <= 1.0


def test_placebo_centred_near_zero():
    """Random-date placebo means are centred near zero on the null tape."""
    prices, _, _ = data.synthetic_panel(bump_bps=0.0, seed=298)
    plc = st.placebo_distribution(prices, n_events=16, n_draws=1000, seed=298)
    assert abs(plc["mean_cars"].mean()) < 50e-4  # within 50 bps of zero


# ---------------------------------------------------------------------------
# Tradable sleeve
# ---------------------------------------------------------------------------
def test_tradable_sleeve_keys(synthetic_null):
    prices, events, _ = synthetic_null
    sl = st.tradable_sleeve(prices, events)
    for k in ("raw_gross", "raw_net", "hedged_gross", "hedged_net"):
        assert k in sl
        assert "mean_bps" in sl[k]


def test_tradable_costs_reduce_net(synthetic_signal):
    """Net (after costs) is always <= gross for the same trade."""
    prices, events, _ = synthetic_signal
    sl = st.tradable_sleeve(prices, events, one_way_bps=20.0)
    assert sl["raw_net"]["mean_bps"] <= sl["raw_gross"]["mean_bps"] + 1e-6
    assert sl["hedged_net"]["mean_bps"] <= sl["hedged_gross"]["mean_bps"] + 1e-6


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys(synthetic_null):
    prices, events, _ = synthetic_null
    s = st.summarize(prices, events, n_placebo=200, seed=298)
    required = {
        "n_events", "mean_car_bps", "car_t", "car_p", "hit_rate",
        "aar_hac_t", "median_beta", "placebo_std_bps", "placebo_p",
        "sleeve_hedged_gross_bps", "sleeve_hedged_net_bps",
    }
    assert required.issubset(set(s.keys()))


def test_summarize_planted_positive():
    prices, events, _ = data.synthetic_panel(
        n_days=900, n_events=40, bump_bps=400.0, seed=298)
    s = st.summarize(prices, events, n_placebo=300, seed=298)
    assert s["mean_car_bps"] > 0
    assert s["car_t"] > 2.0
