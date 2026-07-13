"""Strategy engine invariants and the study's spine: the pre-OpEx drift signal appears
only when we plant it, the placebo isolates a planted anchor, and the real-tape drift is
null (cache-gated)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from charm_decay import data, strategy as st  # noqa: E402

_CACHE_PATH = data._cache_path("SPY", data.DEFAULT_CACHE)
requires_cache = pytest.mark.skipif(
    not os.path.exists(_CACHE_PATH),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# compare_window_vs_baseline invariants
# ---------------------------------------------------------------------------

def test_compare_output_keys(null_tape):
    bars, _ = null_tape
    idx = pd.DatetimeIndex(bars.index)
    res = st.compare_window_vs_baseline(st.daily_return(bars).dropna(),
                                        data.pre_opex_mask(idx), "x")
    assert set(res.keys()) == {"label", "n_window", "n_baseline",
                               "mean_window", "mean_baseline", "diff", "tstat"}


def test_compare_diff_equals_mean_difference(null_tape):
    bars, _ = null_tape
    idx = pd.DatetimeIndex(bars.index)
    res = st.compare_window_vs_baseline(st.daily_return(bars).dropna(),
                                        data.pre_opex_mask(idx))
    assert abs(res["diff"] - (res["mean_window"] - res["mean_baseline"])) < 1e-12


def test_compare_counts_partition(null_tape):
    bars, _ = null_tape
    idx = pd.DatetimeIndex(bars.index)
    r = st.daily_return(bars).dropna()
    res = st.compare_window_vs_baseline(r, data.pre_opex_mask(idx))
    assert res["n_window"] + res["n_baseline"] == len(r)


# ---------------------------------------------------------------------------
# The synthetic positive control — drift appears only when planted
# ---------------------------------------------------------------------------

def test_pre_drift_appears_only_when_planted():
    def pre_t(pb):
        b, _ = data.synthetic_daily(n_years=30, pre_drift_bps=pb, seed=768)
        return st.charm_drift_test(b)["pre"]["tstat"]

    assert abs(pre_t(0.0)) < 2.0
    assert pre_t(10.0) > 3.0


def test_drift_monotone_in_plant():
    ts = [st.charm_drift_test(
        data.synthetic_daily(n_years=30, pre_drift_bps=pb, seed=768)[0]
    )["pre"]["tstat"] for pb in (0.0, 3.0, 6.0, 12.0)]
    assert ts == sorted(ts)  # more planted drift → larger t


def test_asymmetry_keys_and_sign(drift_tape):
    bars, _ = drift_tape
    a = st.pre_post_asymmetry(bars)
    assert set(a.keys()) == {"diff_bps", "mean_pre_bps", "mean_post_bps",
                             "tstat", "n_pre", "n_post"}
    # planted +8 pre / -4 post → positive asymmetry
    assert a["diff_bps"] > 0


def test_placebo_isolates_planted_anchor():
    """On a planted tape the true anchor should be extreme vs fake anchors (small p)."""
    b, _ = data.synthetic_daily(n_years=30, pre_drift_bps=8.0, seed=768)
    pl = st.placebo_randomization(b)
    assert pl["true_t"] > 2.0
    assert pl["p_value"] < 0.1
    assert pl["n_placebo"] > 20


def test_placebo_null_tape_not_extreme():
    """On a null tape the true anchor sits in the body of the placebo cloud."""
    b, _ = data.synthetic_daily(n_years=30, pre_drift_bps=0.0, seed=768)
    pl = st.placebo_randomization(b)
    assert pl["p_value"] > 0.1


# ---------------------------------------------------------------------------
# Overlay / summarize
# ---------------------------------------------------------------------------

def test_overlay_zero_outside_window(null_tape):
    bars, _ = null_tape
    ov = st.charm_overlay_returns(bars)
    idx = pd.DatetimeIndex(ov.index)
    pre = data.pre_opex_mask(idx).reindex(ov.index).fillna(False)
    assert (ov[~pre.values] == 0).all()


def test_summarize_keys_and_short_nan():
    bars, _ = data.synthetic_daily(n_years=10, seed=768)
    s = st.summarize(st.charm_overlay_returns(bars))
    assert set(s.keys()) == {"n", "mean_bps", "sharpe_ann", "cagr", "tstat"}
    assert all(np.isnan(v) for v in st.summarize(pd.Series([0.01, 0.02])).values())


def test_cost_sweep_monotone_decreasing():
    sweep = st.cost_sweep(5.0)
    nets = [m for _, m in sweep]
    assert nets == sorted(nets, reverse=True)
    assert sweep[0][1] == 5.0  # zero cost → gross


# ---------------------------------------------------------------------------
# Cache-gated real-data tests — the study's headline null
# ---------------------------------------------------------------------------

@requires_cache
def test_real_pre_opex_drift_is_null():
    bars = data.fetch_daily("SPY", fetch=False)
    bars = bars[bars.index <= pd.Timestamp("2026-07-10")]
    t = st.charm_drift_test(bars)["pre"]["tstat"]
    assert abs(t) < 2.0, f"Pre-OpEx drift unexpectedly significant: t={t:.2f}"


@requires_cache
def test_real_placebo_not_special():
    bars = data.fetch_daily("SPY", fetch=False)
    bars = bars[bars.index <= pd.Timestamp("2026-07-10")]
    pl = st.placebo_randomization(bars)
    assert pl["p_value"] > 0.2, f"Real anchor unexpectedly extreme: p={pl['p_value']:.2f}"
