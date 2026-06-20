"""The event-study engine's invariants, the right-null inference, and the study's two
spines: (1) the engine detects a run-up premium only when one is planted, and (2) on the
null tape the front-run is indistinguishable from noise (|t| below the bar)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from russell_reconstitution import data, strategy as st  # noqa: E402

IWM_CACHE = os.path.join(data.DEFAULT_CACHE, "bars_IWM_daily.parquet")


# ---- HAC t-stat -----------------------------------------------------------
def test_hac_tstat_zero_mean_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 500)
    assert abs(st.hac_tstat(x, lag=2)) < 3.0


def test_hac_tstat_shifted_mean_large():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 1.0, 500)
    assert st.hac_tstat(x, lag=2) > 5.0


# ---- window construction --------------------------------------------------
def test_build_windows_one_per_event(null_tape):
    bars, truth = null_tape
    win = st.build_windows(bars, runup_days=5)
    assert len(win) == truth["n_events"]
    assert list(win.columns) == ["year", "event_date", "runup_ret", "event_day_ret", "post_ret"]
    # every event_date is on/just before a late-June Friday
    assert (win["event_date"].dt.month == 6).all()


def test_runup_excludes_event_day(null_tape):
    """The run-up window ends at the event close; the event-day return is separate."""
    bars, _ = null_tape
    win = st.build_windows(bars, runup_days=5)
    close = bars["close"]
    idx = bars.index
    # reconstruct one window and check it matches close[loc]/close[loc-5]-1
    loc = idx.searchsorted(win["event_date"].iloc[3], side="right") - 1
    expect = close.iat[loc] / close.iat[loc - 5] - 1.0
    assert np.isclose(win["runup_ret"].iloc[3], expect)


def test_baseline_window_matches_length(null_tape):
    bars, _ = null_tape
    base = st.runup_windows_baseline(bars, runup_days=5)
    # a 5-session block return equals close/close.shift(5)-1
    manual = (bars["close"] / bars["close"].shift(5) - 1.0).dropna()
    assert np.allclose(base.to_numpy(), manual.to_numpy())


def test_june_baseline_is_subset(null_tape):
    bars, _ = null_tape
    full = st.runup_windows_baseline(bars, runup_days=5)
    june = st.runup_windows_baseline(bars, runup_days=5, month=6)
    assert len(june) < len(full)
    assert (june.index.month == 6).all()


# ---- event-study stats ----------------------------------------------------
def test_event_stats_keys(null_tape):
    bars, _ = null_tape
    win = st.build_windows(bars, runup_days=5)
    base = st.runup_windows_baseline(bars, runup_days=5)
    es = st.event_stats(win["runup_ret"], base, n_permutations=500, seed=1)
    for k in ("n", "ev_mean", "base_mean", "excess_mean", "hac_t_excess", "perm_p"):
        assert k in es
    assert 0.0 <= es["perm_p"] <= 1.0


def test_excess_is_event_minus_baseline(null_tape):
    bars, _ = null_tape
    win = st.build_windows(bars, runup_days=5)
    base = st.runup_windows_baseline(bars, runup_days=5)
    es = st.event_stats(win["runup_ret"], base, n_permutations=200, seed=1)
    assert np.isclose(es["excess_mean"], es["ev_mean"] - es["base_mean"], atol=1e-8)


# ---- the two spines -------------------------------------------------------
def test_null_tape_is_not_significant(null_tape):
    """Spine #2: with NO planted premium the front-run is indistinguishable from noise."""
    bars, _ = null_tape
    win = st.build_windows(bars, runup_days=5)
    base = st.runup_windows_baseline(bars, runup_days=5)
    es = st.event_stats(win["runup_ret"], base, n_permutations=2000, seed=1)
    assert abs(es["hac_t_excess"]) < 2.0
    assert es["perm_p"] > 0.05


def test_planted_premium_is_detected(planted_tape):
    """Spine #1: a large planted run-up premium clears the |t|>=2 bar (machinery proof)."""
    bars, _ = planted_tape
    win = st.build_windows(bars, runup_days=5)
    base = st.runup_windows_baseline(bars, runup_days=5)
    es = st.event_stats(win["runup_ret"], base, n_permutations=2000, seed=1)
    assert es["excess_mean"] > 0.0
    assert es["hac_t_excess"] > 2.0


# ---- tradability ----------------------------------------------------------
def test_net_is_gross_minus_cost(null_tape):
    bars, _ = null_tape
    win = st.build_windows(bars, runup_days=5)
    tr = st.tradability(win["runup_ret"], cost_bps_oneway=1.5)
    assert np.isclose(
        tr["net_mean_per_event"], tr["gross_mean_per_event"] - 2.0 * 1.5e-4, atol=1e-9
    )


def test_cost_lowers_net(null_tape):
    bars, _ = null_tape
    win = st.build_windows(bars, runup_days=5)
    lo = st.tradability(win["runup_ret"], cost_bps_oneway=0.0)["net_mean_per_event"]
    hi = st.tradability(win["runup_ret"], cost_bps_oneway=3.0)["net_mean_per_event"]
    assert hi < lo


def test_subperiod_stats_partition(null_tape):
    bars, _ = null_tape
    win = st.build_windows(bars, runup_days=5)
    base = st.runup_windows_baseline(bars, runup_days=5)
    sp = st.subperiod_stats(win, base)
    assert int(sp["n"].sum()) == len(win)


# ---- real tape (cache-gated) ----------------------------------------------
@pytest.mark.skipif(not os.path.exists(IWM_CACHE), reason="offline CI: no IWM cache")
def test_real_iwm_runup_is_null():
    """On the real IWM tape the front-run does NOT clear the bar (the study's verdict)."""
    bars = data.fetch_daily("IWM", fetch=False)
    s = st.summarize(bars, runup_days=5, n_permutations=2000, seed=320)
    assert s["n"] >= 20
    assert abs(s["hac_t_excess"]) < 2.0  # NONE on the Signal axis
