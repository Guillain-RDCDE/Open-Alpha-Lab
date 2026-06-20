"""Event-tagging invariants, the HAC/Welch table, the block-bootstrap CI, the one traded
arm, and the study's two spines: (1) the engine detects a planted speech-day bump and
finds nothing in the null, and (2) the reaction trade applies exactly one execution lag."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jackson_hole import strategy as st  # noqa: E402


# ---- event tagging --------------------------------------------------------
def test_event_window_offset_count(null_tape):
    ret, speech, _ = null_tape
    # each in-range event tags exactly one session at offset 0
    m0 = st.event_window(ret.index, speech, offset=0)
    assert m0.sum() == len(speech)


def test_event_window_offset_shifts(null_tape):
    """offset=+1 tags the session right after offset=0's session."""
    ret, speech, _ = null_tape
    m0 = np.where(st.event_window(ret.index, speech, 0).to_numpy())[0]
    m1 = np.where(st.event_window(ret.index, speech, 1).to_numpy())[0]
    assert np.array_equal(m1, m0 + 1)


def test_event_table_keys(null_tape):
    ret, speech, _ = null_tape
    t = st.event_table(ret, speech, 0)
    assert set(t) >= {"event_mean", "rest_mean", "diff_bps", "welch_t", "hac_t",
                      "win_rate", "n_event", "n_total"}
    assert t["n_event"] == len(speech)


# ---- spine #1: detect the planted effect, stay quiet on the null ----------
def test_detects_planted_bump_quiet_on_null(null_tape, planted_tape):
    r0, sp0, _ = null_tape
    r1, sp1, _ = planted_tape
    t0 = st.event_table(r0, sp0, 0)["hac_t"]
    t1 = st.event_table(r1, sp1, 0)["hac_t"]
    assert abs(t0) < 2.0          # null: indistinguishable from zero
    assert t1 > 2.0               # planted +80 bps: clears the bar


def test_offset_sweep_shape(null_tape):
    ret, speech, _ = null_tape
    sw = st.offset_sweep(ret, speech, offsets=range(-5, 6))
    assert len(sw) == 11
    assert (sw["offset"].to_numpy() == np.arange(-5, 6)).all()


# ---- the bootstrap CI -----------------------------------------------------
def test_bootstrap_ci_brackets_mean_planted(planted_tape):
    """The CI should bracket the (large, planted) event-day mean."""
    ret, speech, _ = planted_tape
    mean = st.event_table(ret, speech, 0)["event_mean"]
    lo, hi = st.block_bootstrap_ci(ret, speech, 0, n_boot=2000, seed=1)
    assert lo <= mean <= hi
    assert lo < hi


def test_bootstrap_ci_straddles_zero_on_null(null_tape):
    """On the null the speech-day CI must straddle zero (no effect)."""
    ret, speech, _ = null_tape
    lo, hi = st.block_bootstrap_ci(ret, speech, 0, n_boot=2000, seed=1)
    assert lo < 0 < hi


# ---- spine #2: the reaction trade and its single execution lag ------------
def test_react_trade_columns_and_count(null_tape):
    ret, speech, _ = null_tape
    led = st.react_trade(ret, speech, hold=1, cost_bps=1.0)
    assert list(led.columns) == ["year", "entry_date", "ret_gross", "ret_net"]
    assert len(led) == len(speech)


def test_react_trade_one_lag(null_tape):
    """The reaction trade earns the NEXT session's return after the speech-day close —
    exactly one shift, not zero and not two."""
    ret, speech, _ = null_tape
    idx = ret.index
    led = st.react_trade(ret, speech, hold=1, cost_bps=0.0)
    for _, row in led.head(5).iterrows():
        pos = idx.searchsorted(row["entry_date"])
        # gross return == the single next session's return
        expected = ret.iloc[pos + 1]
        assert np.isclose(row["ret_gross"], expected, atol=1e-12)


def test_react_trade_cost_lowers_return(null_tape):
    ret, speech, _ = null_tape
    g = st.react_trade(ret, speech, hold=1, cost_bps=0.0)["ret_net"].mean()
    n = st.react_trade(ret, speech, hold=1, cost_bps=5.0)["ret_net"].mean()
    assert n < g


def test_summarize_trades_empty():
    s = st.summarize_trades(pd.DataFrame(columns=["ret_net"]))
    assert s["n"] == 0
    assert np.isnan(s["mean_bps"])


def test_hac_t_zero_mean_is_small():
    """A zero-mean noise series has a HAC t near zero."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000) * 0.01
    assert abs(st._hac_t(x)) < 2.0
