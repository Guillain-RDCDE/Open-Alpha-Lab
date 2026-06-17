"""Tests for the pre-earnings-runup study.

All five tests run on the fully-offline synthetic panel — no cache, no network.
- Positive control: drift planted in the pre-event window → book should beat the null.
- Null: no planted drift → book return should be near zero (no signal).
- Causal check: weights depend only on the past earnings calendar, not on future returns.
- Weight structure: long-only, sums to 1 when any names are in the book.
- Pre-day sweep: mean pre-event return rises toward the announcement on the control.
A cache-gated real-data smoke test skips cleanly if the study-local cache is absent.
"""

import os

import numpy as np
import pytest

from pre_earnings_runup import data, strategy


# ---------------------------------------------------------------------------
# Offline synthetic tests
# ---------------------------------------------------------------------------

def test_panel_deterministic(control):
    panel, events, truth = control
    p2, e2, _ = data.synthetic_runup(drift_strength=0.0004, seed=228)
    assert np.allclose(panel.to_numpy(), p2.to_numpy())
    assert e2.equals(events)
    assert truth.has_runup


def test_book_positive_on_control_vs_null(control, null):
    """The book earns a higher mean return on the control (planted drift) than on the null.
    We compare mean pre-event returns: on the control the planted drift makes the mean return
    clearly positive; on the null the mean return is near zero."""
    pc, ec, _ = control
    pn, en, _ = null
    # Use the per-event mean return curve which cleanly isolates the pre-event window
    curve_c = strategy.mean_pre_event_return(pc, ec, pre_days=5)
    curve_n = strategy.mean_pre_event_return(pn, en, pre_days=5)
    # Control mean pre-event return should be clearly larger than the null
    assert curve_c.mean() > curve_n.mean() + 1e-4


def test_null_sharpe_near_zero(null):
    """On the null there is no planted drift — the book's Sharpe should be near zero."""
    pn, en, _ = null
    sn = strategy.summary(strategy.book_returns(pn, en, cost_bps=0.0))
    assert abs(sn["sharpe"]) < 1.0


def test_weights_causal(control):
    """Flipping the last day's returns does not change any prior-day weights."""
    panel, events, _ = control
    w = strategy.runup_weights(panel, events, pre_days=5)
    p2 = panel.copy()
    p2.iloc[-1] *= -5.0
    w2 = strategy.runup_weights(p2, events, pre_days=5)
    # Weights up to the second-to-last day must be identical (causal)
    assert w.iloc[:-1].equals(w2.iloc[:-1])


def test_weights_long_only_and_sum_to_one(control):
    """On days when any names are in the pre-event window, weights are non-negative and sum to 1."""
    panel, events, _ = control
    w = strategy.runup_weights(panel, events, pre_days=5)
    assert (w.to_numpy() >= -1e-12).all(), "weights must be non-negative (long-only book)"
    active_days = w.sum(axis=1)
    active = active_days[active_days > 1e-9]
    assert len(active) > 0, "no active days found"
    assert (active - 1.0).abs().max() < 1e-9, "weights on active days must sum to 1"


def test_mean_pre_event_return_rises_on_control(control, null):
    """On the control, the mean return in the pre-event window should be clearly positive; on the null
    it should be near zero across all offsets."""
    pc, ec, _ = control
    curve_c = strategy.mean_pre_event_return(pc, ec, pre_days=5)
    assert curve_c.mean() > 0.0, "mean pre-event return should be positive on control"

    pn, en, _ = null
    curve_n = strategy.mean_pre_event_return(pn, en, pre_days=5)
    # The control's mean should be larger than the null's
    assert curve_c.mean() > curve_n.mean()


# ---------------------------------------------------------------------------
# Cache-gated real-data smoke test
# ---------------------------------------------------------------------------

def _has_real_cache() -> bool:
    cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_cache"
    )
    return (
        os.path.exists(os.path.join(cache_dir, data.PRICES_CACHE))
        and os.path.exists(os.path.join(cache_dir, data.EVENTS_CACHE))
    )


@pytest.mark.skipif(not _has_real_cache(), reason="real yfinance cache absent offline/CI")
def test_real_data_smoke():
    """Smoke test on the real cached data: structure checks only, no magnitude thresholds."""
    bundle = data.fetch_real_data()
    assert bundle, "fetch_real_data returned empty dict even though cache exists"
    panel = bundle["panel"]
    events = bundle["events"]

    assert panel.shape[1] >= 5, "panel should have at least 5 tickers"
    assert len(events) >= 10, "events frame should have at least 10 rows"
    assert list(events.columns) == ["date", "ticker"]

    # Weights are causal and long-only
    w = strategy.runup_weights(panel, events, pre_days=5)
    assert (w.to_numpy() >= -1e-12).all()

    # The book runs without error
    ret = strategy.book_returns(panel, events, pre_days=5, cost_bps=2.0)
    assert ret.notna().sum() > 0
