"""The synthetic PEAD panel is deterministic; the long-positive/short-negative-surprise book recovers the
drift gross on the control and finds ~nothing on the null; the signal is dollar-neutral and strictly
causal; the surprise-signed cumulative-abnormal-return curve rises then flattens (the PEAD shape); and the
cost / break-even / holding-period machinery behaves. Fast — no network."""

import os

import numpy as np
import pandas as pd
import pytest

from aftershock import costs, data, extension, strategy


def test_panel_deterministic_and_has_drift(control):
    panel, events, truth = control
    p2, e2, _ = data.synthetic_pead(drift_strength=0.0005, seed=34)
    assert np.allclose(panel.to_numpy(), p2.to_numpy())               # seeded → reproducible
    assert e2.equals(events)                                          # same events too
    assert truth.has_drift


def test_book_recovers_drift_gross(control):
    panel, events, _ = control
    s = strategy.summary(strategy.book_returns(panel, events, cost_bps=0.0))
    assert s["sharpe"] > 1.0                                          # real drift → strong gross


def test_book_flat_on_null(null):
    panel, events, _ = null
    s = strategy.summary(strategy.book_returns(panel, events, cost_bps=0.0))
    assert abs(s["sharpe"]) < 0.8                                     # surprises are noise → no drift


def test_weights_are_dollar_neutral(control):
    panel, events, _ = control
    w = strategy.pead_weights(panel, events, holding_days=50)
    net = w.sum(axis=1).abs()
    assert net.max() < 1e-9                                           # long and short legs net to zero
    gross = w.abs().sum(axis=1)
    assert abs(gross[gross > 0].mean() - 1.0) < 1e-6                  # gross normalised to 1


def test_signal_is_causal(control):
    """pead_weights is lagged: flipping the last day's returns leaves earlier weights untouched, and the
    signal never references a future earnings date."""
    panel, events, _ = control
    w = strategy.pead_weights(panel, events, holding_days=50)
    p2 = panel.copy()
    p2.iloc[-1] *= -5.0
    w2 = strategy.pead_weights(p2, events, holding_days=50)
    assert w.iloc[:-1].equals(w2.iloc[:-1])                           # weights depend only on past info


def test_drift_curve_rises_then_flattens(control, null):
    """The surprise-signed cumulative abnormal return rises over the post-event window on the control
    (under-reaction) and is roughly flat on the null."""
    pc, ec, _ = control
    curve = extension.drift_decay_curve(pc, ec, window=70)
    assert curve.iloc[-1] > curve.iloc[0]                            # net drift in the surprise direction
    assert curve.iloc[60] >= curve.iloc[20] - 1e-12                   # monotone-ish rise, then flattens
    # late-window slope is smaller than early-window slope (the drift decays)
    early = curve.iloc[20] - curve.iloc[0]
    late = curve.iloc[-1] - curve.iloc[50]
    assert late < early
    pn, en, _ = null
    null_curve = extension.drift_decay_curve(pn, en, window=70)
    assert abs(null_curve.iloc[-1]) < abs(curve.iloc[-1])             # null drift is far smaller


def test_costs_and_breakeven(control):
    panel, events, _ = control
    assert strategy.turnover(panel, events) > 0.0
    be = costs.breakeven_cost_bps(panel, events)
    assert be > 0.0                                                   # there is a gross edge to erode
    cs = costs.cost_sweep(panel, events)
    assert cs["sharpe"].iloc[0] >= cs["sharpe"].iloc[-1]             # higher cost ⇒ lower Sharpe


def test_holding_period_sweep(control):
    panel, events, _ = control
    hp = extension.holding_period_sweep(panel, events, holds=[10, 40, 90])
    assert list(hp.columns) == ["gross_sharpe", "net_sharpe", "turnover_per_day"]
    assert len(hp) == 3
    assert hp["turnover_per_day"].loc[90] < hp["turnover_per_day"].loc[10]   # longer hold ⇒ less turnover


# --------------------------------------------------------------------------- #
# Cache-gated real-data smoke test — skips cleanly if the EDGAR/panel cache is absent.
# --------------------------------------------------------------------------- #

def _has_real_cache() -> bool:
    cd = data.DEFAULT_CACHE
    return (os.path.exists(os.path.join(cd, data.EDGAR_CACHE))
            and os.path.exists(os.path.join(cd, data.PANEL_CACHE)))


@pytest.mark.skipif(not _has_real_cache(), reason="real EDGAR/panel cache absent")
def test_real_sue_book_runs_and_is_causal():
    """The real SUE book assembles from the cache, runs the same functions as the synthetic path, and is
    strictly causal. We assert structure (not a magnitude), so it is robust to data revisions."""
    eps = pd.read_parquet(os.path.join(data.DEFAULT_CACHE, data.EDGAR_CACHE))
    deduped = data._dedupe_announcements(eps)
    # earliest-filed dedupe: never more rows than the raw, one row per (ticker, period_end)
    assert len(deduped) <= len(eps)
    assert not deduped.duplicated(["ticker", "period_end"]).any()

    events = data.build_sue_events(eps)
    assert list(events.columns) == ["date", "ticker", "surprise"]
    assert np.isfinite(events["surprise"]).all()

    bundle = data.fetch_earnings_panel()           # cache-first, offline
    panel, ev = bundle["panel"], bundle["events"]
    assert panel.shape[1] > 100 and len(ev) > 1000

    # Subset to keep the smoke test fast (the per-event loops are O(events)); structure is what matters.
    names = sorted(ev["ticker"].unique())[:30]
    sub_ev = ev[ev["ticker"].isin(names)].reset_index(drop=True)
    sub_panel = panel[names]

    # the book runs and is dollar-neutral + causal on the real frame
    w = strategy.pead_weights(sub_panel, sub_ev, holding_days=60)
    assert w.sum(axis=1).abs().max() < 1e-9
    p2 = sub_panel.copy(); p2.iloc[-1] *= -3.0
    w2 = strategy.pead_weights(p2, sub_ev, holding_days=60)
    assert w.iloc[:-1].equals(w2.iloc[:-1])        # weights depend only on the past

    # the drift-decay curve is NaN-safe on the real panel (which has per-name gaps)
    car = extension.drift_decay_curve(sub_panel, sub_ev, window=70)
    assert np.isfinite(car.to_numpy()).all()
