"""Walk-forward forecast skill — must beat the random-phase null on a *real* cycle."""

import numpy as np
import pandas as pd

from vix_cycles import backtest, data


def test_dominant_period_finds_injected(synth):
    series, injected = synth
    P = backtest.dominant_period(series, band=(20.0, 200.0))
    assert min(abs(P - c.period) for c in injected) <= 0.2 * 80.0


def test_direction_skill_beats_null_on_real_cycle(synth):
    series, _ = synth
    r = backtest.oos_direction_skill(series, band=(20.0, 200.0), horizon=20,
                                     min_train=750, step=10, n_null=200, seed=0)
    assert r["skill"] > 0.5
    assert r["p_value"] < 0.10        # the learned phase forecasts better than scrambled


def test_direction_skill_no_edge_on_red_noise(red_noise):
    r = backtest.oos_direction_skill(red_noise, band=(20.0, 200.0), horizon=20,
                                     min_train=750, step=10, n_null=200, seed=0)
    assert r["p_value"] > 0.05        # no learnable phase in structureless noise


def _control_price(series: pd.Series, k: float = 0.5) -> pd.Series:
    """The positive-control 'index': log-price moves opposite the vol *change*, tick for tick.

    This is the alignment the trade harness assumes (and the empirical SPX/VIX relation):
    the index rallies *while* vol falls. A level-driven toy (r ∝ −(v − mean)) pays in the
    bottom half of the cycle instead — a quarter-period out of phase with the harness's
    "long while projected to fall" rule — and would fail even on a planted cycle.
    """
    dv = np.diff(series.values, prepend=series.values[0])
    return pd.Series(np.exp(np.cumsum(-k * dv)), index=series.index, name="px")


def test_phase_trade_runs_and_is_causal(synth):
    series, _ = synth
    res = backtest.phase_trade(series, _control_price(series), band=(20.0, 200.0),
                               min_train=750, refit=5, n_null=50, seed=0)
    assert res.daily.iloc[:750].abs().sum() == 0.0    # no position before min_train: causal
    assert "sharpe_net" in res.stats and "p_value_vs_null" in res.stats


def test_phase_trade_banks_a_planted_cycle(synth):
    """Positive control: on a series with a REAL cycle driving the price, the harness must
    bank it — positive Sharpe, above the random-phase null. Otherwise a null verdict on the
    real VIX would be the harness's deafness, not the market's silence."""
    series, _ = synth
    res = backtest.phase_trade(series, _control_price(series), band=(20.0, 200.0),
                               min_train=750, refit=5, n_null=100, seed=0)
    assert res.stats["sharpe_net"] > 1.0              # the planted cycle is banked…
    assert res.stats["p_value_vs_null"] < 0.05        # …and the learned phase beats scrambled


def test_phase_trade_degenerate_gate_reports_nan_p(synth):
    """A gate that never fires has no return stream: Sharpe and p must be NaN, not a
    spuriously significant (0+1)/(n+1) from comparing NaN against the null."""
    series, _ = synth
    res = backtest.phase_trade(series, _control_price(series), band=(20.0, 200.0),
                               min_train=750, refit=5, n_null=50, seed=0, amp_gate=1e9)
    assert res.stats["exposure"] == 0.0
    assert np.isnan(res.stats["sharpe_net"])
    assert np.isnan(res.stats["p_value_vs_null"])
