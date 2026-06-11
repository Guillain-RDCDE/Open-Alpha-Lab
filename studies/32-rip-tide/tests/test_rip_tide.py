"""The synthetic panel is deterministic; the contrarian book recovers the reversion premium gross and
finds none in the null; the signal is causal and carries exactly one execution lag (signal at the close
of t, position earns t+1); turnover is high; the cost wall bites; and the holding-period rescue and
sub-period table report gross and net, labelled."""

import numpy as np
import pandas as pd

from rip_tide import costs, data, extension, strategy


def test_panel_deterministic_and_has_reversion(revert_ret, revert):
    r2, _ = data.synthetic_reversion(revert_strength=0.06, seed=32)
    assert np.allclose(revert_ret.to_numpy(), r2.to_numpy())          # seeded → reproducible
    assert revert[1].has_reversion


def test_book_recovers_reversion_gross(revert_ret):
    """Gross of costs, the contrarian book extracts the negative-autocorrelation premium."""
    s = strategy.summary(strategy.book_returns(revert_ret, cost_bps=0.0))
    assert s["sharpe"] > 1.0                                          # strong gross on the control


def test_book_flat_on_null(null_ret):
    s = strategy.summary(strategy.book_returns(null_ret, cost_bps=0.0))
    assert abs(s["sharpe"]) < 0.8                                     # nothing to fade in a random walk


def test_signal_is_causal_and_contrarian():
    """reversal_signal at row t uses returns through the close of t (no internal lag) and is the exact
    negative of the momentum sign of the same window."""
    idx = pd.bdate_range("2010-01-04", periods=400)
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.standard_normal((400, 3)) * 0.01, index=idx, columns=["A", "B", "C"])
    sig = strategy.reversal_signal(r, lookbacks=(1, 3, 5))
    assert sig.iloc[0].abs().sum() == 0                              # first row has no past → 0
    # with a 1-day lookback, the signal at t is just the opposite sign of day t's own return
    sig1 = strategy.reversal_signal(r, lookbacks=(1,))
    assert (sig1.iloc[5:] == -np.sign(r.iloc[5:])).all().all()
    # changing the LAST day's return must not change any EARLIER signal (uses no future data)
    r2 = r.copy(); r2.iloc[-1] *= -5
    assert sig.iloc[:-1].equals(strategy.reversal_signal(r2, lookbacks=(1, 3, 5)).iloc[:-1])
    # contrarian: the reversal signal is the negative of a momentum (long recent winner) signal
    prices = (1.0 + r.fillna(0.0)).cumprod()
    mom = np.sign(sum(np.sign(prices / prices.shift(lb) - 1.0) for lb in (1, 3, 5)) / 3).fillna(0.0)
    assert sig.equals((-mom).replace(-0.0, 0.0))


def test_single_execution_lag_catches_the_bounce():
    """The book carries exactly ONE lag: the position decided at the close of t earns t+1. On a strictly
    alternating tape (+1%, −1%, …) a 1-day contrarian must make money every day after warm-up; with an
    accidental double lag it would LOSE every day (it would fade the bounce one day too late)."""
    idx = pd.bdate_range("2010-01-04", periods=300)
    alt = pd.DataFrame({"A": [0.01 * (-1) ** t for t in range(300)]}, index=idx)
    gross = strategy.book_returns(alt, cost_bps=0.0).iloc[100:]      # after the vol warm-up
    assert (gross > 0).all()


def test_turnover_is_high(revert_ret):
    """The whole tension: a 1–5 day signal flips constantly → turnover well above a slow trend book."""
    assert strategy.turnover(revert_ret) > 0.1


def test_cost_wall_and_breakeven(revert_ret):
    cs = costs.cost_sweep(revert_ret)
    assert cs["sharpe"].iloc[0] >= cs["sharpe"].iloc[-1]            # higher cost ⇒ lower Sharpe
    be = costs.breakeven_cost_bps(revert_ret)
    assert be >= 0.0                                                # a finite break-even cost exists


def test_holding_period_sweep_shape(revert_ret):
    sw = extension.holding_period_sweep(revert_ret, holds=[1, 5, 21])
    assert list(sw.columns) == ["gross_sharpe", "net_sharpe", "turnover_per_day"]
    # slowing down cuts turnover
    assert sw["turnover_per_day"].loc[21] < sw["turnover_per_day"].loc[1]


def test_horizon_sweep_shape(revert_ret):
    sw = extension.horizon_sweep(revert_ret)
    assert "gross_sharpe" in sw.columns and "net_sharpe" in sw.columns and len(sw) >= 4


def test_subperiod_sharpe_reports_gross_and_net(revert_ret):
    """Sub-periods must be quoted gross AND net, labelled — never a net number sold as gross."""
    sp = extension.subperiod_sharpe(revert_ret, n_splits=3, cost_bps=2.0)
    assert list(sp.columns)[:2] == ["gross_sharpe", "net_sharpe"] and len(sp) == 3
    assert (sp["net_sharpe"] <= sp["gross_sharpe"] + 1e-9).all()    # cost can only hurt
