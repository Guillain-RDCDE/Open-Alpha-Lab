"""Beat-7 extensions: data winsorizing, the DF gate, stop-loss, cointegration run."""

import numpy as np
import pandas as pd

from pairs_trading import backtest, data, pairs, robustness


def test_clean_panel_kills_bad_print(panel):
    """A single impossible print is winsorized away; sane series are untouched."""
    dirty = panel.copy()
    col = dirty.columns[0]
    # inject a 5,000,000% bad print like the real BMW glitch
    pos = 100
    dirty.iloc[pos, dirty.columns.get_loc(col)] *= 50_000.0
    raw_ret = dirty[col].pct_change().abs().max()
    clean = data.clean_panel(dirty, clip_daily=1.0)
    assert raw_ret > 100                                  # the dirty spike is enormous
    assert clean[col].pct_change().abs().max() <= 1.0 + 1e-9   # clipped to <=100%/day
    # a clean column is essentially unchanged (returns preserved)
    other = panel.columns[1]
    assert np.allclose(panel[other].pct_change().dropna(),
                       clean[other].pct_change().dropna(), atol=1e-9)


def test_df_tstat_separates_reversion_from_walk():
    rng = np.random.default_rng(0)
    n = 300
    # a mean-reverting OU spread -> strongly negative DF t
    ou = np.empty(n); ou[0] = 0.0
    for t in range(1, n):
        ou[t] = 0.8 * ou[t - 1] + rng.standard_normal()
    # a random walk -> DF t near 0 (not significantly negative)
    walk = np.cumsum(rng.standard_normal(n))
    assert pairs.df_tstat(ou) < -2.86                    # rejects unit root
    assert pairs.df_tstat(walk) > -2.86                  # cannot reject


def test_df_tstat_handles_constant_spread():
    assert np.isnan(pairs.df_tstat(np.ones(100)))        # singular -> nan, not a crash


def test_stop_loss_caps_drawdown(panel):
    base = backtest.run(panel, top_n=8, stop_loss=None)
    stopped = backtest.run(panel, top_n=8, stop_loss=0.05)
    # a tight stop never deepens the worst drawdown (max_drawdown is <= 0)
    assert stopped.stats["max_drawdown"] >= base.stats["max_drawdown"] - 1e-9


def test_cointegration_gate_filters_and_runs(panel):
    base = backtest.run(panel, top_n=20, cointegration=False)
    gated = backtest.run(panel, top_n=20, cointegration=True)
    # the gate is a real filter: it changes the traded set (selection differs)
    assert gated.stats["n_trades"] != base.stats["n_trades"] or \
        gated.stats["committed_monthly_net"] != base.stats["committed_monthly_net"]


def test_stop_loss_scan_monotone_columns(panel):
    scan = robustness.stop_loss_scan(panel, top_n=8, stops=(None, 0.05, 0.10))
    assert {"committed_monthly_net", "max_drawdown", "sharpe_net"} <= set(scan.columns)
    assert len(scan) == 3
