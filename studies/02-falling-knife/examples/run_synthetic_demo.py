"""Offline demo — no network needed. Validates the whole toolchain.

We build two synthetic markets that share NDX-like drift and *clustered*
volatility (so -3% days bunch up, exactly like the real index):

    World A — NO dip edge: returns are drift + clustered noise, nothing special
              happens after a crash. A correct toolchain must report excess ≈ 0
              and a deflated Sharpe that says "this is just data-mining".

    World B — a REAL mean-reversion edge is injected after big down days. The same
              toolchain must now light up: positive excess vs the random day, and
              an edge that holds up.

If the tools call A flat and B real, we trust them on the live NDX. Run:

    python examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from falling_knife import data, triggers, exits, eventstudy, benchmark, backtest, robustness, plots

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def synthetic_ohlc(n_days=8000, seed=0, mu_annual=0.10, bounce=0.0):
    """Daily OHLC with drift + AR(1) volatility clustering.

    ``bounce`` injects mean-reversion: a fraction of each day's drop is handed
    back over the next few sessions (0 = no edge, World A; >0 = World B).
    """
    rng = np.random.default_rng(seed)
    mu = mu_annual / 252.0
    # AR(1) log-volatility -> clustered crashes (fat left tail in clusters).
    log_vol = np.zeros(n_days)
    log_vol[0] = np.log(0.011)
    for t in range(1, n_days):
        log_vol[t] = 0.95 * log_vol[t - 1] + 0.05 * np.log(0.011) + 0.15 * rng.standard_normal()
    vol = np.exp(log_vol)

    shocks = rng.standard_normal(n_days) * vol
    r_cc = mu + shocks

    if bounce > 0.0:
        # Hand back part of any drop beyond -2% over the next 3 days.
        add = np.zeros(n_days)
        for t in range(n_days):
            if r_cc[t] < -0.02:
                give = -bounce * r_cc[t]
                for k in (1, 2, 3):
                    if t + k < n_days:
                        add[t + k] += give / 3.0
        r_cc = r_cc + add

    close = 1000.0 * np.cumprod(1.0 + r_cc)
    prev_close = np.concatenate([[1000.0], close[:-1]])
    gap = rng.standard_normal(n_days) * vol * 0.4
    open_ = prev_close * (1.0 + gap)
    hi_noise = np.abs(rng.standard_normal(n_days)) * vol * 0.5
    lo_noise = np.abs(rng.standard_normal(n_days)) * vol * 0.5
    high = np.maximum(open_, close) * (1.0 + hi_noise)
    low = np.minimum(open_, close) * (1.0 - lo_noise)

    idx = pd.bdate_range("1994-01-03", periods=n_days)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


def analyse(name, ohlc):
    print(f"\n{'='*70}\n  {name}\n{'='*70}")
    ret = data.daily_returns(ohlc)

    # 1) Event study on the classic close-to-close -3% trigger (T1).
    raw = triggers.close_to_close(ret)
    events = triggers.first_crossings(raw, cooldown=20)
    es = eventstudy.event_study(ohlc, events, horizon=20, pre=5)
    print(f"\n[Event study T1] {es['n_events']} fresh -3% events")
    print(es["summary"].loc[[1, 3, 5, 10, 20], ["mean", "median", "pct_positive", "tstat"]]
          .to_string(float_format=lambda v: f"{v:.4f}"))

    # 2) The decisive test: conditional vs a random day.
    bench = benchmark.conditional_vs_unconditional(ohlc, events, horizons=(1, 3, 5, 10, 20))
    print("\n[Conditional vs random day]  (p_greater small => beats random)")
    print(bench[["n_events", "mean_cond", "mean_uncond", "excess", "p_greater"]]
          .to_string(float_format=lambda v: f"{v:.4f}"))

    # 3) Block bootstrap on the 5-day excess (respects clustering).
    bb = robustness.block_bootstrap_excess(ohlc, events, horizon=5)
    print(f"\n[Block bootstrap, +5d excess]  mean={bb['mean']:+.4f}  "
          f"95% CI [{bb['ci_low']:+.4f}, {bb['ci_high']:+.4f}]  "
          f"P(excess<=0)={bb['p_excess_le_0']:.2f}")

    # 4) Full family scan (data-mining surface) + deflation of the winner.
    sigs = {n: f(ret) for n, f in triggers.TRIGGERS.items()}
    scan = backtest.family_scan(ohlc, ret, sigs, exits.default_grid(),
                                costs=backtest.CostModel())
    print("\n[Family scan] top 5 by ABSOLUTE Sharpe (this number is a trap, see below):")
    print(scan.head(5).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    best = scan.iloc[0]
    # n_obs = active days behind the winning Sharpe, not the calendar length —
    # an event strategy that is flat >90% of the time has far fewer observations.
    dsr = robustness.deflated_sharpe(best["sharpe"], n_trials=len(scan),
                                     n_obs=int(best["active_days"]))
    print(f"\n[Selection check] deflated Sharpe rules out SELECTION luck only — not drift.")
    print(f"   best raw Sharpe={best['sharpe']:.2f} over {len(scan)} trials, "
          f"{int(best['active_days'])} active days -> deflated PSR={dsr:.2f}")

    # 5) The ONLY verdict that separates a real edge from harvested drift.
    row5 = bench.loc[5]
    real = bool(row5["p_greater"] < 0.05 and row5["excess"] > 0)
    print("\n[EDGE VERDICT] decisive test = excess vs a random day (+5d):")
    print(f"   +5d excess = {row5['excess']:+.3%}   p_greater = {row5['p_greater']:.3f}   -> "
          f"{'REAL dip edge' if real else 'NO dip edge (high Sharpe was just market drift/noise)'}")
    return es, bench, scan


def main():
    print("FALLING-KNIFE — offline validation demo")
    world_a = synthetic_ohlc(seed=1, bounce=0.0)   # no edge
    world_b = synthetic_ohlc(seed=1, bounce=0.6)   # real mean-reversion

    es_a, _, _ = analyse("WORLD A — no dip edge (expect excess ~ 0)", world_a)
    es_b, bench_b, scan_b = analyse("WORLD B — injected mean-reversion (expect positive excess)", world_b)

    # A concrete backtest + cost sweep on World B's classic trigger.
    ret_b = data.daily_returns(world_b)
    events_b = triggers.first_crossings(triggers.close_to_close(ret_b), cooldown=10)
    res = backtest.run(world_b, events_b, exits.ExitRule(max_hold=5), backtest.CostModel())
    print("\n[Backtest World B | T1, hold<=5d]")
    for k in ("n_trades", "cagr", "sharpe", "max_drawdown", "win_rate", "avg_hold_days"):
        print(f"   {k:>16}: {res.stats[k]:.4f}" if isinstance(res.stats[k], float) else f"   {k:>16}: {res.stats[k]}")
    sweep = backtest.cost_sweep(world_b, events_b, exits.ExitRule(max_hold=5))
    print("\n[Cost sweep World B] net CAGR/Sharpe as entry panic-slippage rises:")
    print(sweep.to_string(float_format=lambda v: f"{v:.4f}"))

    # Figures.
    plots.plot_event_study(es_a, title="World A (no edge): path around -3%",
                           path=os.path.join(OUT_DIR, "out_eventstudy_A.png"))
    plots.plot_event_study(es_b, title="World B (real edge): path around -3%",
                           path=os.path.join(OUT_DIR, "out_eventstudy_B.png"))
    plots.plot_equity(res, title="World B backtest (T1, hold<=5d)",
                      path=os.path.join(OUT_DIR, "out_equity_B.png"))
    plots.plot_cost_sweep(sweep, path=os.path.join(OUT_DIR, "out_costsweep_B.png"))
    plots.plot_family_heatmap(scan_b, metric="sharpe",
                              path=os.path.join(OUT_DIR, "out_family_B.png"))
    print("\nSaved figures: out_eventstudy_A/B.png, out_equity_B.png, "
          "out_costsweep_B.png, out_family_B.png")
    print("\nDone. If A reads flat and B reads as a real edge, the toolchain works.")


if __name__ == "__main__":
    main()
