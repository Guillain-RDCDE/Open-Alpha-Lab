"""Headline run for Study 91 (Death-Cross) on the real total-return tape.

Runs the SMA(50/200) crossover timer against buy-and-hold and a matched random-timing
control on SPY (total return), prints the numbers that the README front-card and the
notebooks quote, and stamps the as-of date + content fingerprint. Re-run to reproduce;
match the fingerprint to confirm you hold the same tape.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from death_cross import data, strategy  # noqa: E402

AS_OF = "2026-06-12"
COST_BPS = 5.0


def main() -> None:
    frame = data.load_real("SPY", mode="total_return")
    frame = frame.loc[:AS_OF]
    close = frame["close"]
    fp = data.fingerprint(frame)
    print(f"[data] SPY total-return: {len(close):,} rows  "
          f"{close.index[0].date()} -> {close.index[-1].date()}  "
          f"as-of {AS_OF}  fingerprint={fp}")

    pos = strategy.crossover_position(close, 50, 200, lag=1)
    timer = strategy.backtest(close, pos, cost_bps=COST_BPS)
    bh = strategy.buy_and_hold(close)

    # Matched random-timing controls — average over many seeds for a stable comparison.
    rand_sharpes, rand_cagrs, rand_dds, beats = [], [], [], 0
    n_seeds = 200
    for s in range(n_seeds):
        rp = strategy.matched_random_position(pos, seed=s)
        rb = strategy.backtest(close, rp, cost_bps=COST_BPS)
        rand_sharpes.append(rb["sharpe"]); rand_cagrs.append(rb["cagr"]); rand_dds.append(rb["max_dd"])
        if timer["sharpe"] > rb["sharpe"]:
            beats += 1

    # HAC t-stat on the daily return difference (timer minus buy-and-hold).
    diff = (timer["net"] - bh["net"]).to_numpy()
    t_vs_bh = strategy.hac_tstat(diff)

    def row(name, d):
        print(f"  {name:<22} CAGR {d['cagr']*100:6.2f}%  vol {d['vol']*100:5.1f}%  "
              f"Sharpe {d['sharpe']:.3f}  maxDD {d['max_dd']*100:6.1f}%  "
              f"TiM {d['time_in_market']*100:4.0f}%  switches {d['switches']}")

    print("\n--- Real tape (SPY total return), cost = 5 bps/switch ---")
    row("Death-cross timer", timer)
    row("Buy & hold", bh)
    print(f"  Random-timing coin     CAGR {np.mean(rand_cagrs)*100:6.2f}%  "
          f"Sharpe {np.mean(rand_sharpes):.3f}  maxDD {np.mean(rand_dds)*100:6.1f}%  "
          f"(matched exposure, {n_seeds} seeds)")
    print(f"\n  Timer Sharpe - BH Sharpe   = {timer['sharpe']-bh['sharpe']:+.3f}")
    print(f"  Timer beats random coin    : {beats}/{n_seeds} seeds "
          f"({beats/n_seeds*100:.0f}%) on Sharpe")
    print(f"  HAC t (timer-BH daily ret) = {t_vs_bh:+.2f}")
    print(f"  Drawdown cut vs BH         = {(timer['max_dd']-bh['max_dd'])*100:+.1f} pts")


if __name__ == "__main__":
    main()
