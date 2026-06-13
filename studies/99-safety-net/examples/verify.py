"""Headline run for Study 99 (Safety-Net) on the real total-return tape.

Runs a trailing stop-loss (exit when the close falls X% below its running peak,
re-enter after a fixed cooldown) on SPY (total return) across the X-sweep
(5/10/15/20%), against buy-and-hold and a matched random-exit control, prints the
numbers that the README front-card and the notebooks quote, and stamps the as-of date
+ content fingerprint. Re-run to reproduce; match the fingerprint to confirm you hold
the same tape.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from safety_net import data, strategy  # noqa: E402

AS_OF = "2026-06-12"
COST_BPS = 5.0
COOLDOWN = 21
STOPS = (5.0, 10.0, 15.0, 20.0)
HEADLINE_X = 10.0
N_SEEDS = 200


def main() -> None:
    frame = data.load_real("SPY", mode="total_return")
    frame = frame.loc[:AS_OF]
    close = frame["close"]
    fp = data.fingerprint(frame)
    print(f"[data] SPY total-return: {len(close):,} rows  "
          f"{close.index[0].date()} -> {close.index[-1].date()}  "
          f"as-of {AS_OF}  fingerprint={fp}")

    bh = strategy.buy_and_hold(close)

    def row(name, d, extra=""):
        print(f"  {name:<26} CAGR {d['cagr']*100:6.2f}%  vol {d['vol']*100:5.1f}%  "
              f"Sharpe {d['sharpe']:.3f}  maxDD {d['max_dd']*100:6.1f}%  "
              f"TiM {d['time_in_market']*100:4.0f}%  switches {d['switches']:>3}{extra}")

    print(f"\n--- Real tape (SPY total return), cost = {COST_BPS:.0f} bps/switch, "
          f"cooldown = {COOLDOWN}d ---")
    row("Buy & hold", bh)
    print()

    # X-sweep + matched random-exit control at each X.
    for x in STOPS:
        pos = strategy.trailing_stop_position(close, stop_pct=x, cooldown=COOLDOWN)
        stop = strategy.backtest(close, pos, cost_bps=COST_BPS)
        rand_sharpes, beats = [], 0
        for s in range(N_SEEDS):
            rp = strategy.matched_random_position(pos, seed=s)
            rb = strategy.backtest(close, rp, cost_bps=COST_BPS)
            rand_sharpes.append(rb["sharpe"])
            if stop["sharpe"] > rb["sharpe"]:
                beats += 1
        diff = (stop["net"] - bh["net"]).to_numpy()
        t_vs_bh = strategy.hac_tstat(diff)
        row(f"Trailing stop {x:.0f}%", stop)
        print(f"      vs random-exit coin (matched, {N_SEEDS} seeds): "
              f"mean Sharpe {np.mean(rand_sharpes):.3f}  "
              f"stop beats {beats}/{N_SEEDS} ({beats/N_SEEDS*100:.0f}%)  | "
              f"HAC t(stop-BH) = {t_vs_bh:+.2f}  | "
              f"CAGR gap {(stop['cagr']-bh['cagr'])*100:+.2f} pts  | "
              f"DD cut {(stop['max_dd']-bh['max_dd'])*100:+.1f} pts")


if __name__ == "__main__":
    main()
