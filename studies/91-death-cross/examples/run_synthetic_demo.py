"""Offline positive/negative control for Study 91 (Death-Cross) — no network.

Shows the harness behaves correctly on tapes where we *know* the answer:

- ``regime = 0`` (a constant-drift random walk): there is no bear market to dodge, so
  the timer must NOT beat buy-and-hold on a risk-adjusted basis — it only sheds upside.
- ``regime > 0`` (sticky bull/bear regimes): the death-cross filter should now bank a
  real edge — a higher Sharpe and a smaller drawdown than buy-and-hold.

    python examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from death_cross import data, strategy  # noqa: E402


def _run(regime: float) -> dict:
    frame, truth = data.synthetic_daily(n_days=6000, regime=regime, seed=91)
    close = frame["close"]
    pos = strategy.crossover_position(close, 50, 200, lag=1)
    timer = strategy.backtest(close, pos, cost_bps=5.0)
    bh = strategy.buy_and_hold(close)
    return {"truth": truth, "timer": timer, "bh": bh}


def main() -> None:
    for regime, label in [(0.0, "NULL  (random walk)"), (1.0, "PLANTED regimes")]:
        r = _run(regime)
        t, b = r["timer"], r["bh"]
        print(f"\n=== {label} | frac_bear={r['truth']['frac_bear']:.2f} ===")
        print(f"  timer : Sharpe {t['sharpe']:.3f}  maxDD {t['max_dd']*100:6.1f}%  CAGR {t['cagr']*100:6.2f}%")
        print(f"  B&H   : Sharpe {b['sharpe']:.3f}  maxDD {b['max_dd']*100:6.1f}%  CAGR {b['cagr']*100:6.2f}%")
        verdict = "timer wins" if t["sharpe"] > b["sharpe"] else "timer loses"
        print(f"  -> {verdict} on Sharpe (expected: "
              f"{'win' if regime > 0 else 'lose / tie'})")


if __name__ == "__main__":
    main()
