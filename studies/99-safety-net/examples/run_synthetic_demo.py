"""Offline positive/negative control for Study 99 (Safety-Net) — no network.

Shows the harness behaves correctly on tapes where we *know* the answer, exactly the
split Kaminski & Lo (2014) describe:

- TRENDING tape (sustained bull/bear regimes): declines persist, so a trailing stop
  escapes the downtrend and should BEAT buy-and-hold on a risk-adjusted basis.
- MEAN-REVERTING tape (dips snap back): the stop sells the bottom and misses the
  rebound — it whipsaws and should UNDERPERFORM buy-and-hold.

    python examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from safety_net import data, strategy  # noqa: E402

STOP_PCT = 10.0
COOLDOWN = 21


def _run(close):
    pos = strategy.trailing_stop_position(close, stop_pct=STOP_PCT, cooldown=COOLDOWN)
    stop = strategy.backtest(close, pos, cost_bps=5.0)
    bh = strategy.buy_and_hold(close)
    return stop, bh


def main() -> None:
    trend, _ = data.synthetic_trending(n_days=6000, seed=99)
    mr, _ = data.synthetic_meanrev(n_days=6000, seed=99)

    for label, frame, expect in [
        ("TRENDING  (stops should HELP)", trend, "win"),
        ("MEAN-REV  (stops should HURT)", mr, "lose"),
    ]:
        stop, bh = _run(frame["close"])
        print(f"\n=== {label} ===")
        print(f"  stop  : Sharpe {stop['sharpe']:.3f}  maxDD {stop['max_dd']*100:6.1f}%  CAGR {stop['cagr']*100:6.2f}%")
        print(f"  B&H   : Sharpe {bh['sharpe']:.3f}  maxDD {bh['max_dd']*100:6.1f}%  CAGR {bh['cagr']*100:6.2f}%")
        verdict = "stop wins" if stop["sharpe"] > bh["sharpe"] else "stop loses"
        print(f"  -> {verdict} on Sharpe (expected: {expect})")


if __name__ == "__main__":
    main()
