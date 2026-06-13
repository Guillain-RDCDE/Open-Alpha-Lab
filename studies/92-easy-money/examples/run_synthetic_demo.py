"""Offline positive/negative control for Study 92 (Easy-Money) — no network.

Shows the harness behaves correctly on tapes where we *know* the answer:

- ``carry = 0`` (a zero-drift random walk, no jumps): there is no contango bleed to
  harvest and no spike to fear, so shorting it must NOT earn a risk-adjusted edge —
  Sharpe near zero, no fat left tail.
- ``carry > 0`` (steady decay + rare upward jump-spikes): the short-vol carry should
  now bank a steady drip with a *positive* Sharpe AND carry a clear fat left tail
  (negative skew, high kurtosis, a deep drawdown) — the planted "earn then blow up"
  signature the real tape also shows.

    python examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from easy_money import data, strategy  # noqa: E402


def _run(carry: float) -> dict:
    frame, truth = data.synthetic_vol_etp(n_days=4000, carry=carry, seed=92)
    close = frame["close"]
    bt = strategy.short_vol_backtest(close, leverage=1.0, cost_bps=5.0,
                                     borrow_bps_annual=300.0)
    return {"truth": truth, "bt": bt}


def main() -> None:
    for carry, label in [(0.0, "NULL  (no carry, no jumps)"),
                         (0.003, "PLANTED carry + jump-spikes")]:
        r = _run(carry)
        b = r["bt"]
        print(f"\n=== {label} | n_jumps={r['truth']['n_jumps']} ===")
        print(f"  short-vol: Sharpe {b['sharpe']:+.3f}  CAGR {b['cagr']*100:+6.2f}%  "
              f"maxDD {b['max_dd']*100:6.1f}%")
        print(f"             skew {b['skew']:+.2f}  excess-kurt {b['excess_kurt']:+.1f}  "
              f"worst-day {b['worst_day']*100:+.1f}%")
        if carry > 0:
            verdict = "earns a drip AND has a fat left tail" if (
                b["sharpe"] > 0.3 and b["skew"] < -0.5
            ) else "FAILED to reproduce the signature"
        else:
            verdict = "no edge, no fat tail (as expected)" if (
                abs(b["sharpe"]) < 0.3 and b["skew"] > -0.5
            ) else "UNEXPECTED edge/tail on the null"
        print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
