"""Offline positive/negative control for Study 97 (Balancing-Act) — no network.

Shows the harness behaves correctly on tapes where we *know* the answer. The whole point
of 60/40 is **diversification**, which only exists when the two legs are not perfectly
correlated — so the controls toggle the correlation knob:

- ``corr < 0`` (negatively-correlated legs, matched Sharpes): a fixed-weight blend cuts
  volatility more than return, so its Sharpe must BEAT the better single leg. This is the
  positive control — diversification is real, the harness should bank it.
- ``corr > 0`` (positively-correlated legs): blending earns no diversification, so the
  blend's Sharpe must sit BETWEEN the two legs, never above the best. This is the null.

    PYTHONIOENCODING=utf-8 python examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from balancing_act import data, strategy  # noqa: E402


def _run(corr: float) -> dict:
    frame, truth = data.synthetic_two_asset(n_days=6000, corr=corr, seed=97)
    rets = strategy.to_returns(frame)
    blend = strategy.stats(strategy.rebalanced_blend(
        rets, {"STK": 0.60, "BND": 0.40}, rebalance="annual"))
    stk = strategy.stats(strategy.single_asset(rets, "STK"))
    bnd = strategy.stats(strategy.single_asset(rets, "BND"))
    return {"truth": truth, "blend": blend, "stk": stk, "bnd": bnd}


def main() -> None:
    for corr, label in [(-0.40, "DIVERSIFYING (neg corr)"),
                        (+0.85, "REDUNDANT (pos corr)")]:
        r = _run(corr)
        bl, st, bn = r["blend"], r["stk"], r["bnd"]
        best_leg = max(st["sharpe"], bn["sharpe"])
        print(f"\n=== {label} | realised corr={r['truth']['realised_corr']:+.2f} ===")
        print(f"  60/40 blend : Sharpe {bl['sharpe']:.3f}  vol {bl['vol']*100:5.1f}%  CAGR {bl['cagr']*100:6.2f}%")
        print(f"  STK (100%)  : Sharpe {st['sharpe']:.3f}  vol {st['vol']*100:5.1f}%")
        print(f"  BND (100%)  : Sharpe {bn['sharpe']:.3f}  vol {bn['vol']*100:5.1f}%")
        wins = bl["sharpe"] > best_leg
        print(f"  -> blend Sharpe {'BEATS' if wins else 'does NOT beat'} the best single leg "
              f"(expected: {'beats' if corr < 0 else 'sits between'})")


if __name__ == "__main__":
    main()
