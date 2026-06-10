"""Offline demo — prove the machinery on a synthetic commodity panel with a known hedging premium.

No network, deterministic. On the premium tape, hedging pressure predicts the next return and the
long-short factor pays; on the null, nothing. The contrast proves the engine measures the effect.

    python examples/run_synthetic_demo.py

The real CFTC-COT + futures verdict (the premium has faded) is in docs/results.md.
"""

import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from hedgers_toll import data, hedging, strategy, decompose


def run(label, hs):
    r, hp, truth = data.synthetic_commodities(hp_strength=hs, seed=29)
    pr = hedging.hedging_premium(r, hp)
    cmp = strategy.compare(r, hp, cost_bps=10.0)
    pt = decompose.premium_tstat(r, hp, cost_bps=10.0)
    print(f"\n=== {label} (hp_strength={hs}, premium={truth.has_premium}) ===")
    print(f"  hedging-pressure IC {pr['mean_ic']:+.3f} (t {pr['ic_t']:+.1f}); "
          f"top-minus-bottom {pr['top_minus_bottom_ann_pct']:+.1f}%/yr")
    print(f"  long-short Sharpe {cmp['long_short']['sharpe']:+.2f}, ann {cmp['long_short']['ann_return']:+.1%} "
          f"(HAC t {pt['t_stat']:+.1f}), turnover {cmp['turnover_ann']:.1f}x/yr")


def main():
    print("Hedgers-Toll — hedging pressure on a synthetic commodity panel (premium) vs a null.")
    run("PREMIUM", hs=0.0045)
    run("NULL", hs=0.0)
    print("\nPremium tape: the factor pays at HAC t well past luck. Null: nothing. On the REAL CFTC-COT + "
          "futures tape the premium has faded -- see docs/results.md.")


if __name__ == "__main__":
    main()
