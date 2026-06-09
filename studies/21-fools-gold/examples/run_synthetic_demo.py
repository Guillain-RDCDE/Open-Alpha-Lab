"""Offline demo — prove the machinery on a synthetic tape with a *known* trend.

No network, deterministic. Runs the teardown on a **trend** tape (persistent drift, so a crossover can
ride runs) and the **null** (driftless random walk). The contrast is the point: a crossover that fires
on the trend tape and goes quiet on the null shows the code measures the effect, not itself — and the
null also illustrates how a single random walk can *spuriously* trend, the data-mining trap beat 7
hunts on the real tape.

    python examples/run_synthetic_demo.py
"""

import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from fools_gold import data, cross, strategy, decompose, extension


def run(label, ts):
    close, _ = data.synthetic_prices(trend_strength=ts, seed=21)
    sv = cross.signal_value(close)
    cmp = strategy.compare(close, cost_bps=2.0)
    sp = decompose.spread_tstat(close)
    vb = decompose.vs_buy_hold(close, cost_bps=2.0)
    pg = extension.param_grid(close, cost_bps=2.0)
    print(f"\n=== {label} (trend_strength={ts}) ===")
    print(f"  golden−death spread {sv['spread_ann_pct']:+.1f}%/yr (HAC t {sp['t_stat']:+.1f})")
    print(f"  timing Sharpe {cmp['timing']['sharpe']:+.2f} vs buy-&-hold {cmp['buy_hold']['sharpe']:+.2f} "
          f"(gain {cmp['sharpe_gain']:+.2f}), avg exposure {cmp['avg_exposure']:.0%}")
    print(f"  alpha vs B&H {vb['alpha_ann_pct']:+.1f}% (t{vb['alpha_t']:+.1f}), beta {vb['beta']:.2f}")
    print(f"  parameter grid: {pg['frac_beat_buy_hold']:.0%} of pairs beat B&H, mean gain {pg['mean_gain']:+.2f}")


def main():
    print("Fools-Gold — synthetic proof of the machinery (a real trend vs a driftless null).")
    run("trend", ts=0.0006)
    run("null", ts=0.0)
    print("\nThe real-ETF verdict (it works on the S&P everyone quotes, nowhere else) is in docs/results.md.")


if __name__ == "__main__":
    main()
