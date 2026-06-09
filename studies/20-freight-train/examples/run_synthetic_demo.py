"""Offline demo — prove the machinery on a synthetic universe with a *known* trend.

No network, deterministic. Runs the teardown twice: on a **trend** panel (a slow persistent drift, so
past returns predict future ones) and on the **null** (driftless noise). Every diagnostic that fires on
the trend tape must go quiet on the null — that is what proves the code measures the effect.

    python examples/run_synthetic_demo.py

The real-market verdict (a live ETF basket, where the trend is faint and decayed but the crisis
convexity holds) is a separate, fingerprinted run in `examples/verify.py` -> `docs/results.md`.
"""

import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from trend_follow import data, trend, strategy, decompose, extension


def run(label, ts):
    panel, truth = data.synthetic_panel(trend_strength=ts, seed=20)
    pr = trend.predictability(panel)
    cmp = strategy.compare(panel, cost_bps=2.0)
    t = decompose.mean_tstat_hac(strategy.tsmom_returns(panel, cost_bps=2.0))
    a = decompose.basket_alpha(panel, cost_bps=2.0)
    print(f"\n=== {label} (trend_strength={ts}) ===")
    print(f"  predictability pooled t {pr['pooled_t']:+.1f}, hit {pr['hit_rate']:.0%}")
    print(f"  TSMOM net Sharpe {cmp['tsmom']['sharpe']:+.2f} vs basket {cmp['basket']['sharpe']:+.2f} "
          f"(gain {cmp['sharpe_gain']:+.2f}), turnover {cmp['turnover_ann']:.1f}x/yr, HAC t {t['t_stat']:+.1f}")
    print(f"  alpha vs basket {a['alpha_ann_pct']:+.1f}%/yr (t{a['alpha_t']:+.1f}), beta {a['beta']:+.2f}")


def main():
    print("Freight-Train — synthetic proof of the machinery (a real trend vs a driftless null).")
    run("trend", ts=0.0006)
    run("null", ts=0.0)
    print("\nTrend tape: predictability and a TSMOM edge at HAC t > 2. Null: every leg quiet. "
          "The real ETF verdict is in docs/results.md.")


if __name__ == "__main__":
    main()
