"""Offline demo — prove the machinery on a synthetic cross-asset world with a known macro premium.

No network, deterministic. On the control, the trend in latent macro state predicts asset returns, so the
macro-momentum book earns a real premium and the inflation-hedge tilt pays — *more* in rising-inflation
regimes than falling ones (the beat-7 split). On the null (macro_strength=0) the assets are pure noise and
both books earn ~0.

    python examples/run_synthetic_demo.py

The real cross-asset/FRED verdict is PENDING a reliable FRED macro fetch (see docs/results.md and
examples/verify.py --fetch).
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from barometer import costs, data, extension, strategy


def run(label, macro_strength):
    r, m, truth = data.synthetic_macro(macro_strength=macro_strength, seed=37)
    cmp = strategy.compare(r, m, cost_bps=0.0)
    print(f"\n=== {label} (macro_strength={macro_strength}, has_macro={truth.has_macro}) ===")
    for kind in ("macro_momentum", "inflation"):
        s = cmp[kind]
        print(f"  {kind:15} Sharpe {s['sharpe']:+.2f}  ann {100*s['ann_return']:+.1f}%  "
              f"maxDD {100*s['max_drawdown']:.0f}%  turnover {s['turnover_ann']:.1f}x/yr")
    return r, m


def main():
    print("Barometer — a synthetic cross-asset world where macro momentum predicts returns, vs a null.")
    r, m = run("MACRO PREMIUM", macro_strength=1.0)
    run("NULL (no macro predictability)", macro_strength=0.0)

    print("\nbreak-even cost (slow book → cost is not the threat):")
    print(f"  macro-momentum {costs.breakeven_cost_bps(r, m, 'macro_momentum'):.0f} bp   "
          f"inflation-hedge {costs.breakeven_cost_bps(r, m, 'inflation'):.0f} bp")

    print("\nbeat-7 regime split — does the inflation hedge pay when inflation is rising?")
    for kind, sp in extension.regime_split_both(r, m, cost_bps=0.0).items():
        rising, falling = sp.loc["rising"], sp.loc["falling"]
        print(f"  {kind:15} rising Sharpe {rising['sharpe']:+.2f} ({rising['ann_return_pct']:+.1f}%/yr)  "
              f"falling {falling['sharpe']:+.2f} ({falling['ann_return_pct']:+.1f}%/yr)")
    print("\n  → the inflation-hedge book earns more in the rising-inflation regime (as designed); the\n"
          "    broader macro-momentum book, which also rides growth, is steadier across both.")
    print("\nThe real cross-asset / FRED verdict is PENDING a reliable FRED macro fetch: examples/verify.py --fetch.")


if __name__ == "__main__":
    main()
