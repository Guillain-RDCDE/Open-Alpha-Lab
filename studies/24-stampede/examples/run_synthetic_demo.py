"""Offline demo — prove the machinery on a synthetic momentum panel vs a no-momentum null.

No network, deterministic. On a panel with a baked persistent relative drift, past winners keep winning
and the WML factor earns a large, significant alpha; on the null (no persistence) the same sort earns
nothing. The contrast proves the engine measures momentum, not noise.

    python examples/run_synthetic_demo.py
"""

import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from stampede import data, momentum, strategy, decompose


def run(label, ms):
    panel, _, truth = data.synthetic_panel(mom_strength=ms, seed=24)
    sp = momentum.momentum_spread(panel)
    cmp = strategy.compare(panel, cost_bps=5.0)
    a = decompose.capm_alpha(panel, cost_bps=5.0)
    print(f"\n=== {label} (mom_strength={ms}, momentum={truth.has_momentum}) ===")
    print(f"  winners {sp['winners_ann_pct']:+.1f}%/yr vs losers {sp['losers_ann_pct']:+.1f}%/yr "
          f"(WML spread {sp['wml_ann_pct']:+.1f}%/yr)")
    print(f"  WML net Sharpe {cmp['wml']['sharpe']:+.2f}, CAPM alpha {a['alpha_ann_pct']:+.1f}%/yr "
          f"(HAC t {a['alpha_t']:+.1f}), turnover {cmp['turnover_ann']:.0f}x/yr")


def main():
    print("Stampede — synthetic proof of the machinery (real momentum vs a no-momentum null).")
    run("MOMENTUM", ms=0.0015)
    run("NULL", ms=0.0)
    print("\nMomentum tape: winners out-earn losers at HAC t well past any luck threshold. Null: nothing. "
          "The real S&P 500 verdict (a faint, crash-prone premium) is in docs/results.md.")


if __name__ == "__main__":
    main()
