"""Offline demo — prove the machinery on a synthetic panel where momentum lives in the residual.

No network, deterministic. On the momentum tape, residual winners keep winning and the residual-WML
factor earns a large, significant alpha; on the null, nothing. The contrast proves the residualiser and
the momentum engine measure the effect, not noise.

    python examples/run_synthetic_demo.py
"""

import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from clean_slate import data, momentum, strategy, decompose


def run(label, ms):
    panel, market, truth = data.synthetic_panel(mom_strength=ms, seed=25)
    sp = momentum.momentum_spread(panel, market, residual=True)
    cmp = strategy.compare(panel, market, cost_bps=5.0)
    a = decompose.capm_alpha(panel, market, cost_bps=5.0)
    print(f"\n=== {label} (mom_strength={ms}, momentum={truth.has_momentum}) ===")
    print(f"  residual-WML spread {sp['wml_ann_pct']:+.1f}%/yr | residual Sharpe {cmp['residual']['sharpe']:+.2f} "
          f"(total {cmp['total']['sharpe']:+.2f}) | CAPM alpha {a['alpha_ann_pct']:+.1f}%/yr (HAC t {a['alpha_t']:+.1f})")


def main():
    print("Clean-Slate — residual momentum on a synthetic panel (momentum in the residual) vs a null.")
    run("MOMENTUM", ms=0.0016)
    run("NULL", ms=0.0)
    print("\nMomentum tape: residual winners out-earn losers at HAC t well past luck. Null: nothing. "
          "The real S&P 500 verdict (cleaner than total momentum, still faint) is in docs/results.md.")


if __name__ == "__main__":
    main()
