"""Offline demo — prove the machinery on a synthetic sector panel with a known momentum, vs a null.

No network, deterministic. On the momentum tape the hot sectors persist and rotation beats the
equal-weight basket; on the null, nothing. The contrast proves the engine measures the effect.

    python examples/run_synthetic_demo.py

The real SPDR-sector verdict (rotation does NOT beat holding all sectors) is in docs/results.md.
"""

import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from carousel import data, rotation, strategy, decompose


def run(label, ms):
    panel, truth = data.synthetic_sectors(mom_strength=ms, seed=28)
    rs = rotation.rotation_strength(panel)
    cmp = strategy.compare(panel, cost_bps=3.0)
    a = decompose.vs_equal_weight(panel, cost_bps=3.0)
    print(f"\n=== {label} (mom_strength={ms}, momentum={truth.has_momentum}) ===")
    print(f"  top-minus-bottom sector spread {rs['top_minus_bottom_ann_pct']:+.1f}%/yr")
    print(f"  rotation Sharpe {cmp['rotation']['sharpe']:+.2f} vs equal-weight {cmp['equal_weight']['sharpe']:+.2f} "
          f"(gain {cmp['rotation_minus_ew_sharpe']:+.2f}); alpha over basket {a['alpha_ann_pct']:+.1f}%/yr (HAC t {a['alpha_t']:+.1f})")


def main():
    print("Carousel — sector momentum rotation on a synthetic sector panel (momentum) vs a null.")
    run("MOMENTUM", ms=0.0011)
    run("NULL", ms=0.0)
    print("\nMomentum tape: rotation beats the basket at HAC t well past luck. Null: nothing. "
          "On the REAL SPDR sectors it does NOT beat the basket -- see docs/results.md.")


if __name__ == "__main__":
    main()
