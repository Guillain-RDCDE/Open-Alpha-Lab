"""Offline positive/null control for Study 339 (Convertible-Bonds) — no network.

Plants a genuinely *convex* synthetic 'convertible' (gamma > 0) and a *linear* null
(gamma = 0), then shows the harness recovers a positive, significant convexity coefficient on
the former and nothing on the latter. This is a MACHINERY proof — that the quadratic
regression can detect convexity when it is really there — never market evidence for CWB.

    PYTHONIOENCODING=utf-8 python examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from convertible_bonds import data, strategy  # noqa: E402


def _report(label, gamma):
    frame, truth = data.synthetic_convex(n_days=4000, gamma=gamma, seed=339)
    rets = strategy.to_returns(frame)
    reg = strategy.convexity_regression(rets, "CVT", "IDX")
    bg = strategy.bootstrap_gamma(rets, "CVT", "IDX", block=21, n_boot=1000, seed=1)
    cap = strategy.capture_ratios(rets, "CVT", "IDX")
    print(f"\n--- {label}  (planted gamma = {gamma}) ---")
    print(f"  recovered gamma = {reg['gamma']:+.3f}   HAC t = {reg['t_gamma']:+.2f}")
    print(f"  bootstrap gamma CI [{bg['ci95'][0]:+.3f}, {bg['ci95'][1]:+.3f}]  "
          f"P(gamma>0) = {bg['frac_pos']*100:.0f}%")
    print(f"  up-capture {cap['up_capture']:.3f}  down-capture {cap['down_capture']:.3f}  "
          f"asym {cap['asymmetry']:+.3f}")


def main() -> None:
    print("Study 339 — synthetic convexity control (offline, deterministic)")
    _report("POSITIVE CONTROL (convex)", gamma=0.6)
    _report("NULL (linear blend)", gamma=0.0)
    print("\nThe harness banks planted convexity and finds none in the null. "
          "A synthetic Sharpe/gamma can NEVER back a Signal stamp — see METHODOLOGY.md.")


if __name__ == "__main__":
    main()
