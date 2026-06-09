"""Offline demo — prove the trap on a synthetic random walk (and show the honest filter on real reversion).

No network, deterministic. On a **random walk** there is nothing to find, so the two-sided HP-cycle
book's large Sharpe is, by construction, pure look-ahead — and the one-sided (causal) book confirms it
by earning nothing. On a **mean-reverting** tape the one-sided book finally finds a real (small) edge,
proving the causal filter isn't broken — it just doesn't hallucinate.

    python examples/run_synthetic_demo.py
"""

import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from crystal_ball import data, decompose


def run(label, rho):
    close, truth = data.synthetic_prices(revert_rho=rho, seed=22)
    lb = decompose.lookahead_bias(close, cost_bps=1.0, lam=1e6)
    fl = decompose.future_leakage(close, lam=1e6)
    print(f"\n=== {label} (revert_rho={rho}, real reversion={truth.has_reversion}) ===")
    print(f"  two-sided (peeking) Sharpe {lb['two_sided_sharpe']:+.2f} (HAC t {lb['two_sided_t']:+.1f})")
    print(f"  one-sided (causal)  Sharpe {lb['one_sided_sharpe']:+.2f} (HAC t {lb['one_sided_t']:+.1f})")
    print(f"  cycle vs future 5-day return: two-sided {fl[5]['two_sided_corr']:+.2f} | one-sided {fl[5]['one_sided_corr']:+.2f}")


def main():
    print("Crystal-Ball — the HP-filter look-ahead trap, on a random walk and on real reversion.")
    run("RANDOM-WALK null", rho=1.0)
    run("MEAN-REVERT", rho=0.97)
    print("\nTwo-sided looks great on BOTH (it can't tell signal from noise -- it peeks); one-sided is "
          "flat on the null and only finds an edge where one is real. The real-ETF verdict is in docs/results.md.")


if __name__ == "__main__":
    main()
