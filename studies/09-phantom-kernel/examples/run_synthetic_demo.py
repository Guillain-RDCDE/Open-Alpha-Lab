"""A 30-second tour of Study 09 — the kernel falsification and the tournament, printed.

    python examples/run_synthetic_demo.py
"""

import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from phantom_kernel import experiments as ex, sim


def main():
    print("1) Machinery sanity (World A): the estimator must recover the planted k.")
    print("  ", ex.estimator_recovery(seed=0))

    print("\n2) H1 — is the kernel exponential? (exp vs power-law goodness-of-fit)")
    print(ex.kernel_gof_table(seed=0).to_string())

    print("\n3) H1 — a static k while the truth drifts 4x mis-prices the 'optimal' spread:")
    k = ex.k_instability(seed=0)
    print("   max |spread % error| =", k["max_abs_spread_pct_error"], "%")

    print("\n4) H2 — the tournament (P&L Sharpe), benign vs hostile market:")
    for w in (sim.WORLD_A, sim.WORLD_B):
        t = ex.tournament(w, seed=0)
        print(f"   World {w.name}:")
        print(t[["pnl_sharpe", "inv_std", "n_adverse"]].to_string())


if __name__ == "__main__":
    main()
