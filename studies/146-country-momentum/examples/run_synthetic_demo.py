"""Offline demo — prove the machinery on a synthetic country-momentum panel vs a no-momentum null.

No network, deterministic, exits 0. On a panel with a baked persistent relative drift (past
country winners keep winning), the 12-1 sort earns a positive active return over random rotation
and the equal-weight basket; on the null (no persistent drift) it earns nothing above random.
The contrast proves the engine measures momentum, not noise.

    python examples/run_synthetic_demo.py
"""

import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

import numpy as np

from country_momentum import data, strategy as st


def run(label: str, mom_strength: float, seed: int = 146) -> None:
    panel, truth = data.synthetic_panel(
        n_months=300, mom_strength=mom_strength, seed=seed
    )

    mom_df = st.run_strategy(panel, k=4, cost_bps=0, mode="momentum")
    ew_df  = st.run_strategy(panel, k=4, cost_bps=0, mode="ew")
    rand_df = st.run_strategy(panel, k=4, cost_bps=0, mode="random", seed=0)

    mom_s  = st.summarize(mom_df["r_gross"])
    ew_s   = st.summarize(ew_df["r_gross"])
    rand_s = st.summarize(rand_df["r_gross"])

    active = mom_df["r_gross"] - rand_df["r_gross"]
    active_s = st.summarize(active)

    print(f"\n=== {label} (mom_strength={mom_strength}, has_momentum={truth['has_momentum']}) ===")
    print(
        f"  Momentum top-4:  ann={mom_s['ann_ret']*100:.1f}%  Sharpe={mom_s['sharpe']:+.2f}"
        f"  HAC t={mom_s['tstat']:+.2f}"
    )
    print(
        f"  EW all-country:  ann={ew_s['ann_ret']*100:.1f}%  Sharpe={ew_s['sharpe']:+.2f}"
        f"  HAC t={ew_s['tstat']:+.2f}"
    )
    print(
        f"  Random-rotation: ann={rand_s['ann_ret']*100:.1f}%  Sharpe={rand_s['sharpe']:+.2f}"
        f"  HAC t={rand_s['tstat']:+.2f}"
    )
    print(
        f"  Active (mom-rand): ann={active_s['ann_ret']*100:.1f}%  t={active_s['tstat']:+.2f}"
    )


def main() -> None:
    print("Country-Momentum — synthetic proof of the 12-1 country rotation machinery.")
    print("Planted momentum -> the sort harvests it. Null -> the sort earns nothing.")
    run("MOMENTUM tape", mom_strength=0.04)
    run("NULL tape (no persistent drift)", mom_strength=0.0)
    print(
        "\nKey: with planted momentum the top-4 outperforms random rotation at a clear "
        "edge; on the null, active return vs random is near zero. The real verdict is "
        "in docs/results.md: a weak, cost-sensitive premium that has faded post-2010."
    )


if __name__ == "__main__":
    main()
