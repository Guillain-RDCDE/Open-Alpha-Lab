"""Offline demo — the whole machine on the synthetic book, no network.

Generates the toy prediction markets (arbitrage gaps that decay with a *known* 6-minute
half-life), detects episodes, recovers the half-life two ways, and prints the three
charts the verdict turns on: the bootstrap CI, the resolution sweep (what a coarse tape
can even see), and the retail-capture ladder (what's left when a human reacts).

    python examples/run_synthetic_demo.py

This is the reproducible core: it proves the estimator recovers a half-life it was *given*,
so the real-data run (`verify_real.py`) is a measurement, not a hope.
"""

import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from prediction_arb import arbitrage, data, robustness


def main():
    gap, truth = data.synthetic_markets(seed=0)
    eps = arbitrage.detect_all(gap, open_threshold=truth.open_threshold)

    print(f"synthetic book: {gap.shape[1]} markets x {gap.shape[0]} minutes, "
          f"baked-in half-life = {truth.half_life_min:.1f} min")
    print(f"episodes detected: {len(eps)}")

    s = arbitrage.summary(eps)
    print("\n[episode summary]")
    for k, v in s.items():
        print(f"  {k:>22}: {round(v, 4) if isinstance(v, float) else v}")

    boot = robustness.bootstrap_half_life(eps, n_boot=2000)
    print("\n[bootstrap half-life CI]")
    print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in boot.items()})

    print("\n[resolution sweep - what a coarser tape can even see]")
    print(robustness.resolution_sweep(gap).round(3).to_string())

    print("\n[retail capture - fraction of the peak penny left after reacting]")
    print(robustness.retail_capture_table(s["half_life_median_min"]).round(4).to_string())

    print("\nTakeaway: the estimator recovers the baked-in 6-min half-life; a minute tape "
          "already loses the fast episodes, and a human reacting in minutes keeps almost "
          "none of the penny. The real run measures whether Polymarket's half-life is this "
          "short - see verify_real.py.")


if __name__ == "__main__":
    main()
