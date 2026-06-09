"""Offline demo — prove the machinery on a synthetic cointegrated pair vs a spurious one.

No network, deterministic. On a genuinely **cointegrated** pair the spread half-life is short, the trade
makes money and survives out of sample; on a **spurious** pair (two independent walks) the half-life is
long and the second half breaks. The contrast proves the engine measures the relationship, not noise.

    python examples/run_synthetic_demo.py
"""

import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from broken_tether import data, spread, strategy, decompose


def run(label, rho):
    px, truth = data.synthetic_pair(revert_rho=rho, seed=23)
    st = spread.stationarity(px["A"], px["B"])
    cmp = strategy.compare(px["A"], px["B"], cost_bps=2.0)
    oos = decompose.in_sample_vs_oos(px["A"], px["B"], cost_bps=2.0)
    print(f"\n=== {label} (revert_rho={rho}, cointegrated={truth.is_cointegrated}) ===")
    print(f"  spread half-life {st['half_life_days']:.0f} days (reverting={st['is_reverting']})")
    print(f"  pairs Sharpe {cmp['sharpe']:+.2f} ({cmp['trades']} trades)")
    print(f"  in-sample {oos['first_half_sharpe']:+.2f} -> out-of-sample {oos['second_half_sharpe']:+.2f} "
          f"(survives={oos['survives_oos']})")


def main():
    print("Broken-Tether — a real cointegrated pair vs a spurious one (two independent walks).")
    run("COINTEGRATED", rho=0.93)
    run("SPURIOUS null", rho=1.0)
    spur = decompose.spurious_pairs(n_series=20, seed=1)
    print(f"\nselection trap: {spur['false_positive_rate']:.0%} of independent random-walk pairs LOOK "
          f"cointegrated by chance. The real-ETF verdict is in docs/results.md.")


if __name__ == "__main__":
    main()
