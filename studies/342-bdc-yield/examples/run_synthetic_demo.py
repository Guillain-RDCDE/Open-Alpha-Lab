"""Offline positive-control / null demo for Study 342 (BDC-Yield).

    python examples/run_synthetic_demo.py

No network. Builds two synthetic three-asset worlds and shows the harness reads the BDC
leg as levered equity when it IS (the positive control), and as bond-like when it isn't
(the null). A pipeline that can't tell the two apart proves nothing by finding nothing.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bdc_yield import data, strategy  # noqa: E402


def _report(label, frame, truth):
    rets = strategy.to_returns(frame)
    dspy = strategy.downside_beta(rets["BIZD"], rets["SPY"])
    dbnd = strategy.downside_beta(rets["BIZD"], rets["BND"])
    print(f"\n[{label}]  bdc_beta={truth['bdc_beta']:.2f}  "
          f"corr(BIZD,SPY)={truth['corr_bizd_stk']:+.2f}  "
          f"corr(BIZD,BND)={truth['corr_bizd_bnd']:+.2f}")
    print(f"  downside beta to SPY = {dspy:.2f}   downside beta to BND = {dbnd:.2f}")
    print(f"  fingerprint = {data.fingerprint(frame)}")


def main() -> int:
    pos = data.synthetic_three_asset(n_days=3300, bdc_beta=1.15, seed=342)
    null = data.synthetic_three_asset(n_days=3300, bdc_beta=0.05, seed=342)
    _report("POSITIVE CONTROL — levered private-credit equity", *pos)
    _report("NULL — genuinely bond-like income", *null)
    print("\nHarness check: the positive control's downside beta to stocks must dominate "
          "the null's. If it doesn't, the pipeline is blind.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
