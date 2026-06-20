"""Offline positive-control / null demo for Study 340 (Bank-Loans).

    python examples/run_synthetic_demo.py

Shows the harness reads the loan leg correctly under both regimes:

- positive control (``dur`` low, ``credit_beta`` high): BKLN shrugs off the rate spike
  (the "rate protection" is real) yet falls in the equity crash (credit risk is the catch);
- null (``dur`` high, ``credit_beta`` low): a duration bond — falls in the rate spike,
  rallies in the crash.

No network; deterministic.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bank_loans import data, strategy  # noqa: E402


def _report(label, frame, truth):
    rets = strategy.to_returns(frame)
    b_tlt = strategy.ols_beta(rets["BKLN"], rets["TLT"])
    d_spy = strategy.downside_beta(rets["BKLN"], rets["SPY"])
    print(f"\n[{label}]  dur={truth['dur']:.2f}  credit_beta={truth['credit_beta']:.2f}")
    print(f"  beta BKLN~TLT (rate)        = {b_tlt:+.3f}")
    print(f"  downside beta BKLN~SPY      = {d_spy:+.3f}")
    rate_eps = strategy.asset_drawdowns(rets[["TLT", "BKLN", "IEF", "SPY"]], "TLT", thresh=-0.05)
    if rate_eps:
        e = max(rate_eps, key=lambda z: -z["driver_loss"])
        print(f"  worst rate selloff: TLT {e['driver_loss']*100:+.1f}%  "
              f"BKLN {e['others']['BKLN']*100:+.1f}%")
    crash_eps = strategy.asset_drawdowns(rets[["SPY", "BKLN", "TLT", "IEF"]], "SPY", thresh=-0.10)
    if crash_eps:
        e = max(crash_eps, key=lambda z: -z["driver_loss"])
        print(f"  worst equity crash: SPY {e['driver_loss']*100:+.1f}%  "
              f"BKLN {e['others']['BKLN']*100:+.1f}%  TLT {e['others']['TLT']*100:+.1f}%")


def main() -> int:
    pos = data.synthetic_four_asset(dur=0.05, credit_beta=0.65, seed=340)
    null = data.synthetic_four_asset(dur=12.0, credit_beta=0.05, seed=340)
    _report("POSITIVE CONTROL (floating, credit-loaded)", *pos)
    _report("NULL (duration bond)", *null)
    return 0


if __name__ == "__main__":
    sys.exit(main())
