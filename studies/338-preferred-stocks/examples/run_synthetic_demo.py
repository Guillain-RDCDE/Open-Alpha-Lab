"""Offline positive/null control for Study 338 (Preferred-Stocks).

No network. Plants a preferred leg that is either equity-in-disguise (pref_beta high,
the positive control) or genuinely bond-like (pref_beta low, the null), then shows the
harness reads each correctly via the downside beta to stocks vs to bonds.

    python examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preferred_stocks import data, strategy  # noqa: E402


def _report(label, pref_beta):
    frame, truth = data.synthetic_three_asset(pref_beta=pref_beta, seed=338)
    rets = strategy.to_returns(frame)
    dspy = strategy.downside_beta(rets["PFF"], rets["SPY"])
    dbnd = strategy.downside_beta(rets["PFF"], rets["BND"])
    crash = strategy.window_return(
        rets[["SPY", "PFF", "BND"]],
        str(truth["crash_peak"].date()), str(truth["crash_trough"].date()),
    )
    print(f"\n[{label}]  pref_beta={pref_beta}")
    print(f"  downside beta to SPY = {dspy:.3f}   downside beta to BND = {dbnd:.3f}")
    print(f"  crash-window return:  SPY {crash['SPY']*100:+.1f}%  "
          f"PFF {crash['PFF']*100:+.1f}%  BND {crash['BND']*100:+.1f}%")


def main() -> int:
    print("Synthetic control — can the harness tell equity-in-disguise from bond-like?")
    _report("POSITIVE CONTROL (equity-in-disguise)", 0.90)
    _report("NULL (genuinely bond-like)", 0.05)
    return 0


if __name__ == "__main__":
    sys.exit(main())
