"""Offline, deterministic demo — Study 187 (Three-Soldiers).

No network.  Builds a deterministic synthetic daily tape and shows the study's spine:
Three White Soldiers and Three Black Crows only beat a random-day baseline when the tape
carries real short-term momentum — on a martingale they are a fair coin.  Run:

    python studies/187-three-soldiers/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from three_soldiers import data, strategy as st  # noqa: E402


def _edge(bars, name, horizon=1):
    """Compute excess return (signal − random baseline, bps) for one pattern."""
    result = st.run_pattern_study(bars, horizons=(horizon,), cost_bps=0, seed=42)
    if name not in result:
        return float("nan"), 0, float("nan"), float("nan")
    df = result[name]
    s = st.summarize_pattern(df, horizon=horizon)
    return s.get("excess_bps", float("nan")), s.get("n", 0), s.get("tstat_excess", float("nan")), s.get("mean_bps", float("nan"))


def main() -> None:
    print("Study 187 — Three-Soldiers — synthetic positive control\n")
    print(
        f"{'planted momentum':>18} | {'pattern':>22} | {'n':>5} | "
        f"{'signal bps':>10} | {'excess bps':>10} | {'exc t':>7} | beats baseline?"
    )
    print("-" * 95)

    for mom in (0.00, 0.15, 0.30, -0.15):
        bars, _ = data.synthetic_daily(n_days=1000, momentum=mom, seed=187)
        for name in st.PATTERN_NAMES:
            exc, n, t_exc, sig_bps = _edge(bars, name, horizon=1)
            if n == 0:
                beats = "n/a (no triggers)"
            elif not (exc is not None and not (exc != exc)):  # nan check
                beats = "n/a"
            else:
                beats = "yes" if exc > 0.5 else "no"
            label = name.replace("_", " ")
            print(
                f"{mom:>18.2f} | {label:>22} | {n:>5} | "
                f"{sig_bps:>+10.2f} | {exc:>+10.2f} | {t_exc:>+7.2f} | {beats}"
            )

    print()
    print("The patterns are faithful continuation detectors: their advantage over the")
    print("random baseline rises with planted momentum, and vanishes on a martingale.")
    print("On the real daily tape there is little to find — see docs/results.md.")


if __name__ == "__main__":
    main()
