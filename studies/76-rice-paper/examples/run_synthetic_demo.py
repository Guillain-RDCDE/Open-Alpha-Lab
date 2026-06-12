"""Offline, deterministic demo — Study 76 (Rice-Paper).

No network.  Builds a synthetic daily tape and shows the study's spine in one
screen: Japanese candlestick reversal patterns only edge a random-baseline when
the tape actually carries mean-reversion — and on a martingale (reversion=0)
they are fair coins.  Run:

    python studies/76-rice-paper/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rice_paper import data, strategy as st  # noqa: E402


def _summarise_all(bars, seed=0):
    """Run the full pattern study on ``bars`` and return pattern→summary."""
    results = st.run_pattern_study(bars, horizons=(1, 5), cost_bps=0.0, seed=seed)
    return {
        name: st.summarize_pattern(df, horizon=1)
        for name, df in results.items()
        if df is not None and len(df) > 0
    }


def main() -> None:
    print("Study 76 — Rice-Paper — synthetic positive control\n")
    print(
        f"{'pattern':>20} | {'reversion=0 excess bps':>22} {'t':>6}"
        f" | {'reversion=0.25 excess bps':>25} {'t':>6} | patterns help?"
    )
    print("-" * 96)

    bars_flat, _ = data.synthetic_daily(n_days=800, reversion=0.00, seed=76)
    bars_rev, _ = data.synthetic_daily(n_days=800, reversion=0.25, seed=76)

    sums_flat = _summarise_all(bars_flat, seed=7)
    sums_rev = _summarise_all(bars_rev, seed=7)

    for name in st.PATTERN_NAMES:
        sf = sums_flat.get(name, {})
        sr = sums_rev.get(name, {})
        exc_f = sf.get("excess_bps", float("nan"))
        t_f = sf.get("tstat_excess", float("nan"))
        exc_r = sr.get("excess_bps", float("nan"))
        t_r = sr.get("tstat_excess", float("nan"))
        helps = "yes" if exc_r > exc_f + 0.5 else "no"
        print(
            f"{name:>20} | {exc_f:>+22.2f} {t_f:>+6.2f}"
            f" | {exc_r:>+25.2f} {t_r:>+6.2f} | {helps}"
        )

    print()
    print("Reversal patterns are faithful mean-reversion detectors: they edge the")
    print("random baseline when mean-reversion is planted, and on a martingale they")
    print("are fair coins.  On the real daily tape there is no robust mean-reversion")
    print("for them to harvest — see docs/results.md.")


if __name__ == "__main__":
    main()
