"""Offline, deterministic demo — Study 186 (Morning-Star).

No network.  Builds a synthetic daily tape and shows the study's spine in one
screen: the morning-star and evening-star three-candle reversal patterns only edge
a random-day baseline when the tape actually carries mean-reversion — and on a
martingale (reversion=0) they are fair coins.  Run:

    python studies/186-morning-star/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from morning_star import data, strategy as st  # noqa: E402


def _summarise_all(bars, seed=0):
    """Run the full pattern study on ``bars`` and return pattern→summary."""
    results = st.run_pattern_study(bars, horizons=(1, 5), cost_bps=0.0, seed=seed)
    return {
        name: st.summarize_pattern(df, horizon=1)
        for name, df in results.items()
        if df is not None and len(df) > 0
    }


def main() -> None:
    print("Study 186 - Morning-Star - synthetic positive control\n")
    print(
        f"{'pattern':>14} | {'reversion=0 excess bps':>22} {'t':>6}"
        f" | {'reversion=0.25 excess bps':>25} {'t':>6} | patterns help?"
    )
    print("-" * 88)

    bars_flat, _ = data.synthetic_daily(n_days=1500, reversion=0.00, seed=186)
    bars_rev, _ = data.synthetic_daily(n_days=1500, reversion=0.25, seed=186)

    sums_flat = _summarise_all(bars_flat, seed=7)
    sums_rev = _summarise_all(bars_rev, seed=7)

    any_pattern = False
    for name in st.PATTERN_NAMES:
        sf = sums_flat.get(name, {})
        sr = sums_rev.get(name, {})
        exc_f = sf.get("excess_bps", float("nan"))
        t_f = sf.get("tstat_excess", float("nan"))
        n_f = sf.get("n", 0)
        exc_r = sr.get("excess_bps", float("nan"))
        t_r = sr.get("tstat_excess", float("nan"))
        n_r = sr.get("n", 0)
        helps = "yes" if (n_r > 0 and n_f >= 0 and exc_r > exc_f + 0.5) else "no / too few events"
        print(
            f"{name:>14} | {exc_f:>+22.2f} {t_f:>+6.2f} (n={n_f:3d})"
            f" | {exc_r:>+25.2f} {t_r:>+6.2f} (n={n_r:3d}) | {helps}"
        )
        any_pattern = True

    if not any_pattern:
        print("  (no pattern events on this synthetic tape; patterns are rare by construction)")

    print()
    print("Three-candle reversal patterns are rare by construction. On the real daily tape")
    print("(SPY + S&P 500 names, 2010-2026) the patterns fire ~2-10 times per ticker per year.")
    print("Neither morning-star nor evening-star excess returns clear the Bonferroni bar")
    print(f"(|t| >= 2.50 for {len(st.PATTERN_NAMES)} patterns x 2 horizons = 4 tests).")
    print("See docs/results.md for the full real-tape numbers.")


if __name__ == "__main__":
    main()
