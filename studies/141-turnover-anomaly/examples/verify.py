"""Real-tape verification — Study 141 (Turnover-Anomaly). Regenerates docs/results.md numbers.

Loads the EDGAR shares-outstanding cache and the study-local annual-volume cache,
sorts S&P 500 names into turnover quintiles each year, computes next-year returns,
and runs the quintile-spread vs shuffled-label test.

    python studies/141-turnover-anomaly/examples/verify.py            # cache-only
    python studies/141-turnover-anomaly/examples/verify.py --fetch    # refresh volume cache
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from turnover_anomaly import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    print("Study 141 — Turnover-Anomaly — real tape verification")
    print("Datar-Naik-Radcliffe (1998): high-turnover stocks should underperform\n")

    sig, fwd = data.fetch_panel(fetch=fetch)
    if sig.empty:
        print("ERROR: No data. Run with --fetch once to populate the volume cache.")
        sys.exit(1)

    # Restrict to signal years 2008-2024 (2007 has low coverage; 2025 → 2026 fwd is partial)
    sig = sig.loc[2008:2024]
    fwd = fwd.loc[2008:2024]

    print(f"Signal years: {sig.index.min()}–{sig.index.max()}  "
          f"({len(sig)} years, {sig.notna().sum(axis=1).mean():.0f} avg tickers/year)")
    print(f"Signal fingerprint: {data.fingerprint(sig)}")
    print(f"Fwd-ret fingerprint: {data.fingerprint(fwd)}")
    print()

    res = st.quintile_returns(sig, fwd)

    print("=== Quintile mean annual returns ===")
    for q in range(1, 6):
        col = f"q{q}"
        qmean = float(res[col].mean())
        label = "low-turnover" if q == 1 else ("high-turnover" if q == 5 else f"Q{q}")
        print(f"  Q{q} ({label:>15s}): {qmean:+.4f}")
    print(f"  Market (EW all)  : {float(res['market'].mean()):+.4f}")
    print()

    print("=== Hedge (Q1 low-turnover minus market) ===")
    hedge = st.summarize(res["hedge"])
    print(f"  n={hedge['n']} years  mean={hedge['mean']:+.4f}  vol={hedge['vol']:.4f}  "
          f"Sharpe={hedge['sharpe']:+.4f}  HAC t={hedge['tstat']:+.4f}  hit={hedge['hit_rate']:.0%}")
    print()

    print("=== Spread (Q1 low-turnover minus Q5 high-turnover) ===")
    spread = st.summarize(res["spread"])
    print(f"  n={spread['n']} years  mean={spread['mean']:+.4f}  vol={spread['vol']:.4f}  "
          f"Sharpe={spread['sharpe']:+.4f}  HAC t={spread['tstat']:+.4f}  hit={spread['hit_rate']:.0%}")
    print()

    print("=== Shuffled-label control (500 draws) ===")
    ctrl = st.shuffled_control(sig, fwd, n_shuffles=500, seed=141)
    shuffle_mean = float(ctrl.mean(axis=None).mean())
    shuffle_std = float(ctrl.mean(axis=1).std())
    print(f"  Shuffle Q1-market: mean={shuffle_mean:+.4f}  std(across shuffles)={shuffle_std:.4f}")
    print(f"  Real Q1-market:    mean={hedge['mean']:+.4f}")
    print(f"  Real beats shuffle? {'NO — real is WORSE than shuffle' if hedge['mean'] < shuffle_mean else 'yes'}")
    print()

    print("=== Interpretation ===")
    if spread["mean"] < 0 and abs(spread["tstat"]) >= 2:
        print("  RESULT: Q1 (low-turnover) UNDERPERFORMS Q5 (high-turnover).")
        print("  This is the OPPOSITE of the Datar-Naik-Radcliffe (1998) claim.")
        print("  In this survivorship-biased S&P 500 sample, high-turnover names")
        print("  (growth/momentum stocks that thrived) outperform — likely a")
        print("  combination of survivorship bias and momentum co-movement.")
        print("  The effect is statistically significant but directionally REVERSED.")
    else:
        print(f"  Q1-Q5 spread: {spread['mean']:+.4f}, HAC t={spread['tstat']:+.4f}")
        print("  See docs/results.md for full interpretation.")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
