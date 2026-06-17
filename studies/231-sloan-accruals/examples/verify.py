"""Real-tape verification — Study 231 (Sloan Accruals). Regenerates docs/results.md numbers.

Reads the EDGAR cache, computes the Sloan accruals signal, runs quintile sorts and the
random-portfolio control, and prints the headline statistics. No network access required
— reads from the shared ``_cache/`` parquet files.

    python studies/231-sloan-accruals/examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sloan_accruals import data, strategy as st  # noqa: E402


def main() -> None:
    sig, fwd = data.fetch_panel()
    if sig.empty:
        print("ERROR: EDGAR cache not found. Place _edgar_*.parquet files in _cache/.")
        sys.exit(1)

    print(f"Panel: {len(sig.columns)} tickers, {len(sig)} years "
          f"({sig.index.min()}-{sig.index.max()})")
    print(f"Fingerprint (accruals): {data.fingerprint(sig)}")
    print()

    h = st.quintile_returns(sig, fwd, q=0.20)

    print("=== quintile returns (accruals low->high, forward 1yr) ===")
    for q in ["q1", "q2", "q3", "q4", "q5", "market"]:
        s = st.summary(h[q])
        label = "low-acc" if q == "q1" else ("high-acc" if q == "q5" else
                ("EW mkt" if q == "market" else q))
        print(f"  {label:10s} mean={s['mean']*100:+.1f}%/yr  Sharpe={s['sharpe']:+.2f}"
              f"  HAC t={s['tstat']:+.2f}  hit={s['hit_rate']:.0%}")

    hedge = st.summary(h["hedge"])
    print()
    print("=== long low-accruals / short high-accruals hedge ===")
    print(f"  mean={hedge['mean']*100:+.1f}%/yr  Sharpe={hedge['sharpe']:+.2f}  "
          f"HAC t={hedge['tstat']:+.2f}  hit={hedge['hit_rate']:.0%}  n={hedge['n']}")

    hi_excess = st.summary(h["q5"] - h["market"])
    lo_excess = st.summary(h["q1"] - h["market"])
    print()
    print("=== excess returns vs EW market ===")
    print(f"  Low-acc (q1) excess: mean={lo_excess['mean']*100:+.1f}%/yr  "
          f"HAC t={lo_excess['tstat']:+.2f}")
    print(f"  High-acc (q5) excess: mean={hi_excess['mean']*100:+.1f}%/yr  "
          f"HAC t={hi_excess['tstat']:+.2f}")

    print()
    print("=== random-portfolio null distribution ===")
    rand = st.random_portfolio_returns(sig, fwd, q=0.20, n_draws=500, seed=231)
    actual_lo_excess = float((h["q1"] - h["market"]).mean())
    pct_beaten = float((rand < actual_lo_excess).mean())
    print(f"  Random control: mean={rand.mean()*100:+.2f}%  std={rand.std()*100:.1f}%")
    print(f"  Low-acc portfolio beats {pct_beaten:.0%} of random same-size draws")

    print()
    print("=== year-by-year summary ===")
    print(f"{'year':>6}  {'n':>4}  {'q1 (lo)':>9}  {'q5 (hi)':>9}  {'mkt':>9}  {'hedge':>9}")
    for y, row in h.iterrows():
        print(f"{y:>6}  {int(row['n']):>4d}  {row['q1']*100:>+8.1f}%"
              f"  {row['q5']*100:>+8.1f}%  {row['market']*100:>+8.1f}%"
              f"  {row['hedge']*100:>+8.1f}%")

    print()
    print("SURVIVORSHIP BIAS WARNING: panel covers CURRENT S&P 500 members only.")
    print("All results are UPPER BOUNDS on a live strategy.")


if __name__ == "__main__":
    main()
