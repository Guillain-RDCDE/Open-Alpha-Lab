"""Real-tape verification — Study 197 (Dividend-Payout-Ratio).

Reads the staged Shiller dataset, computes the payout ratio, runs the full
regression suite (HAC OLS, IS/OOS split, quintile analysis), and prints the
headline numbers that populate docs/results.md.

    python studies/197-dividend-payout-ratio/examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dividend_payout_ratio import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 197 — Dividend-Payout-Ratio — real Shiller tape\n")

    df = data.load_shiller()
    fp = data.fingerprint(df)
    n_total = len(df)
    date_start = df.index.min().date()
    date_end = df.index.max().date()

    print(f"Data: Shiller monthly, {date_start} → {date_end}")
    print(f"N monthly obs (with full payout + 10y forward): {n_total}")
    print(f"Fingerprint (Payout series): {fp}")
    print()

    # --- Primary: payout → 10y real earnings growth ---
    reg_eg = st.regress_payout(df, "fwd_10y_eg", "Payout", maxlags=120)
    print("=== Payout → 10y real earnings growth (HAC, 120 lags) ===")
    print(
        f"  slope  = {reg_eg['slope']:+.4f}  t = {reg_eg['t_slope']:+.3f}  R² = {reg_eg['r_squared']:.4f}  N = {reg_eg['n']}"
    )
    print(
        f"  unconditional mean EG = {reg_eg['unconditional_mean']:+.4f}  t(≠0) = {reg_eg['unconditional_t']:+.2f}"
    )

    # --- Primary: payout → 10y real price return ---
    reg_px = st.regress_payout(df, "fwd_10y_price", "Payout", maxlags=120)
    print()
    print("=== Payout → 10y real price return (HAC, 120 lags) ===")
    print(
        f"  slope  = {reg_px['slope']:+.4f}  t = {reg_px['t_slope']:+.3f}  R² = {reg_px['r_squared']:.4f}  N = {reg_px['n']}"
    )

    # --- IS/OOS split ---
    split_eg = st.is_oos_split(df, "fwd_10y_eg", "Payout", split_year=1953, maxlags=120)
    print()
    print("=== IS/OOS split (split 1953) — fwd_10y_eg ===")
    for period in ("is", "oos"):
        r = split_eg[period]
        label = "IS (≤1953)" if period == "is" else "OOS (>1953)"
        print(
            f"  {label}: slope={r['slope']:+.4f}  t={r['t_slope']:+.3f}  R²={r['r_squared']:.4f}  N={r['n']}"
        )

    # --- Non-overlapping robustness ---
    rob_eg = st.non_overlapping_regression(df, "fwd_10y_eg", "Payout")
    print()
    print("=== Non-overlapping (Jan obs only) — fwd_10y_eg ===")
    print(
        f"  slope={rob_eg['slope']:+.4f}  t_HC3={rob_eg['t_hc3']:+.3f}  R²={rob_eg['r_squared']:.4f}  N={rob_eg['n']}"
    )

    # --- Quintile analysis ---
    qm = st.payout_quintile_means(df, "fwd_10y_eg", "Payout")
    print()
    print("=== Quintile means — fwd_10y_eg ===")
    print(qm.to_string())

    # --- Tradability: price return split ---
    split_px = st.is_oos_split(df, "fwd_10y_price", "Payout", split_year=1953, maxlags=120)
    print()
    print("=== IS/OOS split (split 1953) — fwd_10y_price ===")
    for period in ("is", "oos"):
        r = split_px[period]
        label = "IS (≤1953)" if period == "is" else "OOS (>1953)"
        print(
            f"  {label}: slope={r['slope']:+.4f}  t={r['t_slope']:+.3f}  R²={r['r_squared']:.4f}  N={r['n']}"
        )

    print()
    print("=== Verdict ===")
    print("Signal (fwd_10y_eg):  WEAK  — t ≈ 3.5 HAC but < 2 on non-overlapping annual obs")
    print("Signal (fwd_10y_px):  NONE  — t ≈ -1.2, wrong sign vs claim")
    print("Tradability:          MIRAGE — 10-year horizon, not a timing signal")


if __name__ == "__main__":
    main()
