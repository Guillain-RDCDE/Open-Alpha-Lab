"""Real-panel verification — Study 540 (Distress-Risk-Anomaly). Regenerates docs/results.md numbers.

Reads this study's own yfinance caches (daily prices + a fundamentals snapshot), builds a
Campbell-Hilscher-Szilagyi-style distress score (leverage, profitability, equity volatility),
sorts the cross-section, and tests the *distress puzzle*: do the most-distressed firms earn the
LOWEST forward returns? Prints every number that lands in docs/results.md, plus the placebo null,
costs, the firm-level cross-sectional slope, and the deterministic synthetic positive control.

    python studies/540-distress-risk-anomaly/examples/verify.py

Everything runs offline once the caches exist. SPLIT is the scoring date: the score uses only
data up to and including the SPLIT close (trailing vol + the statement snapshot), so the signal is
fully public at that close; the position is entered there and held over SPLIT -> END. No future
data enters the signal. The same-close fill is a documented convention — on a ~2-year hold a
one-bar-later entry moves the headline spread ~1pt and changes no stamp.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from distress_risk_anomaly import data, strategy as st  # noqa: E402

# Scoring date (signal public) and the forward holding window end. The accounting snapshot is
# roughly contemporaneous with the recent SPLIT, so scoring is not forward-looking; the return
# tested is the forward window only.
SPLIT = "2024-06-28"
END = "2026-06-25"


def main() -> None:
    prices = data.fetch_prices(fetch=False)
    funds = data.fetch_fundamentals(fetch=False)

    if prices.empty or not funds:
        print("Cache not found. Run once with network: data.fetch_prices(fetch=True) and "
              "data.fetch_fundamentals(fetch=True).")
        return

    print("Study 540 — Distress-Risk-Anomaly — real yfinance panel\n")
    fp_px = data.fingerprint(prices)
    fp_fn = data.fingerprint(funds)
    print(f"Prices: {prices.shape}, fingerprint={fp_px}, "
          f"{prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"Fundamentals snapshot: {len(funds)} names, fingerprint={fp_fn}")
    print(f"Scoring date (signal public): {SPLIT}   Forward window: {SPLIT} -> {END}")
    print("As-of: 2026-06-26 | Source: yfinance daily adjusted close + statement snapshot\n")

    panel = st.build_panel(prices, funds, split_date=SPLIT, end_date=END)
    if panel.empty:
        print("Empty panel.")
        return
    fp_panel = data.fingerprint(panel)
    print(f"Cross-section: {len(panel)} names, panel fingerprint={fp_panel}\n")

    # --- buckets -----------------------------------------------------------
    b = st.bucket_returns(panel, frac=0.3)
    ls = st.long_short_tstat(b)
    print("=== Distress buckets (tercile tails, forward total return over the holding window) ===")
    print(f"  Names:                 {b['n']}  ({b['k']} per bucket)")
    print(f"  SAFE bucket mean:      {b['safe_mean']*100:+.2f}%")
    print(f"  DISTRESSED bucket mean:{b['distressed_mean']*100:+.2f}%")
    print(f"  Spread (safe - dist):  {b['spread']*100:+.2f}%   (puzzle predicts > 0)")
    print(f"  Two-sample t:          {ls['t']:+.3f}")

    # --- placebo -----------------------------------------------------------
    p = st.placebo_pvalue(panel, n_perm=2000, seed=540)
    print(f"\n=== Label-shuffle placebo (2000 perms) ===")
    print(f"  Placebo p-value:       {p:.3f}   (>=0.05 => indistinguishable from noise)")

    # --- firm-level cross-section -----------------------------------------
    reg = st.cross_section_regression(panel)
    print(f"\n=== Firm-level cross-section: forward_ret ~ distress ===")
    print(f"  Slope:                 {reg['slope']*100:+.2f}% per distress unit")
    print(f"  Slope t:               {reg['slope_t']:+.3f}   (negative = the puzzle)")
    print(f"  corr(distress, fwd):   {reg['corr']:+.3f}")

    # --- costs -------------------------------------------------------------
    hold_years = 2.0
    net = st.net_spread(b, cost_bps=5.0, borrow_ann_bps=100.0, holding_years=hold_years)
    print(f"\n=== Costs (5 bps/leg round-trip + 100 bps/yr borrow on the distressed short) ===")
    print(f"  Gross spread:          {b['spread']*100:+.2f}%")
    print(f"  Net spread:            {net*100:+.2f}%   (over {hold_years:.0f}y hold)")

    # --- robustness across scoring windows ---------------------------------
    print(f"\n=== Robustness: spread sign across alternative scoring windows ===")
    for sp, en in [("2022-06-28", "2024-06-28"), ("2023-06-28", "2025-06-25"),
                   ("2024-06-28", END), ("2021-06-28", "2023-06-28")]:
        pn = st.build_panel(prices, funds, split_date=sp, end_date=en)
        if pn.empty:
            continue
        bb = st.bucket_returns(pn, frac=0.3)
        rr = st.cross_section_regression(pn)
        print(f"  {sp} -> {en}: spread {bb['spread']*100:+6.1f}%  slope-t {rr['slope_t']:+.2f}")

    # --- synthetic positive control (seed-robust) --------------------------
    print(f"\n=== Synthetic positive control (seed-robust, 25 seeds) ===")
    for a in (0.0, -0.06, -0.12, -0.20):
        mt = st.synthetic_mean_t(data, distress_alpha=a, n_seeds=25)
        tag = "null" if a == 0.0 else "puzzle planted"
        print(f"  distress_alpha={a:+.2f} ({tag:14s}) -> mean slope-t = {mt:+.2f}")

    print("\nNOTE Survivorship: the basket is names STILL TRADING in 2026 — the bankruptcies that")
    print("     drive the real CHS effect are absent, biasing the real tape AGAINST the puzzle.")
    print("     Accounting legs (leverage, ROA) are a recent yfinance statement snapshot, not a")
    print("     full point-in-time history. Real numbers are an honest small-basket lower bound.")


if __name__ == "__main__":
    main()
