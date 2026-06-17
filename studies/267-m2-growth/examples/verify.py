"""Real-data verification -- Study 267 (M2-Growth). Regenerates docs/results.md numbers.

Joins the hardcoded M2 YoY series with cached ^GSPC monthly returns and runs the
full analysis: predictive HAC regressions (1-month and 12-month forward), the
quantile sort, and the tradable long/flat rule with costs.

The ^GSPC monthly parquet is read from the study-level _cache/ (cache-only; no
network). Populate it once with:

    python -c "import sys; sys.path.insert(0,'studies/267-m2-growth'); \
        from m2_growth import data; data.fetch_gspc_monthly(fetch=True)"

Then run from anywhere:

    python studies/267-m2-growth/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from m2_growth import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 267 -- M2-Growth -- real-data verification\n")
    print("Data: M2 YoY hardcoded in data.py (FRED M2SL contour, 1960-2025)")
    print("      ^GSPC monthly price index: cache-only _cache/gspc_monthly.parquet\n")

    df = data.build_real_panel(fetch=False)
    fp = data.fingerprint(df["gspc"])
    print(f"Data stamp: {df.index.min().date()} -> {df.index.max().date()}, "
          f"n={len(df)} months, ^GSPC fingerprint={fp}\n")

    s = st.summarize(df)

    print("=== Predictive regression (lagged M2 YoY -> forward ^GSPC return) ===")
    print(f"  1-month  : beta={s['beta_fwd1']:+.5f}  t_OLS={s['t_ols_fwd1']:+.2f}  "
          f"t_HAC={s['t_hac_fwd1']:+.2f}  R2={s['r2_fwd1']:.4f}")
    print(f"  12-month : beta={s['beta_fwd12']:+.5f}  t_OLS={s['t_ols_fwd12']:+.2f}  "
          f"t_HAC={s['t_hac_fwd12']:+.2f}  R2={s['r2_fwd12']:.4f}")
    print("  (the 12-mo naive OLS t looks significant; HAC kills it -- overlapping windows)\n")

    print("=== Quantile sort (next-month return, high-money vs low-money) ===")
    print(f"  High-money bucket mean : {s['hi_mean']:+.2f}%")
    print(f"  Low-money bucket  mean : {s['lo_mean']:+.2f}%")
    print(f"  High - Low spread      : {s['spread']:+.2f}pp  (Welch t = {s['spread_t']:+.2f})\n")

    print("=== Tradable rule (long ^GSPC when lagged M2 > trailing median, else cash) ===")
    print(f"  M2 rule (net) : {s['ann_net']:+.2f}%/yr  Sharpe {s['sharpe_net']:.2f}  "
          f"in-market {s['time_in_market']*100:.0f}%  turnover {s['turnover_per_yr']:.2f}/yr")
    print(f"  Buy & hold    : {s['ann_bah']:+.2f}%/yr  Sharpe {s['sharpe_bah']:.2f}\n")

    print("Verdict: Signal=NONE, Tradability=MIRAGE -- the overlay is contemporaneous, "
          "not predictive; the lagged HAC t is ~0 (and wrong-signed).")


if __name__ == "__main__":
    main()
