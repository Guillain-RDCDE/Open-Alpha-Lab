"""Real-tape verification — Study 337 (Covered-Call-ETF). Regenerates docs/results.md numbers.

Reads the cached monthly total-return panel (JEPI/JEPQ/QYLD/XYLD/RYLD + SPY/QQQ) and the
split-only price+distribution panel, races each buy-write ETF against SPY total return with a
HAC t-stat and block-bootstrap CI, decomposes the distribution into price vs return-of-capital
(the income illusion), and reports up/down capture. Network is touched only with --fetch.

    python studies/337-covered-call-etf/examples/verify.py            # cache-only
    python studies/337-covered-call-etf/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from covered_call_etf import data, strategy as st  # noqa: E402

# Drop any partial current month: the as-of run freezes at end-of-May 2026.
AS_OF_CUTOFF = pd.Timestamp("2026-05-31")
FUNDS = ["JEPI", "JEPQ", "QYLD", "XYLD", "RYLD"]


def main(fetch: bool) -> None:
    panel = data.fetch_panel(fetch=fetch)
    split = data.fetch_price_and_dist(fetch=fetch)
    if panel.empty:
        print("No cached total-return panel. Run with --fetch once to populate the cache.")
        return
    panel = panel[panel.index <= AS_OF_CUTOFF]
    if not split.empty:
        split = split[split.index <= AS_OF_CUTOFF]

    print(f"TR panel: {panel.index.min().date()} -> {panel.index.max().date()}  "
          f"fp={data.fingerprint(panel.fillna(0))}")
    if not split.empty:
        print(f"split   : {split.index.min().date()} -> {split.index.max().date()}  "
              f"fp={data.fingerprint(split.fillna(0))}\n")

    print("=== The race: buy-write vs SPY total return ===")
    for f in FUNDS:
        sub = panel[[f, "SPY"]].dropna()
        if sub.empty:
            continue
        r = st.race(sub[f], sub["SPY"])
        cap = st.capture(sub[f], sub["SPY"])
        print(f"{f:5s} n={r['n']:>3} {sub.index.min().date()}  "
              f"CAGR {r['fund_cagr']*100:5.1f}% vs SPY {r['bench_cagr']*100:5.1f}%  "
              f"spread {r['spread_ann_pct']:+5.1f}%/yr  HAC t={r['spread_t']:+5.2f}  "
              f"CI[{r['spread_ci_lo_bps']:+.0f},{r['spread_ci_hi_bps']:+.0f}]bps  "
              f"Sharpe {r['fund_sharpe']:.2f} vs {r['bench_sharpe']:.2f}  "
              f"up {cap['up_capture']:.2f}/dn {cap['down_capture']:.2f}")

    if not split.empty:
        print("\n=== The income illusion: distribution vs NAV (return of capital) ===")
        for f in FUNDS:
            try:
                price, dist = split[(f, "price")], split[(f, "dist")]
            except KeyError:
                continue
            both = pd.concat([price, dist], axis=1).dropna()
            if both.empty:
                continue
            ill = st.income_illusion(both.iloc[:, 0], both.iloc[:, 1])
            print(f"{f:5s} dist {ill['dist_yield']*100:4.1f}%/yr  "
                  f"price CAGR {ill['price_cagr']*100:+5.1f}%  "
                  f"total {ill['total_cagr']*100:5.1f}%  "
                  f"return-of-capital share {ill['return_of_capital_share']:.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
