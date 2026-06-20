"""Real-tape verification — Study 341 (MLP-Pipelines). Regenerates docs/results.md numbers.

Reads the cached monthly total-return panel (AMLP/MLPA/AMZA + SPY/XLE/USO) and the split-only
price+distribution panel, decomposes each MLP fund's distribution into price vs return-of-capital
(the income illusion), races each fund against SPY total return and XLE total return with a HAC
t-stat and block-bootstrap CI, and measures the energy beta (HAC t on the XLE slope) plus
up/down capture vs energy. Network is touched only with --fetch.

    python studies/341-mlp-pipelines/examples/verify.py            # cache-only
    python studies/341-mlp-pipelines/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mlp_pipelines import data, strategy as st  # noqa: E402

# Drop any partial current month: the as-of run freezes at end-of-May 2026.
AS_OF_CUTOFF = pd.Timestamp("2026-05-31")
FUNDS = ["AMLP", "MLPA", "AMZA"]


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

    print("=== The energy beta: MLP funds regressed on XLE total return ===")
    for f in FUNDS:
        sub = panel[[f, "XLE"]].dropna()
        if sub.empty:
            continue
        eb = st.energy_beta(sub[f], sub["XLE"])
        print(f"{f:5s} n={eb['n']:>3}  beta={eb['beta']:.2f}  HAC t={eb['beta_t']:+.2f}  "
              f"R2={eb['r2']:.2f}  CI[{eb['ci_lo']:.2f},{eb['ci_hi']:.2f}]")

    print("\n=== The race: MLP fund vs SPY total return / vs XLE total return ===")
    for f in FUNDS:
        for bench in ("SPY", "XLE"):
            sub = panel[[f, bench]].dropna()
            if sub.empty:
                continue
            r = st.race(sub[f], sub[bench])
            cap = st.capture(sub[f], sub[bench])
            print(f"{f:5s} vs {bench}: n={r['n']:>3}  CAGR {r['fund_cagr']*100:5.1f}% vs "
                  f"{r['bench_cagr']*100:5.1f}%  spread {r['spread_ann_pct']:+5.1f}%/yr  "
                  f"HAC t={r['spread_t']:+5.2f}  Sh {r['fund_sharpe']:.2f} vs {r['bench_sharpe']:.2f}  "
                  f"DD {r['fund_maxdd']*100:.0f}% vs {r['bench_maxdd']*100:.0f}%  "
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
