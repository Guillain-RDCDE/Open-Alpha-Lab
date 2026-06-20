#!/usr/bin/env python
"""Reproduce the pinned headline numbers for Study 334 (ARK-Innovation).

Reads the cached real daily tapes (ARKK, QQQ, XLK) under ``_cache/`` and prints the
fingerprinted, as-of'd headline table that ``docs/results.md`` quotes. Run once with
``--fetch`` to populate the cache from Yahoo!, then offline thereafter.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # populate the cache, then verify
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ark_innovation import data, strategy as st  # noqa: E402

AS_OF = "2026-06-18"
LAST_COMPLETE_MONTH_END = "2026-06-01"  # exclusive cut → drop the partial month
TICKERS = ["ARKK", "QQQ", "XLK"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="populate the cache from Yahoo!")
    args = ap.parse_args()

    px = {}
    for t in TICKERS:
        px[t] = data.fetch_prices(t, fetch=args.fetch)

    print(f"# Study 334 — ARK-Innovation · as-of {AS_OF}\n")
    print("## Data stamp")
    for t in TICKERS:
        b = px[t]
        print(f"  {t:5s} {b.index[0].date()} -> {b.index[-1].date()}  "
              f"{len(b):,} days  fp={data.fingerprint(b)}")

    cut = pd.Timestamp(LAST_COMPLETE_MONTH_END)
    c = px["ARKK"]["close"][px["ARKK"].index < cut]
    print(f"\n  trimmed ARKK: {c.index[0].date()} -> {c.index[-1].date()}  {len(c):,} bars")

    print("\n## ARKK buy-and-hold arc (total-return proxy)")
    full = st.time_weighted_return(c, st.TRADING_DAYS_PER_YEAR)
    print(f"  full sample CAGR {full['cagr']*100:+.2f}%/yr  total {full['total_return']*100:+.1f}%")
    pk = c.idxmax()
    boom = st.time_weighted_return(c[c.index <= pk], 252)
    bust = st.time_weighted_return(c[c.index >= pk], 252)
    print(f"  inception -> peak ({pk.date()}) CAGR {boom['cagr']*100:+.2f}%/yr  total {boom['total_return']*100:+.0f}%")
    print(f"  peak -> as-of     CAGR {bust['cagr']*100:+.2f}%/yr  total {bust['total_return']*100:+.0f}%")

    print("\n## Overlay race (gross, one-day lag)")
    overlays = {
        "trend 50/200": st.ma_crossover_signal(c, 50, 200),
        "ts-mom 1d": st.ts_momentum_signal(c, 1),
        "mean-rev z<-1": st.mean_reversion_signal(c, 5, 1.0),
    }
    for name, sig in overlays.items():
        d = st.overlay_returns(c, sig, cost_bps=0.0)
        s = st.summarize_overlay(d, "gross")
        print(f"  {name:14s} mean {s['mean_bps']:+.2f} bps/d  Sharpe {s['sharpe']:.2f}  "
              f"HAC t {s['hac_t']:+.2f}  inv {s['pct_invested']*100:.0f}%")
    bh = c.pct_change().fillna(0.0).to_numpy()
    print(f"  {'ARKK B&H':14s} Sharpe {st.sharpe(bh):.2f}  HAC t {st.hac_tstat(bh):+.2f}")

    sig = st.ma_crossover_signal(c, 50, 200)
    d = st.overlay_returns(c, sig, 0.0)
    lo, hi = st.block_bootstrap_ci(d["gross"].to_numpy(), block=21, n_boot=2000)
    print(f"  trend overlay daily-mean 95% block-bootstrap CI: [{lo*1e4:+.2f}, {hi*1e4:+.2f}] bps")

    print("\n## Synthetic behaviour-gap control (machinery proof, not market evidence)")
    for lbl, bust_, chase in [("null (steady grower)", 0.0, 0.0),
                              ("boom-bust + chasing", 0.04, 15.0)]:
        fr, _ = data.synthetic_hype_cycle(n_months=180, bust=bust_, chase=chase, seed=334)
        g = st.behaviour_gap(fr["price"], fr["flow"], st.MONTHS_PER_YEAR)
        print(f"  {lbl:22s} TWR {g['twr_cagr']*100:+.2f}%  DWR {g['dwr_irr']*100:+.2f}%  "
              f"gap {g['gap']*100:+.2f} pts")

    return 0


if __name__ == "__main__":
    sys.exit(main())
