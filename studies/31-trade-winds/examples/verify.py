"""Reproduce the real-data headline run (docs/results.md) — diversified futures, 2000–today.

    python examples/verify.py            # cache-only (offline); prints if cache present
    python examples/verify.py --fetch    # download the futures basket from Yahoo, then run

Fetches the diversified continuous-futures basket, runs the equal-risk time-series-momentum book vs
the long-only basket and 60/40, and prints the crisis-alpha, cost-sweep and portfolio-blend numbers
that docs/results.md fingerprints.
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from trade_winds import costs, data, extension, strategy


def main(fetch: bool) -> None:
    r = data.fetch_futures(fetch=fetch)
    if r.empty:
        print("No cached real data. Re-run with --fetch (needs network) to download the futures basket.")
        return
    try:
        from quantlab import repro
        r = repro.as_of(r, "2026-06-10")  # same as-of as Study 32 — the two studies share one tape
    except Exception:
        pass
    tsmom = strategy.book_returns(r, cost_bps=2.0)
    lob = costs.long_only_basket(r)
    eqcols = [c for c in ("ES=F", "NQ=F", "YM=F") if c in r.columns]
    sixty40 = (0.6 * r["ES=F"] + 0.4 * r["ZN=F"]) if {"ES=F", "ZN=F"} <= set(r.columns) else lob

    def line(nm, x):
        s = strategy.summary(x)
        print(f"  {nm:26} Sharpe {s['sharpe']:5.2f}  CAGR {s['cagr']*100:5.1f}%  vol {s['vol_ann']*100:3.0f}%  "
              f"maxDD {s['max_drawdown']*100:5.0f}%  skew {s['skew']:+.2f}")

    print(f"\nDiversified futures, {r.index[0].date()} - {r.index[-1].date()} ({r.shape[1]} markets, {len(r)} days)\n")
    line("TSMOM (trend book)", tsmom)
    line("long-only basket", lob)
    line("60/40 (ES/ZN)", sixty40)

    print("\nCrisis alpha:", {k: (round(v, 3) if isinstance(v, float) else v)
                              for k, v in costs.crisis_alpha(r, eqcols).items()})
    print("\nCost sweep (bp -> Sharpe/CAGR):\n" + costs.cost_sweep(r).round(3).to_string())
    print("\nSub-period Sharpe (decay):\n" + extension.subperiod_sharpe(r, n_splits=3).round(3).to_string())
    print("\nLookback sweep:\n" + extension.lookback_sweep(r).round(3).to_string())

    eq = r["ES=F"].reindex(tsmom.index)
    print("\nPortfolio blend (the payoff):")
    for nm, x in [("equities only", eq), ("60/40 equity/trend", 0.6 * eq + 0.4 * tsmom),
                  ("60/40 only", sixty40.reindex(tsmom.index)),
                  ("70/30 (60/40 + trend)", 0.7 * sixty40.reindex(tsmom.index) + 0.3 * tsmom)]:
        line(nm, x)

    try:
        from quantlab import repro
        print(f"\nas-of {r.index[-1].date()} · inputs fingerprint {repro.fingerprint(r)}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
