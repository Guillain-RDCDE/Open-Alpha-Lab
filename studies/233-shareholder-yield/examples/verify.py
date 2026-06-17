"""Reproduce the real run (docs/results.md) — shareholder yield on the S&P 500.
    python examples/verify.py         # cache-only (reads shared EDGAR pull)
    python examples/verify.py --fetch # also fetches div-yield via yfinance
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

from shareholder_yield import data, strategy as st


def main(fetch: bool = False) -> None:
    total_yield, fwd, comps = data.fetch_panel(fetch=fetch)
    if total_yield.empty:
        print("No cached EDGAR data — run from a machine with _cache/ populated.")
        return

    h = st.quantile_hedge(total_yield, fwd, q=0.2, long_high=True)
    n = int(total_yield.notna().sum(axis=1).median())
    print(
        f"\nShareholder-yield long-short, {h.index.min()}–{h.index.max()} "
        f"({len(h)} years, ~{n} names/yr)\n"
    )
    print(h.round(3).to_string())
    s = st.summary(h["hedge"])
    high_s = st.summary(h["high"])
    low_s  = st.summary(h["low"])
    mkt_s  = st.summary(st.market_annual(fwd).reindex(h.index))
    print(
        f"\n  hedge (long high yield, short low yield): "
        f"mean {s['mean']:+.2%}/yr  Sharpe {s['sharpe']:.2f}  "
        f"(t~{s['tstat']:.1f})  hit {s['hit_rate']:.0%}"
    )
    print(
        f"  high-yield leg {high_s['mean']:+.2%}  "
        f"low-yield leg {low_s['mean']:+.2%}  "
        f"market {mkt_s['mean']:+.2%}"
    )
    print("  >0 => shareholder yield predicts outperformance; <0 => growth beats yield (inverted)")
    if not comps.empty:
        print(f"\n  Components: mean buyback yield {comps['bby_mean'].mean():+.2%}")
    try:
        from quantlab import repro
        print(f"\ninputs fingerprint {repro.fingerprint(total_yield.fillna(0))}")
    except Exception:
        pass


if __name__ == "__main__":
    main("--fetch" in sys.argv)
