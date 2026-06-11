"""Reproduce the real run (docs/results.md) — the term premium (IEF vs SHY vs BIL).
    python examples/verify.py [--fetch]
"""
from __future__ import annotations
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from downhill import data, strategy as st


def main(fetch):
    ret = data.fetch_panel(fetch=fetch)
    if ret.empty:
        print("No cached real data. Re-run with --fetch."); return
    al = ret[["IEF", "SHY", "BIL"]].dropna()
    print(f"\nTerm premium / riding the curve, {al.index.min().date()}..{al.index.max().date()} ({len(al)} mo)\n")
    for tk, lab in [("IEF","7-10y"),("SHY","1-3y"),("BIL","cash")]:
        s = st.leg_summary(al, tk)
        print(f"  {tk} ({lab}): CAGR {s['cagr']:+.2%}  Sharpe {s['sharpe']:.2f}  vol {s['vol_ann']:.1%}  maxDD {s['max_drawdown']:.0%}")
    tp = st.excess_stats(st.term_premium(ret))
    print(f"\n  term premium IEF-BIL: mean {tp['mean_ann']:+.2%}/yr  Sharpe {tp['sharpe']:+.2f}  (Lo t {tp['tstat']:+.2f})  hit {tp['hit_rate']:.0%}")
    print("  >0 ⇒ a real premium; but compare its Sharpe to just holding cash (BIL) above")
    try:
        from quantlab import repro
        print(f"\nas-of {ret.index[-1].date()} · inputs fingerprint {repro.fingerprint(al)}")
    except Exception: pass


if __name__ == "__main__":
    main("--fetch" in sys.argv)
