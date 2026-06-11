"""Reproduce the real run (docs/results.md) — USMV (min-vol) vs SPY over the common window.
    python examples/verify.py [--fetch]
"""
from __future__ import annotations
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from bunker import data, strategy as st


def main(fetch):
    ret = data.fetch_pairs(fetch=fetch)
    if ret.empty:
        print("No cached real data. Re-run with --fetch."); return
    al = ret[["USMV", "SPY"]].dropna()   # compare over the common window (USMV starts 2011)
    u, m = st.leg_summary(al, "USMV"), st.leg_summary(al, "SPY")
    sp = st.spread_stats(st.spread(ret, "USMV", "SPY"))
    print(f"\nMin-vol (USMV) vs market (SPY), {al.index.min().date()}..{al.index.max().date()} ({sp['n']} mo)\n")
    print(f"  USMV (min-vol): CAGR {u['cagr']:+.2%}  Sharpe {u['sharpe']:.2f}  vol {u['vol_ann']:.1%}  maxDD {u['max_drawdown']:.0%}")
    print(f"  SPY (market):   CAGR {m['cagr']:+.2%}  Sharpe {m['sharpe']:.2f}  vol {m['vol_ann']:.1%}  maxDD {m['max_drawdown']:.0%}")
    print(f"  vol reduction USMV/SPY = {st.vol_reduction(al,'USMV','SPY'):.2f}  (it does cut risk)")
    print(f"  spread USMV-SPY: {sp['mean_ann']:+.2%}/yr  Sharpe {sp['sharpe']:+.2f}  (Lo t {sp['tstat']:+.2f})")
    try:
        from quantlab import repro
        print(f"\nas-of {ret.index[-1].date()} · inputs fingerprint {repro.fingerprint(al)}")
    except Exception: pass


if __name__ == "__main__":
    main("--fetch" in sys.argv)
