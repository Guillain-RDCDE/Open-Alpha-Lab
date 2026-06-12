"""Reproduce the real run (docs/results.md) — QYLD (covered call) vs QQQ vs SPY.
    python examples/verify.py [--fetch]
"""
from __future__ import annotations
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from premium_seller import data, strategy as st


def main(fetch):
    ret = data.fetch_panel(fetch=fetch)
    if ret.empty:
        print("No cached real data. Re-run with --fetch."); return
    al = ret[["QYLD", "QQQ", "SPY"]].dropna()
    print(f"\nCovered-call (QYLD) vs index, {al.index.min().date()}..{al.index.max().date()} ({len(al)} mo)\n")
    for tk, lab in [("QYLD", "covered call"), ("QQQ", "underlying"), ("SPY", "market")]:
        s = st.leg_summary(al, tk)
        print(f"  {tk} ({lab}): CAGR {s['cagr']:+.2%}  Sharpe {s['sharpe']:.2f}  vol {s['vol_ann']:.1%}  maxDD {s['max_drawdown']:.0%}")
    sp = st.spread_stats(st.spread(ret, "QYLD", "QQQ"))
    print(f"\n  spread QYLD-QQQ (vs its own underlying): {sp['mean_ann']:+.2%}/yr  Sharpe {sp['sharpe']:+.2f}  (Lo t {sp['tstat']:+.2f})")
    cap = st.capture(ret, "QYLD", "QQQ")
    print(f"  upside capture {cap['upside_capture']:.0%} (keeps {cap['up_fund']*100:.1f}% of QQQ's {cap['up_underlying']*100:.1f}% up months)")
    print(f"  downside capture {cap['downside_capture']:.0%} (takes {cap['down_fund']*100:.1f}% of QQQ's {cap['down_underlying']*100:.1f}% down months)")
    try:
        from quantlab import repro
        print(f"\nas-of {ret.index[-1].date()} · inputs fingerprint {repro.fingerprint(al)}")
    except Exception: pass


if __name__ == "__main__":
    main("--fetch" in sys.argv)
