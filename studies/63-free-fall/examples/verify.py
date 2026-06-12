"""Reproduce the real run (docs/results.md) — SVXY (short-vol) vs SPY.
    python examples/verify.py [--fetch]
"""
from __future__ import annotations
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from free_fall import data, strategy as st


def main(fetch):
    ret = data.fetch_pair(fetch=fetch)
    if ret.empty:
        print("No cached real data. Re-run with --fetch."); return
    print(f"\nShort-vol (SVXY) vs market (SPY), {ret.index.min().date()}..{ret.index.max().date()} ({len(ret)} days)\n")
    for tk, lab in [("SVXY", "short-vol"), ("SPY", "market")]:
        s = st.summary(ret[tk])
        print(f"  {tk} ({lab}): CAGR {s['cagr']:+.2%}  Sharpe {s['sharpe']:.2f}  vol {s['vol_ann']:.0%}  maxDD {s['max_drawdown']:.0%}  skew {s['skew']:+.1f}")
    wd, wdt = st.worst_day(ret["SVXY"])
    print(f"\n  worst SVXY day: {wd:+.0%} on {wdt.date()} (Volmageddon)")
    c = st.carry_vs_crash(ret["SVXY"])
    print(f"  carry: median day {c['median_day_bp']:+.0f} bp, mean day {c['mean_day_bp']:+.0f} bp")
    print(f"  {c['n_crash_days']} crash day(s) (<-20%) wiped {c['crash_days_total']:+.0%} between them")
    post = ret.loc["2018-05-01":]
    if len(post) > 50:
        s = st.summary(post["SVXY"])
        print(f"  post-Volmageddon (de-levered): CAGR {s['cagr']:+.2%}  Sharpe {s['sharpe']:.2f}  (the carry resumes)")
    try:
        from quantlab import repro
        print(f"\nas-of {ret.index[-1].date()} · inputs fingerprint {repro.fingerprint(ret)}")
    except Exception: pass


if __name__ == "__main__":
    main("--fetch" in sys.argv)
