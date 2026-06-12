"""Reproduce the real run (docs/results.md) — the pre-FOMC drift on SPY.
    python examples/verify.py   # cache-only (reads the shared SPY pull)
"""
from __future__ import annotations
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from fed_drift import data, strategy as st


def main(fetch):
    ret, fomc = data.fetch_panel(fetch=fetch)
    if ret.empty:
        print("No cached SPY data."); return
    t = st.drift_table(ret, fomc, lead=1)
    print(f"\nPre-FOMC drift, SPY {ret.index.min().date()}–{ret.index.max().date()} "
          f"({t['n_total']} sessions, {t['n_pre']} pre-FOMC days)\n")
    print(f"  pre-FOMC day mean   {t['pre_mean']*100:+.3f}%/day")
    print(f"  all other days mean {t['rest_mean']*100:+.3f}%/day   (diff t~{t['tstat']:.2f})")
    print(f"  the {t['n_pre']} pre-FOMC days ({t['n_pre']/t['n_total']:.1%} of sessions) earned "
          f"{t['pre_share']:.1%} of SPY's total cumulative return")
    sp = st.split_by_date(ret, fomc, cut="2011-01-01")
    print("\n  did it survive publication (Lucca-Moench, 2011–15)?")
    for era, tab in sp.items():
        print(f"    {era:17s}: pre {tab['pre_mean']*100:+.3f}%/day vs rest {tab['rest_mean']*100:+.3f}% "
              f"(t~{tab['tstat']:.2f}); pre-FOMC days = {tab['pre_share']:.1%} of return")
    try:
        from quantlab import repro
        print(f"\ninputs fingerprint {repro.fingerprint(ret.to_frame())}")
    except Exception: pass


if __name__ == "__main__":
    main("--fetch" in sys.argv)
