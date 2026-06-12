"""Reproduce the real run (docs/results.md) — yield-curve inversion and forward equity returns.
    python examples/verify.py [--fetch]
"""
from __future__ import annotations
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from inverted import data, strategy as st


def main(fetch):
    d = data.fetch_panel(fetch=fetch)
    if d.empty:
        print("No cached real data. Re-run with --fetch."); return
    slope = st.curve_slope(d["long_yield"], d["short_yield"])
    print(f"\nYield-curve inversion → equities, {d.index.min().date()}..{d.index.max().date()} ({len(d)} mo)\n")
    for h in (12, 18, 24):
        c = st.conditional_forward(slope, d["eq"], horizon=h)
        print(f"  next-{h}m equity: after INVERTED {c['inverted_fwd']:+.1%}  vs NORMAL {c['normal_fwd']:+.1%}  "
              f"(gap {c['gap']:+.1%}, {c['n_inverted']} inverted obs)")
    c = st.conditional_forward(slope, d["eq"], 18)
    print(f"\n  curve inverted {c['inverted_share']:.0%} of months")
    try:
        from quantlab import repro
        print(f"\nas-of {d.index[-1].date()} · inputs fingerprint {repro.fingerprint(d[['long_yield','short_yield']])}")
    except Exception: pass


if __name__ == "__main__":
    main("--fetch" in sys.argv)
