"""Reproduce the real run (docs/results.md) — TQQQ (3x) vs QQQ (1x).
    python examples/verify.py [--fetch]
"""
from __future__ import annotations
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from slow_burn import data, strategy as st


def main(fetch):
    ret = data.fetch_pair(fetch=fetch)
    if ret.empty:
        print("No cached real data. Re-run with --fetch."); return
    print(f"\nTQQQ (3x) vs QQQ (1x), {ret.index.min().date()}..{ret.index.max().date()} ({len(ret)} days)\n")
    for tk, lab in [("QQQ", "1x"), ("TQQQ", "3x actual")]:
        s = st.summary(ret[tk])
        print(f"  {tk} ({lab}): CAGR {s['cagr']:+.2%}  Sharpe {s['sharpe']:.2f}  vol {s['vol_ann']:.0%}  maxDD {s['max_drawdown']:.0%}")
    g = st.decay_gap(ret["QQQ"], L=3.0)
    print(f"\n  naive 3x(QQQ CAGR) = {g['naive_Lx_cagr']:+.1%}  |  3x daily-rebalanced realized = {g['levered_cagr']:+.1%}")
    print(f"  volatility decay (gap) = {g['decay']:+.1%}/yr   vs theory 0.5*L(L-1)*vol^2 = {g['drag_theory']:.1%}/yr")
    print(f"  TQQQ Sharpe {st.summary(ret['TQQQ'])['sharpe']:.2f} <= QQQ Sharpe {st.summary(ret['QQQ'])['sharpe']:.2f} (no risk-adjusted gain)")
    print("  2022:", f"TQQQ {st.summary(ret['TQQQ'].loc['2022'])['cagr']:+.0%}" if '2022' in ret.index.year.astype(str).values else "n/a")
    try:
        from quantlab import repro
        print(f"\nas-of {ret.index[-1].date()} · inputs fingerprint {repro.fingerprint(ret)}")
    except Exception: pass


if __name__ == "__main__":
    main("--fetch" in sys.argv)
