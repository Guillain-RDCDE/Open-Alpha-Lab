"""Reproduce the real run (docs/results.md) — gold as inflation and crisis hedge.
    python examples/verify.py   # cache-only (reads the shared cross-asset + macro pulls)
"""
from __future__ import annotations
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from safe_haven import data, strategy as st


def main(fetch):
    gold_m, eq_ret, cpi_m = data.fetch_panel(fetch=fetch)
    if gold_m.empty:
        print("No cached gold/macro data."); return
    print(f"\nGold the safe haven — GLD/SPY/CPI, {gold_m.index.min().date()}–{gold_m.index.max().date()}\n")
    if not cpi_m.empty:
        ih = st.inflation_hedge(gold_m, cpi_m)
        print("  (1) inflation hedge")
        print(f"      corr(gold YoY, inflation YoY) = {ih['corr']:+.2f}   (n={ih['n']})")
        print(f"      gold YoY in high-inflation months {ih['gold_hi_infl']:+.1%} vs low {ih['gold_lo_infl']:+.1%}  (gap {ih['hi_minus_lo']:+.1%})")
        print(f"      → gold {'tracks' if ih['corr']>0.4 else 'does NOT track'} inflation month-to-month")
    cb = st.crisis_ballast(gold_m.pct_change(), eq_ret)
    print("\n  (2) crisis hedge")
    print(f"      corr(gold, stocks) = {cb['stock_corr']:+.2f}")
    print(f"      in {cb['n_crash']} equity-crash months (SPY<-8%): gold {cb['gold_in_crash']:+.1%} vs SPY {cb['eq_in_crash']:+.1%}")
    print(f"      gold rose in {cb['gold_up_share']:.0%} of those crashes → ballast, not a reliable haven")
    try:
        from quantlab import repro
        print(f"\ninputs fingerprint {repro.fingerprint(gold_m.to_frame('gold'))}")
    except Exception: pass


if __name__ == "__main__":
    main("--fetch" in sys.argv)
