"""Reproduce the real run (docs/results.md) — the skewness (lottery) effect in commodities.
    python examples/verify.py [--fetch]
"""
from __future__ import annotations
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from long_shot import data, strategy as st


def main(fetch):
    daily = data.fetch_panel(fetch=fetch)
    if daily.empty:
        print("No cached real data. Re-run with --fetch."); return
    sig = st.skewness_signal(daily)
    h = st.cross_section_hedge(daily, sig, long_high=False)   # long low-skew
    s = st.stats(h)
    print(f"\nCommodity skewness effect, {h.index.min().date()}..{h.index.max().date()} ({len(h)} months, {daily.shape[1]} commodities)\n")
    print(f"  textbook trade (long low-skew / short high-skew): mean {s['mean_ann']:+.2%}/yr  Sharpe {s['sharpe']:+.2f}  (Lo t={s['tstat']:+.2f})  hit {s['hit_rate']:.0%}")
    print("  >0 ⇒ the lottery/skewness effect works (low-skew wins)")
    print("\n  decay (Sharpe):")
    for lab, sl in [("2009-2017", h.loc[:"2017"]), ("2018-on", h.loc["2018":])]:
        print(f"    {lab}: {st.stats(sl)['sharpe']:+.2f}")
    print("  net Sharpe after costs:")
    for c in (10, 25): print(f"    {c} bp/trade: {st.stats(st.net_of_cost(h, c))['sharpe']:+.2f}")
    try:
        from quantlab import repro
        print(f"\ninputs fingerprint {repro.fingerprint(daily.fillna(0))}")
    except Exception: pass


if __name__ == "__main__":
    main("--fetch" in sys.argv)
