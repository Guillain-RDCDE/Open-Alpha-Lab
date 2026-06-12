"""Reproduce the real run (docs/results.md) — bitcoin as "digital gold".
    python examples/verify.py   # cache-only (reads the BTC/SPY/GLD pull)
"""
from __future__ import annotations
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from digital_gold import data, strategy as st


def main(fetch):
    ret = data.fetch_panel(fetch=fetch)
    if ret.empty:
        print("No cached BTC/SPY/GLD data."); return
    print(f"\nBitcoin as 'digital gold' — BTC/SPY/GLD, {ret.index.min().date()}–{ret.index.max().date()} ({len(ret)} days)\n")
    for nm in ["BTC-USD", "SPY", "GLD"]:
        s = st.annualized(ret[nm])
        print(f"  {nm:8s} ann {s['ann']:+.0%}  vol {s['vol']:.0%}  Sharpe {s['sharpe']:.2f}  maxDD {s['max_drawdown']:.0%}")
    c = st.correlations(ret)
    print(f"\n  corr(BTC, stocks) {c['btc_stock']:+.2f}  |  corr(BTC, gold) {c['btc_gold']:+.2f}  |  corr(stocks, gold) {c['stock_gold']:+.2f}")
    print(f"  BTC–stock corr drifting: {c['btc_stock_first']:+.2f} (first half) → {c['btc_stock_second']:+.2f} (second half)")
    m = (1 + ret).resample("ME").prod() - 1
    cb = st.crisis_behavior(m)
    print(f"\n  in {cb['n_crash']} equity-crash months (SPY<-8%): BTC {cb['btc_in_crash']:+.1%}  gold {cb['gold_in_crash']:+.1%}  stocks {cb['eq_in_crash']:+.1%}")
    print(f"  BTC rose in {cb['btc_up_share']:.0%} of crashes (gold {cb['gold_up_share']:.0%}) → not a safe haven")
    print("\n  a small BTC sleeve added to an all-stock book:")
    sl = st.sleeve_effect(ret["SPY"], ret["BTC-USD"])
    for name, row in sl.iterrows():
        print(f"    {name:8s} ann {row['ann']:+.1%}  Sharpe {row['sharpe']:.2f}  maxDD {row['max_drawdown']:.0%}")
    try:
        from quantlab import repro
        print(f"\ninputs fingerprint {repro.fingerprint(ret)}")
    except Exception: pass


if __name__ == "__main__":
    main("--fetch" in sys.argv)
