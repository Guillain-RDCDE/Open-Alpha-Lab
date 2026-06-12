"""Reproduce the real run (docs/results.md) — net share issuance on the S&P 500.
    python examples/verify.py   # cache-only (reads the shared EDGAR pull)
"""
from __future__ import annotations
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from share_shuffle import data, strategy as st


def main(fetch):
    sig, fwd = data.fetch_panel(fetch=fetch)
    if sig.empty:
        print("No cached EDGAR data."); return
    h = st.quantile_hedge(sig, fwd, q=0.2, long_high=False)   # long LOW issuance / buybacks
    n = int(sig.notna().sum(axis=1).median())
    print(f"\nNet-issuance long-short, {h.index.min()}-{h.index.max()} ({len(h)} years, ~{n} names/yr)\n")
    print(h.round(3).to_string())
    s = st.summary(h["hedge"])
    print(f"\n  hedge (long buyback/low-issuance, short issuers): mean {s['mean']:+.2%}/yr  Sharpe {s['sharpe']:.2f}  (t~{s['tstat']:.1f})  hit {s['hit_rate']:.0%}")
    print(f"  buyback leg {st.summary(h['low'])['mean']:+.2%}  issuer leg {st.summary(h['high'])['mean']:+.2%}  market {st.summary(st.market_annual(fwd).reindex(h.index))['mean']:+.2%}")
    print("  >0 ⇒ the issuance anomaly works; <0 ⇒ issuers beat buybacks (inverted on large caps)")
    try:
        from quantlab import repro
        print(f"\ninputs fingerprint {repro.fingerprint(sig.fillna(0))}")
    except Exception: pass


if __name__ == "__main__":
    main("--fetch" in sys.argv)
