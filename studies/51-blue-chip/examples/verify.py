"""Reproduce the real-data headline run (docs/results.md) — quality (gross profitability) on the S&P 500.

    python examples/verify.py            # cache-only (offline; reads the shared EDGAR pull)
    python examples/verify.py --fetch    # crawl SEC EDGAR for GrossProfit + Assets (slow)
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from blue_chip import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    sig, fwd = data.fetch_panel(fetch=fetch)
    if sig.empty:
        print("No cached EDGAR data. Re-run with --fetch (slow) to crawl SEC EDGAR.")
        return

    h = st.quantile_hedge(sig, fwd, q=0.2, long_high=True)
    n_names = int(sig.notna().sum(axis=1).median())
    print(f"\nQuality (gross profitability) long-short, {h.index.min()}-{h.index.max()} "
          f"({len(h)} years, ~{n_names} names/yr)\n")
    print(h.round(3).to_string())
    s = st.summary(h["hedge"])
    print(f"\n  hedge (long high-GP / short low-GP): mean {s['mean']:+.2%}/yr  Sharpe {s['sharpe']:.2f}  "
          f"(t≈{s['tstat']:.1f})  hit {s['hit_rate']:.0%}")
    print(f"  high-GP leg {st.summary(h['high'])['mean']:+.2%}  low-GP leg {st.summary(h['low'])['mean']:+.2%}  "
          f"market {st.summary(st.market_annual(fwd).reindex(h.index))['mean']:+.2%}")

    try:
        from quantlab import repro
        print(f"\ninputs fingerprint {repro.fingerprint(sig.fillna(0))}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
