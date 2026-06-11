"""Reproduce the real-data headline run (docs/results.md) — asset growth on S&P 500 survivors.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # crawl SEC EDGAR for total assets + Yahoo returns (slow)
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

from growth_spurt import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    ag, fwd = data.fetch_panel(fetch=fetch)
    if ag.empty:
        print("No cached real data. Re-run with --fetch (needs network; the EDGAR crawl is slow).")
        return

    h = st.quantile_hedge(ag, fwd, q=0.2)
    mkt = st.market_annual(fwd)
    n_names = int(ag.notna().sum(axis=1).median())
    print(f"\nAsset-growth long-short, {h.index.min()}-{h.index.max()} "
          f"({len(h)} years, ~{n_names} names/yr, current S&P 500 survivors)\n")
    print(h.round(3).to_string())

    sh = st.summary(h["hedge"])
    sm = st.summary(mkt.reindex(h.index))
    print(f"\n  hedge (long low-growth / short high-growth): mean {sh['mean']:+.2%}/yr  "
          f"Sharpe {sh['sharpe']:.2f}  hit {sh['hit_rate']:.0%}")
    print(f"  long leg mean {st.summary(h['low_growth'])['mean']:+.2%}  short leg mean {st.summary(h['high_growth'])['mean']:+.2%}")
    print(f"  equal-weight market mean {sm['mean']:+.2%}  Sharpe {sm['sharpe']:.2f}")
    print(f"  vendor headline Sharpe: 0.835")

    try:
        from quantlab import repro
        print(f"\ninputs fingerprint {repro.fingerprint(ag.fillna(0))}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
