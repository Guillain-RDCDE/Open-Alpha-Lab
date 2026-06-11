"""Reproduce the real-data headline run (docs/results.md) — high-dividend vs the market.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # download VYM + SPY monthly from Yahoo, then run
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

from yield_trap import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    ret = data.fetch_pairs(fetch=fetch)
    if ret.empty:
        print("No cached real data. Re-run with --fetch (needs network).")
        return

    sp = st.spread(ret, "VYM", "SPY")
    s = st.spread_stats(sp)
    print(f"\nHigh-dividend (VYM) vs market (SPY), {ret.dropna().index.min().date()}..{ret.index.max().date()} ({s['n']} months)\n")
    v, m = st.leg_summary(ret, "VYM"), st.leg_summary(ret, "SPY")
    print(f"  VYM (high dividend): CAGR {v['cagr']:+.2%}  Sharpe {v['sharpe']:.2f}  maxDD {v['max_drawdown']:.0%}")
    print(f"  SPY (market):        CAGR {m['cagr']:+.2%}  Sharpe {m['sharpe']:.2f}  maxDD {m['max_drawdown']:.0%}")
    print(f"  spread VYM-SPY: mean {s['mean_ann']:+.2%}/yr  Sharpe {s['sharpe']:+.2f}  (Lo t={s['tstat']:+.2f})  hit {s['hit_rate']:.0%}")

    try:
        from quantlab import repro
        print(f"\nas-of {ret.index[-1].date()} · inputs fingerprint {repro.fingerprint(ret.dropna())}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
