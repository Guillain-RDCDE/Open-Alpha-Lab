"""Reproduce the real-data headline run (docs/results.md).

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # download ^GSPC (effect) + SPY (tradable) from Yahoo
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

from last_call import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    gspc = data.fetch_prices("^GSPC", fetch=fetch, start_year=1950)
    spy = data.fetch_prices("SPY", fetch=fetch, start_year=1993)
    if gspc.empty:
        print("No cached real data. Re-run with --fetch (needs network).")
        return

    d = st.tom_vs_rest(gspc)
    print(f"\nS&P 500 (^GSPC) {gspc.index.min().date()}..{gspc.index.max().date()} — {d['n_tom']+d['n_rest']} days")
    print(f"  TOM days ({d['tom_share']:.0%} of days): {d['tom_bp']:.1f} bp/day  |  non-TOM: {d['rest_bp']:.1f} bp/day  |  Welch t={d['welch_t']:.1f}")

    print(f"\nTradable on SPY {spy.index.min().date()}..{spy.index.max().date()}:")
    for c in (0.0, 2.0, 5.0):
        s = st.summary(st.tom_returns(spy, cost_bps=c))
        print(f"  TOM-only (cost {c:.0f}bp): CAGR {s['cagr']:+.2%}  Sharpe {s['sharpe']:.2f}  maxDD {s['max_drawdown']:.0%}  in-mkt {s['time_in_market']:.0%}")
    bh = st.summary(st.buy_hold(spy))
    print(f"  Buy & hold SPY:        CAGR {bh['cagr']:+.2%}  Sharpe {bh['sharpe']:.2f}  maxDD {bh['max_drawdown']:.0%}")

    print("\nDecay (TOM vs non-TOM bp/day):")
    for lab, lo, hi in [("1950-1987", 1950, 1987), ("1988-2007", 1988, 2007), ("2008-on", 2008, 2100)]:
        sl = gspc[(gspc.index.year >= lo) & (gspc.index.year <= hi)]
        dd = st.tom_vs_rest(sl)
        print(f"  {lab}: TOM {dd['tom_bp']:5.1f}  non-TOM {dd['rest_bp']:5.1f}")

    try:
        from quantlab import repro
        print(f"\nas-of {gspc.index[-1].date()} · inputs fingerprint {repro.fingerprint(gspc.to_frame())}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
