"""Reproduce the real-data headline run (docs/results.md) — Sell in May on the S&P 500.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # download ^GSPC from Yahoo, then run
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

from summer_lull import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    ret = data.fetch_index(fetch=fetch)
    if ret.empty:
        print("No cached real data. Re-run with --fetch (needs network).")
        return

    sp = st.seasonal_split(ret)
    print(f"\nSell in May, S&P 500 {ret.index.min().date()}..{ret.index.max().date()} ({len(ret)} months)\n")
    print(f"  winter (Nov-Apr): {sp['winter_bp']:.0f} bp/mo (ann {sp['winter_ann']:+.1%}, n={sp['n_winter']})")
    print(f"  summer (May-Oct): {sp['summer_bp']:.0f} bp/mo (ann {sp['summer_ann']:+.1%}, n={sp['n_summer']})")
    print(f"  Welch t (winter-summer): {sp['welch_t']:+.2f}  (|t|>2 to be strong)")

    sim, bh = st.sell_in_may(ret), st.buy_hold(ret)
    print()
    for nm, s in [("Sell-in-May (cash in summer)", sim), ("Buy & hold", bh)]:
        x = st.summary(s)
        print(f"  {nm:30} CAGR {x['cagr']:+.2%}  Sharpe {x['sharpe']:.2f}  maxDD {x['max_drawdown']:.0%}")

    print("\n  decay (winter vs summer bp/mo):")
    for lab, lo, hi in [("1928-1999", 1900, 1999), ("2000-on", 2000, 2100)]:
        sl = ret[(ret.index.year >= lo) & (ret.index.year <= hi)]
        d = st.seasonal_split(sl)
        print(f"    {lab}: winter {d['winter_bp']:.0f}  summer {d['summer_bp']:.0f}")

    try:
        from quantlab import repro
        print(f"\nas-of {ret.index[-1].date()} · inputs fingerprint {repro.fingerprint(ret.to_frame())}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
