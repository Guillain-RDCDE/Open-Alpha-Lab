"""Reproduce the real-data headline run (docs/results.md) — S&P 500, 1950–today.

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

from hangover import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    ret = data.fetch_index(fetch=fetch)
    if ret.empty:
        print("No cached real data. Re-run with --fetch (needs network) to download ^GSPC.")
        return
    tbl = st.annual_table(ret)
    print(f"\nS&P 500 January Barometer, {tbl.index.min()}-{tbl.index.max()} ({len(tbl)} years)\n")
    print(st.conditional_means(tbl).round(4).to_string())
    print(f"\nDirectional accuracy: {st.barometer_accuracy(tbl):.1%}  vs  always-predict-up base rate: {st.base_rate(tbl):.1%}")

    bar, bh = st.barometer_returns(tbl), st.buy_hold_roy(tbl)
    for nm, s in [("Buy & hold (Feb-Dec)", bh), ("Barometer (cash if Jan down)", bar)]:
        d = st.summary(s)
        print(f"  {nm:30} CAGR {d['cagr']:+.2%}  Sharpe {d['sharpe']:.2f}  maxDD {d['max_drawdown']:.1%}")

    for lab, sl in [("1950-1972", tbl[tbl.index <= 1972]), ("1973-on", tbl[tbl.index > 1972])]:
        print(f"  decay {lab}: accuracy {st.barometer_accuracy(sl):.1%}, "
              f"jan-down rest-of-year mean {sl[sl['jan'] <= 0]['roy'].mean():+.2%}")

    try:
        from quantlab import repro
        print(f"\nas-of {ret.index[-1].date()} · inputs fingerprint {repro.fingerprint(ret.to_frame())}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
