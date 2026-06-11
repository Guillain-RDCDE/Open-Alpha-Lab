"""Reproduce the real-data headline run (docs/results.md) — oil → equities.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # download daily CL=F + ^GSPC + ^IRX from Yahoo, then run

The sample is pinned to the desk's as-of (quantlab.repro) and the monthly grid is verified hole-free
(fetch_pair asserts it; this script reports the count). Sharpe convention for the race: excess of the
rolled 13-week T-bill (^IRX) on both legs, like-for-like.
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

import pandas as pd  # noqa: E402

from black_gold import data, strategy as st  # noqa: E402
from quantlab import repro  # noqa: E402


def main(fetch: bool) -> None:
    d = data.fetch_pair(fetch=fetch)
    if d.empty:
        print("No cached real data. Re-run with --fetch (needs network).")
        return
    d = repro.as_of(d)

    months = pd.PeriodIndex(d.index, freq="M")
    holes = pd.period_range(months[0], months[-1], freq="M").difference(months)
    reg = st.predict_regression(d["oil"], d["eq"])
    print(f"\nOil → equities, {d.index.min().date()}..{d.index.max().date()} "
          f"({len(d)} months, {len(holes)} interior holes — every calendar month present)\n")
    print(f"  regress eq_t on oil_(t-1): slope {reg['slope']:+.3f}  r {reg['r']:+.3f}  t={reg['tstat']:+.2f}  "
          "(Driesprong predicts slope < 0; |t| > 2 to be real)")

    timing = st.oil_timing(d["oil"], d["eq"], d["tbill"])
    bh = st.buy_hold(d["eq"])
    print(f"\n  in market {st.time_in_market(d['oil']):.0%} of the time; cash leg earns the 13-week "
          "T-bill (^IRX); Sharpe = excess of T-bill, both legs")
    for nm, s in [("oil timing (long if oil fell)", timing), ("buy & hold S&P", bh)]:
        x = st.summary(s, rf=d["tbill"])
        print(f"  {nm:30} CAGR {x['cagr']:+.2%}  Sharpe {x['sharpe']:.2f}  maxDD {x['max_drawdown']:.0%}")

    print("\n  decay (predictive slope / t):")
    for lab, sl in [("2000-2008", d[d.index.year <= 2008]), ("2009-on", d[d.index.year >= 2009])]:
        rr = st.predict_regression(sl["oil"], sl["eq"])
        print(f"    {lab}: slope {rr['slope']:+.3f}  t={rr['tstat']:+.2f}  (n={rr['n']})")

    print(f"\n{repro.data_stamp('CL=F + ^GSPC + ^IRX monthly', d)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
