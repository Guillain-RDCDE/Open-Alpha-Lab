"""Reproduce the real-data headline run (docs/results.md) — natgas winter seasonality.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # download UNG daily data from Yahoo, then run

The sample covers UNG from inception (2007-05) through the latest complete calendar month.
Monthly returns are built from daily closes resampled to month-end; the grid is verified hole-free.
Fingerprinted run in docs/results.md.
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

from natgas_winter import data, strategy as st  # noqa: E402
from quantlab import repro  # noqa: E402


def main(fetch: bool) -> None:
    d = data.fetch_ung(fetch=fetch)
    if d.empty:
        print("No cached real data. Re-run with --fetch (needs network).")
        return
    d = repro.as_of(d)

    months = pd.PeriodIndex(d.index, freq="M")
    holes = pd.period_range(months[0], months[-1], freq="M").difference(months)
    print(f"\nUNG (United States Natural Gas Fund), {d.index.min().date()}..{d.index.max().date()} "
          f"({len(d)} months, {len(holes)} interior holes)\n")

    # winter premium regression
    reg = st.winter_premium(d["ung"])
    print(f"  winter-dummy regression (Oct-Mar = 1, else 0):")
    print(f"    slope  {reg['slope']:+.4f}  t = {reg['tstat']:+.2f}  n = {reg['n']}")
    print(f"    mean winter (Oct-Mar) return: {reg['mean_winter']:+.2%}/month")
    print(f"    mean non-winter (Apr-Sep) return: {reg['mean_nonwinter']:+.2%}/month")
    print(f"    (negative slope = winter is WORSE, not better; |t| < 2 = not even significant)")

    # strategy vs buy-and-hold
    seasonal = st.seasonal_returns(d["ung"])
    bh = st.buy_hold_returns(d["ung"])
    s_seasonal = st.summary(seasonal)
    s_bh = st.summary(bh)
    print(f"\n  seasonal (long Oct-Mar, cash=0 Apr-Sep):")
    print(f"    CAGR {s_seasonal['cagr']:+.2%}  Sharpe {s_seasonal['sharpe']:.2f}  "
          f"maxDD {s_seasonal['max_drawdown']:.0%}")
    print(f"  buy & hold UNG:")
    print(f"    CAGR {s_bh['cagr']:+.2%}  Sharpe {s_bh['sharpe']:.2f}  "
          f"maxDD {s_bh['max_drawdown']:.0%}")

    # sub-period check
    print(f"\n  sub-period winter premium (slope / t-stat):")
    for lab, sl in [("2007-2015", d[d.index.year <= 2015]), ("2016-2026", d[d.index.year >= 2016])]:
        rr = st.winter_premium(sl["ung"])
        print(f"    {lab}: slope {rr['slope']:+.4f}  t = {rr['tstat']:+.2f}  (n={rr['n']})")

    print(f"\n{repro.data_stamp('UNG monthly', d)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
