"""Reproduce the real-data headline run (docs/results.md) — BAB on a liquid-ETF cross-section.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # download the ETF panel from Yahoo, then run
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

from free_lunch import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    assets, market = data.fetch_etf_panel(fetch=fetch)
    if assets.empty:
        print("No cached real data. Re-run with --fetch (needs network).")
        return

    gross, lev = st.bab_returns(assets, market, return_leverage=True)
    mkt = st.market_monthly(market).loc[gross.index]
    print(f"\nBAB on {assets.shape[1]} ETFs, {gross.index.min().date()}..{gross.index.max().date()} "
          f"({len(gross)} months) — avg long-leg leverage {lev:.2f}x\n")

    def line(nm, s):
        d = st.summary(s)
        print(f"  {nm:34} CAGR {d['cagr']:+.2%}  Sharpe {d['sharpe']:5.2f}  vol {d['vol_ann']:.1%}  maxDD {d['max_drawdown']:.0%}")

    line("BAB gross (free leverage)", gross)
    line("BAB net (fin 3% + borrow 0.5%)", st.bab_returns(assets, market, 0.03, 0.005))
    line("BAB net (fin 5% + borrow 1%)", st.bab_returns(assets, market, 0.05, 0.01))
    line("SPY (the market)", mkt)

    print("\nDecay (gross BAB Sharpe):")
    for lab, sl in [("1999-2012", gross.loc[:"2012"]), ("2013-on", gross.loc["2013":])]:
        print(f"  {lab}: {st.summary(sl)['sharpe']:.2f}")

    try:
        from quantlab import repro
        print(f"\nas-of {gross.index[-1].date()} · inputs fingerprint {repro.fingerprint(assets.join(market))}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
