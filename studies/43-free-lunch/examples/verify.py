"""Reproduce the real-data headline run (docs/results.md) — BAB on a liquid-ETF cross-section.

    python examples/verify.py            # cache-only (offline)
    python examples/verify.py --fetch    # download the ETF panel + ^IRX from Yahoo, then run

The sample is pinned to the desk's as-of (quantlab.repro) on complete months only. Sharpe convention
everywhere: **excess of the 13-week T-bill (^IRX)**, BAB and market alike — the book is self-financed
(borrowed slice at rf + spread, short rebate at rf − fee, every dollar charged once), so its return is
already excess-of-cash; SPY gets the same treatment via ``summary(rf=...)``.
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

from free_lunch import data, strategy as st  # noqa: E402
from quantlab import repro  # noqa: E402


def main(fetch: bool) -> None:
    assets, market, rf_ann = data.fetch_etf_panel(fetch=fetch)
    if assets.empty:
        print("No cached real data. Re-run with --fetch (needs network).")
        return
    cut = pd.Timestamp(repro.DEFAULT_AS_OF)   # complete months strictly before the desk as-of
    assets, market, rf_ann = assets[assets.index < cut], market[market.index < cut], rf_ann[rf_ann.index < cut]
    rf_m = data.monthly_rf(rf_ann)

    gross, lev = st.bab_returns(assets, market, rf_m, return_leverage=True)
    mkt = st.market_monthly(market).loc[gross.index]
    rf = rf_m.reindex(gross.index).fillna(0.0)
    print(f"\nBAB on {assets.shape[1]} ETFs, {gross.index.min().date()}..{gross.index.max().date()} "
          f"({len(gross)} months) — avg long-leg leverage {lev:.2f}x, avg T-bill {rf.mean() * 12:.1%}/yr\n")

    def line(nm, s, srf):
        d = st.summary(s, rf=srf)
        print(f"  {nm:44} CAGR {d['cagr']:+.2%}  Sharpe {d['sharpe']:5.2f}  vol {d['vol_ann']:.1%}  maxDD {d['max_drawdown']:.0%}")

    print("Excess-of-cash race (all Sharpes vs the same T-bill; BAB rows are the funded book, rf + excess):")
    line("BAB gross (self-financed, zero spreads)", gross + rf, rf)
    line("BAB net (spread 1% + borrow fee 0.25%)", st.bab_returns(assets, market, rf_m, 0.01, 0.0025) + rf, rf)
    line("BAB net (spread 2.5% + borrow fee 0.5%)", st.bab_returns(assets, market, rf_m, 0.025, 0.005) + rf, rf)
    line("SPY (the market)", mkt, rf)

    print("\nFinancing-spread sweep (net Sharpe, borrow fee = spread/4):")
    for sp in (0.0, 0.01, 0.02, 0.03, 0.04):
        net = st.bab_returns(assets, market, rf_m, sp, sp / 4)
        print(f"  spread {sp:.1%}: {st.summary(net)['sharpe']:+.2f}")

    print("\nWhat the long leg actually is (the duration confound):")
    tlt = (1.0 + assets["TLT"]).resample("ME").prod() - 1.0
    eq_gross, eq_lev = st.bab_returns(assets[data.EQUITY_ONLY], market, rf_m, return_leverage=True)
    print(f"  corr(BAB gross, TLT) = {gross.corr(tlt.loc[gross.index]):+.2f}   "
          f"corr(BAB gross, SPY) = {gross.corr(mkt):+.2f}")
    print(f"  intra-equity book (drop TLT/IEF/GLD): gross Sharpe {st.summary(eq_gross)['sharpe']:.2f}, "
          f"leverage {eq_lev:.2f}x, corr with TLT {eq_gross.corr(tlt.loc[eq_gross.index]):+.2f}")

    print("\nSub-period gross Sharpe (full book vs intra-equity):")
    for lab, lo, hi in [("2000-2012", None, "2012"), ("2013-on", "2013", None)]:
        s_all = st.summary(gross.loc[lo:hi])["sharpe"]
        s_eq = st.summary(eq_gross.loc[lo:hi])["sharpe"]
        print(f"  {lab}: full {s_all:.2f}   intra-equity {s_eq:.2f}")

    stamp_df = assets.join(market).join(rf_ann)
    print(f"\n{repro.data_stamp('13 ETFs + SPY + ^IRX daily', stamp_df)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
