"""Real-tape verification — Study 912 (Gold + Trend). Regenerates docs/results.md numbers.

Reads cached GLD (gold) and BIL (cash) daily total-return closes, runs the 200-day trend
overlay vs buy-and-hold gold and a random-timing control — every leg excess-of-cash —
and prints the excess-Sharpe advantage, HAC return-difference *t*, bootstrap Sharpe CIs,
drawdowns, the era cut, the calendar-year table and the costed sweep. Network only on
``--fetch``.

    python studies/912-gold-trend-managed/examples/verify.py            # cache-only
    python studies/912-gold-trend-managed/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from gold_trend import data, strategy as st  # noqa: E402

GOLD = "GLD"
CASH = "BIL"
SMA_N = 200
COST_BPS = 5.0


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()
    gold = px[GOLD].dropna()
    cash = px[CASH].dropna()
    common = gold.index.intersection(cash.index)
    gold, cash = gold.loc[common], cash.loc[common]

    print(f"{GOLD} vs {CASH}: {common[0].date()} -> {common[-1].date()}  "
          f"n={len(common)}  fp={data.fingerprint(px[[GOLD, CASH]].loc[common])}")
    print(f"as-of {data.AS_OF}")

    cmp = st.compare(gold, cash, sma_n=SMA_N, cost_bps=COST_BPS)

    def fmt(s):
        return (f"exSharpe={s['sharpe']:+.3f}  CAGR={s['cagr']:+.2%}  "
                f"Vol={s['vol_ann']:.2%}  HAC t={s['tstat']:+.2f}")

    print(f"\nin-gold fraction: {cmp['in_market_frac']:.1%}  |  switches: {cmp['n_switches']}")
    print(f"\nBuy-and-hold gold : {fmt(cmp['bh'])}  MaxDD(abs)={cmp['dd_bh_abs']:+.2%}")
    print(f"Trend overlay     : {fmt(cmp['trend'])}  MaxDD(abs)={cmp['dd_trend_abs']:+.2%}")
    print(f"Random control    : {fmt(cmp['random'])}  MaxDD(abs)={cmp['dd_rand_abs']:+.2%}")
    print(f"\nExcess-Sharpe advantage (trend - BH): {cmp['excess_sharpe_adv']:+.3f}")
    print(f"  HAC t on return diff vs BH    : {cmp['t_diff_vs_bh']:+.2f}")
    print(f"  Excess-Sharpe adv vs random   : {cmp['sharpe_adv_vs_random']:+.3f} (t={cmp['t_diff_vs_random']:+.2f})")
    print(f"  Drawdown improvement vs BH    : {cmp['dd_trend_abs'] - cmp['dd_bh_abs']:+.2%}")

    print("\n=== bootstrap Sharpe CIs (excess-of-cash, 2000 draws, 21-day blocks) ===")
    for tag, e in [("buy-and-hold", cmp["e_bh"]), ("trend", cmp["e_trend"])]:
        ci = st.bootstrap_sharpe_ci(e, seed=912)
        print(f"  {tag:13s}: Sharpe {ci['sharpe']:+.3f}  95% CI [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]  "
              f"frac<0 {ci['frac_negative']:.3f}")

    print("\n=== era cut (split 2016-01-01) ===")
    eras = st.era_cut(gold, cash, split="2016-01-01", sma_n=SMA_N, cost_bps=COST_BPS)
    for tag, e in eras.items():
        if e is None:
            continue
        print(f"  {tag:5s} (n={e['n_days']}): BH exSharpe {e['sharpe_bh']:+.2f} / trend {e['sharpe_trend']:+.2f}"
              f"  adv {e['excess_sharpe_adv']:+.2f} (t={e['t_diff_vs_bh']:+.2f})"
              f"  DD {e['dd_bh_abs']:+.1%}->{e['dd_trend_abs']:+.1%}")

    print("\n=== cost sweep ===")
    for row in st.timer_stats(gold, cash, sma_n=SMA_N):
        print(f"  cost={row['cost_bps']:5.1f}bps  trend exSharpe={row['sharpe_trend']:+.3f}  "
              f"adv={row['excess_sharpe_adv']:+.3f} (t={row['t_diff_vs_bh']:+.2f})  "
              f"DD={row['dd_trend_abs']:+.1%}  switches={row['n_switches']}")

    print("\n=== calendar-year table (BH vs trend, %) ===")
    tbl = st.calendar_year_table(gold, cash, sma_n=SMA_N, cost_bps=COST_BPS)
    for y, row in tbl.iterrows():
        print(f"  {y}: BH {row['bh_pct']:+6.1f}%  trend {row['trend_pct']:+6.1f}%  in-gold {row['in_gold_frac']:.0%}")

    # IAU cross-check (a second gold ETF)
    iau = px["IAU"].dropna()
    ci_common = iau.index.intersection(cash.index)
    cmp_iau = st.compare(iau.loc[ci_common], cash.loc[ci_common], sma_n=SMA_N, cost_bps=COST_BPS)
    print(f"\nIAU cross-check: adv {cmp_iau['excess_sharpe_adv']:+.3f} (t={cmp_iau['t_diff_vs_bh']:+.2f})  "
          f"DD {cmp_iau['dd_bh_abs']:+.1%}->{cmp_iau['dd_trend_abs']:+.1%}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
