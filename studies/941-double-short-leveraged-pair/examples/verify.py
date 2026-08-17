"""Real-tape verification — Study 941 (Short Both Legs). Regenerates docs/results.md.

Reads cached TQQQ (3x long), SQQQ (−3x inverse), QQQ (index benchmark) and BIL (cash)
daily total-return closes, runs the equal-dollar short of both levered legs at four
rebalance schedules, and prints the excess-of-cash harvest, the HAC *t*, the bootstrap
CIs, the residual beta, the tail, the era cut, the calendar-year table, the cost sweep
and — the crux — the **borrow sweep** with its breakeven rate. Network only on ``--fetch``.

    python studies/941-double-short-leveraged-pair/examples/verify.py            # cache-only
    python studies/941-double-short-leveraged-pair/examples/verify.py --fetch    # refresh
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from short_pair import data, strategy as st  # noqa: E402

LONG, SHORT, BENCH, CASH = "TQQQ", "SQQQ", "QQQ", "BIL"
COST_BPS = 2.0
SPLIT = "2018-01-01"


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices().dropna()
    a, b, q, c = px[LONG], px[SHORT], px[BENCH], px[CASH]

    print(f"{LONG}/{SHORT} vs {CASH}: {px.index[0].date()} -> {px.index[-1].date()}  "
          f"n={len(px)}  fp={data.fingerprint(px)}")
    print(f"as-of {data.AS_OF}  |  daily total-return closes (auto_adjust=True)")

    legs = {t: px[t].pct_change().dropna() for t in (LONG, SHORT, BENCH, CASH)}
    print("\n=== the two legs (annualised, arithmetic) ===")
    for t in (LONG, SHORT, BENCH, CASH):
        r = legs[t]
        print(f"  {t:5s}: mean {r.mean()*252:+.2%}  vol {r.std()*np.sqrt(252):.1%}")
    drag = -0.5 * 252 * (legs[LONG].mean() + legs[SHORT].mean()) + 252 * legs[CASH].mean()
    print(f"  implied average fund shortfall vs a costless +/-3x replication financed at "
          f"{CASH}: {drag:+.2%}/yr")
    print("  NOTE: this is an IDENTITY with the daily-reset harvest below (same algebra),")
    print("        not an independent measurement. It is a RESIDUAL: expense ratios +")
    print("        financing spread + the funds' own internal trading + tracking, none of")
    print("        which is measured separately here.")

    print("\n=== which leg does the harvest come from? ===")
    legs_sf = st.implied_leg_shortfall(a, b, c, bench=q)
    print(f"  {LONG} shortfall vs a costless +3x financed at {CASH}: "
          f"{legs_sf['shortfall_long']:+.2%}/yr")
    print(f"  {SHORT} shortfall vs a costless -3x financed at {CASH}: "
          f"{legs_sf['shortfall_short']:+.2%}/yr")
    print(f"  average (= the pair harvest): {legs_sf['average']:+.2%}/yr   "
          f"share carried by the {LONG} leg: {legs_sf['share_from_long']:.0%}")
    print("  READ THIS AGAINST THE BORROW: each leg pays borrow on its own half of the")
    print("  notional, so each leg has its OWN breakeven equal to its own shortfall. The")
    print(f"  crowded, expensive-to-borrow leg ({SHORT}) is the one with the THINNER margin")
    print(f"  ({legs_sf['shortfall_short']:.2%}/yr); the pair only clears because the cheap-to-")
    print(f"  borrow {LONG} leg ({legs_sf['shortfall_long']:.2%}/yr) subsidises it.")
    print("  (Same residual as above, same caveats: undecomposed, and it is a decomposition")
    print("   of an identity, not a second measurement.)")

    print("\n=== headline: equal-dollar short of both legs, daily reset ===")
    cmp = st.compare(a, b, c, bench=q, rebalance="daily", cost_bps=COST_BPS)
    s = cmp["strategy"]
    print(f"  excess-of-cash: mean {cmp['ann_mean']:+.2%}/yr  Sharpe {cmp['sharpe']:+.2f}  "
          f"HAC t {cmp['t_hac']:+.2f}  vol {s['vol_ann']:.2%}")
    print(f"  max drawdown {cmp['max_drawdown']:+.2%}  worst day {s['worst_day']:+.2%}  "
          f"worst month {s['worst_month']:+.2%}")
    print(f"  skew {s['skew']:+.2f}  excess kurtosis {s['excess_kurtosis']:+.1f}  "
          f"turnover {cmp['turnover_ann']:.1f}x/yr")
    print(f"  residual beta to {BENCH}: {cmp['beta']['beta']:+.4f}  "
          f"alpha {cmp['beta']['alpha_ann']:+.2%}/yr (t={cmp['beta']['t_alpha']:+.2f})  "
          f"R2 {cmp['beta']['r2']:.4f}")
    print(f"  {BENCH} excess-of-cash Sharpe {cmp['bench']['sharpe']:+.2f} "
          f"(the thing this is supposed to be neutral to)")

    print("\n=== is it an accrual or a handful of days? (outlier robustness) ===")
    e = cmp["e_nav"]
    top = e.sort_values(ascending=False)
    print(f"  mean {e.mean()*252:+.2%}/yr   median {e.median()*252:+.2%}/yr   "
          f"share of days > 0: {(e > 0).mean():.1%}")
    print("  (CAVEAT: deleting the best observations and then re-running a t-test is NOT")
    print("   valid inference -- it strips the right tail, so the t MECHANICALLY rises even")
    print("   as the mean falls. Read the mean column; the t's below are diagnostics of how")
    print("   smooth what is left is, not significance tests.)")
    for k in (5, 20, 50):
        kept = e.drop(top.index[:k])
        print(f"  drop the {k:2d} best days ({top.iloc[:k].sum()/e.sum():4.1%} of the total): "
              f"mean {kept.mean()*252:+.2%}/yr  HAC t {st.hac_t(kept):+.2f}")
    srt = e.sort_values()
    cut = int(0.01 * len(e))
    print(f"  1% two-sided trimmed mean: {srt.iloc[cut:len(e)-cut].mean()*252:+.2%}/yr")
    ex20 = e[e.index.year != 2020]
    print(f"  excluding 2020 entirely : {ex20.mean()*252:+.2%}/yr  HAC t {st.hac_t(ex20):+.2f}")

    print("\n=== bootstrap CIs (excess-of-cash, 2000 draws, 21-day blocks) ===")
    ci_s = st.bootstrap_sharpe_ci(cmp["e_nav"], seed=941)
    ci_m = st.bootstrap_mean_ci(cmp["e_nav"], seed=941)
    print(f"  Sharpe {ci_s['sharpe']:+.2f}  95% CI [{ci_s['ci_low']:+.2f}, {ci_s['ci_high']:+.2f}]"
          f"  frac<0 {ci_s['frac_negative']:.3f}")
    print(f"  mean   {ci_m['mean_ann']:+.2%}/yr  95% CI [{ci_m['ci_low']:+.2%}, "
          f"{ci_m['ci_high']:+.2%}]  frac<0 {ci_m['frac_negative']:.3f}")

    print("\n=== rebalance schedule race (the short-gamma dial) ===")
    for row in st.frequency_race(a, b, c, bench=q, cost_bps=COST_BPS):
        print(f"  {row['rebalance']:8s}: mean {row['ann_mean']:+.2%}/yr  Sharpe {row['sharpe']:+.2f}"
              f"  vol {row['vol_ann']:.1%}  t {row['t_hac']:+.2f}  DD {row['max_drawdown']:+.1%}"
              f"  worst month {row['worst_month']:+.1%}  beta {row['beta']:+.3f}"
              f"  max gross {row['max_gross']:.1f}x  ruined={row['ruined']}")

    print("\n=== BORROW SWEEP (the crux - borrow is an ASSUMPTION, not tape) ===")
    for row in st.borrow_sweep(a, b, c, rebalance="daily", cost_bps=COST_BPS):
        print(f"  borrow {row['borrow_pct']:5.1f}%/yr: mean {row['ann_mean']:+.2%}/yr  "
              f"Sharpe {row['sharpe']:+.2f}  HAC t {row['t_hac']:+.2f}")
    be = st.breakeven_borrow(a, b, c, rebalance="daily", cost_bps=COST_BPS)
    print(f"  breakeven blended borrow (daily reset): {be:.2f}%/yr")
    for f in ("weekly", "monthly"):
        print(f"  breakeven blended borrow ({f:7s}): "
              f"{st.breakeven_borrow(a, b, c, rebalance=f, cost_bps=COST_BPS):.2f}%/yr")

    print("\n=== account terms: full short rebate vs no rebate (retail) ===")
    print("  (the rebate is not a SOURCE of the harvest: the two legs' own financing carry")
    print("   averages to +1x cash on the liability side, which the interest on the proceeds")
    print("   exactly offsets. Losing the rebate is a COST of retail terms, not a component")
    print("   of the alpha -- it costs the cash rate itself, "
          f"{legs[CASH].mean()*252:+.2%}/yr on this sample.)")
    for tag, reb in [("full rebate", 1.0), ("no rebate ", 0.0)]:
        k = st.compare(a, b, c, rebalance="daily", cost_bps=COST_BPS, rebate=reb)
        eras = st.era_cut(a, b, c, split=SPLIT, cost_bps=COST_BPS, rebate=reb)
        print(f"  {tag}: mean {k['ann_mean']:+.2%}/yr  Sharpe {k['sharpe']:+.2f}  "
              f"t {k['t_hac']:+.2f}  breakeven borrow "
              f"{st.breakeven_borrow(a, b, c, cost_bps=COST_BPS, rebate=reb):.2f}%/yr")
        for tag2, e in eras.items():
            if e:
                print(f"      {tag2:5s} (n={e['n_days']}): mean {e['ann_mean']:+.2%}/yr  "
                      f"Sharpe {e['sharpe']:+.2f}  t {e['t_hac']:+.2f}")

    print("\n=== cost sweep (one-way bps of traded notional, daily reset) ===")
    for row in st.cost_sweep(a, b, c, rebalance="daily"):
        print(f"  cost {row['cost_bps']:5.1f} bps: mean {row['ann_mean']:+.2%}/yr  "
              f"Sharpe {row['sharpe']:+.2f}  t {row['t_hac']:+.2f}  "
              f"turnover {row['turnover_ann']:.1f}x/yr")

    print("\n=== calendar-year table (daily reset, 2 bps, zero borrow) ===")
    tbl = st.calendar_year_table(a, b, c, cost_bps=COST_BPS)
    for y, row in tbl.iterrows():
        print(f"  {y}: NAV {row['nav_pct']:+6.2f}%  cash {row['cash_pct']:+5.2f}%  "
              f"excess {row['excess_pct']:+5.2f}%  worst day {row['worst_day_pct']:+5.2f}%")

    print("\n=== synthetic control (machinery proof only - never supports the stamp) ===")
    for tag, ss in [("costly funds (planted)", 1.0), ("free funds (null)     ", 0.0)]:
        means, ts = [], []
        for prices, truth in data.synthetic_panel(n_seeds=6, signal_strength=ss):
            d = st.synthetic_detect(prices, cost_bps=0.0)
            means.append(d["ann_mean"]); ts.append(d["t_hac"])
        exp = truth["expected_harvest_ann"]
        print(f"  {tag}: planted {exp:+.2%}/yr  recovered {np.mean(means):+.2%}/yr "
              f"(sd {np.std(means, ddof=1):.2%})  |t| max {max(abs(t) for t in ts):.2f}  "
              f"decay in the legs {truth['decay_long_ann']:.1%} / {truth['decay_short_ann']:.1%} per yr")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
