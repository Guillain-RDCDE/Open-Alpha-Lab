"""Real-tape verification — Study 961 (Which Gold). Regenerates docs/results.md numbers.

Reads the cached daily closes of the five physically-backed gold wrappers (GLD, IAU,
GLDM, SGOL, BAR) plus BIL, measures every wrapper's realised tracking difference against
the equal-weight cohort mean, tests the fee ranking against a permutation null, puts a HAC
*t* and a block bootstrap on the cheapest-minus-priciest spread (daily and monthly),
compounds it over a holding horizon, races the ownership rules excess-of-cash, and prints
the liquidity counterweight and the swept assumptions. Network only on ``--fetch``.

    python studies/961-gold-wrapper-cheapest/examples/verify.py            # cache-only
    python studies/961-gold-wrapper-cheapest/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from which_gold import data, strategy as st  # noqa: E402

COHORT = list(data.COHORT)
CHEAP, DEAR = "GLDM", "GLD"
COST_BPS = 2.0


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()
    coh = data.cohort_frame(px)

    start, end = coh.index[0], coh.index[-1]
    print(f"cohort {'/'.join(COHORT)}: {start.date()} -> {end.date()}  n={len(coh)}  "
          f"fp={data.fingerprint(coh)}")
    print(f"as-of {data.AS_OF}  (partial months dropped)")

    fees_sheet = data.fee_vector(COHORT)
    fees_blend = data.fee_vector(COHORT, blended_over=(start, end))
    print("\npublished fee sheet (ASSUMPTION, bp/yr): "
          + "  ".join(f"{t}={fees_sheet[t]:.2f}" for t in COHORT))
    print("time-blended over the window (ASSUMPTION): "
          + "  ".join(f"{t}={fees_blend[t]:.2f}" for t in COHORT))

    floor = st.measurement_floor(coh, COHORT)
    print(f"\nmeasurement floor: pairwise sd {floor['sd_daily_pair_bp']:.2f} bp/day, "
          f"{floor['sd_month_pair_bp']:.1f} bp/month over {floor['n_months']} months "
          f"-> smallest gap resolvable at |t|=2 is {floor['detectable_bpy']:.1f} bp/yr")

    print("\n=== realised tracking difference vs the equal-weight cohort mean ===")
    print("    (t = naive iid; HAC in brackets is ANTI-conservative on this mean-reverting tape)")
    tbl = st.tracking_table(coh, COHORT, fees=fees_sheet)
    for tk, row in tbl.iterrows():
        print(f"  {tk:5s} fee {row['fee_bps']:5.2f}  monthly {row['td_monthly_bpy']:+7.2f} bp/yr "
              f"(t={row['t_monthly']:+5.2f} [HAC {row['t_monthly_hac']:+5.2f}], "
              f"{int(row['pos_months'])}/{int(row['n_months'])} months up)  "
              f"daily {row['td_daily_bpy']:+7.2f}  endpoint {row['td_endpoint_bpy']:+7.2f}")

    tds = tbl["td_monthly_bpy"].to_numpy()
    for tag, fv in [("published sheet", fees_sheet), ("time-blended", fees_blend)]:
        rt = st.rank_test([fv[t] for t in COHORT], tds)
        pt = st.pass_through([fv[t] for t in COHORT], tds)
        print(f"\nfee-rank test ({tag}): Spearman {rt['spearman']:+.4f}  "
              f"exact permutation p={rt['p_perm']:.4f} (floor {rt['p_floor']:.4f}, "
              f"{rt['n_draws']} draws)  |  pass-through slope {pt['slope']:+.3f}  "
              f"R2 {pt['r2']:.3f}")

    print(f"\n=== headline pair: {CHEAP} - {DEAR} "
          f"(measurement is pure tape; the PAIR was chosen off the fee sheet) ===")
    ps = st.pair_spread(coh, CHEAP, DEAR)
    print(f"  monthly  {ps['spread_bpy']:+.2f} bp/yr  naive t {ps['tstat']:+.2f}  "
          f"[HAC t {ps['t_hac']:+.2f}]  {ps['pos_months']}/{ps['n_months']} months up  "
          f"sd {ps['sd_month_bp']:.1f} bp/mo  acf1 {ps['monthly_acf1']:+.3f}")
    print(f"  iid bootstrap   95% CI [{ps['ci_iid_low']:+.2f}, {ps['ci_iid_high']:+.2f}] bp/yr  "
          f"share<0 {ps['frac_negative_iid']:.4f}   <- the conservative interval")
    print(f"  block bootstrap 95% CI [{ps['ci_low']:+.2f}, {ps['ci_high']:+.2f}] bp/yr  "
          f"share<0 {ps['frac_negative']:.4f}   (3-month blocks; inherits the mean reversion)")
    print(f"  positive in {ps['pos_years']}/{ps['n_years']} calendar years")
    print(f"  daily    {ps['daily_bpy']:+.2f} bp/yr  naive t {ps['t_daily_naive']:+.2f}  "
          f"[HAC t {ps['t_daily']:+.2f}]  (sd {ps['sd_daily_bp']:.2f} bp/day, "
          f"acf1 {ps['daily_acf1']:+.3f} -> mean-reverting premium noise)")
    print(f"  endpoint {ps['endpoint_bpy']:+.2f} bp/yr")
    sweep = st.hac_lag_sweep(ps["monthly"].to_numpy())
    print("  HAC bandwidth sweep (monthly): "
          + "  ".join(f"{r['label'][:4]}{r['lags']}={r['tstat']:+.2f}" for r in sweep))
    print("    -> the HAC t climbs with the bandwidth because the series MEAN-REVERTS; "
          "the naive t is the number the verdict rests on.")

    print("\n=== all ten pairs (monthly spread vs published fee gap, naive t) ===")
    ap = st.all_pairs(coh, COHORT, fees=fees_sheet)
    for pair, row in ap.iterrows():
        print(f"  {pair:10s} fee gap {row['fee_gap_bps']:+6.2f}  realised {row['spread_bpy']:+7.2f} bp/yr "
              f"(t={row['tstat']:+5.2f} [HAC {row['t_hac']:+5.2f}], "
              f"{int(row['pos_months'])}/{int(row['n_months'])})")

    print("\n=== era cut (split 2022-01-01, naive t) ===")
    for tag, e in st.era_cut(coh, CHEAP, DEAR).items():
        if e is None:
            continue
        print(f"  {tag:5s} n={e['n_months']:3d} months  {e['spread_bpy']:+.2f} bp/yr  "
              f"t={e['tstat']:+.2f} [HAC {e['t_hac']:+.2f}]  {e['pos_months']}/{e['n_months']} up")

    print(f"\n=== fee-stable sub-window ({data.FEE_STABLE_FROM} ->, today's sheet holds for all five) ===")
    sub = coh[coh.index >= data.FEE_STABLE_FROM]
    ps_sub = st.pair_spread(sub, CHEAP, DEAR)
    tbl_sub = st.tracking_table(sub, COHORT, fees=fees_sheet)
    rt_sub = st.rank_test([fees_sheet[t] for t in COHORT], tbl_sub["td_monthly_bpy"].to_numpy())
    print(f"  {CHEAP}-{DEAR} {ps_sub['spread_bpy']:+.2f} bp/yr  t={ps_sub['tstat']:+.2f} "
          f"[HAC {ps_sub['t_hac']:+.2f}]  "
          f"{ps_sub['pos_months']}/{ps_sub['n_months']} up  |  Spearman {rt_sub['spearman']:+.4f} "
          f"(p={rt_sub['p_perm']:.4f})")

    print("\n=== what the gap is worth to a buy-and-hold holder ===")
    for row in st.holding_cost(ps["spread_bpy"]):
        print(f"  {row['years']:2d} yr: {row['shortfall_pct']:.3f}% of terminal wealth  "
              f"(${row['per_100k_usd']:,.0f} per $100k)")

    print("\n=== ownership race, excess-of-cash (one execution lag on the chase rule) ===")
    race = st.ownership_race(px, COHORT, data.CASH, fees_sheet, cost_bps=COST_BPS)
    for tag, a in race["arms"].items():
        print(f"  {tag:13s} exSharpe {a['excess_sharpe']:.4f}  CAGR {a['cagr']*100:+.3f}%  "
              f"vol {a['vol_ann']*100:.2f}%")
    print(f"  Sharpe gap cheapest-priciest {race['sharpe_gap_cheap_minus_dear']:+.4f}  "
          f"(no t is quoted ON the Sharpe gap; the t below is on the mean daily return diff)")
    print(f"  daily return diff cheapest-priciest: naive t "
          f"{race['t_cheap_minus_dear_naive']:+.2f} [HAC {race['t_cheap_minus_dear']:+.2f}] "
          f"-> the daily view is powerless either way; the monthly estimator is the evidence")
    print(f"  chase-the-winner vs own-cheapest: naive t "
          f"{race['t_chase_minus_cheapest_naive']:+.2f} [HAC {race['t_chase_minus_cheapest']:+.2f}]  "
          f"({race['n_switches']} switches at {COST_BPS:.0f} bp one-way, filled one session "
          f"after the year-end close)")

    print("\n=== counterweight 1: liquidity (real tape, close x volume) ===")
    if data.have_volume():
        dv = data.load_dollar_volume()
        lq = st.liquidity_table(dv, COHORT, trade_usd=100e6, recent_from="2025-07-01")
        for tk, row in lq.iterrows():
            print(f"  {tk:5s} median ${row['median_dvol_usd_m']:8,.0f}m/day  "
                  f"a $100m ticket = {row['days_of_adv']:.2f} days of ADV")
    else:
        print("  (no dollar-volume cache — run with --fetch)")

    print("\n=== counterweight 2: swept round-trip execution differential (ASSUMPTION) ===")
    for row in st.spread_break_even(ps["spread_bpy"]):
        print(f"  round trip {row['round_trip_bps']:4.1f} bp -> repaid in "
              f"{row['break_even_days']:6.1f} days")

    print("\n=== counterweight 3: switching an appreciated taxable holding (ASSUMPTION) ===")
    for row in st.tax_break_even(ps["spread_bpy"]):
        if row["embedded_gain"] in (0.50, 3.00):
            print(f"  tax {row['tax_rate']:.3f}  embedded gain {row['embedded_gain']:.2f}x  "
                  f"costs {row['cost_of_nav']*100:.2f}% of NAV -> repaid in "
                  f"{row['break_even_years']:.1f} years")

    print("\n=== counterweight 4: the long/short version (borrow swept, ASSUMPTION) ===")
    for row in st.long_short_net(ps["spread_bpy"], cost_bps=COST_BPS):
        print(f"  borrow {row['borrow_bpy']:5.1f} bp/yr -> net {row['net_bpy']:+7.2f} bp/yr  "
              f"{'alive' if row['alive'] else 'dead'}")

    print("\n=== synthetic control (machinery proof - never supports the stamp) ===")
    pl_p, pl_t = data.synthetic_panel(signal_strength=1.0, seed=961)
    pl = st.synthetic_detect(pl_p, pl_t)
    print(f"  planted (gap {pl['planted_gap_bpy']:.1f} bp/yr): recovered "
          f"{pl['pair_spread_bpy']:+.2f} bp/yr (t={pl['pair_t']:+.2f})  "
          f"Spearman {pl['spearman']:+.3f} (p={pl['p_perm']:.4f})  "
          f"pass-through {pl['pass_through_slope']:+.3f} (R2 {pl['pass_through_r2']:.3f})")
    nulls = []
    for s in range(8):
        n_p, n_t = data.synthetic_panel(signal_strength=0.0, seed=961 + s)
        nulls.append(st.synthetic_detect(n_p, n_t))
    sp = np.array([n["pair_spread_bpy"] for n in nulls])
    tt = np.array([n["pair_t"] for n in nulls])
    rh = np.array([n["spearman"] for n in nulls])
    print(f"  null x8: spread mean {sp.mean():+.2f} bp/yr (sd {sp.std(ddof=1):.2f}), "
          f"|t|>=2 in {(np.abs(tt) >= 2).sum()}/8, mean Spearman {rh.mean():+.3f}, "
          f"p<0.05 in {(np.array([n['p_perm'] for n in nulls]) < 0.05).sum()}/8")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
