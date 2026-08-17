"""Real-tape verification — Study 960 (The Unstaked ETF). Regenerates docs/results.md.

Reads the cached ETHA / ETHE / ETH-USD / BIL daily closes, measures the realised
tracking difference of each fund against the coin and against the other fund, adds the
two declared ASSUMPTIONS (sponsor fee, forgone staking yield) into the wrapper-cost
ledger, and prints the bootstrap intervals, the era cut, the assumption sweeps, the
borrow sweep on the capture trade and the excess-of-cash race. Network only on
``--fetch``.

    python studies/960-eth-staking-yield-forgone/examples/verify.py           # cache-only
    python studies/960-eth-staking-yield-forgone/examples/verify.py --fetch   # refresh
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from unstaked import data, strategy as st  # noqa: E402

SPLIT = "2025-07-01"


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()
    etha, ethe = px["ETHA"], px["ETHE"]
    coin, cash = px["ETH-USD"], px["BIL"]

    print(f"ETF-era window {px.index[0].date()} -> {px.index[-1].date()}  n={len(px)}  "
          f"fp={data.fingerprint(px)}")
    print(f"as-of {data.AS_OF}   (ETHA/ETHE: no distributions -> close = total return; "
          f"ETH-USD: PRICE-ONLY, unstaked; BIL: total return)")

    print(f"  measured: ETHA, ETHE only. NOT in this cache, not measured: "
          f"{', '.join(data.COHORT_NOT_MEASURED)} -> the pair used is the cohort's "
          f"cheapest-vs-dearest, i.e. its WIDEST fee gap.")

    print("\n=== 1. Tracking difference vs the coin (ETH-USD, price-only, unstaked) ===")
    tds = {}
    for tk, s in (("ETHA", etha), ("ETHE", ethe)):
        td = st.tracking_difference(s, coin)
        ci = st.bootstrap_td_ci(td["diff"])
        tds[tk] = (td, ci)
        print(f"  {tk} vs ETH-USD: {td['ann_td_pct']:+.2f}%/yr (simple-return diff "
              f"{td['ann_td_geo_pct']:+.2f}%/yr)  95% CI [{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}]")
        print(f"      daily sd {td['daily_sd_bps']:.0f} bps  ac(1) {td['autocorr1']:+.2f}  "
              f"iid t {td['iid_t']:+.2f}  HAC t {td['hac_t']:+.2f}  monthly t {td['monthly_t']:+.2f} "
              f"(n={td['monthly_n']})")
        sw = st.resolution_sweep(td["diff"])
        blocks = "  ".join(f"b{b}:{w:.1f}" for b, w in sw["blocks"].items())
        print(f"      RESOLUTION LIMIT IS RULER-DEPENDENT: {blocks}  monthly:{sw['monthly']:.1f}"
              f"  -> +/-{sw['min']:.1f} to +/-{sw['max']:.1f}%/yr "
              f"(headline block=21: +/-{ci['half_width']:.2f})")

    print("\n=== 2. Tracking difference fund vs fund (same coin, same 16:00 stamp) ===")
    td_ff = st.tracking_difference(ethe, etha)
    ci_ff = st.bootstrap_td_ci(td_ff["diff"])
    fee_spread = (data.EXPENSE_RATIOS["ETHE"] - data.EXPENSE_RATIOS["ETHA"]) * 100.0
    print(f"  ETHE - ETHA: {td_ff['ann_td_pct']:+.2f}%/yr (simple-return diff "
          f"{td_ff['ann_td_geo_pct']:+.2f}%/yr)  95% CI [{ci_ff['ci_low']:+.2f}, {ci_ff['ci_high']:+.2f}]")
    print(f"      daily sd {td_ff['daily_sd_bps']:.0f} bps  iid t {td_ff['iid_t']:+.2f} (OVER-WIDE: "
          f"assumes away the -0.47 reversal)  HAC t {td_ff['hac_t']:+.2f} (flattering direction)  "
          f"monthly t {td_ff['monthly_t']:+.2f} <- the ruler quoted")
    sw_ff = st.resolution_sweep(td_ff["diff"])
    print(f"      ruler sweep: " + "  ".join(f"b{b}:{w:.2f}" for b, w in sw_ff["blocks"].items())
          + f"  monthly:{sw_ff['monthly']:.2f}  -> interval excludes zero on every ruler")
    print(f"      ASSUMED documented fee spread: -{fee_spread:.2f}%/yr  -> "
          f"measured is {abs(td_ff['ann_td_pct']) / fee_spread:.0%} of it")

    print("\n=== 3. Wrapper-cost ledger (measured tape + declared ASSUMPTIONS) ===")
    for tk in ("ETHA", "ETHE"):
        td, _ = tds[tk]
        led = st.wrapper_ledger(td["ann_td_pct"], data.EXPENSE_RATIOS[tk],
                                data.STAKING_YIELD_ANN)
        print(f"  {tk}: measured TD vs unstaked coin {led['td_vs_coin_pct']:+.2f}%/yr  |  "
              f"ASSUMED fee {led['fee_pct_assumed']:.2f}%  ASSUMED staking {led['staking_pct_assumed']:.2f}%")
        print(f"      shortfall vs a SELF-STAKING coin holder: {led['total_vs_staked_holder_pct']:+.2f}%/yr; "
              f"staking is {led['staking_share_of_declared_cost']:.0%} of the declared cost")
        narrowest = st.monthly_halfwidth(td["diff"])
        print(f"      measured TD minus the fee we expected: {led['unexplained_pct']:+.2f}%/yr "
              f"(inside even the NARROWEST resolution limit, +/-{narrowest:.2f}%/yr from the "
              f"non-overlapping monthly ruler)")

    print("\n=== 4. Assumption sweeps (the staking yield is NOT tape) ===")
    td_a = tds["ETHA"][0]
    for row in st.staking_sweep(td_a["ann_td_pct"], data.EXPENSE_RATIOS["ETHA"],
                                yields=data.STAKING_SWEEP):
        print(f"  staking {row['staking_pct_assumed']:.2f}%/yr -> ETHA shortfall vs staked holder "
              f"{row['total_vs_staked_holder_pct']:+.2f}%/yr  "
              f"(staking share {row['staking_share_of_declared_cost']:.0%})")
    for row in st.fee_sweep(td_a["ann_td_pct"], data.STAKING_YIELD_ANN,
                            fees=data.ETHA_FEE_RANGE):
        print(f"  ETHA fee {row['fee_pct_assumed']:.2f}%/yr -> staking share of declared cost "
              f"{row['staking_share_of_declared_cost']:.0%}")

    print(f"\n=== 5. Era cut (split {SPLIT}) ===")
    for label, fund in (("ETHA vs coin", etha), ("ETHE vs coin", ethe)):
        eras = st.era_cut(fund, coin, split=SPLIT)
        for tag, e in eras.items():
            if e is None:
                continue
            print(f"  {label:13s} {tag:5s} {e['start']}->{e['end']} n={e['n_days']:3d}: "
                  f"{e['ann_td_pct']:+.2f}%/yr  CI [{e['ci_low']:+.2f}, {e['ci_high']:+.2f}]  "
                  f"monthly t {e['monthly_t']:+.2f}")
    eras_ff = st.era_cut(ethe, etha, split=SPLIT)
    for tag, e in eras_ff.items():
        if e is None:
            continue
        print(f"  ETHE - ETHA   {tag:5s} {e['start']}->{e['end']} n={e['n_days']:3d}: "
              f"{e['ann_td_pct']:+.2f}%/yr  CI [{e['ci_low']:+.2f}, {e['ci_high']:+.2f}]  "
              f"monthly t {e['monthly_t']:+.2f}")

    print("\n=== 6. Calendar-year tracking difference (ETHE - ETHA, %/yr) ===")
    tbl = st.calendar_year_table(ethe, etha)
    for y, row in tbl.iterrows():
        print(f"  {y}: {row['ann_td_pct']:+.2f}%/yr  (n={int(row['n_days'])} sessions)")

    print("\n=== 7. Is the measured spread bankable? long ETHA / short ETHE ===")
    rows = st.borrow_sweep(etha, ethe, cash)
    r0 = rows[0]
    print(f"  gross {r0['ann_gross_pct']:+.2f}%/yr  monthly t {r0['monthly_t_gross']:+.2f} "
          f"({r0['months_positive']}/{r0['monthly_n']} months positive) <- the ruler quoted; "
          f"HAC t {r0['hac_t_gross']:+.2f} (flattering), iid t {r0['iid_t_gross']:+.2f} (over-wide)")
    for row in rows:
        print(f"  borrow {row['borrow_bps_ann']:5.0f} bps/yr, 10 bps one-way: "
              f"net {row['ann_net_pct']:+.2f}%/yr  "
              f"vol {row['vol_ann_pct']:.2f}%  DD {row['max_drawdown']:+.2%}")

    print("\n=== 8. Excess-of-cash race (both arms minus BIL) ===")
    for tk, s in (("ETHA", etha), ("ETHE", ethe)):
        race = st.excess_sharpe_race(s, coin, cash, cost_bps=5.0)
        print(f"  {tk}: exSharpe {race['fund']['sharpe']:+.3f} vs coin {race['coin']['sharpe']:+.3f}  "
              f"adv {race['excess_sharpe_adv']:+.4f}  HAC t {race['t_diff']:+.2f}  "
              f"(iid t {race['iid_t_diff']:+.2f})  fund vol {race['fund']['vol_ann']:.1%}")

    print("\n=== 9. Synthetic control (machinery proof; never supports the stamp) ===")
    planted, truth = data.synthetic_panel(signal_strength=1.0, seed=960)
    d = st.synthetic_detect(planted)
    print(f"  planted spread {truth['spread_planted_pct']:.2f}%/yr -> fund-vs-fund "
          f"{d['fund_vs_fund_pct']:+.2f}%/yr CI [{d['fund_vs_fund_ci'][0]:+.2f}, "
          f"{d['fund_vs_fund_ci'][1]:+.2f}] (monthly t {d['fund_vs_fund_monthly_t']:+.2f})")
    print(f"  same world, fund-vs-coin: {d['fund_vs_coin_pct']:+.2f}%/yr CI "
          f"[{d['fund_vs_coin_ci'][0]:+.2f}, {d['fund_vs_coin_ci'][1]:+.2f}]  -> "
          f"resolution +/-{d['fund_vs_coin_halfwidth']:.2f}%/yr, {d['fund_vs_coin_halfwidth'] / d['fund_vs_fund_halfwidth']:.0f}x wider")
    nulls = np.array([
        st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=960 + s)[0])["fund_vs_fund_pct"]
        for s in range(8)
    ])
    print(f"  null x8 (identical wrappers): fund-vs-fund mean {nulls.mean():+.3f}%/yr "
          f"(sd {nulls.std(ddof=1):.3f}), |est| >= 0.5%/yr in {(np.abs(nulls) >= 0.5).sum()}/8")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
