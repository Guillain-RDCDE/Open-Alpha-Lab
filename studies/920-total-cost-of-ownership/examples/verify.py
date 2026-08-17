"""Real-tape verification — Study 920 (Total Cost of Ownership). Regenerates docs/results.md.

Reads the cached daily total-return closes of three S&P 500 wrappers (SPY, IVV, VOO), two
Nasdaq-100 wrappers (QQQ, QQQM) and the cash leg (BIL), estimates each pair's realised
tracking difference three ways, prints the break-even holding-period curve against a sweep
of round-trip spread assumptions, walks the empirical holding-period race with one
execution lag, sweeps the borrow rate on the long/short version, cuts the sample into eras,
and finishes with the synthetic control. Network only on ``--fetch``.

    python studies/920-total-cost-of-ownership/examples/verify.py            # cache-only
    python studies/920-total-cost-of-ownership/examples/verify.py --fetch    # refresh
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from tco import data, strategy as st  # noqa: E402

try:
    from quantlab import repro  # noqa: E402
except Exception:  # pragma: no cover - the study is self-sufficient without quantlab
    repro = None

SPREAD_GRID = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
HORIZONS = (5, 10, 21, 42, 63, 126, 189, 252, 378, 504, 756, 1008, 1260)


def synthetic_demo() -> None:
    """Offline fallback when the shared cache is absent — clearly labelled as SYNTHETIC.

    Runs the whole pipeline on a planted 6 bp/yr gap and on the null, so a fresh checkout
    can execute this script and see the machinery work without ever reaching the network.
    Nothing printed here is a real-tape number, and it never supports the study's stamp.
    """
    print("no shared _cache found — running the SYNTHETIC demonstration instead.")
    print("Nothing below is a real-tape number. Run with --fetch to populate the cache.\n")
    for tag, ss in [("planted 6 bp/yr gap", 1.0), ("null (identical wrappers)", 0.0)]:
        prices, _ = data.synthetic_daily(signal_strength=ss, gap_bp_yr=6.0, seed=920)
        est = st.td_estimate(prices["liquid"], prices["cheap"], n_boot=500)
        print(f"[SYNTHETIC] {tag}: TD {est['td_ann_bp_yr']:+.2f} bp/yr "
              f"(n={est['n_years']}y, t {est['t_annual']:+.2f}, "
              f"CI [{est['ci_low']:+.2f}, {est['ci_high']:+.2f}])")
        for s in (0.5, 1.0, 3.0):
            d = st.breakeven_days(est["td_ann_bp_yr"], s)
            print(f"[SYNTHETIC]   break-even at {s:.1f} bp extra round trip: "
                  f"{'never' if not np.isfinite(d) else format(d, ',.0f') + ' trading days'}")
    panel, truths = data.synthetic_panel()
    print("\n[SYNTHETIC] panel calibration (planted -> recovered, bp/yr):")
    for k, row in st.panel_calibration(panel, truths, n_boot=300).iterrows():
        print(f"[SYNTHETIC]   {k:12s} {row['planted_bp_yr']:+6.2f} -> {row['recovered_bp_yr']:+6.2f}"
              f"  (t {row['t_annual']:+.2f})")


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    if not data.have_real():
        synthetic_demo()
        return
    px = data.load_prices()

    print(f"as-of {data.AS_OF}   cache {data.DEFAULT_CACHE}")
    if repro is not None:
        print(repro.data_stamp("wrappers", px.dropna(how="all"), asof=data.AS_OF))
    print(f"fingerprint(all) = {data.fingerprint(px)}")
    for tk in data.USABLE:
        s = px[tk].dropna()
        print(f"  {tk:5s} {s.index[0].date()} -> {s.index[-1].date()}  n={len(s):5d}  "
              f"stated ER {data.STATED_ER_BPS.get(tk, float('nan')):.2f} bp/yr (ASSUMPTION)")
    print("  SPLG   EXCLUDED — Yahoo history reset to a single session at build time")

    ests = {}
    print("\n=== realised tracking difference (cheap minus liquid, bp/yr, total return) ===")
    for liquid, cheap, label in data.PAIRS:
        l, c = px[liquid].dropna(), px[cheap].dropna()
        est = st.td_estimate(l, c, n_boot=2000, seed=920)
        ests[(liquid, cheap)] = est
        gap = st.stated_fee_gap_bps(liquid, cheap, data.STATED_ER_BPS)
        print(f"\n{cheap} over {liquid}  [{label}]")
        print(f"  window {est['start'].date()} -> {est['end'].date()}  n={est['n_days']}  "
              f"fp={data.fingerprint(px[[liquid, cheap]].dropna())}")
        print(f"  stated fee gap (PROSPECTUS ASSUMPTION) : {gap:+.2f} bp/yr")
        print(f"  realised TD, cumulative                : {est['td_cum_bp_yr']:+.2f} bp/yr")
        print(f"  realised TD, mean of {est['n_years']:2d} full years    : {est['td_ann_bp_yr']:+.2f} bp/yr  "
              f"(sd {est['td_ann_sd_bp']:.2f}, t = {est['t_annual']:+.2f})")
        print(f"  realised TD, {est['n_months']:3d} months, HAC t       : {est['td_mon_bp_yr']:+.2f} bp/yr  "
              f"(t = {est['t_monthly_hac']:+.2f})")
        print(f"  robust: median {est['td_ann_median_bp']:+.2f}, trimmed mean {est['td_ann_trimmed_bp']:+.2f} "
              f"(t {est['t_trimmed']:+.2f})  worst year {est['worst_year'][0]}: {est['worst_year'][1]:+.1f} bp")
        print(f"  bootstrap 95% CI (annual, block 2)     : [{est['ci_low']:+.2f}, {est['ci_high']:+.2f}] bp/yr  "
              f"(share <= 0: {est['frac_le_zero']:.1%})  <- TOO NARROW at small n")
        print(f"  Student-t 95% CI (df={est['n_years'] - 1:2d}) HONEST     : "
              f"[{est['t_ci_low']:+.2f}, {est['t_ci_high']:+.2f}] bp/yr  "
              f"(years positive {est['years_positive']}/{est['years_counted']})")
        sd_ = st.stub_decomposition(l, c)
        print(f"  stub decomposition (bp of total)       : head {sd_['head_bp']:+.1f} + "
              f"complete years {sd_['years_bp']:+.1f} + tail {sd_['tail_bp']:+.1f} = "
              f"{sd_['total_bp']:+.1f}  [the annual estimator keeps the middle term only]")
        yrs = est["years_to_detect"]
        print(f"  years of tape needed for |t| = 2       : "
              f"{'infinite' if not np.isfinite(yrs) else f'{yrs:.0f}'}  (have {est['n_years']})")
        print("  per-year TD (bp): " + "  ".join(f"{y}:{v:+.1f}" for y, v in est["annual"].items()))

    print("\n=== break-even holding period vs the round-trip spread differential (SWEPT ASSUMPTION) ===")
    for liquid, cheap, _ in data.PAIRS:
        est = ests[(liquid, cheap)]
        curve = st.breakeven_curve(est, SPREAD_GRID)
        print(f"\n{cheap} vs {liquid}   (TD {est['td_ann_bp_yr']:+.2f} bp/yr, "
              f"CI [{est['ci_low']:+.2f}, {est['ci_high']:+.2f}])")
        print(f"  Student-t CI [{est['t_ci_low']:+.2f}, {est['t_ci_high']:+.2f}] "
              f"is the one quoted for pessimistic cases")
        print("  d_spread(bp)   days@point   months@point   days@boot-low   days@t-low   days@t-high")
        for s, row in curve.iterrows():
            def f(x):
                return "never" if not np.isfinite(x) else format(x, ",.0f")
            mo = "n/a" if not np.isfinite(row["months_point"]) else format(row["months_point"], ".1f")
            print(f"  {s:11.1f}   {f(row['days_point']):>10s}   {mo:>12s}   "
                  f"{f(row['days_ci_low_td']):>13s}   {f(row['days_t_low_td']):>10s}   "
                  f"{f(row['days_t_high_td']):>11s}")
        assumed = data.ASSUMED_RT_SPREAD_BPS[cheap] - data.ASSUMED_RT_SPREAD_BPS[liquid]
        print(f"  at the study's indicative differential ({assumed:+.1f} bp): "
              f"{st.breakeven_days(est['td_ann_bp_yr'], assumed):,.0f} trading days")

    print("\n=== empirical holding-period race (one execution lag, entry at close t+1) ===")
    for liquid, cheap, _ in data.PAIRS:
        l, c = px[liquid].dropna(), px[cheap].dropna()
        for dsp in (1.0, 3.0):
            emp = st.empirical_breakeven(l, c, delta_rt_spread_bps=dsp, horizons=HORIZONS)
            print(f"\n{cheap} vs {liquid}, extra round trip {dsp:.1f} bp")
            print("   hold(d)   windows   mean edge(bp)   win rate [95% Wilson]   HAC t")
            for h, row in emp["table"].iterrows():
                print(f"   {h:7d}   {int(row['n_windows']):7d}   {row['mean_edge_bp']:+13.2f}   "
                      f"{row['win_rate']:8.1%} [{row['wilson_low']:.2f},{row['wilson_high']:.2f}]   "
                      f"{row['t_hac']:+6.2f}")
            print(f"   first positive: {emp['first_positive_days']}d   "
                  f"first majority: {emp['first_majority_days']}d   "
                  f"first |t|>=2: {emp['first_significant_days']}d")

    print("\n=== the long/short version — borrow sweep (ASSUMPTION, no tape borrow curve) ===")
    for liquid, cheap, _ in data.PAIRS:
        l, c = px[liquid].dropna(), px[cheap].dropna()
        sw = st.borrow_sweep(l, c, borrow_grid=(0.0, 10.0, 25.0, 50.0, 100.0), cost_bps_one_way=1.0)
        print(f"\nlong {cheap} / short {liquid}  (gross {sw['gross_bp_yr'].iloc[0]:+.2f} bp/yr, "
              f"vol {sw['vol_bp_yr'].iloc[0]:.0f} bp/yr)")
        for b, row in sw.iterrows():
            print(f"   borrow {b:6.1f} bp/yr -> net {row['net_bp_yr']:+7.2f} bp/yr  "
                  f"Sharpe {row['sharpe']:+.2f}  HAC t {row['t_hac']:+.2f}")

    print("\n=== excess-of-cash Sharpe race (BIL) — the wrong instrument, shown anyway ===")
    for liquid, cheap, _ in data.PAIRS:
        r = st.excess_sharpe_race(px[liquid].dropna(), px[cheap].dropna(), px["BIL"].dropna())
        print(f"  {cheap} vs {liquid}: exSharpe {r['sharpe_cheap']:+.4f} vs {r['sharpe_liquid']:+.4f}  "
              f"diff {r['sharpe_diff']:+.4f}  (HAC t on return diff {r['t_diff_hac']:+.2f}, "
              f"arm vol {r['vol_liquid_pct']:.1f}%/yr, n={r['n_days']})")

    print("\n=== era cut ===")
    for liquid, cheap, _ in data.PAIRS:
        l, c = px[liquid].dropna(), px[cheap].dropna()
        common = l.index.intersection(c.index)
        split = "2020-01-01" if common[0] < pd.Timestamp("2016-01-01") else "2023-01-01"
        eras = st.era_cut(l, c, split=split, n_boot=500)
        print(f"  {cheap} vs {liquid} (split {split}):")
        for tag, e in eras.items():
            if e is None:
                print(f"    {tag:5s}: too short")
                continue
            print(f"    {tag:5s} {e['start'].date()}->{e['end'].date()} n={e['n_days']:5d}: "
                  f"TD {e['td_ann_bp_yr']:+.2f} bp/yr (t {e['t_annual']:+.2f}, "
                  f"CI [{e['ci_low']:+.2f}, {e['ci_high']:+.2f}])")

    common_start = str(px["QQQM"].dropna().index[0].date())
    print(f"\n=== the common window ({common_start} ->), the only span on which all six "
          f"wrappers trade — chosen by QQQM's inception, not by any result ===")
    print("  EVERY headline in docs/results.md comes from this block.")
    for liquid, cheap, _ in data.PAIRS:
        l = px[liquid].dropna().loc[common_start:]
        c = px[cheap].dropna().loc[common_start:]
        est = st.td_estimate(l, c, n_boot=2000, seed=920)
        gap = st.stated_fee_gap_bps(liquid, cheap, data.STATED_ER_BPS)
        emp = st.empirical_breakeven(l, c, delta_rt_spread_bps=1.0, horizons=HORIZONS)
        sd_ = st.stub_decomposition(l, c)
        print(f"\n  {cheap} vs {liquid}: stated {gap:+.2f}  realised {est['td_ann_bp_yr']:+.2f} bp/yr "
              f"(n={est['n_years']}y, sd {est['td_ann_sd_bp']:.2f}, t {est['t_annual']:+.2f}, "
              f"{est['years_positive']}/{est['years_counted']} years positive)")
        print(f"     bootstrap CI [{est['ci_low']:+.2f}, {est['ci_high']:+.2f}] (too narrow)   "
              f"Student-t CI [{est['t_ci_low']:+.2f}, {est['t_ci_high']:+.2f}] (honest)")
        print(f"     the other two estimators of the SAME number: cumulative "
              f"{est['td_cum_bp_yr']:+.2f}, monthly x12 {est['td_mon_bp_yr']:+.2f} bp/yr")
        print(f"     stubs the annual estimator drops: head {sd_['head_bp']:+.1f} bp, "
              f"tail {sd_['tail_bp']:+.1f} bp (vs {sd_['years_bp']:+.1f} bp over "
              f"{sd_['n_years']} complete years)")
        print(f"     break-even @1bp: analytic {st.breakeven_days(est['td_ann_bp_yr'], 1.0):,.0f}d / "
              f"empirical {emp['first_positive_days']}d (|t|>=2 at {emp['first_significant_days']}d) / "
              f"pessimistic (t-CI low) "
              f"{'never' if not np.isfinite(st.breakeven_days(est['t_ci_low'], 1.0)) else format(st.breakeven_days(est['t_ci_low'], 1.0), ',.0f') + 'd'}")

    print("\n--- common window: the overlapping race, 1 bp extra round trip ---")
    print("  (HAC t past 252d is bandwidth-inflated — the 63d row is the significance claim)")
    for liquid, cheap, _ in data.PAIRS:
        l = px[liquid].dropna().loc[common_start:]
        c = px[cheap].dropna().loc[common_start:]
        tbl = st.holding_period_table(l, c, (21, 63, 126, 252, 504, 756, 1008), 1.0)
        print(f"  {cheap} over {liquid}: " + "  ".join(
            f"{h}d {r['mean_edge_bp']:+.2f}({r['win_rate']:.0%},t{r['t_hac']:+.2f})"
            for h, r in tbl.iterrows()))

    print("\n--- common window: the long/short harvest, 1 bp one-way on four legs ---")
    print("  (the PLACEBO row is the control: it has no fee gap to harvest)")
    for liquid, cheap, _ in data.PAIRS:
        l = px[liquid].dropna().loc[common_start:]
        c = px[cheap].dropna().loc[common_start:]
        sw = st.borrow_sweep(l, c, borrow_grid=(0.0, 5.0, 10.0, 25.0), cost_bps_one_way=1.0)
        tag = "  <- PLACEBO" if (liquid, cheap) == ("IVV", "VOO") else ""
        print(f"  long {cheap} / short {liquid}: gross {sw['gross_bp_yr'].iloc[0]:+.2f} bp/yr, "
              f"noise {sw['vol_bp_yr'].iloc[0]:.0f} bp/yr, net @5bp {sw['net_bp_yr'].iloc[1]:+.2f}, "
              f"@10bp {sw['net_bp_yr'].iloc[2]:+.2f}, @25bp {sw['net_bp_yr'].iloc[3]:+.2f}{tag}")

    print("\n--- common window: excess-of-cash Sharpe race (BIL, both legs) ---")
    for liquid, cheap, _ in data.PAIRS:
        r = st.excess_sharpe_race(px[liquid].dropna().loc[common_start:],
                                  px[cheap].dropna().loc[common_start:],
                                  px["BIL"].dropna().loc[common_start:])
        print(f"  {cheap} vs {liquid}: exSharpe {r['sharpe_cheap']:+.4f} vs {r['sharpe_liquid']:+.4f}  "
              f"diff {r['sharpe_diff']:+.4f}  (HAC t {r['t_diff_hac']:+.2f}, "
              f"arm vol {r['vol_liquid_pct']:.1f}%/yr, n={r['n_days']})")

    print("\n=== synthetic control (machinery proof — never supports the stamp) ===")
    planted, _ = data.synthetic_daily(signal_strength=1.0, gap_bp_yr=6.0, seed=920)
    null, _ = data.synthetic_daily(signal_strength=0.0, gap_bp_yr=6.0, seed=920)
    dp = st.synthetic_detect(planted, n_boot=1000)
    dn = st.synthetic_detect(null, n_boot=1000)
    print(f"  planted 6.0 bp/yr: recovered {dp['td_ann_bp_yr']:+.2f} bp/yr "
          f"(t {dp['t_annual']:+.2f}, CI [{dp['ci_low']:+.2f}, {dp['ci_high']:+.2f}])")
    print(f"  null    0.0 bp/yr: recovered {dn['td_ann_bp_yr']:+.2f} bp/yr "
          f"(t {dn['t_annual']:+.2f}, CI [{dn['ci_low']:+.2f}, {dn['ci_high']:+.2f}])")
    nulls = np.array([
        st.synthetic_detect(data.synthetic_daily(signal_strength=0.0, seed=920 + s)[0], n_boot=200)["t_annual"]
        for s in range(8)
    ])
    print(f"  null x8 seeds: mean t {nulls.mean():+.2f} (sd {nulls.std(ddof=1):.2f}), "
          f"|t|>=2 on {(np.abs(nulls) >= 2).sum()}/8")
    panel, truths = data.synthetic_panel()
    cal = st.panel_calibration(panel, truths, n_boot=300)
    print("  panel calibration (planted -> recovered, bp/yr):")
    for k, row in cal.iterrows():
        print(f"    {k:12s} {row['planted_bp_yr']:+6.2f} -> {row['recovered_bp_yr']:+6.2f}  "
              f"(t {row['t_annual']:+.2f})")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
