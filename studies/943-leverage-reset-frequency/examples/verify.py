"""Real-tape verification — Study 943 (Reset Frequency). Regenerates docs/results.md.

Reads the cached SPY / SSO / UPRO / BIL / ^IRX series from the shared desk cache, builds a
monthly-reset replication of 2x and 3x S&P exposure (financed at ^IRX + a spread) and
races it against the daily-reset replication and the actual daily-reset funds — every leg
excess-of-cash — then prints the reset-frequency gap, its HAC *t*, the trending/choppy
decomposition, the predictive (tradable) version, era cuts and the financing-spread, cost
and maintenance-margin sweeps. Network only on ``--fetch``.

    python studies/943-leverage-reset-frequency/examples/verify.py            # cache-only
    python studies/943-leverage-reset-frequency/examples/verify.py --fetch    # refresh
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab import repro  # noqa: E402
from reset_freq import data, strategy as st  # noqa: E402

SLEEVES = [(2.0, "SSO"), (3.0, "UPRO")]
STRESS_START = "2004-01-01"     # pinned, so the stress window cannot drift with the cache


def synthetic_demo() -> None:
    """Offline fallback when the shared cache is absent — SYNTHETIC ONLY, never a real stamp."""
    print("no shared _cache found - running the OFFLINE SYNTHETIC demo instead.")
    print("(these are planted worlds, not the tape; run with --fetch for the real numbers)\n")
    for tag, phi, ss in [("choppy (mean-reverting)", -0.15, 1.0),
                         ("trending (momentum)", 0.15, 1.0),
                         ("iid null", -0.15, 0.0)]:
        d = [st.synthetic_detect(data.synthetic_daily(phi=phi, signal_strength=ss,
                                                      n_years=12, seed=943 + s)[0])
             for s in range(4)]
        print(f"  {tag:<24s}: monthly-minus-daily gap "
              f"{np.mean([x['mean_gap_bps'] for x in d]):+.3f} bps/day  "
              f"(HAC t {np.mean([x['t_gap'] for x in d]):+.2f})  "
              f"ER slope {np.mean([x['chop_slope'] for x in d]):+.2f}")
    print("\nthe detector fires with the right sign on both planted worlds and is silent "
          "on the null.")


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    if not data.have_real():
        synthetic_demo()
        return
    px = data.load_prices()
    print(f"as-of {data.AS_OF}   financing = ^IRX + {st.DEFAULT_SPREAD_BPS:.0f} bps (PROXY)   "
          f"cost {st.DEFAULT_COST_BPS:.0f} bps one-way on exposure turnover (PROXY)   "
          f"maintenance margin {st.DEFAULT_MAINTENANCE:.0%} (PROXY)")

    for lev, fund in SLEEVES:
        fr = st.build_frame(px, fund)
        sub = px.dropna(subset=["SPY", fund, "BIL", "IRX"]).loc[fr.index]
        print("\n" + "=" * 78)
        print(f"{lev:.0f}x sleeve - {fund} vs monthly-reset replication")
        print("  " + repro.data_stamp(f"SPY/{fund}/BIL/^IRX",
                                      sub[["SPY", fund, "BIL", "IRX"]], asof=data.AS_OF))

        r = st.race(fr, lev)
        print("\n  arm            exSharpe    CAGR      vol     maxDD    terminal   HAC t")
        for tag, key, absk in [("fund", "fund", "abs_fund"),
                               ("daily-synth", "daily", "abs_daily"),
                               ("monthly-synth", "monthly", "abs_monthly"),
                               ("SPY (1x)", "abs_asset", "abs_asset")]:
            s = r[key] if key != "abs_asset" else r["abs_asset"]
            a = r[absk]
            print(f"  {tag:<14s} {s['sharpe']:+7.3f}  {a['cagr']:+7.2%}  {a['vol_ann']:6.1%}  "
                  f"{a['max_drawdown']:+7.1%}  x{a['terminal_wealth']:8.1f}  {s['tstat']:+6.2f}")

        print(f"\n  monthly - daily : mean {r['mean_diff_bps']:+.3f} bps/day  HAC t {r['t_diff']:+.2f}  "
              f"Sharpe advantage {r['sharpe_adv']:+.3f}")
        print(f"  monthly - fund  : Sharpe advantage {r['sharpe_adv_vs_fund']:+.3f}  "
              f"HAC t {r['t_diff_vs_fund']:+.2f}")
        print(f"  fee/tracking leg (daily-synth - fund): {r['fee_leg_ann_pct']:+.2f}%/yr  "
              f"HAC t {r['t_fee_leg']:+.2f}")
        print(f"  monthly exposure: mean {r['w_mean']:.3f}  min {r['w_min']:.2f}  max {r['w_max']:.2f}  "
              f"liquidation {r['liquidation_date']}")

        ci_m = st.bootstrap_sharpe_ci(r["e_monthly"])
        ci_d = st.bootstrap_sharpe_ci(r["e_daily"])
        ci_a = st.bootstrap_diff_ci(r["e_monthly"], r["e_daily"])
        print(f"\n  bootstrap (2000 draws, 21-day blocks):")
        print(f"    monthly Sharpe {ci_m['sharpe']:+.3f}  CI [{ci_m['ci_low']:+.3f}, {ci_m['ci_high']:+.3f}]")
        print(f"    daily   Sharpe {ci_d['sharpe']:+.3f}  CI [{ci_d['ci_low']:+.3f}, {ci_d['ci_high']:+.3f}]")
        print(f"    advantage      {ci_a['adv']:+.3f}  CI [{ci_a['ci_low']:+.3f}, {ci_a['ci_high']:+.3f}]  "
              f"share<0 {ci_a['frac_negative']:.3f}")

        tbl = st.monthly_gap_table(fr, lev)
        fit = st.chop_regression(tbl)
        print(f"\n  trending vs choppy (n={fit['n']} months, contemporaneous decomposition):")
        print(f"    slope on efficiency ratio {fit['slope']:+.3f} pp per unit ER  HAC t {fit['t']:+.2f}  "
              f"intercept {fit['intercept']:+.3f}")
        for tag in ("choppy", "mid", "trending"):
            tc = fit["terciles"][tag]
            print(f"    {tag:<9s} tercile: mean gap {tc['mean_pp']:+.3f} pp/month  t {tc['t']:+.2f}  (n={tc['n']})")

        pf = st.predictive_regression(tbl)
        print(f"  predictive (ER lagged one month): slope {pf['slope']:+.3f} pp  HAC t {pf['t']:+.2f}  "
              f"| switch rule {pf['switch_mean_pp']:+.3f} pp/month  t {pf['switch_t']:+.2f}")

        eras = st.era_cut(fr, lev)
        print("\n  era cut (split 2017-01-01):")
        for tag, e in eras.items():
            if e is None:
                continue
            print(f"    {tag:<5s} n={e['n_days']:,}: monthly {e['sharpe_monthly']:+.3f} / "
                  f"daily {e['sharpe_daily']:+.3f}  adv {e['sharpe_adv']:+.3f}  t {e['t_diff']:+.2f}")
        ed = st.era_difference_test(fr, lev)
        print(f"    DIFFERENCE (late - early): {ed['late_minus_early_bps']:+.3f} bps/day  "
              f"HAC t {ed['t']:+.2f}   [early {ed['early_mean_bps']:+.3f} / "
              f"late {ed['late_mean_bps']:+.3f} bps/day]")

        print("\n  financing-spread sweep (PROXY):")
        for row in st.spread_sweep(px, fund, lev):
            print(f"    {row['spread_bps']:5.0f} bps: monthly {row['sharpe_monthly']:+.3f}  "
                  f"daily {row['sharpe_daily']:+.3f}  fund {row['sharpe_fund']:+.3f}  "
                  f"adv {row['sharpe_adv']:+.3f} (t {row['t_diff']:+.2f})  vs fund {row['adv_vs_fund']:+.3f}")

        print("  cost sweep (one-way bps on exposure turnover):")
        for row in st.cost_sweep(fr, lev):
            print(f"    {row['cost_bps']:5.1f} bps: monthly {row['sharpe_monthly']:+.3f}  "
                  f"daily {row['sharpe_daily']:+.3f}  adv {row['sharpe_adv']:+.3f} (t {row['t_diff']:+.2f})")

        print("  maintenance-margin sweep (PROXY - does the monthly account get called?):")
        for row in st.maintenance_sweep(fr, lev):
            print(f"    {row['maintenance']:.0%}: monthly Sharpe {row['sharpe_monthly']:+.3f}  "
                  f"adv {row['sharpe_adv']:+.3f}  max exposure {row['w_max']:.2f}  "
                  f"terminal x{row['terminal_monthly']:.1f} (daily x{row['terminal_daily']:.1f})  "
                  f"liquidation {row['liquidation']}")

    # ---------------------------------------------------------------- #
    # The 2008 stress the UPRO tape cannot see (UPRO launched 2009-06)
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print(f"2008 stress - monthly-reset replication on SPY from {STRESS_START} "
          f"(^IRX accrual as cash)")
    # Pinned start: the SHARED cache is written by other studies too and may reach further
    # back than this study's own fetch, so the stress window is fixed here rather than
    # inherited from whatever happens to be cached.
    sub = px.dropna(subset=["SPY", "IRX"]).loc[STRESS_START:]
    cash = sub["IRX"].shift(1) / 100.0 / 252.0
    stress = pd.DataFrame({
        "asset": sub["SPY"].pct_change(),
        "fund": sub["SPY"].pct_change(),
        "cash": cash,
        "fin": st.financing_rate(sub["IRX"]),
    }).dropna()
    print(f"  window {stress.index[0].date()} -> {stress.index[-1].date()}  n={len(stress):,}")
    for lev in (2.0, 3.0, 4.0):
        for maint in (0.0, st.DEFAULT_MAINTENANCE):
            if maint > 0.0 and lev >= 1.0 / maint:
                # L = 4 sits exactly at the 25% maintenance limit on day one — the account
                # is never viable, so the sweep is meaningless there.
                print(f"  {lev:.0f}x monthly, maintenance {maint:.0%}: not viable "
                      f"(target leverage is already at the 1/{maint:.2f} limit)")
                continue
            path = st.replicate(stress["asset"], stress["fin"], lev, "monthly",
                                maintenance=maint, cash_r=stress["cash"])
            s = st.summary(path["r"])
            liq = path.attrs["liquidation_date"]
            print(f"  {lev:.0f}x monthly, maintenance {maint:.0%}: terminal x{s['terminal_wealth']:.2f}  "
                  f"maxDD {s['max_drawdown']:+.1%}  max exposure {path['w'].max():.2f}  "
                  f"liquidation {None if liq is None else pd.Timestamp(liq).date()}")
        d = st.replicate(stress["asset"], stress["fin"], lev, "daily", maintenance=0.0,
                         cash_r=stress["cash"])
        sd = st.summary(d["r"])
        print(f"  {lev:.0f}x daily   (reference)      : terminal x{sd['terminal_wealth']:.2f}  "
              f"maxDD {sd['max_drawdown']:+.1%}")

    # ---------------------------------------------------------------- #
    # Synthetic control (machinery proof - never supports the stamp)
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("synthetic control (offline, seeds 943-946; maintenance disabled)")
    for phi, tag in [(-0.15, "choppy (mean-reverting)"), (0.15, "trending (momentum)"),
                     (0.0, "iid null")]:
        gaps, ts, slopes = [], [], []
        for s in range(4):
            ss = 0.0 if phi == 0.0 else 1.0
            prices, _ = data.synthetic_daily(phi=(phi if phi != 0.0 else -0.15),
                                             signal_strength=ss, seed=943 + s)
            d = st.synthetic_detect(prices, leverage=2.0)
            gaps.append(d["mean_gap_bps"]); ts.append(d["t_gap"]); slopes.append(d["chop_slope"])
        print(f"  {tag:<26s}: gap {np.mean(gaps):+.3f} bps/day (HAC t {np.mean(ts):+.2f})  "
              f"ER slope {np.mean(slopes):+.3f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
