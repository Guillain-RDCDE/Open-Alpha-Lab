"""Reproducible headline run for Study 635 — Coinbase Premium.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached Coinbase+Binance daily
tape under ``_cache/`` if present (the real-tape numbers), and always runs the
synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from coinbase_premium import data, strategy as st

print("# Coinbase Premium — Coinbase BTC-USD close vs Binance BTCUSDT close, UTC daily")
if data.have_real():
    df = data.load_real()
    print(f"[data] cb+bn closes: {len(df):,} days  {df.index.min().date()} -> "
          f"{df.index.max().date()}  as-of {data.AS_OF}  fingerprint={data.fingerprint(df)}")

    prem = st.premium_bps(df)
    ret = st.log_returns(df)

    print("\n# The premium series (bps; positive = Coinbase rich)")
    print(f"  mean {prem.mean():+.2f}  sd {prem.std(ddof=1):.2f}  "
          f"min {prem.min():+.1f}  max {prem.max():+.1f}  AC(1) {prem.autocorr(1):.3f}")
    r0 = st.hac_slope(prem.to_numpy(), ret.to_numpy(), lags=5)
    print(f"  contemporaneous same-day return corr {r0['corr']:.3f}  (HAC t {r0['t']:+.2f})")
    yt = st.yearly_table(prem)
    for y, row in yt.iterrows():
        print(f"  {y}: mean {row['mean_bps']:+7.2f} bps  sd {row['sd_bps']:7.2f}  "
              f"share>0 {row['pos_share']*100:4.0f}%  (n={int(row['n'])})")
    p26 = prem[prem.index.year == 2026]
    print(f"  2026 so far: mean {p26.mean():+.2f} bps, {int((p26 < 0).sum())}/{len(p26)} "
          f"days NEGATIVE (Coinbase persistently cheap)")

    print("\n# Predictive regressions — forward h-day return on today's premium (HAC t, NW lags h+2)")
    for label, kw in [("LEVEL (raw, the claim as stated)", dict(signal="level")),
                      ("LEVEL winsorized 1%/99% (USDT-depeg robustness)",
                       dict(signal="level", winsor=0.01)),
                      ("1-DAY CHANGE", dict(signal="change"))]:
        t = st.predictive_table(df, **kw)
        print(f"  {label}:")
        for h, row in t.iterrows():
            print(f"    h={h}d: beta {row['beta_bps_per_sd']:+7.2f} bps per +1sd   "
                  f"HAC t {row['hac_t']:+5.2f}   corr {row['corr']:+.3f}   (n={int(row['n'])})")
    lo, hi = prem.quantile(0.01), prem.quantile(0.99)
    print(f"  (winsor bounds: {lo:+.1f} / {hi:+.1f} bps)")

    print("\n# Group split — next-day return when trailing 63d z(premium) > 0 vs <= 0")
    z = st.trailing_z(prem, 63)
    nxt = ret.shift(-1)
    a = nxt[z > 0].dropna()
    b = nxt[z <= 0].dropna()
    print(f"  rich days {a.mean()*1e4:+6.2f} bps/d (n={len(a):,})  vs  cheap days "
          f"{b.mean()*1e4:+6.2f} bps/d (n={len(b):,})   Welch t = "
          f"{st.welch_t(a.to_numpy(), b.to_numpy()):+.2f}")

    print("\n# Timing overlay — long BTC next day iff z63(premium) > 0 (one lag; long-only)")
    for cb in (5.0, 10.0, 25.0):
        ov = st.overlay(df, cost_bps=cb)
        at = st.timing_alpha_t(df, cost_bps=cb)
        print(f"  cost {cb:4.1f} bps: gross {ov['gross_bps_d']:+5.2f} -> net "
              f"{ov['net_bps_d']:+5.2f} bps/d (HAC t {ov['net_t']:+5.2f}, Sharpe "
              f"{ov['net_sharpe']:.2f}, ann {ov['net_ann_pct']:+.1f}%)  |  timing alpha "
              f"{ov['alpha_bps_d']:+5.2f} bps/d (HAC t {at:+5.2f})")
    ov10 = st.overlay(df, cost_bps=10.0)
    print(f"  exposure {ov10['exposure']*100:.1f}% of days, {ov10['turnover_per_yr']:.0f} "
          f"position changes/yr  |  buy & hold {ov10['bh_bps_d']:+.2f} bps/d "
          f"(Sharpe {ov10['bh_sharpe']:.2f}, ann {ov10['bh_ann_pct']:+.1f}%)  [rf = 0 on both legs]")

    print("\n  z-window robustness (cost 10 bps):")
    for w in (20, 63, 126):
        ovw = st.overlay(df, z_window=w, cost_bps=10.0)
        atw = st.timing_alpha_t(df, z_window=w, cost_bps=10.0)
        print(f"    z{w:>3}: net {ovw['net_bps_d']:+5.2f} bps/d (t {ovw['net_t']:+5.2f})  "
              f"alpha {ovw['alpha_bps_d']:+5.2f} bps/d (t {atw:+5.2f})")

    rb = st.random_baseline(df, exposure=ov10["exposure"], cost_bps=10.0, n_seeds=20)
    print(f"  random-timing twin ({rb['n_seeds']} seeds, exposure-matched): net "
          f"{rb['mean_bps_d']:+.2f} +/- {rb['sd_bps_d']:.2f} bps/d, mean HAC t "
          f"{rb['mean_hac_t']:+.2f}")

    print("\n# Eras — premium regime + h=1 predictive t + overlay (z63, 10 bps)")
    eras = [("2017-2019", None, "2019-12-31"), ("2020-2021", "2020-01-01", "2021-12-31"),
            ("2022", "2022-01-01", "2022-12-31"), ("2023+", "2023-01-01", None),
            ("2024+ (ETF era)", "2024-01-01", None)]
    for name, s, e in eras:
        es = st.era_summary(st.slice_df(df, s, e))
        print(f"  {name:<16} n={es['n']:4d}  prem {es['prem_mean_bps']:+7.2f} bps "
              f"(sd {es['prem_sd_bps']:6.2f})  pred beta {es['beta_bps_per_sd']:+7.2f} "
              f"bps/sd  t {es['pred_t']:+5.2f}  |  overlay net {es['net_bps_d']:+6.2f} "
              f"bps/d (t {es['net_t']:+5.2f})  alpha {es['alpha_bps_d']:+6.2f} "
              f"(t {es['alpha_t']:+5.2f})")

    print("\n# Third axis — pre-2023 vs 2023+ slope difference (per-bp slope, HAC SEs)")
    pre = st.slice_df(df, None, "2022-12-31")
    post = st.slice_df(df, "2023-01-01", None)

    def _slope(d):
        p = st.premium_bps(d)
        y = st.forward_return(st.log_returns(d), 1)
        r = st.hac_slope(p.to_numpy(), y.to_numpy(), lags=3)
        return r["beta"], r["beta"] / r["t"]

    b1, se1 = _slope(pre)
    b2, se2 = _slope(post)
    t_diff = (b2 - b1) / float(np.hypot(se1, se2))
    print(f"  pre-2023 slope {b1*1e4:+.3f} bps per bp of premium; 2023+ slope "
          f"{b2*1e4:+.3f} bps per bp ({b2/b1:.0f}x)  |  difference t = {t_diff:+.2f}")
else:
    print("(no _cache/btc_cb_bn_daily.parquet — run data.fetch_daily(fetch=True) once)")

print("\n# Synthetic positive control — deterministic, no network (machinery proof only)")
print("  the HAC-slope detector must stay quiet on a pure-noise premium and light up on a")
print("  planted 'premium leads returns' effect.")
for lead in (0.0, 0.003):
    syn = data.synthetic_world(lead=lead)
    pt = st.predictive_table(syn, horizons=(1,), signal="level")
    print(f"  planted lead={lead:.3f}: h=1 beta {pt['beta_bps_per_sd'].iloc[0]:+7.2f} "
          f"bps/sd   HAC t {pt['hac_t'].iloc[0]:+6.2f}")
