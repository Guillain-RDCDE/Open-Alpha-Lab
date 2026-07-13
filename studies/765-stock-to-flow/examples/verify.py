"""Reproducible headline run for Study 765 — Stock-to-Flow.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached BTC-USD tape under ``_cache/``
(fetching once on a cache miss); the S2F curve is a pure function of the hardcoded issuance
schedule, no network. The synthetic control always runs, no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from stock_to_flow import data, strategy as st  # noqa: E402

print("# Stock-to-Flow — does PlanB's scarcity model survive an honest out-of-sample test?")

sf = data.supply_flow_daily()
print(f"S2F curve: {len(sf):,} daily points {sf.index.min().date()} -> {sf.index.max().date()} "
      "(reconstructed from the issuance schedule — stock exact at every halving, flow = "
      "reward x 144 blocks/day x 365; SF doubles at each halving)")

if not data.have_real():
    print("(cache miss — fetching BTC-USD once)")
    data.fetch_btc(end="2026-07-01")

btc = data.load_btc()
df = data.join_price_sf(btc)
print(data_stamp("BTC-USD daily close", df[["price"]], asof=data.AS_OF))
print(f"  SF today ~= {df['sf'].iloc[-1]:.0f} (stock {df['supply'].iloc[-1]/1e6:.2f}M BTC)")

print("\n# 1 · THE FIT THAT FLATTERS — full-sample vs a plain time trend")
m = df.resample("ME").last().dropna()
fmc = st.fit_s2f(m, use_marketcap=True)
race = st.spurious_trend_race(df)
print(f"  steelman  ln(mcap) ~ ln(SF), monthly full sample : R2 = {fmc['r2']:.4f} (n={fmc['n']})")
print(f"  ln(price) ~ ln(SF), daily full sample            : R2 = {race['r2_sf']:.4f}")
print(f"  ln(price) ~ CALENDAR TIME, daily full sample     : R2 = {race['r2_time']:.4f}")
print(f"  corr( ln(SF), time ) = {race['corr_sf_time']:.4f}  ->  the fit is a clock in disguise")

print(f"\n# 2 · OUT-OF-SAMPLE — coefficients FROZEN at publication ({data.PUBLICATION_DATE})")
oos = st.oos_fit_stats(df, data.PUBLICATION_DATE)
print(f"  train {oos['n_train']} days -> a = {oos['a']:.4f}, b = {oos['b']:.4f}")
print(f"  in-sample  R2 = {oos['r2_in']:.4f}  (log RMSE {oos['rmse_in']:.4f})")
print(f"  OUT-OF-SAMPLE R2 = {oos['r2_oos']:.4f}  (log RMSE {oos['rmse_oos']:.4f})  <- collapse")
a, b = oos["a"], oos["b"]
mp = st.model_price(df, a, b)
print("  predicted vs actual (frozen model):")
for d in ["2021-11-30", "2022-11-30", "2024-12-31", "2026-06-30"]:
    ts = pd.Timestamp(d)
    px = float(df.loc[df.index <= ts, "price"].iloc[-1])
    mo = float(mp.loc[df.index <= ts].iloc[-1])
    print(f"    {d}: actual ${px:>8,.0f}   model ${mo:>9,.0f}   actual/model = {px/mo:.2f}")

print("\n# 3 · IS THE VALUATION RESIDUAL TRADABLE? (residual -> forward return, HAC t)")
resid = st.valuation_residual(df, a, b)
prO = st.predictive_regression(df, resid, window_start=data.PUBLICATION_DATE)
prF = st.predictive_regression(df, resid)
for h in prO.index:
    print(f"  +{h:>3d}d: slope {prO.loc[h,'slope']:+.3f}   HAC t (OOS) {prO.loc[h,'hac_t']:+.2f}   "
          f"(full-sample t {prF.loc[h,'hac_t']:+.2f} — the in-sample fit leaking back in)")

print("\n# 4 · TIMER (long when cheap vs model, else cash) vs BUY-AND-HOLD, net of costs")
for ws, lab in [(data.PUBLICATION_DATE, "out-of-sample"), ("2021-11-01", "post-2021 (broke)")]:
    tb = st.timer_backtest(df, resid, threshold=0.0, cost_bps=10.0, window_start=ws)
    pl = st.random_placebo(df, tb["exposure_pct"], cost_bps=10.0, window_start=ws, n_draws=3000)
    print(f"  [{lab}] {tb['window_start']} -> {tb['window_end']} ({tb['years']:.1f}y, "
          f"exposure {tb['exposure_pct']:.0f}%, {tb['n_switches']} switches)")
    print(f"    timer      : net {tb['net_total_pct']:+8.0f}%  (gross {tb['gross_total_pct']:+.0f}%)  "
          f"Sharpe {tb['net_sharpe']:.2f}")
    print(f"    buy & hold : net {tb['bh_total_pct']:+8.0f}%  "
          f"Sharpe {tb['bh_sharpe']:.2f}")
    print(f"    placebo    : mean {pl['mean_total_pct']:+.0f}%  (p95 {pl['p95_total_pct']:+.0f}%) "
          "at matched exposure")

print("\n# 5 · Synthetic positive control — deterministic, no network")
print("  the detector must read ~0 on a null world (beta=0) and recover a planted valuation")
print("  signal. Null checked over 20 seeds (never a single stream).")
null_ts = np.array([st.synthetic_detect(data.synthetic_world(beta=0.0, seed=765 + s))
                    for s in range(20)])
planted = st.synthetic_detect(data.synthetic_world(beta=0.03, seed=765))
print(f"  null (beta=0), 20 seeds: mean HAC t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
print(f"  planted (beta=0.03, seed 765): HAC t = {planted:+.2f}")
