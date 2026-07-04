"""Reproducible headline run for Study 633 — BTC Vol Targeting.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached BTC-USD tape under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from btc_vol_targeting import data, strategy as st

try:
    from quantlab.repro import data_stamp
except Exception:                                    # repo root not importable — degrade
    data_stamp = None

print("# BTC Vol Targeting — weight = min(1.5, target/RV30), daily, one-day lag (yfinance BTC-USD)")
if data.have_real():
    px = data.load_btc()
    ret = data.daily_returns()
    if data_stamp is not None:
        print(data_stamp("BTC-USD daily close", px.to_frame(), asof=data.AS_OF))
    print(f"tape           : {len(px)} daily closes  {px.index.min().date()} -> "
          f"{px.index.max().date()}  ({len(ret)/365:.1f} years, calendar-daily bars, ann = 365)")

    print("\n# Headline race — target 30%, window 30d, cap 1.5x, gross (0 bps), rf = 0, cash at 0%")
    r = st.race(ret, target=st.TARGET, window=st.WINDOW)
    for leg, p in (("vol-targeted", r["strat"]), ("buy & hold", r["bh"])):
        print(f"  {leg:<13}: CAGR {p['cagr_pct']:+7.2f}%  vol {p['vol_ann_pct']:6.2f}%  "
              f"Sharpe {p['sharpe']:+.3f}  maxDD {p['maxdd_pct']:+7.2f}%  "
              f"wealth x{p['wealth_mult']:,.1f}")
    print(f"  sample        : {r['n_days']} days  avg weight {r['avg_w']:.3f}  "
          f"levered {r['share_levered']*100:.1f}% of days (capped {r['share_capped']*100:.1f}%)  "
          f"turnover {r['turnover_ann']:.2f}x NAV/yr")
    print(f"  log-growth gap: {r['growth_gap_ann_pct']:+.2f}%/yr (strat - B&H)  "
          f"HAC t = {r['t_growth_gap']:+.2f}")
    print(f"  HAC alpha     : {r['alpha_ann_pct']:+.2f}%/yr  t = {r['t_alpha']:+.2f}  "
          f"beta = {r['beta']:.3f}  appraisal = {r['appraisal']:+.3f}")

    print("\n# Does it actually deliver ~constant 30% vol? (rolling 30d realized vol of the STRATEGY)")
    vt = st.vol_tracking(ret)
    print(f"  strategy roll vol: median {vt['median_roll_vol_pct']:.1f}%  "
          f"p10-p90 {vt['p10_pct']:.1f}%-{vt['p90_pct']:.1f}%  "
          f"in 20-40% band {vt['share_in_band']*100:.1f}% of days")
    print(f"  buy & hold   vol: median {vt['bh_median_roll_vol_pct']:.1f}%  "
          f"p90 {vt['bh_p90_pct']:.1f}%")

    print("\n# The heart-attack ledger — max drawdown inside each crash window")
    ct = st.crash_table(ret)
    for name, v in ct.items():
        print(f"  {name:<13}: vol-targeted {v['strat']:+7.2f}%  vs  B&H {v['bh']:+7.2f}%")

    print("\n# Parameter grid — targets 20/30/50% x windows 20/30/60d (is it parameter-robust?)")
    for g in st.grid(ret):
        print(f"  target {g['target']*100:>3.0f}%  window {g['window']:>2d}d : "
              f"Sharpe {g['sharpe']:+.3f} (BH {g['sharpe_bh']:+.3f})  "
              f"maxDD {g['maxdd_pct']:+7.2f}% (BH {g['maxdd_bh_pct']:+7.2f}%)  "
              f"CAGR {g['cagr_pct']:+6.2f}% (BH {g['cagr_bh_pct']:+6.2f}%)  "
              f"alpha t={g['t_alpha']:+.2f}  growth-gap t={g['t_growth_gap']:+.2f}")

    print("\n# Costs & whipsaw — one-way bps x |dw| x NAV daily + retail borrow spread on max(w-1,0)")
    for cb, bs in ((0.0, 0.0), (5.0, 0.0), (10.0, 0.02), (20.0, 0.05)):
        rc = st.race(ret, cost_bps=cb, borrow_spread_ann=bs)
        print(f"  cost {cb:>4.0f} bps, borrow {bs*100:>3.0f}%: "
              f"Sharpe {rc['strat']['sharpe']:+.3f}  CAGR {rc['strat']['cagr_pct']:+6.2f}%  "
              f"maxDD {rc['strat']['maxdd_pct']:+7.2f}%  alpha {rc['alpha_ann_pct']:+.2f}%/yr "
              f"t={rc['t_alpha']:+.2f}  growth-gap t={rc['t_growth_gap']:+.2f}")

    print("\n# Placebo — shuffled vol signal (200 seeds; same weight distribution, no timing)")
    pl = st.placebo_shuffle(ret, n_seeds=200)
    print(f"  observed alpha {pl['obs_alpha_ann_pct']:+.2f}%/yr (t={pl['obs_t']:+.2f})  "
          f"placebo mean alpha {pl['placebo_mean_alpha']:+.2f}%/yr "
          f"(mean t {pl['placebo_mean_t']:+.2f} +/- {pl['placebo_sd_t']:.2f})")
    print(f"  p(shuffled alpha >= observed) = {pl['p_alpha']:.3f}")
    print(f"  observed maxDD {pl['obs_maxdd_pct']:+.2f}%  placebo mean maxDD "
          f"{pl['placebo_mean_maxdd_pct']:+.2f}%  "
          f"p(shuffled DD as shallow as observed) = {pl['p_dd']:.3f}")

    print("\n# Third axis — the CAGR give-up (does it keep enough upside to matter?)")
    p_s, p_b = r["strat"], r["bh"]
    print(f"  CAGR: vol-targeted {p_s['cagr_pct']:+.2f}%/yr vs B&H {p_b['cagr_pct']:+.2f}%/yr  "
          f"-> give-up {p_b['cagr_pct']-p_s['cagr_pct']:.2f} pp/yr "
          f"({p_s['cagr_pct']/p_b['cagr_pct']*100:.1f}% of B&H CAGR kept)")
    print(f"  terminal wealth: x{p_s['wealth_mult']:,.1f} vs x{p_b['wealth_mult']:,.1f} "
          f"({p_s['wealth_mult']/p_b['wealth_mult']*100:.1f}% of B&H wealth)")
    print(f"  bought with    : maxDD {p_s['maxdd_pct']:+.2f}% vs {p_b['maxdd_pct']:+.2f}%  "
          f"and vol {p_s['vol_ann_pct']:.1f}% vs {p_b['vol_ann_pct']:.1f}%")
else:
    print("(no _cache/btc_usd.csv — run data.fetch_btc() once to build the cache)")

print("\n# Synthetic control — 20 seeds per world, deterministic, no network")
print("  NULL (risk priced, mu_t ~ sig_t^2): vol targeting must NOT earn alpha;")
print("  PLANTED (leverage effect, mean falls as variance rises): it MUST light up.")
for name, dc in (("null (priced)", 0.0), ("planted (disconnect)", 2.0)):
    sc = st.synthetic_check(disconnect=dc, n_seeds=20)
    print(f"  {name:<21}: mean alpha {sc['mean_alpha_ann_pct']:+7.2f}%/yr  "
          f"mean t {sc['mean_t']:+.2f} +/- {sc['sd_t']:.2f}  "
          f"share t>=2: {sc['share_t_ge_2']*100:.0f}%  (n={sc['n_seeds']} seeds)")
