"""Reproducible headline run for Study 865 — Credit → Equity Lead-Lag.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached four-ETF panel under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from credit_lead import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Credit -> Equity Lead-Lag — does the trailing HY-excess return LEAD next-week SPY?")

if not data.have_real():
    print("(cache miss — fetching the four-ETF panel once via yfinance)")
    data.fetch()

panel = data.load_panel()
wp = st.weekly_prices(panel)
print(f"[data] {panel.shape[1]} ETFs, {len(panel)} daily rows -> {len(wp)} weekly closes  "
      f"{wp.index.min().date()} -> {wp.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint={data.fingerprint(panel)}")
print("  SURVIVORSHIP: HYG/IEF/LQD/SPY are all live, continuously-listed ETFs — no "
      "delisting bias on the tape. The Signal-axis caveat is single-regime overfit risk "
      "(one US 2007-2026 credit history), named on the Signal axis.")

K = 4
print(f"\n# THE SIGNAL — predictive regression: r_SPY[t+1] ~ trailing {K}-week HY-excess trend")
r = st.leadlag_regression(panel, lookback_wk=K)
print(f"  slope beta = {r['beta']:+.4f}  (per-1sigma-trend SPY move {r['per_sd_bps']:+.1f} bps)  "
      f"NW(6) t = {r['beta_t_nw']:+.2f}  R2 = {r['r2']*100:.3f}%  corr = {r['corr']:+.3f}  "
      f"(n = {r['n_weeks']} weeks)")
for k in (1, 2):
    rk = st.leadlag_regression(panel, lookback_wk=k)
    print(f"  cross-check {k}-week trend: per-sigma {rk['per_sd_bps']:+.1f} bps  "
          f"NW t = {rk['beta_t_nw']:+.2f}  R2 = {rk['r2']*100:.3f}%")

print(f"\n# DISCRIMINATION — next-week SPY on risk-on (trend>0) vs risk-off weeks ({K}-week)")
s = st.signal_stats(panel, lookback_wk=K)
print(f"  risk-on weeks ({s['on_frac']:.1%}): SPY next {s['on_bps']:+.1f} bps  |  risk-off: "
      f"{s['off_bps']:+.1f} bps")
print(f"  difference (on - off): {s['diff_bps']:+.1f} bps/week  NW(6) t = {s['t_nw']:+.2f}  "
      f"Welch t = {s['welch_t']:+.2f}")

print("\n# PLACEBO — circular-shift the risk-on labels (1,000 draws)")
pl = st.placebo_pvalue(panel, K)
z = (pl["obs_bps"] - pl["placebo_mean_bps"]) / pl["placebo_sd_bps"]
print(f"  observed {pl['obs_bps']:+.1f} bps vs placebo mean {pl['placebo_mean_bps']:+.1f} "
      f"(sd {pl['placebo_sd_bps']:.1f}) -> {z:+.1f}sigma, p = {pl['p_value']:.3f} "
      f"over {pl['n_draws']:,} draws")

print("\n# ROBUSTNESS — two eras (split 2017-01-01), regression NW t on the slope")
df = st.leadlag_frame(panel, K)
for lo, hi, lbl in [("2007-01-01", "2017-01-01", "2007-2016"),
                    ("2017-01-01", "2027-01-01", "2017-2026")]:
    d = df[(df.index >= lo) & (df.index < hi)]
    beta, t_nw, r2 = st.nw_regression(d["trend"].to_numpy(), d["r_spy_next"].to_numpy())
    print(f"  {lbl}: n={len(d)}  beta {beta:+.4f}  NW t = {t_nw:+.2f}  R2 = {r2*100:.3f}%")

print("\n# THE OVERLAY — SPY when trend>0 else IEF, weekly, costed, vs 100%-SPY buy-and-hold")
for cb in (1.0, 5.0):
    t = st.overlay_stats(panel, K, cost_bps=cb)
    print(f"  cost={cb:>4.1f} bps/leg ({t['n_switches']} switches): net Sharpe {t['net_sharpe']:.3f} "
          f"vs B&H {t['bh_sharpe']:.3f} | net CAGR {t['net_cagr']*100:+.2f}% vs {t['bh_cagr']*100:+.2f}% "
          f"| net maxDD {t['net_maxdd']*100:.1f}% vs {t['bh_maxdd']*100:.1f}% | active "
          f"{t['active_bps']:+.2f} bps/wk (NW t = {t['active_t_nw']:+.2f}), cost drag "
          f"{t['cost_drag_bps_yr']:.1f} bps/yr")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
null_t = np.array([
    st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=865 + s_, n_days=3000), K)["beta_t_nw"]
    for s_ in range(20)
])
print(f"  null (edge=0), 20 seeds: regression NW t mean {null_t.mean():+.2f} "
      f"(sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t) >= 2).sum()}/20 seeds")
p1 = data.synthetic_panel(edge=0.02, seed=865, n_days=3000)
d1 = st.synthetic_detect(p1, K)
print(f"  planted (edge=0.02, seed 865): regression NW t = {d1['beta_t_nw']:+.2f}, "
      f"per-sigma {d1['per_sd_bps']:+.1f} bps, R2 = {d1['r2']*100:.3f}%, "
      f"active NW t = {d1['active_t_nw']:+.2f}, overlay Sharpe {d1['net_sharpe']:+.2f} "
      f"vs B&H {d1['bh_sharpe']:+.2f}")
