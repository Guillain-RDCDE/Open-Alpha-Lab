"""Reproducible headline run for Study 832 — High-Yield Credit Momentum.

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

from hy_credit_momentum import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# High-Yield Credit Momentum — does the credit TREND time equity risk-on/off?")

if not data.have_real():
    print("(cache miss — fetching the four-ETF panel once via yfinance)")
    data.fetch()

panel = data.load_panel()
print(f"[data] {panel.shape[1]} ETFs, {len(panel)} rows  "
      f"{panel.index.min().date()} -> {panel.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint={data.fingerprint(panel)}")
print("  SURVIVORSHIP: HYG/IEF/LQD/SPY are all live, continuously-listed ETFs — no "
      "delisting bias on the tape. The Signal-axis caveat is single-regime overfit risk "
      "(one US 2007-2026 credit history), named on the Signal axis.")

L = 126
print(f"\n# THE SIGNAL — credit trend (HYG excess over IEF, {L}d) discriminates day-t "
      f"equity-excess xs = r_SPY - r_IEF?")
s = st.signal_stats(panel, lookback=L)
print(f"  risk-on days ({s['on_frac']:.1%}): xs {s['on_bps']:+.2f} bps  |  risk-off days: "
      f"xs {s['off_bps']:+.2f} bps")
print(f"  difference (on - off): {s['diff_bps']:+.2f} bps/day  NW(10) t = {s['t_nw']:+.2f}  "
      f"Welch t = {s['welch_t']:+.2f}  (n = {s['n_days']})")
s3 = st.signal_stats(panel, lookback=63)
print(f"  cross-check {63}d (3-month) trend: diff {s3['diff_bps']:+.2f} bps  "
      f"NW t = {s3['t_nw']:+.2f}")

print("\n# PLACEBO — circular-shift the risk-on labels (1,000 draws)")
pl = st.placebo_pvalue(panel, L)
z = (pl["obs_bps"] - pl["placebo_mean_bps"]) / pl["placebo_sd_bps"]
print(f"  observed {pl['obs_bps']:+.2f} bps vs placebo mean {pl['placebo_mean_bps']:+.2f} "
      f"(sd {pl['placebo_sd_bps']:.2f}) -> {z:+.1f}sigma, p = {pl['p_value']:.3f} "
      f"over {pl['n_draws']:,} draws")

print("\n# ROBUSTNESS — two eras (split 2017-01-01)")
df = st.signal_frame(panel, L)
for lo, hi, lbl in [("2007-01-01", "2017-01-01", "2007-2016"),
                    ("2017-01-01", "2027-01-01", "2017-2026")]:
    d = df[(df.index >= lo) & (df.index < hi)]
    on = d.loc[d["risk_on"] == 1, "xs"].to_numpy()
    off = d.loc[d["risk_on"] == 0, "xs"].to_numpy()
    ds = st._regime_diff_series(d["risk_on"].to_numpy(), d["xs"].to_numpy())
    print(f"  {lbl}: n={len(d)}  diff {(on.mean() - off.mean()) * 1e4:+.2f} bps  "
          f"NW t = {st.newey_west_t(ds):+.2f}")

print("\n# THE TIMER — SPY when trend>0 else IEF, costed, vs 100%-SPY buy-and-hold")
for cb in (1.0, 5.0):
    t = st.timer_stats(panel, L, cost_bps=cb)
    print(f"  cost={cb:>4.1f} bps/leg ({t['n_switches']} switches): net Sharpe {t['net_sharpe']:.3f} "
          f"vs B&H {t['bh_sharpe']:.3f} | net CAGR {t['net_cagr']*100:+.2f}% vs {t['bh_cagr']*100:+.2f}% "
          f"| net maxDD {t['net_maxdd']*100:.1f}% vs {t['bh_maxdd']*100:.1f}% | active "
          f"{t['active_bps']:+.2f} bps/day (NW t = {t['active_t_nw']:+.2f}), cost drag "
          f"{t['cost_drag_bps_yr']:.1f} bps/yr")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
null_t = np.array([
    st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=832 + s_, n_days=2200), L)["t_nw"]
    for s_ in range(20)
])
print(f"  null (edge=0), 20 seeds: discrimination NW t mean {null_t.mean():+.2f} "
      f"(sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t) >= 2).sum()}/20 seeds")
p1 = data.synthetic_panel(edge=0.006, seed=832, n_days=2200)
d1 = st.synthetic_detect(p1, L)
print(f"  planted (edge=0.006, seed 832): discrimination NW t = {d1['t_nw']:+.2f}, "
      f"Welch t = {d1['welch_t']:+.2f}, active NW t = {d1['active_t_nw']:+.2f}, "
      f"timer Sharpe {d1['net_sharpe']:+.2f} vs B&H {d1['bh_sharpe']:+.2f}")
