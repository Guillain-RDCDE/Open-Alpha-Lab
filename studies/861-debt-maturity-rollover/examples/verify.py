"""Reproducible headline run for Study 861 — Debt-Maturity Rollover Risk.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached prices + EDGAR events under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with no
network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import fingerprint  # noqa: E402

from debt_maturity import data, strategy as st  # noqa: E402

print("# Debt-Maturity Rollover Risk — do high-short-term-share firms under-earn?")

if not data.have_real():
    print("(cache miss — fetching prices + EDGAR once; this is the only network step)")
    data.fetch_panel()

px, ev = data.load_real()
print(f"[data] prices {px.shape}  events {ev.shape}  names={ev['ticker'].nunique()}  "
      f"ends {ev['end'].min().date()}..{ev['end'].max().date()}  "
      f"as-of {data.AS_OF}  prices_fp={fingerprint(px)}")

print("\n# PRIMARY — calendar-time tercile long-short (long LOW-share / short HIGH-share)")
print("#   (rank fresh signals monthly, long safe / short rollover-risk, earn next month; one lag)")
ls = st.calendar_ls(px, ev, signal_col="st_share", n_buckets=3, min_names=6,
                    staleness_days=400, long_high=False)
s = st.calendar_ls_stats(ls)
print(f"  n_months={s['n_months']}  avg cross-section={s['avg_n']:.1f}  "
      f"span {ls.index.min().date()}..{ls.index.max().date()}")
print(f"  long-short mean = {s['mean_bps']:+.1f} bps/mo (+{s['ann_pct']:.2f}%/yr gross)  "
      f"one-sample t = {s['t_iid']:+.2f}  Newey-West t = {s['t_nw']:+.2f}")
print(f"  Sharpe = {s['sharpe']:.2f}  hit = {s['hit']*100:.0f}%  "
      f"long {s['long_bps']:+.1f} / short {s['short_bps']:+.1f} bps/mo  turnover {s['avg_turnover']:.3f}")
ls200 = st.calendar_ls(px, ev, signal_col="st_share", staleness_days=200, long_high=False)
s200 = st.calendar_ls_stats(ls200)
print(f"  robustness (staleness 200d): mean {s200['mean_bps']:+.1f} bps, NW t = {s200['t_nw']:+.2f}")
ls_sc = st.calendar_ls(px, ev, signal_col="st_debt_assets", staleness_days=400, long_high=False)
s_sc = st.calendar_ls_stats(ls_sc)
print(f"  asset-scaled signal ((DC+LC)/Assets): mean {s_sc['mean_bps']:+.1f} bps, "
      f"NW t = {s_sc['t_nw']:+.2f}, Sharpe {s_sc['sharpe']:.2f}")
for nb, lbl in [(2, "halves"), (4, "quartiles"), (5, "quintiles")]:
    sb = st.calendar_ls_stats(st.calendar_ls(px, ev, signal_col="st_share",
                                             staleness_days=400, n_buckets=nb, long_high=False))
    print(f"  {lbl:>9} (n_buckets={nb}): NW t = {sb['t_nw']:+.2f}")

print("\n# CROSS-CHECK — pooled event drift (tercile sort by signal, 1-day-lag entry)")
for h in st.HORIZONS:
    es = st.event_summary(px, ev, horizon=h, n_buckets=3, n_draws=10_000, long_high=False)
    print(f"  H={h:>3}d: n={es['n_events']}  high(top) {es['top_mean']*100:+.2f}%  "
          f"low(bot) {es['bot_mean']*100:+.2f}%  low-high {es['ls_mean']*100:+.2f}%  "
          f"t={es['t']:+.2f}  win={es['ls_win']*100:.0f}%  placebo p={es['p_placebo']:.4f}")
bm = st.bucket_means(st.event_drift_frame(px, ev, horizon=63), 3)
print(f"  monotonicity (63d terciles low->high share): {[f'{b*100:+.2f}%' for b in bm]}")

print("\n# ERA CUT AT 2022 — does the penalty bite harder when rates rise?")
e = st.era_split(ls, split="2022-01-01")
print(f"  2009-2021: n={e['early_n']}  mean {e['early_bps']:+.1f} bps  NW t = {e['early_t']:+.2f}")
print(f"  2022-2026: n={e['late_n']}  mean {e['late_bps']:+.1f} bps  NW t = {e['late_t']:+.2f}")
for sp in ["2017-01-01", "2019-01-01"]:
    ee = st.era_split(ls, split=sp)
    print(f"  (alt split {sp}: late n={ee['late_n']} {ee['late_bps']:+.1f} bps NW t={ee['late_t']:+.2f})")

print("\n# TRADABILITY — calendar long-short net of costs + short borrow (stressed)")
for cb, bb in [(10.0, 50.0), (20.0, 100.0), (30.0, 200.0), (50.0, 300.0)]:
    net = st.calendar_ls_net(ls, cost_bps=cb, borrow_bps_ann=bb)
    print(f"  cost={cb:>4.1f} bps, borrow={bb:>5.1f} bps/yr: net {net['net_mean_bps']:+.1f} bps/mo "
          f"(+{net['net_ann_pct']:.2f}%/yr), NW t = {net['net_t_nw']:+.2f}, Sharpe {net['net_sharpe']:.2f}")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
print("  the claim detector must NOT fire on a null world (edge=0) and must recover a planted")
print("  penalty. Null checked over 12 seeds (never a single stream).")
null_ts = []
for s_ in range(12):
    p0, e0 = data.synthetic_panel(edge=0.0, seed=861 + s_)
    null_ts.append(st.synthetic_detect(p0, e0)["t_nw"])
null_ts = np.asarray(null_ts)
print(f"  null (edge=0), 12 seeds: mean NW t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/12 seeds")
p1, e1 = data.synthetic_panel(edge=0.15, seed=861)
pl = st.synthetic_detect(p1, e1)
print(f"  planted edge=0.15 (seed 861): mean {pl['mean_bps']:+.1f} bps/mo, NW t = {pl['t_nw']:+.2f}")
