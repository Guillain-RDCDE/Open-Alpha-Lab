"""Reproducible headline run for Study 858 — Margin ÷ Inventory Divergence.

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

from margin_inventory import data, strategy as st  # noqa: E402

print("# Margin / Inventory Divergence — does the Abarbanell-Bushee contradiction sort returns?")

if not data.have_real():
    print("(cache miss — fetching prices + EDGAR once; this is the only network step)")
    data.fetch_panel()

px, ev = data.load_real()
print(f"[data] prices {px.shape}  events {ev.shape}  names={ev['ticker'].nunique()}  "
      f"ends {ev['end'].min().date()}..{ev['end'].max().date()}  "
      f"as-of {data.AS_OF}  prices_fp={fingerprint(px)}")

print("\n# PRIMARY — calendar-time tercile long-short on the divergence signal")
print("#   (rank fresh signals monthly, long clean/top / short contradictory/bottom, +1 lag)")
ls = st.calendar_ls(px, ev, signal_col="divergence", n_buckets=3, min_names=6, staleness_days=200)
s = st.calendar_ls_stats(ls)
print(f"  n_months={s['n_months']}  avg cross-section={s['avg_n']:.1f}  "
      f"span {ls.index.min().date()}..{ls.index.max().date()}")
print(f"  long-short mean = {s['mean_bps']:+.1f} bps/mo ({s['ann_pct']:+.2f}%/yr gross)  "
      f"one-sample t = {s['t_iid']:+.2f}  Newey-West t = {s['t_nw']:+.2f}")
print(f"  Sharpe = {s['sharpe']:.2f}  hit = {s['hit']*100:.0f}%  "
      f"long {s['long_bps']:+.1f} / short {s['short_bps']:+.1f} bps/mo  turnover {s['avg_turnover']:.2f}")
ls120 = st.calendar_ls(px, ev, signal_col="divergence", n_buckets=3, min_names=6, staleness_days=120)
s120 = st.calendar_ls_stats(ls120)
print(f"  robustness (staleness 120d): mean {s120['mean_bps']:+.1f} bps, NW t = {s120['t_nw']:+.2f}")
ls_gap = st.calendar_ls(px, ev, signal_col="inv_sales_gap", n_buckets=3, min_names=6, staleness_days=200)
s_gap = st.calendar_ls_stats(ls_gap)
print(f"  classic inv-vs-sales gap alone (long LOW gap): mean {-s_gap['mean_bps']:+.1f} bps, "
      f"NW t = {-s_gap['t_nw']:+.2f}")

print("\n# CROSS-CHECK — pooled event drift (tercile sort by divergence, 1-day-lag entry)")
for h in st.HORIZONS:
    es = st.event_summary(px, ev, horizon=h, n_buckets=3, n_draws=10_000)
    print(f"  H={h:>3}d: n={es['n_events']}  top {es['top_mean']*100:+.2f}%  bot {es['bot_mean']*100:+.2f}%  "
          f"long-short {es['ls_mean']*100:+.2f}%  t={es['t']:+.2f}  win={es['ls_win']*100:.0f}%  "
          f"placebo p={es['p_placebo']:.4f}")
bm = st.bucket_means(st.event_drift_frame(px, ev, horizon=63), 3)
print(f"  monotonicity (63d terciles low->high): {[f'{b*100:+.2f}%' for b in bm]}")

print("\n# ERA SPLIT — calendar long-short pre/post 2016")
for lbl, lo, hi in [("pre-2016", "2000-01-01", data.ERA_SPLIT),
                    ("2016-2026", data.ERA_SPLIT, "2100-01-01")]:
    sub = ls[(ls.index >= lo) & (ls.index < hi)]["ls"].to_numpy()
    print(f"  {lbl}: n={len(sub)}  mean {sub.mean()*1e4:+.1f} bps  NW t = {st.newey_west_t(sub):+.2f}")

print("\n# THIRD AXIS — does the divergence LEAD next-year gross-margin change?")
q = st.leads_margin(ev)
print(f"  n={q['n']}  slope={q['slope']:+.3f}  t={q['t']:+.2f}  R2={q['r2']:.3f}  corr={q['corr']:+.3f}")
print(f"  next-yr margin change: top divergence tercile {q['top_margin']*100:+.2f}pp  "
      f"vs bottom {q['bot_margin']*100:+.2f}pp  ->  spread {q['spread']*100:+.2f} pp")

print("\n# TRADABILITY — calendar long-short net of costs + short borrow")
for cb, bb in [(10.0, 50.0), (20.0, 100.0)]:
    net = st.calendar_ls_net(ls, cost_bps=cb, borrow_bps_ann=bb)
    print(f"  cost={cb:>4.1f} bps, borrow={bb:>5.1f} bps/yr: net {net['net_mean_bps']:+.1f} bps/mo "
          f"({net['net_ann_pct']:+.2f}%/yr), NW t = {net['net_t_nw']:+.2f}, Sharpe {net['net_sharpe']:.2f}")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
print("  the calendar detector must NOT fire on a null world (edge=0) and must recover a")
print("  planted effect. Null checked over 12 seeds (never a single stream).")
null_ts = []
for s_ in range(12):
    p0, e0 = data.synthetic_panel(edge=0.0, seed=858 + s_)
    null_ts.append(st.synthetic_detect(p0, e0)["t_nw"])
null_ts = np.asarray(null_ts)
print(f"  null (edge=0), 12 seeds: mean NW t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/12 seeds")
p1, e1 = data.synthetic_panel(edge=0.15, seed=858)
pl = st.synthetic_detect(p1, e1)
print(f"  planted edge=0.15 (seed 858): mean {pl['mean_bps']:+.1f} bps/mo, NW t = {pl['t_nw']:+.2f}")
