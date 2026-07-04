"""Reproducible headline run for Study 639 — Gasoline RVP Seasonality.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached RB=F/CL=F/UGA/^IRX daily closes
under ``_cache/`` if present (the real-tape numbers, sliced to the frozen as-of), and always
runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from gasoline_rvp_seasonality import data, strategy as st

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def show_window(label, ws):
    print(f"  {label}: window {ws['win_mean_pct']:+.2f}%/mo vs rest {ws['rest_mean_pct']:+.2f}%/mo"
          f"  gap {ws['gap_pct']:+.2f}%/mo  Welch t (years) = {ws['welch_t_years']:+.2f}"
          f"  | window cum {ws['win_sum_pct']:+.2f}%  t = {ws['t_win_sum']:+.2f}"
          f"  hit {ws['hit_rate']*100:.1f}%  (n_years={ws['n_years']})")


print("# Gasoline RVP Seasonality — RB=F vs CL=F crack seasonal + the UGA curve test (yfinance)")
if data.have_real():
    from quantlab import repro

    px = data.load_real()
    print(repro.data_stamp("RB=F + CL=F + UGA + ^IRX daily closes", px, asof=data.AS_OF))
    mret = data.monthly_returns(px)
    ex = st.excess_series(mret)                      # log(1+RB) - log(1+CL), spliced chains
    print(f"monthly excess (spliced RB - CL, spot-proxy): {len(ex)} months  "
          f"{ex.index.min().date()} -> {ex.index.max().date()}")

    print("\n# Signal — the Feb-Apr run-up on the gasoline-minus-crude spread (spot-proxy)")
    ws = st.window_stats(ex, months=data.RUNUP_MONTHS)
    show_window("Feb-Apr", ws)
    print(f"  pooled-months Welch t = {ws['welch_t_pooled']:+.2f} "
          f"(n = {ws['n_pooled_w']} window months vs {ws['n_pooled_r']} rest)")

    print("\n# Per-calendar-month mean monthly excess (%, one-sample t, n) — Bonferroni-12 bar ~ |t| >= 3.0")
    mt = st.month_table(ex)
    for mo, row in mt.iterrows():
        star = "  <-- survives Bonferroni x12" if abs(row["t"]) >= 3.0 else ""
        print(f"  {MONTH_NAMES[mo-1]}: {row['mean_pct']:+6.2f}%/mo  t={row['t']:+5.2f}  n={int(row['n'])}{star}")

    print("\n# The September switch-back (winter blend legal again after Sep 15)")
    sep = st.single_month_stats(ex, month=data.SWITCHBACK_MONTH)
    print(f"  Sep excess: {sep['mean_pct']:+.2f}%/mo  t = {sep['t']:+.2f}  "
          f"negative in {sep['hit_rate']*100:.1f}% of years  (n_years={sep['n_years']})")

    print("\n# Robustness — window variants + halves split (spot-proxy excess)")
    for label, mos in (("Mar only ", (3,)), ("Mar-Apr  ", (3, 4)),
                       ("Feb-Apr  ", (2, 3, 4)), ("Feb-May  ", (2, 3, 4, 5))):
        w2 = st.window_stats(ex, months=mos)
        print(f"  {label}: cum {w2['win_sum_pct']:+.2f}%  t={w2['t_win_sum']:+.2f}  "
              f"Welch t (years)={w2['welch_t_years']:+.2f}  hit {w2['hit_rate']*100:.0f}%")
    pan = ws["panel"]
    for label, sub in (("2005-2015", pan.loc[:2015]), ("2016-2025", pan.loc[2016:])):
        print(f"  half {label}: mean window cum {sub['win_sum'].mean()*100:+.2f}%  "
              f"t={st.ttest_vs_zero(sub['win_sum'].values):+.2f}  (n={len(sub)})")

    print("\n# Third axis — does the futures curve already price it? (holder UGA vs spliced RB=F)")
    rg = st.roll_gap_stats(mret, months=data.RUNUP_MONTHS)
    print(f"  Feb-Apr: spliced chain {rg['chain_pct']:+.2f}% vs holder (UGA) {rg['holder_pct']:+.2f}%"
          f"  -> roll gap {rg['gap_pct']:+.2f}%/window  paired t = {rg['t']:+.2f}  "
          f"negative in {rg['share_negative']*100:.0f}% of years (n_years={rg['n_years']})")
    rg9 = st.roll_gap_stats(mret, months=(data.SWITCHBACK_MONTH,))
    print(f"  Sep    : spliced chain {rg9['chain_pct']:+.2f}% vs holder (UGA) {rg9['holder_pct']:+.2f}%"
          f"  -> roll gap {rg9['gap_pct']:+.2f}%/month   paired t = {rg9['t']:+.2f}  "
          f"negative in {rg9['share_negative']*100:.0f}% of years (n_years={rg9['n_years']})")

    print("\n# Tradability — what a real holder can actually capture")
    exi = st.excess_series(mret, long="UGA", short="CL=F")
    wsi = st.window_stats(exi, months=data.RUNUP_MONTHS)
    print("  investable crack (long UGA / short CL, monthly rebalanced):")
    show_window("Feb-Apr", wsi)
    sepi = st.single_month_stats(exi, month=data.SWITCHBACK_MONTH)
    print(f"    Sep: {sepi['mean_pct']:+.2f}%/mo  t = {sepi['t']:+.2f}  (n_years={sepi['n_years']})")
    print("  long-only UGA overlay (hold Feb-Apr only, one round trip/yr, one-way bps x NAV):")
    tb = data.tbill_monthly(px)
    for cb in (5.0, 10.0, 20.0):
        ov = st.overlay_stats(mret, cost_bps=cb)
        print(f"    cost={cb:>4.1f} bps: gross {ov['gross_pct']:+.2f}%/window -> net "
              f"{ov['net_pct']:+.2f}%  t = {ov['t_net']:+.2f}  hit {ov['hit_rate']*100:.1f}%"
              f"  (n_years={ov['n_years']})")
    oc = st.overlay_excess_cash(mret, tb, cost_bps=10.0)
    print(f"    net of 10 bps AND the window T-bill (excess-vs-cash): "
          f"{oc['excess_pct']:+.2f}%/window  t = {oc['t']:+.2f}  hit {oc['hit_rate']*100:.1f}%")
else:
    print("(no _cache/rvp_prices.csv — run data.fetch_prices() once to build the cache)")

print("\n# Synthetic control — deterministic, no network (machinery proof only, never evidence)")
print("  the window detector must recover a PLANTED seasonal and stay quiet on the null;")
print("  the roll-gap detector must recover a PLANTED curve-pricing drag and stay quiet on the null.")
print("  (each cell = mean t over 20 seeds — no single-seed baselines)")
N_SEEDS = 20
for amp, drag in ((0.0, 0.0), (0.03, 0.02)):
    wt, st9, gt, wc, gp = [], [], [], [], []
    for s in range(N_SEEDS):
        m = data.synthetic_world(amp=amp, roll_drag=drag, seed=639 + s)
        exs = st.excess_series(m)
        w = st.window_stats(exs, months=data.RUNUP_MONTHS)
        g = st.roll_gap_stats(m, months=data.RUNUP_MONTHS)
        s9 = st.single_month_stats(exs, month=data.SWITCHBACK_MONTH)
        wt.append(w["welch_t_years"]); wc.append(w["win_sum_pct"])
        st9.append(s9["t"]); gt.append(g["t"]); gp.append(g["gap_pct"])
    mean = lambda v: sum(v) / len(v)
    print(f"  amp={amp*100:.0f}%/mo drag={drag*100:.0f}%/mo: Welch t (years)={mean(wt):+.2f}  "
          f"window cum={mean(wc):+.2f}%  Sep t={mean(st9):+.2f}  "
          f"roll-gap={mean(gp):+.2f}% t={mean(gt):+.2f}")
