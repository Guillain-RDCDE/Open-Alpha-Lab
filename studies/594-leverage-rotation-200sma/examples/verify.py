"""Reproducible headline run for Study 594 — Leverage Rotation (TQQQ + 200SMA).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached QQQ/TQQQ/^IRX tape under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pandas as pd

from leverage_rotation_200sma import data, strategy as st

COST_HEADLINE = 5.0     # bps one-way x NAV, the headline retail assumption
N_SEEDS = 40            # random-timer seeds (desk minimum is 20)

print("# Leverage Rotation (TQQQ + 200SMA) — QQQ/TQQQ/^IRX, yfinance, as-of", data.AS_OF)

if data.have_real():
    px = data.load_real()
    try:
        from quantlab.repro import fingerprint
        fp = fingerprint(px.reset_index(drop=True))
    except Exception:
        fp = "(quantlab unavailable)"
    qqq = data.qqq_returns(px)
    rf = data.rf_daily(px)
    sim3 = data.sim3x_returns(qqq, rf)
    tqqq = data.tqqq_returns(px)
    sig = st.sma_signal(px["QQQ"].dropna())
    cash = rf.reindex(qqq.index).ffill()

    print(f"prices      : {px['QQQ'].dropna().shape[0]} QQQ days, "
          f"{px['QQQ'].dropna().index.min().date()} -> {px.index.max().date()} | "
          f"TQQQ from {px['TQQQ'].dropna().index.min().date()} | fingerprint {fp}")

    print("\n# Synthesised 3x vs REAL TQQQ (the audit of the pre-2010 extension)")
    v = data.validate_sim3x(px)
    print(f"  common span {v['start']} -> {v['end']} ({v['n_days']} days)")
    print(f"  daily-return corr = {v['corr']:.4f} | residual drag {v['resid_drag_ann_pct']:+.2f}%/yr "
          f"after the {data.SIM3X_FEE*100:.1f}%/yr all-in fee | RMS TE {v['rms_te_bps']:.1f} bps/day "
          f"| terminal-NAV ratio real/sim {v['nav_ratio']:.3f}")

    # ---------------- headline race ----------------
    rot, nsw = st.rotation_returns(sim3, cash, sig, cost_bps=COST_HEADLINE)
    idx = rot.index
    yrs = len(idx) / st.TRADING_DAYS
    print(f"\n# Headline race — signal live {idx.min().date()} -> {idx.max().date()} "
          f"({yrs:.1f} years, {len(idx)} days), net of {COST_HEADLINE:.0f} bps one-way per switch")
    legs = {
        "rotation (3x above 200SMA, T-bills below)": rot,
        "QQQ buy & hold": qqq.reindex(idx),
        "3x buy & hold (synthesised TQQQ)": sim3.reindex(idx),
    }
    for nm, r in legs.items():
        s = st.perf_stats(r, rf)
        print(f"  {nm:44s} CAGR {s['cagr_pct']:+7.2f}%  vol {s['vol_pct']:5.1f}%  "
              f"Sharpe {s['sharpe']:+.3f}  MaxDD {s['maxdd_pct']:+7.2f}%  x{s['final_mult']:.2f}")
    pos = st.positions(sig, sim3.index)
    print(f"  exposure {pos.mean()*100:.1f}% of days in the market | {nsw} switches "
          f"({nsw/yrs:.1f}/yr)")

    print("\n# The desk t-stats (HAC/Newey-West on the daily return DIFFERENCE)")
    h1 = st.hac_t(rot - qqq.reindex(idx))
    h2 = st.hac_t(rot - sim3.reindex(idx))
    print(f"  rotation - QQQ B&H : mean {h1['mean_bps']:+.2f} bps/day  HAC t = {h1['t']:+.2f}  "
          f"(lags {h1['lags']}, n {h1['n']})")
    print(f"  rotation - 3x  B&H : mean {h2['mean_bps']:+.2f} bps/day  HAC t = {h2['t']:+.2f}")

    print(f"\n# Random-timer baseline — exposure AND switch-count matched, {N_SEEDS} seeds")
    rt = st.random_timer_test(sim3, cash, pos, cost_bps=COST_HEADLINE, n_seeds=N_SEEDS)
    print(f"  seed-averaged Welch t (real vs random daily returns) = {rt['avg_welch_t']:+.2f} "
          f"(sd across seeds {rt['std_welch_t']:.2f})")
    print(f"  real Sharpe {rt['real_sharpe']:+.3f} vs random-timer mean {rt['rand_sharpe_mean']:+.3f} "
          f"| real beats {rt['beat_frac']*100:.0f}% of seeds on Sharpe "
          f"| random-timer mean CAGR {rt['rand_cagr_mean_pct']:+.2f}%")

    print("\n# WHY it works (and doesn't) — next-day QQQ conditioning on the 200SMA state")
    c = st.sma_conditioning(qqq, sig)
    print(f"  mean channel : above {c['mean_above_bps']:+.2f} bps/day (n={c['n_above']}) vs "
          f"below {c['mean_below_bps']:+.2f} bps/day (n={c['n_below']})  Welch t = {c['welch_t_mean']:+.2f}")
    print(f"  vol  channel : above {c['vol_above_pct']:.1f}% ann. vs below {c['vol_below_pct']:.1f}% ann. "
          f"({c['vol_below_pct']/c['vol_above_pct']:.2f}x)  Welch t (sq. returns) = {c['welch_t_var']:+.2f}")

    print("\n# Cost sensitivity (one-way bps x NAV per switch)")
    for cb in (0.0, 2.0, 5.0, 10.0):
        r, _ = st.rotation_returns(sim3, cash, sig, cost_bps=cb)
        s = st.perf_stats(r, rf)
        print(f"  cost {cb:4.1f} bps: CAGR {s['cagr_pct']:+6.2f}%  Sharpe {s['sharpe']:+.3f}  "
              f"MaxDD {s['maxdd_pct']:+.1f}%")

    print("\n# SMA-window robustness (net of 5 bps)")
    for w in (150, 200, 250):
        sg = st.sma_signal(px["QQQ"].dropna(), window=w)
        r, ns = st.rotation_returns(sim3, cash, sg, cost_bps=COST_HEADLINE)
        r = r.reindex(idx).dropna()
        s = st.perf_stats(r, rf)
        h = st.hac_t(r - qqq.reindex(r.index))
        print(f"  SMA {w}: CAGR {s['cagr_pct']:+6.2f}%  MaxDD {s['maxdd_pct']:+.1f}%  "
              f"switches {ns}  HAC t vs QQQ = {h['t']:+.2f}")

    print("\n# Sub-periods (total return / max drawdown inside the window, net of 5 bps)")
    subs = st.SUBPERIODS + [("2020 full year", "2020-01-01", "2020-12-31")]
    for nm, a, b in subs:
        w1 = st.window_stats(rot, a, b)
        w2 = st.window_stats(qqq, a, b)
        w3 = st.window_stats(sim3, a, b)
        psub = pos.loc[a:b]
        nsw_sub = int((psub.values[1:] != psub.values[:-1]).sum())
        print(f"  {nm:22s} rotation {w1['total_pct']:+7.1f}% (dd {w1['maxdd_pct']:+6.1f}%) | "
              f"QQQ {w2['total_pct']:+6.1f}% (dd {w2['maxdd_pct']:+6.1f}%) | "
              f"3x {w3['total_pct']:+6.1f}% (dd {w3['maxdd_pct']:+6.1f}%) | switches {nsw_sub}")

    print("\n# Third axis — $10,000 at the 2000-03-24 QQQ peak: does the dot-com cohort survive?")
    co = st.cohort_2000(rot, qqq.reindex(idx), sim3.reindex(idx))
    for nm, label in [("rotation", "rotation"), ("qqq_bh", "QQQ B&H"), ("tqqq_bh", "3x B&H")]:
        d = co[nm]
        rec = d["recovered_on"] or "NEVER"
        uw = ""
        if d["recovered_on"]:
            uw_yrs = (pd.Timestamp(d["recovered_on"]) - pd.Timestamp("2000-03-24")).days / 365.25
            uw = f" ({uw_yrs:.1f} yrs underwater)"
        print(f"  {label:10s}: trough ${d['trough_value']:>8,.0f}  maxDD {d['maxdd_pct']:+.2f}%  "
              f"made whole {rec}{uw}  final ${d['final_value']:>10,.0f} on {d['end']}")

    print("\n# Cross-check on the REAL TQQQ tape (2010-02-12 onward, net of 5 bps)")
    rot_tq, nsw_tq = st.rotation_returns(tqqq, cash, sig, cost_bps=COST_HEADLINE)
    rot_sim_2010 = rot.reindex(rot_tq.index)
    print(f"  corr(rotation on real TQQQ, rotation on sim 3x) = "
          f"{rot_tq.corr(rot_sim_2010):.4f}")
    for nm, r in [("rotation (real TQQQ)", rot_tq),
                  ("QQQ B&H", qqq.reindex(rot_tq.index)),
                  ("TQQQ B&H (real)", tqqq)]:
        s = st.perf_stats(r, rf)
        print(f"  {nm:22s} CAGR {s['cagr_pct']:+7.2f}%  Sharpe {s['sharpe']:+.3f}  "
              f"MaxDD {s['maxdd_pct']:+6.1f}%")
    h3 = st.hac_t(rot_tq - qqq.reindex(rot_tq.index))
    h4 = st.hac_t(rot_tq - tqqq.reindex(rot_tq.index))
    print(f"  rotation - QQQ  B&H (2010+): HAC t = {h3['t']:+.2f}   <- bull-heavy sub-sample, "
          f"starts AFTER the regime that killed it")
    print(f"  rotation - TQQQ B&H (2010+): HAC t = {h4['t']:+.2f}   <- the rotation LAGGED "
          f"holding TQQQ outright since 2010")
else:
    print("(no _cache/lr200_prices.csv — run data.fetch_prices() once to build the cache)")

print("\n# Synthetic control — deterministic, no network (machinery proof, never market evidence)")
print("  null (effect=0, i.i.d. world): the rotation must NOT beat a matched random timer;")
print("  planted (persistent bear regimes): it must light up.")
for eff in (0.0, 0.008):
    close = data.synthetic_world(effect=eff, n_days=7560)
    r = close.pct_change().dropna()
    zero = pd.Series(0.0, index=r.index)
    sg = st.sma_signal(close)
    p = st.positions(sg, r.index)
    res = st.random_timer_test(r, zero, p, cost_bps=COST_HEADLINE, n_seeds=N_SEEDS)
    cnd = st.sma_conditioning(r, sg)
    tag = "null   " if eff == 0.0 else "planted"
    print(f"  {tag} effect={eff:.3f}: avg Welch t = {res['avg_welch_t']:+.2f} "
          f"({res['n_seeds']} seeds) | real Sharpe {res['real_sharpe']:+.3f} vs random "
          f"{res['rand_sharpe_mean']:+.3f} | mean-channel Welch t = {cnd['welch_t_mean']:+.2f}")
