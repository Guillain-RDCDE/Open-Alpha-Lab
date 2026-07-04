"""Reproducible headline run for Study 607 — Quarterly Refunding Shock.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached ^TNX + TLT tape under
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

from quarterly_refunding_shock import data, strategy as st

print("# Quarterly Refunding Shock — 106 hardcoded QRA dates (TreasuryDirect-derived) "
      "vs ^TNX day-0 moves + TLT holds")

if data.have_real():
    tnx, tyx, tlt = data.load_real()
    try:
        from quantlab import repro
        print(repro.data_stamp("^TNX 10Y yield (%) daily close", tnx.to_frame(), asof=data.AS_OF))
        print(repro.data_stamp("^TYX 30Y yield (%) daily close", tyx.to_frame(), asof=data.AS_OF))
        print(repro.data_stamp("TLT total-return close", tlt.to_frame(), asof=data.AS_OF))
    except ImportError:
        pass

    dy = st.dy_bps(tnx)
    lvl_idx = pd.DatetimeIndex(tnx.index)
    sp = st.clean_split(lvl_idx, data.QRA_DATES, data.FOMC_DATES)
    # translate level-index positions to dy-index positions
    clean = st.align_positions(lvl_idx, dy.index, sp["clean_pos"])
    overlap = st.align_positions(lvl_idx, dy.index, sp["overlap_pos"])
    allq = st.align_positions(lvl_idx, dy.index, sp["qra_pos"])
    base_mask = sp["base_mask"][1:]  # dy index = level index minus the first bar

    n_dates = len(data.QRA_DATES)
    print(f"\nQRA calendar    : {n_dates} announcement dates "
          f"{data.QRA_DATES[0]} -> {data.QRA_DATES[-1]} (all Wednesdays, 08:30 ET)")
    print(f"on the tape     : {len(allq)} QRA day-0 sessions | FOMC-clean {len(clean)} | "
          f"FOMC-overlap {len(overlap)} ({len(overlap)/len(allq)*100:.0f}% of QRA days are "
          f"ALSO FOMC statement days)")
    print(f"baseline        : {int(base_mask.sum())} ordinary sessions "
          f"(no QRA window day, no FOMC day)")

    print("\n# Day-0 — does the 10Y move more on QRA day? (PRIMARY: FOMC-clean QRA days, "
          "Welch t on |dy|; events ~63 sessions apart -> serially uncorrelated, no HAC needed)")
    s_clean = st.day0_stats(dy, clean, base_mask)
    s_all = st.day0_stats(dy, allq, base_mask)
    s_ovl = st.day0_stats(dy, overlap, base_mask)
    for lab, s in [("CLEAN (primary)", s_clean), ("ALL QRA days   ", s_all),
                   ("FOMC-overlap   ", s_ovl)]:
        print(f"  {lab}: n={s['n_event']:>3} | |dy| {s['abs_event']:.2f} bps vs baseline "
              f"{s['abs_base']:.2f} bps (x{s['abs_ratio']:.2f}) Welch t = {s['welch_abs']:+.2f} | "
              f"signed {s['signed_mean']:+.2f} bps t = {s['t_signed']:+.2f}")
    pl = st.placebo_pvalue(dy, clean, base_mask, n_draws=2000, seed=607)
    print(f"  placebo (2,000 random same-size calendars from the baseline): "
          f"observed {pl['obs']:.2f} bps vs draw mean {pl['draw_mean']:.2f} bps, "
          f"p = {pl['p_value']:.3f}")

    # In-tape positive control: FOMC-only days through the SAME pipeline must fire
    qset = set(sp["qra_pos"])
    fomc_only = [p for p in st.event_positions(lvl_idx, data.FOMC_DATES) if p not in qset]
    fomc_only = st.align_positions(lvl_idx, dy.index, fomc_only)
    s_f = st.day0_stats(dy, fomc_only, base_mask)
    print(f"  FOMC-only days (in-tape positive control): n={s_f['n_event']} | "
          f"|dy| {s_f['abs_event']:.2f} bps (x{s_f['abs_ratio']:.2f}) "
          f"Welch t = {s_f['welch_abs']:+.2f} — the same method DOES detect a real "
          f"macro event on this tape")

    # 30Y robustness — "the long end" proper (^TYX), same clean day-0 test
    dy30 = st.dy_bps(tyx)
    lvl30 = pd.DatetimeIndex(tyx.index)
    sp30 = st.clean_split(lvl30, data.QRA_DATES, data.FOMC_DATES)
    clean30 = st.align_positions(lvl30, dy30.index, sp30["clean_pos"])
    bm30 = sp30["base_mask"][1:]
    s30 = st.day0_stats(dy30, clean30, bm30)
    print(f"  30Y (^TYX) robustness, clean day-0: |dy| {s30['abs_event']:.2f} bps vs "
          f"baseline {s30['abs_base']:.2f} bps (x{s30['abs_ratio']:.2f}) "
          f"Welch t = {s30['welch_abs']:+.2f} | signed {s30['signed_mean']:+.2f} bps "
          f"t = {s30['t_signed']:+.2f} (n={s30['n_event']})")

    print("\n# Event window [-1..+3] — |dy| and signed dy by offset (clean QRA days)")
    for w in st.window_profile(dy, clean, base_mask):
        print(f"  day {w['offset']:+d}: |dy| {w['abs_mean']:.2f} bps (baseline "
              f"{w['abs_base']:.2f}, Welch t {w['welch_abs']:+.2f}) | signed "
              f"{w['signed_mean']:+.2f} bps (t {w['t_signed']:+.2f})  n={w['n']}")

    print("\n# The day+2 blip is the JOBS REPORT, not a QRA aftershock "
          "(first-Friday collision check)")
    dj = st.day2_jobs_diagnostic(dy, clean, base_mask)
    print(f"  {dj['n_first_friday']}/{dj['n_day2']} clean-QRA day+2 sessions "
          f"({dj['share_first_friday']*100:.0f}%) are FIRST FRIDAYS — the Employment "
          f"Situation 08:30 ET slot")
    print(f"  day+2 |dy|: all {dj['abs_all']:.2f} bps (Welch t {dj['welch_all']:+.2f}) | "
          f"first-Fridays alone {dj['abs_ff']:.2f} bps | EXCLUDING first-Fridays "
          f"{dj['abs_ex_ff']:.2f} bps (Welch t {dj['welch_ex_ff']:+.2f}, n={dj['n_ex_ff']})")

    print("\n# Third axis — was 2023 a regime change? Vol-normalised |move| "
          "(|dy| / trailing 60-day mean |dy|), eras split at 2023-01-01")
    for lab, pos in [("ALL QRA days", allq), ("FOMC-clean  ", clean)]:
        ec = st.era_compare(dy, pos, base_mask,
                            drop_dates=["2023-08-02", "2023-11-01"])
        print(f"  {lab}: 2000-22 norm {ec['norm_early']:.2f}x (n={ec['n_early']}) | 2023+ "
              f"norm {ec['norm_late']:.2f}x (n={ec['n_late']}) | Welch late-vs-early "
              f"t = {ec['welch_late_vs_early']:+.2f}")
        print(f"    drop Aug-2 + Nov-1 2023: 2023+ norm {ec['norm_late_trim']:.2f}x "
              f"(n={ec['n_late_trim']}), Welch vs early t = {ec['welch_trim_vs_early']:+.2f}")
        print(f"    each era vs its own baseline: early t = "
              f"{ec['welch_early_vs_own_base']:+.2f}, late t = {ec['welch_late_vs_own_base']:+.2f} "
              f"| raw excess ratio early x{ec['ratio_early']:.2f}, late x{ec['ratio_late']:.2f}")
        if lab.startswith("ALL"):
            EC_ALL = ec
    print("  2023+ QRA days, vol-normalised |move| one by one:")
    for d, x in zip(EC_ALL["late_dates"], EC_ALL["late_norm_values"]):
        tag = " <- FOMC same day" if d in set(data.FOMC_DATES) else ""
        print(f"    {d}: {x:.2f}x{tag}")

    print("\n# The 2023 narrative — the four episodes, signed 10Y moves off the tape")
    for d, kind, txt in data.EPISODES_2023:
        ts = pd.Timestamp(d)
        if ts in dy.index:
            print(f"  {d} ({kind}): dy10 = {dy.loc[ts]:+.1f} bps | {txt}")
    jul = float((tnx.loc["2023-08-04"] - tnx.loc["2023-07-28"]) * 100)
    octv = float((tnx.loc["2023-11-03"] - tnx.loc["2023-10-27"]) * 100)
    print(f"  Aug-2023 refunding week (Jul-28 -> Aug-4 close): {jul:+.1f} bps on the 10Y")
    print(f"  Nov-2023 refunding week (Oct-27 -> Nov-3 close): {octv:+.1f} bps on the 10Y")

    print("\n# Tradability — TLT after the QRA (entry at the day-0 close = the ONE lag; "
          "exit +3 sessions; ~4 events/yr)")
    tr = st.tlt_event_trades(tlt, data.QRA_DATES, hold=3)
    print(f"  events: {tr['n_events']} | unconditional 3-day hold "
          f"{tr['uncond_mean_bps']:+.1f} bps/event (t = {tr['uncond_t']:+.2f}; "
          f"vs {tr['n_base']} non-overlapping baseline windows {tr['base_mean_bps']:+.1f} bps, "
          f"Welch t = {tr['welch_vs_base']:+.2f})")
    print(f"  conditional (ride the day-0 sign): {tr['cond_mean_bps']:+.1f} bps/event "
          f"(t = {tr['cond_t']:+.2f})")
    for cb in (2.0, 5.0):
        nt = st.tlt_net(tr, cb)
        print(f"  net @ {cb:.0f} bps one-way: uncond {nt['uncond_net_bps']:+.1f} bps/event "
              f"({nt['uncond_net_ann_bps']:+.0f} bps/yr) | cond {nt['cond_net_bps']:+.1f} "
              f"bps/event ({nt['cond_net_ann_bps']:+.0f} bps/yr)")
else:
    print("(no _cache/qrs_tnx.csv — run data.fetch_tape() once to build the cache)")

print("\n# Synthetic control — deterministic, no network (machinery proof, never market "
      "evidence)")
print("  the |move| detector must NOT fire on the null and must light up on a planted "
      "event-day vol multiplier;")
print("  the signed detector must stay quiet at mean 0 and fire on a planted signed mean.")
for vol_mult, mean_bp in [(1.0, 0.0), (2.0, 0.0), (1.0, 4.0)]:
    lvl, ev_dates = data.synthetic_world(vol_mult=vol_mult, mean_bp=mean_bp, seed=607)
    dy_s = st.dy_bps(lvl)
    li = pd.DatetimeIndex(lvl.index)
    sp_s = st.clean_split(li, ev_dates, [])
    pos_s = st.align_positions(li, dy_s.index, sp_s["clean_pos"])
    bm_s = sp_s["base_mask"][1:]
    s = st.day0_stats(dy_s, pos_s, bm_s)
    print(f"  planted vol x{vol_mult:.1f}, mean {mean_bp:+.1f} bp: |dy| ratio "
          f"x{s['abs_ratio']:.2f} Welch t = {s['welch_abs']:+.2f} | signed "
          f"{s['signed_mean']:+.2f} bps t = {s['t_signed']:+.2f} (n={s['n_event']})")
