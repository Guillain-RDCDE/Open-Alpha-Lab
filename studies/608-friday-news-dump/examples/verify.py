"""Reproducible headline run for Study 608 — Friday News Dump.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached EDGAR filing panel +
acceptance timestamps + yfinance closes under ``_cache/`` if present (the real-tape
numbers), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from friday_news_dump import data, strategy as st

AS_OF = "2026-06-30"        # last complete month
N_PERM = 2000

print("# Friday News Dump — negative 8-Ks (Items 4.02 / 2.06 / 5.02-CEO), 2004-08-23 ->", AS_OF)

if data.have_real():
    try:
        from quantlab.repro import fingerprint
    except Exception:
        fingerprint = None

    ev = data.load_events()
    bad = ev[ev["cls"].isin(data.BAD_CLASSES)]
    ctrl = data.load_control()
    panel = data.load_panel(require_prices=True)
    px = data.load_prices(asof=AS_OF)
    fp_ev = fingerprint(bad[["cik", "weekday"]].astype(float)) if fingerprint else "n/a"
    fp_px = fingerprint(px[["SPY"]]) if fingerprint else "n/a"
    print(f"filings harvested : {len(ev)} total — {len(bad)} negative "
          f"({', '.join(f'{c}={int(n)}' for c, n in bad['cls'].value_counts().items())}), "
          f"{len(ctrl)} earnings-control")
    n_map = int(data.load_panel(require_prices=False)["ticker"].notna().sum())
    print(f"ticker-mapped     : {n_map} of {len(bad)} "
          f"negative filings map to a CURRENT SEC ticker (survivorship: dead firms drop here)")
    print(f"with full tape    : {len(panel)} events (price window day -1..+10 complete)")
    print(f"fingerprints      : events {fp_ev} | SPY tape {fp_px} "
          f"({len(px)} sessions {px.index.min().date()} -> {px.index.max().date()})")

    kept, armat = st.event_ars(panel, px)
    print(f"panel for CARs    : {len(kept)} events, "
          f"{kept['accept_ts'].min().date()} -> {kept['accept_ts'].max().date()}")

    # ------------------------------------------------------------------ #
    print("\n# 1) The DellaVigna-Pollet drift test — Friday vs Mon-Thu filers")
    fri = kept["friday"].to_numpy()
    g = st.group_stats(fri, armat)
    print(f"  day-0 reaction  : Friday {g['ar0_fri_bps']:+7.1f} bps (t={g['ar0_t_fri']:+.2f})  "
          f"Mon-Thu {g['ar0_oth_bps']:+7.1f} bps (t={g['ar0_t_oth']:+.2f})  "
          f"gap Welch t={g['ar0_gap_t']:+.2f}")
    print(f"  drift CAR[+1..+10]: Friday {g['drift_fri_bps']:+7.1f} bps (t={g['drift_t_fri']:+.2f}, "
          f"n={g['n_fri']})  Mon-Thu {g['drift_oth_bps']:+7.1f} bps (t={g['drift_t_oth']:+.2f}, "
          f"n={g['n_oth']})")
    print(f"  THE GAP         : {g['gap_bps']:+7.1f} bps  Welch t={g['gap_t']:+.2f}  "
          f"(1%-winsorized t={g['gap_t_wins']:+.2f})")
    p_perm = st.placebo_gap(fri, armat, n_perm=N_PERM)
    print(f"  label-permutation placebo (one-sided, {N_PERM} draws): p={p_perm:.3f}")
    print(f"  pooled drift (all bad news): {g['pooled_drift_bps']:+7.1f} bps  "
          f"t={g['pooled_drift_t']:+.2f}  (winsorized t={g['pooled_drift_t_wins']:+.2f})")
    print(f"  CAR[0..+10]     : Friday {g['car010_fri_bps']:+7.1f} vs Mon-Thu "
          f"{g['car010_oth_bps']:+7.1f} bps  gap Welch t={g['car010_gap_t']:+.2f}")

    gpm = st.group_stats(kept["friday_pm"].to_numpy(), armat)
    print(f"  Friday-AFTER-CLOSE only (n={gpm['n_fri']}): drift {gpm['drift_fri_bps']:+7.1f} bps "
          f"(t={gpm['drift_t_fri']:+.2f})  gap vs rest {gpm['gap_bps']:+7.1f} bps  "
          f"Welch t={gpm['gap_t']:+.2f}")

    print("\n  robustness — by class and by half-sample (gap = Friday minus Mon-Thu drift):")
    for cls in data.BAD_CLASSES:
        m = (kept["cls"] == cls).to_numpy()
        gc = st.group_stats(fri[m], armat[m])
        print(f"    {cls:12s}: n={m.sum():4d} (fri {gc['n_fri']:3d})  "
              f"gap {gc['gap_bps']:+7.1f} bps  Welch t={gc['gap_t']:+.2f}  "
              f"pooled drift {gc['pooled_drift_bps']:+7.1f} bps (t={gc['pooled_drift_t']:+.2f})")
    for lab, m in [("2004-2015", (kept["accept_ts"] < "2016-01-01").to_numpy()),
                   ("2016-2026", (kept["accept_ts"] >= "2016-01-01").to_numpy())]:
        gc = st.group_stats(fri[m], armat[m])
        print(f"    {lab:12s}: n={m.sum():4d} (fri {gc['n_fri']:3d})  "
              f"gap {gc['gap_bps']:+7.1f} bps  Welch t={gc['gap_t']:+.2f}  "
              f"pooled drift {gc['pooled_drift_bps']:+7.1f} bps (t={gc['pooled_drift_t']:+.2f})")

    # ------------------------------------------------------------------ #
    print("\n# 2) Third axis — is bad news actually dumped on Friday (evening)?")
    s = st.share_stats(bad, ctrl)
    print(f"  n: {s['n_bad']} negative vs {s['n_ctrl']} earnings-control filings")
    for flag, lab in [("friday", "Friday share      "),
                      ("after_close", "after-16:00 share "),
                      ("friday_pm", "Friday-PM share   ")]:
        d = s[flag]
        print(f"  {lab}: bad {d['bad']*100:5.1f}%  vs control {d['ctrl']*100:5.1f}%  "
              f"two-prop z={d['z']:+.2f}")
    print(f"  uniform-calendar null: Fridays are {s['friday_null']*100:.1f}% of EDGAR-open "
          f"days -> bad-news Friday share vs null: binomial z={s['friday_vs_null_z']:+.2f}")
    wd = s["weekday_shares_bad"]
    wdc = s["weekday_shares_ctrl"]
    print("  weekday mix bad  : " + "  ".join(f"{l} {x*100:.1f}%" for l, x in
                                              zip("Mon Tue Wed Thu Fri".split(), wd)))
    print("  weekday mix ctrl : " + "  ".join(f"{l} {x*100:.1f}%" for l, x in
                                              zip("Mon Tue Wed Thu Fri".split(), wdc)))

    # ------------------------------------------------------------------ #
    print("\n# 3) Tradability — short the Friday dump at day-0 close, cover +10, SPY-hedged")
    for flag, lab in [("friday", "all Friday filers"), ("friday_pm", "Friday after-close")]:
        for borrow in (0.05, 0.02):
            t = st.trade_stats(kept, armat, flag=flag, borrow_apr=borrow)
            print(f"  {lab:18s} borrow {borrow*100:.0f}%/yr: n={t['n']:3d} "
                  f"(~{t['per_year']:.0f}/yr)  gross {t['gross_bps']:+7.1f} bps/event "
                  f"(t={t['t_gross']:+.2f}, hit {t['hit']*100:.0f}%)  costs {t['cost_bps']:.0f} bps "
                  f"-> net {t['net_bps']:+7.1f} bps (t={t['t_net']:+.2f})")
else:
    print("(no cached panel — run data.fetch_events/fetch_acceptance/fetch_ticker_map/"
          "fetch_prices once to build _cache/)")

# ---------------------------------------------------------------------- #
print("\n# Synthetic machinery control — deterministic, no network")
print("  null must NOT fire; a planted -300 bps Friday drift and a planted Friday-hiding")
print("  propensity (p_fri 0.20 -> 0.35) must light up. Machinery proof only.")
for lab, kw in [("null              ", dict()),
                ("drift -300 bps Fri", dict(drift_fri=-0.03)),
                ("hide p_fri=0.35   ", dict(p_fri=0.35))]:
    panel, armat = data.synthetic_world(**kw)
    g = st.group_stats(panel["friday"].to_numpy(), armat)
    p = st.placebo_gap(panel["friday"].to_numpy(), armat, n_perm=500)
    print(f"  {lab}: fri share {panel['friday'].mean()*100:4.1f}%  "
          f"gap {g['gap_bps']:+7.1f} bps  Welch t={g['gap_t']:+.2f}  placebo p={p:.3f}")
