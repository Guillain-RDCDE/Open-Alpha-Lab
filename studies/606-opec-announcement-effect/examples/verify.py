"""Reproducible headline run for Study 606 — OPEC Announcement Effect.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached CL=F / BZ=F / USO OHLC
tapes under ``_cache/`` if present (the real-tape numbers), and always runs the
synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from opec_announcement_effect import data, strategy as st

AS_OF = "2026-06-30"          # last complete session of the last complete month
N_BOOT, N_PLACEBO = 5000, 2000

print("# OPEC Announcement Effect — 107 hardcoded OPEC/OPEC+ ministerial decision days, 2000-2026")
print(f"meetings        : {len(data.meeting_dates())} "
      f"({data.meeting_dates().min().date()} -> {data.meeting_dates().max().date()})")

if data.have_real():
    try:
        from quantlab.repro import fingerprint
    except Exception:
        fingerprint = None
    dates = data.meeting_dates()
    tapes = data.load_real(asof=AS_OF)
    for tk, px in tapes.items():
        fp = fingerprint(px) if fingerprint else "n/a"
        print(f"{tk:5s}           : {len(px)} sessions {px.index.min().date()} -> "
              f"{px.index.max().date()}  fingerprint {fp}")

    print("\n# 1) Volatility on decision days (day 0 vs event-free baseline, halo ±5)")
    for tk, px in tapes.items():
        v = st.vol_stats(px, dates, n_boot=N_BOOT, n_placebo=N_PLACEBO)
        print(f"  {tk:5s} n_ev={v['n_events']:3d}: |r| {v['abs_ev_bps']:.1f} vs "
              f"{v['abs_base_bps']:.1f} bps -> x{v['vol_multiple']:.2f} "
              f"[95% CI {v['ci_lo']:.2f}-{v['ci_hi']:.2f}]  Welch t={v['welch_t_abs']:+.2f}  "
              f"BF t={v['bf_t']:+.2f}  var-ratio {v['var_ratio']:.2f}  "
              f"placebo p={v['p_placebo']:.3f}")
        print(f"        intraday range {v['rng_ev_bps']:.1f} vs {v['rng_base_bps']:.1f} bps "
              f"-> x{v['rng_multiple']:.2f}  Welch t={v['welch_t_rng']:+.2f}")

    print("\n# 2) Signed post-decision drift close(-1)->close(+k)")
    for tk, px in tapes.items():
        print(f"  {tk}:")
        for d in st.drift_stats(px, dates):
            print(f"    k={d['k']}: n={d['n']:3d} cum {d['mean_bps']:+7.1f} bps  "
                  f"t_event={d['t_event']:+5.2f}  HAC(NW) t={d['hac_t']:+5.2f}")

    print("\n# 3) Surprise continuation — go with the day-0 sign, hold to close(+5)")
    no2020 = dates[(dates < "2020-01-01") | (dates > "2020-12-31")]
    pre16, post16 = dates[dates < "2016-01-01"], dates[dates >= "2016-01-01"]
    for tk, px in tapes.items():
        c = st.continuation_stats(px, dates, k=5, cost_bps_side=2.0)
        print(f"  {tk:5s} all     : n={c['n']:3d} gross {c['gross_bps']:+7.1f} bps "
              f"t={c['t']:+5.2f} hit={c['hit']*100:.0f}%  "
              f"lag-1-day t={c['t_lag']:+5.2f}")
        for lab, dd in [("ex-2020", no2020), ("pre-2016", pre16), ("2016+", post16)]:
            c2 = st.continuation_stats(px, dd, k=5, cost_bps_side=2.0)
            print(f"        {lab:8s}: n={c2['n']:3d} gross {c2['gross_bps']:+7.1f} bps "
                  f"t={c2['t']:+5.2f} hit={c2['hit']*100:.0f}%")

    print("\n# Costs on the continuation rule (CL=F, ~4 trades/yr)")
    cl = st.continuation_stats(tapes["CL=F"], dates, k=5)
    for side in (2.0, 5.0, 10.0):
        print(f"  {side:>4.1f} bps/side: gross {cl['gross_bps']:+.1f} bps/event -> "
              f"net {cl['gross_bps'] - 2*side:+.1f} bps/event  (t stays {cl['t']:+.2f} — "
              f"never significant)")
else:
    print("(no cached tapes — run data.fetch_prices() once to build _cache/)")

print("\n# Synthetic machinery control — deterministic, no network")
print("  null (vol x1, drift 0) must NOT fire; the folklore planted at full strength")
print("  (vol x2) and a planted +150 bps day-0 drift must light up.")
for lab, vm, dr in [("null           ", 1.0, 0.0),
                    ("vol x2 planted ", 2.0, 0.0),
                    ("vol x2 + drift ", 2.0, 0.015)]:
    px, dd = data.synthetic_world(vol_mult=vm, drift=dr, seed=606)
    v = st.vol_stats(px, dd, n_boot=1000, n_placebo=500)
    d0 = st.drift_stats(px, dd, horizons=(0,))[0]
    print(f"  {lab}: vol x{v['vol_multiple']:.2f}  Welch t={v['welch_t_abs']:+5.2f}  "
          f"placebo p={v['p_placebo']:.3f} | day-0 drift {d0['mean_bps']:+7.1f} bps  "
          f"HAC t={d0['hac_t']:+5.2f}")
