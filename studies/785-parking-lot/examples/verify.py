"""Reproducible headline run for Study 785 — Parking-Lot.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached WMT / SPY tapes under ``_cache/``
(fetching once on a cache miss), and always runs the synthetic control with no network. The
parking-count signal is a **LABELLED PROXY** (see ``data.py``), used ordinally only.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from parking_lot import data as dt, strategy as st  # noqa: E402

print("# Parking-Lot — do satellite parking counts (proxy) beat the earnings print for WMT?")
print("# SIGNAL = LABELLED PROXY of WMT quarterly foot traffic (stylised, ordinal use only — "
      "NOT real satellite data / not a live feed). See data.py.")

ev_tab = dt.parking_events()
print(f"proxy: WMT quarterly foot-traffic index {dt.EVENT_YEARS[0]}->{dt.EVENT_YEARS[-1]} "
      f"({len(ev_tab)} quarters); parking events: "
      f"{(ev_tab['direction']=='busy').sum()} busy, {(ev_tab['direction']=='slow').sum()} slow, "
      f"{(ev_tab['direction']=='flat').sum()} flat")

if not dt.have_real():
    print("(cache miss — fetching WMT + SPY once)")
    dt.fetch()

prices = dt.load_real()
panel = pd.DataFrame({t: s for t, s in prices.items()}).dropna()
print(data_stamp("Parking-Lot panel (WMT + SPY, adjusted/total-return)", panel, asof=dt.AS_OF))

ev = st.build_event_table(prices, cost_bps=5.0)
ev10 = st.build_event_table(prices, cost_bps=10.0)
inc = ev[ev["included"]]
bz, sl = st.busy(ev), st.slow(ev)
print(f"\nevents resolved: {len(inc)} of {len(ev)} parking quarters have WMT+SPY forward "
      f"coverage ({len(bz)} busy, {len(sl)} slow)")
for reason, n in ev[~ev["included"]]["reason"].value_counts().items():
    print(f"  excluded {n:2d}x {reason}")

print("\n# FORWARD TIMING — BUSY-quarter WMT abnormal return (WMT minus SPY, gross), entered at "
      "the print anchor (zero look-ahead)")
for label, col in (("1-week forward (k=5)", "fwd_s"), ("1-month forward (k=21)", "fwd_l")):
    s = st.one_sample_t(bz[col].values)
    hr = st.hit_rate(bz[col].values)
    print(f"  {label:<24s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  t={s['t']:+.3f}  "
          f"hit {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% "
          f"(Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# CONTRAST — SLOW-quarter WMT abnormal return (folklore says these drift DOWN)")
for label, col in (("1-week forward (k=5)", "fwd_s"), ("1-month forward (k=21)", "fwd_l")):
    s = st.one_sample_t(sl[col].values)
    hr = st.hit_rate(sl[col].values)
    print(f"  {label:<24s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  t={s['t']:+.3f}  "
          f"hit {hr['k']}/{hr['n']}={hr['rate']*100:.1f}%")

print("\n# LONG/SHORT — busy-minus-slow (one-sample t of the signed timing P&L; Welch spread)")
for label, col in (("1-week forward (k=5)", "fwd_s"), ("1-month forward (k=21)", "fwd_l")):
    ls = st.longshort_returns(ev, col)
    ost = st.one_sample_t(ls)
    r2 = st.two_sample_t(bz[col].values, sl[col].values)
    rho = st.spearman(inc["yoy"].values, inc[col].values)
    print(f"  {label:<24s} L/S mean={ost['mean']*100:+.3f}%  t={ost['t']:+.3f}  "
          f"(Welch busy-slow {r2['diff']*100:+.3f}%, t={r2['t']:+.3f}; spearman(yoy,fwd)={rho:+.3f})")

print("\n# Sign-shuffle placebo (40 seeds x 250 draws) — is the busy-minus-slow spread luck?")
for label, col in (("1-week L/S", "fwd_s"), ("1-month L/S", "fwd_l")):
    pl = st.placebo_pvalue(ev, col, tail="right")
    print(f"  {label:<12s} (right-tail): observed {pl['obs']*100:+.3f}%  vs placebo mean "
          f"{pl['placebo_mean']*100:+.3f}% (sd {pl['placebo_sd']*100:.3f}%) over "
          f"{pl['n_draws']:,} draws -> p = {pl['p_value']:.4f}")

print("\n# Jackknife (leave-one-out) — 1-month long/short t-stat")
x = st.longshort_returns(ev, "fwd_l")
jk_ts = [st.one_sample_t(np.delete(x, i))["t"] for i in range(len(x))]
print(f"  full-sample t = {st.one_sample_t(x)['t']:+.3f}  |  jackknife t range "
      f"[{min(jk_ts):+.3f}, {max(jk_ts):+.3f}] across {len(x)} leave-one-out draws")

print("\n# TRADABILITY — the busy-minus-slow timing P&L net of costs")
for base, label in (("fwd_s", "1-week L/S"), ("fwd_l", "1-month L/S")):
    g = st.one_sample_t(st.longshort_returns(ev, base))
    n5 = st.one_sample_t(st.longshort_returns(ev, base + "_net"))
    n10 = st.one_sample_t(st.longshort_returns(ev10, base + "_net"))
    print(f"  {label:<12s} gross {g['mean']*100:+.3f}% (t={g['t']:+.2f})  "
          f"net@5bps {n5['mean']*100:+.3f}% (t={n5['t']:+.2f})  "
          f"net@10bps {n10['mean']*100:+.3f}% (t={n10['t']:+.2f})")

print("\n# Synthetic positive control — deterministic, no network")
null_ts = np.array([st.synthetic_detect(bump=0.0, seed=810 + s, k=21)["t"] for s in range(20)])
print(f"  null (bump=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
for bump in (0.02, 0.04):
    planted = st.synthetic_detect(bump=bump, seed=810, k=21)
    print(f"  planted busy/slow link bump=+{bump*100:.0f}% (seed 810): mean L/S {planted['mean']*100:+.3f}%  "
          f"t = {planted['t']:+.2f}  (n={planted['n']} synthetic events)")

print("\n# VERDICT")
print("  (see docs/results.md for the stamped, fingerprinted table)")
