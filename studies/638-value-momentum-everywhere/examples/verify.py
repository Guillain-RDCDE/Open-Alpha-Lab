"""Reproducible headline run for Study 638 — Value-Momentum-Everywhere.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached sleeve panels under
``_cache/`` (the real-tape numbers) and always runs the synthetic control offline.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pandas as pd

from value_momentum_everywhere import data, strategy as st
from quantlab import repro

SPLIT = "2011-12-31"      # AMP's sample ends 2011 / publication 2013 — the justified split


def fmt(x, d=2):
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{d}f}"


def leg_line(name, s):
    return (f"  {name:5s} VAL SR={fmt(s['val']['sharpe'])} t={fmt(s['val']['t'])} | "
            f"MOM SR={fmt(s['mom']['sharpe'])} t={fmt(s['mom']['t'])} | "
            f"COMBO SR={fmt(s['combo']['sharpe'])} t={fmt(s['combo']['t'])} | "
            f"rho(V,M)={fmt(s['rho'])}  n={s['n']}  {s['start']}->{s['end']}")


print("# Value-Momentum-Everywhere — AMP (2013) on four free-data sleeves")
if data.have_real():
    panels = data.load_sleeves()
    allm = pd.concat(panels.values(), axis=1)
    fp = repro.fingerprint(allm.reset_index(), cols=list(allm.columns))
    print(f"as-of {data.AS_OF} (futures cache ends 2026-06-11 -> June 2026 partial, dropped)"
          f" | fingerprint {fp}")
    for k, v in panels.items():
        live = v.dropna(how="all")
        print(f"  {k:5s}: {v.shape[1]} assets, monthly {live.index.min().date()} -> "
              f"{live.index.max().date()}")

    print("\n# Per-sleeve legs — VAL (5y reversal, t-59..t-12), MOM (12-1), 50/50 COMBO")
    print("# (L/S thirds, 1-month execution lag, gross, excess-vs-excess)")
    sleeves = {k: st.sleeve_series(v) for k, v in panels.items()}
    stats = {k: st.sleeve_stats(ss) for k, ss in sleeves.items()}
    for k in data.SLEEVES:
        print(leg_line(k, stats[k]))
    rhos = [stats[k]["rho"] for k in data.SLEEVES]
    print(f"  mean per-sleeve rho(VAL,MOM) = {np.nanmean(rhos):+.2f} "
          f"(AMP report ~-0.5 within asset classes)")

    print("\n# GLOBAL everywhere portfolios — equal-weight across live sleeves (>=2)")
    g = {}
    for leg in ("val", "mom", "combo"):
        g[leg] = st.global_series(sleeves, leg)
        h = st.hac_mean(g[leg])
        print(f"  GLOBAL {leg.upper():5s}: SR={fmt(st.sharpe(g[leg]))}  HAC t={fmt(h['t'])}  "
              f"mean={fmt(h['mean_bps'], 1)} bps/mo  n={h['n']}")

    print("\n# THE CLAIM UNDER TEST — the 50/50 combo everywhere")
    hc = st.hac_mean(g["combo"])
    print(f"  global combo: Sharpe {fmt(st.sharpe(g['combo']))}, HAC t = {fmt(hc['t'])} "
          f"({hc['n']} months) -> fails the t>=2 bar (sign is even negative)")

    print("\n# Sub-periods — AMP in-sample overlap vs post-publication")
    for label, sl in (("<=2011 (AMP overlap)", slice(None, SPLIT)),
                      (">=2012 (post-pub)", slice(SPLIT, None))):
        sub = g["combo"].loc[sl]
        h = st.hac_mean(sub)
        print(f"  {label:22s}: SR={fmt(st.sharpe(sub))}  HAC t={fmt(h['t'])}  n={h['n']}")

    print("\n# Third axis — is the combo just diversification arithmetic?")
    dc = st.diversification_check(g["val"], g["mom"])
    print(f"  global rho(VAL,MOM) = {dc['rho']:+.3f}  (n={dc['n']})")
    print(f"  ingredients: SR_val={fmt(dc['sr_val'])}  SR_mom={fmt(dc['sr_mom'])}")
    print(f"  predicted combo SR = (mu_v+mu_m)/2 / sigma_mix = {fmt(dc['predicted'], 3)}")
    print(f"  realized  combo SR = {fmt(dc['realized'], 3)}   "
          f"(gap {abs(dc['realized'] - dc['predicted']):.4f} -> pure arithmetic)")

    print("\n# Random-signal placebo — same construction, noise ranks, "
          f"{st.PLACEBO_SEEDS} seeds (house rule >=20)")
    pl = st.random_placebo(panels)
    print(f"  mean global-combo Sharpe = {fmt(pl['mean_sharpe'])} "
          f"(sd {pl['sd_sharpe']:.2f}), mean HAC t = {fmt(pl['mean_t'])}, "
          f"frac |t|>=2 = {pl['frac_t_ge_2']*100:.0f}%")

    print("\n# Costs — one-way bps x traded notional, EQ short leg pays 50 bps/yr borrow")
    print(f"  avg monthly one-way traded notional (combo) = "
          f"{st.avg_turnover(sleeves):.2f}x NAV")
    for cb in (5.0, 10.0, 20.0):
        c = st.net_global_combo(sleeves, cost_bps=cb)
        print(f"  cost={cb:>4.0f} bps: gross {fmt(c['gross_ann_pct'])}%/yr -> net "
              f"{fmt(c['net_ann_pct'])}%/yr  net SR={fmt(c['net_sharpe'])}  "
              f"net t={fmt(c['net_t'])}")

    print("\n# Robustness — construction variants (global combo)")
    # (a) halves instead of thirds
    def build(variant):
        legs = {}
        for name, mret in panels.items():
            logr = np.log1p(mret)
            cum = logr.cumsum()

            def window(a, b, minlen):
                w = cum.shift(b) - cum.shift(a)
                cnt = logr.notna().rolling(a - b).sum().shift(b)
                return w.where(cnt >= minlen)

            if variant == "mom 12-0 (no skip)":
                val, mom = st.signals(mret)[0], window(12, 0, 10)
            elif variant == "value incl. last year":
                val, mom = -window(60, 1, 48), st.signals(mret)[1]
            else:
                val, mom = st.signals(mret)
            if variant == "halves not thirds":
                def w_half(sig):
                    w = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
                    for t in sig.index:
                        row = sig.loc[t].dropna()
                        if len(row) < st.MIN_ASSETS:
                            w.loc[t] = np.nan
                            continue
                        k = max(1, len(row) // 2)
                        order = row.sort_values()
                        w.loc[t, order.index[-k:]] = 1.0 / k
                        w.loc[t, order.index[:k]] = -1.0 / k
                    return w
                wv, wm = w_half(val), w_half(mom)
            else:
                wv, wm = st.weights(val), st.weights(mom)
            wc = 0.5 * (wv.fillna(0.0) + wm.fillna(0.0))
            dead = ~(wv.notna().any(axis=1) | wm.notna().any(axis=1))
            wc.loc[dead] = np.nan
            rc, _ = st.portfolio(mret, wc)
            legs[name] = {"combo": rc}
        return st.global_series(legs, "combo")

    for variant in ("halves not thirds", "mom 12-0 (no skip)", "value incl. last year"):
        gv = build(variant)
        h = st.hac_mean(gv)
        print(f"  {variant:24s}: SR={fmt(st.sharpe(gv))}  HAC t={fmt(h['t'])}  n={h['n']}")
else:
    print("(cache missing — run value_momentum_everywhere.data.fetch_all() once)")

print("\n# Synthetic control — deterministic, no network (machinery proof only)")
print("# the SAME pipeline on a 4-sleeve planted world: zero edges must stay flat;")
print("# planted value+momentum edges must light up AND obey the diversification arithmetic.")
for ve, me, label in ((0.0, 0.0, "null (no edges)"),
                      (0.004, 0.004, "planted VAL+MOM edges")):
    world = data.synthetic_world(val_edge=ve, mom_edge=me, seed=638)
    ssl = {k: st.sleeve_series(v) for k, v in world.items()}
    gv = st.global_series(ssl, "val")
    gm = st.global_series(ssl, "mom")
    gc = st.global_series(ssl, "combo")
    h = st.hac_mean(gc)
    dc = st.diversification_check(gv, gm)
    print(f"  {label:22s}: VAL SR={fmt(st.sharpe(gv))} MOM SR={fmt(st.sharpe(gm))} "
          f"COMBO SR={fmt(st.sharpe(gc))} HAC t={fmt(h['t'])} | "
          f"rho={dc['rho']:+.2f} predicted SR={fmt(dc['predicted'])} "
          f"realized={fmt(dc['realized'])}")
