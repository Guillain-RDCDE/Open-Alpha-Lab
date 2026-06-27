"""Reproducible headline run for Study 534 — Revenue-Surprise-Drift (Jegadeesh-Livnat 2006).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached basket prices + EDGAR revenue
events under ``_cache/`` if present (the real-tape numbers), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from revenue_drift import data, strategy as st

NDRAWS = 20_000

print("# Revenue-Surprise-Drift — SUR event study on a fixed large-cap basket "
      "(yfinance prices + EDGAR revenue)")
if data.have_real():
    prices, ev = data.load_real()
    span = (ev["filed"].max() - ev["filed"].min()).days / 365.25
    print(f"events         : {len(ev)}  ({ev['ticker'].nunique()} names, "
          f"{ev['filed'].min().date()} -> {ev['filed'].max().date()}, {span:.1f} years)")
    print(f"surprise       : standardized unexpected revenue (SUR), seasonal RW / own vol")

    print("\n# SUR long-short by revenue-surprise quintile (1-day entry lag, top - bottom)")
    print(f"  {'H':>3} {'n':>5} {'top%':>7} {'bot%':>7} {'LS%':>7} {'win%':>5} "
          f"{'t':>6} {'p_plac':>7}")
    for h in st.HORIZONS:
        s = st.summarize(prices, ev, h, surprise_col="sur", n_draws=NDRAWS)
        print(f"  {h:>3} {s['n_events']:>5} {s['top_mean']*100:>6.2f} "
              f"{s['bot_mean']*100:>6.2f} {s['ls_mean']*100:>6.2f} "
              f"{s['ls_win']*100:>4.0f} {s['t']:>6.2f} {s['p_placebo']:>7.3f}")

    print("\n# Net of one-way costs (10 bps/leg x 4 legs) + 50 bps/yr borrow on the short leg")
    print(f"  {'H':>3} {'gross%':>8} {'net%':>8}")
    for h in st.HORIZONS:
        c = st.net_of_costs(prices, ev, h, cost_bps=10.0, borrow_bps_ann=50.0,
                            surprise_col="sur")
        print(f"  {h:>3} {c['gross']*100:>7.2f} {c['net']*100:>7.2f}")

    print("\n# 20-day drift by SUR quintile (low -> high) — the monotonicity")
    fr = st.event_drift_frame(prices, ev, 20)
    bm = st.bucket_means(fr, "sur")
    print("  " + "  ".join(f"Q{i+1}:{x*100:+.2f}%" for i, x in enumerate(bm)))

    print("\n# Robustness — quintile count at H=20")
    for nb in (3, 5, 10):
        s = st.summarize(prices, ev, 20, surprise_col="sur", n_buckets=nb, n_draws=NDRAWS)
        print(f"  n_buckets={nb:>2}: LS={s['ls_mean']*100:>5.2f}%  t={s['t']:>5.2f}  "
              f"p={s['p_placebo']:.3f}")

    print("\n# Within-quarter block placebo at H=20 (respects filing-season clustering)")
    fr20 = st.event_drift_frame(prices, ev, 20)
    bp = st.block_placebo_pvalue(fr20, "sur", n_draws=5_000)
    print(f"  block-placebo p = {bp:.3f}")

    print("\n# Third axis — does the revenue drift add info BEYOND the EPS surprise?")
    eps_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            "..", "..", "363-pead-drift", "_cache",
                                            "pead_events.csv"))
    if os.path.exists(eps_path):
        import pandas as pd
        eps = pd.read_csv(eps_path, parse_dates=["date"])
        fr20e = st.attach_eps(st.event_drift_frame(prices, ev, 20), eps)
        inc = st.incremental_to_eps(fr20e, "sur")
        print(f"  matched events with an EPS surprise: {inc['n_matched']}")
        for lab, d in inc["strata"].items():
            print(f"    within {lab}: n={d['n']:>4}  SUR LS={d['ls_mean']*100:>6.2f}%  t={d['t']:>6.2f}")
        print(f"  pooled within-EPS-strata SUR long-short: {inc['pooled_ls_mean']*100:+.2f}%  "
              f"t = {inc['pooled_t']:.2f}")
    else:
        print("  (study 363 EPS cache not found — incremental test skipped)")
else:
    print("(no _cache/rev_prices.csv — run data.fetch_panel() once to build the cache)")

print("\n# Synthetic positive control — deterministic, no network")
print("  detector + inference must recover a PLANTED drift and must NOT manufacture")
print("  significance when the true edge is 0.")
for edge in (0.0, 0.08):
    px, e = data.synthetic_rev(edge=edge, seed=534)
    s = st.summarize(px, e, 20, n_draws=5_000)
    print(f"  planted edge={edge:+.2f}: events={s['n_events']:>4}  "
          f"LS20={s['ls_mean']*100:>6.2f}%  t={s['t']:>6.2f}  "
          f"p_placebo={s['p_placebo']:.3f}  win={s['ls_win']*100:.0f}%")
