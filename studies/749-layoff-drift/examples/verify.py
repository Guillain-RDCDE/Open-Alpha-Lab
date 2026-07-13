"""Reproducible headline run for Study 749 — Layoff-Drift.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached daily closes under
``_cache/`` if present (the real-tape event study), and always runs the synthetic
positive control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from layoff_drift import data, strategy as st

print("# Layoff-Drift — short/long-window market-model abnormal returns around layoff news")
if data.have_real():
    prices, events = data.load_real()
    print(f"event tickers cached : {sorted(c for c in prices.columns if c != 'SPY')}")
    print(f"events with price data: {len(events)} of {len(data.LAYOFF_EVENTS)} "
          f"(table fingerprint {data.fingerprint(data.LAYOFF_EVENTS)})")

    panel = st.car_panel(prices, events)          # pop [+1,+3] and drift [+4,+63]
    print("\n# Per-event abnormal returns (market model, SPY benchmark, 1-day entry lag)")
    print(panel.to_string(index=False,
                          formatters={"pop": lambda v: f"{v*100:+.2f}%",
                                      "drift": lambda v: f"{v*100:+.2f}%"}))

    daily = st.pooled_daily_drift(prices, events)
    s = st.summarize(panel, prices=prices, tickers=data.TICKERS, n_draws=8000,
                     hac_series=daily)
    print("\n# The two legs")
    for leg in ("pop", "drift"):
        b = s[leg]
        print(f"  {leg:>6}: n={b['n']:>2}  mean={b['mean_pct']:+6.2f}%  "
              f"win={b['win']*100:>4.0f}%  Welch t={b['t']:+5.2f}  "
              f"placebo p={b['p_placebo']:.3f}")
    print(f"  drift HAC t (pooled daily, Newey-West L=5) = {s['drift']['hac_t']:+.2f} "
          f"(daily mean {s['drift']['daily_mean_bps']:+.2f} bps)")

    print("\n# Tradable book — buy +1 day, hold the drift window, sell (one-way 10 bps)")
    dm = panel["drift"].to_numpy(float)
    nc = st.net_of_costs(float(np.nanmean(dm)))
    print(f"  drift gross={nc['gross_pct']:+.2f}%   net@10bps={nc['net_pct']:+.2f}%")

    print("\n# Robustness — vary the pop and drift windows")
    for pw, dw in [((1, 3), (4, 63)), ((0, 1), (2, 42)), ((1, 5), (6, 126))]:
        pn = st.car_panel(prices, events, pop_win=pw, drift_win=dw)
        pp = st.summarize_leg(pn["pop"].to_numpy(float))
        dd = st.summarize_leg(pn["drift"].to_numpy(float))
        print(f"  pop {str(pw):>7} drift {str(dw):>9}: "
              f"pop {pp['mean_pct']:+.2f}% (t={pp['t']:+.2f}) | "
              f"drift {dd['mean_pct']:+.2f}% (t={dd['t']:+.2f})")

    print("\n# Split — the tech 'efficiency wave' (2022+) vs everything else")
    p = panel.copy()
    p["yr"] = p["announce_date"].dt.year
    for lab, mask in [("2022+ tech wave", p["yr"] >= 2022), ("pre-2022", p["yr"] < 2022)]:
        sub = p.loc[mask]
        pp = st.summarize_leg(sub["pop"].to_numpy(float))
        dd = st.summarize_leg(sub["drift"].to_numpy(float))
        print(f"  {lab:>16} (n={len(sub)}): pop {pp['mean_pct']:+.2f}% (t={pp['t']:+.2f}) | "
              f"drift {dd['mean_pct']:+.2f}% (t={dd['t']:+.2f})")
else:
    print("(no _cache — run data.fetch_prices() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  engine must recover PLANTED pop/drift edges and must NOT manufacture significance")
print("  from ~two dozen events when the true edges are 0.")
for pop_bps, drift_bps in [(0.0, 0.0), (0.0, 400.0)]:
    syn = data.synthetic_events(pop_bps=pop_bps, drift_bps=drift_bps, seed=723)
    pt = st.welch_t(syn["pop"])
    dt = st.welch_t(syn["drift"])
    ht = st.hac_t(syn["daily_drift"])
    print(f"  planted pop={pop_bps:+5.0f}bps drift={drift_bps:+6.0f}bps: "
          f"pop mean={syn['pop'].mean()*100:+.2f}% (t={pt:+.2f}) | "
          f"drift mean={syn['drift'].mean()*100:+.2f}% (t={dt:+.2f}, HAC t={ht:+.2f})")
