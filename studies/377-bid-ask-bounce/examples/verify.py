"""Reproducible headline run for Study 377 — Bid-Ask-Bounce (Roll 1984).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached small-cap prices under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic Roll control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bid_ask_bounce import data, strategy as st


def _panel(n_assets=30, n_days=4000, spread=0.004, edge=0.0, seed=377):
    """A synthetic panel of Roll assets sharing one spread/edge (cross-section for the book)."""
    cols = {f"A{j}": data.synthetic_roll(n_days=n_days, spread=spread, edge=edge,
                                         seed=seed + j)["r_obs"] for j in range(n_assets)}
    return pd.DataFrame(cols).dropna()


print("# Bid-Ask-Bounce — Roll (1984) model + a small-cap real-tape reversal proxy")

if data.have_real():
    R = data.load_real()
    names = [c for c in R.columns if c != "IWM"]
    RN = R[names]
    yrs = (RN.index.max() - RN.index.min()).days / 365.25
    print(f"real panel     : {RN.shape[1]} small/mid-cap names, {len(RN)} days "
          f"({RN.index.min().date()} -> {RN.index.max().date()}, {yrs:.1f} years)")
    print(f"avg lag-1 autocorr (apparent reversion) : {st.avg_lag1(RN):+.4f}")
    print(f"Roll-implied effective spread (round-trip): {st.avg_roll_spread(RN)*1e4:.0f} bps")

    print("\n# One-day reversal book — gross, then net of a one-way half-spread cost")
    print(f"  {'half-spread':>12} {'mean(bps)':>10} {'win':>6} {'Sharpe':>7} {'t':>7} {'boot p':>7}")
    for hbps in (0.0, 5.0, 10.0, 20.0):
        s = st.summarize_book(RN, cost_halfspread=hbps / 1e4)
        print(f"  {hbps:>10.1f}bp {s['mean_bps']:>10.3f} {s['win']*100:>5.1f}% "
              f"{s['sharpe']:>7.2f} {s['t']:>7.2f} {s['p']:>7.3f}")

    # break-even half-spread
    sig = -(RN.shift(1)); sig = sig.sub(sig.mean(axis=1), axis=0)
    gr = sig.abs().sum(axis=1); w = sig.div(gr.where(gr > 0), axis=0).fillna(0.0)
    to = (w - w.shift(1)).abs().sum(axis=1).dropna().mean()
    gm = (w * RN).sum(axis=1).dropna().mean()
    be = gm / to
    print(f"\n# Break-even: gross {gm*1e4:.2f} bps/day at one-way turnover {to:.2f}")
    print(f"  break-even half-spread = {be*1e4:.2f} bps  (round-trip {be*2e4:.2f} bps)")
    print(f"  Roll-implied actual round-trip spread = {st.avg_roll_spread(RN)*1e4:.0f} bps "
          f"-> ~{st.avg_roll_spread(RN)/(2*be):.0f}x wider than break-even")
else:
    print("(no _cache/smallcap_prices.csv — run data.fetch_basket() once to build it)")

print("\n# Synthetic Roll control #1 — the bounce manufactures negative cov1 = -(s/2)^2")
print(f"  {'spread':>8} {'cov1_obs':>12} {'-(s/2)^2':>12} {'ac1_obs':>9} {'roll_est_s':>11}")
for spread in (0.0, 0.005, 0.010, 0.020):
    syn = data.synthetic_roll(spread=spread, edge=0.0, seed=377)
    c1 = st.autocov1(syn["r_obs"].values)
    rs = st.roll_spread(syn["r_obs"].values)
    print(f"  {spread:>8.3f} {c1:>12.3e} {-(spread/2)**2:>12.3e} "
          f"{st.lag1_autocorr(syn['r_obs'].values):>+9.4f} "
          f"{(float('nan') if not np.isfinite(rs) else rs):>11.4f}")

print("\n# Synthetic Roll control #2 — reversal book gross vs net (edge=0 vs a planted edge)")
print("  edge=0 is the PURE BOUNCE (gross>0 but net<0, no tradable edge);")
print("  a large planted reversion survives the spread net of cost.")
half = 0.004 / 2.0
for edge in (0.0, -0.25):
    P = _panel(spread=0.004, edge=edge)
    g = st.summarize_book(P, 0.0)
    n = st.summarize_book(P, half)
    print(f"  edge={edge:+.2f}: gross={g['mean_bps']:>6.2f}bps (t={g['t']:>5.1f})  "
          f"net@20bps-half={n['mean_bps']:>+6.2f}bps (t={n['t']:>6.1f}, Sharpe={n['sharpe']:>5.2f})")
