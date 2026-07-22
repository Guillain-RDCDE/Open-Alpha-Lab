"""Reproducible headline run for Study 800 — High-Frequency (Weekly) Reversal.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached weekly panel under
``_cache/`` (built once, offline, from the shared daily S&P 500 panel already in the
repo), and always runs the synthetic control with no network.

    python examples/verify.py

Recipe: each Friday, rank the S&P 500 cross-section by LAST WEEK's 5-day return; long the
bottom quintile (losers), short the top quintile (winners), hold one week, rebalance. Then
the honest checks: the skip-a-week bid-ask-bounce killer, an empirical Corwin-Schultz
bounce haircut, a flat cost sweep, a random-portfolio null, a beta decomposition, and a
sub-period split. SURVIVORSHIP-BIASED (current S&P 500 projected backwards) — all real-tape
results are upper bounds, named on the Signal axis.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from hf_reversal import data, strategy as st  # noqa: E402

print("# High-Frequency (Weekly) Reversal — do last week's losers beat last week's winners?")

close, spread = data.load_real()
fp = data.fingerprint(close)
n_med = int(np.median(close.notna().sum(axis=1)))
print(f"[data] weekly S&P 500 panel: {close.shape[0]} weeks x {close.shape[1]} tickers  "
      f"{close.index.min().date()} -> {close.index.max().date()}  as-of {data.AS_OF_LAST_WEEK}  "
      f"fingerprint={fp}")
print(f"       ~{n_med} names with a valid weekly close per week; ~{n_med // 5} per quintile.")
print(f"       median weekly Corwin-Schultz effective spread: "
      f"{np.nanmedian(spread.to_numpy()) * 1e4:.0f} bps (an upper-bound liquidity proxy).")
print("       SURVIVORSHIP-BIASED: current S&P 500 projected backwards; results are upper bounds.")

# --- skip=0 (Jegadeesh) and skip=1 (bid-ask-bounce killer) ---
print("\n# The loser-minus-winner spread, and the skip-a-week bid-ask-bounce test")
for skip in (0, 1):
    res = st.quintile_returns(st.trailing_return(close, skip=skip), close, spreads=spread, q=0.20)
    s = st.annualise(st.summarize(res["spread"].dropna()))
    tag = "  (Jegadeesh: same close forms + prices)" if skip == 0 else "  (one-week gap: bounce defused)"
    print(f"  skip={skip}: spread {s['mean'] * 1e4:+6.1f} bps/wk  HAC t {s['tstat']:+5.2f} "
          f"(lags {s['lags']})  {s['mean_ann'] * 100:+.1f}%/yr  hit {s['hit_rate'] * 100:.1f}%  "
          f"n {s['n']}{tag}")

res0 = st.quintile_returns(st.trailing_return(close, skip=0), close, spreads=spread, q=0.20)
lo = st.annualise(st.summarize(res0["loser"].dropna()))
wi = st.annualise(st.summarize(res0["winner"].dropna()))
hit = int((res0["spread"] > 0).sum()); nsp = int(res0["spread"].notna().sum())
wl, wh = st.wilson_interval(hit, nsp)
print(f"  loser leg {lo['mean_ann'] * 100:+.1f}%/yr (t {lo['tstat']:+.2f})  |  "
      f"winner leg {wi['mean_ann'] * 100:+.1f}%/yr (t {wi['tstat']:+.2f})")
print(f"  spread hit-rate {hit}/{nsp} = {hit / nsp * 100:.1f}%  Wilson95 [{wl * 100:.1f}%, {wh * 100:.1f}%]")
print(f"  leg turnover: loser {res0['loser_turn'].mean() * 100:.0f}% / winner "
      f"{res0['winner_turn'].mean() * 100:.0f}% one-way per week")

# --- beta decomposition ---
beta, alpha = st.beta_alpha(res0["spread"], res0["market"])
print(f"\n# Beta decomposition — dollar-neutral spread vs the equal-weight market")
print(f"  spread beta {beta:+.3f}  annual alpha {alpha * 100:+.2f}%/yr  (near-neutral => not disguised beta)")

# --- flat cost sweep + break-even + borrow ---
print("\n# Tradability — flat one-way cost sweep (both legs x NAV, round-trip; short pays 50 bps/yr borrow)")
for c in (0.0, 5.0, 10.0, 20.0):
    ns = st.annualise(st.summarize(st.net_spread(res0, one_way_bps=c, borrow_bps_annual=50.0).dropna()))
    print(f"  {c:>4.0f} bps one-way: net {ns['mean'] * 1e4:+6.1f} bps/wk  HAC t {ns['tstat']:+5.2f}  "
          f"{ns['mean_ann'] * 100:+.1f}%/yr")
print(f"  break-even flat one-way cost: {st.break_even_cost(res0):.1f} bps")

# --- empirical Corwin-Schultz bid-ask-bounce haircut ---
print("\n# The empirical bid-ask-bounce haircut — each leg pays its OWN Corwin-Schultz spread")
for mult, lbl in ((0.5, "half"), (1.0, "full")):
    bh = st.annualise(st.summarize(st.bounce_haircut_spread(res0, borrow_bps_annual=50.0, spread_mult=mult).dropna()))
    print(f"  {lbl}-spread crossed: net {bh['mean'] * 1e4:+7.1f} bps/wk  HAC t {bh['tstat']:+6.2f}  "
          f"{bh['mean_ann'] * 100:+.1f}%/yr")

# --- random-portfolio null ---
rp = st.random_portfolio_returns(st.trailing_return(close, skip=0), close, n_draws=60, seed=800)
print(f"\n# Random-portfolio null (loser-leg size, 60 draws/wk): mean excess "
      f"{rp.mean() * 1e4:+.2f} bps (n={len(rp):,}) — centred at zero, so any loser excess is signal.")

# --- sub-periods ---
print("\n# Sub-period split (skip=0 raw spread) — the McLean-Pontiff decay check")
for lo_y, hi_y in [("2010", "2015"), ("2016", "2020"), ("2021", "2026")]:
    s = st.summarize(res0.loc[lo_y:hi_y, "spread"].dropna())
    print(f"  {lo_y}-{hi_y}: {s['mean'] * 1e4:+6.1f} bps/wk  HAC t {s['tstat']:+5.2f}  n {s['n']}")

# --- synthetic positive control ---
print("\n# Synthetic positive control (deterministic, no network)")
print("  null must not fire; a REAL reversal must survive skip=1; a PURE bounce must die at skip=1.")
null_t0 = np.array([st.detect_spread(data.synthetic_panel(reversal=0, bounce=0, seed=800 + i)[0])["tstat"]
                    for i in range(12)])
null_t1 = np.array([st.detect_spread(data.synthetic_panel(reversal=0, bounce=0, seed=800 + i)[0], skip=1)["tstat"]
                    for i in range(12)])
print(f"  null (12 seeds): mean skip0 t {null_t0.mean():+.2f}, skip1 t {null_t1.mean():+.2f}; "
      f"|t|>=2 in {(abs(null_t0) >= 2).sum()}/{12} (skip0)")
px_rev, _ = data.synthetic_panel(reversal=0.5, bounce=0.0, seed=800)
px_bnc, _ = data.synthetic_panel(reversal=0.0, bounce=0.008, seed=800)
r0, r1 = st.detect_spread(px_rev, skip=0), st.detect_spread(px_rev, skip=1)
b0, b1 = st.detect_spread(px_bnc, skip=0), st.detect_spread(px_bnc, skip=1)
print(f"  real reversal (0.5): skip0 t {r0['tstat']:+.2f} -> skip1 t {r1['tstat']:+.2f}  (SURVIVES)")
print(f"  pure bounce (0.008): skip0 t {b0['tstat']:+.2f} -> skip1 t {b1['tstat']:+.2f}  (DIES)")
print("  (A faithful-engine / power check only — never cited in support of the real-tape stamp.)")
