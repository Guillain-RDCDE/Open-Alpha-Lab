"""Reproducible headline run for Study 790 — Composite Equity Issuance.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached EDGAR-shares x yfinance-prices
panel under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from composite_issuance import data, strategy as st  # noqa: E402

print("# Composite Equity Issuance — do high 5y composite issuers underperform buyback-ers?")

if not data.have_real():
    print("(cache miss — fetching EDGAR shares + yfinance prices once)")
    data.fetch_real()

raw, adj, sh = data.load_real()
raw, adj, sh = data.drop_partial_last_year(raw, adj, sh)
fp = data.fingerprint(raw, adj, sh)
print(f"panel      : year-ends {raw.index.min().year} -> {raw.index.max().year} "
      f"({len(raw)} year-ends, {sh.shape[1]} names) | as-of {data.AS_OF} | fingerprint {fp}")
print("survivorship: a fixed large-cap SURVIVOR basket (current names, projected back) — the "
      "bias is named on the Signal axis; delisted issuers are absent, which biases the sort.")

ls = st.long_short_series(raw, adj, sh, frac=0.3)
print(f"\n# Long-short panel (formation year-end -> next-year total return), {len(ls)} usable years")
print(f"  {'year':>6} {'long(lo)':>9} {'short(hi)':>10} {'L-S':>8}")
for t, row in ls.iterrows():
    print(f"  {t.year:>6} {row['long_ret']*100:>8.1f}% {row['short_ret']*100:>9.1f}% "
          f"{row['ls_ret']*100:>7.1f}%")

s = st.summarize(raw, adj, sh, frac=0.3)
print("\n# Headline long-short (long LOW composite issuance / short HIGH, frac = 0.3 tails)")
print(f"  n_years      : {s['n_years']}")
print(f"  long  mean   : {s['long_mean']*100:>6.2f}% / yr   (low / negative issuance = buybacks)")
print(f"  short mean   : {s['short_mean']*100:>6.2f}% / yr   (high issuance = dilution)")
print(f"  L-S   mean   : {s['ls_mean']*100:>6.2f}% / yr   (median {s['ls_median']*100:.2f}%)")
print(f"  win-rate     : {s['win']*100:>4.0f}%   Wilson95 [{s['win_lo']*100:.0f}%, {s['win_hi']*100:.0f}%]"
      f"   annual Sharpe {s['sharpe']:.2f}")
print(f"  one-sample t : {s['t']:>5.2f}   (need |t| >= 2, RIGHT sign for the claim)")
print(f"  Newey-West t : {s['nw_t']:>5.2f}   (HAC, 1 lag — overlapping 5y windows)")
print(f"  placebo p    : {s['p_placebo']:.4f}   (label-shuffle: P[shuffled L-S >= real])")

print("\n# Label-shuffle placebo detail (20,000 draws)")
pl = st.placebo_pvalue(raw, adj, sh, frac=0.3, n_draws=20_000)
print(f"  observed L-S {pl['obs_mean']*100:+.2f}% vs placebo mean {pl['placebo_mean']*100:+.2f}% "
      f"-> p = {pl['p_value']:.4f}  (the real sort is beaten by the shuffle "
      f"{pl['p_value']*100:.1f}% of the time)")

print("\n# Costs — one-way bps x both legs x turnover + short borrow")
c = st.net_of_costs(raw, adj, sh, frac=0.3, cost_bps=10.0, borrow_ann_bps=50.0)
print(f"  gross L-S    : {c['gross_mean']*100:>6.2f}% / yr")
print(f"  net @ {c['cost_bps']:.0f}bps x2 + {c['borrow_bps']:.0f}bps borrow : "
      f"{c['net_mean']*100:>6.2f}% / yr  (net t {c['net_t']:.2f})")

print("\n# Robustness — quantile-width sweep")
for r in st.width_sweep(raw, adj, sh, fracs=(0.2, 0.3, 0.4)):
    print(f"  frac {r['frac']:.1f}: L-S {r['ls_mean']*100:>6.2f}% / yr  t {r['t']:>5.2f}  "
          f"Sharpe {r['sharpe']:.2f}")

print(f"\n# Era contrast — formation year < {data.ERA_SPLIT} vs >= {data.ERA_SPLIT}")
ec = st.era_contrast(raw, adj, sh, split_year=data.ERA_SPLIT, frac=0.3)
print(f"  early (n={ec['n_early']}): {ec['early_mean']*100:+.2f}% / yr  (t {ec['t_early']:+.2f})")
print(f"  late  (n={ec['n_late']}): {ec['late_mean']*100:+.2f}% / yr  (t {ec['t_late']:+.2f})")
print(f"  Welch t of the difference (late - early): {ec['t_diff']:+.2f}")

print("\n# Synthetic positive control — deterministic, no network")
print("  the sort must NOT fire on a null world (edge=0) and must recover a planted low-issuance")
print("  edge; averaged over 25 seeds (never a single lucky draw).")
for r in st.synthetic_control(data.synthetic_panel, edges=(0.0, 0.12), n_seeds=25):
    print(f"  planted edge {r['edge']:.2f}: mean one-sample t = {r['mean_t']:+.2f}  "
          f"(mean L-S {r['ls_mean']*100:+.2f}% / yr, {r['n_seeds']} seeds)")

print("\n# VERDICT: the claimed factor (long low / short high) prints a SIGNIFICANTLY NEGATIVE")
print("  spread here — high composite issuers OUTPERFORMED on this survivor basket. The")
print("  Daniel-Titman premium does not replicate; it inverts. Signal NONE / Tradability MIRAGE.")
_ = np  # keep the import referenced for parity with sibling verify scripts
