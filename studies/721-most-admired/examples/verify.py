"""Reproducible headline run for Study 721 — Most-Admired.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached admired prices under
``_cache/`` if present (the real-tape numbers, pinned to the as-of), and always runs the
synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from most_admired import data, strategy as st

try:
    from quantlab import repro
    ASOF = "2026-06-30"
except Exception:  # pragma: no cover
    repro = None
    ASOF = None

print("# Most-Admired — Fortune 'World's Most Admired Companies' as a return signal")
print(f"admired table  : {len(data.ADMIRED)} perennial All-Stars (mega-cap); "
      f"spurned proxy {len(data.SPURNED)} survivors; {len(data.DELISTED)} famous low-rep DELISTED (survivorship)")

if data.have_real():
    b = data.load_real()
    if repro is not None:
        b["prices"] = repro.as_of(b["prices"], ASOF)
        cols = sorted([t for t, *_ in data.ADMIRED]) + ["SPY"]
        print(repro.data_stamp("admired month-end (admired+SPY)", b["prices"], cols=cols, asof=ASOF))
    p = b["prices"]

    print("\n# The admiration premium — equal-weight admired book, monthly excess over SPY, HAC t")
    print("#   NAIVE  = own the CURRENT list from day one (look-ahead selection)")
    print("#   LAGGED = own a name only AFTER Fortune first crowns it (no timing look-ahead)")
    print(f"  {'variant':>8} {'n':>4} {'window':>25} {'excess%/yr':>11} {'HAC t':>7} "
          f"{'alpha%/yr':>10} {'alphaT':>7} {'beta':>6} {'sharpe':>7}")
    for lagged, tag in ((False, "NAIVE"), (True, "LAGGED")):
        s = st.summarize(b, lagged=lagged)
        print(f"  {tag:>8} {s['n_months']:>4} {s['start']+'..'+s['end']:>25} "
              f"{s['excess_ann']*100:>10.2f}% {s['hac_t']:>7.2f} "
              f"{s['alpha_ann']*100:>9.2f}% {s['alpha_t']:>7.2f} {s['beta']:>6.2f} {s['sharpe_excess']:>7.2f}")

    print("\n# Placebo — random equal-weight large-cap books vs the admired book's excess over SPY")
    for lagged, start in ((True, "2008-02-01"), (False, "2004-02-01")):
        book = st.admired_book(p, b["admired"], entry=b["entry"], lagged=lagged)
        ex = st.excess_over_market(book, p)
        pv = st.placebo_pvalue(p, data.POOL, k=len(data.ADMIRED), observed_ann=ex.mean() * 12,
                               start=start, n_draws=4000)
        tag = "LAGGED" if lagged else "NAIVE"
        print(f"  {tag:>6}: observed {pv['obs_ann']*100:+6.2f}%/yr  vs random large-cap book "
              f"{pv['placebo_mean_ann']*100:+5.2f}%/yr  ->  p = {pv['p_value']:.3f}  "
              f"(pool full-history n={pv['n_full']})")

    print("\n# Robustness — is the premium the whole roster, or a couple of hindsight winners?")
    print(f"  {'roster':>22} {'excess%/yr':>11} {'HAC t':>7} {'alpha%/yr':>10} {'alphaT':>7} {'beta':>6}")
    no_nv = [r for r in data.ADMIRED if r[0] != "NVDA"]
    variants = [("lagged, all 15", data.ADMIRED),
                ("drop NVDA", no_nv),
                ("drop NVDA & AAPL", [r for r in no_nv if r[0] != "AAPL"])]
    for name, adm in variants:
        book = st.admired_book(p, adm, entry=b["entry"], lagged=True)
        ex = st.excess_over_market(book, p)
        nw = st.newey_west_t(ex.to_numpy())
        mm = st.market_model_alpha(book, p)
        print(f"  {name:>22} {nw['ann']*100:>10.2f}% {nw['t']:>7.2f} "
              f"{mm['alpha_ann']*100:>9.2f}% {mm['alpha_t']:>7.2f} {mm['beta']:>6.2f}")

    print("\n# Long/short: admired (lagged) minus a SURVIVOR spurned proxy (survivorship-biased)")
    ls = st.long_short(p, b["admired"], b["spurned"], entry=b["entry"], lagged=True)
    nw = st.newey_west_t(ls.to_numpy())
    print(f"  admired - spurned = {nw['ann']*100:+.2f}%/yr  HAC t = {nw['t']:.2f}  (n={nw['n']}, not >= 2)")

    print("\n# Costs — the list turns over slowly; frictions are NOT the binding constraint")
    book = st.admired_book(p, b["admired"], entry=b["entry"], lagged=True)
    ex = st.excess_over_market(book, p)
    c = st.net_of_costs(ex, cost_bps=10.0, annual_rebalance_turnover=0.20)
    print(f"  lagged book: gross {c['gross_ann']*100:+.2f}%/yr -> net {c['net_ann']*100:+.2f}%/yr "
          f"(turnover {c['turnover']:.0%}, {c['cost_bps']:.0f} bps one-way; drag {c['drag']*100:.2f}%/yr)")
else:
    print("(no _cache/admired_prices.csv — run data.fetch_prices() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the engine must NOT find a significant premium with edge=0, and MUST light up with a large planted one.")
for edge in (0.0, 0.06):
    syn = data.synthetic_admired(n_names=15, edge_ann=edge, seed=721)
    s = st.summarize(syn, lagged=False)
    print(f"  planted edge={edge*100:>4.0f}%/yr: excess={s['excess_ann']*100:>6.2f}%/yr "
          f"HAC t={s['hac_t']:>6.2f}  alpha={s['alpha_ann']*100:>6.2f}%/yr alpha_t={s['alpha_t']:>6.2f}")
