"""Reproducible headline run for Study 592 — Dual Momentum (GEM).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached ETF closes + ^IRX under
``_cache/`` (the real tape) and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))            # study package
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))  # repo root (quantlab)

from dual_momentum_gem import data, strategy as st

try:
    from quantlab.repro import fingerprint
except Exception:  # pragma: no cover
    fingerprint = None

N_SEEDS = 40


def fmt_perf(p):
    return (f"CAGR {p['cagr']*100:+.2f}%  vol {p['vol']*100:.2f}%  "
            f"Sharpe(excess) {p['sharpe']:.2f}  maxDD {p['max_dd']*100:.1f}%  (n={p['n']})")


print("# Dual Momentum (GEM) — SPY/EFA/AGG(IEF)/^IRX, monthly, yfinance (cache-first)")
if not data.have_real():
    print("(no _cache/gem_prices.csv — run data.fetch() once to build the cache)")
    panel = None
else:
    panel = data.monthly_panel(asof=data.AS_OF)
    gem = st.run_gem(panel)
    bm = st.benchmark_returns(panel, gem.index)
    yrs = len(gem) / 12.0
    fp = fingerprint(panel.dropna()) if fingerprint else "n/a"
    print(f"as-of {data.AS_OF} (last complete month) | fingerprint {fp}")
    print(f"backtest window: {gem.index.min().date()} -> {gem.index.max().date()} "
          f"({len(gem)} months, {yrs:.1f} years)")
    alloc = gem["asset"].value_counts(normalize=True)
    n_sw = int((gem["legs"] == 2).sum())
    print(f"allocation: SPY {alloc.get('SPY',0)*100:.1f}% | EFA {alloc.get('EFA',0)*100:.1f}% | "
          f"BOND {alloc.get('BOND',0)*100:.1f}% of months; {n_sw} switches "
          f"({n_sw/yrs:.2f}/yr)")
    print("bond leg: AGG spliced with IEF before AGG's first monthly return (documented)")
    print("execution: signal from month-end closes, position earns the FOLLOWING month —")
    print("           exactly one one-month decision-to-earn lag, no same-bar fills")

    print("\n# Headline (gross) — GEM vs SPY buy-and-hold vs 60/40, same months")
    pg = st.perf(gem["gross"], bm["RF"])
    ps = st.perf(bm["SPY"], bm["RF"])
    p64 = st.perf(bm["B6040"], bm["RF"])
    print(f"  GEM   : {fmt_perf(pg)}")
    print(f"  SPY   : {fmt_perf(ps)}")
    print(f"  60/40 : {fmt_perf(p64)}")
    print(f"  drawdown ratio GEM/SPY = {pg['max_dd']/ps['max_dd']:.2f} "
          f"('half the drawdown' claim)")
    a_spy = st.active_stats(gem["gross"], bm["SPY"])
    a_64 = st.active_stats(gem["gross"], bm["B6040"])
    print(f"  active GEM-SPY  : {a_spy['mean_bps']:+.1f} bps/mo = {a_spy['ann_pct']:+.2f}%/yr  "
          f"HAC t = {a_spy['hac_t']:+.2f}")
    print(f"  active GEM-60/40: {a_64['mean_bps']:+.1f} bps/mo = {a_64['ann_pct']:+.2f}%/yr  "
          f"HAC t = {a_64['hac_t']:+.2f}")

    print("\n# Costs — one-way bps x NAV per traded leg (2 legs per switch); benchmarks gross")
    for cb in (5.0, 10.0, 20.0):
        g = st.run_gem(panel, cost_bps=cb)
        pf = st.perf(g["net"], bm["RF"])
        a = st.active_stats(g["net"], bm["SPY"])
        print(f"  cost={cb:>4.1f} bps: net CAGR {pf['cagr']*100:+.2f}%  Sharpe {pf['sharpe']:.2f}"
              f"  | active vs SPY {a['ann_pct']:+.2f}%/yr  HAC t = {a['hac_t']:+.2f}")

    print("\n# Lookback placebo grid — is 12 months special, or lucky? (common months)")
    for r in st.lookback_grid(panel):
        print(f"  L={r['lookback']:>2}m: CAGR {r['cagr']*100:+.2f}%  Sharpe {r['sharpe']:.2f}  "
              f"maxDD {r['max_dd']*100:.1f}%  active vs SPY {r['act_ann_pct']:+.2f}%/yr  "
              f"HAC t = {r['hac_t']:+.2f}  (n={r['n']})")

    print(f"\n# Random-switching baseline — GEM's own holdings, shuffled timing, "
          f"{N_SEEDS} seeds (averaged)")
    rb = st.random_switch_baseline(panel, gem, n_seeds=N_SEEDS)
    print(f"  mean Welch t (GEM vs shuffled months) = {rb['welch_t_mean']:+.2f} "
          f"over {rb['n_seeds']} seeds")
    print(f"  shuffled-timing baseline: mean CAGR {rb['base_cagr_mean']*100:+.2f}%  "
          f"mean Sharpe {rb['base_sharpe_mean']:.2f}")
    print(f"  GEM's CAGR beats {rb['beat_share']*100:.1f}% of the {rb['n_seeds']} shuffles "
          f"(GEM CAGR {pg['cagr']*100:+.2f}%)")

    print("\n# Sub-periods — does it live off 2008 alone?")
    subs = [("full sample", {}),
            ("ex-GFC (drop 2007-07..2009-06)", dict(drop=("2007-07-01", "2009-06-30"))),
            ("2010 ->", dict(start="2010-01-01")),
            ("pre-2013", dict(end="2012-12-31")),
            ("post-2013", dict(start="2013-01-01"))]
    for name, kw in subs:
        s = st.subperiod_stats(panel, gem, **kw)
        print(f"  {name:<32} GEM CAGR {s['gem']['cagr']*100:+.2f}% DD {s['gem']['max_dd']*100:.1f}% "
              f"Sh {s['gem']['sharpe']:.2f} | SPY CAGR {s['spy']['cagr']*100:+.2f}% "
              f"DD {s['spy']['max_dd']*100:.1f}% Sh {s['spy']['sharpe']:.2f} | "
              f"active {s['active']['ann_pct']:+.2f}%/yr HAC t = {s['active']['hac_t']:+.2f}")

    print("\n# Third axis — has GEM decayed since publication (2013)?")
    act = (gem["gross"] - bm["SPY"]).dropna()
    pre = act[act.index <= "2012-12-31"]
    post = act[act.index >= "2013-01-01"]
    wt = st.welch_t(pre.values, post.values)
    print(f"  pre-2013 active : {pre.mean()*1e4:+.1f} bps/mo (n={len(pre)})  "
          f"HAC t = {st.hac_tstat(pre):+.2f}")
    print(f"  post-2013 active: {post.mean()*1e4:+.1f} bps/mo (n={len(post)})  "
          f"HAC t = {st.hac_tstat(post):+.2f}")
    print(f"  Welch t of the pre-vs-post difference = {wt:+.2f}")

print("\n# Synthetic machinery control — deterministic, no network (never cited for the stamp)")
print("  null (iid regimes): momentum must NOT beat buy-and-hold; planted persistence: it must.")
for label, pers in (("null   (persistence=0.00)", 0.0), ("planted (persistence=0.94)", 0.94)):
    p = data.synthetic_world(persistence=pers, seed=592)
    g = st.run_gem(p)
    b = st.benchmark_returns(p, g.index)
    a = st.active_stats(g["gross"], b["SPY"])
    pg2 = st.perf(g["gross"], b["RF"])
    ps2 = st.perf(b["SPY"], b["RF"])
    print(f"  {label}: active {a['ann_pct']:+.2f}%/yr  HAC t = {a['hac_t']:+.2f}  "
          f"| maxDD GEM {pg2['max_dd']*100:.0f}% vs B&H {ps2['max_dd']*100:.0f}%  (n={a['n']})")
