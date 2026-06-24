"""Reproducible headline run for Study 426 — Know Sure Thing (KST).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; reads the cached SPY daily tape under
``_cache/`` (the real-tape numbers) and always runs the synthetic control with no network.

    python examples/verify.py

The stamped run pins the as-of to the last fully-closed session (2026-06-23) — the
in-progress current bar is dropped, per house rule (no partial periods in a stamped run).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from know_sure_thing import data, strategy as st

ASOF = "2026-06-23"     # last fully-closed session; drop the in-progress bar
COST_BPS = 1.0
N_PERM = 5000

print("# KST timing rule vs buy-and-hold / SMA-200 / MACD on SPY daily (yfinance)")
if data.have_real("SPY"):
    bars = data.load_real("SPY").loc[:ASOF]
    yrs = len(bars) / st.TRADING_DAYS
    print(f"as-of {ASOF}  rows={len(bars)}  "
          f"{bars.index.min().date()} -> {bars.index.max().date()}  "
          f"{yrs:.1f} years  fingerprint={data.fingerprint(bars)}")

    res = st.run_experiment(bars, cost_bps=COST_BPS, rf_annual=0.0, n_perm=N_PERM)
    print("\n# The race — net excess-of-cash Sharpe (one-way 1 bp x NAV, 1-day lag)")
    print(f"  {'arm':14s} {'Sharpe':>7} {'CAGR%':>7} {'maxDD%':>7} {'HAC t':>7} "
          f"{'expo':>5} {'trades':>7} {'turn/yr':>8}")
    for k in ("kst", "buy_and_hold", "sma200", "macd"):
        s = res[k]
        print(f"  {k:14s} {s['sharpe_excess']:>7.3f} {s['cagr_net']*100:>7.2f} "
              f"{s['maxdd_net']*100:>7.1f} {s['hac_t']:>7.2f} {s['exposure']:>5.2f} "
              f"{s['n_trades']:>7d} {s['ann_turnover']:>8.1f}")

    print(f"\n# Sign-flip permutation placebo on the KST long/flat schedule "
          f"({N_PERM} draws)")
    print(f"  observed gross Sharpe={res['perm']['obs_sharpe']:.3f}  "
          f"p(random sign-schedule >= observed)={res['perm']['p_value']:.4f}")

    kn = res["books"]["kst"]["net"].to_numpy()
    bn = res["books"]["buy_and_hold"]["net"].to_numpy()
    d = kn - bn
    print("\n# Does KST beat buy-and-hold? (KST net minus BAH net, daily)")
    print(f"  spread = {d.mean()*st.TRADING_DAYS*100:+.2f}%/yr   HAC t = {st.hac_t(d):+.2f}")

    print("\n# Cost sweep — KST vs buy-and-hold net excess Sharpe")
    for r in st.cost_sweep(bars):
        print(f"  one-way {r['cost_bps']:>4.1f} bp: KST={r['kst_sharpe']:.3f}  "
              f"BAH={r['bah_sharpe']:.3f}")

    print("\n# Long/short variant (flip to short below the signal line)")
    rls = st.run_experiment(bars, cost_bps=COST_BPS, allow_short=True, n_perm=2000)
    s = rls["kst"]
    print(f"  KST L/S Sharpe={s['sharpe_excess']:.3f}  CAGR={s['cagr_net']*100:.2f}%  "
          f"maxDD={s['maxdd_net']*100:.1f}%  HAC t={s['hac_t']:.2f}")
else:
    print("(no _cache/kst_SPY_1d.parquet — run data.load_real('SPY', fetch=True) once)")

print("\n# Synthetic positive control — deterministic, no network")
print("  value-add = Sharpe/return of (KST net - buy&hold net): must climb toward 0+ as")
print("  planted trend rises, and be NEGATIVE with no trend (timing only sacrifices beta).")
for trend in (0.00, 0.20, 0.35):
    b, _ = data.synthetic_panel(n_days=4000, trend=trend, seed=426)
    r = st.run_experiment(b, cost_bps=COST_BPS, n_perm=1)
    kk = r["books"]["kst"]["net"].to_numpy()
    bb = r["books"]["buy_and_hold"]["net"].to_numpy()
    dd = kk - bb
    print(f"  trend={trend:+.2f}: KST Sh={r['kst']['sharpe_excess']:.3f}  "
          f"BAH Sh={r['buy_and_hold']['sharpe_excess']:.3f}  "
          f"spread={dd.mean()*st.TRADING_DAYS*100:+.2f}%/yr  spread t={st.hac_t(dd):+.2f}")
