"""Reproducible headline run for Study 632 — Crypto Cross-Sectional Momentum.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; reads the cached weekly panel under
``_cache/`` (real-tape numbers) and always runs the synthetic control offline.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from crypto_xs_momentum import data, strategy as st

COSTS = (10.0, 25.0, 50.0)

print("# Crypto XS momentum -- weekly winners-minus-losers quintiles, Binance+yfinance")
if data.have_real():
    p = data.load_real()
    span = (p.index.max() - p.index.min()).days / 365.25
    print(f"panel         : {p.shape[0]} weeks x {p.shape[1]} coins, "
          f"{p.index.min().date()} -> {p.index.max().date()} ({span:.1f} years, "
          f"last complete week ends {(p.index.max() + __import__('pandas').Timedelta(days=6)).date()})")
    print(f"fingerprint   : {data.fingerprint(p)}")
    print(f"delisted names: {len(data.BINANCE_DELISTED)} of {p.shape[1]} "
          f"({', '.join(data.BINANCE_DELISTED)})")

    print("\n# Headline -- k-week formation, 1-week hold, one-week execution lag")
    RES = {}
    for k in (1, 2, 3, 4):
        res = st.run_xs_momentum(p, k=k)
        RES[k] = res
        s = st.summarize(res["wml"])
        ex = st.summarize(res["win"] - res["mkt"])
        star = "  <== headline (the claim as stated)" if k == 1 else ""
        print(f"  k={k}: {s['n']} wks from {res.index.min().date()} | "
              f"WML {s['mean_bps']:+.1f} bps/wk (ann {s['ann_pct']:+.1f}%)  "
              f"HAC t={s['hac_t']:+.2f}  SR={s['sharpe']:.2f}  hit={s['hit_pct']:.1f}% | "
              f"WIN-MKT {ex['mean_bps']:+.1f} bps/wk t={ex['hac_t']:+.2f}{star}")
    res = RES[1]
    sm = st.summarize(res["mkt"])
    sw = st.summarize(res["win"])
    print(f"  context: EW market {sm['mean_bps']:+.1f} bps/wk (HAC t={sm['hac_t']:+.2f}); "
          f"winners leg {sw['mean_bps']:+.1f} bps/wk (t={sw['hac_t']:+.2f}); "
          f"avg cross-section n={res['n'].mean():.1f} coins")

    print("\n# Sub-periods (k=1 WML)")
    for a, b, lab in [("2017-01-01", "2019-12-31", "2017-2019 (early)"),
                      ("2020-01-01", "2021-12-31", "2020-2021 (bull) "),
                      ("2022-01-01", "2022-12-31", "2022      (bear) "),
                      ("2023-01-01", "2026-06-28", "2023-2026 (late) ")]:
        s = st.sub_period(res, a, b)
        print(f"  {lab}: n={s['n']:>3} | {s['mean_bps']:+7.1f} bps/wk "
              f"(ann {s['ann_pct']:+7.1f}%)  HAC t={s['hac_t']:+.2f}  hit={s['hit_pct']:.1f}%")

    print("\n# Third axis -- does it survive the bear? (lagged BTC>30w-SMA regime split)")
    rs = st.regime_split(res, st.btc_regime(p))
    for lab in ("bull", "bear"):
        s = rs[lab]
        print(f"  {lab}: n={s['n']:>3} | WML {s['mean_bps']:+7.1f} bps/wk  "
              f"HAC t={s['hac_t']:+.2f}  SR={s['sharpe']:.2f}")
    print(f"  Welch t (bull - bear difference) = {rs['welch_t_diff']:+.2f}")

    print("\n# Costs -- one-way bps x traded NAV; shorts pay 10%/yr borrow (k=1)")
    print(f"  turnover/wk: winners leg {res['to_win'].mean():.2f}x NAV, "
          f"losers leg {res['to_lose'].mean():.2f}x NAV")
    for c in COSTS:
        nw = st.summarize(st.net_wml(res, c))
        ex = st.summarize(st.net_long_only(res, c) - res["mkt"])
        print(f"  c={c:>4.0f} bps: WML net {nw['mean_bps']:+7.1f} bps/wk "
              f"(ann {nw['ann_pct']:+7.1f}%)  t={nw['hac_t']:+.2f} | "
              f"long-only WIN net minus mkt {ex['mean_bps']:+6.1f} bps/wk  t={ex['hac_t']:+.2f}")

    print("\n# Placebo -- random ranks, same machinery, averaged over 20 seeds")
    pl = st.placebo_random_ranks(p, k=1, n_seeds=20)
    print(f"  shuffled WML mean = {pl['mean_bps']:+.1f} +/- {pl['std_bps']:.1f} bps/wk "
          f"across {pl['n_seeds']} seeds; mean HAC t = {pl['mean_hac_t']:+.2f}")
else:
    print("(no _cache/xs_weekly_panel.parquet -- run data.fetch_panel(fetch=True) once)")

print("\n# Synthetic control -- deterministic, no network (machinery proof only)")
for rho in (0.0, 0.10):
    ps = data.synthetic_panel(rho=rho, seed=632)
    r = st.run_xs_momentum(ps, k=1)
    s = st.summarize(r["wml"])
    lab = "null (no planted momentum)  " if rho == 0 else "planted 1-wk momentum rho=.1"
    print(f"  {lab}: WML {s['mean_bps']:+7.1f} bps/wk  HAC t={s['hac_t']:+.2f}")
