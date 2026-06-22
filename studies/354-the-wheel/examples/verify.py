"""Reproducible headline run for Study 354 — The Wheel.

Prints every number quoted in docs/results.md and frozen into notebooks/build_notebooks.py
(the ``R`` dict). Deterministic. The real-tape block needs the cached SPY+VIX parquet under
``_cache/`` (pass ``--fetch`` once to download it via yfinance); the synthetic positive
control and the Black-Scholes identity run offline with no network.

    python examples/verify.py            # uses the cache
    python examples/verify.py --fetch    # download SPY+VIX first, then run
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from the_wheel import data, strategy as st

AS_OF = "2026-06-18"  # last cached SPY/VIX bar; the stamped sample drops the partial month


def main(do_fetch: bool = False) -> None:
    if do_fetch:
        data.fetch(do_fetch=True)

    df = data.fetch()
    if df.empty:
        print("no cache — run `python examples/verify.py --fetch` once to download SPY+VIX.")
        return

    m = data.monthly(df).iloc[:-1]  # drop the in-progress final month (house rule)
    spy = m["spy_ret"].to_numpy()
    w = st.run_wheel(m)

    print(f"# The Wheel — real tape (SPY total-return + ^VIX as Black-Scholes IV)")
    print(f"as-of (last bar)   : {AS_OF}")
    print(f"months (stamped)   : {len(m)}  [{m.index.min().date()} .. {m.index.max().date()}]")
    print(f"SPY level range    : {m['spy_open'].min():.2f} – {m['spy_close'].max():.2f}")
    print(f"VIX mean / median  : {m['vix_open'].mean()*100:.1f}% / {m['vix_open'].median()*100:.1f}%")
    print(f"avg premium kept/mo: {w['premium'].mean()*100:.2f}%")

    print("\n# Annualised summaries (monthly compounding)")
    for name, r in [("Wheel", w["wheel_ret"]), ("Buy&Hold SPY", spy)]:
        s = st.summary(r)
        print(f"  {name:13s} CAGR {s['cagr']*100:6.2f}%  vol {s['vol']*100:5.1f}%  "
              f"Sharpe {s['sharpe']:.2f}  MDD {s['mdd']*100:6.1f}%  skew {s['skew']:+.2f}  "
              f"final {s['final']:.2f}x")
    cash = st.summary(np.full(len(m), 0.03 / 12))
    print(f"  {'Cash @3%/yr':13s} CAGR {cash['cagr']*100:6.2f}%  final {cash['final']:.2f}x")

    print("\n# Signal — paired t of the Wheel's monthly EXCESS return over buy-and-hold")
    pt = st.paired_t(w["wheel_ret"], spy)
    print(f"  full sample : mean/mo {pt['mean_monthly']*100:+.3f}%  ann {pt['ann_drag']*100:+.2f}%  "
          f"t={pt['t']:+.2f}  n={pt['n']}  win {pt['win_rate']*100:.0f}%")
    half = len(m) // 2
    for lab, sl in [("first half ", slice(0, half)), ("second half", slice(half, None))]:
        p = st.paired_t(w["wheel_ret"].to_numpy()[sl], spy[sl])
        idx = m.index[sl]
        print(f"  {lab} : {idx.min().date()}..{idx.max().date()}  ann {p['ann_drag']*100:+.2f}%  "
              f"t={p['t']:+.2f}  n={p['n']}")

    print("\n# Upside / downside capture — the short-vol signature")
    cap = st.capture(m, w)
    print(f"  up months   (n={cap['n_up']}): SPY {cap['spy_up_mean']*100:+.2f}%  "
          f"Wheel {cap['wheel_up_mean']*100:+.2f}%  capture {cap['up_capture']*100:.0f}%")
    print(f"  down months (n={cap['n_dn']}): SPY {cap['spy_dn_mean']*100:+.2f}%  "
          f"Wheel {cap['wheel_dn_mean']*100:+.2f}%  capture {cap['dn_capture']*100:.0f}%")

    print("\n# Tradability — cost sweep (one-way cost per option written)")
    for c in (0.0, 0.0010, 0.0025, 0.0050):
        wc = st.run_wheel(m, cost_frac=c)
        ptc = st.paired_t(wc["wheel_ret"], spy)
        sc = st.summary(wc["wheel_ret"])
        print(f"  cost {c*100:.2f}%/option: edge ann {ptc['ann_drag']*100:+.2f}%  "
              f"t={ptc['t']:+.2f}  Wheel CAGR {sc['cagr']*100:.2f}%")

    print("\n# Synthetic positive control — the Wheel IS the option payoff")
    print("  (vix_premium=1.0 => option priced at the world's own vol => no VRP)")
    for vp, crash in ((1.0, 0.0), (1.0, 0.01), (1.20, 0.01)):
        S = data.synthetic_months(n=6000, mu_ann=0.07, sigma_ann=0.16,
                                  crash_prob=crash, vix_premium=vp, seed=354)
        ws = st.run_wheel(S)
        ps = st.paired_t(ws["wheel_ret"], S["spy_ret"].to_numpy())
        cs = st.capture(S, ws)
        sk = st.summary(ws["wheel_ret"])["skew"]
        print(f"  vix_prem={vp:.2f} crash={crash*100:.0f}%: edge ann {ps['ann_drag']*100:+.2f}% "
              f"(t={ps['t']:+.1f})  up-cap {cs['up_capture']*100:.0f}%  dn-cap "
              f"{cs['dn_capture']*100:.0f}%  skew {sk:+.2f}")

    print("\n# Black-Scholes ATM identity — engine == closed form (r=0: call==put)")
    fi = st.fair_premium_identity(np.array([0.10, 0.20, 0.40]))
    for s, c, p, cf in zip([0.10, 0.20, 0.40], np.atleast_1d(fi["call"]),
                           np.atleast_1d(fi["put"]), np.atleast_1d(fi["closed_form"])):
        print(f"  sigma={s*100:.0f}%: call={c:.4f}  put={p:.4f}  closed={cf:.4f}  "
              f"match={bool(np.isclose(c, cf) and np.isclose(c, p))}")


if __name__ == "__main__":
    main(do_fetch="--fetch" in sys.argv)
