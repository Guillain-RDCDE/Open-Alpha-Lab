"""Real-tape verification — Study 962 (Do It Yourself). Regenerates docs/results.md.

Reads cached daily total-return closes for XLK / XLE / XLF, their published top-10 names
(today's list and the January-2011 list) and BIL (cash), builds the do-it-yourself
baskets at depths 3 / 5 / 10 under both weighting schemes, and races each against the
fund it is meant to replace: annualised gap, tracking error, HAC *t*, block-bootstrap CI,
excess-of-cash Sharpe race, era cut, cost sweep, rebalance-frequency sweep, calendar-year
table, and the power arithmetic. Network only on ``--fetch``.

    python studies/962-sector-etf-vs-holdings/examples/verify.py            # cache-only
    python studies/962-sector-etf-vs-holdings/examples/verify.py --fetch    # refresh
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from diy_sector import data, strategy as st  # noqa: E402

COST_BPS = 5.0
HEAD_N = 10
HEAD_SCHEME = "eq2011"
SPLIT = "2019-01-01"


def blend(diffs: dict[str, pd.Series]) -> pd.Series:
    """Equal-weight blend of the three per-sector gap series (one honest headline)."""
    return pd.concat(diffs, axis=1).mean(axis=1).dropna()


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()
    cash = px[data.CASH].dropna()

    print(f"as-of {data.AS_OF}  |  sample {px.index[0].date()} -> {px.index[-1].date()}  "
          f"n={len(px)}  fp={data.fingerprint(px)}")
    print(f"tickers: {len(data.TICKERS)} (3 funds + cash + {len(data.TICKERS) - 4} names)")
    print(f"published expense ratio (context only, already inside the fund's tape): "
          f"{data.SECTORS['XLK']['expense_ratio']:.2%}/yr")

    print("\n=== depth x weighting sweep (monthly rebalance, 5 bps one-way) ===")
    print("  fund scheme    N    ann gap   HAC t     TE   exSharpe DIY/fund   worst 12m   turn/yr")
    for f in data.FUNDS:
        for row in st.depth_sweep(px, f, cash, data.basket_weights, cost_bps=COST_BPS):
            print(f"  {row['fund']:4s} {row['scheme']:8s} {row['n']:2d}  "
                  f"{row['ann_gap']:+7.2%}  {row['t_diff']:+5.2f}  {row['tracking_error']:6.2%}  "
                  f"{row['sharpe_diy']:+5.2f}/{row['sharpe_fund']:+5.2f}  "
                  f"{row['worst_12m_shortfall']:+8.1%}  {row['turnover_ann']:6.2f}")

    # ------------------------------------------------------------------ headline
    print(f"\n=== headline: {HEAD_SCHEME} top-{HEAD_N}, equal weight fixed at the sample start ===")
    diffs, diffs_lookahead, diffs_eqnames = {}, {}, {}
    for f in data.FUNDS:
        w = data.basket_weights(f, HEAD_N, HEAD_SCHEME)
        res = st.race(px, f, w, cash, cost_bps=COST_BPS)
        diffs[f] = res["diff"]
        ci = st.bootstrap_gap_ci(res["diff"])
        print(f"  {f}: gap {res['ann_gap']:+.2%} (HAC t {res['t_diff']:+.2f})  "
              f"TE {res['tracking_error']:.2%}  CI [{ci['ci_low']:+.2%}, {ci['ci_high']:+.2%}]  "
              f"corr {res['corr']:.3f}  beta {res['beta']:.2f}")
        print(f"        exSharpe DIY {res['diy']['sharpe']:+.2f} vs fund "
              f"{res['fund_stats']['sharpe']:+.2f} (adv {res['sharpe_adv']:+.2f}, "
              f"t {res['t_sharpe_race']:+.2f})  maxDD {res['dd_diy']:.1%} vs {res['dd_fund']:.1%}  "
              f"worst 12m {res['worst_12m_shortfall']:+.1%}")
        wl = data.basket_weights(f, HEAD_N, "cap2026")
        rl = st.race(px, f, wl, cash, cost_bps=COST_BPS)
        diffs_lookahead[f] = rl["diff"]
        we = data.basket_weights(f, HEAD_N, "eq2026")
        re_ = st.race(px, f, we, cash, cost_bps=COST_BPS)
        diffs_eqnames[f] = re_["diff"]

    d = blend(diffs)
    dl = blend(diffs_lookahead)
    de = blend(diffs_eqnames)
    ci = st.bootstrap_gap_ci(d)
    cil = st.bootstrap_gap_ci(dl)
    ann = (1.0 + d.mean()) ** st.TRADING_DAYS - 1.0
    annl = (1.0 + dl.mean()) ** st.TRADING_DAYS - 1.0
    anne = (1.0 + de.mean()) ** st.TRADING_DAYS - 1.0
    te = float(d.std(ddof=1) * np.sqrt(st.TRADING_DAYS))
    tel = float(dl.std(ddof=1) * np.sqrt(st.TRADING_DAYS))
    print(f"\n  BLEND (equal-weight across the three sectors)")
    print(f"    contemporaneous 2011 list : gap {ann:+.2%}  HAC t {st.hac_t(d):+.2f}  "
          f"TE {te:.2%}  CI [{ci['ci_low']:+.2%}, {ci['ci_high']:+.2%}]  "
          f"worst 12m {st.worst_rolling_shortfall(d):+.1%}")
    print(f"    hindsight 2026 list       : gap {annl:+.2%}  HAC t {st.hac_t(dl):+.2f}  "
          f"TE {tel:.2%}  CI [{cil['ci_low']:+.2%}, {cil['ci_high']:+.2%}]  "
          f"worst 12m {st.worst_rolling_shortfall(dl):+.1%}")

    # --------------------------------------------------- decomposing that difference
    # cap2026 vs eq2011 moves TWO things at once (which names, and how they are
    # weighted). eq2026 (today's names, EQUAL weight) is the control that separates them.
    print(f"\n=== decomposition — hindsight is not the only thing that changes ===")
    print(f"  eq2026 control (today's names, EQUAL weight): gap {anne:+.2%}  "
          f"HAC t {st.hac_t(de):+.2f}")
    print(f"  raw hindsight-minus-contemporaneous difference : {annl - ann:+.2%}/yr")
    print(f"    of which NAME VINTAGE (eq2026 - eq2011, weighting held equal) : {anne - ann:+.2%}/yr")
    print(f"    of which WEIGHTING    (cap2026 - eq2026, names held at 2026)  : {annl - anne:+.2%}/yr")
    print(f"  -> the look-ahead in the membership list is worth {anne - ann:+.2%}/yr; the rest is "
          f"cap- vs equal-weighting and is NOT hindsight.")
    print("  per sector (name-vintage term only):")
    for f in data.FUNDS:
        a_c = (1.0 + diffs[f].mean()) ** st.TRADING_DAYS - 1.0
        a_e = (1.0 + diffs_eqnames[f].mean()) ** st.TRADING_DAYS - 1.0
        a_l = (1.0 + diffs_lookahead[f].mean()) ** st.TRADING_DAYS - 1.0
        print(f"    {f}: eq2011 {a_c:+.2%} -> eq2026 {a_e:+.2%} -> cap2026 {a_l:+.2%}   "
              f"(vintage {a_e - a_c:+.2%}, weighting {a_l - a_e:+.2%})")

    # ------------------------------------------------------------------ power
    fee = data.SECTORS["XLK"]["expense_ratio"]
    years = len(d) / st.TRADING_DAYS
    se_ann = te / np.sqrt(years)
    need_years = (2.0 * te / fee) ** 2
    print(f"\n=== power arithmetic (why this test can never resolve the fee) ===")
    print(f"  blended TE {te:.2%}/yr over {years:.1f} yr -> SE of the annualised gap "
          f"{se_ann:.2%}/yr, i.e. {se_ann / fee:.0f}x the {fee:.2%} fee being hunted")
    print(f"  years of tape needed for |t| = 2 on an {fee:.2%} fee at this TE: {need_years:,.0f}")

    # ------------------------------------------------------------------ eras
    print(f"\n=== era cut (split {SPLIT}) ===")
    for f in data.FUNDS:
        w = data.basket_weights(f, HEAD_N, HEAD_SCHEME)
        for tag, e in st.era_cut(px, f, cash, w, split=SPLIT, cost_bps=COST_BPS).items():
            if e is None:
                continue
            print(f"  {f} {tag:5s} {e['start']} -> {e['end']} (n={e['n_days']}): "
                  f"gap {e['ann_gap']:+.2%} (t {e['t_diff']:+.2f})  TE {e['tracking_error']:.2%}  "
                  f"exSharpe {e['sharpe_diy']:+.2f} vs {e['sharpe_fund']:+.2f}")

    # ------------------------------------------------------------------ sweeps
    print("\n=== cost sweep (XLK, contemporaneous top-10, monthly) ===")
    w_xlk = data.basket_weights("XLK", HEAD_N, HEAD_SCHEME)
    for row in st.cost_sweep(px, "XLK", cash, w_xlk):
        print(f"  {row['cost_bps']:5.1f} bps: gap {row['ann_gap']:+.2%} (t {row['t_diff']:+.2f})  "
              f"TE {row['tracking_error']:.2%}  turn/yr {row['turnover_ann']:.2f}")

    print("\n=== rebalance-frequency sweep (XLK, contemporaneous top-10, 5 bps) ===")
    for row in st.freq_sweep(px, "XLK", cash, w_xlk, cost_bps=COST_BPS):
        print(f"  freq={row['freq']}: gap {row['ann_gap']:+.2%} (t {row['t_diff']:+.2f})  "
              f"TE {row['tracking_error']:.2%}  turn/yr {row['turnover_ann']:.2f}  "
              f"worst 12m {row['worst_12m_shortfall']:+.1%}")

    # ------------------------------------------------- the full contemporaneous grid
    # The rebalance sweep above is XLK-only, so the "only one cell clears |t| = 2" claim
    # is checked here against every fund x depth x frequency combination.
    print("\n=== full contemporaneous grid: how many cells clear |t| = 2? ===")
    grid_hits, grid_n = [], 0
    for f in data.FUNDS:
        for n in (3, 5, 10):
            w = data.basket_weights(f, n, HEAD_SCHEME)
            for row in st.freq_sweep(px, f, cash, w, cost_bps=COST_BPS):
                grid_n += 1
                if abs(row["t_diff"]) >= 2.0:
                    grid_hits.append((f, n, row["freq"], row["ann_gap"], row["t_diff"]))
    print(f"  {len(grid_hits)} of {grid_n} cells (3 funds x 3 depths x 4 frequencies)")
    for f, n, fq, g, t in grid_hits:
        side = "in the fund's favour" if t < 0 else "in the basket's favour"
        print(f"    {f} N={n} freq={fq}: gap {g:+.2%} t {t:+.2f}  ({side})")
    print(f"  expected by chance under a true null at 5% size: {0.05 * grid_n:.1f} cells "
          f"-> {len(grid_hits)} is not evidence of anything")

    print("\n=== calendar-year gap, contemporaneous top-10 (pp of total return) ===")
    tbls = {f: st.calendar_year_table(px, f, cash, data.basket_weights(f, HEAD_N, HEAD_SCHEME),
                                      cost_bps=COST_BPS) for f in data.FUNDS}
    years_idx = sorted(set().union(*[set(t.index) for t in tbls.values()]))
    print("  year  " + "  ".join(f"{f:>8s}" for f in data.FUNDS))
    for y in years_idx:
        cells = []
        for f in data.FUNDS:
            v = tbls[f]["gap_pp"].get(y, float("nan"))
            cells.append(f"{v:+8.1f}")
        print(f"  {y}  " + "  ".join(cells))

    # ------------------------------------------------------------------ synthetic
    print("\n=== synthetic control (machinery proof only — never supports the stamp) ===")
    for ss, tag in [(1.0, "planted fee"), (0.0, "null (no fee)")]:
        gaps, ts = [], []
        for s in range(6):
            p, t = data.synthetic_panel(signal_strength=ss, seed=962 + s)
            det = st.synthetic_detect(p, t, cost_bps=0.0)
            gaps.append(det["ann_gap"])
            ts.append(det["t_diff"])
        gaps, ts = np.array(gaps), np.array(ts)
        p0, t0 = data.synthetic_panel(signal_strength=ss, seed=962)
        det0 = st.synthetic_detect(p0, t0, cost_bps=0.0)
        print(f"  {tag:14s}: planted {t0['planted_fee_ann']:.2%}/yr -> recovered gap "
              f"{gaps.mean():+.3%} (sd {gaps.std(ddof=1):.3%}), mean HAC t {ts.mean():+.2f}, "
              f"|t|>=2 in {(np.abs(ts) >= 2).sum()}/6, TE {det0['tracking_error']:.2%}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
