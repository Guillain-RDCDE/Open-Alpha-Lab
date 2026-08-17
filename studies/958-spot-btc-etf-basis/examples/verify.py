"""Real-tape verification — Study 958 (Spot ETF Basis). Regenerates docs/results.md.

Reads the cached daily total-return closes of BITO (futures wrapper), IBIT and FBTC
(spot wrappers), BTC-USD (the coin) and BIL (cash), then:

  * calibrates the tracking-difference ruler on the two spot wrappers (their drag
    against spot should read their published fee);
  * measures the futures wrapper's drag against spot and against each spot wrapper,
    with both the naive endpoint estimator and the HAC trend slope;
  * turns those drags into an implied annualised futures basis (fee + collateral-yield
    ASSUMPTIONS, swept);
  * runs the era test at the 2024-01-11 spot-ETF launch, a matched twelve-month
    window on each side, and a placebo sweep of arbitrary split dates;
  * prices the long-spot / short-futures harvest across borrow and cost grids.

Network only on ``--fetch``.

    python studies/958-spot-btc-etf-basis/examples/verify.py            # cache-only
    python studies/958-spot-btc-etf-basis/examples/verify.py --fetch    # refresh tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from etf_basis import data, strategy as st  # noqa: E402

FUT = "BITO"
SPOT = "BTC-USD"
LAUNCH = data.LAUNCH


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()
    bito = px[FUT].dropna()
    btc = px[SPOT].dropna()
    ibit = px["IBIT"].dropna()
    fbtc = px["FBTC"].dropna()

    w, r = st.align(bito, btc)
    print(f"{FUT} vs {SPOT}: {w.index[0].date()} -> {w.index[-1].date()}  n={len(w)}  "
          f"fp={data.fingerprint(px[[FUT, SPOT, 'IBIT', 'FBTC', 'BIL']].loc[w.index])}")
    print(f"as-of {data.AS_OF}   launch {LAUNCH}")

    # ------------------------------------------------------------------ ruler
    print("\n=== ruler calibration: the spot wrappers' drag vs the coin ===")
    print("    (should read the published expense ratio; PROXY fees in data.FEES)")
    for tk, s in [("IBIT", ibit), ("FBTC", fbtc)]:
        t_ = st.trend_drag(s, btc)
        nv = st.naive_drag(s, btc)
        print(f"  {tk}: trend {t_['drag_pct']:+.3f}%/yr (se {t_['se_pct']:.3f}, t {t_['t']:+.2f})"
              f"   naive {nv['drag_pct']:+.3f}%/yr (t {nv['t']:+.2f})"
              f"   published fee -{data.FEES[tk] * 100:.2f}%/yr")

    # ------------------------------------------------------------- the drags
    print("\n=== the futures wrapper's drag ===")
    for tag, a, b in [("BITO vs BTC-USD (full)", bito, btc),
                      ("BITO vs BTC-USD (post-launch)", bito, btc),
                      ("BITO vs IBIT", bito, ibit),
                      ("BITO vs FBTC", bito, fbtc)]:
        lo = LAUNCH if "post-launch" in tag else None
        t_ = st.trend_drag(a, b, lo=lo)
        nv = st.naive_drag(a, b, lo=lo)
        mo = st.monthly_drag(a, b, lo=lo)
        print(f"  {tag:30s} n={t_['n']:4d}  trend {t_['drag_pct']:+.3f}%/yr "
              f"(se {t_['se_pct']:.3f}, t {t_['t']:+.2f})   naive {nv['drag_pct']:+.3f}%/yr "
              f"(t {nv['t']:+.2f})   monthly {mo['drag_pct']:+.3f}%/yr "
              f"(t {mo['t']:+.2f}, {mo['n_months']} months)")

    print("\n  NOTE ON INFERENCE: the trend slope regresses a serially dependent level on")
    print("  time, so its HAC t is the most generous of the three. The monthly column is")
    print("  non-overlapping and needs no HAC at all — it is the number the verdict leans on.")
    print("  Residual diagnostics on the trend fit (DF t below -3.4 = stationary residual,")
    print("  so the slope's t is meaningful; near zero = treat that t as an upper bound):")
    for tag, a, b in [("vs BTC-USD", bito, btc), ("vs IBIT", bito, ibit),
                      ("vs FBTC", bito, fbtc)]:
        dg = st.trend_residual_diagnostics(a, b)
        print(f"    BITO {tag:11s} resid AR(1) {dg['ar1']:+.3f}   DF t {dg['df_t']:+.2f}")

    print("\n  HAC bandwidth sensitivity (the residual basis wanders, so long lags matter):")
    for lg in (20, 60, 120, 250):
        a_ = st.trend_drag(bito, btc, lags=lg)
        b_ = st.trend_drag(bito, btc, lo=LAUNCH, lags=lg)
        e_ = st.piecewise_drag(bito, btc, split=LAUNCH, lags=lg)
        m_ = st.piecewise_drag(bito, btc, split=LAUNCH, lo="2023-01-11", hi="2025-01-10", lags=lg)
        print(f"    lags={lg:3d}: full drag t {a_['t']:+7.2f}   post drag t {b_['t']:+7.2f}   "
              f"era change t {e_['t']:+6.2f}   matched-window t {m_['t']:+5.2f}")

    print("\n  block-bootstrap CI on the naive drag (2000 draws, 21-day blocks):")
    for tag, a, b, lo in [("BITO vs BTC-USD full", bito, btc, None),
                          ("BITO vs IBIT", bito, ibit, None)]:
        ci = st.bootstrap_drag_ci(a, b, lo=lo)
        print(f"    {tag:22s} {ci['drag_pct']:+.2f}%/yr  95% CI "
              f"[{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}]")

    # --------------------------------------------------------- implied basis
    print("\n=== implied annualised basis (fee + collateral yield are ASSUMPTIONS) ===")
    for tag, lo, hi in [("pre-launch ", "2021-10-20", "2024-01-10"),
                        ("post-launch", LAUNCH, data.AS_OF)]:
        d = st.trend_drag(bito, btc, lo=lo, hi=hi)
        cash = data.cash_yield(px, lo, hi)
        ib = st.implied_basis(d["drag_pct"], data.FEES[FUT], cash)
        print(f"  {tag}: drag {d['drag_pct']:+.3f}%/yr  cash {cash * 100:.2f}%  "
              f"-> basis {ib['basis_pct']:+.2f}%/yr  (excess of cash {ib['excess_basis_pct']:+.2f}%/yr)")
    cash_post = data.cash_yield(px, LAUNCH, data.AS_OF)
    d_post = st.trend_drag(bito, btc, lo=LAUNCH)["drag_pct"]
    print("  fee sweep on the post-launch basis (BITO fee PROXY 0.95%):")
    for row in st.fee_sweep(d_post, cash_post):
        print(f"    fee {row['fee_pct']:.2f}%  -> basis {row['basis_pct']:+.2f}%/yr  "
              f"(excess {row['excess_basis_pct']:+.2f}%/yr)")

    # ------------------------------------------------------------- era tests
    print("\n=== era test: did the launch compress the carry? ===")
    print("    (positive change = the drag shrank = the carry compressed)")
    full = st.piecewise_drag(bito, btc, split=LAUNCH)
    print(f"  full sample   : pre {full['pre_pct']:+.3f}  post {full['post_pct']:+.3f}  "
          f"change {full['change_pct']:+.3f} (se {full['se_pct']:.3f}, t {full['t']:+.2f})  n={full['n']}")
    matched = st.piecewise_drag(bito, btc, split=LAUNCH, lo="2023-01-11", hi="2025-01-10")
    print(f"  matched +/-12m: pre {matched['pre_pct']:+.3f}  post {matched['post_pct']:+.3f}  "
          f"change {matched['change_pct']:+.3f} (se {matched['se_pct']:.3f}, t {matched['t']:+.2f})  n={matched['n']}")

    print("  matched-window sweep (one window is a CHOICE — here is the whole family;")
    print("   positive change = compression, which is the thing being claimed):")
    for m, row in st.matched_window_sweep(bito, btc, LAUNCH).iterrows():
        print(f"    +/-{m:2d}m: pre {row['pre_pct']:+6.2f}  post {row['post_pct']:+6.2f}  "
              f"change {row['change_pct']:+6.2f} pp (t {row['t']:+6.2f})  n={int(row['n'])}")
    print("   every width is zero or WRONG-SIGNED; none shows compression. The +/-12m row")
    print("   is the flattest of the family, so it is quoted as the headline only because")
    print("   it is the reading MOST favourable to 'the launch changed nothing'.")

    sweep = st.placebo_split_sweep(bito, btc)
    rank = st.placebo_rank(sweep, full["t"])
    print(f"  placebo sweep : {rank['n']} arbitrary split dates, median |t| {rank['median_abs_t']:.2f}, "
          f"max |t| {rank['max_abs_t']:.2f}")
    print(f"                  the launch's |t|={abs(full['t']):.2f} ranks {rank['rank']}/{rank['n']} "
          f"({rank['frac_more_extreme']:.0%} of arbitrary dates are at least as extreme)")

    print("\n=== calendar-year drag, BITO vs BTC-USD (and vs IBIT once it exists) ===")
    tbl = st.annual_drag_table(bito, btc)
    tbl_i = st.annual_drag_table(bito, ibit)
    for y, row in tbl.iterrows():
        vs_i = tbl_i.loc[y, "drag_pct"] if y in tbl_i.index else float("nan")
        cash = data.cash_yield(px, f"{y}-01-01", f"{y}-12-31")
        ib = st.implied_basis(row["drag_pct"], data.FEES[FUT], cash)
        extra = f"  vs IBIT {vs_i:+6.2f}" if np.isfinite(vs_i) else "              "
        print(f"  {y}: drag {row['drag_pct']:+6.2f}%/yr (t {row['t']:+6.2f}, n={int(row['n']):3d})"
              f"{extra}   cash {cash * 100:4.2f}%  implied basis {ib['basis_pct']:+6.2f}%/yr")

    cyc = st.cycle_regression(bito, btc)
    print(f"\n  cycle check (NOT certified): rolling {cyc['window']}d drag on trailing "
          f"{cyc['window']}d BTC log return -> corr {cyc['corr']:+.2f}, slope "
          f"{cyc['slope']:+.2f} %/yr per unit, HAC({cyc['lags']}) t {cyc['t']:+.2f} "
          f"(naive OLS t {cyc['t_ols']:+.2f}; only ~{cyc['n_eff']:.1f} independent windows)")

    # ----------------------------------------------------------- tradability
    print("\n=== harvest: long IBIT / short BITO (monthly reset, one-day lag) ===")
    for row in st.borrow_sweep(ibit, bito):
        print(f"  borrow {row['borrow_pct']:4.1f}%  cost {row['cost_bps']:4.1f}bps: "
              f"{row['ann_pct']:+6.2f}%/yr  vol {row['vol_pct']:4.2f}%  "
              f"Sharpe {row['sharpe']:+5.2f}  HAC t {row['t']:+6.2f}  DD {row['max_dd_pct']:+.2f}%")
    print("  (dollar-neutral and self-financing: the short proceeds fund the long leg and")
    print("   the rebate is ASSUMED to offset the cash given up, so the return above is")
    print("   already excess of cash on both legs; borrow is the incremental loan fee.)")
    base = st.pair_trade(ibit, bito, borrow_ann=0.02, cost_bps=5.0)
    ci = st.bootstrap_sharpe_ci(base["net"])
    print(f"  bootstrap Sharpe CI at 2% borrow / 5 bps: {ci['sharpe']:+.2f} "
          f"[{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}]  share<0 {ci['frac_negative']:.1%}")
    for lg in (20, 60, 250):
        print(f"    HAC({lg:3d}) t on the daily net series: "
              f"{st.newey_west_t(base['net'].to_numpy(), lags=lg):+.2f}")
    mnet = (1.0 + base["net"]).groupby([base.index.year, base.index.month]).prod() - 1.0
    tm = float(mnet.mean() / mnet.std(ddof=1) * np.sqrt(len(mnet)))
    print(f"    non-overlapping monthly cross-check: {mnet.mean() * 12 * 100:+.2f}%/yr, "
          f"plain t {tm:+.2f} over {len(mnet)} months, {(mnet > 0).sum()}/{len(mnet)} positive")
    print("  by year (2% borrow, 5 bps one-way):")
    for y, row in st.pair_by_year(ibit, bito).iterrows():
        print(f"    {y}: gross {row['gross_pct']:+5.2f}%  net {row['net_pct']:+5.2f}%  "
              f"Sharpe {row['sharpe']:+5.2f}  HAC t {row['t']:+5.2f}  n={int(row['n'])}")
    fb = st.pair_summary(st.pair_trade(fbtc, bito, borrow_ann=0.02, cost_bps=5.0), "net")
    print(f"  FBTC cross-check: {fb['ann_pct']:+.2f}%/yr net, Sharpe {fb['sharpe']:+.2f}, "
          f"HAC t {fb['t']:+.2f}")

    # ------------------------------------------------------ synthetic control
    print("\n=== synthetic control (machinery proof — never supports the stamp) ===")
    for tag, ss in [("planted compression", 1.0), ("null (no compression)", 0.0)]:
        prices, truth = data.synthetic_panel(signal_strength=ss, seed=958)
        d = st.synthetic_detect(prices, truth)
        print(f"  {tag:22s}: spot-wrapper drag {d['spot_etf_drag_pct']:+.3f} "
              f"(planted {d['planted_fee_pct']:+.2f})  futures drag {d['fut_drag_trend_pct']:+.2f} "
              f"(planted {d['planted_drag_pre_pct']:+.2f}/{d['planted_drag_post_pct']:+.2f})")
        print(f"  {'':22s}  era change {d['era_change_pct']:+.2f} (t {d['era_t']:+.2f}), "
              f"planted {d['planted_change_pct']:+.2f}   "
              f"[naive drag {d['fut_drag_naive_pct']:+.2f} vs trend {d['fut_drag_trend_pct']:+.2f}]")
    nulls = np.array([
        st.synthetic_detect(*data.synthetic_panel(signal_strength=0.0, seed=958 + s))["era_change_pct"]
        for s in range(8)
    ])
    print(f"  null x8 seeds: era change mean {nulls.mean():+.2f} (sd {nulls.std(ddof=1):.2f}), "
          f"|change| >= 2 pp in {(np.abs(nulls) >= 2.0).sum()}/8")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
