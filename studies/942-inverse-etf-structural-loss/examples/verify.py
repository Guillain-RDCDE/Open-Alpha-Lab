"""Real-tape verification — Study 942 (The Inverse Tax). Regenerates docs/results.md.

Reads the shared cache (SH, PSQ, SDS, SPY, QQQ, BIL, ^IRX total-return closes plus
price-only SPY/QQQ closes), builds the honest directly-short replicate, and prints the
daily gap with its HAC *t*, the accounting decomposition, the break-even rebate credit,
the bootstrap CI, the era and rate-regime cuts, the proxy sweeps, the path-drag table and
the two cross-checks. Network only on ``--fetch``.

    python studies/942-inverse-etf-structural-loss/examples/verify.py            # cache-only
    python studies/942-inverse-etf-structural-loss/examples/verify.py --fetch    # refresh
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from inverse_tax import data, strategy as st  # noqa: E402

CREDIT = 1.0        # PROXY: interest on your own NAV cash, no rebate on the proceeds
BORROW_BPS = 30.0   # PROXY: general-collateral stock-loan fee on SPY / QQQ
COST_BPS = 1.0      # one-way rebalance cost x NAV on the direct book


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()

    print(f"as-of {data.AS_OF}   shared cache: {data.DEFAULT_CACHE}")
    print(f"tape {px.index[0].date()} -> {px.index[-1].date()}  "
          f"fp={data.fingerprint(px.dropna())}")
    print(f"base proxies: credit={CREDIT} x bill rate, borrow={BORROW_BPS} bps, "
          f"rebalance cost={COST_BPS} bps one-way x NAV")

    cc = st.financing_crosscheck(px)
    print(f"\nfinancing cross-check: ^IRX accrual {cc['irx_ann_pct']:.2f}%/yr vs "
          f"BIL total return {cc['bil_ann_pct']:.2f}%/yr "
          f"(diff {cc['diff_ann_pct']:+.2f} pp, n={cc['n_days']})")

    print("\n=== headline: inverse fund minus honest direct short (positive = fund wins) ===")
    races = {}
    for fund, (index, k) in data.FUNDS.items():
        r = st.race(px, fund=fund, index=index, k=k, credit=CREDIT,
                    borrow_bps=BORROW_BPS, cost_bps=COST_BPS)
        races[fund] = r
        print(f"  {fund} ({k:+.0f}x {index}) n={r['n_days']} "
              f"{r['start'].date()}->{r['end'].date()}  "
              f"gap {r['gap_ann_pct']:+.2f}%/yr  HAC t {r['gap_t']:+.2f}  "
              f"daily sd {r['gap_sd_bps']:.1f} bps")

    print("\n=== SAME-MANDATE counterweight: the fund charged the index's dividends too ===")
    print("  (both arms then short the TOTAL-return stream - the strictly like-for-like race)")
    for fund, (index, k) in data.FUNDS.items():
        r = races[fund]
        print(f"  {fund}: gap {r['gap_same_mandate_ann_pct']:+.2f}%/yr  "
              f"HAC t {r['gap_same_mandate_t']:+.2f}   "
              f"(headline {r['gap_ann_pct']:+.2f}, t {r['gap_t']:+.2f})")

    print("\n=== HAC lag sensitivity of the headline SH gap (lags=0 is the i.i.d. t) ===")
    prof = st.hac_lag_profile(races["SH"]["gap"])
    print("  " + "  ".join(
        f"{'auto' if row['lags'] is None else row['lags']}:{row['t']:+.2f}" for row in prof))

    r = races["SH"]
    print("\n=== excess-of-cash (minus BIL) Sharpe race, SH vs the direct short ===")
    abs_fund = st.summary(r["legs"]["r_fund"])
    abs_dir = st.summary(r["r_direct"])
    for tag, se, sa in [("SH (inverse ETF)", r["s_fund"], abs_fund),
                        ("direct short", r["s_direct"], abs_dir)]:
        print(f"  {tag:18s}: exSharpe {se['sharpe']:+.3f}  "
              f"absolute CAGR {sa['cagr']*100:+.2f}%/yr  vol {sa['vol_ann']*100:.1f}%  "
              f"MaxDD {sa['max_drawdown']*100:.1f}%")
    print(f"  Sharpe difference (fund - direct): {r['sharpe_diff']:+.3f}  "
          f"(both arms are short books in a market that rose 5x - the GAP is the quantity)")

    print("\n=== accounting decomposition of the RAW gap (credit=0, borrow=0, cost=0) ===")
    for fund, (index, k) in data.FUNDS.items():
        d = st.decompose(px, fund=fund, index=index, k=k,
                         expense_ratio_pct=data.EXPENSE_RATIO_PCT[fund])
        print(f"  {fund}: raw {d['raw_gap_ann_pct']:+.2f}%/yr = "
              f"dividends not owed {d['dividends_not_owed_pct']:+.2f} "
              f"+ financing {d['financing_passthrough_pct']:+.2f} "
              f"- expense ratio {d['expense_ratio_pct']:.2f} "
              f"+ residual {d['residual_pct']:+.2f}")
        print(f"        regression: beta {d['beta']:.4f} (target 1.0000)  "
              f"gamma {d['gamma']:.3f} rate units  alpha {d['alpha_ann_pct']:+.2f}%/yr  "
              f"(index yield {d['div_yield_ann_pct']:.2f}%/yr, bills {d['rf_ann_pct']:.2f}%/yr)")
        print(f"        spec check: gamma on the PRICE leg {d['gamma_px']:.3f} "
              f"-> financing term {d['financing_passthrough_px_pct']:+.2f}%/yr "
              f"(vs {d['financing_passthrough_pct']:+.2f}); the raw gap is spec-free, "
              f"only the financing/residual SPLIT moves")

    print("\n=== break-even rebate credit (gap = 0) ===")
    for fund, (index, k) in data.FUNDS.items():
        bc = st.breakeven_credit(px, fund=fund, index=index, k=k,
                                 borrow_bps=BORROW_BPS, cost_bps=COST_BPS)
        print(f"  {fund}: c* = {bc:.2f} units of the bill rate  "
              f"(below it the fund wins, above it the direct short wins)")

    print("\n=== bootstrap CI on the annualised gap (2,000 draws, 21-day blocks) ===")
    for fund in data.FUNDS:
        ci = st.bootstrap_gap_ci(races[fund]["gap"], seed=942)
        print(f"  {fund}: {ci['point']:+.2f}%/yr  95% CI "
              f"[{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}]  share<0 {ci['frac_negative']:.3f}")

    print("\n=== era cut (split 2016-01-01) ===")
    for tag, e in st.era_cut(px, fund="SH", index="SPY", k=-1.0, credit=CREDIT,
                             borrow_bps=BORROW_BPS, cost_bps=COST_BPS).items():
        if e:
            print(f"  {tag:6s} n={e['n_days']:5d}  gap {e['gap_ann_pct']:+.2f}%/yr  "
                  f"t {e['gap_t']:+.2f}  (bills {e['rf_ann_pct']:.2f}%/yr)")

    print("\n=== rate-regime cut (prevailing 13-week bill level, known ex ante) ===")
    rr = st.rate_regime_cut(px, fund="SH", index="SPY", k=-1.0, credit=CREDIT,
                            borrow_bps=BORROW_BPS, cost_bps=COST_BPS)
    for tag in ("zirp", "normal"):
        e = rr[tag]
        if e:
            print(f"  {tag:6s} (bills {'<' if tag == 'zirp' else '>='}{rr['threshold_pct']:.0f}%) "
                  f"n={e['n_days']:5d}  gap {e['gap_ann_pct']:+.2f}%/yr  t {e['gap_t']:+.2f}  "
                  f"(bills {e['rf_ann_pct']:.2f}%/yr)")

    print("\n=== proxy sweep: rebate credit x stock-loan fee (SH, %/yr gap) ===")
    for row in st.credit_borrow_sweep(px, fund="SH", index="SPY", k=-1.0, cost_bps=COST_BPS):
        print(f"  credit={row['credit']:.1f}  borrow={row['borrow_bps']:5.0f} bps: "
              f"gap {row['gap_ann_pct']:+.2f}%/yr  t {row['gap_t']:+.2f}")

    print("\n=== rebalance-cost sweep on the direct book (SH) ===")
    for row in st.cost_sweep(px, fund="SH", index="SPY", k=-1.0,
                             credit=CREDIT, borrow_bps=BORROW_BPS):
        print(f"  cost={row['cost_bps']:5.1f} bps: gap {row['gap_ann_pct']:+.2f}%/yr  "
              f"t {row['gap_t']:+.2f}")

    print("\n=== path drag across holding horizons (non-overlapping windows) ===")
    for fund, (index, k) in data.FUNDS.items():
        for row in st.path_drag(px, fund=fund, index=index, k=k, credit=CREDIT,
                                borrow_bps=BORROW_BPS, cost_bps=COST_BPS):
            print(f"  {fund} H={row['horizon_days']:4d}d n={row['n_windows']:4d}: "
                  f"daily-reset - static {row['reset_minus_static_mean_pp']:+.2f} pp "
                  f"(med {row['reset_minus_static_median_pp']:+.2f})  |  "
                  f"fund - daily-reset {row['fund_minus_reset_mean_pp']:+.2f} pp "
                  f"(med {row['fund_minus_reset_median_pp']:+.2f})")

    print("\n=== synthetic control (machinery proof only, never supports the stamp) ===")
    for ss in (1.0, 0.0):
        prices, truth = data.synthetic_daily(signal_strength=ss, seed=942)
        d = st.synthetic_detect(prices)
        print(f"  signal_strength={ss:.0f}: planted gap {truth['expected_gap_ann']*100:+.2f}%/yr "
              f"-> measured {d['gap_ann_pct']:+.2f}%/yr (t {d['gap_t']:+.2f})")
    import numpy as np
    seeds = tuple(942 + s for s in range(12))
    null = [st.synthetic_detect(p) for p, _ in
            data.synthetic_panel(seeds=seeds, signal_strength=0.0)]
    plant = [st.synthetic_detect(p) for p, _ in
             data.synthetic_panel(seeds=seeds, signal_strength=1.0)]
    ng = np.array([d["gap_ann_pct"] for d in null])
    nt = np.array([d["gap_t"] for d in null])
    pg = np.array([d["gap_ann_pct"] for d in plant])
    print(f"  null panel x{len(seeds)}   : gap mean {ng.mean():+.3f}%/yr sd {ng.std(ddof=1):.3f}, "
          f"|t|>=2 on {(np.abs(nt) >= 2).sum()}/{len(seeds)} seeds")
    print(f"  planted panel x{len(seeds)}: gap mean {pg.mean():+.3f}%/yr sd {pg.std(ddof=1):.3f} "
          f"(planted -3.00), recovered with the right sign on "
          f"{(pg < 0).sum()}/{len(seeds)} seeds")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
