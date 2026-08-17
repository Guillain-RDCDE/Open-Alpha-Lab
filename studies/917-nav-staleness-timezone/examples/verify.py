"""Real-tape verification — Study 917 (Stale NAV). Regenerates docs/results.md numbers.

Reads cached daily total-return closes for SPY, five single-country ETFs and the cash legs
(^IRX for the long sample, BIL as the tradable cross-check), then runs: the per-fund lagged
HAC regressions (raw and net of SPY's own next-day move), the domestic SPY-on-SPY control,
the top-decile next-day rule on an equal-weight basket, the era cut at 2010, the cost,
borrow and threshold sweeps, the conservative extra-lag variant, the block bootstrap, and
the synthetic control. Network only on ``--fetch``.

    python studies/917-nav-staleness-timezone/examples/verify.py            # cache-only
    python studies/917-nav-staleness-timezone/examples/verify.py --fetch    # refresh
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from stale_nav import data, strategy as st  # noqa: E402

COST_BPS = 10.0
TOP_Q = 0.90
SPLIT = "2010-01-01"


def synthetic_control() -> None:
    """The offline machinery proof — runs with no cache and no network."""
    print("\n=== synthetic control (machinery proof — never supports the stamp) ===")
    p1, t1 = data.synthetic_panel(signal_strength=1.0, seed=917)
    d1 = st.synthetic_detect(p1, t1)
    print(f"  planted catch-up beta {d1['planted_beta']:+.3f}: recovered "
          f"{d1['beta_mean']:+.3f} (mean HAC t {d1['t_beta_mean']:+.2f}), trigger-day mean "
          f"{d1['mean_on_bps']:+.2f} bps (t {d1['t_on_mean']:+.2f})")
    nulls = [st.synthetic_detect(*data.synthetic_panel(signal_strength=0.0, seed=917 + s))
             for s in range(8)]
    nb = np.array([d["beta_mean"] for d in nulls])
    nt = np.array([d["t_beta_max_abs"] for d in nulls])
    print(f"  null x8: beta mean {nb.mean():+.4f} (sd {nb.std(ddof=1):.4f}), "
          f"max |t| >= 2 on {(nt >= 2).sum()}/8 seeds")


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    if not data.have_real():
        print("No shared studies/_cache present — running the offline synthetic demo only.")
        print("Populate the cache with:  python examples/verify.py --fetch")
        synthetic_control()
        return
    px = data.load_prices()
    funds = list(data.FUNDS)
    rets = px.pct_change()
    spy = rets["SPY"]
    cash = data.cash_returns_from_irx(px["IRX"]).reindex(px.index).ffill()

    basket = st.equal_weight_basket(rets, funds)
    span = px[["SPY"] + funds].dropna(how="all")
    print(f"tape: {span.index[0].date()} -> {span.index[-1].date()}  n={len(span)}  "
          f"fp={data.fingerprint(px[['SPY'] + funds])}")
    print(f"as-of {data.AS_OF}  |  cash = ^IRX 13-week bill PROXY, lagged one day")

    # ------------------------------------------------------------------ #
    print("\n=== per-fund lagged regressions: r_fund[t+1] ~ a + b r_SPY[t] ===")
    tbl = st.per_fund_regressions(rets, funds)
    for f, row in tbl.iterrows():
        print(f"  {f:4s} n={int(row['n']):5d}  raw b={row['beta_raw']:+.4f} "
              f"(t={row['t_raw']:+.2f})   net-of-SPY b={row['beta_rel']:+.4f} "
              f"(t={row['t_rel']:+.2f})")
    dom = st.domestic_control(spy)
    print(f"  DOMESTIC CONTROL  SPY[t+1] ~ SPY[t]: b={dom['beta']:+.4f} "
          f"(t={dom['t_beta']:+.2f}, n={dom['n']}) <- the market's own reversal")
    print(f"  Bonferroni across {len(funds)} funds: |t| >= 2.58 needed for 5% family-wise")

    # ------------------------------------------------------------------ #
    # The net-of-SPY column above subtracts ONE unit of SPY. That is an ASSUMPTION
    # (unit beta) and it is load-bearing, so drop it: refit each fund's contemporaneous
    # beta (IN-SAMPLE — a diagnostic, never a tradable rule) and hedge with that.
    print("\n=== dropping the unit-beta ASSUMPTION: fitted-beta hedge (in-sample) ===")
    print("     b_raw = beta_c x b_domestic (confound)  +  b_hedged (what a timezone story gets)")
    for f in funds:
        d = st.confound_decomposition(rets[f], spy)
        print(f"  {f:4s} beta_c={d['beta_contemp']:+.3f}  raw {d['beta_raw']:+.4f} = "
              f"confound {d['confound']:+.4f} ({d['share']:.0%}) + hedged "
              f"{d['beta_hedged']:+.4f} (t={d['t_hedged']:+.2f})")
    print("  -> under a FITTED hedge two funds clear |t|=2.58 (EWJ, FXI); both are NEGATIVE,")
    print("     i.e. still the opposite of catch-up. No fund is positive in either hedge.")

    # ------------------------------------------------------------------ #
    print("\n=== era cut per fund (split 2010-01-01), net-of-SPY beta ===")
    for f in funds:
        eras = st.era_cut(rets[f], spy, cash, split=SPLIT, cost_bps=COST_BPS, top_q=TOP_Q)
        e, l = eras["early"], eras["late"]
        if e is None or l is None:
            continue
        print(f"  {f:4s} early n={e['n']:5d} rel b={e['beta_rel']:+.4f} (t={e['t_rel']:+.2f})"
              f" [hedged {e['beta_hedged']:+.4f} t={e['t_hedged']:+.2f}]"
              f"   late n={l['n']:5d} rel b={l['beta_rel']:+.4f} (t={l['t_rel']:+.2f})"
              f" [hedged {l['beta_hedged']:+.4f} t={l['t_hedged']:+.2f}]")

    # ------------------------------------------------------------------ #
    print("\n=== the tradable rule on the equal-weight basket (10 bps one-way) ===")
    cmp = st.compare(basket, spy, cash, top_q=TOP_Q, cost_bps=COST_BPS)
    c = cmp["cond"]
    print(f"  trigger days {c['n_on']} ({cmp['trigger_frac']:.1%})  |  turnover "
          f"{cmp['turnover']:.0f} position changes")
    print(f"  basket excess return on trigger days: {c['mean_on_bps']:+.2f} bps/day "
          f"(HAC t={c['t_on']:+.2f})  [GROSS of cost — the claim's best case]")
    print(f"  on all other days                   : {c['mean_off_bps']:+.2f} bps/day "
          f"(difference {c['diff_bps']:+.2f} bps, Welch t={c['t_diff_welch']:+.2f})")
    pos = cmp["position"]
    rel_on = (basket - spy).reindex(pos.index)[pos != 0].dropna()
    print(f"  NET OF SPY's same-day move          : {rel_on.mean() * 1e4:+.2f} bps/day "
          f"(HAC t={st.newey_west_t(rel_on.to_numpy()):+.2f})  <- unit-beta ASSUMPTION")
    # Same statistic with the unit-beta assumption dropped (beta fitted in-sample).
    m = ~(basket.isna() | spy.isna())
    b_c = st.ols_hac(basket[m], spy[m])["beta"]
    hed_on = (basket - b_c * spy).reindex(pos.index)[pos != 0].dropna()
    print(f"  same, hedged at the FITTED beta {b_c:.3f}  : {hed_on.mean() * 1e4:+.2f} bps/day "
          f"(HAC t={st.newey_west_t(hed_on.to_numpy()):+.2f})  <- assumption-free version")
    print("  (both HAC t's are computed on the non-contiguous trigger-day subsample: the")
    print("   Bartlett lags pair draws that are adjacent in the SUBSAMPLE, not in the calendar)")
    for tag, key in [("long rule (the claim)", "long"), ("short mirror", "short"),
                     ("buy & hold basket (untraded, pays no cost)", "bh")]:
        s = cmp[key]
        print(f"  {tag:42s}: exSharpe {s['sharpe']:+.3f}  CAGR {s['cagr']:+.2%}  "
              f"vol {s['vol_ann']:.2%}  HAC t {s['tstat']:+.2f}")
    print("  NB costs are ONE-SIDED by construction: the rule pays 10 bps on every one of")
    print("     its position changes, buy-and-hold pays nothing. That asymmetry is ~2 bps")
    print("     total over 30 years for B&H and runs AGAINST the claim, so it is left in.")

    # ------------------------------------------------------------------ #
    print("\n=== block bootstrap (2000 draws, 21-day blocks) ===")
    on_excess = cmp["e_bh"].reindex(cmp["position"].index)[cmp["position"] != 0].dropna()
    b = st.bootstrap_mean_ci(on_excess, seed=917)
    print(f"  trigger-day mean {b['mean_bps']:+.2f} bps  95% CI "
          f"[{b['ci_low_bps']:+.2f}, {b['ci_high_bps']:+.2f}]  share<0 {b['frac_negative']:.1%}")
    bs = st.bootstrap_sharpe_ci(cmp["e_long"], seed=917)
    print(f"  long-rule exSharpe {bs['sharpe']:+.3f}  95% CI [{bs['ci_low']:+.3f}, "
          f"{bs['ci_high']:+.3f}]  share<0 {bs['frac_negative']:.1%}")

    # ------------------------------------------------------------------ #
    print("\n=== era cut on the basket rule ===")
    eras = st.era_cut(basket, spy, cash, split=SPLIT, cost_bps=COST_BPS, top_q=TOP_Q)
    for tag, e in eras.items():
        if e is None:
            continue
        print(f"  {tag:5s} n={e['n']:5d}  trigger-day mean {e['mean_on_bps']:+.2f} bps "
              f"(t={e['t_on']:+.2f}, n_on={e['n_on']})  rel b={e['beta_rel']:+.4f} "
              f"(t={e['t_rel']:+.2f})  long exSharpe {e['sharpe_long']:+.3f}  "
              f"short {e['sharpe_short']:+.3f}  B&H {e['sharpe_bh']:+.3f}")

    # ------------------------------------------------------------------ #
    print("\n=== cost sweep (basket) ===")
    for row in st.cost_sweep(basket, spy, cash, top_q=TOP_Q):
        print(f"  {row['cost_bps']:5.1f} bps one-way: long exSharpe {row['sharpe_long']:+.3f} "
              f"(t={row['t_long']:+.2f})   short mirror {row['sharpe_short']:+.3f} "
              f"(t={row['t_short']:+.2f})")

    print("\n=== borrow sweep for the short mirror (10 bps cost) ===")
    for row in st.borrow_sweep(basket, spy, cash, cost_bps=COST_BPS, top_q=TOP_Q):
        print(f"  borrow {row['borrow_ann'] * 100:4.1f}%/yr: exSharpe "
              f"{row['sharpe_short']:+.3f} (t={row['t_short']:+.2f})")

    print("\n=== threshold sweep (what counts as a strong US day) ===")
    for row in st.threshold_sweep(basket, spy, cash, cost_bps=COST_BPS):
        print(f"  top {(1 - row['top_q']) * 100:4.0f}%: n_on={row['n_on']:4d} "
              f"mean {row['mean_on_bps']:+.2f} bps (t={row['t_on']:+.2f})  "
              f"long exSharpe {row['sharpe_long']:+.3f}")

    print("\n=== conservative variant: wait one extra day before trading ===")
    x = st.extra_lag_check(basket, spy, cash, cost_bps=COST_BPS, top_q=TOP_Q)
    print(f"  mean {x['mean_on_bps']:+.2f} bps (t={x['t_on']:+.2f}, n={x['n_on']})  "
          f"long exSharpe {x['sharpe_long']:+.3f}")

    # ------------------------------------------------------------------ #
    print("\n=== BIL cross-check (tradable cash leg, 2007+) ===")
    bil = px["BIL"].dropna()
    cash_bil = bil.pct_change()
    idx = basket.index.intersection(cash_bil.index)
    cmp_bil = st.compare(basket.loc[idx], spy.loc[idx], cash_bil.loc[idx],
                         top_q=TOP_Q, cost_bps=COST_BPS)
    cb = cmp_bil["cond"]
    print(f"  {idx[0].date()} -> {idx[-1].date()}  trigger-day mean {cb['mean_on_bps']:+.2f} "
          f"bps (t={cb['t_on']:+.2f})  long exSharpe {cmp_bil['long']['sharpe']:+.3f}  "
          f"B&H {cmp_bil['bh']['sharpe']:+.3f}")

    # ------------------------------------------------------------------ #
    synthetic_control()


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
