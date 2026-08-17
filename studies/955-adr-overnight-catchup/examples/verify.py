"""Real-tape verification — Study 955 (ADR Catch-Up). Regenerates docs/results.md numbers.

Reads the shared cache (eight ADRs, three home indices, three FX pairs, a cash leg), builds
the daily return panel, and runs: the pooled stale-catchup regression, the per-name
loadings, the Fama-MacBeth residual test, the discriminating a_t-vs-x_t regression, the two
costed books plus a no-home-data control, the cost and borrow sweeps, the era and region
cuts, the block-bootstrap CI, and the synthetic control. Network only on ``--fetch``.

    python studies/955-adr-overnight-catchup/examples/verify.py            # cache-only
    python studies/955-adr-overnight-catchup/examples/verify.py --fetch    # refresh tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from adr_catchup import data, strategy as st  # noqa: E402

WINDOW = 252
COST_BPS = 5.0
BORROW_BPS = 50.0


def _line(tag, s):
    return (f"{tag:26s} {s['ann_return']:+7.2%}/yr  Sharpe {s['sharpe']:+.3f}  "
            f"vol {s['vol_ann']:5.1%}  HAC t {s['tstat']:+.2f}  maxDD {s['max_drawdown']:+.1%}")


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()
    panel = st.add_residual(data.build_panel(px), window=WINDOW)

    used = list(data.ADR_PAIRS) + list(data.HOME_INDICES) + list(data.FX)
    print(f"panel: {panel.index.min().date()} -> {panel.index.max().date()}  "
          f"names={panel['adr'].nunique()}  rows={len(panel):,}  "
          f"fp={data.fingerprint(px[used].dropna(how='all'))}")
    print(f"as-of {data.AS_OF}  |  ADR closes are TOTAL-RETURN; home indices are PRICE-ONLY")

    # ------------------------------------------------------------------ #
    print("\n=== 1. is anything stale? pooled a_t ~ x_t + x_{t-1} (HAC) ===")
    stale = st.stale_catchup_test(panel)
    print(f"  alpha {stale['alpha_bps']:+.2f} bp (t={stale['t_alpha']:+.2f})")
    print(f"  same-day home loading b0 = {stale['beta_same_day']:+.4f} (t={stale['t_same_day']:+.1f})")
    print(f"  LAGGED  home loading b1 = {stale['beta_lag']:+.4f} (t={stale['t_lag']:+.2f})   <- THE TEST")
    print(f"  n = {stale['n_obs']:,}")

    print("\n=== per-name loadings ===")
    for adr, row in st.per_name_betas(panel).iterrows():
        print(f"  {adr:5s} {row['region']:7s} n={int(row['n']):5d}  b0={row['beta_same_day']:+.3f}"
              f" (t={row['t_same_day']:+5.1f})   b1={row['beta_lag']:+.4f} (t={row['t_lag']:+5.2f})")

    # ------------------------------------------------------------------ #
    print("\n=== 1b. the index leg vs the FX leg ===")
    for tag, g in [("ALL", panel)] + [(r, panel[panel["region"] == r])
                                      for r in ("Japan", "Europe", "UK")]:
        d = st.leg_decomposition(g)
        print(f"  {tag:7s} h={d['h']:+.3f}(t={d['t_h']:+.1f})  h_lag={d['h_lag']:+.4f}"
              f"(t={d['t_h_lag']:+.2f})   f={d['f']:+.3f}(t={d['t_f']:+.1f})"
              f"  f_lag={d['f_lag']:+.4f}(t={d['t_f_lag']:+.2f})")

    print("\n=== 1c. the tradable coefficient: univariate a_{t+1} ~ x_t ===")
    for tag, g in [("ALL", panel)] + [(r, panel[panel["region"] == r])
                                      for r in ("Japan", "Europe", "UK")]:
        tr = st.tradable_gamma(g)
        rho = st.home_autocorrelation(g)
        stl = st.stale_catchup_test(g)
        print(f"  {tag:7s} beta_lag={stl['beta_lag']:+.4f}(t={stl['t_lag']:+.2f})  ->  "
              f"gamma={tr['gamma']:+.4f}(t={tr['t_gamma']:+.2f})   "
              f"[rho1(x)={rho['rho1']:+.4f} explains the gap]")

    print("\n=== 2. does the un-caught-up residual predict? Fama-MacBeth a_{t+1} ~ e_t ===")
    fm = st.fama_macbeth(panel, xcols=("e",), standardize="rank")
    print(f"  rank-standardised slope {fm['slope_e']*1e4:+.2f} bp  HAC t = {fm['t_e']:+.2f}"
          f"  over {fm['n_dates']:,} dates")
    fm_raw = st.fama_macbeth(panel, xcols=("e",))
    print(f"  (raw slope {fm_raw['slope_e']:+.4f}, t={fm_raw['t_e']:+.2f} — unstable on "
          f"quiet cross-sections, kept only for the record)")
    pool_e = st.pooled_predictive(panel, xcols=("e",))
    print(f"  (pooled, over-optimistic by cross-correlation: {pool_e['coef_e']:+.4f}, "
          f"t={pool_e['t_e']:+.2f})")

    # ------------------------------------------------------------------ #
    print("\n=== 3. is that just plain one-day reversal? pooled a_{t+1} ~ a_t + x_t (HAC) ===")
    disc = st.pooled_predictive(panel, xcols=("a", "x"))
    print(f"  own-move a_t   {disc['coef_a']:+.4f} (t={disc['t_a']:+.2f})   <- plain reversal")
    print(f"  home    x_t    {disc['coef_x']:+.4f} (t={disc['t_x']:+.2f})   <- catch-up, once controlled")
    solo = st.pooled_predictive(panel, xcols=("x",))
    print(f"  home x_t alone {solo['coef_x']:+.4f} (t={solo['t_x']:+.2f})")
    fm_x = st.fama_macbeth(panel, xcols=("x",))
    print(f"  Fama-MacBeth on x_t: slope {fm_x['slope_x']:+.4f} (t={fm_x['t_x']:+.2f})")

    # ------------------------------------------------------------------ #
    print("\n=== 4. the costed books (gross exposure 1, ONE execution lag) ===")
    for name in ("catchup", "residual", "reversal"):
        bt = st.book(panel, name, cost_bps=COST_BPS, borrow_bps=BORROW_BPS)
        g = st.book(panel, name, cost_bps=0.0, borrow_bps=0.0)
        print(f"  [{name}]  turnover {bt['turnover'].mean():.3f}/day  "
              f"net exposure {bt['net_exposure'].mean():+.3f}  short leg {bt['short_exposure'].mean():.3f}")
        print("   " + _line("gross:", st.summary(g["gross"])))
        print("   " + _line(f"net ({COST_BPS:.0f}bp/{BORROW_BPS:.0f}bp):", st.summary(bt["net"])))
        print(f"    breakeven one-way cost: {st.breakeven_cost_bps(panel, name):.2f} bps")

    dn = st.book(panel, "catchup", cost_bps=0.0, borrow_bps=0.0, dollar_neutral=True)
    print("   " + _line("catchup, $-neutral gross:", st.summary(dn["gross"])))
    lin = st.book(panel, "catchup", weighting="linear", cost_bps=0.0, borrow_bps=0.0)
    print("   " + _line("catchup, signal-weighted:", st.summary(lin["gross"])))

    # ------------------------------------------------------------------ #
    print("\n=== bootstrap Sharpe CIs on the gross books (2000 draws, 21-day blocks) ===")
    for name in ("catchup", "residual", "reversal"):
        g = st.book(panel, name, cost_bps=0.0, borrow_bps=0.0)["gross"]
        ci = st.block_bootstrap_ci(g, seed=955)
        print(f"  {name:9s}: Sharpe {ci['point']:+.3f}  95% CI [{ci['ci_low']:+.3f}, "
              f"{ci['ci_high']:+.3f}]  share<0 {ci['frac_negative']:.1%}")

    print("\n=== cost sweep, catch-up book (borrow 50 bps) ===")
    for c, row in st.cost_sweep(panel, name="catchup", borrow_bps=BORROW_BPS).iterrows():
        print(f"  {c:5.1f} bps: {row['ann_return']:+7.2%}/yr  Sharpe {row['sharpe']:+.3f}"
              f"  t {row['tstat']:+.2f}")

    print("\n=== cost sweep, residual book (borrow 50 bps) ===")
    for c, row in st.cost_sweep(panel, name="residual", borrow_bps=BORROW_BPS).iterrows():
        print(f"  {c:5.1f} bps: {row['ann_return']:+7.2%}/yr  Sharpe {row['sharpe']:+.3f}"
              f"  t {row['tstat']:+.2f}")

    print("\n=== borrow sweep (PROXY/ASSUMPTION; cost 5 bps) ===")
    for name in ("catchup", "residual"):
        for b, row in st.borrow_sweep(panel, name=name, cost_bps=COST_BPS).iterrows():
            print(f"  [{name:8s}] {b:5.0f} bps: {row['ann_return']:+7.2%}/yr  "
                  f"Sharpe {row['sharpe']:+.3f}  t {row['tstat']:+.2f}")

    # ------------------------------------------------------------------ #
    print("\n=== era cut (split 2015-01-01) ===")
    for tag, e in st.era_cut(panel, split="2015-01-01").items():
        if e is None:
            continue
        print(f"  {tag:5s} n={e['n_obs']:6,}: b1(lag)={e['beta_lag']:+.4f} (t={e['t_lag']:+.2f})"
              f"  gamma={e['gamma']:+.4f} (t={e['t_gamma']:+.2f})"
              f"  FM(e)={e['fm_slope_e']*1e4:+.2f}bp (t={e['fm_t_e']:+.2f})"
              f"  x|a={e['coef_x_ctrl']:+.4f} (t={e['t_x_ctrl']:+.2f})")
        print(f"          catch-up gross Sharpe {e['gross_sharpe']:+.3f} (t={e['gross_t']:+.2f})"
              f"  |  residual gross Sharpe {e['resid_sharpe']:+.3f} (t={e['resid_t']:+.2f})")

    print("\n=== region cut (Tokyo closes 02:00 ET; London/Frankfurt 11:30 ET) ===")
    for region, row in st.region_cut(panel).iterrows():
        print(f"  {region:7s} n={int(row['n_obs']):6,}: b0={row['beta_same_day']:+.3f}"
              f"  b1(lag)={row['beta_lag']:+.4f} (t={row['t_lag']:+.2f})"
              f"  gamma={row['gamma']:+.4f} (t={row['t_gamma']:+.2f})"
              f"  FM(e)={row['fm_slope_e']*1e4:+.2f}bp (t={row['fm_t_e']:+.2f})"
              f"  x|a={row['coef_x_ctrl']:+.4f} (t={row['t_x_ctrl']:+.2f})")

    print("\n=== Japan, block by block (is the one positive region era-robust?) ===")
    jp = panel[panel["region"] == "Japan"]
    for _, row in st.sub_era_table(jp).iterrows():
        print(f"  {row['from']}-{row['to']} n={int(row['n_obs']):5d}: b0={row['beta_same_day']:+.3f}"
              f"  b1={row['beta_lag']:+.4f} (t={row['t_lag']:+.2f})"
              f"  gamma={row['gamma']:+.4f} (t={row['t_gamma']:+.2f})")
    for tag, m in [("2004-2014", jp.index < pd.Timestamp("2015-01-01")),
                   ("2015-2026", jp.index >= pd.Timestamp("2015-01-01"))]:
        s = st.stale_catchup_test(jp[m])
        g = st.tradable_gamma(jp[m])
        print(f"  {tag} half : b1={s['beta_lag']:+.4f} (t={s['t_lag']:+.2f})"
              f"  gamma={g['gamma']:+.4f} (t={g['t_gamma']:+.2f})")
    print("  Japan-only books:")
    for name in ("catchup", "residual", "reversal"):
        sg = st.summary(st.book(jp, name, cost_bps=0.0, borrow_bps=0.0)["gross"])
        sn = st.summary(st.book(jp, name, cost_bps=COST_BPS, borrow_bps=BORROW_BPS)["net"])
        print(f"    [{name:8s}] gross {sg['ann_return']:+6.2%}/yr Sharpe {sg['sharpe']:+.3f}"
              f" t {sg['tstat']:+.2f} | net {sn['ann_return']:+7.2%}/yr Sharpe {sn['sharpe']:+.3f}"
              f" | breakeven {st.breakeven_cost_bps(jp, name):+.2f} bps")

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    print("\n=== 5. how robust is the ONE positive number? (Japan b1) ===")
    for tag, g in [("ALL", panel), ("Japan", jp)]:
        print(f"  [{tag}] trim / winsorize |x_lag| at the quantile:")
        for _, row in st.tail_sensitivity(g).iterrows():
            label = "full sample" if row["mode"] == "full" else f"{row['mode']:6s} @ {row['q']:.3f}"
            print(f"     {label:18s} n={int(row['n']):6,}  b1={row['beta_lag']:+.4f} "
                  f"(t={row['t_lag']:+.2f})")
    print("  Japan, bucketed by yesterday's home move (the bettable, leverage-free sort):")
    for b, row in st.lag_bucket_table(jp).iterrows():
        print(f"     Q{b} x_lag {row['x_lag_mean']:+.4f} -> ADR {row['a_mean_bps']:+6.2f} bp"
              f"  (naive t {row['naive_t']:+.2f}, n={int(row['n']):,})")
    tbl = st.lag_bucket_table(jp)
    print(f"     Q5-Q1 spread {tbl['a_mean_bps'].iloc[-1] - tbl['a_mean_bps'].iloc[0]:+.2f} bp")
    co = st.consecutive_only(jp)
    print(f"  Japan, rows whose lag really is yesterday ({co['share_kept']:.1%} of rows): "
          f"b1={co['beta_lag']:+.4f} (t={co['t_lag']:+.2f})")
    ci = st.beta_lag_bootstrap(jp, seed=955)
    print(f"  Japan block-bootstrap of b1 (2,000 draws, 21-day date blocks): {ci['point']:+.4f} "
          f"[{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]  share<=0 {ci['frac_le_zero']:.1%}"
          f"  (pooled point {ci['point_pooled']:+.4f})")
    print("  the same knife on the SYNTHETIC planted (linear) lag, for calibration:")
    syn, _ = data.synthetic_panel(signal_strength=1.0, seed=955)
    for _, row in st.tail_sensitivity(syn).iterrows():
        if row["mode"] in ("full", "trim") and row["q"] in (1.0, 0.99, 0.95):
            label = "full sample" if row["mode"] == "full" else f"trim   @ {row['q']:.3f}"
            print(f"     {label:18s} b1={row['beta_lag']:+.4f} (t={row['t_lag']:+.1f})")

    print("\n=== NVO cross-check: the real Danish home pair vs the DAX/EUR PROXY ===")
    try:
        cx = data.load_prices(tickers=data.CROSS_CHECK_TICKERS,
                              start=data.CROSS_CHECK_START)
        for tag, pairs in [("OMXC25 + DKK (real)", data.NVO_CROSS_CHECK),
                           ("GDAXI + EUR (proxy)", {"NVO": data.ADR_PAIRS["NVO"]})]:
            p = st.add_residual(data.build_panel(cx, pairs=pairs))
            s, tg = st.stale_catchup_test(p), st.tradable_gamma(p)
            print(f"  {tag:20s} n={s['n_obs']:5d}  b0={s['beta_same_day']:+.3f}"
                  f"  b1={s['beta_lag']:+.4f} (t={s['t_lag']:+.2f})"
                  f"  gamma={tg['gamma']:+.4f} (t={tg['t_gamma']:+.2f})")
    except FileNotFoundError as exc:
        print(f"  skipped (cross-check tapes not cached): {exc}")

    print("\n=== benchmark context: equal-weight ADR basket, excess-of-cash (^IRX proxy) ===")
    ret = panel.pivot_table(index=panel.index, columns="adr", values="a")
    ew = ret.mean(axis=1)
    cash = data.cash_returns(px).reindex(ew.index).ffill().fillna(0.0)
    print("  " + _line("EW basket, excess-of-cash:", st.summary(ew - cash)))
    # Explicit division, not pct_change: BIL only starts in 2007 and pandas 2.x would pad
    # the gap and print spurious zeros (see data.build_panel).
    bil_px = px["BIL"].astype(float)
    bil = (bil_px / bil_px.shift(1) - 1.0).reindex(ew.index)
    ok = bil.notna()
    print("  " + _line("... same, BIL cash cross-check:", st.summary((ew - bil)[ok])))

    # ------------------------------------------------------------------ #
    print("\n=== synthetic control (machinery proof; never supports the stamp) ===")
    pl, _ = data.synthetic_panel(signal_strength=1.0, seed=955)
    d_pl = st.synthetic_detect(pl)
    print(f"  planted lag : b1={d_pl['beta_lag']:+.4f} (t={d_pl['t_lag']:+.1f})"
          f"  x|a={d_pl['coef_x_ctrl']:+.4f} (t={d_pl['t_x_ctrl']:+.1f})"
          f"  catch-up gross Sharpe {d_pl['gross_sharpe']:+.2f} (t={d_pl['gross_t']:+.1f})")
    b1s, txs, shs = [], [], []
    for s in range(8):
        d = st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=955 + s)[0])
        b1s.append(d["beta_lag"]); txs.append(d["t_x_ctrl"]); shs.append(d["gross_t"])
    b1s, txs, shs = map(np.asarray, (b1s, txs, shs))
    print(f"  null x8     : b1 mean {b1s.mean():+.4f} (sd {b1s.std(ddof=1):.4f}); "
          f"|t(x|a)|>=2 on {(np.abs(txs) >= 2).sum()}/8; |t(book)|>=2 on {(np.abs(shs) >= 2).sum()}/8")
    bo, _ = data.synthetic_panel(signal_strength=0.0, bounce_vol=0.006, seed=955)
    d_bo = st.synthetic_detect(bo)
    print(f"  pure bounce, ZERO catch-up: FM slope(e)={d_bo['fm_slope_e']:+.4f} "
          f"(t={d_bo['fm_t_e']:+.1f}) fires;  b1={d_bo['beta_lag']:+.4f} (t={d_bo['t_lag']:+.2f}) "
          f"stays silent;  x|a={d_bo['coef_x_ctrl']:+.4f} (t={d_bo['t_x_ctrl']:+.1f}) is spurious")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
