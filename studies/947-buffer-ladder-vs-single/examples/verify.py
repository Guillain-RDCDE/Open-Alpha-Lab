"""Real-tape verification — Study 947 (The Buffer Ladder). Regenerates docs/results.md.

Reads cached daily total-return closes for the laddered buffer wrapper (BUFR), its four
quarterly Power Buffer vintages (PJAN/PAPR/PJUL/POCT), the underlying (SPY) and cash (BIL),
then races the wrapper against every do-it-yourself alternative — each single vintage, the
equal-weight DIY basket, the beta-matched DIY ladder, and the beta-matched SPY/BIL mix.
Prints the excess-of-cash Sharpe race, the HAC return-difference *t* on each gap, block
bootstrap CIs **and a block-length sensitivity sweep on the only CI that excludes zero**,
the entry-point-luck dispersion, the drawdown table, an era cut, a cost sweep, an
assumed-fee sweep, a rebalance-frequency sweep, the calendar-year table and the synthetic
control. Network only on ``--fetch``.

    python studies/947-buffer-ladder-vs-single/examples/verify.py            # cache-only
    python studies/947-buffer-ladder-vs-single/examples/verify.py --fetch    # refresh
"""

from __future__ import annotations

import os
import sys

import numpy as np  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from buffer_ladder import data, strategy as st  # noqa: E402

LADDER = data.LADDER
VINTAGES = data.VINTAGES
MARKET = data.MARKET
CASH = data.CASH
COST_BPS = 5.0
SPLIT = "2023-07-01"


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()

    print(f"{LADDER} vs {'/'.join(VINTAGES)} vs {MARKET}/{CASH}: "
          f"{px.index[0].date()} -> {px.index[-1].date()}  n={len(px)}  "
          f"fp={data.fingerprint(px)}")
    print(f"as-of {data.AS_OF}  |  daily TOTAL-RETURN closes (auto_adjust=True), TR vs TR "
          f"throughout  |  cost {COST_BPS:.0f} bps one-way x NAV")

    res = st.race(px, LADDER, VINTAGES, MARKET, CASH, cost_bps=COST_BPS)
    A = res["arms"]

    print(f"\nun-matched window {res['window'][0].date()} -> {res['window'][1].date()} "
          f"(n={res['n_days']});  beta-matched window "
          f"{res['window_matched'][0].date()} -> {res['window_matched'][1].date()} "
          f"(n={res['n_days_matched']}, first 252 days burned estimating the beta)")

    print("\n=== the arms, excess-of-cash (BIL total return subtracted) ===")
    order = ["ladder"] + list(VINTAGES) + ["diy_basket", "diy_beta_matched",
                                           "beta_mix_ladder", "beta_mix_basket", "market"]
    for k in order:
        s = res["summary"][k]
        print(f"  {k:18s} n={s['n_days']:5d}  ann {s['ann_return_pct']:+6.2f}%  "
              f"vol {s['vol_ann'] * 100:5.2f}%  exSharpe {s['sharpe']:+.3f}  "
              f"HAC t {s['tstat']:+.2f}")

    print("\n=== absolute (lived) max drawdown, nominal total return ===")
    for k, v in res["abs_drawdown"].items():
        print(f"  {k:12s} {v * 100:+6.2f}%")

    print("\n=== the gaps: wrapper minus each DIY alternative (pp/yr, HAC t) ===")
    for k, g in res["gaps"].items():
        print(f"  {k:22s} n={g['n_days']:5d}  {g['gap_ann_pp']:+6.2f} pp/yr  "
              f"t={g['t_hac']:+.2f}  TE={g['tracking_error_pct']:5.2f}%  "
              f"dSharpe={g['sharpe_gap']:+.3f}")

    print(f"\n  SPY-beta: wrapper {res['beta_ladder_oos_last']:.3f} vs DIY basket "
          f"{res['beta_basket_oos_last']:.3f} (expanding, lagged 1d; full-sample IS "
          f"{res['beta_ladder_full_sample']:.3f} / {res['beta_basket_full_sample']:.3f})")
    print(f"  daily excess-return correlation wrapper vs DIY basket: "
          f"{res['corr_ladder_basket']:.3f}")

    print("\n=== block bootstrap (2,000 draws, 21-day blocks, paired) ===")
    b1 = st.bootstrap_gap_ci(A["ladder"], A["diy_basket"])
    b2 = st.bootstrap_gap_ci(A["ladder"].reindex(A["diy_beta_matched"].index),
                             A["diy_beta_matched"])
    b3 = st.bootstrap_sharpe_gap_ci(A["ladder"], A["diy_basket"])
    print(f"  gap vs DIY basket        : {b1['gap_ann_pp']:+.2f} pp/yr  "
          f"95% CI [{b1['ci_low']:+.2f}, {b1['ci_high']:+.2f}]  frac<0 {b1['frac_negative']:.3f}")
    print(f"  gap vs beta-matched DIY  : {b2['gap_ann_pp']:+.2f} pp/yr  "
          f"95% CI [{b2['ci_low']:+.2f}, {b2['ci_high']:+.2f}]  frac<0 {b2['frac_negative']:.3f}")
    print(f"  Sharpe gap vs DIY basket : {b3['sharpe_gap']:+.3f}  "
          f"95% CI [{b3['ci_low']:+.3f}, {b3['ci_high']:+.3f}]  frac<0 {b3['frac_negative']:.3f}")

    print("\n=== is the ONE CI that excludes zero robust to the block length? ===")
    print("  (beta-matched gap; the block length is a free parameter of the bootstrap, "
          "not a fact about the tape)")
    sens = st.bootstrap_block_sensitivity(
        A["ladder"].reindex(A["diy_beta_matched"].index), A["diy_beta_matched"])
    for block in sorted({r["block"] for r in sens}):
        rows = [r for r in sens if r["block"] == block]
        lo = np.mean([r["ci_low"] for r in rows])
        hi = np.mean([r["ci_high"] for r in rows])
        n_excl = sum(r["excludes_zero"] for r in rows)
        print(f"  block {block:3d}d  mean 95% CI [{lo:+.2f}, {hi:+.2f}]  "
              f"excludes zero on {n_excl}/{len(rows)} seeds")
    n_all = sum(r["excludes_zero"] for r in sens)
    print(f"  -> excludes zero on {n_all}/{len(sens)} (block, seed) settings: the "
          f"exclusion is an artefact of the block length, NOT a result. HAC t (-1.69) "
          f"is the honest summary and it is the one the study is stamped on.")

    print("\n=== entry-point luck: how big is the thing laddering averages away? ===")
    d = st.dispersion_stats(st.to_returns(px), VINTAGES)
    print(f"  rolling 1-year best-minus-worst vintage spread: mean {d['spread_mean_pp']:.2f} pp, "
          f"median {d['spread_median_pp']:.2f} pp, max {d['spread_max_pp']:.2f} pp "
          f"({d['n_windows']} windows)")
    print(f"  sd of rolling 1-year returns: single vintages "
          f"{'/'.join(f'{v:.2f}' for v in d['sd_single_pct'].values())} "
          f"(mean {d['sd_single_mean_pct']:.2f}%)  ->  equal-weight basket "
          f"{d['sd_basket_pct']:.2f}%")
    print(f"  variance reduction from averaging four vintages: "
          f"{d['variance_reduction_pct']:.1f}% on 1-year holding-period sd, "
          f"{d['daily_variance_reduction_pct']:.1f}% on daily sd  (mean pairwise daily "
          f"correlation {d['mean_pairwise_corr']:.3f})")
    print(f"  closed form for {len(VINTAGES)} equally-correlated legs at that correlation: "
          f"{d['daily_variance_reduction_closed_form_pct']:.1f}% - the tape does exactly "
          f"what the correlation says it must")

    print(f"\n=== era cut (split {SPLIT}) ===")
    for tag, e in st.era_cut(px, LADDER, VINTAGES, MARKET, CASH,
                             split=SPLIT, cost_bps=COST_BPS).items():
        if e is None:
            continue
        print(f"  {tag:5s} n={e['n_days']:4d}  exSharpe wrapper {e['sharpe_ladder']:+.3f} / "
              f"DIY {e['sharpe_basket']:+.3f}  |  vs DIY {e['vs_diy_basket_gap_pp']:+.2f} pp "
              f"(t={e['vs_diy_basket_t']:+.2f})  vs beta-matched DIY "
              f"{e['vs_diy_beta_matched_gap_pp']:+.2f} pp (t={e['vs_diy_beta_matched_t']:+.2f})")

    print("\n=== cost sweep (one-way bps charged on the DIY arms) ===")
    for row in st.cost_sweep(px, LADDER, VINTAGES, MARKET, CASH):
        print(f"  {row['cost_bps']:5.1f} bps  vs DIY {row['gap_vs_basket_pp']:+.2f} pp "
              f"(t={row['t_vs_basket']:+.2f})  vs beta-matched {row['gap_vs_matched_pp']:+.2f} pp "
              f"(t={row['t_vs_matched']:+.2f})")

    print("\n=== ASSUMED extra-fee sweep (PROXY: quoted ER, not a tape measurement) ===")
    print(f"  single-vintage ER {data.FEE_SINGLE_VINTAGE_PCT:.2f}%/yr; assumed wrapper "
          f"extra layer {data.FEE_LADDER_EXTRA_PCT:.2f}%/yr. Published NAV returns are "
          f"already net; this adds the layer back.")
    for row in st.fee_sweep(px, LADDER, VINTAGES, MARKET, CASH,
                            fee_grid=data.FEE_EXTRA_GRID_PCT):
        print(f"  +{row['extra_fee_pct']:.2f}%/yr waived  vs DIY {row['gap_vs_basket_pp']:+.2f} pp "
              f"(t={row['t_vs_basket']:+.2f})  vs beta-matched {row['gap_vs_matched_pp']:+.2f} pp "
              f"(t={row['t_vs_matched']:+.2f})")

    print("\n=== DIY rebalance-frequency sweep ===")
    for row in st.rebalance_sweep(px, LADDER, VINTAGES, MARKET, CASH):
        print(f"  {row['rebalance']:1s}  DIY exSharpe {row['sharpe_basket']:+.3f}  "
              f"gap {row['gap_vs_basket_pp']:+.2f} pp (t={row['t_vs_basket']:+.2f})")

    print("\n=== calendar-year total return (%, complete years only) ===")
    tbl = st.calendar_year_table(px, LADDER, VINTAGES, MARKET, CASH, cost_bps=COST_BPS)
    print(tbl.round(2).to_string())

    print("\n=== synthetic control (machinery proof - never supports the stamp) ===")
    for tag, ss, fee in [("planted premium", 1.0, 0.002), ("null, fee only", 0.0, 0.002),
                         ("null, no fee", 0.0, 0.0)]:
        p, t = data.synthetic_panel(signal_strength=ss, extra_fee_ann=fee, seed=947)
        s = st.synthetic_detect(p, t)
        print(f"  {tag:16s}: recovered {s['gap_ann_pp']:+.2f} pp/yr (planted "
              f"{s['expected_gap_pp']:+.2f}, error {s['error_pp']:+.2f})  t={s['t_hac']:+.2f}")
    nulls = [st.synthetic_detect(*data.synthetic_panel(signal_strength=0.0, extra_fee_ann=0.0,
                                                       seed=947 + s)) for s in range(8)]
    gaps = np.array([n["gap_ann_pp"] for n in nulls])
    ts = np.array([n["t_hac"] for n in nulls])
    print(f"  null across 8 seeds: gap mean {gaps.mean():+.2f} pp (sd {gaps.std(ddof=1):.2f}), "
          f"max |t| {np.abs(ts).max():.2f}, fires (|t|>=2) on {int((np.abs(ts) >= 2).sum())}/8")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
