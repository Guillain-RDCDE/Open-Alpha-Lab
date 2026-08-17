"""Real-tape verification — Study 956 (the ADR custody fee drag). Regenerates docs/results.md.

Loads fifteen ADR / home-line / FX triples from the shared desk cache, screens them for
whether the home line's adjusted close carries dividends at all, and estimates the
annualised total-return shortfall of the ADR against its home line, split into the
price-ratio placebo and the measured income gap. Then the bootstrap CIs, the leave-one-out,
the era cut, the withholding-assumption sweep, the break-threshold sweep, and the one
traded comparison (own the home line instead). Network only on ``--fetch``.

    python studies/956-adr-custody-fee-drag/examples/verify.py            # cache-only
    python studies/956-adr-custody-fee-drag/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from adr_drag import data, strategy as st  # noqa: E402
from quantlab.repro import data_stamp  # noqa: E402


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()

    frames = {p["adr"]: data.load_pair(p) for p in data.PAIRS}
    whts = {p["adr"]: p["wht"] for p in data.PAIRS}
    names = {p["adr"]: p["name"] for p in data.PAIRS}
    ctry = {p["adr"]: p["country"] for p in data.PAIRS}

    stamp_cols = pd.DataFrame({tk: df["adr_tr"] for tk, df in frames.items()}).dropna(how="all")
    print(data_stamp("ADR total-return panel", stamp_cols, asof=data.AS_OF))
    print(f"as-of {data.AS_OF}  |  {len(frames)} pairs loaded  |  "
          f"fingerprint(adr_tr panel) = {data.fingerprint(stamp_cols.fillna(0.0))}")

    # ----------------------------------------------------------------- screen
    kept, report = st.screen_frames(frames)
    print("\n=== coverage screen: does the HOME line's adjusted close carry dividends? ===")
    for tk in frames:
        r = report.loc[tk]
        flag = "keep" if r["pass"] else "DROP"
        print(f"  {flag}  {tk:5s} {names[tk]:15s} {ctry[tk]:15s} "
              f"home yield {float(r['local_yield'])*100:6.2f}%/yr  "
              f"ADR yield {float(r['adr_yield'])*100:6.2f}%/yr  ratio {float(r['yield_ratio']):8.2f}")
    print(f"  -> {len(kept)} of {len(frames)} pairs usable")

    # ----------------------------------------------------------------- panel
    tbl = st.panel_table(kept, whts)
    print("\n=== per-name annualised drag (positive = the ADR holder loses), bp/yr ===")
    print(f"  {'name':6s} {'yrs':>5s} {'seg':>3s} {'total':>8s} {'t':>7s} {'price':>8s} {'t':>7s} "
          f"{'income':>8s} {'t':>7s} {'homeYld':>8s} {'adrYld':>8s} {'c/ADS':>7s}")
    for tk in tbl.index:
        r = tbl.loc[tk]
        print(f"  {tk:6s} {float(r['years']):5.1f} {int(r['n_segments']):3d} "
              f"{float(r['drag_total'])*1e4:8.1f} {float(r['drag_t']):7.2f} "
              f"{float(r['price_drift'])*1e4:8.1f} {float(r['price_drift_t']):7.2f} "
              f"{float(r['income_gap'])*1e4:8.1f} {float(r['income_gap_t']):7.2f} "
              f"{float(r['gross_yield'])*100:7.2f}% {float(r['net_yield'])*100:7.2f}% "
              f"{float(r['income_gap_cents']):7.2f}")

    print("\n=== pooled across names (names are the observations) ===")
    for col, label in (("drag_total", "total drag"), ("price_drift", "price placebo"),
                       ("income_gap", "income gap"), ("income_gap_cents", "income gap c/ADS")):
        p = st.pooled(tbl, col)
        unit = "c/ADS/yr" if "cents" in col else "bp/yr"
        scale = 1.0 if "cents" in col else 1e4
        print(f"  {label:18s} mean {p['mean']*scale:+7.2f} {unit}  median {p['median']*scale:+7.2f}"
              f"  sd {p['sd']*scale:6.2f}  t {p['t']:+5.2f}  positive {int(p['share_positive']*p['n'])}/{p['n']}")

    bs = st.name_bootstrap(tbl, "income_gap")
    print(f"\n  income gap: name-bootstrap 95% CI [{bs['ci_low']*1e4:+.2f}, {bs['ci_high']*1e4:+.2f}] bp/yr"
          f"  share<=0 {bs['frac_le_zero']:.3f}  sign test {bs['n_positive']}/{bs['n']} positive, p = {bs['sign_p']:.4f}")
    bsc = st.name_bootstrap(tbl, "income_gap_cents")
    print(f"  income gap: name-bootstrap 95% CI [{bsc['ci_low']:+.2f}, {bsc['ci_high']:+.2f}] cents/ADS/yr")

    print("\n=== per-name block-bootstrap CI on the income gap (63-day blocks, 600 draws) ===")
    for tk in tbl.index:
        _, _, gap = st.income_series(kept[tk])
        ci = st.bootstrap_drag_ci(gap, break_thresh=1.0, n_boot=600, seed=956)
        print(f"  {tk:5s} gap {ci['drag']*1e4:+7.1f} bp/yr  95% CI [{ci['ci_low']*1e4:+7.1f}, "
              f"{ci['ci_high']*1e4:+7.1f}]  share<=0 {ci['frac_le_zero']:.3f}")

    print("\n=== leave-one-out (cross-name mean of the income gap, bp/yr) ===")
    loo = st.leave_one_out(tbl, "income_gap")
    for tk, r in loo.iterrows():
        print(f"  drop {tk:5s} -> mean {r['mean']*1e4:+6.2f}  median {r['median']*1e4:+6.2f}  t {r['t']:+5.2f}")

    # ----------------------------------------------------------------- eras
    print("\n=== era cut (split 2015-01-01) ===")
    eras = st.era_cut(kept, whts, split="2015-01-01")
    for tag, e in eras.items():
        if e is None:
            continue
        print(f"  {tag:5s} {e['start'].date()} -> {e['end'].date()}  n_names {e['n_names']}: "
              f"income gap mean {e['income_gap']['mean']*1e4:+6.2f} bp/yr (t {e['income_gap']['t']:+5.2f}, "
              f"positive {int(e['income_gap']['share_positive']*e['income_gap']['n'])}/{e['income_gap']['n']})  |  "
              f"price placebo {e['price_drift']['mean']*1e4:+6.2f} bp/yr (t {e['price_drift']['t']:+5.2f})")

    # ----------------------------------------------------------------- sweeps
    print("\n=== ASSUMPTION sweep: the withholding rate (the only non-tape input) ===")
    for row in st.withholding_sweep(kept, whts):
        print(f"  wht scale {row['scale']:4.1f} x treaty  -> residual 'custody' mean "
              f"{row['custody_mean']*1e4:+8.2f} bp/yr (t {row['custody_t']:+6.2f}), "
              f"positive {row['share_positive']*100:.0f}% of names")

    print("\n=== break-threshold sweep (level-shift detector) ===")
    for row in st.break_threshold_sweep(kept, whts):
        print(f"  thresh {row['thresh']:.2f}  total drag mean {row['drag_mean']*1e4:+6.2f} bp/yr "
              f"(t {row['drag_t']:+5.2f})  price placebo {row['price_drift_mean']*1e4:+6.2f} bp/yr  "
              f"mean segments {row['n_segments']:.2f}")

    # ----------------------------------------------------------------- traded
    cash = data.load_prices([data.CASH_TICKER], kind="tr")[data.CASH_TICKER].dropna()
    print("\n=== the only traded leg: own the home line instead of the ADR ===")
    print("    (equal-weight baskets, excess-of-cash vs BIL, long-only so no borrow,")
    print("     one-way FX conversion x NAV executed at t+1, foreign custody swept)")
    for row in st.switch_cost_sweep(kept, cash):
        print(f"  fx {row['fx_cost_bps']:5.1f} bps + custody {row['foreign_custody_bps_per_year']:5.1f} bp/yr"
              f"  -> ADR Sharpe {row['sharpe_adr']:+.3f}  home Sharpe {row['sharpe_local']:+.3f}"
              f"  adv {row['sharpe_adv']:+.3f}  ann diff {row['ann_diff']*1e4:+7.1f} bp"
              f"  HAC t {row['t_diff']:+.2f}  n={row['n_days']}")

    # ----------------------------------------------------------------- synth
    print("\n=== synthetic control (machinery proof only - never supports the stamp) ===")
    for planted in (25.0, 0.0):
        fr, truth = data.synthetic_panel(n_names=10, drag_bps_per_year=planted,
                                         signal_strength=1.0 if planted > 0 else 0.0)
        w = {k: truth["per_name"][k]["wht"] for k in fr}
        d = st.synthetic_detect(fr, w)
        print(f"  planted custody {truth['custody_drag_per_year']*1e4:5.1f} + withholding "
              f"{truth['wht_drag_per_year']*1e4:5.1f} = {truth['total_drag_per_year']*1e4:5.1f} bp/yr"
              f"  ->  income gap {d['income_gap']['mean']*1e4:+6.2f} bp/yr (t {d['income_gap']['t']:+6.2f})"
              f"  residual custody {d['custody']['mean']*1e4:+6.2f}"
              f"  price placebo {d['price_drift']['mean']*1e4:+6.2f} (t {d['price_drift']['t']:+5.2f})")
    nulls = []
    for s in range(8):
        fr, truth = data.synthetic_panel(n_names=10, drag_bps_per_year=25.0,
                                         signal_strength=0.0, seed=956 + 7 * s)
        w = {k: truth["per_name"][k]["wht"] for k in fr}
        nulls.append(st.synthetic_detect(fr, w)["income_gap"]["mean"])
    nulls = np.array(nulls)
    print(f"  null x8 seeds: income gap mean {nulls.mean()*1e4:+6.2f} bp/yr "
          f"(sd {nulls.std(ddof=1)*1e4:.2f}), |mean| >= 5 bp on {(np.abs(nulls) >= 5e-4).sum()}/8")
    fr, truth = data.synthetic_panel(n_names=6, drag_bps_per_year=25.0, ratio_break=0.7)
    w = {k: truth["per_name"][k]["wht"] for k in fr}
    d = st.synthetic_detect(fr, w)
    print(f"  with a planted 0.70-log ADS-ratio break: recovered total drag "
          f"{d['drag']['mean']*1e4:+6.2f} bp/yr (planted {truth['total_drag_per_year']*1e4:.1f})")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
