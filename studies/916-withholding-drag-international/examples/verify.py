"""Real-tape verification — Study 916 (Withholding Drag).

Prints every number quoted in ``docs/results.md`` and frozen into the ``R`` dict in
``notebooks/build_notebooks.py``.

Reads the shared cache (total-return closes plus the price-only / distribution leg),
measures each fund's realised net distribution yield, validates the ruler against the
funds' actual cash dividends, races the broad developed-ex-US funds against a
single-country blend of the same market, and reports the withholding inference as a
sweep over an assumed effective rate. Network only on ``--fetch``.

    python studies/916-withholding-drag-international/examples/verify.py
    python studies/916-withholding-drag-international/examples/verify.py --fetch
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from withholding import data, strategy as st  # noqa: E402

from quantlab import repro  # noqa: E402

FUND = "VEA"
BLEND_COLS = ["EWJ", "EWU", "EWG"]


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()

    tr = data.load_prices()
    dist = data.load_distributions()
    px = pd.DataFrame({k: v["close"] for k, v in dist.items()})
    E = data.EXPENSE_RATIO
    bench_expense = sum(data.BLEND_WEIGHTS[k] * E[k] for k in BLEND_COLS)

    print(f"as-of {data.AS_OF}  |  shared cache {data.DEFAULT_CACHE}")
    print(f"blend {data.BLEND_WEIGHTS} — weighted expense {bench_expense*1e4:.0f} bp "
          f"(ASSUMPTION: fact-sheet fees)")

    # ---------------------------------------------------------------- #
    print("\n=== 1. the ruler: differenced yield vs the funds' actual cash dividends ===")
    for t in list(data.BROAD) + list(data.SINGLE):
        cc = st.distribution_cross_check(tr[t].dropna(), px[t].dropna(),
                                         dist[t]["dividend"])
        print(f"  {t:5s} differenced {cc['differenced_bp']:6.1f} bp/yr   "
              f"declared cash {cc['cash_dividend_bp']:6.1f} bp/yr   "
              f"residual {cc['residual_bp']:+.2f} bp   ({cc['n_ex_dates']} ex-dates)")

    # ---------------------------------------------------------------- #
    print("\n=== 2. resolution check: EFA vs IEFA (near-twins, known 26 bp fee gap) ===")
    pair = ["EFA", "IEFA"]
    c = tr[pair].dropna().index.intersection(px[pair].dropna().index)
    cal = st.fee_calibration(tr.loc[c], px.loc[c], "EFA", "IEFA", E)
    print(f"  {cal['start']} -> {cal['end']}  n={cal['n_days']:,}")
    print(f"  EFA {cal['fund_yield_bp']:.1f} bp/yr vs IEFA {cal['bench_yield_bp']:.1f} bp/yr"
          f"  -> gap {cal['gap_bp']:+.1f} bp (HAC t {cal['t_hac']:+.2f}, "
          f"CI [{cal['ci_low_bp']:+.1f}, {cal['ci_high_bp']:+.1f}])")
    print(f"  fee gap alone would predict {cal['expected_gap_bp']:+.1f} bp "
          f"-> residual {cal['residual_bp']:+.1f} bp (index composition, not measurement)")

    # ---------------------------------------------------------------- #
    print("\n=== 3. headline: broad fund vs single-country blend, realised income yield ===")
    for fund in data.BROAD:
        idx = tr[[fund] + BLEND_COLS].dropna().index.intersection(
            px[[fund] + BLEND_COLS].dropna().index)
        b_tr, b_px = st.blend_index(tr.loc[idx, BLEND_COLS], px.loc[idx, BLEND_COLS],
                                    data.BLEND_WEIGHTS)
        g = st.yield_gap(tr.loc[idx, fund], px.loc[idx, fund], b_tr, b_px,
                         fund_expense=E[fund], bench_expense=bench_expense)
        print(f"  {fund:5s} {g['start']}->{g['end']} n={g['n_days']:,}  "
              f"fund {g['fund_yield_bp']:6.1f}  blend {g['bench_yield_bp']:6.1f}  "
              f"gap {g['gap_bp']:+7.1f} (t {g['t_hac']:+.2f})  "
              f"fee-adj {g['gap_fee_adj_bp']:+7.1f} (t {g['t_hac_fee_adj']:+.2f}, "
              f"CI [{g['ci_low_fee_adj_bp']:+.1f}, {g['ci_high_fee_adj_bp']:+.1f}])")

    # the headline fund, kept for the sections below
    idx = tr[[FUND] + BLEND_COLS].dropna().index.intersection(
        px[[FUND] + BLEND_COLS].dropna().index)
    b_tr, b_px = st.blend_index(tr.loc[idx, BLEND_COLS], px.loc[idx, BLEND_COLS],
                                data.BLEND_WEIGHTS)
    head = st.yield_gap(tr.loc[idx, FUND], px.loc[idx, FUND], b_tr, b_px,
                        fund_expense=E[FUND], bench_expense=bench_expense)
    stamp = tr.loc[idx, [FUND] + BLEND_COLS]
    print(f"\n  data stamp: {FUND}+blend {idx[0].date()} -> {idx[-1].date()}  "
          f"n={len(idx):,}  fingerprint {data.fingerprint(stamp)} "
          f"/ quantlab {repro.fingerprint(stamp)}")

    # ---------------------------------------------------------------- #
    print("\n=== 4. calendar-year income yields (%) — complete years only ===")
    y_fund = st.annual_income_yield(tr.loc[idx, FUND], px.loc[idx, FUND])
    y_bench = st.annual_income_yield(b_tr, b_px)
    for yr in y_fund.index:
        if yr in y_bench.index:
            print(f"  {yr}: {FUND} {y_fund.loc[yr, 'yield_pct']:.2f}%   "
                  f"blend {y_bench.loc[yr, 'yield_pct']:.2f}%   "
                  f"gap {(y_bench.loc[yr, 'yield_pct'] - y_fund.loc[yr, 'yield_pct'])*100:+6.1f} bp")

    # ---------------------------------------------------------------- #
    print("\n=== 5. era cut (split 2017-01-01) ===")
    for tag, e in st.era_cut(tr.loc[idx, FUND], px.loc[idx, FUND], b_tr, b_px,
                             split="2017-01-01", fund_expense=E[FUND],
                             bench_expense=bench_expense).items():
        if e is None:
            continue
        print(f"  {tag:5s} n={e['n_days']:,}  fund {e['fund_yield_bp']:.1f}  "
              f"blend {e['bench_yield_bp']:.1f}  gap {e['gap_bp']:+.1f} (t {e['t_hac']:+.2f})  "
              f"fee-adj {e['gap_fee_adj_bp']:+.1f} (t {e['t_hac_fee_adj']:+.2f})")

    # ---------------------------------------------------------------- #
    print("\n=== 6. blend-weight sweep (the composition ASSUMPTION) ===")
    sets = {
        "EAFE-ish 50/30/20": data.BLEND_WEIGHTS,
        "equal 1/3 each": {"EWJ": 1 / 3, "EWU": 1 / 3, "EWG": 1 / 3},
        "Japan-heavy 70/20/10": {"EWJ": 0.70, "EWU": 0.20, "EWG": 0.10},
        "UK-heavy 30/50/20": {"EWJ": 0.30, "EWU": 0.50, "EWG": 0.20},
    }
    for r in st.blend_weight_sweep(tr.loc[idx, FUND], px.loc[idx, FUND],
                                   tr.loc[idx, BLEND_COLS], px.loc[idx, BLEND_COLS],
                                   sets, fund_expense=E[FUND],
                                   bench_expense=bench_expense):
        print(f"  {r['weights']:22s} blend {r['bench_yield_bp']:6.1f}  "
              f"gap {r['gap_bp']:+7.1f}  fee-adj {r['gap_fee_adj_bp']:+7.1f}  "
              f"(t {r['t_hac']:+.2f})")

    # ---------------------------------------------------------------- #
    print("\n=== 6b. expense-ratio sweep (today's fees on a 19-year window) ===")
    print("  the headline adds back a 47 bp fee gap from 2026 fact sheets; fees FELL")
    print("  over the sample, so the true time-averaged gap is smaller and 47 bp is an")
    print("  UPPER bound on the fee-adjusted (i.e. hypothesis-flattering) gap.")
    for r in st.fee_gap_sweep(head):
        tag = "  <- headline ASSUMPTION" if abs(r["fee_gap_bp"] - 47.0) < 1e-9 else ""
        print(f"    fee gap {r['fee_gap_bp']:5.1f} bp -> fee-adj gap "
              f"{r['gap_fee_adj_bp']:+7.1f} bp (t {r['t_hac']:+.2f}, "
              f"CI [{r['ci_low_bp']:+.1f}, {r['ci_high_bp']:+.1f}]){tag}")

    # ---------------------------------------------------------------- #
    print("\n=== 7. INFERENCE — implied gross yield and drag (NOT measured) ===")
    print(f"  measured {FUND} net distribution yield: {head['fund_yield_bp']:.0f} bp/yr")
    print("  assumed effective withholding rate -> implied gross / implied drag:")
    for r in st.implied_withholding(head["fund_yield_bp"]):
        tag = "  <- central ASSUMPTION" if abs(r["w"] - data.ASSUMED_EFFECTIVE_WITHHOLDING) < 1e-9 else ""
        print(f"    w = {r['w']:.0%}  gross {r['gross_bp']:5.0f} bp  "
              f"drag {r['drag_bp']:5.0f} bp/yr{tag}")

    # ---------------------------------------------------------------- #
    print("\n=== 8. is it bankable? total-return race, excess of BIL, cost sweep ===")
    ridx = idx.intersection(tr["BIL"].dropna().index)
    for row in st.cost_sweep(tr.loc[ridx, FUND], tr.loc[ridx, BLEND_COLS],
                             px.loc[ridx, BLEND_COLS], tr.loc[ridx, "BIL"],
                             data.BLEND_WEIGHTS):
        print(f"  cost {row['cost_bps']:5.1f} bps  {FUND} exSharpe {row['sharpe_fund']:+.3f}  "
              f"blend exSharpe {row['sharpe_bench']:+.3f}  adv {row['sharpe_adv']:+.3f}  "
              f"return adv {row['ret_adv_bp']:+.0f} bp/yr (HAC t {row['t_diff']:+.2f})")
    race = st.total_return_race(tr.loc[ridx, FUND],
                                st.blend_index(tr.loc[ridx, BLEND_COLS],
                                               px.loc[ridx, BLEND_COLS],
                                               data.BLEND_WEIGHTS, cost_bps=5.0)[0],
                                tr.loc[ridx, "BIL"], cost_bps=5.0)
    ci_f = st.bootstrap_sharpe_ci(race["e_fund"])
    ci_b = st.bootstrap_sharpe_ci(race["e_bench"])
    print(f"  bootstrap excess-Sharpe CI: {FUND} {ci_f['sharpe']:+.3f} "
          f"[{ci_f['ci_low']:+.3f}, {ci_f['ci_high']:+.3f}]   "
          f"blend {ci_b['sharpe']:+.3f} [{ci_b['ci_low']:+.3f}, {ci_b['ci_high']:+.3f}]")

    # ---------------------------------------------------------------- #
    print("\n=== 9. synthetic control (machinery proof — never supports the stamp) ===")
    for ss in (1.0, 0.0):
        s_tr, s_px, truth = data.synthetic_panel(signal_strength=ss, seed=916)
        d = st.synthetic_detect(s_tr, s_px, truth, n_boot=600)
        print(f"  signal_strength={ss:.0f}: planted {d['true_gap_bp']:5.1f} bp -> "
              f"measured {d['measured_gap_bp']:+6.1f} bp (t {d['t_hac']:+.2f}, "
              f"CI [{d['ci_low_bp']:+.1f}, {d['ci_high_bp']:+.1f}])")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
