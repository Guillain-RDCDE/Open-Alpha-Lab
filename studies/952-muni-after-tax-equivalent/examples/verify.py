"""Real-tape verification — Study 952 (After-Tax Equivalent). Regenerates docs/results.md.

Reads the cached daily **total-return** and **price-only** closes for the muni legs
(MUB, VTEB, SUB, HYD), the taxable-credit legs (AGG, LQD, VCIT) and cash (BIL),
reconstructs each leg's monthly income return as ``total - price``, and races muni against
taxable **after tax** across the bracket ladder — reporting the break-even effective
marginal rate for every pairing, the HAC *t* and bootstrap CI at the top bracket, the era
cut, and the state / capital-gains / cost / borrow sweeps. Network only on ``--fetch``.

    python studies/952-muni-after-tax-equivalent/examples/verify.py
    python studies/952-muni-after-tax-equivalent/examples/verify.py --fetch
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from after_tax import data, strategy as st  # noqa: E402

PAIRS = [
    ("MUB", "AGG"), ("MUB", "LQD"), ("MUB", "VCIT"),
    ("VTEB", "AGG"), ("VTEB", "VCIT"), ("SUB", "BIL"), ("HYD", "LQD"),
]
HEADLINE = ("MUB", "VCIT")   # the duration-matched, like-for-like credit comparison
TOP = st.tax_profile(fed_rate=0.37)   # 37% + 3.8% NIIT = 40.8% effective


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    tot = data.load_prices()
    pri = data.load_price_only()
    panel = data.decompose(tot, pri, floor_income=True)

    m = panel["total"]
    print(f"monthly panel {m.index[0]} -> {m.index[-1]}  n={len(m)}  "
          f"fp={data.fingerprint(m.fillna(0.0))}")
    print(f"as-of {data.AS_OF}  (partial months dropped)")

    print("\n=== reconstructed income yields (total return - price return, annualised) ===")
    for tk in data.TICKERS:
        s = panel["income"][tk].dropna()
        raw = (panel["total"][tk] - panel["price"][tk]).dropna()
        print(f"  {tk:5s} n={len(s):3d}  income {s.mean() * 12 * 100:5.2f}%/yr  "
              f"price {panel['price'][tk].dropna().mean() * 12 * 100:+5.2f}%/yr  "
              f"(negative-income months before floor: {int((raw < 0).sum())})")

    print("\n=== break-even effective marginal rate (mean-based, closed form) ===")
    print("  NOTE: the point estimate is a ratio of two sample means; see the CI block below.")
    for a, b in PAIRS:
        be = st.breakeven_rate(panel, a, b)
        tag = "point est. > 0" if be["tax_driven"] else "point est. <= 0 (muni wins PRE-tax)"
        print(f"  {a:5s} vs {b:5s} n={be['n_months']:3d}  pre-tax diff {be['pretax_diff_bps']:+6.2f} bps/mo "
              f"(t={be['pretax_t']:+5.2f})  break-even {be['breakeven'] * 100:+6.1f}%   [{tag}]")
    print("  (no pairing's pre-tax difference clears |t| >= 2, so the sign of the "
          "break-even is NOT established by a test)")

    print("\n=== HOW WELL IS THE BREAK-EVEN IDENTIFIED? (block bootstrap on tau*) ===")
    print("  total-return break-even, 95% CI, and the price-leg difference that widens it")
    for a, b in PAIRS:
        ci = st.breakeven_ci(panel, a, b)
        print(f"  {a:5s} vs {b:5s} tau* {ci['breakeven'] * 100:+6.1f}%  "
              f"CI[{ci['ci_low'] * 100:+7.1f},{ci['ci_high'] * 100:+7.1f}]  "
              f"P(tau*<0)={ci['p_below_zero']:.2f}  P(tau*>40.8%)={ci['p_above_top']:.2f}  "
              f"price-leg diff {ci['price_leg_diff_bps']:+5.2f} bps/mo (t={ci['price_leg_t']:+5.2f})")
    print("\n  the income-leg-only (tax-equivalent-yield) break-even — the half the tape DOES pin down")
    for a, b in PAIRS:
        ib = st.income_breakeven(panel, a, b)
        print(f"  {a:5s} vs {b:5s} tau* {ib['breakeven'] * 100:+6.1f}%  "
              f"CI[{ib['ci_low'] * 100:+6.1f},{ib['ci_high'] * 100:+6.1f}]  "
              f"(income {ib['muni_income_ann_pct']:.2f}%/yr vs {ib['taxable_income_ann_pct']:.2f}%/yr)")

    print("\n=== after-tax race at 40.8% (37% + NIIT), excess of after-tax BIL ===")
    for a, b in PAIRS:
        r = st.race(panel, a, b, TOP)
        tag = "excess of after-tax BIL" if r["excess_of_cash"] else "raw after-tax (BIL is the arm)"
        print(f"  {a:5s} vs {b:5s} n={r['n_months']:3d}  diff {r['diff_bps']:+6.2f} bps/mo "
              f"({r['diff_ann_pct']:+5.2f} pp/yr)  HAC t={r['t_diff']:+5.2f}  "
              f"CI[{r['ci_low_bps']:+6.1f},{r['ci_high_bps']:+6.1f}]  "
              f"Sharpe {r['sharpe_muni_gross']:+.2f} gross / {r['sharpe_muni']:+.2f} net "
              f"vs {r['sharpe_taxable']:+.2f}  [{tag}]")

    print("\n=== WHERE DOES THE t-STAT COME FROM? (after-tax diff = pre-tax diff + tax term) ===")
    print("  the tax term is a coupon stream: big in the MEAN, ~nil in the VARIANCE, so it")
    print("  lifts |t| by construction. A |t| >= 2 built this way tests the tax code, not a market.")
    for x, y in PAIRS:
        d = st.tax_constant_decomposition(panel, x, y, TOP)
        print(f"  {x:5s} vs {y:5s} pre-tax {d['pretax_mean_bps']:+6.2f} bps (t={d['pretax_t']:+5.2f})  "
              f"+ tax term {d['tax_mean_bps']:+6.2f} bps (sd {d['tax_sd_bps']:4.2f})  "
              f"= {d['total_mean_bps']:+6.2f} bps (t={d['total_t']:+5.2f})  "
              f"| tax share of mean {d['mean_share_tax'] * 100:6.1f}%, of variance "
              f"{d['var_share_tax'] * 100:5.2f}%")

    a, b = HEADLINE
    print(f"\n=== bracket ladder — {a} vs {b} (state 0%) ===")
    print(st.bracket_sweep(panel, a, b).to_string(index=False,
                                                  float_format=lambda v: f"{v:8.3f}"))
    print(f"\n=== bracket ladder — MUB vs AGG (the Aggregate comparator) ===")
    print(st.bracket_sweep(panel, "MUB", "AGG").to_string(index=False,
                                                          float_format=lambda v: f"{v:8.3f}"))

    print(f"\n=== era cut at 2017-01 (40.8%) ===")
    for pair in [HEADLINE, ("MUB", "AGG"), ("MUB", "LQD")]:
        eras = st.era_cut(panel, pair[0], pair[1], TOP)
        for tag, e in eras.items():
            if e is None:
                continue
            print(f"  {pair[0]}-{pair[1]:5s} {tag:5s} n={e['n_months']:3d}  "
                  f"{e['diff_bps']:+6.2f} bps/mo  t={e['t_diff']:+5.2f}  "
                  f"exSharpe {e['sharpe_muni']:+.2f} vs {e['sharpe_taxable']:+.2f}")

    print(f"\n=== state-tax ASSUMPTION sweep — {a} vs {b} at 37% federal ===")
    print(st.state_sweep(panel, a, b).to_string(index=False,
                                                float_format=lambda v: f"{v:8.3f}"))

    print(f"\n=== capital-gains ASSUMPTION sweep — {a} vs {b} (price leg untaxed by default) ===")
    print(st.capgain_sweep(panel, a, b).to_string(index=False,
                                                  float_format=lambda v: f"{v:8.3f}"))

    print(f"\n=== cost sweep (one-way x NAV, asset-location choice) — {a} vs {b} ===")
    print(st.cost_sweep(panel, a, b).to_string(index=False,
                                               float_format=lambda v: f"{v:8.3f}"))

    print(f"\n=== borrow sweep (long {a} / short {b} spread; the short leg pays borrow) ===")
    print(st.borrow_sweep(panel, a, b).to_string(index=False,
                                                 float_format=lambda v: f"{v:8.3f}"))
    print("  (same sweep on MUB/AGG)")
    print(st.borrow_sweep(panel, "MUB", "AGG").to_string(index=False,
                                                         float_format=lambda v: f"{v:8.3f}"))

    print(f"\n=== income-floor sensitivity — {a} vs {b} ===")
    print(st.income_floor_sensitivity(tot, pri, a, b).to_string(index=False,
                                                                float_format=lambda v: f"{v:8.3f}"))

    print(f"\n=== the one signal arm: trailing-yield switch overlay (one execution lag) ===")
    for pair in [HEADLINE, ("MUB", "AGG")]:
        o = st.switch_overlay(panel, pair[0], pair[1], TOP)
        print(f"  {pair[0]}/{pair[1]:5s} n={o['n_months']:3d}  in-muni {o['in_muni_frac']:.1%}  "
              f"switches {o['n_switches']:2d}  overlay {o['overlay_ann_pct']:+5.2f}%/yr vs "
              f"hold-muni {o['muni_ann_pct']:+5.2f}%/yr  edge {o['edge_bps']:+6.2f} bps/mo "
              f"(t={o['t_edge']:+5.2f})  Sharpe {o['sharpe_overlay']:+.2f} vs {o['sharpe_muni']:+.2f}")

    print("\n=== synthetic control (machinery proof only — never supports the stamp) ===")
    for ss in (1.0, 0.0):
        errs, bes = [], []
        for s in range(8):
            p, truth = data.synthetic_panel(signal_strength=ss, seed=952 + s)
            d = st.synthetic_detect(p)
            bes.append(d["breakeven"])
            errs.append(d["breakeven"] - truth["planted_breakeven"])
        p, truth = data.synthetic_panel(signal_strength=ss, seed=952)
        d = st.synthetic_detect(p)
        print(f"  signal_strength={ss:.1f}: planted break-even {truth['planted_breakeven'] * 100:6.2f}%  "
              f"recovered {d['breakeven'] * 100:6.2f}%  "
              f"(8 seeds: mean err {sum(errs) / len(errs) * 100:+.2f} pp, "
              f"max |err| {max(abs(e) for e in errs) * 100:.2f} pp)")
        print(f"      pre-tax diff {d['pretax_diff_bps']:+6.2f} bps (t={d['pretax_t']:+5.2f})  "
              f"|  15 pp below break-even {d['diff_below_breakeven']:+6.2f} bps, "
              f"15 pp above {d['diff_above_breakeven']:+6.2f} bps (sign must flip)")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
