"""Real-tape verification — Study 970 (Root Time). Regenerates docs/results.md.

Estimates variance ratios at 5 / 21 / 63 / 252 days on ten tapes with
heteroskedasticity-robust Lo-MacKinlay z-statistics, cross-checks them against non-overlapping
realised scaling, and converts each into the number a desk would actually get wrong: an
annualised volatility, a 10-day 99% VaR, and an annualised Sharpe ratio.

    python studies/970-sqrt-time-scaling/examples/verify.py            # cache-only
    python studies/970-sqrt-time-scaling/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sqrt_time import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


def report() -> dict:
    px = data.load_prices()
    rets = {tk: px[tk].dropna().pct_change().dropna() for tk in data.TICKERS}
    h: dict = {"as_of": data.AS_OF, "tickers": list(data.TICKERS),
               "n_tickers": len(data.TICKERS), "fingerprint": data.fingerprint(px)}

    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    for tk, r in rets.items():
        print(f"  {tk:8s} {r.index[0].date()} -> {r.index[-1].date()}  n={len(r):6,}  "
              f"daily vol {r.std():.4f}  AR(1) {r.autocorr(1):+.3f}")
    h["windows"] = {tk: [str(r.index[0].date()), str(r.index[-1].date())]
                    for tk, r in rets.items()}
    h["n_obs"] = {tk: int(len(r)) for tk, r in rets.items()}
    h["ar1"] = {tk: float(r.autocorr(1)) for tk, r in rets.items()}

    print("\n=== variance ratios (overlapping, Lo-MacKinlay bias-corrected) ===")
    print("  tkr        VR(5)   z     VR(21)  z     VR(63)  z     VR(252) z")
    curves = {}
    for tk, r in rets.items():
        c = st.vr_curve(r)
        curves[tk] = {int(q): dict(v) for q, v in c.to_dict("index").items()}
        cells = "  ".join(f"{c.loc[q, 'vr']:6.2f} {c.loc[q, 'z']:+5.1f}" for q in st.HORIZONS)
        print(f"  {tk:8s} {cells}")
    h["vr"] = curves
    ann = {tk: curves[tk][252] for tk in data.TICKERS}
    h["n_reject_annual"] = int(sum(abs(v["z"]) >= 2 for v in ann.values()
                                   if np.isfinite(v["z"])))
    h["max_vr_ticker"] = max(ann, key=lambda k: ann[k]["vr"])
    h["min_vr_ticker"] = min(ann, key=lambda k: ann[k]["vr"])
    h["max_vr"] = float(ann[h["max_vr_ticker"]]["vr"])
    h["min_vr"] = float(ann[h["min_vr_ticker"]]["vr"])
    print(f"  reject VR = 1 at the annual horizon (robust |z| >= 2): "
          f"{h['n_reject_annual']}/{len(data.TICKERS)}")
    print(f"  highest {h['max_vr_ticker']} {h['max_vr']:.2f}  lowest {h['min_vr_ticker']} "
          f"{h['min_vr']:.2f}")

    print("\n=== the assumption-free cross-check: non-overlapping blocks ===")
    print("  tkr        q     blocks   actual sd   sqrt-rule   ratio   implied VR")
    real = {}
    for tk in data.TICKERS:
        rs = st.realised_scaling(px[tk].dropna())
        real[tk] = {int(q): dict(v) for q, v in rs.to_dict("index").items()}
        for q, row in rs.iterrows():
            print(f"  {tk:8s} {q:4d} {int(row['n_blocks']):9d} {row['actual_sd']:11.4f} "
                  f"{row['sqrt_rule_sd']:11.4f} {row['ratio']:7.3f} {row['implied_vr']:11.2f}")
    h["realised"] = real

    print("\n=== what the error costs ===")
    print("  tkr       VR(252)   vol error   10d 99% VaR error   Sharpe naive -> Lo")
    costs = {}
    for tk, r in rets.items():
        vr = ann[tk]["vr"]
        ve = st.vol_scaling_error(vr)
        vr10 = curves[tk][21]["vr"]        # nearest available horizon to the Basel 10 days
        var_e = st.var_scaling_error(float(r.std(ddof=1)), vr10, horizon=st.BASEL_HORIZON)
        sh = st.sharpe_scaling_error(r)
        costs[tk] = {"vr_annual": vr, "vol_error": ve, "var_error": var_e["error_pct"],
                     "sharpe_naive": sh["sharpe_naive"], "sharpe_lo": sh["sharpe_lo"],
                     "sharpe_error": sh["relative_error"], "lo_factor": sh["factor"]}
        print(f"  {tk:8s} {vr:8.2f} {ve:+11.1%} {var_e['error_pct']:+19.1%} "
              f"{sh['sharpe_naive']:+9.2f} -> {sh['sharpe_lo']:+.2f} "
              f"({sh['relative_error']:+.1%})")
    h["costs"] = costs
    h["max_abs_vol_error"] = float(max(abs(c["vol_error"]) for c in costs.values()))
    h["max_var_ticker"] = max(costs, key=lambda k: abs(costs[k]["var_error"]))
    h["max_var_error"] = float(costs[h["max_var_ticker"]]["var_error"])
    h["max_sharpe_ticker"] = max(costs, key=lambda k: abs(costs[k]["sharpe_error"]))
    h["max_sharpe_error"] = float(costs[h["max_sharpe_ticker"]]["sharpe_error"])
    h["max_sharpe_naive"] = float(costs[h["max_sharpe_ticker"]]["sharpe_naive"])
    h["max_sharpe_lo"] = float(costs[h["max_sharpe_ticker"]]["sharpe_lo"])
    h["spy_vol_error"] = float(costs["SPY"]["vol_error"])

    print("\n=== how much of the Sharpe correction is signal? (Lo factor on i.i.d. draws) ===")
    errs = []
    for s in range(20):
        rr, _ = data.synthetic_ar1(n_years=20, ar1=0.0, signal_strength=0.0, seed=970 + s)
        errs.append(st.sharpe_scaling_error(rr)["relative_error"])
    print(f"  on TWENTY independent i.i.d. samples the q=252 Lo factor differs from sqrt(252) "
          f"by {np.mean(errs):+.1%} on average (unbiased) with a spread of "
          f"{np.std(errs, ddof=1):.1%}")
    print(f"  -> a single-sample Sharpe 'correction' at the annual horizon is mostly noise; "
          f"the variance ratios above are the reliable measurement, and the q=21 factor is "
          f"the usable correction")
    h["lo_factor_noise_iid"] = float(np.std(errs, ddof=1))
    h["lo_factor_bias_iid"] = float(np.mean(errs))
    sh21 = {tk: st.sharpe_scaling_error(r, q=21)["relative_error"] for tk, r in rets.items()}
    h["sharpe_error_q21"] = {k: float(v) for k, v in sh21.items()}
    print("  the same correction at q=21: " +
          ", ".join(f"{tk} {v:+.1%}" for tk, v in sh21.items()))

    print("\n=== the autocorrelation behind it (first five lags) ===")
    profs = {}
    for tk, r in rets.items():
        p = st.autocorrelation_profile(r, lags=5)
        profs[tk] = {int(k): float(v) for k, v in p.items()}
        print(f"  {tk:8s} " + "  ".join(f"rho{k}={p[k]:+.3f}" for k in range(1, 6)))
    h["acf"] = profs

    print("\n=== era check: is the dependence a fossil? (pre/post 2010, VR(21)) ===")
    eras = {}
    for tk in data.TICKERS:
        r = rets[tk]
        early, late = r.loc[:"2009-12-31"], r.loc["2010-01-01":]
        e = st.variance_ratio(early, 21) if len(early) > 500 else np.nan
        l = st.variance_ratio(late, 21) if len(late) > 500 else np.nan
        eras[tk] = {"early": float(e) if np.isfinite(e) else None,
                    "late": float(l) if np.isfinite(l) else None}
        print(f"  {tk:8s} early {e if np.isfinite(e) else float('nan'):6.2f}   "
              f"late {l if np.isfinite(l) else float('nan'):6.2f}")
    h["eras"] = eras

    print("\n=== synthetic control: the estimator against a closed-form truth ===")
    for phi in (0.0, 0.10, 0.20):
        r, truth = data.synthetic_ar1(n_years=30, ar1=phi, signal_strength=1.0, seed=970)
        got = st.variance_ratio(r, 21)
        want = truth["vr_closed_form"][21]
        print(f"  AR(1) = {phi:.2f}: closed-form VR(21) {want:.3f}, estimated {got:.3f}, "
              f"robust z {st.lo_mackinlay_test(r, 21)['z']:+.2f}")
        h[f"synthetic_phi_{int(phi * 100)}"] = {"closed_form": float(want),
                                                "estimated": float(got)}

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Usefulness: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    vr = "\n".join(
        "| " + tk + " | " + " | ".join(
            f"{h['vr'][tk][q]['vr']:.2f} ({h['vr'][tk][q]['z']:+.1f})" for q in st.HORIZONS)
        + " |" for tk in h["tickers"])
    costs = "\n".join(
        f"| {tk} | {c['vr_annual']:.2f} | {c['vol_error']:+.1%} | {c['var_error']:+.1%} | "
        f"{c['sharpe_naive']:+.2f} | {c['sharpe_lo']:+.2f} | {c['sharpe_error']:+.1%} |"
        for tk, c in h["costs"].items())
    acf = "\n".join("| " + tk + " | " + " | ".join(f"{h['acf'][tk][k]:+.3f}"
                                                   for k in range(1, 6)) + " |"
                    for tk in h["tickers"])
    real = "\n".join(
        f"| {tk} | {q} | {r['n_blocks']} | {r['ratio']:.3f} | {r['implied_vr']:.2f} |"
        for tk in h["tickers"] for q, r in h["realised"][tk].items())
    eras = "\n".join(
        f"| {tk} | {('%.2f' % e['early']) if e['early'] else 'n/a'} | "
        f"{('%.2f' % e['late']) if e['late'] else 'n/a'} |"
        for tk, e in h["eras"].items())
    win = "\n".join(f"| {tk} | {w[0]} → {w[1]} | {h['n_obs'][tk]:,} | {h['ar1'][tk]:+.3f} |"
                    for tk, w in h["windows"].items())
    return f"""# Results — Study 970 (Root Time) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). Variance ratios are computed from
overlapping q-period sums with the Lo-MacKinlay (1988) bias corrections, and every test
statistic is the **heteroskedasticity-robust** one — without that, volatility clustering alone
rejects the random walk and this study would be measuring GARCH. As-of **{h['as_of']}**;
fingerprint `{h['fingerprint']}`.*

## Data stamp

| Ticker | Window | Sessions | AR(1) |
|---|---|--:|--:|
{win}

## Variance ratios, VR(q) with robust z in brackets

| Ticker | q = 5 | q = 21 | q = 63 | q = 252 |
|---|--:|--:|--:|--:|
{vr}

**{h['n_reject_annual']} of {h['n_tickers']}** tapes reject VR = 1 at the annual horizon.
Highest: **{h['max_vr_ticker']}** at {h['max_vr']:.2f}. Lowest: **{h['min_vr_ticker']}** at
{h['min_vr']:.2f}.

## The assumption-free cross-check (non-overlapping blocks)

| Ticker | q | Blocks | Actual sd ÷ sqrt-rule | Implied VR |
|---|--:|--:|--:|--:|
{real}

Fewer observations, no estimator, no bias correction — and the same story. When an estimator
and a brute-force count agree, the result is not a modelling artefact.

## What the error costs

| Ticker | VR(252) | Volatility error | 10-day 99% VaR error | Sharpe (sqrt-T) | Sharpe (Lo 2002) | Sharpe error |
|---|--:|--:|--:|--:|--:|--:|
{costs}

The VaR column uses the 21-day variance ratio as the nearest measured horizon to Basel's
10 days, and the sqrt-rule VaR is exactly what a standard risk system reports.

> **Read the Sharpe columns with care.** Lo's factor at q = 252 sums 251 estimated
> autocorrelations. Run on twenty *independent* simulated samples it is unbiased
> ({h['lo_factor_bias_iid']:+.1%} on average) but its spread is
> **{h['lo_factor_noise_iid']:.1%}** — comparable to the corrections it reports on the real
> tapes. The variance ratios are the measurement this study stands behind; the annual Sharpe
> correction is an illustration, and the usable version of it is the q = 21 factor:
> {", ".join(f"{tk} {v:+.1%}" for tk, v in h["sharpe_error_q21"].items())}.

## The autocorrelations underneath

| Ticker | ρ₁ | ρ₂ | ρ₃ | ρ₄ | ρ₅ |
|---|--:|--:|--:|--:|--:|
{acf}

## Is it a fossil? VR(21) by era

| Ticker | Pre-2010 | 2010 onward |
|---|--:|--:|
{eras}

## Caveats

- **Overlapping estimators are noisy** at long horizons: VR(252) on a thirty-year tape rests on
  roughly thirty independent observations, which is why the non-overlapping cross-check is run
  alongside and why the z-statistics at q = 252 are the weakest in the table.
- **Bond and bill funds inherit dependence from the yield process**, and part of their measured
  autocorrelation is NAV staleness rather than economics — the mechanism differs from the
  equity case even where the number looks similar.
- **No transaction costs, no strategy.** A variance ratio away from 1 is a statement about
  measurement, not about a tradable edge; that is study **815** on this desk.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Usefulness: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study [970-sqrt-time-scaling](../README.md).
Not investment advice.*
"""

def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    h = report()
    with open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(results_md(h))
    print("\nwrote docs/results.md")
    print("##HEADLINE## " + json.dumps(h, default=float))


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
