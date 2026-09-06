"""Real-tape verification — Study 973 (Different Holidays). Regenerates docs/results.md.

Measures correlation and beta between a US index and four foreign-market ETFs at
daily through monthly frequencies, applies the Dimson and Scholes-Williams corrections, runs a
same-market control through the identical machinery, and prices the consequence as the
volatility a minimum-variance optimiser promises versus the one it delivers.

    python studies/973-calendar-misalignment/examples/verify.py            # cache-only
    python studies/973-calendar-misalignment/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from calendar_gap import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


def report() -> dict:
    px = data.load_prices()
    h: dict = {"as_of": data.AS_OF, "tickers": list(data.TICKERS),
               "n_foreign": len(data.FOREIGN), "control_asset": data.CONTROL,
               "fingerprint": data.fingerprint(px)}

    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    for tk in data.TICKERS:
        s = px[tk].dropna()
        print(f"  {tk:5s} {s.index[0].date()} -> {s.index[-1].date()}  n={len(s):,}")
    h["windows"] = {tk: [str(px[tk].dropna().index[0].date()),
                         str(px[tk].dropna().index[-1].date())] for tk in data.TICKERS}
    h["n_obs"] = {tk: int(px[tk].dropna().shape[0]) for tk in data.TICKERS}

    common = px.dropna()
    print(f"  common window: {common.index[0].date()} -> {common.index[-1].date()} "
          f"({len(common):,} sessions) — all five are New-York-listed and close at 16:00 ET, "
          f"so the QUOTES are synchronous; only the underlying markets' hours differ")
    h["common_window"] = [str(common.index[0].date()), str(common.index[-1].date())]
    h["n_common"] = int(len(common))

    print("\n=== correlation with SPY, by sampling frequency ===")
    print("  asset   " + "  ".join(f"{f:>10s}" for f in st.FREQS) + "      lift")
    tbl = st.bias_table(common, data.US, list(data.FOREIGN) + [data.CONTROL])
    for a, row in tbl.iterrows():
        cells = "  ".join(f"{row.get('corr_' + f, float('nan')):10.3f}" for f in st.FREQS)
        tag = "  <- same-market control" if a == data.CONTROL else ""
        print(f"  {a:6s} {cells} {row['corr_lift']:+9.3f}{tag}")
    h["bias"] = {a: {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
                 for a, row in tbl.to_dict("index").items()}
    foreign = tbl.drop(index=[data.CONTROL])
    h["n_big_lifts"] = int((foreign["corr_lift"] > 0.10).sum())
    h["control_lift"] = float(tbl.loc[data.CONTROL, "corr_lift"])
    h["worst_lift_asset"] = str(foreign["corr_lift"].idxmax())
    h["worst_lift"] = float(foreign["corr_lift"].max())
    h["worst_daily_corr"] = float(foreign.loc[h["worst_lift_asset"], "corr_daily"])
    h["worst_monthly_corr"] = float(foreign.loc[h["worst_lift_asset"], "corr_monthly"])

    print("\n=== beta on SPY: naive, Dimson (1979), Scholes-Williams (1977) ===")
    print("  asset    naive    Dimson      S-W     lead coef    lag coef")
    for a, row in tbl.iterrows():
        tag = "  <- control" if a == data.CONTROL else ""
        print(f"  {a:6s} {row['beta_naive']:8.3f} {row['beta_dimson']:9.3f} "
              f"{row['beta_sw']:8.3f} {row['lead_coef']:+12.3f} {row['lag_coef']:+11.3f}{tag}")
    h["worst_beta_naive"] = float(tbl.loc[h["worst_lift_asset"], "beta_naive"])
    h["worst_beta_dimson"] = float(tbl.loc[h["worst_lift_asset"], "beta_dimson"])
    h["worst_lag_coef"] = float(tbl.loc[h["worst_lift_asset"], "lag_coef"])

    print("\n=== where the shared information sits (lead-lag correlation profile vs SPY) ===")
    rets = st.to_returns(common)
    profiles = {}
    for a in list(data.FOREIGN) + [data.CONTROL]:
        prof = st.lead_lag_profile(rets, data.US, a, max_lag=2)
        profiles[a] = {int(k): float(v) for k, v in prof.items()}
        print(f"  {a:6s} " + "  ".join(f"k={k:+d}: {v:+.3f}" for k, v in prof.items()))
    h["lead_lag"] = profiles
    print("  (a peak at k = +1 means the foreign tape is still absorbing yesterday's US session)")

    print("\n=== the portfolio consequence ===")
    assets = [data.US] + list(data.FOREIGN)
    imp = st.portfolio_impact(common, assets, step_estimate=1, step_truth=21)
    print(f"  minimum-variance book on the five markets, weights from the DAILY covariance:")
    print(f"    promised volatility  : {imp['promised_vol']:.2%}/yr")
    print(f"    delivered at monthly : {imp['delivered_vol']:.2%}/yr  "
          f"({imp['understatement']:+.1%})")
    print(f"    best achievable      : {imp['best_possible_vol']:.2%}/yr  "
          f"(cost of the biased matrix {imp['cost_of_bad_matrix']:+.1%})")
    print(f"    largest weight difference vs the unbiased matrix: {imp['max_weight_gap']:.1%}")
    for a in assets:
        print(f"      {a:5s} daily-matrix weight {imp['weights_estimated'][a]:+7.1%}   "
              f"monthly-matrix weight {imp['weights_truth'][a]:+7.1%}")
    h.update({k: imp[k] for k in ("promised_vol", "delivered_vol", "best_possible_vol",
                                  "understatement", "cost_of_bad_matrix", "max_weight_gap")})
    h["weights"] = {"daily": imp["weights_estimated"], "monthly": imp["weights_truth"]}

    print("\n=== the same machinery on a same-market pair (the null) ===")
    ctrl = st.portfolio_impact(common, [data.US, data.CONTROL], step_estimate=1, step_truth=21)
    print(f"  SPY + {data.CONTROL}: promised {ctrl['promised_vol']:.2%}, delivered "
          f"{ctrl['delivered_vol']:.2%} ({ctrl['understatement']:+.1%})")
    h["control_understatement"] = float(ctrl["understatement"])

    print("\n=== era check: has global trading become more synchronous? ===")
    eras = {}
    for tag, sl in (("pre-2010", common.loc[:"2009-12-31"]),
                    ("2010+", common.loc["2010-01-01":])):
        if len(sl) < 500:
            continue
        t = st.bias_table(sl, data.US, list(data.FOREIGN))
        eras[tag] = {a: float(t.loc[a, "corr_lift"]) for a in data.FOREIGN}
        print(f"  {tag:9s} " + "  ".join(f"{a} {t.loc[a, 'corr_lift']:+.3f}"
                                         for a in data.FOREIGN))
    h["eras"] = eras

    print("\n=== synthetic control: a known correlation with a planted delay ===")
    rng = np.random.default_rng(973)
    for delay, tag in ((0.0, "synchronous"), (0.5, "half a day stale")):
        n = 8000
        f = rng.normal(0, 0.01, n)
        e1, e2 = rng.normal(0, 0.01, n), rng.normal(0, 0.01, n)
        b = (1 - delay) * f + delay * np.concatenate([[0.0], f[:-1]]) + e2
        idx = pd.bdate_range("1995-01-02", periods=n)
        sim = pd.DataFrame({"A": 100 * np.cumprod(1 + f + e1),
                            "B": 100 * np.cumprod(1 + b)}, index=idx)
        agg = st.aggregated_correlation(sim, "A", "B")
        r = st.to_returns(sim)
        print(f"  {tag:18s} daily corr {agg.loc['daily', 'correlation']:.3f} -> monthly "
              f"{agg.loc['monthly', 'correlation']:.3f};  beta naive "
              f"{st.ols_beta(r['B'], r['A']):.3f} -> Dimson "
              f"{st.dimson_beta(r['B'], r['A'])['beta']:.3f}")
        h[f"synthetic_{'stale' if delay else 'sync'}"] = {
            "corr_daily": float(agg.loc["daily", "correlation"]),
            "corr_monthly": float(agg.loc["monthly", "correlation"]),
            "beta_naive": float(st.ols_beta(r["B"], r["A"])),
            "beta_dimson": float(st.dimson_beta(r["B"], r["A"])["beta"])}

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Usefulness: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    freqs = list(st.FREQS)
    head = " | ".join(f.capitalize() for f in freqs)
    dash = "|".join(["--:"] * len(freqs))
    corr = "\n".join(
        "| " + a + (" *(control)*" if a == h["control_asset"] else "") + " | " +
        " | ".join(f"{h['bias'][a].get('corr_' + f) or float('nan'):.3f}" for f in freqs) +
        f" | **{h['bias'][a]['corr_lift']:+.3f}** |" for a in h["bias"])
    betas = "\n".join(
        f"| {a}{' *(control)*' if a == h['control_asset'] else ''} | {r['beta_naive']:.3f} | "
        f"{r['beta_dimson']:.3f} | {r['beta_sw']:.3f} | {r['lead_coef']:+.3f} | "
        f"{r['lag_coef']:+.3f} |" for a, r in h["bias"].items())
    ll = "\n".join("| " + a + " | " + " | ".join(f"{h['lead_lag'][a][str(k)] if str(k) in h['lead_lag'][a] else h['lead_lag'][a][k]:+.3f}"
                                                 for k in (-2, -1, 0, 1, 2)) + " |"
                   for a in h["lead_lag"])
    w = "\n".join(f"| {a} | {h['weights']['daily'][a]:+.1%} | {h['weights']['monthly'][a]:+.1%} |"
                  for a in h["weights"]["daily"])
    eras = "\n".join(f"| {tag} | " + " | ".join(f"{v2:+.3f}" for v2 in d.values()) + " |"
                     for tag, d in h["eras"].items())
    era_head = " | ".join(list(next(iter(h["eras"].values())).keys())) if h["eras"] else ""
    return f"""# Results — Study 973 (Different Holidays) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). Five **New-York-listed** ETFs —
SPY, IWM and four single-country funds — on a common window of {h['n_common']:,} sessions
({h['common_window'][0]} → {h['common_window'][1]}). Because every ticker quotes on the same
exchange and closes at the same minute, there is no quote-time misalignment at all: what
remains is the mismatch in the hours the **underlying** markets were open. As-of
**{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

## Correlation with SPY, by sampling frequency

| Asset | {head} | Lift (monthly − daily) |
|---|{dash}|--:|
{corr}

**{h['n_big_lifts']} of {h['n_foreign']}** foreign tapes gain more than 0.10 of correlation
simply by being measured monthly instead of daily. The same-market control moves
**{h['control_lift']:+.3f}**, which is this machinery's noise floor — the effect is not an
artefact of changing frequency.

## Beta on SPY, three ways

| Asset | Naive | Dimson (1979) | Scholes-Williams (1977) | Lead coefficient | Lag coefficient |
|---|--:|--:|--:|--:|--:|
{betas}

The **lag coefficient** is the mechanism made visible: yesterday's US session still explains
part of today's move in a market that was closed for most of it.

## Where the shared information sits

corr(SPY at *t*, asset at *t+k*):

| Asset | k = −2 | k = −1 | k = 0 | k = +1 | k = +2 |
|---|--:|--:|--:|--:|--:|
{ll}

## The portfolio consequence

Minimum-variance book across the five markets, weights taken from the **daily** covariance
matrix and priced against the **monthly** one:

| | |
|---|--:|
| Volatility promised in-sample | {h['promised_vol']:.2%}/yr |
| Volatility delivered at the monthly horizon | **{h['delivered_vol']:.2%}/yr** |
| Understatement | **{h['understatement']:+.1%}** |
| Best achievable with the unbiased matrix | {h['best_possible_vol']:.2%}/yr |
| Cost of using the biased matrix | {h['cost_of_bad_matrix']:+.1%} |

| Asset | Weight from the daily matrix | Weight from the monthly matrix |
|---|--:|--:|
{w}

The same calculation on the same-market pair (SPY + {h['control_asset']}) understates by
{h['control_understatement']:+.1%}.

## Has global trading become more synchronous?

| Era | {era_head} |
|---|{"|".join(["--:"] * max(len(h["eras"].get(next(iter(h["eras"]), ""), {})), 1))}|
{eras}

## Synthetic control

A pair with a known common factor: synchronous, daily correlation
{h['synthetic_sync']['corr_daily']:.3f} → monthly {h['synthetic_sync']['corr_monthly']:.3f},
beta {h['synthetic_sync']['beta_naive']:.3f} → Dimson
{h['synthetic_sync']['beta_dimson']:.3f}. With half the factor arriving a day late: daily
{h['synthetic_stale']['corr_daily']:.3f} → monthly
{h['synthetic_stale']['corr_monthly']:.3f}, beta
{h['synthetic_stale']['beta_naive']:.3f} → Dimson
{h['synthetic_stale']['beta_dimson']:.3f}. The apparatus finds a planted delay and leaves a
synchronous pair alone.

## Caveats

- **Lower-frequency estimates are noisier.** A monthly correlation over this sample rests on a
  few hundred observations against several thousand daily ones; the lift is large relative to
  that noise, but the standard errors are not zero and the control is the check on it.
- **Part of the "lift" is not timing.** Correlations genuinely rise at longer horizons for
  economic reasons (transitory noise averages out), which is exactly why the same-market
  control is run: whatever it shows is the non-timing part.
- **No currency decomposition.** A US-listed foreign ETF bundles the local market with the
  exchange rate; separating them is study **613**'s territory.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Usefulness: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[973-calendar-misalignment](../README.md). Not investment advice.*
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
