"""Real-tape verification — Study 967 (Window Shopping). Regenerates docs/results.md.

Estimates beta, the mean return and the covariance matrix at every year-end on the
eleven sector SPDRs from rolling windows of 1/2/3/5/10 years and from an expanding window,
scores each against what the next year delivered, and tests the differences with a HAC
Diebold-Mariano. Adds two humbling benchmarks: the cross-sectional grand mean, and Blume
shrinkage.

    python studies/967-rolling-vs-expanding/examples/verify.py            # cache-only
    python studies/967-rolling-vs-expanding/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from window_choice import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


WINDOWS = st.WINDOWS_YEARS


def report() -> dict:
    px = data.load_prices()
    rets = st.to_returns(px)
    sectors = tuple(s for s in data.SECTORS if rets[s].notna().sum() > 500)
    h: dict = {"as_of": data.AS_OF, "tickers": list(data.TICKERS),
               "sectors": list(sectors), "n_sectors": len(sectors),
               "windows": [str(w) for w in WINDOWS]}

    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    for s in data.SECTORS:
        col = px[s].dropna()
        print(f"  {s:5s} {col.index[0].date()} -> {col.index[-1].date()}  n={len(col):,}")
    h["fingerprint"] = data.fingerprint(px)
    h["inceptions"] = {s: str(px[s].dropna().index[0].date()) for s in data.SECTORS}
    h["obs_per_param_1y"] = float(st.TRADING_DAYS /
                                  (len(sectors) * (len(sectors) + 1) / 2) * len(sectors))

    # --------------------------------------------------------------------- beta
    print("\n=== 1. beta: estimate at each year-end, score against next year's realised beta ===")
    bexp = st.beta_experiment(rets, sectors, data.MARKET)
    bsc = st.score(bexp)
    for w, row in bsc.iterrows():
        print(f"  window {str(w):>9s}: MSE {row['mse']:.5f}  MAE {row['mae']:.4f}  "
              f"bias {row['bias']:+.4f}  n={int(row['n'])}")
    best_b = st.best_window(bsc)
    h["beta_scores"] = {str(k): dict(v) for k, v in bsc.to_dict("index").items()}
    h["best_beta"] = str(best_b)
    h["spread_beta"] = float(bsc["mse"].max() / bsc["mse"].min() - 1)
    print(f"  best window: {best_b}   worst/best MSE spread: {h['spread_beta']:.0%}")

    bexp["shrunk_error"] = bexp["estimate"].map(st.blume_shrunk) - bexp["realised"]
    raw_mse = float((bexp[bexp["window"] == best_b]["error"] ** 2).mean())
    shr_mse = float((bexp[bexp["window"] == best_b]["shrunk_error"] ** 2).mean())
    h["blume_gain"] = float(1 - shr_mse / raw_mse)
    print(f"  Blume shrinkage toward 1.0 on the best window: MSE {raw_mse:.5f} -> {shr_mse:.5f} "
          f"({h['blume_gain']:+.0%})")
    dm_b = st.pairwise_dm(bexp, 1, st.EXPANDING)
    print(f"  DM(1 year vs expanding): {dm_b['dm']:+.2f} (p {dm_b['p_value']:.3f}, "
          f"n={dm_b['n']}) — positive means the 1-year window is worse")
    h["dm_beta_1_vs_exp"] = float(dm_b["dm"])

    # --------------------------------------------------------------------- mean
    print("\n=== 2. mean return: the parameter nobody can estimate ===")
    mexp = st.mean_experiment(rets, sectors)
    gm = st.grand_mean_benchmark(rets, sectors)
    both = pd.concat([mexp, gm], ignore_index=True)
    msc = st.score(both)
    for w, row in msc.iterrows():
        ann = row["bias"] * st.TRADING_DAYS
        print(f"  window {str(w):>11s}: MSE {row['mse']:.3e}  bias {ann:+.2%}/yr  "
              f"n={int(row['n'])}")
    best_m = st.best_window(msc.drop(index=["grand mean"]))
    h["mean_scores"] = {str(k): dict(v) for k, v in msc.to_dict("index").items()}
    h["best_mean"] = str(best_m)
    h["spread_mean"] = float(msc.drop(index=["grand mean"])["mse"].max() /
                             msc.drop(index=["grand mean"])["mse"].min() - 1)
    h["grand_mean_ratio"] = float(msc.loc["grand mean", "mse"] / msc.loc[best_m, "mse"])
    print(f"  best window of a sector's OWN history: {best_m}")
    print(f"  the cross-sectional grand mean scores {h['grand_mean_ratio']:.2f}x that MSE "
          f"({'better' if h['grand_mean_ratio'] < 1 else 'worse'})")
    dm_m = st.pairwise_dm(both, "grand mean", best_m)
    print(f"  DM(grand mean vs best own-history window): {dm_m['dm']:+.2f} "
          f"(p {dm_m['p_value']:.3f})")
    h["dm_mean_grand_vs_best"] = float(dm_m["dm"])

    # --------------------------------------------------------------- covariance
    print("\n=== 3. covariance: judged by the portfolio it builds (minimum variance) ===")
    cexp = st.covariance_experiment(rets, sectors)
    csc = cexp.groupby("window").agg(
        realised_vol=("realised_vol", "mean"), predicted_vol=("predicted_vol", "mean"),
        max_weight=("max_weight", "mean"), short_weight=("short_weight", "mean"),
        turnover=("turnover", "mean"), n=("realised_vol", "size"))
    order = [w for w in list(WINDOWS) + [st.EXPANDING] if w in csc.index]
    csc = csc.loc[order]
    for w, row in csc.iterrows():
        print(f"  window {str(w):>9s}: realised vol {row['realised_vol']:6.2%}  "
              f"predicted {row['predicted_vol']:6.2%}  optimism "
              f"{row['predicted_vol'] / row['realised_vol'] - 1:+6.1%}  "
              f"max weight {row['max_weight']:5.1%}  shorts {row['short_weight']:5.1%}  "
              f"turnover {row['turnover']:.2f}")
    best_c = csc["realised_vol"].idxmin()
    h["cov_scores"] = {str(k): dict(v) for k, v in csc.to_dict("index").items()}
    h["best_cov"] = str(best_c)
    h["spread_cov"] = float(csc["realised_vol"].max() / csc["realised_vol"].min() - 1)
    print(f"  lowest realised portfolio vol: window {best_c}  "
          f"(spread across windows {h['spread_cov']:.0%})")

    cexp["err"] = cexp["realised_vol"]
    la = cexp[cexp["window"] == 1].set_index("date")["realised_vol"]
    lb = cexp[cexp["window"] == best_c].set_index("date")["realised_vol"]
    la, lb = la.align(lb, join="inner")
    dm_c = st.diebold_mariano(la.reset_index(drop=True), lb.reset_index(drop=True))
    print(f"  DM(1-year vs best) on realised vol: {dm_c['dm']:+.2f} (p {dm_c['p_value']:.3f})")
    h["dm_cov_1_vs_best"] = float(dm_c["dm"])
    h["max_abs_dm"] = float(max(abs(h["dm_beta_1_vs_exp"]), abs(h["dm_mean_grand_vs_best"]),
                                abs(h["dm_cov_1_vs_best"])))

    print("\n=== the arithmetic behind the covariance result ===")
    n = len(sectors)
    print(f"  {n} sectors -> {n * (n + 1) // 2} parameters in the covariance matrix")
    for w in WINDOWS:
        print(f"    {w:2d}-year window = {w * st.TRADING_DAYS:5d} rows -> "
              f"{w * st.TRADING_DAYS / (n * (n + 1) / 2):5.1f} observations per parameter")
    h["n_params"] = int(n * (n + 1) // 2)

    print("\n=== per-sector beta stability (context for the beta result) ===")
    for s in sectors:
        sub = bexp[(bexp["sector"] == s) & (bexp["window"] == 5)]
        if len(sub) > 3:
            print(f"  {s:5s} realised beta {sub['realised'].min():.2f} -> "
                  f"{sub['realised'].max():.2f} (sd {sub['realised'].std():.2f}), "
                  f"5-year estimate error sd {sub['error'].std():.3f}")
    h["beta_dispersion"] = {s: float(bexp[(bexp["sector"] == s) &
                                          (bexp["window"] == 5)]["realised"].std())
                            for s in sectors}

    print("\n=== synthetic control: stationary world (expanding must win the mean) ===")
    for ss, tag in ((1.0, "planted, stationary"), (0.0, "pure noise")):
        p, _, _ = data.synthetic_panel(n_assets=8, n_years=25, signal_strength=ss, seed=967)
        r = st.to_returns(p)
        sc = st.score(st.mean_experiment(r, tuple(p.columns), windows=(1, 3)))
        print(f"  {tag:20s}: MSE 1y {sc.loc[1, 'mse']:.3e}  3y {sc.loc[3, 'mse']:.3e}  "
              f"expanding {sc.loc[st.EXPANDING, 'mse']:.3e}")
        h[f"synthetic_{'planted' if ss else 'noise'}_exp_beats_1y"] = bool(
            sc.loc[st.EXPANDING, "mse"] < sc.loc[1, "mse"])

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Usefulness: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    b = "\n".join(f"| {w} | {r['mse']:.5f} | {r['mae']:.4f} | {r['bias']:+.4f} | {int(r['n'])} |"
                  for w, r in h["beta_scores"].items())
    m = "\n".join(f"| {w} | {r['mse']:.3e} | {r['bias'] * 252:+.2%} | {int(r['n'])} |"
                  for w, r in h["mean_scores"].items())
    c = "\n".join(
        f"| {w} | {r['realised_vol']:.2%} | {r['predicted_vol']:.2%} | "
        f"{r['predicted_vol'] / r['realised_vol'] - 1:+.1%} | {r['max_weight']:.1%} | "
        f"{r['short_weight']:.1%} | {r['turnover']:.2f} |"
        for w, r in h["cov_scores"].items())
    inc = ", ".join(f"{s} {d}" for s, d in h["inceptions"].items())
    ratio = "\n".join(
        f"| {w}-year | {w * 252:,} | {w * 252 / h['n_params']:.1f} |" for w in (1, 2, 3, 5, 10))
    return f"""# Results — Study 967 (Window Shopping) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). Daily total-return closes for the
eleven Select Sector SPDRs plus SPY and BIL. At every year-end, each parameter is estimated
from a rolling window of 1 / 2 / 3 / 5 / 10 years and from an expanding window, then scored
against what the **following** year delivered. As-of **{h['as_of']}**; panel fingerprint
`{h['fingerprint']}`.*

Sector inceptions (the panel is deliberately ragged, never back-filled): {inc}.

## 1. Beta — estimated at year-end, scored on next year's realised beta

| Window | MSE | MAE | Bias | n |
|---|--:|--:|--:|--:|
{b}

Best window: **{h['best_beta']}**; worst-to-best MSE spread **{h['spread_beta']:.0%}**.
Diebold-Mariano, 1-year versus expanding: **{h['dm_beta_1_vs_exp']:+.2f}** (positive = the
1-year window is worse). Applying **Blume (1971) shrinkage** toward 1.0 on top of the best
window changes its MSE by **{h['blume_gain']:+.0%}** — beta is mean-reverting, and the classic
fix still earns its keep.

## 2. Mean return — the parameter that will not be estimated

| Window | MSE | Bias (annualised) | n |
|---|--:|--:|--:|
{m}

Best window of a sector's **own** history: **{h['best_mean']}**. The **grand mean across all
sectors** — the same number for every sector, using none of its own history — scores
**{h['grand_mean_ratio']:.2f}×** that MSE (Diebold-Mariano
{h['dm_mean_grand_vs_best']:+.2f}). When an estimator that ignores an asset's own returns
beats every window of them, the honest conclusion is not "use a longer window" but "this
parameter is not estimable from price history".

## 3. Covariance — judged by the portfolio it builds

Minimum-variance weights rebuilt each year-end and held for the year:

| Window | Realised vol | Predicted vol | Optimism | Mean max weight | Shorts | Turnover |
|---|--:|--:|--:|--:|--:|--:|
{c}

Lowest realised volatility: window **{h['best_cov']}**; spread across windows
**{h['spread_cov']:.0%}**; Diebold-Mariano 1-year versus best
**{h['dm_cov_1_vs_best']:+.2f}**.

The mechanism is arithmetic, not finance — {h['n_sectors']} assets means
**{h['n_params']} covariance parameters**:

| Window | Rows | Observations per parameter |
|---|--:|--:|
{ratio}

*Optimism* is the gap between the volatility the optimiser promised in-sample and the one the
portfolio actually ran. It is largest exactly where the observations-per-parameter count is
smallest, which is Michaud's error-maximisation in one column.

## Caveats

- **Eleven liquid sector ETFs** is a benign cross-section: no delistings, no illiquidity, and
  a covariance matrix small enough to invert. At 500 assets every conclusion here gets sharper
  and the case for shrinkage (study **975**) becomes overwhelming.
- **Annual re-estimation** is one choice among many; a monthly clock would give more
  observations and more overlap, and the Diebold-Mariano lags would have to grow with it.
- **Realised next-year beta** is itself an estimate, not a truth: part of every MSE here is
  the noise in the target, which inflates all windows equally and understates the differences
  between them.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Usefulness: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[967-rolling-vs-expanding](../README.md). Not investment advice.*
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
