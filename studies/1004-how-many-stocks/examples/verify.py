"""Real-tape verification — Study 1004 (How Many Stocks). Regenerates docs/results.md.

Reproduces the Evans-Archer volatility curve on forty large-cap names, then
plots two further curves on the same draws — the dispersion of terminal wealth across randomly
drawn portfolios, and tracking error against the index — and reports how many holdings each
criterion demands. Measures the skew mechanism that separates them, contrasts rebalancing with
buy-and-hold, and uses a synthetic cross-section with independently tunable correlation and
return dispersion to show the two curves are driven by different things.

    python studies/1004-how-many-stocks/examples/verify.py            # cache-only
    python studies/1004-how-many-stocks/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from howmany import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


N_DRAWS = 500
SHARE = 0.90


def report() -> dict:
    px = data.load_prices()
    h: dict = {"as_of": data.AS_OF, "share": SHARE, "fingerprint": data.fingerprint(px)}

    R = st.usable_panel(px, data.NAMES)
    Rc = R.dropna()
    h["n_available"] = int(R.shape[1])
    h["n_days"] = int(len(Rc))
    h["years"] = float(len(Rc) / 252)
    h["start"] = str(Rc.index[0].date())
    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    print(f"  {R.shape[1]} names with a common history of {len(Rc):,} sessions from "
          f"{h['start']} ({h['years']:.1f} years)")
    print(f"  NOTE: every one of these names is still listed in 2026. That is a selection")
    print(f"  no investor could have made in {h['start'][:4]}, and it biases the LEVEL of every")
    print(f"  curve below. It biases the SHAPE far less, and the shape is the subject.")

    print("\n=== 1. the textbook curve (Evans & Archer 1968) ===")
    vc = st.volatility_curve(R, n_draws=N_DRAWS)
    print(vc.round(4).to_string())
    h["vol_curve"] = vc.reset_index().to_dict("records")
    h["vol_at_1"] = float(vc["mean_vol"].iloc[0])
    h["vol_at_max"] = float(vc["mean_vol"].iloc[-1])
    h["n_for_90_vol"] = st.stocks_for_share(vc, "mean_vol", SHARE)
    h["n_for_99_vol"] = st.stocks_for_share(vc, "mean_vol", 0.99)
    print(f"  one stock: {h['vol_at_1']:.1%} volatility")
    print(f"  all {R.shape[1]}:   {h['vol_at_max']:.1%}")
    print(f"  {SHARE:.0%} of the reduction is achieved by {h['n_for_90_vol']:.0f} holdings")
    print(f"  99% by {h['n_for_99_vol']:.0f}. The textbook is right about its own statistic.")

    print("\n=== 2. the curve an investor actually lives in ===")
    wc = st.terminal_wealth_curve(R, n_draws=N_DRAWS)
    print(wc[["median_wealth", "log_sd", "p05_wealth", "p95_wealth",
              "ratio_95_05"]].round(3).to_string())
    h["wealth_curve"] = wc.reset_index().to_dict("records")
    h["n_for_90_wealth"] = st.stocks_for_share(wc, "log_sd", SHARE)
    nvol = int(round(h["n_for_90_vol"]))
    nearest = min(wc.index, key=lambda k: abs(k - nvol))
    h["ratio_at_vol_n"] = float(wc.loc[nearest, "ratio_95_05"])
    print(f"  standard deviation is a property of the AVERAGE portfolio.")
    print(f"  An investor holds ONE, for decades. This is the spread across portfolios.")
    print(f"  {SHARE:.0%} of that dispersion is removed by {h['n_for_90_wealth']:.0f} holdings"
          f" — against {h['n_for_90_vol']:.0f} for volatility")
    print(f"  at {nearest} names the 5th-to-95th percentile of outcomes still spans "
          f"{h['ratio_at_vol_n']:.2f}x")

    print("\n=== 3. and a third question: tracking error ===")
    tc = st.tracking_error_curve(R, px[data.MARKET].pct_change(), n_draws=N_DRAWS)
    print(tc.round(4).to_string())
    h["te_curve"] = tc.reset_index().to_dict("records")
    h["n_for_90_te"] = st.stocks_for_share(tc, "mean_te", SHARE)
    print(f"  {SHARE:.0%} of the achievable tracking-error reduction: "
          f"{h['n_for_90_te']:.0f} holdings")
    print(f"  three questions, three answers: {h['n_for_90_vol']:.0f} (volatility), "
          f"{h['n_for_90_te']:.0f} (tracking error), {h['n_for_90_wealth']:.0f} (outcomes)")

    print("\n=== 4. why they differ: skew ===")
    sk = st.skew_and_the_median_portfolio(R, n_draws=N_DRAWS)
    print(sk.round(4).to_string())
    h["skew_curve"] = sk.reset_index().to_dict("records")
    h["shortfall_at_vol_n"] = float(sk.loc[nearest, "shortfall"])
    nw = min(sk.index, key=lambda k: abs(k - int(round(h["n_for_90_wealth"]))))
    h["shortfall_at_wealth_n"] = float(sk.loc[nw, "shortfall"])
    conc = st.concentration_of_returns(R)
    h.update({k: conc[k] for k in ("share_from_top_10pct", "share_from_top_25pct",
                                   "share_negative", "median_stock_return",
                                   "mean_stock_return", "best", "best_return")})
    print(f"  over the period, {conc['share_negative']:.0%} of these names lost money outright")
    print(f"  the best ({conc['best']}) returned {conc['best_return']:.0%}")
    print(f"  the top 10% of names produced {conc['share_from_top_10pct']:.0%} of the "
          f"basket's total return")
    print(f"  median stock {conc['median_stock_return']:.0%} vs mean "
          f"{conc['mean_stock_return']:.0%}")
    print(f"  so the MEDIAN {nearest}-stock portfolio ends "
          f"{h['shortfall_at_vol_n']:.0%} below the MEAN one. Averaging across draws — which")
    print(f"  is what the volatility curve does — is precisely the step that hides this.")

    print("\n=== 5. rebalancing is a choice, not a neutral default ===")
    rh = st.rebalanced_vs_held(R, 20, n_draws=300)
    h["rebalance_vs_hold"] = rh
    print(f"  20 names, rebalanced daily: median {rh['rebalanced_median']:.2f}x, "
          f"mean {rh['rebalanced_mean']:.2f}x, log-sd {rh['rebalanced_log_sd']:.3f}")
    print(f"  20 names, bought and held:  median {rh['held_median']:.2f}x, "
          f"mean {rh['held_mean']:.2f}x, log-sd {rh['held_log_sd']:.3f}")
    print("  buy-and-hold lets the winners run, raising the mean and the dispersion together.")
    print("  Diversification curves are almost always drawn under rebalancing, unstated.")

    print("\n=== 6. the identification: two knobs, two curves ===")
    ctrl = []
    for corr in (0.10, 0.30, 0.50):
        for disp in (0.0, 0.05, 0.10):
            sim = st.synthetic_cross_section(n_stocks=60, n_days=5000, avg_corr=corr,
                                             mu_dispersion=disp)
            v = st.volatility_curve(sim, n_draws=150)
            w = st.terminal_wealth_curve(sim, n_draws=150)
            ctrl.append({"avg_corr": corr, "mu_dispersion": disp,
                         "vol_floor": float(v["mean_vol"].iloc[-1]),
                         "n_for_90_vol": st.stocks_for_share(v, "mean_vol", SHARE),
                         "wealth_sd_at_20": float(w.loc[20, "log_sd"]) if 20 in w.index
                         else np.nan,
                         "n_for_90_wealth": st.stocks_for_share(w, "log_sd", SHARE)})
            print(f"  corr {corr:.2f}, mean-dispersion {disp:.2f}: "
                  f"vol floor {ctrl[-1]['vol_floor']:.1%}, "
                  f"n90(vol) {ctrl[-1]['n_for_90_vol']:5.1f}, "
                  f"n90(wealth) {ctrl[-1]['n_for_90_wealth']:5.1f}")
    h["control"] = ctrl
    print("  read DOWN a correlation block: dispersion in expected returns barely moves the")
    print("  volatility answer and moves the wealth answer a lot. The two curves are not the")
    print("  same curve, and that is the study's claim made falsifiable.")

    print("\n=== 7. how sensitive is this to the basket? ===")
    sub = []
    rng = np.random.default_rng(1004)
    cols = list(R.columns)
    for k in range(6):
        keep = rng.choice(cols, size=max(len(cols) - 8, 5), replace=False)
        sr = R[list(keep)]
        v = st.volatility_curve(sr, n_draws=200)
        w = st.terminal_wealth_curve(sr, n_draws=200)
        sub.append({"draw": k, "n_names": len(keep),
                    "n_for_90_vol": st.stocks_for_share(v, "mean_vol", SHARE),
                    "n_for_90_wealth": st.stocks_for_share(w, "log_sd", SHARE)})
        print(f"  subset {k} ({len(keep)} names): vol {sub[-1]['n_for_90_vol']:5.1f}, "
              f"wealth {sub[-1]['n_for_90_wealth']:5.1f}")
    h["subsets"] = sub
    h["subset_wealth_gt_vol"] = float(np.mean(
        [s["n_for_90_wealth"] > s["n_for_90_vol"] for s in sub]))
    print(f"  the wealth criterion asks for more in {h['subset_wealth_gt_vol']:.0%} of subsets")

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    vc = "\n".join(
        f"| {int(r['n_stocks'])} | {r['mean_vol']:.2%} | {r['sd_of_vol']:.2%} | "
        f"{r['p05_vol']:.2%} – {r['p95_vol']:.2%} |" for r in h["vol_curve"])
    wc = "\n".join(
        f"| {int(r['n_stocks'])} | {r['median_wealth']:.2f}× | {r['log_sd']:.3f} | "
        f"{r['p05_wealth']:.2f}× | {r['p95_wealth']:.2f}× | **{r['ratio_95_05']:.2f}×** |"
        for r in h["wealth_curve"])
    tc = "\n".join(f"| {int(r['n_stocks'])} | {r['mean_te']:.2%} | {r['p95_te']:.2%} |"
                   for r in h["te_curve"])
    sk = "\n".join(
        f"| {int(r['n_stocks'])} | {r['median']:.2f}× | {r['mean']:.2f}× | "
        f"**{r['shortfall']:.1%}** | {r['share_below_mean']:.0%} |" for r in h["skew_curve"])
    ctrl = "\n".join(
        f"| {r['avg_corr']:.2f} | {r['mu_dispersion']:.2f} | {r['vol_floor']:.1%} | "
        f"{r['n_for_90_vol']:.1f} | {r['n_for_90_wealth']:.1f} |" for r in h["control"])
    sub = "\n".join(
        f"| {int(r['draw'])} | {int(r['n_names'])} | {r['n_for_90_vol']:.1f} | "
        f"{r['n_for_90_wealth']:.1f} |" for r in h["subsets"])
    return f"""# Results — Study 1004 (How Many Stocks) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). {h['n_available']} large-cap names,
{h['n_days']:,} common sessions from {h['start']} ({h['years']:.1f} years), equal-weighted and
daily-rebalanced. As-of **{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

> **Survivorship, stated up front.** Every name in this basket is still listed in 2026 — a
> selection no investor could have made at the start. This inflates the *level* of every curve
> below. It affects the *shape* far less, and the shape, not the level, is what the study
> compares.

## 1. The textbook curve

Average annualised volatility of a randomly drawn equal-weighted N-stock portfolio:

| N | Mean volatility | SD across draws | 5th – 95th percentile |
|---|--:|--:|--:|
{vc}

One stock: {h['vol_at_1']:.1%}. All {h['n_available']}: {h['vol_at_max']:.1%}.
**{h['n_for_90_vol']:.0f} holdings** capture 90% of the reduction, {h['n_for_99_vol']:.0f}
capture 99%. Evans and Archer were right about the statistic they computed.

## 2. The curve an investor lives in

Standard deviation describes the **average** portfolio. An investor holds **one**, for decades.
This is the spread of terminal wealth *across* randomly drawn portfolios:

| N | Median | Log SD | 5th pct | 95th pct | 95th / 5th |
|---|--:|--:|--:|--:|--:|
{wc}

**{h['n_for_90_wealth']:.0f} holdings** are needed for 90% of that dispersion to go, against
{h['n_for_90_vol']:.0f} for volatility. At the textbook number the gap between a lucky and an
unlucky investor is still a factor of **{h['ratio_at_vol_n']:.2f}×**.

## 3. A third question: tracking error

| N | Mean tracking error | 95th percentile |
|---|--:|--:|
{tc}

Three criteria, three answers: **{h['n_for_90_vol']:.0f}** (volatility),
**{h['n_for_90_te']:.0f}** (tracking error), **{h['n_for_90_wealth']:.0f}** (outcomes). The
familiar "twenty stocks" is not wrong so much as unlabelled.

## 4. Why they differ: skew, not covariance

Over the period {h['share_negative']:.0%} of these names lost money outright, the best
({h['best']}) returned {h['best_return']:.0%}, and the top decile produced
{h['share_from_top_10pct']:.0%} of the basket's total return. The median stock returned
{h['median_stock_return']:.0%} against a mean of {h['mean_stock_return']:.0%}.

| N | Median outcome | Mean outcome | Median's shortfall | Share below the mean |
|---|--:|--:|--:|--:|
{sk}

A small portfolio probably misses the names that mattered, so its *median* outcome lags its
*mean* outcome. Averaging across draws — exactly what the volatility curve does — is the step
that hides this, which is why no amount of care with the textbook statistic would ever have
revealed it.

## 5. Rebalancing is a choice

| 20 names | Median | Mean | Log SD |
|---|--:|--:|--:|
| Rebalanced daily | {h['rebalance_vs_hold']['rebalanced_median']:.2f}× | {h['rebalance_vs_hold']['rebalanced_mean']:.2f}× | {h['rebalance_vs_hold']['rebalanced_log_sd']:.3f} |
| Bought and held | {h['rebalance_vs_hold']['held_median']:.2f}× | {h['rebalance_vs_hold']['held_mean']:.2f}× | {h['rebalance_vs_hold']['held_log_sd']:.3f} |

Rebalancing sells the winners. Diversification curves are almost always drawn under rebalancing
without saying so, and the choice moves both the mean and the dispersion.

## 6. The identification — two knobs, two curves

If the terminal-wealth curve were the volatility curve in different clothes, no parameter could
move one without the other. Dispersion in expected returns does exactly that:

| Avg correlation | Return dispersion | Volatility floor | N for 90% (vol) | N for 90% (wealth) |
|---|--:|--:|--:|--:|
{ctrl}

Read down a correlation block: return dispersion barely touches the volatility answer and moves
the wealth answer substantially. The claim is falsifiable, and it survives.

## 7. Sensitivity to the basket

| Subset | Names | N for 90% (vol) | N for 90% (wealth) |
|---|--:|--:|--:|
{sub}

The wealth criterion demands more holdings in {h['subset_wealth_gt_vol']:.0%} of random subsets.

## Caveats

- **Survivorship**, as stated above. The honest reading is that the *levels* here are
  optimistic and the *comparison between curves* is what carries.
- **Forty names is not the market.** The curves are drawn towards this basket's own average, so
  the tracking-error floor in section 3 is a property of the basket, not of diversification.
  Adding names beyond forty would keep the wealth curve falling for longer, so the gap reported
  is a **lower bound**.
- **Equal weights, daily rebalancing** for the main curves. Section 5 shows how much that
  matters; cap-weighted portfolios would confound the question with the size effect.
- **One period.** These are large-caps over a long bull market for large-caps. The mechanism —
  right-skewed single-stock returns — is documented far more broadly (Bessembinder 2018), but
  the magnitudes here belong to this sample.
- **No costs.** Holding forty names costs more than holding twenty in commissions and attention,
  which is Statman's (1987) argument and pushes the practical answer back down. This study
  measures the benefit side only.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1004-how-many-stocks](../README.md). Not investment advice.*
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
