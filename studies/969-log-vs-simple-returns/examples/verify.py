"""Real-tape verification — Study 969 (Log or Simple). Regenerates docs/results.md.

Measures the arithmetic-versus-geometric gap on eight tapes spanning 0.3% to 65%
annualised volatility, checks the sigma-squared-over-two prediction against what the tapes
actually do, prices the log-weighted-portfolio mistake in terminal wealth, and shows which
statistics move (Sharpe) and which do not (beta).

    python studies/969-log-vs-simple-returns/examples/verify.py            # cache-only
    python studies/969-log-vs-simple-returns/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from log_vs_simple import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


def report() -> dict:
    px = data.load_prices()
    h: dict = {"as_of": data.AS_OF, "tickers": list(data.TICKERS),
               "fingerprint": data.fingerprint(px)}

    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    for tk in data.TICKERS:
        s = px[tk].dropna()
        print(f"  {tk:8s} {s.index[0].date()} -> {s.index[-1].date()}  n={len(s):,}")
    h["windows"] = {tk: [str(px[tk].dropna().index[0].date()),
                         str(px[tk].dropna().index[-1].date())] for tk in data.TICKERS}
    h["n_obs"] = {tk: int(px[tk].dropna().shape[0]) for tk in data.TICKERS}

    print("\n=== 1. the gap between the two means, and what explains it ===")
    dt = st.drag_table(px)
    dt = dt.sort_values("vol_ann")
    print("  tkr        vol    mean simple    mean log        gap    sigma^2/2   residual    "
          "CAGR   skew   kurt")
    for tk, row in dt.iterrows():
        print(f"  {tk:8s} {row['vol_ann']:6.1%} {row['mean_simple_ann']:+13.2%} "
              f"{row['mean_log_ann']:+11.2%} {row['gap_ann']:+10.2%} {row['half_var_ann']:+11.2%} "
              f"{row['residual_ann']:+10.2%} {row['cagr']:+7.1%} {row['skew']:+6.2f} "
              f"{row['excess_kurtosis']:6.1f}")
    h["drag"] = {tk: dict(v) for tk, v in dt.to_dict("index").items()}
    h["max_gap_ann"] = float(dt["gap_ann"].max())
    h["calm_ticker"] = str(dt.index[0])
    h["calm_vol"] = float(dt["vol_ann"].iloc[0])
    h["calm_gap"] = float(dt["gap_ann"].iloc[0])
    h["wild_ticker"] = str(dt["gap_ann"].idxmax())
    h["wild_vol"] = float(dt.loc[h["wild_ticker"], "vol_ann"])
    h["half_var_explains"] = float(dt.loc[h["wild_ticker"], "half_var_ann"] /
                                   dt.loc[h["wild_ticker"], "gap_ann"])
    print(f"  the sigma^2/2 rule explains {h['half_var_explains']:.0%} of the gap on "
          f"{h['wild_ticker']}; the rest is skew and kurtosis (the third and fourth terms of "
          f"the expansion, which nobody quotes)")

    print("\n=== the CAGR every convention would have you report (SPY and the wildest tape) ===")
    ann = {}
    for tk in ("SPY", h["wild_ticker"]):
        R = px[tk].dropna().pct_change().dropna()
        t = st.annualisation_table(R)
        ann[tk] = {m: float(v) for m, v in t["value"].items()}
        print(f"  {tk}:")
        for m, row in t.iterrows():
            print(f"    {m:32s} {row['value']:+9.2%}   ({row['claims']})")
    h["annualisation"] = ann

    print("\n=== 2. the portfolio mistake: weighting LOG returns ===")
    universe = tuple(tk for tk in data.TICKERS if tk not in ("BIL",))
    err = st.portfolio_error(px, universe)
    bonus = st.rebalancing_bonus(px, universe)
    print(f"  equal-weight daily-rebalanced book of {len(universe)} tapes, "
          f"{err['years']:.1f} years of overlap")
    print(f"    correct (weight simple returns): CAGR {err['cagr_correct']:+.2%}")
    print(f"    wrong   (weight log returns)   : CAGR {err['cagr_wrong']:+.2%}")
    print(f"    gap {err['cagr_gap']:+.2%}/yr -> {err['terminal_ratio']:.2f}x of terminal "
          f"wealth; understates on essentially every day: {err['always_understates']}")
    print(f"    the gap IS the diversification return: rebalanced {bonus['rebalanced_cagr']:+.2%} "
          f"minus the weighted GEOMETRIC average of the holdings "
          f"{bonus['buy_hold_geometric_cagr']:+.2%} = {bonus['bonus']:+.2%}/yr")
    print(f"    (against the *arithmetic* average of the holdings it would be "
          f"{bonus['bonus_vs_arithmetic']:+.2%}/yr — a different quantity, routinely confused "
          f"with this one)")
    h["portfolio"] = err
    h["rebalancing_bonus"] = bonus["bonus"]
    h["portfolio_n"] = len(universe)
    h["portfolio_cagr_gap"] = err["cagr_gap"]
    h["portfolio_terminal_ratio"] = err["terminal_ratio"]
    h["portfolio_years"] = err["years"]
    h["understates_always"] = bool(err["always_understates"])

    print("\n  the same mistake on a two-asset book, by volatility of the second leg:")
    pairs = {}
    for tk in ("TLT", "GLD", "QQQ", "TQQQ", "BTC-USD"):
        e = st.portfolio_error(px, ("SPY", tk))
        pairs[tk] = {"cagr_gap": e["cagr_gap"], "years": e["years"]}
        print(f"    SPY + {tk:8s}: {e['cagr_gap']:+.2%}/yr over {e['years']:.1f} years")
    h["pairs"] = pairs

    print("\n=== 3. which statistics move, and which do not ===")
    sg = st.sharpe_gap(px).sort_values("vol_ann")
    for tk, row in sg.iterrows():
        print(f"  {tk:8s} vol {row['vol_ann']:6.1%}  Sharpe simple {row['sharpe_simple']:+.3f}  "
              f"log {row['sharpe_log']:+.3f}  gap {row['gap']:+.3f} "
              f"({row['relative_gap']:+.1%} of the simple value)")
    h["sharpe"] = {tk: dict(v) for tk, v in sg.to_dict("index").items()}
    h["max_sharpe_gap"] = float(sg["gap"].abs().max())

    print("\n  beta (the case where it barely matters):")
    betas = {}
    for tk in ("QQQ", "EEM", "TLT", "GLD", "TQQQ"):
        b = st.beta_gap(px, tk, "SPY")
        betas[tk] = b
        print(f"    {tk:8s} beta on SPY: simple {b['beta_simple']:.3f}  log {b['beta_log']:.3f}  "
              f"({b['relative_gap']:+.2%})")
    h["betas"] = betas

    print("\n=== the shape of the whole thing: drag as a function of volatility ===")
    curve = st.drag_curve()
    for v in (0.10, 0.20, 0.40, 0.80, 1.20):
        row = curve.loc[curve.index[np.argmin(np.abs(curve.index - v))]]
        print(f"  vol {row.name:5.0%}: arithmetic {row['arithmetic']:+.1%}/yr -> "
              f"geometric {row['geometric_approx']:+.1%}/yr "
              f"(drag {row['arithmetic'] - row['geometric_approx']:.1%})")
    h["break_even_vol"] = float(np.sqrt(2 * 0.08))
    print(f"  an 8%/yr arithmetic return is entirely eaten at "
          f"{h['break_even_vol']:.0%} annualised volatility")

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Usefulness: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    drag = "\n".join(
        f"| {tk} | {r['vol_ann']:.1%} | {r['mean_simple_ann']:+.2%} | {r['mean_log_ann']:+.2%} | "
        f"**{r['gap_ann']:+.2%}** | {r['half_var_ann']:+.2%} | {r['residual_ann']:+.2%} | "
        f"{r['cagr']:+.1%} |"
        for tk, r in sorted(h["drag"].items(), key=lambda kv: kv[1]["vol_ann"]))
    sharpe = "\n".join(
        f"| {tk} | {r['vol_ann']:.1%} | {r['sharpe_simple']:+.3f} | {r['sharpe_log']:+.3f} | "
        f"{r['gap']:+.3f} |"
        for tk, r in sorted(h["sharpe"].items(), key=lambda kv: kv[1]["vol_ann"]))
    betas = "\n".join(
        f"| {tk} | {b['beta_simple']:.3f} | {b['beta_log']:.3f} | {b['relative_gap']:+.2%} |"
        for tk, b in h["betas"].items())
    pairs = "\n".join(f"| SPY + {tk} | {p['cagr_gap']:+.2%} | {p['years']:.1f} |"
                      for tk, p in h["pairs"].items())
    ann = "\n".join(
        f"| {m} | {vals['SPY']:+.2%} | {vals_wild:+.2%} |"
        for m, vals, vals_wild in [
            (m, {"SPY": h["annualisation"]["SPY"][m]}, h["annualisation"][h["wild_ticker"]][m])
            for m in h["annualisation"]["SPY"]])
    win = "\n".join(f"| {tk} | {w[0]} → {w[1]} | {h['n_obs'][tk]:,} |"
                    for tk, w in h["windows"].items())
    return f"""# Results — Study 969 (Log or Simple) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). Daily total-return closes for
eight tapes spanning **{h['calm_vol']:.1%} to {h['wild_vol']:.0%}** annualised volatility.
Nothing here is an estimate with a confidence interval — it is arithmetic — so the study
reports magnitudes, not *t*-statistics. As-of **{h['as_of']}**; fingerprint
`{h['fingerprint']}`.*

## Data stamp

| Ticker | Window | Sessions |
|---|---|--:|
{win}

## 1. The gap between the two means

| Ticker | Vol | Mean simple | Mean log | Gap | σ²/2 | Residual | Realised CAGR |
|---|--:|--:|--:|--:|--:|--:|--:|
{drag}

The gap is **half the variance**, to a first approximation — which is why it is invisible on
bills and enormous on bitcoin. On {h['wild_ticker']} the σ²/2 term explains
**{h['half_var_explains']:.0%}** of the observed gap; the remainder is the third and fourth
terms of the expansion (skewness and kurtosis), which the textbook formula drops.

### The same tape, four "annualised returns"

| Method | SPY | {h['wild_ticker']} |
|---|--:|--:|
{ann}

All four are defensible; only the third and fourth describe what a holder's money did.

## 2. The portfolio mistake

Equal-weight, daily-rebalanced book of {h['portfolio_n']} tapes over
{h['portfolio_years']:.1f} years:

| | CAGR |
|---|--:|
| Correct — weight **simple** returns | {h['portfolio']['cagr_correct']:+.2%} |
| Wrong — weight **log** returns | {h['portfolio']['cagr_wrong']:+.2%} |
| **Gap** | **{h['portfolio_cagr_gap']:+.2%}/yr** |

That is **{h['portfolio_terminal_ratio']:.2f}×** of terminal wealth, and it understates on
essentially every single day: **{h['understates_always']}**. Jensen's inequality is a
one-directional error, which makes it the most dangerous kind — it never announces itself as
noise.

The size of the mistake is not arbitrary: it is almost exactly the **rebalancing bonus**
({h['rebalancing_bonus']:+.2%}/yr here), the genuine excess growth a rebalanced book earns over
the average of its holdings. Weighting log returns throws that away.

| Two-asset book | CAGR gap | Years |
|---|--:|--:|
{pairs}

## 3. Which statistics move

| Ticker | Vol | Sharpe (simple) | Sharpe (log) | Gap |
|---|--:|--:|--:|--:|
{sharpe}

| Beta on SPY | Simple | Log | Relative gap |
|---|--:|--:|--:|
{betas}

A Sharpe ratio moves with the convention (both the numerator and the denominator change). A
**beta barely does**, because it is a ratio of covariances and the second-order corrections
largely cancel — which is why nobody has ever noticed.

## The rule

- **Across time → logs.** They add; their mean exponentiates to the CAGR that actually
  happened.
- **Across assets → simple.** A portfolio return is a weighted average of simple returns,
  exactly, and of log returns, never.
- **Reporting a single "average return" → say which one.** At {h['break_even_vol']:.0%}
  annualised volatility an 8%/yr arithmetic mean corresponds to a geometric mean of zero.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Usefulness: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[969-log-vs-simple-returns](../README.md). Not investment advice.*
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
