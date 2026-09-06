"""Real-tape verification — Study 972 (Adjusted or Not). Regenerates docs/results.md.

Loads both adjustment conventions for the same eight tickers, measures the return the
price-only view deletes, counts how often the two views disagree about a cross-sectional
ranking, and runs a 12-1 momentum sleeve twice — ranking on each panel, scoring both on total
returns — so the selection effect is isolated from the income effect.

    python studies/972-adjustment-mode-matters/examples/verify.py            # cache-only
    python studies/972-adjustment-mode-matters/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from adj_mode import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


TOP_K = 3
COST_BPS = 5.0


def report() -> dict:
    panels = data.load_pair()
    tr, px = panels["tr"], panels["pxonly"]
    h: dict = {"as_of": data.AS_OF, "tickers": list(data.TICKERS),
               "fingerprint_tr": data.fingerprint(tr),
               "fingerprint_px": data.fingerprint(px)}

    print(f"as-of {data.AS_OF}   common index {tr.index[0].date()} -> {tr.index[-1].date()}  "
          f"n={len(tr):,}")
    print(f"  fingerprints: total-return {h['fingerprint_tr']}  price-only {h['fingerprint_px']}")
    h["window"] = [str(tr.index[0].date()), str(tr.index[-1].date())]
    h["n_obs"] = int(len(tr))

    print("\n=== 1. what the price-only view deletes ===")
    print("  tkr      CAGR (TR)   CAGR (price)   implied yield   share of return   vol gap")
    yt = st.yield_table(tr, px).sort_values("implied_yield")
    for tk, row in yt.iterrows():
        print(f"  {tk:6s} {row['cagr_tr']:+11.2%} {row['cagr_px']:+14.2%} "
              f"{row['implied_yield']:15.2%} {row['share_of_return']:17.0%} "
              f"{row['vol_tr'] - row['vol_px']:+9.3%}")
    h["yields"] = {tk: dict(v) for tk, v in yt.to_dict("index").items()}
    h["max_implied_yield"] = float(yt["implied_yield"].max())
    h["min_implied_yield"] = float(yt["implied_yield"].min())
    h["max_yield_ticker"] = str(yt["implied_yield"].idxmax())
    h["min_yield_ticker"] = str(yt["implied_yield"].idxmin())
    h["max_share_of_return"] = float(yt["share_of_return"].max())
    h["max_vol_gap"] = float((yt["vol_tr"] - yt["vol_px"]).abs().max())

    print("\n=== 2. risk-adjusted ratios move because the numerator does ===")
    rt = st.risk_table(tr, px)
    for tk, row in rt.iterrows():
        print(f"  {tk:6s} Sharpe {row['sharpe_tr']:+.3f} -> {row['sharpe_px']:+.3f} "
              f"({row['sharpe_gap']:+.3f})   maxDD {row['maxdd_tr']:+.1%} -> "
              f"{row['maxdd_px']:+.1%}   years underwater "
              f"{row['years_underwater_tr']:.1f} -> {row['years_underwater_px']:.1f}")
    h["risk"] = {tk: dict(v) for tk, v in rt.to_dict("index").items()}
    h["max_sharpe_gap"] = float(rt["sharpe_gap"].max())

    print("\n=== 3. the cross-sectional damage ===")
    rank = st.ranking_table(tr, px)
    print(f"  {len(rank)} month-ends compared")
    print(f"  mean Spearman between the two rankings : {rank['spearman'].mean():.3f} "
          f"(worst {rank['spearman'].min():.3f})")
    print(f"  average share of asset PAIRS reordered : {rank['flip_share'].mean():.1%}")
    print(f"  same top-ranked asset                  : {rank['same_top'].mean():.0%} of months")
    print(f"  same bottom-ranked asset               : {rank['same_bottom'].mean():.0%} of months")
    h["ranking"] = {"n_months": int(len(rank)),
                    "mean_spearman": float(rank["spearman"].mean()),
                    "min_spearman": float(rank["spearman"].min()),
                    "mean_flip_share": float(rank["flip_share"].mean()),
                    "same_top_share": float(rank["same_top"].mean()),
                    "same_bottom_share": float(rank["same_bottom"].mean())}
    h["mean_flip_share"] = h["ranking"]["mean_flip_share"]
    h["same_top_share"] = h["ranking"]["same_top_share"]

    print("\n=== 4. the same momentum sleeve, ranked two ways, scored one way ===")
    on_tr = st.momentum_backtest(tr, tr, top_k=TOP_K, cost_bps=COST_BPS)
    on_px = st.momentum_backtest(px, tr, top_k=TOP_K, cost_bps=COST_BPS)
    ys = yt["implied_yield"]
    y_tr = st.holding_yield(on_tr["weights"], ys)
    y_px = st.holding_yield(on_px["weights"], ys)
    for label, res in (("rank on total return", on_tr), ("rank on price only", on_px)):
        print(f"  {label:22s} CAGR {res['cagr']:+.2%}  vol {res['vol']:.2%}  "
              f"Sharpe {res['sharpe']:+.2f}  maxDD {res['max_dd']:+.1%}  "
              f"turnover {res['turnover_ann']:.2f}/yr")
    print(f"  average dividend yield of what each held: total-return signal {y_tr:.2%}, "
          f"price signal {y_px:.2%}  (tilt {y_px - y_tr:+.2%})")
    h["momentum_cagr_tr"] = on_tr["cagr"]
    h["momentum_cagr_px"] = on_px["cagr"]
    h["momentum_cagr_gap"] = on_tr["cagr"] - on_px["cagr"]
    h["momentum_sharpe_tr"] = on_tr["sharpe"]
    h["momentum_sharpe_px"] = on_px["sharpe"]
    h["momentum_sharpe_gap"] = on_tr["sharpe"] - on_px["sharpe"]
    h["momentum_dd_tr"] = on_tr["max_dd"]
    h["momentum_dd_px"] = on_px["max_dd"]
    h["yield_tilt"] = float(y_px - y_tr)

    print("\n=== sensitivity: does the conclusion depend on the sleeve's parameters? ===")
    sweep = []
    for k in (2, 3, 4):
        for lb in (126, 252):
            a = st.momentum_backtest(tr, tr, top_k=k, lookback=lb, cost_bps=COST_BPS)
            b = st.momentum_backtest(px, tr, top_k=k, lookback=lb, cost_bps=COST_BPS)
            sweep.append({"top_k": k, "lookback": lb, "cagr_tr": a["cagr"],
                          "cagr_px": b["cagr"], "gap": a["cagr"] - b["cagr"],
                          "sharpe_gap": a["sharpe"] - b["sharpe"]})
            print(f"  top {k}, lookback {lb}d: TR-ranked {a['cagr']:+.2%} vs price-ranked "
                  f"{b['cagr']:+.2%}  (gap {a['cagr'] - b['cagr']:+.2%}, Sharpe gap "
                  f"{a['sharpe'] - b['sharpe']:+.2f})")
    h["sweep"] = sweep
    h["sweep_tr_wins"] = int(sum(1 for r in sweep if r["gap"] > 0))

    print("\n=== control: the same machinery on synthetic panels ===")
    for ss, tag in ((1.0, "yields 0-6% planted"), (0.0, "no dividends (null)")):
        a, b, truth = data.synthetic_pair(n_years=20, signal_strength=ss, seed=972)
        r = st.ranking_table(a, b)
        yy = st.yield_table(a, b)
        print(f"  {tag:22s} implied yields "
              f"{yy['implied_yield'].min():.2%}..{yy['implied_yield'].max():.2%}  "
              f"mean Spearman {r['spearman'].mean():.3f}  pairs reordered "
              f"{r['flip_share'].mean():.1%}")
        h[f"control_{'planted' if ss else 'null'}"] = {
            "mean_spearman": float(r["spearman"].mean()),
            "flip_share": float(r["flip_share"].mean()),
            "max_yield": float(yy["implied_yield"].max())}

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Usefulness: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    y = "\n".join(
        f"| {tk} | {r['cagr_tr']:+.2%} | {r['cagr_px']:+.2%} | **{r['implied_yield']:.2%}** | "
        f"{r['share_of_return']:.0%} | {r['vol_tr'] - r['vol_px']:+.3%} |"
        for tk, r in sorted(h["yields"].items(), key=lambda kv: kv[1]["implied_yield"]))
    rk = "\n".join(
        f"| {tk} | {r['sharpe_tr']:+.3f} | {r['sharpe_px']:+.3f} | {r['sharpe_gap']:+.3f} | "
        f"{r['maxdd_tr']:+.1%} | {r['maxdd_px']:+.1%} | {r['years_underwater_tr']:.1f} | "
        f"{r['years_underwater_px']:.1f} |"
        for tk, r in h["risk"].items())
    sw = "\n".join(
        f"| {r['top_k']} | {r['lookback']}d | {r['cagr_tr']:+.2%} | {r['cagr_px']:+.2%} | "
        f"{r['gap']:+.2%} | {r['sharpe_gap']:+.2f} |" for r in h["sweep"])
    return f"""# Results — Study 972 (Adjusted or Not) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). Two closes per ticker from the
same provider in one pass: **total return** (`auto_adjust=True`) and **price only**
(`auto_adjust=False`), on a common index of {h['n_obs']:,} sessions
({h['window'][0]} → {h['window'][1]}). As-of **{h['as_of']}**; fingerprints
`{h['fingerprint_tr']}` (TR) and `{h['fingerprint_px']}` (price).*

## 1. What the price-only view deletes

| Ticker | CAGR (total return) | CAGR (price) | Implied yield | Share of total return | Volatility gap |
|---|--:|--:|--:|--:|--:|
{y}

The implied yield *is* the difference between the two conventions — there is nothing else in
it. Volatility barely moves (largest gap {h['max_vol_gap']:.3%}), which is why every
risk-adjusted ratio changes by almost exactly the return that was removed.

## 2. Risk-adjusted ratios

| Ticker | Sharpe (TR) | Sharpe (price) | Gap | Max DD (TR) | Max DD (price) | Years underwater (TR) | (price) |
|---|--:|--:|--:|--:|--:|--:|--:|
{rk}

## 3. The cross-sectional damage

Over **{h['ranking']['n_months']}** month-ends, ranking the universe by 12-1 trailing return:

| | |
|---|--:|
| Mean Spearman between the two rankings | {h['ranking']['mean_spearman']:.3f} |
| Worst month | {h['ranking']['min_spearman']:.3f} |
| Average share of asset pairs reordered | **{h['ranking']['mean_flip_share']:.1%}** |
| Same top-ranked asset | {h['ranking']['same_top_share']:.0%} of months |
| Same bottom-ranked asset | {h['ranking']['same_bottom_share']:.0%} of months |

## 4. The same sleeve, ranked two ways, scored one way

Both arms hold real total-return assets and both collect their dividends; the **only**
difference is which panel the ranking read. That isolates selection from income.

| Signal | CAGR | Sharpe | Max DD | Held yield |
|---|--:|--:|--:|--:|
| Rank on total return | {h['momentum_cagr_tr']:+.2%} | {h['momentum_sharpe_tr']:+.2f} | {h['momentum_dd_tr']:+.1%} | — |
| Rank on price only | {h['momentum_cagr_px']:+.2%} | {h['momentum_sharpe_px']:+.2f} | {h['momentum_dd_px']:+.1%} | {h['yield_tilt']:+.2%} vs the other |

| Top-k | Lookback | CAGR (TR-ranked) | CAGR (price-ranked) | Gap | Sharpe gap |
|---|---|--:|--:|--:|--:|
{sw}

The total-return signal wins **{h['sweep_tr_wins']} of {len(h['sweep'])}** parameter
combinations. A price-only ranking is not a neutral simplification: it is a standing bet
against whatever pays income.

## The control

On a synthetic panel where every asset earns the **same total return** but pays a different
yield, the price-only ranking orders the universe by yield: mean Spearman
{h['control_planted']['mean_spearman']:.3f}, {h['control_planted']['flip_share']:.1%} of pairs
reordered. With dividends switched off entirely the two panels are identical and the rank
correlation is {h['control_null']['mean_spearman']:.3f}.

## Caveats

- **Taxes are not modelled.** A taxable holder does not receive the full dividend, so the true
  answer for them sits between the two conventions — closer to total return for a tax-deferred
  account, closer to price for a high-bracket taxable one.
- **Reinvestment at the close** is the assumption inside every total-return series, including
  this provider's; a real holder reinvests late, at a different price, and pays a spread.
- **Price-only is not always wrong.** Chart patterns, index valuation levels and
  drawdown-in-price-terms are genuine questions about the quoted price. The rule is to know
  which question you are asking.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Usefulness: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[972-adjustment-mode-matters](../README.md). Not investment advice.*
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
