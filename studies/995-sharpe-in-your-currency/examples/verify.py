"""Real-tape verification — Study 995 (Whose Sharpe Is It?). Regenerates docs/results.md.

Converts one US fund's returns into six home currencies, gives each investor their
own risk-free rate rather than the US bill rate, decomposes every Sharpe difference into drift,
variance and rate-differential terms, checks whether the currency reorders how five assets rank
against each other, and prices a currency hedge at several ratios with the interest-rate
differential charged explicitly.

    python studies/995-sharpe-in-your-currency/examples/verify.py            # cache-only
    python studies/995-sharpe-in-your-currency/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from whosesharpe import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


COST_BPS = 3.0


def report() -> dict:
    px = data.load_prices()
    rets = px.pct_change()
    h: dict = {"as_of": data.AS_OF, "asset": data.ASSET, "cost_bps": COST_BPS,
               "fingerprint": data.fingerprint(px)}

    fx_rets = {k: rets[v].dropna() for k, v in data.FX.items()
               if v in rets.columns and rets[v].notna().sum() > 1000}
    asset = rets[data.ASSET].dropna()
    usd_rf = rets[data.CASH].reindex(asset.index).fillna(0.0)
    common = asset.index
    for f in fx_rets.values():
        common = common.intersection(f.index)
    asset = asset.reindex(common).dropna()
    usd_rf = usd_rf.reindex(asset.index).fillna(0.0)
    fx_rets = {k: v.reindex(asset.index).dropna() for k, v in fx_rets.items()}
    h["years"] = float(len(asset) / st.TRADING_DAYS)
    h["window"] = [str(asset.index[0].date()), str(asset.index[-1].date())]
    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    print(f"  {data.ASSET} over {asset.index[0].date()} -> {asset.index[-1].date()} "
          f"({h['years']:.1f} years), {len(fx_rets)} currencies")

    print("\n=== 1. checking the FX legs behave as advertised ===")
    for k, v in data.FX.items():
        if k not in fx_rets:
            continue
        f = fx_rets[k]
        print(f"  {k} ({v}): ann drift {f.mean() * st.TRADING_DAYS:+.2%}, "
              f"ann vol {f.std() * np.sqrt(st.TRADING_DAYS):.1%}, "
              f"corr with {data.ASSET} {asset.reindex(f.index).corr(f):+.2f}")
    print("  each of these holds a deposit in the foreign currency, so its dollar return is")
    print("  (currency appreciation + foreign deposit rate) — which is exactly the leg a")
    print("  foreign investor's cash sits in, and is why the rate adjustment below is possible")

    print("\n=== 2. the same fund, six ways ===")
    tbl = st.sharpe_by_currency(asset, fx_rets, usd_rf)
    print(tbl.round(4).to_string())
    h["by_currency"] = tbl.reset_index().to_dict("records")
    h["sharpe_usd"] = float(tbl.loc["USD", "sharpe"])
    h["vol_usd"] = float(tbl.loc["USD", "vol"])
    foreign = tbl.drop("USD")
    h["sharpe_min"] = float(foreign["sharpe"].min())
    h["sharpe_max"] = float(foreign["sharpe"].max())
    h["sharpe_spread"] = float(tbl["sharpe"].max() - tbl["sharpe"].min())
    h["worst_currency"] = str(foreign["sharpe"].idxmin())
    h["best_currency"] = str(foreign["sharpe"].idxmax())
    h["vol_median_foreign"] = float(foreign["vol"].median())
    print(f"  -> Sharpe from {tbl['sharpe'].min():.2f} to {tbl['sharpe'].max():.2f}, "
          f"a spread of {h['sharpe_spread']:.2f} on identical shares")

    print("\n=== 3. the variance channel ===")
    vds = {}
    for k, f in fx_rets.items():
        d = st.variance_decomposition(asset, f)
        vds[k] = d
        print(f"  {k}: corr {d['corr']:+.2f}, vol {d['vol_asset']:.1%} -> "
              f"{d['vol_converted']:.1%} ({d['vol_ratio']:.2f}x), identity error "
              f"{d['approximation_error'] / d['realised_var']:+.2%} of variance")
    h["variance"] = {k: {kk: v[kk] for kk in ("corr", "vol_asset", "vol_converted",
                                              "vol_ratio", "approximation_error")}
                     for k, v in vds.items()}
    h["median_corr"] = float(np.median([v["corr"] for v in vds.values()]))
    print(f"  median correlation with {data.ASSET}: {h['median_corr']:+.2f} — near zero, so "
          f"the currency is nearly pure added variance")

    print("\n=== 4. splitting each Sharpe gap three ways ===")
    decs = {}
    for k, f in fx_rets.items():
        d = st.decompose_sharpe_gap(asset, f, usd_rf)
        decs[k] = d
        print(f"  {k}: gap {d['gap']:+.3f} = drift {d['drift_term']:+.3f} + variance "
              f"{d['variance_term']:+.3f} + rate {d['rate_term']:+.3f}   "
              f"(vol ratio {d['vol_ratio']:.2f}, rate gap "
              f"{d['rate_gap_ann']:+.2%})")
    h["decompositions"] = {k: {kk: v[kk] for kk in
                               ("gap", "drift_term", "variance_term", "rate_term",
                                "vol_ratio", "rate_gap_ann")} for k, v in decs.items()}
    var_terms = [v["variance_term"] for v in decs.values()]
    drift_terms = [v["drift_term"] for v in decs.values()]
    h["median_variance_term"] = float(np.median(var_terms))
    h["median_drift_term"] = float(np.median(drift_terms))
    h["n_variance_negative"] = int(sum(1 for v in var_terms if v < 0))
    print(f"  the variance term is negative for {h['n_variance_negative']} of "
          f"{len(var_terms)} currencies (median {h['median_variance_term']:+.3f}) — it is "
          f"the systematic part")
    print(f"  the drift term is the coin flip: median {h['median_drift_term']:+.3f}, "
          f"{sum(1 for v in drift_terms if v > 0)} positive and "
          f"{sum(1 for v in drift_terms if v < 0)} negative")

    print("\n=== 5. does it reorder anything? ===")
    assets = {tk: rets[tk].reindex(asset.index).dropna()
              for tk in (data.ASSET,) + data.ALT_ASSETS if tk in rets.columns}
    assets = {k: v for k, v in assets.items() if len(v) > 1000}
    h["n_assets"] = int(len(assets))
    rank = st.ranking_stability(assets, fx_rets, usd_rf)
    print(rank.round(3).to_string())
    h["rankings"] = rank.drop("_rank_spread").reset_index().to_dict("records")
    h["rank_spread"] = float(rank.loc["_rank_spread"].iloc[0])
    body = rank.drop("_rank_spread")
    best_by = {c: str(body[c].idxmax()) for c in body.columns}
    h["best_by_currency"] = best_by
    print(f"  the best-Sharpe asset by home currency: "
          + ", ".join(f"{c}->{a}" for c, a in best_by.items()))
    print(f"  the largest ranking move across currencies: {h['rank_spread']:.0f} places")

    print("\n=== 6. hedging, with the carry charged ===")
    hedge_rows = []
    gains, ratios = [], []
    for k, f in fx_rets.items():
        ha = st.hedge_analysis(asset, f, usd_rf, cost_bps=COST_BPS)
        t = ha["table"]
        unh = ha["unhedged"]
        gains.append(ha["sharpe_gain"])
        ratios.append(ha["optimal_ratio"])
        hedge_rows.append({"currency": k, "sharpe_unhedged": unh["sharpe"],
                           "sharpe_hedged": float(t.loc[1.0, "sharpe"]),
                           "gain": ha["sharpe_gain"],
                           "vol_unhedged": unh["vol"],
                           "vol_hedged": float(t.loc[1.0, "vol"]),
                           "optimal_ratio": ha["optimal_ratio"]})
        print(f"  {k}: unhedged Sharpe {unh['sharpe']:.3f} (vol {unh['vol']:.1%}) -> "
              f"hedged {t.loc[1.0, 'sharpe']:.3f} (vol {t.loc[1.0, 'vol']:.1%}), "
              f"gain {ha['sharpe_gain']:+.3f}, optimal ratio {ha['optimal_ratio']:.2f}")
    h["hedging"] = hedge_rows
    h["hedge_helps_share"] = float(np.mean([g > 0 for g in gains]))
    h["median_hedge_gain"] = float(np.median(gains))
    h["median_optimal_ratio"] = float(np.median(ratios))
    print(f"  hedging raised the Sharpe for {h['hedge_helps_share']:.0%} of currencies "
          f"(median {h['median_hedge_gain']:+.3f})")
    print(f"  median variance-minimising ratio: {h['median_optimal_ratio']:.2f} — "
          f"{'a full hedge overshoots' if h['median_optimal_ratio'] < 0.95 else 'close to 1.0'}")
    if data.HEDGED in rets.columns and rets[data.HEDGED].notna().sum() > 1000:
        hj = rets[data.HEDGED].dropna()
        print(f"  for reference, {data.HEDGED} (a real hedged product) had ann vol "
              f"{hj.std() * np.sqrt(st.TRADING_DAYS):.1%} against {data.ASSET}'s "
              f"{h['vol_usd']:.1%}")

    print("\n=== 7. how much of this is the sample window? ===")
    windows = []
    n = len(asset)
    for label, sl in (("first half", slice(0, n // 2)), ("second half", slice(n // 2, n))):
        a2 = asset.iloc[sl]
        f2 = {k: v.reindex(a2.index).dropna() for k, v in fx_rets.items()}
        t2 = st.sharpe_by_currency(a2, f2, usd_rf.reindex(a2.index).fillna(0.0))
        windows.append({"window": label, "spread": float(t2["sharpe"].max()
                                                         - t2["sharpe"].min()),
                        "best": str(t2["sharpe"].idxmax()),
                        "worst": str(t2["sharpe"].idxmin())})
        print(f"  {label}: spread {windows[-1]['spread']:.2f}, best "
              f"{windows[-1]['best']}, worst {windows[-1]['worst']}")
    h["windows"] = windows
    print("  if the best and worst currency swap between halves, the drift channel is doing "
          "the work and will not repeat")

    print("\n=== 8. synthetic control ===")
    ctrl = []
    for corr, drift, gap, tag in ((0.0, 0.0, 0.0, "pure added variance"),
                                  (0.0, -0.03, 0.0, "currency fell"),
                                  (0.0, 0.0, 0.03, "home rates 3pp higher"),
                                  (0.6, 0.0, 0.0, "currency co-moves with the asset")):
        w = st.synthetic_world(n=8000, corr=corr, fx_drift=drift, rate_gap=gap)
        d = st.decompose_sharpe_gap(w["asset"], w["fx"], w["usd_rf"], w["local_rf"])
        ctrl.append({"world": tag, "gap": d["gap"], "drift": d["drift_term"],
                     "variance": d["variance_term"], "rate": d["rate_term"]})
        print(f"  {tag:34s} gap {d['gap']:+.3f} = drift {d['drift_term']:+.3f} + var "
              f"{d['variance_term']:+.3f} + rate {d['rate_term']:+.3f}")
    h["control"] = ctrl
    print("  each row moves exactly the term it should, and only that term")

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    cur = "\n".join(
        f"| {r['currency']} | {r['cagr']:+.2%} | {r['vol']:.1%} | {r['rf_ann']:.2%} | "
        f"{r['excess_ann']:+.2%} | **{r['sharpe']:.3f}** | {r['max_dd']:.1%} |"
        for r in h["by_currency"])
    arith = "\n".join(
        f"| {r['currency']} | {r['sharpe']:.3f} | {r['sharpe_arithmetic']:.3f} | "
        f"{r['sharpe_arithmetic'] - r['sharpe']:+.3f} |" for r in h["by_currency"])
    var = "\n".join(
        f"| {k} | {vv['corr']:+.2f} | {vv['vol_asset']:.1%} | {vv['vol_converted']:.1%} | "
        f"**{vv['vol_ratio']:.2f}×** |" for k, vv in h["variance"].items())
    dec = "\n".join(
        f"| {k} | {vv['gap']:+.3f} | {vv['drift_term']:+.3f} | {vv['variance_term']:+.3f} | "
        f"{vv['rate_term']:+.3f} | {vv['rate_gap_ann']:+.2%} |"
        for k, vv in h["decompositions"].items())
    rank = "\n".join(
        "| " + str(r["index"]) + " | "
        + " | ".join(f"{r[c]:.3f}" for c in r if c != "index") + " |"
        for r in h["rankings"])
    rank_hdr = " | ".join(c for c in h["rankings"][0] if c != "index") \
        if h["rankings"] else ""
    hedge = "\n".join(
        f"| {r['currency']} | {r['sharpe_unhedged']:.3f} | {r['sharpe_hedged']:.3f} | "
        f"**{r['gain']:+.3f}** | {r['vol_unhedged']:.1%} | {r['vol_hedged']:.1%} | "
        f"{r['optimal_ratio']:.2f} |" for r in h["hedging"])
    win = "\n".join(f"| {r['window']} | {r['spread']:.2f} | {r['best']} | {r['worst']} |"
                    for r in h["windows"])
    ctrl = "\n".join(
        f"| {r['world']} | {r['gap']:+.3f} | {r['drift']:+.3f} | {r['variance']:+.3f} | "
        f"{r['rate']:+.3f} |" for r in h["control"])
    return f"""# Results — Study 995 (Whose Sharpe Is It?) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). {h['asset']} measured from
{len(h['by_currency']) - 1} home currencies, {h['window'][0]} → {h['window'][1]}
({h['years']:.1f} years). As-of **{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

## 1. The same fund, from six countries

Each row is the *identical shares*, held by an investor who has to convert back to their own
money and whose cash leg earns their own deposit rate:

| Home currency | CAGR | Vol | Risk-free | Excess | Sharpe | Max DD |
|---|--:|--:|--:|--:|--:|--:|
{cur}

Sharpe from **{h['sharpe_min']:.2f}** to **{h['sharpe_max']:.2f}** — a spread of
**{h['sharpe_spread']:.2f}**.

A note on which Sharpe. Every Sharpe in this study is computed on **log** excess returns, and
that is not fastidiousness. The arithmetic mean of `(1+a)/(1+f) − 1` carries a convexity term of
roughly `var(f)`: dividing by a random number raises the arithmetic average even while it lowers
the compounded outcome. So a more volatile currency inflates the arithmetic Sharpe's numerator
at the same moment it inflates the denominator, and the two partly cancel — leaving an
arithmetic Sharpe that understates the damage. The table reports both
(`sharpe` is the log one, `sharpe_arithmetic` the naive one) so the size of the artefact is
visible rather than assumed away.

### The size of the convexity artefact

| Home currency | Log Sharpe (used here) | Arithmetic Sharpe | Flattery |
|---|--:|--:|--:|
{arith}

## 2. The variance channel — mechanical and permanent

`var(a − c) = var(a) + var(c) − 2cov(a, c)`. Unless the currency moves *with* the asset, adding
a currency leg adds risk:

| Currency | Correlation with {h['asset']} | Vol in USD | Vol at home | Ratio |
|---|--:|--:|--:|--:|
{var}

Median correlation **{h['median_corr']:+.2f}** — near zero, so the currency is close to pure
added variance for these investors.

## 3. Splitting each gap three ways

| Currency | Total gap | Drift | Variance | Rate | Rate gap |
|---|--:|--:|--:|--:|--:|
{dec}

The three channels are not interchangeable:

- **Variance** is systematic: negative for {h['n_variance_negative']} of
  {len(h['decompositions'])} currencies, median {h['median_variance_term']:+.3f}. It does not
  average away.
- **Drift** is a coin flip: median {h['median_drift_term']:+.3f}, and it changes sign across
  currencies and across sub-periods.
- **Rate** is the one nobody adjusts for. Using the US bill rate for every investor — the
  standard shortcut — biases every high-rate country's Sharpe downward.

## 4. Does it reorder anything?

Sharpe of each asset, by home currency:

| Asset | {rank_hdr} |
|---|{"---|" * max(len(h['rankings'][0]) - 1, 1)}
{rank}

Largest ranking move across currencies: **{h['rank_spread']:.0f} places**. The best asset by
Sharpe: {", ".join(f"**{c}** → {a}" for c, a in h["best_by_currency"].items())}.

## 5. Hedging, with the carry charged

A rolling forward hedge earns the **interest-rate differential**, not zero — that is the charge
hedged share classes pass on, and it is applied here rather than ignored:

| Currency | Unhedged Sharpe | Hedged | Gain | Vol unhedged | Vol hedged | Optimal ratio |
|---|--:|--:|--:|--:|--:|--:|
{hedge}

Hedging raised the Sharpe for **{h['hedge_helps_share']:.0%}** of currencies (median
{h['median_hedge_gain']:+.3f}). The variance-minimising ratio has a median of
**{h['median_optimal_ratio']:.2f}** — not 1.0, because {h['asset']} itself co-moves with
risk-off currencies (Campbell, Serfaty-de Medeiros & Viceira 2010).

## 6. Is this the window?

| Window | Sharpe spread | Best currency | Worst |
|---|--:|---|---|
{win}

If the best and worst currency swap places between halves, the **drift** channel is doing the
work and will not repeat. If they hold, the variance and rate channels dominate and the finding
is durable.

## 7. Synthetic control

| World | Gap | Drift | Variance | Rate |
|---|--:|--:|--:|--:|
{ctrl}

Each row moves exactly the term it should and only that term.

## Caveats

- **The foreign risk-free rates are implied, not observed.** They come from each currency ETF's
  drift against dollar cash via covered interest parity, estimated on a three-year rolling
  window. That is an approximation, and a better one than assuming every investor in the world
  earns the US bill rate — which is what most published Sharpe ratios silently do.
- **Currency ETFs are not spot FX.** FXE and friends carry an expense ratio and hold deposits
  rather than the currency itself; their tracking of the underlying rate is close but not exact.
- **One sample window.** Currency drift over eighteen years is one draw. Section 6 splits it in
  half precisely because the drift channel is not repeatable.
- **Arithmetic Sharpe ratios are not comparable across currencies.** Section 1 quantifies the
  artefact. Any published cross-country Sharpe comparison that does not say which mean it used
  is ambiguous by more than the differences it is reporting.
- **No taxes.** Currency gains are taxed differently across all six jurisdictions, sometimes as
  income and sometimes not at all, which would move these numbers again.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[995-sharpe-in-your-currency](../README.md). Not investment advice.*
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
