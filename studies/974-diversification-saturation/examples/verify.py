"""Real-tape verification — Study 974 (The Nth Asset). Regenerates docs/results.md.

Builds equal-weight portfolios of every size from one to twelve out of a
twelve-asset-class universe, drawing constituents at random hundreds of times per size, and
compares the resulting volatility curve with the closed form implied by the sample's own average
correlation — then converts it into a stopping rule and checks what choosing well (rather than
choosing many) would have bought.

    python studies/974-diversification-saturation/examples/verify.py            # cache-only
    python studies/974-diversification-saturation/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from diversify_n import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


DRAWS = 300
COST_BPS = 5.0
REBALANCE = 21


def report() -> dict:
    px = data.load_prices()
    cash = px[data.CASH]
    universe = [t for t in data.UNIVERSE if px[t].notna().sum() > 1000]
    rets = st.excess_returns(px[universe], cash).dropna(how="any")
    h: dict = {"as_of": data.AS_OF, "tickers": list(data.TICKERS),
               "universe": universe, "n_universe": len(universe),
               "fingerprint": data.fingerprint(px)}

    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    print(f"universe ({len(universe)}): {', '.join(universe)}")
    print(f"common excess-return window: {rets.index[0].date()} -> {rets.index[-1].date()} "
          f"({len(rets):,} sessions)")
    h["window"] = [str(rets.index[0].date()), str(rets.index[-1].date())]
    h["n_obs"] = int(len(rets))

    print("\n=== the correlation matrix everything follows from ===")
    corr = rets.corr()
    iu = np.triu_indices_from(corr.to_numpy(), k=1)
    h["avg_corr"] = float(corr.to_numpy()[iu].mean())
    h["min_corr"] = float(corr.to_numpy()[iu].min())
    h["max_corr"] = float(corr.to_numpy()[iu].max())
    pairs = [(universe[i], universe[j], corr.iloc[i, j]) for i, j in zip(*iu)]
    pairs.sort(key=lambda t: -t[2])
    print(f"  average pairwise correlation {h['avg_corr']:.3f} "
          f"(range {h['min_corr']:.2f} .. {h['max_corr']:.2f})")
    print("  most redundant pairs:")
    for a, b, c in pairs[:4]:
        print(f"    {a:5s} / {b:5s} {c:.3f}")
    print("  most diversifying pairs:")
    for a, b, c in pairs[-4:]:
        print(f"    {a:5s} / {b:5s} {c:+.3f}")
    h["top_pairs"] = [[a, b, float(c)] for a, b, c in pairs[:4]]
    h["bottom_pairs"] = [[a, b, float(c)] for a, b, c in pairs[-4:]]

    print(f"\n=== the diversification curve ({DRAWS} random subsets per size, monthly "
          f"rebalance, {COST_BPS:.0f} bps) ===")
    curve = st.random_subset_curve(rets, draws=DRAWS, cost_bps=COST_BPS,
                                   rebalance=REBALANCE)
    theory = st.theoretical_curve(rets)
    mb = st.marginal_benefit(curve, threshold=0.05)
    print("   k   vol (mean)   p10 - p90        theory    marginal drop   Sharpe   maxDD")
    for k in curve.index:
        print(f"  {k:2d}  {curve.loc[k, 'vol_mean']:10.2%}   "
              f"{curve.loc[k, 'vol_p10']:.2%} - {curve.loc[k, 'vol_p90']:.2%}  "
              f"{theory.loc[k, 'vol_theory']:11.2%}   "
              f"{(mb.loc[k, 'drop_rel'] if np.isfinite(mb.loc[k, 'drop_rel']) else 0):13.1%}   "
              f"{curve.loc[k, 'sharpe_mean']:+6.2f}  {curve.loc[k, 'maxdd_mean']:+7.1%}")
    h["curve"] = {int(k): dict(v) for k, v in curve.to_dict("index").items()}
    h["theory"] = {int(k): float(v) for k, v in theory["vol_theory"].items()}
    h["floor_vol"] = float(theory.attrs["floor_vol"])
    h["vol_k1"] = float(curve.loc[1, "vol_mean"])
    h["vol_kmax"] = float(curve["vol_mean"].iloc[-1])
    h["vol_reduction_total"] = float(1 - h["vol_kmax"] / h["vol_k1"])
    h["third_gain"] = float(mb.loc[3, "drop_rel"])
    h["last_gain"] = float(mb["drop_rel"].iloc[-1])
    print(f"  theoretical floor (average covariance): {h['floor_vol']:.2%} — the full "
          f"portfolio is {abs(h['vol_kmax'] - h['floor_vol']):.2%} from it")

    print("\n=== where does it stop paying? ===")
    stops = {}
    for thr in (0.10, 0.05, 0.02, 0.01):
        stops[thr] = st.stopping_point(curve, threshold=thr)
        print(f"  at a {thr:.0%} relative-improvement threshold: stop at k = {stops[thr]}")
    h["stops"] = {str(k): int(v) for k, v in stops.items()}
    h["stop_5pct"] = int(stops[0.05])
    h["stop_2pct"] = int(stops[0.02])
    k_stop = h["stop_5pct"]
    h["vol_at_stop"] = float(curve.loc[k_stop, "vol_mean"])
    h["tail_gain"] = float(1 - h["vol_kmax"] / curve.loc[k_stop, "vol_mean"])
    h["dispersion_at_k5"] = float((curve.loc[min(5, curve.index.max()), "vol_p90"] -
                                   curve.loc[min(5, curve.index.max()), "vol_p10"]) /
                                  curve.loc[min(5, curve.index.max()), "vol_mean"])
    print(f"  everything beyond k = {k_stop} is worth a further {h['tail_gain']:.1%} of "
          f"volatility")
    print(f"  dispersion across random draws at k = 5: "
          f"{h['dispersion_at_k5']:.0%} of the mean volatility — choosing WHICH five matters "
          f"more than choosing five")

    print("\n=== effective number of bets (Meucci) ===")
    enb = {}
    for k in curve.index:
        subset = universe[:k]
        enb[k] = st.effective_number_of_bets(rets[subset])
    for k in curve.index:
        print(f"  {k:2d} nominal assets -> {enb[k]:5.2f} effective bets")
    h["enb"] = {int(k): float(v) for k, v in enb.items()}
    h["enb_full"] = float(enb[curve.index.max()])

    print("\n=== choosing well vs choosing many (greedy order, IN SAMPLE — an upper bound) ===")
    order = st.greedy_order(rets, cost_bps=COST_BPS, rebalance=REBALANCE)
    g = st.ordered_curve(rets, order, cost_bps=COST_BPS, rebalance=REBALANCE)
    for k in g.index:
        print(f"  {k:2d}  add {g.loc[k, 'added']:5s}  vol {g.loc[k, 'vol']:.2%}  "
              f"(random average {curve.loc[k, 'vol_mean']:.2%})  "
              f"Sharpe {g.loc[k, 'sharpe']:+.2f}  effective bets {g.loc[k, 'enb']:.2f}")
    h["greedy_order"] = order
    h["greedy"] = {int(k): {"added": str(v["added"]), "vol": float(v["vol"]),
                            "sharpe": float(v["sharpe"]), "enb": float(v["enb"])}
                   for k, v in g.to_dict("index").items()}
    h["greedy_vol_at_stop"] = float(g.loc[k_stop, "vol"])
    print(f"  the best {k_stop} assets reached {h['greedy_vol_at_stop']:.2%} against "
          f"{h['vol_at_stop']:.2%} for a random {k_stop} — and the greedy order is chosen with "
          f"hindsight, so treat it as the ceiling, not a plan")

    print("\n=== does the cost of rebalancing twelve assets eat the benefit? ===")
    for cb in (0.0, 5.0, 20.0):
        c = st.random_subset_curve(rets, draws=100, cost_bps=cb, rebalance=REBALANCE)
        print(f"  {cb:4.0f} bps: k=1 Sharpe {c.loc[1, 'sharpe_mean']:+.2f} -> "
              f"k={c.index.max()} Sharpe {c['sharpe_mean'].iloc[-1]:+.2f}  "
              f"(vol {c.loc[1, 'vol_mean']:.2%} -> {c['vol_mean'].iloc[-1]:.2%})")
        h[f"cost_{int(cb)}"] = {"sharpe_1": float(c.loc[1, "sharpe_mean"]),
                                "sharpe_max": float(c["sharpe_mean"].iloc[-1])}

    print("\n=== control: a synthetic panel with a known correlation ===")
    for rho, tag in ((0.0, "independent"), (0.9, "near-duplicates")):
        rng = np.random.default_rng(974)
        n, k = 4000, 8
        sd = 0.20 / np.sqrt(252)
        f = rng.normal(0, sd * np.sqrt(rho), (n, 1))
        sim = pd.DataFrame(f + rng.normal(0, sd * np.sqrt(1 - rho), (n, k)),
                           index=pd.bdate_range("2005-01-03", periods=n),
                           columns=[f"A{i}" for i in range(k)])
        c = st.random_subset_curve(sim, draws=60, cost_bps=0.0)
        print(f"  {tag:16s} vol {c.loc[1, 'vol_mean']:.2%} -> {c['vol_mean'].iloc[-1]:.2%} "
              f"({1 - c['vol_mean'].iloc[-1] / c.loc[1, 'vol_mean']:.0%} reduction), "
              f"effective bets {st.effective_number_of_bets(sim):.2f}")
        h[f"control_rho{int(rho * 10)}"] = {
            "reduction": float(1 - c["vol_mean"].iloc[-1] / c.loc[1, "vol_mean"]),
            "enb": float(st.effective_number_of_bets(sim))}

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Usefulness: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    curve = "\n".join(
        f"| {k} | {r['vol_mean']:.2%} | {r['vol_p10']:.2%} – {r['vol_p90']:.2%} | "
        f"{h['theory'][k]:.2%} | {r['sharpe_mean']:+.2f} | {r['maxdd_mean']:+.1%} | "
        f"{h['enb'][k]:.2f} |"
        for k, r in h["curve"].items())
    greedy = "\n".join(
        f"| {k} | {r['added']} | {r['vol']:.2%} | {h['curve'][k]['vol_mean']:.2%} | "
        f"{r['sharpe']:+.2f} | {r['enb']:.2f} |" for k, r in h["greedy"].items())
    stops = " · ".join(f"{float(t):.0%} → k = {k}" for t, k in h["stops"].items())
    top = "\n".join(f"| {a} / {b} | {c:+.3f} |" for a, b, c in h["top_pairs"])
    bot = "\n".join(f"| {a} / {b} | {c:+.3f} |" for a, b, c in h["bottom_pairs"])
    return f"""# Results — Study 974 (The Nth Asset) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). Equal-weight portfolios of every
size from 1 to {h['n_universe']}, constituents drawn at random 300 times per size, weights
drifting between monthly rebalances, 5 bps per unit of traded notional, all on
**excess-of-cash** returns. Common window {h['window'][0]} → {h['window'][1]}
({h['n_obs']:,} sessions). As-of **{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

Universe: {', '.join(h['universe'])}.

## The correlation matrix everything follows from

Average pairwise correlation **{h['avg_corr']:.3f}** (range {h['min_corr']:.2f} to
{h['max_corr']:.2f}).

| Most redundant pairs | ρ |
|---|--:|
{top}

| Most diversifying pairs | ρ |
|---|--:|
{bot}

## The diversification curve

| k | Volatility (mean) | 10th – 90th percentile | Closed form | Sharpe | Max DD | Effective bets |
|---|--:|---|--:|--:|--:|--:|
{curve}

One asset averages **{h['vol_k1']:.1%}** annualised volatility; the whole universe runs
**{h['vol_kmax']:.1%}** — a **{h['vol_reduction_total']:.0%}** reduction. The closed form
`σ²/k + ρ·σ²·(k−1)/k`, evaluated with this sample's own average variance and correlation, puts
the floor at **{h['floor_vol']:.2%}**; the twelve-asset portfolio is
{abs(h['vol_kmax'] - h['floor_vol']):.2%} away from it. There is nothing exotic happening — the
textbook curve is the curve.

## Where it stops paying

Stopping point by relative-improvement threshold: {stops}.

Everything beyond k = {h['stop_5pct']} is worth a further **{h['tail_gain']:.1%}** of
volatility. Meanwhile the spread across *which* assets you pick at k = 5 is
**{h['dispersion_at_k5']:.0%}** of the mean volatility — the choice of constituents matters
more than the count.

## Choosing well versus choosing many

Greedy order (minimising volatility at each step, **in sample** — an upper bound, not a plan):

| k | Added | Greedy volatility | Random-average volatility | Sharpe | Effective bets |
|---|---|--:|--:|--:|--:|
{greedy}

## The control

On a synthetic panel of eight independent assets the curve delivers a
**{h['control_rho0']['reduction']:.0%}** reduction and {h['control_rho0']['enb']:.2f} effective
bets; on eight near-duplicates (ρ = 0.9) it delivers
**{h['control_rho9']['reduction']:.0%}** and {h['control_rho9']['enb']:.2f}. The machinery
tracks the correlation structure it is given.

## Caveats

- **Volatility is not risk.** The curve measures dispersion; it says nothing about the tail,
  and correlations rise exactly when they matter most (study **578**).
- **Equal weight is a choice.** A risk-parity or minimum-variance version of this curve would
  saturate at a different k; the equal-weight version is the one a reader can replicate.
- **In-sample greedy ordering** is hindsight by construction and is labelled as such — the
  random curve is the honest expectation.
- **Twelve asset classes, one era.** The sample starts after 2007 for the newest funds, so the
  whole curve lives in a single correlation regime.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Usefulness: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[974-diversification-saturation](../README.md). Not investment advice.*
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
