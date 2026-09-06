"""Real-tape verification — Study 976 (The Family Tree). Regenerates docs/results.md.

Builds hierarchical risk parity from scratch — correlation distance, single-linkage
tree, quasi-diagonal order, recursive bisection — and races it against minimum variance,
inverse variance, equal risk contribution and 1/N on three panels, out of sample, with paired
tests and a planted-block Monte Carlo.

    python studies/976-hierarchical-risk-parity/examples/verify.py            # cache-only
    python studies/976-hierarchical-risk-parity/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hrp import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


WINDOW = 252
STEP = 63
COST_BPS = 5.0


def report() -> dict:
    px = data.load_prices()
    rets = px.pct_change()
    panels = {
        "sectors": [c for c in data.SECTORS if rets[c].notna().sum() > 1500],
        "names": [c for c in data.NAMES if rets[c].notna().sum() > 3000],
        "multi-asset": [c for c in data.MULTI if rets[c].notna().sum() > 3000],
    }
    h: dict = {"as_of": data.AS_OF, "window": WINDOW, "step": STEP,
               "fingerprint": data.fingerprint(px),
               "panels": {k: v for k, v in panels.items()},
               "n_names": len(panels["names"])}

    print(f"as-of {data.AS_OF}   window {WINDOW}d every {STEP}d   fp {data.fingerprint(px)}")
    for tag, cols in panels.items():
        print(f"  {tag:12s} {len(cols):3d} assets")

    results, hierarchy = {}, 0
    for tag, cols in panels.items():
        sub = rets[cols].dropna(how="any")
        cov = np.cov(sub.iloc[-WINDOW:].to_numpy(), rowvar=False, ddof=1)
        order = st.cluster_order(cov)
        w_hrp = st.hrp_weights(cov)
        w_iv = st.inverse_variance_weights(cov)
        gap = float(np.abs(w_hrp - w_iv).sum() / 2)     # share of the book that moves
        hierarchy += int(gap > 0.02)
        print(f"\n=== {tag}: {len(cols)} assets, {len(sub):,} sessions "
              f"{sub.index[0].date()} -> {sub.index[-1].date()} ===")
        print(f"  clustering moves {gap:.1%} of the book relative to plain inverse variance")
        print(f"  leaf order: {', '.join(cols[i] for i in order[:min(len(order), 12)])}"
              f"{' ...' if len(order) > 12 else ''}")

        wf = st.walk_forward(sub, window=WINDOW, step=STEP, cost_bps=COST_BPS)
        s = st.summarise(wf)
        print("  method                       realised vol   return  Sharpe  turnover  "
              "max w  eff. N  shorts")
        for m, row in s.iterrows():
            print(f"  {st.METHOD_LABEL[m]:28s} {row['realised_vol']:12.2%} "
                  f"{row['mean_ret']:8.2%} {row['sharpe']:7.2f} {row['turnover']:9.2f} "
                  f"{row['max_weight']:6.1%} {row['effective_n']:7.1f} {row['short']:7.1%}")
        pairs = {other: st.paired_test(wf, "hrp", other)
                 for other in ("min_var", "inv_var", "risk_parity", "equal")}
        for other, p in pairs.items():
            print(f"    HRP vs {st.METHOD_LABEL[other]:26s} vol difference {p['diff']:+.3%}  "
                  f"t {p['t']:+5.2f}  HRP quieter in {p['win_rate']:.0%} of {p['n']}")
        results[tag] = {"summary": {m: dict(v) for m, v in s.to_dict("index").items()},
                        "pairs": pairs, "weight_gap": gap, "order": order,
                        "n_sessions": int(len(sub))}

    h["results"] = results
    h["n_panels_hierarchy_matters"] = int(hierarchy)
    h["weight_gap_wide"] = float(results["names"]["weight_gap"])
    h["weight_gap_multi"] = float(results["multi-asset"]["weight_gap"])
    wide = results["names"]["summary"]
    h["vol_hrp"] = float(wide["hrp"]["realised_vol"])
    h["vol_minvar"] = float(wide["min_var"]["realised_vol"])
    h["vol_invvar"] = float(wide["inv_var"]["realised_vol"])
    h["vol_equal"] = float(wide["equal"]["realised_vol"])
    h["effective_n_hrp"] = float(wide["hrp"]["effective_n"])
    h["effective_n_minvar"] = float(wide["min_var"]["effective_n"])
    h["max_weight_hrp"] = float(wide["hrp"]["max_weight"])
    h["max_weight_minvar"] = float(wide["min_var"]["max_weight"])
    h["turnover_hrp"] = float(wide["hrp"]["turnover"])
    h["turnover_minvar"] = float(wide["min_var"]["turnover"])
    for other in ("min_var", "inv_var", "equal"):
        h[f"t_vs_{other.replace('_', '')}"] = float(
            -results["names"]["pairs"][other]["t"])   # positive = HRP is quieter
    h["t_vs_minvar"] = float(-results["names"]["pairs"]["min_var"]["t"])
    h["t_vs_invvar"] = float(-results["names"]["pairs"]["inv_var"]["t"])
    h["t_vs_equal"] = float(-results["names"]["pairs"]["equal"]["t"])

    print("\n=== window sensitivity (wide panel, realised volatility) ===")
    sens = []
    sub = rets[panels["names"]].dropna(how="any")
    for w in (126, 252, 504):
        s = st.summarise(st.walk_forward(sub, window=w, step=STEP, cost_bps=COST_BPS))
        sens.append({"window": w, **{m: float(s.loc[m, "realised_vol"]) for m in st.METHODS}})
        print(f"  {w:4d}d: " + "  ".join(f"{st.METHOD_LABEL[m][:12]:12s}{s.loc[m, 'realised_vol']:6.2%}"
                                         for m in st.METHODS))
    h["window_sensitivity"] = sens

    print("\n=== Monte Carlo: planted block correlation, the structure HRP is built for ===")
    mc = []
    for rho_in, rho_out, tag in ((0.7, 0.1, "strong blocks"), (0.3, 0.25, "weak blocks"),
                                 (0.3, 0.3, "no block structure")):
        vols = {m: [] for m in st.METHODS}
        for seed in range(12):
            X, cov_true = st.block_panel(n_per_block=10, n_blocks=2, n_obs=250,
                                         rho_in=rho_in, rho_out=rho_out, seed=976 + seed)
            Y, _ = st.block_panel(n_per_block=10, n_blocks=2, n_obs=250, rho_in=rho_in,
                                  rho_out=rho_out, seed=5000 + seed)   # out-of-sample draw
            cov_hat = np.cov(X, rowvar=False, ddof=1)
            for m in st.METHODS:
                w = st.weights_for(m, cov_hat)
                vols[m].append(float(np.std(Y @ w, ddof=1) * np.sqrt(st.TRADING_DAYS)))
        row = {"world": tag, **{m: float(np.mean(v)) for m, v in vols.items()}}
        mc.append(row)
        print(f"  {tag:20s} " + "  ".join(f"{st.METHOD_LABEL[m][:12]:12s}{row[m]:6.2%}"
                                          for m in st.METHODS))
    h["monte_carlo"] = mc

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Usefulness: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    def block(tag):
        s = h["results"][tag]["summary"]
        return "\n".join(
            f"| {st.METHOD_LABEL[m]} | {r['realised_vol']:.2%} | {r['mean_ret']:+.2%} | "
            f"{r['sharpe']:+.2f} | {r['turnover']:.2f} | {r['max_weight']:.1%} | "
            f"{r['effective_n']:.1f} | {r['short']:.1%} |" for m, r in s.items())
    def pairs(tag):
        return "\n".join(
            f"| HRP − {st.METHOD_LABEL[o]} | {p['diff']:+.3%} | {-p['t']:+.2f} | "
            f"{p['win_rate']:.0%} |" for o, p in h["results"][tag]["pairs"].items())
    sens = "\n".join("| " + str(r["window"]) + "d | " +
                     " | ".join(f"{r[m]:.2%}" for m in st.METHODS) + " |"
                     for r in h["window_sensitivity"])
    mc = "\n".join("| " + r["world"] + " | " + " | ".join(f"{r[m]:.2%}" for m in st.METHODS) + " |"
                   for r in h["monte_carlo"])
    heads = " | ".join(st.METHOD_LABEL[m] for m in st.METHODS)
    return f"""# Results — Study 976 (The Family Tree) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). Hierarchical risk parity
implemented from scratch (correlation distance → single-linkage tree → quasi-diagonal order →
recursive bisection), raced against four alternatives on three panels. Rolling
**{h['window']}-day** window re-estimated every **{h['step']}** sessions, one day of execution
lag, 5 bps a rebalance. As-of **{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

## Does the hierarchy change anything?

Share of the book that moves relative to plain inverse-variance weighting:
**{h['weight_gap_wide']:.1%}** on the {h['n_names']}-name panel,
**{h['weight_gap_multi']:.1%}** on the multi-asset panel,
{h['results']['sectors']['weight_gap']:.1%} on sectors. That is the honest measure of what the
*clustering* contributes, as distinct from the risk weighting that comes free with it.

## Eleven sectors

| Method | Realised vol | Return | Sharpe | Turnover | Max weight | Effective N | Shorts |
|---|--:|--:|--:|--:|--:|--:|--:|
{block('sectors')}

| Paired comparison | Vol difference | *t* (positive = HRP quieter) | HRP wins |
|---|--:|--:|--:|
{pairs('sectors')}

## Forty single names — the case HRP is argued for

| Method | Realised vol | Return | Sharpe | Turnover | Max weight | Effective N | Shorts |
|---|--:|--:|--:|--:|--:|--:|--:|
{block('names')}

| Paired comparison | Vol difference | *t* (positive = HRP quieter) | HRP wins |
|---|--:|--:|--:|
{pairs('names')}

## Ten multi-asset sleeves

| Method | Realised vol | Return | Sharpe | Turnover | Max weight | Effective N | Shorts |
|---|--:|--:|--:|--:|--:|--:|--:|
{block('multi-asset')}

| Paired comparison | Vol difference | *t* (positive = HRP quieter) | HRP wins |
|---|--:|--:|--:|
{pairs('multi-asset')}

## Window sensitivity (wide panel, realised volatility)

| Window | {heads} |
|---|--:|--:|--:|--:|--:|
{sens}

## Monte Carlo: where the block structure is planted

Out-of-sample volatility, weights fitted on one 250-day draw and evaluated on an independent
one, averaged over twelve seeds:

| World | {heads} |
|---|--:|--:|--:|--:|--:|
{mc}

The bottom row is the control: with no block structure the tree has nothing to find, and HRP
should — and does — collapse toward inverse-variance weighting.

## Caveats

- **Single linkage** is López de Prado's choice and it is the least stable of the linkage
  methods; average or Ward linkage would give a different tree and a different (usually
  similar) portfolio.
- **No shrinkage.** The covariance matrices here are plain sample estimates, so part of what
  HRP is beating is an estimator this desk already knows how to improve — see study **975**.
  The interesting comparison, HRP against a *shrunk* optimiser, is a fork of this study.
- **Long-only by construction** is doing real work in HRP's favour: Jagannathan & Ma (2003)
  showed that constraint alone is worth much of what a better estimator buys.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Usefulness: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[976-hierarchical-risk-parity](../README.md). Not investment advice.*
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
