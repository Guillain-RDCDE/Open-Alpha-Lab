"""Real-tape verification — Study 977 (Maximum Diversification). Regenerates docs/results.md.

Builds the most diversified portfolio in closed form, checks it against the
minimum-variance portfolio and three free alternatives on two panels out of sample, reports the
diversification ratio it promised and the one it delivered, and pins the degenerate case where
the two objectives provably coincide.

    python studies/977-max-diversification/examples/verify.py            # cache-only
    python studies/977-max-diversification/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from max_div import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


WINDOW = 252
STEP = 63
COST_BPS = 5.0


def report() -> dict:
    px = data.load_prices()
    rets = px.pct_change()
    panels = {"sectors": [c for c in data.SECTORS if rets[c].notna().sum() > 1500],
              "multi-asset": [c for c in data.MULTI if rets[c].notna().sum() > 3000]}
    h: dict = {"as_of": data.AS_OF, "window": WINDOW, "step": STEP,
               "fingerprint": data.fingerprint(px),
               "panels": {k: list(v) for k, v in panels.items()}}

    print(f"as-of {data.AS_OF}   window {WINDOW}d every {STEP}d   fp {data.fingerprint(px)}")
    for tag, cols in panels.items():
        sub = rets[cols].dropna(how="any")
        vols = sub.std() * np.sqrt(st.TRADING_DAYS)
        print(f"  {tag:12s} {len(cols):2d} assets, {len(sub):,} sessions, volatility "
              f"{vols.min():.1%} to {vols.max():.1%}")

    results, gaps = {}, {}
    for tag, cols in panels.items():
        sub = rets[cols].dropna(how="any")
        cov = np.cov(sub.iloc[-WINDOW:].to_numpy(), rowvar=False, ddof=1)
        w_md, w_mv = st.max_div_weights(cov), st.min_variance_weights(cov)
        gaps[tag] = float(np.abs(w_md - w_mv).sum() / 2)
        print(f"\n=== {tag} ===")
        print(f"  most-diversified vs minimum-variance: {gaps[tag]:.1%} of the book differs")
        print(f"  diversification ratio in sample: MDP {st.diversification_ratio(w_md, cov):.3f}, "
              f"min-var {st.diversification_ratio(w_mv, cov):.3f}, "
              f"1/N {st.diversification_ratio(st.equal_weights(cov), cov):.3f}")
        top = pd.Series(w_md, index=cols).sort_values(ascending=False)
        print("  MDP holds: " + ", ".join(f"{k} {v:.0%}" for k, v in top.head(5).items()))

        wf = st.walk_forward(sub, window=WINDOW, step=STEP, cost_bps=COST_BPS)
        s = st.summarise(wf)
        print("  method                          vol    return  Sharpe   DR in  DR out  "
              "slip   turnover  eff.N")
        for m, row in s.iterrows():
            print(f"  {st.METHOD_LABEL[m]:30s} {row['realised_vol']:6.2%} {row['mean_ret']:8.2%} "
                  f"{row['sharpe']:7.2f} {row['dr_in']:7.3f} {row['dr_out']:7.3f} "
                  f"{row['dr_slippage']:+6.1%} {row['turnover']:9.2f} {row['effective_n']:6.1f}")
        pairs = {o: st.paired_test(wf, "max_div", o)
                 for o in ("min_var", "inv_vol", "equal", "risk_parity")}
        for o, p in pairs.items():
            print(f"    MDP vs {st.METHOD_LABEL[o]:30s} vol diff {p['diff']:+.3%}  "
                  f"t {p['t']:+5.2f}  MDP quieter in {p['win_rate']:.0%} of {p['n']}")
        results[tag] = {"summary": {m: dict(v) for m, v in s.to_dict("index").items()},
                        "pairs": pairs, "weight_gap": gaps[tag],
                        "weights": dict(zip(cols, w_md.tolist()))}

    h["results"] = results
    h["weight_gap_sectors"] = gaps["sectors"]
    h["weight_gap_multi"] = gaps["multi-asset"]
    multi = results["multi-asset"]["summary"]
    h["dr_in_maxdiv"] = float(multi["max_div"]["dr_in"])
    h["dr_in_minvar"] = float(multi["min_var"]["dr_in"])
    h["dr_out_maxdiv"] = float(multi["max_div"]["dr_out"])
    h["dr_slippage_maxdiv"] = float(multi["max_div"]["dr_slippage"])
    h["vol_maxdiv"] = float(multi["max_div"]["realised_vol"])
    h["vol_invvol"] = float(multi["inv_vol"]["realised_vol"])
    h["vol_equal"] = float(multi["equal"]["realised_vol"])
    h["turnover_maxdiv"] = float(multi["max_div"]["turnover"])
    h["turnover_invvol"] = float(multi["inv_vol"]["turnover"])
    h["effective_n_maxdiv"] = float(multi["max_div"]["effective_n"])
    h["effective_n_minvar"] = float(multi["min_var"]["effective_n"])
    ts = [-results[t]["pairs"]["inv_vol"]["t"] for t in panels]
    h["best_t_vs_invvol"] = float(max(ts))
    h["beats_invvol_panels"] = int(sum(1 for t in ts if t > 0))

    print("\n=== the degenerate case: equal volatilities ===")
    d = st.degenerate_case(n=10, rho=0.35)
    print(f"  with identical variances the two objectives coincide; largest weight difference "
          f"{d['max_abs_diff']:.2e}")
    h["degenerate_gap"] = float(d["max_abs_diff"])

    print("\n=== the identity that prices the correlation matrix ===")
    sub_multi = rets[panels["multi-asset"]].dropna(how="any")
    vols_now = (sub_multi.iloc[-WINDOW:].std() * np.sqrt(st.TRADING_DAYS)).to_numpy()
    ident = st.equicorrelation_identity(vols_now)
    for rho, row in ident.iterrows():
        print(f"  constant correlation {rho:.1f}: |MDP - inverse volatility| = "
              f"{row['max_abs_diff']:.2e}  (DR {row['dr']:.3f})")
    print("  -> under EQUICORRELATION the most diversified portfolio IS inverse volatility, at "
          "every level. Everything the method earns comes from correlations that DIFFER.")
    h["equicorr_max_gap"] = float(ident["max_abs_diff"].max())
    disp = {tag: st.correlation_dispersion(
        np.cov(rets[cols].dropna(how="any").iloc[-WINDOW:].to_numpy(), rowvar=False, ddof=1))
        for tag, cols in panels.items()}
    for tag, v in disp.items():
        print(f"  correlation dispersion, {tag:12s}: {v:.3f} (weight gap vs inverse "
              f"volatility {np.abs(np.array(list(results[tag]['weights'].values())) - st.inverse_vol_weights(np.cov(rets[panels[tag]].dropna(how='any').iloc[-WINDOW:].to_numpy(), rowvar=False, ddof=1))).sum() / 2:.1%})")
    h["corr_dispersion"] = disp

    print("\n=== does the diversification ratio survive out of sample? ===")
    for tag in panels:
        s = results[tag]["summary"]
        for m in st.METHODS:
            print(f"  {tag:12s} {st.METHOD_LABEL[m]:30s} promised {s[m]['dr_in']:.3f} -> "
                  f"delivered {s[m]['dr_out']:.3f} ({s[m]['dr_slippage']:+.1%})")

    print("\n=== window sensitivity (multi-asset) ===")
    sens = []
    sub = rets[panels["multi-asset"]].dropna(how="any")
    for w in (126, 252, 504):
        s = st.summarise(st.walk_forward(sub, window=w, step=STEP, cost_bps=COST_BPS))
        sens.append({"window": w, **{m: float(s.loc[m, "realised_vol"]) for m in st.METHODS}})
        print(f"  {w:4d}d: " + "  ".join(f"{st.METHOD_LABEL[m][:14]:14s}"
                                         f"{s.loc[m, 'realised_vol']:6.2%}" for m in st.METHODS))
    h["window_sensitivity"] = sens

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
            f"{r['sharpe']:+.2f} | {r['dr_in']:.3f} | {r['dr_out']:.3f} | "
            f"{r['dr_slippage']:+.1%} | {r['turnover']:.2f} | {r['effective_n']:.1f} |"
            for m, r in s.items())
    def pairs(tag):
        return "\n".join(
            f"| MDP − {st.METHOD_LABEL[o]} | {p['diff']:+.3%} | {-p['t']:+.2f} | "
            f"{p['win_rate']:.0%} |" for o, p in h["results"][tag]["pairs"].items())
    sens = "\n".join("| " + str(r["window"]) + "d | " +
                     " | ".join(f"{r[m]:.2%}" for m in st.METHODS) + " |"
                     for r in h["window_sensitivity"])
    heads = " | ".join(st.METHOD_LABEL[m] for m in st.METHODS)
    return f"""# Results — Study 977 (Maximum Diversification) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). The most diversified portfolio
(Choueifaty & Coignard 2008) solved in closed form on the correlation matrix, raced against
minimum variance, inverse volatility, 1/N and equal risk contribution. Rolling
**{h['window']}-day** window every **{h['step']}** sessions, long-only, one day of execution
lag, 5 bps a rebalance. As-of **{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

## Is it a different portfolio?

Share of the book that differs from the minimum-variance solution:
**{h['weight_gap_multi']:.1%}** (multi-asset), **{h['weight_gap_sectors']:.1%}** (sectors). On a
panel of **equal-volatility** assets the two objectives are mathematically the same problem, and
the implementation confirms it: largest weight difference **{h['degenerate_gap']:.1e}**.

## Eleven sectors

| Method | Realised vol | Return | Sharpe | DR in-sample | DR out-of-sample | Slippage | Turnover | Effective N |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
{block('sectors')}

| Paired comparison | Vol difference | *t* (positive = MDP quieter) | MDP wins |
|---|--:|--:|--:|
{pairs('sectors')}

## Ten multi-asset sleeves

| Method | Realised vol | Return | Sharpe | DR in-sample | DR out-of-sample | Slippage | Turnover | Effective N |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
{block('multi-asset')}

| Paired comparison | Vol difference | *t* (positive = MDP quieter) | MDP wins |
|---|--:|--:|--:|
{pairs('multi-asset')}

## Window sensitivity (multi-asset, realised volatility)

| Window | {heads} |
|---|--:|--:|--:|--:|--:|
{sens}

## The identity that prices the correlation matrix

Under a **constant** correlation matrix the closed form collapses: `C⁻¹1 ∝ 1`, so
`w ∝ 1/σ` and the most diversified portfolio *is* inverse-volatility weighting — at every
level of correlation. Checked at ρ = 0.0, 0.2, 0.5 and 0.8 on this panel's volatilities, the
largest weight difference is **{h['equicorr_max_gap']:.1e}**.

Everything the method earns therefore comes from correlations that **differ from one
another**, not from their level. Measured dispersion of the off-diagonal correlations:
{", ".join(f"{tag} **{v:.3f}**" for tag, v in h["corr_dispersion"].items())}. That number, not
the average correlation, is the honest predictor of how far the MDP can stray from the free
alternative.

## What the slippage column means

Every method's diversification ratio is computed twice: once on the covariance matrix it was
optimised on, and once on the matrix that actually materialised over the holding period. The
most diversified portfolio maximises the first by construction; the gap to the second is the
part of the objective that was estimation error. On this desk's convention that column is the
honest measure of whether an optimised quantity is a property of the market or of the sample.

## Caveats

- **Long-only throughout.** Choueifaty's construction is defined for long-only books and the
  constraint is doing real work here (Jagannathan & Ma 2003); an unconstrained MDP is a
  different, wilder object.
- **Sample covariance.** No shrinkage is applied, so part of every method's out-of-sample
  slippage is an estimator problem this desk already knows how to reduce — study **975**.
- **No factor story.** The MDP has an interpretation as a maximum-Sharpe portfolio under the
  assumption that expected returns are proportional to volatility. That assumption is not
  tested here, and it is the whole economic case for the method.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Usefulness: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[977-max-diversification](../README.md). Not investment advice.*
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
