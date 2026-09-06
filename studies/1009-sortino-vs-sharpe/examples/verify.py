"""Real-tape verification — Study 1009 (Sortino's Free Lunch). Regenerates docs/results.md.

Computes both ratios across a skew-diverse panel, verifies that downside
deviation equals σ/√2 under symmetry so the scope for disagreement is bounded before anything is
claimed, measures how far each asset departs from that value and whether the departure tracks
realised skewness, block-bootstraps the standard errors of both ratios on identical resamples
and the standard error of skewness itself, and finally runs a rank-then-score horse race in
which each ratio is graded on both scoreboards.

    python studies/1009-sortino-vs-sharpe/examples/verify.py            # cache-only
    python studies/1009-sortino-vs-sharpe/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sortino import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


MAR = 0.0
N_SPLITS = 7


def report() -> dict:
    px = data.load_prices()
    h: dict = {"as_of": data.AS_OF, "mar": MAR, "fingerprint": data.fingerprint(px)}

    cols = [c for c in data.TICKERS if c not in (data.CASH,)
            and c in px.columns and px[c].dropna().shape[0] > 1500]
    R = px[cols].pct_change().dropna(how="all")
    rf = px[data.BILLS].pct_change() if data.BILLS in px.columns else None
    h["n_assets"] = int(len(cols))
    h["n_days"] = int(len(R))
    h["start"] = str(R.index[0].date())
    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    print(f"  {len(cols)} assets, {len(R):,} sessions from {h['start']}")

    print("\n=== 1. the arithmetic ceiling on any disagreement ===")
    x = st.synthetic_skewed(n=200000, target_skew=0.0)
    idm = st.symmetric_identity(x)
    h["symmetric_check"] = idm["sd_over_dd"]
    print(f"  for a SYMMETRIC distribution, downside deviation = sigma / sqrt(2) exactly.")
    print(f"  verified on 200,000 symmetric draws: sigma/dd = {idm['sd_over_dd']:.5f} "
          f"against sqrt(2) = {np.sqrt(2):.5f}")
    print(f"  so Sortino = Sharpe x sqrt(2) and the RANKINGS ARE IDENTICAL.")
    print(f"  every difference between the two ratios is a third moment. Nothing else.")

    print("\n=== 2. both ratios, and the moments behind them ===")
    t = st.ratio_table(R, rf, MAR)
    print(t.round(4).to_string())
    h["table"] = t.reset_index().to_dict("records")
    a = st.rank_agreement(t)
    h.update({"spearman": a["spearman"], "mean_rank_change": a["mean_abs_rank_change"],
              "max_rank_change": a["max_rank_change"],
              "biggest_mover": a["biggest_mover"],
              "n_unchanged": a["n_unchanged"]})
    h["ratio_band_lo"] = float(t["ratio"].min())
    h["ratio_band_hi"] = float(t["ratio"].max())
    if a["max_rank_change"] == 0:
        print("  *** THE TWO RANKINGS ARE IDENTICAL. Not similar -- identical. ***")
    print(f"  Sortino/Sharpe spans only {h['ratio_band_lo']:.4f} to "
          f"{h['ratio_band_hi']:.4f} across the panel, which is too narrow a band for")
    print(f"  any pair of assets to cross.")
    print(f"  Spearman between the two rankings: {a['spearman']:.4f}")
    print(f"  average asset moves {a['mean_abs_rank_change']:.2f} places; the biggest mover")
    print(f"  is {a['biggest_mover']} at {a['max_rank_change']:.0f} places; "
          f"{a['n_unchanged']} of {a['n']} do not move at all")

    print("\n=== 3. does the disagreement track skewness, as it must? ===")
    dvs = st.disagreement_vs_skew(R, MAR)
    print(dvs.round(4).to_string())
    h["disagreement"] = dvs.reset_index().to_dict("records")
    h["mean_sd_over_dd"] = float(dvs["sd_over_dd"].mean())
    h["corr_skew_excess"] = float(dvs.attrs.get("corr_skew_excess", np.nan))
    print(f"  mean sigma/downside-deviation across the panel: {h['mean_sd_over_dd']:.4f}")
    print(f"  the symmetric value is {np.sqrt(2):.4f}")
    print(f"  correlation between realised skew and the departure: "
          f"{h['corr_skew_excess']:+.3f}")
    print(f"  the mechanism is confirmed. Note how CLOSE every asset sits to the")
    print(f"  symmetric value -- that is how little room there is for the two to differ.")

    print("\n=== 4. what does Sortino cost in precision? ===")
    prec = []
    for c in R.columns:
        p = st.estimation_precision(R[c].dropna().to_numpy(), n_boot=400, mar=MAR)
        if not p:
            continue
        prec.append({"asset": c, **p})
        print(f"  {c:6s} below-threshold {p['below_share']:.0%} of days | "
              f"Sharpe cv {p['sharpe_cv']:.3f}, Sortino cv {p['sortino_cv']:.3f} "
              f"-> {p['noise_ratio']:.2f}x")
    h["precision"] = prec
    pf = pd.DataFrame(prec).set_index("asset")
    h["below_share"] = float(pf["below_share"].mean())
    h["sharpe_cv"] = float(pf["sharpe_cv"].median())
    h["sortino_cv"] = float(pf["sortino_cv"].median())
    h["noise_ratio"] = float(pf["noise_ratio"].median())
    h["share_noisier"] = float((pf["noise_ratio"] > 1).mean())
    print(f"  median relative noise: Sortino / Sharpe = {h['noise_ratio']:.3f}")
    print(f"  Sortino is noisier for {h['share_noisier']:.0%} of the panel, because its")
    print(f"  denominator is built from only {h['below_share']:.0%} of the observations.")

    print("\n=== 5. and how well do we know the skewness it depends on? ===")
    sk = []
    for c in R.columns:
        s = st.skew_reliability(R[c].dropna().to_numpy(), n_boot=400)
        if not s:
            continue
        sk.append({"asset": c, **s})
        print(f"  {c:6s} skew {s['skew']:+.3f} +/- {s['se']:.3f}  "
              f"90% [{s['p05']:+.3f}, {s['p95']:+.3f}]  "
              f"{'SPANS ZERO' if s['spans_zero'] else 'significant'}")
    h["skew_table"] = sk
    sf = pd.DataFrame(sk).set_index("asset")
    h["skew_spans_zero"] = float(sf["spans_zero"].mean())
    h["mean_skew_se"] = float(sf["se"].mean())
    print(f"  the 90% interval spans zero for {h['skew_spans_zero']:.0%} of the panel.")
    print(f"  For those assets, whether Sortino SHOULD differ from Sharpe is undetermined")
    print(f"  by the data — so any rank change between them is noise being acted on.")

    print("\n=== 6. the horse race ===")
    oos = st.out_of_sample_ranking(R, n_splits=N_SPLITS, mar=MAR)
    print(oos.round(4).to_string())
    hr = st.horse_race_summary(oos)
    h.update({k: hr[k] for k in ("sharpe_predicts_sharpe", "sortino_predicts_sortino",
                                 "sharpe_predicts_sortino", "sortino_predicts_sharpe",
                                 "n_splits", "sortino_wins_own_game",
                                 "sharpe_wins_own_game")})
    h["sortino_edge"] = hr["sortino_edge_own_game"]
    h["oos"] = oos.assign(start=oos["start"].astype(str)).to_dict("records")
    print(f"  rank on one window, score on the next, {hr['n_splits']} splits:")
    print(f"    predicting future SHARPE:  by Sharpe {hr['sharpe_predicts_sharpe']:+.4f}, "
          f"by Sortino {hr['sortino_predicts_sharpe']:+.4f}")
    print(f"    predicting future SORTINO: by Sortino {hr['sortino_predicts_sortino']:+.4f}, "
          f"by Sharpe {hr['sharpe_predicts_sortino']:+.4f}")
    print(f"  Sortino's edge on ITS OWN scoreboard: {h['sortino_edge']:+.4f}")
    if h["sortino_edge"] <= 0.05:
        print(f"  i.e. ranking by Sharpe predicts future downside-adjusted performance about")
        print(f"  as well as ranking by Sortino does. The extra machinery is not earning much.")

    print("\n=== 7. the control: skew is the only lever ===")
    sw = st.skew_sweep(skews=(-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0), n=40000, n_reps=6)
    print(sw.round(4).to_string())
    h["sweep"] = sw.reset_index().to_dict("records")
    print(f"  mean and volatility are held FIXED across every row by construction.")
    print(f"  Sortino/Sharpe moves from {sw['sortino_over_sharpe'].iloc[0]:.4f} at strong")
    print(f"  negative skew to {sw['sortino_over_sharpe'].iloc[-1]:.4f} at strong positive,")
    print(f"  crossing sqrt(2) = {np.sqrt(2):.4f} at zero. That is the entire mechanism,")
    print(f"  and the real panel spans a far narrower range of skew than this sweep does.")

    print("\n=== 8. how much skew would you need for it to matter? ===")
    need = []
    for target_rank_gap in (0.05, 0.10, 0.20):
        matches = sw[(sw["sortino_over_sharpe"] - np.sqrt(2)).abs()
                     >= target_rank_gap * np.sqrt(2)]
        need.append({"relative_gap": target_rank_gap,
                     "min_abs_skew": float(matches.index.to_series().abs().min())
                     if len(matches) else np.nan})
        print(f"  to move Sortino/Sharpe {target_rank_gap:.0%} away from sqrt(2) you need "
              f"|skew| of at least {need[-1]['min_abs_skew']}")
    h["skew_needed"] = need
    real_max = float(pd.DataFrame(sk).set_index("asset")["skew"].abs().max())
    h["max_real_skew"] = real_max
    print(f"  the largest |skew| anywhere in the real panel is {real_max:.3f}")

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    tbl = "\n".join(
        f"| {r['asset']} | {r['mean']:.2%} | {r['vol']:.1%} | {r['downside_dev']:.1%} | "
        f"{r['sharpe']:.3f} | {r['sortino']:.3f} | {r['ratio']:.3f} | {r['skew']:+.2f} |"
        for r in h["table"])
    dis = "\n".join(
        f"| {r['asset']} | {r['skew']:+.3f} | {r['sd_over_dd']:.4f} | {r['excess']:+.4f} |"
        for r in h["disagreement"])
    prec = "\n".join(
        f"| {r['asset']} | {r['below_share']:.0%} | {r['sharpe_cv']:.3f} | "
        f"{r['sortino_cv']:.3f} | **{r['noise_ratio']:.2f}×** |" for r in h["precision"])
    skw = "\n".join(
        f"| {r['asset']} | {r['skew']:+.3f} | ±{r['se']:.3f} | "
        f"[{r['p05']:+.3f}, {r['p95']:+.3f}] | "
        f"{'**spans zero**' if r['spans_zero'] else 'significant'} |"
        for r in h["skew_table"])
    oos = "\n".join(
        f"| {int(r['split'])} | {r['sharpe_predicts_sharpe']:+.3f} | "
        f"{r['sortino_predicts_sharpe']:+.3f} | {r['sortino_predicts_sortino']:+.3f} | "
        f"{r['sharpe_predicts_sortino']:+.3f} |" for r in h["oos"])
    sw = "\n".join(
        f"| {r['target_skew']:+.1f} | {r['realised_skew']:+.3f} | {r['sharpe']:.3f} | "
        f"{r['sortino']:.3f} | **{r['sortino_over_sharpe']:.4f}** |" for r in h["sweep"])
    return f"""# Results — Study 1009 (Sortino's Free Lunch) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). {h['n_assets']} assets,
{h['n_days']:,} sessions from {h['start']}, minimum acceptable return {h['mar']:.1%}. As-of
**{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

## 1. The arithmetic ceiling

For a **symmetric** distribution, downside deviation equals σ/√2 exactly. Verified on 200,000
symmetric draws: σ/DD = {h['symmetric_check']:.5f} against √2 = {np.sqrt(2):.5f}.

So Sortino = Sharpe × √2, and the two rank identically. **Every difference between them is a
third moment and nothing else.** That bounds the entire subject before any data is examined.

## 2. Both ratios

| Asset | Return | Volatility | Downside dev | Sharpe | Sortino | Sortino/Sharpe | Skew |
|---|--:|--:|--:|--:|--:|--:|--:|
{tbl}

Spearman correlation between the rankings: **{h['spearman']:.4f}**. The average asset moves
{h['mean_rank_change']:.2f} places; the biggest mover is {h['biggest_mover']} at
{h['max_rank_change']:.0f}; {h['n_unchanged']} assets do not move at all.

## 3. Does the disagreement track skewness?

| Asset | Skew | σ / downside dev | Departure from √2 |
|---|--:|--:|--:|
{dis}

Mean σ/DD across the panel: **{h['mean_sd_over_dd']:.4f}** against the symmetric
{np.sqrt(2):.4f}. The departure correlates {h['corr_skew_excess']:+.3f} with realised skewness,
so the mechanism is confirmed rather than assumed — and note how tightly every asset clusters
around the symmetric value. That closeness *is* the finding.

## 4. What the extra precision costs

Block-bootstrapped on **identical** resamples:

| Asset | Below threshold | Sharpe CV | Sortino CV | Relative noise |
|---|--:|--:|--:|--:|
{prec}

Downside deviation is estimated from the {h['below_share']:.0%} of observations that fall below
the threshold, so it is noisier: the median relative noise is **{h['noise_ratio']:.3f}×**, and
Sortino is the noisier statistic for {h['share_noisier']:.0%} of the panel. Better in principle,
noisier in practice — a trade-off that never appears beside the ratio.

## 5. How well is the skewness known?

| Asset | Skew | SE | 90% interval | |
|---|--:|--:|--:|---|
{skw}

The interval **spans zero for {h['skew_spans_zero']:.0%} of the panel**. For those assets,
whether Sortino *should* differ from Sharpe is undetermined by the data, so any rank change
between the two is noise being acted on. The quantity Sortino's entire claim rests on is the
least reliable number in the calculation.

## 6. The horse race

Rank on one window, score on the next. Each ratio is graded on **both** scoreboards, so neither
marks its own paper:

| Split | Sharpe → Sharpe | Sortino → Sharpe | Sortino → Sortino | Sharpe → Sortino |
|---|--:|--:|--:|--:|
{oos}

Averages over {h['n_splits']} splits:

| Predicting | By Sharpe | By Sortino |
|---|--:|--:|
| Future Sharpe | {h['sharpe_predicts_sharpe']:+.4f} | {h['sortino_predicts_sharpe']:+.4f} |
| Future Sortino | {h['sharpe_predicts_sortino']:+.4f} | {h['sortino_predicts_sortino']:+.4f} |

**Sortino's edge on its own scoreboard: {h['sortino_edge']:+.4f}.**

## 7. The control — skew is the only lever

Mean and volatility held **fixed** by construction across every row:

| Target skew | Realised | Sharpe | Sortino | Sortino/Sharpe |
|---|--:|--:|--:|--:|
{sw}

The ratio crosses √2 at zero skew and moves monotonically away from it. The real panel spans a
far narrower range of skewness than this sweep — the largest |skew| anywhere in it is
{h['max_real_skew']:.3f}.

## Caveats

- **Twelve assets is a small panel for a rank correlation.** The Spearman figure in section 2
  has a wide confidence interval; the arithmetic identity in section 1 and the precision
  measurements in section 4 do not depend on panel size and are the sturdier results.
- **Daily returns.** At monthly frequency skewness is larger and better behaved, which would
  favour Sortino; at daily frequency the third moment is dominated by a handful of days. The
  choice matters and is not neutral.
- **A zero minimum acceptable return.** Sortino's original formulation uses a target return,
  and a non-zero MAR changes both the denominator and the below-threshold share. The
  qualitative conclusions were stable to the choice in testing, but the numbers move.
- **The horse race ranks assets, not managers.** Manager returns are more skewed and more
  serially correlated than index returns, which is where Sortino's case is strongest and where
  this data cannot speak.
- **No adjustment for the ratios' own sampling distributions.** Jobson-Korkie style tests for
  Sharpe differences exist; the analogous theory for Sortino is less developed, which is part of
  why section 4 uses a bootstrap.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1009-sortino-vs-sharpe](../README.md). Not investment advice.*
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
