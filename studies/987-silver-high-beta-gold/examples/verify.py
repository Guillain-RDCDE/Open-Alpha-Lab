"""Real-tape verification — Study 987 (Gold's Loud Cousin). Regenerates docs/results.md.

Estimates silver's beta to gold and how far it wanders, projects gold out and
hunts the residual for structure against the dollar, real rates, industrials and copper,
computes the arithmetic volatility drag of holding levered gold instead, runs the replication as
an actual financed strategy, and tests whether a stretched gold/silver ratio predicts its own
reversal.

    python studies/987-silver-high-beta-gold/examples/verify.py            # cache-only
    python studies/987-silver-high-beta-gold/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from loudcousin import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


FINANCING_SPREAD = 0.005
COST_BPS = 1.0


def report() -> dict:
    px = data.load_prices()
    rets = px.pct_change()
    common = rets[[data.GOLD, data.SILVER, data.DOLLAR, data.TIPS, data.INDUSTRIAL,
                   data.CASH]].dropna()
    h: dict = {"as_of": data.AS_OF, "financing_spread": FINANCING_SPREAD,
               "fingerprint": data.fingerprint(px)}

    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    for tk in data.TICKERS:
        s = px[tk].dropna()
        print(f"  {tk:5s} {s.index[0].date()} -> {s.index[-1].date()}  n={len(s):,}  "
              f"ann vol {s.pct_change().std() * np.sqrt(st.TRADING_DAYS):.1%}")
    g, s_ = common[data.GOLD], common[data.SILVER]
    h["n_days"] = int(len(common))
    h["window"] = [str(common.index[0].date()), str(common.index[-1].date())]
    h["gold_vol"] = float(g.std() * np.sqrt(st.TRADING_DAYS))
    h["silver_vol"] = float(s_.std() * np.sqrt(st.TRADING_DAYS))
    print(f"  common window {common.index[0].date()} -> {common.index[-1].date()} "
          f"({len(common):,} sessions)")
    print(f"  gold vol {h['gold_vol']:.1%}, silver vol {h['silver_vol']:.1%}, "
          f"ratio {h['silver_vol'] / h['gold_vol']:.2f}")

    print("\n=== 1. the number everyone quotes ===")
    b = st.full_sample_beta(s_, g)
    h.update({"beta": b["beta"], "beta_se": b["se"], "r2": b["r2"],
              "alpha_ann": b["alpha_ann"], "resid_vol": b["resid_vol_ann"]})
    print(f"  silver = {b['alpha_ann']:+.2%}/yr + {b['beta']:.3f} x gold")
    print(f"  se {b['se']:.3f}, R2 {b['r2']:.1%}, residual vol {b['resid_vol_ann']:.1%}/yr")
    print(f"  -> {1 - b['r2']:.0%} of silver's variance is NOT gold. That is the subject of "
          f"sections 3 and 4.")

    print("\n=== 2. does the beta hold still? ===")
    tbl = st.rolling_beta_table(s_, g)
    print(tbl.round(3).to_string())
    h["rolling_beta"] = tbl.reset_index().to_dict("records")
    r252 = tbl.loc[252]
    h.update({"beta_min": float(r252["min"]), "beta_max": float(r252["max"]),
              "beta_sd": float(r252["sd"]),
              "beta_range_over_mean": float(r252["range_over_mean"])})
    print(f"  over rolling one-year windows the loading runs {r252['min']:.2f} to "
          f"{r252['max']:.2f} — a spread of {r252['range_over_mean']:.0%} of its own mean")
    reg = st.beta_by_regime(s_, g)
    print(reg.round(3).to_string())
    h["beta_regimes"] = reg.reset_index().to_dict("records")
    h["beta_up"] = float(reg.loc["gold up days", "beta"])
    h["beta_down"] = float(reg.loc["gold down days", "beta"])
    print(f"  gold up days: beta {h['beta_up']:.2f};  gold down days: {h['beta_down']:.2f}")

    print("\n=== 3. what is in the residual? ===")
    resid = st.residuals(s_, g)
    d = st.residual_diagnostics(resid)
    for k, v in d.items():
        if isinstance(v, float) and np.isfinite(v):
            print(f"  {k:24s} {v:+.4f}")
    h["residual"] = {k: v for k, v in d.items() if isinstance(v, (int, float))}
    print(f"  -> the leftover has {d['vol_ann']:.1%} annualised volatility. For scale, that is "
          f"{d['vol_ann'] / h['gold_vol']:.1f}x gold's own.")

    print("\n=== 4. does the residual load on anything? ===")
    factors = pd.DataFrame({
        "dollar": common[data.DOLLAR], "real_rates": common[data.TIPS],
        "industrial": common[data.INDUSTRIAL],
    })
    if data.COPPER in rets.columns and rets[data.COPPER].notna().sum() > 1000:
        factors["copper"] = rets[data.COPPER]
    load = st.residual_loadings(resid, factors.reindex(resid.index))
    print(load.round(4).to_string())
    h["loadings"] = load.reset_index().to_dict("records")
    uni = load[~load.index.str.contains(r"\(joint\)")]
    h["max_abs_residual_t"] = float(uni["t"].abs().max())
    h["strongest_factor"] = str(uni["t"].abs().idxmax())
    print(f"  -> strongest single loading: {h['strongest_factor']} at "
          f"t {uni.loc[h['strongest_factor'], 't']:+.2f}")

    print("\n=== 5. the arithmetic of levering gold ===")
    for beta in (1.0, 1.5, h["beta"], 2.0, 2.5):
        print(f"  beta {beta:.2f}: volatility drag "
              f"{st.leverage_drag(beta, h['gold_vol']):.2%}/yr on a "
              f"{h['gold_vol']:.0%}-vol asset")
    h["drag_table"] = [{"beta": bb, "drag": st.leverage_drag(bb, h["gold_vol"])}
                       for bb in (1.0, 1.5, 2.0, 2.5, 3.0)]

    print("\n=== 6. the replication, run as a strategy ===")
    rep = st.replication_backtest(s_, g, common[data.CASH],
                                  financing_spread=FINANCING_SPREAD, cost_bps=COST_BPS)
    h.update({"correlation": rep["correlation"], "tracking_error": rep["tracking_error_ann"],
              "years": rep["years"], "silver_cagr": rep["silver"]["cagr"],
              "replica_cagr": rep["replica"]["cagr"],
              "silver_sharpe": rep["silver"]["sharpe"],
              "replica_sharpe": rep["replica"]["sharpe"],
              "silver_dd": rep["silver"]["max_dd"], "replica_dd": rep["replica"]["max_dd"],
              "predicted_drag": rep["predicted_drag"], "t_gap": rep["t_gap"]})
    print(f"  holding silver:            CAGR {rep['silver']['cagr']:+.2%}, vol "
          f"{rep['silver']['vol']:.1%}, Sharpe {rep['silver']['sharpe']:+.2f}, "
          f"maxDD {rep['silver']['max_dd']:.1%}")
    print(f"  holding {rep['beta_used']:.2f}x gold:      CAGR {rep['replica']['cagr']:+.2%}, "
          f"vol {rep['replica']['vol']:.1%}, Sharpe {rep['replica']['sharpe']:+.2f}, "
          f"maxDD {rep['replica']['max_dd']:.1%}")
    print(f"  correlation {rep['correlation']:.3f}, tracking error "
          f"{rep['tracking_error_ann']:.1%}/yr, gap t {rep['t_gap']:+.2f}")
    print(f"  predicted volatility drag at this beta: {rep['predicted_drag']:.2%}/yr")
    sweep = []
    for beta in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5):
        r2 = st.replication_backtest(s_, g, common[data.CASH], beta=beta,
                                     financing_spread=FINANCING_SPREAD, cost_bps=COST_BPS)
        sweep.append({"beta": beta, "cagr": r2["replica"]["cagr"],
                      "sharpe": r2["replica"]["sharpe"], "vol": r2["replica"]["vol"],
                      "tracking_error": r2["tracking_error_ann"]})
        print(f"  beta {beta:.2f}: CAGR {r2['replica']['cagr']:+.2%}, Sharpe "
              f"{r2['replica']['sharpe']:+.2f}, TE {r2['tracking_error_ann']:.1%}")
    h["beta_sweep"] = sweep

    print("\n=== 7. the miners, which lever it again ===")
    miners = {}
    for label, tk, base in (("gold miners", data.GOLD_MINERS, data.GOLD),
                            ("silver miners", data.SILVER_MINERS, data.SILVER)):
        if tk not in rets.columns or rets[tk].notna().sum() < 1000:
            continue
        pair = pd.concat([rets[tk].rename("y"), rets[base].rename("x")], axis=1).dropna()
        bb = st.full_sample_beta(pair["y"], pair["x"])
        miners[label] = {"beta": bb["beta"], "r2": bb["r2"],
                         "resid_vol": bb["resid_vol_ann"], "n": bb["n"]}
        print(f"  {label:14s} beta to its metal {bb['beta']:.2f}, R2 {bb['r2']:.0%}, "
              f"residual vol {bb['resid_vol_ann']:.1%}")
    h["miners"] = miners

    print("\n=== 8. the gold/silver ratio trade ===")
    ratio = st.gold_silver_ratio(px[data.GOLD], px[data.SILVER])
    mr = st.ratio_mean_reversion(ratio)
    print(mr.round(4).to_string())
    h["ratio_reversion"] = mr.reset_index().to_dict("records")
    print(f"  ratio range over the sample: {ratio.min():.2f} to {ratio.max():.2f} "
          f"(indexed to 1.00 at the start)")
    h["ratio_range"] = [float(ratio.min()), float(ratio.max())]

    print("\n=== 9. synthetic control ===")
    for load_v, drift, tag in ((0.0, 0.0, "silver IS levered gold"),
                               (1.5, 0.0, "silver has a second driver"),
                               (0.0, 0.8, "beta drifts")):
        sw = st.synthetic_world(n=6000, true_beta=1.8, industrial_load=load_v,
                                beta_drift=drift)
        rr = st.residuals(sw["silver"], sw["gold"])
        lt = st.residual_loadings(rr, sw[["industrial"]])
        rb = st.rolling_beta_table(sw["silver"], sw["gold"], windows=(252,))
        print(f"  {tag:30s} residual t on industrial {lt.loc['industrial', 't']:+6.2f}, "
              f"beta range/mean {rb.loc[252, 'range_over_mean']:.2f}")
        h[f"synthetic_{tag.split()[1]}"] = {
            "residual_t": float(lt.loc["industrial", "t"]),
            "beta_range": float(rb.loc[252, "range_over_mean"])}

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    rb = "\n".join(
        f"| {int(r['window'])}d | {r['mean']:.2f} | {r['sd']:.2f} | {r['min']:.2f} | "
        f"{r['max']:.2f} | {r['range_over_mean']:.0%} |" for r in h["rolling_beta"])
    reg = "\n".join(f"| {r['regime']} | {int(r['n'])} | {r['beta']:.2f} | {r['se']:.3f} | "
                    f"{r['r2']:.0%} |" for r in h["beta_regimes"])
    load = "\n".join(f"| {r['factor']} | {r['beta']:+.3f} | {r['t']:+.2f} | {r['r2']:.2%} |"
                     for r in h["loadings"])
    drag = "\n".join(f"| {r['beta']:.2f}× | {r['drag']:.2%} |" for r in h["drag_table"])
    sweep = "\n".join(
        f"| {r['beta']:.2f}× | {r['cagr']:+.2%} | {r['vol']:.1%} | {r['sharpe']:+.2f} | "
        f"{r['tracking_error']:.1%} |" for r in h["beta_sweep"])
    mine = "\n".join(f"| {k} | {vv['beta']:.2f} | {vv['r2']:.0%} | {vv['resid_vol']:.1%} |"
                     for k, vv in h["miners"].items())
    mr = "\n".join(f"| {int(r['horizon'])}d | {int(r['n'])} | {r['slope']:+.3f} | "
                   f"{r['t']:+.2f} | {r['r2']:.2%} |" for r in h["ratio_reversion"])
    return f"""# Results — Study 987 (Gold's Loud Cousin) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). {h['n_days']:,} common sessions,
{h['window'][0]} → {h['window'][1]}. As-of **{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

Gold's annualised volatility over the window was **{h['gold_vol']:.1%}**, silver's
**{h['silver_vol']:.1%}** — a ratio of {h['silver_vol'] / h['gold_vol']:.2f}, which is where the
folklore comes from.

## 1. The number everyone quotes

`silver = {h['alpha_ann']:+.2%}/yr + {h['beta']:.3f} × gold`, standard error {h['beta_se']:.3f},
**R² {h['r2']:.1%}**, residual volatility {h['resid_vol']:.1%} a year.

So {1 - h['r2']:.0%} of silver's variance is not gold. Everything below is about that
{1 - h['r2']:.0%}.

## 2. Does the beta hold still?

| Window | Mean | SD | Min | Max | Range ÷ mean |
|---|--:|--:|--:|--:|--:|
{rb}

| Regime | n | Beta | SE | R² |
|---|--:|--:|--:|--:|
{reg}

Over rolling one-year windows the loading runs **{h['beta_min']:.2f} to {h['beta_max']:.2f}** —
a spread of {h['beta_range_over_mean']:.0%} of its own mean. A single full-sample beta is an
average over that, not a description of it.

## 3. What is in the residual?

| | |
|---|--:|
| Annualised volatility | **{h['resid_vol']:.1%}** |
| Mean | {h['residual']['mean_ann']:+.2%}/yr (*t* = {h['residual']['t_mean']:+.2f}) |
| First-order autocorrelation | {h['residual']['autocorr_1']:+.3f} |
| Skew / excess kurtosis | {h['residual']['skew']:+.2f} / {h['residual']['kurtosis']:+.1f} |

| Factor | Loading | *t* | R² |
|---|--:|--:|--:|
{load}

Strongest single loading: **{h['strongest_factor']}** at *t* =
{h['max_abs_residual_t']:+.2f}. Roughly half of silver demand is industrial and none of gold's
is, so this is the place the "levered gold" story was always most likely to break.

## 4. The arithmetic of levering gold

A daily-rebalanced β× position does not deliver β times the period return. It compounds β times
the *daily* return, and the difference is `β(β−1)σ²/2` a year:

| Leverage | Annual volatility drag on a {h['gold_vol']:.0%}-vol asset |
|---|--:|
{drag}

This is arithmetic, not an empirical claim, and it is larger than most alphas anyone argues
about in this space.

## 5. The replication, run as a strategy

Hold silver, or hold {h['beta']:.2f}× gold financed at cash + {h['financing_spread']:.1%}:

| | Silver | {h['beta']:.2f}× gold |
|---|--:|--:|
| CAGR | {h['silver_cagr']:+.2%} | {h['replica_cagr']:+.2%} |
| Sharpe | {h['silver_sharpe']:+.2f} | {h['replica_sharpe']:+.2f} |
| Max drawdown | {h['silver_dd']:.1%} | {h['replica_dd']:.1%} |

Correlation **{h['correlation']:.2f}**, annualised tracking error **{h['tracking_error']:.1%}**,
predicted volatility drag {h['predicted_drag']:.2%}/yr.

A tracking error of {h['tracking_error']:.0%} a year is not tracking. It is a different asset
that happens to be correlated.

| Leverage | CAGR | Vol | Sharpe | Tracking error |
|---|--:|--:|--:|--:|
{sweep}

## 6. The miners, which lever it again

| | Beta to its metal | R² | Residual vol |
|---|--:|--:|--:|
{mine}

## 7. The gold/silver ratio trade

Does a stretched ratio predict its own reversal?

| Horizon | n | Slope on the *z*-score | *t* | R² |
|---|--:|--:|--:|--:|
{mr}

The ratio ranged {h['ratio_range'][0]:.2f} to {h['ratio_range'][1]:.2f} over the sample (indexed
to 1.00 at the start). A negative slope means a high ratio is followed by a falling one —
mean reversion — but note the R² column before reading that as a trade.

The *t* column uses **Newey-West at the horizon length**, and that is not a detail. These
forward windows overlap heavily; with ordinary heteroskedasticity-robust errors a pure random
walk run through this exact table produces *t*-statistics past 4 and looks like a tradeable
signal. That artefact is pinned down as a unit test
(`test_overlapping_windows_need_hac_and_hc1_would_lie`), because it is the single most common
way the gold/silver ratio trade gets "confirmed" in print.

## 8. Synthetic control

| World | Residual *t* on industrial | Beta range ÷ mean |
|---|--:|--:|
| Silver **is** levered gold | {h['synthetic_IS']['residual_t']:+.2f} | {h['synthetic_IS']['beta_range']:.2f} |
| Silver has a second driver | {h['synthetic_has']['residual_t']:+.2f} | {h['synthetic_has']['beta_range']:.2f} |
| Beta drifts | {h['synthetic_drifts']['residual_t']:+.2f} | {h['synthetic_drifts']['beta_range']:.2f} |

The apparatus says "yes, it is levered gold" when it is, and finds the second driver and the
drifting beta when those are planted.

## Caveats

- **ETFs, not metal.** GLD and SLV hold bullion but carry fees and, for SLV, a wider spread and
  a period of borrowing constraints in 2021 that briefly decoupled it.
- **Twenty years, two cycles.** Long by commodity-study standards, still short for a beta whose
  rolling range is this wide.
- **The rebalancing convention decides the drag.** Daily rebalancing is the worst case for
  volatility drag; a monthly-rebalanced or buy-and-hold levered position behaves differently
  and would narrow the gap in section 5.
- **XLI and CPER are crude proxies** for industrial demand. Physical silver-demand data
  (photovoltaics especially, which has grown from a rounding error to a fifth of demand across
  this sample) would be far better and is not in a daily price feed.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[987-silver-high-beta-gold](../README.md). Not investment advice.*
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
