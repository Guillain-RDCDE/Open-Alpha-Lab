"""Real-tape verification — Study 982 (The Appetite Gauge). Regenerates docs/results.md.

Measures how much of the high-beta / low-volatility spread is simply market
exposure, builds a beta-neutral version of it, races both against the market's own trend in
univariate and multiple regressions with horizon-lag HAC errors, prices the risk-on/risk-off
rule, and checks the gauge's behaviour in each of the sample's actual drawdowns.

    python studies/982-risk-appetite-ratio/examples/verify.py            # cache-only
    python studies/982-risk-appetite-ratio/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from appetite import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


LOOKBACK = 63
HORIZON = 21
COST_BPS = 2.0


def report() -> dict:
    px = data.load_prices()
    rets = st.to_returns(px)
    common = rets[list(data.TICKERS)].dropna()
    h: dict = {"as_of": data.AS_OF, "lookback": LOOKBACK, "horizon": HORIZON,
               "fingerprint": data.fingerprint(px)}

    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    for tk in data.TICKERS:
        s = px[tk].dropna()
        print(f"  {tk:5s} {s.index[0].date()} -> {s.index[-1].date()}  n={len(s):,}  "
              f"ann vol {s.pct_change().std() * np.sqrt(st.TRADING_DAYS):.1%}")
    years = len(common) / st.TRADING_DAYS
    h["years"] = float(years)
    h["window"] = [str(common.index[0].date()), str(common.index[-1].date())]
    h["n_common"] = int(len(common))
    print(f"  common window {common.index[0].date()} -> {common.index[-1].date()}: "
          f"{len(common):,} sessions ({years:.1f} years) — the binding constraint on every "
          f"t-statistic below")

    print("\n=== 1. how much of the gauge is just the market? ===")
    sb = st.spread_beta(common, data.HIGH_BETA, data.LOW_VOL, data.MARKET)
    print(f"  beta of {data.HIGH_BETA} on {data.MARKET}: {sb['beta_high']:+.2f}")
    print(f"  beta of {data.LOW_VOL} on {data.MARKET}: {sb['beta_low']:+.2f}")
    print(f"  beta of the SPREAD: {sb['beta_of_spread']:+.2f}, correlation "
          f"{sb['corr_with_market']:+.2f}, R^2 on the market {sb['r2_on_market']:.0%}")
    h.update({k: float(v) for k, v in sb.items()})

    sig = st.build_signals(common, data.HIGH_BETA, data.LOW_VOL, data.MARKET)
    resid_corr = sig["beta_neutral"].corr(common[data.MARKET])
    print(f"  after beta-neutralisation the residual gauge correlates {resid_corr:+.3f} with "
          f"the market (was {sig['raw'].corr(common[data.MARKET]):+.3f})")
    h["resid_corr"] = float(resid_corr)

    print(f"\n=== 2. predicting the next {HORIZON} days ===")
    grid = st.univariate_grid(sig, common[data.MARKET])
    print("  signal                                   lookback  horizon     beta       t      R2")
    for _, r in grid.iterrows():
        flag = " *" if abs(r["t"]) >= 2 else ""
        print(f"  {st.SIGNAL_LABEL[r['signal']]:40s} {int(r['lookback']):8d} "
              f"{int(r['horizon']):8d} {r['beta']:+8.3f} {r['t']:+7.2f} {r['r2']:7.4%}{flag}")
    h["grid"] = grid.to_dict("records")
    h["n_cells"] = int(len(grid))
    h["n_hits"] = int((grid["t"].abs() >= 2).sum())
    h["expected_hits"] = st.expected_false_positives(len(grid))
    print(f"  {h['n_hits']} of {h['n_cells']} cells clear |t| = 2; luck gives "
          f"{h['expected_hits']:.1f}")

    print(f"\n=== 3. the horse race (lookback {LOOKBACK}d, horizon {HORIZON}d) ===")
    race = st.horse_race(sig, common[data.MARKET], lookback=LOOKBACK, horizon=HORIZON)
    for _, r in race.iterrows():
        print(f"  {r['specification']:22s} {st.SIGNAL_LABEL[r['signal']]:40s} "
              f"beta {r['beta']:+8.3f}  t {r['t']:+6.2f}  R2 {r['r2']:7.4%}")
    h["race"] = race.to_dict("records")
    uni = race[race["specification"].str.contains("alone")].set_index("signal")
    multi = race[race["specification"] == "the decomposition"].set_index("signal")
    h["t_raw_uni"] = float(uni.loc["raw", "t"])
    h["t_neutral_uni"] = float(uni.loc["beta_neutral", "t"])
    h["t_trend_uni"] = float(uni.loc["market_trend", "t"])
    h["t_neutral_multi"] = float(multi.loc["beta_neutral", "t"])
    h["t_trend_multi"] = float(multi.loc["market_trend", "t"])
    gap = st.decomposition_residual(sig, common, data.HIGH_BETA, data.LOW_VOL, data.MARKET)
    h["decomposition_gap"] = gap
    print(f"  the identity raw = neutral + beta*market holds to {gap:.2e} — which is why the "
          f"three cannot go into one regression, and the decomposition does instead")
    print(f"  -> the raw gauge alone: t {h['t_raw_uni']:+.2f}; beta-neutral alone "
          f"{h['t_neutral_uni']:+.2f}; the market's own trend alone {h['t_trend_uni']:+.2f}")
    print(f"  -> decomposed: neutral {h['t_neutral_multi']:+.2f}, market "
          f"{h['t_trend_multi']:+.2f}")

    print("\n=== 4. the rule ===")
    rules = {}
    for name in st.SIGNALS:
        r = st.timing_rule(common, sig[name], data.MARKET, data.CASH,
                           lookback=LOOKBACK, cost_bps=COST_BPS)
        rules[name] = {k: v for k, v in r.items() if k != "returns"}
        a, b = r["strategy"], r["buy_hold"]
        print(f"  {st.SIGNAL_LABEL[name]:40s} invested {r['time_invested']:.0%}, "
              f"CAGR {a['cagr']:+.2%} vs {b['cagr']:+.2%} ({r['cagr_gap']:+.2%}), "
              f"Sharpe {a['sharpe']:+.2f} vs {b['sharpe']:+.2f}, maxDD {a['max_dd']:.1%} vs "
              f"{b['max_dd']:.1%}, t {r['t_gap']:+.2f}")
    h["rules"] = rules
    head = rules["raw"]
    h.update({"time_invested": head["time_invested"],
              "switches_per_year": head["switches_per_year"],
              "cagr_strategy": head["strategy"]["cagr"], "cagr_hold": head["buy_hold"]["cagr"],
              "cagr_gap": head["cagr_gap"], "sharpe_strategy": head["strategy"]["sharpe"],
              "sharpe_hold": head["buy_hold"]["sharpe"], "sharpe_gap": head["sharpe_gap"],
              "t_gap": head["t_gap"], "dd_strategy": head["strategy"]["max_dd"],
              "dd_hold": head["buy_hold"]["max_dd"]})

    print("\n=== 5. lookback sweep (raw gauge) ===")
    sweep = []
    for lb in (21, 42, 63, 126, 252):
        r = st.timing_rule(common, sig["raw"], data.MARKET, data.CASH, lookback=lb,
                           cost_bps=COST_BPS)
        sweep.append({"lookback": lb, "cagr_gap": r["cagr_gap"], "sharpe_gap": r["sharpe_gap"],
                      "time_invested": r["time_invested"]})
        print(f"  {lb:4d}d: invested {r['time_invested']:.0%}, CAGR gap {r['cagr_gap']:+.2%}, "
              f"Sharpe gap {r['sharpe_gap']:+.2f}")
    h["sweep"] = sweep

    print("\n=== 6. did it warn? the sample's actual drawdowns ===")
    ct = st.crisis_table(sig["raw"], common[data.MARKET], lookback=LOOKBACK)
    for ep, r in ct.iterrows():
        warn = (f"{int(r['sessions_of_warning'])} sessions before the peak"
                if np.isfinite(r["sessions_of_warning"]) else "no warning")
        print(f"  {ep:26s} peak {r['market_peak']}  gauge turned "
              f"{r['signal_turned_negative']}  ({warn})  drawdown {r['drawdown']:.1%}")
    h["crises"] = ct.reset_index().to_dict("records")
    ct_neutral = st.crisis_table(sig["beta_neutral"], common[data.MARKET], lookback=LOOKBACK)
    h["crises_neutral"] = ct_neutral.reset_index().to_dict("records")

    print("\n=== 7. controls: the same test on two other risk-appetite pairs ===")
    extra = {}
    for label, (hi, lo) in {"small caps over the index": ("IWM", "SPY"),
                            "high yield over investment grade": ("HYG", "LQD")}.items():
        if hi in common.columns and lo in common.columns:
            s2 = st.ratio_beta_neutral(common, hi, lo, data.MARKET)
            r = st.hac_regression(st.forward_return(common[data.MARKET], HORIZON),
                                  st.trailing(s2, LOOKBACK).to_frame("x"), lags=HORIZON)
            extra[label] = {"beta": r.get("beta_x", np.nan), "t": r.get("t_x", np.nan),
                            "r2": r.get("r2", np.nan)}
            print(f"  {label:34s} beta-neutral spread predicts with t {r.get('t_x', np.nan):+.2f} "
                  f"(R2 {r.get('r2', np.nan):.4%})")
    h["other_pairs"] = extra

    print("\n=== 8. synthetic control ===")
    for strength, tag in ((1.0, "planted appetite factor"), (0.0, "null: pure beta spread")):
        ts_raw, ts_neu = [], []
        for s in range(6):
            sim = st.synthetic_world(n=4000, appetite_strength=strength, seed=982 + s)
            sg = st.build_signals(sim, "SPHB", "SPLV", "SPY")
            rc = st.horse_race(sg, sim["SPY"], lookback=21, horizon=21)
            uni_s = rc[rc["specification"] == "raw alone"].set_index("signal")
            m = rc[rc["specification"] == "the decomposition"].set_index("signal")
            ts_raw.append(uni_s.loc["raw", "t"])
            ts_neu.append(m.loc["beta_neutral", "t"])
        print(f"  {tag:26s} raw alone t {np.nanmean(ts_raw):+5.2f}, beta-neutral in the "
              f"decomposition t {np.nanmean(ts_neu):+5.2f}")
        h[f"synthetic_{'planted' if strength else 'null'}"] = {
            "t_raw": float(np.nanmean(ts_raw)), "t_neutral": float(np.nanmean(ts_neu))}

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    grid = "\n".join(
        f"| {st.SIGNAL_LABEL[r['signal']]} | {int(r['lookback'])}d | {int(r['horizon'])}d | "
        f"{r['beta']:+.3f} | {r['t']:+.2f} | {r['r2']:.4%} |" for r in h["grid"])
    race = "\n".join(
        f"| {r['specification']} | {st.SIGNAL_LABEL[r['signal']]} | {r['beta']:+.3f} | "
        f"{r['t']:+.2f} | {r['r2']:.4%} |" for r in h["race"])
    rules = "\n".join(
        f"| {st.SIGNAL_LABEL[k]} | {r['time_invested']:.0%} | {r['switches_per_year']:.1f} | "
        f"{r['strategy']['cagr']:+.2%} | {r['cagr_gap']:+.2%} | {r['strategy']['sharpe']:+.2f} | "
        f"{r['strategy']['max_dd']:.1%} | {r['t_gap']:+.2f} |" for k, r in h["rules"].items())
    sweep = "\n".join(
        f"| {r['lookback']}d | {r['time_invested']:.0%} | {r['cagr_gap']:+.2%} | "
        f"{r['sharpe_gap']:+.2f} |" for r in h["sweep"])
    crises = "\n".join(
        f"| {r['episode']} | {r['market_peak']} | {r['signal_turned_negative']} | "
        f"{'' if not np.isfinite(r['sessions_of_warning']) else int(r['sessions_of_warning'])} | "
        f"{r['drawdown']:.1%} |" for r in h["crises"])
    others = "\n".join(f"| {k} | {vv['beta']:+.3f} | {vv['t']:+.2f} | {vv['r2']:.4%} |"
                       for k, vv in h["other_pairs"].items())
    return f"""# Results — Study 982 (The Appetite Gauge) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). SPHB and SPLV — the S&P 500's
high-beta and low-volatility halves — against SPY, on a common window of {h['n_common']:,}
sessions ({h['window'][0]} → {h['window'][1]}, **{h['years']:.1f} years**, the binding
constraint on every *t* below). As-of **{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

## 1. How much of the gauge is the market?

| | |
|---|--:|
| Beta of SPHB on SPY | {h['beta_high']:+.2f} |
| Beta of SPLV on SPY | {h['beta_low']:+.2f} |
| **Beta of the spread** | **{h['beta_of_spread']:+.2f}** |
| Correlation of the spread with the market | {h['corr_with_market']:+.2f} |
| **R² of the spread on the market** | **{h['r2_on_market']:.0%}** |
| Correlation after beta-neutralisation | {h['resid_corr']:+.3f} |

A rising gauge is, {h['r2_on_market']:.0%} of the time, a rising market. Everything that
follows is an attempt to find out whether the remaining {1 - h['r2_on_market']:.0%} says
anything.

## 2. Predicting the market

| Signal | Lookback | Horizon | Slope | HAC *t* | R² |
|---|---|---|--:|--:|--:|
{grid}

**{h['n_hits']} of {h['n_cells']}** cells clear |*t*| = 2 against **{h['expected_hits']:.1f}**
expected by luck.

## 3. The horse race

Each signal alone, then the decomposition, predicting the next {h['horizon']} sessions from a
{h['lookback']}-day trailing average:

| Specification | Signal | Slope | HAC *t* | R² |
|---|---|--:|--:|--:|
{race}

This is the table that settles the study, and it needs one note on why it has the shape it has.
The three signals are **exactly** linearly dependent — `raw = beta_neutral + β · market` is the
definition of the neutralisation, and it holds here to {h['decomposition_gap']:.1e}. Regressing
the market on all three at once is therefore singular and its coefficients are meaningless. The
informative version splits the raw gauge, without remainder, into a market part and a residual
part and lets those two compete: whichever carries the slope is where the raw gauge's content
came from.

The raw gauge alone scores **{h['t_raw_uni']:+.2f}**; beta-neutralised it scores
{h['t_neutral_uni']:+.2f}; and in the decomposition the residual gauge is
**{h['t_neutral_multi']:+.2f}** against the market component's {h['t_trend_multi']:+.2f}.

## 4. The rule

Own the index while the signal's {h['lookback']}-day average is positive, T-bills otherwise,
one day of lag, 2 bps a switch:

| Signal | Time invested | Switches/yr | CAGR | vs buy-and-hold | Sharpe | Max DD | *t* |
|---|--:|--:|--:|--:|--:|--:|--:|
{rules}

| Lookback | Time invested | CAGR gap | Sharpe gap |
|---|--:|--:|--:|
{sweep}

## 5. Did it warn?

| Episode | Market peak | Gauge turned negative | Sessions of warning | Drawdown |
|---|---|---|--:|--:|
{crises}

Four episodes is an anecdote, not a test, and it is labelled as one. It is included because a
leading indicator's whole claim is about turning points, and a full-sample regression has
almost no power at exactly those moments.

## 6. Other risk-appetite pairs, beta-neutralised

| Pair | Slope | HAC *t* | R² |
|---|--:|--:|--:|
{others}

## 7. Synthetic control

With a genuine appetite factor planted (a latent state driving both the residual spread and
next period's market return): raw *t* {h['synthetic_planted']['t_raw']:+.2f}, beta-neutral *t*
**{h['synthetic_planted']['t_neutral']:+.2f}**. With the link removed and only the beta
exposure left: raw {h['synthetic_null']['t_raw']:+.2f}, beta-neutral
**{h['synthetic_null']['t_neutral']:+.2f}**. The apparatus finds a real factor and is not
fooled by a leveraged one.

## Caveats

- **Fifteen years.** SPHB and SPLV launched in 2011. Two crashes and one cycle is not a sample
  on which to settle a debate, and no daily-frequency cleverness changes that.
- **Beta is estimated.** The neutralisation uses a trailing 252-day beta; a different window
  leaves a different residual, and the residual is the object of study.
- **Low-volatility is itself a factor.** The residual spread is not "risk appetite" in any pure
  sense — it is largely the low-volatility anomaly's own factor return (studies **330** and
  **903**), which has its own literature and its own drivers.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[982-risk-appetite-ratio](../README.md). Not investment advice.*
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
