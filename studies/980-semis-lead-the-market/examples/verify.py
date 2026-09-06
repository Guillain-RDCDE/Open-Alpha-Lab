"""Real-tape verification — Study 980 (The Silicon Canary). Regenerates docs/results.md.

Strips the market factor out of two semiconductor ETFs, measures the lead-lag against
the market in both directions, runs predictive regressions with horizon-lag HAC errors over a
lookback × horizon grid, prices the timing rule the folklore implies, and checks the whole
battery against a non-semiconductor control and a planted-lead simulation.

    python studies/980-semis-lead-the-market/examples/verify.py            # cache-only
    python studies/980-semis-lead-the-market/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from semi_lead import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


LOOKBACK = 21
HORIZON = 21
COST_BPS = 2.0
SPLIT = "2013-01-01"


def report() -> dict:
    px = data.load_prices()
    rets = st.to_returns(px).dropna(how="all")
    common = rets[list(data.TICKERS)].dropna()
    h: dict = {"as_of": data.AS_OF, "lookback": LOOKBACK, "horizon": HORIZON,
               "fingerprint": data.fingerprint(px)}

    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    for tk in data.TICKERS:
        s = px[tk].dropna()
        print(f"  {tk:5s} {s.index[0].date()} -> {s.index[-1].date()}  n={len(s):,}  "
              f"ann vol {s.pct_change().std() * np.sqrt(st.TRADING_DAYS):.1%}")
    print(f"  common window: {common.index[0].date()} -> {common.index[-1].date()} "
          f"({len(common):,} sessions)")
    h["window"] = [str(common.index[0].date()), str(common.index[-1].date())]
    h["n_common"] = int(len(common))

    print("\n=== how much of the canary is just the market? ===")
    for s in (data.CANARY,) + data.PEERS:
        beta = st.residual_series(common, s, data.MARKET).dropna()
        raw = common[s]
        print(f"  {s:5s} correlation with {data.MARKET}: raw {raw.corr(common[data.MARKET]):.3f}, "
              f"after removing a rolling beta {beta.corr(common[data.MARKET].reindex(beta.index)):+.3f}")
    h["raw_corr"] = float(common[data.CANARY].corr(common[data.MARKET]))

    print("\n=== lead-lag, both directions (residual signal) ===")
    ll = st.lead_lag_table(common, data.CANARY, data.MARKET, max_lag=5)
    print("  lag   sector leads   market leads   difference")
    for lag, row in ll.iterrows():
        print(f"  {lag:3d}   {row['sector_leads']:+12.4f}   {row['market_leads']:+12.4f}   "
              f"{row['difference']:+10.4f}")
    h["lead_lag"] = {int(k): dict(v) for k, v in ll.to_dict("index").items()}
    h["lead_lag1"] = float(ll.loc[1, "sector_leads"])
    h["market_lead1"] = float(ll.loc[1, "market_leads"])
    h["lead_diff_lag1"] = float(ll.loc[1, "difference"])

    print(f"\n=== the headline regression: trailing {LOOKBACK}d residual -> next {HORIZON}d "
          f"market return ===")
    reg = st.predictive_regression(common, data.CANARY, data.MARKET,
                                   horizon=HORIZON, lookback=LOOKBACK)
    print(f"  slope {reg['beta']:+.3f}  HAC t {reg['t']:+.2f} ({reg['lags']} lags)  "
          f"R2 {reg['r2']:.4%}  n {reg['n']:,}  intercept {reg['alpha']:+.4%}")
    h.update({"beta": float(reg["beta"]), "t_stat": float(reg["t"]),
              "r2": float(reg["r2"]), "n_reg": int(reg["n"])})

    print("\n=== every lookback x horizon cell (the multiple-testing arithmetic) ===")
    grid = st.horizon_grid(common, data.CANARY, data.MARKET)
    for _, row in grid.iterrows():
        flag = " *" if abs(row["t"]) >= 2 else ""
        print(f"  lookback {int(row['lookback']):3d}d, horizon {int(row['horizon']):3d}d: "
              f"beta {row['beta']:+7.3f}  t {row['t']:+6.2f}  R2 {row['r2']:7.4%}{flag}")
    n_hits = int((grid["t"].abs() >= 2).sum())
    print(f"  {n_hits} of {len(grid)} cells clear |t| = 2; luck alone gives "
          f"{st.expected_false_positives(len(grid)):.1f}")
    h["grid"] = grid.to_dict("records")
    h["n_hits"] = n_hits
    h["n_cells"] = int(len(grid))
    h["expected_hits"] = st.expected_false_positives(len(grid))

    print("\n=== does the second semiconductor fund agree? and the tech control? ===")
    peers = st.peer_agreement(common, data.MARKET, data.CASH,
                              [data.CANARY] + list(data.PEERS),
                              lookback=LOOKBACK, horizon=HORIZON)
    for c, row in peers.iterrows():
        tag = "  <- tech control, not a semi" if c == "XLK" else ""
        print(f"  {c:5s} slope {row['beta']:+7.3f}  t {row['t']:+6.2f}  R2 {row['r2']:7.4%}  "
              f"rule CAGR gap {row['cagr_gap']:+7.2%}  Sharpe gap {row['sharpe_gap']:+6.2f}{tag}")
    h["peers"] = {c: dict(v) for c, v in peers.to_dict("index").items()}
    semis = [c for c in peers.index if c in (data.CANARY, "SOXX")]
    h["n_semis_significant"] = int(sum(abs(peers.loc[c, "t"]) >= 2 for c in semis))
    h["xlk_t"] = float(peers.loc["XLK", "t"]) if "XLK" in peers.index else np.nan

    print(f"\n=== the rule: own {data.MARKET} while the canary is strong, else bills "
          f"({COST_BPS:.0f} bps a switch) ===")
    rule = st.timing_rule(common, data.CANARY, data.MARKET, data.CASH,
                          lookback=LOOKBACK, cost_bps=COST_BPS)
    a, b = rule["strategy"], rule["buy_hold"]
    print(f"  invested {rule['time_invested']:.0%} of the time, "
          f"{rule['switches_per_year']:.1f} switches a year, {rule['years']:.1f} years")
    print(f"  rule       CAGR {a['cagr']:+.2%}  vol {a['vol']:.2%}  Sharpe {a['sharpe']:+.2f}  "
          f"maxDD {a['max_dd']:.1%}")
    print(f"  buy & hold CAGR {b['cagr']:+.2%}  vol {b['vol']:.2%}  Sharpe {b['sharpe']:+.2f}  "
          f"maxDD {b['max_dd']:.1%}")
    print(f"  gap {rule['cagr_gap']:+.2%}/yr, Sharpe {rule['sharpe_gap']:+.2f}, "
          f"HAC t on the daily difference {rule['t_gap']:+.2f}")
    h.update({"time_invested": rule["time_invested"],
              "switches_per_year": rule["switches_per_year"],
              "cagr_strategy": a["cagr"], "cagr_hold": b["cagr"],
              "cagr_gap": rule["cagr_gap"], "sharpe_strategy": a["sharpe"],
              "sharpe_hold": b["sharpe"], "sharpe_gap": rule["sharpe_gap"],
              "t_gap": rule["t_gap"], "dd_strategy": a["max_dd"], "dd_hold": b["max_dd"]})

    print("\n=== lookback sweep for the rule ===")
    sweep = []
    for lb in (5, 10, 21, 63, 126):
        r = st.timing_rule(common, data.CANARY, data.MARKET, data.CASH,
                           lookback=lb, cost_bps=COST_BPS)
        sweep.append({"lookback": lb, "cagr_gap": r["cagr_gap"],
                      "sharpe_gap": r["sharpe_gap"], "time_invested": r["time_invested"],
                      "switches": r["switches_per_year"]})
        print(f"  {lb:4d}d: invested {r['time_invested']:.0%}, CAGR gap {r['cagr_gap']:+.2%}, "
              f"Sharpe gap {r['sharpe_gap']:+.2f}, {r['switches_per_year']:.1f} switches/yr")
    h["sweep"] = sweep
    h["sweep_wins"] = int(sum(1 for s in sweep if s["cagr_gap"] > 0))

    print(f"\n=== era cut (split {SPLIT}) ===")
    eras = st.era_split(common, data.CANARY, data.MARKET, data.CASH, split=SPLIT,
                        lookback=LOOKBACK, horizon=HORIZON)
    for era, row in eras.iterrows():
        print(f"  {era:5s} {row['start']} -> {row['end']}  slope {row['beta']:+.3f} "
              f"(t {row['t']:+.2f})  rule CAGR gap {row['cagr_gap']:+.2%}  "
              f"Sharpe gap {row['sharpe_gap']:+.2f}")
    h["eras"] = {k: dict(v) for k, v in eras.to_dict("index").items()}

    print("\n=== synthetic control ===")
    for strength, tag in ((0.4, "planted one-day lead"), (0.0, "null: shared factor only")):
        lls, ts, gaps = [], [], []
        for s in range(6):
            sim = st.synthetic_pair(n=5000, lead_strength=strength, seed=980 + s)
            lls.append(st.lead_lag_table(sim, "SEC", "MKT", max_lag=2).loc[1, "difference"])
            ts.append(st.predictive_regression(sim, "SEC", "MKT", horizon=5, lookback=5)["t"])
            gaps.append(st.timing_rule(sim, "SEC", "MKT", "CASH", lookback=5)["cagr_gap"])
        print(f"  {tag:26s} lead-lag difference {np.mean(lls):+.4f}, mean regression t "
              f"{np.mean(ts):+.2f}, rule CAGR gap {np.mean(gaps):+.2%}")
        h[f"synthetic_{'planted' if strength else 'null'}"] = {
            "lead_diff": float(np.mean(lls)), "t": float(np.mean(ts)),
            "cagr_gap": float(np.mean(gaps))}

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    ll = "\n".join(f"| {k} | {r['sector_leads']:+.4f} | {r['market_leads']:+.4f} | "
                   f"**{r['difference']:+.4f}** |" for k, r in h["lead_lag"].items())
    grid = "\n".join(
        f"| {int(r['lookback'])}d | {int(r['horizon'])}d | {r['beta']:+.3f} | {r['t']:+.2f} | "
        f"{r['r2']:.4%} |" for r in h["grid"])
    peers = "\n".join(
        f"| {c} | {r['beta']:+.3f} | {r['t']:+.2f} | {r['r2']:.4%} | {r['cagr_gap']:+.2%} | "
        f"{r['sharpe_gap']:+.2f} |" for c, r in h["peers"].items())
    sweep = "\n".join(
        f"| {r['lookback']}d | {r['time_invested']:.0%} | {r['cagr_gap']:+.2%} | "
        f"{r['sharpe_gap']:+.2f} | {r['switches']:.1f} |" for r in h["sweep"])
    eras = "\n".join(
        f"| {k} | {r['start']} → {r['end']} | {r['beta']:+.3f} | {r['t']:+.2f} | "
        f"{r['cagr_gap']:+.2%} | {r['sharpe_gap']:+.2f} |" for k, r in h["eras"].items())
    return f"""# Results — Study 980 (The Silicon Canary) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). Daily total-return closes for
SMH, SOXX, XLK, SPY, QQQ, IWM and BIL on a common window of {h['n_common']:,} sessions
({h['window'][0]} → {h['window'][1]}). The signal is the semiconductor return with a **rolling
backward-looking beta on the market removed**, because a raw sector return is
{h['raw_corr']:.0%} correlated with the market and testing it against the market would mostly
measure that. As-of **{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

## Lead-lag, in both directions

| Lag (days) | Semis lead market | Market leads semis | Difference |
|---|--:|--:|--:|
{ll}

The second column is the control. A leading indicator whose reverse statistic is the same size
is not leading anything — and at lag 1 the difference here is **{h['lead_diff_lag1']:+.4f}**.

## The predictive regression

Trailing **{h['lookback']}-day** residual → next **{h['horizon']}-day** market return:
slope **{h['beta']:+.3f}**, HAC *t* = **{h['t_stat']:+.2f}**, R² = **{h['r2']:.4%}**, n =
{h['n_reg']:,}. Standard errors use a Newey-West lag equal to the horizon, because the forward
windows overlap.

| Lookback | Horizon | Slope | HAC *t* | R² |
|---|---|--:|--:|--:|
{grid}

**{h['n_hits']} of {h['n_cells']}** cells clear |*t*| = 2 against **{h['expected_hits']:.1f}**
expected by luck.

## Does the other semiconductor fund agree? Does the tech control?

| Candidate | Slope | *t* | R² | Rule CAGR gap | Rule Sharpe gap |
|---|--:|--:|--:|--:|--:|
{peers}

XLK is not a semiconductor fund. If it scores as well as SMH and SOXX, the claim being
supported is "technology leads", which is a different and much older idea.

## The rule the folklore implies

Own SPY while the canary's trailing {h['lookback']}-day residual strength is positive, hold
T-bills otherwise, one day of execution lag, 2 bps a switch:

| | Rule | Buy and hold |
|---|--:|--:|
| CAGR | **{h['cagr_strategy']:+.2%}** | **{h['cagr_hold']:+.2%}** |
| Sharpe | {h['sharpe_strategy']:+.2f} | {h['sharpe_hold']:+.2f} |
| Worst drawdown | {h['dd_strategy']:.1%} | {h['dd_hold']:.1%} |
| Time invested | {h['time_invested']:.0%} | 100% |
| Switches per year | {h['switches_per_year']:.1f} | 0 |

HAC *t* on the daily return difference: **{h['t_gap']:+.2f}**.

| Lookback | Time invested | CAGR gap | Sharpe gap | Switches/yr |
|---|--:|--:|--:|--:|
{sweep}

The rule beats buy-and-hold on **{h['sweep_wins']} of {len(h['sweep'])}** lookbacks.

## Era cut

| Era | Window | Slope | *t* | Rule CAGR gap | Rule Sharpe gap |
|---|---|--:|--:|--:|--:|
{eras}

## Synthetic control

With a one-day lead planted at strength 0.4: lead-lag difference
**{h['synthetic_planted']['lead_diff']:+.4f}**, mean regression *t*
**{h['synthetic_planted']['t']:+.2f}**, rule CAGR gap
{h['synthetic_planted']['cagr_gap']:+.2%}. With no lead at all (shared factor only):
{h['synthetic_null']['lead_diff']:+.4f}, *t* {h['synthetic_null']['t']:+.2f}, gap
{h['synthetic_null']['cagr_gap']:+.2%}. The apparatus finds a lead when there is one and does
not invent one when there is not.

## Caveats

- **ETFs, not the industry.** SMH and SOXX are cap-weighted funds dominated by a handful of
  names; in recent years one company has driven much of their return, so "semis" and "one
  stock" are increasingly hard to tell apart.
- **No fundamentals.** The economic version of this claim is about *bookings and inventories*,
  which lead prices; a price-only test cannot speak to it.
- **Survivorship in the control.** XLK's composition changed dramatically over the sample, so
  the "tech control" is not a constant object.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[980-semis-lead-the-market](../README.md). Not investment advice.*
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
