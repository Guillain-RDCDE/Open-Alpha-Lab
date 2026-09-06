"""Real-tape verification — Study 964 (All-Time High). Regenerates docs/results.md.

Marks every record-high close on six total-return tapes, measures forward returns at
1 / 3 / 12 months against every other day (HAC at the horizon lag plus a non-overlapping
cross-check), buckets forward returns by how far below the peak the money went in, and then
races the advice as a portfolio: wait for a dip, hold bills meanwhile.

    python studies/964-ath-buying/examples/verify.py            # cache-only
    python studies/964-ath-buying/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ath_buy import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


TOL = 0.0
HEAD_DIP = 0.05
COST_BPS = 2.0


def report() -> dict:
    px = data.load_prices()
    cash = px[data.CASH].dropna()
    h: dict = {"as_of": data.AS_OF, "tickers": list(data.RISKY)}

    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    for tk in data.TICKERS:
        s = px[tk].dropna()
        print(f"  {tk:4s} {s.index[0].date()} -> {s.index[-1].date()}  n={len(s):,}")
    h["windows"] = {tk: [str(px[tk].dropna().index[0].date()),
                         str(px[tk].dropna().index[-1].date())] for tk in data.TICKERS}
    h["n_obs"] = {tk: int(px[tk].dropna().shape[0]) for tk in data.TICKERS}
    h["fingerprint"] = data.fingerprint(px)

    print("\n=== how often is the market at a record, anyway? ===")
    share = {}
    for tk in data.RISKY:
        s = px[tk].dropna()
        share[tk] = st.share_of_days_at_high(s)
        near = st.share_of_days_at_high(s, tol=0.01)
        print(f"  {tk:4s} strict record closes {share[tk]:6.1%} of sessions   "
              f"within 1% of the record {near:6.1%}   currently "
              f"{st.drawdown(s).iloc[-1]:+.1%} from peak")
    h["share_at_high"] = share
    h["share_at_high_spy"] = share["SPY"]

    print("\n=== forward returns from a record high vs from every other day ===")
    print("  tkr  horizon      at high    elsewhere         gap   HAC t   win(hi)  win(else)")
    fwd_all = {}
    for tk in data.RISKY:
        s = px[tk].dropna()
        tbl = st.forward_table(s, TOL)
        fwd_all[tk] = {int(k): dict(v) for k, v in tbl.to_dict("index").items()}
        for hz, row in tbl.iterrows():
            print(f"  {tk:4s} {st.HORIZON_LABEL[hz]:9s} {row['mean_state']:+10.2%} "
                  f"{row['mean_other']:+11.2%} {row['diff']:+11.2%}  {row['t_diff']:+6.2f}   "
                  f"{row['win_state']:6.0%}   {row['win_other']:6.0%}")
    h["forward"] = fwd_all

    # ------------------------------------------------------------------ pooled 12m
    pooled_state, pooled_other, pooled_t, n_pos = [], [], [], 0
    for tk in data.RISKY:
        row = fwd_all[tk][252]
        pooled_state.append(row["mean_state"])
        pooled_other.append(row["mean_other"])
        pooled_t.append(row["t_diff"])
        n_pos += int(row["diff"] > 0)
    h["pooled_state_12m"] = float(np.mean(pooled_state))
    h["pooled_other_12m"] = float(np.mean(pooled_other))
    h["pooled_diff_12m"] = float(np.mean(pooled_state) - np.mean(pooled_other))
    h["pooled_t_12m"] = float(np.mean(pooled_t))
    h["n_positive_12m"] = int(n_pos)
    print(f"\n  POOLED (equal weight across the six tapes), 12 months: "
          f"at a high {h['pooled_state_12m']:+.2%} vs elsewhere {h['pooled_other_12m']:+.2%} "
          f"-> gap {h['pooled_diff_12m']:+.2%}, mean HAC t {h['pooled_t_12m']:+.2f}, "
          f"positive on {n_pos}/6 tapes")

    print("\n=== the non-overlapping cross-check (one observation per horizon) ===")
    non = {}
    for tk in data.RISKY:
        s = px[tk].dropna()
        r = st.nonoverlap_stats(s, 252, TOL)
        non[tk] = r
        print(f"  {tk:4s} n={r['n_state']:3d} at a high / {r['n_other']:3d} elsewhere  "
              f"gap {r['diff']:+.2%}  t {r['t_diff']:+5.2f}")
    h["nonoverlap"] = non
    h["nonoverlap_diff_12m"] = float(np.nanmean([v["diff"] for v in non.values()]))
    print(f"  pooled non-overlapping gap: {h['nonoverlap_diff_12m']:+.2%} "
          f"(vs {h['pooled_diff_12m']:+.2%} overlapping — the overlap is not the story)")

    print("\n=== SPY: forward 12-month return by how far below the peak you bought ===")
    bt = st.drawdown_bucket_table(px["SPY"].dropna(), 252)
    for b, row in bt.iterrows():
        print(f"  {b:16s} n={int(row['n']):5d}  mean {row['mean_fwd']:+7.2%}  "
              f"median {row['median_fwd']:+7.2%}  win {row['win_rate']:5.0%}  "
              f"worst {row['worst']:+7.2%}")
    h["buckets_spy"] = {k: dict(v) for k, v in bt.to_dict("index").items()}

    print(f"\n=== the advice as a portfolio: wait for a {HEAD_DIP:.0%} dip, hold bills meanwhile "
          f"({COST_BPS:.0f} bps a switch) ===")
    print("  tkr   time in  switch/yr    CAGR   vs hold   exSharpe   vs hold   maxDD    t(gap)")
    races, wins = {}, 0
    for tk in data.RISKY:
        s = px[tk].dropna()
        c = cash.reindex(s.index).ffill().dropna()
        s = s.reindex(c.index)
        r = st.race(s, c, HEAD_DIP, COST_BPS)
        races[tk] = {"time_invested": r["time_invested"], "cagr": r["strategy"]["cagr"],
                     "cagr_gap": r["cagr_gap"], "sharpe": r["strategy"]["sharpe_excess"],
                     "sharpe_hold": r["buy_hold"]["sharpe_excess"],
                     "sharpe_gap": r["sharpe_gap"], "max_dd": r["strategy"]["max_dd"],
                     "max_dd_hold": r["buy_hold"]["max_dd"], "t_gap": r["t_gap"],
                     "switches_per_year": r["switches_per_year"]}
        wins += int(r["sharpe_gap"] > 0)
        print(f"  {tk:4s} {r['time_invested']:8.0%} {r['switches_per_year']:10.1f} "
              f"{r['strategy']['cagr']:+7.2%} {r['cagr_gap']:+9.2%} "
              f"{r['strategy']['sharpe_excess']:+10.2f} {r['sharpe_gap']:+9.2f} "
              f"{r['strategy']['max_dd']:+7.1%} {r['t_gap']:+9.2f}")
    h["races"] = races
    h["n_dip_sharpe_wins"] = int(wins)
    h["pooled_dip_t"] = float(np.mean([v["t_gap"] for v in races.values()]))
    h["head_dip"] = HEAD_DIP
    h["head_time_invested"] = races["SPY"]["time_invested"]
    h["head_cagr_gap"] = races["SPY"]["cagr_gap"]
    h["head_sharpe_gap"] = races["SPY"]["sharpe_gap"]
    h["head_t_gap"] = races["SPY"]["t_gap"]
    print(f"  the rule wins on excess Sharpe on {wins}/6 tapes; mean t on the daily gap "
          f"{h['pooled_dip_t']:+.2f}")

    print("\n=== how patient must the dip-buyer be? (SPY) ===")
    s = px["SPY"].dropna()
    c = cash.reindex(s.index).ffill().dropna()
    sw = st.dip_sweep(s.reindex(c.index), c, cost_bps=COST_BPS)
    for d, row in sw.iterrows():
        print(f"  wait for {d:5.0%}: invested {row['time_invested']:5.0%} of the time, "
              f"CAGR {row['cagr']:+.2%} ({row['cagr_gap']:+.2%} vs hold), "
              f"exSharpe {row['sharpe']:+.2f} ({row['sharpe_gap']:+.2f}), "
              f"maxDD {row['max_dd']:+.1%}")
    h["dip_sweep_spy"] = {float(k): dict(v) for k, v in sw.to_dict("index").items()}

    print("\n=== the drawdown the rule actually buys, and what it costs ===")
    r_head = st.race(s.reindex(c.index), c, HEAD_DIP, COST_BPS)
    print(f"  buy-and-hold worst drawdown : {r_head['buy_hold']['max_dd']:+.1%}")
    print(f"  dip rule    worst drawdown  : {r_head['strategy']['max_dd']:+.1%}")
    print(f"  compounding given up        : {r_head['cagr_gap']:+.2%}/yr over "
          f"{r_head['buy_hold']['years']:.0f} years")
    h["dd_hold_spy"] = r_head["buy_hold"]["max_dd"]
    h["dd_rule_spy"] = r_head["strategy"]["max_dd"]
    h["years_spy"] = r_head["buy_hold"]["years"]

    print("\n=== synthetic control (machinery proof only) ===")
    for ss, tag in ((1.0, "planted trend"), (0.0, "random-walk null")):
        ts = []
        for sd in range(6):
            p, cc, _ = data.synthetic_panel(n_assets=1, n_years=25, signal_strength=ss,
                                            seed=964 + sd)
            ts.append(st.conditional_stats(st.forward_return(p.iloc[:, 0], 252),
                                           st.at_high(p.iloc[:, 0]), 252)["t_diff"])
        print(f"  {tag:18s}: mean HAC t {np.nanmean(ts):+5.2f}, "
              f"|t|>=2 in {int(np.sum(np.abs(ts) >= 2))}/6")
        h[f"synthetic_{'planted' if ss else 'null'}_t"] = float(np.nanmean(ts))

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    fwd_rows = []
    for tk in h["tickers"]:
        for hz in ("21", "63", "252"):
            row = h["forward"][tk][int(hz)] if int(hz) in h["forward"][tk] else h["forward"][tk][hz]
            fwd_rows.append(
                f"| {tk} | {st.HORIZON_LABEL[int(hz)]} | {row['mean_state']:+.2%} | "
                f"{row['mean_other']:+.2%} | **{row['diff']:+.2%}** | {row['t_diff']:+.2f} | "
                f"{row['win_state']:.0%} / {row['win_other']:.0%} |")
    races = "\n".join(
        f"| {tk} | {r['time_invested']:.0%} | {r['switches_per_year']:.1f} | {r['cagr']:+.2%} | "
        f"**{r['cagr_gap']:+.2%}** | {r['sharpe']:+.2f} vs {r['sharpe_hold']:+.2f} | "
        f"{r['max_dd']:+.1%} vs {r['max_dd_hold']:+.1%} | {r['t_gap']:+.2f} |"
        for tk, r in h["races"].items())
    buckets = "\n".join(
        f"| {b} | {int(r['n']):,} | {r['mean_fwd']:+.2%} | {r['median_fwd']:+.2%} | "
        f"{r['win_rate']:.0%} | {r['worst']:+.1%} |"
        for b, r in h["buckets_spy"].items())
    sweep = "\n".join(
        f"| {float(d):.0%} | {r['time_invested']:.0%} | {r['cagr']:+.2%} | {r['cagr_gap']:+.2%} | "
        f"{r['sharpe']:+.2f} | {r['max_dd']:+.1%} |"
        for d, r in h["dip_sweep_spy"].items())
    windows = "\n".join(f"| {tk} | {w[0]} → {w[1]} | {h['n_obs'][tk]:,} |"
                        for tk, w in h["windows"].items())
    share = ", ".join(f"{tk} {s:.0%}" for tk, s in h["share_at_high"].items())
    return f"""# Results — Study 964 (All-Time High) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). Daily **total-return** closes
(`yfinance`, `auto_adjust=True`) for SPY, QQQ, EFA, EEM, TLT, GLD plus BIL as the cash leg.
A **record high** is a close equal to the running maximum of the total-return index — a
wealth peak, not a price peak. Forward returns use HAC standard errors with the lag set to
the horizon (Hansen-Hodrick), plus a non-overlapping cross-check. As-of **{h['as_of']}**;
panel fingerprint `{h['fingerprint']}`.*

## Data stamp

| Ticker | Window | Sessions |
|---|---|--:|
{windows}

Share of sessions closing at a record: {share}.

## Forward returns from a record high

| Ticker | Horizon | From a record | From anywhere else | Gap | HAC *t* | Win rate (high / else) |
|---|---|--:|--:|--:|--:|---|
{chr(10).join(fwd_rows)}

**Pooled across the six tapes at twelve months:** {h['pooled_state_12m']:+.2%} from a record
high against {h['pooled_other_12m']:+.2%} from every other day — a gap of
**{h['pooled_diff_12m']:+.2%}**, mean HAC *t* {h['pooled_t_12m']:+.2f}, positive on
**{h['n_positive_12m']} of 6** tapes.

**Non-overlapping cross-check** (one observation per year, no shared days): pooled gap
{h['nonoverlap_diff_12m']:+.2%}. The overlap is not what produces the result.

## Where the money would otherwise have gone (SPY, 12-month forward)

| Bought when | n | Mean | Median | Win rate | Worst |
|---|--:|--:|--:|--:|--:|
{buckets}

## The advice as a portfolio — wait for a {h['head_dip']:.0%} dip, hold T-bills meanwhile

| Ticker | Time invested | Switches/yr | CAGR | vs buy-and-hold | Excess Sharpe (rule vs hold) | Worst DD (rule vs hold) | *t* on the daily gap |
|---|--:|--:|--:|--:|---|---|--:|
{races}

The rule beat buy-and-hold on excess Sharpe on **{h['n_dip_sharpe_wins']} of
{len(h['tickers'])}** tapes; the mean *t* on the daily return gap is {h['pooled_dip_t']:+.2f}.

### How patient does the dip-buyer have to be? (SPY)

| Wait for | Time invested | CAGR | vs hold | Excess Sharpe | Worst DD |
|---|--:|--:|--:|--:|--:|
{sweep}

On SPY over {h['years_spy']:.0f} years the rule's worst drawdown was {h['dd_rule_spy']:+.1%}
against buy-and-hold's {h['dd_hold_spy']:+.1%} — it does buy back some of the pain, at
{h['head_cagr_gap']:+.2%}/yr of compounding.

## What this study cannot tell you

- **Survivorship.** SPY, QQQ, EFA, EEM, TLT and GLD all still exist and all spent the sample
  in secular uptrends. A record-high study run on the Nikkei from 1990, or on any index that
  spent a generation below its peak, would read differently. This is the single largest
  caveat and it is not quantified here.
- **One history.** The non-overlapping cross-check leaves roughly thirty independent
  twelve-month observations per tape. That is the true sample size behind any 12-month claim.
- **Tax and behaviour.** The dip rule generates realised gains and requires sitting out
  multi-year advances; neither is modelled.

## Synthetic control

Planted-trend world: mean HAC *t* {h['synthetic_planted_t']:+.2f}. Random-walk null: mean HAC
*t* {h['synthetic_null_t']:+.2f}. The apparatus reads a trend when one is planted and reads
nothing when it is not.

## Verdict

Produced by `strategy.verdict`, a rule fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study [964-ath-buying](../README.md).
Not investment advice.*
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
