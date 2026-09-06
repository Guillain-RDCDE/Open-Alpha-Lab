"""Real-tape verification — Study 998 (The Moving Target). Regenerates docs/results.md.

Grades a Kalman filter against rolling windows on a synthetic pair whose hedge
ratio is known, then runs every estimator across seven real pairs — including two that should
barely move — measuring spread tightness, mean-reversion speed, and traded performance with the
hedge-rebalancing cost charged explicitly.

    python studies/998-kalman-hedge-ratio/examples/verify.py            # cache-only
    python studies/998-kalman-hedge-ratio/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from movingtarget import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


COST_BPS = 5.0
WINDOWS = (20, 60, 120, 250)
DELTAS = (1e-3, 1e-2, 1e-1, 3e-1)


def report() -> dict:
    px = data.load_prices()
    rets = px.pct_change()
    h: dict = {"as_of": data.AS_OF, "cost_bps": COST_BPS,
               "fingerprint": data.fingerprint(px)}

    pairs = [(a, b) for a, b in data.PAIRS
             if a in rets.columns and b in rets.columns
             and rets[a].notna().sum() > 1000 and rets[b].notna().sum() > 1000]
    h["n_pairs"] = int(len(pairs))
    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    for a, b in pairs:
        d = pd.concat([rets[a], rets[b]], axis=1).dropna()
        print(f"  {a}/{b}: {len(d):,} common sessions, correlation "
              f"{d.iloc[:, 0].corr(d.iloc[:, 1]):+.3f}, full-sample beta "
              f"{st.static_hedge_ratio(d.iloc[:, 0], d.iloc[:, 1]):.3f}")
    first = pd.concat([rets[pairs[0][0]], rets[pairs[0][1]]], axis=1).dropna()
    h["years"] = float(len(first) / st.TRADING_DAYS)

    print("\n=== 1. grading against a KNOWN hedge ratio ===")
    grid = []
    for bvol, tag in ((0.0, "constant beta"), (0.001, "slowly drifting"),
                      (0.004, "fast drifting")):
        w = st.synthetic_pair(n=6000, beta_vol=bvol)
        row = {"world": tag, "beta_vol": bvol}
        for win in WINDOWS:
            te = st.tracking_error(st.rolling_hedge_ratio(w["y"], w["x"], win), w["beta"])
            row[f"rolling{win}"] = te["rmse"]
        for d in DELTAS:
            te = st.tracking_error(st.kalman_hedge_ratio(w["y"], w["x"], delta=d)["beta"],
                                   w["beta"])
            row[f"kalman{d:g}"] = te["rmse"]
        grid.append(row)
        cols = {k: v for k, v in row.items() if k not in ("world", "beta_vol")}
        best = min(cols, key=cols.get)
        print(f"  {tag:18s} best estimator: {best:14s} (RMSE {cols[best]:.4f})")
        print("    " + "  ".join(f"{k} {v:.4f}" for k, v in cols.items()))
    h["tracking_grid"] = grid
    drift = [g for g in grid if g["beta_vol"] > 0.003][0]
    roll_cols = {k: v for k, v in drift.items() if k.startswith("rolling")}
    kal_cols = {k: v for k, v in drift.items() if k.startswith("kalman")}
    h["best_rolling_rmse"] = float(min(roll_cols.values()))
    h["best_rolling_window"] = int(min(roll_cols, key=roll_cols.get).replace("rolling", ""))
    h["kalman_rmse"] = float(min(kal_cols.values()))
    print(f"  -> on a drifting beta the filter's RMSE is {h['kalman_rmse']:.4f} against the "
          f"best window's {h['best_rolling_rmse']:.4f} "
          f"({h['best_rolling_rmse'] / h['kalman_rmse']:.2f}x better)")

    w = st.synthetic_pair(n=8000, beta_vol=0.003)
    te_k = st.tracking_error(st.kalman_hedge_ratio(w["y"], w["x"], delta=1e-1)["beta"],
                             w["beta"])
    te_r = st.tracking_error(st.rolling_hedge_ratio(w["y"], w["x"], 20), w["beta"])
    h["kalman_excess_movement"] = te_k["excess_movement"]
    h["rolling_excess_movement"] = te_r["excess_movement"]
    print(f"  excess movement (estimate volatility / truth volatility): Kalman "
          f"{te_k['excess_movement']:.2f}x, 20-day window {te_r['excess_movement']:.2f}x")
    print("  a good estimator moves as much as the truth does — no more")

    print("\n=== 2. what the filter is, in window terms ===")
    for d in DELTAS:
        ew = st.effective_window(d, 1e-3, x_var=1.5e-4)
        print(f"  delta {d:g} behaves like a ~{ew:.0f}-session window")
    h["effective_windows"] = [{"delta": d,
                               "window": float(st.effective_window(d, 1e-3, 1.5e-4))}
                              for d in DELTAS]
    print("  a Kalman filter is not magic — it is a smoothly-weighted window whose length "
          "adapts to how informative the data has been")

    print("\n=== 3. every estimator on every real pair ===")
    all_rows = []
    for a, b in pairs:
        d = pd.concat([rets[a].rename("y"), rets[b].rename("x")], axis=1).dropna()
        c = st.compare_estimators(d["y"], d["x"], d["y"], d["x"], WINDOWS, DELTAS, COST_BPS)
        print(f"\n  {a}/{b}:")
        print(c.round(4).to_string())
        for est, row in c.iterrows():
            all_rows.append({"pair": f"{a}/{b}", "estimator": est, **row.to_dict()})
    panel = pd.DataFrame(all_rows)
    h["panel"] = panel.to_dict("records")

    print("\n=== 4. who wins? ===")
    wins = 0
    comps = []
    for pair, g in panel.groupby("pair"):
        kal = g[g["estimator"].str.startswith("Kalman")]
        roll = g[g["estimator"].str.startswith("rolling")]
        if kal.empty or roll.empty:
            continue
        bk = kal.loc[kal["spread_std"].idxmin()]
        br = roll.loc[roll["spread_std"].idxmin()]
        won = bool(bk["spread_std"] < br["spread_std"])
        wins += won
        comps.append({"pair": pair, "kalman_spread": float(bk["spread_std"]),
                      "rolling_spread": float(br["spread_std"]),
                      "kalman_sharpe": float(bk["sharpe"]),
                      "rolling_sharpe": float(br["sharpe"]),
                      "kalman_wins": won})
        print(f"  {pair:12s} spread std: Kalman {bk['spread_std']:.5f} vs rolling "
              f"{br['spread_std']:.5f}  {'KALMAN' if won else 'rolling'}")
    h["comparisons"] = comps
    h["kalman_wins_spread"] = float(wins / max(len(comps), 1))
    print(f"  the filter produced the tighter spread on {wins} of {len(comps)} pairs "
          f"({h['kalman_wins_spread']:.0%})")

    static_pairs = [c for c in comps if c["pair"] in ("GLD/IAU", "SPY/IVV")]
    h["static_pair"] = static_pairs[0]["pair"] if static_pairs else "n/a"
    if static_pairs:
        won_static = all(c["kalman_wins"] for c in static_pairs)
        h["static_verdict"] = ("persists, which is a warning sign" if won_static
                               else "disappears, as it should")
        print(f"  on the near-identical pairs, the filter's advantage {h['static_verdict']}")
    else:
        h["static_verdict"] = "could not be tested"

    print("\n=== 5. gross against net ===")
    kal_all = panel[panel["estimator"].str.startswith("Kalman")]
    roll_all = panel[panel["estimator"].str.startswith("rolling")]
    h.update({"kalman_gross_sharpe": float(kal_all["gross_ann"].mean()
                                           / max(kal_all["gross_ann"].std(ddof=1), 1e-9)
                                           if False else kal_all["sharpe"].mean()),
              "rolling_gross_sharpe": float(roll_all["sharpe"].mean()),
              "kalman_turnover": float(kal_all["hedge_turnover"].mean()),
              "rolling_turnover": float(roll_all["hedge_turnover"].mean()),
              "kalman_cost": float(kal_all["cost_ann"].mean()),
              "rolling_cost": float(roll_all["cost_ann"].mean())})
    gross_k, gross_r, net_k, net_r = [], [], [], []
    for a, b in pairs:
        d = pd.concat([rets[a].rename("y"), rets[b].rename("x")], axis=1).dropna()
        kb = st.kalman_hedge_ratio(d["y"], d["x"], delta=1e-2)["beta"]
        rb = st.rolling_hedge_ratio(d["y"], d["x"], 60)
        for beta, gl, nl in ((kb, gross_k, net_k), (rb, gross_r, net_r)):
            free = st.spread_trade(d["y"], d["x"], beta, cost_bps=0.0)
            paid = st.spread_trade(d["y"], d["x"], beta, cost_bps=COST_BPS)
            if "sharpe" in free:
                gl.append(free["sharpe"])
                nl.append(paid["sharpe"])
    h["kalman_gross_sharpe"] = float(np.nanmean(gross_k))
    h["rolling_gross_sharpe"] = float(np.nanmean(gross_r))
    h["kalman_net_sharpe"] = float(np.nanmean(net_k))
    h["rolling_net_sharpe"] = float(np.nanmean(net_r))
    h["best_sharpe"] = float(max(h["kalman_net_sharpe"], h["rolling_net_sharpe"]))
    print(f"  Kalman:  gross Sharpe {h['kalman_gross_sharpe']:.3f} -> net "
          f"{h['kalman_net_sharpe']:.3f}  (hedge turnover {h['kalman_turnover']:.2f}/yr)")
    print(f"  rolling: gross Sharpe {h['rolling_gross_sharpe']:.3f} -> net "
          f"{h['rolling_net_sharpe']:.3f}  (hedge turnover {h['rolling_turnover']:.2f}/yr)")
    print("  the gap between gross and net is what the adaptiveness costs")

    print("\n=== 6. cost sensitivity ===")
    sweep = []
    a, b = pairs[0]
    d = pd.concat([rets[a].rename("y"), rets[b].rename("x")], axis=1).dropna()
    kb = st.kalman_hedge_ratio(d["y"], d["x"], delta=1e-2)["beta"]
    rb = st.rolling_hedge_ratio(d["y"], d["x"], 60)
    for c in (0.0, 2.0, 5.0, 10.0, 25.0):
        tk = st.spread_trade(d["y"], d["x"], kb, cost_bps=c)
        tr = st.spread_trade(d["y"], d["x"], rb, cost_bps=c)
        sweep.append({"cost_bps": c, "kalman": tk.get("sharpe", np.nan),
                      "rolling": tr.get("sharpe", np.nan)})
        print(f"  {c:5.1f} bps: Kalman {tk.get('sharpe', np.nan):+.3f}, rolling "
              f"{tr.get('sharpe', np.nan):+.3f}")
    h["cost_sweep"] = sweep

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    keys = [k for k in h["tracking_grid"][0] if k not in ("world", "beta_vol")]
    grid = "\n".join(
        "| " + r["world"] + " | " + " | ".join(f"{r[k]:.4f}" for k in keys) + " |"
        for r in h["tracking_grid"])
    grid_hdr = " | ".join(keys)
    ew = "\n".join(f"| {r['delta']:g} | ~{r['window']:.0f} sessions |"
                   for r in h["effective_windows"])
    comp = "\n".join(
        f"| {r['pair']} | {r['kalman_spread']:.5f} | {r['rolling_spread']:.5f} | "
        f"{r['kalman_sharpe']:+.2f} | {r['rolling_sharpe']:+.2f} | "
        f"{'**Kalman**' if r['kalman_wins'] else 'rolling'} |" for r in h["comparisons"])
    cs = "\n".join(f"| {r['cost_bps']:.0f} | {r['kalman']:+.3f} | {r['rolling']:+.3f} |"
                   for r in h["cost_sweep"])
    return f"""# Results — Study 998 (The Moving Target) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). {h['n_pairs']} pairs over roughly
{h['years']:.0f} years. As-of **{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

## 1. Graded against a hedge ratio that is known

The only clean way to compare estimators is on data where the truth is planted. RMSE against the
true beta:

| World | {grid_hdr} |
|---|{"---|" * len(keys)}
{grid}

On a genuinely drifting beta the filter's RMSE is **{h['kalman_rmse']:.4f}** against the best
rolling window's {h['best_rolling_rmse']:.4f} (a {h['best_rolling_window']}-day window) —
**{h['best_rolling_rmse'] / h['kalman_rmse']:.2f}× better**. Read the *constant beta* row too:
where nothing moves, adaptation is a liability, and any estimator that still "wins" there is
fitting noise.

The diagnostic that explains it:

| | Estimate volatility ÷ truth volatility |
|---|--:|
| Kalman | {h['kalman_excess_movement']:.2f}× |
| 20-day rolling | {h['rolling_excess_movement']:.2f}× |

A good estimator moves as much as the truth does and no more. A short window thrashes; a long
one lags; the filter's gain adapts between the two.

## 2. What the filter actually is

| delta | Equivalent window |
|---|---|
{ew}

A Kalman filter with a random-walk state is an exponentially-weighted estimator in steady state.
It is not doing something categorically different from a rolling window — it is choosing the
weighting scheme, and the effective length, that a fixed window has to guess in advance.

## 3. On the real pairs

| Pair | Kalman spread SD | Best rolling | Kalman Sharpe | Rolling Sharpe | Winner |
|---|--:|--:|--:|--:|---|
{comp}

The filter produced the tighter spread on **{h['kalman_wins_spread']:.0%}** of pairs. On the
near-identical pairs ({h['static_pair']}, two funds holding the same thing) its advantage
**{h['static_verdict']}** — which is the control that separates genuine adaptation from
wobbling.

## 4. Tracking is not trading

A hedge ratio that follows the truth has to be **traded** to be maintained. Every move in beta
is a trade in the second leg, whether or not the spread position changed. Most comparisons
charge only entry and exit and therefore flatter the adaptive estimator.

| | Kalman | Rolling 60d |
|---|--:|--:|
| Hedge turnover per year | {h['kalman_turnover']:.2f} | {h['rolling_turnover']:.2f} |
| Cost per year | {h['kalman_cost']:.2%} | {h['rolling_cost']:.2%} |
| **Gross Sharpe** | {h['kalman_gross_sharpe']:.3f} | {h['rolling_gross_sharpe']:.3f} |
| **Net Sharpe** | **{h['kalman_net_sharpe']:.3f}** | **{h['rolling_net_sharpe']:.3f}** |

| Cost (bps) | Kalman Sharpe | Rolling Sharpe |
|---|--:|--:|
{cs}

## Caveats

- **The observation variance is estimated from a warm-up slice, not the whole sample.** That
  distinction is not pedantry. Scaling `obs_var` to the data is necessary — without it `delta`
  means something different for every pair — but scaling it to the *full* sample would let the
  hedge ratio used in 2008 depend on what happened in 2024. The suite's causality test exists
  because the full-sample version passed every other check silently.
- **The best Sharpe here is {h['best_sharpe']:.2f}.** Whatever the estimator comparison says,
  none of these pairs constitutes a business over this sample. The study is about the estimator,
  not about pairs trading.
- **The filter's parameters are chosen, not estimated.** `delta` and `obs_var` could be fitted
  by maximum likelihood; here they are swept. A fitted version would be better and would also
  introduce a search dimension of its own — see study **996**.
- **No borrow costs or shorting constraints.** A spread trade is short one leg, and for several
  of these pairs that is neither free nor always possible.
- **Returns, not prices.** The hedge ratio is estimated on returns, so the spread is a
  return-space construct. A cointegration-based version working in log prices is the other
  standard approach and would answer a slightly different question.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[998-kalman-hedge-ratio](../README.md). Not investment advice.*
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
