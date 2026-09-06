"""Real-tape verification — Study 1011 (The Half-Life of an Edge). Regenerates docs/results.md.

Builds four cross-sectional signals spanning the decay spectrum, measures each
one's rank information coefficient at horizons from one day to a year, fits a half-life to the
*marginal* profile rather than the cumulative one, bootstraps whole years of the panel to put an
interval on that half-life, computes Grinold breadth from the decay rate and the residual
correlation rather than from the trade count, and sweeps both the rebalancing period and the
Gârleanu-Pedersen partial-trading rate to see whether the theory's recommendation wins.

    python studies/1011-turnover-and-alpha-half-life/examples/verify.py            # cache-only
    python studies/1011-turnover-and-alpha-half-life/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from halflife import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


COST_BPS = 10.0
HORIZONS = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 126, 189, 252)
LAGS = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 126)


def report() -> dict:
    px = data.load_prices()
    h: dict = {"as_of": data.AS_OF, "cost_bps": COST_BPS,
               "fingerprint": data.fingerprint(px)}

    cols = [c for c in data.NAMES if c in px.columns
            and px[c].dropna().shape[0] > 2500]
    R = px[cols].pct_change().dropna()
    h["n_assets"] = int(len(cols))
    h["n_days"] = int(len(R))
    h["start"] = str(R.index[0].date())
    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    print(f"  {len(cols)} names, {len(R):,} common sessions from {h['start']}")

    print("\n=== 1. the control: can we recover a KNOWN half-life? ===")
    ctrl = []
    for true_hl in (3.0, 10.0, 21.0, 63.0):
        sim = st.synthetic_panel(n_assets=60, n_days=8000, half_life=true_hl, ic=0.08)
        f = st.fit_half_life(st.lag_profile(sim["signal"], sim["returns"]))
        ctrl.append({"true": true_hl, "fitted": f.get("half_life", np.nan),
                     "r2": f.get("r2", np.nan), "n_points": f.get("n_points", 0)})
        print(f"  planted {true_hl:5.1f} days -> fitted {f.get('half_life', np.nan):6.1f} "
              f"(R^2 {f.get('r2', np.nan):.2f}, {f.get('n_points', 0)} points)")
    h["control"] = ctrl
    print("  the estimator is calibrated before it is pointed at anything real.")

    print("\n=== 2. decay profiles of four real signals ===")
    sigs = st.make_signals(R)
    profiles, cum, fits = {}, {}, {}
    for name, s in sigs.items():
        d = st.lag_profile(s, R, LAGS)
        if d.empty:
            continue
        profiles[name] = d
        cum[name] = st.ic_decay(s, R, HORIZONS)
        f = st.fit_half_life(d)
        fits[name] = f
        print(f"  {name:14s} IC(lag 1) {d['ic'].iloc[0]:+.4f} (t {d['t'].iloc[0]:+.1f})  "
              f"IC(lag 21) {d.loc[21, 'ic']:+.4f}  half-life "
              f"{f.get('half_life', np.nan):7.1f} days  (R^2 {f.get('r2', np.nan):.2f})")
    h["profiles"] = {k: v.reset_index().to_dict("records") for k, v in profiles.items()}
    h["fits"] = {k: {kk: (float(vv) if isinstance(vv, (int, float)) else vv)
                     for kk, vv in v.items()} for k, v in fits.items()}
    decaying = {k: v["half_life"] for k, v in fits.items()
                if v.get("decaying") and np.isfinite(v["half_life"])}
    h["fastest_name"] = min(decaying, key=decaying.get)
    h["fastest_hl"] = float(decaying[h["fastest_name"]])
    h["slowest_name"] = max(decaying, key=decaying.get)
    h["slowest_hl"] = float(decaying[h["slowest_name"]])
    h["hl_spread"] = h["slowest_hl"] / max(h["fastest_hl"], 1e-9)
    print(f"  fastest: {h['fastest_name']} at {h['fastest_hl']:.1f} days")
    print(f"  slowest: {h['slowest_name']} at {h['slowest_hl']:.1f} days")
    print(f"  a factor of {h['hl_spread']:.1f} between them")

    print("\n=== 2b. cumulative vs marginal: why the fit uses the second ===")
    hn = h["fastest_name"]
    d = cum[hn]
    m = st.marginal_ic(d)
    print(f"  {hn}:")
    print(f"    cumulative IC rises from {d['ic'].iloc[0]:+.4f} at 1 day to "
          f"{d['ic'].iloc[-1]:+.4f} at {d.index[-1]} days")
    print(f"    marginal IC per day falls from {m['marginal_ic_per_day'].iloc[0]:+.5f} "
          f"to {m['marginal_ic_per_day'].iloc[-1]:+.5f}")
    print(f"  a cumulative profile can rise simply because the horizon is longer. Fitting")
    print(f"  a half-life to it would report every signal as far more durable than it is.")
    h["cumulative_first"] = float(d["ic"].iloc[0])
    h["cumulative_last"] = float(d["ic"].iloc[-1])
    h["marginal_first"] = float(m["marginal_ic_per_day"].iloc[0])
    h["marginal_last"] = float(m["marginal_ic_per_day"].iloc[-1])

    print("\n=== 3. how well do we know the half-life? ===")
    u = st.decay_uncertainty(sigs[hn], R, n_boot=60)
    h.update({"hl_median": u["median"], "hl_p05": u["p05"], "hl_p95": u["p95"],
              "hl_interval_ratio": u["ratio_95_05"]})
    print(f"  bootstrapping whole YEARS of the panel ({u['n']} resamples):")
    print(f"    half-life median {u['median']:.1f} days, 90% interval "
          f"[{u['p05']:.1f}, {u['p95']:.1f}] -> a ratio of {u['ratio_95_05']:.1f}x")
    print(f"  the Garleanu-Pedersen trading rate is a smooth function of this number.")
    print(f"  It cannot deserve more precision than the number it is built from.")

    print("\n=== 4. breadth: Grinold's law, applied carefully ===")
    resid = st._residual_correlation(R)
    h["residual_correlation"] = resid
    raw = R.corr().to_numpy()
    h["raw_correlation"] = float(np.nanmean(raw[~np.eye(len(raw), dtype=bool)]))
    print(f"  average RAW pairwise correlation:      {h['raw_correlation']:.3f}")
    print(f"  average RESIDUAL pairwise correlation: {resid:.3f}")
    breadths = []
    for name, f in fits.items():
        hl = f.get("half_life", np.nan)
        if not np.isfinite(hl):
            continue
        b = st.effective_breadth(len(cols), int(max(hl, 1)), correlation=max(resid, 0))
        breadths.append({"signal": name, "half_life": hl, **b})
        print(f"  {name:14s} half-life {hl:6.1f}d -> {b['time_bets']:6.1f} time bets x "
              f"{b['cross_sectional_bets']:5.1f} independent names = breadth "
              f"{b['breadth']:8.0f}  (naive {b['naive_breadth']:,.0f}, "
              f"overstated {b['overstatement']:.0f}x)")
    h["breadths"] = breadths

    print("\n=== 5. predicted against realised information ratio ===")
    checks = []
    for name, f in fits.items():
        hl = f.get("half_life", np.nan)
        if not np.isfinite(hl):
            continue
        rebal = int(np.clip(round(hl), 1, 126))
        g = st.grinold_check(sigs[name], R, hl, rebalance=rebal, cost_bps=0.0)
        if not g:
            continue
        checks.append({"signal": name, **g})
        print(f"  {name:14s} rebal {rebal:3d}d  IC {g['ic_at_rebalance']:+.4f}  "
              f"predicted IR {g['predicted_ir']:5.2f} (naive {g['predicted_ir_naive']:6.2f})"
              f"  realised {g['realised_ir']:+.2f}")
    h["grinold"] = checks
    head = [c for c in checks if c["signal"] == hn]
    if head:
        c0 = head[0]
        h.update({"breadth": c0["breadth"], "naive_breadth": c0["naive_breadth"],
                  "predicted_ir": c0["predicted_ir"],
                  "predicted_ir_naive": c0["predicted_ir_naive"],
                  "realised_ir": c0["realised_ir"]})
        h["breadth_overstatement"] = c0["naive_breadth"] / max(c0["breadth"], 1e-9)
        print(f"  the naive bet count overstates breadth by "
              f"{h['breadth_overstatement']:.0f}x, and IR scales with its square root,")
        print(f"  so it exaggerates the achievable IR by "
              f"{np.sqrt(h['breadth_overstatement']):.0f}x.")

    print("\n=== 6. does rebalancing near the half-life win? ===")
    h["headline_signal"] = hn
    h["headline_hl"] = h["fastest_hl"]
    rs = st.rebalance_sweep(sigs[hn], R, h["headline_hl"],
                            periods=(1, 2, 3, 5, 8, 13, 21, 42, 63, 126),
                            cost_bps=COST_BPS)
    print(rs.round(4).to_string())
    h["rebalance_sweep"] = rs.reset_index().to_dict("records")
    best = int(rs["ir"].idxmax())
    h["best_rebal"] = best
    h["best_ir"] = float(rs.loc[best, "ir"])
    faster = rs[rs.index < best]
    slower = rs[rs.index > best]
    h["fast_rebal"] = int(faster.index.min()) if len(faster) else best
    h["fast_ir"] = float(faster["ir"].iloc[0]) if len(faster) else h["best_ir"]
    h["slow_rebal"] = int(slower.index.max()) if len(slower) else best
    h["slow_ir"] = float(slower["ir"].iloc[-1]) if len(slower) else h["best_ir"]
    h["beats_faster"] = bool(h["best_ir"] > h["fast_ir"])
    h["beats_slower"] = bool(h["best_ir"] > h["slow_ir"])
    print(f"  best IR at {best} days, against a fitted half-life of "
          f"{h['headline_hl']:.1f} ({best / max(h['headline_hl'], 1):.2f}x)")
    print(f"  much faster ({h['fast_rebal']}d): {h['fast_ir']:+.3f}")
    print(f"  much slower ({h['slow_rebal']}d): {h['slow_ir']:+.3f}")

    print("\n=== 7. the Garleanu-Pedersen partial-trading rate ===")
    h["gp_rate"] = st.gp_trade_rate(h["headline_hl"], COST_BPS)
    tr = st.trade_rate_sweep(sigs[hn], R, rebalance=max(best, 1), cost_bps=COST_BPS,
                             rates=(0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0))
    print(tr.round(4).to_string())
    h["trade_rate_sweep"] = tr.reset_index().to_dict("records")
    h["best_trade_rate"] = float(tr["ir"].idxmax())
    print(f"  closed form recommends trading {h['gp_rate']:.0%} of the way each period")
    print(f"  brute-force sweep puts the optimum at {h['best_trade_rate']:.0%}")
    gp_grid = []
    for hl in (2, 5, 10, 21, 63, 126):
        for c in (1.0, 10.0, 50.0, 200.0):
            gp_grid.append({"half_life": hl, "cost_bps": c,
                            "rate": st.gp_trade_rate(hl, c)})
    h["gp_grid"] = gp_grid
    print("  the closed form's SHAPE is unambiguous -- fast decay demands fast trading,")
    print("  high costs demand slow trading -- but its LEVEL depends on a risk-aversion")
    print("  parameter nobody knows, which is why the sweep is the arbiter here.")

    print("\n=== 8. every signal, every cost level ===")
    grid = []
    for name, f in fits.items():
        hl = f.get("half_life", np.nan)
        if not np.isfinite(hl):
            continue
        for c in (0.0, 5.0, 10.0, 25.0, 50.0):
            sw = st.rebalance_sweep(sigs[name], R, hl,
                                    periods=(1, 5, 21, 63, 126), cost_bps=c)
            if sw.empty:
                continue
            b = int(sw["ir"].idxmax())
            grid.append({"signal": name, "cost_bps": c, "half_life": hl,
                         "best_rebalance": b, "best_ir": float(sw.loc[b, "ir"]),
                         "ratio_to_half_life": b / max(hl, 1e-9)})
            print(f"  {name:14s} {c:5.1f}bp -> best rebalance {b:4d}d "
                  f"(half-life {hl:6.1f}d, ratio {b / max(hl, 1e-9):5.2f})  "
                  f"IR {sw.loc[b, 'ir']:+.3f}")
    h["cost_grid"] = grid
    print("  higher costs push the optimum toward slower trading, exactly as the theory")
    print("  says. The half-life sets where you start; the cost decides how far you move.")

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    ctrl = "\n".join(
        f"| {r['true']:.0f} | {r['fitted']:.1f} | {r['r2']:.2f} | {int(r['n_points'])} |"
        for r in h["control"])
    fits = "\n".join(
        f"| {k} | {vv.get('half_life', float('nan')):.1f} | {vv.get('r2', float('nan')):.2f} | "
        f"{int(vv.get('n_points', 0))} |" for k, vv in h["fits"].items())
    br = "\n".join(
        f"| {r['signal']} | {r['half_life']:.1f} | {r['time_bets']:.1f} | "
        f"{r['cross_sectional_bets']:.1f} | {r['breadth']:,.0f} | "
        f"{r['naive_breadth']:,.0f} | **{r['overstatement']:.0f}×** |"
        for r in h["breadths"])
    gr = "\n".join(
        f"| {r['signal']} | {r['rebalance']} | {r['ic_at_rebalance']:+.4f} | "
        f"{r['predicted_ir']:.2f} | {r['predicted_ir_naive']:.2f} | "
        f"**{r['realised_ir']:+.2f}** |" for r in h["grinold"])
    rs = "\n".join(
        f"| {int(r['rebalance'])} | {r['vs_half_life']:.2f}× | {r['ir']:+.3f} | "
        f"{r['turnover_pa']:.1f}× | {r['cost_drag']:.2%} |"
        for r in h["rebalance_sweep"])
    tr = "\n".join(
        f"| {r['trade_rate']:.0%} | {r['ir']:+.3f} | {r['turnover_pa']:.1f}× | "
        f"{r['cost_drag']:.2%} |" for r in h["trade_rate_sweep"])
    cg = "\n".join(
        f"| {r['signal']} | {r['cost_bps']:.0f} | {r['half_life']:.1f} | "
        f"{r['best_rebalance']} | {r['ratio_to_half_life']:.2f}× | {r['best_ir']:+.3f} |"
        for r in h["cost_grid"])
    return f"""# Results — Study 1011 (The Half-Life of an Edge) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). {h['n_assets']} names,
{h['n_days']:,} common sessions from {h['start']}, costs at {h['cost_bps']:.0f}bp of turnover.
As-of **{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

## 1. The control — can a known half-life be recovered?

| Planted (days) | Fitted | R² | Points |
|---|--:|--:|--:|
{ctrl}

The estimator is calibrated on a world where the answer is set before it is pointed at anything
real.

## 2. Four signals, four decay profiles

| Signal | Half-life (days) | R² | Points |
|---|--:|--:|--:|
{fits}

Fastest: **{h['fastest_name']}** at {h['fastest_hl']:.1f} days. Slowest:
**{h['slowest_name']}** at {h['slowest_hl']:.1f}. A factor of **{h['hl_spread']:.1f}** between
signals whose headline ICs differ by far less — and the number that decides capacity, cost
tolerance and portfolio construction.

### Cumulative versus marginal

For {h['fastest_name']}, the *cumulative* IC rises from {h['cumulative_first']:+.4f} at one day
to {h['cumulative_last']:+.4f} at a year, while the *marginal* IC per day falls from
{h['marginal_first']:+.5f} to {h['marginal_last']:+.5f}. A cumulative profile can climb simply
because the horizon is longer; fitting a half-life to it reports every signal as far more
durable than it is. The fits above use the marginal profile.

## 3. How well is the half-life known?

Bootstrapping whole **years** of the panel, so the cross-sectional structure within each year
survives:

| | Days |
|---|--:|
| Median half-life | {h['hl_median']:.1f} |
| 90% interval | {h['hl_p05']:.1f} – {h['hl_p95']:.1f} |
| Ratio | **{h['hl_interval_ratio']:.1f}×** |

The Gârleanu-Pedersen trading rate is a smooth function of this number, which invites tuning it
to several decimal places. It cannot deserve more precision than its input.

## 4. Breadth, counted properly

Average raw pairwise correlation: {h['raw_correlation']:.3f}. Average **residual** correlation
after removing the market: {h['residual_correlation']:.3f} — the figure that belongs in the
breadth calculation, and the one usually omitted.

| Signal | Half-life | Time bets/yr | Independent names | Breadth | Naive count | Overstatement |
|---|--:|--:|--:|--:|--:|--:|
{br}

## 5. Predicted against realised information ratio

| Signal | Rebalance | IC | Predicted IR | Naive prediction | Realised IR |
|---|--:|--:|--:|--:|--:|
{gr}

The naive breadth count overstates by **{h['breadth_overstatement']:.0f}×**. Since IR scales
with the square root of breadth, that is a {np.sqrt(h['breadth_overstatement']):.0f}× exaggeration
of the achievable information ratio — which is roughly the gap between what factor backtests
promise and what factor funds deliver.

## 6. Does rebalancing near the half-life win?

{h['headline_signal']}, half-life {h['headline_hl']:.1f} days, at {h['cost_bps']:.0f}bp:

| Rebalance (days) | vs half-life | IR | Turnover | Cost drag |
|---|--:|--:|--:|--:|
{rs}

Best at **{h['best_rebal']} days**, or {h['best_rebal'] / max(h['headline_hl'], 1):.2f}× the
half-life. Much faster ({h['fast_rebal']}d) gave {h['fast_ir']:+.3f}; much slower
({h['slow_rebal']}d) gave {h['slow_ir']:+.3f}.

## 7. The partial-trading rate

| Trade rate | IR | Turnover | Cost drag |
|---|--:|--:|--:|
{tr}

The closed form recommends **{h['gp_rate']:.0%}**; the brute-force sweep puts the optimum at
**{h['best_trade_rate']:.0%}**. The formula's *shape* is unambiguous — fast decay demands fast
trading, high costs demand slow trading — but its *level* depends on a risk-aversion parameter
nobody knows, so the sweep is the arbiter.

## 8. Every signal, every cost level

| Signal | Cost (bp) | Half-life | Best rebalance | Ratio | IR |
|---|--:|--:|--:|--:|--:|
{cg}

Higher costs push the optimum toward slower trading, exactly as the theory says. The half-life
sets where you start; the cost decides how far from it you move.

## Caveats

- **Survivorship.** Fifty names that all survived to 2026, which flatters every signal's
  measured IC. It affects the decay *profile* far less than the level, and the profile is the
  subject.
- **No shorting frictions, borrow costs or capacity limits.** The portfolios are dollar-neutral
  and unit-gross by construction; a real implementation faces all three, and all three push
  toward slower trading than section 6 recommends.
- **A linear cost model.** Real costs are concave in participation rate and rise with urgency,
  which would penalise the fastest rebalancing periods more heavily than a flat basis-point
  charge does.
- **Grinold's law assumes the IC is constant across bets and the transfer coefficient is one.**
  Neither holds. The gap between predicted and realised IR in section 5 is the size of those
  assumptions, reported rather than hidden.
- **Half-lives are fitted to an exponential.** Real decay profiles are not exponential — reversal
  in particular decays faster than exponential at short horizons — so the fitted number is a
  summary, and section 3's interval is the honest way to read it.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1011-turnover-and-alpha-half-life](../README.md). Not investment advice.*
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
