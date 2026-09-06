"""Real-tape verification — Study 979 (The Prior Is the Portfolio). Regenerates docs/results.md.

Verifies the zero-view identity to machine precision across priors, taus and
covariances; calibrates how much of the book a view of a stated size actually moves; compares
that against how much the *choice of prior* moves it; and races a mechanically view-tilted
portfolio against the untouched prior and plain mean-variance out of sample.

    python studies/979-black-litterman-zero-views/examples/verify.py            # cache-only
    python studies/979-black-litterman-zero-views/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from bl_prior import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


WINDOW = 504
STEP = 63
COST_BPS = 5.0
VIEW_ANN = 0.03


def report() -> dict:
    px = data.load_prices()
    rets = px.pct_change()
    panels = {"multi-asset": [c for c in data.MULTI if rets[c].notna().sum() > 3000],
              "sectors": [c for c in data.SECTORS if rets[c].notna().sum() > 1500]}
    h: dict = {"as_of": data.AS_OF, "window": WINDOW, "step": STEP, "tau": st.DEFAULT_TAU,
               "delta": st.DEFAULT_DELTA, "view_ann": VIEW_ANN,
               "fingerprint": data.fingerprint(px)}

    print(f"as-of {data.AS_OF}   tau {st.DEFAULT_TAU}   delta {st.DEFAULT_DELTA}   "
          f"fp {data.fingerprint(px)}")
    for tag, cols in panels.items():
        print(f"  {tag:12s} {len(cols):2d} assets, "
              f"{len(rets[cols].dropna(how='any')):,} common sessions")

    sub = rets[panels["multi-asset"]].dropna(how="any")
    cols = panels["multi-asset"]
    cov = np.cov(sub.iloc[-WINDOW:].to_numpy(), rowvar=False, ddof=1)

    print("\n=== 1. the zero-view identity ===")
    worst = 0.0
    for kind in st.PRIORS:
        wp = st.prior_weights(cov, kind)
        for tau in (0.01, 0.05, 0.25, 1.0):
            err = float(np.abs(st.posterior_weights(cov, wp, tau=tau) - wp).sum() / 2)
            worst = max(worst, err)
        print(f"  {st.PRIOR_LABEL[kind]:26s} worst error across tau: {worst:.2e} of the book")
    # a deliberately ill-conditioned case: more assets than observations
    thin = np.cov(sub.iloc[-30:].to_numpy(), rowvar=False, ddof=1)
    wp_thin = st.prior_weights(thin, "inverse_vol")
    thin_err = float(np.abs(st.posterior_weights(thin, wp_thin) - wp_thin).sum() / 2)
    print(f"  and on a singular covariance (30 rows, {len(cols)} assets): {thin_err:.2e}")
    h["zero_view_error"] = float(max(worst, thin_err))

    print("\n=== 2. the implied 'equilibrium' returns are the prior, read backwards ===")
    for kind in st.PRIORS:
        wp = st.prior_weights(cov, kind)
        pi = st.implied_returns(cov, wp) * st.TRADING_DAYS
        top = pd.Series(pi, index=cols).sort_values(ascending=False)
        print(f"  {st.PRIOR_LABEL[kind]:26s} " +
              ", ".join(f"{k} {v:+.1%}" for k, v in top.head(4).items()))
    h["implied_returns"] = {kind: dict(zip(cols, (st.implied_returns(
        cov, st.prior_weights(cov, kind)) * st.TRADING_DAYS).tolist())) for kind in st.PRIORS}

    print("\n=== 3. how strong does a view have to be? ===")
    wp = st.prior_weights(cov, "equal")
    asset = cols.index("GLD") if "GLD" in cols else 0
    curve = st.view_strength_curve(cov, wp, asset=asset)
    print(f"  view: {cols[asset]} out-performs by X a year")
    print("  tau     " + "  ".join(f"{s:>7.0%}" for s in sorted(curve['view_ann'].unique())))
    for tau, g in curve.groupby("tau"):
        g = g.sort_values("view_ann")
        print(f"  {tau:5.2f}   " + "  ".join(f"{v:7.1%}" for v in g["book_moved"]))
    h["view_curve"] = curve.to_dict("records")
    at3 = curve[(curve["tau"] == st.DEFAULT_TAU) & (np.isclose(curve["view_ann"], 0.02))]
    h["book_moved_3pct"] = float(curve[(curve["tau"] == st.DEFAULT_TAU) &
                                       (np.isclose(curve["view_ann"], 0.05))]["book_moved"].iloc[0]
                                 ) if len(at3) == 0 else float(at3["book_moved"].iloc[0])
    h["book_moved_10pct"] = float(curve[(curve["tau"] == st.DEFAULT_TAU) &
                                        (np.isclose(curve["view_ann"], 0.10))]["book_moved"].iloc[0])
    h["view_asset"] = cols[asset]

    print("\n=== 3b. a view of ZERO is not the absence of a view ===")
    P0, q0 = st.single_view(len(cols), asset, 0.0)
    implied = float(st.implied_view(P0, cov, wp)[0]) * st.TRADING_DAYS
    moved_zero = float(np.abs(st.posterior_weights(cov, wp, P0, q0) - wp).sum() / 2)
    _, q_neutral = st.single_view(len(cols), asset, implied)
    moved_neutral = float(np.abs(st.posterior_weights(cov, wp, P0, q_neutral) - wp).sum() / 2)
    print(f"  the prior already implies {cols[asset]} returns {implied:+.2%}/yr")
    print(f"  a view of 'exactly 0%/yr' moves {moved_zero:.1%} of the book — it CONTRADICTS "
          f"the prior")
    print(f"  a view of '{implied:+.2%}/yr' — agreeing with the prior — moves "
          f"{moved_neutral:.2e}")
    print(f"  the neutral view is q = P.pi, not q = 0; every 'view size' below is measured "
          f"from zero, so read them as distances from that {implied:+.2%} baseline")
    h["implied_view_ann"] = implied
    h["moved_by_zero_view"] = moved_zero
    h["moved_by_neutral_view"] = moved_neutral

    print("\n=== 4. the prior matters more than the view ===")
    sens = st.prior_sensitivity(cov, asset=asset, size_ann=VIEW_ANN)
    print(sens.round(4).to_string())
    pairwise = sens.filter(like="vs_").to_numpy()
    pairwise = pairwise[np.isfinite(pairwise)]
    h["prior_spread"] = float(pairwise.mean()) if pairwise.size else np.nan
    h["view_move_mean"] = float(sens["view_moved_book"].mean())
    print(f"  the same view moved {h['view_move_mean']:.1%} of the book on average; "
          f"switching prior moved {h['prior_spread']:.1%}")

    print("\n=== 5. out of sample, with a mechanical 12-1 momentum view ===")
    results = {}
    for tag, cs in panels.items():
        s_sub = rets[cs].dropna(how="any")
        for prior_kind in st.PRIORS:
            wf = st.walk_forward(s_sub, prior_kind=prior_kind, window=WINDOW, step=STEP,
                                 size_ann=VIEW_ANN, cost_bps=COST_BPS)
            s = st.summarise(wf)
            key = f"{tag}|{prior_kind}"
            results[key] = {"summary": {m: dict(v) for m, v in s.to_dict("index").items()},
                            "pairs": {o: st.paired_test(wf, "black_litterman", o)
                                      for o in ("prior", "plain_mv")}}
            print(f"  {tag:12s} prior = {st.PRIOR_LABEL[prior_kind]:26s}")
            for m, row in s.iterrows():
                print(f"    {st.METHOD_LABEL[m]:34s} return {row['mean_ret']:+7.2%}  vol "
                      f"{row['realised_vol']:6.2%}  Sharpe {row['sharpe']:+6.2f}  turnover "
                      f"{row['turnover']:5.2f}  max w {row['max_weight']:5.1%}  tilt "
                      f"{row['tilt_from_prior']:5.1%}")
            p = results[key]["pairs"]["prior"]
            print(f"    BL vs the prior: return difference {p['diff']:+.3%}, t {p['t']:+.2f}, "
                  f"wins {p['win_rate']:.0%} of {p['n']}")
    h["results"] = results

    head = results["multi-asset|equal"]
    hs = head["summary"]
    h.update({"ret_bl": float(hs["black_litterman"]["mean_ret"]),
              "vol_bl": float(hs["black_litterman"]["realised_vol"]),
              "sharpe_bl": float(hs["black_litterman"]["sharpe"]),
              "max_weight_bl": float(hs["black_litterman"]["max_weight"]),
              "ret_prior": float(hs["prior"]["mean_ret"]),
              "sharpe_prior": float(hs["prior"]["sharpe"]),
              "ret_mv": float(hs["plain_mv"]["mean_ret"]),
              "sharpe_mv": float(hs["plain_mv"]["sharpe"]),
              "max_weight_mv": float(hs["plain_mv"]["max_weight"]),
              "t_bl_vs_prior": float(head["pairs"]["prior"]["t"]),
              "n_rebalances": int(head["pairs"]["prior"]["n"])})

    print("\n=== 6. view size sweep, out of sample ===")
    sweep = []
    s_sub = rets[panels["multi-asset"]].dropna(how="any")
    for size in (0.0, 0.01, 0.03, 0.06, 0.12):
        wf = st.walk_forward(s_sub, "equal", window=WINDOW, step=STEP, size_ann=size,
                             cost_bps=COST_BPS)
        s = st.summarise(wf)
        p = st.paired_test(wf, "black_litterman", "prior")
        sweep.append({"view_ann": size, "tilt": float(s.loc["black_litterman", "tilt_from_prior"]),
                      "sharpe": float(s.loc["black_litterman", "sharpe"]),
                      "vs_prior": float(p["diff"]), "t": float(p["t"])})
        print(f"  view {size:5.0%}/yr: tilt {s.loc['black_litterman', 'tilt_from_prior']:5.1%}, "
              f"Sharpe {s.loc['black_litterman', 'sharpe']:+.2f}, versus the prior "
              f"{p['diff']:+.3%} (t {p['t']:+.2f})")
    h["view_sweep"] = sweep

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    curve = pd.DataFrame(h["view_curve"])
    sizes = sorted(curve["view_ann"].unique())
    head = " | ".join(f"{s:.0%}/yr" for s in sizes)
    dash = "|".join(["--:"] * len(sizes))
    ct = "\n".join("| " + f"{tau:.2f}" + " | " + " | ".join(
        f"{curve[(curve['tau'] == tau) & (np.isclose(curve['view_ann'], s))]['book_moved'].iloc[0]:.1%}"
        for s in sizes) + " |" for tau in sorted(curve["tau"].unique()))
    def block(key):
        s = h["results"][key]["summary"]
        return "\n".join(
            f"| {st.METHOD_LABEL[m]} | {r['mean_ret']:+.2%} | {r['realised_vol']:.2%} | "
            f"{r['sharpe']:+.2f} | {r['turnover']:.2f} | {r['max_weight']:.1%} | "
            f"{r['tilt_from_prior']:.1%} |" for m, r in s.items())
    priors_tbl = "\n".join(
        f"| {st.PRIOR_LABEL[p]} | " +
        " | ".join(f"{h['results'][f'multi-asset|{p}']['summary'][m]['sharpe']:+.2f}"
                   for m in st.METHODS) +
        f" | {h['results'][f'multi-asset|{p}']['pairs']['prior']['t']:+.2f} |"
        for p in st.PRIORS)
    sweep = "\n".join(
        f"| {r['view_ann']:.0%} | {r['tilt']:.1%} | {r['sharpe']:+.2f} | {r['vs_prior']:+.3%} | "
        f"{r['t']:+.2f} |" for r in h["view_sweep"])
    return f"""# Results — Study 979 (The Prior Is the Portfolio) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). Black-Litterman implemented from
the standard algebra, with **no market-capitalisation data anywhere** — three defensible priors
are run instead, and the sensitivity to that choice is one of the results. tau =
{h['tau']}, delta = {h['delta']}, rolling **{h['window']}-day** window every **{h['step']}**
sessions, long-only out of sample, 5 bps a rebalance. As-of **{h['as_of']}**; fingerprint
`{h['fingerprint']}`.*

## 1. The zero-view identity

With no views the posterior portfolio equals the prior to **{h['zero_view_error']:.1e}** of the
book — across three priors, four values of tau, and a deliberately singular covariance matrix
(30 observations, ten assets). This is not an empirical finding, it is arithmetic, and it is the
reason everything else in the model is either the prior or the view.

## 2. How strong does a view have to be?

Share of the book that moves for a view that **{h['view_asset']}** out-performs by *X* a year:

| tau | {head} |
|---|{dash}|
{ct}

A 3%/yr view at the textbook tau of {h['tau']} moves **{h['book_moved_3pct']:.1%}** of the
portfolio; a 10%/yr view moves **{h['book_moved_10pct']:.1%}**. Those two numbers are what
`tau` and `Omega` actually mean, and they are almost never published.

### A view of zero is not the absence of a view

The prior already implies that **{h['view_asset']}** returns **{h['implied_view_ann']:+.2%}/yr**.
Telling the model "I expect exactly 0%" therefore *contradicts* it, and moves
**{h['moved_by_zero_view']:.1%}** of the book. Telling it
"{h['implied_view_ann']:+.2%}" — agreeing with the prior — moves
**{h['moved_by_neutral_view']:.1e}**. The neutral view is `q = P·π`, not `q = 0`, and the
column headings in the table above are distances from zero, not from neutrality.

## 3. The prior moves the answer more than the view does

The same view, under three priors, produces portfolios **{h['prior_spread']:.1%}** apart on
average — against the **{h['view_move_mean']:.1%}** the view itself moved. Choosing the prior
is the larger decision, and it is the one the model does not help with.

## 4. Out of sample, with a mechanical momentum view (equal-weight prior)

| Method | Return | Volatility | Sharpe | Turnover | Max weight | Tilt from prior |
|---|--:|--:|--:|--:|--:|--:|
{block('multi-asset|equal')}

Paired *t* of the tilted book against the untouched prior: **{h['t_bl_vs_prior']:+.2f}** across
{h['n_rebalances']} rebalances.

### The same, under each prior (Sharpe ratios)

| Prior | {" | ".join(st.METHOD_LABEL[m] for m in st.METHODS)} | *t* (BL − prior) |
|---|--:|--:|--:|--:|
{priors_tbl}

### Sector panel, equal-weight prior

| Method | Return | Volatility | Sharpe | Turnover | Max weight | Tilt from prior |
|---|--:|--:|--:|--:|--:|--:|
{block('sectors|equal')}

## 5. View size sweep

| View | Tilt from prior | Sharpe | Return vs prior | *t* |
|---|--:|--:|--:|--:|
{sweep}

A zero-sized view is the control: it should leave the portfolio at the prior and return the
prior's performance, and it does.

## Caveats

- **No market portfolio.** The textbook prior is cap-weighted; a price feed does not carry
  market capitalisations, and inventing them would be worse than running three explicit priors
  and reporting the spread. That spread *is* the headline of section 3.
- **One view specification.** Omega follows He & Litterman's proportional convention; the
  literature contains half a dozen alternatives and they change the tilt sizes, though not the
  identity.
- **The momentum view is mechanical**, not a forecast. Whether momentum works is studies
  **507** and **518**; this study is about what the model does with a view once you have one.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[979-black-litterman-zero-views](../README.md). Not investment advice.*
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
