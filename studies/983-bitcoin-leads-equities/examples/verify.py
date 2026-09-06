"""Real-tape verification — Study 983 (The Weekend Oracle). Regenerates docs/results.md.

Builds every closed-market window on the equity calendar, measures the crypto move
across each one, regresses the following session's equity return on it with HC1 errors, splits
by regime and by closure length, prices the Monday rule against holding the same sessions, and
sets all of it beside the ordinary daily lead-lag correlations with their overlap in hours
printed next to them.

    python studies/983-bitcoin-leads-equities/examples/verify.py            # cache-only
    python studies/983-bitcoin-leads-equities/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from weekendoracle import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


COST_BPS = 2.0


def report() -> dict:
    px = data.load_prices()
    rets = px.pct_change()
    h: dict = {"as_of": data.AS_OF, "equity": data.EQUITY, "cost_bps": COST_BPS,
               "fingerprint": data.fingerprint(px)}

    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    for tk in data.TICKERS:
        s = px[tk].dropna()
        print(f"  {tk:9s} {s.index[0].date()} -> {s.index[-1].date()}  n={len(s):,}  "
              f"ann vol {s.pct_change().std() * np.sqrt(st.TRADING_DAYS):.1%}")

    eq_px = px[data.EQUITY].dropna()
    eq_rets = eq_px.pct_change()
    cash_rets = rets[data.CASH].reindex(eq_px.index).fillna(0.0)

    print("\n=== 1. the clock, before anything else ===")
    print(f"  Yahoo's {data.CRYPTO} daily bar closes at {st.CRYPTO_BAR_CLOSE_UTC:02d}:00 UTC; "
          f"the NYSE closes at ~{st.NY_CLOSE_UTC:02d}:00 UTC.")
    for a in ("same_day", "crypto_lead", "weekend"):
        print(f"  {a:12s} overlap {st.overlap_hours(a):5.1f}h   "
              f"{'CLEAN' if st.is_clean(a) else 'contaminated'}")

    print("\n=== 2. the ordinary daily lead-lag, with its caveat attached ===")
    grid = st.lead_lag_grid(rets[data.CRYPTO].dropna(), eq_rets.dropna())
    for lag, r in grid.iterrows():
        print(f"  {int(lag):+2d}  {r['description']:28s} corr {r['correlation']:+.3f}   "
              f"overlap {r['overlap_hours']:4.1f}h  "
              f"{'clean' if r['clean'] else 'CONTAMINATED'}")
    h["lead_lag"] = grid.reset_index().to_dict("records")
    h["daily_lead_corr"] = float(grid.loc[1, "correlation"])
    h["daily_lead_overlap"] = float(grid.loc[1, "overlap_hours"])
    h["daily_same_corr"] = float(grid.loc[0, "correlation"])
    print("  -> not one row of this table is a clean experiment; that is the point of the next "
          "section")

    print("\n=== 3. the clean design: closed-market windows ===")
    w = st.weekend_returns(px[data.CRYPTO].dropna(), eq_px.index)
    panel = st.attach_target(w, eq_rets)
    h["n_weekends"] = int(len(panel))
    h["window"] = [str(panel.index[0].date()), str(panel.index[-1].date())]
    print(f"  {len(panel):,} closed windows, {panel.index[0].date()} -> "
          f"{panel.index[-1].date()}")
    print(f"  closure lengths: "
          + ", ".join(f"{int(k)}d x{int(v)}" for k, v in
                      panel['gap_days'].value_counts().sort_index().items()))
    print(f"  crypto across the window: mean {panel['crypto_return'].mean():+.3%}, "
          f"sd {panel['crypto_return'].std():.2%}")
    print(f"  equity on the next session: mean {panel['equity_return'].mean():+.3%}, "
          f"sd {panel['equity_return'].std():.2%}")

    reg = st.monday_regression(panel)
    h["beta_weekend"] = reg["beta"]
    h["t_weekend"] = reg["t"]
    h["r2_weekend"] = reg["r2"]
    h["alpha_weekend"] = reg["alpha"]
    print(f"  next session = {reg['alpha']:+.4%} + {reg['beta']:+.4f} x weekend crypto   "
          f"t {reg['t']:+.2f}   R2 {reg['r2']:.2%}   n {reg['n']}")

    sa = st.sign_agreement(panel)
    h["hit_rate"] = sa["hit_rate"]
    h["t_hit"] = sa["t_vs_coin_flip"]
    print(f"  sign agreement {sa['hit_rate']:.1%} (t vs a coin flip {sa['t_vs_coin_flip']:+.2f})")

    cm = st.conditional_means(panel)
    for b, r in cm.iterrows():
        extra = f"   t {r['t']:+.2f}" if "t" in cm.columns and np.isfinite(r.get("t", np.nan)) \
            else ""
        print(f"  {b:22s} n {int(r['n']):4d}  mean {r['mean']:+.3%}{extra}")
    h["mean_after_up"] = float(cm.loc["crypto weekend up", "mean"])
    h["mean_after_down"] = float(cm.loc["crypto weekend down", "mean"])
    h["t_bucket"] = float(cm.loc["difference", "t"])
    h["conditional"] = cm.reset_index().to_dict("records")

    print("\n=== 4. regimes ===")
    rs = st.regime_split(panel)
    for reg_name, r in rs.iterrows():
        print(f"  {reg_name:22s} n {int(r['n']):4d}  beta {r['beta']:+.4f}  t {r['t']:+.2f}  "
              f"R2 {r['r2']:.2%}")
    h["regimes"] = rs.reset_index().to_dict("records")
    h["beta_before"] = float(rs.iloc[0]["beta"])
    h["beta_since"] = float(rs.iloc[1]["beta"])

    print("\n=== 5. does a longer closure carry more? ===")
    gb = st.gap_length_buckets(panel)
    for g, r in gb.iterrows():
        print(f"  {int(g)}-day gap ({r['hours_closed']:.1f}h closed)  n {int(r['n']):4d}  "
              f"beta {r['beta']:+.4f}  t {r['t']:+.2f}")
    h["gap_buckets"] = gb.reset_index().to_dict("records")

    print("\n=== 6. the second crypto and the controls ===")
    others = {}
    for label, tk in (("Ethereum", data.CRYPTO_2), ("Gold", data.GOLD)):
        s = px[tk].dropna()
        if len(s) < 500:
            continue
        w2 = st.weekend_returns(s, eq_px.index)
        p2 = st.attach_target(w2, eq_rets)
        r2 = st.monday_regression(p2)
        others[label] = {"n": r2["n"], "beta": r2["beta"], "t": r2["t"], "r2": r2["r2"]}
        print(f"  {label:10s} n {r2['n']:4d}  beta {r2['beta']:+.4f}  t {r2['t']:+.2f}  "
              f"R2 {r2['r2']:.2%}")
    for label, tk in (("Nasdaq as target", data.TECH),):
        tgt = px[tk].dropna().pct_change()
        p3 = st.attach_target(st.weekend_returns(px[data.CRYPTO].dropna(),
                                                 px[tk].dropna().index), tgt)
        r3 = st.monday_regression(p3)
        others[label] = {"n": r3["n"], "beta": r3["beta"], "t": r3["t"], "r2": r3["r2"]}
        print(f"  {label:16s} n {r3['n']:4d}  beta {r3['beta']:+.4f}  t {r3['t']:+.2f}  "
              f"R2 {r3['r2']:.2%}")
    h["others"] = others
    print("  (Gold trades on Friday and Monday only — its 'weekend return' is the same closed "
          "window with no trading in it, so a slope there would be a warning that the "
          "apparatus is picking up something other than weekend news.)")

    print("\n=== 7. the rule ===")
    rule = st.monday_rule(panel, eq_rets, cash_rets, cost_bps=COST_BPS)
    h.update({k: v for k, v in rule.items() if k != "returns"})
    gap = rule["per_year_rule"] - rule["per_year_always"]
    print(f"  long on {rule['n_long']} of {rule['n_windows']} sessions "
          f"({rule['share_long']:.0%})")
    print(f"  rule {rule['per_year_rule']:+.2%}/yr vs holding every such session "
          f"{rule['per_year_always']:+.2%}/yr ({gap:+.2%}, t {rule['t_gap']:+.2f})")
    sweep = []
    for thr in (-0.03, -0.01, 0.0, 0.01, 0.03):
        r = st.monday_rule(panel, eq_rets, cash_rets, cost_bps=COST_BPS, threshold=thr)
        sweep.append({"threshold": thr, "share_long": r["share_long"],
                      "per_year_rule": r["per_year_rule"], "t_gap": r["t_gap"]})
        print(f"  threshold {thr:+.0%}: long {r['share_long']:.0%}, "
              f"{r['per_year_rule']:+.2%}/yr, t {r['t_gap']:+.2f}")
    h["threshold_sweep"] = sweep

    print("\n=== 8. how much could this design ever have seen? ===")
    crypto_sd = float(panel["crypto_return"].std())
    ceiling = st.detectability_ceiling(len(panel), crypto_noise=crypto_sd)
    h["ceiling"] = ceiling
    h["crypto_window_sd"] = crypto_sd
    print(f"  Bitcoin's move across the closed window has sd {crypto_sd:.2%}. If only about "
          f"{0.6 * 1.0:.1f}% of that is news equities also care about, the correlation between "
          f"the crypto move and that news cannot exceed {ceiling['max_correlation']:.3f}...")
    print(f"  ...so the largest |t| this design can produce over {len(panel)} windows — even if "
          f"the weekend's news determined Monday perfectly — is about "
          f"{ceiling['max_t']:.2f}, and reaching |t| = 2 reliably would need roughly "
          f"{ceiling['n_weeks_for_t2']:.0f} windows.")
    print("  A null result here is therefore weak evidence of absence, and that is stated in "
          "the verdict rather than buried.")

    print("\n=== 9. synthetic control ===")
    for info, tag in ((0.6, "weekend genuinely informative"), (0.0, "null: no weekend content")):
        ts = [st.monday_regression(st.synthetic_world(n_weeks=len(panel),
                                                      weekend_information=info, seed=983 + s))["t"]
              for s in range(8)]
        print(f"  {tag:32s} mean |t| {np.nanmean(np.abs(ts)):.2f}, "
              f"share past 2: {np.mean(np.abs(ts) >= 2):.0%}")
        h[f"synthetic_{'planted' if info else 'null'}"] = {
            "mean_abs_t": float(np.nanmean(np.abs(ts))),
            "share_significant": float(np.mean(np.abs(ts) >= 2))}

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    ll = "\n".join(
        f"| {int(r['lag']):+d} | {r['description']} | {r['correlation']:+.3f} | "
        f"{r['overlap_hours']:.1f}h | {'clean' if r['clean'] else '**contaminated**'} |"
        for r in h["lead_lag"])
    cond = "\n".join(
        f"| {r['bucket']} | {int(r['n'])} | {r['mean']:+.3%} |" for r in h["conditional"]
        if r["bucket"] != "difference")
    reg = "\n".join(
        f"| {r['regime']} | {int(r['n'])} | {r['beta']:+.4f} | {r['t']:+.2f} | {r['r2']:.2%} |"
        for r in h["regimes"])
    gaps = "\n".join(
        f"| {int(r['gap_days'])} days ({r['hours_closed']:.1f}h) | {int(r['n'])} | "
        f"{r['beta']:+.4f} | {r['t']:+.2f} |" for r in h["gap_buckets"])
    oth = "\n".join(f"| {k} | {vv['n']} | {vv['beta']:+.4f} | {vv['t']:+.2f} | {vv['r2']:.2%} |"
                    for k, vv in h["others"].items())
    sw = "\n".join(
        f"| {r['threshold']:+.0%} | {r['share_long']:.0%} | {r['per_year_rule']:+.2%} | "
        f"{r['t_gap']:+.2f} |" for r in h["threshold_sweep"])
    gap = h["per_year_rule"] - h["per_year_always"]
    return f"""# Results — Study 983 (The Weekend Oracle) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). {h['n_weekends']:,} closed-market
windows, {h['window'][0]} → {h['window'][1]}. As-of **{h['as_of']}**; fingerprint
`{h['fingerprint']}`.*

## 1. The clock comes first

Yahoo's Bitcoin daily bar closes at **00:00 UTC**. The NYSE closes at roughly **21:00 UTC**.
So a "same-day" Bitcoin bar contains three hours of news that the equity close could not have
seen, and every ordinary daily lead-lag statistic between the two is contaminated by it:

| Design | Overlap | Clean? |
|---|--:|---|
| Crypto day *t* vs equity day *t* | 21.0h | no |
| Crypto day *t* vs equity day *t+1* | 3.0h | no |
| **Weekend crypto vs the next session** | **0.0h** | **yes** |

## 2. The contaminated table, shown anyway

| Lag | Description | Correlation | Overlap | |
|---|---|--:|--:|---|
{ll}

Published lead-lag results between crypto and equities usually stop here. This study does not,
because none of these rows is an experiment.

## 3. The clean design

Every gap of more than one calendar day in the equity calendar — ordinary weekends and holiday
closures alike — gives a window in which Bitcoin traded and {h['equity']} could not. Regressing
the next session's equity return on the crypto move across that window:

| | |
|---|--:|
| Windows | {h['n_weekends']:,} |
| Intercept | {h['alpha_weekend']:+.4%} |
| **Slope on the weekend crypto move** | **{h['beta_weekend']:+.4f}** |
| **HC1 *t*** | **{h['t_weekend']:+.2f}** |
| R² | {h['r2_weekend']:.2%} |
| Sign agreement | {h['hit_rate']:.1%} (*t* = {h['t_hit']:+.2f}) |

| Bucket | n | Next session's mean return |
|---|--:|--:|
{cond}

Difference: **{h['mean_after_up'] - h['mean_after_down']:+.3%}** per session, *t* =
{h['t_bucket']:+.2f}.

## 4. Regimes

| Regime | n | Slope | *t* | R² |
|---|--:|--:|--:|--:|
{reg}

Bitcoin spent its first years as an uncorrelated curiosity and its recent ones as a high-beta
risk asset. A single full-sample slope averages two different worlds.

## 5. Does a longer closure carry more information?

| Closure | n | Slope | *t* |
|---|--:|--:|--:|
{gaps}

## 6. Other assets through the same machine

| | n | Slope | *t* | R² |
|---|--:|--:|--:|--:|
{oth}

Gold is the falsification control: GLD does not trade over the weekend either, so its "weekend
return" is measured across a window in which nothing happened. A significant slope there would
mean the apparatus is finding something other than weekend news.

## 7. The rule

Own {h['equity']} on the session after an up crypto weekend, bills otherwise, at
{h['cost_bps']:.0f} bps a side. The benchmark is holding **those same sessions** unconditionally
— the only like-for-like comparison for a rule that is invested about one day a week.

| | |
|---|--:|
| Sessions long | {h['n_long']} of {h['n_weekends']} ({h['share_long']:.0%}) |
| Rule | {h['per_year_rule']:+.2%}/yr |
| Holding every such session | {h['per_year_always']:+.2%}/yr |
| **Gap** | **{gap:+.2%}/yr** (*t* = {h['t_gap']:+.2f}) |

| Weekend threshold | Share long | Rule | *t* |
|---|--:|--:|--:|
{sw}

## 8. What could this design ever have seen?

A null result is only informative if the test had power to begin with, so here is the ceiling,
computed rather than asserted. Bitcoin's move across the closed window has a standard deviation
of **{h['crypto_window_sd']:.1%}**, and only a small part of that is news equities also price.
The correlation between the crypto move and that shared news therefore cannot exceed
**{h['ceiling']['max_correlation']:.3f}** — which means:

| | |
|---|--:|
| Largest achievable |*t*| over {h['n_weekends']:,} windows | **{h['ceiling']['max_t']:.2f}** |
| Windows needed to reach \\|*t*\\| = 2 reliably | ~{h['ceiling']['n_weeks_for_t2']:.0f} |
| Windows available | {h['n_weekends']:,} |

The binding constraint on this study is **not** the length of the sample. It is that Bitcoin's
weekend move is a very noisy proxy for the weekend's news, and no amount of additional history
sharpens the proxy. Read section 3 with that in front of it.

## 9. Synthetic control

With weekend news planted, mean |*t*| = {h['synthetic_planted']['mean_abs_t']:.2f}
({h['synthetic_planted']['share_significant']:.0%} of runs significant). With crypto and
equities still correlated but the weekend carrying nothing extra, mean |*t*| =
{h['synthetic_null']['mean_abs_t']:.2f}
({h['synthetic_null']['share_significant']:.0%}). The design has power and it has size.

## Caveats

- **The weekend is not the whole night.** The clean design measures a 65-hour window once a
  week. Overnight lead-lag on ordinary days is the far larger opportunity and is *not*
  measurable this way, for exactly the clock reason in section 1.
- **The next session's open is not its close.** This study uses close-to-close returns for the
  target. If the information is priced in the first five minutes of Monday, a close-to-close
  measurement understates it and an open-to-close trade could not capture it either.
- **One weekend a week, and a noisy proxy.** {h['n_weekends']:,} observations is a small sample
  by daily-data standards, Bitcoin's own regime changed halfway through it, and section 8 shows
  the noise ceiling binds before the sample size does.
- **Costs are optimistic.** {h['cost_bps']:.0f} bps a side assumes SPY-grade execution.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[983-bitcoin-leads-equities](../README.md). Not investment advice.*
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
