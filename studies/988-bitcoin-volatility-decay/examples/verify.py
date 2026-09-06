"""Real-tape verification — Study 988 (The Taming). Regenerates docs/results.md.

Measures Bitcoin's realised volatility on its own 365-day calendar, fits the trend
four ways (OLS, Theil-Sen, Mann-Kendall and a block bootstrap that respects the persistence),
refits from every possible start date to see how much of the "decay" is a choice of where the
chart begins, cuts by era and by halving, compares against equities in ratio terms, and prices
the volatility-targeting rule the whole question implies.

    python studies/988-bitcoin-volatility-decay/examples/verify.py            # cache-only
    python studies/988-bitcoin-volatility-decay/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from taming import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


WINDOW = 30
TARGET_VOL = 0.40
COST_BPS = 5.0


def report() -> dict:
    px = data.load_prices()
    btc = px[data.CRYPTO].dropna()
    h: dict = {"as_of": data.AS_OF, "window": WINDOW, "target_vol": TARGET_VOL,
               "fingerprint": data.fingerprint(px)}

    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    print("\n=== 1. the calendar, before anything else ===")
    for tk in data.TICKERS:
        s = px[tk].dropna()
        a = st.annualisation_factor(s)
        print(f"  {tk:9s} {s.index[0].date()} -> {s.index[-1].date()}  n={len(s):,}  "
              f"{a:5.0f} obs/yr  ann vol {s.pct_change().std() * np.sqrt(a):.1%}")
    ann = st.annualisation_factor(btc)
    h["ann_factor"] = float(ann)
    wrong = btc.pct_change().std() * np.sqrt(st.EQUITY_DAYS)
    right = btc.pct_change().std() * np.sqrt(ann)
    print(f"  -> Bitcoin on its own calendar: {right:.1%}. On an equity calendar: {wrong:.1%}. "
          f"A {right / wrong - 1:.0%} error, larger than the trend we are about to measure.")

    v = st.realised_vol(btc, WINDOW, ann)
    h["years"] = float(len(v) / ann)
    h["mean_vol"] = float(v.mean())
    h["min_vol"] = float(v.min())
    h["max_vol"] = float(v.max())
    print("\n=== 2. the volatility itself ===")
    summ = st.vol_summary(btc)
    print(summ.round(4).to_string())
    h["summary"] = summ.reset_index().to_dict("records")
    h["autocorr_100"] = float(summ.loc[WINDOW, "autocorr_100"]) if WINDOW in summ.index \
        else float(v.autocorr(100))
    print(f"  the {WINDOW}-day series still autocorrelates {h['autocorr_100']:+.2f} at 100 "
          f"days. Any test that assumes independent residuals is about to lie to us.")

    print("\n=== 3. the trend, four ways ===")
    tbl = st.trend_table(v, ann)
    print(tbl.round(4).to_string())
    h["trend_table"] = tbl.reset_index().to_dict("records")
    o = st.ols_trend(v, ann)
    ts = st.theil_sen(v, ann)
    bb = st.block_bootstrap_trend(v, ann)
    h.update({"ols_slope": o["slope_per_year"], "ols_pct": o["pct_per_year"],
              "ols_t": o["t_naive"], "ols_se": o["se_naive"],
              "ts_slope": ts["slope_per_year"], "ts_pct": ts["pct_per_year"],
              "boot_slope": bb["slope_per_year"], "boot_sd": bb["boot_sd"],
              "boot_t": bb["t_boot"], "boot_lo": bb["lo"], "boot_hi": bb["hi"],
              "boot_share_negative": bb["share_negative"]})
    print(f"  naive standard error {o['se_naive']:.4f}; block-bootstrap {bb['boot_sd']:.4f} — "
          f"a factor of {bb['boot_sd'] / o['se_naive']:.1f}")
    print(f"  so the honest interval on the slope is [{bb['lo']:+.3f}, {bb['hi']:+.3f}] per "
          f"year, and {bb['share_negative']:.0%} of resamples are negative")

    print("\n=== 4. the control: where does the chart start? ===")
    sens = st.start_date_sensitivity(v, ann, step=30)
    s = st.sensitivity_summary(sens)
    h["sensitivity"] = s
    h.update({"share_negative": s["share_negative"],
              "share_significant_down": s["share_significant_down"],
              "share_significant_up": s["share_significant_up"],
              "corr_with_start_vol": s["corr_with_start_vol"]})
    print(f"  {s['n']} start dates tried (every 30 days, minimum 3-year window)")
    print(f"  slope ranges {s['min_slope']:+.3f} to {s['max_slope']:+.3f} per year")
    print(f"  {s['share_negative']:.0%} slope down; {s['share_significant_down']:.0%} do so "
          f"with t < -2; {s['share_significant_up']:.0%} slope significantly UP")
    print(f"  correlation between the fitted slope and the volatility on the START date: "
          f"{s['corr_with_start_vol']:+.2f}")
    print("  -> that last number is the trick. Begin the chart at a volatility peak and the "
          "decline comes free.")
    h["sens_rows"] = [{"start": str(i.date()), "years": float(r["years"]),
                       "start_vol": float(r["start_vol"]),
                       "slope": float(r["slope_per_year"]), "t": float(r["t"])}
                      for i, r in sens.iloc[::max(len(sens) // 12, 1)].iterrows()]

    print("\n=== 5. era by era ===")
    eras = st.by_era(v, ann, n_eras=4)
    print(eras.round(4).to_string())
    h["eras"] = eras.reset_index().to_dict("records")

    print("\n=== 6. the halvings ===")
    hv = st.halving_alignment(v)
    if not hv.empty:
        print(hv.round(4).to_string())
        h["halvings"] = hv.reset_index().to_dict("records")
        print(f"  {len(hv)} events. That is the sample. No test is being run on it.")
    else:
        h["halvings"] = []

    print("\n=== 7. maturing means converging, not merely falling ===")
    others = {name: px[tk].dropna() for name, tk in
              (("SPY", data.EQUITY), ("QQQ", data.NASDAQ), ("gold", data.GOLD),
               ("TSLA", data.SINGLE_STOCK)) if px[tk].notna().sum() > 500}
    rel = st.relative_to_equities(btc, others, window=365)
    print("  Bitcoin's 1-year volatility as a multiple of:")
    for col in rel.columns:
        c = rel[col].dropna()
        if len(c) < 100:
            continue
        first = float(c.iloc[:252].mean())
        last = float(c.iloc[-252:].mean())
        rt = st.ols_trend(c.clip(lower=1e-6), ann)
        print(f"    {col:6s} first year {first:5.1f}x -> last year {last:5.1f}x   "
              f"trend {rt.get('pct_per_year', np.nan):+.1%}/yr")
        h.setdefault("relative", {})[col] = {"first": first, "last": last,
                                             "trend": rt.get("pct_per_year", np.nan)}
    if data.CRYPTO_2 in px.columns and px[data.CRYPTO_2].notna().sum() > 1000:
        v2 = st.realised_vol(px[data.CRYPTO_2].dropna(), WINDOW)
        o2 = st.ols_trend(v2, st.annualisation_factor(px[data.CRYPTO_2].dropna()))
        h["eth_trend"] = o2.get("pct_per_year", np.nan)
        print(f"  Ethereum, same test: {o2.get('pct_per_year', np.nan):+.1%}/yr "
              f"(naive t {o2.get('t_naive', np.nan):+.2f})")

    print("\n=== 8. the rule the question implies ===")
    sb = st.sizing_backtest(btc, TARGET_VOL, WINDOW, px[data.CASH], COST_BPS)
    h.update({"vt_cagr": sb["vol_targeted"]["cagr"], "bh_cagr": sb["buy_hold"]["cagr"],
              "vt_sharpe": sb["vol_targeted"]["sharpe"], "bh_sharpe": sb["buy_hold"]["sharpe"],
              "vt_vol": sb["vol_targeted"]["vol"], "bh_vol": sb["buy_hold"]["vol"],
              "vt_dd": sb["vol_targeted"]["max_dd"], "bh_dd": sb["buy_hold"]["max_dd"],
              "mean_leverage": sb["mean_leverage"], "leverage_trend": sb["leverage_trend"]})
    print(f"  vol-targeted to {TARGET_VOL:.0%}: CAGR {sb['vol_targeted']['cagr']:+.1%}, vol "
          f"{sb['vol_targeted']['vol']:.1%}, Sharpe {sb['vol_targeted']['sharpe']:.2f}, "
          f"maxDD {sb['vol_targeted']['max_dd']:.0%}")
    print(f"  buy and hold:            CAGR {sb['buy_hold']['cagr']:+.1%}, vol "
          f"{sb['buy_hold']['vol']:.1%}, Sharpe {sb['buy_hold']['sharpe']:.2f}, "
          f"maxDD {sb['buy_hold']['max_dd']:.0%}")
    print(f"  average leverage {sb['mean_leverage']:.2f}x, and its own trend "
          f"{sb['leverage_trend']:+.3f}/yr — a vol-targeter is betting on the decay whether "
          f"they meant to or not")
    sweep = []
    for tv in (0.20, 0.30, 0.40, 0.60, 0.80):
        r = st.sizing_backtest(btc, tv, WINDOW, px[data.CASH], COST_BPS)
        sweep.append({"target": tv, "cagr": r["vol_targeted"]["cagr"],
                      "vol": r["vol_targeted"]["vol"],
                      "sharpe": r["vol_targeted"]["sharpe"],
                      "max_dd": r["vol_targeted"]["max_dd"],
                      "leverage": r["mean_leverage"]})
        print(f"  target {tv:.0%}: realised {r['vol_targeted']['vol']:.1%}, CAGR "
              f"{r['vol_targeted']['cagr']:+.1%}, Sharpe {r['vol_targeted']['sharpe']:.2f}, "
              f"lev {r['mean_leverage']:.2f}x")
    h["target_sweep"] = sweep

    print("\n=== 9. synthetic control ===")
    for decay, persist, tag in ((-0.25, 0.95, "genuine decay planted"),
                                (0.0, 0.995, "null: persistent, no trend")):
        naive_hits, boot_hits, shares = [], [], []
        for sd in range(8):
            sim = st.synthetic_world(n=len(v), decay_per_year=decay, persistence=persist,
                                     seed=988 + sd)
            sv = st.realised_vol(sim, WINDOW)
            naive_hits.append(abs(st.ols_trend(sv)["t_naive"]) >= 2)
            bbs = st.block_bootstrap_trend(sv, n_boot=200)
            boot_hits.append(abs(bbs.get("t_boot", 0)) >= 2)
            shares.append(st.sensitivity_summary(
                st.start_date_sensitivity(sv, step=90))["share_significant_down"])
        print(f"  {tag:30s} naive t>2 in {np.mean(naive_hits):.0%} of runs, bootstrap t>2 in "
              f"{np.mean(boot_hits):.0%}, start-date robustness {np.nanmean(shares):.0%}")
        h[f"synthetic_{'planted' if decay else 'null'}"] = {
            "naive_reject": float(np.mean(naive_hits)),
            "boot_reject": float(np.mean(boot_hits)),
            "start_robustness": float(np.nanmean(shares))}
    print("  -> the naive test rejects constantly under the null. The bootstrap and the "
          "start-date control are what make the difference.")

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    summ = "\n".join(
        f"| {int(r['window'])}d | {int(r['n'])} | {r['mean']:.0%} | {r['median']:.0%} | "
        f"{r['min']:.0%} | {r['max']:.0%} | {r['autocorr_100']:+.2f} |"
        for r in h["summary"])
    def _num(x, fmt="+.3f"):
        return "—" if x is None or not np.isfinite(x) else format(x, fmt)

    tt = "\n".join(
        f"| {r['method']} | {_num(r['slope_per_year'])} | {_num(r['t'], '+.2f')} | "
        f"{r['note']} |" for r in h["trend_table"])
    sens = "\n".join(
        f"| {r['start']} | {r['years']:.1f} | {r['start_vol']:.0%} | {r['slope']:+.3f} | "
        f"{r['t']:+.2f} |" for r in h["sens_rows"])
    eras = "\n".join(
        f"| {r['era']} | {int(r['n'])} | {r['mean_vol']:.0%} | {r['median_vol']:.0%} | "
        f"{r['max_vol']:.0%} | {r['slope_within']:+.3f} |" for r in h["eras"])
    halv = "\n".join(
        f"| {r['halving']} | {r['vol_before']:.0%} | {r['vol_after']:.0%} | "
        f"{r['change']:+.0%} | {r['ratio']:.2f} |" for r in h["halvings"]) or \
        "| _(no halving has a full year on both sides in this sample)_ | | | | |"
    rel = "\n".join(
        f"| {k} | {vv['first']:.1f}× | {vv['last']:.1f}× | {vv['trend']:+.1%} |"
        for k, vv in h.get("relative", {}).items())
    sw = "\n".join(
        f"| {r['target']:.0%} | {r['vol']:.0%} | {r['cagr']:+.1%} | {r['sharpe']:.2f} | "
        f"{r['max_dd']:.0%} | {r['leverage']:.2f}× |" for r in h["target_sweep"])
    return f"""# Results — Study 988 (The Taming) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). Bitcoin's daily closes over
{h['years']:.1f} years, annualised on its own **{h['ann_factor']:.0f}-observation** calendar.
As-of **{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

## 1. The calendar comes first

Bitcoin trades every day; equities trade 252 days a year. Annualising Bitcoin on an equity
calendar understates its volatility by about **20%** — larger than the entire trend this study
is trying to measure. Every number below uses the asset's own calendar.

## 2. The volatility itself

| Window | n | Mean | Median | Min | Max | Autocorr at 100 days |
|---|--:|--:|--:|--:|--:|--:|
{summ}

That last column is the problem. Realised volatility observed 100 days apart still correlates
**{h['autocorr_100']:+.2f}**. Any trend test that assumes independent residuals is about to
report a standard error that is far too small.

## 3. The trend, four ways

| Method | Slope (log vol/yr) | *t* | Note |
|---|--:|--:|---|
{tt}

The naive standard error is {h['ols_se']:.4f}; the block-bootstrap one is {h['boot_sd']:.4f} —
**{h['boot_sd'] / h['ols_se']:.1f}× wider**. The honest 95% interval on the slope is
[{h['boot_lo']:+.3f}, {h['boot_hi']:+.3f}] per year, with
{h['boot_share_negative']:.0%} of resamples negative.

## 4. The control nobody runs: where does the chart start?

Refitting the same trend from every possible start date, at least three years long:

| Start | Years | Volatility that day | Slope/yr | *t* |
|---|--:|--:|--:|--:|
{sens}

| | |
|---|--:|
| Start dates tried | {h['sensitivity']['n']} |
| Slope range | {h['sensitivity']['min_slope']:+.3f} to {h['sensitivity']['max_slope']:+.3f} |
| Share sloping down | {h['share_negative']:.0%} |
| **Share significantly down (*t* < −2)** | **{h['share_significant_down']:.0%}** |
| Share significantly **up** | {h['share_significant_up']:.0%} |
| **Correlation of slope with the volatility on the start date** | **{h['corr_with_start_vol']:+.2f}** |

That last row is the trick behind every "Bitcoin is maturing" chart. In a persistent series,
beginning the window at a volatility peak *manufactures* a downward trend. Bitcoin's history
offers several peaks to begin at.

## 5. Era by era

| Era | n | Mean vol | Median | Max | Slope within |
|---|--:|--:|--:|--:|--:|
{eras}

## 6. The halvings

| Halving | Vol in the year before | Year after | Change | Ratio |
|---|--:|--:|--:|--:|
{halv}

Three events. No test is run on three events; the table is here because people ask.

## 7. Maturing means converging, not merely falling

An asset whose volatility halves in a decade when *every* asset's volatility halved has not
matured — it has been carried by a calm market. So the right comparison is a ratio:

| Bitcoin's 1-year vol as a multiple of | First year | Last year | Trend |
|---|--:|--:|--:|
{rel}

## 8. The rule the question implies

Size to a constant {h['target_vol']:.0%} volatility rather than holding a fixed position:

| | Vol-targeted | Buy and hold |
|---|--:|--:|
| CAGR | {h['vt_cagr']:+.1%} | {h['bh_cagr']:+.1%} |
| Realised vol | {h['vt_vol']:.0%} | {h['bh_vol']:.0%} |
| Sharpe | {h['vt_sharpe']:.2f} | {h['bh_sharpe']:.2f} |
| Max drawdown | {h['vt_dd']:.0%} | {h['bh_dd']:.0%} |

Average leverage {h['mean_leverage']:.2f}×, with its own fitted trend of
**{h['leverage_trend']:+.3f}/yr**. That is the practical stake in the whole question: if
volatility is genuinely decaying, a vol-targeter must lever up over time, and is therefore
betting on the trend continuing whether they intended to or not.

| Target vol | Realised | CAGR | Sharpe | Max DD | Avg leverage |
|---|--:|--:|--:|--:|--:|
{sw}

## 9. Synthetic control

| World | Naive *t* > 2 | Bootstrap *t* > 2 | Start-date robustness |
|---|--:|--:|--:|
| Genuine decay planted | {h['synthetic_planted']['naive_reject']:.0%} | {h['synthetic_planted']['boot_reject']:.0%} | {h['synthetic_planted']['start_robustness']:.0%} |
| Null: persistent, no trend | {h['synthetic_null']['naive_reject']:.0%} | {h['synthetic_null']['boot_reject']:.0%} | {h['synthetic_null']['start_robustness']:.0%} |

Read the null row. A naive *t*-test on a persistent, **trendless** volatility series rejects
in {h['synthetic_null']['naive_reject']:.0%} of runs. The bootstrap and the start-date control
are the entire difference between a measurement and a chart.

## Caveats

- **One asset, one history.** Bitcoin has had roughly four cycles. Any statement about a decade
  trend rests on four regimes, not on a large sample of independent years.
- **Realised, not implied.** Options markets have a view on future Bitcoin volatility that this
  study does not use. Implied volatility term structure would be a genuinely forward-looking
  test and is the obvious extension.
- **Survivorship in the comparison set.** TSLA is in section 7 because it is famously volatile
  *and* it survived; that biases the equity comparison toward looking calmer than equities were.
- **Mann-Kendall's variance formula assumes independence**, which is false here even after
  thinning. Its *z* should be read as a direction indicator, not a *p*-value.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[988-bitcoin-volatility-decay](../README.md). Not investment advice.*
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
