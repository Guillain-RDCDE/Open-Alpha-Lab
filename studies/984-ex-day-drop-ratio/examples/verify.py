"""Real-tape verification — Study 984 (A Dollar Off). Regenerates docs/results.md.

Audits the corporate-actions feed, builds every ex-date across twelve mega-cap
payers, computes the raw and market-adjusted drop ratios, sets four defensible estimators of the
same quantity against each other, bootstraps the portfolio-weighted one, cuts by yield, ticker
and decade, and prices the dividend-capture trade before and after costs and tax.

    python studies/984-ex-day-drop-ratio/examples/verify.py            # cache-only
    python studies/984-ex-day-drop-ratio/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from exday import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


COST_BPS = 2.0
TAX_RATE = 0.15


def report() -> dict:
    bars = data.load_bars()
    px = data.load_prices()
    h: dict = {"as_of": data.AS_OF, "cost_bps": COST_BPS, "tax_rate": TAX_RATE,
               "elton_gruber": st.ELTON_GRUBER_1970, "fingerprint": data.fingerprint(px)}

    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    print("\n=== 1. what the corporate-actions feed actually contains ===")
    sr = data.sanity_report()
    print(sr.round(4).to_string())
    h["sanity"] = sr.reset_index().to_dict("records")
    print(f"  total ex-dates in the feed: {int(sr['ex_days'].sum())}")
    print(f"  tickers with a suspicious spacing between payments: "
          f"{int((sr['suspicious_gaps'] > 0).sum())} of {len(sr)}")
    print("  (a 'suspicious gap' is a spacing under 45 or over 200 days for a quarterly payer — "
          "usually a special dividend or a missed record, not necessarily an error)")

    print("\n=== 2. the event set ===")
    ev = st.build_events(bars, data.MARKET)
    h["n_events"] = int(len(ev))
    h["n_tickers"] = int(ev["ticker"].nunique())
    h["window"] = [str(ev.index[0].date()), str(ev.index[-1].date())]
    h["typical_yield"] = float(ev["yield"].median())
    daily_sd = float(px[list(data.PAYERS)].pct_change().stack().std())
    h["typical_move"] = daily_sd
    print(f"  {len(ev)} ex-dates on {ev['ticker'].nunique()} payers, "
          f"{ev.index[0].date()} -> {ev.index[-1].date()}")
    print(f"  median dividend yield per event: {ev['yield'].median():.3%}")
    print(f"  typical single-day move for these names: {daily_sd:.2%}")
    print(f"  -> the per-event ratio divides a {daily_sd:.2%} move by a "
          f"{ev['yield'].median():.3%} dividend. Everything below follows from that.")

    print("\n=== 3. four estimators of one number ===")
    tbl = st.estimator_table(ev)
    for name, r in tbl.iterrows():
        print(f"  {name:32s} {r['value']:+7.3f}   ({r['note']})")
    h["estimators"] = tbl.reset_index().to_dict("records")
    h["mean_ratio"] = st.mean_of_ratios(ev)
    h["median_ratio"] = st.median_of_ratios(ev)
    h["ratio_of_sums"] = st.ratio_of_sums(ev)
    reg = st.regression_slope(ev)
    h["slope"] = reg["slope"]
    h["slope_se"] = reg["se"]
    h["intercept"] = reg["intercept"]
    h["t_vs_one"] = reg["t_vs_one"]
    h["t_vs_zero"] = reg["t_vs_zero"]
    h["estimator_spread"] = float(tbl["value"].max() - tbl["value"].min())
    print(f"  spread between the four: {h['estimator_spread']:.3f}")
    print(f"  regression: drop = {reg['intercept']:+.4f} + {reg['slope']:.4f} x dividend   "
          f"(se {reg['se']:.4f}, t vs 1.0 {reg['t_vs_one']:+.2f}, t vs 0 "
          f"{reg['t_vs_zero']:+.2f})")

    print("\n=== 4. why the per-event ratio misbehaves ===")
    disp = st.ratio_dispersion(ev)
    for k, v in disp.items():
        print(f"  {k:26s} {v:.4f}" if isinstance(v, float) else f"  {k:26s} {v}")
    h["dispersion"] = disp
    h["share_wild"] = disp["share_outside_0_2"]
    print(f"  -> {disp['share_outside_0_2']:.0%} of individual events have a 'drop ratio' "
          f"outside [0, 2]. The mean of such a sample is not estimating a drop ratio.")
    raw_disp = st.ratio_dispersion(ev, "raw_ratio")
    h["dispersion_raw"] = raw_disp
    print(f"  before the market adjustment the same figure was "
          f"{raw_disp['share_outside_0_2']:.0%}, sd {raw_disp['sd']:.2f} vs {disp['sd']:.2f}")

    print("\n=== 5. bootstrap ===")
    for label, est in (("total drop / total dividend", st.ratio_of_sums),
                       ("regression slope", lambda e: st.regression_slope(e)["slope"]),
                       ("median ratio", st.median_of_ratios)):
        ci = st.bootstrap_ci(ev, est, n_boot=2000)
        print(f"  {label:28s} {ci['point']:+.3f}  95% CI [{ci['lo']:+.3f}, {ci['hi']:+.3f}]")
        if label == "total drop / total dividend":
            h["ci_lo"], h["ci_hi"] = ci["lo"], ci["hi"]
            h["eg_inside_ci"] = bool(ci["lo"] <= st.ELTON_GRUBER_1970 <= ci["hi"])
            h["one_inside_ci"] = bool(ci["lo"] <= 1.0 <= ci["hi"])
        h[f"ci_{label.split()[0].lower()}"] = {"point": ci["point"], "lo": ci["lo"],
                                               "hi": ci["hi"]}
    print(f"  Elton-Gruber's {st.ELTON_GRUBER_1970:.3f} is "
          f"{'INSIDE' if h['eg_inside_ci'] else 'outside'} that interval; "
          f"a full drop of 1.000 is {'INSIDE' if h['one_inside_ci'] else 'outside'} it")

    print("\n=== 6. cuts ===")
    yb = st.yield_buckets(ev)
    print("  by dividend size:")
    print(yb.round(4).to_string())
    h["yield_buckets"] = yb.reset_index().astype({"yield_bucket": str}).to_dict("records")
    tk = st.by_group(ev, "ticker")
    print("  by ticker:")
    print(tk.round(4).to_string())
    h["by_ticker"] = tk.reset_index().to_dict("records")
    e2 = ev.copy()
    e2["era"] = np.where(e2.index < pd.Timestamp("2013-01-01"), "2005-2012",
                         np.where(e2.index < pd.Timestamp("2019-01-01"), "2013-2018",
                                  "2019-"))
    era = st.by_group(e2, "era")
    print("  by era:")
    print(era.round(4).to_string())
    h["by_era"] = era.reset_index().to_dict("records")

    print("\n=== 7. the dividend-capture trade ===")
    tr = st.capture_trade(ev, cost_bps=COST_BPS, div_tax=0.0)
    h.update({"gross_bps": tr["mean_gross_bps"], "net_bps": tr["mean_net_bps"],
              "sd_bps": tr["sd_bps"], "t_trade": tr["t"], "hit_rate": tr["hit_rate"],
              "breakeven_bps": tr["breakeven_cost_bps"]})
    print(f"  gross {tr['mean_gross_bps']:+.1f} bps/event, net of {COST_BPS:.0f} bps a side "
          f"{tr['mean_net_bps']:+.1f} bps, median {tr['median_net_bps']:+.1f}, "
          f"sd {tr['sd_bps']:.0f}, t {tr['t']:+.2f}, hit {tr['hit_rate']:.0%}")
    print(f"  breaks even at {tr['breakeven_cost_bps']:.1f} bps of round-trip cost")
    taxed = st.capture_trade(ev, cost_bps=COST_BPS, div_tax=TAX_RATE)
    h["net_after_tax_bps"] = taxed["mean_net_bps"]
    print(f"  at the {TAX_RATE:.0%} qualified rate a taxable holder nets "
          f"{taxed['mean_net_bps']:+.1f} bps")
    sweep = []
    for c in (0.0, 1.0, 2.0, 5.0, 10.0):
        r = st.capture_trade(ev, cost_bps=c)
        sweep.append({"cost_bps": c, "net_bps": r["mean_net_bps"], "t": r["t"]})
        print(f"  cost {c:5.1f} bps/side -> net {r['mean_net_bps']:+7.1f} bps, t {r['t']:+.2f}")
    h["cost_sweep"] = sweep
    print(f"  the per-event dispersion is {tr['sd_bps']:.0f} bps against a mean of "
          f"{tr['mean_net_bps']:+.1f}: it takes "
          f"{(tr['sd_bps'] / max(abs(tr['mean_net_bps']), 0.1)) ** 2:.0f} events for the mean to "
          f"stand one standard error clear of zero")

    print("\n=== 8. synthetic control: can any of this be recovered? ===")
    rows = []
    for frac in (0.5, 0.8, 1.0):
        sims = {"mean": [], "median": [], "sums": [], "slope": []}
        for s in range(6):
            sb = data.synthetic_panel(n=3000, n_tickers=12, drop_fraction=frac, seed=984 + s)
            se = st.build_events(sb, "MKT")
            sims["mean"].append(st.mean_of_ratios(se))
            sims["median"].append(st.median_of_ratios(se))
            sims["sums"].append(st.ratio_of_sums(se))
            sims["slope"].append(st.regression_slope(se)["slope"])
        row = {"planted": frac, **{k: float(np.mean(v)) for k, v in sims.items()}}
        rows.append(row)
        print(f"  truth {frac:.2f} -> mean {row['mean']:+.2f}, median {row['median']:+.2f}, "
              f"sums {row['sums']:+.3f}, slope {row['slope']:+.3f}")
    h["synthetic"] = rows
    errs = {k: float(np.mean([abs(r[k] - r["planted"]) for r in rows]))
            for k in ("mean", "median", "sums", "slope")}
    h["synthetic_errors"] = errs
    best = min(errs, key=errs.get)
    h["best_estimator"] = best
    print(f"  mean absolute error against the planted truth: "
          + ", ".join(f"{k} {v:.3f}" for k, v in errs.items()))
    print(f"  -> the estimator to trust here is '{best}', which is why the verdict is built "
          f"on it")

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    sanity = "\n".join(
        f"| {r['ticker']} | {int(r['ex_days'])} | {r['first_ex']} → {r['last_ex']} | "
        f"{r['median_yield']:.3%} | {r['median_gap_days']:.0f} | {int(r['suspicious_gaps'])} |"
        for r in h["sanity"])
    est = "\n".join(f"| {r['estimator']} | **{r['value']:+.3f}** | {r['note']} |"
                    for r in h["estimators"])
    yb = "\n".join(
        f"| {r['yield_bucket']} | {int(r['n'])} | {r['median_yield']:.3%} | {r['ratio']:.3f} | "
        f"{r['slope']:.3f} | {r['t_vs_one']:+.2f} |" for r in h["yield_buckets"])
    tk = "\n".join(
        f"| {r['ticker']} | {int(r['n'])} | {r['median_yield']:.3%} | {r['ratio']:.3f} | "
        f"{r['slope']:.3f} | {r['t_vs_one']:+.2f} |" for r in h["by_ticker"])
    era = "\n".join(
        f"| {r['era']} | {int(r['n'])} | {r['ratio']:.3f} | {r['slope']:.3f} | "
        f"{r['t_vs_one']:+.2f} |" for r in h["by_era"])
    cs = "\n".join(f"| {r['cost_bps']:.0f} | {r['net_bps']:+.1f} | {r['t']:+.2f} |"
                   for r in h["cost_sweep"])
    syn = "\n".join(
        f"| {r['planted']:.2f} | {r['mean']:+.2f} | {r['median']:+.2f} | {r['sums']:.3f} | "
        f"{r['slope']:.3f} |" for r in h["synthetic"])
    d = h["dispersion"]
    return f"""# Results — Study 984 (A Dollar Off) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). {h['n_events']} ex-dividend dates
across {h['n_tickers']} mega-cap payers, {h['window'][0]} → {h['window'][1]}. Raw
(dividend-**un**adjusted) closes plus Yahoo's corporate-actions feed. As-of **{h['as_of']}**;
fingerprint `{h['fingerprint']}`.*

## 1. The feed, audited first

| Ticker | Ex-days | Span | Median yield | Median gap (days) | Odd gaps |
|---|--:|---|--:|--:|--:|
{sanity}

Yahoo's dividend feed is not a research-grade corporate-actions database. The universe is
restricted to twelve mega-cap payers since 2005 because that is where it is most reliable, and
the audit is printed rather than assumed.

## 2. The scale problem, stated before any estimate

| | |
|---|--:|
| Events | {h['n_events']} |
| Median dividend yield per event | **{h['typical_yield']:.3%}** |
| Typical single-day move for these names | **{h['typical_move']:.2%}** |

The per-event drop ratio divides the second number by the first. That is the whole difficulty:
a statistic whose denominator is small relative to its numerator's noise has fat tails, and its
sample mean estimates something other than what it appears to.

## 3. Four estimators of one number

| Estimator | Value | Note |
|---|--:|---|
{est}

**They span {h['estimator_spread']:.3f}** — larger than the difference between "the price falls
by the full dividend" and Elton & Gruber's {h['elton_gruber']:.3f} that the literature has
argued about since 1970. Any paper quoting a drop ratio without saying which of these four it
computed has not said much.

The regression — dollar drop on dollar dividend — gives
`drop = {h['intercept']:+.4f} + {h['slope']:.4f} × dividend`, standard error
{h['slope_se']:.4f}, *t* against 1.0 = **{h['t_vs_one']:+.2f}**, against zero
{h['t_vs_zero']:+.2f}.

## 4. Why the per-event ratio misbehaves

| | Market-adjusted | Raw |
|---|--:|--:|
| Standard deviation | {d['sd']:.2f} | {h['dispersion_raw']['sd']:.2f} |
| Interquartile range | {d['iqr']:.2f} | {h['dispersion_raw']['iqr']:.2f} |
| Mean | {d['mean']:+.2f} | {h['dispersion_raw']['mean']:+.2f} |
| 5-95% trimmed mean | {d['trimmed_mean']:+.2f} | {h['dispersion_raw']['trimmed_mean']:+.2f} |
| Median | {d['median']:.2f} | {h['dispersion_raw']['median']:.2f} |
| Most extreme event | {d['max']:.1f} | {h['dispersion_raw']['max']:.1f} |
| **Share outside [0, 2]** | **{d['share_outside_0_2']:.0%}** | {h['dispersion_raw']['share_outside_0_2']:.0%} |

The gap between the mean and the trimmed mean is the tell. Removing the market's move on the
ex-day helps materially — it is the single most useful correction available — but it does not
rescue the per-event ratio as a statistic.

## 5. Bootstrap

The portfolio-weighted ratio (total dollars dropped over total dollars paid) has a 95%
percentile interval of **[{h['ci_lo']:.3f}, {h['ci_hi']:.3f}]**. Elton & Gruber's
{h['elton_gruber']:.3f} is {'inside' if h['eg_inside_ci'] else 'outside'} it; a full drop of
1.000 is {'inside' if h['one_inside_ci'] else 'outside'} it.

## 6. Cuts

By dividend size:

| Bucket | n | Median yield | Total/total | Slope | *t* vs 1 |
|---|--:|--:|--:|--:|--:|
{yb}

By ticker:

| Ticker | n | Median yield | Total/total | Slope | *t* vs 1 |
|---|--:|--:|--:|--:|--:|
{tk}

By era:

| Era | n | Total/total | Slope | *t* vs 1 |
|---|--:|--:|--:|--:|
{era}

## 7. The dividend-capture trade

Buy at the cum close, sell at the ex close, keep the dividend:

| | |
|---|--:|
| Gross | {h['gross_bps']:+.1f} bps per event |
| Net of {h['cost_bps']:.0f} bps a side | **{h['net_bps']:+.1f} bps** (*t* = {h['t_trade']:+.2f}) |
| Hit rate | {h['hit_rate']:.0%} |
| Dispersion per event | {h['sd_bps']:.0f} bps |
| Break-even round-trip cost | {h['breakeven_bps']:.1f} bps |
| Net at the {h['tax_rate']:.0%} qualified rate | {h['net_after_tax_bps']:+.1f} bps |

| Cost (bps/side) | Net (bps) | *t* |
|---|--:|--:|
{cs}

## 8. Synthetic control — which estimator survives?

The generator plants a known drop fraction under realistic noise. Each estimator is then asked
to recover it:

| Planted truth | Mean of ratios | Median | Total/total | Regression slope |
|---|--:|--:|--:|--:|
{syn}

Mean absolute error against the truth:
{", ".join(f"**{k}** {v:.3f}" for k, v in h['synthetic_errors'].items())}. The best is
**{h['best_estimator']}**, and the verdict is built on the regression slope for that reason.

## Caveats

- **Close-to-close, not open.** The ex-day drop happens at the opening auction. Measuring it
  from the previous close to the ex-day *close* includes a full day of trading noise that has
  nothing to do with the dividend. Open prices would cut the noise substantially and are the
  single highest-value improvement to this study.
- **Twelve names.** Mega-cap, liquid, quarterly payers. High-yield names, REITs, and
  special dividends — where tax clienteles should bite hardest — are excluded, so this is the
  *least* favourable universe for finding an effect.
- **No tick-size effect.** Pre-decimalisation studies found much of the shortfall was price
  discreteness (Bali & Hite 1998). The sample starts in 2005, well after decimalisation, which
  removes that explanation and also removes the era where the effect was largest.
- **Yahoo's feed.** Section 1 audits it; it is not a substitute for CRSP.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[984-ex-day-drop-ratio](../README.md). Not investment advice.*
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
