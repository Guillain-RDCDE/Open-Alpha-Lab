"""Real-tape verification — Study 965 (The Range Estimators). Regenerates docs/results.md.

Runs the textbook efficiency claim on simulated bars where the true sigma is known,
then measures on five real tapes how much variance the gap-blind estimators miss, and finally
races every estimator — raw and rescaled — as a forecast of the next month's realised
variance, scored with QLIKE and MSE and compared by Diebold-Mariano.

    python studies/965-range-vol-estimators/examples/verify.py            # cache-only
    python studies/965-range-vol-estimators/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from range_vol import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


WINDOW = 21
HORIZON = 21
BURN = 504


def report() -> dict:
    bars = data.load_all()
    h: dict = {"as_of": data.AS_OF, "tickers": list(data.TICKERS),
               "window": WINDOW, "horizon": HORIZON}

    print(f"as-of {data.AS_OF}   window {WINDOW}d   forecast horizon {HORIZON}d")
    for tk, b in bars.items():
        print(f"  {tk:4s} {b.index[0].date()} -> {b.index[-1].date()}  n={len(b):,}  "
              f"bars dropped={data.bad_bar_count(tk)}  fp={data.fingerprint(b)}")
    h["fingerprints"] = {tk: data.fingerprint(b) for tk, b in bars.items()}
    h["windows"] = {tk: [str(b.index[0].date()), str(b.index[-1].date())]
                    for tk, b in bars.items()}
    h["n_bars"] = {tk: int(len(b)) for tk, b in bars.items()}
    h["dropped"] = {tk: data.bad_bar_count(tk) for tk in data.TICKERS}

    # ------------------------------------------- 1) efficiency, where truth is known
    print("\n=== 1. the textbook claim, in the textbook's world (simulated, no gap) ===")
    sim, tr = data.synthetic_ohlc(n_years=40, overnight_share=0.0, seed=965)
    eff = st.efficiency_table(sim, tr["sigma"])
    for c, row in eff.iterrows():
        print(f"  {st.ESTIMATOR_LABEL[c]:24s} MSE {row['mse']:.3e}  efficiency vs CC "
              f"{row['efficiency_vs_cc']:5.2f}x  mean ratio to truth "
              f"{row['mean_ratio_to_truth']:.2f}")
    h["efficiency"] = {c: dict(v) for c, v in eff.to_dict("index").items()}
    h["efficiency_parkinson"] = float(eff.loc["parkinson", "efficiency_vs_cc"])
    h["efficiency_gk"] = float(eff.loc["garman_klass", "efficiency_vs_cc"])
    h["efficiency_rs"] = float(eff.loc["rogers_satchell", "efficiency_vs_cc"])

    print("\n  the same simulation WITH a 35% overnight gap — the world we actually trade:")
    sim2, tr2 = data.synthetic_ohlc(n_years=40, overnight_share=0.35, seed=965)
    eff2 = st.efficiency_table(sim2, tr2["sigma"])
    for c, row in eff2.iterrows():
        print(f"  {st.ESTIMATOR_LABEL[c]:24s} efficiency {row['efficiency_vs_cc']:5.2f}x  "
              f"mean ratio to truth {row['mean_ratio_to_truth']:.2f}")
    h["efficiency_gapped"] = {c: dict(v) for c, v in eff2.to_dict("index").items()}

    # ------------------------------------------------------- 2) bias on the real tape
    print("\n=== 2. what the gap-blind estimators miss on the real tape ===")
    print("  tkr  overnight share   " + "  ".join(f"{c[:9]:>9s}" for c in st.ESTIMATORS))
    bias, on_share = {}, {}
    for tk, b in bars.items():
        t = st.bias_table(b, WINDOW)
        bias[tk] = {c: float(t.loc[c, "ratio_to_cc"]) for c in st.ESTIMATORS}
        on_share[tk] = st.overnight_share(b)
        print(f"  {tk:4s} {on_share[tk]:15.0%}   " +
              "  ".join(f"{bias[tk][c]:9.2f}" for c in st.ESTIMATORS))
    h["bias_ratio"] = bias
    h["overnight_share"] = on_share
    h["overnight_share_spy"] = on_share["SPY"]
    h["ratio_parkinson_spy"] = bias["SPY"]["parkinson"]
    print("  (1.00 = same average variance as close-to-close; the three gap-blind estimators "
          "should land near 1 minus the overnight share)")

    print("\n  annualised vol each estimator would have you quote today (last value):")
    quotes = {}
    for tk, b in bars.items():
        r = st.rolling_variance(b, WINDOW).dropna().iloc[-1]
        quotes[tk] = {c: float(np.sqrt(max(r[c], 0) * st.TRADING_DAYS)) for c in st.ESTIMATORS}
        print(f"  {tk:4s} " + "  ".join(f"{c[:4]}={quotes[tk][c]:6.1%}" for c in st.ESTIMATORS))
    h["current_quotes"] = quotes

    # -------------------------------------------------------- 3) the forecast race
    print(f"\n=== 3. forecasting the next {HORIZON} days' realised variance (raw) ===")
    print("  tkr  estimator                  QLIKE      MSE      DM vs CC   p")
    raw_races = {}
    for tk, b in bars.items():
        t = st.forecast_race(b, WINDOW, HORIZON, BURN)
        raw_races[tk] = {c: dict(v) for c, v in t.to_dict("index").items()}
        for c, row in t.iterrows():
            print(f"  {tk:4s} {st.ESTIMATOR_LABEL[c]:24s} {row['qlike']:8.4f} "
                  f"{row['mse']:.2e}  {row['dm_vs_cc']:+8.2f}  {row['p_vs_cc']:6.3f}")
    h["race_raw"] = raw_races

    print(f"\n=== 3b. the fair fight: each estimator rescaled on the burn-in only ===")
    print("  tkr  estimator                 scale     QLIKE   DM vs CC   p")
    scaled, wins, dms, gains = {}, 0, [], []
    for tk, b in bars.items():
        t = st.scaled_forecast_race(b, WINDOW, HORIZON, BURN)
        scaled[tk] = {c: dict(v) for c, v in t.to_dict("index").items()}
        for c, row in t.iterrows():
            print(f"  {tk:4s} {st.ESTIMATOR_LABEL[c]:24s} {row['scale']:6.2f} "
                  f"{row['qlike']:9.4f}  {row['dm_vs_cc']:+8.2f}  {row['p_vs_cc']:6.3f}")
        best = t.drop(index=["close_close"])["qlike"].idxmin()
        if t.loc[best, "qlike"] < t.loc["close_close", "qlike"]:
            wins += 1
        dms.append(t.loc[best, "dm_vs_cc"])
        gains.append(1.0 - t.loc[best, "qlike"] / t.loc["close_close", "qlike"])
    h["race_scaled"] = scaled
    h["n_qlike_wins"] = int(wins)
    h["pooled_dm"] = float(np.nanmean(dms))

    # which estimator wins most often, and by how much on SPY
    counts = {}
    for tk in data.TICKERS:
        t = pd.DataFrame(scaled[tk]).T
        counts[t.drop(index=["close_close"])["qlike"].idxmin()] = counts.get(
            t.drop(index=["close_close"])["qlike"].idxmin(), 0) + 1
    best_est = max(counts, key=counts.get)
    spy = pd.DataFrame(scaled["SPY"]).T
    h["best_estimator"] = st.ESTIMATOR_LABEL[best_est]
    h["best_estimator_key"] = best_est
    h["best_estimator_wins"] = int(counts[best_est])
    h["best_qlike_gain"] = float(1.0 - spy.loc[best_est, "qlike"] / spy.loc["close_close", "qlike"])
    print(f"\n  best rescaled estimator by tape: " +
          ", ".join(f"{st.ESTIMATOR_LABEL[k]} x{v}" for k, v in counts.items()))
    print(f"  beats close-to-close on QLIKE on {wins}/{len(data.TICKERS)} tapes; pooled DM "
          f"{h['pooled_dm']:+.2f}; SPY QLIKE improvement {h['best_qlike_gain']:.1%}")

    print("\n=== window sensitivity (SPY, rescaled, best estimator) ===")
    sens = []
    for w in (5, 10, 21, 63):
        t = st.scaled_forecast_race(bars["SPY"], w, HORIZON, BURN)
        g = 1.0 - t.loc[best_est, "qlike"] / t.loc["close_close", "qlike"]
        sens.append({"window": w, "qlike_gain": float(g), "dm": float(t.loc[best_est, "dm_vs_cc"])})
        print(f"  window {w:3d}d: QLIKE improvement {g:+.1%}, DM {t.loc[best_est, 'dm_vs_cc']:+.2f}")
    h["window_sensitivity"] = sens

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Usefulness: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    eff = "\n".join(
        f"| {st.ESTIMATOR_LABEL[c]} | {r['mse']:.3e} | **{r['efficiency_vs_cc']:.2f}x** | "
        f"{r['mean_ratio_to_truth']:.2f} |"
        for c, r in h["efficiency"].items())
    effg = "\n".join(
        f"| {st.ESTIMATOR_LABEL[c]} | {r['efficiency_vs_cc']:.2f}x | {r['mean_ratio_to_truth']:.2f} |"
        for c, r in h["efficiency_gapped"].items())
    bias = "\n".join(
        f"| {tk} | {h['overnight_share'][tk]:.0%} | " +
        " | ".join(f"{h['bias_ratio'][tk][c]:.2f}" for c in st.ESTIMATORS) + " |"
        for tk in h["tickers"])
    scaled = "\n".join(
        f"| {tk} | {st.ESTIMATOR_LABEL[c]} | {r['scale']:.2f} | {r['qlike']:.4f} | "
        f"{r['dm_vs_cc']:+.2f} | {r['p_vs_cc']:.3f} |"
        for tk in h["tickers"] for c, r in h["race_scaled"][tk].items())
    sens = "\n".join(f"| {r['window']}d | {r['qlike_gain']:+.1%} | {r['dm']:+.2f} |"
                     for r in h["window_sensitivity"])
    stamp = "\n".join(f"| {tk} | {h['windows'][tk][0]} → {h['windows'][tk][1]} | "
                      f"{h['n_bars'][tk]:,} | {h['dropped'][tk]} | `{f}` |"
                      for tk, f in h["fingerprints"].items())
    heads = " | ".join(st.ESTIMATOR_LABEL[c].split(" ")[0] for c in st.ESTIMATORS)
    dashes = "|".join(["--:"] * len(st.ESTIMATORS))
    return f"""# Results — Study 965 (The Range Estimators) on simulation and on the real tape

*Generated by [`examples/verify.py`](../examples/verify.py). Daily OHLC(V) bars
(`yfinance`, `auto_adjust=True`) for SPY, QQQ, IWM, GLD, TLT. Estimators: close-to-close,
Parkinson (1980), Garman-Klass (1980), Rogers-Satchell (1991), Yang-Zhang (2000). Rolling
window **{h['window']} sessions**, forecast horizon **{h['horizon']} sessions**, burn-in 504
sessions. As-of **{h['as_of']}**.*

## Data stamp

| Ticker | Window | Bars | Dropped | Fingerprint |
|---|---|--:|--:|---|
{stamp}

## 1. The textbook claim, tested where truth exists

Forty simulated years with **no overnight gap** and a known daily sigma — the world
Parkinson's derivation assumes:

| Estimator | MSE vs the truth | Efficiency vs close-to-close | Mean ratio to truth |
|---|--:|--:|--:|
{eff}

The same simulation **with a 35% overnight gap** — the world that actually exists:

| Estimator | Efficiency vs close-to-close | Mean ratio to truth |
|---|--:|--:|
{effg}

The efficiency claim is not wrong; it is *conditional*, and the condition is the one thing
every equity market violates every night.

## 2. What the gap-blind estimators miss on the real tape

Average variance relative to close-to-close (1.00 = same level):

| Ticker | Overnight share of daily variance | {heads} |
|---|--:|{dashes}|
{bias}

## 3. The forecast race, rescaled (the fair fight)

Each estimator is scaled by the ratio of mean close-to-close variance to its own, estimated
**on the burn-in window only**, then scored on QLIKE against the realised variance of the
next {h['horizon']} sessions. Diebold-Mariano is HAC-corrected because the overlapping
windows make the loss differential strongly autocorrelated.

| Ticker | Estimator | Scale | QLIKE | DM vs CC | p |
|---|---|--:|--:|--:|--:|
{scaled}

Best rescaled estimator overall: **{h['best_estimator']}** (best on
{h['best_estimator_wins']} of {len(h['tickers'])} tapes). It beat close-to-close on QLIKE on
**{h['n_qlike_wins']} of {len(h['tickers'])}** tapes with a pooled DM of
**{h['pooled_dm']:+.2f}**; on SPY the QLIKE improvement is **{h['best_qlike_gain']:.1%}**.

### Window sensitivity (SPY)

| Rolling window | QLIKE improvement | DM |
|---|--:|--:|
{sens}

## What this study does not claim

- **No intraday truth.** Without minute bars there is no realised-variance benchmark on the
  real tape, so section 1's efficiency claim is tested only in simulation and section 3
  scores *forecasts*, not estimates. This is stated rather than papered over.
- **Yahoo's high and low** are consolidated-tape extremes and include prints a real trader
  could not have transacted at; that inflates every range estimator slightly and equally.
- **No costs, no strategy.** This is a measurement study. What a better variance estimate is
  worth in a position-sizing rule is a different question (and this desk's 633 and 898).

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Usefulness: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[965-range-vol-estimators](../README.md). Not investment advice.*
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
