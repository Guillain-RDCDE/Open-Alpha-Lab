"""Real-tape verification — Study 999 (The Break). Regenerates docs/results.md.

Grades a sequential CUSUM against planted change points whose dates are known,
compares its measured delay against the Wald bound, traces the delay/false-alarm trade-off
across thresholds, runs it on the real tape against the episodes everyone would name, contrasts
it with a retrospective segmentation that sees the future, and prices a live switching rule
against the same rule given the break dates in advance.

    python studies/999-cusum-change-points/examples/verify.py            # cache-only
    python studies/999-cusum-change-points/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from thebreak import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


THRESHOLD = 5.0
DRIFT = 0.5
RISK_OFF_DAYS = 21
COST_BPS = 5.0

# Episodes a macro desk would name without argument. Used as a plausibility check on the real
# tape, never as ground truth — the exact "date" of a regime change is not a well-defined thing.
KNOWN_EPISODES = {
    "1998 LTCM": "1998-08-17",
    "2000 dot-com peak": "2000-03-24",
    "2008 Lehman": "2008-09-15",
    "2020 COVID": "2020-02-20",
    "2022 inflation": "2022-01-04",
}


def report() -> dict:
    px = data.load_prices()
    rets = px.pct_change()
    h: dict = {"as_of": data.AS_OF, "threshold": THRESHOLD, "drift": DRIFT,
               "risk_off_days": RISK_OFF_DAYS, "cost_bps": COST_BPS,
               "fingerprint": data.fingerprint(px)}

    assets = {tk: rets[tk].dropna() for tk in data.TICKERS
              if tk != data.CASH and rets[tk].notna().sum() > 1500}
    lead = assets[data.EQUITY]
    h["n_assets"] = int(len(assets))
    h["years"] = float(len(lead) / st.TRADING_DAYS)
    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    for tk, s in assets.items():
        print(f"  {tk:9s} {s.index[0].date()} -> {s.index[-1].date()}  n={len(s):,}")

    print("\n=== 1. graded against breaks whose dates are known ===")
    w = st.synthetic_series(n=6000, break_points=[1500, 3000, 4500], vol_shift=3.0)
    c = st.variance_cusum(w["returns"], DRIFT, THRESHOLD)
    dd = st.detection_delay(c["alarm"], w["breaks"])
    ar = st.alarm_rate(c["alarm"])
    h.update({"detection_rate": dd["detection_rate"], "median_delay": dd["median_delay"],
              "mean_delay": dd["mean_delay"], "alarms_per_year": ar["alarms_per_year"],
              "false_alarms": dd["false_alarms"]})
    print(f"  {dd['n_detected']} of {dd['n_breaks']} planted breaks found "
          f"({dd['detection_rate']:.0%})")
    print(f"  median delay {dd['median_delay']:.0f} sessions, mean "
          f"{dd['mean_delay']:.0f}, worst {dd['max_delay']:.0f}")
    print(f"  {ar['alarms_per_year']:.2f} alarms per year, {dd['false_alarms']} of them false")

    print("\n=== 2. how much of the delay is unavoidable? ===")
    # A 3x volatility shift is a change of about log(3) in log squared returns, against a
    # standard deviation of pi/sqrt(2) for log chi-squared(1).
    change_sd = float(np.log(3.0) / (np.pi / np.sqrt(2)))
    th_delay = st.theoretical_delay(change_sd, THRESHOLD, DRIFT)
    h["theoretical_delay"] = th_delay
    h["delay_vs_theory"] = float(dd["median_delay"] / th_delay) if th_delay > 0 else np.nan
    print(f"  a 3x volatility shift is about {change_sd:.2f} sd in log squared returns")
    print(f"  Wald's identity puts the floor at ~{th_delay:.0f} sessions")
    print(f"  measured {dd['median_delay']:.0f} -> {h['delay_vs_theory']:.2f}x the limit")
    print("  a detector near 1x is not slow; the DATA is slow")

    print("\n=== 3. the trade-off curve ===")
    roc = st.roc_curve(w["returns"], w["breaks"], thresholds=(2, 3, 4, 5, 7, 10, 15, 20),
                       drift=DRIFT)
    print(roc.round(3).to_string())
    h["roc"] = roc.reset_index().to_dict("records")
    lo = roc.index.min()
    h["low_threshold"] = float(lo)
    h["low_delay"] = float(roc.loc[lo, "median_delay"])
    h["low_alarm_rate"] = float(roc.loc[lo, "alarms_per_year"])
    print(f"  there is no best threshold, only a curve: at {lo:.0f} the delay is "
          f"{h['low_delay']:.0f} sessions and the alarm rate {h['low_alarm_rate']:.1f}/yr; "
          f"at {THRESHOLD:.0f} they are {dd['median_delay']:.0f} and "
          f"{ar['alarms_per_year']:.1f}")

    print("\n=== 4. how the delay scales with the size of the change ===")
    scale = []
    for vs in (1.5, 2.0, 3.0, 5.0, 8.0):
        ws = st.synthetic_series(n=5000, break_points=[2500], vol_shift=vs)
        cs = st.variance_cusum(ws["returns"], DRIFT, THRESHOLD)
        d2 = st.detection_delay(cs["alarm"], ws["breaks"])
        csd = float(np.log(vs) / (np.pi / np.sqrt(2)))
        scale.append({"vol_shift": vs, "change_sd": csd,
                      "measured_delay": d2["median_delay"],
                      "theoretical": st.theoretical_delay(csd, THRESHOLD, DRIFT),
                      "detected": d2["n_detected"]})
        print(f"  {vs:.1f}x volatility ({csd:.2f} sd): measured "
              f"{d2['median_delay'] if np.isfinite(d2['median_delay']) else float('nan'):.0f}, "
              f"theory {st.theoretical_delay(csd, THRESHOLD, DRIFT):.0f}, "
              f"detected {d2['n_detected']}/1")
    h["scaling"] = scale

    print("\n=== 5. hindsight against real time ===")
    retro = st.binary_segmentation(w["returns"], max_breaks=5)
    errs = []
    for b in w["breaks"]:
        if retro:
            errs.append(min(abs(int(np.busday_count(b.date(), r.date()))) for r in retro))
    h["retro_error"] = float(np.median(errs)) if errs else np.nan
    h["retro_advantage"] = float(dd["median_delay"] - h["retro_error"]) \
        if np.isfinite(h["retro_error"]) else np.nan
    print(f"  retrospective segmentation placed the breaks within a median "
          f"{h['retro_error']:.0f} sessions")
    print(f"  the sequential detector took {dd['median_delay']:.0f} sessions to fire")
    print(f"  hindsight advantage: {h['retro_advantage']:.0f} sessions")
    print("  a retrospective method answers 'was there a break?'; a sequential one answers")
    print("  'is there a break NOW?'. Papers demonstrate the first and imply the second.")

    print("\n=== 6. on the real tape ===")
    real = {}
    for tk, s in assets.items():
        cc = st.variance_cusum(s, DRIFT, THRESHOLD)
        if cc.empty:
            continue
        arr = st.alarm_rate(cc["alarm"])
        fired = cc.index[cc["alarm"]]
        real[tk] = {"alarms_per_year": arr["alarms_per_year"],
                    "n_alarms": arr["n_alarms"],
                    "dates": [str(d.date()) for d in fired[:40]]}
        print(f"  {tk:9s} {arr['n_alarms']:3d} alarms ({arr['alarms_per_year']:.2f}/yr)")
    h["real_alarms"] = real
    eq_alarms = st.variance_cusum(lead, DRIFT, THRESHOLD)["alarm"]
    fired = eq_alarms.index[eq_alarms]
    print(f"\n  {data.EQUITY} alarms near the episodes everyone would name:")
    ep_rows = []
    for name, date in KNOWN_EPISODES.items():
        d0 = pd.Timestamp(date)
        after = fired[fired >= d0]
        lag = int(np.busday_count(d0.date(), after[0].date())) if len(after) else None
        ep_rows.append({"episode": name, "date": date,
                        "first_alarm": str(after[0].date()) if len(after) else "never",
                        "sessions_later": lag})
        print(f"    {name:22s} {date}  first alarm "
              + (f"{after[0].date()} ({lag} sessions later)" if len(after) else "never"))
    h["episodes"] = ep_rows
    lags = [r["sessions_later"] for r in ep_rows if r["sessions_later"] is not None]
    h["episode_median_lag"] = float(np.median(lags)) if lags else np.nan
    print(f"  median lag behind a named episode: {h['episode_median_lag']:.0f} sessions")
    print("  (these dates are a plausibility check, not ground truth — the 'date' of a regime "
          "change is not a well-defined thing, which is itself part of the problem)")

    print("\n=== 7. what the delay costs ===")
    cash = rets[data.CASH].reindex(lead.index).fillna(0.0)
    live = st.regime_switch_strategy(lead, eq_alarms, cash, RISK_OFF_DAYS, COST_BPS)
    hind = st.hindsight_strategy(lead, [pd.Timestamp(d) for d in KNOWN_EPISODES.values()],
                                 cash, RISK_OFF_DAYS, COST_BPS)
    h.update({"live_cagr": live["strategy"]["cagr"], "live_sharpe": live["strategy"]["sharpe"],
              "live_dd": live["strategy"]["max_dd"],
              "bh_cagr": live["buy_hold"]["cagr"], "bh_sharpe": live["buy_hold"]["sharpe"],
              "bh_dd": live["buy_hold"]["max_dd"],
              "hindsight_cagr": hind["strategy"]["cagr"],
              "hindsight_sharpe": hind["strategy"]["sharpe"],
              "hindsight_dd": hind["strategy"]["max_dd"],
              "time_in_market": live["time_in_market"]})
    print(f"  live detector:   CAGR {live['strategy']['cagr']:+.2%}, Sharpe "
          f"{live['strategy']['sharpe']:.2f}, maxDD {live['strategy']['max_dd']:.1%}, "
          f"invested {live['time_in_market']:.0%}")
    print(f"  buy and hold:    CAGR {live['buy_hold']['cagr']:+.2%}, Sharpe "
          f"{live['buy_hold']['sharpe']:.2f}, maxDD {live['buy_hold']['max_dd']:.1%}")
    print(f"  told the dates:  CAGR {hind['strategy']['cagr']:+.2%}, Sharpe "
          f"{hind['strategy']['sharpe']:.2f}, maxDD {hind['strategy']['max_dd']:.1%}")
    print(f"  -> the gap between live and hindsight is "
          f"{hind['strategy']['sharpe'] - live['strategy']['sharpe']:+.2f} of Sharpe, and it "
          f"IS the delay")

    print("\n=== 8. sweeping the rule ===")
    sweep = []
    for th in (3.0, 5.0, 8.0, 12.0):
        for days in (5, 21, 63):
            al = st.variance_cusum(lead, DRIFT, th)["alarm"]
            r2 = st.regime_switch_strategy(lead, al, cash, days, COST_BPS)
            sweep.append({"threshold": th, "risk_off_days": days,
                          "cagr": r2["strategy"]["cagr"],
                          "sharpe": r2["strategy"]["sharpe"],
                          "max_dd": r2["strategy"]["max_dd"],
                          "time_in_market": r2["time_in_market"]})
            print(f"  threshold {th:5.1f}, off {days:2d}d: CAGR "
                  f"{r2['strategy']['cagr']:+.2%}, Sharpe {r2['strategy']['sharpe']:.2f}, "
                  f"maxDD {r2['strategy']['max_dd']:.1%}, invested "
                  f"{r2['time_in_market']:.0%}")
    h["sweep"] = sweep
    best = max(sweep, key=lambda r: r["sharpe"])
    h["best_sweep_sharpe"] = float(best["sharpe"])
    print(f"  best of {len(sweep)} configurations: Sharpe {best['sharpe']:.2f} vs "
          f"buy-and-hold {h['bh_sharpe']:.2f} — and picking the best of "
          f"{len(sweep)} is itself a search (study 996)")

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    roc = "\n".join(
        f"| {r['threshold']:.0f} | {r['median_delay']:.0f} | {r['detection_rate']:.0%} | "
        f"{r['alarms_per_year']:.2f} | {int(r['false_alarms'])} |" for r in h["roc"])
    scale = "\n".join(
        f"| {r['vol_shift']:.1f}× | {r['change_sd']:.2f} | "
        f"{r['measured_delay']:.0f} | {r['theoretical']:.0f} | {int(r['detected'])} |"
        for r in h["scaling"])
    eps = "\n".join(
        f"| {r['episode']} | {r['date']} | {r['first_alarm']} | "
        f"{r['sessions_later'] if r['sessions_later'] is not None else '—'} |"
        for r in h["episodes"])
    real = "\n".join(f"| {tk} | {vv['n_alarms']} | {vv['alarms_per_year']:.2f} |"
                     for tk, vv in h["real_alarms"].items())
    sweep = "\n".join(
        f"| {r['threshold']:.0f} | {r['risk_off_days']} | {r['cagr']:+.2%} | "
        f"{r['sharpe']:.2f} | {r['max_dd']:.1%} | {r['time_in_market']:.0%} |"
        for r in h["sweep"])
    return f"""# Results — Study 999 (The Break) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). {h['n_assets']} assets over
{h['years']:.1f} years, plus synthetic series with change points at known dates. As-of
**{h['as_of']}**; fingerprint `{h['fingerprint']}`.*

## 1. Graded against breaks whose dates are known

Three change points planted in a 6,000-session series, a 3× volatility shift at each:

| | |
|---|--:|
| Breaks found | **{h['detection_rate']:.0%}** |
| Median delay | **{h['median_delay']:.0f} sessions** |
| Mean delay | {h['mean_delay']:.0f} sessions |
| Alarms per year | {h['alarms_per_year']:.2f} |
| False alarms | {h['false_alarms']} |

## 2. How much of that delay is unavoidable?

A 3× volatility shift is a change of about **{np.log(3.0) / (np.pi / np.sqrt(2)):.2f} standard
deviations** in log squared returns. Wald's identity puts the expected delay of *any* sequential
detector at roughly `threshold / (change − drift)`:

| | |
|---|--:|
| Theoretical floor | ~{h['theoretical_delay']:.0f} sessions |
| Measured | {h['median_delay']:.0f} sessions |
| **Ratio** | **{h['delay_vs_theory']:.2f}×** |

A detector running near 1× is not slow. The **data** is slow: at this signal-to-noise, that many
observations are required before the evidence exists. No algorithm improves on it, and that is
the single most useful thing to know about change-point detection.

## 3. The trade-off, which has no optimum

| Threshold | Median delay | Detection rate | Alarms/yr | False alarms |
|---|--:|--:|--:|--:|
{roc}

There is no best threshold, only a curve. Quoting one point on it without the rest is how these
methods get oversold.

## 4. Delay against the size of the change

| Volatility shift | Change (sd) | Measured delay | Theory | Detected |
|---|--:|--:|--:|--:|
{scale}

## 5. Hindsight against real time

Retrospective segmentation placed the same breaks within a median **{h['retro_error']:.0f}
sessions**; the sequential detector took **{h['median_delay']:.0f}**. The hindsight advantage is
**{h['retro_advantage']:.0f} sessions**.

That gap is the study's spine. A retrospective method answers *"was there a break?"* and a
sequential one answers *"is there a break now?"*. Papers routinely demonstrate the first and
imply the second.

## 6. On the real tape

| Asset | Alarms | Per year |
|---|--:|--:|
{real}

Against the episodes a macro desk would name:

| Episode | Date | First alarm | Sessions later |
|---|---|---|--:|
{eps}

Median lag behind a named episode: **{h['episode_median_lag']:.0f} sessions**. These dates are a
plausibility check, not ground truth — the "date" of a regime change is not a well-defined
thing, which is itself part of the problem.

## 7. What the delay costs

| | CAGR | Sharpe | Max DD |
|---|--:|--:|--:|
| Live detector | {h['live_cagr']:+.2%} | {h['live_sharpe']:.2f} | {h['live_dd']:.1%} |
| Buy and hold | {h['bh_cagr']:+.2%} | {h['bh_sharpe']:.2f} | {h['bh_dd']:.1%} |
| **Told the dates in advance** | **{h['hindsight_cagr']:+.2%}** | **{h['hindsight_sharpe']:.2f}** | {h['hindsight_dd']:.1%} |

The gap between the live and hindsight rows —
**{h['hindsight_sharpe'] - h['live_sharpe']:+.2f} of Sharpe** — is not a failure of the strategy.
It is the price of the delay, and section 2 says most of that delay cannot be removed.

| Threshold | Risk-off days | CAGR | Sharpe | Max DD | Invested |
|---|--:|--:|--:|--:|--:|
{sweep}

The best of those {len(h['sweep'])} configurations reaches a Sharpe of
{h['best_sweep_sharpe']:.2f} — but picking the best of {len(h['sweep'])} is itself a search, and
study **996** is about what that is worth.

## Caveats

- **The hindsight benchmark uses hand-picked episode dates.** They are the dates a macro desk
  would name, not an objective ground truth, and choosing them with hindsight flatters the
  benchmark. That biases *against* the live detector, which is the safe direction for this
  study's conclusion but should be stated.
- **One detector family.** CUSUM is optimal for a known shift size against a known baseline.
  Bayesian online change-point detection (Adams & MacKay 2007) and the PELT algorithm handle
  unknown change sizes better and would likely improve the detection rate at the same alarm
  rate — though not the information bound in section 2.
- **A "bad regime" may not be bad in-sample.** Building this study surfaced it the hard way: a
  planted drift of −0.05% a day inside a 3%-a-day volatility regime is invisible over 1,500
  sessions, because the sample mean's standard error is larger than the drift. The turbulent
  regime came out *positive*. That is not a quirk of the simulation — it is the same reason
  real-world claims about what returns do in a given regime are so much weaker than claims
  about what volatility does, and it is why every detector here targets variance.
- **Regimes are not step functions.** The synthetic world has instantaneous breaks; real regimes
  arrive gradually, which makes both the "true date" and the delay fuzzier than reported here.
- **Volatility, not returns.** The detector is tuned on variance because mean shifts are
  undetectable at daily frequency in any useful time — which is worth stating as a result in
  itself rather than a design choice.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[999-cusum-change-points](../README.md). Not investment advice.*
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
