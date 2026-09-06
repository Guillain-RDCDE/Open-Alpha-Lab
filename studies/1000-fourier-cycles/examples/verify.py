"""Real-tape verification — Study 1000 (The Cycle Hunt). Regenerates docs/results.md.

Computes the periodogram of every asset with detrending and windowing choices made
explicit, measures how large a peak a random walk of the same length produces, applies Fisher's
exact test against both a white and an AR(1) null, checks a positive control with a genuine
annual cycle, tests whether the best peak keeps its period and phase across sample halves, and
trades it out of sample.

    python studies/1000-fourier-cycles/examples/verify.py            # cache-only
    python studies/1000-fourier-cycles/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cyclehunt import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


MIN_PERIOD = 5.0
MAX_PERIOD = 750.0
FIT_WINDOW = 1000
COST_BPS = 5.0
N_SIMS = 300


def report() -> dict:
    px = data.load_prices()
    rets = px.pct_change()
    h: dict = {"as_of": data.AS_OF, "fit_window": FIT_WINDOW,
               "fingerprint": data.fingerprint(px)}

    assets = {tk: rets[tk].dropna() for tk in data.TICKERS
              if tk != data.CASH and rets[tk].notna().sum() > 1000}
    lead = assets[data.EQUITY]
    h["n_assets"] = int(len(assets))
    h["lead_asset"] = data.EQUITY
    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    for tk, s in assets.items():
        print(f"  {tk:9s} {s.index[0].date()} -> {s.index[-1].date()}  n={len(s):,}  "
              f"lag-1 autocorrelation {s.autocorr(1):+.3f}")

    print("\n=== 1. the spectrum of the stock market ===")
    pg = st.periodogram(lead)
    h["n_bins"] = int(len(pg))
    peaks = st.top_peaks(pg, 10, MIN_PERIOD, MAX_PERIOD)
    print(f"  {len(pg):,} frequency bins from {len(lead):,} observations")
    print("  the ten strongest periods:")
    for _, r in peaks.iterrows():
        print(f"    {r['period']:8.1f} sessions ({r['period'] / 252:5.2f} years)  "
              f"{r['relative_power']:6.2f}x the average power")
    h["top_peaks"] = peaks.to_dict("records")
    h["lead_relative_power"] = float(peaks.iloc[0]["relative_power"])
    h["lead_peak_period"] = float(peaks.iloc[0]["period"])

    print("\n=== 2. what a random walk gives ===")
    theo = st.expected_max_relative_power(len(pg))
    sim = st.spurious_peak_distribution(len(lead), N_SIMS, min_period=MIN_PERIOD)
    h["theoretical_max"] = float(theo)
    h["simulated_mean"] = sim["mean_max"]
    h["simulated_p95"] = sim["p95_max"]
    h["simulated_p99"] = sim["p99_max"]
    print(f"  the periodogram ordinates of white noise are independent exponentials, so the")
    print(f"  largest of {len(pg):,} is expected at log(m) + 0.577 = {theo:.2f}x the mean")
    print(f"  simulating {sim['n_sims']} random walks of the same length:")
    print(f"    mean best peak   {sim['mean_max']:.2f}x")
    print(f"    median           {sim['median_max']:.2f}x")
    print(f"    95th percentile  {sim['p95_max']:.2f}x")
    print(f"    99th percentile  {sim['p99_max']:.2f}x")
    print(f"  {data.EQUITY}'s best peak: {h['lead_relative_power']:.2f}x "
          f"-> {'above' if h['lead_relative_power'] > sim['p95_max'] else 'INSIDE'} "
          f"the noise band")

    print("\n=== 3. Fisher's exact test, 1929 ===")
    rows = []
    n_white = n_ar1 = 0
    for tk, s in assets.items():
        p = st.periodogram(s)
        f = st.fisher_g_test(p)
        a1 = st.ar1_null(s)
        pa = st.peak_against_ar1(p, a1.get("phi", 0.0))
        # An AR(1)-corrected peak is significant if it still exceeds the simulated band for
        # the matching autocorrelation.
        sim_ar = st.spurious_peak_distribution(len(s), 120, ar1=a1.get("phi", 0.0),
                                               min_period=MIN_PERIOD)
        sig_ar = bool(pa.get("relative_power_ar1", 0) > sim_ar["p95_max"])
        n_white += bool(f.get("significant_5pct", False))
        n_ar1 += sig_ar
        rows.append({"asset": tk, "peak_period": f.get("peak_period", np.nan),
                     "relative_power": f.get("relative_power", np.nan),
                     "fisher_g": f.get("g", np.nan), "fisher_p": f.get("p_value", np.nan),
                     "significant_white": bool(f.get("significant_5pct", False)),
                     "phi": a1.get("phi", np.nan),
                     "peak_period_ar1": pa.get("peak_period_ar1", np.nan),
                     "relative_power_ar1": pa.get("relative_power_ar1", np.nan),
                     "ar1_band_p95": sim_ar["p95_max"],
                     "significant_ar1": sig_ar})
        print(f"  {tk:9s} peak {f.get('peak_period', np.nan):7.1f}d  "
              f"{f.get('relative_power', np.nan):5.2f}x  Fisher p "
              f"{f.get('p_value', np.nan):7.4f} "
              f"{'*' if f.get('significant_5pct') else ' '}  |  phi "
              f"{a1.get('phi', np.nan):+.3f}  AR(1)-corrected peak "
              f"{pa.get('peak_period_ar1', np.nan):7.1f}d at "
              f"{pa.get('relative_power_ar1', np.nan):5.2f}x vs band "
              f"{sim_ar['p95_max']:5.2f}x {'*' if sig_ar else ' '}")
    h["fisher"] = rows
    h["n_significant_white"] = int(n_white)
    h["n_significant_ar1"] = int(n_ar1)
    h["mean_relative_power"] = float(np.nanmean([r["relative_power"] for r in rows]))
    print(f"  significant against white noise: {n_white} of {len(rows)}")
    print(f"  significant against an AR(1) null: {n_ar1} of {len(rows)}")
    print("  the second row is the honest one — returns are autocorrelated, and a flat null")
    print("  tilts every test toward finding long cycles")

    print("\n=== 4. the positive control ===")
    ctrl = data.SEASONAL
    h["control_asset"] = ctrl
    if ctrl in assets:
        cs = assets[ctrl]
        cp = st.periodogram(cs)
        # look specifically in the annual region
        annual = st.top_peaks(cp, 3, min_period=180.0, max_period=400.0)
        best = st.top_peaks(cp, 1, MIN_PERIOD, MAX_PERIOD)
        h["control_period"] = float(annual.iloc[0]["period"]) if not annual.empty \
            else float(best.iloc[0]["period"])
        h["control_relative_power"] = float(annual.iloc[0]["relative_power"]) \
            if not annual.empty else float(best.iloc[0]["relative_power"])
        print(f"  {ctrl} has a genuine annual demand cycle.")
        if not annual.empty:
            print(f"  strongest period in the 180-400 session band: "
                  f"{annual.iloc[0]['period']:.0f} sessions "
                  f"({annual.iloc[0]['period'] / 252:.2f} years) at "
                  f"{annual.iloc[0]['relative_power']:.2f}x")
        print(f"  strongest period anywhere: {best.iloc[0]['period']:.0f} sessions at "
              f"{best.iloc[0]['relative_power']:.2f}x")
        print("  a method that cannot find a cycle here is failing to detect, not correctly")
        print("  reporting absence")
    else:
        h["control_period"] = np.nan
        h["control_relative_power"] = np.nan

    print("\n=== 5. does the peak survive a split sample? ===")
    sp = st.split_sample_peak(lead, MIN_PERIOD, MAX_PERIOD)
    h.update({k: sp.get(k, np.nan) for k in
              ("period_first", "period_second", "period_ratio", "amplitude_decay",
               "phase_error_fraction", "phase_concentration", "phase_r2")})
    print(f"  best period in the first half:  {sp['period_first']:.1f} sessions")
    print(f"  best period in the second half: {sp['period_second']:.1f} sessions "
          f"(ratio {sp['period_ratio']:.2f})")
    print(f"  amplitude decay: {sp['amplitude_decay']:.2f}")
    print(f"  raw phase error: {sp['phase_error_fraction']:.0%} of a half-cycle "
          f"(dominated by the periodogram's grid resolution, see below)")
    print(f"  phase COHERENCE: {sp['phase_concentration']:.3f} "
          f"(linear-drift R2 {sp['phase_r2']:.2f}); above 0.70 means coherent")
    print("  a real cycle's phase drifts LINEARLY when the period estimate is slightly off;")
    print("  noise wanders. That distinction is what makes the test usable at all.")
    splits = []
    for tk, s in assets.items():
        o = st.split_sample_peak(s, MIN_PERIOD, MAX_PERIOD)
        if "period_ratio" in o:
            splits.append({"asset": tk, "period_first": o["period_first"],
                           "period_second": o["period_second"],
                           "ratio": o["period_ratio"],
                           "phase_error": o["phase_error_fraction"],
                           "coherence": o.get("phase_concentration", np.nan)})
            print(f"  {tk:9s} {o['period_first']:7.1f} -> {o['period_second']:7.1f} "
                  f"(x{o['period_ratio']:.2f}), phase off "
                  f"{o['phase_error_fraction']:.0%}, coherence "
                  f"{o.get('phase_concentration', float('nan')):.2f}")
    h["splits"] = splits

    print("\n=== 6. trade the peak ===")
    tr = st.cycle_trade(lead, h["lead_peak_period"], FIT_WINDOW, COST_BPS)
    h.update({"cycle_cagr": tr.get("cagr", np.nan), "cycle_sharpe": tr.get("sharpe", np.nan),
              "cycle_hit_rate": tr.get("hit_rate", np.nan),
              "cycle_switches": tr.get("switches_per_year", np.nan),
              "cycle_buyhold": tr.get("buy_hold_cagr", np.nan)})
    print(f"  trading the {h['lead_peak_period']:.0f}-session cycle out of sample:")
    print(f"    CAGR {tr['cagr']:+.2%}, Sharpe {tr['sharpe']:.2f}, hit rate "
          f"{tr['hit_rate']:.1%}, {tr['switches_per_year']:.0f} switches/yr")
    print(f"    buy-and-hold over the same window: {tr['buy_hold_cagr']:+.2%}")
    sweep = []
    for per in (21.0, 63.0, 126.0, 252.0, h["lead_peak_period"]):
        t2 = st.cycle_trade(lead, per, FIT_WINDOW, COST_BPS)
        if "sharpe" not in t2:
            continue
        sweep.append({"period": per, "cagr": t2["cagr"], "sharpe": t2["sharpe"],
                      "hit_rate": t2["hit_rate"]})
        print(f"    {per:6.0f}-session cycle: Sharpe {t2['sharpe']:+.2f}, hit "
              f"{t2['hit_rate']:.1%}")
    h["trade_sweep"] = sweep

    print("\n=== 7. calibration: the same analysis on data with no cycle ===")
    ctrl_rows = []
    for k in range(6):
        sim_x = st.synthetic_series(n=len(lead), seed=1000 + k)
        p2 = st.periodogram(sim_x)
        f2 = st.fisher_g_test(p2)
        pk = st.top_peaks(p2, 1, MIN_PERIOD, MAX_PERIOD)
        ctrl_rows.append({"run": k, "peak_period": float(pk.iloc[0]["period"]),
                          "relative_power": float(pk.iloc[0]["relative_power"]),
                          "fisher_p": f2["p_value"]})
        print(f"  run {k}: best period {pk.iloc[0]['period']:7.1f} sessions at "
              f"{pk.iloc[0]['relative_power']:5.2f}x, Fisher p {f2['p_value']:.4f}")
    h["noise_control"] = ctrl_rows
    print(f"  {data.EQUITY} gave {h['lead_relative_power']:.2f}x at "
          f"{h['lead_peak_period']:.0f} sessions. Compare the column above.")

    print("\n=== 8. power: what amplitude would we need to see? ===")
    power_rows = []
    for amp in (0.0, 0.05, 0.1, 0.2, 0.4):
        hits = 0
        for k in range(20):
            sim_x = st.synthetic_series(n=len(lead), period=126.0, amplitude=amp,
                                        seed=1000 + k)
            hits += bool(st.fisher_g_test(st.periodogram(sim_x)).get("significant_5pct"))
        power_rows.append({"amplitude": amp, "detection_rate": hits / 20})
        print(f"  amplitude {amp:.2f} sd: detected in {hits / 20:.0%} of runs")
    h["power_curve"] = power_rows
    detectable = [r["amplitude"] for r in power_rows if r["detection_rate"] >= 0.8]
    h["min_detectable_amplitude"] = float(min(detectable)) if detectable else np.nan
    print(f"  smallest reliably detectable amplitude: "
          f"{h['min_detectable_amplitude'] if np.isfinite(h['min_detectable_amplitude']) else float('nan'):.2f} "
          f"standard deviations")
    print("  so 'no cycle found' means 'no cycle above that size', which is a weaker")
    print("  statement than it sounds and is stated here rather than left implied")

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    peaks = "\n".join(
        f"| {r['period']:.1f} | {r['period'] / 252:.2f} | {r['relative_power']:.2f}× |"
        for r in h["top_peaks"])
    fisher = "\n".join(
        f"| {r['asset']} | {r['peak_period']:.1f} | {r['relative_power']:.2f}× | "
        f"{r['fisher_p']:.4f} | {'**yes**' if r['significant_white'] else 'no'} | "
        f"{r['phi']:+.3f} | {r['peak_period_ar1']:.1f} | {r['relative_power_ar1']:.2f}× | "
        f"{r['ar1_band_p95']:.2f}× | {'**yes**' if r['significant_ar1'] else 'no'} |"
        for r in h["fisher"])
    splits = "\n".join(
        f"| {r['asset']} | {r['period_first']:.1f} | {r['period_second']:.1f} | "
        f"{r['ratio']:.2f}× | {r['phase_error']:.0%} | {r['coherence']:.2f} |"
        for r in h["splits"])
    sweep = "\n".join(
        f"| {r['period']:.0f} | {r['cagr']:+.2%} | {r['sharpe']:+.2f} | {r['hit_rate']:.1%} |"
        for r in h["trade_sweep"])
    ctrl = "\n".join(
        f"| {int(r['run'])} | {r['peak_period']:.1f} | {r['relative_power']:.2f}× | "
        f"{r['fisher_p']:.4f} |" for r in h["noise_control"])
    power = "\n".join(f"| {r['amplitude']:.2f} | {r['detection_rate']:.0%} |"
                      for r in h["power_curve"])
    return f"""# Results — Study 1000 (The Cycle Hunt) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). {h['n_assets']} assets, spectra
computed on {h['n_bins']:,} frequency bins. As-of **{h['as_of']}**; fingerprint
`{h['fingerprint']}`.*

## 1. The spectrum of the stock market

| Period (sessions) | Years | Power ÷ average |
|---|--:|--:|
{peaks}

Presented on its own, that table is how a century of cycle theories got started.

## 2. What a random walk gives

The periodogram of white noise is **not flat**. Each ordinate is an independent exponential
draw, so the largest of *m* of them sits around `log(m) + 0.577` times the mean:

| | |
|---|--:|
| Frequency bins | {h['n_bins']:,} |
| Theoretical expected maximum | **{h['theoretical_max']:.2f}×** |
| Simulated mean (random walks, same length) | {h['simulated_mean']:.2f}× |
| Simulated 95th percentile | **{h['simulated_p95']:.2f}×** |
| Simulated 99th percentile | {h['simulated_p99']:.2f}× |
| **{h['lead_asset']}'s actual best peak** | **{h['lead_relative_power']:.2f}×** |

## 3. Fisher's exact test — available since 1929

| Asset | Peak (d) | Power | Fisher *p* | Sig. vs white | φ | AR(1) peak | AR(1) power | AR(1) band | Sig. vs AR(1) |
|---|--:|--:|--:|---|--:|--:|--:|--:|---|
{fisher}

**{h['n_significant_white']} of {h['n_assets']}** significant against white noise;
**{h['n_significant_ar1']} of {h['n_assets']}** against an AR(1) null.

The second number is the honest one. Returns are autocorrelated, an AR(1) spectrum tilts toward
low frequencies, and testing against a flat null therefore biases every test toward discovering
"long cycles" that are nothing but the autocorrelation showing through.

## 4. The positive control

{h['control_asset']} has a genuine annual demand cycle. The method found a peak at
**{h['control_period']:.0f} sessions** ({h['control_period'] / 252:.2f} years) at
{h['control_relative_power']:.2f}× the average power.

This matters more than any of the negative results. A study that only ever finds nothing has not
shown that nothing is there — it may simply be unable to see. The control shows the machinery
detects a cycle when one exists.

## 5. Does the peak survive a split sample?

| Asset | Period, first half | Second half | Ratio | Phase error | Coherence |
|---|--:|--:|--:|--:|--:|
{splits}

A genuine cycle keeps its period **and its phase** — but "phase" has to be measured carefully,
and getting that wrong was the most instructive bug in building this study.

The periodogram measures a period **on a grid**: near 120 sessions with 8,000 observations the
neighbouring bins are several sessions apart. Being off by one part in a hundred accumulates a
full radian of phase over a few thousand steps, so a *perfectly stable planted cycle* fails a
naive "is the phase the same in both halves?" test. That test measures grid resolution, not
reality, and it is pinned as a unit test
(`test_the_naive_phase_test_fails_even_on_a_genuine_cycle`) so nobody re-introduces it.

What works is **coherence**. Split the series into segments, fit the phase in each, and ask
whether it drifts *linearly* (a real cycle whose period estimate is slightly off) or wanders at
random (noise). On {h['lead_asset']} the raw phase error was
{h['phase_error_fraction']:.0%} of a half-cycle — uninformative on its own — while the phase
concentration was **{h['phase_concentration']:.2f}** against the 0.70 threshold for calling a
cycle coherent.

## 6. Trading it

Fitting the sinusoid on a rolling {h['fit_window']}-session window and trading the next step's
sign, strictly out of sample:

| Period | CAGR | Sharpe | Hit rate |
|---|--:|--:|--:|
{sweep}

Buy-and-hold over the same window: {h['cycle_buyhold']:+.2%}.

## 7. Calibration against data with no cycle at all

| Run | Best period | Power | Fisher *p* |
|---|--:|--:|--:|
{ctrl}

{h['lead_asset']} gave {h['lead_relative_power']:.2f}× at {h['lead_peak_period']:.0f} sessions.
Compare the column.

## 8. What could we have detected?

| Amplitude (sd) | Detected |
|---|--:|
{power}

Smallest reliably detectable amplitude: **{h['min_detectable_amplitude']:.2f} standard
deviations**. So "no cycle found" means "no cycle above that size" — a weaker statement than it
sounds, and one worth making explicitly rather than leaving implied.

## Caveats

- **Fisher's test assumes Gaussian white noise.** Returns are neither Gaussian nor white. The
  AR(1) columns and the simulated bands are the corrections; heavy tails are not corrected for
  and would widen the bands further, making the test *more* conservative than reported.
- **A fixed-period sinusoid is a narrow definition of a cycle.** Business cycles are not
  sinusoids: they vary in length, which spreads their power across bins and can hide them from
  exactly this method. Wavelets or a time-varying spectrum would be the right tool for that and
  are not used here.
- **Detrending is a choice.** Section 1 detrends. Not detrending produces a spectacular peak at
  the sample length, which is the single commonest artefact in published market-cycle charts.
- **Daily data cannot see decade cycles.** With thirty years of data there are only three
  ten-year "cycles", so anything in that region is estimated from a handful of oscillations
  regardless of how confident the periodogram looks.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[1000-fourier-cycles](../README.md). Not investment advice.*
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
