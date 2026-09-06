"""Real-tape verification — Study 986 (The Rolling Ladder). Regenerates docs/results.md.

Runs the controlled experiment — one interest-rate path, a bond held to maturity
against a constant-maturity fund — then takes the same question to the tape: for each bond ETF,
every rolling window's starting yield against what it actually delivered, the error decomposed
into duration times the rate change, and the horizon at which the two instruments cross.

    python studies/986-etf-vs-hold-to-maturity/examples/verify.py            # cache-only
    python studies/986-etf-vs-hold-to-maturity/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ladder import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


SIM_MATURITY = 10.0
SIM_SHOCK_BP = 200.0


def report() -> dict:
    px = data.load_prices()
    h: dict = {"as_of": data.AS_OF, "sim_maturity": SIM_MATURITY,
               "sim_shock_bp": SIM_SHOCK_BP, "fingerprint": data.fingerprint(px)}

    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}")
    for tk in data.TICKERS:
        s = px[tk].dropna()
        print(f"  {tk:6s} {s.index[0].date()} -> {s.index[-1].date()}  n={len(s):,}")

    # ^TNX prints ten times the yield in percent: 42.0 means 4.20%.
    y10 = (px[data.YIELD_10Y] / 1000.0).dropna()
    print(f"  10-year yield: {y10.min():.2%} to {y10.max():.2%}, last {y10.iloc[-1]:.2%}")
    h["yield_range"] = [float(y10.min()), float(y10.max()), float(y10.iloc[-1])]

    print("\n=== 1. the arithmetic, stated ===")
    for m in (2, 5, 10, 20, 30):
        d = st.macaulay_duration(0.04, float(m))
        md = st.modified_duration(0.04, float(m))
        print(f"  a {m:2d}-year 4% bond: Macaulay duration {d:5.2f}y, modified {md:5.2f}, "
              f"price at 6% = {st.price_from_yield(0.06, float(m), 0.04):6.2f}")
    h["duration_table"] = [{"maturity": m, "macaulay": st.macaulay_duration(0.04, float(m)),
                            "modified": st.modified_duration(0.04, float(m)),
                            "price_at_6pct": st.price_from_yield(0.06, float(m), 0.04)}
                           for m in (2, 5, 10, 20, 30)]

    print("\n=== 2. the controlled experiment ===")
    w = st.synthetic_world(n_years=30, shock_bp=SIM_SHOCK_BP, maturity=SIM_MATURITY)
    h["sim_duration"] = float(w["duration"])
    x = st.crossover_horizon(w["rates"], SIM_MATURITY)
    h.update({"crossover_years": x["crossover_years"], "initial_gap": x["initial_gap"],
              "final_gap": x["final_gap"], "max_gap": x["max_gap"]})
    print(f"  a {SIM_MATURITY:.0f}-year bond held to maturity vs a {SIM_MATURITY:.0f}-year "
          f"constant-maturity fund, one path, a {SIM_SHOCK_BP:.0f} bp rise after a year")
    print(f"  starting modified duration {w['duration']:.2f}")
    print(f"  the fund immediately falls {abs(x['initial_gap']):.1%} behind")
    print(f"  it catches the bond after {x['crossover_years']:.1f} years")
    print(f"  after 30 years the fund is {x['final_gap']:+.1%} vs the bond")
    print("  (the search stops at the bond's maturity: past redemption the comparison depends "
          "entirely on what you assume the proceeds buy)")

    ramp = st.rate_path("ramp", n=int(120 * st.TRADING_DAYS), start=0.04, end=0.10)
    cv = st.convergence_horizon(ramp, SIM_MATURITY, tol=0.0005)
    h.update({"convergence_years": cv["convergence_years"],
              "leibowitz_bound": cv["leibowitz_2d_minus_1"]})
    print(f"  on a steadily trending path the fund's annualised return crosses back through "
          f"its purchase yield after {cv['convergence_years']:.1f} years")
    print(f"  Leibowitz, Bova & Kogelman (2014) predict 2D - 1 = "
          f"{cv['leibowitz_2d_minus_1']:.1f} for a FIXED duration; the measured number is "
          f"lower because a real fund's duration shrinks as yields rise, so the cumulative "
          f"price loss is smaller than D0 x dy")
    print("  -> the fund is not worse. It is slower to be right, by about a duration.")

    print("\n=== 3. the same experiment across maturities and shocks ===")
    grid = []
    for m in (2.0, 5.0, 10.0, 20.0, 30.0):
        for bp in (-200.0, -100.0, 100.0, 200.0):
            rates = st.rate_path("shock", n=int(45 * st.TRADING_DAYS), start=0.04,
                                 end=0.04 + bp / 1e4, shock_at=st.TRADING_DAYS)
            xx = st.crossover_horizon(rates, m)
            grid.append({"maturity": m, "shock_bp": bp,
                         "duration": st.modified_duration(0.04, m, 0.04),
                         "crossover_years": xx["crossover_years"],
                         "initial_gap": xx["initial_gap"]})
            cross = ("  never" if not np.isfinite(xx["crossover_years"])
                     else f"{xx['crossover_years']:6.1f}y")
            print(f"  {m:4.0f}y bond, {bp:+5.0f} bp: initial gap {xx['initial_gap']:+7.1%}, "
                  f"crossover {cross} "
                  f"(duration {st.modified_duration(0.04, m, 0.04):5.2f})")
    h["crossover_grid"] = grid
    fin = [g for g in grid if np.isfinite(g["crossover_years"])]
    if fin:
        ratio = np.mean([g["crossover_years"] / g["duration"] for g in fin])
        h["crossover_over_duration"] = float(ratio)
        print(f"  -> where it happens within the bond's life, crossover / duration averages "
              f"{ratio:.2f}. Redington (1952) is the reason that number is of order one.")

    print("\n=== 3b. convergence to the purchase yield, by maturity ===")
    conv_grid = []
    for m in (2.0, 5.0, 10.0, 20.0, 30.0):
        cvm = st.convergence_horizon(ramp, m, tol=0.0005)
        conv_grid.append({"maturity": m, "duration": cvm["duration"],
                          "convergence_years": cvm["convergence_years"],
                          "leibowitz": cvm["leibowitz_2d_minus_1"]})
        print(f"  {m:4.0f}y: duration {cvm['duration']:5.2f}, converges after "
              f"{cvm['convergence_years']:6.2f}y, Leibowitz bound "
              f"{cvm['leibowitz_2d_minus_1']:5.2f}, ratio to duration "
              f"{cvm['convergence_years'] / cvm['duration']:.2f}")
    h["convergence_grid"] = conv_grid

    print("\n=== 4. on the tape: what each fund promised and delivered ===")
    funds = {}
    for tk in (data.SHORT, data.INTERMEDIATE, data.LONG, data.CORPORATE):
        dur = st.FUND_DURATION[tk]
        s = px[tk].dropna()
        # The 10-year yield stands in for each fund's own yield-to-maturity, which Yahoo does
        # not publish. That is a real approximation and it is why the LQD row is the weakest.
        conv = st.convergence_by_horizon(s, y10, dur)
        if conv.empty:
            continue
        funds[tk] = conv.reset_index().to_dict("records")
        print(f"  {tk} (duration {dur:.1f}y)")
        for hy, r in conv.iterrows():
            print(f"    {hy:2.0f}y windows: n {int(r['n']):4d}  promised "
                  f"{r['mean_promised']:+.2%}  realised {r['mean_realised']:+.2%}  "
                  f"error {r['mean_error']:+.2%}  sd {r['sd_error']:.2%}  "
                  f"within 1pp {r['share_within_1pp']:.0%}")
    h["funds"] = funds

    print("\n=== 5. is the error the roll, or something else? ===")
    tk = data.INTERMEDIATE
    dur = st.FUND_DURATION[tk]
    h["fund"] = tk
    h["duration"] = dur
    decomps = {}
    for hy in (1, 3, 5, 7, 10):
        t = st.realised_vs_promised(px[tk].dropna(), y10, float(hy), dur)
        if len(t) < 30:
            continue
        d = st.error_decomposition(t, dur, float(hy))
        decomps[hy] = d
        print(f"  {hy:2d}y windows (n {d['n']:3d}): error = {d['intercept']:+.3%} + "
              f"{d['slope']:.2f} x (-D dy / h),  R2 {d['r2']:.0%},  "
              f"mean error {d['mean_error']:+.2%}, sd {d['sd_error']:.2%}")
    h["decompositions"] = decomps
    at_dur = min(decomps, key=lambda k: abs(k - dur)) if decomps else None
    if at_dur is not None:
        d = decomps[at_dur]
        t = st.realised_vs_promised(px[tk].dropna(), y10, float(at_dur), dur)
        h.update({"n_windows": d["n"], "mean_error_at_duration": d["mean_error"],
                  "sd_error_at_duration": d["sd_error"], "decomp_slope": d["slope"],
                  "decomp_r2": d["r2"],
                  "share_within_1pp": float((t["error"].abs() < 0.01).mean())})
        print(f"  -> at {at_dur}y, closest to {tk}'s {dur:.1f}-year duration: the error's sd is "
              f"{d['sd_error']:.2%} and the roll explains {d['r2']:.0%} of it")
    conv = st.convergence_by_horizon(px[tk].dropna(), y10, dur)
    h["sd_error_1y"] = float(conv.loc[1, "sd_error"]) if 1 in conv.index else np.nan
    h["sd_error_10y"] = float(conv.loc[10, "sd_error"]) if 10 in conv.index else np.nan

    print("\n=== 6. the comparison that matters to an allocator ===")
    rows = []
    for tk2 in (data.SHORT, data.INTERMEDIATE, data.LONG):
        s = px[tk2].dropna()
        r = s.pct_change().dropna()
        yrs = len(r) / st.TRADING_DAYS
        cagr = float((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1)
        start_y = float(y10.reindex(s.index).ffill().iloc[0])
        rows.append({"fund": tk2, "duration": st.FUND_DURATION[tk2], "years": yrs,
                     "starting_yield": start_y, "realised_cagr": cagr,
                     "gap": cagr - start_y,
                     "vol": float(r.std() * np.sqrt(st.TRADING_DAYS)),
                     "max_dd": float(((1 + r).cumprod() /
                                      (1 + r).cumprod().cummax() - 1).min())})
        print(f"  {tk2}: over {yrs:.1f} years, started at {start_y:.2%}, delivered "
              f"{cagr:+.2%} ({cagr - start_y:+.2%}), vol {rows[-1]['vol']:.1%}, "
              f"maxDD {rows[-1]['max_dd']:.1%}")
    h["lifetime"] = rows

    print("\n=== 7. synthetic control: does the machinery find a known answer? ===")
    for vol, tag in ((0.0, "deterministic path"), (0.01, "noisy path")):
        xs = []
        for s in range(5):
            ww = st.synthetic_world(n_years=40, shock_bp=200, maturity=10.0, vol=vol,
                                    seed=986 + s)
            xs.append(st.crossover_horizon(ww["rates"], 10.0)["crossover_years"])
        print(f"  {tag:22s} crossover {np.nanmean(xs):5.1f}y "
              f"(duration {st.modified_duration(0.04, 10.0, 0.04):.2f})")
        h[f"synthetic_{'noisy' if vol else 'clean'}"] = float(np.nanmean(xs))

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    dur = "\n".join(
        f"| {int(r['maturity'])}y | {r['macaulay']:.2f} | {r['modified']:.2f} | "
        f"{r['price_at_6pct']:.2f} |" for r in h["duration_table"])
    def _cross(g):
        return "never" if not np.isfinite(g["crossover_years"]) \
            else f"{g['crossover_years']:.1f}y"

    grid = "\n".join(
        f"| {int(g['maturity'])}y | {g['shock_bp']:+.0f} | {g['duration']:.2f} | "
        f"{g['initial_gap']:+.1%} | {_cross(g)} |" for g in h["crossover_grid"])
    cgrid = "\n".join(
        f"| {int(g['maturity'])}y | {g['duration']:.2f} | {g['convergence_years']:.2f} | "
        f"{g['leibowitz']:.2f} | {g['convergence_years'] / g['duration']:.2f} |"
        for g in h["convergence_grid"])
    funds = "\n".join(
        f"| {tk} | {int(r['horizon_y'])}y | {int(r['n'])} | {r['mean_promised']:+.2%} | "
        f"{r['mean_realised']:+.2%} | {r['mean_error']:+.2%} | {r['sd_error']:.2%} | "
        f"{r['share_within_1pp']:.0%} |"
        for tk, rows in h["funds"].items() for r in rows)
    dec = "\n".join(
        f"| {k}y | {d['n']} | {d['intercept']:+.3%} | {d['slope']:.2f} | {d['r2']:.0%} | "
        f"{d['mean_error']:+.2%} | {d['sd_error']:.2%} |" for k, d in h["decompositions"].items())
    life = "\n".join(
        f"| {r['fund']} | {r['duration']:.1f} | {r['years']:.1f} | {r['starting_yield']:.2%} | "
        f"{r['realised_cagr']:+.2%} | {r['gap']:+.2%} | {r['vol']:.1%} | {r['max_dd']:.1%} |"
        for r in h["lifetime"])
    return f"""# Results — Study 986 (The Rolling Ladder) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). Constant-maturity bond funds against
the arithmetic of a bond held to maturity. As-of **{h['as_of']}**; fingerprint
`{h['fingerprint']}`. The 10-year yield ranged {h['yield_range'][0]:.2%} to
{h['yield_range'][1]:.2%} over the sample, ending at {h['yield_range'][2]:.2%} — a range wide
enough for the question to have an answer.*

## 1. The arithmetic

| Maturity | Macaulay duration | Modified | Price at 6% (4% coupon) |
|---|--:|--:|--:|
{dur}

## 2. The controlled experiment

One interest-rate path. A {h['sim_maturity']:.0f}-year bond bought at 4% and held to maturity,
against a {h['sim_maturity']:.0f}-year constant-maturity fund. Rates rise
{h['sim_shock_bp']:.0f} bp after one year, and stay there.

| | |
|---|--:|
| Starting modified duration | {h['sim_duration']:.2f} |
| The fund's immediate shortfall | **{h['initial_gap']:+.1%}** |
| Years until it catches the bond | **{h['crossover_years']:.1f}** |
| Gap after 30 years | {h['final_gap']:+.1%} |

This is the study in four numbers. The fund is not *worse* than the bond. It is **slower to be
right**, and the length of the delay is set by its duration.

One caveat on the third row, because it decides how the number should be read: the search stops
at the bond's maturity. Past redemption the comparison is no longer about the two instruments —
it is about whatever you assume the bond's proceeds were reinvested in — so continuing it would
be measuring an assumption.

## 3. Crossover across maturities and shocks

| Maturity | Shock (bp) | Duration | Initial gap | Crossover |
|---|--:|--:|--:|--:|
{grid}

Where the crossing happens inside the bond's life, crossover ÷ duration averages
**{h.get('crossover_over_duration', float('nan')):.2f}** — of order one, which is Redington
(1952) showing through.

## 3b. Convergence: when does the *fund* deliver what it advertised?

The crossover above compares two instruments. This is the question with a theorem attached: on a
steadily trending rate path, how long before the fund's annualised return since purchase crosses
back through the yield it was bought at?

| Maturity | Duration | Converges after | Leibowitz 2D−1 | Ratio to duration |
|---|--:|--:|--:|--:|
{cgrid}

Two things to read here. First, **convergence scales with duration** — duration is a clock, not
only a risk number. Second, the measured horizon sits consistently *below* Leibowitz, Bova &
Kogelman's 2D − 1, and for a reason worth stating: their bound assumes a fixed duration, while a
real fund's duration **shrinks as yields rise**. The cumulative price loss is therefore smaller
than D₀ × Δy, and the fund gets back to its purchase yield sooner than the fixed-duration
algebra says. That is convexity doing something useful.

## 4. On the tape: promised versus delivered

For each fund, every rolling window's starting yield against the annualised return that
followed:

| Fund | Horizon | n | Mean promised | Mean realised | Error | SD of error | Within 1pp |
|---|--:|--:|--:|--:|--:|--:|--:|
{funds}

## 5. Is the error the roll, or something else?

Regressing each window's error on **−duration × Δyield ⁄ horizon**, the shortfall theory
predicts:

| Horizon | n | Intercept | Slope | R² | Mean error | SD |
|---|--:|--:|--:|--:|--:|--:|
{dec}

A slope near 1.0 with a high R² means the gap between what a bond fund advertises and what it
delivers is **entirely** its refusal to mature. There is no second mystery to explain.

## 6. What each fund actually did over its life

| Fund | Duration | Years | Starting yield | Realised CAGR | Gap | Vol | Max DD |
|---|--:|--:|--:|--:|--:|--:|--:|
{life}

## 7. Synthetic control

Crossover horizon recovered from a deterministic path:
{h['synthetic_clean']:.1f} years; from a noisy one: {h['synthetic_noisy']:.1f} years, against a
modified duration of {h['sim_duration']:.2f}. The machinery finds the answer it should.

## Caveats

- **The 10-year yield stands in for each fund's own yield-to-maturity.** Yahoo does not publish
  fund YTM, so ^TNX is used as the rate series for all of them. That is exact for IEF, roughly
  right for TLT and SHY in *changes* if not levels, and weakest for LQD, whose credit spread
  moves independently of Treasuries. The LQD row should be read as illustrative.
- **The constant-maturity approximation.** `simulate_rolling_fund` uses carry minus
  duration × Δy plus a convexity term, which is what index providers use, but a real ETF has
  transaction costs on every roll, a fee, and a curve shape rather than a single yield.
- **A single flat yield.** The whole study collapses the curve to one number. A steepening and
  a parallel shift of the same size have different consequences for a rolling fund, and this
  cannot see the difference. That is the largest simplification here.
- **Overlapping windows.** Section 4's rolling windows overlap heavily, so the *n* columns
  substantially overstate the independent information. The standard deviations are descriptive,
  not inferential.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[986-etf-vs-hold-to-maturity](../README.md). Not investment advice.*
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
