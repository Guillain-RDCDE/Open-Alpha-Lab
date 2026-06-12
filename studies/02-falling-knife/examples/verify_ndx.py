"""Live study on the real Nasdaq-100 — crosses ^NDX (since 1985) and QQQ (since 1999).

Requires a network connection (Yahoo! Finance) on first run; afterwards the data
is cached as parquet under ``_cache/`` and reruns are offline.

    python examples/verify_ndx.py

What it answers, end to end:
  1. How does NDX actually behave around a -3% day? (event study, both faces)
  2. Does buying the dip beat buying a random day? (the decisive test)
  3. Does any (trigger x exit) survive costs AND deflation? (family scan)
  4. Is it one lucky regime or a stable effect? (regime split)
The console summary closes with a plain-language verdict, and the headline tables
are written to ``docs/results.md`` with the as-of date and the data fingerprints,
so every number quoted in the README traces to an exact, reproducible run.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

_STUDY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY_DIR, "..", "..")))  # repo root, for quantlab

from falling_knife import data, triggers, exits, eventstudy, benchmark, backtest, robustness, plots
from quantlab.repro import DEFAULT_AS_OF, as_of, data_stamp

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(OUT_DIR), "docs", "results.md")
pd.set_option("display.width", 140)


def study(label, ohlc):
    print(f"\n{'#'*72}\n#  {label}   ({ohlc.index[0].date()} -> {ohlc.index[-1].date()}, "
          f"{len(ohlc)} days)\n{'#'*72}")
    stamp = data_stamp(label.split()[0], ohlc, cols=["Close"])
    print(stamp)
    ret = data.daily_returns(ohlc)

    # Event study on the classic close-to-close -3% trigger.
    events = triggers.first_crossings(triggers.close_to_close(ret), cooldown=20)
    es = eventstudy.event_study(ohlc, events, horizon=20, pre=5)
    print(f"\n[Event study | T1 close-to-close -3%]  {es['n_events']} events")
    print(es["summary"].loc[[1, 3, 5, 10, 20], ["mean", "median", "pct_positive", "tstat"]]
          .to_string(float_format=lambda v: f"{v:.4f}"))

    # Decisive test: conditional vs random day.
    bench = benchmark.conditional_vs_unconditional(ohlc, events)
    print("\n[Conditional vs random day]  (p_greater < 0.05 => genuinely beats random)")
    print(bench[["n_events", "mean_cond", "mean_uncond", "excess", "p_greater"]]
          .to_string(float_format=lambda v: f"{v:.4f}"))

    # Block bootstrap on the 5-day excess.
    bb = robustness.block_bootstrap_excess(ohlc, events, horizon=5)
    print(f"\n[Block bootstrap +5d excess] mean={bb['mean']:+.4f} "
          f"95% CI [{bb['ci_low']:+.4f}, {bb['ci_high']:+.4f}] P(excess<=0)={bb['p_excess_le_0']:.2f}")

    # Family scan + deflation.
    sigs = {n: f(ret) for n, f in triggers.TRIGGERS.items()}
    scan = backtest.family_scan(ohlc, ret, sigs, exits.default_grid(), costs=backtest.CostModel())
    print(f"\n[Family scan] {len(scan)} (trigger x exit) combos, top 6 by net Sharpe:")
    print(scan.head(6).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    best = scan.iloc[0]
    # n_obs = the active days behind the winning Sharpe, NOT the calendar length:
    # these event strategies are flat >90% of the time, and inflating n_obs makes
    # the deflation test trivially easy to pass.
    dsr = robustness.deflated_sharpe(best["sharpe"], n_trials=len(scan),
                                     n_obs=int(best["active_days"]))
    print(f"\n[Deflated Sharpe] best raw={best['sharpe']:.2f} over {len(scan)} trials, "
          f"{int(best['active_days'])} active days "
          f"-> PSR={dsr:.2f} ({'survives selection' if dsr and dsr > 0.95 else 'consistent with luck'})")

    # Concrete backtest of the best combo + regime split + cost sweep.
    cd = max(20, 5)
    best_sig = triggers.first_crossings(sigs[best["trigger"]], cooldown=cd)
    # Reconstruct the winning exit rule from its label is messy; re-run the default
    # hold<=5 as a representative tradeable rule for the equity/regime view.
    res = backtest.run(ohlc, best_sig, exits.ExitRule(max_hold=5), backtest.CostModel())
    print(f"\n[Representative backtest | {best['trigger']}, hold<=5d, retail costs]")
    for k in ("n_trades", "cagr", "sharpe", "max_drawdown", "win_rate", "exposure"):
        v = res.stats[k]
        print(f"   {k:>15}: {v:.4f}" if isinstance(v, float) else f"   {k:>15}: {v}")

    print("\n[Regime split] is it one lucky era?")
    print(robustness.regime_split(res.daily).to_string(float_format=lambda v: f"{v:.4f}"))

    sweep = backtest.cost_sweep(ohlc, best_sig, exits.ExitRule(max_hold=5))
    print("\n[Cost sweep] net CAGR/Sharpe vs entry panic-slippage (bps):")
    print(sweep.to_string(float_format=lambda v: f"{v:.4f}"))

    return {"label": label, "stamp": stamp, "es": es, "bench": bench, "bb": bb,
            "scan": scan, "dsr": dsr, "res": res, "sweep": sweep}


def _write_results(faces: list[dict]) -> None:
    """The reproducible results doc behind the README verdict (house pattern)."""
    def md(df):
        return "```\n" + df.to_string(float_format=lambda v: f"{v:.4f}") + "\n```"

    lines = [
        "# Results — Study 02 (Falling-Knife) on the real Nasdaq-100",
        "",
        "*Generated by [`examples/verify_ndx.py`](../examples/verify_ndx.py). Trigger: T1",
        "close-to-close ≤ −3%, fresh events (20-session cooldown). Prices: Yahoo! Finance,",
        f"mode `split_only` (see [`falling_knife/data.py`](../falling_knife/data.py)). As-of **{DEFAULT_AS_OF}**.*",
        "",
    ]
    for f in faces:
        lines += [f"## {f['label']}", "", f"`{f['stamp']}`", ""]
        lines += [f"### Conditional vs a random day ({f['es']['n_events']} events)",
                  md(f["bench"][["n_events", "mean_cond", "mean_uncond", "excess", "p_greater"]]), ""]
        bb = f["bb"]
        lines += ["### Block bootstrap, +5d excess (clustering-aware)",
                  f"`mean={bb['mean']:+.4f}  95% CI [{bb['ci_low']:+.4f}, {bb['ci_high']:+.4f}]  "
                  f"P(excess<=0)={bb['p_excess_le_0']:.2f}`", ""]
        best = f["scan"].iloc[0]
        lines += ["### Family scan (70 trigger × exit combos) — top cell + deflation",
                  md(f["scan"].head(3)),
                  f"- deflated Sharpe of the winner: raw **{best['sharpe']:.2f}** on "
                  f"**{int(best['active_days'])} active days** over 70 trials -> PSR **{f['dsr']:.2f}** "
                  f"({'survives selection' if f['dsr'] and f['dsr'] > 0.95 else 'consistent with luck'})",
                  ""]
        s = f["res"].stats
        lines += ["### Representative backtest (best trigger, hold<=5d, retail costs)",
                  f"`n_trades={s['n_trades']}  cagr={s['cagr']:+.4f}  sharpe={s['sharpe']:.2f}  "
                  f"max_drawdown={s['max_drawdown']:.1%}  exposure={s['exposure']:.1%}`", ""]
    with open(RESULTS, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nwrote {RESULTS}")


def main():
    print("FALLING-KNIFE — live Nasdaq-100 study (Yahoo! Finance)")
    # Deep history on the spot index; tradeable face on the ETF.
    ndx = as_of(data.fetch("^NDX", mode="split_only"), DEFAULT_AS_OF)
    qqq = as_of(data.fetch("QQQ", mode="split_only"), DEFAULT_AS_OF)

    face_ndx = study("^NDX (spot, deep history)", ndx)
    face_qqq = study("QQQ (tradeable ETF)", qqq)
    bench_ndx, bench_qqq = face_ndx["bench"], face_qqq["bench"]
    es_qqq, scan_qqq, res_qqq, sweep_qqq = (face_qqq["es"], face_qqq["scan"],
                                            face_qqq["res"], face_qqq["sweep"])

    # Figures from the tradeable face.
    plots.plot_event_study(es_qqq, title="QQQ: path around a -3% day",
                           path=os.path.join(OUT_DIR, "out_ndx_eventstudy.png"))
    plots.plot_equity(res_qqq, title="QQQ falling-knife (T1, hold<=5d, retail costs)",
                      path=os.path.join(OUT_DIR, "out_ndx_equity.png"))
    plots.plot_cost_sweep(sweep_qqq, path=os.path.join(OUT_DIR, "out_ndx_costsweep.png"))
    plots.plot_family_heatmap(scan_qqq, metric="sharpe",
                              path=os.path.join(OUT_DIR, "out_ndx_family.png"))

    # Plain-language verdict from the 5-day excess and its significance.
    print(f"\n{'='*72}\n  VERDICT (read with the caveats in the README)\n{'='*72}")
    for name, bench in (("^NDX", bench_ndx), ("QQQ", bench_qqq)):
        if 5 in bench.index:
            row = bench.loc[5]
            beats = row["p_greater"] < 0.05 and row["excess"] > 0
            print(f"  {name}: +5d excess vs random day = {row['excess']:+.3%} "
                  f"(p_greater={row['p_greater']:.3f}) -> "
                  f"{'beats random' if beats else 'NOT distinguishable from buying any day'}")
    print("\nSaved: out_ndx_eventstudy.png, out_ndx_equity.png, out_ndx_costsweep.png, "
          "out_ndx_family.png")

    _write_results([face_ndx, face_qqq])


if __name__ == "__main__":
    main()
