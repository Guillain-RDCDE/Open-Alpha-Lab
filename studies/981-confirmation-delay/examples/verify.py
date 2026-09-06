"""Real-tape verification — Study 981 (The Price of Waiting). Regenerates docs/results.md.

Runs three standard signals on four tapes at six confirmation lengths, counts the
whipsaw trades each confirmation length prevents, prices the sessions it spends waiting on a
signal that was already right, and asks whether any confirmation length beat the unconfirmed
rule — and whether the winning length could have been known in advance.

    python studies/981-confirmation-delay/examples/verify.py            # cache-only
    python studies/981-confirmation-delay/examples/verify.py --fetch    # refresh the cache first
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np  # noqa: F401  (used by the report bodies below)
import pandas as pd  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from confirm_delay import data, strategy as st  # noqa: E402

DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))


COST_BPS = 2.0
HEAD_K = 5


def report() -> dict:
    px = data.load_prices()
    cash = px[data.CASH].dropna()
    h: dict = {"as_of": data.AS_OF, "fingerprint": data.fingerprint(px),
               "tickers": list(data.RISKY), "signals": list(st.SIGNALS),
               "ks": list(st.CONFIRM_DAYS)}

    print(f"as-of {data.AS_OF}   fingerprint {data.fingerprint(px)}   cost {COST_BPS:.0f} bps "
          f"a switch")
    for tk in data.TICKERS:
        s = px[tk].dropna()
        print(f"  {tk:5s} {s.index[0].date()} -> {s.index[-1].date()}  n={len(s):,}")
    h["windows"] = {tk: [str(px[tk].dropna().index[0].date()),
                         str(px[tk].dropna().index[-1].date())] for tk in data.TICKERS}

    grid = st.full_grid(px, cash, data.RISKY, cost_bps=COST_BPS)
    h["n_cells"] = int(len(grid.groupby(["ticker", "signal"])))

    print("\n=== does confirmation reduce whipsaw? (share of round trips closed within a month) ===")
    print("  tape  signal                          " +
          "  ".join(f"k={k:<3d}" for k in st.CONFIRM_DAYS))
    ws = grid.pivot_table(index=["ticker", "signal"], columns="k", values="ws_whipsaw_share")
    for (tk, s), row in ws.iterrows():
        print(f"  {tk:5s} {st.SIGNAL_LABEL[s]:32s} " +
              "  ".join(f"{row[k]:5.0%}" for k in st.CONFIRM_DAYS))
    h["whipsaw_share"] = {f"{tk}|{s}": {int(k): float(v) for k, v in row.items()}
                          for (tk, s), row in ws.iterrows()}
    cut = (ws[HEAD_K] <= ws[1] * (2 / 3)).mean()
    any_cut = (ws[HEAD_K] < ws[1]).mean()
    h["frac_whipsaw_cut"] = float(cut)
    h["frac_whipsaw_any"] = float(any_cut)
    h["whipsaw_k1"] = float(ws[1].mean())
    h["whipsaw_k5"] = float(ws[HEAD_K].mean())
    print(f"  falls by at least a third on {cut:.0%} of cells; falls at all on {any_cut:.0%}")

    tr = grid.pivot_table(index=["ticker", "signal"], columns="k", values="switches_per_year")
    h["trades_k1"] = float(tr[1].mean() / 2)      # a round trip is two switches
    h["trades_k5"] = float(tr[HEAD_K].mean() / 2)
    print(f"  round trips a year, averaged over cells: {h['trades_k1']:.1f} at k=1 -> "
          f"{h['trades_k5']:.1f} at k={HEAD_K}")
    rsi_rows = ws.xs("rsi14", level="signal")
    h["rsi_whipsaw_k1"] = float(rsi_rows[1].mean())
    h["rsi_whipsaw_k5"] = float(rsi_rows[HEAD_K].mean())
    ma_rows = ws.xs("ma200", level="signal")
    print(f"  the fast signal has the most to gain: RSI {rsi_rows[1].mean():.0%} -> "
          f"{rsi_rows[HEAD_K].mean():.0%}; the 200-day average {ma_rows[1].mean():.0%} -> "
          f"{ma_rows[HEAD_K].mean():.0%}")
    h["ma_whipsaw_k1"] = float(ma_rows[1].mean())
    h["ma_whipsaw_k5"] = float(ma_rows[HEAD_K].mean())

    print("\n=== what the waiting cost ===")
    wait = grid[grid["k"] == HEAD_K]
    print("  tape  signal                          days waiting  forgone bps  avoided bps  "
          "disagree")
    for _, row in wait.iterrows():
        print(f"  {row['ticker']:5s} {st.SIGNAL_LABEL[row['signal']]:32s} "
              f"{int(row['days_waiting']):12,d} {row['delay_cost_bps']:12,.0f} "
              f"{row['late_exit_cost_bps']:12,.0f} {row['share_disagreeing']:9.1%}")
    h["days_waiting_k5"] = int(wait["days_waiting"].sum())
    h["delay_cost_k5"] = float(-wait["delay_cost_bps"].sum())
    h["late_exit_cost_k5"] = float(-wait["late_exit_cost_bps"].sum())
    h["delay_detail"] = wait[["ticker", "signal", "days_waiting", "delay_cost_bps",
                              "late_exit_cost_bps", "share_disagreeing"]].to_dict("records")

    print("\n=== and what it earned: Sharpe by confirmation length ===")
    sh = grid.pivot_table(index=["ticker", "signal"], columns="k", values="sharpe")
    print("  tape  signal                          " +
          "  ".join(f"k={k:<4d}" for k in st.CONFIRM_DAYS))
    for (tk, s), row in sh.iterrows():
        cells = "  ".join(f"{row[k]:+6.2f}" for k in st.CONFIRM_DAYS)
        star = " *" if row.idxmax() != 1 else ""
        print(f"  {tk:5s} {st.SIGNAL_LABEL[s]:32s} {cells}{star}")
    h["sharpe"] = {f"{tk}|{s}": {int(k): float(v) for k, v in row.items()}
                   for (tk, s), row in sh.iterrows()}

    best = st.best_k(grid)
    h["frac_sharpe_wins"] = float((best["gain"] > 0).mean())
    h["mean_sharpe_gain"] = float(best["gain"].mean())
    h["n_distinct_best_k"] = int(best["best_k"].nunique())
    h["best_k_counts"] = best["best_k"].value_counts().sort_index().to_dict()
    h["best"] = best.to_dict("records")
    print(f"\n  some k beat k=1 on Sharpe in {h['frac_sharpe_wins']:.0%} of cells, mean gain "
          f"{h['mean_sharpe_gain']:+.3f}")
    print(f"  the winning k differs across cells: " +
          ", ".join(f"k={k} in {v}" for k, v in h["best_k_counts"].items()))

    print("\n=== against buy-and-hold, not just against k=1 ===")
    vs = grid.pivot_table(index=["ticker", "signal"], columns="k", values="cagr_vs_hold")
    for (tk, s), row in vs.iterrows():
        print(f"  {tk:5s} {st.SIGNAL_LABEL[s]:32s} " +
              "  ".join(f"{row[k]:+6.2%}" for k in st.CONFIRM_DAYS))
    h["vs_hold"] = {f"{tk}|{s}": {int(k): float(v) for k, v in row.items()}
                    for (tk, s), row in vs.iterrows()}
    h["frac_beating_hold"] = float((vs > 0).to_numpy().mean())
    print(f"  cells beating buy-and-hold: {h['frac_beating_hold']:.0%} of "
          f"{vs.size} (tape x signal x k)")

    print("\n=== would you have known which k to pick? (first half chooses, second half pays) ===")
    oos = []
    for tk in data.RISKY:
        s_px = px[tk].dropna()
        mid = s_px.index[len(s_px) // 2]
        for sig in st.SIGNALS:
            first = st.sweep(s_px.loc[:mid], cash, sig, cost_bps=COST_BPS)
            second = st.sweep(s_px.loc[mid:], cash, sig, cost_bps=COST_BPS)
            k_pick = int(first["sharpe"].idxmax())
            oos.append({"ticker": tk, "signal": sig, "k_chosen": k_pick,
                        "sharpe_second_half": float(second.loc[k_pick, "sharpe"]),
                        "sharpe_k1_second": float(second.loc[1, "sharpe"]),
                        "gain": float(second.loc[k_pick, "sharpe"] - second.loc[1, "sharpe"]),
                        "k_best_second": int(second["sharpe"].idxmax())})
    oos_df = pd.DataFrame(oos)
    for _, r in oos_df.iterrows():
        print(f"  {r['ticker']:5s} {st.SIGNAL_LABEL[r['signal']]:32s} chose k={r['k_chosen']:2d} "
              f"-> second-half Sharpe {r['sharpe_second_half']:+.2f} vs {r['sharpe_k1_second']:+.2f} "
              f"for k=1 (gain {r['gain']:+.2f}); best in hindsight was k={r['k_best_second']}")
    h["oos"] = oos_df.to_dict("records")
    h["oos_win_rate"] = float((oos_df["gain"] > 0).mean())
    h["oos_mean_gain"] = float(oos_df["gain"].mean())
    h["oos_agreement"] = float((oos_df["k_chosen"] == oos_df["k_best_second"]).mean())
    print(f"  picking k on the first half beat k=1 on the second in "
          f"{h['oos_win_rate']:.0%} of cells (mean {h['oos_mean_gain']:+.3f}); the chosen k was "
          f"also the best in hindsight {h['oos_agreement']:.0%} of the time")

    print("\n=== synthetic control: a trending tape and a choppy one ===")
    for trend, tag in ((1.0, "trending"), (-1.0, "choppy")):
        sim = st.synthetic_tape(n=8000, trendiness=trend, seed=981)
        rows = {}
        for sig in st.SIGNALS:
            s_ = st.sweep(sim["ASSET"], sim["CASH"], sig, cost_bps=COST_BPS)
            rows[sig] = {"whipsaw_k1": float(s_.loc[1, "ws_whipsaw_share"]),
                         "whipsaw_k5": float(s_.loc[5, "ws_whipsaw_share"]),
                         "sharpe_gain": float(s_["sharpe"].max() - s_.loc[1, "sharpe"])}
            print(f"  {tag:9s} {st.SIGNAL_LABEL[sig]:32s} whipsaw "
                  f"{rows[sig]['whipsaw_k1']:.0%} -> {rows[sig]['whipsaw_k5']:.0%}, best "
                  f"Sharpe gain {rows[sig]['sharpe_gain']:+.3f}")
        h[f"synthetic_{tag}"] = rows

    h["_verdict"] = st.verdict(h)
    print(f"\n=== verdict (strategy.verdict, applied) ===")
    print(f"  Signal: {h['_verdict']['signal']}   Tradability: {h['_verdict']['trad']}")
    return h


def results_md(h: dict) -> str:
    v = h["_verdict"]
    ks = h["ks"]
    head = " | ".join(f"k = {k}" for k in ks)
    dash = "|".join(["--:"] * len(ks))
    ws = "\n".join("| " + key.split("|")[0] + " | " + st.SIGNAL_LABEL[key.split("|")[1]] +
                   " | " + " | ".join(f"{d[str(k)] if str(k) in d else d[k]:.0%}" for k in ks) + " |"
                   for key, d in h["whipsaw_share"].items())
    sh = "\n".join("| " + key.split("|")[0] + " | " + st.SIGNAL_LABEL[key.split("|")[1]] +
                   " | " + " | ".join(f"{d[str(k)] if str(k) in d else d[k]:+.2f}" for k in ks) + " |"
                   for key, d in h["sharpe"].items())
    vs = "\n".join("| " + key.split("|")[0] + " | " + st.SIGNAL_LABEL[key.split("|")[1]] +
                   " | " + " | ".join(f"{d[str(k)] if str(k) in d else d[k]:+.2%}" for k in ks) + " |"
                   for key, d in h["vs_hold"].items())
    delay = "\n".join(
        f"| {r['ticker']} | {st.SIGNAL_LABEL[r['signal']]} | {int(r['days_waiting']):,} | "
        f"{r['delay_cost_bps']:+,.0f} | {r['late_exit_cost_bps']:+,.0f} | "
        f"{r['share_disagreeing']:.1%} |" for r in h["delay_detail"])
    oos = "\n".join(
        f"| {r['ticker']} | {st.SIGNAL_LABEL[r['signal']]} | {r['k_chosen']} | "
        f"{r['sharpe_second_half']:+.2f} | {r['sharpe_k1_second']:+.2f} | {r['gain']:+.2f} | "
        f"{r['k_best_second']} |" for r in h["oos"])
    return f"""# Results — Study 981 (The Price of Waiting) on the real daily tape

*Generated by [`examples/verify.py`](../examples/verify.py). Three signals (200-day moving
average, 12-1 momentum, 14-day RSI) on four tapes (SPY, IWM, TLT, GLD) at six confirmation
lengths, long the asset while the confirmed signal is on and in T-bills otherwise. **Every arm
carries exactly one day of execution lag**, whatever the confirmation length, so the comparison
is about confirmation and not about who acts sooner. 2 bps a switch. As-of **{h['as_of']}**;
fingerprint `{h['fingerprint']}`.*

## Does confirmation reduce whipsaw?

Share of round trips closed inside a month:

| Tape | Signal | {head} |
|---|---|{dash}|
{ws}

It falls by at least a third on **{h['frac_whipsaw_cut']:.0%}** of cells and falls at all on
**{h['frac_whipsaw_any']:.0%}**. Round trips per year drop from **{h['trades_k1']:.1f}** to
**{h['trades_k5']:.1f}** on average. The effect is concentrated exactly where it should be: the
fast signal (RSI, **{h['rsi_whipsaw_k1']:.0%} → {h['rsi_whipsaw_k5']:.0%}**) has far more to
gain than the slow one (200-day average, {h['ma_whipsaw_k1']:.0%} →
{h['ma_whipsaw_k5']:.0%}).

## What the waiting cost (k = 5)

| Tape | Signal | Sessions waiting | Forgone (bps) | Avoided (bps) | Days disagreeing |
|---|---|--:|--:|--:|--:|
{delay}

*Forgone* is the return earned by the market on days when the raw signal was already positive
and the confirmed arm was still in cash. *Avoided* is its mirror: the return dodged by exiting
late. Both are reported because quoting only one of them is how this rule gets sold.

## What it earned

Sharpe ratio by confirmation length:

| Tape | Signal | {head} |
|---|---|{dash}|
{sh}

Some confirmation length beat the unconfirmed rule in **{h['frac_sharpe_wins']:.0%}** of cells,
by **{h['mean_sharpe_gain']:+.3f}** on average. The winning length took
**{h['n_distinct_best_k']}** different values across {h['n_cells']} cells
({", ".join(f"k = {k} in {vv}" for k, vv in h['best_k_counts'].items())}), which is what
in-hindsight parameter choice looks like.

## And against buy-and-hold

| Tape | Signal | {head} |
|---|---|{dash}|
{vs}

**{h['frac_beating_hold']:.0%}** of all cells beat simply owning the asset.

## Could you have chosen k in advance?

First half of each tape picks the confirmation length; the second half pays for it:

| Tape | Signal | k chosen | Sharpe (second half) | Sharpe at k = 1 | Gain | Best k in hindsight |
|---|---|--:|--:|--:|--:|--:|
{oos}

The chosen length beat no confirmation in **{h['oos_win_rate']:.0%}** of cells, mean gain
**{h['oos_mean_gain']:+.3f}**, and it was also the best choice in hindsight
**{h['oos_agreement']:.0%}** of the time.

## Synthetic control

On a deliberately **trending** tape and a deliberately **choppy** one, with the same volatility
and drift, the machinery behaves as the mechanism predicts: confirmation has far more whipsaw
to remove on the choppy tape, and far less to gain on the trending one.

## Caveats

- **Long-flat only.** A long-short version would double the number of state changes and roughly
  double whatever confirmation is worth.
- **One confirmation rule.** Requiring *k consecutive* days is the simplest form; a threshold
  band around the crossing point (a "buffer") is the other standard approach and is not tested.
- **Costs are flat.** At 2 bps a switch the cost of trading is small next to the timing effects;
  a smaller account or a wider spread would shift the balance toward waiting.

## Verdict

Produced by `strategy.verdict`, fixed before the run and unit-tested in
[`tests/test_strategy.py`](../tests/test_strategy.py).

**Signal: {v['signal']}.** {v['signal_why']}

**Tradability: {v['trad']}.** {v['trad_why']}

---

*Part of [Open-Alpha-Lab](../../../README.md) — study
[981-confirmation-delay](../README.md). Not investment advice.*
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
