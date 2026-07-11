"""Generate the two narrative notebooks for Study 688 (Concealing Baby Swallow).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached 111-name
basket under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic positive control runs anywhere with no
network.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily OHLCV,
# 111-ticker basket, 1962-01-02 -> 2026-06-30, 4,957 name-years searched).
R = dict(
    n_names=111, name_years=4957, max_years=64.5, oldest="1962-01-02", as_of="2026-06-30",
    min_n=8, n_loose=4, n_strict=0, bonferroni_crit=2.50,
    fp="d60f851e5d11", rows=1_248_608,
    events=[
        dict(ticker="GD", date="1962-06-14", strict=False,
             r1=+404.6, r5=-404.6, r10=+867.1, r20=+867.1),
        dict(ticker="BDX", date="1993-01-06", strict=False,
             r1=+97.4, r5=+32.5, r10=-64.9, r20=+340.9),
        dict(ticker="CL", date="2004-09-23", strict=False,
             r1=-148.4, r5=-141.8, r10=-309.8, r20=+57.6),
        dict(ticker="KMI", date="2011-08-22", strict=False,
             r1=+254.9, r5=+576.7, r10=+601.8, r20=+1157.5),
    ],
    # per-horizon: mean CBS return (bp, n=4 untested), base-rate n/mean/HAC-t, placebo p
    horizon={
        1: dict(cbs_mean=152.1, win=75, base_n=31552, base_mean=10.8, base_t=7.74,
                placebo_mean=9.7, placebo_p=0.084),
        5: dict(cbs_mean=15.7, win=50, base_n=31528, base_mean=85.7, base_t=19.97,
                placebo_mean=81.1, placebo_p=0.6145),
        10: dict(cbs_mean=273.5, win=50, base_n=31516, base_mean=126.2, base_t=19.73,
                 placebo_mean=122.7, placebo_p=0.3045),
        20: dict(cbs_mean=605.8, win=100, base_n=31491, base_mean=220.3, base_t=22.55,
                 placebo_mean=237.3, placebo_p=0.1925),
    },
    # synthetic control
    syn_null_mean=+0.68, syn_null_sd=0.95, syn_null_fire=1, syn_null_seeds=20,
    syn_null_n_lo=422, syn_null_n_hi=494,
    syn_planted_edge=0.05, syn_planted_n=456, syn_planted_mean=544.0,
    syn_planted_base=111.3, syn_planted_delta=432.7, syn_planted_t=14.39,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Too_rare_to_test%3F: Confirmed](https://img.shields.io/badge/Too_rare_to_test%3F-Confirmed-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from concealing_baby_swallow import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PANEL = data.load_real()
else:
    PANEL = None
print("real cache present:", HAVE_REAL, "| basket size:",
      (0 if PANEL is None else len(PANEL)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The rarest candle in the book — does it even exist? 🕯️🐦\n"
            "### The concealing baby swallow — four black candles that are supposed to "
            "mark the bottom, if you can ever find one\n\n"
            + BADGES +
            "Somewhere in the candlestick-charting canon, sandwiched between the famous "
            "patterns everyone's heard of, sits the **concealing baby swallow**: four "
            "black candles in a downtrend, arranged in a very specific way, that are "
            "supposed to mark **capitulation** — the point where the last sellers give up "
            "and the reversal begins.\n\n"
            "We went looking for it. Not in one stock — in **111** of the most liquid US "
            "stocks and ETFs, some with **64 years** of daily history. Here's what we "
            "found.\n\n"
            "> 📓 **Plain-language layer.** Want the exact detector, the *t*-stat "
            "bookkeeping and the synthetic proof the search actually works? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** Every bar of every ticker in the basket is scanned "
            "programmatically for the exact four-candle shape (two cuts: a loose, "
            "practical reading and a strict, literature-close one). House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| How many times did the pattern occur, ever? | **{R['n_loose']}** times "
            "(plain reading), across "
            f"**{R['n_names']}** stocks and **{R['name_years']:,}** stock-years of daily "
            f"data. The strict, book-accurate version occurred **{R['n_strict']}** times. "
            "|\n"
            "| Is that enough to test whether it 'works'? | **No.** Four data points (or "
            "zero) isn't a sample, it's an anecdote. We pre-registered a minimum of "
            f"**{R['min_n']}** occurrences before we'd even compute a statistic — this "
            "claim never got close. |\n"
            "| So is the detector broken? | **No** — on a synthetic world where we "
            "*plant* the exact pattern by hand, the same code finds it instantly and "
            "recovers a planted effect at *t* = "
            f"**{R['syn_planted_t']:.1f}**. It just almost never happens for real. |\n"
            "| Bottom line? | A trading rule you can count on one hand isn't a trading "
            "rule. |\n\n"
            "> You can't grade a test that never gets to run."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Two black marubozu — long bodies, no wicks, one-way selling. Then a "
            "third black candle that gaps down, rallies hard intraday back into the "
            "second candle's body (a long upper shadow — the rally the pattern "
            "**conceals**), but still closes low. Then a fourth black candle that "
            "completely swallows the third — trading above its high and below its low — "
            "and closes at a fresh low anyway. That failed rally, hidden inside a candle "
            "that still closed down, is the last gasp of the sellers. The bottom is "
            "close.\"*\n\n"
            "It's a real, named pattern (Nison's *Japanese Candlestick Charting "
            "Techniques*; Bulkowski's *Encyclopedia of Candlestick Charts*) — and by "
            "common agreement, the rarest one in the book. Bulkowski himself flags it as "
            "too infrequent to reliably rank against his other 100+ patterns."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If this pattern reliably marked capitulation bottoms, it would be one of "
            "the cleanest bullish signals in technical analysis — a precise, mechanical "
            "\"buy here\" flag with a story that actually makes sense (failed rally + new "
            "low = exhausted sellers). The catch, and the reason this study is different "
            "from the desk's other candlestick teardowns: **before we can ask if it "
            "works, we have to check if it ever happens.**"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **Cast the widest net on the desk.** {R['n_names']} tickers — SPY, QQQ, "
            "DIA, IWM, plus over 100 long-listed US large-caps across every sector — "
            f"scanned bar by bar, {R['name_years']:,} stock-years total, back to "
            f"{R['oldest']} where the data goes that far.\n"
            "- **Two honest cuts.** A loose, practical-chartist reading, and a strict cut "
            "that matches the book's language almost word for word (true marubozu, true "
            "gaps, the fourth candle opening exactly where the pattern says it should).\n"
            f"- **A line drawn *before* we looked.** Below **{R['min_n']}** occurrences, "
            "we don't compute a *t*-statistic at all — full stop, no exceptions, decided "
            "before we ran the scan.\n"
            "- **A sanity check.** Build a fake world where we hand-plant the exact "
            "pattern, and prove the code actually finds it (otherwise a zero count would "
            "mean nothing)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The scan.** Every single bar, every single ticker, both cuts."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(PANEL)\n"
            "    n_loose, n_strict = res['n_loose'], res['n_strict']\n"
            "else:\n"
            "    n_loose, n_strict = R['n_loose'], R['n_strict']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['loose cut\\n(plain reading)', 'strict cut\\n(literature-close)'],\n"
            "       [n_loose, n_strict], color=[AMBER, RED], width=.55)\n"
            "ax.axhline(R['min_n'], ls='--', c='k', lw=1.2, label=f\"pre-registered floor (n={R['min_n']})\")\n"
            "for i, v in enumerate([n_loose, n_strict]):\n"
            "    ax.annotate(str(v), (i, v), ha='center', va='bottom', fontsize=13, fontweight='bold')\n"
            "ax.set_ylabel(f\"occurrences across {R['n_names']} tickers, {R['name_years']:,} stock-years\")\n"
            "ax.set_ylim(0, max(R['min_n'] + 2, n_loose + 2))\n"
            "ax.set_title('The whole result, in one bar chart')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'loose: {n_loose}  strict: {n_strict}  (floor: {R[\"min_n\"]})')"
        ),
        md(
            f"That's it. **{R['n_loose']} occurrences**, ever, of the plain reading — and "
            f"**{R['n_strict']}** of the version that actually matches the book. Both sit "
            f"below the line we drew *before* running the scan. There is no statistic to "
            "report because there is no sample.\n\n"
            "**Here they all are** — every single occurrence found, with what happened "
            "next:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ev = st.pool_events(PANEL).sort_values('pos')\n"
            "    rows = []\n"
            "    for _, row in ev.iterrows():\n"
            "        t = row['ticker']; i = int(row['pos'])\n"
            "        d = PANEL[t].index[i]\n"
            "        rows.append((t, str(d.date()), row['ret_1']*1e4, row['ret_5']*1e4,\n"
            "                     row['ret_10']*1e4, row['ret_20']*1e4))\n"
            "else:\n"
            "    rows = [(e['ticker'], e['date'], e['r1'], e['r5'], e['r10'], e['r20'])\n"
            "            for e in R['events']]\n"
            "tbl = pd.DataFrame(rows, columns=['ticker', 'date', '1d bp', '5d bp', '10d bp', '20d bp'])\n"
            "display(tbl.round(1))\n"
            "print('mean 1d:', round(tbl['1d bp'].mean(), 1), 'bp  |  mean 20d:',\n"
            "      round(tbl['20d bp'].mean(), 1), 'bp   -- on n=4, decoration not evidence')"
        ),
        md(
            "No consistent story: one occurrence (KMI, 2011) looks great across every "
            "horizon; another (CL, 2004) loses money at every horizon out to 10 days; the "
            "oldest one (GD, 1962) traded at a split-adjusted **twelve cents a share** — "
            "at that price, 1960s tick sizes are coarse enough to *manufacture* a "
            "zero-shadow \"marubozu\" candle by rounding alone, pattern or no pattern. "
            "Four points, no pattern in the pattern.\n\n"
            "**Is the search itself trustworthy?** Build a fake world where we hand-plant "
            "the exact four-candle shape, and check the same code finds it:"
        ),
        code(
            "d0, _ = data.synthetic_panel(edge=0.0, seed=688)\n"
            "r0 = st.synthetic_detect(d0)\n"
            "d1, _ = data.synthetic_panel(edge=R['syn_planted_edge'], seed=688)\n"
            "r1 = st.synthetic_detect(d1)\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "ax.bar(['no planted effect\\n(null world)', 'planted reversal\\n(seed 688)'],\n"
            "       [r0['welch_t'], r1['welch_t']], color=[GREY, RED], width=.55)\n"
            "ax.axhline(2, ls='--', c='k', lw=1); ax.axhline(-2, ls='--', c='k', lw=1)\n"
            "for i, v in enumerate([r0['welch_t'], r1['welch_t']]):\n"
            "    ax.annotate(f'{v:+.2f}', (i, v), ha='center',\n"
            "                va='bottom' if v > 0 else 'top', fontsize=12)\n"
            "ax.set_ylabel('Welch t (reversal vs base rate)')\n"
            "ax.set_title('The detector works: quiet on nothing, loud on a real signal')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"null: n={r0['n']} events, t={r0['welch_t']:+.2f}  |  \"\n"
            "      f\"planted: n={r1['n']} events, t={r1['welch_t']:+.2f}\")"
        ),
        md(
            f"On a world with no real effect, the code finds hundreds of events and "
            f"correctly reports nothing significant (*t* near zero). On a world where we "
            f"hand-plant a genuine bounce, it finds the same hundreds of events and "
            f"correctly reports a screaming *t* = **{R['syn_planted_t']:.1f}**. The tool "
            "works. It simply never gets a real-world sample to work *with*."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Not \"tested and failed\" — genuinely too rare to test. "
            f"{R['n_loose']} loose-cut and {R['n_strict']} strict-cut occurrences across "
            f"{R['n_names']} tickers and {R['name_years']:,} stock-years, below the "
            f"pre-registered floor of {R['min_n']}.\n"
            "- **Tradability — Mirage.** There's nothing to charge costs against. One "
            "signal every ~15 years, pooled across 111 of the most liquid stocks in the "
            "market, is not a strategy.\n"
            "- **\"Too rare to ever test?\" — Confirmed.** The detector is proven honest "
            "on a synthetic world; the real world simply doesn't offer it a sample."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The general lesson.** The rarer and more precisely-specified a chart "
            "pattern is, the more likely a book's \"it works\" claim rests on a handful of "
            "cherry-picked historical charts rather than anything you could backtest. "
            "Rarity itself should be a red flag, not a mark of sophistication.\n"
            "- **Sibling studies:** [three black crows](../../408-three-black-crows/) "
            "(three candles, common enough to test — and it loses money shorted), "
            "[morning star](../../186-morning-star/) (three candles, common enough to "
            "test — and underperforms random days), [ladder bottom](../../687-ladder-bottom/) "
            "(five candles, the desk's other very-large-basket rarity case). None of them "
            "hit this study's wall: a sample size of essentially zero.\n\n"
            "*Think the pattern is real but we mis-coded the geometry? The detector "
            "(`concealing_baby_swallow/strategy.py`) is short and fully documented — show "
            "us where the definition is wrong, and where in the real tape a corrected "
            "version finds an actual sample.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# The Concealing Baby Swallow — a quantitative teardown 🔬\n"
            "### A pre-registered small-*n* discipline · a 111-ticker, 4,957 stock-year "
            "scan · loose vs strict detector cuts · a base-rate-matched event study · a "
            "20-seed synthetic faithful-engine control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — **a precise four-candle capitulation shape marks the end of a "
            "downtrend** — is one of the desk's cleanest tests of a different, prior "
            "question than usual: not *\"does the effect clear t >= 2\"*, but *\"does the "
            "effect even get a sample to be tested on\"*.\n\n"
            "> ⚠️ **Data note.** 111-ticker daily OHLCV basket (yfinance, `auto_adjust=True`), "
            "1962-01-02 → 2026-06-30, cache-first. No survivorship weighting inside the "
            "event study, but the basket itself is a **survivors** panel — named on the "
            "Signal axis. Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp"] +
            f"`, {R['rows']:,} rows).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | {R['n_loose']} loose-cut / {R['n_strict']} strict-cut "
            f"occurrences across {R['n_names']} tickers, {R['name_years']:,} stock-years — "
            f"below the pre-registered floor **n >= {R['min_n']}**; no *t*-statistic is "
            "computed |\n"
            "| **Tradability** | `MIRAGE` | nothing to charge costs against; ~1 signal per "
            "111-ticker-basket per 15 years |\n"
            "| **Too rare to ever test?** | `CONFIRMED` | synthetic control proves the "
            f"detector unbiased (null 20-seed mean *t* = {R['syn_null_mean']:+.2f}, planted "
            f"*t* = {R['syn_planted_t']:.2f}); the real-tape count is a market property, "
            "not a detector bug |\n\n"
            "> 💡 In plain words: this study's job wasn't to grade a signal — it was to "
            "find out whether there was anything to grade."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a **concealing baby swallow** at bar $t$ (day 4) require, over bars "
            "$t{-}3..t$:\n\n"
            "- **Downtrend context**: $c_{t-3} < c_{t-3-L}$ for lookback $L$.\n"
            "- **Days 1-2 near-marubozu**, both bearish: shadow fraction "
            "$\\le \\epsilon_1$ of range.\n"
            "- **Day 3**: bearish, closes below day 2, small lower shadow "
            "($\\le \\epsilon_2$ of range), but a real upper shadow "
            "($\\ge \\epsilon_3$ of range) that reaches $high_{t-1} > c_{t-2}$ — the "
            "\"concealed\" rally.\n"
            "- **Day 4**: bearish, fully engulfs day 3 ($high_t \\ge high_{t-1}$, "
            "$low_t \\le low_{t-1}$), closes below $c_{t-1}$.\n\n"
            "**H1 (existence).** The shape occurs often enough, on a real, large, "
            "long-history equity tape, to define a testable sample.\n\n"
            "**H2 (reversal, conditional on H1).** Conditional on H1, the long return "
            "after day 4's confirming close, entered at $t{+}1$'s open, exceeds the "
            "matched base rate (same long bet after any four-red-days-in-a-downtrend, "
            "regardless of the precise geometry).\n\n"
            f"**We find H1 false.** {R['n_loose']} loose-cut occurrences (0 strict) "
            f"across {R['n_names']} tickers and {R['name_years']:,} stock-years sits "
            f"below the pre-registered floor of {R['min_n']} — **H2 cannot even be "
            "posed**, let alone tested."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The decisive design choice here isn't a statistic — it's a **stopping "
            f"rule, fixed before the scan ran**: `strategy.MIN_N_FOR_TEST = {R['min_n']}`. "
            "Below that pooled event count, `strategy.welch_t` and `strategy.summarize` "
            "return `None`/`tested=False` rather than computing anything. This matters "
            "because a *t*-statistic on 4 (or 0) observations is not merely weak evidence "
            "— it is *decoration*: with `n=4` a single outlier observation can swing the "
            "mean by hundreds of basis points and there is no meaningful notion of a "
            "sampling distribution to appeal to. Reporting a formal Welch or HAC *t* here "
            "would manufacture false precision, exactly the failure mode "
            "[`METHODOLOGY.md`](../../../METHODOLOGY.md) warns against (\"a *t* on a "
            "handful of observations is theatre, not evidence\")."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {R['n_names']} tickers (SPY/QQQ/DIA/IWM + long-listed US "
            f"large-caps), yfinance daily OHLCV, cache-first, as-of {R['as_of']}.\n"
            f"- **Coverage.** {R['name_years']:,} stock-years searched; oldest bar "
            f"{R['oldest']}; longest single series spans {R['max_years']} years.\n"
            "- **Detector.** Two cuts — loose (plain geometric reading) and strict "
            "(literature-close: true marubozu, true gap-downs, day 4 opening inside day "
            "3's shadow) — both confirmed at day 4's close, no look-ahead.\n"
            "- **Execution.** Enter next session's open (one lag), long-only, hold fixed "
            "horizons 1/5/10/20 sessions.\n"
            "- **Base rate.** Same long bet on every bar matching the coarse downtrend + "
            "four-red-closes context, regardless of the precise geometry — isolates the "
            "pattern's *specific* information from generic post-decline mean reversion.\n"
            f"- **Stopping rule.** No *t*-statistic below n = {R['min_n']} pooled events, "
            "fixed ex ante. Four horizons -> Bonferroni-corrected critical "
            f"|t| = {R['bonferroni_crit']:.2f} where a test *does* run.\n"
            "- **Control.** Synthetic panel with the *exact* engineered four-candle "
            "geometry at a controlled rate and a tunable planted bounce; the null must "
            "not fire across 20 seeds, and a planted edge must be recovered."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The scan — both cuts, against the pre-registered floor"
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(PANEL)\n"
            "    n_loose, n_strict = res['n_loose'], res['n_strict']\n"
            "    years, name_years = res['years'], res['name_years']\n"
            "else:\n"
            "    n_loose, n_strict = R['n_loose'], R['n_strict']\n"
            "    years, name_years = R['max_years'], R['name_years']\n"
            "print(f\"loose cut: {n_loose} occurrences | strict cut: {n_strict} occurrences\")\n"
            "print(f\"basket: {R['n_names']} tickers, {name_years:,.0f} name-years, \"\n"
            "      f\"max span {years:.1f} years\")\n"
            "print(f\"pre-registered floor: n >= {R['min_n']} -> \"\n"
            "      f\"{'BELOW: no t-statistic computed' if n_loose < R['min_n'] else 'cleared'}\")"
        ),
        md(
            "### 4b · The base rate — context, not evidence\n\n"
            "The matched base rate (same long bet on every bar in the coarse "
            "\"four-red-days-in-a-downtrend\" context) is well-populated and carries a "
            "real, HAC-significant positive drift — the ordinary post-decline bounce, "
            "unrelated to the specific pattern claim:"
        ),
        code(
            "hs = sorted(R['horizon'])\n"
            "base_means = [R['horizon'][h]['base_mean'] for h in hs]\n"
            "base_ts = [R['horizon'][h]['base_t'] for h in hs]\n"
            "cbs_means = [R['horizon'][h]['cbs_mean'] for h in hs]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.4))\n"
            "a1.bar([str(h) for h in hs], base_means, color=GREY, width=.55)\n"
            "for i, v in enumerate(base_means): a1.annotate(f'{v:.1f}', (i, v), ha='center', va='bottom')\n"
            "a1.set_title('Base rate (n > 31,000): the generic bounce'); a1.set_ylabel('mean return (bp)')\n"
            "a1.set_xlabel('horizon (sessions)')\n"
            "a2.bar([str(h) for h in hs], cbs_means, color=AMBER, width=.55)\n"
            "for i, v in enumerate(cbs_means): a2.annotate(f'{v:.1f}', (i, v), ha='center', va='bottom')\n"
            "a2.set_title('The 4 CBS occurrences (UNTESTED, n=4)'); a2.set_xlabel('horizon (sessions)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h in hs:\n"
            "    d = R['horizon'][h]\n"
            "    print(f\"h={h:>2d}: base n={d['base_n']:,} mean={d['base_mean']:+.1f}bp \"\n"
            "          f\"(HAC t={d['base_t']:+.2f})  |  CBS mean={d['cbs_mean']:+.1f}bp \"\n"
            "          f\"(n=4, win={d['win']}%, untested)\")"
        ),
        md(
            "> 💡 In plain words: don't be fooled by the right-hand panel's big numbers — "
            "with n = 4 they're barely more informative than four coin flips. The left "
            "panel is the real, testable finding: *any* long bet after a run of red days "
            "in a downtrend tends to bounce, whether or not the specific candle geometry "
            "shows up. That bounce is not this pattern's edge — it's the market's."
        ),
        md(
            "### 4c · The four occurrences, and the honest placebo\n\n"
            "Every loose-cut occurrence, in full — and a descriptive-only (not "
            "certifying) label-shuffle placebo against the base-rate pool."
        ),
        code(
            "ev_rows = [(e['ticker'], e['date'], e['r1'], e['r5'], e['r10'], e['r20'])\n"
            "           for e in R['events']]\n"
            "tbl = pd.DataFrame(ev_rows, columns=['ticker', 'date', '1d bp', '5d bp', '10d bp', '20d bp'])\n"
            "display(tbl)\n"
            "print()\n"
            "for h in hs:\n"
            "    d = R['horizon'][h]\n"
            "    print(f\"h={h:>2d}: observed {d['cbs_mean']:+.1f}bp vs placebo mean \"\n"
            "          f\"{d['placebo_mean']:+.1f}bp (2,000 draws of 4)  ->  p = {d['placebo_p']:.4f}\")"
        ),
        md(
            "> 💡 In plain words: none of the four placebo *p*-values is remotely "
            "significant, even before any multiple-testing correction (Bonferroni at k=4 "
            "would need *p* < 0.0125). This is printed for transparency only — a *p*-value "
            "computed from a 4-observation sample is not a certifying statistic under any "
            "reading of the desk's inference bar; it exists so a skeptical reader can see "
            "we didn't quietly run the test and hide an inconvenient result."
        ),
        md(
            "### 4d · Faithful-engine & power control — proving the search itself works\n\n"
            "A synthetic panel with the *exact* engineered four-candle geometry (true "
            "marubozu, true gap-and-fail rally, true full engulf) planted at a controlled "
            "rate, with a tunable post-pattern bounce. The null (`edge=0`) is checked over "
            "**20 seeds** — never a single stream."
        ),
        code(
            "null_ts, null_ns = [], []\n"
            "for s_ in range(20):\n"
            "    d0, _ = data.synthetic_panel(edge=0.0, seed=688 + s_)\n"
            "    r0 = st.synthetic_detect(d0, seed=688 + s_)\n"
            "    null_ts.append(r0['welch_t']); null_ns.append(r0['n'])\n"
            "null_ts = np.asarray(null_ts, dtype=float)\n"
            "d1, _ = data.synthetic_panel(edge=R['syn_planted_edge'], seed=688)\n"
            "r1 = st.synthetic_detect(d1, seed=688)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (edge=0), 20 seeds')\n"
            "ax.scatter([1], [r1['welch_t']], color=RED, s=90, zorder=5,\n"
            "           label=f\"planted edge={R['syn_planted_edge']}\")\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Welch t (reversal vs base rate)')\n"
            "ax.set_title('Control: quiet on nulls (1/20 fires, ~nominal rate), loud on a planted edge')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds, events/seed '\n"
            "      f'{min(null_ns)}-{max(null_ns)}')\n"
            "print(f\"planted: n={r1['n']} events, mean {r1['mean_bps']:+.1f}bp vs base \"\n"
            "      f\"{r1['base_bps']:+.1f}bp (delta {r1['delta_bps']:+.1f}bp)  \"\n"
            "      f\"Welch t = {r1['welch_t']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector fires "
            f"(\\|t\\| >= 2) in **{R['syn_null_fire']}/{R['syn_null_seeds']}** seeds — "
            "almost exactly the *nominal* 5% false-positive rate of that threshold, i.e. "
            "no systematic bias toward manufacturing significance. On a world with a "
            f"genuine planted edge it recovers *t* = **{R['syn_planted_t']:.2f}** cleanly "
            f"from {R['syn_planted_n']} found events (out of ~640 planted blocks — the "
            "detector's loose cut has real recall). **The machinery is sound.** The "
            f"real tape's {R['n_loose']}-occurrence count is therefore a fact about how "
            "rare this candle geometry actually is in live markets, not an artifact of "
            "an over-strict or buggy rule."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — {R['n_loose']} loose-cut / {R['n_strict']} strict-cut "
            f"occurrences across {R['n_names']} tickers and {R['name_years']:,} "
            f"stock-years, below the pre-registered floor n >= {R['min_n']}. Per "
            "[`METHODOLOGY.md`](../../../METHODOLOGY.md), `WEAK` is for a real, if "
            "fragile, point estimate; here there is no testable point estimate at all — "
            "the honest grade is `NONE`, not a manufactured *t* on four data points.\n"
            "- **Tradability `MIRAGE`** — no venue, no capacity, no book: roughly one "
            "signal every 15 years pooled across the desk's largest single-pattern "
            "basket. There is nothing to charge costs against.\n"
            "- **\"Too rare to ever test?\" `CONFIRMED`** — the synthetic control proves "
            "the detector recovers a planted edge cleanly and stays within the nominal "
            "false-positive rate on nulls; the real-tape scarcity is therefore a genuine "
            "market fact, and this specific claim cannot be falsified or confirmed at any "
            "practical sample size."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson: rarity is itself informative.** A pattern this "
            "combinatorially restrictive (two near-perfect marubozu *and* a precise "
            "overlap-then-full-engulf geometry) will, by construction, almost never fire "
            "on real data. Popular technical-analysis literature rarely reports "
            "occurrence counts alongside \"it works\" claims — this study suggests that "
            "omission is not an accident for the rarest formations.\n"
            "- **A natural extension** would loosen the geometry parametrically "
            "(`marubozu_frac`, `warn_shadow_frac`, `tail_frac` in `strategy.cbs_flags`) "
            "and trace out occurrence count vs. how far the loosened definition drifts "
            "from the book's language — the point at which a testable sample appears is "
            "itself informative about how much of the \"pattern\" survives dilution.\n"
            "- **Dedup map:** [408-three-black-crows](../../408-three-black-crows/) "
            "(three candles, common enough to test, bearish, loses shorted), "
            "[186-morning-star](../../186-morning-star/) (three candles, common enough to "
            "test, underperforms random days), "
            "[687-ladder-bottom](../../687-ladder-bottom/) (five candles, the desk's other "
            "very-large-basket rarity case, without this study's precise overlap/engulf "
            "geometry). None of them hit this study's wall: an unusable sample size.\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "02_for_the_quants.ipynb")


def _meta():
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }


def _write(nb, name):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
