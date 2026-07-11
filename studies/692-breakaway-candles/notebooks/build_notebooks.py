"""Generate the two narrative notebooks for Study 692 (Breakaway Candles).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily
tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (yfinance daily OHLCV,
# SPY + 60 large-caps, ~25.0 years, as-of 2026-06-30; loose/strict breakaway detector).
R = dict(
    years=25.0, n_names=61, n_bars=383_080,
    n_loose=44, n_strict=5, n_bullish=27, n_bearish=17,
    bonferroni_crit=2.50,
    combined={
        1: dict(n=44, mean=-57.1, win=43.2, base=-0.2, delta=-56.8, t=-2.16, p=0.993, net=-67.1),
        5: dict(n=44, mean=13.6, win=43.2, base=1.1, delta=12.5, t=0.24, p=0.409, net=3.6),
        10: dict(n=44, mean=113.6, win=63.6, base=0.4, delta=113.2, t=1.44, p=0.070, net=103.6),
        20: dict(n=44, mean=325.8, win=56.8, base=6.7, delta=319.1, t=2.39, p=0.002, net=315.8),
    },
    strict={
        1: dict(n=5, mean=-168.2, delta=-168.0, p=0.993),
        5: dict(n=5, mean=-318.6, delta=-319.5, p=0.979),
        10: dict(n=5, mean=-82.2, delta=-82.5, p=0.640),
        20: dict(n=5, mean=587.7, delta=581.1, p=0.041),
    },
    bullish={
        1: dict(n=27, mean=-116.2, base=2.8, delta=-119.0, t=-3.43, p=1.000),
        5: dict(n=27, mean=42.2, base=27.7, delta=14.5, t=0.19, p=0.414),
        10: dict(n=27, mean=256.4, base=55.5, delta=200.9, t=2.15, p=0.026),
        20: dict(n=27, mean=670.8, base=120.0, delta=550.8, t=3.33, p=0.001),
    },
    bearish={
        1: dict(n=17, mean=36.8, base=-2.6, delta=39.4, t=1.39, p=0.096),
        5: dict(n=17, mean=-31.9, base=-19.6, delta=-12.3, t=-0.18, p=0.567),
        10: dict(n=17, mean=-113.2, base=-42.5, delta=-70.6, t=-0.57, p=0.743),
        20: dict(n=17, mean=-222.1, base=-81.5, delta=-140.6, t=-0.93, p=0.814),
    },
    best_worst=[
        ("BDX", "2002-06-11", "2002-06-17", -1792.7),
        ("BDX", "2011-08-17", "2011-08-23", -598.1),
        ("LOW", "2002-07-18", "2002-07-24", +1584.6),
        ("DIS", "2025-04-03", "2025-04-09", +1821.9),
        ("CAT", "2010-08-23", "2010-08-27", +2026.0),
    ],
    syn_null_mean=+0.58, syn_null_sd=0.90, syn_null_fire=2, syn_null_seeds=20,
    syn_edge02_t=+4.43, syn_edge04_t=+7.96,
    fp_panel="9693c3ea7807", fp_spy="85f86a841de4",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Works_both_directions_equally%3F: Busted](https://img.shields.io/badge/Works_both_directions_equally%3F-Busted-8b949e?style=flat-square)\n\n"
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

from breakaway_candles import data, strategy as st

HAVE_REAL = data.have_real()
PANEL = data.load_real() if HAVE_REAL else None
print("real cache present:", HAVE_REAL, "| names:", (0 if PANEL is None else len(PANEL)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# A gap, a run, a reversal — does the \"breakaway\" candle really break away? 🕳️\n"
            "### The five-candle breakaway pattern — a chart shape that looks unmistakable and "
            "trades almost nothing\n\n"
            + BADGES +
            "Chartists love this one: a stock is falling, it **gaps down** one morning, keeps "
            "falling for two more days — then a big **reversal candle** shows up and closes "
            "*back through the gap*, as if the whole down-leg never happened. The story writes "
            "itself: the sellers who chased the gap are now underwater, the buyers who waited are "
            "vindicated, and the trend is supposedly over. Works upside-down too, in an uptrend.\n\n"
            "That's the claim we test: **the gap-then-run-then-reversal shape marks the end of "
            "the trend it interrupted.** Six independent things have to line up for this pattern "
            "to even *exist* — which is exactly why it turns out to be almost too rare to trust.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the Bonferroni correction and the "
            "symmetry check? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** SPY + 60 long-listed large-caps, ~25 years, yfinance daily bars. "
            "Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the breakaway shape mark a real reversal? | **No — or at least, we can't "
            f"prove it.** Across **{R['n_loose']}** confirmed breakaways (both directions "
            "pooled) over 61 names and 25 years, no forward horizon (1/5/10/20 days) clears the "
            "statistical bar once you correct for testing four horizons at once. |\n"
            "| How rare is it? | **Very.** {} bullish + {} bearish events in a quarter-century "
            "across 61 stocks — about **once every 35 ticker-years**. The stricter, "
            "textbook-faithful version fires only **5** times, total. |\n"
            "| Does it at least work one direction? | **Only on paper, and only if you don't ask "
            "hard questions.** The bullish side alone *looks* significant at 20 days — until you "
            "notice its first day is a sharp loss, and its best trades cluster on famous "
            "market-crash bottoms rather than a stock-specific signal. |\n"
            "| What about the bearish (short) version? | **Nothing.** Every horizon is "
            "statistically indistinguishable from noise, and three of four point estimates are "
            "*negative* — the opposite of what the pattern claims. |\n\n"
            "> A shape the eye recognizes instantly, a force the tape refuses to obey."
            .format(R["n_bullish"], R["n_bearish"])
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A long candle in the direction of the trend, a gap that opens even further that "
            "way, two more days pressing the move, and then — a long candle slams back through "
            "the gap. That's a breakaway: the last gasp of the old trend, and the start of the "
            "new one.\"*\n\n"
            "It's in every candlestick textbook (Nison's own phrase is that the reversal candle "
            "\"closes within the area of the window\" — it erases the last leg of the run). The "
            "gap is supposed to show conviction; the reversal candle is supposed to show that "
            "conviction was wrong."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this is a free, mechanical early-warning system: spot the shape, take the "
            "reversal trade the next morning, and you're on the right side of a trend change "
            "before most of the market notices. Five-candle reversal patterns are exactly the "
            "kind of thing chart-pattern services sell subscriptions on — so it's worth asking, "
            "plainly: does the shape actually predict anything, or does the eye just like a "
            "clean picture?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **The detector.** A downtrend (or uptrend) into the pattern, a long first candle, "
            "a *clean* gap that stays open, two candles running further, then a long reversal "
            "candle closing back through the gap-day's own high/low. We test the plain reading "
            "**and** a stricter, textbook-faithful cut side by side.\n"
            "- **The comparison.** Not \"did the stock go up\" — but did it beat the **same "
            "trade** on an *ordinary* day already sitting in that same downtrend or uptrend, "
            "shape or no shape.\n"
            "- **The correction.** Four forward windows (1/5/10/20 days) means four chances to "
            "get lucky — so the bar to clear moves up (Bonferroni).\n"
            "- **The symmetry check.** The claim is bidirectional. If it's a real reversal force "
            "and not a story about up-markets drifting up, both the bullish *and* bearish version "
            "should work."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline** — both directions pooled, the pre-registered test."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.combined_experiment(PANEL)\n"
            "    hs = st.HORIZONS\n"
            "    ts = [res['per_horizon'][h]['welch_t'] for h in hs]\n"
            "    deltas = [res['per_horizon'][h]['delta_bps'] for h in hs]\n"
            "else:\n"
            "    hs = (1, 5, 10, 20)\n"
            "    ts = [R['combined'][h]['t'] for h in hs]\n"
            "    deltas = [R['combined'][h]['delta'] for h in hs]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "cols = [RED if abs(t) >= R['bonferroni_crit'] else GREY for t in ts]\n"
            "ax.bar([str(h) for h in hs], ts, color=cols, width=.55)\n"
            "for i, t in enumerate(ts): ax.annotate(f'{t:+.2f}', (i, t), ha='center',\n"
            "    va='bottom' if t >= 0 else 'top')\n"
            "ax.axhline(R['bonferroni_crit'], ls='--', c=RED, lw=1, label='Bonferroni bar (k=4)')\n"
            "ax.axhline(-R['bonferroni_crit'], ls='--', c=RED, lw=1)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('forward horizon (trading days)'); ax.set_ylabel('Welch t vs base rate')\n"
            "ax.set_title('Nothing clears the bar — closest is day 20')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print({h: round(t, 2) for h, t in zip(hs, ts)})"
        ),
        md(
            f"With {R['n_loose']} confirmed breakaways (both directions), the closest any "
            f"horizon gets to significance is day 20 (*t* = {R['combined'][20]['t']:+.2f}) — "
            f"short of the Bonferroni-corrected bar of **{R['bonferroni_crit']:.2f}**. And day 1 "
            f"is actually **negative** (*t* = {R['combined'][1]['t']:+.2f}): on average, the "
            "breakaway trade *loses* money on its very first session.\n\n"
            "**Second, is it at least one-sided?** Split by direction:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rb = st.run_experiment(PANEL, 'bullish')\n"
            "    rr = st.run_experiment(PANEL, 'bearish')\n"
            "    tb = [rb['per_horizon'][h]['welch_t'] for h in hs]\n"
            "    tr = [rr['per_horizon'][h]['welch_t'] for h in hs]\n"
            "else:\n"
            "    tb = [R['bullish'][h]['t'] for h in hs]\n"
            "    tr = [R['bearish'][h]['t'] for h in hs]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "x = np.arange(len(hs)); w = 0.35\n"
            "ax.bar(x - w/2, tb, width=w, color=AMBER, label=f\"bullish (n={R['n_bullish']})\")\n"
            "ax.bar(x + w/2, tr, width=w, color=GREY, label=f\"bearish (n={R['n_bearish']})\")\n"
            "ax.axhline(R['bonferroni_crit'], ls='--', c=RED, lw=1)\n"
            "ax.axhline(-R['bonferroni_crit'], ls='--', c=RED, lw=1)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([str(h) for h in hs])\n"
            "ax.set_xlabel('forward horizon (days)'); ax.set_ylabel('Welch t vs base rate')\n"
            "ax.set_title('Bullish alone nearly clears the bar at 20d; bearish never does')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('bullish', {h: round(t,2) for h,t in zip(hs, tb)})\n"
            "print('bearish', {h: round(t,2) for h,t in zip(hs, tr)})"
        ),
        md(
            f"The bullish side alone reaches *t* = **{R['bullish'][20]['t']:+.2f}** at 20 days — "
            "past the bar! But look at day 1 for that *same* bullish side: "
            f"*t* = **{R['bullish'][1]['t']:+.2f}** — a sharp loss. And the bearish mirror, which "
            "should show the same effect shorting into strength, shows **nothing**: every "
            f"horizon is under \\|1.4\\|, and "
            f"{sum(1 for h in (1, 5, 10, 20) if R['bearish'][h]['delta'] < 0)} "
            "of 4 point estimates are outright negative.\n\n"
            "**Third — is the bullish \"signal\" the candle, or the calendar?**"
        ),
        code(
            "recs = R['best_worst']\n"
            "labels = [f\"{t}\\n{d0}\" for t, d0, d1, r in recs]\n"
            "vals = [r for t, d0, d1, r in recs]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "cols = [RED if v < 0 else GREEN for v in vals]\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "for i, v in enumerate(vals): ax.annotate(f'{v:+.0f} bps', (i, v), ha='center',\n"
            "    va='bottom' if v >= 0 else 'top', fontsize=8)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('20-day forward return (bps)')\n"
            "ax.set_title('Best/worst bullish breakaways -- notice the dates')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "LOW's best trade fires in **July 2002** (the post-dot-com bottom), DIS's fires in "
            "**April 2025** (the tariff-selloff bottom), and CAT's fires weeks after the **2010 "
            "correction low**. These aren't obviously about the *candle* — they're about the "
            "*calendar*: the pattern happened to fire near some of the most famous market-wide "
            "bounces of the last 25 years. That's not a stock-specific reversal signal; that's "
            "riding a handful of historic dip-buying opportunities and calling it a pattern."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The pre-registered, bidirectional test never clears the "
            "corrected bar at any horizon; the one cell that nominally does (bullish, 20 days) "
            "fails its own day-1 reaction and its own \"is it the candle or the calendar\" check.\n"
            "- **Tradability — Mirage.** 44 events over 61 stocks and 25 years is a curiosity, "
            "not a strategy. Costs aren't even the binding constraint — there just aren't enough "
            "trades to matter.\n"
            "- **Works both directions equally? — Busted.** Bullish rides crash-bottom "
            "coincidences without certifying; bearish does nothing at all."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Rarity is the real finding.** Six independent conditions (trend, long candle, "
            "clean gap, two-day run, long reversal candle, crossing back through the gap) almost "
            "never line up together — the pattern's own strictness is what starves it of a "
            "usable sample, long before any economic mechanism gets a chance to show up.\n"
            "- **Sibling studies:** the [island reversal](../../417-island-reversal/) (a two-gap "
            "bracket, no run), [mind the gap](../../74-mind-the-gap/) (does *any* gap fill?) and "
            "the [ladder bottom](../../687-ladder-bottom/) (this study's closest structural "
            "cousin — a no-gap, one-directional five-candle figure) all land in the same place: "
            "chart patterns built from rare, compound conditions rarely survive contact with a "
            "large enough sample.\n\n"
            "*Think the strict cut is onto something? The honest next step is more history or "
            "more names, not a looser detector — five events will never be enough to trust, no "
            "matter how good they look.*"
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
            "# Breakaway Candles — a quantitative teardown 🔬\n"
            "### The loose/strict detector · the trend-matched base rate · Bonferroni across 4 "
            "horizons · the bullish/bearish symmetry check · a label-shuffle placebo · costs · a "
            "synthetic planted-drift control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — **a gap-then-run-then-reversal five-candle shape marks a trend "
            "reversal, in either direction** — is tested mechanically, pooled across sides as "
            "pre-registered, then unpacked by side as the desk's own symmetry myth-check.\n\n"
            "> ⚠️ **Data note.** Daily OHLCV, SPY + 60 long-listed US large-caps (the shared "
            "685/687 basket), ~25.0 years, 383,080 total bars, yfinance, cached; as-of "
            "2026-06-30. **Survivors basket** — named on the Signal axis. Panel fingerprint "
            "`" + R["fp_panel"] + "`, methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | combined (bidirectional) headline: max \\|Welch *t*\\| = "
            f"**{R['combined'][20]['t']:.2f}** at 20d vs Bonferroni bar "
            f"**{R['bonferroni_crit']:.2f}**; day 1 is **{R['combined'][1]['t']:+.2f}** "
            "(negative); strict cut n = {} (untestable) |\n"
            f"| **Tradability** | `MIRAGE` | n = {R['n_loose']} loose events / 61 names / "
            f"{R['years']:.0f} years (~1 per 35 ticker-years); strict n = {R['n_strict']} |\n"
            "| **Symmetric?** | `BUSTED` | bullish 20d *t* = "
            f"**{R['bullish'][20]['t']:+.2f}** but day-1 *t* = **{R['bullish'][1]['t']:+.2f}**; "
            f"bearish max \\|*t*\\| = {max(abs(R['bearish'][h]['t']) for h in (1,5,10,20)):.2f} "
            "(never significant) |\n\n"
            "> 💡 In plain words: the shape is real (you can point at it on a chart); the "
            "forward edge is not (you can't point at a number that survives scrutiny)."
            .format(R["n_strict"])
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let candle $t{-}4$ be a long-bodied trend candle, $t{-}3$ gap cleanly beyond its "
            "extreme, $t{-}3 \\to t{-}2 \\to t{-}1$ make monotonically declining/advancing closes "
            "(the \"run\"), and $t$ be a long-bodied candle in the *opposite* direction closing "
            "back past $t{-}3$'s own high/low (the \"reversal through the gap\"). The claims:\n\n"
            "- **H₁ (reversal, pooled).** The bidirectional reversal-return mean beats the "
            "trend-matched base rate, at a horizon that survives multiple-testing correction.\n"
            "- **H₂ (symmetry).** Both the bullish *and* bearish version work — a real reversal "
            "force, not a drift story.\n"
            "- **H₃ (robustness).** The literature-closer strict cut, if anything, works *better* "
            "than the loose cut (a cleaner shape should carry a cleaner signal).\n\n"
            "We find **H₁ not supported** (no horizon clears the corrected bar), **H₂ busted** "
            "(one side alone nominally clears the bar, the other shows nothing), and **H₃ "
            "untestable** (n = 5, below the desk's own n = 8 floor)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The claim is explicitly **bidirectional**, so the pre-registered primary test pools "
            "bullish (long, downtrend context) and bearish (short, uptrend context) events into "
            "one sample — both already sign-adjusted to \"trade P&L\", so pooling is "
            "apples-to-apples. The decisive statistic is a **Welch *t*** of the reversal-return "
            "mean vs the trend-matched base-rate mean (never a one-sample *t* against zero, which "
            "would just measure the basket's unconditional drift). Four horizons (1/5/10/20 days) "
            "means a **Bonferroni**-corrected critical value (k=4, "
            f"\\|*t*\\| ≥ {R['bonferroni_crit']:.2f}), not a naive 2.0. A **label-shuffle "
            "placebo** (2,000 draws from the pooled base-rate pool) is the honest control for the "
            "basket's own drift. Below **n = 8** pooled events, no *t*-stat is computed at all — "
            "decoration, not evidence."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Basket.** SPY + 60 long-listed large-caps, ~{R['years']:.1f} years, "
            f"{R['n_bars']:,} total bars, yfinance daily OHLCV, as-of 2026-06-30.\n"
            "- **Detector.** Loose cut: 6 conditions (context, long candle 1, clean gap, "
            "monotone 2-day run, long candle 5, reversal through the gap). Strict cut adds a "
            "bigger gap, long bodies (≥55% of range) on candles 1 & 5, and a full gap fill.\n"
            "- **Base rate.** The same directional bet on every bar sharing the trend context, "
            "shape or no shape — isolates the pattern's own information from trend-context mean "
            "reversion.\n"
            "- **Headline.** Combined (pooled) Welch *t* + HAC *t* + Bonferroni(k=4) + "
            "label-shuffle placebo, at 5/10 bps one-way costs.\n"
            "- **Symmetry.** The same machinery run separately per side — the desk's own myth-"
            "check for a bidirectional claim.\n"
            "- **Execution.** One documented lag: confirm at candle 5's close, enter next "
            "session's open.\n"
            "- **Control.** A deterministic, bar-by-bar synthetic panel with forced breakaway "
            "blocks (alternating sides) and a tunable planted post-reversal drift; the null must "
            "not fire across many seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline — combined, pre-registered\n\n"
            "Both directions pooled; Welch *t* of the reversal mean vs the trend-matched base "
            "rate, per horizon, against the Bonferroni-corrected bar."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.combined_experiment(PANEL)\n"
            "    rows = res['per_horizon']\n"
            "else:\n"
            "    rows = {h: dict(ladder=dict(n=R['combined'][h]['n'], mean_bps=R['combined'][h]['mean'],\n"
            "                                win_rate=R['combined'][h]['win']/100, net_bps=R['combined'][h]['net']),\n"
            "                    base=dict(mean_bps=R['combined'][h]['base']),\n"
            "                    delta_bps=R['combined'][h]['delta'], welch_t=R['combined'][h]['t'],\n"
            "                    placebo_p=R['combined'][h]['p'])\n"
            "            for h in st.HORIZONS}\n"
            "for h in st.HORIZONS:\n"
            "    r = rows[h]; s = r['ladder']\n"
            "    print(f\"H={h:>2}  n={s['n']:>3}  mean={s['mean_bps']:+7.1f} bps  \"\n"
            "          f\"base={r['base']['mean_bps']:+6.1f}  delta={r['delta_bps']:+7.1f}  \"\n"
            "          f\"welch_t={r['welch_t']:+.2f}  placebo_p={r['placebo_p']:.3f}\")"
        ),
        code(
            "hs = list(st.HORIZONS)\n"
            "ts = [rows[h]['welch_t'] for h in hs]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "cols = [RED if abs(t) >= R['bonferroni_crit'] else GREY for t in ts]\n"
            "ax.bar([str(h) for h in hs], ts, color=cols, width=.55)\n"
            "for i, t in enumerate(ts): ax.annotate(f'{t:+.2f}', (i, t), ha='center',\n"
            "    va='bottom' if t >= 0 else 'top')\n"
            "ax.axhline(R['bonferroni_crit'], ls='--', c=RED, lw=1)\n"
            "ax.axhline(-R['bonferroni_crit'], ls='--', c=RED, lw=1)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('Welch t (reversal vs base rate)')\n"
            "ax.set_title(f\"Bonferroni bar (k=4) = {R['bonferroni_crit']:.2f} -- nothing clears it\")\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: {R['n_loose']} pooled events, four looks, none survive "
            f"correction. Day 20 (*t* = {R['combined'][20]['t']:+.2f}) is the closest and its own "
            f"placebo *p* = {R['combined'][20]['p']:.3f} is suggestively low — but a single "
            "nominally-suggestive cell out of four pre-registered looks is exactly what the "
            "Bonferroni correction exists to catch."
        ),
        md(
            "### 4b · The strict cut — too rare to test\n\n"
            "The literature-closer cut (bigger gap, genuinely long candles 1 & 5, a full gap "
            "fill) is stricter by design — and correspondingly rarer."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ev = st.pool_events(PANEL, 'bullish')\n"
            "    ev_r = st.pool_events(PANEL, 'bearish')\n"
            "    import pandas as pd\n"
            "    both = pd.concat([ev, ev_r], ignore_index=True) if len(ev) or len(ev_r) else pd.DataFrame()\n"
            "    n_strict = int(both['strict'].sum()) if len(both) else 0\n"
            "else:\n"
            "    n_strict = R['n_strict']\n"
            "print(f'strict-cut events basket-wide: {n_strict}  (desk floor for a t-stat: n=8)')\n"
            "for h in st.HORIZONS:\n"
            "    sr = R['strict'][h]\n"
            "    print(f\"  H={h:>2}  n={sr['n']}  mean={sr['mean']:+7.1f} bps  \"\n"
            "          f\"delta={sr['delta']:+7.1f}  placebo_p={sr['p']:.3f}  (t-stat: n/a, n<8)\")"
        ),
        md(
            "> 💡 In plain words: five events in a quarter-century, basket-wide. A 20-day point "
            f"estimate of {R['strict'][20]['delta']:+.0f} bps on five trades is a coin flip's "
            "worth of anecdotes dressed up as a finding — the desk's own floor exists precisely "
            "to stop a number like this from being reported as if it meant something."
        ),
        md(
            "### 4c · The symmetry check — does it work both ways?\n\n"
            "A real bidirectional reversal figure should show a comparable effect on both sides. "
            "Run the identical machinery per side:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rb = st.run_experiment(PANEL, 'bullish'); rr = st.run_experiment(PANEL, 'bearish')\n"
            "    tb = [rb['per_horizon'][h]['welch_t'] for h in st.HORIZONS]\n"
            "    tr = [rr['per_horizon'][h]['welch_t'] for h in st.HORIZONS]\n"
            "else:\n"
            "    tb = [R['bullish'][h]['t'] for h in st.HORIZONS]\n"
            "    tr = [R['bearish'][h]['t'] for h in st.HORIZONS]\n"
            "hs = list(st.HORIZONS)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "x = np.arange(len(hs)); w = .35\n"
            "ax.bar(x - w/2, tb, width=w, color=AMBER, label=f\"bullish (n={R['n_bullish']})\")\n"
            "ax.bar(x + w/2, tr, width=w, color=GREY, label=f\"bearish (n={R['n_bearish']})\")\n"
            "ax.axhline(R['bonferroni_crit'], ls='--', c=RED, lw=1)\n"
            "ax.axhline(-R['bonferroni_crit'], ls='--', c=RED, lw=1)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([str(h) for h in hs])\n"
            "ax.set_ylabel('Welch t vs base rate'); ax.legend()\n"
            "ax.set_title('Bullish edges toward significance; bearish never does')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('bullish', {h: round(t,2) for h,t in zip(hs, tb)})\n"
            "print('bearish', {h: round(t,2) for h,t in zip(hs, tr)})"
        ),
        md(
            f"> 💡 In plain words: bullish 20d *t* = **{R['bullish'][20]['t']:+.2f}** clears both "
            f"the naive and the Bonferroni bar — but its own day-1 reaction is "
            f"*t* = **{R['bullish'][1]['t']:+.2f}**, a sharp loss, which is a strange first act "
            "for a \"the trend is over\" signal. The bearish mirror never exceeds "
            f"\\|{max(abs(R['bearish'][h]['t']) for h in (1,5,10,20)):.2f}\\| at any horizon, with "
            "negative point estimates at 3 of 4 — the pattern that's supposed to be symmetric "
            "isn't."
        ),
        md(
            "### 4d · Is the bullish edge the candle, or the calendar?\n\n"
            "Best/worst confirmed bullish breakaways, 20-day forward:"
        ),
        code(
            "recs = R['best_worst']\n"
            "for tkr, d0, d1, r20 in recs:\n"
            "    print(f'  {tkr}: block {d0} -> {d1}   20d {r20:+.1f} bps')\n"
            "labels = [f'{t}\\n{d0}' for t, d0, d1, r in recs]\n"
            "vals = [r for t, d0, d1, r in recs]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "cols = [RED if v < 0 else GREEN for v in vals]\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('20d forward (bps)')\n"
            "ax.set_title('3 of 5 extreme events sit on famous market-wide bottoms')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "> 💡 In plain words: LOW (2002-07), DIS (2025-04) and CAT (weeks after the 2010 "
            "correction low) sit on or near famous market-wide bounces — not obviously a "
            "name-specific candle signal. The same confound sibling study 687-ladder-bottom "
            "documented on its own strict cut."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Deterministic, bar-by-bar synthetic panel: forced 5-bar breakaway blocks "
            "(alternating bullish/bearish), TUNABLE planted post-reversal drift. The null "
            f"(edge = 0) is checked over **{R['syn_null_seeds']} seeds** — never a single stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    d, _truth = data.synthetic_panel(edge=0.0, seed=692 + s_)\n"
            "    r = st.synthetic_detect_combined(d, horizon=20, seed=692 + s_)\n"
            "    if r['welch_t'] is not None:\n"
            "        null_ts.append(r['welch_t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "d, _truth = data.synthetic_panel(edge=0.02, seed=692)\n"
            "planted_02 = st.synthetic_detect_combined(d, horizon=20, seed=692)['welch_t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(len(null_ts)) + np.linspace(-.12, .12, len(null_ts)), null_ts,\n"
            "           color=GREY, s=40, label='null worlds (edge=0), 20 seeds')\n"
            "ax.scatter([1], [planted_02], color=RED, s=90, zorder=5, label='planted edge=0.02')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Welch t (reversal vs base rate)')\n"
            "ax.set_title('The detector recovers a planted drift; the null is mostly quiet')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/{len(null_ts)} seeds  |  '\n"
            "      f'planted (edge=0.02) t = {planted_02:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"*t* = **{R['syn_null_mean']:+.2f}** (sd {R['syn_null_sd']:.2f}) and fires "
            f"\\|*t*\\| ≥ 2 in **{R['syn_null_fire']}/{R['syn_null_seeds']}** seeds — not perfectly "
            "quiet, a modestly elevated false-positive rate we attribute to the deliberately rare "
            "event counts even a 40-name/3,500-day synthetic panel produces, and we say so "
            "plainly rather than smooth it over. A planted drift is recovered cleanly "
            f"(*t* = {R['syn_edge02_t']:+.2f} at edge=0.02, {R['syn_edge04_t']:+.2f} at "
            "edge=0.04). *(A faithful-engine / power check only — never cited in support of the "
            "real-tape stamp, and this study's real-tape stamp is `NONE` regardless.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the pre-registered, bidirectional headline never clears the "
            f"Bonferroni bar (\\|*t*\\| ≥ {R['bonferroni_crit']:.2f}) at any of 4 horizons "
            f"(best: {R['combined'][20]['t']:+.2f} at 20d); day 1 is "
            f"**{R['combined'][1]['t']:+.2f}** (a negative whipsaw); the strict cut (n="
            f"{R['n_strict']}) is untestable.\n"
            f"- **Tradability `MIRAGE`** — {R['n_loose']} events over {R['n_names']} names and "
            f"{R['years']:.0f} years (~1 per 35 ticker-years); costs are not the binding "
            "constraint, event frequency is.\n"
            f"- **Symmetric? `BUSTED`** — bullish 20d nominally clears the bar "
            f"(*t* = {R['bullish'][20]['t']:+.2f}) but fails its own day-1 and "
            "candle-vs-calendar checks; bearish shows nothing at any horizon."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson is combinatorial rarity.** Six conditions compounding — "
            "context, a long candle, a clean gap, a monotone run, another long candle, a "
            "cross-back — is why five-candle reversal figures across this desk (breakaway, "
            "ladder bottom, island reversal, three methods) all struggle for sample size long "
            "before any economic story gets tested.\n"
            "- **Where a believer goes next:** more names (international large-caps), a longer "
            "tape (pre-2001), or accepting the loose cut as the only testable version and "
            "reporting its honest `NONE` rather than reaching for the strict cut's n=5 headline.\n"
            "- **Dedup map:** [417-island-reversal](../../417-island-reversal/) (two-gap "
            "bracket, no run), [74-mind-the-gap](../../74-mind-the-gap/) (any single gap "
            "filling), [455-three-methods](../../455-three-methods/) (a continuation pause, "
            "no gap), [687-ladder-bottom](../../687-ladder-bottom/) (this study's structural "
            "cousin — the loose/strict/Bonferroni idiom reused directly, on a no-gap, "
            "one-directional shape).\n\n"
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
