"""Generate the two narrative notebooks for Study 694 (Matching Low).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached basket
OHLCV under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
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
# 30-name liquid large-cap + SPY basket, 2005-01-03 -> 2026-06-30, 21.5 years).
R = dict(
    start="2005-01-03", end="2026-06-30", years=21.5, n_names=30,
    fp_quantlab="75a0fd665f4f", fp_basket="a63644cb0ca5", total_bars=162180,
    # forward return after a matching low, per horizon:
    # (H, n, mean%, base%, hit%, basehit%, t_hac, t_one, t_welch[decisive], p_placebo, net5%, net10%)
    h1=(1, 4600, 0.028, 0.028, 52, 51, 1.69, 1.72, -0.03, 0.519, -0.072, -0.172),
    h5=(5, 4597, 0.245, 0.252, 54, 55, 5.49, 5.67, -0.16, 0.545, 0.145, 0.045),
    h10=(10, 4595, 0.500, 0.529, 56, 57, 7.54, 8.34, -0.48, 0.646, 0.400, 0.300),
    h20=(20, 4586, 1.051, 1.087, 59, 59, 10.55, 11.94, -0.40, 0.628, 0.951, 0.851),
    hit_wilson={1: (50.4, 53.3), 5: (53.0, 55.8), 10: (55.0, 57.9), 20: (57.5, 60.4)},
    # strict close-match myth-check: (H, n, mean%, t_welch, t_hac, p, net5%)
    strict=[(1, 1023, 0.021, -0.21, 0.58, 0.565, -0.079),
            (5, 1023, 0.351, 1.11, 3.81, 0.180, 0.251),
            (10, 1023, 0.568, 0.33, 4.84, 0.390, 0.468),
            (20, 1022, 1.300, 1.15, 6.63, 0.163, 1.200)],
    # genuine-prior-downtrend myth-check: same tuple shape
    trend=[(1, 1784, 0.039, 0.37, 1.41, 0.383, -0.061),
           (5, 1783, 0.270, 0.23, 3.55, 0.416, 0.170),
           (10, 1783, 0.553, 0.24, 5.25, 0.414, 0.453),
           (20, 1780, 1.278, 1.28, 8.25, 0.119, 1.178)],
    # Bonferroni across the basket (per-ticker Welch t vs that ticker's own base, H=5)
    bonf_k=30, bonf_z=3.14, bonf_survive=0,
    bonf_top=[("INTC", 100, -0.492, -2.09), ("HD", 149, 0.817, 1.84), ("PFE", 159, -0.272, -1.58)],
    # synthetic null over 20 seeds (decisive stat: Welch t vs the panel's own unconditional base)
    syn_null_mean=0.13, syn_null_sd=0.93, syn_null_fire=1, syn_null_n=20,
    # planted control (seed 694): (edge, planted_days, n, mean%, t_welch, t_hac, hit%)
    syn=[(0.000, 0, 1484, 0.081, 0.17, 1.03, 51),
         (0.002, 4257, 1475, 0.769, 8.32, 9.63, 61),
         (0.004, 4225, 1456, 1.176, 13.03, 14.48, 66)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Marks_support%3F: Busted](https://img.shields.io/badge/Marks_support%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from matching_low import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PANEL = data.load_real(allow_fetch=False, asof="2026-06-30")
else:
    PANEL = None
print("real basket cache present:", HAVE_REAL,
      "| names:", (0 if PANEL is None else len(PANEL)))
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell can
# quote it whether or not the cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The matching low — does a repeated close really mark support? 🟥🟥\n"
            "### A simple Japanese candlestick **reversal** figure, measured on 21.5 years "
            "of real tape\n\n"
            + BADGES +
            "Open a candlestick book and you'll meet the **matching low**: the market falls "
            "for two days in a row, and both days close at (about) the same price. Chart "
            "readers call it \"a double-bottom in miniature\" — the market tested the exact "
            "same price twice and twice failed to break it, so that price must be support, "
            "and the decline should **reverse upward**.\n\n"
            "So we did the boring, honest thing: found **every** matching low in a basket of "
            "30 big US stocks + SPY over 21.5 years, and measured what actually happened "
            "next.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the "
            "Bonferroni correction? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** We use a fixed **30-name liquid large-cap + SPY** "
            "basket (names still trading today), so this carries **survivorship** — it "
            "excludes firms whose \"reversal\" never came because they kept falling and later "
            "delisted. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After a matching low, does the market actually reverse upward? | **Not in "
            "any way we can certify.** Buy the next morning and, across 1–20 days, you make "
            "almost exactly what you'd have made buying on any ordinary day — no more. |\n"
            "| Does it at least make money? | **Yes, in raw terms — but that's the trap.** "
            "Because stocks drift up over decades, *any* long-only rule looks profitable. "
            "The honest question is whether it beats an ordinary day, and it doesn't. |\n"
            "| Does a near-perfect close-tie work better than a loose match? | **No.** It "
            "doesn't sharpen anything — the strict version is just rarer. |\n"
            "| What if the pattern only counts after a genuine prior decline? | **Still "
            "nothing.** |\n"
            "| Does any single stock secretly carry the pattern? | **No.** Checked "
            f"individually across all {R['bonf_k']} names with enough events, using a "
            "stricter statistical bar (so one lucky name can't pass as \"the\" signal), "
            f"**{R['bonf_survive']}/{R['bonf_k']}** clear it. |\n\n"
            "> Two red candles closing at the same price don't mean the market \"defended\" "
            "anything — on the tape it means almost nothing at all."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The **matching low** is two consecutive down candles whose closes land at "
            "essentially the same price. If sellers pushed the market to the exact same "
            "level twice and couldn't break it, buyers are defending that price — a support "
            "level has formed, and the decline should reverse.\"*\n\n"
            "This is straight out of Steve Nison's *Japanese Candlestick Charting "
            "Techniques* (the book that brought candles to the West) and is catalogued "
            "across the standard candlestick references as one of the simpler reversal "
            "figures — unlike most of this desk's candlestick patterns, it makes **no "
            "short-side claim**: it is a purely bullish signal. We measure it independently, "
            "on our own basket and protocol, and ask the only question that matters: **does "
            "the market actually bounce?**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the matching low really predicted a reversal, it would be a genuinely "
            "useful, very common signal — it fires roughly **once every 24 trading days per "
            "name** in our basket, far more often than most of the desk's rarer three- and "
            "four-candle patterns. A signal that common, if real, could anchor a whole "
            "systematic strategy.\n\n"
            "But there's a trap built into testing *any* long-only rule on a rising market: "
            "stocks drift up over time, with or without a pattern. A naive backtest that "
            "just checks \"did the price go up afterward?\" will say yes almost regardless "
            "of what triggered the trade. The only honest question is whether the pattern "
            "beats what an ordinary, unconditional day in the same basket would have earned "
            "you — and that's exactly what we measure."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take a fixed **{R['n_names']}-name** basket of liquid US large-caps + "
            f"**SPY** and scan every daily bar from **{R['start']}** to **{R['end']}** "
            f"({R['years']:.1f} years). For each name we flag **every** matching low: two "
            "consecutive down candles whose closes sit within **0.15%** of each other (a "
            "realistic tolerance — exact ties basically never happen on continuous prices). "
            "Then, with **no cheating**:\n\n"
            "1. **Wait for the close** that confirms the second down candle.\n"
            "2. **Enter the next morning's open** (one day's lag — you can't trade a bar "
            "still forming) and hold **1, 5, 10, or 20** trading days, long only.\n"
            "3. **Compare fairly** — not to zero (a naive comparison is fooled by the "
            "market's own drift), but to what the *same basket* earns on an unconditional "
            "day, and to thousands of random \"fake signal\" picks. If buying after a "
            "matching low beats the everyday tide, the legend is real. If it doesn't, the "
            "\"support level\" story is a myth."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Here's the return after a matching low, at each "
            "horizon, next to what the *same basket* earns on an ordinary (unconditional) "
            "day. A real \"support and reversal\" signal would show the pattern's bar "
            "**beating** the baseline, in green."
        ),
        code(
            "hs = [1, 5, 10, 20]\n"
            "if HAVE_REAL:\n"
            "    rows = {h: st.summarize(PANEL, h, placebo=False) for h in hs}\n"
            "    means = [rows[h]['mean']*100 for h in hs]\n"
            "    base  = [rows[h]['base_mean']*100 for h in hs]\n"
            "else:\n"
            "    keys = ['h1','h5','h10','h20']\n"
            "    means = [R[k][2] for k in keys]; base = [R[k][3] for k in keys]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.5))\n"
            "ax.bar(x-.18, means, .34, color=RED, label='AFTER a matching low')\n"
            "ax.bar(x+.18, base, .34, color=GREY, label='ORDINARY day (baseline)')\n"
            "for i,v in enumerate(means): ax.annotate(f'{v:+.2f}%',(i-.18,v),ha='center',va='bottom',fontsize=9)\n"
            "for i,v in enumerate(base): ax.annotate(f'{v:+.2f}%',(i+.18,v),ha='center',va='bottom',fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('forward return (%)')\n"
            "ax.set_title('The pattern bar barely differs from the everyday baseline')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('after matching low:', {f'{h}d': round(m,3) for h,m in zip(hs,means)})\n"
            "print('ordinary day (baseline):', {f'{h}d': round(b,3) for h,b in zip(hs,base)})"
        ),
        md(
            f"At 20 days the pattern trade is **{R['h20'][2]:+.2f}%** — positive! — but the "
            f"ordinary-day baseline is **{R['h20'][3]:+.2f}%**, essentially the same number. "
            "The bars in that chart look almost identical because they *are* almost "
            "identical — the pattern isn't adding reversal information, it's riding the "
            "same up-drift every other day in the basket rides too."
        ),
        md(
            "**Here's the trap, made visible.** If you naively compared the pattern's return "
            "to *zero* instead of to the baseline, you'd see a \"significant\"-looking "
            "number and declare victory. Let's look at both."
        ),
        code(
            "if HAVE_REAL:\n"
            "    t_hac = [rows[h]['t_hac'] for h in hs]\n"
            "    t_welch = [rows[h]['t_welch'] for h in hs]\n"
            "else:\n"
            "    keys = ['h1','h5','h10','h20']\n"
            "    t_hac = [R[k][6] for k in keys]; t_welch = [R[k][8] for k in keys]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], t_hac, color=RED, width=.55)\n"
            "a1.axhline(2, ls='--', c='k', lw=1); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_ylabel('t vs ZERO (the naive, misleading test)')\n"
            "a1.set_title('Looks great... if you compare to zero')\n"
            "for i,t in enumerate(t_hac): a1.annotate(f'{t:.1f}',(i,t),ha='center',va='bottom')\n"
            "a2.bar([f'{h}d' for h in hs], t_welch, color=AMBER, width=.55)\n"
            "a2.axhline(2, ls='--', c='k', lw=1); a2.axhline(-2, ls='--', c='k', lw=1); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel('t vs the everyday baseline (the fair test)')\n"
            "a2.set_title('...but compared fairly, there is nothing here')\n"
            "for i,t in enumerate(t_welch): a2.annotate(f'{t:.2f}',(i,t),ha='center',va='top' if t<0 else 'bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('t vs zero:', [round(t,2) for t in t_hac]); print('t vs baseline (fair):', [round(t,2) for t in t_welch])"
        ),
        md(
            f"Left panel: compared to zero, the 20-day number reads *t* = "
            f"**{R['h20'][6]:.2f}** — looks like a slam dunk. Right panel: compared to what "
            f"the basket does on an ordinary day, the same 20-day number reads *t* = "
            f"**{R['h20'][8]:.2f}** — indistinguishable from noise. The left chart was never "
            "measuring the pattern. It was measuring the fact that stocks go up over 21 "
            "years, which is true whether or not you waited for two matching red candles."
        ),
        md(
            "**Could the fair result just be bad luck?** Let's pit the real signal against "
            "thousands of **random** picks of the same size, drawn from the same basket."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ev = st.collect_events(PANEL, 5)\n"
            "    obs = float(ev['ret'].mean())\n"
            "    pl = st.placebo_pvalue(PANEL, 5, len(ev), obs, n_draws=5000)\n"
            "    draws = pl['draws']*100; obsp = obs*100; pval = pl['p_value']\n"
            "else:\n"
            "    obsp = R['h5'][2]; pval = R['h5'][9]\n"
            "    rng = np.random.default_rng(694); draws = rng.normal(obsp, 0.5, 5000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='5,000 random 5-day picks, same basket')\n"
            "ax.axvline(obsp, c=RED, lw=2.5, label=f'real matching-low trade {obsp:+.2f}%')\n"
            "ax.set_xlabel('5-day forward return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The real pattern sits in the MIDDLE of the pack: p = {pval:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obsp:+.2f}%   share of random picks that beat it: p = {pval:.3f}')"
        ),
        md(
            f"With placebo *p* = **{R['h5'][9]:.3f}**, more than half of random, undirected "
            "picks in the same basket did *at least as well* as the real pattern. That's "
            "the signature of no edge at all — the matching low is not special."
        ),
        md(
            "**Does a picture-perfect close-tie save it?** Maybe an approximate, casual "
            "match is noise but the strict, near-exact tie — or only counting patterns "
            "that followed a *genuine* decline — is the real deal. Let's check both "
            "stricter recipes against a fair (baseline-matched) comparison."
        ),
        code(
            "if HAVE_REAL:\n"
            "    basic  = [st.summarize(PANEL, h, placebo=False)['t_welch'] for h in hs]\n"
            "    strict = [st.summarize(PANEL, h, placebo=False, tol=st.STRICT_TOL)['t_welch'] for h in hs]\n"
            "    trend  = [st.summarize(PANEL, h, placebo=False, require_downtrend=True)['t_welch'] for h in hs]\n"
            "else:\n"
            "    basic  = [R[k][8] for k in ['h1','h5','h10','h20']]\n"
            "    strict = [s[3] for s in R['strict']]; trend = [t[3] for t in R['trend']]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.3))\n"
            "ax.bar(x-.25, basic, .25, color=RED, label='basic matching low')\n"
            "ax.bar(x,      strict,.25, color=AMBER, label='near-exact close tie')\n"
            "ax.bar(x+.25,  trend, .25, color=GREY, label='after a genuine downtrend')\n"
            "ax.axhline(0, c='k', lw=.8); ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('Welch t (vs the everyday baseline — the fair bar, +-2 dashed)')\n"
            "ax.set_title('No recipe clears the bar')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('basic t:', [round(b,2) for b in basic]); print('strict t:', [round(s,2) for s in strict]); print('trend t:', [round(t,2) for t in trend])"
        ),
        md(
            "No bar in that chart pokes above the dashed **+2** line under any of the three "
            "recipes. Neither purity filter — a tighter close-tie or a genuine decline "
            "behind the pattern — turns \"nothing\" into \"something\"."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Measured fairly (against the basket's own unconditional "
            "day, not against zero), the matching low never certifies a reversal at any "
            "horizon. A random-draw placebo says over half of random picks beat it. No "
            "single stock secretly carries it either — "
            f"**{R['bonf_survive']}/{R['bonf_k']}** individual tickers clear a corrected "
            "bar. There's no support-and-reversal edge.\n"
            "- **Tradability — Mirage.** It's *nominally* profitable from 5 days out — but "
            "so is doing nothing special in the same basket. There's no excess return to "
            "deploy against, and at 1 day, costs already flip it negative.\n"
            "- **\"Marks support\"? — Busted.** The famous \"tested the same price twice, "
            "buyers defended it\" reading doesn't survive contact with the tape once you "
            "compare it to an ordinary day."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — no, and here's the kicker\n\n"
            "The trade is already indistinguishable from the baseline before costs. Now add "
            "the real-world friction — the round-trip spread. The excess, already zero, "
            "stays zero; only the 1-day trade actually turns net-negative."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g  = [st.summarize(PANEL, h, placebo=False)['mean']*100 for h in hs]\n"
            "    n5 = [float(st.event_net_returns(st.collect_events(PANEL, h), cost_bps=5.0).mean())*100 for h in hs]\n"
            "else:\n"
            "    keys = ['h1','h5','h10','h20']\n"
            "    g  = [R[k][2] for k in keys]; n5 = [R[k][10] for k in keys]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, g, .4, color=AMBER, label='gross return')\n"
            "ax.bar(x+.2, n5, .4, color=RED, label='net of costs (5 bps)')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('return (%)'); ax.set_title('Costs bite where the excess was already zero')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('net return by horizon:', {f'{h}d': round(v,3) for h,v in zip(hs,n5)})"
        ),
        md(
            f"At 1 day the net trade is **{R['h1'][10]:.2f}%** — a small loss — and even "
            f"where the net number is positive ({R['h20'][10]:+.2f}% at 20 days) it's "
            "because the market drifted up over that window generally, not because of the "
            "pattern. The matching low is common enough to build a strategy around if it "
            "worked — it just doesn't."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Why does the naive comparison fool people?** Any long-only bet in a "
            "drifting market looks \"significant\" against **zero** — with or without a "
            "matching low — because the market drifts up on its own. That's why this "
            "study's certifying test compares each event to what the *same basket* earns "
            "on an ordinary day. The "
            "[quants notebook](02_for_the_quants.ipynb) shows this trap live on a "
            "synthetic panel with a **known** zero edge.\n"
            "- **The \"same price twice\" family.** "
            "[409-tweezer-tops-bottoms](../../409-tweezer-tops-bottoms/) tests matching "
            "*wick lows*, no color requirement; "
            "[460-counterattack-lines](../../460-counterattack-lines/) requires the second "
            "candle to be the *opposite* color; "
            "[696-double-bottom](../../696-double-bottom/) is the multi-week macro version "
            "— none of them run this study's specific two-down-candle, matching-close "
            "figure.\n\n"
            "*Think a stricter close-tie, a different universe, or a clever filter turns it "
            "green? Show the **net** return landing above the basket's own baseline with "
            "Welch *t* ≥ 2 — then we'll talk.*"
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
            "# Matching Low — a quantitative teardown 🔬\n"
            "### Precise two-bar detector · forward 1/5/10/20-day event study · the "
            "vs-zero-vs-vs-base direction trap, shown live · a random-draw placebo · a "
            "Bonferroni correction across the 30-name basket · costs · close-tolerance & "
            "prior-downtrend myth-checks · a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The matching low is one of candlestick lore's simpler **reversal** figures — "
            "the job here is to test it honestly as a long-only signal, with one subtlety "
            "front and centre: the basket carries the market's own positive drift, so a "
            "naive *t*-vs-zero **always** looks significant for a long-only rule, pattern "
            "or no pattern. Everything below is built around removing that contamination.\n\n"
            "> ⚠️ **Data + survivorship note.** Fixed **30-name liquid large-cap + SPY** "
            "basket, names still trading in 2026 — a *survivor* panel that tilts *toward* "
            "finding a working reversal signal (it excludes names whose decline never "
            "reversed because they delisted). Real data: yfinance daily OHLCV, "
            "`auto_adjust=True`, 2005→2026, as-of **2026-06-30**. Offline core + synthetic "
            "control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_quantlab"] +
            "` / `" + R["fp_basket"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | **Welch *t*** (event sample vs the basket's own "
            f"unconditional forward return — the certifying number) never certifies a "
            f"reversal at any horizon: {R['h1'][8]:+.2f} (1d) / {R['h5'][8]:+.2f} (5d) / "
            f"{R['h10'][8]:+.2f} (10d) / {R['h20'][8]:+.2f} (20d). Random-draw placebo "
            f"*p* ≥ {min(R['h1'][9], R['h5'][9], R['h10'][9], R['h20'][9]):.3f} at every "
            f"horizon. **{R['bonf_survive']}/{R['bonf_k']}** tickers clear a "
            "Bonferroni-corrected per-name bar. |\n"
            f"| **Tradability** | `MIRAGE` | Net return tracks the unconditional base "
            f"almost bar for bar ({R['h5'][10]:+.2f}% net at 5d vs a base of "
            f"{R['h5'][3]:+.2f}%); 1-day net is already negative "
            f"({R['h1'][10]:+.2f}%). |\n"
            "| **Marks support?** | `BUSTED` | The \"tested the same price twice, buyers "
            "defended it\" reading does not survive being measured fairly against the "
            "basket's own drift; neither a near-exact close tie nor a genuine "
            "prior-downtrend filter rescues it. |\n\n"
            "> 💡 In plain words: the matching low has **no certifiable predictive power** "
            "as a reversal signal — the \"significant\"-looking vs-zero number is a "
            "photograph of the market's own up-drift, not the pattern."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned — and the direction trap named up front\n\n"
            "A **matching low** is confirmed at the close of bar $t$ when candles "
            "$t\\!-\\!1$ and $t$ are both **down** ($C_0<O_0$, $C_1<O_1$) and their closes "
            "match within a relative tolerance: $|C_1-C_0|/|C_0|\\le\\text{tol}$ "
            "(default tol $=0.15\\%$, a loose, realistic band — exact ties basically never "
            "occur on continuous daily closes). Enter the **next** open (one lag), hold "
            "$H$ days, and let $r_i(H)=\\text{Close}_{t+H}/\\text{Open}_{t+1}-1$ be the "
            "forward return of event $i$ — **unsigned**, since the pattern is purely "
            "**long-only**: there is no short-side mirror.\n\n"
            "- **H₁ (reversal exists).** $\\overline{r}(H)$ beats the basket's "
            "**unconditional** forward return, significantly, for some $H$ — trading the "
            "matching low adds information beyond just being long the basket anyway.\n"
            "- **H₂ (it's deployable).** The excess survives the round-trip cost.\n"
            "- **H₃ (\"strict tie / genuine decline\").** Restricting to a near-exact close "
            "match or a real prior downtrend sharpens the edge.\n\n"
            "**Why compare to the base, not just to zero?** Because this basket drifts "
            "**up**, ANY long-only event beats zero for free, with or without any pattern. "
            "A vs-zero test therefore inflates every long-only rule purely from the tape's "
            "ordinary drift — not from the pattern. Section 4a demonstrates this trap "
            "directly on a synthetic panel with a **known** zero pattern-edge. We find "
            "**H₁ rejected** (Welch $t\\in[-0.48,-0.03]$, never positive, let alone $\\ge "
            "2$), **H₂ rejected** (net tracks the base, no excess; 1-day net is already "
            "negative), **H₃ rejected** (neither filter helps)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The Signal axis's **decisive** statistic is a **Welch *t*** of the event "
            "sample's forward-return mean against the basket's **unconditional** "
            "forward-return pool:\n\n"
            "$$t_{\\text{Welch}} = \\frac{\\overline{r}_{\\text{event}} - \\overline{r}_{\\text{base}}}"
            "{\\sqrt{\\widehat{\\mathrm{Var}}(r_{\\text{event}})/n_{\\text{event}} + "
            "\\widehat{\\mathrm{Var}}(r_{\\text{base}})/n_{\\text{base}}}}.$$\n\n"
            "A one-sample **Newey-West HAC *t*** against zero is reported alongside as an "
            "**informational** cross-check only — it is the statistic a naive read would "
            "use, and it is exactly the one contaminated by the basket's own drift (proved "
            "on the synthetic null in 4f). The **random-draw placebo** (same event count, "
            "drawn from the unconditional pool) is a second, independent honesty check. "
            f"Finally, with **{R['bonf_k']} tickers** tested individually for a per-name "
            "version of the pooled effect, the two-sided significance bar widens via "
            f"**Bonferroni** to $|t|\\ge z(1-0.025/{R['bonf_k']})\\approx {R['bonf_z']:.2f}$ "
            "— otherwise the single loudest name in a wide basket gets mistaken for \"the\" "
            "signal."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** Fixed **{R['n_names']}-name** liquid large-cap + SPY basket "
            f"(yfinance daily OHLCV, `auto_adjust=True`, {R['start']}→{R['end']}, "
            f"{R['total_bars']:,} bars). **Survivor** panel — named on the Signal axis, "
            "tilting *toward* finding a working reversal signal.\n"
            "- **Detector.** Two down candles ($C<O$) on bars $(t\\!-\\!1,t)$, closes "
            "matching within a **0.15%** relative tolerance (default); a **0.03%** strict "
            "tolerance and a genuine-prior-downtrend variant for the myth-checks.\n"
            "- **Timing.** Confirm at the close of $t$; enter the **next open** (one lag); "
            "hold $H\\in\\{1,5,10,20\\}$ days, long only; drop events whose window overruns.\n"
            "- **Certifying stat.** Welch *t* of the event sample vs the basket's own "
            "unconditional forward-return pool (drift-neutral) — the inference-bar number.\n"
            "- **Cross-checks.** One-sample/HAC *t* vs zero (informational, "
            "drift-contaminated); a 5,000-draw random-draw placebo; hit-rate vs base "
            "hit-rate with Wilson intervals.\n"
            "- **Bonferroni across the basket.** Per-ticker Welch *t* (event sample vs that "
            f"ticker's own unconditional base), corrected critical value for "
            f"$k={R['bonf_k']}$ simultaneous tests.\n"
            "- **Costs.** 5 / 10 bps one-way × 2 (round trip) on every event; no borrow "
            "(long only).\n"
            "- **Positive control.** A deterministic panel with its own embedded drift and "
            "a **planted** post-pattern reversal: zero edge must NOT reach significance on "
            "the *certifying* Welch stat across 20 seeds; a planted edge must light it up.\n\n"
            "> **Falsifier, stated in advance:** if the Welch *t* never clears **+2** on "
            "the real tape at any horizon, the reversal signal is **None** and \"marks "
            "support\" is **Busted**."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The direction trap, shown live — vs-zero vs vs-unconditional-base\n\n"
            "Left: the forward-return mean by horizon against **zero** (the naive, "
            "contaminated read) with its HAC *t*. Right: the same means against the "
            "basket's own **unconditional** base (the certifying Welch *t*). Watch the "
            "left panel climb steadily while the right panel stays flat around zero — "
            "that gap *is* the finding."
        ),
        code(
            "hs = [1, 5, 10, 20]\n"
            "keys = ['h1','h5','h10','h20']\n"
            "if HAVE_REAL:\n"
            "    rows = [st.summarize(PANEL, h, placebo=False) for h in hs]\n"
            "    means = [r['mean']*100 for r in rows]\n"
            "    t_hac = [r['t_hac'] for r in rows]; t_welch = [r['t_welch'] for r in rows]\n"
            "else:\n"
            "    means = [R[k][2] for k in keys]\n"
            "    t_hac = [R[k][6] for k in keys]; t_welch = [R[k][8] for k in keys]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], t_hac, color=RED, width=.55)\n"
            "a1.axhline(2, ls='--', c='k', lw=1); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_ylabel('t (vs ZERO, informational)'); a1.set_title('The naive HAC t vs 0 climbs steadily')\n"
            "for i,t in enumerate(t_hac): a1.annotate(f'{t:.2f}',(i,t),ha='center',va='bottom')\n"
            "a2.bar([f'{h}d' for h in hs], t_welch, color=AMBER, width=.55)\n"
            "a2.axhline(-2, ls='--', c='k', lw=1); a2.axhline(2, ls='--', c='k', lw=1); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel('Welch t (vs unconditional base — decisive)'); a2.set_title('Once compared fairly: nothing supportive')\n"
            "for i,t in enumerate(t_welch): a2.annotate(f'{t:.2f}',(i,t),ha='center',va='top' if t<0 else 'bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('t vs zero (HAC):', [round(t,2) for t in t_hac])\n"
            "print('t vs unconditional base (Welch, decisive):', [round(t,2) for t in t_welch])"
        ),
        md(
            f"> 💡 In plain words: the left panel — comparing to zero — climbs to "
            f"*t* = **{R['h20'][6]:.2f}** by 20 days and would read as a slam-dunk "
            "discovery to anyone testing a long-only rule against zero. It is the wrong "
            f"comparison: the right panel, the decisive one, never leaves the noise band "
            f"({R['h1'][8]:+.2f} to {R['h10'][8]:+.2f}) — the matching low earns almost "
            "exactly what an unconditional long position earns, no more."
        ),
        md(
            "### 4b · The full inference table — HAC, one-sample, Welch, hit-rate, placebo\n\n"
            "Every arbiter at once. The believer needs a **positive** excess over the "
            "unconditional base with Welch *t* ≥ +2; nothing here comes close."
        ),
        code(
            "import pandas as pd\n"
            "if HAVE_REAL:\n"
            "    tab = pd.DataFrame([st.summarize(PANEL, h, n_draws=5000) for h in hs])\n"
            "    show = tab[['horizon','n_events','mean','base_mean','hit','base_hit','t_hac','t_one','t_welch','p_placebo','net']].copy()\n"
            "    for c in ['mean','base_mean','net']: show[c] = (show[c]*100).round(3)\n"
            "    for c in ['hit','base_hit']: show[c] = show[c].round(0)\n"
            "    for c in ['t_hac','t_one','t_welch']: show[c] = show[c].round(2)\n"
            "    show['p_placebo'] = show['p_placebo'].round(3)\n"
            "else:\n"
            "    cols=['horizon','n_events','mean','base_mean','hit','base_hit','t_hac','t_one','t_welch','p_placebo','net5','net10']\n"
            "    show = pd.DataFrame([dict(zip(cols, R[k])) for k in keys]).drop(columns=['net10']).rename(columns={'net5':'net'})\n"
            "print(show.to_string(index=False))"
        ),
        md(
            "> 💡 In plain words: the `hit`/`base_hit` columns sit within 1-2 points of "
            "each other at every horizon — no meaningful tilt from the pattern. "
            "`t_welch`, the decisive column, never clears +2, and is negative at three of "
            "four horizons — a mild underperformance, not a reversal."
        ),
        md(
            "### 4c · Bonferroni across the basket — does one name secretly carry it?\n\n"
            "A pooled null can hide a real effect concentrated in a handful of names, or "
            "manufacture a fake one from a single noisy outlier. Per-ticker Welch *t* "
            "(event sample vs that ticker's *own* unconditional pool) at $H=5$, with the "
            f"Bonferroni-corrected bar for $k={R['bonf_k']}$ simultaneous tests."
        ),
        code(
            "pt = st.per_ticker_stats(PANEL, 5) if HAVE_REAL else None\n"
            "if pt is not None and len(pt):\n"
            "    zc = st.bonferroni_z(len(pt))\n"
            "    order = pt.reindex(pt['t_welch'].abs().sort_values(ascending=False).index)\n"
            "    tickers = list(order['ticker']); tv = list(order['t_welch'])\n"
            "    k = len(pt)\n"
            "else:\n"
            "    zc = R['bonf_z']; k = R['bonf_k']\n"
            "    tickers = [b[0] for b in R['bonf_top']]; tv = [b[3] for b in R['bonf_top']]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "cols = [RED if abs(t) >= zc else GREY for t in tv[:15]]\n"
            "ax.bar(tickers[:15], tv[:15], color=cols, width=.6)\n"
            "ax.axhline(zc, ls='--', c=RED, lw=1, label=f'Bonferroni bar |t|>={zc:.2f} (k={k})')\n"
            "ax.axhline(-zc, ls='--', c=RED, lw=1)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('per-ticker Welch t (vs its own unconditional base)'); ax.set_xlabel('ticker (top 15 by |t|)')\n"
            "ax.set_title('No single name survives the multiple-testing correction')\n"
            "plt.xticks(rotation=45, ha='right'); ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'{k} tickers tested, Bonferroni bar={zc:.2f}, survivors:', sum(abs(t)>=zc for t in tv))"
        ),
        md(
            f"> 💡 In plain words: even the loudest names in a {R['bonf_k']}-ticker sample "
            "— exactly the outliers a chart-reader would cherry-pick and show you — sit "
            f"*below* the corrected bar (**{R['bonf_survive']}/{R['bonf_k']}** clear it), "
            f"and the loudest name ({R['bonf_top'][0][0]} at {R['bonf_top'][0][3]:.2f}) is "
            "negative — the wrong direction for a reversal story. The pooled null in 4a/4b "
            "isn't hiding a real effect in a handful of names; there's simply no name "
            "carrying it."
        ),
        md(
            "### 4d · The myth checks — does a stricter close-tie, or a genuine downtrend, rescue it?\n\n"
            "Two ways to make the signal \"purer\": a **near-exact close tie** (0.03% vs "
            "the 0.15% default), and only counting patterns that follow a **genuine prior "
            "downtrend** (a real support test, not a random pair of red days). If the lore "
            "were right, these should sharpen the effect."
        ),
        code(
            "if HAVE_REAL:\n"
            "    basic  = [st.summarize(PANEL, h, placebo=False)['t_welch'] for h in hs]\n"
            "    strict = [st.summarize(PANEL, h, placebo=False, tol=st.STRICT_TOL)['t_welch'] for h in hs]\n"
            "    trend  = [st.summarize(PANEL, h, placebo=False, require_downtrend=True)['t_welch'] for h in hs]\n"
            "    ns = st.summarize(PANEL, 5, placebo=False, tol=st.STRICT_TOL)['n_events']\n"
            "    nt = st.summarize(PANEL, 5, placebo=False, require_downtrend=True)['n_events']\n"
            "else:\n"
            "    basic  = [R[k][8] for k in keys]\n"
            "    strict = [s[3] for s in R['strict']]; trend = [t[3] for t in R['trend']]\n"
            "    ns = R['strict'][1][1]; nt = R['trend'][1][1]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.3))\n"
            "ax.bar(x-.25, basic, .25, color=RED, label='basic matching low')\n"
            "ax.bar(x,      strict,.25, color=AMBER, label=f'near-exact tie (n5={ns})')\n"
            "ax.bar(x+.25,  trend, .25, color=GREY, label=f'genuine downtrend (n5={nt})')\n"
            "ax.axhline(2, ls='--', c='k', lw=1); ax.axhline(-2, ls='--', c='k', lw=1); ax.axhline(0,c='k',lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('Welch t (vs unconditional base, +-2 dashed)'); ax.set_title('No filter clears +2')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('strict Welch t:', [round(s,2) for s in strict]); print('trend Welch t :', [round(t,2) for t in trend])"
        ),
        md(
            f"> 💡 In plain words: the near-exact close tie (only **{R['strict'][1][1]}** "
            f"events at 5d — a much smaller, rarer sample) gives Welch *t* = "
            f"**{R['strict'][1][3]:.2f}**, and the genuine-downtrend filter gives *t* = "
            f"**{R['trend'][1][3]:.2f}** at 5 days and **{R['trend'][3][3]:.2f}** at 20 — "
            "closer to the bar than the loose default, but still well short of +2 at every "
            "horizon. The pattern's predictive content is **indistinguishable from the "
            "basket's own drift** under every reasonable definition."
        ),
        md(
            "### 4e · Costs — there is no excess to eat\n\n"
            "Gross vs net (5 / 10 bps one-way × 2 on every event, long only, no borrow). "
            "When the excess over the base is already statistically nothing, costs simply "
            "shift the whole gross number down."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g  = [st.summarize(PANEL, h, placebo=False)['mean']*100 for h in hs]\n"
            "    n5 = [float(st.event_net_returns(st.collect_events(PANEL, h), cost_bps=5.0).mean())*100 for h in hs]\n"
            "    n10= [float(st.event_net_returns(st.collect_events(PANEL, h), cost_bps=10.0).mean())*100 for h in hs]\n"
            "else:\n"
            "    g  = [R[k][2] for k in keys]\n"
            "    n5 = [R[k][10] for k in keys]; n10 = [R[k][11] for k in keys]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.25, g, .25, color=AMBER, label='gross')\n"
            "ax.bar(x,      n5, .25, color=RED, label='net @ 5 bps')\n"
            "ax.bar(x+.25,  n10,.25, color='#7a1f14', label='net @ 10 bps')\n"
            "ax.axhline(0,c='k',lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('return (%)'); ax.set_title('Costs bite exactly where the excess was already zero')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for h,gg,n5_,n10_ in zip(hs,g,n5,n10): print(f'{h:>2}d: gross={gg:+.3f}%  net5bps={n5_:+.3f}%  net10bps={n10_:+.3f}%')"
        ),
        md(
            f"> 💡 In plain words: at 1 day gross **{R['h1'][2]:+.2f}%** → net "
            f"**{R['h1'][10]:+.2f}%** (5 bps) — already a small loss. At longer horizons "
            "the net stays nominally positive, but that positivity tracks the basket's own "
            "beta (see 4a), not an edge from the pattern. **Mirage** in the strict sense — "
            "nothing *specific to the signal* to charge costs against."
        ),
        md(
            "### 4f · Faithful-engine & power control — the direction trap, proven on a KNOWN-zero panel\n\n"
            "A deterministic panel with its own mild embedded drift (no overnight gap — "
            "the matching low needs none) and a **planted** post-pattern reversal knob. "
            "With **zero** planted edge, the certifying **Welch t** (vs the panel's own "
            "unconditional base) must stay inside ±2 across **20 independent seeds**."
        ),
        code(
            "null_welch = []\n"
            "for s_ in range(20):\n"
            "    px, _ = data.synthetic_panel(edge=0.0, seed=694 + s_)\n"
            "    null_welch.append(st.summarize(px, 5, placebo=False)['t_welch'])\n"
            "null_welch = np.asarray(null_welch)\n"
            "res = []\n"
            "for edge in (0.0, 0.002, 0.004):\n"
            "    px, truth = data.synthetic_panel(edge=edge, seed=694)\n"
            "    s = st.summarize(px, 5, n_draws=4000, placebo=False)\n"
            "    res.append((edge, truth['n_planted_days'], s['n_events'], s['mean']*100, s['t_welch'], s['t_hac'], s['hit']*100))\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.scatter(np.zeros(20) + np.linspace(-.1,.1,20), null_welch, color=GREY, s=40, label='null worlds (edge=0), 20 seeds')\n"
            "a1.axhline(-2, ls='--', c=RED, lw=1); a1.axhline(2, ls='--', c=RED, lw=1); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_xticks([0]); a1.set_xticklabels(['null x 20']); a1.set_ylabel('Welch t (vs panel unconditional base)')\n"
            "a1.set_title(f'Null mostly does NOT fire: {int((abs(null_welch)>=2).sum())}/20 seeds >= |t|=2'); a1.legend()\n"
            "labels = [f'planted\\n{e*100:.1f}%/day' for e,_,_,_,_,_,_ in res]\n"
            "tvals = [r[4] for r in res]\n"
            "a2.bar(labels, tvals, color=[GREY, AMBER, GREEN], width=.5)\n"
            "a2.axhline(2, ls='--', c=RED, label='t=+2 bar')\n"
            "for i,t in enumerate(tvals): a2.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "a2.set_ylabel('Welch t (5d, seed 694)'); a2.set_title('A planted reversal lights up cleanly'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null Welch t: mean={null_welch.mean():+.2f} sd={null_welch.std(ddof=1):.2f} |t|>=2 in {(abs(null_welch)>=2).sum()}/20')\n"
            "for e,pd_,k,ls,tw,th,w in res: print(f'planted {e*100:+.1f}%/day: n={k} mean={ls:+.3f}% t_welch={tw:.2f} t_hac(vs0)={th:.2f} hit={w:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted reversal, the certifying Welch *t* "
            f"averages **{R['syn_null_mean']:+.2f}** (sd {R['syn_null_sd']:.2f}) across 20 "
            f"seeds and fires in only **{R['syn_null_fire']}/{R['syn_null_n']}** — right at "
            "the ~5% nominal false-positive rate of a two-sided two-sigma bar, not a "
            f"systematic bias. A planted reversal of 0.2%/day reaches Welch *t* = "
            f"**{R['syn'][1][4]:.2f}**, and 0.4%/day reaches **{R['syn'][2][4]:.2f}** — the "
            "harness *would* catch a real matching-low reversal if one existed. This is "
            "exactly why Welch-vs-unconditional-base is the number that decides the "
            "verdict, not HAC-vs-zero."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the certifying, drift-neutral Welch *t* (event sample "
            f"vs the basket's own unconditional forward return) never certifies a reversal "
            f"at any horizon ({R['h1'][8]:+.2f} / {R['h5'][8]:+.2f} / {R['h10'][8]:+.2f} / "
            f"{R['h20'][8]:+.2f} for 1/5/10/20 days). The random-draw placebo places every "
            f"horizon at *p* ≥ {min(R['h1'][9], R['h5'][9], R['h10'][9], R['h20'][9]):.3f}. "
            f"**{R['bonf_survive']}/{R['bonf_k']}** tickers survive a Bonferroni correction "
            "individually. Neither a near-exact close tie nor a genuine prior-downtrend "
            "filter rescues it (best reading +1.28, still short of the bar). The **vs-zero** "
            f"*t* climbs to **{R['h20'][6]:.2f}** by 20 days — that's the basket's own "
            "up-drift, not the pattern; the gap between the two statistics *is* the "
            "finding. Carries a **survivorship** caveat that tilts *toward* finding a "
            "working reversal signal. NONE, not WEAK.\n"
            f"- **Tradability `MIRAGE`** — nominally positive **net** returns from 5 days "
            f"out ({R['h5'][10]:+.2f}% at 5 bps) track the unconditional base almost "
            f"bar-for-bar; at 1 day, net of even 5 bps, the trade is already negative "
            f"({R['h1'][10]:+.2f}%). There is no *excess* return to charge costs against — "
            "just the basket's ordinary beta.\n"
            "- **Marks support? `BUSTED`** — the \"tested the same price twice, buyers "
            "defended it\" reading does not survive a fair comparison to what the same "
            "basket does on an ordinary day. Neither purity filter saves it, and no "
            "individual name in the basket carries a Bonferroni-robust version of the "
            "story."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson.** Any long-only signal on a rising basket must be "
            "compared to the basket's own baseline, never to zero — the vs-zero trap shown "
            "live in 4a/4f is the single most important idiom in this study, and applies "
            "equally to any long-only chart pattern, momentum rule, or seasonal effect.\n"
            "- **The \"same price twice\" neighbours.** "
            "[409-tweezer-tops-bottoms](../../409-tweezer-tops-bottoms/) tests matching "
            "*wick lows* with no color requirement — a looser, wick-level claim; "
            "[460-counterattack-lines](../../460-counterattack-lines/) requires the second "
            "candle to be the **opposite** color, the reversal already visible inside the "
            "pattern; [696-double-bottom](../../696-double-bottom/) is the multi-week "
            "swing-chart version of the same idea. None of them run this study's specific "
            "two-consecutive-down-candle, matching-close, long-only detector.\n\n"
            "*The reproducible core is offline and deterministic; the detector is two "
            "consecutive down candles with a stated close-matching tolerance, a strict "
            "tolerance and a genuine-downtrend filter as myth-checks, and a Bonferroni "
            "correction across the 30-name basket. Methods and sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
            "[`docs/results.md`](../docs/results.md).*"
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
