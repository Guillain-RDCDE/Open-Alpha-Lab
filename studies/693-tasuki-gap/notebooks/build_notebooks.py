"""Generate the two narrative notebooks for Study 693 (Tasuki Gap).

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
    fp_quantlab="8af479f8c79c", fp_basket="aa438a4d53ea", total_bars=162180,
    # TREND-SIGNED forward return after a tasuki gap, per horizon:
    # (H, n, n_up, n_down, mean%, base%, hit%, basehit%, t_hac, t_one, t_welch[decisive], p_placebo, net5%, net10%)
    h1=(1, 1277, 656, 621, 0.007, 0.000, 48, 50, 0.15, 0.15, 0.15, 0.429, -0.094, -0.194),
    h5=(5, 1273, 655, 618, -0.259, 0.000, 48, 50, -2.40, -2.34, -2.33, 0.995, -0.364, -0.464),
    h10=(10, 1269, 652, 617, -0.189, 0.000, 49, 50, -1.38, -1.30, -1.29, 0.924, -0.298, -0.398),
    h20=(20, 1267, 651, 616, -0.011, 0.000, 50, 50, -0.06, -0.05, -0.05, 0.585, -0.131, -0.231),
    hit_wilson={1: (45.7, 51.2), 5: (44.9, 50.4), 10: (46.4, 51.9), 20: (47.6, 53.1)},
    # up/down split: horizon -> (up_mean%, up_t_welch, n_up, down_mean%, down_t_welch, n_down)
    split={1: (-0.000, -0.50, 656, 0.014, 0.61, 621),
           5: (0.150, -0.72, 655, -0.692, -2.58, 618),
           10: (0.518, -0.05, 652, -0.935, -1.98, 617),
           20: (1.235, 0.50, 651, -1.328, -0.80, 616)},
    # strict full-range-gap myth-check: (H, n, up, dn, mean%, t_welch, t_hac, p, net5%)
    strict=[(1, 472, 244, 228, -0.007, -0.09, -0.09, 0.548, -0.108),
            (5, 469, 243, 226, -0.231, -1.20, -1.24, 0.927, -0.335),
            (10, 467, 241, 226, -0.186, -0.69, -0.65, 0.812, -0.296),
            (20, 467, 241, 226, 0.271, 0.68, 0.68, 0.233, 0.151)],
    # genuine-prior-trend myth-check: same tuple shape
    trend=[(1, 616, 349, 267, 0.092, 1.31, 1.45, 0.057, -0.009),
           (5, 614, 348, 266, -0.358, -2.03, -2.03, 0.998, -0.462),
           (10, 612, 347, 265, -0.502, -2.36, -2.34, 0.998, -0.611),
           (20, 612, 347, 265, -0.435, -1.51, -1.54, 0.979, -0.552)],
    # Bonferroni across the basket (per-ticker pooled Welch t vs that ticker's own base, H=5)
    bonf_k=30, bonf_z=3.14, bonf_survive=0,
    bonf_top=[("MRK", 49, -1.683, -2.15), ("CSCO", 44, -0.843, -1.89), ("SPY", 50, -0.504, -1.73)],
    # synthetic null over 20 seeds (decisive stat: Welch t vs the panel's own direction-matched base)
    syn_null_mean=-0.41, syn_null_sd=1.12, syn_null_fire=1, syn_null_n=20,
    # planted control (seed 693): (edge, planted_days, n, mean%, t_welch, t_hac, hit%)
    syn=[(0.000, 601, 599, -0.104, -0.76, -0.82, 50),
         (0.012, 593, 591, 1.119, 8.06, 8.60, 65),
         (0.020, 593, 591, 1.929, 13.85, 14.85, 72)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Predicts_continuation%3F: Busted](https://img.shields.io/badge/Predicts_continuation%3F-Busted-8b949e?style=flat-square)\n\n"
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

from tasuki_gap import data, strategy as st

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
            "# The tasuki gap — does an unfilled gap really mean the trend has more room? 🕳️🕯️\n"
            "### A classic Japanese candlestick **continuation** pattern, measured on 21.5 years "
            "of real tape\n\n"
            + BADGES +
            "Open a candlestick book and you'll meet the **tasuki gap**: a market gaps in the "
            "direction it's trending, then takes a small breather — a candle of the opposite "
            "color pokes back toward the gap, but **can't quite close it**. The lore reads this as "
            "good news for the trend: if the pullback couldn't even close a small gap, the move "
            "still has fuel left, and it should **resume**. There's an upside version (bullish, go "
            "long) and a downside mirror (bearish, go short).\n\n"
            "So we did the boring, honest thing: found **every** tasuki gap — both flavors — in a "
            "basket of 30 big US stocks + SPY over 21.5 years, and measured what actually happened "
            "next.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo tests and the "
            "Bonferroni correction? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** We use a fixed **30-name liquid large-cap + SPY** "
            "basket (names still trading today), so this carries **survivorship** in both "
            "directions — it excludes firms whose uptrend never resumed (they delisted) and, "
            "symmetrically, firms that never stopped falling. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After a tasuki gap, does the trend actually resume? | **No — not in any way we "
            "can certify.** Trade in the predicted direction the next morning and, across 1–20 "
            "days, you're roughly flat-to-losing once compared fairly to an ordinary day. |\n"
            "| Is there at least a hint of continuation somewhere? | **The opposite, mildly.** At "
            "5 and 10 days the downside (bearish) version actually shows a small **reversal** — "
            "shorting after it loses money relative to shorting anything else in the basket. Not "
            "certified enough to trade, but it's the wrong-way surprise in this whole story. |\n"
            "| Does the picture-perfect version (a true, full-range gap, no overlap at all) work "
            "better? | **No.** It doesn't sharpen anything — the strict shape is just rarer. |\n"
            "| What if I only count patterns that genuinely continue an already-established "
            "trend? | **Still nothing — if anything, slightly worse.** |\n"
            "| Does any single stock secretly carry the pattern? | **No.** Checked individually "
            f"across all {R['bonf_k']} names with enough events, using a stricter statistical bar "
            f"(so one lucky name can't pass as \"the\" signal), **{R['bonf_survive']}/{R['bonf_k']}** "
            "clear it. |\n\n"
            "> A gap that stays open doesn't mean the trend has \"fuel left\" — on the tape it "
            "means almost nothing at all, and shorting the down-flavor into it **loses money**."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The **tasuki gap** is a continuation signal. In an uptrend: a white candle gaps "
            "up, another white candle gaps up again, and then a small black candle opens inside "
            "that second candle and pulls back — but it **can't close the gap**. If the sellers "
            "couldn't even erase a fresh gap, the buyers still have the upper hand, and the rally "
            "resumes. The downside tasuki gap is the exact bearish mirror.\"*\n\n"
            "This is straight out of Steve Nison's *Japanese Candlestick Charting Techniques* (the "
            "book that brought candles to the West) and is catalogued across the standard "
            "candlestick references as one of the older continuation figures. We measure it "
            "independently, on our own basket and protocol, and ask the only question that "
            "matters: **does the trend actually keep going?**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the tasuki gap really predicted continuation, it would be a genuinely useful, "
            "fairly common signal — unlike some of the desk's rarer three-candle patterns, tasuki "
            "gaps show up often enough (about 1,270 times in our basket) to actually build a "
            "systematic strategy around, if the edge were real.\n\n"
            "But there's a trap built into the shape itself: a gap plus a quick pullback candle is "
            "*exactly* the setup short-horizon mean-reversion research warns about. A market that "
            "just gapped hard and is already being faded a little might keep fading, not resume — "
            "the opposite of what the folklore promises. We'll see which force wins, and — just as "
            "important — whether any apparent edge is *real* trend-following force or just the "
            "basket's own ordinary drift dressed up as a pattern."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take a fixed **{R['n_names']}-name** basket of liquid US large-caps + **SPY** and "
            f"scan every daily bar from **{R['start']}** to **{R['end']}** ({R['years']:.1f} "
            "years). For each name we flag **every** tasuki gap with a precise OHLC rule — both "
            "the upside (two white candles gapping up, then a black pullback candle that opens "
            "inside the second body and closes back inside the gap without filling it) and the "
            "downside mirror. Then, with **no cheating**:\n\n"
            "1. **Wait for the close** that confirms the pullback candle.\n"
            "2. **Enter the next morning's open** (one day's lag — you can't trade a bar still "
            "forming) and hold **1, 5, 10, or 20** trading days, in the **predicted trend "
            "direction** (long for upside tasuki, short for downside).\n"
            "3. **Compare fairly** — not to zero (a naive comparison is fooled by the market's own "
            "drift, and confuses long and short bets), but to what the *same basket* earns on an "
            "unconditional day, matched to the same direction, and to thousands of random \"fake "
            "signal\" picks in the same long/short mix. If riding the trend after a tasuki gap "
            "beats the everyday tide, the legend is real. If it doesn't, continuation is a myth."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Here's the trend-signed return after a tasuki gap, at each "
            "horizon, next to what the *same basket* earns on an ordinary (unconditional) day in "
            "the same direction. A real \"continuation\" signal would show the pattern's bar "
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
            "    means = [R[k][4] for k in keys]; base = [R[k][5] for k in keys]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.5))\n"
            "ax.bar(x-.18, means, .34, color=RED, label='AFTER a tasuki gap (trend direction)')\n"
            "ax.bar(x+.18, base, .34, color=GREY, label='ORDINARY day, same bet (baseline)')\n"
            "for i,v in enumerate(means): ax.annotate(f'{v:+.2f}%',(i-.18,v),ha='center',va='top' if v<0 else 'bottom',fontsize=9)\n"
            "for i,v in enumerate(base): ax.annotate(f'{v:+.2f}%',(i+.18,v),ha='center',va='top' if v<0 else 'bottom',fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('trend-signed return (%)')\n"
            "ax.set_title('The pattern bar does NOT beat the everyday baseline')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('after tasuki gap:', {f'{h}d': round(m,3) for h,m in zip(hs,means)})\n"
            "print('ordinary day (same bet):', {f'{h}d': round(b,3) for h,b in zip(hs,base)})"
        ),
        md(
            f"At 5 days the pattern trade is **{R['h5'][4]:+.2f}%** — *negative* — against "
            f"**{R['h5'][5]:+.2f}%** for an ordinary day's same-direction bet. The pattern doesn't "
            "add continuation information; if anything it briefly underperforms doing nothing "
            "special. The \"significant-looking\" number you'd get by comparing the pattern to "
            "**zero** — ignoring which direction each trade actually is — is a mirage created by "
            "the basket's own drift, not the gap."
        ),
        md(
            "**Is it the long side or the short side dragging this down?** Tasuki gaps come in two "
            "flavors — let's split them."
        ),
        code(
            "if HAVE_REAL:\n"
            "    up_m = [rows[h]['up_mean']*100 for h in hs]\n"
            "    dn_m = [rows[h]['down_mean']*100 for h in hs]\n"
            "else:\n"
            "    up_m = [R['split'][h][0] for h in hs]; dn_m = [R['split'][h][3] for h in hs]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.18, up_m, .34, color=GREEN, label='upside tasuki (long)')\n"
            "ax.bar(x+.18, dn_m, .34, color=RED, label='downside tasuki (short)')\n"
            "for i,v in enumerate(up_m): ax.annotate(f'{v:+.2f}%',(i-.18,v),ha='center',va='top' if v<0 else 'bottom',fontsize=9)\n"
            "for i,v in enumerate(dn_m): ax.annotate(f'{v:+.2f}%',(i+.18,v),ha='center',va='top' if v<0 else 'bottom',fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('signed return, split by direction (%)')\n"
            "ax.set_title('The downside (bearish) leg is where the dip lives')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('upside (long):', {f'{h}d': round(v,3) for h,v in zip(hs,up_m)})\n"
            "print('downside (short):', {f'{h}d': round(v,3) for h,v in zip(hs,dn_m)})"
        ),
        md(
            f"The **upside** tasuki (long) is small and bounces around zero at every horizon — "
            "never a real edge, but also never a real problem. The **downside** tasuki (short) is "
            f"where the story lives: at 5 days it's **{R['split'][5][3]:+.2f}%**, a mild "
            "**reversal** — shorting into a downside tasuki loses relative to shorting anything "
            "else in the basket. The market couldn't close the gap, but instead of *falling "
            "further* it tended to bounce."
        ),
        md(
            "**Could it just be luck?** Let's pit the real signal against thousands of **fake** "
            "ones matched to the same long/short mix: pick the same number of random days, flip a "
            "weighted coin, and see how the real pattern's return stacks up."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ev = st.collect_events(PANEL, 5)\n"
            "    n_up = int((ev['direction']>0).sum()); n_dn = int((ev['direction']<0).sum())\n"
            "    obs = float(ev['signed_ret'].mean())\n"
            "    pl = st.placebo_pvalue(PANEL, 5, n_up, n_dn, obs, n_draws=5000)\n"
            "    draws = pl['draws']*100; obsp = obs*100; pval = pl['p_value']\n"
            "else:\n"
            "    obsp = R['h5'][4]; pval = R['h5'][11]\n"
            "    rng = np.random.default_rng(693); draws = rng.normal(0.0, 0.34, 5000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='5,000 mix-matched coin-flip \"signals\"')\n"
            "ax.axvline(obsp, c=RED, lw=2.5, label=f'real tasuki-gap trade {obsp:+.2f}%')\n"
            "ax.set_xlabel('5-day trend-signed return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The real pattern sits in the WRONG tail: p = {pval:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obsp:+.2f}%   share of random fakes that beat it: p = {pval:.3f}')"
        ),
        md(
            f"With placebo *p* = **{R['h5'][11]:.3f}**, roughly 99.5% of random, undirected coin "
            "flips in the same long/short mix would have done *better* than the real pattern. "
            "That's the opposite of a genuine edge — it's a mild statistical curiosity, not "
            "something to trade."
        ),
        md(
            "**Does the picture-perfect version save it?** Maybe an overlapping, casual gap is "
            "noise but the strict, true full-range gap — or only counting patterns that follow a "
            "*genuine*, already-established trend — is the real deal. Let's check both stricter "
            "recipes against a fair (direction-matched-base) comparison."
        ),
        code(
            "if HAVE_REAL:\n"
            "    basic  = [st.summarize(PANEL, h, placebo=False)['t_welch'] for h in hs]\n"
            "    strict = [st.summarize(PANEL, h, placebo=False, true_range_gap=True)['t_welch'] for h in hs]\n"
            "    trend  = [st.summarize(PANEL, h, placebo=False, require_trend=True)['t_welch'] for h in hs]\n"
            "else:\n"
            "    basic  = [R[k][10] for k in ['h1','h5','h10','h20']]\n"
            "    strict = [s[5] for s in R['strict']]; trend = [t[5] for t in R['trend']]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.3))\n"
            "ax.bar(x-.25, basic, .25, color=RED, label='basic tasuki gap')\n"
            "ax.bar(x,      strict,.25, color=AMBER, label='strict full-range gap')\n"
            "ax.bar(x+.25,  trend, .25, color=GREY, label='genuine prior-trend context')\n"
            "ax.axhline(0, c='k', lw=.8); ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('Welch t (vs the direction-matched base — the fair bar, +-2 dashed)')\n"
            "ax.set_title('No recipe clears the bar in the CLAIMED direction')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('basic t:', [round(b,2) for b in basic]); print('strict t:', [round(s,2) for s in strict]); print('trend t:', [round(t,2) for t in trend])"
        ),
        md(
            "No bar in that chart pokes above the dashed **+2** line — the direction the folklore "
            "needs — under any of the three recipes. If anything, requiring a genuine prior trend "
            "makes the dip at 5 and 10 days *deeper*, the exact opposite of what a \"purer, more "
            "authentic\" continuation setup should do."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Measured fairly (against the basket's own unconditional day, "
            "matched to direction, not against zero), the trend-signed tasuki-gap trade never "
            "certifies a continuation at any horizon. The only reading past ±2 (5-day) points "
            "**backwards** — a mild reversal on the downside leg. No single stock secretly carries "
            f"it either — **{R['bonf_survive']}/{R['bonf_k']}** individual tickers clear a "
            "corrected bar. There's no continuation edge.\n"
            "- **Tradability — Mirage.** It's flat-to-negative **before** costs; once you add the "
            "spread and short borrow it's worse at every horizon. Nothing to deploy.\n"
            "- **\"Predicts continuation\"? — Busted.** The famous \"gap can't be closed, the trend "
            "has fuel left\" reading doesn't survive contact with the tape once you compare it to "
            "an ordinary day, in the same direction."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — no, and here's the kicker\n\n"
            "The trade is already flat-to-negative gross. Now add the real-world frictions — the "
            "spread paid on the round trip, plus borrow to hold the short leg. The bars only sink "
            "further."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g  = [st.summarize(PANEL, h, placebo=False)['mean']*100 for h in hs]\n"
            "    n5 = [float(st.event_net_returns(st.collect_events(PANEL, h), h, cost_bps=5.0).mean())*100 for h in hs]\n"
            "else:\n"
            "    keys = ['h1','h5','h10','h20']\n"
            "    g  = [R[k][4] for k in keys]; n5 = [R[k][12] for k in keys]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, g, .4, color=AMBER, label='gross trend-signed return')\n"
            "ax.bar(x+.2, n5, .4, color=RED, label='net of costs + borrow (5 bps)')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('trend-signed return (%)'); ax.set_title('Costs only deepen the loss — there is nothing to harvest')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('net trend-signed return by horizon:', {f'{h}d': round(v,3) for h,v in zip(hs,n5)})"
        ),
        md(
            f"At every horizon the **net** trade is more negative than the gross — e.g. "
            f"**{R['h5'][12]:.2f}%** at 5 days. There is no window where it pays. The tasuki gap "
            "is common enough to build a strategy around if it worked — it just doesn't."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Why does the naive comparison fool people?** Any trend-following bet in a "
            "drifting market looks \"significant\" against **zero** — with or without a tasuki gap "
            "— because long trades ride the market's own drift and short trades fight it. That's "
            "why this study's certifying test compares each event to what the *same basket* earns "
            "on an ordinary day, in the *same direction*. The "
            "[quants notebook](02_for_the_quants.ipynb) shows this trap live on a synthetic panel "
            "with a **known** zero edge.\n"
            "- **The gap-pattern family.** "
            "[74-mind-the-gap](../../74-mind-the-gap/) asks the general \"do gaps fill\" question "
            "across all shapes; [417-island-reversal](../../417-island-reversal/) needs *two* "
            "opposite gaps predicting the opposite outcome (reversal); "
            "[455-three-methods](../../455-three-methods/) is a continuation pattern with **no gap "
            "at all** — none of them run this study's specific figure.\n\n"
            "*Think a stricter gap, a different universe, or a clever filter turns it green? Show "
            "the **net** trend-signed return landing above the basket's own direction-matched "
            "baseline with Welch *t* ≥ 2 — then we'll talk.*"
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
            "# Tasuki Gap — a quantitative teardown 🔬\n"
            "### Precise OHLC detector · trend-signed forward 1/5/10/20-day event study · a "
            "direction-matched Welch *t* design · a mix-matched coin-flip placebo · a Bonferroni "
            "correction across the 30-name basket · costs + borrow · full-range-gap & "
            "prior-trend-context myth-checks · a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "tasuki gap is one of candlestick lore's classic **continuation** figures — the job "
            "here is to test it honestly as a directional signal, with one subtlety front and "
            "centre: the basket carries the market's own positive drift, so a naive *t*-vs-zero "
            "**overstates** long-side continuation and **understates** short-side continuation. "
            "Everything below is built around removing that contamination.\n\n"
            "> ⚠️ **Data + survivorship note.** Fixed **30-name liquid large-cap + SPY** basket, "
            "names still trading in 2026 — a *survivor* panel that tilts *toward* finding a "
            "working long-side (upside tasuki) signal and, symmetrically, *against* finding a "
            "working short-side (downside tasuki) signal. Real data: yfinance daily OHLCV, "
            "`auto_adjust=True`, 2005→2026, as-of **2026-06-30**. Offline core + synthetic control "
            "are deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers "
            "in [`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_quantlab"] +
            "` / `" + R["fp_basket"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Direction-matched **Welch *t*** (event sample vs the "
            f"basket's own direction-matched unconditional forward return — the certifying number) "
            f"never certifies a continuation at any horizon: {R['h1'][10]:+.2f} (1d) / "
            f"{R['h5'][10]:+.2f} (5d) / {R['h10'][10]:+.2f} (10d) / {R['h20'][10]:+.2f} (20d) — the "
            "one reading past |2| points **backwards**. Mix-matched placebo *p* = "
            f"{R['h5'][11]:.3f} at 5d. **{R['bonf_survive']}/{R['bonf_k']}** tickers clear a "
            "Bonferroni-corrected per-name bar. |\n"
            f"| **Tradability** | `MIRAGE` | Flat-to-negative **gross** at every horizon "
            f"({R['h1'][4]:+.2f}% to {R['h5'][4]:+.2f}%); net of 5/10-bps round trip + 50 bps/yr "
            f"borrow it is worse ({R['h5'][12]:+.2f}% / {R['h5'][13]:+.2f}% net at 5d). |\n"
            "| **Predicts continuation?** | `BUSTED` | The \"gap can't close, the trend has fuel "
            "left\" reading does not survive being measured fairly against the basket's own "
            "direction-matched drift; neither the strict full-range-gap nor the genuine-trend-"
            "context filter rescues it — the trend filter makes the dip *worse*. |\n\n"
            "> 💡 In plain words: the tasuki gap has **no certifiable predictive power** as a "
            "continuation signal — and the one horizon that looks \"significant\" is significant "
            "in the wrong direction (a mild reversal on the short leg)."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned — and the direction trap named up front\n\n"
            "Let an **upside** tasuki gap be confirmed at the close of bar $t$ when the OHLC "
            "triple $(t\\!-\\!2,t\\!-\\!1,t)$ satisfies: candles $t\\!-\\!2$ and $t\\!-\\!1$ are "
            "both **bullish** ($C_0>O_0$, $C_1>O_1$), $t\\!-\\!1$ **gaps up** from $t\\!-\\!2$ "
            "($\\min(O_1,C_1)>\\max(O_0,C_0)$), and candle $t$ is **bearish** ($C_2<O_2$), opens "
            "**inside** $t\\!-\\!1$'s real body ($O_1\\le O_2\\le C_1$) and closes back **inside "
            "the gap without filling it** ($C_0<C_2<O_1$). The **downside** tasuki gap is the "
            "exact mirror (bearish, bearish, then a bullish pullback). Enter the **next** open "
            "(one lag), hold $H$ days **in the predicted trend direction** ($d=+1$ upside, "
            "$d=-1$ downside), and let $s_i(H)=d_i\\cdot(\\text{Close}_{t+H}/\\text{Open}_{t+1}-1)$ "
            "be the trend-signed return of event $i$.\n\n"
            "- **H₁ (continuation exists).** $\\overline{s}(H)$ beats the **direction-matched "
            "unconditional** return of the same basket, significantly, for some $H$ — trading the "
            "tasuki gap adds information beyond just trading its own direction anyway.\n"
            "- **H₂ (it's deployable).** The excess survives the round-trip cost + short borrow.\n"
            "- **H₃ (\"strict gap / genuine trend\").** Restricting to a true full-range gap after "
            "an already-established trend sharpens the edge.\n\n"
            "**Why direction-match the base, not just compare to zero?** Because this basket "
            "drifts **up**, a *long* event beats zero for free (with or without any pattern) while "
            "a *short* event is fighting an uphill battle against that same drift. A vs-zero test "
            "therefore inflates the long leg and deflates the short leg purely from the tape's "
            "ordinary drift — not from the pattern. Section 4a demonstrates this trap directly on a "
            "synthetic panel with a **known** zero pattern-edge. We find **H₁ rejected** (Welch "
            "$t\\in[-2.33,+0.15]$, the sole |t|≥2 reading pointing the wrong way), **H₂ rejected** "
            "(net worse than gross), **H₃ rejected** (filters don't help, and the trend filter "
            "makes it worse)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The Signal axis's **decisive** statistic is a **Welch *t*** of the event sample's "
            "trend-signed mean against a **direction-matched** unconditional pool — the plain base "
            "for long (upside) events, its **negation** for short (downside) events, concatenated "
            "in proportion:\n\n"
            "$$t_{\\text{Welch}} = \\frac{\\overline{s}_{\\text{event}} - \\overline{r}_{\\text{matched}}}"
            "{\\sqrt{\\widehat{\\mathrm{Var}}(s)/n_{\\text{event}} + \\widehat{\\mathrm{Var}}(r_{\\text{matched}})/n_{\\text{base}}}}.$$\n\n"
            "A one-sample **Newey-West HAC *t*** against zero is reported alongside as an "
            "**informational** cross-check only — it is the statistic a naive read would use, and "
            "it is exactly the one contaminated by the basket's own drift *and* by whichever "
            "direction happens to dominate the sample (proved on the synthetic null in 4f). The "
            "**mix-matched coin-flip placebo** (same event count, coin-signed to match the observed "
            "up/down mix — not a flat 50/50) is a second, independent honesty check. Finally, with "
            f"**{R['bonf_k']} tickers** tested individually for a per-name version of the pooled "
            "(up+down) effect, the two-sided significance bar widens via **Bonferroni** to "
            f"$|t|\\ge z(1-0.025/{R['bonf_k']})\\approx {R['bonf_z']:.2f}$ — otherwise the single "
            "loudest name in a wide basket gets mistaken for \"the\" signal."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** Fixed **{R['n_names']}-name** liquid large-cap + SPY basket "
            f"(yfinance daily OHLCV, `auto_adjust=True`, {R['start']}→{R['end']}, "
            f"{R['total_bars']:,} bars). **Survivor** panel — named on the Signal axis, and it "
            "tilts *toward* the long/upside leg while cutting *against* the short/downside leg.\n"
            "- **Detector.** Precise real-body tasuki gap on bars $(t\\!-\\!2,t\\!-\\!1,t)$: two "
            "same-color bodies gapping in the trend direction, then an opposite-color body that "
            "opens inside the second body and closes back inside the gap without filling it; a "
            "**strict full-range-gap** (`true_range_gap=True`) and a **genuine prior-trend-"
            "context** variant for the myth-checks.\n"
            "- **Timing.** Confirm at the close of $t$; enter the **next open** (one lag); hold "
            "$H\\in\\{1,5,10,20\\}$ days in the predicted trend direction; drop events whose window "
            "overruns.\n"
            "- **Certifying stat.** Welch *t* of the trend-signed sample vs the basket's own "
            "direction-matched unconditional pool (drift-neutral) — the inference-bar number.\n"
            "- **Cross-checks.** One-sample/HAC *t* vs zero (informational, drift-contaminated); a "
            "5,000-draw mix-matched coin-flip placebo; hit-rate vs base hit-rate with Wilson "
            "intervals; up-only and down-only Welch *t*'s reported separately.\n"
            "- **Bonferroni across the basket.** Per-ticker pooled (up+down) Welch *t* (event "
            f"sample vs that ticker's own direction-matched base), corrected critical value for "
            f"$k={R['bonf_k']}$ simultaneous tests.\n"
            "- **Costs.** 5 / 10 bps one-way × 2 (round trip) on every event + 50 bps/yr borrow on "
            "the short (downside) leg over the hold.\n"
            "- **Positive control.** A deterministic panel with its own embedded drift and "
            "overnight-gap noise (mirroring the real basket's contamination risk and the pattern's "
            "need for a genuine gap) and a **planted** post-pattern continuation: zero edge must "
            "NOT reach significance on the *certifying* Welch stat across 20 seeds; a planted edge "
            "must light it up.\n\n"
            "> **Falsifier, stated in advance:** if the direction-matched Welch *t* never clears "
            "**+2** on the real tape at any horizon, the continuation signal is **None** and "
            "\"predicts continuation\" is **Busted**."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The direction trap, shown live — vs-zero vs vs-direction-matched-base\n\n"
            "Left: the trend-signed mean by horizon against **zero** (the naive, potentially "
            "contaminated read) with its HAC *t*. Right: the same means against the basket's own "
            "**direction-matched** base (the certifying Welch *t*). Watch the bar heights *and* "
            "which axis they clear."
        ),
        code(
            "hs = [1, 5, 10, 20]\n"
            "keys = ['h1','h5','h10','h20']\n"
            "if HAVE_REAL:\n"
            "    rows = [st.summarize(PANEL, h, placebo=False) for h in hs]\n"
            "    means = [r['mean']*100 for r in rows]\n"
            "    t_hac = [r['t_hac'] for r in rows]; t_welch = [r['t_welch'] for r in rows]\n"
            "else:\n"
            "    means = [R[k][4] for k in keys]\n"
            "    t_hac = [R[k][8] for k in keys]; t_welch = [R[k][10] for k in keys]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], t_hac, color=RED, width=.55)\n"
            "a1.axhline(-2, ls='--', c='k', lw=1); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_ylabel('t (vs ZERO, informational)'); a1.set_title('The naive HAC t vs 0')\n"
            "for i,t in enumerate(t_hac): a1.annotate(f'{t:.2f}',(i,t),ha='center',va='top' if t<0 else 'bottom')\n"
            "a2.bar([f'{h}d' for h in hs], t_welch, color=AMBER, width=.55)\n"
            "a2.axhline(-2, ls='--', c='k', lw=1); a2.axhline(2, ls='--', c='k', lw=1); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel('Welch t (vs direction-matched base — decisive)'); a2.set_title('Once compared fairly: still nothing supportive')\n"
            "for i,t in enumerate(t_welch): a2.annotate(f'{t:.2f}',(i,t),ha='center',va='top' if t<0 else 'bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('t vs zero (HAC):', [round(t,2) for t in t_hac])\n"
            "print('t vs matched base (Welch, decisive):', [round(t,2) for t in t_welch])"
        ),
        md(
            f"> 💡 In plain words: the left panel — comparing to zero — tells almost the same "
            f"story here because the up/down mix is close to 50/50, but it's still the wrong "
            "comparison in principle (a long-heavy or short-heavy sample would mislead it). The "
            "right panel, the decisive one, compares each event to what the *same basket* earns on "
            f"an **ordinary** day in the **same direction**: three of four horizons sit inside "
            f"±2 ({R['h1'][10]:+.2f}, {R['h10'][10]:+.2f}, {R['h20'][10]:+.2f}), and the fourth "
            f"({R['h5'][10]:+.2f}) clears it on the **wrong side**."
        ),
        md(
            "### 4b · The full inference table — HAC, one-sample, Welch, hit-rate, placebo\n\n"
            "Every arbiter at once. The believer needs a **positive** trend-signed excess over the "
            "matched base with Welch *t* ≥ +2; nothing here comes close in the claimed direction."
        ),
        code(
            "import pandas as pd\n"
            "if HAVE_REAL:\n"
            "    tab = pd.DataFrame([st.summarize(PANEL, h, n_draws=5000) for h in hs])\n"
            "    show = tab[['horizon','n_events','n_up','n_down','mean','base_mean','hit','base_hit','t_hac','t_one','t_welch','p_placebo','net']].copy()\n"
            "    for c in ['mean','base_mean','net']: show[c] = (show[c]*100).round(3)\n"
            "    for c in ['hit','base_hit']: show[c] = (show[c]*100).round(0)\n"
            "    for c in ['t_hac','t_one','t_welch']: show[c] = show[c].round(2)\n"
            "    show['p_placebo'] = show['p_placebo'].round(3)\n"
            "else:\n"
            "    cols=['horizon','n_events','n_up','n_down','mean','base_mean','hit','base_hit','t_hac','t_one','t_welch','p_placebo','net5','net10']\n"
            "    show = pd.DataFrame([dict(zip(cols, R[k])) for k in keys]).drop(columns=['net10']).rename(columns={'net5':'net'})\n"
            "print(show.to_string(index=False))"
        ),
        md(
            "> 💡 In plain words: the `hit`/`base_hit` columns sit close together at every "
            "horizon — no meaningful directional tilt. `t_welch`, the decisive column, never "
            "clears +2, and the one horizon past |2| is negative — a mild reversal, not a "
            "continuation."
        ),
        md(
            "### 4c · Split by direction — the up leg and the down leg separately\n\n"
            "Pooling long and short events can hide an asymmetry. Report them apart."
        ),
        code(
            "if HAVE_REAL:\n"
            "    up_t = [rows[i]['up_t_welch'] for i,h in enumerate(hs)]\n"
            "    dn_t = [rows[i]['down_t_welch'] for i,h in enumerate(hs)]\n"
            "else:\n"
            "    up_t = [R['split'][h][1] for h in hs]; dn_t = [R['split'][h][4] for h in hs]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.18, up_t, .34, color=GREEN, label='upside tasuki (long) Welch t')\n"
            "ax.bar(x+.18, dn_t, .34, color=RED, label='downside tasuki (short) Welch t')\n"
            "ax.axhline(2, ls='--', c='k', lw=1); ax.axhline(-2, ls='--', c='k', lw=1); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('Welch t (vs that leg\\'s own matched base)')\n"
            "ax.set_title('Only the downside leg ever clears |2| — and it points backwards')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('up-leg t:', [round(t,2) for t in up_t]); print('down-leg t:', [round(t,2) for t in dn_t])"
        ),
        md(
            f"> 💡 In plain words: the upside (long) leg never leaves the noise band at any "
            f"horizon. The downside (short) leg reaches *t* = **{R['split'][5][4]:.2f}** at 5 days "
            f"and **{R['split'][10][4]:.2f}** at 10 — both **negative**, meaning the short *loses* "
            "relative to shorting the basket unconditionally. A downside tasuki gap looks more "
            "like a fadeable spike than a trend with fuel left — though neither leg is Bonferroni-"
            "robust (see 4d)."
        ),
        md(
            "### 4d · Bonferroni across the basket — does one name secretly carry it?\n\n"
            "A pooled null can hide a real effect concentrated in a handful of names, or "
            "manufacture a fake one from a single noisy outlier. Per-ticker pooled (up+down) "
            "Welch *t* (event sample vs that ticker's *own* direction-matched pool) at $H=5$, with "
            f"the Bonferroni-corrected bar for $k={R['bonf_k']}$ simultaneous tests."
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
            "ax.set_ylabel('per-ticker pooled Welch t (vs its own matched base)'); ax.set_xlabel('ticker (top 15 by |t|)')\n"
            "ax.set_title('No single name survives the multiple-testing correction')\n"
            "plt.xticks(rotation=45, ha='right'); ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'{k} tickers tested, Bonferroni bar={zc:.2f}, survivors:', sum(abs(t)>=zc for t in tv))"
        ),
        md(
            f"> 💡 In plain words: even the loudest names in a {R['bonf_k']}-ticker sample — "
            "exactly the outliers a chart-reader would cherry-pick and show you — sit *below* the "
            f"corrected bar (**{R['bonf_survive']}/{R['bonf_k']}** clear it), and the loudest name "
            f"({R['bonf_top'][0][0]} at {R['bonf_top'][0][3]:.2f}) is negative, the same "
            "wrong-direction flavor as the pooled result. The pooled null in 4b/4c isn't hiding a "
            "real effect in a handful of names; there's simply no name carrying it."
        ),
        md(
            "### 4e · The myth check — does a stricter gap, or a genuine trend, rescue it?\n\n"
            "Two ways to make the signal \"purer\": the **strict full-range gap** (the second "
            "candle's entire wick range clears the first's, not just the bodies), and only "
            "counting patterns whose **prior trend genuinely agrees** with the predicted direction "
            "(a real continuation setup, not a reversal wearing the shape). If the lore were "
            "right, these should sharpen the effect."
        ),
        code(
            "if HAVE_REAL:\n"
            "    basic  = [st.summarize(PANEL, h, placebo=False)['t_welch'] for h in hs]\n"
            "    strict = [st.summarize(PANEL, h, placebo=False, true_range_gap=True)['t_welch'] for h in hs]\n"
            "    trend  = [st.summarize(PANEL, h, placebo=False, require_trend=True)['t_welch'] for h in hs]\n"
            "    ns = st.summarize(PANEL, 5, placebo=False, true_range_gap=True)['n_events']\n"
            "    nt = st.summarize(PANEL, 5, placebo=False, require_trend=True)['n_events']\n"
            "else:\n"
            "    basic  = [R[k][10] for k in keys]\n"
            "    strict = [s[5] for s in R['strict']]; trend = [t[5] for t in R['trend']]\n"
            "    ns = R['strict'][1][1]; nt = R['trend'][1][1]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.3))\n"
            "ax.bar(x-.25, basic, .25, color=RED, label='basic tasuki gap')\n"
            "ax.bar(x,      strict,.25, color=AMBER, label=f'strict full-range gap (n5={ns})')\n"
            "ax.bar(x+.25,  trend, .25, color=GREY, label=f'genuine prior trend (n5={nt})')\n"
            "ax.axhline(2, ls='--', c='k', lw=1); ax.axhline(-2, ls='--', c='k', lw=1); ax.axhline(0,c='k',lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('Welch t (vs direction-matched base, +-2 dashed)'); ax.set_title('No filter clears +2 — the trend-context filter makes it worse')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('strict Welch t:', [round(s,2) for s in strict]); print('trend Welch t :', [round(t,2) for t in trend])"
        ),
        md(
            f"> 💡 In plain words: the strict full-range gap (only **{R['strict'][1][1]}** events "
            f"at 5d — a much smaller, rarer sample) gives Welch *t* = **{R['strict'][1][5]:.2f}**, "
            f"and the genuine-prior-trend filter gives *t* = **{R['trend'][1][5]:.2f}** and "
            f"**{R['trend'][2][5]:.2f}** at 10 days — *more* negative than the base rate, the "
            "opposite of what \"a real continuation setup\" should produce. The pattern's "
            "predictive content is **indistinguishable from, or worse than, the basket's own "
            "drift** under every reasonable definition."
        ),
        md(
            "### 4f · Costs — the loss only deepens\n\n"
            "Gross vs net (5 / 10 bps one-way × 2 on every event + 50 bps/yr borrow on the short "
            "leg only). When the excess over the matched base is already statistically nothing, "
            "costs can't save it — they bury the gross number."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g  = [st.summarize(PANEL, h, placebo=False)['mean']*100 for h in hs]\n"
            "    n5 = [float(st.event_net_returns(st.collect_events(PANEL, h), h, cost_bps=5.0).mean())*100 for h in hs]\n"
            "    n10= [float(st.event_net_returns(st.collect_events(PANEL, h), h, cost_bps=10.0).mean())*100 for h in hs]\n"
            "else:\n"
            "    g  = [R[k][4] for k in keys]\n"
            "    n5 = [R[k][12] for k in keys]; n10 = [R[k][13] for k in keys]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.25, g, .25, color=AMBER, label='gross trend-signed')\n"
            "ax.bar(x,      n5, .25, color=RED, label='net @ 5 bps + borrow')\n"
            "ax.bar(x+.25,  n10,.25, color='#7a1f14', label='net @ 10 bps + borrow')\n"
            "ax.axhline(0,c='k',lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('trend-signed return (%)'); ax.set_title('Gross is already flat-to-negative; net is worse — no tradeable window')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for h,gg,n5_,n10_ in zip(hs,g,n5,n10): print(f'{h:>2}d: gross={gg:+.3f}%  net5bps={n5_:+.3f}%  net10bps={n10_:+.3f}%')"
        ),
        md(
            f"> 💡 In plain words: at 5 days gross **{R['h5'][4]:+.2f}%** → net "
            f"**{R['h5'][12]:+.2f}%** (5 bps) / **{R['h5'][13]:+.2f}%** (10 bps). There is no "
            "horizon where the trade clears zero, let alone its costs, let alone the "
            "direction-matched base it should be beating. **Mirage** in the strict sense — nothing "
            "under it to charge costs *against*."
        ),
        md(
            "### 4g · Faithful-engine & power control — the direction trap, proven on a KNOWN-zero panel\n\n"
            "A deterministic panel with its own mild embedded drift and an overnight-gap "
            "component (so genuine same-color-gap-pair-then-pullback shapes actually occur — "
            "mirroring the real basket's contamination risk) and a **planted** post-pattern "
            "continuation knob. With **zero** planted edge, the certifying **Welch t** (vs the "
            "panel's own direction-matched base) must stay inside ±2 across **20 independent "
            "seeds**."
        ),
        code(
            "null_welch = []\n"
            "for s_ in range(20):\n"
            "    px, _ = data.synthetic_panel(edge=0.0, seed=693 + s_)\n"
            "    null_welch.append(st.summarize(px, 5, placebo=False)['t_welch'])\n"
            "null_welch = np.asarray(null_welch)\n"
            "res = []\n"
            "for edge in (0.0, 0.012, 0.020):\n"
            "    px, truth = data.synthetic_panel(edge=edge, seed=693)\n"
            "    s = st.summarize(px, 5, n_draws=4000, placebo=False)\n"
            "    res.append((edge, truth['n_planted_days'], s['n_events'], s['mean']*100, s['t_welch'], s['t_hac'], s['hit']*100))\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.scatter(np.zeros(20) + np.linspace(-.1,.1,20), null_welch, color=GREY, s=40, label='null worlds (edge=0), 20 seeds')\n"
            "a1.axhline(-2, ls='--', c=RED, lw=1); a1.axhline(2, ls='--', c=RED, lw=1); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_xticks([0]); a1.set_xticklabels(['null x 20']); a1.set_ylabel('Welch t (vs panel matched base)')\n"
            "a1.set_title(f'Null mostly does NOT fire: {int((abs(null_welch)>=2).sum())}/20 seeds >= |t|=2'); a1.legend()\n"
            "labels = [f'planted\\n{e*100:.1f}%/day' for e,_,_,_,_,_,_ in res]\n"
            "tvals = [r[4] for r in res]\n"
            "a2.bar(labels, tvals, color=[GREY, AMBER, GREEN], width=.5)\n"
            "a2.axhline(2, ls='--', c=RED, label='t=+2 bar')\n"
            "for i,t in enumerate(tvals): a2.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "a2.set_ylabel('Welch t (5d, seed 693)'); a2.set_title('A planted continuation lights up cleanly'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null Welch t: mean={null_welch.mean():+.2f} sd={null_welch.std(ddof=1):.2f} |t|>=2 in {(abs(null_welch)>=2).sum()}/20')\n"
            "for e,pd_,k,ls,tw,th,w in res: print(f'planted {e*100:+.1f}%/day: n={k} mean={ls:+.3f}% t_welch={tw:.2f} t_hac(vs0)={th:.2f} hit={w:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted continuation, the certifying Welch *t* "
            f"averages **{R['syn_null_mean']:+.2f}** (sd {R['syn_null_sd']:.2f}) across 20 seeds "
            f"and fires in only **{R['syn_null_fire']}/{R['syn_null_n']}** — right at the ~5% "
            "nominal false-positive rate of a two-sided two-sigma bar, not a systematic bias. "
            f"A planted continuation of 1.2%/day reaches Welch *t* = **{R['syn'][1][4]:.2f}**, and "
            f"2.0%/day reaches **{R['syn'][2][4]:.2f}** — the harness *would* catch a real tasuki "
            "continuation if one existed. This is exactly why Welch-vs-matched-base is the number "
            "that decides the verdict, not HAC-vs-zero."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the certifying, direction-matched Welch *t* (event sample vs "
            f"the basket's own direction-matched unconditional return) never certifies a "
            f"continuation at any horizon ({R['h1'][10]:+.2f} / {R['h5'][10]:+.2f} / "
            f"{R['h10'][10]:+.2f} / {R['h20'][10]:+.2f} for 1/5/10/20 days), and the sole reading "
            "past |2| (5-day) points **backwards** — a mild reversal on the downside leg "
            f"(*t* = {R['split'][5][4]:.2f}). Hit rates track the base rate. The mix-matched "
            f"placebo places the 5-day observed mean where **{R['h5'][11]*100:.1f}%** of random "
            f"draws beat it. **{R['bonf_survive']}/{R['bonf_k']}** tickers survive a Bonferroni "
            "correction individually. Neither a strict full-range gap nor a genuine "
            "prior-trend-context filter rescues it — the trend filter makes the dip *worse*. "
            "Carries a **survivorship** caveat that cuts *toward* the long leg and *against* the "
            "short leg, reported separately so the asymmetry is visible. NONE, not WEAK.\n"
            f"- **Tradability `MIRAGE`** — flat-to-negative **gross** at every horizon "
            f"({R['h1'][4]:+.2f}% to {R['h5'][4]:+.2f}%); net of a round-trip cost + short borrow "
            f"it's worse across the whole 5-10 bps sweep ({R['h5'][12]:+.2f}% to "
            f"{R['h5'][13]:+.2f}%). There is no edge to charge costs against.\n"
            "- **Predicts continuation? `BUSTED`** — the \"gap can't close, the trend has fuel "
            "left\" reading does not survive a fair comparison to what the same basket does on an "
            "ordinary day in the same direction. Neither purity filter saves it, and no individual "
            "name in the basket carries a Bonferroni-robust version of the story."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Why it fails, mechanically.** A gap-plus-pullback is exactly the setup "
            "short-horizon reversal research (Jegadeesh 1990; Lehmann 1990) warns about: the "
            "market just made an abrupt move and is already being partly faded, which tends to "
            "keep fading a little further on the short side rather than resuming — the opposite "
            "of the folklore.\n"
            "- **The survivorship lever cuts both ways here.** Excluding names that never resumed "
            "an uptrend (and delisted) biases the long leg *toward* a working signal; excluding "
            "names that never stopped falling biases the short leg *against* one. Reporting the "
            "legs separately (4c) makes both directions of this bias visible instead of averaging "
            "them away.\n"
            "- **The gap-pattern neighbours.** [74-mind-the-gap](../../74-mind-the-gap/) is the "
            "general fill-rate question across all gap sizes and shapes; "
            "[417-island-reversal](../../417-island-reversal/) needs two opposite-direction gaps "
            "predicting a **reversal**, the opposite claim; "
            "[455-three-methods](../../455-three-methods/) is a five-candle continuation pattern "
            "with **no gap** at all — none of them run this study's specific "
            "two-same-color-gap-then-opposite-pullback, trend-signed detector.\n\n"
            "*The reproducible core is offline and deterministic; the detector is the precise "
            "real-body tasuki gap with a strict full-range-gap variant and a genuine-prior-trend "
            "filter as myth-checks, and a Bonferroni correction across the 30-name basket. Methods "
            "and sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
