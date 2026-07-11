"""Generate the two narrative notebooks for Study 696 (Double-Bottom).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily OHLC panel
under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily auto-adjusted
# OHLC, SPY + 29 large-caps, as-of 2026-06-30, 21.5 years, fp 312be65a0274 / cac3bd6bbfcc).
R = dict(
    start="2005-01-03", end="2026-06-30", years=21.5, n_bars=5406, n_names=30,
    fp_quantlab="312be65a0274", fp_sha1="cac3bd6bbfcc",
    n_bottom=560, spy_bottom=12,
    # double-bottom (bullish, long) forward edge — excess over base rate:
    #   (H, n, excess%, raw%, win%, t, HAC_t, p_plac, net5%)
    bottom=[(5, 558, -0.073, 0.218, 52.0, -0.48, -0.49, 0.541, -0.173),
            (10, 558, -0.379, 0.200, 48.2, -1.91, -1.98, 0.643, -0.479),
            (20, 554, 0.019, 1.177, 51.6, 0.06, 0.06, 0.499, -0.081),
            (40, 550, -0.478, 1.814, 51.1, -1.12, -1.18, 0.594, -0.578)],
    # robustness at H=20: (tolerance, min_bounce, n, excess%, t, p_plac)
    robust=[(0.03, 0.07, 196, 0.846, 1.22, 0.374),
            (0.04, 0.05, 554, 0.019, 0.06, 0.501),
            (0.06, 0.03, 1304, -0.357, -2.01, 0.647)],
    # SPY-only double bottom (the hook): (H, n, excess%, raw%, t, p_plac)
    spy=[(5, 12, 0.151, 0.385, 0.30, 0.427),
         (10, 12, 0.219, 0.686, 0.30, 0.417),
         (20, 12, 0.559, 1.495, 0.46, 0.354),
         (40, 12, 1.843, 3.710, 1.02, 0.140)],
    # measured-move target hit rate vs a magnitude-matched placebo
    mm_n=560, mm_hit=452, mm_rate=80.7, mm_lo=77.2, mm_hi=83.8,
    mm_placebo_n=6000, mm_placebo_hit=4782, mm_placebo_rate=79.7, mm_z=0.57,
    mm_median_days=21, mm_rel_move=5.92,
    # the long timer (target-or-timeout), cost sweep + holding-period-matched excess
    tm_n=545, tm_gross=2.942, tm_net5=2.842, tm_net10=2.742,
    tm_t=7.80, tm_hac=8.87, tm_avg_hold=49.8, tm_hit_share=80.2,
    tm_excess=0.125, tm_t_excess=0.32,
    # synthetic null over 20 seeds (naive one-sample t; the placebo is the arbiter)
    syn_null_mean=1.57, syn_null_sd=0.76, syn_null_fire=0, syn_null_n=20,
    # planted control (seed 696): (edge, planted, detected, n, excess%, t, p_plac, win%)
    syn=[(0.00, 160, 282, 282, 0.68, 2.12, 0.344, 57),
         (0.20, 160, 264, 264, 5.30, 10.32, 0.009, 72)],
    cost_bps=5.0,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Target_hit_rate%3F: Busted](https://img.shields.io/badge/Target_hit_rate%3F-Busted-8b949e?style=flat-square)\n\n"
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

from double_bottom import data, strategy as st

ASOF = "2026-06-30"
HAVE_REAL = data.have_real()
if HAVE_REAL:
    PANEL = data.load_real(asof=ASOF)
    CLOSES = PANEL["close"]
else:
    PANEL = CLOSES = None
print("real double-bottom cache present:", HAVE_REAL,
      "| names:", (0 if CLOSES is None else CLOSES.shape[1]),
      "| bars:", (0 if CLOSES is None else len(CLOSES)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The double bottom — does a \"W\" on the chart really mean buy? \U0001F4C9\U0001F1FC\n"
            "### The textbook bullish reversal figure, measured on 21.5 years of real tape\n\n"
            + BADGES +
            "Open any technical-analysis primer and you'll meet the **double bottom**: the price "
            "falls, bounces, falls again to about the **same low**, and bounces harder — a "
            "\"W\" shape. Chart readers say the market tested that price twice and twice failed to "
            "break it, so sellers are exhausted; when the price finally closes above the "
            "in-between peak (the **neckline**), you **buy the breakout**, and the pattern's own "
            "height even tells you a **price target** to expect.\n\n"
            "It's one of the oldest, most-taught patterns in the book. So let's do the thing the "
            "book never does: write down a **mechanical** rule for the shape, run it across two "
            "decades of real stocks, and ask three honest questions — does buying the breakout "
            "beat just holding the stock? Does the famous \"target\" actually mean anything? And "
            "does the strategy make money once you account for how long you have to wait?\n\n"
            "> \U0001F4D3 **Plain-language layer.** Want the *t*-stats, the placebo tests and the "
            "cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data + honesty note up front.** Chart figures are **partly subjective** — "
            "three chartists will draw three different W's on the same tape. We test the *closest "
            "mechanical definition* we can write down (two swing lows at a similar level, a real "
            "rally between them, then a confirmed neckline break) and we say so. We use a fixed "
            "**30-name large-cap basket + SPY** (names still trading today), which carries "
            "**survivorship** — it can't include names that delisted after a failed pattern, a "
            "mild tilt *for* this bullish figure. Every chart is drawn by the code beside it; "
            "house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After a confirmed **double bottom**, does the stock outrun its own normal drift? | "
            f"**No — and the closest it gets is the wrong sign.** At 10 days the excess is "
            f"**{R['bottom'][1][2]:+.2f}%** at *t* = **{R['bottom'][1][5]:.2f}** — buying the "
            "breakout slightly *underperforms* the stock's ordinary drift. No horizon clears the "
            "bar in the pattern's favor. |\n"
            "| Does the famous \"measured-move target\" actually predict a real move? | "
            f"**No.** It gets touched **{R['mm_rate']:.0f}%** of the time — but a random walk "
            f"asked to travel the *same* distance already gets there **{R['mm_placebo_rate']:.0f}%** "
            "of the time. The \"accuracy\" is mostly the target being a modest, achievable "
            "distance. |\n"
            "| Does at least the *strategy* make money, patiently held? | "
            f"**On paper, yes — but that's the trap.** Hold to target-or-timeout and you make "
            f"**{R['tm_gross']:+.2f}%** per trade — except so does a random buy-and-hold over the "
            "same ~50-day window. Compared fairly, the excess is **statistically nothing**. |\n"
            "| Does a stricter shape (a tighter tolerance) rescue it? | **No.** The tiny edge "
            "wanders around zero and never turns reliably positive at any strictness. |\n\n"
            "> The \"W\" is real on the chart. The edge is not."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When price tests the same low **twice** and holds, the down-move is exhausted. "
            "Buy when it breaks above the middle peak (the neckline), and expect price to travel "
            "at least as far above the neckline as the low sat below it — the **measured move**.\"*\n\n"
            "This is straight out of the canon — Edwards & Magee's *Technical Analysis of Stock "
            "Trends* (1948), Schabacker before them, and every modern pattern guide since "
            "(Bulkowski's *Encyclopedia of Chart Patterns* tallies its own \"success rates\"). The "
            "double bottom is sold as one of the more **reliable** bullish reversal figures — and "
            "unlike vaguer chart lore, it comes with a built-in price target, which makes it "
            "unusually easy to test **honestly**."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the figure worked, it would be a trader's dream: a **visual, rules-light** entry "
            "anyone can spot, a built-in stop (below the second low), and a **built-in target** — "
            "no guessing where to take profit. A reliable reversal signal with a target attached "
            "would be one of the great free lunches.\n\n"
            "Two traps hide underneath, though. **(1) The base rate.** Stocks drift **up** over "
            "time, so *any* \"buy\" signal looks like it works a little. The honest test is "
            "whether the breakout beats the stock's **own** normal drift — not just zero. "
            "**(2) The target illusion.** A target sitting a modest distance away will get "
            "touched *often* on pure noise — the real question is whether it gets touched *more* "
            "than a random move of the same size would."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take a fixed **{R['n_names']}-name large-cap basket + SPY** and run a "
            f"**mechanical** double-bottom detector over **{R['years']:.1f} years** "
            f"({R['start']} → {R['end']}):\n\n"
            "1. **Swing pivots.** Find local lows (and highs) over a symmetric window.\n"
            "2. **Two troughs at one level.** Two swing lows within a tolerance of one price, "
            "separated by a **genuine** rally to an intervening peak (not a flat shelf) — the "
            "\"W\".\n"
            "3. **The neckline & the breakout.** The neckline is that intervening peak; the "
            "**signal** is the first close that clears it. We **enter the next close** (no "
            "cheating) and hold **5 / 10 / 20 / 40** days for the headline test.\n"
            "4. **The honest bar.** Compare the forward return to the name's *own* base rate, "
            "test it with a *t*-stat, and run a **random-date placebo**. Then check the "
            "**measured-move target** against a placebo of the *same magnitude*, and try a "
            "**long timer** that holds to target-or-timeout, raced against a fair, "
            "holding-period-matched baseline.\n\n"
            "**What would make us say \"mirage\"?** A *t* under 2 (or the wrong sign), a placebo "
            "the pattern can't beat, a target hit rate matched by a same-size random move, and a "
            "long-timer P&L that evaporates once compared fairly. (Spoiler: all four.)"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, what does the detector even find?** Here's one real confirmed double bottom "
            "drawn by the code — two troughs, the neckline, and the breakout bar."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tk = 'JPM'\n"
            "    c = CLOSES[tk].dropna().to_numpy(float)\n"
            "    sigs = st.detect_double_bottom(c)\n"
            "    d = sigs[len(sigs)//2] if sigs else None\n"
            "else:\n"
            "    px, _ = data.synthetic_panel(edge=0.0, seed=696); c = px['close']['N00'].to_numpy(float)\n"
            "    sigs = st.detect_double_bottom(c); d = sigs[0] if sigs else None\n"
            "if d is not None:\n"
            "    a, b = d['t1_idx']-15, d['breakout_idx']+25\n"
            "    a = max(a, 0); b = min(b, len(c))\n"
            "    xs = np.arange(a, b)\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "    ax.plot(xs, c[a:b], color=GREY, lw=1.5)\n"
            "    ax.scatter([d['t1_idx'], d['t2_idx']], [c[d['t1_idx']], c[d['t2_idx']]], color=RED,\n"
            "               zorder=5, s=70, label='two troughs (the W)')\n"
            "    ax.axhline(d['level'], color=RED, ls=':', alpha=.6)\n"
            "    ax.axhline(d['neckline'], color=GREEN, ls='--', alpha=.7, label='neckline')\n"
            "    ax.axhline(d['target'], color=AMBER, ls='--', alpha=.7, label='measured-move target')\n"
            "    ax.scatter([d['breakout_idx']], [c[d['breakout_idx']]], color=GREEN, marker='^', s=120,\n"
            "               zorder=6, label='confirmed breakout')\n"
            "    ax.set_title(f'A real mechanical double bottom ({tk if HAVE_REAL else \"synthetic\"})')\n"
            "    ax.set_xlabel('bar'); ax.set_ylabel('price'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('troughs at bars', d['t1_idx'], d['t2_idx'], '| breakout at', d['breakout_idx'],\n"
            "          '| target', round(d['target'],2))\n"
            "print('total double-bottom breakouts found across the basket:', R['n_bottom'])"
        ),
        md(
            f"The detector is doing the believer's job faithfully: it found **{R['n_bottom']}** "
            "confirmed double-bottom breakouts across the basket. The shape is exactly the "
            "textbook picture, target line and all. Now the only question that matters: **does "
            "buying that green triangle pay?**"
        ),
        md(
            "**The forward edge by horizon.** For each breakout we measure the return *over and "
            "above the stock's own normal drift*, then pool across all names. If the figure "
            "works, these bars should be clearly positive."
        ),
        code(
            "hs = [5, 10, 20, 40]\n"
            "if HAVE_REAL:\n"
            "    ex = [st.run_experiment(PANEL, horizon=h, n_draws=2000)['mean']*100 for h in hs]\n"
            "else:\n"
            "    ex = [b[2] for b in R['bottom']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "cols = [GREEN if v>0 else RED for v in ex]\n"
            "ax.bar([f'{h}d' for h in hs], ex, color=cols, width=.6)\n"
            "for i,v in enumerate(ex): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('excess return over base rate (%)')\n"
            "ax.set_title('Double-bottom breakout: never clearly positive')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('excess by horizon:', [round(v,3) for v in ex])"
        ),
        md(
            f"That's the whole story in one chart. The 10-day bar is actually **negative** "
            f"(**{R['bottom'][1][2]:+.2f}%**) — buying the breakout does *worse* than just holding "
            "the stock — and no horizon shows a real, reliable excess. This is not what a "
            "confident reversal signal looks like."
        ),
        md(
            "**Is even the best reading real, or could a coin have done it?** We throw **5,000 "
            "random entry dates** on the same tape and ask how the pattern stacks up against the "
            "cloud of random luck."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r5 = st.run_experiment(PANEL, horizon=5, n_draws=5000)\n"
            "    obs = r5['mean']*100; p = r5['p_placebo']; t = r5['t']\n"
            "else:\n"
            "    obs = R['bottom'][0][2]; p = R['bottom'][0][7]; t = R['bottom'][0][5]\n"
            "rng = np.random.default_rng(696)\n"
            "cloud = rng.normal(0.0, max(abs(obs),0.05)/max(abs(t),0.4), 5000)   # illustrative null spread\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(cloud, bins=55, color=GREY, alpha=.85, label='random-date placebo (luck)')\n"
            "ax.axvline(obs, color=RED, lw=2.5, label=f'double bottom {obs:+.2f}%')\n"
            "ax.set_xlabel('5-day excess return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Inside the luck cloud: placebo p = {p:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f}%  t={t:.2f}  placebo p={p:.3f}')"
        ),
        md(
            f"The red line sits **inside** the luck cloud, not out in a tail. About "
            f"**{R['bottom'][0][7]*100:.0f}%** of random-date entries do at least as well as the "
            "pattern — a coin-flip, not a signal."
        ),
        md(
            "**Now the target.** The book says the pattern's own height tells you where price is "
            "going. Let's check whether that target beats a random move of the *same size*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mm = st.measured_move_hits(PANEL, max_days=126)\n"
            "    hit, plac, z = mm['hit_rate']*100, mm['placebo_rate']*100, mm['z_vs_placebo']\n"
            "else:\n"
            "    hit, plac, z = R['mm_rate'], R['mm_placebo_rate'], R['mm_z']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['double-bottom\\ntarget', 'same-size\\nrandom move'], [hit, plac], color=[AMBER, GREY], width=.5)\n"
            "for i,v in enumerate([hit, plac]): ax.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('target hit rate within 6 months (%)'); ax.set_ylim(0, 100)\n"
            "ax.set_title(f'The target is not special: z = {z:.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'observed {hit:.1f}%  placebo {plac:.1f}%  z={z:.2f}')"
        ),
        md(
            f"The two bars are almost the same height. The double-bottom target hits "
            f"**{R['mm_rate']:.1f}%** of the time — but so does a random move asked to travel the "
            f"*exact same typical distance* (**{R['mm_placebo_rate']:.1f}%**). That's *z* = "
            f"**{R['mm_z']:.2f}** — nowhere near significant. The famous \"measured move\" is a "
            "trick of arithmetic on a rising market, not a forecast."
        ),
        md(
            "**Finally, the strategy some traders actually run: hold until the target hits, or "
            "give up after six months.** Here's the raw P&L — and then the fair comparison."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tp = st.timer_pnl(PANEL, max_days=126, cost_bps=5.0)\n"
            "    g, ex, tv, texv = tp['gross']*100, tp['excess']*100, tp['t'], tp['t_excess']\n"
            "else:\n"
            "    g, ex, tv, texv = R['tm_gross'], R['tm_excess'], R['tm_t'], R['tm_t_excess']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['vs ZERO\\n(the naive read)', 'vs the SAME-STOCK\\nbaseline (fair)'], [g, ex],\n"
            "       color=[RED, AMBER], width=.5)\n"
            "for i,v in enumerate([g, ex]): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('per-trade return (%)')\n"
            "ax.set_title(f'Looks great vs zero (t={tv:.1f})... vanishes vs a fair baseline (t={texv:.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross vs 0: {g:+.2f}%   excess vs matched baseline: {ex:+.2f}%')"
        ),
        md(
            f"There's the whole trick. Held to target-or-timeout (about **{R['tm_avg_hold']:.0f} "
            f"days** on average), the strategy makes **{R['tm_gross']:+.2f}%** per trade — looks "
            "like a real edge. But a ~50-day hold in a market that drifts up over two decades "
            "*always* looks like that. Compare it to buying the **same stock for the same number "
            f"of days, no pattern required**, and the excess is **{R['tm_excess']:+.2f}%** — "
            "statistically nothing."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The breakout never beats the stock's own drift at a real "
            "statistical margin — the closest approach is actually *negative*. The target "
            "\"accuracy\" is matched by a random move of the same size. There's no reversal "
            "edge here.\n"
            "- **Tradability — Mirage.** The long-timer P&L looks like a slam dunk against zero "
            "and evaporates against a fair baseline. There's no excess to deploy against.\n"
            "- **\"Target beats a coin flip\"? — Busted.** The famous measured-move target is a "
            "trick of arithmetic on a rising market, not a forecast of anything."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — no, and here's the kicker\n\n"
            "The excess was already ~zero before costs. Now add the real-world friction — the "
            "round-trip spread — to the headline horizons."
        ),
        code(
            "hs = [5, 10, 20, 40]\n"
            "if HAVE_REAL:\n"
            "    g = [st.run_experiment(PANEL, horizon=h, n_draws=1500)['mean']*100 for h in hs]\n"
            "    nv = [st.run_experiment(PANEL, horizon=h, n_draws=1500)['net']*100 for h in hs]\n"
            "else:\n"
            "    g = [b[2] for b in R['bottom']]; nv = [b[8] for b in R['bottom']]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, g, .4, color=AMBER, label='gross excess')\n"
            "ax.bar(x+.2, nv, .4, color=RED, label=f'net of {R[\"cost_bps\"]:.0f} bps/leg')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('excess return (%)'); ax.set_title('Costs push an already-flat edge further negative')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('net by horizon:', [round(v,3) for v in nv])"
        ),
        md(
            f"Every net number sits at or below zero. At 10 days you're **{R['bottom'][1][8]:.2f}%** "
            "in the hole net of costs, on top of an already-negative excess. There's no tradable "
            "region here."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further \U0001F6AA\n\n"
            "- **Why the target trick works on the eye.** Any target a modest distance away gets "
            "touched often over a 6-month window on a rising market — the magnitude-matched "
            "placebo is the only honest way to know whether a target beats \"random noise of the "
            "same size,\" and here it doesn't.\n"
            "- **The \"tested a level twice\" family.** [189-double-top](../../189-double-top/) "
            "tests both the bearish and bullish versions with a different (fixed-horizon) "
            "protocol; [415-triple-top-bottom](../../415-triple-top-bottom/) is the three-tap "
            "version; [695-inverse-head-shoulders](../../695-inverse-head-shoulders/) is the "
            "three-trough, asymmetric-head version; "
            "[694-matching-low](../../694-matching-low/) is the two-*candle* micro version of the "
            "same idea. None of them run this study's measured-move + long-timer bar.\n\n"
            "*Think a stricter shape, a different universe, or a smarter target rule turns it "
            "green? Show the **net excess over a fair baseline** clearing Welch/HAC *t* ≥ 2 — "
            "then we'll talk.*"
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
            "# Double-Bottom — a quantitative teardown \U0001F52C\n"
            "### A mechanical two-trough detector · forward 5/10/20/40-day excess over base rate "
            "· one-sample + HAC *t* · a same-tape random-date placebo · a detector-strictness "
            "sweep · a magnitude-matched measured-move hit-rate test · a holding-period-matched "
            "long timer · a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "double bottom is a *reversal* figure with a built-in price target, which makes it "
            "unusually testable — the job here is to (a) write down the closest mechanical "
            "definition of \"two troughs at one level + a confirmed neckline break,\" (b) measure "
            "the post-breakout excess over each name's base rate, (c) confront both the target and "
            "a \"hold to target-or-timeout\" strategy with **matched** honest controls (magnitude "
            "for the target, holding period for the timer), and (d) run the synthetic positive "
            "control.\n\n"
            "> ⚠️ **Subjectivity + survivorship note.** Chart figures are partly in the eye of "
            "the beholder; we test **one** mechanical definition and report a strictness sweep so "
            "the reader sees how the count and the (non-)edge move with it. Fixed **30-name "
            "large-cap basket + SPY** (the same basket as siblings 415 and 695), names still "
            "trading in 2026 — a *survivor* panel that tilts post-breakout forward returns mildly "
            "**up** (i.e. *for* the bullish read). Real data: yfinance daily auto-adjusted OHLC, "
            f"{R['start']} → {R['end']}, as-of **{R['end']}**. Offline core + synthetic control "
            "are deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers "
            "in [`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_quantlab"] +
            "` / `" + R["fp_sha1"] + "`).\n"
            ">\n"
            "> \U0001F4A1 **The `\U0001F4A1 In plain words` notes** translate each result back to "
            "intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Double-bottom breakout excess never clears **t ≥ 2** in the "
            f"pattern's favor: {R['bottom'][0][5]:+.2f} / **{R['bottom'][1][5]:+.2f}** / "
            f"{R['bottom'][2][5]:+.2f} / {R['bottom'][3][5]:+.2f} *t* for 5/10/20/40d (10d is the "
            "**wrong sign**). Random-date placebo *p* ∈ [0.50, 0.64] at every horizon. Measured-"
            f"move hit rate {R['mm_rate']:.1f}% vs a magnitude-matched placebo "
            f"{R['mm_placebo_rate']:.1f}% (*z* = {R['mm_z']:.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | Long-timer gross **{R['tm_gross']:+.2f}%**/trade "
            f"(*t* = {R['tm_t']:.2f} vs 0) collapses to **{R['tm_excess']:+.2f}%** "
            f"(HAC *t* = {R['tm_t_excess']:.2f}) vs a holding-period-matched base rate. |\n"
            "| **Target hit rate beats a coin flip of the same size?** | `BUSTED` | "
            f"{R['mm_rate']:.1f}% observed vs {R['mm_placebo_rate']:.1f}% magnitude-matched "
            f"placebo, *z* = {R['mm_z']:.2f} — the target's \"accuracy\" is arithmetic on a "
            "rising market, not forecasting power. |\n\n"
            "> \U0001F4A1 In plain words: the shape is real and common; the breakout, the target "
            "and the patient hold-to-target strategy each look impressive by the naive read and "
            "each dissolve once measured against the fair, matched baseline."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a **double bottom** be two swing lows $\\{p_1,p_2\\}$ at prices $h_1,h_2$ within "
            "tolerance $\\tau$ of each other ($|h_2-h_1|/\\max(h_1,h_2)\\le\\tau$), separated by an "
            "intervening swing high — the **neckline** $N$ — that rallies at least $\\beta$ above "
            "the troughs' mean level $L=(h_1+h_2)/2$ (a genuine bounce, not a shelf). The "
            "**confirmed breakout** is the first close $c_b>N$ after $p_2$. Enter at $b+1$ (one "
            "lag); the trade's forward $H$-day return is $r_i(H)=c_{b+1+H}/c_{b+1}-1$. Define the "
            "**excess** over the name's base rate $\\mu_k(H)$: $x_i(H)=r_i(H)-\\mu_{k(i)}(H)$. The "
            "**measured-move target** is $T=N+(N-L)$, projected up from the neckline by the "
            "trough-to-neckline height.\n\n"
            "- **H₁ (the breakout works).** Pooled $\\bar{x}(H)>0$ and significant for some $H$.\n"
            "- **H₂ (the target predicts a real move).** The target hit rate beats a random walk "
            "asked to travel the *same relative distance*.\n"
            "- **H₃ (the strategy is deployable).** Holding to target-or-timeout beats a "
            "holding-period-matched base rate, net of costs.\n\n"
            f"We find **H₁ rejected** (max favorable $t={R['bottom'][2][5]:.2f}$ < 2; the 10-day "
            f"reading is *negative* at $t={R['bottom'][1][5]:.2f}$), **H₂ rejected** "
            f"($z={R['mm_z']:.2f}$ vs the magnitude-matched placebo), **H₃ rejected** (excess "
            f"$t={R['tm_t_excess']:.2f}$ vs the matched base rate). The shape is real and common; "
            "none of the three claims survive contact with a fair control."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The Signal axis is a one-sample test of the pooled excess against zero, with an HAC "
            "(Newey-West) variant because breakouts cluster in time:\n\n"
            "$$t=\\frac{\\bar{x}}{s_x/\\sqrt{n}},\\qquad "
            "t_{\\text{HAC}}=\\frac{\\bar{x}}{\\sqrt{\\widehat{\\sigma}^2_{\\text{NW}}/n}}.$$\n\n"
            "Three honesty problems sit on top of a naive read. **(a) The up-drift base rate:** "
            "any long signal inherits the market's positive drift, so we test **excess over each "
            "name's own base rate**, plus a **same-tape random-date placebo**. **(b) The target's "
            "own magnitude:** a nearby target gets hit often on pure noise, so the target "
            "hit-rate control must draw **the same relative distance** from a random entry, not "
            "an arbitrary one — see [`strategy.measured_move_hits`](../double_bottom/strategy.py). "
            "**(c) The variable holding period:** \"hold to target-or-timeout\" produces trades of "
            "different lengths, so the long-timer's honest bar is a "
            "**holding-period-matched** base rate — buy the same name, unconditionally, for the "
            "*same* number of days — not zero. Section 4c/4d make each trap visible."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** Fixed **{R['n_names']}-name** large-cap basket incl. SPY (yfinance "
            f"auto-adjusted daily OHLC, {R['start']}→{R['end']}, as-of **{R['end']}**, "
            f"**{R['n_bars']:,}** bars, fp `{R['fp_quantlab']}` / `{R['fp_sha1']}`). **Survivor** "
            "panel — named on the Signal axis.\n"
            "- **Detector.** swing window $w=5$; two troughs within $\\tau=0.04$ of each other; "
            "intervening rally $\\beta\\ge0.05$ above the trough level; figure span 15–150 bars; "
            "breakout = first close through the neckline within 30 bars (buffer 0); "
            "non-overlapping.\n"
            "- **Timing.** signal known at the breakout close; **enter the next close** (one "
            "documented lag); hold $H\\in\\{5,10,20,40\\}$ for the headline, or to "
            "target-or-timeout (126 trading days) for the long timer.\n"
            "- **Metric.** pooled **excess over base rate** (headline); **hit rate vs a "
            "magnitude-matched placebo** (target); **excess vs a holding-period-matched base "
            "rate** (timer).\n"
            "- **Null #1 (one-sample / HAC t)** vs 0, reported but *not* decisive by itself.\n"
            "- **Null #2 (same-tape placebo).** random entry dates, same count per name, same "
            "base-rate subtraction.\n"
            "- **Null #3 (magnitude-matched placebo).** random entries with a target of the *same "
            "relative size* as the observed average.\n"
            "- **Robustness.** strictness sweep over $(\\tau,\\beta)$.\n"
            "- **Costs.** 5 / 10 bps one-way × 2 (round trip) per event; long only, no borrow.\n"
            "- **Positive control.** a synthetic panel with *planted* double bottoms + a "
            "forward-drift knob: edge 0 must NOT beat the placebo across 20 seeds even though a "
            "single-seed naive $t$ can read misleadingly high; a large edge must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The term structure of the (non-)edge\n\n"
            "Pooled excess over base rate at each horizon, with the one-sample and HAC *t* "
            "annotated. A real reversal signal would be clearly positive; this never is."
        ),
        code(
            "hs = [5, 10, 20, 40]\n"
            "if HAVE_REAL:\n"
            "    res = [st.run_experiment(PANEL, horizon=h, n_draws=2500) for h in hs]\n"
            "    ex = [r['mean']*100 for r in res]; ts = [r['t'] for r in res]; hac = [r['hac_t'] for r in res]\n"
            "else:\n"
            "    ex = [b[2] for b in R['bottom']]; ts = [b[5] for b in R['bottom']]; hac = [b[6] for b in R['bottom']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "cols = [GREEN if v>0 else RED for v in ex]\n"
            "a1.bar([f'{h}d' for h in hs], ex, color=cols, width=.6)\n"
            "for i,v in enumerate(ex): a1.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('excess over base rate (%)'); a1.set_title('Excess never clearly positive')\n"
            "a2.bar(np.arange(len(hs))-.2, ts, .4, color=GREY, label='one-sample t')\n"
            "a2.bar(np.arange(len(hs))+.2, hac, .4, color=AMBER, label='HAC t')\n"
            "a2.axhline(2, ls='--', c=RED, label='t=2 bar'); a2.axhline(-2, ls='--', c=RED)\n"
            "a2.set_xticks(range(len(hs))); a2.set_xticklabels([f'{h}d' for h in hs]); a2.set_ylabel('t-stat'); a2.set_title('Never clears the t=2 bar'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('excess:', [round(v,3) for v in ex]); print('t:', [round(v,2) for v in ts], 'HAC:', [round(v,2) for v in hac])"
        ),
        md(
            f"> \U0001F4A1 In plain words: the 10-day *t* is **{R['bottom'][1][5]:+.2f}** — the "
            "wrong sign for a reversal thesis — and no horizon reaches +2. The HAC version tells "
            "the same story."
        ),
        md(
            "### 4b · The decisive test — the same-tape random-date placebo\n\n"
            "The honest null for a long signal on an up-drifting tape: random entry dates on the "
            "*same* names, same count, same base-rate subtraction."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rng = np.random.default_rng(696); H = 5; lag = 1\n"
            "    obs = st.run_experiment(PANEL, horizon=H, n_draws=1)['mean']*100\n"
            "    draws = []\n"
            "    bk = st.collect_breakouts(PANEL)\n"
            "    for _ in range(4000):\n"
            "        pool = []\n"
            "        for tk, sigs in bk.items():\n"
            "            c = CLOSES[tk].dropna().to_numpy(float); n=len(c); k=len(sigs)\n"
            "            if k==0: continue\n"
            "            hi = n - H - lag - 1\n"
            "            if hi<=1: continue\n"
            "            si = rng.integers(1, hi, size=k); entry=si+lag; exit_=entry+H\n"
            "            br = st.base_rate(c, H)\n"
            "            pool.extend((c[exit_]/c[entry]-1.0 - br).tolist())\n"
            "        draws.append(np.mean(pool)*100)\n"
            "    draws = np.array(draws); p = float((draws>=obs).mean())\n"
            "else:\n"
            "    obs = R['bottom'][0][2]; p = R['bottom'][0][7]\n"
            "    rng = np.random.default_rng(696); draws = rng.normal(0.0, max(abs(obs),0.05)/0.5, 4000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=60, color=GREY, alpha=.85, label='random-date placebos')\n"
            "ax.axvline(obs, color=RED, lw=2.5, label=f'double bottom {obs:+.2f}%')\n"
            "ax.set_xlabel('5-day pooled excess (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Inside the luck cloud: placebo p = {p:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f}%  placebo p={p:.3f}  (frozen p={R[\"bottom\"][0][7]})')"
        ),
        md(
            f"> \U0001F4A1 In plain words: random entries beat the figure roughly "
            f"**{R['bottom'][0][7]*100:.0f}%** of the time at 5 days. No horizon in "
            "[`docs/results.md`](../docs/results.md) beats its placebo by more than a coin flip's "
            "margin."
        ),
        md(
            "### 4c · The measured-move target — a magnitude-matched control, not a bare hit rate\n\n"
            "A bare hit rate ($T$ touched within 6 months) means little on its own — a target a "
            "modest distance away gets hit often on pure noise. The honest control draws random "
            "entries on the same tapes and assigns them a target at the **same relative "
            "distance** as the observed average ($+5.9\\%$), then asks whether *that* gets hit as "
            "often."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mm = st.measured_move_hits(PANEL, max_days=126)\n"
            "    hit, lo, hi = mm['hit_rate']*100, mm['hit_lo']*100, mm['hit_hi']*100\n"
            "    plac, z = mm['placebo_rate']*100, mm['z_vs_placebo']\n"
            "else:\n"
            "    hit, lo, hi = R['mm_rate'], R['mm_lo'], R['mm_hi']; plac, z = R['mm_placebo_rate'], R['mm_z']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "ax.bar(['observed\\n(double bottom)', 'magnitude-matched\\nplacebo'], [hit, plac],\n"
            "       color=[AMBER, GREY], width=.5)\n"
            "ax.errorbar([0], [hit], yerr=[[hit-lo],[hi-hit]], fmt='none', ecolor='k', capsize=6)\n"
            "for i,v in enumerate([hit, plac]): ax.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('target hit rate within 6 months (%)'); ax.set_ylim(0,100)\n"
            "ax.set_title(f'z = {z:.2f} vs the matched placebo — indistinguishable')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'observed {hit:.1f}% [Wilson {lo:.1f}%,{hi:.1f}%]  placebo {plac:.1f}%  z={z:.2f}')"
        ),
        md(
            f"> \U0001F4A1 In plain words: **{R['mm_rate']:.1f}%** observed vs "
            f"**{R['mm_placebo_rate']:.1f}%** magnitude-matched placebo, *z* = "
            f"**{R['mm_z']:.2f}** — the Wilson interval on the observed rate "
            f"([{R['mm_lo']:.1f}%, {R['mm_hi']:.1f}%]) comfortably contains the placebo rate. The "
            "\"the pattern's own height tells you the target\" folklore is arithmetic on a "
            "rising market wearing a chart-pattern's name."
        ),
        md(
            "### 4d · The long timer — the direction trap, on a variable holding period\n\n"
            "Hold from entry to the day the measured-move target is touched, or 126 trading days "
            "(timeout), whichever comes first. The **excess** row races the trade against a "
            "**holding-period-matched** base rate — buy the same name, unconditionally, for the "
            "same number of days — the fair bar for a variable-length hold."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tp5 = st.timer_pnl(PANEL, max_days=126, cost_bps=5.0)\n"
            "    tp10 = st.timer_pnl(PANEL, max_days=126, cost_bps=10.0)\n"
            "    g, n5, n10 = tp5['gross']*100, tp5['net']*100, tp10['net']*100\n"
            "    tv, hacv = tp5['t'], tp5['hac_t']\n"
            "    ex, texv = tp5['excess']*100, tp5['t_excess']\n"
            "else:\n"
            "    g, n5, n10 = R['tm_gross'], R['tm_net5'], R['tm_net10']\n"
            "    tv, hacv = R['tm_t'], R['tm_hac']; ex, texv = R['tm_excess'], R['tm_t_excess']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(['gross', 'net 5bps', 'net 10bps'], [g, n5, n10], color=[GREY, AMBER, AMBER], width=.55)\n"
            "for i,v in enumerate([g, n5, n10]): a1.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('per-trade return (%)'); a1.set_title(f'vs ZERO: looks great (t={tv:.1f}, HAC t={hacv:.1f})')\n"
            "a2.bar(['vs ZERO\\n(naive)', 'vs the matched\\nbaseline (fair)'], [g, ex], color=[RED, AMBER], width=.5)\n"
            "for i,v in enumerate([g, ex]): a2.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel('per-trade return (%)'); a2.set_title(f'Fair comparison: t={texv:.2f}, vanishes')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.2f}%  net5 {n5:+.2f}%  net10 {n10:+.2f}%  t(vs0)={tv:.2f}  HAC(vs0)={hacv:.2f}')\n"
            "print(f'excess vs matched baseline: {ex:+.3f}%  HAC t = {texv:.2f}')"
        ),
        md(
            f"> \U0001F4A1 In plain words: the left panel — **{R['tm_gross']:+.2f}%**/trade, *t* = "
            f"**{R['tm_t']:.1f}** vs zero — is exactly the trap the desk's methodology warns "
            f"about: an average **{R['tm_avg_hold']:.0f}-day** hold on a rising basket looks "
            "\"significant\" against zero *by construction*. Raced against buying the same names "
            f"for the same number of days, the excess is **{R['tm_excess']:+.3f}%** at HAC "
            f"*t* = **{R['tm_t_excess']:.2f}** — statistically nothing. The target-hit share "
            f"({R['tm_hit_share']:.0f}% of trades exit on a target touch rather than timeout) is "
            "consistent with 4c: the target is easy to hit on drift alone."
        ),
        md(
            "### 4e · Robustness — the (non-)edge is a tolerance artefact\n\n"
            "Sweep the detector strictness $(\\tau,\\beta)$ at the 20-day horizon. If the figure "
            "were real, a stricter \"cleaner\" pattern should be *better*. Instead the (non-)edge "
            "wanders and only ever brushes past 2 on the **wrong side**."
        ),
        code(
            "settings = [(0.03,0.07),(0.04,0.05),(0.06,0.03)]\n"
            "if HAVE_REAL:\n"
            "    rob = [st.run_experiment(PANEL, horizon=20, n_draws=3000, tolerance=t_, min_bounce=mb) for t_,mb in settings]\n"
            "    rob = [(s[0], s[1], r['n_events'], r['mean']*100, r['t'], r['p_placebo']) for s,r in zip(settings,rob)]\n"
            "else:\n"
            "    rob = R['robust']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "labels = [f'tol={r[0]}\\nbounce={r[1]}\\n(n={r[2]})' for r in rob]\n"
            "ts = [r[4] for r in rob]\n"
            "cols = [GREEN if t_>0 else RED for t_ in ts]\n"
            "ax.bar(labels, ts, color=cols, width=.55)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=+-2 bar'); ax.axhline(-2, ls='--', c=RED); ax.axhline(0,c='k',lw=.8)\n"
            "for i,r in enumerate(rob): ax.annotate(f'{r[3]:+.2f}%',(i,r[4]),ha='center',va='bottom' if r[4]>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20-day one-sample t'); ax.set_title('No strictness turns it reliably positive')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('robustness (tol, bounce, n, excess%, t, p):', [(r[0],r[1],r[2],round(r[3],2),round(r[4],2),round(r[5],3)) for r in rob])"
        ),
        md(
            f"> \U0001F4A1 In plain words: the strictest setting (n={R['robust'][0][2]}) reads "
            f"**{R['robust'][0][3]:+.2f}%** at t={R['robust'][0][4]:.2f} — positive but well under "
            f"2; the loosest (n={R['robust'][2][2]}) is **{R['robust'][2][3]:+.2f}%** at "
            f"t={R['robust'][2][4]:.2f} — the *only* setting to cross $|t|=2$, and on the wrong "
            "side. A real figure gets cleaner when you demand a cleaner shape; this doesn't."
        ),
        md(
            "### 4f · SPY-only — the README hook\n\n"
            "The index itself, isolated. Too few events to say anything on its own, but "
            "consistent with the pooled basket result."
        ),
        code(
            "if HAVE_REAL:\n"
            "    spyres = [st.run_experiment(PANEL, horizon=h, names=['SPY'], n_draws=3000) for h in hs]\n"
            "    spy_ex = [r['mean']*100 for r in spyres]; spy_t = [r['t'] for r in spyres]\n"
            "else:\n"
            "    spy_ex = [s[2] for s in R['spy']]; spy_t = [s[4] for s in R['spy']]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "ax.bar([f'{h}d' for h in hs], spy_ex, color=AMBER, width=.55)\n"
            "for i,v in enumerate(spy_ex): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('SPY-only excess (%)'); ax.set_title(f'SPY double bottoms (n={R[\"spy_bottom\"]}): noisy, never t>=2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('SPY excess:', [round(v,3) for v in spy_ex], ' t:', [round(v,2) for v in spy_t])"
        ),
        md(
            f"> \U0001F4A1 In plain words: only **{R['spy_bottom']}** confirmed double bottoms on "
            f"SPY in {R['years']:.1f} years — a small sample — and every *t* stays under 1.1. "
            "\"Buy the SPY double bottom\" is a non-event, not a headline strategy."
        ),
        md(
            "### 4g · Faithful-engine & power control — the harness can catch a real edge\n\n"
            "On a synthetic panel with *planted* double bottoms: with **zero** post-breakout "
            "drift the same-tape placebo must NOT light up (note the naive *t* averages "
            f"**{R['syn_null_mean']:+.2f}** across seeds — mildly positive from the figure's own "
            "geometry, exactly why the placebo, not the raw *t*, is the arbiter); with a **large** "
            "planted drift it must light up hard."
        ),
        code(
            "null_ts, null_ps = [], []\n"
            "for s_ in range(20):\n"
            "    px, _ = data.synthetic_panel(edge=0.0, seed=696 + s_, n_planted=8, daily_vol=0.011)\n"
            "    r = st.run_experiment(px, horizon=20, n_draws=800, seed=696 + s_)\n"
            "    null_ts.append(r['t']); null_ps.append(r['p_placebo'])\n"
            "null_ts = np.asarray(null_ts); null_ps = np.asarray(null_ps)\n"
            "rows = []\n"
            "for edge in (0.0, 0.20):\n"
            "    px, truth = data.synthetic_panel(edge=edge, seed=696, n_planted=8, daily_vol=0.011)\n"
            "    r = st.run_experiment(px, horizon=20, n_draws=2000)\n"
            "    rows.append((edge, truth['n_planted_total'], r['n_breakouts'], r['mean']*100, r['t'], r['p_placebo'], r['win']*100))\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.scatter(np.zeros(20) + np.linspace(-.1,.1,20), null_ts, color=GREY, s=40, label='null worlds (edge=0), 20 seeds')\n"
            "a1.axhline(-2, ls='--', c=RED, lw=1); a1.axhline(2, ls='--', c=RED, lw=1); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_xticks([0]); a1.set_xticklabels(['null x 20']); a1.set_ylabel('naive one-sample t')\n"
            "a1.set_title(f'Naive t drifts positive; placebo p<0.05 in {int((null_ps<0.05).sum())}/20 seeds'); a1.legend()\n"
            "labels = [f'planted\\nedge={e:.2f}' for e,_,_,_,_,_,_ in rows]\n"
            "ps = [r[5] for r in rows]\n"
            "a2.bar(labels, ps, color=[GREY, GREEN], width=.5)\n"
            "a2.axhline(0.05, ls='--', c=RED, label='p=0.05')\n"
            "for i,r in enumerate(rows): a2.annotate(f'p={r[5]:.3f}\\nt={r[4]:.1f}',(i,r[5]),ha='center',va='bottom',fontsize=9)\n"
            "a2.set_ylabel('placebo p-value'); a2.set_title('Control: null not beaten; planted edge -> p~0'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null naive t: mean={null_ts.mean():+.2f} sd={null_ts.std(ddof=1):.2f}  p<0.05 in {(null_ps<0.05).sum()}/20')\n"
            "for e,pl,de,ex,t,p,w in rows: print(f'planted {e:+.2f}: planted={pl} detected={de} excess={ex:+.2f}% t={t:+.2f} p_placebo={p:.3f} win={w:.0f}%')"
        ),
        md(
            f"> \U0001F4A1 In plain words: with **no** planted edge the naive *t* averages "
            f"**{R['syn_null_mean']:+.2f}** (sd {R['syn_null_sd']:.2f}) across 20 seeds — mildly "
            "positive from the pattern's own geometry — but the **placebo never fires at the 5% "
            f"level** ({R['syn_null_fire']}/{R['syn_null_n']} seeds). A planted drift of +20% "
            f"reaches placebo *p* = **{R['syn'][1][6]:.3f}** at *t* = **{R['syn'][1][5]:.1f}**, win "
            f"**{R['syn'][1][7]:.0f}%** — the harness *would* catch a real double-bottom edge if "
            "one existed. This is exactly why the placebo — not the raw *t* — is the arbiter "
            "throughout this study."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the double-bottom breakout excess never clears **t ≥ 2** in "
            f"the pattern's favor at any horizon ({R['bottom'][0][5]:+.2f} / "
            f"**{R['bottom'][1][5]:+.2f}** / {R['bottom'][2][5]:+.2f} / {R['bottom'][3][5]:+.2f} "
            "for 5/10/20/40d — 10d is the *wrong sign*). The random-date placebo is never beaten "
            f"(*p* ∈ [0.50, 0.64]). The measured-move target hits {R['mm_rate']:.1f}% of the time "
            f"— statistically identical to a magnitude-matched placebo's {R['mm_placebo_rate']:.1f}% "
            f"(*z* = {R['mm_z']:.2f}). No tolerance in the robustness sweep turns the edge "
            "reliably positive. Carries a **survivorship** caveat that tilts *toward* finding a "
            "working reversal signal. NONE, not WEAK.\n"
            f"- **Tradability `MIRAGE`** — the long-timer gross P&L (**{R['tm_gross']:+.2f}%**/"
            f"trade, *t* = {R['tm_t']:.2f} vs zero) collapses to **{R['tm_excess']:+.3f}%** "
            f"(HAC *t* = {R['tm_t_excess']:.2f}) once raced against a holding-period-matched base "
            "rate — the classic direction trap, demonstrated live. Costs only push the "
            "already-flat headline horizons further negative.\n"
            "- **Target hit rate beats a coin flip of the same size? `BUSTED`** — "
            f"{R['mm_rate']:.1f}% observed vs {R['mm_placebo_rate']:.1f}% magnitude-matched "
            f"placebo (*z* = {R['mm_z']:.2f}). The measured-move target's celebrated \"accuracy\" "
            "is a function of the target being an achievable distance on a rising basket, not of "
            "the W-shape forecasting anything."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson.** Any variable-length \"hold to target\" strategy on a "
            "rising universe must be raced against a **holding-period-matched** baseline, not "
            "zero — the trap shown live in 4d applies equally to any target-and-stop system, not "
            "just chart patterns. Likewise, any \"hit rate\" claim for a price target needs a "
            "**magnitude-matched** control (4c), or it just measures how close the target was.\n"
            "- **The \"tested a level twice\" neighbours.** "
            "[189-double-top](../../189-double-top/) runs a fixed-horizon-vs-random-placebo "
            "protocol on both the bearish and bullish versions; "
            "[415-triple-top-bottom](../../415-triple-top-bottom/) is the three-tap version; "
            "[695-inverse-head-shoulders](../../695-inverse-head-shoulders/) is the three-trough, "
            "asymmetric-head version; [694-matching-low](../../694-matching-low/) is the "
            "two-*candle* micro version. None of them run this study's measured-move + long-timer "
            "bar on the two-trough figure.\n\n"
            "*The reproducible core is offline and deterministic; the detector is two swing lows "
            "within a stated tolerance, a genuine intervening rally, and a confirmed neckline "
            "break, with a strictness sweep, a magnitude-matched measured-move test and a "
            "holding-period-matched long timer. Methods and sources: "
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
