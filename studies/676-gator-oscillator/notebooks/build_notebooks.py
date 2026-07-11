"""Generate the two narrative notebooks for Study 676 (Gator Oscillator).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached basket
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance 30-name basket,
# SPY + 29 liquid US large-caps, 2000-01-03 -> 2026-06-30; MIN_SLEEP=3 wake definition).
R = dict(
    start="2000-01-03", end="2026-06-30", n_names=30, min_sleep=3,
    event={
        1: dict(n_wake=346, n_dir=192, n_bull=120, n_bear=72, mean=0.013, base=0.030,
                win=52.1, t_welch=-0.17, t_hac=0.13, p=0.4682, net=-0.088,
                mean_abs=1.001, base_abs=1.043, t_abs=-0.83, p_abs=0.7434),
        5: dict(n_wake=346, n_dir=192, n_bull=120, n_bear=72, mean=-0.009, base=0.233,
                win=49.5, t_welch=-1.09, t_hac=-0.04, p=0.5100, net=-0.114,
                mean_abs=2.497, base_abs=2.645, t_abs=-1.20, p_abs=0.8424),
        10: dict(n_wake=346, n_dir=192, n_bull=120, n_bear=72, mean=-0.119, base=0.479,
                 win=46.9, t_welch=-1.88, t_hac=-0.40, p=0.6342, net=-0.229,
                 mean_abs=3.364, base_abs=3.785, t_abs=-2.41, p_abs=0.9772),
        20: dict(n_wake=346, n_dir=192, n_bull=120, n_bear=72, mean=0.604, base=0.976,
                 win=51.0, t_welch=-0.73, t_hac=1.35, p=0.1356, net=0.484,
                 mean_abs=5.169, base_abs=5.431, t_abs=-0.99, p_abs=0.8092),
    },
    hold_days=10,
    bh=dict(cagr=8.28, sharpe=0.412, vol=19.32, dd=-55.19, t=2.42),
    wake=dict(cagr=4.26, sharpe=2.548, vol=1.64, dd=-5.67, t=15.22),
    fan=dict(cagr=2.26, sharpe=0.126, vol=17.69, dd=-37.48, t=0.75),
    t_wake_vs_bh=-1.16, t_wake_vs_fan=0.65,
    placebo_p=0.512, placebo_obs=2.136,
    n_wakes_spy=13, in_pos_frac=0.90,
    cost10_cagr=4.23, cost10_sharpe=2.531,
    syn_null_mean=0.41, syn_null_sd=0.78, syn_null_fire=0, syn_null_seeds=10,
    syn_planted_t=3.20, syn_planted_welch=2.54, syn_planted_mean=7.02,
    syn_planted_base=0.67, syn_planted_win=78.6, syn_planted_n=14,
    fp_basket="7b64d4cd54dd", fp_spy="eed0b5d1f41e",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Catches_trend_starts%3F: Busted](https://img.shields.io/badge/Catches_trend_starts%3F-Busted-8b949e?style=flat-square)\n\n"
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

from gator_oscillator import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PANEL = data.load_panel()
    SPY = PANEL["SPY"]
else:
    PANEL = SPY = None
print("real cache present:", HAVE_REAL, "| basket names:",
      (0 if PANEL is None else len(PANEL)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the sleeping alligator really wake up right before it eats? 🐊💤\n"
            "### The Gator Oscillator — a color-coded histogram built from a folk indicator, "
            "tested on its own terms\n\n"
            + BADGES +
            "Bill Williams' **Alligator** is three wiggly, overlapping moving-average lines that "
            "are supposed to represent a sleepy alligator's jaw, teeth and lips. The **Gator "
            "Oscillator** turns that mess of lines into something simpler to eyeball: two little "
            "bar charts that go **green when the lines are spreading apart** (the mouth is "
            "opening — hungry, ready to eat a trend) and **red when they're converging** (the "
            "mouth is shut — asleep, nothing happening).\n\n"
            "The trading idea is irresistibly simple: *wait for the gator to sleep for a while, "
            "then the moment both bars turn green, jump in — you've caught the trend at the "
            "start.* We test exactly that, on real market data, honestly.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** The Gator is literally built from the same three lines as "
            "[study 421 (the Alligator)](../../421-williams-alligator/) — same fan, same "
            "shifts. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the gator waking up predict which way price goes next? | **No.** Across "
            f"**{R['event'][10]['n_wake']}** genuine wake events (a real sleep, not a one-day "
            "flicker) on 30 stocks over 26 years, the move *after* a wake looks just like an "
            "ordinary day — no horizon (1, 5, 10 or 20 days later) clears the statistical bar. |\n"
            "| Does it at least predict a *bigger* move is coming, even if not which way? | "
            "**No — if anything the opposite.** At the 10-day mark, moves after a wake are "
            "*smaller* than average, not bigger. |\n"
            f"| Can you actually trade it? | **It looks amazing, then it doesn't.** A real timer "
            f"on SPY posts a jaw-dropping Sharpe of **{R['wake']['sharpe']:.2f}** vs "
            f"buy-and-hold's **{R['bh']['sharpe']:.2f}** — but that's a trick of the numbers "
            f"(it's sitting in cash **{R['in_pos_frac']*100:.0f}%** of the time). Test it "
            "properly and the advantage disappears. |\n"
            "| So is the Alligator/Gator family completely useless? | The Alligator itself "
            "(study 421) at least shows *some* trend-following behavior, even if it doesn't "
            "beat a plain moving average. The Gator adds nothing on top of that. |\n\n"
            "> A green bar means the alligator's mouth is opening. It doesn't mean there's "
            "anything worth eating."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When the Gator's bars are red, the market is consolidating — stay out. The "
            "instant both bars turn green together, a new trend is beginning. Get in, ride it "
            "in the direction the Alligator's lines are already fanned, and get out when the "
            "gator goes back to sleep.\"*\n\n"
            "It's one of the most-taught indicators on retail trading platforms (it ships as a "
            "default on MetaTrader). It has an intuitive mechanical story: **the Gator bars are "
            "literally just the rate-of-change of the Alligator's own spread** — so a green bar "
            "genuinely does mean \"the lines are pulling apart faster.\" The question is whether "
            "that geometric fact translates into anything you could trade on."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the Gator really flags the *start* of a trend, it would be a genuinely useful "
            "timing tool — better than just being in the Alligator's fan all the time (study "
            "421's approach), because you'd only take a position right as the move begins "
            "instead of riding the whole choppy fan-in/fan-out cycle. That's worth testing "
            "properly rather than eyeballing a chart: does watching the color change actually "
            "buy you anything over the plain Alligator?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **A real definition of \"waking.\"** A single-day red-to-green flip on both bars "
            "happens on roughly **1 day in 4** — that's not \"the market slept, then woke up,\" "
            "that's just noise. We require the mouth to have been genuinely shut (both bars red) "
            f"for **{R['min_sleep']}+ days in a row** before counting a wake.\n"
            f"- **The comparison.** Pool every qualifying wake across **{R['n_names']}** liquid "
            "stocks + SPY (2000→2026), sign the next 1/5/10/20-day return by whichever way the "
            "Alligator's lines are fanned, and compare it to an *ordinary* day on the same "
            "stocks.\n"
            "- **The trade check.** Build an actual SPY timer: buy (or short) the day after a "
            "wake, hold two weeks, pay real trading costs — then race it against simply buying "
            "and holding, and against just staying in the Alligator's fan the whole time."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** The signed return after a wake, at four different "
            "horizons, next to what an ordinary day on the same stocks returns."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(PANEL, horizons=st.HORIZONS, n_draws=1000)\n"
            "    hs = list(res['horizon']); means = list(res['mean_signed']*100)\n"
            "    bases = list(res['base_mean']*100)\n"
            "else:\n"
            "    hs = sorted(R['event']); means = [R['event'][h]['mean'] for h in hs]\n"
            "    bases = [R['event'][h]['base'] for h in hs]\n"
            "x = np.arange(len(hs)); w = 0.35\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "ax.bar(x - w/2, means, width=w, color=RED, label='after a wake')\n"
            "ax.bar(x + w/2, bases, width=w, color=GREY, label='ordinary day (same stocks)')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean forward return (%)')\n"
            "ax.set_title('No horizon shows the gator waking up predicts direction')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print({h: (round(m,3), round(b,3)) for h,m,b in zip(hs, means, bases)})"
        ),
        md(
            f"The 10-day bar (the same window the timer trades) shows the biggest gap — wake "
            f"events return **{R['event'][10]['mean']:+.3f}%** vs **{R['event'][10]['base']:+.3f}%** "
            "for an ordinary day — but that's a smaller number *behind* the ordinary day, and "
            f"it's still not statistically distinguishable from luck (the quants notebook shows "
            f"why: Welch *t* = {R['event'][10]['t_welch']:.2f}, short of the ±2 bar).\n\n"
            "**What about the \"a bigger move is coming\" fallback** — maybe the Gator can't call "
            "direction, but does it at least flag more action ahead?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    a, b = res['mean_abs']*100, res['base_abs']*100\n"
            "else:\n"
            "    a = [R['event'][h]['mean_abs'] for h in hs]\n"
            "    b = [R['event'][h]['base_abs'] for h in hs]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "ax.bar(x - w/2, a, width=w, color=AMBER, label='|move| after a wake')\n"
            "ax.bar(x + w/2, b, width=w, color=GREY, label='|move| on an ordinary day')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('mean absolute forward return (%)')\n"
            "ax.set_title('The gator waking does not mean a bigger move is coming')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('10-day: after wake', round(list(a)[2] if HAVE_REAL else a[2],3),\n"
            "      '  ordinary', round(list(b)[2] if HAVE_REAL else b[2],3))"
        ),
        md(
            "If anything, the 10-day move after a wake is *smaller* than an ordinary day's — "
            "the opposite of \"trend capture.\" Not a robust pattern across every horizon, but "
            "there is certainly no sign of the promised bigger move.\n\n"
            "**Finally, the trade.** This is where the story gets genuinely tricky — because on "
            "the surface, it looks like the Gator *works beautifully*:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    out = st.run_timer_experiment(SPY, placebo=False)\n"
            "    bh_s, wk_s = out['bh']['sharpe'], out['wake']['sharpe']\n"
            "    bh_c, wk_c = out['bh']['cagr']*100, out['wake']['cagr']*100\n"
            "else:\n"
            "    bh_s, wk_s = R['bh']['sharpe'], R['wake']['sharpe']\n"
            "    bh_c, wk_c = R['bh']['cagr'], R['wake']['cagr']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar(['buy & hold','gator timer'], [bh_s, wk_s], color=[GREY, GREEN], width=.55)\n"
            "for i,v in enumerate([bh_s, wk_s]): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "a1.set_title('Sharpe: the gator timer looks amazing...'); a1.set_ylabel('Sharpe ratio')\n"
            "a2.bar(['buy & hold','gator timer'], [bh_c, wk_c], color=[GREY, RED], width=.55)\n"
            "for i,v in enumerate([bh_c, wk_c]): a2.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_title('...but it actually makes HALF the money'); a2.set_ylabel('CAGR (%)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Sharpe: BH {bh_s:.2f}  Gator timer {wk_s:.2f}  |  CAGR: BH {bh_c:+.1f}%  Gator timer {wk_c:+.1f}%')"
        ),
        md(
            f"A Sharpe of **{R['wake']['sharpe']:.2f}** against buy-and-hold's **{R['bh']['sharpe']:.2f}** "
            "looks like a knockout. But look at the CAGR chart right next to it: the gator timer "
            f"makes **{R['wake']['cagr']:+.1f}%/yr**, buy-and-hold makes **{R['bh']['cagr']:+.1f}%/yr** "
            f"— roughly *double*. How can the \"worse\" strategy have the better Sharpe? Because "
            f"it's sitting in cash **{R['in_pos_frac']*100:.0f}%** of the time (only "
            f"**{R['n_wakes_spy']}** genuine wakes fired on SPY in 26.5 years), earning a smooth, "
            "riskless 4%/yr on almost every day. A portfolio that's mostly cash has tiny "
            "volatility almost by definition — and Sharpe rewards low volatility, no matter "
            "where it comes from. Test it the honest way (does it actually beat buy-and-hold, "
            "statistically?) and the advantage vanishes — the quants notebook shows exactly how."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** No horizon (1/5/10/20 days) after a genuine wake shows a "
            "statistically real directional edge, and the \"bigger move coming\" fallback "
            "doesn't hold up either.\n"
            "- **Tradability — Mirage.** The eye-catching Sharpe on a real SPY timer is a "
            "side-effect of sitting mostly in cash, not a genuine trading edge — it doesn't "
            "beat buy-and-hold once tested properly, and doesn't beat simply staying in the "
            "Alligator's fan the whole time either.\n"
            "- **\"Does the gator's awakening catch the start of a trend?\" — Busted.** We proved "
            "the detector *can* catch a real planted trend in a controlled experiment — so it's "
            "not that the method is broken. It's that on real markets, there's nothing there to "
            "catch."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The Gator adds a layer of decoration, not information.** It's built entirely "
            "from the same three lines as [study 421](../../421-williams-alligator/) — any "
            "signal it could carry is already inside those lines. If a trend-following signal "
            "exists in the Alligator fan, watching for the specific moment the histogram changes "
            "color doesn't seem to time it any better.\n"
            "- **Why the Sharpe trap matters beyond this study.** Any strategy that's mostly in "
            "cash will tend to show an inflated Sharpe against a fully-invested benchmark — "
            "always check the *difference* is statistically real, and always look at the actual "
            "return, not just the ratio.\n"
            "- **Sibling studies:** [421-williams-alligator](../../421-williams-alligator/) (the "
            "fan itself), [184-williams-fractals](../../184-williams-fractals/) (a different "
            "Williams pivot marker), [420-awesome-oscillator](../../420-awesome-oscillator/) and "
            "[474-accelerator-oscillator](../../474-accelerator-oscillator/) (Williams' momentum "
            "cousins) — none of them is this study.\n\n"
            "*Think the Gator works on a different holding period or a different market? Show a "
            "signed forward-return edge that clears |t| ≥ 2 net of costs — then we'll talk.*"
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
            "# The Gator Oscillator — a quantitative teardown 🔬\n"
            "### A pooled wake-event study across 1/5/10/20-day horizons · a magnitude "
            "(trend-capture) test · a real SPY timer vs buy-and-hold and vs the plain Alligator "
            "· a block-permutation placebo that unmasks a Sharpe artifact · a synthetic "
            "positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The Gator Oscillator is not an independent signal — it is the rate-of-change of "
            "sibling [421-williams-alligator](../421-williams-alligator/)'s own three SMMAs, "
            "plotted as a green/red histogram. The job here is a narrow, honest question: does "
            "watching that histogram's color change — specifically the \"wake\" transition — add "
            "a real, tradable timing edge over the fan it's built from?\n\n"
            "> ⚠️ **Data note.** 30-name basket (SPY + 29 liquid US large-caps), yfinance daily "
            "OHLCV, `auto_adjust=True`, 2000-01-03 → 2026-06-30. **Survivors basket** — named on "
            "the Signal axis. Methods in [`docs/references.md`](../docs/references.md), numbers "
            "in [`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_basket"] +
            "` basket / `" + R["fp_spy"] + "` SPY).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | signed 10-day wake return Welch **t = {R['event'][10]['t_welch']:.2f}**, "
            f"HAC **t = {R['event'][10]['t_hac']:.2f}**; no horizon clears \\|t\\| ≥ 2; magnitude "
            f"test at 10d: Welch **t = {R['event'][10]['t_abs']:.2f}** (wrong sign) |\n"
            f"| **Tradability** | `MIRAGE` | Sharpe-diff *t* (wake vs BH) = **{R['t_wake_vs_bh']:.2f}**; "
            f"vs the plain Alligator fan *t* = **{R['t_wake_vs_fan']:.2f}**; block-permutation "
            f"placebo **p = {R['placebo_p']:.3f}** |\n"
            f"| **Catches trend starts?** | `BUSTED` | machinery detects a planted trend at "
            f"*t* = **{R['syn_planted_t']:.2f}**; real tape shows nothing directional |\n\n"
            "> 💡 In plain words: the Gator's histogram is a decoration on top of the Alligator's "
            "own lines — geometrically real, statistically empty."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $J_t, Te_t, L_t$ be the Alligator's Jaw/Teeth/Lips (SMMA(13/8/5) of median "
            "price, forward-shifted 8/5/3 bars — identical to sibling 421). The Gator "
            "histograms are $U_t = |J_t - Te_t|$ (upper) and $Lo_t = |Te_t - L_t|$ (lower), each "
            "colored GREEN if taller than $t-1$, RED otherwise. A **wake** at $t$ requires both "
            "$U$ and $Lo$ green at $t$ and both red for $\\geq 3$ consecutive prior bars ($t-1$ "
            "back through $t-3$) — a genuine sleep, not a single-day flicker (which fires on "
            "~24% of days and is not the claim). The claims:\n\n"
            "- **H₁ (direction).** $E[\\text{signed fwd ret} \\mid \\text{wake}] \\gg 0$ at some "
            "horizon, where the sign comes from the concurrent Alligator fan.\n"
            "- **H₂ (magnitude / trend-capture).** $E[|\\text{fwd ret}| \\mid \\text{wake}] > "
            "E[|\\text{fwd ret}|]$ unconditionally — a wake flags *action ahead*, regardless of "
            "direction.\n"
            "- **H₃ (capture).** A real timer built on the wake signal beats buy-and-hold **and** "
            "beats simply being in the Alligator fan continuously (421's rule) net of costs.\n\n"
            "We find **H₁ not supported** at any of the four horizons tested, **H₂ not "
            "supported** (and reversed in sign at 10 days), **H₃ not supported** once the "
            "Sharpe-difference is tested properly rather than eyeballed."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Wake events are pooled across a **30-name basket** so the event study has real "
            "power (346 events, 192 with a directional fan). Because events on the same name "
            "can cluster in time, the primary statistic is a **HAC (Newey-West) one-sample *t*** "
            "on the signed forward return; a **Welch *t*** against the unconditional base rate "
            "(every bar of every name, same window) checks the split isn't just \"stocks tend to "
            "rise\"; a **5,000-draw label-shuffle placebo** (random bars, fair-coin sign) checks "
            "the observed mean isn't a lucky draw of the same size. The magnitude test uses the "
            "same machinery on $|\\text{fwd ret}|$. The timer's Sharpe race uses a **Sharpe-"
            "difference HAC *t*** (Jobson-Korkie style) and a **2,000-draw circular-block-"
            "permutation placebo** that reshuffles the *timing* of the same sparse exposure "
            "pattern — the only honest way to ask \"could this Sharpe gap be luck.\""
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Basket.** {R['n_names']} names {R['start']} → {R['end']}, survivors panel "
            "(named on Signal).\n"
            f"- **Wake definition.** Both Gator histograms green after ≥ {R['min_sleep']} "
            "consecutive both-red bars — fixed once, reused in the event study, the timer, and "
            "the synthetic control.\n"
            "- **Headline.** HAC *t* + Welch *t* + 5,000-draw label-shuffle placebo, at "
            "1/5/10/20-day horizons, signed by the concurrent Alligator fan.\n"
            "- **Magnitude.** Same machinery on |forward return|, unsigned.\n"
            f"- **Execution.** One documented lag: wake/fan known at close $t$; event study "
            f"enters open $t+1$; timer signal shifted once, held {R['hold_days']} sessions.\n"
            "- **Timer.** SPY only, 5/10 bps one-way costs, 50 bps/yr borrow on shorts, flat "
            "4%/yr cash-leg proxy while flat (matches sibling 421's convention) — raced against "
            "buy-and-hold **and** the plain \"always in the fan\" rule.\n"
            "- **Control.** Synthetic trend-persistence panels (20 series each, the synthetic "
            "analogue of the real basket); the null must not fire across 10 independent seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline event study — signed forward return by horizon\n\n"
            "HAC *t*, Welch *t* vs the unconditional base rate, and the label-shuffle placebo, "
            "at each horizon."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(PANEL, horizons=st.HORIZONS, n_draws=1000)\n"
            "    print(res[['horizon','n_wake','n_dir','mean_signed','base_mean','t_hac','t_welch','p_placebo']]\n"
            "          .to_string(index=False))\n"
            "    hs = list(res['horizon']); t_hacs = list(res['t_hac']); t_welchs = list(res['t_welch'])\n"
            "else:\n"
            "    hs = sorted(R['event'])\n"
            "    t_hacs = [R['event'][h]['t_hac'] for h in hs]\n"
            "    t_welchs = [R['event'][h]['t_welch'] for h in hs]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "x = np.arange(len(hs)); w = 0.35\n"
            "ax.bar(x - w/2, t_hacs, width=w, color=RED, label='HAC t (one-sample vs 0)')\n"
            "ax.bar(x + w/2, t_welchs, width=w, color=AMBER, label='Welch t (vs unconditional)')\n"
            "ax.axhline(2, ls='--', c='k', lw=1); ax.axhline(-2, ls='--', c='k', lw=1)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('t-statistic'); ax.set_title('No horizon crosses the |t| >= 2 bar (dashed)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: 346 wake events, 192 with a directional fan (120 bull / 72 "
            f"bear). Every *t* sits inside [−2, +2]; the 10-day horizon comes closest (Welch "
            f"*t* = {R['event'][10]['t_welch']:.2f}) and is still short. H₁ fails at every "
            "horizon tested."
        ),
        md(
            "### 4b · The magnitude (trend-capture) test\n\n"
            "Unsigned: does a wake predict a bigger move, regardless of direction?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    a = list(res['mean_abs']*100); b = list(res['base_abs']*100)\n"
            "    ta = list(res['t_welch_abs'])\n"
            "else:\n"
            "    a = [R['event'][h]['mean_abs'] for h in hs]\n"
            "    b = [R['event'][h]['base_abs'] for h in hs]\n"
            "    ta = [R['event'][h]['t_abs'] for h in hs]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(x - w/2, a, width=w, color=AMBER, label='|fwd ret| after wake')\n"
            "a1.bar(x + w/2, b, width=w, color=GREY, label='|fwd ret| unconditional')\n"
            "a1.set_xticks(x); a1.set_xticklabels([f'{h}d' for h in hs])\n"
            "a1.set_ylabel('mean |forward return| (%)'); a1.legend()\n"
            "a1.set_title('Magnitude: no \"bigger move ahead\" signal')\n"
            "a2.bar(x, ta, color=[RED if abs(t)>=2 else GREY for t in ta], width=.6)\n"
            "a2.axhline(2, ls='--', c='k', lw=1); a2.axhline(-2, ls='--', c='k', lw=1)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_xticks(x); a2.set_xticklabels([f'{h}d' for h in hs])\n"
            "a2.set_ylabel('Welch t (magnitude)'); a2.set_title('10-day: significant, WRONG sign')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('t_welch_abs by horizon:', {h: round(t,2) for h,t in zip(hs, ta)})"
        ),
        md(
            f"> 💡 In plain words: at 10 days, moves after a wake are *smaller* than an ordinary "
            f"day's (Welch *t* = {R['event'][10]['t_abs']:.2f}) — the opposite of what \"trend "
            "capture\" predicts. It's one horizon out of four (not robust across the window, and "
            "the placebo assigns it no special standing — p = 0.977), but there is certainly no "
            "supporting evidence for H₂ anywhere."
        ),
        md(
            "### 4c · The third axis — a real timer, and the Sharpe trap it walks into\n\n"
            "Enter the fan direction the bar after a wake, hold 10 sessions, one execution lag, "
            "5 bps one-way costs, 50 bps/yr borrow on shorts, flat 4%/yr cash-leg proxy while "
            "flat (SPY, 2000→2026)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    out = st.run_timer_experiment(SPY, placebo=True, n_draws=800)\n"
            "    bh_s, wk_s, fan_s = out['bh']['sharpe'], out['wake']['sharpe'], out['fan']['sharpe']\n"
            "    bh_c, wk_c, fan_c = out['bh']['cagr']*100, out['wake']['cagr']*100, out['fan']['cagr']*100\n"
            "    twb, twf = out['t_wake_vs_bh'], out['t_wake_vs_fan']\n"
            "    pval, pobs = out['placebo_p'], out['placebo_obs']\n"
            "    n_w, ipf = out['n_wakes'], out['in_pos_frac']*100\n"
            "else:\n"
            "    bh_s, wk_s, fan_s = R['bh']['sharpe'], R['wake']['sharpe'], R['fan']['sharpe']\n"
            "    bh_c, wk_c, fan_c = R['bh']['cagr'], R['wake']['cagr'], R['fan']['cagr']\n"
            "    twb, twf = R['t_wake_vs_bh'], R['t_wake_vs_fan']\n"
            "    pval, pobs = R['placebo_p'], R['placebo_obs']\n"
            "    n_w, ipf = R['n_wakes_spy'], R['in_pos_frac']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.3))\n"
            "a1.bar(['buy&hold','gator timer','always in fan'], [bh_s, wk_s, fan_s],\n"
            "       color=[GREY, GREEN, AMBER], width=.6)\n"
            "for i,v in enumerate([bh_s, wk_s, fan_s]): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('Sharpe'); a1.set_title('Raw Sharpe: gator timer looks best...')\n"
            "a2.bar(['buy&hold','gator timer','always in fan'], [bh_c, wk_c, fan_c],\n"
            "       color=[GREY, GREEN, AMBER], width=.6)\n"
            "for i,v in enumerate([bh_c, wk_c, fan_c]): a2.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('CAGR (%)'); a2.set_title('...but earns the LEAST money')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Sharpe-diff t (wake vs BH) = {twb:+.2f}   (wake vs fan) = {twf:+.2f}')\n"
            "print(f'block-permutation placebo p = {pval:.3f}  (observed diff {pobs:+.3f})')\n"
            "print(f'n wakes on SPY = {n_w}   in-position fraction = {ipf:.2f}%')"
        ),
        md(
            f"> 💡 In plain words: the wake timer's Sharpe ({R['wake']['sharpe']:.2f}) beats "
            f"buy-and-hold's ({R['bh']['sharpe']:.2f}) only because it's a **{R['in_pos_frac']*100:.0f}%**-"
            f"cash strategy — {R['n_wakes_spy']} events in 26.5 years — with near-zero realized "
            "volatility from the flat cash leg. Its CAGR is *half* buy-and-hold's "
            f"({R['wake']['cagr']:+.1f}% vs {R['bh']['cagr']:+.1f}%). The honest tests both say "
            f"\"no edge\": Sharpe-difference HAC *t* vs buy-and-hold = **{R['t_wake_vs_bh']:+.2f}** "
            f"(not significant), and a block-permutation placebo shows a random reshuffle of the "
            f"same sparse in/out pattern matches or beats the observed Sharpe gap **"
            f"{R['placebo_p']*100:.0f}%** of the time. Against the plain \"always in the fan\" "
            f"rule (421's approach), the difference is **{R['t_wake_vs_fan']:+.2f}** — also not "
            "significant. H₃ fails on every honest test."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "A trend-persistence knob (`edge`), same family as sibling 421; each panel pools 20 "
            "independent synthetic series (the synthetic analogue of the real 30-name basket) so "
            "the wake-event detector has enough events to compute a meaningful *t*. The null "
            "(`edge=0`) is checked over **10 independent panel seeds** — never a single stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(10):\n"
            "    pan = data.synthetic_multi_panel(edge=0.0, seed=676 + s_)\n"
            "    null_ts.append(st.summarize(pan, horizon=st.HOLD_DAYS, placebo=False)['t_hac'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "pan = data.synthetic_multi_panel(edge=8.0, seed=676)\n"
            "sy = st.summarize(pan, horizon=st.HOLD_DAYS, placebo=False)\n"
            "planted_t = sy['t_hac']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(10) + np.linspace(-.12,.12,10), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (edge=0), 10 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5,\n"
            "           label='planted trend (edge=8.0)')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 10', 'planted'])\n"
            "ax.set_ylabel('HAC t (signed wake return)')\n"
            "ax.set_title('Control: no null fires; a planted trend lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts)>=2).sum()}/10 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 10 null panels the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and **never** crosses the "
            f"bar; a planted multi-week trend (`edge=8.0`) reads t = {R['syn_planted_t']:.2f} "
            f"(win rate {R['syn_planted_win']:.0f}%, n = {R['syn_planted_n']}). The machinery is "
            "unbiased and *can* catch a real trend — the flat real-tape result is a genuine "
            "\"nothing there,\" not a broken detector. *(A faithful-engine / power check only — "
            "never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — no horizon (1/5/10/20 days) of the signed post-wake return "
            f"clears \\|*t*\\| ≥ 2 (HAC or Welch); the magnitude/trend-capture fallback shows the "
            f"same at best, and the wrong sign at 10 days (Welch *t* = {R['event'][10]['t_abs']:.2f}).\n"
            f"- **Tradability `MIRAGE`** — a real SPY timer's headline Sharpe "
            f"({R['wake']['sharpe']:.2f}) is a cash-dominance artifact ({R['in_pos_frac']*100:.0f}% "
            f"flat, {R['n_wakes_spy']} events in 26.5 years); the Sharpe-difference *t* vs "
            f"buy-and-hold is {R['t_wake_vs_bh']:+.2f} and vs the plain Alligator fan "
            f"{R['t_wake_vs_fan']:+.2f}, neither significant, and a block-permutation placebo "
            f"(p = {R['placebo_p']:.3f}) confirms it.\n"
            "- **\"Catches the start of a trend?\" `BUSTED`** — the synthetic control proves the "
            f"machinery *can* catch a planted trend (t = {R['syn_planted_t']:.2f}); on the real "
            "tape, watching the Gator's color change adds nothing over the Alligator fan it's "
            "built from."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson: a derived transform carries no new information by "
            "construction.** The Gator is the rate-of-change of the Alligator's own spread — any "
            "edge it could have was already fully contained in the three lines it's built from. "
            "The honest prior, going in, should have been \"at best equal to 421, likely worse "
            "after the extra filtering throws away most days as `not a wake'.\" The data agrees.\n"
            "- **The Sharpe trap generalises.** Any strategy that's mostly in cash inflates its "
            "Sharpe ratio against a fully-invested benchmark purely through variance shrinkage — "
            "always test the *difference* with HAC inference and a permutation placebo, never "
            "read a raw Sharpe gap at face value.\n"
            "- **Dedup map:** [421-williams-alligator](../../421-williams-alligator/) (the fan "
            "itself, run continuously — Weak/Mirage), "
            "[184-williams-fractals](../../184-williams-fractals/) (a different Williams pivot "
            "marker — None/Mirage), [420-awesome-oscillator](../../420-awesome-oscillator/) and "
            "[474-accelerator-oscillator](../../474-accelerator-oscillator/) (Williams' momentum "
            "cousins, unrelated construction — both None/Mirage). The whole Williams *Trading "
            "Chaos* toolkit reads the same way on this desk: intuitive stories, empty statistics.\n\n"
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
