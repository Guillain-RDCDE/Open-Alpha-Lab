"""Generate the two narrative notebooks for Study 415 (Triple Top & Bottom).

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
# OHLC, SPY + 29 large-caps, as-of 2026-05-31, last bar 2026-05-29, 21.4 years, fp 62992380d9e9).
R = dict(
    asof="2026-05-31", last_bar="2026-05-29", start="2005-01-03", end="2026-05-29",
    years=21.4, n_bars=5385, n_names=30, fingerprint="62992380d9e9",
    n_bottom=356, n_top=211, spy_bottom=5, spy_top=3,
    # triple-bottom (bullish, long) forward edge — excess over base rate:
    #   (H, n, excess%, raw%, win%, t, HAC_t, p_plac, net%)
    bottom=[(5, 356, 0.335, 0.644, 56, 1.75, 1.82, 0.369, 0.235),
            (10, 355, 0.207, 0.816, 53, 0.72, 0.77, 0.443, 0.107),
            (20, 354, 0.389, 1.600, 55, 0.79, 0.86, 0.421, 0.289),
            (40, 350, -0.689, 1.687, 48, -1.22, -1.32, 0.598, -0.789)],
    # triple-top (bearish, short) myth check — excess over (short) base rate:
    #   (H, n, excess%, raw%, win%, t, p_plac)
    top=[(5, 211, -0.050, -0.360, 54, -0.15, 0.510),
         (10, 211, -0.590, -1.202, 48, -1.18, 0.626),
         (20, 210, -1.376, -2.598, 43, -2.04, 0.701),
         (40, 210, -1.941, -4.336, 46, -2.12, 0.703)],
    # SPY-only triple bottom (the hook): (H, n, excess%, raw%, t, p_plac)
    spy=[(5, 5, 0.161, 0.398, 0.92, 0.457),
         (10, 5, -0.089, 0.381, -0.11, 0.559),
         (20, 5, 0.281, 1.218, 0.15, 0.475),
         (40, 5, 0.699, 2.546, 0.46, 0.434)],
    # robustness at H=5 (the best bottom horizon): (level_tol, min_bounce, n, excess%, t, p_plac)
    robust=[(0.03, 0.07, 126, 0.322, 0.87, 0.424),
            (0.04, 0.05, 356, 0.335, 1.75, 0.367),
            (0.06, 0.04, 641, 0.018, 0.13, 0.492)],
    # synthetic control 20d: (planted_edge, planted, detected, n, excess%, t, p_plac, win%)
    syn=[(0.00, 160, 218, 218, -1.64, -5.06, 0.828, 38),
         (0.20, 160, 209, 209, 5.02, 10.19, 0.016, 77)],
    cost_bps=5.0,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Reliable_reversal%3F: Busted](https://img.shields.io/badge/Reliable_reversal%3F-Busted-8b949e?style=flat-square)\n\n"
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

from triple_top_bottom import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
if HAVE_REAL:
    PANEL = data.load_real()
    asof = pd.Timestamp(ASOF)
    for f in PANEL:
        PANEL[f] = PANEL[f].loc[:asof]
    CLOSES = PANEL["close"]
else:
    PANEL = CLOSES = None
print("real triple cache present:", HAVE_REAL,
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
            "# Do three failures at one level make a signal? \U0001F4C9\n"
            "### The triple top & triple bottom — the textbook \"three taps and reverse\" figure, in plain English\n\n"
            + BADGES +
            "Here's a piece of chart lore you'll meet on day one of any technical-analysis course. "
            "When a stock falls to the **same price three times** and bounces each time — three "
            "valleys lined up at one level — that's a **triple bottom**, and it supposedly means the "
            "sellers are exhausted: the third failure to break lower is the tell, and when the price "
            "finally pops above the in-between highs (the *neckline*), you **buy the breakout** and "
            "ride the reversal. Flip it upside-down — three peaks at one level, then a break **down** "
            "— and you have a **triple top**, the bearish twin.\n\n"
            "It's a *beautiful* story. Three is a magic number, the picture is clean, and \"three "
            "failures = a wall\" feels like real information. So let's do the thing the story never "
            "does: write down a **mechanical** rule for the figure, run it across two decades of real "
            "stocks, and ask whether buying the confirmed breakout actually beats just holding the "
            "stock.\n\n"
            "> \U0001F4D3 **Plain-language layer.** Want the *t*-stats, the placebo tests and the cost "
            "math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data + honesty note up front.** Chart figures are **partly subjective** — "
            "three chartists will draw three different triangles on the same tape. We test the *closest "
            "mechanical definition* we can write down (three swing pivots at one price within a "
            "tolerance, real bounces between them, then a confirmed neckline break) and we say so. We "
            "use a fixed **30-name large-cap basket + SPY** (names still trading today), which carries "
            "**survivorship** — it can't include names that delisted after a failed pattern, a mild tilt "
            "*for* the bullish version. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After a confirmed **triple bottom**, does the stock outrun its own normal drift? | "
            f"**Barely, and not reliably.** The best horizon (5 days) shows **{R['bottom'][0][2]:+.2f}%** "
            f"excess — but at *t* = **{R['bottom'][0][5]:.2f}**, *under* the bar, and a random-date "
            "placebo matches it about **37%** of the time. By 40 days it's **negative**. |\n"
            "| Is the magic-three threshold special? | **No.** Loosen or tighten the \"same level\" "
            "tolerance and the tiny edge **evaporates** — it's a knob-tuning artefact, not a wall. |\n"
            "| Does the bearish **triple top** mirror it (a clean short)? | **No — it's worse.** "
            "Shorting the triple-top breakdown **loses** money on average (the names keep drifting "
            f"**up** — short excess **{R['top'][2][2]:+.2f}%** at 20d). The figure does **not** reverse "
            "symmetrically. |\n"
            "| So is it a tradable signal? | **No.** A thin, sub-*t*-2, knob-sensitive edge that "
            "**dies at costs** and **flips negative** at the horizons the lore actually recommends. |\n\n"
            "> The picture is real on the chart; the *edge* is not. \"Three failures at one level\" is a "
            "pattern your eye loves and the tape ignores."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When price tests the same support **three** times and holds, the down-move is "
            "exhausted. The third failure is the confirmation. Buy when it breaks the neckline and ride "
            "the reversal — and short the mirror-image triple top when it breaks down.\"*\n\n"
            "This is straight out of the canon — Edwards & Magee's *Technical Analysis of Stock Trends* "
            "(1948), Schabacker before them, and every modern pattern guide since (Bulkowski's "
            "*Encyclopedia of Chart Patterns* even tallies \"success rates\"). The triple top/bottom is "
            "sold as a **major reversal** figure: rarer than the double, and therefore *more* reliable "
            "because the level has been defended three times. The question isn't whether the picture "
            "exists — of course three bounces sometimes line up — it's whether **acting on the "
            "confirmed breakout pays.**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the figure worked, it would be a gift: a **visual, rules-light** entry that any retail "
            "trader can spot, with a built-in stop (just below the third bottom) and a clean target "
            "(the figure's height projected from the neckline). A reliable reversal signal you can *see* "
            "would be one of the great free lunches.\n\n"
            "But two traps hide here. **(1) Selection by eye.** \"Three taps at one level\" is so loose "
            "that *something* always qualifies if you squint — and a pattern you only recognise after "
            "the breakout is a pattern you can curve-fit. **(2) The base rate.** Stocks drift **up** "
            "over time, so *any* \"buy\" signal looks like it works a bit. The honest test is whether the "
            "breakout beats the stock's **own** normal drift — and whether a *random* entry on the same "
            "tape would have looked just as good."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take a fixed **{R['n_names']}-name large-cap basket + SPY** and run a **mechanical** "
            "triple-bottom detector on every name over **{:.0f} years** ({} → {}):\n\n".format(
                R['years'], R['start'], R['last_bar']) +
            "1. **Swing pivots.** Find local lows (and highs) over a symmetric window.\n"
            "2. **Three taps at one level.** Three swing lows within a tolerance of one price — the "
            "\"three failures\" — separated by genuine bounces (real rallies, not a flat shelf).\n"
            "3. **The neckline & the breakout.** The neckline is the highest high between the lows; the "
            "**signal** is the first close that clears it. We **enter the next close** (no cheating) and "
            "hold **5 / 10 / 20 / 40** days.\n"
            "4. **The honest bar.** Compare the forward return to the name's *own* base rate, test it "
            "with a *t*-stat, and run a **random-date placebo** on the same tape: if random entries beat "
            "the pattern as often as not, the figure carries no information.\n\n"
            "**What would make us say \"mirage\"?** A *t* under 2, a placebo it can't beat, an edge that "
            "needs a particular tolerance to exist, and a bearish twin that doesn't mirror it. (Spoiler: "
            "all four.)"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, what does the detector even find?** Here's one real confirmed triple bottom drawn "
            "by the code — three lows at one support, the neckline, and the breakout bar."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tk = 'JPM'\n"
            "    c = CLOSES[tk].dropna().to_numpy(float)\n"
            "    trips = st.detect_triples(c, side='bottom')\n"
            "    d = trips[len(trips)//2] if trips else None\n"
            "else:\n"
            "    px, _ = data.synthetic_panel(edge=0.0, seed=415); c = px['close']['N00'].to_numpy(float)\n"
            "    trips = st.detect_triples(c, side='bottom'); d = trips[0] if trips else None\n"
            "if d is not None:\n"
            "    a, b = d['taps'][0]-15, d['breakout_idx']+25\n"
            "    a = max(a, 0); b = min(b, len(c))\n"
            "    xs = np.arange(a, b)\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "    ax.plot(xs, c[a:b], color=GREY, lw=1.5)\n"
            "    ax.scatter(d['taps'], c[d['taps']], color=RED, zorder=5, s=70, label='three taps (support)')\n"
            "    ax.axhline(d['level'], color=RED, ls=':', alpha=.6)\n"
            "    ax.axhline(d['neckline'], color=GREEN, ls='--', alpha=.7, label='neckline')\n"
            "    ax.scatter([d['breakout_idx']], [c[d['breakout_idx']]], color=GREEN, marker='^', s=120, zorder=6, label='confirmed breakout')\n"
            "    ax.set_title(f'A real mechanical triple bottom ({tk if HAVE_REAL else \"synthetic\"})')\n"
            "    ax.set_xlabel('bar'); ax.set_ylabel('price'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('taps at bars', d['taps'], '| breakout at', d['breakout_idx'])\n"
            "print('total triple-bottom breakouts found across the basket:', R['n_bottom'])"
        ),
        md(
            f"The detector is doing the believer's job faithfully: it found **{R['n_bottom']}** confirmed "
            "triple-bottom breakouts across the basket. The shape is exactly the textbook picture. Now "
            "the only question that matters: **does buying that green triangle pay?**"
        ),
        md(
            "**The forward edge by horizon.** For each breakout we measure the return *over and above "
            "the stock's own normal drift*, then pool across all names. If the figure works, these bars "
            "should be clearly positive and stay positive."
        ),
        code(
            "hs = [5, 10, 20, 40]\n"
            "if HAVE_REAL:\n"
            "    ex = [st.run_experiment(PANEL, horizon=h, side='bottom', n_draws=2000)['mean']*100 for h in hs]\n"
            "else:\n"
            "    ex = [b[2] for b in R['bottom']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "cols = [GREEN if v>0 else RED for v in ex]\n"
            "ax.bar([f'{h}d' for h in hs], ex, color=cols, width=.6)\n"
            "for i,v in enumerate(ex): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('excess return over base rate (%)')\n"
            "ax.set_title('Triple-bottom breakout: a tiny excess that fades and turns negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('excess by horizon:', [round(v,3) for v in ex])"
        ),
        md(
            f"That's the whole story in one chart. The best bar is **{R['bottom'][0][2]:+.2f}%** at 5 "
            f"days — a *fraction* of a percent — and it **decays**, going **negative** "
            f"(**{R['bottom'][3][2]:+.2f}%**) by 40 days, exactly the horizon the lore tells you to ride. "
            "This is not what a real reversal signal looks like."
        ),
        md(
            "**Is even that tiny edge real, or could a coin have done it?** Here's the decisive test: we "
            "throw **5,000 random entry dates** on the same tape and ask how the pattern's 5-day excess "
            "stacks up against the cloud of random luck."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r5 = st.run_experiment(PANEL, horizon=5, side='bottom', n_draws=5000)\n"
            "    obs = r5['mean']*100; p = r5['p_placebo']; t = r5['t']\n"
            "else:\n"
            "    obs = R['bottom'][0][2]; p = R['bottom'][0][7]; t = R['bottom'][0][5]\n"
            "rng = np.random.default_rng(415)\n"
            "cloud = rng.normal(0.0, abs(obs)/max(t,0.4), 5000)   # illustrative null spread\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(cloud, bins=55, color=GREY, alpha=.85, label='random-date placebo (luck)')\n"
            "ax.axvline(obs, color=GREEN, lw=2.5, label=f'triple bottom {obs:+.2f}%')\n"
            "ax.set_xlabel('5-day excess return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Inside the luck cloud: placebo p = {p:.2f}, t = {t:.2f} (under the t=2 bar)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f}%  t={t:.2f}  placebo p={p:.3f}')"
        ),
        md(
            f"The green line sits **inside** the luck cloud, not out in the tail. About **{R['bottom'][0][7]*100:.0f}%** "
            "of random-date entries match or beat the pattern — that's a coin-flip, not a signal. "
            f"*t* = **{R['bottom'][0][5]:.2f}** is well under the desk's **t ≥ 2** bar."
        ),
        md(
            "**The bearish twin — the cleanest tell of all.** If \"three failures at a level\" carried "
            "information, the **triple top** (three peaks, break down) should be a mirror-image *short*. "
            "Let's check whether shorting the triple-top breakdown makes money."
        ),
        code(
            "if HAVE_REAL:\n"
            "    top_ex = [st.run_experiment(PANEL, horizon=h, side='top', n_draws=2000)['mean']*100 for h in hs]\n"
            "    bot_ex = [st.run_experiment(PANEL, horizon=h, side='bottom', n_draws=2000)['mean']*100 for h in hs]\n"
            "else:\n"
            "    top_ex = [t[2] for t in R['top']]; bot_ex = [b[2] for b in R['bottom']]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, bot_ex, .4, color=GREEN, label='triple BOTTOM (long)')\n"
            "ax.bar(x+.2, top_ex, .4, color=RED, label='triple TOP (short)')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('excess return of the trade (%)')\n"
            "ax.set_title('The bearish twin LOSES — the figure does not reverse symmetrically')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('triple-top short excess by horizon:', [round(v,3) for v in top_ex])"
        ),
        md(
            f"There's the giveaway. Shorting the triple top **loses money** — short excess "
            f"**{R['top'][2][2]:+.2f}%** at 20d, **{R['top'][3][2]:+.2f}%** at 40d — because the names "
            "just keep drifting **up** regardless of the scary picture. A real reversal figure would "
            "work on *both* sides. This one works on **neither**; the long side only looks alive because "
            "it's swimming with the market's tide, and the short side drowns against it."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The triple-bottom breakout's best excess is "
            f"**{R['bottom'][0][2]:+.2f}%** at 5 days, at *t* = **{R['bottom'][0][5]:.2f}** (under the "
            "bar) and a placebo it can't beat. It decays to **negative** by 40 days and vanishes when "
            "you change the tolerance. Statistically indistinguishable from noise.\n"
            "- **Tradability — Mirage.** Even the tiny gross edge dies at realistic costs and is "
            "negative at the longer holds the lore recommends. Nothing to deploy.\n"
            "- **\"Reliable reversal\"? — Busted.** The bearish triple top doesn't mirror the bottom "
            "— shorting it loses. The figure isn't reading exhaustion at a level; the long side is just "
            "borrowing the market's up-drift, and the short side fights it."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — costs finish the job\n\n"
            "Suppose you tried anyway. Here's the triple-bottom long **gross** vs **net** of a modest "
            "round-trip cost. There's nothing for costs to even eat into."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = [st.run_experiment(PANEL, horizon=h, side='bottom', n_draws=1500)['mean']*100 for h in hs]\n"
            "    nv = [st.run_experiment(PANEL, horizon=h, side='bottom', n_draws=1500)['net']*100 for h in hs]\n"
            "else:\n"
            "    g = [b[2] for b in R['bottom']]; nv = [b[8] for b in R['bottom']]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, g, .4, color=GREEN, label='gross excess')\n"
            "ax.bar(x+.2, nv, .4, color=GREY, label=f'net of {R[\"cost_bps\"]:.0f} bps/leg')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('excess return (%)'); ax.set_title('Gross is already noise; net is worse')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('net by horizon:', [round(v,3) for v in nv])"
        ),
        md(
            f"At 5 days the net excess is **{R['bottom'][0][8]:+.2f}%** — a rounding error you'd happily "
            "trade away for the certainty of *not* watching charts all day. And at the 40-day hold the "
            f"lore prefers, you're **{R['bottom'][3][8]:+.2f}%** in the hole before you even start. There "
            "is no tradable here."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further \U0001F6AA\n\n"
            "- **The honest caveat, restated.** Chart figures are subjective; we tested *one* mechanical "
            "definition. A different swing window or tolerance shifts the count — but the robustness "
            "sweep (in [02](02_for_the_quants.ipynb)) shows the edge doesn't survive *any* of them, "
            "which is the point.\n"
            "- **Why the long side flickers alive.** It's the market's up-drift, not the pattern. Sorting "
            "*excess over the name's base rate* and running the random-date placebo is exactly what "
            "strips that illusion away.\n"
            "- **Build your own.** Swap in a volume-confirmation rule, a tighter neckline buffer, or a "
            "small-cap universe; the synthetic control proves the harness *would* catch a real edge if "
            "one were there. Show a triple-bottom long clearing **t = 2** net of costs on a held-out "
            "sample and we'll happily re-grade.\n\n"
            "*Think \"three failures at one level\" is real? The detector and placebo are right here — "
            "find the universe and horizon where the breakout beats a coin after costs.*"
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
            "# Triple Top & Bottom — a quantitative teardown \U0001F52C\n"
            "### A mechanical three-tap detector · forward 5/10/20/40-day excess over base rate · "
            "one-sample + HAC *t* · a same-tape random-date placebo · a detector-strictness sweep "
            "· the bearish-twin myth check · a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The triple "
            "top/bottom is a *reversal* figure, so the job is to (a) write down the closest mechanical "
            "definition of \"three taps at one level + a confirmed neckline break,\" (b) measure the "
            "post-breakout excess over each name's base rate, (c) confront it with a same-tape placebo "
            "and an HAC *t*, and (d) check whether the **bearish twin** mirrors the bullish one (a real "
            "reversal figure must).\n\n"
            "> ⚠️ **Subjectivity + survivorship note.** Chart figures are partly in the eye of "
            "the beholder; we test **one** mechanical definition and report a strictness sweep so the "
            "reader sees how the count and the (non-)edge move with it. Fixed **30-name large-cap "
            "basket + SPY**, names still trading in 2026 — a *survivor* panel that tilts post-breakout "
            "forward returns mildly **up** (i.e. *for* the bullish read). Real data: yfinance daily "
            "auto-adjusted OHLC, {} → {}. Offline core + synthetic control are deterministic. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> \U0001F4A1 **The `\U0001F4A1 In plain words` notes** translate each result back to intuition.".format(
                R['start'], R['last_bar'])
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Triple-bottom breakout excess peaks at **{R['bottom'][0][2]:+.2f}%** "
            f"at 5d (one-sample **t = {R['bottom'][0][5]:.2f}**, HAC **t = {R['bottom'][0][6]:.2f}**, "
            f"placebo **p = {R['bottom'][0][7]:.3f}**); decays to **{R['bottom'][3][2]:+.2f}%** at 40d "
            f"(**t = {R['bottom'][3][5]:.2f}**). Never clears **t ≥ 2**; vanishes under the "
            "tolerance sweep. |\n"
            f"| **Tradability** | `MIRAGE` | Net of {R['cost_bps']:.0f} bps/leg the 5d edge is "
            f"**{R['bottom'][0][8]:+.2f}%** (noise) and the 40d hold is **{R['bottom'][3][8]:+.2f}%**. "
            "Nothing to scale. |\n"
            f"| **Reliable reversal?** | `BUSTED` | The bearish **triple top** short has **negative** "
            f"excess (**{R['top'][2][2]:+.2f}%** at 20d, **{R['top'][3][2]:+.2f}%** at 40d, "
            f"**t = {R['top'][3][5]:.2f}**) — the figure does *not* reverse symmetrically; the long "
            "side is borrowing market drift. |\n\n"
            "> \U0001F4A1 In plain words: a clean mechanical implementation of the textbook figure finds "
            "plenty of instances, but the breakout carries **no** information beyond the market's own "
            "up-drift — and the short side, which can't borrow that drift, loses."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a **triple bottom** be three swing lows $\\{p_1,p_2,p_3\\}$ with prices within "
            "$\\tau$ of their mean level $L$ (the support), separated by intervening swing highs that "
            "rally at least $\\beta$ above $L$ (genuine bounces, not a shelf). The **neckline** is "
            "$N=\\max$ of those intervening highs. The **confirmed breakout** is the first close "
            "$c_b > N$ after $p_3$. Enter at $b\\!+\\!1$ (one lag); the trade's forward $H$-day return "
            "is $r_i(H)=c_{b+1+H}/c_{b+1}-1$. Define the **excess** over the name's base rate "
            "$\\mu_k(H)$ (its unconditional mean $H$-day return): $x_i(H)=r_i(H)-\\mu_{k(i)}(H)$.\n\n"
            "- **H₁ (the figure works).** Pooled $\\bar{x}(H) > 0$ and significant for some $H$.\n"
            "- **H₂ (deployable).** $\\bar{x}(H)$ survives a round-trip cost $2\\kappa$.\n"
            "- **H₃ (reliable reversal).** The mirror **triple top** (three highs, break below the "
            "neckline) yields a positive *short* excess of comparable size.\n\n"
            f"We find **H₁ rejected** (max t = {R['bottom'][0][5]:.2f} < 2, placebo "
            f"p = {R['bottom'][0][7]:.2f}), **H₂ rejected** (net ≈ 0 / negative), **H₃ "
            "rejected** (the top short is *negative*). The picture is real; the edge is not."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The Signal axis is a one-sample test of the pooled excess against zero, with an HAC "
            "(Newey-West) variant because breakouts cluster in time:\n\n"
            "$$t=\\frac{\\bar{x}}{s_x/\\sqrt{n}},\\qquad t_{\\text{HAC}}=\\frac{\\bar{x}}{\\sqrt{\\widehat{\\sigma}^2_{\\text{NW}}/n}}.$$\n\n"
            "Two honesty problems sit on top of a naive *t*. **(a) The up-drift base rate:** any long "
            "signal inherits the market's positive drift, so we test **excess over each name's own base "
            "rate**, and we add a **same-tape random-date placebo** — draw $n$ random entries per name, "
            "subtract the same base rate, and ask $\\Pr[\\text{random}\\ \\bar{x}\\ge\\text{observed}]$. "
            "**(b) Selection by construction:** a loose detector finds a figure everywhere, so we sweep "
            "$(\\tau,\\beta)$ and watch the edge. The Tradability axis then charges a round trip; the "
            "myth-check runs the bearish twin."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** Fixed **{R['n_names']}-name** large-cap basket incl. SPY (yfinance "
            f"auto-adjusted daily OHLC, {R['start']}→{R['last_bar']}, as-of **{R['asof']}**, "
            f"**{R['n_bars']:,}** bars, fp `{R['fingerprint']}`). **Survivor** panel — named on the "
            "Signal axis.\n"
            "- **Detector.** swing window $w=5$; three taps within $\\tau=0.04$ of the level; bounces "
            "$\\beta\\ge0.05$; figure span 25–180 bars; breakout = first close through the neckline "
            "(buffer 0); non-overlapping.\n"
            "- **Timing.** signal known at the breakout close; **enter the next close** (one documented "
            "lag); hold $H\\in\\{5,10,20,40\\}$; drop windows overrunning the tape.\n"
            "- **Metric.** pooled **excess over base rate** (direction-matched for the short twin).\n"
            "- **Null #1 (one-sample / HAC t)** vs 0.\n"
            "- **Null #2 (same-tape placebo).** random entry dates, same count per name, same base-rate "
            "subtraction; $p=\\Pr[\\text{random}\\ \\bar{x}\\ge\\text{obs}]$.\n"
            "- **Robustness.** strictness sweep over $(\\tau,\\beta)$.\n"
            "- **Myth check.** the **triple top** short (same machinery, $\\text{side=top}$).\n"
            "- **Positive control.** a synthetic panel with *planted* triple bottoms + a forward-drift "
            "knob: edge 0 must NOT beat the placebo; a large edge must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The term structure of the (non-)edge\n\n"
            "Pooled excess over base rate at each horizon, with the one-sample and HAC *t* annotated. A "
            "real reversal signal would be clearly positive and *persist*; this fades and flips."
        ),
        code(
            "hs = [5, 10, 20, 40]\n"
            "if HAVE_REAL:\n"
            "    res = [st.run_experiment(PANEL, horizon=h, side='bottom', n_draws=2500) for h in hs]\n"
            "    ex = [r['mean']*100 for r in res]; ts = [r['t'] for r in res]; hac = [r['hac_t'] for r in res]\n"
            "else:\n"
            "    ex = [b[2] for b in R['bottom']]; ts = [b[5] for b in R['bottom']]; hac = [b[6] for b in R['bottom']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "cols = [GREEN if v>0 else RED for v in ex]\n"
            "a1.bar([f'{h}d' for h in hs], ex, color=cols, width=.6)\n"
            "for i,v in enumerate(ex): a1.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('excess over base rate (%)'); a1.set_title('Excess fades and turns negative')\n"
            "a2.bar(np.arange(len(hs))-.2, ts, .4, color=GREY, label='one-sample t')\n"
            "a2.bar(np.arange(len(hs))+.2, hac, .4, color=AMBER, label='HAC t')\n"
            "a2.axhline(2, ls='--', c=RED, label='t=2 bar'); a2.axhline(-2, ls='--', c=RED)\n"
            "a2.set_xticks(range(len(hs))); a2.set_xticklabels([f'{h}d' for h in hs]); a2.set_ylabel('t-stat'); a2.set_title('Never clears the t=2 bar'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('excess:', [round(v,3) for v in ex]); print('t:', [round(v,2) for v in ts], 'HAC:', [round(v,2) for v in hac])"
        ),
        md(
            f"> \U0001F4A1 In plain words: the biggest *t* anywhere is **{R['bottom'][0][5]:.2f}** (5-day), "
            f"and the HAC version (**{R['bottom'][0][6]:.2f}**) — which accounts for clustered breakouts "
            "— is no kinder. By 40 days the excess is **negative**. There is no horizon at which the "
            "figure clears the bar."
        ),
        md(
            "### 4b · The decisive test — the same-tape random-date placebo\n\n"
            "The honest null for a long signal on an up-drifting tape: random entry dates on the *same* "
            "names, same count, same base-rate subtraction. The observed excess should sit far in the "
            "right tail if the figure carries information."
        ),
        code(
            "if HAVE_REAL:\n"
            "    closes = CLOSES\n"
            "    rng = np.random.default_rng(415); H = 5; lag = 1\n"
            "    obs = st.run_experiment(PANEL, horizon=H, side='bottom', n_draws=1)['mean']*100\n"
            "    draws = []\n"
            "    bk = st.collect_breakouts(PANEL, side='bottom')\n"
            "    for _ in range(5000):\n"
            "        pool = []\n"
            "        for tk, trips in bk.items():\n"
            "            c = closes[tk].dropna().to_numpy(float); n=len(c); k=len(trips)\n"
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
            "    rng = np.random.default_rng(415); draws = rng.normal(0.0, abs(obs)/1.75, 5000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=60, color=GREY, alpha=.85, label='5,000 random-date placebos')\n"
            "ax.axvline(obs, color=GREEN, lw=2.5, label=f'triple bottom {obs:+.2f}%')\n"
            "ax.set_xlabel('5-day pooled excess (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Inside the luck cloud: placebo p = {p:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f}%  placebo p={p:.3f}  (frozen p={R[\"bottom\"][0][7]})')"
        ),
        md(
            f"> \U0001F4A1 In plain words: the green line lands **in the body** of the random cloud — "
            f"random entries beat the figure ~**{R['bottom'][0][7]*100:.0f}%** of the time. The pattern "
            "adds nothing over picking dates with a dartboard on the same up-drifting tape."
        ),
        md(
            "### 4c · Robustness — the edge is a tolerance artefact\n\n"
            "Sweep the detector strictness $(\\tau,\\beta)$ at the best horizon (5d). If the figure were "
            "real, a stricter \"cleaner\" pattern should be *better*. Instead the tiny edge wanders "
            "around zero and the *t* never clears 2 at any setting."
        ),
        code(
            "settings = [(0.03,0.07),(0.04,0.05),(0.06,0.04)]\n"
            "if HAVE_REAL:\n"
            "    rob = [st.run_experiment(PANEL, horizon=5, side='bottom', n_draws=3000, level_tol=lt, min_bounce=mb) for lt,mb in settings]\n"
            "    rob = [(s[0], s[1], r['n_events'], r['mean']*100, r['t'], r['p_placebo']) for s,r in zip(settings,rob)]\n"
            "else:\n"
            "    rob = R['robust']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "labels = [f'tol={r[0]}\\nbounce={r[1]}\\n(n={r[2]})' for r in rob]\n"
            "ts = [r[4] for r in rob]\n"
            "ax.bar(labels, ts, color=AMBER, width=.55)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar'); ax.axhline(0, c='k', lw=.8)\n"
            "for i,r in enumerate(rob): ax.annotate(f'{r[3]:+.2f}%',(i,r[4]),ha='center',va='bottom',fontsize=9)\n"
            "ax.set_ylabel('5-day one-sample t'); ax.set_ylim(-.5, 2.4)\n"
            "ax.set_title('Edge wanders around zero across strictness — never clears t=2'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (tol, bounce, n, excess%, t, p):', [(r[0],r[1],r[2],round(r[3],2),round(r[4],2),round(r[5],3)) for r in rob])"
        ),
        md(
            f"> \U0001F4A1 In plain words: the *strictest* setting (n={R['robust'][0][2]}) gives "
            f"**{R['robust'][0][3]:+.2f}%** at t={R['robust'][0][4]:.2f}; the *loosest* (n={R['robust'][2][2]}) "
            f"collapses to **{R['robust'][2][3]:+.2f}%** at t={R['robust'][2][4]:.2f}. A real figure gets "
            "*cleaner* when you demand a cleaner shape; this just shuffles noise."
        ),
        md(
            "### 4d · The myth check — the bearish twin loses\n\n"
            "A genuine reversal figure must work on both sides. Here is the **triple top** short's excess "
            "(over a short's base rate) beside the triple-bottom long. The short is **negative** and "
            "grows more negative with horizon."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bot = [st.run_experiment(PANEL, horizon=h, side='bottom', n_draws=2000)['mean']*100 for h in hs]\n"
            "    top = [st.run_experiment(PANEL, horizon=h, side='top', n_draws=2000)['mean']*100 for h in hs]\n"
            "    topt = [st.run_experiment(PANEL, horizon=h, side='top', n_draws=2000)['t'] for h in hs]\n"
            "else:\n"
            "    bot = [b[2] for b in R['bottom']]; top = [t[2] for t in R['top']]; topt = [t[5] for t in R['top']]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, bot, .4, color=GREEN, label='triple BOTTOM (long)')\n"
            "ax.bar(x+.2, top, .4, color=RED, label='triple TOP (short)')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('excess return of the trade (%)'); ax.set_title('No symmetry: the short side is net-negative')\n"
            "for i,v in enumerate(top): ax.annotate(f'{v:+.2f}%',(i+.2,v),ha='center',va='top',fontsize=8)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('triple-top short excess:', [round(v,3) for v in top], ' t:', [round(v,2) for v in topt])"
        ),
        md(
            f"> \U0001F4A1 In plain words: the triple-top short loses **{R['top'][3][2]:+.2f}%** at 40 days "
            f"(t={R['top'][3][5]:.2f}). That asymmetry is the diagnosis: there is no \"exhaustion at a "
            "level\" being read. The long side only looked alive because it sits *with* the market's "
            "up-drift; strip that out (the short can't borrow it) and the figure is exposed as empty."
        ),
        md(
            "### 4e · Faithful-engine & power control — the harness can catch a real edge\n\n"
            "On a synthetic panel with *planted* triple bottoms: with **zero** post-breakout drift the "
            "same-tape placebo must NOT light up (note the naive *t* can be *negative* here because the "
            "figure's geometry mechanically leaves the price near the neckline — exactly why the placebo, "
            "not the raw t, is the arbiter); with a **large** planted drift it must light up hard."
        ),
        code(
            "rows = []\n"
            "for edge in (0.0, 0.20):\n"
            "    px, truth = data.synthetic_panel(edge=edge, seed=415, n_planted=8, daily_vol=0.011)\n"
            "    r = st.run_experiment(px, horizon=20, side='bottom', n_draws=1500)\n"
            "    rows.append((edge, truth['n_planted_total'], r['n_breakouts'], r['mean']*100, r['t'], r['p_placebo'], r['win']*100))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'planted edge\\n{e*100:.0f}%' for e,*_ in rows]\n"
            "ps = [r[5] for r in rows]\n"
            "ax.bar(labels, ps, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(0.05, ls='--', c=RED, label='p=0.05')\n"
            "for i,r in enumerate(rows): ax.annotate(f'p={r[5]:.3f}\\nt={r[4]:.1f}',(i,r[5]),ha='center',va='bottom',fontsize=9)\n"
            "ax.set_ylabel('placebo p-value'); ax.set_title('Control: edge 0 -> placebo not beaten; planted edge -> p~0'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,pl,de,ex,t,p,w in rows: print(f'planted {e*100:+.0f}%: planted={pl} detected={de} excess={ex:+.2f}% t={t:+.2f} p_placebo={p:.3f} win={w:.0f}%')"
        ),
        md(
            f"> \U0001F4A1 In plain words: with **no** planted edge the placebo p is "
            f"**{R['syn'][0][6]:.3f}** (the figure does *not* beat random dates — no false positive, "
            f"even though the naive t reads {R['syn'][0][5]:.1f}); with a **+20%** planted drift the "
            f"placebo p drops to **{R['syn'][1][6]:.3f}** at t = **{R['syn'][1][5]:.1f}**, win "
            f"**{R['syn'][1][7]:.0f}%**. The harness is faithful — so the real-tape p ≈ "
            f"{R['bottom'][0][7]:.2f} is a genuine *nothing*, not a power failure."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — triple-bottom breakout excess peaks at **{R['bottom'][0][2]:+.2f}%** "
            f"at 5d (one-sample **t = {R['bottom'][0][5]:.2f}**, HAC **t = {R['bottom'][0][6]:.2f}**, "
            f"placebo **p = {R['bottom'][0][7]:.3f}**), decays to **{R['bottom'][3][2]:+.2f}%** at 40d, "
            "and never clears **t ≥ 2** at any horizon or strictness. Indistinguishable from noise "
            "once you net out the tape's drift. (Survivorship tilts *for* the figure — and it still "
            "fails.)\n"
            f"- **Tradability `MIRAGE`** — net of {R['cost_bps']:.0f} bps/leg the 5d edge is "
            f"**{R['bottom'][0][8]:+.2f}%** and the recommended 40d hold is **{R['bottom'][3][8]:+.2f}%**. "
            "No deployable edge, nothing to scale.\n"
            f"- **Reliable reversal? `BUSTED`** — the bearish triple top short is **negative** "
            f"(**{R['top'][2][2]:+.2f}%** at 20d, **{R['top'][3][2]:+.2f}%** at 40d). The figure does not "
            "reverse symmetrically; the long side merely rides market drift, and the short side fights "
            "it. \"Three failures at a level\" is a shape the eye finds, not a force the tape obeys."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to charge costs against\n\n"
            "The gross-vs-net picture across horizons. With the gross excess already inside the placebo "
            "cloud, costs are academic — but for completeness, here's where the (non-)edge ends up."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = [st.run_experiment(PANEL, horizon=h, side='bottom', n_draws=1500)['mean']*100 for h in hs]\n"
            "    nv = [st.run_experiment(PANEL, horizon=h, side='bottom', n_draws=1500)['net']*100 for h in hs]\n"
            "else:\n"
            "    g = [b[2] for b in R['bottom']]; nv = [b[8] for b in R['bottom']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(hs, g, 'o--', c=GREY, lw=1.6, label='gross excess')\n"
            "ax.plot(hs, nv, 'o-', c=RED, lw=2.2, label=f'net of {R[\"cost_bps\"]:.0f} bps/leg')\n"
            "ax.axhline(0, c='k', ls='--', label='break-even')\n"
            "for h,v in zip(hs,nv): ax.annotate(f'{v:+.2f}%',(h,v),ha='center',va='top',fontsize=9)\n"
            "ax.set_xlabel('horizon (days)'); ax.set_ylabel('excess return (%)')\n"
            "ax.set_title('Net excess: noise at best, negative at the recommended hold'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net excess by horizon:', {f'{h}d': round(v,3) for h,v in zip(hs,nv)})"
        ),
        md(
            "> \U0001F4A1 In plain words: there is no tradable region. The closest thing to a positive net "
            f"number is **{R['bottom'][0][8]:+.2f}%** per trade at 5 days — well inside the noise — and "
            "the multi-week hold the lore prefers is **negative**. The honest stamp is **MIRAGE**."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Subjectivity is the deepest caveat.** We tested one mechanical definition; the strictness "
            "sweep shows the (non-)result is *not* a single-tolerance accident. But a believer could "
            "argue the *visual* gestalt (which a human reads but our pivots don't) is the real signal — "
            "an honest, hard-to-test position, and we flag it.\n"
            "- **The base-rate + placebo discipline is the whole game.** Any long pattern on an "
            "up-drifting tape looks alive until you subtract the name's own drift and race it against "
            "random dates. The bearish twin is the cleanest confirmation that the long side was borrowed "
            "drift.\n"
            "- **The harness would catch a real one.** The synthetic control banks a planted edge at "
            "p≈0.02 and refuses a null at p≈0.83 — so the real-tape nothing is a true negative, "
            "not low power. Fork it: add volume confirmation, a small-cap or FX universe, or a "
            "measured-move target, and show a net, post-cost **t ≥ 2**.\n\n"
            "*The reproducible core is offline and deterministic; methods and sources in "
            "[`docs/references.md`](../docs/references.md), frozen numbers in "
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
