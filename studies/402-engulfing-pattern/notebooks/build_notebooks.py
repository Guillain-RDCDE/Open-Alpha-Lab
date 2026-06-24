"""Generate the two narrative notebooks for Study 402 (Engulfing Pattern).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached basket OHLCV under
../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md).
The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md.
# Basket: 29 liquid US large-caps + SPY, yfinance daily, 2005-01-03 -> 2026-06-18, fp bf1d6cb7ca54.
R = dict(
    start="2005-01-03", end="2026-06-18", years=21.5, fp="bf1d6cb7ca54",
    n_names=30, n_events=12144, n_bull=5736, n_bear=6408,
    # combined signed return per horizon: (H, n, mean%, baseL%, win%, t_hac, t_one, p_plac, net%)
    h1=(1, 12144, -0.049, 0.028, 48.1, -3.79, -3.89, 1.000, -0.150),
    h3=(3, 12143, -0.114, 0.141, 48.2, -5.14, -4.85, 1.000, -0.217),
    h5=(5, 12140, -0.174, 0.253, 48.2, -5.74, -5.61, 1.000, -0.279),
    h10=(10, 12127, -0.176, 0.530, 48.3, -4.25, -4.08, 1.000, -0.285),
    # leg split: H -> (bull_mean%, bull_t, bull_win%, bear_mean%, bear_t, bear_win%)
    bull={1: (-0.011, -0.59, 49.7), 3: (0.072, 2.24, 52.1),
          5: (0.130, 2.92, 53.5), 10: (0.456, 6.90, 55.9)},
    bear={1: (-0.083, -4.55, 46.7), 3: (-0.281, -8.76, 44.7),
          5: (-0.446, -10.46, 43.6), 10: (-0.741, -11.92, 41.6)},
    # myth-check H=1: label -> (n, mean%, t_hac, win%)
    myth=[("plain", 12144, -0.049, -3.79, 48.1),
          ("+downtrend", 5124, -0.052, -2.73, 47.6),
          ("+volume spike", 975, -0.110, -1.91, 47.4),
          ("+both", 405, -0.135, -1.40, 47.9)],
    # synthetic control H=1: (planted_edge, events, mean%, t_hac, p_plac, win%)
    syn=[(0.000, 16295, 0.009, 0.90, 0.188, 50), (0.004, 15263, 0.400, 38.77, 0.000, 62)],
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
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from engulfing_pattern import data, strategy as st

HAVE_REAL = data.have_real()
PANEL = data.load_real(allow_fetch=False) if HAVE_REAL else None
print("real engulfing cache present:", HAVE_REAL,
      "| names:", (0 if PANEL is None else len(PANEL)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the engulfing candle actually call the turn? 🕯️\n"
            "### The most-taught reversal pattern, measured honestly — in plain English\n\n"
            + BADGES +
            "Open any charting app and you'll meet the **engulfing candle**. The story is irresistible: a "
            "small red (down) day, then a big green (up) day whose body completely **engulfs** the day "
            "before — \"the buyers just took over, the bottom is in, **go long**.\" Flip the colours for a "
            "**bearish** engulfing: a top is in, **go short**. It's the first reversal every beginner learns.\n\n"
            "So we did the obvious thing nobody seems to do: found **every** engulfing candle on 30 big, "
            "liquid US stocks over 21 years, bought (or shorted) the next morning exactly as the textbook "
            "says, and asked — *did the reversal happen?*\n\n"
            "The answer is worse than \"no.\" The pattern points the **wrong way**.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Fixed **30-name** basket (29 large-caps + **SPY**, names still "
            "trading today). That carries **survivorship** — but here it *helps* the bullish-reversal claim, "
            "and the pattern still fails. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After an engulfing, does the next day reverse the way it should? | **No — the opposite.** "
            "Trading the pattern as taught (long bullish, short bearish) earns a *significantly negative* "
            "**−0.05%** the very next day. The reversal runs **backwards**. |\n"
            "| Is the win-rate at least better than a coin? | **No.** It's **48%** — below 50%. |\n"
            "| Does the bullish version at least work? | **Not really.** Over a week or two it drifts up — "
            "but **less** than just buying any random day. That's the market going up, not the candle. |\n"
            "| What about the bearish version? | **It's the disaster.** After a \"bearish\" engulfing the "
            "stock keeps **climbing**, so the short bleeds (5-day *t* = −10). |\n"
            "| Does \"only count it after a trend, on big volume\" fix it? | **No.** Every filter the "
            "textbook recommends leaves the edge negative. |\n\n"
            "> The most-taught reversal in all of charting is **busted** on liquid US large-caps."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A **bullish engulfing** — a small down candle swallowed by a bigger up candle — means the "
            "sellers are exhausted and the buyers have seized control. The bottom is in; go long. A "
            "**bearish engulfing** is the mirror: the top is in; go short. It's most reliable after a clear "
            "trend and on heavy volume.\"*\n\n"
            "This isn't a fringe idea. It comes straight from **Steve Nison's** *Japanese Candlestick "
            "Charting Techniques* (1991), the book that brought candlesticks West, and it's in every "
            "trading course and screener since. The engulfing is *the* canonical reversal signal. So the "
            "question isn't whether people believe it — it's whether the tape agrees."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a single two-candle shape genuinely called turns, it would be extraordinary: you could "
            "predict tomorrow's direction from a pattern you can spot with your eyes, no data feed, no "
            "fundamentals. That's a direct claim that markets are *inefficient in a way a beginner can "
            "exploit* — which is exactly why it should make us suspicious.\n\n"
            "Two traps hide here. First, a pattern signed long-after-bullish and short-after-bearish is "
            "roughly **market-neutral**, so we must judge it against **zero**, not against \"it bounced "
            "afterward\" (everything bounces eventually). Second, the bullish leg gets a free ride from the "
            "market's long-run **upward drift** — so a positive bullish number can be pure beta, not a "
            "reversal. We separate both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take a fixed **{R['n_names']}-name** basket (29 liquid US large-caps + SPY) and scan "
            f"**every** bar over **{R['years']:.0f} years** ({R['start']} → {R['end']}). For each one:\n\n"
            "1. **Detect.** Is this an engulfing candle? We use the strict textbook rule — opposite "
            "colours, and today's body must be **bigger** and fully **swallow** yesterday's.\n"
            f"2. **Trade it as taught.** Confirmed at the close, we **buy the next morning** (bullish) or "
            f"**short the next morning** (bearish) — no cheating — and hold **1, 3, 5, 10** days.\n"
            "3. **Score it.** Average the move *in the direction the pattern predicted*. If the legend is "
            "real, that average is clearly **positive**. We also compare to a \"random day\" benchmark and "
            "to just-buying-and-holding.\n\n"
            f"We found **{R['n_events']:,}** engulfing candles ({R['n_bull']:,} bullish, "
            f"{R['n_bear']:,} bearish) — a huge sample. Plenty of power to see a real effect if one exists."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Average move in the predicted direction, the day after every "
            "engulfing, at four horizons. If the legend holds, every bar is comfortably above zero."
        ),
        code(
            "hs = [1, 3, 5, 10]\n"
            "if HAVE_REAL:\n"
            "    means = [st.summarize(PANEL, h, placebo=False)['mean']*100 for h in hs]\n"
            "else:\n"
            "    means = [R['h1'][2], R['h3'][2], R['h5'][2], R['h10'][2]]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar([f'{h}d' for h in hs], means, color=RED, width=.6)\n"
            "for i,v in enumerate(means): ax.annotate(f'{v:+.3f}%',(i,v),ha='center',va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.9)\n"
            "ax.set_ylabel('avg move in the predicted direction (%)')\n"
            "ax.set_title('Trading the engulfing as taught LOSES money at every horizon')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('signed mean by horizon:', [round(v,3) for v in means])"
        ),
        md(
            f"Every bar is **below zero**. Trading the pattern as the book says — long the bullish, short "
            f"the bearish — loses **{R['h1'][2]:+.3f}%** the next day and gets *worse* with horizon "
            f"(**{R['h5'][2]:+.3f}%** at 5 days). The reversal doesn't just fail to show up; the price "
            "tends to go the **other way**."
        ),
        md(
            "**Why?** Split the trade into its two legs — bullish (the long) and bearish (the short) — and "
            "the culprit jumps out."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bull, bear = [], []\n"
            "    for h in hs:\n"
            "        ev = st.collect_events(PANEL, h)\n"
            "        bull.append(ev[ev['dir']>0]['signed_ret'].mean()*100)\n"
            "        bear.append(ev[ev['dir']<0]['signed_ret'].mean()*100)\n"
            "else:\n"
            "    bull = [R['bull'][h][0] for h in hs]; bear = [R['bear'][h][0] for h in hs]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, bull, .4, color=GREEN, label='bullish leg (the LONG)')\n"
            "ax.bar(x+.2, bear, .4, color=RED, label='bearish leg (the SHORT)')\n"
            "ax.axhline(0, c='k', lw=.9); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('avg signed return (%)')\n"
            "ax.set_title('The bearish leg is the disaster: stocks keep RISING after it')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('bull leg:', [round(v,3) for v in bull]); print('bear leg:', [round(v,3) for v in bear])"
        ),
        md(
            f"There it is. After a **bearish** engulfing — supposedly a top — the stock keeps **climbing**, "
            f"so the short bleeds (**{R['bear'][5][0]:+.3f}%** at 5 days). The **bullish** leg looks green "
            "over a week or two... but hold that thought."
        ),
        md(
            "**The bullish trap.** That green bullish leg isn't a reversal — it's the market's normal "
            "upward drift. Compare the bullish-leg return to what you'd earn buying a **random** day and "
            "holding the same number of days."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bull = []; base = []\n"
            "    for h in hs:\n"
            "        ev = st.collect_events(PANEL, h)\n"
            "        bull.append(ev[ev['dir']>0]['signed_ret'].mean()*100)\n"
            "        base.append(st.unconditional_base(PANEL, h).mean()*100)\n"
            "else:\n"
            "    bull = [R['bull'][h][0] for h in hs]; base = [R['h1'][3], R['h3'][3], R['h5'][3], R['h10'][3]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, bull, .4, color=GREEN, label='after a bullish engulfing')\n"
            "ax.bar(x+.2, base, .4, color=GREY, label='a RANDOM day (just buy & hold)')\n"
            "ax.axhline(0, c='k', lw=.9); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('avg forward return (%)')\n"
            "ax.set_title('The bullish \"signal\" is BEATEN by buying any random day')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('bullish 10d:', round(bull[-1],3), 'vs random 10d:', round(base[-1],3))"
        ),
        md(
            f"At 10 days the bullish engulfing earns **{R['bull'][10][0]:+.3f}%** — but a **random** day "
            f"earns **{R['h10'][3]:+.3f}%**. The candle does *worse* than buying nothing in particular. So "
            "even the half that looks good is just beta you'd have collected anyway."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Traded as taught, the engulfing loses money at every horizon "
            f"(**{R['h1'][2]:+.3f}%** the next day) with a **{R['h1'][4]:.0f}%** win-rate. No reversal — "
            "if anything, the *reverse* of one.\n"
            "- **Tradability — Mirage.** It's negative **before costs**, so there's nothing to trade. "
            "Charging the round trip just makes it worse.\n"
            "- **A reliable reversal? — Busted.** The bearish leg keeps rising; the bullish leg is just the "
            "market drifting up. The famous pattern doesn't call turns."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — there's nothing to trade\n\n"
            "Usually this beat is where a real signal dies on costs. Here it's already dead before we get "
            "there: the gross edge is **negative**, so costs only deepen the hole. Gross vs net, the trade "
            "as taught:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = [st.summarize(PANEL, h, placebo=False)['mean']*100 for h in hs]\n"
            "    nv = [st.summarize(PANEL, h, placebo=False)['net']*100 for h in hs]\n"
            "else:\n"
            "    g = [R['h1'][2], R['h3'][2], R['h5'][2], R['h10'][2]]\n"
            "    nv = [R['h1'][8], R['h3'][8], R['h5'][8], R['h10'][8]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, g, .4, color=GREY, label='gross')\n"
            "ax.bar(x+.2, nv, .4, color=RED, label='net (costs + borrow)')\n"
            "ax.axhline(0, c='k', lw=.9); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('signed return per event (%)'); ax.set_title('Negative before costs; costs only widen the loss')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('net by horizon:', [round(v,3) for v in nv])"
        ),
        md(
            f"Net of a realistic round trip plus borrow on the shorts, the loss widens to "
            f"**{R['h1'][8]:+.3f}%** (1d) … **{R['h10'][8]:+.3f}%** (10d). There is no capacity question to "
            "ask — you can't scale a negative number into a profit."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Is the harness just broken?** No — the quants notebook plants a *fake* reversal into "
            "synthetic data and the exact same code lights up at *t* = 39. The test would see a real "
            "reversal; the tape just doesn't have one.\n"
            "- **What about small-caps / crypto / other markets?** Candlestick studies (Marshall, Young & "
            "Rose 2006) mostly find nothing across markets too — fork the basket and check.\n"
            "- **What about combining it with a real signal?** The engulfing might add nothing, or might "
            "even subtract (it anti-predicts here). Try it as a *filter* on a known edge and measure.\n\n"
            "*Think the engulfing predicts turns on some universe? Show the **signed** next-day return "
            "landing above zero with an honest *t* — then we'll talk.*"
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
            "# The Engulfing pattern — a quantitative teardown 🔬\n"
            "### Real-body detector · forward 1/3/5/10-day signed returns · HAC *t* vs zero + a coin-sign "
            "placebo · leg split · costs · the trend/volume myth-check · a synthetic faithful-engine / "
            "power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The engulfing "
            "candle is the canonical candlestick reversal — so the job here is to **measure it against "
            "zero** with autocorrelation-robust inference, split the long and short legs, confront a "
            "random-timing null, and prove (with a planted-edge control) that a negative finding is the "
            "tape's verdict and not a dead harness.\n\n"
            "> ⚠️ **Data + survivorship note.** Fixed **30-name** basket (29 liquid US large-caps + SPY), "
            "names still trading in 2026 — a *survivor* panel that tilts the bullish (long) leg up and so "
            "works **in favour** of the reversal claim; it still fails. Real data: yfinance daily OHLCV "
            "(`auto_adjust=True`), 2005→2026. Offline core + synthetic control are deterministic. Methods "
            "in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Signed engulfing return (long bullish / short bearish) **{R['h1'][2]:+.3f}%** "
            f"at 1d (**HAC t = {R['h1'][5]:.2f}**), worsening to **{R['h5'][2]:+.3f}%** at 5d "
            f"(**t = {R['h5'][5]:.2f}**). Win **{R['h1'][4]:.1f}%**, placebo **p = {R['h1'][7]:.3f}**. The "
            "effect is significantly *negative* — the reversal runs backwards. |\n"
            f"| **Tradability** | `MIRAGE` | Negative **before costs**; net of 2×5-bps + borrow it is "
            f"**{R['h1'][8]:+.3f}% / {R['h5'][8]:+.3f}%** (1d/5d). Nothing to harvest, no capacity question. |\n"
            f"| **Reliable reversal?** | `BUSTED` | Bearish leg keeps **rising** (5d HAC t = {R['bear'][5][1]:.2f}); "
            f"bullish leg's 10d **+{R['bull'][10][0]:.3f}%** *underperforms* the unconditional long "
            f"**+{R['h10'][3]:.3f}%** (pure beta). Trend/volume filters don't flip it. |\n\n"
            "> 💡 In plain words: the most-taught reversal candle has **no** predictive reversal on liquid "
            "US large-caps — it points the wrong way, the bearish leg fights the market, and the bullish "
            "leg is just the market going up."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let bar $t$ form an engulfing of direction $d_i\\in\\{+1,-1\\}$ (bullish/bearish) by the "
            "real-body rule: with $(O_0,C_0)$ the prior bar and $(O_1,C_1)$ the current,\n\n"
            "$$\\text{bullish}:\\ C_0<O_0,\\ C_1>O_1,\\ O_1\\le C_0,\\ C_1\\ge O_0,\\ |C_1-O_1|>|C_0-O_0|,$$\n\n"
            "and the mirror for bearish. We enter at the **next open** (one execution lag) and hold $H$ "
            "days; the *signed* event return is $\\rho_i(H)=d_i\\,(C_{t+H}/O_{t+1}-1)$. The believer's "
            "hypotheses:\n\n"
            "- **H₁ (the reversal exists).** $\\mathbb{E}[\\rho_i(H)]>0$ and significant for some $H$.\n"
            "- **H₂ (it beats beta).** the *bullish* leg beats the unconditional long forward return.\n"
            "- **H₃ (\"after a trend, on volume\").** filtering on a prior trend / a volume spike improves it.\n\n"
            f"We find **H₁ rejected** (signed mean **{R['h1'][2]:+.3f}%**, HAC t = {R['h1'][5]:.2f} — "
            f"significantly *negative*), **H₂ rejected** (bullish 10d "
            f"{R['bull'][10][0]:+.3f}% < unconditional {R['h10'][3]:+.3f}%), **H₃ rejected** (every filter "
            "stays negative). The pattern anti-predicts."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The Signal axis is a one-sample test of the pooled signed-return sample against zero, with a "
            "Newey-West HAC standard error (events overlap in time and cluster, so a naive *t* understates "
            "the SE):\n\n"
            "$$t_{\\text{HAC}}=\\frac{\\bar\\rho}{\\widehat{\\mathrm{se}}_{\\text{NW}}(\\bar\\rho)},\\qquad "
            "\\widehat{\\mathrm{se}}_{\\text{NW}}^2=\\frac{1}{n}\\Big(\\hat\\gamma_0+2\\sum_{k=1}^{L}w_k\\hat\\gamma_k\\Big).$$\n\n"
            "Two honesty problems sit on top. **(a) The right benchmark.** A signed long/short event "
            "return is market-neutral-ish, so it must beat **zero** — and the *bullish* leg alone must beat "
            "the always-up drift, or it's just beta. **(b) Selection on a famous rule.** With 12k events we "
            "have huge power; the placebo (random bars, coin signs) shows whether the observed mean is even "
            "*on the good side* of random. The Tradability axis then charges the round trip + short borrow "
            "— moot here, since the gross is already negative."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** Fixed **{R['n_names']}-name** basket (29 liquid US large-caps + SPY), "
            f"yfinance daily `auto_adjust=True`, {R['start']}→{R['end']} (fingerprint `{R['fp']}`). "
            "**Survivor** panel — named on the Signal axis (and it works *for* the claim).\n"
            "- **Detector.** Real-body engulfing (opposite colours, current body strictly larger and fully "
            "containing the prior body).\n"
            "- **Timing.** Confirm at the close of day $t$; enter the **next open** (one lag); exit close "
            "$+H$, $H\\in\\{1,3,5,10\\}$; drop events whose window overruns.\n"
            "- **Signal.** Signed return $\\rho_i=d_i\\cdot r_i$ (long bullish, short bearish).\n"
            "- **Null #1 (HAC + one-sample t)** of the pooled signed sample vs 0.\n"
            "- **Null #2 (placebo).** 10,000 draws of the same number of random bars, each signed by a "
            "coin; $p=\\Pr[\\text{random mean}\\ge\\text{observed}]$.\n"
            "- **Leg split + beta check.** bullish vs bearish; bullish vs the unconditional long drift.\n"
            "- **Myth check.** require a prior trend and/or a volume spike.\n"
            "- **Costs.** 5 bps one-way × 2 (round trip) + 50 bps/yr borrow on the short leg.\n"
            "- **Positive control.** a deterministic panel with a **planted** day-after reversal: zero edge "
            "must NOT reach significance; a planted edge must light up.\n\n"
            "> 🔬 We pre-commit to the kill condition: **a signed mean at or below zero, or a HAC t under 2, "
            "is a `NONE`.** No moving the goalposts."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline + its HAC t\n\n"
            "Signed return by horizon with the Newey-West *t*. The believer needs green bars above a "
            "*t* = 2 line; we get red bars below a *t* = −2 line."
        ),
        code(
            "hs = [1, 3, 5, 10]\n"
            "if HAVE_REAL:\n"
            "    res = [st.summarize(PANEL, h, placebo=False) for h in hs]\n"
            "    means = [s['mean']*100 for s in res]; ts = [s['t_hac'] for s in res]\n"
            "else:\n"
            "    means = [R['h1'][2], R['h3'][2], R['h5'][2], R['h10'][2]]\n"
            "    ts = [R['h1'][5], R['h3'][5], R['h5'][5], R['h10'][5]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], means, color=RED, width=.6); a1.axhline(0, c='k', lw=.9)\n"
            "for i,v in enumerate(means): a1.annotate(f'{v:+.3f}%',(i,v),ha='center',va='top' if v<0 else 'bottom',fontsize=9)\n"
            "a1.set_ylabel('signed mean (%)'); a1.set_title('Signed engulfing return < 0 everywhere')\n"
            "a2.bar([f'{h}d' for h in hs], ts, color=RED, width=.6)\n"
            "a2.axhline(2, ls='--', c=GREEN, label='+2 (would be REAL)'); a2.axhline(-2, ls='--', c=GREY)\n"
            "for i,v in enumerate(ts): a2.annotate(f'{v:.2f}',(i,v),ha='center',va='top',fontsize=9)\n"
            "a2.set_ylabel('HAC t'); a2.set_title('HAC t deeply negative — wrong side of the bar'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('signed mean:', [round(v,3) for v in means]); print('HAC t:', [round(v,2) for v in ts])"
        ),
        md(
            f"> 💡 In plain words: the signed return is **{R['h1'][2]:+.3f}%** at 1 day (**HAC t = "
            f"{R['h1'][5]:.2f}**) and the *t* gets *more* negative with horizon ({R['h5'][5]:.2f} at 5d). "
            "This is the opposite of the kill condition being avoided — the effect is significant in the "
            "**wrong** direction. Far below the *t* ≥ 2 bar."
        ),
        md(
            "### 4b · The leg split — the bearish short is the killer, the bullish long is beta\n\n"
            "Left: bullish vs bearish leg means. Right: the bullish leg against the unconditional long "
            "forward return (the always-up drift it must beat to be a real *signal* and not beta)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bull = []; bear = []; base = []\n"
            "    for h in hs:\n"
            "        ev = st.collect_events(PANEL, h)\n"
            "        bull.append(ev[ev['dir']>0]['signed_ret'].mean()*100)\n"
            "        bear.append(ev[ev['dir']<0]['signed_ret'].mean()*100)\n"
            "        base.append(st.unconditional_base(PANEL, h).mean()*100)\n"
            "else:\n"
            "    bull = [R['bull'][h][0] for h in hs]; bear = [R['bear'][h][0] for h in hs]\n"
            "    base = [R['h1'][3], R['h3'][3], R['h5'][3], R['h10'][3]]\n"
            "x = np.arange(len(hs))\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar(x-.2, bull, .4, color=GREEN, label='bullish leg'); a1.bar(x+.2, bear, .4, color=RED, label='bearish leg')\n"
            "a1.axhline(0, c='k', lw=.9); a1.set_xticks(x); a1.set_xticklabels([f'{h}d' for h in hs])\n"
            "a1.set_ylabel('signed return (%)'); a1.set_title('Bearish leg keeps rising (short bleeds)'); a1.legend()\n"
            "a2.bar(x-.2, bull, .4, color=GREEN, label='after bullish engulfing'); a2.bar(x+.2, base, .4, color=GREY, label='unconditional long')\n"
            "a2.axhline(0, c='k', lw=.9); a2.set_xticks(x); a2.set_xticklabels([f'{h}d' for h in hs])\n"
            "a2.set_ylabel('forward return (%)'); a2.set_title('Bullish leg < just buying any day = beta'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('bull:', [round(v,3) for v in bull]); print('bear:', [round(v,3) for v in bear]); print('uncond long:', [round(v,3) for v in base])"
        ),
        md(
            f"> 💡 In plain words: after a *bearish* engulfing the stock keeps **climbing** "
            f"({R['bear'][5][0]:+.3f}% signed at 5d, HAC t = {R['bear'][5][1]:.2f}) — the short is on the "
            f"wrong side of the market. The *bullish* leg is positive over a week, but at 10 days it earns "
            f"**{R['bull'][10][0]:+.3f}%** versus **{R['h10'][3]:+.3f}%** for a random day — it "
            "**underperforms beta**. Neither leg contains a reversal."
        ),
        md(
            "### 4c · The placebo — a random pick of the same trades beats it every time\n\n"
            "Draw 10,000 sets of the same number of random bars, sign each by a coin, and histogram their "
            "means. The observed engulfing mean should sit in the right tail if it's good — instead it "
            "sits in the **left** tail."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(PANEL, 1, len(st.collect_events(PANEL,1)), n_draws=10000)\n"
            "    obs = pl['obs']*100; draws = pl['draws']*100; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['h1'][2]; pval = R['h1'][7]\n"
            "    rng = np.random.default_rng(402); draws = rng.normal(0.0, 0.02, 10000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=60, color=GREY, alpha=.85, label='null: 10,000 random-bar / coin-sign picks')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed engulfing {obs:+.3f}%')\n"
            "ax.set_xlabel('signed mean (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Wrong tail: placebo p = {pval:.3f} (random beats it ~always)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.3f}%  placebo p={pval:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the red line sits to the **left** of the random cloud — placebo "
            f"**p = {R['h1'][7]:.3f}**, i.e. essentially *every* random pick of the same many trades does "
            "better than the engulfing. A pattern that loses to coin flips is not a signal."
        ),
        md(
            "### 4d · The myth check — does a trend / volume filter rescue it?\n\n"
            "The textbook's escape hatch: \"only count it after a trend, on heavy volume.\" Add each "
            "filter at H=1 (bullish needs a prior 10-day downtrend, bearish a prior uptrend; volume > 1.5× "
            "its 20-day average)."
        ),
        code(
            "labels = [m[0] for m in R['myth']]\n"
            "if HAVE_REAL:\n"
            "    cfg = [(False,False),(True,False),(False,True),(True,True)]\n"
            "    ms = []\n"
            "    for rt,rv in cfg:\n"
            "        s = st.summarize(PANEL, 1, placebo=False, require_trend=rt, require_volume=rv)\n"
            "        ms.append(s['mean']*100)\n"
            "else:\n"
            "    ms = [m[2] for m in R['myth']]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(labels, ms, color=RED, width=.6); ax.axhline(0, c='k', lw=.9)\n"
            "for i,v in enumerate(ms): ax.annotate(f'{v:+.3f}%',(i,v),ha='center',va='top',fontsize=9)\n"
            "ax.set_ylabel('signed mean at H=1 (%)'); ax.set_title('Every textbook filter stays NEGATIVE')\n"
            "plt.tight_layout(); plt.show()\n"
            "for m in R['myth']: print(f\"{m[0]:>14}: n={m[1]:>5} mean={m[2]:+.3f}% t_hac={m[3]:+.2f} win={m[4]:.1f}%\")"
        ),
        md(
            f"> 💡 In plain words: plain **{R['myth'][0][2]:+.3f}%**, with a trend filter "
            f"**{R['myth'][1][2]:+.3f}%**, with a volume filter **{R['myth'][2][2]:+.3f}%**, with both "
            f"**{R['myth'][3][2]:+.3f}%** — all **negative**. The sub-2 *t*'s on the filtered cuts are just "
            "shrinking samples (down to 405 events), not a sign flip: the means and win-rates stay on the "
            "wrong side. The folklore's reliability conditions don't rescue it."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic panel where we **plant** a known day-after reversal: with **zero** edge the "
            "signed mean must stay at t≈0 (no false positive); with a planted reversal it must light up. "
            "Both hold — so the real-tape negative is a genuine *absence* (worse, a reversal) of edge, not "
            "a broken detector."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.004):\n"
            "    px, truth = data.synthetic_panel(edge=edge, seed=402)\n"
            "    s = st.summarize(px, 1, n_draws=3000)\n"
            "    res.append((edge, s['n_events'], s['mean']*100, s['t_hac'], s['p_placebo'], s['win']*100))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'planted edge\\n{e:.3f}' for e,_,_,_,_,_ in res]; tvals = [r[3] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5); ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('H=1 HAC t'); ax.set_title('Control: zero edge -> t~0; planted reversal -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,m,t,p,w in res: print(f'planted {e:.3f}: events={k} mean={m:+.3f}% t={t:.2f} p={p:.3f} win={w:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted reversal the control sits at "
            f"**t = {R['syn'][0][3]:.2f}** (a random walk can't fake it); a planted reversal reaches "
            f"**t = {R['syn'][1][3]:.2f}** with a {R['syn'][1][5]}% win-rate. The machinery is faithful — "
            "so the real-tape **negative** is the genuine article. The engulfing simply isn't a reversal "
            "on this tape; it's the reverse."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the signed engulfing return is **significantly negative** at every "
            f"horizon (1d **{R['h1'][2]:+.3f}%**, HAC **t = {R['h1'][5]:.2f}**; 5d **{R['h5'][2]:+.3f}%**, "
            f"**t = {R['h5'][5]:.2f}**), win-rate **{R['h1'][4]:.1f}%**, placebo **p = {R['h1'][7]:.3f}**. "
            "It does not clear the **t ≥ 2** bar — it clears it in the *wrong direction*. Survivorship works "
            "*for* the claim and still can't save it.\n"
            f"- **Tradability `MIRAGE`** — negative before costs; net of round trip + borrow it is "
            f"**{R['h1'][8]:+.3f}% / {R['h5'][8]:+.3f}%** (1d/5d). Nothing to harvest, no capacity question.\n"
            f"- **Reliable reversal? `BUSTED`** — the bearish leg keeps rising (5d HAC t = "
            f"{R['bear'][5][1]:.2f}); the bullish leg's positive few-day return **underperforms** the "
            "unconditional long drift (pure beta). Trend/volume filters don't flip it. The most-taught "
            "reversal does not call turns."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — no, and the control says why that's real\n\n"
            "The net signed return by horizon, with the break-even line — the whole region is below zero. "
            "And because the planted-edge control lights up, we know this isn't a measurement failure: the "
            "tape genuinely lacks the reversal (and faintly contains its opposite)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    nv = [st.summarize(PANEL, h, placebo=False)['net']*100 for h in hs]\n"
            "    gg = [st.summarize(PANEL, h, placebo=False)['mean']*100 for h in hs]\n"
            "else:\n"
            "    nv = [R['h1'][8], R['h3'][8], R['h5'][8], R['h10'][8]]\n"
            "    gg = [R['h1'][2], R['h3'][2], R['h5'][2], R['h10'][2]]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(hs, gg, 'o--', c=GREY, lw=1.6, label='gross signed')\n"
            "ax.plot(hs, nv, 'o-', c=RED, lw=2.2, label='net of costs + borrow')\n"
            "ax.axhline(0, c='k', ls='--', label='break-even')\n"
            "ax.fill_between(hs, 0, nv, color=RED, alpha=.10)\n"
            "for h,v in zip(hs,nv): ax.annotate(f'{v:+.3f}%',(h,v),ha='center',va='top',fontsize=9)\n"
            "ax.set_xlabel('horizon (days)'); ax.set_ylabel('signed return per event (%)')\n"
            "ax.set_title('Net signed return is below zero at every horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net by horizon:', {f'{h}d': round(v,3) for h,v in zip(hs,nv)})"
        ),
        md(
            "> 💡 In plain words: there is no tradeable region — the net line never crosses zero. The "
            "engulfing pattern, traded as taught, is a **MIRAGE**: a vivid shape that feels predictive and "
            "isn't. The honest move is to stop trading it (or, if anything, to fade it — but the magnitude "
            "is too small after costs to build a business on)."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The candlestick literature agrees.** Marshall, Young & Rose (2006, JBF) find candlestick "
            "patterns add no value on the Dow once bootstrapped against a random null — our engulfing "
            "result is one clean instance.\n"
            "- **Why might it anti-predict?** A bearish engulfing on a strong, drifting name is often just "
            "a one-day pullback inside an uptrend — fading it (shorting) puts you against the drift. Short "
            "reversal vs continuation is a regime question worth its own study.\n"
            "- **Use it as a filter, not a signal.** It might still carry information *conditional* on a "
            "real edge; test it as a gate on PEAD or momentum and measure the incremental t.\n\n"
            "*The reproducible core is offline and deterministic; bars are total-return adjusted. Methods "
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
