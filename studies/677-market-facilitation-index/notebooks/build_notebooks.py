"""Generate the two narrative notebooks for Study 677 (Market Facilitation Index).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY +
basket tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SPY + 5-ETF
# basket, RAW OHLCV 1993-02-01 -> 2026-06-30; as-of 2026-06-30).
R = dict(
    start="1993-02-01", end="2026-06-30",
    tickers=["SPY", "QQQ", "DIA", "IWM", "XLE", "GLD"],
    fp_spy="77c7e57287b6", fp_qqq="f8a90001e6b4", fp_dia="2b099f19b356",
    fp_iwm="c02ce892c294", fp_xle="fca8869d5f64", fp_gld="e20ec7413593",
    n_spy=8410, n_classified=8409,
    # SPY headline: state -> (n, fwd_bps, rest_fwd_bps, t_fwd, cont_bps, rest_cont_bps, t_cont, hit_pct)
    spy=dict(
        green=(1954, 6.64, 4.19, +0.84, -4.29, -1.80, -0.85, 46.5),
        fade=(2002, 3.68, 5.10, -0.48, 2.35, -3.86, +2.08, 49.6),
        fake=(2256, 6.76, 4.02, +0.94, -6.07, -1.02, -1.74, 49.1),
        squat=(2196, 2.01, 5.73, -1.26, -1.19, -2.80, +0.54, 48.6),
    ),
    placebo_green_gap=-2.48, placebo_green_p=0.4008,
    placebo_squat_gap=+1.60, placebo_squat_p=0.5842,
    # pooled: state -> (n, fwd_bps, t_fwd, cont_bps, t_cont, hit_pct)
    pooled=dict(
        green=(8864, 5.56, +0.63, -3.03, -0.33, 48.5),
        fade=(9218, 5.43, +0.54, -1.16, +1.10, 48.8),
        fake=(11894, 6.83, +1.89, -4.88, -2.07, 49.2),
        squat=(11364, 1.32, -2.89, -1.00, +1.35, 48.3),
    ),
    # per-ticker: ticker -> (green_t, green_n, squat_t, squat_n)
    per_ticker=dict(
        SPY=(-0.85, 1954, +0.54, 2196), QQQ=(-0.37, 1699, +1.31, 1680),
        DIA=(-0.02, 1432, +0.92, 2057), IWM=(+0.21, 1396, -0.13, 1800),
        XLE=(+0.02, 1234, +0.58, 2168), GLD=(+0.39, 1149, -0.23, 1463),
    ),
    # timer: rule -> (sharpe_strat, sharpe_bench, diff_bps, hac_t, flips_yr)
    timer=dict(
        green=(0.33, 0.65, -3.630, -3.55, 52.0),
        squat_avoid=(0.61, 0.65, -0.966, -1.53, 55.5),
        sma=(0.75, 0.65, -0.719, -0.97, 0.5),
    ),
    cost_sweep=[(0.0, 0.45, -3.14), (1.0, 0.33, -3.55), (2.0, 0.21, -3.95), (5.0, -0.15, -5.17)],
    perm_gap=-0.314, perm_p=0.211,
    syn_null_green_mean=+0.35, syn_null_green_sd=0.87, syn_null_green_fire=1,
    syn_null_squat_mean=-0.17, syn_null_squat_sd=0.76, syn_null_squat_fire=0,
    syn_planted_green_bps=+70.42, syn_planted_green_t=+17.23,
    syn_planted_squat_bps=-63.65, syn_planted_squat_t=-22.14,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Green/Squat?: Busted](https://img.shields.io/badge/Green%2FSquat%3F-Busted-8b949e?style=flat-square)\n\n"
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

from market_facilitation_index import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    BASKET = data.load_basket()
    SPY_BARS = BASKET["SPY"]
    DF = st.day_frame(SPY_BARS)
else:
    BASKET = SPY_BARS = DF = None
print("real cache present:", HAVE_REAL, "| tickers:", data.TICKERS,
      "| SPY bars:", (0 if DF is None else len(DF)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"

STATE_ORDER = ["green", "fade", "fake", "squat"]
STATE_LABEL = {"green": "Green", "fade": "Fade", "fake": "Fake", "squat": "Squat"}


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do the four colors of a trading bar tell you what happens next? 🚦\n"
            "### Bill Williams' Market Facilitation Index — a 30-year-old retail rule "
            "that never got checked\n\n"
            + BADGES +
            "Bill Williams, the trading author behind the \"Alligator\" and \"chaos "
            "theory\" school of technical analysis, built a simple ratio: how far price "
            "moved (High minus Low) divided by how much volume it took to move it. Cross "
            "that ratio's day-to-day direction against volume's own day-to-day direction "
            "and you get four \"colors\": **Green** — price and volume agree, the market "
            "is \"in gear,\" the move should *continue*. **Squat** — lots of volume, no "
            "progress, the market is \"coiling\" before a *reversal*.\n\n"
            "It's taught in trading courses to this day. Nobody seems to have actually "
            "checked it against 33 years of data. We did.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost "
            "math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** SPY headline + a 5-ETF basket (QQQ, DIA, IWM, XLE, "
            "GLD), raw daily bars, 1993→2026. Every chart is drawn by the code beside "
            "it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does a Green bar predict the next day continues the trend? | **No.** "
            f"The \"continuation score\" on Green days is actually **negative** "
            f"({R['spy']['green'][4]:+.1f} bps) — the opposite of what the rule "
            "claims — and it's nowhere close to statistically real. |\n"
            "| Does a Squat bar predict a reversal? | **No.** Barely different from a "
            f"normal day ({R['spy']['squat'][4]:+.1f} bps), and again not real "
            "statistically. |\n"
            "| Does the sign at least agree across different markets? | **No.** Green is "
            "negative on SPY, QQQ and DIA but positive on IWM, XLE and GLD — a coin flip, "
            "not a pattern. |\n"
            "| Can you trade the \"ride the Green days\" idea? | **You'd lose money "
            "doing it** — significantly worse than just holding the index, even before "
            "paying a single cent of trading costs. |\n\n"
            "> Four colors, thirty years of data, zero signal."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When price range and volume both expand together, the market is "
            "'in gear' — ride it. When volume explodes but price goes nowhere, the "
            "market is 'squatting,' loading up for a violent move — the trend is about "
            "to snap.\"*\n\n"
            "It has the shape of a good idea: volume is supposed to confirm price "
            "moves, and BW-MFI turns that intuition into a simple, mechanical rule "
            "anyone can read off a chart with no math beyond \"is this bar bigger or "
            "smaller than the last one, on more or less volume.\""
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it worked, this would be one of the simplest tradeable edges on the "
            "desk: no model, no optimization, just four labels read straight off daily "
            "bars. It's taught as gospel in retail technical-analysis courses. If it's "
            "noise dressed up as a pattern, that's worth knowing too — and it's a "
            "template for how to check *any* categorical chart-pattern claim honestly."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **Label every day.** Compute BW-MFI = (High−Low)/Volume for SPY and five "
            "other ETFs, compare it and volume to the day before, and sort every day "
            "into Green / Fade / Fake / Squat.\n"
            "- **Score the next day.** For each color, does tomorrow's return tend to "
            "*continue* today's direction (a positive \"continuation score\") or "
            "*reverse* it (negative)?\n"
            "- **The luck check.** Reshuffle which days get which color label 2,000 "
            "times — how often does a random subset of days score this well by chance?\n"
            "- **The replication check.** Does the sign of the effect agree across six "
            "different, independently-traded markets?\n"
            "- **The trade check.** Actually trade it — long only on Green days — and "
            "see if it beats simply holding the index, after costs."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Average continuation score by bar color on SPY — "
            "positive means \"tomorrow agreed with today,\" negative means \"tomorrow "
            "reversed it.\""
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.state_stats(DF)\n"
            "    conts = [s[st_]['cont_bps'] for st_ in " + repr(STATE_ORDER) + "]\n"
            "else:\n"
            "    conts = [R['spy'][st_][4] for st_ in " + repr(STATE_ORDER) + "]\n"
            "labels = " + repr([STATE_LABEL[s] for s in STATE_ORDER]) + "\n"
            "cols = [RED if l in ('Green','Squat') else GREY for l in labels]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(labels, conts, color=cols, width=.6)\n"
            "for i, v in enumerate(conts):\n"
            "    ax.annotate(f'{v:+.1f} bps', (i, v), ha='center', va='top' if v < 0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('continuation score (bps): + = continues, - = reverses')\n"
            "ax.set_title('Green should be positive, Squat should be negative. Neither is.')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({l: round(c, 2) for l, c in zip(labels, conts)})"
        ),
        md(
            f"Green should be **positive** (continuation) — it's **{R['spy']['green'][4]:+.1f}"
            " bps**, negative, backwards. Squat should be clearly **negative** "
            f"(reversal) — it's **{R['spy']['squat'][4]:+.1f} bps**, barely different "
            "from zero. Neither bar is even close to the *t* = 2 bar the desk requires "
            "to call something real, and a 2,000-shuffle placebo confirms both are "
            f"indistinguishable from randomly relabeling the days "
            f"(*p* = {R['placebo_green_p']:.2f} for Green, {R['placebo_squat_p']:.2f} "
            "for Squat).\n\n"
            "**Does it at least hold up on other markets?**"
        ),
        code(
            "tickers = " + repr(list(R["per_ticker"].keys())) + "\n"
            "if HAVE_REAL:\n"
            "    gts, sts = [], []\n"
            "    for t in tickers:\n"
            "        d = st.day_frame(BASKET[t]); ss = st.state_stats(d)\n"
            "        gts.append(ss['green']['welch_t_cont']); sts.append(ss['squat']['welch_t_cont'])\n"
            "else:\n"
            "    gts = [R['per_ticker'][t][0] for t in tickers]\n"
            "    sts = [R['per_ticker'][t][2] for t in tickers]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "x = np.arange(len(tickers)); w = .35\n"
            "ax.bar(x - w/2, gts, width=w, color=AMBER, label='Green (should be > 0)')\n"
            "ax.bar(x + w/2, sts, width=w, color=GREY, label='Squat (should be < 0)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels(tickers)\n"
            "ax.set_ylabel('continuation-score Welch t')\n"
            "ax.set_title('Six markets, no consistent sign, nobody clears the dashed t=2 line')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('green t:', dict(zip(tickers, [round(v,2) for v in gts])))\n"
            "print('squat t:', dict(zip(tickers, [round(v,2) for v in sts])))"
        ),
        md(
            "The sign of \"Green\" flips between negative (SPY, QQQ, DIA) and positive "
            "(IWM, XLE, GLD) depending on which ETF you look at — and \"Squat\" does the "
            "same thing in reverse. That's the signature of noise, not a six-way "
            "replication of a real pattern.\n\n"
            "**Finally, the trade.** What if you actually built a strategy — go long "
            "only on Green days, the \"in gear, ride it\" reading?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = st.run_timer(DF, cost_bps=1.0)\n"
            "    gs, bs = r['green']['sharpe_strat'], r['green']['sharpe_bench']\n"
            "    ss_, bs2 = r['sma']['sharpe_strat'], r['sma']['sharpe_bench']\n"
            "else:\n"
            "    gs, bs = R['timer']['green'][0], R['timer']['green'][1]\n"
            "    ss_, bs2 = R['timer']['sma'][0], R['timer']['sma'][1]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['Green filter\\n(BW-MFI)', 'buy & hold', 'SMA(50/200)\\n(no volume at all)'],\n"
            "       [gs, bs, ss_], color=[RED, GREY, AMBER], width=.55)\n"
            "for i, v in enumerate([gs, bs, ss_]):\n"
            "    ax.annotate(f'{v:.2f}', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('annualised net Sharpe (SPY, 1 bp cost)')\n"
            "ax.set_title('Riding the \"in gear\" days loses to doing nothing special at all')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Green filter Sharpe {gs:.2f} vs buy-hold {bs:.2f} vs SMA {ss_:.2f}')"
        ),
        md(
            f"The Green-only filter posts a net Sharpe of **{R['timer']['green'][0]:.2f}** "
            f"against buy-and-hold's **{R['timer']['green'][1]:.2f}** — and the gap is "
            "statistically real (just not in the direction anyone selling the rule would "
            "want): it **loses significantly**, even *before* paying a single cent in "
            "trading costs. A plain two-line moving-average crossover, using none of "
            f"BW-MFI's volume machinery, beats both at **{R['timer']['sma'][0]:.2f}**."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Neither Green's continuation claim nor Squat's "
            "reversal claim clears the statistical bar, both point the wrong way on "
            "the headline tape, and the sign isn't even stable across six different "
            "markets.\n"
            "- **Tradability — Mirage.** The one tradeable version of the idea actively "
            "loses to simply holding the index — significantly, and before costs.\n"
            "- **\"Green flags continuation, Squat flags reversal\"? — Busted.** Not "
            "weak, not mixed — busted, on every test we ran."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Why might a plausible-sounding rule fail this completely?** BW-MFI's "
            "own four states are already unbalanced by construction — because MFI is a "
            "*ratio* of range to volume, a volume spike mechanically tends to *lower* "
            "MFI even when nothing informative is happening, which is exactly why Fake "
            "and Squat are the two most common colors in the data. The rule may be "
            "confusing a mechanical artefact of the ratio for a market signal.\n"
            "- **A cleaner volume-confirmation test** would separate the ratio's "
            "mechanical bias from any genuine price-volume information — the natural "
            "sequel to this study.\n"
            "- **Sibling studies:** the closest relative, "
            "[Ease of Movement](../../424-ease-of-movement/), tests the same "
            "range-over-volume ratio as a continuous signal (and finds a real but "
            "unremarkable trend effect, no better than a moving average) — worth "
            "reading next.\n\n"
            "*Think the classifier itself is wrong, not the claim? The code is right "
            "here — fork it, change the state rule, show a net, certifiable edge after "
            "costs, then we'll talk.*"
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
            "# Market Facilitation Index — a quantitative teardown 🔬\n"
            "### The 4-state classifier · continuation-score Welch/placebo splits · "
            "pooled & per-ticker replication · a NET timer race with HAC *t* · a "
            "20-seed synthetic positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "Bill Williams' BW-MFI names four bar \"colors\" from the cross of two "
            "bar-to-bar sign changes — MFI's own direction, volume's own direction — and "
            "claims Green predicts continuation, Squat predicts reversal. Neither claim "
            "has a peer-reviewed anchor; this is a direct, first-principles test.\n\n"
            "> ⚠️ **Data note.** SPY + 5-ETF basket (QQQ, DIA, IWM, XLE, GLD), RAW daily "
            "OHLCV for the BW-MFI ratio, adjusted close for every return, "
            "1993-02-01 → 2026-06-30. No survivorship — every tape is a single "
            "continuously-traded ETF. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp_spy"] +
            "` for SPY).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Green cont-score Welch **t = {R['spy']['green'][6]:.2f}** "
            f"(SPY), **t = {R['pooled']['green'][4]:.2f}** (pooled 6 tickers, n="
            f"{R['pooled']['green'][0]:,}); Squat **t = {R['spy']['squat'][6]:.2f}** (SPY), "
            f"**t = {R['pooled']['squat'][4]:.2f}** (pooled); placebo *p* = "
            f"{R['placebo_green_p']:.2f} / {R['placebo_squat_p']:.2f} |\n"
            f"| **Tradability** | `MIRAGE` | Green-filter timer HAC **t = "
            f"{R['timer']['green'][3]:.2f}** vs buy-hold, net Sharpe "
            f"{R['timer']['green'][0]:.2f} vs {R['timer']['green'][1]:.2f}, still "
            f"significant at 0 bp cost |\n"
            "| **Green/Squat claim?** | `BUSTED` | wrong sign on the headline tape, no "
            "consistent sign across 6 tickers, placebo confirms noise |\n\n"
            "> 💡 In plain words: a plausible-sounding volume-confirmation rule that "
            "simply isn't in the data, on either side of its own claim."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\text{MFI}_t = (H_t - L_t)/V_t$. Define\n"
            "$\\text{state}(t) \\in \\{\\text{green, fade, fake, squat}\\}$ from "
            "$\\text{sign}(\\Delta\\text{MFI}_t) \\times \\text{sign}(\\Delta V_t)$: "
            "green = (+,+), fade = (−,−), fake = (+,−), squat = (−,+). Let "
            "$r_t$ be day $t$'s return and $\\text{fwd}_t = r_{t+1}$. The **continuation "
            "score** is $c_t = \\text{sign}(r_t) \\cdot \\text{fwd}_t$.\n\n"
            "- **H₁ (Green continuation).** $E[c_t \\mid \\text{state}(t)=\\text{green}] > "
            "0$, and above the unconditional mean.\n"
            "- **H₂ (Squat reversal).** $E[c_t \\mid \\text{state}(t)=\\text{squat}] < 0$, "
            "and below the unconditional mean.\n"
            "- **H₃ (replication).** The sign of H₁/H₂ should hold across independent "
            "tickers — a categorical claim this simple has no excuse to be "
            "instrument-specific.\n"
            "- **H₄ (capture).** A Green-only long/flat timer, net of costs, should beat "
            "buy-and-hold.\n\n"
            "We find **H₁ rejected** (wrong sign, *t* = −0.85 SPY / −0.33 pooled), "
            "**H₂ rejected** (wrong sign, *t* = +0.54 SPY / +1.35 pooled), **H₃ "
            "rejected** (sign flips 3-of-6 tickers for both states) and **H₄ rejected** "
            "in the strongest possible way — the timer loses *significantly*."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The continuation score at consecutive days overlaps by one observation "
            "(day *t*'s fwd return is day *t+1*'s own return), so within-state serial "
            "dependence is mild but present; the **Welch t** on the group split is the "
            "planned primary, and every conditional mean carries a **2,000-draw "
            "label-shuffle placebo** — reassign the state label to a random same-size "
            "subset of days, holding every continuation score fixed, so the null "
            "distribution reflects the study's own sample, not a textbook formula. The "
            "timer's daily return difference against buy-and-hold gets a **Newey-West "
            "HAC** *t* (returns are autocorrelated) plus its own **sign-permutation** "
            "placebo. Pooling across six tickers and reporting the per-ticker split "
            "separately guards against a categorical claim laundering a single lucky "
            "instrument into six-fold \"confirmation.\""
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** SPY + 5-ETF basket, RAW OHLCV {R['start']} → {R['end']} "
            "(as-of, last complete month). No survivorship (single continuously-traded "
            "ETFs).\n"
            "- **Classifier.** `state(t)` from bar-to-bar MFI/Volume sign crosses — "
            "known at close *t*, needs only that bar and the one before.\n"
            "- **Headline.** Welch *t* (forward return + continuation score) per state "
            "vs all other days, plus the label-shuffle placebo.\n"
            "- **Replication.** Pooled 6-ticker version, then a per-ticker sign check.\n"
            "- **Execution.** One lag: color known at close *t* earns the return of "
            "*t+1* (`fwd_ret`) or the timer position applied with one `shift`.\n"
            "- **Timer.** Green filter + Squat avoidance + an SMA(50/200) benchmark, "
            "NET of one-way costs × NAV per flip, HAC *t* vs buy-and-hold, cost sweep, "
            "sign-permutation placebo.\n"
            "- **Control.** Synthetic tape, range/volume independent of the return path, "
            "TUNABLE planted Green-continuation/Squat-reversal knob; the null must not "
            "fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline split and its placebo (SPY)\n\n"
            "Forward return and continuation score by state, Welch *t* vs all other "
            "days."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.state_stats(DF)\n"
            "else:\n"
            "    s = {k: dict(zip(['n','fwd_bps','rest_fwd_bps','welch_t_fwd','cont_bps',\n"
            "                      'rest_cont_bps','welch_t_cont','cont_hit'],\n"
            "                     list(v[:7]) + [v[7]/100]))\n"
            "         for k, v in R['spy'].items()}\n"
            "for k in " + repr(STATE_ORDER) + ":\n"
            "    v = s[k]\n"
            "    print(f\"{k:6s} n={v['n']:5d}  fwd={v['fwd_bps']:+6.2f}bps t={v['welch_t_fwd']:+5.2f}"
            "   cont={v['cont_bps']:+6.2f}bps t={v['welch_t_cont']:+5.2f}\")\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "labels = " + repr([STATE_LABEL[x] for x in STATE_ORDER]) + "\n"
            "conts = [s[x]['cont_bps'] for x in " + repr(STATE_ORDER) + "]\n"
            "ts = [s[x]['welch_t_cont'] for x in " + repr(STATE_ORDER) + "]\n"
            "cols = [RED if abs(t) >= 2 else GREY for t in ts]\n"
            "ax.bar(labels, conts, color=cols, width=.6)\n"
            "for i, (v, t) in enumerate(zip(conts, ts)):\n"
            "    ax.annotate(f'{v:+.1f}bps\\n(t={t:+.2f})', (i, v), ha='center',\n"
            "                va='top' if v < 0 else 'bottom', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('continuation score (bps)')\n"
            "ax.set_title('No state clears the |t|>=2 bar (all grey = not significant)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: Green's continuation score is **{R['spy']['green'][4]:+.2f} "
            f"bps** at *t* = **{R['spy']['green'][6]:+.2f}** — negative, the wrong sign, and "
            f"nowhere near significant. Squat's is **{R['spy']['squat'][4]:+.2f} bps** at "
            f"*t* = **{R['spy']['squat'][6]:+.2f}** — also not significant. The label-shuffle "
            f"placebo (2,000 draws) puts Green's gap at *p* = **{R['placebo_green_p']:.4f}** "
            f"and Squat's at *p* = **{R['placebo_squat_p']:.4f}** — both squarely inside a "
            "random-relabeling null."
        ),
        md(
            "### 4b · Pooled across the basket and the per-ticker replication check\n\n"
            "Six tickers pooled (~50k state-days), then split back out to see whether "
            "the *sign* is even stable."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pooled = st.pooled_frame(BASKET)\n"
            "    sp = st.state_stats(pooled)\n"
            "    tickers = list(BASKET.keys())\n"
            "    gts = [st.state_stats(st.day_frame(BASKET[t]))['green']['welch_t_cont'] for t in tickers]\n"
            "    sts = [st.state_stats(st.day_frame(BASKET[t]))['squat']['welch_t_cont'] for t in tickers]\n"
            "else:\n"
            "    sp = {k: dict(zip(['n','fwd_bps','welch_t_fwd','cont_bps','welch_t_cont','cont_hit'], v))\n"
            "          for k, v in R['pooled'].items()}\n"
            "    tickers = list(R['per_ticker'].keys())\n"
            "    gts = [R['per_ticker'][t][0] for t in tickers]\n"
            "    sts = [R['per_ticker'][t][2] for t in tickers]\n"
            "print(f\"pooled green: n={sp['green']['n']:,} cont={sp['green']['cont_bps']:+.2f}bps \"\n"
            "      f\"t={sp['green']['welch_t_cont']:+.2f}\")\n"
            "print(f\"pooled squat: n={sp['squat']['n']:,} cont={sp['squat']['cont_bps']:+.2f}bps \"\n"
            "      f\"t={sp['squat']['welch_t_cont']:+.2f}\")\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "x = np.arange(len(tickers)); w = .35\n"
            "ax.bar(x - w/2, gts, width=w, color=AMBER, label='Green t (claim: > 0)')\n"
            "ax.bar(x + w/2, sts, width=w, color=GREY, label='Squat t (claim: < 0)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels(tickers)\n"
            "ax.set_ylabel('continuation-score Welch t')\n"
            "ax.set_title('Sign instability across six independent tickers')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: pooling to ~{R['pooled']['green'][0]:,} Green state-days "
            f"barely moves the needle (*t* = {R['pooled']['green'][4]:+.2f}), and Squat "
            f"pooled sits at *t* = {R['pooled']['squat'][4]:+.2f} — still under the bar. The "
            "per-ticker split is the real tell: Green is negative on SPY/QQQ/DIA and "
            "positive on IWM/XLE/GLD; Squat flips the same way. A genuine categorical "
            "effect this simple has no principled reason to reverse sign by instrument — "
            "this is what six independent noise draws look like."
        ),
        md(
            "### 4c · The timer — trading the \"continuation\" half, net of costs\n\n"
            "Green filter (long only on Green days) vs Squat avoidance vs the "
            "volume-free SMA(50/200) benchmark, all raced against buy-and-hold with one "
            "execution lag and one-way costs × NAV per flip."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = st.run_timer(DF, cost_bps=1.0)\n"
            "    names = ['green', 'squat_avoid', 'sma']\n"
            "    ss_ = [r[n]['sharpe_strat'] for n in names]\n"
            "    bh = r['green']['sharpe_bench']\n"
            "    ts = [r[n]['hac_t_diff'] for n in names]\n"
            "else:\n"
            "    names = ['green', 'squat_avoid', 'sma']\n"
            "    ss_ = [R['timer'][n][0] for n in names]\n"
            "    bh = R['timer']['green'][1]\n"
            "    ts = [R['timer'][n][3] for n in names]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.4))\n"
            "labels = ['Green filter', 'Squat avoid', 'SMA(50/200)']\n"
            "a1.bar(labels + ['buy & hold'], ss_ + [bh], color=[RED, GREY, AMBER, GREEN], width=.6)\n"
            "for i, v in enumerate(ss_ + [bh]): a1.annotate(f'{v:.2f}', (i, v), ha='center', va='bottom')\n"
            "a1.set_ylabel('net annualised Sharpe'); a1.set_title('Net Sharpe vs buy-and-hold')\n"
            "a2.bar(labels, ts, color=[RED if abs(t) >= 2 else GREY for t in ts], width=.55)\n"
            "for i, t in enumerate(ts): a2.annotate(f'{t:+.2f}', (i, t), ha='center', va='top' if t < 0 else 'bottom')\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel('HAC t (strategy - buy&hold)'); a2.set_title('Is the gap statistically real?')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the Green filter's net Sharpe (**{R['timer']['green'][0]:.2f}**) "
            f"is not just below buy-and-hold's (**{R['timer']['green'][1]:.2f}**) — the gap is "
            f"**statistically significant** (HAC *t* = **{R['timer']['green'][3]:.2f}**), and "
            "the cost sweep below shows it isn't a costs story. Squat avoidance "
            f"(*t* = {R['timer']['squat_avoid'][3]:.2f}) is a wash. The SMA(50/200) crossover "
            f"— zero volume machinery — posts the best Sharpe of the three "
            f"(**{R['timer']['sma'][0]:.2f}**)."
        ),
        md(
            "### 4d · Cost sweep and the sign-permutation placebo\n\n"
            "Is the Green filter's underperformance a costs story, or structural?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sweep = [(cb, st.run_timer(DF, cost_bps=cb)['green']['sharpe_strat'],\n"
            "              st.run_timer(DF, cost_bps=cb)['green']['hac_t_diff'])\n"
            "             for cb in (0.0, 1.0, 2.0, 5.0)]\n"
            "    pv = st.permutation_pvalue_timer(DF, cost_bps=1.0, n_perm=1000)\n"
            "else:\n"
            "    sweep = R['cost_sweep']\n"
            "    pv = {'real_gap': R['perm_gap'], 'p_value': R['perm_p']}\n"
            "cbs = [x[0] for x in sweep]; shs = [x[1] for x in sweep]; hts = [x[2] for x in sweep]\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "ax2 = ax.twinx()\n"
            "ax.plot(cbs, shs, 'o-', color=AMBER, label='net Sharpe')\n"
            "ax2.plot(cbs, hts, 's--', color=RED, label='HAC t vs buy-hold')\n"
            "ax2.axhline(-2, ls=':', c=RED, lw=1)\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('net Sharpe', color=AMBER)\n"
            "ax2.set_ylabel('HAC t', color=RED)\n"
            "ax.set_title('Already significant at ZERO cost — not a costs story')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"real Sharpe gap {pv['real_gap']:+.3f}, sign-permutation p = {pv['p_value']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: even at **0 bp** the Green filter is already "
            f"significantly behind buy-and-hold (*t* = {R['cost_sweep'][0][2]:.2f}) — the "
            "underperformance is structural (missing most of the tape's drift), costs just "
            f"make it worse. The sign-permutation placebo puts the real Sharpe gap's "
            f"unluckiness at *p* = **{R['perm_p']:.2f}** — the timer isn't even an unusually "
            "bad draw of its own exposure schedule."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic tape: range and volume are two independent AR(1)-in-logs "
            "processes (unrelated to the return path), classified into the same four "
            "states; a TUNABLE knob plants a Green-continuation / Squat-reversal "
            "effect. Null (planted=0) checked over **20 seeds**."
        ),
        code(
            "null_g, null_s = [], []\n"
            "for s_ in range(20):\n"
            "    bars, _ = data.synthetic_world(planted=0.0, seed=677 + s_)\n"
            "    sy = st.synthetic_detect(bars)\n"
            "    null_g.append(sy['green']['welch_t_cont']); null_s.append(sy['squat']['welch_t_cont'])\n"
            "null_g = np.asarray(null_g); null_s = np.asarray(null_s)\n"
            "bars_p, _ = data.synthetic_world(planted=0.6, seed=677)\n"
            "syp = st.synthetic_detect(bars_p)\n"
            "planted_g, planted_s = syp['green']['welch_t_cont'], syp['squat']['welch_t_cont']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_g, color=GREY, s=36,\n"
            "           label='Green, null x 20 seeds')\n"
            "ax.scatter(np.ones(20) + np.linspace(-.12, .12, 20), null_s, color=GREY, s=36, marker='s',\n"
            "           label='Squat, null x 20 seeds')\n"
            "ax.scatter([0], [planted_g], color=RED, s=100, zorder=5, label='Green, planted=0.6')\n"
            "ax.scatter([1], [planted_s], color=RED, s=100, zorder=5, marker='s', label='Squat, planted=0.6')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['Green', 'Squat'])\n"
            "ax.set_ylabel('continuation-score Welch t')\n"
            "ax.set_title('Control: null stays inert, a planted effect lights up hard')\n"
            "ax.legend(fontsize=8); plt.tight_layout(); plt.show()\n"
            "print(f'null green: mean t={null_g.mean():+.2f} (sd {null_g.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_g)>=2).sum()}/20')\n"
            "print(f'null squat: mean t={null_s.mean():+.2f} (sd {null_s.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_s)>=2).sum()}/20')\n"
            "print(f'planted=0.6: green t={planted_g:+.2f}  squat t={planted_s:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"*t* = {R['syn_null_green_mean']:+.2f} (Green) / {R['syn_null_squat_mean']:+.2f} "
            f"(Squat) and almost never crosses the bar ({R['syn_null_green_fire']}/20 and "
            f"{R['syn_null_squat_fire']}/20); a modest planted effect reads *t* = "
            f"{R['syn_planted_green_t']:+.2f} / {R['syn_planted_squat_t']:+.2f}. The harness "
            "is unbiased and has the power to find this exact pattern — the real-tape "
            "near-zero *t*'s above are the genuine absence of an effect, not a blind "
            "detector. *(A faithful-engine / power check only — never cited in support of "
            "the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — Green continuation-score *t* = "
            f"**{R['spy']['green'][6]:+.2f}** (SPY), **{R['pooled']['green'][4]:+.2f}** "
            f"(pooled); Squat *t* = **{R['spy']['squat'][6]:+.2f}** (SPY), "
            f"**{R['pooled']['squat'][4]:+.2f}** (pooled). Both under **t = 2**, both on "
            "the wrong side of the claim on the headline tape, placebo *p* = "
            f"{R['placebo_green_p']:.2f} / {R['placebo_squat_p']:.2f}, and the sign "
            "flips ticker-by-ticker with no consistent pattern.\n"
            f"- **Tradability `MIRAGE`** — the Green-filter timer loses to buy-and-hold "
            f"at HAC *t* = **{R['timer']['green'][3]:.2f}** net, still significant at 0 "
            "bp cost; a volume-free SMA(50/200) beats it outright.\n"
            "- **\"Green flags continuation, Squat flags reversal\"? `BUSTED`** — both "
            "predictions fail on the tape, fail the placebo, fail to replicate across "
            "six markets, and the one strategy built on the claim's \"good\" half "
            "actively underperforms doing nothing at all."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **A mechanical confound worth chasing.** BW-MFI is a *ratio* of range to "
            "volume — a volume spike mechanically tends to lower MFI even absent any "
            "informative price action, which is exactly why Fake and Squat dominate the "
            "day-count on every tape here. That skew is baked into the ratio's "
            "definition, not a market fact; separating it from any genuine "
            "price-volume information is the natural sequel.\n"
            "- **Fake's incidental pooled crossing** (*t* = −2.07, not part of the "
            "pre-registered Green/Squat pair) doesn't survive the per-ticker check — "
            "flagged here only so it isn't silently discarded, not promoted to a "
            "finding.\n"
            "- **Dedup map:** [424-ease-of-movement](../../424-ease-of-movement/) (the "
            "closest relative — same range/volume ratio, tested continuously, finds a "
            "real but SMA-equivalent trend effect), "
            "[418-money-flow-index](../../418-money-flow-index/) and "
            "[423-force-index](../../423-force-index/) (different volume-oscillator "
            "constructions), [676-gator-oscillator](../../676-gator-oscillator/) (same "
            "author, a moving-average construct with no volume term).\n\n"
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
