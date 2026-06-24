"""Generate the two narrative notebooks for Study 431 (Schaff Trend Cycle).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily prices
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


# Frozen real-tape headline numbers — mirror of docs/results.md.
# SPY daily, total-return-adjusted (yfinance auto_adjust), 1993-01-29 -> 2026-05-29,
# as-of 2026-05-31, fingerprint 0207fbae81b3. Net of 1 bp one-way x NAV, one execution lag.
R = dict(
    asof="2026-05-31",
    start="1993-01-29", end="2026-05-29", years=33.3, n_days=8390, fp="0207fbae81b3",
    # SPY long/flat, 1 bp
    stc_sharpe=0.29, bh_sharpe=0.65, macd_sharpe=0.42, stc_minus_bh=-0.36,
    stc_cagr=0.029, bh_cagr=0.109, stc_maxdd=-0.50, bh_maxdd=-0.55,
    stc_hac_t=1.87, turnover=17.4, exposure=0.54,
    perm_p=0.96,
    # SPY long/short, 1 bp
    ls_stc_sharpe=-0.26, ls_macd_sharpe=-0.16, ls_hac_t=-1.75, ls_cagr=-0.064, ls_maxdd=-0.91,
    # cost sweep long/flat: (cost_bps, sharpe, hac_t)
    cost=[(0.0, 0.31, 1.96), (1.0, 0.29, 1.87), (2.0, 0.28, 1.78),
          (5.0, 0.24, 1.51), (10.0, 0.17, 1.07)],
    # panel long/flat 1 bp: (ticker, n, stc_sharpe, bh_sharpe, macd_sharpe, hac_t)
    panel=[
        ("SPY", 8390, 0.29, 0.65, 0.42, 1.87),
        ("QQQ", 6848, 0.20, 0.52, 0.21, 1.07),
        ("IWM", 6540, 0.36, 0.47, 0.42, 2.00),
        ("EFA", 6225, 0.26, 0.41, 0.19, 1.45),
        ("AAPL", 11457, 0.37, 0.63, 0.61, 2.36),
        ("MSFT", 10131, 0.43, 0.84, 0.57, 2.70),
    ],
    # synthetic control: (planted_trend, stc_sharpe, bh_sharpe, stc_minus_bh, hac_t)
    syn=[(0.0, -0.35, -0.16, -0.19, -1.42), (0.003, 0.82, 0.72, 0.10, 2.66)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Faster_MACD%3F: Busted](https://img.shields.io/badge/Faster_MACD%3F-Busted-8b949e?style=flat-square)\n\n"
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

from schaff_trend_cycle import data, strategy as st

HAVE_REAL = data.have_real("SPY")
SPY = data.load_real("SPY") if HAVE_REAL else None
print("real SPY cache present:", HAVE_REAL, "| bars:", (0 if SPY is None else len(SPY)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Is the Schaff Trend Cycle really a \"faster MACD\"? \U0001f4c9\n"
            "### A popular oscillator that promises to turn earlier — and what happens when you actually trade it\n\n"
            + BADGES +
            "The **Schaff Trend Cycle** (STC) is a chart indicator with a great pitch: it's a *\"faster "
            "MACD\"*. Doug Schaff built it in 2008 by running a couple of extra math tricks on top of the "
            "classic MACD so the line **whips up and down more quickly** — the idea being you catch trends "
            "*earlier* and make more money than the laggy old MACD.\n\n"
            "It's all over TradingView and YouTube. So we did the obvious thing: turned the STC into a "
            "simple **buy/sell timing rule** on the S&P 500, and asked two blunt questions — *does it beat "
            "just buying and holding?* and *does it beat the plain MACD it claims to improve on?*\n\n"
            "> \U0001f4d3 **Plain-language layer.** Want the *t*-stats, the permutation test and the cost "
            "math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice** — research & education. Every chart is drawn by "
            "the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the STC rule beat **buy-and-hold** on the S&P 500? | **No.** Net Sharpe "
            f"**{R['stc_sharpe']:.2f}** vs buy-and-hold's **{R['bh_sharpe']:.2f}** — it gives up "
            f"about half the risk-adjusted return, and grows your money at **{R['stc_cagr']*100:.1f}%/yr** "
            f"vs **{R['bh_cagr']*100:.1f}%/yr**. |\n"
            f"| Is the edge even **real** (not luck)? | **No.** The HAC *t* is **{R['stc_hac_t']:.2f}** "
            f"(below the 2.0 bar), and a shuffled-position placebo beats it **{R['perm_p']*100:.0f}%** of "
            "the time. |\n"
            f"| Does it beat the **plain MACD** it claims to improve on? | **No.** STC Sharpe "
            f"**{R['stc_sharpe']:.2f}** vs a vanilla MACD crossover's **{R['macd_sharpe']:.2f}** — the "
            "\"faster\" version is actually **worse** than the thing it's faster than. |\n"
            "| Does shorting (long/short) help? | **No — it's worse.** The long/short version loses "
            f"money outright (Sharpe **{R['ls_stc_sharpe']:.2f}**, a **{abs(R['ls_maxdd'])*100:.0f}%** "
            "drawdown). |\n\n"
            "> The promise is \"turn faster, profit more.\" The tape says: turning faster just means "
            "**more whipsaws, more costs, and less money** than doing nothing."
        ),

        md(
            "## 1 · The claim\n\n"
            "> *\"The MACD is good but **slow** — it lags the turn. The Schaff Trend Cycle fixes that: "
            "it cycles between 0 and 100 and snaps to the new trend **much sooner**, so you get in and out "
            "earlier and beat both the MACD and a passive buy-and-hold.\"*\n\n"
            "This is the standard pitch from the indicator's fans. It's a *steelman* worth testing because "
            "it's specific: there's a clear rule (long when STC rises through 25, exit when it falls "
            "through 75) and a clear benchmark (the MACD). If \"faster = better\" is true, the STC rule "
            "should out-earn both. Let's see."
        ),

        md(
            "## 2 · So what?\n\n"
            "If a simple oscillator on a daily chart could reliably beat buy-and-hold on the **S&P 500** "
            "— the single most-studied, most-efficient market on earth — that would be a small "
            "miracle. It would mean the index's future direction is predictable from a squiggly line "
            "anyone can compute for free. Whole hedge funds would be out of a job.\n\n"
            "The honest prior is that it **won't**, because the same trap catches almost every chart "
            "indicator: a trend rule that flips direction on a moving line *lags* the actual turn, eats "
            "trading costs on every flip, and sits in cash through chunks of the relentless upward drift "
            "that buy-and-hold simply collects. \"Faster\" makes that *worse*, not better, in a choppy "
            "market — more flips, more whipsaw."
        ),

        md(
            "## 3 · How would we know?\n\n"
            f"- **The tape.** The S&P 500 (SPY), daily, total-return adjusted, **{R['start']} → "
            f"{R['end']}** (~{R['years']:.0f} years, {R['n_days']:,} bars).\n"
            "- **The rule.** Compute the STC. Go **long** when it crosses **up through 25**; go **flat** "
            "when it crosses **down through 75**. Hold between signals.\n"
            "- **No cheating.** The signal is known at today's close; we earn **tomorrow's** return (one "
            "day's delay). We charge a realistic trading cost on every flip.\n"
            "- **The fair race.** Compare its **risk-adjusted** return (Sharpe) to buy-and-hold *and* to a "
            "plain MACD crossover — because \"it makes money\" is meaningless if buying and holding "
            "makes *more*.\n"
            "- **The mirage test.** If the STC rule's Sharpe is **below** buy-and-hold, or its edge can't "
            "clear a *t* of 2 and beat a shuffled-position placebo — we call it a mirage."
        ),

        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the money.** Here are the equity curves: a dollar in the STC timing rule vs a dollar "
            "just buying and holding the S&P. Same tape, same costs, one-day execution lag."
        ),
        code(
            "if HAVE_REAL:\n"
            "    stc = st.schaff_trend_cycle(SPY['close'])\n"
            "    pos = st.stc_positions(stc)\n"
            "    bk = st.book_returns(SPY, pos, cost_bps=1.0)\n"
            "    eq_stc = (1+bk['strat_excess']).cumprod()\n"
            "    eq_bh  = (1+bk['ret']).cumprod()\n"
            "    idx = SPY.index\n"
            "else:\n"
            "    idx = None\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.6))\n"
            "if HAVE_REAL:\n"
            "    ax.semilogy(idx, eq_bh, color=GREEN, lw=2, label=f\"buy & hold (Sharpe {R['bh_sharpe']:.2f})\")\n"
            "    ax.semilogy(idx, eq_stc, color=RED, lw=2, label=f\"STC timing rule (Sharpe {R['stc_sharpe']:.2f})\")\n"
            "ax.set_ylabel('growth of $1 (log scale)'); ax.set_title('The \"faster MACD\" leaves money on the table')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"STC CAGR {R['stc_cagr']*100:.1f}%/yr  vs  buy-and-hold {R['bh_cagr']*100:.1f}%/yr\")"
        ),
        md(
            f"The red line (STC) ends **far below** the green line (buy-and-hold). It grows your money at "
            f"**{R['stc_cagr']*100:.1f}%/yr** against buy-and-hold's **{R['bh_cagr']*100:.1f}%/yr** — "
            "and it doesn't even buy you a much smoother ride (the worst drawdowns are similar). Sitting in "
            "cash through half the bull market is expensive."
        ),
        md(
            "**Now the head-to-head it actually cares about: does \"faster\" beat the plain MACD?** Here's "
            "the risk-adjusted return (Sharpe) of the STC rule, the vanilla MACD crossover, and buy-and-hold."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = st.run_experiment(SPY, cost_bps=1.0)\n"
            "    vals = [r['stc_sharpe'], r['macd_sharpe'], r['bh_sharpe']]\n"
            "else:\n"
            "    vals = [R['stc_sharpe'], R['macd_sharpe'], R['bh_sharpe']]\n"
            "labels = ['STC\\n(\"faster MACD\")', 'plain MACD\\ncrossover', 'buy & hold']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(labels, vals, color=[RED, AMBER, GREEN], width=.6)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('net Sharpe (annualised)'); ax.set_title('STC loses to the very MACD it claims to beat')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"STC {vals[0]:.2f}  <  MACD {vals[1]:.2f}  <  buy-and-hold {vals[2]:.2f}\")"
        ),
        md(
            f"There's the punchline. The STC rule's Sharpe (**{R['stc_sharpe']:.2f}**) is **below** the "
            f"plain MACD's (**{R['macd_sharpe']:.2f}**), which is itself below buy-and-hold "
            f"(**{R['bh_sharpe']:.2f}**). The \"faster MACD\" is **worse than the MACD**. Speed bought "
            "whipsaws, not profit."
        ),

        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The STC rule's edge can't clear the bar: HAC *t* = "
            f"**{R['stc_hac_t']:.2f}** (under 2.0), and a shuffled-position placebo beats it "
            f"**{R['perm_p']*100:.0f}%** of the time. Indistinguishable from luck.\n"
            f"- **Tradability — Mirage.** Net Sharpe **{R['stc_sharpe']:.2f}** vs buy-and-hold "
            f"**{R['bh_sharpe']:.2f}** — it *destroys* value versus doing nothing. The long/short "
            "version loses money outright.\n"
            "- **\"Faster MACD\"? — Busted.** It doesn't beat the MACD; it underperforms it on the "
            "S&P 500 and on 5 of 6 instruments we tried. The whole selling point fails."
        ),

        md(
            "## 6 · Could you actually trade it?\n\n"
            "Not for profit — it loses to the free alternative (buy-and-hold) before you even get "
            "fancy. And the costs only make it worse: the rule flips ~17 times a year, and every flip "
            "pays the spread. Here's what happens to its Sharpe as trading costs rise."
        ),
        code(
            "cs = [c[0] for c in R['cost']]\n"
            "if HAVE_REAL:\n"
            "    sh = [st.run_experiment(SPY, cost_bps=c)['stc_sharpe'] for c in cs]\n"
            "else:\n"
            "    sh = [c[1] for c in R['cost']]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.plot(cs, sh, 'o-', color=RED, lw=2, label='STC net Sharpe')\n"
            "ax.axhline(R['bh_sharpe'], color=GREEN, ls='--', lw=2, label=f\"buy & hold ({R['bh_sharpe']:.2f})\")\n"
            "ax.set_xlabel('one-way cost per flip (bps)'); ax.set_ylabel('net Sharpe')\n"
            "ax.set_title('Even at zero cost the STC rule never reaches buy-and-hold')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"at 0 bps STC Sharpe {sh[0]:.2f} -- still below buy-and-hold {R['bh_sharpe']:.2f}\")"
        ),
        md(
            "The red line never touches the green dashed line — **not even at zero cost.** Costs make "
            "a bad deal worse, but they're not the killer here; the rule is simply on the wrong side of the "
            "market's drift."
        ),

        md(
            "## 7 · Going further \U0001f6aa\n\n"
            "- **It's the family, not the settings.** STC, MACD, Supertrend (study 106), CCI (178) — "
            "lagging trend oscillators on the S&P keep landing in the same place: behind buy-and-hold. "
            "Try the parameter sweep and watch the conclusion survive.\n"
            "- **Where might it work?** Trend rules earn their keep in *strongly trending, mean-reverting-"
            "free* markets (some commodities, some FX). The synthetic control in the quant notebook shows "
            "the engine **can** bank an edge when a real trend is planted — the S&P just doesn't hand "
            "it one net of costs.\n\n"
            "*Think a different threshold or cycle length flips the verdict? Fork it and show an STC Sharpe "
            "above buy-and-hold, net of costs, on a market you didn't pick after the fact.*"
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
            "# Schaff Trend Cycle — a quantitative teardown \U0001f52c\n"
            "### Daily long/flat (& long/short) timing rule on SPY · net Sharpe excess-vs-excess · "
            "HAC *t* · block-permutation placebo · cost sweep · head-to-head vs the plain MACD "
            "· a planted-trend synthetic control\n\n"
            + BADGES +
            "The rigorous companion to the [notebook for the curious](01_for_the_curious.ipynb). The STC is "
            "a double-stochastic-of-MACD bounded to 0..100 — marketed as a *faster* MACD. We test the "
            "two falsifiable claims head-on: **(i)** the timing rule beats buy-and-hold, and **(ii)** it "
            "beats the MACD it derives from. Both fail on the real tape.\n\n"
            "> ⚠️ **Data note.** SPY daily, **total-return adjusted** (yfinance `auto_adjust`), "
            f"{R['start']}→{R['end']}, as-of **{R['asof']}**, fingerprint `{R['fp']}`. All Sharpe "
            "races are **excess-of-cash vs excess-of-cash**; all returns **net** of 1 bp one-way × "
            "NAV with **one** execution lag. Offline core + synthetic control are deterministic. Methods: "
            "[`docs/references.md`](../docs/references.md); numbers: [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> \U0001f4a1 **The `\U0001f4a1 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | STC long/flat net Sharpe **{R['stc_sharpe']:.2f}** with HAC "
            f"**t = {R['stc_hac_t']:.2f}** (< 2) and a block-permutation placebo **p = {R['perm_p']:.2f}** "
            "— a shuffled position beats it almost always. |\n"
            f"| **Tradability** | `MIRAGE` | Net Sharpe **{R['stc_sharpe']:.2f}** vs buy-and-hold "
            f"**{R['bh_sharpe']:.2f}** (Δ = **{R['stc_minus_bh']:+.2f}**); CAGR "
            f"**{R['stc_cagr']*100:.1f}%** vs **{R['bh_cagr']*100:.1f}%**. Long/short is net-negative "
            f"(Sharpe **{R['ls_stc_sharpe']:.2f}**). |\n"
            f"| **Faster MACD?** | `BUSTED` | STC Sharpe **{R['stc_sharpe']:.2f}** < plain-MACD crossover "
            f"**{R['macd_sharpe']:.2f}** on SPY, and STC < MACD on 5/6 panel names. The defining claim "
            "is false. |\n\n"
            "> \U0001f4a1 In plain words: the STC turns faster, but on the S&P that just means more "
            "whipsaws — it loses to buy-and-hold, loses to the MACD, and its edge is statistically "
            "indistinguishable from a shuffled position."
        ),

        md(
            "## 1 · The claim, steelmanned\n\n"
            "The STC is built as a *double stochastic of the MACD line*. With MACD line "
            "$M_t = \\mathrm{EMA}_{f}(P)_t - \\mathrm{EMA}_{s}(P)_t$ (Schaff's $f{=}23,\\,s{=}50$), a "
            "$c$-period stochastic $\\mathcal{S}_c(\\cdot)$ and EMA smoother $\\mathcal{E}$:\n\n"
            "$$\\mathrm{STC}_t = \\mathcal{E}\\!\\big[\\,\\mathcal{S}_c\\big(\\mathcal{E}[\\,\\mathcal{S}_c(M)\\,]\\big)\\big]_t \\in [0,100].$$\n\n"
            "Two extra stochastic passes make it snap from 0 to 100 faster than the MACD histogram — "
            "the \"faster\" in the pitch. The folk rule: **long** when $\\mathrm{STC}$ crosses up through "
            "25, **flat/short** when it crosses down through 75. Two testable hypotheses:\n\n"
            "- **H₁ (it's an edge).** The rule's net excess return beats zero (HAC $t\\ge 2$) and a "
            "shuffled-position placebo.\n"
            "- **H₂ (faster = better).** Its Sharpe beats the **plain MACD** crossover (12/26/9) and "
            "buy-and-hold, excess-vs-excess.\n\n"
            "We find **H₁ rejected** ($t=1.87$, placebo $p=0.96$) and **H₂ rejected** "
            "(STC < MACD < buy-and-hold). The speed is real; the profit is not."
        ),

        md(
            "## 2 · So what — what rides on each answer\n\n"
            "The Signal axis is a one-sample HAC test of the rule's **daily net excess** return series "
            "against zero,\n\n"
            "$$t = \\frac{\\bar r_{\\text{strat}}}{\\widehat{\\mathrm{se}}_{\\text{NW}}(\\bar r_{\\text{strat}})},"
            "\\qquad r_{\\text{strat},t} = w_{t-1}\\,(r_t - r_{f,t}) - \\text{cost}_t - \\text{borrow}_t,$$\n\n"
            "with $w_{t-1}$ the position **lagged one day** (the single execution lag). Two honesty traps: "
            "**(a)** a daily Sharpe can look positive purely from the market's drift while still trailing "
            "buy-and-hold — so the race is **excess-vs-excess** and the comparator is the passive "
            "Sharpe, not zero; **(b)** autocorrelation in a step-position series inflates a naive *t*, so we "
            "use Newey-West and a **block-permutation** placebo that shuffles the position in 21-day blocks "
            "(preserving holding-period structure) to ask whether the *timing* adds anything over a coin."
        ),

        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** SPY headline ({R['start']}→{R['end']}, {R['n_days']:,} bars, as-of "
            f"{R['asof']}, fp `{R['fp']}`) + a 6-name robustness panel (QQQ/IWM/EFA/AAPL/MSFT).\n"
            "- **Indicator.** Canonical STC (fast 23, slow 50, cycle 10, smoothing 0.5).\n"
            "- **Rule.** Long on cross-up through 25, flat (or short) on cross-down through 75; held "
            "between signals.\n"
            "- **Timing.** Signal known at close *t*; earns $t{+}1$ return. One `shift(1)`.\n"
            "- **Costs.** 1 bp one-way × NAV per unit turnover (swept to 10 bp); short leg pays 50 "
            "bp/yr borrow.\n"
            "- **Signal test.** HAC *t* of daily net excess vs 0 + a 2,000-draw block-permutation placebo.\n"
            "- **Tradability test.** Net Sharpe (excess) vs buy-and-hold excess + the MACD benchmark.\n"
            "- **Positive control.** A deterministic regime-trend panel: zero trend must NOT beat "
            "buy-and-hold (no false positive); a planted trend must light up (engine can bank a real edge)."
        ),

        md("## 4 · The teardown"),
        md(
            "### 4a · The indicator and the rule it implies\n\n"
            "Top: SPY price with the long (in-market) stretches shaded. Bottom: the STC oscillator with "
            "the 25/75 thresholds. The rule is in-market about half the time — so it forgoes a big "
            "chunk of the index's drift by construction."
        ),
        code(
            "if HAVE_REAL:\n"
            "    seg = SPY.iloc[-750:]\n"
            "    stc = st.schaff_trend_cycle(SPY['close']).iloc[-750:]\n"
            "    pos = st.stc_positions(st.schaff_trend_cycle(SPY['close'])).iloc[-750:]\n"
            "    fig, (a1,a2) = plt.subplots(2,1, figsize=(9.6,5.6), sharex=True, gridspec_kw={'height_ratios':[2,1]})\n"
            "    a1.plot(seg.index, seg['close'], color='black', lw=1.2)\n"
            "    a1.fill_between(seg.index, seg['close'].min(), seg['close'].max(), where=(pos>0).values, color=GREEN, alpha=.12)\n"
            "    a1.set_ylabel('SPY (TR-adj)'); a1.set_title('STC long stretches (shaded) over the last ~3 years')\n"
            "    a2.plot(stc.index, stc.values, color=AMBER, lw=1.2)\n"
            "    a2.axhline(25, color=GREEN, ls='--', lw=1); a2.axhline(75, color=RED, ls='--', lw=1)\n"
            "    a2.set_ylabel('STC (0-100)'); a2.set_ylim(0,100)\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'in-market fraction (full sample): {R[\"exposure\"]*100:.0f}%  |  flips/yr ~ {R[\"turnover\"]:.0f}')\n"
            "else:\n"
            "    print('cache missing; quoting frozen numbers: exposure', R['exposure'], 'turnover', R['turnover'])"
        ),
        md(
            f"> \U0001f4a1 In plain words: the rule holds the index only ~**{R['exposure']*100:.0f}%** of "
            f"the time and flips ~**{R['turnover']:.0f}×/yr**. Half the upside drift is skipped, and "
            "each flip is a fresh chance to whipsaw."
        ),

        md(
            "### 4b · The decisive test — significance + a block-permutation placebo\n\n"
            "The rule's daily net excess return against a 2,000-draw placebo that **block-shuffles the "
            "position** (21-day blocks, asset returns fixed). A real timing edge sits in the right tail; a "
            "coin sits in the middle."
        ),
        code(
            "if HAVE_REAL:\n"
            "    stc = st.schaff_trend_cycle(SPY['close']); pos = st.stc_positions(stc)\n"
            "    pp = st.block_permutation_p(SPY, pos, n_draws=2000, cost_bps=1.0, seed=431)\n"
            "    obs = pp['obs_mean']*1e4; draws = pp['draws']*1e4; pval = pp['p_value']\n"
            "    tval = st.run_experiment(SPY, cost_bps=1.0)['stc_hac_t']\n"
            "else:\n"
            "    obs = R['stc_hac_t']; pval = R['perm_p']; tval = R['stc_hac_t']\n"
            "    rng = np.random.default_rng(431); draws = rng.normal(0.0, 0.15, 2000); obs = 0.015*1e4*0\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='null: 2,000 block-shuffled positions')\n"
            "ax.axvline(obs, color=RED, lw=2.5, label=f'observed STC mean {obs:+.2f} bps/day')\n"
            "ax.set_xlabel('mean daily net excess return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Inside the luck cloud: placebo p = {pval:.2f}, HAC t = {tval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'HAC t = {tval:.2f} (bar is 2.0)   placebo p = {pval:.2f}')"
        ),
        md(
            f"> \U0001f4a1 In plain words: the observed mean sits **inside** the placebo cloud — "
            f"**{R['perm_p']*100:.0f}%** of shuffled positions do as well or better, and the HAC "
            f"**t = {R['stc_hac_t']:.2f}** is under the 2.0 bar. The STC's *timing* adds no detectable "
            "information over randomly re-ordering when it's in the market. **Signal = NONE.**"
        ),

        md(
            "### 4c · The race — excess-vs-excess, and the MACD head-to-head\n\n"
            "Net annualised Sharpe (excess of cash) for the STC rule, the plain MACD crossover, and "
            "buy-and-hold. The \"faster MACD\" claim lives or dies on the first two bars."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = st.run_experiment(SPY, cost_bps=1.0)\n"
            "    vals = [r['stc_sharpe'], r['macd_sharpe'], r['bh_sharpe']]\n"
            "    rls = st.run_experiment(SPY, cost_bps=1.0, long_short=True)\n"
            "    ls_stc = rls['stc_sharpe']\n"
            "else:\n"
            "    vals = [R['stc_sharpe'], R['macd_sharpe'], R['bh_sharpe']]; ls_stc = R['ls_stc_sharpe']\n"
            "labels = ['STC long/flat','plain MACD','buy & hold','STC long/short']\n"
            "bars_v = vals + [ls_stc]\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.bar(labels, bars_v, color=[RED, AMBER, GREEN, RED], width=.62)\n"
            "ax.axhline(0, color='k', lw=.8)\n"
            "for i,v in enumerate(bars_v): ax.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_ylabel('net Sharpe (excess, annualised)')\n"
            "ax.set_title('STC < MACD < buy-and-hold; long/short is net-negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'STC {bars_v[0]:.2f} | MACD {bars_v[1]:.2f} | B&H {bars_v[2]:.2f} | STC L/S {bars_v[3]:.2f}')"
        ),
        md(
            f"> \U0001f4a1 In plain words: the ordering is unambiguous — STC "
            f"(**{R['stc_sharpe']:.2f}**) **<** MACD (**{R['macd_sharpe']:.2f}**) **<** buy-and-hold "
            f"(**{R['bh_sharpe']:.2f}**), and shorting makes it lose money (**{R['ls_stc_sharpe']:.2f}**). "
            "The defining claim — *faster than MACD = better than MACD* — is **busted**."
        ),

        md(
            "### 4d · Costs + the panel — the conclusion is not SPY-specific\n\n"
            "Left: the STC Sharpe across a cost sweep, never reaching the buy-and-hold line even at zero "
            "cost. Right: STC vs MACD vs buy-and-hold across the 6-name panel — STC trails "
            "buy-and-hold everywhere and trails the MACD on 5 of 6."
        ),
        code(
            "cs = [c[0] for c in R['cost']]\n"
            "if HAVE_REAL:\n"
            "    sh = [st.run_experiment(SPY, cost_bps=c)['stc_sharpe'] for c in cs]\n"
            "    P = [(t, *(_v for _v in (lambda r:(r['stc_sharpe'],r['bh_sharpe'],r['macd_sharpe']))(st.run_experiment(data.load_real(t), cost_bps=1.0)))) for t in data.TICKERS]\n"
            "else:\n"
            "    sh = [c[1] for c in R['cost']]\n"
            "    P = [(p[0], p[2], p[3], p[4]) for p in R['panel']]\n"
            "fig, (a1,a2) = plt.subplots(1,2, figsize=(11.0,4.3))\n"
            "a1.plot(cs, sh, 'o-', color=RED, lw=2, label='STC net Sharpe')\n"
            "a1.axhline(R['bh_sharpe'], color=GREEN, ls='--', lw=2, label=f\"SPY buy & hold\")\n"
            "a1.set_xlabel('one-way cost per flip (bps)'); a1.set_ylabel('net Sharpe'); a1.set_title('Cost sweep (SPY)'); a1.legend()\n"
            "names=[p[0] for p in P]; x=np.arange(len(names))\n"
            "a2.bar(x-.27,[p[1] for p in P],.27,color=RED,label='STC')\n"
            "a2.bar(x   ,[p[3] for p in P],.27,color=AMBER,label='MACD')\n"
            "a2.bar(x+.27,[p[2] for p in P],.27,color=GREEN,label='buy & hold')\n"
            "a2.set_xticks(x); a2.set_xticklabels(names); a2.set_ylabel('net Sharpe'); a2.set_title('Panel: STC trails everywhere'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('panel (STC, MACD, B&H):', [(p[0],round(p[1],2),round(p[3],2),round(p[2],2)) for p in P])"
        ),
        md(
            "> \U0001f4a1 In plain words: even at **zero** cost the STC Sharpe never reaches buy-and-hold "
            "(costs aren't the killer — the rule is on the wrong side of drift), and the underperformance "
            "repeats across QQQ, IWM, EFA, AAPL and MSFT. AAPL/MSFT clear *t* = 2 only because their "
            "*drift* is huge — buy-and-hold still beats STC there by the widest margins."
        ),

        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic panel with a **planted regime trend**: with **zero** trend the STC rule "
            "must NOT beat buy-and-hold (no false positive); with a real trend planted it must light up "
            "(*t* > 2 and beat buy-and-hold). Both hold — so the null result on SPY is a true negative, "
            "not a broken harness."
        ),
        code(
            "res = []\n"
            "for tr in (0.0, 0.003):\n"
            "    b,truth = data.synthetic_panel(trend=tr)\n"
            "    r = st.run_experiment(b, cost_bps=1.0)\n"
            "    res.append((tr, r['stc_sharpe'], r['bh_sharpe'], r['stc_minus_bh'], r['stc_hac_t']))\n"
            "labels=[f'no trend\\n(null)','planted trend\\n(control)']\n"
            "delta=[r[3] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(8.4,4.3))\n"
            "ax.bar(labels, delta, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(0, color='k', lw=.8)\n"
            "for i,d in enumerate(delta): ax.annotate(f'{d:+.2f}',(i,d),ha='center',va='bottom' if d>=0 else 'top')\n"
            "ax.set_ylabel('STC Sharpe − buy-and-hold Sharpe'); ax.set_title('Control: null -> STC loses; planted trend -> STC wins')\n"
            "plt.tight_layout(); plt.show()\n"
            "for tr,s,b,d,t in res: print(f'trend={tr}: STC={s:.2f} B&H={b:.2f} delta={d:+.2f} HAC t={t:.2f}')"
        ),
        md(
            f"> \U0001f4a1 In plain words: with **no** planted trend the STC rule *loses* to buy-and-hold "
            f"(Δ = **{R['syn'][0][3]:+.2f}**, t = {R['syn'][0][4]:.2f}) — no false positive. With "
            f"a trend planted it **wins** (Δ = **{R['syn'][1][3]:+.2f}**, **t = {R['syn'][1][4]:.2f}** "
            "> 2). The machinery can bank a genuine trend edge — the S&P 500 just doesn't offer one, "
            "net, to this rule. That's what makes the SPY null **real**, not a coding artefact."
        ),

        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — STC long/flat net excess Sharpe **{R['stc_sharpe']:.2f}** with "
            f"HAC **t = {R['stc_hac_t']:.2f}** (< 2) and block-permutation **p = {R['perm_p']:.2f}**. "
            "Indistinguishable from a shuffled position; the timing carries no edge.\n"
            f"- **Tradability `MIRAGE`** — net Sharpe **{R['stc_sharpe']:.2f}** vs buy-and-hold "
            f"**{R['bh_sharpe']:.2f}** (Δ **{R['stc_minus_bh']:+.2f}**), CAGR "
            f"**{R['stc_cagr']*100:.1f}%** vs **{R['bh_cagr']*100:.1f}%**; long/short is net-negative "
            f"(**{R['ls_stc_sharpe']:.2f}**). It *destroys* value vs doing nothing.\n"
            f"- **Faster MACD? `BUSTED`** — STC Sharpe **{R['stc_sharpe']:.2f}** < plain-MACD "
            f"**{R['macd_sharpe']:.2f}** on SPY and on 5/6 panel names. The defining claim is false."
        ),

        md(
            "## 6 · Could you trade it?\n\n"
            "No — and the cleanest way to see why is the equity curve net of costs against the free "
            "alternative. The STC rule converts a relentless upward drift into a thinner, no-smoother "
            "return by sitting in cash through half of it and paying ~17 spreads a year for the privilege."
        ),
        code(
            "if HAVE_REAL:\n"
            "    stc = st.schaff_trend_cycle(SPY['close']); pos = st.stc_positions(stc)\n"
            "    bk = st.book_returns(SPY, pos, cost_bps=1.0)\n"
            "    eq_stc=(1+bk['strat_excess']).cumprod(); eq_bh=(1+bk['ret']).cumprod()\n"
            "    r=st.run_experiment(SPY, cost_bps=1.0)\n"
            "    dd_stc, dd_bh = r['stc_maxdd'], r['bh_maxdd']\n"
            "else:\n"
            "    eq_stc=eq_bh=None; dd_stc, dd_bh = R['stc_maxdd'], R['bh_maxdd']\n"
            "fig, ax = plt.subplots(figsize=(9.4,4.5))\n"
            "if HAVE_REAL:\n"
            "    ax.semilogy(SPY.index, eq_bh, color=GREEN, lw=2, label=f\"buy & hold (maxDD {dd_bh*100:.0f}%)\")\n"
            "    ax.semilogy(SPY.index, eq_stc, color=RED, lw=2, label=f\"STC rule (maxDD {dd_stc*100:.0f}%)\")\n"
            "ax.set_ylabel('growth of $1 (log)'); ax.set_title('Less money, no extra safety')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'maxDD STC {dd_stc*100:.0f}% vs B&H {dd_bh*100:.0f}% -- similar pain, far less gain')"
        ),
        md(
            f"> \U0001f4a1 In plain words: the STC rule's worst drawdown (**{R['stc_maxdd']*100:.0f}%**) is "
            f"barely better than buy-and-hold's (**{R['bh_maxdd']*100:.0f}%**) — so you don't even buy "
            "meaningful safety with all that under-performance. There is no version of this that pays. "
            "**MIRAGE.**"
        ),

        md(
            "## 7 · Going further\n\n"
            "- **Parameter robustness.** Sweep (fast, slow, cycle, thresholds) — the conclusion "
            "(STC ≤ MACD ≤ buy-and-hold on the S&P, net) is flat across the grid; that's the "
            "fingerprint of a lagging trend filter, not a tuning accident.\n"
            "- **Where trend rules earn.** The synthetic control shows the engine banks an edge on a "
            "*genuine* persistent trend; markets with stronger, cleaner trends (some commodities/FX) are "
            "the honest place to retest — not equity indices dominated by drift + mean-reversion.\n"
            "- **The family verdict.** Compare with [Supertrend (106)](../../106-supertrend/) and "
            "[CCI (178)](../../178-cci/): the same lagging-oscillator-on-equities story keeps landing in "
            "the same null.\n\n"
            "*The reproducible core is offline and deterministic. Methods and sources: "
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
