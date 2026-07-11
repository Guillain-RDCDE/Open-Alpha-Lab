"""Generate the two narrative notebooks for Study 678 (Random-Walk-Index).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY/QQQ/IWM/DIA/GLD
tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily total-return
# OHLC, SPY + QQQ/IWM/DIA/GLD basket, 2005-01-03 -> 2026-06-30).
R = dict(
    start="2005-01-03", end="2026-06-30", periods=(2, 3, 4, 5, 6),
    n_flag=3067, n_rest=2336, flag_freq=56.8,
    flag_bps=+2.41, rest_bps=+8.07, gap_bps=-5.66,
    welch_t=-1.64, nw_t=-1.74,
    hit_up=1691, hit_pct=55.1, wilson=(53.4, 56.9),
    placebo_obs=+2.41, placebo_mean=+4.85, placebo_sd=1.41, placebo_p=0.9591, placebo_draws=20000,
    # cross-instrument pooled + per-ticker
    per_ticker={
        "SPY": (3067, +2.41, +8.07, -5.66, -1.64, -1.74, 55.1),
        "QQQ": (3013, +4.98, +8.84, -3.86, -0.99, -1.06, 55.9),
        "IWM": (2783, +1.06, +8.40, -7.34, -1.76, -1.82, 52.4),
        "DIA": (2955, +3.84, +5.25, -1.41, -0.44, -0.47, 55.4),
        "GLD": (2771, +5.04, +4.24, +0.80, +0.26, +0.27, 52.4),
    },
    pooled_n=27015, pooled_flag_bps=+3.47, pooled_rest_bps=+6.92, pooled_t=-2.16,
    # the book
    gross_ret=+81.76, gross_sharpe=+0.301,
    net5_ret=+0.55, net5_sharpe=+0.060,
    net10_ret=-44.39, net10_sharpe=-0.181,
    bh_ret=+831.34, bh_sharpe=+0.644,
    exposure=56.8, n_trades=1184, ann_cost=2.76,
    # random-entry control
    rc_mean_ret=+98.60, rc_sd_ret=58.86, rc_mean_sharpe=+0.280, rc_sd_sharpe=0.105,
    rc_beat_ret_pct=90, rc_beat_sharpe_pct=95,
    # synthetic control
    syn_null_mean=-0.03, syn_null_sd=1.09, syn_null_fire=1, syn_planted_t=+9.54,
    syn_planted_hit=64.6,
    fp_spy="98e6db1a1e26", fp_qqq="70e1bc4a725e", fp_iwm="7c02d693031f",
    fp_dia="f1faf16222de", fp_gld="a0f0673984c4",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Beats_a_coin%3F: Busted](https://img.shields.io/badge/Beats_a_coin%3F-Busted-8b949e?style=flat-square)\n\n"
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

from random_walk_index import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    BASKET = data.load_basket()
    SPY_RAW = BASKET[data.HEADLINE]
    DF = st.day_frame(SPY_RAW)
else:
    BASKET = SPY_RAW = DF = None
print("real cache present:", HAVE_REAL, "| SPY tape days:", (0 if DF is None else len(DF)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does \"it moved farther than a random walk\" actually mean it will keep moving? 🎲📈\n"
            "### The Random Walk Index — a clever idea from 1990 that turns out to lose to a "
            "coin flip\n\n"
            + BADGES +
            "Every trader has felt it: a stock breaks out and it *feels* different from the "
            "usual daily wiggle — bigger, more purposeful, less random. In 1990, Michael Poulos "
            "gave that feeling a formula: compare how far the price actually moved to how far a "
            "**pure random walk**, with the market's own recent choppiness, would be expected to "
            "wander over the same stretch. If the real move is bigger than the random-walk "
            "benchmark — the **Random Walk Index reads above 1** — the claim is that this is a "
            "*real* trend, not noise, and you should ride it.\n\n"
            "That's the claim we test: *can you tell a real trend from noise just by measuring "
            "how far price has already gone?* Spoiler: measuring distance travelled is not the "
            "same as forecasting where it goes next.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** SPY plus a QQQ/IWM/DIA/GLD basket, daily total-return-adjusted "
            "OHLC, 2005→2026. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does a big RWI reading predict a better *next* day? | **No — if anything, "
            f"worse.** SPY sessions flagged \"trending\" (RWI-high > 1) return **{R['flag_bps']:+.2f} "
            f"bps** the next session; unflagged sessions return **{R['rest_bps']:+.2f} bps**. "
            "Pooled across five markets the gap is *statistically real* — and runs **backwards**. |\n"
            f"| Is the flag at least rare enough to mean something? | **No.** It fires on "
            f"**{R['flag_freq']:.0f}%** of all sessions — barely different from a coin flip. |\n"
            f"| Can you actually trade the \"long only when RWI says trend\" rule? | You can, "
            f"but it captures only **{R['gross_ret']:.0f}%** of buy & hold's **{R['bh_ret']:.0f}%** "
            "total return over 21.5 years *before costs*, and goes **negative** after realistic "
            "trading costs. |\n"
            f"| Does it at least beat picking days at random? | **No.** A random selection of "
            f"days with the *same* time-in-market beats the real rule in **{R['rc_beat_ret_pct']}%** "
            "of trials. |\n\n"
            "> Distance travelled isn't a crystal ball."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A random walk with a given day-to-day choppiness (its Average True Range) can "
            "only be expected to wander so far in *n* days. If price has actually moved *farther* "
            "than that — RWI > 1 — the move isn't noise, it's a real trend. Ride it.\"*\n\n"
            "It's an elegant idea: instead of eyeballing a chart, put a number on \"this doesn't "
            "look random.\" The formula borrows real math — over *n* independent random steps of "
            "typical size *ATR*, a pure random walk's expected net displacement scales like "
            "*ATR × √n*, the same square-root-of-time logic behind a Sharpe ratio. A move that "
            "clears that benchmark should, in principle, be less likely under pure randomness."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If this worked, it would be a genuinely useful, mechanical filter: a single number "
            "that separates \"real move, keep riding\" from \"just noise, stand aside\" — no "
            "chart-reading required, works on any instrument with a High/Low/Close. It's built "
            "into essentially every charting platform (StockCharts, TradingView) for exactly this "
            "reason. So we ask the only question that matters: **does a session flagged this way "
            "actually pay better than one that isn't?**"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **The indicator.** RWI-high, the max reading across five short lookbacks (2 to 6 "
            "sessions) — Poulos' own scanning method.\n"
            "- **The comparison.** The return of the session *after* a flag vs the session after "
            "no flag — with one honest execution lag (the flag needs today's own High/Low, so "
            "you can only act at today's close, earning tomorrow's return).\n"
            "- **The luck check.** Pick a random subset of sessions the same size as the flagged "
            "set, 20,000 times — how often does *that* do as well?\n"
            "- **The fair race.** Not vs buy & hold (that just measures \"was the market up\") "
            "but vs a **random-entry control matched for time-in-market** — same exposure, same "
            "trading frequency, no timing skill. That's the bar a *timing* rule actually has to "
            "clear."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Average next-session return, flagged vs not, on SPY."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.trend_day_stats(DF)\n"
            "    fp, rp = s['flag_bps'], s['rest_bps']\n"
            "else:\n"
            "    fp, rp = R['flag_bps'], R['rest_bps']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.4))\n"
            "ax.bar(['RWI-high > 1\\n(\"trend\" flagged)','no flag\\n(everything else)'], [fp, rp],\n"
            "       color=[RED, GREY], width=.6)\n"
            "for i,v in enumerate([fp, rp]): ax.annotate(f'{v:+.2f} bps',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('average next-session return (bps)')\n"
            "ax.set_title('The \"real trend\" flag pays LESS than an ordinary day')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'flag day {fp:+.2f} bps   no-flag day {rp:+.2f} bps')"
        ),
        md(
            f"That's the opposite of the claim: **{R['flag_bps']:+.2f} bps** the day after a "
            f"flag vs **{R['rest_bps']:+.2f} bps** otherwise. And the flag isn't rare — it fires "
            f"on **{R['flag_freq']:.0f}%** of all sessions, because taking the *maximum* of five "
            "overlapping lookback windows makes a raw \"> 1\" threshold much easier to clear than "
            "it sounds.\n\n"
            "**Is SPY just unlucky?** Same test on four more liquid markets:"
        ),
        code(
            "tickers = list(R['per_ticker'])\n"
            "if HAVE_REAL:\n"
            "    gaps, ts = [], []\n"
            "    for t in tickers:\n"
            "        d = st.day_frame(BASKET[t])\n"
            "        r = st.trend_day_stats(d)\n"
            "        gaps.append(r['gap_bps']); ts.append(r['welch_t'])\n"
            "else:\n"
            "    gaps = [R['per_ticker'][t][3] for t in tickers]\n"
            "    ts = [R['per_ticker'][t][4] for t in tickers]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "cols = [RED if g < 0 else GREEN for g in gaps]\n"
            "ax.bar(tickers, gaps, color=cols, width=.6)\n"
            "for i,(g,t_) in enumerate(zip(gaps, ts)):\n"
            "    ax.annotate(f'{g:+.1f} bps\\n(t={t_:+.2f})',(i,g),ha='center',\n"
            "        va='top' if g<0 else 'bottom', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('flag-day minus no-flag-day next-session return (bps)')\n"
            "ax.set_title('Four of five markets: flagging \"trend\" picks a WORSE day')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('gaps (bps):', dict(zip(tickers, [round(g,2) for g in gaps])))"
        ),
        md(
            f"Pooled across all five (n = {R['pooled_n']:,}), the wrong-signed gap is "
            f"**statistically real** (Welch *t* = **{R['pooled_t']:+.2f}**) — this isn't just "
            "noise, it's a small, genuine headwind. Flag a day as \"real trend, not random\" and "
            "you have mechanically selected a session that pays a little *less* than average, "
            "not more — a signature of short-term mean reversion after a sharp burst, not "
            "continuation.\n\n"
            "**Now the actual trade.** Be long only when the flag says trend:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    bt5 = st.backtest(DF, cost_bps=5.0)\n"
            "    g, n5 = bt5['bh_total_return_pct'], bt5['total_return_pct']\n"
            "else:\n"
            "    g, n5 = R['bh_ret'], R['net5_ret']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.4))\n"
            "ax.bar(['buy & hold SPY','RWI-high timer\\n(net 5 bps)'], [g, n5],\n"
            "       color=[GREEN, RED], width=.55)\n"
            "for i,v in enumerate([g, n5]): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('total return, 2005 -> 2026 (%)')\n"
            "ax.set_title('The timer captures a fraction of buy & hold — and that is BEFORE 10bps costs')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'buy & hold {g:+.1f}%   RWI timer (net 5bps) {n5:+.1f}%')"
        ),
        md(
            f"Buy & hold turns $1 into roughly **${1+R['bh_ret']/100:.2f}**; the RWI-timed book, "
            f"net of a realistic 5 bps one-way cost, barely breaks even (**{R['net5_ret']:+.1f}%** "
            f"total) — and at 10 bps it's **{R['net10_ret']:+.1f}%**, a loss. The rule trades "
            f"**{R['n_trades']:,} times** over 21.5 years flipping in and out of a "
            f"**{R['flag_freq']:.0f}%**-of-the-time exposure, and every flip costs money.\n\n"
            "**But maybe any timer that's invested 57% of the time in a rising market would look "
            "OK — is this really about the *timing*, or just about being in the market at all?** "
            "That's the fair question, and the fair test is a **random-entry control**: pick "
            "random days with the exact same total time-in-market and the same trading frequency, "
            "and see how the real rule stacks up."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rc = st.random_control_backtest(DF, cost_bps=5.0, block_size=21, n_seeds=20)\n"
            "    draws = rc['draws_total_return_pct']\n"
            "else:\n"
            "    rng = np.random.default_rng(678)\n"
            "    draws = rng.normal(R['rc_mean_ret'], R['rc_sd_ret'], 20)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.hist(draws, bins=12, color=GREY, alpha=.85,\n"
            "        label='random-entry control (20 seeds, same exposure & turnover)')\n"
            "ax.axvline(R['net5_ret'], c=RED, lw=2.5, label=f\"RWI timer (net 5bps): {R['net5_ret']:+.1f}%\")\n"
            "ax.set_xlabel('total return, 2005 -> 2026 (%)')\n"
            "ax.set_ylabel('draws')\n"
            "ax.set_title(f\"Random days with the SAME time-in-market beat the RWI timer \"\n"
            "             f\"in {R['rc_beat_ret_pct']}% of draws\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"random-entry mean {draws.mean():+.1f}%  vs RWI timer {R['net5_ret']:+.1f}%\")"
        ),
        md(
            f"A random selection of days with the *exact same* time-in-market and trading "
            f"frequency **beats** the real RWI-timed book in **{R['rc_beat_ret_pct']}%** of "
            f"trials by total return and **{R['rc_beat_sharpe_pct']}%** by Sharpe. The RWI flag "
            "isn't a neutral coin flip that happens not to help — it's actively worse than one."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Flag days pay less than no-flag days on SPY, and the "
            f"wrong-signed gap is statistically real pooled across five markets (Welch "
            f"*t* = {R['pooled_t']:+.2f}). A random day-count-matched subset beats the flag's own "
            "mean 96% of the time.\n"
            "- **Tradability — Mirage.** The book captures a tenth of buy & hold's return before "
            "costs and goes negative after realistic costs.\n"
            "- **\"Beats a coin?\" — Busted.** A same-exposure random-entry control outperforms "
            "the real rule on both return and Sharpe most of the time."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The general lesson.** Measuring how far price has *already* moved, however "
            "cleverly scaled, describes the past — it does not, by itself, forecast the future. "
            "This desk has now tested four different \"is this a real trend\" statistics (ADX, "
            "the Vertical-Horizontal-Filter, a rolling Hurst exponent, and now the RWI) and every "
            "one loses to a plain random or ungated baseline.\n"
            "- **Where a range statistic like the RWI might still earn its keep** is as a "
            "*volatility-regime* filter for position sizing (bigger true range → smaller size), "
            "not as a directional trigger — a genuinely different question from \"which way does "
            "it go next.\"\n"
            "- **Sibling studies:** [108-adx-filter](../../108-adx-filter/), "
            "[484-vertical-horizontal-filter](../../484-vertical-horizontal-filter/) and "
            "[397-hurst-regime](../../397-hurst-regime/) — the same family, different formula, "
            "same result.\n\n"
            "*Think you can build a trend-strength gate that actually works? Show a net, "
            "certifiable edge against a random-entry control matched for exposure — after costs "
            "— then we'll talk.*"
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
            "# The Random Walk Index — a quantitative teardown 🔬\n"
            "### Flag-day Welch/HAC splits · a matched-count random-day placebo · cross-instrument "
            "pooling · the fair (block-shuffled, exposure-matched) trading control · cost sweeps "
            "· a 20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "Poulos' **Random Walk Index** compares realized price displacement to the expected "
            "displacement of a pure random walk of the same Average True Range — "
            "RWI-high(n) = (High_t − Low_{t−n}) / (ATR_n(t) × √n), reported as the max over "
            "n = 2..6 — and claims RWI-high > 1 flags a real, ride-able trend. The job here is "
            "to measure that honestly, on the real tape, against a fair control.\n\n"
            "> ⚠️ **Data note.** SPY + QQQ/IWM/DIA/GLD daily total-return-adjusted OHLC "
            "(2005→2026), yfinance, cached. No survivorship anywhere in this study (index ETFs / "
            "a commodity tracker, no cross-sectional panel). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_spy"] + "` / `" +
            R["fp_qqq"] + "` / `" + R["fp_iwm"] + "` / `" + R["fp_dia"] + "` / `" + R["fp_gld"] +
            "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | flag-day next-session return **{R['flag_bps']:+.2f} bps** "
            f"vs {R['rest_bps']:+.2f} bps (SPY Welch *t* = {R['welch_t']:+.2f}); pooled 5-market "
            f"Welch *t* = **{R['pooled_t']:+.2f}** (wrong-signed, clears the bar); random-day "
            f"placebo *p* = {R['placebo_p']:.4f} |\n"
            f"| **Tradability** | `MIRAGE` | timer +{R['gross_ret']:.1f}% gross / "
            f"{R['net5_ret']:+.1f}% net(5bp) / {R['net10_ret']:+.1f}% net(10bp) vs buy&hold "
            f"+{R['bh_ret']:.1f}%; random-entry control beats it in {R['rc_beat_ret_pct']}% of "
            "draws |\n"
            f"| **Beats a coin?** | `BUSTED` | exposure-matched random control mean "
            f"{R['rc_mean_ret']:+.1f}% (sd {R['rc_sd_ret']:.1f}) vs timer {R['net5_ret']:+.1f}% |\n\n"
            "> 💡 In plain words: the indicator measures a real, well-defined displacement "
            "statistic — it just isn't predictive, and pooled it's mildly predictive of the "
            "*wrong* thing."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $D_{n,t} = \\text{High}_t - \\text{Low}_{t-n}$ be the realized *n*-session "
            "high-low displacement and $\\overline{ATR}_n(t)$ the simple *n*-bar average True "
            "Range. Under a pure random walk with i.i.d. step size ~ $\\overline{ATR}_n$, the "
            "expected net displacement over *n* steps scales like $\\overline{ATR}_n \\sqrt{n}$ "
            "(the standard Brownian-motion identity). Poulos defines\n\n"
            "$$\\mathrm{RWI\\text{-}high}(n,t) = \\frac{D_{n,t}}{\\overline{ATR}_n(t)\\sqrt{n}}, "
            "\\qquad \\mathrm{RWI\\text{-}high}(t) = \\max_{n \\in \\{2..6\\}} "
            "\\mathrm{RWI\\text{-}high}(n,t)$$\n\n"
            "and the operational claim: **flag $\\{\\mathrm{RWI\\text{-}high}(t) > 1\\}$ predicts "
            "a real, ride-able uptrend continuation.**\n\n"
            "- **H₁ (predictive).** $E[\\text{fwd ret} \\mid \\text{flag}_t] > E[\\text{fwd ret} "
            "\\mid \\lnot\\text{flag}_t]$ — flagged sessions pay more, going forward.\n"
            "- **H₂ (rare event).** The flag should fire on a minority of sessions — a "
            "\"statistically non-random\" claim loses force if it fires on a coin-flip fraction "
            "of the tape.\n"
            "- **H₃ (tradable).** A long timer built on the flag beats a fair, exposure-matched "
            "control net of realistic costs.\n\n"
            f"We find **H₁ rejected and wrong-signed pooled** (*t* = {R['pooled_t']:+.2f}), "
            f"**H₂ rejected** ({R['flag_freq']:.0f}% flag frequency), **H₃ rejected** (loses to "
            f"buy & hold *and* to a random-entry control in {R['rc_beat_ret_pct']}% of draws)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Flag/no-flag is a **daily, effectively non-overlapping split** (one observation per "
            "session), so the planned primary is a **Welch t** on the next-session-return group "
            "split. Daily returns carry mild serial correlation, so we cross-check with a "
            "**Newey-West (5-lag) t** on the flag-dummy regression. A **matched-count random-day "
            "placebo** (20,000 draws) asks the honest question a raw *t*-stat can't: would *any* "
            "day-count-matched random subset do this well? And because a timing rule invested "
            "57% of the time in a 21-year bull market will show a positive *number* almost by "
            "construction, the tradable-book comparison is against a **block-shuffled, "
            "exposure- and turnover-matched random-entry control**, not buy & hold alone — buy & "
            "hold answers \"was the market up,\" the block-shuffle control answers the question "
            "that actually matters: **does the timing add anything on top of being invested?**"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Indicator.** RWI-high, max over n = {R['periods']} (Poulos' own short-lookback "
            "scan), simple (non-Wilder) ATR.\n"
            f"- **Tape.** SPY + QQQ/IWM/DIA/GLD daily total-return-adjusted OHLC {R['start']} -> "
            f"{R['end']}. As-of 2026-06-30 (last complete month).\n"
            "- **Execution (one documented lag).** Flag computed from data through close(t) "
            "(needs today's own High/Low) -> entered at that same close -> earns "
            "close(t)->close(t+1). A single shift, applied once.\n"
            "- **Headline.** Welch t + NW(5) t + Wilson hit rate + 20-seed x 1,000-draw "
            "matched-count placebo, on SPY and pooled across the basket.\n"
            "- **The book.** 0/1 position on the lagged flag, one-way x NAV cost per leg (0/5/10 "
            "bps), vs buy & hold and vs the block-shuffled random-entry control (21-day blocks, "
            "20 seeds).\n"
            "- **Control.** Synthetic two-regime Markov world, planted trend-persistence-edge "
            "knob; the null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline split and its placebo (SPY)\n\n"
            "Welch t on next-session return (flag vs no-flag), NW t on the dummy regression, "
            "hit rate, and the matched-count random-day null. In the notebook we run a lighter "
            "placebo (4 seeds x 300 draws) and quote the canonical 20,000-draw p from "
            "`results.md`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.trend_day_stats(DF)\n"
            "    print(f\"flag {s['flag_bps']:+.2f} bps  (n={s['n_flag']})  vs  no-flag \"\n"
            "          f\"{s['rest_bps']:+.2f} bps  (n={s['n_rest']})\")\n"
            "    print(f\"Welch t = {s['welch_t']:+.2f}   NW(5) t = {s['nw_t']:+.2f}   \"\n"
            "          f\"hit {s['hit_up']}/{s['n_flag']} = {s['hit_rate']*100:.1f}%  \"\n"
            "          f\"Wilson [{s['hit_lo']*100:.1f}%, {s['hit_hi']*100:.1f}%]\")\n"
            "    pl = st.placebo_pvalue(DF, n_draws_per_seed=300, n_seeds=4)\n"
            "    obs = pl['obs'] * 1e4\n"
            "    rng = np.random.default_rng(678)\n"
            "    draws = rng.choice(DF['fwd_ret'].dropna().values, size=(1200, s['n_flag'])).mean(axis=1) * 1e4\n"
            "else:\n"
            "    obs = R['placebo_obs']\n"
            "    rng = np.random.default_rng(678)\n"
            "    draws = rng.normal(R['placebo_mean'], R['placebo_sd'], 1200)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85,\n"
            "        label='null: random day-count-matched subsets (light in-notebook run)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed flag-day mean {obs:+.2f} bps')\n"
            "ax.set_xlabel('mean next-session return of a random matched-size subset (bps)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"The flag sits BELOW the luck cloud: canonical p = {R['placebo_p']:.4f} \"\n"
            "             '(20 seeds x 1,000 draws)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical placebo (results.md): mean {R['placebo_mean']:+.2f} bps, \"\n"
            "      f\"sd {R['placebo_sd']:.2f}, p = {R['placebo_p']:.4f}\")"
        ),
        md(
            f"> 💡 In plain words: the observed **{R['flag_bps']:+.2f} bps** sits *below* the "
            f"null's center ({R['placebo_mean']:+.2f} ± {R['placebo_sd']:.2f} bps); "
            f"**p = {R['placebo_p']:.4f}** means a random day-count-matched subset does at least "
            "as well as the flag **96% of the time**. H₁ is not merely unsupported on SPY alone — "
            "the flag actively selects a below-average subset."
        ),
        md(
            "### 4b · Cross-instrument pooling — one lucky tape can't carry this\n\n"
            "The same split on SPY, QQQ, IWM, DIA and GLD, per-ticker and pooled."
        ),
        code(
            "tickers = list(R['per_ticker'])\n"
            "if HAVE_REAL:\n"
            "    xi = st.cross_instrument_stats(BASKET)\n"
            "    rows = [(t, xi['per_ticker'][t]['n_flag'], xi['per_ticker'][t]['flag_bps'],\n"
            "             xi['per_ticker'][t]['rest_bps'], xi['per_ticker'][t]['gap_bps'],\n"
            "             xi['per_ticker'][t]['welch_t'], xi['per_ticker'][t]['nw_t'],\n"
            "             xi['per_ticker'][t]['hit_rate']*100) for t in tickers]\n"
            "    pooled_t = xi['pooled']['welch_t']\n"
            "    pooled_flag, pooled_rest = xi['pooled']['flag_bps'], xi['pooled']['rest_bps']\n"
            "else:\n"
            "    rows = [(t, *R['per_ticker'][t]) for t in tickers]\n"
            "    pooled_t, pooled_flag, pooled_rest = R['pooled_t'], R['pooled_flag_bps'], R['pooled_rest_bps']\n"
            "gaps = [r[4] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "cols = [RED if g < 0 else GREEN for g in gaps]\n"
            "ax.bar(tickers, gaps, color=cols, width=.6)\n"
            "for i,(t_,*rest) in enumerate(rows):\n"
            "    ax.annotate(f'{rest[3]:+.1f}\\n(t={rest[4]:+.2f})',(i,rest[3]),ha='center',\n"
            "        va='top' if rest[3]<0 else 'bottom', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('flag minus no-flag next-session return (bps)')\n"
            f"ax.set_title(f'Pooled Welch t = {{pooled_t:+.2f}} (wrong-signed, n={R['pooled_n']:,})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pooled: flag {pooled_flag:+.2f} bps vs rest {pooled_rest:+.2f} bps  t={pooled_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: four of five markets show a negative gap; pooled "
            f"(n = {R['pooled_n']:,}) the Welch *t* is **{R['pooled_t']:+.2f}** — it *clears* "
            "the ±2 desk bar, in the direction that says \"the flag picks worse days,\" not "
            "better ones. This is the strongest single number in the study: not \"we found "
            "nothing,\" but \"we found the opposite of the claim, and it's real.\""
        ),
        md(
            "### 4c · The book — gross, net, and the fair control\n\n"
            "The RWI-high long timer (one-day lag, position 0/1, cost x NAV per leg on every "
            "position change) vs buy & hold *and* vs a block-shuffled random-entry control that "
            "preserves the timer's exposure and turnover profile while destroying the flag's "
            "real calendar placement."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows_bt = [st.backtest(DF, cost_bps=cb) for cb in (0.0, 5.0, 10.0)]\n"
            "    g, n5, n10 = [r['total_return_pct'] for r in rows_bt]\n"
            "    bh = rows_bt[0]['bh_total_return_pct']\n"
            "    rc = st.random_control_backtest(DF, cost_bps=5.0, block_size=21, n_seeds=20)\n"
            "    rc_draws = rc['draws_total_return_pct']\n"
            "else:\n"
            "    g, n5, n10, bh = R['gross_ret'], R['net5_ret'], R['net10_ret'], R['bh_ret']\n"
            "    rng = np.random.default_rng(678)\n"
            "    rc_draws = rng.normal(R['rc_mean_ret'], R['rc_sd_ret'], 20)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.4))\n"
            "a1.bar(['gross','net 5bp','net 10bp','buy&hold'], [g, n5, n10, bh],\n"
            "       color=[GREY, AMBER, RED, GREEN], width=.6)\n"
            "for i,v in enumerate([g, n5, n10, bh]): a1.annotate(f'{v:+.0f}%',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('total return, 2005->2026 (%)')\n"
            "a1.set_title('Costs turn a losing race into a loss')\n"
            "a2.hist(rc_draws, bins=10, color=GREY, alpha=.85, label='random-entry control')\n"
            "a2.axvline(n5, c=RED, lw=2.5, label=f'RWI timer net5bp: {n5:+.1f}%')\n"
            "a2.set_xlabel('total return (%)'); a2.set_title('...and loses to random entry too')\n"
            "a2.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.1f}%  net5 {n5:+.1f}%  net10 {n10:+.1f}%  buy&hold {bh:+.1f}%')\n"
            "print(f'random-entry control: mean {rc_draws.mean():+.1f}%  vs timer {n5:+.1f}%')"
        ),
        md(
            f"> 💡 In plain words: even **before** costs the timer captures only "
            f"**{R['gross_ret']:.0f}%** against buy & hold's **{R['bh_ret']:.0f}%** — being long "
            f"only **{R['exposure']:.0f}%** of the time already forgoes most of the compounding, "
            f"and **{R['n_trades']:,} position changes** over 21.5 years bleed real cost. At "
            f"10 bps one-way it's **{R['net10_ret']:+.1f}%** — a loss. And the block-shuffled "
            f"control, matched for the *same* exposure and turnover, beats the real rule in "
            f"**{R['rc_beat_ret_pct']}%** of 20 draws by total return and "
            f"**{R['rc_beat_sharpe_pct']}%** by Sharpe. **H₃ rejected; Tradability = MIRAGE, "
            "third axis = BUSTED.**"
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "Two-regime (trend/chop) Markov world, both regimes sharing the same daily shock "
            "scale, TUNABLE planted trend-persistence edge. The null (edge = 0) is checked over "
            "**20 seeds** — never a single stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    ohlc = data.synthetic_world(edge=0.0, seed=678 + s_)\n"
            "    null_ts.append(st.synthetic_detect(ohlc)['welch_t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "ohlc = data.synthetic_world(edge=0.006, seed=678)\n"
            "planted = st.synthetic_detect(ohlc)\n"
            "planted_t = planted['welch_t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (edge=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5,\n"
            "           label='planted trend-persistence edge')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Welch t (flag vs no-flag)')\n"
            "ax.set_title('Control: the null stays quiet, a planted edge lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and crosses the ±2 bar in "
            f"only {R['syn_null_fire']}/20 seeds — right at the ~5% nominal false-positive rate, "
            f"not biased. A generously sized planted trend-persistence edge reads "
            f"t = {R['syn_planted_t']:.2f} with a {R['syn_planted_hit']:.0f}% hit rate. The "
            "machinery works and can find a real edge when one exists — the real-tape null (and "
            "wrong-signed pooled result) is the genuine article, not a broken detector."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — RWI-high > 1 fires on {R['flag_freq']:.0f}% of SPY sessions "
            f"and does not predict a better next-session return: SPY alone runs backwards at "
            f"*t* = {R['welch_t']:+.2f}, and pooled across five instruments the wrong-signed gap "
            f"clears the bar (Welch *t* = {R['pooled_t']:+.2f}). A matched-count random-day "
            f"placebo beats the flag's own mean {R['placebo_p']*100:.0f}% of the time.\n"
            f"- **Tradability `MIRAGE`** — the long timer returns {R['gross_ret']:+.1f}% gross vs "
            f"buy & hold's {R['bh_ret']:+.1f}%, and turns net-negative ({R['net10_ret']:+.1f}%) "
            "at 10 bps one-way costs.\n"
            f"- **\"Beats a coin?\" `BUSTED`** — a block-shuffled, exposure-matched random-entry "
            f"control beats the real timer in {R['rc_beat_ret_pct']}% of draws by total return "
            f"and {R['rc_beat_sharpe_pct']}% by Sharpe. The rule loses to picking days at random."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson is about the whole trend-strength-gate family.** ADX, the "
            "Vertical-Horizontal-Filter, a rolling Hurst exponent and now the RWI all measure "
            "*something real* about recent price behavior — and none of them forecast what "
            "happens next. \"This move doesn't look random\" and \"this move will continue\" are "
            "different claims; the indicators only ever test the first.\n"
            "- **A genuinely different next test:** does the RWI carry information as a "
            "*volatility-regime* / position-sizing input (larger realized-vs-ATR displacement -> "
            "smaller size, expecting reversion) rather than a directional trigger? The pooled "
            "wrong-signed result here is a hint in that direction, not a tested claim.\n"
            "- **Dedup map:** [108-adx-filter](../../108-adx-filter/) (Wilder's ADX gate), "
            "[484-vertical-horizontal-filter](../../484-vertical-horizontal-filter/) (White's "
            "VHF gate), [397-hurst-regime](../../397-hurst-regime/) (rolling R/S Hurst switch) — "
            "four formulas, the same \"is this a real trend\" question, the same answer.\n\n"
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
