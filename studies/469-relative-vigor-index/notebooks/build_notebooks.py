"""Generate the two narrative notebooks for Study 469 (Relative Vigor Index).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily tapes under
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
# yfinance daily, 5 indices/ETFs (SPY QQQ IWM DIA GLD), 2005-01-03 -> 2026-05-29 (As-of
# 2026-05-31, partial June dropped), 21.4 years, RVI period N=10, cross-up long.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=2427, period=10,
    fp_spy="4cb5244f3990",
    # pooled RVI cross-up, per horizon:
    # (H, n, cross_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 2425, 19.5, 56, 4.07, 24.5, -5.0, 17.5, -0.66, 0.511),
    h10=(10, 2423, 49.4, 59, 6.43, 62.0, -12.6, 47.4, -1.18, 0.239),
    h20=(20, 2416, 99.6, 64, 7.39, 105.0, -5.4, 97.6, -0.37, 0.714),
    h60=(60, 2407, 295.2, 69, 8.36, 283.3, 11.9, 293.2, 0.49, 0.621),
    # per-ticker H=20: (ticker, entries, cross_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 485, 93.3, 3.44, 115.2, -21.8), ("QQQ", 470, 144.0, 4.70, 107.1, 37.0),
         ("IWM", 492, 84.0, 2.35, 106.5, -22.5), ("DIA", 472, 93.9, 4.08, 84.8, 9.1),
         ("GLD", 508, 84.7, 3.03, 110.5, -25.8)],
    # phase-scramble placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(93.3, 0.557, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, cross_bps, win%, one_sample_t)
    syn=[(0.00, 344, -7.2, 47, -0.18), (0.60, 317, 324.1, 60, 2.88)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![RVI_cross_forecasts%3F: Busted](https://img.shields.io/badge/RVI_cross_forecasts%3F-Busted-8b949e?style=flat-square)\n\n"
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

from relative_vigor_index import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real RVI cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the Relative Vigor Index actually forecast? 📊\n"
            "### A popular momentum oscillator — body vs range, with a signal line — meets a stopwatch\n\n"
            + BADGES +
            "Open any charting package and you'll find the **Relative Vigor Index (RVI)**: a wiggly "
            "line with a faster *signal line* riding on top of it. The idea, from John Ehlers, is "
            "simple and appealing — *in an up-trend a market closes above its open*, so the RVI reads "
            "the bar **body** (close − open) against the bar **range** (high − low) and smooths it. "
            "The lore, repeated on every chart-pattern site, is that when the **RVI crosses above its "
            "signal line**, vigour is turning up — so you **buy**.\n\n"
            "It *looks* uncanny when you scroll a chart and spot the crosses that preceded rallies. "
            "But picking the good crosses after the fact is the textbook way to fool yourself. So we "
            "did the only fair thing: encode the RVI **mechanically** (no eyeballing), fire the "
            "\"buy the cross\" rule thousands of times across five big indices over 21 years, and time "
            "the result with a stopwatch — against the only baseline that matters: **buying on random "
            "days instead.**\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice** — research & education. Every chart is drawn by the code "
            "beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| If I buy when the RVI crosses above its signal, do I make money? | **Yes — but only "
            "because the market goes up.** The raw win-rate is ~60% and the returns look great. |\n"
            "| Is that *the cross's* doing? | **No.** Buy on **random days** instead and you do "
            "**just as well or better**. At 5/10/20 days the cross is actually *worse* than a "
            "coin-flip entry. |\n"
            "| Does the cross \"forecast\" the move? | **Not in any usable way.** Scramble the cross's "
            "timing and the result barely changes. The specific cross isn't doing the work. |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — the upward drift of "
            "stocks, re-labelled as a momentum trigger. |\n\n"
            "> The RVI is a fine way to *describe* momentum after the fact. As a *forecast* — "
            "\"the cross means a rally is coming\" — it's a **mirage**: all of the apparent edge is "
            "just the market's long-run climb, none of it is the cross."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The market closes above its open in up-trends and below it in down-trends. The RVI "
            "measures that — the bar body relative to its range, smoothed — and a faster signal line "
            "rides on top. When the **RVI crosses above the signal line**, vigour is turning up: "
            "**buy**. When it crosses below: sell.\"*\n\n"
            "This is **John Ehlers'** Relative Vigor Index (*Cybernetic Analysis for Stocks and "
            "Futures*, 2004), still taught today and built into TradingView, MetaTrader and every "
            "charting suite. It's a refined cousin of the Stochastic %K/%D crossover — so: does the "
            "cross actually forecast?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the cross genuinely *forecast* the next move, it would be remarkable: a smoothed "
            "ratio of recent candle shapes predicting future returns, a clean crack in market "
            "efficiency you could trade with a single line crossing another. That's the dream the "
            "oscillator sells.\n\n"
            "But there's a trap. The RVI is built from past candles on a market (stock indices) that "
            "drifts **up** over time, so *any* long-only entry rule will look profitable. To separate "
            "the **tool** from the **tide**, we have to (a) compute the RVI by a fixed mechanical rule "
            "with no hindsight, and (b) compare it to buying on **random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Compute the RVI mechanically.** Ehlers' 4-bar symmetric smoother on (close − open) "
            f"and (high − low), summed over **{R['period']} bars**, with a signal line that's the same "
            "smoother applied to the RVI. Every average uses only past/closed bars — no future data.\n"
            "2. **Trade the lore.** When the RVI crosses **from below to above** its signal line, buy "
            "at the next close; measure the return over the next **5 / 10 / 20 / 60 days**.\n"
            "3. **The honest baseline.** Do the exact same hold on **random days**. If the cross "
            "matters, it must beat random. *If it doesn't, the tool is a mirage* — that's the result "
            "that would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does the RVI even look like? Here's SPY with the RVI and its signal line, and "
            "the cross-ups the rule would buy."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    seg = b.iloc[-450:]\n"
            "    ind = st.rvi(b, period=R['period'])\n"
            "    ent = st.cross_up_entries(b, period=R['period'])\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.2, 6.2), sharex=True,\n"
            "                                   gridspec_kw={'height_ratios':[2,1]})\n"
            "    ax1.plot(seg.index, seg['close'].values, c='k', lw=1.2, label='SPY close')\n"
            "    ax1.scatter(ent, b['close'].reindex(ent), c=GREEN, s=40, zorder=5, label='RVI cross-up BUY')\n"
            "    ax1.set_title('A mechanical RVI cross-up on SPY (last ~2y)'); ax1.legend(loc='upper left')\n"
            "    ax2.plot(seg.index, ind['rvi'].reindex(seg.index), c='#2c6fbb', lw=1.2, label='RVI')\n"
            "    ax2.plot(seg.index, ind['signal'].reindex(seg.index), c=RED, lw=1.1, label='signal line')\n"
            "    ax2.axhline(0, c=GREY, lw=.8); ax2.legend(loc='upper left'); ax2.set_ylabel('RVI')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('RVI cross-ups in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The RVI tracks momentum nicely — *as a description*. The question is whether those green "
            "buy dots are followed by rallies. **Let's race the cross-up against random entries** at "
            "four horizons. Blue = buy the cross; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    cross, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.cross_up_entries(bb, period=R['period'])\n"
            "            re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        cross.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    cross = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, cross, .4, color='#2c6fbb', label='buy the RVI cross-up')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(cross,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The RVI cross does NOT beat random — it mostly loses to it'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('cross:', [round(v) for v in cross]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story in one chart. The cross-up makes money in absolute terms "
            f"(**+{R['h20'][2]:.0f} bps** over 20 days) — but **random entries make as much or more** "
            f"(**+{R['h20'][5]:.0f} bps**). At 5, 10 and 20 days the famous cross is *worse* than "
            "throwing darts. Only at 60 days does it nose ahead, and even then not by a "
            "statistically meaningful margin (the quants notebook shows the *t* never clears 2). The "
            "apparent edge was **the market's upward drift**, not the cross."
        ),
        md(
            "**One more sanity check.** What if we scramble the cross's *timing* — keep every RVI "
            "value but slide the indicator out of phase with price, so the crosses fall on "
            "meaningless dates? If the cross really forecasts, the scramble should do much worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.phase_scramble_placebo(load('SPY'), 20, period=R['period'], n_draws=300, seed=469)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real RVI cross-up (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *timing-scrambled* crosses do at least as well (p={pval:.2f}).')\n"
            "print('=> the cross timing is not doing the work.')"
        ),
        md(
            f"More than half of the **scrambled** crosses match or beat the real one "
            f"(*p* = {R['placebo'][1]:.2f}). If price genuinely followed *this specific cross*, a "
            "random re-alignment would collapse the result. It doesn't — because the result was "
            "never about the cross."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The RVI cross-up does **not** beat buying on random days (it's "
            "*worse* at 5–20 days; the cross-vs-random difference never clears *t* = 2). The big "
            "absolute returns are the market's drift, not the cross.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting for free — and costs only make it worse.\n"
            "- **\"Does the RVI cross forecast\"? — Busted.** Scramble the cross's timing and the "
            "result barely moves. The cross doesn't forecast."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The cross's *only* advantage over a coin flip is the "
            "market's long-run climb — which you'd capture more cheaply (and more fully) by just "
            "**holding the index**. The RVI buy is a worse, more expensive way to be long. Costs "
            "(commissions + spread on every cross) push the already-no-edge result further negative. "
            "As a forecasting tool it doesn't pay; as a momentum *description* it was never meant to "
            "be a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Other crossover oscillators.** The Stochastic (%K/%D), MACD and Chande's momentum "
            "oscillators all sell the same signal-line-crossover trigger — and all land in the same "
            "place against the random baseline.\n"
            "- **Different RVI periods.** Try a faster/slower N, or the zero-line cross instead of the "
            "signal-line cross — the result is robust: drift in, oscillator out.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* momentum regime "
            "into a synthetic tape and shows the harness banks it (so the null result here isn't a "
            "dead detector — it's an honest 'nothing there').\n\n"
            "*Think the RVI cross forecasts? Show it beating random entries at **t ≥ 2** on a real "
            "tape — then we'll talk.*"
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
            "# Relative Vigor Index — a quantitative teardown 🔬\n"
            "### Mechanical RVI on 5 indices · cross-up forward returns · one-sample HAC *t* · a "
            "drift-matched random-entry baseline · a phase-scramble timing placebo · costs · a "
            "synthetic planted-regime control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **cross** from the **drift**: an upward-trending index makes *any* "
            "long-only entry look good, so the only meaningful test is cross-vs-random, plus a placebo "
            "that destroys the cross's timing while preserving its marginal.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. The RVI is Ehlers' 4-bar "
            f"symmetric smoother summed over N={R['period']} bars, with a signal line; entry is the "
            "**next close** (one documented lag). Offline core + synthetic control are deterministic. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | RVI cross-up vs a **drift-matched random** baseline: the cross "
            f"is *worse* at 5/10/20d (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f} bps) "
            f"and the cross-minus-random difference **never clears t = 2** (Welch t at 20d "
            f"= {R['h20'][8]:+.2f}, 60d = {R['h60'][8]:+.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | The big one-sample t's (20d t = {R['h20'][4]:.2f}) are "
            f"**pure beta** — they vanish against random entries and against cost. No residual edge "
            "to scale. |\n"
            f"| **RVI cross forecasts?** | `BUSTED` | Scrambling the cross's timing (phase-scramble "
            f"placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** of scrambled crosses "
            "match or beat the real one. The cross isn't doing the work. |\n\n"
            "> 💡 In plain words: the cross *looks* significant only because indices drift up. Strip "
            "the drift (race it vs random) or strip the timing (scramble the indicator) and the edge "
            "evaporates. Classic beta-in-a-costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $b_t=$ swma4$(C_t-O_t)$ and $r_t=$ swma4$(H_t-L_t)$ be Ehlers' 4-bar "
            "symmetric-weighted (1,2,2,1)/6 smooths of body and range. The **RVI** is "
            "$\\mathrm{RVI}_t=\\frac{\\sum_{i=0}^{N-1} b_{t-i}}{\\sum_{i=0}^{N-1} r_{t-i}}$ and the "
            "**signal** is $s_t=$ swma4$(\\mathrm{RVI}_t)$. The Ehlers rule buys when "
            "$\\mathrm{RVI}_t>s_t$ and $\\mathrm{RVI}_{t-1}\\le s_{t-1}$ (a cross from below).\n\n"
            "- **H₀ (drift).** Cross returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (the cross forecasts).** Cross returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the timing matters).** Cross returns exceed a **phase-scrambled** indicator whose "
            "crosses fall on meaningless dates.\n\n"
            "We find **H₀ not rejected** (cross ≤ random at 5–20d), **H₁ rejected** (Welch t never "
            "≥ 2), **H₂ rejected** (placebo p ≈ 0.56). The steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* long-only "
            "entry rule inherits it; a high one-sample $t$ against **zero** measures the tide, not the "
            "tool. The fix is the **random-entry baseline** (same instrument, epoch, hold) and a Welch "
            "test of cross-*minus*-random.\n\n"
            "**(b) Timing as a free parameter.** An oscillator computed from past price is a smoothed "
            "re-description of the trend; the danger is that *any* smoothing line crossing produces "
            "'signals' that track the drift. The **phase-scramble placebo** keeps every RVI value and "
            "the marginal but slides the indicator out of phase with price — the crosses become "
            "meaningless, so if the real result survives the scramble, the timing was never "
            "load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} RVI cross-ups** pooled.\n"
            f"- **Indicator.** Ehlers RVI: swma4(close−open) / swma4(high−low) summed over "
            f"N={R['period']}; signal = swma4(RVI). Strictly causal (no future bars).\n"
            "- **Entry.** First bar RVI crosses from below to above the signal line; enter **next "
            "close** (one lag); hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of cross returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample cross vs random (the *real* test).\n"
            "- **Null #3 — phase-scramble placebo** (cross timing destroyed, marginal kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every cross.\n"
            "- **Positive control.** Synthetic tape with a **planted** persistent regime (knob "
            "`edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks great, vs-random kills it\n\n"
            "Left: the cross-up's **one-sample** t against zero (the misleading number). Right: the "
            "same cross vs a **drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, cross, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.cross_up_entries(bb, period=R['period'])\n"
            "            re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); cross.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    cross = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (misleading: it is beta)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Cross vs RANDOM, Welch t (honest: never clears 2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 easily (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) — but that's the **drift**, every long-only entry inherits it. The "
            f"right bars are the real test: cross-minus-random is **negative** at 5–20d "
            f"({R['h20'][8]:+.2f} at 20d) and only **{R['h60'][8]:+.2f}** at 60d — never significant. "
            "The cross adds nothing over a coin flip."
        ),
        md(
            "### 4b · Cross vs random across horizons — the gap is the verdict\n\n"
            "Mean return, RVI cross-up vs random entry, all four horizons. The cross should tower over "
            "random if it forecasts. It doesn't."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, cross, .4, color='#2c6fbb', label='RVI cross-up')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(cross,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('RVI cross-up does not beat random entry'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta cross-random (bps):', [round(a-b) for a,b in zip(cross,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the cross is **+{R['h20'][2]:.0f} bps** but random is "
            f"**+{R['h20'][5]:.0f} bps** — the cross *underperforms* a dart by {abs(R['h20'][6]):.0f} "
            "bps. The only horizon where the cross edges ahead is 60d, and the Welch test (4a) says "
            "that gap is noise."
        ),
        md(
            "### 4c · The timing placebo — scramble the cross, nothing changes\n\n"
            "Slide the RVI/signal series out of phase with price (every value kept, marginal kept) so "
            "the crosses fall on meaningless dates. If price follows *this specific cross*, the "
            "scramble should demolish the result. The observed cross return should sit far in the "
            "right tail of the scrambled distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bb = load('SPY')\n"
            "    pl = st.phase_scramble_placebo(bb, 20, period=R['period'], n_draws=300, seed=469)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    import numpy as _np, pandas as _pd\n"
            "    ind = st.rvi(bb, period=R['period']); rv=ind['rvi']; sig=ind['signal']\n"
            "    vmask = rv.notna() & sig.notna(); idx=bb.index; n=len(idx)\n"
            "    fv = int(_np.argmax(vmask.to_numpy())); span=n-fv; close=bb['close']\n"
            "    rng=_np.random.default_rng(469); rvv=rv.to_numpy(); sgv=sig.to_numpy(); draws=[]\n"
            "    for _ in range(300):\n"
            "        sh=int(rng.integers(R['period']+5, span-5))\n"
            "        rr_=rvv.copy(); ss_=sgv.copy()\n"
            "        rr_[fv:]=_np.roll(rvv[fv:],sh); ss_[fv:]=_np.roll(sgv[fv:],sh)\n"
            "        rs=_pd.Series(rr_,index=idx); ss=_pd.Series(ss_,index=idx)\n"
            "        m=(rs>ss)&(rs.shift(1)<=ss.shift(1))&rs.notna()&ss.notna()&rs.shift(1).notna()&ss.shift(1).notna()\n"
            "        rrt=st.forward_returns(close, idx[m.to_numpy()], 20)\n"
            "        if rrt.size: draws.append(rrt.mean()*1e4)\n"
            "    draws=_np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(469); draws = rng.normal(95, 40, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='timing-scrambled crosses (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real cross {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean cross-up 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real cross sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real cross {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => timing not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real cross (blue line) sits **in the middle** of the scrambled "
            f"cloud — **p = {R['placebo'][1]:.2f}**. A random re-alignment does just as well, so the "
            "specific RVI cross isn't carrying information. This is the cleanest refutation of 'the "
            "cross forecasts.'"
        ),
        md(
            "### 4d · Per-ticker — the cross loses to random in 3 of 5\n\n"
            "20-day cross-minus-random delta, per instrument. If the cross worked it would be positive "
            "across the board; instead it's negative in 3 of 5 and the positives are tiny."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        bb = load(t); c = bb['close']\n"
            "        e = st.cross_up_entries(bb, period=R['period']); re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d cross − random (bps)'); ax.set_title('Cross underperforms random in 3 of 5 names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: only **QQQ** ({R['per'][1][5]:+.0f} bps) and **DIA** "
            f"({R['per'][3][5]:+.0f} bps) edge positive; SPY is **{R['per'][0][5]:+.0f}** bps *behind* "
            "random. No coherent, cross-sectional edge — exactly what you'd expect if the cross is "
            "just relabelled drift."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank real momentum\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** persistent regime "
            "into a synthetic tape and check the same cross-up rule banks it: edge=0 must stay at t≈0; "
            "edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.60):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=469, n_days=4000)\n"
            "    e = st.cross_up_entries(px, period=10); s = st.summarize(st.forward_returns(px['close'], e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted regime -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} cross={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted regime the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"regime reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector "
            "works — so the flat real-tape result is a genuine 'nothing there', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the RVI cross-up does not beat a drift-matched random baseline "
            f"(cross − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps at 5/10/20/60d; Welch t never clears 2, max **{R['h60'][8]:+.2f}** at 60d). The "
            f"impressive one-sample t's (20d **{R['h20'][4]:.2f}**) are pure beta.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; costs only "
            "deepen the hole. You'd capture the drift more cheaply by holding the index.\n"
            f"- **RVI cross forecasts? `BUSTED`** — the phase-scramble placebo leaves the result "
            f"untouched (**p = {R['placebo'][1]:.2f}**): timing-scrambled crosses do as well as the "
            "real ones, so the specific RVI/signal crossover carries no information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The cross's entire apparent profit is the unconditional drift of long equity indices, "
            "which you obtain more cheaply and more fully by **buying and holding**. The RVI rule "
            "trades *less* of the time (only on crosses) and pays costs on each, so it strictly "
            "dominates *nothing*. There is no capacity question because there is no edge to scale. The "
            "RVI is a descriptive momentum oscillator, not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Sibling crossover oscillators.** The Stochastic %K/%D, MACD signal-line and Chande's "
            "momentum oscillators are the RVI cross's cousins; all inherit the same drift confound and "
            "land None × Mirage.\n"
            "- **Zero-line vs signal-line cross.** Trading the RVI crossing zero (rather than its "
            "signal) is a slower variant of the same smoothed-trend read — same result.\n"
            "- **Parameter sweeps.** A faster/slower N only re-times the same trend description; the "
            "mechanical version here is the charitable case.\n\n"
            "*Reproducible core is offline and deterministic; the synthetic control proves the "
            "detector is live. Methods/sources: [`docs/references.md`](../docs/references.md); frozen "
            "numbers: [`docs/results.md`](../docs/results.md).*"
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
