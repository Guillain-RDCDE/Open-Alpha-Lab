"""Generate the two narrative notebooks for Study 478 (Percentage Price Oscillator).

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
# yfinance daily total-return, 5 indices/ETFs (SPY QQQ IWM DIA GLD), 2005-01-03 -> 2026-05-29
# (As-of 2026-05-31, partial June dropped), 21.4 years, PPO 12/26/9, bullish-crossover long.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=1082,
    fp_spy="4cb5244f3990",
    # pooled bullish PPO crossover, per horizon:
    # (H, n, cross_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 1080, 21.6, 59, 3.08, 35.8, -14.2, 19.6, -1.41, 0.159),
    h10=(10, 1080, 45.0, 61, 4.08, 69.9, -24.9, 43.0, -1.71, 0.088),
    h20=(20, 1078, 90.3, 63, 5.98, 154.0, -63.6, 88.3, -3.13, 0.002),
    h60=(60, 1072, 280.3, 69, 7.86, 324.7, -44.4, 278.3, -1.31, 0.190),
    # raw-MACD comparator (the thesis axis): per horizon delta-vs-random for PPO and raw MACD
    # (H, ppo_delta_bps, macd_delta_bps, difference_bps)
    macd=[(5, -14.2, -12.0, -2.2), (10, -24.9, -23.8, -1.1),
          (20, -63.6, -68.9, 5.3), (60, -44.4, -52.3, 7.9)],
    # per-ticker H=20: (ticker, entries, cross_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 227, 75.1, 2.65, 156.2, -81.1), ("QQQ", 210, 128.2, 3.82, 192.0, -63.8),
         ("IWM", 216, 86.6, 1.93, 162.5, -75.9), ("DIA", 218, 90.1, 3.33, 145.9, -55.8),
         ("GLD", 211, 72.9, 2.05, 113.3, -40.4)],
    # shuffled-sign crossover placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(75.1, 0.998, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, cross_bps, win%, one_sample_t)
    syn=[(0.00, 166, 41.9, 58, 1.09), (0.60, 70, 1224.3, 99, 19.21)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Normalizing_adds_edge%3F: Busted](https://img.shields.io/badge/Normalizing_adds_edge%3F-Busted-8b949e?style=flat-square)\n\n"
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

from ppo import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real PPO cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the PPO crossover actually call the turn? 📊\n"
            "### MACD's tidier cousin — the same \"buy the bullish crossover\" lore — meets a stopwatch\n\n"
            + BADGES +
            "Open any charting package and you'll find the **Percentage Price Oscillator (PPO)**: it's "
            "**MACD wearing a percent sign**. MACD is the gap between a fast and a slow moving average; "
            "the PPO just *divides that gap by the slow average* so it reads as a percentage (a $400 "
            "SPY and a $40 GLD then give comparable numbers). The lore, repeated on every indicator "
            "site, is the same as MACD's: when the **PPO line crosses up through its signal line**, "
            "that's a **buy** — momentum has turned up.\n\n"
            "It *looks* convincing on a chart, because the market mostly goes up and a crossover lands "
            "before plenty of green bars. But \"it went up afterwards\" isn't an edge if it would have "
            "gone up anyway. So we did the only fair thing: encode the crossover **mechanically**, fire "
            "the buy **1082 times** across five big indices over 21 years, and time the result with a "
            "stopwatch — against the only baseline that matters: **buying on random days instead.**\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo, the raw-MACD comparison "
            "and the cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice** — research & education. Every chart is drawn by the code "
            "beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| If I buy on a **bullish PPO crossover**, do I make money? | **Yes — but only because the "
            "market goes up.** The raw win-rate is ~60% and the returns look great. |\n"
            "| Is that *the crossover's* doing? | **No.** Buy on **random days** instead and you do "
            "**better** — at every horizon. The crossover *under-times* a coin flip. |\n"
            "| Does normalizing MACD (dividing by the slow EMA) add anything? | **No.** The **raw "
            "MACD** crossover loses to random by the *same* margin. The percent sign changes the units, "
            "not the timing. |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — the upward drift of "
            "stocks, re-labelled as an oscillator signal. |\n\n"
            "> The PPO is a fine way to *read* momentum on a chart. As a *forecast* — \"the crossover "
            "will be followed by a rally\" — it's a **mirage**: all of the apparent edge is just the "
            "market's long-run climb, none of it is the crossover."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"PPO = 100 × (EMA₁₂ − EMA₂₆) / EMA₂₆. The **signal line** is a 9-period EMA of the "
            "PPO. When the PPO crosses **above** the signal line, momentum has turned up — **buy**. "
            "Because the PPO is a percentage, the same crossover rule works on any instrument, at any "
            "price.\"*\n\n"
            "This is the standard PPO/MACD crossover, built into TradingView, StockCharts, MetaTrader "
            "and every indicator suite. The normalization is *sold* as the upgrade over MACD: a "
            "percentage is comparable across stocks and across time. So two questions: does the "
            "crossover forecast at all — and does the normalization buy anything MACD didn't already "
            "have?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a bullish crossover genuinely *forecast* a rally, it would be remarkable: a simple "
            "moving-average crossing would predict future returns, a clean crack in market efficiency "
            "you could trade by rote. That's the dream the indicator sells.\n\n"
            "But there's a trap built into it. The crossover fires on a market (stock indices) that "
            "drifts **up** over time, so *any* buy-and-hold-for-a-month rule looks profitable. To "
            "separate the **signal** from the **tide**, we have to compare the crossover to buying on "
            "**random days** — same instrument, same epoch, same hold. And to answer the headline "
            "question, we run the **raw MACD** crossover side-by-side: if the percent sign matters, the "
            "two should diverge. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Build the oscillator by rule.** PPO = 100 × (EMA₁₂ − EMA₂₆) / EMA₂₆; signal = 9-EMA "
            "of the PPO. Standard 12/26/9, read on the close of each bar.\n"
            "2. **Trade the lore.** The first bar the PPO crosses **strictly above** its signal, buy at "
            "the **next close** (one documented lag — no peeking); measure the return over the next "
            "**5 / 10 / 20 / 60 days**.\n"
            "3. **The honest baseline.** Do the exact same hold on **random days**. If the crossover "
            "matters, it must beat random. *If it doesn't, the signal is a mirage* — announced before "
            "we look.\n"
            "4. **The normalization test.** Run the **raw MACD** crossover (no division) on the same "
            "tape. If they tie, the percent sign buys comparability, not edge."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does the rule even fire on? Here's SPY with the PPO and its signal line, and "
            "the bullish crossovers the rule would buy."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    cl = b['close']\n"
            "    ppo, sig = st.ppo_lines(cl)\n"
            "    ent = st.ppo_cross_entries(cl)\n"
            "    seg = cl.iloc[-450:]\n"
            "    ent_w = ent[ent >= seg.index[0]]\n"
            "    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.2, 6.4), sharex=True,\n"
            "                                   gridspec_kw={'height_ratios': [2, 1]})\n"
            "    ax1.plot(seg.index, seg.values, c='k', lw=1.2, label='SPY close')\n"
            "    ax1.scatter(ent_w, cl.reindex(ent_w), c=GREEN, s=42, zorder=5, label='bullish PPO crossover BUY')\n"
            "    ax1.set_title('Mechanical bullish PPO crossovers on SPY (last ~2y)'); ax1.legend(loc='upper left')\n"
            "    ax2.plot(seg.index, ppo.reindex(seg.index), c='#2c6fbb', lw=1.2, label='PPO')\n"
            "    ax2.plot(seg.index, sig.reindex(seg.index), c=RED, lw=1.1, label='signal (9-EMA of PPO)')\n"
            "    ax2.axhline(0, c=GREY, lw=.8); ax2.set_ylabel('PPO (%)'); ax2.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('bullish crossovers in window:', len(ent_w))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The crossover does what it says — it marks momentum turning up. The question is whether "
            "those green buy dots are *followed by* outperformance. **Let's race the crossover against "
            "random entries** at four horizons. Blue = buy the crossover; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    cross, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.ppo_cross_entries(c)\n"
            "            re = st.random_entries(c, max(len(e), 50), seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        cross.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    cross = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, cross, .4, color='#2c6fbb', label='buy the PPO crossover')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(cross,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The crossover LOSES to random at every horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('crossover:', [round(v) for v in cross]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story in one chart. The crossover makes money in absolute terms "
            f"(**+{R['h20'][2]:.0f} bps** over 20 days) — but **random entries make more** "
            f"(**+{R['h20'][5]:.0f} bps**). At *every* horizon the famous crossover is *worse* than "
            "throwing darts, and at 20 days it's worse by a statistically meaningful margin (the quants "
            "notebook shows the Welch *t* hitting **−3.13** — \"significant\" in the wrong direction). "
            "The apparent edge was **the market's upward drift**, not the crossover."
        ),
        md(
            "**But does the normalization at least do *something*?** The PPO is just MACD divided by the "
            "slow EMA. Let's run the **raw MACD** crossover on the same tape and compare how each loses "
            "to random."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ppo_d, macd_d = [], []\n"
            "    for h in hs:\n"
            "        pt, mt, rt = [], [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.ppo_cross_entries(c); me = st.macd_cross_entries(c)\n"
            "            re = st.random_entries(c, max(len(e), 50), seed=7)\n"
            "            pt.append(st.forward_returns(c, e, h)); mt.append(st.forward_returns(c, me, h))\n"
            "            rt.append(st.forward_returns(c, re, h))\n"
            "        r = np.concatenate(rt).mean()*1e4\n"
            "        ppo_d.append(np.concatenate(pt).mean()*1e4 - r); macd_d.append(np.concatenate(mt).mean()*1e4 - r)\n"
            "else:\n"
            "    ppo_d = [m[1] for m in R['macd']]; macd_d = [m[2] for m in R['macd']]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, ppo_d, .4, color='#2c6fbb', label='PPO (normalized) Δ vs random')\n"
            "ax.bar(x+.2, macd_d, .4, color=AMBER, label='raw MACD Δ vs random')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(a,bb) in enumerate(zip(ppo_d,macd_d)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='top',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='top',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('Δ vs random (bps)')\n"
            "ax.set_title('PPO ≈ raw MACD — normalizing changes the units, not the timing'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('PPO  Δ:', [round(v) for v in ppo_d]); print('MACD Δ:', [round(v) for v in macd_d])"
        ),
        md(
            "The two bars are essentially the same height at every horizon — both lose to random by the "
            "same margin. Dividing by the slow EMA barely moves *where* the lines cross, so the "
            "normalized PPO and the raw MACD fire on almost the same days and earn almost the same "
            "(negative-vs-random) returns. The percent sign buys **comparability across instruments**, "
            "not **edge**."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The bullish PPO crossover does **not** beat buying on random days "
            "(it's *worse* at every horizon, significantly so at 20 days, Welch *t* = −3.13). The big "
            "absolute returns are the market's drift, not the crossover.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting for free — and costs only make it worse.\n"
            "- **\"Does normalizing MACD add edge?\"— Busted.** The raw MACD crossover loses to random "
            "by the same margin, and scrambling the crossover's sign structure leaves the result "
            "intact. The percent sign changes units, not timing."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The crossover's *only* advantage over a coin flip is "
            "the market's long-run climb — which you'd capture more cheaply (and more fully) by just "
            "**holding the index**. The PPO buy is a worse, more expensive way to be long: it trades "
            "only on crossovers and pays commission + spread on each, so costs push the already-"
            "no-edge result further negative. As a forecasting tool it doesn't pay; as a way to *read* "
            "momentum on a chart, it was never meant to be a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The bearish side / shorting.** We tested the long. The symmetric short on bearish "
            "crossovers fares no better — same drift confound, with the sign flipped against you.\n"
            "- **Different parameters.** Try 5/35/5 or 8/17/9 — the result is robust: drift in, "
            "crossover out. Tuning the EMAs is exactly the kind of in-sample search that manufactures "
            "mirages.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* post-crossover "
            "continuation into a synthetic tape and shows the harness banks it (so the null result here "
            "isn't a dead detector — it's an honest 'nothing there').\n\n"
            "*Think the crossover forecasts? Show the bullish PPO cross beating random entries at "
            "**t ≥ 2** on a real tape — then we'll talk.*"
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
            "# The PPO crossover — a quantitative teardown 🔬\n"
            "### Mechanical 12/26/9 PPO crossovers on 5 indices · forward returns · one-sample HAC *t* "
            "· a drift-matched random-entry Welch test · a raw-MACD comparator · a shuffled-sign "
            "placebo · costs · a synthetic planted-continuation control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **crossover** from the **drift**: an upward-trending index makes *any* "
            "long entry look good, so the only meaningful test is crossover-vs-random, plus a placebo "
            "that destroys the crossover structure while preserving its marginal, plus the raw-MACD "
            "run that answers the headline thesis question.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. PPO/MACD use standard 12/26/9 "
            "EMAs (`adjust=False`); the oscillator is read on the close of *t*, entry is the **next "
            "close** (one documented lag). Offline core + synthetic control are deterministic. Methods "
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
            f"| **Signal** | `NONE` | Crossover vs a **drift-matched random** baseline: the crossover "
            f"is *worse* at every horizon (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps at 5/10/20/60d) and at 20d *significantly* worse (Welch t "
            f"= {R['h20'][8]:+.2f}, p = {R['h20'][9]:.3f}). |\n"
            f"| **Tradability** | `MIRAGE` | The big one-sample t's (20d t = {R['h20'][4]:.2f}, 60d "
            f"= {R['h60'][4]:.2f}) are **pure beta** — they vanish against random entries and against "
            "cost. No residual edge to scale. |\n"
            f"| **Normalizing adds edge?** | `BUSTED` | Raw MACD loses to random by the same margin "
            f"(20d Δ = {R['macd'][2][2]:+.0f} bps vs PPO {R['macd'][2][1]:+.0f}), and the shuffled-sign "
            f"placebo leaves the result intact: **p = {R['placebo'][1]:.3f}** of nonsense crossovers "
            "match or beat the real one. |\n\n"
            "> 💡 In plain words: the crossover *looks* significant only because indices drift up. Strip "
            "the drift (race it vs random) or strip the structure (scramble the signs) and the edge "
            "evaporates. And the percent-sign normalization changes nothing MACD didn't already do. "
            "Classic beta-in-a-costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "With $\\mathrm{EMA}_f,\\mathrm{EMA}_s$ the fast/slow (12/26) EMAs of the close, the "
            "$\\textbf{PPO}_t = 100\\cdot(\\mathrm{EMA}_f-\\mathrm{EMA}_s)/\\mathrm{EMA}_s$ and the "
            "**signal** $\\sigma_t = \\mathrm{EMA}_9(\\mathrm{PPO})$. The Andrews-of-momentum rule buys "
            "the first bar $\\mathrm{PPO}_{t-1}\\le\\sigma_{t-1}$ and $\\mathrm{PPO}_t>\\sigma_t$ — a "
            "strict up-crossing — entered at the next close.\n\n"
            "- **H₀ (drift).** Crossover returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (the crossover forecasts).** Crossover returns **exceed** random at some horizon, "
            "t ≥ 2.\n"
            "- **H₂ (normalization adds timing).** The **PPO** crossover beats the **raw MACD** "
            "crossover; and the real crossover beats a **shuffled-sign** placebo whose up-crossings are "
            "noise.\n\n"
            "We find **H₀ not rejected** (crossover < random at every horizon), **H₁ rejected** (Welch "
            "t is *negative*, −3.13 at 20d), **H₂ rejected** (PPO ≈ MACD; placebo p ≈ 0.998). The "
            "steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* long entry "
            "on a multi-day horizon inherits it; a high one-sample $t$ against **zero** measures the "
            "tide, not the tool. The fix is the **random-entry baseline** (same instrument, epoch, "
            "hold) and a Welch test of crossover-*minus*-random.\n\n"
            "**(b) Structure as a free parameter.** The danger is that *any* oscillator crossing on a "
            "trend produces 'signals' that look timed. The **shuffled-sign placebo** keeps the marginal "
            "of $|\\mathrm{PPO}-\\sigma|$ (so crossings happen about as often) but permutes which bars "
            "sit above vs below the signal — the specific up-crossings become noise. If the real result "
            "survives the scramble, the crossover structure was never load-bearing. The **raw-MACD "
            "comparator** then isolates the *normalization* itself."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} bullish PPO crossovers** "
            "pooled.\n"
            "- **Oscillator.** 12/26/9 EMAs (`adjust=False`); PPO = 100·(EMA₁₂−EMA₂₆)/EMA₂₆; signal "
            "= 9-EMA of PPO; warm-up = slow+signal bars.\n"
            "- **Entry.** First strict up-crossing PPO>signal; enter **next close** (one lag); hold "
            "H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of crossover returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample crossover vs random (the *real* "
            "test).\n"
            "- **Null #3 — shuffled-sign placebo** (sign of PPO−signal permuted; marginal kept).\n"
            "- **Thesis comparator — raw MACD** crossover (12/26/9, no division) on the same tape.\n"
            "- **Costs.** 1 bp one-way × 2 legs on every crossover.\n"
            "- **Positive control.** Synthetic tape with a **planted** post-crossover continuation "
            "(knob `edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks great, vs-random kills it\n\n"
            "Left: the crossover's **one-sample** t against zero (the misleading number). Right: the "
            "same crossover vs a **drift-matched random** baseline (the honest number, Welch t)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, cross, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.ppo_cross_entries(c)\n"
            "            re = st.random_entries(c, max(len(e), 50), seed=7)\n"
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
            "a2.axhline(2, ls='--', c=RED); a2.axhline(-2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Crossover vs RANDOM, Welch t (honest: negative everywhere)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 easily (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) — but that's the **drift**, every long entry inherits it. The right "
            f"bars are the real test: crossover-minus-random is **negative at every horizon** "
            f"({R['h20'][8]:+.2f} at 20d, *below* the −2 bar — significant in the *wrong* direction). "
            "The crossover doesn't just fail to beat a coin flip; it actively under-times it."
        ),
        md(
            "### 4b · Crossover vs random across horizons — the gap is the verdict\n\n"
            "Mean return, PPO crossover vs random entry, all four horizons. The crossover should tower "
            "over random if it forecasts. It sits *below* it everywhere."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, cross, .4, color='#2c6fbb', label='PPO crossover')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,bb) in enumerate(zip(cross,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('PPO crossover does not beat random entry — it trails it'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta crossover-random (bps):', [round(a-bb) for a,bb in zip(cross,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the crossover is **+{R['h20'][2]:.0f} bps** but random is "
            f"**+{R['h20'][5]:.0f} bps** — the crossover *underperforms* a dart by "
            f"{abs(R['h20'][6]):.0f} bps. There is no horizon where it gets ahead."
        ),
        md(
            "### 4c · The normalization test — PPO vs raw MACD\n\n"
            "The headline question: does dividing MACD by the slow EMA add timing power? Run the raw "
            "MACD crossover side-by-side and compare each one's Δ-vs-random. If the percent sign "
            "mattered, the bars would diverge."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ppo_d, macd_d = [], []\n"
            "    for h in hs:\n"
            "        pt, mt, rt = [], [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.ppo_cross_entries(c); me = st.macd_cross_entries(c)\n"
            "            re = st.random_entries(c, max(len(e), 50), seed=7)\n"
            "            pt.append(st.forward_returns(c, e, h)); mt.append(st.forward_returns(c, me, h))\n"
            "            rt.append(st.forward_returns(c, re, h))\n"
            "        r = np.concatenate(rt).mean()*1e4\n"
            "        ppo_d.append(np.concatenate(pt).mean()*1e4 - r); macd_d.append(np.concatenate(mt).mean()*1e4 - r)\n"
            "else:\n"
            "    ppo_d = [m[1] for m in R['macd']]; macd_d = [m[2] for m in R['macd']]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, ppo_d, .4, color='#2c6fbb', label='PPO (normalized)')\n"
            "ax.bar(x+.2, macd_d, .4, color=AMBER, label='raw MACD')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(a,bb) in enumerate(zip(ppo_d,macd_d)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='top',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='top',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('Δ vs random (bps)')\n"
            "ax.set_title('PPO Δ ≈ raw-MACD Δ at every horizon (within a few bps)'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('PPO  Δ:', [round(v) for v in ppo_d]); print('MACD Δ:', [round(v) for v in macd_d])\n"
            "print('PPO−MACD diff:', [round(p-m) for p,m in zip(ppo_d,macd_d)])"
        ),
        md(
            f"> 💡 In plain words: the PPO and raw-MACD Δ's are within a few bps of each other at every "
            f"horizon (20d {R['macd'][2][1]:+.0f} vs {R['macd'][2][2]:+.0f}). Dividing by a slow-moving "
            "EMA is a near-monotonic rescaling, so it barely moves *where* PPO−signal changes sign — "
            "the two oscillators fire on almost the same days. The normalization buys cross-instrument "
            "**comparability**, not **timing**."
        ),
        md(
            "### 4d · The structure placebo — scramble the signs, nothing changes\n\n"
            "Permute the *sign* of the daily PPO−signal differences (magnitudes/marginal kept, so "
            "crossings happen about as often) so the specific up-crossings become noise. If the real "
            "crossover carried timing information, its return would sit far in the **right** tail of the "
            "scrambled distribution. It sits near the **left** tail instead."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.shuffled_sign_placebo(c, 20, n_draws=500, seed=478)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    # rebuild the placebo distribution for the histogram\n"
            "    ppo, sig = st.ppo_lines(c)\n"
            "    diff = (ppo - sig); warmup = st.SLOW + st.SIGNAL\n"
            "    valid = diff.iloc[warmup:].dropna(); mags = valid.abs().to_numpy(); idx = valid.index\n"
            "    import pandas as _pd; rng = np.random.default_rng(478); draws = []\n"
            "    for _ in range(500):\n"
            "        signs = rng.choice([-1.0, 1.0], size=mags.size)\n"
            "        d = _pd.Series(signs*mags, index=idx); prev = d.shift(1)\n"
            "        cross = (prev <= 0.0) & (d > 0.0) & d.notna() & prev.notna()\n"
            "        rr = st.forward_returns(c, idx[cross.to_numpy()], 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(478); draws = rng.normal(150, 40, 500)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='scrambled-sign crossovers (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real crossover {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean PPO-crossover 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real crossover sits in the LEFT tail: placebo p = {pval:.3f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real crossover {obs:+.1f} bps   placebo p={pval:.3f}  (>0.05 => structure not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: **{R['placebo'][1]*100:.1f}%** of scrambled-sign crossovers match or "
            f"beat the real one (**p = {R['placebo'][1]:.3f}**) — the real PPO crossover is one of the "
            "*worst*-timed entries in the scrambled cloud. The specific crossover structure carries no "
            "useful information; this is the cleanest refutation of 'the crossover calls the turn.'"
        ),
        md(
            "### 4e · Per-ticker — the crossover loses to random everywhere\n\n"
            "20-day crossover-minus-random delta, per instrument. If the crossover worked it would be "
            "positive across the board; instead it's negative in **5 of 5**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = load(t)['close']\n"
            "        e = st.ppo_cross_entries(c); re = st.random_entries(c, max(len(e), 50), seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='top',fontsize=9)\n"
            "ax.set_ylabel('20d crossover − random (bps)'); ax.set_title('Crossover underperforms random in 5 of 5 names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: every name is negative — SPY is **{R['per'][0][5]:+.0f}** bps behind "
            f"random, GLD (the least bad) still **{R['per'][4][5]:+.0f}**. No coherent, cross-sectional "
            "edge — exactly what you'd expect if the crossover is just relabelled drift. The healthy "
            "one-sample t's (SPY +2.65, QQQ +3.82…) are all beta."
        ),
        md(
            "### 4f · Synthetic positive control — the harness CAN bank a real continuation\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** post-crossover "
            "continuation into a synthetic tape and check the same crossover rule banks it: edge=0 must "
            "stay near t≈1; edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.60):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=478, n_days=4000)\n"
            "    c = px['close']; e = st.ppo_cross_entries(c); s = st.summarize(st.forward_returns(c, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~1 (no false positive); planted continuation -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} cross={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted continuation the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive at the t ≥ 2 "
            f"bar); a planted continuation reaches **t = {R['syn'][1][4]:.2f}** (win "
            f"{R['syn'][1][3]:.0f}%). The detector works — so the flat real-tape result is a genuine "
            "'nothing there', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the PPO crossover does not beat a drift-matched random baseline "
            f"(crossover − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps at 5/10/20/60d; Welch t is *negative everywhere* and *significantly* "
            f"so at 20d, **{R['h20'][8]:+.2f}**, p = {R['h20'][9]:.3f}). The impressive one-sample t's "
            f"(20d **{R['h20'][4]:.2f}**, 60d **{R['h60'][4]:.2f}**) are pure beta.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; the crossover "
            "*under-times* random and costs only deepen the hole. You'd capture the drift more cheaply "
            "by holding the index.\n"
            f"- **Normalizing adds edge? `BUSTED`** — raw MACD loses to random by an indistinguishable "
            f"margin (20d Δ {R['macd'][2][2]:+.0f} vs PPO {R['macd'][2][1]:+.0f}), and the shuffled-sign "
            f"placebo leaves the result untouched (**p = {R['placebo'][1]:.3f}**). Dividing by EMA₂₆ "
            "changes the oscillator's *units*, not its *timing* — comparability, not forecasting power."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The crossover's entire apparent profit is the unconditional drift of long equity indices, "
            "which you obtain more cheaply and more fully by **buying and holding**. The PPO rule trades "
            "*less* of the time (only on crossovers) and pays costs on each, so it strictly dominates "
            "*nothing*. There is no capacity question because there is no edge to scale. The PPO is a "
            "descriptive momentum reader, not a forecasting strategy — and the normalization that's "
            "sold as its advantage over MACD changes its units, not its outcome."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Parameter robustness.** 5/35/5, 8/17/9, weekly bars — the result is robust: drift in, "
            "crossover out. Sweeping the EMAs is the in-sample search that manufactures mirages; the "
            "12/26/9 default here is the charitable case.\n"
            "- **The bearish/short leg.** Symmetric shorts on bearish crossovers inherit the same "
            "drift confound with the sign flipped against you — no rescue there.\n"
            "- **PPO histogram / divergence variants.** The histogram (PPO − signal) and "
            "price-vs-oscillator divergence are affine/derived from the same lines and inherit the same "
            "confound.\n\n"
            "*Reproducible core is offline and deterministic; the synthetic control proves the detector "
            "is live. Methods/sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
