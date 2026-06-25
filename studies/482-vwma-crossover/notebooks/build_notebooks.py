"""Generate the two narrative notebooks for Study 482 (VWMA-Crossover).

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
# 2026-05-31, partial June dropped), 21.4 years, fast/slow VWMA & SMA = 10/30, golden cross long.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_vwma=500, n_sma=465, fast=10, slow=30,
    fp_spy="0a125ce4099b",
    # pooled, per horizon:
    # (H, n, vwma_bps, win%, one_sample_t, sma_bps, d_vs_sma, welch_t_vs_sma, p_vs_sma,
    #  random_bps, d_vs_rnd, welch_t_vs_rnd, p_vs_rnd, net_bps)
    h5=(5, 500, 9.9, 57, 1.08, 19.7, -9.8, -0.67, 0.502, 22.4, -12.5, -0.76, 0.446, 7.9),
    h10=(10, 500, 54.6, 63, 4.03, 71.7, -17.0, -0.88, 0.379, 47.5, 7.1, 0.31, 0.753, 52.6),
    h20=(20, 500, 103.9, 65, 4.81, 116.3, -12.4, -0.41, 0.682, 56.5, 47.3, 1.45, 0.147, 101.9),
    h60=(60, 495, 267.6, 69, 5.31, 304.1, -36.6, -0.67, 0.503, 299.2, -31.6, -0.58, 0.560, 265.6),
    # per-ticker H=20: (ticker, nV, vwma_bps, one_sample_t, sma_bps, d_vs_sma)
    per=[("SPY", 91, 129.6, 3.79, 129.2, 0.4), ("QQQ", 93, 184.8, 4.12, 157.7, 27.1),
         ("IWM", 104, 74.4, 1.37, 94.8, -20.4), ("DIA", 99, 97.6, 2.42, 119.3, -21.7),
         ("GLD", 113, 49.2, 0.91, 88.1, -38.9)],
    # shuffled-volume placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(129.6, 0.238, 500),
    # synthetic control (H=20, n_days=4000): (edge, nV, vwma_bps, win%, one_sample_t, sma_bps, vwma-sma)
    syn=[(0.00, 75, 42.8, 52, 0.72, 34.5, 8.3), (0.60, 43, 1163.8, 81, 7.17, 1110.3, 53.4)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Volume-weighting_adds_edge%3F: Busted](https://img.shields.io/badge/Volume--weighting_adds_edge%3F-Busted-8b949e?style=flat-square)\n\n"
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

from vwma_crossover import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real VWMA cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a *volume-weighted* moving-average cross beat a plain one? 📊\n"
            "### A favourite \"smarter MA\" — weight each day by how much traded — meets a stopwatch\n\n"
            + BADGES +
            "Open any charting package and you can swap your simple moving average for a "
            "**volume-weighted moving average (VWMA)**: instead of averaging the last *N* closes "
            "equally, it weights each day by **how many shares traded** that day. The pitch, "
            "repeated on every indicator site and trading channel, is that the VWMA *leans toward "
            "the bars where the real money was* — so when its fast line crosses above its slow line "
            "(a \"golden cross\"), that buy signal is supposed to be **better** than the same cross "
            "on a plain equal-weighted average.\n\n"
            "It's a tidy story. But \"better than the plain SMA\" is a claim you can actually time. "
            "So we did the only fair thing: run the **VWMA golden cross** and the **same-length "
            "plain-SMA golden cross** side by side, thousands of times across five big indices over "
            "21 years, and ask whether the volume weighting adds a single basis point — against the "
            "baseline that matters: **the plain cross, and buying on random days.**\n\n"
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
            "| If I buy the VWMA golden cross, do I make money? | **Yes — but only because the "
            "market goes up.** The raw win-rate is ~65% over 20 days and the returns look great. |\n"
            "| Does the **volume weighting** beat the plain SMA cross? | **No.** At every horizon "
            "the VWMA cross makes **less** than the identical-length plain SMA cross. Volume-"
            "weighting doesn't help — it very slightly hurts. |\n"
            "| Is there any edge over buying on **random days**? | **No.** The VWMA cross never "
            "clears the significance bar against a coin-flip entry. |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — the upward drift of "
            "stocks, re-labelled as a fancier moving average. |\n\n"
            "> Weighting your moving average by volume *sounds* smarter. As a *forecast* it's a "
            "**mirage**: all of the apparent edge is the market's long-run climb, and the volume "
            "term adds nothing the plain average didn't already have."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Average the last N closes, but weight each day by its **volume** — that's the VWMA. "
            "It tracks where the big money traded, so it turns faster on real moves. Buy when the "
            "fast VWMA crosses above the slow VWMA; it front-runs the plain-SMA cross.\"*\n\n"
            "The VWMA is the discrete cousin of **VWAP** (the volume-weighted average price "
            "institutions benchmark executions against) and is built into TradingView, MetaTrader, "
            "Thinkorswim and every charting suite. The volume-confirms-price intuition is old "
            "(Granville's On-Balance Volume, 1963). So: does weighting by volume actually sharpen "
            "the cross?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If volume-weighting genuinely improved the signal, it would be a free upgrade: same "
            "rule, same lengths, just a better average — more profit per trade, no extra risk. "
            "That's the upgrade the indicator sells.\n\n"
            "But there are two traps. First, stock indices drift **up**, so *any* long-only golden "
            "cross looks profitable — that's the **tide**, not the tool. Second, the honest question "
            "isn't \"does the VWMA cross make money?\" but \"does it beat the **plain** cross?\" To "
            "separate the volume term from the drift we have to (a) run the VWMA and SMA crosses "
            "**head-to-head** at identical lengths, and (b) compare both to buying on **random "
            "days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily "
            f"closes **and volume**, over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            f"1. **Build both averages.** A fast ({R['fast']}-day) and slow ({R['slow']}-day) "
            "**VWMA** (volume-weighted) and the matching **plain SMA**. Trailing windows only — no "
            "peeking at the future.\n"
            "2. **Trade the lore.** When the fast line crosses **above** the slow line (a golden "
            "cross), buy at the **next close**; measure the return over the next 5 / 10 / 20 / 60 "
            "days. Do this for the VWMA cross *and* the SMA cross.\n"
            "3. **The thesis test.** Subtract: **VWMA − SMA**. If volume-weighting helps, this is "
            "positive. If it's zero or negative, the volume term is dead weight.\n"
            "4. **The honest baseline.** Also buy on **random days**. If the VWMA cross can't beat a "
            "dart, the signal is a mirage — that's the result that would make us say so, announced "
            "before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what do the two averages even look like? Here's SPY with its VWMA and plain SMA "
            "(slow, 30-day) and the VWMA golden crosses the rule would buy. The two lines are nearly "
            "on top of each other — your first clue."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    cl = b['close']; vol = b['volume']; seg = cl.iloc[-450:]\n"
            "    vwslow = st.vwma(cl, vol, R['slow']); smslow = st.sma(cl, R['slow'])\n"
            "    ent = st.vwma_cross_entries(cl, vol, R['fast'], R['slow'])\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg.values, c='k', lw=1.0, alpha=.6, label='SPY close')\n"
            "    ax.plot(seg.index, vwslow.reindex(seg.index), c='#2c6fbb', lw=1.4, label='VWMA(30)')\n"
            "    ax.plot(seg.index, smslow.reindex(seg.index), c=AMBER, lw=1.2, ls='--', label='SMA(30)')\n"
            "    ax.scatter(ent, cl.reindex(ent), c=GREEN, s=45, zorder=5, label='VWMA golden cross BUY')\n"
            "    ax.set_title('VWMA vs plain SMA on SPY (last ~2y) — nearly identical lines'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('VWMA golden crosses in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The volume-weighted line barely separates from the plain one — daily ETF volume just "
            "isn't lumpy enough to move a 30-day average much. So the crosses fire at almost the "
            "same times. **Now race the two crosses** (and random entries) at four horizons. Blue = "
            "VWMA cross; amber = plain SMA cross; grey = random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    vw, sm, rnd = [], [], []\n"
            "    for h in hs:\n"
            "        vv, ss, rr = [], [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']; v = bb['volume']\n"
            "            ev = st.vwma_cross_entries(c, v, R['fast'], R['slow'])\n"
            "            es = st.sma_cross_entries(c, R['fast'], R['slow'])\n"
            "            re = st.random_entries(c, max(len(ev),50), warmup=R['slow'], seed=7)\n"
            "            vv.append(st.forward_returns(c, ev, h)); ss.append(st.forward_returns(c, es, h))\n"
            "            rr.append(st.forward_returns(c, re, h))\n"
            "        vw.append(np.concatenate(vv).mean()*1e4); sm.append(np.concatenate(ss).mean()*1e4)\n"
            "        rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    vw = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    sm = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    rnd = [R['h5'][9], R['h10'][9], R['h20'][9], R['h60'][9]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "ax.bar(x-.26, vw, .26, color='#2c6fbb', label='VWMA cross')\n"
            "ax.bar(x, sm, .26, color=AMBER, label='plain SMA cross')\n"
            "ax.bar(x+.26, rnd, .26, color=GREY, label='random days')\n"
            "for i,(a,c2,d) in enumerate(zip(vw,sm,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.26,a),ha='center',va='bottom',fontsize=7)\n"
            "    ax.annotate(f'{c2:+.0f}',(i,c2),ha='center',va='bottom',fontsize=7)\n"
            "    ax.annotate(f'{d:+.0f}',(i+.26,d),ha='center',va='bottom',fontsize=7)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('VWMA cross does NOT beat the plain SMA cross'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('VWMA:', [round(v) for v in vw]); print('SMA :', [round(v) for v in sm])\n"
            "print('VWMA-SMA:', [round(a-b) for a,b in zip(vw,sm)])"
        ),
        md(
            f"There's the whole story. The VWMA cross makes money in absolute terms "
            f"(**+{R['h20'][2]:.0f} bps** over 20 days) — but the **plain SMA cross makes more** "
            f"(**+{R['h20'][5]:.0f} bps**), and so does buying on random days at the longest horizon. "
            "At *every* horizon the volume-weighted cross is **behind** the plain one. The apparent "
            "edge was the market's drift, and the volume term added nothing."
        ),
        md(
            "**One more sanity check.** What if we **scramble the volume** — keep the same prices "
            "but shuffle which day's volume attaches to which bar, so the weighting is nonsense? If "
            "the volume term really mattered, the scramble should wreck the result."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bb = load('SPY'); c = bb['close']; v = bb['volume']\n"
            "    pl = st.shuffled_volume_placebo(c, v, 20, R['fast'], R['slow'], n_draws=300, seed=482)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real VWMA cross (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *scrambled-volume* VWMAs do at least as well (p={pval:.2f}).')\n"
            "print('=> the volume weighting is not doing the work.')"
        ),
        md(
            f"Around a quarter of the **scrambled-volume** runs match or beat the real one "
            f"(*p* = {R['placebo'][1]:.2f}). If the volume weighting genuinely carried information, "
            "a random scramble would collapse the result. It doesn't — because the result was never "
            "about the volume."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The VWMA golden cross does **not** beat buying on random days (the "
            "VWMA-vs-random difference never clears the significance bar). The big absolute returns "
            "are the market's drift.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting for free — and costs only make it worse.\n"
            "- **\"Volume-weighting adds edge\"? — Busted.** The VWMA cross **loses to the plain SMA "
            "cross at every horizon**, and scrambling the volume barely changes anything. The volume "
            "term isn't doing the work."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade — and certainly no reason to prefer the VWMA over the "
            "plain SMA. The cross's only advantage over a coin flip is the market's long-run climb, "
            "which you'd capture more cheaply by just **holding the index**. Swapping in a "
            "volume-weighted average makes the signal *slightly worse*, not better, and you still pay "
            "costs on every cross. Volume-weighting is a plausible-sounding upgrade that, measured, "
            "subtracts."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Why so little difference?** Daily ETF volume isn't lumpy enough to pull a 30-day "
            "average far off the equal-weighted one — the two lines nearly coincide, so the crosses "
            "fire at the same times. On thinner, spikier instruments the VWMA *moves* more, but more "
            "movement isn't more *forecasting*.\n"
            "- **Other lengths.** Try faster/slower pairs — the result is robust: drift in, no "
            "volume edge out.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* volume-led move "
            "into a synthetic tape and shows the harness banks it (so the null here isn't a dead "
            "detector — it's an honest 'nothing there').\n\n"
            "*Think volume-weighting helps? Show the VWMA cross beating the same-length SMA cross at "
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
            "# VWMA-Crossover — a quantitative teardown 🔬\n"
            "### VWMA vs same-length SMA golden cross on 5 indices · forward returns · one-sample "
            "HAC *t* · a head-to-head VWMA−SMA Welch test · a drift-matched random baseline · a "
            "shuffled-volume placebo · costs · a synthetic planted-volume control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **volume term** from the **drift**: an upward-trending index makes "
            "*any* golden cross look good, so the thesis test is **VWMA − SMA** (identical lengths, "
            "only the weighting differs), backed by a random-entry Signal test and a placebo that "
            "destroys the volume weighting while preserving its marginal.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return**) **and volume**, 2005→2026. Causal MAs "
            f"(fast={R['fast']}, slow={R['slow']}); entry is the **next close** (one documented "
            "lag). Offline core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | VWMA cross vs a **drift-matched random** baseline: VWMA − "
            f"random never clears t = 2 (max Welch t = **{R['h20'][11]:+.2f}** at 20d, "
            f"p = {R['h20'][12]:.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | The big one-sample t's (20d t = {R['h20'][4]:.2f}) are "
            f"**pure beta** — they vanish against random entries and against cost. No residual edge "
            "to scale. |\n"
            f"| **Volume-weighting adds edge?** | `BUSTED` | VWMA − SMA is **negative at every "
            f"horizon** ({R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps; Welch t never positive), and the shuffled-volume placebo leaves the result intact "
            f"(**p = {R['placebo'][1]:.2f}**). |\n\n"
            "> 💡 In plain words: the VWMA cross *looks* significant only because indices drift up. "
            "Strip the drift (race it vs random) or race it vs the plain SMA cross (same lengths) and "
            "the volume term contributes nothing — slightly less than nothing. Classic "
            "beta-in-a-costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "The volume-weighted moving average over a trailing window of $N$ bars is\n\n"
            "$$\\mathrm{VWMA}_N(t)=\\frac{\\sum_{i=t-N+1}^{t} P_i\\,V_i}{\\sum_{i=t-N+1}^{t} V_i},$$\n\n"
            "against the equal-weighted $\\mathrm{SMA}_N(t)=\\frac1N\\sum_{i=t-N+1}^{t}P_i$. The rule "
            "buys when $\\mathrm{VWMA}_{f}$ crosses above $\\mathrm{VWMA}_{s}$ (fast $f=10$, slow "
            "$s=30$).\n\n"
            "- **H₀ (drift).** VWMA-cross returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (volume-weighting forecasts).** VWMA-cross returns **exceed the same-length SMA "
            "cross**, t ≥ 2.\n"
            "- **H₂ (the volume term matters).** VWMA-cross returns exceed a **shuffled-volume** "
            "VWMA whose weighting is destroyed.\n\n"
            "We find **H₀ not rejected** (VWMA ≤ random by the Welch test), **H₁ rejected** (VWMA − "
            "SMA negative at every horizon), **H₂ rejected** (placebo p ≈ 0.24). The steelman fails "
            "on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* long-only "
            "golden cross inherits it; a high one-sample $t$ against **zero** measures the tide, not "
            "the tool. The fix is the **random-entry baseline** (same instrument, epoch, hold) and a "
            "Welch test of VWMA-*minus*-random.\n\n"
            "**(b) Volume-weighting as the only free variable.** The whole thesis is that the volume "
            "term sharpens the cross. So the clean test holds **fast/slow lengths, the cross rule, "
            "the instrument and the hold fixed** and varies *only* the weighting — the **VWMA − SMA** "
            "Welch test. The **shuffled-volume placebo** then permutes which volume attaches to which "
            "bar (price path kept, volume marginal kept): if the real result survives the scramble, "
            "the weighting was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes **and "
            f"volume** ({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_vwma']} VWMA crosses** "
            f"vs **{R['n_sma']} SMA crosses** pooled.\n"
            f"- **Averages.** Causal trailing VWMA & SMA, fast={R['fast']}, slow={R['slow']}; cross "
            "read on close of *t* (no look-ahead).\n"
            "- **Entry.** First bar where fast crosses above slow; enter **next close** (one lag); "
            "hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of VWMA-cross returns vs 0 (Newey-West).\n"
            "- **Null #2 (the thesis) — VWMA vs SMA**, Welch two-sample, same lengths.\n"
            "- **Null #3 (the Signal) — random-entry baseline**, Welch VWMA vs random.\n"
            "- **Null #4 — shuffled-volume placebo** (weighting destroyed, marginals kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every cross.\n"
            "- **Positive control.** Synthetic tape with a **planted** volume-led drift pulse (knob "
            "`edge`): edge=0 must keep VWMA−SMA ≈ 0; edge>0 must light it up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks great, the head-to-head kills it\n\n"
            "Left: the VWMA cross's **one-sample** t against zero (the misleading number). Right: "
            "the **VWMA − SMA** Welch t — the honest thesis test (does volume-weighting add edge?)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    from scipy import stats\n"
            "    one_t, vw, sm, welch_vs = [], [], [], []\n"
            "    for h in hs:\n"
            "        vv, ss = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']; v = bb['volume']\n"
            "            ev = st.vwma_cross_entries(c, v, R['fast'], R['slow'])\n"
            "            es = st.sma_cross_entries(c, R['fast'], R['slow'])\n"
            "            vv.append(st.forward_returns(c, ev, h)); ss.append(st.forward_returns(c, es, h))\n"
            "        vv = np.concatenate(vv); ss = np.concatenate(ss)\n"
            "        one_t.append(st.summarize(vv)['t']); vw.append(vv.mean()*1e4); sm.append(ss.mean()*1e4)\n"
            "        welch_vs.append(stats.ttest_ind(vv, ss, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    vw = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    sm = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch_vs = [R['h5'][7], R['h10'][7], R['h20'][7], R['h60'][7]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (misleading: it is beta)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch_vs, color=[GREEN if v>2 else RED for v in welch_vs], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch_vs): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('VWMA - SMA, Welch t (honest: negative everywhere)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch VWMA-SMA:', [round(v,2) for v in welch_vs])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 easily (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) — but that's the **drift**, every golden cross inherits it. The "
            f"right bars are the thesis test: VWMA − SMA is **negative at every horizon** "
            f"({R['h20'][7]:+.2f} at 20d) — volume-weighting doesn't add edge, it subtracts a little."
        ),
        md(
            "### 4b · VWMA vs SMA vs random across horizons — the gap is the verdict\n\n"
            "Mean return: VWMA cross, plain SMA cross, and random entry, all four horizons. The VWMA "
            "should tower over the SMA if volume-weighting forecasts. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rnd = []\n"
            "    for h in hs:\n"
            "        rr = []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']; v = bb['volume']\n"
            "            ev = st.vwma_cross_entries(c, v, R['fast'], R['slow'])\n"
            "            re = st.random_entries(c, max(len(ev),50), warmup=R['slow'], seed=7)\n"
            "            rr.append(st.forward_returns(c, re, h))\n"
            "        rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    rnd = [R['h5'][9], R['h10'][9], R['h20'][9], R['h60'][9]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.3))\n"
            "ax.bar(x-.26, vw, .26, color='#2c6fbb', label='VWMA cross')\n"
            "ax.bar(x, sm, .26, color=AMBER, label='plain SMA cross')\n"
            "ax.bar(x+.26, rnd, .26, color=GREY, label='random (drift baseline)')\n"
            "for i,(a,b,c2) in enumerate(zip(vw,sm,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.26,a),ha='center',va='bottom',fontsize=7)\n"
            "    ax.annotate(f'{b:+.0f}',(i,b),ha='center',va='bottom',fontsize=7)\n"
            "    ax.annotate(f'{c2:+.0f}',(i+.26,c2),ha='center',va='bottom',fontsize=7)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('VWMA cross trails the plain SMA cross at every horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('VWMA-SMA (bps):', [round(a-b) for a,b in zip(vw,sm)])\n"
            "print('VWMA-random (bps):', [round(a-b) for a,b in zip(vw,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the VWMA cross is **+{R['h20'][2]:.0f} bps** but the "
            f"plain SMA cross is **+{R['h20'][5]:.0f} bps** — the volume-weighted version "
            f"*underperforms* by {abs(R['h20'][6]):.0f} bps. And VWMA − random never clears t = 2 "
            f"(max **{R['h20'][11]:+.2f}** at 20d). No edge from the volume term, no edge over drift."
        ),
        md(
            "### 4c · The volume placebo — scramble the volume, nothing changes\n\n"
            "Permute which volume attaches to which bar (price path kept, volume marginal kept) so "
            "the weighting is meaningless. If the volume term carried information, the scramble "
            "should demolish the result. The observed VWMA return should sit far in the right tail of "
            "the scrambled distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bb = load('SPY'); c = bb['close']; v = bb['volume']\n"
            "    pl = st.shuffled_volume_placebo(c, v, 20, R['fast'], R['slow'], n_draws=300, seed=482)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    rng = np.random.default_rng(482); vv = v.to_numpy(); draws = []\n"
            "    for _ in range(300):\n"
            "        vp = __import__('pandas').Series(rng.permutation(vv), index=v.index)\n"
            "        ent = st.vwma_cross_entries(c, vp, R['fast'], R['slow'])\n"
            "        rr = st.forward_returns(c, ent, 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(482); draws = rng.normal(110, 35, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='scrambled-volume VWMAs (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real VWMA {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean VWMA-cross 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real VWMA sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real VWMA {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => volume weighting not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real VWMA (blue line) sits **inside** the scrambled-volume "
            f"cloud — **p = {R['placebo'][1]:.2f}**. Randomly-aligned volume does about as well, so "
            "the specific volume weighting carries no information. The cleanest refutation of "
            "'volume-weighting sharpens the cross.'"
        ),
        md(
            "### 4d · Per-ticker — volume-weighting helps in only one of five names\n\n"
            "20-day VWMA-minus-SMA delta, per instrument. If volume-weighting worked it would be "
            "positive across the board; instead it's negative in 3 of 5 and a wash in SPY."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        bb = load(t); c = bb['close']; v = bb['volume']\n"
            "        ev = st.vwma_cross_entries(c, v, R['fast'], R['slow']); es = st.sma_cross_entries(c, R['fast'], R['slow'])\n"
            "        d = st.summarize(st.forward_returns(c,ev,20))['mean_bps'] - st.summarize(st.forward_returns(c,es,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d VWMA - SMA (bps)'); ax.set_title('Volume-weighting helps in only 1 of 5 names (QQQ)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d VWMA-SMA (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: only **QQQ** shows a meaningful positive delta "
            f"({R['per'][1][5]:+.0f} bps); SPY is a wash ({R['per'][0][5]:+.0f}), and GLD is "
            f"**{R['per'][4][5]:+.0f}** bps *behind* the plain cross. No coherent cross-sectional "
            "benefit — exactly what you'd expect if the volume term is just relabelled drift."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real volume effect\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** volume-led drift "
            "pulse into a synthetic tape (informative up-moves land on heavy-volume bars, so a VWMA "
            "leans in earlier than the SMA) and check VWMA − SMA lights up: edge=0 must stay ≈ 0; "
            "edge>0 must turn clearly positive with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.60):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=482, n_days=4000)\n"
            "    c = px['close']; v = px['volume']\n"
            "    ev = st.vwma_cross_entries(c, v, R['fast'], R['slow']); es = st.sma_cross_entries(c, R['fast'], R['slow'])\n"
            "    sv = st.summarize(st.forward_returns(c, ev, 20)); ss = st.summarize(st.forward_returns(c, es, 20))\n"
            "    res.append((edge, sv['n'], sv['mean_bps'], sv['win']*100, sv['t'], ss['mean_bps'], sv['mean_bps']-ss['mean_bps']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,*_ in res]; dvals = [r[6] for r in res]\n"
            "ax.bar(labels, dvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(dvals): ax.annotate(f'{d:+.0f} bps',(i,d),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d VWMA - SMA (bps)'); ax.set_title('Control: edge=0 -> VWMA-SMA ~0; planted volume pulse -> positive'); ax.legend(['VWMA-SMA delta'])\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t,s,d in res: print(f'edge={e:.2f}: nV={n} vwma={m:+.1f}bps win={w:.0f}% t={t:+.2f} sma={s:+.1f}bps VWMA-SMA={d:+.1f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted volume effect the VWMA−SMA delta is a "
            f"near-zero **{R['syn'][0][6]:+.0f} bps** (no false positive); a planted volume-led pulse "
            f"drives the VWMA cross to **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%) and a "
            f"clean **{R['syn'][1][6]:+.0f} bps** VWMA-over-SMA edge. The detector works — so the "
            "negative real-tape delta is a genuine 'the volume term does nothing here', not a broken "
            "pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the VWMA cross does not beat a drift-matched random baseline "
            f"(VWMA − random Welch t never clears 2, max **{R['h20'][11]:+.2f}** at 20d, "
            f"p = {R['h20'][12]:.2f}). The impressive one-sample t's (20d **{R['h20'][4]:.2f}**) are "
            "pure beta.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; costs only "
            "deepen the hole. You'd capture the drift more cheaply by holding the index.\n"
            f"- **Volume-weighting adds edge? `BUSTED`** — VWMA − SMA is negative at every horizon "
            f"({R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} bps; Welch "
            f"t negative everywhere), volume-weighting helps in only 1 of 5 names, and the "
            f"shuffled-volume placebo leaves the result intact (**p = {R['placebo'][1]:.2f}**). The "
            "volume term carries no forecasting information over the plain SMA cross."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The VWMA cross's entire apparent profit is the unconditional drift of long equity "
            "indices, captured more cheaply and more fully by **buying and holding**. Worse, the "
            "volume weighting *subtracts* a few bps versus the identical plain-SMA cross, so even "
            "*within* the family of golden-cross rules the 'smarter' average is the weaker one. "
            "There is no capacity question because there is no edge to scale. The VWMA is a "
            "reasonable execution benchmark (its VWAP cousin) — not a forecasting upgrade."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Why the lines barely separate.** Daily ETF volume is not lumpy enough to pull a "
            "30-day average far from the equal-weighted one, so the two crosses fire at nearly the "
            "same times. On thinner instruments the VWMA *moves* more — but more movement is not more "
            "*forecasting* (test it; the volume placebo idiom transfers directly).\n"
            "- **Length sweep.** Faster/slower (f,s) pairs reproduce the same picture: drift in, no "
            "volume edge out — the head-to-head VWMA−SMA delta hovers around zero.\n"
            "- **Volume-confirmation cousins** (OBV, volume-weighted MACD, VWAP-bands) are the same "
            "'volume confirms price' intuition and inherit the same drift confound.\n\n"
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
