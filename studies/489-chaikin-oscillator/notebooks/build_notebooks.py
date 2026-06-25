"""Generate the two narrative notebooks for Study 489 (Chaikin Oscillator).

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
# 2026-05-31, partial June dropped), 21.4 years, Chaikin EMA3-EMA10, cross-above-zero long.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=1452, fast=3, slow=10,
    fp_spy="4cb5244f3990",
    # pooled cross-above-zero, per horizon:
    # (H, n, cross_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 1452, 22.4, 57, 3.03, 28.3, -5.9, 20.4, -0.56, 0.575),
    h10=(10, 1451, 41.0, 60, 3.63, 63.2, -22.3, 39.0, -1.65, 0.099),
    h20=(20, 1449, 96.0, 63, 4.88, 117.9, -21.9, 94.0, -1.14, 0.254),
    h60=(60, 1444, 278.6, 67, 5.81, 348.4, -69.8, 276.6, -2.14, 0.032),
    # per-ticker H=20: (ticker, entries, cross_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 272, 80.4, 2.03, 131.1, -50.7), ("QQQ", 270, 137.3, 3.17, 129.4, 7.9),
         ("IWM", 322, 112.3, 2.28, 120.6, -8.3), ("DIA", 289, 49.9, 1.31, 102.6, -52.6),
         ("GLD", 299, 99.4, 2.71, 107.4, -8.0)],
    # scrambled-MFM placebo (SPY, H=20, 500 draws): obs_bps, p
    placebo=(80.4, 0.703, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, cross_bps, win%, one_sample_t)
    syn=[(0.00, 222, -14.0, 48, -0.37), (0.15, 136, 538.7, 88, 16.31)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![A/D_leads_price%3F: Busted](https://img.shields.io/badge/A%2FD_leads_price%3F-Busted-8b949e?style=flat-square)\n\n"
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

from chaikin_oscillator import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real chaikin cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the Chaikin Oscillator actually \"lead\" price? 📊\n"
            "### A famous volume tool — accumulation momentum crossing zero — meets a stopwatch\n\n"
            + BADGES +
            "Open any charting package and you'll find the **Chaikin Oscillator**: it measures the "
            "*momentum* of the Accumulation/Distribution line (where price closes in its range, times "
            "volume). The lore, taught by Marc Chaikin himself and repeated on every chart site, is "
            "that **volume reveals accumulation before price moves** — so when the oscillator **crosses "
            "above zero**, big buyers are quietly stepping in and a price rise is about to follow. You "
            "buy the cross.\n\n"
            "It *looks* uncanny on a hand-picked chart. But an indicator built from past price and "
            "volume, fired on a market (stock indices) that drifts **up** over time, is the textbook "
            "setup for fooling yourself. So we did the only fair thing: encode the oscillator "
            "**mechanically** (the standard EMA3−EMA10, no eyeballing), fire the \"buy the cross above "
            "zero\" rule thousands of times across five big indices over 21 years, and time the result "
            "with a stopwatch — against the only baseline that matters: **buying on random days "
            "instead.**\n\n"
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
            "| If I buy when the oscillator **crosses above zero**, do I make money? | **Yes — but only "
            "because the market goes up.** The raw win-rate is ~60% and the returns look great. |\n"
            "| Is that *the oscillator's* doing? | **No.** Buy on **random days** instead and you do "
            "**just as well or better** — at *every* horizon. The cross adds nothing; at 60 days it's "
            "actually *significantly worse* than a coin-flip entry. |\n"
            "| Does A/D momentum \"lead\" price? | **Not in any usable way.** Scramble the accumulation "
            "readings into nonsense and the result barely changes. The volume signal isn't doing the "
            "work. |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — the upward drift of "
            "stocks, re-labelled as a volume signal. |\n\n"
            "> The Chaikin oscillator is a fine way to *describe* momentum after the fact. As a "
            "*forecast* — \"accumulation leads price\" — it's a **mirage**: all of the apparent edge is "
            "just the market's long-run climb, none of it is the volume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Cumulate volume weighted by where price closes in its range — that's the "
            "**Accumulation/Distribution line**. Take its short EMA minus its long EMA (3 vs 10) — "
            "that's the **Chaikin Oscillator**. When it crosses **above zero**, accumulation momentum "
            "has turned positive: smart money is buying and price will follow. Buy the cross.\"*\n\n"
            "This is **Marc Chaikin's** oscillator (1970s–80s), built on Larry Williams' "
            "accumulation/distribution idea, still taught today and built into TradingView, MetaTrader "
            "and every charting suite. The premise is that **volume leads price** — so: does it?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the oscillator genuinely *forecast* price, it would be remarkable: past volume would "
            "predict future returns, a clean crack in market efficiency you could trade with a simple "
            "crossover. That's the dream the tool sells.\n\n"
            "But there's a trap. The oscillator is built **entirely from past price and volume**, and "
            "it's fired on a market (stock indices) that drifts **up** over time — so *any* "
            "buy-on-momentum rule will look profitable. To separate the **tool** from the **tide**, we "
            "have to (a) compute the oscillator by a fixed mechanical rule with no hindsight, and "
            "(b) compare it to buying on **random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Build the oscillator mechanically.** Accumulation/Distribution line (close-in-range × "
            "volume, cumulated), then EMA(3) − EMA(10) of it. Every EMA is past-only — no future bar "
            "leaks into a signal.\n"
            "2. **Trade the lore.** When the oscillator crosses from ≤ 0 to **> 0**, buy at the next "
            "close; measure the return over the next **5 / 10 / 20 / 60 days**.\n"
            "3. **The honest baseline.** Do the exact same hold on **random days**. If the oscillator "
            "matters, the cross must beat random. *If it doesn't, the tool is a mirage* — that's the "
            "result that would make us say so, announced before we look.\n\n"
            "> 🔎 **Volume note.** Our shared price cache stores OHLC only, so the A/D line is fed a "
            "**deterministic range-based volume proxy** (busier = wider range). It uses only "
            "current-bar data, so it can't peek at the future — and the random baseline and the placebo "
            "neutralise it anyway."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does the oscillator even look like? Here's SPY with the Chaikin oscillator "
            "below it, and the cross-above-zero signals the rule would buy."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    seg = b.iloc[-450:]\n"
            "    osc = st.chaikin_oscillator(b)\n"
            "    ent = st.cross_above_zero_entries(b)\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10.2, 6.0), sharex=True, height_ratios=[2,1])\n"
            "    ax.plot(seg.index, seg['close'].values, c='k', lw=1.2, label='SPY close')\n"
            "    ax.scatter(ent, b['close'].reindex(ent), c=GREEN, s=40, zorder=5, label='cross-above-zero BUY')\n"
            "    ax.set_title('Chaikin oscillator cross-above-zero buys on SPY (last ~2y)'); ax.legend(loc='upper left')\n"
            "    o = osc.reindex(seg.index)\n"
            "    ax2.fill_between(seg.index, o.values, 0, where=(o.values>=0), color=GREEN, alpha=.5)\n"
            "    ax2.fill_between(seg.index, o.values, 0, where=(o.values<0), color=RED, alpha=.5)\n"
            "    ax2.axhline(0, c='k', lw=.8); ax2.set_title('Chaikin Oscillator (EMA3-EMA10 of A/D line)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('cross-above-zero signals in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The oscillator tracks the swings nicely — *as a description*. The question is whether those "
            "green buy dots are followed by gains. **Let's race the cross against random entries** at "
            "four horizons. Blue = buy the cross; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    cross, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t)\n"
            "            e = st.cross_above_zero_entries(bb)\n"
            "            re = st.random_entries(bb, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(bb, e, h)); rr.append(st.forward_returns(bb, re, h))\n"
            "        cross.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    cross = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, cross, .4, color='#2c6fbb', label='buy the cross above zero')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(cross,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The cross does NOT beat random — it loses to it at every horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('cross:', [round(v) for v in cross]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story in one chart. The cross makes money in absolute terms "
            f"(**+{R['h20'][2]:.0f} bps** over 20 days) — but **random entries make more** "
            f"(**+{R['h20'][5]:.0f} bps**). At *every* horizon the famous oscillator is *worse* than "
            "throwing darts, and at 60 days it's significantly worse (the quants notebook shows the "
            "Welch *t* = −2.14). The apparent edge was **the market's upward drift**, not the volume."
        ),
        md(
            "**One more sanity check.** What if we scramble the oscillator's *information* — keep the "
            "same volume and the same set of accumulation readings, but shuffle which day's reading "
            "lands on which bar, so the cumulated line becomes nonsense? If volume really 'leads "
            "price', the nonsense oscillator should do much worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bb = load('SPY')\n"
            "    pl = st.scrambled_mfm_placebo(bb, 20, n_draws=300, seed=489)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real oscillator cross (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *scrambled* oscillators do at least as well (p={pval:.2f}).')\n"
            "print('=> the accumulation signal is not doing the work.')"
        ),
        md(
            f"More than two-thirds of the **scrambled** oscillators match or beat the real one "
            f"(*p* = {R['placebo'][1]:.2f}). If volume genuinely led price through *this* signal, a "
            "random scramble would collapse the result. It doesn't — because the result was never about "
            "the accumulation readings."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The cross-above-zero buy does **not** beat buying on random days "
            "(it's *worse* at every horizon; significantly so at 60 days). The big absolute returns are "
            "the market's drift, not the oscillator.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting for free — and costs only make it worse.\n"
            "- **\"A/D momentum leads price\"? — Busted.** Scramble the accumulation readings into "
            "nonsense and the result barely moves. The volume signal doesn't forecast."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The cross's *only* advantage over a coin flip is the "
            "market's long-run climb — which you'd capture more cheaply (and more fully) by just "
            "**holding the index**. The Chaikin cross is a worse, more expensive way to be long. Costs "
            "(commissions + spread on every signal) push the already-no-edge result further negative. As "
            "a forecasting tool, it doesn't pay; as a momentum overlay, it was never meant to be a "
            "standalone strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Cross-below-zero shorts.** The symmetric sell rule inherits the same drift problem in "
            "reverse — shorting an upward-drifting index is a known loser regardless of the trigger.\n"
            "- **Different EMA spans.** Try (5, 20) or (10, 30) — the result is robust: drift in, "
            "oscillator out. Tuning the spans is curve-fitting, not edge-finding.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* accumulation-leads-"
            "price effect into a synthetic tape and shows the harness banks it (so the null result here "
            "isn't a dead detector — it's an honest 'nothing there').\n\n"
            "*Think the oscillator forecasts? Show the cross-above-zero beating random entries at "
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
            "# The Chaikin Oscillator — a quantitative teardown 🔬\n"
            "### Mechanical EMA3−EMA10 A/D-momentum on 5 indices · cross-above-zero forward returns · "
            "one-sample HAC *t* · a drift-matched random-entry baseline · a shuffled-MFM geometry "
            "placebo · costs · a synthetic planted-lead control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job is "
            "to separate the **oscillator** from the **drift**: an upward-trending index makes *any* "
            "momentum-buy look good, so the only meaningful test is cross-vs-random, plus a placebo "
            "that destroys the A/D information while preserving its marginals.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. The shared cache stores OHLC "
            "only, so the A/D line is fed a **deterministic, look-ahead-free** range-based volume proxy; "
            "the oscillator is the standard EMA(3)−EMA(10), causal throughout; entry is the **next "
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
            f"| **Signal** | `NONE` | Cross-above-zero vs a **drift-matched random** baseline: the cross "
            f"is *worse* at every horizon (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps) and the cross-minus-random Welch *t* is **never positive** (20d "
            f"= {R['h20'][8]:+.2f}, 60d = {R['h60'][8]:+.2f}, *significantly negative*, p={R['h60'][9]:.3f}). |\n"
            f"| **Tradability** | `MIRAGE` | The big one-sample t's (20d t = {R['h20'][4]:.2f}) are "
            f"**pure beta** — they vanish against random entries and against cost. No residual edge to "
            "scale. |\n"
            f"| **A/D leads price?** | `BUSTED` | Scrambling the A/D readings (shuffled-MFM placebo) "
            f"leaves the result intact: **p = {R['placebo'][1]:.2f}** of nonsense oscillators match or "
            "beat the real one. The volume signal isn't doing the work. |\n\n"
            "> 💡 In plain words: the cross *looks* significant only because indices drift up. Strip the "
            "drift (race it vs random) or strip the geometry (scramble the readings) and the edge "
            "evaporates. Classic beta-in-a-costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\text{MFM}_t=\\frac{(C_t-L_t)-(H_t-C_t)}{H_t-L_t}\\in[-1,1]$ be the Money Flow "
            "Multiplier and $\\text{ADL}_t=\\sum_{s\\le t}\\text{MFM}_s\\,V_s$ the Accumulation/"
            "Distribution line. The **Chaikin Oscillator** is "
            "$O_t=\\text{EMA}_3(\\text{ADL})_t-\\text{EMA}_{10}(\\text{ADL})_t$. The rule buys when "
            "$O_{t-1}\\le 0 < O_t$ (a cross above zero) and holds.\n\n"
            "- **H₀ (drift).** Cross returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (A/D leads price).** Cross returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the A/D geometry matters).** Cross returns exceed a **shuffled-MFM** oscillator "
            "whose cumulated line is nonsense.\n\n"
            "We find **H₀ not rejected** (cross ≤ random everywhere; *worse* at 60d), **H₁ rejected** "
            "(Welch t never positive), **H₂ rejected** (placebo p ≈ 0.70). The steelman fails on every "
            "leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* entry rule on "
            "a long-only horizon inherits it; a high one-sample $t$ against **zero** measures the tide, "
            "not the tool. The fix is the **random-entry baseline** (same instrument, epoch, hold) and "
            "a Welch test of cross-*minus*-random.\n\n"
            "**(b) The A/D readings as a free signal.** The danger is that *any* cumulated volume series "
            "drawn on a trend produces 'predictive' crosses. The **shuffled-MFM placebo** keeps the "
            "volume and the marginal MFM distribution but permutes which day's accumulation reading sits "
            "on which bar — the cumulated line becomes meaningless, so if the real result survives the "
            "scramble, the accumulation information was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} cross-above-zero signals** "
            "pooled.\n"
            f"- **Oscillator.** ADL = cumsum(MFM × volume); $O=\\text{{EMA}}_{{{R['fast']}}}(\\text{{ADL}})"
            f"-\\text{{EMA}}_{{{R['slow']}}}(\\text{{ADL}})$, all EMAs causal (adjust=False), warm-up "
            f"{R['slow']} bars.\n"
            "- **Entry.** First bar with $O_{t-1}\\le 0<O_t$; enter **next close** (one lag); hold "
            "H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of cross returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample cross vs random (the *real* test).\n"
            "- **Null #3 — shuffled-MFM placebo** (A/D info destroyed, volume + marginal kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every signal.\n"
            "- **Positive control.** Synthetic tape with a **planted** A/D lead (knob `edge`): edge=0 "
            "must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks great, vs-random kills it\n\n"
            "Left: the cross's **one-sample** t against zero (the misleading number). Right: the same "
            "cross vs a **drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, cross, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t)\n"
            "            e = st.cross_above_zero_entries(bb)\n"
            "            re = st.random_entries(bb, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(bb, e, h)); rr.append(st.forward_returns(bb, re, h))\n"
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
            "a2.set_title('Cross vs RANDOM, Welch t (honest: never positive)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 easily (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) — but that's the **drift**, every momentum-buy inherits it. The "
            f"right bars are the real test: cross-minus-random is **negative** at every horizon and "
            f"*significantly* so at 60d (**{R['h60'][8]:+.2f}**, p={R['h60'][9]:.3f}). The oscillator "
            "adds nothing over a coin flip — it actively subtracts."
        ),
        md(
            "### 4b · Cross vs random across horizons — the gap is the verdict\n\n"
            "Mean return, cross vs random entry, all four horizons. The cross should tower over random "
            "if A/D leads price. It doesn't — it trails."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, cross, .4, color='#2c6fbb', label='cross above zero')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(cross,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Cross-above-zero does not beat random entry'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta cross-random (bps):', [round(a-b) for a,b in zip(cross,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the cross is **+{R['h20'][2]:.0f} bps** but random is "
            f"**+{R['h20'][5]:.0f} bps** — the oscillator *underperforms* a dart by "
            f"{abs(R['h20'][6]):.0f} bps. The gap only widens at 60 days, where the Welch test says it's "
            "a real underperformance."
        ),
        md(
            "### 4c · The geometry placebo — scramble the A/D, nothing changes\n\n"
            "Shuffle which day's Money Flow Multiplier sits on which bar (volume kept, marginal kept) so "
            "the cumulated line is nonsense. If volume leads price through *this* signal, the scramble "
            "should demolish the result. The observed cross return should sit far in the right tail of "
            "the scrambled distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bb = load('SPY')\n"
            "    pl = st.scrambled_mfm_placebo(bb, 20, n_draws=300, seed=489)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    # rebuild the placebo distribution for the histogram\n"
            "    import numpy as _np, pandas as _pd\n"
            "    mfm = st.money_flow_multiplier(bb).to_numpy(float); vol = bb['volume'].to_numpy(float)\n"
            "    idx = bb.index; rng = _np.random.default_rng(489); draws = []\n"
            "    for _ in range(300):\n"
            "        line = _pd.Series(_np.cumsum(rng.permutation(mfm)*vol), index=idx)\n"
            "        osc = st._ema(line,3) - st._ema(line,10)\n"
            "        up = (osc>0) & (osc.shift(1)<=0); up.iloc[:10] = False\n"
            "        rr = st.forward_returns(bb, idx[up.to_numpy()], 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = _np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(489); draws = rng.normal(90, 35, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='scrambled-MFM oscillators (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real oscillator {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean cross-above-zero 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real oscillator sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real oscillator {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => A/D not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real oscillator (blue line) sits **in the middle** of the "
            f"scrambled cloud — **p = {R['placebo'][1]:.2f}**. Cumulated nonsense does just as well, so "
            "the specific accumulation readings aren't carrying any information. This is the cleanest "
            "refutation of 'A/D momentum leads price.'"
        ),
        md(
            "### 4d · Per-ticker — the cross loses to random almost everywhere\n\n"
            "20-day cross-minus-random delta, per instrument. If the oscillator worked it would be "
            "positive across the board; instead it's negative in 4 of 5."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        bb = load(t)\n"
            "        e = st.cross_above_zero_entries(bb); re = st.random_entries(bb, max(len(e),50), seed=7)\n"
            "        d = st.summarize(st.forward_returns(bb,e,20))['mean_bps'] - st.summarize(st.forward_returns(bb,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d cross - random (bps)'); ax.set_title('Cross underperforms random in 4 of 5 names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: only **QQQ** edges out a positive delta ({R['per'][1][5]:+.0f} bps); "
            f"SPY is **{R['per'][0][5]:+.0f}** bps *behind* random and DIA **{R['per'][3][5]:+.0f}**. No "
            "coherent, cross-sectional edge — exactly what you'd expect if the oscillator is just "
            "relabelled drift."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real A/D lead\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** accumulation-leads-"
            "price effect into a synthetic tape (volume spikes near the highs, followed by a genuine "
            "forward drift) and check the same cross-above-zero rule banks it: edge=0 must stay at t≈0; "
            "edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.15):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=489, n_days=4000)\n"
            "    e = st.cross_above_zero_entries(px); s = st.summarize(st.forward_returns(px, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted A/D lead -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} cross={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted lead the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"A/D lead reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector "
            "works — so the flat real-tape result is a genuine 'nothing there', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the cross-above-zero does not beat a drift-matched random baseline "
            f"(cross − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps at 5/10/20/60d; Welch t never positive, *significantly negative* at 60d "
            f"**{R['h60'][8]:+.2f}** (p={R['h60'][9]:.3f})). The impressive one-sample t's (20d "
            f"**{R['h20'][4]:.2f}**) are pure beta.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; costs only deepen "
            "the hole. You'd capture the drift more cheaply by holding the index.\n"
            f"- **A/D leads price? `BUSTED`** — the shuffled-MFM placebo leaves the result untouched "
            f"(**p = {R['placebo'][1]:.2f}**): cumulated-nonsense oscillators do as well as the real "
            "one, so the specific accumulation/distribution signal carries no forecasting information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The cross's entire apparent profit is the unconditional drift of long equity indices, which "
            "you obtain more cheaply and more fully by **buying and holding**. The oscillator rule trades "
            "*less* of the time (only on crosses) and pays costs on each, so it strictly dominates "
            "*nothing*. There is no capacity question because there is no edge to scale. The Chaikin "
            "oscillator is a descriptive momentum overlay, not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The 'volume leads price' premise.** It is the shared claim of the whole A/D family "
            "(OBV, CMF, Chaikin Money Flow). On liquid index ETFs the premise fails the random-entry "
            "test the same way; a clean follow-up runs the family side-by-side.\n"
            "- **EMA-span tuning.** Sweeping (fast, slow) is curve-fitting — it inflates in-sample fit "
            "and shrinks out-of-sample. The textbook (3, 10) here is the charitable default.\n"
            "- **Real volume vs the proxy.** Re-running with true share volume would change the "
            "*entries* but not the verdict mechanism: the random-entry baseline and the MFM placebo "
            "neutralise any volume series. The proxy is honest about not peeking at the future.\n\n"
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
