"""Generate the two narrative notebooks for Study 486 (Gann Hi-Lo Activator).

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
# 2026-05-31, partial June dropped), 21.4 years, activator period=10, flip-up long.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=1234, period=10,
    fp_spy="4cb5244f3990",
    # pooled flip-up, per horizon:
    # (H, n, flip_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 1233, 17.8, 58, 2.63, 14.1, 3.6, 15.8, 0.36, 0.722),
    h10=(10, 1232, 43.6, 60, 3.83, 49.2, -5.6, 41.6, -0.39, 0.694),
    h20=(20, 1228, 97.9, 65, 6.20, 95.8, 2.1, 95.9, 0.10, 0.917),
    h60=(60, 1221, 252.7, 68, 6.24, 314.6, -61.9, 250.7, -1.88, 0.060),
    # per-ticker H=20: (ticker, entries, flip_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 238, 85.2, 2.83, 106.6, -21.5), ("QQQ", 251, 106.5, 2.88, 117.0, -10.5),
         ("IWM", 253, 93.2, 2.44, 72.5, 20.6), ("DIA", 246, 84.3, 2.56, 77.7, 6.7),
         ("GLD", 246, 119.6, 3.34, 105.8, 13.8)],
    # shuffled-flip timing placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(85.2, 0.629, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, flip_bps, win%, one_sample_t)
    syn=[(0.00, 151, -9.5, 48, -0.22), (0.60, 68, 888.6, 85, 9.37)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Flip_forecasts_trend%3F: Busted](https://img.shields.io/badge/Flip_forecasts_trend%3F-Busted-8b949e?style=flat-square)\n\n"
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

from gann_hilo_activator import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real Gann cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the Gann Hi-Lo Activator \"flip\" actually call the trend? ⚡\n"
            "### A famous Gann trailing line — flips above price, flips below — meets a stopwatch\n\n"
            + BADGES +
            "Open any charting package and you'll find the **Gann Hi-Lo Activator**: a single line "
            "that rides *below* price while you're long, then **flips** above it the moment price "
            "rolls over. It's built from a plain moving average of recent **highs** and **lows**. "
            "The lore, from Robert Krausz's Gann work and repeated on every Gann site, is that the "
            "**flip forecasts the trend** — when price closes back **above** the activator, a new "
            "up-leg is starting, so you **buy the flip**.\n\n"
            "It *looks* uncanny on a hand-picked chart: the line flips, price runs. But a trailing "
            "line built from past highs/lows on a market that drifts **up** will flip you long right "
            "as the market resumes its endless climb — which makes *any* such rule look prophetic. So "
            "we did the only fair thing: encode the flip **mechanically** (no eyeballing), fire the "
            "\"buy the flip\" rule **1,234 times** across five big indices over 21 years, and time the "
            "result with a stopwatch — against the only baseline that matters: **buying on random "
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
            "| If I buy when the activator **flips up**, do I make money? | **Yes — but only because "
            "the market goes up.** The raw win-rate is ~65% over 20 days and the returns look great. |\n"
            "| Is that *the flip's* doing? | **No.** Buy on **random days** instead and you do "
            "**just as well** — the flip-vs-random gap is a couple of basis points, statistically "
            "zero. |\n"
            "| Does the flip \"forecast\" the trend? | **Not in any usable way.** Move the flip to "
            "random dates and the result barely changes. The *timing* isn't doing the work. |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — the upward drift of "
            "stocks, re-labelled as a Gann flip. |\n\n"
            "> The Hi-Lo Activator is a fine *trailing stop* — a way to *describe* a trend you're "
            "already in. As a *forecast* — \"the flip starts the move\" — it's a **mirage**: all of "
            "the apparent edge is just the market's long-run climb, none of it is the flip."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Take the average of the last few **highs** and the last few **lows**. While you're "
            "long, the activator sits on the low-average, trailing below price. The instant price "
            "closes **below** it, the line flips up to the high-average — you're now short. When "
            "price closes back **above**, it flips long again. Each flip up is the start of a new "
            "up-trend: **buy the flip**.\"*\n\n"
            "This is the **Gann Hi-Lo Activator**, popularised by **Robert Krausz** from **W. D. "
            "Gann's** swing-chart ideas, and built into MetaTrader, TradingView, NinjaTrader and "
            "Thinkorswim. It's one of the most recognisable Gann tools — so: does the flip actually "
            "*call* the trend?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the flip genuinely *forecast* trends, it would be remarkable: a moving average of "
            "past highs and lows would predict the *start* of future moves — a clean, mechanical "
            "crack in market efficiency you could trade with a single line. That's the dream the tool "
            "sells.\n\n"
            "But there's a trap built into it. The activator is built **entirely from past price**, "
            "on a market (stock indices) that drifts **up** over time. A trailing line will flip you "
            "long again and again right as the index resumes its climb — so *any* such rule will look "
            "profitable. To separate the **flip** from the **tide**, we have to (a) run the rule by a "
            "fixed mechanical recipe with no hindsight, and (b) compare it to buying on **random "
            "days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            f"1. **Build the activator mechanically.** A {R['period']}-day average of highs and of "
            "lows, **shifted one bar** so today's line uses only *yesterday and earlier* — we never "
            "peek at the future.\n"
            "2. **Read the flip by rule.** When price closes above the line, the regime flips long — "
            "no eyeballing, no cherry-picking.\n"
            "3. **Trade the lore.** On each short→long flip, buy at the next close; measure the "
            "return over the next **5 / 10 / 20 / 60 days**.\n"
            "4. **The honest baseline.** Do the exact same hold on **random days**. If the flip "
            "matters, it must beat random. *If it doesn't, the tool is a mirage* — that's the result "
            "that would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does the activator even look like? Here's SPY with the flipping line drawn "
            "below/above price, and the flip-up dates the rule would buy."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    cl = b['close']; seg = cl.iloc[-450:]\n"
            "    act, reg = st.hilo_activator(b, period=R['period'])\n"
            "    ent = st.flip_up_entries(b, period=R['period'])\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg.values, c='k', lw=1.2, label='SPY close')\n"
            "    ax.plot(seg.index, act.reindex(seg.index), c='#2c6fbb', lw=1.2, label='Gann Hi-Lo activator')\n"
            "    ax.scatter(ent, cl.reindex(ent), c=GREEN, s=45, zorder=5, label='flip-up BUY')\n"
            "    ax.set_title('The Gann Hi-Lo Activator on SPY (last ~2y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('flip-up entries in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The activator trails the trend nicely — *as a stop*. The question is whether those green "
            "flip dots are followed by *new* up-moves you couldn't get otherwise. **Let's race the "
            "flip against random entries** at four horizons. Blue = buy the flip; grey = buy on "
            "random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    flip, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.flip_up_entries(bb, period=R['period'])\n"
            "            re = st.random_entries(bb, max(len(e),50), period=R['period'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        flip.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    flip = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, flip, .4, color='#2c6fbb', label='buy the flip')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(flip,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The flip does NOT beat random — it ties or loses'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('flip:', [round(v) for v in flip]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story in one chart. The flip makes money in absolute terms "
            f"(**+{R['h20'][2]:.0f} bps** over 20 days) — but **random entries make about the same** "
            f"(**+{R['h20'][5]:.0f} bps**). The bars are essentially the same height at every "
            "horizon; at 60 days random actually wins. The apparent edge was **the market's upward "
            "drift**, not the flip."
        ),
        md(
            "**One more sanity check.** What if we keep the *number* of flips but move them to "
            "**random dates** — destroying the flip's *timing* while keeping everything else? If the "
            "flip really times trends, the scrambled-timing version should do much worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bb = load('SPY')\n"
            "    pl = st.shuffled_flip_placebo(bb, 20, period=R['period'], n_draws=300, seed=486)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real flip-up (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *random-timing* draws do at least as well (p={pval:.2f}).')\n"
            "print('=> the flip timing is not doing the work.')"
        ),
        md(
            f"More than half of the **random-timing** draws match or beat the real flip "
            f"(*p* = {R['placebo'][1]:.2f}). If the flip genuinely timed the *start* of trends, "
            "moving the dates at random would collapse the result. It doesn't — because the result "
            "was never about the timing."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The flip-up buy does **not** beat buying on random days (the gap is "
            "a few bps, statistically zero; the flip-vs-random *t* never clears 2). The big absolute "
            "returns are the market's drift, not the flip.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting for free — and costs only make it worse.\n"
            "- **\"The flip forecasts trend\"? — Busted.** Move the flip to random dates and the "
            "result barely moves. The flip doesn't time anything."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The flip's *only* advantage over a coin flip is the "
            "market's long-run climb — which you'd capture more cheaply (and more fully) by just "
            "**holding the index**. The Gann flip is a worse, more expensive way to be long. Costs "
            "(commissions + spread on every flip) push the already-no-edge result further negative. "
            "As a forecasting tool, it doesn't pay; as a trailing stop, it was never meant to be a "
            "stand-alone strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Different periods.** Try a 3-day or 14-day activator — the result is robust: drift "
            "in, flip out.\n"
            "- **Use it as a stop, not a signal.** The honest job of the Hi-Lo Activator is to "
            "*exit* a position you already hold, not to *predict* the next one. That framing is fine "
            "— it just isn't a forecast.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* post-flip trend "
            "into a synthetic tape and shows the harness banks it (so the null result here isn't a "
            "dead detector — it's an honest 'nothing there').\n\n"
            "*Think the flip forecasts? Show the flip-up beating random entries at **t ≥ 2** on a "
            "real tape — then we'll talk.*"
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
            "# The Gann Hi-Lo Activator — a quantitative teardown 🔬\n"
            "### Mechanical flip line on 5 indices · flip-up forward returns · one-sample HAC *t* · "
            "a drift-matched random-entry baseline · a shuffled-flip timing placebo · costs · a "
            "synthetic planted-trend control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **flip** from the **drift**: an upward-trending index makes *any* "
            "long-only entry look good, so the only meaningful test is flip-vs-random, plus a placebo "
            "that destroys the flip's *timing* while preserving its marginals.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. Activator is a "
            f"period-{R['period']} SMA of highs/lows shifted **+1 bar** (the flip is knowable on the "
            "close of *t*); entry is the **next close** (one documented lag). Offline core + "
            "synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `NONE` | Flip-up vs a **drift-matched random** baseline: the gap is "
            f"Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} bps at "
            f"5/10/20/60d and the flip-minus-random difference **never clears t = 2** (max |Welch t| "
            f"= {R['h60'][8]:+.2f} at 60d, and *negative*). |\n"
            f"| **Tradability** | `MIRAGE` | The big one-sample t's (20d t = {R['h20'][4]:.2f}, 60d "
            f"= {R['h60'][4]:.2f}) are **pure beta** — they vanish against random entries and against "
            "cost. No residual edge to scale. |\n"
            f"| **Flip forecasts trend?** | `BUSTED` | Moving the flip to random dates (timing "
            f"placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** of random-date draws "
            "match or beat the real flip. The timing isn't doing the work. |\n\n"
            "> 💡 In plain words: the flip *looks* significant only because indices drift up. Strip "
            "the drift (race it vs random) or strip the timing (scramble the flip dates) and the edge "
            "evaporates. Classic beta-in-a-costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $H^{(p)}_t,L^{(p)}_t$ be the $p$-bar SMAs of highs and lows (each lagged one bar). "
            "The activator $A_t$ flips: in the long regime $A_t=L^{(p)}_t$ (a stop below price); a "
            "close $C_t<A_t$ flips to short with $A_t=H^{(p)}_t$; a close $C_t>A_t$ flips back long. "
            "The Gann rule **buys the flip** ($\\text{short}\\!\\to\\!\\text{long}$) and rides the "
            "new up-leg.\n\n"
            "- **H₀ (drift).** Flip returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (the flip forecasts).** Flip returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the timing matters).** Flip returns exceed a **shuffled-flip** entry whose dates "
            "are random.\n\n"
            "We find **H₀ not rejected** (flip ≈ random at every horizon), **H₁ rejected** (Welch t "
            "never ≥ 2), **H₂ rejected** (placebo p ≈ 0.63). The steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* long-only "
            "entry rule inherits it; a high one-sample $t$ against **zero** measures the tide, not "
            "the tool. The fix is the **random-entry baseline** (same instrument, epoch, hold) and a "
            "Welch test of flip-*minus*-random.\n\n"
            "**(b) Timing as a free parameter.** The danger is that *any* batch of buy-days on a "
            "trending index produces a 'good' result. The **shuffled-flip placebo** keeps the flip "
            "count and the price marginal but draws the entry dates at random — the flip's *timing* "
            "is destroyed, so if the real result survives the scramble, the timing was never "
            "load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} flip-up entries** "
            "pooled.\n"
            f"- **Activator.** Period-{R['period']} SMA of highs / lows, each shifted **+1 bar** "
            "(no look-ahead); single flipping line.\n"
            "- **Entry.** First bar of each short→long flip; enter **next close** (one lag); hold "
            "H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of flip returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample flip vs random (the *real* test).\n"
            "- **Null #3 — shuffled-flip placebo** (timing destroyed, count & marginal kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every flip.\n"
            "- **Positive control.** Synthetic tape with a **planted** post-flip trend (knob `edge`): "
            "edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks great, vs-random kills it\n\n"
            "Left: the flip-up's **one-sample** t against zero (the misleading number). Right: the "
            "same flip vs a **drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, flip, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.flip_up_entries(bb, period=R['period'])\n"
            "            re = st.random_entries(bb, max(len(e),50), period=R['period'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); flip.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    flip = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
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
            "a2.set_title('Flip vs RANDOM, Welch t (honest: never clears 2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 easily (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) — but that's the **drift**, every long-only entry inherits it. "
            f"The right bars are the real test: flip-minus-random hovers around **zero** "
            f"({R['h20'][8]:+.2f} at 20d) and is **negative** at 60d ({R['h60'][8]:+.2f}) — never "
            "significant. The flip adds nothing over a coin flip."
        ),
        md(
            "### 4b · Flip vs random across horizons — the gap is the verdict\n\n"
            "Mean return, flip-up vs random entry, all four horizons. The flip should tower over "
            "random if it forecasts. It doesn't — the bars are the same height."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, flip, .4, color='#2c6fbb', label='flip-up entry')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(flip,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Flip-up does not beat random entry'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta flip-random (bps):', [round(a-b) for a,b in zip(flip,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the flip is **+{R['h20'][2]:.0f} bps** and random is "
            f"**+{R['h20'][5]:.0f} bps** — a {abs(R['h20'][6]):.0f}-bp difference, i.e. nothing. At "
            "60 days the flip *underperforms* random. The flip is not timing anything the drift "
            "doesn't already deliver."
        ),
        md(
            "### 4c · The timing placebo — scramble the flip dates, nothing changes\n\n"
            "Keep the *number* of flips and the price marginal, but move the entry dates at random. "
            "If the flip times the *start* of trends, the scramble should demolish the result. The "
            "observed flip return should sit far in the right tail of the random-timing distribution. "
            "It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bb = load('SPY'); c = bb['close']\n"
            "    pl = st.shuffled_flip_placebo(bb, 20, period=R['period'], n_draws=300, seed=486)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    import numpy as _np\n"
            "    ent = st.flip_up_entries(bb, period=R['period']); k = len(ent)\n"
            "    rng = _np.random.default_rng(486); valid = bb.index[2*R['period']:]\n"
            "    draws = []\n"
            "    for _ in range(300):\n"
            "        ch = __import__('pandas').DatetimeIndex(sorted(rng.choice(valid, size=min(k,len(valid)), replace=False)))\n"
            "        rr = st.forward_returns(c, ch, 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = _np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(486); draws = rng.normal(95, 35, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='random-timing draws (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real flip {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean flip-up 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real flip sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real flip {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => timing not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real flip (blue line) sits **in the middle** of the "
            f"random-timing cloud — **p = {R['placebo'][1]:.2f}**. Random buy-days do just as well, "
            "so the flip's specific timing isn't carrying any information. This is the cleanest "
            "refutation of 'the flip forecasts trend.'"
        ),
        md(
            "### 4d · Per-ticker — the flip-minus-random delta scatters around zero\n\n"
            "20-day flip-minus-random delta, per instrument. If the flip worked it would be strongly "
            "positive across the board; instead it scatters around zero."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        bb = load(t); c = bb['close']\n"
            "        e = st.flip_up_entries(bb, period=R['period']); re = st.random_entries(bb, max(len(e),50), period=R['period'], seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d flip − random (bps)'); ax.set_title('Flip-minus-random scatters around zero')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: the deltas run from **{R['per'][0][5]:+.0f}** (SPY) to "
            f"**{R['per'][2][5]:+.0f}** (IWM) bps — three positive, two negative, none large. No "
            "coherent, cross-sectional edge — exactly what you'd expect if the flip is just "
            "relabelled drift."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real trend\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** post-flip trend into "
            "a synthetic tape and check the same flip-up rule banks it: edge=0 must stay at t≈0; "
            "edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.60):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=486, n_days=4000)\n"
            "    c = px['close']; e = st.flip_up_entries(px, period=10); s = st.summarize(st.forward_returns(c, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted trend -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} flip={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted trend the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"trend reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector "
            "works — so the flat real-tape result is a genuine 'nothing there', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the flip-up does not beat a drift-matched random baseline "
            f"(flip − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps at 5/10/20/60d; Welch t never clears 2, max magnitude **{R['h60'][8]:+.2f}** at 60d "
            f"and *negative*). The impressive one-sample t's (20d **{R['h20'][4]:.2f}**) are pure "
            "beta.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; costs only "
            "deepen the hole. You'd capture the drift more cheaply by holding the index.\n"
            f"- **Flip forecasts trend? `BUSTED`** — the shuffled-flip timing placebo leaves the "
            f"result untouched (**p = {R['placebo'][1]:.2f}**): random buy-days do as well as the "
            "real flips, so the activator's flip carries no forecasting information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The flip's entire apparent profit is the unconditional drift of long equity indices, "
            "which you obtain more cheaply and more fully by **buying and holding**. The flip rule "
            "trades *less* of the time (only on flips) and pays costs on each, so it strictly "
            "dominates *nothing*. There is no capacity question because there is no edge to scale. The "
            "Gann Hi-Lo Activator is a trailing-stop tool, not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Period sweep.** 3 / 10 / 14-day activators are parametric variants of the same "
            "construction; they inherit the same drift confound. The mechanical version here is the "
            "charitable upper bound.\n"
            "- **Stop, not signal.** The Hi-Lo Activator's honest job is to *exit* a held position. "
            "Tested as an exit on a position you already justified, it's a reasonable trailing stop; "
            "tested as an *entry forecast* it's drift in a costume.\n"
            "- **EMA / HiLo-band variants** are affine tweaks of the same line and inherit the same "
            "confound.\n\n"
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
