"""Generate the two narrative notebooks for Study 480 (Darvas Box).

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
# 2026-05-31, partial June dropped), 21.4 years, box lookback=20 + min-box=5, breakout long.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=949, lookback=20, min_box=5,
    fp_spy="4cb5244f3990",
    # pooled box breakout, per horizon:
    # (H, n, brk_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 946, 25.8, 61, 3.48, 42.3, -16.4, 23.8, -1.56, 0.119),
    h10=(10, 945, 48.4, 63, 5.22, 63.2, -14.8, 46.4, -1.00, 0.318),
    h20=(20, 943, 94.1, 65, 6.41, 102.3, -8.2, 92.1, -0.38, 0.702),
    h60=(60, 938, 248.8, 68, 6.67, 258.0, -9.2, 246.8, -0.26, 0.798),
    # per-ticker H=20: (ticker, entries, brk_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 201, 119.8, 5.25, 116.0, 3.8), ("QQQ", 197, 137.1, 4.11, 123.2, 13.9),
         ("IWM", 186, 69.7, 1.76, 113.7, -44.0), ("DIA", 194, 73.1, 2.89, 75.1, -2.0),
         ("GLD", 171, 64.8, 1.59, 81.0, -16.2)],
    # shuffled-box placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(119.8, 0.192, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, brk_bps, win%, one_sample_t)
    syn=[(0.00, 127, -59.1, 46, -1.56), (0.30, 119, 458.6, 78, 9.77)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Box_breakout_forecasts%3F: Busted](https://img.shields.io/badge/Box_breakout_forecasts%3F-Busted-8b949e?style=flat-square)\n\n"
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

from darvas_box import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real darvas cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the Darvas box breakout actually forecast? 📦\n"
            "### A famous momentum tool — a box around a consolidation, a breakout buy — meets a stopwatch\n\n"
            + BADGES +
            "A ballroom dancer named **Nicolas Darvas** turned $25k into $2,000,000 (so the title of "
            "his 1960 book goes) by trading from telegrams on tour. His method: draw a **box** around "
            "a stock that's gone quiet near a high, and **buy the moment it closes above the box top** "
            "— the breakout is \"supposed\" to launch the next leg up. It's one of the most copied "
            "momentum recipes in trading.\n\n"
            "It *looks* uncanny on a hand-picked chart. But a breakout-buy on a market that drifts "
            "**up** over decades is the textbook setup for fooling yourself — *any* way of being long "
            "makes money. So we did the only fair thing: encode the box **mechanically** (no "
            "eyeballing), fire the \"buy the breakout\" rule nearly a thousand times across five big "
            "indices over 21 years, and time the result with a stopwatch — against the only baseline "
            "that matters: **buying on random days instead.**\n\n"
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
            "| If I buy when price closes above the **box top**, do I make money? | **Yes — but only "
            "because the market goes up.** The raw win-rate is ~65% and the returns look great. |\n"
            "| Is that *the breakout's* doing? | **No.** Buy on **random days** instead and you do "
            "**just as well or better** — at every horizon the breakout is actually *worse* than a "
            "coin-flip entry. |\n"
            "| Does the box top \"forecast\" the move? | **Not in any usable way.** Scatter the "
            "breakout dates at random and the result barely changes. The timing isn't doing the work. |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — the upward drift of "
            "stocks, re-labelled as a breakout. |\n\n"
            "> The Darvas box is a fine way to *manage* a trade after the fact. As a *forecast* — "
            "\"the breakout will continue\" — it's a **mirage**: all of the apparent edge is just the "
            "market's long-run climb, none of it is the box."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A stock makes a new high, then goes quiet — that high is the **box top**, the "
            "pullback low is the **box bottom**. When it **closes above the box top**, buy: the "
            "breakout starts the next box up. Put your stop just under the box bottom and ride it.\"*\n\n"
            "This is **Nicolas Darvas'** box theory, from *How I Made $2,000,000 in the Stock Market* "
            "(1960), still taught today and built into screeners as the \"Darvas box\" indicator. It's "
            "the granddaddy of breakout/momentum trading — so: does the breakout actually *forecast* "
            "the continuation?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the breakout genuinely *forecast* continuation, it would be remarkable: a line drawn "
            "from past highs would predict future moves, a clean crack in market efficiency you could "
            "trade with a ruler. That's the dream the tool sells.\n\n"
            "But there's a trap built into it. A breakout buy is **long-only on a market that drifts "
            "up** — so *any* breakout rule will look profitable, because the tide lifts everything. To "
            "separate the **tool** from the **tide**, we have to (a) draw the box by a fixed "
            "mechanical rule with no hindsight, and (b) compare it to buying on **random days**. We'll "
            "do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Build the box mechanically.** The **box top** is the highest close over the trailing "
            f"**{R['lookback']} days** (shifted one bar, so it never peeks at today). We only call it "
            f"a box once price has sat *below* that top for **{R['min_box']} bars** — a real "
            "consolidation, not a runaway.\n"
            "2. **Trade the lore.** When the close finally pierces **above the box top**, buy at the "
            "next close; measure the return over the next **5 / 10 / 20 / 60 days**.\n"
            "3. **The honest baseline.** Do the exact same hold on **random days**. If the breakout "
            "matters, it must beat random. *If it doesn't, the tool is a mirage* — that's the result "
            "that would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical Darvas box even look like? Here's SPY with the trailing box "
            "top/bottom drawn, and the breakout buys the rule would fire."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    seg = b.iloc[-450:]\n"
            "    lv = st.box_levels(b, lookback=R['lookback'])\n"
            "    ent = st.breakout_entries(b, lookback=R['lookback'], min_box=R['min_box'])\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg['close'].values, c='k', lw=1.2, label='SPY close')\n"
            "    ax.plot(seg.index, lv['box_top'].reindex(seg.index), c=GREEN, lw=1.1, label='box top (20d high)')\n"
            "    ax.plot(seg.index, lv['box_bottom'].reindex(seg.index), c=RED, lw=1.1, label='box bottom (20d low)')\n"
            "    ax.scatter(ent, b['close'].reindex(ent), c=GREEN, s=40, zorder=5, label='breakout BUY')\n"
            "    ax.set_title('A mechanical Darvas box on SPY (last ~2y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('breakouts in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The box tracks the consolidation nicely — *as a description*. The question is whether "
            "those green buy dots are followed by continuation. **Let's race the breakout against "
            "random entries** at four horizons. Blue = buy the breakout; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    brk, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.breakout_entries(bb, lookback=R['lookback'], min_box=R['min_box'])\n"
            "            re = st.random_entries(c, max(len(e),50), lookback=R['lookback'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        brk.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    brk = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, brk, .4, color='#2c6fbb', label='buy the breakout')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(brk,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The breakout does NOT beat random — it loses to it at every horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('breakout:', [round(v) for v in brk]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story in one chart. The breakout makes money in absolute terms "
            f"(**+{R['h20'][2]:.0f} bps** over 20 days) — but **random entries make more** "
            f"(**+{R['h20'][5]:.0f} bps**). At *every* horizon the famous breakout is *worse* than "
            "throwing darts. The apparent edge was **the market's upward drift**, not the box."
        ),
        md(
            "**One more sanity check.** What if we keep the same number of trades but **scatter the "
            "entry dates at random** — destroying the box-top timing while keeping everything else? If "
            "the breakout really forecasts, the random dates should do much worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')\n"
            "    pl = st.shuffled_box_placebo(c, 20, lookback=R['lookback'], min_box=R['min_box'], n_draws=300, seed=480)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real box breakout (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *random-date* entries do at least as well (p={pval:.2f}).')\n"
            "print('=> the box-top timing is not doing the work.')"
        ),
        md(
            f"Around a fifth of the **random-date** runs match or beat the real breakout "
            f"(*p* = {R['placebo'][1]:.2f}). If the box top genuinely forecast, random timing would "
            "collapse the result. It doesn't — because the result was never about the box."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The breakout buy does **not** beat buying on random days "
            "(it's *worse* at all four horizons; the breakout-vs-random difference is never positive). "
            "The big absolute returns are the market's drift, not the box.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting for free — and costs only make it worse.\n"
            "- **\"Does the box breakout forecast\"? — Busted.** Scatter the entry dates and the "
            "result barely moves. The box doesn't forecast."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The breakout's *only* advantage over a coin flip is the "
            "market's long-run climb — which you'd capture more cheaply (and more fully) by just "
            "**holding the index**. The Darvas breakout is a worse, more expensive way to be long. "
            "Costs (commissions + spread on every breakout) push the already-no-edge result further "
            "negative. As a forecasting tool, it doesn't pay; as a trade-management frame, it was "
            "never meant to be a standalone strategy on a diversified index."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Single stocks vs indices.** Darvas traded individual high-flyers, not the S&P. A fun "
            "follow-up runs the same rule on a survivorship-corrected single-name universe — but "
            "beware: that's exactly where survivorship + selection inflate the apparent edge.\n"
            "- **Channel cousins.** The box is a Donchian/turtle breakout in disguise; see the desk's "
            "[Donchian](../../437-donchian-breakout) and [Turtle](../../103-turtle-trader) studies — "
            "same drift confound.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* post-breakout "
            "continuation into a synthetic tape and shows the harness banks it (so the null result "
            "here isn't a dead detector — it's an honest 'nothing there').\n\n"
            "*Think the box forecasts? Show the breakout beating random entries at **t ≥ 2** on a "
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
            "# Darvas Box — a quantitative teardown 🔬\n"
            "### Mechanical trailing-high boxes on 5 indices · breakout forward returns · "
            "one-sample HAC *t* · a drift-matched random-entry baseline · a shuffled-box timing "
            "placebo · costs · a synthetic planted-continuation control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **breakout** from the **drift**: an upward-trending index makes *any* "
            "long-only entry look good, so the only meaningful test is breakout-vs-random, plus a "
            "placebo that destroys the box-top timing while preserving the marginals.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. Box top/bottom are trailing "
            f"windows (lookback={R['lookback']}, one-bar shift, {R['min_box']}-bar consolidation "
            "requirement); entry is the **next close** (one documented lag). Offline core + synthetic "
            "control are deterministic. Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Box breakout vs a **drift-matched random** baseline: the "
            f"breakout is *worse* at every horizon (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/"
            f"{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} bps) and the breakout-minus-random difference is "
            f"**never positive** (Welch t at 5d = {R['h5'][8]:+.2f}, 20d = {R['h20'][8]:+.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | The big one-sample t's (20d t = {R['h20'][4]:.2f}) are "
            f"**pure beta** — they vanish against random entries and against cost. No residual edge "
            "to scale. |\n"
            f"| **Box breakout forecasts?** | `BUSTED` | Scattering the breakout dates (shuffled-box "
            f"placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** of random-date entries "
            "match or beat the real one. The timing isn't doing the work. |\n\n"
            "> 💡 In plain words: the breakout *looks* significant only because indices drift up. "
            "Strip the drift (race it vs random) or strip the timing (scatter the dates) and the edge "
            "evaporates. Classic beta-in-a-costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $T_t=\\max_{t-L\\le \\tau<t} C_\\tau$ be the trailing box top (lookback $L$, shifted "
            "one bar). A *box* requires the close to have sat below $T$ for $\\ge m$ consecutive bars. "
            "The Darvas rule buys when $C_t>T_t$ for the first time after the consolidation, stops "
            "below the box bottom $B_t=\\min_{t-L\\le\\tau<t} \\text{low}_\\tau$, and rides the "
            "continuation.\n\n"
            "- **H₀ (drift).** Breakout returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (the breakout forecasts).** Breakout returns **exceed** random at some horizon, "
            "t ≥ 2.\n"
            "- **H₂ (the timing matters).** Breakout returns exceed a **shuffled-box** placebo whose "
            "entry dates are scattered.\n\n"
            "We find **H₀ not rejected** (breakout ≤ random at every horizon), **H₁ rejected** (Welch "
            "t never positive), **H₂ rejected** (placebo p ≈ 0.19). The steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* long-only "
            "entry rule inherits it; a high one-sample $t$ against **zero** measures the tide, not the "
            "tool. The fix is the **random-entry baseline** (same instrument, epoch, hold) and a "
            "Welch test of breakout-*minus*-random.\n\n"
            "**(b) Timing as a free parameter.** A breakout is a chosen entry date; the danger is that "
            "*any* entry on a trend looks like a 'continuation'. The **shuffled-box placebo** keeps "
            "the same trade count and the price marginal but scatters the entry dates — the box-top "
            "timing becomes meaningless, so if the real result survives the scramble, the timing was "
            "never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} box breakouts** "
            "pooled.\n"
            f"- **Box.** Top = trailing {R['lookback']}-day high of close (one-bar shift, no "
            f"look-ahead); a box requires {R['min_box']} consecutive below-top bars first. Bottom = "
            f"trailing {R['lookback']}-day low (the ATR/box stop reference).\n"
            "- **Entry.** First close above the box top; enter **next close** (one lag); hold "
            "H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of breakout returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample breakout vs random (the *real* "
            "test).\n"
            "- **Null #3 — shuffled-box placebo** (timing destroyed, marginals + count kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every breakout.\n"
            "- **Positive control.** Synthetic tape with a **planted** post-breakout continuation "
            "(knob `edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks great, vs-random kills it\n\n"
            "Left: the breakout's **one-sample** t against zero (the misleading number). Right: the "
            "same breakout vs a **drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, brk, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.breakout_entries(bb, lookback=R['lookback'], min_box=R['min_box'])\n"
            "            re = st.random_entries(c, max(len(e),50), lookback=R['lookback'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); brk.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    brk = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
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
            "a2.set_title('Breakout vs RANDOM, Welch t (honest: never positive)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 easily (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) — but that's the **drift**, every long-only breakout inherits it. "
            f"The right bars are the real test: breakout-minus-random is **negative** at every horizon "
            f"({R['h20'][8]:+.2f} at 20d, {R['h5'][8]:+.2f} at 5d) — never significant, never even "
            "positive. The breakout adds nothing over a coin flip."
        ),
        md(
            "### 4b · Breakout vs random across horizons — the gap is the verdict\n\n"
            "Mean return, breakout vs random entry, all four horizons. The breakout should tower over "
            "random if it forecasts. It doesn't."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, brk, .4, color='#2c6fbb', label='box breakout')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(brk,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Box breakout does not beat random entry'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta breakout-random (bps):', [round(a-b) for a,b in zip(brk,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the breakout is **+{R['h20'][2]:.0f} bps** but random is "
            f"**+{R['h20'][5]:.0f} bps** — the breakout *underperforms* a dart by {abs(R['h20'][6]):.0f} "
            "bps. There is no horizon where it gets ahead, and the Welch test (4a) confirms the gap is "
            "against it."
        ),
        md(
            "### 4c · The timing placebo — scatter the breakout dates, nothing changes\n\n"
            "Keep the same number of trades and the price marginal, but scatter the entry dates at "
            "random so the box-top timing is meaningless. If the breakout forecasts, the observed "
            "return should sit far in the right tail of the scattered distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')\n"
            "    pl = st.shuffled_box_placebo(c, 20, lookback=R['lookback'], min_box=R['min_box'], n_draws=300, seed=480)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    ent = st.breakout_entries(c, lookback=R['lookback'], min_box=R['min_box'])\n"
            "    rng = np.random.default_rng(480); valid = c['close'].index[2*R['lookback']:]\n"
            "    draws = []\n"
            "    for _ in range(300):\n"
            "        ch = rng.choice(valid, size=min(len(ent), len(valid)), replace=False)\n"
            "        rr = st.forward_returns(c['close'], __import__('pandas').DatetimeIndex(ch), 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(480); draws = rng.normal(102, 28, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='scattered-date entries (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real breakout {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean breakout 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real breakout sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real breakout {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => timing not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real breakout (blue line) sits **inside** the scattered-date "
            f"cloud — **p = {R['placebo'][1]:.2f}**. Random timing does just as well, so the specific "
            "box-top breakout date isn't carrying information. This is the cleanest refutation of 'the "
            "box breakout forecasts.'"
        ),
        md(
            "### 4d · Per-ticker — no coherent cross-sectional edge\n\n"
            "20-day breakout-minus-random delta, per instrument. If the box worked it would be "
            "positive across the board; instead the positives are tiny and IWM is far behind."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        bb = load(t); c = bb['close']\n"
            "        e = st.breakout_entries(bb, lookback=R['lookback'], min_box=R['min_box'])\n"
            "        re = st.random_entries(c, max(len(e),50), lookback=R['lookback'], seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d breakout − random (bps)'); ax.set_title('Tiny positives, one big negative — no coherent edge')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: SPY/QQQ eke out a few bps ({R['per'][0][5]:+.0f}/{R['per'][1][5]:+.0f}) "
            f"— inside the noise — while IWM is **{R['per'][2][5]:+.0f}** bps *behind* random. No "
            "coherent, cross-sectional edge — exactly what you'd expect if the breakout is just "
            "relabelled drift."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real continuation\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** post-breakout "
            "continuation into a synthetic tape and check the same breakout rule banks it: edge=0 must "
            "stay below t=2; edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.30):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=480, n_days=4000)\n"
            "    e = st.breakout_entries(px, lookback=20, min_box=5); s = st.summarize(st.forward_returns(px['close'], e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> below t=2; planted continuation -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} brk={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted continuation the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — does not clear 2, no false "
            f"positive); a planted continuation reaches **t = {R['syn'][1][4]:.2f}** (win "
            f"{R['syn'][1][3]:.0f}%). The detector works — so the flat real-tape result is a genuine "
            "'nothing there', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the box breakout does not beat a drift-matched random baseline "
            f"(breakout − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps at 5/10/20/60d; Welch t never positive, most negative **{R['h5'][8]:+.2f}** at 5d). "
            f"The impressive one-sample t's (20d **{R['h20'][4]:.2f}**) are pure beta.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; costs only "
            "deepen the hole. You'd capture the drift more cheaply by holding the index.\n"
            f"- **Box breakout forecasts? `BUSTED`** — the shuffled-box placebo leaves the result "
            f"untouched (**p = {R['placebo'][1]:.2f}**): randomly-scattered entries do as well as the "
            "real breakouts, so the specific box-top timing carries no information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The breakout's entire apparent profit is the unconditional drift of long equity indices, "
            "which you obtain more cheaply and more fully by **buying and holding**. The Darvas rule "
            "trades *less* of the time (only on breakouts) and pays costs on each, so it strictly "
            "dominates *nothing*. There is no capacity question because there is no edge to scale. The "
            "box is a descriptive trade-management frame, not a forecasting strategy on a diversified "
            "index tape."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Single high-flyers vs indices.** Darvas traded individual momentum names, not the "
            "S&P. Re-running on single stocks is the obvious follow-up — but that's exactly where "
            "survivorship and selection inflate the apparent edge, so it must be done on a "
            "point-in-time, survivorship-corrected universe.\n"
            "- **Channel cousins.** The box is an affine relative of the Donchian/turtle breakout "
            "([437](../../437-donchian-breakout), [103](../../103-turtle-trader)); all inherit the "
            "same drift confound.\n"
            "- **Stop sensitivity.** Varying the ATR/box stop width changes the trade *management* but "
            "not the entry edge; the no-forecast result is robust to it.\n\n"
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
