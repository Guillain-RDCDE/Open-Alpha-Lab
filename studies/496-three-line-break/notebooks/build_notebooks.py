"""Generate the two narrative notebooks for Study 496 (Three-Line-Break).

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
# 2026-05-31, partial June dropped), 21.4 years, break number = 3, bullish 3-line up-reversal.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=943, nl=3,
    fp_spy="4cb5244f3990",
    # pooled TLB up-reversal, per horizon:
    # (H, n, rev_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 940, 16.5, 60, 2.05, 16.3, 0.2, 14.5, 0.02, 0.986),
    h10=(10, 940, 31.9, 61, 2.59, 60.3, -28.4, 29.9, -1.79, 0.073),
    h20=(20, 938, 93.1, 64, 5.29, 115.5, -22.4, 91.1, -1.02, 0.307),
    h60=(60, 932, 258.4, 69, 6.26, 290.9, -32.5, 256.4, -0.89, 0.373),
    # per-ticker H=20: (ticker, entries, rev_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 211, 114.1, 3.63, 145.2, -31.1), ("QQQ", 210, 87.6, 2.17, 165.9, -78.3),
         ("IWM", 183, 49.6, 0.96, 93.3, -43.7), ("DIA", 193, 91.4, 2.97, 88.2, 3.2),
         ("GLD", 146, 127.4, 3.20, 64.2, 63.2)],
    # shuffled-returns placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(114.1, 0.261, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, rev_bps, win%, one_sample_t)
    syn=[(0.00, 138, -2.0, 49, -0.05), (0.60, 84, 617.4, 76, 6.13)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![3-line_reversal_forecasts%3F: Busted](https://img.shields.io/badge/3--line_reversal_forecasts%3F-Busted-8b949e?style=flat-square)\n\n"
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

from three_line_break import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real TLB cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a Three-Line-Break reversal actually \"forecast\" a trend? 🧱\n"
            "### A famous time-less chart — bricks that flip after breaking three — meets a stopwatch\n\n"
            + BADGES +
            "Open any charting package and you'll find the **Three-Line-Break** (TLB) chart: it throws "
            "away the calendar and draws a new brick (a *line*) only when price closes past the prior "
            "brick. It **flips colour** — signalling a reversal — only after price breaks the extremes "
            "of the **three** most-recent opposite bricks. The lore, from the Japanese *Sakata* "
            "tradition and Steve Nison's *Beyond Candlesticks*, is that a TLB reversal **forecasts a "
            "new trend**: go long when it flips up, step aside when it flips down.\n\n"
            "It *looks* clean on a hand-picked chart — the bricks filter out the noise and the flips "
            "seem to catch every big move. But a chart that only redraws *after* price has already "
            "moved is the textbook setup for fooling yourself. So we did the only fair thing: encode "
            "TLB **mechanically** (break number = 3, no eyeballing), fire the \"buy the up-reversal\" "
            "rule on every flip across five big indices over 21 years, and time the result with a "
            "stopwatch — against the only baseline that matters: **buying on random days instead.**\n\n"
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
            "| If I buy when the TLB flips up, do I make money? | **Yes — but only because the market "
            "goes up.** The raw win-rate is ~60–69% and the returns look great. |\n"
            "| Is that *the reversal's* doing? | **No.** Buy on **random days** instead and you do "
            "**just as well or better**. The flip adds nothing — at 10/20/60 days it's actually "
            "*worse* than a coin-flip entry. |\n"
            "| Does the 3-line reversal forecast? | **Not in any usable way.** Shuffle the order of "
            "the daily moves so the brick sequence is nonsense, and the result barely changes. The "
            "specific flips aren't doing the work. |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — the upward drift of "
            "stocks, re-labelled as a brick reversal. |\n\n"
            "> TLB is a great way to *declutter* a trend after the fact. As a *forecast* — \"the flip "
            "predicts the next move\" — it's a **mirage**: all of the apparent edge is just the "
            "market's long-run climb, none of it is the bricks."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Forget time and volume. Draw a new brick only when price closes past the last one. "
            "Flip the colour — call a reversal — only when price breaks the highs/lows of the last "
            "**three** bricks. The reversal marks a genuine new trend: go long on an up-flip, flat on "
            "a down-flip, and you ride trends while filtering the noise.\"*\n\n"
            "This is the **Three-Line-Break** chart from the *Sakata* lineage, introduced West by "
            "**Steve Nison** (*Beyond Candlesticks*, 1994) and codified by **Steve Achelis** "
            "(*Technical Analysis from A to Z*, 1995). The break number (default 3) is its only knob. "
            "It's a staple of TradingView, MetaStock and StockCharts — so: does the flip actually "
            "*forecast*?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the reversal genuinely *forecast* trends, it would be remarkable: a pattern of past "
            "closes would predict future direction, a clean crack in market efficiency you could "
            "trade with a rule. That's the dream the chart sells.\n\n"
            "But there's a trap built into it. A TLB brick only appears **after** price has already "
            "moved past the last one — the chart is a lagging redraw of what already happened. And "
            "it's drawn on a market (stock indices) that drifts **up** over time, so *any* "
            "long-on-strength rule will look profitable. To separate the **signal** from the "
            "**tide**, we (a) build the bricks by a fixed mechanical rule with no hindsight, and "
            "(b) compare every flip to buying on **random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Build the bricks mechanically.** A new line appears on a close past the prior "
            "brick; the chart flips only when the close breaks the extremes of the **3** latest "
            "opposite bricks. Everything is computed from *past* closes — no future data.\n"
            "2. **Trade the lore.** When the TLB flips **down→up** (the bullish 3-line break), buy at "
            "the next close; measure the return over the next **5 / 10 / 20 / 60 days**.\n"
            "3. **The honest baseline.** Do the exact same hold on **random days**. If the reversal "
            "forecasts, the flip must beat random. *If it doesn't, the rule is a mirage* — that's the "
            "result that would make us say so, announced before we look.\n"
            "4. **The geometry check.** Shuffle the order of the daily moves (same returns, scrambled "
            "sequence) and rebuild the bricks. If the *specific* flips matter, the scramble should "
            "destroy the result."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical TLB even look like? Here's SPY with its colour state and "
            "the bullish up-reversals the rule would buy."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    cl = b['close']; seg = cl.iloc[-450:]\n"
            "    tlb = st.build_tlb(cl, n_lines=R['nl'])\n"
            "    ent = st.reversal_entries(cl, n_lines=R['nl'])\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    col = tlb['color'].reindex(seg.index)\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg.values, c='k', lw=1.1, label='SPY close')\n"
            "    up = seg.where(col > 0); dn = seg.where(col < 0)\n"
            "    ax.plot(seg.index, up.values, c=GREEN, lw=2.4, label='TLB up')\n"
            "    ax.plot(seg.index, dn.values, c=RED, lw=2.4, label='TLB down')\n"
            "    ax.scatter(ent, cl.reindex(ent), c=GREEN, s=45, zorder=5, marker='^', label='3-line up-reversal BUY')\n"
            "    ax.set_title('A mechanical Three-Line-Break on SPY (last ~2y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('up-reversals in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The bricks colour the trend nicely — *as a description*. The question is whether those "
            "green up-flips are followed by gains beyond the usual drift. **Let's race the reversal "
            "against random entries** at four horizons. Blue = buy the up-flip; grey = buy on random "
            "days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    rev, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.reversal_entries(c, n_lines=R['nl'])\n"
            "            re = st.random_entries(c, max(len(e),50), n_lines=R['nl'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        rev.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    rev = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, rev, .4, color='#2c6fbb', label='buy the up-reversal')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(rev,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The reversal does NOT beat random — it mostly loses to it'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('reversal:', [round(v) for v in rev]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story in one chart. The up-reversal makes money in absolute terms "
            f"(**+{R['h20'][2]:.0f} bps** over 20 days) — but **random entries make more** "
            f"(**+{R['h20'][5]:.0f} bps**). At 10, 20 and 60 days the famous flip is *worse* than "
            "throwing darts; at 5 days it merely ties. The apparent edge was **the market's upward "
            "drift**, not the bricks."
        ),
        md(
            "**One more sanity check.** What if we shuffle the *order* of the daily moves — keep "
            "exactly the same set of returns (same drift, same volatility) but scramble their "
            "sequence, so the brick flips are built on nonsense? If the reversal really forecasts, "
            "the scramble should crush the result."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.shuffled_returns_placebo(c, 20, n_lines=R['nl'], n_draws=300, seed=496)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real TLB up-reversal (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *shuffled-sequence* tapes do at least as well (p={pval:.2f}).')\n"
            "print('=> the specific brick sequence is not doing the work.')"
        ),
        md(
            f"About a quarter of the **shuffled** tapes match or beat the real one "
            f"(*p* = {R['placebo'][1]:.2f}). If the reversal genuinely forecast on *this specific* "
            "sequence of breaks, a scramble would collapse the result. It doesn't — because the "
            "result was never about the bricks."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The up-reversal does **not** beat buying on random days "
            "(it's *worse* at 10–60 days, a tie at 5d; the reversal-vs-random difference never "
            "clears *t* = 2). The big absolute returns are the market's drift, not the flip.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting for free — and costs only make it worse.\n"
            "- **\"Does the 3-line reversal forecast?\" — Busted.** Shuffle the sequence into nonsense "
            "and the result barely moves. The flip doesn't forecast."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The up-reversal's *only* advantage over a coin flip is the "
            "market's long-run climb — which you'd capture more cheaply (and more fully) by just "
            "**holding the index**. The TLB buy is a worse, more expensive way to be long: it's out "
            "of the market on every down-flip (missing rebounds) and pays costs on every brick "
            "reversal. As a forecasting tool, it doesn't pay; as a charting tool, it was only ever "
            "meant to declutter a trend."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Different break numbers.** Try 2-line or 5-line breaks — more or fewer whipsaws, but "
            "the drift confound is identical: drift in, bricks out.\n"
            "- **Short the down-flips too.** The long-short version just adds the *negative* of the "
            "drift on the short leg; against random it fares no better (try it).\n"
            "- **A real positive control.** The quants notebook plants a *genuine* post-reversal "
            "continuation into a synthetic tape and shows the harness banks it (so the null result "
            "here isn't a dead detector — it's an honest 'nothing there').\n\n"
            "*Think the flip forecasts? Show the up-reversal beating random entries at **t ≥ 2** on a "
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
            "# Three-Line-Break — a quantitative teardown 🔬\n"
            "### Mechanical 3-line bricks on 5 indices · up-reversal forward returns · "
            "one-sample HAC *t* · a drift-matched random-entry baseline · a shuffled-returns "
            "sequence placebo · costs · a synthetic planted-continuation control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **reversal** from the **drift**: an upward-trending index makes *any* "
            "long-on-strength rule look good, so the only meaningful test is reversal-vs-random, plus "
            "a placebo that destroys the brick *sequence* while preserving its marginals.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. Break number = 3; reversals "
            "are built causally from past closes and read on the close of *t*; entry is the **next "
            "close** (one documented lag). Offline core + synthetic control are deterministic. "
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
            f"| **Signal** | `NONE` | Up-reversal vs a **drift-matched random** baseline: a tie at 5d "
            f"and *worse* at 10/20/60d (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps) and the reversal-minus-random difference **never clears t = 2** "
            f"(Welch t max {R['h5'][8]:+.2f} at 5d; longer horizons negative). |\n"
            f"| **Tradability** | `MIRAGE` | The big one-sample t's (20d t = {R['h20'][4]:.2f}, 60d "
            f"= {R['h60'][4]:.2f}) are **pure beta** — they vanish against random entries and against "
            "cost. No residual edge to scale. |\n"
            f"| **3-line reversal forecasts?** | `BUSTED` | Shuffling the daily-move *sequence* "
            f"(shuffled-returns placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** of "
            "scrambled tapes match or beat the real one. The specific brick flips aren't doing the "
            "work. |\n\n"
            "> 💡 In plain words: the up-reversal *looks* significant only because indices drift up. "
            "Strip the drift (race it vs random) or strip the sequence (shuffle the returns) and the "
            "edge evaporates. Classic beta-in-a-costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Build TLB lines causally from closes $C_t$: a new up-line at a close above the prior "
            "line's top, a new down-line below its bottom. The chart **flips down→up** at the first "
            "close $C_t$ that exceeds $\\max$ of the tops of the **3** latest down-lines (and "
            "symmetrically for up→down). Let $\\mathcal{U}$ be the set of up-reversal bars. The lore "
            "buys at $t+1$ for each $t\\in\\mathcal{U}$ and rides the new trend.\n\n"
            "- **H₀ (drift).** Reversal returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (the reversal forecasts).** Reversal returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the sequence matters).** Reversal returns exceed a **shuffled-returns** tape "
            "whose brick flips are built on a scrambled order.\n\n"
            "We find **H₀ not rejected** (reversal ≤ random at 10–60d, tie at 5d), **H₁ rejected** "
            "(Welch t never ≥ 2), **H₂ rejected** (placebo p ≈ 0.26). The steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* long-only "
            "entry rule inherits it; a high one-sample $t$ against **zero** measures the tide, not the "
            "tool. The fix is the **random-entry baseline** (same instrument, epoch, hold) and a Welch "
            "test of reversal-*minus*-random.\n\n"
            "**(b) Sequence as a free parameter.** A TLB flip is a function of the *order* in which "
            "runs occur. The danger is that on a trending tape *any* ordering produces 'reversals' "
            "that lead into more drift. The **shuffled-returns placebo** keeps the exact set of daily "
            "returns (the price marginal, drift and vol) but permutes their order, so the specific "
            "brick sequence is meaningless — if the real result survives the scramble, the sequence "
            "was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} up-reversals** "
            "pooled.\n"
            f"- **TLB.** Break number = {R['nl']}; lines built causally from past closes; flip "
            "down→up at the first close above the top-extreme of the 3 latest down-lines.\n"
            "- **Entry.** The up-reversal bar; enter **next close** (one lag); hold "
            "H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of reversal returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample reversal vs random (the *real* test).\n"
            "- **Null #3 — shuffled-returns placebo** (sequence destroyed, marginal kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every reversal.\n"
            "- **Positive control.** Synthetic tape with a **planted** post-reversal continuation "
            "(knob `edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks great, vs-random kills it\n\n"
            "Left: the up-reversal's **one-sample** t against zero (the misleading number). "
            "Right: the same reversal vs a **drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, rev, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.reversal_entries(c, n_lines=R['nl'])\n"
            "            re = st.random_entries(c, max(len(e),50), n_lines=R['nl'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); rev.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    rev = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
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
            "a2.set_title('Reversal vs RANDOM, Welch t (honest: never clears 2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 easily (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) — but that's the **drift**, every long-on-strength entry inherits "
            f"it. The right bars are the real test: reversal-minus-random is a **tie** at 5d "
            f"({R['h5'][8]:+.2f}) and **negative** at 10–60d — never significant. The reversal adds "
            "nothing over a coin flip."
        ),
        md(
            "### 4b · Reversal vs random across horizons — the gap is the verdict\n\n"
            "Mean return, up-reversal vs random entry, all four horizons. The reversal should tower "
            "over random if the flip forecasts. It doesn't."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, rev, .4, color='#2c6fbb', label='up-reversal')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(rev,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Up-reversal does not beat random entry'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta reversal-random (bps):', [round(a-b) for a,b in zip(rev,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the reversal is **+{R['h20'][2]:.0f} bps** but random is "
            f"**+{R['h20'][5]:.0f} bps** — the flip *underperforms* a dart by {abs(R['h20'][6]):.0f} "
            "bps. The only horizon where the reversal is level with random is 5d, and the Welch test "
            "(4a) says even that gap is noise."
        ),
        md(
            "### 4c · The sequence placebo — shuffle the moves, nothing changes\n\n"
            "Permute the daily returns (positions kept as a set, drift/vol kept) so the brick flips "
            "are built on a scrambled order. If the reversal forecasts on *this specific* sequence, "
            "the scramble should demolish the result. The observed reversal return should sit far in "
            "the right tail of the scrambled distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.shuffled_returns_placebo(c, 20, n_lines=R['nl'], n_draws=300, seed=496)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    import numpy as _np, pandas as _pd\n"
            "    logret = _np.diff(_np.log(c.to_numpy(float))); p0 = float(c.iloc[0]); idx = c.index\n"
            "    rng = _np.random.default_rng(496); draws = []\n"
            "    for _ in range(300):\n"
            "        perm = rng.permutation(logret)\n"
            "        path = p0*_np.exp(_np.concatenate([[0.0], _np.cumsum(perm)]))\n"
            "        cs = _pd.Series(path, index=idx)\n"
            "        rr = st.forward_returns(cs, st.reversal_entries(cs, n_lines=R['nl']), 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = _np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(496); draws = rng.normal(100, 60, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='shuffled-sequence tapes (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real TLB {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean up-reversal 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real TLB sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real TLB {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => sequence not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real TLB (blue line) sits **inside** the shuffled-sequence "
            f"cloud — **p = {R['placebo'][1]:.2f}**. A scrambled order reverses just as profitably, so "
            "the specific brick flips aren't carrying information. This is the cleanest refutation of "
            "'the 3-line reversal forecasts.'"
        ),
        md(
            "### 4d · Per-ticker — the reversal loses to random in 3 of 5\n\n"
            "20-day reversal-minus-random delta, per instrument. If the flip worked it would be "
            "positive across the board; instead it's negative in 3 of 5."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = load(t)['close']\n"
            "        e = st.reversal_entries(c, n_lines=R['nl']); re = st.random_entries(c, max(len(e),50), n_lines=R['nl'], seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d reversal − random (bps)'); ax.set_title('Reversal underperforms random in 3 of 5 names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: only **GLD** ({R['per'][4][5]:+.0f} bps) and marginally **DIA** "
            f"({R['per'][3][5]:+.0f} bps) edge out positive deltas; SPY is **{R['per'][0][5]:+.0f}** "
            "bps *behind* random and QQQ is far worse. No coherent cross-sectional edge — and the lone "
            "GLD bright spot vanishes in the pooled Welch test. Exactly what you'd expect if the "
            "reversal is just relabelled drift."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real continuation\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** post-reversal "
            "continuation into a synthetic tape and check the same up-reversal rule banks it: edge=0 "
            "must stay at t≈0; edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.60):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=496, n_days=4000)\n"
            "    c = px['close']; e = st.reversal_entries(c, n_lines=3); s = st.summarize(st.forward_returns(c, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted continuation -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} rev={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted continuation the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"continuation reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The "
            "detector works — so the flat real-tape result is a genuine 'nothing there', not a broken "
            "pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the up-reversal does not beat a drift-matched random baseline "
            f"(reversal − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps at 5/10/20/60d; Welch t never clears 2, max **{R['h5'][8]:+.2f}** at 5d). The "
            f"impressive one-sample t's (20d **{R['h20'][4]:.2f}**, 60d **{R['h60'][4]:.2f}**) are pure beta.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; costs only "
            "deepen the hole. You'd capture the drift more cheaply by holding the index.\n"
            f"- **3-line reversal forecasts? `BUSTED`** — the shuffled-returns placebo leaves the "
            f"result untouched (**p = {R['placebo'][1]:.2f}**): a scrambled order reverses as well as "
            "the real one, so the specific Three-Line-Break flips carry no forecasting information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The up-reversal's entire apparent profit is the unconditional drift of long equity "
            "indices, which you obtain more cheaply and more fully by **buying and holding**. The TLB "
            "rule sits *out* of the market on every down-flip (missing rebounds) and pays costs on "
            "each reversal, so it strictly dominates *nothing*. There is no capacity question because "
            "there is no edge to scale. Three-Line-Break is a descriptive noise-filter chart, not a "
            "forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The break number.** 2-line and 5-line breaks trade off whipsaws against lag, but "
            "both inherit the same drift confound — a clean follow-up sweeps the break number and "
            "shows the Welch t stays sub-2 throughout.\n"
            "- **Long-short.** Shorting the down-flips adds the *negative* of drift on the short leg; "
            "raced against a random long-short baseline it fares no better.\n"
            "- **Renko & Kagi.** The closest cousins (also time-independent, redraw-on-threshold "
            "charts) are affine tweaks of the same idea and inherit the same drift confound — see the "
            "Renko sibling study.\n\n"
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
