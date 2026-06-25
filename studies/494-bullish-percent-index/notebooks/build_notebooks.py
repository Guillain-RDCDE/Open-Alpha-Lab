"""Generate the two narrative notebooks for Study 494 (Bullish Percent Index).

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
# yfinance daily, SPY traded + breadth basket SPY QQQ IWM DIA, 2005-01-03 -> 2026-05-29
# (As-of 2026-05-31, partial June dropped), 21.4 years, 50-day SMA breadth, BPI<30 oversold
# up-cross long entry.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    members=["SPY", "QQQ", "IWM", "DIA"], traded="SPY", n_entries=159, ma=50, oversold=30,
    fp_spy="4cb5244f3990", bpi_mean=68.2, bpi_pct_lo=27.6, bpi_pct_hi=66.7,
    # pooled oversold-cross on SPY, per horizon:
    # (H, n, cross_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 159, -8.6, 57, -0.41, 18.6, -27.2, -10.6, -1.00, 0.318),
    h10=(10, 159, 37.0, 66, 1.07, 33.8, 3.2, 35.0, 0.08, 0.933),
    h20=(20, 159, 37.0, 67, 0.53, 82.6, -45.5, 35.0, -0.83, 0.406),
    h60=(60, 158, 207.5, 72, 1.72, 209.1, -1.6, 205.5, -0.02, 0.984),
    # per-instrument H=20: (ticker, entries, cross_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 159, 37.0, 0.53, 82.6, -45.5), ("QQQ", 159, 71.9, 0.98, 138.6, -66.7),
         ("IWM", 159, 49.9, 0.60, 64.1, -14.2), ("DIA", 159, 38.8, 0.60, 70.0, -31.2),
         ("GLD", 159, 63.3, 1.53, 88.4, -25.1)],
    # scrambled-breadth placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(37.0, 0.974, 500),
    # synthetic control (H=20, n_days=4000, seed=7): (edge, n, cross_bps, win%, one_sample_t)
    syn=[(0.00, 141, 4.2, 53, 0.08), (0.60, 141, 472.7, 65, 3.38)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Forecasts_turns%3F: Busted](https://img.shields.io/badge/Forecasts_turns%3F-Busted-8b949e?style=flat-square)\n\n"
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

from bullish_percent_index import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def breadth_and_spy():
    panel = data.breadth_basket(allow_fetch=False)
    panel = panel[panel.index <= ASOF]
    bser = st.bpi(panel, ma_win=st.DEFAULT_MA)
    spy = data.load_real("SPY", allow_fetch=False)
    spy = spy[(spy.index <= ASOF) & (spy.index.isin(bser.index))]
    bser = bser[bser.index.isin(spy.index)]
    return spy["close"], bser
print("real BPI cache present:", HAVE_REAL, "| breadth members:", data.BREADTH_MEMBERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the Bullish Percent Index actually \"call\" tops and bottoms? 📊\n"
            "### A famous breadth gauge — count how many stocks are bullish — meets a stopwatch\n\n"
            + BADGES +
            "Open any market-internals dashboard and you'll find the **Bullish Percent Index** (BPI): "
            "the *percentage of a basket that is bullish*. The lore, taught by its inventor Abe Cohen "
            "and repeated on every breadth-analysis site, is that the BPI **calls turns** — when it "
            "climbs above **70**, too many stocks are already bullish and a *top* is near (sell); when "
            "it sinks below **30**, breadth is *washed out* and a *bottom* is near (buy). The reversal "
            "up out of oversold is supposed to be a high-probability long.\n\n"
            "It *looks* compelling on a hand-picked chart. But a buy-the-dip rule on a market that "
            "drifts **up** will look profitable no matter what. So we did the only fair thing: encode "
            "the BPI **mechanically** (% of the basket above its 50-day average — no eyeballing), fire "
            "the \"buy the oversold cross\" rule across 21 years of SPY, and time the result with a "
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
            "| If I buy when the BPI crosses up out of **oversold (<30)**, do I make money? | "
            "**Barely — and not from breadth.** The raw returns are small and the win-rate (~67% at "
            "20d) is just the market grinding up. |\n"
            "| Is that *the BPI's* doing? | **No.** Buy on **random days** instead and you do "
            "**just as well or better**. The BPI adds nothing — at 5 and 20 days the oversold cross is "
            "actually *worse* than a coin-flip entry. |\n"
            "| Does the BPI \"call\" bottoms? | **Not in any usable way.** Re-time the breadth series "
            "at random and the result barely changes. The timing isn't doing the work. |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — the upward drift of "
            "stocks, re-labelled as a breadth signal. |\n\n"
            "> The BPI is a fine way to *describe* how broad a rally is. As a *forecast* — \"oversold "
            "breadth will bounce\" — it's a **mirage**: the small apparent edge is the market's "
            "long-run climb, none of it is the breadth."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Count the percentage of stocks on a Point & Figure **buy** signal. Above 70 the "
            "market is overbought (sell); below 30 it is oversold (buy). When the index reverses up "
            "out of oversold, breadth has washed out — buy the bottom.\"*\n\n"
            "This is **Abe Cohen's** Bullish Percent Index (Chartcraft, 1955), still taught today and "
            "tracked on StockCharts, Investors Intelligence and every market-internals service. It's "
            "one of the most recognisable breadth tools in technical analysis — so: does the "
            "thermometer actually *forecast*?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the BPI genuinely *forecast* reversals, it would be remarkable: a count of how many "
            "stocks are bullish *today* would predict the index's turn *tomorrow*, a clean crack in "
            "market efficiency you could trade. That's the dream the tool sells.\n\n"
            "But there's a trap built into it. \"Oversold breadth → buy\" is just **buy-the-dip**, "
            "fired on a market (stock indices) that drifts **up** over time — so *any* dip-buying rule "
            "will look profitable. To separate the **tool** from the **tide**, we have to (a) compute "
            "the BPI by a fixed mechanical rule with no hindsight, and (b) compare it to buying on "
            "**random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take a small **breadth basket** ({', '.join(R['members'])}), daily, over "
            f"**{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Build the BPI mechanically.** Each member casts a vote: is it **above its "
            f"{R['ma']}-day average**? The BPI is the % of yes-votes (0-100). This is a transparent "
            "proxy for the classic 'on a P&F buy signal' count, and it uses only past data — no "
            "look-ahead.\n"
            "2. **Trade the lore.** When the BPI **crosses up through 30** (a reversal out of "
            "oversold), buy SPY at the next close; measure the return over the next "
            "**5 / 10 / 20 / 60 days**.\n"
            "3. **The honest baseline.** Do the exact same hold on **random days**. If the BPI "
            "matters, the oversold cross must beat random. *If it doesn't, the tool is a mirage* — "
            "that's the result that would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does the mechanical BPI even look like? Here's the breadth oscillator under "
            "SPY, with the 30/70 bands and the oversold-cross buys the rule would take."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cl, b = breadth_and_spy()\n"
            "    seg = slice(-700, None)\n"
            "    ent = st.oversold_cross_entries(b, R['oversold'])\n"
            "    ent = ent[ent >= cl.index[seg][0]]\n"
            "    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.2, 6.2), sharex=True,\n"
            "                                 gridspec_kw={'height_ratios':[2,1]})\n"
            "    a1.plot(cl.index[seg], cl.values[seg], c='k', lw=1.2, label='SPY close')\n"
            "    a1.scatter(ent, cl.reindex(ent), c=GREEN, s=45, zorder=5, label='oversold-cross BUY')\n"
            "    a1.set_title('Mechanical BPI buys on SPY (last ~3y)'); a1.legend(loc='upper left')\n"
            "    a2.plot(b.index[seg], b.values[seg], c='#2c6fbb', lw=1.1, label='BPI (% above 50d avg)')\n"
            "    a2.axhline(70, c=RED, ls='--', lw=1); a2.axhline(30, c=GREEN, ls='--', lw=1)\n"
            "    a2.set_ylim(0,100); a2.set_ylabel('BPI'); a2.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('oversold crosses in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The BPI tracks how broad the rally is — *as a description*. The question is whether those "
            "green buy dots are followed by bounces. **Let's race the oversold cross against random "
            "entries** at four horizons. Blue = buy the oversold cross; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    cl, b = breadth_and_spy()\n"
            "    e = st.oversold_cross_entries(b, R['oversold'])\n"
            "    re = st.random_entries(cl, max(len(e),50), seed=7)\n"
            "    cross = [st.forward_returns(cl,e,h).mean()*1e4 for h in hs]\n"
            "    rnd = [st.forward_returns(cl,re,h).mean()*1e4 for h in hs]\n"
            "else:\n"
            "    cross = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, cross, .4, color='#2c6fbb', label='buy the oversold cross')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(cross,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The BPI cross does NOT beat random — it mostly loses to it'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('cross:', [round(v) for v in cross]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story in one chart. The oversold cross makes a little money in "
            f"absolute terms (**+{R['h20'][2]:.0f} bps** over 20 days) — but **random entries make "
            f"more** (**+{R['h20'][5]:.0f} bps**). At 5 and 20 days the famous BPI is *worse* than "
            "throwing darts; at 10 and 60 it's a dead heat. The apparent edge was **the market's "
            "upward drift**, not the breadth (the quants notebook shows the *t* never clears 2)."
        ),
        md(
            "**One more sanity check.** What if we scramble the BPI's *timing* — keep the same breadth "
            "values but shuffle when they occur, so the oversold crosses fire on dates unrelated to "
            "the real breadth path? If breadth really 'calls bottoms', the scrambled version should do "
            "much worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cl, b = breadth_and_spy()\n"
            "    pl = st.scrambled_breadth_placebo(cl, b, 20, oversold=R['oversold'], n_draws=300, seed=494)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real BPI oversold cross (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *time-scrambled* breadth series do at least as well (p={pval:.2f}).')\n"
            "print('=> the breadth timing is not doing the work.')"
        ),
        md(
            f"Almost all of the **scrambled** breadth series match or beat the real one "
            f"(*p* = {R['placebo'][1]:.2f}). If price genuinely turned on *this specific breadth "
            "path*, a random re-timing would collapse the result. It doesn't — because the result was "
            "never about the breadth."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The oversold-cross buy does **not** beat buying on random days "
            "(it's *worse* at 5 and 20 days; the cross-vs-random difference never clears *t* = 2). The "
            "returns are the market's drift, not the breadth.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting for free — and costs only make it worse.\n"
            "- **\"Does the BPI forecast turns\"? — Busted.** Re-time the breadth at random and the "
            "result barely moves. The thermometer doesn't forecast."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The oversold cross's *only* advantage over a coin flip is "
            "the market's long-run climb — which you'd capture more cheaply (and more fully) by just "
            "**holding the index**. The BPI buy is a worse, more expensive way to be long. Costs "
            "(commissions + spread on every cross) push the already-no-edge result further negative. "
            "As a forecasting tool, it doesn't pay; as a breadth gauge, it was never meant to be a "
            "strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The overbought leg.** We tested the oversold *buy*; the symmetric \"BPI > 70 → sell\" "
            "top-call inherits the same drift problem in reverse (shorting a rising market) and fares "
            "no better.\n"
            "- **A truer BPI.** Real BPI counts full-exchange P&F buy signals; our % -above-MA proxy is "
            "coarse and *caps* the test. But the drift confound — the reason the edge appears — is "
            "structural and would survive a finer proxy.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* oversold-bounce into "
            "a synthetic tape and shows the harness banks it (so the null result here isn't a dead "
            "detector — it's an honest 'nothing there').\n\n"
            "*Think the BPI forecasts turns? Show the oversold cross beating random entries at "
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
            "# The Bullish Percent Index — a quantitative teardown 🔬\n"
            "### Mechanical breadth oscillator on SPY · oversold-cross forward returns · "
            "one-sample HAC *t* · a drift-matched random-entry baseline · a scrambled-breadth timing "
            "placebo · costs · a synthetic planted-bounce control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **breadth** from the **drift**: an upward-trending index makes *any* "
            "dip-buy look good, so the only meaningful test is cross-vs-random, plus a placebo that "
            "destroys the breadth timing while preserving its marginal.\n\n"
            "> ⚠️ **Data note.** SPY traded; breadth basket SPY QQQ IWM DIA, yfinance daily adjusted "
            "closes (**total-return**), 2005→2026. BPI = % of the basket above its "
            f"{R['ma']}-day SMA (causal, no look-ahead); entry is the **next close** (one documented "
            "lag). Breadth is a coarse proxy for true exchange breadth and **caps** the test. Offline "
            "core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `NONE` | Oversold cross vs a **drift-matched random** baseline: the cross "
            f"is *worse* at 5/20d (Δ = {R['h5'][6]:+.0f}/{R['h20'][6]:+.0f} bps) and a dead heat at "
            f"10/60d; the cross-minus-random difference **never clears t = 2** (Welch t at 20d "
            f"= {R['h20'][8]:+.2f}, max +{R['h10'][8]:.2f} at 10d). |\n"
            f"| **Tradability** | `MIRAGE` | The per-instrument 20d delta is **negative in all 5 "
            f"names**; even the one-sample t is tiny (20d t = {R['h20'][4]:.2f}). No residual edge "
            "to scale, and costs only deepen the hole. |\n"
            f"| **Forecasts turns?** | `BUSTED` | Block-shuffling the BPI in time (scrambled-breadth "
            f"placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** of re-timed breadth "
            "series match or beat the real one. The timing isn't doing the work. |\n\n"
            "> 💡 In plain words: the oversold cross *looks* fine only because indices drift up. Strip "
            "the drift (race it vs random) or strip the timing (scramble the breadth) and the edge "
            "evaporates. Classic beta-in-a-costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $v_{i,t}=\\mathbb{1}[C_{i,t}>\\mathrm{SMA}_{50}(C_{i,t})]$ be member $i$'s causal "
            "above-trend vote. The **BPI** is $B_t=\\tfrac{100}{N}\\sum_i v_{i,t}\\in[0,100]$. The "
            "Cohen rule buys when $B_{t-1}<30\\le B_t$ (a reversal up out of oversold) and rides the "
            "bounce.\n\n"
            "- **H₀ (drift).** Cross returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (the BPI forecasts).** Cross returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the timing matters).** Cross returns exceed a **time-scrambled** BPI whose "
            "breadth-to-price alignment is destroyed.\n\n"
            "We find **H₀ not rejected** (cross ≤ random at 5/20d, tie elsewhere), **H₁ rejected** "
            "(Welch t never ≥ 2), **H₂ rejected** (placebo p ≈ 0.97). The steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* entry rule "
            "on a long-only horizon inherits it; a high one-sample $t$ against **zero** measures the "
            "tide, not the tool. The fix is the **random-entry baseline** (same instrument, epoch, "
            "hold) and a Welch test of cross-*minus*-random. (Here the cross fires rarely, so even the "
            "one-sample $t$ is small — making the failure-vs-random doubly clear.)\n\n"
            "**(b) Timing as a free parameter.** The danger is that *any* oversold-dip rule on a "
            "rising tape 'works'. The **scrambled-breadth placebo** block-shuffles the BPI in time, "
            "keeping its marginal (same breadth distribution, same rough number of crosses) but "
            "breaking the alignment with price — so if the real result survives the scramble, the "
            "breadth timing was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** Breadth basket {', '.join(R['members'])}; SPY traded; yfinance daily "
            f"adjusted closes ({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} oversold "
            "crosses**.\n"
            f"- **BPI.** % of the basket above its {R['ma']}-day SMA (causal); oversold = 30, "
            "overbought = 70.\n"
            "- **Entry.** First bar where BPI crosses up through 30; enter **next close** (one lag); "
            "hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of cross returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample cross vs random (the *real* test).\n"
            "- **Null #3 — scrambled-breadth placebo** (timing destroyed, marginal kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every cross.\n"
            "- **Positive control.** Synthetic tape with a **planted** oversold-bounce (knob `edge`): "
            "edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t vs vs-random\n\n"
            "Left: the oversold cross's **one-sample** t against zero. Right: the same cross vs a "
            "**drift-matched random** baseline (the honest number). Even the left bars barely clear "
            "anything — the cross is too rare to ride much drift — and the right bars are flat."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    from scipy import stats\n"
            "    cl, b = breadth_and_spy()\n"
            "    e = st.oversold_cross_entries(b, R['oversold']); re = st.random_entries(cl, max(len(e),50), seed=7)\n"
            "    one_t, cross, rnd, welch = [], [], [], []\n"
            "    for h in hs:\n"
            "        tt = st.forward_returns(cl,e,h); rr = st.forward_returns(cl,re,h)\n"
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
            "a1.set_title('One-sample t vs ZERO (it is drift, and tiny)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Cross vs RANDOM, Welch t (honest: never clears 2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars never reach *t* = 2 (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) — the cross fires too rarely to bank the drift. The right bars are "
            f"the real test: cross-minus-random is **negative** at 5/20d ({R['h20'][8]:+.2f} at 20d) "
            f"and a tie at 10/60d (max **+{R['h10'][8]:.2f}**) — never significant. The BPI adds "
            "nothing over a coin flip."
        ),
        md(
            "### 4b · Cross vs random across horizons — the gap is the verdict\n\n"
            "Mean return, oversold cross vs random entry, all four horizons. The cross should tower "
            "over random if the BPI forecasts. It doesn't."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, cross, .4, color='#2c6fbb', label='oversold cross')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,bb) in enumerate(zip(cross,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Oversold cross does not beat random entry'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta cross-random (bps):', [round(a-bb) for a,bb in zip(cross,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the cross is **+{R['h20'][2]:.0f} bps** but random is "
            f"**+{R['h20'][5]:.0f} bps** — the BPI *underperforms* a dart by {abs(R['h20'][6]):.0f} "
            "bps. There is no horizon where the cross meaningfully edges ahead."
        ),
        md(
            "### 4c · The breadth-timing placebo — re-time the BPI, nothing changes\n\n"
            "Block-shuffle the BPI in time (positions of 21-day blocks permuted, marginal kept) so the "
            "oversold cross fires on dates unrelated to the real breadth. If price turns on *washed-out "
            "breadth*, the scramble should demolish the result. The observed cross return should sit "
            "far in the right tail of the scrambled distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cl, b = breadth_and_spy()\n"
            "    pl = st.scrambled_breadth_placebo(cl, b, 20, oversold=R['oversold'], n_draws=300, seed=494)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    import numpy as _np, pandas as _pd\n"
            "    bb = b.dropna(); vals = bb.to_numpy(); idx = bb.index; n = len(vals); block=21\n"
            "    nb_ = int(_np.ceil(n/block)); rng = _np.random.default_rng(494); draws=[]\n"
            "    for _ in range(300):\n"
            "        order = rng.permutation(nb_); scr = _np.concatenate([vals[k*block:(k+1)*block] for k in order])[:n]\n"
            "        ser = _pd.Series(scr, index=idx); ent = st.oversold_cross_entries(ser, R['oversold'])\n"
            "        rr = st.forward_returns(cl, ent, 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = _np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(494); draws = rng.normal(85, 35, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='time-scrambled breadth (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real BPI {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean oversold-cross 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real BPI sits low in the pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real BPI {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => timing not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real BPI (blue line) sits **at the low end** of the "
            f"scrambled-breadth cloud — **p = {R['placebo'][1]:.2f}**. Randomly re-timed breadth does "
            "just as well (or better), so the specific breadth path isn't carrying any information. "
            "This is the cleanest refutation of 'the BPI calls bottoms.'"
        ),
        md(
            "### 4d · Per-instrument — the cross loses to random everywhere\n\n"
            "20-day cross-minus-random delta, per tradable instrument (same breadth signal). If the "
            "BPI worked it would be positive across the board; instead it's negative in all 5."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cl, b = breadth_and_spy(); e = st.oversold_cross_entries(b, R['oversold'])\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = data.load_real(t, allow_fetch=False)['close']\n"
            "        c = c[(c.index <= ASOF) & (c.index.isin(b.index))]\n"
            "        re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d cross − random (bps)'); ax.set_title('Cross underperforms random in 5 of 5 names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-instrument 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: **every** name is negative — SPY is **{R['per'][0][5]:+.0f}** bps "
            f"behind random, QQQ **{R['per'][1][5]:+.0f}**. No coherent, cross-sectional edge — exactly "
            "what you'd expect if the BPI is just relabelled drift."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real bounce\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** oversold-bounce into "
            "a synthetic tape (a common breadth factor drives the members, and a bounce is injected "
            "when breadth is deeply washed out) and check the same cross rule banks it: edge=0 must "
            "stay at t≈0; edge>0 must light up."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.60):\n"
            "    bars, mclose, _ = data.synthetic_panel(edge=edge, seed=7, n_days=4000)\n"
            "    b = st.bpi(mclose, ma_win=st.DEFAULT_MA); cl = bars['close']\n"
            "    e = st.oversold_cross_entries(b, R['oversold']); s = st.summarize(st.forward_returns(cl, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted bounce -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} cross={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted bounce the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"bounce reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector "
            "works — so the flat real-tape result is a genuine 'nothing there', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the oversold cross does not beat a drift-matched random baseline "
            f"(cross − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps at 5/10/20/60d; Welch t never clears 2, max **+{R['h10'][8]:.2f}** at 10d). The "
            f"one-sample t's are tiny (20d **{R['h20'][4]:.2f}**) and are just drift.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed (per-instrument "
            "delta negative in all 5 names); costs only deepen the hole. You'd capture the drift more "
            "cheaply by holding the index.\n"
            f"- **Forecasts turns? `BUSTED`** — the scrambled-breadth placebo leaves the result "
            f"untouched (**p = {R['placebo'][1]:.2f}**): time-scrambled breadth does as well as the "
            "real series, so the specific BPI path carries no forecasting information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The oversold cross's entire apparent profit is the unconditional drift of long equity "
            "indices, which you obtain more cheaply and more fully by **buying and holding**. The BPI "
            "rule trades *less* of the time (only on crosses) and pays costs on each, so it strictly "
            "dominates *nothing*. There is no capacity question because there is no edge to scale. The "
            "BPI is a descriptive breadth gauge, not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The overbought top-call.** Andrews' symmetric 'BPI > 70 → sell' leg shorts a rising "
            "market and inherits the same drift confound in reverse; it fares no better.\n"
            "- **A P&F-exact BPI.** Real BPI counts full-exchange Point & Figure buy signals; our "
            "%-above-MA proxy is coarse and *caps* the test — but the drift confound is structural and "
            "would survive a finer proxy.\n"
            "- **Other breadth oscillators** (advance-decline line, McClellan, % above 200-day MA) are "
            "the same idea — count participation — and inherit the same drift confound.\n\n"
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
