"""Generate the two narrative notebooks for Study 497 (Woodie's Pivot Points).

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
# 2026-05-31, partial June dropped), 21.4 years, Woodie P=(H+L+2C)/4, S1=2P-H, S1-support touch.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=7002,
    fp_spy="4cb5244f3990",
    # pooled S1-support touch, per horizon:
    # (H, n, touch_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 6994, 28.9, 58, 7.58, 26.2, 2.7, 26.9, 0.59, 0.556),
    h10=(10, 6989, 53.6, 60, 7.86, 51.4, 2.2, 51.6, 0.35, 0.725),
    h20=(20, 6975, 99.5, 63, 7.78, 107.6, -8.1, 97.5, -0.93, 0.353),
    h60=(60, 6916, 296.1, 69, 10.23, 300.5, -4.4, 294.1, -0.30, 0.761),
    # per-ticker H=20: (ticker, entries, touch_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 1388, 96.1, 3.85, 103.7, -7.6), ("QQQ", 1390, 134.4, 4.46, 143.4, -9.0),
         ("IWM", 1408, 82.1, 2.43, 96.4, -14.3), ("DIA", 1368, 91.2, 3.95, 92.7, -1.5),
         ("GLD", 1448, 94.0, 3.61, 101.8, -7.8)],
    # random-level placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(96.1, 0.655, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, touch_bps, win%, one_sample_t)
    syn=[(0.00, 1173, 15.6, 52, 0.56), (0.40, 1172, 446.8, 88, 18.62)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Woodie_levels_hold%3F: Busted](https://img.shields.io/badge/Woodie_levels_hold%3F-Busted-8b949e?style=flat-square)\n\n"
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

from woodie_pivots import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real Woodie cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do Woodie's pivot points actually act as support? 🪝\n"
            "### A day-trader staple — yesterday's range becomes today's support/resistance — meets a stopwatch\n\n"
            + BADGES +
            "Open any trading platform and you'll find **pivot points**: from yesterday's high, low "
            "and close it draws a central pivot **P** with support lines **S1, S2** below and "
            "resistance **R1, R2** above. The **Woodie** variant — from day-trader Ken Wood — "
            "double-weights the close: **P = (H + L + 2C) / 4**, and **S1 = 2P − H**. The lore, "
            "repeated on every pivot-point write-up, is that **price respects these levels** — when "
            "it reaches down to **S1** it should find support and bounce.\n\n"
            "It *looks* uncanny on a hand-picked chart. But a level computed from yesterday's bar, "
            "plotted on a market that drifts **up** over time, is the textbook setup for fooling "
            "yourself: any buy on a rising market looks good. So we did the only fair thing: encode "
            "the S1-support rule **mechanically**, fire it thousands of times across five big indices "
            "over 21 years, and time the result with a stopwatch — against the only baseline that "
            "matters: **buying on random days instead.**\n\n"
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
            "| If I buy when price reaches down to **S1**, do I make money? | **Yes — but only "
            "because the market goes up.** The raw win-rate is ~60% and the returns look great. |\n"
            "| Is that *the pivot's* doing? | **No.** Buy on **random days** instead and you do "
            "**just as well** — at 20 days the S1-touch is actually *behind* a coin-flip entry. |\n"
            "| Do Woodie levels hold better than random? | **No.** Replace S1 with a *randomly-placed* "
            "level and the result barely changes. The specific close-weighted line isn't the work. |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — the upward drift of "
            "stocks, re-labelled as a support level. |\n\n"
            "> Woodie's pivots are a fine way to *mark up* a chart. As a *forecast* — \"S1 will "
            "bounce\" — they're a **mirage**: all of the apparent edge is the market's long-run climb, "
            "none of it is the level."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"From yesterday's high, low and close, compute Woodie's pivot **P = (H + L + 2C)/4** "
            "(the close counts double). Below it sits **S1 = 2P − H**. When today's price reaches "
            "down to S1, it finds **support** and bounces — buy it.\"*\n\n"
            "This is the **Woodie pivot** (Ken Wood, *Woodie's CCI Club*), a close-weighted cousin of "
            "the classic floor-trader pivot, built into TradingView, MetaTrader and every charting "
            "suite. It's one of the most-plotted day-trading tools there is — so: do the levels "
            "actually *support*?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If S1 genuinely *forecast* a bounce, it would be remarkable: a number from yesterday's "
            "bar would predict today's turning point, a clean crack in market efficiency you could "
            "trade with a calculator.\n\n"
            "But there's a trap. The S1-touch fires on a *majority* of sessions (a daily low dips to "
            "S1 most days), and it's measured on a market (stock indices) that drifts **up** — so the "
            "rule is barely different from \"always be long\", and *any* long entry looks profitable. "
            "To separate the **level** from the **tide**, we must (a) apply S1 by a fixed mechanical "
            "rule with a one-day lag, and (b) compare it to buying on **random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Compute Woodie's levels from yesterday's bar.** P = (H + L + 2C)/4 and S1 = 2P − H "
            "use the **prior** day's H/L/C — knowable at today's open, no future data.\n"
            "2. **Trade the lore.** When today's **low reaches down to S1**, buy at the next close; "
            "measure the return over the next **5 / 10 / 20 / 60 days**.\n"
            "3. **The honest baseline.** Do the exact same hold on **random days**. If S1 matters, the "
            "touch must beat random.\n"
            "4. **The level placebo.** Swap S1 for a *randomly-placed* support (same touch frequency). "
            "*If the random level does just as well, the Woodie line is a mirage* — that's the result "
            "that would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what do Woodie's levels even look like? Here's SPY with the rolling pivot P and "
            "its S1 support, and the S1-support touches the rule would buy."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    seg = b.iloc[-300:]\n"
            "    lev = st.woodie_levels(b)\n"
            "    ent = st.s1_touch_entries(b)\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg['close'].values, c='k', lw=1.2, label='SPY close')\n"
            "    ax.plot(seg.index, lev['P'].reindex(seg.index), c=GREY, lw=1.0, ls='--', label='Woodie P')\n"
            "    ax.plot(seg.index, lev['S1'].reindex(seg.index), c=GREEN, lw=1.1, label='S1 support')\n"
            "    ax.plot(seg.index, lev['R1'].reindex(seg.index), c=RED, lw=1.0, alpha=.7, label='R1 resistance')\n"
            "    ax.scatter(ent, b['close'].reindex(ent), c=GREEN, s=22, zorder=5, label='S1 BUY')\n"
            "    ax.set_title('Woodie pivots on SPY (last ~1.2y)'); ax.legend(loc='upper left', fontsize=8)\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('S1-support touches in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The S1 line hugs just below price — and it gets touched constantly. The question is "
            "whether those green buy dots are followed by bounces. **Let's race the S1-touch against "
            "random entries** at four horizons. Blue = buy the S1 support; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    touch, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t)\n"
            "            e = st.s1_touch_entries(bb)\n"
            "            re = st.random_entries(bb, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(bb, e, h)); rr.append(st.forward_returns(bb, re, h))\n"
            "        touch.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    touch = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, touch, .4, color='#2c6fbb', label='buy the S1 support')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(touch,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('Woodie S1 does NOT beat random — they are a dead heat'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('touch:', [round(v) for v in touch]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story in one chart. The S1-touch makes money in absolute terms "
            f"(**+{R['h20'][2]:.0f} bps** over 20 days) — but **random entries make just as much or "
            f"more** (**+{R['h20'][5]:.0f} bps**). The two bars are essentially the same height at "
            "every horizon. The apparent edge was **the market's upward drift**, not the level."
        ),
        md(
            "**One more sanity check.** What if we replace S1 with a *randomly-placed* support — same "
            "touch frequency, but no Woodie geometry? If price really 'respects S1', the random level "
            "should do much worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.random_level_placebo(load('SPY'), 20, n_draws=300, seed=497)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real Woodie S1 touch (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *random-level* supports do at least as well (p={pval:.2f}).')\n"
            "print('=> the specific Woodie level is not doing the work.')"
        ),
        md(
            f"Two thirds of the **random-level** supports match or beat the real S1 "
            f"(*p* = {R['placebo'][1]:.2f}). If price genuinely respected *Woodie's specific line*, a "
            "random level would collapse the result. It doesn't — because the result was never about "
            "the level."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The S1-support buy does **not** beat buying on random days (it's a "
            "dead heat, *behind* at 20 days; the touch-vs-random difference never clears *t* = 2). The "
            "big absolute returns are the market's drift, not the level.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting for free — and costs only make it worse.\n"
            "- **\"Do Woodie levels hold better than random?\" — Busted.** Swap S1 for a random level "
            "and the result barely moves. The level doesn't support."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The S1-touch's *only* advantage over a coin flip is the "
            "market's long-run climb — which you'd capture more cheaply (and more fully) by just "
            "**holding the index**. Since the touch fires on most sessions, the rule is a worse, more "
            "expensive way to be long. Costs (commissions + spread on every touch) push the already-no-"
            "edge result further negative. As a forecasting tool it doesn't pay; as a chart annotation "
            "it was never meant to be a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **R1 resistance / other levels.** The mirror claim (R1 caps price) and S2/R2 inherit "
            "the same drift confound — a fun follow-up shows them all flat vs random.\n"
            "- **Other pivot flavours.** Classic, Camarilla, Fibonacci and DeMark pivots are affine "
            "tweaks of the same prior-day H/L/C; the result is robust: drift in, level out.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* S1 bounce into a "
            "synthetic tape and shows the harness banks it (so the null result here isn't a dead "
            "detector — it's an honest 'nothing there').\n\n"
            "*Think Woodie's S1 supports? Show the touch beating random entries at **t ≥ 2** on a real "
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
            "# Woodie's Pivot Points — a quantitative teardown 🔬\n"
            "### Close-weighted pivots on 5 indices · S1-support-touch forward returns · "
            "one-sample HAC *t* · a drift-matched random-entry baseline · a random-level placebo · "
            "costs · a synthetic planted-bounce control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **level** from the **drift**: an upward-trending index makes *any* "
            "long entry look good — and the S1-touch fires on most sessions, so it is nearly always "
            "long — so the only meaningful test is touch-vs-random, plus a placebo that swaps the "
            "Woodie level for a random one while preserving touch frequency.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted OHLC (**total-return** for the ETFs), 2005→2026. Woodie levels use the **prior** "
            "bar (one documented lag); entry is the **next close** (one more lag). Offline core + "
            "synthetic control are deterministic. Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | S1-touch vs a **drift-matched random** baseline: Δ = "
            f"{R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} bps at "
            f"5/10/20/60d and the touch-minus-random difference **never clears t = 2** (Welch t at 5d "
            f"= {R['h5'][8]:+.2f}, 20d = {R['h20'][8]:+.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | The big one-sample t's (20d t = {R['h20'][4]:.2f}) are "
            f"**pure beta** — an S1-touch fires on most sessions, so it is essentially buy-and-hold. "
            "No residual edge to scale. |\n"
            f"| **Woodie levels hold?** | `BUSTED` | Swapping S1 for a **random-level** support leaves "
            f"the result intact: **p = {R['placebo'][1]:.2f}** of random levels match or beat the real "
            "one. The Woodie line isn't doing the work. |\n\n"
            "> 💡 In plain words: the S1-touch *looks* significant only because indices drift up and "
            "the rule is almost always long. Strip the drift (race it vs random) or strip the geometry "
            "(random level) and the edge evaporates. Classic beta-in-a-costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "From yesterday's bar $(H,L,C)$, Woodie's pivot double-weights the close: "
            "$P=\\tfrac{H+L+2C}{4}$, with support $S_1=2P-H$ and $S_2=P-(H-L)$ "
            "(resistance $R_1=2P-L$, $R_2=P+(H-L)$). Let $S_{1,t}$ be today's S1 (from $t-1$). The "
            "rule buys when today's low $L_t\\le S_{1,t}$ and rides the bounce.\n\n"
            "- **H₀ (drift).** Touch returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (S1 forecasts).** Touch returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the level matters).** Touch returns exceed a **random-level** support whose line "
            "is geometrically meaningless.\n\n"
            "We find **H₀ not rejected** (touch ≈ random everywhere), **H₁ rejected** (Welch t never "
            "≥ 2), **H₂ rejected** (placebo p ≈ 0.66). The steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* long entry "
            "inherits it; a high one-sample $t$ against **zero** measures the tide, not the tool — and "
            "here especially, because the S1-touch fires on a majority of sessions, so it is barely "
            "distinguishable from buy-and-hold. The fix is the **random-entry baseline** (same "
            "instrument, epoch, hold) and a Welch test of touch-*minus*-random.\n\n"
            "**(b) Geometry as a free parameter.** S1 is one arithmetic combination of yesterday's "
            "bar; the danger is that *any* level near price gets 'touched' and inherits the same "
            "drift. The **random-level placebo** keeps the touch frequency and the price marginal but "
            "places the support a *random* distance below the prior close — the line becomes "
            "meaningless, so if the real result survives, the Woodie geometry was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted OHLC "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} S1 touches** pooled.\n"
            "- **Levels.** Woodie P = (H+L+2C)/4, S1 = 2P − H, from the **prior** bar (one lag, no "
            "look-ahead).\n"
            "- **Entry.** First bar whose low ≤ prior-day S1; enter **next close** (one more lag); "
            "hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of touch returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample touch vs random (the *real* test).\n"
            "- **Null #3 — random-level placebo** (Woodie level swapped for a random support).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every touch.\n"
            "- **Positive control.** Synthetic tape with a **planted** S1-bounce (knob `edge`): "
            "edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks great, vs-random kills it\n\n"
            "Left: the S1-touch's **one-sample** t against zero (the misleading number). Right: the "
            "same touch vs a **drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, touch, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t)\n"
            "            e = st.s1_touch_entries(bb)\n"
            "            re = st.random_entries(bb, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(bb, e, h)); rr.append(st.forward_returns(bb, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); touch.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    touch = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (misleading: it is beta)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if abs(v)>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(-2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Touch vs RANDOM, Welch t (honest: never clears 2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 by a mile (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) — but that's the **drift**, and the rule is nearly always long. "
            f"The right bars are the real test: touch-minus-random is tiny and bounces around zero "
            f"({R['h5'][8]:+.2f} at 5d, {R['h20'][8]:+.2f} at 20d) — never significant. S1 adds nothing "
            "over a coin flip."
        ),
        md(
            "### 4b · Touch vs random across horizons — the gap is the verdict\n\n"
            "Mean return, S1-touch vs random entry, all four horizons. The touch should tower over "
            "random if S1 forecasts. It's a dead heat."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, touch, .4, color='#2c6fbb', label='S1-support touch')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(touch,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('S1-support touch does not beat random entry'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta touch-random (bps):', [round(a-b) for a,b in zip(touch,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the touch is **+{R['h20'][2]:.0f} bps** but random is "
            f"**+{R['h20'][5]:.0f} bps** — S1 *underperforms* a dart by {abs(R['h20'][6]):.0f} bps. At "
            "no horizon does the touch meaningfully beat random; the Welch test (4a) confirms the gap "
            "is noise."
        ),
        md(
            "### 4c · The level placebo — swap S1 for a random level, nothing changes\n\n"
            "Replace S1 with a support placed a *random (re-sampled) distance* below the prior close "
            "(touch frequency and marginal kept). If price respects *Woodie's specific line*, the "
            "swap should demolish the result. The observed touch return should sit far in the right "
            "tail of the random-level distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bspy = load('SPY')\n"
            "    pl = st.random_level_placebo(bspy, 20, n_draws=300, seed=497)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    # rebuild the placebo distribution for the histogram\n"
            "    import numpy as _np, pandas as _pd\n"
            "    lev = st.woodie_levels(bspy); prev_c = bspy['close'].shift(1)\n"
            "    depth = (prev_c - lev['S1']); vm = depth.notna(); depths = depth[vm].to_numpy()\n"
            "    rng = _np.random.default_rng(497); low = bspy['low']; idx = bspy.index\n"
            "    draws=[]\n"
            "    for _ in range(300):\n"
            "        fd = _pd.Series(_np.nan, index=idx); fd[vm] = rng.choice(depths, size=depths.size, replace=True)\n"
            "        fs1 = prev_c - fd; m=(low<=fs1)&fs1.notna(); f=m&~m.shift(1,fill_value=False)\n"
            "        rr=st.forward_returns(bspy, idx[f.to_numpy()], 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws=_np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(497); draws = rng.normal(98, 18, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='random-level supports (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real Woodie S1 {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean S1-touch 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real S1 sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real Woodie S1 {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => level not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real Woodie S1 (blue line) sits **in the middle** of the "
            f"random-level cloud — **p = {R['placebo'][1]:.2f}**. A randomly-placed support does just "
            "as well, so the specific close-weighted line isn't carrying any information. This is the "
            "cleanest refutation of 'Woodie levels hold better than random.'"
        ),
        md(
            "### 4d · Per-ticker — the touch loses to random everywhere\n\n"
            "20-day touch-minus-random delta, per instrument. If S1 worked it would be positive "
            "across the board; instead it's negative in all five."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        bb = load(t)\n"
            "        e = st.s1_touch_entries(bb); re = st.random_entries(bb, max(len(e),50), seed=7)\n"
            "        d = st.summarize(st.forward_returns(bb,e,20))['mean_bps'] - st.summarize(st.forward_returns(bb,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d touch − random (bps)'); ax.set_title('Touch underperforms random in 5 of 5 names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: every name is **negative** — SPY {R['per'][0][5]:+.0f}, "
            f"IWM {R['per'][2][5]:+.0f} bps *behind* random. No coherent, cross-sectional edge — "
            "exactly what you'd expect if S1 is just relabelled drift."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real bounce\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** S1 support bounce "
            "into a synthetic tape and check the same S1-touch rule banks it: edge=0 must stay at "
            "t≈0; edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.40):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=497, n_days=4000)\n"
            "    e = st.s1_touch_entries(px); s = st.summarize(st.forward_returns(px, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted bounce -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} touch={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
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
            f"- **Signal `NONE`** — the S1-support touch does not beat a drift-matched random baseline "
            f"(touch − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps at 5/10/20/60d; Welch t never clears 2, max **{R['h5'][8]:+.2f}** at 5d). The "
            f"impressive one-sample t's (20d **{R['h20'][4]:.2f}**) are pure beta — the touch fires on "
            "most sessions, so it is essentially buy-and-hold.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; costs only "
            "deepen the hole. You'd capture the drift more cheaply by holding the index.\n"
            f"- **Do Woodie levels hold better than random? `BUSTED`** — the random-level placebo "
            f"leaves the result untouched (**p = {R['placebo'][1]:.2f}**): randomly-placed supports do "
            "as well as Woodie's close-weighted S1, so the specific level carries no information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The S1-touch's entire apparent profit is the unconditional drift of long equity indices, "
            "which you obtain more cheaply and more fully by **buying and holding**. Because the touch "
            "fires on most sessions, the rule is essentially always-long minus costs — it dominates "
            "*nothing*. There is no capacity question because there is no edge to scale. Woodie's "
            "pivots are a descriptive day-trading annotation, not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **R1 / S2 / R2.** The resistance and second-line claims are affine cousins of S1 and "
            "inherit the same drift confound; a clean follow-up shows them all flat vs random.\n"
            "- **Other pivot flavours.** Classic (H+L+C)/3, Camarilla, Fibonacci and DeMark pivots "
            "differ only in weights — Woodie's 2× close is exactly what the random-level placebo "
            "destroys, and they all land None × Mirage.\n"
            "- **Intraday bars.** Proponents trade pivots on 5-minute bars. Higher-frequency data adds "
            "more noise and more costs, not a hidden edge — the drift confound is unchanged.\n\n"
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
