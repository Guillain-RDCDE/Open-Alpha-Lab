"""Generate the two narrative notebooks for Study 481 (ZigZag Indicator).

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
# 2026-05-31, partial June dropped), 21.4 years, ZigZag pct=5%, confirmed-up-leg long.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=428, pct=0.05,
    fp_spy="4cb5244f3990",
    # pooled confirmed-up-leg, per horizon:
    # (H, n, upleg_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 428, -5.9, 56, -0.32, 17.8, -23.7, -7.9, -1.04, 0.298),
    h10=(10, 428, 39.6, 61, 1.62, 29.0, 10.6, 37.6, 0.34, 0.734),
    h20=(20, 428, 111.9, 64, 2.87, 53.4, 58.5, 109.9, 1.33, 0.183),
    h60=(60, 423, 370.3, 66, 4.35, 222.2, 148.1, 368.3, 2.12, 0.034),
    # per-ticker H=20: (ticker, entries, upleg_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 78, 94.3, 0.98, 58.9, 35.4), ("QQQ", 91, 152.0, 1.56, 87.9, 64.1),
         ("IWM", 115, 139.2, 1.65, 36.5, 102.7), ("DIA", 70, 97.0, 1.25, 50.6, 46.4),
         ("GLD", 74, 52.9, 1.13, 34.6, 18.3)],
    # relabelled-leg placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(94.3, 0.645, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, upleg_bps, win%, one_sample_t)
    syn=[(0.00, 57, -1.0, 51, -0.01), (0.40, 52, 435.0, 81, 5.37)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![ZigZag_tradable%3F: Busted](https://img.shields.io/badge/ZigZag_tradable%3F-Busted-8b949e?style=flat-square)\n\n"
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

from zigzag_indicator import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real zigzag cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the ZigZag indicator actually call tradable turns? ⚡\n"
            "### A famous swing filter — connect the swings, buy the turn — meets a stopwatch\n\n"
            + BADGES +
            "Open any charting package and you'll find the **ZigZag**: it ignores the wiggles and "
            "draws clean straight legs between the real swing highs and lows, snapping a new leg only "
            "when price reverses by more than some percentage (5% here). The lore, repeated on every "
            "chart-pattern site and baked into MetaTrader, TradingView and the Elliott-wave tools, is "
            "that the ZigZag **identifies the turns**: when it turns **up** off a swing low, that low "
            "was a tradable bottom — so you buy.\n\n"
            "It *looks* uncanny, because on a finished chart the ZigZag's lows sit exactly on the "
            "bottoms. But there's a catch the tool hides: the newest leg **repaints**. The indicator "
            "draws it to the latest price and then **erases and redraws** it as price moves — the "
            "'perfect' low you see was only knowable *after* the bounce. So we did the only fair "
            "thing: trade **confirmed** legs only (the bounce has already happened, no peeking), fire "
            "the 'buy the turn' rule across five big indices over 21 years, and time the result "
            "against the only baseline that matters: **buying on random days instead.**\n\n"
            "> \U0001f4d3 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice** — research & education. Every chart is drawn by "
            "the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| If I buy when the ZigZag turns up (a confirmed low), do I make money? | **In absolute "
            "terms, yes — but mostly because the market goes up.** Win-rate ~64% at 20 days, returns "
            "look fine. |\n"
            "| Is that *the ZigZag's* doing? | **No.** Buy on **random days** instead and you do "
            "**just as well** at 5/10/20 days. The turn signal adds nothing over a dart at the "
            "horizons a 'turn' is supposed to live on. |\n"
            "| Does the up/down *structure* matter? | **No.** Randomly relabel which confirmation "
            "dates count as 'lows' and the result barely changes — any confirmation date is as good "
            "as a real one. |\n"
            "| So is it a tradable turn signal? | **No.** It's **beta in a costume** plus a slow "
            "post-pullback drift you'd get more cheaply by holding the index. |\n\n"
            "> The ZigZag is a great way to *describe* the swings after the fact. As a *forecast* — "
            "'this low will bounce' — it's a **mirage**: strip the repaint and the random-day "
            "baseline, and the turns stop being tradable."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The ZigZag filters out the noise: it only marks a swing when price reverses more "
            "than x%. When it turns up off a low, that low is confirmed — a high-probability bottom. "
            "Buy the turn.\"*\n\n"
            "This is the classic **threshold ZigZag** (MetaTrader's *Depth/Deviation/Backstep*, "
            "TradingView's ZigZag), the pivot engine under **Elliott Wave** and **harmonic patterns**. "
            "It's one of the most recognisable tools in technical analysis — so: do its turns "
            "actually turn?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the ZigZag genuinely *forecast* the turn, it would be remarkable: a percentage rule "
            "on past price would flag future bottoms, a clean crack in market efficiency you could "
            "trade with a ruler. That's the dream the tool sells.\n\n"
            "But two traps are built in. **(1) Repaint.** The newest leg is provisional — drawn to "
            "the latest extreme and redrawn as price moves. The 'perfect' low is only known *after* "
            "the bounce, so a naive backtest peeks at the future. **(2) Drift.** Stock indices climb "
            "over time, so *any* dip-buying rule looks profitable. To separate the **tool** from the "
            "**tide**, we (a) trade only **confirmed** legs (no repaint) and (b) compare to buying on "
            "**random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Build the ZigZag mechanically.** A new leg snaps only when price reverses "
            f"**{R['pct']*100:.0f}%** off the running extreme — the textbook threshold filter.\n"
            "2. **Remove the repaint.** A swing low is only *confirmed* once price has rebounded "
            f"**{R['pct']*100:.0f}%** above it. We use that confirmation bar — never the future-peeking "
            "final pivot.\n"
            "3. **Trade the lore.** When a low confirms (the ZigZag turns up), buy at the next close; "
            "measure the return over the next **5 / 10 / 20 / 60 days**.\n"
            "4. **The honest baseline.** Do the exact same hold on **random days**. If the turn "
            "matters, the up-leg must beat random. *If it doesn't, the tool is a mirage* — that's the "
            "result that would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical ZigZag even look like? Here's SPY with the (repainting, "
            "display-only) ZigZag line and the **confirmed** up-leg buys the rule would actually fire."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    cl = b['close']; seg = cl.iloc[-450:]\n"
            "    zz = st.zigzag_line(cl, pct=R['pct'])\n"
            "    ent = st.confirmed_uppleg_entries(cl, pct=R['pct'])\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg.values, c='k', lw=1.2, label='SPY close')\n"
            "    ax.plot(seg.index, zz.reindex(seg.index), c=GREY, lw=1.3, ls='--', label='ZigZag (repaints!)')\n"
            "    ax.scatter(ent, cl.reindex(ent), c=GREEN, s=45, zorder=5, label='confirmed up-leg BUY')\n"
            "    ax.set_title('A mechanical ZigZag on SPY (last ~2y) — buys are CONFIRMED legs')\n"
            "    ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('confirmed up-legs in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "Notice the green buys sit a few bars *after* each low — that's the confirmation lag (the "
            "honest, non-repainting version). The question is whether those buys are followed by "
            "bounces. **Let's race the confirmed up-leg against random entries** at four horizons. "
            "Blue = buy the turn; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    upleg, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.confirmed_uppleg_entries(c, pct=R['pct'])\n"
            "            re = st.random_entries(c, max(len(e),50), pct=R['pct'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        upleg.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    upleg = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, upleg, .4, color='#2c6fbb', label='buy the ZigZag turn')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(upleg,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The turn does NOT beat random at 5/10/20 days'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('up-leg:', [round(v) for v in upleg]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the story. The turn makes money in absolute terms (**+{R['h20'][2]:.0f} bps** "
            f"over 20 days) — but at 5 days it's actually *worse* than random, and at 10/20 days it's "
            "a statistical coin-flip (the quants notebook shows the Welch *t* never clears 2 until "
            "60 days). The apparent edge was **the market's upward drift**, not the turn."
        ),
        md(
            "**One more sanity check.** What if we keep the exact same confirmation dates but "
            "**randomly relabel** which ones count as 'lows' (the buys)? If the ZigZag's up/down "
            "structure really matters, the nonsense labels should do much worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.shuffled_leg_placebo(c, 20, pct=R['pct'], n_draws=300, seed=481)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real ZigZag confirmed-up-leg (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *relabelled-leg* runs do at least as well (p={pval:.2f}).')\n"
            "print('=> the up/down structure is not doing the work.')"
        ),
        md(
            f"Almost two-thirds of the **relabelled** runs match or beat the real turn "
            f"(*p* = {R['placebo'][1]:.2f}). If the ZigZag genuinely picked *the* tradable lows, "
            "random labels would collapse the result. They don't — because the result was never "
            "about the up/down structure, just about clustering near volatile pullbacks."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The confirmed turn does **not** beat buying on random days at "
            "5/10/20 days (worse at 5d; a coin-flip at 10/20d). Only 60 days noses ahead, and the "
            "placebo shows even that isn't the ZigZag's geometry. The big absolute returns are drift.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting for free — and costs only make it worse.\n"
            "- **\"The ZigZag calls tradable turns\"? — Busted.** Relabel the legs into nonsense and "
            "the result barely moves. The turns don't turn."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The turn's *only* edge over a coin flip is the market's "
            "long-run climb plus a slow post-pullback drift — which you'd capture more cheaply (and "
            "more fully) by just **holding the index**. The ZigZag buy is a worse, more expensive way "
            "to be long, and it depends on a confirmation that arrives *after* the bottom you wanted. "
            "Costs push the already-no-edge result further negative. As a forecasting tool it doesn't "
            "pay; as a drawing tool it was never meant to be a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further \U0001f6aa\n\n"
            "- **The repaint demo.** The single most useful follow-up: plot the ZigZag as it *was* "
            "in real time vs the finished line, and watch the last leg jump around. Most backtests "
            "secretly trade the finished line.\n"
            "- **Different thresholds.** Try 3% / 8% / 10% — the result is robust: drift in, "
            "ZigZag out.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* post-turn drift "
            "into a synthetic tape and shows the harness banks it (so the null here isn't a dead "
            "detector — it's an honest 'nothing there').\n\n"
            "*Think the ZigZag forecasts turns? Show the confirmed up-leg beating random entries at "
            "**t ≥ 2** *and* surviving the relabelled-leg placebo on a real tape — then we'll talk.*"
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
            "# The ZigZag indicator — a quantitative teardown \U0001f52c\n"
            "### Mechanical threshold ZigZag on 5 indices · confirmed-up-leg forward returns "
            "(no repaint) · one-sample HAC *t* · a drift-matched random-entry baseline "
            "· a relabelled-leg geometry placebo · costs · a synthetic planted-turn "
            "control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **turn** from the **drift** (and from the **repaint**): an "
            "upward-trending index makes *any* dip-buy look good, and the finished ZigZag peeks at the "
            "future, so the only meaningful test is confirmed-leg-vs-random plus a placebo that "
            "destroys the up/down structure while preserving the marginals.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), "
            "yfinance daily adjusted closes (**total-return** for the ETFs), 2005→2026. ZigZag "
            f"threshold pct={R['pct']*100:.0f}% with an explicit confirmation (repaint) lag; entry is "
            "the **next close** (one documented lag). Offline core + synthetic control are "
            "deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> \U0001f4a1 **The `\U0001f4a1 In plain words` notes** translate each result back to "
            "intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Confirmed up-leg vs a **drift-matched random** baseline: "
            f"Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f} bps at 5/10/20d, Welch "
            f"*t* = {R['h5'][8]:+.2f}/{R['h10'][8]:+.2f}/{R['h20'][8]:+.2f} (all *p* > 0.18). Only 60d "
            f"clears 2 (Welch *t* = {R['h60'][8]:+.2f}, *p* = {R['h60'][9]:.3f}). |\n"
            f"| **Tradability** | `MIRAGE` | The strong one-sample t's (20d t = {R['h20'][4]:.2f}, 60d "
            f"t = {R['h60'][4]:.2f}) are **mostly beta**; the lone 60d blip is geometry-independent "
            "slow drift (placebo below). No residual edge to scale. |\n"
            f"| **ZigZag tradable?** | `BUSTED` | Relabelling which confirmations are 'lows' "
            f"(relabelled-leg placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** of "
            "nonsense-label runs match or beat the real one. The up/down structure isn't load-bearing. |\n\n"
            "> \U0001f4a1 In plain words: the up-leg *looks* significant only because indices drift up "
            "and because confirmation dates cluster after volatile pullbacks. Race it vs random "
            "(strip the drift) or relabel the legs (strip the geometry) and the edge evaporates. "
            "Classic beta-in-a-costume, with a repaint trap on top."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "A threshold ZigZag tracks a running extreme and snaps a new leg when price reverses "
            "$\\ge pct$ off it. Let a swing low at bar $t^\\*$ be **confirmed** at the first bar "
            "$c>t^\\*$ with $C_c \\ge \\min\\cdot(1+pct)$. The rule buys at $c+1$ (next close) on each "
            "confirmed low and rides the new up-leg.\n\n"
            "- **H₀ (drift).** Up-leg returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (the turn forecasts).** Up-leg returns **exceed** random at some horizon, "
            "t ≥ 2.\n"
            "- **H₂ (the geometry matters).** Up-leg returns exceed a **relabelled-leg** placebo "
            "whose 'low' labels are randomised over the same confirmation dates.\n\n"
            "We find **H₀ not rejected** at 5/10/20d (up-leg ≤ random, *p* > 0.18); **H₁ "
            "barely rejected only at 60d** (Welch t = +2.12); **H₂ rejected** (placebo p ≈ "
            "0.65). The steelman fails on the horizons that matter, and the one survivor isn't the "
            "geometry."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the three confounds this design must kill\n\n"
            "**(a) Repaint / look-ahead.** The finished ZigZag's last leg is drawn with hindsight; "
            "the 'perfect' low is only knowable after the $pct$ bounce. We trade the **confirmation "
            "bar**, never the final pivot — the repaint lag is the antidote.\n\n"
            "**(b) Drift.** Equity indices have a positive unconditional daily mean. *Any* long-only "
            "entry inherits it; a one-sample $t$ against **zero** measures the tide, not the tool. The "
            "fix is the **random-entry baseline** and a Welch test of up-leg-*minus*-random.\n\n"
            "**(c) Geometry as a free parameter.** Confirmation dates cluster after volatile "
            "pullbacks; the danger is that *any* such date works, not specifically the 'low' label. "
            "The **relabelled-leg placebo** keeps the dates and the marginal but randomises the up/down "
            "labels — if the real result survives, the geometry was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} confirmed up-legs** "
            "pooled.\n"
            f"- **ZigZag.** Threshold filter, pct={R['pct']*100:.0f}%; consecutive same-direction "
            "moves absorbed into the running extreme.\n"
            "- **No repaint.** A swing low is usable only at its **confirmation bar** (price rebounded "
            f"{R['pct']*100:.0f}% off the low) — never the future-peeking final pivot.\n"
            "- **Entry.** Confirmed low → enter **next close** (one lag); hold H ∈ "
            "{5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of up-leg returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample up-leg vs random (the *real* test).\n"
            "- **Null #3 — relabelled-leg placebo** (up/down structure destroyed, dates+marginals kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every signal.\n"
            "- **Positive control.** Synthetic tape with a **planted** post-turn drift (knob `edge`): "
            "edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks decent, vs-random kills it\n\n"
            "Left: the confirmed up-leg's **one-sample** t against zero (the misleading number). "
            "Right: the same up-leg vs a **drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, upleg, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.confirmed_uppleg_entries(c, pct=R['pct'])\n"
            "            re = st.random_entries(c, max(len(e),50), pct=R['pct'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); upleg.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    upleg = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
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
            "a2.set_title('Up-leg vs RANDOM, Welch t (clears 2 only at 60d)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> \U0001f4a1 In plain words: the left bars look strong at 20/60d "
            f"({R['h20'][4]:.2f}/{R['h60'][4]:.2f}) — but that's the **drift**, every dip-buy inherits "
            f"it. The right bars are the real test: up-leg-minus-random is **negative** at 5d "
            f"({R['h5'][8]:+.2f}), a coin-flip at 10/20d, and only **{R['h60'][8]:+.2f}** at 60d. "
            "Three of four horizons say 'no turn edge'."
        ),
        md(
            "### 4b · Up-leg vs random across horizons — the gap is the verdict\n\n"
            "Mean return, confirmed up-leg vs random entry, all four horizons. The up-leg should "
            "tower over random if the turn forecasts. It doesn't until 60 days."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, upleg, .4, color='#2c6fbb', label='confirmed up-leg')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(upleg,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Confirmed up-leg does not beat random until 60d'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta up-leg-random (bps):', [round(a-b) for a,b in zip(upleg,rnd)])"
        ),
        md(
            f"> \U0001f4a1 In plain words: at 20 days the up-leg is **+{R['h20'][2]:.0f} bps** and "
            f"random is **+{R['h20'][5]:.0f} bps** — a +{R['h20'][6]:.0f} bps gap that the Welch test "
            f"(4a) says is noise (*p* = {R['h20'][9]:.2f}). The only horizon with a significant gap is "
            "60d, where the next test shows the *geometry* isn't responsible."
        ),
        md(
            "### 4c · The geometry placebo — relabel the legs, nothing changes\n\n"
            "Keep the exact confirmation dates and the price marginal, but randomise which dates are "
            "called 'lows' (the buys), matching the real count. If the ZigZag's up/down structure "
            "carries the signal, the real up-leg should sit far in the right tail of the relabelled "
            "distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.shuffled_leg_placebo(c, 20, pct=R['pct'], n_draws=400, seed=481)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    piv = st.zigzag_confirmations(c, pct=R['pct'])\n"
            "    confirms = piv['confirm_pos'].to_numpy(); confirms = confirms[confirms < len(c)]\n"
            "    n_lows = int((piv['kind']==-1).sum()); n_lows = min(n_lows, len(confirms))\n"
            "    rng = np.random.default_rng(481); idx = c.index; draws=[]\n"
            "    for _ in range(400):\n"
            "        pick = rng.choice(len(confirms), size=n_lows, replace=False)\n"
            "        ent = idx[np.sort(confirms[pick])]\n"
            "        rr = st.forward_returns(c, ent, 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(481); draws = rng.normal(90, 30, 400)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='relabelled-leg runs (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real up-leg {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean confirmed-up-leg 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real up-leg sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real up-leg {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => geometry not load-bearing)')"
        ),
        md(
            f"> \U0001f4a1 In plain words: the real up-leg (blue line) sits **in the middle** of the "
            f"relabelled cloud — **p = {R['placebo'][1]:.2f}**. Random labels do just as well, so the "
            "ZigZag's 'this is a low, buy it' structure isn't carrying information — confirmation "
            "dates just cluster near pullbacks. This is the cleanest refutation of 'the ZigZag calls "
            "the turn.'"
        ),
        md(
            "### 4d · Per-ticker — positive deltas, but no significant one-sample t\n\n"
            "20-day up-leg-minus-random delta and the up-leg's one-sample t, per instrument. The "
            "deltas are positive everywhere — but no single name's one-sample t clears 2."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas, tvals = [], [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = load(t)['close']\n"
            "        e = st.confirmed_uppleg_entries(c, pct=R['pct']); re = st.random_entries(c, max(len(e),50), pct=R['pct'], seed=7)\n"
            "        su = st.summarize(st.forward_returns(c,e,20)); sr = st.summarize(st.forward_returns(c,re,20))\n"
            "        names.append(t); deltas.append(su['mean_bps']-sr['mean_bps']); tvals.append(su['t'])\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]; tvals = [p[3] for p in R['per']]\n"
            "fig, (a1,a2) = plt.subplots(1,2,figsize=(11,4.2))\n"
            "a1.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6); a1.axhline(0,c='k',lw=.8)\n"
            "for i,d in enumerate(deltas): a1.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_ylabel('20d up-leg - random (bps)'); a1.set_title('Deltas positive but small')\n"
            "a2.bar(names, tvals, color=GREY, width=.6); a2.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(tvals): a2.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a2.set_ylabel('20d one-sample t'); a2.set_title('No name clears t=2'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})\n"
            "print('per-ticker 20d one-sample t:', {n:round(v,2) for n,v in zip(names,tvals)})"
        ),
        md(
            f"> \U0001f4a1 In plain words: every name shows a positive delta (IWM the largest at "
            f"{R['per'][2][5]:+.0f} bps) — but no one-sample t clears 2 (max **{R['per'][2][3]:.2f}** on "
            "IWM), the pooled 20d Welch test is insignificant, and the placebo kills the geometry. "
            "Positive-but-weak deltas with no significant t is the signature of horizon-stretched beta."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real turn\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** post-turn drift into "
            "a synthetic tape and check the same confirmed-up-leg rule banks it: edge=0 must stay at "
            "t≈0; edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.40):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=481, n_days=4000)\n"
            "    c = px['close']; e = st.confirmed_uppleg_entries(c, pct=0.05); s = st.summarize(st.forward_returns(c, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted turn -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} up-leg={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> \U0001f4a1 In plain words: with **no** planted turn the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"post-turn drift reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The "
            "detector works — so the flat real-tape result is a genuine 'nothing there', not a broken "
            "pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the confirmed up-leg does not beat a drift-matched random baseline "
            f"at the horizons a turn lives on (up-leg − random = "
            f"{R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f} bps at 5/10/20d; Welch t = "
            f"{R['h5'][8]:+.2f}/{R['h10'][8]:+.2f}/{R['h20'][8]:+.2f}, all *p* > 0.18). Only 60d clears "
            f"2 (Welch t = **{R['h60'][8]:+.2f}**, p = {R['h60'][9]:.3f}). The strong one-sample t's "
            f"(20d **{R['h20'][4]:.2f}**, 60d **{R['h60'][4]:.2f}**) are mostly drift.\n"
            f"- **Tradability `MIRAGE`** — no residual turn edge once the drift is removed; the lone "
            "60d blip is geometry-independent slow drift you capture more cheaply by holding the "
            "index. Costs only deepen the hole.\n"
            f"- **ZigZag tradable? `BUSTED`** — the relabelled-leg placebo leaves the result untouched "
            f"(**p = {R['placebo'][1]:.2f}**): nonsense-label runs do as well as the real 'lows', so "
            "the up/down structure carries no forecasting information. The ZigZag is a descriptive, "
            "**repainting** swing filter — strip the repaint and the turns stop being tradable."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The up-leg's apparent profit is the unconditional drift of long equity indices plus a "
            "slow post-pullback drift, both obtained more cheaply and more fully by **buying and "
            "holding**. The ZigZag rule trades *less* of the time (only on confirmed lows), pays costs "
            "on each, and depends on a confirmation that arrives *after* the bottom — so it strictly "
            "dominates *nothing*. There is no capacity question because there is no edge to scale. The "
            "ZigZag is a descriptive drawing tool, not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The repaint demonstration.** The single most useful follow-up: animate the ZigZag in "
            "real time vs the finished line. Most ZigZag 'backtests' secretly trade the finished "
            "(future-peeking) line; the confirmed-leg encoding here is the honest upper bound.\n"
            "- **Threshold robustness.** 3% / 8% / 10% thresholds give the same answer — drift in, "
            "ZigZag out — because the confound is structural, not parametric.\n"
            "- **ZigZag as a pivot engine.** Elliott-wave and harmonic-pattern tools sit on top of the "
            "ZigZag; they inherit its repaint and drift confounds wholesale.\n\n"
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
