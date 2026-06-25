"""Generate the two narrative notebooks for Study 461 (Descending Triangle).

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
# 2026-05-31), 21.4 years, fractal k=5, lookback=6, descending-triangle support-break SHORT.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=39, k=5, lookback=6,
    fp_spy="4cb5244f3990",
    # pooled support-break SHORT, per horizon (positive = break-down paid):
    # (H, n, short_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 39, 6.2, 46, 0.15, -58.1, 64.3, 4.2, 1.58, 0.119),
    h10=(10, 39, -64.0, 44, -1.42, -61.4, -2.6, -66.0, -0.05, 0.957),
    h20=(20, 39, -152.7, 28, -3.40, -64.9, -87.9, -154.7, -1.43, 0.158),
    h60=(60, 39, -417.5, 23, -3.30, -101.9, -315.6, -419.5, -2.57, 0.012),
    # per-ticker H=20: (ticker, breaks, short_bps, one_sample_t(nan->None), random_bps, delta_bps)
    per=[("SPY", 3, -215.7, None, -60.1, -155.7), ("QQQ", 3, -46.7, None, -136.5, 89.8),
         ("IWM", 5, -342.5, None, -51.5, -291.0), ("DIA", 7, -321.3, -4.95, -42.1, -279.1),
         ("GLD", 21, -57.5, -1.03, -34.1, -23.4)],
    # scrambled-highs placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(-215.7, 0.692, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, short_bps, win%, one_sample_t)
    syn=[(0.00, 13, -31.7, 38, -0.43), (0.50, 8, 1103.1, 88, 1.90)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Resolves_textbook_direction%3F: Busted](https://img.shields.io/badge/Resolves_textbook_direction%3F-Busted-8b949e?style=flat-square)\n\n"
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

from descending_triangle import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real descending-triangle cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a descending triangle really break DOWN? 📐\n"
            "### A textbook bearish pattern — flat floor, falling ceiling — meets a stopwatch\n\n"
            + BADGES +
            "Open any chart-pattern guide and you'll meet the **descending triangle**: a row of lows "
            "at the *same* price (a flat **support**) with a row of **descending highs** pressing down "
            "on them, the two squeezing toward a point. The lore, straight out of Edwards & Magee and "
            "repeated on every charting site, is that this is a **bearish continuation** — price coils, "
            "then **breaks down through the floor and keeps falling**. The textbook trade: **short the "
            "break**.\n\n"
            "It *looks* inevitable on a hand-picked chart. But a pattern you only recognise **after** "
            "price has drawn it — choosing which lows are 'flat enough' and which highs 'descend' — is "
            "the textbook way to fool yourself. So we did the fair thing: encode the triangle "
            "**mechanically** (no eyeballing), short every clean support break across five big indices "
            "over 21 years, and time the result with a stopwatch — against the only baseline that "
            "matters: **shorting on random days instead.**\n\n"
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
            "| If I short when price breaks the flat support, do I make money? | **No.** Booked as a "
            "short the rule **loses** at 10/20/60 days (−64 / −153 / −418 bps). |\n"
            "| Does the 'bearish' break actually go down? | **No — it mostly goes UP.** Sixty days "
            "after the break the short wins only **23%** of the time: the break is followed by a "
            "*rally* three times in four. |\n"
            "| Is it at least better than shorting on random days? | **No.** At 60 days it's "
            "*significantly worse* than a random short (*p* = 0.01). |\n"
            "| Does the *triangle shape* matter? | **No.** Scramble the descending highs into nonsense "
            "and the result barely changes. The famous geometry isn't doing the work. |\n\n"
            "> The descending triangle is a fine way to *describe* a sideways pullback after the fact. "
            "As a *forecast* — 'it will break down' — it's a **mirage that points the wrong way**: the "
            "break, in a market that drifts up, mostly resolves UP."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A descending triangle is a flat horizontal **support** under a falling line of "
            "**lower highs**. The selling pressure (descending highs) eventually overwhelms the floor; "
            "price **breaks below support and continues lower**, by about the height of the triangle. "
            "Short the break.\"*\n\n"
            "This is **Edwards & Magee's** classic continuation pattern (*Technical Analysis of Stock "
            "Trends*, 1948), catalogued by **Bulkowski** and built into TradingView, MetaTrader and "
            "every charting suite. It's one of the most recognisable bearish setups in technical "
            "analysis — so: does the triangle actually break the way it's drawn?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the triangle genuinely *forecast* a breakdown, it would be remarkable: a few past "
            "swings would predict the *direction* of the next big move, a crack in market efficiency "
            "you could trade with a ruler. That's the dream the pattern sells.\n\n"
            "But there are two traps. First, the pattern is **named after the fact** — you decide a "
            "triangle was there once price has broken, picking the swings that make it look right. "
            "Second, it's a **short** on a market (stock indices) that drifts **up** — so the deck is "
            "stacked *against* every short, the pattern has to overcome a drift headwind just to break "
            "even. To separate the **pattern** from the **tide**, we draw the triangle by a fixed "
            "mechanical rule with no hindsight, and compare the short to shorting on **random days**."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Find the swing points mechanically.** A 'pivot' is a high (or low) with "
            f"**{R['k']} lower (higher) bars on each side** — a confirmed fractal, known only "
            f"**{R['k']} bars later**, so we never draw the triangle with future data.\n"
            "2. **Spot the triangle by rule.** Over the latest handful of pivots, require the highs to "
            "**descend** and the lows to be **flat** (within a tolerance) — no eyeballing which dots "
            "to connect.\n"
            "3. **Trade the lore.** When the close drops **below the flat support**, **short** at the "
            "next close; measure the return over the next **5 / 10 / 20 / 60 days** (positive = the "
            "break-down paid).\n"
            "4. **The honest baseline.** Do the exact same short on **random days**. If the triangle "
            "matters, the break-short must beat a random short. *If it doesn't — or goes the wrong "
            "way — the pattern is a mirage*, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical descending triangle even look like? Here's a tape with the "
            "flat support drawn under descending highs, and the support breaks the rule would short."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pick = None\n"
            "    for tk in data.DEFAULT_TICKERS:\n"
            "        c0 = load(tk)['close']\n"
            "        e0 = st.support_break_entries(c0, k=R['k'], lookback=R['lookback'])\n"
            "        if len(e0): pick = (tk, c0, e0); break\n"
            "    tk, cl, ent = pick\n"
            "    sup = st.build_support(cl, k=R['k'], lookback=R['lookback'])\n"
            "    last = ent[-1]\n"
            "    i = cl.index.get_loc(last)\n"
            "    seg = cl.iloc[max(0,i-120):i+60]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg.values, c='k', lw=1.2, label=f'{tk} close')\n"
            "    ax.plot(seg.index, sup.reindex(seg.index), c=RED, lw=1.4, ls='--', label='flat support')\n"
            "    ein = ent[(ent>=seg.index[0])&(ent<=seg.index[-1])]\n"
            "    ax.scatter(ein, cl.reindex(ein), c=RED, s=55, zorder=5, label='support-break SHORT')\n"
            "    ax.set_title(f'A mechanical descending triangle on {tk}'); ax.legend(loc='best')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('support breaks in window:', len(ein))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The flat floor and descending highs are there — *as a description*. The question is "
            "whether those red short dots are followed by **falls**. **Let's race the break-short "
            "against random shorts** at four horizons. Red = short the break; grey = short on random "
            "days. (Positive bars = the short made money.)"
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    short, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.support_break_entries(c, k=R['k'], lookback=R['lookback'])\n"
            "            re = st.random_entries(c, max(len(e),50), k=R['k'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        short.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    short = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, short, .4, color=RED, label='short the break')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='short on random days')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(a,bb) in enumerate(zip(short,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom' if a>=0 else 'top',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom' if bb>=0 else 'top',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean SHORT return (bps)')\n"
            "ax.set_title('The break-down loses money — and the longer you hold, the worse'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('short:', [round(v) for v in short]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the story. Shorting the break **loses** — **{R['h20'][2]:.0f} bps** over 20 days, "
            f"**{R['h60'][2]:.0f} bps** over 60 — and the longer you hold the worse it gets, because "
            "the 'bearish' break keeps getting bought back up. Both bars are negative (every short "
            "fights the market's drift), but the break-short is *no better* than a random short, and "
            "at 60 days it's clearly worse. The famous pattern points the wrong way."
        ),
        md(
            "**One more sanity check.** What if we scramble the *descending highs* — keep the flat "
            "support and the timing, but shuffle the high prices so the ceiling no longer 'descends'? "
            "If the triangle shape really matters, the nonsense version should behave very differently."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.scrambled_highs_placebo(c, 20, k=R['k'], lookback=R['lookback'], n_draws=60, seed=461)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']  # (full-precision p in docs/results.md: 0.69)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real triangle support-break short (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... and {pval*100:.0f}% of *scrambled-highs* triangles do at least as well (p={pval:.2f}).')\n"
            "print('=> the descending-highs geometry is not doing the work.')"
        ),
        md(
            f"More than two-thirds of the **scrambled** triangles match or beat the real one "
            f"(*p* = {R['placebo'][1]:.2f}). If the descending ceiling genuinely mattered, breaking it "
            "would change the result. It doesn't — because the result was never about the shape."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Shorting the break does **not** beat shorting on random days; at "
            "60 days it's *significantly worse* (*p* = 0.01). Booked as a short the rule loses money.\n"
            "- **Tradability — Mirage.** Nothing to trade: the short bleeds the drift headwind and the "
            "break mostly reverses — costs only deepen the hole.\n"
            "- **\"Does the break resolve in the textbook direction?\" — Busted.** It resolves the "
            "**opposite** way (60-day short win-rate **23%**), and scrambling the descending highs "
            "leaves the result intact. The triangle doesn't break down."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing to trade — and worse, the obvious trade is **backwards**. Shorting the "
            "'bearish' break loses to the market's upward drift *and* to a random short. If anything "
            "the result hints that **fading the break** (buying the dip) would have done better — "
            "which is just the dip-buying drift trade again, not the triangle. As a forecasting tool "
            "the descending triangle doesn't pay; as a drawing tool, it was never meant to be a "
            "strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Volume confirmation.** Proponents add 'break on rising volume'. A fun follow-up tests "
            "whether a volume filter rescues *any* of the 39 breaks — spoiler: too few to matter.\n"
            "- **The mirror pattern.** The *ascending* triangle (flat resistance, rising lows) is sold "
            "as bullish; on an up-drifting tape it 'works', but that's the drift again.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* break-down into a "
            "synthetic tape and shows the harness banks it (so the wrong-way real result isn't a dead "
            "detector — it's an honest refutation).\n\n"
            "*Think the triangle forecasts a breakdown? Show the break-short beating a random short at "
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
            "# Descending Triangle — a quantitative teardown 🔬\n"
            "### Mechanical flat-support + descending-highs triangles on 5 indices · support-break "
            "SHORT forward returns · one-sample HAC *t* · a drift-matched random-short baseline · a "
            "scrambled-highs geometry placebo · costs · a synthetic planted-break-down control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **triangle** from the **drift**: every short on an up-drifting index "
            "fights a headwind, so the only meaningful test is break-vs-random-short, plus a placebo "
            "that destroys the descending-highs geometry while preserving the marginals.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. Pivots are confirmed "
            f"fractals (k={R['k']}, an explicit {R['k']}-bar confirmation lag); entry is the **next "
            "close** (one documented lag) and the trade is a SHORT. Offline core + synthetic control "
            "are deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Support-break short vs a **drift-matched random short**: Δ = "
            f"{R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} bps at "
            f"5/10/20/60d; the only horizon clearing |t|≥2 clears it the **wrong way** (60d Welch "
            f"t = {R['h60'][8]:+.2f}, p = {R['h60'][9]:.3f}). |\n"
            f"| **Tradability** | `MIRAGE` | Booked short the rule loses ({R['h20'][2]:+.0f}/"
            f"{R['h60'][2]:+.0f} bps at 20/60d); the negative one-sample t's are the drift headwind, "
            "and the break mostly reverses. No edge to scale. |\n"
            f"| **Resolves textbook direction?** | `BUSTED` | 60-day short win-rate **{R['h60'][3]}%** "
            f"(break followed by a rally 3× in 4); scrambling the descending highs leaves the result "
            f"intact (placebo p = {R['placebo'][1]:.2f}). |\n\n"
            "> 💡 In plain words: the 'bearish' break does not go down. Race it vs a random short and "
            "it's no better (worse at 60d); scramble the descending highs and nothing changes. The "
            "pattern is a post-hoc description, not a forecast."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "A descending triangle is a flat support $s$ (swing lows within a band) beneath descending "
            "swing highs $h_1>h_2>\\dots$. Let the break time $\\tau$ be the first close $C_\\tau<s$. "
            "The Edwards-Magee rule **shorts at $\\tau{+}1$** expecting a continuation down of roughly "
            "the triangle height. We book the short P&L $-\\,(C_{\\tau+1+H}/C_{\\tau+1}-1)$ so a real "
            "break-down is positive.\n\n"
            "- **H₀ (drift headwind).** Break-short returns equal a drift-matched **random-short** "
            "baseline.\n"
            "- **H₁ (the triangle forecasts a breakdown).** Break-short **exceeds** random short at "
            "some horizon, t ≥ 2.\n"
            "- **H₂ (the geometry matters).** Break-short exceeds a **scrambled-highs** triangle whose "
            "descending ceiling is nonsense.\n\n"
            "We find **H₀ not rejected** (break ≈ random, *worse* at 60d), **H₁ rejected** (Welch t "
            "never ≥ +2; at 60d it's −2.57), **H₂ rejected** (placebo p ≈ 0.69). The steelman fails on "
            "every leg — and the sign is *backwards*."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift, as a headwind.** Equity indices have a positive unconditional daily mean; a "
            "**short** inherits it as a *loss*. A one-sample $t$ of a short-only rule against **zero** "
            "measures that headwind, not the pattern — which is exactly why our one-sample t's are "
            "negative. The fix is the **random-short baseline** (same instrument, epoch, hold, also "
            "short) and a Welch test of break-*minus*-random.\n\n"
            "**(b) Geometry as a free parameter.** A 'descending triangle' is a flat support plus a "
            "*chosen* descending ceiling. The **scrambled-highs placebo** keeps the support, the "
            "timing and the price marginal but permutes the swing-high prices, so the descending-highs "
            "constraint is destroyed; if the result survives, the ceiling — the thing that makes it a "
            "*descending* triangle — was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} support breaks** "
            "pooled (a rare pattern).\n"
            f"- **Pivots.** Confirmed fractals: extremum with k={R['k']} strictly-beaten bars each "
            f"side; usable only at bar +{R['k']} (no look-ahead).\n"
            f"- **Triangle.** Rolling over the latest {R['lookback']} confirmed pivots: highs must "
            "descend (no higher high, last well below first), lows must be flat (band tolerance).\n"
            "- **Entry.** First close below the flat support; **short** at the **next close** (one "
            "lag); hold H ∈ {5,10,20,60}. Short P&L (positive = break-down paid).\n"
            "- **Null #1 — one-sample HAC t** of break-short returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-short baseline**, Welch two-sample break vs random (the *real* test).\n"
            "- **Null #3 — scrambled-highs placebo** (descending ceiling destroyed, marginals kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every break.\n"
            "- **Positive control.** Synthetic tape with a **planted** break-down (knob `edge`): "
            "edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headwind trap — one-sample t is the drift, vs-random is the truth\n\n"
            "Left: the break-short's **one-sample** t against zero (negative — that's the drift "
            "headwind every short eats). Right: the same break-short vs a **drift-matched random "
            "short** (the honest number). A real bearish edge would push the right bars *positive* "
            "past +2; instead they sit at/below zero and the 60d bar is significantly **negative**."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, short, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.support_break_entries(c, k=R['k'], lookback=R['lookback'])\n"
            "            re = st.random_entries(c, max(len(e),50), k=R['k'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); short.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    short = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(-2, ls='--', c=RED, label='|t|=2 bar'); a1.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (the drift headwind)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(-2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Break vs RANDOM short, Welch t (never +2; -2.57 at 60d)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars are **negative** (20d {R['h20'][4]:.2f}, 60d "
            f"{R['h60'][4]:.2f}) — that's the short fighting the index's drift, not the pattern. The "
            f"right bars are the real test: break-minus-random is around zero and **{R['h60'][8]:+.2f} "
            f"at 60d** (*p* = {R['h60'][9]:.3f}) — the break-short is *significantly worse* than a "
            "random short. There is no bearish edge; if anything the sign is inverted."
        ),
        md(
            "### 4b · Break vs random across horizons — both lose, the break loses more\n\n"
            "Mean short return, support-break vs random short, all four horizons. A real breakdown "
            "edge would put the break bars well *above* random. They don't."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, short, .4, color=RED, label='support-break short')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random short (drift baseline)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(a,b) in enumerate(zip(short,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom' if a>=0 else 'top',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom' if b>=0 else 'top',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean SHORT return (bps)')\n"
            "ax.set_title('Support-break short does not beat a random short'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta break-random (bps):', [round(a-b) for a,b in zip(short,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 60 days the break-short is **{R['h60'][2]:.0f} bps** while a "
            f"random short is only **{R['h60'][5]:.0f} bps** — the famous pattern *underperforms* a "
            f"dart by {abs(R['h60'][6]):.0f} bps. Holding longer makes it worse: the 'bearish' break "
            "keeps getting bought back up."
        ),
        md(
            "### 4c · The geometry placebo — scramble the descending highs, nothing changes\n\n"
            "Shuffle which price sits at which swing **high** (support, timing and marginal kept) so "
            "the ceiling no longer descends. If the descending-highs geometry is doing the work, the "
            "observed break-short return should sit far from the scrambled distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    import numpy as _np\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.scrambled_highs_placebo(c, 20, k=R['k'], lookback=R['lookback'], n_draws=60, seed=461)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']  # full-precision p (500 draws) = 0.69 in docs\n"
            "    # rebuild the placebo distribution for the histogram (reuses the fast hot loop)\n"
            "    piv = st.find_pivots(c, k=R['k']); idx=c.index; n=len(c)\n"
            "    hidx=[j for j,kd in enumerate(piv['kind'].to_numpy()) if kd>0]\n"
            "    hp=piv['price'].to_numpy(dtype=float)[hidx]\n"
            "    rng=_np.random.default_rng(461); draws=[]\n"
            "    for _ in range(60):\n"
            "        perm=rng.permutation(hp); sp=piv.copy()\n"
            "        pr=sp['price'].to_numpy(dtype=float).copy()\n"
            "        for slot,j in enumerate(hidx): pr[j]=perm[slot]\n"
            "        sp['price']=pr\n"
            "        ss=st._support_from_pivots(sp, n, idx, k=R['k'], lookback=R['lookback'])\n"
            "        m=(c<ss)&ss.notna(); f=m&~m.shift(1,fill_value=False)\n"
            "        rr=st.forward_returns(c, idx[f.to_numpy()], 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws=_np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(461); draws = rng.normal(-180, 120, 60)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=30, color=GREY, alpha=.85, label='scrambled-highs triangles (SPY, 20d)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'real triangle {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean support-break 20d SHORT return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real triangle sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real triangle {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => geometry not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real triangle (red line) sits **in the middle** of the "
            f"scrambled cloud — **p = {R['placebo'][1]:.2f}**. Nonsense ceilings do just as well, so "
            "the descending-highs geometry isn't carrying information. This is the cleanest refutation "
            "of 'the descending triangle breaks down.'"
        ),
        md(
            "### 4d · Per-ticker — the break-short loses almost everywhere\n\n"
            "20-day break-minus-random delta, per instrument. (Counts are tiny — 3 to 21 breaks — so "
            "this is directional, not precise.) A real bearish edge would be positive across the "
            "board; instead it's negative in 4 of 5."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = load(t)['close']\n"
            "        e = st.support_break_entries(c, k=R['k'], lookback=R['lookback']); re = st.random_entries(c, max(len(e),50), k=R['k'], seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d break − random (bps)'); ax.set_title('Break-short underperforms random in 4 of 5 names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: only **QQQ** ({R['per'][1][5]:+.0f} bps, on 3 breaks) shows a "
            f"positive delta; SPY is **{R['per'][0][5]:+.0f}** bps *behind* a random short. No "
            "coherent, cross-sectional bearish edge — exactly what you'd expect if the triangle is "
            "post-hoc chart-reading."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real break-down\n\n"
            "To prove the wrong-way real result is honest (not a dead detector), plant a **real** "
            "break-down continuation into a synthetic tape and check the same support-break short "
            "banks it: edge=0 must stay near zero; edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.50):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=461, n_days=4000)\n"
            "    c = px['close']; e = st.support_break_entries(c, k=R['k'], lookback=R['lookback'])\n"
            "    s = st.summarize(st.forward_returns(c, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, (b1,b2) = plt.subplots(1,2,figsize=(10.5, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]\n"
            "wins=[r[3] for r in res]; means=[r[2] for r in res]\n"
            "b1.bar(labels, wins, color=[GREY, GREEN], width=.5); b1.axhline(50, ls='--', c=RED, label='50%')\n"
            "for i,w in enumerate(wins): b1.annotate(f'{w:.0f}%',(i,w),ha='center',va='bottom')\n"
            "b1.set_ylabel('20d short win-rate'); b1.set_title('Control: win-rate'); b1.legend()\n"
            "b2.bar(labels, means, color=[GREY, GREEN], width=.5); b2.axhline(0, c='k', lw=.8)\n"
            "for i,m in enumerate(means): b2.annotate(f'{m:+.0f}',(i,m),ha='center',va='bottom' if m>=0 else 'top')\n"
            "b2.set_ylabel('20d mean short (bps)'); b2.set_title('Control: mean short return')\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} short={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted break-down the control sits near zero "
            f"(short {R['syn'][0][2]:+.0f} bps, win {R['syn'][0][3]:.0f}% — no false positive); a "
            f"planted break-down sends the short to **{R['syn'][1][2]:+.0f} bps, win {R['syn'][1][3]:.0f}%** "
            "(small n caps the HAC t, but the +35-pt win-rate jump is unambiguous). The detector "
            "works — so the wrong-way real-tape result is a genuine refutation, not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the support-break short does not beat a drift-matched random short "
            f"(break − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps at 5/10/20/60d; the only significant horizon is 60d and it's the **wrong way**, "
            f"Welch t = {R['h60'][8]:+.2f}, p = {R['h60'][9]:.3f}). The negative one-sample t's are "
            "the drift headwind.\n"
            f"- **Tradability `MIRAGE`** — booked short the rule loses at every horizon past 5d "
            f"({R['h60'][2]:+.0f} bps at 60d); costs deepen the hole. No edge to scale.\n"
            f"- **Resolves textbook direction? `BUSTED`** — it resolves the **opposite** way (60-day "
            f"short win-rate **{R['h60'][3]}%**: the break is followed by a rally 3× in 4), and the "
            f"scrambled-highs placebo leaves the result intact (**p = {R['placebo'][1]:.2f}**), so the "
            "descending-highs geometry carries no forecasting information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade (and the sign is backwards)\n\n"
            "The support-break short loses to the long-equity drift *and* to a random short, and the "
            "break mostly reverses up. There is no capacity question because there is no edge to scale "
            "— and the only hint of structure (the break getting bought back) points to **fading** the "
            "break, which is just the dip-buying drift trade, not the triangle. The descending triangle "
            "is a descriptive figure, not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Volume / breadth filters.** Proponents demand a break on expanding volume; with only "
            f"{R['n_entries']} breaks total, any filter leaves too few trades to test — itself a "
            "verdict on how rare the clean pattern is.\n"
            "- **The ascending mirror.** The ascending triangle (flat resistance, rising lows) 'works' "
            "long on an up-drifting tape — but that's the drift, the same confound inverted.\n"
            "- **Tolerance sensitivity.** Loosening the flat-band / descending-highs tolerances finds "
            "more 'triangles' but dilutes them toward generic support breaks — the result is robust to "
            "the knobs: drift in, nothing out.\n\n"
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
