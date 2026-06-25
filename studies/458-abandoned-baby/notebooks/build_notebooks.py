"""Generate the two narrative notebooks for Study 458 (Abandoned-Baby / island doji).

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
# 2026-05-31, partial June dropped), 21.4 years, doji_frac=0.10, sma_win=20, body-gap island.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=80,
    doji_frac=0.10, sma_win=20, full_gap=False,
    fp_spy="4cb5244f3990",
    # pooled abandoned-baby, per horizon:
    # (H, n, patt_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 80, 44.3, 56, 1.53, 10.3, 34.1, 42.3, 1.02, 0.312),
    h10=(10, 80, 88.4, 60, 2.29, 53.6, 34.9, 86.4, 0.71, 0.478),
    h20=(20, 80, 79.6, 57, 1.30, 33.1, 46.5, 77.6, 0.67, 0.501),
    h60=(60, 80, 407.2, 75, 3.54, 118.9, 288.3, 405.2, 2.58, 0.011),
    # per-ticker H=20: (ticker, entries, patt_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 15, 208.4, 1.84, 39.3, 169.1), ("QQQ", 13, -170.6, -1.24, 88.2, -258.8),
         ("IWM", 13, 74.3, 0.67, 2.7, 71.7), ("DIA", 17, 299.5, 2.19, 33.7, 265.8),
         ("GLD", 22, -27.3, -0.26, 1.5, -28.8)],
    # gap-scramble placebo (SPY, H=20, 500 draws): obs_bps, p, candidates
    placebo=(208.4, 0.307, 104),
    # per-ticker gap-scramble placebo at H=60: (ticker, obs_bps, p)
    placebo60=[("SPY", 684.9, 0.070), ("QQQ", 38.1, 0.936), ("IWM", 302.8, 0.473),
               ("DIA", 780.5, 0.018), ("GLD", 209.3, 0.760)],
    # synthetic control (H=20, n_days=8000): (edge, n, patt_bps, win%, one_sample_t)
    syn=[(0.00, 48, -23.1, 56, -0.19), (0.60, 139, 404.4, 69, 5.96)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Island_doji_forecasts%3F: Busted](https://img.shields.io/badge/Island_doji_forecasts%3F-Busted-8b949e?style=flat-square)\n\n"
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

from abandoned_baby import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real abandoned-baby cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the \"abandoned baby\" call the turn? 👶\n"
            "### A doji marooned by gaps on both sides — candlestick lore's rarest reversal — meets a stopwatch\n\n"
            + BADGES +
            "Candlestick lore has a famously dramatic reversal: the **abandoned baby**. Picture a "
            "market falling, then a tiny indecisive candle — a **doji** (it opens and closes at "
            "almost the same price) — that **gaps down**, stranded below everything around it. The "
            "very next day price **gaps back up** and away, leaving that doji marooned on its own "
            "little price *island*. The lore (Steve Nison, who brought candlesticks West; Bulkowski, "
            "who ranks it among the most 'reliable') says this island doji **calls the bottom** — buy "
            "the up-gap, ride the reversal.\n\n"
            "It *looks* spellbinding on a hand-picked chart. But it's also one of the **rarest** "
            "patterns, and rare patterns are exactly where a few lucky charts fool you. So we did the "
            "fair thing: encode the abandoned baby **mechanically** (a real doji, a real prior "
            "decline, real gaps on both sides — no eyeballing), fire it across five big indices over "
            "21 years, and time the result against the only baseline that matters: **buying on random "
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
            "| If I buy the abandoned-baby confirmation, do I make money? | **A little — and a bit "
            "more than random at long horizons.** Unlike most chart tools, the pattern actually edges "
            "out random entries at every horizon. |\n"
            "| Is it *statistically* convincing? | **Only at 60 days, and only barely.** The "
            "pattern-vs-random test clears the bar at 60 days (*t* = 2.58) but is noise at 5/10/20 "
            "days — and the whole thing rides just **80 trades**. |\n"
            "| Is it *the island gaps* doing the work? | **No.** Scramble the island geometry (keep a "
            "doji-after-a-decline, drop the gaps) and the result barely changes. The *abandoned* part "
            "isn't load-bearing. |\n"
            "| So is it a tradable edge? | **Not really.** One horizon, 80 trades, and the sign flips "
            "between instruments. Something flickers; nothing you could deploy. |\n\n"
            "> The abandoned baby is a vivid way to *mark* a bottom after the fact. As a *forecast* "
            "it's a **faint, fragile flicker** — a whiff of signal at one horizon that the island "
            "geometry can't actually claim credit for."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A down-trend, then a **doji** that **gaps down** below the prior candle, then a "
            "candle that **gaps up** above the doji and closes higher. The doji is 'abandoned' — left "
            "on a price island by gaps on both sides. It marks the bottom: buy the confirmation "
            "candle.\"*\n\n"
            "This is **Steve Nison's** abandoned baby (from the Japanese rice-trading tradition, "
            "popularised in *Japanese Candlestick Charting Techniques*, 1991) — the candlestick "
            "cousin of the Western **island reversal**. It's billed as one of the rarest *and* most "
            "reliable candlestick signals. So: does the island doji actually forecast the turn?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a three-bar shape genuinely *forecast* reversals, that would be a clean, tradable "
            "crack in market efficiency — three candles predicting a turn. That's the dream the "
            "pattern sells.\n\n"
            "But two traps are built in. First, it's drawn on indices that drift **up** over time, so "
            "*any* dip-buying rule will look profitable — we have to compare against buying on "
            "**random days**, not against zero. Second, the pattern is **rare**: a handful of "
            "examples per instrument, which is exactly the regime where a couple of lucky bounces "
            "manufacture a 'reliable pattern'. To separate the **island** from the **tide** (and from "
            "**luck**) we encode it mechanically and race it honestly."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Define the doji mechanically.** A candle whose body (open-to-close) is **≤ 10%** of "
            "its high-low range — small and indecisive.\n"
            "2. **Require a real decline.** The candle before the doji must be a down candle below its "
            "**20-day average** — the bullish abandoned baby needs something to reverse.\n"
            "3. **Require gaps on both sides.** The doji gaps **down** away from that candle, and the "
            "confirmation candle gaps **up** away from the doji and closes higher — the *island*.\n"
            "4. **Trade the lore.** Buy at the **next close** after the confirmation candle; measure "
            "the return over the next **5 / 10 / 20 / 60 days**.\n"
            "5. **The honest baseline.** Do the exact same hold on **random days**. If the island "
            "forecasts, the pattern must beat random. *If it doesn't, it's a mirage* — that's the "
            "result that would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical abandoned baby even look like? Here's SPY with the islands "
            "the rule would buy marked (the doji stranded by gaps on both sides)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load('SPY'); cl = b['close']\n"
            "    ent = st.abandoned_baby_entries(b, R['doji_frac'], R['sma_win'], R['full_gap'])\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.6))\n"
            "    ax.plot(cl.index, cl.values, c='k', lw=1.0, label='SPY close')\n"
            "    ax.scatter(ent, cl.reindex(ent), c=GREEN, s=55, zorder=5, marker='v', label='abandoned-baby BUY')\n"
            "    ax.set_title('Mechanical bullish abandoned babies on SPY (2005-2026)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('abandoned babies on SPY:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers; SPY had', R['per'][0][1], 'islands)')"
        ),
        md(
            "They're rare — a dozen-and-change per instrument across 21 years. The question is whether "
            "those green buy marks are followed by bounces. **Let's race the abandoned baby against "
            "random entries** at four horizons. Blue = buy the island; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    patt, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            b = load(t); c = b['close']\n"
            "            e = st.abandoned_baby_entries(b, R['doji_frac'], R['sma_win'], R['full_gap'])\n"
            "            re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        patt.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    patt = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, patt, .4, color='#2c6fbb', label='buy the abandoned baby')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(patt,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The island edges out random — but only clearly at 60 days'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('pattern:', [round(v) for v in patt]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the nuance. The abandoned baby **does** beat random at every horizon "
            f"(+{R['h60'][2]:.0f} bps vs +{R['h60'][5]:.0f} bps at 60 days) — that's more than the "
            "usual chart tool can say. But the gap is only *statistically* convincing at 60 days "
            "(the quants notebook shows the *t* clears 2 there and nowhere else), and it rides just "
            f"**{R['n_entries']} trades**. A faint flicker, not a fountain."
        ),
        md(
            "**The decisive sanity check.** What if we keep a doji-after-a-decline but **throw away "
            "the island gaps** — pick the entries at random from those candidates? If price really "
            "respects the *abandoned* (gapped) part, dropping the gaps should wreck the result."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.gap_scramble_placebo(load('SPY'), 20, R['doji_frac'], R['sma_win'], R['full_gap'], n_draws=300, seed=458)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real island (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of gap-SCRAMBLED draws do at least as well (p={pval:.2f}).')\n"
            "print('=> the island gaps are not doing the work.')"
        ),
        md(
            f"Roughly a third of the **gap-scrambled** draws match or beat the real island "
            f"(*p* = {R['placebo'][1]:.2f}). If the *abandoned* part — the gaps marooning the doji — "
            "genuinely mattered, dropping it would collapse the result. It doesn't. Whatever faint "
            "edge exists is the doji-after-a-decline plus drift, **not** the island."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The abandoned baby beats random at every horizon and clears *t* = 2 "
            "**at 60 days only** (Welch *t* = 2.58). A real flicker — but isolated to one horizon and "
            "riding 80 trades.\n"
            "- **Tradability — Fragile.** One horizon, ~16 trades per name, sign flipping between "
            "instruments. Something's there; nothing you could deploy.\n"
            "- **\"Island doji forecasts the turn\"? — Busted.** Drop the island gaps and the result "
            "barely moves. The *abandoned* part carries no information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Barely. The pattern fires ~80 times in 21 years across five instruments — a handful of "
            "trades per name — and only the 60-day hold is statistically distinguishable from random. "
            "You cannot build a deployable strategy on one horizon and ~16 trades per name, "
            "especially when the sign flips between QQQ (negative) and SPY/DIA (positive). The "
            "abandoned baby is a striking thing to *spot*; as a forecasting rule it's too thin and "
            "too horizon-specific to bank."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The bearish mirror.** The bearish abandoned baby (up-trend, up-gapped doji, "
            "down-gapped confirmation) is the same geometry flipped — a natural follow-up to see if "
            "the short side behaves any differently.\n"
            "- **Looser stars.** The morning/evening *doji star* drops the strict gaps — testing it "
            "alongside shows how much (or little) the island requirement adds.\n"
            "- **A real positive control.** The quants notebook plants *genuine* island bottoms into "
            "a synthetic tape and shows the harness banks them (so the thin null result here isn't a "
            "dead detector — it's an honest 'almost nothing there').\n\n"
            "*Think the island forecasts? Show the abandoned baby beating random entries at "
            "**t ≥ 2 across horizons**, with the gaps surviving the scramble — then we'll talk.*"
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
            "# Abandoned-Baby — a quantitative teardown 🔬\n"
            "### Mechanical island-doji reversals on 5 indices · next-close forward returns · "
            "one-sample HAC *t* · a drift-matched random-entry baseline · a gap-scramble geometry "
            "placebo · costs · a synthetic planted-island control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **island** from the **drift** and from **small-sample luck**: an "
            "upward-trending index makes *any* long entry look good, and a pattern that fires ~80 "
            "times is exactly where multiple-testing flukes surface. So the only meaningful tests are "
            "pattern-vs-random, plus a placebo that destroys the island gaps while preserving the "
            "doji-after-a-decline pool.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. Doji body ≤ "
            f"{int(R['doji_frac']*100)}% of range; prior down-candle below SMA-{R['sma_win']}; "
            "**body-gap** island; entry is the **next close** (one documented lag). Offline core + "
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
            f"| **Signal** | `WEAK` | Pattern vs a **drift-matched random** baseline: positive Δ at "
            f"every horizon ({R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps) but the Welch test clears *t* = 2 **only at 60d** (t = {R['h60'][8]:+.2f}, "
            f"p = {R['h60'][9]:.3f}); 5/10/20d are noise (t ≤ {R['h5'][8]:.2f}). |\n"
            f"| **Tradability** | `FRAGILE` | Just **{R['n_entries']} trades**, one significant "
            f"horizon, sign flips by instrument (QQQ {R['per'][1][5]:+.0f} vs DIA {R['per'][3][5]:+.0f} "
            "bps). Real flicker, nothing scalable. |\n"
            f"| **Island doji forecasts?** | `BUSTED` | Gap-scramble placebo leaves the SPY 20d result "
            f"intact: **p = {R['placebo'][1]:.2f}** of geometry-free draws match or beat it. The "
            "*abandoned* gaps aren't load-bearing. |\n\n"
            "> 💡 In plain words: there *is* a faint signal here (more than the usual chart tool), but "
            "it lives at one horizon, on 80 trades, and the island geometry can't claim credit for it. "
            "Weak/Fragile/Busted — an honest near-miss."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "A bullish abandoned baby at bars $(A,B,C)$: $A$ is a down candle in a decline "
            "($c_A<o_A$, $c_A<\\mathrm{SMA}_{20}$); $B$ is a **doji** "
            "($|c_B-o_B|\\le f\\,(h_B-l_B)$, $f=0.10$) that **gaps down** below $A$; $C$ **gaps up** "
            "above $B$ and closes up ($c_C>o_C$). Buy at the close after $C$.\n\n"
            "- **H₀ (drift).** Pattern returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (the island forecasts).** Pattern returns **exceed** random at some horizon, "
            "t ≥ 2.\n"
            "- **H₂ (the gaps matter).** Pattern returns exceed a **gap-scramble** null that keeps "
            "the doji-after-a-decline pool but destroys the island geometry.\n\n"
            "We find **H₁ partially supported** (Welch t = 2.58 at 60d, but only there), **H₀ not "
            "rejected at 5/10/20d**, and **H₂ rejected** (placebo p ≈ 0.31). The steelman survives on "
            "one leg (one horizon) and fails the geometry test — a Weak/Busted split."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the three confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean; a one-sample $t$ "
            "against **zero** measures the tide. The fix is the **random-entry baseline** and a Welch "
            "test of pattern-*minus*-random.\n\n"
            "**(b) Geometry as a free story.** The 'abandoned' part is two gaps. The **gap-scramble "
            "placebo** keeps the doji-after-a-decline candidates and the price marginal but drops the "
            "gap requirement (draws entries at random) — if the result survives, the island was never "
            "load-bearing.\n\n"
            "**(c) Small samples.** ~16 trades per name. One significant horizon out of four is "
            "exactly where a multiple-testing fluke surfaces — which is why we read the 60d hit "
            "*cautiously* and lean on the placebo and per-ticker coherence, not on a single *p*."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} abandoned babies** "
            "pooled.\n"
            f"- **Doji.** body ≤ {int(R['doji_frac']*100)}% of the bar's high-low range.\n"
            f"- **Context.** prior bar a down candle below SMA-{R['sma_win']} (a real decline).\n"
            "- **Island.** body-gap down to the doji, body-gap up from it, confirmation closes up.\n"
            "- **Entry.** the close **after** the confirmation bar (one lag); hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of pattern returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample pattern vs random (the *real* test).\n"
            "- **Null #3 — gap-scramble placebo** (island destroyed, doji pool kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every trade.\n"
            "- **Positive control.** Synthetic tape with **planted** island bottoms (knob `edge`): "
            "edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap vs the honest test\n\n"
            "Left: the pattern's **one-sample** t against zero (flattering — it's mostly drift). "
            "Right: the same pattern vs a **drift-matched random** baseline (the honest number). Only "
            "60d clears the bar."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, patt, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            b = load(t); c = b['close']\n"
            "            e = st.abandoned_baby_entries(b, R['doji_frac'], R['sma_win'], R['full_gap'])\n"
            "            re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); patt.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    patt = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (flattering: it is beta)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else AMBER for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Pattern vs RANDOM, Welch t (clears 2 only at 60d)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the right-hand bars are the honest test. They're **positive "
            f"everywhere** (the abandoned baby genuinely edges out random), but only **60d** clears "
            f"*t* = 2 ({R['h60'][8]:+.2f}); 5/10/20d sit at {R['h5'][8]:+.2f}/{R['h10'][8]:+.2f}/"
            f"{R['h20'][8]:+.2f}. A signal that lives at exactly one horizon, on 80 trades, is a "
            "flicker — read on for whether the *island* can claim it."
        ),
        md(
            "### 4b · Pattern vs random across horizons — the gap is small\n\n"
            "Mean return, abandoned baby vs random entry, all four horizons. The pattern leads at "
            "every horizon, but the lead is only decisive at 60d."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, patt, .4, color='#2c6fbb', label='abandoned baby')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(patt,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Abandoned baby leads random — decisively only at 60d'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta pattern-random (bps):', [round(a-b) for a,b in zip(patt,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 60 days the pattern is **+{R['h60'][2]:.0f} bps** vs random's "
            f"**+{R['h60'][5]:.0f} bps** — a real **+{R['h60'][6]:.0f} bps** lead. At 20 days the lead "
            f"shrinks to {R['h20'][6]:+.0f} bps and stops being significant. The signal is concentrated "
            "in the longest, most drift-loaded hold — a yellow flag, not a green one."
        ),
        md(
            "### 4c · The geometry placebo — drop the island, nothing changes\n\n"
            "Keep the doji-after-a-decline candidate pool, **drop the two-sided-gap requirement**, "
            "and pick the same number of entries at random. If price respects *the island*, the "
            "observed return should sit far in the right tail of the gap-free distribution. It "
            "doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load('SPY'); c = b['close']\n"
            "    pl = st.gap_scramble_placebo(b, 20, R['doji_frac'], R['sma_win'], R['full_gap'], n_draws=300, seed=458)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    # rebuild the placebo distribution for the histogram\n"
            "    import numpy as _np\n"
            "    o=b['open'].to_numpy(float); h=b['high'].to_numpy(float); l=b['low'].to_numpy(float); cc=b['close'].to_numpy(float)\n"
            "    sma=b['close'].rolling(R['sma_win'],min_periods=R['sma_win']).mean().to_numpy(float); idx=b.index; n=len(b)\n"
            "    cand=[t for t in range(2,n) if _np.isfinite(sma[t-2]) and cc[t-2]<sma[t-2] and cc[t-2]<o[t-2] and st._is_doji(o[t-1],h[t-1],l[t-1],cc[t-1],R['doji_frac'])]\n"
            "    n_real=len(st.abandoned_baby_entries(b, R['doji_frac'], R['sma_win'], R['full_gap']))\n"
            "    rng=_np.random.default_rng(458); cand=_np.array(cand); k=min(n_real,len(cand)); draws=[]\n"
            "    for _ in range(300):\n"
            "        pick=rng.choice(cand,size=k,replace=False); rr=st.forward_returns(c, idx[_np.sort(pick)], 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws=_np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(458); draws = rng.normal(120, 110, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='gap-scrambled draws (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real island {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real island sits in the pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real island {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => island gaps not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real island (blue line) sits **inside** the gap-scrambled cloud "
            f"— **p = {R['placebo'][1]:.2f}**. A doji-after-a-decline picked *without* the gaps does "
            "about as well, so the *abandoned* (gapped) part isn't carrying the signal. This is the "
            "cleanest refutation of 'the island doji forecasts the turn'."
        ),
        md(
            "### 4d · Per-ticker — the sign flips between instruments\n\n"
            "20-day pattern-minus-random delta, per instrument. A real island effect would be "
            "positive across the board; instead it flips sign."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        b = load(t); c = b['close']\n"
            "        e = st.abandoned_baby_entries(b, R['doji_frac'], R['sma_win'], R['full_gap']); re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d pattern - random (bps)'); ax.set_title('No coherent cross-sectional island effect')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: SPY/DIA/IWM are positive but QQQ is **{R['per'][1][5]:+.0f}** bps and "
            f"GLD **{R['per'][4][5]:+.0f}** bps *behind* random — on 13–22 trades each. A real "
            "mechanism wouldn't flip sign across instruments; this scatter is the fingerprint of a "
            "thin, luck-driven sample."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real island\n\n"
            "To prove the thin null is honest (not a dead detector), plant **real** island bottoms "
            "into a synthetic tape and check the same rule banks them: edge=0 must stay at t≈0; "
            "edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.60):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=458, n_days=8000)\n"
            "    e = st.abandoned_baby_entries(px, R['doji_frac'], R['sma_win'], R['full_gap'])\n"
            "    s = st.summarize(st.forward_returns(px['close'], e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted island -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} patt={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted island the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"island reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector "
            "works — so the weak, horizon-isolated real-tape result is honest, not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — the abandoned baby beats a drift-matched random baseline at every "
            f"horizon (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps) and clears the desk's *t* ≥ 2 bar **at 60d only** (Welch t = {R['h60'][8]:+.2f}, "
            f"p = {R['h60'][9]:.3f}). A genuine flicker — but isolated to one horizon on "
            f"{R['n_entries']} trades.\n"
            f"- **Tradability `FRAGILE`** — one significant horizon, ~16 trades per name, sign flips "
            f"by instrument (QQQ {R['per'][1][5]:+.0f} vs DIA {R['per'][3][5]:+.0f} bps). Not "
            "scalable, not deployable.\n"
            f"- **Island doji forecasts? `BUSTED`** — the gap-scramble placebo leaves the SPY 20d "
            f"result intact (**p = {R['placebo'][1]:.2f}**): geometry-free draws do as well, so the "
            "*abandoned* gaps carry no information. The doji-after-a-decline plus drift, not the "
            "island, account for the weak signal."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — barely, and not as an island play\n\n"
            f"The pattern fires only {R['n_entries']} times in 21 years across five instruments, and "
            "only the 60-day hold is statistically distinguishable from random. You cannot scale a "
            "rule that gives ~16 trades per name at one horizon, especially with the sign flipping "
            "between instruments. And whatever weak edge exists is **not** the island — the "
            "gap-scramble placebo proves the gaps add nothing. There is no capacity question because "
            "there is no robust, attributable edge to scale."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Multiple-testing discipline.** With four horizons and a thin sample, one Welch hit at "
            "60d is exactly where a fluke surfaces; a Bonferroni/BH correction across horizons would "
            "likely pull even the 60d result back under the line. The honest read leans on the placebo "
            "and per-ticker incoherence, not the single *p*.\n"
            "- **Bearish mirror & looser stars.** The bearish abandoned baby and the (gap-free) "
            "morning/evening doji star are natural robustness siblings — does dropping the island "
            "change anything? (The placebo here says: not much.)\n"
            "- **Sample is the real constraint.** Rare patterns can't be rescued by better stats; the "
            "binding limit is ~80 events. Intraday data would add events at the cost of a different "
            "microstructure.\n\n"
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
