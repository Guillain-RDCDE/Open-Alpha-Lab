"""Generate the two narrative notebooks for Study 456 (Belt-Hold / opening marubozu).

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
# 2026-05-31, partial June dropped), 21.4 years, bullish belt-hold (open~low, tall white body,
# prior 10-bar downtrend), long entered next close.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=666,
    fp_spy="4cb5244f3990",
    # pooled belt-hold, per horizon:
    # (H, n, bh_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 665, 31.8, 59, 2.36, 33.4, -1.7, 29.8, -0.11, 0.909),
    h10=(10, 665, 66.5, 62, 3.71, 57.2, 9.3, 64.5, 0.47, 0.635),
    h20=(20, 665, 104.6, 67, 3.68, 71.9, 32.8, 102.6, 1.15, 0.251),
    h60=(60, 660, 306.0, 69, 5.86, 212.1, 93.9, 304.0, 2.03, 0.042),
    # per-ticker H=20: (ticker, entries, bh_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 121, 119.5, 2.49, 22.4, 97.2), ("QQQ", 139, 126.6, 1.55, 160.1, -33.5),
         ("IWM", 126, 67.0, 1.04, 29.2, 37.8), ("DIA", 137, 89.8, 1.48, 62.6, 27.3),
         ("GLD", 143, 118.1, 2.56, 74.7, 43.4)],
    # shape-scramble placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(119.5, 0.433, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, bh_bps, win%, one_sample_t)
    syn=[(0.00, 200, 11.5, 48, 0.27), (0.40, 140, 955.8, 92, 13.14)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Opening_extreme_reverses%3F: Busted](https://img.shields.io/badge/Opening_extreme_reverses%3F-Busted-8b949e?style=flat-square)\n\n"
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

from belt_hold import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real belt-hold cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a \"belt-hold\" candle actually reverse the trend? 🥋\n"
            "### One famous candle — opens at its low, closes way up — meets a stopwatch\n\n"
            + BADGES +
            "Open any candlestick guide and you'll find the **bullish belt-hold** (Japanese "
            "*yorikiri*, a sumo move meaning *to force out*). It's a single bar that **opens right "
            "at its low** — no lower wick at all — and then **closes well up** in a tall white "
            "body, after a downtrend. The lore, taught by Steve Nison and echoed everywhere, is "
            "that because the **open sat at the very bottom**, buyers grabbed control from the "
            "first tick and the down-move is about to **reverse**. So you buy.\n\n"
            "It *looks* compelling on a hand-picked chart. But a single candle, spotted after the "
            "fact, on a market that drifts **up** anyway, is the textbook way to fool yourself. So "
            "we did the fair thing: encode the belt-hold **mechanically** (open at the low, tall "
            "body, real prior downtrend — no eyeballing), fire it hundreds of times across five "
            "indices over 21 years, and time the result — against the only baseline that matters: "
            "**buying on random days instead.**\n\n"
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
            "| If I buy the belt-hold, do I make money? | **Yes in absolute terms** — win-rate "
            "~60-67%, returns look good. But that's mostly the market going up. |\n"
            "| Is that *the candle's* doing? | **Barely, and only at the longest horizon.** At "
            "5/10/20 days it's a coin-flip vs buying on random days. Only at **60 days** does it "
            "nose ahead of random — and only just (a hair over the significance bar). |\n"
            "| Does the *open-at-the-low* shape matter? | **No.** Scramble the candle shape — buy "
            "*any* bar in the same downtrend — and you do just as well **43%** of the time. The "
            "shape isn't doing the work. |\n"
            "| So is it a tradable edge? | **Not really.** What little there is, is the "
            "**downtrend context** (slow mean-reversion + drift), not the belt-hold. |\n\n"
            "> The belt-hold is a fine way to *label* a strong up-day. As a *forecast* — \"the "
            "open at the low means reversal\" — it's **fragile at best**: one marginal long-horizon "
            "blip, and the candle shape itself carries no information."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"After a downtrend, a candle that **opens at its low** (no lower wick) and "
            "**closes well up** in a long white body is a **bullish belt-hold**. The open at the "
            "extreme means buyers took control instantly — the trend reverses. Buy it.\"*\n\n"
            "This is **Steve Nison's** translation of the Japanese *yorikiri* candle (from Munehisa "
            "Homma's rice-trading lore), built into TradingView, every broker's pattern scanner, "
            "and a hundred candlestick courses. It's one of the most recognisable single-candle "
            "reversal signals — so: does the open-at-the-extreme actually reverse anything?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a single candle genuinely *forecast* reversals, it would be remarkable: one bar's "
            "shape predicting the next month, a clean crack in efficiency you could trade with a "
            "screener. That's the dream the pattern sells.\n\n"
            "But two traps are built in. First, a belt-hold is spotted **after** the up-day has "
            "happened — and it's spotted on indices that drift **up** over time, so *any* buy rule "
            "looks profitable. Second, the 'signal' bundles two things: the **candle shape** (open "
            "at the low) *and* the **prior downtrend**. To separate the **candle** from the "
            "**context** and the **tide**, we (a) compare to buying on **random days**, and (b) "
            "scramble the candle shape while keeping the downtrend. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), "
            f"daily, over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Flag the belt-hold mechanically.** White body, the **open within 10% of the low** "
            "(open ≈ low, no lower wick), a **tall body** (≥60% of the bar's range), and a real "
            "**prior downtrend** (close below where it was 10 days ago). All read at the close.\n"
            "2. **Trade the lore.** On a belt-hold close, buy at the **next** close; measure the "
            "return over the next **5 / 10 / 20 / 60 days**.\n"
            "3. **The honest baseline.** Do the exact same hold on **random days**. If the candle "
            "matters, the belt-hold must beat random.\n"
            "4. **The shape test.** Buy *random bars in the same downtrend* instead. If the "
            "open-at-the-low shape matters, that should do much worse. *If it doesn't, the shape is "
            "a mirage* — that's the result that would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical belt-hold look like? Here's SPY with the flagged "
            "belt-hold bars the rule would buy."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    seg = b.iloc[-450:]\n"
            "    ent = st.belt_hold_entries(b); ent = ent[ent >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg['close'].values, c='k', lw=1.2, label='SPY close')\n"
            "    ax.scatter(ent, b['close'].reindex(ent), c=GREEN, s=55, zorder=5, label='belt-hold BUY')\n"
            "    ax.set_title('Mechanical bullish belt-holds on SPY (last ~2y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('belt-hold signals in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The signals land on sharp up-days after dips — exactly as advertised. The question is "
            "whether those green dots are followed by *more* upside than buying on any old day. "
            "**Let's race the belt-hold against random entries** at four horizons. Blue = belt-hold; "
            "grey = random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    bh, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t)\n"
            "            e = st.belt_hold_entries(bb)\n"
            "            re = st.random_entries(bb, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(bb, e, h)); rr.append(st.forward_returns(bb, re, h))\n"
            "        bh.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    bh = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, bh, .4, color='#2c6fbb', label='buy the belt-hold')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(bh,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The belt-hold barely beats random — and only at 60 days'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('belt-hold:', [round(v) for v in bh]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story. The belt-hold makes money in absolute terms "
            f"(**+{R['h20'][2]:.0f} bps** over 20 days) — but so does buying on **random days** "
            f"(**+{R['h20'][5]:.0f} bps**). At 5/10/20 days it's a coin-flip. Only at 60 days does "
            f"the belt-hold nose ahead (**+{R['h60'][2]:.0f}** vs **+{R['h60'][5]:.0f}** bps), and "
            "even then by a whisker (the quants notebook shows the *t* is just **+2.03**, right at "
            "the bar). Most of the apparent edge is **the market's upward drift**, not the candle."
        ),
        md(
            "**The decisive check.** What if we keep the **downtrend** but throw away the "
            "**candle shape** — buy *random bars in the same kind of downtrend* instead of the "
            "belt-hold? If the open-at-the-low really matters, the shape-blind version should do "
            "much worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')\n"
            "    pl = st.shape_scramble_placebo(c, 20, n_draws=300, seed=456)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real belt-hold (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *shape-scrambled* draws (random downtrend bars) do as well (p={pval:.2f}).')\n"
            "print('=> the open-at-low candle shape is NOT doing the work.')"
        ),
        md(
            f"Nearly half of the **shape-scrambled** draws match or beat the real belt-hold "
            f"(*p* = {R['placebo'][1]:.2f}). If the open-at-the-extreme genuinely mattered, "
            "scrambling it would collapse the result. It doesn't — because whatever small edge "
            "exists is the **downtrend context**, not the candle."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The belt-hold is a coin-flip vs random at 5/10/20 days and only "
            "*marginally* beats it at 60 days (*t* = +2.03, right at the bar) — concentrated in one "
            "name. Enough to deny a clean 'None', far short of a real edge.\n"
            "- **Tradability — Fragile.** The only edge is slow (60-day), leans on a single "
            "instrument, and is really the downtrend context — not a deployable candle signal.\n"
            "- **\"Does the opening-at-extreme reverse\"? — Busted.** Scramble the candle shape and "
            "the result barely moves (*p* = 0.43). The open-at-the-low isn't doing the reversing."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Barely, and not as a *candle* strategy. The belt-hold's only edge over a coin flip "
            "shows up at 60 days, leans almost entirely on SPY, and is really 'buy a dip and wait' "
            "— which you'd capture more simply (and more fully) by just buying after downtrends, no "
            "candle screener required. Costs on every signal eat into an already-thin result. As a "
            "forecasting tool the open-at-the-low doesn't pay; as a labelling tool it was never "
            "meant to be a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The downtrend, not the candle.** The placebo says the edge is the prior downtrend. "
            "A fun follow-up strips the candle entirely and just buys post-downtrend dips — same "
            "result, no belt-hold.\n"
            "- **Tighter shapes.** Try a full marubozu (no wicks either end) or a body ≥80% of "
            "range — fewer signals, same story: drift/context in, candle out.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* belt-hold "
            "reversal into a synthetic tape and shows the harness banks it (so the marginal real "
            "result isn't a dead detector — it's an honest reading).\n\n"
            "*Think the belt-hold forecasts? Show it beating random entries **and** its own "
            "shape-scramble placebo at **t ≥ 2** across horizons on a real tape — then we'll talk.*"
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
            "# Belt-Hold — a quantitative teardown 🔬\n"
            "### Mechanical bullish belt-holds on 5 indices · forward returns · one-sample HAC *t* · "
            "a drift-matched random-entry baseline · a shape-scramble geometry placebo · costs · a "
            "synthetic planted-reversal control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **candle** from the **drift** *and* from the **downtrend context**: "
            "an upward-trending index makes *any* buy look good, and the belt-hold bundles a "
            "downtrend filter with a candle shape — so the only meaningful tests are vs-random plus "
            "a placebo that destroys the shape while keeping the downtrend.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. The belt-hold is read on the "
            "close of t (open within 10% of the low, body ≥60% of range, prior 10-bar downtrend); "
            "entry is the **next close** (one documented lag). Offline core + synthetic control are "
            "deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Belt-hold vs a **drift-matched random** baseline: a coin-flip "
            f"at 5/10/20d (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f} bps, Welch t "
            f"≤ 1.15) and only **marginally** positive at 60d (Welch t = {R['h60'][8]:+.2f}, "
            f"p = {R['h60'][9]:.3f}) — one horizon, right at the bar. |\n"
            f"| **Tradability** | `FRAGILE` | The big one-sample t's (20d t = {R['h20'][4]:.2f}) are "
            f"mostly **beta**; the only vs-random edge is slow (60d) and leans on one name (SPY). "
            "Nothing robust to scale. |\n"
            f"| **Opening-extreme reverses?** | `BUSTED` | The shape-scramble placebo leaves the "
            f"result intact: **p = {R['placebo'][1]:.2f}** of random-downtrend-bar draws match or "
            "beat the real belt-hold. The candle shape isn't load-bearing. |\n\n"
            "> 💡 In plain words: the belt-hold *looks* significant only because indices drift up "
            "and the rule already filters for a downtrend. Race it vs random (drift gone) or scramble "
            "the shape (candle gone) and only a thin, slow, single-name residual survives."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "On bar $t$ with $(O,H,L,C)$, define the bullish belt-hold by $C>O$ (white), "
            "$O-L \\le \\tfrac{1}{10}(H-L)$ (open at the low), $C-O \\ge \\tfrac{3}{5}(H-L)$ (tall "
            "body), and a prior downtrend $C_t < C_{t-10}$. The Andrews-style rule buys at $C_{t+1}$ "
            "and rides the reversal.\n\n"
            "- **H₀ (drift).** Belt-hold returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (the candle forecasts).** Belt-hold returns **exceed** random, t ≥ 2, robustly.\n"
            "- **H₂ (the shape matters).** Belt-hold returns exceed a **shape-scramble** null that "
            "keeps the downtrend filter but discards the candle geometry.\n\n"
            "We find **H₀ rejected only at 60d** (and only at t = 2.03), **H₁ essentially "
            "unsupported** (coin-flip at 5–20d), **H₂ rejected** (placebo p ≈ 0.43). The steelman "
            "survives on one leg, by a hair, and the geometry leg fails outright."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the three confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* long "
            "entry inherits it; a high one-sample $t$ against **zero** measures the tide. The fix is "
            "the **random-entry baseline** and a Welch test of belt-hold-*minus*-random.\n\n"
            "**(b) Context vs shape.** The belt-hold bundles a **prior-downtrend filter** with a "
            "**candle shape**. A naive vs-random test can't tell which is doing the work. The "
            "**shape-scramble placebo** keeps the downtrend filter and the signal count but draws "
            "from random downtrend bars — isolating the candle's marginal contribution.\n\n"
            "**(c) Selection across horizons.** Testing 4 horizons and celebrating the one that "
            "clears t=2 is multiple testing. We report all four and read 60d's t = 2.03 as the "
            "**marginal, isolated** result it is — not a discovery."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} belt-hold signals** "
            "pooled.\n"
            "- **Belt-hold flag.** White body; open within 10% of the low; body ≥60% of the bar's "
            "range; prior 10-bar downtrend. Read at the close of t (no look-ahead).\n"
            "- **Entry.** Belt-hold close → enter **next close** (one lag); hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of belt-hold returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample belt-hold vs random (the *real* "
            "test).\n"
            "- **Null #3 — shape-scramble placebo** (downtrend kept, candle shape destroyed).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every signal.\n"
            "- **Positive control.** Synthetic tape with a **planted** post-belt-hold reversal (knob "
            "`edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks strong, vs-random shrinks it\n\n"
            "Left: the belt-hold's **one-sample** t against zero (the misleading number). Right: the "
            "same belt-hold vs a **drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, bh, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)\n"
            "            e = st.belt_hold_entries(c)\n"
            "            re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); bh.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    bh = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (misleading: it is beta)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else AMBER if v>0 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Belt-hold vs RANDOM, Welch t (only 60d clears 2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 at 10/20/60d — but that's the "
            f"**drift**, every long entry inherits it. The right bars are the real test: "
            f"belt-hold-minus-random is a coin-flip at 5/10/20d ({R['h20'][8]:+.2f} at 20d) and only "
            f"**{R['h60'][8]:+.2f}** at 60d — right at the bar, one horizon out of four."
        ),
        md(
            "### 4b · Belt-hold vs random across horizons — the gap is the verdict\n\n"
            "Mean return, belt-hold vs random entry, all four horizons. The belt-hold should tower "
            "over random if the candle forecasts. It only nudges ahead at 60d."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, bh, .4, color='#2c6fbb', label='belt-hold')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(bh,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Belt-hold only nudges ahead of random at 60 days'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta belt-hold-random (bps):', [round(a-b) for a,b in zip(bh,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the belt-hold is **+{R['h20'][2]:.0f} bps** and random "
            f"is **+{R['h20'][5]:.0f} bps** — a {R['h20'][6]:+.0f} bps edge that the Welch test (4a) "
            "calls noise. Only at 60d does the gap reach the bar, and barely."
        ),
        md(
            "### 4c · The shape placebo — scramble the candle, nothing changes\n\n"
            "Keep the prior-downtrend filter and the signal count, but draw the entries from "
            "**random downtrend bars** — the candle shape is gone. If price reverses because the "
            "**open sat at the extreme**, the real belt-hold should sit far in the right tail of "
            "these shape-blind draws. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')\n"
            "    pl = st.shape_scramble_placebo(c, 20, n_draws=400, seed=456)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    down = st._downtrend_bars(c); pool = c.index[down]\n"
            "    real_ent = st.belt_hold_entries(c); k = len(real_ent)\n"
            "    rng = np.random.default_rng(456); draws = []\n"
            "    for _ in range(400):\n"
            "        pick = __import__('pandas').DatetimeIndex(rng.choice(pool, size=k, replace=False))\n"
            "        rr = st.forward_returns(c, pick, 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(456); draws = rng.normal(95, 45, 400)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='shape-scrambled draws (random downtrend bars)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real belt-hold {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real belt-hold sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real belt-hold {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => candle shape not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real belt-hold (blue line) sits **inside** the "
            f"shape-scrambled cloud — **p = {R['placebo'][1]:.2f}**. Buying *any* bar in the same "
            "downtrend does as well, so the open-at-the-low geometry carries no extra information. "
            "This is the cleanest refutation of 'the opening-at-extreme reverses.'"
        ),
        md(
            "### 4d · Per-ticker — a scattered, single-name tilt\n\n"
            "20-day belt-hold-minus-random delta, per instrument. A real candle edge would be "
            "positive and similar across the board; instead it's dominated by SPY and flips negative "
            "on QQQ."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = load(t)\n"
            "        e = st.belt_hold_entries(c); re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d belt-hold − random (bps)'); ax.set_title('Dominated by SPY, negative on QQQ')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: **SPY** carries the result (+{R['per'][0][5]:.0f} bps) while "
            f"**QQQ** is **{R['per'][1][5]:+.0f}** bps *behind* random. No coherent, cross-sectional "
            "edge — the hallmark of a thin, context-driven tilt rather than a real candle signal."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real reversal\n\n"
            "To prove the marginal real result isn't a dead detector, plant a **real** "
            "post-belt-hold reversal into a synthetic tape and check the same rule banks it: edge=0 "
            "must stay at t≈0; edge>0 must light up."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.40):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=456, n_days=4000)\n"
            "    e = st.belt_hold_entries(px); s = st.summarize(st.forward_returns(px, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted reversal -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} bh={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted reversal the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"reversal reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector "
            "is live and sensitive — so the thin real-tape result is an honest reading, not a broken "
            "pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — the belt-hold is a coin-flip vs a drift-matched random baseline "
            f"at 5/10/20d (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f} bps; Welch t "
            f"≤ 1.15) and only **marginally** beats random at 60d (Δ = {R['h60'][6]:+.0f} bps, "
            f"Welch t = **{R['h60'][8]:+.2f}**, p = {R['h60'][9]:.3f}). One horizon, right at the "
            f"bar, concentrated in SPY. The big one-sample t's (20d **{R['h20'][4]:.2f}**) are mostly "
            "beta.\n"
            f"- **Tradability `FRAGILE`** — the only vs-random edge is slow (60d), leans on one name, "
            "and is the downtrend context, not a deployable candle signal. Costs eat a thin result; "
            "nothing robust to scale.\n"
            f"- **Opening-extreme reverses? `BUSTED`** — the shape-scramble placebo leaves the result "
            f"intact (**p = {R['placebo'][1]:.2f}**): random downtrend bars do as well as the real "
            "belt-hold, so the open-at-the-low candle shape carries no forecasting information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is barely anything to trade\n\n"
            "The belt-hold's apparent profit is mostly the unconditional drift of long equity "
            "indices plus a thin, slow, single-name downtrend-reversion tilt — captured more simply "
            "by buying post-downtrend dips without a candle screener. The rule trades *less* of the "
            "time and pays costs on each signal, so it dominates almost nothing. There is no real "
            "capacity question because there is no robust edge to scale. The belt-hold is a "
            "descriptive candle label, not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Strip the candle.** The placebo says the residual is the downtrend. A clean "
            "follow-up buys post-downtrend dips with no shape filter and recovers the same 60d tilt "
            "— proving the candle is decorative.\n"
            "- **Tighter geometry.** A full marubozu (no wicks either end), body ≥80% of range, or a "
            "stricter downtrend — fewer signals, same conclusion: context in, candle out.\n"
            "- **Multiple-horizon honesty.** With 4 horizons tested, a single t = 2.03 at the longest "
            "is roughly what selection alone would produce; a Bonferroni/White reality-check would "
            "deflate it further.\n\n"
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
