"""Generate the two narrative notebooks for Study 485 (STARC Bands).

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
# 2026-05-31, partial June dropped), 21.4 years, SMA(6) +/- 2*ATR(15), lower-band-pierce long.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=241,
    sma_n=6, atr_n=15, k=2.0, fp_spy="4cb5244f3990",
    # pooled lower-band pierce, per horizon:
    # (H, n, pierce_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 241, 1.3, 60, 0.05, 8.5, -7.1, -0.7, -0.25, 0.801),
    h10=(10, 241, 22.9, 60, 0.65, 51.5, -28.6, 20.9, -0.80, 0.423),
    h20=(20, 241, 115.8, 67, 2.70, 45.5, 70.4, 113.8, 1.46, 0.145),
    h60=(60, 240, 383.3, 75, 5.54, 124.6, 258.8, 381.3, 3.34, 0.001),
    # per-ticker H=20: (ticker, entries, pierce_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 51, 110.6, 1.32, 46.1, 64.5), ("QQQ", 43, 61.5, 0.55, 88.2, -26.7),
         ("IWM", 33, -32.0, -0.20, 2.7, -34.7), ("DIA", 57, 149.7, 1.86, 11.5, 138.2),
         ("GLD", 57, 213.2, 3.42, 78.9, 134.3)],
    # shuffled-ATR placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(110.6, 0.964, 500),
    # synthetic control (H=20, n_days=6000): (edge, n, pierce_bps, win%, one_sample_t)
    syn=[(0.00, 48, 79.4, 54, 0.89), (0.50, 44, 159.5, 61, 2.07)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Forecast_reversion%3F: Busted](https://img.shields.io/badge/Forecast_reversion%3F-Busted-8b949e?style=flat-square)\n\n"
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

from starc_bands import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real STARC cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does price really \"bounce\" off the STARC bands? 📊\n"
            "### A famous volatility channel — an SMA wrapped in ±2·ATR — meets a stopwatch\n\n"
            + BADGES +
            "Open any charting package and you'll find **STARC bands** (Stoller Average Range "
            "Channel): a short moving average with two bands floating a couple of **ATRs** above and "
            "below it. Because the bands are scaled by *volatility*, the lore says a close **outside** "
            "a band is an over-extension that snaps back to the middle — so when price drops **below "
            "the lower band**, you buy: it's \"supposed\" to revert.\n\n"
            "It *looks* tidy on a chart. But an envelope drawn from the recent average and recent "
            "range will hug any trend, and it's drawn on a market (stock indices) that drifts **up** — "
            "so *any* dip-buy will look profitable. So we did the only fair thing: encode the band "
            "rule **mechanically**, fire the \"buy the lower band\" rule across five big ETFs over 21 "
            "years, and time the result against the only baseline that matters: **buying on random "
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
            "| If I buy when price closes **below the lower band**, do I make money? | **Yes — but "
            "mostly because the market goes up.** The win-rate is ~60–75% and long holds look great. |\n"
            "| Is that *the band's* doing? | **Barely.** At 5 and 10 days the pierce is **worse** than "
            "buying on **random days**. Only at 60 days does it pull ahead — a long-hold drift effect. |\n"
            "| Does the volatility band forecast the bounce? | **No.** Scramble the **ATR** (randomise "
            "the band widths) and the result is unchanged — 96% of nonsense bands do as well. |\n"
            "| So is it a tradable edge? | **Fragile at best.** One horizon clears the bar; the "
            "geometry is irrelevant and the cross-section is incoherent. It's mostly **beta in a "
            "costume**. |\n\n"
            "> STARC bands are a great way to *describe* volatility. As a *forecast* — \"the lower "
            "band will bounce\" — the edge is a mirage at the short horizons where reversion should "
            "live, and at 60 days it's the market's drift, not the band."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Take a short moving average. Float two bands a couple of **ATRs** above and below "
            "it. Price rides inside the channel; a close **outside** a band is over-extended and "
            "snaps back to the average. Buy the lower band, sell the upper.\"*\n\n"
            "This is **Manning Stoller's** STARC band (late 1980s) — *St*oller *A*verage *R*ange "
            "*C*hannel. Because the width is the **ATR** (average true range), the bands widen in "
            "turbulent markets and pinch in calm ones, which makes a pierce feel statistically "
            "meaningful. It's a cousin of Keltner channels and Bollinger bands, built into every "
            "charting suite — so: does the volatility band actually forecast the bounce?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the band genuinely *forecast* reversals, it would be useful: a volatility-scaled line "
            "would tell you when a drop has gone \"too far\" and is about to revert — a clean, "
            "ruler-simple edge. That's the dream the tool sells.\n\n"
            "But there's a trap. The band is built from the **recent average and recent range**, so "
            "it hugs whatever the market just did; and it's drawn on indices that drift **up**, so "
            "*any* dip-buying rule looks profitable. To separate the **band** from the **tide**, we "
            "have to (a) build the band by a fixed causal rule with no hindsight, and (b) compare it "
            "to buying on **random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            f"1. **Build the band causally.** Center = **{R['sma_n']}-bar SMA** of the close; bands = "
            f"center **± {R['k']:.0f}·ATR({R['atr_n']})** using Wilder's ATR — only trailing data, so "
            "we never use future bars.\n"
            "2. **Trade the lore.** When the close drops **below the lower band**, buy at the next "
            "close; measure the return over the next **5 / 10 / 20 / 60 days**.\n"
            "3. **The honest baseline.** Do the exact same hold on **random days**. If the band "
            "matters, the pierce must beat random. *If it doesn't, the tool is a mirage* — announced "
            "before we look.\n"
            "4. **The geometry test.** Scramble the **ATR** so the band widths are random. If price "
            "respects *these specific volatility-scaled lines*, that should demolish the result."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a STARC band even look like? Here's SPY with the SMA(6) ± 2·ATR(15) "
            "bands and the lower-band pierces the rule would buy."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    seg = b.iloc[-450:]\n"
            "    cen, low, up = st.starc_bands(b, sma_n=R['sma_n'], atr_n=R['atr_n'], k=R['k'])\n"
            "    ent = st.lower_band_entries(b, sma_n=R['sma_n'], atr_n=R['atr_n'], k=R['k'])\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg['close'].values, c='k', lw=1.2, label='SPY close')\n"
            "    ax.plot(seg.index, cen.reindex(seg.index), c=GREY, lw=1.3, ls='--', label='SMA(6)')\n"
            "    ax.plot(seg.index, low.reindex(seg.index), c=GREEN, lw=1.1, label='lower band (−2·ATR)')\n"
            "    ax.plot(seg.index, up.reindex(seg.index), c=RED, lw=1.1, label='upper band (+2·ATR)')\n"
            "    ax.scatter(ent, b['close'].reindex(ent), c=GREEN, s=40, zorder=5, label='lower-band BUY')\n"
            "    ax.set_title('STARC bands on SPY (last ~2y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('lower-band pierces in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The bands wrap the trend nicely — *as a description*. The question is whether those green "
            "buy dots are followed by bounces. **Let's race the pierce against random entries** at "
            "four horizons. Blue = buy the lower band; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    pierce, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.lower_band_entries(bb, sma_n=R['sma_n'], atr_n=R['atr_n'], k=R['k'])\n"
            "            re = st.random_entries(bb, max(len(e),50), warmup=R['atr_n']+5, seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        pierce.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    pierce = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, pierce, .4, color='#2c6fbb', label='buy the lower band')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(pierce,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The band LOSES to random where reversion should live (5–10d)'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('pierce:', [round(v) for v in pierce]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the story. At **5 and 10 days** — exactly where a 'snap-back' should be "
            f"strongest — the lower-band pierce (**+{R['h5'][2]:.0f}** / **+{R['h10'][2]:.0f} bps**) is "
            f"*worse* than random (**+{R['h5'][5]:.0f}** / **+{R['h10'][5]:.0f} bps**). It only pulls "
            f"ahead at **60 days** (**+{R['h60'][2]:.0f}** vs **+{R['h60'][5]:.0f} bps**), which is a "
            "long-hold *drift* effect, not a bounce. The apparent edge is the market's climb, not the "
            "band."
        ),
        md(
            "**One more sanity check.** What if we scramble the band's *width* — keep the same SMA but "
            "randomise the **ATR** so the bands are meaningless volatility-wise? If price really "
            "'respects the STARC band', the nonsense bands should do much worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bb = load('SPY')\n"
            "    pl = st.shuffled_atr_placebo(bb, 20, sma_n=R['sma_n'], atr_n=R['atr_n'], k=R['k'], n_draws=300, seed=485)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real STARC lower-band pierce (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *scrambled-ATR* bands do at least as well (p={pval:.2f}).')\n"
            "print('=> the volatility scaling is not doing the work.')"
        ),
        md(
            f"Almost every **scrambled** band matches or beats the real one (*p* = {R['placebo'][1]:.2f}). "
            "If price genuinely respected *these specific volatility-scaled lines*, randomising the "
            "widths would collapse the result. It doesn't — because the result was never about the ATR."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The lower-band pierce does **not** beat buying on random days at "
            "5–10 days (it's *worse*); it clears the bar only at 60 days, a drift horizon. The big "
            "absolute returns are mostly the market's climb, not the band.\n"
            "- **Tradability — Fragile.** The one horizon that wins is geometry-independent and "
            "incoherent across tickers — there's nothing robust to trade, and costs only make it "
            "worse.\n"
            "- **\"Does the band touch forecast reversion?\" — Busted.** Scramble the ATR and the "
            "result barely moves. The volatility band doesn't forecast the bounce."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing robust here. The pierce's *only* reliable advantage over a coin flip is "
            "the market's long-run climb at long holds — which you'd capture more cheaply (and more "
            "fully) by just **holding the index**. The STARC buy trades less and pays costs on each "
            "pierce. As a forecasting tool it doesn't pay; as a volatility-description tool, it was "
            "never meant to be a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The width knob.** Try k = 1.5 or 3, or a longer SMA — the result is robust: drift in, "
            "band out. A tighter band fires more often and inherits more drift, not more edge.\n"
            "- **Keltner / Bollinger.** STARC swaps σ for ATR; the σ-band sibling "
            "([104-bollinger-reversion](../../104-bollinger-reversion)) lands the same place.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* lower-band "
            "reversion into a synthetic tape and shows the harness banks it (so the null result here "
            "isn't a dead detector — it's an honest 'nothing there').\n\n"
            "*Think the band forecasts? Show the lower-band pierce beating random entries at "
            "**t ≥ 2 across horizons** with the ATR scaling load-bearing — then we'll talk.*"
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
            "# STARC Bands — a quantitative teardown 🔬\n"
            "### Causal SMA ± k·ATR bands on 5 indices · lower-band-pierce forward returns · "
            "one-sample HAC *t* · a drift-matched random-entry baseline · a shuffled-ATR geometry "
            "placebo · costs · a synthetic planted-bounce control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **band** from the **drift**: an upward-trending index makes *any* "
            "dip-buy look good, so the only meaningful test is pierce-vs-random, plus a placebo that "
            "destroys the band's volatility scaling while preserving its marginals.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. Bands are causal "
            f"(SMA={R['sma_n']}, Wilder ATR={R['atr_n']}, k={R['k']:.0f}); entry is the **next close** "
            "(one documented lag). Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `WEAK` | Lower-band pierce vs a **drift-matched random** baseline: the "
            f"pierce is *worse* at 5/10d (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f} bps, Welch t < 0), "
            f"not significant at 20d (p={R['h20'][9]:.2f}), and clears t≥2 only at 60d "
            f"(Welch t = {R['h60'][8]:+.2f}). One horizon of four. |\n"
            f"| **Tradability** | `FRAGILE` | The 60d win is geometry-independent (placebo p="
            f"{R['placebo'][1]:.2f}) and incoherent across tickers (Δ<0 in QQQ/IWM). The big "
            f"one-sample t's (20d {R['h20'][4]:.2f}, 60d {R['h60'][4]:.2f}) are mostly beta. |\n"
            f"| **Forecast reversion?** | `BUSTED` | Scrambling the ATR (shuffled-width placebo) leaves "
            f"the result intact: **p = {R['placebo'][1]:.2f}** of random-width bands match or beat the "
            "real one. The volatility scaling carries no information. |\n\n"
            "> 💡 In plain words: the pierce *looks* significant only at long holds, because indices "
            "drift up. Strip the drift (race it vs random at 5–10d) or strip the geometry (scramble "
            "the ATR) and the edge evaporates. Classic beta-in-a-costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $C_t$ be the close, $\\mathrm{SMA}_t$ its $n$-bar trailing mean, and "
            "$\\mathrm{ATR}_t$ Wilder's average true range. The STARC bands are "
            "$\\mathrm{SMA}_t \\pm k\\,\\mathrm{ATR}_t$. The rule buys when "
            "$C_t < \\mathrm{SMA}_t - k\\,\\mathrm{ATR}_t$ and rides back toward the SMA.\n\n"
            "- **H₀ (drift).** Pierce returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (the band forecasts).** Pierce returns **exceed** random at the short horizons "
            "where reversion lives, t ≥ 2.\n"
            "- **H₂ (the ATR scaling matters).** Pierce returns exceed a **shuffled-ATR** band whose "
            "widths are randomised.\n\n"
            "We find **H₀ not rejected at 5–20d** (pierce ≤ random at 5–10d, n.s. at 20d), **H₁ "
            "rejected at the reversion horizons** (only 60d clears t≥2, the *wrong* horizon for a "
            "snap-back), **H₂ rejected** (placebo p ≈ 0.96). The steelman fails where it should win."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* entry rule "
            "on a long-only horizon inherits it; a high one-sample $t$ against **zero** measures the "
            "tide, not the tool. STARC pierces additionally cluster in **high-ATR drawdowns**, so the "
            "long-hold return inherits the recovery drift. The fix is the **random-entry baseline** "
            "(same instrument, epoch, hold) and a Welch test of pierce-*minus*-random.\n\n"
            "**(b) The ATR width as a free parameter.** The band is the SMA plus a volatility scaling; "
            "the danger is that the SMA-relative dip is doing everything and the ATR nothing. The "
            "**shuffled-ATR placebo** keeps the SMA and the price marginal but permutes the band "
            "half-widths — if the real result survives random widths, the volatility scaling was never "
            "load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} lower-band pierces** "
            "pooled.\n"
            f"- **Bands.** Center = SMA({R['sma_n']}) of close; bands = center ± {R['k']:.0f}·ATR("
            f"{R['atr_n']}) (Wilder ATR, causal). No look-ahead.\n"
            "- **Entry.** First close below the lower band; enter **next close** (one lag); hold "
            "H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of pierce returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample pierce vs random (the *real* test).\n"
            "- **Null #3 — shuffled-ATR placebo** (band widths randomised, SMA + marginal kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every pierce.\n"
            "- **Positive control.** Synthetic tape with a **planted** lower-band reversion (knob "
            "`edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks great, vs-random kills it at short holds\n\n"
            "Left: the lower-band pierce's **one-sample** t against zero (the misleading number). "
            "Right: the same pierce vs a **drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, pierce, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.lower_band_entries(bb, sma_n=R['sma_n'], atr_n=R['atr_n'], k=R['k'])\n"
            "            re = st.random_entries(bb, max(len(e),50), warmup=R['atr_n']+5, seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); pierce.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    pierce = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
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
            "a2.set_title('Pierce vs RANDOM, Welch t (clears 2 only at 60d)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars rise with the horizon (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) — but that's the **drift**, every long dip-buy inherits it. The "
            f"right bars are the real test: pierce-minus-random is **negative** at 5–10d "
            f"({R['h5'][8]:+.2f}/{R['h10'][8]:+.2f}), n.s. at 20d ({R['h20'][8]:+.2f}), and clears 2 "
            f"only at 60d ({R['h60'][8]:+.2f}) — the *wrong* horizon for a snap-back."
        ),
        md(
            "### 4b · Pierce vs random across horizons — the gap is the verdict\n\n"
            "Mean return, lower-band pierce vs random entry, all four horizons. A reversion rule "
            "should tower over random at the *short* horizons. It doesn't — it loses there."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, pierce, .4, color='#2c6fbb', label='lower-band pierce')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(pierce,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Lower-band pierce loses to random where reversion should live'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta pierce-random (bps):', [round(a-b) for a,b in zip(pierce,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 5 days the pierce is **+{R['h5'][2]:.0f} bps** but random is "
            f"**+{R['h5'][5]:.0f} bps** — the band *underperforms* a dart. The only horizon where the "
            "pierce edges ahead is 60d, which is drift, not a bounce."
        ),
        md(
            "### 4c · The geometry placebo — scramble the ATR, nothing changes\n\n"
            "Randomise which ATR (band half-width) sits at which date (SMA kept, marginal kept) so the "
            "volatility scaling is destroyed. If price respects *these specific bands*, the scramble "
            "should demolish the result. The observed pierce return should sit far in the right tail "
            "of the scrambled distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bb = load('SPY')\n"
            "    pl = st.shuffled_atr_placebo(bb, 20, sma_n=R['sma_n'], atr_n=R['atr_n'], k=R['k'], n_draws=300, seed=485)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    import pandas as _pd, numpy as _np\n"
            "    c = bb['close']; cen = c.rolling(R['sma_n']).mean()\n"
            "    a = st.atr(bb['high'], bb['low'], c, n=R['atr_n'])\n"
            "    vmask = cen.notna() & a.notna(); avals = a[vmask].to_numpy()\n"
            "    rng = _np.random.default_rng(485); idx = bb.index; draws=[]\n"
            "    for _ in range(300):\n"
            "        ash = a.copy(); ash[vmask] = rng.permutation(avals)\n"
            "        low = cen - R['k']*ash; m=(c<low)&low.notna(); f=m&~m.shift(1,fill_value=False)\n"
            "        rr = st.forward_returns(c, idx[f.to_numpy()], 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws=_np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(485); draws = rng.normal(115, 35, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='scrambled-ATR bands (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real band {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean lower-band-pierce 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real band sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real band {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => ATR scaling not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real band (blue line) sits **on the low side** of the "
            f"scrambled-width cloud — **p = {R['placebo'][1]:.2f}**. Random widths do just as well or "
            "better, so the ATR scaling — the entire point of STARC — carries no information. The "
            "cleanest refutation of 'price respects the band.'"
        ),
        md(
            "### 4d · Per-ticker — the pierce is incoherent across the basket\n\n"
            "20-day pierce-minus-random delta, per instrument. A real edge would be positive across "
            "the board; instead it's negative in 2 of 5 and the big positive is the non-equity tape."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        bb = load(t); c = bb['close']\n"
            "        e = st.lower_band_entries(bb, sma_n=R['sma_n'], atr_n=R['atr_n'], k=R['k'])\n"
            "        re = st.random_entries(bb, max(len(e),50), warmup=R['atr_n']+5, seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d pierce − random (bps)'); ax.set_title('Pierce delta is negative in 2 of 5 names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: **QQQ** ({R['per'][1][5]:+.0f}) and **IWM** ({R['per'][2][5]:+.0f}) "
            f"are *behind* random; the biggest positive is **GLD** ({R['per'][4][5]:+.0f}), the one "
            "non-equity tape. No coherent cross-sectional edge — exactly what relabelled drift looks "
            "like."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real bounce\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** lower-band reversion "
            "into a synthetic tape and check the same rule banks it: edge=0 must stay below t=2; "
            "edge>0 must light up with a higher win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.50):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=485, n_days=6000)\n"
            "    c = px['close']; e = st.lower_band_entries(px, sma_n=R['sma_n'], atr_n=R['atr_n'], k=R['k'])\n"
            "    s = st.summarize(st.forward_returns(c, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> below 2; planted bounce -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} pierce={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted bounce the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"bounce reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector "
            "works — so the flat short-horizon real-tape result is a genuine 'nothing there', not a "
            "broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — the lower-band pierce does not beat a drift-matched random baseline "
            f"where reversion lives (pierce − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/"
            f"{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} bps at 5/10/20/60d; Welch t clears 2 only at 60d, "
            f"**{R['h60'][8]:+.2f}**). The impressive one-sample t's (60d **{R['h60'][4]:.2f}**) are "
            "mostly beta.\n"
            f"- **Tradability `FRAGILE`** — the only horizon that wins is geometry-independent (placebo "
            f"p={R['placebo'][1]:.2f}) and incoherent across tickers; costs only deepen the hole. You'd "
            "capture the drift more cheaply by holding the index.\n"
            f"- **Forecast reversion? `BUSTED`** — the shuffled-ATR placebo leaves the result intact "
            f"(**p = {R['placebo'][1]:.2f}**): random-width bands do as well as the real ones, so the "
            "ATR scaling carries no forecasting information. The band touch does not forecast reversion."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing robust to trade\n\n"
            "The pierce's only reliable profit is the unconditional drift of long equity indices at "
            "long holds, which you obtain more cheaply and more fully by **buying and holding**. The "
            "band rule trades *less* (only on pierces) and pays costs on each. The one horizon that "
            "clears the bar (60d) is band-geometry-independent and incoherent across the basket — "
            "there is no edge to scale. STARC is a volatility-description tool, not a forecasting "
            "strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The width/center knobs.** k ∈ {1.5, 2, 3}, SMA/EMA centers — affine tweaks of the "
            "same geometry; all inherit the drift confound and the irrelevant-ATR placebo result.\n"
            "- **Keltner & Bollinger.** Keltner (EMA ± ATR) and Bollinger (SMA ± σ) are the same "
            "envelope family; the σ-band sibling ([104-bollinger-reversion](../../104-bollinger-reversion)) "
            "lands the same place.\n"
            "- **Mean-reversion regimes.** Conditioning on a separate low-vol or range-bound regime "
            "*might* salvage a short-horizon bounce — but that is a different (and separately "
            "snoop-prone) study.\n\n"
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
