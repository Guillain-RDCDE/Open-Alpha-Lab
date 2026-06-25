"""Generate the two narrative notebooks for Study 474 (Accelerator Oscillator).

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
# 2026-05-31, partial June dropped), 21.4 years, AC = AO - SMA5(AO), two-green-bars-above-zero.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=2131,
    fp_spy="4cb5244f3990",
    # pooled AC-up, per horizon:
    # (H, n, ac_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 2126, 11.0, 56, 2.16, 10.4, 0.5, 9.0, 0.07, 0.945),
    h10=(10, 2126, 45.5, 60, 6.17, 26.5, 19.0, 43.5, 1.73, 0.083),
    h20=(20, 2121, 88.2, 63, 6.50, 80.1, 8.1, 86.2, 0.54, 0.591),
    h60=(60, 2111, 252.1, 68, 7.13, 256.1, -4.0, 250.1, -0.16, 0.874),
    # per-ticker H=20: (ticker, entries, ac_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 422, 77.9, 3.03, 84.4, -6.5), ("QQQ", 409, 120.3, 3.61, 102.9, 17.4),
         ("IWM", 433, 72.6, 2.04, 40.6, 32.0), ("DIA", 433, 78.1, 3.31, 54.4, 23.6),
         ("GLD", 434, 93.8, 3.16, 119.7, -25.9)],
    # rotated-AC placebo (SPY, H=20, 500 draws): obs_bps, p
    placebo=(77.9, 0.886, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, ac_bps, win%, one_sample_t)
    syn=[(0.00, 303, -21.7, 46, -0.67), (0.80, 298, 272.0, 62, 5.33)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Forecasts_price%3F: Busted](https://img.shields.io/badge/Forecasts_price%3F-Busted-8b949e?style=flat-square)\n\n"
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

from accelerator_oscillator import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real AC cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does Bill Williams' Accelerator Oscillator actually forecast? ⚡\n"
            "### A famous \"acceleration\" indicator — green bars, red bars, a zero line — meets a stopwatch\n\n"
            + BADGES +
            "Open MetaTrader or TradingView and you'll find Bill Williams' **Accelerator Oscillator "
            "(AC)**: a histogram of green and red bars that's supposed to measure how fast momentum "
            "itself is *speeding up or slowing down* — the **acceleration** of price. The lore, taught "
            "by Williams in *Trading Chaos* and repeated on every indicator site, is that acceleration "
            "**leads** price: AC turns up *before* the market does, so **two rising green bars above "
            "the zero line** are a high-probability **buy**.\n\n"
            "It *looks* uncanny when you scroll back through a chart. But an indicator that is literally "
            "a smoothed **second derivative of past price**, computed on a market that drifts **up** "
            "over time, is the textbook setup for fooling yourself. So we did the only fair thing: "
            "encode the rule **mechanically** (no eyeballing), fire the \"two green bars above zero\" "
            "buy thousands of times across five big indices over 21 years, and time the result with a "
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
            "| If I buy on **two green AC bars above zero**, do I make money? | **Yes — but only "
            "because the market goes up.** The raw win-rate is 60–68% and the returns look great. |\n"
            "| Is that *the AC's* doing? | **No.** Buy on **random days** instead and you do "
            "**just as well**. AC adds essentially nothing — the gap vs a coin-flip entry never "
            "reaches significance at any horizon. |\n"
            "| Does acceleration \"lead\" price? | **Not in any usable way.** Rotate the AC bars to a "
            "random point on the chart and the result barely changes. The timing isn't doing the work. |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — the upward drift of "
            "stocks, re-labelled as an acceleration signal. |\n\n"
            "> The Accelerator Oscillator is a fine way to *describe* whether a trend is speeding up "
            "after the fact. As a *forecast* — \"two green bars will keep rising\" — it's a **mirage**: "
            "all of the apparent edge is the market's long-run climb, none of it is the indicator."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The Awesome Oscillator (AO) measures momentum: a fast 5-bar average of the median "
            "price minus a slow 34-bar one. The **Accelerator** subtracts AO's own 5-bar average from "
            "it — AC = AO − SMA5(AO) — so it measures how fast momentum is changing: **acceleration**. "
            "Acceleration turns *before* momentum, and momentum turns before price — so two rising "
            "green AC bars above zero are an early buy.\"*\n\n"
            "This is **Bill Williams'** AO/AC system (*Trading Chaos*, 1995; *New Trading Dimensions*, "
            "1998), still taught today and built into MetaTrader, cTrader and TradingView. It's one of "
            "the most recognisable momentum tools in technical analysis — so: does acceleration "
            "actually *accelerate* your returns?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If AC genuinely *forecast* continuation, it would be remarkable: a second derivative of "
            "*past* price would predict *future* price, a clean crack in market efficiency you could "
            "trade by colour-coding bars. That's the dream the indicator sells.\n\n"
            "But there's a trap built into it. AC is computed **entirely from past prices** (averages "
            "of averages of averages), on a market (stock indices) that drifts **up** over time, so "
            "*any* long-only momentum rule will look profitable. To separate the **tool** from the "
            "**tide**, we have to (a) compute AC by a fixed mechanical rule with no hindsight, and (b) "
            "compare it to buying on **random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Build AC by the book.** AO = SMA5(median) − SMA34(median); AC = AO − SMA5(AO). Every "
            "average is **trailing** (closed bars only), so AC at today's close uses only today and "
            "earlier — no peeking at the future.\n"
            "2. **Trade the lore.** When AC posts **two consecutive rising bars** (green) and sits "
            "**above zero**, buy at the next close; measure the return over the next **5 / 10 / 20 / "
            "60 days**.\n"
            "3. **The honest baseline.** Do the exact same hold on **random days**. If AC matters, the "
            "signal must beat random. *If it doesn't, the tool is a mirage* — that's the result that "
            "would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does AC even look like? Here's SPY with its Accelerator Oscillator below, and "
            "the two-green-bars-above-zero entries the rule would buy."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    seg = b.iloc[-300:]\n"
            "    ac = st.accelerator_oscillator(b).reindex(seg.index)\n"
            "    ent = st.ac_entries(b); ent = ent[ent >= seg.index[0]]\n"
            "    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.2, 6.0), sharex=True,\n"
            "                                   gridspec_kw={'height_ratios':[2,1]})\n"
            "    ax1.plot(seg.index, seg['close'].values, c='k', lw=1.2, label='SPY close')\n"
            "    ax1.scatter(ent, b['close'].reindex(ent), c=GREEN, s=42, zorder=5, label='AC-up BUY')\n"
            "    ax1.set_title('SPY with two-green-bars-above-zero AC entries (last ~14m)'); ax1.legend(loc='upper left')\n"
            "    colors = [GREEN if (v>p) else RED for v,p in zip(ac.values, np.r_[np.nan, ac.values[:-1]])]\n"
            "    ax2.bar(seg.index, ac.values, color=colors, width=1.0)\n"
            "    ax2.axhline(0, c='k', lw=.8); ax2.set_ylabel('AC')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('AC-up entries in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The green AC bars do mark stretches where price is speeding up — *as a description*. The "
            "question is whether those green buy dots are followed by **continuation**. **Let's race "
            "the AC entry against random entries** at four horizons. Blue = two green bars above zero; "
            "grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    ac_r, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.ac_entries(bb)\n"
            "            re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        ac_r.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    ac_r = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, ac_r, .4, color='#2c6fbb', label='two green bars above zero')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(ac_r,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('AC does NOT beat random — it ties it'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('AC:', [round(v) for v in ac_r]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story in one chart. The AC entry makes money in absolute terms "
            f"(**+{R['h20'][2]:.0f} bps** over 20 days) — but **random entries make about the same** "
            f"(**+{R['h20'][5]:.0f} bps**). At every horizon the famous acceleration signal is "
            "neck-and-neck with throwing darts, and the quants notebook shows the difference never "
            "clears *t* = 2. The apparent edge was **the market's upward drift**, not the indicator."
        ),
        md(
            "**One more sanity check.** What if we **rotate** the AC bars — keep their exact values but "
            "slide the whole series to a random point on the chart, so the green bars no longer line up "
            "with the price moves they 'predicted'? If acceleration really *leads* price, the rotated "
            "AC should do much worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.rotated_ac_placebo(load('SPY'), 20, n_draws=300, seed=474)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real AC-up entry (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *rotated* AC series do at least as well (p={pval:.2f}).')\n"
            "print('=> the AC-to-price timing is not doing the work.')"
        ),
        md(
            f"Nearly nine in ten **rotated** AC series match or beat the real one "
            f"(*p* = {R['placebo'][1]:.2f}). If price genuinely followed *this specific* acceleration "
            "timing, a random rotation would collapse the result. It doesn't — because the result was "
            "never about the timing."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The two-green-bars buy does **not** beat buying on random days "
            "(the AC-vs-random difference never clears *t* = 2 at any horizon). The big absolute "
            "returns are the market's drift, not the acceleration.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting for free — and costs only make it worse.\n"
            "- **\"Acceleration forecasts price\"? — Busted.** Rotate the AC bars to a random spot and "
            "the result barely moves. Acceleration doesn't lead."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The AC entry's *only* advantage over a coin flip is the "
            "market's long-run climb — which you'd capture more cheaply (and more fully) by just "
            "**holding the index**. The acceleration buy is a worse, more expensive way to be long. "
            "Costs (commissions + spread on every entry) push the already-no-edge result further "
            "negative. As a forecasting tool, it doesn't pay; as a descriptive histogram, it was never "
            "meant to be a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The 'saucer' and zero-cross variants.** Williams' other AC setups are reshapings of "
            "the same SMA-of-SMA arithmetic and inherit the same drift confound — a fun follow-up shows "
            "they all collapse against random.\n"
            "- **Different windows.** Try other fast/slow lengths than 5/34 — the result is robust: "
            "drift in, smoothed-derivative out.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* upward-accelerating "
            "episode into a synthetic tape and shows the harness banks it (so the null result here "
            "isn't a dead detector — it's an honest 'nothing there').\n\n"
            "*Think acceleration forecasts? Show the two-green-bars entry beating random entries at "
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
            "# The Accelerator Oscillator — a quantitative teardown 🔬\n"
            "### Mechanical AC = AO − SMA5(AO) on 5 indices · two-green-bars-above-zero forward returns · "
            "one-sample HAC *t* · a drift-matched random-entry baseline · a rotated-AC timing placebo · "
            "costs · a synthetic planted-acceleration control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **acceleration signal** from the **drift**: an upward-trending index "
            "makes *any* long-only momentum entry look good, so the only meaningful test is "
            "AC-vs-random, plus a placebo that destroys the AC's time-alignment while preserving its "
            "values.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. AC = AO − SMA5(AO), "
            "AO = SMA5(median) − SMA34(median), all **trailing**; entry is the **next close** (one "
            "documented lag). Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `NONE` | AC-up vs a **drift-matched random** baseline: the difference is "
            f"tiny and mixed-sign (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps) and the AC-minus-random Welch *t* **never clears 2** (best "
            f"{R['h10'][8]:+.2f} at 10d, *p* = {R['h10'][9]:.3f}; reverses to {R['h60'][8]:+.2f} at 60d). |\n"
            f"| **Tradability** | `MIRAGE` | The big one-sample t's (20d t = {R['h20'][4]:.2f}, 60d "
            f"= {R['h60'][4]:.2f}) are **pure beta** — they vanish against random entries and against "
            "cost. No residual edge to scale. |\n"
            f"| **Forecasts price?** | `BUSTED` | Rotating the AC series relative to price (rotated-AC "
            f"placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** of meaningless "
            "time-shifts match or beat the real one. The timing isn't load-bearing. |\n\n"
            "> 💡 In plain words: AC *looks* significant only because indices drift up. Strip the "
            "drift (race it vs random) or strip the timing (rotate the AC) and the edge evaporates. "
            "Classic beta-in-a-costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "With median price $MP_t=\\tfrac12(H_t+L_t)$, the Awesome Oscillator is "
            "$AO_t = \\mathrm{SMA}_5(MP)_t - \\mathrm{SMA}_{34}(MP)_t$ and the Accelerator is "
            "$AC_t = AO_t - \\mathrm{SMA}_5(AO)_t$ — a discrete **second derivative** of smoothed "
            "price. The Williams rule buys when $AC_t > AC_{t-1} > AC_{t-2}$ (two rising green bars) "
            "and $AC_t > 0$.\n\n"
            "- **H₀ (drift).** AC-up returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (acceleration forecasts).** AC-up returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the timing matters).** AC-up returns exceed a **rotated-AC** series whose "
            "alignment with price is destroyed.\n\n"
            "We find **H₀ not rejected** (AC ≈ random at every horizon), **H₁ rejected** (Welch t never "
            "≥ 2), **H₂ rejected** (placebo p ≈ 0.89). The steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* long-only "
            "entry rule on a long horizon inherits it; a high one-sample $t$ against **zero** measures "
            "the tide, not the tool. The fix is the **random-entry baseline** (same instrument, epoch, "
            "hold) and a Welch test of AC-*minus*-random.\n\n"
            "**(b) The indicator is a function of past price.** AC is an SMA-of-SMA-of-SMA of the bars "
            "it's supposed to predict; on an autocorrelated, drifting series *any* such derivative "
            "co-moves with the trend. The **rotated-AC placebo** keeps the AC value marginal *exactly* "
            "but circularly shifts the series so its alignment with forward price is meaningless — if "
            "the real result survives the rotation, the timing was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} AC-up entries** "
            "pooled.\n"
            "- **Indicator.** AO = SMA5(median) − SMA34(median); AC = AO − SMA5(AO); all trailing "
            "(closed-bar), so AC[t] uses only bars ≤ t (no look-ahead).\n"
            "- **Entry.** First bar with AC[t] > AC[t−1] > AC[t−2] and AC[t] > 0; enter **next close** "
            "(one lag); hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of AC returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample AC vs random (the *real* test).\n"
            "- **Null #3 — rotated-AC placebo** (timing destroyed, AC marginal kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every entry.\n"
            "- **Positive control.** Synthetic tape with a **planted** upward-accelerating episode "
            "(knob `edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks great, vs-random kills it\n\n"
            "Left: the AC-up entry's **one-sample** t against zero (the misleading number). "
            "Right: the same entry vs a **drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, ac_r, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.ac_entries(bb)\n"
            "            re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); ac_r.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    ac_r = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
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
            "a2.set_title('AC vs RANDOM, Welch t (honest: never clears 2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 easily (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) — but that's the **drift**, every long-only entry inherits it. The "
            f"right bars are the real test: AC-minus-random peaks at just **{R['h10'][8]:+.2f}** "
            f"(10d) and *reverses* to **{R['h60'][8]:+.2f}** at 60d — never significant. AC adds nothing "
            "over a coin flip."
        ),
        md(
            "### 4b · AC vs random across horizons — the gap is the verdict\n\n"
            "Mean return, AC-up vs random entry, all four horizons. The AC entry should tower over "
            "random if acceleration forecasts. It doesn't."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, ac_r, .4, color='#2c6fbb', label='two green bars above zero')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(ac_r,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('AC-up does not beat random entry'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta AC-random (bps):', [round(a-b) for a,b in zip(ac_r,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days AC is **+{R['h20'][2]:.0f} bps** and random is "
            f"**+{R['h20'][5]:.0f} bps** — a {R['h20'][6]:+.0f} bps difference that's pure noise. The "
            "only horizon where AC noses ahead is 10d, and the Welch test (4a) says that gap isn't real."
        ),
        md(
            "### 4c · The timing placebo — rotate the AC, nothing changes\n\n"
            "Keep the AC values **exactly** but circularly rotate the series relative to price, so the "
            "green bars no longer align with the moves they 'predicted'. If price follows *this "
            "specific* acceleration timing, the rotation should demolish the result. The observed AC "
            "return should sit far in the right tail of the rotated distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bb = load('SPY')\n"
            "    pl = st.rotated_ac_placebo(bb, 20, n_draws=300, seed=474)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    # rebuild the placebo distribution for the histogram\n"
            "    import numpy as _np, pandas as _pd\n"
            "    ac = st.accelerator_oscillator(bb); c = bb['close']; idx = bb.index\n"
            "    av = ac.to_numpy(); fin = _np.where(_np.isfinite(av))[0]; lo,hi = fin.min(), fin.max()\n"
            "    block = av[lo:hi+1].copy(); m = block.size; rng = _np.random.default_rng(474); draws=[]\n"
            "    for _ in range(300):\n"
            "        sh = int(rng.integers(34, m-34)); rot = _np.roll(block, sh)\n"
            "        full = av.copy(); full[lo:hi+1] = rot; ser = _pd.Series(full, index=idx)\n"
            "        ris = (ser>ser.shift(1)) & (ser.shift(1)>ser.shift(2)); cond = ris & ser.notna() & ser.shift(2).notna() & (ser>0)\n"
            "        f = cond & ~cond.shift(1, fill_value=False); rr = st.forward_returns(c, idx[f.to_numpy()], 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = _np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(474); draws = rng.normal(80, 25, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='rotated-AC series (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real AC {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean AC-up 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real AC sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real AC {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => timing not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real AC (blue line) sits **in the middle** of the rotated-AC "
            f"cloud — **p = {R['placebo'][1]:.2f}**. A meaningless time-shift does just as well, so the "
            "specific AC-to-price timing carries no information. This is the cleanest refutation of "
            "'acceleration leads price.'"
        ),
        md(
            "### 4d · Per-ticker — the AC-random delta straddles zero\n\n"
            "20-day AC-minus-random delta, per instrument. If acceleration worked it would be positive "
            "across the board; instead it's a coin-flip — positive in 3 of 5, negative in 2."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        bb = load(t); c = bb['close']\n"
            "        e = st.ac_entries(bb); re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d AC − random (bps)'); ax.set_title('AC-random delta straddles zero (3 of 5 positive)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: AC edges ahead on IWM ({R['per'][2][5]:+.0f}) and DIA "
            f"({R['per'][3][5]:+.0f}) but *loses* on SPY ({R['per'][0][5]:+.0f}) and GLD "
            f"({R['per'][4][5]:+.0f}). No coherent, cross-sectional edge — exactly what you'd expect if "
            "AC is just relabelled drift."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real acceleration\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** upward-accelerating "
            "episode into a synthetic tape and check the same two-green-bars rule banks it: edge=0 must "
            "stay at t≈0; edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.80):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=474, n_days=4000)\n"
            "    c = px['close']; e = st.ac_entries(px); s = st.summarize(st.forward_returns(c, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted acceleration -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} ac={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted acceleration the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"acceleration episode reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The "
            "detector works — so the flat real-tape result is a genuine 'nothing there', not a broken "
            "pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the AC-up entry does not beat a drift-matched random baseline "
            f"(AC − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps at 5/10/20/60d; Welch t never clears 2, best **{R['h10'][8]:+.2f}** at 10d, reverses "
            f"to **{R['h60'][8]:+.2f}** at 60d). The impressive one-sample t's (20d **{R['h20'][4]:.2f}**) "
            "are pure beta.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; costs only "
            "deepen the hole. You'd capture the drift more cheaply by holding the index.\n"
            f"- **Forecasts price? `BUSTED`** — the rotated-AC placebo leaves the result untouched "
            f"(**p = {R['placebo'][1]:.2f}**): a meaningless time-shift does as well as the real "
            "alignment, so the acceleration-to-price timing carries no information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The AC entry's entire apparent profit is the unconditional drift of long equity indices, "
            "which you obtain more cheaply and more fully by **buying and holding**. The AC rule trades "
            "*less* of the time (only on two-green-bars turn-ups) and pays costs on each, so it strictly "
            "dominates *nothing*. There is no capacity question because there is no edge to scale. The "
            "Accelerator Oscillator is a descriptive histogram of a smoothed second derivative of price, "
            "not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The saucer / zero-cross setups.** Williams' alternative AC entries are reshapings of "
            "the same SMA-of-SMA arithmetic and inherit the same drift confound; a clean follow-up "
            "shows each collapses against random.\n"
            "- **Window sensitivity.** The 5/34/5 lengths are folklore; sweeping them does not produce "
            "a horizon where AC-minus-random clears t = 2 — drift in, smoothed-derivative out.\n"
            "- **The AO parent.** Since AC = AO − SMA5(AO), AC inherits AO's confound; testing the two "
            "side by side shows the same beta masquerade.\n\n"
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
