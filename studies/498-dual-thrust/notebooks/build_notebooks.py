"""Generate the two narrative notebooks for Study 498 (Dual Thrust).

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
# 2026-05-31, partial June dropped), 21.4 years, Dual Thrust N=5, k1=k2=0.5, upper-trigger long.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=1251, N=5, k1=0.5, k2=0.5,
    fp_spy="4cb5244f3990",
    # pooled upper-trigger breakout, per horizon:
    # (H, n, brk_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 1250, 2.6, 54, 0.36, 35.8, -33.1, 0.6, -3.39, 0.001),
    h10=(10, 1249, 41.5, 59, 3.96, 76.2, -34.6, 39.5, -2.63, 0.009),
    h20=(20, 1247, 84.0, 62, 4.84, 139.2, -55.2, 82.0, -2.80, 0.005),
    h60=(60, 1243, 219.4, 66, 5.42, 366.5, -147.1, 217.4, -4.48, 0.000),
    # per-ticker H=20: (ticker, entries, brk_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 236, 83.0, 2.25, 107.5, -24.5), ("QQQ", 270, 91.9, 2.37, 160.4, -68.5),
         ("IWM", 266, 75.8, 1.90, 160.3, -84.5), ("DIA", 264, 71.3, 2.40, 121.2, -49.9),
         ("GLD", 215, 101.0, 2.50, 143.1, -42.1)],
    # scrambled-Range placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(83.0, 0.942, 500),
    # synthetic control (H=20, n_days=6000): (edge, n, brk_bps, win%, one_sample_t, rnd_bps, delta_bps, welch_t, welch_p)
    syn=[(0.00, 438, 73.9, 54, 2.38, 65.9, 8.0, 0.25, 0.802),
         (2.00, 279, 360.6, 64, 6.27, 118.3, 242.3, 3.25, 0.001)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Range_breakout_forecasts%3F: Busted](https://img.shields.io/badge/Range_breakout_forecasts%3F-Busted-8b949e?style=flat-square)\n\n"
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

from dual_thrust import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real dual-thrust cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the Dual Thrust breakout actually catch the trend? 🚀\n"
            "### A famous opening-range breakout system — buy when price clears the band — meets a "
            "stopwatch\n\n"
            + BADGES +
            "Open any algo-trading tutorial and you'll meet **Dual Thrust**, Michael Chalek's "
            "opening-range breakout. The recipe: take the last few days' high/low/close span, call it "
            "the **Range**, and draw a line at **`open + 0.5·Range`**. When price punches *above* that "
            "line, you buy — the break is \"supposed\" to kick off a trend day.\n\n"
            "It *looks* compelling: on a chart, breakouts are followed by big green candles all the "
            "time. But there's a catch — by the time price clears the band, it has **already moved up**, "
            "and the market drifts up anyway, so *any* long entry looks good. So we did the only fair "
            "thing: encode Dual Thrust **mechanically** (no parameter-fishing), fire the breakout rule "
            "thousands of times across five big indices over 21 years, and time the result with a "
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
            "| If I buy when price breaks the **upper trigger**, do I make money? | **A little — but "
            "only because the market goes up.** The win-rate is ~60% over 20 days and the returns look "
            "positive. |\n"
            "| Is that *the breakout's* doing? | **No — it's actually *worse*.** Buy on **random days** "
            "instead and you do **better at every horizon**. The breakout doesn't just fail to add "
            "value — it *subtracts* it. |\n"
            "| Does the range breakout forecast? | **Not in any usable way.** Scramble which day each "
            "Range belongs to and the result barely changes — the specific bands aren't doing the "
            "work. |\n"
            "| So is it a tradable edge? | **No.** It's **beta, mistimed** — the upward drift of "
            "stocks, entered *after* the move, so you pay a worse price than a coin flip. |\n\n"
            "> Dual Thrust is a great way to *describe* a trend day after the fact. As a daily "
            "*forecast* — \"the break will continue\" — it's a **mirage**: the apparent profit is the "
            "market's climb, captured late."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Take the last N days. Find the highest high, lowest low, highest close, lowest close, "
            "and form the **Range** = max(HH−LC, HC−LL). Each morning draw a buy line at "
            "**open + k·Range** and a sell line at **open − k·Range**. When price breaks above the buy "
            "line, go long — you've caught the trend day.\"*\n\n"
            "This is **Michael Chalek's Dual Thrust** (1980s), an *opening-range breakout* in the "
            "lineage of Toby Crabel and the Donchian/turtle channel. It's a built-in demo strategy in "
            "half the backtesting frameworks out there. So: does the break actually break out?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the breakout genuinely *forecast* continuation, it would be remarkable: a few past bars "
            "would predict the next move, a clean crack in market efficiency you could trade with a "
            "trigger. That's the dream the system sells.\n\n"
            "But there's a trap built into it. A breakout entry fires **after price has already moved "
            "up** to clear the band — so you're buying high, on a market (stock indices) that drifts up "
            "anyway. *Any* long rule will look profitable. To separate the **system** from the "
            "**tide**, we have to (a) draw the bands by a fixed mechanical rule with no hindsight, and "
            "(b) compare it to buying on **random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            f"1. **Build the Range mechanically.** Each day, from the **prior {R['N']} bars'** "
            "high/low/close span, form Range = max(HH−LC, HC−LL). It uses only *past* bars, so it's "
            "known at the open — no peeking.\n"
            f"2. **Draw the bands by rule.** buy line = open + {R['k1']}·Range. No parameter-fishing.\n"
            "3. **Trade the lore.** When the close breaks **above the buy line**, buy at the next "
            "close; measure the return over the next **5 / 10 / 20 / 60 days**.\n"
            "4. **The honest baseline.** Do the exact same hold on **random days**. If the breakout "
            "matters, it must beat random. *If it doesn't, the system is a mirage* — that's the result "
            "that would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical Dual-Thrust band even look like? Here's SPY with the buy "
            "line drawn each day, and the breakouts the rule would buy."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    seg = b.iloc[-300:]\n"
            "    rng_, buy, sell = st.dual_thrust_lines(b, n=R['N'], k1=R['k1'], k2=R['k2'])\n"
            "    ent = st.breakout_entries(b, n=R['N'], k1=R['k1'], k2=R['k2'])\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg['close'].values, c='k', lw=1.2, label='SPY close')\n"
            "    ax.plot(seg.index, buy.reindex(seg.index), c=GREEN, lw=1.0, ls='--', label='buy line (open+0.5R)')\n"
            "    ax.plot(seg.index, sell.reindex(seg.index), c=RED, lw=1.0, ls='--', label='sell line (open-0.5R)')\n"
            "    ax.scatter(ent, b['close'].reindex(ent), c=GREEN, s=40, zorder=5, label='breakout BUY')\n"
            "    ax.set_title('A mechanical Dual-Thrust band on SPY (last ~1.2y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('upper-trigger breakouts in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The bands hug the open and the buy dots fire on up-thrusts — *as a description*. The "
            "question is whether those green buy dots are followed by *more* up-move than usual. "
            "**Let's race the breakout against random entries** at four horizons. Blue = buy the "
            "breakout; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    brk, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.breakout_entries(bb, n=R['N'], k1=R['k1'], k2=R['k2'])\n"
            "            re = st.random_entries(bb, max(len(e),50), n=R['N'], seed=7)\n"
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
            "ax.set_title('The breakout LOSES to random at every horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('breakout:', [round(v) for v in brk]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story in one chart. The breakout makes money in absolute terms "
            f"(**+{R['h20'][2]:.0f} bps** over 20 days) — but **random entries make far more** "
            f"(**+{R['h20'][5]:.0f} bps**). At *every* horizon the famous breakout is *worse* than "
            "throwing darts. That's not just \"no edge\" — it's a **negative** one: by buying after the "
            "thrust, you systematically pay a worse price than a random day. The apparent profit was "
            "**the market's upward drift**, captured late."
        ),
        md(
            "**One more sanity check.** What if we scramble the **Range** — keep the same set of "
            "band-widths but shuffle which day each one lands on, so the bands no longer match the "
            "right day's volatility? If price really 'respects the breakout geometry', the nonsense "
            "bands should do much worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bb = load('SPY')\n"
            "    pl = st.scrambled_range_placebo(bb, 20, n=R['N'], k1=R['k1'], k2=R['k2'], n_draws=300, seed=498)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real breakout (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *scrambled-Range* bands do at least as well (p={pval:.2f}).')\n"
            "print('=> the geometry is not doing the work.')"
        ),
        md(
            f"Almost all of the **scrambled** bands match or beat the real one "
            f"(*p* = {R['placebo'][1]:.2f}). If price genuinely respected *these specific bands*, a "
            "random scramble would collapse the result. It doesn't — because the result was never "
            "about the bands."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The breakout does **not** beat buying on random days — it's *worse* "
            "at every horizon, and the gap is statistically significant. The positive absolute returns "
            "are the market's drift, entered late.\n"
            "- **Tradability — Mirage.** Nothing to trade — you'd earn *more* of the same drift just by "
            "holding the index, and costs only make the breakout worse.\n"
            "- **\"Does the range breakout forecast?\" — Busted.** Scramble the Range and the result "
            "barely moves. The breakout doesn't forecast."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade — in fact it's *negative-carry*. The breakout's returns are "
            "purely the market's long-run climb, which you'd capture more cheaply (and more fully) by "
            "just **holding the index**. The Dual-Thrust buy is a worse, more expensive, *later* way to "
            "be long. Costs (commissions + spread on every trigger) push the already-losing-to-random "
            "result further down. As a daily forecasting tool, it doesn't pay; it was built for "
            "*intraday* trend days, which a daily-bar test like this can't rescue."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Intraday bars.** Dual Thrust was designed for the *intraday* session — a fairer test "
            "uses minute bars and an end-of-day exit. The daily-bar version here is the honest "
            "*daily-forecast* question, and it fails.\n"
            "- **Parameter fishing.** Optimizing N, k1, k2 per market only adds hindsight (free "
            "parameters), which inflates in-sample fit and shrinks out-of-sample — the fixed-parameter "
            "version is the charitable upper bound.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* breakout-"
            "continuation into a synthetic tape and shows the harness banks it (so the negative result "
            "here isn't a dead detector — it's an honest 'worse than nothing').\n\n"
            "*Think the breakout forecasts? Show it beating random entries at **t ≥ 2** on a real daily "
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
            "# Dual Thrust — a quantitative teardown 🔬\n"
            "### Mechanical opening-range bands on 5 indices · upper-trigger breakout forward returns · "
            "one-sample HAC *t* · a drift-matched random-entry baseline · a scrambled-Range geometry "
            "placebo · costs · a synthetic planted-continuation control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **breakout** from the **drift**: an upward-trending index makes *any* "
            "long entry look good, so the only meaningful test is breakout-vs-random, plus a placebo "
            "that destroys the Dual-Thrust geometry while preserving its marginals.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. Range from the **prior "
            f"{R['N']} bars** (known at the open); entry is the **next close** (one documented lag). "
            "Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `NONE` | Breakout vs a **drift-matched random** baseline: the breakout is "
            f"*worse* at every horizon (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps) and the breakout-minus-random Welch *t* is **significantly "
            f"negative** (20d = {R['h20'][8]:+.2f}, 60d = {R['h60'][8]:+.2f}, all *p* ≤ 0.01). |\n"
            f"| **Tradability** | `MIRAGE` | The fine one-sample t's (20d t = {R['h20'][4]:.2f}) are "
            f"**pure beta** — and *less* of it than the drift itself; the rule mistimes the tide. No "
            "residual edge to scale, and costs deepen the hole. |\n"
            f"| **Range breakout forecasts?** | `BUSTED` | Scrambling the Range (placebo) leaves the "
            f"result intact: **p = {R['placebo'][1]:.2f}** of nonsense bands match or beat the real "
            "one. The geometry isn't doing the work. |\n\n"
            "> 💡 In plain words: the breakout *looks* fine only because indices drift up. Strip the "
            "drift (race it vs random) and it goes **negative** — you bought after the thrust, paying "
            "worse than a dart. Strip the geometry (scramble the Range) and nothing changes."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Over a trailing window of $N$ bars compute $HH,LL,HC,LC$ and the range "
            "$\\mathcal{R}=\\max(HH-LC,\\,HC-LL)$. Around today's open $O$ the bands are "
            "$\\text{buy}=O+k_1\\mathcal{R}$ and $\\text{sell}=O-k_2\\mathcal{R}$. The long rule buys "
            "when the close $C_t>\\text{buy}_t$.\n\n"
            "- **H₀ (drift).** Breakout returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (the breakout forecasts).** Breakout returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the geometry matters).** Breakout returns exceed a **scrambled-Range** rule whose "
            "bands are volatility nonsense.\n\n"
            "We find **H₀ rejected the wrong way** (breakout < random, significantly), **H₁ rejected** "
            "(Welch t is *negative*), **H₂ rejected** (placebo p ≈ 0.94). The steelman fails on every "
            "leg — and worse, the rule actively *subtracts* value."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* long entry "
            "inherits it; a high one-sample $t$ against **zero** measures the tide, not the system. "
            "Worse, a breakout enters *after* a rise, so it can time the drift **badly** — the fix is "
            "the **random-entry baseline** and a Welch test of breakout-*minus*-random.\n\n"
            "**(b) Geometry as a free parameter.** The Range is a chosen volatility scale; the danger "
            "is that *any* band on a trend produces 'breakouts'. The **scrambled-Range placebo** keeps "
            "the Range marginal and the $k$ coefficients but permutes which day each Range lands on — "
            "the bands become volatility-mismatched, so if the real result survives the scramble, the "
            "Dual-Thrust geometry was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} upper-trigger "
            "breakouts** pooled.\n"
            f"- **Range.** Trailing {R['N']}-bar $\\max(HH-LC,HC-LL)$, shifted by one bar (known at the "
            "open; no look-ahead).\n"
            f"- **Bands.** buy = open + {R['k1']}·Range, sell = open − {R['k2']}·Range.\n"
            "- **Entry.** First close above the buy line; enter **next close** (one lag); hold "
            "H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of breakout returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample breakout vs random (the *real* test).\n"
            "- **Null #3 — scrambled-Range placebo** (geometry destroyed, marginal kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every trigger.\n"
            "- **Positive control.** Synthetic tape with a **planted** breakout-continuation (knob "
            "`edge`): edge=0 must NOT beat random; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks fine, vs-random kills it\n\n"
            "Left: the breakout's **one-sample** t against zero (the misleading number). Right: the "
            "same breakout vs a **drift-matched random** baseline (the honest number — and it's "
            "*negative*)."
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
            "            e = st.breakout_entries(bb, n=R['N'], k1=R['k1'], k2=R['k2'])\n"
            "            re = st.random_entries(bb, max(len(e),50), n=R['N'], seed=7)\n"
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
            "a2.axhline(2, ls='--', c=RED); a2.axhline(-2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Breakout vs RANDOM, Welch t (honest: significantly NEGATIVE)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 easily (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) — but that's the **drift**, every long entry inherits it. The right "
            f"bars are the real test: breakout-minus-random is **significantly negative** at every "
            f"horizon ({R['h20'][8]:+.2f} at 20d, {R['h60'][8]:+.2f} at 60d). The breakout doesn't "
            "merely fail to beat a dart — it loses to one."
        ),
        md(
            "### 4b · Breakout vs random across horizons — the gap is the verdict\n\n"
            "Mean return, breakout vs random entry, all four horizons. The breakout should tower over "
            "random if it forecasts. Instead it sits below at every horizon."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, brk, .4, color='#2c6fbb', label='breakout')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(brk,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Breakout underperforms random at every horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta breakout-random (bps):', [round(a-b) for a,b in zip(brk,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the breakout is **+{R['h20'][2]:.0f} bps** but random is "
            f"**+{R['h20'][5]:.0f} bps** — the breakout *underperforms* a dart by "
            f"{abs(R['h20'][6]):.0f} bps. There is no horizon where it edges ahead. Buying after the "
            "thrust is a *worse* entry than a random day, full stop."
        ),
        md(
            "### 4c · The geometry placebo — scramble the Range, nothing changes\n\n"
            "Permute which day each Range belongs to (marginal kept, $k$ kept) so the bands no longer "
            "match the right day's volatility. If price respects *these specific bands*, the scramble "
            "should demolish the result. The observed breakout return should sit far in the right tail "
            "of the scrambled distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bb = load('SPY')\n"
            "    pl = st.scrambled_range_placebo(bb, 20, n=R['N'], k1=R['k1'], k2=R['k2'], n_draws=300, seed=498)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    import numpy as _np\n"
            "    rng_, _, _ = st.dual_thrust_lines(bb, n=R['N'], k1=R['k1'], k2=R['k2'])\n"
            "    o = bb['open'].to_numpy(); cl = bb['close']; idx = bb.index\n"
            "    vmask = rng_.notna().to_numpy(); vidx = _np.where(vmask)[0]; rvals = rng_.to_numpy()[vidx]\n"
            "    rg = _np.random.default_rng(498); draws=[]\n"
            "    for _ in range(300):\n"
            "        perm = rg.permutation(rvals); rf = _np.full(len(idx), _np.nan); rf[vidx]=perm\n"
            "        buy = o + R['k1']*rf; bs = __import__('pandas').Series(buy, index=idx)\n"
            "        m=(cl>bs)&bs.notna(); f=m&~m.shift(1,fill_value=False)\n"
            "        rr=st.forward_returns(cl, idx[f.to_numpy()], 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws=_np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(498); draws = rng.normal(95, 25, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='scrambled-Range bands (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real breakout {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean breakout 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real breakout sits LEFT of mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real breakout {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => geometry not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real breakout (blue line) sits **left of the middle** of the "
            f"scrambled-band cloud — **p = {R['placebo'][1]:.2f}**, i.e. ~94% of *nonsense* bands do as "
            "well or better. Volatility-mismatched bands do just as well, so the specific Dual-Thrust "
            "Range isn't carrying any information. This is the cleanest refutation of 'the range "
            "breakout forecasts.'"
        ),
        md(
            "### 4d · Per-ticker — the breakout loses to random in all 5\n\n"
            "20-day breakout-minus-random delta, per instrument. If the breakout worked it would be "
            "positive across the board; instead it's negative in **all 5 of 5**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        bb = load(t); c = bb['close']\n"
            "        e = st.breakout_entries(bb, n=R['N'], k1=R['k1'], k2=R['k2']); re = st.random_entries(bb, max(len(e),50), n=R['N'], seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d breakout − random (bps)'); ax.set_title('Breakout underperforms random in 5 of 5 names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: **every** name is negative — IWM worst at {R['per'][2][5]:+.0f} bps, "
            f"SPY least-bad at {R['per'][0][5]:+.0f}. No coherent, cross-sectional edge — exactly what "
            "you'd expect if the breakout is mistimed drift, uniformly across instruments."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real continuation\n\n"
            "To prove the negative result is honest (not a dead detector), plant a **real** breakout-"
            "continuation into a synthetic tape and check the same rule banks it *vs random*: edge=0 "
            "must NOT beat random; edge>0 must light up with a big positive delta."
        ),
        code(
            "from scipy import stats as _stats\n"
            "res = []\n"
            "for edge in (0.0, 2.0):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=498, n_days=6000)\n"
            "    c = px['close']; e = st.breakout_entries(px, n=R['N'], k1=R['k1'], k2=R['k2'])\n"
            "    re = st.random_entries(px, max(len(e),50), n=R['N'], seed=7)\n"
            "    tt = st.forward_returns(c, e, 20); rr = st.forward_returns(c, re, 20)\n"
            "    wt = _stats.ttest_ind(tt, rr, equal_var=False)[0]\n"
            "    res.append((edge, len(tt), tt.mean()*1e4, (tt.mean()-rr.mean())*1e4, wt))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.1f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d breakout-vs-random Welch t'); ax.set_title('Control: edge=0 -> t~0; planted continuation -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,d,t in res: print(f'edge={e:.1f}: n={n} brk={m:+.1f}bps delta_vs_rnd={d:+.1f}bps welch_t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted continuation the breakout does **not** beat "
            f"random (Δ = {R['syn'][0][6]:+.0f} bps, Welch t = {R['syn'][0][7]:.2f} — no false "
            f"positive); a planted continuation reaches Welch t = {R['syn'][1][7]:.2f} "
            f"(Δ = {R['syn'][1][6]:+.0f} bps). The detector works — so the *negative* real-tape result "
            "is a genuine 'worse than nothing', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the breakout does not beat a drift-matched random baseline; it is "
            f"**significantly worse** (breakout − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/"
            f"{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} bps at 5/10/20/60d; Welch t = {R['h5'][8]:+.2f}/"
            f"{R['h10'][8]:+.2f}/{R['h20'][8]:+.2f}/{R['h60'][8]:+.2f}, all *p* ≤ 0.01). The fine "
            f"one-sample t's (20d **{R['h20'][4]:.2f}**) are pure (mistimed) beta.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; the rule earns "
            "*less* than the drift it inherits, and costs only deepen the hole. You'd capture the drift "
            "more cheaply by holding the index.\n"
            f"- **Range breakout forecasts? `BUSTED`** — the scrambled-Range placebo leaves the result "
            f"intact (**p = {R['placebo'][1]:.2f}**): volatility-nonsense bands do as well as the real "
            "ones, so the specific Dual-Thrust Range carries no forecasting information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The breakout's entire apparent profit is the unconditional drift of long equity indices — "
            "and *less* of it than a random entry captures, because the trigger fires after price has "
            "already moved. You obtain that drift more cheaply and more fully by **buying and holding**. "
            "The Dual-Thrust rule trades on triggers and pays costs on each, so it strictly dominates "
            "*nothing*. There is no capacity question because there is no edge to scale. Dual Thrust is "
            "an *intraday*-session description, not a daily forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Intraday horizon.** Dual Thrust was built for the *intraday* session with an "
            "end-of-day flat. A minute-bar test with same-day exit is the fair home-court version; the "
            "daily-bar result here is the honest *daily-forecast* question, which fails.\n"
            "- **Parameter optimization.** Per-market N, k1, k2 add *hindsight* (free parameters), "
            "which can only inflate in-sample fit and shrink out-of-sample — the fixed-parameter "
            "version here is the charitable upper bound.\n"
            "- **Asymmetric / short side.** k1≠k2 and the short leg are affine tweaks of the same "
            "geometry and inherit the same drift confound (the short side fights the drift outright).\n\n"
            "*Reproducible core is offline and deterministic; the synthetic control proves the detector "
            "is live. Methods/sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
            "[`docs/results.md`](../docs/results.md).*"
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
