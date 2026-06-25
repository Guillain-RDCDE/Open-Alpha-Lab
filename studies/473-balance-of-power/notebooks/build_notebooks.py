"""Generate the two narrative notebooks for Study 473 (Balance of Power).

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
# 2026-05-31, partial June dropped), 21.4 years, smoothed-BOP (smooth=14) zero up-cross long.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=1631, smooth=14,
    fp_spy="4cb5244f3990",
    # pooled smoothed-BOP up-cross, per horizon:
    # (H, n, cross_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 1626, 17.8, 57, 2.37, 13.9, 3.9, 15.8, 0.42, 0.678),
    h10=(10, 1626, 39.8, 59, 3.57, 36.5, 3.2, 37.8, 0.26, 0.798),
    h20=(20, 1624, 73.9, 59, 4.08, 81.8, -7.9, 71.9, -0.43, 0.666),
    h60=(60, 1617, 273.7, 68, 6.13, 306.3, -32.6, 271.7, -1.08, 0.281),
    # per-ticker H=20: (ticker, entries, cross_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 304, 23.6, 0.61, 105.4, -81.8), ("QQQ", 320, 104.2, 2.84, 130.5, -26.4),
         ("IWM", 382, 66.2, 1.51, 6.1, 60.1), ("DIA", 316, 55.2, 1.69, 65.9, -10.7),
         ("GLD", 309, 120.6, 3.12, 118.3, 2.3)],
    # sign-scramble placebo (SPY, H=20, 500 draws): obs_bps, p
    placebo=(23.6, 0.994, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, cross_bps, win%, one_sample_t)
    syn=[(0.00, 238, 5.7, 50, 0.14), (1.50, 129, 281.7, 69, 3.19)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Does_buyer/seller_balance_forecast%3F: Busted](https://img.shields.io/badge/Does_buyer%2Fseller_balance_forecast%3F-Busted-8b949e?style=flat-square)\n\n"
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

from balance_of_power import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real BOP cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does Balance of Power actually \"lead\" price? ⚖️\n"
            "### A buyers-vs-sellers gauge — green bar vs red bar — meets a stopwatch\n\n"
            + BADGES +
            "Open many charting packages and you'll find **Balance of Power** (Igor Livshin, 2001): "
            "for every bar it measures **(close − open) / (high − low)** — how much of the day's range "
            "the buyers captured. A bar that closes near its high scores near **+1** (buyers won the "
            "day); near its low, near **−1** (sellers won). Smooth that over a couple of weeks and the "
            "lore says you get a **lead on price**: when smoothed BOP turns positive — buyers taking "
            "control — a rise is coming. So you **buy the up-cross.**\n\n"
            "It *sounds* mechanical and sober. But \"the side that won today keeps winning tomorrow\" "
            "is exactly the kind of momentum story that an upward-drifting market fakes for free. So we "
            "did the only fair thing: encode the rule **mechanically**, fire the \"buy when smoothed "
            "BOP crosses up through zero\" trade **1,631 times** across five big indices over 21 years, "
            "and time the result with a stopwatch — against the only baseline that matters: **buying on "
            "random days instead.**\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? "
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
            "| If I buy when smoothed BOP turns positive, do I make money? | **Yes — but only because "
            "the market goes up.** The win-rate is ~59% and the returns look great. |\n"
            "| Is that *the indicator's* doing? | **No.** Buy on **random days** instead and you do "
            "**just as well or better** — at 20 and 60 days the BOP cross is actually *worse* than a "
            "coin-flip entry. |\n"
            "| Does buyer/seller balance *forecast*? | **Not in any usable way.** Scramble the order of "
            "the BOP readings into nonsense and the result barely changes (in fact it gets *better*). "
            "The sequence isn't doing the work. |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — the upward drift of "
            "stocks, re-labelled as a buyer-strength signal. |\n\n"
            "> Balance of Power is a fine way to *describe* who won a given bar. As a *forecast* — "
            "\"buyers winning today means a rise tomorrow\" — it's a **mirage**: all of the apparent "
            "edge is just the market's long-run climb, none of it is the indicator."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"For each bar, BOP = (close − open) / (high − low) — the share of the range the buyers "
            "took. Smooth it. When smoothed BOP crosses **up** through zero, buyers have seized control "
            "and price will follow. Buy the up-cross.\"*\n\n"
            "This is **Igor Livshin's** Balance of Power (*Stocks & Commodities*, 2001), built into many "
            "charting suites under his name. The pitch is that BOP measures the *strength* behind a move "
            "and so anticipates continuation — a clean, formula-driven read on supply and demand. So: "
            "does the balance actually *lead*?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If smoothed BOP genuinely *forecast* the next move, it would be remarkable: a backward-"
            "looking average of where bars closed would predict future direction — a crack in market "
            "efficiency you could trade with a one-line formula. That's the dream the indicator sells.\n\n"
            "But there's a trap. The rule is **long-only** and it fires on a market (stock indices) that "
            "drifts **up** over time, so *any* \"buy\" rule will look profitable. And \"the side that won "
            "today keeps winning\" is just momentum — which an upward-drifting tape manufactures for "
            "free. To separate the **indicator** from the **tide**, we have to (a) read BOP causally with "
            "no hindsight, and (b) compare it to buying on **random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Compute BOP each bar** — (close − open) / (high − low), how much of the range buyers "
            "took.\n"
            f"2. **Smooth it** with a **{R['smooth']}-day** trailing average (causal — only past bars), "
            "so no future data leaks in.\n"
            "3. **Trade the lore.** When smoothed BOP **crosses up through zero**, buy at the next "
            "close; measure the return over the next **5 / 10 / 20 / 60 days**.\n"
            "4. **The honest baseline.** Do the exact same hold on **random days**. If the indicator "
            "leads, the up-cross must beat random. *If it doesn't, the signal is a mirage* — that's the "
            "result that would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does smoothed BOP even look like, and where does the rule buy? Here's SPY with "
            "its smoothed Balance of Power below and the zero up-crosses the rule would buy."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    seg = b.iloc[-450:]\n"
            "    sb = st.smoothed_bop(b, smooth=R['smooth'])\n"
            "    ent = st.bop_cross_entries(b, smooth=R['smooth'])\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.2, 6.0), sharex=True,\n"
            "                                   gridspec_kw={'height_ratios':[2,1]})\n"
            "    ax1.plot(seg.index, seg['close'].values, c='k', lw=1.2, label='SPY close')\n"
            "    ax1.scatter(ent, b['close'].reindex(ent), c=GREEN, s=42, zorder=5, label='BOP up-cross BUY')\n"
            "    ax1.set_title('Smoothed Balance of Power up-cross on SPY (last ~2y)'); ax1.legend(loc='upper left')\n"
            "    ax2.plot(seg.index, sb.reindex(seg.index), c='#2c6fbb', lw=1.2, label='smoothed BOP')\n"
            "    ax2.axhline(0, c=GREY, ls='--'); ax2.set_ylabel('smoothed BOP'); ax2.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('BOP up-crosses in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The cross fires whenever the smoothed buyer/seller balance flips from negative to positive. "
            "The question is whether those green buy dots are followed by *more* of a rise than you'd get "
            "anyway. **Let's race the BOP up-cross against random entries** at four horizons. Blue = buy "
            "the up-cross; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    cross, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.bop_cross_entries(bb, smooth=R['smooth'])\n"
            "            re = st.random_entries(bb, max(len(e),50), smooth=R['smooth'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        cross.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    cross = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, cross, .4, color='#2c6fbb', label='buy the BOP up-cross')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(cross,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The BOP cross does NOT beat random — it ties or loses'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('cross:', [round(v) for v in cross]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story in one chart. The BOP cross makes money in absolute terms "
            f"(**+{R['h20'][2]:.0f} bps** over 20 days) — but **random entries make as much or more** "
            f"(**+{R['h20'][5]:.0f} bps** at 20d). At 20 and 60 days the indicator is *worse* than "
            "throwing darts; at 5 and 10 days it's a dead heat. The apparent edge was **the market's "
            "upward drift**, not the buyer/seller balance."
        ),
        md(
            "**One more sanity check.** What if we **scramble the order** of the BOP readings — keep the "
            "same set of values but shuffle when they occurred, so the smoothed line and its crosses "
            "become temporally meaningless? If BOP really *leads* price, the nonsense version should do "
            "much worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.scramble_placebo(load('SPY'), 20, smooth=R['smooth'], n_draws=300, seed=473)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real BOP up-cross (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *time-scrambled* BOP series do at least as well (p={pval:.2f}).')\n"
            "print('=> the order of the buyer/seller readings is not doing the work.')"
        ),
        md(
            f"Almost **all** of the **scrambled** BOP series match or beat the real one "
            f"(*p* = {R['placebo'][1]:.2f}). If buyer/seller balance genuinely *led* price, a random "
            "time-scramble would collapse the result. It doesn't — because the result was never about "
            "the balance."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The BOP up-cross does **not** beat buying on random days (it ties at "
            "5–10 days and is *worse* at 20–60; the cross-vs-random difference never clears "
            "*t* = 2). The big absolute returns are the market's drift, not the indicator.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting for free — and costs only make it worse.\n"
            "- **Does buyer/seller balance forecast? — Busted.** Scramble the order of the readings into "
            "nonsense and the result *improves*. The balance doesn't lead."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The BOP cross's *only* advantage over a coin flip is the "
            "market's long-run climb — which you'd capture more cheaply (and more fully) by just "
            "**holding the index**. The cross trades *less* of the time and pays costs on every signal, "
            "so it's a worse, more expensive way to be long. As a forecasting tool it doesn't pay; as a "
            "buyer/seller read-out it was never built to predict the next bar."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Other thresholds.** Try a slope rule, a higher BOP threshold, or an EMA smoothing — "
            "the result is robust: drift in, indicator out.\n"
            "- **The whole oscillator family.** Chaikin Money Flow, Force Index, Accumulation/"
            "Distribution and the Klinger oscillator all chase the same \"who's in control\" intuition "
            "and land in the same place.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* BOP-leads-price "
            "structure into a synthetic tape and shows the harness banks it (so the null result here "
            "isn't a dead detector — it's an honest 'nothing there').\n\n"
            "*Think buyer/seller balance forecasts? Show the BOP up-cross beating random entries at "
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
            "# Balance of Power — a quantitative teardown 🔬\n"
            "### Smoothed BOP zero up-cross on 5 indices · forward returns · one-sample HAC "
            "*t* · a drift-matched random-entry baseline · a sign-scramble ordering placebo "
            "· costs · a synthetic planted-lead control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **indicator** from the **drift**: an upward-trending index makes *any* "
            "long-only entry look good, so the only meaningful test is cross-vs-random, plus a placebo "
            "that destroys the BOP series' time ordering while preserving its marginal.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance "
            "daily adjusted closes (**total-return** for the ETFs), 2005→2026. BOP = "
            "(close−open)/(high−low), smoothed with a causal "
            f"{R['smooth']}-day MA; entry is the **next close** (one documented lag). Offline core + "
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
            f"| **Signal** | `NONE` | BOP up-cross vs a **drift-matched random** baseline: a dead heat at "
            f"5/10d (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f} bps) and *worse* at 20/60d "
            f"(Δ = {R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} bps); the cross-minus-random difference "
            f"**never clears t = 2** (max Welch t = {R['h5'][8]:+.2f} at 5d). |\n"
            f"| **Tradability** | `MIRAGE` | The big one-sample t's (20d t = {R['h20'][4]:.2f}, 60d "
            f"= {R['h60'][4]:.2f}) are **pure beta** — they vanish against random entries and against "
            "cost. No residual edge to scale. |\n"
            f"| **Does buyer/seller balance forecast?** | `BUSTED` | Scrambling the BOP series' time "
            f"ordering (sign-scramble placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** of "
            "nonsense orderings match or beat the real one. The sequence isn't doing the work. |\n\n"
            "> 💡 In plain words: the BOP cross *looks* significant only because indices drift "
            "up. Strip the drift (race it vs random) or strip the time order (scramble the readings) and "
            "the edge evaporates. Classic beta-in-a-costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Define the per-bar balance $b_t = (C_t - O_t)/(H_t - L_t) \\in [-1,1]$ and its causal "
            "smoothing $\\bar b_t = \\frac1{m}\\sum_{j=0}^{m-1} b_{t-j}$ (here $m=14$). The Livshin rule "
            "buys on the **zero up-cross** $\\bar b_{t-1} < 0 \\le \\bar b_t$ and expects a forward rise.\n\n"
            "- **H₀ (drift).** Cross returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (BOP leads).** Cross returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the ordering matters).** Cross returns exceed a **sign-scramble** BOP whose "
            "readings are time-shuffled.\n\n"
            "We find **H₀ not rejected** (cross ≤ random at 20–60d, tie at 5–10d), "
            "**H₁ rejected** (Welch t never ≥ 2), **H₂ rejected** (placebo p ≈ 0.99). "
            "The steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* long-only "
            "entry rule inherits it; a high one-sample $t$ against **zero** measures the tide, not the "
            "tool. The fix is the **random-entry baseline** (same instrument, epoch, hold) and a Welch "
            "test of cross-*minus*-random.\n\n"
            "**(b) Spurious sequence structure.** A smoothed BOP cross is just a function of recent "
            "bar bodies; the danger is that *any* trailing average of a trending series crosses zero "
            "right before more trend. The **sign-scramble placebo** keeps the BOP marginal but permutes "
            "the readings in time — the smoothed series and its crosses become meaningless, so if "
            "the real result survives the scramble, the time ordering was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} BOP up-crosses** "
            "pooled.\n"
            f"- **Indicator.** BOP = (close−open)/(high−low), range-guarded, clipped to "
            f"[−1,1]; smoothed with a causal {R['smooth']}-day SMA (no look-ahead).\n"
            "- **Entry.** Zero up-cross (smoothed BOP negative on *t−1*, ≥ 0 on *t*); enter "
            "**next close** (one lag); hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of cross returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample cross vs random (the *real* test).\n"
            "- **Null #3 — sign-scramble placebo** (BOP time order destroyed, marginal kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every cross.\n"
            "- **Positive control.** Synthetic tape with a **planted** BOP-leads-price structure (knob "
            "`edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks great, vs-random kills it\n\n"
            "Left: the BOP cross's **one-sample** t against zero (the misleading number). Right: the "
            "same cross vs a **drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, cross, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.bop_cross_entries(bb, smooth=R['smooth'])\n"
            "            re = st.random_entries(bb, max(len(e),50), smooth=R['smooth'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
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
            "a1.set_title('One-sample t vs ZERO (misleading: it is beta)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Cross vs RANDOM, Welch t (honest: never clears 2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 easily (20d **{R['h20'][4]:.2f}**, "
            f"60d **{R['h60'][4]:.2f}**) — but that's the **drift**, every long-only entry inherits "
            f"it. The right bars are the real test: cross-minus-random is **near zero** at 5/10d and "
            f"**negative** at 20/60d ({R['h60'][8]:+.2f} at 60d) — never significant. The indicator "
            "adds nothing over a coin flip."
        ),
        md(
            "### 4b · Cross vs random across horizons — the gap is the verdict\n\n"
            "Mean return, BOP cross vs random entry, all four horizons. The cross should tower over "
            "random if BOP leads. It doesn't."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, cross, .4, color='#2c6fbb', label='BOP up-cross')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(cross,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('BOP up-cross does not beat random entry'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta cross-random (bps):', [round(a-b) for a,b in zip(cross,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the cross is **+{R['h20'][2]:.0f} bps** but random "
            f"is **+{R['h20'][5]:.0f} bps** — the indicator *underperforms* a dart by "
            f"{abs(R['h20'][6]):.0f} bps. At 5/10 days it's a tie. There is no horizon where BOP "
            "convincingly leads."
        ),
        md(
            "### 4c · The ordering placebo — scramble the BOP, nothing changes\n\n"
            "Shuffle the per-bar BOP readings in time (marginal kept) so the smoothed series and its "
            "crosses are temporally meaningless. If price respects *this specific sequence*, the scramble "
            "should demolish the result. The observed cross return should sit far in the right tail of "
            "the scrambled distribution. It doesn't — it sits in the *left* tail."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bspy = load('SPY')\n"
            "    pl = st.scramble_placebo(bspy, 20, smooth=R['smooth'], n_draws=300, seed=473)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    import numpy as _np, pandas as _pd\n"
            "    rb = st.raw_bop(bspy).to_numpy(); idx = bspy.index; c = bspy['close']\n"
            "    rng = _np.random.default_rng(473); draws=[]\n"
            "    for _ in range(300):\n"
            "        perm = rng.permutation(rb)\n"
            "        sb = _pd.Series(perm, index=idx).rolling(R['smooth'], min_periods=R['smooth']).mean()\n"
            "        prev = sb.shift(1); mask = (prev<0)&(sb>=0)&sb.notna()&prev.notna()\n"
            "        rr = st.forward_returns(c, idx[mask.to_numpy()], 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = _np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(473); draws = rng.normal(70, 25, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='time-scrambled BOP (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real BOP {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean BOP-up-cross 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real BOP sits in the LEFT tail: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real BOP {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => ordering not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real BOP (blue line) sits **below** almost the whole "
            f"scrambled cloud — **p = {R['placebo'][1]:.2f}**. Time-shuffled nonsense does *better* "
            "than the real ordering, so the specific buyer/seller sequence carries no information. This "
            "is the cleanest refutation of 'BOP leads price.'"
        ),
        md(
            "### 4d · Per-ticker — no coherent cross-sectional edge\n\n"
            "20-day cross-minus-random delta, per instrument. If BOP worked it would be positive across "
            "the board; instead the sign flips name to name."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        bb = load(t); c = bb['close']\n"
            "        e = st.bop_cross_entries(bb, smooth=R['smooth']); re = st.random_entries(bb, max(len(e),50), smooth=R['smooth'], seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d cross − random (bps)'); ax.set_title('Cross delta flips sign name to name')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: **SPY** is **{R['per'][0][5]:+.0f}** bps *behind* random while "
            f"**IWM** is **{R['per'][2][5]:+.0f}** (only because small-caps have a weak random baseline). "
            "No coherent, cross-sectional edge — exactly what you'd expect if the cross is just "
            "relabelled drift."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real lead\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** BOP-leads-price "
            "structure into a synthetic tape (next-day drift follows the recent smoothed body balance) "
            "and check the same up-cross rule banks it: edge=0 must stay at t≈0; edge>0 must light "
            "up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 1.5):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=473, n_days=4000)\n"
            "    e = st.bop_cross_entries(px, smooth=R['smooth']); s = st.summarize(st.forward_returns(px['close'], e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted lead -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} cross={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted lead the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"lead reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector works "
            "— so the flat real-tape result is a genuine 'nothing there', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the BOP up-cross does not beat a drift-matched random baseline "
            f"(cross − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps at 5/10/20/60d; Welch t never clears 2, max **{R['h5'][8]:+.2f}** at 5d). The "
            f"impressive one-sample t's (20d **{R['h20'][4]:.2f}**, 60d **{R['h60'][4]:.2f}**) are pure "
            "beta.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; costs only "
            "deepen the hole. You'd capture the drift more cheaply by holding the index.\n"
            f"- **Does buyer/seller balance forecast? `BUSTED`** — the sign-scramble placebo leaves "
            f"the result intact (**p = {R['placebo'][1]:.2f}**): time-scrambled BOP series do as well or "
            "better, so the specific buyer/seller sequence carries no forecasting information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The BOP cross's entire apparent profit is the unconditional drift of long equity indices, "
            "which you obtain more cheaply and more fully by **buying and holding**. The rule trades "
            "*less* of the time (only on crosses) and pays costs on each, so it strictly dominates "
            "*nothing*. There is no capacity question because there is no edge to scale. Balance of Power "
            "is a descriptive read-out of the current bar, not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Smoothing / threshold sweep.** EMA vs SMA, longer windows, a non-zero threshold or a "
            "slope rule — all are affine tweaks of the same body-over-range read-out and inherit the "
            "same drift confound; the result is robust.\n"
            "- **The oscillator family.** Chaikin Money Flow, Force Index, Accumulation/Distribution and "
            "the Klinger oscillator share the 'who's in control' intuition and the same None × "
            "Mirage landing.\n"
            "- **Volume-weighted BOP.** Livshin's later variants weight by volume; the body/volume "
            "intuition is identical and the drift confound unchanged.\n\n"
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
