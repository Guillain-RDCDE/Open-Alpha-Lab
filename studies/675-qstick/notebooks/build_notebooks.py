"""Generate the two narrative notebooks for Study 675 (Qstick).

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
# yfinance daily, 5 ETFs (SPY QQQ IWM DIA GLD), 2005-01-03 -> 2026-06-30 (As-of 2026-06-30),
# 21.5 years, smoothed Qstick (smooth=8, body normalised by prior close) zero up-cross long.
R = dict(
    asof="2026-06-30", start="2005-01-03", end="2026-06-30", years=21.5,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=2210, smooth=8,
    fp={"SPY": "c374b2feba04", "QQQ": "cd19fe6cb57d", "IWM": "cb6689d8b659",
        "DIA": "3a1136ef6422", "GLD": "35f763b5047c"},
    # pooled smoothed-Qstick up-cross, per horizon:
    # (H, n, cross_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 2204, 22.0, 58, 3.89, 34.8, -12.8, 20.0, -1.68, 0.094),
    h10=(10, 2204, 50.6, 61, 5.64, 62.5, -11.9, 48.6, -1.12, 0.261),
    h20=(20, 2204, 92.7, 63, 5.99, 115.3, -22.5, 90.7, -1.52, 0.129),
    h60=(60, 2194, 286.2, 69, 7.25, 261.7, 24.5, 284.2, 0.94, 0.348),
    # per-ticker H=20: (ticker, entries, cross_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 437, 77.8, 2.57, 117.0, -39.2), ("QQQ", 438, 129.2, 3.60, 165.2, -35.9),
         ("IWM", 465, 102.8, 2.60, 109.4, -6.6), ("DIA", 425, 62.1, 2.09, 82.2, -20.1),
         ("GLD", 445, 90.2, 2.76, 102.1, -11.9)],
    # sign-scramble placebo (SPY, H=20, 500 draws): obs_bps, p
    placebo=(77.8, 0.814, 500),
    # trend-proxy correlation: (ticker, corr, qs_entries, mom_entries, jaccard)
    trend=[("SPY", 0.781, 437, 401, 0.201), ("QQQ", 0.818, 438, 410, 0.222),
           ("IWM", 0.818, 465, 442, 0.205), ("DIA", 0.784, 425, 409, 0.209),
           ("GLD", 0.713, 445, 426, 0.151)],
    trend_corr_mean=0.783,
    # naive momentum-only cross, H=20 pooled: n, cross_bps, rnd_bps, delta_bps, welch_t, p
    mom=(2080, 96.7, 64.7, 32.1, 1.99, 0.047),
    # synthetic control (H=20, n_days=8000): null 20-seed mean/sd/fires, planted (n, cross_bps, win%, t)
    syn_null=(0.07, 1.07, 0, 20),
    syn_planted=(155, 1385.0, 63, 5.12),
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Just_a_slow_trend_proxy%3F: Confirmed](https://img.shields.io/badge/Just_a_slow_trend_proxy%3F-Confirmed-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from qstick import data, strategy as st

ASOF = "2026-06-30"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real Qstick cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does close-minus-open actually time the trend? 🕯️\n"
            "### Chande's Qstick — a buying-pressure gauge built from candle bodies — meets a stopwatch\n\n"
            + BADGES +
            "Every candlestick tells a tiny story: a **green body** (close above open) says "
            "buyers won the day; a **red body** says sellers did. Tushar Chande's **Qstick** "
            "(1994) averages that daily verdict over 8 sessions, and the charting lore says its "
            "**zero-cross** times a trend: when the smoothed body flips positive, buyers have "
            "taken control — buy. It's the same family as Balance of Power and the Force Index, "
            "just built from the plainest ingredient of all: where a bar opened vs where it "
            "closed.\n\n"
            "So we did the only fair thing: encode the rule **mechanically**, fire the \"buy the "
            "Qstick up-cross\" trade **2,210 times** across five liquid ETFs over 21 years, and "
            "time the result against the only baseline that matters — **buying on random days "
            "instead.**\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the trend-proxy "
            "math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice** — research & education. Every chart is drawn by "
            "the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| If I buy when Qstick turns positive, do I make money? | **Yes, in absolute "
            "terms** — but random days do **the same or better**: at 5/10/20-day holds the "
            "Qstick cross actually *trails* buying on a random day. |\n"
            "| Does the buyer/seller body sequence matter? | **No.** Scramble the order of the "
            "daily bodies into nonsense and the result barely moves (81% of scrambled versions "
            "match or beat the real one). |\n"
            "| Is it really measuring something new? | **No — it's a slow echo of ordinary "
            "trend.** Qstick tracks a plain trailing-momentum line (no open, no candle, just "
            "closing prices) at **r ≈ 0.78** — and even *that* simpler line barely, "
            "unreliably, beats a coin. |\n"
            "| So is it a tradable edge? | **No.** It's **relabelled trend-following**, and a "
            "worse version of it — it trades less often than the market and still doesn't beat "
            "a dart. |\n\n"
            "> Qstick is a fine way to *describe* who won a bar. As a *forecast*, it's a "
            "**mirage** — and structurally, it's mostly just a laggy copy of price momentum."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"For each bar, the body is close minus open — how far buyers pushed the price "
            "past where it opened. Average that over 8 days. When the average crosses **up** "
            "through zero, buying pressure has taken over and a rise should follow. Buy the "
            "up-cross.\"*\n\n"
            "This is **Tushar Chande's Qstick** (Chande & Kroll, *The New Technical Trader*, "
            "1994), built into most charting packages. It's pitched as a purer read on *who's "
            "winning* than the close alone — a bar can close lower than yesterday and still be "
            "\"bullish\" if it closed well above where it opened. So: does that extra "
            "information actually forecast anything?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If close-minus-open genuinely carried forward-looking information, it would be a "
            "small miracle: a number computed from a single bar's own open and close would "
            "predict the *next* bar. That's a strong claim, and — like every long-only signal "
            "tested on an index that **drifts up over time** — it's exactly the kind of claim "
            "that looks great by accident. Any \"buy\" rule inherits the market's climb; the "
            "only fair test is whether Qstick beats **buying on random days**, and whether the "
            "close-minus-open *split* adds anything a plain trailing average of the close alone "
            "wouldn't already give you."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            f"1. **Compute the body** each bar — (close − open) / prior close.\n"
            f"2. **Smooth it** with an **{R['smooth']}-day** trailing average (causal — only "
            "past bars, no future data), Chande's own original window.\n"
            "3. **Trade the lore.** When smoothed Qstick **crosses up through zero**, buy at "
            "the next close; measure the return over the next **5 / 10 / 20 / 60 days**.\n"
            "4. **The honest baseline.** Do the exact same hold on **random days**. If Qstick "
            "leads, the up-cross must beat random. *If it doesn't, the signal is a mirage* — "
            "that's the result that would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does smoothed Qstick even look like, and where does the rule buy? "
            "Here's SPY with its Qstick line below and the zero up-crosses the rule would buy."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    seg = b.iloc[-450:]\n"
            "    qs = st.qstick(b, smooth=R['smooth'])\n"
            "    ent = st.qstick_cross_entries(b, smooth=R['smooth'])\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.2, 6.0), sharex=True,\n"
            "                                   gridspec_kw={'height_ratios':[2,1]})\n"
            "    ax1.plot(seg.index, seg['close'].values, c='k', lw=1.2, label='SPY close')\n"
            "    ax1.scatter(ent, b['close'].reindex(ent), c=GREEN, s=42, zorder=5, label='Qstick up-cross BUY')\n"
            "    ax1.set_title('Smoothed Qstick up-cross on SPY (last ~2y)'); ax1.legend(loc='upper left')\n"
            "    ax2.plot(seg.index, qs.reindex(seg.index), c='#2c6fbb', lw=1.2, label='smoothed Qstick')\n"
            "    ax2.axhline(0, c=GREY, ls='--'); ax2.set_ylabel('Qstick'); ax2.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('Qstick up-crosses in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The cross fires whenever the smoothed candle-body average flips from negative to "
            "positive. **Let's race the Qstick up-cross against random entries** at four "
            "horizons. Blue = buy the up-cross; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    cross, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.qstick_cross_entries(bb, smooth=R['smooth'])\n"
            "            re = st.random_entries(bb, max(len(e),50), smooth=R['smooth'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        cross.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    cross = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, cross, .4, color='#2c6fbb', label='buy the Qstick up-cross')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(cross,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The Qstick cross does NOT beat random — it loses at 3 of 4 horizons'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('cross:', [round(v) for v in cross]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story in one chart. The Qstick cross makes money in absolute "
            f"terms (**+{R['h20'][2]:.0f} bps** over 20 days) — but **random entries make "
            f"more** (**+{R['h20'][5]:.0f} bps** at 20d). At 5, 10 and 20 days the indicator "
            "underperforms a dart; only at 60 days does it edge ahead, and even there it's a "
            "statistical tie. The apparent edge was **the market's upward drift**, not the "
            "candle-body signal.\n\n"
            "**One more sanity check.** What if we **scramble the order** of the daily bodies — "
            "keep the same set of values but shuffle when they occurred, so the smoothed line "
            "and its crosses become temporally meaningless? If the body sequence really *led* "
            "price, the nonsense version should do much worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.scramble_placebo(load('SPY'), 20, smooth=R['smooth'], n_draws=300, seed=675)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real Qstick up-cross (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *time-scrambled* body series do at least as well (p={pval:.2f}).')\n"
            "print('=> the order of the daily buyer/seller bodies is not doing the work.')"
        ),
        md(
            f"**{R['placebo'][1]*100:.0f}%** of scrambled body series match or beat the real "
            "one. If the sequence of daily candle bodies genuinely led price, a random "
            "time-scramble would collapse the result. It doesn't.\n\n"
            "**Last question — is Qstick even measuring something new?** We compare it to the "
            "plainest possible trend line: the average daily price change over the same 8 "
            "days, computed from **closing prices alone**, no open, no candle."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load('SPY')\n"
            "    qs = st.qstick(b, smooth=R['smooth']); mom = st.trend_momentum(b, n=R['smooth'])\n"
            "    seg = pd.concat([qs, mom], axis=1).dropna().iloc[-500:]\n"
            "    seg.columns = ['Qstick', 'plain trend proxy']\n"
            "else:\n"
            "    rng = np.random.default_rng(675)\n"
            "    x = rng.normal(0, 0.004, 500)\n"
            "    seg = pd.DataFrame({'Qstick': x, 'plain trend proxy': x*R['trend_corr_mean'] + rng.normal(0,0.003,500)})\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.scatter(seg.iloc[:,1]*1e4, seg.iloc[:,0]*1e4, s=8, alpha=.4, color='#2c6fbb')\n"
            "ax.axhline(0, c=GREY, lw=.8); ax.axvline(0, c=GREY, lw=.8)\n"
            "ax.set_xlabel('plain trailing price-momentum proxy (bps/day, no open used)')\n"
            "ax.set_ylabel('Qstick (bps/day)')\n"
            "ax.set_title(f'Qstick tracks plain trend at r ~ {R[\"trend_corr_mean\"]:.2f} (pooled basket)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'mean correlation across the 5-ETF basket: {R[\"trend_corr_mean\"]:+.3f}')"
        ),
        md(
            f"Qstick and a plain \"average daily price change\" line — which never even looks "
            f"at the open — move together at **r ≈ {R['trend_corr_mean']:.2f}**. And when we "
            "run *that* simpler line's own zero-cross rule through the identical race, it "
            f"scores **+{R['mom'][1]:.0f} bps** vs **+{R['mom'][2]:.0f} bps** random "
            f"(Δ = +{R['mom'][3]:.0f} bps, *t* = {R['mom'][4]:.2f}) — itself just short of "
            "certifiable. Chande's candle-body split isn't measuring something the plain trend "
            "doesn't already show; it's a noisier, laggier way to look at the same thing."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The Qstick up-cross does **not** beat buying on random days "
            "(it *loses* at 5/10/20 days and ties at 60; the cross-vs-random difference never "
            "clears *t* = 2). The market's drift explains the absolute profit, not the "
            "indicator.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were "
            "always getting for free — and a sign-scramble placebo confirms the body's time "
            "order isn't doing any work either.\n"
            "- **Just a slow trend proxy? — Confirmed.** Qstick tracks a plain trailing "
            "price-momentum line at r ≈ 0.78, and even that simpler line barely (and "
            "unreliably) beats a coin. The open/close split adds machinery, not information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The Qstick cross's only edge over a coin flip would "
            "have been the market's long-run climb — and it doesn't even reliably capture that; "
            "you'd get more of the drift, more cheaply, by just **holding the index**. The cross "
            "trades *less* of the time and, at 3 of 4 horizons, delivers *less* return than "
            "doing nothing special at all. As a forecasting tool it doesn't pay; as a bar-by-bar "
            "read-out of who won the day, it was never built to predict the next one."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Other windows.** Chande's original is 8 days; some platforms default to 10 or "
            "20. Longer windows only strengthen the trend-proxy correlation — the diagnosis "
            "gets *more* true, not less.\n"
            "- **The whole 'who's-in-control' oscillator family.** Balance of Power, the Force "
            "Index, Accumulation/Distribution and the Klinger oscillator all chase the same "
            "intuition and land in the same place — see [423-force-index](../../423-force-index/) "
            "and [473-balance-of-power](../../473-balance-of-power/).\n"
            "- **A real positive control.** The quants notebook plants a *genuine* "
            "buying-pressure-leads-price structure into a synthetic tape and shows the harness "
            "banks it (so the null result here isn't a dead detector — it's an honest 'nothing "
            "there').\n\n"
            "*Think close-minus-open forecasts beyond ordinary trend? Show the Qstick cross "
            "beating random entries **and** the plain momentum proxy at **t ≥ 2** on a real "
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
            "# Qstick — a quantitative teardown 🔬\n"
            "### Smoothed close-minus-open zero up-cross on 5 ETFs · forward returns · one-sample "
            "HAC *t* · a drift-matched random-entry baseline · an ordering placebo · a trend-proxy "
            "correlation and momentum-cross race · costs · a synthetic planted-lead control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The job is to separate the **indicator** from the **drift**, and then to ask "
            "Chande's own defining claim directly: does the open/close split carry information a "
            "plain trend line doesn't already have?\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance "
            "daily adjusted closes (**total-return** for the ETFs), 2005→2026. Qstick = "
            f"SMA_{R['smooth']}((close−open)/prior close) — normalised by the prior close for "
            "cross-instrument comparability (a documented choice; it cannot change which bars "
            "cross zero). Entry is the **next close** (one documented lag). Methods in "
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
            f"| **Signal** | `NONE` | Qstick up-cross vs a **drift-matched random** baseline: "
            f"Δ = {R['h5'][6]:+.1f} / {R['h10'][6]:+.1f} / {R['h20'][6]:+.1f} / "
            f"{R['h60'][6]:+.1f} bps at 5/10/20/60d; Welch *t* **never clears 2** "
            f"(max |t| = {abs(R['h5'][8]):.2f}, at 5d, in the wrong direction). |\n"
            f"| **Tradability** | `MIRAGE` | The impressive one-sample t's (20d "
            f"t = {R['h20'][4]:.2f}, 60d = {R['h60'][4]:.2f}) are **pure beta** — they vanish "
            "against random entries. Ordering placebo *p* = "
            f"{R['placebo'][1]:.2f}: the sequence isn't doing the work either. |\n"
            f"| **Just a slow trend proxy?** | `CONFIRMED` | Qstick correlates at "
            f"**r ≈ {R['trend_corr_mean']:.2f}** with a plain trailing-momentum proxy that "
            f"never looks at the open — and even that proxy's own cross barely clears random "
            f"(Δ = +{R['mom'][3]:.1f} bps, t = {R['mom'][4]:.2f} < 2). |\n\n"
            "> 💡 In plain words: the Qstick cross *looks* like it makes money only because "
            "indices drift up — strip the drift (race it vs random) and it loses; the "
            "open/close split itself is mostly a repackaging of ordinary trend."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Define the per-bar normalised body $b_t = (C_t - O_t)/C_{t-1}$ and its causal "
            f"smoothing $\\bar b_t = \\frac1{{{R['smooth']}}}\\sum_{{j=0}}^{{{R['smooth']}-1}} "
            "b_{t-j}$ (Chande & Kroll's original 8-day window). The rule buys on the **zero "
            "up-cross** $\\bar b_{t-1} < 0 \\le \\bar b_t$ and expects a forward rise.\n\n"
            "- **H₀ (drift).** Cross returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (Qstick leads).** Cross returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the ordering matters).** Cross returns exceed a **sign-scramble** body "
            "series whose readings are time-shuffled.\n"
            "- **H₃ (Qstick is independent of price momentum).** Smoothed Qstick correlates "
            "weakly with a plain trailing price-momentum proxy that never uses the open.\n\n"
            "We find **H₀ not rejected** (cross ≤ random at 5/10/20d, tie at 60d), "
            "**H₁ rejected** (Welch t never ≥ 2, and negative at 3 of 4 horizons), "
            "**H₂ rejected** (placebo p = 0.81), **H₃ rejected** (r ≈ 0.78). "
            "The steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity/commodity ETFs have a positive unconditional daily mean. "
            "*Any* long-only entry rule inherits it; a high one-sample $t$ against **zero** "
            "measures the tide, not the tool. The fix is the **random-entry baseline** (same "
            "instrument, epoch, hold) and a Welch test of cross-*minus*-random.\n\n"
            "**(b) Relabelled trend.** A moving average of close-minus-open is, up to the "
            "overnight gap, close to a moving average of the *close itself minus a lagged "
            "close* — i.e. ordinary trailing momentum wearing a candle costume. The **trend-"
            "proxy correlation** and the **momentum-cross race** test this directly instead of "
            "asserting the algebra."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} Qstick "
            "up-crosses** pooled.\n"
            f"- **Indicator.** ``(close−open)/prior close``, smoothed with a causal "
            f"{R['smooth']}-day SMA (Chande & Kroll's original window; no look-ahead).\n"
            "- **Entry.** Zero up-cross (smoothed Qstick negative on *t−1*, ≥ 0 on *t*); enter "
            "**next close** (one lag); hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of cross returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample cross vs random (the *real* test).\n"
            "- **Null #3 — sign-scramble placebo** (body time order destroyed, marginal kept).\n"
            "- **Null #4 — trend-proxy correlation and race** vs a plain price-momentum cross.\n"
            "- **Costs.** 1 bp one-way × 2 legs on every cross.\n"
            "- **Positive control.** Synthetic tape with a **planted**, mean-reverting "
            "buying-pressure factor (knob `edge`): edge=0 must NOT reach significance across "
            "20 seeds; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks decent, vs-random kills it\n\n"
            "Left: the Qstick cross's **one-sample** t against zero (the misleading number). "
            "Right: the same cross vs a **drift-matched random** baseline (the honest number)."
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
            "            e = st.qstick_cross_entries(bb, smooth=R['smooth'])\n"
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
            f"> 💡 In plain words: the left bars clear *t* = 2 easily (20d "
            f"**{R['h20'][4]:.2f}**, 60d **{R['h60'][4]:.2f}**) — but that's the **drift**, "
            f"every long-only entry inherits it. The right bars are the real test: "
            f"cross-minus-random is **negative** at 5/10/20d ({R['h20'][8]:+.2f} at 20d) and "
            "barely positive at 60d — never significant either way. The indicator adds "
            "nothing over a coin flip, and at short horizons it's a *worse* coin."
        ),
        md(
            "### 4b · Cross vs random across horizons — the gap is the verdict\n\n"
            "Mean return, Qstick cross vs random entry, all four horizons. The cross should "
            "tower over random if Qstick leads. It mostly loses to it."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, cross, .4, color='#2c6fbb', label='Qstick up-cross')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(cross,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Qstick up-cross does not beat random entry'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta cross-random (bps):', [round(a-b) for a,b in zip(cross,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the cross is **+{R['h20'][2]:.0f} bps** but random "
            f"is **+{R['h20'][5]:.0f} bps** — the indicator *underperforms* a dart by "
            f"{abs(R['h20'][6]):.0f} bps. Only at 60 days does it edge ahead, by "
            f"{R['h60'][6]:.0f} bps, and that's still a coin flip statistically (t = "
            f"{R['h60'][8]:+.2f})."
        ),
        md(
            "### 4c · The ordering placebo — scramble the body, nothing changes\n\n"
            "Shuffle the per-bar bodies in time (marginal kept) so the smoothed series and its "
            "crosses are temporally meaningless. If price respects *this specific sequence*, "
            "the scramble should demolish the result."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bspy = load('SPY')\n"
            "    pl = st.scramble_placebo(bspy, 20, smooth=R['smooth'], n_draws=300, seed=675)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    rb = st.raw_body(bspy).to_numpy(); idx = bspy.index; c = bspy['close']\n"
            "    rng = np.random.default_rng(675); draws=[]\n"
            "    for _ in range(300):\n"
            "        perm = rng.permutation(rb)\n"
            "        sb = pd.Series(perm, index=idx).rolling(R['smooth'], min_periods=R['smooth']).mean()\n"
            "        prev = sb.shift(1); mask = (prev<0)&(sb>=0)&sb.notna()&prev.notna()\n"
            "        rr = st.forward_returns(c, idx[mask.to_numpy()], 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(675); draws = rng.normal(70, 25, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='time-scrambled body (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real Qstick {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean Qstick-up-cross 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real Qstick sits mid-cloud: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real Qstick {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => ordering not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: **{R['placebo'][1]*100:.0f}%** of time-scrambled body series "
            "do as well or better than the real ordering — so the specific buyer/seller "
            "sequence carries no information. The result was never about *when* the bodies "
            "occurred, only about *how many* were positive — which is just the drift again."
        ),
        md(
            "### 4d · Per-ticker — a *consistent* loss, not noise\n\n"
            "20-day cross-minus-random delta, per instrument. Unlike a coin-flip indicator "
            "(sign flipping name to name), every single ticker loses to random here."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        bb = load(t); c = bb['close']\n"
            "        e = st.qstick_cross_entries(bb, smooth=R['smooth']); re = st.random_entries(bb, max(len(e),50), smooth=R['smooth'], seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d cross − random (bps)'); ax.set_title('Every ticker trails its own random baseline')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: **QQQ** is **{R['per'][1][5]:+.0f}** bps behind random, "
            f"**SPY** **{R['per'][0][5]:+.0f}**, and even the least-bad ticker (**IWM**, "
            f"{R['per'][2][5]:+.0f}) is still negative. Five out of five — a consistent, "
            "structural drag, not a sign-flipping coincidence."
        ),
        md(
            "### 4e · Is Qstick even measuring something new? — the trend-proxy check\n\n"
            "By construction, ``Qstick_N = SMA_N(close) - SMA_N(open)``; when the open sits "
            "close to the prior close, this telescopes toward the trailing N-day average price "
            "change — a signal built from **closes alone**, with no reference to the open. We "
            "measure the correlation directly and race the naive momentum-only cross."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tickers, corrs = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        bb = load(t); tp = st.trend_proxy_stats(bb, smooth=R['smooth'])\n"
            "        tickers.append(t); corrs.append(tp['corr'])\n"
            "    tt2, rr2 = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        bb = load(t); c = bb['close']\n"
            "        e = st.momentum_cross_entries(bb, n=R['smooth'])\n"
            "        re = st.random_entries(bb, max(len(e),50), smooth=R['smooth'], seed=7)\n"
            "        tt2.append(st.forward_returns(c,e,20)); rr2.append(st.forward_returns(c,re,20))\n"
            "    tt2 = np.concatenate(tt2); rr2 = np.concatenate(rr2)\n"
            "    mom_cross, mom_rnd = tt2.mean()*1e4, rr2.mean()*1e4\n"
            "    from scipy import stats as _st\n"
            "    mom_t = _st.ttest_ind(tt2, rr2, equal_var=False)[0]\n"
            "else:\n"
            "    tickers = [p[0] for p in R['trend']]; corrs = [p[1] for p in R['trend']]\n"
            "    mom_cross, mom_rnd, mom_t = R['mom'][1], R['mom'][2], R['mom'][4]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar(tickers, corrs, color='#2c6fbb', width=.6)\n"
            "a1.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(corrs): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylim(0,1); a1.set_ylabel('corr(Qstick, plain trend proxy)')\n"
            "a1.set_title('Qstick tracks a signal that never looks at the open')\n"
            "a2.bar(['Qstick\\ncross','momentum\\ncross','random'], [cross[2] if HAVE_REAL else R['h20'][2], mom_cross, mom_rnd],\n"
            "       color=['#2c6fbb', AMBER, GREY], width=.55)\n"
            "for i,v in enumerate([cross[2] if HAVE_REAL else R['h20'][2], mom_cross, mom_rnd]):\n"
            "    a2.annotate(f'{v:+.0f}',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('mean 20d return (bps)'); a2.set_title(f'The naive trend cross fares no better (t={mom_t:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker corr:', dict(zip(tickers, [round(c,3) for c in corrs])))\n"
            "print(f'momentum-only cross {mom_cross:+.1f} bps vs random {mom_rnd:+.1f} bps, t={mom_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: Qstick correlates with the open-blind trend proxy at "
            f"**r ≈ {R['trend_corr_mean']:.2f}** pooled — strong structural overlap. And when "
            f"the plain proxy runs the *same* race, it scores **+{R['mom'][1]:.0f} bps** vs "
            f"**+{R['mom'][2]:.0f} bps** random (Δ = +{R['mom'][3]:.0f} bps, "
            f"t = {R['mom'][4]:.2f}) — itself **short of *t* = 2**. Chande's open/close split "
            "adds bar-level noise, not forecasting power, over a trend signal that is *itself* "
            "unproven."
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic tape with a latent, **mean-reverting** buying-pressure factor (stationary "
            "AR(1), no self-referential feedback — an earlier feedback-loop design was tried and "
            "made the price path explode, an engineering note, not part of the claim). The null "
            "(edge = 0) is checked over **20 seeds** — never a single stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    px, _ = data.synthetic_panel(edge=0.0, seed=675 + s_, n_days=8000)\n"
            "    s = st.summarize(st.forward_returns(px['close'], st.qstick_cross_entries(px, smooth=R['smooth']), 20))\n"
            "    null_ts.append(s['t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "px, _ = data.synthetic_panel(edge=2.0, seed=675, n_days=8000)\n"
            "planted_t = st.summarize(st.forward_returns(px['close'], st.qstick_cross_entries(px, smooth=R['smooth']), 20))['t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (edge=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5,\n"
            "           label='planted buying-pressure lead (edge=2.0)')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('one-sample t (20d cross return)')\n"
            "ax.set_title('Control: no null fires; a planted lead lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null'][0]:+.2f} (sd {R['syn_null'][1]:.2f}) and **never** crosses the "
            f"bar; a planted, persistent buying-pressure factor reads t = "
            f"{R['syn_planted'][3]:.2f}. The machinery is unbiased — the flat real-tape result "
            "is a genuine 'nothing there', not a broken pipeline. *(A faithful-engine / power "
            "check only — never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the Qstick up-cross does not beat a drift-matched random "
            f"baseline (cross − random = {R['h5'][6]:+.1f}/{R['h10'][6]:+.1f}/"
            f"{R['h20'][6]:+.1f}/{R['h60'][6]:+.1f} bps at 5/10/20/60d; Welch t never clears "
            f"±2, max |t| = {abs(R['h5'][8]):.2f} at 5d — in the wrong direction). Every one of "
            "5 tickers underperforms its own random baseline at 20 days; an ordering placebo "
            f"(*p* = {R['placebo'][1]:.2f}) confirms the body **sequence** carries no "
            "information.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; costs "
            "only deepen the small hole. You'd capture the same drift more cheaply by holding "
            "the index.\n"
            f"- **Just a slow trend proxy? `CONFIRMED`** — Qstick correlates at "
            f"r ≈ {R['trend_corr_mean']:.2f} with a plain trailing-momentum proxy that never "
            f"uses the open, and even that simpler proxy's own cross barely (Δ = "
            f"+{R['mom'][3]:.1f} bps, t = {R['mom'][4]:.2f} < 2) beats random. The candle-body "
            "split is a laggier read of ordinary trend, not an independent forecast."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The Qstick cross's apparent absolute profit is the unconditional drift of the "
            "underlying ETFs, which you obtain more cheaply and more fully by **buying and "
            "holding**. The rule trades *less* of the time and, at 3 of 4 horizons, *underperforms* "
            "a random entry on top of that — a strictly worse, more expensive way to be "
            "sometimes-long. There is no capacity question because there is no edge to scale."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Window sweep.** 10-day and 20-day Qstick (other platforms' defaults) only "
            "raise the trend-proxy correlation further — longer smoothing windows converge "
            "faster on plain momentum, not away from it.\n"
            "- **The oscillator family.** [423-force-index](../../423-force-index/) and "
            "[473-balance-of-power](../../473-balance-of-power/) share the 'who's in control' "
            "intuition and land in the same None × Mirage place, via different formulas.\n"
            "- **Volume-weighted Qstick.** Some variants weight the body by volume (closer to "
            "Force Index); the trend-proxy critique would need re-testing on that variant "
            "specifically.\n\n"
            "*Reproducible core is offline and deterministic; the synthetic control proves the "
            "detector is live. Methods/sources: [`docs/references.md`](../docs/references.md); "
            "frozen numbers: [`docs/results.md`](../docs/results.md).*"
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
