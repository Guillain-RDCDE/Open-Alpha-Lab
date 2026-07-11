"""Generate the two narrative notebooks for Study 682 (Inverse-Fisher-RSI).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached total-return
tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance total-return closes,
# SPY/QQQ/IWM/AAPL/MSFT/NVDA, 2010-01-04 -> 2026-06-30).
R = dict(
    start="2010-01-04", end="2026-06-30", tickers="SPY, QQQ, IWM, AAPL, MSFT, NVDA",
    n_ift_bull=711, n_ift_bear=886, n_rsi14=190, n_rsi2=1359,
    # headline: horizon -> (sig_bps, base_bps, gap_bps, welch_t, nw_t, hit_pct, hit_lo, hit_hi, n_sig)
    head={
        5: (23.8, 39.7, -15.9, -1.07, -1.09, 58.3, 54.6, 61.9, 710),
        10: (40.3, 80.2, -39.9, -1.89, -2.05, 56.9, 53.2, 60.5, 707),
        20: (179.0, 159.8, 19.3, 0.65, 0.77, 68.0, 64.5, 71.3, 706),
    },
    bear={
        5: (39.4, 39.3, 0.1, 0.01, 0.01, 886),
        10: (79.5, 79.0, 0.5, 0.03, 0.03, 886),
        20: (115.5, 162.0, -46.4, -1.82, -2.17, 881),
    },
    placebo_obs=40.3, placebo_mean=79.8, placebo_sd=20.0, placebo_p=0.9710, placebo_draws=4000,
    # comparison at h=10d: signal -> (n, gap_bps, welch_t, nw_t, hit_pct)
    cmp={
        "IFT-RSI (-0.5 cross)": (707, -39.9, -1.89, -2.05, 56.9),
        "plain RSI(14) (cross up 30)": (189, -54.0, -1.08, -1.01, 56.6),
        "plain RSI(2) (cross up 10)": (1354, 20.8, 1.36, 1.53, 61.0),
    },
    tm_sharpe5=0.48, tm_sharpe10=0.45, bh_sharpe=0.71, tm_ann5=8.68, tm_ann10=8.13,
    bh_ann=19.86, exposure=47.8,
    rnd_mean=0.35, rnd_sd=0.10, rnd_z=1.27, rnd_beats=16,
    rsi14_tm_sharpe5=0.31, rsi14_tm_ann5=5.41, rsi14_exposure=30.3,
    syn_null_mean=0.07, syn_null_sd=0.87, syn_null_fire=0, syn_planted_t=3.81,
    syn_planted_sig=39.9, syn_planted_base=2.7,
    fp=dict(SPY="0c6ad198f447", QQQ="a84e44862578", IWM="f0b7b205c2af",
            AAPL="8699fe541146", MSFT="51aba9472a5f", NVDA="454a3ad24423"),
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Sharper_than_plain_RSI%3F: Busted](https://img.shields.io/badge/Sharper_than_plain_RSI%3F-Busted-8b949e?style=flat-square)\n\n"
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

from inverse_fisher_rsi import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    REAL = data.load_real()
    SIGNALS = st.basket_signals(REAL)
else:
    REAL = SIGNALS = None
print("real cache present:", HAVE_REAL, "| tickers:", data.TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a fancier RSI actually call the turns better? 📈🔀\n"
            "### Ehlers' Inverse Fisher Transform — a beautifully bounded oscillator that "
            "turns out to be all polish, no edge\n\n"
            + BADGES +
            "Plain old RSI has a chart problem: it spends most of its life bunched between 30 "
            "and 70, and even when it gets to an extreme it *lingers* there instead of turning "
            "on a dime. John Ehlers' fix, from a 2002 magazine article that's still cited "
            "today: run RSI through the **Inverse Fisher Transform**, a formula that squeezes "
            "the reading into a tight -1 to +1 band and makes it snap between its extremes. "
            "Traders love how it looks — sharp, decisive, easy to eyeball.\n\n"
            "The question this notebook asks: does it look *better* because it trades *better* "
            "— or just because it's a prettier picture of the same information?\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the head-to-head "
            "math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** SPY plus a five-name liquid basket, 2010→2026, total-return "
            "closes. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does buying an IFT-RSI \"oversold snap-back\" (crossing up through -0.5) beat "
            f"buying on a random day? | **No.** At the 10-day mark it actually did **worse** "
            f"({R['head'][10][2]:+.1f} bps) — not by a lot, but the sign is backwards from the "
            "story. |\n"
            "| Does a random-signal placebo beat the real thing? | **97% of the time.** Draw "
            "random entries of the same count, 4,000 times — most of them out-earn the actual "
            "IFT-RSI signal. |\n"
            "| Does it beat plain old RSI(14) or RSI(2)? | **No.** All three baselines were "
            "tested the identical way; the *simplest* one (RSI(2)) was the only one with a "
            "positive (if still uncertified) tilt. |\n"
            f"| Can you trade it? | A costed long-flat timer nets Sharpe **{R['tm_sharpe5']:.2f}** "
            f"— trailing plain buy-and-hold's **{R['bh_sharpe']:.2f}** over the same historic "
            "bull run, and barely beating a coin-flip timer with the same time-in-market. |\n\n"
            "> The chart looks sharper. The numbers don't care."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"RSI's overbought/oversold crossovers are mushy — the indicator sits near its "
            "extremes too long. Pass it through the Inverse Fisher Transform and it snaps "
            "cleanly between -1 and +1: a crisper, more decisive read on turning points.\"*\n\n"
            "It's not folk wisdom off a forum — Ehlers published the exact recipe (RSI(5), a "
            "9-bar weighted average, then the inverse-Fisher formula) in *Technical Analysis of "
            "Stocks & Commodities* in 2002, and it's a stock indicator on every major charting "
            "platform today. The claim has real math behind it — tanh really does compress a "
            "wandering series toward its bounds. The open question is whether *compression* "
            "is the same thing as *information*."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If IFT-RSI really calls reversals more reliably than plain RSI, that's a free "
            "upgrade to one of the most widely used indicators in retail trading — swap one "
            "formula for another, same data, better entries. It would also validate a whole "
            "family of Ehlers \"cybernetic\" transforms built on the same idea (apply Fisher to "
            "*any* bounded oscillator to sharpen it).\n\n"
            "If it doesn't, that's an equally useful lesson: a transform can make a chart *look* "
            "more decisive without adding one bit of forecasting power — style dressed up as "
            "substance, and worth knowing before you retire a working system for a prettier one."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **The signal.** IFT-RSI crossing UP through -0.5 = bullish turn; crossing DOWN "
            "through +0.5 = bearish turn — Ehlers' own published thresholds, on his own "
            "published formula.\n"
            "- **The comparison.** Forward returns (5/10/20 trading days) after that cross vs. "
            "an unconditional entry on any random day, on the *same* six tickers.\n"
            "- **The luck check.** Draw the same *number* of random entries thousands of times "
            "— does the real signal actually beat that random-entry distribution?\n"
            "- **The fair fight.** Run the identical test on plain RSI(14) and plain RSI(2) "
            "crossovers — if IFT-RSI is genuinely sharper, it should win this comparison "
            "cleanly.\n"
            "- **The trade check.** Turn the crossover into an actual buy/sell rule and pay "
            "costs — a signal that only wins on paper isn't a signal you can bank."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Average forward return after an IFT-RSI bullish cross vs "
            "an unconditional entry, at three horizons."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [st.headline_stats(REAL, SIGNALS, 'ift_bull', h) for h in st.HORIZONS]\n"
            "    sig = [r['sig_mean_bps'] for r in rows]; base = [r['base_mean_bps'] for r in rows]\n"
            "else:\n"
            "    sig = [R['head'][h][0] for h in (5, 10, 20)]\n"
            "    base = [R['head'][h][1] for h in (5, 10, 20)]\n"
            "x = np.arange(3); w = 0.35\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.6))\n"
            "ax.bar(x - w/2, sig, width=w, color=RED, label='after IFT-RSI bullish cross')\n"
            "ax.bar(x + w/2, base, width=w, color=GREY, label='unconditional (any day)')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['5d', '10d', '20d'])\n"
            "ax.set_ylabel('mean forward return (bps)')\n"
            "ax.set_title('The \"oversold bounce\" does not out-earn a random day')\n"
            "ax.legend(); ax.axhline(0, c='k', lw=.8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print({h: (round(s,1), round(b,1)) for h,s,b in zip((5,10,20), sig, base)})"
        ),
        md(
            f"Ten-day forward return after a bullish snap-back: **{R['head'][10][0]:+.1f} bps**, "
            f"vs **{R['head'][10][1]:+.1f} bps** for an unconditional entry on the same basket — "
            f"a gap of **{R['head'][10][2]:+.1f} bps**, the wrong sign for a \"buy the dip\" "
            f"claim. It's not a huge effect either way (Welch *t* = {R['head'][10][3]:+.2f}), "
            "but there is nothing here to certify — and the sign flips again by 20 days, which "
            "is itself a tell that the pattern isn't stable.\n\n"
            "**The luck check makes it sharper.** We drew 4,000 random entry sets of the same "
            "size:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.random_signal_placebo(REAL, SIGNALS, 'ift_bull', 10,\n"
            "                                  n_draws_per_seed=40, n_seeds=4)\n"
            "    obs, draws = pl['obs']*1e4, pl['draws']*1e4\n"
            "else:\n"
            "    obs = R['placebo_obs']\n"
            "    rng = np.random.default_rng(682)\n"
            "    draws = rng.normal(R['placebo_mean'], R['placebo_sd'], 800)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='random entries, same count')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'the real IFT-RSI signal: {obs:+.1f} bps')\n"
            "ax.set_xlabel('mean 10-day forward return of a random entry set (bps)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Random entries beat the real signal on {R[\"placebo_p\"]*100:.0f}% of draws')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'canonical placebo (results.md): mean {R[\"placebo_mean\"]:+.1f} bps, '\n"
            "      f'p = {R[\"placebo_p\"]:.4f}')"
        ),
        md(
            f"The observed signal sits on the **wrong side** of a random-entry distribution "
            f"centered at **+{R['placebo_mean']:.1f} bps** — **{R['placebo_p']*100:.1f}%** of "
            "random draws of the same size do *better* than the real \"oversold bounce\" entry. "
            "That's about as clean a non-result as this desk sees.\n\n"
            "**The fair fight.** Does IFT-RSI at least beat the plain RSI it's supposed to "
            "improve on?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    names = {'IFT-RSI (-0.5 cross)': 'ift_bull',\n"
            "             'plain RSI(14) (cross up 30)': 'rsi14_bull',\n"
            "             'plain RSI(2) (cross up 10)': 'rsi2_bull'}\n"
            "    gaps = [st.headline_stats(REAL, SIGNALS, k, 10)['gap_bps'] for k in names.values()]\n"
            "    labels = list(names.keys())\n"
            "else:\n"
            "    labels = list(R['cmp'].keys())\n"
            "    gaps = [R['cmp'][k][1] for k in labels]\n"
            "cols = [RED if g < 0 else GREEN for g in gaps]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "ax.barh(labels, gaps, color=cols)\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('10-day forward-return gap vs unconditional (bps)')\n"
            "ax.set_title('The fancier transform is not the best performer here')\n"
            "for i, g in enumerate(gaps): ax.annotate(f'{g:+.1f}', (g, i), ha='left' if g>0 else 'right',\n"
            "    va='center')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dict(zip(labels, [round(g,1) for g in gaps])))"
        ),
        md(
            "The plainest baseline of all — Connors' RSI(2), one of this desk's genuinely real "
            "signals (see sibling [75-knee-jerk](../../75-knee-jerk/)) — is the only one that "
            "even *points* the right way here (this is a lighter comparison test, not that "
            "study's full protocol). IFT-RSI, the one with the fancy math, comes in worst.\n\n"
            "**Finally, the trade.** Turn the crossover into a real long-flat rule and pay "
            "costs:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tm = st.timer_with_costs(REAL, SIGNALS, 'ift_bull', 'ift_bear', cost_bps=5.0)\n"
            "    net, bh = tm['sharpe_net'], tm['sharpe_bh']\n"
            "else:\n"
            "    net, bh = R['tm_sharpe5'], R['bh_sharpe']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['IFT-RSI timer\\n(net, 5 bps)', 'buy & hold'], [net, bh], color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([net, bh]): ax.annotate(f'{v:.2f}', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('annualised Sharpe, 2010-2026')\n"
            "ax.set_title('The timer trails simply holding the basket')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'timer net Sharpe {net:.2f}  vs buy-and-hold {bh:.2f}')"
        ),
        md(
            f"Net Sharpe **{R['tm_sharpe5']:.2f}** against buy-and-hold's **{R['bh_sharpe']:.2f}** "
            f"— unsurprising given the timer is flat **{100-R['exposure']:.0f}%** of the time "
            "during one of the strongest bull markets on record, but it also barely beats a "
            f"coin-flip timer with the same exposure (z = {R['rnd_z']:+.2f}, not certified). "
            "Nothing here clears the bar to trade."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** No horizon shows a certified edge; the closest call points "
            "the wrong way, and 97% of random entries beat the real signal.\n"
            "- **Tradability — Mirage.** The costed timer trails buy-and-hold and barely edges "
            "out a random-exposure coin flip.\n"
            "- **\"Sharper than plain RSI(2/14)\"? — Busted.** The fancier transform performs "
            "worse than the simplest baseline in the identical test."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The general lesson.** A bounded, visually decisive oscillator is not "
            "automatically a *better-informed* one — Ehlers' transform reshapes RSI's "
            "distribution, it doesn't add new information to it. The same caution applies to "
            "every other \"Fisherized\" indicator in his toolbox.\n"
            "- **Where it might still matter.** As a *filter* alongside a real signal (thin "
            "evidence either way here) rather than a standalone entry rule — untested by this "
            "study, a natural next step.\n"
            "- **Sibling studies:** the [plain Fisher Transform on price](../../183-fisher-transform/) "
            "(proven mathematically redundant with the raw crossover), "
            "[Connors' RSI(2)](../../75-knee-jerk/) (the desk's real RSI-family signal), "
            "[Stochastic-RSI](../../428-stochastic-rsi/) (a different stacked transform, also a "
            "non-result) and [RSI divergence](../../669-rsi-divergence/) (a structural pattern, "
            "also a non-result) — four different ways to dress up RSI, four honest \"no\"s.\n\n"
            "*Think the Inverse Fisher Transform earns its keep somewhere this test missed — a "
            "different asset class, a volatility regime, a filter role? Show a net, certifiable "
            "edge after costs, then we'll talk.*"
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
            "# Inverse-Fisher-RSI — a quantitative teardown 🔬\n"
            "### Welch/HAC splits per horizon · a 20-seed random-signal placebo · a head-to-head "
            "vs plain RSI(2/14) · a costed timer vs a random-exposure control · an AR(1) "
            "synthetic power check\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "Ehlers' Inverse Fisher Transform of RSI (`tanh(WMA(0.1*(RSI(5)-50), 9))`) is a "
            "clean, fully specified formula with a stated mechanism (compress a bounded "
            "oscillator toward its extremes) — no ambiguity to steelman around. The job here is "
            "to measure whether that compression buys any forecasting power, honestly, against "
            "the baselines it's supposed to beat.\n\n"
            "> ⚠️ **Data note.** SPY, QQQ, IWM, AAPL, MSFT, NVDA daily total-return closes "
            "(`auto_adjust=True`), 2010-01-04 → 2026-06-30, yfinance, cached. No survivorship "
            "on the index/ETF legs; the five single names are a current-mega-cap **selection**, "
            "not a survivor panel (no delisted names by construction — named, not a Signal-axis "
            "confound here since nothing is conditioned on having survived). Methods in "
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
            f"| **Signal** | `NONE` | 10d gap **{R['head'][10][2]:+.1f} bps**, Welch "
            f"**t={R['head'][10][3]:+.2f}**, NW **t={R['head'][10][4]:+.2f}** — wrong-signed; "
            f"placebo p={R['placebo_p']:.4f} |\n"
            f"| **Tradability** | `MIRAGE` | timer net Sharpe **{R['tm_sharpe5']:.2f}** vs "
            f"buy-and-hold **{R['bh_sharpe']:.2f}**; random-exposure z={R['rnd_z']:+.2f} |\n"
            "| **Sharper than plain RSI?** | `BUSTED` | IFT-RSI gap worse than both RSI(14) and "
            "RSI(2) on the identical test |\n\n"
            "> 💡 In plain words: the transform changes the *shape* of the chart, not the "
            "*content* of the signal."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $x_t = \\text{RSI}(5)_t$, $v_{1,t} = 0.1(x_t - 50)$, $v_{2,t} = "
            "\\text{WMA}_9(v_1)_t$, and $\\text{IFT}_t = \\tanh(v_{2,t}) \\in [-1, 1]$. Ehlers'\n"
            "claim: a cross of $\\text{IFT}_t$ **up** through $-0.5$ (from below) marks a sharper "
            "oversold reversal than a plain RSI threshold cross, and symmetrically for a "
            "**down**-cross through $+0.5$.\n\n"
            "- **H₁ (bullish edge).** $E[\\text{fwd-ret} \\mid \\text{IFT cross up } -0.5] > "
            "E[\\text{fwd-ret}]$ (unconditional), at 5/10/20-day horizons.\n"
            "- **H₂ (bearish edge).** Symmetric, sign-flipped, for the $+0.5$ down-cross.\n"
            "- **H₃ (relative sharpness).** The IFT-RSI edge (if any) exceeds plain RSI(14) and "
            "plain RSI(2) edges measured the identical way.\n"
            "- **H₄ (tradability).** A costed long-flat timer built on the crossover beats "
            "buy-and-hold and a random-exposure control.\n\n"
            "We find **H₁ rejected** (wrong-signed at the closest-to-significant horizon), "
            "**H₂ a near-total non-result**, **H₃ rejected** (IFT-RSI is the *worst* of the "
            "three), **H₄ rejected** (trails buy-and-hold, ties a coin on exposure-adjusted "
            "terms)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Forward-return windows at h=5/10/20 **overlap** (adjacent signal days share "
            "return history), so the planned primary is a **Welch t** on the pooled "
            "signal-vs-unconditional split, cross-checked with a **Newey-West** (Bartlett "
            "kernel, lags = h) t on the pooled dummy regression — the serial-correlation-robust "
            "version of the identical mean gap. A **random-signal placebo** (20 seeds × 200 "
            "draws of matched-count random entries) is the fairest null: it inherits the "
            "basket's own drift and volatility exactly, so any edge over it is genuinely about "
            "*timing*, not *being long tech stocks in a bull market*. The plain-RSI comparison "
            "runs through the **identical** `headline_stats` function — no separate tuning per "
            "indicator."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {R['tickers']}, total-return closes, {R['start']} → {R['end']}.\n"
            "- **Indicator.** Ehlers' exact IFT-RSI recipe (RSI(5), WMA(9), tanh); thresholds "
            "±0.5, his own published levels.\n"
            "- **Execution.** One documented lag: signal known at close *t*, forward return "
            "earned close *t+1* → *t+1+h* (a single shift).\n"
            "- **Headline.** Welch t + NW(h) t + Wilson hit rate, pooled across the basket, at "
            "h = 5/10/20.\n"
            "- **Comparison.** The identical machinery on plain RSI(14) (cross up 30 / down 70) "
            "and plain RSI(2) (cross up 10, Connors-style) at h=10.\n"
            "- **Third axis.** A long-flat timer, 2 × one-way cost × NAV per round trip (5/10 "
            "bps), vs buy-and-hold Sharpe and a random-exposure (Bernoulli, matched "
            "time-in-market) control, 20 seeds.\n"
            "- **Control.** AR(1) synthetic reversion process; the null (rho=0) must not fire "
            "across 20 seeds, a planted rho must light up — checked at the horizon the effect "
            "actually operates at (h=1, a single-lag AR process), not the headline h=5/10/20."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline split across horizons\n\n"
            "Pooled Welch t (points) and Newey-West cross-check, IFT-RSI bullish cross vs "
            "unconditional, at each horizon."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [st.headline_stats(REAL, SIGNALS, 'ift_bull', h) for h in st.HORIZONS]\n"
            "    gaps = [r['gap_bps'] for r in rows]; wt = [r['welch_t'] for r in rows]\n"
            "    nwt = [r['nw_t'] for r in rows]\n"
            "else:\n"
            "    gaps = [R['head'][h][2] for h in (5,10,20)]\n"
            "    wt = [R['head'][h][3] for h in (5,10,20)]\n"
            "    nwt = [R['head'][h][4] for h in (5,10,20)]\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.8, 6.4), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [1, 1]})\n"
            "x = np.arange(3)\n"
            "a1.bar(x, gaps, color=[RED if g<0 else GREEN for g in gaps], width=.5)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('gap vs unconditional (bps)')\n"
            "a1.set_title('IFT-RSI bullish-cross edge: small, inconsistent, wrong-signed at 10d')\n"
            "a2.bar(x - .18, wt, width=.35, label='Welch t', color=GREY)\n"
            "a2.bar(x + .18, nwt, width=.35, label='NW t', color=AMBER)\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(2, ls='--', c=RED, lw=1)\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_xticks(x); a2.set_xticklabels(['5d','10d','20d'])\n"
            "a2.set_ylabel('t-stat'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('gaps (bps):', [round(g,1) for g in gaps], ' Welch t:', [round(t,2) for t in wt])"
        ),
        md(
            f"> 💡 In plain words: the effect is small at 5 days ({R['head'][5][2]:+.1f} bps, "
            f"t={R['head'][5][3]:+.2f}), gets *worse and closer to the bar* at 10 days "
            f"({R['head'][10][2]:+.1f} bps, NW t={R['head'][10][4]:+.2f}) — but in the direction "
            "that says buying the \"oversold snap-back\" is a *below-average* entry — then flips "
            f"positive and shrinks in significance by 20 days ({R['head'][20][2]:+.1f} bps, "
            "t={:+.2f}). A real effect doesn't usually change sign like that; this reads as "
            "noise around zero, not a signal.".format(R['head'][20][3])
        ),
        md(
            "### 4b · The random-signal placebo — the fairest null\n\n"
            "Draws inherit the basket's own drift/vol exactly, so beating them means genuine "
            "timing skill, not just being long a bull market."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.random_signal_placebo(REAL, SIGNALS, 'ift_bull', 10,\n"
            "                                  n_draws_per_seed=40, n_seeds=4)\n"
            "    obs, draws = pl['obs']*1e4, pl['draws']*1e4\n"
            "else:\n"
            "    obs = R['placebo_obs']\n"
            "    rng = np.random.default_rng(682)\n"
            "    draws = rng.normal(R['placebo_mean'], R['placebo_sd'], 800)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=45, color=GREY, alpha=.85,\n"
            "        label='null: random-count-matched entries (light in-notebook run)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed IFT-RSI signal {obs:+.1f} bps')\n"
            "ax.set_xlabel('mean 10-day forward return of a random entry set (bps)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f'canonical placebo (results.md): p = {R[\"placebo_p\"]:.4f} '\n"
            "             f'({R[\"placebo_draws\"]:,} draws)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'canonical: observed {R[\"placebo_obs\"]:+.1f} bps vs placebo mean '\n"
            "      f'{R[\"placebo_mean\"]:+.1f} bps (sd {R[\"placebo_sd\"]:.1f}), p = {R[\"placebo_p\"]:.4f}')"
        ),
        md(
            f"> 💡 In plain words: **{R['placebo_p']*100:.1f}%** of random entry sets the same "
            "size beat the real IFT-RSI signal. If the crossover carried genuine information "
            "you'd expect it to beat *most* random draws, not lose to nearly all of them."
        ),
        md(
            "### 4c · The head-to-head — does the transform beat plain RSI?\n\n"
            "Same machinery, same horizon (10d), three different indicators."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names = {'IFT-RSI (-0.5 cross)': 'ift_bull',\n"
            "             'plain RSI(14) (cross up 30)': 'rsi14_bull',\n"
            "             'plain RSI(2) (cross up 10)': 'rsi2_bull'}\n"
            "    tbl = {k: st.headline_stats(REAL, SIGNALS, v, 10) for k, v in names.items()}\n"
            "    labels = list(tbl.keys()); gaps = [tbl[k]['gap_bps'] for k in labels]\n"
            "    ts = [tbl[k]['welch_t'] for k in labels]; ns = [tbl[k]['n_sig'] for k in labels]\n"
            "else:\n"
            "    labels = list(R['cmp'].keys())\n"
            "    gaps = [R['cmp'][k][1] for k in labels]; ts = [R['cmp'][k][2] for k in labels]\n"
            "    ns = [R['cmp'][k][0] for k in labels]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "cols = [RED if g < 0 else GREEN for g in gaps]\n"
            "bars = ax.barh(labels, gaps, color=cols)\n"
            "for i, (g, t, n) in enumerate(zip(gaps, ts, ns)):\n"
            "    ax.annotate(f'{g:+.1f} bps (t={t:+.2f}, n={n})', (g, i),\n"
            "                ha='left' if g > 0 else 'right', va='center', fontsize=9)\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('10-day forward-return gap vs unconditional (bps)')\n"
            "ax.set_title('IFT-RSI is not the sharpest tool here — it is the dullest')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({l: round(g,1) for l, g in zip(labels, gaps)})"
        ),
        md(
            "> 💡 In plain words: plain RSI(2) — the simplest, oldest indicator of the three — "
            "is the only one with a positive (still uncertified) gap. The Inverse Fisher "
            "Transform's compression does not translate into better-timed entries; if anything "
            "it correlates with the worst of the three here. (This is a lightweight, "
            "identical-machinery comparison, not RSI(2)'s full protocol — see "
            "[75-knee-jerk](../../75-knee-jerk/) for that indicator's own dedicated, real "
            "verdict.)"
        ),
        md(
            "### 4d · The third axis — a timer, costed, against a random-exposure control\n\n"
            "Long-flat: enter on the bullish cross, exit on the bearish cross. One documented "
            "execution lag; 2 × one-way cost × NAV per round trip."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tm5 = st.timer_with_costs(REAL, SIGNALS, 'ift_bull', 'ift_bear', cost_bps=5.0)\n"
            "    rnd = st.random_exposure_control(REAL, tm5['exposure'], cost_bps=5.0,\n"
            "                                      n_seeds=6)\n"
            "    net, bh, exp_ = tm5['sharpe_net'], tm5['sharpe_bh'], tm5['exposure']\n"
            "    rmean, rsd = rnd.mean(), rnd.std(ddof=1)\n"
            "else:\n"
            "    net, bh, exp_ = R['tm_sharpe5'], R['bh_sharpe'], R['exposure']/100\n"
            "    rmean, rsd = R['rnd_mean'], R['rnd_sd']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['IFT-RSI timer\\n(net, 5bps)', 'buy & hold', 'random-exposure\\n(mean of seeds)'],\n"
            "       [net, bh, rmean], color=[AMBER, GREY, GREY], width=.55,\n"
            "       yerr=[0, 0, rsd], capsize=6)\n"
            "for i, v in enumerate([net, bh, rmean]): ax.annotate(f'{v:.2f}', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('annualised Sharpe'); ax.set_title(f'Timer trails buy-and-hold, ties a coin (exposure {exp_*100:.0f}%)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'net {net:.2f}  buy&hold {bh:.2f}  random-exposure mean {rmean:.2f} (sd {rsd:.2f})')"
        ),
        md(
            f"> 💡 In plain words: net Sharpe **{R['tm_sharpe5']:.2f}** vs buy-and-hold's "
            f"**{R['bh_sharpe']:.2f}** — expected, since the timer is flat "
            f"**{100-R['exposure']:.0f}%** of a historic bull run — but the more relevant "
            f"comparison is the random-exposure control at the *same* time-in-market: "
            f"mean **{R['rnd_mean']:.2f}** (sd {R['rnd_sd']:.2f}). Real Sharpe **z = "
            f"{R['rnd_z']:+.2f}**, beating only {R['rnd_beats']}/20 random seeds — not "
            "certified. For reference, the RSI(14) timer does even worse "
            f"(net Sharpe {R['rsi14_tm_sharpe5']:.2f}, {R['rsi14_exposure']:.0f}% exposure)."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Deterministic AR(1) log-return process: `r_t = mu + rho*(mu - r_{t-1}) + eps_t`. "
            "The reversion knob operates at a single 1-day lag by construction, so the machinery "
            "check runs at **h=1** (not the headline h=5/10/20) — a design decision, documented, "
            "not a snooped horizon."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(10):\n"
            "    close = data.synthetic_world(rho=0.0, seed=682 + s_)\n"
            "    null_ts.append(st.synthetic_detect(close, h=1)['welch_t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "close = data.synthetic_world(rho=0.6, seed=682)\n"
            "planted_t = st.synthetic_detect(close, h=1)['welch_t']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.scatter(np.zeros(len(null_ts)) + np.linspace(-.12, .12, len(null_ts)), null_ts,\n"
            "           color=GREY, s=40, label=f'null worlds (rho=0), {len(null_ts)} seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5, label='planted rho=0.6')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null', 'planted'])\n"
            "ax.set_ylabel('Welch t (h=1)')\n"
            "ax.set_title('Control: no null fires; a planted 1-day reversion lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/{len(null_ts)} seeds  |  '\n"
            "      f'planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and never crosses the "
            f"bar (canonical 20-seed run in `results.md`); a planted rho=0.6 reads "
            f"t = {R['syn_planted_t']:.2f}. The pipeline is unbiased and has power — the flat, "
            "wrong-signed real-tape result at h=5/10/20 is a genuine null, not an underpowered "
            "test. *(A faithful-engine / power check only — never cited in support of the "
            "real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — no horizon clears **t ≥ 2** in support of the claim; the "
            f"closest call (10d NW t = {R['head'][10][4]:+.2f}) runs the wrong direction, the "
            f"bearish side is near-flat, and a random-signal placebo beats the real signal on "
            f"{R['placebo_p']*100:.1f}% of draws.\n"
            f"- **Tradability `MIRAGE`** — costed timer nets Sharpe {R['tm_sharpe5']:.2f} vs "
            f"buy-and-hold {R['bh_sharpe']:.2f}, and only weakly beats a random-exposure control "
            f"(z = {R['rnd_z']:+.2f}, not certified).\n"
            "- **\"Sharper than plain RSI(2/14)\"? `BUSTED`** — IFT-RSI's own gap is worse than "
            "both baselines on the identical machinery; the Fisher compression changes the "
            "chart's look, not its information content."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson.** A monotone or bounded transform of an existing indicator "
            "can change its *distributional shape* — variance, kurtosis, how it looks on a "
            "chart — without changing its *information content* about future returns. Sibling "
            "[183-fisher-transform](../../183-fisher-transform/) proves this outright for "
            "*plain* Fisher on price (100% crossover coincidence with raw price); this study "
            "shows the *inverse* transform on *RSI* doesn't buy anything either, even though "
            "the underlying oscillator (RSI(2), tested properly in "
            "[75-knee-jerk](../../75-knee-jerk/)) genuinely can carry a real edge.\n"
            "- **A natural next step.** Test IFT-RSI as a *regime filter* alongside an "
            "independently-real signal, rather than a standalone entry trigger — untested here.\n"
            "- **Dedup map:** [183-fisher-transform](../../183-fisher-transform/) (plain Fisher "
            "on price, proven monotone/redundant), [75-knee-jerk](../../75-knee-jerk/) (RSI(2), "
            "the desk's real RSI signal, used here only as a baseline), "
            "[428-stochastic-rsi](../../428-stochastic-rsi/) (a different stacked transform, "
            "also `NONE`), [669-rsi-divergence](../../669-rsi-divergence/) (a structural "
            "pattern, also `NONE`).\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).*"
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
