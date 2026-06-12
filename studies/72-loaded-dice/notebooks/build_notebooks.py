"""Generate the two narrative notebooks for Study 72 (Loaded-Dice).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
5-minute parquet under ../_cache/ if present and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-12).
R = dict(
    n=3801, tpd=11.3, tpy=2851,
    cross_mean=-0.39, cross_win=49.6, cross_t=-1.12, cross_skew=-0.00,
    cross_sharpe=-0.91, ci_lo=-2.39, ci_hi=0.69, frac_neg=88,
    rand_mean=0.14, rand_win=50.3, rand_t=0.36, rand_lo=-1.06, rand_hi=0.23,
    cross_minus_rand=-0.53,
    naive14_win=80.1, naive14_mean=0.40, naive14_skew=-2.37, naive14_t=1.20,
    nostop_win=93.5, nostop_mean=0.47, nostop_skew=-8.03, nostop_t=1.35,
    net05=-0.89, net05_t=-2.55, net10=-1.39, net10_t=-3.98, net20=-2.39,
    gross_ann=-11.2, net10_ann=-39.7,
    nostop_tp=477, nostop_eod=19, nostop_sl=1,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble — imports, the basket, and small pooled helpers.
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from loaded_dice import data, strategy as st

TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA", "ES=F", "NQ=F"]

def _have_cache():
    return all(os.path.exists(data._cache_path(t, data.DEFAULT_CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

def pooled(tp, sl, cost, seed=None):
    \"\"\"Pool barrier trades across the basket (cached real tapes).\"\"\"
    frames = []
    for t in TICKERS:
        b = data.fetch_5m(t, fetch=False)
        ent = st.crossover_entries(b["close"])
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(b, ent, tp_R=tp, sl_R=sl, cost_bps=cost, directions=dirs))
    return pd.concat(frames, ignore_index=True)

print("real 5m cache present:" , HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Loaded-Dice — can a moving-average crossover beat a coin? 🎲\n"
            "### The 5-minute SMA(5/10) scalp, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats a coin%3F: Busted](https://img.shields.io/badge/Beats_a_coin%3F-Busted-8b949e?style=flat-square)\n\n"
            "Here's a recipe you'll find in a thousand day-trading videos: on the 5-minute chart, "
            "when the fast 5-bar average crosses the slow 10-bar average, jump in the way it crossed, "
            "grab a few dollars, and do it again — *all day*. It's pitched as a **dice roll with the "
            "odds nudged your way** by the trend. This notebook asks the only question that matters: "
            "is the die actually loaded, or is it just a coin wearing a costume?\n\n"
            "> 📓 **This is the plain-language layer.** Want the t-stats, the bootstrap intervals and "
            "the cost maths? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the crossover point the right way? | **No.** With a fair (symmetric) "
            f"target & stop it wins **{R['cross_win']}%** of the time — a coin. |\n"
            "| Does it beat an actual coin flip? | **No.** A random-direction entry on the *same* "
            f"trades does just as well (better, by a hair). |\n"
            "| 'But my win-rate is 90%+!' | That's the trap. A tiny target with no real stop "
            f"manufactures a **{R['nostop_win']}%** win-rate — and a tail that eats it. |\n"
            "| Could you trade it? | **No.** At ~11 trades/day, costs turn the coin into a "
            f"**{R['net10_ann']:.0f}%/yr** loser. |\n\n"
            "> 💡 The trend doesn't load the die. The high win-rate people brag about is an "
            "accounting trick of *where they put the exit*, not an edge."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Plot SMA(5) and SMA(10) on a 5-minute chart. Go **long** when 5 crosses **above** "
            "10, **short** when it crosses **below**. Take a few dollars of profit and repeat. The "
            "moving averages keep you on the right side of the short-term trend — it's a coin flip, "
            "but a coin flip with an edge.\"*\n\n"
            "It's an honest-sounding idea. Markets do trend sometimes, and a crossover is the "
            "simplest possible trend detector. The bet is that **very-short-term momentum** leaks "
            "enough information into the next few minutes to tilt a near-random outcome your way."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it's true, it's a money printer: a simple, mechanical, all-day rule anyone can run. "
            "If it's false, thousands of people are paying spreads and commissions ~11 times a day "
            "to play a coin flip with a house edge **against** them. The difference between those two "
            "worlds is one honest test — so let's run it."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Two rules keep us honest:\n\n"
            "1. **Make the bet symmetric.** If your profit target and your stop are the same "
            "distance away (here: one *ATR* — the bar's typical range — on each side), then a coin "
            "scores zero. Only a *real* sense of direction can beat that. Asymmetric exits (small "
            "target, far stop) hide the truth behind a pretty win-rate.\n"
            "2. **Race it against an actual coin.** Take the exact same entry moments and flip a "
            "coin for the direction. If the crossover knows something, it beats the coin. If it "
            "doesn't, it won't.\n\n"
            "We run it on **eight liquid markets** (SPY, QQQ, IWM, AAPL, TSLA, NVDA, ES, NQ) over "
            "~60 days of 5-minute bars — the most history the data vendor gives at this resolution."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the trap that fools everyone.** Here's what happens to your *win-rate* and "
            "your *risk* as you shrink the profit target and push the stop further away — the "
            "literal 'grab a few dollars and let it ride' recipe:"
        ),
        code(
            "rows = []\n"
            "labels = [('1.0R / 1.0R (fair)', 1.0, 1.0), ('0.5R / 2.0R', 0.5, 2.0),\n"
            "          ('0.25R / ~no stop', 0.25, 8.0)]\n"
            "if HAVE_REAL:\n"
            "    for lbl, tp, sl in labels:\n"
            "        s = st.summarize(pooled(tp, sl, 0.0), 'ret_gross')\n"
            "        rows.append((lbl, s['win_rate']*100, s['skew'], s['mean_bps']))\n"
            "else:\n"
            "    rows = [('1.0R / 1.0R (fair)', R['cross_win'], R['cross_skew'], R['cross_mean']),\n"
            "            ('0.5R / 2.0R', R['naive14_win'], R['naive14_skew'], R['naive14_mean']),\n"
            "            ('0.25R / ~no stop', R['nostop_win'], R['nostop_skew'], R['nostop_mean'])]\n"
            "tbl = pd.DataFrame(rows, columns=['exit rule', 'win-rate %', 'P&L skew', 'mean bps/trade'])\n\n"
            "fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))\n"
            "ax[0].bar(tbl['exit rule'], tbl['win-rate %'], color=[GREEN, AMBER, RED])\n"
            "ax[0].axhline(50, ls='--', c='k', lw=1); ax[0].set_ylabel('win-rate %')\n"
            "ax[0].set_title('Win-rate looks amazing...'); ax[0].tick_params(axis='x', rotation=20)\n"
            "ax[1].bar(tbl['exit rule'], tbl['P&L skew'], color=[GREEN, AMBER, RED])\n"
            "ax[1].set_ylabel('P&L skew (negative = fat left tail)')\n"
            "ax[1].set_title('...because the losses got huge'); ax[1].tick_params(axis='x', rotation=20)\n"
            "plt.tight_layout(); plt.show()\n"
            "tbl.round(2)"
        ),
        md(
            "The 'no stop' version wins **~93%** of its trades. It also has a P&L skew of about "
            "**−8**: almost every trade banks a sliver, and the rare loser is a catastrophe. On SPY "
            f"that arm exits **{R['nostop_tp']} take-profits, {R['nostop_eod']} flat-at-close, and "
            f"just {R['nostop_sl']} stop** — picking up pennies in front of a steamroller. A high "
            "win-rate is **not** evidence of an edge; it's a choice about where you hide the loss."
        ),
        md(
            "**Now the honest test: the fair 1:1 bet, crossover vs a coin.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = st.summarize(pooled(1, 1, 0.0), 'ret_gross')\n"
            "    r = st.summarize(pooled(1, 1, 0.0, seed=72), 'ret_gross')\n"
            "    cm, cw, rm, rw = c['mean_bps'], c['win_rate']*100, r['mean_bps'], r['win_rate']*100\n"
            "else:\n"
            "    cm, cw, rm, rw = R['cross_mean'], R['cross_win'], R['rand_mean'], R['rand_win']\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars = ax.bar(['SMA crossover\\n(with the trend)', 'Random direction\\n(a coin)'],\n"
            "              [cm, rm], color=[RED, GREY], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('gross profit per trade (bps)')\n"
            "ax.set_title('The fair bet: the crossover does NOT beat a coin')\n"
            "for b, w in zip(bars, [cw, rw]):\n"
            "    ax.annotate(f'win {w:.1f}%', (b.get_x()+b.get_width()/2, 0),\n"
            "                ha='center', va='bottom' if b.get_height()<0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'crossover: {cm:+.2f} bps/trade   |   coin: {rm:+.2f} bps/trade')"
        ),
        md(
            f"On the real tape the crossover earns **{R['cross_mean']:+.2f} bps per trade** — "
            "essentially zero, and if anything slightly *negative*. The coin earns "
            f"**{R['rand_mean']:+.2f}**. They are the same bet. The trend filter adds nothing.\n\n"
            "> 💡 Why? At the 5-minute scale, prices don't really trend — they jiggle. The little "
            "wiggles that make a 5/10 crossover fire are mostly noise and 'bid-ask bounce', so by "
            "the time you've entered, the next wiggle is just as likely to go against you."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The crossover direction carries no real information; it loses a "
            "(statistically invisible) sliver and ties a coin.\n"
            "- **Tradability — Mirage.** Even if it were flat, ~11 trades/day of costs make it a "
            f"**{R['net10_ann']:.0f}%/yr** loser.\n"
            "- **Beats a coin? — Busted.** The 'loaded die' isn't loaded. The famous 90% win-rate is "
            "a trick of the exit, not a glimpse of the future."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "This is where even a *flat* coin goes to die. Add a realistic round-trip cost and run "
            "the fair version ~11 times a day:"
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    net = [st.summarize(pooled(1, 1, c), 'ret_net')['mean_bps'] for c in costs]\n"
            "else:\n"
            "    net = [R['cross_mean'], R['net05'], R['net10'], R['net20'], R['cross_mean']-5+0.39]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net, 0, where=[n<0 for n in net], color=RED, alpha=.12)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net profit per trade (bps)')\n"
            "ax.set_title('Every bit of cost just digs the hole deeper')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross is already {net[0]:+.2f} bps — there is no positive break-even cost.')"
        ),
        md(
            "The line starts below zero and only falls. Because the gross edge is already negative, "
            "**there is no cost low enough to make it work** — the usual 'it dies at the costs line' "
            "story doesn't even apply, because it was never alive."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Is *anything* findable at 5 minutes?** The companion notebook shows a synthetic "
            "control: when we *plant* real momentum in a fake tape, this exact rule finds it. So the "
            "machine works — the real market just has nothing to find here.\n"
            "- **The opposite bet.** If short-term moves *reverse* rather than continue, you'd fade "
            "the crossover, not follow it — that's [Study 32 — Rip-Tide](../../32-rip-tide/).\n"
            "- **One timeframe up.** The daily 50/200 'golden cross' is "
            "[Study 21 — Fools-Gold](../../21-fools-gold/) — same family, same ending.\n\n"
            "*Think the die really is loaded? Fork this, change the timeframe or the instrument, and "
            "show a fair-bet edge that beats the coin and clears the costs. That's the bar.*"
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
            "# Loaded-Dice — a quantitative teardown 🔬\n"
            "### Real 5-minute tape · barrier backtest · random-direction control · HAC inference · turnover costs\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats a coin%3F: Busted](https://img.shields.io/badge/Beats_a_coin%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether the "
            "SMA(5/10) crossover direction beats a random-direction control on a symmetric-barrier "
            "backtest, across eight liquid 5-minute tapes, and what turnover does to it.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo 5-minute bars, ~60-day rolling window, "
            "as-of 2026-06-12; the offline core and tests run on a deterministic synthetic tape. "
            "Methods & sources in [`docs/references.md`](../docs/references.md), reproducible numbers "
            "in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Cross gross **{R['cross_mean']:+.2f} bps/trade**, HAC "
            f"*t* = **{R['cross_t']:+.2f}**; no edge over a random-direction control "
            f"(Δ = {R['cross_minus_rand']:+.2f} bps). |\n"
            f"| **Tradability** | `MIRAGE` | ~{R['tpd']} trades/day → significant loser at 1 bp "
            f"(*t* = {R['net10_t']:+.2f}), **{R['net10_ann']:.0f}%/yr** net. |\n"
            f"| **Beats a coin?** | `BUSTED` | Fixed-tick arm: win **{R['nostop_win']}%**, skew "
            f"**{R['nostop_skew']:+.1f}**, mean *t* = {R['nostop_t']:+.2f} — a skew illusion. |\n\n"
            "> 💡 Three failures, one cause: there is no exploitable serial dependence in 5-minute "
            "returns at this horizon, so the rule is a fair (slightly tax-disadvantaged) die."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be the bar log-return and $x_t = \\mathrm{sign}(\\mathrm{SMA}_5 - "
            "\\mathrm{SMA}_{10})_t$ the crossover state. The recipe asserts:\n\n"
            "- **H₁ (signal).** $\\mathbb{E}[\\,d\\cdot r_{t\\to\\text{exit}}\\,] > 0$ where $d$ is "
            "the cross direction — i.e. the cross carries directional information, *measured "
            "symmetrically* so a coin scores 0.\n"
            "- **H₂ (beats random).** That expectation exceeds the same statistic with $d$ replaced "
            "by an i.i.d. fair coin on identical entries.\n"
            "- **H₃ (tradable).** It survives a realistic round-trip cost at the rule's natural "
            "turnover.\n\n"
            "We reject H₁ and H₂ on the real tape and H₃ trivially."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The crossover scalp is the most-recommended retail intraday rule there is. If H₁–H₃ "
            "held, it would be a rare free lunch. The interesting failure isn't *that* it loses — "
            "it's *how*: the win-rate the recipe sells (90%+) is real and completely uninformative, "
            "a textbook demonstration that **win-rate is not expectancy** once the exit is "
            "asymmetric and the P&L is skewed."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Entry.** Cross of SMA(5)/SMA(10) on closes up to $t$; **enter at $t{+}1$'s open** "
            "(no look-ahead).\n"
            "- **Exit (symmetric).** Barriers at $\\pm 1\\,\\mathrm{ATR}(20)$ from entry; flat at "
            "the 16:00 close; a bar that straddles both barriers is filled at the **stop** "
            "(conservative).\n"
            "- **Control.** Identical entries, direction $\\in\\{-1,+1\\}$ drawn i.i.d. (seeded).\n"
            "- **Inference.** Newey–West HAC *t* on per-trade returns; circular block-bootstrap "
            "Sharpe CI; per-instrument breakdown; a cost sweep at the rule's turnover.\n"
            "- **Positive control.** A synthetic tape with a tunable bar-level AR(1) momentum, to "
            "confirm the engine recovers an edge *when one exists*.\n\n"
            "Eight tapes: SPY, QQQ, IWM, AAPL, TSLA, NVDA, ES=F, NQ=F."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The symmetric bet, per instrument — no tape carries a signal\n\n"
            "Per-instrument HAC *t* on the gross per-trade return. If the rule worked anywhere, a "
            "bar would clear +2."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in TICKERS:\n"
            "        b = data.fetch_5m(t, fetch=False); ent = st.crossover_entries(b['close'])\n"
            "        s = st.summarize(st.run_trades(b, ent, tp_R=1, sl_R=1, cost_bps=0), 'ret_gross')\n"
            "        recs.append((t, s['n_trades'], s['win_rate']*100, s['mean_bps'], s['tstat']))\n"
            "    perinst = pd.DataFrame(recs, columns=['ticker','n','win%','mean_bps','t'])\n"
            "    p = pooled(1,1,0); ps = st.summarize(p,'ret_gross')\n"
            "else:\n"
            "    perinst = pd.DataFrame({'ticker':TICKERS,\n"
            "        't':[-0.13,0.06,-1.36,-0.80,-0.45,-0.47,-0.22,0.24]})\n"
            "    perinst['mean_bps']=np.nan; perinst['win%']=np.nan; perinst['n']=np.nan\n"
            "    ps = dict(mean_bps=R['cross_mean'], tstat=R['cross_t'], win_rate=R['cross_win']/100)\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "col = [RED if abs(v)<2 else GREEN for v in perinst['t']]\n"
            "ax.bar(perinst['ticker'], perinst['t'], color=col)\n"
            "for s in (2,-2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('HAC t-stat (gross mean/trade)')\n"
            "ax.set_title('Eight markets, eight non-results (|t| < 2 everywhere)')\n"
            "ax.set_ylim(-3, 3); plt.tight_layout(); plt.show()\n"
            "print(f\"pooled: {ps['mean_bps']:+.2f} bps/trade, HAC t = {ps['tstat']:+.2f}\")\n"
            "perinst.round(2)"
        ),
        md(
            "> 💡 Every instrument sits inside the ±2 band, most of them *below* zero. Pooling "
            f"3,801 trades gives **{R['cross_mean']:+.2f} bps**, HAC *t* = **{R['cross_t']:+.2f}** — "
            "the extra power of pooling buys us confidence that the edge is *absent*, not merely "
            "noisy."
        ),
        md(
            "### 4b · Crossover vs the coin — same bet, with a bootstrap band\n\n"
            "The random-direction control on identical entries, with a circular block-bootstrap "
            "95% CI on the annualised Sharpe of the crossover arm."
        ),
        code(
            "if HAVE_REAL:\n"
            "    C = pooled(1,1,0); cm = st.summarize(C,'ret_gross')\n"
            "    rand_means = [st.summarize(pooled(1,1,0,seed=s),'ret_gross')['mean_bps'] for s in range(20)]\n"
            "    tpy = int(round(cm['n_trades']/len(TICKERS)/42*252))\n"
            "    ci = stats.sharpe_ci_bootstrap(pd.Series(C['ret_gross'].values), periods_per_year=tpy, seed=72)\n"
            "    cmean, clo, chi, fn = cm['mean_bps'], ci['ci_low'], ci['ci_high'], ci['frac_negative']*100\n"
            "else:\n"
            "    rand_means=[R['rand_mean']]; cmean,clo,chi,fn = R['cross_mean'],R['ci_lo'],R['ci_hi'],R['frac_neg']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(rand_means, bins=12, color=GREY, alpha=.65, label='random-control means (20 seeds)')\n"
            "ax.axvline(cmean, c=RED, lw=2.5, label=f'crossover: {cmean:+.2f} bps')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('gross mean (bps/trade)'); ax.set_ylabel('count')\n"
            "ax.set_title('The crossover lands inside the coin\\'s own noise band'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'annualised Sharpe 95% CI: [{clo:+.2f}, {chi:+.2f}]  ({fn:.0f}% of resamples < 0)')"
        ),
        md(
            f"The crossover's gross Sharpe is **{R['cross_sharpe']:+.2f}**, bootstrap 95% CI "
            f"**[{R['ci_lo']:+.2f}, {R['ci_hi']:+.2f}]** ({R['frac_neg']}% of resamples negative). "
            "It is statistically indistinguishable from — and point-estimate *worse* than — a coin."
        ),
        md(
            "### 4c · The fixed-tick illusion — win-rate ≠ expectancy\n\n"
            "Shrink the target, widen the stop: the win-rate climbs to the famous 90s, the skew "
            "collapses, and the mean stays pinned at noise."
        ),
        code(
            "specs = [('1.0R/1.0R', 1.0,1.0), ('0.5R/2.0R', 0.5,2.0), ('0.25R/8R', 0.25,8.0)]\n"
            "if HAVE_REAL:\n"
            "    tt = [(l,)+tuple(st.summarize(pooled(tp,sl,0),'ret_gross')[k] for k in ('win_rate','mean_bps','skew','tstat')) for l,tp,sl in specs]\n"
            "    tt = pd.DataFrame(tt, columns=['rule','win','mean_bps','skew','t']); tt['win']*=100\n"
            "else:\n"
            "    tt = pd.DataFrame({'rule':['1.0R/1.0R','0.5R/2.0R','0.25R/8R'],\n"
            "        'win':[R['cross_win'],R['naive14_win'],R['nostop_win']],\n"
            "        'mean_bps':[R['cross_mean'],R['naive14_mean'],R['nostop_mean']],\n"
            "        'skew':[R['cross_skew'],R['naive14_skew'],R['nostop_skew']],\n"
            "        't':[R['cross_t'],R['naive14_t'],R['nostop_t']]})\n"
            "tt.round(2)"
        ),
        md(
            "> 💡 The win-rate is a deterministic function of the barrier ratio, not of any "
            "forecasting skill: with target $a$ and stop $b$ on a driftless path the hit-probability "
            "of the target is ≈ $b/(a+b)$. Set $b=8a$ and you 'win' ~89% by construction — while the "
            "expectancy stays zero and the variance migrates entirely into the left tail "
            f"(skew **{R['nostop_skew']:+.1f}**)."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — pooled gross {R['cross_mean']:+.2f} bps/trade, HAC "
            f"*t* {R['cross_t']:+.2f}; Δ vs control {R['cross_minus_rand']:+.2f} bps; every "
            "instrument |*t*| < 1.4.\n"
            f"- **Tradability `MIRAGE`** — gross Sharpe {R['cross_sharpe']:+.2f}; net "
            f"*t* {R['net10_t']:+.2f} at 1 bp; {R['net10_ann']:.0f}%/yr.\n"
            f"- **Beats a coin? `BUSTED`** — {R['nostop_win']}% win-rate at skew "
            f"{R['nostop_skew']:+.1f}, mean *t* {R['nostop_t']:+.2f}."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — turnover does the killing\n\n"
            "Per-trade net mean and its HAC *t* across round-trip cost, at ~11 trades/day."
        ),
        code(
            "costs = [0.0,0.5,1.0,2.0,5.0]\n"
            "if HAVE_REAL:\n"
            "    sw = [st.summarize(pooled(1,1,c),'ret_net') for c in costs]\n"
            "    mean=[s['mean_bps'] for s in sw]; tst=[s['tstat'] for s in sw]\n"
            "else:\n"
            "    mean=[R['cross_mean'],R['net05'],R['net10'],R['net20'],R['cross_mean']-4.61]\n"
            "    tst=[R['cross_t'],R['net05_t'],R['net10_t'],-6.84,-15.4]\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.plot(costs, mean, 'o-', c=RED, lw=2, label='net mean (bps/trade)')\n"
            "ax2 = ax.twinx(); ax2.plot(costs, tst, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(-2, ls=':', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (bps)', color=RED)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('No break-even cost exists: it starts negative and certifies as a loser')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('break-even round-trip cost: none (gross is already below zero)')"
        ),
        md(
            "> 💡 Contrast with the desk's *real-but-untradable* studies, where a positive gross "
            "edge dies at a finite break-even (e.g. Rubber-Band's IBS snap-back). Here the gross is "
            "negative, so the break-even is undefined — costs are the executioner, not the cause."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "Is the *engine* capable of finding an edge, or does it always print 'none'? Plant a "
            "bar-level AR(1) momentum in a synthetic 5-minute tape and sweep it: the crossover's "
            "advantage over the coin should turn on exactly when real persistence appears."
        ),
        code(
            "moms = [-0.15,-0.10,-0.05,0.0,0.05,0.10,0.15,0.20]\n"
            "edge = []\n"
            "for m in moms:\n"
            "    b,_ = data.synthetic_5m(n_days=120, momentum=m, seed=72)\n"
            "    ent = st.crossover_entries(b['close'])\n"
            "    c = st.summarize(st.run_trades(b,ent,tp_R=1,sl_R=1,cost_bps=0),'ret_gross')['mean_bps']\n"
            "    r = st.summarize(st.run_trades(b,ent,tp_R=1,sl_R=1,cost_bps=0,\n"
            "        directions=st.random_directions(len(ent),seed=1)),'ret_gross')['mean_bps']\n"
            "    edge.append(c-r)\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.plot(moms, edge, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('planted bar-level AR(1) momentum'); ax.set_ylabel('cross − coin (bps/trade)')\n"
            "ax.set_title('The engine works: it harvests momentum when momentum exists')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At momentum 0 the edge is ~0; it rises with persistence and is negative under reversion.')"
        ),
        md(
            "The crossover advantage is monotone in the planted momentum and crosses zero right at "
            "a martingale — so the engine is a faithful momentum detector. The real-tape verdict is "
            "therefore a statement about the **market**, not the method: at 5-minute fidelity there "
            "is no momentum here to harvest. Forks worth trying: an explicit reversal sign "
            "([Rip-Tide](../../32-rip-tide/)), an intraday-momentum *time window* (Gao et al. 2018) "
            "rather than every cross, or a higher timeframe ([Fools-Gold](../../21-fools-gold/))."
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
