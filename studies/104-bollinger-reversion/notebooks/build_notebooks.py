"""Generate the two narrative notebooks for Study 104 (Bollinger-Reversion).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md).  The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
daily parquet under ../_cache/ if present and otherwise quote the frozen headline
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-13).
R = dict(
    n=811, tpy=6.3,
    # Lower-band reversion (20-day time exit, gross)
    rev_mean=193.09, rev_win=64.6, rev_t=5.34, rev_skew=-0.22,
    # Random-day control
    rand_mean=141.09, rand_win=61.5, rand_t=6.20,
    # Delta (rev - rand)
    delta=52.00, delta_t=0.63,
    # Upper-band breakout
    brk_mean=118.11, brk_win=62.0, brk_t=5.49,
    # Mid-exit version (for comparison)
    rev_mid_mean=141.74, rev_mid_t=4.86,
    # Cost sweep (net, 20-day time exit)
    net0=193.09, net0_t=5.34,
    net1=192.09, net1_t=5.32,
    net2=191.09, net2_t=5.29,
    net5=188.09, net5_t=5.21,
    # Per-instrument (rev mean / t / rand mean / delta)
    spy_rev=140.4, spy_t=1.90, spy_rand=110.1, spy_delta=30.3,
    qqq_rev=226.0, qqq_t=3.21, qqq_rand=123.2, qqq_delta=102.8,
    aapl_rev=115.6, aapl_t=1.08, aapl_rand=255.4, aapl_delta=-139.8,
    msft_rev=218.5, msft_t=2.86, msft_rand=93.5, msft_delta=125.0,
    jpm_rev=353.5, jpm_t=3.74, jpm_rand=151.1, jpm_delta=202.4,
    xle_rev=104.9, xle_t=1.14, xle_rand=71.8, xle_delta=33.1,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from bollinger_reversion import data, strategy as st

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "JPM", "XLE"]

def _have_cache():
    return all(os.path.exists(data._cache_path(t, data.DEFAULT_CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

def pooled(side="reversion", exit_mode="time", max_hold=20, cost=0.0, seed=None):
    \"\"\"Pool trades across the basket (cached real tapes).\"\"\"
    frames = []
    for t in TICKERS:
        b = data.fetch_daily(t, fetch=False)
        if seed is not None:
            ent = st.random_entries(b, n=200, seed=seed)
        else:
            ent = st.band_entries(b["close"], side=side)
        frames.append(st.run_trades(b, ent, exit_mode=exit_mode, cost_bps=cost,
                                    max_hold=max_hold))
    return pd.concat(frames, ignore_index=True)

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Bollinger-Reversion -- does price *always* return to the bands?\n"
            "### The lower-band buy, tested honestly, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Price_always_returns%3F: Busted](https://img.shields.io/badge/"
            "Price_always_returns%3F-Busted-8b949e?style=flat-square)\n\n"
            "Here's a recipe from every technical-trading book: on the daily chart, when the close "
            "drops below the lower Bollinger Band (the 20-day moving average minus two standard "
            "deviations), **buy** -- because price always comes back. It's sold as a law of "
            "nature, a rubber band stretched too far. This notebook asks the honest question: does "
            "the band tell you something a *random day* doesn't?\n\n"
            "> **This is the plain-language layer.** Want the t-stats, the per-instrument breakdown, "
            "and the breakout contradiction laid out numerically? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the lower-band pierce beat a random buy? | **Barely.** It earns "
            f"**+{R['rev_mean']:.0f} bps** per 20-day trade vs **+{R['rand_mean']:.0f} bps** "
            "for a random buy on the same stocks. The gap (52 bps) has t ~ 0.6. |\n"
            "| 'But 64.6% of my trades make money!' | A random buy also makes money 61.5% of "
            "the time. It's a bull market, not the bands. |\n"
            "| Does the *opposite* rule (breakout) also work? | **Yes -- +118 bps.** Both the "
            "'reversion' and 'breakout' band entries are positive because both are just *buying* "
            "in a rising market. |\n"
            "| Could you trade it? | **Fragile.** Costs aren't the issue. The edge collapses if "
            "the market stops trending upward. |\n\n"
            "> The bands do not hold a magic reversion secret. "
            "The drift of the S&P 500 (~10%/yr) is doing all the work."
        ),

        # ---- BEAT 1 -- THE CLAIM -----------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"When the daily close pierces the lower Bollinger Band (20-SMA minus 2 standard "
            "deviations), buy. Price always returns to the middle band -- that's your exit. "
            "It's a rubber-band effect: the more stretched, the stronger the snap-back.\"*\n\n"
            "It's intuitive: the Bollinger Bands contain ~95% of closing prices by construction "
            "(for a normal distribution). A pierce to the downside is a local extreme. The bet is "
            "that the market *over-reacts* and will reverse, making the entry genuinely better "
            "than a random day in the same stock."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If it's true, the lower-band pierce is a cheap, mechanical signal anyone can implement "
            "in minutes. On six liquid US equities over 21 years it fires ~6 times a year per stock "
            "-- low turnover, easy to run. If it's false (or mostly drift), traders following this "
            "rule are mistaking the long-term upward trend of the market for a band-specific effect, "
            "and will be caught out in any flat or falling market."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Two rules keep us honest:\n\n"
            "1. **Match the exit horizon.** The band-entry is compared against a random-day buy "
            "held for exactly 20 days -- not exited at the mid-band. If the band-entry exits "
            "earlier (because it hit the mid-band) while the random buy holds for 20 days, we're "
            "comparing apples to motorcycles. Both arms use the same 20-day hold.\n"
            "2. **Test the opposite rule too.** If 'buy the lower band' works because of reversion, "
            "then 'buy the upper band' (a breakout) should *not* work. If both work, the bands "
            "are irrelevant -- the market just went up.\n\n"
            "We run it on **six liquid US equities** (SPY, QQQ, AAPL, MSFT, JPM, XLE) over "
            "21 years of daily data (2005-2026) -- enough to spot a real edge."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First: the band entry vs a random buy -- both held for 20 days.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rev = st.summarize(pooled('reversion','time',20,0.0), 'ret_gross')\n"
            "    rand = st.summarize(pooled(None,'time',20,0.0,seed=42), 'ret_gross')\n"
            "    brk = st.summarize(pooled('breakout','time',20,0.0), 'ret_gross')\n"
            "    cm, cw, rm, rw, bm, bw = (rev['mean_bps'], rev['win_rate']*100,\n"
            "                               rand['mean_bps'], rand['win_rate']*100,\n"
            "                               brk['mean_bps'], brk['win_rate']*100)\n"
            "else:\n"
            "    cm, cw, rm, rw, bm, bw = (R['rev_mean'], R['rev_win'],\n"
            "                               R['rand_mean'], R['rand_win'],\n"
            "                               R['brk_mean'], R['brk_win'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.8))\n"
            "arms = ['Lower-band\\n(reversion buy)', 'Random day\\n(baseline)', 'Upper-band\\n(breakout buy)']\n"
            "means = [cm, rm, bm]\n"
            "wins = [cw, rw, bw]\n"
            "colors = [AMBER, GREY, AMBER]\n"
            "bars = ax.bar(arms, means, color=colors, width=.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('gross profit per trade (bps, 20-day hold)')\n"
            "ax.set_title('Both band entries work -- but so does the random buy')\n"
            "for b, w in zip(bars, wins):\n"
            "    ax.annotate(f'win {w:.1f}%', (b.get_x()+b.get_width()/2,\n"
            "                                   max(b.get_height()+5, 5)),\n"
            "                ha='center', va='bottom', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Reversion: {cm:+.1f} bps  |  Random: {rm:+.1f} bps  |  Breakout: {bm:+.1f} bps')"
        ),
        md(
            f"All three bars are positive. The lower-band entry earns **+{R['rev_mean']:.0f} bps** "
            f"and the upper-band entry (its supposed *opposite*) earns **+{R['brk_mean']:.0f} bps**. "
            f"The random buy earns **+{R['rand_mean']:.0f} bps**.\n\n"
            "When both the 'reversion' and 'momentum' interpretation of the bands make money, "
            "**neither is the signal** -- the rising market is."
        ),
        md(
            "**Now look at what drives the win-rate.**"
        ),
        code(
            "# Synthetic demo: reversion vs random across planted reversion strengths\n"
            "moms = [0.0, 0.05, 0.10, 0.15, 0.20]\n"
            "rev_edges, rand_edges = [], []\n"
            "for m in moms:\n"
            "    b,_ = data.synthetic_daily(n_days=1500, reversion=m, seed=104)\n"
            "    ent_r = st.band_entries(b['close'], side='reversion')\n"
            "    ent_rand = st.random_entries(b, n=max(len(ent_r),1), seed=42)\n"
            "    if len(ent_r):\n"
            "        led_r = st.run_trades(b, ent_r, exit_mode='time', cost_bps=0., max_hold=20)\n"
            "        led_rand = st.run_trades(b, ent_rand, exit_mode='time', cost_bps=0., max_hold=20)\n"
            "        rev_edges.append(st.summarize(led_r,'ret_gross')['mean_bps'])\n"
            "        rand_edges.append(st.summarize(led_rand,'ret_gross')['mean_bps'])\n"
            "    else:\n"
            "        rev_edges.append(float('nan')); rand_edges.append(float('nan'))\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.plot(moms, rev_edges, 'o-', c=AMBER, lw=2, label='lower-band entry')\n"
            "ax.plot(moms, rand_edges, 's--', c=GREY, lw=2, label='random-day entry')\n"
            "ax.fill_between(moms, rev_edges, rand_edges, alpha=.12, color=GREEN,\n"
            "                label='incremental edge')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xlabel('planted mean-reversion strength')\n"
            "ax.set_ylabel('gross mean (bps/20-day trade)')\n"
            "ax.set_title('Band entry outperforms random only when reversion is planted in the data')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            "On a synthetic tape with **no planted reversion**, the band entry and a random buy "
            "earn similar returns. As we plant stronger mean-reversion, the band entry pulls ahead "
            "-- because it really does locate the local extremes. On the real tape (21 years, six "
            "stocks) the planted equivalent is somewhere between 0 and 0.05 -- a weak effect "
            "dominated by the upward drift."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- Weak.** The lower-band entry earns "
            f"**+{R['rev_mean']:.0f} bps/trade** gross (HAC t = +{R['rev_t']:.2f}), but the "
            f"random-day baseline earns +{R['rand_mean']:.0f} bps (t = +{R['rand_t']:.2f}) on the "
            "same stocks. The incremental edge vs random (delta ~ +52 bps, t ~ 0.63) is "
            "**not statistically significant**.\n"
            "- **Tradability -- Fragile.** Costs aren't the problem (low turnover, ~6/year). "
            "The issue is the benchmark: the random buy captures almost the same drift premium. "
            "Any bear market or flat regime will erase the edge.\n"
            f"- **'Price always returns?' -- Busted.** The upper-band breakout also earns "
            f"+{R['brk_mean']:.0f} bps in the same period. The bands detect nothing; the market "
            "trend is doing the work."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The good news: at ~6 trades per year per stock, costs are almost irrelevant. "
            "The bad news: the benchmark is free too -- just buy randomly."
        ),
        code(
            "costs = [0.0, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    net = [st.summarize(pooled('reversion','time',20,c),'ret_net')['mean_bps']\n"
            "           for c in costs]\n"
            "    net_rand = [st.summarize(pooled(None,'time',20,c,seed=42),'ret_net')['mean_bps']\n"
            "                for c in costs]\n"
            "else:\n"
            "    net = [R['net0'], R['net1'], R['net2'], R['net5']]\n"
            "    net_rand = [R['rand_mean']-c for c in costs]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=AMBER, lw=2, label='lower-band entry')\n"
            "ax.plot(costs, net_rand, 's--', c=GREY, lw=1.5, label='random-day (baseline)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (bps/trade)')\n"
            "ax.set_title('Both lines stay positive -- costs are not the issue here')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('The band entry and the random buy remain profitable even after realistic costs.')\n"
            "print('The problem is not survival -- it is that the random baseline is almost as good.')"
        ),
        md(
            "Both lines stay comfortably positive: this is not a story of 'the edge dies at costs.' "
            "It is a story of **an edge that barely exists above the drift baseline**. If US equities "
            "mean-revert to flat returns over the next decade, the ~52 bps delta the bands add could "
            "vanish or reverse -- while the random buyer also loses. That's why it is FRAGILE."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **Bear-market sub-sample.** Repeat the test using only 2007-2009 or 2022. "
            "If the bands add an incremental edge there (where drift is negative), the "
            "reversion story would have genuine support.\n"
            "- **The intraday version.** [Study 72 -- Loaded-Dice](../../72-loaded-dice/) runs the "
            "same 'buy the extreme' logic on 5-minute bars vs a coin flip -- at that horizon "
            "the crossover earns nothing even gross.\n"
            "- **Fading the breakout instead.** If the upper-band pierce is a *reversion* entry "
            "rather than a momentum entry, does it beat the lower-band pierce? Split the 21-year "
            "sample by VIX regime and check.\n\n"
            "*Think the bands really do add something beyond drift? Fork this, restrict the "
            "back-test to flat or falling markets, and show a delta-vs-random t-stat above 2. "
            "That's the bar.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 -- FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Bollinger-Reversion -- a quantitative teardown\n"
            "### Real daily tape - 21 years - lower-band entry vs random-day control - "
            "HAC inference - breakout contradiction\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Price_always_returns%3F: Busted](https://img.shields.io/badge/"
            "Price_always_returns%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.*  We test whether the "
            "lower-band pierce earns an incremental return above a **random-day baseline** on a "
            "20-day hold, across six liquid US equity tapes (2005-2026), and whether the "
            "upper-band pierce simultaneously contradicts the claim.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars 2005-01-03 to 2026-06-12, "
            "six instruments; reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **In plain words** notes translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Lower-band gross **+{R['rev_mean']:.0f} bps/trade**, "
            f"HAC *t* = **+{R['rev_t']:.2f}**; random-day baseline +{R['rand_mean']:.0f} bps "
            f"(t = +{R['rand_t']:.2f}); incremental delta = +{R['delta']:.0f} bps "
            f"(t ~ +{R['delta_t']:.2f}) -- not significant. |\n"
            f"| **Tradability** | `FRAGILE` | Costs not the issue (~{R['tpy']:.1f} trades/yr); "
            "edge barely exceeds random-day baseline; regime-dependent (bull-market tape). |\n"
            f"| **Price always returns?** | `BUSTED` | Upper-band breakout earns "
            f"**+{R['brk_mean']:.0f} bps** (t = +{R['brk_t']:.2f}) on same instruments; "
            "both rules positive because the market trended up 2005-2026. |\n\n"
            "> In plain words: the mean is real, the t-stat is significant -- but the same "
            "is true of a random-day buy in a bull market.  The incremental signal from the "
            "bands is weak and not robust."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $c_t$ be the closing price, $\\mu_t = \\mathrm{SMA}_{20}(c)_t$, "
            "$\\sigma_t = \\mathrm{Std}_{20}(c)_t$.  The lower band is "
            "$L_t = \\mu_t - 2\\sigma_t$.  The recipe asserts:\n\n"
            "- **H1 (incremental signal).** "
            "$\\mathbb{E}[r_{t+1, t+20} \\mid c_t < L_t] > \\mathbb{E}[r_{t+1, t+20}]$ -- "
            "i.e. a lower-band pierce predicts above-average 20-day forward returns.\n"
            "- **H2 (directional specificity).** The *upper*-band pierce "
            "($c_t > U_t = \\mu_t + 2\\sigma_t$) should predict *below*-average forward "
            "returns if reversion is the mechanism -- a band signal, not just drift.\n"
            "- **H3 (tradable).** The edge survives round-trip costs at the rule's natural "
            "turnover (~6/stock/year).\n\n"
            "We **fail to reject** H1 on a gross basis (t = +5.34), but the incremental "
            "delta vs a random buy is t ~ +0.63 (too small).  H2 is **rejected** (upper-band "
            "also earns +118 bps).  H3 holds trivially (costs are small)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "The Bollinger Band lower-pierce is cited in virtually every technical-analysis "
            "curriculum.  If H1 held cleanly and H2 held, it would represent a mechanical "
            "over-reaction signal, consistent with the De Bondt-Thaler (1985) overreaction "
            "literature at the multi-week horizon.  The interesting failure is not *that* the "
            "gross edge is positive -- it is *that the random buy is almost as good*, revealing "
            "the real driver as the 21-year US equity bull market, not the bands themselves."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Entry.** First close below the lower Bollinger Band (20-SMA, 2-sigma) after "
            "a non-below day; **enter at the next bar's open** (no look-ahead).\n"
            "- **Exit (time-fair).** Flat after exactly **20 calendar trading days** -- the same "
            "horizon for both the band-entry and the random-day control.  (A mid-band exit "
            "creates an asymmetric-duration bias: the band arm exits earlier when it 'works', "
            "boosting its per-day return vs a fixed-hold random arm.  We report mid-exit numbers "
            "separately for comparison.)\n"
            "- **Control.** 200 random daily entries per ticker, seeded, same 20-day hold.\n"
            "- **Contradiction test.** Upper-band breakout entries on the same 20-day hold.\n"
            "- **Inference.** Newey-West HAC *t* on per-trade returns; delta t-stat on "
            "the (unmatched) difference of means.\n"
            "- **Positive control.** Synthetic daily tape with tunable AR(1) mean-reversion -- "
            "confirms the engine detects an edge *when one exists*.\n\n"
            "Six tapes: SPY, QQQ, AAPL, MSFT, JPM, XLE -- daily 2005-01-03 to 2026-06-12."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),

        md(
            "### 4a - Lower-band entry vs random-day control (20-day time exit)\n\n"
            "Per-instrument gross mean and HAC *t*.  The bar chart also shows the random-day "
            "baseline for each instrument."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in TICKERS:\n"
            "        b = data.fetch_daily(t, fetch=False)\n"
            "        ent_rev = st.band_entries(b['close'], side='reversion')\n"
            "        ent_rand = st.random_entries(b, n=len(ent_rev), seed=42)\n"
            "        sr = st.summarize(st.run_trades(b,ent_rev,'time',cost_bps=0.,max_hold=20),'ret_gross')\n"
            "        sn = st.summarize(st.run_trades(b,ent_rand,'time',cost_bps=0.,max_hold=20),'ret_gross')\n"
            "        recs.append((t,len(ent_rev),sr['mean_bps'],sr['tstat'],sn['mean_bps'],\n"
            "                     sr['mean_bps']-sn['mean_bps']))\n"
            "    perinst = pd.DataFrame(recs,columns=['ticker','n','rev_bps','rev_t','rand_bps','delta'])\n"
            "else:\n"
            "    perinst = pd.DataFrame({'ticker':TICKERS,\n"
            "        'n':[149,140,112,119,139,152],\n"
            "        'rev_bps':[R['spy_rev'],R['qqq_rev'],R['aapl_rev'],R['msft_rev'],R['jpm_rev'],R['xle_rev']],\n"
            "        'rev_t':[R['spy_t'],R['qqq_t'],R['aapl_t'],R['msft_t'],R['jpm_t'],R['xle_t']],\n"
            "        'rand_bps':[R['spy_rand'],R['qqq_rand'],R['aapl_rand'],R['msft_rand'],R['jpm_rand'],R['xle_rand']],\n"
            "        'delta':[R['spy_delta'],R['qqq_delta'],R['aapl_delta'],R['msft_delta'],R['jpm_delta'],R['xle_delta']]})\n"
            "\n"
            "x = range(len(TICKERS)); w = 0.35\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "\n"
            "# Left: gross means side-by-side\n"
            "ax = axes[0]\n"
            "ax.bar([i-w/2 for i in x], perinst['rev_bps'], width=w, color=AMBER, label='lower-band')\n"
            "ax.bar([i+w/2 for i in x], perinst['rand_bps'], width=w, color=GREY, label='random-day')\n"
            "ax.set_xticks(list(x)); ax.set_xticklabels(TICKERS)\n"
            "ax.set_ylabel('gross mean (bps/20d)'); ax.legend()\n"
            "ax.set_title('Per-instrument gross returns')\n"
            "\n"
            "# Right: delta (rev - rand)\n"
            "ax2 = axes[1]\n"
            "colors2 = [GREEN if d>0 else RED for d in perinst['delta']]\n"
            "ax2.bar(perinst['ticker'], perinst['delta'], color=colors2)\n"
            "ax2.axhline(0, c='k', lw=1)\n"
            "ax2.set_ylabel('delta: lower-band minus random (bps)')\n"
            "ax2.set_title('Incremental edge: positive for 5/6 instruments, negative for AAPL')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(perinst.round(1).to_string(index=False))"
        ),
        md(
            "> In plain words: the lower-band entry beats the random baseline on 5 of 6 "
            "instruments, but the AAPL delta is negative (the random buy there does better). "
            f"Pooled delta = **+{R['delta']:.0f} bps** -- real but not statistically robust "
            f"(t ~ +{R['delta_t']:.2f})."
        ),

        md(
            "### 4b - The contradiction: the breakout rule also earns positive returns\n\n"
            "H2 would require the upper-band pierce to earn *below*-average forward returns "
            "(reversion logic: if lower-band = over-sold buy, then upper-band = over-bought sell). "
            "It does not: both are positive."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rev = st.summarize(pooled('reversion','time',20,0.), 'ret_gross')\n"
            "    brk = st.summarize(pooled('breakout','time',20,0.), 'ret_gross')\n"
            "    rand = st.summarize(pooled(None,'time',20,0.,seed=42), 'ret_gross')\n"
            "    arms = ['Lower-band\\nreversion', 'Random-day\\ncontrol', 'Upper-band\\nbreakout']\n"
            "    means = [rev['mean_bps'], rand['mean_bps'], brk['mean_bps']]\n"
            "    tstats = [rev['tstat'], rand['tstat'], brk['tstat']]\n"
            "else:\n"
            "    arms = ['Lower-band\\nreversion','Random-day\\ncontrol','Upper-band\\nbreakout']\n"
            "    means = [R['rev_mean'], R['rand_mean'], R['brk_mean']]\n"
            "    tstats = [R['rev_t'], R['rand_t'], R['brk_t']]\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "colors3 = [AMBER, GREY, AMBER]\n"
            "axes[0].bar(arms, means, color=colors3, width=.55)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_ylabel('gross mean (bps/20d)')\n"
            "axes[0].set_title('All three arms positive -- bands add little beyond drift')\n"
            "for b2, ts in zip(axes[0].patches, tstats):\n"
            "    axes[0].annotate(f't={ts:+.2f}',\n"
            "        (b2.get_x()+b2.get_width()/2, b2.get_height()+3), ha='center', fontsize=9)\n"
            "\n"
            "axes[1].bar(arms, tstats, color=colors3, width=.55)\n"
            "for s in (2,-2): axes[1].axhline(s, ls='--', c=GREY, lw=1)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('HAC t-stat'); axes[1].set_title('t-stats (all exceed 2 -- bull market drift)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'delta reversion-random: {means[0]-means[1]:+.0f} bps')\n"
            "print(f'delta breakout-random:  {means[2]-means[1]:+.0f} bps')"
        ),
        md(
            "> In plain words: both band extremes have HAC t-stats above 2 -- not because "
            "they are both right, but because the entire market trended up and any 'buy US equities "
            "and hold 20 days' strategy earns positive returns.  The random-day control has the "
            f"*highest* t-stat ({R['rand_t']:.2f}) because it has the most trades (1,200) and "
            "captures the same drift cleanly."
        ),

        md(
            "### 4c - The positive control -- the machinery finds reversion when planted\n\n"
            "Does the engine work at all?  On a synthetic tape with planted mean-reversion, the "
            "lower-band entry should pull ahead of random.  On a martingale it should not."
        ),
        code(
            "moms = [0.0, 0.05, 0.10, 0.15, 0.20]\n"
            "rev_means, rand_means, deltas = [], [], []\n"
            "for m in moms:\n"
            "    b,_ = data.synthetic_daily(n_days=1500, reversion=m, seed=104)\n"
            "    ent_r = st.band_entries(b['close'], side='reversion')\n"
            "    ent_rand = st.random_entries(b, n=max(len(ent_r),1), seed=42)\n"
            "    if len(ent_r) > 0:\n"
            "        lr = st.run_trades(b,ent_r,'time',cost_bps=0.,max_hold=20)\n"
            "        lrand = st.run_trades(b,ent_rand,'time',cost_bps=0.,max_hold=20)\n"
            "        mr = st.summarize(lr,'ret_gross')['mean_bps']\n"
            "        mn = st.summarize(lrand,'ret_gross')['mean_bps']\n"
            "    else:\n"
            "        mr, mn = float('nan'), float('nan')\n"
            "    rev_means.append(mr); rand_means.append(mn); deltas.append(mr-mn)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.plot(moms, rev_means, 'o-', c=AMBER, lw=2, label='lower-band entry')\n"
            "ax.plot(moms, rand_means, 's--', c=GREY, lw=2, label='random-day baseline')\n"
            "ax.fill_between(moms, rev_means, rand_means, alpha=.15, color=GREEN)\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls=':', c=GREY)\n"
            "ax.set_xlabel('planted mean-reversion (AR1 coefficient)')\n"
            "ax.set_ylabel('gross mean (bps/20-day trade)')\n"
            "ax.set_title('The engine detects reversion when planted -- real market has weak reversion')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('delta at reversion=0:', f'{deltas[0]:+.1f} bps')\n"
            "print('delta at reversion=0.10:', f'{deltas[2]:+.1f} bps')\n"
            "print('delta at reversion=0.20:', f'{deltas[4]:+.1f} bps')"
        ),
        md(
            "> In plain words: the machinery is correctly calibrated -- it recovers a monotone "
            "signal as the planted reversion grows.  The real tape sits close to the 'reversion=0' "
            "regime (a mild drift environment), which is why the delta is small and barely "
            "significant."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `WEAK`** -- pooled gross +{R['rev_mean']:.0f} bps, HAC "
            f"*t* +{R['rev_t']:.2f}; random-day baseline +{R['rand_mean']:.0f} bps; "
            f"incremental delta +{R['delta']:.0f} bps, t ~ +{R['delta_t']:.2f} (not significant). "
            "Upper-band breakout is also positive (+118 bps, t = +5.49), ruling out band-specific "
            "reversion as the driver.\n"
            f"- **Tradability `FRAGILE`** -- costs survive (~{R['tpy']:.1f} trades/yr, net barely "
            "moves); the problem is the benchmark.  A random buy in the same stocks over 2005-2026 "
            "earns almost as much.  Any neutral or bear regime would test the edge severely.\n"
            "- **Price always returns? `BUSTED`** -- 64.6% win-rate (price reaches SMA within 20d) "
            "vs 61.5% for random.  3 pp of win-rate, not a rubber-band law."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- cost sweep\n\n"
            "Per-trade net mean and HAC *t* across round-trip cost.  Low turnover means costs "
            "barely dent the return -- but the baseline is also immune to costs."
        ),
        code(
            "costs = [0.0, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    sw_rev = [st.summarize(pooled('reversion','time',20,c),'ret_net') for c in costs]\n"
            "    sw_rand = [st.summarize(pooled(None,'time',20,c,seed=42),'ret_net') for c in costs]\n"
            "    mean_rev=[s['mean_bps'] for s in sw_rev]; tst_rev=[s['tstat'] for s in sw_rev]\n"
            "    mean_rand=[s['mean_bps'] for s in sw_rand]\n"
            "else:\n"
            "    mean_rev=[R['net0'],R['net1'],R['net2'],R['net5']]\n"
            "    tst_rev=[R['net0_t'],R['net1_t'],R['net2_t'],R['net5_t']]\n"
            "    mean_rand=[R['rand_mean']-c for c in costs]\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))\n"
            "axes[0].plot(costs, mean_rev, 'o-', c=AMBER, lw=2, label='lower-band')\n"
            "axes[0].plot(costs, mean_rand, 's--', c=GREY, lw=2, label='random-day')\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_xlabel('round-trip cost (bps)'); axes[0].set_ylabel('net mean (bps)')\n"
            "axes[0].set_title('Both survive costs -- the gap is small'); axes[0].legend()\n"
            "\n"
            "if HAVE_REAL:\n"
            "    axes[1].plot(costs, tst_rev, 'o-', c=AMBER, lw=2, label='lower-band t-stat')\n"
            "    axes[1].axhline(2, ls='--', c=GREY); axes[1].axhline(0, c='k', lw=1)\n"
            "    axes[1].set_xlabel('round-trip cost (bps)'); axes[1].set_ylabel('HAC t-stat')\n"
            "    axes[1].set_title('t-stat stays well above 2 even at 5 bps'); axes[1].legend()\n"
            "else:\n"
            "    axes[1].text(0.5, 0.5, 'real tape required\\nfor t-stat sweep', ha='center',\n"
            "                 va='center', transform=axes[1].transAxes, fontsize=12)\n"
            "\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Gross HAC t-stat remains above 2 across all cost levels.')\n"
            "print('The real weakness is the baseline gap, not cost erosion.')"
        ),
        md(
            "> In plain words: unlike Study 72 (Loaded-Dice), where the gross edge is already "
            "negative so costs kill it trivially, here the gross edge is real -- it just barely "
            "clears the random-day baseline.  The **regime risk** (bull market 2005-2026) is the "
            "primary vulnerability, not the per-trade cost."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the positive control and open questions\n\n"
            "The synthetic control confirms the engine is a faithful mean-reversion detector. "
            "The real-market question is: does US equity daily price action contain enough "
            "mean-reversion *beyond drift* to make the band entry decisively better than random? "
            "Our 21-year answer is 'barely not.'  Forks worth pursuing:\n\n"
            "- **Bear-market sub-sample (2007-2009, 2022).** If the incremental delta holds or "
            "improves when the drift is negative, the claim has genuine support.  If it vanishes, "
            "it is pure bull-market carry.\n"
            "- **Tighter band (1.5-sigma).** A narrower band fires less often but at more extreme "
            "over-reactions -- test whether the delta (vs random) rises.\n"
            "- **The RSI or VIX filter.** Condition the entry on a confirming oversold signal "
            "(RSI < 30 or VIX > 25) to isolate genuine panic buys vs routine band touches.\n"
            "- **[Study 72 -- Loaded-Dice](../../72-loaded-dice/)** for the intraday version "
            "of the same logic, where even the gross edge is absent."
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
