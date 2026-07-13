"""Generate the two narrative notebooks for Study 767 (VIX9D-Term).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-07-12).
R = dict(
    n=3901,
    date_start="2011-01-04",
    date_end="2026-07-10",
    fp="5953babc50d3",
    frac_contango=0.731,
    # Q5-Q1 spread
    spread_1d=-2.79, t_spread_1d=-0.35,
    spread_5d=-10.25, t_spread_5d=-0.36,
    spread_21d=-67.86, t_spread_21d=-1.10,
    # Quintile means h=1
    q1_mean=8.73, q1_t=1.65,
    q2_mean=5.16, q2_t=1.33,
    q3_mean=2.68, q3_t=0.79,
    q4_mean=3.25, q4_t=1.17,
    q5_mean=5.94, q5_t=2.80,
    # Timing overlay
    active_mean=3.64, active_sharpe=0.83, active_t=3.41,
    passive_mean=5.27, passive_sharpe=0.78, passive_t=3.42,
    spread_mean=-1.64, spread_t=-1.45,
    switches_yr=37.0,
    # Regime stats
    n_contango=2851, pct_contango=73.1, mean_contango=4.97, sharpe_contango=0.97,
    n_back=1048, pct_back=26.9, mean_back=6.04, sharpe_back=0.60,
    # Annualised spread
    spread_ann=-4.1,
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
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from vix9d_term import data, strategy as st

# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-07-12).
R = dict(
    n=3901, date_start="2011-01-04", date_end="2026-07-10", fp="5953babc50d3",
    frac_contango=0.731,
    spread_1d=-2.79, t_spread_1d=-0.35,
    spread_5d=-10.25, t_spread_5d=-0.36,
    spread_21d=-67.86, t_spread_21d=-1.10,
    q1_mean=8.73, q1_t=1.65, q2_mean=5.16, q2_t=1.33,
    q3_mean=2.68, q3_t=0.79, q4_mean=3.25, q4_t=1.17, q5_mean=5.94, q5_t=2.80,
    active_mean=3.64, active_sharpe=0.83, active_t=3.41,
    passive_mean=5.27, passive_sharpe=0.78, passive_t=3.42,
    spread_mean=-1.64, spread_t=-1.45, switches_yr=37.0, spread_ann=-4.1,
    n_contango=2851, pct_contango=73.1, mean_contango=4.97, sharpe_contango=0.97,
    n_back=1048, pct_back=26.9, mean_back=6.04, sharpe_back=0.60,
)

def _have_cache():
    return os.path.exists(data._cache_path(data.DEFAULT_CACHE))

HAVE_REAL = _have_cache()
print("real daily cache present:", HAVE_REAL)
"""

BOOT_REAL = """\
if HAVE_REAL:
    df = data.fetch_daily(fetch=False)
    print(f"Tape: N={len(df)}, {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Fraction contango: {(df['slope']>0).mean():.3f}")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# VIX9D-Term — does the *short end* of the VIX curve time the market?\n"
            "### The 9-day vs 30-day contango/backwardation slope as a risk-off timer, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_buy-and-hold%3F: No](https://img.shields.io/badge/Beats_buy--and--hold%3F-No-8b949e?style=flat-square)\n\n"
            "Two volatility numbers sit at the front of the curve: **^VIX9D** (9-day implied "
            "vol) and **^VIX** (30-day). The 9-day tenor is the twitchiest point on the whole "
            "volatility surface — it reacts to *this afternoon's* headline. When ^VIX9D < ^VIX "
            "the front end slopes up (*contango*, the normal calm state); when ^VIX9D spikes "
            "*above* ^VIX the very short end *inverts* into backwardation, and vol-traders read "
            "it as acute near-term fear — a classic risk-off alert. The ratio log(^VIX / ^VIX9D) "
            "is tracked as a clean, continuous signal. This notebook asks the only question that "
            "matters: **does an inverted front end actually tell you the S&P is about to fall?**\n\n"
            "> **This is the plain-language layer.** Want the quintile tables, the HAC t-stats "
            "and the cost maths? That's **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),
        code(BOOT_REAL),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does steep contango predict *better* SPY returns? | **No.** "
            f"The Q5 (steepest contango) minus Q1 (deepest backwardation) spread is "
            f"**{R['spread_1d']:+.2f} bps** at 1 day — *wrong sign*, *t* = {R['t_spread_1d']:+.2f}. |\n"
            "| Does a timing rule (long in contango, flat when inverted) beat buy-and-hold? | "
            f"**No.** The timer trails by **{abs(R['spread_ann']):.1f}%/yr** at zero cost. |\n"
            "| Is an inverted front end actually dangerous for equities? | **The opposite (for raw "
            f"returns).** The {R['pct_back']:.1f}% of days in backwardation average "
            f"**{R['mean_back']:.1f} bps/day** — *more* than the contango mean of "
            f"{R['mean_contango']:.1f} bps/day. |\n\n"
            "> The 9-day slope tells you something real — about **volatility**. Calm (contango) "
            "days really are steadier (higher Sharpe). But it does **not** tell you which way "
            "the stock market will go tomorrow, and a timer that flees the inverted days flees "
            "the higher-returning ones."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *'When the front of the VIX curve is upward-sloping (9-day below 30-day) it's a "
            "green light — be long. When the 9-day pops above the 30-day, the market is pricing "
            "in imminent stress. Get defensive, cut equity exposure. The inverted front end has "
            "flagged every scare since 2011.'*\n\n"
            "It is a reasonable-sounding idea grounded in real behaviour. The 9-day tenor spikes "
            "in a scare because ultra-short-dated options are the cheapest possible tail hedge, "
            "so a jump above the 30-day genuinely reflects a rush for near-term protection. The "
            "slope is real information — the question is whether it *predicts* subsequent equity "
            "returns or just reflects the fear already in the tape."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the claim were true, the short-end VIX slope would be a near-free market-timing "
            "signal available from any charting platform at zero data cost, refreshing intraday. "
            "If it is false, a large number of practitioners are cutting equity exposure exactly "
            "when the subsequent returns turn out to be *highest* — paying the opportunity cost of "
            "missing post-scare bounces, and churning ~37 times a year to do it. The stakes are "
            "high enough to warrant an honest 15-year test."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three tests, two rules of rigour:\n\n"
            "1. **Quintile sort.** Divide every day into one of five contango/backwardation "
            "buckets (quintile 1 = deepest backwardation, 5 = steepest contango) and compare "
            "the SPY return over the next 1, 5, and 21 trading days for each bucket. The "
            "claim predicts a monotone ascent from Q1 to Q5.\n"
            "2. **Binary timer.** Be long SPY when yesterday's slope was positive, flat "
            "otherwise. Compare to always being long (buy-and-hold). The claim predicts the "
            "timer wins.\n"
            "3. **No look-ahead.** The slope at today's close forms the signal; the forward "
            "return starts tomorrow. No bar of information is used that wasn't available.\n\n"
            "Tape: ^VIX9D, ^VIX, SPY daily, January 2011 (^VIX9D history start) through July 2026 "
            f"— {R['n']:,} observations."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, what the short-end VIX term structure looks like in practice:**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)\n"
            "    axes[0].plot(df.index, df['VIX'], label='^VIX (30-day)', color=AMBER, lw=1, alpha=0.8)\n"
            "    axes[0].plot(df.index, df['VIX9D'], label='^VIX9D (9-day)', color=RED, lw=1, alpha=0.7)\n"
            "    axes[0].set_ylabel('VIX index level'); axes[0].legend(); axes[0].set_title('^VIX9D vs ^VIX (2011–2026)')\n"
            "    axes[1].fill_between(df.index, df['slope'], 0,\n"
            "        where=(df['slope'] >= 0), color=GREEN, alpha=0.4, label='Contango (slope>0)')\n"
            "    axes[1].fill_between(df.index, df['slope'], 0,\n"
            "        where=(df['slope'] < 0), color=RED, alpha=0.6, label='Backwardation (slope<0)')\n"
            "    axes[1].set_ylabel('slope = log(VIX/VIX9D)'); axes[1].legend()\n"
            "    axes[1].set_title(f'Short-end slope — {R[\"frac_contango\"]*100:.0f}% of time in contango')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print(f'Real tape not cached. Headlines: N={R[\"n\"]}, {R[\"date_start\"]} to {R[\"date_end\"]}')\n"
            "    print('Fraction in contango: " + f"{round(R['frac_contango']*100)}" + "%')"
        ),
        md(
            f"The front end is in contango **{R['frac_contango']*100:.0f}%** of the time — but "
            "notice how much *more* often it inverts than the 3-month curve (Study 111's "
            "^VIX/^VIX3M slope sat in contango ~90% of the time). The 9-day tenor flips into "
            "backwardation on any scare: not just the big crises (2015 August, 2018 Volmageddon, "
            "2020 COVID, 2022, 2025) but dozens of one- and two-day headline wobbles. These are "
            "the episodes the signal says to 'go defensive' — and many are followed by prompt "
            "recoveries."
        ),
        md(
            "**Now the key test: do contango days predict better next-day SPY returns?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tbl = st.quintile_table(df, horizons=[1])\n"
            "    means = [tbl[tbl['quintile']==q]['mean_bps'].values[0] for q in range(1,6)]\n"
            "    tstats = [tbl[tbl['quintile']==q]['t_stat'].values[0] for q in range(1,6)]\n"
            "else:\n"
            f"    means = [{R['q1_mean']}, {R['q2_mean']}, {R['q3_mean']}, {R['q4_mean']}, {R['q5_mean']}]\n"
            f"    tstats = [{R['q1_t']}, {R['q2_t']}, {R['q3_t']}, {R['q4_t']}, {R['q5_t']}]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "bars = ax.bar([f'Q{i}' for i in range(1,6)], means, color=[RED,AMBER,GREY,AMBER,GREEN])\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b, t in zip(bars, tstats):\n"
            "    ax.annotate(f't={t:+.1f}', (b.get_x()+b.get_width()/2, b.get_height()),\n"
            "                ha='center', va='bottom', fontsize=9)\n"
            "ax.set_xlabel('Slope quintile (Q1=deepest backwardation, Q5=steepest contango)')\n"
            "ax.set_ylabel('Next-day SPY mean return (bps)')\n"
            "ax.set_title('U-shaped, not a rising staircase — the claim fails')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Predicted pattern: Q1 < Q2 < Q3 < Q4 < Q5')\n"
            "print(f'Actual: Q1={means[0]:.1f} Q2={means[1]:.1f} Q3={means[2]:.1f} Q4={means[3]:.1f} Q5={means[4]:.1f}')"
        ),
        md(
            f"There is no rising staircase. The shape is **U-shaped**: the two *extremes* — "
            f"deep backwardation (Q1, **{R['q1_mean']:+.1f} bps**) and steep contango (Q5, "
            f"**{R['q5_mean']:+.1f} bps**) — both beat the flat middle (Q3, {R['q3_mean']:+.1f} bps). "
            "But Q1 (the 'danger' bucket) actually earns *more* than Q5 (the 'all-clear'). The "
            f"Q5-Q1 spread is **{R['spread_1d']:+.2f} bps** with HAC t = **{R['t_spread_1d']:+.2f}**: "
            "consistent with zero and pointing the wrong way."
        ),
        md(
            "**The timing overlay — long in contango, flat when the front end inverts:**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ov = st.timing_overlay(df, horizon=1, cost_bps=0.0)\n"
            "    cum_active = ov['r_active'].cumsum() * 100\n"
            "    cum_passive = ov['r_passive'].cumsum() * 100\n"
            "    sa = st.summarize(ov['r_active']); sp = st.summarize(ov['r_passive'])\n"
            "    act_m, act_sh, pas_m, pas_sh = sa['mean_bps'], sa['sharpe_ann'], sp['mean_bps'], sp['sharpe_ann']\n"
            "else:\n"
            f"    act_m, act_sh, pas_m, pas_sh = {R['active_mean']}, {R['active_sharpe']}, {R['passive_mean']}, {R['passive_sharpe']}\n"
            "    cum_active = cum_passive = None\n"
            "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            "if cum_active is not None:\n"
            "    ax.plot(ov.index, cum_passive, label=f'Buy-and-hold: Sharpe {pas_sh:+.2f}', color=GREEN, lw=1.5)\n"
            "    ax.plot(ov.index, cum_active, label=f'Slope timer: Sharpe {act_sh:+.2f}', color=RED, lw=1.5)\n"
            "    ax.set_ylabel('Cumulative log return (%)')\n"
            "    ax.set_title('Slope timer vs buy-and-hold — the timer sits out the rebounds')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "else:\n"
            f"    print(f'Timer: {{act_m:+.2f}} bps/day, Sharpe {{act_sh:+.2f}}')\n"
            f"    print(f'Passive: {{pas_m:+.2f}} bps/day, Sharpe {{pas_sh:+.2f}}')"
        ),
        md(
            f"The timing overlay earns **{R['active_mean']:+.2f} bps/day** (Sharpe {R['active_sharpe']:+.2f}) "
            f"vs buy-and-hold's **{R['passive_mean']:+.2f} bps/day** (Sharpe {R['passive_sharpe']:+.2f}). "
            f"The timer trails by **~{abs(R['spread_ann']):.1f}%/yr** at zero cost. The reason is "
            "visible in the chart: the biggest equity gains in the sample happen right after the "
            "front-end inversions the timer classifies as 'danger'. By sitting in cash during "
            "those ~27% of days, the timer misses the very returns it was supposed to protect "
            "against losing — and it churns constantly getting in and out."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Q5-Q1 spread is {R['spread_1d']:+.2f} bps (wrong sign), "
            f"HAC *t* = {R['t_spread_1d']:+.2f} at 1 day; negative at all horizons tested; the "
            "quintile pattern is U-shaped, not monotone. The short-end slope does not forecast "
            "SPY *direction*.\n"
            f"- **Tradability — Mirage.** The timer trails buy-and-hold by {abs(R['spread_ann']):.1f}%/yr "
            f"at zero cost; ~{R['switches_yr']:.0f} switches/yr mean every bp of friction widens the "
            "gap, and by 5 bps/switch the underperformance is itself significant.\n"
            f"- **Beats buy-and-hold? — No.** Backwardation days average {R['mean_back']:.1f} bps/day "
            f"(*more* than contango's {R['mean_contango']:.1f} bps/day in raw terms) — the market "
            "rebounds hardest from the exact episodes the slope flags as most dangerous."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            f"Even if the spread were positive, the turnover arithmetic would sink it. The 9-day "
            f"slope switches regimes ~{R['switches_yr']:.0f} times per year — nearly three times "
            "the churn of the 3-month curve — and each switch is a round-trip trade on the full "
            "SPY position. At realistic ETF round-trip costs:"
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    spreads = []\n"
            "    for c in costs:\n"
            "        ov2 = st.timing_overlay(df, horizon=1, cost_bps=c)\n"
            "        spreads.append(st.summarize(ov2['r_spread'])['mean_bps'])\n"
            "else:\n"
            f"    spreads = [{R['spread_mean']}, {R['spread_mean']}-0.07, {R['spread_mean']}-0.14, "
            f"{R['spread_mean']}-0.29, {R['spread_mean']}-0.73]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "ax.plot(costs, spreads, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, spreads, 0, where=[s<0 for s in spreads], color=RED, alpha=.12)\n"
            "ax.set_xlabel('Round-trip cost per switch (bps)'); ax.set_ylabel('Spread (bps/day)')\n"
            "ax.set_title('Gross spread is already negative — costs make it worse')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The slope timer underperforms buy-and-hold before costs; the heavy switching makes it worse.')"
        ),
        md(
            "The line starts below zero and only falls. Because the gross spread is already "
            "negative, **there is no switching cost low enough to make the timer competitive** "
            "with buying and holding. This is the strongest version of the 'Mirage' verdict: "
            "not 'dies at costs' but 'never lived' — and the high turnover only deepens the loss."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Is the slope useful for anything?** Yes — for **volatility**, not direction. "
            "On a risk-adjusted basis contango days *are* calmer (Sharpe 0.97 vs 0.60). Signals "
            "that trade *volatility* (variance swaps, VIX-ETP roll harvesting) can use that; a "
            "signal that trades equity *direction* cannot.\n"
            "- **The 3-month cousin.** The desk's [Study 111 — VIX-Term-Structure]"
            "(../../111-vix-term-structure/) runs the identical test on the ^VIX/^VIX3M slope "
            "and reaches the same verdict — but with a third of the switching, so the front-end "
            "version studied here is *strictly worse* to trade.\n"
            "- **What about VIX *level* mean-reversion?** VIX above 30 is statistically followed "
            "by higher SPY returns (the Connors/Alvarez literature), but that is the *level* of "
            "fear, not the slope between tenors — a separate claim.\n\n"
            "*Convinced the short-end slope works on a different horizon or instrument? Fork "
            "this, change the parameters, and show a positive Q5-Q1 spread that clears the "
            "|t| ≥ 2 bar. That's the standard.*"
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
            "# VIX9D-Term — a quantitative teardown\n"
            "### Daily tape · rolling quintile sort · binary timing overlay · HAC inference · regime breakdown\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_buy-and-hold%3F: No](https://img.shields.io/badge/Beats_buy--and--hold%3F-No-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "Same seven beats, every claim now carrying its standard error. We test whether "
            "the short-end slope (log(^VIX / ^VIX9D)) predicts SPY forward returns via a "
            "rolling quintile sort, a binary timing overlay, and a regime decomposition, "
            "over the full 2011–2026 daily tape.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, 2011-01-04 through "
            "2026-07-10 (^VIX9D history to present). Methods & sources in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **`> In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),
        code(BOOT_REAL),

        # ---- BEAT 0 — VERDICT UP FRONT ----------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Q5-Q1 spread **{R['spread_1d']:+.2f} bps** (wrong sign), "
            f"HAC *t* = **{R['t_spread_1d']:+.2f}** at h=1d; negative at all horizons, U-shaped quintiles. |\n"
            f"| **Tradability** | `MIRAGE` | Timer trails buy-and-hold **{abs(R['spread_ann']):.1f}%/yr** at "
            f"zero cost; spread *t* = **{R['spread_t']:+.2f}**, ~{R['switches_yr']:.0f} switches/yr, no positive break-even. |\n"
            f"| **Beats buy-and-hold?** | `NO` | Backwardation ({R['pct_back']:.1f}% of tape) "
            f"averages **{R['mean_back']:.1f} bps/day** raw vs contango's {R['mean_contango']:.1f} — "
            "the claimed danger signal earns *more* raw return. |\n\n"
            "> **In plain words:** the slope is real information about what the options market "
            "thinks about *volatility* (contango days have Sharpe 0.97 vs 0.60), but the claim "
            "that it predicts equity *direction* is backwards on this tape. Backwardation "
            "episodes — where the timer goes flat — produce the highest average raw returns."
        ),

        # ---- BEAT 1 — STEELMANNED CLAIM ----------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $s_t = \\log(\\text{VIX}_t / \\text{VIX9D}_t)$ be the short-end slope on day $t$ "
            "and $r_{t+1:t+h}$ the h-day forward SPY log-return. The term-structure timer asserts:\n\n"
            "- **H₁ (quintile forecast).** $\\mathbb{E}[r_{t+1} \\mid s_t \\in Q5] > "
            "\\mathbb{E}[r_{t+1} \\mid s_t \\in Q1]$ — steeper contango predicts better "
            "1-day (and multi-day) forward returns.\n"
            "- **H₂ (timing overlay).** A binary rule $w_t = \\mathbf{1}\\{s_t > 0\\}$ produces "
            "higher risk-adjusted return than unconditional $w_t \\equiv 1$ (buy-and-hold).\n"
            "- **H₃ (regime dominance).** Conditioning on contango delivers materially "
            "higher *returns* than conditioning on backwardation.\n\n"
            "We reject H₁ (spread negative, wrong sign), H₂ (timer underperforms), and "
            "H₃ (backwardation sub-sample delivers higher per-day *raw* returns — though, "
            "revealingly, *lower* Sharpe: the slope forecasts vol, not direction)."
        ),

        # ---- BEAT 2 — SO WHAT --------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "The short-end VIX slope is cited in practitioner research and widely used by "
            "retail vol-traders as a fast regime filter — faster than the 3-month curve because "
            "the 9-day tenor moves intraday. The stakes: a practitioner who cuts equity exposure "
            "on every front-end inversion misses the post-scare rebounds on this 15-year tape "
            "and churns ~37×/yr doing it. An honest test resolves whether the signal is "
            "actionable for direction or a case of economically-intuitive-but-empirically-absent "
            "forecasting."
        ),

        # ---- BEAT 3 — PROTOCOL -------------------------------------------------
        md(
            "## 3 · The protocol\n\n"
            "- **Signal.** $s_t = \\log(\\text{VIX}_t / \\text{VIX9D}_t)$, computed "
            "from Yahoo daily closes. Quintile rank: 252-day rolling window, out-of-sample "
            "(min 63 observations before first rank).\n"
            "- **Forward return.** Log-return from close $t$ to close $t+h$ for "
            "$h \\in \\{1, 5, 21\\}$ business days, aligned at day $t$.\n"
            "- **Spread test.** Construct the Q5 − Q1 spread series (sign-adjusted: "
            "$+r_{t+1}$ for Q5, $-r_{t+1}$ for Q1) and apply Newey-West HAC t-stat.\n"
            "- **Timer.** Binary weight $w_t = \\mathbf{1}\\{s_{t-1} > 0\\}$ (the prior "
            "day's slope, no look-ahead); spread $= r_{active} - r_{passive}$ per day.\n"
            "- **Inference.** Newey-West HAC t-stat throughout; cost sweep on switching.\n"
            "- **Positive control.** Synthetic tape with tunable planted signal confirms the "
            "engine recovers an edge when one exists."
        ),

        # ---- BEAT 4 — TEARDOWN -------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Quintile forward-return table — all horizons\n\n"
            "The claim predicts a monotone ascent Q1 < Q2 < Q3 < Q4 < Q5 at each horizon. "
            "HAC t-stats on per-day quintile means."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tbl = st.quintile_table(df, horizons=[1, 5, 21])\n"
            "    pivot = tbl.pivot(index='quintile', columns='horizon', values='mean_bps')\n"
            "    tpivot = tbl.pivot(index='quintile', columns='horizon', values='t_stat')\n"
            "else:\n"
            f"    pivot = pd.DataFrame({{\n"
            f"        1: [{R['q1_mean']},{R['q2_mean']},{R['q3_mean']},{R['q4_mean']},{R['q5_mean']}],\n"
            f"        5: [31.98, 32.63, 29.33, 15.63, 21.73],\n"
            f"        21: [151.33, 121.06, 121.89, 72.17, 83.47]}},\n"
            f"        index=[1,2,3,4,5])\n"
            f"    tpivot = pd.DataFrame({{\n"
            f"        1: [{R['q1_t']},{R['q2_t']},{R['q3_t']},{R['q4_t']},{R['q5_t']}],\n"
            f"        5: [1.59, 2.66, 2.94, 1.74, 3.25],\n"
            f"        21: [3.66, 3.93, 4.06, 2.44, 3.75]}},\n"
            f"        index=[1,2,3,4,5])\n"
            "fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))\n"
            "for ax, h in zip(axes, [1, 5, 21]):\n"
            "    vals = pivot[h].values\n"
            "    cols = [GREEN if v == max(vals) else (RED if v == min(vals) else GREY) for v in vals]\n"
            "    ax.bar([f'Q{i}' for i in range(1,6)], vals, color=cols)\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.set_title(f'h = {h} day(s)')\n"
            "    ax.set_ylabel('mean bps' if h==1 else '')\n"
            "    for xi, (v, t) in enumerate(zip(vals, tpivot[h].values)):\n"
            "        ax.text(xi, v + abs(v)*0.05, f't={t:+.1f}', ha='center', fontsize=8)\n"
            "plt.suptitle('Q5-Q1 spread is negative at every horizon — predicted direction WRONG', y=1.02)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('Q5-Q1 spread: h=1: {R['spread_1d']:+.2f}bps t={R['t_spread_1d']:+.2f} | h=5: {R['spread_5d']:+.2f}bps t={R['t_spread_5d']:+.2f} | h=21: {R['spread_21d']:+.2f}bps t={R['t_spread_21d']:+.2f}')"
        ),
        md(
            f"> **In plain words:** at every horizon tested, Q5 (steepest contango) earns "
            f"*less* than Q1 (deepest backwardation). The 1-day spread is "
            f"**{R['spread_1d']:+.2f} bps** with *t* = **{R['t_spread_1d']:+.2f}** — consistent "
            "with zero and pointing in the wrong direction. The pattern is *U-shaped*: the "
            "highest-vol tails (Q1 deep backwardation and Q5 steep contango) both out-earn the "
            "flat middle. If anything, the deep-backwardation bucket (the 'danger' one) has the "
            "single largest 1-day mean — the opposite of what the timer needs."
        ),
        md(
            "### 4b · Binary timing overlay — long in contango, flat in backwardation\n\n"
            "The active arm vs buy-and-hold; HAC t-stat on the daily spread."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ov = st.timing_overlay(df, horizon=1, cost_bps=0.0)\n"
            "    sa = st.summarize(ov['r_active'])\n"
            "    sp = st.summarize(ov['r_passive'])\n"
            "    ss = st.summarize(ov['r_spread'])\n"
            "    act_m, act_sh, act_t = sa['mean_bps'], sa['sharpe_ann'], sa['t_stat']\n"
            "    pas_m, pas_sh, pas_t = sp['mean_bps'], sp['sharpe_ann'], sp['t_stat']\n"
            "    spr_m, spr_t = ss['mean_bps'], ss['t_stat']\n"
            "    n_sw = int(ov['switched'].sum()); sw_yr = n_sw/len(ov)*252\n"
            "else:\n"
            f"    act_m, act_sh, act_t = {R['active_mean']}, {R['active_sharpe']}, {R['active_t']}\n"
            f"    pas_m, pas_sh, pas_t = {R['passive_mean']}, {R['passive_sharpe']}, {R['passive_t']}\n"
            f"    spr_m, spr_t = {R['spread_mean']}, {R['spread_t']}\n"
            f"    sw_yr = {R['switches_yr']}\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.bar(['Slope timer\\n(active)', 'Buy-and-hold\\n(passive)'],\n"
            "       [act_m, pas_m], color=[RED, GREEN], width=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean bps/day'); ax.set_title(f'Timer trails by {spr_m:+.2f} bps/day (t={spr_t:+.2f})')\n"
            "for y, lab in [(act_m, f'Sharpe {act_sh:+.2f}'), (pas_m, f'Sharpe {pas_sh:+.2f}')]:\n"
            "    ax.annotate(lab, (0 if y==act_m else 1, y), ha='center', va='bottom' if y>0 else 'top', fontsize=10)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Spread: {spr_m:+.2f} bps/day, HAC t = {spr_t:+.2f}, switches: {sw_yr:.1f}/yr')"
        ),
        md(
            f"> **In plain words:** the timer earns {R['active_mean']:.1f} bps/day vs {R['passive_mean']:.1f} "
            f"for buy-and-hold — a gap of **{R['spread_mean']:+.2f} bps/day** ({R['spread_ann']:.1f}%/yr). "
            f"The spread t-stat is {R['spread_t']:+.2f}: not conventionally significant, but "
            f"consistently negative. With {R['pct_back']:.1f}% of days in backwardation "
            f"(~{round(R['pct_back']/100*252):.0f} days/yr), the timer sits in cash roughly one "
            "day in four — and those days disproportionately include the strongest post-scare "
            "rebounds."
        ),
        md(
            "### 4c · Regime decomposition — the key finding\n\n"
            "Full-sample vs contango vs backwardation daily return statistics. Watch the "
            "*raw mean* and the *Sharpe* diverge — that is the whole story."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rs = st.regime_stats(df)\n"
            "    regime_labels = rs['regime'].tolist()\n"
            "    regime_means = rs['mean_bps'].tolist()\n"
            "    regime_sharpes = rs['sharpe_ann'].tolist()\n"
            "    regime_ns = rs['n'].tolist()\n"
            "else:\n"
            f"    regime_labels = ['Full sample', 'Contango\\n(slope>0)', 'Backwardation\\n(slope<0)']\n"
            f"    regime_means = [{R['passive_mean']:.2f}, {R['mean_contango']:.2f}, {R['mean_back']:.2f}]\n"
            f"    regime_sharpes = [0.77, {R['sharpe_contango']:.2f}, {R['sharpe_back']:.2f}]\n"
            f"    regime_ns = [{R['n']}, {R['n_contango']}, {R['n_back']}]\n"
            "fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "short_labels = [l.split('\\n')[0] for l in regime_labels]\n"
            "cols = [GREY, GREEN, RED]\n"
            "bars = ax.bar(short_labels, regime_means, color=cols, width=0.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('Mean SPY return (bps/day)')\n"
            "ax.set_title('RAW return: backwardation wins')\n"
            "for b, n in zip(bars, regime_ns):\n"
            "    ax.annotate(f'n={n:,}', (b.get_x()+b.get_width()/2, b.get_height()+0.1),\n"
            "                ha='center', va='bottom', fontsize=9)\n"
            "b2 = ax2.bar(short_labels, regime_sharpes, color=cols, width=0.5)\n"
            "ax2.axhline(0, c='k', lw=1); ax2.set_ylabel('Annualised Sharpe')\n"
            "ax2.set_title('Risk-ADJUSTED: contango wins (it is calmer)')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('Contango: mean {R['mean_contango']:+.2f} bps/day, Sharpe {R['sharpe_contango']:.2f}')\n"
            f"print('Backwardation: mean {R['mean_back']:+.2f} bps/day, Sharpe {R['sharpe_back']:.2f}')\n"
            f"print('Raw-return ratio: {R['mean_back']/R['mean_contango']:.2f}x higher in backwardation')"
        ),
        md(
            f"> **In plain words:** here is the honest subtlety. On *raw* return, backwardation "
            f"averages **{R['mean_back']:.1f} bps/day** vs {R['mean_contango']:.1f} for contango — "
            f"the 'stress' regime pays more. But on *Sharpe*, contango wins "
            f"({R['sharpe_contango']:.2f} vs {R['sharpe_back']:.2f}) because backwardation days are "
            "far more volatile. So the slope *does* carry real information — about **volatility**. "
            "The timer, though, harvests *raw* return, and by fleeing the more-volatile "
            "backwardation days it forgoes their higher mean. Pointing a volatility signal at "
            "direction is the category error. The backwardation episodes here (2015, 2018, 2020, "
            "2022, 2025) are followed by rapid policy-driven mean-reversions — the classic "
            "volatility-risk-premium rebound where being *long* risk during stress pays."
        ),

        # ---- BEAT 5 — VERDICT --------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — Q5-Q1 spread {R['spread_1d']:+.2f} bps (wrong sign), HAC *t* "
            f"{R['t_spread_1d']:+.2f} at h=1d; negative at h=5d (*t*={R['t_spread_5d']:+.2f}) and "
            f"h=21d (*t*={R['t_spread_21d']:+.2f}). The slope does not predict forward direction.\n"
            f"- **Tradability `MIRAGE`** — timer trails buy-and-hold {R['spread_mean']:+.2f} bps/day "
            f"({R['spread_ann']:.1f}%/yr) at zero cost; spread t={R['spread_t']:+.2f}; "
            f"~{R['switches_yr']:.0f} switches/yr and no positive break-even switching cost.\n"
            f"- **Beats buy-and-hold? `NO`** — backwardation days ({R['pct_back']:.1f}% of tape) "
            f"earn {R['mean_back']:.1f} bps/day raw vs {R['mean_contango']:.1f} in contango; the timer "
            "sits flat during the higher-raw-return regime."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----------------------------------------
        md(
            "## 6 · Could you trade it? — cost sweep\n\n"
            f"~{R['switches_yr']:.0f} switches/yr — nearly 3× the churn of the 3-month curve — "
            "each a round-trip on the full SPY position:"
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    sw = [st.summarize(st.timing_overlay(df, cost_bps=c)['r_spread']) for c in costs]\n"
            "    spr_means = [s['mean_bps'] for s in sw]\n"
            "    spr_tstats = [s['t_stat'] for s in sw]\n"
            "else:\n"
            f"    spr_means = [-1.64, -1.71, -1.78, -1.93, -2.37]\n"
            f"    spr_tstats = [-1.45, -1.51, -1.58, -1.70, -2.09]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, spr_means, 'o-', c=RED, lw=2, label='spread mean (bps/day)')\n"
            "ax2 = ax.twinx()\n"
            "ax2.plot(costs, spr_tstats, 's--', c=GREY, lw=1.5, label='HAC t-stat')\n"
            "ax2.axhline(-2, ls=':', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Round-trip switching cost (bps)'); ax.set_ylabel('Spread mean (bps/day)', color=RED)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('Gross spread negative — and by 5 bps the underperformance is significant')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('t at 5 bps/switch:', spr_tstats[-1], '(< -2: significantly WORSE than buy-and-hold)')"
        ),
        md(
            "> **In plain words:** unlike a study where a positive gross edge dies at a "
            "finite break-even cost, here the gross is already negative — **there is no "
            "positive break-even cost**. The timer was never competitive with buy-and-hold; "
            "and because the 9-day slope churns so heavily, by ~5 bps/switch the "
            "underperformance is itself statistically significant (*t* ≈ −2.1)."
        ),

        # ---- BEAT 7 — GOING FURTHER ----------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Does the engine *find* a signal when one exists? Plant a synthetic slope-return "
            "relationship and confirm the Q5-Q1 spread turns on:"
        ),
        code(
            "signals = [0.00, 0.02, 0.05, 0.10]\n"
            "spreads_ctrl = []\n"
            "for sig in signals:\n"
            "    df_s, _ = data.synthetic_daily(n_days=3000, contango_signal=sig, seed=767)\n"
            "    res = st.top_minus_bottom(df_s, horizon=1)\n"
            "    spreads_ctrl.append(res['spread_bps'])\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(signals, spreads_ctrl, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('Planted contango signal strength')\n"
            "ax.set_ylabel('Q5-Q1 spread (bps/day)')\n"
            "ax.set_title('The engine is a faithful signal detector — the real market has nothing to find')\n"
            "plt.tight_layout(); plt.show()\n"
            "for sig, spr in zip(signals, spreads_ctrl):\n"
            "    print(f'signal={sig:.2f}: Q5-Q1 spread = {spr:+.2f} bps/day')"
        ),
        md(
            "The Q5-Q1 spread rises monotonically with planted signal strength and is near "
            "zero at the null — confirming the quintile engine is a faithful detector. The "
            "real-tape verdict is therefore a statement about the **market**: the ^VIX9D/^VIX "
            "short-end slope contains no exploitable forward-*direction* information.\n\n"
            "**Forks worth trying:**\n"
            "- The 3-month cousin ([Study 111 — VIX-Term-Structure](../../111-vix-term-structure/)): "
            "same verdict, a third of the switching — strictly less bad to trade.\n"
            "- Trading the *volatility* the slope actually forecasts (variance swaps, VIX-ETP "
            "roll) rather than equity direction.\n"
            "- Conditioning on the *change* in slope rather than its level: a sudden inversion "
            "may behave differently from a persistently inverted front end.\n"
            "- VIX *level* mean-reversion timing (VIX > 30 → better forward returns): a "
            "different and better-documented effect."
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
