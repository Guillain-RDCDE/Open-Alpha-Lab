"""Generate the two narrative notebooks for Study 111 (VIX-Term-Structure).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-13).
R = dict(
    n=4640,
    date_start="2008-01-03",
    date_end="2026-06-12",
    fp="e78f42c8162e",
    frac_contango=0.897,
    # Q5-Q1 spread
    spread_1d=-0.98, t_spread_1d=-0.08,
    spread_5d=-11.61, t_spread_5d=-0.36,
    spread_21d=-47.25, t_spread_21d=-0.47,
    # Quintile means h=1
    q1_mean=3.23, q1_t=0.61,
    q2_mean=5.12, q2_t=1.20,
    q3_mean=9.76, q3_t=2.91,
    q4_mean=2.81, q4_t=1.01,
    q5_mean=2.25, q5_t=1.08,
    # Timing overlay
    active_mean=3.20, active_sharpe=0.57, active_t=2.50,
    passive_mean=4.25, passive_sharpe=0.54, passive_t=2.64,
    spread_mean=-1.05, spread_t=-1.01,
    switches_yr=14.4,
    # Regime stats
    n_contango=4159, pct_contango=89.7, mean_contango=3.57, sharpe_contango=0.60,
    n_back=479, pct_back=10.3, mean_back=10.64, sharpe_back=0.62,
    # Annualised spread
    spread_ann=-2.6,
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

from vix_term_structure import data, strategy as st

# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-13).
R = dict(
    n=4640, date_start="2008-01-03", date_end="2026-06-12", fp="e78f42c8162e",
    frac_contango=0.897,
    spread_1d=-0.98, t_spread_1d=-0.08,
    spread_5d=-11.61, t_spread_5d=-0.36,
    spread_21d=-47.25, t_spread_21d=-0.47,
    q1_mean=3.23, q1_t=0.61, q2_mean=5.12, q2_t=1.20,
    q3_mean=9.76, q3_t=2.91, q4_mean=2.81, q4_t=1.01, q5_mean=2.25, q5_t=1.08,
    active_mean=3.20, active_sharpe=0.57, active_t=2.50,
    passive_mean=4.25, passive_sharpe=0.54, passive_t=2.64,
    spread_mean=-1.05, spread_t=-1.01, switches_yr=14.4, spread_ann=-2.6,
    n_contango=4159, pct_contango=89.7, mean_contango=3.57, sharpe_contango=0.60,
    n_back=479, pct_back=10.3, mean_back=10.64, sharpe_back=0.62,
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
            "# VIX-Term-Structure — does the VIX curve predict market direction?\n"
            "### The contango/backwardation slope as a market-timing signal, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_buy-and-hold%3F: No](https://img.shields.io/badge/Beats_buy--and--hold%3F-No-8b949e?style=flat-square)\n\n"
            "Every market day, two VIX numbers flash on Bloomberg terminals: **^VIX** "
            "(30-day implied vol) and **^VIX3M** (93-day implied vol). When VIX < VIX3M the "
            "curve is in *contango* — the market expects volatility to remain elevated longer, "
            "which is said to be risk-on. When VIX > VIX3M the curve *inverts* — the market is "
            "pricing in acute near-term stress, the classic risk-off alert. The ratio "
            "log(VIX3M / VIX) is widely tracked as a clean, continuous signal. This notebook "
            "asks the only question that matters: **does it actually predict which way the S&P "
            "500 will go?**\n\n"
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
            "| Does a timing rule (long in contango, flat in backwardation) beat buy-and-hold? | "
            f"**No.** The timer trails by **{R['spread_ann']:.1f}%/yr** at zero cost. |\n"
            "| Is backwardation actually dangerous for equities? | **The opposite.** "
            f"The {R['pct_back']:.1f}% of days in backwardation average "
            f"**{R['mean_back']:.1f} bps/day** — nearly 3x the contango mean of "
            f"{R['mean_contango']:.1f} bps/day. |\n\n"
            "> The slope tells you what the options market thinks about volatility. "
            "It does **not** tell you which way the stock market will go tomorrow."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *'When the VIX curve is in contango it's a green light — buy the dip, be long. "
            "When it inverts into backwardation, something is breaking. Get defensive, reduce "
            "equity exposure. The term structure has been a reliable stress indicator since 2008.'*\n\n"
            "It is a reasonable-sounding idea grounded in real institutional behaviour. The "
            "short end of the VIX curve spikes in crises because short-dated options are the "
            "cheapest tail hedge, so a spike above the 3-month level really does reflect "
            "acute demand for protection. The slope is real information — the question is "
            "whether it *predicts* subsequent equity returns or just reflects what is already "
            "priced in."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the claim were true, the VIX term structure would be a near-free market-timing "
            "signal available from any charting platform with zero data cost. If it is false, "
            "a large number of practitioners are reducing equity exposure exactly when the "
            "subsequent returns turn out to be highest — paying the opportunity cost of "
            "missing post-crisis bounces. The stakes are high enough to warrant an honest "
            "18-year test."
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
            "Tape: ^VIX, ^VIX3M, SPY daily, January 2008 (^VIX3M launch) through June 2026 "
            f"— {R['n']:,} observations."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, what the VIX term structure looks like in practice:**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)\n"
            "    axes[0].plot(df.index, df['VIX'], label='^VIX', color=RED, lw=1, alpha=0.8)\n"
            "    axes[0].plot(df.index, df['VIX3M'], label='^VIX3M', color=AMBER, lw=1, alpha=0.8)\n"
            "    axes[0].set_ylabel('VIX index level'); axes[0].legend(); axes[0].set_title('^VIX vs ^VIX3M (2008–2026)')\n"
            "    axes[1].fill_between(df.index, df['slope'], 0,\n"
            "        where=(df['slope'] >= 0), color=GREEN, alpha=0.4, label='Contango (slope>0)')\n"
            "    axes[1].fill_between(df.index, df['slope'], 0,\n"
            "        where=(df['slope'] < 0), color=RED, alpha=0.6, label='Backwardation (slope<0)')\n"
            "    axes[1].set_ylabel('slope = log(VIX3M/VIX)'); axes[1].legend()\n"
            "    axes[1].set_title(f'Term-structure slope — {R[\"frac_contango\"]*100:.0f}% of time in contango')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print(f'Real tape not cached. Headlines: N={R[\"n\"]}, {R[\"date_start\"]} to {R[\"date_end\"]}')\n"
            "    print('Fraction in contango: " + f"{round(R['frac_contango']*100)}" + "%')"
        ),
        md(
            f"The market is in contango **{R['frac_contango']*100:.0f}%** of the time. "
            "Backwardation is rare, concentrated in the big crises: 2008–09 financial crisis, "
            "2011 Euro stress, 2020 COVID crash, 2022 rate shock. These are the episodes the "
            "signal says to 'go defensive' — but they are also followed by some of the "
            "strongest recoveries."
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
            "colors = [RED if m < 0 else GREEN for m in means]\n"
            "bars = ax.bar([f'Q{i}' for i in range(1,6)], means, color=[RED,AMBER,GREEN,AMBER,RED])\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b, t in zip(bars, tstats):\n"
            "    ax.annotate(f't={t:+.1f}', (b.get_x()+b.get_width()/2, b.get_height()),\n"
            "                ha='center', va='bottom', fontsize=9)\n"
            "ax.set_xlabel('Slope quintile (Q1=deepest backwardation, Q5=steepest contango)')\n"
            "ax.set_ylabel('Next-day SPY mean return (bps)')\n"
            "ax.set_title('No monotone ascent from Q1 to Q5 — the claim fails')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Predicted pattern: Q1 < Q2 < Q3 < Q4 < Q5')\n"
            "print(f'Actual: Q1={means[0]:.1f} Q2={means[1]:.1f} Q3={means[2]:.1f} Q4={means[3]:.1f} Q5={means[4]:.1f}')"
        ),
        md(
            f"There is no monotone ascent. Q5 (steepest contango) earns just "
            f"**{R['q5_mean']:+.1f} bps/day** — the *second lowest* of the five buckets. "
            f"Q1 (deepest backwardation) earns {R['q1_mean']:+.1f} bps — more than Q5. The "
            "Q5-Q1 spread is "
            f"**{R['spread_1d']:+.2f} bps** with HAC t = **{R['t_spread_1d']:+.2f}**: "
            "consistent with zero and pointing in the wrong direction."
        ),
        md(
            "**The timing overlay — long in contango, flat when backwardated:**"
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
            "backwardation episodes the timer classifies as 'danger'. By sitting in cash during "
            "those periods, the timer misses the very returns it was supposed to protect against "
            "losing."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Q5-Q1 spread is {R['spread_1d']:+.2f} bps (wrong sign), "
            f"HAC *t* = {R['t_spread_1d']:+.2f} at 1 day; negative at all horizons tested. "
            "The VIX term structure slope does not forecast SPY returns.\n"
            f"- **Tradability — Mirage.** The timer trails buy-and-hold by {abs(R['spread_ann']):.1f}%/yr "
            f"at zero cost; ~{R['switches_yr']:.0f} switches/yr mean every bp of friction widens the gap.\n"
            f"- **Beats buy-and-hold? — No.** Backwardation days average {R['mean_back']:.1f} bps/day "
            f"(nearly 3x the contango mean of {R['mean_contango']:.1f} bps/day) — the market "
            "rebounds hardest from the exact episodes the slope flags as most dangerous."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Even if the spread were positive, the turnover arithmetic would matter. The slope "
            "switches regimes ~14 times per year; each switch is a round-trip trade on the full "
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
            f"    spreads = [{R['spread_mean']}, {R['spread_mean']}-0.02, {R['spread_mean']}-0.05, "
            f"{R['spread_mean']}-0.11, {R['spread_mean']}-0.28]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "ax.plot(costs, spreads, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, spreads, 0, where=[s<0 for s in spreads], color=RED, alpha=.12)\n"
            "ax.set_xlabel('Round-trip cost per switch (bps)'); ax.set_ylabel('Spread (bps/day)')\n"
            "ax.set_title('Gross spread is already negative — costs make it worse')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The slope timer underperforms buy-and-hold before costs; costs compound the problem.')"
        ),
        md(
            "The line starts below zero and only falls. Because the gross spread is already "
            "negative, **there is no switching cost low enough to make the timer competitive** "
            "with buying and holding. This is the strongest version of the 'Mirage' verdict: "
            "not 'dies at costs' but 'never lived'."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Is the slope useful for anything?** The desk's [Study 92 — Easy-Money]"
            "(../../92-easy-money/) shows that *shorting* the volatility ETP in contango "
            "(harvesting the VIX futures term premium) does generate a real gross edge — "
            "but via a completely different mechanism (roll yield on the ETP, not equity "
            "direction timing).\n"
            "- **Does VIX level (not slope) work?** [Study 86 — Tail-Radar](../../86-tail-radar/) "
            "finds the same story for the CBOE SKEW index: a valid volatility-surface "
            "signal with no equity-return forecasting power.\n"
            "- **What about VIX-level mean reversion?** VIX above 30 is statistically "
            "followed by higher SPY returns (the Connors/Alvarez literature), but that is "
            "the *level* of fear, not the slope between maturities — a separate claim.\n\n"
            "*Convinced the slope works on a different horizon or instrument? Fork this, "
            "change the parameters, and show a positive Q5-Q1 spread that clears the |t| ≥ 2 "
            "bar. That's the standard.*"
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
            "# VIX-Term-Structure — a quantitative teardown\n"
            "### Daily tape · rolling quintile sort · binary timing overlay · HAC inference · regime breakdown\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_buy-and-hold%3F: No](https://img.shields.io/badge/Beats_buy--and--hold%3F-No-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "Same seven beats, every claim now carrying its standard error. We test whether "
            "the VIX/VIX3M slope (log(^VIX3M / ^VIX)) predicts SPY forward returns via a "
            "rolling quintile sort, a binary timing overlay, and a regime decomposition, "
            "over the full 2008–2026 daily tape.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, 2008-01-03 through "
            "2026-06-12 (^VIX3M launch to present). Methods & sources in "
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
            f"HAC *t* = **{R['t_spread_1d']:+.2f}** at h=1d; negative at all horizons. |\n"
            f"| **Tradability** | `MIRAGE` | Timer trails buy-and-hold **{abs(R['spread_ann']):.1f}%/yr** at "
            f"zero cost; spread *t* = **{R['spread_t']:+.2f}**, no positive break-even. |\n"
            f"| **Beats buy-and-hold?** | `NO` | Backwardation ({R['pct_back']:.1f}% of tape) "
            f"averages **{R['mean_back']:.1f} bps/day** vs contango's {R['mean_contango']:.1f} — "
            "the claimed danger signal is followed by higher returns. |\n\n"
            "> **In plain words:** the slope is real information about what the options market "
            "thinks, but the claim that contango predicts good equity returns is backwards on "
            "this tape. Backwardation episodes (where the timer goes flat) produce the highest "
            "average returns."
        ),

        # ---- BEAT 1 — STEELMANNED CLAIM ----------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $s_t = \\log(\\text{VIX3M}_t / \\text{VIX}_t)$ be the slope on day $t$ and "
            "$r_{t+1:t+h}$ the h-day forward SPY log-return. The term-structure timer asserts:\n\n"
            "- **H₁ (quintile forecast).** $\\mathbb{E}[r_{t+1} \\mid s_t \\in Q5] > "
            "\\mathbb{E}[r_{t+1} \\mid s_t \\in Q1]$ — steeper contango predicts better "
            "1-day (and multi-day) forward returns.\n"
            "- **H₂ (timing overlay).** A binary rule $w_t = \\mathbf{1}\\{s_t > 0\\}$ produces "
            "higher risk-adjusted return than unconditional $w_t \\equiv 1$ (buy-and-hold).\n"
            "- **H₃ (regime dominance).** Conditioning on contango delivers materially "
            "higher Sharpe than conditioning on backwardation.\n\n"
            "We reject H₁ (spread is negative, wrong sign), H₂ (timer underperforms), and "
            "H₃ (backwardation sub-sample delivers higher per-day returns)."
        ),

        # ---- BEAT 2 — SO WHAT --------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "The VIX term structure is cited in practitioner research and widely used by "
            "retail vol-traders as a regime filter. The stakes: a practitioner who reduces "
            "equity exposure in backwardation misses the post-crisis rebounds on this 18-year "
            "tape. An honest test resolves whether the signal is actionable or a case of "
            "economically-intuitive-but-empirically-absent forecasting."
        ),

        # ---- BEAT 3 — PROTOCOL -------------------------------------------------
        md(
            "## 3 · The protocol\n\n"
            "- **Signal.** $s_t = \\log(\\text{VIX3M}_t / \\text{VIX}_t)$, computed "
            "from Yahoo daily closes. Quintile rank: 252-day rolling window, out-of-sample "
            "(min 63 observations before first rank).\n"
            "- **Forward return.** Log-return from close $t$ to close $t+h$ for "
            "$h \\in \\{1, 5, 21\\}$ business days, aligned at day $t$.\n"
            "- **Spread test.** Construct the Q5 − Q1 spread series (sign-adjusted: "
            "$+r_{t+1}$ for Q5, $-r_{t+1}$ for Q1) and apply Newey-West HAC t-stat.\n"
            "- **Timer.** Binary weight $w_t = \\mathbf{1}\\{s_{t-1} > 0\\}$ (the prior "
            "day's slope, no look-ahead); spread $= r_{active} - r_{passive}$ per day.\n"
            "- **Inference.** Newey-West HAC t-stat throughout; block-bootstrap Sharpe CI "
            "on the active arm.\n"
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
            f"        5: [23.4, 8.7, 50.7, 18.9, 11.8],\n"
            f"        21: [127.6, 81.1, 111.5, 67.4, 80.4]}},\n"
            f"        index=[1,2,3,4,5])\n"
            f"    tpivot = pd.DataFrame({{\n"
            f"        1: [{R['q1_t']},{R['q2_t']},{R['q3_t']},{R['q4_t']},{R['q5_t']}],\n"
            f"        5: [1.06, 0.52, 4.97, 2.09, 1.49],\n"
            f"        21: [2.58, 1.82, 3.81, 2.63, 3.41]}},\n"
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
            "with zero and pointing in the wrong direction. Q3 (the near-flat transition zone) "
            "shows the highest mean at h=5d and h=21d, likely reflecting the equity drift in "
            "the post-2009 bull market that concentrated there by coincidence."
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
            f"The spread t-stat is {R['spread_t']:+.2f}: not significant, but consistently negative. "
            f"With only {R['pct_back']:.1f}% of days in backwardation (~{round(R['pct_back']/100*252):.0f} days/yr), "
            "the timer sits in cash for ~47 days/year — and those days disproportionately "
            "include the strongest post-crisis rebounds."
        ),
        md(
            "### 4c · Regime decomposition — the key finding\n\n"
            "Full-sample vs contango vs backwardation daily return statistics."
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
            f"    regime_sharpes = [0.55, {R['sharpe_contango']:.2f}, {R['sharpe_back']:.2f}]\n"
            f"    regime_ns = [{R['n']}, {R['n_contango']}, {R['n_back']}]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "short_labels = [l.split('\\n')[0] for l in regime_labels]\n"
            "cols = [GREY, GREEN, RED]\n"
            "bars = ax.bar(short_labels, regime_means, color=cols, width=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean SPY return (bps/day)')\n"
            "ax.set_title('Backwardation days average HIGHER returns than contango days')\n"
            "for b, sh, n in zip(bars, regime_sharpes, regime_ns):\n"
            "    ax.annotate(f'n={n:,}\\nSharpe {sh:+.2f}', (b.get_x()+b.get_width()/2, b.get_height()+0.1),\n"
            "                ha='center', va='bottom', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('Contango mean: {R['mean_contango']:+.2f} bps/day | Backwardation mean: {R['mean_back']:+.2f} bps/day')\n"
            f"print('Ratio: {R['mean_back']/R['mean_contango']:.1f}x higher returns in backwardation')"
        ),
        md(
            f"> **In plain words:** backwardation episodes average **{R['mean_back']:.1f} bps/day** "
            f"vs {R['mean_contango']:.1f} for contango — the 'stress' regime delivers nearly 3x the "
            "daily return. The difference is not statistically significant (small backwardation "
            "sample), but the direction is robustly wrong. The most likely explanation: the "
            "backwardation episodes in this sample (2008-09, 2020, 2022) are followed by "
            "rapid mean-reversions driven by policy responses — classic volatility-risk-premium "
            "episodes where being *long* risk during stress pays handsomely."
        ),

        # ---- BEAT 5 — VERDICT --------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — Q5-Q1 spread {R['spread_1d']:+.2f} bps (wrong sign), HAC *t* "
            f"{R['t_spread_1d']:+.2f} at h=1d; negative at h=5d (*t*={R['t_spread_5d']:+.2f}) and "
            f"h=21d (*t*={R['t_spread_21d']:+.2f}). The slope does not predict forward returns.\n"
            f"- **Tradability `MIRAGE`** — timer trails buy-and-hold {R['spread_mean']:+.2f} bps/day "
            f"({R['spread_ann']:.1f}%/yr) at zero cost; spread t={R['spread_t']:+.2f}; no positive "
            "break-even switching cost.\n"
            f"- **Beats buy-and-hold? `NO`** — backwardation days ({R['pct_back']:.1f}% of tape) "
            f"earn {R['mean_back']:.1f} bps/day vs {R['mean_contango']:.1f} in contango; the timer "
            "sits flat during the highest-returning regime."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----------------------------------------
        md(
            "## 6 · Could you trade it? — cost sweep\n\n"
            f"~{R['switches_yr']:.0f} switches/yr at ~14 bps each side in SPY (spread + commissions):"
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    sw = [st.summarize(st.timing_overlay(df, cost_bps=c)['r_spread']) for c in costs]\n"
            "    spr_means = [s['mean_bps'] for s in sw]\n"
            "    spr_tstats = [s['t_stat'] for s in sw]\n"
            "else:\n"
            f"    spr_means = [{R['spread_mean']:.3f}, {R['spread_mean']:.3f}-0.02, "
            f"{R['spread_mean']:.3f}-0.05, {R['spread_mean']:.3f}-0.11, {R['spread_mean']:.3f}-0.28]\n"
            f"    spr_tstats = [{R['spread_t']:.2f}, {R['spread_t']:.2f}-0.03, "
            f"{R['spread_t']:.2f}-0.05, {R['spread_t']:.2f}-0.11, {R['spread_t']:.2f}-0.27]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, spr_means, 'o-', c=RED, lw=2, label='spread mean (bps/day)')\n"
            "ax2 = ax.twinx()\n"
            "ax2.plot(costs, spr_tstats, 's--', c=GREY, lw=1.5, label='HAC t-stat')\n"
            "ax2.axhline(-2, ls=':', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Round-trip switching cost (bps)'); ax.set_ylabel('Spread mean (bps/day)', color=RED)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('Gross spread negative — no break-even cost exists')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "> **In plain words:** unlike a study where a positive gross edge dies at a "
            "finite break-even cost, here the gross is already negative — **there is no "
            "positive break-even cost**. The timer was never competitive with buy-and-hold; "
            "switching costs make it strictly worse."
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
            "    df_s, _ = data.synthetic_daily(n_days=3000, contango_signal=sig, seed=111)\n"
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
            "real-tape verdict is therefore a statement about the **market**: the VIX/VIX3M "
            "slope contains no exploitable forward-return information.\n\n"
            "**Forks worth trying:**\n"
            "- Shorting volatility ETPs in contango ([Study 92 — Easy-Money](../../92-easy-money/)): "
            "the same term-structure information harvested via roll yield, not equity direction.\n"
            "- VIX *level* mean-reversion timing (VIX > 30 → better forward returns): a "
            "different and better-documented effect.\n"
            "- Conditioning on the *change* in slope rather than its level: stress *increases* "
            "may predict reversals even if the level does not."
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
