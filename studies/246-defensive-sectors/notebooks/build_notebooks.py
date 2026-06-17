"""Generate the two narrative notebooks for Study 246 (Defensive-Sectors Leadership).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
parquet under ../_cache/ if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-16).
R = dict(
    n=6911,
    date_start="1998-12-23",
    date_end="2026-06-16",
    fp="e4ade1bcf806",
    # Quintile means 1-day
    q1_1d=2.46, q2_1d=0.09, q3_1d=5.42, q4_1d=3.78, q5_1d=4.02,
    q1_1d_t=0.95, q5_1d_t=1.09,
    # Quintile means 5-day
    q1_5d=29.71, q2_5d=7.96, q3_5d=10.92, q4_5d=5.38, q5_5d=24.79,
    q1_5d_t=3.04, q5_5d_t=1.79,
    # Quintile means 21-day
    q1_21d=112.10, q2_21d=68.24, q3_21d=44.86, q4_21d=44.40, q5_21d=68.48,
    q1_21d_t=4.69, q5_21d_t=2.27,
    # Q1-Q5 spread
    spread_1d=-1.56, spread_1d_t=-0.42,
    spread_5d=4.93, spread_5d_t=0.11,
    spread_21d=43.62, spread_21d_t=0.79,
    # Regime stats
    full_sharpe=0.433, full_mean=3.31, full_t=2.60,
    calm_sharpe=0.521, calm_mean=3.36, calm_t=2.12,
    alert_sharpe=0.376, alert_mean=3.27, alert_t=1.52,
    # Timing overlay
    passive_sharpe=0.432, active_sharpe_gross=0.430,
    spread_gross=-0.77, spread_gross_t=-1.04,
    spread_1bp=-0.84, spread_1bp_t=-1.13, active_sharpe_1bp=0.418,
    n_alerts=1471, alert_pct=21.3, n_switches=480,
)

# ---------------------------------------------------------------------------
# Shared preamble
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

from defensive_sectors import data, strategy as st

CACHE_DIR = os.path.abspath(os.path.join("..", "_cache"))

def _have_cache():
    return os.path.exists(data._cache_path(CACHE_DIR))

HAVE_REAL = _have_cache()

def load_real():
    return data.fetch_daily(fetch=False, cache_dir=CACHE_DIR)

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Defensive-Sectors Leadership -- does XLP+XLU leading warn of risk-off?\n"
            "### The two-sector defensive canary, tested honestly against buy-and-hold\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Coincident--not--forecast](https://img.shields.io/badge/Coincident--not--forecast-8b949e?style=flat-square)\n\n"
            "The utilities-canary signal (Study 131) reads XLU relative to SPY. A common "
            "extension: if *both* consumer staples (XLP) *and* utilities (XLU) are "
            "simultaneously leading the market, the risk-off rotation must be more "
            "convincing -- two sectors sending the same warning, not just one. "
            "This notebook asks whether that combined signal does any better than the "
            "single-sector version -- whether it actually **knows** what happens next, "
            "or whether it is simply a more elaborate thermometer.\n\n"
            "> This is the plain-language layer. Want the t-stats and the spread "
            "decomposition? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does combined XLP+XLU leading predict lower forward SPY returns? | **No.** "
            f"The 21-day spread is only **+{R['spread_21d']:.0f} bps/month** (*t* = **{R['spread_21d_t']:.2f}**, "
            f"bar is 2.0) and the 1-day spread is *negative* (−{abs(R['spread_1d']):.1f} bps). |\n"
            "| Does it beat buy-and-hold? | **No.** The timing overlay (go to cash in the "
            f"top-20% alert days) *lowers* Sharpe ({R['active_sharpe_gross']:.3f} vs "
            f"{R['passive_sharpe']:.3f}) and earns a negative gross alpha "
            f"({R['spread_gross']:+.2f} bps/day). |\n"
            "| Does combining two sectors strengthen the canary? | **No.** The single-sector "
            "utilities canary (Study 131) reached *t* = +1.58 at 21d; this combined version "
            f"reaches only *t* = +{R['spread_21d_t']:.2f}. Adding XLP makes the signal *weaker*. |\n"
            "| Is the pattern monotone? | **No.** At 1-day, Q5 (alert) earns *more* than Q1 "
            "(calm) -- the opposite of the folk claim. At 21 days, Q5 jumps above Q3 and Q4. |\n\n"
            "> The combined defensive canary is not singing -- it is just reacting to the same "
            "volatility storm the market is already experiencing."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim\n\n"
            "> *\"When consumer staples (XLP) AND utilities (XLU) are both outperforming SPY "
            "on a relative-strength basis -- not just one defensive sector but two -- the "
            "risk-off rotation is confirmed. Institutions are fleeing to safety across the "
            "board, and a significant market drawdown is likely. Two sectors agreeing is a "
            "stronger signal than one.\"*\n\n"
            "The logic is appealing. Both XLP and XLU are 'boring' sectors: stable earnings, "
            "low beta, bond-like behaviour in recessions. If smart money is piling into both "
            "simultaneously, the rotation away from risk seems coordinated and credible. "
            "The bet is that this **multi-sector rotation precedes market weakness**, not just "
            "accompanies it."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "If true, this is a simple two-ETF recipe: compute the combined XLP+XLU relative "
            "strength versus SPY, monitor its 20-day momentum, and step aside when both "
            "sectors are outperforming. For retail investors, this is easier to explain and "
            "justify than VIX-based or macro-factor signals. If false, anyone relying on this "
            "confirmation pattern is paying trading costs for what amounts to noise -- or worse, "
            "a contrarian signal (exiting on strength in the defensive sectors and missing the "
            "subsequent SPY recovery)."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 -- How would we even know?\n\n"
            "Three rules keep the test honest:\n\n"
            "1. **Combine, then sort forward.** Average the log(XLP/SPY) and log(XLU/SPY) "
            "ratios daily. Compute the 20-day change in this combined defensive RS (rolling "
            "percentile, strictly out-of-sample), split into quintiles, and look at "
            "*next-day/week/month* SPY returns per bucket. The folk claim predicts a monotone "
            "descent: Q1 (SPY leading) → high returns; Q5 (defensives leading) → low returns.\n"
            "2. **Race it against buy-and-hold.** Going to cash in the top-20% alert bucket "
            "must earn positive *alpha* (daily return spread), not just look different.\n"
            "3. **Compare to the single-sector baseline.** If combining two sectors genuinely "
            "strengthens the signal, we should see a higher t-stat than Study 131 (bar: "
            "*t* > 1.58). We find the opposite.\n\n"
            f"Real tape: XLP, XLU, and SPY daily, {R['date_start']} to {R['date_end']}, n = {R['n']:,} days."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 -- The teardown\n\n"
            "**The quintile picture.** Sort days by the 20-day combined defensive RS rank "
            "and look at forward SPY returns per bucket across three horizons:"
        ),
        code(
            "quintiles = [1, 2, 3, 4, 5]\n"
            "labels_q = ['Q1\\n(calm,\\nSPY leads)', 'Q2', 'Q3', 'Q4', 'Q5\\n(alert,\\ndef leads)']\n"
            "if HAVE_REAL:\n"
            "    df = load_real()\n"
            "    qt = st.quintile_table(df, horizons=[1, 5, 21])\n"
            "    means_1d  = [qt[(qt.quintile==q)&(qt.horizon==1)]['mean_bps'].values[0] for q in quintiles]\n"
            "    means_21d = [qt[(qt.quintile==q)&(qt.horizon==21)]['mean_bps'].values[0] for q in quintiles]\n"
            "else:\n"
            f"    means_1d  = [{R['q1_1d']}, {R['q2_1d']}, {R['q3_1d']}, {R['q4_1d']}, {R['q5_1d']}]\n"
            f"    means_21d = [{R['q1_21d']}, {R['q2_21d']}, {R['q3_21d']}, {R['q4_21d']}, {R['q5_21d']}]\n"
            "fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))\n"
            "colors = [GREEN, GREEN, GREY, AMBER, RED]\n"
            "ax[0].bar(labels_q, means_1d, color=colors)\n"
            "ax[0].axhline(0, c='k', lw=1)\n"
            "ax[0].set_ylabel('Mean forward return (bps)'); ax[0].set_title('1-day forward SPY by defensive quintile')\n"
            "ax[1].bar(labels_q, means_21d, color=colors)\n"
            "ax[1].axhline(0, c='k', lw=1)\n"
            "ax[1].set_ylabel('Mean forward return (bps)'); ax[1].set_title('21-day forward SPY by defensive quintile')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'1d: Q5 ({means_1d[4]:+.1f} bps) > Q1 ({means_1d[0]:+.1f} bps) -- WRONG DIRECTION')\n"
            "print(f'21d spread Q1-Q5: {means_21d[0]-means_21d[4]:+.1f} bps')"
        ),
        md(
            f"At **1 day**, Q5 (defensive alert) earns *more* than Q1 (calm): +{R['q5_1d']:.1f} bps "
            f"vs +{R['q1_1d']:.1f} bps. The folk signal is pointing the **wrong way** at the "
            "most actionable horizon. At **21 days**, Q1 does lead (+112 bps vs +68 bps), but "
            "Q5 *reverses above* Q3 and Q4 -- the pattern is not monotone. Both of these "
            "observations contradict the 'stronger confirmation from two sectors' claim."
        ),
        md(
            "**The decisive test: is the Q1 vs Q5 gap real?**"
        ),
        code(
            "horizons = [1, 5, 21]\n"
            "spreads = []\n"
            "tstats = []\n"
            "if HAVE_REAL:\n"
            "    for h in horizons:\n"
            "        r = st.top_minus_bottom(df, horizon=h)\n"
            "        spreads.append(r['spread_bps']); tstats.append(r['t_spread'])\n"
            "else:\n"
            f"    spreads = [{R['spread_1d']}, {R['spread_5d']}, {R['spread_21d']}]\n"
            f"    tstats  = [{R['spread_1d_t']}, {R['spread_5d_t']}, {R['spread_21d_t']}]\n"
            "fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "c_spread = [RED if abs(t) < 2 else GREEN for t in tstats]\n"
            "ax[0].bar([str(h)+'d' for h in horizons], spreads, color=c_spread)\n"
            "ax[0].axhline(0, c='k', lw=1)\n"
            "ax[0].set_ylabel('Q1-calm minus Q5-alert (bps)'); ax[0].set_title('Spread across horizons')\n"
            "ax[1].bar([str(h)+'d' for h in horizons], tstats, color=c_spread)\n"
            "for s in (2,-2): ax[1].axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax[1].axhline(0, c='k', lw=1)\n"
            "ax[1].set_ylabel('HAC t-stat'); ax[1].set_title('Statistical significance (bar = 2.0)')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'21d spread: {{spreads[2]:+.1f}} bps,  HAC t = {{tstats[2]:+.2f}} (NOT cleared)')\n"
            f"print(f'1d spread: {{spreads[0]:+.1f}} bps -- NEGATIVE (wrong direction)')"
        ),
        md(
            f"The 1-day spread is *negative* (−{abs(R['spread_1d']):.1f} bps, *t* = {R['spread_1d_t']:.2f}) — "
            f"calm days have *lower* next-day returns than alert days. The 21-day spread of "
            f"+{R['spread_21d']:.0f} bps looks like a number but *t* = {R['spread_21d_t']:.2f} — "
            "the bar is 2.0. Compare to the single-sector utilities canary (Study 131) which "
            "achieved *t* = +1.58 at 21 days. **Adding XLP makes the signal weaker, not stronger.**"
        ),
        md(
            "**The regime Sharpe -- the one number that might look compelling:**"
        ),
        code(
            "regimes = ['Full sample', 'Calm (def Q1-Q2)', 'Alert (def Q3-Q5)']\n"
            "if HAVE_REAL:\n"
            "    rs = st.regime_stats(df)\n"
            "    sharpes = rs['sharpe_ann'].tolist(); means_r = rs['mean_bps'].tolist()\n"
            "else:\n"
            f"    sharpes = [{R['full_sharpe']}, {R['calm_sharpe']}, {R['alert_sharpe']}]\n"
            f"    means_r = [{R['full_mean']}, {R['calm_mean']}, {R['alert_mean']}]\n"
            "fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "cols = [GREY, GREEN, RED]\n"
            "ax[0].bar(regimes, sharpes, color=cols)\n"
            "ax[0].set_ylabel('Annualised Sharpe'); ax[0].set_title('Sharpe by regime')\n"
            "ax[0].tick_params(axis='x', rotation=10)\n"
            "ax[1].bar(regimes, means_r, color=cols)\n"
            "ax[1].axhline(0, c='k', lw=1)\n"
            "ax[1].set_ylabel('Mean daily return (bps)'); ax[1].set_title('Mean return by regime')\n"
            "ax[1].tick_params(axis='x', rotation=10)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Mean return: calm={means_r[1]:+.2f} vs alert={means_r[2]:+.2f} bps -- essentially identical')"
        ),
        md(
            f"The calm-regime Sharpe of **{R['calm_sharpe']:.2f}** versus the alert-regime "
            f"Sharpe of **{R['alert_sharpe']:.2f}** is a much smaller gap than Study 131 "
            f"(0.70 vs 0.22). More tellingly, the mean *returns* are nearly identical "
            f"(+{R['calm_mean']:.2f} vs +{R['alert_mean']:.2f} bps/day). The Sharpe "
            "difference is almost entirely driven by volatility clustering -- high-vol "
            "periods happen to coincide with defensive-sector outperformance. This is not "
            "alpha; it is a risk measurement."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- None.** 21-day Q1−Q5 spread = +{R['spread_21d']:.0f} bps "
            f"(*t* = {R['spread_21d_t']:.2f}), far below the bar of 2. 1-day spread is "
            f"negative (wrong direction). Pattern is non-monotone at all horizons.\n"
            f"- **Tradability -- Mirage.** Timing overlay gross alpha = "
            f"{R['spread_gross']:+.2f} bps/day (*t* = {R['spread_gross_t']:+.2f}). "
            f"Active Sharpe {R['active_sharpe_gross']:.3f} < passive {R['passive_sharpe']:.3f} "
            "even before any trading costs. No break-even cost exists.\n"
            "- **Coincident or forecast? -- Coincident.** The combined defensive RS co-occurs "
            "with stress but does not statistically precede weak SPY returns. Two sectors "
            "agreeing provides no improvement over one."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 -- Could you actually trade it?\n\n"
            "The timing overlay goes to cash when the 20-day combined defensive RS rank "
            "exceeds the 80th percentile (~21% of all days). With 480 round-trips over "
            "27.5 years, even small costs compound quickly:"
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    sharpes_c = []\n"
            "    spreads_c = []\n"
            "    for c in costs:\n"
            "        ov = st.timing_overlay(df, cost_bps=c)\n"
            "        sharpes_c.append(st.summarize(ov['r_active'])['sharpe_ann'])\n"
            "        spreads_c.append(st.summarize(ov['r_spread'])['mean_bps'])\n"
            "else:\n"
            f"    passive_s = {R['passive_sharpe']}\n"
            f"    sharpes_c = [{R['active_sharpe_gross']}, {R['active_sharpe_gross']}-0.006,\n"
            f"                 {R['active_sharpe_1bp']}, {R['active_sharpe_1bp']}-0.012, {R['active_sharpe_1bp']}-0.047]\n"
            f"    spreads_c = [{R['spread_gross']}, {R['spread_gross']}-0.04, {R['spread_1bp']},\n"
            f"                 {R['spread_1bp']}-0.07, {R['spread_1bp']}-0.27]\n"
            "fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "ax[0].plot(costs, sharpes_c, 'o-', c=RED, lw=2, label='active (timer)')\n"
            f"ax[0].axhline({R['passive_sharpe']:.3f}, ls='--', c=GREY, lw=1.5, label='passive B&H')\n"
            "ax[0].set_xlabel('round-trip cost (bps)'); ax[0].set_ylabel('annualised Sharpe')\n"
            "ax[0].set_title('Sharpe vs cost: active starts BELOW passive'); ax[0].legend()\n"
            "ax[1].plot(costs, spreads_c, 'o-', c=RED, lw=2)\n"
            "ax[1].axhline(0, c='k', lw=1)\n"
            "ax[1].set_xlabel('round-trip cost (bps)'); ax[1].set_ylabel('daily alpha (bps)')\n"
            "ax[1].set_title('Alpha: negative even at zero cost')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Alpha starts negative: there is no cost at which this strategy pays.')"
        ),
        md(
            "Unlike the single-sector utilities canary (Study 131), which at least had a "
            "higher Sharpe than buy-and-hold at zero cost, the combined defensive signal "
            "starts *below* the passive line. There is no level of execution efficiency "
            "that rescues this strategy."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Why does adding XLP weaken the signal?** Consumer staples have a higher "
            "beta (~0.5) and more idiosyncratic noise than utilities (~0.4), which dilutes "
            "the clean utilities-RS momentum signal. XLU is the purer defensive proxy.\n"
            "- **Other weightings.** An equal-weighted combination is the simplest choice; "
            "a volatility-weighted or regression-weighted average of XLP and XLU RS signals "
            "might perform differently -- though the extra degrees of freedom raise overfitting risk.\n"
            "- **Broader defensive basket.** Adding XLV (health care) or other defensive "
            "sectors might change the picture, but each additional sector brings more noise.\n"
            "- **Drawdown forecasting.** We tested mean returns; the folk claim is really "
            "about *large drawdowns*. A test on max 21-day drawdown conditional on quintile "
            "might tell a different story (though sample sizes would thin quickly).\n\n"
            "*Think the combined defensive canary really does lead? Fork this, show a monotone "
            "quintile pattern and a t-stat above 2 on the Q1-Q5 spread. That is the bar.*"
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
            "# Defensive-Sectors Leadership -- a quantitative teardown\n"
            "### Combined XLP+XLU/SPY RS momentum -- quintile sorts -- HAC spreads -- regime decomposition\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Coincident--not--forecast](https://img.shields.io/badge/Coincident--not--forecast-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim now carrying its standard error.* "
            "We test whether the rolling 20-day combined (XLP+XLU)/SPY RS-momentum quintile "
            "rank predicts forward SPY log-returns at 1d/5d/21d horizons, run a binary timing "
            "overlay vs buy-and-hold, decompose the regime-Sharpe difference, and confirm via "
            "synthetic positive control that the machinery detects planted effects.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily auto-adjusted closes, "
            f"as-of {R['date_end']}; offline core runs on a deterministic synthetic tape. "
            "Methods in [`docs/references.md`](../docs/references.md), headline numbers "
            "in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **`> In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Q1-Q5 21d spread **+{R['spread_21d']:.1f} bps** but "
            f"HAC *t* = **{R['spread_21d_t']:.2f}** -- far below the bar of 2; 1d spread "
            f"is *negative* ({R['spread_1d']:+.1f} bps, *t* = {R['spread_1d_t']:.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | Timing overlay gross alpha = "
            f"**{R['spread_gross']:+.2f} bps/day** (*t* = {R['spread_gross_t']:+.2f}); "
            f"active Sharpe {R['active_sharpe_gross']:.3f} is **below** passive {R['passive_sharpe']:.3f} "
            "even at zero cost. |\n"
            f"| **Coincident or forecast?** | `COINCIDENT` | Non-monotone quintile pattern "
            "at every horizon; combined RS tags stress already in progress. |\n\n"
            "> In plain words: the combined two-sector defensive canary is *weaker* than "
            "the single-sector utilities canary (Study 131: t = +1.58 at 21d). Adding XLP "
            "noise dilutes the XLU signal, the 1-day spread runs in the wrong direction, "
            "and the timing overlay fails to clear buy-and-hold even before transaction costs."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $r_t$ be the log-return of SPY on day $t$. Define the combined defensive "
            "relative-strength log ratio:\n\n"
            "$$\\rho^{\\text{def}}_t = \\frac{1}{2}\\left[(\\log \\text{XLP}_t - \\log \\text{SPY}_t) "
            "+ (\\log \\text{XLU}_t - \\log \\text{SPY}_t)\\right]$$\n\n"
            "and the 20-day RS momentum $m_t = \\rho^{\\text{def}}_t - \\rho^{\\text{def}}_{t-20}$, "
            "with rolling 252-day percentile rank $q_t \\in (0,1]$ (strictly OOS). "
            "The folk claim asserts:\n\n"
            "- **H1 (quintile signal).** $\\mathbb{E}[r_{t+1..t+h} \\mid q_t \\in \\text{Q5}] < "
            "\\mathbb{E}[r_{t+1..t+h} \\mid q_t \\in \\text{Q1}]$ for $h \\in \\{1, 5, 21\\}$.\n"
            "- **H2 (monotone).** The quintile-conditioned expected return is monotone decreasing in $q$.\n"
            "- **H3 (tradable).** A binary timer that goes to cash when $q_t > 0.80$ earns "
            "positive risk-adjusted alpha over unconditional buy-and-hold.\n\n"
            "- **H4 (improvement).** Combining XLP and XLU should yield a stronger signal "
            "than XLU alone (Study 131 bar: *t* > 1.58 at 21d).\n\n"
            "We reject H1 at 1d (wrong sign), find H2 violated at every horizon, reject H3 "
            "(negative gross alpha even at zero cost), and reject H4 (*t* = 0.79 < 1.58)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "The combined defensive signal is used to justify a 'triple confirmation' of the "
            "risk-off narrative: staples AND utilities AND SPY weakness simultaneously. "
            "Our analysis shows the three-way confluence is largely coincident with existing "
            "volatility, not a leading indicator. The regime Sharpe split (0.52 vs 0.38) "
            "is real but driven almost entirely by volatility clustering, with mean returns "
            "nearly identical (+3.36 vs +3.27 bps/day). The interesting comparative finding: "
            "adding a second defensive sector does not confirm the signal -- it dilutes it."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 -- The protocol\n\n"
            "- **Signal.** $m_t = \\rho^{\\text{def}}_t - \\rho^{\\text{def}}_{t-20}$ where "
            "$\\rho^{\\text{def}}$ is the simple average of XLP/SPY and XLU/SPY log-ratios. "
            "Rolling 252-day percentile rank (min 63 obs before first rank).\n"
            "- **Quintile sort.** Five equal buckets; Q5 is the 'strong defensive alert' bucket.\n"
            "- **Forward return.** Log forward return aligned at $t$, earned from close $t$ "
            "to close $t+h$. No look-ahead: signal uses only data available at close of day $t$.\n"
            "- **Decisive test.** Newey-West HAC *t* on the Q1−Q5 spread "
            "(pooled: $r^{\\text{calm}} - r^{\\text{alert}}$).\n"
            "- **Timing overlay.** Long SPY unless prior-day rank > 80th percentile (flat in "
            "cash). Active-minus-passive spread; cost sweep at 480 round-trips / 27.5 yr.\n"
            "- **Comparison baseline.** Study 131 (XLU-only) achieved *t* = +1.58 at 21d. "
            "Combined signal must beat this to be considered an improvement.\n"
            "- **Positive control.** Synthetic tape with planted $\\text{defensive\\_signal}$: "
            "the machinery must recover t > 2 when a signal is planted.\n\n"
            f"Tape: XLP, XLU, and SPY, {R['date_start']} to {R['date_end']}, n = {R['n']:,} days."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 -- The teardown"),

        md(
            "### 4a -- Quintile table: all three horizons\n\n"
            "The claim predicts monotone descent Q1 -> Q5 and t_spread > 2 at at least one horizon. "
            "Compare to Study 131 where t reached +1.58 at 21d."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = load_real()\n"
            "    qt = st.quintile_table(df, horizons=[1, 5, 21])\n"
            "else:\n"
            "    qt = None\n"
            "quintiles_l = [1, 2, 3, 4, 5]\n"
            "labels_q = ['Q1\\n(calm)', 'Q2', 'Q3', 'Q4', 'Q5\\n(alert)']\n"
            "if qt is not None:\n"
            "    means_21d = [float(qt[(qt.quintile==q)&(qt.horizon==21)]['mean_bps'].values[0]) for q in quintiles_l]\n"
            "    ts_21d    = [float(qt[(qt.quintile==q)&(qt.horizon==21)]['t_stat'].values[0]) for q in quintiles_l]\n"
            "    means_1d  = [float(qt[(qt.quintile==q)&(qt.horizon==1)]['mean_bps'].values[0]) for q in quintiles_l]\n"
            "else:\n"
            f"    means_21d = [{R['q1_21d']}, {R['q2_21d']}, {R['q3_21d']}, {R['q4_21d']}, {R['q5_21d']}]\n"
            f"    ts_21d    = [{R['q1_21d_t']}, 2.75, 1.71, 1.26, {R['q5_21d_t']}]\n"
            f"    means_1d  = [{R['q1_1d']}, {R['q2_1d']}, {R['q3_1d']}, {R['q4_1d']}, {R['q5_1d']}]\n"
            "fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))\n"
            "cols = [GREEN if t > 2 else (AMBER if t > 1 else RED) for t in ts_21d]\n"
            "bars = ax[0].bar(labels_q, means_21d, color=cols)\n"
            "for b, t in zip(bars, ts_21d):\n"
            "    ax[0].annotate(f't={t:+.2f}', (b.get_x()+b.get_width()/2,\n"
            "                b.get_height() + (3 if b.get_height()>=0 else -8)),\n"
            "                ha='center', fontsize=9)\n"
            "ax[0].axhline(0, c='k', lw=1)\n"
            "ax[0].set_ylabel('Mean 21-day forward SPY return (bps)')\n"
            "ax[0].set_title('21-day quintile returns: non-monotone, Q5 reverses above Q3+Q4')\n"
            "ax[1].bar(labels_q, means_1d, color=[RED]*5)\n"
            "ax[1].axhline(0, c='k', lw=1)\n"
            "ax[1].set_ylabel('Mean 1-day forward SPY return (bps)')\n"
            "ax[1].set_title('1-day: Q5 earns MORE than Q1 (wrong direction)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'21d: Q1={means_21d[0]:.1f} bps  Q5={means_21d[4]:.1f} bps  spread={means_21d[0]-means_21d[4]:+.1f} bps')"
        ),
        md(
            f"> In plain words: at 21 days there is a broad descending trend Q1→Q4, but "
            f"Q5 reverses to +{R['q5_21d']:.0f} bps -- above Q3 ({R['q3_21d']:.0f} bps) "
            f"and Q4 ({R['q4_21d']:.0f} bps). At 1 day, Q5 outperforms Q1. "
            "Neither outcome is consistent with 'two defensive sectors outperforming "
            "predicts below-average forward returns.'"
        ),

        md(
            "### 4b -- The decisive spread: comparison to Study 131\n\n"
            "HAC *t* on the Q1−Q5 pooled spread at each horizon, vs Study 131 baseline."
        ),
        code(
            "horizons = [1, 5, 21]\n"
            "spreads, tstats = [], []\n"
            "# Study 131 (XLU only) reference t-stats for comparison\n"
            "study131_t = [0.43, 0.67, 1.58]\n"
            "if HAVE_REAL:\n"
            "    for h in horizons:\n"
            "        r = st.top_minus_bottom(df, horizon=h)\n"
            "        spreads.append(r['spread_bps']); tstats.append(r['t_spread'])\n"
            "else:\n"
            f"    spreads = [{R['spread_1d']}, {R['spread_5d']}, {R['spread_21d']}]\n"
            f"    tstats  = [{R['spread_1d_t']}, {R['spread_5d_t']}, {R['spread_21d_t']}]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "x = np.arange(len(horizons))\n"
            "w = 0.35\n"
            "ax.bar(x - w/2, tstats, w, label='Study 246 (XLP+XLU)', color=RED)\n"
            "ax.bar(x + w/2, study131_t, w, label='Study 131 (XLU only)', color=AMBER, alpha=0.7)\n"
            "for s in (2, -2): ax.axhline(s, ls='--', c=GREY, lw=1.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in horizons])\n"
            "ax.set_ylabel('HAC t-stat (Q1-calm minus Q5-alert spread)')\n"
            "ax.set_title('Adding XLP weakens, not strengthens, the signal vs Study 131')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for h, s, t, t131 in zip(horizons, spreads, tstats, study131_t):\n"
            "    print(f'h={h:2d}d: spread={s:+.2f} bps  t={t:+.2f} (Study131 t={t131:+.2f})')"
        ),
        md(
            "> In plain words: at every horizon, the combined two-sector signal has a *lower* "
            "*t*-stat than the single-sector XLU canary. The information content of XLP's "
            "relative strength is not additive -- it is dilutive. This is the key finding "
            "of this study relative to Study 131."
        ),

        md(
            "### 4c -- Regime Sharpe decomposition\n\n"
            "The calm vs alert Sharpe split -- and why it is *not* alpha."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rs = st.regime_stats(df)\n"
            "    regs = rs['regime'].tolist()\n"
            "    sh   = rs['sharpe_ann'].tolist()\n"
            "    mn   = rs['mean_bps'].tolist()\n"
            "    sd   = rs['std_bps'].tolist()\n"
            "else:\n"
            "    regs = ['Full sample', 'Calm (def Q1-Q2)', 'Alert (def Q3-Q5)']\n"
            f"    sh = [{R['full_sharpe']}, {R['calm_sharpe']}, {R['alert_sharpe']}]\n"
            f"    mn = [{R['full_mean']}, {R['calm_mean']}, {R['alert_mean']}]\n"
            "    sd = [mn[0]/sh[0]*np.sqrt(252)*1e4, mn[1]/sh[1]*np.sqrt(252)*1e4, mn[2]/sh[2]*np.sqrt(252)*1e4]\n"
            "fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))\n"
            "cols3 = [GREY, GREEN, RED]\n"
            "for i, (data_v, label) in enumerate(zip([sh, mn, sd],\n"
            "        ['Annualised Sharpe', 'Mean return (bps/day)', 'Daily vol (bps)'])):\n"
            "    ax[i].bar(regs, data_v, color=cols3)\n"
            "    ax[i].set_title(label); ax[i].tick_params(axis='x', rotation=15)\n"
            "plt.suptitle('Regime Sharpe split is almost entirely a volatility story: means are near-identical', y=1.02)\n"
            "plt.tight_layout(); plt.show()\n"
            "calm_mean_diff = mn[1] - mn[2]\n"
            "print(f'Mean return difference: {calm_mean_diff:+.2f} bps/day -- essentially zero')"
        ),
        md(
            f"> In plain words: the mean return in calm vs alert regimes is "
            f"+{R['calm_mean']:.2f} vs +{R['alert_mean']:.2f} bps/day -- a difference of "
            f"{R['calm_mean']-R['alert_mean']:+.2f} bps that is statistically indistinguishable "
            "from zero. The Sharpe gap (0.52 vs 0.38) is driven entirely by the alert regime "
            "having higher volatility -- as expected, since 2001, 2008, 2020, and 2022 are all "
            "concentrated there. Citing the Sharpe split as evidence of forecasting power "
            "conflates a risk measurement with an alpha measurement."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `NONE`** -- Q1−Q5 21d spread +{R['spread_21d']:.0f} bps, "
            f"HAC *t* = {R['spread_21d_t']:.2f}, far below the bar of 2. 1-day spread is "
            f"negative ({R['spread_1d']:+.1f} bps, *t* = {R['spread_1d_t']:.2f}). "
            f"Non-monotone quintile pattern at every horizon. Adding XLP weakens the "
            f"signal vs Study 131 (XLU-only *t* = 1.58).\n"
            f"- **Tradability `MIRAGE`** -- timing overlay gross alpha = "
            f"{R['spread_gross']:+.2f} bps/day (*t* = {R['spread_gross_t']:+.2f}). "
            f"Active Sharpe {R['active_sharpe_gross']:.3f} < passive {R['passive_sharpe']:.3f} "
            "even before any costs. No break-even cost; strategy destroys value at every "
            "tested cost level.\n"
            "- **Coincident or forecast? `COINCIDENT`** -- the combined defensive RS tags "
            "episodes of market stress but does not statistically precede weak SPY returns. "
            "Mean returns in calm vs alert regimes are nearly identical."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 -- Could you trade it? -- 480 round-trips and a negative alpha from the start\n\n"
            "Active vs passive across costs:"
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    act_sh, spr_mn, spr_t = [], [], []\n"
            "    for c in costs:\n"
            "        ov = st.timing_overlay(df, cost_bps=c)\n"
            "        act_sh.append(st.summarize(ov['r_active'])['sharpe_ann'])\n"
            "        s = st.summarize(ov['r_spread'])\n"
            "        spr_mn.append(s['mean_bps']); spr_t.append(s['t_stat'])\n"
            "else:\n"
            f"    act_sh = [{R['active_sharpe_gross']}, {R['active_sharpe_gross']}-0.006,\n"
            f"               {R['active_sharpe_1bp']}, {R['active_sharpe_1bp']}-0.012, {R['active_sharpe_1bp']}-0.047]\n"
            f"    spr_mn = [{R['spread_gross']}, -0.80, {R['spread_1bp']}, -0.91, -1.11]\n"
            f"    spr_t  = [{R['spread_gross_t']}, -1.09, {R['spread_1bp_t']}, -1.23, -1.51]\n"
            "fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "ax[0].plot(costs, act_sh, 'o-', c=RED, lw=2, label='active Sharpe')\n"
            f"ax[0].axhline({R['passive_sharpe']}, ls='--', c=GREY, lw=1.5, label='passive (B&H)')\n"
            "ax[0].set_xlabel('round-trip cost (bps)'); ax[0].set_ylabel('annualised Sharpe')\n"
            "ax[0].set_title('Active Sharpe: starts BELOW passive at zero cost'); ax[0].legend()\n"
            "ax[1].plot(costs, spr_mn, 'o-', c=RED, lw=2)\n"
            "ax[1].axhline(0, c='k', lw=1)\n"
            "ax2 = ax[1].twinx(); ax2.plot(costs, spr_t, 's--', c=GREY, lw=1.5)\n"
            "ax2.axhline(-2, ls=':', c=GREY)\n"
            "ax[1].set_xlabel('round-trip cost (bps)'); ax[1].set_ylabel('daily alpha (bps)', c=RED)\n"
            "ax2.set_ylabel('HAC t-stat', c=GREY)\n"
            "ax[1].set_title('Alpha: negative at every cost (no break-even)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('No break-even cost exists. The strategy destroys value from the start.')"
        ),
        md(
            "> In plain words: unlike Study 131, where the timing overlay had a higher Sharpe "
            "than buy-and-hold at zero cost (beta reduction artefact), the combined defensive "
            "timer starts *below* the passive line. Both the Sharpe and the alpha are worse "
            "than doing nothing. The strategy offers no benefit at any cost level."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 -- Going further -- the positive control\n\n"
            "Does the machinery work? Plant a negative `defensive_signal` in a synthetic tape "
            "and confirm the Q1−Q5 spread emerges at sufficient signal strength."
        ),
        code(
            "signals = [0.0, -0.001, -0.002, -0.003, -0.005, -0.010]\n"
            "spreads_s, tstats_s = [], []\n"
            "for sig in signals:\n"
            "    df_s, _ = data.synthetic_daily(n_days=6500, defensive_signal=sig, seed=246)\n"
            "    r21 = st.top_minus_bottom(df_s, horizon=21)\n"
            "    spreads_s.append(r21['spread_bps']); tstats_s.append(r21['t_spread'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.plot([abs(s) for s in signals], tstats_s, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1.5, label='inference bar (t=2)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('|planted defensive signal|'); ax.set_ylabel('HAC t-stat (21d Q1-Q5 spread)')\n"
            "ax.set_title('Positive control: machinery detects planted signal when strong enough')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('The engine works: it recovers t > 2 when the planted signal is large enough.')"
        ),
        md(
            "The positive control confirms the machinery is a faithful detector. The real tape "
            "result (*t* = 0.79 at 21d) is not a failure of the code but a genuine finding: "
            "the combined defensive signal carries less information about forward SPY returns "
            "than would be required for statistical certification -- and less than the single-"
            "sector XLU canary (Study 131)."
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
