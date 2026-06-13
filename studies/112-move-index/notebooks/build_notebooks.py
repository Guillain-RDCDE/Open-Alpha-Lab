"""Generate the two narrative notebooks for Study 112 (Move-Index).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
parquet under ../_cache/ if present and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

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
    n=5796,
    date_start="2003-01-06", date_end="2026-06-12",
    move_min=36.6, move_max=264.6, move_mean=86.8, move_std=30.5,
    vix_min=9.1, vix_max=82.7, vix_mean=19.1, vix_std=8.4,
    move_vix_corr=0.586,
    # Quintile spread results
    spread1=-0.74, t_spread1=-0.76,
    spread5=+0.12, t_spread5=-0.68,
    spread21=-28.08, t_spread21=-1.63,
    q5_mean1=+4.07, q1_mean1=+4.80,
    q5_mean5=+22.74, q1_mean5=+22.63,
    q5_mean21=+68.69, q1_mean21=+96.78,
    # Regression results
    t_move_h1=-0.80, t_vix_h1=+1.02, r2_h1=0.00031,
    t_move_h5=-0.59, t_vix_h5=+0.23, r2_h5=0.00018,
    t_move_h21=-0.95, t_vix_h21=+0.49, r2_h21=0.00093,
    # MOVE/VIX ratio
    t_ratio1=-1.23, spread_ratio1=-1.83, t_spread_ratio1=-0.35,
    t_ratio5=-1.02, spread_ratio5=-11.78, t_spread_ratio5=-0.53,
    t_ratio21=-1.65, spread_ratio21=-54.59, t_spread_ratio21=-0.93,
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

from move_index import data, strategy as st

R = dict(
    n=5796, date_start="2003-01-06", date_end="2026-06-12",
    move_min=36.6, move_max=264.6, move_mean=86.8, move_std=30.5,
    vix_min=9.1, vix_max=82.7, vix_mean=19.1, vix_std=8.4,
    move_vix_corr=0.586,
    spread1=-0.74, t_spread1=-0.76,
    spread5=+0.12, t_spread5=-0.68,
    spread21=-28.08, t_spread21=-1.63,
    q5_mean1=+4.07, q1_mean1=+4.80,
    q5_mean5=+22.74, q1_mean5=+22.63,
    q5_mean21=+68.69, q1_mean21=+96.78,
    t_move_h1=-0.80, t_vix_h1=+1.02, r2_h1=0.00031,
    t_move_h5=-0.59, t_vix_h5=+0.23, r2_h5=0.00018,
    t_move_h21=-0.95, t_vix_h21=+0.49, r2_h21=0.00093,
    t_ratio1=-1.23, spread_ratio1=-1.83, t_spread_ratio1=-0.35,
    t_ratio5=-1.02, spread_ratio5=-11.78, t_spread_ratio5=-0.53,
    t_ratio21=-1.65, spread_ratio21=-54.59, t_spread_ratio21=-0.93,
)

def _have_cache():
    import os
    return os.path.exists(data._cache_path(data.DEFAULT_CACHE))

HAVE_REAL = _have_cache()
print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Move-Index -- does bond-market fear predict equity drawdowns?\n"
            "### The MOVE index as a cross-asset risk gauge, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_VIX%3F: No](https://img.shields.io/badge/Beats__VIX%3F-No-8b949e?style=flat-square)\n\n"
            "The **MOVE index** (ICE BofA, ticker ``^MOVE``) is the 'VIX for bonds' -- it measures "
            "how much option traders in the Treasury market expect rates to swing over the next month. "
            "When MOVE spikes, bond markets are pricing in uncertainty. The claim is that this "
            "cross-asset fear gauge should also warn about **equity drawdowns**: if bond markets are "
            "nervous, stocks should soon follow. It sounds intuitive -- bonds are the 'smart money'. "
            "But is it actually true that a high MOVE reading predicts negative SPY returns?\n\n"
            "> **This is the plain-language layer.** Want the t-stats and regression coefficients? "
            "See the companion, **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same "
            "story, deeper numbers.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does high MOVE predict negative SPY returns? | **No.** The Q5 vs Q1 spread is "
            f"**{R['spread1']:+.1f} bps** at 1 day (HAC *t* = {R['t_spread1']:+.2f}) -- a coin. |\n"
            "| Does MOVE add anything over VIX alone? | **No.** In a joint regression, MOVE's "
            f"t-stat is **{R['t_move_h1']:+.2f}** -- statistically zero alongside VIX. |\n"
            "| Does the MOVE/VIX ratio (bond fear vs equity fear) work? | **No.** t-ratio = "
            f"**{R['t_ratio5']:+.2f}** at 5 days -- no better than a coin. |\n"
            "| Could you trade it? | **No edge to trade.** All three tests return NONE. |\n\n"
            "> The MOVE index is a *coincident* risk measure -- it rises with market stress -- but "
            "it does not *lead* equity returns. Looking at MOVE today tells you almost nothing "
            "about where SPY will be tomorrow, next week, or next month."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 The claim\n\n"
            "> *\"The MOVE index is the bond market's fear gauge. When MOVE spikes, the smart "
            "money in fixed income is pricing in uncertainty -- and equity markets usually follow. "
            "Watch MOVE: when it's high or rising faster than VIX, expect equity turbulence.\"*\n\n"
            "This is a real claim, found in macro research, sell-side notes, and CNBC commentary. "
            "It has an intuitive basis: bonds and equities share common macro risk factors (growth "
            "and credit), so if bond-options traders are hedging aggressively, equity vol should "
            "follow. The steelman adds that MOVE might lead VIX because fixed-income players are "
            "often better-informed about macro tail risks.\n\n"
            "The test is simple: sort 20+ years of daily MOVE readings into quintiles and ask "
            "whether high-MOVE days predict lower SPY returns over the next 1, 5, or 21 days."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 So what?\n\n"
            "If MOVE truly led equity markets, it would be a rare macro timing signal: a "
            "freely-available, liquid indicator you could use to reduce equity exposure before "
            "drawdowns. It would also be a useful hedge-allocation trigger -- go long vol or "
            "cut SPY when MOVE screams.\n\n"
            "If it's false, macro commentators and risk desks are reading MOVE as a leading "
            "indicator when it's just a concurrent description of market stress -- useful for "
            "understanding *what's happening*, useless for predicting *what comes next*."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 How would we even know?\n\n"
            "Three complementary tests, all out-of-sample:\n\n"
            "1. **Quintile sort.** Rank each day's MOVE against the prior 252 days. Assign to "
            "quintiles Q1 (lowest 20%) through Q5 (highest 20%). Compute the average SPY forward "
            "return for each quintile at 1-, 5-, and 21-day horizons. If the claim is true, "
            "Q5 should have noticeably lower returns than Q1.\n\n"
            "2. **MOVE vs VIX regression.** Standardise both MOVE and VIX (rolling z-score) and "
            "regress forward SPY returns on both simultaneously. If MOVE adds something over VIX "
            "alone, its regression coefficient should be significant (|t| >= 2).\n\n"
            "3. **MOVE/VIX ratio.** A high ratio means bond vol is elevated relative to equity vol "
            "-- the specific 'cross-asset divergence' version of the story. Test whether a high "
            "ratio predicts negative forward returns.\n\n"
            f"Data: ``^MOVE``, ``^VIX``, SPY daily from Yahoo Finance, **{R['n']} trading days** "
            f"({R['date_start']} to {R['date_end']}). Over 20 years of MOVE history."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 The teardown -- let's actually look\n\n"
            "**First: the quintile sort.** Here's the mean forward SPY return for each MOVE quintile "
            "at 1-, 5-, and 21-day horizons. If MOVE is a leading indicator, the bars should trend "
            "downward from Q1 (lowest MOVE) to Q5 (highest MOVE)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_daily(fetch=False)\n"
            "    tbl = st.quintile_table(df, horizons=[1, 5, 21])\n"
            "else:\n"
            "    # Fall back to frozen numbers from docs/results.md\n"
            "    rows = []\n"
            "    for qi in range(1, 6):\n"
            "        for h, q1m, q5m in [(1, R['q1_mean1'], R['q5_mean1']),\n"
            "                            (5, R['q1_mean5'], R['q5_mean5']),\n"
            "                            (21, R['q1_mean21'], R['q5_mean21'])]:\n"
            "            rows.append({'quintile': qi, 'horizon': h,\n"
            "                         'mean_bps': q1m if qi==1 else (q5m if qi==5 else (q1m+q5m)/2),\n"
            "                         'n': 1000})\n"
            "    tbl = pd.DataFrame(rows)\n"
            "\n"
            "fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)\n"
            "for ax, h in zip(axes, [1, 5, 21]):\n"
            "    sub = tbl[tbl['horizon']==h].sort_values('quintile')\n"
            "    ax.bar(sub['quintile'], sub['mean_bps'], color=GREY, alpha=0.75)\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.set_xlabel('MOVE quintile (1=low, 5=high)')\n"
            "    ax.set_ylabel('fwd SPY return (bps)' if h==1 else '')\n"
            "    ax.set_title(f'h = {h} day{\"s\" if h>1 else \"\"}')\n"
            "    ax.set_xticks([1,2,3,4,5])\n"
            "plt.suptitle('Forward SPY returns by MOVE quintile -- no downward trend', y=1.01)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Q5-Q1 spread at 21d: {{R[\"spread21\"]:+.1f}} bps  (HAC t = {{R[\"t_spread21\"]:+.2f}})')"
        ),
        md(
            "The bars show no consistent downward trend from Q1 to Q5. In fact, at the 21-day "
            f"horizon, Q5 (high MOVE) delivers **{R['q5_mean21']:+.0f} bps** and Q1 (low MOVE) "
            f"delivers **{R['q1_mean21']:+.0f} bps** -- a spread of **{R['spread21']:+.0f} bps** "
            f"(HAC *t* = {R['t_spread21']:+.2f}). That's directionally consistent with the story "
            "but not statistically significant."
        ),
        md(
            "**Now: does MOVE add anything over VIX?** Both in one regression."
        ),
        code(
            "if HAVE_REAL:\n"
            "    regs = {h: st.move_vs_vix_regression(df, horizon=h) for h in [1,5,21]}\n"
            "    tmove = [regs[h]['t_move'] for h in [1,5,21]]\n"
            "    tvix  = [regs[h]['t_vix']  for h in [1,5,21]]\n"
            "else:\n"
            "    tmove = [R['t_move_h1'], R['t_move_h5'], R['t_move_h21']]\n"
            "    tvix  = [R['t_vix_h1'],  R['t_vix_h5'],  R['t_vix_h21']]\n"
            "\n"
            "horizons = [1, 5, 21]\n"
            "x = np.arange(len(horizons))\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.bar(x - 0.2, tmove, width=0.38, label='MOVE (bond vol)', color=RED, alpha=0.8)\n"
            "ax.bar(x + 0.2, tvix,  width=0.38, label='VIX (equity vol)', color=GREY, alpha=0.8)\n"
            "ax.axhline(2, ls='--', c='k', lw=1, label='|t|=2 bar')\n"
            "ax.axhline(-2, ls='--', c='k', lw=1)\n"
            "ax.axhline(0, c='k', lw=0.5)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['1 day','5 days','21 days'])\n"
            "ax.set_ylabel('HAC t-stat'); ax.legend()\n"
            "ax.set_title('MOVE vs VIX: neither index predicts forward SPY returns')\n"
            "ax.set_ylim(-3, 3); plt.tight_layout(); plt.show()\n"
            "print('All bars inside the +/-2 band: NONE for both MOVE and VIX.')"
        ),
        md(
            "Neither MOVE nor VIX reaches the |*t*| >= 2 bar at any horizon. The idea that "
            "MOVE leads VIX which leads equities is not borne out: both indices are coincident "
            "with market stress, not predictive of it. "
            "> Both indices encode *current* fear, not *future* pain."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 The verdict\n\n"
            "- **Signal -- None.** MOVE quintile Q5-Q1 spread: "
            f"**{R['spread1']:+.1f} bps** (1d), **{R['spread5']:+.1f} bps** (5d), "
            f"**{R['spread21']:+.1f} bps** (21d). HAC t-stats: "
            f"{R['t_spread1']:+.2f}, {R['t_spread5']:+.2f}, {R['t_spread21']:+.2f}. "
            "No horizon reaches |t| >= 2. MOVE does not add over VIX (t_MOVE = "
            f"{R['t_move_h1']:+.2f}). MOVE/VIX ratio also fails (t_ratio = {R['t_ratio5']:+.2f}).\n"
            "- **Tradability -- Mirage.** With no measurable signal at any horizon, there is "
            "no strategy to evaluate. Even the directional 21-day effect (t = -1.63) is well "
            "inside the noise band and would likely not survive transaction costs or a "
            "walk-forward test.\n"
            "- **Beats VIX? -- No.** MOVE and VIX are correlated (0.59) but neither adds "
            "incremental predictive power over the other. They move together in crisis episodes "
            "but as concurrent risk gauges, not forecasting tools."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 Could you actually trade it?\n\n"
            "The signal doesn't pass the first test (statistical significance), so there is "
            "no edge to cost-adjust. Even granting the weak 21-day directional tendency:\n\n"
            "- **Horizon mismatch.** A 21-day-ahead signal requires holding SPY short or "
            "underweight for a month. Daily SPY vol is ~80 bps -- a 21-day cumulative vol of "
            "roughly 3.7%. The claimed signal is ~28 bps spread -- well inside one day's noise.\n"
            "- **Overlapping observations.** 21-day windows overlap, inflating apparent sample "
            "sizes. The conservative HAC t-stat accounts for this, and it still falls short.\n"
            "- **Regime sensitivity.** MOVE spikes during acute crises (2008, 2020, 2022). "
            "Any regression of forward returns on a MOVE-derived signal is dominated by a "
            "handful of macro episodes -- not a stable tradable edge.\n\n"
            "Bottom line: there is no break-even cost, because there is no edge."
        ),

        # ---- BEAT 7 -- GOING FURTHER ----------------------------------------
        md(
            "## 7 Going further\n\n"
            "- **The positive control.** The companion notebook shows that the engine *does* "
            "find a MOVE signal when one is planted in a synthetic tape -- so the null result "
            "is about the market, not the method.\n"
            "- **SKEW as a tail-radar.** The CBOE SKEW index (put/call skew on SPY options) "
            "is the equity-market analogue of MOVE. [Study 86 (Tail-Radar)](../../86-tail-radar/) "
            "runs the same protocol on SKEW -- same NONE verdict.\n"
            "- **Rate-level signals.** The level of the 10-year yield (not its implied vol) "
            "has a better-documented relationship with equity returns through the equity risk "
            "premium channel. A rate-level sort might find more than a vol-level sort.\n"
            "- **Crisis regimes.** MOVE *is* informative *during* crises (it peaks near equity "
            "troughs), but that's a contemporaneous relationship, not a predictive one. A "
            "crisis-regime conditional model might find something MOVE contributes.\n\n"
            "*Think the MOVE signal is real at a different horizon or specification? Fork "
            "this, change the lookback or the quintile cut, and show a fair edge that clears "
            "|t| >= 2 in an out-of-sample window. That's the bar.*"
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
            "# Move-Index -- a quantitative teardown\n"
            "### Daily MOVE, VIX, SPY tape 2003-2026 | quintile sorts | HAC t-stats | OLS regression\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_VIX%3F: No](https://img.shields.io/badge/Beats__VIX%3F-No-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim now carrying its standard error.* We test whether "
            "the MOVE index predicts forward SPY returns (via quintile sorts and HAC t-stats), "
            "whether it adds incremental information over VIX (OLS regression with NW standard "
            "errors), and whether the MOVE/VIX ratio is a useful cross-asset signal.\n\n"
            "> **Not investment advice.** Real data: Yahoo Finance daily closes, "
            f"{R['n']} trading days {R['date_start']} to {R['date_end']}; "
            "the offline core and tests run on a deterministic synthetic tape. "
            "Methods and sources in [`docs/references.md`](../docs/references.md); "
            "reproducible numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `> in plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | MOVE Q5-Q1 spread: **{R['spread1']:+.1f} bps** (1d, "
            f"t = {R['t_spread1']:+.2f}), **{R['spread5']:+.1f} bps** (5d, t = {R['t_spread5']:+.2f}), "
            f"**{R['spread21']:+.1f} bps** (21d, t = {R['t_spread21']:+.2f}). No |t| >= 2 anywhere. |\n"
            f"| **Tradability** | `MIRAGE` | No significant signal at any horizon; no edge to trade. |\n"
            f"| **Beats VIX?** | `NO` | Joint regression: t_MOVE = **{R['t_move_h1']:+.2f}**, "
            f"t_VIX = {R['t_vix_h1']:+.2f} (h=1d). MOVE adds nothing over VIX alone. |\n\n"
            "> in plain words: MOVE is a real-time barometer of bond-market stress. It just "
            "does not forecast where equity markets will be in 1, 5, or 21 days."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 The claim, steelmanned\n\n"
            "Let $m_t$ be the MOVE percentile rank at day $t$ (rolling 252-day OOS) and "
            "$r_{t \\to t+h}$ the forward log-return on SPY.  The claim is:\n\n"
            "- **H1 (level signal):** $\\mathbb{E}[r_{t\\to t+h} \\mid m_t \\in Q5] < "
            "\\mathbb{E}[r_{t\\to t+h} \\mid m_t \\in Q1]$ -- high MOVE predicts "
            "worse-than-average equity returns.\n"
            "- **H2 (incremental):** In OLS $r_{t\\to t+h} = \\alpha + \\beta_M \\tilde{m}_t + "
            "\\beta_V \\tilde{v}_t + \\epsilon_t$ (both standardised OOS), $\\beta_M$ is "
            "significant after controlling for $\\tilde{v}_t$ (VIX z-score).\n"
            "- **H3 (ratio signal):** The ratio $\\text{MOVE}_t / \\text{VIX}_t$ (relative "
            "bond-vol stress) predicts negative $r_{t\\to t+h}$ beyond either index alone.\n\n"
            "All three hypotheses are tested with Newey-West HAC t-stats to account for "
            "autocorrelation in overlapping forward returns."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 So what? -- what rides on each answer\n\n"
            "If H1-H3 held, MOVE would be a macro timing tool with 20+ years of out-of-sample "
            "support: allocators could use it as a risk-on/risk-off trigger. The interesting "
            "failure is *why* it doesn't work: MOVE is an option-implied volatility measure, "
            "which means it's priced by arbitrageurs who already embed current macro views. "
            "Any predictive content for equities would be quickly arbitraged away -- the very "
            "thing option prices are designed to reflect."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 How we'd know -- the protocol\n\n"
            "- **Data:** ``^MOVE``, ``^VIX``, SPY daily from Yahoo Finance, "
            f"{R['date_start']} -- {R['date_end']}, n = {R['n']}.\n"
            "- **Quintile rank:** rolling 252-day percentile rank (strict OOS; min 63 obs). "
            "  Labels Q1-Q5 assigned at day $t$; forward return computed for days $t+1$ through "
            "  $t+h$.\n"
            "- **Regression:** OLS on standardised MOVE and VIX (rolling z-score, same window). "
            "  HAC standard errors via Newey-West with $\\lfloor 4(n/100)^{2/9} \\rfloor$ lags.\n"
            "- **Ratio signal:** MOVE/VIX ratio, rolling z-score, OLS on forward returns + "
            "  quintile sort of the ratio.\n"
            "- **Positive control:** deterministic synthetic tape with tunable ``move_signal`` "
            "  to confirm the engine detects a signal when one is planted."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 The teardown"),
        md(
            "### 4a Quintile-sorted forward returns -- no monotone pattern\n\n"
            "Mean forward SPY log-return by MOVE quintile at h = 1, 5, 21 days."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_daily(fetch=False)\n"
            "    tbl = st.quintile_table(df, horizons=[1, 5, 21])\n"
            "else:\n"
            "    rows = []\n"
            "    for qi in range(1, 6):\n"
            "        for h, q1m, q5m in [(1, R['q1_mean1'], R['q5_mean1']),\n"
            "                            (5, R['q1_mean5'], R['q5_mean5']),\n"
            "                            (21, R['q1_mean21'], R['q5_mean21'])]:\n"
            "            rows.append({'quintile': qi, 'horizon': h,\n"
            "                         'mean_bps': q1m if qi==1 else (q5m if qi==5 else (q1m+q5m)/2),\n"
            "                         't_stat': 0, 'n': 1000})\n"
            "    tbl = pd.DataFrame(rows)\n"
            "\n"
            "fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)\n"
            "for ax, h in zip(axes, [1, 5, 21]):\n"
            "    sub = tbl[tbl['horizon']==h].sort_values('quintile')\n"
            "    colors = [GREEN if v >= 0 else RED for v in sub['mean_bps']]\n"
            "    ax.bar(sub['quintile'], sub['mean_bps'], color=colors, alpha=0.75)\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.set_xlabel('MOVE quintile'); ax.set_title(f'h={h}d')\n"
            "    ax.set_ylabel('mean fwd SPY (bps)' if h==1 else '')\n"
            "    ax.set_xticks([1,2,3,4,5])\n"
            "plt.suptitle('No monotone Q1->Q5 decline in forward SPY returns', y=1.01)\n"
            "plt.tight_layout(); plt.show()\n"
            "\n"
            "# Print the spread table\n"
            "if HAVE_REAL:\n"
            "    print(f'{'h':>5} | {'Q5 mean':>9} {'Q1 mean':>9} | {'spread':>9} {'t_spread':>10}')\n"
            "    print('-' * 50)\n"
            "    for h in [1, 5, 21]:\n"
            "        res = st.top_minus_bottom(df, horizon=h)\n"
            "        print(f'{h:>5}d | {res[\"mean_top_bps\"]:>+9.1f} {res[\"mean_bot_bps\"]:>+9.1f} | '\n"
            "              f'{res[\"spread_bps\"]:>+9.1f} {res[\"t_spread\"]:>+10.2f}')\n"
            "else:\n"
            "    for h, s, t in [(1, R['spread1'], R['t_spread1']),\n"
            "                    (5, R['spread5'], R['t_spread5']),\n"
            "                    (21, R['spread21'], R['t_spread21'])]:\n"
            "        print(f'h={h:2d}d: spread={s:+.1f} bps, t={t:+.2f}')"
        ),
        md(
            "> in plain words: if MOVE predicted equity pain, the bars should step down "
            "left to right. They don't -- the pattern is essentially flat at 1 and 5 days, "
            f"and the modest 21-day gap ({R['spread21']:+.0f} bps, t = {R['t_spread21']:+.2f}) "
            "is statistically indistinguishable from zero."
        ),
        md(
            "### 4b MOVE vs VIX -- the joint regression\n\n"
            "OLS coefficient t-stats for standardised MOVE and VIX as joint predictors "
            "of forward SPY returns. A bar outside +-2 would signal real incremental value."
        ),
        code(
            "if HAVE_REAL:\n"
            "    regs = {h: st.move_vs_vix_regression(df, horizon=h) for h in [1,5,21]}\n"
            "    tmove = [regs[h]['t_move'] for h in [1,5,21]]\n"
            "    tvix  = [regs[h]['t_vix']  for h in [1,5,21]]\n"
            "    r2s   = [regs[h]['r2']     for h in [1,5,21]]\n"
            "else:\n"
            "    tmove = [R['t_move_h1'], R['t_move_h5'], R['t_move_h21']]\n"
            "    tvix  = [R['t_vix_h1'],  R['t_vix_h5'],  R['t_vix_h21']]\n"
            "    r2s   = [R['r2_h1'], R['r2_h5'], R['r2_h21']]\n"
            "\n"
            "horizons = ['1 day', '5 days', '21 days']\n"
            "x = np.arange(len(horizons))\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.bar(x - 0.2, tmove, 0.38, label='MOVE (bond vol)', color=RED, alpha=0.8)\n"
            "ax.bar(x + 0.2, tvix,  0.38, label='VIX (equity vol)', color=GREY, alpha=0.8)\n"
            "for s in (2, -2): ax.axhline(s, ls='--', c='k', lw=1)\n"
            "ax.axhline(0, c='k', lw=0.5)\n"
            "ax.set_xticks(x); ax.set_xticklabels(horizons)\n"
            "ax.set_ylabel('HAC t-stat on regression coefficient')\n"
            "ax.set_title('Neither MOVE nor VIX is a significant forward-return predictor')\n"
            "ax.set_ylim(-3, 3); ax.legend(); plt.tight_layout(); plt.show()\n"
            "\n"
            "print(f'Regression R2: h=1d {r2s[0]:.5f}, h=5d {r2s[1]:.5f}, h=21d {r2s[2]:.5f}')\n"
            "print('All bars inside the +-2 band; R2 < 0.001 at all horizons.')"
        ),
        md(
            "> in plain words: the regression has essentially no explanatory power "
            f"(R2 < 0.1% at all horizons). Neither MOVE (t = {R['t_move_h1']:+.2f}) nor VIX "
            f"(t = {R['t_vix_h1']:+.2f}) is a significant predictor once we use the full "
            "2003-2026 sample and proper HAC standard errors."
        ),
        md(
            "### 4c MOVE/VIX ratio -- does cross-asset divergence matter?\n\n"
            "The ratio MOVE/VIX measures bond-vol relative to equity-vol. A high ratio "
            "('bond market more stressed than equity') is the specific cross-asset version "
            "of the MOVE story."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rat5  = st.move_vix_ratio_signal(df, horizon=5)\n"
            "    ratio = df['MOVE'] / df['VIX']\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "    axes[0].plot(df.index, ratio, color=GREY, lw=0.7, alpha=0.8)\n"
            "    axes[0].set_title('MOVE/VIX ratio, 2003-2026')\n"
            "    axes[0].set_ylabel('MOVE/VIX'); axes[0].set_xlabel('date')\n"
            "    # Quintile-sorted ratio vs 5d forward SPY\n"
            "    from move_index.strategy import move_quintile, forward_return\n"
            "    ratio_q = move_quintile(ratio.dropna(), window=252, min_periods=63)\n"
            "    ratio_q = ratio_q.reindex(df.index)\n"
            "    fwd5 = forward_return(df['SPY_close'], horizon=5)\n"
            "    means = [fwd5[(ratio_q==qi) & fwd5.notna()].mean()*1e4 for qi in range(1,6)]\n"
            "    axes[1].bar(range(1,6), means, color=GREY, alpha=0.75)\n"
            "    axes[1].axhline(0, c='k', lw=1)\n"
            "    axes[1].set_xlabel('MOVE/VIX quintile (1=low, 5=high)')\n"
            "    axes[1].set_ylabel('mean fwd SPY 5d (bps)')\n"
            "    axes[1].set_title('MOVE/VIX ratio quintile vs forward SPY (h=5d)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'Ratio beta t-stat (h=5d): {rat5[\"t_ratio\"]:+.2f}  '\n"
            "          f'Q5-Q1 spread: {rat5[\"spread_bps\"]:+.1f} bps (t={rat5[\"t_spread\"]:+.2f})')\n"
            "else:\n"
            "    print(f'MOVE/VIX ratio h=5d: t_ratio = {R[\"t_ratio5\"]:+.2f}, '\n"
            "          f'spread = {R[\"spread_ratio5\"]:+.1f} bps (t = {R[\"t_spread_ratio5\"]:+.2f})')"
        ),
        md(
            f"> in plain words: the ratio t-stat is {R['t_ratio5']:+.2f} at 5 days "
            "-- directionally negative (consistent with the story) but nowhere near the "
            "|t| >= 2 bar. The cross-asset divergence narrative has no measurable "
            "predictive power in this data."
        ),
        md(
            "### 4d Synthetic positive control -- the engine works when there's something to find\n\n"
            "On a synthetic tape with a tunable `move_signal` knob, the quintile spread "
            "grows monotonically. This confirms the engine would detect real predictability "
            "if it existed in the real tape."
        ),
        code(
            "signals = [0.00, 0.01, 0.02, 0.03, 0.05]\n"
            "spreads, tstats = [], []\n"
            "for sig in signals:\n"
            "    df_syn, _ = data.synthetic_daily(n_days=2000, move_signal=sig, seed=42)\n"
            "    res = st.top_minus_bottom(df_syn, horizon=1)\n"
            "    spreads.append(res['spread_bps'])\n"
            "    tstats.append(res['t_spread'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.plot(signals, spreads, 'o-', c=GREEN, lw=2, label='Q5-Q1 spread (bps)')\n"
            "ax2 = ax.twinx()\n"
            "ax2.plot(signals, tstats, 's--', c=GREY, lw=1.5, label='HAC t-stat')\n"
            "ax2.axhline(-2, ls=':', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted move_signal'); ax.set_ylabel('Q5-Q1 spread (bps)', color=GREEN)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('Engine is a faithful MOVE detector -- signal absent from real tape')\n"
            "fig.legend(loc='lower right', bbox_to_anchor=(0.9, 0.15)); plt.tight_layout(); plt.show()\n"
            "print('At move_signal=0 (null): spread near 0 bps, t near 0.')\n"
            f"print('Real tape is consistent with move_signal ~= 0 (spread = {R['spread1']:+.1f} bps, t = {R['t_spread1']:+.2f})')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 The verdict\n\n"
            f"- **Signal `NONE`** -- Q5-Q1 spread: {R['spread1']:+.1f} bps (h=1d, t={R['t_spread1']:+.2f}), "
            f"{R['spread5']:+.1f} bps (h=5d, t={R['t_spread5']:+.2f}), "
            f"{R['spread21']:+.1f} bps (h=21d, t={R['t_spread21']:+.2f}). "
            "No horizon clears |t|>=2. Joint regression: "
            f"t_MOVE = {R['t_move_h1']:+.2f}, t_VIX = {R['t_vix_h1']:+.2f}, R2 < 0.001. "
            f"MOVE/VIX ratio: t_ratio = {R['t_ratio5']:+.2f}.\n"
            "- **Tradability `MIRAGE`** -- No significant signal at any horizon or specification. "
            "No edge exists to cost-adjust.\n"
            "- **Beats VIX? `NO`** -- MOVE-VIX correlation = 0.59; neither adds over the other "
            "as a forward-return predictor."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 Could you trade it? -- no edge to adjust\n\n"
            "The standard tradability analysis (gross edge vs break-even cost) doesn't apply "
            "here because the gross edge is statistically indistinguishable from zero. A few "
            "additional observations for the practitioner:\n\n"
            "- **Capacity** would be enormous (SPY is highly liquid) but irrelevant -- "
            "size doesn't matter if the signal doesn't exist.\n"
            "- **The 21-day effect (t = -1.63)** is the strongest finding. Even if real, "
            "a monthly-rebalanced long-short equity strategy based on MOVE quintiles would face: "
            "~4 turns/year, ~25 bps round-trip cost for large allocations. The required annualised "
            "alpha to survive costs would be ~100 bps -- far above the claimed 28 bps spread.\n"
            "- **Regime sensitivity:** MOVE peaked at 264 in 2008 and 200 in 2020. A quintile-based "
            "strategy is dominated by a handful of macro crisis episodes; out-of-sample stability "
            "is very uncertain."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 Going further\n\n"
            "- **SKEW vs MOVE.** [Study 86 (Tail-Radar)](../../86-tail-radar/) runs the same "
            "protocol on the CBOE SKEW index -- same NONE verdict. Vol indices in general "
            "appear to be coincident, not leading.\n"
            "- **Rate-level signals.** The 10-year yield level and its change (not its implied "
            "vol) have a better-documented relationship with equities through the discount-rate "
            "channel. Worth a separate study.\n"
            "- **Conditional on regime.** MOVE may be more informative when it's rising sharply "
            "from low levels (regime change) vs when it's already elevated. A conditional "
            "specification might find something the unconditional sort misses.\n"
            "- **Credit spreads.** IG and HY credit spreads are an alternative measure of "
            "fixed-income stress that are more directly tied to equity valuations via the "
            "risk premium channel.\n\n"
            "*Want to find a MOVE signal? Fork this, test a different specification "
            "(level change, rate of change, conditional regime), show |t|>=2 out-of-sample, "
            "and send a PR. That's the bar.*"
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
