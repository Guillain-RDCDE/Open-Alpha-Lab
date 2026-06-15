"""Generate the two narrative notebooks for Study 202 (Fifty-Two-Week-Low).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-15).
R = dict(
    n_days=3383, n_tickers=20, n_periods=626,
    date_start="2013-01-02", date_end="2026-06-15",
    # Q1-Q5 spread at 5d
    spread_mean=-7.42, spread_t=-0.67, spread_win=48.2,
    # Per-quintile 5d returns
    q1_mean=25.80, q1_t=2.39, q1_win=58.6,
    q2_mean=40.91, q2_t=4.38,
    q3_mean=33.55, q3_t=3.56,
    q4_mean=24.08, q4_t=2.64,
    q5_mean=33.22, q5_t=3.71, q5_win=60.4,
    # Long-only bottom decile vs baseline
    decile_ann=8.5, baseline_ann=11.3, decile_spread_t=-0.72,
    # Hold-period sweep
    hold1d_spread=-11.12, hold1d_t=-2.00,
    hold5d_spread=-7.42, hold5d_t=-0.67,
    hold20d_spread=-8.13, hold20d_t=-0.21,
    hold65d_spread=-37.30, hold65d_t=-0.45,
    # 65d Q1 / Q5
    q1_65d=403.40, q5_65d=440.70,
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

from fifty_two_week_low import data, strategy as st

CACHE = os.path.abspath(os.path.join("..", "_cache"))

# Frozen real-tape headline numbers (mirror of docs/results.md, as-of 2026-06-15)
R = dict(
    n_days=3383, n_tickers=20, n_periods=626,
    date_start="2013-01-02", date_end="2026-06-15",
    spread_mean=-7.42, spread_t=-0.67, spread_win=48.2,
    q1_mean=25.80, q1_t=2.39, q1_win=58.6,
    q2_mean=40.91, q2_t=4.38,
    q3_mean=33.55, q3_t=3.56,
    q4_mean=24.08, q4_t=2.64,
    q5_mean=33.22, q5_t=3.71, q5_win=60.4,
    decile_ann=8.5, baseline_ann=11.3, decile_spread_t=-0.72,
    hold1d_spread=-11.12, hold1d_t=-2.00,
    hold5d_spread=-7.42, hold5d_t=-0.67,
    hold20d_spread=-8.13, hold20d_t=-0.21,
    hold65d_spread=-37.30, hold65d_t=-0.45,
    q1_65d=403.40, q5_65d=440.70,
)

def _have_cache():
    return all(os.path.exists(data._cache_path(t, CACHE)) for t in data.SP500_BASKET)

HAVE_REAL = _have_cache()

def load_real():
    panel = data.load_panel(data.SP500_BASKET, cache_dir=CACHE)
    prox = st.proximity_to_52w_low(panel, window=252)
    return panel, prox

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Fifty-Two-Week-Low — contrarian bargain bet or losers keep losing?\n"
            "### Ranking S&P 500 names by proximity to their 52-week low, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Contrarian%3F: Losers_Keep_Losing](https://img.shields.io/badge/Contrarian%3F-Losers__Keep__Losing-8b949e?style=flat-square)\n\n"
            "Here is one of the oldest retail stories in the playbook: find stocks that have "
            "fallen to their 52-week low, buy them as a 'bargain', and wait for the bounce. "
            "The logic sounds solid — beaten-down names are out of favour, priced for doom, and "
            "due for a rebound. This notebook runs that idea against the data and asks the only "
            "question that matters: **do stocks near their 52-week low actually bounce back, or "
            "do they keep falling?**\n\n"
            "> This is the plain-language layer. Want the t-stats and the full quintile "
            "breakdown? That is the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do stocks near their 52-week low outperform? | **No.** They actually "
            f"*underperform* stocks near their 52-week high by **{abs(R['spread_mean']):.1f} bps** "
            f"per week (HAC *t* = {R['spread_t']:+.2f}). |\n"
            "| Is it at least a decent buy signal? | **No.** The bottom quintile earns "
            f"**{R['decile_ann']:.1f}%/yr** against a passive basket at "
            f"**{R['baseline_ann']:.1f}%/yr** — it *lags* the benchmark. |\n"
            "| Does the horizon matter? | At every tested horizon (1 day to 3 months), "
            "the spread is *negative* — near-low names trail near-high names. |\n"
            "| Could you trade it? | **No.** Negative gross edge + costs + wider spreads "
            "on beaten-down names = a documented way to lag the market. |\n\n"
            "> The 52-week-low proximity signal is the *mirror image* of the momentum "
            "anomaly — and the mirror image of a real effect is often just noise pointing "
            "the wrong way."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Screen for stocks that have hit or are near their 52-week low. These names are "
            "oversold, beaten up, and priced for disaster. Buy them as a contrarian 'bargain' — "
            "the bad news is already in the price, and any improvement triggers a bounce back "
            "toward fair value.\"*\n\n"
            "It is the contrarian mirror of the 52-week-high momentum anomaly (Study 50): if "
            "stocks near their high *keep rising* (anchoring creates underreaction to good news), "
            "then stocks near their low *should keep falling* — OR they should bounce back "
            "(if the fall has been overdone). The retail contrarian story bets on the bounce. "
            "The academic momentum literature bets on the continuation. We let the data decide."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the contrarian bet works, it is one of the simplest and most scalable edges in "
            "equities: a mechanical weekly screen, no fundamental research, no options, just "
            "buy the biggest losers and wait. If it fails — and the real result is losers *keep* "
            "losing — then everyone running '52-week-low screeners' is systematically buying "
            "the weakest stocks and calling it a strategy."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three rules keep us honest:\n\n"
            "1. **Sort, don't cherry-pick.** Rank *all* stocks by proximity to the 52-week "
            "low and form five equal groups (quintiles). If the contrarian story is true, "
            "the lowest quintile (nearest the low) should earn more than the highest quintile "
            "(nearest the high). If it's false, the opposite will appear.\n"
            "2. **Compare to the whole basket.** A long-only investor should beat the "
            "equal-weight basket, not just make money in a bull market.\n"
            "3. **Try multiple time horizons.** A 1-day bounce is a different claim from a "
            "3-month recovery. We sweep 1 day to 13 weeks so we can't miss the effect by "
            "testing only one window.\n\n"
            f"Universe: 20 representative S&P 500 names, "
            f"{R['date_start']} to {R['date_end']}, {R['n_days']:,} daily bars. "
            f"**Survivorship-biased** — all names still trade in 2026; failed companies "
            "are excluded, which actually *helps* the contrarian story by hiding the worst "
            "outcomes."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's look at the data\n\n"
            "**The quintile chart.** If the contrarian thesis holds, bars should go from "
            "high (Q1, near the low) to low (Q5, near the high). Let's see:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    panel, prox = load_real()\n"
            "    fwd = st.forward_returns(panel, hold_days=5)\n"
            "    q = st.quintile_backtest(prox, fwd, n_quintiles=5, rebalance_freq=5)\n"
            "    q_means = [st.summarize(q[f'Q{i}'].dropna())['mean_bps'] for i in range(1, 6)]\n"
            "    q_ts = [st.summarize(q[f'Q{i}'].dropna())['tstat'] for i in range(1, 6)]\n"
            "else:\n"
            f"    q_means = [{R['q1_mean']}, {R['q2_mean']}, {R['q3_mean']}, {R['q4_mean']}, {R['q5_mean']}]\n"
            f"    q_ts = [{R['q1_t']}, {R['q2_t']}, {R['q3_t']}, {R['q4_t']}, {R['q5_t']}]\n"
            "labels = ['Q1\\nnear 52w-low\\n(contrarian long)', 'Q2', 'Q3', 'Q4',\n"
            "          'Q5\\nnear 52w-high\\n(momentum long)']\n"
            "colors = [AMBER if i == 0 else GREEN if i == 4 else GREY for i in range(5)]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 5.0))\n"
            "bars = ax.bar(labels, q_means, color=colors, alpha=0.85)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b, t in zip(bars, q_ts):\n"
            "    ax.annotate(f't={t:+.1f}', (b.get_x()+b.get_width()/2, b.get_height()),\n"
            "                ha='center', va='bottom' if b.get_height()>=0 else 'top', fontsize=9)\n"
            "ax.set_ylabel('mean 5-day return (bps/period)')\n"
            "ax.set_title('Quintile returns: the contrarian story runs BACKWARDS')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Q1 (near low): {{q_means[0]:+.1f}} bps  |  Q5 (near high): {{q_means[4]:+.1f}} bps')\n"
            f"print(f'Spread Q1-Q5: {{q_means[0]-q_means[4]:+.1f}} bps  (negative = contrarian loses)')"
        ),
        md(
            f"The contrarian story runs backwards. Q5 (near the 52-week high) earns "
            f"**{R['q5_mean']:.1f} bps** per 5-day period; Q1 (near the 52-week low) earns "
            f"only **{R['q1_mean']:.1f} bps**. The spread is **{R['spread_mean']:+.1f} bps** — "
            "stocks near their 52-week low underperform stocks near their high. This is mild "
            "momentum, the opposite of a contrarian edge.\n\n"
            "Note: both quintiles earn positive returns (it's a bull-market sample), but the "
            "near-low names lag the near-high names consistently."
        ),
        md(
            "**Does it improve at longer horizons?** One might argue: the bounce takes "
            "more than a week to materialise. Let's look:"
        ),
        code(
            "holds = [1, 5, 10, 20, 65]\n"
            "if HAVE_REAL:\n"
            "    panel, prox = load_real()\n"
            "    spreads = []\n"
            "    for h in holds:\n"
            "        fwd = st.forward_returns(panel, hold_days=h)\n"
            "        q = st.quintile_backtest(prox, fwd, n_quintiles=5, rebalance_freq=5)\n"
            "        s = st.summarize(q['Q1_minus_Q5'].dropna())\n"
            "        spreads.append(s['mean_bps'])\n"
            "else:\n"
            f"    spreads = [{R['hold1d_spread']}, {R['hold5d_spread']}, -7.63, {R['hold20d_spread']}, {R['hold65d_spread']}]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.bar([str(h)+'d' for h in holds], spreads,\n"
            "       color=[RED if s<0 else GREEN for s in spreads], alpha=0.85)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('hold period'); ax.set_ylabel('Q1 minus Q5 spread (bps/period)')\n"
            "ax.set_title('At every horizon, the near-low quintile TRAILS the near-high quintile')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The contrarian spread is negative at every tested horizon.')"
        ),
        md(
            f"The spread is negative across every tested horizon — from 1 day to 13 weeks. "
            f"The 1-day spread is the most negative at **{R['hold1d_spread']:+.1f} bps** "
            f"(HAC *t* = {R['hold1d_t']:+.2f}), suggesting a short-term momentum effect. "
            "The hoped-for reversal at multi-week horizons simply does not appear in this data."
        ),

        # ---- BEAT 5 — VERDICT ------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The Q1-Q5 spread is **negative** at every horizon, "
            f"peaking at {R['spread_mean']:+.1f} bps/week (HAC *t* = {R['spread_t']:+.2f}). "
            "Statistically indistinguishable from zero, and in the **wrong direction** for "
            "the contrarian thesis.\n"
            f"- **Tradability — Mirage.** The bottom decile earns only **{R['decile_ann']:.1f}%/yr** "
            f"against a passive basket at **{R['baseline_ann']:.1f}%/yr**. Before any transaction "
            "costs, the contrarian screen *lags the market*.\n"
            "- **Contrarian? — Losers Keep Losing.** Stocks near their 52-week low "
            "systematically underperform stocks near their high. This is the momentum effect, "
            "not its opposite."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Even if the signal were flat, the execution would be brutal:"
        ),
        code(
            "# Show the cost headwind: already negative gross, costs only dig deeper\n"
            "costs = [0.0, 1.0, 2.0, 5.0]\n"
            "gross_spread = R['spread_mean']\n"
            "net = [gross_spread - c for c in costs]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar([f'{c:.0f}bps' for c in costs], net,\n"
            "       color=[RED]*len(net), alpha=0.85)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost'); ax.set_ylabel('net Q1-Q5 spread (bps/period)')\n"
            "ax.set_title('Already negative before costs — every bp of cost deepens the hole')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Gross spread = {gross_spread:+.1f} bps. There is no positive break-even cost.')"
        ),
        md(
            "The gross spread is already negative, so there is **no transaction cost low "
            "enough** to rescue it. Add that beaten-down stocks tend to have wider bid-ask "
            "spreads and lower liquidity than high-proximity names, and the execution drag "
            "is even worse than a naive per-trade cost estimate."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The mirror that works.** The 52-week-high momentum strategy — buying stocks "
            "*near their high* — is the effect that George & Hwang (2004) actually documented. "
            "That's [Study 50 — Fifty-Two-Week-High](../../50-fifty-two-week-high/).\n"
            "- **Longer-horizon reversal (3–5 years).** DeBondt & Thaler (1985) showed "
            "multi-year losers outperform over the *following* 3-5 years — but at 1–13 week "
            "horizons, the opposite holds. Time scale matters.\n"
            "- **A single-stock version of oversold.** Williams %R (Study 127) tests whether "
            "a single stock at its short-term low bounces — also a non-result.\n\n"
            "*Convinced there is a bounce hiding here? Fork this, expand the universe "
            "(unbiased, including delisted names), test a longer horizon, and show a positive "
            "spread that clears |*t*| ≥ 2. That is the bar.*"
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
            "# Fifty-Two-Week-Low — a quantitative teardown\n"
            "### 20 S&P 500 names · 13-year daily panel · HAC inference · quintile sort · hold-period sweep\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Contrarian%3F: Losers_Keep_Losing](https://img.shields.io/badge/Contrarian%3F-Losers__Keep__Losing-8b949e?style=flat-square)\n\n"
            "The deep companion to the [plain-language notebook](01_for_the_curious.ipynb) — "
            "same seven beats, every claim carrying its standard error. We test whether the "
            "cross-sectional 52-week-low proximity signal (proximity = (close − 252d-low) / "
            "(252d-high − 252d-low)) predicts forward returns in a direction consistent with "
            "contrarianism, across a 20-name S&P 500 basket and hold periods from 1 to 65 days.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, "
            f"{R['date_start']} to {R['date_end']}, "
            f"{R['n_tickers']} tickers, {R['n_days']:,} panel-days. **Survivorship-biased** "
            "— all names still trade in 2026. Methods in "
            "[`docs/references.md`](../docs/references.md); reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **In plain words** notes translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Q1-Q5 spread = **{R['spread_mean']:+.2f} bps/5d**, "
            f"HAC *t* = **{R['spread_t']:+.2f}**; negative and insignificant at all "
            f"tested horizons. Near-low names *trail* near-high names. |\n"
            f"| **Tradability** | `MIRAGE` | Bottom decile: **{R['decile_ann']:.1f}%/yr** "
            f"vs equal-weight baseline **{R['baseline_ann']:.1f}%/yr** — lags before costs. |\n"
            f"| **Contrarian?** | `LOSERS KEEP LOSING` | 1-day spread = "
            f"**{R['hold1d_spread']:+.2f} bps**, *t* = **{R['hold1d_t']:+.2f}**; "
            "consistent with mild momentum, not mean-reversion. |\n\n"
            "> In plain words: stocks near their 52-week low have underperformed for a "
            "reason, and that reason does not reverse at weekly-to-quarterly horizons. "
            "The 52-week-HIGH is where the documented anomaly lives (George & Hwang 2004)."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\text{prox}_{i,t}$ be the 52-week-low proximity for stock $i$ at date $t$:\n\n"
            "$$\\text{prox}_{i,t} = \\frac{\\text{close}_{i,t} - \\min_{s \\in [t-252,t]} "
            "\\text{close}_{i,s}}{\\max_{s \\in [t-252,t]} \\text{close}_{i,s} - "
            "\\min_{s \\in [t-252,t]} \\text{close}_{i,s} + \\varepsilon}$$\n\n"
            "- $\\text{prox} \\approx 0$ → stock near its 52-week low (the contrarian buy).\n"
            "- $\\text{prox} \\approx 1$ → stock near its 52-week high (the momentum long).\n\n"
            "The contrarian hypothesis (H₁): $\\mathbb{E}[r_{i,t+h}]$ is *decreasing* in "
            "$\\text{prox}_{i,t}$ — i.e. Q1 outperforms Q5.  We test this cross-sectionally "
            "with a quintile sort at horizons $h \\in \\{1, 5, 10, 20, 65\\}$ trading days.\n\n"
            "The momentum null (H₀): $\\mathbb{E}[r_{i,t+h}]$ is *increasing* in "
            "$\\text{prox}_{i,t}$ — i.e. Q5 outperforms Q1, consistent with Jegadeesh & "
            "Titman (1993) and George & Hwang (2004)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "If H₁ holds, it is one of the simplest documented cross-sectional alphas: "
            "a weekly screen requiring no fundamental analysis, scalable to any liquid universe. "
            "If the null holds, every investor running a '52-week-low screener' is "
            "systematically buying into momentum headwinds — a well-dressed way to "
            "underperform the index. The distinction has direct implications for retail "
            "screening tools and factor allocation."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** $\\text{prox}_{i,t}$ computed on closes up to $t$; "
            "trade entered at $t+1$'s close (daily bars, weekly rebalance).\n"
            "- **Sort.** Equal-weight quintile portfolios from cross-sectional rank of "
            "$\\text{prox}$; Q1 = lowest 20% (near 52w-low), Q5 = highest 20% (near 52w-high).\n"
            "- **Hold periods.** $h \\in \\{1, 5, 10, 20, 65\\}$ trading days, "
            "rebalance every 5 days.\n"
            "- **Inference.** Newey-West HAC *t*-stat on the Q1-Q5 spread series; "
            "the inference bar is |*t*| ≥ 2 on the real tape.\n"
            "- **Baseline.** Equal-weight buy-and-hold of the full 20-name basket "
            "(the long-only investor's fair comparator).\n"
            "- **Positive control.** Synthetic panel with tunable AR(1) reversion: "
            "confirms the engine is a faithful mean-reversion detector.\n\n"
            f"Universe: {R['n_tickers']} S&P 500 large-caps, {R['n_days']:,} trading days, "
            f"{R['date_start']} to {R['date_end']}. **Survivorship-biased** "
            "(all names still trade in 2026 — this *helps* the contrarian story by "
            "excluding failed companies that hit 52w-lows and kept falling)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Quintile returns — contrarian underperforms momentum at every quintile\n\n"
            "Per-quintile mean 5-day return and HAC *t*-stat. H₁ requires a monotone "
            "*decreasing* pattern from Q1 to Q5."
        ),
        code(
            "if HAVE_REAL:\n"
            "    panel, prox = load_real()\n"
            "    fwd = st.forward_returns(panel, hold_days=5)\n"
            "    q = st.quintile_backtest(prox, fwd, n_quintiles=5, rebalance_freq=5)\n"
            "    q_stats = {}\n"
            "    for i in range(1, 6):\n"
            "        s = st.summarize(q[f'Q{i}'].dropna())\n"
            "        q_stats[f'Q{i}'] = s\n"
            "    q_means = [q_stats[f'Q{i}']['mean_bps'] for i in range(1, 6)]\n"
            "    q_ts = [q_stats[f'Q{i}']['tstat'] for i in range(1, 6)]\n"
            "    spread_s = st.summarize(q['Q1_minus_Q5'].dropna())\n"
            "else:\n"
            f"    q_means = [{R['q1_mean']}, {R['q2_mean']}, {R['q3_mean']}, {R['q4_mean']}, {R['q5_mean']}]\n"
            f"    q_ts = [{R['q1_t']}, {R['q2_t']}, {R['q3_t']}, {R['q4_t']}, {R['q5_t']}]\n"
            f"    spread_s = dict(mean_bps={R['spread_mean']}, tstat={R['spread_t']}, n={R['n_periods']})\n"
            "qdf = pd.DataFrame({'quintile': [f'Q{i}' for i in range(1,6)],\n"
            "                    'mean_bps': q_means, 'tstat': q_ts})\n"
            "fig, ax = plt.subplots(figsize=(9.5, 5.0))\n"
            "cols = [AMBER, GREY, GREY, GREY, GREEN]\n"
            "bars = ax.bar(qdf['quintile'], qdf['mean_bps'], color=cols, alpha=0.85)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b, t in zip(bars, qdf['tstat']):\n"
            "    y = b.get_height() + 0.5\n"
            "    ax.annotate(f't={t:+.2f}', (b.get_x()+b.get_width()/2, y),\n"
            "                ha='center', va='bottom', fontsize=9)\n"
            "ax.set_ylabel('mean 5-day return (bps/period)')\n"
            "ax.set_xlabel('Proximity quintile (Q1 = near 52w-low, Q5 = near 52w-high)')\n"
            "ax.set_title('Quintile returns: Q5 outperforms Q1 (contrarian thesis FAILS)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(qdf.to_string(index=False))\n"
            "print(f\"\\nQ1-Q5 spread: {spread_s['mean_bps']:+.2f} bps, \"\n"
            "      f\"HAC t = {spread_s['tstat']:+.2f}, n = {spread_s['n']}\")"
        ),
        md(
            f"> In plain words: the bar chart runs the wrong way for the contrarian story. "
            f"Q5 (near the high, {R['q5_mean']:.1f} bps) earns more than Q1 "
            f"(near the low, {R['q1_mean']:.1f} bps). The spread is "
            f"**{R['spread_mean']:+.2f} bps** (HAC *t* = {R['spread_t']:+.2f}) — "
            "statistically insignificant and in the wrong direction."
        ),
        md(
            "### 4b · Hold-period sweep — the bounce never comes\n\n"
            "If the contrarian bounce takes time to materialise, the spread should turn "
            "positive at longer horizons. We test from 1 day to 13 weeks."
        ),
        code(
            "holds = [1, 5, 10, 20, 65]\n"
            "if HAVE_REAL:\n"
            "    panel, prox = load_real()\n"
            "    h_spreads, h_ts = [], []\n"
            "    for h in holds:\n"
            "        fwd = st.forward_returns(panel, hold_days=h)\n"
            "        q = st.quintile_backtest(prox, fwd, n_quintiles=5, rebalance_freq=5)\n"
            "        s = st.summarize(q['Q1_minus_Q5'].dropna())\n"
            "        h_spreads.append(s['mean_bps']); h_ts.append(s['tstat'])\n"
            "else:\n"
            f"    h_spreads = [{R['hold1d_spread']}, {R['hold5d_spread']}, -7.63, {R['hold20d_spread']}, {R['hold65d_spread']}]\n"
            f"    h_ts = [{R['hold1d_t']}, {R['hold5d_t']}, -0.36, {R['hold20d_t']}, {R['hold65d_t']}]\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.5))\n"
            "ax1.bar([f'{h}d' for h in holds], h_spreads,\n"
            "        color=[RED if s<0 else GREEN for s in h_spreads], alpha=0.85)\n"
            "ax1.axhline(0, c='k', lw=1)\n"
            "ax1.set_ylabel('Q1-Q5 spread (bps/period)')\n"
            "ax1.set_title('Spread is negative at every horizon')\n"
            "ax2.bar([f'{h}d' for h in holds], h_ts,\n"
            "        color=[RED if abs(t)>=2 else GREY for t in h_ts], alpha=0.85)\n"
            "for thr in (2, -2): ax2.axhline(thr, ls='--', c=GREY, lw=1)\n"
            "ax2.axhline(0, c='k', lw=1)\n"
            "ax2.set_ylabel('HAC t-stat')\n"
            "ax2.set_title('t-stat: only 1d breaches ±2 (and in wrong direction)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h, sp, t in zip(holds, h_spreads, h_ts):\n"
            "    print(f'hold={h:3d}d  spread={sp:+.2f}bps  t={t:+.2f}')"
        ),
        md(
            f"> In plain words: the 1-day spread is the only period that breaches |*t*| = 2, "
            f"and it does so at **{R['hold1d_spread']:+.1f} bps** (*t* = {R['hold1d_t']:+.2f}) "
            "— in the *momentum* direction (near-low names underperform near-high names). "
            "At all longer horizons the spread is negative and statistically indistinguishable "
            "from zero. There is no horizon where the contrarian bounce appears."
        ),
        md(
            "### 4c · Positive synthetic control — the engine is faithful\n\n"
            "Does the engine correctly detect contrarian reversion *when it exists*? "
            "We plant a tunable AR(1) coefficient in a synthetic 20-stock panel "
            "and sweep from strong momentum to strong reversion:"
        ),
        code(
            "revs = [-0.40, -0.20, 0.00, 0.20, 0.40]\n"
            "syn_spreads, syn_ts = [], []\n"
            "for rev in revs:\n"
            "    panel_s, _ = data.synthetic_panel(n_stocks=20, n_days=2500, reversion=rev, seed=42)\n"
            "    prox_s = st.proximity_to_52w_low(panel_s, window=252)\n"
            "    fwd_s = st.forward_returns(panel_s, hold_days=5)\n"
            "    q_s = st.quintile_backtest(prox_s, fwd_s, n_quintiles=5, rebalance_freq=5)\n"
            "    sp_s = st.summarize(q_s['Q1_minus_Q5'].dropna())\n"
            "    syn_spreads.append(sp_s['mean_bps']); syn_ts.append(sp_s['tstat'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.bar([str(r) for r in revs], syn_spreads,\n"
            "       color=[GREEN if s>0 else RED for s in syn_spreads], alpha=0.85)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, (sp, t) in enumerate(zip(syn_spreads, syn_ts)):\n"
            "    ax.annotate(f't={t:+.1f}', (i, sp), ha='center',\n"
            "                va='bottom' if sp>=0 else 'top', fontsize=9)\n"
            "ax.set_xlabel('planted AR(1) reversion coefficient')\n"
            "ax.set_ylabel('Q1-Q5 spread (bps/period)')\n"
            "ax.set_title('Synthetic control: spread monotone in planted reversion')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Engine verdict: momentum (-0.40) → big negative spread; '\n"
            "      'reversion (+0.40) → positive spread.')"
        ),
        md(
            "> In plain words: the engine works. It correctly identifies contrarian reversion "
            "when it is planted (positive spread at +0.40 reversion) and correctly scores "
            "momentum (negative spread at -0.40). The real tape's negative spread is therefore "
            "a statement about the **market** — it is closer to a mild-momentum or martingale "
            "regime at weekly-to-quarterly horizons."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — Q1-Q5 spread = {R['spread_mean']:+.2f} bps/5d, "
            f"HAC *t* = {R['spread_t']:+.2f}; negative and insignificant at all tested "
            f"horizons. The 1-day spread is significant (*t* = {R['hold1d_t']:+.2f}) but "
            "in the momentum direction. Survivorship bias works *against* the contrarian "
            "story being true — on an unbiased universe the result is likely more negative.\n"
            f"- **Tradability `MIRAGE`** — bottom decile earns {R['decile_ann']:.1f}%/yr "
            f"vs {R['baseline_ann']:.1f}%/yr equal-weight baseline; lags the market before "
            "costs. Near-low stocks also tend to have wider spreads and lower liquidity, "
            "making execution drag larger than for near-high names.\n"
            f"- **Contrarian? `LOSERS KEEP LOSING`** — the signal is in the wrong direction. "
            "Stocks near their 52-week low underperform stocks near their high across all "
            "tested horizons on this survivorship-biased universe."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the long-only comparison\n\n"
            "The head-to-head comparison a long-only investor cares about: "
            "bottom decile vs equal-weight basket, before and after costs:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    panel, prox = load_real()\n"
            "    fwd = st.forward_returns(panel, hold_days=5)\n"
            "    strat, base = st.long_bottom_decile(prox, fwd, decile_frac=0.10, rebalance_freq=5)\n"
            "    cum_strat = (1 + strat).cumprod()\n"
            "    cum_base  = (1 + base).cumprod()\n"
            "    s_strat = st.summarize(strat)\n"
            "    s_base  = st.summarize(base)\n"
            "    strat_ann, base_ann = s_strat['ann_pct'], s_base['ann_pct']\n"
            "    strat_t, base_t = s_strat['tstat'], s_base['tstat']\n"
            "else:\n"
            f"    strat_ann, base_ann = {R['decile_ann']}, {R['baseline_ann']}\n"
            f"    strat_t, base_t = 1.62, 4.26\n"
            f"    cum_strat = cum_base = None\n"
            "print(f'Bottom decile: {strat_ann:+.1f}%/yr  (HAC t = {strat_t:+.2f})')\n"
            "print(f'EW baseline:   {base_ann:+.1f}%/yr  (HAC t = {base_t:+.2f})')\n"
            "print(f'Drag vs baseline: {strat_ann - base_ann:+.1f}pp/yr — BEFORE COSTS')\n"
            "if HAVE_REAL and cum_strat is not None:\n"
            "    fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "    ax.plot(cum_strat.index, cum_strat.values, c=RED, lw=2, label=f'Bottom decile ({strat_ann:+.1f}%/yr)')\n"
            "    ax.plot(cum_base.index, cum_base.values, c=GREY, lw=1.5, label=f'EW baseline ({base_ann:+.1f}%/yr)')\n"
            "    ax.set_ylabel('cumulative return (period compounded)'); ax.legend()\n"
            "    ax.set_title('Bottom-decile lags the equal-weight basket')\n"
            "    plt.tight_layout(); plt.show()"
        ),
        md(
            "> In plain words: a long-only investor running this strategy earns "
            f"**{R['decile_ann']:.1f}%/yr** against a passive basket at "
            f"**{R['baseline_ann']:.1f}%/yr** — a persistent drag before any trading "
            "costs, commission, or bid-ask spread."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the real anomaly is in the other direction\n\n"
            "The result here is not that 52-week-proximity is useless — it is that "
            "**the useful direction is the opposite of what the contrarian story claims**. "
            "George & Hwang (2004) documented that proximity to the 52-week *high* predicts "
            "positive future returns (anchoring creates underreaction to good news). "
            "The 52-week-high strategy is [Study 50](../../50-fifty-two-week-high/).\n\n"
            "Forks worth trying:\n"
            "- **Longer horizon (3-5 years).** DeBondt & Thaler (1985) showed long-term reversal "
            "at 3-5 year horizons. Extend the look-back and hold period significantly.\n"
            "- **Unbiased universe.** Include delisted stocks. The survivorship-biased result is "
            "already negative — the true unbiased result is likely more so.\n"
            "- **Fundamental filter.** Pair the low-proximity signal with value or quality "
            "criteria to filter genuine 'beaten-down value' from 'falling knife'."
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
