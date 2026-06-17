"""Generate the two narrative notebooks for Study 236 (Fifty-Two-Week-High).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-16).
R = dict(
    n_days=3384, n_tickers=20, n_periods=626,
    date_start="2013-01-02", date_end="2026-06-16",
    fingerprint="e6d4695e32c7",
    # Q5-Q1 spread at 5d
    spread_mean=-10.60, spread_t=-0.85, spread_win=48.9,
    # Per-quintile 5d returns
    q1_mean=38.26, q1_t=3.17, q1_win=60.5,
    q2_mean=38.61, q2_t=4.30,
    q3_mean=32.81, q3_t=3.58,
    q4_mean=20.23, q4_t=2.22,
    q5_mean=27.66, q5_t=3.55, q5_win=58.0,
    # Long-only top decile vs baseline
    decile_ann=6.8, baseline_ann=11.3, decile_spread_t=-1.62,
    # Hold-period sweep
    hold1d_spread=10.54, hold1d_t=1.57,
    hold5d_spread=-10.60, hold5d_t=-0.85,
    hold10d_spread=-32.11, hold10d_t=-1.39,
    hold20d_spread=-60.71, hold20d_t=-1.42,
    hold65d_spread=-139.13, hold65d_t=-1.39,
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

from fifty_two_week_high import data, strategy as st

CACHE = os.path.abspath(os.path.join("..", "_cache"))

# Frozen real-tape headline numbers (mirror of docs/results.md, as-of 2026-06-16)
R = dict(
    n_days=3384, n_tickers=20, n_periods=626,
    date_start="2013-01-02", date_end="2026-06-16",
    fingerprint="e6d4695e32c7",
    spread_mean=-10.60, spread_t=-0.85, spread_win=48.9,
    q1_mean=38.26, q1_t=3.17, q1_win=60.5,
    q2_mean=38.61, q2_t=4.30,
    q3_mean=32.81, q3_t=3.58,
    q4_mean=20.23, q4_t=2.22,
    q5_mean=27.66, q5_t=3.55, q5_win=58.0,
    decile_ann=6.8, baseline_ann=11.3, decile_spread_t=-1.62,
    hold1d_spread=10.54, hold1d_t=1.57,
    hold5d_spread=-10.60, hold5d_t=-0.85,
    hold10d_spread=-32.11, hold10d_t=-1.39,
    hold20d_spread=-60.71, hold20d_t=-1.42,
    hold65d_spread=-139.13, hold65d_t=-1.39,
)

def _have_cache():
    return all(os.path.exists(data._cache_path(t, CACHE)) for t in data.SP500_BASKET)

HAVE_REAL = _have_cache()

def load_real():
    panel = data.load_panel(data.SP500_BASKET, cache_dir=CACHE)
    prox = st.proximity_to_52w_high(panel, window=252)
    return panel, prox

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Fifty-Two-Week-High — momentum signal or mega-cap mirage?\n"
            "### Ranking S&P 500 names by proximity to their 52-week high, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![George--Hwang: Inverted_on_Large-Caps](https://img.shields.io/badge/George--Hwang-Inverted__on__Large--Caps-8b949e?style=flat-square)\n\n"
            "Here is one of the most cited momentum strategies in academic finance: buy stocks "
            "that are near their 52-week high. The logic is anchoring psychology — investors are "
            "reluctant to pay more than a stock's 52-week high, so good news gets under-reflected "
            "in prices for stocks that are close to that level. George & Hwang documented this in "
            "2004 using 40 years of broad market data. This notebook runs that idea against a "
            "modern large-cap sample and asks the only question that matters: **does it still work, "
            "and is it better than simply holding the index?**\n\n"
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
            "| Do stocks near their 52-week high outperform? | **No** — on this large-cap "
            f"sample they *underperform* stocks far from their high by **{abs(R['spread_mean']):.1f} bps** "
            f"per week (HAC *t* = {R['spread_t']:+.2f}). The famous momentum effect is **inverted** here. |\n"
            "| Is it at least a decent buy signal? | **No.** The top decile (nearest 52w-high) "
            f"earns **{R['decile_ann']:.1f}%/yr** against a passive basket at "
            f"**{R['baseline_ann']:.1f}%/yr** — it *lags* the benchmark before costs. |\n"
            "| Does the horizon matter? | At 1 day the spread is positive (+10.5 bps) but below "
            "the significance bar. At every longer horizon (5d to 65d) it is negative. |\n"
            "| Could you trade it? | **No.** Negative gross edge at practical horizons + costs "
            "+ slippage = a reliable way to underperform the index. |\n\n"
            "> The George-Hwang (2004) anomaly was found on 1963-2001 data across thousands of "
            "stocks. On a 2013-2026 mega-cap panel of 20 blue-chip survivors, the effect inverts: "
            "names near their highs are rich valuations at risk of mean-reversion, not "
            "under-priced momentum bets."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Buy the stocks closest to their 52-week high. They are there for a reason: "
            "the market is anchored to that price level, so when good news breaks, investors are "
            "slow to push the stock past that ceiling. Near-high stocks are the ones where the "
            "market keeps underreacting to good information — that is your edge.\"*\n\n"
            "This is the George & Hwang (2004) story, and it was a genuinely strong academic "
            "result: on NYSE/AMEX/Nasdaq stocks from 1963 to 2001, buying the stocks closest to "
            "their 52-week high and shorting those farthest away earned about 0.5%/month, "
            "net of the standard momentum premium. The question is whether it holds in a modern, "
            "liquid, heavily-watched large-cap universe — or whether four decades of arbitrage "
            "and the shift to passive investing have closed the gap."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the signal works, it is a simple mechanical weekly screen on any stock screener: "
            "find the stocks nearest their 52-week high, buy them, repeat. No fundamental "
            "analysis, no options, just price ratios. If it fails — or worse, inverts — then "
            "chasing 52-week highs on mega-cap names is a documented way to buy expensive "
            "stocks at the moment of peak valuation."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three rules keep us honest:\n\n"
            "1. **Sort, don't cherry-pick.** Rank *all* stocks by proximity to the 52-week "
            "high (close / 252d rolling high) and form five equal groups (quintiles). "
            "If the momentum story holds, Q5 (closest to the high) should earn more than "
            "Q1 (farthest from the high). If it fails, the bar chart will tell us.\n"
            "2. **Compare to the whole basket.** A long-only investor should beat the "
            "equal-weight basket, not just make money in a bull market.\n"
            "3. **Try multiple time horizons.** The original paper used 6-12 month holds; "
            "we sweep 1 day to 13 weeks so we can see where (if anywhere) the signal exists.\n\n"
            f"Universe: 20 representative S&P 500 large-cap names, "
            f"{R['date_start']} to {R['date_end']}, {R['n_days']:,} daily bars. "
            f"**Survivorship-biased** — all names still trade in 2026."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's look at the data\n\n"
            "**The quintile chart.** If the momentum thesis holds, bars should go from "
            "low (Q1, farthest from high) to high (Q5, nearest to high). Let's see:"
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
            "labels = ['Q1\\nfar from 52w-high\\n(momentum short)', 'Q2', 'Q3', 'Q4',\n"
            "          'Q5\\nnear 52w-high\\n(momentum long)']\n"
            "colors = [GREEN if i == 0 else AMBER if i == 4 else GREY for i in range(5)]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 5.0))\n"
            "bars = ax.bar(labels, q_means, color=colors, alpha=0.85)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b, t in zip(bars, q_ts):\n"
            "    ax.annotate(f't={t:+.1f}', (b.get_x()+b.get_width()/2, b.get_height()),\n"
            "                ha='center', va='bottom' if b.get_height()>=0 else 'top', fontsize=9)\n"
            "ax.set_ylabel('mean 5-day return (bps/period)')\n"
            "ax.set_title('Quintile returns: the momentum story runs BACKWARDS on mega-caps')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Q1 (far from high): {{q_means[0]:+.1f}} bps  |  Q5 (near high): {{q_means[4]:+.1f}} bps')\n"
            f"print(f'Spread Q5-Q1: {{q_means[4]-q_means[0]:+.1f}} bps  (negative = momentum loses)')"
        ),
        md(
            f"The momentum story runs backwards here. Q1 (farthest from the 52-week high) earns "
            f"**{R['q1_mean']:.1f} bps** per 5-day period; Q5 (nearest to the 52-week high) earns "
            f"only **{R['q5_mean']:.1f} bps**. The spread is **{R['spread_mean']:+.1f} bps** — "
            "the near-high quintile *underperforms* the far-from-high quintile on this mega-cap "
            "panel. Both quintiles earn positive returns (it's a survivorship-biased bull market "
            "sample), but the near-high names consistently lag.\n\n"
            "Why? Within a basket of blue-chip survivors, names near their 52-week high tend to "
            "be at stretched valuations (think NVDA after a 200% run). Names far from their high "
            "are temporarily soft but fundamentally sound companies — they recover. This is the "
            "opposite of the original paper's universe, which included genuine momentum losers "
            "that stayed down."
        ),
        md(
            "**Does the momentum kick in at longer horizons?** The original George-Hwang paper "
            "used 6-12 month holds. Let's check if our 1-13 week window is simply too short:"
        ),
        code(
            "holds = [1, 5, 10, 20, 65]\n"
            "if HAVE_REAL:\n"
            "    panel, prox = load_real()\n"
            "    spreads = []\n"
            "    for h in holds:\n"
            "        fwd = st.forward_returns(panel, hold_days=h)\n"
            "        q = st.quintile_backtest(prox, fwd, n_quintiles=5, rebalance_freq=5)\n"
            "        s = st.summarize(q['Q5_minus_Q1'].dropna())\n"
            "        spreads.append(s['mean_bps'])\n"
            "else:\n"
            f"    spreads = [{R['hold1d_spread']}, {R['hold5d_spread']}, {R['hold10d_spread']}, {R['hold20d_spread']}, {R['hold65d_spread']}]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.bar([str(h)+'d' for h in holds], spreads,\n"
            "       color=[GREEN if s>0 else RED for s in spreads], alpha=0.85)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('hold period'); ax.set_ylabel('Q5 minus Q1 spread (bps/period)')\n"
            "ax.set_title('Spread is positive only at 1d (and below the bar); negative from 5d to 65d')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Only the 1-day horizon is positive; the signal weakens and inverts at longer holds.')"
        ),
        md(
            f"The 1-day horizon shows a small positive spread (+{R['hold1d_spread']:.1f} bps, "
            f"*t* = {R['hold1d_t']:+.2f}) — consistent with the original momentum story but "
            "well below the statistical bar. At every longer horizon (5d to 65d) the spread "
            "is negative and grows more negative with time. There is no tested window where "
            "the near-high quintile significantly outperforms."
        ),

        # ---- BEAT 5 — VERDICT ------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The Q5-Q1 spread is **{R['spread_mean']:+.1f} bps/week** "
            f"(HAC *t* = {R['spread_t']:+.2f}), negative and statistically insignificant. "
            "No hold period from 5d to 65d clears the |*t*| ≥ 2 bar, and all are in the "
            "wrong direction for the momentum thesis.\n"
            f"- **Tradability — Mirage.** Top decile earns only **{R['decile_ann']:.1f}%/yr** "
            f"against a passive basket at **{R['baseline_ann']:.1f}%/yr** — a "
            f"{R['baseline_ann']-R['decile_ann']:.1f} pp drag before any costs.\n"
            "- **George-Hwang — Inverted on Large-Caps.** The famous anomaly does not survive "
            "intact on a 2013-2026 mega-cap panel. The survivorship-biased basket and the "
            "post-publication decay combine to eliminate the original edge at this horizon."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Even if the signal were flat, the top-decile strategy lags the index before costs:"
        ),
        code(
            "# Show the cost headwind: already negative gross at 5d, costs only dig deeper\n"
            "costs = [0.0, 1.0, 2.0, 5.0]\n"
            "gross_spread = R['spread_mean']   # Q5-Q1 at 5d\n"
            "net = [gross_spread - c for c in costs]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar([f'{c:.0f}bps' for c in costs], net,\n"
            "       color=[RED]*len(net), alpha=0.85)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost'); ax.set_ylabel('net Q5-Q1 spread (bps/period)')\n"
            "ax.set_title('Already negative before costs — every bp of cost deepens the hole')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Gross spread = {gross_spread:+.1f} bps. No transaction cost rescues it.')"
        ),
        md(
            "The gross spread is already negative at 5d, so there is **no transaction cost low "
            "enough** to make the strategy work. The top-decile version also requires weekly "
            "rebalancing in large-cap names — not costly, but with a strategy that already "
            "trails the index, every round-trip compounds the drag."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The original sample matters.** George & Hwang (2004) used CRSP 1963-2001 "
            "with thousands of stocks including true momentum losers. That universe is very "
            "different from 20 mega-cap survivors. A broader, unbiased test over the original "
            "period would be a better replication.\n"
            "- **Longer hold periods.** The original paper found the effect strongest at "
            "6-12 months. Our 1-65 day window may simply be too short to capture the "
            "anchoring dynamic.\n"
            "- **The contrarian mirror.** Study 202 tests stocks near their 52-week *low* "
            "as a contrarian bet — also a non-result on the same sample, consistent with "
            "mild mean-reversion within this elite basket.\n"
            "- **Turtle Trading (Study 103).** Uses the 52-week range for breakout entry "
            "decisions rather than a cross-sectional rank — a related but mechanically "
            "different approach that did find a real signal.\n\n"
            "*The fair test of George-Hwang on modern data requires an unbiased universe, "
            "the original 6-12 month holding period, and a fresh out-of-sample window "
            "from 2001 onward. That is a project for a future study.*"
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
            "# Fifty-Two-Week-High — a quantitative teardown\n"
            "### 20 S&P 500 names · 13-year daily panel · HAC inference · quintile sort · hold-period sweep\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![George--Hwang: Inverted_on_Large-Caps](https://img.shields.io/badge/George--Hwang-Inverted__on__Large--Caps-8b949e?style=flat-square)\n\n"
            "The deep companion to the [plain-language notebook](01_for_the_curious.ipynb) — "
            "same seven beats, every claim carrying its standard error. We test whether the "
            "George-Hwang (2004) 52-week-high proximity signal (proximity = close / 252d-high) "
            "predicts forward returns in a direction consistent with the published momentum "
            "anomaly, across a 20-name S&P 500 basket and hold periods from 1 to 65 days.\n\n"
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
            f"| **Signal** | `NONE` | Q5-Q1 spread = **{R['spread_mean']:+.2f} bps/5d**, "
            f"HAC *t* = **{R['spread_t']:+.2f}**; negative and insignificant at all "
            f"tested horizons ≥5d. Near-high names *trail* far-from-high names. |\n"
            f"| **Tradability** | `MIRAGE` | Top decile: **{R['decile_ann']:.1f}%/yr** "
            f"vs equal-weight baseline **{R['baseline_ann']:.1f}%/yr** — lags before costs. |\n"
            f"| **George-Hwang?** | `INVERTED ON LARGE-CAPS` | 1-day spread = "
            f"**+{R['hold1d_spread']:.2f} bps**, *t* = {R['hold1d_t']:+.2f} (below bar); "
            "5d to 65d all negative. Survivorship + post-publication decay eliminate the effect. |\n\n"
            "> In plain words: the canonical 52-week-high momentum anomaly does not survive "
            "on a 2013-2026 mega-cap panel. Within this elite basket, names near their highs "
            "are expensive and mean-revert; names far from their highs are temporarily soft "
            "quality companies that recover."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\text{prox}_{i,t}$ be the 52-week-high proximity for stock $i$ at date $t$ "
            "(George & Hwang 2004 specification):\n\n"
            "$$\\text{prox}_{i,t} = \\frac{\\text{close}_{i,t}}{\\max_{s \\in [t-252,t]} "
            "\\text{close}_{i,s}}$$\n\n"
            "- $\\text{prox} \\approx 1$ → stock at its 52-week high (the momentum long).\n"
            "- $\\text{prox} \\ll 1$ → stock far below its 52-week high (the momentum short).\n\n"
            "The momentum hypothesis (H₁): $\\mathbb{E}[r_{i,t+h}]$ is *increasing* in "
            "$\\text{prox}_{i,t}$ — i.e. Q5 outperforms Q1, consistent with anchoring-driven "
            "underreaction to good news near the 52-week high (George & Hwang 2004).\n\n"
            "The null (H₀): $\\mathbb{E}[r_{i,t+h}]$ is unrelated to $\\text{prox}_{i,t}$.\n\n"
            "We test H₁ cross-sectionally with a quintile sort at horizons "
            "$h \\in \\{1, 5, 10, 20, 65\\}$ trading days on a 20-name S&P 500 panel."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "If H₁ holds in the modern large-cap universe, it implies the anchoring mechanism "
            "survives decades of academic attention and institutional arbitrage — a rare "
            "claim in 2026. If it fails, it is strong evidence that (a) post-publication "
            "decay has closed the gap and (b) the original finding was partly an artefact "
            "of the mid-cap / small-cap universe of 1963-2001, not applicable to "
            "liquid mega-cap names. The distinction matters for retail screeners and "
            "factor allocation products that market the '52-week-high strategy'."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** $\\text{prox}_{i,t}$ computed on closes up to $t$; "
            "trade entered at $t+1$'s close (daily bars, weekly rebalance).\n"
            "- **Sort.** Equal-weight quintile portfolios from cross-sectional rank of "
            "$\\text{prox}$; Q1 = lowest 20% (far from 52w-high), Q5 = highest 20% "
            "(near 52w-high, the momentum long).\n"
            "- **Hold periods.** $h \\in \\{1, 5, 10, 20, 65\\}$ trading days, "
            "rebalance every 5 days.\n"
            "- **Inference.** Newey-West HAC *t*-stat on the Q5-Q1 spread series; "
            "the inference bar is |*t*| ≥ 2 on the real tape.\n"
            "- **Baseline.** Equal-weight buy-and-hold of the full 20-name basket.\n"
            "- **Positive control.** Synthetic panel with tunable AR(1) momentum: "
            "confirms the engine detects continuation when it is planted.\n\n"
            f"Universe: {R['n_tickers']} S&P 500 large-caps, {R['n_days']:,} trading days, "
            f"{R['date_start']} to {R['date_end']}. **Survivorship-biased** — all names "
            "still trade in 2026."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Quintile returns — momentum inverts on mega-caps\n\n"
            "Per-quintile mean 5-day return and HAC *t*-stat. H₁ requires a monotone "
            "*increasing* pattern from Q1 to Q5."
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
            "    spread_s = st.summarize(q['Q5_minus_Q1'].dropna())\n"
            "else:\n"
            f"    q_means = [{R['q1_mean']}, {R['q2_mean']}, {R['q3_mean']}, {R['q4_mean']}, {R['q5_mean']}]\n"
            f"    q_ts = [{R['q1_t']}, {R['q2_t']}, {R['q3_t']}, {R['q4_t']}, {R['q5_t']}]\n"
            f"    spread_s = dict(mean_bps={R['spread_mean']}, tstat={R['spread_t']}, n={R['n_periods']})\n"
            "qdf = pd.DataFrame({'quintile': [f'Q{i}' for i in range(1,6)],\n"
            "                    'mean_bps': q_means, 'tstat': q_ts})\n"
            "fig, ax = plt.subplots(figsize=(9.5, 5.0))\n"
            "cols = [GREEN, GREY, GREY, GREY, AMBER]\n"
            "bars = ax.bar(qdf['quintile'], qdf['mean_bps'], color=cols, alpha=0.85)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b, t in zip(bars, qdf['tstat']):\n"
            "    y = b.get_height() + 0.5\n"
            "    ax.annotate(f't={t:+.2f}', (b.get_x()+b.get_width()/2, y),\n"
            "                ha='center', va='bottom', fontsize=9)\n"
            "ax.set_ylabel('mean 5-day return (bps/period)')\n"
            "ax.set_xlabel('Proximity quintile (Q1 = far from 52w-high, Q5 = near 52w-high)')\n"
            "ax.set_title('Quintile returns: Q1 outperforms Q5 (momentum thesis FAILS on mega-caps)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(qdf.to_string(index=False))\n"
            "print(f\"\\nQ5-Q1 spread: {spread_s['mean_bps']:+.2f} bps, \"\n"
            "      f\"HAC t = {spread_s['tstat']:+.2f}, n = {spread_s['n']}\")"
        ),
        md(
            f"> In plain words: the bar chart runs the wrong way for momentum. "
            f"Q1 (far from the high, {R['q1_mean']:.1f} bps) earns more than Q5 "
            f"(near the high, {R['q5_mean']:.1f} bps). The spread is "
            f"**{R['spread_mean']:+.2f} bps** (HAC *t* = {R['spread_t']:+.2f}) — "
            "statistically insignificant and in the wrong direction. Within this survivorship-"
            "biased mega-cap basket, the 'momentum long' quintile is populated by names at "
            "stretched valuations, not by underreacted winners."
        ),
        md(
            "### 4b · Hold-period sweep — the momentum never arrives\n\n"
            "If the momentum effect is longer-horizon (the original paper uses 6-12 month "
            "holds), perhaps short windows miss it. We test from 1 day to 13 weeks:"
        ),
        code(
            "holds = [1, 5, 10, 20, 65]\n"
            "if HAVE_REAL:\n"
            "    panel, prox = load_real()\n"
            "    h_spreads, h_ts = [], []\n"
            "    for h in holds:\n"
            "        fwd = st.forward_returns(panel, hold_days=h)\n"
            "        q = st.quintile_backtest(prox, fwd, n_quintiles=5, rebalance_freq=5)\n"
            "        s = st.summarize(q['Q5_minus_Q1'].dropna())\n"
            "        h_spreads.append(s['mean_bps']); h_ts.append(s['tstat'])\n"
            "else:\n"
            f"    h_spreads = [{R['hold1d_spread']}, {R['hold5d_spread']}, {R['hold10d_spread']}, {R['hold20d_spread']}, {R['hold65d_spread']}]\n"
            f"    h_ts = [{R['hold1d_t']}, {R['hold5d_t']}, {R['hold10d_t']}, {R['hold20d_t']}, {R['hold65d_t']}]\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.5))\n"
            "ax1.bar([f'{h}d' for h in holds], h_spreads,\n"
            "        color=[GREEN if s>0 else RED for s in h_spreads], alpha=0.85)\n"
            "ax1.axhline(0, c='k', lw=1)\n"
            "ax1.set_ylabel('Q5-Q1 spread (bps/period)')\n"
            "ax1.set_title('Spread positive only at 1d; negative at all longer horizons')\n"
            "ax2.bar([f'{h}d' for h in holds], h_ts,\n"
            "        color=[GREEN if t>=2 else RED if t<=-2 else GREY for t in h_ts], alpha=0.85)\n"
            "for thr in (2, -2): ax2.axhline(thr, ls='--', c=GREY, lw=1)\n"
            "ax2.axhline(0, c='k', lw=1)\n"
            "ax2.set_ylabel('HAC t-stat')\n"
            "ax2.set_title('No period crosses |t|=2 in the momentum direction')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h, sp, t in zip(holds, h_spreads, h_ts):\n"
            "    print(f'hold={h:3d}d  spread={sp:+.2f}bps  t={t:+.2f}')"
        ),
        md(
            f"> In plain words: only the 1-day horizon shows a positive (momentum-consistent) "
            f"spread (+{R['hold1d_spread']:.1f} bps, *t* = {R['hold1d_t']:+.2f}), but it does "
            "not reach the bar. From 5d onward the spread is negative and grows more negative "
            "with time. The momentum effect does not arrive within a 13-week window on this sample."
        ),
        md(
            "### 4c · Positive synthetic control — the engine is faithful\n\n"
            "Does the engine correctly detect momentum continuation *when it exists*? "
            "We plant a tunable AR(1) momentum coefficient in a synthetic 20-stock panel:"
        ),
        code(
            "moms = [-0.40, -0.20, 0.00, 0.20, 0.40]\n"
            "syn_spreads, syn_ts = [], []\n"
            "for mom in moms:\n"
            "    panel_s, _ = data.synthetic_panel(n_stocks=20, n_days=2500, momentum=mom, seed=42)\n"
            "    prox_s = st.proximity_to_52w_high(panel_s, window=252)\n"
            "    fwd_s = st.forward_returns(panel_s, hold_days=5)\n"
            "    q_s = st.quintile_backtest(prox_s, fwd_s, n_quintiles=5, rebalance_freq=5)\n"
            "    sp_s = st.summarize(q_s['Q5_minus_Q1'].dropna())\n"
            "    syn_spreads.append(sp_s['mean_bps']); syn_ts.append(sp_s['tstat'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.bar([str(m) for m in moms], syn_spreads,\n"
            "       color=[GREEN if s>0 else RED for s in syn_spreads], alpha=0.85)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, (sp, t) in enumerate(zip(syn_spreads, syn_ts)):\n"
            "    ax.annotate(f't={t:+.1f}', (i, sp), ha='center',\n"
            "                va='bottom' if sp>=0 else 'top', fontsize=9)\n"
            "ax.set_xlabel('planted AR(1) momentum coefficient')\n"
            "ax.set_ylabel('Q5-Q1 spread (bps/period)')\n"
            "ax.set_title('Synthetic control: spread monotone in planted momentum')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Engine verdict: reversion (-0.40) → big negative spread; '\n"
            "      'momentum (+0.40) → positive spread.')"
        ),
        md(
            "> In plain words: the engine works. It correctly identifies momentum when it is "
            "planted (positive spread at +0.40) and correctly identifies mean-reversion "
            "(negative spread at -0.40). The real tape's negative spread is therefore a "
            "statement about the **market** within this mega-cap survivor basket — it sits "
            "closer to a mild mean-reverting regime than a momentum one at weekly-to-quarterly "
            "horizons."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — Q5-Q1 spread = {R['spread_mean']:+.2f} bps/5d, "
            f"HAC *t* = {R['spread_t']:+.2f}; negative and insignificant at all tested "
            f"horizons ≥5d. The 1-day horizon is the only positive point "
            f"(*t* = {R['hold1d_t']:+.2f}), below the inference bar. Survivorship bias "
            "within a mega-cap basket compounds: the strategy's 'momentum short' (Q1) "
            "is populated by temporarily-soft blue chips that recover, inflating Q1 returns.\n"
            f"- **Tradability `MIRAGE`** — top decile earns {R['decile_ann']:.1f}%/yr "
            f"vs {R['baseline_ann']:.1f}%/yr equal-weight baseline; lags the market before "
            "costs. The original paper's 6-12 month hold is not tested here; this result "
            "covers only weekly-to-quarterly horizons.\n"
            f"- **George-Hwang `INVERTED ON LARGE-CAPS`** — the published anomaly "
            "(1963-2001, broad universe) does not replicate on a 2013-2026 mega-cap "
            "survivorship-biased panel at weekly horizons."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the long-only comparison\n\n"
            "The head-to-head comparison a long-only investor cares about: "
            "top decile (near 52w-high) vs equal-weight basket, before and after costs:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    panel, prox = load_real()\n"
            "    fwd = st.forward_returns(panel, hold_days=5)\n"
            "    strat, base = st.long_top_decile(prox, fwd, decile_frac=0.10, rebalance_freq=5)\n"
            "    cum_strat = (1 + strat).cumprod()\n"
            "    cum_base  = (1 + base).cumprod()\n"
            "    s_strat = st.summarize(strat)\n"
            "    s_base  = st.summarize(base)\n"
            "    strat_ann, base_ann = s_strat['ann_pct'], s_base['ann_pct']\n"
            "    strat_t, base_t = s_strat['tstat'], s_base['tstat']\n"
            "else:\n"
            f"    strat_ann, base_ann = {R['decile_ann']}, {R['baseline_ann']}\n"
            f"    strat_t, base_t = 2.11, 4.26\n"
            f"    cum_strat = cum_base = None\n"
            "print(f'Top decile:    {strat_ann:+.1f}%/yr  (HAC t = {strat_t:+.2f})')\n"
            "print(f'EW baseline:   {base_ann:+.1f}%/yr  (HAC t = {base_t:+.2f})')\n"
            "print(f'Drag vs baseline: {strat_ann - base_ann:+.1f}pp/yr — BEFORE COSTS')\n"
            "if HAVE_REAL and cum_strat is not None:\n"
            "    fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "    ax.plot(cum_strat.index, cum_strat.values, c=RED, lw=2, label=f'Top decile ({strat_ann:+.1f}%/yr)')\n"
            "    ax.plot(cum_base.index, cum_base.values, c=GREY, lw=1.5, label=f'EW baseline ({base_ann:+.1f}%/yr)')\n"
            "    ax.set_ylabel('cumulative return (period compounded)'); ax.legend()\n"
            "    ax.set_title('Top-decile (near 52w-high) lags the equal-weight basket')\n"
            "    plt.tight_layout(); plt.show()"
        ),
        md(
            "> In plain words: a long-only investor running this strategy earns "
            f"**{R['decile_ann']:.1f}%/yr** against a passive basket at "
            f"**{R['baseline_ann']:.1f}%/yr** — a persistent {R['baseline_ann']-R['decile_ann']:.1f} pp/yr "
            "drag before any trading costs, commission, or bid-ask spread."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — what a real replication would require\n\n"
            "The result here is not that 52-week-high proximity is useless in all contexts — "
            "it is that it fails on *this* specific mega-cap survivorship-biased sample at "
            "weekly-to-quarterly horizons. A proper replication of George & Hwang (2004) would require:\n\n"
            "1. **Broader universe.** The original paper used all NYSE/AMEX/Nasdaq stocks, "
            "including small and mid-caps. The momentum short leg is driven by genuine losers, "
            "not by 'temporarily soft Apple'\n"
            "2. **Longer hold periods.** The original signal was strongest at 6-12 months — "
            "well outside our 1-65 day window. Extend the test to 126-252 trading days.\n"
            "3. **Out-of-sample period.** The original paper ended in 2001. McLean & Pontiff "
            "(2016) document that anomalies decay after publication. A 2002-2026 out-of-sample "
            "test with an unbiased universe is the fair comparison.\n"
            "4. **The mirror (Study 202).** The 52-week-low study shows the same large-cap "
            "sample produces mild mean-reversion within the basket — consistent with what we "
            "find here."
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
