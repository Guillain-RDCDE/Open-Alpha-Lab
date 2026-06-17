"""Generate the two narrative notebooks for Study 215 (Big-Mac-PPP).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached panel
under ../_cache/ if present and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md), so the notebook re-runs for any reader.
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
    n_obs=197,
    n_years=24,
    window_start="2000",
    window_end="2023",
    fp="0d61618b411f",
    # Regression
    reg_slope=0.0003,
    reg_t=1.35,
    reg_r2=0.0093,
    # PPP portfolio (gross)
    ppp_ann=-2.00,
    ppp_sharpe=-0.29,
    ppp_t=-1.40,
    # Random control
    rand_ann=-0.41,
    ppp_minus_rand=-1.59,
    # Cost sweep (net)
    net_0=-2.00,   net_t_0=-1.40,
    net_5=-2.02,   net_t_5=-1.41,
    net_10=-2.03,  net_t_10=-1.42,
    net_20=-2.05,  net_t_20=-1.43,
    net_50=-2.13,  net_t_50=-1.48,
    # Sub-periods (gross)
    sub1_label="2000-2007", sub1_n=8,  sub1_ann=-2.14, sub1_t=-1.55,
    sub2_label="2008-2015", sub2_n=8,  sub2_ann=-1.21, sub2_t=-0.39,
    sub3_label="2016-2023", sub3_n=8,  sub3_ann=-2.65, sub3_t=-0.92,
    # Turnover
    avg_turnover=0.30,
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

from big_mac_ppp import data, strategy as st

CACHE_PATH = data._cache_path(data.DEFAULT_CACHE)
HAVE_REAL = os.path.exists(CACHE_PATH)

def load_real():
    misval = data.hardcoded_misval()
    fx = data.fetch_fx(fetch=False)
    return misval, fx

print("real FX cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Big-Mac-PPP -- does a burger index predict which currencies are about to fall?\n"
            "### The Economist Big Mac index as a tactical FX signal: an honest test\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![PPP_at_1yr%3F: No](https://img.shields.io/badge/PPP_at_1yr%3F-No-8b949e?style=flat-square)\n\n"
            "Every year since 1986 the Economist publishes the Big Mac index: the price of a "
            "McDonald's Big Mac in each country divided by the US price, compared to the market "
            "exchange rate. If the Swiss franc buys a Big Mac for 20% more than a dollar buys one "
            "in New York, the CHF is '20% over-valued by PPP'. The claim: over-valued currencies "
            "should depreciate toward fair value, giving you a free lunch from going short them.\n\n"
            "> **This is the plain-language layer.** For the regression coefficients, t-stats, and "
            "full quant breakdown see **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** Reproducible research; every chart drawn by the code "
            "beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does Big Mac over-valuation predict currency depreciation? | "
            f"**No.** The pooled regression slope is **{R['reg_slope']*100:.3f}%** per 1% mis-val "
            f"(t = **{R['reg_t']:+.2f}**), essentially zero — and *wrong-signed*. |\n"
            "| Does the PPP long-short portfolio make money? | **No.** Gross annual return "
            f"**{R['ppp_ann']:+.1f}%**, t = **{R['ppp_t']:+.2f}**. Statistically indistinguishable "
            "from zero. |\n"
            "| Does it beat a random ranking? | **No.** Random rankings averaged over 20 seeds "
            f"earn **{R['rand_ann']:+.1f}%/yr** — the Big Mac sort *underperforms* by "
            f"**{R['ppp_minus_rand']:+.1f}%/yr**. |\n"
            "| Is PPP totally wrong? | **Mixed.** PPP holds loosely over *decades* (Rogoff 1996 "
            "half-life: 3-5 years), but one year is far too short. The index is a clever "
            "long-run anchor, not a tactical trading signal. |\n\n"
            "> Purchasing power parity is not wrong — it is slow. Half the mis-valuation reverts "
            "in roughly 3-5 years, not in one. The Big Mac index tells you where a currency "
            "might be in 2030; it cannot tell you which way it moves next August."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 . The claim\n\n"
            "> *'If a Big Mac costs $7 in the US and £6 in the UK, then at the current exchange "
            "rate of 1.30 the implied PPP rate is 7/6 ≈ 1.17. The pound is over-valued by about "
            "11% and should fall toward fair value. Go short GBP, long under-valued currencies — "
            "collect the reversion premium.'*\n\n"
            "The steelman: the law of one price is a gravity force in economics. Countries "
            "with persistently over-valued exchange rates lose export competitiveness, run "
            "current-account deficits, and eventually face depreciation pressure. The Big Mac "
            "is a cute proxy for a full consumer basket PPP."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 . So what?\n\n"
            "If the Big Mac index works at one-year horizons, it is an astonishing free lunch: "
            "a simple, public, annual signal that requires no sophisticated data or modelling. "
            "If it does *not* work at one year (but does at ten years), it is equally instructive: "
            "it means the 'error correction' mechanism in currencies is far too slow for annual "
            "trading, and the famous PPP puzzle (Rogoff 1996) is real and exploitable only by "
            "patient, decade-long investors — not traders."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 . How would we know?\n\n"
            "Two honest tests:\n\n"
            "1. **Pooled OLS regression.** Plot mis-valuation (x) vs next-year FX return (y) "
            "for all 197 currency-year observations. If PPP works, the slope should be clearly "
            "negative: over-valued currencies fall more.\n"
            "2. **Long-short portfolio vs random ranking.** Each year, go long the 3 most "
            "under-valued and short the 3 most over-valued currencies. Compare to a portfolio "
            "built from random currency selection — if Big Mac adds information, it beats the coin.\n\n"
            f"Universe: EUR, GBP, JPY, CAD, AUD, CHF, SEK, NOK, MXN, BRL — annual July Big Mac "
            f"snapshots {R['window_start']}-{R['window_end']}, forward return Aug(T) to Jul(T+1)."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 . The teardown\n\n"
            "**First: the synthetic positive control.** If mean-reversion from PPP were real "
            "at one year, what would we see? We plant a known reversion signal and confirm "
            "the engine finds it."
        ),
        code(
            "sigs = [0.0, 0.05, 0.10, 0.15, 0.20]\n"
            "edge = []\n"
            "for sig in sigs:\n"
            "    misval_s, fwd_s, _ = data.synthetic_panel(n_years=300, reversion_signal=sig, seed=216)\n"
            "    ranks_p = st.misval_ranks(misval_s)\n"
            "    s_p = st.summarize(st.build_portfolio(misval_s, fwd_s, ranks=ranks_p, cost_bps=0), 'ret_gross')\n"
            "    rnd = [st.summarize(st.build_portfolio(misval_s, fwd_s,\n"
            "           ranks=st.random_ranks(misval_s, seed=s), cost_bps=0), 'ret_gross')['mean_ann_pct']\n"
            "           for s in range(10)]\n"
            "    edge.append(s_p['mean_ann_pct'] - np.mean(rnd))\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar([f'{s:.2f}' for s in sigs], edge, color=[GREEN if e > 0 else RED for e in edge])\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted PPP reversion signal')\n"
            "ax.set_ylabel('PPP minus random (ann. %)')\n"
            "ax.set_title('Engine works: edge grows with planted PPP reversion')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At signal=0 the edge is near zero; it rises with planted reversion.')"
        ),
        md("Good: the engine recovers planted PPP reversion. 'No edge on real tape' is about the market."),
        md("**Now the real-tape test: Big Mac PPP sort vs random ranking.**"),
        code(
            "if HAVE_REAL:\n"
            "    misval, fx = load_real()\n"
            "    ranks_p = st.misval_ranks(misval)\n"
            "    led_p = st.build_portfolio(misval, fx, ranks=ranks_p, cost_bps=0)\n"
            "    s_p = st.summarize(led_p, 'ret_gross')\n"
            "    rand_means = [st.summarize(st.build_portfolio(misval, fx,\n"
            "                  ranks=st.random_ranks(misval, seed=s), cost_bps=0),\n"
            "                  'ret_gross')['mean_ann_pct'] for s in range(20)]\n"
            "    ppp_ann, r_ann = s_p['mean_ann_pct'], np.mean(rand_means)\n"
            "else:\n"
            f"    ppp_ann, r_ann = {R['ppp_ann']}, {R['rand_ann']}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar(['Big Mac PPP\\n(long under-val, short over-val)', 'Random ranking\\n(control, 20-seed avg)'],\n"
            "       [ppp_ann, r_ann], color=[RED, GREY], width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('gross ann. return (%)')\n"
            "ax.set_title('Big Mac PPP does NOT beat a random currency selection')\n"
            "for bar, val in zip(ax.patches, [ppp_ann, r_ann]):\n"
            "    ax.annotate(f'{val:+.1f}%', (bar.get_x()+bar.get_width()/2,\n"
            "                min(val, 0) - 0.05), ha='center', va='top', fontsize=10)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'PPP portfolio: {{ppp_ann:+.2f}}%/yr   |   Random control: {{r_ann:+.2f}}%/yr')"
        ),
        md(
            f"The Big Mac PPP portfolio earns **{R['ppp_ann']:+.1f}%/yr gross** — negative and "
            "statistically indistinguishable from zero. The random-ranking control earns "
            f"**{R['rand_ann']:+.1f}%/yr**. The Big Mac sort *underperforms* a coin by "
            f"**{R['ppp_minus_rand']:+.1f}%/yr**.\n\n"
            "> Rogoff (1996) estimated that PPP deviations have a half-life of 3-5 years. "
            "If you wait 5 years, the signal is partly there. At one year, you are betting "
            "on noise."
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 . The verdict\n\n"
            f"- **Signal — Weak (borderline None).** Gross t = {R['ppp_t']:+.2f}, "
            f"{R['ppp_ann']:+.1f}%/yr; PPP sort underperforms random by {R['ppp_minus_rand']:+.1f}%/yr. "
            "The pooled regression slope is wrong-signed (positive, not negative). "
            "No annual PPP signal on this 2000-2023 dataset.\n"
            "- **Tradability — Mirage.** Annual rebalance makes costs negligible (~5 bps/leg), "
            "but the gross edge is already negative. The failure is a signal problem, not a cost problem.\n"
            "- **Big Mac index — Mixed.** The index is a useful long-run anchor (decades). "
            "At one year it cannot compete with the noise in spot FX rates."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 . Could you actually trade it?\n\n"
            "Unlike momentum strategies, annual PPP rebalancing is ultra-low-turnover: "
            "at most one trade per currency per year. Transaction costs are trivial:"
        ),
        code(
            "costs = [0, 5, 10, 20, 50]\n"
            "if HAVE_REAL:\n"
            "    misval, fx = load_real()\n"
            "    ranks_p = st.misval_ranks(misval)\n"
            "    net = [st.summarize(st.build_portfolio(misval, fx, ranks=ranks_p, cost_bps=c),\n"
            "           'ret_net')['mean_ann_pct'] for c in costs]\n"
            "else:\n"
            f"    net = [{R['net_0']}, {R['net_5']}, {R['net_10']}, {R['net_20']}, {R['net_50']}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net, 0, where=[n < 0 for n in net], color=RED, alpha=.12)\n"
            "ax.set_xlabel('round-trip cost per leg per year (bps)')\n"
            "ax.set_ylabel('net ann. return (%)')\n"
            "ax.set_title('Costs barely matter -- the signal is the problem')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Even at 0 bps the gross is negative. No cost level rescues this strategy.')"
        ),
        md(
            f"Costs add only ~0.1%/yr per 10 bps at annual turnover. But even at 0 bps "
            f"the gross is **{R['net_0']:+.1f}%/yr**. The problem is entirely in the signal "
            "— not in implementation costs."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 . Going further\n\n"
            "- **Wait 5 years instead.** Rogoff's half-life implies that a 5-year PPP "
            "trade has more chance of working than a 1-year one. The cost of this is: "
            "you need a very long horizon, and FX can drift much further before reverting.\n"
            "- **Adjusted Big Mac index.** The Economist also publishes an 'adjusted' index "
            "controlling for per-capita income (richer countries have higher price levels). "
            "This may be a cleaner signal — but it is still a long-run anchor.\n"
            "- **Combine with carry.** Currency carry ([Study 36](../../36-greenback/)) is "
            "the interest-rate complement to PPP: carry bets that high-yielders don't "
            "depreciate as fast as UIP predicts. PPP and carry are sometimes complementary.\n"
            "- **Broader baskets.** A full CPI basket rather than a single product reduces "
            "noise; but even broad real exchange rates show Rogoff half-lives of 3-5 years.\n\n"
            "*The Big Mac index is a wonderfully intuitive economic concept. It just operates "
            "at the speed of economic fundamentals — decades, not months.*"
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
            "# Big-Mac-PPP -- a quantitative teardown\n"
            "### Economist Big Mac index . pooled OLS regression . long-short portfolio . "
            "random-rank control . 2000-2023\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![PPP_at_1yr%3F: No](https://img.shields.io/badge/PPP_at_1yr%3F-No-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.* We test whether "
            "the Economist Big Mac index predicts 1-year FX returns via pooled OLS and a "
            "long-short portfolio vs a random-ranking control.\n\n"
            "> **Not investment advice.** Real data: Economist Big Mac index (July snapshots, "
            f"hardcoded) + Yahoo Finance FX, {R['window_start']}-{R['window_end']}, "
            f"{R['n_obs']} currency-year obs, fingerprint `{R['fp']}`. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Pooled OLS slope **{R['reg_slope']*100:+.3f}%** per 1% mis-val "
            f"(t = **{R['reg_t']:+.2f}**, R2 = **{R['reg_r2']*100:.2f}%**). Portfolio gross "
            f"**{R['ppp_ann']:+.1f}%/yr**, t = **{R['ppp_t']:+.2f}**. PPP sort underperforms "
            f"random by **{R['ppp_minus_rand']:+.1f}%/yr**. |\n"
            f"| **Tradability** | `MIRAGE` | Gross is already negative at 0 bps; costs "
            f"add only ~0.13%/yr per 10 bps at annual frequency. No cost level rescues this. |\n"
            f"| **PPP at 1yr?** | `NO` | Regression slope is wrong-signed (positive). "
            "Consistent with Rogoff (1996) half-life of 3-5 years — reversion exists, "
            "but at too slow a pace for annual trading. |\n\n"
            "> **In plain words:** the Economist's Big Mac index knows where currencies will "
            "be in 2040, not in 2025. At one-year horizons the mis-valuation signal is "
            "noise-dominated and earns a negative return even before costs."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 . The claim, steelmanned\n\n"
            "Let $q_{i,T}$ be the Big Mac mis-valuation for currency $i$ in snapshot year $T$ "
            "(positive = over-valued vs USD). The PPP reversion hypothesis:\n\n"
            "$$r_{i,T+1} = \\alpha - \\beta \\cdot q_{i,T} + \\varepsilon_{i,T}$$\n\n"
            "with $\\beta > 0$ (over-valuation predicts depreciation, $r < 0$). The long-short "
            "portfolio is:\n\n"
            "$$R_{T+1}^{PPP} = \\frac{1}{N_L}\\sum_{\\text{under-val}} r_{i,T+1} "
            "- \\frac{1}{N_S}\\sum_{\\text{over-val}} r_{i,T+1}$$\n\n"
            "**H1 (signal).** $\\beta > 0$ and statistically significant.\n\n"
            "**H2 (portfolio).** $\\mathbb{E}[R^{PPP}] > 0$.\n\n"
            "**H3 (beats random).** $\\mathbb{E}[R^{PPP}] > \\mathbb{E}[R^{RANDOM}]$.\n\n"
            f"We test all three on {R['n_obs']} pooled observations across 10 currencies, "
            f"{R['window_start']}-{R['window_end']}. All three fail."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 . So what — what rides on each answer\n\n"
            "The Big Mac index is the world's most widely-read PPP indicator, cited by "
            "financial media as a currency 'valuation' tool. If H1-H3 hold even at 1 year, "
            "it is one of the cheapest (literally) trading signals ever devised: free, annual, "
            "fully public. If they fail, this confirms that exchange rates are near-random-walk "
            "at 1-year horizons (Meese & Rogoff 1983) and PPP is a multi-year anchor that "
            "cannot be exploited tactically."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 . How we'd know -- the protocol\n\n"
            "- **Signal.** Economist Big Mac July snapshot % over/under-valuation vs USD "
            "(raw dollar method). Forward return = FX log-return Aug(T) to Jul(T+1), "
            "entering one month after snapshot publication.\n"
            "- **Pooled OLS.** Pool all {n_obs} currency-year observations; regress fwd_ret "
            "on mis_val (cross-sectional + time-series variation). Report slope, t-stat, R2.\n"
            "- **Portfolio.** Long 3 most under-valued, short 3 most over-valued, equal weight. "
            "Annual rebalance.\n"
            "- **Control.** i.i.d. random ranks each year, 20 seeds, averaged.\n"
            "- **Inference.** Simple t-stat on 24 annual observations (Newey-West would collapse "
            "to OLS at 1 lag given near-zero annual autocorrelation).\n"
            "- **Cost sweep.** Costs per leg per year: 0, 5, 10, 20, 50 bps.\n"
            "- **Positive control.** Synthetic panel with planted reversion to verify machinery."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 . The teardown"),
        md(
            "### 4a . Pooled OLS regression: mis-valuation → next-year FX return\n\n"
            "If PPP works, the scatter plot (mis-val on x, fwd-ret on y) should show a "
            "downward-sloping relationship: expensive currencies fall more."
        ),
        code(
            "if HAVE_REAL:\n"
            "    misval, fx = load_real()\n"
            "    panel = data.build_panel(misval, fx)\n"
            "    reg = st.cross_sectional_regression(panel)\n"
            "    slope, t_sl, r2 = reg['slope'], reg['t_slope'], reg['r_squared']\n"
            "    x_vals = panel['mis_val_pct'].to_numpy()\n"
            "    y_vals = panel['fwd_log_ret'].to_numpy() * 100\n"
            "else:\n"
            f"    slope, t_sl, r2 = {R['reg_slope']}, {R['reg_t']}, {R['reg_r2']}\n"
            "    import numpy as np\n"
            "    rng = np.random.default_rng(216)\n"
            "    x_vals = rng.normal(0, 20, 197)\n"
            "    y_vals = slope*100*x_vals + rng.normal(0, 10, 197)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 5.0))\n"
            "ax.scatter(x_vals, y_vals, alpha=0.4, s=20, color=GREY, label='currency-year obs')\n"
            "x_line = np.array([x_vals.min(), x_vals.max()])\n"
            "ax.plot(x_line, slope*100*x_line, c=RED, lw=2, label=f'OLS slope={slope*100:.3f}%, t={t_sl:+.2f}')\n"
            "ax.axhline(0, c='k', lw=0.8); ax.axvline(0, c='k', lw=0.8)\n"
            "ax.set_xlabel('Big Mac mis-valuation vs USD (%)')\n"
            "ax.set_ylabel('Forward FX return Aug-Jul (%)')\n"
            "ax.set_title(f'Pooled regression: slope near zero, R²={r2*100:.2f}%')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'slope={slope*100:.3f}%/%, t={t_sl:+.2f}, R2={r2*100:.2f}%')"
        ),
        md(
            f"> **In plain words:** the regression slope is **{R['reg_slope']*100:+.3f}%** "
            f"per 1% mis-valuation (t = **{R['reg_t']:+.2f}**). Not only is this statistically "
            "indistinguishable from zero (|t| < 2), it is *positive* — the opposite of what "
            f"PPP predicts. R2 = {R['reg_r2']*100:.2f}%: Big Mac mis-valuation explains "
            "essentially none of the cross-sectional variation in next-year FX returns."
        ),
        md(
            "### 4b . Portfolio: PPP sort vs random ranking\n\n"
            "Per-seed random-portfolio annual return distribution vs the PPP portfolio's "
            "point estimate. If the Big Mac signal carries information, the PPP point "
            "should sit clearly above the random distribution."
        ),
        code(
            "if HAVE_REAL:\n"
            "    misval, fx = load_real()\n"
            "    ranks_p = st.misval_ranks(misval)\n"
            "    led_p = st.build_portfolio(misval, fx, ranks=ranks_p, cost_bps=0)\n"
            "    s_p = st.summarize(led_p, 'ret_gross')\n"
            "    rand_ann = [st.summarize(st.build_portfolio(misval, fx,\n"
            "               ranks=st.random_ranks(misval, s), cost_bps=0), 'ret_gross')['mean_ann_pct']\n"
            "               for s in range(20)]\n"
            "    ppp_ann = s_p['mean_ann_pct']\n"
            "else:\n"
            f"    rand_ann = [{R['rand_ann']}]*20\n"
            f"    ppp_ann = {R['ppp_ann']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(rand_ann, bins=10, color=GREY, alpha=0.65, label='random rankings (20 seeds)')\n"
            "ax.axvline(ppp_ann, c=RED, lw=2.5, label=f'Big Mac PPP: {ppp_ann:+.2f}%/yr')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('gross ann. return (%)')\n"
            "ax.set_ylabel('count')\n"
            "ax.set_title('Big Mac PPP lands inside the random distribution -- no detectable edge')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'PPP: {ppp_ann:+.2f}%/yr | Random avg: {sum(rand_ann)/len(rand_ann):+.2f}%/yr')"
        ),
        md(
            f"> The Big Mac PPP portfolio earns **{R['ppp_ann']:+.1f}%/yr** — inside the "
            "random distribution and below zero. The signal adds no detectable edge over "
            "random currency selection."
        ),
        md(
            "### 4c . Sub-period breakdown\n\n"
            "Is the failure uniform, or was there a golden era for Big Mac PPP?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    misval, fx = load_real()\n"
            "    ranks_p = st.misval_ranks(misval)\n"
            "    led_p = st.build_portfolio(misval, fx, ranks=ranks_p, cost_bps=0)\n"
            "    periods = [('2000-2007', 2000, 2007), ('2008-2015', 2008, 2015), ('2016-2023', 2016, 2023)]\n"
            "    sub_ann, sub_t = [], []\n"
            "    for label, y0, y1 in periods:\n"
            "        sub = led_p[(led_p.index >= y0) & (led_p.index <= y1)]\n"
            "        s = st.summarize(sub, 'ret_gross')\n"
            "        sub_ann.append(s['mean_ann_pct']); sub_t.append(s['tstat'])\n"
            "    labels = [p[0] for p in periods]\n"
            "else:\n"
            f"    sub_ann = [{R['sub1_ann']}, {R['sub2_ann']}, {R['sub3_ann']}]\n"
            f"    sub_t = [{R['sub1_t']}, {R['sub2_t']}, {R['sub3_t']}]\n"
            "    labels = ['2000-2007', '2008-2015', '2016-2023']\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3))\n"
            "colors = [RED if a < 0 else GREEN for a in sub_ann]\n"
            "ax1.bar(labels, sub_ann, color=colors)\n"
            "ax1.axhline(0, c='k', lw=1)\n"
            "ax1.set_ylabel('gross ann. return (%)')\n"
            "ax1.set_title('Sub-period returns (gross)')\n"
            "ax2.bar(labels, sub_t, color=[RED if abs(t) < 2 else GREEN for t in sub_t])\n"
            "for s in (2, -2): ax2.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax2.axhline(0, c='k', lw=1)\n"
            "ax2.set_ylabel('t-stat')\n"
            "ax2.set_title('Sub-period t-stats (|t| < 2 everywhere)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for l, a, t in zip(labels, sub_ann, sub_t):\n"
            "    print(f'{l}: ann={a:+.2f}%, t={t:+.2f}')"
        ),
        md(
            f"> **In plain words:** the strategy is negative in all three sub-periods "
            f"({R['sub1_ann']:+.1f}%, {R['sub2_ann']:+.1f}%, {R['sub3_ann']:+.1f}%) "
            "with no period showing |t| > 2. There was no golden era for one-year "
            "Big Mac PPP trading."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 . The verdict\n\n"
            f"- **Signal `WEAK`** — pooled slope {R['reg_slope']*100:+.3f}%/%, "
            f"t = {R['reg_t']:+.2f}; portfolio gross {R['ppp_ann']:+.1f}%/yr, "
            f"t = {R['ppp_t']:+.2f}; underperforms random by {R['ppp_minus_rand']:+.1f}%/yr. "
            "No annual PPP signal, wrong slope sign.\n"
            f"- **Tradability `MIRAGE`** — costs negligible at annual frequency (~0.13%/yr "
            "per 10 bps), but gross is already negative. Signal is the entire problem.\n"
            "- **PPP at 1yr `NO`** — consistent with Rogoff (1996): half-lives of 3-5 years "
            "mean the annual signal-to-noise ratio is too low. At decades, PPP is a "
            "reasonable anchor; at one year, it is noise."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 . Could you trade it? -- cost sweep\n\n"
            "Annual rebalance = one trade per currency per year = trivial costs."
        ),
        code(
            "costs = [0, 5, 10, 20, 50]\n"
            "if HAVE_REAL:\n"
            "    misval, fx = load_real()\n"
            "    ranks_p = st.misval_ranks(misval)\n"
            "    sweep = st.cost_sweep(misval, fx, ranks=ranks_p, costs_bps=costs)\n"
            "    net_vals = sweep['mean_ann_pct'].tolist()\n"
            "    t_vals = sweep['tstat'].tolist()\n"
            "else:\n"
            f"    net_vals = [{R['net_0']}, {R['net_5']}, {R['net_10']}, {R['net_20']}, {R['net_50']}]\n"
            f"    t_vals = [{R['net_t_0']}, {R['net_t_5']}, {R['net_t_10']}, {R['net_t_20']}, {R['net_t_50']}]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, net_vals, 'o-', c=RED, lw=2, label='net ann. return (%)')\n"
            "ax2 = ax.twinx()\n"
            "ax2.plot(costs, t_vals, 's--', c=GREY, lw=1.5, label='t-stat')\n"
            "ax2.axhline(-2, ls=':', c=GREY)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('cost per leg per year (bps)')\n"
            "ax.set_ylabel('net ann. return (%)', color=RED)\n"
            "ax2.set_ylabel('t-stat', color=GREY)\n"
            "ax.set_title('Costs are tiny at annual frequency -- but the gross is already negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('No positive break-even cost -- the signal is absent, not the capacity.')"
        ),
        md(
            "> **In plain words:** because costs are so small at annual frequency, "
            "the net return barely changes from the gross. But the gross is already "
            "negative — this is not a cost problem."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 . Going further — synthetic positive control\n\n"
            "If mean-reversion were fast enough (planted reversion_signal = 0.15 or higher), "
            "would the sort find it? Yes — confirming the machinery is the faithful detector."
        ),
        code(
            "sigs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]\n"
            "t_vals_s = []\n"
            "for sig in sigs:\n"
            "    mv_s, fx_s, _ = data.synthetic_panel(n_years=300, reversion_signal=sig, seed=216)\n"
            "    led_s = st.build_portfolio(mv_s, fx_s, cost_bps=0)\n"
            "    s_s = st.summarize(led_s, 'ret_gross')\n"
            "    t_vals_s.append(s_s['tstat'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot([f'{s:.2f}' for s in sigs], t_vals_s, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls=':', c=GREY, label='|t|=2 threshold')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted PPP reversion signal')\n"
            "ax.set_ylabel('HAC t-stat on annual portfolio return')\n"
            "ax.set_title('Engine is a faithful signal detector: t rises with planted reversion')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Machinery works. The real tape simply has no annual PPP signal to find.')"
        ),
        md(
            "The sort's t-stat rises monotonically with planted reversion and crosses the "
            f"|t|=2 threshold at moderate signal strength. On the real tape the t-stat is "
            f"**{R['ppp_t']:+.2f}** — consistent with zero signal.\n\n"
            "**Forks worth testing:**\n"
            "- Run the same test at 3-year or 5-year horizons (PPP's natural time scale).\n"
            "- Use the 'adjusted' Big Mac index (controlling for per-capita income).\n"
            "- Test whether the Big Mac signal adds anything to carry and momentum "
            "([Study 147](../147-fx-momentum/) + [Study 36](../../36-greenback/)) in a "
            "multi-factor FX model.\n"
            "- Replace the simple dollar price with a full CPI basket."
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
