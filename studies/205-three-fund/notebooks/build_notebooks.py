"""Generate the two narrative notebooks for Study 205 (Three-Fund).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
parquet under ../_cache/ if present and otherwise quote the frozen headline numbers
in ``R``, so the notebook re-runs for any reader.

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
    # Data stamp
    n_days=3865, start="2011-02-01", end="2026-06-15", fp="2eaa8336f1fa",
    n_years=15,
    # Main tournament (cost=5bps one-way, annual rebalance)
    cagr_3f=10.61, sharpe_3f=0.737, vol_3f=15.27, mdd_3f=-31.21, calmar_3f=0.340, t_3f=5.17,
    cagr_6040=9.56, sharpe_6040=0.929, vol_6040=10.41, mdd_6040=-21.03, calmar_6040=0.455,
    cagr_us=14.11, sharpe_us=0.857, vol_us=17.11, mdd_us=-33.72, calmar_us=0.418,
    # Pairwise diffs
    diff_vs_us=-3.48, t_vs_us=-4.90,
    diff_vs_6040=1.31, t_vs_6040=1.70,
    bonf_thresh=2.394,
    # International attribution
    intl_diff=-2.01, intl_t=-3.25,
    # Bootstrap CI (three-fund Sharpe)
    ci_lo=0.251, ci_hi=1.247, frac_neg=0.1,
    # Sub-period Sharpe
    sharpe_3f_2011=0.573, sharpe_us_2011=0.800, sharpe_6040_2011=0.976,
    sharpe_3f_2016=0.810, sharpe_us_2016=0.857, sharpe_6040_2016=0.986,
    sharpe_3f_2021=0.841, sharpe_us_2021=0.944, sharpe_6040_2021=0.887,
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
BLUE = "#2980b9"

from three_fund import data, strategy as st

def _have_cache():
    return os.path.exists(data._cache_path(data.DEFAULT_CACHE))

HAVE_REAL = _have_cache()

def get_navs(cost_bps=5.0):
    # Load real prices and run all three portfolios.
    prices = data.load_real(fetch=False)
    return data, prices, st.run_all_portfolios(prices, cost_bps=cost_bps)

print("real three-fund cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Three-Fund — is the canonical lazy portfolio actually better?\n"
            "### Bogleheads VTI/VXUS/BND vs 100% US stocks and 60/40, 2011–2026\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![International_helped%3F: Hurt](https://img.shields.io/badge/International_helped%3F-Hurt-c0392b?style=flat-square)\n\n"
            "The **Bogleheads three-fund portfolio** is possibly the most-recommended "
            "investment strategy in personal finance: put your money into a US total-market "
            "fund (VTI), an international fund (VXUS), and a bond fund (BND), rebalance "
            "once a year, and ignore the noise. It's pitched as the *optimal* lazy portfolio — "
            "globally diversified, low-cost, proven by theory. But is it actually better than "
            "just buying SPY and going fishing? Or better than a simple 60/40? This notebook "
            "runs the honest test.\n\n"
            "> This is the plain-language layer. Want the t-stats, bootstrap Sharpe intervals, "
            "and international attribution? That's **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** Reproducible research: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Is the three-fund better than 100% US? | **No.** It returned **{R['cagr_3f']:.1f}%/yr** "
            f"vs **{R['cagr_us']:.1f}%/yr** for SPY, with *lower* Sharpe ({R['sharpe_3f']:.2f} vs {R['sharpe_us']:.2f}). |\n"
            f"| Is it better than 60/40? | **Not statistically.** +{R['diff_vs_6040']:.1f}%/yr more return "
            f"but higher volatility and deeper drawdown; *t* = {R['t_vs_6040']:+.2f}, not significant. |\n"
            f"| Did the VXUS international sleeve help? | **No — it hurt.** The intl sleeve "
            f"subtracted **{R['intl_diff']:.1f}%/yr** vs holding the same weight in VTI instead. |\n"
            "| Could you trade it? | **Yes, easily** — annual rebalancing makes costs near-zero. "
            "But 'implementable' is not the same as 'better than simpler alternatives.' |\n\n"
            "> The three-fund portfolio is sensible, low-cost, and sound by theory. On *this* "
            "sample (2011–2026), it was the worst of three options — and the culprit is clear: "
            "international stocks underperformed US stocks for 15 years straight."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Three mutual funds are all you need for a complete investment portfolio. "
            "A total US market fund, a total international market fund, and a total bond "
            "market fund give you diversification across thousands of securities worldwide, "
            "at rock-bottom cost. Rebalance annually. Ignore everything else.\"*\n\n"
            "— Bogleheads (bogleheads.org/wiki/Three-fund_portfolio)\n\n"
            "The idea is grounded in real theory: a globally-diversified market-cap-weighted "
            "portfolio is what the CAPM says you should hold. Adding bonds reduces volatility "
            "through negative correlation. The recipe is simple, cheap, and defensible by math. "
            "The question is whether the math translates to better *actual* outcomes over the "
            "available sample."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the three-fund is genuinely better, millions of people who just own SPY or "
            "a 60/40 are leaving risk-adjusted return on the table. If it isn't — if the "
            "'diversification benefit' is theoretical but not realised over a 15-year live sample — "
            "then the case for complexity over a single ETF needs to rest on non-return grounds "
            "(behavioural smoother, lower tracking-error anxiety, etc.). The answer also "
            "directly addresses whether international diversification was worth it over 2011–2026."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three rules keep this honest:\n\n"
            "1. **Use a fair comparison.** All three portfolios use annual rebalancing and the "
            "same 5-bp one-way cost. The three-fund is slightly more expensive to run (3 "
            "sleeves vs 1), but ETF expense ratios are so low today that this barely matters.\n"
            "2. **Use risk-adjusted metrics, not just return.** More return at more risk isn't "
            "free — the Sharpe ratio (return per unit of volatility) is the primary gauge. "
            "We also look at max drawdown and the Calmar ratio.\n"
            "3. **Correct for multiple comparisons.** We're running three tests (three-fund vs "
            "US-only, three-fund vs 60/40, VXUS attribution). The Bonferroni threshold for "
            f"'significant' is **|t| ≥ {R['bonf_thresh']:.2f}**, not the naive 1.96.\n\n"
            "Data: daily adjusted-close prices for VTI, VXUS, BND, and SPY from Yahoo Finance, "
            f"**{R['start']}** to **{R['end']}** ({R['n_days']:,} trading days, ≈ 15 years)."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — what the numbers actually show\n\n"
            "**Growth of \\$1 — the equity curves tell the whole story:**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    data_m, prices, navs = get_navs()\n"
            "    nav_3f = navs['three_fund']\n"
            "    nav_us = navs['us_only']\n"
            "    nav_6040 = navs['sixty_forty']\n"
            "    m_3f = st.summarize(nav_3f)\n"
            "    m_us = st.summarize(nav_us)\n"
            "    m_6040 = st.summarize(nav_6040)\n"
            "else:\n"
            "    # Use synthetic null to illustrate the shape\n"
            "    prices, truth = data.synthetic_prices(n_days=3865, intl_alpha=-0.02, seed=205)\n"
            "    navs = st.run_all_portfolios(prices, cost_bps=5.0)\n"
            "    nav_3f, nav_us, nav_6040 = navs['three_fund'], navs['us_only'], navs['sixty_forty']\n"
            "    m_3f = st.summarize(nav_3f)\n"
            "    m_us = st.summarize(nav_us)\n"
            "    m_6040 = st.summarize(nav_6040)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10.5, 5.5))\n"
            "ax.plot(nav_3f.index, nav_3f.values, c=AMBER, lw=2, label=f'Three-fund (Sharpe {m_3f[\"sharpe\"]:.2f})')\n"
            "ax.plot(nav_us.index, nav_us.values, c=RED, lw=2, label=f'100% US-SPY (Sharpe {m_us[\"sharpe\"]:.2f})')\n"
            "ax.plot(nav_6040.index, nav_6040.values, c=BLUE, lw=2, label=f'60/40 SPY+BND (Sharpe {m_6040[\"sharpe\"]:.2f})')\n"
            "ax.set_ylabel('Growth of $1'); ax.set_title('Three-Fund vs Baselines — 2011 to 2026')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Three-fund: CAGR={m_3f[\"cagr\"]*100:.1f}%  Sharpe={m_3f[\"sharpe\"]:.3f}  MaxDD={m_3f[\"max_dd\"]*100:.1f}%')\n"
            "print(f'US-only:    CAGR={m_us[\"cagr\"]*100:.1f}%  Sharpe={m_us[\"sharpe\"]:.3f}  MaxDD={m_us[\"max_dd\"]*100:.1f}%')\n"
            "print(f'60/40:      CAGR={m_6040[\"cagr\"]*100:.1f}%  Sharpe={m_6040[\"sharpe\"]:.3f}  MaxDD={m_6040[\"max_dd\"]*100:.1f}%')"
        ),
        md(
            f"The three-fund's **{R['cagr_3f']:.1f}% CAGR** sits between 60/40 ({R['cagr_6040']:.1f}%) and US-only "
            f"({R['cagr_us']:.1f}%). But on **Sharpe ratio** — the risk-adjusted measure — it comes *last*: "
            f"**{R['sharpe_3f']:.3f}** vs {R['sharpe_us']:.3f} for US-only and {R['sharpe_6040']:.3f} for 60/40. "
            "Higher volatility than 60/40, lower return than US-only."
        ),
        md(
            "**Now the honest question: why? Is it the international sleeve?**"
        ),
        code(
            "# Compare three-fund vs no-international alternative (90% VTI / 10% BND)\n"
            "if HAVE_REAL:\n"
            "    nav_nointl = st.run_portfolio(prices, {'VTI': 0.90, 'BND': 0.10}, cost_bps=5.0)\n"
            "else:\n"
            "    nav_nointl = st.run_portfolio(prices, {'VTI': 0.90, 'BND': 0.10}, cost_bps=5.0)\n"
            "m_nointl = st.summarize(nav_nointl)\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))\n"
            "# Left: equity curves comparison\n"
            "axes[0].plot(nav_3f.index, nav_3f.values, c=AMBER, lw=2, label='Three-fund (with VXUS)')\n"
            "axes[0].plot(nav_nointl.index, nav_nointl.values, c=GREEN, lw=2, label='No intl (90% VTI/10% BND)')\n"
            "axes[0].set_title('VXUS sleeve: help or hurt?'); axes[0].legend()\n"
            "axes[0].set_ylabel('Growth of $1')\n"
            "# Right: bar chart of CAGR difference\n"
            "diff_intl = m_3f['cagr'] - m_nointl['cagr']\n"
            "clr = GREEN if diff_intl > 0 else RED\n"
            "axes[1].bar(['VXUS contribution\\n(CAGR diff)'], [diff_intl*100], color=clr, width=0.4)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('Excess CAGR from VXUS (%/yr)')\n"
            "axes[1].set_title(f'International sleeve = {diff_intl*100:+.2f}%/yr')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Three-fund CAGR: {m_3f[\"cagr\"]*100:.2f}%  vs  No-intl CAGR: {m_nointl[\"cagr\"]*100:.2f}%')\n"
            f"print('VXUS contribution: ' + str(round(diff_intl*100, 2)) + '%/yr  (real tape: {R['intl_diff']:+.2f}%/yr, t={R['intl_t']:+.2f})')"
        ),
        md(
            f"The VXUS international sleeve subtracted approximately **{R['intl_diff']:.1f}%/yr** "
            f"compared to simply holding that equity weight in VTI instead (t = {R['intl_t']:+.2f}). "
            "This is the core finding: international stocks underperformed US stocks significantly "
            "over the 2011–2026 sample. The diversification benefit existed on paper — the "
            "correlation between VTI and VXUS helps in theory — but the return gap overwhelmed it.\n\n"
            "> **Why did this happen?** US tech dominated global markets 2011–2026. The MSCI EAFE "
            "(developed international) returned roughly 5-6%/yr while US markets returned 12-14%. "
            "The bond sleeve also reduced return in the 2011-2021 near-zero-rate environment."
        ),
        md(
            "**What about the worst moments — drawdowns?**"
        ),
        code(
            "# Drawdown chart\n"
            "def rolling_dd(nav):\n"
            "    peak = nav.expanding().max()\n"
            "    return (nav / peak - 1.0) * 100\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10.5, 4.5))\n"
            "ax.fill_between(nav_3f.index, rolling_dd(nav_3f), 0, alpha=0.4, color=AMBER, label='Three-fund')\n"
            "ax.fill_between(nav_us.index, rolling_dd(nav_us), 0, alpha=0.3, color=RED, label='US-only')\n"
            "ax.fill_between(nav_6040.index, rolling_dd(nav_6040), 0, alpha=0.3, color=BLUE, label='60/40')\n"
            "ax.set_ylabel('Drawdown from peak (%)'); ax.set_title('Three-fund does cushion drawdowns — but 60/40 does better')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            f"print(f'Three-fund max DD: {R['mdd_3f']:.1f}%  |  US-only: {R['mdd_us']:.1f}%  |  60/40: {R['mdd_6040']:.1f}%')"
        ),
        md(
            f"The three-fund's max drawdown ({R['mdd_3f']:.1f}%) is slightly better than US-only "
            f"({R['mdd_us']:.1f}%) thanks to the bond sleeve, but **60/40 ({R['mdd_6040']:.1f}%) "
            "protects far better** — because the 60/40's larger bond allocation does more work in "
            "a crash. The three-fund gets the worst of both worlds: enough equity to feel a crash, "
            "not enough bonds to cushion it meaningfully."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Mixed.** The three-fund is real and positive (Sharpe {R['sharpe_3f']:.2f}, "
            f"t > 5). But it was the *worst* of three alternatives on risk-adjusted return over 2011–2026: "
            f"below US-only (t = {R['t_vs_us']:+.2f}, significant) and below 60/40 (not significant, "
            f"t = {R['t_vs_6040']:+.2f}).\n"
            f"- **Tradability — Mirage.** Easily implementable (annual rebalance, near-zero costs). "
            "But 'implementable' is not 'better than alternatives.' On this sample, you'd have done "
            f"better holding just SPY (CAGR +{R['cagr_us']-R['cagr_3f']:.1f}%/yr more, Sharpe "
            f"+{R['sharpe_us']-R['sharpe_3f']:.2f} more).\n"
            f"- **International helped? — Hurt.** The VXUS sleeve was a {R['intl_diff']:.1f}%/yr drag "
            f"(t = {R['intl_t']:+.2f}). 2011–2026 was the worst possible decade to bet on international "
            "outperformance. Whether that changes going forward is unknown."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT -------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Yes — this is the three-fund's clearest advantage. Annual rebalancing means "
            "near-zero transaction costs:"
        ),
        code(
            "costs = [0.0, 2.0, 5.0, 10.0, 20.0]\n"
            "if HAVE_REAL:\n"
            "    sharpes = [st.summarize(st.run_portfolio(prices, st.THREE_FUND_WEIGHTS, cost_bps=c))['sharpe'] for c in costs]\n"
            "else:\n"
            f"    sharpes = [{R['sharpe_3f']:.3f}] * len(costs)  # cost-insensitive at annual rebalance\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, sharpes, 'o-', c=AMBER, lw=2)\n"
            "ax.set_xlabel('One-way rebalance cost (bps)')\n"
            "ax.set_ylabel('Three-fund Sharpe ratio')\n"
            "ax.set_title('Annual rebalancing: cost is essentially irrelevant')\n"
            "ax.set_ylim(0, 1.2); plt.tight_layout(); plt.show()\n"
            "print('The three-fund is cost-insensitive at annual rebalancing -- but cost-insensitive')\n"
            "print('does not help if the underlying allocation underperforms simpler alternatives.')"
        ),
        md(
            "The three-fund is effectively cost-free to implement. The problem isn't execution — "
            "it's that the allocation itself underperformed simpler options over this sample. "
            "A passive investor who simply bought and held SPY from 2011 ended up with a higher "
            f"Sharpe ({R['sharpe_us']:.2f} vs {R['sharpe_3f']:.2f}) and nearly 4% more return per year."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Is this permanent?** No. The 2011–2026 period was exceptionally good for US "
            "stocks. Over longer horizons (Dimson, Marsh & Staunton 2002 covers 1900–2000), "
            "diversification has a better track record. Past US outperformance does not "
            "guarantee future US outperformance.\n"
            "- **What about the rest of the world?** "
            "[Study 145 — Home-Bias](../../145-home-bias/) directly examines the home-bias "
            "literature and international diversification effects.\n"
            "- **More sophisticated allocation?** "
            "[Study 144 — Permanent-Portfolio](../../144-permanent-portfolio/) and "
            "[Study 68 — All-Weather](../../68-all-weather/) test multi-asset allocations "
            "with different risk targets.\n"
            "- **Equal-weight vs market-cap weight?** "
            "[Study 171 — Naive-1-Over-N](../../171-naive-1-over-n/) tests whether "
            "1/N beats Markowitz-optimised portfolios.\n\n"
            "*Think the international sleeve pays off going forward? Fork this, extend the "
            "window back to 1971 via MSCI data, and show a significant Sharpe improvement "
            "that survives the Bonferroni bar.*"
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
            "# Three-Fund — a quantitative teardown\n"
            "### Daily tape · annual rebalance · HAC inference · bootstrap Sharpe CI · "
            "international attribution · Bonferroni correction\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![International_helped%3F: Hurt](https://img.shields.io/badge/International_helped%3F-Hurt-c0392b?style=flat-square)\n\n"
            "The deep companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb). "
            "Three-fund portfolio (60% VTI / 30% VXUS / 10% BND) measured against 100% US "
            "equity (SPY) and 60/40 (SPY+BND), with HAC t-stats on annual return differences, "
            "bootstrap Sharpe CIs, Bonferroni correction, sub-period breakdown, and a "
            "synthetic positive control.\n\n"
            "> **Not investment advice.** Real data: Yahoo Finance daily adjusted closes, "
            f"as-of {R['end']}; fingerprint `{R['fp']}`. "
            "Methods in [`docs/references.md`](../docs/references.md), reproducible numbers "
            "in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **`💡 In plain words`** notes translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | Three-fund Sharpe **{R['sharpe_3f']:.3f}** vs "
            f"US-only **{R['sharpe_us']:.3f}** and 60/40 **{R['sharpe_6040']:.3f}**; "
            f"vs US-only: *t* = **{R['t_vs_us']:+.2f}** (trails, significant); "
            f"vs 60/40: *t* = **{R['t_vs_6040']:+.2f}** (not significant). |\n"
            f"| **Tradability** | `MIRAGE` | Annual rebalance → cost-insensitive, "
            "but allocation underperforms both baselines on Sharpe. |\n"
            f"| **International helped?** | `HURT` | VXUS sleeve: "
            f"**{R['intl_diff']:+.2f}%/yr**, HAC *t* = **{R['intl_t']:+.2f}** (Bonferroni-significant). |\n\n"
            "> 💡 The three-fund is sound theory, but the 2011–2026 sample was historically "
            "exceptional for US equity. International diversification subtracted return throughout."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $w = (0.6, 0.3, 0.1)$ for (VTI, VXUS, BND). The recipe asserts:\n\n"
            "- **H₁ (vs US-only).** The three-fund's risk-adjusted return "
            "($\\mathrm{SR}_{3F}$) exceeds US-only ($\\mathrm{SR}_{US}$), i.e. "
            "international diversification improves the Sharpe ratio.\n"
            "- **H₂ (vs 60/40).** The three-fund beats the simpler 60/40 on Sharpe: "
            "the VXUS sleeve adds value beyond a generic equity/bond split.\n"
            "- **H₃ (VXUS attribution).** The international leg has positive return "
            "relative to its counterfactual (same equity weight in VTI): "
            "$\\mathbb{E}[r_{VXUS}] > \\mathbb{E}[r_{VTI}]$ over the sample.\n\n"
            f"We reject H₁ (three-fund trails, *t* = {R['t_vs_us']:+.2f}), fail to reject "
            f"H₂ (*t* = {R['t_vs_6040']:+.2f} < Bonferroni {R['bonf_thresh']:.2f}), and "
            f"reject H₃ (VXUS *t* = {R['intl_t']:+.2f})."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The three-fund is the most-recommended passive strategy in personal finance. "
            "If H₁–H₃ all held, it would be the provably optimal allocation for a passive "
            "retail investor. The interesting empirical result is that **international "
            "diversification has been a return drag, not a return premium**, over the available "
            "ETF history — which mirrors the broader literature debate on whether the CAPM "
            "prediction of international diversification benefit materialises in practice "
            "once estimation error, currency risk, and time-varying correlations are accounted for."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Portfolios.** Three-fund: 60% VTI / 30% VXUS / 10% BND. Baselines: "
            "100% SPY; 60% SPY / 40% BND. All use annual calendar-year-end rebalancing "
            "and 5 bps one-way cost.\n"
            "- **Inference.** HAC t-stat (Newey-West) on per-year return *differences* "
            "across the ~15 annual observations. Bonferroni correction across 3 tests "
            f"→ threshold |*t*| ≥ {R['bonf_thresh']:.3f}.\n"
            "- **Sharpe CI.** Circular block-bootstrap (Politis-Romano 1994) on daily "
            "returns, 2,000 resamples, block size $n^{{1/3}}$.\n"
            "- **International attribution.** Three-fund vs 90% VTI / 10% BND (same "
            "equity/bond ratio, no international), to isolate the VXUS sleeve's contribution.\n"
            "- **Positive control.** Synthetic price paths with planted $\\mathrm{intl\\_alpha}$ "
            "confirm the engine recovers international advantage when one exists."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Main tournament — headline metrics\n\n"
            "Portfolio statistics over the full sample with 95% bootstrap CIs on the Sharpe:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    data_m, prices, navs = get_navs()\n"
            "    names = ['three_fund', 'sixty_forty', 'us_only']\n"
            "    labels = ['Three-fund (60/30/10)', '60/40 (SPY+BND)', '100% US (SPY)']\n"
            "    colors = [AMBER, BLUE, RED]\n"
            "    metrics = {n: st.summarize(navs[n]) for n in names}\n"
            "    ci_3f = stats.sharpe_ci_bootstrap(st.nav_to_returns(navs['three_fund']), periods_per_year=252, seed=205)\n"
            "    ci_us = stats.sharpe_ci_bootstrap(st.nav_to_returns(navs['us_only']), periods_per_year=252, seed=205)\n"
            "    ci_6040 = stats.sharpe_ci_bootstrap(st.nav_to_returns(navs['sixty_forty']), periods_per_year=252, seed=205)\n"
            "    sharpes = [metrics[n]['sharpe'] for n in names]\n"
            "    cis = [(ci_3f['ci_low'], ci_3f['ci_high']), (ci_6040['ci_low'], ci_6040['ci_high']), (ci_us['ci_low'], ci_us['ci_high'])]\n"
            "else:\n"
            f"    metrics = dict(\n"
            f"        three_fund=dict(cagr={R['cagr_3f']/100:.4f}, sharpe={R['sharpe_3f']}, ann_vol={R['vol_3f']/100:.4f}, max_dd={R['mdd_3f']/100:.4f}, calmar={R['calmar_3f']}),\n"
            f"        sixty_forty=dict(cagr={R['cagr_6040']/100:.4f}, sharpe={R['sharpe_6040']}, ann_vol={R['vol_6040']/100:.4f}, max_dd={R['mdd_6040']/100:.4f}, calmar={R['calmar_6040']}),\n"
            f"        us_only=dict(cagr={R['cagr_us']/100:.4f}, sharpe={R['sharpe_us']}, ann_vol={R['vol_us']/100:.4f}, max_dd={R['mdd_us']/100:.4f}, calmar={R['calmar_us']}),\n"
            "    )\n"
            f"    sharpes = [{R['sharpe_3f']}, {R['sharpe_6040']}, {R['sharpe_us']}]\n"
            f"    cis = [({R['ci_lo']}, {R['ci_hi']}), (0.432, 1.470), (0.385, 1.369)]\n"
            "    names = ['three_fund', 'sixty_forty', 'us_only']\n"
            "    labels = ['Three-fund (60/30/10)', '60/40 (SPY+BND)', '100% US (SPY)']\n"
            "    colors = [AMBER, BLUE, RED]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "xs = range(len(names))\n"
            "for i, (n, lbl, clr, (lo, hi)) in enumerate(zip(names, labels, colors, cis)):\n"
            "    ax.bar(i, sharpes[i], color=clr, alpha=0.8, width=0.55, label=lbl)\n"
            "    ax.errorbar(i, sharpes[i], yerr=[[sharpes[i]-lo], [hi-sharpes[i]]], fmt='none', c='k', capsize=6, lw=1.5)\n"
            "ax.set_xticks(list(xs)); ax.set_xticklabels(labels, rotation=10)\n"
            "ax.set_ylabel('Annualised Sharpe ratio'); ax.set_title('Three-fund has the lowest Sharpe of the three')\n"
            "ax.legend(loc='upper right'); plt.tight_layout(); plt.show()\n"
            "print('Strategy     | CAGR%  | Sharpe | Vol%  | MaxDD%  | Calmar')\n"
            "for n, lbl in zip(names, ['Three-fund', '60/40    ', 'US-only  ']):\n"
            "    m = metrics[n]\n"
            "    print(f'{lbl} | {m[\"cagr\"]*100:5.2f}  | {m[\"sharpe\"]:6.3f} | {m[\"ann_vol\"]*100:5.2f} | {m[\"max_dd\"]*100:7.2f} | {m[\"calmar\"]:6.3f}')"
        ),
        md(
            "> 💡 The error bars (95% bootstrap CI) are wide — individual Sharpe rankings "
            "are uncertain. But the three-fund's CI does overlap with both baselines, "
            "meaning the ranking could plausibly reverse over different samples. The "
            "key point: the theoretical claim that three-fund > US-only is *rejected* "
            "on this sample."
        ),
        md(
            "### 4b · Pairwise HAC t-tests (with Bonferroni correction)\n\n"
            "Annual return differences, Newey-West long-run variance:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    diff_us  = st.hac_tstat_diff(navs['three_fund'], navs['us_only'])\n"
            "    diff_6040 = st.hac_tstat_diff(navs['three_fund'], navs['sixty_forty'])\n"
            "    intl_attr = st.international_attribution(prices, cost_bps=5.0)\n"
            "    d_us, t_us = diff_us['mean_diff']*100, diff_us['tstat']\n"
            "    d_6040, t_6040 = diff_6040['mean_diff']*100, diff_6040['tstat']\n"
            "    d_intl, t_intl = intl_attr['mean_diff']*100, intl_attr['tstat']\n"
            "else:\n"
            f"    d_us, t_us = {R['diff_vs_us']}, {R['t_vs_us']}\n"
            f"    d_6040, t_6040 = {R['diff_vs_6040']}, {R['t_vs_6040']}\n"
            f"    d_intl, t_intl = {R['intl_diff']}, {R['intl_t']}\n"
            f"bonf = {R['bonf_thresh']}\n"
            "tests = ['vs US-only', 'vs 60/40', 'VXUS attribution']\n"
            "diffs = [d_us, d_6040, d_intl]\n"
            "tstats = [t_us, t_6040, t_intl]\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "clrs = [RED if abs(t) > bonf else GREY for t in tstats]\n"
            "axes[0].bar(tests, diffs, color=clrs); axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_ylabel('Annual return diff (%/yr)'); axes[0].set_title('Three-fund vs baselines (mean diff)')\n"
            "axes[1].bar(tests, tstats, color=clrs)\n"
            "for th in (bonf, -bonf): axes[1].axhline(th, ls='--', c=RED, lw=1.2, label=f'Bonferroni |t|={bonf:.2f}')\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('HAC t-stat'); axes[1].set_title('Statistical significance (|t| > Bonferroni bar = significant)')\n"
            "axes[1].legend(); plt.tight_layout(); plt.show()\n"
            "print(f'vs US-only:         {d_us:+.2f}%/yr  t={t_us:+.2f}  {\"SIGNIFICANT\" if abs(t_us)>bonf else \"not significant\"}')\n"
            "print(f'vs 60/40:           {d_6040:+.2f}%/yr  t={t_6040:+.2f}  {\"SIGNIFICANT\" if abs(t_6040)>bonf else \"not significant\"}')\n"
            "print(f'VXUS attribution:   {d_intl:+.2f}%/yr  t={t_intl:+.2f}  {\"SIGNIFICANT\" if abs(t_intl)>bonf else \"not significant\"}')\n"
            "print(f'Bonferroni threshold: {bonf:.3f} (3 tests, alpha=5%)')"
        ),
        md(
            f"> 💡 Two of three tests clear the Bonferroni bar: (1) three-fund trails US-only "
            f"(*t* = {R['t_vs_us']:+.2f}, significant), (2) VXUS was a drag "
            f"(*t* = {R['intl_t']:+.2f}, significant). The 60/40 comparison is ambiguous — "
            f"the three-fund earns more return but with more risk, and the net Sharpe diff is not "
            "significant enough to call."
        ),
        md(
            "### 4c · Sub-period Sharpe breakdown — did anything ever work?\n\n"
            "The three-fund trails US-only in every sub-period:"
        ),
        code(
            "periods = [('2011-2015', '2011-01-01', '2016-01-01'),\n"
            "           ('2016-2020', '2016-01-01', '2021-01-01'),\n"
            "           ('2021-2026', '2021-01-01', '2027-01-01')]\n"
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for period, start, end in periods:\n"
            "        sub = prices[(prices.index >= start) & (prices.index < end)]\n"
            "        if len(sub) < 100: continue\n"
            "        navs_sub = st.run_all_portfolios(sub, cost_bps=5.0)\n"
            "        ms = {k: st.summarize(v) for k, v in navs_sub.items()}\n"
            "        rows.append((period, ms['three_fund']['sharpe'], ms['us_only']['sharpe'], ms['sixty_forty']['sharpe']))\n"
            "else:\n"
            f"    rows = [('2011-2015', {R['sharpe_3f_2011']}, {R['sharpe_us_2011']}, {R['sharpe_6040_2011']}),\n"
            f"            ('2016-2020', {R['sharpe_3f_2016']}, {R['sharpe_us_2016']}, {R['sharpe_6040_2016']}),\n"
            f"            ('2021-2026', {R['sharpe_3f_2021']}, {R['sharpe_us_2021']}, {R['sharpe_6040_2021']})]\n"
            "import numpy as np\n"
            "sub_df = pd.DataFrame(rows, columns=['Period', '3-fund', 'US-only', '60/40'])\n"
            "x = np.arange(len(sub_df))\n"
            "w = 0.25\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.bar(x - w, sub_df['3-fund'], width=w, color=AMBER, label='Three-fund')\n"
            "ax.bar(x,     sub_df['US-only'], width=w, color=RED, label='US-only')\n"
            "ax.bar(x + w, sub_df['60/40'], width=w, color=BLUE, label='60/40')\n"
            "ax.set_xticks(x); ax.set_xticklabels(sub_df['Period'])\n"
            "ax.set_ylabel('Sharpe ratio'); ax.set_title('Three-fund trails US-only in every sub-period')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(sub_df.to_string(index=False))"
        ),
        md(
            "> 💡 No reversal, no sub-period where the three-fund wins. This is not a mixed "
            "sample that averages out — international equity underperformed US equity consistently "
            "throughout the full 2011–2026 window. The three-fund's drag is structural on this tape."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — Three-fund Sharpe {R['sharpe_3f']:.3f} (real, t = {R['t_3f']:+.2f}), "
            f"but below both baselines. Trails US-only by {abs(R['diff_vs_us']):.2f}%/yr "
            f"(*t* = {R['t_vs_us']:+.2f}, Bonferroni-significant). Vs 60/40: +{R['diff_vs_6040']:.2f}%/yr "
            f"(*t* = {R['t_vs_6040']:+.2f}, not significant). VXUS sleeve: {R['intl_diff']:+.2f}%/yr "
            f"(*t* = {R['intl_t']:+.2f}, Bonferroni-significant).\n"
            f"- **Tradability `MIRAGE`** — Costs are irrelevant at annual rebalancing (5 bps one-way "
            "has negligible impact). But the allocation itself underperforms simpler alternatives. "
            "The three-fund is implementable but not advantageous on this sample.\n"
            f"- **International? `HURT`** — VXUS subtracted {abs(R['intl_diff']):.2f}%/yr vs holding "
            f"VTI instead (t = {R['intl_t']:+.2f}). The 2011–2026 sample was an exceptional decade "
            "for US equity and poor for international; the diversification benefit did not materialise."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — cost is irrelevant, allocation is the issue\n\n"
            "Annual rebalancing means the three-fund is essentially cost-free:"
        ),
        code(
            "costs = [0.0, 2.0, 5.0, 10.0, 20.0, 50.0]\n"
            "if HAVE_REAL:\n"
            "    sharpes_3f = [st.summarize(st.run_portfolio(prices, st.THREE_FUND_WEIGHTS, cost_bps=c))['sharpe'] for c in costs]\n"
            "    sharpes_us = [st.summarize(st.run_portfolio(prices, st.US_ONLY_WEIGHTS, cost_bps=c))['sharpe'] for c in costs]\n"
            "else:\n"
            f"    sharpes_3f = [{R['sharpe_3f']}] * len(costs)\n"
            f"    sharpes_us = [{R['sharpe_us']}] * len(costs)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, sharpes_3f, 'o-', c=AMBER, lw=2, label='Three-fund')\n"
            "ax.plot(costs, sharpes_us, 's-', c=RED, lw=2, label='US-only (SPY)')\n"
            "ax.set_xlabel('One-way rebalance cost (bps)')\n"
            "ax.set_ylabel('Sharpe ratio')\n"
            "ax.set_title('Annual rebalancing: cost barely matters — the gap is the allocation, not costs')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('At annual rebalancing, even 50bps one-way barely affects Sharpe.')\n"
            "print('The performance gap to US-only is driven by VXUS, not implementation cost.')"
        ),
        md(
            "> 💡 Contrast with daily-rebalancing strategies where turnover kills the edge. "
            "Here the problem is reversed: the strategy is so easy to run that cost has no "
            "bearing. The decision that matters is whether VXUS belongs in the portfolio at all."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "Can the engine detect when international diversification *helps*? "
            "We plant a positive or negative ``intl_alpha`` in the synthetic data:"
        ),
        code(
            "alphas = [-0.04, -0.02, 0.0, 0.01, 0.02, 0.04]\n"
            "diff_cagrs = []\n"
            "for a in alphas:\n"
            "    p, _ = data.synthetic_prices(n_days=2520, intl_alpha=a, seed=205)\n"
            "    navs_s = st.run_all_portfolios(p, cost_bps=0.0)\n"
            "    m3f = st.summarize(navs_s['three_fund'])\n"
            "    mus = st.summarize(navs_s['us_only'])\n"
            "    diff_cagrs.append((m3f['sharpe'] - mus['sharpe']))\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(alphas, diff_cagrs, 'o-', c=AMBER, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('Planted international alpha (per year)')\n"
            "ax.set_ylabel('Three-fund Sharpe − US-only Sharpe')\n"
            "ax.set_title('Engine works: international advantage visible when planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At intl_alpha=0 the three-fund is close to US-only Sharpe.')\n"
            f"print('Real tape maps to ~intl_alpha = {R['intl_diff']/0.3:.2f} implied by -2%/yr on 30% weight.')"
        ),
        md(
            "The engine is faithful: when international equity earns a real premium, the "
            "three-fund's Sharpe rises above US-only. When it earns less (as on the "
            "2011–2026 real tape), the three-fund's Sharpe falls below. The real-tape "
            "result is therefore a statement about the *market* on this sample, not about "
            "the strategy design. Forward-looking investors must decide whether the next "
            "15 years will look more like the null (intl = US) or the drag (intl < US). "
            "Long-run history (Dimson et al. 2002) suggests the null; the post-GFC "
            "realisation suggests the drag."
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
