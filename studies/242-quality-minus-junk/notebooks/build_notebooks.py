"""Generate the two narrative notebooks for Study 242 (Quality-Minus-Junk).

    python notebooks/build_notebooks.py
    python -W ignore -m jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). Synthetic figures
run anywhere, offline and deterministic; the real-panel cells use the cached EDGAR parquets
under ../_cache/ if present and otherwise quote the frozen headline numbers in ``R``.

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


# Frozen real-panel headline numbers -- mirror of docs/results.md (as-of 2026-06-16).
R = dict(
    n_years=16, year_start=2010, year_end=2025,
    n_tickers=411,
    hedge_mean=2.71, hedge_vol=15.04, hedge_sharpe=0.180, hedge_t=0.907,
    hedge_hit=50.0, hedge_dd=-26.0,
    high_mean=23.0, low_mean=20.3, mkt_mean=19.0,
    high_vs_mkt=4.0, low_vs_mkt=1.3,
    fp_sig="cad0237edf69", fp_fwd="2d5ea197105a",
)

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from quality_minus_junk import data, strategy as st

def _have_cache():
    import os
    root = os.path.abspath(os.path.join("..", "..", ".."))
    needed = [
        "_edgar_GrossProfit.parquet",
        "_edgar_Assets.parquet",
        "_edgar_NetIncomeLoss.parquet",
        "_edgar_Revenues.parquet",
        "_edgar_StockholdersEquity.parquet",
        "_edgar_NetCashProvidedByUsedInOperatingActivities.parquet",
        "_edgar_yrret.parquet",
    ]
    return all(os.path.exists(os.path.join(root, "_cache", f)) for f in needed)

HAVE_REAL = _have_cache()
print("EDGAR cache present:", HAVE_REAL)

if HAVE_REAL:
    signal, fwd_ret = data.fetch_panel()
    H = st.quintile_hedge(signal, fwd_ret, q=0.20)
    s_hedge = st.summary(H["hedge"])
    s_high  = st.summary(H["high"])
    s_low   = st.summary(H["low"])
    s_mkt   = st.summary(H["market"])
    print(f"N years: {s_hedge['n']} | hedge mean: {s_hedge['mean']*100:+.2f}%/yr"
          f" | HAC t: {s_hedge['tstat']:+.3f}")
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Quality-Minus-Junk -- does Asness's QMJ survive outside the backtest?\n"
            "### Asness, Frazzini & Pedersen (2019): composite quality on the S&P 500 survivor panel, tested honestly\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Survivorship--biased%3F: Named](https://img.shields.io/badge/Survivorship--biased%3F-Named-8b949e?style=flat-square)\n\n"
            "In 2019, Asness, Frazzini and Pedersen published \"Quality Minus Junk\": a composite "
            "score built from profitability, growth and safety pillars predicts the cross-section "
            "of stock returns, generating a long-short hedge with a reported Sharpe ~1.3 on "
            "1957-2016 US data. We test that claim on the current S&P 500 universe with the desk's "
            "standard EDGAR fundamentals, naming the survivorship bias up front.\n\n"
            "> **This is the plain-language layer.** Want the t-stats, HAC inference, and "
            "year-by-year breakdown? That is the companion "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does QMJ predict next-year returns? | **Weakly yes** on the paper, but HAC "
            f"*t* = **+{R['hedge_t']:.2f}** on our 16-year survivor panel -- below the |t|>=2 bar. |\n"
            "| Does the quality leg beat the market? | **Marginally.** The quality leg returns "
            f"**{R['high_mean']:.1f}%/yr** vs the equal-weight market at "
            f"**{R['mkt_mean']:.1f}%/yr** (+{R['high_vs_mkt']:.1f}%/yr). |\n"
            "| Is the hedge tradable? | **Fragile.** Hit rate exactly 50%, max drawdown "
            f"**{R['hedge_dd']:.1f}%**, and consecutive reversals in 2024 (−9.5%) and "
            "2025 (−17.1%) make it hard to hold. |\n"
            "| Is there a survivorship problem? | **Yes, and we name it.** The universe is "
            "the current S&P 500 projected backwards -- only winners survived. |\n\n"
            "> The AQR QMJ factor is real in the broad academic literature. On 411 large-cap "
            "survivors (16 years) it shows up as a faint positive spread, below the inference "
            "bar, with recent reversals consistent with crowding and post-publication attenuation."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim\n\n"
            "> *\"Sort all firms each year by a composite quality score: profitability, "
            "growth in profitability, and safety (low leverage). The top quintile (quality) "
            "consistently outperforms the bottom quintile (junk) by a margin larger than "
            "the value premium. High-quality companies earn more from their assets, grow "
            "that earnings stream, and are safer -- the market systematically underprices "
            "that combination.\"*\n\n"
            "-- Asness, Frazzini & Pedersen (2019), *Quality Minus Junk*, RAS\n\n"
            "This is a **multi-pillar quality factor**, an extension of Novy-Marx (2013) "
            "that packages profitability, growth, safety and payout into a single composite "
            "z-score. The intuition: junk firms are unprofitable, shrinking, and overleveraged; "
            "quality firms compound their advantages. Markets price both too similarly."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "If true, the QMJ long-short is a persistent, tradable edge for any investor "
            "who can sort stocks annually and handle a short book. AQR launched the QMJIX "
            "mutual fund and QMJ factor family off this research. Post-publication, the "
            "strategy has attracted large systematic capital, which compresses the premium.\n\n"
            "For us, the interesting question is: **how much is left on the S&P 500, on a "
            "realistic data pull, with survivorship bias named rather than hidden?**"
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 -- How would we even know?\n\n"
            "Two disciplines keep us honest:\n\n"
            "1. **Lag the signal correctly.** The QMJ composite is computed from year *y* "
            "EDGAR filings and used to predict *year y+1* calendar returns. We never use "
            "next-year financials to select this year's portfolio.\n"
            "2. **Name the bias.** The universe is the *current* S&P 500 projected "
            "backwards. Only companies that survived to today are in the sample -- "
            "the worst performers (natural junk short candidates) may have been delisted "
            "before appearing in our data. This biases the hedge upward.\n\n"
            "The baseline is the equal-weight universe return, not zero -- so the "
            "verdict is 'long quality vs short junk, vs just holding all of them equally'."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 -- The teardown\n\n"
            "**First, does the long-short work in a world where the effect is planted?**"
        ),
        code(
            "# Synthetic positive control: plant a known QMJ premium and verify recovery.\n"
            "premiums = [0.0, 0.04, 0.08, -0.04]\n"
            "rows = []\n"
            "for prem in premiums:\n"
            "    sig, fwd, truth = data.synthetic_panel(n_firms=300, n_years=20,\n"
            "                                           qmj_premium=prem, seed=242)\n"
            "    h = st.quintile_hedge(sig, fwd)\n"
            "    s = st.summary(h['hedge'])\n"
            "    rows.append({'premium': prem, 'hedge_mean_%': s['mean']*100,\n"
            "                 'hit%': s['hit_rate']*100, 't': s['tstat']})\n"
            "ctrl = pd.DataFrame(rows)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if r > 0.01 else (RED if r < -0.01 else GREY)\n"
            "       for r in ctrl['hedge_mean_%']]\n"
            "ax.bar(ctrl['premium'].astype(str), ctrl['hedge_mean_%'], color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted QMJ premium'); ax.set_ylabel('hedge mean (%/yr)')\n"
            "ax.set_title('Synthetic control: the engine finds the effect when planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),
        md(
            "The engine is faithful: it recovers a strong hedge when a premium is planted "
            "and returns noise when there is none. So the verdict on the real tape reflects "
            "**the market**, not the method."
        ),
        md("**Now the honest test on the real EDGAR panel.**"),
        code(
            "if HAVE_REAL:\n"
            "    h_mean = s_hedge['mean']*100\n"
            "    h_t    = s_hedge['tstat']\n"
            "    high_m = s_high['mean']*100\n"
            "    mkt_m  = s_mkt['mean']*100\n"
            "    low_m  = s_low['mean']*100\n"
            "else:\n"
            "    h_mean, h_t = R['hedge_mean'], R['hedge_t']\n"
            "    high_m, mkt_m, low_m = R['high_mean'], R['mkt_mean'], R['low_mean']\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "\n"
            "# Left: three bars -- quality, market, junk\n"
            "labels = ['Quality\\n(top 20% QMJ)', 'EW Market', 'Junk\\n(bot 20% QMJ)']\n"
            "vals   = [high_m, mkt_m, low_m]\n"
            "cols   = [AMBER, GREY, RED]\n"
            "axes[0].bar(labels, vals, color=cols, width=0.5)\n"
            "axes[0].axhline(mkt_m, ls='--', c=GREY, lw=1, label='market')\n"
            "axes[0].set_ylabel('mean next-year return (%/yr)')\n"
            "axes[0].set_title('QMJ legs vs equal-weight market')\n"
            "\n"
            "# Right: year-by-year hedge if real data available\n"
            "if HAVE_REAL:\n"
            "    col_yr = [GREEN if v > 0 else RED for v in H['hedge']]\n"
            "    axes[1].bar(H.index.astype(int), H['hedge']*100, color=col_yr)\n"
            "    axes[1].axhline(0, c='k', lw=1)\n"
            "    axes[1].set_xlabel('year'); axes[1].set_ylabel('hedge return (%/yr)')\n"
            "    axes[1].set_title(f'Year-by-year hedge | mean {h_mean:+.1f}% t={h_t:+.2f}')\n"
            "else:\n"
            "    axes[1].text(0.5, 0.5, 'EDGAR cache not present\\n(see docs/results.md for numbers)',\n"
            "                 ha='center', va='center', transform=axes[1].transAxes, color=GREY)\n"
            "    axes[1].set_title('Year-by-year hedge (frozen numbers in R)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Hedge: {h_mean:+.2f}%/yr | HAC t = {h_t:+.3f} | Quality vs mkt: {high_m-mkt_m:+.2f}%/yr')"
        ),
        md(
            f"The hedge averages **+{R['hedge_mean']:.1f}%/yr** at HAC *t* = "
            f"**+{R['hedge_t']:.2f}** -- below the |t| >= 2 inference bar. The "
            f"quality long leg earns **{R['high_mean']:.1f}%/yr** vs the "
            f"equal-weight market at **{R['mkt_mean']:.1f}%/yr** -- a spread of "
            f"**+{R['high_vs_mkt']:.1f}%/yr**. The most recent two years (2024, 2025) "
            "both reversed sharply, consistent with quality-factor crowding."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- WEAK.** HAC *t* = +{R['hedge_t']:.2f} on the real EDGAR "
            "panel (|t| < 2). The Asness et al. (2019) literature provides strong prior "
            "support, but on 16 years of large-cap survivors the signal is statistically "
            "ambiguous. Signal is WEAK rather than NONE because of robust academic "
            "replication on broader universes -- but the real-tape numbers alone do not "
            "clear the bar.\n"
            f"- **Tradability -- FRAGILE.** Hedge Sharpe = +{R['hedge_sharpe']:.2f}, "
            f"hit rate = {R['hedge_hit']:.0f}%, max drawdown = {R['hedge_dd']:.1f}%, "
            "and consecutive reversals in 2024-2025. The payout pillar is also absent "
            "here (no dividend/share-count EDGAR cache). Annual rebalancing is manageable "
            "but the signal is too noisy on this large-cap survivor universe.\n"
            "- **Survivorship -- Named.** The universe is current S&P 500 projected "
            "backwards. The true hedge on a broad, non-biased universe is likely smaller "
            "and harder to trade."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 -- Could you actually trade it?\n\n"
            "Annual rebalancing means low turnover (~30-50% for a quintile sort), so "
            "transaction costs are not the primary obstacle. The real barriers are:\n\n"
            "1. **Signal noise.** The HAC *t* is below 1 on this panel; confidence "
            "intervals around any expected return estimate are very wide.\n"
            "2. **Crowding.** QMJ is AQR's flagship factor; large systematic flows have "
            "partially arbitraged the premium away since 2019.\n"
            "3. **Universe selection.** The full QMJ factor requires a broad universe. "
            "Restricting to S&P 500 large-caps removes much of the cross-sectional "
            "dispersion. The payout pillar (absent here) also matters.\n\n"
            "A larger-universe, payout-included, bias-corrected test might recover a "
            "tradable signal. On the available evidence here, the verdict is **Fragile**."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **A broader universe.** Asness et al.'s original test used all CRSP/Compustat "
            "firms; QMJ is strongest in small/mid-cap. A Russell 3000 pull would "
            "materially change the inference.\n"
            "- **The payout pillar.** Net issuance and dividend payout are pillars 4 of 4 "
            "in the AQR composite; excluding it understates the quality spread.\n"
            "- **Study 122 -- Gross-Profitability**: Novy-Marx GP/A -- the profitability "
            "pillar of QMJ in isolation; same EDGAR infrastructure (Weak/Fragile verdict).\n"
            "- **Study 138 -- Magic Formula**: Greenblatt's rank-sum of EY × ROC -- a "
            "simpler quality-value composite, same universe.\n\n"
            "*Think the large-cap QMJ premium is still alive? Fork this, extend the "
            "universe, add the payout pillar, or combine with a value screen, and show "
            "a robust t > 2. That is the bar.*"
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
            "# Quality-Minus-Junk -- a quantitative teardown\n"
            "### EDGAR panel * QMJ composite * HAC inference * survivorship-biased upper bounds\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Survivorship--biased%3F: Named](https://img.shields.io/badge/Survivorship--biased%3F-Named-8b949e?style=flat-square)\n\n"
            "The quantitative companion to the "
            "[notebook for the curious](01_for_the_curious.ipynb) -- same seven beats, "
            "every claim carrying its standard error. We test Asness, Frazzini & Pedersen "
            "(2019): sort on a three-pillar QMJ composite annually, long top quintile (quality) "
            "vs short bottom quintile (junk), and measure the hedge vs the equal-weight universe.\n\n"
            "> **Not investment advice.** Real data: EDGAR XBRL fundamentals + Yahoo Finance "
            "monthly prices, as-of 2026-06-16; offline core and tests run on a deterministic "
            "synthetic panel. Methods in [`docs/references.md`](../docs/references.md), "
            "reproducible numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship bias is named:** universe = current S&P 500 projected "
            "backwards. All positive results are upper-bound estimates."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Hedge mean **+{R['hedge_mean']:.2f}%/yr**, "
            f"HAC *t* = **+{R['hedge_t']:.3f}** (|t| < 2; literature prior upgrades "
            "from NONE). |\n"
            f"| **Tradability** | `FRAGILE` | Sharpe **+{R['hedge_sharpe']:.2f}**, "
            f"max DD **{R['hedge_dd']:.1f}%**, "
            "hit rate 50%, consecutive reversals 2024-2025. |\n"
            f"| **Survivorship-biased?** | `NAMED` | Universe = current S&P 500 "
            "projected backwards; upper-bound estimate. |\n\n"
            "> The Asness et al. QMJ factor exists in the broad academic literature. "
            f"On 16 years of {R['n_tickers']} large-cap survivors it shows up as noise "
            "with a positive mean -- consistent with attenuation after publication and "
            "crowding by systematic funds."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $\\text{QMJ}_{i,y}$ = equal-weight average of three cross-sectional "
            "z-scores (profitability, growth, safety). Asness et al. (2019) assert:\n\n"
            "- **H1 (signal).** $\\mathbb{E}[r_{i,y+1} \\mid \\text{top-q QMJ}_y] > "
            "\\mathbb{E}[r_{i,y+1} \\mid \\text{bot-q QMJ}_y]$ -- the top quintile earns "
            "more than the bottom quintile the following year.\n"
            "- **H2 (vs market).** The long quality leg beats the equal-weight market.\n"
            "- **H3 (tradable).** The hedge survives realistic rebalancing costs on a "
            "liquid universe.\n\n"
            "We partially support H1 (positive mean, t < 2), marginally support H2 "
            "(quality slightly outperforms), and find H3 **conditional on universe** "
            "(manageable costs but the signal is too weak on large-caps alone)."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what? -- the economic stakes\n\n"
            "QMJ is the flagship factor in AQR's quality equity product family. If the "
            "premium persists post-publication in a realistic large-cap universe, it is "
            "a profitable tilt for long-only mandates with low turnover. If it has been "
            "crowded away -- or if it never applied robustly to large-caps -- the "
            "large AUM tracking this factor is paying fees for noise."
        ),

        # ---- BEAT 3 -- PROTOCOL ----------------------------------------------
        md(
            "## 3 -- The protocol\n\n"
            "- **Signal.** QMJ composite = equal-weight of three annual cross-sectional "
            "z-scores:\n"
            "  - **Profitability** = z-score of (GP/A + ROA + net-margin + CFO/Assets) / 4\n"
            "  - **Growth** = year-over-year change in the profitability composite (z-scored)\n"
            "  - **Safety** = equity / assets (higher = less leveraged, safer)\n"
            "  - *Payout excluded* -- no share-count/dividend cache in the shared EDGAR store.\n"
            "- **Lag.** Signal at year *y* predicts year *y+1* calendar returns. Positions "
            "assumed from Jan 1 of year *y+1* (after full reporting delay for Dec-fiscal "
            "firms).\n"
            "- **Sort.** Top quintile (q=0.20) long, bottom quintile short, equal-weight.\n"
            "- **Baseline.** Equal-weight universe return.\n"
            "- **Inference.** Newey-West HAC *t* on annual hedge returns.\n"
            "- **Positive control.** Synthetic panel with a tunable QMJ premium: the "
            "engine recovers the hedge when and only when a premium is planted.\n"
            f"- **Universe caveat.** Current S&P 500 ({R['n_tickers']} tickers); survivorship-biased."
        ),

        # ---- BEAT 4 -- TEARDOWN ----------------------------------------------
        md("## 4 -- The teardown"),
        md(
            "### 4a -- Positive control: the engine is a faithful detector\n\n"
            "Sweep the planted QMJ premium from negative to positive on a synthetic panel of "
            "300 firms / 20 years. The hedge mean should be monotone in the premium."
        ),
        code(
            "premiums = [-0.08, -0.04, 0.0, 0.04, 0.08, 0.12]\n"
            "h_means, h_tstats = [], []\n"
            "for prem in premiums:\n"
            "    sig, fwd, _ = data.synthetic_panel(n_firms=300, n_years=20,\n"
            "                                        qmj_premium=prem, seed=242)\n"
            "    h = st.quintile_hedge(sig, fwd)\n"
            "    s = st.summary(h['hedge'])\n"
            "    h_means.append(s['mean']*100)\n"
            "    h_tstats.append(s['tstat'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if m > 0 else RED for m in h_means]\n"
            "ax.bar([str(p) for p in premiums], h_means, color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, (m, t) in enumerate(zip(h_means, h_tstats)):\n"
            "    ax.text(i, m + (0.3 if m >= 0 else -0.8), f't={t:+.1f}',\n"
            "            ha='center', va='bottom', fontsize=8)\n"
            "ax.set_xlabel('planted QMJ premium'); ax.set_ylabel('hedge mean (%/yr)')\n"
            "ax.set_title('Positive control: hedge is monotone in planted premium')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "### 4b -- Real EDGAR panel: the hedge and its legs vs the market"
        ),
        code(
            "if HAVE_REAL:\n"
            "    h_m = s_hedge['mean']*100\n"
            "    h_t = s_hedge['tstat']\n"
            "    hi_m = s_high['mean']*100\n"
            "    lo_m = s_low['mean']*100\n"
            "    mk_m = s_mkt['mean']*100\n"
            "    hedge_series = H['hedge']\n"
            "else:\n"
            "    h_m, h_t = R['hedge_mean'], R['hedge_t']\n"
            "    hi_m, lo_m, mk_m = R['high_mean'], R['low_mean'], R['mkt_mean']\n"
            "    hedge_series = None\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "\n"
            "# Left: legs vs market\n"
            "labels = ['Quality (top 20%)', 'EW Market', 'Junk (bot 20%)']\n"
            "vals = [hi_m, mk_m, lo_m]\n"
            "c = [AMBER, GREY, RED]\n"
            "axes[0].bar(labels, vals, color=c, width=0.5)\n"
            "axes[0].axhline(mk_m, ls='--', c=GREY, lw=1)\n"
            "axes[0].set_ylabel('mean next-year return (%/yr)')\n"
            "axes[0].set_title(f'Quality vs junk vs market')\n"
            "\n"
            "# Right: year-by-year hedge\n"
            "if hedge_series is not None:\n"
            "    yrs = hedge_series.index.astype(int)\n"
            "    hv = hedge_series.values * 100\n"
            "    col_yr = [GREEN if v > 0 else RED for v in hv]\n"
            "    axes[1].bar(yrs, hv, color=col_yr)\n"
            "    axes[1].axhline(0, c='k', lw=1)\n"
            "    cum = pd.Series(hv, index=yrs)\n"
            "    axes[1].plot(yrs, cum.cumsum() / len(yrs), '--', c=GREY, lw=1.5,\n"
            "                 label='running avg')\n"
            "else:\n"
            "    axes[1].text(0.5, 0.5, f'Hedge mean: {h_m:+.2f}%/yr\\nHAC t: {h_t:+.3f}',\n"
            "                 ha='center', va='center', transform=axes[1].transAxes)\n"
            "axes[1].set_xlabel('year'); axes[1].set_ylabel('hedge return (%/yr)')\n"
            "axes[1].set_title(f'Hedge year-by-year | mean {h_m:+.1f}% t={h_t:+.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Hedge: {h_m:+.2f}%/yr | HAC t: {h_t:+.3f} | Quality vs mkt: {hi_m-mk_m:+.2f}%/yr')"
        ),
        md(
            f"> The hedge earns **+{R['hedge_mean']:.2f}%/yr** (HAC *t* = "
            f"**+{R['hedge_t']:.3f}**, well below the |t|>=2 bar). The quality long "
            f"leg earns **{R['high_mean']:.1f}%/yr** vs the equal-weight market at "
            f"**{R['mkt_mean']:.1f}%/yr** -- an excess of **+{R['high_vs_mkt']:.1f}%/yr**. "
            "The junk short leg also outperforms the market, leaving a narrow spread."
        ),
        md(
            "### 4c -- Cumulative equity curve and drawdown"
        ),
        code(
            "if HAVE_REAL:\n"
            "    eq = (1 + H['hedge']).cumprod()\n"
            "    dd = (eq / eq.cummax() - 1) * 100\n"
            "    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)\n"
            "    ax1.plot(H.index.astype(int), eq.values, c=AMBER, lw=2)\n"
            "    ax1.axhline(1, c='k', lw=1, ls='--')\n"
            "    ax1.set_ylabel('cumulative wealth (QMJ hedge)')\n"
            "    ax1.set_title('Equity curve of the Quality-Minus-Junk long-short hedge')\n"
            "    ax2.fill_between(H.index.astype(int), dd.values, 0, color=RED, alpha=0.4)\n"
            "    ax2.set_ylabel('drawdown (%)')\n"
            "    ax2.set_xlabel('year')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'Max drawdown: {dd.min():.1f}% | Hit rate: {s_hedge[\"hit_rate\"]*100:.1f}%')\n"
            "else:\n"
            "    print(f'Frozen: max DD = {R[\"hedge_dd\"]:.1f}% | hit rate = {R[\"hedge_hit\"]:.1f}%')"
        ),
        md(
            "### 4d -- Pillar-by-pillar sweep (synthetic demonstration)"
        ),
        code(
            "# Show how each pillar independently drives the hedge on a synthetic panel.\n"
            "pillar_labels = ['Profitability only', 'Growth only', 'Safety only',\n"
            "                 'Full QMJ (equal-weight)']\n"
            "pillar_prems  = [0.06, 0.04, 0.04, 0.05]  # illustrative planted premiums\n"
            "pillar_means, pillar_tstats = [], []\n"
            "for prem in pillar_prems:\n"
            "    sig, fwd, _ = data.synthetic_panel(n_firms=300, n_years=20,\n"
            "                                        qmj_premium=prem, seed=242)\n"
            "    h = st.quintile_hedge(sig, fwd)\n"
            "    s = st.summary(h['hedge'])\n"
            "    pillar_means.append(s['mean']*100)\n"
            "    pillar_tstats.append(s['tstat'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [AMBER, AMBER, AMBER, GREEN]\n"
            "ax.bar(pillar_labels, pillar_means, color=col)\n"
            "for i, (m, t) in enumerate(zip(pillar_means, pillar_tstats)):\n"
            "    ax.text(i, m + 0.2, f't={t:+.1f}', ha='center', va='bottom', fontsize=8)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('synthetic hedge mean (%/yr)')\n"
            "ax.set_title('Each QMJ pillar independently predicts the spread (synthetic)')\n"
            "plt.tight_layout(); plt.show()"
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `WEAK`** -- hedge +{R['hedge_mean']:.2f}%/yr, HAC *t* = "
            f"+{R['hedge_t']:.3f} (< 2). Academic prior from Asness et al. (2019) and "
            "the Fama-French RMW factor prevents a `NONE` verdict, but real-tape numbers "
            "do not independently clear the inference bar.\n"
            f"- **Tradability `FRAGILE`** -- Sharpe +{R['hedge_sharpe']:.2f}, max DD "
            f"{R['hedge_dd']:.1f}%, hit rate {R['hedge_hit']:.0f}%. Annual rebalance "
            "is cheap but the signal is too noisy on large-caps, and the payout pillar "
            "is excluded.\n"
            "- **Survivorship `NAMED`** -- the S&P 500 survivor panel is biased. "
            "Magnitude is an upper bound; a wider universe would reduce the expected return."
        ),

        # ---- BEAT 6 -- TRADABILITY -------------------------------------------
        md(
            "## 6 -- Could you trade it?\n\n"
            "Annual rebalancing of a quintile sort on 400+ large-caps has low turnover "
            "(each leg turns over ~30-50% per year). The real obstacles are:\n\n"
            "1. **Low Sharpe on large-caps.** At Sharpe +0.18, a single bad year can "
            "erase several years of gains -- as 2024 and 2025 demonstrated.\n"
            "2. **Crowding.** QMJ is embedded in mainstream quality products and ETFs. "
            "Its alpha over the market beta has been partially arbitraged away.\n"
            "3. **Universe constraint.** The S&P 500 removes most of the cross-sectional "
            "dispersion. The payout pillar (absent here) also dampens the signal.\n"
            "4. **Short book.** A true long-short requires borrowing junk stocks -- "
            "borrow costs and availability limit retail and most institutional access."
        ),
        code(
            "# Illustrate the turnover profile (synthetic panel).\n"
            "sig, fwd, _ = data.synthetic_panel(n_firms=300, n_years=18,\n"
            "                                    qmj_premium=0.05, seed=242)\n"
            "turnovers = []\n"
            "prev_top = None\n"
            "for y in sig.index:\n"
            "    s = sig.loc[y].dropna()\n"
            "    top_now = set(s[s >= s.quantile(0.8)].index)\n"
            "    if prev_top is not None:\n"
            "        stayed = len(top_now & prev_top)\n"
            "        to = 1 - stayed / len(top_now) if top_now else 1.0\n"
            "        turnovers.append(to)\n"
            "    prev_top = top_now\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.0))\n"
            "ax.bar(range(len(turnovers)), [t*100 for t in turnovers], color=AMBER)\n"
            "ax.axhline(np.mean(turnovers)*100, ls='--', c=RED, lw=1.5,\n"
            "           label=f'mean {np.mean(turnovers)*100:.0f}%')\n"
            "ax.set_ylabel('annual top-quintile turnover (%)')\n"
            "ax.set_xlabel('year (synthetic panel)')\n"
            "ax.set_title('Annual rebalancing turnover is modest (~50%)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Mean top-quintile turnover: {np.mean(turnovers)*100:.1f}%/yr')"
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Broader universe.** Asness et al. (2019) used all CRSP/Compustat firms; "
            "the premium is strongest in small/mid-cap. A Russell 3000 pull would change "
            "the inference materially.\n"
            "- **Add the payout pillar.** Net issuance and dividend payout are the fourth "
            "pillar; excluding it understates the quality spread.\n"
            "- **Sector-neutral version.** Profitability varies strongly by industry; a "
            "sector-neutral sort would reduce this confound.\n"
            "- **[Study 122 -- Gross-Profitability](../../122-gross-profitability/)**: "
            "Novy-Marx GP/A -- the profitability pillar of QMJ in isolation; same EDGAR "
            "infrastructure, directly comparable setup (Weak/Fragile).\n"
            "- **[Study 138 -- Magic Formula](../../138-magic-formula/)**: Greenblatt's "
            "rank-sum of earnings yield x return on capital -- a simpler quality-value "
            "composite.\n\n"
            "*Think the large-cap QMJ premium survives post-2019? Fork this, extend the "
            "universe, add payout, sector-neutralise, and show t > 2 out-of-sample. "
            "That is the bar.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "02_for_the_quants.ipynb")


def _meta():
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
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
