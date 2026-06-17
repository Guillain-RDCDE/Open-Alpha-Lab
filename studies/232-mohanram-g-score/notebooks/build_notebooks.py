"""Generate the two narrative notebooks for Study 232 (Mohanram G-score).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). Synthetic
figures run anywhere, offline and deterministic; the real-panel cells use the cached
EDGAR parquets under _cache/ if present and otherwise quote the frozen headline
numbers in ``R``.

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
    n_years=19, year_start=2007, year_end=2025,
    n_tickers=318,
    hedge_mean=-2.98, hedge_vol=8.60, hedge_sharpe=-0.347, hedge_t=-1.613,
    hedge_hit=47.4, hedge_dd=-47.9,
    high_mean=14.53, low_mean=17.52, mkt_mean=16.00,
    high_vs_mkt=-1.47, low_vs_mkt=1.51,
    fp_sig="a9801f04b364", fp_fwd="d1ecef461fd1",
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

from mohanram_g_score import data, strategy as st

def _have_cache():
    root = os.path.abspath(os.path.join("..", "..", ".."))
    required = [
        os.path.join(root, "_cache", "_edgar_NetIncomeLoss.parquet"),
        os.path.join(root, "_cache", "_edgar_Assets.parquet"),
        os.path.join(root, "_cache", "_edgar_NetCashProvidedByUsedInOperatingActivities.parquet"),
        os.path.join(root, "_cache", "_edgar_Revenues.parquet"),
        os.path.join(root, "_cache", "_edgar_yrret.parquet"),
    ]
    return all(os.path.exists(p) for p in required)

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
            "# Mohanram G-score -- can 8 fundamental signals find the growth stocks that actually deliver?\n"
            "### Mohanram (2005): the growth-stock quality screen on the S&P 500 survivor panel, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Can%20G--score%20find%20growth%20winners%3F: BUSTED](https://img.shields.io/badge/Can%20G--score%20find%20growth%20winners%3F-BUSTED-8b949e?style=flat-square)\n\n"
            "In 2005, Partha Mohanram proposed the G-score: eight binary accounting signals "
            "that together identify which growth (low book-to-market) stocks have the "
            "fundamental support to actually deliver -- as opposed to glamour stocks priced "
            "purely on investor optimism. The G-score is the growth counterpart to Piotroski's "
            "F-score for value stocks. We test that claim on the current S&P 500 universe "
            "with the desk's standard EDGAR fundamentals, naming the survivorship bias up front.\n\n"
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
            "| Does the G-score predict which growth stocks will outperform? | **No.** "
            f"The hedge averages **{R['hedge_mean']:+.2f}%/yr** at HAC *t* = "
            f"**{R['hedge_t']:+.2f}** -- *in the wrong direction*. |\n"
            "| Does the high-G leg beat the market? | **No.** The long leg returns "
            f"**{R['high_mean']:.1f}%/yr**, trailing the equal-weight market at "
            f"**{R['mkt_mean']:.1f}%/yr** by **{abs(R['high_vs_mkt']):.1f}%/yr**. |\n"
            "| Does low-G underperform? | **Also no.** Low-G firms returned "
            f"**{R['low_mean']:.1f}%/yr**, *beating* the market by "
            f"**+{R['low_vs_mkt']:.1f}%/yr**. The signal inverts. |\n"
            "| Is this a data or method problem? | Partly. G6/G7 are approximated "
            "(no R&D/advertising in the desk cache), and the S&P 500 survivor panel "
            "only includes the already-successful large-cap universe. |\n\n"
            "> On large-cap S&P 500 survivors, the Mohanram G-score is inverted: "
            "firms with stronger accounting fundamentals *earn less* than their weaker "
            "peers. This is consistent with the market correctly pricing (or even "
            "over-discounting) fundamental quality in large-cap growth names."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim\n\n"
            "> *\"Among low book-to-market (growth) stocks, those with strong accounting "
            "fundamentals -- profitable, stable earnings, conservative investment, "
            "innovative -- earn significantly higher returns than those without. "
            "A composite G-score of eight binary signals separates the winners from "
            "the losers within the glamour category.\"*\n\n"
            "-- Mohanram (2005), *Review of Accounting Studies*\n\n"
            "The G-score is a mirror of Piotroski's F-score (Study 65), but aimed at "
            "the opposite end of the book-to-market spectrum. Where the F-score finds "
            "the few value stocks that are genuinely turning around, the G-score finds "
            "the few growth stocks that actually have the earnings power to justify "
            "their premium valuation. Eight signals: three on profitability (G1-G3), "
            "two on earnings stability (G4-G5), and three on growth conservatism "
            "(G6-G8). Each is 1 if the firm clears the bar, 0 otherwise."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "If the G-score works, it is a free long-only tilt: buy the growth stocks "
            "with real accounting support, avoid the glamour stocks with air behind "
            "the price. Annual rebalancing from annual 10-K filings -- cheap to operate "
            "if the signal is there.\n\n"
            "Post-publication attenuation is always a concern: the G-score is 20 years "
            "old, well-known, and has been absorbed into systematic growth quality "
            "strategies. The question is whether anything survives on the large-cap "
            "universe that most institutional investors can actually access."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 -- How would we even know?\n\n"
            "Two disciplines keep us honest:\n\n"
            "1. **Lag the signal correctly.** The G-score is computed from year *y* "
            "EDGAR 10-K filings and used to predict *year y+1* calendar returns. "
            "We never use next-year financials.\n"
            "2. **Name the bias.** The universe is the *current* S&P 500 projected "
            "backwards -- only companies that survived to today are in the sample. "
            "This inflates any positive long-leg estimate but does not explain a "
            "negative hedge direction.\n\n"
            "Two honest caveats:\n\n"
            "- **G6/G7 substitution.** R&D intensity and advertising intensity (the "
            "original Mohanram G6/G7) are not in the desk's EDGAR cache. We substitute "
            "revenue-growth and asset-turnover-growth. This may under- or over-estimate "
            "the true G-score effect.\n"
            "- **Universe.** Mohanram (2005) sorted on all COMPUSTAT firms. We sort on "
            "S&P 500 large-caps only. The paper's effect was strongest in smaller growth "
            "stocks with high analyst coverage -- precisely the stocks excluded here."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 -- The teardown\n\n"
            "**First, does the long-short work in a world where the effect is planted?**"
        ),
        code(
            "# Synthetic positive control: plant a known G-score premium and verify recovery.\n"
            "premiums = [-0.06, 0.0, 0.04, 0.08]\n"
            "rows = []\n"
            "for prem in premiums:\n"
            "    sig, fwd, truth = data.synthetic_panel(n_firms=300, n_years=20,\n"
            "                                            g_premium=prem, seed=232)\n"
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
            "ax.set_xlabel('planted G-score premium'); ax.set_ylabel('hedge mean (%/yr)')\n"
            "ax.set_title('Synthetic control: the engine finds the effect when planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),
        md(
            "The engine is faithful: when a premium is planted, the hedge recovers it. "
            "When there is none, it returns noise around zero. So the negative result "
            "on the real tape reflects **the market** and **the signal**, not the method."
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
            "# Left: three bars -- high, market, low\n"
            "labels = ['High G-score\\n(top 20%)', 'EW Market', 'Low G-score\\n(bot 20%)']\n"
            "vals   = [high_m, mkt_m, low_m]\n"
            "cols   = [RED, GREY, AMBER]  # high-G underperforms, low-G outperforms\n"
            "axes[0].bar(labels, vals, color=cols, width=0.5)\n"
            "axes[0].axhline(mkt_m, ls='--', c=GREY, lw=1, label='market')\n"
            "axes[0].set_ylabel('mean next-year return (%/yr)')\n"
            "axes[0].set_title('Signal inverts: low-G beats high-G')\n"
            "\n"
            "# Right: year-by-year hedge\n"
            "if HAVE_REAL:\n"
            "    col_yr = [GREEN if v > 0 else RED for v in H['hedge']]\n"
            "    axes[1].bar(H.index.astype(int), H['hedge']*100, color=col_yr)\n"
            "    axes[1].axhline(0, c='k', lw=1)\n"
            "    axes[1].set_xlabel('year'); axes[1].set_ylabel('hedge return (%/yr)')\n"
            "    axes[1].set_title(f'Year-by-year hedge | mean {h_mean:+.1f}% t={h_t:+.2f}')\n"
            "else:\n"
            "    axes[1].text(0.5, 0.5,\n"
            "                 f'Hedge mean: {h_mean:+.2f}%/yr\\nHAC t: {h_t:+.3f}\\n(inverted)',\n"
            "                 ha='center', va='center', transform=axes[1].transAxes, color=RED)\n"
            "    axes[1].set_title('Year-by-year hedge (frozen numbers in R)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Hedge: {h_mean:+.2f}%/yr | HAC t = {h_t:+.3f} | High vs mkt: {high_m-mkt_m:+.2f}%/yr')"
        ),
        md(
            f"The hedge averages **{R['hedge_mean']:+.1f}%/yr** at HAC *t* = "
            f"**{R['hedge_t']:+.2f}** -- *in the wrong direction*. The "
            f"high-G leg earns **{R['high_mean']:.1f}%/yr** while the "
            f"equal-weight market earns **{R['mkt_mean']:.1f}%/yr** -- the "
            f"long leg **trails** the market by **{abs(R['high_vs_mkt']):.1f}%/yr**. "
            f"The low-G leg earns **{R['low_mean']:.1f}%/yr**, "
            f"**beating** the market by +{R['low_vs_mkt']:.1f}%/yr. "
            "The signal is inverted on large-cap survivors."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- NONE.** HAC *t* = {R['hedge_t']:+.2f} on the real EDGAR "
            "panel, in the **negative** direction. High-G stocks underperform low-G "
            "stocks by 3%/yr. This is the opposite of the original claim. The signal "
            "does not work on large-cap S&P 500 survivors.\n"
            f"- **Tradability -- MIRAGE.** Negative expected return at Sharpe "
            f"{R['hedge_sharpe']:+.2f}, max drawdown {R['hedge_dd']:.1f}%, hit rate "
            f"{R['hedge_hit']:.1f}%. There is no alpha to trade.\n"
            "- **Why the inversion?** On large-cap survivors, growth firms are already "
            "priced for quality: the market has identified and priced the fundamentally "
            "strong ones. The 'cheap growth' universe where Mohanram (2005) found the "
            "effect -- smaller, under-followed firms with analyst mispricing -- is "
            "absent here."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 -- Could you actually trade it?\n\n"
            "You would not want to. On this panel, the trade is short high-G, long "
            "low-G -- the reverse of the paper's recommendation. But:\n\n"
            "1. **The |t| is below 2.** At HAC *t* = -1.61, the inversion is not "
            "statistically distinguishable from noise. You cannot confidently trade "
            "either direction.\n"
            "2. **Data limitations.** G6 (R&D) and G7 (advertising) are approximated. "
            "A full implementation might change the direction, though likely not enough "
            "to cross the inference bar on large-caps.\n"
            "3. **Wrong universe.** The published G-score effect requires small/mid-cap "
            "growth stocks with optimistic analyst coverage. S&P 500 large-caps are "
            "heavily analysed and arbitraged; there is less room for fundamental-quality "
            "mispricings to persist."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **A broader universe.** Test on all EDGAR-available firms (Russell 3000 "
            "style): the paper's effect was strongest among the small/mid-cap growth "
            "segment where analyst mispricing is larger.\n"
            "- **Add R&D and advertising.** Fetching `ResearchAndDevelopmentExpense` "
            "and `AdvertisingExpense` from data.sec.gov would complete the original G6/G7.\n"
            "- **Sector-neutral.** Growth-stock accounting metrics vary enormously by "
            "sector; a sector-neutral G-score sort would remove confounding sector bets.\n"
            "- **[Study 65 -- Scorecard](../../65-scorecard/)**: Piotroski's F-score on "
            "the value-stock counterpart -- the twin signal that powers the same idea "
            "on the other side of the book-to-market spectrum.\n"
            "- **[Study 122 -- Gross-Profitability](../../122-gross-profitability/)**: "
            "Novy-Marx GP/A factor -- a simpler one-signal version of the profitability "
            "dimension; also Weak on large-caps.\n\n"
            "*Think the G-score works in a broader, bias-corrected universe? Fork this, "
            "extend to small-caps, add the missing R&D signals, and show t > 2. "
            "That is the bar.*"
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
            "# Mohanram G-score -- a quantitative teardown\n"
            "### EDGAR panel * 8-signal composite * HAC inference * survivorship-biased upper bounds\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Can%20G--score%20find%20growth%20winners%3F: BUSTED](https://img.shields.io/badge/Can%20G--score%20find%20growth%20winners%3F-BUSTED-8b949e?style=flat-square)\n\n"
            "The quantitative companion to the "
            "[notebook for the curious](01_for_the_curious.ipynb) -- same seven beats, "
            "every claim carrying its standard error. We test Mohanram (2005): sort on "
            "the 8-signal G-score annually, long top quintile vs short bottom quintile, "
            "and measure the hedge vs the equal-weight universe.\n\n"
            "> **Not investment advice.** Real data: EDGAR XBRL fundamentals "
            "(NetIncomeLoss, Assets, CFO, Revenues) + Yahoo Finance monthly prices, "
            "as-of 2026-06-16; offline core and tests run on a deterministic synthetic "
            "panel. Methods in [`docs/references.md`](../docs/references.md), "
            "reproducible numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship bias is named:** universe = current S&P 500 projected "
            "backwards. All positive results are upper-bound estimates; negative results "
            "are not explained by this bias."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Hedge mean **{R['hedge_mean']:+.2f}%/yr**, "
            f"HAC *t* = **{R['hedge_t']:+.3f}** -- inverted. High-G underperforms low-G. |\n"
            f"| **Tradability** | `MIRAGE` | Sharpe **{R['hedge_sharpe']:+.2f}**, "
            f"max DD **{R['hedge_dd']:.1f}%**, hit rate **{R['hedge_hit']:.1f}%**. "
            "Negative expected return. |\n"
            f"| **Can G-score find winners?** | `BUSTED` | Signal inverts on large-cap "
            "survivors; high-G firms underperform by -3%/yr. |\n\n"
            "> The G-score was designed for small/mid-cap growth stocks with analyst "
            "mispricing. On 318 S&P 500 large-cap survivors, the fundamental quality "
            "differential is already priced in -- or the market has overshot it."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $G_{i,y}$ = Mohanram G-score in {0,...,8} for firm $i$ in year $y$. "
            "Mohanram (2005) asserts:\n\n"
            "- **H1 (signal).** $\\mathbb{E}[r_{i,y+1} \\mid \\text{top-q } G_y] > "
            "\\mathbb{E}[r_{i,y+1} \\mid \\text{bot-q } G_y]$ -- the top-quintile G-score "
            "firms earn more the following year.\n"
            "- **H2 (vs market).** The long leg beats the equal-weight market.\n"
            "- **H3 (tradable).** The hedge survives realistic rebalancing costs.\n\n"
            "We reject H1 (signal inverts, t = -1.61), reject H2 (long leg lags the "
            "market), and H3 is moot (negative expected return before costs)."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what? -- the economic stakes\n\n"
            "The G-score is the growth-stock counterpart to Piotroski's F-score, now "
            "embedded in systematic growth-quality screens. If it works, it is a low-"
            "turnover tilt for any growth-oriented mandate. If it fails -- or inverts -- "
            "on large-caps, it may actually be harvesting a **risk premium in reverse**: "
            "the safest, most profitable large-cap growth firms may be the *most* "
            "expensively priced, leaving less upside."
        ),

        # ---- BEAT 3 -- PROTOCOL ----------------------------------------------
        md(
            "## 3 -- The protocol\n\n"
            "- **Signal.** G-score from year-*y* EDGAR 10-K filings (NI, Assets, CFO, Rev).\n"
            "  - G1: ROA > cross-sectional median\n"
            "  - G2: CFO/Assets > cross-sectional median\n"
            "  - G3: Accruals (NI-CFO)/Assets < cross-sectional median\n"
            "  - G4: Rolling-3yr ROA std < cross-sectional median\n"
            "  - G5: Rolling-3yr CFO/A std < cross-sectional median\n"
            "  - G6: Revenue growth > 0 (proxy for growth conservatism)\n"
            "  - G7: Asset-turnover growth > 0 (proxy for efficiency investment)\n"
            "  - G8: ROA > 0 (absolute profitability)\n"
            "- **Lag.** Signal at year *y* predicts year *y+1* calendar returns. "
            "Positions from Jan 1 of year *y+1*.\n"
            "- **Sort.** Top quintile (q=0.20) long, bottom quintile short, equal-weight.\n"
            "- **Baseline.** Equal-weight universe return.\n"
            "- **Inference.** Newey-West HAC *t* on 19 annual hedge returns.\n"
            "- **Universe.** Current S&P 500 (318 tickers with all four EDGAR fields); "
            "survivorship-biased."
        ),

        # ---- BEAT 4 -- TEARDOWN ----------------------------------------------
        md("## 4 -- The teardown"),
        md(
            "### 4a -- Positive control: the engine detects planted premiums faithfully"
        ),
        code(
            "premiums = [-0.08, -0.04, 0.0, 0.04, 0.08, 0.12]\n"
            "h_means, h_tstats = [], []\n"
            "for prem in premiums:\n"
            "    sig, fwd, _ = data.synthetic_panel(n_firms=300, n_years=20,\n"
            "                                        g_premium=prem, seed=232)\n"
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
            "    ax.text(i, m + (0.2 if m >= 0 else -0.6), f't={t:+.1f}',\n"
            "            ha='center', va='bottom', fontsize=8)\n"
            "ax.set_xlabel('planted G-score premium'); ax.set_ylabel('hedge mean (%/yr)')\n"
            "ax.set_title('Positive control: hedge is monotone in planted premium')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "### 4b -- Real EDGAR panel: the signal is inverted"
        ),
        code(
            "if HAVE_REAL:\n"
            "    h_m  = s_hedge['mean']*100\n"
            "    h_t  = s_hedge['tstat']\n"
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
            "labels = ['High G-score', 'EW Market', 'Low G-score']\n"
            "vals = [hi_m, mk_m, lo_m]\n"
            "c = [RED, GREY, AMBER]  # inverted: high underperforms, low outperforms\n"
            "axes[0].bar(labels, vals, color=c, width=0.5)\n"
            "axes[0].axhline(mk_m, ls='--', c=GREY, lw=1)\n"
            "axes[0].set_ylabel('mean next-year return (%/yr)')\n"
            "axes[0].set_title(f'Signal inverted: high G trails market by {hi_m-mk_m:+.1f}%/yr')\n"
            "\n"
            "# Right: year-by-year hedge\n"
            "if hedge_series is not None:\n"
            "    yrs = hedge_series.index.astype(int)\n"
            "    hv = hedge_series.values * 100\n"
            "    col_yr = [GREEN if v > 0 else RED for v in hv]\n"
            "    axes[1].bar(yrs, hv, color=col_yr)\n"
            "    axes[1].axhline(0, c='k', lw=1)\n"
            "    axes[1].plot(yrs, pd.Series(hv).expanding().mean().values, '--',\n"
            "                 c=GREY, lw=1.5, label='expanding mean')\n"
            "    axes[1].legend(fontsize=8)\n"
            "else:\n"
            "    axes[1].text(0.5, 0.5,\n"
            "                 f'Hedge mean: {h_m:+.2f}%/yr\\nHAC t: {h_t:+.3f}\\n(INVERTED)',\n"
            "                 ha='center', va='center', transform=axes[1].transAxes, color=RED)\n"
            "axes[1].set_xlabel('year'); axes[1].set_ylabel('hedge return (%/yr)')\n"
            "axes[1].set_title(f'Hedge year-by-year | mean {h_m:+.1f}% t={h_t:+.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "h_vol = s_hedge['vol']*100 if HAVE_REAL else R['hedge_vol']\n"
            "h_sr  = s_hedge['sharpe'] if HAVE_REAL else R['hedge_sharpe']\n"
            "print(f'Hedge: {h_m:+.2f}%/yr | vol: {h_vol:.1f}% | Sharpe: {h_sr:+.3f} | HAC t: {h_t:+.3f}')"
        ),
        md(
            f"> The hedge earns **{R['hedge_mean']:+.2f}%/yr** (HAC *t* = "
            f"**{R['hedge_t']:+.3f}**) -- below the |t| >= 2 bar and in the **wrong "
            "direction**. The high-G long leg earns "
            f"**{R['high_mean']:.1f}%/yr** vs the equal-weight market at "
            f"**{R['mkt_mean']:.1f}%/yr** -- the long leg **trails** the market by "
            f"**{abs(R['high_vs_mkt']):.1f}%/yr**. The low-G short leg earns "
            f"**{R['low_mean']:.1f}%/yr**, *outperforming* the market by "
            f"+{R['low_vs_mkt']:.1f}%/yr."
        ),
        md(
            "### 4c -- Cumulative equity curve and drawdown"
        ),
        code(
            "if HAVE_REAL:\n"
            "    eq = (1 + H['hedge']).cumprod()\n"
            "    dd = (eq / eq.cummax() - 1) * 100\n"
            "    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)\n"
            "    ax1.plot(H.index.astype(int), eq.values, c=RED, lw=2)\n"
            "    ax1.axhline(1, c='k', lw=1, ls='--')\n"
            "    ax1.set_ylabel('cumulative wealth (hedge)')\n"
            "    ax1.set_title('Equity curve: high-G vs low-G long-short (declining)')\n"
            "    ax2.fill_between(H.index.astype(int), dd.values, 0, color=RED, alpha=0.4)\n"
            "    ax2.set_ylabel('drawdown (%)')\n"
            "    ax2.set_xlabel('year')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'Max drawdown: {dd.min():.1f}% | Hit rate: {s_hedge[\"hit_rate\"]*100:.1f}%')\n"
            "else:\n"
            "    print(f'Frozen: max DD = {R[\"hedge_dd\"]:.1f}% | hit rate = {R[\"hedge_hit\"]:.1f}%')"
        ),
        md(
            "### 4d -- G-score component analysis (synthetic)"
        ),
        code(
            "# Illustrate how individual G-score components contribute in the synthetic world\n"
            "sig, fwd, _ = data.synthetic_panel(n_firms=200, n_years=18,\n"
            "                                    g_premium=0.08, seed=232)\n"
            "h = st.quintile_hedge(sig, fwd)\n"
            "# Show score distribution over time\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "ax1.hist(sig.values.flatten(), bins=9, range=(-0.5, 8.5),\n"
            "         color=AMBER, edgecolor='white', rwidth=0.8)\n"
            "ax1.set_xlabel('G-score (0-8)'); ax1.set_ylabel('firm-years')\n"
            "ax1.set_title('G-score distribution (synthetic, Binomial(8,0.5))')\n"
            "ax1.set_xticks(range(9))\n"
            "\n"
            "# Quintile means under premium\n"
            "all_scores = sig.values.flatten()\n"
            "all_rets   = fwd.values.flatten()\n"
            "bins = np.nanpercentile(all_scores, [0, 20, 40, 60, 80, 100])\n"
            "qmeans = []\n"
            "for lo, hi in zip(bins[:-1], bins[1:]):\n"
            "    mask = (all_scores >= lo) & (all_scores < hi)\n"
            "    qmeans.append(all_rets[mask].mean() * 100)\n"
            "ax2.bar(range(1, 6), qmeans, color=[RED,RED,GREY,AMBER,GREEN])\n"
            "ax2.axhline(np.nanmean(all_rets)*100, ls='--', c=GREY, lw=1.5, label='market')\n"
            "ax2.set_xlabel('G-score quintile (1=lowest, 5=highest)')\n"
            "ax2.set_ylabel('mean next-year return (%/yr)')\n"
            "ax2.set_title(f'Planted premium=0.08: returns rise with G-score')\n"
            "ax2.legend(); plt.tight_layout(); plt.show()"
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `NONE`** -- hedge {R['hedge_mean']:+.2f}%/yr, HAC *t* = "
            f"{R['hedge_t']:+.3f} (negative direction). High-G stocks underperform "
            "low-G stocks by -3%/yr on this 19-year survivor panel. The original "
            "Mohanram (2005) effect does not survive on large-cap S&P 500 survivors "
            "over this period.\n"
            f"- **Tradability `MIRAGE`** -- Sharpe {R['hedge_sharpe']:+.2f}, max DD "
            f"{R['hedge_dd']:.1f}%, hit rate {R['hedge_hit']:.1f}%. The direction of "
            "the signal would require shorting high-quality growth stocks, which is "
            "expensive and fundamentally unattractive.\n"
            "- **Survivorship `Named`** -- the survivor bias should inflate the long-leg "
            "estimate; it does not rescue the signal direction. The inversion is robust "
            "to this known bias."
        ),

        # ---- BEAT 6 -- TRADABILITY -------------------------------------------
        md(
            "## 6 -- Could you trade it?\n\n"
            "The nominal trade from the data -- short high-G, long low-G -- is not "
            "actionable:\n\n"
            "1. **|t| below 2.** The negative signal is not statistically "
            "distinguishable from noise. Confidence intervals span zero.\n"
            "2. **Wrong direction economically.** Shorting the most profitable, most "
            "stable large-cap growth firms is a structural risk -- these firms have "
            "strong balance sheets and tend to be expensive to borrow.\n"
            "3. **Universe problem.** The published effect is specific to the small/mid-cap "
            "growth universe. S&P 500 inclusion already screens for the quality "
            "characteristics that the G-score is designed to find."
        ),
        code(
            "# Show that turnover is manageable (annual sort, ~50% turnover on top quintile)\n"
            "sig, fwd, _ = data.synthetic_panel(n_firms=200, n_years=18,\n"
            "                                    g_premium=0.06, seed=232)\n"
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
            "ax.set_title('Annual G-score turnover is modest -- but the signal is inverted')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Mean top-quintile turnover: {np.mean(turnovers)*100:.1f}%/yr')\n"
            "print('Note: low turnover is a cost advantage, but irrelevant when the signal inverts.')"
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Broader universe.** Mohanram (2005) used all COMPUSTAT firms; the effect "
            "was strongest among small/mid-cap growth names with high analyst coverage "
            "and more room for mispricing. A non-survivorship-biased Russell 3000 pull "
            "would materially change the inference.\n"
            "- **Full G6/G7.** Fetching `ResearchAndDevelopmentExpense` and "
            "`AdvertisingExpense` from data.sec.gov would complete the original "
            "specification and remove the substitution caveat.\n"
            "- **Filter to low book-to-market.** Mohanram specifically targeted *growth* "
            "(low BM) stocks. Applying the G-score only to the low-BM quintile may "
            "restore the original effect within its intended domain.\n"
            "- **[Study 65 -- Scorecard](../../65-scorecard/)**: Piotroski's F-score -- "
            "the value-stock counterpart; same EDGAR infrastructure, same infrastructure.\n"
            "- **[Study 122 -- Gross-Profitability](../../122-gross-profitability/)**: "
            "Novy-Marx GP/A -- a single-signal quality factor with a similar conclusion "
            "on large-caps: Weak/Fragile.\n\n"
            "*Think the G-score works for growth stocks in small/mid-cap? Fork this, "
            "restrict to low-BM names, add R&D, broaden the universe, and show t > 2. "
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
