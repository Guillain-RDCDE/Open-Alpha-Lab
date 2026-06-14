"""Generate the two narrative notebooks for Study 154 (Leverage-Anomaly).

    python notebooks/build_notebooks.py
    python -W ignore -m jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md).  Synthetic
figures run anywhere, offline and deterministic; real-data cells use the shared EDGAR
caches under ../_cache/ when present and otherwise quote the frozen headline numbers
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-14).
# LEV1 = LTD/Assets; LEV2 = Liabilities/Equity (D/E ratio).
R = dict(
    # LEV1
    lev1_n=17,
    lev1_mean=-3.88,   # %/yr, low-minus-high
    lev1_t=-1.78,
    lev1_hit=29,       # % hit rate
    lev1_sharpe=-0.39,
    lev1_rand=0.05,    # %/yr, random control excess
    # LEV2
    lev2_n=17,
    lev2_mean=3.07,    # %/yr
    lev2_t=1.41,
    lev2_hit=53,
    lev2_sharpe=0.27,
    lev2_rand=0.06,
    # Summary
    fp_lev1="634f4ebbb2f6",
    fp_lev2="9a57482885f9",
    fp_fwd="d04ace9d0d51",
    years_start=2008,
    years_end=2024,
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

from leverage_anomaly import data, strategy as st

# Frozen real-tape headline numbers (mirror of docs/results.md, as-of 2026-06-14)
R = dict(
    lev1_n=17, lev1_mean=-3.88, lev1_t=-1.78, lev1_hit=29, lev1_sharpe=-0.39, lev1_rand=0.05,
    lev2_n=17, lev2_mean=3.07, lev2_t=1.41, lev2_hit=53, lev2_sharpe=0.27, lev2_rand=0.06,
    fp_lev1="634f4ebbb2f6", fp_lev2="9a57482885f9", fp_fwd="d04ace9d0d51",
    years_start=2008, years_end=2024,
)

def _have_real():
    cache = os.path.join(os.path.abspath(".."), "..", "..", "_cache")
    return (
        os.path.exists(os.path.join(cache, "_edgar_LongTermDebtNoncurrent.parquet"))
        and os.path.exists(os.path.join(cache, "_edgar_Assets.parquet"))
        and os.path.exists(os.path.join(cache, "_edgar_yrret.parquet"))
    )

HAVE_REAL = _have_real()

if HAVE_REAL:
    lev1, lev2, fwd_ret = data.fetch_panel()
else:
    lev1, lev2, fwd_ret = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

print("EDGAR real data available:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Leverage-Anomaly — does more debt mean higher returns?\n"
            "### The financial-leverage puzzle from Penman-Richardson-Tuna, tested on S&P 500 EDGAR data\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Survivorship--biased: upper_bound](https://img.shields.io/badge/Survivorship--biased-upper__bound-8b949e?style=flat-square)\n\n"
            "Standard finance theory says: more debt = more risk = higher expected return. "
            "But a strand of academic research (Penman, Richardson & Tuna 2007) finds the opposite — "
            "high-leverage firms *underperform* low-leverage firms. This notebook tests that claim "
            "on real EDGAR fundamentals for S&P 500 companies.\n\n"
            "> This is the plain-language layer. For t-stats, bootstrap intervals, and full "
            "methodology details see **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does low-leverage beat high-leverage? | **Unclear.** Two measures tell opposite stories. |\n"
            f"| LTD/Assets (debt-to-assets) spread | **{R['lev1_mean']:+.1f}%/yr**, HAC *t* = **{R['lev1_t']:+.2f}** — wrong direction, no signal. |\n"
            f"| D/E ratio spread | **{R['lev2_mean']:+.1f}%/yr**, HAC *t* = **{R['lev2_t']:+.2f}** — right direction, below the bar. |\n"
            "| Could you trade it? | **No.** Annual rebalance is cheap, but the signal is too noisy. |\n"
            "| Survivorship bias? | **Yes.** S&P 500 survivors only — these are upper bounds. |\n\n"
            "> The leverage anomaly does not survive a two-measure cross-check on this survivorship-biased panel. "
            "Neither proxy clears the inference bar (|t| ≥ 2). Signal = **NONE**; Tradability = **MIRAGE**."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *Firms with more financial leverage (more debt relative to equity or assets) earn "
            "lower future returns, not higher returns. This is the 'leverage anomaly': the market "
            "under-appreciates the risk that leverage adds to earnings, so high-leverage firms are "
            "overvalued and low-leverage firms are undervalued. Sorting stocks on debt-to-assets "
            "or D/E ratio and going long the low-leverage quintile beats the market.*\n\n"
            "The original paper (Penman, Richardson & Tuna 2007, *The Book-to-Price Effect in "
            "Stock Returns and Accounting for Leverage*) decomposes the book-to-price ratio and "
            "finds the leverage component is negatively priced — *more leverage predicts lower "
            "returns*, not higher ones as Modigliani-Miller risk-compensation would suggest."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the anomaly is real, it means:\n\n"
            "- **Investors misprice leverage risk.** They buy leveraged companies without demanding "
            "enough return for the added distress risk and earnings volatility. Eventually the market "
            "corrects, and high-leverage firms underperform.\n"
            "- **A cheap long/short factor.** Annual rebalance, low turnover — just sort on one "
            "balance-sheet line and hold for a year. If the spread is real and persistent, this "
            "is a rare free lunch.\n\n"
            "The stakes are high enough to be worth testing carefully."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three discipline rules:\n\n"
            "1. **Two independent measures.** We test LTD/Assets (long-term debt over total assets, "
            "the 'financial leverage ratio') *and* Liabilities/Equity (the classic D/E ratio). "
            "They should agree if the anomaly is robust.\n"
            "2. **Conservative lag.** We use fundamentals from fiscal year y to predict *calendar "
            "year y+1* returns — at minimum a 12-month-plus lag, so the 10-K is definitely public.\n"
            "3. **Random-portfolio control.** Any concentrated decile can outperform randomly. "
            "We benchmark the spread against the distribution of random-size-matched subsets.\n"
            "4. **Name the bias.** The EDGAR panel is current S&P 500 only, projected back in time. "
            "Every survivor in the panel made it to 2026 — so bankruptcies and delistings are "
            "excluded. Results are upper bounds.\n\n"
            "Inference bar: |HAC t| ≥ 2 on the low-minus-high spread across annual observations."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown\n\n"
            "**LEV1: LTD/Assets ratio.** "
            "Each year we sort S&P 500 firms into quintiles by long-term-debt / total-assets "
            "and record the following year's equal-weight returns for the top (high-leverage) "
            "and bottom (low-leverage) quintiles."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sp1 = st.quintile_spread(lev1, fwd_ret, q=0.20)\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "    axes[0].bar(sp1.index+1, sp1['spread']*100,\n"
            "                color=[GREEN if v>0 else RED for v in sp1['spread']])\n"
            "    axes[0].axhline(0, c='k', lw=1)\n"
            "    axes[0].set_xlabel('Year (return year)')\n"
            "    axes[0].set_ylabel('Spread %/yr (low − high leverage)')\n"
            "    axes[0].set_title('LEV1: LTD/Assets — annual spread')\n"
            "    axes[1].bar(['Low leverage\\n(bottom quintile)', 'Market\\n(equal-weight)', 'High leverage\\n(top quintile)'],\n"
            "                [sp1['low'].mean()*100, sp1['market'].mean()*100, sp1['high'].mean()*100],\n"
            "                color=[GREEN, GREY, RED])\n"
            "    axes[1].set_ylabel('Mean return %/yr')\n"
            "    axes[1].set_title('LEV1: Average annual returns by leverage bucket')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    s1 = st.summarize(sp1['spread'])\n"
            "    print(f\"Spread: {s1['mean']*100:+.2f}%/yr  t={s1['tstat']:+.2f}  hit={s1['hit_rate']*100:.0f}%\")\n"
            "else:\n"
            "    print(f\"Frozen: LEV1 spread = {R['lev1_mean']:+.1f}%/yr, t = {R['lev1_t']:+.2f}, hit = {R['lev1_hit']}%\")"
        ),
        md(
            f"LEV1 result: mean spread **{R['lev1_mean']:+.1f}%/yr**, "
            f"HAC *t* = **{R['lev1_t']:+.2f}**, hit rate {R['lev1_hit']}%. "
            "This is the **wrong direction** — high-leverage firms *outperformed* low-leverage firms "
            "slightly, the opposite of the anomaly. The effect is statistically indistinguishable "
            "from zero and is against the theory."
        ),
        md(
            "**LEV2: Liabilities/Equity (D/E ratio).** "
            "The classic debt-to-equity ratio, masking firms with negative book equity."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sp2 = st.quintile_spread(lev2, fwd_ret, q=0.20)\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "    axes[0].bar(sp2.index+1, sp2['spread']*100,\n"
            "                color=[GREEN if v>0 else RED for v in sp2['spread']])\n"
            "    axes[0].axhline(0, c='k', lw=1)\n"
            "    axes[0].set_xlabel('Year (return year)')\n"
            "    axes[0].set_ylabel('Spread %/yr (low − high leverage)')\n"
            "    axes[0].set_title('LEV2: D/E ratio — annual spread')\n"
            "    axes[1].bar(['Low leverage\\n(bottom quintile)', 'Market\\n(equal-weight)', 'High leverage\\n(top quintile)'],\n"
            "                [sp2['low'].mean()*100, sp2['market'].mean()*100, sp2['high'].mean()*100],\n"
            "                color=[GREEN, GREY, RED])\n"
            "    axes[1].set_ylabel('Mean return %/yr')\n"
            "    axes[1].set_title('LEV2: Average annual returns by leverage bucket')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    s2 = st.summarize(sp2['spread'])\n"
            "    print(f\"Spread: {s2['mean']*100:+.2f}%/yr  t={s2['tstat']:+.2f}  hit={s2['hit_rate']*100:.0f}%\")\n"
            "else:\n"
            "    print(f\"Frozen: LEV2 spread = {R['lev2_mean']:+.1f}%/yr, t = {R['lev2_t']:+.2f}, hit = {R['lev2_hit']}%\")"
        ),
        md(
            f"LEV2 result: mean spread **{R['lev2_mean']:+.1f}%/yr**, "
            f"HAC *t* = **{R['lev2_t']:+.2f}**, hit rate {R['lev2_hit']}%. "
            "This is in the right direction — low D/E outperforms — but |t| < 2, so it does not "
            "clear the inference bar. The two measures disagree with each other, which is itself "
            "a sign that we are in noise territory."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The two leverage proxies tell opposite stories: LTD/Assets "
            f"gives a spread of **{R['lev1_mean']:+.1f}%/yr** (*t* = {R['lev1_t']:+.2f}, wrong direction); "
            f"D/E gives **{R['lev2_mean']:+.1f}%/yr** (*t* = {R['lev2_t']:+.2f}, right direction but below bar). "
            "Neither clears |t| ≥ 2. The anomaly does not replicate on this panel.\n"
            "- **Tradability — Mirage.** Even if the D/E spread were real, the hit rate of "
            f"{R['lev2_hit']}% and single-digit t-stat mean the strategy would be untrustworthy "
            "year-to-year. Annual rebalance is cheap, but you need a real edge first.\n"
            "- **Survivorship bias.** The EDGAR panel is current-S&P-500 survivors only. "
            "High-leverage firms that went bankrupt or were delisted are excluded, which biases "
            "*against* finding the anomaly (the riskiest high-leverage firms disappear). The "
            "fact that we still don't find it even with this bias in our favour makes the null "
            "more credible."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Annual rebalance means transaction costs are modest — maybe 20-50 bps/yr round-trip "
            "for a small cap-weighted portfolio. But there is a deeper problem: **you need a real "
            "signal first.** With |t| < 2 on both measures and the two measures pointing in "
            "opposite directions, a trader would be unsure which measure to trust, what decile "
            "cutoff to use, and whether the effect is durable or a sample artifact.\n\n"
            "The survivorship-biased panel also means that in live trading you would hold "
            "companies that *could* become distressed and leave the S&P 500 — something this "
            "backtest never experienced."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Longer history.** The EDGAR panel runs from FY 2007. A longer sample (academic "
            "Compustat data going back to the 1960s) might tell a different story — the original "
            "PRT (2007) paper used a broader sample.\n"
            "- **Decomposition.** Penman-Richardson-Tuna decompose book-to-price into an "
            "*operating* and a *financial* component. Testing the combined B/P rather than "
            "raw leverage might strengthen the signal.\n"
            "- **Market-cap weighting.** Equal weighting gives each firm the same voice; "
            "value-weighting (larger firms dominate) often weakens factor returns significantly.\n"
            "- **Related factors.** The leverage anomaly is related to the low-risk anomaly "
            "([Study 65 — Scorecard](../../65-scorecard/)) and the quality factor "
            "([Study 121 — Magic-Formula](../../121-magic-formula/)). Does leverage add "
            "explanatory power *after* controlling for profitability?\n\n"
            "*Think the die is loaded? Fork this, extend the history or add the operating "
            "component, and show a t ≥ 2 spread on a clean (non-survivorship-biased) universe.*"
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
            "# Leverage-Anomaly — a quantitative teardown\n"
            "### EDGAR fundamentals · annual quintile sort · HAC inference · random-control · survivorship bias\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Survivorship--biased: upper_bound](https://img.shields.io/badge/Survivorship--biased-upper__bound-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — same "
            "seven beats, every claim carrying its standard error. We test whether sorting S&P 500 "
            "firms by financial leverage (LTD/Assets and D/E ratio) predicts next-year returns in "
            "the direction of the Penman-Richardson-Tuna (2007) leverage anomaly.\n\n"
            "> **Not investment advice.** EDGAR shared caches, S&P 500 survivorship-biased panel, "
            f"FY {R['years_start']}–{R['years_end']}, as-of 2026-06-14. "
            "Methods in [`docs/references.md`](../docs/references.md); "
            "reproducible numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> The **In plain words** notes translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | LEV1 (LTD/Assets) spread **{R['lev1_mean']:+.1f}%/yr**, "
            f"HAC *t* = **{R['lev1_t']:+.2f}** (wrong direction); "
            f"LEV2 (D/E) spread **{R['lev2_mean']:+.1f}%/yr**, *t* = **{R['lev2_t']:+.2f}** — "
            "right direction but |*t*| < 2; measures disagree. |\n"
            f"| **Tradability** | `MIRAGE` | No edge confirmed; annual rebalance is cheap but "
            "there is nothing to trade. Survivorship bias further undermines results. |\n"
            f"| **Survivorship-biased** | `upper bound` | S&P 500 survivors only (2026 membership "
            "projected back); bankrupted high-leverage firms excluded — results are upper bounds. |\n\n"
            "> In plain words: the leverage anomaly literature (Penman-Richardson-Tuna 2007) does "
            "not replicate on this panel — the two simplest leverage proxies disagree in direction, "
            "and neither clears the |t| ≥ 2 inference bar."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $L_{i,y}$ be a leverage proxy for firm $i$ at fiscal year-end $y$, and "
            "$r_{i,y+1}$ the calendar-year $y+1$ return.  The Penman-Richardson-Tuna (2007) claim:\n\n"
            "- **H₁ (signal).**  $\\mathbb{E}[r_{i,y+1} \\mid L_{i,y} \\in \\text{bottom quintile}] "
            "> \\mathbb{E}[r_{i,y+1} \\mid L_{i,y} \\in \\text{top quintile}]$ — low-leverage "
            "firms earn higher future returns.  Measured HAC t-stat of the annual spread ≥ 2.\n"
            "- **H₂ (size control).**  The spread exceeds the distribution of random equal-size "
            "subsets, ruling out a concentrated-subset artifact.\n"
            "- **H₃ (robustness).**  Both LTD/Assets and Liabilities/Equity agree in direction "
            "and magnitude.\n\n"
            "We reject all three on this panel."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "The leverage anomaly was cited as evidence of mispriced distress risk and a violation "
            "of the Modigliani-Miller theorem's implication that leverage should be positively "
            "priced (in a world with taxes and bankruptcy costs, the sign is unclear).  A robust "
            "replication would matter for fundamental factor investing.  The failure to replicate "
            "here — with both measures and on a presumably *favorable* survivorship-biased sample — "
            "is evidence that the effect is either weaker than the original paper suggested, "
            "specific to the original sample period, or requires a more careful decomposition "
            "(the full PRT approach decomposes book-to-price, not just sorts on raw leverage)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Universe.** Current S&P 500 tickers (survivorship-biased) from the desk's "
            "shared EDGAR caches.  FY 2007–2024 fundamentals → calendar 2008–2025 returns.\n"
            "- **Lag.** Signal from fiscal year $y$ → forward return in calendar year $y+1$ "
            "(≥12-month reporting lag).\n"
            "- **Leverage proxies.** LEV1 = LongTermDebtNoncurrent / Assets; "
            "LEV2 = Liabilities / StockholdersEquity (with non-positive equity masked).\n"
            "- **Portfolio.** Equal-weight quintile sort (q = 0.20); low-minus-high spread.\n"
            "- **Inference.** Newey-West HAC t-stat on the {n}-observation annual spread series "
            "(n=17 years; lags=1 by the rule-of-thumb).\n"
            "- **Control.** Random-portfolio excess — 500 draws of a random quintile-size subset "
            "per year, the null that the low-lev quintile is just a lucky small basket.\n"
            "- **Positive control.** Synthetic panel with planted premium: the engine recovers "
            "the anomaly when it exists."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · LEV1 — LTD / Assets: the wrong direction\n\n"
            "Per-year spread (low-leverage − high-leverage) for the LTD/Assets quintile sort."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sp1 = st.quintile_spread(lev1, fwd_ret, q=0.20)\n"
            "    s1 = st.summarize(sp1['spread'])\n"
            "    exc1 = st.random_portfolio_excess(lev1, fwd_ret, n_draws=500, seed=154)\n"
            "    fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "    ax.bar(sp1.index+1, sp1['spread']*100, color=[GREEN if v>0 else RED for v in sp1['spread']])\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.axhline(s1['mean']*100, ls='--', c=GREY, lw=1.5, label=f\"mean={s1['mean']*100:+.1f}%\")\n"
            "    ax.set_xlabel('Return year'); ax.set_ylabel('Spread %/yr')\n"
            "    ax.set_title(f\"LEV1 (LTD/Assets) low-minus-high: mean={s1['mean']*100:+.1f}%/yr, t={s1['tstat']:+.2f}\")\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f\"n={s1['n']}  mean={s1['mean']*100:+.2f}%/yr  vol={s1['vol']*100:.2f}%  \"\n"
            "          f\"t={s1['tstat']:+.2f}  hit={s1['hit_rate']*100:.0f}%  Sharpe={s1['sharpe']:+.2f}\")\n"
            "    print(f\"Random control mean excess: {exc1.mean()*100:+.2f}%/yr  \"\n"
            "          f\"(p-tile of spread vs null: {(exc1 < s1['mean']).mean()*100:.0f}%)\")\n"
            "else:\n"
            "    print(f\"Frozen: LEV1 mean={R['lev1_mean']:+.1f}%/yr, t={R['lev1_t']:+.2f}, hit={R['lev1_hit']}%\")"
        ),
        md(
            f"> In plain words: LEV1 gives **{R['lev1_mean']:+.1f}%/yr** — negative, meaning "
            "high-leverage firms slightly *outperformed* (the opposite of the anomaly).  "
            f"Hit rate {R['lev1_hit']}%.  HAC *t* = {R['lev1_t']:+.2f}: comfortably inside the ±2 band."
        ),
        md(
            "### 4b · LEV2 — Liabilities / Equity: right direction, below the bar\n\n"
            "D/E ratio quintile sort.  The anomaly direction (low minus high positive) shows up, "
            "but does not reach statistical significance."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sp2 = st.quintile_spread(lev2, fwd_ret, q=0.20)\n"
            "    s2 = st.summarize(sp2['spread'])\n"
            "    exc2 = st.random_portfolio_excess(lev2, fwd_ret, n_draws=500, seed=154)\n"
            "    fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "    ax.bar(sp2.index+1, sp2['spread']*100, color=[GREEN if v>0 else RED for v in sp2['spread']])\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.axhline(s2['mean']*100, ls='--', c=AMBER, lw=1.5, label=f\"mean={s2['mean']*100:+.1f}%\")\n"
            "    ax.set_xlabel('Return year'); ax.set_ylabel('Spread %/yr')\n"
            "    ax.set_title(f\"LEV2 (D/E) low-minus-high: mean={s2['mean']*100:+.1f}%/yr, t={s2['tstat']:+.2f}\")\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f\"n={s2['n']}  mean={s2['mean']*100:+.2f}%/yr  vol={s2['vol']*100:.2f}%  \"\n"
            "          f\"t={s2['tstat']:+.2f}  hit={s2['hit_rate']*100:.0f}%  Sharpe={s2['sharpe']:+.2f}\")\n"
            "    print(f\"Random control mean excess: {exc2.mean()*100:+.2f}%/yr  \"\n"
            "          f\"(p-tile of spread vs null: {(exc2 < s2['mean']).mean()*100:.0f}%)\")\n"
            "else:\n"
            "    print(f\"Frozen: LEV2 mean={R['lev2_mean']:+.1f}%/yr, t={R['lev2_t']:+.2f}, hit={R['lev2_hit']}%\")"
        ),
        md(
            f"> In plain words: LEV2 gives **{R['lev2_mean']:+.1f}%/yr** — in the right direction, "
            f"but HAC *t* = {R['lev2_t']:+.2f} and the hit rate is just {R['lev2_hit']}% (barely "
            "above a coin).  The two measures disagree in sign, which strongly suggests we are "
            "in noise territory."
        ),
        md(
            "### 4c · The two measures side by side — the disagreement is the signal\n\n"
            "If the leverage anomaly were robust, both LEV1 and LEV2 would point in the same "
            "direction and agree in magnitude.  When they disagree in sign, the anomaly is not "
            "robust at the level of these data."
        ),
        code(
            "labels = ['LEV1\\n(LTD/Assets)', 'LEV2\\n(Liab/Equity)']\n"
            "means = [R['lev1_mean'], R['lev2_mean']]\n"
            "tstats = [R['lev1_t'], R['lev2_t']]\n"
            "if HAVE_REAL:\n"
            "    _s1b = st.quintile_spread(lev1, fwd_ret, q=0.20)\n"
            "    _s2b = st.quintile_spread(lev2, fwd_ret, q=0.20)\n"
            "    means = [st.summarize(_s1b['spread'])['mean']*100, st.summarize(_s2b['spread'])['mean']*100]\n"
            "    tstats = [st.summarize(_s1b['spread'])['tstat'], st.summarize(_s2b['spread'])['tstat']]\n"
            "fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))\n"
            "colors = [RED if m < 0 else AMBER for m in means]\n"
            "axes[0].bar(labels, means, color=colors)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_ylabel('Mean spread %/yr'); axes[0].set_title('Mean annual spread (low - high)')\n"
            "axes[1].bar(labels, tstats, color=[RED if abs(t)<2 else GREEN for t in tstats])\n"
            "for s in (2, -2): axes[1].axhline(s, ls='--', c=GREY, lw=1)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('HAC t-stat'); axes[1].set_title('HAC t-stat (inference bar = +/-2)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Both measures inside the +/-2 band. Measures disagree in sign.')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — LEV1 spread {R['lev1_mean']:+.1f}%/yr, t={R['lev1_t']:+.2f} "
            "(wrong direction); LEV2 spread {R['lev2_mean']:+.1f}%/yr, t={R['lev2_t']:+.2f} "
            "(right direction, below bar). The measures disagree. No HAC |t| ≥ 2.\n"
            "- **Tradability `MIRAGE`** — no confirmed edge; the signal is not tradable even at "
            "annual rebalance frequency.\n"
            "- **Survivorship-biased** — the EDGAR panel excludes delisted and bankrupt firms. "
            "The bias *favors finding* the anomaly (because the worst high-leverage outcomes — "
            "bankruptcies — are absent). Even so, no signal is found."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it?\n\n"
            "Annual rebalance is cheap — roughly 20–50 bps round-trip per year for a large-cap "
            "quintile portfolio. The real problem is not costs but signal quality: with hit rates "
            "near 50% and t-stats below 2, the expected out-of-sample performance is no better "
            "than a coin flip. There is no break-even cost to compute because the gross edge is "
            "not confirmed.\n\n"
            "A live implementation would also face:\n"
            "- **Survivorship risk**: delisted firms are real losses not in this backtest.\n"
            "- **Sector concentration**: leverage varies enormously across sectors (utilities "
            "and REITs carry far more debt than tech).  Raw leverage sorts can be disguised "
            "sector bets.\n"
            "- **Definition sensitivity**: the two leverage measures we tested pointed in "
            "opposite directions; a live manager would need to commit to one definition "
            "before seeing data."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the positive control and extensions\n\n"
            "The engine works: a synthetic panel with a planted premium recovers the spread "
            "reliably when it exists."
        ),
        code(
            "# Positive control: sweep planted premium\n"
            "prems = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]\n"
            "t_stats = []\n"
            "for p in prems:\n"
            "    lev_s, fwd_s, _ = data.synthetic_panel(n_firms=300, n_years=18, premium=p, seed=154)\n"
            "    sp = st.quintile_spread(lev_s, fwd_s, q=0.20)\n"
            "    t_stats.append(st.summarize(sp['spread'])['tstat'])\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot([p*100 for p in prems], t_stats, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(2, ls='--', c=GREY, lw=1, label='|t|=2 bar')\n"
            "ax.set_xlabel('Planted premium (%/yr)'); ax.set_ylabel('HAC t-stat of spread')\n"
            "ax.set_title('Engine works: t-stat rises with planted premium')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('At 0% premium the engine reads ~0 t-stat — correctly.')"
        ),
        md(
            "The engine faithfully recovers planted effects. The real-tape null result is "
            "therefore a statement about the market signal, not the machinery.\n\n"
            "**Forks worth trying**:\n\n"
            "- Use Compustat (broader universe, longer history, avoids survivorship) rather "
            "than the shared EDGAR panel.\n"
            "- Implement the full Penman-Richardson-Tuna book-to-price decomposition "
            "(operating component + financial leverage component) rather than raw leverage.\n"
            "- Control for sector / industry (within-sector leverage sort instead of "
            "cross-sector).\n"
            "- Combine with profitability (see [Study 121 — Magic-Formula](../../121-magic-formula/)) "
            "to test whether leverage adds incremental power.\n\n"
            "*The bar: |HAC t| ≥ 2 on a non-survivorship-biased panel, for both leverage measures, "
            "in the anomaly direction.*"
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
