"""Generate the two narrative notebooks for Study 535 (Mispricing-Score).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats. The synthetic positive control runs
anywhere, offline and deterministic; the real-panel cells use the cached
yfinance parquets under ../_cache/ if present and otherwise quote the frozen
headline numbers in ``R`` (mirror of docs/results.md, as-of 2026-06-27).
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


# Frozen real-panel headline numbers — mirror of docs/results.md (as-of 2026-06-27).
R = dict(
    n_months=178, year_start=2011, year_end=2025, n_tickers=45,
    fp_prices="b6d4a14d508b", fp_spy="5bb6e49fae4d",
    ls_gross=-7.68, ls_gross_t=-1.843, ls_gross_sharpe=-0.473, ls_gross_vol=16.25,
    ls_hit=43.3, ls_dd=-80.3, turnover=44.8,
    ls_net=-9.50, ls_net_t=-2.285, ls_net_sharpe=-0.585,
    long_mean=15.45, long_t=4.26, short_names=23.13, short_book=-23.13, short_book_t=-5.04,
    placebo_obs=-7.68, placebo_null_mean=-0.13, placebo_null_sd=3.05, placebo_p=0.007,
    cut_tercile=-4.69, cut_tercile_t=-1.49,
    cut_quintile=-7.68, cut_quintile_t=-1.84,
    cut_decile=-11.66, cut_decile_t=-2.08,
    ctrl={-0.20: -5.84, -0.10: -2.11, 0.0: -0.03, 0.10: 2.73, 0.20: 4.42, 0.30: 4.33},
    annual={
        2011: 9.1, 2012: 11.3, 2013: -14.3, 2014: -12.1, 2015: -22.9,
        2016: -3.6, 2017: -14.4, 2018: -6.8, 2019: -12.8, 2020: -24.2,
        2021: 3.6, 2022: 28.6, 2023: -9.9, 2024: -22.5, 2025: -21.6,
    },
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from mispricing_score import data, strategy as st

def _have_cache():
    study = os.path.abspath("..")
    return (os.path.exists(os.path.join(study, "_cache", "msp_prices.parquet"))
            and os.path.exists(os.path.join(study, "_cache", "msp_spy.parquet")))

HAVE_REAL = _have_cache()
print("yfinance cache present:", HAVE_REAL)

if HAVE_REAL:
    prices, spy = data.fetch_panel()
    result = st.long_short(prices)
    s_ls   = st.summary(result["ls_ret"])
    s_long = st.summary(result["long_ret"])
    s_sb   = st.summary(-result["short_ret"])
    net    = st.apply_costs(result); s_net = st.summary(net)
    print(f"N months: {s_ls['n']} | L-S gross: {s_ls['mean']*100:+.2f}%/yr"
          f" | HAC t: {s_ls['tstat']:+.3f}")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Mispricing-Score — does averaging eleven anomalies beat any one?\n"
            "### Stambaugh-Yu-Yuan (2015): the composite mispricing measure, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Short--leg edge%3F: Busted](https://img.shields.io/badge/Short--leg%20edge%3F-Busted-8b949e?style=flat-square)\n\n"
            "In 2015, Robert Stambaugh, Jianfeng Yu and Yu Yuan published a tidy idea: take "
            "eleven well-known stock-market anomalies, **average their rankings** into a single "
            "'mispricing score', and sort on it. The composite, they showed, predicts returns "
            "**better than any single anomaly** — and the edge lives almost entirely in the "
            "**short leg** (the overpriced names), because short-sale frictions let overpricing "
            "persist longer than underpricing.\n\n"
            "We test the composite on a 45-name large-cap S&P 500 **survivor** basket, building "
            "the six anomalies you can compute from price alone, and we name the survivorship "
            "bias up front — because here it is the whole story.\n\n"
            "> **This is the plain-language layer.** Want the rank construction, the placebo "
            "p-value and the leg split with standard errors? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the composite score sort the cross-section? | **Yes — really.** A "
            f"label-shuffle placebo puts the spread far in the tail (**p = {R['placebo_p']:.3f}**). |\n"
            "| Does the long-short make money? | **No — it loses.** "
            f"**{R['ls_gross']:.2f}%/yr** gross, **{R['ls_net']:.2f}%/yr** net of costs. |\n"
            "| Is the edge in the short leg, as SYY say? | **The opposite.** Shorting the "
            f"'overpriced' names loses **{R['short_book']:.1f}%/yr** — they are the surviving "
            "mega-cap winners. |\n"
            "| Is there a survivorship problem? | **Yes, and it is the entire result.** The "
            "failed overpriced names that would make the short leg pay have delisted. |\n\n"
            "> The score genuinely *orders* stocks — but on a survivor basket it orders them the "
            "**wrong way** for a long-short. The only honest residue is the long (cheap) leg as a "
            f"long-only tilt: **+{R['long_mean']:.1f}%/yr** (*t* = +{R['long_t']:.2f})."
        ),

        md(
            "## 1 — The claim\n\n"
            "> *\"Combining anomalies into a single mispricing measure produces a composite that "
            "predicts the cross-section of returns more reliably than any individual anomaly, "
            "with most of the predictability coming from the short (overpriced) side.\"*\n\n"
            "— Stambaugh, Yu & Yuan (2015), *Journal of Finance*\n\n"
            "The intuition: each anomaly is a noisy proxy for the same latent thing — mispricing. "
            "Average eleven noisy proxies and the noise diversifies away, leaving a cleaner "
            "signal. And because optimists can buy freely but pessimists must pay to short, "
            "**overpricing** is the part arbitrage leaves on the table."
        ),

        md(
            "## 2 — So what?\n\n"
            "If true, the composite is a one-number quality/cheapness gauge that dominates the "
            "factor zoo, and the money is in shorting glamour. Every multi-factor 'quality' fund "
            "is implicitly making this bet. The questions for us:\n\n"
            "1. **Does the composite sort at all on a realistic large-cap panel?**\n"
            "2. **Does the long-short actually pay — and in which leg?**\n"
            "3. **What does survivorship do to the short leg?**"
        ),

        md(
            "## 3 — How would we even know?\n\n"
            "Three disciplines:\n\n"
            "1. **Build the score on past data only.** Every anomaly uses a trailing window; the "
            "portfolio enters the close **one day after** the signal is public — one execution "
            "lag, no look-ahead.\n"
            "2. **Shuffle the labels.** If a random permutation of the score sorts as well as the "
            "real score, there is no signal. (Spoiler: it doesn't — the real score sorts.)\n"
            "3. **Name the survivorship bias.** The universe is the *current* S&P 500 projected "
            "backwards. The worst overpriced names — the engine of the SYY short leg — have "
            "delisted and are gone."
        ),

        md(
            "## 4 — The teardown\n\n"
            "**First, does the composite engine work when a mispricing premium is planted?**"
        ),
        code(
            "# Synthetic positive control: plant a state-dependent mispricing premium\n"
            "# (overpriced = high-vol names that then underperform) and sweep it.\n"
            "premiums = [-0.20, -0.10, 0.0, 0.10, 0.20, 0.30]\n"
            "ctrl_means = []\n"
            "for prem in premiums:\n"
            "    p_s, _spy, _t = data.synthetic_panel(n_stocks=45, n_days=2200,\n"
            "                                          mispricing_premium=prem, seed=0)\n"
            "    res_s = st.long_short(p_s, min_stocks=10, quantile=0.20)\n"
            "    ctrl_means.append(st.summary(res_s['ls_ret'])['mean']*100 if not res_s.empty else np.nan)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if m > 0.5 else (RED if m < -0.5 else GREY) for m in ctrl_means]\n"
            "ax.bar([str(p) for p in premiums], ctrl_means, color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted mispricing premium'); ax.set_ylabel('composite L-S mean (%/yr)')\n"
            "ax.set_title('Synthetic control: engine recovers a planted premium (monotone, ~0 at null)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('seed-0 sweep:', [round(m,2) for m in ctrl_means])\n"
            "print('15-seed averaged (frozen):', R_CTRL)"
        ),
        md(
            "The engine is faithful: the long-short mean rises monotonically with the planted "
            "premium and is ~0 at the null. So whatever happens on the real tape reflects "
            "**the market**, not a broken detector."
        ),
        md("**Now the honest test on the real yfinance survivor panel.**"),
        code(
            "if HAVE_REAL:\n"
            "    ls_g = s_ls['mean']*100; ls_t = s_ls['tstat']\n"
            "    long_m = s_long['mean']*100; sb_m = s_sb['mean']*100\n"
            "    short_names_m = st.summary(result['short_ret'])['mean']*100\n"
            "else:\n"
            "    ls_g, ls_t = R['ls_gross'], R['ls_gross_t']\n"
            "    long_m, sb_m, short_names_m = R['long_mean'], R['short_book'], R['short_names']\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))\n"
            "labels = ['Long\\n(cheap quintile)', '\"Overpriced\"\\nnames (owned)',\n"
            "          'Short BOOK\\n(short them)']\n"
            "vals = [long_m, short_names_m, sb_m]\n"
            "cols = [GREEN, GREY, RED]\n"
            "axes[0].bar(labels, vals, color=cols, width=0.6)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_ylabel('mean annual return (%/yr)')\n"
            "axes[0].set_title('The short leg is the WORST place to be short')\n"
            "\n"
            "yrs = list(R['annual'].keys()); av = list(R['annual'].values())\n"
            "if HAVE_REAL:\n"
            "    rc = result.copy(); rc['year'] = rc.index.year\n"
            "    a = rc.groupby('year')['ls_ret'].apply(lambda x: float((1+x).prod()-1))*100\n"
            "    yrs, av = list(a.index), list(a.values)\n"
            "axes[1].bar(yrs, av, color=[GREEN if v > 0 else RED for v in av])\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_xlabel('year'); axes[1].set_ylabel('composite L-S (%/yr)')\n"
            "axes[1].set_title(f'Year-by-year | mean {ls_g:+.1f}%  t={ls_t:+.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Composite L-S: {ls_g:+.2f}%/yr | HAC t = {ls_t:+.3f}')"
        ),
        md(
            f"The composite long-short **loses {R['ls_gross']:.2f}%/yr** (HAC *t* = "
            f"{R['ls_gross_t']:.2f}). The 'overpriced' short leg returned **+{R['short_names']:.1f}%/yr** "
            "— the *best* basket to own — because on survivors the high-momentum, high-vol, "
            "near-52-week-high names are the NVDA/AAPL cohort that kept winning. The one good year "
            f"({R['annual'][2022]:+.1f}% in 2022) is the single moment glamour cracked."
        ),

        md(
            "## 5 — The verdict\n\n"
            f"- **Signal — NONE.** The score *sorts* (placebo p = {R['placebo_p']:.3f}), but the "
            f"long-short earns no positive *t* ≥ 2 — the only significant *t* is **negative** "
            f"({R['ls_gross_t']:.2f} gross, {R['ls_net_t']:.2f} net). No real positive edge here.\n"
            f"- **Tradability — MIRAGE.** Loses gross ({R['ls_gross']:.2f}%/yr) and net "
            f"({R['ls_net']:.2f}%/yr), Sharpe {R['ls_gross_sharpe']:.2f}, max drawdown {R['ls_dd']:.0f}%.\n"
            "- **Short-leg edge — BUSTED.** On survivors, shorting the 'overpriced' names loses "
            f"{R['short_book']:.1f}%/yr (*t* = {R['short_book_t']:.2f}). The delisted overpriced "
            "names that would make it pay are absent — the exact bias a clean SYY test must avoid."
        ),

        md(
            "## 6 — Could you actually trade it?\n\n"
            "Not as a long-short on this universe. The obstacles:\n\n"
            "1. **Survivorship inverts the short leg.** The whole SYY premium is short-side, and "
            "the short side is exactly what a survivor basket destroys.\n"
            "2. **Costs deepen the loss.** Net is *more* negative than gross — there is no edge "
            "for costs to erode, only a loss to amplify.\n"
            "3. **−80% drawdown.** Shorting the decade's biggest winners is a recipe for ruin.\n\n"
            "The only salvageable piece is the **long (cheap) leg as a long-only tilt** "
            f"(+{R['long_mean']:.1f}%/yr, *t* +{R['long_t']:.2f}) — and even that is dominated by "
            "the survivor mega-cap rally, not by cheapness."
        ),

        md(
            "## 7 — Going further\n\n"
            "- **A real, delisting-complete universe.** The SYY result needs the failed overpriced "
            "names. A CRSP/Compustat pull including delistings would be the honest test.\n"
            "- **Add the fundamental anomalies.** Accruals, net issuance, profitability and "
            "distress need clean per-name financials — see "
            "[231 Sloan-Accruals](../../231-sloan-accruals/) and "
            "[244 Asset-Growth](../../244-asset-growth/) for two of the eleven, stand-alone.\n"
            "- **The members it claims to beat:** "
            "[238 Betting-Against-Beta](../../238-betting-against-beta/), "
            "[330 Low-Volatility](../../330-low-volatility-anomaly/).\n\n"
            "*Think the SYY composite survives once you add back the delisted names and the "
            "fundamental anomalies? Fork this, pull a delisting-complete universe, and show a "
            "positive short-leg *t* > 2 net of borrow. That is the bar.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _inject_ctrl(nb)
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Mispricing-Score — a quantitative teardown\n"
            "### composite anomaly ranks · placebo null · leg split · HAC inference\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Short--leg edge%3F: Busted](https://img.shields.io/badge/Short--leg%20edge%3F-Busted-8b949e?style=flat-square)\n\n"
            "The quantitative companion to the [notebook for the curious](01_for_the_curious.ipynb) "
            "— same seven beats, every claim carrying its standard error. We test Stambaugh-Yu-Yuan "
            "(2015): six price-computable anomaly ranks → composite mispricing score → quintile "
            "long-short, monthly, one-day execution lag.\n\n"
            "> **Not investment advice.** Real data: yfinance daily adjusted-close, 45 S&P 500 "
            "large-caps, 2010–2025, as-of 2026-06-27. Methods in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship bias is named:** universe = current S&P 500 projected backwards. The "
            "delisted overpriced names — the engine of the SYY short leg — are absent."
        ),
        code(BOOT),

        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | L-S **{R['ls_gross']:.2f}%/yr** gross, HAC *t* = "
            f"**{R['ls_gross_t']:.3f}**; net **{R['ls_net']:.2f}%/yr**, *t* **{R['ls_net_t']:.3f}** — "
            "the only significant *t* is *against* the strategy. |\n"
            f"| **Tradability** | `MIRAGE` | Sharpe **{R['ls_gross_sharpe']:.3f}**, max DD "
            f"**{R['ls_dd']:.0f}%**, costs deepen the loss. |\n"
            f"| **Short-leg edge?** | `BUSTED` | Short book **{R['short_book']:.1f}%/yr** "
            f"(*t* = {R['short_book_t']:.2f}); the 'overpriced' names are survivor winners. |\n\n"
            f"> The composite *sorts* (placebo p = {R['placebo_p']:.3f}) — but adversely. The long "
            f"(cheap) leg is a clean long-only tilt (+{R['long_mean']:.1f}%/yr, *t* +{R['long_t']:.2f}); "
            "everything else is the survivor mega-cap rally."
        ),

        md(
            "## 1 — The claim, steelmanned\n\n"
            "Let $S_i$ = the average cross-sectional percentile rank of stock $i$ across the "
            "anomalies, oriented so high $S_i$ = more overpriced. Stambaugh-Yu-Yuan (2015) assert:\n\n"
            "- **H1 (signal).** $S_i$ predicts returns: the long-short "
            "$\\bar r^{\\text{low }S} - \\bar r^{\\text{high }S} > 0$.\n"
            "- **H2 (short-leg).** Most of that comes from the short (high-$S$) leg, because "
            "arbitrage asymmetry (Miller 1977) leaves overpricing uncorrected.\n"
            "- **H3 (dominance).** The composite beats any single anomaly.\n\n"
            "On a survivor panel we **reject H1** (negative, insignificant-in-direction), "
            "**reject H2** (the short leg is the worst leg), and cannot test H3 cleanly. The "
            "composite *does* carry information (placebo p = 0.007) — pointed the wrong way."
        ),

        md(
            "## 2 — So what? — the economic stakes\n\n"
            "The SYY composite underpins a generation of 'quality' and 'mispricing' factor "
            "products, most of which lean on the short-glamour side. If that side inverts the "
            "moment you restrict to tradable large-cap survivors, the marketed edge is an "
            "artefact of the delisted tail — not something a long-only or large-cap fund can bank."
        ),

        md(
            "## 3 — The protocol\n\n"
            "- **Signals (six, price-only).** 12-1 momentum, MAX/lottery (top-5 daily returns "
            "last month), 6-month realised vol, 12-month price run-up, distance to 52-week high, "
            "short-term reversal — each oriented high = overpriced.\n"
            "- **Composite.** Cross-sectional percentile rank of each, averaged (SYY combination).\n"
            "- **Sort.** Monthly quintiles; long the bottom (cheap), short the top (overpriced).\n"
            "- **Execution.** Enter the close **one day after** the signal date; hold one month.\n"
            "- **Costs.** 10 bps/side × turnover × 2 legs + 75 bps/yr borrow on the short leg.\n"
            "- **Inference.** Newey-West HAC *t* on monthly returns; a 150-shuffle label "
            "permutation placebo.\n"
            "- **Universe caveat.** 45 large-cap S&P 500 survivors; the short leg is biased."
        ),

        md("## 4 — The teardown"),
        md(
            "### 4a — Positive control: the composite engine is a faithful detector\n\n"
            "Sweep the planted mispricing premium; the long-short mean should be monotone and "
            "~0 at the null."
        ),
        code(
            "premiums = [-0.20, -0.10, 0.0, 0.10, 0.20, 0.30]\n"
            "means = []\n"
            "for prem in premiums:\n"
            "    p_s, _spy, _t = data.synthetic_panel(n_stocks=45, n_days=2200,\n"
            "                                          mispricing_premium=prem, seed=0)\n"
            "    res_s = st.long_short(p_s, min_stocks=10, quantile=0.20)\n"
            "    means.append(st.summary(res_s['ls_ret'])['mean']*100 if not res_s.empty else np.nan)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar([str(p) for p in premiums], means,\n"
            "       color=[GREEN if m > 0 else RED for m in means])\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted premium'); ax.set_ylabel('composite L-S mean (%/yr)')\n"
            "ax.set_title('Positive control: monotone in the planted premium')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('seed-0:', [round(m,2) for m in means])\n"
            "print('15-seed averaged (frozen):', R_CTRL)"
        ),

        md(
            "### 4b — The composite score components\n\n"
            "What do the six anomaly signals look like at the latest snapshot, and how does the "
            "composite rank order names?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets = prices.pct_change()\n"
            "    asof = prices.index[-1]\n"
            "    sig = st._anomaly_signals(prices, asof, rets=rets)\n"
            "    score = st.mispricing_score(sig).sort_values()\n"
            "    fig, ax = plt.subplots(figsize=(10, 4.6))\n"
            "    col = [GREEN if s < 0.4 else (RED if s > 0.6 else GREY) for s in score.values]\n"
            "    ax.bar(range(len(score)), score.values, color=col)\n"
            "    ax.set_xticks(range(len(score)))\n"
            "    ax.set_xticklabels(score.index, rotation=90, fontsize=7)\n"
            "    ax.axhline(0.5, c='k', lw=1, ls='--')\n"
            "    ax.set_ylabel('composite mispricing score (0=cheap, 1=overpriced)')\n"
            "    ax.set_title('Composite score across the universe (latest snapshot)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('cheapest:', list(score.index[:5]))\n"
            "    print('most overpriced:', list(score.index[-5:]))\n"
            "else:\n"
            "    print('Composite score requires the real cache.')"
        ),

        md("### 4c — Real panel: the long-short and its legs"),
        code(
            "if HAVE_REAL:\n"
            "    ls_g = s_ls['mean']*100; ls_t = s_ls['tstat']\n"
            "    ls_n = s_net['mean']*100; ls_nt = s_net['tstat']\n"
            "    long_m = s_long['mean']*100; long_t = s_long['tstat']\n"
            "    sb_m = s_sb['mean']*100; sb_t = s_sb['tstat']\n"
            "    short_names_m = st.summary(result['short_ret'])['mean']*100\n"
            "else:\n"
            "    ls_g, ls_t = R['ls_gross'], R['ls_gross_t']\n"
            "    ls_n, ls_nt = R['ls_net'], R['ls_net_t']\n"
            "    long_m, long_t = R['long_mean'], R['long_t']\n"
            "    sb_m, sb_t = R['short_book'], R['short_book_t']\n"
            "    short_names_m = R['short_names']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.6))\n"
            "labels = ['Long (cheap)', '\"Overpriced\" owned', 'Short book', 'L-S gross', 'L-S net']\n"
            "vals = [long_m, short_names_m, sb_m, ls_g, ls_n]\n"
            "cols = [GREEN, GREY, RED, RED, RED]\n"
            "ax.bar(labels, vals, color=cols)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('mean annual return (%/yr)')\n"
            "ax.set_title('Long leg wins; short leg and the L-S lose')\n"
            "plt.xticks(rotation=15)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Long:  {long_m:+.2f}%/yr (t={long_t:+.2f})')\n"
            "print(f'Short book: {sb_m:+.2f}%/yr (t={sb_t:+.2f})')\n"
            "print(f'L-S gross: {ls_g:+.2f}%/yr (t={ls_t:+.2f}) | net: {ls_n:+.2f}%/yr (t={ls_nt:+.2f})')"
        ),

        md("### 4d — The placebo: does the score sort, or is it noise?"),
        code(
            "# A small live placebo (40 shuffles for speed); the frozen 150-shuffle p is quoted.\n"
            "if HAVE_REAL:\n"
            "    obs = result['ls_ret'].mean()*12*100\n"
            "    nulls = []\n"
            "    rng = np.random.default_rng(535)\n"
            "    for _ in range(40):\n"
            "        s = int(rng.integers(0, 2**31-1))\n"
            "        sh = st.long_short(prices, shuffle_seed=s)\n"
            "        if not sh.empty:\n"
            "            nulls.append(sh['ls_ret'].mean()*12*100)\n"
            "    nulls = np.array(nulls)\n"
            "    fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "    ax.hist(nulls, bins=15, color=GREY, edgecolor='white', alpha=0.8, label='shuffled null')\n"
            "    ax.axvline(obs, c=RED, lw=2.5, label=f'observed {obs:+.1f}%/yr')\n"
            "    ax.axvline(0, c='k', lw=1, ls='--')\n"
            "    ax.set_xlabel('composite L-S mean (%/yr)'); ax.set_ylabel('count')\n"
            "    ax.set_title('Placebo: the real score sorts far from the shuffled null')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'live 40-shuffle: null mean {nulls.mean():+.2f}%, sd {nulls.std():.2f}%')\n"
            "    print(f'frozen 150-shuffle p-value: {R[\"placebo_p\"]:.3f} (obs {R[\"placebo_obs\"]:+.2f}%, '\n"
            "          f'null {R[\"placebo_null_mean\"]:+.2f}% sd {R[\"placebo_null_sd\"]:.2f}%)')\n"
            "else:\n"
            "    print(f'frozen placebo p = {R[\"placebo_p\"]:.3f}')"
        ),
        md(
            f"> The label-shuffle null is centred at ~0 ({R['placebo_null_mean']:+.2f}%/yr); the "
            f"observed {R['placebo_obs']:+.2f}%/yr sits far in the tail (**p = {R['placebo_p']:.3f}**). "
            "The composite **genuinely sorts** — it just sorts the wrong way for a long-short on "
            "survivors."
        ),

        md("### 4e — Robustness: sharper cutoffs deepen the loss"),
        code(
            "if HAVE_REAL:\n"
            "    cuts = [0.30, 0.20, 0.10]; rows = []\n"
            "    for q in cuts:\n"
            "        r = st.long_short(prices, quantile=q)\n"
            "        s = st.summary(r['ls_ret'])\n"
            "        rows.append({'cutoff': q, 'mean_%': s['mean']*100, 't': s['tstat']})\n"
            "    cut_df = pd.DataFrame(rows)\n"
            "else:\n"
            "    cut_df = pd.DataFrame({'cutoff': [0.30, 0.20, 0.10],\n"
            "        'mean_%': [R['cut_tercile'], R['cut_quintile'], R['cut_decile']],\n"
            "        't': [R['cut_tercile_t'], R['cut_quintile_t'], R['cut_decile_t']]})\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(cut_df['cutoff'], cut_df['mean_%'], 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.invert_xaxis()\n"
            "ax.set_xlabel('tail cutoff (smaller = more extreme)'); ax.set_ylabel('L-S mean (%/yr)')\n"
            "ax.set_title('More extreme overpriced tail -> deeper loss (structural, not a fluke)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(cut_df.round(3).to_string(index=False))"
        ),

        md(
            "## 5 — The verdict\n\n"
            f"- **Signal `NONE`** — L-S {R['ls_gross']:.2f}%/yr gross (HAC *t* {R['ls_gross_t']:.3f}), "
            f"{R['ls_net']:.2f}%/yr net (*t* {R['ls_net_t']:.3f}). The composite sorts "
            f"(placebo p = {R['placebo_p']:.3f}) but earns no positive *t* ≥ 2 — the only "
            "significance is adverse. Literature cannot upgrade a wrong-signed result.\n"
            f"- **Tradability `MIRAGE`** — Sharpe {R['ls_gross_sharpe']:.3f}, max DD {R['ls_dd']:.0f}%, "
            "costs deepen the loss.\n"
            f"- **Short-leg edge `BUSTED`** — short book {R['short_book']:.1f}%/yr "
            f"(*t* {R['short_book_t']:.2f}). Survivorship deletes the failed overpriced names that "
            "drive the SYY short leg; the survivors are the decade's winners."
        ),

        md(
            "## 6 — Could you trade it?\n\n"
            "Only the long (cheap) leg, and only as a long-only tilt — not the headline long-short. "
            "The short leg needs the delisted tail the survivor universe removes; without it, "
            "shorting 'overpriced' glamour is shorting the winners. Net-of-cost the long-short is "
            "untradeable (loss, −80% drawdown). The long leg "
            f"(+{R['long_mean']:.1f}%/yr, *t* +{R['long_t']:.2f}) is real but is largely the "
            "survivor mega-cap rally, not cheapness per se."
        ),

        md(
            "## 7 — Going further\n\n"
            "- **Delisting-complete universe (CRSP/Compustat).** The only honest SYY test — it "
            "restores the short leg's fuel.\n"
            "- **Add the four fundamental anomalies** (accruals, net issuance, profitability, "
            "distress): [231 Sloan-Accruals](../../231-sloan-accruals/), "
            "[244 Asset-Growth](../../244-asset-growth/).\n"
            "- **The single anomalies it claims to dominate:** "
            "[238 Betting-Against-Beta](../../238-betting-against-beta/), "
            "[330 Low-Volatility](../../330-low-volatility-anomaly/), "
            "[363 PEAD-Drift](../../363-pead-drift/).\n\n"
            "*Think the composite survives once the delisted overpriced names are back and the "
            "fundamental anomalies are added? Fork this, pull a delisting-complete universe, and "
            "show a positive short-leg *t* > 2 net of borrow. That is the bar.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _inject_ctrl(nb)
    _write(nb, "02_for_the_quants.ipynb")


def _inject_ctrl(nb):
    """Make the frozen real-number dict ``R`` (and ``R_CTRL``) available to the
    notebook runtime: prepend ``R = {...}`` to the BOOT cell (the first code
    cell), and ``R_CTRL = {...}`` to the control cell that prints it."""
    booted = False
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        if not booted:
            c.source = "R = " + repr(R) + "\n" + c.source
            booted = True
        if "R_CTRL" in c.source and "R_CTRL =" not in c.source:
            c.source = "R_CTRL = " + repr(R["ctrl"]) + "\n" + c.source
    if not booted:
        raise RuntimeError("no code cell found to inject R")


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
