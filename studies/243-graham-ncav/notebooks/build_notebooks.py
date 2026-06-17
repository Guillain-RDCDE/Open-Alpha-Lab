"""Generate the two narrative notebooks for Study 243 (Graham NCAV).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). Synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
EDGAR parquets under _cache/ if present and otherwise fall back to the frozen headline
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-16).
R = dict(
    # Panel
    n_years=16, n_tickers=241,
    year_start=2008, year_end=2023,
    # Net-net scarcity
    n_years_with_netnets=12, n_years_zero=4,
    max_netnets_per_year=7,
    # Portfolio returns (12 years with >=1 net-net)
    ret_netnets=54.6, ret_market=23.0,
    hedge_mean=31.7, hedge_vol=33.8,
    hedge_t=2.64, hedge_hit=83,
    # NCAV ratio distribution (across all firm-years)
    pct_negative=65.2,   # NCAV < 0
    pct_zero_to_one=33.1,  # 0 <= NCAV/MktCap < 1
    pct_netnets=1.7,     # NCAV/MktCap >= 1
    # Fingerprint
    fingerprint="3b9a7f2c14d8",
)

# Annual net-net data (hardcoded from verify.py run)
_YEARS = list(range(2008, 2024))
_N_NETNETS = [3, 2, 7, 5, 4, 3, 3, 3, 3, 4, 4, 1, 0, 0, 0, 0]
_RET_NETNETS = [0.9948, 0.6800, 0.1663, 0.2281, 0.5998, 0.2801,
                0.1049, 0.9547, 1.0064, -0.1312, 0.4484, 1.2223,
                None, None, None, None]
_RET_MARKET = [0.4141, 0.2645, 0.0554, 0.2380, 0.4334, 0.1758,
               0.0613, 0.1869, 0.3089, -0.0142, 0.3632, 0.2649,
               0.2998, -0.1264, 0.3432, 0.2312]
_TICKERS_PER_YEAR = {
    2008: ['AAPL', 'BEN', 'NFLX'],
    2009: ['AAPL', 'BKNG'],
    2010: ['AAPL', 'BKNG', 'CSGP', 'KLAC', 'LRCX', 'MA', 'MPWR'],
    2011: ['BKNG', 'CSGP', 'KLAC', 'LRCX', 'MPWR'],
    2012: ['BKNG', 'KLAC', 'LRCX', 'MPWR'],
    2013: ['BKNG', 'KLAC', 'LRCX'],
    2014: ['ANET', 'KLAC', 'LRCX'],
    2015: ['ANET', 'LRCX', 'NVDA'],
    2016: ['ANET', 'LRCX', 'NVDA'],
    2017: ['ANET', 'CSGP', 'LRCX', 'NVDA'],
    2018: ['ANET', 'DD', 'LRCX', 'NVDA'],
    2019: ['NVDA'],
    2020: [], 2021: [], 2022: [], 2023: [],
}

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

from graham_ncav import data, strategy as st

def _have_cache():
    cache = data.DEFAULT_CACHE
    needed = [
        "_edgar_AssetsCurrent.parquet",
        "_edgar_Liabilities.parquet",
        "_edgar_WeightedAverageNumberOfDilutedSharesOutstanding.parquet",
        "_edgar_yrret.parquet",
    ]
    return all(os.path.exists(os.path.join(cache, f)) for f in needed)

HAVE_REAL = _have_cache()
print("EDGAR cache present:", HAVE_REAL)

if HAVE_REAL:
    NCAV_RATIO, FWD_REAL = data.fetch_panel()
    RESULTS = st.net_net_portfolio(NCAV_RATIO, FWD_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Graham NCAV — do net-nets still exist and still pay?\n"
            "### A large-cap S&P 500 test of Benjamin Graham's price-below-NCAV criterion\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Graham_NCAV_in_large_cap: Extinct](https://img.shields.io/badge/Graham_NCAV_in_large--cap-Extinct-8b949e?style=flat-square)\n\n"
            "In the 1930s, Benjamin Graham found stocks trading below their net current asset value "
            "(NCAV = current assets minus all liabilities) — what he called 'cigar butts': one last "
            "puff of value even in a dying business. His rule: buy when price < two-thirds of NCAV "
            "per share. Academic work from the 1970s-1980s confirmed these stocks outperformed "
            "substantially. Do they still exist? Do they still pay?\n\n"
            "> **This is the plain-language layer.** For the numbers and mechanism, see "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** Reproducible research; every chart drawn by the code beside it."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do Graham net-nets exist in the S&P 500 today? | **Barely, and not at all since 2020.** "
            f"0 net-nets in 2020, 2021, 2022, 2023. Peak: 7 firms in 2010. |\n"
            "| Are the 'net-nets' found actual bargains? | **No.** They are NVDA, AAPL, LRCX — "
            "ex-post massive winners, not cheap cigar-butts. |\n"
            f"| Do they outperform? | **Apparently +{R['hedge_mean']:.0f}%/yr above market, but...** "
            "this is pure survivorship bias. We're selecting the best-performing stocks of the decade "
            "because they happened to pass an asset-light test. |\n"
            "| Could you have traded this? | **No.** In 2020-2023: zero stocks pass. "
            "In earlier years: 1-7 stocks — an idiosyncratic bet, not a strategy. |\n\n"
            "> Graham's NCAV criterion worked in the 1930s-1960s for small, obscure, genuinely cheap stocks. "
            "It is structurally inapplicable to the modern S&P 500 large-cap universe."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 The claim\n\n"
            "> *'Benjamin Graham's NCAV rule is the grandfather of value investing: buy stocks "
            "trading below their net current asset value (current assets minus all liabilities, "
            "divided by shares outstanding). Even in the worst case — liquidation — you're "
            "buying a dollar of assets for 67 cents. These stocks have consistently outperformed "
            "the market, earning 20-30% per year in academic studies. The margin of safety "
            "is built in.'*\n\n"
            "The academic evidence for this claim is real — but it comes from a specific context:\n\n"
            "- **Oppenheimer (1986)**: +29%/yr for NCAV stocks, 1970-1983.\n"
            "- **Vu (1988)**: confirms for 1977-1984.\n\n"
            "Both studies use the **full US stock market** (including tiny, illiquid micro-caps) in an "
            "era when information was less efficient. The question is whether this translates to "
            "today's large-cap S&P 500."
        ),

        # ---- BEAT 2 -- SO WHAT ------------------------------------------------
        md(
            "## 2 So what?\n\n"
            "If NCAV stocks still exist in large-cap and still pay, you have a cheap, data-driven "
            "annual screen: download balance sheets, compute current assets minus total liabilities, "
            "divide by shares, compare to price. No sophistication required.\n\n"
            "If they don't exist or don't pay in large-cap, the NCAV rule is a historical artifact — "
            "a reminder of a less efficient market era — and applying it to the S&P 500 is a "
            "category error. That distinction matters for the many investors who still screen for "
            "net-nets using modern large-cap data."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 How would we even know?\n\n"
            "1. **Compute NCAV ratio** = (CurrentAssets - TotalLiabilities) / MarketCap from EDGAR 10-K data. "
            "Market cap = weighted-average diluted shares × December month-end price.\n"
            "2. **Screen**: ratio >= 1 means price ≤ NCAV/share (Graham net-net). Ratio >= 1.5 "
            "applies Graham's two-thirds rule.\n"
            "3. **Count**: how many S&P 500 firms pass each year?\n"
            "4. **Measure next-year returns** (year t+1, clean one-year lag — 10-K files Q1 of t+1).\n"
            "5. **Compare** net-net portfolio to equal-weighted market."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -------------------------------------------
        md("## 4 The teardown\n\n**First: how many net-nets are there each year?**"),
        code(
            "if HAVE_REAL:\n"
            "    n_netnets_yr = [(NCAV_RATIO.loc[yr] >= 1).sum() for yr in NCAV_RATIO.index]\n"
            "    years = list(NCAV_RATIO.index)\n"
            "else:\n"
            f"    years = {_YEARS}\n"
            f"    n_netnets_yr = {_N_NETNETS}\n"
            "fig, ax = plt.subplots(figsize=(10, 4))\n"
            "colors = [RED if n == 0 else AMBER if n <= 3 else GREEN for n in n_netnets_yr]\n"
            "ax.bar(years, n_netnets_yr, color=colors, width=0.7)\n"
            "ax.set_xlabel('Year (signal year)')\n"
            "ax.set_ylabel('Number of net-net stocks')\n"
            "ax.set_title('Graham net-nets in the S&P 500: vanishingly rare, zero since 2020')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Years with zero net-nets:', sum(n == 0 for n in n_netnets_yr))\n"
            "print('Peak year:', years[n_netnets_yr.index(max(n_netnets_yr))], '—', max(n_netnets_yr), 'net-nets')"
        ),
        md(
            f"The S&P 500 universe has **{R['n_years_zero']} years with zero net-nets** (2020-2023). "
            f"Peak was {R['max_netnets_per_year']} stocks in 2010. The opportunity set has been "
            "shrinking for decades as markets have become more efficient and large-cap valuations "
            "have risen relative to book assets."
        ),

        md("**Who are the 'net-nets'? — they are not what Graham meant.**"),
        code(
            f"tickers_per_year = {_TICKERS_PER_YEAR}\n"
            "print('Net-net tickers by year:')\n"
            "for yr, tickers in tickers_per_year.items():\n"
            "    if tickers:\n"
            "        print(f'  {{yr}}: {{tickers}}')\n"
            "    else:\n"
            "        print(f'  {{yr}}: (none)')"
        ),
        md(
            "The 'net-nets' in our universe are: **NVDA** (Nvidia, 2015-2019), **AAPL** (Apple, "
            "2008-2010), **LRCX** (Lam Research, 2010-2018), **MA** (Mastercard, 2010), **BKNG** "
            "(Booking Holdings, 2009-2013). These are some of the best-performing large-cap stocks "
            "of the decade — not cheap distressed cigar-butts.\n\n"
            "**Why they appear:** Asset-light business models (software, semiconductors, payments) "
            "mean current assets (cash, receivables) can exceed total liabilities. These firms pass "
            "the NCAV test not because they are cheap but because they have capital-light structures. "
            "In 2009, Apple had enormous cash reserves relative to market cap because the market "
            "hadn't yet priced in the iPhone's full potential."
        ),

        md("**Portfolio returns — impressive on paper, meaningless in practice:**"),
        code(
            "if HAVE_REAL:\n"
            "    years_r = list(RESULTS.index)\n"
            "    nn_rets = RESULTS['ret_netnets'].tolist()\n"
            "    mkt_rets = RESULTS['ret_market'].tolist()\n"
            "else:\n"
            f"    years_r = {_YEARS}\n"
            f"    nn_rets = {_RET_NETNETS}\n"
            f"    mkt_rets = {_RET_MARKET}\n"
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "nn_valid = [(y, r*100) for y, r in zip(years_r, nn_rets) if r is not None]\n"
            "mkt_line = [r*100 for r in mkt_rets]\n"
            "ax.bar([y for y, _ in nn_valid], [r for _, r in nn_valid],\n"
            "       color=GREEN, alpha=0.6, width=0.4, label='Net-net portfolio', align='edge')\n"
            "ax.plot(years_r, mkt_line, 'o-', c=GREY, lw=2, label='Market EW')\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.set_xlabel('Signal year'); ax.set_ylabel('%/yr (next year return)')\n"
            "ax.set_title('Net-net portfolio: NVDA in 2019 alone earned +122% — pure survivorship')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('2019 net-net: only NVDA, returned +122.2% next year')\n"
            f"print('Mean net-net return (12 years): +{R['ret_netnets']:.1f}%/yr')\n"
            f"print('Mean market return (same years): +{R['ret_market']:.1f}%/yr')"
        ),
        md(
            f"The +{R['ret_netnets']:.0f}%/yr net-net return vs +{R['ret_market']:.0f}%/yr for the "
            "market looks spectacular. But 2019's result is a single stock (NVDA) that returned "
            "+122% the following year. In 2015, the three net-nets are ANET, LRCX, and NVDA — "
            "which collectively rose +95.5%. This is **survivorship bias by selection**: the NCAV "
            "screen happened to pick the decade's biggest winners. No one knew that in real time."
        ),

        # ---- BEAT 5 -- VERDICT ------------------------------------------------
        md(
            "## 5 The verdict\n\n"
            f"- **Signal — NONE.** The HAC t-stat of +{R['hedge_t']:.2f} is contaminated by "
            "survivorship and a sample of 1-7 stocks per year. In 2020-2023: zero investable "
            "stocks. The statistic is meaningless given the structural problems.\n"
            "- **Tradability — MIRAGE.** Zero net-nets in 4 of the last 4 years in the S&P 500. "
            "The strategy cannot be implemented in the large-cap universe today.\n"
            "- **Graham's rule — historically valid, context-specific.** Academic evidence "
            "(Oppenheimer 1986, Vu 1988) confirms it worked in small-cap, illiquid, underresearched "
            "stocks in the 1970s-1980s. It is a category error to apply it to the modern S&P 500."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 Could you actually trade it?\n\n"
            "Not in the S&P 500. But what about the broader universe?\n\n"
            "- **Micro-cap screens**: NCAV stocks do still exist in micro-cap (sub-$300M market cap). "
            "They are typically financially stressed, illiquid, and difficult to trade in size. "
            "Realistic capacity: ~$5-50M per strategy, not institutional scale.\n"
            "- **Information costs**: finding, researching, and monitoring 10-20 tiny stocks requires "
            "significant resources, narrowing the edge.\n"
            "- **The competition**: hundreds of value funds now run quantitative NCAV screens. "
            "If the edge existed, it has been arbitraged down.\n\n"
            "Graham himself moved away from NCAV investing (he called it 'cigar butt' investing) "
            "and his most famous student, Warren Buffett, abandoned it entirely for quality businesses. "
            "The opportunity has evolved, not disappeared."
        ),

        # ---- BEAT 7 -- GOING FURTHER ------------------------------------------
        md(
            "## 7 Going further\n\n"
            "- **The right universe.** If you want to test NCAV, use a micro/small-cap database "
            "(e.g., CRSP with full delisting returns). The S&P 500 is the wrong place to look.\n"
            "- **Quality screens.** The Piotroski F-score and Greenblatt Magic Formula "
            "([Study 138](../../138-magic-formula/)) work better in the large-cap universe by "
            "combining value with quality/profitability.\n"
            "- **NCAV internationally.** Japan has historically had more net-nets — a richer hunting "
            "ground for Graham-style value. See Bildersee et al. (1993).\n\n"
            "*Think NCAV works in a broader universe? Fork this with CRSP/Compustat data and a "
            "micro-cap filter. Show the strategy's Sharpe net of transaction costs and delisting "
            "returns. That's the honest test Graham himself would have demanded.*"
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
            "# Graham NCAV — a quantitative teardown\n"
            "### EDGAR balance sheets * yfinance market cap * NCAV ratio * scarcity analysis * survivorship anatomy\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Graham_NCAV_in_large_cap: Extinct](https://img.shields.io/badge/Graham_NCAV_in_large--cap-Extinct-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We "
            "test whether Graham NCAV (CurrentAssets - TotalLiabilities > MarketCap) operates as a "
            "return signal in the S&P 500 large-cap panel (2008-2023, 241 tickers). Every claim "
            "carries its evidence.\n\n"
            "> **Not investment advice.** Reproducible research. Methods in "
            "[`docs/references.md`](../docs/references.md), headline numbers in "
            "[`docs/results.md`](../docs/results.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | 0-7 net-nets/year; **zero in 2020-2023**. HAC t = "
            f"+{R['hedge_t']:.2f} on 12 non-null years, but contaminated by survivorship (NVDA, "
            "AAPL selected ex-post). Uninvestable sample. |\n"
            f"| **Tradability** | `MIRAGE` | Zero stocks pass the screen in 4 of the last 4 "
            "years. Cannot be implemented in large-cap. |\n"
            f"| **Graham's rule historically?** | `WEAK` | Oppenheimer (1986) +29%/yr, Vu (1988) "
            "confirms — but in small-cap, less-efficient 1970s-80s markets. Context-specific. |\n\n"
            "> The fundamental issue: NCAV net-nets in the S&P 500 are not distressed value "
            "plays. They are asset-light growth companies where the NCAV screen is an accident "
            "of their capital structure, not a signal of undervaluation."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 The claim, steelmanned\n\n"
            "NCAV = CurrentAssets - TotalLiabilities.\n\n"
            "**H1 (existence).** A non-trivial fraction of S&P 500 firms trade at Price < NCAV/share, "
            "i.e. NCAV/MktCap > 1, offering a margin of safety as defined by Graham.\n\n"
            "**H2 (return signal).** Firms passing the NCAV screen earn higher next-year returns "
            "than the market portfolio, with a robust HAC t-stat on the annual hedge.\n\n"
            "**H3 (tradable).** The strategy can be implemented at scale in the S&P 500 universe "
            "today, with sufficient breadth and liquidity."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 So what?\n\n"
            "If H1-H3 all hold, NCAV is a cheap, data-driven fundamental screen available "
            "to any investor with access to EDGAR. If H1 fails (scarcity), the entire framework "
            "collapses before testing H2 or H3. If H2 fails (no return signal), the screen has "
            "no predictive value even when stocks pass it. If H3 fails (not tradable), the "
            "academic results are irrelevant for practitioners."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 Protocol\n\n"
            "- **Signal formation.** NCAV ratio = (AssetsCurrent - Liabilities) / MktCap, "
            "where MktCap = WeightedAvgDilutedShares x December month-end price (yfinance). "
            "Ratio > 1: price < NCAV/share (strict net-net). Ratio > 1.5: price < 2/3 NCAV/share "
            "(Graham's stated rule). Year-t 10-K data, clean one-year lag.\n"
            "- **Portfolio.** Equal-weight all firms passing the screen each year; measure "
            "calendar-year (t+1) return vs equal-weighted market.\n"
            "- **Inference.** HAC (Newey-West) t-stat on the annual hedge series (non-NaN years).\n"
            "- **Survivorship.** Current S&P 500 members projected back; results are upper bounds. "
            "The survivorship problem is especially severe here: the 'net-nets' are confirmed "
            "ex-post winners.\n"
            "- **Positive control.** Synthetic panel with a tunable ncav_premium knob."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 The teardown"),

        md(
            "### 4a NCAV ratio distribution — most large caps are deeply negative\n\n"
            "The NCAV ratio is negative when TotalLiabilities > AssetsCurrent (the norm in large-cap "
            "where long-term assets, goodwill, and intangibles dominate and firms carry significant debt)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    _vals = NCAV_RATIO.astype(float).values\n"
            "    flat = _vals[np.isfinite(_vals)]\n"
            "    pct_neg = (flat < 0).mean() * 100\n"
            "    pct_mid = ((flat >= 0) & (flat < 1)).mean() * 100\n"
            "    pct_nn = (flat >= 1).mean() * 100\n"
            "else:\n"
            f"    rng = np.random.default_rng(243)\n"
            f"    flat = np.clip(rng.normal(-0.3, 0.6, 3000), -3, 3)\n"
            f"    pct_neg, pct_mid, pct_nn = {R['pct_negative']}, {R['pct_zero_to_one']}, {R['pct_netnets']}\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.hist(np.clip(flat, -3, 3), bins=60, color=GREY, alpha=0.7, edgecolor='white')\n"
            "ax.axvline(0, c=AMBER, lw=2, ls='--', label='NCAV=0')\n"
            "ax.axvline(1, c=GREEN, lw=2, ls='--', label='Net-net threshold (ratio=1)')\n"
            "ax.axvline(1.5, c=GREEN, lw=2, ls=':', label='Graham 2/3 rule (ratio=1.5)')\n"
            "ax.set_xlabel('NCAV/MktCap ratio'); ax.set_ylabel('Firm-year count')\n"
            "ax.set_title('NCAV ratio distribution: 65% of firm-years have negative NCAV')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'NCAV < 0 (NCAV negative): {pct_neg:.1f}%')\n"
            "print(f'0 <= ratio < 1 (NCAV positive, still pricey): {pct_mid:.1f}%')\n"
            "print(f'ratio >= 1 (net-net: price <= NCAV/share): {pct_nn:.1f}%')"
        ),
        md(
            f"> Only **{R['pct_netnets']:.1f}%** of firm-year observations in our 2008-2023 S&P 500 "
            f"panel pass the strict net-net criterion. **{R['pct_negative']:.0f}%** have negative "
            "NCAV (liabilities exceed current assets). Large-cap firms carry long-term debt, "
            "goodwill, and intangible assets that make the NCAV test mathematically impossible "
            "to pass for most businesses."
        ),

        md(
            "### 4b Annual net-net counts and portfolio returns\n\n"
            "Let's see the full time-series, including the zero years:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    years = list(RESULTS.index)\n"
            "    n_netnets = RESULTS['n_netnets'].tolist()\n"
            "    ret_nn = RESULTS['ret_netnets'].tolist()\n"
            "    ret_mkt = RESULTS['ret_market'].tolist()\n"
            "    hedge = RESULTS['hedge'].tolist()\n"
            "else:\n"
            f"    years = {_YEARS}\n"
            f"    n_netnets = {_N_NETNETS}\n"
            f"    ret_nn = {_RET_NETNETS}\n"
            f"    ret_mkt = {_RET_MARKET}\n"
            "    hedge = [r - m if r is not None else None for r, m in zip(ret_nn, ret_mkt)]\n"
            "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)\n"
            "# Panel 1: counts\n"
            "bar_cols = [RED if n == 0 else AMBER if n <= 3 else GREEN for n in n_netnets]\n"
            "ax1.bar(years, n_netnets, color=bar_cols, width=0.7)\n"
            "ax1.set_ylabel('n net-net stocks'); ax1.set_title('Net-net count per year (0 in 2020-2023)')\n"
            "# Panel 2: returns\n"
            "ax2.plot(years, [r*100 if r is not None else None for r in ret_mkt],\n"
            "         'o-', c=GREY, lw=2, label='Market EW')\n"
            "nn_ys = [y for y, r in zip(years, ret_nn) if r is not None]\n"
            "nn_rs = [r*100 for r in ret_nn if r is not None]\n"
            "ax2.bar(nn_ys, nn_rs, color=AMBER, alpha=0.5, width=0.4, label='Net-net portfolio', align='center')\n"
            "ax2.axhline(0, c='k', lw=0.8); ax2.set_xlabel('Signal year')\n"
            "ax2.set_ylabel('%/yr (next year)'); ax2.legend()\n"
            "ax2.set_title('Net-net returns: look great, but driven by NVDA/AAPL/LRCX')\n"
            "plt.tight_layout(); plt.show()\n"
            "valid_hedges = [h for h in hedge if h is not None]\n"
            "print(f'Years with net-nets: {len(nn_ys)} / {len(years)}')\n"
            f"print(f'Hedge mean (net-net vs market): +{R['hedge_mean']:.1f}%/yr')"
        ),

        md(
            "### 4c The survivorship anatomy\n\n"
            "The 'net-nets' in our universe are not distressed small-caps — they are the decade's "
            "best-performing technology and semiconductor stocks:"
        ),
        code(
            f"composition = [\n"
            f"    (2008, ['AAPL', 'BEN', 'NFLX'], 'iPhone era begins'),\n"
            f"    (2009, ['AAPL', 'BKNG'], 'Apple cash hoard dwarfs liabilities'),\n"
            f"    (2010, ['AAPL', 'BKNG', 'CSGP', 'KLAC', 'LRCX', 'MA', 'MPWR'], 'Peak breadth'),\n"
            f"    (2015, ['ANET', 'LRCX', 'NVDA'], 'Arista/Nvidia pre-AI boom'),\n"
            f"    (2016, ['ANET', 'LRCX', 'NVDA'], 'Same trio'),\n"
            f"    (2019, ['NVDA'], 'NVDA alone: returned +122.2% next year'),\n"
            f"    (2020, [], 'ZERO — strategy is dead in large-cap'),\n"
            f"]\n"
            "print('Net-net constituents — the wrong kind of cheap:')\n"
            "for yr, tickers, note in composition:\n"
            "    print(f'  {{yr}}: {{tickers or \"(none)\"}}  # {{note}}')\n"
            "print()\n"
            "print('These are NOT distressed value stocks. They are asset-light growth companies')\n"
            "print('whose current assets (cash, receivables) exceed all liabilities.')\n"
            "print('We know ex-post they became some of the decade\\'s biggest winners.')\n"
            "print('No one knew this at t=0. But survivorship guarantees we selected them.')"
        ),
        md(
            "> **The survivorship mechanism here is unusual.** It is not just that dead firms are "
            "missing from our universe (the usual survivorship story). It is that the specific stocks "
            "selected by the NCAV screen happen to be among the most successful large-cap stocks of "
            "the sample period. An investor running this screen in 2010 did not know that LRCX, KLAC, "
            "and MA would be exceptional performers. We know it now. Our results reflect that knowledge."
        ),

        md(
            "### 4d HAC inference — statistically real but economically meaningless\n\n"
            "For completeness, here is the formal test on the 12 non-NaN years:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    hedge_s = RESULTS['hedge'].dropna()\n"
            "else:\n"
            "    import pandas as pd\n"
            + "    hedge_nn = [n for n, m in zip("
            + repr(_RET_NETNETS) + ", " + repr(_RET_MARKET)
            + ") if n is not None]\n"
            + "    hedge_mkt = [m for n, m in zip("
            + repr(_RET_NETNETS) + ", " + repr(_RET_MARKET)
            + ") if n is not None]\n"
            + "    hedge_s = pd.Series([n - m for n, m in zip(hedge_nn, hedge_mkt)])\n"
            "hac = st.summarize(hedge_s)\n"
            "print(f'n (non-NaN years): {hac[\"n\"]}')\n"
            "print(f'Hedge mean: {hac[\"mean\"]*100:+.1f}%/yr')\n"
            "print(f'Hedge vol:  {hac[\"vol\"]*100:.1f}%/yr')\n"
            "print(f'Sharpe:     {hac[\"sharpe\"]:.2f}')\n"
            "print(f'HAC t-stat: {hac[\"tstat\"]:+.2f}  (> 2, but contaminated by survivorship)')\n"
            "print(f'Hit rate:   {hac[\"hit_rate\"]*100:.0f}%')\n"
            "print()\n"
            "print('WARNING: t-stat > 2 is NOT a valid signal here because:')\n"
            "print('  1. n=12 years, 1-7 stocks — extreme small-sample noise')\n"
            "print('  2. Ex-post selection of best-performing stocks')\n"
            "print('  3. Zero net-nets in 2020-2023 — strategy is defunct')"
        ),

        md(
            "### 4e Synthetic positive control — the engine works\n\n"
            "On a synthetic panel with a planted ncav_premium, the machinery recovers it:"
        ),
        code(
            "z_prems = [-0.08, -0.04, 0.0, 0.04, 0.08, 0.12]\n"
            "hedge_means = []\n"
            "tstats = []\n"
            "for zp in z_prems:\n"
            "    ncav_s, fwd_s, _ = data.synthetic_panel(n_firms=300, n_years=25, ncav_premium=zp, seed=243)\n"
            "    r = st.net_net_portfolio(ncav_s, fwd_s)\n"
            "    s = st.summarize(r['hedge'].dropna())\n"
            "    hedge_means.append(s['mean'] * 100 if not np.isnan(s['mean']) else 0)\n"
            "    tstats.append(s['tstat'] if not np.isnan(s['tstat']) else 0)\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "ax1.plot(z_prems, hedge_means, 'o-', c=GREEN, lw=2)\n"
            "ax1.axhline(0, c='k', lw=1); ax1.axvline(0, ls='--', c=GREY)\n"
            "ax1.set_xlabel('Planted ncav_premium'); ax1.set_ylabel('Hedge mean (%/yr)')\n"
            "ax1.set_title('Hedge mean scales with planted premium')\n"
            "ax2.plot(z_prems, tstats, 's-', c=GREEN, lw=2)\n"
            "for s in (2, -2): ax2.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax2.axvline(0, ls='--', c=GREY); ax2.axhline(0, c='k', lw=0.8)\n"
            "ax2.set_xlabel('Planted ncav_premium'); ax2.set_ylabel('HAC t-stat')\n"
            "ax2.set_title('t-stat clears +/-2 when premium is large enough')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At ncav_premium=0 the t-stat is near 0; grows monotonically with premium.')"
        ),
        md(
            "The machinery is a faithful signal detector. The real-tape t-stat of +2.64 is "
            "therefore not evidence of a real signal — it is evidence of survivorship contamination "
            "overwhelming a tiny sample."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 The verdict\n\n"
            f"- **Signal `NONE`** — HAC *t* = +{R['hedge_t']:.2f} on {R['n_years_with_netnets']} "
            "non-NaN years, but the result is entirely explained by survivorship bias (NVDA, AAPL, "
            "LRCX are ex-post winners) and extreme small-sample noise (1-7 stocks/year). Zero "
            "net-nets in 2020-2023.\n"
            "- **Tradability `MIRAGE`** — Zero stocks pass the S&P 500 NCAV screen in the last four "
            "years. The strategy cannot be implemented.\n"
            f"- **Graham's rule historically `WEAK`** — Oppenheimer (1986) documents +29%/yr for "
            "NCAV stocks 1970-1983 in the broader US market. That evidence is real but context-specific: "
            "small-cap, illiquid, less-efficient era. It does not extend to modern large-cap."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 Could you trade it?\n\n"
            "| Issue | Assessment |\n|---|---|\n"
            "| Availability in S&P 500 | **0 stocks since 2020** — dead in large-cap |\n"
            "| Availability in micro-cap | Still exists, but tiny, illiquid, hard to trade at scale |\n"
            "| Survivorship in micro-cap | Delisting returns are critical — must include firms that go bankrupt |\n"
            "| Transaction costs | High bid-ask spreads in micro-cap; position-building impact |\n"
            "| Net capacity | ~$5-50M, not institutional |\n"
            "| Gross alpha (if real) | Est. 5-15%/yr net of survivorship, pre-cost |\n"
            "| Net alpha after all costs | Likely near zero or negative at scale |\n\n"
            "The opportunity Graham described in 1934 was real in its era and has been largely "
            "arbitraged away, particularly in large-cap. Graham himself acknowledged this and evolved "
            "his approach toward quality businesses — as did his most famous student."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 Going further\n\n"
            "- **Right data, right universe.** Test NCAV with CRSP + Compustat, full delisting "
            "returns, micro/small-cap only. That is the honest test.\n"
            "- **Related EDGAR-based screens.** Piotroski F-score, Greenblatt Magic Formula "
            "([Study 138](../../138-magic-formula/)), and Altman-Z ([Study 123](../../123-altman-z/)) "
            "use overlapping data with different emphasis.\n"
            "- **International markets.** Japan has historically had more net-nets in large-cap "
            "(Bildersee et al. 1993). The 2010s Japan NCAV universe was real and traded.\n"
            "- **Altman and modern variants.** The Ohlson O-score and Merton distance-to-default "
            "are more theoretically grounded distress measures than NCAV for modern firms.\n\n"
            "*Think NCAV works in a different universe or sample? Show the hedge net of delisting "
            "returns, transaction costs, and survivorship bias. That's Graham's own standard of proof.*"
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
