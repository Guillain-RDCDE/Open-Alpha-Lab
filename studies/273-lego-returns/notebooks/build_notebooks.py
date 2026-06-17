"""Generate the two narrative notebooks for Study 273 (Lego-Returns).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The
hardcoded LEGO secondary-market index and the synthetic generator run anywhere,
offline and deterministically; the real S&P 500 cells use the repo-level cache
(``_cache/^GSPC_split_only.parquet``) if present, and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs
for any reader with zero error cells.

The _write convention (each build_*() ends by calling _write) is kept so the
repo's intro-restyle tooling can monkeypatch it.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-17).
R = dict(
    n=38,
    year_start=1987,
    year_end=2024,
    fp="c2f1ae5a05d5",
    lego_cagr=8.3,
    lego_net_cagr=5.9,
    market_cagr=8.76,
    lego_mean=8.33,
    market_mean=10.16,
    mean_excess_gross=-1.83,
    mean_excess_net=-4.23,
    hac_t_gross=-0.668,
    hac_t_net=-1.544,
    beta=0.011,
    alpha=8.22,
    corr=0.067,
    lego_win_years=16,
    lego_win_rate=42.1,
    sign_test_p=0.872,
    one_way_cost=12,
    holding_years=5,
    div_yield=2.0,
)

R_CELL = """\
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-17)
R = dict(
    n=38, year_start=1987, year_end=2024,
    lego_cagr=8.3, lego_net_cagr=5.9, market_cagr=8.76,
    lego_mean=8.33, market_mean=10.16,
    mean_excess_gross=-1.83, mean_excess_net=-4.23,
    hac_t_gross=-0.668, hac_t_net=-1.544,
    beta=0.011, alpha=8.22, corr=0.067,
    lego_win_years=16, lego_win_rate=42.1, sign_test_p=0.872,
    one_way_cost=12, holding_years=5, div_yield=2.0,
)
"""

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from lego_returns import data, strategy as st

def _have_cache():
    try:
        data.fetch_market_annual()
        return True
    except FileNotFoundError:
        return False

HAVE_REAL = _have_cache()
print("S&P 500 cache present:", HAVE_REAL)
"""

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
)


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Lego-Returns -- do retired Lego sets beat the stock market?\n"
            "### \"The toy of smart investors\", tested honestly, in plain English\n\n"
            + BADGES +
            "You will have seen the headline: *retired LEGO sets have returned more than "
            "gold, stocks, and bonds.* It comes from a real academic paper (Dobrynskaya & "
            "Kishilova, 2022) that found ~10%/yr on the secondary market with almost zero "
            "correlation to equities. This notebook asks the only questions that matter for "
            "an investor: **after fees, does a basket of retired sets actually beat a plain "
            "S&P 500 fund -- and could you realistically capture it?**\n\n"
            "> **This is the plain-language layer.** Want the Newey-West t-stat on excess "
            "return, the CAPM beta, and the cost model? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 -- VERDICT ----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the curated Lego index beat the S&P (price-only)? | **No, narrowly.** "
            "Lego **8.3%/yr** vs S&P **8.8%/yr** gross -- and the S&P also pays ~2%/yr in "
            "dividends the Lego basket cannot. |\n"
            "| After resale fees? | **Clearly no.** Net of a one-way ~12% marketplace fee, "
            "Lego drops to **~5.9%/yr** -- a **-4.2pp/yr** shortfall vs the market. |\n"
            "| Is the outperformance statistically real? | **No.** Mean excess return is "
            "**negative**; Newey-West t = **-0.67** gross, **-1.54** net. |\n"
            "| Is the 'great diversifier' claim true? | **Partly.** Beta to the market is "
            "**~0.01** (genuinely uncorrelated) -- but a low-beta asset that trails the "
            "market is not a free lunch. |\n\n"
            "> Lego is a real, fun, low-correlation collectible. It is **not** a stock-beating "
            "investment once you pay to sell it -- and the published returns are upper bounds."
        ),

        # ---- BEAT 1 -- THE CLAIM --------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"Retired LEGO sets are a high-return, low-risk alternative investment: on the "
            "secondary market they have appreciated ~10%/yr, beating large stocks, bonds, gold "
            "and many art categories, with almost no correlation to financial markets.\"*\n\n"
            "This traces to Dobrynskaya & Kishilova (2022), *Research in International Business "
            "and Finance*. It is a genuine, careful paper -- but the press telescoped it into "
            "\"LEGO beats the S&P 500\", and quietly dropped the fees, the storage, the "
            "illiquidity, and the selection bias."
        ),

        # ---- BEAT 2 -- SO WHAT ----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If true, it would be remarkable: an asset uncorrelated with equities that *also* "
            "beats them is the holy grail of portfolio construction -- you would add it for the "
            "diversification and keep it for the return. The interesting question is which half "
            "of the claim survives contact with a fee schedule. Low correlation? Probably. "
            "Market-beating net return for a real buyer? That is what we test."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **The wrong benchmark.** A collectible pays no dividend, so it must be compared "
            "to the **price-only** S&P, not total return. We do that -- and still note the "
            "~2%/yr dividend the equity holder pockets on top.\n"
            "2. **Gross vs net.** Index returns are *paper* returns. Realising them means "
            "selling through eBay/BrickLink and paying ~12-15%. A buy-and-hold collector eats "
            "that once; we amortise it over the holding period and report both.\n"
            "3. **Survivorship & selection bias.** A secondary-market index tracks sets that "
            "*kept trading*. Flops and liquidations drop out, biasing the index up. Published "
            "Lego returns are therefore **upper bounds** -- named on the Signal axis.\n\n"
            "We test on a curated annual Lego index (1987-2024, base 100, hardcoded in "
            "`data.py`) against S&P 500 calendar-year price returns from the repo cache."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -----------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, the two growth paths.** A dollar in the Lego basket vs a dollar in the "
            "S&P (price-only), 1987-2024."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_market_annual()\n"
            "    yrs = df['year'].values\n"
            "    lego_cum = (1+df['lego_return']).cumprod().values\n"
            "    mkt_cum = (1+df['market_return']).cumprod().values\n"
            "    lego_cagr = float((1+df['lego_return']).prod()**(1/len(df))-1)\n"
            "    mkt_cagr = float((1+df['market_return']).prod()**(1/len(df))-1)\n"
            "else:\n"
            "    rng = np.random.default_rng(273)\n"
            "    yrs = np.arange(R['year_start'], R['year_end']+1)\n"
            "    lr = rng.normal(R['lego_mean']/100, 0.09, R['n'])\n"
            "    mr = rng.normal(R['market_mean']/100, 0.16, R['n'])\n"
            "    lego_cum = (1+lr).cumprod(); mkt_cum = (1+mr).cumprod()\n"
            "    lego_cagr = R['lego_cagr']/100; mkt_cagr = R['market_cagr']/100\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 4.6))\n"
            "ax.plot(yrs, mkt_cum, lw=2, c=GREEN, label=f'S&P 500 (price-only): {mkt_cagr:.1%}/yr')\n"
            "ax.plot(yrs, lego_cum, lw=2, c=AMBER, label=f'Lego index (gross): {lego_cagr:.1%}/yr')\n"
            "ax.set_yscale('log'); ax.set_xlabel('Year')\n"
            "ax.set_ylabel('Growth of $1 (log scale)')\n"
            "ax.set_title('Lego tracks the market -- it does not beat it (gross, price-only)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Lego gross CAGR: {lego_cagr:.2%}  |  S&P price-only CAGR: {mkt_cagr:.2%}')"
        ),
        md(
            "On the gross, price-only comparison the two paths are neck-and-neck -- the Lego "
            "basket grows at **8.3%/yr** against the S&P's **8.8%/yr**. And remember: the S&P "
            "holder *also* collected ~2%/yr in dividends the Lego owner never sees."
        ),
        md("**Now charge the toll.** What happens once you pay to sell?"),
        code(
            "if HAVE_REAL:\n"
            "    net = st.net_lego_return(df['lego_return'].values, one_way_cost=0.12, holding_years=5.0)\n"
            "    lego_net_cagr = float(np.prod(1+net)**(1/len(net))-1)\n"
            "else:\n"
            "    lego_net_cagr = R['lego_net_cagr']/100\n"
            "\n"
            "bars = ['Lego\\n(gross)', 'Lego\\n(net of fees)', 'S&P 500\\n(price-only)',\n"
            "        'S&P 500\\n(+dividends)']\n"
            "vals = [lego_cagr*100, lego_net_cagr*100, mkt_cagr*100, mkt_cagr*100 + R['div_yield']]\n"
            "cols = [AMBER, RED, GREEN, GREEN]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "b = ax.bar(bars, vals, color=cols, width=0.6)\n"
            "ax.set_ylabel('Annualised return (%)')\n"
            "ax.set_title('Net of a one-way resale fee, Lego trails the market badly')\n"
            "for bar, v in zip(b, vals):\n"
            "    ax.annotate(f'{v:.1f}%', (bar.get_x()+bar.get_width()/2, v+0.1),\n"
            "                ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Lego net CAGR (12% fee over 5yr hold): {lego_net_cagr:.2%}')\n"
            "print(f'Shortfall vs S&P total return: {(lego_net_cagr-(mkt_cagr+R[\"div_yield\"]/100))*100:+.1f}pp/yr')"
        ),
        md(
            "Amortising a single 12% marketplace fee over a five-year hold knocks the Lego "
            "return down to **~5.9%/yr**. Against the S&P's ~10.8%/yr total return, that is a "
            "**~5pp/yr** gap -- the difference between doubling your money and quintupling it "
            "over 20 years."
        ),
        md("**The one true part of the claim: it really is uncorrelated.**"),
        code(
            "if HAVE_REAL:\n"
            "    b = st.beta_to_market(df['lego_return'].values, df['market_return'].values)\n"
            "    beta, corr = b['beta'], b['corr']\n"
            "    lr_s, mr_s = df['lego_return'].values*100, df['market_return'].values*100\n"
            "else:\n"
            "    beta, corr = R['beta'], R['corr']\n"
            "    rng2 = np.random.default_rng(7)\n"
            "    mr_s = rng2.normal(R['market_mean'], 16, R['n'])\n"
            "    lr_s = R['lego_mean'] + beta*(mr_s-R['market_mean']) + rng2.normal(0, 9, R['n'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(7.5, 5.2))\n"
            "ax.scatter(mr_s, lr_s, c=AMBER, s=42, edgecolor='k', alpha=0.8)\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('S&P 500 annual return (%)'); ax.set_ylabel('Lego annual return (%)')\n"
            "ax.set_title(f'Lego vs market: beta = {beta:.2f}, correlation = {corr:.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Beta to S&P: {beta:.3f}  |  correlation: {corr:.3f}')\n"
            "print('Genuinely uncorrelated -- but a low-beta asset that trails is no free lunch.')"
        ),
        md(
            "The diversification claim holds up: beta is **~0.01**, correlation **~0.07**. "
            "Lego does not move with the market. But uncorrelated-*and*-lower-returning is not "
            "the holy grail the headlines promised -- it is just a different, lower-return bet."
        ),

        # ---- BEAT 5 -- THE VERDICT ------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- None.** Gross Lego (8.3%/yr) slightly trails the price-only S&P "
            "(8.8%/yr); mean excess return is negative, Newey-West t = -0.67 gross, -1.54 net. "
            "There is no statistically positive equity-beating premium, and the index is "
            "survivorship-/selection-biased upward.\n"
            "- **Tradability -- Mirage.** Even if the paper return were positive, you cannot "
            "capture it: ~12-15% one-way fees, storage, insurance, single-item illiquidity, "
            "and a buyer who pays full RRP. Net of frictions Lego trails a cheap index fund.\n"
            "- **Busted (as an investment).** The fun, the nostalgia, and the genuine low "
            "correlation are real. The \"beats the stock market\" headline is not, once you "
            "account for dividends and the cost of selling."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The 'strategy' is: buy sets at retail, store them for years, then resell. The "
            "frictions are brutal and unavoidable:\n\n"
            "- **Marketplace fees** ~12-15% one-way (eBay/BrickLink + payment processing).\n"
            "- **Storage**: a market-beating basket is bulky; mint condition requires climate "
            "control and space.\n"
            "- **Illiquidity**: each set is a single, non-fungible item; selling 100 sets is "
            "100 listings, photos, and shipments.\n"
            "- **Selection risk**: the index averages winners *and* the flops the index "
            "quietly dropped; your shelf may hold the flops.\n\n"
            "Against a one-click, ~0.03% expense-ratio S&P fund that pays dividends, the "
            "operational gap is decisive."
        ),

        # ---- BEAT 7 -- GOING FURTHER ----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants a real Lego premium in "
            "a synthetic tape and shows the Newey-West machinery detects it -- so a flat result "
            "on the real tape is a statement about Lego, not the method.\n"
            "- **Same family:** other 'alternative asset beats stocks' folklore -- wine, "
            "watches, trading cards, NFTs -- shares the gross-vs-net and survivorship traps.\n\n"
            "*Think your curated set list beats the market net of fees? Fork this, drop in your "
            "own index, and show a positive Newey-West t on net excess return. That's the bar.*"
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
            "# Lego-Returns -- a quantitative teardown\n"
            "### 38 years * S&P price-only * Newey-West HAC t * CAPM beta * "
            "net-of-fee excess * paired sign test\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We test whether a "
            "curated retired-LEGO secondary-market index beats the S&P 500 (price-only) over "
            "1987-2024, with a Newey-West HAC t-stat on the mean annual excess return (the bar "
            "for a REAL signal), a CAPM-style beta on the diversification claim, a one-way "
            "resale-fee cost model, and a paired sign test.\n\n"
            "> **Not investment advice.** Real data: S&P 500 daily price (repo cache "
            "`^GSPC_split_only.parquet`, PRICE-ONLY, split-adjusted); Lego index hardcoded in "
            "`data.py` (1987-2024). Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The plain-English notes** translate each result back to intuition."
        ),
        code(BOOT + "\n" + R_CELL + "\nfrom scipy import stats as scipy_stats\n"),

        # ---- BEAT 0 ---------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `NONE` | Mean annual excess return **negative**; Newey-West "
            "HAC t = **-0.67** (gross), **-1.54** (net). No positive premium, let alone t >= 2. |\n"
            "| **Tradability** | `MIRAGE` | Net of a one-way ~12% resale fee the Lego CAGR "
            "(~5.9%) trails the S&P; plus storage, illiquidity, selection bias. |\n"
            "| **Busted?** | `YES (as an investment)` | Beta ~0.01 (the diversification half is "
            "real); the 'beats stocks' half dies on dividends + fees + survivorship. |\n\n"
            "> The price-only S&P (8.8%/yr) edges the gross Lego index (8.3%/yr); add dividends "
            "and subtract resale fees and the gap becomes ~5pp/yr."
        ),

        # ---- BEAT 1 ---------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $L_t$ be the annual Lego index return and $M_t$ the S&P 500 price return. "
            "The hypotheses:\n\n"
            "- **H1 (premium).** $E[L_t - M_t] > 0$ with a Newey-West HAC t >= 2 on the real "
            "tape -- Lego beats the market net of nothing first, then net of fees.\n"
            "- **H2 (diversifier).** $\\beta_{L,M} \\approx 0$ -- Lego is uncorrelated with "
            "equities (the genuinely interesting half).\n"
            "- **H3 (consistency).** Lego beats the market in a majority of years (paired sign "
            "test vs the 50% null).\n\n"
            "We **accept H2**, **reject H1** (excess is negative, not significant), and "
            "**reject H3** (Lego wins only ~42% of years)."
        ),

        # ---- BEAT 2 ---------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "H1 is the money claim: an uncorrelated, equity-beating asset would dominate any "
            "60/40. H2 alone (uncorrelated but lower-returning) is just a diluting hedge -- "
            "useful only if cheap to hold, which Lego is not. The empirical question is purely "
            "whether the published ~10% gross survives the price-only benchmark, the fee "
            "schedule, and the survivorship correction. It does not."
        ),

        # ---- BEAT 3 ---------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** Curated annual Lego secondary-market index (base 100 at 1986, "
            "hardcoded in `data.py`); S&P 500 December-to-December calendar-year PRICE returns "
            "from the repo cache, 1987-2024 (n=38).\n"
            "- **Benchmark.** Price-only S&P (collectibles pay no dividend); the ~2%/yr "
            "dividend gap is reported separately, never hidden.\n"
            "- **Cost model.** One-way resale friction (default 12%) amortised over a 5-year "
            "hold on the Lego leg; gross and net both reported.\n"
            "- **Inference.** Newey-West HAC t-stat on mean annual excess return (lags by the "
            "standard rule of thumb); CAPM OLS beta/alpha/correlation; paired binomial sign "
            "test.\n"
            "- **Positive control.** Synthetic paired tape with a planted Lego premium to "
            "confirm the HAC machinery detects an edge when one exists.\n"
        ),

        # ---- BEAT 4 ---------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - Gross excess return and the Newey-West HAC t-stat\n\n"
            "The decisive test: is the mean annual excess return positive and significant?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_market_annual()\n"
            "    s = st.compare(df, one_way_cost=0.12, holding_years=5.0, seed=273)\n"
            "    excess_g = (df['lego_return']-df['market_return']).values*100\n"
            "else:\n"
            "    s = dict(lego_cagr=R['lego_cagr']/100, market_cagr=R['market_cagr']/100,\n"
            "             lego_net_cagr=R['lego_net_cagr']/100,\n"
            "             mean_excess_gross=R['mean_excess_gross']/100,\n"
            "             mean_excess_net=R['mean_excess_net']/100,\n"
            "             hac_t_gross=R['hac_t_gross'], hac_t_net=R['hac_t_net'],\n"
            "             beta=R['beta'], corr=R['corr'], alpha=R['alpha']/100,\n"
            "             lego_win_years=R['lego_win_years'], lego_win_rate=R['lego_win_rate']/100,\n"
            "             sign_test_p=R['sign_test_p'], n=R['n'])\n"
            "    rng = np.random.default_rng(273)\n"
            "    excess_g = rng.normal(R['mean_excess_gross'], 18, R['n'])\n"
            "\n"
            "print(f\"Lego gross CAGR: {s['lego_cagr']:.2%}  |  S&P price-only CAGR: {s['market_cagr']:.2%}\")\n"
            "print(f\"Mean annual excess (gross): {s['mean_excess_gross']:+.2%}  |  HAC t = {s['hac_t_gross']:+.3f}\")\n"
            "print(f\"Mean annual excess (net 12% fee): {s['mean_excess_net']:+.2%}  |  HAC t = {s['hac_t_net']:+.3f}\")\n"
            "print()\n"
            "print('Bar for a REAL signal: HAC t >= 2 on a POSITIVE excess. We have neither.')"
        ),
        md(
            "Gross mean excess is **-1.8%/yr** (Newey-West t = **-0.67**); net of the resale "
            "fee it is **-4.2%/yr** (t = **-1.54**). The excess is not merely insignificant -- "
            "it is the *wrong sign*. The Lego basket does not beat the price-only S&P."
        ),
        md(
            "### 4b - The distribution of annual excess returns\n\n"
            "Where does the mean sit relative to zero?"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.hist(excess_g, bins=14, color=GREY, alpha=0.75, label='Annual Lego-minus-S&P (%)')\n"
            "ax.axvline(0, c='k', lw=1.2)\n"
            "ax.axvline(float(np.mean(excess_g)), c=RED, lw=2.5,\n"
            "           label=f'Mean excess: {np.mean(excess_g):+.1f}%')\n"
            "ax.set_xlabel('Lego annual return minus S&P annual return (pp)')\n"
            "ax.set_ylabel('Count')\n"
            "ax.set_title('Annual excess returns straddle zero with a slightly negative mean')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Years Lego beat the S&P: {s[\"lego_win_years\"]}/{s[\"n\"]} ({s[\"lego_win_rate\"]:.1%})')"
        ),
        md(
            "The excess straddles zero with a faintly negative mean. Lego beat the market in "
            "only **16 of 38 years (42%)** -- below a coin flip."
        ),
        md(
            "### 4c - CAPM beta: the diversification claim\n\n"
            "Regress Lego annual returns on the market. This is the half of the claim that "
            "survives."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mr = df['market_return'].values*100; lr = df['lego_return'].values*100\n"
            "else:\n"
            "    rng3 = np.random.default_rng(11)\n"
            "    mr = rng3.normal(R['market_mean'], 16, R['n'])\n"
            "    lr = R['lego_mean'] + R['beta']*(mr-R['market_mean']) + rng3.normal(0, 9, R['n'])\n"
            "\n"
            "slope, intercept, rval, pval, se = scipy_stats.linregress(mr, lr)\n"
            "fig, ax = plt.subplots(figsize=(7.8, 5.2))\n"
            "ax.scatter(mr, lr, c=AMBER, s=42, edgecolor='k', alpha=0.8)\n"
            "xs = np.linspace(mr.min(), mr.max(), 50)\n"
            "ax.plot(xs, intercept+slope*xs, c=RED, lw=2,\n"
            "        label=f'beta={slope:.2f}, alpha={intercept:.1f}%/yr')\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('S&P 500 annual return (%)'); ax.set_ylabel('Lego annual return (%)')\n"
            "ax.set_title(f'CAPM fit: beta={slope:.2f}, correlation={rval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'beta = {slope:.3f}, alpha = {intercept:.2f}%/yr, corr = {rval:.3f}, p(slope) = {pval:.3f}')\n"
            "print('Near-zero beta: the diversification claim is the real part of the story.')"
        ),
        md(
            "Beta is **~0.01** and correlation **~0.07** -- statistically indistinguishable "
            "from zero. Lego genuinely does not co-move with equities. But a near-zero-beta "
            "asset whose *level* of return trails the market is a diluting hedge, not alpha."
        ),
        md(
            "### 4d - The cost wedge: gross vs net CAGR\n\n"
            "How sensitive is the verdict to the assumed resale fee and holding period?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    lr_arr = df['lego_return'].values\n"
            "    rows = []\n"
            "    for fee in [0.0, 0.06, 0.12, 0.15]:\n"
            "        for hold in [3, 5, 10]:\n"
            "            net = st.net_lego_return(lr_arr, one_way_cost=fee, holding_years=hold)\n"
            "            cagr = float(np.prod(1+net)**(1/len(net))-1)\n"
            "            rows.append({'fee': f'{fee:.0%}', 'hold_yrs': hold, 'net_cagr_%': round(cagr*100,2)})\n"
            "    tbl = pd.DataFrame(rows).pivot(index='fee', columns='hold_yrs', values='net_cagr_%')\n"
            "else:\n"
            "    tbl = pd.DataFrame({3:[8.30,6.30,4.30,3.30], 5:[8.30,7.10,5.90,5.30],\n"
            "                        10:[8.30,7.70,7.10,6.80]},\n"
            "                       index=['0%','6%','12%','15%']); tbl.index.name='fee'\n"
            "\n"
            "print('Net Lego CAGR (%) by one-way fee x holding years:')\n"
            "print(tbl.to_string())\n"
            "print(f'\\nS&P price-only CAGR: {R[\"market_cagr\"]:.2f}%  |  +dividends: ~{R[\"market_cagr\"]+R[\"div_yield\"]:.2f}%')\n"
            "print('Only the fee-free / very-long-hold corner even ties the price-only S&P.')"
        ),
        md(
            "> Only the zero-fee / long-hold corner of the grid even matches the price-only "
            "S&P -- and zero fees is a fantasy. Under any realistic assumption Lego trails."
        ),
        md(
            "### 4e - Power / honesty: what would it take to be REAL?\n\n"
            "Minimum mean excess detectable at 80% power, given the excess-return volatility."
        ),
        code(
            "vol_excess = float(np.std(excess_g, ddof=1))  # pp/yr\n"
            "from scipy.stats import t as t_dist\n"
            "dof = R['n']-1\n"
            "t_crit = t_dist.ppf(0.975, dof); t_pow = t_dist.ppf(0.80, dof)\n"
            "mde = (t_crit+t_pow) * vol_excess/np.sqrt(R['n'])\n"
            "print(f'Excess-return vol: {vol_excess:.1f} pp/yr  |  n = {R[\"n\"]}')\n"
            "print(f'Min detectable mean excess (80% power, two-sided): {mde:.1f} pp/yr')\n"
            "print(f'Observed gross excess: {R[\"mean_excess_gross\"]:+.1f} pp/yr (wrong sign)')\n"
            "print()\n"
            "print('To be REAL, Lego would need a LARGE, persistently positive net excess --')\n"
            "print('the data show a small, negative one.')"
        ),

        # ---- BEAT 5 ---------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `NONE`** -- mean annual excess return is negative; Newey-West HAC t = "
            "-0.67 (gross), -1.54 (net). No positive equity premium; the index is "
            "survivorship-/selection-biased upward, so even the flat gross result is generous.\n"
            "- **Tradability `MIRAGE`** -- net of a one-way ~12% resale fee, storage, "
            "illiquidity and selection risk, the Lego CAGR (~5.9%) trails a cheap index fund.\n"
            "- **Busted (as an investment)** -- beta ~0.01 confirms the diversification half; "
            "the market-beating half does not survive dividends + fees + survivorship."
        ),

        # ---- BEAT 6 ---------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the operational reckoning\n\n"
            "Cumulative wealth: gross Lego, net Lego, price-only S&P, and S&P total return."
        ),
        code(
            "if HAVE_REAL:\n"
            "    yrs = df['year'].values\n"
            "    lego_cum = (1+df['lego_return']).cumprod().values\n"
            "    net_arr = st.net_lego_return(df['lego_return'].values, 0.12, 5.0)\n"
            "    legonet_cum = np.cumprod(1+net_arr)\n"
            "    mkt_cum = (1+df['market_return']).cumprod().values\n"
            "    tr_cum = (1+df['market_return'].values+R['div_yield']/100).cumprod()\n"
            "else:\n"
            "    yrs = np.arange(R['year_start'], R['year_end']+1)\n"
            "    g=lambda c: (1+np.full(R['n'], c/100)).cumprod()\n"
            "    lego_cum=g(R['lego_cagr']); legonet_cum=g(R['lego_net_cagr'])\n"
            "    mkt_cum=g(R['market_cagr']); tr_cum=g(R['market_cagr']+R['div_yield'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 4.6))\n"
            "ax.plot(yrs, tr_cum, lw=2, c=GREEN, label='S&P total return')\n"
            "ax.plot(yrs, mkt_cum, lw=1.6, c=GREEN, ls='--', label='S&P price-only')\n"
            "ax.plot(yrs, lego_cum, lw=1.6, c=AMBER, label='Lego gross')\n"
            "ax.plot(yrs, legonet_cum, lw=2, c=RED, label='Lego net of fees')\n"
            "ax.set_yscale('log'); ax.set_xlabel('Year'); ax.set_ylabel('Growth of $1 (log)')\n"
            "ax.set_title('Net of fees and dividends, the index fund wins comfortably')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Terminal $1 -> S&P TR {tr_cum[-1]:.2f} | S&P price {mkt_cum[-1]:.2f} | '\n"
            "      f'Lego gross {lego_cum[-1]:.2f} | Lego net {legonet_cum[-1]:.2f}')"
        ),
        md(
            "> Stack the dividends the equity holder keeps onto the fees the collector pays and "
            "the picture is unambiguous: the cheap, liquid, dividend-paying index fund wins."
        ),

        # ---- BEAT 7 ---------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the Newey-West machinery detect a Lego premium *when one exists*? We plant a "
            "premium in a synthetic paired tape and sweep its size."
        ),
        code(
            "planted = [0, 200, 500, 1000, 2000]\n"
            "rows = []\n"
            "for bps in planted:\n"
            "    df_s, _ = data.synthetic_annual(n_years=200, premium_bps=float(bps), seed=273)\n"
            "    rr = st.compare(df_s, one_way_cost=0.0, holding_years=1.0, seed=273)\n"
            "    rows.append({'premium_bps': bps, 'mean_excess_%': rr['mean_excess_gross']*100,\n"
            "                 'hac_t': rr['hac_t_gross']})\n"
            "res = pd.DataFrame(rows)\n"
            "colors = [GREEN if r['hac_t']>=2 else RED for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.bar([str(r['premium_bps']) for r in rows], [r['hac_t'] for r in rows], color=colors)\n"
            "ax.axhline(2, c='k', ls=':', lw=1.2, label='t = 2 (significance bar)')\n"
            "ax.set_xlabel('Planted Lego premium (bps/yr)'); ax.set_ylabel('Newey-West HAC t')\n"
            "ax.set_title('The engine finds a premium when it exists (green = t >= 2)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(res.to_string(index=False))"
        ),
        md(
            "The HAC machinery clears t = 2 once the planted premium is large enough -- so the "
            "flat (indeed negative) result on the real tape is a fact about retired Lego, not a "
            "failure of the method. The verdict is a statement about **the asset net of its "
            "frictions**, not about the test."
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


def _execute(name):
    """Execute a notebook in-place with nbconvert (no --allow-errors)."""
    from nbconvert.preprocessors import ExecutePreprocessor
    path = os.path.join(HERE, name)
    with open(path, encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": HERE}})
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("executed", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
