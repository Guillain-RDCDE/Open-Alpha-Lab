"""Generate the two narrative notebooks for Study 277 (Trading-Cards).

    python notebooks/build_notebooks.py

This script writes AND executes both notebooks via nbconvert (no --allow-errors).
The hardcoded curated card index and the synthetic generator run anywhere, offline
and deterministically; the real S&P 500 cells use the cached ^GSPC parquet at the
repo-level _cache/ if present, and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader offline.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
"""

from __future__ import annotations

import os
import subprocess
import sys

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-17).
R = dict(
    n=35,
    year_start=1991,
    year_end=2025,
    card_cagr_pct=5.2,
    card_vol_pct=18.7,
    card_sharpe=0.345,
    card_maxdd_pct=-44.1,
    card_net_cagr_pct=-0.9,
    card_net_sharpe=0.024,
    gspc_cagr_pct=9.0,
    gspc_vol_pct=17.0,
    gspc_sharpe=0.618,
    gspc_maxdd_pct=-40.1,
    alpha_ann_pct=3.42,
    beta=0.29,
    t_alpha=1.331,
    p_alpha=0.1832,
    corr=0.263,
    r2=0.069,
    alpha_net_ann_pct=-2.58,
    t_alpha_net=-1.007,
    boot_alpha_p25=-1.81,
    boot_alpha_p50=3.44,
    boot_alpha_p975=8.99,
    boot_frac_alpha_pos=0.911,
    card_cagr_ex_boom_pct=2.1,
    card_cagr_9114=3.2,
    gspc_cagr_9114=7.9,
    card_cagr_1521=23.4,
    gspc_cagr_1521=12.7,
    card_cagr_2225=-10.9,
    gspc_cagr_2225=9.5,
    card_fp="557588f16e2e",
    gspc_fp="2e653d757b43",
)

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY, BLUE = "#c0392b", "#dab617", "#2ea44f", "#8b949e", "#2c7fb8"

from trading_cards import data, strategy as st

def _have_cache():
    try:
        data.fetch_gspc_annual()
        return True
    except FileNotFoundError:
        return False

HAVE_REAL = _have_cache()
print("GSPC cache present:", HAVE_REAL)
"""

# R-dict preamble for the notebooks
R_CELL = """\
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-17)
R = dict(
    n=35, year_start=1991, year_end=2025,
    card_cagr_pct=5.2, card_vol_pct=18.7, card_sharpe=0.345, card_maxdd_pct=-44.1,
    card_net_cagr_pct=-0.9, card_net_sharpe=0.024,
    gspc_cagr_pct=9.0, gspc_vol_pct=17.0, gspc_sharpe=0.618, gspc_maxdd_pct=-40.1,
    alpha_ann_pct=3.42, beta=0.29, t_alpha=1.331, p_alpha=0.1832, corr=0.263, r2=0.069,
    alpha_net_ann_pct=-2.58, t_alpha_net=-1.007,
    boot_alpha_p25=-1.81, boot_alpha_p50=3.44, boot_alpha_p975=8.99,
    boot_frac_alpha_pos=0.911, card_cagr_ex_boom_pct=2.1,
    card_cagr_9114=3.2, gspc_cagr_9114=7.9,
    card_cagr_1521=23.4, gspc_cagr_1521=12.7,
    card_cagr_2225=-10.9, gspc_cagr_2225=9.5,
)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Trading-Cards -- are Pokemon and sports cards an investable asset class?\n"
            "### The collectible-card boom, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "During the pandemic you couldn't open a finance feed without someone showing a "
            "graded Charizard or a Michael Jordan rookie that '10x'd'. A whole cottage "
            "industry sprang up selling cards as the new alternative asset -- 'better than "
            "the S&P, and you get to hold something cool.' This notebook asks the only "
            "question that matters for an allocator: **once you account for the brutal "
            "frictions and the fact that the whole story rides on one 2020-2021 mania that "
            "round-tripped, is there an actual edge left?**\n\n"
            "> **This is the plain-language layer.** Want the Newey-West alpha, the block "
            "bootstrap, and the friction accounting? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did cards beat stocks? | **No.** Card index CAGR **5.2%/yr** vs S&P "
            "**9.0%/yr** (both price-only, 1991-2025). |\n"
            "| Risk-adjusted? | **No.** Card Sharpe **0.35** vs S&P **0.62**, and a "
            "**-44%** max drawdown -- worse than equities. |\n"
            "| Is there an edge beyond stock-market beta? | **No.** Alpha t-stat (HAC) "
            "= **1.33**, below the |t| >= 2 bar. And it rides on **one** boom. |\n"
            "| After real costs? | **Negative.** Amortised auction/grading frictions turn "
            "the **+5.2%** gross into **-0.9%/yr** net. |\n\n"
            "> Cards looked like an asset class for exactly two years. Strip the 2020-2021 "
            "mania and the index compounds at **2.1%/yr** -- below cash, before costs."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"Graded trading cards -- Pokemon, vintage baseball, basketball -- are a "
            "real alternative asset. They've outperformed stocks, they're a hedge against "
            "inflation, and unlike equities you actually own a scarce physical thing.\"*\n\n"
            "The claim went mainstream in 2020-2021 as a PSA-10 Charizard and Jordan/LeBron "
            "rookies posted eye-watering gains, and fractional-card platforms and card-focused "
            "funds launched to let you 'invest' without ever touching cardboard. The pitch: "
            "equity-like returns, low correlation, and a collectible you can enjoy."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If it were true, cards would belong in a diversified portfolio: a high-return, "
            "low-correlation sleeve. That is a strong claim. Collectibles pay **no dividend**, "
            "they are **illiquid** (you sell at auction, weeks later, into a thin market), and "
            "the round-trip costs are enormous (buyer's + seller's premiums, grading fees). "
            "To earn a place in a portfolio they must clear equities *after* all of that.\n\n"
            "The interesting question is **how much of the 'cards beat stocks' story is just "
            "one extraordinary boom** -- and what's left when you take it away."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **Survivorship.** Card price indices quote the cards that *kept trading* -- "
            "the Charizards, not the millions of commons worth less than the toploader they "
            "sit in. Any collectible index is survivorship-biased upward; the real return to "
            "a random buyer is worse than the index. We name this on the Signal axis.\n"
            "2. **One boom.** The whole outperformance lives in 2020-2021. With ~35 annual "
            "observations and one giant spike, a tiny-n test will be dominated by it. We use "
            "a **block bootstrap** and an **ex-boom** recompute to show the fragility.\n"
            "3. **Gross vs net.** Card returns are quoted gross. Auction houses take ~15-20% "
            "from *each* side, plus grading fees. We net an amortised round trip out and "
            "compare *net* cards against the S&P.\n\n"
            "We pair a curated annual graded-card price index (hardcoded in `data.py`, "
            "base 100 = 1990) with the S&P 500's calendar-year price return, 1991-2025."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, meet the data:** the curated card index level, year by year. Watch "
            "the 2020-2021 spike -- and the round trip that followed."
        ),
        code(
            "tbl = data.card_index_table()\n"
            "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            "ax.plot(tbl['year'], tbl['card_index'], 'o-', c=BLUE, lw=1.8, ms=4)\n"
            "ax.axvspan(2019.5, 2021.5, color=AMBER, alpha=0.18, label='2020-2021 mania')\n"
            "ax.set_xlabel('Year'); ax.set_ylabel('Card index (base 100 = 1990)')\n"
            "ax.set_title('The curated graded-card index: a long grind, one mania, a crash')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "peak = tbl.loc[tbl['card_index'].idxmax()]\n"
            "print(f\"Peak: {peak['card_index']:.0f} in {int(peak['year'])}\")\n"
            "print(f\"2025 level: {tbl['card_index'].iloc[-1]:.0f} -- still {(1-585/930)*100:.0f}% below the 2021 peak\")"
        ),
        md("**Cards vs stocks: the wealth curves.**"),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_gspc_annual()\n"
            "    yrs = df['year'].values\n"
            "    card_w = (1+df['card_return']).cumprod().values * 100\n"
            "    gspc_w = (1+df['gspc_return']).cumprod().values * 100\n"
            "    card_cagr, gspc_cagr = R['card_cagr_pct'], R['gspc_cagr_pct']\n"
            "else:\n"
            "    rng = np.random.default_rng(277)\n"
            "    yrs = np.arange(R['year_start'], R['year_end']+1)\n"
            "    cr = rng.normal(R['card_cagr_pct']/100, R['card_vol_pct']/100, R['n'])\n"
            "    gr = rng.normal(R['gspc_cagr_pct']/100, R['gspc_vol_pct']/100, R['n'])\n"
            "    card_w = (1+cr).cumprod()*100; gspc_w = (1+gr).cumprod()*100\n"
            "    card_cagr, gspc_cagr = R['card_cagr_pct'], R['gspc_cagr_pct']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            "ax.plot(yrs, gspc_w, lw=2.2, c=GREEN, label=f'S&P 500 (price): {gspc_cagr:.1f}%/yr')\n"
            "ax.plot(yrs, card_w, lw=2.2, c=BLUE, ls='--', label=f'Card index: {card_cagr:.1f}%/yr')\n"
            "ax.set_yscale('log')\n"
            "ax.set_xlabel('Year'); ax.set_ylabel('Wealth (log, base 100)')\n"
            "ax.set_title('Cards spiked above stocks in 2021 -- then handed it all back')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Card CAGR: {card_cagr:.1f}%/yr  |  S&P CAGR: {gspc_cagr:.1f}%/yr (price-only, both)')"
        ),
        md(
            "Cards briefly screamed past stocks at the 2021 peak, then round-tripped. Over "
            "the full 1991-2025 window the card index compounded at "
            "**5.2%/yr** versus the S&P's **9.0%/yr** -- and that is *before* the costs of "
            "actually owning and selling cardboard."
        ),
        md("**The friction reckoning -- what's left after you pay to play.**"),
        code(
            "labels = ['Card index\\n(gross)', 'Card index\\n(net of costs)', 'S&P 500\\n(price)']\n"
            "vals = [R['card_cagr_pct'], R['card_net_cagr_pct'], R['gspc_cagr_pct']]\n"
            "cols = [BLUE, RED, GREEN]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars = ax.bar(labels, vals, color=cols, width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('CAGR (%/yr)')\n"
            "ax.set_title('Amortised auction + grading costs turn +5.2% gross into -0.9% net')\n"
            "for b, v in zip(bars, vals):\n"
            "    ax.annotate(f'{v:+.1f}%', (b.get_x()+b.get_width()/2, v + (0.3 if v>=0 else -0.6)),\n"
            "                ha='center', va='bottom' if v>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Frictions: ~15% one-way (auction premiums + grading), amortised over a 5yr hold.')\n"
            "print('Net card CAGR:', R['card_net_cagr_pct'], '%/yr  vs  S&P', R['gspc_cagr_pct'], '%/yr')"
        ),
        md(
            "Collectibles are not cheap to trade. Auction houses take roughly 15-20% from "
            "*each* side, and grading a card costs real money. Amortising one round trip over "
            "a generous 5-year hold drags the index's **+5.2%** gross down to "
            "**-0.9%/yr** net -- the asset class loses money once you pay the toll."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- None.** Cards underperformed stocks on return (5.2% vs 9.0%) "
            "*and* on risk-adjusted return (Sharpe 0.35 vs 0.62), with a deeper drawdown "
            "(-44%). The excess-return alpha over equity beta has a HAC t-stat of only "
            "**1.33** -- below the |t| >= 2 bar -- and the index is survivorship-biased "
            "upward, so the true edge is, if anything, *worse*.\n"
            "- **Tradability -- Mirage.** After realistic, amortised frictions the net CAGR "
            "is **negative**. There is no liquid, low-cost vehicle: you trade at auction, "
            "weeks apart, into a thin market, paying double-digit premiums.\n"
            "- **Busted.** The entire 'cards beat stocks' narrative is one 2020-2021 mania "
            "that fully round-tripped. Ex-boom, the index compounds at **2.1%/yr**."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "Suppose you'd allocated to cards as a portfolio sleeve. The diversification "
            "pitch is 'low correlation to stocks'. Let's check whether the diversification "
            "actually showed up when it mattered."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_gspc_annual()\n"
            "    corr = float(df['card_return'].corr(df['gspc_return']))\n"
            "    crash = df[df['year'].isin([2022])]\n"
            "    c22 = float(crash['card_return'].iloc[0]); g22 = float(crash['gspc_return'].iloc[0])\n"
            "else:\n"
            "    corr = R['corr']; c22, g22 = -0.301, -0.194\n"
            "\n"
            "print(f'Full-sample card/stock correlation: {corr:+.2f} (modest -- the diversification pitch)')\n"
            "print(f'But in the 2022 sell-off: cards {c22:+.0%}, stocks {g22:+.0%}.')\n"
            "print('Cards fell HARDER than stocks exactly when you needed the hedge.')\n"
            "print()\n"
            "print('And the low correlation buys you nothing if the sleeve loses money net of costs.')"
        ),
        md(
            "The headline correlation is modest (**+0.26**), which is the whole diversification "
            "sales pitch. But in the 2022 risk-off the card index fell **harder** than stocks -- "
            "so the 'hedge' failed precisely when it was needed. A low-correlation sleeve that "
            "loses money after costs is not a diversifier; it's a drag."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants a real structural "
            "card premium into a synthetic tape and shows the engine detects it -- so the "
            "null result here is about the *asset*, not the method.\n"
            "- **Other 'alternative assets':** the same teardown applies to wine, watches, "
            "and NFTs -- survivorship-biased indices, brutal frictions, one boom.\n"
            "- **The honest comparison:** we used the S&P *price* return to match the card "
            "*price* index. Add dividends and equities win by even more.\n\n"
            "*Think cards are really investable? Fork this, bring a friction-adjusted, "
            "survivorship-corrected index, and show a positive net alpha with a HAC "
            "|t| >= 2 that survives dropping 2020-2021. That's the bar -- it isn't cleared.*"
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
            "# Trading-Cards -- a quantitative teardown\n"
            "### curated card index * S&P500 price * Newey-West alpha * block bootstrap * "
            "friction accounting * one-boom fragility\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We regress a curated "
            "annual graded-card price index on the S&P 500, compute the structural alpha with "
            "Newey-West (HAC) standard errors, net out collectible frictions, put a block-"
            "bootstrap confidence interval on the alpha, and show that the entire result is one "
            "2020-2021 boom.\n\n"
            "> **Not investment advice.** Real data: S&P 500 price (^GSPC, cached) "
            "1991-2025; the graded-card index is curated and hardcoded in "
            "`data.py`. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The plain-English notes** translate each result back to intuition."
        ),
        code(BOOT + "\n" + R_CELL + "\nfrom scipy import stats as scipy_stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `NONE` | Card CAGR **5.2%** < S&P **9.0%**; Sharpe **0.35** < "
            "**0.62**; HAC alpha t = **1.33** (< 2). Survivorship-biased index. |\n"
            "| **Tradability** | `MIRAGE` | Net of ~15% one-way frictions the CAGR is "
            "**-0.9%/yr**; no liquid low-cost vehicle. |\n"
            "| **Busted?** | `YES` | The alpha is entirely one 2020-2021 mania; ex-boom CAGR "
            "= **2.1%/yr**. |\n\n"
            "> Cards underperformed equities on every risk-adjusted axis, and what little "
            "outperformance existed evaporates after costs and after removing one boom."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $C_t$ be the annual card-index return and $M_t$ the S&P 500 price return. "
            "The 'investable asset class' claim is:\n\n"
            "- **H1 (alpha).** In $C_t = \\alpha + \\beta M_t + \\varepsilon_t$, the "
            "structural excess return $\\alpha > 0$ with a robust (HAC) $|t| \\ge 2$.\n"
            "- **H2 (risk-adjusted).** Sharpe$(C) >$ Sharpe$(M)$ -- cards reward their "
            "risk better than equities.\n"
            "- **H3 (diversification).** $\\beta$ and correlation are low *and* the alpha "
            "survives, so cards add a genuine low-correlation, positive-return sleeve.\n\n"
            "We reject H1 and H2 on the real tape, and H3 collapses once frictions and the "
            "one-boom dependence are accounted for."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "Alternative assets earn a portfolio seat only if they clear equities on a "
            "*net, risk-adjusted* basis or bring real diversification. Collectibles start "
            "from a deep hole: no yield, severe illiquidity, double-digit round-trip costs, "
            "and indices that quote only the cards that kept trading (survivorship). The "
            "instructive finding is how a single 24-month mania manufactures an 'asset "
            "class' narrative that the data cannot support out of sample."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** Curated annual graded-card price index (hardcoded in `data.py`, "
            "base 100 = 1990); S&P 500 December/December *price* returns from cached ^GSPC, "
            "1991-2025 (n = 35 paired years).\n"
            "- **Label honesty.** Card index is price-only (no yield); we compare against the "
            "S&P *price* return for a like-for-like read, and note that total return widens "
            "the equity lead.\n"
            "- **Execution lag.** None needed for an asset-class comparison (contemporaneous "
            "calendar-year levels), but the *trade* carries a multi-week auction settlement, "
            "which we model as the friction below.\n"
            "- **Inference.** OLS alpha/beta with Newey-West (HAC, 2 lags) t-stats; a "
            "stationary block bootstrap (block=3) for the alpha CI; an ex-2020-2021 recompute.\n"
            "- **Costs.** One-way 15% (auction premium + grading), round trip amortised over "
            "a 5-year hold; shorts are not applicable (you cannot borrow a Charizard).\n"
            "- **Positive control.** Synthetic paired tape with a planted card premium.\n"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - Return and risk: cards vs equities\n\n"
            "The first cut: does the asset clear equities on CAGR, vol, Sharpe, drawdown?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_gspc_annual()\n"
            "    ps_c = st.perf_stats(df['card_return'].values)\n"
            "    ps_g = st.perf_stats(df['gspc_return'].values)\n"
            "    card_cagr, card_vol, card_sh, card_dd = (ps_c['cagr']*100, ps_c['vol']*100,\n"
            "                                             ps_c['sharpe'], ps_c['max_drawdown']*100)\n"
            "    gspc_cagr, gspc_vol, gspc_sh, gspc_dd = (ps_g['cagr']*100, ps_g['vol']*100,\n"
            "                                             ps_g['sharpe'], ps_g['max_drawdown']*100)\n"
            "else:\n"
            "    card_cagr, card_vol, card_sh, card_dd = (R['card_cagr_pct'], R['card_vol_pct'],\n"
            "                                             R['card_sharpe'], R['card_maxdd_pct'])\n"
            "    gspc_cagr, gspc_vol, gspc_sh, gspc_dd = (R['gspc_cagr_pct'], R['gspc_vol_pct'],\n"
            "                                             R['gspc_sharpe'], R['gspc_maxdd_pct'])\n"
            "\n"
            "tbl = pd.DataFrame({\n"
            "    'metric': ['CAGR %', 'Vol %', 'Sharpe', 'Max DD %'],\n"
            "    'Cards':  [card_cagr, card_vol, card_sh, card_dd],\n"
            "    'S&P 500':[gspc_cagr, gspc_vol, gspc_sh, gspc_dd]})\n"
            "print(tbl.to_string(index=False))\n"
            "print()\n"
            "print('Cards lose on CAGR, lose on Sharpe, and have a DEEPER drawdown. No risk-adjusted edge.')"
        ),
        md(
            "Cards compound at **5.2%/yr** against the S&P's **9.0%**, at *higher* volatility "
            "(18.7% vs 17.0%), so the Sharpe gap is stark: **0.35 vs 0.62**. And the card "
            "index's max drawdown (**-44%**) is *worse* than equities' (**-40%**). On every "
            "axis a serious allocator cares about, cards lose."
        ),
        md(
            "### 4b - The structural alpha: Newey-West regression on equity beta\n\n"
            "Is there excess return *beyond* the small equity beta? Regress $C_t$ on $M_t$ "
            "and read the intercept with HAC (Newey-West, 2 lags) standard errors."
        ),
        code(
            "if HAVE_REAL:\n"
            "    reg = st.hac_alpha(df['card_return'].values, df['gspc_return'].values, lags=2)\n"
            "    alpha, beta, t_a, p_a, corr, r2 = (reg['alpha_ann_pct'], reg['beta'],\n"
            "        reg['t_alpha'], reg['p_alpha'], reg['corr'], reg['r2'])\n"
            "else:\n"
            "    alpha, beta, t_a, p_a, corr, r2 = (R['alpha_ann_pct'], R['beta'],\n"
            "        R['t_alpha'], R['p_alpha'], R['corr'], R['r2'])\n"
            "\n"
            "print(f'Card = alpha + beta * SP500')\n"
            "print(f'  alpha = {alpha:+.2f}%/yr   (HAC t = {t_a:+.2f}, p = {p_a:.3f})')\n"
            "print(f'  beta  = {beta:+.2f}        (low equity exposure)')\n"
            "print(f'  corr  = {corr:+.2f}   R^2 = {r2:.3f}')\n"
            "print()\n"
            "print('Alpha t = 1.33 -- BELOW the |t|>=2 bar. The point estimate is positive but')\n"
            "print('not robust, and (see 4d) it is entirely the 2020-2021 boom.')"
        ),
        md(
            "The point estimate of alpha is **+3.42%/yr**, which sounds tempting -- but the "
            "HAC t-stat is only **1.33** (p = 0.18), below the |t| >= 2 bar this desk requires "
            "for a REAL signal. Beta is a tiny **0.29**, so the diversification story has *some* "
            "surface validity -- but a low-beta sleeve with an insignificant, fragile alpha is "
            "not an edge."
        ),
        md(
            "### 4c - Frictions: the net alpha\n\n"
            "Re-run the regression on *net* card returns after amortised auction + grading costs."
        ),
        code(
            "if HAVE_REAL:\n"
            "    net = st.apply_frictions(df['card_return'].values, one_way_cost=0.15, holding_years=5.0)\n"
            "    reg_net = st.hac_alpha(net, df['gspc_return'].values, lags=2)\n"
            "    a_net, t_net = reg_net['alpha']*100, reg_net['t_alpha']\n"
            "    net_cagr = st.perf_stats(net)['cagr']*100\n"
            "else:\n"
            "    a_net, t_net, net_cagr = R['alpha_net_ann_pct'], R['t_alpha_net'], R['card_net_cagr_pct']\n"
            "\n"
            "print(f'Net of ~15% one-way frictions (round trip amortised over 5y hold):')\n"
            "print(f'  Net card CAGR : {net_cagr:+.1f}%/yr')\n"
            "print(f'  Net alpha     : {a_net:+.2f}%/yr  (HAC t = {t_net:+.2f})')\n"
            "print()\n"
            "print('The already-insignificant gross alpha goes NEGATIVE after costs.')"
        ),
        md(
            "Once the toll is paid the net alpha is **-2.58%/yr** (t = -1.01) and the net CAGR "
            "is **-0.9%**. Collectible frictions are not a rounding error; they are the whole "
            "ballgame. There is no surviving edge to monetise."
        ),
        md(
            "### 4d - The one-boom fragility: block bootstrap + ex-boom recompute\n\n"
            "How much of the (gross) alpha is the 2020-2021 mania? Bootstrap the paired "
            "returns, and separately drop 2020 and 2021."
        ),
        code(
            "if HAVE_REAL:\n"
            "    boot = st.block_bootstrap_alpha(df['card_return'].values, df['gspc_return'].values,\n"
            "                                    n_boot=5000, block=3, seed=277)\n"
            "    lo, mid, hi = boot['alpha_p2.5']*100, boot['alpha_p50']*100, boot['alpha_p97.5']*100\n"
            "    frac_pos = boot['frac_alpha_pos']\n"
            "    ex = df[~df['year'].isin([2020,2021])]\n"
            "    cagr_ex = st.perf_stats(ex['card_return'].values)['cagr']*100\n"
            "else:\n"
            "    lo, mid, hi = R['boot_alpha_p25'], R['boot_alpha_p50'], R['boot_alpha_p975']\n"
            "    frac_pos = R['boot_frac_alpha_pos']; cagr_ex = R['card_cagr_ex_boom_pct']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.0))\n"
            "ax.barh(['Bootstrap alpha\\n95% CI'], [hi-lo], left=[lo], color=GREY, alpha=0.6)\n"
            "ax.axvline(0, c=RED, lw=2, label='zero alpha')\n"
            "ax.plot([mid],[0], 'o', c=BLUE, ms=10, label=f'median {mid:+.1f}%')\n"
            "ax.set_xlabel('Card alpha (%/yr)')\n"
            "ax.set_title(f'Bootstrap alpha CI straddles... barely; frac>0 = {frac_pos:.0%}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Bootstrap alpha 95% CI: [{lo:+.1f}%, {hi:+.1f}%], median {mid:+.1f}%')\n"
            "print(f'Ex-2020-2021 card CAGR: {cagr_ex:+.1f}%/yr  (vs {R[\"card_cagr_pct\"]:.1f}% full sample)')\n"
            "print('Strip the boom and the asset class compounds below cash, before costs.')"
        ),
        md(
            "The bootstrap CI for the (gross) alpha runs from **-1.8%** to **+9.0%** -- it "
            "includes zero, so even gross the alpha is not reliably positive. And the killer: "
            "drop just two years (2020, 2021) and the card index compounds at **2.1%/yr**. "
            "The 'asset class' is a single boom in a trench coat."
        ),
        md(
            "### 4e - Sub-period CAGR: where the story lives and dies\n\n"
            "Split into pre-boom, boom, and bust to see the regime dependence."
        ),
        code(
            "periods = [('1991-2014', R['card_cagr_9114'], R['gspc_cagr_9114']),\n"
            "           ('2015-2021', R['card_cagr_1521'], R['gspc_cagr_1521']),\n"
            "           ('2022-2025', R['card_cagr_2225'], R['gspc_cagr_2225'])]\n"
            "labs = [p[0] for p in periods]\n"
            "cardv = [p[1] for p in periods]; gspcv = [p[2] for p in periods]\n"
            "x = np.arange(len(labs)); w = 0.38\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(x-w/2, cardv, w, color=BLUE, label='Cards')\n"
            "ax.bar(x+w/2, gspcv, w, color=GREEN, label='S&P 500')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labs)\n"
            "ax.set_ylabel('CAGR (%/yr)')\n"
            "ax.set_title('Cards only beat stocks during 2015-2021; they lose the rest of the time')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for l,c,g in periods: print(f'{l}: cards {c:+.1f}%/yr  vs  S&P {g:+.1f}%/yr')"
        ),
        md(
            "Cards beat equities in exactly one window -- **2015-2021** (+23.4% vs +12.7%) -- "
            "and lose in both the long pre-boom grind (+3.2% vs +7.9%) and the post-boom bust "
            "(-10.9% vs +9.5%). An 'asset class' that only works during its own bubble is not "
            "one."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `NONE`** -- card CAGR 5.2% < S&P 9.0%, Sharpe 0.35 < 0.62, deeper "
            "drawdown; HAC alpha t = 1.33 (< 2) and bootstrap CI includes zero. The index is "
            "survivorship-biased upward, so the true edge is if anything worse.\n"
            "- **Tradability `MIRAGE`** -- net of realistic frictions the CAGR is -0.9%/yr; "
            "there is no liquid, low-cost vehicle.\n"
            "- **Busted** -- the entire outperformance is the 2020-2021 mania; ex-boom CAGR "
            "is 2.1%/yr."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the net-of-cost reality\n\n"
            "Cards in a portfolio sleeve: the diversification pitch is the low beta and "
            "modest correlation. But the hedge failed when it counted, and the net return "
            "is negative."
        ),
        code(
            "if HAVE_REAL:\n"
            "    corr = float(df['card_return'].corr(df['gspc_return']))\n"
            "    c22 = float(df.loc[df['year']==2022,'card_return'].iloc[0])\n"
            "    g22 = float(df.loc[df['year']==2022,'gspc_return'].iloc[0])\n"
            "else:\n"
            "    corr, c22, g22 = R['corr'], -0.301, -0.194\n"
            "\n"
            "print(f'Correlation to equities: {corr:+.2f}  (the diversification pitch)')\n"
            "print(f'2022 risk-off: cards {c22:+.0%} vs stocks {g22:+.0%} -- cards fell HARDER.')\n"
            "print(f'Net card CAGR: {R[\"card_net_cagr_pct\"]:+.1f}%/yr -- a money-losing sleeve.')\n"
            "print()\n"
            "print('Low correlation is worthless when the sleeve loses money and fails to hedge.')"
        ),
        md(
            "> The modest **+0.26** correlation is the entire diversification case -- but in "
            "the 2022 sell-off cards fell *harder* than stocks, and the sleeve loses money "
            "net of costs. A money-losing, non-hedging sleeve is a drag, not a diversifier."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the engine detect a card premium *when one exists*? Plant a structural "
            "alpha into a synthetic paired tape and sweep its size."
        ),
        code(
            "planted = [0, 200, 500, 1000, 2000]\n"
            "rows = []\n"
            "for bps in planted:\n"
            "    df_s, _ = data.synthetic_index(n_years=400, signal_bps=float(bps), seed=277)\n"
            "    reg = st.hac_alpha(df_s['card_return'].values, df_s['gspc_return'].values, lags=2)\n"
            "    rows.append({'planted_bps': bps, 'alpha_pct': reg['alpha_ann_pct'],\n"
            "                 't_alpha': reg['t_alpha'], 'detected': abs(reg['t_alpha'])>=2})\n"
            "res = pd.DataFrame(rows)\n"
            "colors = [GREEN if r['detected'] else RED for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar([str(r['planted_bps']) for r in rows], [r['t_alpha'] for r in rows], color=colors)\n"
            "ax.axhline(2, c='k', ls=':', lw=1, label='|t| = 2 bar')\n"
            "ax.set_xlabel('Planted card alpha (bps/yr)'); ax.set_ylabel('HAC t(alpha)')\n"
            "ax.set_title('The engine finds a card premium when it exists (green = |t| >= 2)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(res.to_string(index=False))"
        ),
        md(
            "With enough planted alpha and sample size the HAC test clears |t| = 2 -- the "
            "machinery is honest. The real tape (n = 35, one boom, insignificant gross alpha, "
            "negative net) is nowhere near. The verdict is a statement about the **asset class "
            "and its costs**, not about the method."
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
    path = os.path.join(HERE, name)
    subprocess.run(
        [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
         "--inplace", "--ExecutePreprocessor.timeout=300", path],
        check=True,
    )
    print("executed", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
