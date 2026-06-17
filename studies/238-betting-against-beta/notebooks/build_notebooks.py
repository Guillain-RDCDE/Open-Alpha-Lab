"""Generate the two narrative notebooks for Study 238 (Betting-Against-Beta).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). Synthetic figures
run anywhere, offline and deterministic; the real-panel cells use the cached yfinance parquets
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
    n_months=178, year_start=2011, year_end=2025,
    n_tickers=96,
    bab_mean=4.44, bab_vol=13.93, bab_sharpe=0.319, bab_t=1.386,
    bab_hit=55.6, bab_dd=-27.1,
    low_beta_mean=13.86, high_beta_mean=19.88, mkt_mean=14.08,
    avg_beta_low=0.641, avg_beta_high=1.215,
    avg_leverage=1.57, avg_deleverage=0.82,
    fp_prices="11f0bf581250", fp_spy="482009a8da65",
    annual={
        2011: 31.3, 2012: 5.2, 2013: 5.8, 2014: 13.8, 2015: -5.5,
        2016: 2.1, 2017: 4.1, 2018: -9.4, 2019: 20.3, 2020: -17.7,
        2021: 10.3, 2022: 28.6, 2023: -25.9, 2024: 9.7, 2025: -3.3,
    }
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

from betting_against_beta import data, strategy as st

def _have_cache():
    import os
    study = os.path.abspath("..")
    return (os.path.exists(os.path.join(study, "_cache", "bab_prices.parquet"))
            and os.path.exists(os.path.join(study, "_cache", "bab_spy.parquet")))

HAVE_REAL = _have_cache()
print("yfinance cache present:", HAVE_REAL)

if HAVE_REAL:
    prices, spy = data.fetch_panel()
    result = st.bab_portfolio(prices, spy, beta_window=252, min_stocks=20)
    s_bab  = st.summary(result["bab_ret"])
    s_low  = st.summary(result["low_beta_ret"])
    s_high = st.summary(result["high_beta_ret"])
    s_mkt  = st.summary(result["mkt_ret"])
    print(f"N months: {s_bab['n']} | BAB mean: {s_bab['mean']*100:+.2f}%/yr"
          f" | HAC t: {s_bab['tstat']:+.3f}")
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Betting-Against-Beta -- does leveraging low-beta print money?\n"
            "### Frazzini-Pedersen (2014): the leverage-constraint premium, tested honestly\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Survivorship--biased%3F: Named](https://img.shields.io/badge/Survivorship--biased%3F-Named-8b949e?style=flat-square)\n\n"
            "In 2014, Andrea Frazzini and Lasse Pedersen published a striking idea: because "
            "many investors cannot (or will not) use leverage, they crowd into high-beta stocks "
            "to get the market exposure they want. That crowding inflates high-beta prices and "
            "depresses their future returns. The other side of the trade -- buying low-beta "
            "stocks with leverage -- should earn an above-market risk-adjusted return. "
            "They called it **Betting Against Beta (BAB)**.\n\n"
            "We test that claim on a large-cap S&P 500 survivor panel using yfinance daily "
            "prices, naming the survivorship bias and the leverage assumption up front.\n\n"
            "> **This is the plain-language layer.** Want the rolling betas, bootstrap CIs, "
            "and year-by-year breakdown? See the companion "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does BAB earn a premium? | **Weakly yes** in theory and on broad universes, "
            f"but HAC *t* = **+{R['bab_t']:.2f}** on our 15-year survivor panel -- below "
            "the |t|>=2 bar. |\n"
            "| Is the low-beta leg cheap after leveraging? | **Not clear.** The low-beta leg "
            f"earns **{R['low_beta_mean']:.1f}%/yr** vs SPY at **{R['mkt_mean']:.1f}%/yr** "
            "-- before financing costs. |\n"
            "| Is it tradable? | **Fragile.** Leverage of ~1.6x is required; financing "
            f"costs are not modelled. The 2023 reversal was **{R['annual'][2023]:.1f}%**. |\n"
            "| Is there a survivorship problem? | **Yes, and we name it.** Failed high-beta "
            "stocks (a natural short candidate) have been deleted from the universe. |\n\n"
            "> The BAB premium is real in the academic literature on broad, unleveraged-free "
            "universes. On 96 large-cap survivors with a leverage cap, the signal is "
            "statistically marginal and the practical implementation is fraught."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim\n\n"
            "> *\"Leverage-constrained investors hold too many high-beta stocks. Those stocks "
            "are overpriced and earn low future returns. Low-beta stocks are underpriced. "
            "A portfolio long leveraged low-beta and short deleveraged high-beta -- beta-neutral "
            "by construction -- earns a positive return: Betting Against Beta.\"*\n\n"
            "-- Frazzini & Pedersen (2014), *Journal of Financial Economics*\n\n"
            "The intuition is elegant: the CAPM predicts that all stocks should lie on the "
            "Security Market Line (higher beta = higher expected return). Black (1972) showed "
            "that if borrowing is restricted, the line is **flatter** than CAPM predicts. "
            "Frazzini-Pedersen operationalise this: buy low-beta with leverage, sell "
            "high-beta with deleverage, hold a zero-beta portfolio."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "If true, the BAB factor is a source of alpha uncorrelated with the broad market. "
            "It has been incorporated into multi-factor models, risk-parity funds, and "
            "volatility-targeting strategies. AQR (Frazzini and Pedersen's employer) runs "
            "billions of dollars in BAB-adjacent strategies.\n\n"
            "The key questions for us are:\n"
            "1. **Does it survive on a realistic large-cap panel?**\n"
            "2. **What leverage assumption is needed -- and is it realistic?**\n"
            "3. **What is the survivorship bias?**"
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 -- How would we even know?\n\n"
            "Three disciplines keep us honest:\n\n"
            "1. **Roll betas on past data only.** The rolling 252-day beta is computed on "
            "a trailing window -- no future information leaks into the portfolio formation.\n"
            "2. **State the leverage cap.** The original paper assumes unconstrained leverage. "
            "We cap at 2.0x and report the average leverage actually applied.\n"
            "3. **Name the survivorship bias.** The universe is the *current* S&P 500 projected "
            "backwards. Failed companies -- which tend to be high-beta -- are absent. This "
            "biases the short leg's contribution downward (the 'bad high-beta' names we'd short "
            "are not in our sample)."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 -- The teardown\n\n"
            "**First, does the BAB engine work when the effect is planted?**"
        ),
        code(
            "# Synthetic positive control: plant a known BAB premium and verify recovery.\n"
            "premiums = [0.0, 0.04, 0.08, -0.04]\n"
            "rows = []\n"
            "for prem in premiums:\n"
            "    sr, mkt, truth = data.synthetic_panel(n_stocks=100, n_days=2500,\n"
            "                                           bab_premium=prem, seed=238)\n"
            "    prices_s = (1 + sr).cumprod() * 100\n"
            "    spy_s = (1 + mkt).cumprod() * 100\n"
            "    res = st.bab_portfolio(prices_s, spy_s, beta_window=252, min_stocks=10)\n"
            "    if res.empty:\n"
            "        rows.append({'premium': prem, 'bab_mean_%': float('nan'), 't': float('nan')})\n"
            "        continue\n"
            "    s = st.summary(res['bab_ret'])\n"
            "    rows.append({'premium': prem, 'bab_mean_%': s['mean']*100, 't': s['tstat']})\n"
            "ctrl = pd.DataFrame(rows)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if r > 0.5 else (RED if r < -0.5 else GREY)\n"
            "       for r in ctrl['bab_mean_%'].fillna(0)]\n"
            "ax.bar(ctrl['premium'].astype(str), ctrl['bab_mean_%'].fillna(0), color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted BAB premium'); ax.set_ylabel('BAB mean return (%/yr)')\n"
            "ax.set_title('Synthetic control: engine finds the effect when planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),
        md(
            "The engine is faithful: it recovers a positive BAB return when a premium is "
            "planted and returns noise when there is none. So the verdict on the real tape "
            "reflects **the market**, not the method."
        ),
        md("**Now the honest test on the real yfinance panel.**"),
        code(
            "if HAVE_REAL:\n"
            "    bab_mean = s_bab['mean']*100\n"
            "    bab_t    = s_bab['tstat']\n"
            "    low_m    = s_low['mean']*100\n"
            "    high_m   = s_high['mean']*100\n"
            "    mkt_m    = s_mkt['mean']*100\n"
            "else:\n"
            "    bab_mean, bab_t = R['bab_mean'], R['bab_t']\n"
            "    low_m, high_m, mkt_m = R['low_beta_mean'], R['high_beta_mean'], R['mkt_mean']\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "\n"
            "# Left: three bars -- low-beta, SPY, high-beta\n"
            "labels = ['Low-beta\\n(long leg)', 'SPY', 'High-beta\\n(short leg)']\n"
            "vals   = [low_m, mkt_m, high_m]\n"
            "cols   = [AMBER, GREY, RED]\n"
            "axes[0].bar(labels, vals, color=cols, width=0.5)\n"
            "axes[0].axhline(mkt_m, ls='--', c=GREY, lw=1, label='SPY')\n"
            "axes[0].set_ylabel('mean annual return (%/yr, unlevered)')\n"
            "axes[0].set_title('Raw leg returns: high-beta wins in raw terms')\n"
            "\n"
            "# Right: year-by-year BAB\n"
            "if HAVE_REAL:\n"
            "    result_plot = result.copy()\n"
            "    result_plot['year'] = result_plot.index.year\n"
            "    annual_bab = result_plot.groupby('year')['bab_ret'].apply(\n"
            "        lambda x: float((1+x).prod()-1))\n"
            "    col_yr = [GREEN if v > 0 else RED for v in annual_bab]\n"
            "    axes[1].bar(annual_bab.index, annual_bab.values*100, color=col_yr)\n"
            "    axes[1].axhline(0, c='k', lw=1)\n"
            "    axes[1].set_xlabel('year'); axes[1].set_ylabel('BAB return (%/yr)')\n"
            "    axes[1].set_title(f'Year-by-year BAB | mean {bab_mean:+.1f}% t={bab_t:+.2f}')\n"
            "else:\n"
            "    yrs = list(R['annual'].keys())\n"
            "    vals_a = list(R['annual'].values())\n"
            "    col_yr = [GREEN if v > 0 else RED for v in vals_a]\n"
            "    axes[1].bar(yrs, vals_a, color=col_yr)\n"
            "    axes[1].axhline(0, c='k', lw=1)\n"
            "    axes[1].set_title(f'Year-by-year BAB (frozen) | mean {bab_mean:+.1f}%')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'BAB: {bab_mean:+.2f}%/yr | HAC t = {bab_t:+.3f}')"
        ),
        md(
            f"The BAB portfolio earns **+{R['bab_mean']:.1f}%/yr** at HAC *t* = "
            f"**+{R['bab_t']:.2f}** -- below the |t| >= 2 inference bar. The 2023 reversal "
            f"({R['annual'][2023]:.1f}%) shows that the BAB premium can reverse sharply when "
            "momentum/tech stocks surge and low-volatility strategies underperform."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- WEAK.** HAC *t* = +{R['bab_t']:.2f} on the real panel "
            "(|t| < 2). The Frazzini-Pedersen (2014) literature provides strong prior "
            "support, but on 15 years of large-cap survivors the signal is statistically "
            "ambiguous. WEAK rather than NONE because of robust academic replication on "
            "broader, unbiased universes.\n"
            f"- **Tradability -- FRAGILE.** BAB Sharpe = +{R['bab_sharpe']:.2f}, max "
            f"drawdown = {R['bab_dd']:.1f}%, average leverage = {R['avg_leverage']:.2f}x. "
            "Financing costs (the borrowing spread above the risk-free rate) are not "
            "modelled -- they would reduce the premium further. The 2023 reversal of "
            f"{R['annual'][2023]:.1f}% is a sobering reminder.\n"
            "- **Survivorship -- Named.** Failed high-beta firms are absent. The true "
            "BAB return on a complete universe would likely be smaller."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 -- Could you actually trade it?\n\n"
            "The strategy has several practical obstacles:\n\n"
            "1. **Leverage financing costs.** The long leg requires ~1.6x leverage. The "
            "spread between the borrowing rate and the risk-free rate erodes returns "
            "directly. In a high-rate environment this can flip the sign.\n"
            "2. **Short leg mechanics.** Shorting individual stocks requires hard-to-borrow "
            "fees and recall risk. High-beta stocks (growth, tech) are often expensive to "
            "borrow.\n"
            "3. **Crowding.** BAB is a well-known factor; AQR and many systematic funds "
            "run variants. Crowding compresses the premium and amplifies reversals when "
            "deleveraging occurs simultaneously.\n"
            "4. **Monthly rebalancing turnover.** Rolling beta ranks change month by month, "
            "generating meaningful two-sided turnover and trading costs.\n\n"
            "A larger-universe, cost-adjusted, unconstrained implementation might recover "
            "the Frazzini-Pedersen result. On this evidence alone, the verdict is **Fragile**."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Broader universe.** The original paper used all US stocks (1984--2012); "
            "the premium is strongest in small-cap. A Russell 3000 pull would materially "
            "change the inference.\n"
            "- **Cross-asset BAB.** Frazzini-Pedersen (2014) also documents the premium "
            "across bonds, currencies, and commodities -- the leverage-constraint mechanism "
            "is not equity-specific.\n"
            "- **Risk parity.** [Study 144 -- Permanent Portfolio](../../144-permanent-portfolio/) "
            "is a related idea: tilt toward less risky assets (bonds, gold) rather than "
            "levering up low-beta equities.\n"
            "- **Low-volatility investing.** The Blitz-van Vliet low-volatility anomaly "
            "is closely related: sort by realised vol instead of beta. Often similar "
            "results with less leverage.\n\n"
            "*Think the BAB premium survives in large-caps on a realistic cost basis? "
            "Fork this, add financing costs, expand the universe to Russell 3000, and "
            "show t > 2 net of borrow. That is the bar.*"
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
            "# Betting-Against-Beta -- a quantitative teardown\n"
            "### yfinance panel * rolling beta * BAB construction * HAC inference\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Survivorship--biased%3F: Named](https://img.shields.io/badge/Survivorship--biased%3F-Named-8b949e?style=flat-square)\n\n"
            "The quantitative companion to the "
            "[notebook for the curious](01_for_the_curious.ipynb) -- same seven beats, "
            "every claim carrying its standard error. We test Frazzini-Pedersen (2014): "
            "rank by rolling 252-day beta, long leveraged low-beta half, short deleveraged "
            "high-beta half, monthly rebalance, leverage cap 2.0x.\n\n"
            "> **Not investment advice.** Real data: yfinance daily adjusted-close, 96 "
            "S&P 500 large-caps, 2010-2025, as-of 2026-06-16. Methods in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship bias is named:** universe = current S&P 500 projected "
            "backwards. All positive results are upper-bound estimates."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | BAB mean **+{R['bab_mean']:.2f}%/yr**, "
            f"HAC *t* = **+{R['bab_t']:.3f}** (|t| < 2; literature prior upgrades "
            "from NONE). |\n"
            f"| **Tradability** | `FRAGILE` | Sharpe **+{R['bab_sharpe']:.3f}**, "
            f"max DD **{R['bab_dd']:.1f}%**, avg leverage **{R['avg_leverage']:.2f}x** "
            "(financing costs not modelled). |\n"
            "| **Survivorship-biased?** | `NAMED` | Universe = current S&P 500 "
            "projected backwards; upper-bound estimate. |\n\n"
            "> The Frazzini-Pedersen (2014) BAB factor is well-documented on broad, "
            "unconstrained universes. On 15 years of 96 large-cap survivors with a "
            "2.0x leverage cap, the signal is marginal -- consistent with crowding and "
            "constraint-loosening post-publication."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $\\beta_{i,t}$ = trailing 252-day OLS beta of stock $i$ vs SPY at "
            "time $t$. Frazzini-Pedersen (2014) assert:\n\n"
            "- **H1 (signal).** Low-beta stocks earn higher *risk-adjusted* returns than "
            "high-beta stocks. Formally: $\\alpha^\\text{BAB}_t > 0$ where "
            "$\\alpha^\\text{BAB}$ is the return to $L_t \\cdot r^{\\text{low}}_t "
            "- D_t \\cdot r^{\\text{high}}_t$ with $L_t = 1/\\bar\\beta^L$ "
            "(leverage) and $D_t = 1/\\bar\\beta^H$ (deleverage).\n"
            "- **H2 (vs market).** The BAB portfolio is ex-ante beta-neutral, so outperformance "
            "vs zero is true alpha -- not market exposure in disguise.\n"
            "- **H3 (tradable).** The premium survives financing costs and rebalancing.\n\n"
            "We partially confirm H1 (positive mean, t=1.39), partially confirm H2 "
            "(the leverage makes the portfolio approximately beta-neutral), and reject H3 "
            "on this panel (financing costs un-modelled, crowding risk high)."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what? -- the economic stakes\n\n"
            "BAB is one of the most influential ideas in systematic investing since 2010. "
            "AQR's diversified alternatives funds, risk-parity products, and low-volatility "
            "ETFs all embed variants of the BAB logic. If the premium has attenuated -- or "
            "if it never applied cleanly to large-caps -- those fee-laden products are selling "
            "a story the data no longer supports."
        ),

        # ---- BEAT 3 -- PROTOCOL ----------------------------------------------
        md(
            "## 3 -- The protocol\n\n"
            "- **Signal.** Rolling 252-day OLS beta of each stock's daily excess return "
            "on SPY's daily excess return. Risk-free rate = 3%/yr constant.\n"
            "- **Ranking.** Monthly, split all stocks at the median beta.\n"
            "- **Long leg.** Equal-weight low-beta half; leverage = min(1/avg_beta_low, 2.0).\n"
            "- **Short leg.** Equal-weight high-beta half; deleverage = 1/avg_beta_high.\n"
            "- **BAB return.** leverage * low_beta_return - deleverage * high_beta_return.\n"
            "- **Baseline.** SPY total return (not the equal-weight universe) for context.\n"
            "- **Inference.** Newey-West HAC *t* on monthly BAB returns.\n"
            "- **Universe caveat.** 96 large-cap S&P 500 names (yfinance); "
            "survivorship-biased."
        ),

        # ---- BEAT 4 -- TEARDOWN ----------------------------------------------
        md("## 4 -- The teardown"),
        md(
            "### 4a -- Positive control: the BAB engine is a faithful detector\n\n"
            "Sweep the planted BAB premium from negative to positive on a synthetic panel. "
            "The BAB mean should be monotone in the premium."
        ),
        code(
            "premiums = [-0.08, -0.04, 0.0, 0.04, 0.08, 0.12]\n"
            "bab_means, bab_tstats = [], []\n"
            "for prem in premiums:\n"
            "    sr, mkt, _ = data.synthetic_panel(n_stocks=100, n_days=2500,\n"
            "                                       bab_premium=prem, seed=238)\n"
            "    prices_s = (1 + sr).cumprod() * 100\n"
            "    spy_s = (1 + mkt).cumprod() * 100\n"
            "    res = st.bab_portfolio(prices_s, spy_s, beta_window=252, min_stocks=10)\n"
            "    if res.empty:\n"
            "        bab_means.append(float('nan')); bab_tstats.append(float('nan'))\n"
            "        continue\n"
            "    s = st.summary(res['bab_ret'])\n"
            "    bab_means.append(s['mean']*100)\n"
            "    bab_tstats.append(s['tstat'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if m > 0 else RED for m in bab_means]\n"
            "ax.bar([str(p) for p in premiums], bab_means, color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, (m, t) in enumerate(zip(bab_means, bab_tstats)):\n"
            "    ax.text(i, m + (0.3 if m >= 0 else -0.8),\n"
            "            f't={t:+.1f}' if not (t != t) else 'NaN',\n"
            "            ha='center', va='bottom', fontsize=8)\n"
            "ax.set_xlabel('planted BAB premium'); ax.set_ylabel('BAB mean return (%/yr)')\n"
            "ax.set_title('Positive control: BAB mean is monotone in planted premium')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "### 4b -- Rolling beta distribution\n\n"
            "What does the beta distribution look like on the real panel? We expect "
            "a right-skewed distribution centred near 1.0."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets = prices.pct_change().dropna(how='all')\n"
            "    spy_r = spy.pct_change().dropna()\n"
            "    betas = st.rolling_betas(rets.sub(st.RF_DAILY),\n"
            "                              spy_r.sub(st.RF_DAILY))\n"
            "    # Last available snapshot\n"
            "    last_beta = betas.iloc[-1].dropna()\n"
            "    fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "    ax.hist(last_beta, bins=30, color=AMBER, edgecolor='white', alpha=0.8)\n"
            "    ax.axvline(last_beta.median(), c=RED, lw=2, ls='--',\n"
            "               label=f'median={last_beta.median():.2f}')\n"
            "    ax.axvline(1.0, c=GREY, lw=1.5, ls=':', label='beta=1')\n"
            "    ax.set_xlabel('rolling 252-day beta vs SPY')\n"
            "    ax.set_ylabel('count')\n"
            "    ax.set_title('Beta distribution of S&P 500 large-caps (latest snapshot)')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'Median beta: {last_beta.median():.3f}')\n"
            "    print(f'Mean beta:   {last_beta.mean():.3f}')\n"
            "    print(f'Range:       {last_beta.min():.3f} -- {last_beta.max():.3f}')\n"
            "else:\n"
            "    print(f'Frozen: avg beta low={R[\"avg_beta_low\"]:.3f} | avg beta high={R[\"avg_beta_high\"]:.3f}')"
        ),
        md(
            "### 4c -- Real panel: BAB returns and legs"
        ),
        code(
            "if HAVE_REAL:\n"
            "    bab_m = s_bab['mean']*100; bab_t = s_bab['tstat']\n"
            "    lo_m = s_low['mean']*100; hi_m = s_high['mean']*100; mk_m = s_mkt['mean']*100\n"
            "    result_plot = result.copy()\n"
            "    result_plot['year'] = result_plot.index.year\n"
            "    annual_bab = result_plot.groupby('year')['bab_ret'].apply(\n"
            "        lambda x: float((1+x).prod()-1))\n"
            "else:\n"
            "    bab_m, bab_t = R['bab_mean'], R['bab_t']\n"
            "    lo_m, hi_m, mk_m = R['low_beta_mean'], R['high_beta_mean'], R['mkt_mean']\n"
            "    annual_bab = pd.Series(R['annual']) / 100\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "\n"
            "# Left: legs vs market\n"
            "labels = ['Low-beta\\n(long leg)', 'SPY', 'High-beta\\n(short leg)']\n"
            "vals = [lo_m, mk_m, hi_m]\n"
            "c = [AMBER, GREY, RED]\n"
            "axes[0].bar(labels, vals, color=c, width=0.5)\n"
            "axes[0].axhline(mk_m, ls='--', c=GREY, lw=1)\n"
            "axes[0].set_ylabel('mean annual return (%/yr, unlevered)')\n"
            "axes[0].set_title('Raw legs: high-beta earns more (pre-deleverage)')\n"
            "\n"
            "# Right: year-by-year BAB\n"
            "yrs = annual_bab.index\n"
            "vals_a = annual_bab.values * 100\n"
            "col_yr = [GREEN if v > 0 else RED for v in vals_a]\n"
            "axes[1].bar(yrs, vals_a, color=col_yr)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_xlabel('year'); axes[1].set_ylabel('BAB return (%/yr)')\n"
            "axes[1].set_title(f'BAB year-by-year | mean {bab_m:+.1f}% t={bab_t:+.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'BAB: {bab_m:+.2f}%/yr | vol: '\n"
            "      f'{s_bab[\"vol\"]*100 if HAVE_REAL else R[\"bab_vol\"]:.1f}% | '\n"
            "      f'Sharpe: {s_bab[\"sharpe\"] if HAVE_REAL else R[\"bab_sharpe\"]:+.3f} | '\n"
            "      f'HAC t: {bab_t:+.3f}')"
        ),
        md(
            f"> The BAB portfolio earns **+{R['bab_mean']:.2f}%/yr** at HAC *t* = "
            f"**+{R['bab_t']:.3f}** -- below the |t|>=2 bar. The positive mean is "
            f"consistent with the Frazzini-Pedersen mechanism but is statistically "
            f"indistinguishable from zero on this 15-year large-cap survivor panel."
        ),
        md(
            "### 4d -- Cumulative equity curve and drawdown"
        ),
        code(
            "if HAVE_REAL:\n"
            "    eq = (1 + result['bab_ret']).cumprod()\n"
            "    dd = (eq / eq.cummax() - 1) * 100\n"
            "    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)\n"
            "    ax1.plot(result.index, eq.values, c=AMBER, lw=2)\n"
            "    ax1.axhline(1, c='k', lw=1, ls='--')\n"
            "    ax1.set_ylabel('cumulative wealth (BAB)')\n"
            "    ax1.set_title('Equity curve of the BAB portfolio (monthly)')\n"
            "    ax2.fill_between(result.index, dd.values, 0, color=RED, alpha=0.4)\n"
            "    ax2.set_ylabel('drawdown (%)')\n"
            "    ax2.set_xlabel('date')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'Max drawdown: {dd.min():.1f}% | Hit rate: {s_bab[\"hit_rate\"]*100:.1f}%')\n"
            "else:\n"
            "    print(f'Frozen: max DD = {R[\"bab_dd\"]:.1f}% | hit rate = {R[\"bab_hit\"]:.1f}%')"
        ),
        md(
            "### 4e -- Leverage sensitivity\n\n"
            "How much does the BAB return change as we vary the leverage cap? "
            "This sensitivity analysis shows whether the result depends critically on "
            "the leverage assumption."
        ),
        code(
            "if HAVE_REAL:\n"
            "    caps = [1.0, 1.5, 2.0, 2.5, 3.0]\n"
            "    cap_results = []\n"
            "    for cap in caps:\n"
            "        res_c = st.bab_portfolio(prices, spy, beta_window=252,\n"
            "                                 min_stocks=20, leverage_cap=cap)\n"
            "        if not res_c.empty:\n"
            "            s_c = st.summary(res_c['bab_ret'])\n"
            "            cap_results.append({'cap': cap, 'mean_%': s_c['mean']*100,\n"
            "                                'sharpe': s_c['sharpe'], 't': s_c['tstat']})\n"
            "    cap_df = pd.DataFrame(cap_results)\n"
            "    fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "    ax.plot(cap_df['cap'], cap_df['mean_%'], 'o-', c=AMBER, lw=2)\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.set_xlabel('leverage cap (long leg)'); ax.set_ylabel('BAB mean return (%/yr)')\n"
            "    ax.set_title('BAB mean return vs leverage cap')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(cap_df.round(3).to_string(index=False))\n"
            "else:\n"
            "    print('Leverage sensitivity requires real cache. Frozen baseline:',\n"
            "          f'mean={R[\"bab_mean\"]:.2f}%/yr at cap={2.0}x')"
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `WEAK`** -- BAB +{R['bab_mean']:.2f}%/yr, HAC *t* = "
            f"+{R['bab_t']:.3f} (< 2). Academic prior from Frazzini-Pedersen (2014) "
            "and cross-asset replication prevents a `NONE` verdict, but real-tape numbers "
            "on 15 years of large-cap survivors do not independently clear the bar.\n"
            f"- **Tradability `FRAGILE`** -- Sharpe +{R['bab_sharpe']:.3f}, max DD "
            f"{R['bab_dd']:.1f}%, avg leverage {R['avg_leverage']:.2f}x. Financing "
            "costs (the borrowing spread) are not modelled -- they erode the gross return. "
            "The 2023 reversal (-25.9%) from momentum/tech surging shows the strategy "
            "can suffer severely in high-beta rallies.\n"
            "- **Survivorship `NAMED`** -- the S&P 500 survivor panel is biased. Failed "
            "high-beta firms (natural short candidates) are absent. Magnitude is an upper bound."
        ),

        # ---- BEAT 6 -- TRADABILITY -------------------------------------------
        md(
            "## 6 -- Could you trade it?\n\n"
            "Monthly rebalancing generates two-sided turnover (~30-50% for a median split). "
            "The real obstacles are:\n\n"
            "1. **Leverage financing spread.** Borrowing at Rf + spread reduces the gross "
            "return. In 2022-2024 (Fed Funds 4-5%), this spread was material.\n"
            "2. **Short borrow fees.** High-beta stocks (tech, growth, biotech) are "
            "expensive to borrow. Short recall risk adds operational complexity.\n"
            "3. **Crowding.** BAB strategies are widely known and implemented by large AMs. "
            "Simultaneous deleveraging (as in March 2020 or Q4 2023) can cause sharp "
            "reversals, hurting all BAB players at once.\n"
            "4. **Universe constraint.** The premium is well-documented in small/mid-cap; "
            "restricting to S&P 500 removes much of the cross-sectional beta dispersion."
        ),
        code(
            "# Leverage profile over time (synthetic for offline demo)\n"
            "if HAVE_REAL:\n"
            "    fig, ax = plt.subplots(figsize=(10, 3.5))\n"
            "    ax.plot(result.index, result['leverage'], c=AMBER, lw=1.5, label='long leverage')\n"
            "    ax.plot(result.index, result['deleverage'], c=GREY, lw=1.5, ls='--',\n"
            "            label='short deleverage')\n"
            "    ax.axhline(1.0, c='k', lw=0.8, ls=':')\n"
            "    ax.set_ylabel('leverage / deleverage factor')\n"
            "    ax.set_xlabel('date')\n"
            "    ax.set_title('BAB leverage profile over time')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'Avg leverage: {result[\"leverage\"].mean():.2f}x | '\n"
            "          f'Avg deleverage: {result[\"deleverage\"].mean():.2f}x')\n"
            "else:\n"
            "    print(f'Frozen: avg leverage={R[\"avg_leverage\"]:.2f}x | '\n"
            "          f'avg deleverage={R[\"avg_deleverage\"]:.2f}x')"
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Broader universe.** Frazzini-Pedersen (2014) used all US stocks from 1984. "
            "The premium is strongest in small/mid-cap where leverage constraints bind more.\n"
            "- **Cross-asset BAB.** The paper documents BAB in bonds, FX, and commodities. "
            "A multi-asset test (without equity survivorship bias) would be more informative.\n"
            "- **Net of financing costs.** Explicitly model the borrowing spread (e.g., "
            "LIBOR/SOFR + 50bps for margin). This can significantly reduce the net premium.\n"
            "- **Low-volatility variant.** Sort by realised volatility instead of beta. "
            "Similar results with less leverage required.\n"
            "- **[Study 144 -- Permanent Portfolio](../../144-permanent-portfolio/)**: a "
            "risk-parity-adjacent multi-asset allocation that avoids equity beta concentration.\n"
            "- **[Study 122 -- Gross-Profitability](../../122-gross-profitability/)**: "
            "quality factor using the same desk infrastructure (rolling sort, equal-weight, "
            "HAC inference).\n\n"
            "*Think BAB survives in large-caps net of financing costs? Fork this, add a "
            "realistic borrowing spread, expand to Russell 3000, and show t > 2 net of costs. "
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
