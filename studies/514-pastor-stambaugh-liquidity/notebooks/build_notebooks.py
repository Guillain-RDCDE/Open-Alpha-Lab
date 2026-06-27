"""Generate the two narrative notebooks for Study 514 (Pastor-Stambaugh Liquidity Risk).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). Synthetic
figures run anywhere, offline and deterministic; the real-panel cells use the cached
yfinance parquets under ../_cache/ if present and otherwise quote the frozen headline
numbers in ``R`` (mirror of docs/results.md, as-of 2026-06-26).
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


# Frozen real-panel headline numbers -- mirror of docs/results.md (as-of 2026-06-26).
R = dict(
    n_months=155, year_start=2013, year_end=2025, n_tickers=45,
    ls_mean=4.28, ls_vol=11.81, ls_sharpe=0.362, ls_t=1.466,
    ls_hit=55.5, ls_dd=-19.8,
    long_mean=23.71, short_mean=19.43, avg_turnover=0.21,
    net_mean=3.52, net_sharpe=0.298, net_t=1.207,
    placebo_mean=-0.20, placebo_std=2.37, placebo_p=0.070,
    fp_close="1060fab16f5d", fp_dv="7157e92920a6", fp_spy="5af4907fc2a8",
    window_sweep={24: (3.46, 1.019), 36: (4.28, 1.466), 48: (2.81, 0.820)},
    breadth={"tertile": (4.28, 1.466), "quintile": (3.34, 0.799)},
    ctrl={-0.08: (-11.98, -2.722), 0.0: (-0.38, -0.087),
          0.06: (8.22, 1.864), 0.12: (17.22, 3.824)},
    annual={2013: 10.8, 2014: 2.1, 2015: 9.0, 2016: 1.9, 2017: -4.3, 2018: 7.5,
            2019: 3.3, 2020: -4.4, 2021: -3.9, 2022: 5.3, 2023: 1.9, 2024: 16.1,
            2025: 3.9},
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

from pastor_stambaugh_liquidity import data, strategy as st

def _have_cache():
    study = os.path.abspath("..")
    return all(os.path.exists(os.path.join(study, "_cache", f))
               for f in ("ps_close.parquet", "ps_dollar_volume.parquet", "ps_spy.parquet"))

HAVE_REAL = _have_cache()
print("yfinance cache present:", HAVE_REAL)

if HAVE_REAL:
    close, dv, spy = data.fetch_panel()
    res = st.liquidity_sort(close, dv, spy, window=st.BETA_WINDOW_M)
    res_net = st.liquidity_sort(close, dv, spy, window=st.BETA_WINDOW_M,
                                cost_bps=st.COST_BPS, borrow_ann_bps=st.BORROW_ANN_BPS)
    s_ls   = st.summary(res["ls_gross"])
    s_long = st.summary(res["long_ret"])
    s_short= st.summary(res["short_ret"])
    s_net  = st.summary(res_net["ls_net"])
    print(f"N months: {s_ls['n']} | LS mean: {s_ls['mean']*100:+.2f}%/yr"
          f" | HAC t: {s_ls['tstat']:+.3f}")
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Pastor-Stambaugh Liquidity Risk -- does loading on liquidity shocks pay?\n"
            "### Pastor & Stambaugh (2003): the liquidity-risk premium, tested honestly\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Survives--gauntlet%3F: Busted](https://img.shields.io/badge/Survives--gauntlet%3F-Busted-c0392b?style=flat-square)\n\n"
            "In 2003, Lubos Pastor and Robert Stambaugh published a subtle idea: it is not how "
            "*illiquid* a stock is that should pay you, but how much its return **swings with "
            "market-wide liquidity shocks**. When liquidity suddenly dries up across the whole "
            "market (a 'flight to liquidity'), some stocks crash harder than others. Those "
            "liquidity-sensitive names -- high *liquidity beta* -- are scary to hold, so the "
            "theory says they must compensate you with a higher expected return.\n\n"
            "This is a *risk loading*, not an illiquidity *level*. It is deliberately distinct "
            "from [Study 140 -- Amihud-Illiquidity](../../140-amihud-illiquidity/), which tests "
            "whether the *level* of a stock's illiquidity pays. Here we test the **beta** on "
            "aggregate liquidity innovations.\n\n"
            "> **This is the plain-language layer.** Want the aggregate-liquidity series, the "
            "gamma cross-section, the placebo null and the cost ledger? See "
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
            "| Do high liquidity-beta stocks earn a premium? | **Weakly, in the right "
            f"direction:** **+{R['ls_mean']:.2f}%/yr**, but HAC *t* = **+{R['ls_t']:.2f}** "
            "-- below the |t|>=2 bar. |\n"
            "| Is the placebo null reassuring? | **Suggestive, not decisive.** Shuffling the "
            f"liquidity-beta labels gives mean ~0; the real signal sits at *p* = "
            f"**{R['placebo_p']:.3f}**. |\n"
            "| Is it tradable? | **Fragile.** Net **+"
            f"{R['net_mean']:.2f}%/yr**, net *t* = +{R['net_t']:.2f}; low turnover so costs "
            "are mild, but the gross signal already misses. |\n"
            "| Is there a survivorship problem? | **Severely, and we name it.** The natural "
            "carriers of liquidity risk -- illiquid, distressed, delisted names -- are absent. |\n\n"
            "> The liquidity-risk premium is real in the academic literature on a broad CRSP "
            "universe. On 45 of the most liquid stocks alive, the signal is statistically "
            "marginal and the honest gauntlet busts it."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim\n\n"
            "> *\"A stock's expected return depends on the sensitivity of its return to "
            "innovations in aggregate liquidity. Stocks with higher liquidity betas have "
            "higher expected returns.\"*\n\n"
            "-- Pastor & Stambaugh (2003), *Journal of Political Economy*\n\n"
            "The intuition: in a liquidity crisis, you most want assets you can sell. Stocks "
            "that fall hardest exactly when liquidity evaporates are the worst possible thing "
            "to own -- so rational investors demand a premium to hold them. That premium is the "
            "**liquidity-risk** factor. Pastor-Stambaugh found the top-decile liquidity-beta "
            "stocks beat the bottom decile by ~7.5%/yr from 1966-1999."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "If true, liquidity risk is a priced factor uncorrelated with the usual market, "
            "size, and value exposures -- a genuine source of premium and a key input to "
            "risk models, especially for funds that must hold liquid books. It is one of the "
            "canonical 'macro' risk factors alongside the market and is sold inside many "
            "multi-factor products.\n\n"
            "Our questions:\n"
            "1. **Does it survive on a realistic large-cap panel?**\n"
            "2. **Does it beat a label-shuffle placebo?**\n"
            "3. **What is the survivorship bias -- and which way does it cut?**"
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 -- How would we even know?\n\n"
            "Three disciplines keep us honest:\n\n"
            "1. **Estimate gammas on past data only.** Each name's liquidity beta uses a "
            "trailing 36-month window; the holding month is always strictly after the window. "
            "Positions enter at the prior month-end close -- **one execution lag**, no "
            "look-ahead.\n"
            "2. **Run a placebo.** Shuffle the liquidity-beta labels across names and re-sort. "
            "If the sort is real, the shuffled long-short collapses to ~0 and the real signal "
            "is an outlier in that null.\n"
            "3. **Name the survivorship bias.** The universe is the *current* large-cap set "
            "projected backwards. The natural carriers of liquidity risk -- small, illiquid, "
            "ultimately delisted names -- are gone. This is the *most favourable* possible "
            "test, so a miss here is damning."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md("## 4 -- The teardown\n\n**First, does the engine work when the effect is planted?**"),
        code(
            "# Synthetic positive control: plant a known liquidity-risk premium and recover it.\n"
            "ctrl = st.control_sweep([-0.08, 0.0, 0.06, 0.12], window=30)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if m > 0 else RED for m in ctrl['ls_mean_ann']*100]\n"
            "ax.bar(ctrl['premium'].astype(str), ctrl['ls_mean_ann']*100, color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, (m, t) in enumerate(zip(ctrl['ls_mean_ann']*100, ctrl['tstat'])):\n"
            "    ax.text(i, m + (0.4 if m >= 0 else -1.2), f't={t:+.1f}',\n"
            "            ha='center', va='bottom', fontsize=8)\n"
            "ax.set_xlabel('planted liquidity-risk premium')\n"
            "ax.set_ylabel('long-short mean (%/yr)')\n"
            "ax.set_title('Synthetic control: engine recovers the planted premium, right sign')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),
        md(
            "The engine is faithful: zero at the null, monotone in the planted premium, correct "
            "sign, a clean *t* > 3 at a strong premium. So the verdict on the real tape reflects "
            "**the market**, not a broken detector."
        ),
        md("**Now the honest test on the real yfinance panel.**"),
        code(
            "if HAVE_REAL:\n"
            "    ls_mean = s_ls['mean']*100; ls_t = s_ls['tstat']\n"
            "    long_m = s_long['mean']*100; short_m = s_short['mean']*100\n"
            "    res_plot = res.copy(); res_plot['year'] = res_plot.index.year\n"
            "    annual = res_plot.groupby('year')['ls_gross'].apply(lambda x: float((1+x).prod()-1))\n"
            "else:\n"
            "    ls_mean, ls_t = R['ls_mean'], R['ls_t']\n"
            "    long_m, short_m = R['long_mean'], R['short_mean']\n"
            "    annual = pd.Series(R['annual'])/100\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))\n"
            "labels = ['High-gamma\\n(long leg)', 'Low-gamma\\n(short leg)']\n"
            "axes[0].bar(labels, [long_m, short_m], color=[AMBER, GREY], width=0.5)\n"
            "axes[0].set_ylabel('mean annual return (%/yr)')\n"
            "axes[0].set_title('Raw legs: both big in a bull market; the edge is the gap')\n"
            "yrs = list(annual.index); vals = [annual[y]*100 for y in yrs]\n"
            "col_yr = [GREEN if v > 0 else RED for v in vals]\n"
            "axes[1].bar(yrs, vals, color=col_yr); axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_xlabel('year'); axes[1].set_ylabel('long-short (%/yr)')\n"
            "axes[1].set_title(f'Year-by-year | mean {ls_mean:+.1f}%  t={ls_t:+.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Long-short: {ls_mean:+.2f}%/yr | HAC t = {ls_t:+.3f}')"
        ),
        md(
            f"High-gamma stocks out-earn low-gamma by **+{R['ls_mean']:.1f}%/yr** -- in the "
            f"Pastor-Stambaugh direction -- but HAC *t* = **+{R['ls_t']:.2f}** is below the "
            "|t|>=2 bar. The premium is real-ish in sign, ambiguous in significance."
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- WEAK.** HAC *t* = +{R['ls_t']:.2f} (|t| < 2), placebo *p* = "
            f"{R['placebo_p']:.3f}. Right direction, suggestive placebo, strong academic prior "
            "-- but it does not independently clear the bar. WEAK, not NONE.\n"
            f"- **Tradability -- FRAGILE.** Net +{R['net_mean']:.2f}%/yr, Sharpe "
            f"+{R['net_sharpe']:.2f}, max DD {R['ls_dd']:.1f}%. Low turnover keeps costs mild, "
            "but the gross signal already misses and it is fragile to design choices.\n"
            "- **Honest gauntlet -- BUSTED.** On 45 of the most liquid stocks alive (the "
            "most favourable possible test) the premium still fails |t|>=2 net of costs."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 -- Could you actually trade it?\n\n"
            "Practical obstacles:\n\n"
            "1. **The premium lives in illiquid names.** The theory's juice is in small, "
            "thin stocks -- exactly where trading costs explode and exactly what our liquid "
            "survivor basket excludes.\n"
            "2. **Gamma is noisy.** Liquidity betas estimated on 36 months of monthly data "
            "have wide standard errors; the cross-sectional ranking is unstable.\n"
            "3. **Crowding and decay.** Published in 2003 and widely known; Ben-Rephael et al. "
            "(2015) document a diminishing liquidity premium as markets deepened.\n"
            "4. **Short borrow.** Shorting the low-gamma leg carries borrow costs we charge "
            "but that vary in stressed markets."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Broader universe.** The original paper used the full CRSP cross-section; the "
            "premium concentrates in small/illiquid names a large-cap basket cannot reach.\n"
            "- **Better liquidity series.** Pastor-Stambaugh use a signed-volume return-reversal "
            "measure; we use an Amihud average. Korajczyk-Sadka (2008) show the magnitude "
            "depends heavily on the measure.\n"
            "- **[Study 140 -- Amihud-Illiquidity](../../140-amihud-illiquidity/)**: the "
            "illiquidity *level* cousin -- a different channel.\n"
            "- **[Study 238 -- Betting-Against-Beta](../../238-betting-against-beta/)**: another "
            "priced risk-loading sort on the same desk infrastructure.\n\n"
            "*Think liquidity risk pays net of costs on a complete universe? Fork this, pull a "
            "Russell 3000 with delistings, build the signed-volume PS series, and show t > 2 "
            "net of borrow. That is the bar.*"
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
            "# Pastor-Stambaugh Liquidity Risk -- a quantitative teardown\n"
            "### aggregate-liquidity series * gamma cross-section * placebo null * costs * HAC\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Survives--gauntlet%3F: Busted](https://img.shields.io/badge/Survives--gauntlet%3F-Busted-c0392b?style=flat-square)\n\n"
            "The quantitative companion to the "
            "[notebook for the curious](01_for_the_curious.ipynb) -- same seven beats, every "
            "claim carrying its standard error. We test Pastor-Stambaugh (2003): build an "
            "Amihud-style aggregate liquidity series, estimate trailing 36-month liquidity "
            "betas, sort high-minus-low gamma into a monthly dollar-neutral book, one "
            "execution lag.\n\n"
            "> **Not investment advice.** Real data: yfinance daily close + volume, 45 "
            "large-caps, 2010-2025, as-of 2026-06-26. Methods in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship bias is named:** universe = current large-caps projected "
            "backwards; the illiquid carriers of liquidity risk are absent, so all positive "
            "results are upper-bound estimates."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | LS mean **+{R['ls_mean']:.2f}%/yr**, HAC *t* = "
            f"**+{R['ls_t']:.3f}** (|t| < 2), placebo *p* = **{R['placebo_p']:.3f}**. |\n"
            f"| **Tradability** | `FRAGILE` | Net **+{R['net_mean']:.2f}%/yr**, net *t* = "
            f"**+{R['net_t']:.3f}**, Sharpe **+{R['net_sharpe']:.3f}**, max DD "
            f"**{R['ls_dd']:.1f}%**. |\n"
            "| **Survives gauntlet?** | `BUSTED` | 45 of the most liquid stocks alive; the "
            "liquidity-risk carriers are absent -- most favourable test still fails. |\n\n"
            "> The Pastor-Stambaugh liquidity-risk factor is documented on a broad CRSP "
            "universe (1966-1999). On 13 years of 45 large-cap survivors it is marginal -- "
            "consistent with Ben-Rephael et al. (2015) on the diminishing liquidity premium "
            "and Li-Novy-Marx-Velikov (2019) on its fragility in large-caps."
        ),

        # ---- BEAT 1 -- CLAIM -------------------------------------------------
        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $L_t$ be the innovation in aggregate market liquidity and $\\gamma_i$ the "
            "loading of stock $i$'s excess return on $L_t$ (controlling for the market). "
            "Pastor-Stambaugh (2003) assert:\n\n"
            "- **H1 (signal).** $E[r_i] $ is increasing in $\\gamma_i$: high liquidity-beta "
            "stocks earn more. The long-high / short-low gamma portfolio has positive mean.\n"
            "- **H2 (a risk story).** The premium is compensation for covarying badly with "
            "liquidity crises, not a market-beta artefact -- so we control for the market "
            "when estimating $\\gamma_i$.\n"
            "- **H3 (tradable).** The premium survives costs and borrow.\n\n"
            "We find H1 in the *right direction but below significance* (t=1.47), H2 holds by "
            "construction (gammas are market-residual loadings), and reject H3 (net t=1.21)."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what? -- the economic stakes\n\n"
            "Liquidity risk is one of the canonical priced macro factors. If it has attenuated "
            "in large-caps -- or never applied cleanly outside the illiquid tail -- then "
            "liquidity-factor sleeves in multi-factor products are charging for an exposure "
            "the liquid-universe data no longer supports."
        ),

        # ---- BEAT 3 -- PROTOCOL ----------------------------------------------
        md(
            "## 3 -- The protocol\n\n"
            "- **Aggregate liquidity.** Daily Amihud illiquidity |ret|/dollar-volume per name, "
            "averaged within month and across names; sign-flipped and z-scored to a liquidity "
            "*level*. Monthly first difference = the liquidity *shock*.\n"
            "- **Gamma.** Trailing 36-month OLS of each name's monthly excess return on the "
            "liquidity shock, controlling for the market excess return.\n"
            "- **Sort.** Monthly rank by trailing gamma; long top tertile, short bottom "
            "tertile, equal-weight, dollar-neutral.\n"
            "- **Execution lag.** Gamma public at prior month-end close; hold the following "
            "month. No look-ahead.\n"
            "- **Costs.** 10 bps one-way x turnover (both legs) + 50 bps/yr borrow on the short.\n"
            "- **Inference.** Newey-West HAC *t* on the monthly long-short; a 200-permutation "
            "label-shuffle placebo *p*.\n"
            "- **Universe caveat.** 45 large-cap survivors; survivorship-biased."
        ),

        # ---- BEAT 4 -- TEARDOWN ----------------------------------------------
        md("## 4 -- The teardown"),
        md(
            "### 4a -- Positive control: the engine is a faithful detector\n\n"
            "Sweep the planted liquidity-risk premium from negative to positive. The "
            "long-short mean should be monotone and correctly signed."
        ),
        code(
            "ctrl = st.control_sweep([-0.08, -0.04, 0.0, 0.06, 0.12], window=30)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if m > 0 else RED for m in ctrl['ls_mean_ann']*100]\n"
            "ax.bar(ctrl['premium'].astype(str), ctrl['ls_mean_ann']*100, color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, (m, t) in enumerate(zip(ctrl['ls_mean_ann']*100, ctrl['tstat'])):\n"
            "    ax.text(i, m + (0.4 if m >= 0 else -1.2), f't={t:+.1f}',\n"
            "            ha='center', va='bottom', fontsize=8)\n"
            "ax.set_xlabel('planted premium'); ax.set_ylabel('long-short mean (%/yr)')\n"
            "ax.set_title('Positive control: monotone in the planted premium')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),
        md(
            "### 4b -- The aggregate liquidity series and its shocks\n\n"
            "What does market-wide liquidity look like over the sample? Spikes down are "
            "liquidity dry-ups (e.g. 2020 COVID, 2022 rate shock)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    liq = st.aggregate_liquidity(close, dv)\n"
            "    shock = st.liquidity_shock(liq)\n"
            "    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)\n"
            "    a1.plot(liq.index, liq.values, c=AMBER, lw=1.5)\n"
            "    a1.axhline(0, c='k', lw=0.8, ls='--')\n"
            "    a1.set_ylabel('aggregate liquidity (z)')\n"
            "    a1.set_title('Aggregate market liquidity (higher = more liquid)')\n"
            "    a2.bar(shock.index, shock.values,\n"
            "           color=[GREEN if v > 0 else RED for v in shock.values], width=20)\n"
            "    a2.axhline(0, c='k', lw=0.8)\n"
            "    a2.set_ylabel('liquidity shock'); a2.set_xlabel('date')\n"
            "    a2.set_title('Monthly liquidity innovations (the risk factor)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'Liquidity shock std: {shock.std():.4f} | months: {len(shock)}')\n"
            "else:\n"
            "    print('Aggregate-liquidity figure requires the real cache.')"
        ),
        md(
            "### 4c -- The cross-section of liquidity betas\n\n"
            "Latest gamma snapshot across names: a roughly symmetric spread around zero is "
            "what we want for a long-short sort."
        ),
        code(
            "if HAVE_REAL:\n"
            "    stk_ex, mkt_ex = st.monthly_excess(close, spy)\n"
            "    shock = st.liquidity_shock(st.aggregate_liquidity(close, dv))\n"
            "    gammas = st.liquidity_betas(stk_ex, mkt_ex, shock, window=st.BETA_WINDOW_M)\n"
            "    last_g = gammas.iloc[-1].dropna()\n"
            "    fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "    ax.hist(last_g, bins=18, color=AMBER, edgecolor='white', alpha=0.85)\n"
            "    ax.axvline(last_g.median(), c=RED, lw=2, ls='--', label=f'median={last_g.median():.2f}')\n"
            "    ax.axvline(0, c=GREY, lw=1.5, ls=':', label='gamma=0')\n"
            "    ax.set_xlabel('liquidity beta (gamma)'); ax.set_ylabel('count')\n"
            "    ax.set_title('Cross-section of liquidity betas (latest snapshot)')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'gamma range: {last_g.min():.2f} -- {last_g.max():.2f} | n={len(last_g)}')\n"
            "else:\n"
            "    print('Gamma cross-section requires the real cache.')"
        ),
        md("### 4d -- Real panel: long-short returns, equity curve, drawdown"),
        code(
            "if HAVE_REAL:\n"
            "    ls_m = s_ls['mean']*100; ls_t = s_ls['tstat']\n"
            "    eq = (1 + res['ls_gross']).cumprod()\n"
            "    dd = (eq/eq.cummax() - 1)*100\n"
            "    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)\n"
            "    a1.plot(res.index, eq.values, c=AMBER, lw=2); a1.axhline(1, c='k', lw=1, ls='--')\n"
            "    a1.set_ylabel('cumulative wealth (LS)')\n"
            "    a1.set_title(f'Long-short equity curve | mean {ls_m:+.1f}%  HAC t={ls_t:+.2f}')\n"
            "    a2.fill_between(res.index, dd.values, 0, color=RED, alpha=0.4)\n"
            "    a2.set_ylabel('drawdown (%)'); a2.set_xlabel('date')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'Max DD: {dd.min():.1f}% | Hit rate: {s_ls[\"hit_rate\"]*100:.1f}% | '\n"
            "          f'Sharpe: {s_ls[\"sharpe\"]:+.3f}')\n"
            "else:\n"
            "    print(f'Frozen: LS {R[\"ls_mean\"]:+.2f}%/yr | t {R[\"ls_t\"]:+.2f} | '\n"
            "          f'DD {R[\"ls_dd\"]:.1f}% | Sharpe {R[\"ls_sharpe\"]:+.2f}')"
        ),
        md(
            "### 4e -- The placebo: shuffle the gamma labels\n\n"
            "If the sort is real, permuting the liquidity-beta labels across names destroys "
            "it: the placebo long-short means cluster on zero and the real signal is an "
            "outlier in that null."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pb = st.placebo_pvalue(close, dv, spy, window=st.BETA_WINDOW_M, n_perm=200)\n"
            "    arr = np.array(pb['placebo'])*100\n"
            "    fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "    ax.hist(arr, bins=30, color=GREY, edgecolor='white', alpha=0.8, label='placebo')\n"
            "    ax.axvline(pb['real_mean_ann']*100, c=AMBER, lw=2.5,\n"
            "               label=f'real {pb[\"real_mean_ann\"]*100:+.2f}%')\n"
            "    ax.axvline(0, c='k', lw=1, ls='--')\n"
            "    ax.set_xlabel('long-short mean (%/yr)'); ax.set_ylabel('count')\n"
            "    ax.set_title(f'Label-shuffle placebo (200 perms) | p={pb[\"p_value\"]:.3f}')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'Real {pb[\"real_mean_ann\"]*100:+.2f}%/yr | placebo mean '\n"
            "          f'{pb[\"placebo_mean\"]*100:+.2f}% std {pb[\"placebo_std\"]*100:.2f}% | '\n"
            "          f'p={pb[\"p_value\"]:.3f}')\n"
            "else:\n"
            "    print(f'Frozen placebo: mean {R[\"placebo_mean\"]:+.2f}% std {R[\"placebo_std\"]:.2f}% '\n"
            "          f'| p={R[\"placebo_p\"]:.3f}')"
        ),
        md(
            "### 4f -- Robustness: window and breadth, gross vs net\n\n"
            "Every variant is positive and in the predicted direction, and **none clears "
            "|t| >= 2.** Costs (low-turnover) shave only ~0.76%/yr."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for w in (24, 36, 48):\n"
            "        rw = st.liquidity_sort(close, dv, spy, window=w)\n"
            "        sw = st.summary(rw['ls_gross'])\n"
            "        rows.append({'variant': f'window {w}m', 'mean_%': sw['mean']*100, 't': sw['tstat']})\n"
            "    for q, lbl in ((1/3, 'tertile'), (0.2, 'quintile')):\n"
            "        rq = st.liquidity_sort(close, dv, spy, window=st.BETA_WINDOW_M, quantile=q)\n"
            "        sq = st.summary(rq['ls_gross'])\n"
            "        rows.append({'variant': lbl, 'mean_%': sq['mean']*100, 't': sq['tstat']})\n"
            "    rows.append({'variant': 'net (cost+borrow)', 'mean_%': s_net['mean']*100, 't': s_net['tstat']})\n"
            "    rob = pd.DataFrame(rows)\n"
            "    fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "    col = [GREEN if t >= 2 else AMBER for t in rob['t']]\n"
            "    ax.barh(rob['variant'], rob['t'], color=col)\n"
            "    ax.axvline(2, c=RED, lw=1.5, ls='--', label='|t|=2 bar')\n"
            "    ax.axvline(0, c='k', lw=1)\n"
            "    ax.set_xlabel('HAC t-stat'); ax.set_title('Robustness: nothing clears |t|=2')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(rob.round(3).to_string(index=False))\n"
            "else:\n"
            "    for w,(m,t) in R['window_sweep'].items():\n"
            "        print(f'window {w}m: {m:+.2f}%/yr  t={t:+.3f}')"
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `WEAK`** -- LS +{R['ls_mean']:.2f}%/yr, HAC *t* = +{R['ls_t']:.3f} "
            f"(< 2), placebo *p* = {R['placebo_p']:.3f}. Right direction, suggestive placebo, "
            "strong prior from Pastor-Stambaugh (2003) -- but it does not independently clear "
            "the bar. WEAK, not NONE.\n"
            f"- **Tradability `FRAGILE`** -- net +{R['net_mean']:.2f}%/yr, net *t* = "
            f"+{R['net_t']:.3f}, Sharpe +{R['net_sharpe']:.3f}, max DD {R['ls_dd']:.1f}%. "
            "Low turnover, mild costs -- but the gross signal already misses and it is "
            "fragile to window/breadth.\n"
            "- **Honest gauntlet `BUSTED`** -- 45 of the most liquid stocks alive is the "
            "*most favourable* possible test of a liquidity-risk premium, and it still fails "
            "|t| >= 2 net of costs. On a complete universe with delistings it would more "
            "likely shrink than grow."
        ),

        # ---- BEAT 6 -- TRADABILITY -------------------------------------------
        md(
            "## 6 -- Could you trade it?\n\n"
            "1. **The juice is in illiquid names.** The premium concentrates exactly where "
            "trading costs explode -- and exactly what a liquid survivor basket excludes.\n"
            "2. **Noisy gammas.** 36 monthly observations give wide standard errors; the "
            "ranking is unstable, inflating effective turnover beyond our measured 0.21.\n"
            "3. **Decay.** Ben-Rephael et al. (2015) document a diminishing liquidity premium; "
            "Li-Novy-Marx-Velikov (2019) find it fragile in large-caps post-2000.\n"
            "4. **Short borrow & stress.** The short low-gamma leg's borrow cost spikes in "
            "exactly the liquidity crises the factor is supposed to harvest."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Full CRSP universe with delistings.** The original premium lives in the "
            "illiquid tail and needs the names we deleted.\n"
            "- **Signed-volume PS series.** Replace the Amihud average with Pastor-Stambaugh's "
            "own return-reversal / signed-volume liquidity measure (Korajczyk-Sadka 2008 show "
            "the measure matters a lot).\n"
            "- **[Study 140 -- Amihud-Illiquidity](../../140-amihud-illiquidity/)**: the "
            "illiquidity *level* premium -- the level cousin of this risk loading.\n"
            "- **[Study 238 -- Betting-Against-Beta](../../238-betting-against-beta/)** and "
            "**[Study 330 -- Low-Volatility-Anomaly](../../330-low-volatility-anomaly/)**: "
            "sibling risk-sort studies on the same desk engine.\n\n"
            "*Think liquidity risk pays net of costs on a complete universe? Fork this, pull a "
            "Russell 3000 with delistings, build the signed-volume PS series, and show t > 2 "
            "net of borrow. That is the bar.*"
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
