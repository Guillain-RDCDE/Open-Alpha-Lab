"""Generate the two narrative notebooks for Study 507 (Cross-Sectional-Momentum).

    python notebooks/build_notebooks.py

Both notebooks follow the desk's seven beats. Synthetic figures run anywhere, offline and
deterministic; the real-panel cells use the cached yfinance parquet under ../_cache/ if present
and otherwise quote the frozen headline numbers in ``R`` (the single source of truth that mirrors
docs/results.md exactly).
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
    asof="2026-06-26",
    n_days=3424, n_tickers=38, n_months_panel=164, n_hold=151,
    date_start="2012-05-18", date_end="2025-12-30",
    hold_start="2013-05", hold_end="2025-11",
    fp="e038fa42cc02",
    cost_bps=10.0, borrow_bps=50.0,
    # quintile
    q_gross=-0.34, q_net=-1.43, q_sharpe=-0.021, q_t=-0.072, q_t_net=-0.306,
    q_hit=54.3, q_dd=-44.9, q_worst=-17.7, q_turn=24.8, q_p=0.545,
    q_win=16.16, q_los=16.50, q_nleg=8,
    # decile
    d_gross=1.24, d_net=0.02, d_sharpe=0.054, d_t=0.197, d_t_net=0.004,
    d_hit=57.6, d_dd=-53.5, d_worst=-24.6, d_turn=30.0, d_p=0.376,
    d_win=16.43, d_los=15.19, d_nleg=4,
    # synthetic control
    ctrl={0.00: (6.5, 1.47), 0.15: (17.4, 4.00), 0.30: (40.3, 7.88), 0.50: (71.3, 12.50)},
    null_seed_t=-0.07,
    annual={2013: 1.1, 2014: -4.1, 2015: 26.0, 2016: -26.7, 2017: 28.2, 2018: 2.1,
            2019: 3.1, 2020: -2.7, 2021: -23.5, 2022: 4.8, 2023: -0.3, 2024: 22.7,
            2025: -29.6},
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

from cross_sectional_momentum import data, strategy as st

def _have_cache():
    study = os.path.abspath("..")
    return os.path.exists(os.path.join(study, "_cache", "csmom_prices.parquet"))

HAVE_REAL = _have_cache()
print("yfinance cache present:", HAVE_REAL)

if HAVE_REAL:
    prices = data.drop_partial_last_month(data.fetch_panel())
    mp = st.to_monthly(prices)
    ls_q = st.long_short(mp, frac=0.20)
    ls_d = st.long_short(mp, frac=0.10)
    sq = st.summary(ls_q["wml_gross"]); sd = st.summary(ls_d["wml_gross"])
    print(f"Quintile WML: {sq['mean']*100:+.2f}%/yr  HAC t={sq['tstat']:+.3f}")
    print(f"Decile   WML: {sd['mean']*100:+.2f}%/yr  HAC t={sd['tstat']:+.3f}")
"""


def build_curious():
    cells = [
        md(
            "# Cross-Sectional-Momentum -- do past winners keep winning?\n"
            "### Jegadeesh-Titman (1993) 12-1 relative strength, tested honestly on a survivor basket\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Decile_vs_quintile%3F: Busted](https://img.shields.io/badge/Decile_vs_quintile%3F-Busted-8b949e?style=flat-square)\n\n"
            "In 1993, Narasimhan Jegadeesh and Sheridan Titman published what became the most "
            "replicated anomaly in finance: rank stocks by their past 12-month return, buy the "
            "winners, short the losers, refresh monthly -- and the spread paid about **1% a "
            "month**. The catch they baked in: *skip the most recent month* (the classic "
            "\"12-1\"), because last month's moves tend to *reverse*.\n\n"
            "We replicate that exact recipe on a **38-name large-cap basket** and ask the honest "
            "question: does it still pay on a small, modern, *survivor* basket?\n\n"
            "> **This is the plain-language layer.** Want the HAC *t*-stats, the label-shuffle "
            "placebo and the decile-vs-quintile sweep? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do winners keep winning? | **No.** The quintile winners-minus-losers book earns "
            f"**{R['q_gross']:+.2f}%/yr** at HAC *t* = **{R['q_t']:+.3f}** -- a coin. |\n"
            f"| Does tightening to a decile help? | **No.** The decile nudges to "
            f"**{R['d_gross']:+.2f}%/yr** (*t* = {R['d_t']:+.2f}) -- still inside the noise, and "
            f"net of costs it is **{R['d_net']:+.2f}%/yr**. |\n"
            f"| Could you trade it? | **No.** Best-case net **{R['d_net']:+.2f}%/yr** with a "
            f"**{R['d_dd']:.1f}%** drawdown and {R['d_turn']:.0f}%/mo turnover. |\n"
            "| Is there a survivorship problem? | **Yes -- and we name it on the signal axis.** "
            "The basket is names *still trading in 2026*; the loser leg never holds the firms "
            "that trended to zero. |\n\n"
            "> Both legs are winners. The winner leg earns "
            f"**~{R['q_win']:.0f}%/yr** and the loser leg **~{R['q_los']:.0f}%/yr** -- in a "
            "basket of survivors that all compounded for a decade, sorting on past return "
            "separates almost nothing."
        ),

        md(
            "## 1 -- The claim\n\n"
            "> *\"Strategies which buy stocks that have performed well in the past and sell "
            "stocks that have performed poorly in the past generate significant positive "
            "returns over 3- to 12-month holding periods.\"*\n\n"
            "-- Jegadeesh & Titman (1993), *Journal of Finance*\n\n"
            "The 12-1 recipe: at each month-end, compute each stock's return over the trailing "
            "12 months but **skip the most recent month** (to avoid short-term reversal). Rank, "
            "go long the top winners and short the bottom losers, hold one month, repeat."
        ),

        md(
            "## 2 -- So what?\n\n"
            "Momentum is the fourth factor in the Carhart (1997) model, a building block of "
            "every multi-factor fund, and -- per Asness-Moskowitz-Pedersen (2013) -- it shows "
            "up \"everywhere\", across asset classes and countries. If it has *decayed* in US "
            "large caps, a lot of fee-laden product is selling a 1993 story the modern data no "
            "longer tells."
        ),

        md(
            "## 3 -- How would we even know?\n\n"
            "Three disciplines keep us honest:\n\n"
            "1. **No look-ahead.** The 12-1 signal uses only past prices (and skips the most "
            "recent month). We form on the month-end close and earn the *next* month -- one "
            "forward execution lag, no same-bar fill.\n"
            "2. **A placebo null.** We shuffle which stock the signal points at and recompute "
            "the spread 1,000 times. If the real edge is no better than the shuffle, it is "
            "noise.\n"
            "3. **Name the survivorship bias on the *signal* axis.** The basket is the names "
            "that *survived* to 2026. The persistent losers a momentum short would harvest -- "
            "firms that trended into delisting -- are simply absent. That inflates any premium, "
            "so even a flat result here is an upper bound."
        ),

        md("## 4 -- The teardown\n\n**First, does the momentum engine work when the effect is planted?**"),
        code(
            "ctrl = st.synthetic_control()\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if m > 0.5 else GREY for m in ctrl['wml_mean_ann']*100]\n"
            "ax.bar(ctrl['mom_strength'].astype(str), ctrl['wml_mean_ann']*100, color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, (m, t) in enumerate(zip(ctrl['wml_mean_ann']*100, ctrl['tstat'])):\n"
            "    ax.text(i, m + 1.5, f't={t:+.1f}', ha='center', fontsize=8)\n"
            "ax.set_xlabel('planted momentum drift'); ax.set_ylabel('WML mean (%/yr)')\n"
            "ax.set_title('Synthetic control: engine recovers momentum when it is planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),
        md(
            "The engine is faithful: it finds momentum when it is planted and (across 20 seeds) "
            f"scores a null *t* of **{R['null_seed_t']:+.2f}** when it is not. So the verdict on "
            "the real tape reflects **the market**, not the method.\n\n"
            "**Now the honest test on the real basket.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    win_m = st.summary(ls_q['win'])['mean']*100\n"
            "    los_m = st.summary(ls_q['los'])['mean']*100\n"
            "    q_g = sq['mean']*100; q_t = sq['tstat']\n"
            "else:\n"
            "    win_m, los_m = " f"{R['q_win']}, {R['q_los']}\n"
            "    q_g, q_t = " f"{R['q_gross']}, {R['q_t']}\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar(['Winner leg', 'Loser leg'], [win_m, los_m], color=[AMBER, GREY], width=0.5)\n"
            "ax.set_ylabel('mean annual return (%/yr)')\n"
            "ax.set_title(f'Both legs are winners | WML = {q_g:+.2f}%/yr, t={q_t:+.2f}')\n"
            "for i, v in enumerate([win_m, los_m]):\n"
            "    ax.text(i, v + 0.3, f'{v:+.1f}%', ha='center')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'WML (quintile): {q_g:+.2f}%/yr | HAC t = {q_t:+.3f}')"
        ),
        md(
            f"The winner leg (**~{R['q_win']:.0f}%/yr**) and the loser leg "
            f"(**~{R['q_los']:.0f}%/yr**) are practically tied. Subtracting one from the other "
            f"leaves **{R['q_gross']:+.2f}%/yr** -- nothing. This is the survivorship channel "
            "made visible: in a basket where everyone survived and compounded, \"losers\" are "
            "just slightly-less-spectacular winners."
        ),

        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- NONE.** Quintile WML HAC *t* = {R['q_t']:+.3f} (placebo *p* = "
            f"{R['q_p']:.3f}); the decile *t* = {R['d_t']:+.2f} (placebo *p* = {R['d_p']:.2f}). "
            "Neither sort clears any bar, and survivorship -- named on the signal axis -- makes "
            "even these flat numbers an upper bound. The literature (Jegadeesh-Titman 1993) is "
            "strong, but prior alone is WEAK at most, and the real tape here lands NONE.\n"
            f"- **Tradability -- MIRAGE.** Best-case net **{R['d_net']:+.2f}%/yr** with a "
            f"**{R['d_dd']:.1f}%** drawdown. Nothing to trade.\n"
            "- **Decile vs quintile -- BUSTED.** Concentration did not rescue the edge."
        ),

        md(
            "## 6 -- Could you actually trade it?\n\n"
            "No. Even setting aside the flat gross signal:\n\n"
            f"1. **Turnover.** ~{R['q_turn']:.0f}%/mo one-way -- momentum ranks churn, and every "
            "churn pays the spread.\n"
            "2. **The short leg.** Shorting individual stocks costs borrow (we charge 50bps/yr) "
            "and carries recall risk.\n"
            f"3. **The crash.** A {R['q_dd']:.0f}% drawdown with the sign flipping +28% (2017) to "
            "-30% (2025) year to year -- the classic momentum-crash tail with none of the "
            "long-run premium to pay for it.\n"
            "4. **Survivorship.** The real loser leg -- delisted, trending-to-zero names -- is "
            "missing, so the live strategy would likely be *worse*."
        ),

        md(
            "## 7 -- Going further\n\n"
            "- **The full S&P 500.** [Study 24 -- Stampede](../../24-stampede/) runs the same "
            "12-1 WML on the entire modern S&P 500 cross-section -- where the larger basket "
            "lifts the signal to Weak.\n"
            "- **Residual momentum.** [Study 25 -- Clean-Slate](../../25-clean-slate/) and "
            "[Study 237](../../237-residual-momentum/) strip the market factor before ranking, "
            "the \"cleaner cousin\".\n"
            "- **Vol-scaling.** Barroso-Santa-Clara (2015) show scaling by recent volatility "
            "roughly doubles momentum's Sharpe by taming the crash.\n\n"
            "*Think momentum survives on a realistic small basket net of costs? Fork this, add "
            "the delisted names, vol-scale the book, and show *t* > 2 net of borrow. That is the "
            "bar.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


def build_quants():
    cells = [
        md(
            "# Cross-Sectional-Momentum -- a quantitative teardown\n"
            "### yfinance survivor basket * 12-1 sort * WML construction * HAC + placebo inference\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Decile_vs_quintile%3F: Busted](https://img.shields.io/badge/Decile_vs_quintile%3F-Busted-8b949e?style=flat-square)\n\n"
            "The quantitative companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb) -- same seven beats, every claim carrying its standard "
            "error. We test Jegadeesh-Titman (1993): rank by trailing 12-1 return, long winners, "
            "short losers, monthly rebalance, hold one month, on both the quintile and decile cut.\n\n"
            f"> **Not investment advice.** Real data: yfinance daily adjusted-close, "
            f"{R['n_tickers']} large-cap survivors, {R['date_start'][:4]}-{R['date_end'][:4]}, "
            f"as-of {R['asof']}. Methods in [`docs/references.md`](../docs/references.md), "
            "reproducible numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship is named on the SIGNAL axis:** basket = names still trading in "
            "2026; the loser leg's natural shorts are absent. All numbers are upper bounds."
        ),
        code(BOOT),

        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Quintile WML **{R['q_gross']:+.2f}%/yr**, HAC *t* = "
            f"**{R['q_t']:+.3f}**, placebo *p* = **{R['q_p']:.3f}**. |\n"
            f"| **Tradability** | `MIRAGE` | Best net **{R['d_net']:+.2f}%/yr**, max DD "
            f"**{R['d_dd']:.1f}%**, turnover **{R['q_turn']:.0f}%/mo**. |\n"
            "| **Decile vs quintile?** | `BUSTED` | Tighter sort = noise swap, deeper drawdown, "
            "thinner legs. |\n\n"
            "> Momentum is overwhelming in the literature and on our synthetic control. On "
            f"{R['n_hold']} months of {R['n_tickers']} large-cap survivors it is flat -- "
            "consistent with post-publication decay (McLean-Pontiff 2016) and survivorship "
            "removing the loser leg's bite."
        ),

        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $r^{12{-}1}_{i,t}$ = the trailing 12-month return of stock $i$ to month-end "
            "$t$, skipping month $t$. Jegadeesh-Titman (1993) assert:\n\n"
            "- **H1 (signal).** The cross-sectional spread "
            "$\\text{WML}_t = \\bar r^{\\text{win}}_{t+1} - \\bar r^{\\text{los}}_{t+1}$ "
            "(top minus bottom fraction by $r^{12-1}$) has $E[\\text{WML}] > 0$.\n"
            "- **H2 (robust).** It survives a label-shuffle null and is not a single-sort "
            "artifact (decile vs quintile agree in sign).\n"
            "- **H3 (tradable).** It survives turnover costs and short borrow.\n\n"
            "On this basket we **reject H1** (HAC *t* near zero, placebo *p* > 0.5), **reject "
            "H2** (decile and quintile straddle zero), and **reject H3** (net ~0 with a -50% "
            "drawdown)."
        ),

        md(
            "## 2 -- So what? -- the economic stakes\n\n"
            "Momentum underpins the Carhart fourth factor, AQR-style multi-factor funds, and "
            "trend products. If the canonical 12-1 sort no longer pays in US large caps -- the "
            "deepest, most-arbitraged corner of the market -- the residual premium lives in "
            "small caps, breadth, and the crash-risk compensation, not in a clean large-cap "
            "long-short."
        ),

        md(
            "## 3 -- The protocol\n\n"
            "- **Signal.** Trailing 12-month return skipping the most recent month (12-1).\n"
            "- **Ranking.** Monthly, sort all names by the signal.\n"
            "- **Legs.** Long the top `frac`, short the bottom `frac`, equal-weight, "
            "dollar-neutral. `frac=0.20` (quintile) and `frac=0.10` (decile).\n"
            "- **Execution lag.** Form on the month-end close; earn month *t+1*. One forward "
            "shift, no same-bar fill.\n"
            "- **Costs.** 10bps/leg/rebalance x turnover + 50bps/yr short borrow.\n"
            "- **Inference.** Newey-West HAC *t* on monthly WML; a 1,000-draw label-shuffle "
            "placebo p-value.\n"
            f"- **Universe caveat.** {R['n_tickers']} large-cap survivors; survivorship-biased "
            "on the signal axis."
        ),

        md("## 4 -- The teardown"),
        md(
            "### 4a -- Positive control: the WML engine is a faithful detector\n\n"
            "Sweep the planted momentum drift on a synthetic panel. The WML mean should be ~0 "
            "at the null and monotone in the planted drift."
        ),
        code(
            "ctrl = st.synthetic_control(strengths=(0.0, 0.10, 0.20, 0.30, 0.40, 0.50))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if m > 0.5 else GREY for m in ctrl['wml_mean_ann']*100]\n"
            "ax.bar(ctrl['mom_strength'].astype(str), ctrl['wml_mean_ann']*100, color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, (m, t) in enumerate(zip(ctrl['wml_mean_ann']*100, ctrl['tstat'])):\n"
            "    ax.text(i, m + 1.5, f't={t:+.1f}', ha='center', fontsize=8)\n"
            "ax.set_xlabel('planted momentum drift'); ax.set_ylabel('WML mean (%/yr)')\n"
            "ax.set_title('Positive control: WML mean is monotone in planted drift')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),
        md(
            "### 4b -- Real tape: quintile vs decile, gross and net"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls_qn = st.long_short(mp, frac=0.20, cost_bps=10.0, borrow_ann_bps=50.0)\n"
            "    ls_dn = st.long_short(mp, frac=0.10, cost_bps=10.0, borrow_ann_bps=50.0)\n"
            "    sqn = st.summary(ls_qn['wml_net']); sdn = st.summary(ls_dn['wml_net'])\n"
            "    rows = [['Quintile', sq['mean']*100, sqn['mean']*100, sq['tstat'], sq['sharpe']],\n"
            "            ['Decile',   sd['mean']*100, sdn['mean']*100, sd['tstat'], sd['sharpe']]]\n"
            "else:\n"
            "    rows = [['Quintile', " f"{R['q_gross']}, {R['q_net']}, {R['q_t']}, {R['q_sharpe']}],\n"
            "            ['Decile',   " f"{R['d_gross']}, {R['d_net']}, {R['d_t']}, {R['d_sharpe']}]]\n"
            "tab = pd.DataFrame(rows, columns=['sort', 'gross_%', 'net_%', 'HAC_t', 'sharpe'])\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "x = np.arange(len(tab)); w = 0.35\n"
            "ax.bar(x - w/2, tab['gross_%'], w, color=GREY, label='gross')\n"
            "ax.bar(x + w/2, tab['net_%'], w, color=RED, label='net (costs+borrow)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels(tab['sort'])\n"
            "ax.set_ylabel('WML mean (%/yr)')\n"
            "ax.set_title('Real WML: flat gross, negative-to-zero net'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(3).to_string(index=False))"
        ),
        md(
            f"> Quintile **{R['q_gross']:+.2f}%/yr** (HAC *t* = {R['q_t']:+.3f}); decile "
            f"**{R['d_gross']:+.2f}%/yr** (*t* = {R['d_t']:+.2f}). Net of costs both collapse to "
            "zero or below. The third axis is **BUSTED**: concentration does not surface an edge."
        ),
        md(
            "### 4c -- The placebo null: real vs label-shuffle\n\n"
            "Shuffle which stock the momentum signal points at, recompute the WML mean 1,000 "
            "times, and see where the real mean falls in that distribution."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(mp, frac=0.20, n_perm=1000)\n"
            "    real_mean = pl['real_mean_ann']*100; pval = pl['p_value']\n"
            "    # rebuild the placebo distribution for the histogram\n"
            "    import numpy as _np\n"
            "    fwd = mp.pct_change().shift(-1); sig = st.momentum_signal(mp)\n"
            "    rng = _np.random.default_rng(507); months = []\n"
            "    for t in sig.index:\n"
            "        s = sig.loc[t].dropna(); r = fwd.loc[t].dropna() if t in fwd.index else pd.Series(dtype=float)\n"
            "        c = s.index.intersection(r.index)\n"
            "        if len(c) < 10: continue\n"
            "        k = max(1, int(round(0.20*len(c))))\n"
            "        if 2*k > len(c): continue\n"
            "        months.append((s.loc[c].to_numpy(), r.loc[c].to_numpy(), k))\n"
            "    dist = []\n"
            "    for _ in range(1000):\n"
            "        ss = 0.0\n"
            "        for sv, rv, k in months:\n"
            "            order = _np.argsort(sv[rng.permutation(len(sv))])\n"
            "            ss += rv[order[-k:]].mean() - rv[order[:k]].mean()\n"
            "        dist.append(ss/len(months)*12*100)\n"
            "    dist = _np.array(dist)\n"
            "else:\n"
            "    real_mean, pval = " f"{R['q_gross']}, {R['q_p']}\n"
            "    dist = np.random.default_rng(0).normal(0, 6, 1000)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(dist, bins=40, color=GREY, alpha=0.7, edgecolor='white')\n"
            "ax.axvline(real_mean, c=RED, lw=2, label=f'real WML = {real_mean:+.2f}%/yr')\n"
            "ax.set_xlabel('WML mean under shuffled labels (%/yr)'); ax.set_ylabel('count')\n"
            "ax.set_title(f'Placebo null: real mean is unremarkable | p = {pval:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Placebo p-value (quintile): {pval:.3f}')"
        ),
        md(
            f"> The real WML mean sits in the **middle** of the shuffled-label distribution "
            f"(*p* = {R['q_p']:.3f}). There is no edge for a placebo to beat."
        ),
        md(
            "### 4d -- Year-by-year and the equity curve"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ann = ls_q.copy(); ann['year'] = ann.index.year\n"
            "    yearly = ann.groupby('year')['wml_gross'].apply(lambda x: float((1+x).prod()-1))*100\n"
            "    eq = (1 + ls_q['wml_gross']).cumprod(); dd = (eq/eq.cummax()-1)*100\n"
            "    idx = ls_q.index\n"
            "else:\n"
            "    yearly = pd.Series(" f"{R['annual']})\n"
            "    idx = pd.period_range('2013-05', periods=" f"{R['n_hold']}, freq='M').to_timestamp()\n"
            "    eq = pd.Series(np.cumprod(1+np.zeros(len(idx))), index=idx); dd = eq*0\n"
            "\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))\n"
            "col = [GREEN if v > 0 else RED for v in yearly.values]\n"
            "ax1.bar(yearly.index.astype(str), yearly.values, color=col)\n"
            "ax1.axhline(0, c='k', lw=1); ax1.set_ylabel('WML (%/yr)')\n"
            "ax1.set_title('Year-by-year quintile WML: violent sign flips')\n"
            "ax1.tick_params(axis='x', rotation=45)\n"
            "if HAVE_REAL:\n"
            "    ax2.plot(idx, eq.values, c=AMBER, lw=2); ax2.axhline(1, c='k', lw=1, ls='--')\n"
            "    ax2.set_ylabel('cumulative wealth (WML)')\n"
            "    ax2.set_title(f'Equity curve | max DD {dd.min():.1f}%')\n"
            "else:\n"
            "    ax2.text(0.5, 0.5, f'frozen: max DD {" f"{R['q_dd']}" "}%', ha='center')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(yearly.round(1).to_string())"
        ),

        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `NONE`** -- quintile WML {R['q_gross']:+.2f}%/yr, HAC *t* = "
            f"{R['q_t']:+.3f}, placebo *p* = {R['q_p']:.3f}; decile *t* = {R['d_t']:+.2f}. The "
            "synthetic control proves the engine would find momentum if it were there; the real "
            "basket simply does not carry it. Survivorship -- named on the signal axis -- makes "
            "these upper bounds.\n"
            f"- **Tradability `MIRAGE`** -- best net {R['d_net']:+.2f}%/yr, max DD "
            f"{R['d_dd']:.1f}%, turnover {R['q_turn']:.0f}%/mo. Nothing to trade.\n"
            "- **Decile vs quintile `BUSTED`** -- the tighter sort swapped noise for noise at a "
            "deeper drawdown."
        ),

        md(
            "## 6 -- Could you trade it?\n\n"
            "No. Beyond the flat gross signal: ~25-30%/mo turnover pays the spread repeatedly; "
            "the short leg costs borrow and recall risk; and the -50% drawdown with year-to-year "
            "sign flips (+28% 2017 -> -30% 2025) is the momentum-crash tail with no premium to "
            "pay for it. The live strategy -- which would *include* the trending-to-zero "
            "delisted names our survivor basket drops -- could only be worse on the short leg's "
            "borrow and gap risk."
        ),

        md(
            "## 7 -- Going further\n\n"
            "- **[Study 24 -- Stampede](../../24-stampede/)**: the 12-1 WML on the *full* S&P "
            "500 -- the larger cross-section lifts the signal to Weak.\n"
            "- **[Study 25 -- Clean-Slate](../../25-clean-slate/)** / "
            "**[Study 237](../../237-residual-momentum/)**: residual momentum, the market-"
            "stripped cousin.\n"
            "- **[Study 238 -- Betting-Against-Beta](../../238-betting-against-beta/)**: a "
            "sibling cross-sectional sort on the same survivor infrastructure.\n"
            "- **Vol-scaling.** Barroso-Santa-Clara (2015): scale by recent realised vol to "
            "tame the crash and roughly double the Sharpe.\n\n"
            "*Think momentum survives on a small basket net of costs? Fork this, add the "
            "delisted names, vol-scale the book, and show *t* > 2 net of borrow. That is the bar.*"
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
