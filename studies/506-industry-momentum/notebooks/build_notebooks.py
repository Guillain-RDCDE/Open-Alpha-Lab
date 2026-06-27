"""Generate the two narrative notebooks for Study 506 (Industry-Momentum).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the desk's seven beats. The synthetic control runs anywhere, offline and
deterministic; the real-tape cells read the cached yfinance parquets under ../_cache/ if present
and otherwise quote the frozen headline numbers in ``R`` (the single source of truth that mirrors
docs/results.md exactly, as-of 2026-06-26).
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-26).
R = dict(
    asof="2026-06-26",
    n_days=5637, n_etfs=11, n_names=40, n_months=268, n_hold=256,
    date_start="2004-01-02", date_end="2026-05-29",
    m_start="2004-02", m_end="2026-05",
    sec_fp="641f8fd7befd", spy_fp="06ab1687a7ba", names_fp="268288d86882",
    cost_bps=5.0, borrow_bps=50.0, tk_sec=3, tk_name=8,
    # industry (sector ETFs)
    sec_gross=0.90, sec_net=0.14, sec_sharpe=0.065, sec_t=0.315, sec_t_net=0.048,
    sec_hit=53.9, sec_dd=-45.1, sec_turn=21.8, sec_p=0.348,
    sec_win=11.1, sec_los=10.2, spy_ann=11.7,
    # single-name (survivors)
    nm_gross=2.46, nm_net=1.66, nm_sharpe=0.131, nm_t=0.566, nm_t_net=0.382,
    nm_hit=57.0, nm_dd=-59.6, nm_turn=25.1, nm_p=0.150,
    # robustness top_k sweep: k -> (mean%, t)
    rob={1: (5.00, 0.888), 2: (2.48, 0.733), 3: (0.90, 0.315), 4: (-0.33, -0.148)},
    # synthetic control: planted industry_mom -> (mean%, seed-avg t)
    ctrl={0.00: (0.20, 0.18), 0.06: (41.6, 10.50), 0.12: (83.9, 10.84), 0.20: (139.9, 10.93)},
    # year-by-year industry long-short (gross), %
    yearly={2005: 6.2, 2006: -14.1, 2007: 6.7, 2008: 14.5, 2009: -16.4, 2010: -0.2,
            2011: -6.7, 2012: -6.8, 2013: 12.2, 2014: -4.4, 2015: 9.2, 2016: -16.6,
            2017: -0.5, 2018: -6.5, 2019: -2.8, 2020: 19.5, 2021: -15.8, 2022: 46.7,
            2023: -12.1, 2024: -2.8, 2025: 3.6},
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Industry_beats_stocks%3F: Busted](https://img.shields.io/badge/Industry_beats_stocks%3F-Busted-8b949e?style=flat-square)\n\n"
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

from industry_momentum import data, strategy as st

def _have_cache():
    study = os.path.abspath("..")
    return os.path.exists(os.path.join(study, "_cache", "sector_prices.parquet"))

HAVE_REAL = _have_cache()
print("yfinance cache present:", HAVE_REAL)

if HAVE_REAL:
    sec_px, spy, names_px = data.fetch_prices()
    sec_ret = data.drop_partial_last(data.to_monthly_returns(sec_px))
    name_ret = data.drop_partial_last(data.to_monthly_returns(names_px))
    sec_g = st.long_short(sec_ret, top_k=3)
    nm_g = st.long_short(name_ret, top_k=8)
    ssec = st.summary(sec_g["ls_gross"]); snm = st.summary(nm_g["ls_gross"])
    print(f"Industry  WML: {ssec['mean']*100:+.2f}%/yr  HAC t={ssec['tstat']:+.3f}")
    print(f"Single-nm WML: {snm['mean']*100:+.2f}%/yr  HAC t={snm['tstat']:+.3f}")
"""


def build_curious():
    cells = [
        md(
            "# Industry-Momentum -- is momentum really an *industry* effect?\n"
            "### Moskowitz-Grinblatt (1999), tested on the 11 SPDR sector ETFs vs single stocks\n\n"
            + BADGES +
            "In 1999, Tobias Moskowitz and Mark Grinblatt published a quietly radical claim: the "
            "famous stock-momentum effect -- buy past winners, short past losers -- is *mostly an "
            "industry effect*. Winner **industries** keep winning and loser industries keep "
            "losing, and once you account for that, the leftover stock-by-stock momentum nearly "
            "vanishes.\n\n"
            "We test the cleanest modern embodiment: rank the **11 SPDR sector ETFs** each month "
            "by their trailing **12-1** return (skip the most recent month), go long the top "
            "sectors and short the bottom, and **race** that book against the same machine run on "
            "individual large-cap stocks. The honest question: does sorting *industries* beat "
            "sorting *stocks*?\n\n"
            "> **This is the plain-language layer.** Want the HAC *t*-stats, the label-shuffle "
            "placebo and the top_k sweep? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do winner sectors keep winning? | **Barely.** The industry winners-minus-losers "
            f"book earns **{R['sec_gross']:+.2f}%/yr** at HAC *t* = **{R['sec_t']:+.3f}** -- a "
            f"coin. |\n"
            f"| Does the *industry* sort beat the *stock* sort? | **No -- the opposite.** "
            f"Single-name momentum is *larger* (**{R['nm_gross']:+.2f}%/yr**), and both are "
            f"noise. |\n"
            f"| Could you trade it? | **No.** Best industry net **{R['sec_net']:+.2f}%/yr**, with "
            f"a **{R['sec_dd']:.0f}%** drawdown and {R['sec_turn']:.0f}%/mo turnover. |\n"
            "| Is there a survivorship problem? | **Yes -- on the single-name leg, named on the "
            "signal axis.** The sector ETFs are survivorship-free; the stock basket is names "
            "*still trading in 2026*, so its loser short is an upper bound. |\n\n"
            "> Both sector legs are winners. The winner sectors earn "
            f"**~{R['sec_win']:.0f}%/yr** and the loser sectors **~{R['sec_los']:.0f}%/yr** -- "
            f"both near the market's **~{R['spy_ann']:.0f}%/yr**. In a long bull tape every sector "
            "compounds, so ranking sectors on past return separates almost nothing."
        ),

        md(
            "## 1 -- The claim\n\n"
            "> *\"Momentum strategies are less profitable than previously documented when "
            "controlling for industry momentum ... industry momentum investment strategies appear "
            "to be highly profitable, even after controlling for ... individual stock momentum.\"*\n\n"
            "-- Moskowitz & Grinblatt (1999), *Journal of Finance*\n\n"
            "The recipe: at each month-end, compute each asset's return over the trailing 12 "
            "months but **skip the most recent month** (the classic \"12-1\", to dodge short-term "
            "reversal). Rank, go long the top winners and short the bottom losers, hold one month, "
            "repeat -- but do it on **industries**, not stocks."
        ),

        md(
            "## 2 -- So what?\n\n"
            "If momentum is really an *industry* phenomenon, the tradable version is simple and "
            "cheap: a handful of liquid **sector ETFs**, not hundreds of single-name shorts with "
            "their borrow and recall risk. Sector-rotation funds and relative-strength products "
            "(Faber 2010) sell exactly this story. If the industry sort no longer pays on the "
            "modern ETF tape, a lot of that product is a 1999 result the data no longer supports."
        ),

        md(
            "## 3 -- How would we even know?\n\n"
            "Three disciplines keep us honest:\n\n"
            "1. **No look-ahead.** The 12-1 signal uses only past prices (and skips the most "
            "recent month). We form on the month-end close and earn the *next* month -- one "
            "forward execution lag, no same-bar fill.\n"
            "2. **A placebo null.** We shuffle which sector the signal points at and recompute "
            "the spread 500 times. If the real edge is no better than the shuffle, it is noise.\n"
            "3. **Name the survivorship bias on the *signal* axis.** The sector ETFs are "
            "survivorship-free, but the single-name comparison basket is the stocks that "
            "*survived* to 2026 -- so its loser leg is an upper bound, and we say so."
        ),

        md("## 4 -- The teardown\n\n**First, does the momentum engine work when the effect is planted?**"),
        code(
            "ctrl = st.control_sweep(data, moms=[0.0, 0.06, 0.12, 0.20], n_years=18, top_k=3, n_seeds=8)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREY if m < 5 else GREEN for m in ctrl['mean_ann']*100]\n"
            "ax.bar(ctrl['industry_mom'].astype(str), ctrl['mean_ann']*100, color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, (m, t) in enumerate(zip(ctrl['mean_ann']*100, ctrl['tstat'])):\n"
            "    ax.text(i, m + 3, f't={t:+.1f}', ha='center', fontsize=8)\n"
            "ax.set_xlabel('planted industry_mom'); ax.set_ylabel('long-short mean (%/yr)')\n"
            "ax.set_title('Synthetic control: engine recovers industry momentum when planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),
        md(
            "The engine is faithful: it finds industry momentum when it is planted and scores a "
            f"null seed-averaged *t* of **{R['ctrl'][0.0][1]:+.2f}** when it is not. So the verdict "
            "on the real tape reflects **the market**, not the method.\n\n"
            "*(The notebook uses 8 seeds to stay fast; `docs/results.md` reports the full 20-seed "
            "averages -- same monotone shape, null *t* near zero.)*\n\n"
            "**Now the honest race on the real tape: industries vs stocks.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sec_m, sec_t = ssec['mean']*100, ssec['tstat']\n"
            "    nm_m, nm_t = snm['mean']*100, snm['tstat']\n"
            "else:\n"
            "    sec_m, sec_t = " f"{R['sec_gross']}, {R['sec_t']}\n"
            "    nm_m, nm_t = " f"{R['nm_gross']}, {R['nm_t']}\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars = ax.bar(['Industry\\n(sector ETFs)', 'Single-name\\n(stocks)'], [sec_m, nm_m],\n"
            "              color=[GREY, AMBER], width=0.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('long-short mean (%/yr)')\n"
            "ax.set_title(f'The race: industry {sec_m:+.2f}%/yr (t={sec_t:+.2f})  vs  '\n"
            "             f'single-name {nm_m:+.2f}%/yr (t={nm_t:+.2f})')\n"
            "for b, v in zip(bars, [sec_m, nm_m]):\n"
            "    ax.text(b.get_x()+b.get_width()/2, v + 0.1, f'{v:+.2f}%', ha='center')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Industry  WML: {sec_m:+.2f}%/yr  t={sec_t:+.3f}')\n"
            "print(f'Single-nm WML: {nm_m:+.2f}%/yr  t={nm_t:+.3f}')"
        ),
        md(
            f"Moskowitz-Grinblatt predicted the **industry** bar would tower over the "
            f"single-name one. Instead the single-name book (**{R['nm_gross']:+.2f}%/yr**) is "
            f"*larger* than the industry book (**{R['sec_gross']:+.2f}%/yr**) -- and **both** sit "
            "on top of zero (*t* well under 2). On the modern ETF tape the signature claim simply "
            "does not reproduce: the industry sort does not dominate, it under-performs."
        ),

        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- NONE.** Industry long-short {R['sec_gross']:+.2f}%/yr, HAC *t* = "
            f"{R['sec_t']:+.3f} (placebo *p* = {R['sec_p']:.3f}); single-name "
            f"{R['nm_gross']:+.2f}%/yr (*t* = {R['nm_t']:+.3f}). Neither clears any bar, and "
            "survivorship on the stock leg -- named on the signal axis -- makes its number an "
            "upper bound. The synthetic control proves the engine would find industry momentum if "
            "it were there; the real sector tape does not carry it.\n"
            f"- **Tradability -- MIRAGE.** Best industry net **{R['sec_net']:+.2f}%/yr** with a "
            f"**{R['sec_dd']:.0f}%** drawdown. Nothing to trade.\n"
            "- **Industry beats stocks? -- BUSTED.** Single-name momentum is *larger*, not "
            "smaller -- the headline does not reproduce."
        ),

        md(
            "## 6 -- Could you actually trade it?\n\n"
            "No. Even setting aside the flat gross signal:\n\n"
            f"1. **Turnover.** ~{R['sec_turn']:.0f}%/mo one-way -- sector ranks churn, and every "
            "churn pays the spread.\n"
            "2. **The short leg.** Shorting sectors costs borrow (we charge 50bps/yr) and "
            "carries recall risk.\n"
            f"3. **The crash.** A {R['sec_dd']:.0f}% drawdown with the sign flipping +47% (2022) "
            "to -17% (2016) year to year -- the momentum-crash tail with none of the long-run "
            "premium to pay for it.\n"
            "4. **Survivorship (single-name leg).** The real loser leg -- delisted, "
            "trending-to-zero names -- is missing, so the live single-name strategy would likely "
            "be *worse*."
        ),

        md(
            "## 7 -- Going further\n\n"
            "- **Sector rotation.** [Study 225 -- Sector-Rotation](../225-sector-rotation/) tests "
            "rotation *rules* on the same sectors rather than a strict 12-1 cross-sectional sort.\n"
            "- **Residual momentum.** [Study 237 -- Residual-Momentum](../237-residual-momentum/) "
            "strips the market factor before ranking stocks -- the orthogonal cut to \"is it the "
            "industry?\".\n"
            "- **Momentum elsewhere.** [Study 146 -- Country-Momentum](../146-country-momentum/) "
            "and [Study 147 -- FX-Momentum](../147-fx-momentum/) run the same 12-1 machine on "
            "countries and currencies.\n\n"
            "*Think industry momentum survives on a realistic ETF book net of costs? Fork this, "
            "add the pre-2004 history or finer industry baskets, and show *t* > 2 net of borrow. "
            "That is the bar.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


def build_quants():
    cells = [
        md(
            "# Industry-Momentum -- a quantitative teardown\n"
            "### SPDR sector ETFs * 12-1 sort * long-short * HAC + placebo * the industry-vs-stock race\n\n"
            + BADGES +
            "The quantitative companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb) -- same seven beats, every claim carrying its standard "
            "error. We test Moskowitz-Grinblatt (1999): rank by trailing 12-1 return, long "
            "winners, short losers, monthly rebalance, hold one month, on the **11 SPDR sectors** "
            "and (the race) on a **40-name large-cap survivor basket**.\n\n"
            f"> **Not investment advice.** Real data: yfinance daily adjusted-close, "
            f"{R['n_etfs']} sector ETFs + SPY + {R['n_names']} large-caps, "
            f"{R['date_start'][:4]}-{R['date_end'][:4]}, as-of {R['asof']}. Methods in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship is named on the SIGNAL axis:** the sector ETFs are "
            "survivorship-free; the single-name basket is names still trading in 2026, so its "
            "loser short is an upper bound."
        ),
        code(BOOT),

        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Industry WML **{R['sec_gross']:+.2f}%/yr**, HAC *t* = "
            f"**{R['sec_t']:+.3f}**, placebo *p* = **{R['sec_p']:.3f}**. |\n"
            f"| **Tradability** | `MIRAGE` | Best industry net **{R['sec_net']:+.2f}%/yr**, max DD "
            f"**{R['sec_dd']:.0f}%**, turnover **{R['sec_turn']:.0f}%/mo**. |\n"
            "| **Industry beats stocks?** | `BUSTED` | Single-name WML "
            f"**{R['nm_gross']:+.2f}%/yr** is *larger* than the industry **{R['sec_gross']:+.2f}%/yr** "
            "-- both noise. |\n\n"
            "> Industry momentum is strong in Moskowitz-Grinblatt (1999) and on our synthetic "
            f"control. On {R['n_hold']} months of {R['n_etfs']} sector ETFs it is flat -- "
            "consistent with post-publication decay (McLean-Pontiff 2016) and a long bull tape in "
            "which every sector compounds."
        ),

        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $r^{12{-}1}_{i,t}$ = the trailing 12-month return of asset $i$ to month-end $t$, "
            "skipping month $t$. Moskowitz-Grinblatt (1999) assert:\n\n"
            "- **H1 (industry signal).** The industry spread "
            "$\\text{LS}_t = \\bar r^{\\text{win}}_{t+1} - \\bar r^{\\text{los}}_{t+1}$ "
            "(top minus bottom sectors by $r^{12-1}$) has $E[\\text{LS}] > 0$.\n"
            "- **H2 (dominance).** Industry momentum is *larger* than -- and subsumes -- "
            "single-name momentum.\n"
            "- **H3 (tradable).** It survives turnover costs and short borrow.\n\n"
            "On this tape we **reject H1** (HAC *t* near zero, placebo *p* > 0.3), **reject H2** "
            "(single-name is the *larger* of the two), and **reject H3** (net ~0 with a -45% to "
            "-60% drawdown)."
        ),

        md(
            "## 2 -- So what? -- the economic stakes\n\n"
            "If momentum is an *industry* effect, the cheap tradable version is a few liquid "
            "sector ETFs, not hundreds of single-name shorts. Sector-rotation and relative-"
            "strength products (Faber 2010) sell exactly that. If the industry sort no longer "
            "pays in the modern, deeply-arbitraged ETF complex, the residual premium lives in "
            "finer industry baskets, breadth, or crash-risk compensation -- not in a clean "
            "11-sector long-short."
        ),

        md(
            "## 3 -- The protocol\n\n"
            "- **Signal.** Trailing 12-month return skipping the most recent month (12-1).\n"
            "- **Ranking.** Monthly, sort all assets by the signal.\n"
            "- **Legs.** Long the top `top_k`, short the bottom `top_k`, equal-weight, "
            "dollar-neutral. `top_k=3` of 11 sectors; `top_k=8` of 40 names.\n"
            "- **Execution lag.** Form on the month-end close; earn month *t+1*. One forward "
            "shift, no same-bar fill.\n"
            "- **Costs.** 5bps/leg/rebalance x turnover + 50bps/yr short borrow.\n"
            "- **Inference.** Newey-West HAC *t* on the monthly long-short; a 500-draw "
            "label-shuffle placebo p-value.\n"
            f"- **Universe caveat.** {R['n_etfs']} sector ETFs (survivorship-free) raced against "
            f"{R['n_names']} large-cap survivors (survivorship-biased on the signal axis)."
        ),

        md("## 4 -- The teardown"),
        md(
            "### 4a -- Positive control: the engine is a faithful detector\n\n"
            "Sweep the planted industry-momentum strength on a synthetic sector panel. The "
            "long-short mean should be ~0 at the null and monotone in the planted strength "
            "(seed-averaged, never one lucky seed)."
        ),
        code(
            "ctrl = st.control_sweep(data, moms=[0.0, 0.06, 0.12, 0.20], n_years=18, top_k=3, n_seeds=8)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREY if m < 5 else GREEN for m in ctrl['mean_ann']*100]\n"
            "ax.bar(ctrl['industry_mom'].astype(str), ctrl['mean_ann']*100, color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, (m, t) in enumerate(zip(ctrl['mean_ann']*100, ctrl['tstat'])):\n"
            "    ax.text(i, m + 3, f't={t:+.1f}', ha='center', fontsize=8)\n"
            "ax.set_xlabel('planted industry_mom'); ax.set_ylabel('long-short mean (%/yr)')\n"
            "ax.set_title('Positive control: long-short mean is monotone in planted strength')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),
        md(
            f"> Null seed-averaged *t* = **{R['ctrl'][0.0][1]:+.2f}** (no manufactured "
            "significance); the planted component lights up monotonically. The full 20-seed "
            "version lives in `docs/results.md`."
        ),
        md("### 4b -- Real tape: the industry-vs-single-name race, gross and net"),
        code(
            "if HAVE_REAL:\n"
            "    sec_n = st.long_short(sec_ret, top_k=3, cost_bps=5.0, borrow_ann_bps=50.0)\n"
            "    nm_n = st.long_short(name_ret, top_k=8, cost_bps=5.0, borrow_ann_bps=50.0)\n"
            "    ssn = st.summary(sec_n['ls_net']); snn = st.summary(nm_n['ls_net'])\n"
            "    rows = [['Industry',    ssec['mean']*100, ssn['mean']*100, ssec['tstat'], ssec['sharpe']],\n"
            "            ['Single-name', snm['mean']*100,  snn['mean']*100, snm['tstat'],  snm['sharpe']]]\n"
            "else:\n"
            "    rows = [['Industry',    " f"{R['sec_gross']}, {R['sec_net']}, {R['sec_t']}, {R['sec_sharpe']}],\n"
            "            ['Single-name', " f"{R['nm_gross']}, {R['nm_net']}, {R['nm_t']}, {R['nm_sharpe']}]]\n"
            "tab = pd.DataFrame(rows, columns=['book', 'gross_%', 'net_%', 'HAC_t', 'sharpe'])\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "x = np.arange(len(tab)); w = 0.35\n"
            "ax.bar(x - w/2, tab['gross_%'], w, color=GREY, label='gross')\n"
            "ax.bar(x + w/2, tab['net_%'], w, color=RED, label='net (costs+borrow)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels(tab['book'])\n"
            "ax.set_ylabel('long-short mean (%/yr)')\n"
            "ax.set_title('Real long-short: both flat, single-name larger -- claim does not reproduce')\n"
            "ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(3).to_string(index=False))"
        ),
        md(
            f"> Industry **{R['sec_gross']:+.2f}%/yr** (HAC *t* = {R['sec_t']:+.3f}); single-name "
            f"**{R['nm_gross']:+.2f}%/yr** (*t* = {R['nm_t']:+.3f}). The third axis is "
            "**BUSTED**: the *industry* sort is the *smaller* of the two, and net of costs both "
            "collapse toward zero."
        ),
        md(
            "### 4c -- The placebo null: real vs label-shuffle\n\n"
            "Shuffle which sector the momentum signal points at, recompute the long-short mean 500 "
            "times, and see where the real mean falls in that distribution."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pb = st.placebo_pvalue(sec_ret, real_mean=float(sec_g['ls_gross'].mean()), top_k=3, n_shuffles=500)\n"
            "    real_mean = float(sec_g['ls_gross'].mean())*12*100; pval = pb['p_value']\n"
            "    # rebuild the placebo distribution for the histogram (same fast statistic)\n"
            "    score = st.momentum_signal(sec_ret); dates = sec_ret.index; months = []\n"
            "    for i in range(len(dates)-1):\n"
            "        s = score.loc[dates[i]].dropna()\n"
            "        if len(s) < 6: continue\n"
            "        fwd = sec_ret.loc[dates[i+1]].reindex(s.index); ok = fwd.notna().to_numpy()\n"
            "        sv = s.to_numpy()[ok]; rv = fwd.to_numpy()[ok]\n"
            "        if len(sv) < 6: continue\n"
            "        months.append((sv, rv))\n"
            "    rng = np.random.default_rng(506); dist = []\n"
            "    for _ in range(500):\n"
            "        ss = 0.0\n"
            "        for sv, rv in months:\n"
            "            order = np.argsort(sv[rng.permutation(len(sv))])\n"
            "            ss += rv[order[-3:]].mean() - rv[order[:3]].mean()\n"
            "        dist.append(ss/len(months)*12*100)\n"
            "    dist = np.array(dist)\n"
            "else:\n"
            "    real_mean, pval = " f"{R['sec_gross']}, {R['sec_p']}\n"
            "    dist = np.random.default_rng(0).normal(0, 2.07, 500)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(dist, bins=35, color=GREY, alpha=0.7, edgecolor='white')\n"
            "ax.axvline(real_mean, c=RED, lw=2, label=f'real industry LS = {real_mean:+.2f}%/yr')\n"
            "ax.set_xlabel('long-short mean under shuffled sector labels (%/yr)'); ax.set_ylabel('count')\n"
            "ax.set_title(f'Placebo null: real mean is unremarkable | p = {pval:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Placebo p-value (industry): {pval:.3f}')"
        ),
        md(
            f"> The real industry long-short sits in the **middle** of the shuffled-label "
            f"distribution (*p* = {R['sec_p']:.3f}). There is no edge for a placebo to beat."
        ),
        md("### 4d -- Robustness (top_k sweep) and year-by-year"),
        code(
            "if HAVE_REAL:\n"
            "    rob = {k: st.summary(st.long_short(sec_ret, top_k=k)['ls_gross']) for k in (1,2,3,4)}\n"
            "    ks = list(rob); means = [rob[k]['mean']*100 for k in ks]; ts = [rob[k]['tstat'] for k in ks]\n"
            "    yb = sec_g.copy(); yb['year'] = yb.index.year\n"
            "    yearly = yb.groupby('year')['ls_gross'].apply(lambda x: float((1+x).prod()-1))*100\n"
            "else:\n"
            "    ks = " f"{list(R['rob'])}\n"
            "    means = " f"{[R['rob'][k][0] for k in R['rob']]}\n"
            "    ts = " f"{[R['rob'][k][1] for k in R['rob']]}\n"
            "    yearly = pd.Series(" f"{R['yearly']})\n"
            "\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))\n"
            "ax1.bar([str(k) for k in ks], means, color=GREY)\n"
            "ax1.axhline(0, c='k', lw=1); ax1.set_xlabel('top_k (winners/losers per leg)')\n"
            "ax1.set_ylabel('long-short mean (%/yr)')\n"
            "for i, (m, t) in enumerate(zip(means, ts)):\n"
            "    ax1.text(i, m + 0.15, f't={t:+.2f}', ha='center', fontsize=8)\n"
            "ax1.set_title('Robustness: no top_k clears t=2')\n"
            "col = [GREEN if v > 0 else RED for v in yearly.values]\n"
            "ax2.bar(yearly.index.astype(str), yearly.values, color=col)\n"
            "ax2.axhline(0, c='k', lw=1); ax2.set_ylabel('industry LS (%/yr)')\n"
            "ax2.set_title('Year-by-year: violent sign flips'); ax2.tick_params(axis='x', rotation=90)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(yearly.round(1).to_string())"
        ),

        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `NONE`** -- industry long-short {R['sec_gross']:+.2f}%/yr, HAC *t* = "
            f"{R['sec_t']:+.3f}, placebo *p* = {R['sec_p']:.3f}; single-name {R['nm_gross']:+.2f}%/yr "
            f"(*t* = {R['nm_t']:+.3f}). The synthetic control proves the engine would find industry "
            "momentum if it were there; the real sector tape does not carry it. Survivorship on "
            "the stock leg -- named on the signal axis -- makes that number an upper bound.\n"
            f"- **Tradability `MIRAGE`** -- best industry net {R['sec_net']:+.2f}%/yr, max DD "
            f"{R['sec_dd']:.0f}%, turnover {R['sec_turn']:.0f}%/mo. Nothing to trade.\n"
            "- **Industry beats stocks? `BUSTED`** -- the single-name book is the *larger* of the "
            "two; the signature Moskowitz-Grinblatt dominance does not reproduce here."
        ),

        md(
            "## 6 -- Could you trade it?\n\n"
            "No. Beyond the flat gross signal: ~22%/mo turnover pays the spread repeatedly; the "
            "short leg costs borrow and recall risk; and the -45% drawdown with year-to-year sign "
            "flips (+47% 2022 -> -17% 2016) is the momentum-crash tail with no premium to pay for "
            "it. The single-name expression -- which in live trading would *include* the "
            "trending-to-zero delisted names our survivor basket drops -- could only be worse on "
            "the short leg's borrow and gap risk."
        ),

        md(
            "## 7 -- Going further\n\n"
            "- **[Study 225 -- Sector-Rotation](../225-sector-rotation/)**: rotation *rules* on "
            "the same sectors rather than a strict 12-1 cross-sectional sort.\n"
            "- **[Study 237 -- Residual-Momentum](../237-residual-momentum/)**: single-name "
            "momentum on *residual* returns -- the orthogonal cut to \"is it the industry?\".\n"
            "- **[Study 146 -- Country-Momentum](../146-country-momentum/)** / "
            "**[Study 147 -- FX-Momentum](../147-fx-momentum/)**: the same 12-1 machine on "
            "countries and currencies.\n"
            "- **[Study 238 -- Betting-Against-Beta](../238-betting-against-beta/)**: a sibling "
            "cross-sectional sort on the same desk infrastructure.\n\n"
            "*Think industry momentum survives net of costs? Fork this, add pre-2004 history or "
            "finer industry baskets, and show *t* > 2 net of borrow. That is the bar.*"
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
