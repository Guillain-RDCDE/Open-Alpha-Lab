"""Generate the two narrative notebooks for Study 527 (Organizational-Capital).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats. The synthetic figures run anywhere, offline and
deterministic; the real-tape cells use the cached EDGAR + yfinance data under ../_cache/ if
present, otherwise they quote the frozen headline numbers in ``R`` (which mirror
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-26).
R = dict(
    n_months=203, year_start=2009, year_end=2025, n_firms_fund=38, n_priced=39,
    fp_fund="0acb116f3220", fp_px="26dc45209cc1",
    ls_gross=-1.68, ls_vol=7.60, ls_sharpe=-0.221, ls_t=-0.923,
    ls_hit=52.2, ls_dd=-31.4,
    high_leg=14.96, low_leg=16.65,
    ls_net=-2.28, ls_net_t=-1.248,
    placebo_p=0.326, placebo_real=-0.001403, placebo_null=0.000127,
    placebo_nullstd=0.001373, placebo_n=500,
    delta_sweep={0.10: (-1.71, -0.920), 0.15: (-1.68, -0.923),
                 0.20: (-2.06, -1.130), 0.30: (-1.93, -1.051)},
    annual={2009: -14.9, 2010: -3.0, 2011: 3.4, 2012: 8.2, 2013: -3.1,
            2014: 0.8, 2015: -0.2, 2016: -10.2, 2017: 3.7, 2018: 6.2,
            2019: -1.5, 2020: 2.7, 2021: 15.1, 2022: -7.8, 2023: -10.1,
            2024: -3.9, 2025: -13.4},
    synth=[(-0.06, -10.17, -4.41), (-0.03, -5.38, -2.36), (0.00, -0.60, -0.27),
           (0.03, 4.19, 1.83), (0.06, 8.98, 3.92), (0.10, 15.36, 6.62)],
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

from organizational_capital import data, strategy as st

def _have_cache():
    study = os.path.abspath("..")
    return (os.path.exists(os.path.join(study, "_cache", "oc_fundamentals.parquet"))
            and os.path.exists(os.path.join(study, "_cache", "oc_prices.parquet")))

HAVE_REAL = _have_cache()
print("EDGAR + yfinance cache present:", HAVE_REAL)

if HAVE_REAL:
    fund = data.fetch_fundamentals()
    prices = data.fetch_prices()
    oc = st.org_capital_stock(fund)
    monthly = prices.resample("ME").last().pct_change().dropna(how="all")
    ls = st.long_short(oc, monthly)
    s_ls = st.summary(ls["ls"]); s_hi = st.summary(ls["high"]); s_lo = st.summary(ls["low"])
    print(f"N months: {s_ls['n']} | LS: {s_ls['mean']*100:+.2f}%/yr | HAC t: {s_ls['tstat']:+.3f}")
"""


def build_curious():
    cells = [
        md(
            "# Organizational-Capital -- does accumulated know-how earn a premium?\n"
            "### Eisfeldt-Papanikolaou (2013): SG&A as investment, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Premium-sign: Busted](https://img.shields.io/badge/Premium--sign-Busted-8b949e?style=flat-square)\n\n"
            "In 2013, Andrea Eisfeldt and Dimitris Papanikolaou published a clever idea: a firm's "
            "**organizational capital** -- the know-how, processes, and culture built up over years "
            "of spending on Selling, General & Administrative (SG&A) -- is a real, productive asset, "
            "even though accountants expense it. They argued high-org-capital firms should earn "
            "**higher** returns, because that capital is partly embodied in *talent that can walk "
            "out the door*: shareholders bear that mobility risk and demand a premium.\n\n"
            "We capitalize SG&A by perpetual inventory, scale by assets, and sort a large-cap "
            "**survivor** basket -- naming the survivorship bias and applying one execution lag.\n\n"
            "> **This is the plain-language layer.** Want the perpetual-inventory build, the HAC "
            "*t*, the placebo, and the 12-seed control? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool."
        ),
        code(BOOT),
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do high-OC firms out-earn low-OC firms? | **No, on this tape.** The high-minus-low "
            f"long-short earns **{R['ls_gross']:+.2f}%/yr**, HAC *t* = **{R['ls_t']:+.2f}** -- "
            "below the bar *and the wrong sign*. |\n"
            "| Is the sign what E-P predicted? | **No.** They predict high-OC **>** low-OC; here "
            f"the high-OC half ({R['high_leg']:.1f}%/yr) **trailed** the low-OC half "
            f"({R['low_leg']:.1f}%/yr). |\n"
            "| Could you trade it? | **No.** The gross spread is already negative; costs only "
            f"deepen it to **{R['ls_net']:+.2f}%/yr**. |\n"
            "| Is the engine broken? | **No.** The synthetic control recovers a planted premium "
            "with the right sign -- so the flip is **the tape**, not the method. |\n\n"
            "> The org-capital premium is a genuine, well-cited *risk story*. On a 39-name large-cap "
            "survivor basket it simply does not show up -- if anything it points the wrong way."
        ),
        md(
            "## 1 -- The claim\n\n"
            "> *\"Organization capital is a production factor embodied in the firm's key talent. "
            "Since this talent is mobile and has outside options, shareholders of high-org-capital "
            "firms bear more risk and earn higher average returns.\"*\n\n"
            "-- Eisfeldt & Papanikolaou (2013), *Journal of Finance*\n\n"
            "The accounting trick: treat **SG&A as investment**, not pure expense. Accumulate it "
            "into a stock that depreciates ~15%/yr (a perpetual inventory, exactly as you'd build "
            "a physical-capital stock from capex), then scale by book assets to compare across "
            "firm sizes. Sort. The high pile is predicted to pay."
        ),
        md(
            "## 2 -- So what?\n\n"
            "If real, org-capital is an intangible risk factor that standard book-value accounting "
            "misses entirely -- and a whole literature (Peters-Taylor 2017; Eisfeldt-Falato-Xiaolan "
            "2023) builds on capitalizing SG&A and R&D. The questions for us:\n\n"
            "1. **Does the premium survive on a realistic large-cap panel?**\n"
            "2. **Does it even have the right sign?**\n"
            "3. **What does survivorship do to the risk story?** (The high-risk tail E-P load on "
            "the long leg are exactly the firms that *delist* -- and they're missing here.)"
        ),
        md(
            "## 3 -- How would we even know?\n\n"
            "Three disciplines keep us honest:\n\n"
            "1. **Capitalize SG&A by perpetual inventory** with a published recipe (delta=15%, "
            "steady-state seed) -- no free parameters tuned to the answer.\n"
            "2. **One execution lag.** The annual rank uses fiscal-year fundamentals already public "
            "(prior year); positions enter the month after and are held 12 months. No look-ahead.\n"
            "3. **Name the survivorship bias.** The basket is current large-caps still trading in "
            "2026. Delisted firms -- the natural high-risk long-leg names -- are absent. Any positive "
            "result would be an **upper bound**."
        ),
        md("## 4 -- The teardown\n\n**First, does the engine work when the premium is planted?**"),
        code(
            "ctrl = st.synthetic_control()\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if m > 0.3 else (RED if m < -0.3 else GREY) for m in ctrl['mean_ls_pct']]\n"
            "ax.bar(ctrl['premium'].astype(str), ctrl['mean_ls_pct'], color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted OC premium'); ax.set_ylabel('long-short (%/yr, 12-seed avg)')\n"
            "ax.set_title('Synthetic control: engine recovers the planted premium, right sign')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(2).to_string(index=False))"
        ),
        md(
            "The engine is faithful: the long-short is monotone in the planted premium and has the "
            "**correct sign**. So whatever the real tape says reflects **the market**, not a bug."
        ),
        md("**Now the honest test on the real EDGAR + yfinance tape.**"),
        code(
            "if HAVE_REAL:\n"
            "    hi, lo = s_hi['mean']*100, s_lo['mean']*100\n"
            "    lsm, lst = s_ls['mean']*100, s_ls['tstat']\n"
            "    ls2 = ls.copy(); ls2['year'] = pd.DatetimeIndex(ls2.index).year\n"
            "    annual = ls2.groupby('year')['ls'].apply(lambda x: float((1+x).prod()-1))*100\n"
            "else:\n"
            "    hi, lo = R['high_leg'], R['low_leg']\n"
            "    lsm, lst = R['ls_gross'], R['ls_t']\n"
            "    annual = pd.Series(R['annual'])\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))\n"
            "axes[0].bar(['High-OC\\n(long)', 'Low-OC\\n(short)'], [hi, lo], color=[AMBER, GREY], width=.5)\n"
            "axes[0].set_ylabel('mean annual return (%/yr)')\n"
            "axes[0].set_title('Legs: high-OC TRAILS low-OC (wrong sign)')\n"
            "col_yr = [GREEN if v > 0 else RED for v in annual.values]\n"
            "axes[1].bar(annual.index, annual.values, color=col_yr)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_xlabel('year'); axes[1].set_ylabel('long-short (%/yr)')\n"
            "axes[1].set_title(f'Year-by-year | mean {lsm:+.1f}% t={lst:+.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'High-OC minus low-OC: {lsm:+.2f}%/yr | HAC t = {lst:+.3f}')"
        ),
        md(
            f"The high-minus-low spread earns **{R['ls_gross']:+.2f}%/yr** at HAC *t* = "
            f"**{R['ls_t']:+.2f}** -- below the bar and pointing the *wrong* way. High-OC large-caps "
            "did not out-earn low-OC ones over 2009-2025; they slightly **under**performed."
        ),
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- NONE.** {R['ls_gross']:+.2f}%/yr, HAC *t* = {R['ls_t']:+.2f}, wrong sign; "
            f"the label-shuffle placebo can't separate it from noise (*p* = {R['placebo_p']:.2f}).\n"
            f"- **Tradability -- MIRAGE.** The gross spread is negative; net of costs "
            f"{R['ls_net']:+.2f}%/yr. There is nothing to trade.\n"
            "- **Premium-sign -- BUSTED.** E-P predict a positive premium; on this survivor basket "
            "the sign flips. The synthetic control proves it's the tape, not the engine."
        ),
        md(
            "## 6 -- Could you actually trade it?\n\n"
            "No. Even setting aside the wrong sign:\n\n"
            "1. **The spread is negative gross**, and costs (transaction + short borrow) make it "
            "worse.\n"
            "2. **Survivorship cuts the wrong way.** The risk story lives in firms that *might* lose "
            "their talent and fail -- but failed firms are exactly the ones deleted from a "
            "current-large-cap basket.\n"
            "3. **SG&A is noisy across sectors.** A retailer and a software firm spend on utterly "
            "different SG&A; the cross-sectional sort mixes structural business-model differences "
            "with any true org-capital signal."
        ),
        md(
            "## 7 -- Going further\n\n"
            "- **Broaden the universe.** The original paper used the full Compustat cross-section "
            "with delisted firms. A survivor-free, microcap-inclusive panel is the real test.\n"
            "- **Industry-neutralize.** Sort *within* industry so SG&A intensity reflects org-capital, "
            "not business model.\n"
            "- **[Study 238 -- Betting-Against-Beta](../../238-betting-against-beta/)** and "
            "**[Study 330 -- Low-Volatility-Anomaly](../../330-low-volatility-anomaly/)**: two other "
            "risk-premium claims where the headline survives the literature but not a small survivor tape.\n\n"
            "*Think org-capital pays once you fix survivorship and industry-neutralize? Fork this, "
            "add delisted names, and show t > 2 with the right sign net of costs. That is the bar.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


def build_quants():
    cells = [
        md(
            "# Organizational-Capital -- a quantitative teardown\n"
            "### EDGAR SG&A * perpetual inventory * annual sort * HAC inference\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Premium-sign: Busted](https://img.shields.io/badge/Premium--sign-Busted-8b949e?style=flat-square)\n\n"
            "The quantitative companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "We test Eisfeldt-Papanikolaou (2013): capitalize SG&A by perpetual inventory "
            "(delta=15%, steady-state seed), scale by assets, sort annually, long the high-OC half / "
            "short the low-OC half, one execution lag, HAC inference.\n\n"
            "> **Not investment advice.** Real data: EDGAR 10-K SG&A + assets (~18-19y) and yfinance "
            "daily adjusted-close, ~39 large-cap survivors, 2009-2025, as-of 2026-06-26. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship bias is named:** basket = current large-caps still trading in 2026."
        ),
        code(BOOT),
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | LS **{R['ls_gross']:+.2f}%/yr**, HAC *t* = "
            f"**{R['ls_t']:+.3f}** (wrong sign, \\|t\\|<2); placebo *p* = **{R['placebo_p']:.2f}**. |\n"
            f"| **Tradability** | `MIRAGE` | Net of costs **{R['ls_net']:+.2f}%/yr** "
            f"(t={R['ls_net_t']:+.2f}); Sharpe **{R['ls_sharpe']:+.3f}**, max DD **{R['ls_dd']:.1f}%**. |\n"
            "| **Premium-sign** | `BUSTED` | E-P predict high-OC > low-OC; here "
            f"high {R['high_leg']:.1f}%/yr **<** low {R['low_leg']:.1f}%/yr. |\n\n"
            "> The org-capital premium is well-documented on the broad, delisted-inclusive "
            "Compustat cross-section. On 17 years of ~39 large-cap survivors it is absent and "
            "wrong-signed -- consistent with survivorship truncating the high-risk long leg."
        ),
        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Build org-capital by perpetual inventory: "
            "$OC_t = (1-\\delta)\\,OC_{t-1} + SG\\&A_t$, $\\delta=0.15$, seeded at "
            "$OC_0 = SG\\&A_1/(g+\\delta)$. Scale: the sort variable is $OC/\\text{Assets}$. "
            "Eisfeldt-Papanikolaou (2013) assert:\n\n"
            "- **H1 (signal).** High-$OC/A$ firms earn higher average returns: "
            "$E[r^{\\text{high}} - r^{\\text{low}}] > 0$.\n"
            "- **H2 (risk).** That premium compensates talent-mobility risk -- it is *not* mispricing.\n\n"
            "We **reject H1** on this panel (the spread is negative and insignificant), which makes "
            "H2 moot here."
        ),
        md(
            "## 2 -- So what? -- the economic stakes\n\n"
            "Capitalizing intangibles (SG&A -> org-capital, R&D -> knowledge-capital) is now standard "
            "in asset pricing (Peters-Taylor 2017). If the org-capital *return* premium does not "
            "replicate out of the discovery universe, factor models that lean on it inherit the risk."
        ),
        md(
            "## 3 -- The protocol\n\n"
            "- **Fundamentals.** EDGAR 10-K annual SG&A (combined, or Selling&Marketing + G&A summed) "
            "and year-end total assets, ~18-19y per firm.\n"
            "- **Org-capital.** Perpetual inventory, delta=0.15, steady-state seed (g=0.10).\n"
            "- **Sort.** Annually split the basket at the median of $OC/\\text{Assets}$.\n"
            "- **Long-short.** Equal-weight high half minus low half; held 12 months; one execution lag.\n"
            "- **Inference.** Newey-West HAC *t* on monthly long-short returns.\n"
            "- **Placebo.** Shuffle $OC/\\text{Assets}$ labels within each year (null).\n"
            "- **Costs.** 5 bps/leg one-way at annual rebalance (100% turnover) + 50 bps/yr short borrow.\n"
            "- **Universe caveat.** ~39 large-cap survivors; survivorship-biased."
        ),
        md("## 4 -- The teardown"),
        md(
            "### 4a -- Positive control: the engine is a faithful detector (12-seed avg)\n\n"
            "Sweep the planted OC premium; the long-short HAC *t*, averaged over 12 seeds, should be "
            "monotone and correctly-signed."
        ),
        code(
            "ctrl = st.synthetic_control()\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if m > 0 else RED for m in ctrl['mean_t']]\n"
            "ax.bar(ctrl['premium'].astype(str), ctrl['mean_t'], color=col)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(2, c=GREY, ls='--', lw=1, label='|t|=2')\n"
            "ax.axhline(-2, c=GREY, ls='--', lw=1)\n"
            "ax.set_xlabel('planted OC premium'); ax.set_ylabel('mean HAC t (12 seeds)')\n"
            "ax.set_title('Positive control: HAC t monotone & correctly signed in the premium')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),
        md(
            "### 4b -- The org-capital cross-section on the real tape\n\n"
            "What does $OC/\\text{Assets}$ look like across the basket in the latest year?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    last_y = oc['fyear'].max()\n"
            "    snap = oc[oc['fyear'] == last_y].set_index('ticker')['oc_assets'].sort_values()\n"
            "    fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            "    colors = [GREY if v <= snap.median() else AMBER for v in snap.values]\n"
            "    ax.barh(snap.index, snap.values, color=colors)\n"
            "    ax.axvline(snap.median(), c=RED, ls='--', lw=1.5, label=f'median={snap.median():.2f}')\n"
            "    ax.set_xlabel('organizational capital / assets'); ax.set_title(f'OC/assets cross-section ({last_y})')\n"
            "    ax.legend(); ax.tick_params(axis='y', labelsize=7)\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'High-OC (amber) = long leg, low-OC (grey) = short leg. n={len(snap)}')\n"
            "else:\n"
            "    print('Cross-section requires the real cache.')"
        ),
        md("### 4c -- Real long-short: returns, legs, equity curve"),
        code(
            "if HAVE_REAL:\n"
            "    lsm, lst = s_ls['mean']*100, s_ls['tstat']\n"
            "    hi, lo = s_hi['mean']*100, s_lo['mean']*100\n"
            "    eq = (1 + ls['ls']).cumprod(); dd = (eq/eq.cummax()-1)*100\n"
            "else:\n"
            "    lsm, lst, hi, lo = R['ls_gross'], R['ls_t'], R['high_leg'], R['low_leg']\n"
            "    eq = pd.Series(dtype=float); dd = pd.Series(dtype=float)\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "axes[0].bar(['High-OC', 'Low-OC'], [hi, lo], color=[AMBER, GREY], width=.5)\n"
            "axes[0].set_ylabel('mean annual return (%/yr)')\n"
            "axes[0].set_title(f'Legs | LS {lsm:+.2f}%/yr  t={lst:+.2f}')\n"
            "if HAVE_REAL:\n"
            "    axes[1].plot(eq.index, eq.values, c=RED, lw=1.8)\n"
            "    axes[1].axhline(1, c='k', lw=1, ls='--')\n"
            "    axes[1].set_title('High-minus-low equity curve (drifts down)')\n"
            "    axes[1].set_ylabel('cumulative wealth')\n"
            "else:\n"
            "    axes[1].text(.5, .5, 'real cache needed', ha='center')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'LS {lsm:+.2f}%/yr | HAC t {lst:+.3f} | high {hi:.2f}% vs low {lo:.2f}%')"
        ),
        md(
            "### 4d -- Placebo: shuffle the org-capital labels within each year\n\n"
            "If the OC ranking carried real information, the true long-short would sit in the tail of "
            "the label-shuffled null. It does not."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pb = st.placebo_pvalue(oc, monthly, n_shuffles=400)\n"
            "    real_mo = pb['real_mean']*100\n"
            "    print(f\"placebo p = {pb['p_value']:.3f}  (real {real_mo:+.4f}%/mo vs \"\n"
            "          f\"null {pb['null_mean']*100:+.4f}% +/- {pb['null_std']*100:.4f}%, n={pb['n_shuffles']})\")\n"
            "    p = pb['p_value']\n"
            "else:\n"
            "    p = R['placebo_p']; real_mo = R['placebo_real']*100\n"
            "    print(f'Frozen placebo p = {p:.3f}')\n"
            "print('p >= 0.05 -> the real long-short is indistinguishable from a shuffled-label null.')"
        ),
        md("### 4e -- Robustness: does the depreciation rate change the verdict?"),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for d in (0.10, 0.15, 0.20, 0.30):\n"
            "        s_d = st.summary(st.long_short(st.org_capital_stock(fund, delta=d), monthly)['ls'])\n"
            "        rows.append({'delta': d, 'LS_%/yr': s_d['mean']*100, 'HAC_t': s_d['tstat']})\n"
            "    sweep = pd.DataFrame(rows)\n"
            "else:\n"
            "    sweep = pd.DataFrame([{'delta': d, 'LS_%/yr': v[0], 'HAC_t': v[1]}\n"
            "                          for d, v in R['delta_sweep'].items()])\n"
            "fig, ax = plt.subplots(figsize=(9, 4.0))\n"
            "ax.plot(sweep['delta'], sweep['LS_%/yr'], 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('org-capital depreciation rate (delta)'); ax.set_ylabel('LS (%/yr)')\n"
            "ax.set_title('Negative & insignificant across every plausible delta')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(sweep.round(3).to_string(index=False))"
        ),
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `NONE`** -- LS {R['ls_gross']:+.2f}%/yr, HAC *t* = {R['ls_t']:+.3f} "
            f"(wrong sign, \\|t\\|<2), placebo *p* = {R['placebo_p']:.2f}, negative across every delta. "
            "No literature upgrade: the published premium lives on a delisted-inclusive universe we "
            "cannot reproduce here, and the *sign itself* fails.\n"
            f"- **Tradability `MIRAGE`** -- net {R['ls_net']:+.2f}%/yr, Sharpe {R['ls_sharpe']:+.3f}, "
            f"max DD {R['ls_dd']:.1f}%. Nothing to trade.\n"
            "- **Premium-sign `BUSTED`** -- the synthetic control recovers the planted, correctly-"
            "signed premium, so the real-tape sign flip is the market (and the survivor truncation), "
            "not the engine."
        ),
        md(
            "## 6 -- Could you trade it?\n\n"
            "No. The spread is negative gross and net; the borrow on the short leg and the annual "
            "rebalance turnover only add cost. The deeper problem is structural: the risk story "
            "puts the premium on firms that *might fail by losing their talent* -- and those firms "
            "are exactly the ones a current-large-cap basket deletes."
        ),
        md(
            "## 7 -- Going further\n\n"
            "- **Delisted-inclusive Compustat panel** -- the only honest replication of the original.\n"
            "- **Within-industry sort** -- strip out the consumer-staples-vs-software SG&A gap.\n"
            "- **Value-weight & microcaps** -- where Hou-Xue-Zhang (2020) find most anomalies live or die.\n"
            "- **[Study 238 -- Betting-Against-Beta](../../238-betting-against-beta/)** / "
            "**[Study 330 -- Low-Volatility-Anomaly](../../330-low-volatility-anomaly/)**: same desk "
            "machinery on adjacent risk-premium claims.\n\n"
            "*Think org-capital pays once survivorship and industry are fixed? Fork this, add the "
            "delisted tail, and show t > 2 with the right sign net of costs. That is the bar.*"
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
