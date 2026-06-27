"""Generate the two narrative notebooks for Study 530 (Book-To-Market-Value).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats. Synthetic figures run anywhere, offline and
deterministic; the real-panel cells use the cached yfinance parquets under ../_cache/ if
present and otherwise quote the frozen headline numbers in ``R`` (which mirror docs/results.md
exactly, as-of 2026-06-27).
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


# Frozen real-panel headline numbers -- the ONE dict mirroring docs/results.md (as-of 2026-06-27).
R = dict(
    n_years=3, sig_years=[2022, 2023, 2024], n_tickers=40,
    hedge_mean=-5.62, hedge_vol=22.21, hedge_sharpe=-0.253, hedge_t=-0.554,
    hedge_hit=33.3, hedge_dd=-13.3,
    value_leg=26.23, growth_leg=31.85, market=23.02,
    placebo_p=0.583, placebo_null_mean=-0.37, placebo_null_std=10.15,
    turnover=15.6, cost_drag=0.03, borrow_drag=0.50,
    gross_mean=-5.62, net_mean=-6.15, net_t=-0.606,
    fp_bm="e42cf31942ef", fp_fwd="fe6de140bdb3",
    annual={2022: -23.0, 2023: -13.3, 2024: 19.4},  # signal year -> hedge%
    ctrl={-0.05: (-13.47, -14.83), 0.0: (-0.49, -0.54),
          0.05: (12.49, 13.64), 0.10: (25.48, 27.30)},
    ctrl_seed_t_mean=12.9, ctrl_seed_t_min=7.2,
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from book_to_market_value import data, strategy as st

def _have_cache():
    study = os.path.abspath("..")
    return (os.path.exists(os.path.join(study, "_cache", "bm_panel.parquet"))
            and os.path.exists(os.path.join(study, "_cache", "fwd_ret.parquet")))

HAVE_REAL = _have_cache()
print("yfinance cache present:", HAVE_REAL)

if HAVE_REAL:
    bm, fwd = data.fetch_panel()
    sort = st.quintile_sort(bm, fwd)
    s_hedge = st.summary(sort["hedge"])
    s_long  = st.summary(sort["long"])
    s_short = st.summary(sort["short"])
    s_mkt   = st.summary(sort["market"])
    print(f"N years: {s_hedge['n']} | hedge mean: {s_hedge['mean']*100:+.2f}%/yr"
          f" | HAC t: {s_hedge['tstat']:+.3f}")
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Book-To-Market-Value -- do cheap stocks really beat expensive ones?\n"
            "### Fama-French (1992): the canonical value premium, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Value-survived%3F: Busted](https://img.shields.io/badge/Value--survived%3F-Busted-8b949e?style=flat-square)\n\n"
            "In 1992 Eugene Fama and Kenneth French published the single most famous result in "
            "empirical finance: sort stocks by **book-to-market** -- the accountant's book value "
            "of a company divided by its stock-market price -- and the cheap stocks (high "
            "book-to-market, 'value') beat the expensive ones (low book-to-market, 'growth'). "
            "That spread became the **HML** factor, the 'V' in every value fund's pitch deck.\n\n"
            "We test that claim on a 40-name large-cap survivor basket using yfinance balance "
            "sheets and prices -- naming the survivorship bias and a brutal data limitation up "
            "front.\n\n"
            "> **This is the plain-language layer.** Want the quintile sort, HAC inference, "
            "label-shuffle placebo, and seed-robust synthetic control? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does value beat growth here? | **No -- the opposite.** The long-value / "
            f"short-growth hedge earns **{R['hedge_mean']:+.2f}%/yr**, the *wrong* sign, "
            f"HAC *t* = **{R['hedge_t']:+.2f}**. |\n"
            "| Is it statistically real? | **No.** A label-shuffle placebo gives "
            f"*p* = **{R['placebo_p']:.2f}** -- indistinguishable from a coin flip. |\n"
            "| Is it tradable? | **No.** Net of costs **"
            f"{R['net_mean']:+.2f}%/yr**, and the entire test rests on **{R['n_years']} "
            "annual observations**. |\n"
            "| Is the method broken? | **No.** The engine recovers a clean value premium "
            f"when one is planted (synthetic *t* > {R['ctrl_seed_t_min']:.0f} across seeds). "
            "The market just didn't pay value over 2023-2025. |\n\n"
            "> The value premium is real and famous in the long historical record. On a "
            "large-cap survivor basket over the AI / mega-cap-growth era -- the only window "
            "free fundamentals allow -- it shows up **negative and statistically zero**."
        ),

        md(
            "## 1 -- The claim\n\n"
            "> *\"Average returns on stocks are strongly related to book-to-market equity. "
            "High book-to-market (value) stocks have higher average returns than low "
            "book-to-market (growth) stocks, and this is not captured by market beta.\"*\n\n"
            "-- Fama & French (1992), *Journal of Finance*\n\n"
            "**Book-to-market** is just `book value of equity / market value of equity`. A high "
            "ratio means the market prices the company near (or below) its accounting net worth "
            "-- a cheap, unloved, often-distressed 'value' stock. A low ratio means the market "
            "pays a big premium over book -- an expensive, high-expectations 'growth' stock. "
            "Fama-French sort on exactly this and long the cheap, short the expensive."
        ),

        md(
            "## 2 -- So what?\n\n"
            "If true, value is a free lunch: a risk-adjusted return uncorrelated with simply "
            "owning the market. It underpins trillions of dollars of value-tilted funds, smart-"
            "beta ETFs, and the entire Fama-French three-factor model used to grade every "
            "active manager on Earth.\n\n"
            "The questions for us:\n"
            "1. **Does it show up on a realistic large-cap panel?**\n"
            "2. **Does it survive a placebo and real costs?**\n"
            "3. **What does the free-data limitation do to our power?**"
        ),

        md(
            "## 3 -- How would we even know?\n\n"
            "Three disciplines keep us honest:\n\n"
            "1. **One execution lag, no look-ahead.** The B/M signal for fiscal year Y uses "
            "the 10-K (public by ~spring Y+1); we hold the position for the *full* calendar "
            "year Y+1. The fundamentals are stale before we trade on them.\n"
            "2. **A placebo.** We shuffle the B/M labels within each year and recompute the "
            "hedge 2000 times. If the real hedge sits inside that noise band, there is no signal.\n"
            "3. **Name the limits.** The basket is survivors only (failed value traps deleted), "
            "and yfinance gives only ~4-5 annual balance sheets -- so the real test is brutally "
            "short. We say so loudly."
        ),

        md(
            "## 4 -- The teardown\n\n"
            "**First, does the value engine work when the premium is planted?**"
        ),
        code(
            "ctrl = st.synthetic_control(premiums=(-0.05, 0.0, 0.05, 0.10))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if m > 0.5 else (RED if m < -0.5 else GREY) for m in ctrl['hedge_mean_%']]\n"
            "ax.bar(ctrl['premium'].astype(str), ctrl['hedge_mean_%'], color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, (m, t) in enumerate(zip(ctrl['hedge_mean_%'], ctrl['tstat'])):\n"
            "    ax.text(i, m + (0.5 if m >= 0 else -1.5), f't={t:+.1f}', ha='center', fontsize=8)\n"
            "ax.set_xlabel('planted value (HML) premium'); ax.set_ylabel('recovered hedge (%/yr)')\n"
            "ax.set_title('Synthetic control: engine finds value when it is really there')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(2).to_string(index=False))"
        ),
        md(
            "The engine is faithful: positive premium in, positive hedge out (monotone); zero "
            "in, noise out. So whatever the real tape says reflects **the market**, not a "
            "broken detector."
        ),
        md("**Now the honest test on the real yfinance panel.**"),
        code(
            "if HAVE_REAL:\n"
            "    val, gro, mkt = s_long['mean']*100, s_short['mean']*100, s_mkt['mean']*100\n"
            "    hedge_m, hedge_t = s_hedge['mean']*100, s_hedge['tstat']\n"
            "    annual = (sort['hedge']*100)\n"
            "    annual.index = [int(y) for y in sort.index]\n"
            "else:\n"
            "    val, gro, mkt = R['value_leg'], R['growth_leg'], R['market']\n"
            "    hedge_m, hedge_t = R['hedge_mean'], R['hedge_t']\n"
            "    annual = pd.Series(R['annual'])\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "labels = ['Value Q5\\n(long)', 'Universe', 'Growth Q1\\n(short)']\n"
            "axes[0].bar(labels, [val, mkt, gro], color=[AMBER, GREY, RED], width=0.5)\n"
            "axes[0].set_ylabel('mean annual return (%/yr)')\n"
            "axes[0].set_title('Growth (Q1) out-earned value (Q5) over the window')\n"
            "col_yr = [GREEN if v > 0 else RED for v in annual.values]\n"
            "axes[1].bar(annual.index.astype(str), annual.values, color=col_yr)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_xlabel('signal year'); axes[1].set_ylabel('hedge (Q5-Q1) %/yr')\n"
            "axes[1].set_title(f'Value hedge | mean {hedge_m:+.1f}% t={hedge_t:+.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Value hedge: {hedge_m:+.2f}%/yr | HAC t = {hedge_t:+.3f}')"
        ),
        md(
            f"The value hedge earns **{R['hedge_mean']:+.1f}%/yr** at HAC *t* = "
            f"**{R['hedge_t']:+.2f}** -- the *wrong sign* and far inside noise. Two of three "
            "years (2022 and 2023 signal) were sharply negative as mega-cap growth ran; only "
            "the 2024 signal year paid value. This is the textbook 'value is dead' regime."
        ),

        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- NONE.** Hedge {R['hedge_mean']:+.2f}%/yr, HAC *t* = "
            f"{R['hedge_t']:+.2f}, placebo *p* = {R['placebo_p']:.2f}. Insignificant *and* the "
            "wrong sign -- literature cannot upgrade that.\n"
            f"- **Tradability -- MIRAGE.** Net {R['net_mean']:+.2f}%/yr on "
            f"{R['n_years']} annual observations. Nothing to trade.\n"
            "- **Did value survive? -- BUSTED** on this modern survivor tape, even though the "
            "engine demonstrably works."
        ),

        md(
            "## 6 -- Could you actually trade it?\n\n"
            "Even setting aside the negative sign, the obstacles are decisive:\n\n"
            "1. **The data is too short.** Three annual cross-sections is not a backtest; it is "
            "an anecdote. yfinance simply does not expose enough fundamental history.\n"
            "2. **Survivorship flatters the long leg.** Deep-value names that went bankrupt -- "
            "the natural value longs -- are absent, so even our negative result is an "
            "*upper bound* on what was achievable.\n"
            "3. **Regime risk.** Value can underperform for a decade (2010-2020). Sizing a "
            "long-short value book through that requires conviction most investors lack.\n\n"
            "On this evidence the practical verdict is **Mirage**."
        ),

        md(
            "## 7 -- Going further\n\n"
            "- **A real fundamental database.** Compustat/CRSP back to the 1920s -- with "
            "delisted firms -- is where the value premium actually lives. The decisive limit "
            "here is the free data source, not the method.\n"
            "- **Other value lenses.** [Study 124 -- Cash-Flow-Yield](../../124-cash-flow-yield/) "
            "(cash-based value), [Study 121 -- Magic-Formula](../../121-magic-formula/) "
            "(value + quality), [Study 243 -- Graham-NCAV](../../243-graham-ncav/) (deep "
            "liquidation value).\n"
            "- **The same machinery on risk.** [Study 238 -- Betting-Against-Beta](../../238-betting-against-beta/) "
            "uses the identical annual sort + HAC inference on beta instead of B/M.\n\n"
            "*Think value clears t > 2 net of costs on a survivorship-free, multi-decade "
            "panel? Fork this, swap in Compustat with delistings, and show it. That is the bar.*"
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
            "# Book-To-Market-Value -- a quantitative teardown\n"
            "### yfinance panel * B/M quintile sort * HAC inference * label-shuffle placebo\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Value-survived%3F: Busted](https://img.shields.io/badge/Value--survived%3F-Busted-8b949e?style=flat-square)\n\n"
            "The quantitative companion to the "
            "[notebook for the curious](01_for_the_curious.ipynb) -- same seven beats, every "
            "claim carrying its standard error. We test Fama-French (1992): rank by "
            "book-to-market, long the high-B/M value quintile (Q5), short the low-B/M growth "
            "quintile (Q1), one-year reporting lag.\n\n"
            "> **Not investment advice.** Real data: yfinance balance sheets + adjusted close, "
            "40 S&P 500 large-caps, as-of 2026-06-27. Methods in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship bias is named:** basket = current large-caps projected backwards. "
            "**Data limitation:** yfinance exposes only ~4-5 annual balance sheets, so the real "
            "panel is just 3 cross-sectional years -- HAC inference is low-powered by construction."
        ),
        code(BOOT),

        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | hedge **{R['hedge_mean']:+.2f}%/yr** (wrong sign), "
            f"HAC *t* = **{R['hedge_t']:+.3f}**, placebo *p* = **{R['placebo_p']:.3f}**. |\n"
            f"| **Tradability** | `MIRAGE` | net **{R['net_mean']:+.2f}%/yr**, Sharpe "
            f"**{R['hedge_sharpe']:+.3f}**, **{R['n_years']}** annual obs. |\n"
            "| **Value survived?** | `BUSTED` | negative over the AI/mega-cap-growth window; "
            f"synthetic control is faithful (seed-robust *t* > {R['ctrl_seed_t_min']:.0f}). |\n\n"
            "> The most famous factor in finance, replicated honestly on the only window free "
            "fundamentals allow, lands a textbook None x Mirage -- and the regime Busts the "
            "third axis."
        ),

        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $BM_{i,Y}$ = book equity / market cap of stock $i$ at fiscal-year-end $Y$. "
            "Fama-French (1992) assert:\n\n"
            "- **H1 (signal).** The cross-sectional rank of $BM$ predicts next-year return: "
            "$\\mathbb{E}[r_{i,Y+1} \\mid BM_{i,Y}]$ is increasing in $BM$. The long-short "
            "$\\text{hedge}_Y = \\bar r^{Q5}_{Y+1} - \\bar r^{Q1}_{Y+1} > 0$.\n"
            "- **H2 (not beta).** The premium is not explained by market beta -- it is a "
            "distinct factor (HML).\n"
            "- **H3 (tradable).** It survives costs and turnover.\n\n"
            "On this tape we **reject H1** (hedge is negative, *t* = -0.55, *p* = 0.58), so "
            "H2/H3 are moot -- there is no premium to attribute or trade."
        ),

        md(
            "## 2 -- So what? -- the economic stakes\n\n"
            "Value is the cornerstone of factor investing. The Fama-French three-factor model "
            "is the default benchmark for grading active managers; trillions sit in value-"
            "tilted smart-beta. If the premium is absent in large-caps over the recent regime, "
            "those products were -- over this window -- selling a story the tape did not honour."
        ),

        md(
            "## 3 -- The protocol\n\n"
            "- **Signal.** B/M = `Stockholders Equity` / (`Ordinary Shares Number` × "
            "fiscal-year-end adjusted close), winsorised to (0, 5).\n"
            "- **Ranking.** Annually, cut the 40 names into B/M quintiles. Q5 = value (long), "
            "Q1 = growth (short).\n"
            "- **Hedge.** Equal-weight Q5 forward return minus equal-weight Q1 forward return.\n"
            "- **Execution lag.** Signal year Y -> held over the full calendar year Y+1 "
            "(after the 10-K is public). Exactly one lag, no same-bar fill.\n"
            "- **Inference.** Newey-West HAC *t* on the annual hedge series.\n"
            "- **Placebo.** Shuffle B/M labels within each year, 2000 draws; two-sided *p*.\n"
            "- **Costs.** 10 bps one-way × turnover + 50 bps/yr borrow on the short leg.\n"
            "- **Universe caveat.** 40 large-cap survivors; only 3 usable years (yfinance limit)."
        ),

        md("## 4 -- The teardown"),
        md(
            "### 4a -- Positive control: the B/M engine is a faithful detector\n\n"
            "Sweep the planted HML premium from negative to positive on a synthetic panel. The "
            "recovered hedge should be monotone in the premium."
        ),
        code(
            "premiums = (-0.08, -0.04, 0.0, 0.04, 0.08, 0.12)\n"
            "ctrl = st.synthetic_control(premiums=premiums)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if m > 0 else RED for m in ctrl['hedge_mean_%']]\n"
            "ax.bar(ctrl['premium'].astype(str), ctrl['hedge_mean_%'], color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, (m, t) in enumerate(zip(ctrl['hedge_mean_%'], ctrl['tstat'])):\n"
            "    ax.text(i, m + (0.5 if m >= 0 else -2.0), f't={t:+.0f}', ha='center', fontsize=8)\n"
            "ax.set_xlabel('planted HML premium'); ax.set_ylabel('recovered hedge (%/yr)')\n"
            "ax.set_title('Positive control: hedge is monotone in the planted premium')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(2).to_string(index=False))"
        ),
        md(
            "### 4b -- Seed robustness of the control\n\n"
            "A single lucky RNG seed proves nothing. Average the synthetic *t* over 20 seeds "
            "at a planted +5% premium (and check the null)."
        ),
        code(
            "ts_planted, ts_null = [], []\n"
            "for seed in range(20):\n"
            "    bm_s, fwd_s, _ = data.synthetic_panel(hml_premium=0.05, seed=seed)\n"
            "    ts_planted.append(st.summary(st.quintile_sort(bm_s, fwd_s, min_firms=20)['hedge'])['tstat'])\n"
            "    bm0, fwd0, _ = data.synthetic_panel(hml_premium=0.0, seed=seed)\n"
            "    ts_null.append(st.summary(st.quintile_sort(bm0, fwd0, min_firms=20)['hedge'])['tstat'])\n"
            "ts_planted, ts_null = np.array(ts_planted), np.array(ts_null)\n"
            "print(f'Planted +5%: mean t = {ts_planted.mean():+.2f}, min t = {ts_planted.min():+.2f}, '\n"
            "      f'all>2: {bool((ts_planted>2).all())}')\n"
            "print(f'Null  0%:    mean t = {ts_null.mean():+.2f}, frac |t|>2 = {(np.abs(ts_null)>2).mean():.2f}')\n"
            "fig, ax = plt.subplots(figsize=(9.0, 3.8))\n"
            "ax.hist(ts_planted, bins=10, color=GREEN, alpha=.7, label='planted +5%')\n"
            "ax.hist(ts_null, bins=10, color=GREY, alpha=.7, label='null 0%')\n"
            "ax.axvline(2, c=RED, ls='--', lw=1.5, label='|t|=2 bar')\n"
            "ax.set_xlabel('synthetic HAC t'); ax.set_ylabel('count'); ax.legend()\n"
            "ax.set_title('Engine clears the bar only when value is really planted')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "### 4c -- Real panel: the quintile sort and the hedge"
        ),
        code(
            "if HAVE_REAL:\n"
            "    hedge_m, hedge_t = s_hedge['mean']*100, s_hedge['tstat']\n"
            "    qcols = [c for c in sort.columns if c.startswith('Q')]\n"
            "    qmeans = sort[qcols].mean()*100\n"
            "    annual = (sort['hedge']*100); annual.index = [int(y) for y in sort.index]\n"
            "else:\n"
            "    hedge_m, hedge_t = R['hedge_mean'], R['hedge_t']\n"
            "    qmeans = pd.Series({'Q1': R['growth_leg'], 'Q5': R['value_leg']})\n"
            "    annual = pd.Series(R['annual'])\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "axes[0].bar(qmeans.index, qmeans.values, color=[RED]+[GREY]*(len(qmeans)-2)+[AMBER] if len(qmeans)>2 else [RED, AMBER])\n"
            "axes[0].set_ylabel('mean fwd return (%/yr)'); axes[0].set_xlabel('B/M quintile (Q1 growth .. Q5 value)')\n"
            "axes[0].set_title('Quintile means: no monotone value gradient')\n"
            "col_yr = [GREEN if v > 0 else RED for v in annual.values]\n"
            "axes[1].bar(annual.index.astype(str), annual.values, color=col_yr)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_xlabel('signal year'); axes[1].set_ylabel('hedge (Q5-Q1) %/yr')\n"
            "axes[1].set_title(f'Value hedge | mean {hedge_m:+.1f}% t={hedge_t:+.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Hedge: {hedge_m:+.2f}%/yr | HAC t = {hedge_t:+.3f}')"
        ),
        md(
            "### 4d -- Label-shuffle placebo\n\n"
            "Is the (negative) hedge distinguishable from noise? Permute the B/M labels within "
            "each year 2000 times and rebuild the null distribution of the mean hedge."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pb = st.placebo_pvalue(bm, fwd, n_shuffles=2000)\n"
            "    real_m, null_m, null_s, pval = (pb['real_mean']*100, pb['null_mean']*100,\n"
            "                                    pb['null_std']*100, pb['p_value'])\n"
            "else:\n"
            "    real_m, null_m, null_s, pval = (R['hedge_mean'], R['placebo_null_mean'],\n"
            "                                    R['placebo_null_std'], R['placebo_p'])\n"
            "x = np.linspace(null_m-3*null_s, null_m+3*null_s, 200)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.0))\n"
            "from math import pi\n"
            "ax.plot(x, np.exp(-0.5*((x-null_m)/null_s)**2)/(null_s*(2*pi)**.5), c=GREY, lw=2, label='shuffle null')\n"
            "ax.axvline(real_m, c=RED, lw=2, label=f'real hedge {real_m:+.1f}%')\n"
            "ax.axvline(0, c='k', lw=0.8, ls=':')\n"
            "ax.set_xlabel('mean hedge (%/yr)'); ax.set_ylabel('density'); ax.legend()\n"
            "ax.set_title(f'Real hedge sits inside the noise (placebo p = {pval:.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Real {real_m:+.2f}% | null {null_m:+.2f}% (std {null_s:.2f}%) | p = {pval:.3f}')"
        ),
        md(
            "### 4e -- Costs: gross vs net"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cs = st.apply_costs(sort, bm)\n"
            "    turn, cd, bd = cs['avg_turnover']*100, cs['cost_drag']*100, cs['borrow_drag']*100\n"
            "    gm, nm = cs['gross_mean']*100, cs['net_mean']*100\n"
            "else:\n"
            "    turn, cd, bd = R['turnover'], R['cost_drag'], R['borrow_drag']\n"
            "    gm, nm = R['gross_mean'], R['net_mean']\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.0))\n"
            "ax.bar(['gross', 'net of costs'], [gm, nm], color=[GREY, RED], width=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('hedge mean (%/yr)')\n"
            "ax.set_title(f'Costs push an already-negative hedge lower (turnover {turn:.0f}%/yr)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Turnover {turn:.1f}%/yr | cost {cd:.2f}% | borrow {bd:.2f}% | '\n"
            "      f'gross {gm:+.2f}% -> net {nm:+.2f}%')"
        ),

        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `NONE`** -- hedge {R['hedge_mean']:+.2f}%/yr, HAC *t* = "
            f"{R['hedge_t']:+.3f}, placebo *p* = {R['placebo_p']:.3f}. Insignificant and the "
            "wrong sign; the Fama-French prior cannot rescue a negative real-tape result.\n"
            f"- **Tradability `MIRAGE`** -- net {R['net_mean']:+.2f}%/yr, Sharpe "
            f"{R['hedge_sharpe']:+.3f}, on {R['n_years']} annual observations. The thin "
            "fundamental history is the binding constraint.\n"
            "- **Value survived? `BUSTED`** -- negative over the modern survivor window, even "
            "though the synthetic control is faithful and seed-robust."
        ),

        md(
            "## 6 -- Could you trade it?\n\n"
            "No -- and not only because of the sign:\n\n"
            "1. **Three annual cross-sections** is an anecdote, not a backtest. yfinance simply "
            "does not expose enough fundamental history for credible inference.\n"
            "2. **Survivorship flatters the long leg**: bankrupt deep-value names (the natural "
            "value longs) are deleted, so even the negative result is an upper bound.\n"
            "3. **Multi-year regime risk**: value underperformed for most of 2010-2020; the "
            "drawdown-tolerance required is extreme."
        ),

        md(
            "## 7 -- Going further\n\n"
            "- **A real fundamentals database.** Compustat/CRSP with delistings, back to the "
            "1920s, is where the premium actually lives -- and where the survivorship and "
            "sample-length limits here are fixed.\n"
            "- **B/M sub-variants.** Operating-leverage-adjusted B/M, intangible-adjusted book "
            "value (a live debate on why value 'broke' post-2010).\n"
            "- **Sibling value lenses on this desk:** "
            "[124 Cash-Flow-Yield](../../124-cash-flow-yield/), "
            "[121 Magic-Formula](../../121-magic-formula/), "
            "[243 Graham-NCAV](../../243-graham-ncav/).\n"
            "- **Same machinery, different signal:** "
            "[238 Betting-Against-Beta](../../238-betting-against-beta/).\n\n"
            "*Think value clears t > 2 net of costs on a survivorship-free, multi-decade panel? "
            "Fork this, swap in Compustat with delistings, and show it. That is the bar.*"
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
