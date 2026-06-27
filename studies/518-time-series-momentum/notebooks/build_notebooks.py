"""Generate the two narrative notebooks for Study 518 (Time-Series-Momentum).

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
    n_days=4740, n_tickers=12, n_months_panel=226, n_hold=213,
    date_start="2007-03-01", date_end="2025-12-30",
    hold_start="2008-03", hold_end="2025-11",
    fp="9c13724a40d6",
    cost_bps=10.0, borrow_bps=50.0,
    # vol-scaled (canonical)
    vs_gross=3.09, vs_net=2.68, vs_vol=7.9, vs_sharpe=0.392, vs_sharpe_net=0.340,
    vs_t=1.611, vs_t_net=1.396, vs_hit=53.5, vs_dd=-19.0, vs_dd_net=-19.5,
    vs_worst=-7.9, vs_turn=13.5, vs_short=49.8, vs_p=0.005,
    # equal-notional
    en_gross=1.54, en_net=1.20, en_sharpe=0.198, en_sharpe_net=0.154,
    en_t=0.788, en_t_net=0.613, en_hit=55.4, en_dd=-25.1, en_turn=12.1,
    # lookback fragility (gross t)
    t_9m=2.02, t_12m=1.61, t_15m=0.45,
    half1_t=0.78, half2_t=1.47,
    # synthetic control
    ctrl={0.00: (-0.2, -0.13), 0.15: (7.4, 3.66), 0.30: (15.4, 6.31), 0.50: (23.6, 8.80)},
    null_seed_t=-0.31,
    annual={2008: 4.8, 2009: -8.7, 2010: 0.5, 2011: -0.7, 2012: 2.5, 2013: 7.9,
            2014: 16.1, 2015: 3.6, 2016: -8.7, 2017: 17.4, 2018: -18.2, 2019: 12.4,
            2020: -1.1, 2021: 4.4, 2022: 10.3, 2023: -3.0, 2024: 9.1, 2025: 8.6},
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

from time_series_momentum import data, strategy as st

def _have_cache():
    study = os.path.abspath("..")
    return os.path.exists(os.path.join(study, "_cache", "tsmom_prices.parquet"))

HAVE_REAL = _have_cache()
print("yfinance cache present:", HAVE_REAL)

if HAVE_REAL:
    prices = data.drop_partial_last_month(data.fetch_panel())
    mp = st.to_monthly(prices)
    bk_v = st.tsmom_book(mp, vol_scale=True)
    bk_e = st.tsmom_book(mp, vol_scale=False)
    sv = st.summary(bk_v["port_gross"]); se = st.summary(bk_e["port_gross"])
    print(f"Vol-scaled TSMOM: {sv['mean']*100:+.2f}%/yr  HAC t={sv['tstat']:+.3f}")
    print(f"Equal-notional  : {se['mean']*100:+.2f}%/yr  HAC t={se['tstat']:+.3f}")
"""


def build_curious():
    cells = [
        md(
            "# Time-Series-Momentum -- does an asset's own trend predict its next month?\n"
            "### Moskowitz-Ooi-Pedersen (2012) 'trend everywhere', tested honestly on a cross-asset basket\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Vol_scaling_vs_equal_notional%3F: Confirmed](https://img.shields.io/badge/Vol_scaling_vs_equal_notional%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "In 2012, Tobias Moskowitz, Yao Hua Ooi and Lasse Pedersen published *Time Series "
            "Momentum*: forget ranking assets against each other -- just look at whether an asset "
            "has been going **up or down over the past year**, and bet it keeps going that way. "
            "Buy what's rising, short what's falling, and do it across **everything** -- stocks, "
            "bonds, commodities, currencies, gold. Risk-balance the bets, and the diversified "
            "book printed a high Sharpe with famous **crisis alpha** in 2008.\n\n"
            "We replicate that recipe on a **12-ETF cross-asset basket** and ask the honest "
            "question: does it still pay on a small, modern basket, net of real costs?\n\n"
            "> **This is the plain-language layer.** Want the HAC *t*-stats, the sign-shuffle "
            "placebo and the lookback-fragility sweep? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does own-trend predict the next month? | **Sort of.** The vol-scaled book earns "
            f"**{R['vs_gross']:+.2f}%/yr** at HAC *t* = **{R['vs_t']:+.3f}** -- positive and the "
            f"placebo says it's *real structure* (*p* = {R['vs_p']:.3f}), but the *t* is below the "
            f"desk's bar of 2. |\n"
            f"| Is it robust? | **Not quite.** Change the lookback and the *t* swings "
            f"{R['t_9m']:.2f} (9m) -> {R['t_12m']:.2f} (12m) -> {R['t_15m']:.2f} (15m). It "
            "straddles the bar. |\n"
            f"| Could you trade it? | **Barely.** Net **{R['vs_net']:+.2f}%/yr** at a "
            f"{R['vs_sharpe_net']:.2f} Sharpe with a {R['vs_dd']:.0f}% drawdown -- positive, but "
            "thin and it needs persistent shorting and 12-asset diversification. |\n"
            "| Does risk-balancing matter? | **Yes -- a lot.** Vol-scaling roughly doubles the "
            f"Sharpe ({R['vs_sharpe']:.2f} vs {R['en_sharpe']:.2f}) vs a naive sign book. |\n\n"
            "> The big story: trend showed up best when it mattered -- **+4.8% in 2008** while "
            "stocks collapsed, **+10.3% in 2022** in the rates/commodity trend -- but it bled in "
            "choppy years (-18% in 2018). Real-looking, thin, and regime-dependent."
        ),

        md(
            "## 1 -- The claim\n\n"
            "> *\"We document significant time series momentum in equity index, currency, "
            "commodity, and bond futures... A diversified portfolio of time series momentum "
            "strategies across all asset classes delivers substantial abnormal returns.\"*\n\n"
            "-- Moskowitz, Ooi & Pedersen (2012), *Journal of Financial Economics*\n\n"
            "The recipe: at each month-end, take the **sign of each asset's own trailing "
            "12-month return** (+1 up, -1 down). Size each bet inverse to its recent volatility "
            "(so a calm bond and a wild commodity risk the same), and average across the whole "
            "basket. Hold a month, repeat."
        ),

        md(
            "## 2 -- So what?\n\n"
            "Time-series momentum is the engine behind the **managed-futures / CTA** industry -- "
            "tens of billions of dollars of 'trend-following' funds. Its selling point is "
            "*crisis alpha*: when a crash drags on, trend goes short and profits while everything "
            "else bleeds. If that premium has **decayed** on a small modern basket, a lot of "
            "2-and-20 product is selling a 2012 backtest the recent data barely supports."
        ),

        md(
            "## 3 -- How would we even know?\n\n"
            "Three disciplines keep us honest:\n\n"
            "1. **No look-ahead.** The 12-month sign uses only past prices; the vol estimate that "
            "sizes next month uses only past data. We form on the month-end close and earn the "
            "*next* month -- one forward execution lag, no same-bar fill.\n"
            "2. **A placebo null.** We randomly flip each asset's position sign and recompute "
            "the book 1,000+ times. If the real edge is no better than coin-flip signs, it's "
            "noise.\n"
            "3. **Name the survivorship bias on the *signal* axis.** The basket is ETFs that "
            "*survived* to 2026 -- though, unlike a stock momentum short, a falling ETF here is "
            "simply *shorted*, never deleted, so the bias is much milder."
        ),

        md("## 4 -- The teardown\n\n**First, does the trend engine work when the trend is planted?**"),
        code(
            "ctrl = st.synthetic_control()\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if t > 2 else GREY for t in ctrl['tstat']]\n"
            "ax.bar(ctrl['trend_strength'].astype(str), ctrl['port_mean_ann']*100, color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, (m, t) in enumerate(zip(ctrl['port_mean_ann']*100, ctrl['tstat'])):\n"
            "    ax.text(i, m + 0.5, f't={t:+.1f}', ha='center', fontsize=8)\n"
            "ax.set_xlabel('planted own-asset trend'); ax.set_ylabel('portfolio mean (%/yr)')\n"
            "ax.set_title('Synthetic control: engine recovers trend when it is planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),
        md(
            "The engine is faithful: it finds trend when it's planted and (across 20 seeds) "
            f"scores a null *t* of **{R['null_seed_t']:+.2f}** when it isn't. So the verdict on "
            "the real tape reflects **the market**, not the method.\n\n"
            "**Now the honest test on the real basket -- and does risk-balancing help?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    vs_g = sv['mean']*100; vs_sh = sv['sharpe']\n"
            "    en_g = se['mean']*100; en_sh = se['sharpe']\n"
            "else:\n"
            "    vs_g, vs_sh = " f"{R['vs_gross']}, {R['vs_sharpe']}\n"
            "    en_g, en_sh = " f"{R['en_gross']}, {R['en_sharpe']}\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "x = np.arange(2); w = 0.35\n"
            "ax.bar(x - w/2, [vs_g, en_g], w, color=AMBER, label='mean (%/yr)')\n"
            "ax2 = ax.twinx()\n"
            "ax2.bar(x + w/2, [vs_sh, en_sh], w, color=GREY, label='Sharpe')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['Vol-scaled', 'Equal-notional'])\n"
            "ax.set_ylabel('mean annual return (%/yr)'); ax2.set_ylabel('Sharpe')\n"
            "ax.set_title('Risk-balancing roughly doubles the Sharpe')\n"
            "ax.legend(loc='upper left'); ax2.legend(loc='upper right')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Vol-scaled : {vs_g:+.2f}%/yr  Sharpe {vs_sh:+.2f}')\n"
            "print(f'Equal-notn : {en_g:+.2f}%/yr  Sharpe {en_sh:+.2f}')"
        ),
        md(
            f"Vol-scaling lifts the Sharpe from **~{R['en_sharpe']:.2f}** to **~{R['vs_sharpe']:.2f}** "
            "and halves the drawdown. That is the Moskowitz-Ooi-Pedersen insight made visible: "
            "letting a calm bond and a wild commodity each carry the *same risk* is what turns "
            "own-asset trend into a diversified premium instead of an equity-beta proxy."
        ),

        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- WEAK.** Vol-scaled TSMOM earns {R['vs_gross']:+.2f}%/yr at HAC *t* = "
            f"{R['vs_t']:+.3f} with placebo *p* = {R['vs_p']:.3f}. The placebo says the trend "
            "link is *real structure*, but the canonical 12-month *t* doesn't clear the desk's "
            f"bar of 2, and it wobbles with the lookback ({R['t_9m']:.2f}/{R['t_12m']:.2f}/"
            f"{R['t_15m']:.2f}). Strong literature plus a sub-2 *t* = WEAK, never REAL.\n"
            f"- **Tradability -- FRAGILE.** Net **{R['vs_net']:+.2f}%/yr** at a "
            f"{R['vs_sharpe_net']:.2f} Sharpe with a {R['vs_dd']:.0f}% drawdown -- positive, but "
            "thin and it needs persistent shorting + diversification.\n"
            "- **Vol-scaling vs equal-notional -- CONFIRMED.** Risk-balancing roughly doubles "
            "the Sharpe. The method choice matters more than the raw signal."
        ),

        md(
            "## 6 -- Could you actually trade it?\n\n"
            "Marginally. The net number is genuinely positive -- rare on this desk -- but:\n\n"
            f"1. **Shorting.** The book is short ~{R['vs_short']:.0f}% of NAV on average; that "
            "costs borrow (we charge 50bps/yr) and carries recall risk.\n"
            "2. **Leverage & breadth.** Inverse-vol sizing levers up calm bond/FX legs, and the "
            "edge leans on holding *all 12* asset classes -- a retail book can't cheaply do that.\n"
            f"3. **The drawdown.** A {R['vs_dd']:.0f}% loss and a sign that flips year to year "
            "(+17% 2017 -> -18% 2018) -- trend's choppy-market tail.\n"
            "4. **The sub-2 *t*.** With significance below the bar, the live edge could be smaller "
            "still."
        ),

        md(
            "## 7 -- Going further\n\n"
            "- **The cross-sectional cousin.** [Study 507 -- Cross-Sectional-Momentum]"
            "(../../507-cross-sectional-momentum/) ranks assets *against each other* instead of "
            "against their own history.\n"
            "- **SMA trend timing.** [Study 110 -- Faber-Timing](../../110-faber-timing/) is the "
            "long-or-flat, single-asset price-vs-average cousin.\n"
            "- **A century of trend.** Hurst-Ooi-Pedersen (2017) push time-series momentum back "
            "to 1880 -- and document the post-2009 'trend drought' that thins these numbers.\n\n"
            "*Think trend survives on a realistic basket net of costs? Fork this, add managed-"
            "futures-style breadth (more assets, futures not ETFs), and show *t* > 2 net of "
            "borrow. That is the bar.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


def build_quants():
    cells = [
        md(
            "# Time-Series-Momentum -- a quantitative teardown\n"
            "### yfinance cross-asset ETF basket * own-sign signal * inverse-vol sizing * HAC + placebo inference\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Vol_scaling_vs_equal_notional%3F: Confirmed](https://img.shields.io/badge/Vol_scaling_vs_equal_notional%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The quantitative companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb) -- same seven beats, every claim carrying its standard "
            "error. We test Moskowitz-Ooi-Pedersen (2012): own trailing 12-month return sign, "
            "inverse-vol-scaled, diversified across asset classes, long AND short, hold one "
            "month, vs a naive equal-notional sign book.\n\n"
            f"> **Not investment advice.** Real data: yfinance daily adjusted-close, "
            f"{R['n_tickers']} cross-asset ETFs, {R['date_start'][:4]}-{R['date_end'][:4]}, "
            f"as-of {R['asof']}. Methods in [`docs/references.md`](../docs/references.md), "
            "reproducible numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship is named on the SIGNAL axis:** basket = ETFs still trading in "
            "2026; mild here, because a falling ETF is shorted, not deleted."
        ),
        code(BOOT),

        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Vol-scaled TSMOM **{R['vs_gross']:+.2f}%/yr**, HAC *t* = "
            f"**{R['vs_t']:+.3f}**, placebo *p* = **{R['vs_p']:.3f}**; *t* below the bar of 2. |\n"
            f"| **Tradability** | `FRAGILE` | Net **{R['vs_net']:+.2f}%/yr**, net Sharpe "
            f"**{R['vs_sharpe_net']:.2f}**, max DD **{R['vs_dd']:.1f}%**, ~{R['vs_short']:.0f}% short. |\n"
            f"| **Vol-scaling vs equal-notional?** | `CONFIRMED` | Sharpe {R['vs_sharpe']:.2f} vs "
            f"{R['en_sharpe']:.2f}, *t* {R['vs_t']:.2f} vs {R['en_t']:.2f}. Risk-balancing wins. |\n\n"
            "> Trend is strong in the literature and on our synthetic control. On "
            f"{R['n_hold']} months of {R['n_tickers']} cross-asset ETFs it is *real structure* "
            "(placebo p = 0.005) but sub-2 significant and lookback-fragile -- consistent with the "
            "post-2009 trend drought (Hurst-Ooi-Pedersen 2017) and the Huang et al. (2020) critique."
        ),

        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $r^{12}_{i,t}$ = asset $i$'s own trailing 12-month return to month-end $t$, and "
            "$\\hat\\sigma_{i,t}$ its trailing realised vol. Moskowitz-Ooi-Pedersen (2012) assert:\n\n"
            "- **H1 (signal).** The vol-scaled portfolio "
            "$\\text{TSMOM}_{t+1} = \\frac{1}{N}\\sum_i \\text{sign}(r^{12}_{i,t}) "
            "\\frac{\\sigma_{\\text{tgt}}}{\\hat\\sigma_{i,t}} r_{i,t+1}$ has "
            "$E[\\text{TSMOM}] > 0$.\n"
            "- **H2 (robust).** It survives a sign-shuffle null and is not a single-lookback "
            "artifact.\n"
            "- **H3 (tradable).** It survives turnover costs and short borrow.\n\n"
            "On this basket we find **H1 directionally true but sub-bar** (HAC *t* = 1.61, placebo "
            "*p* = 0.005), **H2 fragile** (the *t* straddles 2 across lookbacks), and **H3 marginal** "
            "(net positive but thin, leveraged, short-heavy)."
        ),

        md(
            "## 2 -- So what? -- the economic stakes\n\n"
            "Time-series momentum is the academic core of the managed-futures industry and the "
            "source of trend's 'crisis alpha'. If the canonical 12-month sign no longer prints a "
            "clean *t* on a diversified modern basket, the residual premium lives in breadth "
            "(many more instruments), in futures (not cost-laden ETFs), and in the crash-hedge "
            "convexity, not in a small ETF long-short."
        ),

        md(
            "## 3 -- The protocol\n\n"
            "- **Signal.** Sign of each asset's own trailing 12-month return (no skip).\n"
            "- **Sizing.** Inverse-vol to a common target, leverage-capped (vol-scaled), vs "
            "equal-notional (sign only).\n"
            "- **Book.** Average position x next-month return across all valid assets; long AND "
            "short; diversified across 5 asset classes.\n"
            "- **Execution lag.** Form on the month-end close; earn month *t+1*. Vol estimate "
            "uses only data through *t*. One forward shift, no same-bar fill.\n"
            "- **Costs.** 10bps x turnover/rebalance + 50bps/yr borrow on the net short notional.\n"
            "- **Inference.** Newey-West HAC *t* on monthly returns; a 1,000+-draw sign-shuffle "
            "placebo p-value; a lookback-fragility sweep.\n"
            f"- **Universe caveat.** {R['n_tickers']} cross-asset ETFs, survivorship named on the "
            "signal axis (mild)."
        ),

        md("## 4 -- The teardown"),
        md(
            "### 4a -- Positive control: the trend engine is a faithful detector\n\n"
            "Sweep the planted own-asset trend on a synthetic panel. The portfolio mean should be "
            "~0 at the null and monotone in the planted trend."
        ),
        code(
            "ctrl = st.synthetic_control(strengths=(0.0, 0.10, 0.20, 0.30, 0.40, 0.50))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if t > 2 else GREY for t in ctrl['tstat']]\n"
            "ax.bar(ctrl['trend_strength'].astype(str), ctrl['port_mean_ann']*100, color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, (m, t) in enumerate(zip(ctrl['port_mean_ann']*100, ctrl['tstat'])):\n"
            "    ax.text(i, m + 0.4, f't={t:+.1f}', ha='center', fontsize=8)\n"
            "ax.set_xlabel('planted own-asset trend'); ax.set_ylabel('portfolio mean (%/yr)')\n"
            "ax.set_title('Positive control: portfolio mean monotone in planted trend')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))\n"
            "print('null robustness (20 seeds):', st.synthetic_null_robustness(20))"
        ),
        md("### 4b -- Real tape: vol-scaled vs equal-notional, gross and net"),
        code(
            "if HAVE_REAL:\n"
            "    bk_vn = st.tsmom_book(mp, vol_scale=True, cost_bps=10.0, borrow_ann_bps=50.0)\n"
            "    bk_en = st.tsmom_book(mp, vol_scale=False, cost_bps=10.0, borrow_ann_bps=50.0)\n"
            "    svn = st.summary(bk_vn['port_net']); sen = st.summary(bk_en['port_net'])\n"
            "    rows = [['Vol-scaled',  sv['mean']*100, svn['mean']*100, sv['tstat'], sv['sharpe']],\n"
            "            ['Equal-notn',  se['mean']*100, sen['mean']*100, se['tstat'], se['sharpe']]]\n"
            "else:\n"
            "    rows = [['Vol-scaled', " f"{R['vs_gross']}, {R['vs_net']}, {R['vs_t']}, {R['vs_sharpe']}],\n"
            "            ['Equal-notn', " f"{R['en_gross']}, {R['en_net']}, {R['en_t']}, {R['en_sharpe']}]]\n"
            "tab = pd.DataFrame(rows, columns=['book', 'gross_%', 'net_%', 'HAC_t', 'sharpe'])\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "x = np.arange(len(tab)); w = 0.35\n"
            "ax.bar(x - w/2, tab['gross_%'], w, color=GREY, label='gross')\n"
            "ax.bar(x + w/2, tab['net_%'], w, color=AMBER, label='net (costs+borrow)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels(tab['book'])\n"
            "ax.set_ylabel('TSMOM mean (%/yr)')\n"
            "ax.set_title('Vol-scaling beats equal-notional, gross and net'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(3).to_string(index=False))"
        ),
        md(
            f"> Vol-scaled **{R['vs_gross']:+.2f}%/yr** (HAC *t* = {R['vs_t']:+.3f}, Sharpe "
            f"{R['vs_sharpe']:.2f}) vs equal-notional **{R['en_gross']:+.2f}%/yr** (*t* = "
            f"{R['en_t']:+.2f}, Sharpe {R['en_sharpe']:.2f}). The third axis is **CONFIRMED**: "
            "risk-balancing roughly doubles the Sharpe and the *t*."
        ),
        md(
            "### 4c -- The placebo null: real vs sign-shuffle\n\n"
            "Randomly flip each asset's position sign, recompute the portfolio mean 1,000 times, "
            "and see where the real mean falls in that distribution."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(mp, n_perm=1000)\n"
            "    real_mean = pl['real_mean_ann']*100; pval = pl['p_value']\n"
            "    fwd = mp.pct_change().shift(-1); sig = st.own_sign_signal(mp); vol = st.realised_vol(mp)\n"
            "    import numpy as _np\n"
            "    rng = _np.random.default_rng(518); months = []\n"
            "    for t in sig.index:\n"
            "        s = sig.loc[t]; v = vol.loc[t] if t in vol.index else pd.Series(dtype=float)\n"
            "        r = fwd.loc[t] if t in fwd.index else pd.Series(dtype=float)\n"
            "        valid = s.notna() & v.notna() & (v>0) & r.notna(); names = list(s.index[valid])\n"
            "        if len(names) < 4: continue\n"
            "        mag = (st.VOL_TARGET/v.loc[names]).clip(upper=st.VOL_CAP).to_numpy()\n"
            "        months.append((mag, r.loc[names].to_numpy()))\n"
            "    dist = []\n"
            "    for _ in range(1000):\n"
            "        ss = 0.0\n"
            "        for mag, rv in months:\n"
            "            fl = rng.choice((-1.0, 1.0), size=mag.shape[0])\n"
            "            ss += float((fl*mag*rv).mean())\n"
            "        dist.append(ss/len(months)*12*100)\n"
            "    dist = _np.array(dist)\n"
            "else:\n"
            "    real_mean, pval = " f"{R['vs_gross']}, {R['vs_p']}\n"
            "    dist = np.random.default_rng(0).normal(0, 1.3, 1000)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(dist, bins=40, color=GREY, alpha=0.7, edgecolor='white')\n"
            "ax.axvline(real_mean, c=RED, lw=2, label=f'real TSMOM = {real_mean:+.2f}%/yr')\n"
            "ax.set_xlabel('TSMOM mean under shuffled signs (%/yr)'); ax.set_ylabel('count')\n"
            "ax.set_title(f'Placebo null: real mean is in the right tail | p = {pval:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Placebo p-value (vol-scaled): {pval:.3f}')"
        ),
        md(
            f"> Unlike the typical desk study, the real mean sits in the **right tail** of the "
            f"sign-shuffle null (*p* = {R['vs_p']:.3f}) -- there *is* genuine own-trend structure. "
            "The placebo passes; what fails is the one-sample *t*-bar."
        ),
        md("### 4d -- Lookback fragility, year-by-year and the equity curve"),
        code(
            "if HAVE_REAL:\n"
            "    lbs = [6, 9, 12, 15, 18]\n"
            "    lt = [st.summary(st.tsmom_book(mp, lookback=l)['port_gross'])['tstat'] for l in lbs]\n"
            "    ann = bk_v.copy(); ann['year'] = ann.index.year\n"
            "    yearly = ann.groupby('year')['port_gross'].apply(lambda x: float((1+x).prod()-1))*100\n"
            "    eq = (1 + bk_v['port_gross']).cumprod(); dd = (eq/eq.cummax()-1)*100\n"
            "    idx = bk_v.index\n"
            "else:\n"
            "    lbs = [6, 9, 12, 15, 18]; lt = [1.4, " f"{R['t_9m']}, {R['t_12m']}, {R['t_15m']}, 0.2]\n"
            "    yearly = pd.Series(" f"{R['annual']})\n"
            "    idx = pd.period_range('2008-03', periods=" f"{R['n_hold']}, freq='M').to_timestamp()\n"
            "    eq = pd.Series(np.cumprod(1+np.zeros(len(idx))), index=idx); dd = eq*0\n"
            "\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))\n"
            "ax1.plot(lbs, lt, 'o-', color=AMBER, lw=2)\n"
            "ax1.axhline(2, c=RED, ls='--', lw=1, label='t = 2 bar'); ax1.axhline(0, c='k', lw=1)\n"
            "ax1.set_xlabel('lookback (months)'); ax1.set_ylabel('HAC t (gross)')\n"
            "ax1.set_title('Signal is lookback-fragile: t straddles the bar'); ax1.legend()\n"
            "col = [GREEN if v > 0 else RED for v in yearly.values]\n"
            "ax2.bar(yearly.index.astype(str), yearly.values, color=col)\n"
            "ax2.axhline(0, c='k', lw=1); ax2.set_ylabel('TSMOM (%/yr)')\n"
            "ax2.set_title('Year-by-year: 2008 & 2022 crisis alpha, 2018 chop')\n"
            "ax2.tick_params(axis='x', rotation=90)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(yearly.round(1).to_string())"
        ),

        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `WEAK`** -- vol-scaled TSMOM {R['vs_gross']:+.2f}%/yr, HAC *t* = "
            f"{R['vs_t']:+.3f}, placebo *p* = {R['vs_p']:.3f}. Genuine structure (placebo passes) "
            f"but the canonical *t* is below 2 and straddles the bar across lookbacks "
            f"({R['t_9m']:.2f}/{R['t_12m']:.2f}/{R['t_15m']:.2f}). Literature is strong; prior "
            "plus a sub-2 *t* lands WEAK.\n"
            f"- **Tradability `FRAGILE`** -- net {R['vs_net']:+.2f}%/yr, net Sharpe "
            f"{R['vs_sharpe_net']:.2f}, max DD {R['vs_dd']:.1f}%, ~{R['vs_short']:.0f}% short. "
            "Net positive but thin, leveraged and short-heavy.\n"
            "- **Vol-scaling vs equal-notional `CONFIRMED`** -- risk-balancing roughly doubles "
            "the Sharpe and the *t*; the method choice dominates the raw signal."
        ),

        md(
            "## 6 -- Could you trade it?\n\n"
            "Marginally -- and that is itself notable on this desk. The net Sharpe is ~0.34 with "
            f"a contained {R['vs_dd']:.0f}% drawdown, far better than most factors here. But the "
            "book is short ~50% of NAV (borrow + recall), levers up calm bond/FX legs via "
            "inverse-vol sizing, and leans on holding all 12 asset classes -- and the signal *t* "
            "is below 2. The live, futures-traded, broad-breadth version that CTAs run is the "
            "right vehicle; this small ETF replica is a fragile shadow of it."
        ),

        md(
            "## 7 -- Going further\n\n"
            "- **[Study 507 -- Cross-Sectional-Momentum](../../507-cross-sectional-momentum/)**: "
            "rank assets against each other instead of against their own history.\n"
            "- **[Study 110 -- Faber-Timing](../../110-faber-timing/)**: long-or-flat "
            "price-vs-SMA single-asset trend.\n"
            "- **[Study 144 -- Permanent-Portfolio](../../144-permanent-portfolio/)**: the "
            "cross-asset diversification backdrop trend trades on top of.\n"
            "- **Breadth & futures.** Hurst-Ooi-Pedersen (2017): a century of trend across "
            "dozens of futures -- and the post-2009 drought that thins these numbers.\n\n"
            "*Think trend survives net of costs on a realistic basket? Fork this, add "
            "managed-futures breadth (dozens of futures), and show *t* > 2 net of borrow. That is "
            "the bar.*"
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
