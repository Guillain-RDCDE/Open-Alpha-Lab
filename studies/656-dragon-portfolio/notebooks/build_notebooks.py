"""Generate the two narrative notebooks for Study 656 (Dragon Portfolio).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY/TLT/
GLD/DBC/VXX/SHY tapes under ../_cache/ and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md). The synthetic crisis-hedge control runs anywhere
with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SPY/TLT/GLD/
# DBC/VXX/SHY, 2002-01-02 -> 2026-06-30; DBMF side-check 2019-05-07 ->).
R = dict(
    asof="2026-06-30", fp_prices="3dd12240ccde",
    core_start="2007-02-01", full_start="2018-03-01",
    core_n=4883, full_n=2094,
    # core window (Dragon-lite, ex-vol)
    dl_cagr=7.82, dl_vol=9.22, dl_sharpe=0.668, dl_dd=-26.4, dl_w12=-21.0,
    p6040_core_cagr=8.30, p6040_core_vol=11.21, p6040_core_sharpe=0.601,
    p6040_core_dd=-31.4, p6040_core_w12=-26.2,
    awl_core_cagr=7.12, awl_core_vol=10.13, awl_core_sharpe=0.551,
    awl_core_dd=-27.9, awl_core_w12=-24.8,
    spy_core_cagr=10.90, spy_core_vol=19.67, spy_core_sharpe=0.519,
    spy_core_dd=-55.2, spy_core_w12=-47.4,
    # full window (Dragon-full, 5 sleeve)
    df_cagr=-1.28, df_vol=15.03, df_sharpe=-0.148, df_dd=-45.0, df_w12=-24.7,
    p6040_full_cagr=8.66, p6040_full_vol=12.15, p6040_full_sharpe=0.596,
    p6040_full_dd=-27.7, p6040_full_w12=-25.8,
    awl_full_cagr=9.64, awl_full_vol=10.14, awl_full_sharpe=0.790,
    awl_full_dd=-18.6, awl_full_w12=-13.8,
    spy_full_cagr=14.72, spy_full_vol=19.18, spy_full_sharpe=0.703,
    spy_full_dd=-33.7, spy_full_w12=-19.7,
    # 2020 crash + 2022 both-down year
    c2020_dl=3.71, c2020_6040=3.67, c2020_awl=-2.93, c2020_spy=-9.22, c2020_df=33.35,
    y2022_dl=-9.32, y2022_6040=-23.32, y2022_awl=-8.73, y2022_spy=-18.18, y2022_df=-10.90,
    # VXX standalone
    vxx_cagr=-40.7, vxx_vol=73.7, vxx_dd=-99.5, vxx_cum=-98.8, vxx_covid=271.6,
    # inference
    hac_lite_gap=-0.049, hac_lite_t=-0.37, hac_lite_n=233,
    hac_full_gap=-0.801, hac_full_t=-1.65, hac_full_n=100,
    boot_lite_pt=0.067, boot_lite_lo=-0.257, boot_lite_hi=0.373, boot_lite_win=67,
    boot_full_pt=-0.745, boot_full_lo=-1.987, boot_full_hi=0.357, boot_full_win=9,
    boot_awl_pt=-0.938, boot_awl_lo=-1.975, boot_awl_hi=0.086, boot_awl_win=3,
    # cost sensitivity
    cost0_cagr=-1.24, cost0_sharpe=-0.146, cost5_cagr=-1.28, cost5_sharpe=-0.148,
    cost10_cagr=-1.31, cost10_sharpe=-0.151,
    # DBMF side-check
    dbmf_trend_cagr=9.4, dbmf_trend_vol=15.3, dbmf_real_cagr=9.2, dbmf_real_vol=12.4,
    dbmf_corr=0.28,
    # synthetic control
    syn_null_t=-0.22, syn_null_sd=1.20, syn_null_fire=1, syn_planted_t=30.93,
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![A_cheap_ETF_Dragon%3F: Busted](https://img.shields.io/badge/A_cheap_ETF_Dragon%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from dragon_portfolio import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PX = data.prices_frame(tickers=("SPY", "TLT", "GLD", "DBC", "VXX", "SHY"))
    RET = st.build_returns(PX)
    RF = RET["SHY"]
else:
    PX = RET = RF = None
print("real cache present:", HAVE_REAL, "| tickers:",
      (0 if PX is None else PX.notna().sum().to_dict()))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Can one portfolio survive both a crash AND an inflation spike? 🐉\n"
            "### The Dragon Portfolio — a real diversification story, wrapped around one "
            "sleeve you (probably) can't buy cheaply\n\n"
            + BADGES +
            "Chris Cole, a hedge-fund manager known for tail-risk trading, built a **100-year "
            "backtest** arguing the classic 60/40 stocks-and-bonds mix has a blind spot: it "
            "has nothing that *profits* when everything else craters at once — 2008, March "
            "2020. His fix, the \"Dragon Portfolio\", adds two extra sleeves most portfolios "
            "skip: **commodity trend-following** and **long volatility** — insurance that "
            "pays off exactly when stocks and bonds fail together.\n\n"
            "We can't rebuild his 100-year backtest (that data isn't public). What we *can* "
            "do is buy the cheapest available version with real ETFs and see what actually "
            "happens on the tape since 2007 — including the two best real-world tests we "
            "have: the 2020 crash and the 2022 inflation shock.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the bootstrap and the "
            "cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the mix genuinely diversify? | **Somewhat — but not provably.** Without "
            f"the volatility sleeve, it beats 60/40 on risk-adjusted return "
            f"(Sharpe **{R['dl_sharpe']:.2f}** vs {R['p6040_core_sharpe']:.2f}) with a smaller "
            "drawdown — but the gap isn't statistically certified. |\n"
            f"| Did it survive the 2020 crash? | **Spectacularly.** The full 5-sleeve version "
            f"gained **+{R['c2020_df']:.0f}%** while the S&P fell {R['c2020_spy']:.0f}% in "
            "the same stretch — almost entirely thanks to the long-volatility sleeve. |\n"
            f"| Did it survive 2022 (stocks AND bonds down)? | **Yes, clearly** — Dragon lost "
            f"only **{R['y2022_df']:.0f}%** vs 60/40's **{R['y2022_6040']:.0f}%**. |\n"
            f"| So what's the catch? | **The insurance sleeve is brutally expensive.** The "
            f"only cheap way to buy \"long volatility\" (the VXX ETF) has bled "
            f"**{R['vxx_cum']:.0f}%** of its value since 2018 — so the *whole* 5-sleeve "
            f"portfolio actually **lost money** ({R['df_cagr']:.1f}%/yr) over the only "
            "window we can test it on, despite winning both crises. |\n\n"
            "> It wins the battles. On this tape, it hasn't yet won the war."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Every portfolio built from stocks and bonds alone has a blind spot: a "
            "regime where both fail together. Add gold and commodity trend for inflation, "
            "and long volatility for a deflationary crash, and you get a portfolio built to "
            "profit — not just survive — in every regime.\"* — Chris Cole's published "
            "allocation: **24% stocks, 18% bonds, 19% gold, 18% commodity trend, "
            "21% long volatility.**\n\n"
            "It's a genuinely different idea from most \"all-weather\" recipes: instead of "
            "just spreading risk around (Ray Dalio's approach), it deliberately buys "
            "**convexity** — an asset that goes *up a lot* exactly when everything else "
            "craters."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If this works as advertised, it's the closest thing to a free lunch investing "
            "offers: real crisis protection that doesn't cost you the return you'd expect "
            "from insurance. If it doesn't — if the \"insurance\" sleeve just bleeds money "
            "every year it isn't paid off — then the Dragon Portfolio is really just a "
            "clever repackaging of a very old, very expensive idea: buying puts."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **The proxies.** SPY (stocks), TLT (long bonds), GLD (gold), a 12-month "
            "trend-following overlay on DBC (commodities) for the trend sleeve, and VXX for "
            "the long-vol sleeve — named up front as a **crude, decaying** stand-in; nobody "
            "can cheaply buy what Cole's paper actually models.\n"
            f"- **The windows.** Without VXX, we can test back to **{R['core_start']}** (it "
            f"sees 2008 too). With the real VXX sleeve, yfinance's own data only reaches back "
            f"to **{R['full_start']}** — VXX the *product* dates to 2009, but the ticker was "
            "relaunched in 2018 and that's as far back as the data goes.\n"
            "- **The comparisons.** 60/40 (the mainstream benchmark) and a simple equal-weight "
            "\"All-Weather-lite\" of the same four non-vol assets.\n"
            "- **The stress tests.** The 2020 COVID crash and the 2022 stocks-and-bonds year — "
            "the two cleanest real-world tests of exactly the claim being made."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, without the expensive sleeve** — does the diversification kernel "
            "(stocks + bonds + gold + trend) even help?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    dlite = st.blended_portfolio(RET, st.DRAGON_LITE, cost_bps=5.0, start=data.CORE_START)\n"
            "    p6040 = st.blended_portfolio(RET, st.SIXTY_FORTY, cost_bps=5.0, start=data.CORE_START)\n"
            "    dl_s, p6_s = st.portfolio_stats(dlite, rf=RF), st.portfolio_stats(p6040, rf=RF)\n"
            "    dl_sh, p6_sh, dl_dd, p6_dd = dl_s['sharpe'], p6_s['sharpe'], dl_s['max_dd']*100, p6_s['max_dd']*100\n"
            "else:\n"
            "    dl_sh, p6_sh, dl_dd, p6_dd = R['dl_sharpe'], R['p6040_core_sharpe'], R['dl_dd'], R['p6040_core_dd']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar(['Dragon-lite','60/40'], [dl_sh, p6_sh], color=[GREEN, GREY], width=.55)\n"
            "for i,v in enumerate([dl_sh, p6_sh]): a1.annotate(f'{v:.3f}',(i,v),ha='center',va='bottom')\n"
            "a1.set_title('Sharpe (excess of cash)'); a1.axhline(0, c='k', lw=.8)\n"
            "a2.bar(['Dragon-lite','60/40'], [dl_dd, p6_dd], color=[GREEN, GREY], width=.55)\n"
            "for i,v in enumerate([dl_dd, p6_dd]): a2.annotate(f'{v:.1f}%',(i,v),ha='center',va='top')\n"
            "a2.set_title('Max drawdown'); a2.axhline(0, c='k', lw=.8)\n"
            "plt.suptitle('2007-2026, no long-vol sleeve: better Sharpe, smaller drawdown')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Dragon-lite Sharpe {dl_sh:.3f} vs 60/40 {p6_sh:.3f}  |  '\n"
            "      f'MaxDD {dl_dd:.1f}% vs {p6_dd:.1f}%')"
        ),
        md(
            f"Yes — the plain-vanilla kernel genuinely diversifies: Sharpe **{R['dl_sharpe']:.2f}** "
            f"vs 60/40's **{R['p6040_core_sharpe']:.2f}**, and a smaller drawdown "
            f"(**{R['dl_dd']:.0f}%** vs **{R['p6040_core_dd']:.0f}%**) — for a return forfeit of "
            "less than half a point a year. The quants notebook shows this gap isn't "
            "statistically certified with 19 years of data, but it's a real, honest edge on "
            "paper.\n\n"
            "**Now add the real long-volatility sleeve** — the actual, full 5-sleeve Dragon:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    dfull = st.blended_portfolio(RET, st.DRAGON_FULL, cost_bps=5.0, start=data.FULL_START)\n"
            "    p6f = st.blended_portfolio(RET, st.SIXTY_FORTY, cost_bps=5.0, start=data.FULL_START)\n"
            "    df_s, p6f_s = st.portfolio_stats(dfull, rf=RF), st.portfolio_stats(p6f, rf=RF)\n"
            "    df_cagr, p6f_cagr = df_s['cagr']*100, p6f_s['cagr']*100\n"
            "else:\n"
            "    df_cagr, p6f_cagr = R['df_cagr'], R['p6040_full_cagr']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['Dragon-full\\n(5 sleeve, w/ VXX)','60/40'], [df_cagr, p6f_cagr],\n"
            "       color=[RED, GREY], width=.55)\n"
            "for i,v in enumerate([df_cagr, p6f_cagr]): ax.annotate(f'{v:+.2f}%/yr',(i,v),\n"
            "    ha='center', va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('CAGR')\n"
            "ax.set_title('Add the real long-vol sleeve: the whole portfolio LOSES money')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Dragon-full CAGR {df_cagr:+.2f}%/yr vs 60/40 {p6f_cagr:+.2f}%/yr')"
        ),
        md(
            f"The full portfolio's CAGR is **{R['df_cagr']:.2f}%/yr — negative** — over the "
            f"only window we can test (2018-2026), while 60/40 made **{R['p6040_full_cagr']:.1f}%/yr** "
            "in the same stretch. That's the whole story in one bar chart. Why? Because the "
            "only cheap way to hold \"long volatility\" is a product that structurally bleeds:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    vxx_ret = RET['VXX'].dropna()\n"
            "    cum = (1.0 + vxx_ret).cumprod()\n"
            "else:\n"
            "    idx = pd.period_range('2018-01', periods=100, freq='M').to_timestamp()\n"
            "    cum = pd.Series(np.geomspace(1.0, 1.0 + R['vxx_cum']/100, len(idx)), index=idx)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.plot(cum.index, cum.values, color=RED, lw=1.3)\n"
            "ax.set_yscale('log')\n"
            "ax.axhline(1.0, c='k', lw=.7, ls='--')\n"
            "ax.set_title(f'VXX: -{abs(R[\"vxx_cum\"]):.0f}% cumulative since 2018, one +{R[\"vxx_covid\"]:.0f}% spike and all')\n"
            "ax.set_ylabel('$1 invested (log scale)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'VXX cumulative return since inception: {R[\"vxx_cum\"]:+.1f}%   '\n"
            "      f'COVID spike (Feb19-Mar23 2020): {R[\"vxx_covid\"]:+.1f}%')"
        ),
        md(
            f"VXX lost **{abs(R['vxx_cum']):.0f}%** of its value since 2018 — while paying off "
            f"**+{R['vxx_covid']:.0f}%** in the six weeks of the COVID crash. That single spike "
            "is nowhere near enough to offset eight years of bleed at 21% portfolio weight. "
            "**This is the honest core finding:** the insurance is real (it fires exactly "
            "when it's supposed to) but it isn't cheap, and on this tape it hasn't yet paid "
            "for itself.\n\n"
            "**And yet — look at the two crisis episodes side by side:**"
        ),
        code(
            "labels = ['2020 crash\\n(Jan-Apr)', '2022\\n(stocks & bonds down)']\n"
            "dragon = [R['c2020_df'], R['y2022_df']]\n"
            "sixforty = [R['c2020_6040'], R['y2022_6040']]\n"
            "x = np.arange(2); w = 0.35\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(x - w/2, dragon, w, label='Dragon-full', color=GREEN)\n"
            "ax.bar(x + w/2, sixforty, w, label='60/40', color=GREY)\n"
            "for i,v in enumerate(dragon): ax.annotate(f'{v:+.1f}%',(i-w/2,v),ha='center',va='bottom')\n"
            "for i,v in enumerate(sixforty): ax.annotate(f'{v:+.1f}%',(i+w/2,v),ha='center',va='bottom' if v>0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('return'); ax.legend()\n"
            "ax.set_title('The two regimes the Dragon is built for: it wins both')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'2020: Dragon {R[\"c2020_df\"]:+.1f}% vs 60/40 {R[\"c2020_6040\"]:+.1f}%')\n"
            "print(f'2022: Dragon {R[\"y2022_df\"]:+.1f}% vs 60/40 {R[\"y2022_6040\"]:+.1f}%')"
        ),
        md(
            "It genuinely wins both episodes — the deflationary crash AND the inflationary "
            "both-down year. That's real, useful diversification behavior. The catch is that "
            "\"winning the crisis\" and \"making money overall\" are two different questions, "
            "and on the tape we have, the answer to the second one is currently no."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Mixed.** Real diversification without the vol sleeve (better "
            "Sharpe, smaller drawdown than 60/40) — but not statistically certified. Real "
            "crisis-window wins in 2020 and 2022 with the vol sleeve — but the aggregate "
            "Sharpe over that same window is negative.\n"
            "- **Tradability — Fragile.** Cheap and liquid, but the full published "
            "allocation lost money outright over the only window we can test.\n"
            "- **\"A cheap ETF Dragon just works?\" — Busted.** The idea is sound; the "
            "cheap implementation of the insurance sleeve is the part that fails."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The real Dragon Portfolio doesn't use VXX.** Cole's own firm runs an "
            "actively-managed options/variance book for the long-vol sleeve — sized, rolled "
            "and hedged dynamically. A buy-and-hold ETN is the cheapest possible substitute, "
            "and cheapest is not free.\n"
            "- **We haven't seen a real secular inflation regime.** 2022 was one loud year; "
            "the 1970s were a decade. The claim that this mix survives a *multi-year* "
            "inflationary regime remains genuinely untested on public data.\n"
            "- **Sibling studies:** [68-all-weather](../../68-all-weather/) (risk parity, no "
            "vol sleeve), [144-permanent-portfolio](../../144-permanent-portfolio/) (no "
            "trend or vol), [617-crash-insurance-cost](../../617-crash-insurance-cost/) "
            "(what standalone tail insurance costs), [655-ivy-portfolio](../../655-ivy-portfolio/) "
            "(trend-times everything, no vol).\n\n"
            "*Think you can build the long-vol sleeve cheaper than VXX and keep the crisis "
            "payoff? Show a net Sharpe improvement over the whole sample — then we'll talk.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# The Dragon Portfolio — a quantitative teardown 🔬\n"
            "### Core vs full 5-sleeve backtests · HAC & block-bootstrap inference · the "
            "2020/2022 crisis anatomy · the VXX decay diagnostic · the DBMF proxy-quality "
            "check · a synthetic crisis-alpha control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "Cole's claim — a 5-sleeve mix (equity/bonds/gold/trend/long-vol, published "
            "weights 24/18/19/18/21%) survives both inflationary and deflationary regimes — "
            "is tested on the only public tape available: liquid ETFs since 2007 (ex-vol) or "
            "2018 (with VXX). No claim is made about the un-testable 100-year secular cycle.\n\n"
            "> ⚠️ **Data note.** SPY/TLT/GLD/DBC/VXX/SHY daily auto-adjusted closes (yfinance, "
            f"cached, fingerprint `{R['fp_prices']}`); DBMF (2019→) as a side-check only. "
            f"**Named data quirk:** VXX the product dates to 2009-01-30, but yfinance's own "
            f"tape starts 2018-01-25 (the 2018 'Series B' relaunch) — the true binding "
            "constraint on the full 5-sleeve window. No cross-sectional basket anywhere, so "
            "no basket survivorship; VXX's own relaunch history is named instead. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | Dragon-lite Sharpe **{R['dl_sharpe']:.3f}** vs 60/40 "
            f"**{R['p6040_core_sharpe']:.3f}** (HAC *t* on the monthly gap = "
            f"**{R['hac_lite_t']:.2f}**, uncertified); Dragon-full Sharpe **{R['df_sharpe']:.3f}** "
            f"vs 60/40 **{R['p6040_full_sharpe']:.3f}** over 2018-2026 (bootstrap Dragon wins "
            f"only **{R['boot_full_win']}%** of resamples) |\n"
            f"| **Tradability** | `FRAGILE` | Dragon-full CAGR **{R['df_cagr']:.2f}%/yr** "
            f"(negative); VXX standalone **{R['vxx_cum']:.1f}%** cumulative since 2018 |\n"
            f"| **Cheap ETF Dragon?** | `BUSTED` | wins both crisis episodes "
            f"(2020 **+{R['c2020_df']:.0f}%**, 2022 **{R['y2022_df']:.1f}%** vs 60/40 "
            f"**{R['y2022_6040']:.1f}%**) but the aggregate Sharpe over the same window is "
            "negative |\n\n"
            "> 💡 In plain words: the diversification kernel is real; the cheap version of "
            "the insurance sleeve isn't paying for itself yet."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $w = (0.24, 0.18, 0.19, 0.18, 0.21)$ be Cole's published weights over "
            "(SPY, TLT, GLD, TREND, VXX), monthly rebalanced. The claims:\n\n"
            "- **H₁ (ex-vol diversification).** The 4-sleeve kernel (equity/bonds/gold/trend, "
            "renormalised to 100%) improves risk-adjusted return over 60/40 — higher Sharpe, "
            "smaller drawdown — for a small CAGR forfeit.\n"
            "- **H₂ (crisis convexity).** The long-vol sleeve pays off sharply in a "
            "deflationary liquidity crisis (2020) without destroying the aggregate return.\n"
            "- **H₃ (inflation cushioning).** The mix cushions a stocks-and-bonds-both-down "
            "year (2022) better than 60/40.\n"
            "- **H₄ (cheap tradability).** The whole 5-sleeve mix, built from liquid ETFs "
            "at realistic costs, delivers this without a large aggregate return sacrifice.\n\n"
            "We find **H₁ directionally supported, not certified** (HAC *t* = "
            f"{R['hac_lite_t']:.2f}); **H₂ strongly supported** (VXX "
            f"+{R['vxx_covid']:.0f}% in the spike window); **H₃ supported** "
            f"(Dragon-full {R['y2022_df']:.1f}% vs 60/40 {R['y2022_6040']:.1f}% in 2022); "
            f"**H₄ REJECTED** — Dragon-full's CAGR is negative ({R['df_cagr']:.2f}%/yr) over "
            "the only testable window."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Two structurally different questions get two different tests. **Aggregate "
            "risk-adjusted comparison** (Dragon vs 60/40 over the whole sample) uses a "
            "**Newey-West HAC *t*** on the mean monthly return gap (monthly, matching the "
            "rebalance cadence and taming daily autocorrelation) plus a **circular "
            "block-bootstrap** CI on the Sharpe *difference* (21-day blocks, 2,000 "
            "resamples — preserves volatility clustering an i.i.d. bootstrap would destroy). "
            "**Single crisis episodes** (2020, 2022) are reported as exactly that — narrative "
            "evidence, n=1 each, no *t*-stat claimed on them (house rule: no conditional "
            "claim without uncertainty applies to *aggregate* claims; single historical "
            "episodes are named as episodes, not certified as a distribution)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Proxies.** SPY/TLT/GLD/DBC(+12-mo trend overlay)/VXX, monthly rebalance, "
            "5 bps one-way cost per rebalance leg (10 bps sensitivity shown), no shorts/no "
            "borrow.\n"
            f"- **Windows.** Core (ex-vol): **{R['core_start']} → {R['asof']}**, n={R['core_n']:,} "
            f"days. Full (w/ VXX): **{R['full_start']} → {R['asof']}**, n={R['full_n']:,} days "
            "— bound by yfinance's VXX tape, not the product's nominal 2009 launch.\n"
            "- **Execution.** The commodity-trend flag (DBC's trailing 12-month return, "
            "known at the prior month's close) decides the following month's position — "
            "the study's only forecast, calendar-known, zero look-ahead. Every other sleeve "
            "is an exogenous constant weight.\n"
            "- **Inference.** HAC *t* on the monthly Dragon-minus-60/40 gap; circular "
            "block-bootstrap CI on the Sharpe difference.\n"
            "- **Cross-checks.** VXX standalone decay/spike diagnostic; DBMF (real "
            "managed-futures ETF) correlation check on the trend proxy.\n"
            "- **Control.** Synthetic monthly 5-asset world, planted crisis-alpha knob; the "
            "null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The core (ex-vol) headline and its inference\n\n"
            "Fixed-weight blend, monthly rebalance, HAC *t* on the mean monthly return gap "
            "vs 60/40, bootstrap CI on the Sharpe difference."
        ),
        code(
            "if HAVE_REAL:\n"
            "    dlite = st.blended_portfolio(RET, st.DRAGON_LITE, cost_bps=5.0, start=data.CORE_START)\n"
            "    p6040c = st.blended_portfolio(RET, st.SIXTY_FORTY, cost_bps=5.0, start=data.CORE_START)\n"
            "    awlc = st.blended_portfolio(RET, st.ALL_WEATHER_LITE, cost_bps=5.0, start=data.CORE_START)\n"
            "    spyc = st.spy_only(RET, start=data.CORE_START)\n"
            "    rows = [('Dragon-lite', dlite), ('60/40', p6040c), ('All-Weather-lite', awlc), ('SPY', spyc)]\n"
            "    stats = {n: st.portfolio_stats(s, rf=RF) for n, s in rows}\n"
            "    hac = st.hac_diff_monthly(dlite, p6040c)\n"
            "    boot = st.bootstrap_sharpe_diff(dlite, p6040c, rf=RF, seed=656)\n"
            "else:\n"
            "    stats = {'Dragon-lite': {'sharpe': R['dl_sharpe'], 'max_dd': R['dl_dd']/100, 'cagr': R['dl_cagr']/100},\n"
            "             '60/40': {'sharpe': R['p6040_core_sharpe'], 'max_dd': R['p6040_core_dd']/100, 'cagr': R['p6040_core_cagr']/100},\n"
            "             'All-Weather-lite': {'sharpe': R['awl_core_sharpe'], 'max_dd': R['awl_core_dd']/100, 'cagr': R['awl_core_cagr']/100},\n"
            "             'SPY': {'sharpe': R['spy_core_sharpe'], 'max_dd': R['spy_core_dd']/100, 'cagr': R['spy_core_cagr']/100}}\n"
            "    hac = {'t': R['hac_lite_t'], 'mean_diff_monthly': R['hac_lite_gap']/100, 'n_months': R['hac_lite_n']}\n"
            "    boot = {'point': R['boot_lite_pt'], 'ci95': (R['boot_lite_lo'], R['boot_lite_hi']), 'frac_a_wins': R['boot_lite_win']/100}\n"
            "names = list(stats.keys())\n"
            "sharpes = [stats[n]['sharpe'] for n in names]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "cols = [GREEN if n == 'Dragon-lite' else GREY for n in names]\n"
            "ax.bar(names, sharpes, color=cols, width=.6)\n"
            "for i,v in enumerate(sharpes): ax.annotate(f'{v:.3f}',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('Sharpe (excess of cash)')\n"
            "ax.set_title(f'Core window ({R[\"core_start\"]} -> {R[\"asof\"]}), ex-vol kernel')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"HAC t (Dragon-lite - 60/40, monthly): {hac['t']:+.2f}  \"\n"
            "      f\"(mean gap {hac['mean_diff_monthly']*100:+.3f}%/mo, n={hac['n_months']})\")\n"
            "print(f\"bootstrap Sharpe diff: {boot['point']:+.3f}  \"\n"
            "      f\"CI95=[{boot['ci95'][0]:+.3f}, {boot['ci95'][1]:+.3f}]  \"\n"
            "      f\"Dragon wins {boot['frac_a_wins']*100:.0f}% of resamples\")"
        ),
        md(
            f"> 💡 In plain words: Dragon-lite's Sharpe ({R['dl_sharpe']:.3f}) beats 60/40's "
            f"({R['p6040_core_sharpe']:.3f}) on the chart, but the HAC *t* on the underlying "
            f"monthly gap is only **{R['hac_lite_t']:.2f}** and the bootstrap CI "
            f"[{R['boot_lite_lo']:+.3f}, {R['boot_lite_hi']:+.3f}] straddles zero — Dragon "
            f"wins **{R['boot_lite_win']}%** of resamples, better than a coin flip but nowhere "
            "near certified. H₁: directionally real, not statistically proven with 19 years "
            "of data."
        ),
        md(
            "### 4b · The full 5-sleeve Dragon — where the vol sleeve bites\n\n"
            "Same construction, real VXX at 21% weight, window bound by yfinance's VXX tape "
            "(2018-03-01 onward)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    dfull = st.blended_portfolio(RET, st.DRAGON_FULL, cost_bps=5.0, start=data.FULL_START)\n"
            "    p6040f = st.blended_portfolio(RET, st.SIXTY_FORTY, cost_bps=5.0, start=data.FULL_START)\n"
            "    awlf = st.blended_portfolio(RET, st.ALL_WEATHER_LITE, cost_bps=5.0, start=data.FULL_START)\n"
            "    rows_f = [('Dragon-full', dfull), ('60/40', p6040f), ('All-Weather-lite', awlf)]\n"
            "    stats_f = {n: st.portfolio_stats(s, rf=RF) for n, s in rows_f}\n"
            "    hac_f = st.hac_diff_monthly(dfull, p6040f)\n"
            "    boot_f = st.bootstrap_sharpe_diff(dfull, p6040f, rf=RF, seed=656)\n"
            "    boot_awl = st.bootstrap_sharpe_diff(dfull, awlf, rf=RF, seed=656)\n"
            "else:\n"
            "    stats_f = {'Dragon-full': {'sharpe': R['df_sharpe'], 'cagr': R['df_cagr']/100},\n"
            "               '60/40': {'sharpe': R['p6040_full_sharpe'], 'cagr': R['p6040_full_cagr']/100},\n"
            "               'All-Weather-lite': {'sharpe': R['awl_full_sharpe'], 'cagr': R['awl_full_cagr']/100}}\n"
            "    hac_f = {'t': R['hac_full_t'], 'mean_diff_monthly': R['hac_full_gap']/100, 'n_months': R['hac_full_n']}\n"
            "    boot_f = {'point': R['boot_full_pt'], 'ci95': (R['boot_full_lo'], R['boot_full_hi']), 'frac_a_wins': R['boot_full_win']/100}\n"
            "    boot_awl = {'point': R['boot_awl_pt'], 'ci95': (R['boot_awl_lo'], R['boot_awl_hi']), 'frac_a_wins': R['boot_awl_win']/100}\n"
            "names_f = list(stats_f.keys())\n"
            "cagrs_f = [stats_f[n]['cagr']*100 for n in names_f]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "cols = [RED if n == 'Dragon-full' else GREY for n in names_f]\n"
            "ax.bar(names_f, cagrs_f, color=cols, width=.6)\n"
            "for i,v in enumerate(cagrs_f): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('CAGR')\n"
            "ax.set_title(f'Full window ({R[\"full_start\"]} -> {R[\"asof\"]}), real VXX sleeve')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"HAC t (Dragon-full - 60/40, monthly): {hac_f['t']:+.2f}  \"\n"
            "      f\"(mean gap {hac_f['mean_diff_monthly']*100:+.3f}%/mo, n={hac_f['n_months']})\")\n"
            "print(f\"bootstrap Sharpe diff vs 60/40: {boot_f['point']:+.3f}  \"\n"
            "      f\"CI95=[{boot_f['ci95'][0]:+.3f}, {boot_f['ci95'][1]:+.3f}]  \"\n"
            "      f\"Dragon wins {boot_f['frac_a_wins']*100:.0f}% of resamples\")\n"
            "print(f\"bootstrap Sharpe diff vs All-Weather-lite: {boot_awl['point']:+.3f}  \"\n"
            "      f\"CI95=[{boot_awl['ci95'][0]:+.3f}, {boot_awl['ci95'][1]:+.3f}]  \"\n"
            "      f\"Dragon wins {boot_awl['frac_a_wins']*100:.0f}% of resamples\")"
        ),
        md(
            f"> 💡 In plain words: neither comparison formally clears significance — the CIs "
            f"still straddle zero — but the point estimates lean hard negative "
            f"(Dragon wins only **{R['boot_full_win']}%** of resamples vs 60/40, "
            f"**{R['boot_awl_win']}%** vs All-Weather-lite) and the HAC *t* vs 60/40 is "
            f"**{R['hac_full_t']:.2f}**, closing in on but not crossing the bar. Honest read: "
            "\"probably worse on this tape, not certainly\" — H₄ (cheap tradability) is "
            "rejected on the point estimate even without full certification."
        ),
        md(
            "### 4c · The VXX decay diagnostic — why the full sleeve bites\n\n"
            "The mechanism, isolated: VXX's own buy-and-hold return."
        ),
        code(
            "if HAVE_REAL:\n"
            "    vxx_ret = RET['VXX'].dropna()\n"
            "    vxx_s = st.portfolio_stats(vxx_ret)\n"
            "    cum = float((1.0 + vxx_ret).prod() - 1.0) * 100\n"
            "    covid = st.window_return(vxx_ret, '2020-02-19', '2020-03-23') * 100\n"
            "else:\n"
            "    vxx_s = {'cagr': R['vxx_cagr']/100, 'vol': R['vxx_vol']/100, 'max_dd': R['vxx_dd']/100}\n"
            "    cum, covid = R['vxx_cum'], R['vxx_covid']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['cumulative\\nsince 2018','COVID spike\\n(6 weeks)'], [cum, covid],\n"
            "       color=[RED, GREEN], width=.55)\n"
            "for i,v in enumerate([cum, covid]): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('return')\n"
            "ax.set_title(f'VXX: CAGR {vxx_s[\"cagr\"]*100:+.1f}%/yr, vol {vxx_s[\"vol\"]*100:.0f}%, '\n"
            "             f'MaxDD {vxx_s[\"max_dd\"]*100:.0f}%')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'VXX CAGR {vxx_s[\"cagr\"]*100:+.1f}%/yr  cumulative {cum:+.1f}%  '\n"
            "      f'COVID spike {covid:+.1f}%')"
        ),
        md(
            f"> 💡 In plain words: an 8.4-year, **{R['vxx_dd']:.0f}%**-drawdown asset that "
            f"pays off **+{R['vxx_covid']:.0f}%** in six weeks once. At 21% portfolio weight "
            "that single payoff is not remotely enough to offset the compounding daily bleed "
            "in between. H₂ (crisis convexity) is confirmed as a *mechanism*; H₄ (cheap "
            "tradability) fails because the mechanism's cost, paid continuously, currently "
            "exceeds its payoff, realized rarely."
        ),
        md(
            "### 4d · The DBMF side-check — is the trend proxy even fair?\n\n"
            "Real managed-futures ETF (DBMF, 2019→) vs our single-index DBC trend overlay, "
            "over their overlap."
        ),
        code(
            "if HAVE_REAL and 'DBMF' in data.load_real():\n"
            "    real = data.load_real()\n"
            "    dbmf_ret = real['DBMF'].pct_change().dropna()\n"
            "    trend_ret = RET['TREND'].dropna()\n"
            "    common = dbmf_ret.index.intersection(trend_ret.index)\n"
            "    corr = float(np.corrcoef(dbmf_ret.loc[common], trend_ret.loc[common])[0, 1])\n"
            "    t_cagr = st.portfolio_stats(trend_ret.loc[common])['cagr'] * 100\n"
            "    d_cagr = st.portfolio_stats(dbmf_ret.loc[common])['cagr'] * 100\n"
            "else:\n"
            "    corr, t_cagr, d_cagr = R['dbmf_corr'], R['dbmf_trend_cagr'], R['dbmf_real_cagr']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.2))\n"
            "ax.bar(['our TREND proxy','real DBMF'], [t_cagr, d_cagr], color=[AMBER, GREY], width=.55)\n"
            "for i,v in enumerate([t_cagr, d_cagr]): ax.annotate(f'{v:+.1f}%/yr',(i,v),ha='center',va='bottom')\n"
            "ax.set_title(f'Similar CAGR, but daily correlation is only {corr:+.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'correlation: {corr:+.2f}  |  TREND proxy {t_cagr:+.1f}%/yr  vs  DBMF {d_cagr:+.1f}%/yr')"
        ),
        md(
            f"> 💡 In plain words: **{R['dbmf_corr']:+.2f}** correlation despite similar "
            "headline CAGR — a single long/flat overlay on one commodity index is a "
            "materially cruder proxy for Cole's diversified multi-market trend sleeve than "
            "the ticker choice alone would suggest. Named, not hidden."
        ),
        md(
            "### 4e · Synthetic crisis-alpha control — we know the truth here\n\n"
            "Deterministic monthly 5-asset world; the test isolates the mechanism directly — "
            "does the Dragon-weighted TREND+VOL sub-sleeve pay off MORE in crisis months than "
            "normal months? (Not \"Dragon vs 60/40\" — a lower equity weight alone would win "
            "any stock crash with zero genuine crisis alpha, which would make a naive null "
            "fire on a pure beta artefact.) Null checked over **20 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    frame, truth = data.synthetic_world(hedge_strength=0.0, seed=656 + s_)\n"
            "    null_ts.append(st.synthetic_crisis_test(frame, truth)['welch_t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "frame, truth = data.synthetic_world(hedge_strength=1.0, seed=656)\n"
            "planted_t = st.synthetic_crisis_test(frame, truth)['welch_t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (hedge=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5, label='planted hedge=1.0')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Welch t (crisis vs normal months, hedge sleeve)')\n"
            "ax.set_title('Control: the null is silent, a planted hedge lights up hard')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector fires in "
            f"**{R['syn_null_fire']}/20** seeds — about the 5% nominal false-positive rate of "
            f"a two-sided \\|t\\|≥2 test, not a systematic bias — and a planted crisis-alpha "
            f"effect reads t = **{R['syn_planted_t']:.1f}**. The machinery is unbiased; the "
            "real-tape result above is the genuine article, not an artefact of the test. "
            "*(A faithful-engine / power check only — never cited in support of the "
            "real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — ex-vol, Dragon-lite's Sharpe advantage over 60/40 "
            f"({R['dl_sharpe']:.3f} vs {R['p6040_core_sharpe']:.3f}) is real-looking but "
            f"uncertified (HAC *t* = {R['hac_lite_t']:.2f}, bootstrap CI crosses zero). With "
            f"the real VXX sleeve, the aggregate Sharpe over the only testable window "
            f"(2018-2026) turns negative ({R['df_sharpe']:.3f}), even as the same sleeve "
            f"delivers {R['c2020_df']:+.0f}% in the 2020 crash and cushions 2022 "
            f"({R['y2022_df']:.1f}% vs {R['y2022_6040']:.1f}%). Genuinely split by regime and "
            "by leg — the textbook use of the `MIXED` stamp.\n"
            f"- **Tradability `FRAGILE`** — cheap, liquid ETFs, negligible cost drag "
            f"(±0.03pp/yr from 0→10 bps), but the published 5-sleeve allocation lost money "
            f"outright ({R['df_cagr']:.2f}%/yr) over 8+ years; the insurance premium (VXX "
            f"{R['vxx_cum']:.1f}% cumulative) has not yet been repaid by the payoff it "
            "exists for.\n"
            "- **\"A cheap ETF Dragon just works?\" `BUSTED`** — the diversification "
            "mechanism is genuine; the *cheap* implementation of the long-vol sleeve is "
            "the part the myth gets wrong."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The obvious next step is options, not an ETN.** A dynamically-sized OTM SPX "
            "put ladder (à la Spitznagel's *Safe Haven*) would isolate convexity without "
            "paying VXX's structural roll cost every month it isn't paid off — the natural "
            "sequel study.\n"
            "- **A longer synthetic secular-cycle test** (replicated bond/gold/commodity "
            "indices reaching further back, in the spirit of Cole's own paper) would let the "
            "desk test the 1970s-style inflation decade this real tape simply doesn't "
            "contain.\n"
            "- **Dedup map:** [68-all-weather](../../68-all-weather/) (risk parity, no vol "
            "sleeve), [144-permanent-portfolio](../../144-permanent-portfolio/) (no trend or "
            "vol), [617-crash-insurance-cost](../../617-crash-insurance-cost/) (standalone "
            "tail-hedge cost), [655-ivy-portfolio](../../655-ivy-portfolio/) (trend-times "
            "everything, no vol).\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).*"
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
