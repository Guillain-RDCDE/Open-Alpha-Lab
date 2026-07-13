"""Generate the two narrative notebooks for Study 713 ("classic cars beat equities?").

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic: real-tape cells read the cached month-end
yfinance pulls under ../_cache/ for the equity proxies + benchmarks and the hardcoded
(cited, approximate) collector-car index from the package; on a cache miss they fall back
to the frozen headline numbers in ``R`` (mirroring docs/results.md). The synthetic control
runs anywhere.
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


# Frozen headline numbers — mirror of docs/results.md (car index hardcoded/cited/approx;
# equity proxies + SPY total-return month-end Adj Close, ^GSPC price-only, via yfinance,
# as-of 2025-12-31).
R = dict(
    win="2005 → 2025",
    idx_levels={2005: 100, 2006: 118, 2007: 138, 2008: 140, 2009: 148, 2010: 172,
                2011: 205, 2012: 245, 2013: 300, 2014: 355, 2015: 372, 2016: 375,
                2017: 380, 2018: 378, 2019: 372, 2020: 380, 2021: 410, 2022: 480,
                2023: 490, 2024: 470, 2025: 476},
    idx_yoy={2006: 18.0, 2007: 16.9, 2008: 1.4, 2009: 5.7, 2010: 16.2, 2011: 19.2,
             2012: 19.5, 2013: 22.4, 2014: 18.3, 2015: 4.8, 2016: 0.8, 2017: 1.3,
             2018: -0.5, 2019: -1.6, 2020: 2.2, 2021: 7.9, 2022: 17.1, 2023: 2.1,
             2024: -4.1, 2025: 1.3},
    peak_date="2023-12", peak_level=490, peak_trough=-4.1,
    idx_cagr=8.11, idx_vol=8.8, idx_sharpe=0.96, idx_mdd=-4.1,
    # the appraisal-smoothing tell
    rho=0.638, vol_obs=8.8, vol_desmoothed=18.6, sharpe_obs=0.96, sharpe_desmoothed=0.34,
    # benchmarks, year-end
    spy_cagr_ye=10.91, spy_vol_ye=17.3, spy_sharpe_ye=0.72, spy_mdd_ye=-36.8,
    gspc_cagr=8.88, gspc_vol=17.1, gspc_sharpe=0.61, gspc_mdd=-38.5,
    # annual excess
    excess_tr_mean=-3.97, excess_tr_t=-0.899, excess_tr_p=0.380, excess_tr_n=20,
    excess_po_mean=-1.94, excess_po_t=-0.442, excess_po_p=0.663, excess_po_n=20,
    # tradable proxies (monthly vs SPY TR)
    race_cagr=22.58, race_vol=28.3, race_sharpe=0.86, race_mdd=-29.1,
    race_alpha=10.24, race_beta=1.00, race_t=1.28, race_p=0.202, race_n=122,
    aml_cagr=-43.13, aml_vol=55.2, aml_sharpe=-0.70, aml_mdd=-98.3,
    aml_alpha=-47.87, aml_beta=1.54, aml_t=-2.78, aml_p=0.007, aml_n=86,
    spy_cagr=10.61, spy_vol=14.8, spy_sharpe=0.76, spy_mdd=-50.8,
    # carry haircut on the index gross CAGR
    carry_gross=8.11, carry_spread=-3.49, carry_carry=-2.50, carry_net=1.73,
    # synthetic control
    syn_peak=388, syn_end=381, syn_cagr=7.05, syn_sharpe=1.05, syn_mdd=-3.0,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Cars_beat_equities%3F: Busted](https://img.shields.io/badge/Cars_beat_equities%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from classic_car_index import data, strategy as st

HAVE_PROXIES = data.have_proxies()
IDX = data.load_car_index()                          # hardcoded, cited, APPROXIMATE proxy
PROX = data.load_proxies() if HAVE_PROXIES else None
print("equity/benchmark cache present:", HAVE_PROXIES,
      "| car-index years:", IDX.index[0].year, "->", IDX.index[-1].year)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do classic cars beat the stock market? 🏎️\n"
            "### The \"collector cars are an asset class\" pitch, in plain English\n\n"
            + BADGES +
            "You've seen the headline in every wealth-management brochure: *\"forget stocks — a "
            "Ferrari 250 GTO, a Porsche 911, a Mercedes 300 SL Gullwing only goes **up**. Classic cars "
            "are the best-performing luxury asset of the decade. They beat the S&P — and you get to "
            "**drive** them.\"* And the indices seem to agree: the collector-car market really did melt "
            "up through the 2010s, and it barely wobbled when stocks crashed.\n\n"
            "This notebook lines the collector-car index up next to the S&P 500 — on return, on risk, "
            "and on what it actually **costs to buy, store, insure and sell a car** — and asks the only "
            "question that matters: *would you have been richer in an index fund?* (Spoiler: the "
            "\"smooth ride\" is mostly an accident of how the index is built.)\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the Newey-West alpha, the de-smoothing "
            "and the cost algebra? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice — and a data note.** Real collector-car indices (HAGI Top, the "
            "Knight Frank Luxury Investment Index, Hagerty) aren't free to pull, so the car line below "
            "is a **small, clearly-cited, approximate** reconstruction of public reporting — a "
            "**proxy**, never presented as the live index. Every chart is drawn by the code beside it; "
            "house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did classic-car prices really moon? | **Yes — for a decade.** The index ran from 100 "
            f"(2005) to ~372 by 2015 — the great 2009–2015 melt-up that made cars the poster child of "
            "\"passion assets.\" That part is real. |\n"
            "| Did it keep beating stocks? | **No.** Over 2005–2025 the car index compounded at "
            f"**~{R['idx_cagr']:.1f}%/yr** vs **~{R['spy_cagr_ye']:.1f}%/yr** for the S&P (total "
            "return). After 2015 it mostly went sideways. |\n"
            "| But the ride was so smooth! | **That's an illusion.** A car index barely updates between "
            f"rare auctions, so it *looks* low-risk. Un-smear it and its true volatility (**~"
            f"{R['vol_desmoothed']:.0f}%**) is *higher* than the S&P's, and its risk-adjusted score "
            "*halves*. |\n"
            "| Could you at least buy the trade? | **Not cleanly.** The only listed \"car\" stocks are a "
            f"coin flip: Ferrari **soared** (+{R['race_cagr']:.0f}%/yr) while Aston Martin **cratered** "
            f"({R['aml_cagr']:.0f}%/yr). And once you pay auction fees + storage + insurance, the car "
            f"index's gross ~{R['carry_gross']:.1f}%/yr shrinks to **~{R['carry_net']:.1f}%/yr** — "
            "cash, basically. |\n\n"
            "> The 2010s boom was real. The *asset class that beats stocks* was not. The S&P won on "
            "total return, on true risk-adjusted return, and — by a mile — net of what a car costs to own."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Collector cars are a store of value that beats the stock market. The Historic "
            "Automobile Group index and the Knight Frank Luxury Investment Index show classic cars up "
            "over **185%** in a decade — the best-performing luxury asset there is. Buy the right "
            "chassis, keep it in the garage, sell it for more. And unlike a stock, it doesn't crash "
            "when the market does.\"*\n\n"
            "It's a *steelman-able* claim. Through the 2010s the blue-chip collector market genuinely "
            "melted up — a 1962 Ferrari 250 GTO changed hands for tens of millions, air-cooled Porsche "
            "911s tripled, and the published indices barely dipped in 2008 or 2020. For a decade, "
            "\"cars beat stocks\" looked like simple arithmetic."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it were durably true, it would be a big deal: an asset that beats equities, is "
            "uncorrelated with your portfolio, hedges inflation — *and* you get to drive it on a Sunday. "
            "That's the exact pitch of every classic-car fund and auction-house wealth report. But \"it "
            "went up for a decade\" and \"it's an asset class that beats the S&P\" are different "
            "statements. The first is about a **boom**; the second is a claim about the **long-run, "
            "risk-adjusted, net-of-cost** return. We can check the second directly — and the smoothness "
            "of the ride turns out to be the biggest tell of all."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Four honest comparisons, each against the S&P 500:\n\n"
            "1. **The car index vs the S&P.** Put the (cited, approximate) collector-car index next to "
            "the S&P on the same 2005–2025 clock — return, volatility, worst drawdown. And do it "
            "*fairly*: a car index is a **price** index, so we race it against the S&P **price-only** "
            "(`^GSPC`) — then note the S&P also pays **dividends** you'd have pocketed.\n"
            "2. **The smoothness trap.** A collectible index is built from rare, lagged appraisals, so "
            "it reports a suspiciously smooth line. We **un-smooth** it and see what the real volatility "
            "was.\n"
            "3. **The thing you can actually buy.** You can't buy \"the car index.\" You *can* buy "
            "**Ferrari** and **Aston Martin**. Do they deliver the car trade — or just a wild ride?\n"
            "4. **The cost of ownership.** A car isn't an ETF: auction houses take a cut on the way in "
            "*and* out, and it sits insured, stored and serviced for years. Charge that, and see what's "
            "left.\n\n"
            "**What would make us say \"asset class\"?** The index beats the S&P on *true, "
            "net-of-cost, risk-adjusted* return. Anything less is a boom story."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: the melt-up and the plateau.** Here is the (approximate, cited) collector-car "
            "index — a roaring decade, then a long flat line."
        ),
        code(
            "yrs = list(R['idx_levels'].keys())\n"
            "lv = [float(IDX.loc[f'{y}-12-31']) for y in yrs]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.plot(yrs, lv, 'o-', c=AMBER, lw=2.4, label='collector-car index (proxy)')\n"
            "ax.axvspan(2009, 2015, color=GREEN, alpha=.08)\n"
            "ax.annotate('the 2009–2015\\nmelt-up', (2012, 240), color='#3a7d44', fontsize=9, ha='center')\n"
            "ax.axvspan(2016, 2025, color=GREY, alpha=.10)\n"
            "ax.annotate('...then a decade\\nof plateau', (2020, 415), color='#555', fontsize=9, ha='center')\n"
            "ax.axhline(100, ls=':', c=GREY)\n"
            "ax.set_xlabel('year-end'); ax.set_ylabel('index level (base 100 = 2005)')\n"
            "ax.set_title('The classic-car \"asset class\": a decade of boom, a decade of drift'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('levels:', {y:int(round(v)) for y,v in zip(yrs, lv)})\n"
            "print(f\"from the {R['peak_date']} high the index is {R['peak_trough']:+.0f}% (a plateau, not a crash)\")"
        ),
        md(
            "It nearly **quadrupled** into 2015 — a genuinely spectacular decade. And then it stopped. "
            "From 2016 on it drifts: a small pandemic-era bump in 2022, a cooling in 2024, but no second "
            "leg. A store of value? Sure. A machine that keeps *beating stocks*? Only if you stop the "
            "clock in 2015."
        ),
        md(
            "**Now the race: cars vs the S&P.** Same money, same years — who's richer at the end? "
            "(We use the S&P's *total-return* line, dividends reinvested, because that's what an index "
            "fund actually hands you.)"
        ),
        code(
            "if HAVE_PROXIES:\n"
            "    spy = PROX['SPY']; spy_ye = spy.resample('YE').last()\n"
            "    spy_ye = spy_ye[(spy_ye.index.year>=2005)&(spy_ye.index.year<=2025)]\n"
            "    spy_norm = spy_ye/spy_ye.iloc[0]*100\n"
            "    sx = [d.year for d in spy_norm.index]; sy = list(spy_norm.values)\n"
            "else:\n"
            "    sx, sy = yrs, [100*(1+R['spy_cagr_ye']/100)**(y-2005) for y in yrs]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.plot(yrs, lv, 'o-', c=AMBER, lw=2.2, label=f\"classic cars  ({R['idx_cagr']:.1f}%/yr)\")\n"
            "ax.plot(sx, sy, 's-', c=GREEN, lw=2.2, label=f\"S&P 500 total return  ({R['spy_cagr_ye']:.1f}%/yr)\")\n"
            "ax.set_xlabel('year-end'); ax.set_ylabel('$100 invested at end-2005')\n"
            "ax.set_title('Cars vs the S&P: the boom kept them close, the plateau lost the race'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"cars end at ${lv[-1]/lv[0]*100:,.0f}  vs  S&P ${sy[-1]:,.0f}  (per $100 at end-2005)\")"
        ),
        md(
            f"The S&P wins: **~{R['spy_cagr_ye']:.1f}%/yr** vs **~{R['idx_cagr']:.1f}%/yr** for the "
            "cars. The gaudy 2009–2015 run kept the race close for a while, but the decade of plateau "
            "that followed — while stocks kept compounding — settled it. And this is *before* we've "
            "charged a single cost of actually owning a car."
        ),
        md(
            "**\"But cars are so much less risky — look how smooth that line is!\"** Here's the trick. "
            "A car index only moves when cars *sell*, and blue-chip cars sell rarely, at lagged, "
            "appraised prices. So the index looks calm the way a painting on your wall looks calm — "
            "you're just not getting a price quote every second. Un-smooth it (undo that lag) and the "
            "real bumpiness reappears:"
        ),
        code(
            "ds = st.desmooth_returns(IDX)\n"
            "labels = ['reported\\n(as published)', 'de-smoothed\\n(true)', 'S&P 500']\n"
            "vols = [ds['vol_obs']*100, ds['vol_desmoothed']*100, R['spy_vol_ye']]\n"
            "shs  = [ds['sharpe_obs'], ds['sharpe_desmoothed'], R['spy_sharpe_ye']]\n"
            "x = np.arange(3); fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.1))\n"
            "ax[0].bar(x, vols, .6, color=[AMBER, RED, GREEN]); ax[0].set_xticks(x); ax[0].set_xticklabels(labels)\n"
            "ax[0].set_ylabel('annual volatility %'); ax[0].set_title('The \"smooth ride\" is a mirage')\n"
            "ax[1].bar(x, shs, .6, color=[AMBER, RED, GREEN]); ax[1].set_xticks(x); ax[1].set_xticklabels(labels)\n"
            "ax[1].set_ylabel('risk-adjusted score (Sharpe)'); ax[1].set_title('...and so is the high Sharpe')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"serial correlation of yearly car returns: {ds['rho']:.2f} (a smooth, laggy index)\")\n"
            "print(f\"reported vol {ds['vol_obs']*100:.0f}%  ->  true vol {ds['vol_desmoothed']*100:.0f}%   |   \"\n"
            "      f\"reported Sharpe {ds['sharpe_obs']:.2f}  ->  true Sharpe {ds['sharpe_desmoothed']:.2f}\")"
        ),
        md(
            f"There it is. The car index's returns are **{R['rho']:.2f} correlated year-to-year** — the "
            "fingerprint of an appraisal-smoothed series. Undo the smoothing and its \"low\" "
            f"{R['vol_obs']:.0f}% volatility roughly **doubles to ~{R['vol_desmoothed']:.0f}%** — *higher* "
            f"than the S&P — and its flattering Sharpe **falls from {R['sharpe_obs']:.2f} to "
            f"{R['sharpe_desmoothed']:.2f}**, roughly *half* the S&P's. The calm was never real; it was "
            "the reporting."
        ),
        md(
            "**\"Fine — I'll buy the car *stocks*.\"** You can: Ferrari (`RACE`) and Aston Martin "
            "(`AML.L`). Do they hand you the collector-car trade — or a coin flip?"
        ),
        code(
            "names = ['Ferrari\\n(RACE)', 'Aston Martin\\n(AML.L)', 'S&P 500']\n"
            "cagrs = [R['race_cagr'], R['aml_cagr'], R['spy_cagr']]\n"
            "mdds  = [R['race_mdd'], R['aml_mdd'], R['spy_mdd']]\n"
            "x = np.arange(3); fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, cagrs, .4, color=[GREEN, RED, GREY], label='CAGR %/yr')\n"
            "ax.bar(x+.2, mdds, .4, color=RED, alpha=.5, label='worst drawdown %')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels(names)\n"
            "ax.set_ylabel('percent'); ax.set_title('The buyable proxies: a juggernaut and a wreck — not \"the car market\"'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"Ferrari: CAGR {R['race_cagr']:+.1f}%/yr — but it's a luxury-goods juggernaut, not a resale-price feed\")\n"
            "print(f\"Aston Martin: CAGR {R['aml_cagr']:+.1f}%/yr, drawdown {R['aml_mdd']:.0f}% — nearly a wipeout\")"
        ),
        md(
            "This is the whole problem with \"just buy the car stocks.\" **Ferrari** was a magnificent "
            "investment — but it's a high-margin luxury-goods brand printing money on new cars and "
            "merchandise, *not* a barometer of what a vintage 275 GTB fetches at auction. **Aston "
            f"Martin** nearly went to zero ({R['aml_mdd']:.0f}% drawdown). Same \"collector-car\" story, "
            "opposite outcomes: it's a coin flip on the company, not exposure to the trade."
        ),
        md(
            "**The part the pitch never mentions: it costs real money to own a car.** Auction houses "
            "take a buyer's premium *and* a seller's commission (~20%+ round-trip), and the car sits "
            "insured, climate-stored and serviced for years. Charge that against the index's gross "
            "return:"
        ),
        code(
            "labels = ['gross\\nreturn', 'auction\\nround-trip', 'storage,\\ninsurance,\\nservicing', 'NET to\\nyou']\n"
            "vals = [R['carry_gross'], R['carry_spread'], R['carry_carry'], R['carry_net']]\n"
            "cols = [AMBER, RED, RED, (RED if R['carry_net']<0 else '#b8860b')]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(R['spy_cagr_ye'], ls='--', c=GREEN, alpha=.7)\n"
            "ax.annotate(f\"S&P {R['spy_cagr_ye']:.0f}%/yr\", (3.1, R['spy_cagr_ye']), color=GREEN, fontsize=9, va='center')\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_ylabel('% per year'); ax.set_title('Where the classic-car \"return\" goes once you actually own one')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"gross {R['carry_gross']:+.1f}%/yr  ->  NET {R['carry_net']:+.1f}%/yr after auction spread + carry\")"
        ),
        md(
            f"A gross **{R['carry_gross']:.1f}%/yr** that already lost to the S&P shrinks to "
            f"**~{R['carry_net']:.1f}%/yr** — roughly a savings account — the moment you pay to buy, "
            "sell, store and insure the metal. And unlike a stock, a car pays you **no dividend** while "
            "you wait; it charges you rent."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Cars returned ~{R['idx_cagr']:.1f}%/yr vs ~{R['spy_cagr_ye']:.1f}%/yr "
            "for the S&P — they *under*-performed. The famous \"low risk\" is an appraisal-smoothing "
            "artifact; un-smoothed, the car index is *more* volatile than stocks with roughly *half* "
            "the Sharpe.\n"
            "- **Tradability — Mirage.** Net of auction fees + storage + insurance the gross return "
            "collapses to cash-like, and the only listed proxies are a juggernaut (Ferrari) and a wreck "
            "(Aston Martin) — no clean way to buy the trade.\n"
            "- **Cars beat equities? — Busted.** On total return, on *true* risk-adjusted return, and "
            "on net-of-cost, the index fund wins. The boom was real; the asset class was not."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the honest bottom line\n\n"
            "Imagine two people at the end of 2005, each with \\$10,000. One buys an S&P index fund. "
            "The other becomes a classic-car investor — buys, insures, stores, services, and eventually "
            "sells, paying the real frictions. Where do they land twenty years later?"
        ),
        code(
            "start = 10_000.0; yrs_h = 20\n"
            "spy_end = start*(1+R['spy_cagr_ye']/100)**yrs_h\n"
            "car_gross = start*(1+R['idx_cagr']/100)**yrs_h\n"
            "car_net = start*(1+R['carry_net']/100)**yrs_h   # net of spread + carry\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(['S&P index fund', 'car index\\n(gross, untradable)', 'car investor\\n(net of costs)'],\n"
            "       [spy_end, car_gross, car_net], color=[GREEN, AMBER, RED], width=.6)\n"
            "for i,v in enumerate([spy_end, car_gross, car_net]): ax.annotate(f'${v:,.0f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('value of $10,000 after 20 years')\n"
            "ax.set_title('Same $10k, end-2005 -> end-2025'); plt.tight_layout(); plt.show()\n"
            "print(f'S&P index fund: ${spy_end:,.0f}   |   car (gross): ${car_gross:,.0f}   |   car investor (net): ${car_net:,.0f}')"
        ),
        md(
            "The index-fund investor turns \\$10k into roughly **\\$80,000** doing nothing. The car "
            "investor — after auction fees, insurance and a decade-long plateau — ends up nearer "
            "**\\$14,000**, barely ahead of a savings account. The people who really got rich in cars "
            "bought a specific blue-chip chassis *before* 2015 and sold near the top: a handful of "
            "winners, narrated as an asset class. You only hear from them."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Pull the real index yourself.** Our car line is a cited *approximation*; HAGI, Knight "
            "Frank and Hagerty publish the live series. Swap one in — the shape (and the verdict) won't "
            "move, but you'll have the exact tape. And *always* de-smooth it before quoting a Sharpe.\n"
            "- **The collectibles pattern.** Watches, wine, art, sneakers, whisky: every \"passion "
            "asset\" tells the same story — a real boom, a smooth-looking index, brutal carry, equities "
            "win net of cost (see [docs/references.md](../docs/references.md)).\n"
            "- **The sibling study.** [Study 358 — Watches](../../358-watch-index/) is this exact shape "
            "in a different garage.\n\n"
            "*Think a specific chassis (a 250 GTO, a Carrera RS 2.7) beat the S&P net of every cost? "
            "Pull its auction history, charge the fees and the carry, and show it — then check it wasn't "
            "just one lucky car bought near the bottom.*"
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
            "# Classic cars as an asset class — a quantitative teardown 🔬\n"
            "### Car index vs S&P, price-only AND total-return (CAGR / vol / MDD + annual-excess *t*) · "
            "the appraisal-smoothing de-bias · Newey-West proxy alpha · the auction-spread + carry "
            "haircut on NAV · a synthetic positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We test "
            "the strongest tradable form of \"collector cars beat the S&P\": (H₁) the collector-car "
            "index out-returns the S&P; (H₂) its low measured volatility survives de-smoothing; (H₃) a "
            "buyable proxy carries alpha vs the market; (H₄) it survives the cost of owning the metal. "
            "We find **H₁ rejected** (it *under*-performs), **H₂ rejected** (the low vol is a smoothing "
            "artifact), **H₃ rejected** (no significant alpha; the proxies are a barbell), **H₄ "
            "rejected** (cash-like net of costs).\n\n"
            "> ⚠️ **Not investment advice — data provenance.** The car index is **hardcoded, cited, "
            "approximate** (public HAGI Top / Knight Frank Luxury Investment Index / Hagerty reporting "
            "— a *labelled proxy*, never the live feed). `RACE`, `AML.L`, `SPY` are month-end Adj Close "
            "(total-return) via yfinance; `^GSPC` is the S&P **price-only** index (as-of 2025-12-31). "
            "Offline core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md); numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Car index CAGR **{R['idx_cagr']:.1f}%** vs S&P total-return "
            f"**{R['spy_cagr_ye']:.1f}%** / price-only **{R['gspc_cagr']:.1f}%** (2005–2025); annual "
            f"excess vs TR **{R['excess_tr_mean']:+.1f}%/yr**, *t* = **{R['excess_tr_t']:+.2f}** "
            f"(n={R['excess_tr_n']}). De-smoothed Sharpe **{R['sharpe_desmoothed']:.2f}** ≪ S&P "
            f"**{R['spy_sharpe_ye']:.2f}**. |\n"
            f"| **Tradability** | `MIRAGE` | Gross index CAGR **{R['carry_gross']:+.1f}%** → **NET "
            f"{R['carry_net']:+.1f}%/yr** after a 22% auction round-trip + 2.5%/yr carry. Proxies are a "
            f"barbell: RACE α *t*={R['race_t']:+.2f} (n.s.), AML.L α *t*={R['aml_t']:+.2f} "
            f"(sig. **negative**). |\n"
            f"| **Cars beat equities?** | `BUSTED` | S&P wins CAGR ({R['spy_cagr_ye']:.1f} vs "
            f"{R['idx_cagr']:.1f}), *de-smoothed* Sharpe ({R['spy_sharpe_ye']:.2f} vs "
            f"{R['sharpe_desmoothed']:.2f}), and net-of-cost. The price-only near-tie dies on dividends "
            "+ carry. |\n\n"
            "> 💡 In plain words: the car market under-performed stocks, its famous \"low risk\" is an "
            "artifact of how the index is built, the only things you can buy are a lottery ticket, and "
            "ownership costs turn the gross return into cash. No axis survives."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let the collector-car index level be $I_t$ and the benchmark $B_t$. The claim is a joint "
            "hypothesis:\n\n"
            "- **H₁ (it out-returns).** Annual excess $\\;x_t = r^I_t - r^B_t\\;$ has "
            "$\\mathbb{E}[x_t] > 0$ with $t > 2$ — cars beat stocks.\n"
            "- **H₂ (the low risk is real).** The index's low measured volatility is not a smoothing "
            "artifact: after AR(1) un-smoothing $r^u_t = (r_t - \\rho r_{t-1})/(1-\\rho)$, the Sharpe "
            "stays above the S&P's.\n"
            "- **H₃ (it's buyable with alpha).** For a tradable proxy $P$, the intercept $\\alpha$ in "
            "$r^P_t = \\alpha + \\beta r^B_t + \\varepsilon_t$ is positive with a Newey-West *t* > 2.\n"
            "- **H₄ (it survives carry).** The net CAGR after the auction round-trip spread $s$ over "
            "hold $h$ and annual carry $c$ stays competitive: "
            "$(1+g)(1+((1-s)^{1/h}-1))(1-c)-1$ beats the S&P.\n\n"
            "The 2009–2015 melt-up is the steelman: for a decade H₁ held *in-sample*. The test is "
            "whether it holds over the **full cycle, de-smoothed, net of cost** — an asset class or a boom."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "If H₁–H₄ held, collector cars would be a genuine diversifier: equity-beating return, "
            "low-correlation, inflation-proof, drivable. But each leg is separately falsifiable. H₁ is a "
            "**return race** on a common clock — and it must be *fair*: a price index against the S&P "
            "**price-only**, because comparing a no-dividend car index to *total-return* equities "
            "quietly hands the cars the dividend yield stocks actually paid. H₂ is the "
            "**appraisal-smoothing** problem that inflates every private-asset Sharpe (real estate, "
            "PE, art). H₃ asks whether the *only investable expression* delivers anything beyond beta. "
            "H₄ is the **ownership tax**: a ~20%+ auction round-trip and years of insurance, storage "
            "and servicing carry. The asset-class claim needs all four."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Car index (proxy).** A hardcoded, cited, **approximate** annual level (base 100 @ "
            "2005), reconstructed from public HAGI Top / Knight Frank Luxury Investment Index / Hagerty "
            "reporting (2009–2015 melt-up; 2016–2020 plateau; 2022 bump; 2023–24 cooling). *Labelled a "
            "proxy* — its path is defensible, its precise year-end values are not a live feed.\n"
            "- **Benchmarks, labelled.** `SPY` (dividend-adjusted, **total return**) and `^GSPC` "
            "(**price-only**), both month-end via yfinance, cached. The car index is price-only, so the "
            "*fair* race is vs `^GSPC`; `SPY` shows the real equity outcome.\n"
            "- **De-smoothing.** First-order AR(1) un-smoothing of the annual index returns; report "
            "$\\rho$, reported vs de-smoothed vol and Sharpe. `REAL` low-risk needs the Sharpe to "
            "survive.\n"
            "- **Equity proxies.** `RACE` (from its 2015-10 IPO) and `AML.L` (from its 2018-10 IPO), "
            "monthly, Newey-West (6-lag) alpha vs `SPY`. Survivorship is **not** a concern (named "
            "tickers), but each series starts at its IPO — a stated look-ahead caveat.\n"
            "- **Cost (beat 6).** Charge a 22% auction round-trip over a 7y hold + 2.5%/yr carry "
            "**once on NAV**; net CAGR.\n"
            "- **Positive control.** A deterministic boom-then-plateau path with a *planted* drift; the "
            "engine must recover the up-sign and a finite Sharpe.\n"
            "- **What would make us say \"asset class\":** H₁ *t* > 2 **or** a proxy alpha *t* > 2, "
            "**and** a de-smoothed Sharpe above the S&P, **and** a net-of-cost CAGR that beats it. We "
            "find none of these."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The race — car index vs the S&P, priced fairly\n\n"
            "Year-end levels rebased to \\$100. The car index is a *price* index, so we show BOTH the "
            "price-only S&P (`^GSPC`, the apples-to-apples race) and the total-return S&P (`SPY`, the "
            "real equity outcome). Annual-excess *t* against each."
        ),
        code(
            "yrs = list(R['idx_levels'].keys())\n"
            "lv = np.array([float(IDX.loc[f'{y}-12-31']) for y in yrs])\n"
            "si = st.summarize(IDX, periods_per_year=1.0)\n"
            "if HAVE_PROXIES:\n"
            "    def ye(t):\n"
            "        s = PROX[t].resample('YE').last(); return s[(s.index.year>=2005)&(s.index.year<=2025)]\n"
            "    spy_ye, gspc_ye = ye('SPY'), ye('^GSPC')\n"
            "    spy = (spy_ye/spy_ye.iloc[0]*100).values; gspc = (gspc_ye/gspc_ye.iloc[0]*100).values\n"
            "    sx = [d.year for d in spy_ye.index]\n"
            "    ss = st.summarize(spy_ye, periods_per_year=1.0); sp = st.summarize(gspc_ye, periods_per_year=1.0)\n"
            "    ae_tr = st.annual_excess_t(IDX, PROX['SPY']); ae_po = st.annual_excess_t(IDX, PROX['^GSPC'])\n"
            "else:\n"
            "    spy = np.array([100*(1+R['spy_cagr_ye']/100)**(y-2005) for y in yrs])\n"
            "    gspc = np.array([100*(1+R['gspc_cagr']/100)**(y-2005) for y in yrs]); sx=yrs\n"
            "    si={'cagr':R['idx_cagr']/100,'vol':R['idx_vol']/100,'mdd':R['idx_mdd']/100}\n"
            "    ss={'cagr':R['spy_cagr_ye']/100,'vol':R['spy_vol_ye']/100,'mdd':R['spy_mdd_ye']/100}\n"
            "    sp={'cagr':R['gspc_cagr']/100,'vol':R['gspc_vol']/100,'mdd':R['gspc_mdd']/100}\n"
            "    ae_tr={'mean_excess':R['excess_tr_mean']/100,'t':R['excess_tr_t'],'n':R['excess_tr_n']}\n"
            "    ae_po={'mean_excess':R['excess_po_mean']/100,'t':R['excess_po_t'],'n':R['excess_po_n']}\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.5))\n"
            "ax.plot(yrs, lv, 'o-', c=AMBER, lw=2.2, label=f\"car index  CAGR {si['cagr']*100:.1f}%\")\n"
            "ax.plot(sx, gspc, '^-', c='#7fb069', lw=1.8, label=f\"S&P price-only  {sp['cagr']*100:.1f}%\")\n"
            "ax.plot(sx, spy, 's-', c=GREEN, lw=2.2, label=f\"S&P total return  {ss['cagr']*100:.1f}%\")\n"
            "ax.set_xlabel('year-end'); ax.set_ylabel('rebased to 100 @ 2005')\n"
            "ax.set_title('H1: the car index under-performs the S&P (both on price and on total return)'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"car index : CAGR {si['cagr']*100:6.2f}%  vol {si['vol']*100:5.1f}%  maxDD {si['mdd']*100:6.1f}%\")\n"
            "print(f\"S&P TR    : CAGR {ss['cagr']*100:6.2f}%  vol {ss['vol']*100:5.1f}%  maxDD {ss['mdd']*100:6.1f}%\")\n"
            "print(f\"S&P PO    : CAGR {sp['cagr']*100:6.2f}%  vol {sp['vol']*100:5.1f}%  maxDD {sp['mdd']*100:6.1f}%\")\n"
            "print(f\"annual excess vs TR : mean {ae_tr['mean_excess']*100:+.2f}%/yr  t={ae_tr['t']:+.3f}  (n={ae_tr['n']})\")\n"
            "print(f\"annual excess vs PO : mean {ae_po['mean_excess']*100:+.2f}%/yr  t={ae_po['t']:+.3f}  (n={ae_po['n']})\")"
        ),
        md(
            f"> 💡 In plain words: even against the *price-only* S&P the car index loses "
            f"(**{R['idx_cagr']:.1f}%** vs **{R['gspc_cagr']:.1f}%**), and against the S&P you'd "
            f"actually own — dividends reinvested — it loses by ~3 points a year "
            f"(**{R['spy_cagr_ye']:.1f}%**). The annual excess is **{R['excess_tr_mean']:+.1f}%/yr** vs "
            f"TR (*t* = **{R['excess_tr_t']:+.2f}**, n={R['excess_tr_n']}) and "
            f"**{R['excess_po_mean']:+.1f}%/yr** vs price-only (*t* = **{R['excess_po_t']:+.2f}**) — no "
            "*t* ≥ 2 in the cars' favour on either clock. H₁ rejected."
        ),
        md(
            "### 4b · The smoothness trap — de-biasing the appraisal index\n\n"
            "The car index's headline Sharpe *beats* the S&P — because its volatility is measured off a "
            "smoothed, serially-correlated appraisal series. AR(1) un-smoothing recovers the true risk. "
            "$\\rho$ near the observed **0.64** is the tell; a large vol/Sharpe gap means the "
            "\"risk-adjusted out-performance\" is a measurement artifact."
        ),
        code(
            "ds = st.desmooth_returns(IDX)\n"
            "cats = ['reported', 'de-smoothed', 'S&P 500 (TR)']\n"
            "vols = [ds['vol_obs']*100, ds['vol_desmoothed']*100, R['spy_vol_ye']]\n"
            "shs  = [ds['sharpe_obs'], ds['sharpe_desmoothed'], R['spy_sharpe_ye']]\n"
            "x = np.arange(3); fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.2))\n"
            "ax[0].bar(x, vols, .6, color=[AMBER, RED, GREEN]); ax[0].set_xticks(x); ax[0].set_xticklabels(cats, fontsize=9)\n"
            "ax[0].set_ylabel('annual volatility %'); ax[0].set_title(f\"vol: {ds['vol_obs']*100:.0f}% reported -> {ds['vol_desmoothed']*100:.0f}% true\")\n"
            "ax[1].bar(x, shs, .6, color=[AMBER, RED, GREEN]); ax[1].set_xticks(x); ax[1].set_xticklabels(cats, fontsize=9)\n"
            "ax[1].axhline(0, c='k', lw=.8); ax[1].set_ylabel('Sharpe'); ax[1].set_title(f\"Sharpe: {ds['sharpe_obs']:.2f} -> {ds['sharpe_desmoothed']:.2f}\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"AR(1) rho of annual car returns : {ds['rho']:+.3f}\")\n"
            "print(f\"vol    reported {ds['vol_obs']*100:5.1f}%  ->  de-smoothed {ds['vol_desmoothed']*100:5.1f}%  (S&P {R['spy_vol_ye']:.1f}%)\")\n"
            "print(f\"Sharpe reported {ds['sharpe_obs']:+.2f}  ->  de-smoothed {ds['sharpe_desmoothed']:+.2f}  (S&P {R['spy_sharpe_ye']:+.2f})\")"
        ),
        md(
            f"> 💡 In plain words: the yearly car returns are **{R['rho']:.2f}** auto-correlated — a "
            "textbook smoothed private-asset series. Un-smoothing roughly **doubles** the volatility "
            f"({R['vol_obs']:.0f}% → {R['vol_desmoothed']:.0f}%, now *above* the S&P) and **halves** the "
            f"Sharpe ({R['sharpe_obs']:.2f} → {R['sharpe_desmoothed']:.2f}, now *below* the S&P's "
            f"{R['spy_sharpe_ye']:.2f}). H₂ rejected: the \"low-risk asset class\" is a reporting "
            "illusion — the same bias that flatters real-estate and private-equity indices."
        ),
        md(
            "### 4c · The buyable proxies — alpha, or a barbell of beta?\n\n"
            "Newey-West (6-lag) regression of each proxy's **monthly** return on `SPY`. `REAL` needs "
            "$t_\\alpha \\ge 2$ in the cars' favour — and a *coherent* sign across the two."
        ),
        code(
            "if HAVE_PROXIES:\n"
            "    spy_r = PROX['SPY'].pct_change().dropna()\n"
            "    rows = {}\n"
            "    for t in ['RACE','AML.L']:\n"
            "        s = st.summarize(PROX[t]); nw = st.newey_west_alpha_t(PROX[t].pct_change().dropna(), spy_r, 6)\n"
            "        rows[t] = dict(cagr=s['cagr']*100, sharpe=s['sharpe'], mdd=s['mdd']*100,\n"
            "                       alpha=nw['alpha_ann']*100, beta=nw['beta'], t=nw['t_alpha'])\n"
            "else:\n"
            "    rows = {'RACE':dict(cagr=R['race_cagr'],sharpe=R['race_sharpe'],mdd=R['race_mdd'],alpha=R['race_alpha'],beta=R['race_beta'],t=R['race_t']),\n"
            "            'AML.L':dict(cagr=R['aml_cagr'],sharpe=R['aml_sharpe'],mdd=R['aml_mdd'],alpha=R['aml_alpha'],beta=R['aml_beta'],t=R['aml_t'])}\n"
            "labels=list(rows); alphas=[rows[t]['alpha'] for t in labels]; ts=[rows[t]['t'] for t in labels]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "cols=[GREEN if a>0 else RED for a in alphas]\n"
            "ax.bar(labels, alphas, .5, color=cols)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised alpha vs SPY (%)')\n"
            "for i,t in enumerate(labels): ax.annotate(f\"t={ts[i]:+.2f}\",(i,alphas[i]),ha='center',va='bottom' if alphas[i]>=0 else 'top')\n"
            "ax.set_title('H3: a barbell, not a signal — no coherent, significant alpha'); ax.margins(y=.15)\n"
            "plt.tight_layout(); plt.show()\n"
            "for t in labels: r=rows[t]; print(f\"{t:7s} CAGR {r['cagr']:7.2f}%  Sharpe {r['sharpe']:+.2f}  maxDD {r['mdd']:6.1f}%  alpha {r['alpha']:+7.2f}%/yr  beta {r['beta']:.2f}  NW t {r['t']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: **Ferrari** posts a big alpha (**{R['race_alpha']:+.0f}%/yr**) but it "
            f"does **not** clear significance (*t*={R['race_t']:+.2f}) — and it's a luxury-goods "
            "juggernaut, not a resale-price feed. **Aston Martin** posts a *significantly negative* "
            f"alpha (**{R['aml_alpha']:+.0f}%/yr**, *t*={R['aml_t']:+.2f}, a {R['aml_mdd']:.0f}% "
            "drawdown). Two \"collector-car\" equities, opposite signs: there is no coherent, tradable "
            "alpha here — just idiosyncratic single-stock risk. H₃ rejected."
        ),
        md(
            "### 4d · The ownership tax — net of the auction spread + carry\n\n"
            "A physical car pays a ~22% auction round-trip (buyer's premium + seller's commission, "
            "transport/inspection folded in) over a ~7-year hold, plus ~2.5%/yr carry (specialist "
            "insurance + climate storage + maintenance). Charge it once on the index's gross CAGR (a "
            "*generous* read — an index level is a mid-market appraisal, not net-to-seller)."
        ),
        code(
            "h = st.net_of_carry_cagr(si['cagr'], round_trip_spread=0.22, hold_years=7.0, carry_per_year=0.025)\n"
            "steps = ['gross','after\\nspread','after\\ncarry']\n"
            "running = [h['gross_cagr']*100,\n"
            "           ((1+h['gross_cagr'])*(1+h['spread_drag_annual'])-1)*100,\n"
            "           h['net_cagr']*100]\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "cols=[AMBER, AMBER, ('#b8860b' if running[-1]>0 else RED)]\n"
            "ax.bar(steps, running, .55, color=cols)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(R['spy_cagr_ye'], ls='--', c=GREEN, alpha=.7, label='S&P TR CAGR')\n"
            "for i,v in enumerate(running): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('CAGR %/yr'); ax.set_title('H4: net of real frictions, the car return is cash-like'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"gross {h['gross_cagr']*100:+.2f}%  - spread {h['spread_drag_annual']*100:+.2f}%/yr  - carry {h['carry_per_year']*100:+.2f}%/yr  =  NET {h['net_cagr']*100:+.2f}%/yr  (S&P TR {R['spy_cagr_ye']:.1f}%)\")"
        ),
        md(
            f"> 💡 In plain words: the gross **{R['carry_gross']:.1f}%** — already a loser to the S&P — "
            f"falls to **~{R['carry_net']:.1f}%/yr** once you pay the auction spread and years of carry: "
            "roughly a savings account, at a fraction of the S&P's compounding. And this is *charitable* "
            "— an appraisal level is mid-market, so a real seller nets less still, and the car pays no "
            "dividend while a stock does. H₄ rejected. **MIRAGE**."
        ),
        md(
            "### 4e · Positive control — the engine recovers a planted boom\n\n"
            "A deterministic boom-then-plateau (planted +16%/yr for a decade, then +1%/yr, σ=6%/yr, "
            "seed 713). The harness must recover the up-sign and a finite Sharpe — proving the nulls "
            "above are real, not a broken pipeline."
        ),
        code(
            "syn = data.synthetic_boom()\n"
            "s = st.summarize(syn, periods_per_year=1.0); cr = st.control_recovers(syn, planted_sign=1)\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.1))\n"
            "ax.plot(syn.index, syn.values, 'o-', c=GREY, lw=2)\n"
            "ax.axhline(100, ls=':', c=GREY)\n"
            "ax.set_ylabel('synthetic level'); ax.set_title('Planted boom: engine recovers sign + Sharpe (machinery proof)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"peak {syn.max():.0f} -> end {syn.iloc[-1]:.0f}  recovered CAGR {s['cagr']*100:+.2f}%  Sharpe {s['sharpe']:.2f}  maxDD {s['mdd']*100:.1f}%  sign_ok={cr['sign_ok']}\")"
        ),
        md(
            "> 💡 In plain words: the engine banks the planted signal (recovered CAGR ~+7%, sign "
            "correct). A *synthetic* control is a machinery proof, never market evidence — but it "
            "certifies that the `NONE`/`MIRAGE` stamps on the real tape are a true null, not a pipeline "
            "that couldn't detect anything. (Note the synthetic *also* prints a flattering Sharpe from "
            "its own smoothness — which is exactly beat 4b's point.)"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — car index CAGR {R['idx_cagr']:.1f}% vs S&P {R['spy_cagr_ye']:.1f}% "
            f"(TR) / {R['gspc_cagr']:.1f}% (price-only); annual excess {R['excess_tr_mean']:+.1f}%/yr, "
            f"*t* = {R['excess_tr_t']:+.2f} vs TR (n={R['excess_tr_n']}). The flattering Sharpe "
            f"({R['sharpe_obs']:.2f}) is an appraisal-smoothing artifact — de-smoothed it is "
            f"{R['sharpe_desmoothed']:.2f}, *below* the S&P's {R['spy_sharpe_ye']:.2f}. No robust "
            "*t* ≥ 2 anywhere in the cars' favour.\n"
            f"- **Tradability `MIRAGE`** — gross {R['carry_gross']:+.1f}% → net **{R['carry_net']:+.1f}%/yr** "
            "after a 22% auction round-trip + carry; the only buyable proxies are a barbell (RACE "
            f"α *t*={R['race_t']:+.2f} n.s., AML.L α *t*={R['aml_t']:+.2f} significantly negative, "
            f"{R['aml_mdd']:.0f}% drawdown). No clean, scalable way to own the trade.\n"
            f"- **Cars beat equities? `BUSTED`** — the S&P wins CAGR, de-smoothed Sharpe, drawdown and "
            "net-of-cost. The price-only near-tie evaporates once you count the dividends stocks pay and "
            "the carry cars charge. The winners are pre-2015 buyers of specific chassis: survivorship, "
            "not an asset class."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the capacity & cost reality\n\n"
            "Terminal wealth of \\$10,000 invested end-2005: S&P index fund vs the car index gross "
            "(untradable) vs a car investor paying the real frictions (net CAGR from 4d). Capacity is "
            "the other wall: each car is a bespoke, illiquid, ~20%+-spread lot — there is no scalable "
            "book, and the blue-chip end is a handful of unique chassis."
        ),
        code(
            "start=10_000.0; yrs_h=20\n"
            "paths={'S&P index fund':R['spy_cagr_ye']/100, 'car investor (net)':R['carry_net']/100,\n"
            "       'car index (gross, untradable)':R['idx_cagr']/100}\n"
            "fig, ax = plt.subplots(figsize=(9.2,4.3))\n"
            "labels=list(paths); ends=[start*(1+g)**yrs_h for g in paths.values()]\n"
            "cols=[GREEN, RED, AMBER]\n"
            "ax.bar(labels, ends, .55, color=cols)\n"
            "for i,v in enumerate(ends): ax.annotate(f'${v:,.0f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('value of $10,000 after 20 years'); ax.set_title('Net of cost, the car investor lands near a savings account')\n"
            "plt.tight_layout(); plt.show()\n"
            "for l,g in paths.items(): print(f\"{l:34s} ${start*(1+g)**yrs_h:>10,.0f}  ({g*100:+.1f}%/yr)\")"
        ),
        md(
            "> 💡 In plain words: even the *gross, untradable* index trails the S&P; the *tradable* "
            "version (net of spread + carry) ends near a cash account. And capacity is fatal — a "
            "collector car is a one-off illiquid lot with a ~20%+ round-trip and years of carry, the "
            "antithesis of a scalable strategy. There is no sizing or venue that turns this into an edge."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Swap in the live index.** Replace the hardcoded series with the HAGI Top / Knight "
            "Frank / Hagerty feed and re-run; the *t*-stats sharpen but the sign won't flip — and "
            "**always de-smooth before quoting a Sharpe**.\n"
            "- **Per-chassis dispersion.** The aggregate hides survivorship: test individual references "
            "(250 GTO, Carrera RS 2.7, 300 SL) — the winners are a thin selected tail, the bias points "
            "*for* the claim, so correct for it.\n"
            "- **The collectibles prior.** Dimson–Spaenjers on emotional assets, and the "
            "appraisal-smoothing literature (Geltner; Getmansky–Lo–Makarov): private-asset indices "
            "systematically understate risk and under-perform equities net of carry "
            "([docs/references.md](../docs/references.md)). Cars are not the exception.\n\n"
            "*The reproducible core is offline and deterministic; the car index is a **cited, "
            "approximate proxy** and the equity tickers are **labelled proxies** for the trade. "
            "Methods: [`docs/references.md`](../docs/references.md); frozen numbers: "
            "[`docs/results.md`](../docs/results.md).*"
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
