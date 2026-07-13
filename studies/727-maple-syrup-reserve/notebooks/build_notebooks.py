"""Generate the two narrative notebooks for Study 727 ("the maple-syrup reserve as a trade").

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic: real-tape cells read the cached month-end
yfinance pulls under ../_cache/ for the equity proxies (RSI.TO, SB=F, ^GSPTSE) and the
hardcoded (cited, approximate) PPAQ maple price from the package; on a cache miss they
fall back to the frozen headline numbers in ``R`` (mirroring docs/results.md). The
synthetic sugaring-season control runs anywhere.
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


# Frozen headline numbers — mirror of docs/results.md. Maple price hardcoded/cited/approx
# (annual CAD/lb); equity proxies month-end Adj Close via yfinance, as-of 2026-06-01.
R = dict(
    maple_win="2008 → 2024",
    maple_levels={2008: 2.60, 2009: 2.65, 2010: 2.72, 2011: 2.85, 2012: 2.92, 2013: 2.94,
                  2014: 2.90, 2015: 2.86, 2016: 2.85, 2017: 2.88, 2018: 2.85, 2019: 2.84,
                  2020: 2.85, 2021: 2.86, 2022: 3.02, 2023: 3.35, 2024: 3.60},
    maple_cagr=2.05, maple_vol=3.5, maple_sharpe=0.61, maple_mdd=-3.4,
    tsx_cagr_ye=4.45, tsx_vol_ye=11.9, tsx_mdd_ye=-11.6,
    excess_mean=-3.00, excess_t=-0.906, excess_p=0.381, excess_n=14,
    rsi_cagr=9.39, rsi_vol=15.7, rsi_sharpe=0.65, rsi_mdd=-24.4,
    rsi_alpha=6.56, rsi_beta=0.50, rsi_t=1.83, rsi_p=0.069, rsi_n=196,
    sb_cagr=-4.52, sb_vol=33.4, sb_sharpe=0.00, sb_mdd=-69.4,
    sb_alpha=1.71, sb_beta=-0.21, sb_t=0.19, sb_p=0.848, sb_n=167,
    tsx_cagr=7.25, tsx_vol=11.9, tsx_sharpe=0.65, tsx_mdd=-22.7,
    month_mean={1: -0.02, 2: 1.46, 3: 1.80, 4: 0.58, 5: 0.23, 6: 0.87,
                7: 2.16, 8: -0.52, 9: 1.17, 10: -1.54, 11: 2.46, 12: 1.52},
    month_thac={1: -0.02, 2: 1.76, 3: 2.58, 4: 1.00, 5: 0.19, 6: 1.60,
                7: 5.12, 8: -0.52, 9: 1.88, 10: -1.64, 11: 1.97, 12: 2.09},
    season_mean=1.28, rest_mean=0.70, season_spread=0.58, season_t=0.81,
    n_season=51, n_rest=145,
    ci_lo=-0.71, ci_hi=1.95, ci_point=0.58,
    mar_thac=2.58, jul_thac=5.12, feb_thac=1.76,
    timer_gross_cagr=9.11, timer_net_cagr=8.78, bh_rsi_cagr=9.38, bh_tsx_cagr=7.24,
    timer_gross_sharpe=0.22, timer_net_sharpe=0.18, bh_rsi_sharpe=0.16,
    world_share=72, reserve_mlb=100, heist_year="2011-2012", heist_cad_m=18.7, heist_tonnes=3000,
    syn_planted=6.0, syn_spread=3.30, syn_t=3.77, syn_sign_ok=1,
    fp_proxies="2f0f50f97e59", fp_maple="123be1d12a41", as_of="2026-06-01",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Just_a_curio%3F: Confirmed](https://img.shields.io/badge/Just_a_curio%3F-Confirmed-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))     # the repo root (quantlab)
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY, SYRUP = "#c0392b", "#dab617", "#2ea44f", "#8b949e", "#d1651a"

from maple_syrup_reserve import data, strategy as st
try:
    from quantlab import repro
    ASOF = repro.DEFAULT_AS_OF
except Exception:
    repro = None; ASOF = "2026-06-01"

HAVE_PROXIES = data.have_proxies()
MAPLE = data.load_maple_price()                      # hardcoded, cited, APPROXIMATE proxy
if HAVE_PROXIES:
    PROX = data.load_proxies()
    if repro is not None:
        PROX = {t: repro.as_of(s.to_frame("x")).iloc[:, 0] for t, s in PROX.items()}
else:
    PROX = None
print("equity-proxy cache present:", HAVE_PROXIES,
      "| maple-price years:", MAPLE.index[0].year, "->", MAPLE.index[-1].year, "| as-of", ASOF)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"

MONTH_ABBR = "['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Can you trade maple syrup? 🍁\n"
            "### Quebec runs a *strategic maple-syrup reserve* — so is there a maple trade, or just a great story?\n\n"
            + BADGES +
            "It sounds made up, but it's real: the Canadian province of Quebec — which makes about "
            f"**{R['world_share']}% of the world's maple syrup** — runs a **Global Strategic Maple Syrup "
            "Reserve**, a warehouse full of barrels that a producers' cartel uses to prop up the price, "
            "exactly the way a central bank hoards gold or OPEC hoards oil. In "
            f"**{R['heist_year']}** thieves siphoned **~C${R['heist_cad_m']}M** of syrup out of it — the "
            "\"Great Canadian Maple Syrup Heist.\" A strategic reserve, a cartel, a heist... surely "
            "there's a *trade* in here somewhere?\n\n"
            "This notebook goes looking. Is the maple price something a holder gets paid to own? Is "
            "there anything you can actually **buy**? And since sap runs on a late-winter schedule, is "
            "there a **sugaring-season seasonal**? Spoiler in the badges above — but the *why* is the fun "
            "part.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the Newey-West alpha and the "
            "block-bootstrap? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice — and a data note.** There is **no** maple exchange or maple "
            "futures; the world price is *administered* by the Quebec cartel. So the price line below is "
            "a **small, clearly-cited, approximate** reconstruction of the negotiated bulk price — a "
            "**proxy**, never a live feed. The one buyable name is **Rogers Sugar** (`RSI.TO`), a sugar "
            "refiner that also bottles maple — a *labelled proxy*, not a barrel of syrup. Every chart is "
            "drawn by the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is the reserve real? | **Completely.** A cartel (the PPAQ) stockpiles millions of pounds "
            f"of syrup to defend the price; the {R['heist_year']} heist really happened. |\n"
            "| Does the maple price reward a holder? | **Barely.** The administered bulk price crept "
            f"**~{R['maple_cagr']:.0f}%/yr** over 2008–2024 — *below* the S&P/TSX's "
            f"**~{R['tsx_cagr_ye']:.0f}%/yr**, with almost no movement at all. It's a set price, not a "
            "market. |\n"
            "| Can you actually buy the trade? | **Not really.** No maple futures exist. The only listed "
            f"name is **Rogers Sugar** — a *sugar* company that happens to bottle maple. It did fine "
            f"(**{R['rsi_cagr']:.0f}%/yr**) but that's a defensive dividend stock, **not the maple price**. |\n"
            "| Is there a sugaring-season seasonal? | **No.** Feb–Apr beat the rest of the year by only "
            f"**{R['season_spread']:.2f}%/mo** (*t* = {R['season_t']:.2f}); the confidence band runs from "
            f"**{R['ci_lo']:.1f}%** to **+{R['ci_hi']:.1f}%** — straight through zero. |\n\n"
            "> The reserve is a wonderful story. The *trade* is a mirage: a set price you can't buy, a "
            "proxy that's really just a sugar stock, and a season that's noise. **A curio, confirmed.**"
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Maple syrup is a strategic commodity. Quebec runs a reserve to control its price like "
            "OPEC controls oil — so maple is a real, ownable soft commodity, with a natural "
            "sugaring-season rhythm you could trade.\"*\n\n"
            "It is a *steelman-able* claim, because every ingredient is true. Quebec produces the large "
            f"majority of the world's syrup (~{R['world_share']}%). Its cartel — **Producteurs et "
            "productrices acéricoles du Québec** — really does run a **strategic reserve** of up to "
            f"~{R['reserve_mlb']} million pounds and really does negotiate the bulk price. And the "
            f"**{R['heist_year']} heist** (≈ C${R['heist_cad_m']}M, ~{R['heist_tonnes']:,} tonnes) proved "
            "the barrels are worth stealing. If *that* isn't a commodity market, what is?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If maple were a genuine tradable commodity, it would be a lovely little uncorrelated corner "
            "of a portfolio — a real-asset store of value with a weather-driven seasonal, the kind of "
            "thing a macro fund would love. But \"there is a cartel and a reserve\" and \"there is a "
            "*trade*\" are very different statements. A managed price can be rock-steady precisely "
            "*because* it's managed — which is the opposite of an opportunity. And a reserve exists to "
            "**remove** volatility, not to create the swings a trader needs. We can check whether any of "
            "the three tradable stories — hold it, buy a proxy, time the season — actually pays."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three honest checks, each against the S&P/TSX Composite (the CAD-home benchmark, since the "
            "one buyable name is Canadian):\n\n"
            "1. **Does holding maple pay?** Put the (cited, approximate) administered bulk price next to "
            "the TSX on the same clock — return, volatility, drawdown.\n"
            "2. **Can you buy it?** You can't buy \"maple.\" You *can* buy **Rogers Sugar** (`RSI.TO`), "
            "the one listed firm with a maple-bottling arm. Does it hand you a *maple* return — or just "
            "a sugar-stock return? (We also check `SB=F`, sugar futures — the nearest sweetener, a pure "
            "placebo.)\n"
            "3. **Is there a season?** Sap runs Feb–Apr. Do those months pay more, and could you time "
            "them net of cost?\n\n"
            "**What would make us say \"a real maple trade\"?** A robust signal (*t* ≥ 2) that maple "
            "beats stocks, *or* real alpha in the proxy, *or* a significant sugaring-season edge. Anything "
            "less is a curio."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: what does an *administered* price look like?** Here is the (approximate, cited) "
            "PPAQ bulk price of maple syrup — the number the cartel negotiates each year."
        ),
        code(
            "yrs = list(R['maple_levels'].keys())\n"
            "lv = [float(MAPLE.loc[f'{y}-12-31']) for y in yrs] if HAVE_PROXIES or True else list(R['maple_levels'].values())\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.plot(yrs, lv, 'o-', c=SYRUP, lw=2.4, label='PPAQ bulk maple price (proxy, CAD/lb)')\n"
            "ax.set_xlabel('year'); ax.set_ylabel('CAD per pound')\n"
            "ax.set_ylim(0, max(lv)*1.25)\n"
            "ax.annotate('set by the cartel,\\ndefended by the reserve', (2016, lv[8]),\n"
            "            textcoords='offset points', xytext=(-6, 40), color=GREY, fontsize=9,\n"
            "            arrowprops=dict(arrowstyle='->', color=GREY))\n"
            "ax.set_title('Maple syrup: an administered price, not a market'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"maple CAGR ~{R['maple_cagr']:.1f}%/yr,  annual vol ~{R['maple_vol']:.1f}% \"\n"
            "      f\"(a real commodity swings 25-40%/yr)\")"
        ),
        md(
            f"That flat, orderly creep — **~{R['maple_cagr']:.0f}%/yr** with **~{R['maple_vol']:.0f}%** "
            "annual volatility — is the whole story. A freely-traded soft commodity (coffee, sugar, "
            "cocoa) swings 25–40% a year. Maple barely moves because a committee *sets* it and a "
            "warehouse full of barrels defends it. Steady is nice for a producer; for a trader it means "
            "**there is nothing to trade**."
        ),
        md(
            "**Now the race: did holding maple beat stocks?** Same years, rebased to 100."
        ),
        code(
            "if HAVE_PROXIES:\n"
            "    tsx_ye = PROX['^GSPTSE'].resample('YE').last()\n"
            "    tsx_ye = tsx_ye[(tsx_ye.index.year>=2008)&(tsx_ye.index.year<=2024)]\n"
            "    tx = [d.year for d in tsx_ye.index]; ty = list((tsx_ye/tsx_ye.iloc[0]*100).values)\n"
            "else:\n"
            "    tx = yrs; ty = [100*(1+R['tsx_cagr_ye']/100)**(y-2008) for y in yrs]\n"
            "mn = [100*v/lv[0] for v in lv]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.plot(yrs, mn, 'o-', c=SYRUP, lw=2.2, label=f\"maple price  ({R['maple_cagr']:.0f}%/yr)\")\n"
            "ax.plot(tx, ty, 's-', c=GREEN, lw=2.2, label=f\"S&P/TSX  ({R['tsx_cagr_ye']:.0f}%/yr)\")\n"
            "ax.set_xlabel('year'); ax.set_ylabel('$100 invested at end-2008')\n"
            "ax.set_title('Holding maple vs holding the index: stocks win'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"maple ~{R['maple_cagr']:.1f}%/yr  vs  TSX ~{R['tsx_cagr_ye']:.1f}%/yr  |  \"\n"
            "      f\"annual excess {R['excess_mean']:+.1f}%/yr (t={R['excess_t']:+.2f}, n={R['excess_n']})\")"
        ),
        md(
            f"The index roughly **doubled** the maple price's growth (**~{R['tsx_cagr_ye']:.0f}%** vs "
            f"**~{R['maple_cagr']:.0f}%/yr**). The administered price didn't just fail to beat stocks — it "
            "trailed them, while being *designed* not to move. A store of value that grows at 2% a year "
            "is a savings account, not a trade."
        ),
        md(
            "**\"Fine — I'll buy the maple *company*.\"** You essentially can't: there's one listed name "
            "with real maple exposure, **Rogers Sugar** (`RSI.TO`), and it's mostly a *sugar* refiner. "
            "Here's how it, sugar futures, and the index stack up."
        ),
        code(
            "if HAVE_PROXIES:\n"
            "    rows = {}\n"
            "    for t in ['RSI.TO','SB=F','^GSPTSE']:\n"
            "        s = st.summarize(PROX[t]); rows[t] = (s['cagr']*100, s['mdd']*100)\n"
            "else:\n"
            "    rows = {'RSI.TO':(R['rsi_cagr'],R['rsi_mdd']),'SB=F':(R['sb_cagr'],R['sb_mdd']),'^GSPTSE':(R['tsx_cagr'],R['tsx_mdd'])}\n"
            "names = ['Rogers Sugar\\n(RSI.TO)', 'sugar futures\\n(SB=F, placebo)', 'S&P/TSX']\n"
            "cagrs = [rows['RSI.TO'][0], rows['SB=F'][0], rows['^GSPTSE'][0]]\n"
            "mdds  = [rows['RSI.TO'][1], rows['SB=F'][1], rows['^GSPTSE'][1]]\n"
            "x = np.arange(3); fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, cagrs, .4, color=[GREEN if c>0 else RED for c in cagrs], label='CAGR %/yr')\n"
            "ax.bar(x+.2, mdds, .4, color=RED, alpha=.5, label='worst drawdown %')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels(names)\n"
            "ax.set_ylabel('percent'); ax.set_title('The \"buyable\" maple trade is a sugar stock (and a sugar-futures disaster)'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"RSI.TO CAGR {rows['RSI.TO'][0]:.1f}% (a defensive sugar refiner, beta {R['rsi_beta']:.2f})\")\n"
            "print(f\"SB=F   CAGR {rows['SB=F'][0]:.1f}% — the nearest tradable sweetener, nothing to do with maple\")"
        ),
        md(
            f"Rogers Sugar did fine (**{R['rsi_cagr']:.0f}%/yr**) — but as a low-beta, dividend-paying "
            "*sugar* refiner, not because of maple: its price and the maple bulk price barely relate. "
            f"And the one actual traded sweetener, sugar futures, **lost {abs(R['sb_cagr']):.0f}%/yr** "
            f"with a **{R['sb_mdd']:.0f}%** drawdown. Neither is a maple trade; one is a decent dividend "
            "stock, the other is a wood-chipper for money."
        ),
        md(
            "**Last: the sugaring season.** Sap runs Feb–Apr. If there's a maple rhythm, those months "
            "should light up. Here are Rogers Sugar's average monthly returns, sugaring months in "
            "maple-orange."
        ),
        code(
            "mm = [R['month_mean'][m] for m in range(1,13)]\n"
            "abbr = " + MONTH_ABBR + "\n"
            "cols = [SYRUP if m in (2,3,4) else GREY for m in range(1,13)]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.3))\n"
            "ax.bar(abbr, mm, color=cols, width=.7)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('avg monthly return %')\n"
            "ax.annotate('July?! nothing to do\\nwith sugaring', (6, mm[6]), textcoords='offset points',\n"
            "            xytext=(-10, 6), color=RED, fontsize=9)\n"
            "ax.set_title('Rogers Sugar by calendar month — the sugaring window (orange) is unremarkable')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"Feb-Apr avg {R['season_mean']:.2f}%/mo vs rest {R['rest_mean']:.2f}%/mo  \"\n"
            "      f\"-> spread {R['season_spread']:+.2f}%/mo (t={R['season_t']:.2f})\")"
        ),
        md(
            f"The sugaring months (orange) are fine but forgettable: Feb–Apr averaged "
            f"**{R['season_mean']:.2f}%/mo** vs **{R['rest_mean']:.2f}%/mo** for the rest — a "
            f"**+{R['season_spread']:.2f}%/mo** gap that's easily noise (*t* = {R['season_t']:.2f}). The "
            f"biggest month is **July** — which has *nothing* to do with sugaring. When your strongest "
            "\"seasonal\" is a month with no story, you're looking at randomness, not a rhythm."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The administered price grew ~{R['maple_cagr']:.0f}%/yr (below stocks, "
            "and by design barely moving); the only buyable proxy is a sugar stock with no maple link; "
            "the sugaring season is statistically flat. Nothing here beats noise in maple's favour.\n"
            "- **Tradability — Mirage.** There is no maple market to trade — no exchange, no futures, a "
            "committee-set price. The 'trade' collapses to owning a dividend sugar stock, and timing its "
            "'season' loses to just holding it.\n"
            "- **Just a curio? — Confirmed.** A brilliant story (cartel, reserve, heist) with no edge "
            "attached. Enjoy it on the pancakes."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the honest bottom line\n\n"
            "Suppose you're determined to \"trade maple\" at the end of 2009 with \\$10,000. You can't "
            "buy syrup, so you buy the closest thing — Rogers Sugar — and you cleverly *time the "
            "sugaring season*. Does the cleverness pay, versus just holding the stock, or the index?"
        ),
        code(
            "start = 10_000.0; yrs_h = 16\n"
            "paths = {'time the sugaring season\\n(RSI.TO, net of cost)': R['timer_net_cagr']/100,\n"
            "         'just hold Rogers Sugar': R['bh_rsi_cagr']/100,\n"
            "         'just hold the index (TSX)': R['bh_tsx_cagr']/100}\n"
            "labels=list(paths); ends=[start*(1+g)**yrs_h for g in paths.values()]\n"
            "fig, ax = plt.subplots(figsize=(9.2,4.3))\n"
            "ax.bar(labels, ends, .55, color=[RED, AMBER, GREEN])\n"
            "for i,v in enumerate(ends): ax.annotate(f'${v:,.0f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('value of $10,000 after 16 years')\n"
            "ax.set_title('Timing the \"maple season\" underperforms just holding the stock')\n"
            "plt.tight_layout(); plt.show()\n"
            "for l,g in paths.items(): print(f\"{l.split(chr(10))[0]:34s} ${start*(1+g)**yrs_h:>10,.0f}  ({g*100:+.1f}%/yr)\")"
        ),
        md(
            "The clever seasonal timer ends up **behind** the do-nothing 'just hold the stock' — the "
            "season it's timing isn't there, so all it does is add trading costs and step out of good "
            "months. And remember: none of this is *maple*. It's a sugar stock. The maple part of the "
            "story bought you exactly nothing. The people who 'made money in maple' are the cartel and "
            "the thieves — not investors."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Pull the real PPAQ series.** Our price line is a cited *approximation* of the negotiated "
            "bulk price; the PPAQ publishes the actual conventions. Swap it in — the *flatness* (and the "
            "verdict) won't move.\n"
            "- **The administered-commodity pattern.** Anything with a marketing board or a strategic "
            "reserve — orange juice, rare-earths, even the US oil reserve — tends to be *managed* toward "
            "calm, which is the opposite of a trade (see [docs/references.md](../docs/references.md)).\n"
            "- **The other soft-commodity curios.** Compare with [Study 307 — Coffee-Seasonality](../../307-coffee-seasonality/) "
            "and [Study 358 — Watch-Index](../../358-watch-index/): a real futures market with a real "
            "(tiny) seasonal, and a collectible with no tradable index — maple sits below both.\n\n"
            "*Think a specific maple-linked name or a tighter sugaring window hides an edge? Pull it, "
            "charge the spread, and show it — then check your best 'season' isn't just July in disguise.*"
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
            "# The maple-syrup reserve as a trade — a quantitative teardown 🔬\n"
            "### Administered price vs TSX (CAGR/vol/MDD + an annual-excess *t*) · Newey-West proxy "
            "alpha (RSI.TO + an SB=F placebo) · per-month HAC *t* & a Feb–Apr Welch test with a "
            "block-bootstrap CI · a costed sugaring-season timer · a synthetic-season positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We test the "
            "strongest tradable form of \"the maple reserve is a commodity trade\": (H₁) the administered "
            "bulk price out-returns the TSX; (H₂) a listed proxy carries alpha; (H₃) there is a tradable "
            "sugaring-season (Feb–Apr) seasonal. We find **H₁ rejected** (it *under*-performs a managed, "
            "near-flat price), **H₂ rejected** (the one proxy's mild edge is defensive sugar beta, |t|<2, "
            "and unrelated to maple), **H₃ rejected** (spread *t*≈0.8, bootstrap CI spans 0).\n\n"
            "> ⚠️ **Not investment advice — data provenance.** The maple price is **hardcoded, cited, "
            "approximate** (PPAQ negotiated bulk price — a *labelled proxy*, never a feed; there is no "
            "maple exchange). Equity proxies `RSI.TO`, `SB=F`, `^GSPTSE` are month-end Adj Close via "
            f"yfinance, as-of {R['as_of']} (fingerprint `{R['fp_proxies']}`). Offline core + synthetic "
            "control are deterministic. Methods in [`docs/references.md`](../docs/references.md); numbers "
            "in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Maple price CAGR **{R['maple_cagr']:.1f}%** vs TSX "
            f"**{R['tsx_cagr_ye']:.1f}%** (2008–2024); annual excess **{R['excess_mean']:+.1f}%/yr**, "
            f"*t* = **{R['excess_t']:+.2f}** (n={R['excess_n']}). Proxy RSI.TO alpha *t* = "
            f"**{R['rsi_t']:+.2f}** (< 2, and it's sugar-beta not maple); SB=F *t* = **{R['sb_t']:+.2f}**. |\n"
            f"| **Tradability** | `MIRAGE` | No maple exchange/futures — an administered price. The "
            f"buyable proxy is a defensive sugar refiner (β={R['rsi_beta']:.2f}); the sugaring-season "
            f"timer nets **{R['timer_net_cagr']:.1f}%/yr**, *below* buy-&-hold RSI.TO "
            f"**{R['bh_rsi_cagr']:.1f}%**. |\n"
            f"| **Just a curio?** | `CONFIRMED` | Feb–Apr minus rest = **{R['season_spread']:+.2f}%/mo** "
            f"(*t* = {R['season_t']:.2f}); block-bootstrap 95% CI **[{R['ci_lo']:+.1f}%, "
            f"{R['ci_hi']:+.1f}%]** spans 0. The loudest month is **July** (HAC *t*={R['jul_thac']:.1f}) — "
            "pure noise. |\n\n"
            "> 💡 In plain words: the price is set by a committee and barely moves; the only thing you can "
            "buy is a sugar stock whose edge is a low-beta premium, not maple; and the 'season' is a "
            "coin-flip. There is no axis on which 'a maple trade' survives."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let the administered bulk price be $M_t$ and the benchmark $B_t$ (`^GSPTSE`). The claim is a "
            "joint hypothesis:\n\n"
            "- **H₁ (holding it pays).** Annual excess $x_t = r^M_t - r^B_t$ has $\\mathbb{E}[x_t] > 0$ "
            "with $t > 2$ — maple rewards a holder better than stocks.\n"
            "- **H₂ (it's buyable with alpha).** For the tradable proxy $P$ (`RSI.TO`), the intercept "
            "$\\alpha$ in $r^P_t = \\alpha + \\beta r^B_t + \\varepsilon_t$ is positive with a "
            "Newey-West *t* > 2 — *and* attributable to maple.\n"
            "- **H₃ (a tradable season).** Sugaring-month (Feb–Apr) mean return exceeds the rest with "
            "$t \\ge 2$ and a bootstrap CI clear of 0, and a costed timer beats buy-&-hold.\n\n"
            "The reserve/cartel is the steelman: a real, price-defending institution around a real "
            "agricultural good. The test is whether that institution creates a *tradable* edge — or "
            "whether it exists precisely to **iron the tradability out**."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "If H₁–H₃ held, maple would be a genuine uncorrelated real-asset sleeve with a weather "
            "seasonal — a macro curiosity worth a small allocation. But each leg is separately "
            "falsifiable. H₁ is a **return race** on a common clock against a price whose *whole purpose* "
            "is stability. H₂ asks whether the only investable expression (a listed sugar refiner) "
            "delivers anything beyond market beta — and, critically, whether any edge is *maple* or just "
            "a **defensive-stock premium** misattributed to maple. H₃ is a **calendar** test with a hard "
            "multiple-comparisons trap: test 12 months and one will look significant by chance. Failing "
            "any leg downgrades the whole thing to a curio."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Maple price (proxy).** A hardcoded, cited, **approximate** annual CAD/lb series "
            "(2008–2024) reconstructed from PPAQ negotiated-price reporting. *Labelled a proxy* — its "
            "flat *shape* is the defensible fact, its exact year values are not a live feed.\n"
            "- **Equity proxies.** `RSI.TO` (Rogers Sugar), `SB=F` (sugar futures, placebo), `^GSPTSE` "
            "month-end Adj Close (yfinance, cached, sliced to the as-of). Survivorship is **not** a "
            "concern (named tickers, not a screen); currency is matched (RSI.TO and the benchmark are "
            "both CAD).\n"
            "- **Signal tests.** (i) Paired annual-excess *t* of $r^M - r^B$ (small-$n$, weak by "
            "construction). (ii) **Newey-West (6-lag) HAC** *t* of the proxy alpha vs TSX — `REAL` needs "
            "*t* ≥ 2 *and* a maple attribution. (iii) Per-month HAC *t*, a Feb–Apr Welch test, and a "
            "**circular block-bootstrap** (12-month blocks) CI on the season-minus-rest spread.\n"
            "- **Cost (beat 6).** The sugaring timer enters once / exits once a year; charge one-way × "
            "NAV (15 bp/leg), flat months earning the benchmark, Sharpe excess-of-benchmark both legs.\n"
            "- **Positive control.** A deterministic monthly world with a *planted* Feb–Apr premium; the "
            "engine must recover its sign with *t* > 2 — proof a null on the real tape is a real null.\n"
            "- **What would make us say \"a maple trade\":** H₁ *t* > 2, **or** a maple-attributable proxy "
            "alpha *t* > 2, **or** a season spread *t* > 2 with a CI clear of 0. We find none."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Does holding maple pay? — administered price vs TSX\n\n"
            "Year-end levels rebased to 100; CAGR, vol, MDD and the paired annual-excess *t* in the "
            "print."
        ),
        code(
            "yrs = list(R['maple_levels'].keys())\n"
            "lv = np.array([float(MAPLE.loc[f'{y}-12-31']) for y in yrs])\n"
            "si = st.summarize(MAPLE, periods_per_year=1.0)\n"
            "if HAVE_PROXIES:\n"
            "    tsx_ye = PROX['^GSPTSE'].resample('YE').last()\n"
            "    tsx_ye = tsx_ye[(tsx_ye.index.year>=2008)&(tsx_ye.index.year<=2024)]\n"
            "    ss = st.summarize(tsx_ye, periods_per_year=1.0)\n"
            "    ae = st.annual_excess_t(MAPLE, PROX['^GSPTSE'])\n"
            "    tx=[d.year for d in tsx_ye.index]; ty=(tsx_ye/tsx_ye.iloc[0]*100).values\n"
            "else:\n"
            "    ss={'cagr':R['tsx_cagr_ye']/100,'vol':R['tsx_vol_ye']/100,'mdd':R['tsx_mdd_ye']/100}\n"
            "    ae={'mean_excess':R['excess_mean']/100,'t':R['excess_t'],'n':R['excess_n']}\n"
            "    tx=yrs; ty=[100*(1+R['tsx_cagr_ye']/100)**(y-2008) for y in yrs]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.plot(yrs, lv/lv[0]*100, 'o-', c=SYRUP, lw=2.2, label=f\"maple  CAGR {si['cagr']*100:.1f}%, vol {si['vol']*100:.0f}%\")\n"
            "ax.plot(tx, ty, 's-', c=GREEN, lw=2.2, label=f\"TSX  CAGR {ss['cagr']*100:.1f}%, vol {ss['vol']*100:.0f}%\")\n"
            "ax.set_xlabel('year'); ax.set_ylabel('rebased to 100 @ 2008')\n"
            "ax.set_title('H1: the administered maple price UNDER-performs the TSX'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"maple: CAGR {si['cagr']*100:.2f}%  vol {si['vol']*100:.1f}%  maxDD {si['mdd']*100:.1f}%\")\n"
            "print(f\"TSX  : CAGR {ss['cagr']*100:.2f}%  vol {ss['vol']*100:.1f}%  maxDD {ss['mdd']*100:.1f}%\")\n"
            "print(f\"annual excess (maple-TSX): mean {ae['mean_excess']*100:+.2f}%/yr  t={ae['t']:+.3f}  (n={ae['n']})\")"
        ),
        md(
            f"> 💡 In plain words: maple compounds at **{R['maple_cagr']:.1f}%** vs the TSX's "
            f"**{R['tsx_cagr_ye']:.1f}%**, at a *fraction* of the volatility (~{R['maple_vol']:.0f}% vs "
            f"~{R['tsx_vol_ye']:.0f}%) — because a committee sets it. The mean annual excess is "
            f"**{R['excess_mean']:+.1f}%/yr**, *t* = **{R['excess_t']:+.2f}** (n={R['excess_n']}): no "
            "evidence maple out-returns stocks, and the point estimate leans negative. H₁ rejected. The "
            "low vol isn't a free lunch; it's the *signature of an administered price*, and the return to "
            "match."
        ),
        md(
            "### 4b · Is the buyable proxy alpha — or misattributed sugar beta?\n\n"
            "Newey-West (6-lag) regression of each proxy's **monthly** return on the TSX. `REAL` needs "
            "$t_\\alpha \\ge 2$ **and** a maple story."
        ),
        code(
            "if HAVE_PROXIES:\n"
            "    bench_r = PROX['^GSPTSE'].pct_change().dropna()\n"
            "    rows = {}\n"
            "    for t in ['RSI.TO','SB=F']:\n"
            "        s = st.summarize(PROX[t]); nw = st.newey_west_alpha_t(PROX[t].pct_change().dropna(), bench_r, 6)\n"
            "        rows[t] = dict(cagr=s['cagr']*100, sharpe=s['sharpe'], mdd=s['mdd']*100,\n"
            "                       alpha=nw['alpha_ann']*100, beta=nw['beta'], t=nw['t_alpha'])\n"
            "else:\n"
            "    rows = {'RSI.TO':dict(cagr=R['rsi_cagr'],sharpe=R['rsi_sharpe'],mdd=R['rsi_mdd'],alpha=R['rsi_alpha'],beta=R['rsi_beta'],t=R['rsi_t']),\n"
            "            'SB=F':dict(cagr=R['sb_cagr'],sharpe=R['sb_sharpe'],mdd=R['sb_mdd'],alpha=R['sb_alpha'],beta=R['sb_beta'],t=R['sb_t'])}\n"
            "labels=list(rows); alphas=[rows[t]['alpha'] for t in labels]; ts=[rows[t]['t'] for t in labels]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "cols=[GREEN if a>0 else RED for a in alphas]\n"
            "ax.bar(labels, alphas, .5, color=cols)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised alpha vs TSX (%)')\n"
            "for i,t in enumerate(labels): ax.annotate(f\"t={ts[i]:+.2f}\",(i,alphas[i]),ha='center',va='bottom' if alphas[i]>=0 else 'top')\n"
            "ax.axhspan(-100,-100,color='none')\n"
            "ax.set_title('H2: proxy alpha does not clear |t|>=2 (and RSI.TO is sugar-beta, not maple)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for t in labels: r=rows[t]; print(f\"{t:7s} CAGR {r['cagr']:6.2f}%  Sharpe {r['sharpe']:.2f}  maxDD {r['mdd']:6.1f}%  alpha {r['alpha']:+.2f}%/yr  beta {r['beta']:+.2f}  NW t {r['t']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: Rogers Sugar shows a **{R['rsi_alpha']:+.1f}%/yr** alpha but *t* = "
            f"**{R['rsi_t']:+.2f}** — **below the *t* ≥ 2 bar**, so `WEAK` at most. And it's the wrong "
            f"*kind* of edge: a **β={R['rsi_beta']:.2f}**, dividend-paying refiner earns a classic "
            "low-beta/defensive premium that has nothing to do with the (flat) maple price — attributing "
            "it to maple is a **misattribution**. The genuine tradable sweetener, sugar futures, has "
            f"*t* = **{R['sb_t']:+.2f}** (nothing). H₂ rejected."
        ),
        md(
            "### 4c · The sugaring season — per-month HAC *t*, Welch test, bootstrap CI\n\n"
            "Per-calendar-month one-sample HAC *t* of RSI.TO monthly returns (the multiple-comparisons "
            "trap made explicit), then the Feb–Apr vs rest Welch test and a circular-block-bootstrap CI "
            "on the spread."
        ),
        code(
            "if HAVE_PROXIES:\n"
            "    rsi_r = PROX['RSI.TO'].pct_change().dropna()\n"
            "    ms = st.month_stats(rsi_r); thac = [ms.loc[m,'tstat_hac'] for m in range(1,13)]\n"
            "    sea = st.season_tstat(rsi_r); ci = st.season_bootstrap_ci(rsi_r, n_boot=5000)\n"
            "else:\n"
            "    thac = [R['month_thac'][m] for m in range(1,13)]\n"
            "    sea = {'season_mean':R['season_mean']/100,'rest_mean':R['rest_mean']/100,'spread':R['season_spread']/100,'tstat':R['season_t']}\n"
            "    ci = {'lo':R['ci_lo']/100,'hi':R['ci_hi']/100,'point':R['ci_point']/100}\n"
            "abbr = " + MONTH_ABBR + "\n"
            "cols = [SYRUP if m in (2,3,4) else GREY for m in range(1,13)]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.3))\n"
            "ax.bar(abbr, thac, color=cols, width=.7)\n"
            "ax.axhline(2, ls='--', c=RED, alpha=.6, label='|t|=2'); ax.axhline(-2, ls='--', c=RED, alpha=.6)\n"
            "ax.axhline(3, ls=':', c='k', alpha=.5, label='|t|=3 (Bonferroni, 12 months)')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('per-month HAC t-stat')\n"
            "ax.set_title('4c: sugaring months (orange) do not stand out; July does (and means nothing)'); ax.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"Feb-Apr mean {sea['season_mean']*100:+.2f}%/mo  rest {sea['rest_mean']*100:+.2f}%/mo  \"\n"
            "      f\"spread {sea['spread']*100:+.2f}%/mo  t={sea['tstat']:+.2f}\")\n"
            "print(f\"block-bootstrap 95% CI on spread: [{ci['lo']*100:+.2f}%, {ci['hi']*100:+.2f}%]  (point {ci['point']*100:+.2f}%)\")"
        ),
        md(
            f"> 💡 In plain words: no sugaring month clears even |t|=2 after you remember you tested "
            f"**twelve** months (Bonferroni pushes the bar to ~|t|=3). March pokes to HAC "
            f"*t*={R['mar_thac']:.2f} — but so, far louder, does **July** (*t*={R['jul_thac']:.2f}), which "
            f"has no sugaring story at all: that's the multiple-comparisons trap in one chart. Pooled, "
            f"Feb–Apr beat the rest by **{R['season_spread']:+.2f}%/mo**, *t*={R['season_t']:.2f}, with a "
            f"bootstrap CI of **[{R['ci_lo']:+.1f}%, {R['ci_hi']:+.1f}%]** straddling 0. H₃ rejected."
        ),
        md(
            "### 4d · The costed sugaring-season timer vs buy-and-hold\n\n"
            "Long RSI.TO in Feb–Apr, hold the TSX otherwise (cash-of-market); one-way 15 bp/leg × NAV, "
            "Sharpe excess-of-benchmark on both legs — like-for-like."
        ),
        code(
            "if HAVE_PROXIES:\n"
            "    rsi_r = PROX['RSI.TO'].pct_change().dropna(); bench_r = PROX['^GSPTSE'].pct_change().dropna()\n"
            "    timer = st.seasonal_timer(rsi_r, bench_r); net = st.apply_costs(timer, 2, 15)\n"
            "    res = {}\n"
            "    for lab, r in [('timer (gross)',timer),('timer (net)',net),('buy&hold RSI',rsi_r),('buy&hold TSX',bench_r)]:\n"
            "        s = st.summary_ret(r, rf=bench_r); res[lab]=(s['cagr']*100, s['sharpe'])\n"
            "else:\n"
            "    res = {'timer (gross)':(R['timer_gross_cagr'],R['timer_gross_sharpe']),'timer (net)':(R['timer_net_cagr'],R['timer_net_sharpe']),\n"
            "           'buy&hold RSI':(R['bh_rsi_cagr'],R['bh_rsi_sharpe']),'buy&hold TSX':(R['bh_tsx_cagr'],float('nan'))}\n"
            "labels=list(res); cg=[res[l][0] for l in labels]\n"
            "fig, ax = plt.subplots(figsize=(9.2,4.3))\n"
            "ax.bar(labels, cg, .55, color=[AMBER, RED, GREEN, GREY])\n"
            "for i,v in enumerate(cg): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('CAGR %/yr'); ax.set_title('4d: timing the season LOSES to just holding the stock')\n"
            "plt.tight_layout(); plt.show()\n"
            "for l in labels: print(f\"{l:16s} CAGR {res[l][0]:+.2f}%  Sharpe(exc-TSX) {res[l][1]:.2f}\")"
        ),
        md(
            f"> 💡 In plain words: the gross timer ({R['timer_gross_cagr']:.1f}%/yr) already trails "
            f"buy-&-hold RSI.TO ({R['bh_rsi_cagr']:.1f}%/yr); **net of cost it's worse still** "
            f"({R['timer_net_cagr']:.1f}%/yr). Stepping out of the market to 'wait for sugaring season' "
            "just forfeits good non-spring months and pays spreads to do it. There is no costed version "
            "of the seasonal that adds value — the honest Tradability stamp is `MIRAGE`."
        ),
        md(
            "### 4e · Positive control — the engine recovers a planted season\n\n"
            f"A deterministic monthly world with a *planted* {R['syn_planted']:.0f}%/yr Feb–Apr premium "
            "(seed 727). The seasonality engine must recover the up-sign with *t* > 2 — proving the nulls "
            "in 4c are real, not a broken pipeline."
        ),
        code(
            "world, truth = data.synthetic_world()\n"
            "sea_s = st.season_tstat(world['ret']); cr = st.control_recovers(world['ret'], planted_sign=1)\n"
            "ms_s = st.month_stats(world['ret']); thac_s = [ms_s.loc[m,'tstat_hac'] for m in range(1,13)]\n"
            "abbr = " + MONTH_ABBR + "\n"
            "cols = [GREEN if m in (2,3,4) else GREY for m in range(1,13)]\n"
            "fig, ax = plt.subplots(figsize=(9.4,4.2))\n"
            "ax.bar(abbr, thac_s, color=cols, width=.7)\n"
            "ax.axhline(2, ls='--', c=RED, alpha=.6); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('per-month HAC t (synthetic)'); ax.set_title('Planted Feb-Apr premium: the engine lights it up (machinery proof)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"planted {truth['spring_premium']*100:.0f}%/yr -> recovered season-minus-rest {cr['spread']*100:+.2f}%/mo  t={sea_s['tstat']:+.2f}  sign_ok={cr['sign_ok']}\")"
        ),
        md(
            f"> 💡 In plain words: when a Feb–Apr premium really is there, the exact same code finds it "
            f"loudly — recovered spread **{R['syn_spread']:+.2f}%/mo**, *t* = **{R['syn_t']:.2f}**, sign "
            "correct. A *synthetic* control is a machinery proof, never market evidence — but it certifies "
            "that the flat sugaring result on RSI.TO is a **true null**, not a pipeline that couldn't "
            "detect a season."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — administered maple price CAGR {R['maple_cagr']:.1f}% vs TSX "
            f"{R['tsx_cagr_ye']:.1f}%; annual excess {R['excess_mean']:+.1f}%/yr, *t* = {R['excess_t']:+.2f} "
            f"(n={R['excess_n']}); proxy alpha *t* = {R['rsi_t']:+.2f} (< 2, and defensive-sugar not maple); "
            f"season spread *t* = {R['season_t']:.2f}. No robust *t* ≥ 2 tied to maple anywhere.\n"
            f"- **Tradability `MIRAGE`** — there is no maple market (no exchange, no futures, an "
            f"administered price). The only buyable expression is a defensive sugar refiner "
            f"(β={R['rsi_beta']:.2f}); the sugaring-season timer nets {R['timer_net_cagr']:.1f}%/yr, "
            f"*below* simply holding it ({R['bh_rsi_cagr']:.1f}%/yr).\n"
            f"- **Just a curio? `CONFIRMED`** — a superb story (cartel, {R['reserve_mlb']}M-lb reserve, "
            f"C${R['heist_cad_m']}M heist) with no tradable edge. The reserve exists to *remove* the "
            "volatility a trader would need."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the capacity & cost reality\n\n"
            "Terminal wealth of \\$10,000 from end-2009: the sugaring-season timer (net) vs just holding "
            "Rogers Sugar vs the index. Capacity is the deeper wall — there is no maple instrument to "
            "size *into*: no futures, no ETF, no forward market; the physical bulk price is set by "
            "negotiation and the reserve, not by anything you can transact."
        ),
        code(
            "start=10_000.0; yrs_h=16\n"
            "paths={'sugaring timer (net)':R['timer_net_cagr']/100,'hold Rogers Sugar':R['bh_rsi_cagr']/100,\n"
            "       'hold TSX':R['bh_tsx_cagr']/100,'hold maple price (untradable)':R['maple_cagr']/100}\n"
            "labels=list(paths); ends=[start*(1+g)**yrs_h for g in paths.values()]\n"
            "fig, ax = plt.subplots(figsize=(9.4,4.3))\n"
            "ax.bar(labels, ends, .6, color=[RED, AMBER, GREEN, GREY])\n"
            "for i,v in enumerate(ends): ax.annotate(f'${v:,.0f}',(i,v),ha='center',va='bottom', fontsize=9)\n"
            "ax.set_ylabel('value of $10,000 after 16 years'); ax.set_title('No sizing turns \"maple\" into an edge')\n"
            "plt.xticks(rotation=12); plt.tight_layout(); plt.show()\n"
            "for l,g in paths.items(): print(f\"{l:32s} ${start*(1+g)**yrs_h:>10,.0f}  ({g*100:+.1f}%/yr)\")"
        ),
        md(
            "> 💡 In plain words: the *clever* seasonal timer is the **worst** of the buyable options; "
            "the 'untradable maple price' sits at the bottom precisely because it's engineered to be "
            "dull. And capacity is fatal in a way costs aren't: even if a season existed, there is no "
            "maple future or ETF to express it — you'd be trading a sugar equity as a costume. There is "
            "no venue or size that makes this a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Swap in the live PPAQ conventions.** Replace the hardcoded annual series with the "
            "producers' published bulk-price schedule; the *t*-stats barely move because the series is "
            "flat *by design*.\n"
            "- **Attribution, done properly.** Decompose RSI.TO into sugar-refining margin, dividend "
            "yield and the (small) maple-bottling segment; the outperformance loads on the defensive/"
            "dividend factors, not on the maple bulk price — the misattribution made quantitative.\n"
            "- **The administered-commodity family.** OJ (the FCOJ board), rare-earths, the US SPR: "
            "managed markets are *engineered* toward low volatility, the opposite of tradability. Compare "
            "the real-futures seasonal in [Study 307 — Coffee-Seasonality](../../307-coffee-seasonality/) "
            "and the no-tradable-index collectible in [Study 358 — Watch-Index](../../358-watch-index/) "
            "([docs/references.md](../docs/references.md)).\n\n"
            "*The reproducible core is offline and deterministic; the maple price is a **cited, "
            "approximate proxy** and the equity tickers are **labelled proxies**. Methods: "
            "[`docs/references.md`](../docs/references.md); frozen numbers + fingerprints: "
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
