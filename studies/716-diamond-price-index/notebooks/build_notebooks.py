"""Generate the two narrative notebooks for Study 716 ("short the diamond").

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic: real-tape cells read the cached month-end
yfinance pulls under ../_cache/ for the equity proxies and the hardcoded (cited,
approximate) natural-diamond price index from the package; on a cache miss they fall back
to the frozen headline numbers in ``R`` (mirroring docs/results.md). The synthetic collapse
control runs anywhere.
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


# Frozen headline numbers — mirror of docs/results.md (price index hardcoded/cited/approx;
# equity proxies month-end Adj Close via yfinance, as-of 2025-12-31).
R = dict(
    win="2018 → 2025",
    idx_levels={2018: 100, 2019: 94, 2020: 96, 2021: 118, 2022: 122, 2023: 104, 2024: 92, 2025: 90},
    idx_yoy={2019: -6.0, 2020: 2.1, 2021: 22.9, 2022: 3.4, 2023: -14.8, 2024: -11.5, 2025: -2.2},
    peak_date="2022-02", peak_level=132, peak_trough=-31.8,
    idx_cagr=-1.49, idx_vol=12.4, idx_sharpe=-0.07, idx_mdd=-26.2,
    spy_cagr_ye=17.18, spy_vol_ye=16.9, spy_mdd_ye=-18.2,
    excess_mean=-19.27, excess_t=-2.302, excess_p=0.061, excess_n=7,
    sig_cagr=17.05, sig_vol=66.2, sig_sharpe=0.61, sig_mdd=-77.8,
    sig_alpha=7.32, sig_beta=1.91, sig_t=0.32, sig_p=0.753, sig_n=84,
    luc_cagr=-24.27, luc_vol=49.6, luc_sharpe=-0.32, luc_mdd=-88.8,
    luc_alpha=-26.00, luc_beta=0.82, luc_t=-1.73, luc_p=0.087, luc_n=84,
    luc_total=-85.7,
    spy_cagr=17.18, spy_vol=16.6, spy_sharpe=1.04, spy_mdd=-23.9,
    short_cagr=-22.59, short_vol=49.6, short_sharpe=-0.22, short_mdd=-93.5,
    short_borrow=30, short_nb_cagr=1.91, short_nb_total=13.9,
    haircut_gross=-1.49, haircut_spread=-14.76, haircut_insure=-0.50, haircut_net=-16.45,
    syn_peak=201, syn_end=86, syn_back_cagr=-17.01, syn_sharpe=-0.08, syn_mdd=-58.6,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Diamonds_collapsing%3F: Confirmed](https://img.shields.io/badge/Diamonds_collapsing%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from diamond_price_index import data, strategy as st

HAVE_PROXIES = data.have_proxies()
IDX = data.load_price_index()                        # hardcoded, cited, APPROXIMATE proxy
PROX = data.load_proxies() if HAVE_PROXIES else None
print("equity-proxy cache present:", HAVE_PROXIES,
      "| price-index years:", IDX.index[0].year, "->", IDX.index[-1].year)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Can you get rich shorting diamonds? 💎\n"
            "### \"Lab-grown is killing the natural diamond\" — a real collapse, tested as a trade\n\n"
            + BADGES +
            "You've seen the headlines: lab-grown diamonds are chemically identical, cost a fraction, "
            "and their wholesale price has **collapsed ~80–90%**. Natural-diamond prices peaked in "
            "early 2022 and have been sliding ever since. So the pitch writes itself: *\"the diamond is "
            "dying — **short the diamond**, short the miners, or buy the beaten-down miner for the "
            "rebound.\"* De Beers is cutting prices; the whole complex looks like a falling knife you "
            "could profit from.\n\n"
            "This notebook does something the pitch never does: it asks whether you could actually **make "
            "money** on the collapse. We line the (cited, approximate) natural-diamond price index up "
            "next to the S&P 500, then look at the only things you can really trade — the jeweler "
            "**Signet** and a pure-play **diamond miner** — and charge the frictions (borrow, spreads) "
            "the pitch forgets.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the Newey-West alpha and the borrow "
            "algebra? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice — and a data note.** Real polished-diamond price indices "
            "(Rapaport RAPI, IDEX, Zimnisky) aren't free to pull, so the price line below is a **small, "
            "clearly-cited, approximate** reconstruction of public reporting — a **proxy**, never "
            "presented as the live index. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Are natural-diamond prices really collapsing? | **Yes.** The index peaked near "
            f"**{R['peak_date']}** and round-tripped **{R['peak_trough']:.0f}%** to a lower level by "
            "2025, as lab-grown wholesale prices fell ~80–90%. The collapse is **real**. |\n"
            "| So can you short it? | **Not really.** There's no diamond future. Shorting the miner "
            f"means borrowing a **{R['luc_total']:.0f}%** penny-stock — and after the borrow fee that "
            f"short **loses {abs(R['short_cagr']):.0f}%/yr** anyway. |\n"
            "| Buy the beaten-down miner for a rebound? | **No rebound.** The pure-play miner kept "
            f"falling — down **{R['luc_total']:.0f}%** over the window, a **{R['luc_mdd']:.0f}%** "
            "drawdown. A value trap, not a bounce. |\n"
            "| Is a diamond at least a store of value? | **The opposite.** Buy retail, sell near "
            f"wholesale: the physical stone's return is **~{R['haircut_net']:.0f}%/yr** net of the "
            "resale haircut. A diamond is a spending decision, not an asset. |\n\n"
            "> The diagnosis is dead right — diamonds *are* collapsing. But being right about the "
            "collapse and getting **paid** for it are two very different things. The S&P beat every leg."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Lab-grown diamonds are chemically identical and cost 90% less. They will destroy the "
            "natural-diamond business — prices are already falling and De Beers is slashing. Short the "
            "diamond complex, short the miners, and you'll ride the whole thing down. Or wait for "
            "capitulation and buy a miner for cents on the dollar.\"*\n\n"
            "It's a *steelman-able* claim, and the first half is simply **true**: lab-grown grew from a "
            "novelty to a huge share of the engagement market in a few years, its wholesale price "
            "cratered ~80–90%, and natural polished prices (Rapaport's 1ct RAPI, the IDEX index) fell "
            "roughly **-18% in 2023** and **-11% in 2024** off an early-2022 peak. De Beers cut rough "
            "prices repeatedly. The *diagnosis* is one of the best-supported bearish stories in luxury."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a real, obvious, multi-year price collapse could be turned into a trade, it would be a "
            "money machine — a slow-motion short with a known catalyst. That's exactly why the pitch is "
            "seductive. But *\"prices are falling\"* and *\"you can profit from prices falling\"* are "
            "different statements. A diamond isn't a stock you can short; the listed proxies are a "
            "jeweler and a handful of tiny miners, each with its own story; and shorting anything costs "
            "**borrow**. Between the diagnosis and the P&L sits a wall of microstructure — and that's "
            "where we look."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three honest comparisons, each against the S&P 500 (SPY):\n\n"
            "1. **The price index vs SPY.** Put the (cited, approximate) natural-diamond price index "
            "next to SPY on the same 2018–2025 clock — did diamonds hold value, or fall?\n"
            "2. **The things you can actually trade.** You can't short \"the diamond.\" You *can* trade "
            "**Signet** (the big US diamond jeweler) and **Lucara** (a pure-play miner). Long or short, "
            "do they hand you a *diamond* edge — or just idiosyncratic stock risk?\n"
            "3. **The cost of the trade.** Shorting a penny-stock miner pays a fat borrow fee; owning a "
            "physical diamond pays a brutal retail→resale spread. Charge those, and see what's left.\n\n"
            "**What would make us say \"tradable edge\"?** A listed leg that clears a real *t* ≥ 2 in the "
            "trade's favour **and** survives borrow/spread. Anything less is a correct diagnosis with no "
            "P&L attached."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: the collapse is real.** Here is the (approximate, cited) natural-diamond price "
            "index — the boom into 2022 and the slide since."
        ),
        code(
            "yrs = list(R['idx_levels'].keys())\n"
            "lv = [float(IDX.loc[f'{y}-12-31']) for y in yrs]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "ax.plot(yrs, lv, 'o-', c=AMBER, lw=2.4, label='natural-diamond price index (proxy)')\n"
            "ax.axvline(2022, ls='--', c=RED, alpha=.6)\n"
            "ax.annotate('early-2022\\npeak', (2022, max(lv)), textcoords='offset points',\n"
            "            xytext=(8, -4), color=RED, fontsize=9)\n"
            "ax.axhline(100, ls=':', c=GREY)\n"
            "ax.set_xlabel('year-end'); ax.set_ylabel('index level (base 100 = 2018)')\n"
            "ax.set_title('The natural diamond: a boom, then a lab-grown-driven slide'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('levels:', {y:int(round(v)) for y,v in zip(yrs, lv)})\n"
            "print(f\"down {R['peak_trough']:.0f}% from the early-2022 peak to the 2025 level\")"
        ),
        md(
            f"The slide is real: **{R['peak_trough']:.0f}%** off the early-2022 top, and still drifting "
            "lower in 2025. Lab-grown really is eating the natural stone's lunch. So far the bear pitch "
            "looks completely right — which is exactly why the next three charts matter."
        ),
        md(
            "**But is a diamond a store of value?** Same money, same years — the price index vs the S&P."
        ),
        code(
            "if HAVE_PROXIES:\n"
            "    spy = PROX['SPY']; spy_ye = spy.resample('YE').last()\n"
            "    spy_ye = spy_ye[(spy_ye.index.year>=2018)&(spy_ye.index.year<=2025)]\n"
            "    spy_norm = spy_ye/spy_ye.iloc[0]*100\n"
            "    sx = [d.year for d in spy_norm.index]; sy = list(spy_norm.values)\n"
            "else:\n"
            "    sx, sy = yrs, [100*(1+R['spy_cagr_ye']/100)**(y-2018) for y in yrs]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "ax.plot(yrs, lv, 'o-', c=AMBER, lw=2.2, label=f\"diamonds  ({R['idx_cagr']:+.0f}%/yr)\")\n"
            "ax.plot(sx, sy, 's-', c=GREEN, lw=2.2, label=f\"S&P 500  ({R['spy_cagr_ye']:+.0f}%/yr)\")\n"
            "ax.set_xlabel('year-end'); ax.set_ylabel('$100 invested at end-2018')\n"
            "ax.set_title('Diamonds vs the S&P: not an asset class'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"diamonds CAGR ~{R['idx_cagr']:.1f}%  vs  SPY CAGR ~{R['spy_cagr_ye']:.1f}%  |  \"\n"
            "      f\"diamonds maxDD {R['idx_mdd']:.0f}%  vs  SPY {R['spy_mdd_ye']:.0f}%\")"
        ),
        md(
            f"As an *asset*, the diamond was a disaster: **{R['idx_cagr']:+.1f}%/yr** while the S&P "
            f"compounded **{R['spy_cagr_ye']:+.1f}%/yr**. That confirms the diagnosis — but it also kills "
            "the mirror-image dream (\"diamonds as a hedge / store of value\"). The interesting question "
            "is the *other* side: if diamonds fell, can you get **paid** to be short?"
        ),
        md(
            "**\"Fine — I'll short the miner / buy the miner.\"** Meet the two things you can actually "
            "trade: Signet (the jeweler) and Lucara (a pure-play miner). Here's how they did."
        ),
        code(
            "names = ['Signet\\n(jeweler)', 'Lucara\\n(miner)', 'S&P 500']\n"
            "cagrs = [R['sig_cagr'], R['luc_cagr'], R['spy_cagr']]\n"
            "mdds  = [R['sig_mdd'], R['luc_mdd'], R['spy_mdd']]\n"
            "x = np.arange(3); fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "ax.bar(x-.2, cagrs, .4, color=[GREEN if c>=0 else RED for c in cagrs], label='CAGR %/yr')\n"
            "ax.bar(x+.2, mdds, .4, color=RED, alpha=.55, label='worst drawdown %')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels(names)\n"
            "ax.set_ylabel('percent'); ax.set_title('The tradable legs: a rising jeweler, a collapsing miner'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"SIG:    CAGR {R['sig_cagr']:+.1f}%/yr but maxDD {R['sig_mdd']:.0f}% (beta {R['sig_beta']:.1f})\")\n"
            "print(f\"LUC.TO: CAGR {R['luc_cagr']:+.1f}%/yr, total {R['luc_total']:.0f}%, maxDD {R['luc_mdd']:.0f}%\")"
        ),
        md(
            f"Notice the split: the **jeweler rose** ({R['sig_cagr']:+.0f}%/yr — jewelers happily sell "
            f"lab-grown too) while the **miner collapsed** ({R['luc_total']:.0f}%). \"Short the diamond\" "
            "would mean shorting the miner — and directionally, you'd have been right. So did the short "
            "print money?"
        ),
        md(
            "**Here's the trap. It costs money to be short.** A tiny, cratering miner is *hard to "
            "borrow* — call it ~30%/yr. Worse, a falling penny-stock is wildly volatile: it had "
            "**+40%-plus up-months** that gut a short. Let's build the short book honestly."
        ),
        code(
            "labels = ['miner\\ntotal', 'short (no\\nborrow)', 'short (net of\\n30%/yr borrow)', 'S&P 500']\n"
            "vals = [R['luc_total'], R['short_nb_total'], R['short_cagr']*7, R['spy_cagr']*7]\n"
            "# express as total-ish % over the ~7y window for a like-for-like eyeball\n"
            "cols = [RED, AMBER, RED, GREEN]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('approx. cumulative %')\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.0f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_title('Being right on direction ≠ making money')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"miner fell {R['luc_total']:.0f}%, yet a short book made only ~{R['short_nb_total']:+.0f}% GROSS\")\n"
            "print(f\"net of 30%/yr borrow the short book LOSES ~{R['short_cagr']:+.0f}%/yr\")"
        ),
        md(
            f"There it is. The miner **fell {R['luc_total']:.0f}%**, yet the short book netted only "
            f"**~{R['short_nb_total']:+.0f}% gross** over seven years — because those +40% up-months "
            "wreck a short (volatility drag). Then the **borrow fee** turns the whole thing **negative "
            f"({R['short_cagr']:+.0f}%/yr)**. You called the collapse perfectly and still lost."
        ),
        md(
            "**And the physical stone?** The \"diamonds are forever / a store of value\" side never "
            "charges the ugliest fact: you buy at **retail** and sell near **wholesale**. Charge that "
            "resale haircut on the index's gross return:"
        ),
        code(
            "labels = ['gross\\nreturn', 'resale\\nhaircut', 'insurance\\n& storage', 'NET to\\nyou']\n"
            "vals = [R['haircut_gross'], R['haircut_spread'], R['haircut_insure'], R['haircut_net']]\n"
            "cols = [AMBER, RED, RED, RED]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('% per year')\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_title('Where a diamond \"investment\" really goes')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"gross {R['haircut_gross']:+.1f}%/yr  ->  NET {R['haircut_net']:+.1f}%/yr after resale spread + carry\")"
        ),
        md(
            f"A gross **{R['haircut_gross']:+.1f}%/yr** that already lost to the S&P becomes "
            f"**~{R['haircut_net']:.0f}%/yr** once you pay to buy-retail-sell-wholesale and insure the "
            "stone. A diamond is a purchase, not a position."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** No listed leg gives you a harvestable *diamond* edge: the jeweler is "
            f"high-beta beta ({R['sig_cagr']:+.0f}%/yr, alpha *t*={R['sig_t']:+.2f}), the miner is "
            f"idiosyncratic single-mine risk (alpha *t*={R['luc_t']:+.2f}), and the short book loses net "
            "of borrow.\n"
            "- **Tradability — Mirage.** No diamond future; the short pays a fat borrow and gets eaten by "
            "volatility drag; the physical stone's resale haircut is brutal. Nothing scales.\n"
            "- **Diamonds collapsing? — Confirmed.** The one thing that *is* true: natural-diamond prices "
            "really did collapse as lab-grown scaled. The diagnosis is right; the trade is not there."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the honest bottom line\n\n"
            "Imagine four people at the end of 2018, each with \\$10,000. One buys an S&P index fund. "
            "One shorts the diamond miner (paying borrow). One buys the miner for the \"rebound.\" One "
            "buys a physical diamond as a \"store of value.\" Where do they land by end-2025?"
        ),
        code(
            "start=10_000.0; yrs_h=7\n"
            "paths={'S&P index fund':R['spy_cagr']/100, 'short the miner (net of borrow)':R['short_cagr']/100,\n"
            "       'buy the miner (rebound)':R['luc_cagr']/100, 'own a physical diamond':R['haircut_net']/100}\n"
            "fig, ax = plt.subplots(figsize=(9.4,4.4))\n"
            "labels=list(paths); ends=[start*(1+g)**yrs_h for g in paths.values()]\n"
            "cols=[GREEN, RED, RED, RED]\n"
            "ax.bar(range(len(labels)), ends, .6, color=cols)\n"
            "ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=12, ha='right', fontsize=8)\n"
            "for i,v in enumerate(ends): ax.annotate(f'${v:,.0f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('value of $10,000 after 7 years'); ax.set_title('Every way to \"trade the diamond collapse\" lost to the index fund')\n"
            "plt.tight_layout(); plt.show()\n"
            "for l,g in paths.items(): print(f\"{l:34s} ${start*(1+g)**yrs_h:>10,.0f}  ({g*100:+.1f}%/yr)\")"
        ),
        md(
            "The index-fund investor roughly **triples** their money doing nothing. Every single "
            "diamond-collapse expression — the short, the contrarian long, the physical stone — ends up "
            "**worth less than the stake**. The collapse was real and obvious, and it still couldn't be "
            "turned into a dollar. That's the whole lesson."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Pull the real index yourself.** Our price line is a cited *approximation*; Rapaport / "
            "IDEX / Zimnisky publish the live series (paywalled). Swap it in — the shape (and the "
            "verdict) won't move.\n"
            "- **Try a cleaner short.** Anglo American (De Beers' parent) or a basket of miners instead "
            "of one penny-stock — but you'll dilute the \"diamond\" exposure into diversified mining "
            "beta, and the borrow/vol problem doesn't vanish.\n"
            "- **The collapsing-commodity pattern.** Real, obvious, multi-year price declines that still "
            "aren't tradable are everywhere — see [docs/references.md](../docs/references.md).\n\n"
            "*Think there's a clean way to be paid for the diamond collapse — a specific miner, a "
            "specific hedge? Pull its history, charge the borrow and the spread, and show it beats the "
            "index fund. That's the bar.*"
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
            "# Short the diamond — a quantitative teardown 🔬\n"
            "### Price index vs SPY (CAGR / vol / MDD + an annual-excess *t*) · Newey-West proxy "
            "alpha · a borrow-charged short book · the retail→resale haircut · a synthetic collapse "
            "positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We test the "
            "strongest tradable form of \"lab-grown is killing the natural diamond, short it\": (H₁) the "
            "natural-diamond price index falls vs SPY (the *diagnosis*); (H₂) a listed proxy carries a "
            "harvestable alpha — long the survivor or short the loser — vs the market; (H₃) the short "
            "survives borrow and the physical stone survives the resale spread. We find **H₁ confirmed** "
            "(the diagnosis is real), **H₂ rejected** (no significant alpha; \\|*t*\\| < 2), **H₃ "
            "rejected** (the short loses net of borrow; the stone loses net of spread).\n\n"
            "> ⚠️ **Not investment advice — data provenance.** The price index is **hardcoded, cited, "
            "approximate** (public Rapaport RAPI / IDEX / Zimnisky reporting — a *labelled proxy*, never "
            "the live feed). Equity proxies `SIG`, `LUC.TO`, `SPY` are month-end Adj Close via yfinance "
            "(as-of 2025-12-31). Offline core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md); numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | No tradable leg clears \\|*t*\\| ≥ 2 in the trade's favour: SIG "
            f"alpha *t*=**{R['sig_t']:+.2f}**, LUC.TO alpha *t*=**{R['luc_t']:+.2f}**. The index's "
            f"−19%/yr shortfall (*t*={R['excess_t']:+.2f}) is on a **hardcoded proxy** and measures the "
            f"*diagnosis*, not a harvestable edge. |\n"
            f"| **Tradability** | `MIRAGE` | Short the miner net of 30%/yr borrow: **{R['short_cagr']:+.1f}%/yr** "
            f"(the miner fell {R['luc_total']:.0f}% yet the short lost — vol drag + borrow). Physical stone "
            f"net of resale spread: **{R['haircut_net']:+.1f}%/yr**. No diamond future; nothing scales. |\n"
            f"| **Diamonds collapsing?** | `CONFIRMED` | The price index round-tripped **{R['peak_trough']:.0f}%** "
            f"off its early-2022 peak; lab-grown wholesale fell ~80–90%. The *diagnosis* is real — it just "
            "isn't a trade. |\n\n"
            "> 💡 In plain words: the bear thesis on diamonds is **correct** and yet **untradable**. "
            "Every listed expression is idiosyncratic beta, the short is eaten by borrow and volatility "
            "drag, and the physical stone is a spending decision. Being right about a collapse and "
            "getting paid for it are different problems."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let the natural-diamond price index be $I_t$ and the benchmark $B_t$ (SPY). The bear-trade "
            "claim is a joint hypothesis:\n\n"
            "- **H₁ (the collapse is real).** Annual $r^I_t < r^B_t$ with the index falling in level — "
            "the diagnosis. *(This one we expect to hold.)*\n"
            "- **H₂ (a listed leg monetises it).** For a tradable proxy $P$, the intercept $\\alpha$ in "
            "$r^P_t = \\alpha + \\beta r^B_t + \\varepsilon_t$ clears a Newey-West \\|*t*\\| > 2 in the "
            "trade's favour (positive for a survivor long, negative for a short).\n"
            "- **H₃ (the trade survives frictions).** A short book net of borrow $b$ keeps a positive "
            "CAGR; **and** the physical stone's net CAGR after the retail→resale round-trip $s$ over "
            "hold $h$ and carry $c$ stays positive.\n\n"
            "The steelman is that H₁ is *unusually* well-supported — this is not a manufactured bubble, "
            "it's a genuine technological displacement. The test is whether that rare, correct macro "
            "call converts into P&L through H₂ and H₃. It does not."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "If H₁–H₃ held, this would be the rarest thing in markets: a slow, obvious, correctly-"
            "diagnosed decline you could simply be short. But each leg is separately falsifiable. H₁ is "
            "a **level check** — did diamonds fall? H₂ asks whether the *only investable expressions* "
            "(listed equities) carry anything beyond market beta — you cannot short \"the index\" or a "
            "polished stone directly. H₃ is the **microstructure tax**: a short pays borrow (punishing on "
            "an illiquid penny-stock miner) and suffers convexity against it on up-moves, while a physical "
            "diamond carries a ~50–70% retail→resale round-trip. The trade needs all three; H₁ alone is a "
            "diagnosis, not a position."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Price index (proxy).** A hardcoded, cited, **approximate** annual level (base 100 @ "
            "2018), reconstructed from public Rapaport RAPI / IDEX / Zimnisky / press figures (early-2022 "
            "peak; ~−18%/−11% for 2023/24). *Labelled a proxy* — its path is defensible, its precise "
            "year-end values are not a live feed, and its smoothness **inflates** any *t* built on it.\n"
            "- **Equity proxies.** `SIG`, `LUC.TO`, `SPY` month-end Adj Close (yfinance, cached). "
            "Survivorship is **not** the concern (named tickers, not a screen) — but note the *opposite* "
            "selection risk: Lucara is **one** surviving small miner, so its idiosyncratic path can't be "
            "read as \"diamond beta,\" a caveat that travels with the stamp.\n"
            "- **Signal test.** (i) Paired annual-excess $t$ of $r^I - r^B$ (small-$n$, on the proxy "
            "index — a *diagnosis* statistic, not a tradable one). (ii) **Newey-West (6-lag) HAC** $t$ of "
            "each proxy's alpha vs SPY — the bar for `REAL` is \\|*t*\\| ≥ 2 in the trade's favour.\n"
            "- **Cost (beat 6).** A $1 short book charged **30%/yr borrow** monthly (hard-to-borrow "
            "penny-stock); and the physical stone's retail→resale round-trip (55% over a 5y hold) + carry "
            "**once on NAV**.\n"
            "- **Positive control.** A deterministic peak-and-**collapse** path with a *planted negative* "
            "post-peak drift; the engine must recover the **down**-sign — proof a null on the real tape is "
            "a real null, not a broken harness.\n"
            "- **What would make us say \"tradable\":** a proxy alpha \\|*t*\\| > 2 in the trade's favour "
            "**and** a positive net-of-friction CAGR. We find neither."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The diagnosis — price index vs SPY (H₁)\n\n"
            "Year-end levels rebased to \\$100, both series on one clock. CAGR, vol and max-drawdown in "
            "the print. This leg we *expect* to confirm — diamonds fell."
        ),
        code(
            "yrs = list(R['idx_levels'].keys())\n"
            "lv = np.array([float(IDX.loc[f'{y}-12-31']) for y in yrs])\n"
            "if HAVE_PROXIES:\n"
            "    spy_ye = PROX['SPY'].resample('YE').last()\n"
            "    spy_ye = spy_ye[(spy_ye.index.year>=2018)&(spy_ye.index.year<=2025)]\n"
            "    spy = (spy_ye/spy_ye.iloc[0]*100).values; sx=[d.year for d in spy_ye.index]\n"
            "    si = st.summarize(IDX, periods_per_year=1.0)\n"
            "    ss = st.summarize(spy_ye, periods_per_year=1.0)\n"
            "    ae = st.annual_excess_t(IDX, PROX['SPY'])\n"
            "else:\n"
            "    spy = np.array([100*(1+R['spy_cagr_ye']/100)**(y-2018) for y in yrs]); sx=yrs\n"
            "    si={'cagr':R['idx_cagr']/100,'vol':R['idx_vol']/100,'mdd':R['idx_mdd']/100}\n"
            "    ss={'cagr':R['spy_cagr_ye']/100,'vol':R['spy_vol_ye']/100,'mdd':R['spy_mdd_ye']/100}\n"
            "    ae={'mean_excess':R['excess_mean']/100,'t':R['excess_t'],'n':R['excess_n']}\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.plot(yrs, lv, 'o-', c=AMBER, lw=2.2, label=f\"diamond index  CAGR {si['cagr']*100:+.1f}%, vol {si['vol']*100:.0f}%\")\n"
            "ax.plot(sx, spy, 's-', c=GREEN, lw=2.2, label=f\"SPY  CAGR {ss['cagr']*100:+.1f}%, vol {ss['vol']*100:.0f}%\")\n"
            "ax.set_xlabel('year-end'); ax.set_ylabel('rebased to 100 @ 2018')\n"
            "ax.set_title('H1 (confirmed): the diamond index falls while SPY compounds'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"index: CAGR {si['cagr']*100:+.2f}%  vol {si['vol']*100:.1f}%  maxDD {si['mdd']*100:.1f}%\")\n"
            "print(f\"SPY  : CAGR {ss['cagr']*100:+.2f}%  vol {ss['vol']*100:.1f}%  maxDD {ss['mdd']*100:.1f}%\")\n"
            "print(f\"annual excess (idx-SPY): mean {ae['mean_excess']*100:+.2f}%/yr  t={ae['t']:+.3f}  (n={ae['n']})\")"
        ),
        md(
            f"> 💡 In plain words: the diamond index compounds at **{R['idx_cagr']:+.1f}%** against SPY's "
            f"**{R['spy_cagr_ye']:+.1f}%** — a **{R['excess_mean']:+.1f}%/yr** shortfall, *t* = "
            f"**{R['excess_t']:+.2f}** (n={R['excess_n']}). H₁ **confirmed**: diamonds collapsed. But read "
            "the *t* honestly — it lives on a **hardcoded, smooth proxy** series, so it certifies the "
            "*diagnosis*, not a tradable signal; a real noisy tape would print a smaller number, and no "
            "amount of \"diamonds fell\" tells you how to get **paid**. That is H₂/H₃'s job."
        ),
        md(
            "### 4b · The buyable proxies — is there a *diamond* alpha? (H₂)\n\n"
            "Newey-West (6-lag) regression of each proxy's **monthly** return on SPY. `REAL` needs "
            "\\|$t_\\alpha$\\| ≥ 2 in the trade's favour."
        ),
        code(
            "if HAVE_PROXIES:\n"
            "    spy_r = PROX['SPY'].pct_change().dropna()\n"
            "    rows = {}\n"
            "    for t in ['SIG','LUC.TO']:\n"
            "        s = st.summarize(PROX[t]); nw = st.newey_west_alpha_t(PROX[t].pct_change().dropna(), spy_r, 6)\n"
            "        rows[t] = dict(cagr=s['cagr']*100, sharpe=s['sharpe'], mdd=s['mdd']*100,\n"
            "                       alpha=nw['alpha_ann']*100, beta=nw['beta'], t=nw['t_alpha'])\n"
            "else:\n"
            "    rows = {'SIG':dict(cagr=R['sig_cagr'],sharpe=R['sig_sharpe'],mdd=R['sig_mdd'],alpha=R['sig_alpha'],beta=R['sig_beta'],t=R['sig_t']),\n"
            "            'LUC.TO':dict(cagr=R['luc_cagr'],sharpe=R['luc_sharpe'],mdd=R['luc_mdd'],alpha=R['luc_alpha'],beta=R['luc_beta'],t=R['luc_t'])}\n"
            "labels=list(rows); alphas=[rows[t]['alpha'] for t in labels]; ts=[rows[t]['t'] for t in labels]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "cols=[GREEN if abs(tt)>=2 else GREY for tt in ts]\n"
            "ax.bar(labels, alphas, .5, color=cols)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised alpha vs SPY (%)')\n"
            "for i,t in enumerate(labels): ax.annotate(f\"t={ts[i]:+.2f}\",(i,alphas[i]),ha='center',va='bottom' if alphas[i]>=0 else 'top')\n"
            "ax.set_title('H2 (rejected): no proxy alpha clears |t|>=2 in the trade\\'s favour')\n"
            "plt.tight_layout(); plt.show()\n"
            "for t in labels: r=rows[t]; print(f\"{t:7s} CAGR {r['cagr']:+7.2f}%  Sharpe {r['sharpe']:+.2f}  maxDD {r['mdd']:6.1f}%  alpha {r['alpha']:+.2f}%/yr  beta {r['beta']:.2f}  NW t {r['t']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: Signet is **β≈{R['sig_beta']:.1f}** beta with an insignificant "
            f"**{R['sig_alpha']:+.0f}%/yr** alpha (*t*={R['sig_t']:+.2f}) — a jeweler that sells lab-grown "
            f"too, it *rose* with the market. Lucara has a large *negative* alpha but only *t*="
            f"**{R['luc_t']:+.2f}** — sub-2, and it is **one** mine's idiosyncratic story (a single asset "
            "in Botswana), not a clean \"diamond factor.\" **Neither clears \\|t\\| ≥ 2**: H₂ rejected. "
            "There is no listed instrument that hands you the diamond collapse as alpha."
        ),
        md(
            "### 4c · The short book — being right on direction, net of borrow (H₃, part 1)\n\n"
            "\"Short the diamond\" = short the miner. Build a \\$1 short book on LUC.TO, earn $-r$ each "
            "month, pay a **30%/yr** hard-to-borrow fee. The miner *cratered* — does the short pay?"
        ),
        code(
            "if HAVE_PROXIES:\n"
            "    luc_r = PROX['LUC.TO'].pct_change().dropna()\n"
            "    sb0 = st.short_book_from_returns(luc_r, borrow_annual=0.0)\n"
            "    sb  = st.short_book_from_returns(luc_r, borrow_annual=0.30)\n"
            "    luc_total = PROX['LUC.TO'].iloc[-1]/PROX['LUC.TO'].iloc[0]-1\n"
            "else:\n"
            "    sb0={'cagr':R['short_nb_cagr']/100,'mdd':-0.762}; sb={'cagr':R['short_cagr']/100,'vol':R['short_vol']/100,'mdd':R['short_mdd']/100}\n"
            "    luc_total=R['luc_total']/100\n"
            "bars=['miner\\ntotal return','short book\\n(no borrow)','short book\\n(net 30%/yr)']\n"
            "vals=[luc_total*100, sb0['cagr']*100*7, sb['cagr']*100*7]\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.bar(bars, vals, .55, color=[RED, AMBER, RED])\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('approx cumulative % over ~7y')\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.0f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_title('H3a (rejected): a -86% miner, yet the short loses net of borrow')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"miner total {luc_total*100:+.1f}%  |  short no-borrow CAGR {sb0['cagr']*100:+.2f}%/yr (maxDD {sb0['mdd']*100:.1f}%)\")\n"
            "print(f\"short net of 30%/yr borrow: CAGR {sb['cagr']*100:+.2f}%/yr  vol {sb['vol']*100:.1f}%  maxDD {sb['mdd']*100:.1f}%\")"
        ),
        md(
            f"> 💡 In plain words: the miner fell **{R['luc_total']:.0f}%**, yet the *gross* short book "
            f"compounded at only **{R['short_nb_cagr']:+.1f}%/yr** — the +40%-plus up-months impose a "
            f"vicious **volatility drag** (a short's payoff is concave in the underlying). Charge the "
            f"**30%/yr** borrow and it flips to **{R['short_cagr']:+.1f}%/yr** with a **{R['short_mdd']:.0f}%** "
            "drawdown. A perfectly correct directional call that **lost money**. H₃(a) rejected."
        ),
        md(
            "### 4d · The physical stone — net of the retail→resale haircut (H₃, part 2)\n\n"
            "The \"store of value\" side never charges the round-trip: buy at **retail**, sell near "
            "**wholesale** (~55% over a 5y hold) plus carry. Charge it once on the index's gross CAGR "
            "(a *generous* read — the index level is already a wholesale-ish quote)."
        ),
        code(
            "h = st.net_of_resale_cagr(si['cagr'] if HAVE_PROXIES else R['idx_cagr']/100, round_trip_spread=0.55, hold_years=5.0, insure_per_year=0.005)\n"
            "steps = ['gross','after\\nresale spread','after\\ncarry']\n"
            "running = [h['gross_cagr']*100,\n"
            "           ((1+h['gross_cagr'])*(1+h['spread_drag_annual'])-1)*100,\n"
            "           h['net_cagr']*100]\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.3))\n"
            "ax.bar(steps, running, .55, color=[AMBER, RED, RED])\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(R['spy_cagr_ye'], ls='--', c=GREEN, alpha=.6, label='SPY CAGR')\n"
            "for i,v in enumerate(running): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_ylabel('CAGR %/yr'); ax.set_title('H3b (rejected): net of the resale haircut, deeply negative'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"gross {h['gross_cagr']*100:+.2f}%  - resale {h['spread_drag_annual']*100:+.2f}%/yr  - carry {h['insure_per_year']*100:+.2f}%/yr  =  NET {h['net_cagr']*100:+.2f}%/yr\")"
        ),
        md(
            f"> 💡 In plain words: the gross **{R['haircut_gross']:+.1f}%** — already a loser to SPY — "
            f"becomes **{R['haircut_net']:+.1f}%/yr** once you pay the retail→resale round-trip and carry. "
            "A diamond is a consumption good priced like one: the resale spread alone is larger than a "
            "decade of any plausible appreciation. H₃(b) rejected. **MIRAGE** is the only honest stamp."
        ),
        md(
            "### 4e · Positive control — the engine recovers a planted collapse\n\n"
            "A deterministic peak-and-collapse path (planted modest boom +20% then bust −22% CAGR, "
            "σ=4.5%/mo, seed 716). The harness must recover the **down**-sign in the back half — proving "
            "the nulls above are real, not a pipeline that can't see a trend."
        ),
        code(
            "syn = data.synthetic_collapse()\n"
            "s = st.summarize(syn); cr = st.control_recovers(syn, planted_sign=-1)\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.2))\n"
            "ax.plot(syn.index, syn.values, c=GREY, lw=2)\n"
            "ax.axhline(100, ls=':', c=GREY)\n"
            "ax.set_ylabel('synthetic level'); ax.set_title('Planted collapse: engine recovers the down-sign (machinery proof)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"peak {syn.max():.0f} -> end {syn.iloc[-1]:.0f}  back-half CAGR {cr['back_cagr']*100:+.2f}%  full Sharpe {s['sharpe']:+.2f}  maxDD {s['mdd']*100:.1f}%  sign_ok={cr['sign_ok']}\")"
        ),
        md(
            "> 💡 In plain words: the engine banks the planted **down**-trend (back-half CAGR ≈ −17%, "
            "sign correct). A *synthetic* control is a machinery proof, never market evidence — but it "
            "certifies that the `NONE`/`MIRAGE` stamps on the real tape are a true null, not a pipeline "
            "that couldn't detect a collapse. The collapse is *there*; the tradable edge is not."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — no listed leg clears \\|*t*\\| ≥ 2 in the trade's favour (SIG "
            f"*t*={R['sig_t']:+.2f}, LUC.TO *t*={R['luc_t']:+.2f}); the short book loses net of borrow. "
            f"The index's −19%/yr shortfall (*t*={R['excess_t']:+.2f}) rides on a **hardcoded proxy** and "
            "certifies the *diagnosis*, not a harvestable signal.\n"
            f"- **Tradability `MIRAGE`** — short LUC.TO net of 30%/yr borrow = **{R['short_cagr']:+.1f}%/yr** "
            f"(a {R['luc_total']:.0f}% miner, and the short *still* lost); physical stone net of resale "
            "spread = **{:.1f}%/yr**. No diamond future, illiquid penny-stock borrow, nothing scales.\n".format(R['haircut_net']) +
            "- **Diamonds collapsing? `CONFIRMED`** — natural-diamond prices really did round-trip "
            f"**{R['peak_trough']:.0f}%** off the early-2022 peak as lab-grown scaled. The diagnosis is "
            "the best-supported part of the whole pitch — and the only part that's true."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the capacity & cost reality\n\n"
            "Terminal wealth of \\$10,000 invested end-2018 across every diamond-collapse expression vs "
            "the S&P index fund. Capacity is the other wall: the pure-play short is a **\\$0.20 "
            "penny-stock** — there is no borrow to source at size, and no listed diamond future to "
            "express the macro cleanly."
        ),
        code(
            "start=10_000.0; yrs_h=7\n"
            "paths={'S&P index fund':R['spy_cagr']/100, 'short the miner (net borrow)':R['short_cagr']/100,\n"
            "       'buy the miner (rebound)':R['luc_cagr']/100, 'own a physical diamond':R['haircut_net']/100}\n"
            "fig, ax = plt.subplots(figsize=(9.4,4.4))\n"
            "labels=list(paths); ends=[start*(1+g)**yrs_h for g in paths.values()]\n"
            "cols=[GREEN, RED, RED, RED]\n"
            "ax.bar(range(len(labels)), ends, .6, color=cols)\n"
            "ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=12, ha='right', fontsize=8)\n"
            "for i,v in enumerate(ends): ax.annotate(f'${v:,.0f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('value of $10,000 after 7 years'); ax.set_title('Every diamond-collapse trade lost to the index fund')\n"
            "plt.tight_layout(); plt.show()\n"
            "for l,g in paths.items(): print(f\"{l:32s} ${start*(1+g)**yrs_h:>10,.0f}  ({g*100:+.1f}%/yr)\")"
        ),
        md(
            "> 💡 In plain words: the correct macro call — *diamonds are collapsing* — could not be "
            "monetised on **any** leg. The short loses to borrow + vol drag, the contrarian long is a "
            "value trap, the physical stone is a consumption good. And capacity is fatal: a \\$0.20 "
            "penny-stock has no shortable float at size and there is no diamond future. There is no "
            "venue, sizing, or instrument that turns this diagnosis into an edge."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Swap in the live index.** Replace the hardcoded series with Rapaport RAPI / IDEX / "
            "Zimnisky levels and re-run; the diagnosis sharpens, the tradability verdict won't move.\n"
            "- **A diversified short.** Anglo American (De Beers' parent) or a miner basket lowers the "
            "borrow/vol problem but dilutes the diamond signal into mining beta — test whether *any* "
            "weighting recovers a positive net-of-borrow CAGR (we doubt it).\n"
            "- **The 'right diagnosis, no trade' family.** Correctly-called, obvious, multi-year declines "
            "that still can't be shorted for profit — a recurring desk theme "
            "([docs/references.md](../docs/references.md)).\n\n"
            "*The reproducible core is offline and deterministic; the price index is a **cited, "
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
