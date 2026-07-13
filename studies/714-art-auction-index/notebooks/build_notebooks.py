"""Generate the two narrative notebooks for Study 714 ("contemporary art is an asset class").

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic: real-tape cells read the cached month-end
yfinance pulls under ../_cache/ for the equity proxies and the hardcoded (cited,
approximate) art auction index from the package; on a cache miss they fall back to the
frozen headline numbers in ``R`` (mirroring docs/results.md). The synthetic bubble control
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


# Frozen headline numbers — mirror of docs/results.md (art index hardcoded/cited/approx;
# equity proxies month-end Adj Close via yfinance, as-of 2025-12-31).
R = dict(
    win="2000 → 2025",
    idx_levels={2000: 100, 2001: 103, 2002: 100, 2003: 108, 2004: 128, 2005: 158,
                2006: 205, 2007: 268, 2008: 232, 2009: 150, 2010: 205, 2011: 248,
                2012: 250, 2013: 292, 2014: 330, 2015: 312, 2016: 292, 2017: 330,
                2018: 352, 2019: 345, 2020: 322, 2021: 430, 2022: 478, 2023: 430,
                2024: 385, 2025: 400},
    peak_date="2022-05", peak_level=490, macklowe_musd=922, crash_0709=-44.0, corr_2224=-19.5,
    idx_cagr=5.70, idx_vol=17.1, idx_sharpe=0.41, idx_mdd=-44.0,
    spy_cagr_ye=8.76, spy_vol_ye=17.8, spy_mdd_ye=-36.8,
    excess_mean=-3.27, excess_t=-0.687, excess_p=0.499, excess_n=25,
    mchn_cagr=-6.85, mchn_vol=31.5, mchn_sharpe=-0.07, mchn_mdd=-95.5,
    mchn_alpha=-7.74, mchn_beta=0.60, mchn_t=-1.11, mchn_p=0.268, mchn_n=294,
    ker_cagr=4.84, ker_vol=32.6, ker_sharpe=0.31, ker_mdd=-78.9,
    ker_alpha=-0.15, ker_beta=1.12, ker_t=-0.03, ker_p=0.977, ker_n=311,
    spy_cagr=8.23, spy_vol=15.1, spy_sharpe=0.60, spy_mdd=-50.8,
    carry_gross=5.70, carry_spread=-4.58, carry_insure=-1.00, carry_net=-0.15,
    syn_peak=589, syn_end=175, syn_cagr=5.31, syn_sharpe=0.39, syn_mdd=-70.3,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Art_beats_stocks%3F: Busted](https://img.shields.io/badge/Art_beats_stocks%3F-Busted-8b949e?style=flat-square)\n\n"
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

from art_auction_index import data, strategy as st

HAVE_PROXIES = data.have_proxies()
IDX = data.load_art_index()                          # hardcoded, cited, APPROXIMATE proxy
PROX = data.load_proxies() if HAVE_PROXIES else None
print("equity-proxy cache present:", HAVE_PROXIES,
      "| art-index years:", IDX.index[0].year, "->", IDX.index[-1].year)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does contemporary art beat the S&P? 🖼️\n"
            "### The \"art is an asset class\" pitch, in plain English\n\n"
            + BADGES +
            "You've heard it from every gallerist and wealth manager: *\"forget stocks — a Basquiat, "
            "a Warhol, a hot young painter only goes **up**. Art is an asset class now, it's "
            "uncorrelated, it's inflation-proof, and the Artprice index has beaten the S&P for "
            "decades.\"* For a couple of records-breaking seasons it even looked unstoppable: a single "
            "collection (Macklowe) made **$922 million** at Sotheby's in 2021–22.\n\n"
            "Then the room went quiet — global auction turnover fell **~27%** in 2024. This notebook "
            "lines the art market up next to the S&P 500 — on return, on risk, and on what it actually "
            "**costs to buy and sell a painting at auction** — and asks the only question that matters: "
            "*would you have been richer in an index fund?*\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the Newey-West alpha and the "
            "buyer's-premium algebra? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice — and a data note.** Real art indices (Artprice's "
            "*Contemporary Art Market*, Sotheby's Mei Moses) aren't free to pull, so the art line "
            "below is a **small, clearly-cited, approximate** reconstruction of public reporting — a "
            "**proxy**, never presented as the live index. Every chart is drawn by the code beside it; "
            "house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did art prices really run? | **Yes — in waves.** The (approximate, cited) index ran "
            "from 100 (2000) to ~268 by 2007, then again to a **~478** peak in 2022 (records season — "
            f"the Macklowe collection alone made **\\${R['macklowe_musd']}M**). The mania was real. |\n"
            "| Did it hold? | **Not cleanly.** It **crashed ~44%** in 2008–09 and gave back ~20% again "
            "in 2023–24. Two round-trips, not a straight line. |\n"
            "| Did art beat the S&P? | **No — it lost.** Over 2000–2025 the index compounded at "
            f"**~{R['idx_cagr']:.0f}%/yr** vs **~{R['spy_cagr_ye']:.0f}%/yr** for SPY — with a "
            "*deeper* drawdown. |\n"
            "| Could you at least buy the trade? | **Barely, and badly.** Sotheby's, Christie's and "
            "Phillips are all **private** — you can't buy an auction house. The listed proxies are a "
            f"**{R['mchn_mdd']:.0f}%** disaster (Art Basel's owner) and alpha-free luxury beta; and "
            f"once you pay the ~25% buyer's premium to transact, art's return goes **negative "
            f"({R['carry_net']:+.1f}%/yr)**. |\n\n"
            "> The mania was real. The *asset class* was not. The S&P won every column that matters."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Blue-chip and contemporary art — Basquiat, Warhol, Richter, the hot names at "
            "Sotheby's and Christie's — is a store of value that beats the stock market. The Artprice "
            "index and the Mei Moses index prove art compounds like equities, uncorrelated, "
            "inflation-proof. The rich have always known art is money on the wall.\"*\n\n"
            "It's a *steelman-able* claim: art really did melt up — twice. Between 2003 and 2007 the "
            "secondary market roughly tripled off cheap credit and a new class of global collectors, "
            "and again in 2021–22 post-COVID stimulus and speculation drove record seasons "
            f"(the Macklowe collection made **\\${R['macklowe_musd']}M**). For a moment, in each wave, "
            "\"art beats stocks\" was simply a fact."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it were durably true, it would be a big deal: an asset that beats equities *and* you "
            "get to hang it on the wall, uncorrelated with your stock portfolio, a hedge against "
            "inflation. That's the pitch every gallery, art-fund and private bank leans on. But \"it "
            "ran for a few seasons\" and \"it's an asset class that beats the S&P\" are very different "
            "statements. The first is about a **boom**; the second is a claim about the **long-run, "
            "risk-adjusted, net-of-cost** return. We can check the second directly."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three honest comparisons, each against the S&P 500 (SPY):\n\n"
            "1. **The art index vs SPY.** Put the (cited, approximate) auction price index next to SPY "
            "on the same 2000–2025 clock — return, volatility, worst drawdown.\n"
            "2. **The thing you can actually buy.** You can't buy \"the art index\" — and you can't buy "
            "an auction house either (Sotheby's, Christie's, Phillips all went private). You *can* buy "
            "**MCH Group** (organiser of Art Basel) and **Kering** (whose owner Pinault controls "
            "Christie's). Do they deliver the art trade's return — or just stock-market risk?\n"
            "3. **The cost of transacting.** A painting isn't an ETF: you buy at hammer **plus ~25% "
            "buyer's premium** and sell **minus ~10% seller's commission**, and it hangs insured for "
            "years. Charge that, and see what's left.\n\n"
            "**What would make us say \"asset class\"?** The index beats the S&P on *risk-adjusted, "
            "net-of-cost* return. Anything less is a boom story."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: the two round-trips.** Here is the (approximate, cited) art auction index — the "
            "2000s melt-up, the 2008 crash, the recovery, the 2021–22 records, and the 2023–24 "
            "give-back."
        ),
        code(
            "yrs = list(R['idx_levels'].keys())\n"
            "lv = [float(IDX.loc[f'{y}-12-31']) for y in yrs]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "ax.plot(yrs, lv, 'o-', c=AMBER, lw=2.2, ms=4, label='art auction index (proxy)')\n"
            "ax.axvline(2008, ls='--', c=RED, alpha=.5); ax.axvline(2022, ls='--', c=RED, alpha=.5)\n"
            "ax.annotate('2008-09\\ncrash', (2009, 150), textcoords='offset points', xytext=(6, -2), color=RED, fontsize=9)\n"
            "ax.annotate('2022 records,\\nthen -20%', (2022, 478), textcoords='offset points', xytext=(-64, -6), color=RED, fontsize=9)\n"
            "ax.axhline(100, ls=':', c=GREY)\n"
            "ax.set_xlabel('year-end'); ax.set_ylabel('index level (base 100 = 2000)')\n"
            "ax.set_title('The art \"asset class\": two melt-ups, two round-trips'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('levels:', {y:int(round(v)) for y,v in zip(yrs, lv)})\n"
            "print(f\"crash 2007->2009: {R['crash_0709']:.0f}%  |  correction 2022->2024: {R['corr_2224']:.0f}%\")"
        ),
        md(
            f"It ran hard — but gave it back, twice: **{R['crash_0709']:.0f}%** in the financial crisis, "
            f"another **{R['corr_2224']:.0f}%** in 2023–24. If you bought near a peak — exactly when the "
            "\"art beats stocks\" headlines were loudest — you spent years underwater. That's not how a "
            "store of value behaves; it's how a boom behaves."
        ),
        md(
            "**Now the race: art vs the S&P.** Same money, same 25 years — who's richer at the end?"
        ),
        code(
            "if HAVE_PROXIES:\n"
            "    spy = PROX['SPY']; spy_ye = spy.resample('YE').last()\n"
            "    spy_ye = spy_ye[(spy_ye.index.year>=2000)&(spy_ye.index.year<=2025)]\n"
            "    spy_norm = spy_ye/spy_ye.iloc[0]*100\n"
            "    sx = [d.year for d in spy_norm.index]; sy = list(spy_norm.values)\n"
            "else:\n"
            "    sx, sy = yrs, [100*(1+R['spy_cagr_ye']/100)**(y-2000) for y in yrs]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "ax.plot(yrs, lv, 'o-', c=AMBER, lw=2.0, ms=3, label=f\"art  ({R['idx_cagr']:.0f}%/yr)\")\n"
            "ax.plot(sx, sy, 's-', c=GREEN, lw=2.0, ms=3, label=f\"S&P 500  ({R['spy_cagr_ye']:.0f}%/yr)\")\n"
            "ax.set_xlabel('year-end'); ax.set_ylabel('$100 invested at end-2000')\n"
            "ax.set_title('Art vs the S&P: the index fund pulls away'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"art CAGR ~{R['idx_cagr']:.1f}%  vs  SPY CAGR ~{R['spy_cagr_ye']:.1f}%  |  \"\n"
            "      f\"art maxDD {R['idx_mdd']:.0f}%  vs  SPY {R['spy_mdd_ye']:.0f}%\")"
        ),
        md(
            f"The S&P didn't just win — it **compounded away**: **~{R['spy_cagr_ye']:.0f}%/yr** vs "
            f"**~{R['idx_cagr']:.0f}%/yr**, with a *shallower* worst drop on the same annual clock. More "
            "return, less pain. Even the two gaudy booms aren't enough to save the art line over the "
            "full quarter-century."
        ),
        md(
            "**\"Fine — I'll buy the art *business*.\"** Except you can't buy Sotheby's, Christie's or "
            "Phillips — all three are **private**. The closest listed things are MCH Group (Art Basel) "
            "and Kering (whose owner controls Christie's). Do they hand you the art trade — or a wilder "
            "ride to a worse place?"
        ),
        code(
            "names = ['MCH Group\\n(Art Basel)', 'Kering\\n(Christie\\'s owner)', 'S&P 500']\n"
            "cagrs = [R['mchn_cagr'], R['ker_cagr'], R['spy_cagr']]\n"
            "mdds  = [R['mchn_mdd'], R['ker_mdd'], R['spy_mdd']]\n"
            "x = np.arange(3); fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, cagrs, .4, color=[RED,AMBER,GREEN], label='CAGR %/yr')\n"
            "ax.bar(x+.2, mdds, .4, color=RED, alpha=.5, label='worst drawdown %')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels(names)\n"
            "ax.set_ylabel('percent'); ax.set_title('The buyable proxies: more risk, less reward'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"MCHN.SW: CAGR {R['mchn_cagr']:.1f}% and maxDD {R['mchn_mdd']:.0f}% (beta {R['mchn_beta']:.1f})\")\n"
            "print(f\"KER.PA : CAGR {R['ker_cagr']:.1f}%, high-vol luxury beta, no alpha\")"
        ),
        md(
            f"MCH Group — the one *directly* art-market listed equity — **lost money** and fell "
            f"**{R['mchn_mdd']:.0f}%** peak-to-trough: the art-fair business is a terrible proxy for art "
            "prices going up. Kering underperformed the S&P at twice the volatility. Neither pays you "
            "for the extra white-knuckle risk."
        ),
        md(
            "**The part the pitch never mentions: it costs a fortune to transact art.** You buy at "
            "hammer **plus ~25% buyer's premium**, sell **minus ~10% seller's commission**, and insure "
            "it for years. That's a ~28% round-trip haircut. Charge it against the art index's gross "
            "return:"
        ),
        code(
            "labels = ['gross\\nreturn', 'buyer+seller\\npremium', 'insurance\\n& storage', 'NET to\\nyou']\n"
            "vals = [R['carry_gross'], R['carry_spread'], R['carry_insure'], R['carry_net']]\n"
            "cols = [AMBER, RED, RED, (RED if R['carry_net']<0 else GREEN)]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('% per year')\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_title('Where the art \"return\" goes once you actually transact')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"gross {R['carry_gross']:+.1f}%/yr  ->  NET {R['carry_net']:+.1f}%/yr after premium + carry\")"
        ),
        md(
            f"There it is. A gross **{R['carry_gross']:.1f}%/yr** that already lost to the S&P turns "
            f"**negative ({R['carry_net']:+.1f}%/yr)** the moment you pay the auction house to buy, sell "
            "and hold the painting. A boring T-bill beat it; the index fund laps it."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Art returned ~{R['idx_cagr']:.0f}%/yr vs ~{R['spy_cagr_ye']:.0f}%/yr "
            "for the S&P — it *under*-performed, with a deeper drawdown. No evidence it beats stocks.\n"
            "- **Tradability — Mirage.** Illiquid, high-premium; the gross return goes **negative** "
            "after the ~25% buyer's premium, the auction houses are all private, and the buyable "
            "proxies are alpha-free (one lost 95%).\n"
            "- **Art beats the S&P? — Busted.** Every column — return, risk-adjusted return, drawdown, "
            "net-of-cost — goes to the index fund."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the honest bottom line\n\n"
            "Imagine two people at the end of 2000, each with \\$10,000. One buys an S&P index fund. "
            "The other becomes an art investor — buys at auction, insures, stores, and eventually sells, "
            "paying the real premiums. Where do they land by end-2025?"
        ),
        code(
            "start = 10_000.0; yrs_h = 25\n"
            "spy_end = start*(1+R['spy_cagr_ye']/100)**yrs_h\n"
            "art_end = start*(1+R['carry_net']/100)**yrs_h   # net of premium + carry\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(['S&P index fund', 'art investor\\n(net of premiums)'], [spy_end, art_end],\n"
            "       color=[GREEN, RED], width=.55)\n"
            "for i,v in enumerate([spy_end, art_end]): ax.annotate(f'${v:,.0f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('value of $10,000 after 25 years')\n"
            "ax.set_title('Same $10k, end-2000 -> end-2025'); plt.tight_layout(); plt.show()\n"
            "print(f'S&P index fund: ${spy_end:,.0f}   |   art investor (net): ${art_end:,.0f}')"
        ),
        md(
            "The index-fund investor turns \\$10k into roughly **\\$80k** doing nothing. The art investor "
            "— after buyer's premiums, seller's commissions, insurance and two round-trips — ends up "
            "with **roughly what they started with**. The people who really made money in art bought a "
            "specific name early and sold it at a record: survivors of a boom, not asset-class "
            "investors. You only see the winners on the auction-room screen."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Pull the real index yourself.** Our art line is a cited *approximation*; Artprice and "
            "Sotheby's Mei Moses publish the real repeat-sales series. Swap it in — the shape (and the "
            "verdict) won't move, but you'll have the exact tape.\n"
            "- **The collectibles pattern.** Watches, wine, sneakers, trading cards: every \"passion "
            "asset\" tells the same story — real spikes, brutal carry, equities win net of cost (see "
            "[docs/references.md](../docs/references.md)).\n"
            "- **The sibling study.** [Study 358 — Watches](../../358-watch-index/) is the exact same "
            "shape in a different collectible: a real boom, a round-trip, and a mirage net of the "
            "dealer spread.\n\n"
            "*Think a specific artist (an early Basquiat, a hot MFA graduate) beat the S&P net of every "
            "premium? Pull its repeat-sale history, charge the 25% buyer's premium both ways, and show "
            "it — then check it wasn't just one lucky lot near the bottom.*"
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
            "# Contemporary art as an asset class — a quantitative teardown 🔬\n"
            "### Art index vs SPY (CAGR / vol / MDD + an annual-excess *t*) · Newey-West proxy "
            "alpha · the buyer's-premium + carry haircut on NAV · a synthetic bubble positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We test "
            "the strongest tradable form of \"contemporary art beats the S&P\": (H₁) the secondary-"
            "market auction index out-returns SPY risk-adjusted; (H₂) a buyable proxy carries alpha vs "
            "the market; (H₃) it survives the transaction + carry cost of owning physical art. We find "
            "**H₁ rejected** (it *under*-performs), **H₂ rejected** (no significant alpha), **H₃ "
            "rejected** (negative net of costs).\n\n"
            "> ⚠️ **Not investment advice — data provenance.** The art index is **hardcoded, "
            "cited, approximate** (public Artprice / Sotheby's Mei Moses reporting — a *labelled "
            "proxy*, never the live feed). Equity proxies `MCHN.SW`, `KER.PA`, `SPY` are month-end "
            "Adj Close via yfinance (as-of 2025-12-31). Offline core + synthetic control are "
            "deterministic. Methods in [`docs/references.md`](../docs/references.md); numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Art index CAGR **{R['idx_cagr']:.1f}%** vs SPY "
            f"**{R['spy_cagr_ye']:.1f}%** (2000–2025); mean annual excess **{R['excess_mean']:+.1f}%/yr**, "
            f"*t* = **{R['excess_t']:+.2f}** (n={R['excess_n']}). Proxies: no significant alpha "
            f"(MCHN NW *t*={R['mchn_t']:+.2f}, KER *t*={R['ker_t']:+.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | Gross index CAGR **{R['carry_gross']:+.1f}%** → **NET "
            f"{R['carry_net']:+.1f}%/yr** after a ~25% buyer's premium + 10% seller's commission + 1%/yr "
            f"carry. Auction houses all private; buyable proxy MCHN.SW maxDD **{R['mchn_mdd']:.0f}%**. |\n"
            f"| **Art beats the S&P?** | `BUSTED` | SPY wins CAGR ({R['spy_cagr']:.1f} vs "
            f"{R['idx_cagr']:.1f}), Sharpe ({R['spy_sharpe']:.2f} vs {R['idx_sharpe']:.2f}), maxDD "
            f"({R['spy_mdd_ye']:.0f} vs {R['idx_mdd']:.0f}), and net-of-cost. Every column. |\n\n"
            "> 💡 In plain words: the art market under-performed stocks with a deeper drawdown, the "
            "only thing you can buy is alpha-free (often disastrous) beta, and the auction-house "
            "frictions turn even the gross return negative. There is no axis on which \"art beats the "
            "S&P\" survives."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let the secondary-market index level be $I_t$ and the benchmark $B_t$ (SPY). The claim "
            "is a joint hypothesis:\n\n"
            "- **H₁ (it out-returns).** Annual excess $\\;x_t = r^I_t - r^B_t\\;$ has $\\mathbb{E}[x_t] > 0$ "
            "with $t > 2$ — art beats stocks *risk-adjusted*.\n"
            "- **H₂ (it's buyable with alpha).** For a tradable proxy $P$, the intercept $\\alpha$ in "
            "$r^P_t = \\alpha + \\beta r^B_t + \\varepsilon_t$ is positive with a Newey-West *t* > 2.\n"
            "- **H₃ (it survives premiums).** The net CAGR after a buyer's premium $b$ and seller's "
            "commission $c$ (round-trip multiple $(1-c)/(1+b)$) over hold $h$ and annual carry $k$ "
            "stays positive: $(1+g)\\big((\\tfrac{1-c}{1+b})^{1/h}\\big)(1-k)-1 > 0$.\n\n"
            "The 2000s and 2021–22 booms are the steelman: in each wave H₁ held *in-sample*. The test "
            "is whether it holds over the **full cycle, risk-adjusted, net of cost** — i.e. whether "
            "it's an asset class or a boom."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "If H₁–H₃ held, art would be a genuine diversifier: equity-beating return, hangable, "
            "low-correlation, inflation-proof — the exact pitch. But each leg is separately falsifiable. "
            "H₁ is a **return race** on a common clock. H₂ asks whether the *only investable expression* "
            "(listed equities) delivers anything beyond market beta — because you cannot custody \"the "
            "index,\" and, uniquely here, **you cannot even buy the auction house** (Sotheby's, "
            "Christie's, Phillips are all private). H₃ is the **microstructure tax**: a physical lot "
            "carries a ~25% buyer's premium *and* a ~10% seller's commission with a multi-year holding "
            "period plus insurance carry, frictions an ETF never pays. The asset-class claim needs all "
            "three; failing any one downgrades it to a boom narrative."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Art index (proxy).** A hardcoded, cited, **approximate** annual level (base 100 @ "
            "2000), reconstructed from public Artprice / Sotheby's Mei Moses / press figures (2000s "
            "melt-up, 2008–09 ~−44% crash, 2014 peak, 2021–22 records, 2023–24 ~−20% correction). "
            "*Labelled a proxy* — its path is defensible, its precise year-end values are not a live "
            "feed.\n"
            "- **Equity proxies.** `MCHN.SW` (from its 2001 listing), `KER.PA`, `SPY` month-end Adj "
            "Close (yfinance, cached). Survivorship is **not** a concern here (named tickers, not a "
            "screen); the relevant caveat is the *reverse* — the cleanest expressions (the auction "
            "houses) **left** the tape (Sotheby's taken private 2019), so the listed set is a "
            "second-best, stated.\n"
            "- **Signal test.** (i) Paired annual-excess $t$ of $r^I - r^B$ (n≈25, still weak by "
            "construction). (ii) **Newey-West (6-lag) HAC** $t$ of the proxy alpha vs SPY — the bar "
            "for `REAL` is *t* ≥ 2 in art's favour.\n"
            "- **Cost (beat 6).** Charge the ~25% buyer's premium + ~10% seller's commission (a ~28% "
            "round-trip over a 7y hold) + 1%/yr carry **once on NAV**; net CAGR.\n"
            "- **Positive control.** A deterministic bubble-and-round-trip path with a *planted* boom/"
            "bust drift; the engine must recover the up-sign and a finite Sharpe — proof a null on the "
            "real tape is a real null, not a broken harness.\n"
            "- **What would make us say \"asset class\":** H₁ *t* > 2 **or** a proxy alpha *t* > 2, **and** "
            "a positive net-of-cost CAGR. We find none of these."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The race — art index vs SPY, risk-adjusted\n\n"
            "Year-end levels rebased to \\$100, both series on one clock. CAGR, vol and max-drawdown "
            "in the print."
        ),
        code(
            "yrs = list(R['idx_levels'].keys())\n"
            "lv = np.array([float(IDX.loc[f'{y}-12-31']) for y in yrs])\n"
            "if HAVE_PROXIES:\n"
            "    spy_ye = PROX['SPY'].resample('YE').last()\n"
            "    spy_ye = spy_ye[(spy_ye.index.year>=2000)&(spy_ye.index.year<=2025)]\n"
            "    spy = (spy_ye/spy_ye.iloc[0]*100).values; sx=[d.year for d in spy_ye.index]\n"
            "    si = st.summarize(IDX, periods_per_year=1.0)\n"
            "    ss = st.summarize(spy_ye, periods_per_year=1.0)\n"
            "    ae = st.annual_excess_t(IDX, PROX['SPY'])\n"
            "else:\n"
            "    spy = np.array([100*(1+R['spy_cagr_ye']/100)**(y-2000) for y in yrs]); sx=yrs\n"
            "    si={'cagr':R['idx_cagr']/100,'vol':R['idx_vol']/100,'mdd':R['idx_mdd']/100}\n"
            "    ss={'cagr':R['spy_cagr_ye']/100,'vol':R['spy_vol_ye']/100,'mdd':R['spy_mdd_ye']/100}\n"
            "    ae={'mean_excess':R['excess_mean']/100,'t':R['excess_t'],'n':R['excess_n']}\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "ax.plot(yrs, lv, 'o-', c=AMBER, lw=2.0, ms=3, label=f\"art index  CAGR {si['cagr']*100:.1f}%, vol {si['vol']*100:.0f}%\")\n"
            "ax.plot(sx, spy, 's-', c=GREEN, lw=2.0, ms=3, label=f\"SPY  CAGR {ss['cagr']*100:.1f}%, vol {ss['vol']*100:.0f}%\")\n"
            "ax.set_xlabel('year-end'); ax.set_ylabel('rebased to 100 @ 2000')\n"
            "ax.set_title('H1: the art index UNDER-performs SPY over the cycle'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"art: CAGR {si['cagr']*100:.2f}%  vol {si['vol']*100:.1f}%  maxDD {si['mdd']*100:.1f}%\")\n"
            "print(f\"SPY: CAGR {ss['cagr']*100:.2f}%  vol {ss['vol']*100:.1f}%  maxDD {ss['mdd']*100:.1f}%\")\n"
            "print(f\"annual excess (art-SPY): mean {ae['mean_excess']*100:+.2f}%/yr  t={ae['t']:+.3f}  (n={ae['n']})\")"
        ),
        md(
            f"> 💡 In plain words: the art index compounds at **{R['idx_cagr']:.1f}%** against SPY's "
            f"**{R['spy_cagr_ye']:.1f}%**, with a *deeper* drawdown on the same annual clock — it loses "
            f"on return and on risk. The mean annual excess is **{R['excess_mean']:+.1f}%/yr**, *t* = "
            f"**{R['excess_t']:+.2f}** (n={R['excess_n']}): comfortably inside noise, and the point "
            "estimate is negative. H₁ rejected; the honest stamp is `NONE`, leaning negative."
        ),
        md(
            "### 4b · The buyable proxies — is there alpha, or just beta?\n\n"
            "Newey-West (6-lag) regression of each proxy's **monthly** return on SPY. `REAL` needs "
            "$t_\\alpha \\ge 2$ in art's favour. (And note the cleanest expressions aren't here at all "
            "— the auction houses are private.)"
        ),
        code(
            "if HAVE_PROXIES:\n"
            "    spy_r = PROX['SPY'].pct_change().dropna()\n"
            "    rows = {}\n"
            "    for t in ['MCHN.SW','KER.PA']:\n"
            "        s = st.summarize(PROX[t]); nw = st.newey_west_alpha_t(PROX[t].pct_change().dropna(), spy_r, 6)\n"
            "        rows[t] = dict(cagr=s['cagr']*100, sharpe=s['sharpe'], mdd=s['mdd']*100,\n"
            "                       alpha=nw['alpha_ann']*100, beta=nw['beta'], t=nw['t_alpha'])\n"
            "else:\n"
            "    rows = {'MCHN.SW':dict(cagr=R['mchn_cagr'],sharpe=R['mchn_sharpe'],mdd=R['mchn_mdd'],alpha=R['mchn_alpha'],beta=R['mchn_beta'],t=R['mchn_t']),\n"
            "            'KER.PA':dict(cagr=R['ker_cagr'],sharpe=R['ker_sharpe'],mdd=R['ker_mdd'],alpha=R['ker_alpha'],beta=R['ker_beta'],t=R['ker_t'])}\n"
            "labels=list(rows); alphas=[rows[t]['alpha'] for t in labels]; ts=[rows[t]['t'] for t in labels]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "cols=[GREEN if a>0 else RED for a in alphas]\n"
            "ax.bar(labels, alphas, .5, color=cols)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised alpha vs SPY (%)')\n"
            "for i,t in enumerate(labels): ax.annotate(f\"t={ts[i]:+.2f}\",(i,alphas[i]),ha='center',va='bottom' if alphas[i]>=0 else 'top')\n"
            "ax.set_title('H2: no significant alpha in either buyable proxy (|t|<2)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for t in labels: r=rows[t]; print(f\"{t:8s} CAGR {r['cagr']:6.2f}%  Sharpe {r['sharpe']:+.2f}  maxDD {r['mdd']:6.1f}%  alpha {r['alpha']:+.2f}%/yr  beta {r['beta']:.2f}  NW t {r['t']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: MCH Group — the one *directly* art-market listed equity (Art Basel) "
            f"— **lost {abs(R['mchn_cagr']):.0f}%/yr** and drew down **{R['mchn_mdd']:.0f}%**, with a "
            f"negative (insignificant) alpha, *t*={R['mchn_t']:+.2f}. Kering is high-vol luxury beta "
            f"(*t*={R['ker_t']:+.2f}). **Neither clears *t* ≥ 2** in art's favour: H₂ rejected. The only "
            "investable expressions of the trade are beta you could have bought cheaper as SPY — when "
            "they made money at all."
        ),
        md(
            "### 4c · The microstructure tax — net of the buyer's premium + carry\n\n"
            "A physical lot pays a ~25% buyer's premium on the way in and a ~10% seller's commission on "
            "the way out — a round-trip multiple of $0.90/1.25 = 0.72$ (a ~28% haircut) over a ~7-year "
            "hold, plus ~1%/yr insurance/storage. Charge it once on the index's gross CAGR (a *generous* "
            "read — it ignores that the index level itself is a hammer print, not net-to-seller)."
        ),
        code(
            "h = st.net_of_premium_cagr(si['cagr'], buyers_premium=0.25, sellers_commission=0.10, hold_years=7.0, insure_per_year=0.01)\n"
            "steps = ['gross','after\\npremium','after\\ncarry']\n"
            "running = [h['gross_cagr']*100,\n"
            "           ((1+h['gross_cagr'])*(1+h['spread_drag_annual'])-1)*100,\n"
            "           h['net_cagr']*100]\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.3))\n"
            "cols=[AMBER, AMBER, (RED if running[-1]<0 else GREEN)]\n"
            "ax.bar(steps, running, .55, color=cols)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(R['spy_cagr_ye'], ls='--', c=GREEN, alpha=.6, label='SPY CAGR')\n"
            "for i,v in enumerate(running): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_ylabel('CAGR %/yr'); ax.set_title('H3: net of real frictions, the art return is ~zero/negative'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"gross {h['gross_cagr']*100:+.2f}%  x round-trip {h['round_trip_mult']:.3f} => drag {h['spread_drag_annual']*100:+.2f}%/yr  - carry {h['insure_per_year']*100:+.2f}%/yr  =  NET {h['net_cagr']*100:+.2f}%/yr\")"
        ),
        md(
            f"> 💡 In plain words: the gross **{R['carry_gross']:.1f}%** — already a loser to SPY — goes "
            f"**negative ({R['carry_net']:+.1f}%/yr)** once you pay the premium and carry to actually own "
            "the canvas. And this is *charitable*: a published index tracks hammer prices, so a real "
            "buyer's cost basis is higher and a seller's proceeds lower still. H₃ rejected. **MIRAGE** "
            "is the only honest stamp."
        ),
        md(
            "### 4d · Positive control — the engine recovers a planted bubble\n\n"
            "A deterministic bubble-and-round-trip (planted boom +35% / bust −22% CAGR, σ=4.5%/mo, seed "
            "714). The harness must recover the up-sign and a finite Sharpe — proving the nulls above "
            "are real, not a broken pipeline."
        ),
        code(
            "syn = data.synthetic_bubble()\n"
            "s = st.summarize(syn); cr = st.control_recovers(syn, planted_sign=1)\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.2))\n"
            "ax.plot(syn.index, syn.values, c=GREY, lw=2)\n"
            "ax.axhline(100, ls=':', c=GREY)\n"
            "ax.set_ylabel('synthetic level'); ax.set_title('Planted bubble: engine recovers sign + Sharpe (machinery proof)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"peak {syn.max():.0f} -> end {syn.iloc[-1]:.0f}  recovered CAGR {s['cagr']*100:+.2f}%  Sharpe {s['sharpe']:.2f}  maxDD {s['mdd']*100:.1f}%  sign_ok={cr['sign_ok']}\")"
        ),
        md(
            "> 💡 In plain words: the engine banks the planted signal (recovered CAGR ~+5%, Sharpe "
            "~0.4, sign correct). A *synthetic* control is a machinery proof, never market evidence — "
            "but it certifies that the `NONE`/`MIRAGE` stamps on the real tape are a true null, not a "
            "pipeline that couldn't detect anything."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — art index CAGR {R['idx_cagr']:.1f}% vs SPY {R['spy_cagr_ye']:.1f}%; "
            f"annual excess {R['excess_mean']:+.1f}%/yr, *t* = {R['excess_t']:+.2f} (n={R['excess_n']}); "
            f"proxy alphas insignificant (MCHN *t*={R['mchn_t']:+.2f}, KER *t*={R['ker_t']:+.2f}). No "
            "robust *t* ≥ 2 anywhere in art's favour — and the point estimates lean negative.\n"
            f"- **Tradability `MIRAGE`** — gross {R['carry_gross']:+.1f}% → net **{R['carry_net']:+.1f}%/yr** "
            f"after a ~25% buyer's premium + carry; the auction houses are all private, and the buyable "
            f"proxy MCHN.SW lost money with a {R['mchn_mdd']:.0f}% drawdown. Illiquid, high-premium, "
            "two round-trips.\n"
            f"- **Art beats the S&P? `BUSTED`** — SPY wins CAGR, Sharpe ({R['spy_sharpe']:.2f} vs "
            f"{R['idx_sharpe']:.2f}), drawdown and net-of-cost. The success stories are early buyers of a "
            "specific name who sold at a record: survivorship, not an asset class."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the capacity & cost reality\n\n"
            "Terminal wealth of \\$10,000 invested end-2000, S&P index fund vs an art investor paying "
            "the real premiums (net CAGR from 4c). Capacity is the other wall: each lot is a bespoke, "
            "illiquid, ~28%-round-trip transaction with no fungible unit — there is no scalable book, "
            "and no listed auction house to hold instead."
        ),
        code(
            "start=10_000.0; yrs_h=25\n"
            "paths={'S&P index fund':R['spy_cagr_ye']/100, 'art investor (net)':R['carry_net']/100,\n"
            "       'art index (gross, untradable)':R['idx_cagr']/100}\n"
            "fig, ax = plt.subplots(figsize=(9.2,4.3))\n"
            "labels=list(paths); ends=[start*(1+g)**yrs_h for g in paths.values()]\n"
            "cols=[GREEN, RED, AMBER]\n"
            "ax.bar(labels, ends, .55, color=cols)\n"
            "for i,v in enumerate(ends): ax.annotate(f'${v:,.0f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('value of $10,000 after 25 years'); ax.set_title('Net of cost, the art investor barely breaks even')\n"
            "plt.tight_layout(); plt.show()\n"
            "for l,g in paths.items(): print(f\"{l:34s} ${start*(1+g)**yrs_h:>10,.0f}  ({g*100:+.1f}%/yr)\")"
        ),
        md(
            "> 💡 In plain words: even the *gross, untradable* index trails the S&P badly; the "
            "*tradable* version (net of premium + carry) ends **roughly where it started**. And "
            "capacity is fatal — each lot is a one-off illiquid trade with a ~28% round-trip and no "
            "listed vehicle (the auction houses left the tape), the antithesis of a scalable strategy. "
            "There is no sizing or venue that turns this into an edge."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Swap in the live index.** Replace the hardcoded series with the Artprice Global Index "
            "or the Sotheby's Mei Moses repeat-sales series and re-run; the *t*-stats sharpen but the "
            "sign won't flip.\n"
            "- **Per-artist dispersion.** The aggregate hides survivorship: test individual names "
            "(Basquiat, Richter, a hot young cohort) — the winners are a thin selected tail, the bias "
            "points *for* the claim, so correct for it.\n"
            "- **The collectibles prior.** Mei & Moses (2002), Renneboog & Spaenjers (2013), Dimson & "
            "Spaenjers: art earns lower risk-adjusted returns than equities net of carry across the "
            "board ([docs/references.md](../docs/references.md)). Contemporary art is not the "
            "exception.\n\n"
            "*The reproducible core is offline and deterministic; the art index is a **cited, "
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
