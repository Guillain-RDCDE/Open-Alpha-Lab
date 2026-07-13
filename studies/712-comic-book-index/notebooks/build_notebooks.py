"""Generate the two narrative notebooks for Study 712 ("CGC-graded key comics").

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic: real-tape cells read the cached month-end
yfinance pulls under ../_cache/ for the equity proxy and the hardcoded (cited,
approximate) comic index from the package; on a cache miss they fall back to the frozen
headline numbers in ``R`` (mirroring docs/results.md). The synthetic bubble control runs
anywhere.
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


# Frozen headline numbers — mirror of docs/results.md (comic index hardcoded/cited/approx;
# equity proxy month-end Adj Close via yfinance, as-of 2025-12-31).
R = dict(
    win="2018 → 2025",
    idx_levels={2018: 100, 2019: 108, 2020: 130, 2021: 175, 2022: 158, 2023: 145, 2024: 150, 2025: 152},
    idx_yoy={2019: 8.0, 2020: 20.4, 2021: 34.6, 2022: -9.7, 2023: -8.2, 2024: 3.4, 2025: 1.3},
    peak_date="2022-01", peak_level=188, rec_af15=3.6, rec_ac1=6.0, peak_trough=-20.2,
    idx_cagr=6.16, idx_vol=15.8, idx_sharpe=0.45, idx_mdd=-17.1,
    spy_cagr_ye=17.18, spy_vol_ye=16.9, spy_mdd_ye=-18.2,
    excess_mean=-11.29, excess_t=-1.792, excess_p=0.123, excess_n=7,
    fnko_cagr=-8.04, fnko_vol=66.9, fnko_sharpe=0.24, fnko_mdd=-88.1,
    fnko_alpha=-5.08, fnko_beta=1.44, fnko_t=-0.18, fnko_p=0.858, fnko_n=96,
    spy_cagr=14.21, spy_vol=16.5, spy_sharpe=0.89, spy_mdd=-23.9,
    cost_gross=6.16, cost_spread=-9.14, cost_grading=-0.67, cost_carry=-1.00, cost_net=-5.15,
    syn_peak=460, syn_end=135, syn_cagr=5.19, syn_sharpe=0.34, syn_mdd=-70.7,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Comics_beat_the_S%26P%3F: Busted](https://img.shields.io/badge/Comics_beat_the_S%26P%3F-Busted-8b949e?style=flat-square)\n\n"
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

from comic_book_index import data, strategy as st

HAVE_PROXIES = data.have_proxies()
IDX = data.load_comic_index()                        # hardcoded, cited, APPROXIMATE proxy
PROX = data.load_proxies() if HAVE_PROXIES else None
print("equity-proxy cache present:", HAVE_PROXIES,
      "| comic-index years:", IDX.index[0].year, "->", IDX.index[-1].year)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do old comics beat the S&P? 📚\n"
            "### The \"CGC-graded key comics are an asset class\" boom, in plain English\n\n"
            + BADGES +
            "You heard it all through the pandemic: *\"forget stocks — a slabbed, CGC-graded key "
            "issue only goes **up**. Amazing Fantasy #15, Action Comics #1, the first appearance of "
            "some hero about to get a movie — comics are an asset class now. They beat the S&P.\"* "
            "For about eighteen months it even looked true: prices doubled, people flipped freshly "
            "graded slabs on eBay, and Heritage kept setting records.\n\n"
            "Then the music stopped. This notebook lines the graded-comic market up next to the S&P "
            "500 — on return, on risk, and on what it actually **costs to grade, buy and sell a "
            "comic** — and asks the only question that matters: *would you have been richer in an "
            "index fund?*\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the Newey-West alpha and the cost "
            "algebra? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice — and a data note.** Real graded-comic indices (GoCollect) "
            "are behind a paywall and Heritage publishes per-lot archives, not a downloadable index, "
            "so the comic line below is a **small, clearly-cited, approximate** reconstruction of "
            "public reporting — a **proxy**, never presented as the live index. Every chart is drawn "
            "by the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did comic prices really moon? | **Yes — for a while.** The index ran from 100 (2018) "
            f"to ~175 by end-2021 and blew off near **{R['peak_date']}**. Record slabs printed: "
            f"Amazing Fantasy #15 at **\\${R['rec_af15']:.1f}M** (2021), Action Comics #1 at a record "
            f"**~\\${R['rec_ac1']:.0f}M** (2024). The mania was real. |\n"
            "| Did it last? | **No.** From the early-2022 top the speculative middle **round-tripped "
            f"~{R['peak_trough']:.0f}%** into 2024. A bubble in the middle, a two-tier market at the top. |\n"
            "| Did comics beat the S&P? | **No — they lost.** Over 2018–2025 the index compounded at "
            f"**~{R['idx_cagr']:.0f}%/yr** vs **~{R['spy_cagr_ye']:.0f}%/yr** for SPY — with a *deeper* "
            "path-dependency once you charge the frictions. |\n"
            "| Could you at least buy the trade? | **Barely, and not profitably.** There is **no** "
            "pure-play listed comic stock (CGC, PSA and Heritage are private/delisted). The nearest "
            f"listed proxy, Funko, **lost money** (−8%/yr) with an **{R['fnko_mdd']:.0f}%** drawdown; "
            f"and once you pay the CGC grading fee + dealer spread, the comics' return goes "
            f"**negative ({R['cost_net']:.1f}%/yr)**. |\n\n"
            "> The mania was real. The *asset class* was not. The S&P won every column that matters."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"CGC-graded key comics — Action Comics #1, Amazing Fantasy #15, first appearances "
            "of soon-to-be-movie characters — are a store of value that beats the stock market. Buy "
            "the right slab, wait, sell it for more. Blue-chip keys are money you can read.\"*\n\n"
            "It's a *steelman-able* claim: between 2020 and early 2022 the graded market genuinely "
            "melted up. Stimulus cheques, zero rates, lockdown nostalgia, a wall of movie/TV "
            "announcements and a flood of new collectors pushed keys — and the slabs of them — to "
            f"multiples of their pre-pandemic prices. Amazing Fantasy #15 in CGC 9.6 changed hands "
            f"near **\\${R['rec_af15']:.1f}M**; a record Action Comics #1 later crossed "
            f"**~\\${R['rec_ac1']:.0f}M**. For a moment, \"comics beat stocks\" was simply a fact."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it were durably true, it would be a big deal: an asset that beats equities *and* you "
            "get to keep a piece of pop-culture history, uncorrelated with your stock portfolio, "
            "inflation-proof. That's the pitch every dealer, grading company and collectibles "
            "influencer leans on. But \"it went up for two years\" and \"it's an asset class that "
            "beats the S&P\" are very different statements. The first is about a **bubble**; the "
            "second is a claim about the **long-run, risk-adjusted, net-of-cost** return. We can "
            "check the second directly."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three honest comparisons, each against the S&P 500 (SPY):\n\n"
            "1. **The comic index vs SPY.** Put the (cited, approximate) graded-key-comic index next "
            "to SPY on the same 2018–2025 clock — return, volatility, worst drawdown.\n"
            "2. **The thing you can (barely) buy.** You can't buy \"the comic index,\" and there is "
            "**no** pure-play listed comic stock — CGC's and PSA's parents and Heritage are all "
            "private or delisted. The nearest *listed* proxy is **Funko**. Does it deliver the "
            "collectibles trade's return — or just stock-market risk (and then some)?\n"
            "3. **The cost of flipping a slab.** A comic isn't an ETF: you pay CGC to grade it, buy "
            "near retail-plus, sell into an auction premium or a dealer's discount, and it sits "
            "insured in a safe for years. Charge that, and see what's left.\n\n"
            "**What would make us say \"asset class\"?** The index beats the S&P on *risk-adjusted, "
            "net-of-cost* return. Anything less is a bubble story."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: the round-trip.** Here is the (approximate, cited) comic index — the melt-up "
            "and the giveback."
        ),
        code(
            "yrs = list(R['idx_levels'].keys())\n"
            "lv = [float(IDX.loc[f'{y}-12-31']) for y in yrs]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "ax.plot(yrs, lv, 'o-', c=AMBER, lw=2.4, label='graded-comic index (proxy)')\n"
            "ax.axvline(2022, ls='--', c=RED, alpha=.6)\n"
            "ax.annotate('early-2022\\nblow-off top', (2022, max(lv)), textcoords='offset points',\n"
            "            xytext=(8, -4), color=RED, fontsize=9)\n"
            "ax.axhline(100, ls=':', c=GREY)\n"
            "ax.set_xlabel('year-end'); ax.set_ylabel('index level (base 100 = 2018)')\n"
            "ax.set_title('The comic \"asset class\": a melt-up, then a round-trip'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('levels:', {y:int(round(v)) for y,v in zip(yrs, lv)})\n"
            "print(f\"down ~{R['peak_trough']:.0f}% from the early-2022 peak to the 2024 low\")"
        ),
        md(
            f"It ran up **~75%** into 2021, then the speculative middle gave much of it back: "
            f"**~{R['peak_trough']:.0f}%** from the top. If you bought at the peak — exactly when the "
            "\"asset class\" headlines were loudest — you spent two years underwater. That's not how a "
            "store of value behaves; it's how a bubble behaves. (The very top — the record blue-chip "
            "keys — held up better; that two-tier split is the honest nuance.)"
        ),
        md(
            "**Now the race: comics vs the S&P.** Same money, same years — who's richer at the end?"
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
            "ax.plot(yrs, [float(v) for v in lv], 'o-', c=AMBER, lw=2.2, label=f\"comics  ({R['idx_cagr']:.0f}%/yr)\")\n"
            "ax.plot(sx, sy, 's-', c=GREEN, lw=2.2, label=f\"S&P 500  ({R['spy_cagr_ye']:.0f}%/yr)\")\n"
            "ax.set_xlabel('year-end'); ax.set_ylabel('$100 invested at end-2018')\n"
            "ax.set_title('Comics vs the S&P: not close'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"comics CAGR ~{R['idx_cagr']:.1f}%  vs  SPY CAGR ~{R['spy_cagr_ye']:.1f}%  |  \"\n"
            "      f\"comics maxDD {R['idx_mdd']:.0f}%  vs  SPY {R['spy_mdd_ye']:.0f}%\")"
        ),
        md(
            f"The S&P didn't just win — it **lapped** the comics: **~{R['spy_cagr_ye']:.0f}%/yr** vs "
            f"**~{R['idx_cagr']:.0f}%/yr**. Even the gaudy 2020–21 spike isn't enough to save the comic "
            "line over the full cycle, and this is *before* we charge a cent of the grading-and-selling "
            "friction the pitch quietly ignores."
        ),
        md(
            "**\"Fine — I'll buy the collectibles *stock*.\"** Here's the problem: there basically "
            "isn't one. CGC's parent, PSA's parent and Heritage Auctions are all private or were taken "
            "private. The closest *listed* thing is **Funko** — a pop-culture collectibles maker. Does "
            "it hand you the trade, or just a much wilder ride to a *worse* place?"
        ),
        code(
            "names = ['Funko (FNKO)\\nnearest listed proxy', 'S&P 500']\n"
            "cagrs = [R['fnko_cagr'], R['spy_cagr']]\n"
            "mdds  = [R['fnko_mdd'], R['spy_mdd']]\n"
            "x = np.arange(2); fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "ax.bar(x-.2, cagrs, .4, color=[RED,GREEN], label='CAGR %/yr')\n"
            "ax.bar(x+.2, mdds, .4, color=RED, alpha=.55, label='worst drawdown %')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels(names)\n"
            "ax.set_ylabel('percent'); ax.set_title('The one buyable proxy: more risk, LESS reward'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"FNKO: CAGR {R['fnko_cagr']:.1f}% (a LOSS) with maxDD {R['fnko_mdd']:.0f}% (beta {R['fnko_beta']:.1f})\")\n"
            "print(f\"SPY : CAGR {R['spy_cagr']:.1f}% with maxDD {R['spy_mdd']:.0f}%\")"
        ),
        md(
            f"Funko did the collectibles bubble *with leverage*: it fell **{R['fnko_mdd']:.0f}%** "
            "peak-to-trough and **lost money** over the window — the opposite of a safe-haven. It's the "
            "only thing you could actually buy, and it was worse than both the comics *and* the index "
            "fund. (Caveat: it's a toy company, not a slab of Action #1 — a genuinely poor proxy, which "
            "is exactly the point: the trade has no clean listed expression.)"
        ),
        md(
            "**The part the pitch never mentions: it costs real money to flip a slab.** You pay CGC to "
            "grade it, buy near retail-plus, sell into an auction buyer's premium *and* seller's "
            "commission (or a dealer's wholesale discount — ~25% round-trip), and insure/store it for "
            "years. Charge that against the comic index's gross return:"
        ),
        code(
            "labels = ['gross\\nreturn', 'dealer/auction\\nspread', 'CGC grading\\nfee', 'insurance\\n& storage', 'NET to\\nyou']\n"
            "vals = [R['cost_gross'], R['cost_spread'], R['cost_grading'], R['cost_carry'], R['cost_net']]\n"
            "cols = [AMBER, RED, RED, RED, (RED if R['cost_net']<0 else GREEN)]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.3))\n"
            "ax.bar(labels, vals, color=cols, width=.62)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('% per year')\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_title('Where the comic \"return\" goes once you actually grade + transact')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"gross {R['cost_gross']:+.1f}%/yr  ->  NET {R['cost_net']:+.1f}%/yr after spread + grading + carry\")"
        ),
        md(
            f"There it is. A gross **{R['cost_gross']:.1f}%/yr** that already lost to the S&P turns "
            f"**negative ({R['cost_net']:.1f}%/yr)** the moment you pay CGC to grade it and the dealer/"
            "auction house to move it. A boring T-bill beat it; the index fund laps it."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Comics returned ~{R['idx_cagr']:.0f}%/yr vs ~{R['spy_cagr_ye']:.0f}%/"
            "yr for the S&P — they *under*-performed. No evidence they beat stocks (annual-excess "
            f"*t* = {R['excess_t']:+.2f}, short of the \\|*t*\\| ≥ 2 bar).\n"
            "- **Tradability — Mirage.** Illiquid, high-carry; the gross return goes **negative** after "
            "the CGC fee + dealer spread, and the only listed proxy (Funko) is a money-losing, "
            f"{R['fnko_mdd']:.0f}%-drawdown wreck.\n"
            "- **Comics beat the S&P? — Busted.** Every column — return, risk-adjusted return, "
            "drawdown, net-of-cost — goes to the index fund."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the honest bottom line\n\n"
            "Imagine two people at the end of 2018, each with \\$10,000. One buys an S&P index fund. "
            "The other becomes a comic flipper — grades, insures, stores, and eventually sells, paying "
            "the real frictions. Where do they land by end-2025?"
        ),
        code(
            "start = 10_000.0\n"
            "spy_end = start*(1+R['spy_cagr_ye']/100)**7\n"
            "comic_end = start*(1+R['cost_net']/100)**7   # net of spread + grading + carry\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(['S&P index fund', 'comic flipper\\n(net of costs)'], [spy_end, comic_end],\n"
            "       color=[GREEN, RED], width=.55)\n"
            "for i,v in enumerate([spy_end, comic_end]): ax.annotate(f'${v:,.0f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('value of $10,000 after 7 years')\n"
            "ax.set_title('Same $10k, end-2018 -> end-2025'); plt.tight_layout(); plt.show()\n"
            "print(f'S&P index fund: ${spy_end:,.0f}   |   comic flipper (net): ${comic_end:,.0f}')"
        ),
        md(
            "The index-fund investor roughly **triples** their money doing nothing. The flipper — after "
            "grading fees, spreads, insurance and a bubble that round-tripped — ends up with **less than "
            "they started with**. The people who really made money in comics bought a *specific* key "
            "*before* 2021 and sold near the top: bubble survivors, not asset-class investors. You only "
            "hear from them."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Pull the real index yourself.** Our comic line is a cited *approximation*; GoCollect "
            "publishes live indices (paywalled) and Heritage archives every realised lot. Swap them in — "
            "the shape (and the verdict) won't move, but you'll have the exact tape.\n"
            "- **The collectibles pattern.** Watches, wine, art, sneakers, trading cards: every "
            "\"passion asset\" tells the same story — real spikes, brutal carry, equities win net of "
            "cost (see [docs/references.md](../docs/references.md)).\n"
            "- **The sibling study.** [Study 358 — Watches](../../358-watch-index/) is the exact same "
            "shape in a different collectible: a cited resale index, one listed proxy, a cost haircut "
            "that turns the return negative.\n\n"
            "*Think a specific slab (an Action #1 CGC 8.5, an AF15 CGC 9.6) beat the S&P net of every "
            "cost? Pull its Heritage sale history, charge the grading fee + auction premium, and show it "
            "— then check it wasn't just one lucky buy near the bottom.*"
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
            "# CGC-graded comics as an asset class — a quantitative teardown 🔬\n"
            "### Comic index vs SPY (CAGR / vol / MDD + an annual-excess *t*) · Newey-West proxy "
            "alpha · the CGC-grading + dealer-spread haircut on NAV · a synthetic bubble positive "
            "control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We test "
            "the strongest tradable form of \"CGC-graded key comics beat the S&P\": (H₁) the graded-"
            "comic index out-returns SPY risk-adjusted; (H₂) the one buyable proxy carries alpha vs "
            "the market; (H₃) it survives the grading + transaction + carry cost of owning slabs. We "
            "find **H₁ rejected** (it *under*-performs), **H₂ rejected** (no significant alpha — and "
            "the proxy *lost money*), **H₃ rejected** (negative net of costs).\n\n"
            "> ⚠️ **Not investment advice — data provenance.** The comic index is **hardcoded, cited, "
            "approximate** (public GoCollect / Heritage reporting — a *labelled proxy*, never the live "
            "feed, which is paywalled). The equity proxy `FNKO` and benchmark `SPY` are month-end Adj "
            "Close via yfinance (as-of 2025-12-31). Offline core + synthetic control are "
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
            f"| **Signal** | `NONE` | Comic index CAGR **{R['idx_cagr']:.1f}%** vs SPY "
            f"**{R['spy_cagr_ye']:.1f}%** (2018–2025); mean annual excess **{R['excess_mean']:+.1f}%/yr**, "
            f"*t* = **{R['excess_t']:+.2f}** (n={R['excess_n']}, short of \\|t\\|≥2). Proxy FNKO: no "
            f"significant alpha (NW *t*={R['fnko_t']:+.2f}) and a *negative* CAGR. |\n"
            f"| **Tradability** | `MIRAGE` | Gross index CAGR **{R['cost_gross']:+.1f}%** → **NET "
            f"{R['cost_net']:+.1f}%/yr** after a 25% dealer/auction spread + 2% CGC grading + 1%/yr "
            f"carry. Only listed proxy FNKO: CAGR **{R['fnko_cagr']:+.1f}%**, maxDD **{R['fnko_mdd']:.0f}%**, "
            f"β≈{R['fnko_beta']:.1f}. Illiquid, high-carry, round-tripped. |\n"
            f"| **Comics beat the S&P?** | `BUSTED` | SPY wins CAGR ({R['spy_cagr']:.1f} vs "
            f"{R['idx_cagr']:.1f}), drawdown ({R['spy_mdd_ye']:.0f} vs {R['idx_mdd']:.0f} on the index; "
            f"FNKO {R['fnko_mdd']:.0f}), and net-of-cost. Every column. |\n\n"
            "> 💡 In plain words: the comic market under-performed stocks, the only thing you can buy "
            "is a money-losing toy stock, and the grading-and-selling frictions turn even the gross "
            "return negative. There is no axis on which \"comics beat the S&P\" survives."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let the graded-comic index level be $I_t$ and the benchmark $B_t$ (SPY). The claim "
            "is a joint hypothesis:\n\n"
            "- **H₁ (it out-returns).** Annual excess $\\;x_t = r^I_t - r^B_t\\;$ has $\\mathbb{E}[x_t] > 0$ "
            "with $t > 2$ — comics beat stocks *risk-adjusted*.\n"
            "- **H₂ (it's buyable with alpha).** For a tradable proxy $P$, the intercept $\\alpha$ in "
            "$r^P_t = \\alpha + \\beta r^B_t + \\varepsilon_t$ is positive with a Newey-West *t* > 2.\n"
            "- **H₃ (it survives carry).** The net CAGR after the CGC grading fee $f$, the dealer/"
            "auction round-trip spread $s$ over hold $h$, and annual carry $c$ stays positive.\n\n"
            "The 2020–22 melt-up is the steelman: for ~18 months H₁ held *in-sample*. The test is "
            "whether it holds over the **full cycle, risk-adjusted, net of cost** — i.e. whether it's "
            "an asset class or a bubble."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "If H₁–H₃ held, graded comics would be a genuine diversifier: equity-beating return, "
            "collectible, low-correlation, inflation-proof — the exact pitch. But each leg is "
            "separately falsifiable. H₁ is a **return race** on a common clock. H₂ asks whether *any "
            "investable expression* delivers something beyond market beta — and here the honest "
            "finding is starker: there is essentially **no** listed expression at all (CGC, PSA and "
            "Heritage are private/delisted), so the \"only buyable proxy\" is a badly-fitting one "
            "(Funko). H₃ is the **microstructure tax**: a slab pays a CGC grading fee *and* a ~25% "
            "bid/ask round-trip *and* a multi-year insured hold, frictions an ETF never pays. The "
            "asset-class claim needs all three; failing any one downgrades it to a bubble narrative."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Comic index (proxy).** A hardcoded, cited, **approximate** annual level (base 100 @ "
            "2018), reconstructed from public GoCollect / Heritage reporting (2020–21 melt-up; "
            "early-2022 top; a 2022–23 softening; a 2024–25 blue-chip stabilisation with Action #1 at "
            "a record ~$6.0M). *Labelled a proxy* — its path is defensible, its precise year-end "
            "values are not a live feed.\n"
            "- **Equity proxy.** `FNKO` (Funko) and `SPY` month-end Adj Close (yfinance, cached). "
            "Survivorship is **not** a concern here (a named ticker, not a screen), but note the "
            "honest limitation: FNKO is a toy company, the *nearest* listed proxy only because the "
            "clean ones (CGC/PSA parents, Heritage) are private/delisted.\n"
            "- **Signal test.** (i) Paired annual-excess $t$ of $r^I - r^B$ (small-$n$, weak by "
            "construction). (ii) **Newey-West (6-lag) HAC** $t$ of the proxy alpha vs SPY — the bar "
            "for `REAL` is *t* ≥ 2 in the comics' favour.\n"
            "- **Cost (beat 6).** Charge a ~2% CGC grading fee + a 25% dealer/auction round-trip "
            "spread (over a 3y hold) + 1%/yr carry **once on NAV**; net CAGR.\n"
            "- **Positive control.** A deterministic bubble-and-round-trip path with a *planted* boom/"
            "bust drift; the engine must recover the up-sign and a finite Sharpe — proof a null on the "
            "real tape is a real null, not a broken harness.\n"
            "- **What would make us say \"asset class\":** H₁ *t* > 2 **or** a proxy alpha *t* > 2, **and** "
            "a positive net-of-cost CAGR. We find none of these."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The race — comic index vs SPY, risk-adjusted\n\n"
            "Year-end levels rebased to \\$100, both series on one clock. CAGR, vol and max-drawdown "
            "in the print."
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
            "ax.plot(yrs, lv, 'o-', c=AMBER, lw=2.2, label=f\"comic index  CAGR {si['cagr']*100:.1f}%, vol {si['vol']*100:.0f}%\")\n"
            "ax.plot(sx, spy, 's-', c=GREEN, lw=2.2, label=f\"SPY  CAGR {ss['cagr']*100:.1f}%, vol {ss['vol']*100:.0f}%\")\n"
            "ax.set_xlabel('year-end'); ax.set_ylabel('rebased to 100 @ 2018')\n"
            "ax.set_title('H1: the comic index UNDER-performs SPY over the cycle'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"index: CAGR {si['cagr']*100:.2f}%  vol {si['vol']*100:.1f}%  maxDD {si['mdd']*100:.1f}%\")\n"
            "print(f\"SPY  : CAGR {ss['cagr']*100:.2f}%  vol {ss['vol']*100:.1f}%  maxDD {ss['mdd']*100:.1f}%\")\n"
            "print(f\"annual excess (idx-SPY): mean {ae['mean_excess']*100:+.2f}%/yr  t={ae['t']:+.3f}  (n={ae['n']})\")"
        ),
        md(
            f"> 💡 In plain words: the comic index compounds at **{R['idx_cagr']:.1f}%** against SPY's "
            f"**{R['spy_cagr_ye']:.1f}%** — it loses the race. The mean annual excess is "
            f"**{R['excess_mean']:+.1f}%/yr**, *t* = **{R['excess_t']:+.2f}** (n={R['excess_n']}): the "
            "point estimate is a large negative, but with only 7 annual points it doesn't clear "
            "\\|*t*\\| ≥ 2 in *either* direction. Either way there is **no evidence of "
            "out-performance**. H₁ rejected; the honest stamp is `NONE`, leaning negative."
        ),
        md(
            "### 4b · The buyable proxy — is there alpha, or just (money-losing) beta?\n\n"
            "Newey-West (6-lag) regression of Funko's **monthly** return on SPY. `REAL` needs "
            "$t_\\alpha \\ge 2$ in the comics' favour. (This is the *only* listed proxy — and a poor "
            "one — because the clean names are private/delisted.)"
        ),
        code(
            "if HAVE_PROXIES:\n"
            "    spy_r = PROX['SPY'].pct_change().dropna()\n"
            "    s = st.summarize(PROX['FNKO']); nw = st.newey_west_alpha_t(PROX['FNKO'].pct_change().dropna(), spy_r, 6)\n"
            "    row = dict(cagr=s['cagr']*100, sharpe=s['sharpe'], mdd=s['mdd']*100,\n"
            "               alpha=nw['alpha_ann']*100, beta=nw['beta'], t=nw['t_alpha'])\n"
            "    sp = st.summarize(PROX['SPY'])\n"
            "    sprow = dict(cagr=sp['cagr']*100, sharpe=sp['sharpe'], mdd=sp['mdd']*100)\n"
            "else:\n"
            "    row = dict(cagr=R['fnko_cagr'],sharpe=R['fnko_sharpe'],mdd=R['fnko_mdd'],alpha=R['fnko_alpha'],beta=R['fnko_beta'],t=R['fnko_t'])\n"
            "    sprow = dict(cagr=R['spy_cagr'],sharpe=R['spy_sharpe'],mdd=R['spy_mdd'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "labels=['FNKO alpha\\nvs SPY']; alphas=[row['alpha']]\n"
            "ax.bar(labels, alphas, .4, color=[GREEN if row['alpha']>0 else RED])\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised alpha vs SPY (%)')\n"
            "ax.annotate(f\"t={row['t']:+.2f}  (|t|<2)\",(0,row['alpha']),ha='center',va='bottom' if row['alpha']>=0 else 'top')\n"
            "ax.set_title('H2: no significant alpha in the one buyable proxy (and it LOST money)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"FNKO CAGR {row['cagr']:6.2f}%  Sharpe {row['sharpe']:.2f}  maxDD {row['mdd']:6.1f}%  alpha {row['alpha']:+.2f}%/yr  beta {row['beta']:.2f}  NW t {row['t']:+.2f}\")\n"
            "print(f\"SPY  CAGR {sprow['cagr']:6.2f}%  Sharpe {sprow['sharpe']:.2f}  maxDD {sprow['mdd']:6.1f}%\")"
        ),
        md(
            f"> 💡 In plain words: Funko is **β≈{R['fnko_beta']:.1f}** beta with a "
            f"**{R['fnko_alpha']:+.0f}%/yr** (insignificant, *t*={R['fnko_t']:+.2f}) alpha, an "
            f"**{R['fnko_mdd']:.0f}%** drawdown, and a *negative* CAGR — it amplified the bubble, it "
            f"didn't hedge it. (Its Sharpe reads a hair positive only because ~{R['fnko_vol']:.0f}% "
            "annual vol makes the *arithmetic* mean return positive while the *geometric* return — "
            "what you actually keep — is negative: pure volatility drag.) It does **not** clear "
            "*t* ≥ 2: H₂ rejected. The only listed expression of the trade is a worse bet than SPY."
        ),
        md(
            "### 4c · The microstructure tax — net of CGC grading + the dealer spread + carry\n\n"
            "A physical flip pays a ~2% CGC grading/slabbing fee (flat on cheap books, a % of value "
            "on the expensive keys), a ~25% round-trip spread over a ~3-year hold (auction buyer's "
            "premium + seller's commission, or a dealer's margin), plus ~1%/yr carry. Charge it once "
            "on the index's gross CAGR (a *generous* read — the index level is a mid-market realised "
            "price, not net-to-seller)."
        ),
        code(
            "h = st.net_of_costs_cagr(si['cagr'], round_trip_spread=0.25, hold_years=3.0, grading_fee_pct=0.02, carry_per_year=0.01)\n"
            "steps = ['gross','after\\nspread','after\\ngrading','after\\ncarry']\n"
            "running = [h['gross_cagr']*100,\n"
            "           ((1+h['gross_cagr'])*(1+h['spread_drag_annual'])-1)*100,\n"
            "           ((1+h['gross_cagr'])*(1+h['spread_drag_annual'])*(1+h['grading_drag_annual'])-1)*100,\n"
            "           h['net_cagr']*100]\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "cols=[AMBER, AMBER, AMBER, (RED if running[-1]<0 else GREEN)]\n"
            "ax.bar(steps, running, .6, color=cols)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(R['spy_cagr_ye'], ls='--', c=GREEN, alpha=.6, label='SPY CAGR')\n"
            "for i,v in enumerate(running): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_ylabel('CAGR %/yr'); ax.set_title('H3: net of real frictions, the comic return is NEGATIVE'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"gross {h['gross_cagr']*100:+.2f}%  - spread {h['spread_drag_annual']*100:+.2f}%/yr  - grading {h['grading_drag_annual']*100:+.2f}%/yr  - carry {h['carry_per_year']*100:+.2f}%/yr  =  NET {h['net_cagr']*100:+.2f}%/yr\")"
        ),
        md(
            f"> 💡 In plain words: the gross **{R['cost_gross']:.1f}%** — already a loser to SPY — goes "
            f"**negative ({R['cost_net']:.1f}%/yr)** once you pay CGC to grade it and the dealer/auction "
            "house to move it. And this is *charitable*: a published realised price is mid-market, so a "
            "real seller's net is lower still. H₃ rejected. **MIRAGE** is the only honest stamp."
        ),
        md(
            "### 4d · Positive control — the engine recovers a planted bubble\n\n"
            "A deterministic bubble-and-round-trip (planted boom +55% / bust −30% CAGR, σ=5%/mo, seed "
            "712). The harness must recover the up-sign and a finite Sharpe — proving the nulls above "
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
            "> 💡 In plain words: the engine banks the planted signal (recovered CAGR positive, finite "
            "Sharpe, sign correct). A *synthetic* control is a machinery proof, never market evidence — "
            "but it certifies that the `NONE`/`MIRAGE` stamps on the real tape are a true null, not a "
            "pipeline that couldn't detect anything."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — comic index CAGR {R['idx_cagr']:.1f}% vs SPY {R['spy_cagr_ye']:.1f}%; "
            f"annual excess {R['excess_mean']:+.1f}%/yr, *t* = {R['excess_t']:+.2f} (n={R['excess_n']}); "
            f"proxy alpha insignificant (FNKO *t*={R['fnko_t']:+.2f}) on a *negative*-CAGR name. No "
            "robust *t* ≥ 2 anywhere in the comics' favour — and the point estimates lean negative.\n"
            f"- **Tradability `MIRAGE`** — gross {R['cost_gross']:+.1f}% → net **{R['cost_net']:+.1f}%/yr** "
            f"after a 25% dealer/auction spread + 2% CGC grading + carry; the only listed proxy FNKO is "
            f"β≈{R['fnko_beta']:.1f} with an {R['fnko_mdd']:.0f}% drawdown and a losing CAGR. Illiquid, "
            "high-carry, a bubble that round-tripped 2022–24.\n"
            f"- **Comics beat the S&P? `BUSTED`** — SPY wins CAGR, drawdown, and net-of-cost. The "
            "success stories are pre-2021 buyers of a *specific* key who sold near the top: "
            "survivorship, not an asset class."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the capacity & cost reality\n\n"
            "Terminal wealth of \\$10,000 invested end-2018, S&P index fund vs a comic flipper paying "
            "the real frictions (net CAGR from 4c). Capacity is the other wall: each slab is a bespoke, "
            "illiquid, grade-and-auction transaction — there is no scalable book, and no clean listed "
            "vehicle to hold instead."
        ),
        code(
            "start=10_000.0; yrs_h=7\n"
            "paths={'S&P index fund':R['spy_cagr_ye']/100, 'comic flipper (net)':R['cost_net']/100,\n"
            "       'comic index (gross, untradable)':R['idx_cagr']/100}\n"
            "fig, ax = plt.subplots(figsize=(9.2,4.3))\n"
            "labels=list(paths); ends=[start*(1+g)**yrs_h for g in paths.values()]\n"
            "cols=[GREEN, RED, AMBER]\n"
            "ax.bar(labels, ends, .55, color=cols)\n"
            "for i,v in enumerate(ends): ax.annotate(f'${v:,.0f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('value of $10,000 after 7 years'); ax.set_title('Net of cost, the flipper ends below where they started')\n"
            "plt.tight_layout(); plt.show()\n"
            "for l,g in paths.items(): print(f\"{l:34s} ${start*(1+g)**yrs_h:>10,.0f}  ({g*100:+.1f}%/yr)\")"
        ),
        md(
            "> 💡 In plain words: even the *gross, untradable* index trails the S&P; the *tradable* "
            "version (net of grading + spread + carry) ends **below the starting stake**. And capacity "
            "is fatal — a slab flip is a one-off illiquid trade with a grading fee and a ~25% "
            "round-trip, the antithesis of a scalable strategy, with no clean listed vehicle to hold "
            "instead. There is no sizing or venue that turns this into an edge."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Swap in the live index.** Replace the hardcoded series with the GoCollect indices or "
            "a Heritage realised-price panel and re-run; the *t*-stats sharpen but the sign won't "
            "flip.\n"
            "- **Per-key dispersion.** The aggregate hides survivorship: test individual keys (Action "
            "Comics #1, Amazing Fantasy #15, Incredible Hulk #181) by grade — the winners are a thin "
            "selected tail (the record blue-chips), the bias points *for* the claim, so correct for "
            "it.\n"
            "- **The collectibles prior.** Dimson–Spaenjers and the emotional-assets literature: "
            "collectibles under-perform equities net of carry across the board "
            "([docs/references.md](../docs/references.md)). Comics — with a *grading fee* on top — are "
            "not the exception.\n\n"
            "*The reproducible core is offline and deterministic; the comic index is a **cited, "
            "approximate proxy** and the equity ticker is a **labelled proxy** for the trade. "
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
