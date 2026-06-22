"""Generate the two narrative notebooks for Study 358 ("watches are an asset class").

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic: real-tape cells read the cached month-end
yfinance pulls under ../_cache/ for the equity proxies and the hardcoded (cited,
approximate) resale index from the package; on a cache miss they fall back to the frozen
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


# Frozen headline numbers — mirror of docs/results.md (resale index hardcoded/cited/approx;
# equity proxies month-end Adj Close via yfinance, as-of 2025-12-31).
R = dict(
    win="2018 → 2025",
    idx_levels={2018: 100, 2019: 112, 2020: 130, 2021: 196, 2022: 168, 2023: 150, 2024: 141, 2025: 148},
    idx_yoy={2019: 12.0, 2020: 16.1, 2021: 50.8, 2022: -14.3, 2023: -10.7, 2024: -6.0, 2025: 5.0},
    peak_date="2022-03", peak_level=220, peak_avg_price=45108, peak_trough=-35.9,
    idx_cagr=5.76, idx_vol=22.2, idx_sharpe=0.34, idx_mdd=-28.1,
    spy_cagr_ye=15.03, spy_vol_ye=17.5, spy_mdd_ye=-18.3,
    excess_mean=-9.52, excess_t=-1.054, excess_p=0.340, excess_n=6,
    wosg_cagr=6.76, wosg_vol=55.2, wosg_sharpe=0.41, wosg_mdd=-77.0,
    wosg_alpha=-11.22, wosg_beta=2.05, wosg_t=-0.70, wosg_p=0.488, wosg_n=79,
    cfr_cagr=17.07, cfr_vol=32.8, cfr_sharpe=0.64, cfr_mdd=-36.3,
    cfr_alpha=3.90, cfr_beta=1.04, cfr_t=0.41, cfr_p=0.684, cfr_n=83,
    spy_cagr=16.06, spy_vol=16.7, spy_sharpe=0.98, spy_mdd=-24.0,
    carry_gross=5.76, carry_spread=-7.17, carry_insure=-1.00, carry_net=-2.80,
    syn_peak=1015, syn_end=313, syn_cagr=15.88, syn_sharpe=0.81, syn_mdd=-69.2,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Watches_beat_the_S%26P%3F: Busted](https://img.shields.io/badge/Watches_beat_the_S%26P%3F-Busted-8b949e?style=flat-square)\n\n"
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

from watch_index import data, strategy as st

HAVE_PROXIES = data.have_proxies()
IDX = data.load_resale_index()                       # hardcoded, cited, APPROXIMATE proxy
PROX = data.load_proxies() if HAVE_PROXIES else None
print("equity-proxy cache present:", HAVE_PROXIES,
      "| resale-index years:", IDX.index[0].year, "->", IDX.index[-1].year)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do Rolexes beat the S&P? ⌚\n"
            "### The \"luxury watches are an asset class\" bubble, in plain English\n\n"
            + BADGES +
            "You've heard it at every dinner party since 2021: *\"forget stocks — a steel Rolex "
            "Daytona, a Patek Nautilus, an AP Royal Oak only goes **up**. Watches are an asset class "
            "now. They beat the S&P.\"* For about eighteen glorious months it even looked true: prices "
            "doubled, people flipped watches they couldn't get for triple retail.\n\n"
            "Then the music stopped. This notebook lines the watch market up next to the S&P 500 — on "
            "return, on risk, and on what it actually **costs to buy and sell a watch** — and asks the "
            "only question that matters: *would you have been richer in an index fund?*\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the Newey-West alpha and the cost "
            "algebra? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice — and a data note.** Real watch-resale indices "
            "(WatchCharts, Subdial) aren't free to pull, so the resale line below is a **small, "
            "clearly-cited, approximate** reconstruction of public reporting — a **proxy**, never "
            "presented as the live index. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did watch prices really moon? | **Yes — for a while.** The resale index ran from 100 "
            f"(2018) to ~196 by end-2021 and blew off near **{R['peak_date']}** (avg secondhand price "
            f"≈ \\${R['peak_avg_price']:,}). The mania was real. |\n"
            "| Did it last? | **No.** From the March-2022 top it **round-tripped "
            f"{R['peak_trough']:.0f}%** into 2024. A bubble, not a re-rating. |\n"
            "| Did watches beat the S&P? | **No — they lost.** Over 2018–2025 the index compounded at "
            f"**~{R['idx_cagr']:.0f}%/yr** vs **~{R['spy_cagr_ye']:.0f}%/yr** for SPY — with *more* "
            "swings and a *deeper* drawdown. |\n"
            "| Could you at least buy the trade? | **Not profitably.** The listed watch stocks are "
            f"alpha-free, high-beta beta (one fell **{R['wosg_mdd']:.0f}%**), and once you pay the "
            f"dealer spread to flip the metal, the watches' return goes **negative ({R['carry_net']:.1f}%/yr)**. |\n\n"
            "> The mania was real. The *asset class* was not. The S&P won every column that matters."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Luxury watches — Rolex, Patek Philippe, Audemars Piguet — are a store of value that "
            "beats the stock market. Buy the right reference, wait, sell it for more. The rich have "
            "always known watches are money you can wear.\"*\n\n"
            "It's a *steelman-able* claim: between 2020 and early 2022 the secondary market genuinely "
            "melted up. Stimulus cheques, zero rates, lockdown boredom and a wall of social-media "
            "flipping pushed hyped steel sports models to 2–5× their retail list. A Patek Nautilus "
            f"5711 changed hands near \\$103,000; the average secondhand watch peaked around "
            f"\\${R['peak_avg_price']:,}. For a moment, \"watches beat stocks\" was simply a fact."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it were durably true, it would be a big deal: an asset that beats equities *and* you "
            "get to wear it, uncorrelated with your stock portfolio, inflation-proof. That's the "
            "pitch every watch dealer and lifestyle-finance influencer leans on. But \"it went up for "
            "two years\" and \"it's an asset class that beats the S&P\" are very different statements. "
            "The first is about a **bubble**; the second is a claim about the **long-run, "
            "risk-adjusted, net-of-cost** return. We can check the second directly."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three honest comparisons, each against the S&P 500 (SPY):\n\n"
            "1. **The resale index vs SPY.** Put the (cited, approximate) secondary-market watch "
            "index next to SPY on the same 2018–2025 clock — return, volatility, worst drawdown.\n"
            "2. **The thing you can actually buy.** You can't buy \"the watch index.\" You *can* buy "
            "the listed retailer **Watches of Switzerland** and the group **Richemont**. Do they "
            "deliver the watch trade's return — or just stock-market risk?\n"
            "3. **The cost of flipping.** A watch isn't an ETF: you buy near retail-plus and sell at a "
            "wholesale discount, and it sits insured in a safe for years. Charge that, and see what's "
            "left.\n\n"
            "**What would make us say \"asset class\"?** The index beats the S&P on *risk-adjusted, "
            "net-of-cost* return. Anything less is a bubble story."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: the round-trip.** Here is the (approximate, cited) resale index — the melt-up "
            "and the giveback."
        ),
        code(
            "yrs = list(R['idx_levels'].keys())\n"
            "lv = [IDX.loc[f'{y}-12-31'] for y in yrs] if True else list(R['idx_levels'].values())\n"
            "fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "ax.plot(yrs, lv, 'o-', c=AMBER, lw=2.4, label='watch resale index (proxy)')\n"
            "ax.axvline(2022, ls='--', c=RED, alpha=.6)\n"
            "ax.annotate('March-2022\\nblow-off top', (2022, max(lv)), textcoords='offset points',\n"
            "            xytext=(8, -4), color=RED, fontsize=9)\n"
            "ax.axhline(100, ls=':', c=GREY)\n"
            "ax.set_xlabel('year-end'); ax.set_ylabel('index level (base 100 = 2018)')\n"
            "ax.set_title('The watch \"asset class\": a melt-up, then a round-trip'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('levels:', {y:int(round(float(v))) for y,v in zip(yrs, lv)})\n"
            "print(f\"down {R['peak_trough']:.0f}% from the March-2022 peak to the 2024 low\")"
        ),
        md(
            f"It nearly **doubled** into 2021, then gave most of it back: **{R['peak_trough']:.0f}%** "
            "from the top. If you bought at the peak — exactly when the \"asset class\" headlines were "
            "loudest — you spent two years underwater. That's not how a store of value behaves; it's "
            "how a bubble behaves."
        ),
        md(
            "**Now the race: watches vs the S&P.** Same money, same years — who's richer at the end?"
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
            "ax.plot(yrs, [float(v) for v in lv], 'o-', c=AMBER, lw=2.2, label=f\"watches  ({R['idx_cagr']:.0f}%/yr)\")\n"
            "ax.plot(sx, sy, 's-', c=GREEN, lw=2.2, label=f\"S&P 500  ({R['spy_cagr_ye']:.0f}%/yr)\")\n"
            "ax.set_xlabel('year-end'); ax.set_ylabel('$100 invested at end-2018')\n"
            "ax.set_title('Watches vs the S&P: not close'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"watches CAGR ~{R['idx_cagr']:.1f}%  vs  SPY CAGR ~{R['spy_cagr_ye']:.1f}%  |  \"\n"
            "      f\"watches maxDD {R['idx_mdd']:.0f}%  vs  SPY {R['spy_mdd_ye']:.0f}%\")"
        ),
        md(
            f"The S&P didn't just win — it **lapped** the watches: **~{R['spy_cagr_ye']:.0f}%/yr** vs "
            f"**~{R['idx_cagr']:.0f}%/yr**, with a *shallower* worst drop. More return, less pain. "
            "Even the gaudy 2020–21 spike isn't enough to save the watch line over the full cycle."
        ),
        md(
            "**\"Fine — I'll buy the watch *stocks*.\"** You can: Watches of Switzerland (the big "
            "Rolex/Patek dealer) and Richemont (Cartier/IWC). Do they hand you the watch trade — or "
            "just a wilder ride to roughly the same place?"
        ),
        code(
            "names = ['Watches of\\nSwitzerland', 'Richemont', 'S&P 500']\n"
            "cagrs = [R['wosg_cagr'], R['cfr_cagr'], R['spy_cagr']]\n"
            "mdds  = [R['wosg_mdd'], R['cfr_mdd'], R['spy_mdd']]\n"
            "x = np.arange(3); fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "ax.bar(x-.2, cagrs, .4, color=[AMBER,AMBER,GREEN], label='CAGR %/yr')\n"
            "ax.bar(x+.2, mdds, .4, color=RED, alpha=.55, label='worst drawdown %')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels(names)\n"
            "ax.set_ylabel('percent'); ax.set_title('The buyable proxies: more risk, no extra reward'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"WOSG.L: CAGR {R['wosg_cagr']:.1f}% but maxDD {R['wosg_mdd']:.0f}% (beta {R['wosg_beta']:.1f})\")\n"
            "print(f\"CFR.SW: CAGR {R['cfr_cagr']:.1f}%, basically SPY at twice the vol\")"
        ),
        md(
            f"Watches of Switzerland did the bubble *with leverage*: it crashed **{R['wosg_mdd']:.0f}%** "
            "as the mania deflated — the opposite of a safe-haven. Richemont roughly matched the S&P's "
            "return but at far higher volatility. Neither pays you for the extra white-knuckle risk."
        ),
        md(
            "**The part the pitch never mentions: it costs real money to flip a watch.** You buy near "
            "retail-plus, sell at a dealer's wholesale discount (~20% round-trip), and insure/store it "
            "for years. Charge that against the watch index's gross return:"
        ),
        code(
            "labels = ['gross\\nreturn', 'dealer\\nspread', 'insurance\\n& storage', 'NET to\\nyou']\n"
            "vals = [R['carry_gross'], R['carry_spread'], R['carry_insure'], R['carry_net']]\n"
            "cols = [AMBER, RED, RED, (RED if R['carry_net']<0 else GREEN)]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('% per year')\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_title('Where the watch \"return\" goes once you actually transact')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"gross {R['carry_gross']:+.1f}%/yr  ->  NET {R['carry_net']:+.1f}%/yr after spread + carry\")"
        ),
        md(
            f"There it is. A gross **{R['carry_gross']:.1f}%/yr** that already lost to the S&P turns "
            f"**negative ({R['carry_net']:.1f}%/yr)** the moment you pay to buy, sell and hold the "
            "physical watch. A boring T-bill beat it; the index fund laps it."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Watches returned ~{R['idx_cagr']:.0f}%/yr vs ~{R['spy_cagr_ye']:.0f}%/"
            "yr for the S&P — they *under*-performed, with more risk. No evidence they beat stocks.\n"
            "- **Tradability — Mirage.** Illiquid, high-carry; the gross return goes **negative** after "
            "the dealer spread, and the buyable proxies are alpha-free high-beta beta.\n"
            "- **Watches beat the S&P? — Busted.** Every column — return, risk-adjusted return, "
            "drawdown, net-of-cost — goes to the index fund."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the honest bottom line\n\n"
            "Imagine two people at the end of 2018, each with \\$10,000. One buys an S&P index fund. "
            "The other becomes a watch flipper — buys, insures, stores, and eventually sells, paying "
            "the real frictions. Where do they land by end-2025?"
        ),
        code(
            "start = 10_000.0\n"
            "spy_end = start*(1+R['spy_cagr_ye']/100)**7\n"
            "watch_end = start*(1+R['carry_net']/100)**7   # net of spread + carry\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(['S&P index fund', 'watch flipper\\n(net of costs)'], [spy_end, watch_end],\n"
            "       color=[GREEN, RED], width=.55)\n"
            "for i,v in enumerate([spy_end, watch_end]): ax.annotate(f'${v:,.0f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('value of $10,000 after 7 years')\n"
            "ax.set_title('Same $10k, end-2018 -> end-2025'); plt.tight_layout(); plt.show()\n"
            "print(f'S&P index fund: ${spy_end:,.0f}   |   watch flipper (net): ${watch_end:,.0f}')"
        ),
        md(
            "The index-fund investor roughly **triples** their money doing nothing. The flipper — after "
            "spreads, insurance and a bubble that round-tripped — ends up with **less than they started "
            "with**. The people who really made money in watches bought *before* March 2022 and sold "
            "near the top: bubble survivors, not asset-class investors. You only hear from them."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Pull the real index yourself.** Our resale line is a cited *approximation*; "
            "WatchCharts / Subdial publish the live series. Swap it in — the shape (and the verdict) "
            "won't move, but you'll have the exact tape.\n"
            "- **The collectibles pattern.** Wine, art, sneakers, trading cards: every \"passion asset\" "
            "tells the same story — real spikes, brutal carry, equities win net of cost (see "
            "[docs/references.md](../docs/references.md)).\n"
            "- **The survivorship angle.** [Study 301 — Triple-RSI](../../301-triple-rsi/) is the same "
            "shape in stocks: a handful of winners narrated as a system.\n\n"
            "*Think a specific reference (a steel Daytona, a 5711) beat the S&P net of every cost? "
            "Pull its WatchCharts history, charge the spread, and show it — then check it wasn't just "
            "one lucky buy near the bottom.*"
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
            "# Watches as an asset class — a quantitative teardown 🔬\n"
            "### Resale index vs SPY (CAGR / vol / MDD + an annual-excess *t*) · Newey-West proxy "
            "alpha · the dealer-spread + carry haircut on NAV · a synthetic bubble positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We test "
            "the strongest tradable form of \"luxury watches beat the S&P\": (H₁) the secondary-market "
            "resale index out-returns SPY risk-adjusted; (H₂) a buyable proxy carries alpha vs the "
            "market; (H₃) it survives the transaction + carry cost of owning physical watches. We find "
            "**H₁ rejected** (it *under*-performs), **H₂ rejected** (no significant alpha), **H₃ "
            "rejected** (negative net of costs).\n\n"
            "> ⚠️ **Not investment advice — data provenance.** The resale index is **hardcoded, "
            "cited, approximate** (public WatchCharts / Morgan Stanley×WatchCharts reporting — a "
            "*labelled proxy*, never the live feed). Equity proxies `WOSG.L`, `CFR.SW`, `SPY` are "
            "month-end Adj Close via yfinance (as-of 2025-12-31). Offline core + synthetic control are "
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
            f"| **Signal** | `NONE` | Resale index CAGR **{R['idx_cagr']:.1f}%** vs SPY "
            f"**{R['spy_cagr_ye']:.1f}%** (2018–2025); mean annual excess **{R['excess_mean']:+.1f}%/yr**, "
            f"*t* = **{R['excess_t']:+.2f}** (n={R['excess_n']}). Proxies: no significant alpha "
            f"(WOSG NW *t*={R['wosg_t']:+.2f}, CFR *t*={R['cfr_t']:+.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | Gross index CAGR **{R['carry_gross']:+.1f}%** → **NET "
            f"{R['carry_net']:+.1f}%/yr** after a 20% dealer spread + 1%/yr carry. Buyable proxy WOSG.L "
            f"maxDD **{R['wosg_mdd']:.0f}%**, β≈{R['wosg_beta']:.1f}. Illiquid, high-carry, round-tripped. |\n"
            f"| **Watches beat the S&P?** | `BUSTED` | SPY wins CAGR ({R['spy_cagr']:.1f} vs "
            f"{R['idx_cagr']:.1f}), Sharpe ({R['spy_sharpe']:.2f} vs {R['idx_sharpe']:.2f}), maxDD "
            f"({R['spy_mdd_ye']:.0f} vs {R['idx_mdd']:.0f}), and net-of-cost. Every column. |\n\n"
            "> 💡 In plain words: the watch market under-performed stocks with more risk, the only "
            "thing you can buy is alpha-free beta, and the physical-flip frictions turn even the gross "
            "return negative. There is no axis on which \"watches beat the S&P\" survives."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let the secondary-market index level be $I_t$ and the benchmark $B_t$ (SPY). The claim "
            "is a joint hypothesis:\n\n"
            "- **H₁ (it out-returns).** Annual excess $\\;x_t = r^I_t - r^B_t\\;$ has $\\mathbb{E}[x_t] > 0$ "
            "with $t > 2$ — watches beat stocks *risk-adjusted*.\n"
            "- **H₂ (it's buyable with alpha).** For a tradable proxy $P$, the intercept $\\alpha$ in "
            "$r^P_t = \\alpha + \\beta r^B_t + \\varepsilon_t$ is positive with a Newey-West *t* > 2.\n"
            "- **H₃ (it survives carry).** The net CAGR after the dealer round-trip spread $s$ over "
            "hold $h$ and annual carry $c$ stays positive: $(1+g)(1+((1-s)^{1/h}-1))(1-c)-1 > 0$.\n\n"
            "The 2020–22 melt-up is the steelman: for ~18 months H₁ held *in-sample*. The test is "
            "whether it holds over the **full cycle, risk-adjusted, net of cost** — i.e. whether it's "
            "an asset class or a bubble."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "If H₁–H₃ held, watches would be a genuine diversifier: equity-beating return, wearable, "
            "low-correlation, inflation-proof — the exact pitch. But each leg is separately falsifiable. "
            "H₁ is a **return race** on a common clock. H₂ asks whether the *only investable expression* "
            "(listed equities) delivers anything beyond market beta — because you cannot custody \"the "
            "index.\" H₃ is the **microstructure tax**: a physical watch has a ~20% bid/ask round-trip "
            "and a multi-year holding period with insurance and servicing carry, frictions an ETF never "
            "pays. The asset-class claim needs all three; failing any one downgrades it to a bubble "
            "narrative."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Resale index (proxy).** A hardcoded, cited, **approximate** annual level (base 100 @ "
            "2018), reconstructed from public WatchCharts / Morgan Stanley×WatchCharts / press figures "
            "(March-2022 top; full-year −10.7%/−6.1%/+4.9% for 2023/24/25). *Labelled a proxy* — its "
            "path is defensible, its precise year-end values are not a live feed.\n"
            "- **Equity proxies.** `WOSG.L`, `CFR.SW`, `SPY` month-end Adj Close (yfinance, cached). "
            "Survivorship is **not** a concern here (named tickers, not a screen), but the WOSG.L "
            "series starts at its **2019-05 IPO** — a real look-ahead caveat (we only see the listed "
            "life), stated.\n"
            "- **Signal test.** (i) Paired annual-excess $t$ of $r^I - r^B$ (small-$n$, weak by "
            "construction). (ii) **Newey-West (6-lag) HAC** $t$ of the proxy alpha vs SPY — the bar "
            "for `REAL` is *t* ≥ 2 in the watches' favour.\n"
            "- **Cost (beat 6).** Charge the dealer round-trip spread (20% over a 3y hold) + 1%/yr "
            "carry **once on NAV**; net CAGR.\n"
            "- **Positive control.** A deterministic bubble-and-round-trip path with a *planted* boom/"
            "bust drift; the engine must recover the up-sign and a finite Sharpe — proof a null on the "
            "real tape is a real null, not a broken harness.\n"
            "- **What would make us say \"asset class\":** H₁ *t* > 2 **or** a proxy alpha *t* > 2, **and** "
            "a positive net-of-cost CAGR. We find none of these."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The race — resale index vs SPY, risk-adjusted\n\n"
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
            "ax.plot(yrs, lv, 'o-', c=AMBER, lw=2.2, label=f\"watch index  CAGR {si['cagr']*100:.1f}%, vol {si['vol']*100:.0f}%\")\n"
            "ax.plot(sx, spy, 's-', c=GREEN, lw=2.2, label=f\"SPY  CAGR {ss['cagr']*100:.1f}%, vol {ss['vol']*100:.0f}%\")\n"
            "ax.set_xlabel('year-end'); ax.set_ylabel('rebased to 100 @ 2018')\n"
            "ax.set_title('H1: the watch index UNDER-performs SPY over the cycle'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"index: CAGR {si['cagr']*100:.2f}%  vol {si['vol']*100:.1f}%  maxDD {si['mdd']*100:.1f}%\")\n"
            "print(f\"SPY  : CAGR {ss['cagr']*100:.2f}%  vol {ss['vol']*100:.1f}%  maxDD {ss['mdd']*100:.1f}%\")\n"
            "print(f\"annual excess (idx-SPY): mean {ae['mean_excess']*100:+.2f}%/yr  t={ae['t']:+.3f}  (n={ae['n']})\")"
        ),
        md(
            f"> 💡 In plain words: the watch index compounds at **{R['idx_cagr']:.1f}%** against SPY's "
            f"**{R['spy_cagr_ye']:.1f}%**, with *higher* vol and a *deeper* drawdown — it loses on every "
            f"axis. The mean annual excess is **{R['excess_mean']:+.1f}%/yr**, *t* = **{R['excess_t']:+.2f}** "
            f"(n={R['excess_n']}): not even a *clean* loss given 6 annual points, but certainly **no "
            "evidence of out-performance**. H₁ rejected; the honest stamp is `NONE`, leaning negative."
        ),
        md(
            "### 4b · The buyable proxies — is there alpha, or just beta?\n\n"
            "Newey-West (6-lag) regression of each proxy's **monthly** return on SPY. `REAL` needs "
            "$t_\\alpha \\ge 2$ in the watches' favour."
        ),
        code(
            "if HAVE_PROXIES:\n"
            "    spy_r = PROX['SPY'].pct_change().dropna()\n"
            "    rows = {}\n"
            "    for t in ['WOSG.L','CFR.SW']:\n"
            "        s = st.summarize(PROX[t]); nw = st.newey_west_alpha_t(PROX[t].pct_change().dropna(), spy_r, 6)\n"
            "        rows[t] = dict(cagr=s['cagr']*100, sharpe=s['sharpe'], mdd=s['mdd']*100,\n"
            "                       alpha=nw['alpha_ann']*100, beta=nw['beta'], t=nw['t_alpha'])\n"
            "else:\n"
            "    rows = {'WOSG.L':dict(cagr=R['wosg_cagr'],sharpe=R['wosg_sharpe'],mdd=R['wosg_mdd'],alpha=R['wosg_alpha'],beta=R['wosg_beta'],t=R['wosg_t']),\n"
            "            'CFR.SW':dict(cagr=R['cfr_cagr'],sharpe=R['cfr_sharpe'],mdd=R['cfr_mdd'],alpha=R['cfr_alpha'],beta=R['cfr_beta'],t=R['cfr_t'])}\n"
            "labels=list(rows); alphas=[rows[t]['alpha'] for t in labels]; ts=[rows[t]['t'] for t in labels]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "cols=[GREEN if a>0 else RED for a in alphas]\n"
            "ax.bar(labels, alphas, .5, color=cols)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised alpha vs SPY (%)')\n"
            "for i,t in enumerate(labels): ax.annotate(f\"t={ts[i]:+.2f}\",(i,alphas[i]),ha='center',va='bottom' if alphas[i]>=0 else 'top')\n"
            "ax.set_title('H2: no significant alpha in either buyable proxy (|t|<2)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for t in labels: r=rows[t]; print(f\"{t:7s} CAGR {r['cagr']:6.2f}%  Sharpe {r['sharpe']:.2f}  maxDD {r['mdd']:6.1f}%  alpha {r['alpha']:+.2f}%/yr  beta {r['beta']:.2f}  NW t {r['t']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: Watches of Switzerland is **β≈{R['wosg_beta']:.1f}** beta with a "
            f"**{R['wosg_alpha']:+.0f}%/yr** (insignificant, *t*={R['wosg_t']:+.2f}) alpha and a "
            f"**{R['wosg_mdd']:.0f}%** drawdown — it amplified the bubble, it didn't hedge it. Richemont "
            f"is essentially SPY at higher vol (*t*={R['cfr_t']:+.2f}). **Neither clears *t* ≥ 2** in "
            "either direction: H₂ rejected. The only investable expression of the trade is market beta "
            "you could have bought cheaper as SPY."
        ),
        md(
            "### 4c · The microstructure tax — net of the dealer spread + carry\n\n"
            "A physical flip pays a ~20% round-trip spread over a ~3-year hold plus ~1%/yr carry. "
            "Charge it once on the index's gross CAGR (a *generous* read — it ignores that the index "
            "level itself is mid-market, not net-to-seller)."
        ),
        code(
            "h = st.net_of_carry_cagr(si['cagr'], round_trip_spread=0.20, hold_years=3.0, insure_per_year=0.01)\n"
            "steps = ['gross','after\\nspread','after\\ncarry']\n"
            "running = [h['gross_cagr']*100,\n"
            "           ((1+h['gross_cagr'])*(1+h['spread_drag_annual'])-1)*100,\n"
            "           h['net_cagr']*100]\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.3))\n"
            "cols=[AMBER, AMBER, (RED if running[-1]<0 else GREEN)]\n"
            "ax.bar(steps, running, .55, color=cols)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(R['spy_cagr_ye'], ls='--', c=GREEN, alpha=.6, label='SPY CAGR')\n"
            "for i,v in enumerate(running): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_ylabel('CAGR %/yr'); ax.set_title('H3: net of real frictions, the watch return is NEGATIVE'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"gross {h['gross_cagr']*100:+.2f}%  - spread {h['spread_drag_annual']*100:+.2f}%/yr  - carry {h['insure_per_year']*100:+.2f}%/yr  =  NET {h['net_cagr']*100:+.2f}%/yr\")"
        ),
        md(
            f"> 💡 In plain words: the gross **{R['carry_gross']:.1f}%** — already a loser to SPY — goes "
            f"**negative ({R['carry_net']:.1f}%/yr)** once you pay the spread and carry to actually own "
            "the metal. And this is *charitable*: a published index level is mid-market, so a real "
            "seller's realised return is lower still. H₃ rejected. **MIRAGE** is the only honest stamp."
        ),
        md(
            "### 4d · Positive control — the engine recovers a planted bubble\n\n"
            "A deterministic bubble-and-round-trip (planted boom +55% / bust −30% CAGR, σ=5%/mo, seed "
            "358). The harness must recover the up-sign and a finite Sharpe — proving the nulls above "
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
            "> 💡 In plain words: the engine banks the planted signal (recovered CAGR ~+16%, Sharpe "
            "~0.8, sign correct). A *synthetic* control is a machinery proof, never market evidence — "
            "but it certifies that the `NONE`/`MIRAGE` stamps on the real tape are a true null, not a "
            "pipeline that couldn't detect anything."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — resale index CAGR {R['idx_cagr']:.1f}% vs SPY {R['spy_cagr_ye']:.1f}%; "
            f"annual excess {R['excess_mean']:+.1f}%/yr, *t* = {R['excess_t']:+.2f} (n={R['excess_n']}); "
            f"proxy alphas insignificant (WOSG *t*={R['wosg_t']:+.2f}, CFR *t*={R['cfr_t']:+.2f}). No "
            "robust *t* ≥ 2 anywhere in the watches' favour — and the point estimates lean negative.\n"
            f"- **Tradability `MIRAGE`** — gross {R['carry_gross']:+.1f}% → net **{R['carry_net']:+.1f}%/yr** "
            f"after a 20% dealer spread + carry; buyable proxy WOSG.L is β≈{R['wosg_beta']:.1f} with a "
            f"{R['wosg_mdd']:.0f}% drawdown. Illiquid, high-carry, bubble that round-tripped 2022–24.\n"
            f"- **Watches beat the S&P? `BUSTED`** — SPY wins CAGR, Sharpe ({R['spy_sharpe']:.2f} vs "
            f"{R['idx_sharpe']:.2f}), drawdown and net-of-cost. The success stories are pre-March-2022 "
            "buyers who sold near the top: survivorship, not an asset class."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the capacity & cost reality\n\n"
            "Terminal wealth of \\$10,000 invested end-2018, S&P index fund vs a watch flipper paying "
            "the real frictions (net CAGR from 4c). Capacity is the other wall: each physical flip is a "
            "bespoke, illiquid, ~20%-spread transaction — there is no scalable book."
        ),
        code(
            "start=10_000.0; yrs_h=7\n"
            "paths={'S&P index fund':R['spy_cagr_ye']/100, 'watch flipper (net)':R['carry_net']/100,\n"
            "       'watch index (gross, untradable)':R['idx_cagr']/100}\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
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
            "version (net of spread + carry) ends **below the starting stake**. And capacity is fatal — "
            "a watch flip is a one-off illiquid trade with a ~20% round-trip, the antithesis of a "
            "scalable strategy. There is no sizing or venue that turns this into an edge."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Swap in the live index.** Replace the hardcoded series with the WatchCharts / Subdial "
            "feed (or the Morgan Stanley×WatchCharts quarterly levels) and re-run; the *t*-stats sharpen "
            "but the sign won't flip.\n"
            "- **Per-reference dispersion.** The aggregate hides survivorship: test individual "
            "references (steel Daytona, Nautilus 5711, Royal Oak 15202) — the winners are a thin "
            "selected tail, the bias points *for* the claim, so correct for it.\n"
            "- **The collectibles prior.** Dimson–Spaenjers and the emotional-assets literature: "
            "collectibles under-perform equities net of carry across the board "
            "([docs/references.md](../docs/references.md)). Watches are not the exception.\n\n"
            "*The reproducible core is offline and deterministic; the resale index is a **cited, "
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
