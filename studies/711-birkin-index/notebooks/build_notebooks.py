"""Generate the two narrative notebooks for Study 711 ("a Birkin beats the S&P and gold").

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic: real-tape cells read the cached month-end
yfinance pulls under ../_cache/ for the equity/gold proxies and the hardcoded (cited,
approximate) Birkin resale index from the package; on a cache miss they fall back to the
frozen headline numbers in ``R`` (mirroring docs/results.md). The synthetic compounder
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


# Frozen headline numbers — mirror of docs/results.md (Birkin index hardcoded/cited/approx;
# equity+gold proxies month-end Adj Close via yfinance, as-of 2025-12-31).
R = dict(
    win="2015 → 2025",
    idx_levels={2015: 100, 2016: 106, 2017: 113, 2018: 121, 2019: 128, 2020: 134,
                2021: 150, 2022: 162, 2023: 166, 2024: 158, 2025: 163},
    idx_yoy={2016: 6.0, 2017: 6.6, 2018: 7.1, 2019: 5.8, 2020: 4.7, 2021: 11.9,
             2022: 8.0, 2023: 2.5, 2024: -4.8, 2025: 3.2},
    myth_cagr=14.2, myth_window="1980–2015", myth_sp500=8.7,
    idx_cagr=5.01, idx_vol=4.4, idx_sharpe=1.16, idx_mdd=-4.8,
    spy_cagr_ye=14.71, spy_vol_ye=15.7, spy_mdd_ye=-18.2,
    gld_cagr_ye=14.60, gld_vol_ye=19.9, gld_mdd_ye=-4.9,
    excess_spy_mean=-10.71, excess_spy_t=-1.952, excess_spy_p=0.083, excess_n=10,
    excess_gld_mean=-10.88, excess_gld_t=-1.524, excess_gld_p=0.162,
    rms_cagr=20.85, rms_vol=24.5, rms_sharpe=0.90, rms_mdd=-35.0,
    rms_alpha=11.88, rms_beta=0.76, rms_t=1.93, rms_p=0.055, rms_n=131,
    mc_cagr=16.85, mc_vol=24.1, mc_sharpe=0.77, mc_mdd=-46.8,
    mc_alpha=6.16, mc_beta=0.89, mc_t=0.93, mc_p=0.354,
    ker_cagr=7.97, ker_vol=31.3, ker_sharpe=0.40, ker_mdd=-74.3,
    ker_alpha=-1.64, ker_beta=1.00, ker_t=-0.18, ker_p=0.855,
    spy_cagr=13.84, spy_vol=15.0, spy_sharpe=0.95, spy_mdd=-23.9,
    gld_cagr=11.28, gld_vol=14.3, gld_sharpe=0.82, gld_mdd=-18.1,
    carry_gross=5.01, carry_spread=-11.21, carry_insure=-0.50, carry_net=-7.23,
    syn_peak=316, syn_end=270, syn_cagr=10.54, syn_sharpe=1.00, syn_mdd=-21.8,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Beats_stocks_%26_gold%3F: Busted](https://img.shields.io/badge/Beats_stocks_%26_gold%3F-Busted-8b949e?style=flat-square)\n\n"
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
GOLD = "#c9a227"

from birkin_index import data, strategy as st

HAVE_PROXIES = data.have_proxies()
IDX = data.load_resale_index()                       # hardcoded, cited, APPROXIMATE proxy
PROX = data.load_proxies() if HAVE_PROXIES else None
print("equity/gold-proxy cache present:", HAVE_PROXIES,
      "| Birkin-index years:", IDX.index[0].year, "->", IDX.index[-1].year)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a Birkin beat the S&P — and even gold? 👜\n"
            "### The \"Hermès handbags are the best-performing asset on earth\" claim, in plain English\n\n"
            + BADGES +
            "You've seen the headline: *\"a Hermès Birkin has out-returned the S&P 500 **and** gold — "
            "14% a year, and it's never had a down year.\"* It comes from a 2016 study that got "
            "recycled everywhere, from luxury blogs to a Credit Suisse note. On its face it's a jaw- "
            "dropper: the ultimate handbag, quietly beating Wall Street *and* the oldest safe-haven "
            "asset there is.\n\n"
            "This notebook lines the Birkin up next to the S&P 500 **and** gold — on return, on risk, "
            "and on what it actually **costs to buy and sell a bag** — and asks the only question that "
            "matters: *would you have been richer in an index fund (or a gold ETF)?*\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the Newey-West alpha and the cost "
            "algebra? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice — and a data note.** There's no free, live handbag-resale "
            "index to pull, so the Birkin line below is a **small, clearly-cited, approximate** "
            "reconstruction of public reporting — a **proxy**, never presented as the live index. "
            "Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do Birkins hold their value? | **Yes — beautifully.** The resale line barely wobbles "
            f"(worst drawdown just **{R['idx_mdd']:.0f}%**). As a *store of value* the bag is genuinely "
            "steady. |\n"
            "| Did it return 14%/yr like the headline? | **Not on the modern tape.** Over 2015–2025 the "
            f"(cited, approximate) resale index compounded at **~{R['idx_cagr']:.0f}%/yr** — a third of "
            f"the famous **{R['myth_cagr']:.0f}%** Baghunter number from a cherry-picked 1980–2015 window. |\n"
            "| Did it beat the S&P and gold? | **No — it lost to both.** ~"
            f"{R['idx_cagr']:.0f}%/yr for the bag vs **~{R['spy_cagr_ye']:.0f}%/yr** for the S&P and "
            f"**~{R['gld_cagr_ye']:.0f}%/yr** for gold. |\n"
            "| Could you at least buy the trade? | **Not the bag, profitably.** Once you pay the ~30% "
            f"consignment spread to flip it, the bag's return goes **negative ({R['carry_net']:.1f}%/yr)**. "
            "The one thing that *did* beat the S&P was Hermès's **stock** — not the handbag. |\n\n"
            "> The bag is a lovely *store of value*. It is not the *return machine* the headline sells — "
            "and net of the spread, it doesn't even keep up with cash."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"An Hermès Birkin is the best investment you can make. A 2016 study found Birkins "
            f"returned **{R['myth_cagr']:.0f}% a year** over three-and-a-half decades — beating the S&P "
            f"500 (**{R['myth_sp500']:.0f}%**) and gold — and, unlike stocks, the Birkin **never had a "
            "single down year**. It's money you can carry.\"*\n\n"
            "It's a *steelman-able* claim. Hermès deliberately under-supplies the Birkin: you can't "
            "just walk in and buy one, so a thick secondary market trades them at boutique-plus. Retail "
            "prices ratchet up 5–10% almost every year, dragging resale with them. For a genuinely "
            "scarce, brand-controlled object, \"it only goes up\" isn't crazy on its face — which is "
            "exactly why the claim spreads."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it were durably true it would be extraordinary: an asset that beats equities **and** "
            "gold, has almost no drawdown, *and* you get to wear it to dinner. That's a free lunch with "
            "a lunch on top. But \"it appreciated smoothly\" and \"it beats the S&P and gold\" are very "
            "different statements. A thing can be a wonderful **store of value** — low volatility, "
            "keeps pace with inflation — and still be a poor **return** relative to owning the stock "
            "market. The headline quietly swaps one for the other. We can check the real one directly."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three honest comparisons, each against **both** the S&P 500 (SPY) and gold (GLD) — the two "
            "assets the claim names:\n\n"
            "1. **The resale index vs SPY and gold.** Put the (cited, approximate) Birkin index next to "
            "SPY and GLD on the same 2015–2025 clock — return, volatility, worst drawdown.\n"
            "2. **The thing you can actually buy.** You can't custody \"the Birkin index.\" You *can* buy "
            "the maison: **Hermès (RMS.PA)** itself, plus **LVMH** and **Kering**. Do they deliver the "
            "bag trade's return — or just stock-market risk?\n"
            "3. **The cost of flipping.** A Birkin isn't an ETF: you buy at a premium and sell through "
            "consignment at a ~30% haircut, and it sits insured in a closet for years. Charge that, and "
            "see what's left.\n\n"
            "**What would make us say \"asset class\"?** The bag beats the S&P *and* gold on "
            "*risk-adjusted, net-of-cost* return. Anything less and the headline is a marketing number."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: the bag's price path.** Here is the (approximate, cited) Birkin resale index — "
            "smooth, steady, and nothing like the S&P's ride."
        ),
        code(
            "yrs = list(R['idx_levels'].keys())\n"
            "lv = [float(IDX.loc[f'{y}-12-31']) for y in yrs]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "ax.plot(yrs, lv, 'o-', c=AMBER, lw=2.4, label='Birkin resale index (proxy)')\n"
            "ax.axhline(100, ls=':', c=GREY)\n"
            "ax.annotate('Baghunter myth:\\n14%/yr, never down', (yrs[1], 150), color=RED, fontsize=9)\n"
            "ax.set_xlabel('year-end'); ax.set_ylabel('index level (base 100 = 2015)')\n"
            "ax.set_title('The Birkin \"asset\": smooth and steady — but only ~5%/yr'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('levels:', {y:int(round(v)) for y,v in zip(yrs, lv)})\n"
            "print(f\"CAGR ~{R['idx_cagr']:.1f}%/yr, worst drawdown only {R['idx_mdd']:.0f}% — a real store of value\")"
        ),
        md(
            f"It rose steadily to a **{R['idx_mdd']:.0f}%** worst-drawdown — genuinely calm. But the "
            f"slope is the tell: **~{R['idx_cagr']:.0f}%/yr**, not the **{R['myth_cagr']:.0f}%** of the "
            "viral headline. That famous number came from a 1980–2015 window that conveniently ended "
            "*before* the modern data — and counted only the bags that kept trading. Steady? Yes. A "
            "return monster? No."
        ),
        md(
            "**Now the race: the bag vs the S&P *and* gold.** Same money, same years — who's richest at "
            "the end?"
        ),
        code(
            "if HAVE_PROXIES:\n"
            "    def ye_norm(t):\n"
            "        s = PROX[t].resample('YE').last(); s = s[(s.index.year>=2015)&(s.index.year<=2025)]\n"
            "        return [d.year for d in s.index], list((s/s.iloc[0]*100).values)\n"
            "    sx, sy = ye_norm('SPY'); gx, gy = ye_norm('GLD')\n"
            "else:\n"
            "    sx = gx = yrs\n"
            "    sy = [100*(1+R['spy_cagr_ye']/100)**(y-2015) for y in yrs]\n"
            "    gy = [100*(1+R['gld_cagr_ye']/100)**(y-2015) for y in yrs]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "ax.plot(yrs, lv, 'o-', c=AMBER, lw=2.2, label=f\"Birkin  ({R['idx_cagr']:.0f}%/yr)\")\n"
            "ax.plot(sx, sy, 's-', c=GREEN, lw=2.2, label=f\"S&P 500  ({R['spy_cagr_ye']:.0f}%/yr)\")\n"
            "ax.plot(gx, gy, '^-', c=GOLD, lw=2.2, label=f\"gold  ({R['gld_cagr_ye']:.0f}%/yr)\")\n"
            "ax.set_xlabel('year-end'); ax.set_ylabel('$100 invested at end-2015')\n"
            "ax.set_title('Birkin vs the S&P vs gold: the bag comes last'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"Birkin CAGR ~{R['idx_cagr']:.1f}%  |  SPY ~{R['spy_cagr_ye']:.1f}%  |  gold ~{R['gld_cagr_ye']:.1f}%\")"
        ),
        md(
            f"Not close. Both the S&P (**~{R['spy_cagr_ye']:.0f}%/yr**) and gold (**~{R['gld_cagr_ye']:.0f}"
            f"%/yr**) roughly *tripled* your money; the bag barely more than *doubled*. The headline had "
            "it exactly backwards — over the decade you can actually measure, the Birkin came **last** "
            "of the three."
        ),
        md(
            "**\"Fine — I'll buy the luxury *stocks*.\"** You can: Hermès itself (the maker of the "
            "Birkin), plus LVMH and Kering. Do they hand you the bag trade — or just a wilder ride?"
        ),
        code(
            "names = ['Hermès\\n(RMS.PA)', 'LVMH\\n(MC.PA)', 'Kering\\n(KER.PA)', 'S&P 500']\n"
            "cagrs = [R['rms_cagr'], R['mc_cagr'], R['ker_cagr'], R['spy_cagr']]\n"
            "mdds  = [R['rms_mdd'], R['mc_mdd'], R['ker_mdd'], R['spy_mdd']]\n"
            "x = np.arange(4); fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, cagrs, .4, color=[GREEN,AMBER,RED,GREEN], label='CAGR %/yr')\n"
            "ax.bar(x+.2, mdds, .4, color=RED, alpha=.55, label='worst drawdown %')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels(names)\n"
            "ax.set_ylabel('percent'); ax.set_title('The buyable proxies: one winner, and a lot of risk'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"RMS.PA: CAGR {R['rms_cagr']:.1f}% (beat SPY!) but maxDD {R['rms_mdd']:.0f}%\")\n"
            "print(f\"KER.PA: CAGR {R['ker_cagr']:.1f}%, maxDD {R['ker_mdd']:.0f}% — luxury beta can bite hard\")"
        ),
        md(
            f"Here's the honest twist: **Hermès the *stock* did beat the S&P** (~{R['rms_cagr']:.0f}%/yr) "
            f"— but Kering **crashed {R['ker_mdd']:.0f}%**, and none of these is *the bag*. Buying the "
            "maison is a bet on a whole luxury company (Gucci's slump sank Kering), not on your Birkin's "
            "resale value. We'll put a *t*-stat on that Hermès out-performance next door."
        ),
        md(
            "**The part the pitch never mentions: it costs real money to flip a Birkin.** You buy at a "
            "premium and sell through consignment at a ~30% haircut, plus insurance and authentication. "
            "Charge that against the bag's gross return:"
        ),
        code(
            "labels = ['gross\\nreturn', 'consignment\\nspread', 'insurance\\n& carry', 'NET to\\nyou']\n"
            "vals = [R['carry_gross'], R['carry_spread'], R['carry_insure'], R['carry_net']]\n"
            "cols = [AMBER, RED, RED, (RED if R['carry_net']<0 else GREEN)]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('% per year')\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_title('Where the Birkin \"return\" goes once you actually transact')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"gross {R['carry_gross']:+.1f}%/yr  ->  NET {R['carry_net']:+.1f}%/yr after 30% spread + carry\")"
        ),
        md(
            f"There it is. A gross **{R['carry_gross']:.1f}%/yr** that already lost to stocks and gold "
            f"turns **negative ({R['carry_net']:.1f}%/yr)** the moment you pay the consignment spread to "
            "buy and sell the physical bag. A savings account beat it."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The bag returned ~{R['idx_cagr']:.0f}%/yr vs ~{R['spy_cagr_ye']:.0f}% "
            f"(S&P) and ~{R['gld_cagr_ye']:.0f}% (gold) — it *lost* to both. No evidence it beats "
            "stocks or gold; the one leg that came close, Hermès's *stock*, isn't the handbag.\n"
            "- **Tradability — Mirage.** Illiquid, high-spread; the gross return goes **negative** after "
            "the ~30% consignment haircut, and the buyable proxies are single-stock luxury beta.\n"
            "- **Beats stocks & gold? — Busted.** On return and net-of-cost the bag loses to both the "
            "assets the headline names. It's a fine store of value, not a return machine."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the honest bottom line\n\n"
            "Imagine three people at the end of 2015, each with \\$15,000 (roughly one entry Birkin). "
            "One buys an S&P index fund, one buys gold, one becomes a bag flipper — buys, insures, and "
            "eventually consigns it, paying the real ~30% spread. Where do they land by end-2025?"
        ),
        code(
            "start = 15_000.0\n"
            "spy_end = start*(1+R['spy_cagr_ye']/100)**10\n"
            "gld_end = start*(1+R['gld_cagr_ye']/100)**10\n"
            "bag_end = start*(1+R['carry_net']/100)**10   # net of spread + carry\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.bar(['S&P index fund', 'gold ETF', 'Birkin flipper\\n(net of costs)'],\n"
            "       [spy_end, gld_end, bag_end], color=[GREEN, GOLD, RED], width=.6)\n"
            "for i,v in enumerate([spy_end, gld_end, bag_end]): ax.annotate(f'${v:,.0f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('value of $15,000 after 10 years')\n"
            "ax.set_title('Same $15k, end-2015 -> end-2025'); plt.tight_layout(); plt.show()\n"
            "print(f'S&P: ${spy_end:,.0f}   gold: ${gld_end:,.0f}   Birkin (net): ${bag_end:,.0f}')"
        ),
        md(
            "The index-fund and gold buyers roughly **triple** their money doing nothing. The flipper — "
            "after the consignment spread and carry — ends up with **less than they started with**. The "
            "people who really \"made money on a Birkin\" bought a rare Himalaya at retail and sold it at "
            "auction: a lottery ticket on scarcity, not an asset class you can systematically own."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Pull a real handbag index.** Our resale line is a cited *approximation*; the Knight "
            "Frank Luxury Investment Index and Art Market Research publish handbag series. Swap one in — "
            "the shape (and the verdict) won't move, but you'll have the exact tape.\n"
            "- **The one Birkin that *did* win.** Auction-grade Himalaya / diamond-hardware bags have "
            "smashed records — but that's a thin, selected tail (survivorship), not the steel-Daytona-"
            "of-handbags you can actually source. Test one reference and correct for selection.\n"
            "- **The collectibles pattern.** Watches, wine, art, sneakers: every \"passion asset\" tells "
            "the same story — real steadiness, brutal carry, equities win net of cost (see "
            "[Study 358 — Watch-Index](../../358-watch-index/) and "
            "[docs/references.md](../docs/references.md)).\n\n"
            "*Think a specific reference (a Himalaya, a diamond Kelly) beat the S&P net of every cost? "
            "Pull its auction history, charge the consignment spread, and show it — then check it wasn't "
            "just one lucky lot near a record.*"
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
            "# A Birkin vs the S&P and gold — a quantitative teardown 🔬\n"
            "### Resale index vs SPY & GLD (CAGR / vol / MDD + annual-excess *t*) · Newey-West maison "
            "alpha · the ~30% consignment haircut on NAV · a synthetic compounder positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We test the "
            "strongest tradable form of \"a Birkin beats the S&P and gold\": (H₁) the secondary-market "
            "resale index out-returns SPY *and* GLD risk-adjusted; (H₂) a buyable maison proxy carries "
            "alpha vs the market; (H₃) it survives the transaction + carry cost of owning a physical "
            "bag. We find **H₁ rejected** (it *under*-performs both), **H₂ not established** (the best "
            "leg, RMS.PA, is *t* = 1.93 < 2), **H₃ rejected** (negative net of the consignment spread).\n\n"
            "> ⚠️ **Not investment advice — data provenance.** The Birkin resale index is **hardcoded, "
            "cited, approximate** (Baghunter 2016 headline + Knight Frank Luxury Investment Index + press "
            "reporting — a *labelled proxy*, never the live feed). Equity/gold proxies `RMS.PA`, `MC.PA`, "
            "`KER.PA`, `SPY`, `GLD` are month-end Adj Close via yfinance (as-of 2025-12-31). Offline core "
            "+ synthetic control are deterministic. Methods in [`docs/references.md`](../docs/references.md); "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Resale index CAGR **{R['idx_cagr']:.1f}%** vs SPY "
            f"**{R['spy_cagr_ye']:.1f}%** and GLD **{R['gld_cagr_ye']:.1f}%** (2015–2025); annual excess "
            f"**{R['excess_spy_mean']:+.1f}%/yr** vs SPY (*t* = **{R['excess_spy_t']:+.2f}**), "
            f"**{R['excess_gld_mean']:+.1f}%/yr** vs GLD (*t* = **{R['excess_gld_t']:+.2f}**), n={R['excess_n']}. "
            f"Best maison alpha (RMS.PA) NW *t* = **{R['rms_t']:+.2f}** — under the bar. |\n"
            f"| **Tradability** | `MIRAGE` | Gross index CAGR **{R['carry_gross']:+.1f}%** → **NET "
            f"{R['carry_net']:+.1f}%/yr** after a 30% consignment spread + carry. Buyable proxies are "
            f"single-stock luxury beta (KER.PA maxDD **{R['ker_mdd']:.0f}%**). Illiquid, wide-spread. |\n"
            f"| **Beats stocks & gold?** | `BUSTED` | The bag loses the CAGR race to **both** SPY "
            f"({R['spy_cagr_ye']:.1f}) and GLD ({R['gld_cagr_ye']:.1f}), and net-of-cost goes negative. "
            "Every column the headline names. |\n\n"
            "> 💡 In plain words: the handbag is a lovely low-vol *store of value* but a poor *return* — "
            "it trailed both stocks and gold, the only tradable expression is single-name luxury beta, "
            "and the ~30% consignment friction turns even the gross return negative. The famous "
            f"{R['myth_cagr']:.0f}%/yr is a cherry-picked, survivorship-laden 1980–2015 marketing figure."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let the secondary-market Birkin level be $I_t$ and the benchmarks $B_t$ (SPY) and $G_t$ "
            "(GLD). The claim is a joint hypothesis:\n\n"
            "- **H₁ (it out-returns).** Annual excess $\\;x_t = r^I_t - r^B_t\\;$ (and vs gold) has "
            "$\\mathbb{E}[x_t] > 0$ with $t > 2$ — the bag beats stocks *and* gold, risk-adjusted.\n"
            "- **H₂ (it's buyable with alpha).** For a tradable maison proxy $P$, the intercept "
            "$\\alpha$ in $r^P_t = \\alpha + \\beta r^B_t + \\varepsilon_t$ is positive with a "
            "Newey-West *t* > 2.\n"
            "- **H₃ (it survives carry).** The net CAGR after the consignment round-trip spread $s$ over "
            "hold $h$ and annual carry $c$ stays positive: $(1+g)(1+((1-s)^{1/h}-1))(1-c)-1 > 0$.\n\n"
            "The Baghunter 2016 figure ($\\sim$14.2%/yr, 1980–2015, \"never a down year\") is the "
            "steelman. The test is whether it holds over a **modern, measurable, risk-adjusted, net-of-"
            "cost** window — i.e. whether it's an asset class or a marketing statistic."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "If H₁–H₃ held, the Birkin would be a genuine diversifier: it would beat *both* the growth "
            "asset (equities) *and* the classic inflation hedge (gold), wearable, near-zero drawdown — "
            "the exact pitch. But each leg is separately falsifiable. H₁ is a **return race** on a "
            "common clock against two named benchmarks. H₂ asks whether the *only investable expression* "
            "(listed maisons) delivers anything beyond market beta — you cannot custody \"the index.\" "
            "H₃ is the **microstructure tax**: a physical bag has a ~30% consignment/dealer round-trip "
            "and a multi-year hold with insurance and authentication carry, frictions an ETF never pays. "
            "The asset-class claim needs all three; failing any one downgrades it to folklore."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Resale index (proxy).** A hardcoded, cited, **approximate** annual level (base 100 @ "
            "2015), reconstructed from public reporting (Hermès primary hikes; the 2020–22 resale melt-"
            "up; the 2023–24 luxury-handbag *cooling* Knight Frank flagged). *Labelled a proxy* — its "
            "path is defensible, its precise year-end values are not a live feed.\n"
            "- **Equity/gold proxies.** `RMS.PA`, `MC.PA`, `KER.PA`, `SPY`, `GLD` month-end Adj Close "
            "(yfinance, cached). Survivorship is **not** a concern here (named tickers, not a screen); "
            "the resale index's *level* survivorship — bags that stopped trading drop out — is named, "
            "and it biases the claim **upward**.\n"
            "- **Signal test.** (i) Paired annual-excess $t$ of $r^I - r^B$ vs SPY *and* GLD (small-$n$, "
            "weak by construction). (ii) **Newey-West (6-lag) HAC** $t$ of each maison alpha vs SPY — "
            "the bar for `REAL` is *t* ≥ 2 in the bag's favour.\n"
            "- **Cost (beat 6).** Charge the consignment round-trip spread (30% over a 3y hold) + "
            "0.5%/yr carry **once on NAV**; net CAGR.\n"
            "- **Positive control.** A deterministic *steady-compounder* path with a *planted* drift; the "
            "engine must recover the up-sign and a finite Sharpe — proof a null on the real tape is a "
            "real null, not a broken harness.\n"
            "- **What would make us say \"asset class\":** H₁ *t* > 2 vs **both** benchmarks **or** a "
            "maison alpha *t* > 2, **and** a positive net-of-cost CAGR. We find none of these."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The race — resale index vs SPY *and* gold, risk-adjusted\n\n"
            "Year-end levels rebased to \\$100, all three on one clock. CAGR, vol and max-drawdown in "
            "the print; the paired annual-excess *t* against each benchmark."
        ),
        code(
            "yrs = list(R['idx_levels'].keys())\n"
            "lv = np.array([float(IDX.loc[f'{y}-12-31']) for y in yrs])\n"
            "si = st.summarize(IDX, periods_per_year=1.0)\n"
            "if HAVE_PROXIES:\n"
            "    def ye(t):\n"
            "        s = PROX[t].resample('YE').last(); return s[(s.index.year>=2015)&(s.index.year<=2025)]\n"
            "    spy_ye, gld_ye = ye('SPY'), ye('GLD')\n"
            "    ss, sg = st.summarize(spy_ye, 1.0), st.summarize(gld_ye, 1.0)\n"
            "    aes, aeg = st.annual_excess_t(IDX, PROX['SPY']), st.annual_excess_t(IDX, PROX['GLD'])\n"
            "    spy = (spy_ye/spy_ye.iloc[0]*100).values; gld = (gld_ye/gld_ye.iloc[0]*100).values\n"
            "    sx = [d.year for d in spy_ye.index]\n"
            "else:\n"
            "    spy = np.array([100*(1+R['spy_cagr_ye']/100)**(y-2015) for y in yrs])\n"
            "    gld = np.array([100*(1+R['gld_cagr_ye']/100)**(y-2015) for y in yrs]); sx=yrs\n"
            "    si={'cagr':R['idx_cagr']/100,'vol':R['idx_vol']/100,'mdd':R['idx_mdd']/100}\n"
            "    ss={'cagr':R['spy_cagr_ye']/100,'vol':R['spy_vol_ye']/100,'mdd':R['spy_mdd_ye']/100}\n"
            "    sg={'cagr':R['gld_cagr_ye']/100,'vol':R['gld_vol_ye']/100,'mdd':R['gld_mdd_ye']/100}\n"
            "    aes={'mean_excess':R['excess_spy_mean']/100,'t':R['excess_spy_t'],'n':R['excess_n']}\n"
            "    aeg={'mean_excess':R['excess_gld_mean']/100,'t':R['excess_gld_t'],'n':R['excess_n']}\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.5))\n"
            "ax.plot(yrs, lv, 'o-', c=AMBER, lw=2.2, label=f\"Birkin  CAGR {si['cagr']*100:.1f}%, vol {si['vol']*100:.0f}%\")\n"
            "ax.plot(sx, spy, 's-', c=GREEN, lw=2.2, label=f\"SPY  CAGR {ss['cagr']*100:.1f}%, vol {ss['vol']*100:.0f}%\")\n"
            "ax.plot(sx, gld, '^-', c=GOLD, lw=2.2, label=f\"GLD  CAGR {sg['cagr']*100:.1f}%, vol {sg['vol']*100:.0f}%\")\n"
            "ax.set_xlabel('year-end'); ax.set_ylabel('rebased to 100 @ 2015')\n"
            "ax.set_title('H1: the Birkin index UNDER-performs both SPY and gold'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"Birkin: CAGR {si['cagr']*100:.2f}%  vol {si['vol']*100:.1f}%  maxDD {si['mdd']*100:.1f}%\")\n"
            "print(f\"SPY   : CAGR {ss['cagr']*100:.2f}%  vol {ss['vol']*100:.1f}%  maxDD {ss['mdd']*100:.1f}%\")\n"
            "print(f\"GLD   : CAGR {sg['cagr']*100:.2f}%  vol {sg['vol']*100:.1f}%  maxDD {sg['mdd']*100:.1f}%\")\n"
            "print(f\"excess idx-SPY: {aes['mean_excess']*100:+.2f}%/yr  t={aes['t']:+.3f}  (n={aes['n']})\")\n"
            "print(f\"excess idx-GLD: {aeg['mean_excess']*100:+.2f}%/yr  t={aeg['t']:+.3f}  (n={aeg['n']})\")"
        ),
        md(
            f"> 💡 In plain words: the Birkin compounds at **{R['idx_cagr']:.1f}%** against SPY's "
            f"**{R['spy_cagr_ye']:.1f}%** and gold's **{R['gld_cagr_ye']:.1f}%** — it loses *both* races. "
            f"The mean annual excess is **{R['excess_spy_mean']:+.1f}%/yr** vs SPY (*t* = "
            f"**{R['excess_spy_t']:+.2f}**) and **{R['excess_gld_mean']:+.1f}%/yr** vs gold (*t* = "
            f"**{R['excess_gld_t']:+.2f}**), n={R['excess_n']}. Its saving grace is *low vol* — a genuine "
            "store-of-value trait — but on **return**, H₁ is rejected against both benchmarks. The "
            "honest stamp is `NONE`, leaning negative."
        ),
        md(
            "### 4b · The buyable maisons — is there alpha, or just luxury beta?\n\n"
            "Newey-West (6-lag) regression of each maison's **monthly** return on SPY. `REAL` needs "
            "$t_\\alpha \\ge 2$ — the bar the *bag* itself never approaches, so this is the claim's best "
            "shot."
        ),
        code(
            "if HAVE_PROXIES:\n"
            "    spy_r = PROX['SPY'].pct_change().dropna()\n"
            "    rows = {}\n"
            "    for t in ['RMS.PA','MC.PA','KER.PA']:\n"
            "        s = st.summarize(PROX[t]); nw = st.newey_west_alpha_t(PROX[t].pct_change().dropna(), spy_r, 6)\n"
            "        rows[t] = dict(cagr=s['cagr']*100, sharpe=s['sharpe'], mdd=s['mdd']*100,\n"
            "                       alpha=nw['alpha_ann']*100, beta=nw['beta'], t=nw['t_alpha'])\n"
            "else:\n"
            "    rows = {'RMS.PA':dict(cagr=R['rms_cagr'],sharpe=R['rms_sharpe'],mdd=R['rms_mdd'],alpha=R['rms_alpha'],beta=R['rms_beta'],t=R['rms_t']),\n"
            "            'MC.PA':dict(cagr=R['mc_cagr'],sharpe=R['mc_sharpe'],mdd=R['mc_mdd'],alpha=R['mc_alpha'],beta=R['mc_beta'],t=R['mc_t']),\n"
            "            'KER.PA':dict(cagr=R['ker_cagr'],sharpe=R['ker_sharpe'],mdd=R['ker_mdd'],alpha=R['ker_alpha'],beta=R['ker_beta'],t=R['ker_t'])}\n"
            "labels=list(rows); alphas=[rows[t]['alpha'] for t in labels]; ts=[rows[t]['t'] for t in labels]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "cols=[GREEN if a>0 else RED for a in alphas]\n"
            "ax.bar(labels, alphas, .5, color=cols)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised alpha vs SPY (%)')\n"
            "for i,t in enumerate(labels): ax.annotate(f\"t={ts[i]:+.2f}\",(i,alphas[i]),ha='center',va='bottom' if alphas[i]>=0 else 'top')\n"
            "ax.set_title('H2: even the best maison (RMS.PA) is t=1.93 < 2 — not established')\n"
            "plt.tight_layout(); plt.show()\n"
            "for t in labels: r=rows[t]; print(f\"{t:7s} CAGR {r['cagr']:6.2f}%  Sharpe {r['sharpe']:.2f}  maxDD {r['mdd']:6.1f}%  alpha {r['alpha']:+.2f}%/yr  beta {r['beta']:.2f}  NW t {r['t']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: this is the claim's strongest card, and it's still short. Hermès "
            f"(**RMS.PA**) *did* out-return SPY with a **{R['rms_alpha']:+.0f}%/yr** alpha — but at "
            f"Newey-West **t = {R['rms_t']:+.2f}**, just **under** the *t* ≥ 2 bar, so `REAL` is not "
            f"earned (it reads `WEAK` at most). LVMH is insignificant (*t*={R['mc_t']:+.2f}) and Kering "
            f"is *negative* (*t*={R['ker_t']:+.2f}, maxDD **{R['ker_mdd']:.0f}%**). And crucially: this "
            "is the *equity of the company*, driven by watches, leather, beauty, fashion and buybacks — "
            "**not the resale price of a Birkin**. H₂ is not established."
        ),
        md(
            "### 4c · The microstructure tax — net of the consignment spread + carry\n\n"
            "A physical flip pays a ~30% round-trip spread over a ~3-year hold plus ~0.5%/yr carry. "
            "Charge it once on the index's gross CAGR (a *generous* read — it ignores that the quoted "
            "resale level is a dealer ask, not net-to-seller)."
        ),
        code(
            "gross = si['cagr']\n"
            "h = st.net_of_carry_cagr(gross, round_trip_spread=0.30, hold_years=3.0, insure_per_year=0.005)\n"
            "steps = ['gross','after\\nspread','after\\ncarry']\n"
            "running = [h['gross_cagr']*100,\n"
            "           ((1+h['gross_cagr'])*(1+h['spread_drag_annual'])-1)*100,\n"
            "           h['net_cagr']*100]\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.3))\n"
            "cols=[AMBER, AMBER, (RED if running[-1]<0 else GREEN)]\n"
            "ax.bar(steps, running, .55, color=cols)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(R['spy_cagr_ye'], ls='--', c=GREEN, alpha=.6, label='SPY CAGR')\n"
            "ax.axhline(R['gld_cagr_ye'], ls='--', c=GOLD, alpha=.6, label='GLD CAGR')\n"
            "for i,v in enumerate(running): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_ylabel('CAGR %/yr'); ax.set_title('H3: net of real frictions, the Birkin return is NEGATIVE'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"gross {h['gross_cagr']*100:+.2f}%  - spread {h['spread_drag_annual']*100:+.2f}%/yr  - carry {h['insure_per_year']*100:+.2f}%/yr  =  NET {h['net_cagr']*100:+.2f}%/yr\")"
        ),
        md(
            f"> 💡 In plain words: the gross **{R['carry_gross']:.1f}%** — already a loser to both SPY and "
            f"gold — goes **negative ({R['carry_net']:.1f}%/yr)** once you pay the ~30% consignment "
            "spread and carry to own the physical bag. And this is *charitable*: a quoted resale level is "
            "a dealer ask, so a real seller's realised return is lower still. H₃ rejected. **MIRAGE** is "
            "the only honest stamp."
        ),
        md(
            "### 4d · Positive control — the engine recovers a planted compounder\n\n"
            "A deterministic steady-compounder (planted +12% CAGR, σ=3%/mo, seed 711) — the \"it only "
            "goes up\" path the pitch imagines. The harness must recover the up-sign and a finite Sharpe, "
            "proving the nulls above are real, not a broken pipeline."
        ),
        code(
            "syn = data.synthetic_compounder()\n"
            "s = st.summarize(syn); cr = st.control_recovers(syn, planted_sign=1)\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.2))\n"
            "ax.plot(syn.index, syn.values, c=GREY, lw=2)\n"
            "ax.axhline(100, ls=':', c=GREY)\n"
            "ax.set_ylabel('synthetic level'); ax.set_title('Planted compounder: engine recovers sign + Sharpe (machinery proof)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"peak {syn.max():.0f} -> end {syn.iloc[-1]:.0f}  recovered CAGR {s['cagr']*100:+.2f}%  Sharpe {s['sharpe']:.2f}  maxDD {s['mdd']*100:.1f}%  sign_ok={cr['sign_ok']}\")"
        ),
        md(
            "> 💡 In plain words: the engine banks the planted signal (recovered CAGR ~+11%, Sharpe ~1.0, "
            "sign correct). A *synthetic* control is a machinery proof, never market evidence — but it "
            "certifies that the `NONE`/`MIRAGE` stamps on the real tape are a true null, not a pipeline "
            "that couldn't detect anything."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — Birkin index CAGR {R['idx_cagr']:.1f}% vs SPY {R['spy_cagr_ye']:.1f}% "
            f"and GLD {R['gld_cagr_ye']:.1f}%; annual excess {R['excess_spy_mean']:+.1f}%/yr (*t* = "
            f"{R['excess_spy_t']:+.2f}) vs SPY and {R['excess_gld_mean']:+.1f}%/yr (*t* = "
            f"{R['excess_gld_t']:+.2f}) vs gold. The single closest leg, RMS.PA equity, is NW *t* = "
            f"{R['rms_t']:+.2f} < 2 — no robust *t* ≥ 2 anywhere in the bag's favour, and it isn't the bag.\n"
            f"- **Tradability `MIRAGE`** — gross {R['carry_gross']:+.1f}% → net **{R['carry_net']:+.1f}%/yr** "
            f"after a 30% consignment spread + carry; the buyable proxies are single-stock luxury beta "
            f"(KER.PA maxDD {R['ker_mdd']:.0f}%). Illiquid, wide-spread, no scalable book.\n"
            f"- **Beats stocks & gold? `BUSTED`** — the bag loses the return race to **both** named "
            f"benchmarks and goes negative net-of-cost. The famous {R['myth_cagr']:.0f}%/yr is a "
            "cherry-picked 1980–2015 window with survivorship baked in — a marketing statistic, not a "
            "tradable asset class."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the capacity & cost reality\n\n"
            "Terminal wealth of \\$15,000 invested end-2015 (≈ one entry Birkin): S&P index fund vs gold "
            "ETF vs a bag flipper paying the real frictions (net CAGR from 4c). Capacity is the other "
            "wall: each bag is a bespoke, illiquid, ~30%-spread consignment — there is no scalable book."
        ),
        code(
            "start=15_000.0; yrs_h=10\n"
            "paths={'S&P index fund':R['spy_cagr_ye']/100, 'gold ETF':R['gld_cagr_ye']/100,\n"
            "       'Birkin flipper (net)':R['carry_net']/100,\n"
            "       'Birkin index (gross, untradable)':R['idx_cagr']/100}\n"
            "fig, ax = plt.subplots(figsize=(9.4,4.4))\n"
            "labels=list(paths); ends=[start*(1+g)**yrs_h for g in paths.values()]\n"
            "cols=[GREEN, GOLD, RED, AMBER]\n"
            "ax.bar(labels, ends, .6, color=cols)\n"
            "for i,v in enumerate(ends): ax.annotate(f'${v:,.0f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_xticklabels(labels, rotation=12, ha='right')\n"
            "ax.set_ylabel('value of $15,000 after 10 years'); ax.set_title('Net of cost, the flipper ends below where they started')\n"
            "plt.tight_layout(); plt.show()\n"
            "for l,g in paths.items(): print(f\"{l:34s} ${start*(1+g)**yrs_h:>10,.0f}  ({g*100:+.1f}%/yr)\")"
        ),
        md(
            "> 💡 In plain words: even the *gross, untradable* index trails both the S&P and gold; the "
            "*tradable* version (net of the consignment spread + carry) ends **below the starting stake**. "
            "And capacity is fatal — a Birkin flip is a one-off illiquid trade with a ~30% round-trip, "
            "the antithesis of a scalable strategy. There is no sizing or venue that turns this into an "
            "edge."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Swap in a live handbag index.** Replace the hardcoded series with the Knight Frank "
            "Luxury Investment Index handbag component or an Art Market Research handbag index and re-"
            "run; the *t*-stats sharpen but the sign won't flip.\n"
            "- **Per-reference dispersion.** The aggregate hides survivorship: test individual references "
            "(Himalaya, diamond Kelly, standard Togo 30) — the record-setters are a thin selected tail, "
            "the bias points *for* the claim, so correct for it.\n"
            "- **The collectibles prior.** Dimson–Spaenjers and the emotional-assets literature: "
            "collectibles under-perform equities net of carry across the board (see "
            "[Study 358 — Watch-Index](../../358-watch-index/) and "
            "[docs/references.md](../docs/references.md)). Handbags are not the exception.\n\n"
            "*The reproducible core is offline and deterministic; the resale index is a **cited, "
            "approximate proxy** and the equity/gold tickers are **labelled proxies** for the trade. "
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
