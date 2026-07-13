"""Generate the two narrative notebooks for Study 715 ("vinyl is back — a trend to trade?").

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic: real-tape cells read the cached month-end
yfinance pulls under ../_cache/ for the equity proxies and the hardcoded (cited,
approximate) RIAA vinyl-revenue index from the package; on a cache miss they fall back to
the frozen headline numbers in ``R`` (mirroring docs/results.md). The synthetic revival
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


# Frozen headline numbers — mirror of docs/results.md (vinyl index hardcoded/cited/approx;
# equity proxies month-end Adj Close via yfinance, as-of 2025-12-31).
R = dict(
    vinyl_win="2010 → 2024",
    vinyl_musd={2010: 89, 2014: 321, 2018: 419, 2019: 479, 2020: 643, 2021: 1003,
                2022: 1225, 2023: 1353, 2024: 1400},
    vinyl_yoy={2019: 14.3, 2020: 34.2, 2021: 56.0, 2022: 22.1, 2023: 10.4, 2024: 3.5},
    vinyl_cagr=21.75, vinyl_vol=19.5, vinyl_sharpe=1.19, vinyl_mdd=-8.1,
    spy_cagr_ye=17.18, spy_vol_ye=16.9, spy_mdd_ye=-18.2,
    excess_mean=4.91, excess_t=0.456, excess_p=0.667, excess_n=6,
    vinyl_share=7.0, streaming_share=84.0,
    wmg_cagr=2.76, wmg_vol=36.4, wmg_sharpe=0.25, wmg_mdd=-52.2,
    wmg_alpha=-11.52, wmg_beta=1.24, wmg_t=-1.23, wmg_p=0.223, wmg_n=66,
    spot_cagr=18.14, spot_vol=46.8, spot_sharpe=0.58, spot_mdd=-74.9,
    spot_alpha=1.98, spot_beta=1.65, spot_t=0.12, spot_p=0.908, spot_n=92,
    umg_cagr=1.30, umg_vol=27.3, umg_sharpe=0.18, umg_mdd=-24.2,
    umg_alpha=-6.44, umg_beta=0.85, umg_t=-0.66, umg_p=0.513, umg_n=51,
    spy_cagr=13.59, spy_vol=16.5, spy_sharpe=0.86, spy_mdd=-23.9,
    carry_gross=21.75, carry_spread=-6.89, carry_storage=-1.00, carry_net=12.23,
    carry_flat_net=-7.82,
    syn_peak=596, syn_end=558, syn_cagr=18.86, syn_sharpe=1.28, syn_mdd=-15.3,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Already_priced_in%3F: Confirmed](https://img.shields.io/badge/Already_priced_in%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from vinyl_revival import data, strategy as st

HAVE_PROXIES = data.have_proxies()
IDX = data.load_vinyl_index()                        # hardcoded, cited, APPROXIMATE proxy
REV = data.vinyl_revenue_musd()                      # raw RIAA $M
PROX = data.load_proxies() if HAVE_PROXIES else None
print("equity-proxy cache present:", HAVE_PROXIES,
      "| vinyl-revenue years:", REV.index[0].year, "->", REV.index[-1].year)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Vinyl is back — can you *trade* it? 🎵\n"
            "### The \"records are back, ride the boom\" story, in plain English\n\n"
            + BADGES +
            "You've heard it in every record shop and half the finance podcasts: *\"vinyl is "
            "**back**. Sales go up every single year. Forget stocks — this is a trend you can "
            "ride.\"* And here's the twist most teardowns miss: **the trend is completely real.** "
            "U.S. vinyl revenue has climbed for eighteen straight years, from a near-dead $89M in "
            "2010 to about **$1.4 billion** in 2024. Vinyl outsells CDs now.\n\n"
            "So this one is different. The usual question — *\"is the trend real?\"* — the answer is "
            "**yes**. The real question is the sneaky one: *if a trend is real, does that mean you "
            "can make money on it?* This notebook lines the vinyl boom up next to the S&P 500 and "
            "asks what happens when you actually try to **buy** it.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the Newey-West alpha and the cost "
            "algebra? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice — and a data note.** The RIAA revenue database isn't free "
            "to pull, so the vinyl line below is a **small, clearly-cited, approximate** "
            "reconstruction of public RIAA year-end reports — a **proxy for the trend**, and it's "
            "*industry revenue*, never a price you can buy. Every chart is drawn by the code beside "
            "it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is the vinyl revival real? | **Yes — completely.** Revenue climbed from "
            f"\\${R['vinyl_musd'][2010]}M (2010) to ~\\${R['vinyl_musd'][2024]/1000:.1f}B (2024), "
            f"~**{R['vinyl_cagr']:.0f}%/yr** — 18 straight years. It even *out-grew* the S&P. |\n"
            "| So you can trade it, right? | **No — you can't buy the trend.** The RIAA revenue "
            "series isn't a stock or an ETF. Nobody pays it out. |\n"
            "| What *can* you buy? | The record labels & streamers. And two of the three "
            f"(**Warner {R['wmg_cagr']:.1f}%/yr**, **Universal {R['umg_cagr']:.1f}%**) badly "
            f"**lagged** the S&P (**{R['spy_cagr']:.1f}%**). The one that beat it (Spotify) did it "
            "on *streaming*, not vinyl. |\n"
            "| Could you flip records yourself? | **Not profitably.** A revived market presses "
            "*more* records, so your copy doesn't get rarer — and once you pay the marketplace fees "
            f"to sell, a collector nets **negative ({R['carry_flat_net']:.1f}%/yr)**. |\n\n"
            "> The boom is real. The *trade* is a mirage. A famous 18-year trend is not a secret."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Vinyl is back and it's not stopping. Records outsell CDs, pressing plants can't "
            "keep up, every artist drops a coloured-vinyl variant. This is a real, growing market — "
            "get exposure and ride the trend.\"*\n\n"
            "It's a *steelman-able* claim, and unusually so: the underlying trend genuinely exists. "
            "The RIAA — the industry's own scorekeeper — has vinyl revenue rising every year since "
            f"2006, past **\\${R['vinyl_musd'][2021]/1000:.1f}B in 2021** and to ~**\\${R['vinyl_musd'][2024]/1000:.1f}B "
            "in 2024**. Vinyl passed CDs in revenue in 2020 and in units in 2022. Unlike most \"asset "
            "class\" stories, nobody has to squint to see the growth. That's exactly what makes it a "
            "good test: **a real trend is the strongest possible case for \"you can trade it.\"**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a real, visible, years-long trend were also a *tradable* one, investing would be "
            "easy: read the headlines, buy the theme, collect. That's the whole premise of "
            "\"thematic\" investing. But \"vinyl revenue went up 20%/yr\" and \"you can earn 20%/yr "
            "on vinyl\" are wildly different statements. The first is about an **industry's sales**; "
            "the second is a claim about a **priced, buyable, net-of-cost return**. The gap between "
            "them is where most theme-chasing money quietly dies — and vinyl is a clean place to "
            "watch it happen."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three honest comparisons, each against the S&P 500 (SPY):\n\n"
            "1. **The revenue trend vs SPY.** Put the (cited, approximate) RIAA vinyl-revenue line "
            "next to SPY — did the *trend* out-grow stocks? (Spoiler: yes. Hold that thought.)\n"
            "2. **The thing you can actually buy.** You can't buy \"vinyl revenue.\" You *can* buy "
            "**Warner Music**, **Universal Music** and **Spotify**. Do they hand you the vinyl "
            "trade's growth — or just stock-market risk?\n"
            "3. **The cost of collecting.** The only way to *hold* vinyl directly is to buy and sell "
            "physical records. Charge the marketplace spread and storage, and see what's left.\n\n"
            "**What would make us say \"tradable\"?** A listed proxy that beats the S&P *because of "
            "vinyl* (real alpha), or a collector strategy that clears its costs. Anything less is a "
            "real trend you simply can't monetise."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: the trend is real.** Here is the (approximate, cited) RIAA vinyl-revenue line "
            "— an 18-year climb, no bubble-and-bust, just up."
        ),
        code(
            "yrs = [int(y) for y in REV.index.year]\n"
            "rev = [float(v) for v in REV.values]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "ax.plot(yrs, rev, 'o-', c=AMBER, lw=2.4, label='U.S. vinyl revenue (RIAA, proxy)')\n"
            "ax.axhline(643, ls=':', c=GREY)\n"
            "ax.annotate('2020: vinyl passes CDs\\nin revenue', (2020, 643), textcoords='offset points',\n"
            "            xytext=(8, 10), color=GREY, fontsize=9)\n"
            "ax.set_xlabel('year'); ax.set_ylabel('revenue ($M, retail value)')\n"
            "ax.set_title('The vinyl revival is completely real: 18 years up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('revenue $M:', {y:int(v) for y,v in zip(yrs, rev)})\n"
            "print(f\"CAGR ~{R['vinyl_cagr']:.0f}%/yr, {yrs[0]}->{yrs[-1]}  (a real, durable trend)\")"
        ),
        md(
            f"No trick here: vinyl revenue compounded at ~**{R['vinyl_cagr']:.0f}%/yr** for a decade "
            "and a half. That's *faster* than the stock market. If \"a real trend you can trade\" "
            "were ever true, this is the case that should prove it. So let's try to buy it."
        ),
        md(
            "**\"Great — I'll buy the vinyl *stocks*.\"** You can't buy the revenue line, but you can "
            "buy the labels that press the records (Warner, Universal) and the streamer vinyl is "
            "supposedly beating (Spotify). Do they deliver the trend?"
        ),
        code(
            "names = ['Warner\\nMusic', 'Universal\\nMusic', 'Spotify', 'S&P 500']\n"
            "cagrs = [R['wmg_cagr'], R['umg_cagr'], R['spot_cagr'], R['spy_cagr']]\n"
            "mdds  = [R['wmg_mdd'], R['umg_mdd'], R['spot_mdd'], R['spy_mdd']]\n"
            "x = np.arange(4); fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "cols = [AMBER, AMBER, AMBER, GREEN]\n"
            "ax.bar(x-.2, cagrs, .4, color=cols, label='CAGR %/yr')\n"
            "ax.bar(x+.2, mdds, .4, color=RED, alpha=.55, label='worst drawdown %')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels(names)\n"
            "ax.set_ylabel('percent'); ax.set_title('The buyable proxies: two lagged the S&P, none is \"vinyl\"'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"Warner  {R['wmg_cagr']:+.1f}%/yr (maxDD {R['wmg_mdd']:.0f}%)  vs  S&P {R['spy_cagr']:+.1f}%/yr\")\n"
            "print(f\"Universal {R['umg_cagr']:+.1f}%/yr (maxDD {R['umg_mdd']:.0f}%)\")\n"
            "print(f\"Spotify {R['spot_cagr']:+.1f}%/yr but maxDD {R['spot_mdd']:.0f}% — that's streaming, not vinyl\")"
        ),
        md(
            f"Look at that. The two pure record labels — Warner (**{R['wmg_cagr']:.1f}%/yr**) and "
            f"Universal (**{R['umg_cagr']:.1f}%**) — **lagged** the S&P's **{R['spy_cagr']:.1f}%** "
            "badly, and with much deeper drops. Spotify beat the market, but Spotify is a *streaming* "
            "company — the exact thing vinyl is supposedly a rebellion against — and it fell "
            f"**{R['spot_mdd']:.0f}%** along the way. The 20%/yr vinyl growth is nowhere in what you "
            "can buy."
        ),
        md(
            "**Why?** Because vinyl is a rounding error inside these companies. Streaming is the whole "
            "game."
        ),
        code(
            "shares = [R['vinyl_share'], R['streaming_share'], 100 - R['vinyl_share'] - R['streaming_share']]\n"
            "labels = [f\"vinyl\\n{R['vinyl_share']:.0f}%\", f\"streaming\\n{R['streaming_share']:.0f}%\", 'other\\nformats']\n"
            "fig, ax = plt.subplots(figsize=(6.8, 4.6))\n"
            "ax.pie(shares, labels=labels, colors=[AMBER, GREEN, GREY], autopct='%1.0f%%',\n"
            "       startangle=90, wedgeprops=dict(width=.45))\n"
            "ax.set_title('U.S. recorded-music revenue mix (2024): vinyl is ~7%')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"vinyl ~{R['vinyl_share']:.0f}% of the pie; streaming ~{R['streaming_share']:.0f}%.\")\n"
            "print('A doubling of vinyl barely moves a major label whose P&L is streaming.')"
        ),
        md(
            f"Vinyl is ~**{R['vinyl_share']:.0f}%** of the industry; streaming is ~**{R['streaming_share']:.0f}%**. "
            "Even a *spectacular* vinyl year is a decimal-place event for Warner or Universal. The "
            "market prices them on streaming — so \"buy the label to ride vinyl\" is buying a "
            "battleship to feel a breeze."
        ),
        md(
            "**Last resort: collect the records yourself.** Surely *that* captures the boom? Here's "
            "the part the pitch never mentions — what it costs, and why a rising market works "
            "*against* you."
        ),
        code(
            "labels = ['revenue\\ngrowth\\n(fantasy)', 'realistic\\ngross\\n(~flat)', 'after fees\\n& storage\\n(realistic)']\n"
            "vals = [R['carry_gross'], 0.0, R['carry_flat_net']]\n"
            "cols = [GREY, AMBER, RED]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('% per year to a collector')\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_title('Flipping records: a revived market presses MORE of them')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"industry revenue grew {R['carry_gross']:+.1f}%/yr, but that's SALES, not YOUR record's price.\")\n"
            "print(f\"reissues expand supply -> per-record resale ~flat -> after spread+storage: {R['carry_flat_net']:+.1f}%/yr\")"
        ),
        md(
            "Here's the catch nobody says out loud: when a market *revives*, the factories press "
            "**more** copies. Your record doesn't get rarer — the opposite. So per-record resale is "
            "roughly flat, and once you pay the marketplace fee and the condition discount to sell, "
            f"the collector nets **negative ({R['carry_flat_net']:.1f}%/yr)**. The 20%/yr was the "
            "*industry's* sales, never your shelf."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The trend is real (revenue ~{R['vinyl_cagr']:.0f}%/yr) — but "
            "revenue isn't a return you can earn, and nothing you can *buy* delivers it (no "
            "significant alpha in any proxy).\n"
            "- **Tradability — Mirage.** Two of three label/streamer stocks lagged the S&P; the "
            "winner was streaming beta; flipping records nets negative. There's no way to cash the "
            "trend.\n"
            "- **Already priced in? — Confirmed.** Vinyl is ~7% of a streaming business, fully "
            "reflected (or too small to matter) in the tape. A famous 18-year trend is not an edge."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the honest bottom line\n\n"
            "Imagine two people at the start of 2019, each with \\$10,000. One buys an S&P index "
            "fund. The other goes all-in on the vinyl theme — say, splitting into the listed music "
            "majors. Where do they land?"
        ),
        code(
            "start = 10_000.0; yrs_h = 7\n"
            "spy_end = start*(1+R['spy_cagr']/100)**yrs_h\n"
            "# 'vinyl theme' = equal split of the two pure labels (the honest 'ride vinyl' basket)\n"
            "theme_cagr = (R['wmg_cagr']+R['umg_cagr'])/2/100\n"
            "theme_end = start*(1+theme_cagr)**yrs_h\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(['S&P index fund', 'vinyl theme\\n(Warner+Universal)'], [spy_end, theme_end],\n"
            "       color=[GREEN, AMBER], width=.55)\n"
            "for i,v in enumerate([spy_end, theme_end]): ax.annotate(f'${v:,.0f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('value of $10,000 after 7 years')\n"
            "ax.set_title('Riding the \"real trend\" left you behind the boring index'); plt.tight_layout(); plt.show()\n"
            "print(f'S&P index fund: ${spy_end:,.0f}   |   vinyl-theme labels: ${theme_end:,.0f}')"
        ),
        md(
            "The index-fund investor roughly **doubles** their money doing nothing. The theme "
            "investor — who was *right* about the trend — ends up well behind, because being right "
            "about the industry and getting paid for it are different things. The people who \"made "
            "money on vinyl\" mostly *sell* records (labels, pressing plants, shops) — they don't "
            "*invest* in the trend, they run a business on it. You only hear the tidy version."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Pull the real RIAA series yourself.** Our vinyl line is a cited *approximation*; the "
            "RIAA year-end database has the exact figures. Swap them in — the shape (and the verdict) "
            "won't move.\n"
            "- **The theme-investing pattern.** Cannabis, EVs, \"metaverse,\" clean energy: every "
            "real, visible trend spawns the same pitch — and the same gap between a growing industry "
            "and a buyable return (see [docs/references.md](../docs/references.md)).\n"
            "- **The passion-asset cousin.** [Study 358 — Watches](../../358-watch-index/) is the "
            "same shape with a *fake* trend (a bubble); vinyl is the harder case — a *real* one — "
            "and it still doesn't trade.\n\n"
            "*Think a specific pressing (a first-press original, a rare colour variant) beat the S&P "
            "net of every fee? Pull its Discogs history, charge the marketplace spread, and show it — "
            "then check it wasn't just one lucky find near the bottom.*"
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
            "# The vinyl revival — a quantitative teardown 🔬\n"
            "### Revenue index vs SPY (CAGR / vol / MDD + an annual-excess *t*) · Newey-West proxy "
            "alpha · the collector-carry haircut on NAV · a synthetic revival positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We test "
            "the strongest tradable form of \"vinyl is back, ride it\": (H₁) the vinyl-revenue *trend* "
            "out-grows SPY; (H₂) a buyable proxy carries alpha vs the market; (H₃) a collector "
            "strategy survives the transaction + carry cost. We find **H₁ true but untradable** "
            "(revenue growth is not a return), **H₂ rejected** (no significant proxy alpha), **H₃ "
            "rejected** (negative net of costs) — and the boom is already in the price.\n\n"
            "> ⚠️ **Not investment advice — data provenance.** The vinyl index is **hardcoded, "
            "cited, approximate** (public RIAA year-end reports — a *labelled proxy for the trend*, "
            "and *industry revenue*, never a price). Equity proxies `WMG`, `SPOT`, `UMG.AS`, `SPY` "
            "are month-end Adj Close via yfinance (as-of 2025-12-31). Offline core + synthetic "
            "control are deterministic. Methods in [`docs/references.md`](../docs/references.md); "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Vinyl revenue CAGR **{R['vinyl_cagr']:.1f}%** > SPY "
            f"**{R['spy_cagr_ye']:.1f}%** (a real trend) — but revenue growth isn't a return; annual "
            f"excess *t* = **{R['excess_t']:+.2f}** (n={R['excess_n']}), and no proxy alpha clears "
            f"|t|≥2 (WMG {R['wmg_t']:+.2f}, SPOT {R['spot_t']:+.2f}, UMG {R['umg_t']:+.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | Two proxies *lagged* SPY (WMG {R['wmg_cagr']:+.1f}%, UMG "
            f"{R['umg_cagr']:+.1f}% vs {R['spy_cagr']:+.1f}%); the winner SPOT is β≈{R['spot_beta']:.2f} "
            f"streaming with a {R['spot_mdd']:.0f}% drawdown; collector flip nets "
            f"**{R['carry_flat_net']:+.1f}%/yr**. |\n"
            f"| **Already priced in?** | `CONFIRMED` | Vinyl ~{R['vinyl_share']:.0f}% of a "
            f"~{R['streaming_share']:.0f}%-streaming business; no vinyl-attributable premium in any "
            "listed proxy. A public 18-year trend is not an edge. |\n\n"
            "> 💡 In plain words: the revival is genuinely real and even out-grew stocks — but it's a "
            "*revenue* trend you can't custody, the only things you can buy are streaming-driven beta "
            "with no vinyl alpha, and flipping records loses to costs. \"Trade the vinyl trend\" is a "
            "category error the market already reflects."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let the vinyl-revenue level be $V_t$ and the benchmark $B_t$ (SPY). The claim is a "
            "joint hypothesis:\n\n"
            "- **H₁ (the trend out-grows).** Annual excess $\\;x_t = r^V_t - r^B_t\\;$ has "
            "$\\mathbb{E}[x_t] > 0$ — vinyl grows faster than stocks. *(Note: $r^V$ is revenue "
            "growth, a category measure, not an investable return.)*\n"
            "- **H₂ (it's buyable with alpha).** For a tradable proxy $P$, the intercept $\\alpha$ in "
            "$r^P_t = \\alpha + \\beta r^B_t + \\varepsilon_t$ is positive with a Newey-West *t* > 2.\n"
            "- **H₃ (a collector clears costs).** The net CAGR after the marketplace round-trip "
            "spread $s$ over hold $h$ and storage carry $c$ stays positive: "
            "$(1+g)(1+((1-s)^{1/h}-1))(1-c)-1 > 0$.\n\n"
            "The steelman is unusually strong: unlike a bubble, H₁ is *durably* true — 18 consecutive "
            "up years. The test is whether **truth of the trend implies a tradable, net-of-cost "
            "edge** — i.e. whether \"real trend\" and \"investable\" are the same thing. They are not."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "If H₁–H₃ held, thematic investing would be a free lunch: identify a visible growing "
            "trend, buy it, win. But each leg is separately falsifiable. H₁ is a **growth race** on a "
            "common clock — and here it's the *plaintiff's best evidence*, because the trend really "
            "did out-grow SPY. H₂ asks whether the *only investable expression* (listed equities) "
            "delivers anything beyond market beta — because you cannot custody \"the revenue series,\" "
            "and vinyl is a single-digit slice of a streaming-dominated P&L. H₃ is the "
            "**microstructure tax** on the one *direct* holding (physical records), compounded by a "
            "supply response an ETF never faces: a revived market **presses more units**. The "
            "tradable claim needs H₂ or H₃; a true H₁ alone is a growing industry you cannot bank."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Vinyl index (proxy).** A hardcoded, cited, **approximate** annual RIAA level (base "
            "100 @ 2010), reconstructed from public year-end reports (CDs overtaken in revenue 2020, "
            "in units 2022; ~$1.4B by 2024). *Labelled a proxy* — its path is defensible, its precise "
            "values are not a live feed, and it is **revenue, not price**.\n"
            "- **Equity proxies.** `WMG`, `SPOT`, `UMG.AS`, `SPY` month-end Adj Close (yfinance, "
            "cached). Survivorship is **not** a concern (named tickers, not a screen), but each "
            "series starts at its **listing date** (SPOT 2018-04, WMG 2020-06, UMG 2021-09) — a real "
            "look-ahead caveat (we see only the listed life), stated.\n"
            "- **Signal test.** (i) Paired annual-excess $t$ of $r^V - r^B$ (small-$n$, and a "
            "*growth* not a *return* comparison — weak by construction and by category). (ii) "
            "**Newey-West (6-lag) HAC** $t$ of each proxy alpha vs SPY — the bar for `REAL` is "
            "*t* ≥ 2 in the trade's favour.\n"
            "- **Cost (beat 6).** Charge the collector round-trip spread (30% over a 5y hold) + 1%/yr "
            "storage **once on NAV**, under both a charitable (records appreciate at revenue growth) "
            "and a realistic (flat per-record resale) gross.\n"
            "- **Positive control.** A deterministic revival path with a *planted* positive drift; "
            "the engine must recover the up-sign and a finite Sharpe — proof a null on the real tape "
            "is a real null, not a broken harness.\n"
            "- **What would make us say \"tradable\":** a proxy alpha *t* > 2 **or** a positive "
            "net-of-cost collector CAGR. We find neither."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The growth race — vinyl revenue vs SPY\n\n"
            "The RIAA revenue index (base 100 @ 2010) on its own clock; SPY year-end for scale. CAGR, "
            "vol and the annual-excess *t* in the print. **This is a growth comparison, not a "
            "return you can earn.**"
        ),
        code(
            "yrs = [int(y) for y in IDX.index.year]\n"
            "lv = np.array([float(v) for v in IDX.values])\n"
            "si = st.summarize(IDX, periods_per_year=1.0)\n"
            "if HAVE_PROXIES:\n"
            "    spy_ye = PROX['SPY'].resample('YE').last()\n"
            "    spy_ye = spy_ye[(spy_ye.index.year>=2018)&(spy_ye.index.year<=2025)]\n"
            "    ss = st.summarize(spy_ye, periods_per_year=1.0)\n"
            "    ae = st.annual_excess_t(IDX, PROX['SPY'])\n"
            "else:\n"
            "    ss = {'cagr':R['spy_cagr_ye']/100,'vol':R['spy_vol_ye']/100,'mdd':R['spy_mdd_ye']/100}\n"
            "    ae = {'mean_excess':R['excess_mean']/100,'t':R['excess_t'],'n':R['excess_n']}\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.plot(yrs, lv, 'o-', c=AMBER, lw=2.2, label=f\"vinyl revenue index  CAGR {si['cagr']*100:.1f}%\")\n"
            "ax.axhline(100, ls=':', c=GREY)\n"
            "ax.set_xlabel('year'); ax.set_ylabel('base 100 @ 2010')\n"
            "ax.set_title('H1: the vinyl TREND out-grew SPY — but revenue is not a return'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"vinyl: CAGR {si['cagr']*100:.2f}%  vol {si['vol']*100:.1f}%  maxDD {si['mdd']*100:.1f}%\")\n"
            "print(f\"SPY  : CAGR {ss['cagr']*100:.2f}%  vol {ss['vol']*100:.1f}%  maxDD {ss['mdd']*100:.1f}%\")\n"
            "print(f\"annual excess (vinyl-SPY): mean {ae['mean_excess']*100:+.2f}%/yr  t={ae['t']:+.3f}  (n={ae['n']})\")"
        ),
        md(
            f"> 💡 In plain words: vinyl revenue compounded at **{R['vinyl_cagr']:.1f}%** against SPY's "
            f"**{R['spy_cagr_ye']:.1f}%** — the trend genuinely *out-grew* stocks. But this is the "
            "plaintiff's best exhibit and it still doesn't help: the annual excess is "
            f"**{R['excess_mean']:+.1f}%/yr**, *t* = **{R['excess_t']:+.2f}** (n={R['excess_n']}) — "
            "not even a clean growth win — and, decisively, **revenue growth is not an investable "
            "return.** There is no security that pays the RIAA series. H₁ is *true and irrelevant*."
        ),
        md(
            "### 4b · The buyable proxies — is there alpha, or just beta?\n\n"
            "Newey-West (6-lag) regression of each proxy's **monthly** return on SPY. `REAL` needs "
            "$t_\\alpha \\ge 2$ in the trade's favour."
        ),
        code(
            "if HAVE_PROXIES:\n"
            "    spy_r = PROX['SPY'].pct_change().dropna()\n"
            "    rows = {}\n"
            "    for t in ['WMG','SPOT','UMG.AS']:\n"
            "        s = st.summarize(PROX[t]); nw = st.newey_west_alpha_t(PROX[t].pct_change().dropna(), spy_r, 6)\n"
            "        rows[t] = dict(cagr=s['cagr']*100, sharpe=s['sharpe'], mdd=s['mdd']*100,\n"
            "                       alpha=nw['alpha_ann']*100, beta=nw['beta'], t=nw['t_alpha'])\n"
            "else:\n"
            "    rows = {'WMG':dict(cagr=R['wmg_cagr'],sharpe=R['wmg_sharpe'],mdd=R['wmg_mdd'],alpha=R['wmg_alpha'],beta=R['wmg_beta'],t=R['wmg_t']),\n"
            "            'SPOT':dict(cagr=R['spot_cagr'],sharpe=R['spot_sharpe'],mdd=R['spot_mdd'],alpha=R['spot_alpha'],beta=R['spot_beta'],t=R['spot_t']),\n"
            "            'UMG.AS':dict(cagr=R['umg_cagr'],sharpe=R['umg_sharpe'],mdd=R['umg_mdd'],alpha=R['umg_alpha'],beta=R['umg_beta'],t=R['umg_t'])}\n"
            "labels=list(rows); alphas=[rows[t]['alpha'] for t in labels]; ts=[rows[t]['t'] for t in labels]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "cols=[GREEN if a>0 else RED for a in alphas]\n"
            "ax.bar(labels, alphas, .5, color=cols)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised alpha vs SPY (%)')\n"
            "for i,t in enumerate(labels): ax.annotate(f\"t={ts[i]:+.2f}\",(i,alphas[i]),ha='center',va='bottom' if alphas[i]>=0 else 'top')\n"
            "ax.set_title('H2: no significant alpha in any buyable proxy (|t|<2)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for t in labels: r=rows[t]; print(f\"{t:7s} CAGR {r['cagr']:6.2f}%  Sharpe {r['sharpe']:.2f}  maxDD {r['mdd']:6.1f}%  alpha {r['alpha']:+.2f}%/yr  beta {r['beta']:.2f}  NW t {r['t']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: the two pure labels carry *negative* point-alpha (WMG "
            f"{R['wmg_alpha']:+.0f}%/yr, *t*={R['wmg_t']:+.2f}; UMG {R['umg_alpha']:+.0f}%/yr, "
            f"*t*={R['umg_t']:+.2f}) — they under-performed the market they're exposed to. Spotify's "
            f"alpha is **{R['spot_alpha']:+.0f}%/yr but *t*={R['spot_t']:+.2f}** — statistically zero, "
            f"and it's β≈{R['spot_beta']:.2f} *streaming* beta with a {R['spot_mdd']:.0f}% drawdown, "
            "not vinyl. **None clears |t| ≥ 2**: H₂ rejected. The only investable expressions of the "
            "trade are market beta you could have bought cheaper as SPY."
        ),
        md(
            "### 4c · The microstructure tax — a collector's net of spread + carry\n\n"
            "The one *direct* holding is physical records. A flip pays a ~30% round-trip spread "
            "(marketplace fee + condition discount) over a ~5-year hold plus ~1%/yr storage. We show "
            "it under two grosses: **charitable** (records appreciate at the full revenue-growth "
            "rate) and **realistic** (per-record resale ≈ flat, because a revived market presses more "
            "units)."
        ),
        code(
            "hc = st.net_of_collector_carry(si['cagr'], round_trip_spread=0.30, hold_years=5.0, storage_per_year=0.01)\n"
            "h0 = st.net_of_collector_carry(0.0, round_trip_spread=0.30, hold_years=5.0, storage_per_year=0.01)\n"
            "cats = ['charitable gross\\n(= revenue growth)', 'charitable NET', 'realistic gross\\n(~flat resale)', 'realistic NET']\n"
            "vals = [hc['gross_cagr']*100, hc['net_cagr']*100, 0.0, h0['net_cagr']*100]\n"
            "cols = [GREY, AMBER, GREY, RED]\n"
            "fig, ax = plt.subplots(figsize=(9.4,4.3))\n"
            "ax.bar(cats, vals, .6, color=cols)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(R['spy_cagr'], ls='--', c=GREEN, alpha=.6, label='SPY CAGR')\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_ylabel('collector CAGR %/yr'); ax.set_title('H3: realistically, a collector nets NEGATIVE'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"charitable: gross {hc['gross_cagr']*100:+.2f}%  - spread {hc['spread_drag_annual']*100:+.2f}%/yr  - storage {hc['storage_per_year']*100:+.2f}%/yr  =  NET {hc['net_cagr']*100:+.2f}%/yr\")\n"
            "print(f\"realistic : gross {h0['gross_cagr']*100:+.2f}%  =>  NET {h0['net_cagr']*100:+.2f}%/yr\")"
        ),
        md(
            f"> 💡 In plain words: even the *charitable* fantasy (net **{R['carry_net']:+.1f}%/yr**) "
            "still trails SPY and rests on a false premise — the RIAA line is *industry revenue*, and "
            "a booming market **expands supply**, so your specific record does not appreciate at "
            f"~22%/yr. On the realistic read (flat resale), the collector nets "
            f"**{R['carry_flat_net']:+.1f}%/yr** after spread + storage. H₃ rejected. **MIRAGE** is "
            "the only honest stamp."
        ),
        md(
            "### 4d · Positive control — the engine recovers a planted revival\n\n"
            "A deterministic revival path (planted +20% CAGR, σ=4%/mo, seed 715). The harness must "
            "recover the up-sign and a finite Sharpe — proving the nulls above are real, not a broken "
            "pipeline."
        ),
        code(
            "syn = data.synthetic_revival()\n"
            "s = st.summarize(syn); cr = st.control_recovers(syn, planted_sign=1)\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.2))\n"
            "ax.plot(syn.index, syn.values, c=GREY, lw=2)\n"
            "ax.axhline(100, ls=':', c=GREY)\n"
            "ax.set_ylabel('synthetic level'); ax.set_title('Planted revival: engine recovers sign + Sharpe (machinery proof)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"peak {syn.max():.0f} -> end {syn.iloc[-1]:.0f}  recovered CAGR {s['cagr']*100:+.2f}%  Sharpe {s['sharpe']:.2f}  maxDD {s['mdd']*100:.1f}%  sign_ok={cr['sign_ok']}\")"
        ),
        md(
            "> 💡 In plain words: the engine banks the planted signal (recovered CAGR ~+19%, Sharpe "
            "~1.3, sign correct). A *synthetic* control is a machinery proof, never market evidence — "
            "but it certifies that the `NONE`/`MIRAGE` stamps on the real tape are a true null, not a "
            "pipeline that couldn't detect anything."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — vinyl revenue CAGR {R['vinyl_cagr']:.1f}% > SPY "
            f"{R['spy_cagr_ye']:.1f}% (a genuine trend), but revenue growth is not a return, the "
            f"annual excess is unclean (*t* = {R['excess_t']:+.2f}, n={R['excess_n']}), and proxy "
            f"alphas are insignificant (WMG *t*={R['wmg_t']:+.2f}, SPOT *t*={R['spot_t']:+.2f}, UMG "
            f"*t*={R['umg_t']:+.2f}). No robust *t* ≥ 2 anywhere in the trade's favour.\n"
            f"- **Tradability `MIRAGE`** — two proxies lagged SPY (WMG {R['wmg_cagr']:+.1f}%, UMG "
            f"{R['umg_cagr']:+.1f}% vs {R['spy_cagr']:+.1f}%); the winner SPOT is β≈{R['spot_beta']:.2f} "
            f"streaming with a {R['spot_mdd']:.0f}% drawdown and zero alpha; a collector nets "
            f"**{R['carry_flat_net']:+.1f}%/yr** once a supply-expanding market meets the spread.\n"
            f"- **Already priced in? `CONFIRMED`** — vinyl is ~{R['vinyl_share']:.0f}% of a "
            f"~{R['streaming_share']:.0f}%-streaming business; no listed proxy carries a "
            "vinyl-attributable premium. A famous, public, 18-year trend is not a secret you can "
            "trade."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the capacity & cost reality\n\n"
            "Terminal wealth of \\$10,000 invested at each proxy's listing, vs SPY, and the collector "
            "path (realistic net from 4c). Capacity is the other wall: the *labels* are liquid but "
            "aren't \"vinyl,\" and the *records* are a bespoke, illiquid, ~30%-spread market with no "
            "scalable book."
        ),
        code(
            "start=10_000.0; yrs_h=7\n"
            "paths={'S&P index fund':R['spy_cagr']/100, 'music labels (WMG+UMG avg)':(R['wmg_cagr']+R['umg_cagr'])/2/100,\n"
            "       'record collector (net)':R['carry_flat_net']/100, 'vinyl revenue (untradable)':R['vinyl_cagr']/100}\n"
            "fig, ax = plt.subplots(figsize=(9.6,4.3))\n"
            "labels=list(paths); ends=[start*(1+g)**yrs_h for g in paths.values()]\n"
            "cols=[GREEN, AMBER, RED, GREY]\n"
            "ax.bar(labels, ends, .6, color=cols)\n"
            "for i,v in enumerate(ends): ax.annotate(f'${v:,.0f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('value of $10,000 after 7 years'); ax.set_xticklabels(labels, rotation=12, ha='right')\n"
            "ax.set_title('The only line that beat SPY (grey) is the one you cannot buy')\n"
            "plt.tight_layout(); plt.show()\n"
            "for l,g in paths.items(): print(f\"{l:30s} ${start*(1+g)**yrs_h:>10,.0f}  ({g*100:+.1f}%/yr)\")"
        ),
        md(
            "> 💡 In plain words: the *untradable* revenue line (grey) is the only one that beats the "
            "S&P — and you cannot buy it. The tradable label basket trails; the collector ends below "
            "the stake. Capacity seals it: buying labels dilutes vinyl into single-digit-percent of a "
            "streaming P&L, and the physical market is a ~30%-round-trip, one-off illiquid trade. "
            "There is no venue or size that turns a real trend into an edge."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Swap in the live RIAA series.** Replace the hardcoded levels with the exact RIAA "
            "year-end figures and re-run; the *t*-stats shift a hair but the sign and the category "
            "error (revenue ≠ return) don't move.\n"
            "- **Per-pressing dispersion.** The aggregate hides survivorship: test individual records "
            "on Discogs (a rare first-press vs a current reissue) — the winners are a thin selected "
            "tail and a revived market floods supply, so correct for both.\n"
            "- **The thematic-investing prior.** A real, visible trend (EVs, cannabis, AI, vinyl) is "
            "public information; efficient-markets logic says it's in the price before you arrive "
            "([docs/references.md](../docs/references.md)). Vinyl is a clean demonstration.\n\n"
            "*The reproducible core is offline and deterministic; the vinyl index is a **cited, "
            "approximate proxy** for industry revenue and the equity tickers are **labelled proxies** "
            "for the trade. Methods: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
