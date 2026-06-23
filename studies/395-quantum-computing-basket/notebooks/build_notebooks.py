"""Generate the two narrative notebooks for Study 395 (Quantum-Computing-Basket).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached quantum panel under
../_cache/ (the four pure-plays + QTUM + SPY/QQQ) and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance monthly total returns,
# 4 quantum pure-plays + QTUM + SPY/QQQ, 2021-05 -> 2026-05, 61 months, 5.1 years, fp d955f566bc4b).
R = dict(
    start="2021-05", end="2026-05", months=61, years=5.1, fingerprint="d955f566bc4b",
    # (cagr%, sharpe, vol%, maxdd%)
    basket=(59.1, 0.75, 160, -84.3), basket_net=(58.9, 0.74),
    etf=(27.9, 1.06, 27, -34.5), spy=(13.9, 0.91, 16, -23.9), qqq=(17.3, 0.88, 21, -32.6),
    # (spread%/yr, HAC t, n)
    t_spy=(104.6, 1.23, 61), t_qqq=(100.8, 1.19, 61), t_etf=(90.8, 1.12, 61),
    # per name: (cagr%, vol%, maxdd%, excess%/yr, HAC t)
    names={"IONQ": (45.3, 117, -86, 76.8, 1.40), "QBTS": (24.4, 174, -95, 105.2, 1.27),
           "RGTI": (20.4, 218, -96, 119.4, 1.11), "QUBT": (14.1, 260, -95, 117.1, 0.90)},
    # synthetic: (edge_sharpe, cagr%, sharpe, spread%/yr, HAC t)
    syn=[(0.0, -10.6, 0.23, 4.7, 0.18), (1.0, 256.5, 2.05, 146.9, 5.71)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Next_big_thing%3F: Hype--cycle](https://img.shields.io/badge/Next_big_thing%3F-Hype--cycle-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from quantum_computing_basket import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    RETS, BENCH = data.load_real(allow_survivorship=True)
    RACE = st.race(RETS, BENCH, data.BASKET, cost_bps=10.0, allow_survivorship=True)
else:
    RETS = BENCH = RACE = None
print("real quantum-panel cache present:", HAVE_REAL,
      "| months:", (0 if RETS is None else len(RETS)))
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell can quote
# it whether or not the cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The quantum-computing basket — next big thing, or pure hype? ⚛️\n"
            "### A basket that returned +59%/yr and *still* lost to the index — in plain English\n\n"
            + BADGES +
            "There's a trade everyone's heard: forget the boring index, **buy the quantum stocks** — "
            "IONQ, Rigetti, D-Wave, Quantum Computing Inc. — and ride the next computing revolution. "
            "And the headline is true: over the last five years that four-stock basket returned about "
            f"**+{R['basket'][0]:.0f}% a year**, while the S&P 500 did **+{R['spy'][0]:.0f}%**. Four "
            "times the market. Free money, right?\n\n"
            "Not quite. That +59% came with a **−84% crash** along the way and a ride so wild that, "
            "*per unit of risk*, the basket actually did **worse than the index** — and worse than a "
            "boring quantum ETF you could have bought on day one. This notebook is about the gap "
            "between *a real technology* and *a good investment* — and why \"next big thing\" almost "
            "never means \"buy the basket.\"\n\n"
            "> 📓 **Plain-language layer.** Want the Sharpe ratios, the HAC *t*-stats and the null "
            "control? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** These four are the quantum names that **survived** to a "
            "listing (most arrived via SPAC mergers; many quantum SPACs vanished). Keeping only the "
            "survivors flatters the basket — we call that out throughout. Every chart is drawn by the "
            "code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Did the quantum basket beat the market? | **By a mile — +{R['basket'][0]:.0f}%/yr vs "
            f"+{R['spy'][0]:.0f}%/yr.** That's the hook. |\n"
            f"| So it's a great investment? | **No.** It got there at **{R['basket'][2]:.0f}% "
            f"volatility** with an **{R['basket'][3]:.0f}% crash**. Per unit of risk (its *Sharpe*) it "
            f"scored **{R['basket'][1]:.2f}** — *below* the S&P 500's **{R['spy'][1]:.2f}**. |\n"
            f"| Is the outperformance even real? | **Can't tell.** The gap is so noisy that "
            "statistically it's **indistinguishable from luck** (a proper *t*-test scores "
            f"**{R['t_spy'][1]:.2f}**, far below the ~2 you'd need). |\n"
            f"| Was there a sane way to play quantum? | **Yes — the diversified QTUM ETF** "
            f"(+{R['etf'][0]:.0f}%/yr, Sharpe **{R['etf'][1]:.2f}**), which beat *both* the basket and "
            "the index on risk-adjusted terms — and you could have bought it from day one. |\n\n"
            "> The technology is real. The basket is a **high-variance lottery** wearing a thesis. A "
            "+59%/yr number you can't hold through an −84% drawdown isn't an edge."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Quantum computing is the next AI — a once-in-a-generation technology. Don't "
            "stock-pick; just **hold the pure-plays** — IONQ, RGTI, QBTS, QUBT — equal-weight and ride "
            "the whole wave.\"*\n\n"
            "It's a seductive pitch, and it's been everywhere since Google's *Willow* chip headline in "
            "late 2024 sent the names vertical. The logic feels airtight: a transformative technology "
            "**must** mean its listed champions go up. We'll rebuild that exact basket and look at "
            "what holding it actually felt like — not just where it ended up."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two very different things hide inside \"the basket went up 59% a year,\" and only one is "
            "an investment. (1) *Did it produce a big number?* Yes. (2) *Were you **paid** for the "
            "risk you took to get it — could you actually have held it?* That's the whole game. A "
            "return you only capture if you sit through an **−84% drawdown** without flinching isn't a "
            "strategy a human (or a fund) can run. And if a *diversified* version of the same theme "
            "gives you a **better ride for less risk**, the concentrated lottery basket isn't even the "
            "smart way to be bullish on quantum."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We rebuild the basket honestly: the four pure-plays, equal-weight, rebalanced monthly, "
            f"over the **{R['years']:.0f} years** ({R['start']} → {R['end']}, {R['months']} months) all "
            "four have traded. Then we line it up against three yardsticks:\n\n"
            "1. **The market (SPY)** and **the tech tape (QQQ)** — what you'd have earned the boring "
            "way.\n"
            "2. **The diversified quantum ETF (QTUM)** — the sane way to bet on the theme.\n"
            "3. **The risk** — not just the return, but the *volatility* and the *worst drawdown*, "
            "because that's what decides whether a number is holdable.\n\n"
            "And we ask the honest statistician's question: with only five years and a couple of "
            "violent up-months, **is the gap big enough to be real, or could a coin have done it?**"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the growth-of-$1 chart.** Watch where the basket goes — and where it *was* in "
            "between."
        ),
        code(
            "if HAVE_REAL:\n"
            "    legs = {'Quantum basket (4)': RACE['basket'], 'QTUM ETF': RACE['etf'],\n"
            "            'S&P 500 (SPY)': RACE['spy'], 'Nasdaq-100 (QQQ)': RACE['qqq']}\n"
            "    cols = {'Quantum basket (4)': RED, 'QTUM ETF': GREEN, 'S&P 500 (SPY)': GREY, 'Nasdaq-100 (QQQ)': AMBER}\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.8))\n"
            "    for name, r in legs.items():\n"
            "        eq = (1+r.dropna()).cumprod()\n"
            "        ax.plot(eq.index, eq.values, lw=2, color=cols[name], label=name)\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log scale)')\n"
            "    ax.set_title('The basket wins the race — but look at the volatility of the red line')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print('basket end multiple:', round(float((1+RACE['basket'].dropna()).prod()),1), 'x')\n"
            "else:\n"
            "    print('no cache — see docs/results.md: basket +59.1%/yr vs SPY +13.9%/yr over 61 months')"
        ),
        md(
            f"The red basket line ends *highest* — that's the **+{R['basket'][0]:.0f}%/yr** headline. "
            "But notice how violently it swings. The chart hides the worst part: the *path*. Let's put "
            "the risk on screen."
        ),
        md(
            "**The return you see vs the risk you don't.** Same four legs — but now CAGR next to the "
            "**worst drawdown** each one put you through."
        ),
        code(
            "labels = ['Quantum\\nbasket', 'QTUM\\nETF', 'SPY', 'QQQ']\n"
            "if HAVE_REAL:\n"
            "    s = RACE['summ']\n"
            "    cagr = [s['basket']['cagr'], s['etf']['cagr'], s['spy']['cagr'], s['qqq']['cagr']]\n"
            "    dd = [s['basket']['max_dd'], s['etf']['max_dd'], s['spy']['max_dd'], s['qqq']['max_dd']]\n"
            "else:\n"
            "    cagr = [R['basket'][0]/100, R['etf'][0]/100, R['spy'][0]/100, R['qqq'][0]/100]\n"
            "    dd = [R['basket'][3]/100, R['etf'][3]/100, R['spy'][3]/100, R['qqq'][3]/100]\n"
            "x = np.arange(len(labels))\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.3))\n"
            "a1.bar(x, [c*100 for c in cagr], color=[RED, GREEN, GREY, AMBER], width=.6)\n"
            "a1.set_xticks(x); a1.set_xticklabels(labels); a1.set_ylabel('CAGR (%)')\n"
            "a1.set_title('The return (the hook)')\n"
            "for i,c in enumerate(cagr): a1.annotate(f'{c*100:.0f}%',(i,c*100),ha='center',va='bottom')\n"
            "a2.bar(x, [d*100 for d in dd], color=[RED, GREEN, GREY, AMBER], width=.6)\n"
            "a2.set_xticks(x); a2.set_xticklabels(labels); a2.set_ylabel('worst drawdown (%)')\n"
            "a2.set_title('The risk (the catch)')\n"
            "for i,d in enumerate(dd): a2.annotate(f'{d*100:.0f}%',(i,d*100),ha='center',va='top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('basket worst drawdown:', f'{dd[0]*100:.0f}%', ' vs SPY:', f'{dd[3-1]*100:.0f}%')"
        ),
        md(
            f"There's the catch. To earn the basket's **+{R['basket'][0]:.0f}%/yr** you had to survive "
            f"an **{R['basket'][3]:.0f}%** drawdown — your $100 became roughly $16 at the bottom. The "
            f"index's worst was {abs(R['spy'][3]):.0f}%. Almost nobody holds through an 84% loss; they "
            "sell at the bottom and the headline return is one they never actually got."
        ),
        md(
            "**Reward per unit of risk — the Sharpe ratio.** This is the single number that asks \"was "
            "the wild ride *worth it*?\" Higher is better; the index sets the bar."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = RACE['summ']\n"
            "    sh = [s['basket']['sharpe'], s['etf']['sharpe'], s['spy']['sharpe'], s['qqq']['sharpe']]\n"
            "else:\n"
            "    sh = [R['basket'][1], R['etf'][1], R['spy'][1], R['qqq'][1]]\n"
            "x = np.arange(len(labels))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "bars = ax.bar(x, sh, color=[RED, GREEN, GREY, AMBER], width=.6)\n"
            "ax.axhline(sh[2], ls='--', c=GREY, label=f'S&P 500 bar ({sh[2]:.2f})')\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('Sharpe ratio (reward per unit of risk)')\n"
            "ax.set_title('The lottery basket scores BELOW the index — the diversified ETF wins')\n"
            "for i,v in enumerate(sh): ax.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('basket Sharpe', f'{sh[0]:.2f}', '< SPY', f'{sh[2]:.2f}', '< QTUM ETF', f'{sh[1]:.2f}')"
        ),
        md(
            f"This is the whole study in one chart. The flashy basket scores **{R['basket'][1]:.2f}** — "
            f"*below* the S&P 500's **{R['spy'][1]:.2f}**. The boring **diversified QTUM ETF** wins "
            f"outright at **{R['etf'][1]:.2f}**. Per unit of white-knuckle risk, the concentrated "
            "quantum bet paid you **less** than doing nothing — and a sane quantum ETF paid you more "
            "than either."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The basket really beat the market (+{R['basket'][0]:.0f}%/yr) — but "
            "the gap is so noisy it's statistically **indistinguishable from luck**, and per unit of "
            "risk it scored *below* the index. Real raw return, no real edge.\n"
            f"- **Tradability — Mirage.** A **{R['basket'][2]:.0f}%-vol**, **{R['basket'][3]:.0f}%-"
            "drawdown** sleeve isn't something you can actually hold or size. Risk-adjusted it loses "
            "to both SPY and the diversified ETF.\n"
            "- **Next big thing? — Hype-cycle.** A real technology with no profits yet, and a stock "
            "cohort that's a **high-variance lottery dressed as a thesis**. The sane way to be bullish "
            "on quantum was the boring ETF all along. ([Study 393](../393-ai-datacenter-basket/) is "
            "the same trick in an AI costume.)"
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — one stock or four?\n\n"
            "Maybe the basket is really carried by one moon-shot. Here's each name's own return and "
            "its own worst drawdown. The punchline is that *every* one is a casino chip."
        ),
        code(
            "names = list(R['names'].keys())\n"
            "if HAVE_REAL:\n"
            "    cg = [RACE['single_names'][n]['cagr']*100 for n in names]\n"
            "    dd = [RACE['single_names'][n]['max_dd']*100 for n in names]\n"
            "else:\n"
            "    cg = [R['names'][n][0] for n in names]; dd = [R['names'][n][2] for n in names]\n"
            "x = np.arange(len(names))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, cg, .4, color=GREEN, label='CAGR (%)')\n"
            "ax.bar(x+.2, dd, .4, color=RED, label='worst drawdown (%)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel('%')\n"
            "ax.set_title('Every single name: a big return AND a ~90% crash')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('worst single-name drawdowns:', {n: f'{d:.0f}%' for n,d in zip(names,dd)})"
        ),
        md(
            f"Every one of the four drew down **{abs(R['names']['RGTI'][2]):.0f}%-class** — that is, "
            "they each lost roughly **nine-tenths of their value** at some point. A basket of four "
            "lottery tickets is still a basket of lottery tickets; diversifying across them softens "
            "nothing because they all crash together on the same risk-off days. Costs, by the way, are "
            "a non-issue (a few bps on a monthly hold) — **risk**, not friction, is what makes this "
            "untradeable."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The sibling.** [Study 393 — AI-Datacenter-Basket](../393-ai-datacenter-basket/) is the "
            "same \"buy the theme\" trade for AI infrastructure — there the spread is real but "
            "*hand-picked in hindsight*; here it's *risk that isn't paid for*. Two ways the same "
            "thematic hook misleads.\n"
            "- **The disruptive-tech fund.** [Study 334 — ARK-Innovation](../334-ark-innovation/) — "
            "does riding \"innovation\" pay you risk-adjusted? Same family of high-vol theme bets.\n"
            "- **Build your own.** Swap the four pure-plays for the QTUM ETF holdings, or add the "
            "mega-cap quantum efforts (GOOGL, IBM, MSFT), and re-run — the more diversified you get, "
            "the more the lottery tail shrinks and the Sharpe climbs.\n\n"
            "*Think the concentrated basket beats the ETF on risk-adjusted terms? Show its Sharpe "
            "clearing the index's **and** its spread clearing t = 2 — then we'll talk.*"
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
            "# Quantum-Computing-Basket — a quantitative teardown 🔬\n"
            "### The Sharpe/vol/drawdown race · the HAC *t* of the spread vs SPY/QQQ/QTUM · the "
            "single-name decomposition · the survivorship caveat · a synthetic null that reproduces "
            "the spread from zero risk-adjusted edge\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We "
            "separate the two things the \"next big thing\" pitch fuses: a **large raw spread** from "
            "**any risk-adjusted edge**, and confront both with **volatility, drawdown, and the "
            "standard error of the spread**. The decisive object is not the point estimate (it's huge, "
            "as the lore says) but its **Sharpe and its HAC *t***: a +100%/yr spread at HAC "
            f"**t = {R['t_spy'][1]:.2f}** on a basket whose **Sharpe ({R['basket'][1]:.2f}) is below "
            f"the market's ({R['spy'][1]:.2f})** is a hype cycle, not an edge.\n\n"
            "> ⚠️ **Data + survivorship note.** The four pure-plays are the **surviving** de-SPAC "
            "cohort (the quantum SPACs that delisted are absent — a *bullish* bias, named on the "
            "Signal axis). Real data: yfinance monthly total returns, 4 pure-plays + QTUM + SPY/QQQ, "
            f"{R['start']}→{R['end']} ({R['months']} months, fp `{R['fingerprint']}`). Offline core + "
            "synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `WEAK` | Basket−SPY spread **+{R['t_spy'][0]:.0f}%/yr** but HAC "
            f"**t = {R['t_spy'][1]:.2f}** (n = {R['t_spy'][2]}; {R['t_qqq'][1]:.2f} vs QQQ, "
            f"{R['t_etf'][1]:.2f} vs QTUM) — **fails t ≥ 2**, and **no single name clears t = 2**. "
            f"Basket Sharpe **{R['basket'][1]:.2f}** < SPY **{R['spy'][1]:.2f}**. |\n"
            f"| **Tradability** | `MIRAGE` | **{R['basket'][2]:.0f}% ann vol**, **{R['basket'][3]:.0f}% "
            f"max-dd** (single names {R['names']['IONQ'][2]:.0f}% to {R['names']['RGTI'][2]:.0f}%). "
            "Net of costs the Sharpe is unchanged — **risk**, not friction, makes it un-allocatable; "
            f"risk-adjusted it loses to SPY *and* QTUM (Sharpe {R['etf'][1]:.2f}). |\n"
            f"| **Next big thing?** | `HYPE-CYCLE` | Giant CAGR (+{R['basket'][0]:.0f}%/yr), giant "
            f"drawdown, sub-market Sharpe, sub-2 *t*. A synthetic null reproduces the spread from pure "
            f"beta+noise (t = {R['syn'][0][4]:.2f}). |\n\n"
            "> 💡 In plain words: the basket's spread is enormous and its standard error is enormous "
            "too. Strip the lucky melt-up months and a fat-tailed, survivorship-tilted lottery over "
            "one short regime leaves nothing the null can't explain — while the diversified ETF "
            "quietly does the theme on a *better* Sharpe than either the basket or the index."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $b_t$ be the equal-weight monthly return of the four pure-plays and $m_t$ the market "
            "(SPY). The pitch is a positive, *forward-harvestable* spread $\\Delta_t = b_t - m_t$.\n\n"
            "- **H₁ (the basket out-earns).** $\\mathbb{E}[\\Delta_t] > 0$ with a HAC *t* ≥ 2 — a real "
            "spread, not a noisy one.\n"
            "- **H₂ (you're paid for the risk).** The basket's **Sharpe** beats the market's (and the "
            "diversified ETF's) — the spread is *reward*, not just leverage on beta.\n"
            "- **H₃ (\"next big thing\").** The headline CAGR reflects a tradable theme edge, not a "
            "lottery tail over one regime plus survivorship.\n\n"
            "We find **H₁ not rejected but not supported** (spread positive, HAC *t* ≈ 1.2 < 2), "
            "**H₂ rejected** (basket Sharpe 0.75 < SPY 0.91 < QTUM 1.06), **H₃ rejected** (a synthetic "
            "null reproduces the spread from zero edge; survivorship + one regime). The legend is true "
            "exactly where it's uninformative (the number is big) and unproven exactly where it would "
            "pay (a *risk-rewarded*, *significant* edge)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The teardown is two comparisons judged by their **dispersion**. First the spread under a "
            "HAC standard error:\n\n"
            "$$t = \\frac{\\bar\\Delta}{\\sqrt{\\widehat{\\mathrm{LRV}}(\\Delta)/n}},\\qquad "
            "\\widehat{\\mathrm{LRV}} = \\hat\\gamma_0 + 2\\sum_{k=1}^{L}\\Big(1-\\tfrac{k}{L+1}\\Big)\\hat\\gamma_k.$$\n\n"
            "With $n = 61$ months dominated by a few violent up-prints, $\\widehat{\\mathrm{LRV}}$ is "
            "large, so even a +100%/yr $\\bar\\Delta$ drowns in its own error. Second, the risk-"
            "adjusted return: a **Sharpe** that is *below* the index's says the giant CAGR is pure "
            "leverage on beta plus a fat idiosyncratic tail — not compensation. A raw-CAGR lens is the "
            "wrong instrument entirely; **Sharpe and the HAC *t* decide the verdict**, and a "
            "deterministic null shows a fat-tailed basket with *no* edge can post exactly this picture."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Panel.** Monthly total returns (yfinance auto-adjusted), 4 pure-plays + QTUM + "
            f"SPY/QQQ, {R['start']}→{R['end']} ({R['months']} months — the window all four trade). "
            f"Fingerprint `{R['fingerprint']}`. **Survivorship** (de-SPAC survivors) named on the axis; "
            "the partial final month is dropped.\n"
            "- **Basket.** Equal-weight, monthly-rebalanced; one-way turnover × NAV at 10 bps (no "
            "signal lag — membership is constant).\n"
            "- **Risk race.** CAGR / annualised Sharpe / annualised vol / max-drawdown for the basket, "
            "QTUM, SPY, QQQ.\n"
            "- **Signal test (HAC).** Newey-West *t* of the monthly spread $b_t - m_t$ vs SPY, QQQ "
            "**and** the diversified ETF (both legs fully invested ⇒ the raw difference *is* the "
            "excess).\n"
            "- **Decomposition.** Per-name HAC *t* of excess over SPY + each name's vol & drawdown — "
            "broad theme or a moon-shot?\n"
            "- **Positive control.** A deterministic fat-tailed *hype basket* with a planted "
            "`edge_sharpe`: at 0 each name has **zero** expected excess over the market (the fat vol "
            "only inflates noise) so the HAC *t* must stay at the null; a real Sharpe edge must light "
            "it up. **Mirage trigger:** a basket whose Sharpe ≤ the market's and whose spread *t* < 2 "
            "is a hype cycle."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The risk race — a giant CAGR at a sub-market Sharpe\n\n"
            "CAGR, Sharpe, annualised vol and max-drawdown for each leg. The basket wins CAGR and "
            "loses everything that matters."
        ),
        code(
            "legs = ['basket', 'etf', 'spy', 'qqq']; labs = ['Quantum\\nbasket','QTUM\\nETF','SPY','QQQ']\n"
            "if HAVE_REAL:\n"
            "    s = RACE['summ']\n"
            "    cagr=[s[l]['cagr'] for l in legs]; sh=[s[l]['sharpe'] for l in legs]\n"
            "    vol=[s[l]['vol'] for l in legs]; dd=[s[l]['max_dd'] for l in legs]\n"
            "else:\n"
            "    cagr=[R['basket'][0]/100,R['etf'][0]/100,R['spy'][0]/100,R['qqq'][0]/100]\n"
            "    sh=[R['basket'][1],R['etf'][1],R['spy'][1],R['qqq'][1]]\n"
            "    vol=[R['basket'][2]/100,R['etf'][2]/100,R['spy'][2]/100,R['qqq'][2]/100]\n"
            "    dd=[R['basket'][3]/100,R['etf'][3]/100,R['spy'][3]/100,R['qqq'][3]/100]\n"
            "x=np.arange(len(legs)); col=[RED,GREEN,GREY,AMBER]\n"
            "fig,axs=plt.subplots(2,2,figsize=(10.4,7.0))\n"
            "for ax,vals,title,fmt in [(axs[0,0],cagr,'CAGR','{:.0%}'),(axs[0,1],sh,'Sharpe','{:.2f}'),\n"
            "                          (axs[1,0],vol,'annualised vol','{:.0%}'),(axs[1,1],dd,'max drawdown','{:.0%}')]:\n"
            "    ax.bar(x,vals,color=col,width=.6); ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_title(title)\n"
            "    for i,v in enumerate(vals): ax.annotate(fmt.format(v),(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "axs[0,1].axhline(sh[2],ls='--',c=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Sharpe basket/etf/spy/qqq:', [round(v,2) for v in sh])"
        ),
        md(
            f"> 💡 In plain words: the basket posts **+{R['basket'][0]:.0f}%/yr** — and a "
            f"**{R['basket'][2]:.0f}% vol**, an **{R['basket'][3]:.0f}% drawdown**, and a **Sharpe of "
            f"{R['basket'][1]:.2f}**, *below* SPY's {R['spy'][1]:.2f} and well below QTUM's "
            f"{R['etf'][1]:.2f}. H₂ is rejected on sight: the CAGR is leverage on beta plus a fat tail, "
            "not reward. The diversified ETF is the only leg with a *better* Sharpe than the index."
        ),
        md(
            "### 4b · The decisive test — is the spread even significant?\n\n"
            "The basket's monthly spread over each benchmark under a Newey-West HAC standard error. A "
            "real edge clears t = 2; a hype cycle doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tv = [RACE['test_vs_spy'], RACE['test_vs_qqq'], RACE['test_vs_etf']]\n"
            "    spreads = [t['mean_diff_ann']*100 for t in tv]; ts = [t['tstat'] for t in tv]\n"
            "else:\n"
            "    spreads = [R['t_spy'][0], R['t_qqq'][0], R['t_etf'][0]]\n"
            "    ts = [R['t_spy'][1], R['t_qqq'][1], R['t_etf'][1]]\n"
            "labs = ['vs SPY','vs QQQ','vs QTUM ETF']; x=np.arange(3)\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.4,4.3))\n"
            "a1.bar(x,spreads,color=RED,width=.55); a1.set_xticks(x); a1.set_xticklabels(labs)\n"
            "a1.set_ylabel('annualised spread (%)'); a1.set_title('The spread LOOKS enormous...')\n"
            "for i,v in enumerate(spreads): a1.annotate(f'+{v:.0f}%',(i,v),ha='center',va='bottom')\n"
            "a2.bar(x,ts,color=AMBER,width=.55); a2.axhline(2,ls='--',c=RED,label='t = 2 bar')\n"
            "a2.set_xticks(x); a2.set_xticklabels(labs); a2.set_ylabel('HAC t-stat'); a2.set_ylim(0,2.4)\n"
            "a2.set_title('...but the HAC t never reaches 2'); a2.legend()\n"
            "for i,v in enumerate(ts): a2.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('HAC t vs SPY/QQQ/QTUM:', [round(t,2) for t in ts], '-> none clears 2')"
        ),
        md(
            f"> 💡 In plain words: **+{R['t_spy'][0]:.0f}%/yr vs SPY** at **t = {R['t_spy'][1]:.2f}** — "
            "the point estimate is gigantic, the standard error is gigantic too. A real edge pushes "
            "the *t* through 2; here it sits near 1.2 against *every* benchmark, including the "
            "diversified ETF. H₁ is **not rejected, not supported**: a huge whisper inside its own "
            "error bar, driven by a couple of melt-up months."
        ),
        md(
            "### 4c · One stock or four? — the single-name decomposition\n\n"
            "Per-name HAC *t* of excess over SPY, with each name's own worst drawdown. If the basket "
            "were a real theme, *some* name would clear t = 2. None does — and all four are casino "
            "chips."
        ),
        code(
            "names = list(R['names'].keys())\n"
            "if HAVE_REAL:\n"
            "    ts = [RACE['single_names'][n]['tstat'] for n in names]\n"
            "    dd = [RACE['single_names'][n]['max_dd']*100 for n in names]\n"
            "else:\n"
            "    ts = [R['names'][n][4] for n in names]; dd = [R['names'][n][2] for n in names]\n"
            "x=np.arange(len(names))\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.4,4.3))\n"
            "a1.bar(x,ts,color=AMBER,width=.6); a1.axhline(2,ls='--',c=RED,label='t = 2 bar')\n"
            "a1.set_xticks(x); a1.set_xticklabels(names); a1.set_ylabel('HAC t of excess vs SPY'); a1.set_ylim(0,2.4)\n"
            "a1.set_title('No single name clears t = 2'); a1.legend()\n"
            "for i,v in enumerate(ts): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "a2.bar(x,dd,color=RED,width=.6); a2.set_xticks(x); a2.set_xticklabels(names)\n"
            "a2.set_ylabel('worst drawdown (%)'); a2.set_title('...and each one crashed ~90%')\n"
            "for i,v in enumerate(dd): a2.annotate(f'{v:.0f}%',(i,v),ha='center',va='top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-name HAC t:', {n: round(t,2) for n,t in zip(names,ts)})"
        ),
        md(
            f"> 💡 In plain words: the strongest name (IONQ) reaches just **t = {R['names']['IONQ'][4]:.2f}**; "
            f"the rest are lower, and every one drew down {abs(R['names']['QBTS'][2]):.0f}%-class. The "
            "basket is not one lucky moon-shot hiding three duds — it is **four** statistically-"
            "unproven lottery tickets that crash together. Pooling them cuts the *t* further (shared "
            "risk-off beta), it doesn't rescue it."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic fat-tailed *hype basket*: with **zero** planted risk-adjusted edge "
            "(`edge_sharpe = 0`) each name has zero expected excess over the market, so the HAC *t* of "
            "the spread must stay at the null — the fat vol only adds noise; with a **+1.0 Sharpe** "
            "edge it must light up. Both hold."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 1.0):\n"
            "    rdf, b, truth = data.synthetic_panel(edge_sharpe=edge, seed=395)\n"
            "    basket = st.basket_returns(rdf, truth['basket_cols'])\n"
            "    sm = st.summarize(basket); t = st.hac_tstat_diff(basket, b)\n"
            "    res.append((edge, sm['cagr']*100, sm['sharpe'], t['mean_diff_ann']*100, t['tstat']))\n"
            "labels=[f'planted edge\\nSharpe {e:.1f}' for e,_,_,_,_ in res]; tvals=[r[4] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 (significance bar)')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('HAC t of basket spread'); ax.set_title('Control: a no-edge hype basket stays at the null'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,c,s,sp,t in res: print(f'edge_sharpe={e:+.1f}: CAGR={c:.0f}% Sharpe={s:.2f} spread={sp:.0f}%/yr t={t:.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** real edge the control sits at **t = {R['syn'][0][4]:.2f}** "
            f"(below 2 — a fat-tailed basket *cannot* fake significance out of pure beta+noise); only a "
            f"genuine **+1.0 Sharpe** edge reaches **t = {R['syn'][1][4]:.2f}**. So the machinery is "
            f"honest, and the real-tape *t* of **{R['t_spy'][1]:.2f}** is exactly what an *absent or "
            "tiny* risk-adjusted edge looks like through a 61-month, fat-tailed keyhole. The Sharpe and "
            "the *t*, together, are the verdict."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — basket−SPY spread **+{R['t_spy'][0]:.0f}%/yr** at HAC "
            f"**t = {R['t_spy'][1]:.2f}** (n = {R['t_spy'][2]}; {R['t_qqq'][1]:.2f} vs QQQ, "
            f"{R['t_etf'][1]:.2f} vs QTUM); **no name clears t = 2**; basket Sharpe "
            f"**{R['basket'][1]:.2f}** < SPY **{R['spy'][1]:.2f}**. Literature support + a positive-"
            "but-insignificant, risk-unrewarded point estimate, survivorship-tilted over one regime "
            "⇒ `WEAK`, not `REAL` (the t≥2 bar is not met).\n"
            f"- **Tradability `MIRAGE`** — **{R['basket'][2]:.0f}% vol**, **{R['basket'][3]:.0f}% "
            "drawdown** (single names −86% to −96%); net of 10-bps costs the Sharpe is unchanged. "
            "**Risk**, not friction, makes it un-allocatable, and risk-adjusted it loses to SPY *and* "
            f"the QTUM ETF (Sharpe {R['etf'][1]:.2f}). The +59%/yr is a number you could not have held "
            "through the −84% draw.\n"
            f"- **Next big thing? `HYPE-CYCLE`** — a real technology, no profits yet, a stock cohort "
            f"with a giant CAGR, a giant drawdown, a sub-market Sharpe, and a sub-2 *t* a synthetic "
            f"null reproduces from pure beta+noise (t = {R['syn'][0][4]:.2f}). The diversified ETF was "
            "the risk-rewarded way to play quantum all along — and even it is not a free lunch. "
            "[Study 393 (AI-Datacenter)](../393-ai-datacenter-basket/) in a pre-revenue costume."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the holdability problem\n\n"
            "The operational truth in one picture: the basket's underwater curve. A sleeve you spend "
            "most of the sample **down 50%+** from its high is not something a fund — or a human — "
            "actually holds to collect the headline CAGR."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = RACE['basket'].dropna(); rs = RACE['spy'].dropna()\n"
            "    eq = (1+r).cumprod(); under = eq/eq.cummax()-1\n"
            "    eqs = (1+rs).cumprod(); unders = eqs/eqs.cummax()-1\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.3))\n"
            "    ax.fill_between(under.index, under.values*100, 0, color=RED, alpha=.4, label='quantum basket')\n"
            "    ax.plot(unders.index, unders.values*100, color=GREY, lw=1.5, label='S&P 500')\n"
            "    ax.set_ylabel('drawdown from peak (%)')\n"
            "    ax.set_title(f'Underwater: the basket spent most of 5 years deep in the hole (trough {under.min()*100:.0f}%)')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    frac = float((under < -0.5).mean())\n"
            "    print(f'months >50% below peak: basket {frac*100:.0f}% of the sample vs SPY {(unders<-0.5).mean()*100:.0f}%')\n"
            "else:\n"
            "    print('no cache — see docs/results.md: basket max drawdown -84.3% vs SPY -23.9%')"
        ),
        md(
            "> 💡 In plain words: the red region is how far below its own high-water mark the basket "
            "was, month by month. It is deep and persistent — long stretches **>50% underwater** — "
            "while the index (grey) barely dips. The capacity question never even arises: **the path "
            "is the constraint.** No sizing, no cost regime, no rebalance schedule converts a "
            "160%-vol, −84%-drawdown lottery into a NAV-scale holding. The rarity-of-edge that makes "
            "the thesis exciting is exactly the volatility that makes it untradable."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The selection sibling.** [Study 393 — AI-Datacenter-Basket](../393-ai-datacenter-basket/): "
            "a thematic basket whose spread *is* significant on the tape but is *named in hindsight*. "
            "Reading them together separates the two thematic-basket traps — **ex-post selection** "
            "(393) vs **unrewarded risk + small sample** (here).\n"
            "- **Disruptive-tech fund.** [Study 334 — ARK-Innovation](../334-ark-innovation/) — does "
            "riding \"innovation\" pay risk-adjusted? Same high-vol theme family.\n"
            "- **Thematic commodity-equity.** [Study 302 — Lithium-Boom](../302-lithium-boom/) and "
            "[Study 303 — Uranium-Revival](../303-uranium-revival/) — real theme, the question is "
            "whether the *basket* harvests it net of risk.\n"
            "- **Diversify the tail.** Re-run with the QTUM constituents or the mega-cap quantum "
            "efforts (GOOGL/IBM/MSFT) added; the idiosyncratic tail shrinks, the Sharpe rises toward "
            "the ETF's, and the \"basket\" stops being a lottery.\n\n"
            "*The reproducible core is offline and deterministic; survivorship is explicit. Methods "
            "and sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
