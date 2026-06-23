"""Generate the two narrative notebooks for Study 400 (Patent-Intensity).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached EDGAR intensity +
yfinance returns under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (EDGAR R&D-intensity proxy + SPY,
# 40-name field, 2005-02 -> 2026-05, 256 months, 21.3 years, fingerprint 66d4e99b7bf1).
R = dict(
    start="2005-02", end="2026-05", months=256, years=21.3, field=40, intensity_names=39,
    fingerprint="66d4e99b7bf1",
    long_members=["ADBE", "AMGN", "BMY", "CSCO", "GILD", "IBM", "INTC", "LLY", "MRK", "ORCL",
                  "PFE", "QCOM", "TXN"],
    short_members=["AXP", "C", "COST", "DUK", "HD", "KO", "LOW", "MCD", "MMM", "T", "USB", "VZ",
                   "WMT"],
    # books: (cagr%, sharpe, maxdd%, mean_ann%)
    long=(13.64, 0.94, -36.9, 13.95),
    short=(9.71, 0.72, -46.1, 10.35),
    long_short=(2.98, 0.31, -26.6, 3.60),
    spy=(11.11, 0.78, -50.8, 11.70),
    long_net_cagr=13.61,
    # signal tests: (mean_ann%, hac_t, n)
    test_ls=(3.60, 1.50, 256),
    test_long_spy=(2.25, 1.21, 256),
    test_short_spy=(-1.35, -0.76, 256),
    # random blind long/short control
    rand_mean=0.08, rand_med=0.05, rand_sd=1.77, ls_pctile=97.9,
    # robustness: (frac, k_per_leg, ls_mean_ann%, hac_t)
    robust=[(0.50, 20, 4.12, 2.18), (1 / 3, 13, 3.60, 1.50),
            (0.25, 10, 3.28, 1.29), (0.20, 8, 3.28, 1.13)],
    # costs+borrow: gross and net long-short (mean_ann%, t)
    ls_gross=(3.60, 1.50), ls_net=(2.60, 1.08),
    # synthetic: (edge, ls_mean_ann%, hac_t)
    syn=[(0.0, -1.34, -0.69), (0.06, 4.33, 2.23)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Innovation_premium%3F: Misattributed](https://img.shields.io/badge/Innovation_premium%3F-Misattributed-8b949e?style=flat-square)\n\n"
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

from patent_intensity import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    INTEN, RETS, SPY = data.load_real(allow_survivorship_bias=True)
    RACE = st.race(INTEN, RETS, SPY, frac=1/3, report_lag=1, cost_bps=10.0,
                   borrow_bps=100.0, n_draws=2000)
else:
    INTEN = RETS = SPY = RACE = None
print("real EDGAR+yfinance cache present:", HAVE_REAL,
      "| field names:", (0 if RETS is None else RETS.shape[1]))
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell can
# quote it whether or not the cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Buy the inventors, short the dinosaurs — is there an *innovation premium*? 💡\n"
            "### Do the companies that spend the most on inventing quietly beat the market?\n\n"
            + BADGES +
            "Here's a story that sells funds: the firms pouring the most into **research and "
            "development** — the chip designers, the drug labs, the software houses — are building "
            "moats the market is too slow to price. So a portfolio that goes **long the most "
            "R&D-intensive innovators and short the least-intensive incumbents** (the banks, the "
            "soda makers, the oil majors) should quietly harvest an *innovation premium*.\n\n"
            "It's a lovely idea with real academic papers behind it. This notebook builds the "
            "portfolio on audited data and asks the only two questions that matter: **is the extra "
            "return actually there**, and **is it really about inventing — or is it just 'tech beats "
            "value' wearing a lab coat?**\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the random-split control and the "
            "borrow-cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Real *patent counts* aren't free, so we use the standard "
            "stand-in: **R&D spending ÷ revenue** (\"R&D intensity\") straight from companies' SEC "
            "filings — we call it a proxy throughout. Every chart is drawn by the code beside it; "
            "house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do the big spenders beat the penny-pinchers? | **Yes, a bit.** The high-R&D tertile "
            f"out-earns the low-R&D tertile by **+{R['test_ls'][0]:.1f}%/yr**. |\n"
            "| Is that a real edge? | **Can't say.** Over 21 years the gap's *t*-stat is "
            f"**{R['test_ls'][1]:.2f}** — below the **2** you need to rule out luck. It only crosses "
            "2 when you *barely* sort, and fades the moment you concentrate. |\n"
            "| Could you trade it? | **Not really.** Shorting the low-R&D names costs **borrow**, "
            f"which drags the gap to **+{R['ls_net'][0]:.1f}%/yr at t = {R['ls_net'][1]:.2f}** — "
            "indistinguishable from zero. |\n"
            "| Then is it an *innovation* premium at all? | **No — it's mislabelled.** The high-R&D "
            "side *is* tech & pharma; the low-R&D side *is* banks & staples. You've reinvented the "
            "**growth-vs-value** tilt and called it 'innovation.' |\n\n"
            "> R&D intensity sorts stocks onto a real axis — but that axis is *style*, not a secret "
            "innovation edge, and it isn't even statistically significant on its own."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The companies inventing the future — the highest R&D-and-patent intensity — are "
            "compounding intangible moats the market under-prices. Hold the inventors, short the "
            "dinosaurs, and collect the innovation premium.\"*\n\n"
            "It isn't a crank claim. Chan, Lakonishok & Sougiannis (2001) found R&D-heavy firms can "
            "earn higher subsequent returns; the whole 'disruption / innovation factor' ETF industry "
            "is built on the vibe. The seduction is intuitive — *surely* the firms building tomorrow "
            "are a better bet than the ones milking yesterday. We'll build exactly that bet and look."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a clean *innovation premium* existed, it would be a free lunch you could harvest "
            "forever: rank the market by R&D intensity once a year, tilt toward the top, and beat the "
            "index with a story every client loves. But two traps hide inside it. **One:** the most "
            "R&D-intensive firms are almost all *tech, semis and pharma* — so 'long innovation' is "
            "secretly 'long growth,' a tilt you can already buy for a few basis points. **Two:** a "
            "long/short has to *short* the low-R&D names, and shorting isn't free — you pay to borrow "
            "them. A premium that's really style beta, and that borrow costs eat, isn't an edge — "
            "it's a relabelled, leaky version of something you already own."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We build the portfolio on **audited** numbers. For a fixed **{R['field']}-name** "
            "large-cap field — chosen by *sector* so the mix spans inventors and incumbents, **not** "
            "by who won — we pull each firm's **R&D ÷ revenue** from its SEC filings (the patent "
            "proxy). Each year:\n\n"
            "1. **Rank & sort.** Go *long* the third with the highest R&D intensity, *short* the "
            "third with the lowest — using only data the filings had *already reported* (a one-year "
            "lag, no peeking ahead).\n"
            "2. **Measure the gap.** Over 21 years, does long-minus-short actually pay — and is the "
            "gap big enough, relative to its wobble, to not be luck?\n"
            "3. **Check it's not just 'any split'.** Compare the R&D split to *thousands of random* "
            "long/shorts of the same names. If a random split pays the same, R&D isn't doing the "
            "work. If only the R&D split pays — but it's the tech-vs-bank axis — it's *style*, not "
            "invention."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, who's on each side?** Here's the long (high-R&D) and short (low-R&D) tertile — "
            "and notice it reads like a sector map, not a stock-picker's secret list."
        ),
        code(
            "if HAVE_REAL:\n"
            "    last = max(RACE['members']); lo, sh = RACE['members'][last]\n"
            "    x = st.known_intensity(INTEN, last)\n"
            "else:\n"
            "    lo, sh = R['long_members'], R['short_members']\n"
            "    x = None\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "if x is not None:\n"
            "    xl = x[lo].sort_values(); xs = x[sh].sort_values()\n"
            "    ax.barh([f'{t}' for t in xl.index], xl.values*100, color=GREEN, label='LONG  high-R&D')\n"
            "    ax.barh([f'{t}' for t in xs.index], -xs.values*100-2, color=GREY, label='SHORT low-R&D')\n"
            "    ax.set_xlabel('R&D / revenue  (%)   — long side right, short side left')\n"
            "else:\n"
            "    ax.barh(lo, [1]*len(lo), color=GREEN, label='LONG  high-R&D (tech/pharma)')\n"
            "    ax.barh(sh, [-1]*len(sh), color=GREY, label='SHORT low-R&D (banks/staples)')\n"
            "    ax.set_xlabel('(install the cache to see real intensities)')\n"
            "ax.axvline(0, c='k', lw=.8); ax.set_title('The R&D split is a sector map: tech/pharma vs banks/staples')\n"
            "ax.legend(loc='lower right'); plt.tight_layout(); plt.show()\n"
            "print('LONG :', lo)\nprint('SHORT:', sh)"
        ),
        md(
            "Long: chips (INTC, TXN, QCOM, CSCO), software (ADBE, ORCL, IBM), pharma/biotech (LLY, "
            "MRK, PFE, BMY, AMGN, GILD). Short: banks (C, USB, AXP), staples & retail (KO, WMT, "
            "COST, HD, LOW, MCD), telco/utility (T, VZ, DUK), industrials (MMM). That's not an "
            "innovation list — **that's the growth-vs-value lineup.** Hold that thought."
        ),
        md(
            "**Does the gap pay?** The long (high-R&D), the short leg (low-R&D), the long-minus-short "
            "spread, and the market — growth of \\$1 over 21 years."
        ),
        code(
            "if HAVE_REAL:\n"
            "    L, S, LS, MKT = RACE['long'], RACE['short'], RACE['long_short'], RACE['spy']\n"
            "    eqs = {'Long (high-R&D)': (1+L).cumprod(), 'Short leg (low-R&D)': (1+S).cumprod(),\n"
            "           'SPY (market)': (1+MKT).cumprod()}\n"
            "    idx = L.index\n"
            "else:\n"
            "    idx = np.arange(R['months'])\n"
            "    g = lambda m: (1+np.full(R['months'], (1+m/100)**(1/12)-1)).cumprod()\n"
            "    eqs = {'Long (high-R&D)': g(R['long'][3]), 'Short leg (low-R&D)': g(R['short'][3]),\n"
            "           'SPY (market)': g(R['spy'][3])}\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "for (name, eq), c in zip(eqs.items(), [GREEN, GREY, RED]):\n"
            "    ax.plot(idx, np.asarray(eq), c=c, lw=2, label=name)\n"
            "ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "ax.set_title(f\"High-R&D beats low-R&D by ~{R['test_ls'][0]:.1f}%/yr — but only just\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"long CAGR {R['long'][0]:.1f}%  short CAGR {R['short'][0]:.1f}%  \"\n"
            "      f\"spread {R['long_short'][3]:+.1f}%/yr  SPY {R['spy'][0]:.1f}%\")"
        ),
        md(
            f"The high-R&D side *does* finish ahead — **+{R['test_ls'][0]:.1f}%/yr** over the low-R&D "
            "side. So the directional claim isn't nothing. The question is whether "
            f"**+{R['test_ls'][0]:.1f}%/yr** over 21 wobbly years is a *signal* or just the kind of "
            "gap luck throws up. That's the next chart."
        ),
        md(
            "**Is +3.6%/yr more than luck — and more than *any* split?** We draw 2,000 *random* "
            "long/shorts from the same names (no R&D used) and see where the R&D split lands."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rand = RACE['rand_ls_spread']*100; obs = R['test_ls'][0]; pct = RACE['ls_pctile']\n"
            "else:\n"
            "    rng = np.random.default_rng(400); rand = rng.normal(R['rand_mean'], R['rand_sd'], 2000)\n"
            "    obs = R['test_ls'][0]; pct = R['ls_pctile']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(rand, bins=50, color=GREY, alpha=.85, label='2,000 RANDOM long/shorts (no R&D)')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'the R&D split ({obs:+.1f}%/yr)')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('long-short spread (%/yr)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'R&D sorts onto a REAL axis (it beats {pct:.0f}% of random splits)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'random split pays ~0%/yr; the R&D split pays {obs:+.1f}%/yr -> R&D is NOT a generic split')\n"
            "print(f\"...but its OWN t-stat is {R['test_ls'][1]:.2f} (below 2): real axis, not significant return\")"
        ),
        md(
            f"Two things at once, and both are honest. The R&D split lands at the "
            f"**{R['ls_pctile']:.0f}th percentile** of random splits — so R&D is **not** a coin "
            "flip; it really does pick a persistent axis a random split misses. **But** that axis is "
            "the *tech-vs-value* axis (look at the names), and its own return gap, judged over 21 "
            f"years, has a *t*-stat of just **{R['test_ls'][1]:.2f}** — below the **2** bar. A real "
            "*style* tilt, not a significant *innovation* edge."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** High-R&D beats low-R&D by **+{R['test_ls'][0]:.1f}%/yr** and the "
            f"split is a real axis — but the gap fails the significance bar (HAC "
            f"*t* = {R['test_ls'][1]:.2f}) and only clears it when you *barely* sort. Real-ish, but "
            "not certifiable.\n"
            f"- **Tradability — Mirage.** Shorting the low-R&D names costs borrow, which drops the "
            f"gap to **+{R['ls_net'][0]:.1f}%/yr at t = {R['ls_net'][1]:.2f}** — a net edge "
            "indistinguishable from zero.\n"
            "- **An innovation premium? — Misattributed.** The high side *is* tech/pharma, the low "
            "side *is* banks/staples. You haven't found an innovation factor — you've relabelled "
            "**growth-vs-value**."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the borrow bites\n\n"
            "Forget significance for a second. Even taking the +3.6%/yr at face value, a long/short "
            "must *short* the low-R&D names — and you pay a borrow fee to do it. Here's the gap "
            "before and after a modest 1%/yr borrow on the short leg."
        ),
        code(
            "g, gt = R['ls_gross']; n, nt = R['ls_net']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "bars = ax.bar(['gross\\n(no borrow)', 'net\\n(+1%/yr borrow)'], [g, n],\n"
            "              color=[GREEN, RED], width=.55)\n"
            "for b, v, t in zip(bars, [g, n], [gt, nt]):\n"
            "    ax.annotate(f'{v:+.1f}%/yr\\nt={t:.2f}', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('long-short spread (%/yr)')\n"
            "ax.set_ylim(0, max(g, n)*1.35)\n"
            "ax.set_title('Borrow turns an insignificant +3.6%/yr into a still-insignificant +2.6%/yr')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.1f}%/yr (t={gt:.2f}) -> net {n:+.1f}%/yr (t={nt:.2f}): both below t=2')"
        ),
        md(
            "Turnover is trivial (you rebalance once a year), so trading cost isn't the killer. "
            "**Borrow is.** And the long-*only* version — just hold the high-R&D names, no shorting "
            f"— beats SPY by a mere **+{R['test_long_spy'][0]:.1f}%/yr (t = {R['test_long_spy'][1]:.2f})**, "
            "almost all of which is the growth tilt you can buy as a cheap style ETF. There's no "
            "deployable innovation edge to harvest here."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **It's the *efficiency*, not the *spending*.** Hirshleifer-Hsu-Li (2013) show the "
            "premium that survives is **patents per R&D dollar** (innovative *efficiency*), not gross "
            "R&D intensity. Our proxy measures input, not output — swap in real patent/citation data "
            "and re-test.\n"
            "- **Strip the style.** Regress the long-short on a growth-vs-value factor and ask if any "
            "*residual* innovation alpha survives. (Our random-split control already hints the answer "
            "is 'mostly style.')\n"
            "- **Thematic siblings.** [Study 393 — AI-Datacenter-Basket](../../393-ai-datacenter-basket/) "
            "is the *selection* version of the same trap; here the trap is *attribution*.\n\n"
            "*Think the innovation premium is real and distinct from growth? Show the long-short "
            "clearing t = 2 on a concentrated split, net of borrow, with the style beta stripped out "
            "— then we'll talk.*"
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
            "# Patent-Intensity — a quantitative teardown 🔬\n"
            "### R&D-intensity ranking with a reporting lag · long-short & long-minus-SPY HAC *t* · "
            "a random-blind-split sector control · fraction robustness · costs + short-borrow · a "
            "synthetic planted-premium / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We split "
            "the 'innovation premium' into the two questions a label fuses: is the long-high / "
            "short-low R&D-intensity spread **statistically real** (HAC *t* ≥ 2 on the tape), and is "
            "it a *distinct innovation factor* or **sector/style beta** that costs and short-borrow "
            "erase? The decisive objects are the **HAC t** of the spread, the **random-split** "
            "control (is it the R&D signal or any split of a mixed field?), and a **fraction sweep** "
            "(does it survive concentration?).\n\n"
            "> ⚠️ **Data + proxy note.** Issued-patent counts aren't on a free feed; we use **reported "
            "R&D / revenue** (SEC EDGAR `companyfacts`) as the audited proxy — input intensity, not "
            "output quality (the distinction Hirshleifer-Hsu-Li 2013 show is decisive). Returns: "
            "yfinance monthly total return, 2005→2026. Survivorship is named on the Signal axis (the "
            "basket is current-membership, but the bias is largely *common to both legs* of the "
            "long/short). Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `WEAK` | long-short **+{R['test_ls'][0]:.2f}%/yr**, HAC "
            f"**t = {R['test_ls'][1]:.2f}** (n = {R['test_ls'][2]}); clears t = 2 *only* at the "
            f"\"halves\" split ({R['robust'][0][3]:.2f}), fails at tertile/quartile/quintile. |\n"
            f"| **Tradability** | `MIRAGE` | net of 10 bps turnover **+ 100 bps/yr short-borrow** the "
            f"spread is **+{R['ls_net'][0]:.2f}%/yr at t = {R['ls_net'][1]:.2f}** — zero, "
            "statistically. Long-only beats SPY by only "
            f"+{R['test_long_spy'][0]:.2f}%/yr (t = {R['test_long_spy'][1]:.2f}). |\n"
            f"| **Innovation premium?** | `MISATTRIBUTED` | the split sits at the "
            f"**{R['ls_pctile']:.0f}th pct** of random splits — a *real* axis, but it is the "
            "growth/tech-vs-value style axis (long = semis/software/pharma, short = "
            "banks/staples/retail), not a patent-specific alpha. |\n\n"
            "> 💡 In plain words: R&D intensity sorts stocks onto a genuine cross-sectional axis (not "
            "noise) — but that axis is *style*, the spread doesn't clear significance, and borrow "
            "finishes whatever is left."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $x_{i,y}$ be firm $i$'s R&D intensity (R&D / revenue) reported for fiscal year $y$. "
            "Form, at the start of year $Y$, the long-high / short-low book from $x_{i,Y-1}$ (a "
            "one-year reporting lag — no look-ahead):\n\n"
            "$$\\text{LS}_Y = \\frac1k\\!\\!\\sum_{i\\in \\text{top-}k} r_{i,Y} \\;-\\; "
            "\\frac1k\\!\\!\\sum_{i\\in \\text{bot-}k} r_{i,Y}.$$\n\n"
            "- **H₁ (the premium is real).** $\\mathbb{E}[\\text{LS}] > 0$ with a HAC "
            "$t \\ge 2$ — a *significant* innovation spread.\n"
            "- **H₂ (it's deployable).** The spread survives one-way costs **and the borrow you pay "
            "to be short**, and holds up when you *concentrate* (tighter tertiles).\n"
            "- **H₃ (it's *innovation*).** The spread is distinct from the style/sector axis — it is "
            "not reproduced by 'long tech, short banks' generically.\n\n"
            "We find **H₁ not supported** (+3.60%/yr but $t = 1.50$, and significant only at the "
            "barely-sorted halves), **H₂ rejected** (net of borrow $t = 1.08$; long-only is mostly "
            "growth beta), **H₃ rejected** (the spread *is* the style axis — 97.9th pct of random "
            "splits, but those names *are* growth-vs-value). The label is true where it's "
            "uninformative (tech has out-earned value) and unproven where it would pay (a *distinct, "
            "significant* invention alpha)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The teardown is one decomposition: a long-short mean against zero, judged by its **HAC "
            "standard error**, and against the **sampling distribution of any split** of the field.\n\n"
            "$$t_{\\text{HAC}} = \\frac{\\overline{\\text{LS}}}{\\widehat{\\text{se}}_{\\text{NW}}"
            "(\\overline{\\text{LS}})}, \\qquad "
            "\\text{pctile} = \\Pr_{\\text{random split}}\\!\\big[\\overline{\\text{LS}}_{\\text{rand}} "
            "< \\overline{\\text{LS}}_{\\text{R\\&D}}\\big].$$\n\n"
            "A high **percentile** says R&D loads on a real axis (not noise); a low **t** says that "
            "axis's *return* gap is inside its own error bar. Both can be true at once — and here they "
            "are. The honest read needs both numbers: the percentile kills 'it's just luck,' the "
            "*t* kills 'it's a certified edge,' and the names kill 'it's about *invention*.'"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Intensity panel.** Annual `ResearchAndDevelopmentExpense` / revenue from SEC EDGAR "
            f"`companyfacts` for a fixed {R['field']}-name field ({R['intensity_names']} with a "
            "series), 10-K full-year facts only; missing R&D floored to 0 (banks genuinely don't "
            "patent). **Proxy** for patent intensity — input, not output.\n"
            "- **Books.** Each year, equal-weight top-tertile (long) / bottom-tertile (short) by "
            "intensity *known at formation* (fiscal-year $Y-1$, a 1-year reporting lag). Annual "
            "rebalance.\n"
            "- **Null #1 (HAC t).** Newey-West t of the monthly long-short and long-minus-SPY means "
            "(`REAL` needs $t \\ge 2$).\n"
            "- **Null #2 (random split).** 2,000 blind $k$-long/$k$-short draws of the same names; "
            "where the R&D split sits is the sector/heterogeneity control.\n"
            "- **Robustness.** Fraction sweep halves→quintiles; report-lag 1 vs 2.\n"
            "- **Costs.** 10 bps one-way turnover **+ a 100 bps/yr borrow on the short leg** (a "
            "long/short pays to be short).\n"
            "- **Positive control.** A deterministic panel with a *planted* annual premium `edge`: "
            "the harness must recover a large edge **and** must NOT manufacture significance at "
            "`edge = 0`."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The point estimate — positive, small, and it fails the bar\n\n"
            "The long-short and long-minus-SPY spreads with their HAC *t*. Positive, but neither "
            "clears 2; the short leg merely lags the market."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [('Long - short\\n(innovation)', RACE['test_ls']),\n"
            "            ('Long - SPY', RACE['test_long_vs_spy']),\n"
            "            ('Short - SPY', RACE['test_short_vs_spy'])]\n"
            "    labels = [r[0] for r in rows]; means = [r[1]['mean_ann']*100 for r in rows]\n"
            "    ts = [r[1]['tstat'] for r in rows]\n"
            "else:\n"
            "    labels = ['Long - short\\n(innovation)', 'Long - SPY', 'Short - SPY']\n"
            "    means = [R['test_ls'][0], R['test_long_spy'][0], R['test_short_spy'][0]]\n"
            "    ts = [R['test_ls'][1], R['test_long_spy'][1], R['test_short_spy'][1]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))\n"
            "cols = [GREEN if m > 0 else GREY for m in means]\n"
            "a1.bar(labels, means, color=cols, width=.6); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_ylabel('mean spread (%/yr)'); a1.set_title('Spreads: small, mostly the long leg')\n"
            "for i, m in enumerate(means): a1.annotate(f'{m:+.1f}', (i, m), ha='center', va='bottom')\n"
            "a2.bar(labels, [abs(t) for t in ts], color=AMBER, width=.6)\n"
            "a2.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i, t in enumerate(ts): a2.annotate(f't={t:.2f}', (i, abs(t)), ha='center', va='bottom')\n"
            "a2.set_ylabel('|HAC t|'); a2.set_ylim(0, 2.4); a2.set_title('None of them clears t = 2'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('long-short', f\"{R['test_ls'][0]:+.2f}%/yr t={R['test_ls'][1]:.2f}\",\n"
            "      '| long-SPY', f\"{R['test_long_spy'][0]:+.2f}%/yr t={R['test_long_spy'][1]:.2f}\")"
        ),
        md(
            f"> 💡 In plain words: the innovation spread is **+{R['test_ls'][0]:.2f}%/yr** at "
            f"**t = {R['test_ls'][1]:.2f}** — positive but *not* significant. And it's lopsided: "
            f"long-minus-SPY is +{R['test_long_spy'][0]:.2f}%/yr (t = {R['test_long_spy'][1]:.2f}) "
            f"while short-minus-SPY is {R['test_short_spy'][0]:.2f}%/yr "
            f"(t = {R['test_short_spy'][1]:.2f}) — the 'premium' is mostly the long leg's mild "
            "out-performance, not a clean two-sided edge."
        ),
        md(
            "### 4b · Is it the R&D signal, or any split of a mixed field?\n\n"
            "Draw 2,000 *blind* long/shorts (same leg sizes, names random, **no R&D used**). Where "
            "the R&D split lands tells us whether intensity carries information — and the answer is a "
            "careful *yes, but*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rand = RACE['rand_ls_spread']*100; obs = RACE['test_ls']['mean_ann']*100\n"
            "    pct = RACE['ls_pctile']\n"
            "else:\n"
            "    rng = np.random.default_rng(400); rand = rng.normal(R['rand_mean'], R['rand_sd'], 2000)\n"
            "    obs = R['test_ls'][0]; pct = R['ls_pctile']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(rand, bins=60, color=GREY, alpha=.85, label='2,000 blind random long/shorts')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'R&D split {obs:+.1f}%/yr')\n"
            "ax.axvline(np.mean(rand), c=RED, ls='--', label=f'random mean {np.mean(rand):+.1f}%/yr')\n"
            "ax.set_xlabel('long-short spread (%/yr)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'R&D split is at the {pct:.0f}th pct — a REAL axis, not a generic split')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'random split ~0%/yr (no info); R&D split {obs:+.1f}%/yr at {pct:.0f}th pctile')\n"
            "print('=> R&D loads on a persistent axis -- but the names show that axis is growth-vs-value (style)')"
        ),
        md(
            f"> 💡 In plain words: a *random* split pays ≈0%/yr, so the R&D split's "
            f"+{R['test_ls'][0]:.1f}%/yr (the **{R['ls_pctile']:.0f}th percentile**) is **not** a "
            "generic artefact — intensity really does load on a persistent cross-sectional axis. The "
            "catch is *which* axis: the long names are semis/software/pharma and the short names are "
            "banks/staples/retail. That is the **growth-vs-value** axis. The control rescues R&D from "
            "'pure noise' and convicts it of 'style beta' in the same breath. Combined with 4a's "
            "$t = 1.50$: a real *style* tilt, an unproven *return*."
        ),
        md(
            "### 4c · Robustness — fragile to specification\n\n"
            "Tighten the sort from halves to quintiles. The spread clears $t = 2$ **only** at the "
            "coarsest split (where each 'leg' is half the field — barely a tilt); every concentrated, "
            "tradable split fails."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob = []\n"
            "    for frac in (0.50, 1/3, 0.25, 0.20):\n"
            "        b = st.intensity_books(INTEN, RETS, frac=frac)\n"
            "        t = st.hac_tstat(b['long_short']); k = len(b['members'][max(b['members'])][0])\n"
            "        rob.append((frac, k, t['mean_ann']*100, t['tstat']))\n"
            "else:\n"
            "    rob = R['robust']\n"
            "labels = ['halves', 'tertiles', 'quartiles', 'quintiles']\n"
            "ks = [r[1] for r in rob]; tt = [r[3] for r in rob]; ms = [r[2] for r in rob]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "bars = ax.bar([f'{l}\\nk={k}' for l, k in zip(labels, ks)], tt,\n"
            "              color=[GREEN if t >= 2 else AMBER for t in tt], width=.6)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for b, m, t in zip(bars, ms, tt):\n"
            "    ax.annotate(f'{m:+.1f}%/yr\\nt={t:.2f}', (b.get_x()+b.get_width()/2, t), ha='center', va='bottom')\n"
            "ax.set_ylabel('HAC t of long-short'); ax.set_ylim(0, 2.6)\n"
            "ax.set_title('Clears t=2 only when you barely sort; fails when you concentrate'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (frac, k, mean%/yr, t):', [(round(r[0],2), r[1], round(r[2],2), round(r[3],2)) for r in rob])"
        ),
        md(
            f"> 💡 In plain words: at the **halves** split (k = {R['robust'][0][1]}/leg — basically "
            f"'top half vs bottom half') the t is {R['robust'][0][3]:.2f}, just over the bar. Sort "
            f"into **tertiles** and it's {R['robust'][1][3]:.2f}; **quintiles**, "
            f"{R['robust'][3][3]:.2f}. A genuine factor gets *stronger* as you concentrate on its "
            "extremes; this gets *weaker*. That inversion is the signature of a `WEAK`, "
            "specification-fragile effect (and the report-lag isn't the culprit — lag 2 gives "
            "t = 1.53)."
        ),
        md(
            "### 4d · Costs + short-borrow — the little that's there, leaks\n\n"
            "Turnover is trivial (annual rebalance), but a long/short pays **borrow** on its short "
            "leg. A modest 100 bps/yr borrow drops the already-insignificant gross spread further."
        ),
        code(
            "g, gt = R['ls_gross']; n, nt = R['ls_net']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "bars = ax.bar(['gross', 'net\\n(+100 bps/yr borrow)'], [g, n], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('long-short spread (%/yr)'); ax.set_ylim(0, g*1.4)\n"
            "for b, v, t in zip(bars, [g, n], [gt, nt]):\n"
            "    ax.annotate(f'{v:+.2f}%/yr\\nt={t:.2f}', (b.get_x()+b.get_width()/2, v), ha='center', va='bottom')\n"
            "ax.set_title('Borrow erodes an insignificant spread to a more-insignificant one')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.2f}%/yr (t={gt:.2f}) -> net {n:+.2f}%/yr (t={nt:.2f})')"
        ),
        md(
            f"> 💡 In plain words: gross **+{R['ls_gross'][0]:.2f}%/yr (t = {R['ls_gross'][1]:.2f})** → "
            f"net **+{R['ls_net'][0]:.2f}%/yr (t = {R['ls_net'][1]:.2f})**. The borrow isn't the only "
            "problem — the gross was never significant — but it confirms there is no costed, "
            "shortable edge to allocate to. The deployable object is the long-only growth tilt, which "
            "a style ETF gives you cheaper."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "A deterministic panel (40 names, 18 years) where the first half are persistently "
            "high-intensity, the market beta is common to both legs, and `edge` plants the *true* "
            "annual long-high-minus-short-low premium. With `edge = 0` the long-short must stay "
            "insignificant; a large planted edge must light up."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.06):\n"
            "    i2, r2, b2, truth = data.synthetic_panel(edge=edge)\n"
            "    bk = st.intensity_books(i2, r2); t = st.hac_tstat(bk['long_short'])\n"
            "    res.append((edge, t['mean_ann']*100, t['tstat']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'edge = {e*100:.0f}%/yr\\n({\"NULL\" if e==0 else \"large planted\"})' for e,_,_ in res]\n"
            "tvals = [r[2] for r in res]\n"
            "bars = ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 bar'); ax.axhline(0, c='k', lw=.8)\n"
            "for b, (e, m, t) in zip(bars, res):\n"
            "    ax.annotate(f'{m:+.1f}%/yr\\nt={t:.2f}', (b.get_x()+b.get_width()/2, t),\n"
            "                ha='center', va='bottom' if t >= 0 else 'top')\n"
            "ax.set_ylabel('HAC t of long-short'); ax.set_title('Control: no false positive at edge=0; lights up when planted')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for e, m, t in res: print(f'edge={e:.2f}: long-short {m:+.2f}%/yr  HAC t={t:.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted premium the control sits at "
            f"**t = {R['syn'][0][2]:.2f}** (no false positive — the harness doesn't conjure an edge); "
            f"only a **large** +6%/yr planted premium reaches **t = {R['syn'][1][2]:.2f}**. So the "
            f"machinery is honest, and the real-tape **t = {R['test_ls'][1]:.2f}** reads as exactly "
            "what an *absent-or-tiny* edge looks like through a 21-year keyhole — not a bug, just a "
            "weak tape."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — long-short **+{R['test_ls'][0]:.2f}%/yr** at HAC "
            f"**t = {R['test_ls'][1]:.2f}** (n = {R['test_ls'][2]}); significant *only* at the "
            f"barely-sorted halves ({R['robust'][0][3]:.2f}), fragile to concentration. Published "
            "support + a positive-but-insignificant, specification-fragile estimate ⇒ `WEAK`, not "
            "`REAL`.\n"
            f"- **Tradability `MIRAGE`** — net of 10 bps turnover **+ 100 bps/yr short-borrow** the "
            f"spread is **+{R['ls_net'][0]:.2f}%/yr (t = {R['ls_net'][1]:.2f})** — zero, "
            "statistically. The only deployable object is a long-only growth tilt "
            f"(+{R['test_long_spy'][0]:.2f}%/yr vs SPY, t = {R['test_long_spy'][1]:.2f}) you can buy "
            "more cheaply as a style ETF.\n"
            f"- **Innovation premium? `MISATTRIBUTED`** — the split is a *real* axis "
            f"(**{R['ls_pctile']:.0f}th pct** of random splits) but it is the growth/tech-vs-value "
            "style axis (long = semis/software/pharma, short = banks/staples/retail), not a "
            "patent-specific alpha. The premium is attributed to *invention*; it is *style beta*."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the power curve\n\n"
            "The operational truth in one picture: how big would the *true* long-short premium have "
            "to be for a $T$-month study to detect it at $t = 2$, given the spread's realised "
            "volatility? Our 256-month tape can only certify edges far above the ~3.6%/yr we see."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = RACE['long_short'].dropna(); sd_m = ls.std(ddof=1)\n"
            "    obs_ann = RACE['test_ls']['mean_ann']*100\n"
            "else:\n"
            "    sd_m = 0.045; obs_ann = R['test_ls'][0]\n"
            "Ts = np.arange(36, 600)\n"
            "# min annual mean detectable at t=2 (iid approx): 2*sd_month*12/sqrt(T)\n"
            "min_det = 2.0 * sd_m * 12.0 / np.sqrt(Ts)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(Ts, min_det*100, c=AMBER, lw=2, label='annual premium needed for t=2')\n"
            "ax.axhline(obs_ann, c=GREEN, ls='--', label=f'observed spread ~{obs_ann:.1f}%/yr')\n"
            "ax.axvline(R['months'], c=GREY, ls=':', label=f\"our T={R['months']} months\")\n"
            "ax.set_xlabel('months of data T'); ax.set_ylabel('long-short premium (%/yr)')\n"
            "ax.set_title('Detection floor vs the real spread: under-powered at our T'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "need = 2.0*sd_m*12.0/np.sqrt(R['months'])*100\n"
            "print(f'at T={R[\"months\"]} you need ~{need:.1f}%/yr for t=2; observed ~{obs_ann:.1f}%/yr '\n"
            "      f'-> under-powered by ~{need/obs_ann:.1f}x')"
        ),
        md(
            "> 💡 In plain words: the amber curve is the **minimum detectable premium** at our "
            "sample length; the green line is what we actually see. The observed spread sits *below* "
            "the detection floor for 21 years of data — to certify a +3.6%/yr long-short at t = 2 "
            "you'd need several times the history (or a much cleaner, lower-vol spread). Even granting "
            "the literature a *real* small premium, this tape cannot prove it — and the random-split "
            "control says what little is there is **style**, not invention. There is no sort, no "
            "horizon, and no cost assumption on this field that turns audited R&D intensity into a "
            "distinct, significant, tradable innovation edge."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Input vs output.** Hirshleifer-Hsu-Li (2013): the surviving premium is **innovative "
            "*efficiency*** (patents/citations per R&D dollar), not gross intensity. Our proxy is "
            "input-only — plug in real patent-value data (Kogan-Papanikolaou-Seru-Stoffman) and "
            "re-test H₃.\n"
            "- **Style-strip.** Regress the long-short on an explicit growth-vs-value factor (or "
            "HML); the random-split control predicts little residual alpha survives — confirm it.\n"
            "- **Wider field.** 40 names is a transparent slice; run the same ranking on the full "
            "S&P 500 with sector-neutralisation. The power improves, but the *attribution* question "
            "(innovation vs style) is the one that decides the verdict.\n\n"
            "*The reproducible core is offline and deterministic; intensity is an explicit proxy "
            "(R&D / revenue), not issued patents. Methods and sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
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
