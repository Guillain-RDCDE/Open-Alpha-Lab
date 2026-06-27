"""Generate the two narrative notebooks for Study 525 (R-And-D-Intensity).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached EDGAR fundamentals +
yfinance prices under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
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


# Frozen real-tape headline numbers — mirror of docs/results.md (EDGAR R&D/ME signal + SPY,
# 40-name field, 2005-02 -> 2026-05, 256 return months / 220 holding months, fingerprint 8d048dab164a).
R = dict(
    start="2005-02", end="2026-05", months=220, ret_months=256, years=21.3, field=40,
    rd_names=24, fingerprint="8d048dab164a", turnover=1.6,
    long_members=["ADBE", "AMGN", "BMY", "CSCO", "DE", "GILD", "IBM", "INTC", "LLY", "MRK",
                  "ORCL", "PFE", "QCOM"],
    short_members=["AXP", "DUK", "GS", "HD", "JPM", "KO", "LOW", "MCD", "T", "USB", "VZ", "WFC",
                   "WMT"],
    # books: (cagr%, sharpe, maxdd%, mean_ann%)
    long=(16.90, 1.06, -39.6, 16.99),
    short=(11.77, 0.80, -38.7, 12.38),
    long_short=(3.92, 0.37, -26.3, 4.61),
    spy=(11.11, 0.78, -50.8, 11.70),
    # signal tests: (mean_ann%, hac_t, n)
    test_ls=(4.61, 1.69, 220),
    test_long_spy=(4.56, 2.52, 220),
    test_short_spy=(-0.05, -0.02, 220),
    # placebo label-shuffle null
    null_mean=0.41, null_sd=1.84, placebo_pctile=99.8, placebo_p=0.005,
    # CLS denominator contrast: R&D/ME vs R&D/sales (mean_ann%, t)
    me_ls=(4.61, 1.69), sales_ls=(3.99, 1.40),
    # robustness: (frac, ls_mean_ann%, hac_t)
    robust=[(0.50, 3.97, 1.70), (1 / 3, 4.61, 1.69), (0.25, 5.66, 1.87), (0.20, 6.86, 2.24)],
    # costs+borrow: gross and net long-short (mean_ann%, t)
    ls_gross=(4.61, 1.69), ls_net=(3.59, 1.32),
    # synthetic: (edge, ls_mean_ann%, hac_t)
    syn=[(0.0, -0.43, -0.16), (0.06, 5.57, 2.01)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![ME_vs_sales: Confirmed](https://img.shields.io/badge/ME_vs_sales-Confirmed-8b949e?style=flat-square)\n\n"
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

from r_and_d_intensity import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    REAL = data.load_real(allow_survivorship_bias=True)
    RETS, SPY = REAL["returns"], REAL["spy"]
    SIGS = data.build_signals(REAL, report_lag=1)
    # keep the live placebo light so the notebook executes quickly; results.md uses 400 draws
    RACE = st.race(REAL, SIGS, frac=1/3, cost_bps=10.0, borrow_bps=100.0, n_shuffles=120)
else:
    REAL = RETS = SPY = SIGS = RACE = None
print("real EDGAR+yfinance cache present:", HAVE_REAL,
      "| field names:", (0 if RETS is None else RETS.shape[1]))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Buy the R&D-cheap, short the rest — is there a hidden *R&D premium*? 🔬\n"
            "### Do the firms whose research budget is big *relative to their price tag* quietly beat the market?\n\n"
            + BADGES +
            "A famous finance paper (Chan, Lakonishok & Sougiannis, 2001) says yes — but with a "
            "twist. It's not the firms that *spend* the most on R&D that win; it's the firms whose "
            "R&D is large **relative to how cheaply the market prices them** (R&D ÷ market value). "
            "The idea: those firms are quietly building intangible moats the market is too slow to "
            "pay up for. So a portfolio **long the high-R&D-to-market-cap names, short the low** "
            "should harvest an *R&D premium*.\n\n"
            "This notebook builds exactly that portfolio on audited data and asks the three questions "
            "that matter: **is the extra return actually there**, **could you trade it**, and **does "
            "scaling R&D by *price* really beat scaling it by *sales*** (the version "
            "[study 400](../../400-patent-intensity/) found weak)?\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the label-shuffle placebo and the "
            "borrow-cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** We pull R&D, revenue and share counts straight from "
            "companies' **SEC filings**, and market cap from price × shares. Firms with no R&D line "
            "(banks, oil, soda) genuinely spend ~0, so they land on the short side. Every chart is "
            "drawn by the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do high-R&D/price names beat low-R&D/price names? | **Yes, a bit.** The top tertile "
            f"out-earns the bottom by **+{R['test_ls'][0]:.1f}%/yr**, and it's **not luck of the "
            f"draw** — a shuffled version of the signal pays ~0 (placebo p = {R['placebo_p']:.3f}). |\n"
            "| Is that a certifiable edge? | **Not quite.** Over 21 years the gap's *t*-stat is "
            f"**{R['test_ls'][1]:.2f}** — below the **2** you need. It only crosses 2 at the very "
            "tightest sort, and the whole gap is the *long* side; the short side is just the market. |\n"
            "| Could you trade it? | **Not really.** Shorting the low-R&D names costs **borrow**, "
            f"which drags the gap to **+{R['ls_net'][0]:.1f}%/yr at t = {R['ls_net'][1]:.2f}** — "
            "indistinguishable from zero. |\n"
            "| Does scaling by *price* beat scaling by *sales*? | **Yes — exactly as the paper "
            f"says.** R&D/market-cap (+{R['me_ls'][0]:.1f}%/yr) edges out R&D/sales "
            f"(+{R['sales_ls'][0]:.1f}%/yr). The ranking holds — but both are below the bar. |\n\n"
            "> R&D/market-cap sorts stocks onto a *real* axis — but that axis is mostly **cheap "
            "value/growth**, the gap isn't statistically certifiable, and borrow eats what's left."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The firms whose R&D is big relative to their market value are quietly compounding "
            "intangible moats the market under-prices. Hold them, short the rest, collect the R&D "
            "premium.\"*\n\n"
            "It isn't a crank claim — it's a 2001 *Journal of Finance* paper. The key word is "
            "**relative to market value**: a firm spending \\$1bn on R&D while priced at \\$10bn "
            "(R&D/ME = 10%) is the bet, not a giant priced at \\$3tn. That's really a *cheapness* "
            "signal dressed as an innovation one — which is the first thing to keep an eye on."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a clean R&D/ME premium existed, you could rank the market once a month, tilt toward "
            "the R&D-cheap names, and beat the index with a story clients love. But two traps hide "
            "inside it. **One:** scaling by *price* mechanically tilts you toward **cheap stocks** — "
            "so 'long R&D/ME' is partly just 'long value,' a tilt you can buy for a few basis points. "
            "**Two:** a long/short has to *short* the low-R&D names, and shorting isn't free — you pay "
            "to borrow them. A premium that's really a value tilt, and that borrow costs eat, isn't an "
            "edge — it's a relabelled, leaky version of something you already own."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We build the portfolio on **audited** numbers. For a fixed **{R['field']}-name** "
            "large-cap field — chosen by *sector* so the mix spans R&D-heavy and R&D-light, **not** by "
            "who won — each month we compute **R&D ÷ (price × shares)** using only data the filings "
            "had *already reported* (a one-year lag) and the price *known that month*. Then:\n\n"
            "1. **Rank & sort.** Go *long* the third with the highest R&D/market-cap, *short* the "
            "third with the lowest — and enter the trade *next* month (no peeking, no same-bar fill).\n"
            "2. **Measure the gap.** Over 21 years, does long-minus-short actually pay — and is the gap "
            "big enough, relative to its wobble, to not be luck?\n"
            "3. **Shuffle the labels.** Scramble *which name* gets *which R&D/ME score* and rebuild "
            "the bet, hundreds of times. If the real bet beats almost all the scrambles, the signal is "
            "doing real work; if not, it was just any-old-split of a mixed field."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, who's on each side?** Here's the long (high-R&D/ME) and short (low-R&D/ME) "
            "tertile — and notice it reads like a sector map, not a stock-picker's secret list."
        ),
        code(
            "if HAVE_REAL:\n"
            "    last = SIGS['rd_me'].dropna(how='all').index[-2]\n"
            "    lo, sh = st.tertile_members(SIGS['rd_me'].loc[last], frac=1/3)\n"
            "    x = SIGS['rd_me'].loc[last]\n"
            "else:\n"
            "    lo, sh = R['long_members'], R['short_members']; x = None\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "if x is not None:\n"
            "    xl = x[lo].sort_values(); xs = x[sh].sort_values()\n"
            "    ax.barh(list(xl.index), xl.values*100, color=GREEN, label='LONG  high R&D/ME')\n"
            "    ax.barh(list(xs.index), -xs.values*100-0.3, color=GREY, label='SHORT low R&D/ME')\n"
            "    ax.set_xlabel('R&D / market-cap  (%)   — long side right, short side left')\n"
            "else:\n"
            "    ax.barh(lo, [1]*len(lo), color=GREEN, label='LONG  high R&D/ME (tech/pharma)')\n"
            "    ax.barh(sh, [-1]*len(sh), color=GREY, label='SHORT low R&D/ME (banks/staples)')\n"
            "    ax.set_xlabel('(install the cache to see real R&D/ME)')\n"
            "ax.axvline(0, c='k', lw=.8); ax.set_title('The R&D/ME split is a sector map: tech/pharma vs banks/staples')\n"
            "ax.legend(loc='lower right'); plt.tight_layout(); plt.show()\n"
            "print('LONG :', sorted(lo))\nprint('SHORT:', sorted(sh))"
        ),
        md(
            "Long: chips (INTC, TXN, QCOM, CSCO), software (ADBE, ORCL, IBM), pharma/biotech (LLY, "
            "MRK, PFE, BMY, AMGN, GILD), plus DE. Short: banks (JPM, WFC, GS, USB, AXP), staples & "
            "retail (KO, WMT, HD, LOW, MCD), telco/utility (T, VZ, DUK). That's not an innovation "
            "list — **that's the growth-pharma-vs-value-financials lineup.** Hold that thought."
        ),
        md(
            "**Does the gap pay?** The long (high-R&D/ME), the short leg (low-R&D/ME), and the market "
            "— growth of \\$1 over the holding window."
        ),
        code(
            "if HAVE_REAL:\n"
            "    L, S, MKT = RACE['long'], RACE['short'], RACE['spy'].reindex(RACE['long'].index).dropna()\n"
            "    L = L.reindex(MKT.index); S = S.reindex(MKT.index)\n"
            "    eqs = {'Long (high R&D/ME)': (1+L).cumprod(), 'Short leg (low R&D/ME)': (1+S).cumprod(),\n"
            "           'SPY (market)': (1+MKT).cumprod()}\n"
            "    idx = MKT.index\n"
            "else:\n"
            "    idx = np.arange(R['months'])\n"
            "    g = lambda m: (1+np.full(R['months'], (1+m/100)**(1/12)-1)).cumprod()\n"
            "    eqs = {'Long (high R&D/ME)': g(R['long'][3]), 'Short leg (low R&D/ME)': g(R['short'][3]),\n"
            "           'SPY (market)': g(R['spy'][3])}\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "for (name, eq), c in zip(eqs.items(), [GREEN, GREY, RED]):\n"
            "    ax.plot(idx, np.asarray(eq), c=c, lw=2, label=name)\n"
            "ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "ax.set_title(f\"High-R&D/ME beats low-R&D/ME by ~{R['test_ls'][0]:.1f}%/yr — but the short leg IS the market\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"long CAGR {R['long'][0]:.1f}%  short CAGR {R['short'][0]:.1f}%  \"\n"
            "      f\"spread {R['long_short'][3]:+.1f}%/yr  SPY {R['spy'][0]:.1f}%\")"
        ),
        md(
            f"The high-R&D/ME side finishes well ahead — **+{R['test_ls'][0]:.1f}%/yr** over the low "
            "side. But look closely: the short leg tracks the market almost exactly. So the 'spread' "
            "is really *the long leg beating the market*, with the short leg adding nothing. The "
            f"question is whether **+{R['test_ls'][0]:.1f}%/yr** over 21 wobbly years is a *signal* or "
            "just luck — the next chart settles it."
        ),
        md(
            "**Is +4.6%/yr more than luck?** We scramble *which name gets which R&D/ME score* 120 "
            "times and rebuild the bet. If the real bet beats almost all the scrambles, R&D/ME is "
            "carrying real information."
        ),
        code(
            "if HAVE_REAL:\n"
            "    null = RACE['placebo_null']*100; obs = R['test_ls'][0]; pct = RACE['placebo_pctile']\n"
            "else:\n"
            "    rng = np.random.default_rng(525); null = rng.normal(R['null_mean'], R['null_sd'], 400)\n"
            "    obs = R['test_ls'][0]; pct = R['placebo_pctile']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(null, bins=40, color=GREY, alpha=.85, label='label-shuffled bets (signal scrambled)')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'the real R&D/ME bet ({obs:+.1f}%/yr)')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('long-short spread (%/yr)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'R&D/ME carries real information (it beats {pct:.0f}% of scrambles)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'scrambled signal pays ~0%/yr; the real signal pays {obs:+.1f}%/yr -> NOT a generic split')\n"
            "print(f\"...but its OWN t-stat is {R['test_ls'][1]:.2f} (below 2): real axis, not certifiable return\")"
        ),
        md(
            "Two things at once, both honest. The real R&D/ME bet lands far in the tail of the "
            f"scrambles (**{R['placebo_pctile']:.0f}th percentile, p = {R['placebo_p']:.3f}**) — so "
            "R&D/ME is **not** a coin flip; it really picks a persistent axis. **But** that axis is "
            "the *cheap-value/growth* axis (look at the names), and its own return gap, judged over 21 "
            f"years, has a *t*-stat of just **{R['test_ls'][1]:.2f}** — below the **2** bar. A real "
            "*style* tilt, not a certifiable *innovation* edge."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** High-R&D/ME beats low by **+{R['test_ls'][0]:.1f}%/yr** and the "
            f"signal survives the label-shuffle (p = {R['placebo_p']:.3f}) — but the gap fails the "
            f"significance bar (HAC *t* = {R['test_ls'][1]:.2f}) and only clears it at the tightest "
            "sort. Real-ish axis, not a certifiable return.\n"
            f"- **Tradability — Mirage.** Shorting the low-R&D names costs borrow, which drops the "
            f"gap to **+{R['ls_net'][0]:.1f}%/yr at t = {R['ls_net'][1]:.2f}** — a net edge "
            "indistinguishable from zero.\n"
            f"- **R&D/ME vs R&D/sales — Confirmed.** Scaling by *price* "
            f"(+{R['me_ls'][0]:.1f}%/yr) does beat scaling by *sales* (+{R['sales_ls'][0]:.1f}%/yr), "
            "exactly as the paper predicts — but both are below the bar, so it's a ranking, not a "
            "rescue."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the borrow bites\n\n"
            "Forget significance for a second. Even taking the +4.6%/yr at face value, a long/short "
            "must *short* the low-R&D names — and you pay a borrow fee to do it. Here's the gap before "
            "and after a modest 1%/yr borrow on the short leg."
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
            "ax.set_title('Borrow turns an insignificant +4.6%/yr into a still-insignificant +3.6%/yr')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.1f}%/yr (t={gt:.2f}) -> net {n:+.1f}%/yr (t={nt:.2f}): both below t=2')"
        ),
        md(
            "Turnover is small (the market-cap denominator only nudges the ranking month to month), so "
            "trading cost isn't the killer. **Borrow is.** And the long-*only* version — just hold the "
            f"high-R&D/ME names — beats SPY by **+{R['test_long_spy'][0]:.1f}%/yr "
            f"(t = {R['test_long_spy'][1]:.2f})**, which *is* significant — but it's a value/growth "
            "tilt you can buy as a cheap style ETF, not a secret innovation edge."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **It's mostly *value*.** Because the denominator is price, R&D/ME tilts toward cheap "
            "stocks — regress the long-short on a value factor and ask whether any *residual* R&D "
            "alpha survives. (Our scramble test already hints the signal is real-but-style.)\n"
            "- **The sales cousin.** [Study 400 — Patent-Intensity](../../400-patent-intensity/) "
            "sorts on R&D ÷ *sales* and finds it weaker — this study confirms the paper's claim that "
            "the *price* denominator is the better one.\n"
            "- **Output, not input.** Hirshleifer-Hsu-Li (2013): the premium that most survives is "
            "patents *per R&D dollar* (efficiency), not gross R&D. Swap in real patent data and "
            "re-test.\n\n"
            "*Think the R&D/ME premium is real and distinct from value? Show the long-short clearing "
            "t = 2 at a tradable tertile, net of borrow, with the value beta stripped out — then we'll "
            "talk.*"
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
            "# R&D-Intensity — a quantitative teardown 🔬\n"
            "### Monthly R&D/market-cap ranking with reporting + execution lags · long-short & "
            "long-minus-SPY HAC *t* · a label-shuffle placebo · the R&D/ME-vs-R&D/sales contrast · "
            "fraction robustness · costs + short-borrow · a synthetic planted-premium / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We test "
            "the Chan-Lakonishok-Sougiannis (2001) premium: is the long-high / short-low "
            "**R&D-to-market-cap** spread **statistically real** (HAC *t* ≥ 2 **and** placebo "
            "survival), is it **tradable** net of monthly turnover + short-borrow, and does **R&D/ME "
            "beat R&D/sales** as CLS claim? The decisive objects are the **HAC t** of the spread, the "
            "**label-shuffle placebo** (is it the signal or any split?), and the **denominator "
            "contrast**.\n\n"
            "> ⚠️ **Data note.** R&D, revenue and shares from SEC EDGAR `companyfacts`; market cap = "
            "price × shares (price from yfinance). R&D-light names (banks/staples/energy) floored to "
            "~0. One reporting lag (FY Y-1) + one execution lag (enter next month). Survivorship is "
            "named on the Signal axis (current-membership basket; bias largely *common to both legs*). "
            "Offline core + synthetic control are deterministic. Methods in "
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
            f"**t = {R['test_ls'][1]:.2f}** (n = {R['test_ls'][2]}); survives placebo "
            f"(p = {R['placebo_p']:.3f}) but clears t = 2 *only* at the quintile "
            f"({R['robust'][3][2]:.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | net of 10 bps turnover **+ 100 bps/yr short-borrow** the "
            f"spread is **+{R['ls_net'][0]:.2f}%/yr at t = {R['ls_net'][1]:.2f}** — zero, "
            "statistically. Long-only beats SPY by "
            f"+{R['test_long_spy'][0]:.2f}%/yr (t = {R['test_long_spy'][1]:.2f}, a value tilt). |\n"
            f"| **R&D/ME vs R&D/sales** | `CONFIRMED` | R&D/market-cap "
            f"(**+{R['me_ls'][0]:.2f}%/yr, t = {R['me_ls'][1]:.2f}**) out-ranks R&D/sales "
            f"(**+{R['sales_ls'][0]:.2f}%/yr, t = {R['sales_ls'][1]:.2f}**) — the CLS ordering — but "
            "both below the bar. |\n\n"
            "> 💡 In plain words: R&D/ME sorts stocks onto a genuine axis (placebo p = 0.005) and the "
            "price denominator does beat the sales one — but that axis is *value*, the spread doesn't "
            "clear significance, and borrow finishes whatever is left."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $x_{i,t}$ be firm $i$'s R&D/ME at month $t$: $x_{i,t} = "
            "\\text{R\\&D}_{i,Y-1} / (P_{i,t}\\,S_{i,Y-1})$ — last reported R&D over current market "
            "cap (a 1-year reporting lag on fundamentals, contemporaneous price). Form, at month $t$, "
            "the long-high / short-low book and hold it through $t+1$ (one execution lag):\n\n"
            "$$\\text{LS}_{t+1} = \\frac1k\\!\\!\\sum_{i\\in \\text{top-}k} r_{i,t+1} \\;-\\; "
            "\\frac1k\\!\\!\\sum_{i\\in \\text{bot-}k} r_{i,t+1}.$$\n\n"
            "- **H₁ (the premium is real).** $\\mathbb{E}[\\text{LS}] > 0$ with HAC $t \\ge 2$ **and** "
            "the spread sits in the tail of a label-shuffle null.\n"
            "- **H₂ (it's deployable).** The spread survives one-way costs **and the borrow you pay to "
            "be short**.\n"
            "- **H₃ (CLS denominator).** R&D/ME (scaled by market value) beats R&D/sales (scaled by "
            "revenue).\n\n"
            "We find **H₁ split** (placebo passes, $p = 0.005$; but $t = 1.69 < 2$ — `WEAK`), **H₂ "
            "rejected** (net of borrow $t = 1.32$; the deployable object is a long-only value tilt), "
            "**H₃ supported** (R&D/ME +4.61%/yr > R&D/sales +3.99%/yr, the CLS ordering — though both "
            "sub-2). The label is true where it's uninformative (R&D-cheap value has out-earned) and "
            "unproven where it would pay (a *certifiable, tradable* R&D alpha)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The teardown is one decomposition: a long-short mean against zero, judged by its **HAC "
            "standard error**, and against the **sampling distribution under shuffled labels**.\n\n"
            "$$t_{\\text{HAC}} = \\frac{\\overline{\\text{LS}}}{\\widehat{\\text{se}}_{\\text{NW}}"
            "(\\overline{\\text{LS}})}, \\qquad "
            "p_{\\text{placebo}} = \\Pr_{\\pi}\\!\\big[|\\overline{\\text{LS}}_{\\pi}| \\ge "
            "|\\overline{\\text{LS}}|\\big],$$\n\n"
            "where $\\pi$ permutes the signal's name-labels. A small **placebo p** says R&D/ME loads "
            "on a real axis (not noise); a low **t** says that axis's *return* gap is inside its own "
            "error bar. Both are true here. The honest read needs both: the placebo kills 'it's just "
            "luck,' the *t* kills 'it's a certified edge,' and the names kill 'it's about *invention*' "
            "(it's *value*)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Signal panel.** Monthly R&D/ME for a fixed {R['field']}-name field "
            f"({R['rd_names']} with an R&D series), R&D from 10-K full-year facts (FY Y-1), shares "
            "from the same filing, price as of the month. R&D-light names floored to 0.\n"
            "- **Books.** Each month, equal-weight top-tertile (long) / bottom-tertile (short); enter "
            "*next* month (1-month execution lag), monthly rebalance.\n"
            "- **Null #1 (HAC t).** Newey-West t of the monthly long-short and long-minus-SPY means "
            "(`REAL` needs $t \\ge 2$).\n"
            "- **Null #2 (placebo).** Permute the signal's name-labels; rebuild the long-short; the "
            "real spread must sit in the tail.\n"
            "- **H₃ contrast.** Same machinery on R&D/sales (study-400 signal) vs R&D/ME.\n"
            "- **Robustness.** Fraction sweep halves→quintiles.\n"
            "- **Costs.** 10 bps one-way turnover **+ a 100 bps/yr borrow on the short leg**.\n"
            "- **Positive control.** A deterministic panel with a *planted* annual premium `edge`: "
            "recover a large edge **and** NOT manufacture significance at `edge = 0`."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The point estimate — positive, and it's all the long leg\n\n"
            "The long-short and long-minus-SPY spreads with their HAC *t*. The long-short misses 2; "
            "long-minus-SPY clears it; the short leg is *the market*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [('Long - short\\n(R&D/ME)', RACE['test_ls']),\n"
            "            ('Long - SPY', RACE['test_long_vs_spy']),\n"
            "            ('Short - SPY', RACE['test_short_vs_spy'])]\n"
            "    labels = [r[0] for r in rows]; means = [r[1]['mean_ann']*100 for r in rows]\n"
            "    ts = [r[1]['tstat'] for r in rows]\n"
            "else:\n"
            "    labels = ['Long - short\\n(R&D/ME)', 'Long - SPY', 'Short - SPY']\n"
            "    means = [R['test_ls'][0], R['test_long_spy'][0], R['test_short_spy'][0]]\n"
            "    ts = [R['test_ls'][1], R['test_long_spy'][1], R['test_short_spy'][1]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))\n"
            "cols = [GREEN if m > 0 else GREY for m in means]\n"
            "a1.bar(labels, means, color=cols, width=.6); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_ylabel('mean spread (%/yr)'); a1.set_title('Spreads: the edge is the long leg')\n"
            "for i, m in enumerate(means): a1.annotate(f'{m:+.1f}', (i, m), ha='center', va='bottom')\n"
            "a2.bar(labels, [abs(t) for t in ts], color=[GREEN if abs(t)>=2 else AMBER for t in ts], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i, t in enumerate(ts): a2.annotate(f't={t:.2f}', (i, abs(t)), ha='center', va='bottom')\n"
            "a2.set_ylabel('|HAC t|'); a2.set_ylim(0, 3.0); a2.set_title('Only long-minus-SPY clears t = 2'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('long-short', f\"{R['test_ls'][0]:+.2f}%/yr t={R['test_ls'][1]:.2f}\",\n"
            "      '| long-SPY', f\"{R['test_long_spy'][0]:+.2f}%/yr t={R['test_long_spy'][1]:.2f}\",\n"
            "      '| short-SPY', f\"{R['test_short_spy'][0]:+.2f}%/yr t={R['test_short_spy'][1]:.2f}\")"
        ),
        md(
            f"> 💡 In plain words: the R&D/ME spread is **+{R['test_ls'][0]:.2f}%/yr** at "
            f"**t = {R['test_ls'][1]:.2f}** — positive but *not* significant. And it's one-sided: "
            f"long-minus-SPY is +{R['test_long_spy'][0]:.2f}%/yr (t = {R['test_long_spy'][1]:.2f}, "
            f"significant) while short-minus-SPY is {R['test_short_spy'][0]:.2f}%/yr "
            f"(t = {R['test_short_spy'][1]:.2f}) — the short leg simply *is* the market, so the "
            "'premium' is the long leg's value/growth out-performance, not a clean two-sided edge."
        ),
        md(
            "### 4b · Is it the signal, or any split of a mixed field? — the label-shuffle placebo\n\n"
            "Permute *which name gets which R&D/ME score* (120 draws here; results.md uses 400) and "
            "rebuild the long-short. Where the real spread lands tells us whether the signal carries "
            "information — and the answer is a careful *yes*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    null = RACE['placebo_null']*100; obs = RACE['test_ls']['mean_ann']*100\n"
            "    pct = RACE['placebo_pctile']; pv = RACE['placebo_p']\n"
            "else:\n"
            "    rng = np.random.default_rng(525); null = rng.normal(R['null_mean'], R['null_sd'], 400)\n"
            "    obs = R['test_ls'][0]; pct = R['placebo_pctile']; pv = R['placebo_p']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(null, bins=40, color=GREY, alpha=.85, label='label-shuffled long-shorts')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'R&D/ME spread {obs:+.1f}%/yr')\n"
            "ax.axvline(np.mean(null), c=RED, ls='--', label=f'shuffle mean {np.mean(null):+.1f}%/yr')\n"
            "ax.set_xlabel('long-short spread (%/yr)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'R&D/ME at the {pct:.0f}th pct of shuffles (placebo p={pv:.3f}) — a REAL axis')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'shuffled signal ~0%/yr (no info); R&D/ME {obs:+.1f}%/yr at {pct:.0f}th pctile, p={pv:.3f}')\n"
            "print('=> signal loads on a persistent axis -- but the names show that axis is cheap value/growth')"
        ),
        md(
            f"> 💡 In plain words: a *shuffled* signal pays ≈0%/yr, so the R&D/ME spread's "
            f"+{R['test_ls'][0]:.1f}%/yr (the **{R['placebo_pctile']:.0f}th percentile, "
            f"p = {R['placebo_p']:.3f}**) is **not** a generic artefact — R&D/ME really loads on a "
            "persistent cross-sectional axis. The catch is *which* axis: scaling by price tilts toward "
            "cheap stocks, and the long names are cheap-intangibles tech/pharma. That is the **value** "
            "axis. The placebo rescues R&D/ME from 'noise' and convicts it of 'value beta' in the same "
            "breath. Combined with 4a's $t = 1.69$: a real *style* tilt, an unproven *return*."
        ),
        md(
            "### 4c · The CLS denominator contrast — does scaling by price beat scaling by sales?\n\n"
            "The paper's signature claim: R&D ÷ **market value** predicts where R&D ÷ **sales** does "
            "not. Same machinery, both denominators."
        ),
        code(
            "if HAVE_REAL:\n"
            "    me = RACE['test_ls']; sa = RACE['test_sales_ls']\n"
            "    me_m, me_t = me['mean_ann']*100, me['tstat']; sa_m, sa_t = sa['mean_ann']*100, sa['tstat']\n"
            "else:\n"
            "    me_m, me_t = R['me_ls']; sa_m, sa_t = R['sales_ls']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.2))\n"
            "a1.bar(['R&D / market-cap\\n(CLS)', 'R&D / sales\\n(study 400)'], [me_m, sa_m],\n"
            "       color=[GREEN, GREY], width=.55)\n"
            "for i, m in enumerate([me_m, sa_m]): a1.annotate(f'{m:+.2f}%/yr', (i, m), ha='center', va='bottom')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('long-short (%/yr)'); a1.set_title('R&D/ME edges out R&D/sales')\n"
            "a2.bar(['R&D / market-cap', 'R&D / sales'], [me_t, sa_t], color=AMBER, width=.55)\n"
            "a2.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i, t in enumerate([me_t, sa_t]): a2.annotate(f't={t:.2f}', (i, t), ha='center', va='bottom')\n"
            "a2.set_ylabel('HAC t'); a2.set_ylim(0, 2.4); a2.set_title('...but both below the bar'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'R&D/ME {me_m:+.2f}%/yr (t={me_t:.2f}) vs R&D/sales {sa_m:+.2f}%/yr (t={sa_t:.2f})')"
        ),
        md(
            f"> 💡 In plain words: R&D/ME (**+{R['me_ls'][0]:.2f}%/yr, t = {R['me_ls'][1]:.2f}**) does "
            f"beat R&D/sales (**+{R['sales_ls'][0]:.2f}%/yr, t = {R['sales_ls'][1]:.2f}**) — both in "
            "spread and in *t* — exactly the Chan-Lakonishok-Sougiannis ordering: the *price* "
            "denominator (cheapness) is the better signal. **H₃ supported.** But the win is thin "
            "(≈0.6%/yr) and both fail the bar, so it confirms CLS's *ranking* of the two signals "
            "without rescuing either to significance."
        ),
        md(
            "### 4d · Robustness — and 4e costs + short-borrow\n\n"
            "Fraction sweep: unlike R&D/sales (study 400, which weakened on concentration), R&D/ME "
            "*strengthens* — reaching $t = 2.24$ at the tightest quintile. Then borrow."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob = []\n"
            "    for frac in (0.50, 1/3, 0.25, 0.20):\n"
            "        b = st.signal_books(SIGS['rd_me'], RETS, frac=frac)\n"
            "        t = st.hac_tstat(b['long_short']); rob.append((frac, t['mean_ann']*100, t['tstat']))\n"
            "else:\n"
            "    rob = R['robust']\n"
            "labels = ['halves', 'tertiles', 'quartiles', 'quintiles']\n"
            "tt = [r[2] for r in rob]; ms = [r[1] for r in rob]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "bars = a1.bar(labels, tt, color=[GREEN if t >= 2 else AMBER for t in tt], width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for b, m, t in zip(bars, ms, tt):\n"
            "    a1.annotate(f'{m:+.1f}%/yr\\nt={t:.2f}', (b.get_x()+b.get_width()/2, t), ha='center', va='bottom')\n"
            "a1.set_ylabel('HAC t of long-short'); a1.set_ylim(0, 2.6)\n"
            "a1.set_title('Strengthens on concentration; clears t=2 only at quintile'); a1.legend()\n"
            "g, gt = R['ls_gross']; n, nt = R['ls_net']\n"
            "bars2 = a2.bar(['gross', 'net\\n(+100bps borrow)'], [g, n], color=[GREEN, RED], width=.5)\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('long-short (%/yr)'); a2.set_ylim(0, g*1.4)\n"
            "for b, v, t in zip(bars2, [g, n], [gt, nt]):\n"
            "    a2.annotate(f'{v:+.2f}%/yr\\nt={t:.2f}', (b.get_x()+b.get_width()/2, v), ha='center', va='bottom')\n"
            "a2.set_title('Borrow erodes the thin gross spread')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (frac, mean%/yr, t):', [(round(r[0],2), round(r[1],2), round(r[2],2)) for r in rob])\n"
            "print(f'costs: gross {g:+.2f}%/yr (t={gt:.2f}) -> net {n:+.2f}%/yr (t={nt:.2f})')"
        ),
        md(
            f"> 💡 In plain words: the spread *grows* as you concentrate — halves "
            f"{R['robust'][0][2]:.2f}, tertile {R['robust'][1][2]:.2f}, quintile "
            f"**{R['robust'][3][2]:.2f}** — a *better* signature than the sales signal, but it clears "
            "2 only at the 8-name-per-leg extreme where idiosyncratic risk is highest, not at the "
            f"tradable tertile. And borrow drops gross **+{R['ls_gross'][0]:.2f}%/yr "
            f"(t = {R['ls_gross'][1]:.2f})** → net **+{R['ls_net'][0]:.2f}%/yr "
            f"(t = {R['ls_net'][1]:.2f})**: no costed, shortable edge to allocate to."
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "A deterministic panel (40 names, 18 years) where the first half are persistently "
            "high-R&D/ME, the market beta is common to both legs, and `edge` plants the *true* annual "
            "long-high-minus-short-low premium. With `edge = 0` the long-short must stay insignificant;"
            " a large planted edge must light up."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.06):\n"
            "    s2, r2, b2, truth = data.synthetic_panel(edge=edge)\n"
            "    bk = st.signal_books(s2, r2); t = st.hac_tstat(bk['long_short'])\n"
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
            f"**t = {R['syn'][0][2]:.2f}** (no false positive); only a **large** +6%/yr planted "
            f"premium reaches **t = {R['syn'][1][2]:.2f}**. So the machinery is honest, and the "
            f"real-tape **t = {R['test_ls'][1]:.2f}** reads as exactly what a *real-but-small* edge "
            "looks like through a 21-year keyhole — not a bug, just a weak tape."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — long-short **+{R['test_ls'][0]:.2f}%/yr** at HAC "
            f"**t = {R['test_ls'][1]:.2f}** (n = {R['test_ls'][2]}); survives the label-shuffle "
            f"placebo (p = {R['placebo_p']:.3f}, {R['placebo_pctile']:.0f}th pct) but clears 2 *only* "
            f"at the quintile ({R['robust'][3][2]:.2f}). Published + placebo-real but insignificant on "
            "the tradable tertile ⇒ `WEAK`, not `REAL`.\n"
            f"- **Tradability `MIRAGE`** — net of 10 bps turnover **+ 100 bps/yr short-borrow** the "
            f"spread is **+{R['ls_net'][0]:.2f}%/yr (t = {R['ls_net'][1]:.2f})** — zero, "
            "statistically. The only deployable object is a long-only value tilt "
            f"(+{R['test_long_spy'][0]:.2f}%/yr vs SPY, t = {R['test_long_spy'][1]:.2f}) you can buy "
            "cheaper as a style ETF.\n"
            f"- **R&D/ME vs R&D/sales `CONFIRMED`** — scaling by market value "
            f"(+{R['me_ls'][0]:.2f}%/yr, t = {R['me_ls'][1]:.2f}) out-ranks scaling by sales "
            f"(+{R['sales_ls'][0]:.2f}%/yr, t = {R['sales_ls'][1]:.2f}), the Chan-Lakonishok-"
            "Sougiannis ordering — but the win is thin and both fail the bar."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the power curve\n\n"
            "How big would the *true* long-short premium have to be for a $T$-month study to detect it "
            "at $t = 2$, given the spread's realised volatility? Our tape can only certify edges well "
            "above the ~4.6%/yr we see at the tradable tertile."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = RACE['long_short'].dropna(); sd_m = ls.std(ddof=1)\n"
            "    obs_ann = RACE['test_ls']['mean_ann']*100\n"
            "else:\n"
            "    sd_m = 0.04; obs_ann = R['test_ls'][0]\n"
            "Ts = np.arange(36, 600)\n"
            "min_det = 2.0 * sd_m * 12.0 / np.sqrt(Ts)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(Ts, min_det*100, c=AMBER, lw=2, label='annual premium needed for t=2')\n"
            "ax.axhline(obs_ann, c=GREEN, ls='--', label=f'observed spread ~{obs_ann:.1f}%/yr')\n"
            "ax.axvline(R['months'], c=GREY, ls=':', label=f\"our T={R['months']} months\")\n"
            "ax.set_xlabel('months of data T'); ax.set_ylabel('long-short premium (%/yr)')\n"
            "ax.set_title('Detection floor vs the real spread: under-powered at the tradable tertile'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "need = 2.0*sd_m*12.0/np.sqrt(R['months'])*100\n"
            "print(f'at T={R[\"months\"]} you need ~{need:.1f}%/yr for t=2 at the tertile; observed ~{obs_ann:.1f}%/yr')"
        ),
        md(
            "> 💡 In plain words: the amber curve is the **minimum detectable premium** at our sample "
            "length; the green line is what we see at the tradable tertile. The observed spread sits "
            "*below* the detection floor — to certify it at t = 2 you'd need more history (or to "
            "concentrate into the noisy quintile, which we did, where it just clears). Even granting "
            "the literature a *real* small premium, this tape can't prove it at a deployable sort — "
            "and the placebo says the axis is **value**, not pure invention. Net of borrow there is no "
            "sort, horizon, or cost assumption on this field that turns R&D/ME into a distinct, "
            "significant, tradable edge."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Strip the value.** R&D/ME tilts toward cheap stocks by construction — regress the "
            "long-short on an explicit value (HML) factor and ask if any *residual* R&D alpha "
            "survives. The placebo predicts little does.\n"
            "- **The sales cousin.** [Study 400 — Patent-Intensity](../../400-patent-intensity/) is "
            "the R&D/sales leg of the same contrast — this study confirms CLS that the *price* "
            "denominator is the stronger of the two.\n"
            "- **Output, not input.** Hirshleifer-Hsu-Li (2013): the surviving premium is patents per "
            "R&D dollar (efficiency); plug in real patent-value data and re-test.\n\n"
            "*The reproducible core is offline and deterministic; R&D/ME is the audited CLS signal "
            "(R&D ÷ market value). Methods and sources: "
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
