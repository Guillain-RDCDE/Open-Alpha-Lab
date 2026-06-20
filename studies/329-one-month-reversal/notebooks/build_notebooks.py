"""Generate the two narrative notebooks for Study 329 (One-Month-Reversal).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cache-first
monthly panel (``data.load_real``) if a cache is present and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any
reader. Every executed cell that shows synthetic output banners it as synthetic.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-15,
# panel trimmed to last complete month 2026-05-31, fingerprint fa033b6cedf6).
R = dict(
    n_months=435, n_stocks=398, n_quintile=80,
    loser_ann=24.73, winner_ann=18.18,
    spread_bps=54.6, spread_ann=6.55, spread_t=2.40, spread_hit=0.520,
    loser_xret_ann=5.51, loser_xret_t=3.10,
    spread_beta=0.313, spread_alpha_ann=0.54, loser_beta=1.290, winner_beta=0.977,
    # microstructure-safe skip=1
    skip1_bps=-4.9, skip1_ann=-0.58, skip1_t=-0.21,
    # sub-periods (skip=0 spread)
    sub_90_bps=134.9, sub_90_t=2.67, sub_90_n=154,
    sub_03_bps=23.6, sub_03_t=1.07, sub_03_n=144,
    sub_15_bps=-3.1, sub_15_t=-0.10, sub_15_n=137,
    # costs
    turnover=78.2, breakeven_bps=17.4,
    net5_bps=39.0, net5_t=1.72, net10_bps=23.3, net10_t=1.03, net20_bps=-8.0, net20_t=-0.35,
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from one_month_reversal import data, strategy as st

# Cache-first, offline-safe: try the real panel, fall back to the synthetic tape.
try:
    prices = data.load_real()
    res0 = st.quintile_returns(st.trailing_return(prices, skip=0), prices, q=0.20)
    res1 = st.quintile_returns(st.trailing_return(prices, skip=1), prices, q=0.20)
    spread = res0["spread"].dropna()
    HAVE_REAL = True
    print(f"REAL tape loaded: {prices.shape[0]} months x {prices.shape[1]} tickers "
          f"({prices.index[0].date()} -> {prices.index[-1].date()}, fp {data.fingerprint(prices)})")
except (FileNotFoundError, Exception) as exc:
    HAVE_REAL = False
    prices = res0 = res1 = spread = None
    print("No real cache (offline) -- cells will run the SYNTHETIC tape and quote frozen R numbers.")
    print("  reason:", type(exc).__name__)

print("HAVE_REAL:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# One-Month-Reversal -- do last month's losers beat last month's winners?\n"
            "### Jegadeesh (1990), tested honestly on the modern S&P 500 tape\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Bid--ask bounce: Confirmed](https://img.shields.io/badge/Bid--ask_bounce-Confirmed-8b949e?style=flat-square)\n\n"
            "In 1990, Narasimhan Jegadeesh showed that stock returns *reverse* month to "
            "month: the names that fell the most last month tend to bounce the most this "
            "month, and last month's winners give some back. Rank the market by last "
            "month's return, buy the losers, sell the winners -- and on paper you collect "
            "around 2% a month. It is one of the most-cited short-horizon anomalies in all "
            "of finance. This notebook asks whether you could actually keep any of it.\n\n"
            "> This is the plain-language layer. The t-stats, the bid-ask-bounce test, and "
            "the cost wall live in the companion "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** Reproducible research: every chart is drawn by the "
            "code beside it; real headline numbers are pinned in "
            "[docs/results.md](../docs/results.md). House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do last month's losers beat last month's winners? | **On paper, yes.** "
            f"+{R['spread_ann']:.1f}%/yr spread, HAC *t* = +{R['spread_t']:.2f} over 1990-2026 "
            "-- it clears the bar. |\n"
            "| Is it just hidden market risk (beta)? | **No** -- the long-short book is "
            f"nearly market-neutral (beta {R['spread_beta']:.2f}). This one is genuinely a "
            "cross-sectional anomaly, not disguised leverage. |\n"
            f"| Is it real information? | **Mostly no.** Leave a one-month gap and it "
            f"vanishes ( *t* = {R['skip1_t']:.2f} ) -- it is the bid-ask bounce, a "
            "measurement illusion, not a forecast. |\n"
            f"| Is it still there? | **No.** *t* = +{R['sub_90_t']:.2f} in the 1990s, "
            f"then {R['sub_15_t']:+.2f} in 2015-2026 -- dead for two decades. |\n"
            f"| Could you trade it? | **No.** ~{R['turnover']:.0f}% turnover a month; "
            f"break-even ~{R['breakeven_bps']:.0f} bps; costs eat it well before you do. |\n\n"
            "> The reversal was real and market-neutral -- which makes it the most "
            "*interesting* mirage on the desk. It just turns out to be the bid-ask spread "
            "measured in a mirror, and even that died around 2002."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *'Individual security returns exhibit strong negative serial correlation at "
            "the monthly horizon. A portfolio that buys prior-month losers and sells "
            "prior-month winners earns large, reliable profits.'*\n"
            "> -- Jegadeesh (1990), Journal of Finance\n\n"
            "The logic: the crowd over-pushes a stock's one-month move (panic-selling a "
            "bad month, chasing a good one), and the move partly unwinds the next month. "
            "Whoever takes the other side -- buying the panic, selling the euphoria -- is "
            "paid for providing that liquidity. The key word is **one month**: this is the "
            "*opposite* of 12-month momentum, and the opposite of the 3-5 year reversal in "
            "[Study 196](../../196-long-term-reversal/)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "A reliable, **market-neutral** 6%/yr would be a genuinely valuable building "
            "block: it does not need the market to go up, it diversifies a long book, and "
            "it is mechanical. That is the holy grail of quant equity -- a pure "
            "cross-sectional edge with no beta attached.\n\n"
            "If instead the 'edge' is just the bid-ask bounce -- prices ping-ponging "
            "between the bid and the ask, manufacturing fake reversals out of thin air -- "
            "then it is untradable by construction: the very thing that creates the signal "
            "(the spread) is also the cost of harvesting it."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Four clean tests, announced before we run them:\n\n"
            "1. **The spread.** Rank every stock by last month's return, buy the bottom "
            "20% (losers), sell the top 20% (winners). Is the spread reliably positive?\n"
            "2. **The one-month-gap test.** Re-run it but skip a month between forming the "
            "signal and holding the trade. A *real* forecast survives a small gap; a "
            "**bid-ask-bounce illusion** does not (the bounce only links *adjacent* prints).\n"
            "3. **Did it survive publication?** Split 1990-2002 / 2003-2014 / 2015-2026.\n"
            "4. **Could you trade it?** Measure the turnover and charge realistic costs.\n\n"
            f"Data: ~{R['n_stocks']} current S&P 500 names, monthly closes 1990-2026 "
            f"({R['n_months']} rebalance months). **Survivorship-biased**: the universe is "
            "today's S&P 500 projected backwards, so every firm survived to 2026. The true "
            "loser leg (full of firms that delisted) would do worse -- all results are upper bounds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 - The teardown\n\n"
            "**First: the loser-winner race.** Does buying last month's losers and selling "
            "last month's winners actually make money?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize(spread)\n"
            "    spread_ann, spread_t = s['mean']*12*100, s['tstat']\n"
            "    eq = (1 + spread).cumprod()\n"
            "    tag = 'REAL TAPE'\n"
            "else:\n"
            f"    spread_ann, spread_t = {R['spread_ann']}, {R['spread_t']}\n"
            "    # synthetic illustration of a positive but noisy monthly spread\n"
            "    rng = np.random.default_rng(329)\n"
            "    idx = pd.period_range('1990-01', periods=435, freq='M').to_timestamp(how='end')\n"
            f"    r_sp = rng.normal({R['spread_bps']/1e4:.5f}, 0.025, 435)\n"
            "    eq = pd.Series((1 + r_sp).cumprod(), index=idx)\n"
            "    tag = 'SYNTHETIC TAPE (offline) -- headline numbers from docs/results.md'\n\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(eq, c=GREEN, lw=2, label='Loser - Winner spread (dollar-neutral)')\n"
            "ax.axhline(1, c=GREY, ls='--', lw=1)\n"
            "ax.set_ylabel('Cumulative growth of $1 (long-short)'); ax.legend()\n"
            "ax.set_title(f'One-month reversal spread -- {tag}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'[{tag}]  spread = {spread_ann:+.1f}%/yr   HAC t = {spread_t:+.2f}')"
        ),
        md(
            f"On the full sample the spread is **+{R['spread_ann']:.1f}%/yr** with HAC "
            f"*t* = +{R['spread_t']:.2f} -- it clears the bar of 2.0. So far, so good for "
            "Jegadeesh. And (we check in the quant notebook) it is nearly market-neutral, "
            "so this is *not* the beta-in-disguise trap that sank the long-term reversal. "
            "It looks like the real thing.\n\n"
            "**Now the killer test: does it survive a one-month gap?**"
        ),
        code(
            "labels = ['skip=0\\n(last month, adjacent)', 'skip=1\\n(one-month gap)']\n"
            "if HAVE_REAL:\n"
            "    t0 = st.summarize(res0['spread'])['tstat']\n"
            "    t1 = st.summarize(res1['spread'])['tstat']\n"
            "    tag = 'REAL TAPE'\n"
            "else:\n"
            f"    t0, t1 = {R['spread_t']}, {R['skip1_t']}\n"
            "    tag = 'SYNTHETIC/frozen'\n\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.5))\n"
            "cols = [GREEN if t > 2 else AMBER if t > 1 else RED for t in (t0, t1)]\n"
            "bars = ax.bar(labels, [t0, t1], color=cols)\n"
            "ax.axhline(2, ls='--', c=GREEN, lw=1, label='inference bar (t=2)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b, t in zip(bars, (t0, t1)):\n"
            "    ax.annotate(f't={t:+.2f}', (b.get_x()+b.get_width()/2, t + (0.1 if t>=0 else -0.25)), ha='center')\n"
            "ax.set_ylabel('Spread HAC t-stat'); ax.legend()\n"
            "ax.set_title(f'Leave one month between signal and trade -> the edge evaporates ({tag})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'[{tag}]  skip=0 t={t0:+.2f}   skip=1 t={t1:+.2f}')"
        ),
        md(
            f"There it is. With the standard recipe the spread is *t* = +{R['spread_t']:.2f}; "
            f"insert a single month between forming the signal and holding the trade and it "
            f"collapses to *t* = {R['skip1_t']:.2f} -- **statistically nothing**. A genuine "
            "one-month forecast would barely notice a one-month gap. What *does* die when "
            "you break the adjacency is the **bid-ask bounce**: a stock that happened to "
            "close on the bid prints a low return and 'reverts' upward next print purely "
            "because the next trade lands on the ask. No information -- just the spread, "
            "measured in a mirror.\n\n"
            "**And has it survived publication?**"
        ),
        code(
            "periods = ['1990-2002', '2003-2014', '2015-2026']\n"
            "if HAVE_REAL:\n"
            "    rec = [(st.summarize(res0['spread'][a:b])['mean']*1e4,\n"
            "            st.summarize(res0['spread'][a:b])['tstat'])\n"
            "           for a, b in [('1990','2002'),('2003','2014'),('2015','2026')]]\n"
            "    bps, ts = zip(*rec); tag = 'REAL TAPE'\n"
            "else:\n"
            f"    bps = ({R['sub_90_bps']}, {R['sub_03_bps']}, {R['sub_15_bps']})\n"
            f"    ts  = ({R['sub_90_t']}, {R['sub_03_t']}, {R['sub_15_t']})\n"
            "    tag = 'frozen'\n\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "cols = [GREEN if t > 2 else AMBER if t > 1 else RED for t in ts]\n"
            "bars = ax.bar(periods, bps, color=cols)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b, t in zip(bars, ts):\n"
            "    ax.annotate(f't={t:+.2f}', (b.get_x()+b.get_width()/2, b.get_height()+ (3 if b.get_height()>=0 else -10)), ha='center')\n"
            "ax.set_ylabel('Spread (bps/month)')\n"
            "ax.set_title(f'A 1990s phenomenon: dead since ~2002 ({tag})')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"The whole full-sample t-stat is carried by the 1990s (+{R['sub_90_bps']:.0f} "
            f"bps/mo, *t* = +{R['sub_90_t']:.2f}). By 2003-2014 it had shrunk to *t* = "
            f"+{R['sub_03_t']:.2f}, and in the last decade it is slightly **negative** "
            f"(*t* = {R['sub_15_t']:.2f}). Decimalisation (penny spreads from 2001) and "
            "stat-arb desks competed the liquidity rent away -- exactly the post-publication "
            "fade you would predict."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal -- MIXED.** Real on the raw full sample (*t* = +{R['spread_t']:.2f}) "
            f"and genuinely market-neutral (beta {R['spread_beta']:.2f}) -- but **None** once "
            f"you defuse the bid-ask bounce (skip=1 *t* = {R['skip1_t']:.2f}) or look past "
            f"2002 (2015-2026 *t* = {R['sub_15_t']:.2f}). The certified part is the 1990s "
            "liquidity rent; the clean, modern part is zero.\n"
            f"- **Tradability -- MIRAGE.** ~{R['turnover']:.0f}% monthly turnover; "
            f"break-even ~{R['breakeven_bps']:.0f} bps; the signal evaporates a month out."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "Even pretending the gross edge were reliably there, a one-month signal means "
            "you replace almost the whole book every month:"
        ),
        code(
            "one_way = [0, 5, 10, 20]\n"
            "if HAVE_REAL:\n"
            "    nets = [(c, st.summarize(st.net_spread(res0, c))['mean']*1e4) for c in one_way]\n"
            "    to = res0['loser_turn'].mean()*100; be = st.break_even_cost(res0)\n"
            "    tag = 'REAL TAPE'\n"
            "else:\n"
            f"    nets = [(0, {R['spread_bps']}), (5, {R['net5_bps']}), (10, {R['net10_bps']}), (20, {R['net20_bps']})]\n"
            f"    to, be = {R['turnover']}, {R['breakeven_bps']}\n"
            "    tag = 'frozen'\n\n"
            "cs, nm = zip(*nets)\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(cs, nm, 'o-', c=AMBER, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(cs, nm, 0, where=[n < 0 for n in nm], color=RED, alpha=.12)\n"
            "ax.axvline(be, ls='--', c=GREY); ax.annotate(f'break-even ~{be:.0f}bps', (be, max(nm)*0.5))\n"
            "ax.set_xlabel('One-way transaction cost (bps)'); ax.set_ylabel('Net spread (bps/month)')\n"
            "ax.set_title(f'At ~{to:.0f}% monthly turnover, ~15-20 bps wipes out the spread ({tag})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'[{tag}] turnover {to:.0f}%/mo  break-even {be:.0f}bps one-way')"
        ),
        md(
            f"With ~{R['turnover']:.0f}% one-way turnover, the break-even cost is only "
            f"~{R['breakeven_bps']:.0f} bps. At an optimistic 10 bps the net spread is "
            f"+{R['net10_bps']:.0f} bps/mo (*t* = {R['net10_t']:.2f}, below 1); at a "
            f"realistic all-in 20 bps it is **{R['net20_bps']:+.0f} bps/mo** -- a loss. And "
            "remember: the gross edge we are eroding is bid-ask bounce, which means the "
            "real-world spread you would pay *is* the signal you are chasing. You cannot "
            "buy a dollar with a dollar."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The horizon spectrum.** Daily fade-against-peers is "
            "[Study 33 -- Slingshot](../../33-slingshot/); the 3-5 year reversal is "
            "[Study 196 -- Long-Term-Reversal](../../196-long-term-reversal/). Together "
            "they map the full return-autocorrelation curve.\n"
            "- **Liquidity-conditioned.** Avramov-Chordia-Goyal (2006) show the reversal "
            "lives in illiquid names (where you cannot trade it). A fork: sort by "
            "liquidity and confirm the edge concentrates exactly where costs are worst.\n"
            "- **The synthetic control.** The quant notebook plants a known reversal in a "
            "fake market and shows the engine recovers it monotonically -- so the real-tape "
            "'mostly nothing' is a fact about the market, not a broken detector.\n\n"
            "*Think a liquidity-screened or weekly version survives the one-month-gap test "
            "and clears the bar net of costs? Fork it and show a *t* > 2 after the gap. "
            "That's the bar.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 -- FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# One-Month-Reversal -- a quantitative teardown\n"
            "### Monthly S&P 500 quintile spread * the bid-ask-bounce test * beta-neutrality * decay * HAC inference\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Bid--ask bounce: Confirmed](https://img.shields.io/badge/Bid--ask_bounce-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "Same seven beats, every claim carrying its standard error. We test whether the "
            "Jegadeesh (1990) one-month loser-minus-winner spread survives (a) a HAC t-test, "
            "(b) a one-month-gap (skip=1) microstructure test, (c) a beta decomposition, "
            "(d) a sub-period split, and (e) realistic transaction costs -- and we plant a "
            "known reversal in a synthetic market to confirm the engine is a faithful detector.\n\n"
            "> **Not investment advice.** Real data: shared S&P 500 monthly closes, "
            "1990-01-31 to 2026-05-31 (as-of 2026-06-15, fingerprint `fa033b6cedf6`). "
            "Methods in [`docs/references.md`](../docs/references.md); numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship bias**: the universe is the current S&P 500 projected "
            "backwards; all real-tape results are upper bounds."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | Full-sample spread HAC *t* = **+{R['spread_t']:.2f}** "
            f"and beta-neutral (spread beta **{R['spread_beta']:.2f}**); but skip=1 "
            f"*t* = **{R['skip1_t']:.2f}** and 2015-2026 *t* = **{R['sub_15_t']:.2f}**. |\n"
            f"| **Tradability** | `MIRAGE` | ~{R['turnover']:.0f}% one-way monthly turnover; "
            f"break-even **{R['breakeven_bps']:.0f} bps**; net *t* < 1 by 10 bps. |\n"
            "| **Bid-ask bounce** | `CONFIRMED` | A one-month gap removes the entire effect "
            "(Roll 1984; Lo-MacKinlay 1990). |\n\n"
            "> The reversal is real, market-neutral, and 1990s-only -- a microstructure "
            "rent, not a forecast, and competed away two decades ago."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            r"Let $r_{i,t}$ be stock $i$'s simple return in month $t$. The Jegadeesh (1990) "
            "claim is three joint hypotheses:\n\n"
            r"- **H1 (spread).** $\mathbb{E}[r^{\text{loser}}_{t} - r^{\text{winner}}_{t}] > 0$, "
            r"where losers/winners are the bottom/top quintile by $r_{i,t-1}$."
            "\n"
            r"- **H2 (information, not bounce).** The spread survives a one-month gap: "
            r"forming on $r_{i,t-2}$ and holding in $t$ still pays. A genuine forecast is "
            "robust to a small lag; a bid-ask-bounce artefact is not.\n"
            r"- **H3 (persistence).** The effect is stable across sub-periods."
            "\n\n"
            f"We **accept H1** on the full sample (+{R['spread_ann']:.1f}%/yr, "
            f"*t* = +{R['spread_t']:.2f}) and add that it is beta-neutral -- but we "
            f"**reject H2** (skip=1 *t* = {R['skip1_t']:.2f}) and **reject H3** "
            f"(2015-2026 *t* = {R['sub_15_t']:.2f}). The certified component is the 1990s "
            "microstructure rent."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "A reliable market-neutral 6%/yr is the quant-equity grail. The *interesting* "
            "failure here is not absence -- the spread clears the bar and is not beta -- but "
            "**mechanism**: the edge is the bid-ask bounce (a pure accounting artefact of "
            "transaction prices alternating between bid and ask), so the cost of harvesting "
            "it is identically the source of it. That is the cleanest possible example of a "
            "signal that is statistically real and economically untradable."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - The protocol\n\n"
            "- **Universe**: current S&P 500 with >=60 monthly closes (survivorship-biased); "
            f"~{R['n_stocks']} names/month, ~{R['n_quintile']} per quintile.\n"
            "- **Signal**: last month's simple return, lagged one month "
            "(`strategy.trailing_return`, `skip=0`). The microstructure test re-runs with "
            "`skip=1` (a one-month gap).\n"
            "- **Portfolios**: equal-weight bottom/top quintiles; dollar-neutral L-S spread.\n"
            "- **Inference**: Newey-West HAC *t* on the monthly spread; beta via OLS on the "
            "equal-weight market; sub-period split (1990-2002 / 2003-2014 / 2015-2026); "
            "random-portfolio null of the same quintile size.\n"
            "- **Costs**: one-way x NAV across both legs, round-trip, from the realised "
            "loser-leg turnover; break-even one-way cost reported.\n"
            "- **Positive control**: synthetic panel with a tunable reversal coefficient "
            "(`data.synthetic_panel`), to confirm the engine recovers a planted signal and "
            "reads ~0 on the null.\n\n"
            "**Mirage trigger (declared in advance):** if the spread does not survive the "
            "skip=1 gap *and* the post-2014 sub-period, we call the certified signal a "
            "1990s microstructure rent, not a live edge."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - The spread and its market-neutrality\n\n"
            "The dollar-neutral loser-minus-winner spread, its HAC t, and the beta of each "
            "leg (the test that separates this from Study 196's beta-laden long-term reversal)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize(spread)\n"
            "    beta_s, alpha_frac = st.beta_alpha(res0['spread'], res0['market'])\n"
            "    alpha_s = alpha_frac * 100  # annualised fraction -> %/yr\n"
            "    beta_l, _ = st.beta_alpha(res0['loser'], res0['market'])\n"
            "    beta_w, _ = st.beta_alpha(res0['winner'], res0['market'])\n"
            "    spread_ann, spread_t = s['mean']*12*100, s['tstat']\n"
            "    tag = 'REAL TAPE'\n"
            "else:\n"
            f"    spread_ann, spread_t = {R['spread_ann']}, {R['spread_t']}\n"
            f"    beta_s, alpha_s = {R['spread_beta']}, {R['spread_alpha_ann']}\n"
            f"    beta_l, beta_w = {R['loser_beta']}, {R['winner_beta']}\n"
            "    tag = 'frozen'\n\n"
            "print(f'[{tag}] spread {spread_ann:+.2f}%/yr  HAC t={spread_t:+.2f}')\n"
            "print(f'[{tag}] spread (L-S) beta={beta_s:.3f}  alpha={alpha_s:+.2f}%/yr')\n"
            "print(f'[{tag}] loser beta={beta_l:.3f}  winner beta={beta_w:.3f}')\n\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "ax.bar(['Loser leg', 'Winner leg', 'L-S spread'], [beta_l, beta_w, beta_s],\n"
            "       color=[GREEN, RED, GREY])\n"
            "ax.axhline(1, ls='--', c=GREY, lw=1); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Beta vs equal-weight market')\n"
            "ax.set_title(f'The long-short spread is near beta-neutral -> not disguised leverage ({tag})')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> Spread *t* = +{R['spread_t']:.2f} clears the inference bar. Crucially the "
            f"L-S beta is **{R['spread_beta']:.2f}** with alpha +{R['spread_alpha_ann']:.2f}%/yr "
            "-- so, unlike the long-term reversal (Study 196, where the loser leg was a "
            "beta-1.3 bet), this is a *genuine* cross-sectional anomaly. H1 holds. The "
            "question is whether it is information."
        ),
        md(
            "### 4b - The bid-ask-bounce test (skip=0 vs skip=1)\n\n"
            "The decisive experiment: insert one month between formation and holding. "
            "Information survives a small lag; the Roll (1984) bounce, which links only "
            "*adjacent* prints, does not."
        ),
        code(
            "if HAVE_REAL:\n"
            "    t0 = st.summarize(res0['spread'])['tstat']\n"
            "    m0 = st.summarize(res0['spread'])['mean']*1e4\n"
            "    t1 = st.summarize(res1['spread'])['tstat']\n"
            "    m1 = st.summarize(res1['spread'])['mean']*1e4\n"
            "    tag = 'REAL TAPE'\n"
            "else:\n"
            f"    t0, m0 = {R['spread_t']}, {R['spread_bps']}\n"
            f"    t1, m1 = {R['skip1_t']}, {R['skip1_bps']}\n"
            "    tag = 'frozen'\n\n"
            "tbl = pd.DataFrame({'spread (bps/mo)': [m0, m1], 'HAC t': [t0, t1]},\n"
            "                   index=['skip=0 (adjacent)', 'skip=1 (one-month gap)'])\n"
            "print(f'[{tag}]'); print(tbl.round(2).to_string())\n\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "cols = [GREEN if t > 2 else RED for t in (t0, t1)]\n"
            "bars = ax.bar(['skip=0', 'skip=1'], [t0, t1], color=cols)\n"
            "ax.axhline(2, ls='--', c=GREEN, lw=1); ax.axhline(0, c='k', lw=1)\n"
            "for b, t in zip(bars, (t0, t1)):\n"
            "    ax.annotate(f't={t:+.2f}', (b.get_x()+b.get_width()/2, t + (0.1 if t>=0 else -0.25)), ha='center')\n"
            "ax.set_ylabel('Spread HAC t-stat')\n"
            "ax.set_title(f'One-month gap -> t collapses to ~0: the bounce, not a forecast ({tag})')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> **The verdict on H2.** skip=0 *t* = +{R['spread_t']:.2f} -> skip=1 "
            f"*t* = {R['skip1_t']:.2f}. The entire effect lives in the adjacency of the "
            "formation close and the holding month, which is the signature of bid-ask "
            "bounce (Lo-MacKinlay 1990 decompose contrarian profits and attribute the own-"
            "autocovariance share precisely to non-synchronous trading and the spread). "
            "Reject H2: the clean, information-bearing component is statistically zero."
        ),
        md(
            "### 4c - Sub-period decay and the random-portfolio null"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rec = [(lab, st.summarize(res0['spread'][a:b])['mean']*1e4,\n"
            "            st.summarize(res0['spread'][a:b])['tstat'],\n"
            "            st.summarize(res0['spread'][a:b])['n'])\n"
            "           for lab,a,b in [('1990-2002','1990','2002'),('2003-2014','2003','2014'),('2015-2026','2015','2026')]]\n"
            "    rand = st.random_portfolio_returns(st.trailing_return(prices), prices, n_draws=80, seed=329)\n"
            "    rand_bps = rand.mean()*1e4\n"
            "    tag = 'REAL TAPE'\n"
            "else:\n"
            f"    rec = [('1990-2002',{R['sub_90_bps']},{R['sub_90_t']},{R['sub_90_n']}),\n"
            f"           ('2003-2014',{R['sub_03_bps']},{R['sub_03_t']},{R['sub_03_n']}),\n"
            f"           ('2015-2026',{R['sub_15_bps']},{R['sub_15_t']},{R['sub_15_n']})]\n"
            "    rand_bps = -1.4; tag = 'frozen'\n\n"
            "labs, bps, ts, ns = zip(*rec)\n"
            "print(f'[{tag}]'); print(pd.DataFrame({'bps/mo': bps, 't': ts, 'n': ns}, index=labs).round(2).to_string())\n"
            "print(f'[{tag}] random-portfolio null mean excess = {rand_bps:+.1f} bps (centred ~0: {abs(rand_bps)<5})')\n\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "cols = [GREEN if t > 2 else AMBER if t > 1 else RED for t in ts]\n"
            "bars = ax.bar(labs, bps, color=cols)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b, t in zip(bars, ts):\n"
            "    ax.annotate(f't={t:+.2f}', (b.get_x()+b.get_width()/2, b.get_height()+(3 if b.get_height()>=0 else -10)), ha='center')\n"
            "ax.set_ylabel('Spread (bps/month)')\n"
            "ax.set_title(f'Post-publication decay: significant only in the 1990s ({tag})')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> The spread is significant only in 1990-2002 (*t* = +{R['sub_90_t']:.2f}), "
            f"fades to +{R['sub_03_t']:.2f} in 2003-2014, and is {R['sub_15_t']:+.2f} in "
            "2015-2026 -- the Khandani-Lo (2007) / McLean-Pontiff (2016) competition-and-"
            "decimalisation fade. The random-portfolio null sits at zero, confirming the "
            "loser excess (where it exists) is signal, not concentration. Reject H3."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `MIXED`** -- H1 accepted (full-sample spread *t* = +{R['spread_t']:.2f}, "
            f"beta-neutral {R['spread_beta']:.2f}); H2 rejected (skip=1 *t* = {R['skip1_t']:.2f}); "
            f"H3 rejected (2015-2026 *t* = {R['sub_15_t']:.2f}). Real as a 1990s microstructure "
            "rent; None as a clean, modern, information-bearing signal. Survivorship-biased "
            "upper bound.\n"
            f"- **Tradability `MIRAGE`** -- ~{R['turnover']:.0f}% one-way monthly turnover; "
            f"break-even ~{R['breakeven_bps']:.0f} bps; net spread *t* = {R['net10_t']:.2f} at "
            f"10 bps, {R['net20_bps']:+.0f} bps/mo (loss) at 20 bps. The harvesting cost is the "
            "signal source.\n"
            "- **Bid-ask bounce `CONFIRMED`** -- the one-month gap removes the entire effect."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md("## 6 - Could you trade it? -- turnover and cost sweep"),
        code(
            "one_way = [0, 5, 10, 20, 30]\n"
            "if HAVE_REAL:\n"
            "    rows = [(c, st.summarize(st.net_spread(res0, c))['mean']*1e4,\n"
            "             st.summarize(st.net_spread(res0, c))['tstat']) for c in one_way]\n"
            "    to = res0['loser_turn'].mean()*100; be = st.break_even_cost(res0)\n"
            "    tag = 'REAL TAPE'\n"
            "else:\n"
            f"    base = {R['spread_bps']}; to = {R['turnover']}; be = {R['breakeven_bps']}\n"
            "    rows = [(c, base - 2*(to/100)*2*c, None) for c in one_way]\n"
            "    tag = 'frozen'\n\n"
            "cs, nm, nt = zip(*rows)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.plot(cs, nm, 'o-', c=AMBER, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(cs, nm, 0, where=[n < 0 for n in nm], color=RED, alpha=.12)\n"
            "ax.axvline(be, ls='--', c=GREY); ax.annotate(f'break-even ~{be:.0f}bps', (be, max(nm)*0.4))\n"
            "ax.set_xlabel('One-way transaction cost (bps)'); ax.set_ylabel('Net spread (bps/month)')\n"
            "ax.set_title(f'Cost wall: ~{to:.0f}% turnover -> break-even ~{be:.0f}bps ({tag})')\n"
            "plt.tight_layout(); plt.show()\n"
            "for c, n, t in rows:\n"
            "    print(f'  {c:2d} bps one-way -> net {n:+.1f} bps/mo' + (f'  t={t:+.2f}' if t is not None else ''))"
        ),
        md(
            f"> Break-even ~{R['breakeven_bps']:.0f} bps one-way. Net *t* drops below 1 by "
            "10 bps and the spread turns negative by 20 bps -- and that is *before* the "
            "deeper problem: the gross edge is bid-ask bounce, so the realistic spread cost "
            "is not an external tax but the very thing being measured. The strategy is "
            "self-cancelling."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the engine work, or does it always print MIXED? Plant a one-month reversal "
            "coefficient directly in a synthetic market and sweep it."
        ),
        code(
            "revs = [-0.04, -0.02, 0.0, 0.02, 0.04, 0.06]\n"
            "edge = []\n"
            "for rev in revs:\n"
            "    p, _, _ = data.synthetic_panel(n_firms=150, n_months=250, reversal=rev, seed=329)\n"
            "    rs = st.quintile_returns(st.trailing_return(p, skip=0), p, q=0.20)\n"
            "    s = st.summarize(rs['spread'].dropna())\n"
            "    edge.append((rev, s['mean']*12*100, s['tstat']))\n\n"
            "rv, me, te = zip(*edge)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.plot(rv, me, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('Planted one-month reversal coefficient'); ax.set_ylabel('Spread (%/yr)')\n"
            "ax.set_title('SYNTHETIC control: engine is monotone in the planted signal, ~0 at the null')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('[SYNTHETIC]')\n"
            "for r, m, t in edge:\n"
            "    print(f'  rev={r:+.2f}: spread={m:+.2f}%/yr  t={t:+.2f}')"
        ),
        md(
            "The spread is monotone in the planted reversal and crosses zero at the null "
            "(rev = 0): the engine is a faithful detector. So the real-tape result -- a "
            "full-sample signal that is *entirely* bid-ask bounce and dead since 2002 -- is "
            "a fact about the **market**, not the method. Forks worth trying: a "
            "liquidity-screened cross-section (Avramov-Chordia-Goyal 2006), a weekly Lehmann "
            "(1990) version with the same gap test, or a microstructure-clean reversal built "
            "on *quote midpoints* rather than trade prices."
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
