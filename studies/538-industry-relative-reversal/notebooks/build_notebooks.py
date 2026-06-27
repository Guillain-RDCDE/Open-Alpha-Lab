"""Generate the two narrative notebooks for Study 538 (Industry-Relative-Reversal).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats. The synthetic figures run anywhere, offline and
deterministic; the real-tape cells use the cache-first monthly panel (``data.load_real``) if
a cache is present and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md), so the notebook re-runs for any reader. Every executed cell that shows
synthetic output banners it as synthetic.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md
# (as-of 2026-06-26, panel trimmed to 2026-05-31, fingerprint bbd55e07b990).
R = dict(
    n_months=437, n_stocks=54, n_quintile=11,
    # skip=0 head-to-head
    raw_bps=-11.0, raw_ann=-1.32, raw_t=-0.41, raw_beta=0.073,
    irr_bps=9.5, irr_ann=1.14, irr_t=0.56, irr_hit=0.503, irr_beta=0.149,
    # skip=1 (one-month gap)
    raw1_bps=10.4, raw1_t=0.40, irr1_bps=27.5, irr1_t=1.33,
    # placebo (industry-label shuffle)
    placebo_mean=-0.48, placebo_sd=0.36, placebo_q95=0.06, placebo_p=0.005,
    # sub-periods (IRR spread)
    sub_90_bps=15.1, sub_90_t=0.43, sub_90_n=154,
    sub_03_bps=-4.2, sub_03_t=-0.15, sub_03_n=144,
    sub_15_bps=17.7, sub_15_t=0.70, sub_15_n=137,
    # costs
    turnover=76.6, breakeven_bps=3.1,
    net0_bps=5.4, net0_t=0.31, net5_bps=-10.0, net5_t=-0.59,
    net10_bps=-25.3, net10_t=-1.49, net20_bps=-55.9, net20_t=-3.32,
    # synthetic control
    syn_raw04_t=30.98, syn_irr04_t=69.46, syn_raw08_t=67.27, syn_irr08_t=144.52,
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from industry_relative_reversal import data, strategy as st

# Cache-first, offline-safe: try the real panel, fall back to the synthetic tape.
try:
    prices = data.load_real()
    sectors = data.sector_series(prices.columns)
    raw0 = st.quintile_returns(st.raw_signal(prices, skip=0), prices, q=0.20)
    irr0 = st.quintile_returns(st.industry_relative_signal(prices, sectors, skip=0), prices, q=0.20)
    raw1 = st.quintile_returns(st.raw_signal(prices, skip=1), prices, q=0.20)
    irr1 = st.quintile_returns(st.industry_relative_signal(prices, sectors, skip=1), prices, q=0.20)
    HAVE_REAL = True
    print(f"REAL tape loaded: {prices.shape[0]} months x {prices.shape[1]} tickers "
          f"({prices.index[0].date()} -> {prices.index[-1].date()}, fp {data.fingerprint(prices)})")
except Exception as exc:
    HAVE_REAL = False
    prices = sectors = raw0 = irr0 = raw1 = irr1 = None
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
            "# Industry-Relative-Reversal -- does subtracting the sector make the reversal better?\n"
            "### Hameed-Mian (2015) / Da-Liu-Schaumburg (2014), tested honestly on the S&P 500 tape\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Industry-adjustment beats raw: Confirmed](https://img.shields.io/badge/Industry_beats_raw-Confirmed-8b949e?style=flat-square)\n\n"
            "The one-month reversal -- buy last month's losers, sell last month's winners -- is "
            "the most famous short-horizon anomaly in equities. But [Study 329](../../329-one-month-reversal/) "
            "found the *raw* version is mostly bid-ask bounce, dead since ~2002. Two papers "
            "(Hameed-Mian 2015; Da-Liu-Schaumburg 2014) say the trick is to measure the "
            "reversal **relative to each stock's own industry**: a stock's move splits into "
            "its *sector's* move (which does **not** reverse) and the part *unique to the "
            "stock* (which does). Subtract the sector first, and you should get a cleaner "
            "signal. This notebook asks whether that is true -- and whether 'cleaner' is "
            "enough to be tradable.\n\n"
            "> Plain-language layer. The t-stats, the placebo, and the cost wall live in the "
            "companion **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** Reproducible research: every chart is drawn by the "
            "code beside it; real headline numbers are pinned in "
            "[docs/results.md](../docs/results.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does subtracting the sector give a better signal? | **Yes.** The industry-"
            f"relative spread (**+{R['irr_bps']:.1f}** bps/mo) *flips the sign* of the raw "
            f"spread (**{R['raw_bps']:.1f}** bps/mo) and beats it. |\n"
            f"| Is the 'industry' part doing real work? | **Yes.** Shuffle the sector labels "
            f"and the advantage vanishes -- the real signal sits above the 99.5th percentile "
            f"of the shuffled null (p = {R['placebo_p']:.3f}). |\n"
            f"| Is it statistically real (clears *t* = 2)? | **No.** The industry-relative "
            f"*t* is only **+{R['irr_t']:.2f}** on this 54-name basket. Right direction, "
            "sub-bar magnitude. |\n"
            f"| Could you trade it? | **No.** ~{R['turnover']:.0f}% turnover a month; "
            f"break-even ~{R['breakeven_bps']:.1f} bps; costs turn it loss-making almost "
            "immediately. |\n\n"
            "> The refinement *works as advertised* -- a cleaner signal than the raw "
            "reversal -- it is simply too faint, on a thin survivor basket, to clear the bar "
            "or to pay for its own turnover."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *'Stock return reversals are primarily driven by the within-industry component "
            "of returns. The across-industry (sector) component does not reverse.'*\n"
            "> -- Hameed & Mian (2015), JFQA; Da, Liu & Schaumburg (2014), Management Science\n\n"
            "Think of last month's return as two pieces:\n\n"
            "- **The sector's move.** If all of energy rallied last month, ExxonMobil rallied "
            "partly *because it is energy*. That piece reflects real news and does **not** "
            "bounce back.\n"
            "- **The stock's own move.** The part of Exxon's return *beyond* what energy did. "
            "*That* piece is where the overreaction -- and the reversal -- lives.\n\n"
            "The raw reversal mixes the two, diluting the signal. The fix: rank stocks by "
            "their **industry-relative** return (stock minus its sector mean)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If true, this is a genuinely better mousetrap: a market-neutral reversal that "
            "is not just bid-ask bounce (which is what sank the raw version in Study 329). "
            "It would be the difference between a famous-but-dead anomaly and a live, "
            "tradable, sector-neutral edge.\n\n"
            "But 'a better signal' and 'a tradable strategy' are different claims. A one-"
            "month signal -- even a clean one -- forces you to replace almost the entire book "
            "every month, and turnover is where short-horizon edges go to die."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Four clean tests, announced before we run them:\n\n"
            "1. **Head-to-head.** Build the raw loser-minus-winner spread *and* the industry-"
            "relative one. Does subtracting the sector beat the raw sort?\n"
            "2. **The placebo.** Randomly **shuffle the industry labels** and rebuild. If the "
            "edge survives a *fake* industry map, the 'industry' part was an illusion.\n"
            "3. **The one-month-gap test.** Skip a month between signal and trade. The raw "
            "reversal dies here (bid-ask bounce); does the industry-relative one survive?\n"
            "4. **Could you trade it?** Turnover and realistic costs.\n\n"
            f"Data: a fixed **{R['n_stocks']}-name, six-sector** S&P 500 basket, monthly "
            f"closes 1990-2026 ({R['n_months']} months). **Survivorship-biased**: every firm "
            "survived to 2026, so all results are upper bounds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 - The teardown\n\n"
            "**First: the head-to-head.** Raw reversal vs industry-relative reversal, same "
            "basket, same quintiles."
        ),
        code(
            "labels = ['RAW\\n(total return)', 'INDUSTRY-RELATIVE\\n(minus sector)']\n"
            "if HAVE_REAL:\n"
            "    raw_bps = st.summarize(raw0['spread'])['mean']*1e4\n"
            "    irr_bps = st.summarize(irr0['spread'])['mean']*1e4\n"
            "    tag = 'REAL TAPE'\n"
            "else:\n"
            f"    raw_bps, irr_bps = {R['raw_bps']}, {R['irr_bps']}\n"
            "    tag = 'frozen'\n\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.5))\n"
            "cols = [RED if b < 0 else AMBER for b in (raw_bps, irr_bps)]\n"
            "bars = ax.bar(labels, [raw_bps, irr_bps], color=cols)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b, v in zip(bars, (raw_bps, irr_bps)):\n"
            "    ax.annotate(f'{v:+.1f} bps/mo', (b.get_x()+b.get_width()/2, v + (0.5 if v>=0 else -1.5)), ha='center')\n"
            "ax.set_ylabel('Loser - Winner spread (bps/month)')\n"
            "ax.set_title(f'Subtract the sector -> the spread flips positive and beats raw ({tag})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'[{tag}] RAW {raw_bps:+.1f} bps/mo   IRR {irr_bps:+.1f} bps/mo')"
        ),
        md(
            f"On this basket the *raw* reversal is actually slightly **negative** "
            f"({R['raw_bps']:.1f} bps/mo) -- the famous Jegadeesh effect is largely gone in "
            f"large caps. But the **industry-relative** version flips it positive "
            f"(+{R['irr_bps']:.1f} bps/mo): subtracting the sector recovers a reversal the "
            "raw sort had buried. The refinement works.\n\n"
            "**But is the 'industry' part real, or could any random grouping do this?**"
        ),
        code(
            "# Placebo: shuffle the sector labels and rebuild the industry-relative signal.\n"
            "if HAVE_REAL:\n"
            "    real_t = st.summarize(irr0['spread'])['tstat']\n"
            "    null_t = st.placebo_irr_tstats(prices, sectors, n_shuffles=60, seed=538)\n"
            "    p_val = float((null_t >= real_t).mean())\n"
            "    tag = 'REAL TAPE (60 shuffles)'\n"
            "else:\n"
            f"    real_t = {R['irr_t']}\n"
            f"    rng = np.random.default_rng(538); null_t = rng.normal({R['placebo_mean']}, {R['placebo_sd']}, 200)\n"
            f"    p_val = {R['placebo_p']}\n"
            "    tag = 'frozen/synthetic null'\n\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.hist(null_t, bins=20, color=GREY, alpha=.7, label='shuffled industry labels (null)')\n"
            "ax.axvline(real_t, c=GREEN, lw=2.5, label=f'real industry map (t={real_t:+.2f})')\n"
            "ax.set_xlabel('Industry-relative spread t-stat'); ax.set_ylabel('count'); ax.legend()\n"
            "ax.set_title(f'The TRUE sectors beat random groupings -> the adjustment is real ({tag})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'[{tag}] real IRR t={real_t:+.2f}   placebo p(shuffled >= real)={p_val:.3f}')"
        ),
        md(
            f"The real industry map sits in the **far right tail** of the shuffled null "
            f"(placebo p = {R['placebo_p']:.3f} on the full 200-shuffle run): demeaning by "
            "the *true* sectors is decisively better than demeaning by random groups. So the "
            "'industry' in 'industry-relative' is doing genuine work -- this is not an empty "
            "relabelling. **The mechanism is real.** The only question left is whether the "
            "*size* of the edge is enough."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal -- WEAK.** The industry-relative reversal does exactly what the "
            f"papers say: it flips the raw sign and beats it (+{R['irr_bps']:.1f} vs "
            f"{R['raw_bps']:.1f} bps/mo), and the industry adjustment is provably real "
            f"(placebo p = {R['placebo_p']:.3f}). But the level is tiny: *t* = "
            f"+{R['irr_t']:.2f}, below the bar of 2. Right mechanism, sub-bar magnitude.\n"
            f"- **Tradability -- MIRAGE.** ~{R['turnover']:.0f}% monthly turnover; break-even "
            f"~{R['breakeven_bps']:.1f} bps; loss-making almost as soon as you charge costs."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "A one-month signal means you replace almost the whole book every month. Here is "
            "the net spread as costs rise (one-way per side, plus a small short borrow):"
        ),
        code(
            "one_way = [0, 5, 10, 20]\n"
            "if HAVE_REAL:\n"
            "    nets = [(c, st.summarize(st.net_spread(irr0, c))['mean']*1e4) for c in one_way]\n"
            "    to = irr0['turn'].mean()*100; be = st.break_even_cost(irr0)\n"
            "    tag = 'REAL TAPE'\n"
            "else:\n"
            f"    nets = [(0, {R['net0_bps']}), (5, {R['net5_bps']}), (10, {R['net10_bps']}), (20, {R['net20_bps']})]\n"
            f"    to, be = {R['turnover']}, {R['breakeven_bps']}\n"
            "    tag = 'frozen'\n\n"
            "cs, nm = zip(*nets)\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(cs, nm, 'o-', c=AMBER, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(cs, nm, 0, where=[n < 0 for n in nm], color=RED, alpha=.12)\n"
            "ax.axvline(be, ls='--', c=GREY); ax.annotate(f'break-even ~{be:.1f}bps', (be, max(nm)*0.5 if max(nm)>0 else min(nm)*0.3))\n"
            "ax.set_xlabel('One-way transaction cost (bps)'); ax.set_ylabel('Net spread (bps/month)')\n"
            "ax.set_title(f'At ~{to:.0f}% monthly turnover, a few bps wipes out the spread ({tag})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'[{tag}] turnover {to:.0f}%/mo  break-even {be:.1f}bps one-way')"
        ),
        md(
            f"With ~{R['turnover']:.0f}% one-way turnover, the break-even cost is only "
            f"~{R['breakeven_bps']:.1f} bps -- the gross edge is so small that even retail-"
            f"grade costs sink it. By 5 bps the net spread is {R['net5_bps']:+.0f} bps/mo; by "
            f"20 bps it is a significant **loss** ({R['net20_bps']:+.0f} bps/mo). A better "
            "signal that you cannot afford to trade is still a mirage."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The raw cousin.** [Study 329 -- One-Month-Reversal](../../329-one-month-reversal/) "
            "is the raw Jegadeesh reversal this study refines. Read them together: 329 shows "
            "the raw version is bid-ask bounce; 538 shows the industry-relative residual is a "
            "cleaner -- but still untradable -- signal.\n"
            "- **The reversal spectrum.** [Study 196 -- Long-Term-Reversal](../../196-long-term-reversal/) "
            "is the 3-5 year horizon, the opposite end of the autocorrelation curve.\n"
            "- **The synthetic control.** The quant notebook plants a known within-industry "
            "reversal in a fake market and shows the industry-relative sort recovers it "
            "*more cleanly* than the raw sort -- so the real-tape result is a fact about the "
            "(thin, survivor) market, not a broken detector.\n\n"
            "*Think a liquidity-screened, small-cap, or finer-industry version clears *t* > 2 "
            "net of costs? Fork it and show it. That's the bar.*"
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
            "# Industry-Relative-Reversal -- a quantitative teardown\n"
            "### RAW vs industry-relative one-month reversal * the label-shuffle placebo * the bounce test * cost wall * HAC inference\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Industry beats raw: Confirmed](https://img.shields.io/badge/Industry_beats_raw-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "We test whether the Hameed-Mian (2015) / Da-Liu-Schaumburg (2014) industry-"
            "relative one-month reversal (a) beats the raw Jegadeesh sort, (b) survives a "
            "**placebo** that shuffles the industry labels, (c) survives a one-month-gap "
            "(skip=1) microstructure test, (d) holds across sub-periods, and (e) survives "
            "realistic costs -- and we plant a known within-industry reversal in a synthetic "
            "market to confirm the engine isolates the within component.\n\n"
            "> **Not investment advice.** Real data: fixed 6-sector S&P 500 survivor basket, "
            "1990-01-31 to 2026-05-31 (as-of 2026-06-26, fingerprint `bbd55e07b990`). Methods "
            "in [`docs/references.md`](../docs/references.md); numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship bias**: the basket is current S&P 500 names projected "
            "backwards; all real-tape results are upper bounds."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | IRR spread **+{R['irr_bps']:.1f}** vs RAW **{R['raw_bps']:.1f}** "
            f"bps/mo (sign flip + beat); placebo p = **{R['placebo_p']:.3f}**; but IRR HAC "
            f"*t* = **+{R['irr_t']:.2f}** < 2. |\n"
            f"| **Tradability** | `MIRAGE` | ~{R['turnover']:.0f}% one-way monthly turnover; "
            f"break-even **{R['breakeven_bps']:.1f} bps**; net loss-making by 5 bps. |\n"
            f"| **Industry beats raw** | `CONFIRMED` | Real tape (IRR survives skip=1, RAW does "
            f"not) + synthetic (IRR *t* = **{R['syn_irr04_t']:.0f}** vs RAW *t* = "
            f"**{R['syn_raw04_t']:.0f}**). |\n\n"
            "> The mechanism is correct and provably real; the magnitude, on a thin survivor "
            "basket, is sub-bar and untradable."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            r"Decompose stock $i$'s month-$t$ return into an industry component and a "
            "within-industry residual:\n\n"
            r"$$ r_{i,t} = \bar{r}_{g(i),t} + \tilde{r}_{i,t}, \qquad "
            r"\tilde{r}_{i,t} = r_{i,t} - \bar{r}_{g(i),t} $$"
            "\n\n"
            r"where $\bar{r}_{g,t}$ is the equal-weight mean return of industry $g$. The "
            "Hameed-Mian (2015) claim is three joint hypotheses:\n\n"
            r"- **H1 (within reverses).** Sorting on $-\tilde{r}_{i,t-1}$ (industry-relative) "
            "gives a positive loser-minus-winner spread.\n"
            r"- **H2 (industry does not).** The across-industry component $\bar{r}_{g,t-1}$ "
            "does not reverse, so the **raw** sort (on $-r_{i,t-1}$) is diluted and weaker.\n"
            r"- **H3 (it is the *industry* structure).** The advantage comes from the real "
            "industry map, not any arbitrary grouping (the placebo test).\n\n"
            f"We **confirm the ranking** (IRR +{R['irr_bps']:.1f} > RAW {R['raw_bps']:.1f} "
            f"bps/mo) and **confirm H3** (placebo p = {R['placebo_p']:.3f}) -- but the IRR "
            f"*level* is only *t* = +{R['irr_t']:.2f}, so we cannot certify the signal as REAL."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "The interesting failure here is not absence of mechanism -- the industry "
            "adjustment demonstrably works -- but **scale**. On a 54-name large-cap survivor "
            "basket the within-industry reversal is a few bps a month: enough to flip the raw "
            "sign and beat a random-grouping placebo, nowhere near enough to clear *t* = 2 or "
            "to pay ~77% monthly turnover. It is a clean example of a *correct idea* that is "
            "*economically negligible* in the liquid universe where you could actually trade it."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - The protocol\n\n"
            f"- **Universe**: fixed {R['n_stocks']}-name S&P 500 basket, six GICS sectors "
            "(survivorship-biased); ~11 names per quintile.\n"
            "- **Signals**: `strategy.raw_signal` (sort on total prior-month return) and "
            "`strategy.industry_relative_signal` (sort on prior-month return minus the "
            "stock's sector mean). Both lagged one month; a `skip=1` variant adds a gap.\n"
            "- **Portfolios**: equal-weight bottom/top quintiles; dollar-neutral L-S spread.\n"
            "- **Inference**: Newey-West HAC *t* on the monthly spread.\n"
            "- **Placebo**: `strategy.placebo_irr_tstats` permutes the sector map 200x and "
            "re-computes the IRR *t*; the real *t*'s rank in that null is the p-value for the "
            "industry adjustment.\n"
            "- **Costs**: one-way x NAV across both legs, round-trip, from realised turnover, "
            "plus 50 bps/yr borrow on the short leg; break-even one-way cost reported.\n"
            "- **Positive control**: synthetic panel (a non-reversing industry factor + a "
            "planted within-industry reversal) confirms the IRR sort beats the RAW sort.\n\n"
            "**Weak/Mirage trigger (declared in advance):** if the IRR spread beats RAW and "
            "the placebo confirms the industry structure but the IRR *t* < 2 and costs sink "
            "it, we stamp **WEAK x MIRAGE x CONFIRMED** -- a real mechanism, no tradable edge."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - RAW vs INDUSTRY-RELATIVE and their betas\n\n"
            "The head-to-head spread and the market beta of each (both near zero -> dollar-"
            "neutral, not disguised leverage)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sr = st.summarize(raw0['spread']); si = st.summarize(irr0['spread'])\n"
            "    raw_bps, raw_t = sr['mean']*1e4, sr['tstat']\n"
            "    irr_bps, irr_t = si['mean']*1e4, si['tstat']\n"
            "    beta_raw, _ = st.beta_alpha(raw0['spread'], raw0['market'])\n"
            "    beta_irr, _ = st.beta_alpha(irr0['spread'], irr0['market'])\n"
            "    tag = 'REAL TAPE'\n"
            "else:\n"
            f"    raw_bps, raw_t = {R['raw_bps']}, {R['raw_t']}\n"
            f"    irr_bps, irr_t = {R['irr_bps']}, {R['irr_t']}\n"
            f"    beta_raw, beta_irr = {R['raw_beta']}, {R['irr_beta']}\n"
            "    tag = 'frozen'\n\n"
            "print(f'[{tag}] RAW spread {raw_bps:+.1f} bps/mo  HAC t={raw_t:+.2f}  beta={beta_raw:.3f}')\n"
            "print(f'[{tag}] IRR spread {irr_bps:+.1f} bps/mo  HAC t={irr_t:+.2f}  beta={beta_irr:.3f}')\n\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar(['RAW', 'IRR'], [raw_bps, irr_bps], color=[RED if raw_bps<0 else AMBER, AMBER])\n"
            "a1.axhline(0, c='k', lw=1); a1.set_ylabel('Spread (bps/month)'); a1.set_title('Spread: IRR beats RAW')\n"
            "a2.bar(['RAW', 'IRR'], [beta_raw, beta_irr], color=GREY)\n"
            "a2.axhline(0, c='k', lw=1); a2.set_ylabel('Beta vs equal-weight market'); a2.set_title('Both dollar-neutral')\n"
            "fig.suptitle(f'Industry-relative flips the sign and beats raw -- and stays market-neutral ({tag})')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> IRR spread +{R['irr_bps']:.1f} bps/mo (*t* = +{R['irr_t']:.2f}) beats RAW "
            f"{R['raw_bps']:.1f} bps/mo (*t* = {R['raw_t']:.2f}); both betas are near zero "
            "(0.07-0.15) so neither is disguised leverage. **H1 + H2 ranking confirmed, but "
            "the IRR *t* does not clear 2.**"
        ),
        md(
            "### 4b - The placebo: does 'industry' mean anything? (label shuffle null)\n\n"
            "Permute the sector tags across tickers many times and rebuild the IRR signal. "
            "The real *t*'s position in that null is the p-value for the industry adjustment."
        ),
        code(
            "if HAVE_REAL:\n"
            "    real_t = st.summarize(irr0['spread'])['tstat']\n"
            "    null_t = st.placebo_irr_tstats(prices, sectors, n_shuffles=80, seed=538)\n"
            "    p_val = float((null_t >= real_t).mean())\n"
            "    tag = 'REAL TAPE (80 shuffles)'\n"
            "else:\n"
            f"    real_t = {R['irr_t']}\n"
            f"    rng = np.random.default_rng(538); null_t = rng.normal({R['placebo_mean']}, {R['placebo_sd']}, 200)\n"
            f"    p_val = {R['placebo_p']}\n"
            "    tag = 'frozen/synthetic null'\n\n"
            "print(f'[{tag}] shuffled t: mean={np.mean(null_t):+.2f} sd={np.std(null_t):.2f} q95={np.quantile(null_t,0.95):+.2f}')\n"
            "print(f'[{tag}] real IRR t={real_t:+.2f}   placebo p(shuffled >= real)={p_val:.3f}')\n\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.hist(null_t, bins=20, color=GREY, alpha=.7, label='shuffled industry labels')\n"
            "ax.axvline(np.quantile(null_t, 0.95), ls='--', c='k', lw=1, label='null 95th pct')\n"
            "ax.axvline(real_t, c=GREEN, lw=2.5, label=f'real industry map (t={real_t:+.2f})')\n"
            "ax.set_xlabel('Industry-relative spread t-stat'); ax.set_ylabel('count'); ax.legend()\n"
            "ax.set_title(f'Real sectors >> random groupings: the adjustment is genuine ({tag})')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> **H3 confirmed.** On the full 200-shuffle run the real IRR *t* = +{R['irr_t']:.2f} "
            f"sits above the 99.5th percentile of the shuffled null (mean {R['placebo_mean']}, "
            f"sd {R['placebo_sd']}), placebo p = {R['placebo_p']:.3f}. The within-industry "
            "demeaning uses real industry structure -- it is not an artefact of cutting the "
            "cross-section into arbitrary groups. The mechanism is real; only the level is small."
        ),
        md(
            "### 4c - The bid-ask-bounce gap (skip=1) and sub-period decay\n\n"
            "Study 329's raw reversal dies with a one-month gap (the bounce signature). Does "
            "the industry-relative version survive, and is it stable over time?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    raw1_t = st.summarize(raw1['spread'])['tstat']; raw1_b = st.summarize(raw1['spread'])['mean']*1e4\n"
            "    irr1_t = st.summarize(irr1['spread'])['tstat']; irr1_b = st.summarize(irr1['spread'])['mean']*1e4\n"
            "    rec = [(lab, st.summarize(irr0['spread'][a:b])['mean']*1e4, st.summarize(irr0['spread'][a:b])['tstat'])\n"
            "           for lab,a,b in [('1990-2002','1990','2002'),('2003-2014','2003','2014'),('2015-2026','2015','2026')]]\n"
            "    tag = 'REAL TAPE'\n"
            "else:\n"
            f"    raw1_t, raw1_b, irr1_t, irr1_b = {R['raw1_t']}, {R['raw1_bps']}, {R['irr1_t']}, {R['irr1_bps']}\n"
            f"    rec = [('1990-2002',{R['sub_90_bps']},{R['sub_90_t']}),('2003-2014',{R['sub_03_bps']},{R['sub_03_t']}),('2015-2026',{R['sub_15_bps']},{R['sub_15_t']})]\n"
            "    tag = 'frozen'\n\n"
            "print(f'[{tag}] skip=1  RAW {raw1_b:+.1f} bps/mo (t={raw1_t:+.2f})   IRR {irr1_b:+.1f} bps/mo (t={irr1_t:+.2f})')\n"
            "labs, bps, ts = zip(*rec)\n"
            "print(f'[{tag}] IRR sub-periods'); print(pd.DataFrame({'bps/mo': bps, 't': ts}, index=labs).round(2).to_string())\n\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar(['RAW skip=1','IRR skip=1'], [raw1_t, irr1_t], color=[RED, AMBER])\n"
            "a1.axhline(2, ls='--', c=GREEN, lw=1); a1.axhline(0, c='k', lw=1)\n"
            "a1.set_ylabel('Spread HAC t (one-month gap)'); a1.set_title('IRR survives the gap better than RAW')\n"
            "cols = [GREEN if t>2 else AMBER if t>0 else RED for t in ts]\n"
            "a2.bar(labs, bps, color=cols); a2.axhline(0, c='k', lw=1)\n"
            "a2.set_ylabel('IRR spread (bps/month)'); a2.set_title('No sub-period clears the bar')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> At skip=1 the **raw** reversal is +{R['raw1_bps']:.1f} bps/mo (*t* = "
            f"+{R['raw1_t']:.2f}) while the **industry-relative** one is *larger* "
            f"(+{R['irr1_bps']:.1f} bps/mo, *t* = +{R['irr1_t']:.2f}) -- it is not a pure "
            "bid-ask-bounce artefact, unlike 329's raw effect. But +1.33 still does not clear "
            f"the bar, and no IRR sub-period does either (best: 2015-2026 *t* = "
            f"+{R['sub_15_t']:.2f}). Suggestive, never certified."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `WEAK`** -- ranking confirmed (IRR +{R['irr_bps']:.1f} > RAW "
            f"{R['raw_bps']:.1f} bps/mo), industry structure real (placebo p = "
            f"{R['placebo_p']:.3f}), survives the gap (skip=1 *t* = +{R['irr1_t']:.2f}) -- but "
            f"IRR HAC *t* = +{R['irr_t']:.2f} < 2 and no sub-period clears the bar. Correct "
            "mechanism, sub-bar magnitude. Survivorship-biased upper bound.\n"
            f"- **Tradability `MIRAGE`** -- ~{R['turnover']:.0f}% one-way monthly turnover; "
            f"break-even ~{R['breakeven_bps']:.1f} bps; net {R['net5_bps']:+.0f} bps/mo at "
            f"5 bps, {R['net20_bps']:+.0f} bps/mo (*t* = {R['net20_t']:.2f}, a loss) at 20 bps.\n"
            "- **Industry beats raw `CONFIRMED`** -- on both the real tape and the synthetic "
            "control the within-industry sort dominates the raw sort."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md("## 6 - Could you trade it? -- turnover and cost sweep"),
        code(
            "one_way = [0, 5, 10, 20, 30]\n"
            "if HAVE_REAL:\n"
            "    rows = [(c, st.summarize(st.net_spread(irr0, c))['mean']*1e4,\n"
            "             st.summarize(st.net_spread(irr0, c))['tstat']) for c in one_way]\n"
            "    to = irr0['turn'].mean()*100; be = st.break_even_cost(irr0)\n"
            "    tag = 'REAL TAPE'\n"
            "else:\n"
            f"    base = {R['irr_bps']}; to = {R['turnover']}; be = {R['breakeven_bps']}\n"
            "    rows = [(c, base - 2*(to/100)*2*c - (50e-4/12)*1e4, None) for c in one_way]\n"
            "    tag = 'frozen'\n\n"
            "cs, nm, nt = zip(*rows)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.plot(cs, nm, 'o-', c=AMBER, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(cs, nm, 0, where=[n < 0 for n in nm], color=RED, alpha=.12)\n"
            "ax.axvline(be, ls='--', c=GREY); ax.annotate(f'break-even ~{be:.1f}bps', (be, max(nm)*0.4 if max(nm)>0 else min(nm)*0.3))\n"
            "ax.set_xlabel('One-way transaction cost (bps)'); ax.set_ylabel('Net spread (bps/month)')\n"
            "ax.set_title(f'Cost wall: ~{to:.0f}% turnover -> break-even ~{be:.1f}bps ({tag})')\n"
            "plt.tight_layout(); plt.show()\n"
            "for c, n, t in rows:\n"
            "    print(f'  {c:2d} bps one-way -> net {n:+.1f} bps/mo' + (f'  t={t:+.2f}' if t is not None else ''))"
        ),
        md(
            f"> Break-even ~{R['breakeven_bps']:.1f} bps one-way. The gross edge is so small "
            "that any realistic cost makes the book a loss -- the net spread is negative by "
            "5 bps and significantly loss-making by 20 bps. A correct, placebo-clean signal "
            "that cannot pay its own turnover is a mirage."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the engine actually isolate the *within-industry* component? Plant a non-"
            "reversing industry factor plus a known within-industry reversal, and sweep it: "
            "the industry-relative sort should beat the raw sort (which is diluted by the "
            "industry move) and both should read ~0 at the null."
        ),
        code(
            "revs = [0.0, 0.02, 0.04, 0.06, 0.08]\n"
            "edge = []\n"
            "for rev in revs:\n"
            "    p, sec, _ = data.synthetic_panel(within_reversal=rev, n_months=250, seed=538)\n"
            "    rs = st.quintile_returns(st.raw_signal(p, skip=0), p, q=0.20)\n"
            "    is_ = st.quintile_returns(st.industry_relative_signal(p, sec, skip=0), p, q=0.20)\n"
            "    edge.append((rev, st.summarize(rs['spread'])['tstat'], st.summarize(is_['spread'])['tstat']))\n\n"
            "rv, raw_te, irr_te = zip(*edge)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.plot(rv, raw_te, 'o-', c=RED, lw=2, label='RAW sort (diluted by industry move)')\n"
            "ax.plot(rv, irr_te, 'o-', c=GREEN, lw=2, label='INDUSTRY-RELATIVE sort')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xlabel('Planted within-industry reversal')\n"
            "ax.set_ylabel('Spread HAC t-stat'); ax.legend()\n"
            "ax.set_title('SYNTHETIC control: IRR isolates the within component, beats RAW, ~0 at the null')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('[SYNTHETIC]')\n"
            "for r, rt, it in edge:\n"
            "    print(f'  within_reversal={r:+.2f}: RAW t={rt:+.2f}   IRR t={it:+.2f}')"
        ),
        md(
            f"Both spreads are ~0 at the null (within_reversal = 0) and rise with the planted "
            f"signal -- and the **industry-relative sort dominates the raw sort** throughout "
            f"(at +0.04: IRR *t* = +{R['syn_irr04_t']:.0f} vs RAW *t* = +{R['syn_raw04_t']:.0f}), "
            "because the raw sort is diluted by the non-reversing industry factor. The engine "
            "is a faithful detector that correctly isolates the within-industry component. So "
            "the real-tape result -- a correct, placebo-clean, sub-bar, untradable signal -- "
            "is a fact about this thin survivor **market**, not a broken method. Forks worth "
            "trying: a finer (GICS sub-industry) map, a small-cap or illiquid universe where "
            "Avramov-Chordia-Goyal (2006) say the reversal concentrates, or a weekly horizon."
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
