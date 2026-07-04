"""Generate the two narrative notebooks for Study 592 (Dual Momentum — GEM).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached ETF closes +
^IRX under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SPY/EFA/AGG(IEF)/^IRX,
# 2002-09-30 -> 2026-06-30, 286 months, as-of 2026-06-30, fingerprint 837a69932f70).
R = dict(
    start="2002-09-30", end="2026-06-30", months=286, years=23.8,
    fingerprint="837a69932f70", asof="2026-06-30",
    alloc=dict(SPY=50.0, EFA=31.1, BOND=18.9), switches=35, switches_yr=1.47,
    # leg -> (CAGR %, vol %, Sharpe excess, max DD %)
    gem=(9.87, 12.15, 0.70, -20.7),
    spy=(11.22, 14.77, 0.68, -50.8),
    b6040=(8.27, 9.34, 0.72, -32.3),
    dd_ratio=0.41,
    # active: (bps/mo, %/yr, HAC t)
    act_spy=(-13.3, -1.59, -0.64),
    act_6040=(14.8, 1.77, 0.89),
    # costs: (one-way bps, net CAGR %, net Sharpe, active vs SPY %/yr, HAC t)
    costs=[(5, 9.71, 0.69, -1.74, -0.70),
           (10, 9.54, 0.68, -1.89, -0.75),
           (20, 9.22, 0.65, -2.19, -0.86)],
    # grid: (lookback, CAGR %, Sharpe, max DD %, active %/yr, HAC t)
    grid=[(3, 10.46, 0.80, -19.2, -1.18, -0.52),
          (6, 8.97, 0.66, -24.9, -2.49, -0.95),
          (9, 8.79, 0.64, -20.1, -2.65, -1.06),
          (12, 9.87, 0.70, -20.7, -1.59, -0.64)],
    # random-switching baseline (40 seeds, averaged)
    rb=dict(n_seeds=40, welch_t=0.26, base_cagr=8.53, base_sharpe=0.54, beat_pct=77.5),
    # subperiods: (label, GEM CAGR, GEM DD, GEM Sh, SPY CAGR, SPY DD, SPY Sh, active %/yr, HAC t)
    subs=[("full sample", 9.87, -20.7, 0.70, 11.22, -50.8, 0.68, -1.59, -0.64),
          ("ex-GFC (drop 2007-07..2009-06)", 11.00, -20.7, 0.78, 14.62, -23.9, 0.95, -3.43, -2.02),
          ("2010 ->", 8.57, -20.7, 0.59, 14.23, -23.9, 0.90, -5.32, -3.37),
          ("pre-2013", 10.80, -17.3, 0.79, 6.46, -50.8, 0.38, 3.53, 0.71),
          ("post-2013", 9.16, -20.7, 0.63, 15.01, -23.9, 0.94, -5.51, -2.96)],
    # decay third axis
    pre_bps=29.4, pre_n=124, pre_t=0.71,
    post_bps=-45.9, post_n=162, post_t=-2.96, decay_welch=2.02,
    # synthetic control: (label, active %/yr, HAC t, GEM DD %, B&H DD %)
    syn=[("null (iid regimes)", -1.58, -1.24, -37, -61),
         ("planted (persistence 0.94)", 4.67, 2.86, -28, -93)],
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Decayed_since_2013%3F: Confirmed](https://img.shields.io/badge/Decayed_since_2013%3F-Confirmed-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from dual_momentum_gem import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PANEL = data.monthly_panel(asof=data.AS_OF)
    GEM = st.run_gem(PANEL)
    BM = st.benchmark_returns(PANEL, GEM.index)
else:
    PANEL = GEM = BM = None
print("real GEM cache present:", HAVE_REAL,
      "| months:", (0 if GEM is None else len(GEM)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Dual Momentum: the retail strategy that promised the market's return with half the pain 🌍\n"
            "### Antonacci's GEM — does the most famous switch-three-ETFs recipe actually work?\n\n"
            + BADGES +
            "In 2014 Gary Antonacci published *Dual Momentum Investing*, and it became **the** retail "
            "allocation recipe: once a month, look at the last 12 months. If US stocks beat T-bills, own "
            "the better of **US stocks (SPY)** or **international stocks (EFA)**. If they didn't, hide in "
            "**bonds (AGG)**. That's it — one look, at most one trade, per month. The pitch: **the "
            "market's return with half the drawdown**, because the 12-month filter walks you out of "
            "crashes like 2008 before the worst of them.\n\n"
            "It's cheap, simple and backtested to 1974. So we put the *live-ETF* version — the one you "
            "could actually have run — on the desk.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the placebo grid and the shuffled-"
            "timing baseline? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Broad index ETFs — no survivorship. But the honest tape only "
            "starts in **2002** (EFA's first full momentum window): the celebrated 1974-2011 backtest "
            "years are *not* observable with live funds. Every chart is drawn by the code beside it; "
            "house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Half the drawdown? | **True.** Worst fall −20.7% vs the market's −50.8% — better than "
            "half. The filter really did sidestep 2008. |\n"
            "| Beats buy-and-hold? | **No.** Over 2002-2026 GEM made **9.9%/yr vs SPY's 11.2%/yr** — it "
            "*lags* the market, and the gap is statistical noise at best. |\n"
            "| Is it all one lucky call? | **Essentially yes — 2008.** Remove the two crisis years and "
            "GEM *significantly underperforms*. From 2010 on it lost **−5.3%/yr** to the market. |\n"
            "| Did publishing kill it? | **It flipped.** Before 2013: mildly positive. After: "
            "**−5.5%/yr vs SPY** — whipsawed in 2015-16, out for the 2020 rebound, lagging 2023-25. |\n"
            "| Is a boring 60/40 worse? | **No — it's a wash.** 60/40's risk-adjusted score (Sharpe "
            "0.72) actually edges GEM's (0.70), with zero cleverness. |\n\n"
            "> One honest sentence: **GEM is a drawdown-reduction machine that paid for it with the "
            "market's upside — not a market-beating machine.**"
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Combine **absolute momentum** (are stocks beating cash over the last 12 months?) with "
            "**relative momentum** (US or international — own the stronger). You'll capture bull markets, "
            "sidestep bear markets, and beat buy-and-hold with roughly half the maximum drawdown.\"*\n\n"
            "That's Antonacci's **Global Equities Momentum**, the flagship of *Dual Momentum Investing* "
            "(2014) and the 2012-13 *Risk Premia Harvesting* paper. It is probably the most-followed "
            "tactical allocation recipe among retail investors — three ETF tickers and a spreadsheet."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a one-line monthly rule beat the market with half the pain, every pension fund and every "
            "index investor should run it — the equity premium would be free *and* comfortable. The "
            "counter-story: trend rules buy their crash protection with **whipsaws** (sell low after "
            "falls, buy back high after rebounds), so over a long tape the insurance premium eats the "
            "outperformance. Which story does the live tape support?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We rebuild GEM exactly as prescribed, on the real funds: **{R['months']} months** "
            f"({R['start']} → {R['end']}, {R['years']:.1f} years). Each month-end: SPY's 12-month return "
            "vs T-bills (^IRX, known a month in advance) — pass: hold the better of SPY/EFA; fail: hold "
            "AGG (IEF before 2003). The decision earns the **following** month — one clean execution "
            "lag, no crystal ball. Then we race it against **just holding SPY** and against a boring "
            "**60/40**, count every switch's cost, try other lookbacks (3/6/9 months), scramble the "
            "timing 40 different ways, and cut the sample around 2008 and around the 2013 publication."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The whole story in one chart.** Growth of $1, log scale — and the drawdown that sells "
            "the strategy."
        ),
        code(
            "if HAVE_REAL:\n"
            "    eq_gem = (1+GEM['gross']).cumprod(); eq_spy = (1+BM['SPY']).cumprod(); eq_64 = (1+BM['B6040']).cumprod()\n"
            "    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.5, 7.6), sharex=True,\n"
            "                                 gridspec_kw={'height_ratios': [2.2, 1]})\n"
            "    a1.plot(eq_gem, c=AMBER, lw=1.8, label='GEM (dual momentum)')\n"
            "    a1.plot(eq_spy, c=GREY, lw=1.8, label='SPY buy & hold')\n"
            "    a1.plot(eq_64, c=GREEN, lw=1.4, ls=':', label='60/40')\n"
            "    a1.set_yscale('log'); a1.legend(); a1.set_ylabel('growth of $1 (log)')\n"
            "    a1.set_title('GEM dodged 2008 beautifully - then spent 15 years paying it back')\n"
            "    dd_g = eq_gem/eq_gem.cummax()-1; dd_s = eq_spy/eq_spy.cummax()-1\n"
            "    a2.fill_between(dd_s.index, dd_s*100, 0, color=GREY, alpha=.5, label='SPY drawdown')\n"
            "    a2.fill_between(dd_g.index, dd_g*100, 0, color=AMBER, alpha=.6, label='GEM drawdown')\n"
            "    a2.set_ylabel('drawdown (%)'); a2.legend(loc='lower right')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'final $1 -> GEM ${eq_gem.iloc[-1]:.2f}  SPY ${eq_spy.iloc[-1]:.2f}  60/40 ${eq_64.iloc[-1]:.2f}')\n"
            "    print(f'max drawdown  GEM {dd_g.min()*100:.1f}%  SPY {dd_s.min()*100:.1f}%')\n"
            "else:\n"
            "    print('cache missing - frozen numbers:', 'GEM CAGR', R['gem'][0], '% vs SPY', R['spy'][0], '%; DD', R['gem'][3], 'vs', R['spy'][3])"
        ),
        md(
            f"Both halves of the sales pitch are visible. **The drawdown half is true**: GEM's worst "
            f"fall is **{R['gem'][3]:.1f}%** vs SPY's **{R['spy'][3]:.1f}%** — better than half. **The "
            f"return half is not**: GEM compounds at **{R['gem'][0]:.1f}%/yr vs SPY's "
            f"{R['spy'][0]:.1f}%/yr**. You paid ~1.6%/yr for the smoother ride — and a plain 60/40 "
            f"(Sharpe {R['b6040'][2]:.2f} vs GEM's {R['gem'][2]:.2f}) bought comfort just as "
            "efficiently with zero timing."
        ),
        md(
            "**When did it win, when did it lose?** The running gap between GEM and just holding SPY — "
            "with the 2013 publication marked."
        ),
        code(
            "if HAVE_REAL:\n"
            "    act = (GEM['gross'] - BM['SPY'])\n"
            "    cum = act.cumsum()*100\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(cum, c=AMBER, lw=1.8)\n"
            "    ax.axhline(0, c=GREY, lw=1)\n"
            "    ax.axvline(pd.Timestamp('2013-01-01'), c=RED, ls='--', label='strategy published (2013)')\n"
            "    ax.annotate('the one great sidestep:\\n2008', xy=(pd.Timestamp('2009-06-30'), cum.loc['2009-06-30']),\n"
            "                xytext=(pd.Timestamp('2011-06-30'), cum.max()*0.55), arrowprops=dict(arrowstyle='->'))\n"
            "    ax.set_ylabel('cumulative gap vs SPY (percentage points)')\n"
            "    ax.set_title('Everything GEM ever won came from 2008 - and publication marks the top')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'pre-2013 edge: {act[act.index<=\"2012-12-31\"].mean()*1e4:+.1f} bps/mo | '\n"
            "          f'post-2013: {act[act.index>=\"2013-01-01\"].mean()*1e4:+.1f} bps/mo')\n"
            "else:\n"
            "    print('cache missing - frozen:', R['pre_bps'], 'bps/mo pre-2013,', R['post_bps'], 'bps/mo post-2013')"
        ),
        md(
            f"The line tells the decade-by-decade truth: one huge win (dodging 2008), then a long, "
            f"steady bleed. Before 2013 GEM edged SPY by **{R['pre_bps']:+.0f} bps/month** (not "
            f"statistically significant even then); after publication it *lost* "
            f"**{R['post_bps']:.0f} bps/month** — about **−5.5%/yr** — getting whipsawed in 2015-16, "
            "sitting in bonds through the v-shaped 2020 rebound, and lagging the 2023-25 bull."
        ),
        md(
            "**Maybe 12 months is the wrong knob?** Same rule, lookbacks of 3, 6, 9 and 12 months — the "
            "gap vs buy-and-hold for each."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = st.lookback_grid(PANEL)\n"
            "    lbs = [r['lookback'] for r in rows]; acts = [r['act_ann_pct'] for r in rows]\n"
            "else:\n"
            "    lbs = [g[0] for g in R['grid']]; acts = [g[4] for g in R['grid']]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar([f'{l} months' for l in lbs], acts, color=[GREY, GREY, GREY, AMBER], width=.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, v in enumerate(acts): ax.annotate(f'{v:+.1f}%/yr', (i, v), ha='center', va='top' if v<0 else 'bottom')\n"
            "ax.set_ylabel('GEM minus SPY (%/yr)')\n"
            "ax.set_title('No lookback beats buy-and-hold - 12 months is not hiding a better version')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('active vs SPY by lookback:', dict(zip(lbs, [round(a,2) for a in acts])))"
        ),
        md(
            "Every bar is below zero. The canonical 12 months isn't unlucky — **there is no lookback on "
            "this tape that turns GEM into a market-beater**. (12 is actually the second-best of the "
            "four, so the recipe wasn't sabotaged by its own parameter either.)"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Mixed.** \"Half the drawdown\" is real: **{R['gem'][3]:.0f}% vs "
            f"{R['spy'][3]:.0f}%**. \"Beats buy-and-hold\" is not: **−1.6%/yr vs SPY** over 24 years, "
            "and *significantly* negative once you remove 2008 or start at 2010.\n"
            "- **Tradability — Mirage.** Not because it's hard — three giant ETFs, ~1.5 trades a year, "
            "costs of a few bps — but because the thing being sold (market-beating returns with half "
            "the risk) isn't on the deployable tape. What you can deploy is beta plus a 2008 story, and "
            "a 60/40 matches its risk-adjusted result with zero effort.\n"
            "- **Decayed since publication? — Confirmed.** Mildly positive before 2013, **−5.5%/yr** "
            "after — the flip itself is statistically significant."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The 1974-2011 backtest is real — as a backtest.** Trend rules genuinely shone across "
            "1970s-2000s bear markets. The live-fund era (2002→) contains exactly one long bear GEM "
            "could sidestep (2008) and three fast ones it couldn't (2015-16, 2020, 2022-partial). "
            "Whether the next bear is slow (GEM wins) or fast (GEM whipsaws) is the whole bet.\n"
            "- **Drawdown control is not worthless** — for an investor who would capitulate at −50%, a "
            "−21% worst case has behavioural value. But name the price: ~1.6%/yr of expected return, "
            "the same trade a 60/40 offers passively.\n"
            "- **Siblings on this desk:** [country momentum](../../146-country-momentum/) (the "
            "cross-sectional leg alone) and [time-series momentum on futures]"
            "(../../518-time-series-momentum/) (the absolute leg, diversified). GEM is those two ideas "
            "packaged into a retail three-ticker product — and the packaging is where the promise "
            "outruns the tape.\n\n"
            "*Think the 2013 split is unfair? Pick any split you like — from 2010 onward the "
            "underperformance is −5.3%/yr with a HAC t of −3.4. Show us a live-tradable GEM variant "
            "with a positive active t and we'll re-open the file.*"
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
            "# Dual Momentum (GEM) — a quantitative teardown 🔬\n"
            "### HAC *t* on monthly active returns · excess-vs-excess Sharpe races · a 3/6/9/12 lookback "
            "placebo grid · a 40-seed shuffled-timing baseline · GFC-excision and publication-decay "
            "splits · a regime-persistence synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). GEM "
            "(Antonacci 2012-2014) is the composite retail allocation strategy built on absolute + "
            "relative momentum; distinct from this desk's cross-sectional "
            "[country momentum](../../146-country-momentum/) and futures-panel "
            "[time-series momentum](../../518-time-series-momentum/) studies.\n\n"
            "> ⚠️ **Data note.** Live-ETF tape only: SPY/EFA/AGG (IEF-spliced pre-2003) + ^IRX, "
            f"{R['start']} → {R['end']} ({R['months']} months). **No survivorship** (broad index ETFs); "
            "the material caveat is the **sample start** — the 1974-2011 years of the book's backtest "
            "are unobservable with live funds. Offline core + synthetic control are deterministic. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (as-of " + R["asof"] + ", fingerprint `" + R["fingerprint"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | *Real on the drawdown-halving* (maxDD **{R['gem'][3]:.1f}%** vs "
            f"SPY **{R['spy'][3]:.1f}%**, ratio {R['dd_ratio']:.2f}) · *None on the beats-B&H* (active "
            f"**{R['act_spy'][1]:+.2f}%/yr**, HAC **t = {R['act_spy'][2]:+.2f}**; ex-GFC t = −2.02; "
            "post-2013 t = −2.96; no lookback in the grid positive). |\n"
            f"| **Tradability** | `MIRAGE` | Deployment is trivial ({R['switches_yr']:.2f} switches/yr "
            "≈ 3-16 bps/yr) — but post-publication the deployable product cost **−5.51%/yr vs SPY** "
            f"(t = −2.96) and its Sharpe ({R['gem'][2]:.2f}) is matched by a no-timing 60/40 "
            f"({R['b6040'][2]:.2f}). |\n"
            f"| **Decayed since 2013?** | `CONFIRMED` | Pre **{R['pre_bps']:+.1f}** vs post "
            f"**{R['post_bps']:+.1f} bps/mo**; difference Welch **t = {R['decay_welch']:+.2f}**. |\n\n"
            "> 💡 In plain words: the crash-avoidance is real, the outperformance is not, and what "
            "remains after publication is significantly *negative* alpha in the world's most liquid "
            "wrapper."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{S}_t, r^{E}_t, r^{B}_t, r^{f}_t$ be monthly total returns of SPY, EFA, bonds and "
            "T-bills. At each month-end $t-1$ form 12-month compounds "
            "$M^{x}_{t-1} = \\prod_{k=1}^{12}(1+r^{x}_{t-k})-1$. GEM's position for month $t$:\n\n"
            "$$w_t = \\begin{cases} \\arg\\max\\{M^{S}, M^{E}\\} & \\text{if } M^{S}_{t-1} > M^{f}_{t-1} \\\\ "
            "\\text{BOND} & \\text{otherwise.} \\end{cases}$$\n\n"
            "- **H₁ (outperformance).** Mean monthly active return $\\overline{r^{GEM}-r^{SPY}} > 0$ "
            "with HAC *t* ≥ 2.\n"
            "- **H₂ (half the drawdown).** $\\text{maxDD}(GEM) \\le \\tfrac12\\,\\text{maxDD}(SPY)$.\n"
            "- **H₃ (not one lucky call).** H₁ survives excising 2007-07→2009-06.\n"
            "- **H₄ (no decay).** Active returns pre- and post-publication (2013) indistinguishable.\n\n"
            "We find **H₂ supported** (0.41 < 0.5), **H₁ rejected** (t = −0.64, wrong sign), **H₃ "
            "rejected** (ex-GFC active *significantly negative*, t = −2.02), **H₄ rejected** "
            "(Welch t = +2.02 on the flip)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Monthly active returns of a switching strategy are serially correlated (positions persist "
            "for months), so the outperformance test uses a **Newey-West (HAC)** *t* on the mean active "
            "return, Bartlett kernel, lag $\\lfloor 4(n/100)^{2/9}\\rfloor$. Sharpe races are "
            "**excess-of-T-bill on both sides** (a switching strategy that parks in bonds must not get "
            "credit for the cash yield). Sub-period contrasts carry a **Welch t of the difference** on "
            "a *justified* split (the 2013 publication date — pre-registered by the literature, not "
            "snooped). And any random baseline is **averaged over ≥ 20 seeds** — here 40 — because "
            "single-seed baselines are banned on this desk."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** yfinance total-return closes, {R['start']} → {R['end']} ({R['months']} months "
            "— starts at EFA's first full 12-month formation window). Bond leg = AGG, **IEF-spliced** "
            "before Sept 2003. Risk-free = ^IRX yield from the **prior** month-end (no look-ahead).\n"
            "- **Execution.** Signal from month-end closes; position earns the **following** month — "
            "exactly **one** decision-to-earn lag, documented. No same-bar fills.\n"
            "- **Costs.** One-way bps × NAV per traded leg; a switch = 2 legs. Grid 5/10/20 bps. "
            "Long-only — no borrow.\n"
            f"- **Benchmarks.** SPY buy-and-hold and monthly-rebalanced 60/40, same {R['months']} "
            "months, gross.\n"
            "- **Robustness.** Lookback placebo grid 3/6/9/12 on common months; 40-seed permutation of "
            "GEM's own holdings (same allocation, random timing); GFC excision; 2013 decay split.\n"
            "- **Machinery control.** Two-state Markov synthetic world, persistence knob; the null "
            "(i.i.d. regimes) must not light up, the planted world must."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Headline race — CAGR, excess Sharpe, drawdown, HAC t\n\n"
            "GEM vs SPY vs 60/40 on identical months; active-return t-stats below."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pg = st.perf(GEM['gross'], BM['RF']); ps = st.perf(BM['SPY'], BM['RF']); p64 = st.perf(BM['B6040'], BM['RF'])\n"
            "    a_spy = st.active_stats(GEM['gross'], BM['SPY']); a_64 = st.active_stats(GEM['gross'], BM['B6040'])\n"
            "    rows = [('GEM', pg), ('SPY', ps), ('60/40', p64)]\n"
            "    for n, p in rows:\n"
            "        print(f\"{n:>6}: CAGR {p['cagr']*100:+.2f}%  vol {p['vol']*100:.2f}%  Sharpe(excess) {p['sharpe']:.2f}  maxDD {p['max_dd']*100:.1f}%\")\n"
            "    print(f'  DD ratio GEM/SPY = {pg[\"max_dd\"]/ps[\"max_dd\"]:.2f}')\n"
            "    print(f'  active GEM-SPY  : {a_spy[\"ann_pct\"]:+.2f}%/yr  HAC t = {a_spy[\"hac_t\"]:+.2f}')\n"
            "    print(f'  active GEM-60/40: {a_64[\"ann_pct\"]:+.2f}%/yr  HAC t = {a_64[\"hac_t\"]:+.2f}')\n"
            "    tvals = [a_spy['hac_t'], a_64['hac_t']]\n"
            "else:\n"
            "    tvals = [R['act_spy'][2], R['act_6040'][2]]\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.2))\n"
            "ax.bar(['GEM - SPY', 'GEM - 60/40'], tvals, color=[AMBER, GREY], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = +2 bar'); ax.axhline(-2, ls='--', c=RED)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, v in enumerate(tvals): ax.annotate(f't={v:+.2f}', (i, v), ha='center', va='bottom' if v>0 else 'top')\n"
            "ax.set_ylabel('HAC t of mean monthly active return'); ax.set_ylim(-3.2, 3.2)\n"
            "ax.set_title('Neither race clears the bar - and the SPY race has the wrong sign')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: GEM lags SPY by **{R['act_spy'][1]:+.2f}%/yr** (HAC "
            f"t = {R['act_spy'][2]:+.2f}) and beats 60/40 by a non-significant "
            f"{R['act_6040'][1]:+.2f}%/yr (t = {R['act_6040'][2]:+.2f}). The one number that *is* "
            f"dramatic — maxDD **{R['gem'][3]:.1f}% vs {R['spy'][3]:.1f}%** — is the drawdown claim, "
            "and it holds (ratio 0.41 < ½). H₂ yes, H₁ no."
        ),
        md(
            "### 4b · Lookback placebo grid — is 12 special, or lucky?\n\n"
            "Same engine and months, only the formation window moves. If 12 were data-mined luck, "
            "neighbours would collapse; if momentum were robust here, several would be positive."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = st.lookback_grid(PANEL)\n"
            "    grid = [(r['lookback'], r['cagr']*100, r['sharpe'], r['max_dd']*100, r['act_ann_pct'], r['hac_t']) for r in rows]\n"
            "else:\n"
            "    grid = R['grid']\n"
            "for g in grid:\n"
            "    print(f'L={g[0]:>2}m: CAGR {g[1]:+.2f}%  Sharpe {g[2]:.2f}  maxDD {g[3]:.1f}%  active {g[4]:+.2f}%/yr  HAC t {g[5]:+.2f}')\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar([f'{g[0]}m' for g in grid], [g[5] for g in grid], color=[GREY]*3+[AMBER], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(2, ls='--', c=RED, label='t = +2 bar'); ax.axhline(-2, ls='--', c=RED)\n"
            "for i, g in enumerate(grid): ax.annotate(f'{g[5]:+.2f}', (i, g[5]), ha='center', va='top')\n"
            "ax.set_xlabel('formation lookback'); ax.set_ylabel('HAC t, active vs SPY'); ax.set_ylim(-3, 3)\n"
            "ax.set_title('The whole grid is negative: 12m is neither special nor lucky'); ax.legend()\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "> 💡 In plain words: **every** lookback lags buy-and-hold (active t between −0.5 and "
            "−1.1). The canonical 12 months is second-best — the published parameter wasn't a lucky "
            "draw, and no neighbouring parameter rescues the claim. There is simply no positive active "
            "return in this family on this tape."
        ),
        md(
            "### 4c · Random-switching baseline — 40 seeds, averaged\n\n"
            "Permute GEM's own monthly holdings (same assets, same 50/31/19 allocation, random timing), "
            "one permutation per seed, 40 seeds. If GEM's switch *dates* carry information, it should "
            "beat its own scrambled self."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rb = st.random_switch_baseline(PANEL, GEM, n_seeds=40)\n"
            "    print(f\"mean Welch t over {rb['n_seeds']} seeds : {rb['welch_t_mean']:+.2f}\")\n"
            "    print(f\"baseline mean CAGR / Sharpe    : {rb['base_cagr_mean']*100:+.2f}% / {rb['base_sharpe_mean']:.2f}\")\n"
            "    print(f\"GEM CAGR beats {rb['beat_share']*100:.1f}% of shuffles\")\n"
            "    wt, bc, bs, bp = rb['welch_t_mean'], rb['base_cagr_mean']*100, rb['base_sharpe_mean'], rb['beat_share']*100\n"
            "else:\n"
            "    wt, bc, bs, bp = R['rb']['welch_t'], R['rb']['base_cagr'], R['rb']['base_sharpe'], R['rb']['beat_pct']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.2))\n"
            "ax.bar(['shuffled timing\\n(40-seed mean)', 'GEM'], [bc, R['gem'][0]], color=[GREY, AMBER], width=.5)\n"
            "for i, v in enumerate([bc, R['gem'][0]]): ax.annotate(f'{v:.2f}%', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('CAGR (%)')\n"
            "ax.set_title(f'GEM beats its scrambled self (77% of seeds) - but mean Welch t = {wt:+.2f}')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: GEM's timing is better than random — it beats "
            f"**{R['rb']['beat_pct']:.0f}%** of its own scrambles and adds ~1.3 pp CAGR over their mean "
            f"(**{R['rb']['base_cagr']:.2f}%**) — that's the 2008 sidestep showing up. But the averaged "
            f"Welch t is **{R['rb']['welch_t']:+.2f}**: point-estimate skill, nowhere near the bar. "
            "(The scrambles also show how much of GEM's *risk reduction* is pure allocation: holding "
            "bonds 19% of the time at random already cuts vol and drawdown.)"
        ),
        md(
            "### 4d · Sub-periods and the decay split — does it live off 2008 alone?\n\n"
            "Excise the GFC (2007-07 → 2009-06), start at 2010, and split at the 2013 publication."
        ),
        code(
            "if HAVE_REAL:\n"
            "    subs = [('full', {}), ('ex-GFC', dict(drop=('2007-07-01','2009-06-30'))),\n"
            "            ('2010->', dict(start='2010-01-01')), ('pre-2013', dict(end='2012-12-31')),\n"
            "            ('post-2013', dict(start='2013-01-01'))]\n"
            "    res = []\n"
            "    for name, kw in subs:\n"
            "        s = st.subperiod_stats(PANEL, GEM, **kw)\n"
            "        res.append((name, s['active']['ann_pct'], s['active']['hac_t']))\n"
            "        print(f\"{name:<10} active {s['active']['ann_pct']:+.2f}%/yr  HAC t = {s['active']['hac_t']:+.2f}\")\n"
            "    act = (GEM['gross'] - BM['SPY']).dropna()\n"
            "    pre = act[act.index <= '2012-12-31']; post = act[act.index >= '2013-01-01']\n"
            "    print(f'decay Welch t (pre vs post 2013) = {st.welch_t(pre.values, post.values):+.2f}')\n"
            "else:\n"
            "    res = [(s[0], s[7], s[8]) for s in R['subs']]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "names = [r[0] for r in res]; ts = [r[2] for r in res]\n"
            "ax.bar(names, ts, color=[AMBER if t>=-2 else RED for t in ts], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(-2, ls='--', c=RED, label='t = -2')\n"
            "for i, r in enumerate(res): ax.annotate(f'{r[1]:+.1f}%/yr\\nt={r[2]:+.2f}', (i, r[2]), ha='center', va='top' if r[2]<0 else 'bottom', fontsize=9)\n"
            "ax.set_ylabel('HAC t, active vs SPY'); ax.set_ylim(-4.4, 2.2)\n"
            "ax.set_title('Remove 2008 and GEM significantly UNDERPERFORMS'); ax.legend()\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "> 💡 In plain words: the full-sample −0.64 hides a starker structure. **Ex-GFC: "
            "−3.43%/yr at t = −2.02. From 2010: −5.32%/yr at t = −3.37. Post-2013: −5.51%/yr at "
            "t = −2.96.** Everything GEM ever won it won in 2008; the decay flip (pre "
            f"{R['pre_bps']:+.1f} → post {R['post_bps']:+.1f} bps/mo) clears the bar at Welch "
            f"t = {R['decay_welch']:+.2f}. H₃ and H₄ both rejected — and the third axis stamps "
            "**CONFIRMED** on decay."
        ),
        md(
            "### 4e · Costs — eliminated as an excuse\n\n"
            "One-way bps × NAV per traded leg, 2 legs per switch, at 5/10/20 bps."
        ),
        code(
            "if HAVE_REAL:\n"
            "    for cb in (5.0, 10.0, 20.0):\n"
            "        g = st.run_gem(PANEL, cost_bps=cb)\n"
            "        pf = st.perf(g['net'], BM['RF']); a = st.active_stats(g['net'], BM['SPY'])\n"
            "        print(f'cost {cb:>4.1f} bps: net CAGR {pf[\"cagr\"]*100:+.2f}%  Sharpe {pf[\"sharpe\"]:.2f}  active vs SPY {a[\"ann_pct\"]:+.2f}%/yr (t {a[\"hac_t\"]:+.2f})')\n"
            "    n_sw = int((GEM['legs']==2).sum())\n"
            "    print(f'switches: {n_sw} in {len(GEM)} months = {n_sw/len(GEM)*12:.2f}/yr')\n"
            "else:\n"
            "    for c in R['costs']: print(f'cost {c[0]:>3} bps: net CAGR {c[1]:+.2f}%  Sharpe {c[2]:.2f}  active {c[3]:+.2f}%/yr (t {c[4]:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: {R['switches_yr']:.2f} switches/yr cost 3–16 bps/yr — the gross-to-"
            "net gap is a rounding error. **Costs are not why GEM lags; the signal is.** That matters "
            "for the Tradability stamp: this is a *Mirage* of the claim, not a friction story."
        ),
        md(
            "### 4f · Machinery control — the engine can find the effect when it exists\n\n"
            "Two-state bull/bear Markov world, unconditional equity premium positive in both worlds "
            "(so hiding in bonds can never win mechanically). Knob = regime persistence. Null "
            "(persistence 0): past returns carry nothing — GEM must not beat buy-and-hold. Planted "
            "(persistence 0.94, ~2.8-year regimes): 12-month momentum must light up."
        ),
        code(
            "res = []\n"
            "for label, pers in (('null (iid)', 0.0), ('planted (0.94)', 0.94)):\n"
            "    p = data.synthetic_world(persistence=pers, seed=592)\n"
            "    g = st.run_gem(p); b = st.benchmark_returns(p, g.index)\n"
            "    a = st.active_stats(g['gross'], b['SPY'])\n"
            "    pg2 = st.perf(g['gross'], b['RF']); ps2 = st.perf(b['SPY'], b['RF'])\n"
            "    res.append((label, a['ann_pct'], a['hac_t']))\n"
            "    print(f\"{label:<15} active {a['ann_pct']:+.2f}%/yr  HAC t = {a['hac_t']:+.2f}  \"\n"
            "          f\"maxDD GEM {pg2['max_dd']*100:.0f}% vs B&H {ps2['max_dd']*100:.0f}%\")\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.2))\n"
            "ax.bar([r[0] for r in res], [r[2] for r in res], color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = +2 bar'); ax.axhline(0, c='k', lw=1)\n"
            "for i, r in enumerate(res): ax.annotate(f't={r[2]:+.2f}', (i, r[2]), ha='center', va='bottom')\n"
            "ax.set_ylabel('HAC t, GEM vs buy-and-hold')\n"
            "ax.set_title('Null stays down; planted persistence lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: in a world with **no** trend the engine reads "
            f"**t = {R['syn'][0][2]:+.2f}** (it pays the whipsaw drag, exactly as theory predicts — no "
            f"false positive); plant multi-year regimes and it reads **t = {R['syn'][1][2]:+.2f}** with "
            f"the drawdown cut from {R['syn'][1][4]}% to {R['syn'][1][3]}%. The harness can bank a real "
            "trend — so the flat-to-negative real-tape verdict is the tape's fault, not the code's. "
            "*(Machinery proof only — never cited in support of a stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — *Real on the drawdown-halving:* maxDD **{R['gem'][3]:.1f}% vs "
            f"{R['spy'][3]:.1f}%** (ratio {R['dd_ratio']:.2f} < ½), the mechanical fruit of parking "
            f"19% of months in bonds. *None on the beats-buy-and-hold:* active "
            f"**{R['act_spy'][1]:+.2f}%/yr, HAC t = {R['act_spy'][2]:+.2f}**, significantly negative "
            "ex-GFC (t = −2.02) and post-2013 (t = −2.96); the entire 3/6/9/12 grid is negative; the "
            "40-seed shuffled-timing Welch t averages +0.26. No survivorship (index ETFs); the caveat "
            "is the 2002 sample start.\n"
            f"- **Tradability `MIRAGE`** — execution is as easy as it gets "
            f"({R['switches_yr']:.2f} switches/yr, 3–16 bps/yr, unlimited capacity), which is precisely "
            "why the stamp is red: the deployable product has been **beta minus 5.5%/yr since "
            f"publication**, and its risk-adjusted profile (Sharpe {R['gem'][2]:.2f}) is matched by a "
            f"no-timing 60/40 ({R['b6040'][2]:.2f}). What's tradable isn't the claim; what's claimed "
            "isn't on the tape.\n"
            f"- **Decayed since 2013? `CONFIRMED`** — pre **{R['pre_bps']:+.1f}** bps/mo "
            f"(t = {R['pre_t']:+.2f}, never significant) vs post **{R['post_bps']:+.1f}** bps/mo "
            f"(t = {R['post_t']:+.2f}); difference Welch **t = {R['decay_welch']:+.2f}**. Textbook "
            "McLean-Pontiff."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The bear-shape bet.** GEM's value is concentrated in *slow* bears (2008: 17 months "
            "peak-to-trough). Fast bears (2020: 1 month) and v-recoveries are pure whipsaw for a "
            "12-month rule. A tape with more slow bears — 1974, 2000-02 — is exactly what the book's "
            "pre-ETF backtest contains and ours cannot certify.\n"
            "- **Excess-vs-excess kept us honest.** GEM parks in bonds 19% of the time; racing raw "
            "Sharpe would gift it the cash yield. On excess Sharpe the race is a dead heat with SPY "
            "(0.70 vs 0.68) and a loss to 60/40 (0.72).\n"
            "- **Siblings.** The absolute-momentum leg *diversified across a futures panel* is "
            "[518-time-series-momentum](../../518-time-series-momentum/); the relative leg across many "
            "countries is [146-country-momentum](../../146-country-momentum/). GEM compresses both into "
            "two equity tickers — concentrating exactly the whipsaw risk diversification would have "
            "spread.\n\n"
            "*The reproducible core is offline and deterministic; the signal is Antonacci's official "
            "GEM tree with one documented execution lag. Methods and sources: "
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
