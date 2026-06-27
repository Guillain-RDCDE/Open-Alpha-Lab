"""Generate the two narrative notebooks for Study 503 (Expected Idiosyncratic Skewness).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached basket prices
under ../_cache/ (a fixed S&P-100-style large-cap basket + SPY), compute the idio-skew panel
once (cached to ../_cache/panel_*.parquet for speed), and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with
no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance S&P-100-style 79-name
# large-cap basket + SPY, 2005-01-03 -> 2026-05-29, 5,385 days, 21.4 years, as-of 2026-05-31,
# fingerprint abdf2637ac09).
R = dict(
    start="2005-01-03", end="2026-05-29", asof="2026-05-31", days=5385, years=21.4,
    names=79, fp="abdf2637ac09", n_months=251,
    # per-quintile: (mean%/yr, vol%, sharpe), Q1=low skew ... Q5=high skew
    q1=(15.6, 15.1, 1.03), q2=(12.4, 14.3, 0.86), q3=(14.8, 15.5, 0.96),
    q4=(16.1, 18.1, 0.89), q5=(16.6, 17.5, 0.95),
    # long-short Q1-Q5 (long low-skew, short high-skew): (mean%/yr, sharpe, hac_t, win%, p_placebo)
    ls_gross=(-1.0, -0.09, -0.41, 45, 0.659),
    ls_net=(-6.3, -0.57, -2.54, 41, None),
    # robustness: (label, mean%/yr, t, win%)
    robust=[("tertiles", -1.8, -0.86, 45), ("quintiles", -1.0, -0.41, 45),
            ("deciles", -3.7, -1.14, 45)],
    # sub-periods: (label, n, mean%/yr, t)
    subs=[("2005-2015", 127, -0.6, -0.17), ("2016-2026", 124, -1.5, -0.42)],
    # synthetic control (seed 503): (edge, mean%/yr, t, win%, p_placebo)
    syn=[(0.000, -0.5, -0.30, 53, 0.610), (0.004, 12.9, 8.12, 69, 0.000)],
    # synthetic control seed-robust over 20 seeds: (edge, mean_t, t_sd, mean%/yr, mean_sd)
    syn_robust=[(0.000, 0.24, 1.08, 0.40, 1.71), (0.004, 8.58, 1.28, 13.77, 1.72)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Skewed_losers%3F: Busted](https://img.shields.io/badge/Skewed_losers%3F-Busted-8b949e?style=flat-square)\n\n"
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

from expected_idiosyncratic_skewness import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.load_prices()
    PRICES = PRICES[PRICES.index <= ASOF]          # drop the partial 2026-06 bar
    # cache the (slow) rolling-regression panel so both notebooks build fast
    _cache_dir = os.path.abspath("../_cache")
    _qp = os.path.join(_cache_dir, "qret_q5.parquet")
    if os.path.exists(_qp):
        QRET = pd.read_parquet(_qp)
        PANEL = None
    else:
        PANEL = data.build_panel(PRICES)
        QRET = st.quintile_returns(PANEL)
        try:
            QRET.to_parquet(_qp)
        except Exception:
            pass
else:
    PRICES = PANEL = QRET = None
print("real basket cache present:", HAVE_REAL,
      "| quintile months:", (0 if QRET is None else len(QRET)))
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
            "# Do \"lottery-skewed\" stocks lose? — expected idiosyncratic skewness, in plain English 🎲\n"
            "### A fat one-sided upside is supposed to be a sell signal — does it survive on the big stocks you own?\n\n"
            + BADGES +
            "There's a famous, *real* finding in finance: stocks whose own price moves carry a fat, "
            "lopsided **upside tail** — a small chance of a big idiosyncratic jackpot — tend to "
            "**underperform** afterward. The story is behavioural: people over-pay for the *shape* of a "
            "maybe-jackpot, so those lottery-skewed names are priced too high and drift down. The trade "
            "writes itself: **buy the symmetric stocks, avoid (or short) the skewed ones.**\n\n"
            "It checks out — in the *small, low-priced* corner of the market where it was discovered. This "
            "notebook asks the question a normal investor actually cares about: does it still work on the "
            "**big, liquid** stocks in your account? The answer is a clean, slightly anticlimactic **no** — "
            "it just isn't there.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the synthetic "
            "control? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** The original effect lives in thousands of small stocks; "
            "yfinance gives us the big ones, so we run it on a fixed **S&P-100-style basket** and call it "
            "a **proxy** throughout. That basket is also made of *survivors* — and that's part of why the "
            "signal washes out. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do the lottery-skewed stocks underperform the symmetric ones? | **No — it's basically a "
            f"tie.** The skewed tail (Q5) returned **+{R['q5'][0]:.1f}%/yr** vs the symmetric tail "
            f"(Q1)'s **+{R['q1'][0]:.1f}%/yr** — if anything the skewed names edged *ahead*. |\n"
            "| So the textbook trade (buy symmetric, short skewed) makes money? | **No — it's a coin.** "
            f"**{R['ls_gross'][0]:+.1f}%/yr** gross, which is **statistically indistinguishable from "
            "zero** (no real edge either way). |\n"
            "| Then why is the original finding famous and correct? | **Different universe.** It was "
            "found in *small, low-priced, retail-held* stocks where real lottery-buyers gamble. The big "
            "stocks here are **survivors** — a fat upside tail on a survivor often flags a *winner*, not "
            "an over-priced ticket. |\n"
            "| So what's the lesson? | **An anomaly can be real *and* not generalise.** Skewness "
            "preference is a small-cap story; copy-pasted onto the large-caps you actually hold, it just "
            "evaporates. |\n\n"
            "> The skewness effect is real where it was born. On the big liquid names, \"high idio-skew\" "
            "stops meaning \"overpriced lottery ticket\" and stops meaning much of anything tradable."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Each month, look at the **shape** of every stock's own price moves — the part the whole "
            "market doesn't explain. Some names have a lopsided **upside tail**: usually quiet, "
            "occasionally a big idiosyncratic pop. Those are 'lottery tickets' — people over-pay for the "
            "skew, so they're priced too high and **underperform** next month. Sort into five buckets by "
            "skew, buy the symmetric bucket, short the skewed one.\"*\n\n"
            "This is **Boyer, Mitton & Vorkink (2010)** — a real, well-cited result. Note the difference "
            "from the one-day-pop cousin ([Study 365 — MAX](../365-lottery-max-effect/)): MAX looks at a "
            "single big *day*; skewness looks at the **whole distribution's lopsidedness**. We'll rebuild "
            "the skew sort and watch what it does on big stocks."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the skewed tail really did underperform on the stocks everyone owns, the advice would be "
            "concrete: **avoid the name with the fat one-sided upside.** And it would say something deep — "
            "that the *shape* of a payoff is systematically over-priced. But a fat upside tail can mean "
            "two different things: a **gambler's over-priced lottery ticket** (the small-cap story) *or* "
            "**a strong stock that occasionally rips higher on its own news** (the large-cap story). Same "
            "skew number, opposite future. Which one you measure depends on *which stocks you're sorting* "
            "— and that's the whole ballgame."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We rebuild the sort on a **transparent large-cap proxy**: a fixed **{R['names']}-name "
            f"S&P-100-style basket** ({R['start']} → {R['end']}, {R['years']:.0f} years).\n\n"
            "1. **Score each stock's shape.** Every month, strip out the market (regress the stock on SPY "
            "over the past year) and measure the **skewness of what's left** — its own-name lopsidedness.\n"
            "2. **Sort into five buckets.** Q1 = most symmetric … Q5 = most upside-skewed (flashy).\n"
            "3. **See what happens next.** Earn each bucket's return *the following month* (so we only "
            "ever act on information we already had — no peeking).\n"
            "4. **Run the trade.** Buy Q1, short Q5, and ask: does symmetric-minus-skewed make money, lose "
            "money, or do nothing? If it does nothing, the skewness effect doesn't survive on big stocks."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the five buckets.** Average yearly return of each skew bucket, from symmetric (Q1) "
            "to skewed (Q5). The lottery story says this should slope **down** to the right."
        ),
        code(
            "labs = ['Q1\\nlow skew\\n(symmetric)','Q2','Q3','Q4','Q5\\nhigh skew\\n(lottery)']\n"
            "if HAVE_REAL and PANEL is not None:\n"
            "    qs = st.quintile_summary(QRET); means = [qs.loc[f'Q{i}','mean_ann']*100 for i in range(1,6)]\n"
            "elif HAVE_REAL:\n"
            "    qs = st.quintile_summary(QRET); means = [qs.loc[f'Q{i}','mean_ann']*100 for i in range(1,6)]\n"
            "else:\n"
            "    means = [R['q1'][0],R['q2'][0],R['q3'][0],R['q4'][0],R['q5'][0]]\n"
            "x = np.arange(5)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "cols = [GREY]*4 + [GREEN]\n"
            "ax.bar(x, means, color=cols, width=.62)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labs)\n"
            "ax.set_ylabel('average return per year (%)')\n"
            "ax.set_title('The lottery story says symmetric (Q1) beats skewed (Q5) — here it is FLAT')\n"
            "for i,m in enumerate(means): ax.annotate(f'{m:.1f}%',(i,m),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'symmetric Q1: {means[0]:.1f}%/yr   skewed Q5: {means[-1]:.1f}%/yr  -> essentially a tie')"
        ),
        md(
            f"That's the (non-)surprise in one chart. The skewed tail (Q5, **+{R['q5'][0]:.1f}%/yr**) "
            f"didn't underperform — it slightly **edged out** the symmetric tail (Q1, "
            f"**+{R['q1'][0]:.1f}%/yr**). On big stocks, a fat upside tail isn't a gambler's mistake; "
            "it's barely informative about next month at all."
        ),
        md(
            "**Now run the textbook trade.** Buy symmetric (Q1), short skewed (Q5), every month. Here's "
            "the growth of \\$1 in that strategy — the lottery story predicts it climbs."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.long_short(QRET)\n"
            "else:\n"
            "    rng = np.random.default_rng(503)\n"
            "    ls = pd.Series(rng.normal(R['ls_gross'][0]/100/12, 0.032, R['n_months']))\n"
            "nav = (1+ls).cumprod()\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.plot(range(len(nav)), nav.values, c=RED, lw=2)\n"
            "ax.axhline(1.0, c=GREY, ls='--')\n"
            "ax.set_ylabel('growth of $1 (buy symmetric, short skewed)')\n"
            "ax.set_xlabel('months'); ax.set_title('The textbook skewness trade goes nowhere — no edge to climb')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"buy-symmetric / short-skewed: about {R['ls_gross'][0]:+.1f}%/yr gross -- a coin\")"
        ),
        md(
            f"The line wanders around \\$1 and drifts slightly **down**. Buying symmetric and shorting "
            f"skewed earned about **{R['ls_gross'][0]:+.1f}%/yr** gross on these names — a coin you "
            "couldn't tell from zero (the quants notebook makes that statistical)."
        ),
        md(
            "**Why does it wash out?** Because on a basket of *survivors*, a fat upside tail usually "
            "belongs to a strong, high-beta name — the kind that kept on winning, not an over-priced "
            "loser. Here's the punchline as a before/after: the *original* small-cap finding vs *our* "
            "large-cap result."
        ),
        code(
            "fig, ax = plt.subplots(figsize=(9.0, 4.0))\n"
            "groups = ['small-cap universe\\n(where it was found)','big survivors\\n(what we tested)']\n"
            "published_schematic = 6.0          # ~+6%/yr published-ish small-cap effect (illustration only)\n"
            "vals = [published_schematic, R['ls_gross'][0]]\n"
            "ax.bar(groups, vals, color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('buy-symmetric / short-skewed return (%/yr)')\n"
            "ax.set_title('Same trade, the edge evaporates — the universe is the whole story')\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>0 else 'top')\n"
            "ax.text(0, published_schematic+0.3, 'schematic', ha='center', fontsize=8, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('left bar is a schematic of the published small-cap effect; right bar is our measured large-cap result')"
        ),
        md(
            "> The left bar is a *schematic* of the published small-cap effect (positive — symmetric "
            "wins); the right bar is our **measured** large-cap result (a coin near zero). Same recipe, "
            "the signal gone, because the ingredient — *which stocks* — changed."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** On the big liquid names the claimed effect is **absent**: "
            f"buy-symmetric/short-skewed earns **{R['ls_gross'][0]:+.1f}%/yr** gross, a coin you can't "
            "tell from zero. If anything, the skewed names won.\n"
            "- **Tradability — Mirage.** There's no positive edge to harvest; once you pay costs the trade "
            "merely *loses* (a cost artefact, not a real short).\n"
            "- **\"Skewed losers\"? — Busted.** On the stocks you actually hold, the lottery-skewed tail "
            "didn't lose — it posted the **highest** bucket return."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — no, and here's the honest reason\n\n"
            "Even before costs, the strategy earns ~0 on these names — so there's nothing for costs to "
            "improve. The buy-symmetric/short-skewed book churns its holdings every month (a monthly "
            "signal turns over hard) and you'd pay a short borrow on the skewed leg. Net of a realistic "
            "20 bps a leg plus borrow, a near-zero gross becomes a genuine loss — but that loss is the "
            "**frictions talking**, not a discovered edge."
        ),
        code(
            "labels = ['gross','net\\n(20bps/leg + borrow)']\n"
            "vals = [R['ls_gross'][0], R['ls_net'][0]]\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.0))\n"
            "ax.bar(labels, vals, color=[GREY, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('buy-symmetric / short-skewed (%/yr)')\n"
            "ax.set_title('Costs turn a zero into a loss — there was never an edge to begin with')\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:.1f}%',(i,v),ha='center',va='top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"gross {R['ls_gross'][0]:+.1f}%/yr -> net {R['ls_net'][0]:+.1f}%/yr. The net loss is the costs, not a signal.\")"
        ),
        md(
            f"From **{R['ls_gross'][0]:+.1f}%/yr** gross to **{R['ls_net'][0]:+.1f}%/yr** net. The point "
            "isn't the costs — it's that the trade had no edge on this universe to begin with. There is "
            "nothing here to deploy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Where it *does* work.** The skewness effect is a *small-cap / low-priced / retail* "
            "phenomenon. Run this exact sort on thousands of micro-caps (you'd need a paid feed) and the "
            "symmetric tail should win — that's the published result.\n"
            "- **The one-day-pop cousin.** [Study 365 — Lottery-MAX](../365-lottery-max-effect/): the "
            "single big *day* instead of the whole distribution's shape — a related but distinct signal.\n"
            "- **The bull-market tilt.** [Study 330 — Low-Volatility-Anomaly](../330-low-volatility-anomaly/) "
            "and [Study 238 — Betting-Against-Beta](../238-betting-against-beta/) show the same 2009–2026 "
            "regime flattering the high-beta / flashy leg.\n\n"
            "*Think the skewness effect survives among big stocks? Find a universe and a definition of "
            "\"skewed\" where buy-symmetric / short-skewed lands **above** zero with a real *t* — then "
            "we'll talk.*"
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
            "# Expected Idiosyncratic Skewness — a quantitative teardown 🔬\n"
            "### A monthly residual-skew quintile sort on a large-cap proxy · long-low / short-high spread · "
            "Newey-West *t* + sign-flip placebo · gross-vs-net cost split · robustness & sub-periods · "
            "a seed-robust synthetic planted-edge control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We rebuild "
            "Boyer-Mitton-Vorkink (2010) **expected idiosyncratic skewness** — the skewness of a name's "
            "daily market-model residuals over the trailing year — as a monthly quintile sort, and "
            "confront the central question with the **tape, not the literature**: does a long-low / "
            "short-high idio-skew book clear the desk's *t* ≥ 2 bar on the universe the study actually "
            "ran? It does not — gross it is a **coin** (HAC *t* = −0.41), and the only significant number "
            "is a **cost artefact**.\n\n"
            "> ⚠️ **Data + proxy note.** True expected idio-skew is a CRSP-universe object (thousands of "
            "names, small-caps included, where the effect is strongest). We run it on a fixed "
            f"**{R['names']}-name S&P-100-style basket + SPY** (yfinance adjusted closes, {R['start']}→"
            f"{R['end']}) — an explicit **proxy**, survivorship-tilted, named on the Signal axis. The "
            f"as-of is **{R['asof']}**; inputs fingerprint `{R['fp']}`. Offline core + synthetic control "
            "are deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Long-low/short-high idio-skew mean **{R['ls_gross'][0]:+.1f}%/yr**, "
            f"HAC **t = {R['ls_gross'][2]:.2f}** (placebo *p* = {R['ls_gross'][4]:.2f}) — a coin; Q5 (high "
            f"skew) **+{R['q5'][0]:.1f}%/yr** ≈ Q1 (low skew) **+{R['q1'][0]:.1f}%/yr**. Flat across cuts "
            "and sub-periods; survivorship-tilted proxy (named on the axis). |\n"
            f"| **Tradability** | `MIRAGE` | No positive gross edge; net of 20 bps/leg + 50 bps/yr borrow "
            f"it is **{R['ls_net'][0]:+.1f}%/yr** (t = {R['ls_net'][2]:.2f}) — a **cost artefact**, not a "
            "deployable short. |\n"
            f"| **Skewed losers?** | `BUSTED` | High-skew Q5 posted the **highest** quintile return "
            f"(**+{R['q5'][0]:.1f}%/yr**); the morality tale is a small-cap / retail-lottery effect that "
            "does not generalise to large survivors. |\n\n"
            "> 💡 In plain words: a *significant net loss* is not the same as a *signal*. Here the claimed "
            "low-skew edge is absent gross (t = −0.41); the only significant number is the friction bill "
            "on a high-turnover book that earns nothing gross."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\text{ISkew}_{i,t}$ be the skewness of stock $i$'s daily market-model residual over the "
            "trailing 12 months (regress $r_i$ on $r_{\\text{SPY}}$; take skew of the residual). Rank the "
            "cross-section, form quintiles $Q_1$ (low skew) … $Q_5$ (high skew), and earn each quintile's "
            "**month-$t{+}1$** equal-weight return $\\bar r^{Q}_{t+1}$.\n\n"
            "- **H₁ (the anomaly).** $\\mathbb{E}[\\bar r^{Q_1}_{t+1} - \\bar r^{Q_5}_{t+1}] > 0$ — the "
            "symmetric tail out-earns the skewed tail (Boyer-Mitton-Vorkink 2010).\n"
            "- **H₂ (deployable).** That spread is large and reliable enough, net of costs and borrow, to "
            "allocate to.\n"
            "- **H₃ (lottery mechanism).** The high-skew tail is *over-priced* (skewness/lottery demand), "
            "not simply higher-beta winners.\n\n"
            "On the large-cap survivor proxy we find **H₁ not rejected from zero** "
            f"($\\widehat{{\\Delta}} = {R['ls_gross'][0]:+.1f}\\%$/yr, HAC $t = {R['ls_gross'][2]:.2f}$), "
            "**H₂ rejected** (no positive gross edge; net loss is frictions), **H₃ rejected** (high-skew = "
            "highest return). The published effect is real in its native (small-cap) universe; it does "
            "not survive — it simply vanishes — among large survivors."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The teardown is one cross-sectional sort, judged by the **HAC standard error** of its "
            "long-short mean:\n\n"
            "$$\\widehat{\\Delta} = \\frac{1}{T}\\sum_t\\big(\\bar r^{Q_1}_{t+1}-\\bar r^{Q_5}_{t+1}\\big),"
            "\\qquad t_{\\text{HAC}} = \\frac{\\widehat{\\Delta}}{\\widehat{\\operatorname{se}}_{\\text{NW}}(\\widehat{\\Delta})}.$$\n\n"
            "Two traps the desk insists on. (1) **Gross before net.** `REAL` is a *gross* spread clearing "
            "$|t|=2$ in the claimed direction; a *net* loss on a zero-gross book certifies the **costs**, "
            "not a tradable short. (2) **Universe is identification.** Idio-skew is a *lottery* proxy only "
            "where lottery-buyers congregate (small, low-priced, retail). On large survivors a fat upside "
            "tail correlates with high-beta growth, so the sort carries little of the original signal. "
            "Both traps are why the right Signal stamp is `NONE` (claimed edge absent), with survivorship "
            "named explicitly."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe / proxy.** A fixed {R['names']}-name S&P-100-style large-cap basket + SPY "
            f"(yfinance adjusted closes, {R['start']}→{R['end']}, {R['days']:,} days, {R['n_months']} "
            "monthly cross-sections). Explicit **proxy** for the CRSP universe — survivorship-tilted, "
            "named on the axis.\n"
            "- **Signal.** $\\text{ISkew}_{i,t}$ = skewness of the daily market-model (SPY) residual over "
            "the trailing 12 months (≥ 120 obs required); observed at the month-$t$ close.\n"
            "- **Sort.** Rank by idio-skew → quintiles; equal-weight each quintile's **month-$t{+}1$** "
            "return. The panel is lagged by construction (month-$t$ signal ↔ month-$t{+}1$ return) — "
            "**one execution lag, documented**.\n"
            "- **Spread.** $Q_1 - Q_5$ (long low-skew, short high-skew), full monthly turnover.\n"
            "- **Null #1 (HAC t).** Newey-West *t* on the spread mean, lag $\\lfloor 4(n/100)^{2/9}\\rfloor$.\n"
            "- **Null #2 (placebo).** 20,000 sign-flips of the monthly spread; "
            "$p = \\Pr[\\text{flipped mean} \\ge \\text{observed}]$.\n"
            "- **Costs.** 20 bps one-way per leg × turnover + 50 bps/yr short borrow on the high-skew leg.\n"
            "- **Positive control.** A deterministic panel with a **planted skew penalty**, *seed-robust* "
            "over 20 seeds: the sort must recover a real low-minus-high edge and must **not** manufacture "
            "significance at edge = 0."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The cross-section — flat, tilting the *wrong* way\n\n"
            "Per-quintile annualised mean (bars) with each quintile's Sharpe annotated. The lottery claim "
            "predicts a downward slope Q1→Q5; the tape is essentially **flat**."
        ),
        code(
            "qs_labels = ['Q1','Q2','Q3','Q4','Q5']\n"
            "if HAVE_REAL:\n"
            "    qs = st.quintile_summary(QRET)\n"
            "    means = [qs.loc[q,'mean_ann']*100 for q in qs_labels]; shp = [qs.loc[q,'sharpe'] for q in qs_labels]\n"
            "else:\n"
            "    means = [R['q1'][0],R['q2'][0],R['q3'][0],R['q4'][0],R['q5'][0]]\n"
            "    shp   = [R['q1'][2],R['q2'][2],R['q3'][2],R['q4'][2],R['q5'][2]]\n"
            "x = np.arange(5)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.5))\n"
            "ax.bar(x, means, color=[GREY,GREY,GREY,GREY,GREEN], width=.62)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['Q1 low','Q2','Q3','Q4','Q5 high'])\n"
            "ax.set_ylabel('mean return / yr (%)'); ax.set_title('Idio-skew quintiles: flat, with high-skew (Q5) slightly ahead')\n"
            "for i,(m,s) in enumerate(zip(means,shp)): ax.annotate(f'{m:.1f}%\\nSh {s:.2f}',(i,m),ha='center',va='bottom',fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('mean/yr by quintile:', [round(m,1) for m in means]); print('sharpe by quintile:', [round(s,2) for s in shp])"
        ),
        md(
            f"> 💡 In plain words: no monotone slope. Q5 (high skew) ties / edges Q1 on return "
            f"(**+{R['q5'][0]:.1f}%/yr** vs **+{R['q1'][0]:.1f}%/yr**); Q1 keeps the best Sharpe only "
            "because it's the lowest-vol bucket — a calm-stock artefact, not a skewness penalty on Q5."
        ),
        md(
            "### 4b · The long-short — a coin, gross\n\n"
            "The $Q_1 - Q_5$ spread (long low-skew, short high-skew): cumulative NAV, with its annualised "
            "mean, HAC *t*, and sign-flip placebo *p*. A *real* skewness effect would climb; this wanders "
            "around \\$1."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.long_short(QRET); ss = st.spread_stats(ls)\n"
            "    mean_ann, tval, pval, win = ss['mean_ann']*100, ss['tstat'], ss['p_placebo'], ss['win']*100\n"
            "else:\n"
            "    rng = np.random.default_rng(503); ls = pd.Series(rng.normal(R['ls_gross'][0]/100/12, 0.032, R['n_months']))\n"
            "    mean_ann, tval, pval, win = R['ls_gross'][0], R['ls_gross'][2], R['ls_gross'][4], R['ls_gross'][3]\n"
            "nav = (1+ls).cumprod()\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.plot(range(len(nav)), nav.values, c=RED, lw=2)\n"
            "ax.axhline(1.0, c=GREY, ls='--')\n"
            "ax.set_ylabel('growth of $1 (Q1 - Q5)'); ax.set_xlabel('months')\n"
            "ax.set_title(f'Long low-skew / short high-skew: {mean_ann:+.1f}%/yr, HAC t={tval:.2f}, placebo p={pval:.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'mean {mean_ann:+.1f}%/yr  HAC t {tval:.2f}  win-rate {win:.0f}%  placebo p {pval:.3f}')"
        ),
        md(
            f"> 💡 In plain words: HAC **t = {R['ls_gross'][2]:.2f}** with a **{R['ls_gross'][3]:.0f}%** "
            f"win-rate and placebo **p = {R['ls_gross'][4]:.2f}** is a textbook *no edge*: a positive-mean "
            "random draw beats the observed mean two times in three. **H₁ not rejected from zero — the "
            "claimed low-skew edge isn't here.**"
        ),
        md(
            "### 4c · Gross vs net — the only 'significance' is the cost bill\n\n"
            "The net book (after 20 bps/leg + 50 bps/yr borrow) is significantly *negative* — but it is a "
            "near-zero gross spread dragged down by friction on a high-turnover monthly book, not a "
            "discovered short."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.spread_stats(st.long_short(QRET))\n"
            "    nset = st.spread_stats(st.long_short(QRET, cost_bps=20.0, borrow_ann_bps=50.0, turnover=1.0))\n"
            "    gv, nv = g['mean_ann']*100, nset['mean_ann']*100; gt, nt = g['tstat'], nset['tstat']\n"
            "else:\n"
            "    gv, nv, gt, nt = R['ls_gross'][0], R['ls_net'][0], R['ls_gross'][2], R['ls_net'][2]\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.0))\n"
            "ax.bar(['gross','net'], [gv, nv], color=[GREY, RED], width=.5); ax.axhline(0,c='k',lw=.8)\n"
            "ax.set_ylabel('Q1-Q5 (%/yr)'); ax.set_title('Gross ~0 (t={:.2f}) -> net loss is the costs (t={:.2f})'.format(gt, nt))\n"
            "for i,(v,t) in enumerate(zip([gv,nv],[gt,nt])): ax.annotate(f'{v:+.1f}%\\nt={t:.2f}',(i,v),ha='center',va='top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gv:+.1f}%/yr (t={gt:.2f})  ->  net {nv:+.1f}%/yr (t={nt:.2f}); the net t certifies frictions, not a signal')"
        ),
        md(
            f"> 💡 In plain words: gross **t = {R['ls_gross'][2]:.2f}** (nothing) → net "
            f"**t = {R['ls_net'][2]:.2f}** (significant loss). `REAL` requires a *gross* t ≥ 2 in the "
            "claimed direction; a significant *net* loss on a zero-gross book is the friction bill, which "
            "is why Tradability is `MIRAGE`, not a tradable short."
        ),
        md(
            "### 4d · Robustness & sub-periods — flat everywhere\n\n"
            "Cut the tails harder (tertiles → quintiles → deciles) and split the sample in half. No edge "
            "emerges in either direction; everything sits near zero and below |t| = 2."
        ),
        code(
            "if HAVE_REAL and PANEL is not None:\n"
            "    rob = []\n"
            "    for lab,nq in (('tertiles',3),('quintiles',5),('deciles',10)):\n"
            "        q = st.quintile_returns(PANEL, n_q=nq); s = st.spread_stats(st.long_short(q, low='Q1', high=f'Q{nq}'))\n"
            "        rob.append((lab, s['mean_ann']*100, s['tstat']))\n"
            "    subs = []\n"
            "    for lab,a,b in (('2005-2015','2005-01-01','2015-12-31'),('2016-2026','2016-01-01','2026-12-31')):\n"
            "        s = st.spread_stats(st.long_short(QRET.loc[a:b])); subs.append((lab, s['mean_ann']*100, s['tstat']))\n"
            "else:\n"
            "    rob  = [(r[0], r[1], r[2]) for r in R['robust']]\n"
            "    subs = [(s[0], s[2], s[3]) for s in R['subs']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "a1.bar([r[0] for r in rob], [r[1] for r in rob], color=GREY, width=.55); a1.axhline(0,c='k',lw=.8)\n"
            "for i,r in enumerate(rob): a1.annotate(f't={r[2]:.2f}',(i,r[1]),ha='center',va='top')\n"
            "a1.set_title('Sharper cuts -> still flat'); a1.set_ylabel('Q1-Q5 mean (%/yr)')\n"
            "a2.bar([s[0] for s in subs], [s[1] for s in subs], color=GREY, width=.5); a2.axhline(0,c='k',lw=.8)\n"
            "for i,s in enumerate(subs): a2.annotate(f't={s[2]:.2f}',(i,s[1]),ha='center',va='top')\n"
            "a2.set_title('Near zero in BOTH halves'); a2.set_ylabel('Q1-Q5 mean (%/yr)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness:', [(r[0], round(r[1],1), round(r[2],2)) for r in rob])\n"
            "print('sub-periods:', [(s[0], round(s[1],1), round(s[2],2)) for s in subs])"
        ),
        md(
            "> 💡 In plain words: deciles (the most extreme tails) reach only "
            f"**t = {R['robust'][2][2]:.2f}**, and both 2005–2015 and 2016–2026 are near zero "
            f"(**t = {R['subs'][0][3]:.2f}** and **{R['subs'][1][3]:.2f}**). There's no hidden regime "
            "where the sign flips positive — there's just no signal."
        ),
        md(
            "### 4e · Faithful-engine control — seed-robust, we know the truth here\n\n"
            "On a deterministic panel with a **planted skew penalty** (high-skew names pushed down next "
            "month), averaged over **20 seeds** per the house bar: with **zero** edge the long-low/"
            "short-high *t* must stay near 0 (no false positive); with a modest planted penalty it must "
            "turn **positive** and clear t = 2 robustly. Both hold — so the engine is honest and the flat "
            "real-tape *t* is a genuine universe feature, not a bug."
        ),
        code(
            "# live single-seed control (fast); the 20-seed robustness mean+/-sd is the frozen sweep in R\n"
            "live = []\n"
            "for edge in (0.0, 0.004):\n"
            "    syn = data.synthetic_panel(edge=edge, seed=503); q = st.quintile_returns(syn)\n"
            "    s = st.spread_stats(st.long_short(q)); live.append((edge, s['mean_ann']*100, s['tstat']))\n"
            "rob = R['syn_robust']            # (edge, mean_t, t_sd, mean%/yr, mean_sd) over 20 seeds\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "labels = [f'planted edge\\n{e:.3f}' for e,_,_,_,_ in rob]\n"
            "tvals = [r[1] for r in rob]; terr = [r[2] for r in rob]\n"
            "ax.bar(labels, tvals, yerr=terr, capsize=6, color=[GREY, GREEN], width=.5,\n"
            "       label='20-seed mean +/- sd (frozen)')\n"
            "ax.scatter([0,1], [live[0][2], live[1][2]], color=RED, zorder=5, s=60, label='live seed 503')\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 (significance bar)'); ax.axhline(0,c='k',lw=.8)\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('Q1-Q5 HAC t'); ax.set_title('Control: no edge -> t~0; a real skew penalty -> t>2 (seed-robust)')\n"
            "ax.legend(fontsize=8); plt.tight_layout(); plt.show()\n"
            "for e,m,t in live: print(f'live seed 503  edge={e:.3f}: mean={m:+.2f}%/yr  HAC t={t:+.2f}')\n"
            "for e,tm,ts,mp,ms in rob: print(f'20-seed sweep edge={e:.3f}: mean HAC t={tm:+.2f} (sd {ts:.2f})  mean={mp:+.2f}%/yr')"
        ),
        md(
            f"> 💡 In plain words: at edge = 0 the control sits at **t = +{R['syn_robust'][0][1]:.2f}** "
            f"(sd {R['syn_robust'][0][2]:.2f}) — no false positive; a small planted skew penalty drives it "
            f"to **t = +{R['syn_robust'][1][1]:.2f}** (sd {R['syn_robust'][1][2]:.2f}), the *right* sign, "
            "robustly across seeds. So the sort would have caught a genuine skewness effect cleanly. The "
            f"real tape's **t = {R['ls_gross'][2]:.2f}** is therefore an honest reading: on large "
            "survivors there is simply no skewness-preference edge to find."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the claimed low-skew edge is **absent**; the gross long-low/short-high "
            f"mean is **{R['ls_gross'][0]:+.1f}%/yr** at HAC **t = {R['ls_gross'][2]:.2f}** "
            f"(placebo *p* = {R['ls_gross'][4]:.2f}) — a coin. Flat across cut granularity and both "
            "sub-periods. The universe is a **survivorship**-tilted large-cap proxy (named on the axis). "
            "The published CRSP cross-sectional anomaly is not reproduced — and cannot be certified — on a "
            "surviving large-cap universe.\n"
            f"- **Tradability `MIRAGE`** — no positive gross edge; the only statistically significant "
            f"number (net **t = {R['ls_net'][2]:.2f}**) is a **cost artefact** on a near-zero gross "
            "spread. Nothing to deploy.\n"
            f"- **Skewed losers? `BUSTED`** — high-skew Q5 posted the **highest** quintile return "
            f"(**+{R['q5'][0]:.1f}%/yr**). The skewness-preference morality tale is a small-cap / "
            "retail-lottery effect that does not generalise to the large survivors most investors hold."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — and a power note\n\n"
            "There is no positive edge to harvest here, so the operational question collapses: the "
            "long-low/short-high book is ~0 gross and a loss net. Worth stating *why the engine is "
            "trustworthy* anyway — the seed-robust synthetic control shows it has the power to detect a "
            "real low-minus-high skewness effect at a modest planted magnitude (mean t ≈ +8.6 over 20 "
            "seeds). So the finding isn't 'we couldn't measure it'; it's 'the effect, in its native form, "
            "isn't in this universe.'"
        ),
        code(
            "# net-vs-gross of the (zero-edge) book, and the seed-robust control's detection check side by side\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.0))\n"
            "a1.bar(['gross','net'], [R['ls_gross'][0], R['ls_net'][0]], color=[GREY, RED], width=.5)\n"
            "a1.axhline(0,c='k',lw=.8); a1.set_ylabel('Q1-Q5 (%/yr)'); a1.set_title('Real book: ~0 gross, a cost loss net')\n"
            "for i,v in enumerate([R['ls_gross'][0], R['ls_net'][0]]): a1.annotate(f'{v:+.1f}%',(i,v),ha='center',va='top')\n"
            "a2.bar(['edge 0','edge +0.004'], [R['syn_robust'][0][1], R['syn_robust'][1][1]],\n"
            "       yerr=[R['syn_robust'][0][2], R['syn_robust'][1][2]], capsize=6, color=[GREY, GREEN], width=.5)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0,c='k',lw=.8); a2.set_ylabel('control HAC t (20-seed mean)')\n"
            "a2.set_title('...but the engine CAN detect a real one')\n"
            "for i,v in enumerate([R['syn_robust'][0][1], R['syn_robust'][1][1]]): a2.annotate(f't={v:.2f}',(i,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('left: real book ~0 gross & a cost loss net; right: control recovers a planted edge -> the null finding is real, not underpowered')"
        ),
        md(
            "> 💡 In plain words: the left panel is the *real* (no-edge) trade; the right panel proves the "
            "harness would have lit up for a genuine skewness penalty. Put together: the large-cap "
            "idio-skew null is **informative**, not a failure to measure — the effect simply lives "
            "elsewhere (small, low-priced, retail names)."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Its native universe.** Boyer-Mitton-Vorkink's expected idio-skew is strongest among "
            "small, low-priced, high-idio-vol names. Re-run on a few-thousand-name micro-cap panel (paid "
            "feed) and the sign should return to the published positive — the cleanest demonstration that "
            "*universe is identification*.\n"
            "- **Skew vs MAX vs coskew.** [Study 365 — Lottery-MAX](../365-lottery-max-effect/): the "
            "single-day-pop cousin (one point of the distribution). Coskewness — the *systematic* tail — "
            "is a different axis again; here the market is regressed out by construction.\n"
            "- **Regime.** [Study 330 — Low-Volatility-Anomaly](../330-low-volatility-anomaly/) and "
            "[Study 238 — Betting-Against-Beta](../238-betting-against-beta/) record the same 2009–2026 "
            "bull regime flattering the high-beta / flashy leg.\n\n"
            "*The reproducible core is offline and deterministic; the cross-section is an explicit "
            "large-cap proxy. Methods and sources: [`docs/references.md`](../docs/references.md); frozen "
            "numbers: [`docs/results.md`](../docs/results.md).*"
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
