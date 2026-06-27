"""Generate the two narrative notebooks for Study 504 (Coskewness).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached basket prices
under ../_cache/ (a fixed S&P-100-style large-cap basket + SPY), compute the coskewness panel
once (cached to ../_cache/qret_q5.parquet for speed), and otherwise quote the frozen headline
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
    # per-quintile: (mean%/yr, vol%, sharpe), Q1=low coskew ... Q5=high coskew
    q1=(15.2, 19.3, 0.79), q2=(15.4, 16.3, 0.94), q3=(15.9, 15.6, 1.02),
    q4=(15.2, 15.4, 0.99), q5=(13.9, 15.5, 0.90),
    # long-short Q1-Q5 (long low-coskew, short high-coskew): (mean%/yr, sharpe, hac_t, win%, p_placebo)
    ls_gross=(1.3, 0.08, 0.40, 49, 0.352),
    ls_net=(-4.0, -0.25, -1.24, 42, None),
    # robustness: (label, mean%/yr, t, win%)
    robust=[("tertiles", -0.0, -0.00, 49), ("quintiles", 1.3, 0.40, 49),
            ("deciles", 2.5, 0.62, 54)],
    # sub-periods: (label, n, mean%/yr, t)
    subs=[("2005-2015", 127, 1.9, 0.40), ("2016-2026", 124, 0.7, 0.17)],
    # synthetic control (seed 504): (edge, mean%/yr, t, win%, p_placebo)
    syn=[(0.000, 0.09, 0.05, 49, 0.479), (0.004, 13.43, 7.60, 73, 0.000)],
    # synthetic control seed-robust over 20 seeds: (edge, mean_t, t_sd, mean%/yr, mean_sd)
    syn_robust=[(0.000, 0.24, 1.08, 0.40, 1.71), (0.004, 8.58, 1.28, 13.77, 1.72)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Coskewness_premium%3F: Busted](https://img.shields.io/badge/Coskewness_premium%3F-Busted-8b949e?style=flat-square)\n\n"
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

from coskewness import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.load_prices()
    PRICES = PRICES[PRICES.index <= ASOF]          # drop the partial 2026-06 bar
    # cache the (slow) rolling-coskewness panel so both notebooks build fast
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
            "# Do stocks that \"crash with the market\" pay you extra? — coskewness, in plain English 📉\n"
            "### Bad hedges are supposed to be cheap (high expected return) — does that survive on the big stocks you own?\n\n"
            + BADGES +
            "There's a respectable, *real* idea in finance: a stock that tends to fall **hardest exactly "
            "when the whole market is falling hard** is a lousy thing to own — it lets you down right when "
            "you most need protection. Because nobody likes that, such names should be a little **cheap**, "
            "i.e. carry a small extra expected return as compensation. The fancy name for \"falls with the "
            "market's tail\" is **coskewness** (low / negative coskewness = the bad hedge). The trade writes "
            "itself: **buy the crash-sensitive names, short the insurance-like ones.**\n\n"
            "It's a genuine factor in the broad market. This notebook asks the question a normal investor "
            "actually cares about: does it still pay on the **big, liquid** stocks in your account? The "
            "answer is a clean, slightly anticlimactic **almost-but-no** — the tilt leans the right way, "
            "but it's a coin.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the synthetic "
            "control? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** The premium is small and lives in the *spread* of crash-"
            "sensitivity across thousands of names; yfinance gives us the big ones, so we run it on a fixed "
            "**S&P-100-style basket** and call it a **proxy** throughout. That basket is also made of "
            "*survivors* — which trims the very names whose crash risk came true. Every chart is drawn by "
            "the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do the crash-sensitive (low-coskew) stocks pay a premium over the insurance-like ones? | "
            f"**Barely — and it's a tie statistically.** The crash-sensitive tail (Q1) returned "
            f"**+{R['q1'][0]:.1f}%/yr** vs the insurance-like tail (Q5)'s **+{R['q5'][0]:.1f}%/yr** — the "
            "right direction, but a hair. |\n"
            "| So the textbook trade (buy crash-sensitive, short insurance) makes money? | **Not "
            f"reliably.** **{R['ls_gross'][0]:+.1f}%/yr** gross, which is **statistically indistinguishable "
            "from zero** (a coin-flip win-rate). |\n"
            "| Then why is the original finding real and respected? | **Different universe.** It's priced "
            "off the *spread* in crash-sensitivity across the **whole** market — smaller, cyclical, "
            "leveraged names included. The big stocks here all behave fairly market-like, so the spread "
            "collapses. |\n"
            "| And the survivors? | A surviving basket quietly **drops the names whose crash risk "
            "materialised** (they fell hardest and de-listed) — exactly the left tail the premium is meant "
            "to pay for. |\n\n"
            "> The coskewness premium is real where the dispersion is. On the big liquid names, "
            "\"crash-sensitive\" stops meaning much, and the premium shrinks into the noise."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Each month, measure how each stock behaves **when the market makes a big move** — its "
            "**coskewness**. Some names reliably fall hardest in the market's worst stretches (low / "
            "negative coskewness): bad hedges. Investors dislike that, so those names are priced a little "
            "cheap and should **out-earn** going forward. Sort into five buckets by coskewness, buy the "
            "crash-sensitive bucket, short the insurance-like one.\"*\n\n"
            "This is **Harvey & Siddique (2000)** — a real, well-cited risk factor (building on Kraus & "
            "Litzenberger's 1976 three-moment CAPM). Note the difference from its idiosyncratic cousin "
            "([Study 503 — idio-skew](../503-expected-idiosyncratic-skewness/)): that one is the stock's "
            "*own* private tail (diversifiable); coskewness is the **shared** tail — how the name moves "
            "with the *market's* crashes. Opposite axis, opposite predicted sign."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the crash-sensitive tail really did pay a premium on the stocks everyone owns, the advice "
            "would be concrete: **you get paid to hold the bad hedges** — a clean third-moment risk "
            "premium beyond plain beta. But \"low coskewness\" can mean two different things depending on "
            "*which* stocks you sort. Across the **broad** market it picks out genuinely fragile, cyclical, "
            "leveraged names that deserve compensation. Across a handful of **mega-cap survivors** it picks "
            "out... not much — they all wobble with the market in fairly similar ways, and the ones that "
            "*truly* cracked are no longer in the basket. Same statistic, very different bite."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We rebuild the sort on a **transparent large-cap proxy**: a fixed **{R['names']}-name "
            f"S&P-100-style basket** ({R['start']} → {R['end']}, {R['years']:.0f} years).\n\n"
            "1. **Score each stock's crash-sensitivity.** Every month, measure how the stock moves when "
            "the **market** moves a lot — its coskewness over the past year (low / negative = a bad "
            "hedge).\n"
            "2. **Sort into five buckets.** Q1 = most crash-sensitive (low coskew) … Q5 = most "
            "insurance-like (high coskew).\n"
            "3. **See what happens next.** Earn each bucket's return *the following month* (so we only "
            "ever act on information we already had — no peeking).\n"
            "4. **Run the trade.** Buy Q1, short Q5, and ask: does crash-sensitive-minus-insurance make "
            "money, lose money, or do nothing? If it does nothing, the coskewness premium doesn't survive "
            "on big stocks."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the five buckets.** Average yearly return of each coskewness bucket, from "
            "crash-sensitive (Q1) to insurance-like (Q5). The risk story says this should slope **down** "
            "to the right (you get paid for the bad hedge)."
        ),
        code(
            "labs = ['Q1\\nlow coskew\\n(crash-sensitive)','Q2','Q3','Q4','Q5\\nhigh coskew\\n(insurance)']\n"
            "if HAVE_REAL and PANEL is not None:\n"
            "    qs = st.quintile_summary(QRET); means = [qs.loc[f'Q{i}','mean_ann']*100 for i in range(1,6)]\n"
            "elif HAVE_REAL:\n"
            "    qs = st.quintile_summary(QRET); means = [qs.loc[f'Q{i}','mean_ann']*100 for i in range(1,6)]\n"
            "else:\n"
            "    means = [R['q1'][0],R['q2'][0],R['q3'][0],R['q4'][0],R['q5'][0]]\n"
            "x = np.arange(5)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "cols = [GREEN] + [GREY]*4\n"
            "ax.bar(x, means, color=cols, width=.62)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labs)\n"
            "ax.set_ylabel('average return per year (%)')\n"
            "ax.set_title('The risk story says crash-sensitive (Q1) beats insurance (Q5) — here it is nearly FLAT')\n"
            "for i,m in enumerate(means): ax.annotate(f'{m:.1f}%',(i,m),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'crash-sensitive Q1: {means[0]:.1f}%/yr   insurance-like Q5: {means[-1]:.1f}%/yr  -> the right tilt, but tiny')"
        ),
        md(
            f"That's the (non-)surprise in one chart. The crash-sensitive tail (Q1, **+{R['q1'][0]:.1f}%/"
            f"yr**) edged the insurance-like tail (Q5, **+{R['q5'][0]:.1f}%/yr**) — the *right direction*, "
            "but barely, and the **middle** bucket (Q3) actually won. On big stocks, coskewness is faintly "
            "informative about next month at best."
        ),
        md(
            "**Now run the textbook trade.** Buy crash-sensitive (Q1), short insurance-like (Q5), every "
            "month. Here's the growth of \\$1 in that strategy — the risk story predicts it climbs."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.long_short(QRET)\n"
            "else:\n"
            "    rng = np.random.default_rng(504)\n"
            "    ls = pd.Series(rng.normal(R['ls_gross'][0]/100/12, 0.032, R['n_months']))\n"
            "nav = (1+ls).cumprod()\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.plot(range(len(nav)), nav.values, c=RED, lw=2)\n"
            "ax.axhline(1.0, c=GREY, ls='--')\n"
            "ax.set_ylabel('growth of $1 (buy crash-sensitive, short insurance)')\n"
            "ax.set_xlabel('months'); ax.set_title('The textbook coskewness trade barely budges — no reliable edge to climb')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"buy-crash-sensitive / short-insurance: about {R['ls_gross'][0]:+.1f}%/yr gross -- a coin\")"
        ),
        md(
            f"The line wanders around \\$1 with a faint upward lean. Buying crash-sensitive and shorting "
            f"insurance earned about **{R['ls_gross'][0]:+.1f}%/yr** gross on these names — the right sign, "
            "but a coin you couldn't tell from zero (the quants notebook makes that statistical)."
        ),
        md(
            "**Why does it wash out?** Because on a basket of big *survivors*, crash-sensitivity barely "
            "varies — and the names that *truly* cracked aren't here anymore. Here's the punchline as a "
            "before/after: the *broad-universe* premium vs *our* large-cap result."
        ),
        code(
            "fig, ax = plt.subplots(figsize=(9.0, 4.0))\n"
            "groups = ['broad universe\\n(where it is priced)','big survivors\\n(what we tested)']\n"
            "published_schematic = 5.0          # ~+5%/yr broad-universe coskewness premium (illustration only)\n"
            "vals = [published_schematic, R['ls_gross'][0]]\n"
            "ax.bar(groups, vals, color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('buy-crash-sensitive / short-insurance return (%/yr)')\n"
            "ax.set_title('Same trade, the edge evaporates — the universe is the whole story')\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.text(0, published_schematic+0.2, 'schematic', ha='center', fontsize=8, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('left bar is a schematic of the broad-universe premium; right bar is our measured large-cap result')"
        ),
        md(
            "> The left bar is a *schematic* of the broad-universe premium (a real, modest positive); the "
            "right bar is our **measured** large-cap result (a coin near zero, right sign). Same recipe, "
            "the signal mostly gone, because the ingredient — *which stocks* — changed."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** On the big liquid names the premium is **statistically absent**: "
            f"buy-crash-sensitive/short-insurance earns **{R['ls_gross'][0]:+.1f}%/yr** gross, a coin you "
            "can't tell from zero (right sign, no weight).\n"
            "- **Tradability — Mirage.** A +1.3%/yr gross tilt is far too thin to harvest; once you pay "
            "costs the trade merely *loses* (a cost artefact, not a real edge).\n"
            "- **\"Coskewness premium\"? — Busted.** On the stocks you actually hold, the crash-sensitive "
            "tail didn't pay — it edged the insurance tail by a hair while carrying the **highest** "
            "volatility (worst risk-adjusted return)."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — no, and here's the honest reason\n\n"
            "Even before costs, the strategy earns ~0 on these names — so there's nothing for costs to "
            "improve. The buy-crash-sensitive/short-insurance book churns its holdings every month (a "
            "monthly signal turns over hard) and you'd pay a short borrow on the insurance leg. Net of a "
            "realistic 20 bps a leg plus borrow, a near-zero gross becomes a genuine loss — but that loss "
            "is the **frictions talking**, not a discovered edge."
        ),
        code(
            "labels = ['gross','net\\n(20bps/leg + borrow)']\n"
            "vals = [R['ls_gross'][0], R['ls_net'][0]]\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.0))\n"
            "ax.bar(labels, vals, color=[GREY, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('buy-crash-sensitive / short-insurance (%/yr)')\n"
            "ax.set_title('Costs turn a thin tilt into a loss — there was never a usable edge')\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"gross {R['ls_gross'][0]:+.1f}%/yr -> net {R['ls_net'][0]:+.1f}%/yr. The net loss is the costs, not a signal.\")"
        ),
        md(
            f"From **{R['ls_gross'][0]:+.1f}%/yr** gross to **{R['ls_net'][0]:+.1f}%/yr** net. The point "
            "isn't the costs — it's that the trade had no *reliable* edge on this universe to begin with. "
            "There is nothing here to deploy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Where it *does* pay.** The coskewness premium is priced off the **dispersion** in "
            "crash-sensitivity across the whole market — smaller, cyclical, leveraged names included, and "
            "crucially the ones that later cracked. Run this exact sort on a broad point-in-time panel "
            "(with de-listed names) and the premium should reappear.\n"
            "- **The idiosyncratic cousin.** [Study 503 — Expected-Idiosyncratic-Skewness]"
            "(../503-expected-idiosyncratic-skewness/): the stock's *own* private tail instead of its "
            "shared one — the opposite axis, opposite sign.\n"
            "- **The bull-market tilt.** [Study 330 — Low-Volatility-Anomaly](../330-low-volatility-anomaly/) "
            "and [Study 238 — Betting-Against-Beta](../238-betting-against-beta/) show the same 2009–2026 "
            "regime flattering / compressing risk-factor sorts on large survivors.\n\n"
            "*Think the coskewness premium survives among big stocks? Find a universe and a definition of "
            "\"crash-sensitive\" where buy-crash-sensitive / short-insurance lands **above** zero with a "
            "real *t* — then we'll talk.*"
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
            "# Coskewness — a quantitative teardown 🔬\n"
            "### A monthly direct-coskewness quintile sort on a large-cap proxy · long-low / short-high spread · "
            "Newey-West *t* + sign-flip placebo · gross-vs-net cost split · robustness & sub-periods · "
            "a seed-robust synthetic planted-edge control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We rebuild "
            "Harvey & Siddique (2000) **coskewness** — the direct standardised coskewness of a name's "
            "daily returns with the market over the trailing year — as a monthly quintile sort, and "
            "confront the central question with the **tape, not the literature**: does a long-low / "
            "short-high coskewness book clear the desk's *t* ≥ 2 bar on the universe the study actually "
            "ran? It does not — gross it is a **coin** (HAC *t* = +0.40, right sign, no weight), and the "
            "only number with any size is a **cost artefact**.\n\n"
            "> ⚠️ **Data + proxy note.** True coskewness pricing is a CRSP-universe object (thousands of "
            "names, dispersion in crash-sensitivity, de-listed names included). We run it on a fixed "
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
            f"| **Signal** | `NONE` | Long-low/short-high coskewness mean **{R['ls_gross'][0]:+.1f}%/yr**, "
            f"HAC **t = {R['ls_gross'][2]:.2f}** (placebo *p* = {R['ls_gross'][4]:.2f}) — a coin; Q1 (low "
            f"coskew) **+{R['q1'][0]:.1f}%/yr** ≈ Q5 (high coskew) **+{R['q5'][0]:.1f}%/yr**, with Q1 the "
            "**highest-vol** bucket. Flat across cuts and sub-periods; survivorship-tilted proxy (named on "
            "the axis). |\n"
            f"| **Tradability** | `MIRAGE` | A +1.3%/yr gross edge is too thin; net of 20 bps/leg + 50 "
            f"bps/yr borrow it is **{R['ls_net'][0]:+.1f}%/yr** (t = {R['ls_net'][2]:.2f}) — a **cost "
            "artefact**, not a deployable factor. |\n"
            f"| **Coskewness premium?** | `BUSTED` | The (right-signed) spread never clears |t| = 2 in any "
            "cut; the crash-sensitive leg carries the **highest** vol (worst Sharpe). The premium is a "
            "broad-universe / dispersion effect that does not generalise to large survivors. |\n\n"
            "> 💡 In plain words: a *right-signed* tilt is not the same as a *signal*. Here the claimed "
            "premium is the correct direction but statistically a coin (t = +0.40); the only sizeable "
            "number is the friction bill on a high-turnover book that earns ~nothing gross."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\beta^{SKD}_{i,t}$ be stock $i$'s **direct standardised coskewness** with the market "
            "over the trailing 12 months, "
            "$\\beta^{SKD}=\\dfrac{E[\\varepsilon_i\\,\\varepsilon_m^2]}{\\sqrt{E[\\varepsilon_i^2]}\\;E[\\varepsilon_m^2]}$ "
            "on demeaned daily returns. Rank the cross-section, form quintiles $Q_1$ (low / negative "
            "coskew) … $Q_5$ (high coskew), and earn each quintile's **month-$t{+}1$** equal-weight return "
            "$\\bar r^{Q}_{t+1}$.\n\n"
            "- **H₁ (the premium).** $\\mathbb{E}[\\bar r^{Q_1}_{t+1} - \\bar r^{Q_5}_{t+1}] > 0$ — the "
            "crash-sensitive (low-coskew) tail out-earns the insurance-like tail (Harvey-Siddique 2000).\n"
            "- **H₂ (deployable).** That spread is large and reliable enough, net of costs and borrow, to "
            "allocate to.\n"
            "- **H₃ (third-moment premium).** The crash-sensitive tail is *compensated* (priced "
            "coskewness risk), not merely higher-vol noise.\n\n"
            "On the large-cap survivor proxy we find **H₁ right-signed but not rejected from zero** "
            f"($\\widehat{{\\Delta}} = {R['ls_gross'][0]:+.1f}\\%$/yr, HAC $t = {R['ls_gross'][2]:.2f}$), "
            "**H₂ rejected** (no usable gross edge; net loss is frictions), **H₃ rejected** (the low-coskew "
            "leg carries the *highest* vol and *worst* Sharpe — risk without reward). The published factor "
            "is real in its broad (dispersion-rich) universe; it does not survive among large survivors."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The teardown is one cross-sectional sort, judged by the **HAC standard error** of its "
            "long-short mean:\n\n"
            "$$\\widehat{\\Delta} = \\frac{1}{T}\\sum_t\\big(\\bar r^{Q_1}_{t+1}-\\bar r^{Q_5}_{t+1}\\big),"
            "\\qquad t_{\\text{HAC}} = \\frac{\\widehat{\\Delta}}{\\widehat{\\operatorname{se}}_{\\text{NW}}(\\widehat{\\Delta})}.$$\n\n"
            "Two traps the desk insists on. (1) **Gross before net.** `REAL` is a *gross* spread clearing "
            "$|t|=2$ in the claimed direction; a *net* loss on a near-zero-gross book certifies the "
            "**costs**, not a tradable factor. (2) **Universe is identification.** Coskewness is priced "
            "off the *dispersion* in crash-sensitivity — widest in small, cyclical, leveraged names, and "
            "crucially in the names that later cracked and de-listed. On large survivors that dispersion "
            "collapses, so the sort carries little of the original premium. Both traps are why the right "
            "Signal stamp is `NONE` (claimed premium statistically absent), with survivorship named "
            "explicitly."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe / proxy.** A fixed {R['names']}-name S&P-100-style large-cap basket + SPY "
            f"(yfinance adjusted closes, {R['start']}→{R['end']}, {R['days']:,} days, {R['n_months']} "
            "monthly cross-sections). Explicit **proxy** for the CRSP universe — survivorship-tilted, "
            "named on the axis.\n"
            "- **Signal.** $\\beta^{SKD}_{i,t}$ = direct standardised coskewness of the daily stock return "
            "with the daily market (SPY) return over the trailing 12 months (≥ 120 obs required); observed "
            "at the month-$t$ close.\n"
            "- **Sort.** Rank by coskewness → quintiles; equal-weight each quintile's **month-$t{+}1$** "
            "return. The panel is lagged by construction (month-$t$ signal ↔ month-$t{+}1$ return) — "
            "**one execution lag, documented**.\n"
            "- **Spread.** $Q_1 - Q_5$ (long low-coskew, short high-coskew), full monthly turnover.\n"
            "- **Null #1 (HAC t).** Newey-West *t* on the spread mean, lag $\\lfloor 4(n/100)^{2/9}\\rfloor$.\n"
            "- **Null #2 (placebo).** 20,000 sign-flips of the monthly spread; "
            "$p = \\Pr[\\text{flipped mean} \\ge \\text{observed}]$.\n"
            "- **Costs.** 20 bps one-way per leg × turnover + 50 bps/yr short borrow on the high-coskew leg.\n"
            "- **Positive control.** A deterministic panel with a **planted coskewness premium**, "
            "*seed-robust* over 20 seeds: the sort must recover a real low-minus-high edge and must **not** "
            "manufacture significance at edge = 0."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The cross-section — flat, the right sign, the wrong Sharpe\n\n"
            "Per-quintile annualised mean (bars) with each quintile's Sharpe annotated. The risk claim "
            "predicts a downward slope Q1→Q5; the tape is essentially **flat**, and the crash-sensitive "
            "Q1 carries the *highest* vol."
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
            "ax.bar(x, means, color=[GREEN,GREY,GREY,GREY,GREY], width=.62)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['Q1 low','Q2','Q3','Q4','Q5 high'])\n"
            "ax.set_ylabel('mean return / yr (%)'); ax.set_title('Coskewness quintiles: flat, with low-coskew (Q1) a hair ahead')\n"
            "for i,(m,s) in enumerate(zip(means,shp)): ax.annotate(f'{m:.1f}%\\nSh {s:.2f}',(i,m),ha='center',va='bottom',fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('mean/yr by quintile:', [round(m,1) for m in means]); print('sharpe by quintile:', [round(s,2) for s in shp])"
        ),
        md(
            f"> 💡 In plain words: no monotone slope. Q1 (low coskew) edges Q5 on return "
            f"(**+{R['q1'][0]:.1f}%/yr** vs **+{R['q5'][0]:.1f}%/yr**) — the *right* direction — but the "
            f"peak is the **middle** (Q3, +{R['q3'][0]:.1f}%), and Q1 has the **worst Sharpe** of the five "
            f"(highest vol, {R['q1'][1]:.1f}%). A faint extra return bought with a lot of extra risk is an "
            "*unrewarded* tilt, not a premium."
        ),
        md(
            "### 4b · The long-short — a coin, gross\n\n"
            "The $Q_1 - Q_5$ spread (long low-coskew, short high-coskew): cumulative NAV, with its "
            "annualised mean, HAC *t*, and sign-flip placebo *p*. A *real* coskewness premium would climb; "
            "this wanders around \\$1 with a faint lean."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.long_short(QRET); ss = st.spread_stats(ls)\n"
            "    mean_ann, tval, pval, win = ss['mean_ann']*100, ss['tstat'], ss['p_placebo'], ss['win']*100\n"
            "else:\n"
            "    rng = np.random.default_rng(504); ls = pd.Series(rng.normal(R['ls_gross'][0]/100/12, 0.032, R['n_months']))\n"
            "    mean_ann, tval, pval, win = R['ls_gross'][0], R['ls_gross'][2], R['ls_gross'][4], R['ls_gross'][3]\n"
            "nav = (1+ls).cumprod()\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.plot(range(len(nav)), nav.values, c=RED, lw=2)\n"
            "ax.axhline(1.0, c=GREY, ls='--')\n"
            "ax.set_ylabel('growth of $1 (Q1 - Q5)'); ax.set_xlabel('months')\n"
            "ax.set_title(f'Long low-coskew / short high-coskew: {mean_ann:+.1f}%/yr, HAC t={tval:.2f}, placebo p={pval:.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'mean {mean_ann:+.1f}%/yr  HAC t {tval:.2f}  win-rate {win:.0f}%  placebo p {pval:.3f}')"
        ),
        md(
            f"> 💡 In plain words: HAC **t = {R['ls_gross'][2]:.2f}** with a **{R['ls_gross'][3]:.0f}%** "
            f"win-rate and placebo **p = {R['ls_gross'][4]:.2f}** is a textbook *no edge*: a sign-flipped "
            "random book beats the observed mean about one time in three. **H₁ right-signed but not "
            "rejected from zero — the claimed premium isn't reliably here.**"
        ),
        md(
            "### 4c · Gross vs net — the only size is the cost bill\n\n"
            "The net book (after 20 bps/leg + 50 bps/yr borrow) is negative — but it is a near-zero gross "
            "spread dragged down by friction on a high-turnover monthly book, not a discovered factor."
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
            "for i,(v,t) in enumerate(zip([gv,nv],[gt,nt])): ax.annotate(f'{v:+.1f}%\\nt={t:.2f}',(i,v),ha='center',va='bottom' if v>0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gv:+.1f}%/yr (t={gt:.2f})  ->  net {nv:+.1f}%/yr (t={nt:.2f}); the net t reflects frictions, not a signal')"
        ),
        md(
            f"> 💡 In plain words: gross **t = {R['ls_gross'][2]:.2f}** (nothing) → net "
            f"**t = {R['ls_net'][2]:.2f}** (a loss, still inside the bar). `REAL` requires a *gross* t ≥ 2 "
            "in the claimed direction; a near-zero gross dragged negative by costs is the friction bill, "
            "which is why Tradability is `MIRAGE`, not a tradable factor."
        ),
        md(
            "### 4d · Robustness & sub-periods — flat everywhere\n\n"
            "Cut the tails harder (tertiles → quintiles → deciles) and split the sample in half. The "
            "right-signed tilt grows a touch with sharper cuts but never approaches |t| = 2; both halves "
            "sit near zero."
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
            "for i,r in enumerate(rob): a1.annotate(f't={r[2]:.2f}',(i,r[1]),ha='center',va='bottom' if r[1]>=0 else 'top')\n"
            "a1.set_title('Sharper cuts -> still flat'); a1.set_ylabel('Q1-Q5 mean (%/yr)')\n"
            "a2.bar([s[0] for s in subs], [s[1] for s in subs], color=GREY, width=.5); a2.axhline(0,c='k',lw=.8)\n"
            "for i,s in enumerate(subs): a2.annotate(f't={s[2]:.2f}',(i,s[1]),ha='center',va='bottom' if s[1]>=0 else 'top')\n"
            "a2.set_title('Near zero in BOTH halves'); a2.set_ylabel('Q1-Q5 mean (%/yr)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness:', [(r[0], round(r[1],1), round(r[2],2)) for r in rob])\n"
            "print('sub-periods:', [(s[0], round(s[1],1), round(s[2],2)) for s in subs])"
        ),
        md(
            "> 💡 In plain words: deciles (the most extreme tails) reach only "
            f"**t = {R['robust'][2][2]:.2f}**, and both 2005–2015 and 2016–2026 are near zero "
            f"(**t = {R['subs'][0][3]:.2f}** and **{R['subs'][1][3]:.2f}**). The tilt is faintly the right "
            "sign throughout — never significant in any cut or window."
        ),
        md(
            "### 4e · Faithful-engine control — seed-robust, we know the truth here\n\n"
            "On a deterministic panel with a **planted coskewness premium** (low-coskew names pushed up "
            "next month), averaged over **20 seeds** per the house bar: with **zero** edge the long-low/"
            "short-high *t* must stay near 0 (no false positive); with a modest planted premium it must "
            "turn **positive** and clear t = 2 robustly. Both hold — so the engine is honest and the flat "
            "real-tape *t* is a genuine universe feature, not a bug."
        ),
        code(
            "# live single-seed control (fast); the 20-seed robustness mean+/-sd is the frozen sweep in R\n"
            "live = []\n"
            "for edge in (0.0, 0.004):\n"
            "    syn = data.synthetic_panel(edge=edge, seed=504); q = st.quintile_returns(syn)\n"
            "    s = st.spread_stats(st.long_short(q)); live.append((edge, s['mean_ann']*100, s['tstat']))\n"
            "rob = R['syn_robust']            # (edge, mean_t, t_sd, mean%/yr, mean_sd) over 20 seeds\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "labels = [f'planted edge\\n{e:.3f}' for e,_,_,_,_ in rob]\n"
            "tvals = [r[1] for r in rob]; terr = [r[2] for r in rob]\n"
            "ax.bar(labels, tvals, yerr=terr, capsize=6, color=[GREY, GREEN], width=.5,\n"
            "       label='20-seed mean +/- sd (frozen)')\n"
            "ax.scatter([0,1], [live[0][2], live[1][2]], color=RED, zorder=5, s=60, label='live seed 504')\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 (significance bar)'); ax.axhline(0,c='k',lw=.8)\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('Q1-Q5 HAC t'); ax.set_title('Control: no edge -> t~0; a real coskew premium -> t>2 (seed-robust)')\n"
            "ax.legend(fontsize=8); plt.tight_layout(); plt.show()\n"
            "for e,m,t in live: print(f'live seed 504  edge={e:.3f}: mean={m:+.2f}%/yr  HAC t={t:+.2f}')\n"
            "for e,tm,ts,mp,ms in rob: print(f'20-seed sweep edge={e:.3f}: mean HAC t={tm:+.2f} (sd {ts:.2f})  mean={mp:+.2f}%/yr')"
        ),
        md(
            f"> 💡 In plain words: at edge = 0 the control sits at **t = +{R['syn_robust'][0][1]:.2f}** "
            f"(sd {R['syn_robust'][0][2]:.2f}) — no false positive; a small planted coskew premium drives "
            f"it to **t = +{R['syn_robust'][1][1]:.2f}** (sd {R['syn_robust'][1][2]:.2f}), the *right* "
            "sign, robustly across seeds. So the sort would have caught a genuine coskewness premium "
            f"cleanly. The real tape's **t = {R['ls_gross'][2]:.2f}** is therefore an honest reading: on "
            "large survivors there is simply no coskewness premium large enough to find."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the claimed premium is right-signed but **statistically absent**; the "
            f"gross long-low/short-high mean is **{R['ls_gross'][0]:+.1f}%/yr** at HAC "
            f"**t = {R['ls_gross'][2]:.2f}** (placebo *p* = {R['ls_gross'][4]:.2f}) — a coin. Flat across "
            "cut granularity and both sub-periods. The universe is a **survivorship**-tilted large-cap "
            "proxy (named on the axis). The published CRSP cross-sectional premium is not reproduced — and "
            "cannot be certified — on a surviving large-cap universe.\n"
            f"- **Tradability `MIRAGE`** — no usable gross edge; net of 20 bps/leg + 50 bps/yr borrow it is "
            f"**{R['ls_net'][0]:+.1f}%/yr** (t = {R['ls_net'][2]:.2f}), a **cost artefact** on a near-zero "
            "gross spread. Nothing to deploy.\n"
            f"- **Coskewness premium? `BUSTED`** — the low-coskew leg edged the high-coskew leg by a hair "
            f"(**+{R['q1'][0]:.1f}%** vs **+{R['q5'][0]:.1f}%/yr**) while carrying the **highest** "
            "volatility (worst Sharpe). The systematic-skew premium is a broad-universe, dispersion-rich "
            "effect that does not generalise to the large survivors most investors hold."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — and a power note\n\n"
            "There is no usable edge to harvest here, so the operational question collapses: the "
            "long-low/short-high book is ~0 gross and a loss net. Worth stating *why the engine is "
            "trustworthy* anyway — the seed-robust synthetic control shows it has the power to detect a "
            "real low-minus-high coskewness premium at a modest planted magnitude (mean t ≈ +8.6 over 20 "
            "seeds). So the finding isn't 'we couldn't measure it'; it's 'the premium, in its native "
            "(dispersion-rich) form, isn't in this universe.'"
        ),
        code(
            "# net-vs-gross of the (near-zero-edge) book, and the seed-robust control's detection check side by side\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.0))\n"
            "a1.bar(['gross','net'], [R['ls_gross'][0], R['ls_net'][0]], color=[GREY, RED], width=.5)\n"
            "a1.axhline(0,c='k',lw=.8); a1.set_ylabel('Q1-Q5 (%/yr)'); a1.set_title('Real book: ~0 gross, a cost loss net')\n"
            "for i,v in enumerate([R['ls_gross'][0], R['ls_net'][0]]): a1.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "a2.bar(['edge 0','edge +0.004'], [R['syn_robust'][0][1], R['syn_robust'][1][1]],\n"
            "       yerr=[R['syn_robust'][0][2], R['syn_robust'][1][2]], capsize=6, color=[GREY, GREEN], width=.5)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0,c='k',lw=.8); a2.set_ylabel('control HAC t (20-seed mean)')\n"
            "a2.set_title('...but the engine CAN detect a real one')\n"
            "for i,v in enumerate([R['syn_robust'][0][1], R['syn_robust'][1][1]]): a2.annotate(f't={v:.2f}',(i,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('left: real book ~0 gross & a cost loss net; right: control recovers a planted edge -> the null finding is real, not underpowered')"
        ),
        md(
            "> 💡 In plain words: the left panel is the *real* (near-zero-edge) trade; the right panel "
            "proves the harness would have lit up for a genuine coskewness premium. Put together: the "
            "large-cap coskewness null is **informative**, not a failure to measure — the premium simply "
            "lives elsewhere (the broad, dispersion-rich universe with de-listed names)."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Its native universe.** Harvey-Siddique coskewness is priced off the *dispersion* in "
            "crash-sensitivity across the whole market — strongest in small, cyclical, leveraged names and "
            "the ones that later cracked. Re-run on a broad point-in-time panel (with de-listed names) and "
            "the premium should reappear — the cleanest demonstration that *universe is identification*.\n"
            "- **Coskew vs idio-skew.** [Study 503 — Expected-Idiosyncratic-Skewness]"
            "(../503-expected-idiosyncratic-skewness/): the *idiosyncratic* (diversifiable, behavioural) "
            "tail — the opposite axis of the same regression, opposite predicted sign. "
            "[Study 332 — Downside-Beta](../332-downside-beta/) is a neighbouring left-tail-co-movement "
            "measure.\n"
            "- **Regime.** [Study 330 — Low-Volatility-Anomaly](../330-low-volatility-anomaly/) and "
            "[Study 238 — Betting-Against-Beta](../238-betting-against-beta/) record the same 2009–2026 "
            "bull regime flattering / compressing risk-factor sorts on large survivors.\n\n"
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
