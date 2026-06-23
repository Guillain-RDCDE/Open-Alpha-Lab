"""Generate the two narrative notebooks for Study 365 (Lottery / MAX effect).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached basket prices
under ../_cache/ (a fixed S&P-100-style large-cap basket) and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs
anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance S&P-100-style 66-name
# large-cap basket, 2005-01-03 -> 2026-05-29, 5,385 days, 21.4 years, as-of 2026-05-31,
# fingerprint 5c0c1743c8d7).
R = dict(
    start="2005-01-03", end="2026-05-29", asof="2026-05-31", days=5385, years=21.4,
    names=66, fp="5c0c1743c8d7", n_months=256,
    # per-quintile: (mean%/yr, vol%, sharpe), Q1=low MAX ... Q5=high MAX
    q1=(12.2, 12.6, 0.97), q2=(11.7, 14.4, 0.81), q3=(14.9, 16.8, 0.89),
    q4=(13.3, 20.1, 0.66), q5=(29.8, 26.3, 1.13),
    # long-short Q1-Q5 (long low-MAX, short high-MAX): (mean%/yr, sharpe, hac_t, win%, p_placebo)
    ls_gross=(-17.6, -0.78, -3.44, 42, 1.00),
    ls_net=(-22.9, -1.01, -4.48, 39, 1.00),
    # robustness: (label, mean%/yr, t, win%)
    robust=[("tertiles", -10.8, -2.69, 41), ("quintiles", -17.6, -3.44, 42),
            ("deciles", -27.2, -4.10, 39)],
    # sub-periods: (label, n, mean%/yr, t)
    subs=[("2005-2015", 132, -12.6, -1.61), ("2016-2026", 124, -22.9, -3.58)],
    # synthetic control: (edge, mean%/yr, t, win%, p_placebo)
    syn=[(0.000, -0.1, -0.06, 49, 0.520), (0.002, 5.7, 3.98, 59, 0.000)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Lottery_losers%3F: Busted](https://img.shields.io/badge/Lottery_losers%3F-Busted-8b949e?style=flat-square)\n\n"
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

from lottery_max_effect import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.load_prices()
    PRICES = PRICES[PRICES.index <= ASOF]          # drop the partial 2026-06 bar
    PANEL = data.build_panel(PRICES)
    QRET = st.quintile_returns(PANEL)
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
            "# Do \"lottery\" stocks lose? — the MAX effect, in plain English 🎰\n"
            "### The flashy one-day pop is supposed to be a sell signal — does it survive on the big stocks you own?\n\n"
            + BADGES +
            "There's a famous, *real* finding in finance: stocks that just had a big, eye-catching "
            "**one-day pop** — the kind that feels like a winning lottery ticket — tend to "
            "**underperform** afterward. The story is behavioural: people over-pay for the thrill of a "
            "maybe-jackpot, so those flashy names are priced too high and drift down. The trade writes "
            "itself: **buy the boring stocks, avoid (or short) the flashy ones.**\n\n"
            "It checks out — in the *small, illiquid* corner of the market where it was discovered. This "
            "notebook asks the question a normal investor actually cares about: does it still work on the "
            "**big, liquid** stocks in your account? The answer is a clean, surprising **no** — it "
            "literally flips.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the synthetic "
            "control? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** The original effect lives in thousands of tiny stocks; "
            "yfinance gives us the big ones, so we run it on a fixed **S&P-100-style basket** and call it "
            "a **proxy** throughout. That basket is also made of *survivors* — and, as we'll see, that's "
            "exactly why the result flips. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do the flashy (high-MAX) stocks underperform the boring ones? | **No — the opposite.** On "
            f"big stocks the flashy tail (Q5) returned **+{R['q5'][0]:.0f}%/yr** vs the boring tail "
            f"(Q1)'s **+{R['q1'][0]:.0f}%/yr**. |\n"
            "| So the textbook trade (buy boring, short flashy) makes money? | **It loses — a lot.** "
            f"**{R['ls_gross'][0]:.0f}%/yr**, and that's *statistically real* (it's reliably negative, "
            "not noise). |\n"
            "| Then why is the original finding famous and correct? | **Different universe.** It was "
            "found in *tiny, illiquid* stocks where real lottery-buyers gamble. The big stocks here are "
            "**survivors** — and a big one-day pop on a survivor usually flags a future *winner* "
            "(think the names that led the bull market), not a loser. |\n"
            "| So what's the lesson? | **An anomaly can be real *and* not generalise.** The lottery "
            "effect is a small-cap story; copy-pasted onto the large-caps you actually hold, it inverts. |\n\n"
            "> The MAX effect is real where it was born. On the big liquid names, \"high MAX\" stops "
            "meaning \"overpriced lottery ticket\" and starts meaning \"the stock that's been winning.\""
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Each month, look at every stock's single **best day** over the past month — its **MAX**. "
            "The stocks with the biggest pops are 'lottery tickets': people over-pay for the chance of "
            "another jackpot, so they're priced too high and **underperform** next month. Sort into five "
            "buckets by MAX, buy the lowest bucket, short the highest.\"*\n\n"
            "This is **Bali, Cakici & Whitelaw (2011)** — a real, well-cited result. The intuition is "
            "lovely: a flashy recent payoff is catnip for gamblers, gamblers over-pay, over-priced "
            "stocks drift down. We'll rebuild the exact sort and watch what it does on big stocks."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the flashy tail really did underperform on the stocks everyone owns, the advice would be "
            "concrete and valuable: **avoid the stock that just spiked.** And it would say something deep "
            "about markets — that the *thrill* of a payoff is systematically over-priced. But there's a "
            "trap hiding in the word \"MAX.\" A big one-day jump can mean two very different things: a "
            "**gambler's over-priced lottery ticket** (the small-cap story) *or* **a strong stock having "
            "a great day** (the large-cap story). Same number, opposite future. Which one MAX captures "
            "depends entirely on *which stocks you're sorting* — and that's the whole ballgame."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We rebuild the sort on a **transparent large-cap proxy**: a fixed **{R['names']}-name "
            f"S&P-100-style basket** ({R['start']} → {R['end']}, {R['years']:.0f} years).\n\n"
            "1. **Score each stock.** Every month, find each name's single best day that month — its MAX.\n"
            "2. **Sort into five buckets.** Q1 = lowest MAX (boring) … Q5 = highest MAX (flashy).\n"
            "3. **See what happens next.** Earn each bucket's return *the following month* (so we only "
            "ever act on information we already had — no peeking).\n"
            "4. **Run the trade.** Buy Q1, short Q5, and ask: does boring-minus-flashy make money, lose "
            "money, or do nothing? If it makes money, the lottery effect survives on big stocks. If it "
            "*loses*, the flashy names are actually the winners here."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the five buckets.** Average yearly return of each MAX bucket, from boring (Q1) to "
            "flashy (Q5). The lottery story says this should slope **down** to the right."
        ),
        code(
            "labs = ['Q1\\nlow MAX\\n(boring)','Q2','Q3','Q4','Q5\\nhigh MAX\\n(flashy)']\n"
            "if HAVE_REAL:\n"
            "    qs = st.quintile_summary(QRET); means = [qs.loc[f'Q{i}','mean_ann']*100 for i in range(1,6)]\n"
            "else:\n"
            "    means = [R['q1'][0],R['q2'][0],R['q3'][0],R['q4'][0],R['q5'][0]]\n"
            "x = np.arange(5)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "cols = [GREY]*4 + [GREEN]\n"
            "ax.bar(x, means, color=cols, width=.62)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labs)\n"
            "ax.set_ylabel('average return per year (%)')\n"
            "ax.set_title('The lottery story says boring (Q1) beats flashy (Q5) — here it is BACKWARDS')\n"
            "for i,m in enumerate(means): ax.annotate(f'{m:.0f}%',(i,m),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'boring Q1: {means[0]:.0f}%/yr   flashy Q5: {means[-1]:.0f}%/yr  -> flashy WINS')"
        ),
        md(
            f"That's the surprise in one chart. The flashy tail (Q5, **+{R['q5'][0]:.0f}%/yr**) didn't "
            f"underperform — it nearly **tripled** the boring tail (Q1, **+{R['q1'][0]:.0f}%/yr**). On big "
            "stocks, the one-day pop isn't a gambler's mistake; it's a fingerprint of the stocks that "
            "have been *winning*."
        ),
        md(
            "**Now run the textbook trade.** Buy boring (Q1), short flashy (Q5), every month. Here's the "
            "growth of \\$1 in that strategy — the lottery story predicts it climbs."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.long_short(QRET)\n"
            "else:\n"
            "    # reconstruct a representative path from the frozen mean/vol (~22%/yr vol) for the picture\n"
            "    rng = np.random.default_rng(365)\n"
            "    ls = pd.Series(rng.normal(R['ls_gross'][0]/100/12, 0.066, R['n_months']))\n"
            "nav = (1+ls).cumprod()\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.plot(range(len(nav)), nav.values, c=RED, lw=2)\n"
            "ax.axhline(1.0, c=GREY, ls='--')\n"
            "ax.set_ylabel('growth of $1 (buy boring, short flashy)')\n"
            "ax.set_xlabel('months'); ax.set_title('The textbook lottery trade bleeds — it does the opposite of the legend')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"buy-boring / short-flashy: about {R['ls_gross'][0]:.0f}%/yr -- a steady loss\")"
        ),
        md(
            f"The line goes **down**. Buying boring and shorting flashy lost about "
            f"**{R['ls_gross'][0]:.0f}%/yr** on these names — the textbook trade run in reverse. You'd "
            "have done better doing literally the opposite."
        ),
        md(
            "**Why does it flip?** Because on a basket of *survivors*, a big one-day pop usually belongs "
            "to a strong, high-momentum name — the kind that kept on winning. Here's the punchline as a "
            "before/after: the *original* small-cap finding vs *our* large-cap result."
        ),
        code(
            "fig, ax = plt.subplots(figsize=(9.0, 4.0))\n"
            "groups = ['small-cap universe\\n(where it was found)','big survivors\\n(what we tested)']\n"
            "published_schematic = 7.0          # ~+7%/yr published-ish small-cap effect (illustration only)\n"
            "vals = [published_schematic, R['ls_gross'][0]]\n"
            "ax.bar(groups, vals, color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('buy-boring / short-flashy return (%/yr)')\n"
            "ax.set_title('Same trade, opposite sign — the universe is the whole story')\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.0f}%',(i,v),ha='center',va='bottom' if v>0 else 'top')\n"
            "ax.text(0, published_schematic+0.4, 'schematic', ha='center', fontsize=8, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('left bar is a schematic of the published small-cap effect; right bar is our measured large-cap result')"
        ),
        md(
            "> The left bar is a *schematic* of the published small-cap effect (positive — boring wins); "
            "the right bar is our **measured** large-cap result (negative — flashy wins). Same recipe, "
            "opposite outcome, because the ingredient — *which stocks* — changed."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** On the big liquid names the claimed effect is **absent**, and the only "
            f"thing that's statistically real is the *wrong* sign (buy-boring/short-flashy loses "
            f"**{R['ls_gross'][0]:.0f}%/yr**). The flip is driven by **survivorship** — high-MAX = the "
            "decade's winners.\n"
            "- **Tradability — Mirage.** The only real spread here is a *losing* one; there's no "
            "deployable boring-minus-flashy edge on these stocks.\n"
            "- **\"Lottery losers\"? — Busted.** On the stocks you actually hold, the flashy tail didn't "
            "lose — it had the **highest return and the highest Sharpe**."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — no, and here's the honest reason\n\n"
            "Forget significance for a second. Even before costs, the strategy *loses* on these names — "
            "and costs only make it worse. The buy-boring/short-flashy book churns its holdings every "
            "month (a one-month signal turns over hard) and you'd pay a short borrow on the flashy leg. "
            "Net of a realistic 20 bps a leg plus borrow, the loss deepens."
        ),
        code(
            "labels = ['gross','net\\n(20bps/leg + borrow)']\n"
            "vals = [R['ls_gross'][0], R['ls_net'][0]]\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.0))\n"
            "ax.bar(labels, vals, color=[GREY, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('buy-boring / short-flashy (%/yr)')\n"
            "ax.set_title('Costs do not save a trade that is already on the wrong side')\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:.0f}%',(i,v),ha='center',va='top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"gross {R['ls_gross'][0]:.0f}%/yr -> net {R['ls_net'][0]:.0f}%/yr. The sign was never the costs' fault.\")"
        ),
        md(
            f"From **{R['ls_gross'][0]:.0f}%/yr** gross to **{R['ls_net'][0]:.0f}%/yr** net. The point "
            "isn't the costs — it's that the trade is pointing the wrong way on this universe to begin "
            "with. There is nothing here to deploy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Where it *does* work.** The lottery effect is a *small-cap / illiquid* phenomenon. Run "
            "this exact sort on a few thousand micro-caps (you'd need a paid feed) and the boring tail "
            "should win — that's the published result.\n"
            "- **The beta cousin.** [Study 238 — Betting-Against-Beta](../238-betting-against-beta/): "
            "on big stocks, sorting by MAX collapses into sorting by beta, which is its own story.\n"
            "- **The bull-market tilt.** [Study 330 — Low-Volatility-Anomaly](../330-low-volatility-anomaly/) "
            "and [Study 53 — Jackpot](../53-jackpot/) show the same 2009–2026 regime flattering the "
            "high-beta / flashy leg.\n\n"
            "*Think the lottery effect survives among big stocks? Find a universe and a definition of "
            "\"flashy\" where buy-boring / short-flashy lands **above** zero with a real *t* — then "
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
            "# The Lottery / MAX effect — a quantitative teardown 🔬\n"
            "### A monthly MAX quintile sort on a large-cap proxy · long-low / short-high spread · "
            "Newey-West *t* + sign-flip placebo · robustness & sub-periods · costs · a synthetic planted-edge control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We rebuild "
            "Bali-Cakici-Whitelaw (2011) MAX — each name's maximum daily return over the prior month — "
            "as a monthly quintile sort, and confront the central question with the **tape, not the "
            "literature**: does a long-low / short-high MAX book clear the desk's *t* ≥ 2 bar on the "
            "universe the study actually ran? It does not — it is significantly **negative**, an "
            "inversion driven by **survivorship** in a large-cap basket.\n\n"
            "> ⚠️ **Data + proxy note.** True MAX is a CRSP-universe object (thousands of names, "
            "small-caps included, where the effect is strongest). We run it on a fixed "
            f"**{R['names']}-name S&P-100-style basket** (yfinance adjusted closes, {R['start']}→"
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
            f"| **Signal** | `NONE` | Long-low/short-high MAX mean **{R['ls_gross'][0]:.1f}%/yr**, HAC "
            f"**t = {R['ls_gross'][2]:.2f}** — significant but the **wrong sign**; Q5 (high MAX) "
            f"**+{R['q5'][0]:.0f}%/yr** ≫ Q1 (low MAX) **+{R['q1'][0]:.0f}%/yr**. Survivorship-driven "
            "inversion (named on the axis). |\n"
            f"| **Tradability** | `MIRAGE` | The only real spread is a losing one; net of 20 bps/leg + "
            f"50 bps/yr borrow it is **{R['ls_net'][0]:.1f}%/yr** (t = {R['ls_net'][2]:.2f}). Nothing "
            "deployable. |\n"
            f"| **Lottery losers?** | `BUSTED` | High-MAX posted the **highest** return *and* Sharpe "
            f"({R['q5'][2]:.2f}); the morality tale is a small-cap / illiquid effect that does not "
            "generalise to large survivors. |\n\n"
            "> 💡 In plain words: a *significant* spread is not the same as *the claimed* spread. Here the "
            "claimed low-MAX edge is absent, and what's significant is its inverse — because on survivors, "
            "MAX is a momentum/beta proxy, not a lottery proxy."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\text{MAX}_{i,t}$ be stock $i$'s maximum daily return over month $t$. Rank the "
            "cross-section, form quintiles $Q_1$ (low MAX) … $Q_5$ (high MAX), and earn each quintile's "
            "**month-$t{+}1$** equal-weight return $\\bar r^{Q}_{t+1}$.\n\n"
            "- **H₁ (the anomaly).** $\\mathbb{E}[\\bar r^{Q_1}_{t+1} - \\bar r^{Q_5}_{t+1}] > 0$ — the "
            "boring tail out-earns the flashy tail (Bali-Cakici-Whitelaw 2011).\n"
            "- **H₂ (deployable).** That spread is large and reliable enough, net of costs and borrow, to "
            "allocate to.\n"
            "- **H₃ (lottery mechanism).** The high-MAX tail is *over-priced* (skewness/lottery demand), "
            "not simply higher-beta winners.\n\n"
            "On the large-cap survivor proxy we find **H₁ rejected with the wrong sign** "
            f"($\\widehat{{\\Delta}} = {R['ls_gross'][0]:.1f}\\%$/yr, HAC $t = {R['ls_gross'][2]:.2f}$), "
            "**H₂ rejected** (the only real spread is negative), **H₃ rejected** (high-MAX = highest "
            "Sharpe). The published effect is real in its native (small-cap) universe; it does not "
            "survive — it inverts — among large survivors."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The teardown is one cross-sectional sort, judged by the **HAC standard error** of its "
            "long-short mean:\n\n"
            "$$\\widehat{\\Delta} = \\frac{1}{T}\\sum_t\\big(\\bar r^{Q_1}_{t+1}-\\bar r^{Q_5}_{t+1}\\big),"
            "\\qquad t_{\\text{HAC}} = \\frac{\\widehat{\\Delta}}{\\widehat{\\operatorname{se}}_{\\text{NW}}(\\widehat{\\Delta})}.$$\n\n"
            "Two traps the desk insists on. (1) **Sign matters.** A spread that clears $|t|=2$ with the "
            "*wrong* sign refutes the claim — it does not certify it; `REAL` is earned only **in the "
            "claimed direction**. (2) **Universe is identification.** MAX is a *lottery* proxy only where "
            "lottery-buyers congregate (small, illiquid, low-priced). On survivors it correlates with "
            "short-horizon momentum and market beta, so the sort silently becomes a beta sort. Both "
            "traps are why the right Signal stamp is `NONE` (claimed edge absent), with survivorship "
            "named explicitly."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe / proxy.** A fixed {R['names']}-name S&P-100-style large-cap basket (yfinance "
            f"adjusted closes, {R['start']}→{R['end']}, {R['days']:,} days, {R['n_months']} monthly "
            "cross-sections). Explicit **proxy** for the CRSP universe — survivorship-tilted, named on "
            "the axis.\n"
            "- **Signal.** $\\text{MAX}_{i,t}$ = max daily return in month $t$ (≥ 15 trading days "
            "required); observed at the month-$t$ close.\n"
            "- **Sort.** Rank by MAX → quintiles; equal-weight each quintile's **month-$t{+}1$** return. "
            "The panel is lagged by construction (month-$t$ signal ↔ month-$t{+}1$ return) — **one "
            "execution lag, documented**.\n"
            "- **Spread.** $Q_1 - Q_5$ (long low-MAX, short high-MAX), full monthly turnover.\n"
            "- **Null #1 (HAC t).** Newey-West *t* on the spread mean, lag $\\lfloor 4(n/100)^{2/9}\\rfloor$.\n"
            "- **Null #2 (placebo).** 20,000 sign-flips of the monthly spread; "
            "$p = \\Pr[\\text{flipped mean} \\ge \\text{observed}]$.\n"
            "- **Costs.** 20 bps one-way per leg × turnover + 50 bps/yr short borrow on the high-MAX leg.\n"
            "- **Positive control.** A deterministic panel with a **planted lottery penalty**: the sort "
            "must recover a real low-minus-high edge and must **not** manufacture significance at edge = 0."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The cross-section — monotone the *wrong* way\n\n"
            "Per-quintile annualised mean (bars) with each quintile's Sharpe annotated. The lottery "
            "claim predicts a downward slope Q1→Q5; the tape slopes **up**."
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
            "ax.set_ylabel('mean return / yr (%)'); ax.set_title('MAX quintiles: high-MAX (Q5) has the highest return AND Sharpe')\n"
            "for i,(m,s) in enumerate(zip(means,shp)): ax.annotate(f'{m:.0f}%\\nSh {s:.2f}',(i,m),ha='center',va='bottom',fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('mean/yr by quintile:', [round(m,1) for m in means]); print('sharpe by quintile:', [round(s,2) for s in shp])"
        ),
        md(
            f"> 💡 In plain words: the slope runs the wrong way and the high-MAX tail wins on *both* axes "
            f"(return **+{R['q5'][0]:.0f}%/yr**, Sharpe **{R['q5'][2]:.2f}**). On survivors, the biggest "
            "one-day poppers are the momentum/beta winners — the antithesis of an over-priced lottery tail."
        ),
        md(
            "### 4b · The long-short — significantly the wrong sign\n\n"
            "The $Q_1 - Q_5$ spread (long low-MAX, short high-MAX): cumulative NAV, with its annualised "
            "mean, HAC *t*, and sign-flip placebo *p*. A *real* lottery effect would climb; this bleeds."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.long_short(QRET); ss = st.spread_stats(ls)\n"
            "    mean_ann, tval, pval, win = ss['mean_ann']*100, ss['tstat'], ss['p_placebo'], ss['win']*100\n"
            "else:\n"
            "    rng = np.random.default_rng(365); ls = pd.Series(rng.normal(R['ls_gross'][0]/100/12, 0.066, R['n_months']))\n"
            "    mean_ann, tval, pval, win = R['ls_gross'][0], R['ls_gross'][2], R['ls_gross'][4], R['ls_gross'][3]\n"
            "nav = (1+ls).cumprod()\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.plot(range(len(nav)), nav.values, c=RED, lw=2)\n"
            "ax.axhline(1.0, c=GREY, ls='--')\n"
            "ax.set_ylabel('growth of $1 (Q1 - Q5)'); ax.set_xlabel('months')\n"
            "ax.set_title(f'Long low-MAX / short high-MAX: {mean_ann:.0f}%/yr, HAC t={tval:.2f}, placebo p={pval:.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'mean {mean_ann:.1f}%/yr  HAC t {tval:.2f}  win-rate {win:.0f}%  placebo p {pval:.3f}')"
        ),
        md(
            f"> 💡 In plain words: HAC **t = {R['ls_gross'][2]:.2f}** with a **negative** mean and a "
            f"**{R['ls_gross'][3]:.0f}%** win-rate (below a coin) is a real result pointing the *opposite* "
            "way to the claim. Placebo *p* ≈ 1.00: a positive-mean random draw essentially never beats "
            "this, because the truth here is firmly negative. **H₁ rejected, wrong sign.**"
        ),
        md(
            "### 4c · Robustness & sub-periods — the inversion is structural\n\n"
            "Cut the tails harder (tertiles → quintiles → deciles) and split the sample in half. If this "
            "were noise it would wash out; instead it **strengthens** into the extremes and holds in both "
            "halves."
        ),
        code(
            "if HAVE_REAL:\n"
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
            "a1.bar([r[0] for r in rob], [r[1] for r in rob], color=RED, width=.55); a1.axhline(0,c='k',lw=.8)\n"
            "for i,r in enumerate(rob): a1.annotate(f't={r[2]:.1f}',(i,r[1]),ha='center',va='top')\n"
            "a1.set_title('Sharper cuts -> MORE negative'); a1.set_ylabel('Q1-Q5 mean (%/yr)')\n"
            "a2.bar([s[0] for s in subs], [s[1] for s in subs], color=RED, width=.5); a2.axhline(0,c='k',lw=.8)\n"
            "for i,s in enumerate(subs): a2.annotate(f't={s[2]:.1f}',(i,s[1]),ha='center',va='top')\n"
            "a2.set_title('Negative in BOTH halves'); a2.set_ylabel('Q1-Q5 mean (%/yr)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness:', [(r[0], round(r[1],1), round(r[2],2)) for r in rob])\n"
            "print('sub-periods:', [(s[0], round(s[1],1), round(s[2],2)) for s in subs])"
        ),
        md(
            "> 💡 In plain words: deciles (the most extreme tails) give the *most* negative spread "
            f"(**{R['robust'][2][1]:.0f}%/yr**, t = {R['robust'][2][2]:.1f}), and both 2005–2015 and "
            "2016–2026 are negative (stronger in the FANG-led second half). A noise fluke doesn't behave "
            "like this — the inversion is a structural feature of the survivor universe."
        ),
        md(
            "### 4d · Faithful-engine control — we know the truth here\n\n"
            "On a deterministic panel with a **planted lottery penalty** (high-MAX names pushed down next "
            "month): with **zero** edge the long-low/short-high *t* must stay near 0 (no false positive); "
            "with a modest planted penalty it must turn **positive** and clear t = 2. Both hold — so the "
            "engine is honest and the negative real-tape *t* is a genuine universe feature, not a bug."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.002):\n"
            "    syn = data.synthetic_panel(edge=edge, seed=365); q = st.quintile_returns(syn)\n"
            "    s = st.spread_stats(st.long_short(q)); res.append((edge, s['mean_ann']*100, s['tstat']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "labels = [f'planted edge\\n{e:.3f}' for e,_,_ in res]; tvals = [r[2] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 (significance bar)'); ax.axhline(0,c='k',lw=.8)\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('Q1-Q5 Welch/HAC t'); ax.set_title('Control: no edge -> t~0; a real lottery penalty -> t>2')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for e,m,t in res: print(f'planted edge={e:.3f}: mean={m:+.2f}%/yr  HAC t={t:.2f}')"
        ),
        md(
            f"> 💡 In plain words: at edge = 0 the control sits at **t = {R['syn'][0][2]:.2f}** (no false "
            f"positive); a small planted lottery penalty drives it to **t = +{R['syn'][1][2]:.2f}**, the "
            "*right* sign. So the sort would have caught a genuine MAX effect cleanly. The real tape's "
            f"**t = {R['ls_gross'][2]:.2f}** is therefore an honest reading: on large survivors there is "
            "no positive lottery edge to find — there's a negative beta/survivorship one."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the claimed low-MAX edge is **absent**; the long-low/short-high mean "
            f"is **{R['ls_gross'][0]:.1f}%/yr** at HAC **t = {R['ls_gross'][2]:.2f}** — significant in "
            "the **wrong** direction. `REAL` requires *t* ≥ 2 *in the claimed direction* on this tape; a "
            "wrong-sign result refutes, not certifies. The inversion is **survivorship** (high-MAX = the "
            "decade's winners), named on the axis.\n"
            f"- **Tradability `MIRAGE`** — the only statistically real spread is a losing one; net of "
            f"20 bps/leg + 50 bps/yr borrow it is **{R['ls_net'][0]:.1f}%/yr** "
            f"(t = {R['ls_net'][2]:.2f}). Nothing to deploy.\n"
            f"- **Lottery losers? `BUSTED`** — high-MAX posted the highest return *and* Sharpe "
            f"({R['q5'][2]:.2f}). The lottery morality tale is a small-cap / illiquid effect that does "
            "not generalise to the large survivors most investors hold."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — and a power note\n\n"
            "There is no positive edge to harvest here, so the operational question collapses: the "
            "long-low/short-high book is negative gross and worse net. Worth stating *why the engine is "
            "trustworthy* anyway — the synthetic control shows it has the power to detect a real "
            "low-minus-high effect at a modest planted magnitude. So the finding isn't 'we couldn't "
            "measure it'; it's 'the effect, in its native form, isn't in this universe.'"
        ),
        code(
            "# net-vs-gross of the (losing) book, and the control's detection check side by side\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.0))\n"
            "a1.bar(['gross','net'], [R['ls_gross'][0], R['ls_net'][0]], color=[GREY, RED], width=.5)\n"
            "a1.axhline(0,c='k',lw=.8); a1.set_ylabel('Q1-Q5 (%/yr)'); a1.set_title('The real book loses, gross and net')\n"
            "for i,v in enumerate([R['ls_gross'][0], R['ls_net'][0]]): a1.annotate(f'{v:.0f}%',(i,v),ha='center',va='top')\n"
            "a2.bar(['edge 0','edge +0.002'], [R['syn'][0][2], R['syn'][1][2]], color=[GREY, GREEN], width=.5)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0,c='k',lw=.8); a2.set_ylabel('control HAC t')\n"
            "a2.set_title('...but the engine CAN detect a real one')\n"
            "for i,v in enumerate([R['syn'][0][2], R['syn'][1][2]]): a2.annotate(f't={v:.2f}',(i,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('left: real book is negative gross & net; right: control recovers a planted edge -> the null finding is real, not underpowered')"
        ),
        md(
            "> 💡 In plain words: the left panel is the *real* (losing) trade; the right panel proves the "
            "harness would have lit up for a genuine lottery penalty. Put together: the large-cap MAX "
            "null is **informative**, not a failure to measure — the effect simply lives elsewhere "
            "(small, illiquid names), and the names you hold give you its mirror image."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Its native universe.** Bali-Cakici-Whitelaw's MAX is strongest among small, low-priced, "
            "high-idio-vol names. Re-run on a few-thousand-name micro-cap panel (paid feed) and the sign "
            "should return to the published positive — the cleanest demonstration that *universe is "
            "identification*.\n"
            "- **MAX vs beta.** [Study 238 — Betting-Against-Beta](../238-betting-against-beta/): on "
            "large-caps the MAX sort largely *is* a beta sort; orthogonalising MAX against beta is the "
            "natural next cut.\n"
            "- **Regime.** [Study 330 — Low-Volatility-Anomaly](../330-low-volatility-anomaly/) and "
            "[Study 53 — Jackpot](../53-jackpot/) record the same 2009–2026 bull regime flattering the "
            "high-beta / flashy leg.\n\n"
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
