"""Generate the two narrative notebooks for Study 717 (Person-of-the-Year).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily closes
under ../_cache/ (each honoree ticker + SPY) and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere
with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily total-return
# closes; hardcoded 4-honoree Person-of-the-Year census; as-of 2026-07-12; market-model CAR,
# SPY benchmark; 12-month canonical window [+1,+252]).
R = dict(
    asof="2026-07-12", fingerprint="ff8a712ac1d1", n_used=4, n_dropped=2,
    # per-name 12m: (ticker, label, CAR_pct, runup_pct, short_gross_pct, short_net_pct, squeeze_pct, direct)
    names=[
        ("AMZN", "Bezos '99", -234.1, 49.9, 81.1, 76.0, 8.6, True),
        ("MSFT", "Gates '05", -11.7, 0.8, -13.3, -18.4, 14.0, False),
        ("TSLA", "Musk '21", -146.3, 58.4, 49.6, 44.5, 25.2, True),
        ("DJT", "Trump '24", -149.8, 106.9, 70.7, 65.6, 17.3, False),
    ],
    pooled=dict(mean_pct=-135.5, t=-2.95, curse=100, placebo_left=0.030,
                placebo_two=0.044, null_mean=-8.1),
    # horizon sweep: (label, mean_pct, t, placebo_left)
    horizon=[("1m", -1.5, -0.18, 0.477), ("3m", -31.4, -3.31, 0.062),
             ("6m", -70.3, -3.11, 0.024), ("12m", -135.5, -2.95, 0.027)],
    # leave-one-out 12m: (drop, mean_pct, t)
    loo=[("AMZN", -102.6, -2.26), ("MSFT", -176.7, -6.16),
         ("TSLA", -131.9, -2.03), ("DJT", -130.7, -2.02)],
    # the confound: corr, regression CAR = a + b*runup, residual mean/t
    confound=dict(corr=-0.583, a=-0.689, b=-1.233, resid_mean=-0.0, resid_t=-0.00),
    # short economics (averages) + worst adverse excursion
    short=dict(gross_avg=47.0, net_avg=41.9, worst_squeeze=25.2),
    # synthetic control: (planted_bps, mean_pct, t)
    syn=[(0.0, -37.6, -1.26), (-3000.0, -67.6, -2.26), (-12000.0, -157.6, -5.27)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Cover_curse%3F: Misattributed](https://img.shields.io/badge/Cover_curse%3F-Misattributed-8b949e?style=flat-square)\n\n"
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

from person_of_the_year import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES, EVENTS = data.load_real()
    PANEL = st.car_panel(PRICES, EVENTS, with_runup=True)   # canonical 12-month CAR + run-up
else:
    PRICES = EVENTS = PANEL = None
print("real price cache present:", HAVE_REAL,
      "| honorees with data:", (0 if PANEL is None else len(PANEL)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the TIME cover jinx the stock? 🗞️\n"
            "### The magazine-cover curse, tested on every business Person of the Year\n\n"
            + BADGES +
            "There's an old trading-floor superstition: the moment a company or its CEO lands a "
            "triumphant magazine cover, the top is in. And no cover is bigger than TIME's **Person of "
            "the Year**, unveiled every December. So we asked the blunt version: when TIME crowns a "
            "**business** honoree, does the stock get *cursed*?\n\n"
            "The honest answer is a great little trap. Every single business Person of the Year — Bezos "
            "in 1999, Gates in 2005, Musk in 2021, Trump in 2024 — **did** underperform the market over "
            "the next year. Four for four. The curse looks *real*. Then you notice one thing about who "
            "TIME picks, and the whole spell breaks.\n\n"
            "> 📓 **Plain-language layer.** Want the market-model math, the *t*-stats and the placebo "
            "test? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Most Persons of the Year are politicians or abstract groups "
            "(\"The Protester\", \"You\") with no stock at all. We **hardcode the transparent census** of "
            "the honorees who ran a public company — just four in 25 years — and name the ones who "
            "weren't public yet (Zuckerberg's Facebook in 2010). Every chart is drawn by the code beside "
            "it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did the honorees' stocks fall after the cover? | **Yes — all four.** One year later the "
            f"average was **{R['pooled']['mean_pct']:.0f}% below the market** (a real, significant "
            "decline). |\n"
            "| So the cover curse is real? | **Not as a *curse*.** TIME crowns people at their **peak** "
            "— the honorees had just run up 50–107%. Take out that prior run-up and the \"curse\" is "
            "**exactly zero**. |\n"
            "| Could I have shorted them and won? | **On these four, yes** (net **+42%** on average). But "
            f"**four** tradable honorees in 25 years isn't a strategy, and you'd be shorting names that "
            f"squeeze **+{R['short']['worst_squeeze']:.0f}%** against you first. |\n"
            "| Is there a *magazine* edge to trade? | **No.** What you'd be trading is plain "
            "**momentum-reversion** — extended stocks cooling off — which has nothing to do with the "
            "cover. |\n\n"
            "> The stocks really did fall. But the cover didn't *cause* it — it just **marked the "
            "peak**, the way a thermometer marks a fever without giving you one."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When they put the CEO on the cover, sell — the magazine cover curse always gets them.\"*\n\n"
            "It's repeated about basically every glamour stock, and TIME's **Person of the Year** is the "
            "ultimate cover: a mid-December coronation the whole world sees. The seduction is that it's a "
            "**clean, dated event** — you know the day, you know the direction (someone just got crowned) "
            "— so surely you can fade it. We'll take that seriously: measure each honoree's **abnormal "
            "return** (the stock's move *minus the market's*) over the year *after* the cover drops."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a famous cover reliably marked a top, that would be a free short — and something spooky "
            "about markets, since a *magazine* can't move a company's cash flows. But there's a much more "
            "boring explanation lying in wait, and telling the two apart is the whole game. A magazine "
            "gets a cover-worthy story by finding someone at their **zenith** — the stock that already "
            "tripled, the founder already on every screen. If those zenith stocks then cool off, you'll "
            "see a \"curse\" that is really just **regression to the mean** — and it was never about the "
            "cover. So the question isn't *\"did they fall?\"* (they did). It's *\"did they fall because "
            "of the cover, or because of who gets chosen?\"*"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We hardcode the **{R['n_used']} business Persons of the Year** (as-of {R['asof']}) and run "
            "a textbook **event study**:\n\n"
            "1. **Subtract the market.** For each stock, fit a line — *how it normally moves with the "
            "S&P* — over a calm window **before** the cover. The **abnormal return** is whatever it did "
            "beyond that.\n"
            "2. **Add up the year after.** Cumulate that abnormal return over the 12 months following the "
            "announcement. Negative = the \"curse.\"\n"
            "3. **Then run the one check that matters.** Line up each honoree's *post-cover* move against "
            "how much the stock had *already run up* going in. If the fallers are exactly the ones who'd "
            "risen most, the curse is just **selection** — and we'll have caught it."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the raw picture: four covers, one year later.** Each bar is a business Person of "
            "the Year's abnormal return over the following 12 months. They're all below zero — this is "
            "the chart that keeps the superstition alive."
        ),
        code(
            "if HAVE_REAL:\n"
            "    p = PANEL.sort_values('car')\n"
            "    cars = p['car'].values*100\n"
            "    labs = [f\"{t}\\n{h.split('(')[0].strip()}\" for t,h in zip(p['ticker'], p['honoree'])]\n"
            "else:\n"
            "    order = sorted(R['names'], key=lambda r: r[2])\n"
            "    cars = np.array([r[2] for r in order]); labs=[f\"{r[0]}\\n{r[1]}\" for r in order]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.8))\n"
            "ax.bar(labs, cars, color=RED, width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('12-month abnormal return vs market (%)')\n"
            "for i,v in enumerate(cars): ax.annotate(f'{v:.0f}%',(i,v),ha='center',va='top',fontsize=9)\n"
            "ax.set_title('Every business Person of the Year fell behind the market the next year')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'{len(cars)}/{len(cars)} honorees negative — average {cars.mean():.0f}% — the curse LOOKS airtight')"
        ),
        md(
            f"Four for four, averaging **{R['pooled']['mean_pct']:.0f}%** behind the market. If you "
            "stopped here you'd \"confirm\" the curse and go short the next cover. So let's not stop "
            "here — let's ask *when* the fall happened, and *who* fell."
        ),
        md(
            "**When does the fall happen?** A magazine *jinx* should hit around the cover. Here's the "
            "average abnormal return at 1, 3, 6 and 12 months out. Watch the first bar."
        ),
        code(
            "labs=[h[0] for h in R['horizon']]; means=[h[1] for h in R['horizon']]\n"
            "if HAVE_REAL:\n"
            "    means=[st.car_panel(PRICES, EVENTS, window=w)['car'].mean()*100 for w in [(1,21),(1,63),(1,126),(1,252)]]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "cols=[GREY if abs(m)<5 else RED for m in means]\n"
            "ax.bar(labs, means, color=cols, width=.6); ax.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(means): ax.annotate(f'{v:.0f}%',(i,v),ha='center',va='top',fontsize=9)\n"
            "ax.set_ylabel('average abnormal return (%)'); ax.set_xlabel('months after the cover')\n"
            "ax.set_title('Nothing in month one — the decline is a slow bleed over the year')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'1 month: {means[0]:.0f}% (basically zero). The \"curse\" only shows up over quarters.')"
        ),
        md(
            f"The first month is a **shrug** ({R['horizon'][0][1]:.0f}%). No cover-day shock at all — the "
            "decline dribbles out over quarters. That's already the wrong shape for a *jinx* and the "
            "right shape for **an extended stock slowly cooling off.** Which brings us to the one chart "
            "that ends the argument."
        ),
        md(
            "**Who fell? The ones who'd already flown.** For each honoree, the horizontal axis is how "
            "much the stock had **run up in the year before** the cover; the vertical axis is what it did "
            "**after**. If the curse were about the magazine, these would be unrelated. They are not."
        ),
        code(
            "if HAVE_REAL:\n"
            "    run = PANEL['runup'].values*100; car = PANEL['car'].values*100; tks=PANEL['ticker'].tolist()\n"
            "else:\n"
            "    run=np.array([r[3] for r in R['names']]); car=np.array([r[2] for r in R['names']]); tks=[r[0] for r in R['names']]\n"
            "b,a = np.polyfit(run, car, 1)\n"
            "xs=np.linspace(run.min()-10, run.max()+10, 50)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 5.0))\n"
            "ax.scatter(run, car, s=90, color=RED, zorder=3)\n"
            "for x,y,t in zip(run,car,tks): ax.annotate(t,(x,y),xytext=(6,6),textcoords='offset points',fontsize=9)\n"
            "ax.plot(xs, a+b*xs, color=GREY, ls='--', label=f'fit: more prior run-up -> deeper fall')\n"
            "ax.axhline(0,c='k',lw=.6); ax.axvline(0,c='k',lw=.6)\n"
            "ax.set_xlabel('prior-year run-up going INTO the cover (%)')\n"
            "ax.set_ylabel('abnormal return the year AFTER (%)')\n"
            "ax.set_title('The fall tracks the prior run-up — TIME crowned them at the peak'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'the stock crowned with NO run-up (MSFT/Gates, +1%) barely fell; the rocket-ships fell hardest')"
        ),
        md(
            f"There's the tell. The honoree with essentially **no** prior run-up — Gates' Microsoft, up "
            "0.8% — barely moved afterward. The rocket-ships (Bezos after +50%, Musk after +58%, Trump's "
            f"DJT after +107%) fell in near-lockstep with how high they'd flown (correlation "
            f"**{R['confound']['corr']:.2f}**). The magazine isn't cursing anyone. **TIME crowns stocks "
            "at their zenith, and zeniths revert.** The cover is the thermometer, not the fever."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The decline is real on the tape (**{R['pooled']['mean_pct']:.0f}%** at "
            f"one year, all four down) — but it's absent at one month, rests on **four** events, and "
            "**disappears entirely** once you account for the honorees' prior run-up. Real-looking, "
            "selection-driven.\n"
            "- **Tradability — Mirage.** The shorts happened to win big on these four names, so cost "
            "isn't the killer — but four tradable honorees in 25 years is a scrapbook, not a portfolio, "
            "and you'd be shorting the most-squeezable stocks alive.\n"
            "- **\"Cover curse?\" — Misattributed.** The stocks fell, but the cover didn't do it. It's "
            "**mean-reversion of extended winners** wearing a magazine's face."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — count the covers\n\n"
            "Forget significance. How *many* shots does this give you? Here's the timeline of tradable "
            "business Persons of the Year over a quarter-century."
        ),
        code(
            "yrs=[1999,2005,2021,2024]; labels=['AMZN/Bezos','MSFT/Gates','TSLA/Musk','DJT/Trump']\n"
            "fig, ax = plt.subplots(figsize=(9.6, 2.6))\n"
            "ax.eventplot(yrs, colors=RED, lineoffsets=0, linelengths=.8)\n"
            "for y,l in zip(yrs,labels): ax.annotate(l,(y,.45),ha='center',fontsize=8,rotation=0)\n"
            "ax.set_yticks([]); ax.set_xlim(1998, 2026); ax.set_ylim(-.6,.9)\n"
            "ax.set_title('Four tradable business honorees in 25 years — about one every six years'); ax.set_xlabel('year')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Even if each carried a clean signal, one trade every ~6 years is not a strategy.')"
        ),
        md(
            f"Roughly **one tradable business honoree every six years** — and to collect the \"curse\" "
            f"you'd hold a 12-month short on a name that can (and did — TSLA ran **+"
            f"{R['short']['worst_squeeze']:.0f}%** first) rip against you before it falls. Too **few**, "
            "too **dangerous**, and it's not even a magazine edge — just momentum you could fade "
            "diversified across hundreds of names."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🗞️\n\n"
            "- **A cleaner dated catalyst.** [Study 391 — CEO-Turnover](../391-ceo-turnover/) runs the "
            "same event-study machinery on CEO firings — there the move is priced before you can act; "
            "here the move is *selection*, not the event.\n"
            "- **The confound, on purpose.** [Study 344 — Backtest-Overfitting](../344-backtest-overfitting/) "
            "shows how an effect that clears the bar can dissolve under the one obvious control — exactly "
            "what the prior-run-up regression does here.\n"
            "- **Build your own.** Swap our four-name census for *every* business magazine cover you can "
            "date (BusinessWeek, Forbes, Fortune) — more events tighten the error bars, but the "
            "\"crowned at the peak\" selection won't go away; it's how covers get chosen.\n\n"
            "*Think the cover itself carries a curse? Show a honoree bucket landing below zero **after** "
            "you've matched them to equally-run-up stocks that never got a cover — then we'll talk.*"
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
            "# Person-of-the-Year — a quantitative cover-curse teardown 🔬\n"
            "### Market-model CARs by horizon · a placebo mid-December null · the prior-run-up "
            "regression that zeroes the residual · borrow-aware short economics · a synthetic power "
            "control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We confront "
            "a *significant* result honestly: the pooled 12-month abnormal return of the four business "
            "Persons of the Year is **−135%**, Welch **t = −2.95**, placebo **p = 0.03**, 4/4 negative "
            "— it clears the desk's |t| ≥ 2 bar. Then we show why it is nonetheless **Weak** and "
            "**Misattributed**: it is absent at one month, hostage to a four-event sample, and — "
            "decisively — its residual is **zero** once we control for the honorees' prior-year run-up. "
            "TIME selects on the zenith; zeniths mean-revert.\n\n"
            "> ⚠️ **Data + label note.** Cover-effect data would need every business cover ever printed; "
            "we use the hardcoded census of the **four** tradable business Persons of the Year (AMZN'99, "
            "MSFT'05, TSLA'21, DJT'24; the `direct`/`linked` label is the believers' framing, subjective "
            "at the margin; META'10 & DJT'16 drop for lack of a public stock — a survivorship note on "
            "the Signal axis). Real data: yfinance daily total-return closes, each ticker + SPY. Offline "
            "core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `WEAK` | Pooled 12m CAR **{R['pooled']['mean_pct']:.0f}%**, Welch "
            f"**t = {R['pooled']['t']:.2f}**, placebo **p = {R['pooled']['placebo_left']:.2f}**, "
            f"**4/4** negative — clears |t| ≥ 2 raw. But **1-month t = {R['horizon'][0][2]:.2f}** (no "
            f"event shock), n = 4, and the **residual after prior run-up is {R['confound']['resid_mean']:.0f}% "
            f"at t = {R['confound']['resid_t']:.2f}**. |\n"
            f"| **Tradability** | `MIRAGE` | Realized shorts net **+{R['short']['net_avg']:.0f}%** avg "
            "(costs aren't the killer) — but ~4 events/25yr (no capacity), worst squeeze "
            f"**+{R['short']['worst_squeeze']:.0f}%** (path-risk), and it's short-momentum **beta**. |\n"
            f"| **Cover curse?** | `MISATTRIBUTED` | corr(prior run-up, post CAR) = "
            f"**{R['confound']['corr']:.2f}**; regressing it out leaves **zero**. Zenith mean-reversion, "
            "not a magazine jinx. |\n\n"
            "> 💡 In plain words: the honorees' stocks really did underperform — but the magazine didn't "
            "cause it. TIME picks people at their peak, peaks revert, and once you subtract the part "
            "explained by how far they'd already run, there is no cover effect left to trade."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "For honoree $i$ with announcement day $0$, fit the market model "
            "$r_{i,t} = \\alpha_i + \\beta_i\\, r_{m,t} + \\varepsilon_{i,t}$ on a clean estimation "
            "window $[-125,-5]$, then the **abnormal return** is "
            "$AR_{i,t} = r_{i,t} - (\\hat\\alpha_i + \\hat\\beta_i\\, r_{m,t})$ and the post-coronation "
            "**CAR** over $[\\tau_1,\\tau_2]$ is $\\mathrm{CAR}_i = \\sum_{t=\\tau_1}^{\\tau_2} AR_{i,t}$, "
            "with $\\tau_1 = +1$ (you can act no earlier than the next close).\n\n"
            "- **H₁ (curse exists).** $\\mathbb{E}[\\mathrm{CAR}] < 0$ over the year after the cover.\n"
            "- **H₂ (it's the *cover*).** The decline survives controlling for the honoree's **prior-year "
            "run-up** $\\rho_i$ — i.e. the residual of $\\mathrm{CAR}_i = a + b\\,\\rho_i + u_i$ is still "
            "negative.\n"
            "- **H₃ (it's deployable).** A borrow-aware short of the honorees is a repeatable, "
            "survivable strategy.\n\n"
            "We find **H₁ supported raw** (pooled *t* = −2.95, placebo *p* = 0.03), **H₂ rejected** "
            "(residual mean −0.0%, *t* = −0.00), **H₃ rejected** (four events, ruinous path-risk, and "
            "the tradable object is momentum-reversion). The legend is *true as a description* (they "
            "fell) and *false as a mechanism* (the cover didn't do it)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The entire teardown is one small-sample mean and one control regression. The pooled test is\n\n"
            "$$t = \\frac{\\bar{\\mathrm{CAR}}}{s/\\sqrt{k}},\\qquad k = 4,$$\n\n"
            "and with four heavy-tailed CARs a couple-of-hundred-percent mean can look significant *or* "
            "insignificant depending on one name — so we stress it with **leave-one-out** and a "
            "**placebo null** (random mid-December windows on the same tickers). But the number that "
            "decides *interpretation* is the slope $b$ in\n\n"
            "$$\\mathrm{CAR}_i = a + b\\,\\rho_i + u_i,$$\n\n"
            "where $\\rho_i$ is the prior-year run-up. If the honorees are **selected on** $\\rho$ (a "
            "magazine crowns winners) and $\\rho$ predicts reversal, then a raw $\\bar{\\mathrm{CAR}}<0$ "
            "is *mechanically* produced by selection, and the cover-specific effect is the **residual "
            "$\\bar{u}$** — which is what we actually report."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Honoree census.** Hardcoded **{R['n_used']}** tradable business Persons of the Year "
            f"(ticker, mid-Dec announce date, `direct`/`linked`), as-of {R['asof']}, fingerprint "
            f"`{R['fingerprint']}`; {R['n_dropped']} business picks dropped for no public stock at the "
            "cover (named on the Signal axis).\n"
            "- **Market model.** $r = \\alpha + \\beta\\,r_{\\mathrm{SPY}}$ on $[-125,-5]$ (120-day "
            "estimation, 5-day gap so the cover can't leak into the fit); total-return series both legs.\n"
            "- **CAR window.** Canonical $[+1,+252]$ (12 months); sweep over 1m/3m/6m/12m.\n"
            "- **Null #1 (Welch t).** Pooled mean vs 0, plus leave-one-out.\n"
            "- **Null #2 (placebo).** Random non-event mid-December $(\\text{ticker},\\text{date})$ "
            "windows on the same names; one-sided-left $p = \\Pr[\\text{random mean} \\le \\text{observed}]$.\n"
            "- **The confound control.** Regress post-CAR on the prior-year run-up; report the residual.\n"
            "- **Tradable variant.** Short at **+1 day**, hold 12 months, pay 5%/yr borrow + 10 bps; "
            "report the worst adverse excursion (squeeze).\n"
            "- **Positive control.** A deterministic four-event panel with a **plantable drift edge**: "
            "the engine must recover a large planted curse **and** must not fake one at zero edge."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The horizon sweep — significance is a slow bleed, not an event\n\n"
            "Pooled Welch *t* by window, against the |t| = 2 bar. A magazine *jinx* would spike near the "
            "cover; instead there is **nothing** at 1 month and the *t* only builds as the window widens "
            "— the signature of drift, not a dated shock."
        ),
        code(
            "wins=[(1,21),(1,63),(1,126),(1,252)]; labs=[h[0] for h in R['horizon']]\n"
            "if HAVE_REAL:\n"
            "    ts=[st.welch_t(st.car_panel(PRICES, EVENTS, window=w)['car'].to_numpy()) for w in wins]\n"
            "else:\n"
            "    ts=[h[2] for h in R['horizon']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "cols=[GREY if abs(t)<2 else RED for t in ts]\n"
            "ax.bar(labs, ts, color=cols, width=.6)\n"
            "ax.axhline(-2, ls='--', c=RED, label='t = -2 (significance bar)'); ax.axhline(0,c='k',lw=.8)\n"
            "for i,t in enumerate(ts): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='top',fontsize=9)\n"
            "ax.set_ylabel('pooled Welch t'); ax.set_xlabel('window after the cover')\n"
            "ax.set_title('Nothing at 1 month; the t only builds over quarters'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('t by window:', {l: round(t,2) for l,t in zip(labs,ts)})"
        ),
        md(
            f"> 💡 In plain words: at 1 month the pooled t is **{R['horizon'][0][2]:.2f}** — the cover "
            f"day does nothing. The decline accretes to **{R['pooled']['t']:.2f}** only by 12 months. "
            "That's exactly what a slowly-cooling extended stock looks like, and exactly what a "
            "cover-day jinx does *not* look like."
        ),
        md(
            "### 4b · The decisive control — regress out the prior run-up\n\n"
            "Left: post-CAR vs prior-year run-up — the honorees fall in proportion to how far they'd "
            "flown. Right: the pooled decline **raw** vs the **residual** after removing run-up. The raw "
            "bar clears the bar; the residual is a flat zero."
        ),
        code(
            "if HAVE_REAL:\n"
            "    run=PANEL['runup'].to_numpy(); car=PANEL['car'].to_numpy(); tks=PANEL['ticker'].tolist()\n"
            "else:\n"
            "    run=np.array([r[3]/100 for r in R['names']]); car=np.array([r[2]/100 for r in R['names']]); tks=[r[0] for r in R['names']]\n"
            "b,a=np.polyfit(run, car, 1); resid=car-(a+b*run)\n"
            "raw_t=st.welch_t(car); res_t=st.welch_t(resid); corr=np.corrcoef(run,car)[0,1]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(11.2,4.4))\n"
            "xs=np.linspace(run.min()-.1, run.max()+.1, 50)\n"
            "a1.scatter(run*100, car*100, s=80, color=RED, zorder=3)\n"
            "for x,y,t in zip(run*100,car*100,tks): a1.annotate(t,(x,y),xytext=(5,5),textcoords='offset points',fontsize=8)\n"
            "a1.plot(xs*100,(a+b*xs)*100,ls='--',c=GREY,label=f'corr={corr:.2f}')\n"
            "a1.axhline(0,c='k',lw=.6); a1.set_xlabel('prior-year run-up (%)'); a1.set_ylabel('post-cover CAR (%)')\n"
            "a1.set_title('Post-fall tracks prior run-up'); a1.legend()\n"
            "a2.bar(['raw pooled','residual\\n(run-up out)'], [car.mean()*100, resid.mean()*100], color=[RED, GREEN], width=.55)\n"
            "a2.axhline(0,c='k',lw=.8); a2.set_ylabel('mean CAR (%)')\n"
            "a2.annotate(f't={raw_t:.2f}',(0,car.mean()*100),ha='center',va='top',fontsize=9)\n"
            "a2.annotate(f't={res_t:.2f}',(1,resid.mean()*100),ha='center',va='bottom',fontsize=9)\n"
            "a2.set_title('The cover-specific residual is ZERO')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'raw pooled {car.mean()*100:.0f}% (t={raw_t:.2f}) -> residual {resid.mean()*100:.0f}% (t={res_t:.2f})')"
        ),
        md(
            f"> 💡 In plain words: the correlation between how far a honoree had run up and how far it "
            f"fell is **{R['confound']['corr']:.2f}** — tight and negative. Fit that line and the "
            f"leftover, cover-specific piece is **{R['confound']['resid_mean']:.0f}%** at "
            f"**t = {R['confound']['resid_t']:.2f}**. The magazine explains *nothing* the prior run-up "
            "didn't already. This is the whole verdict in one regression."
        ),
        md(
            "### 4c · Placebo + leave-one-out — the raw effect is real but four-event-fragile\n\n"
            "Left: the pooled mean against random mid-December windows on the same names. Right: the "
            "12-month *t* dropping each honoree in turn. The raw decline survives both — which is *not* "
            "evidence for a cover effect, only that all four picks were extended stocks."
        ),
        code(
            "if HAVE_REAL:\n"
            "    car=PANEL['car'].to_numpy()\n"
            "    null=st.placebo_car_dist(PRICES, data.TICKERS, k=len(car), n_draws=6000, seed=717)\n"
            "    obs=car.mean(); pval=st.placebo_pvalue(obs, null, 'left')\n"
            "    tks=PANEL['ticker'].tolist(); loo=[(t, st.welch_t(np.delete(car,i))) for i,t in enumerate(tks)]\n"
            "else:\n"
            "    rng=np.random.default_rng(717); null=rng.normal(R['pooled']['null_mean']/100, .55, 6000)\n"
            "    obs=R['pooled']['mean_pct']/100; pval=R['pooled']['placebo_left']\n"
            "    loo=[(r[0], r[2]) for r in R['loo']]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(11.2,4.3))\n"
            "a1.hist(null*100, bins=50, color=GREY, alpha=.85, label='random mid-Dec windows')\n"
            "a1.axvline(obs*100, c=RED, lw=2.5, label=f'the four honorees ({obs*100:.0f}%)')\n"
            "a1.set_xlabel('pooled mean CAR (%)'); a1.set_ylabel('freq'); a1.set_title(f'Placebo p(left) = {pval:.2f}'); a1.legend()\n"
            "a2.bar([l[0] for l in loo], [l[1] for l in loo], color=AMBER, width=.6)\n"
            "a2.axhline(-2, ls='--', c=RED); a2.axhline(0,c='k',lw=.8)\n"
            "for i,(_,t) in enumerate(loo): a2.annotate(f'{t:.2f}',(i,t),ha='center',va='top',fontsize=9)\n"
            "a2.set_ylabel('pooled t, dropping this name'); a2.set_title('Leave-one-out: no single outlier')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'placebo p(left)={pval:.3f} | leave-one-out t:', {l[0]:round(l[1],2) for l in loo})"
        ),
        md(
            f"> 💡 In plain words: the four honorees sit in the **left tail** of random mid-December "
            f"windows (placebo *p* = **{R['pooled']['placebo_left']:.2f}**), and dropping any one name "
            "keeps |t| above 2. The raw decline is *not* a single-outlier artifact — every honoree fell. "
            "But robustness of the *raw* number is not a cover effect; it just says all four picks were "
            "extended names, which (per 4b) is the entire story."
        ),
        md(
            "### 4d · Tradable short + synthetic power control\n\n"
            "Left: realized short P&L per honoree (short at +1 day, hold 12m), gross vs net of 5%/yr "
            "borrow + 10 bps — costs are *not* the killer. Right: the synthetic control — with **zero** "
            "planted drift the four-event *t* stays inside ±2 (no false curse); only a large planted "
            "decline lights up."
        ),
        code(
            "if HAVE_REAL:\n"
            "    gs=[]; ns=[]; tks=[]\n"
            "    for e in EVENTS:\n"
            "        s=PRICES[e['ticker']].dropna(); pos=int(np.searchsorted(s.index, e['announce_date']))\n"
            "        path=s.values[pos+1:pos+1+252]\n"
            "        if len(path)<10: continue\n"
            "        raw12=path[-1]/s.values[pos+1]-1; nc=st.net_of_costs(raw12, borrow_ann=0.05)\n"
            "        gs.append(nc['gross_pct']); ns.append(nc['net_pct']); tks.append(e['ticker'])\n"
            "else:\n"
            "    tks=[r[0] for r in R['names']]; gs=[r[4] for r in R['names']]; ns=[r[5] for r in R['names']]\n"
            "res=[(e, st.summarize_bucket(data.synthetic_events(curse_bps=e, seed=717)['car'])['t']) for e in (0.0,-3000.0,-12000.0)]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(11.4,4.3))\n"
            "x=np.arange(len(tks))\n"
            "a1.bar(x-.2, gs, .4, color=GREEN, label='gross'); a1.bar(x+.2, ns, .4, color=AMBER, label='net @5%+10bps')\n"
            "a1.axhline(0,c='k',lw=.8); a1.set_xticks(x); a1.set_xticklabels(tks)\n"
            "a1.set_ylabel('short P&L over 12m (%)'); a1.set_title(f'Shorts won on these names (avg net +{np.mean(ns):.0f}%) — cost is not the killer'); a1.legend()\n"
            "labs=['planted 0\\n(null)','planted -3000bps','planted -12000bps\\n(~real)']; dts=[r[1] for r in res]\n"
            "a2.bar(labs, dts, color=[GREY, AMBER, RED], width=.6)\n"
            "a2.axhline(-2, ls='--', c=RED); a2.axhline(0,c='k',lw=.8)\n"
            "for i,t in enumerate(dts): a2.annotate(f't={t:.2f}',(i,t),ha='center',va='top',fontsize=8)\n"
            "a2.set_ylabel('four-event Welch t'); a2.set_title('Control: n=4 fakes no curse at zero edge')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('synthetic t:', {e:round(t,2) for e,t in res})"
        ),
        md(
            f"> 💡 In plain words: shorting these four *worked* (net **+{R['short']['net_avg']:.0f}%** "
            "avg) — so we do not pretend costs kill it. And the control is honest: with **no** planted "
            f"edge the four-event t is **{R['syn'][0][2]:.2f}** (inside ±2, no false positive); only a "
            f"planted decline near the real magnitude reaches **t = {R['syn'][2][2]:.2f}**. The machinery "
            "recovers real effects and doesn't invent them — the real decline is real, and (per 4b) it's "
            "selection."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — pooled 12m CAR **{R['pooled']['mean_pct']:.0f}%**, Welch "
            f"**t = {R['pooled']['t']:.2f}**, placebo **p = {R['pooled']['placebo_left']:.2f}**, 4/4 "
            f"negative: it clears |t| ≥ 2 raw. But it is absent at 1 month "
            f"(**t = {R['horizon'][0][2]:.2f}**), built on **four** events, and its **residual after the "
            f"prior-run-up control is {R['confound']['resid_mean']:.0f}% at t = {R['confound']['resid_t']:.2f}**. "
            "Significant raw, dissolved by the one obvious confound ⇒ WEAK, not REAL.\n"
            f"- **Tradability `MIRAGE`** — realized shorts net **+{R['short']['net_avg']:.0f}%** avg, so "
            "cost isn't the constraint; but ~4 tradable honorees in 25 years is no capacity, the worst "
            f"adverse excursion is **+{R['short']['worst_squeeze']:.0f}%** (a margin-callable squeeze), "
            "and the object is short-momentum **beta** you'd hold diversified, not one cover star at a "
            "time.\n"
            f"- **Cover curse? `MISATTRIBUTED`** — corr(prior run-up, post CAR) = "
            f"**{R['confound']['corr']:.2f}**; regress it out and the cover-specific effect is **zero**. "
            "The decline is **zenith mean-reversion**, not a magazine jinx."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — selection is the whole edge\n\n"
            "The operational truth in one picture: the honorees' raw fall (red) versus the part left "
            "after you strip out prior run-up (green). Everything tradable lives in the red bar — and the "
            "red bar *is* momentum-reversion, available diversified and cheap, not a magazine secret."
        ),
        code(
            "if HAVE_REAL:\n"
            "    run=PANEL['runup'].to_numpy(); car=PANEL['car'].to_numpy()\n"
            "    b,a=np.polyfit(run,car,1); resid=car-(a+b*run)\n"
            "    explained=car.mean()-resid.mean()\n"
            "else:\n"
            "    car=np.array([r[2]/100 for r in R['names']]); explained=car.mean()-R['confound']['resid_mean']/100; resid=np.array([R['confound']['resid_mean']/100])\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['raw fall'], [car.mean()*100], color=RED, width=.5, label='what you SEE')\n"
            "ax.bar(['explained\\nby prior run-up'], [explained*100], color=GREY, width=.5, label='selection')\n"
            "ax.bar(['cover-specific\\nresidual'], [resid.mean()*100], color=GREEN, width=.5, label='what the COVER adds')\n"
            "ax.axhline(0,c='k',lw=.8); ax.set_ylabel('mean CAR (%)')\n"
            "ax.set_title('The tradable fall is selection/momentum; the cover adds ~0'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'raw {car.mean()*100:.0f}% = selection {explained*100:.0f}% + cover-residual {resid.mean()*100:.0f}%')"
        ),
        md(
            "> 💡 In plain words: decompose the fall and it is almost entirely the piece a momentum "
            "model would have predicted from the prior run-up; the cover-specific slice is a rounding "
            "error. So even the *ex-post* profitable shorts weren't a cover trade — they were a "
            "concentrated, path-risky bet on mean-reversion of four rocket-ships. **The rarity and the "
            "zenith-selection that make the story vivid are exactly what make it untradable as a "
            "magazine effect.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The dated-catalyst mirror.** [Study 391 — CEO-Turnover](../391-ceo-turnover/): the same "
            "market-model event study on CEO firings — there the move is priced before you can act; here "
            "it's selection, not the event.\n"
            "- **The confound as the lesson.** [Study 344 — Backtest-Overfitting](../344-backtest-overfitting/) "
            "— an effect that clears *t* = 2 raw can vanish under one honest control, which is precisely "
            "the prior-run-up regression here.\n"
            "- **Better data.** Replace the four-name census with a full panel of dated business covers "
            "(BusinessWeek/Forbes/Fortune/TIME), matched to non-cover control firms of equal prior "
            "run-up; the error bars tighten and the residual can be estimated properly — but the "
            "selection-on-the-zenith that manufactures the curse will not go away.\n\n"
            "*The reproducible core is offline and deterministic; the honoree census is an explicit "
            "hardcoded, cited stand-in. Methods and sources: [`docs/references.md`](../docs/references.md); "
            "frozen numbers: [`docs/results.md`](../docs/results.md).*"
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
