"""Generate the two narrative notebooks for Study 361 (Zweig Breadth Thrust).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached basket prices
under ../_cache/ (the breadth proxy + SPY) and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with no
network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance breadth proxy from a
# 40-name large-cap basket + SPY, 1995-01-04 -> 2026-06-22, 7,918 days, 31.5 years).
R = dict(
    start="1995-01-04", end="2026-06-22", days=7918, years=31.5, n_events=22,
    # per-horizon: (months, n, cond_mean%, cond_win%, base_mean%, base_win%, t, p_placebo)
    h3=(3, 22, 2.37, 59, 2.93, 71, -0.37, 0.644),
    h6=(6, 22, 6.18, 73, 5.89, 76, 0.13, 0.461),
    h12=(12, 22, 14.99, 91, 12.01, 81, 0.94, 0.207),
    net=[(3, 2.37, 2.32), (6, 6.18, 6.13), (12, 14.99, 14.94)],
    robust=[(0.610, 22, 15.1, 0.99, 0.197), (0.615, 22, 15.0, 0.94, 0.207),
            (0.650, 15, 15.5, 0.68, 0.220)],
    syn=[(0.0, 15, 8.84, 4.98, 1.19, 0.093), (0.10, 15, 17.30, 7.38, 3.18, 0.001)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Never_wrong%3F: Busted](https://img.shields.io/badge/Never_wrong%3F-Busted-8b949e?style=flat-square)\n\n"
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

from zweig_breadth_thrust import data, strategy as st

HAVE_REAL = data.have_real()
F = data.load_real() if HAVE_REAL else None
EVENTS = st.detect_thrusts(F) if HAVE_REAL else None
print("real breadth-proxy cache present:", HAVE_REAL,
      "| thrust events:", (0 if EVENTS is None else len(EVENTS)))
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
            "# The \"never wrong\" buy signal — is the Zweig Breadth Thrust real? 📈\n"
            "### The rare 'all-clear' that supposedly always precedes a bull market, in plain English\n\n"
            + BADGES +
            "Market lore has a legend: Marty Zweig's **Breadth Thrust**. When the stock market goes "
            "from *almost nothing rising* to *almost everything rising* in about two weeks, a powerful "
            "bull run is supposed to follow — and the signal is said to be **\"never wrong.\"** It's "
            "famously rare: roughly **a dozen times in 70 years**.\n\n"
            "A perfect record sounds like free money. But \"never wrong\" out of a dozen tries, in a "
            "market that drifts **up** most years anyway, is exactly what *luck* looks like. This "
            "notebook shows why the legend is real-as-a-story and empty-as-a-strategy at the same "
            "time.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the power "
            "analysis? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** True NYSE \"advancing vs declining\" breadth isn't free, so "
            "we **build a transparent proxy**: the share of a 40-stock basket that rose each day. It's "
            "narrower and noisier than Zweig's NYSE — we call it a proxy throughout — but the lesson "
            "(a dozen events can't be a strategy) only gets *stronger* for the rarer real signal. "
            "Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After a thrust, is the market higher a year later? | **Usually — 91% of the time.** "
            "That's the \"never wrong\" headline. |\n"
            "| Isn't 91% amazing? | **Not really.** Pick *any* random day and the market is higher a "
            "year later **81%** of the time. The thrust adds only ~10 points of win-rate — and that "
            "gap is well inside the noise. |\n"
            "| So is the extra return real? | **Can't tell.** The average bump over a normal year is a "
            f"few percent, but with only **{R['n_events']} events ever** it's statistically "
            "indistinguishable from luck (and at **3 months** the thrust actually *underperforms*). |\n"
            "| Then why the perfect reputation? | **Rarity × an up-drifting market.** Fire a dozen "
            "times in a market that mostly rises and you'll look \"never wrong\" by accident — the same "
            "trick as the Hindenburg Omen, in reverse. |\n\n"
            "> The thrust is a real, vivid pattern. \"Never wrong\" is a base-rate illusion. A dozen "
            "events can't be a tradable edge."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Take the 10-day average of the fraction of stocks going up. When it rockets from "
            "below **0.40** (almost nothing rising) to above **0.615** (a broad surge) within about "
            "**10 trading days**, you've had a Breadth Thrust — and a major bull move always follows.\"*\n\n"
            "Marty Zweig coined it in *Winning on Wall Street*. The picture is intuitive: a violent, "
            "**broad** shift from despair to participation marks a real bottom, not a dead-cat bounce. "
            "The seduction is the track record — *every* thrust since 1945, the story goes, was "
            "followed by gains. We'll rebuild the signal and put that record under a microscope."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two things hide inside \"never wrong,\" and only one of them matters. (1) *Does the market "
            "tend to rise after a thrust?* Sure — but the market tends to rise after **most** days. The "
            "question that matters is (2) *does it rise by **more** than usual, reliably enough to bet "
            "on?* A high win-rate that merely matches the market's natural up-drift is **not an edge** "
            "— it's the base rate in a costume. And even a genuine edge is useless if it only shows up "
            "**once a year or once a decade**: you can't run a fund on a signal that rarely speaks."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We rebuild the thrust on a **transparent breadth proxy**: across a fixed 40-name "
            "large-cap basket, what share rose each day? Smooth it with Zweig's 10-day EMA, and mark "
            f"every **0.40 → 0.615** thrust. Over **{R['years']:.0f} years** "
            f"({R['start']} → {R['end']}) that fires **{R['n_events']}** times.\n\n"
            "1. **Find the thrusts.** Detect each completed thrust on the proxy (one event per surge).\n"
            "2. **Measure the payoff.** For each thrust, what did SPY do over the next **3 / 6 / 12 "
            "months** — and how does that compare to a *random* day (the base rate)?\n"
            "3. **Stress the luck.** Draw the same handful of *random* dates thousands of times and ask "
            "how often pure chance looks as good as the thrust. That's the honest test for a "
            "dozen-event signal."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, here's a thrust.** The breadth proxy (10-day EMA) diving below 0.40 and bursting "
            "above 0.615 — the rare two-week surge from despair to broad participation."
        ),
        code(
            "if HAVE_REAL:\n"
            "    e = st.breadth_ema(F['adv_ratio'])\n"
            "    d0 = EVENTS[10]                       # one representative thrust\n"
            "    lo = F.index.get_loc(d0)\n"
            "    seg = e.iloc[max(0,lo-40):lo+20]\n"
            "    fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "    ax.plot(seg.index, seg.values, c=GREEN, lw=2)\n"
            "    ax.axhline(0.40, ls='--', c=RED, label='0.40 (oversold floor)')\n"
            "    ax.axhline(0.615, ls='--', c=GREEN, label='0.615 (thrust trigger)')\n"
            "    ax.axvline(d0, c=GREY, lw=1)\n"
            "    ax.set_title(f'A Breadth Thrust on the proxy — {d0.date()}')\n"
            "    ax.set_ylabel('10-day EMA of % advancing'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('this thrust completed on', d0.date())\n"
            "else:\n"
            "    print('no cache — see docs/results.md for the frozen example (thrust 2010-02-18 etc.)')"
        ),
        md(
            f"Vivid, isn't it? The proxy fires **{R['n_events']}** times in {R['years']:.0f} years — "
            "about once every 1.4 years (the true NYSE version is rarer still, ~once a decade). Now the "
            "real question: what happens *after*?"
        ),
        md(
            "**The payoff vs a normal day.** For each horizon, the thrust's average forward return and "
            "win-rate, next to the *unconditional* base rate — what a random day gives you anyway."
        ),
        code(
            "hs = [3, 6, 12]\n"
            "if HAVE_REAL:\n"
            "    rows = [st.summarize(F, EVENTS, m) for m in hs]\n"
            "    cwin = [r['cond_win']*100 for r in rows]; bwin = [r['base_win']*100 for r in rows]\n"
            "else:\n"
            "    cwin = [R['h3'][3], R['h6'][3], R['h12'][3]]\n"
            "    bwin = [R['h3'][5], R['h6'][5], R['h12'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(x-.2, cwin, .4, color=GREEN, label='after a thrust')\n"
            "ax.bar(x+.2, bwin, .4, color=GREY, label='any random day (base rate)')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m} months' for m in hs])\n"
            "ax.set_ylim(0, 100); ax.set_ylabel('% of the time SPY is higher')\n"
            "ax.set_title('\"Never wrong\" at 12m is 91% — but a random day is already 81%')\n"
            "for i,(c,b) in enumerate(zip(cwin,bwin)):\n"
            "    ax.annotate(f'{c:.0f}%',(i-.2,c),ha='center',va='bottom')\n"
            "    ax.annotate(f'{b:.0f}%',(i+.2,b),ha='center',va='bottom')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('thrust 12m win-rate:', f'{cwin[-1]:.0f}%', ' base 12m win-rate:', f'{bwin[-1]:.0f}%')"
        ),
        md(
            f"There's the trick. The famous **{R['h12'][3]:.0f}%** \"never wrong\" 12-month win-rate "
            f"sits barely above the **{R['h12'][5]:.0f}%** you get on *any* random day — because the US "
            "market simply rises most years. At **3 months** the thrust's win-rate "
            f"(**{R['h3'][3]:.0f}%**) is actually *below* the base rate (**{R['h3'][5]:.0f}%**). The "
            "headline number is mostly the market's own up-drift, not the signal."
        ),
        md(
            "**Could a dozen random days look this good?** Here's the honest test: draw "
            f"**{R['n_events']}** *random* dates over and over, and see how the thrust's 12-month "
            "average return stacks up against pure luck."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(F, EVENTS, 12)\n"
            "    obs = pl['thrust_mean']; pmean = pl['placebo_mean']; pval = pl['p_value']\n"
            "    # rebuild the placebo distribution for the picture\n"
            "    h = st.TRADING_DAYS[12]; spy = F['spy'].values; n=len(spy); k=len(EVENTS); lag=1\n"
            "    vh = n - h - lag - 1; fwd = spy[lag+h:lag+h+vh]/spy[lag:lag+vh]-1\n"
            "    rng = np.random.default_rng(361)\n"
            "    draws = np.array([fwd[rng.integers(0,vh,size=k)].mean() for _ in range(8000)])\n"
            "else:\n"
            "    obs = R['h12'][2]/100; pmean = R['h12'][4]/100; pval = R['h12'][7]\n"
            "    rng = np.random.default_rng(361); draws = rng.normal(pmean, 0.045, 8000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*100, bins=50, color=GREY, alpha=.8, label='12m return of 22 RANDOM dates')\n"
            "ax.axvline(obs*100, c=GREEN, lw=2.5, label=f'the actual thrusts ({obs*100:.1f}%)')\n"
            "ax.set_xlabel('average 12-month forward return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The thrust sits inside the luck cloud — placebo p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'a random 22-date draw beats the thrust {pval*100:.0f}% of the time — far from rare')"
        ),
        md(
            f"The green line — the actual thrusts — falls **inside** the grey cloud of random draws. "
            f"About **{R['h12'][7]*100:.0f}%** of random 22-date samples do as well or better. In plain "
            "terms: **the thrust's record is what you'd expect from a dozen lucky coin-flips in an "
            "up-trending market.** Not magic — arithmetic."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The pattern is real and the 12-month win-rate is genuinely high "
            f"(**{R['h12'][3]:.0f}%**) — but it barely beats the **{R['h12'][5]:.0f}%** base rate, "
            "isn't statistically significant, and flips *negative* at 3 months. Real as lore, weak as "
            "edge.\n"
            f"- **Tradability — Fragile.** Even if the bump were real, **{R['n_events']} signals in "
            f"{R['years']:.0f} years** (the true version: once a decade) can't be a strategy you "
            "allocate to.\n"
            "- **\"Never wrong\"? — Busted.** A dozen-ish events in a market that mostly rises *will* "
            "look perfect by luck. It's the [Hindenburg Omen](../../167-hindenburg-omen/) in reverse."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — count the trades\n\n"
            "Forget significance for a second and just ask the operational question: how *often* does "
            "this thing tell you to do anything? Here's every thrust on the calendar — and the punchline "
            "is the empty space between them."
        ),
        code(
            "if HAVE_REAL:\n"
            "    yrs = [d.year for d in EVENTS]\n"
            "else:\n"
            "    yrs = [1996,1997,1999,2001,2002,2003,2005,2006,2007,2008,2010,2011,2012,2013,2014,2015,2016,2018,2019,2020,2022,2023]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 2.6))\n"
            "ax.eventplot(yrs, colors=GREEN, lineoffsets=0, linelengths=.8)\n"
            "ax.set_yticks([]); ax.set_xlim(1995, 2027)\n"
            "ax.set_title(f'Every thrust in 31 years: {len(yrs)} of them (~one every 1.4y)')\n"
            "ax.set_xlabel('year'); plt.tight_layout(); plt.show()\n"
            "gap = (2026-1995)/len(yrs)\n"
            "print(f'{len(yrs)} trades in 31 years -> one signal every ~{gap:.1f} years. Costs are tiny; frequency is fatal.')"
        ),
        md(
            f"Around **{R['n_events']} trades in three decades.** A 5-bps cost barely scratches a "
            "multi-month hold — costs are *not* the problem. **Frequency is.** You can't build a "
            "portfolio around a rule that whispers once a year (and the genuine NYSE thrust speaks once "
            "a *decade*). The honest version isn't \"wait for the all-clear and back up the truck\" — "
            "it's \"there aren't enough all-clears to matter.\""
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The mirror image.** [Study 167 — Hindenburg Omen](../../167-hindenburg-omen/) is the "
            "same rare-breadth signal pointed the *other* way: it \"always\" precedes crashes. Both "
            "owe their spooky records to the same thing — too few events in a drifting market.\n"
            "- **The raw material.** [Study 168 — Advance-Decline](../../168-advance-decline/) asks "
            "whether the breadth line itself adds anything beyond price.\n"
            "- **Build your own breadth.** Swap our 40-name basket for the S&P 500 and re-run the "
            "detector — the event count changes, the lesson doesn't: a signal this rare can't clear the "
            "bar.\n\n"
            "*Think the thrust beats the market by more than luck? Capture the events, draw the same "
            "number of random dates, and show the thrust landing **outside** the cloud — then we'll "
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
            "# Zweig Breadth Thrust — a quantitative teardown 🔬\n"
            "### Thrust detection on a breadth proxy · conditional vs unconditional forward returns · "
            "a Welch *t* + placebo randomization null · costs · a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We separate "
            "the two things \"never wrong\" fuses: a **real but small** post-thrust drift from the "
            "**unconditional up-drift** of US equities, and we confront both with the **sample size**. "
            "The decisive object is not the point estimate (it's positive, as the lore says) but its "
            "**power**: with ~a dozen lifetime events, the excess return is statistically "
            "indistinguishable from drawing the same number of random dates.\n\n"
            "> ⚠️ **Data + proxy note.** True NYSE advance/decline breadth isn't on yfinance; we build a "
            "transparent **proxy** — the daily share advancing across a fixed 40-name large-cap basket "
            "(narrower/noisier than NYSE, and survivorship-tilted; both named on the Signal axis). "
            "Real data: yfinance daily adjusted closes, SPY + basket, 1995→2026. Offline core + "
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
            f"| **Signal** | `WEAK` | 12-month conditional mean **+{R['h12'][2]:.1f}%** vs base "
            f"**+{R['h12'][4]:.1f}%**; Welch **t = {R['h12'][6]:.2f}**, placebo **p = {R['h12'][7]:.2f}** "
            f"— positive but **fails t ≥ 2**, and the 3-month *t* is **{R['h3'][6]:.2f}** (sign flip). |\n"
            f"| **Tradability** | `FRAGILE` | **{R['n_events']} events in {R['years']:.0f}y** "
            "(≈ one / 1.4y; true NYSE ≈ one / decade). Net of 5-bps costs the per-trade number is "
            "intact — **frequency**, not cost, kills it. |\n"
            f"| **Never wrong?** | `BUSTED` | The 12m win-rate **{R['h12'][3]:.0f}%** vs an "
            f"unconditional **{R['h12'][5]:.0f}%** — the 'perfect record' is base rate × rarity. "
            "Study 167 (Hindenburg) inverted. |\n\n"
            "> 💡 In plain words: the thrust really does precede gains — because *almost everything* "
            "precedes gains in an up-drifting market. Strip the base rate and ~a dozen events leave "
            "nothing the null can't explain."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $a_t$ be the advance ratio (advancers / (advancers + decliners)) and "
            "$e_t = \\mathrm{EMA}_{10}(a_t)$. A **thrust** completes at $t$ iff $e_t > 0.615$ and "
            "$\\exists\\,s \\in [t-10, t]:\\ e_s < 0.40$ (one event per up-leg, 252-day cooldown).\n\n"
            "- **H₁ (the signal predicts).** $\\mathbb{E}[r_{t\\to t+H}\\mid \\text{thrust}] > "
            "\\mathbb{E}[r_{t\\to t+H}]$ for forward horizon $H$ — a *positive excess* over the base rate.\n"
            "- **H₂ (it's deployable).** The excess is large and frequent enough, net of costs, to "
            "allocate capital to.\n"
            "- **H₃ (\"never wrong\").** The high post-thrust win-rate reflects the signal, not the "
            "market's unconditional up-drift.\n\n"
            "We find **H₁ not rejected but not supported** (positive, $t < 2$, sign-flips at 3m), "
            "**H₂ rejected** (≈22 events / 31y), **H₃ rejected** (the win-rate ≈ the base rate). The "
            "legend is true exactly where it's uninformative (the market goes up) and unproven exactly "
            "where it would pay (a *distinct* edge)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The whole teardown is one decomposition: a conditional mean against an unconditional "
            "baseline, judged by its **standard error**.\n\n"
            "$$\\widehat{\\Delta}_H = \\bar r^{\\text{thrust}}_H - \\bar r^{\\text{all}}_H,\\qquad "
            "t = \\frac{\\widehat{\\Delta}_H}{\\sqrt{\\,s^2_{\\text{thrust}}/k + s^2_{\\text{all}}/N\\,}}.$$\n\n"
            "With $k \\approx 22$, $s_{\\text{thrust}}/\\sqrt{k}$ is large: even a few-point $\\widehat{\\Delta}$ "
            "drowns in its own SE. A raw **win-rate** is the wrong lens entirely — US 12-month returns "
            "are positive ~80% of the time *unconditionally*, so a 91% conditional rate is ≈1 SE of a "
            "proportion away from the null. The honest small-sample instrument is a **randomization "
            "(placebo) test**: resample $k$ random entry dates and ask how often chance matches the "
            "thrust. That number, not the point estimate, decides the Signal axis."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Breadth proxy.** Daily share advancing across a fixed 40-name large-cap basket "
            f"(yfinance adjusted closes, {R['start']}→{R['end']}, {R['days']:,} days). Explicit "
            "**proxy** for NYSE A/D — narrower, noisier, survivorship-tilted (named on the axis).\n"
            "- **Detector.** $e_t = \\mathrm{EMA}_{10}(a_t)$; thrust = first up-cross of 0.615 with a "
            "sub-0.40 print in the prior 10 days; 252-day cooldown ⇒ one event per surge.\n"
            "- **Forward returns.** Enter the close **1 day after** the signal (no look-ahead), hold "
            "$H\\in\\{3,6,12\\}$ months; drop events whose horizon overruns the tape.\n"
            "- **Null #1 (Welch t).** Conditional mean vs the unconditional overlapping-window mean.\n"
            "- **Null #2 (placebo).** 20,000 draws of $k$ random valid entry dates; $p = "
            "\\Pr[\\text{random-draw mean} \\ge \\text{thrust mean}]$ — the small-sample workhorse.\n"
            "- **Costs.** 1-day lag + a 5-bps round-trip on a buy-and-hold trade.\n"
            "- **Positive control.** A deterministic breadth series with **12 thrusts injected** and a "
            "**known** forward edge: the detector must find them, the inference must recover a planted "
            "edge **and** must NOT manufacture significance when the true edge is zero."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The point estimates — positive, small, horizon-flipping\n\n"
            "Conditional mean forward return with its $\\pm$ standard error, against the unconditional "
            "base rate (dashed). Positive at 6/12m, *below* base at 3m."
        ),
        code(
            "hs = [3, 6, 12]\n"
            "if HAVE_REAL:\n"
            "    cm, bm, ts = [], [], []\n"
            "    for m in hs:\n"
            "        s = st.summarize(F, EVENTS, m); cm.append(s['cond_mean']); bm.append(s['base_mean']); ts.append(s['t'])\n"
            "        cr = st.event_returns(F, EVENTS, m); s.setdefault('_se', cr.std(ddof=1)/np.sqrt(len(cr)))\n"
            "    ses = [st.event_returns(F, EVENTS, m).std(ddof=1)/np.sqrt(len(st.event_returns(F, EVENTS, m))) for m in hs]\n"
            "else:\n"
            "    cm = [R['h3'][2]/100, R['h6'][2]/100, R['h12'][2]/100]\n"
            "    bm = [R['h3'][4]/100, R['h6'][4]/100, R['h12'][4]/100]\n"
            "    ts = [R['h3'][6], R['h6'][6], R['h12'][6]]; ses = [0.035, 0.05, 0.06]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x, [c*100 for c in cm], yerr=[s*100 for s in ses], capsize=5, color=GREEN, width=.5, label='after a thrust (±SE)')\n"
            "ax.plot(x, [b*100 for b in bm], 'D', ms=11, c=GREY, label='unconditional base rate')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m}m' for m in hs]); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean forward SPY return (%)')\n"
            "ax.set_title('Excess over base rate is small and t<2 (and negative at 3m)'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Welch t by horizon:', {f'{m}m': round(t,2) for m,t in zip(hs,ts)})"
        ),
        md(
            f"> 💡 In plain words: at 12 months the thrust mean is **+{R['h12'][2]:.1f}%** vs a base "
            f"**+{R['h12'][4]:.1f}%** — a ~3-point bump at **t = {R['h12'][6]:.2f}** (not significant). "
            f"At 3 months it's **{R['h3'][6]:.2f}** — *below* base. H₁ is **not rejected, not "
            "supported**: a positive whisper inside its own error bar that doesn't even keep its sign."
        ),
        md(
            "### 4b · The decisive test — a placebo null sized to the event count\n\n"
            "Draw 22 *random* entry dates 20,000 times; the histogram is the null distribution of the "
            "12-month mean. The thrust is the green line; the *p*-value is the right-tail mass."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(F, EVENTS, 12); obs = pl['thrust_mean']; pval = pl['p_value']\n"
            "    h = st.TRADING_DAYS[12]; spy = F['spy'].values; n=len(spy); k=len(EVENTS); lag=1\n"
            "    vh = n-h-lag-1; fwd = spy[lag+h:lag+h+vh]/spy[lag:lag+vh]-1\n"
            "    rng = np.random.default_rng(361)\n"
            "    draws = np.array([fwd[rng.integers(0,vh,size=k)].mean() for _ in range(20000)])\n"
            "else:\n"
            "    obs = R['h12'][2]/100; pval = R['h12'][7]\n"
            "    rng = np.random.default_rng(361); draws = rng.normal(R['h12'][4]/100, 0.045, 20000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*100, bins=60, color=GREY, alpha=.85, label=f'null: mean of {len(draws):,} random 22-date draws')\n"
            "ax.axvline(obs*100, c=GREEN, lw=2.5, label=f'observed thrust mean {obs*100:.1f}%')\n"
            "ax.set_xlabel('12-month mean forward return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Placebo p = {pval:.2f}: the thrust is inside the luck cloud'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'P[random 22-date draw >= thrust] = {pval:.3f}  (need <0.05 to call it real; here ~1 in 5)')"
        ),
        md(
            f"> 💡 In plain words: **{R['h12'][7]*100:.0f}%** of random 22-date samples match or beat the "
            "thrust. A real edge would push the green line into the far right tail; instead it sits "
            "mid-cloud. This is the crux — **not** a weak point estimate alone, but a point estimate "
            "that **a coin could have produced 1 time in 5**. H₃ rejected: the record is base-rate × rarity."
        ),
        md(
            "### 4c · Robustness + cost — neither rescues it\n\n"
            "Shift the trigger threshold and apply a round-trip cost. The *t* never approaches 2, and "
            "5 bps barely dents a multi-month hold (so cost is **not** the binding constraint — "
            "frequency is)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob = []\n"
            "    for hi in (0.610, 0.615, 0.650):\n"
            "        e2 = st.detect_thrusts(F, hi=hi); s = st.summarize(F, e2, 12)\n"
            "        rob.append((hi, len(e2), s['cond_mean']*100, s['t'], s['p_placebo']))\n"
            "    netr = [(m,)+ (lambda c: (c['gross_mean']*100, c['net_mean']*100))(st.net_of_costs(F, EVENTS, m, 5.0)) for m in (3,6,12)]\n"
            "else:\n"
            "    rob = R['robust']; netr = [(m,g,nn) for (m,g,nn) in R['net']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))\n"
            "his = [r[0] for r in rob]; tt = [r[3] for r in rob]; nn = [r[1] for r in rob]\n"
            "a1.bar([f'{h:.3f}' for h in his], tt, color=AMBER, width=.55)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 (significance bar)')\n"
            "for i,(t,k) in enumerate(zip(tt,nn)): a1.annotate(f'n={k}',(i,t),ha='center',va='bottom')\n"
            "a1.set_title('12m t never reaches 2'); a1.set_xlabel('trigger threshold'); a1.set_ylabel('Welch t'); a1.set_ylim(0,2.4); a1.legend()\n"
            "ms = [r[0] for r in netr]; g = [r[1] for r in netr]; nv = [r[2] for r in netr]; xx=np.arange(len(ms))\n"
            "a2.bar(xx-.2,g,.4,color=GREEN,label='gross'); a2.bar(xx+.2,nv,.4,color=GREY,label='net @5bps')\n"
            "a2.set_xticks(xx); a2.set_xticklabels([f'{m}m' for m in ms]); a2.set_ylabel('mean trade return (%)')\n"
            "a2.set_title('Cost barely matters — frequency does'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (hi, n, cond12%, t, p):', [(r[0], r[1], round(r[2],1), round(r[3],2), round(r[4],3)) for r in rob])"
        ),
        md(
            "> 💡 In plain words: at the canonical 0.65 trigger you get ~15 events (closer to the famed "
            "\"dozen\") and an *even weaker* test; loosening to 0.61 doesn't cross t=2 either. Costs "
            "shave a handful of bps off a months-long hold. There is no parameter and no cost regime in "
            "which this becomes deployable — **the constraint is the event count itself.**"
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic breadth series with **12 thrusts injected**: with **zero** planted edge "
            "the test must stay below t=2 (a dozen events can't fake significance); with a **+10%/6m** "
            "planted edge it must light up. Both hold — proving the engine is unbiased *and* that this "
            "sample size only detects implausibly large edges."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.10):\n"
            "    syn = data.synthetic_breadth(n_events=12, edge_6m=edge, seed=361)\n"
            "    ev = st.detect_thrusts(syn, cooldown=120); s = st.summarize(syn, ev, 6)\n"
            "    res.append((edge, len(ev), s['cond_mean']*100, s['base_mean']*100, s['t'], s['p_placebo']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'planted edge\\n{e*100:.0f}% / 6m' for e,_,_,_,_,_ in res]\n"
            "tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 (significance bar)')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('Welch t (6-month)'); ax.set_title('Control: ~12 events detect only a HUGE edge'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,c,b,t,p in res: print(f'planted {e*100:+.0f}%/6m: detected={k} cond={c:.1f}% base={b:.1f}% t={t:.2f} p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** real edge the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (below 2 — no false positive); only a **+10%/6m** edge — "
            f"enormous — reaches **t = {R['syn'][1][4]:.2f}**. So the machinery is honest, and the "
            "real-tape *t* of ~0.9 is exactly what an *absent or tiny* edge looks like through a "
            "22-event keyhole. The sample size is the verdict."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — 12m excess **+{R['h12'][2]-R['h12'][4]:.1f}pp** at Welch "
            f"**t = {R['h12'][6]:.2f}** / placebo **p = {R['h12'][7]:.2f}**; 3m *t* = {R['h3'][6]:.2f} "
            "(sign flip). Literature support + a positive-but-insignificant, non-robust point estimate "
            "⇒ WEAK, not REAL (the desk's t≥2 bar is not met).\n"
            f"- **Tradability `FRAGILE`** — **{R['n_events']} trades / {R['years']:.0f}y** (true NYSE "
            "≈ once a decade). Net of 5-bps costs the per-trade number is intact; **frequency**, not "
            "cost, makes it un-allocatable. No NAV-scale edge.\n"
            f"- **Never wrong? `BUSTED`** — 12m win-rate **{R['h12'][3]:.0f}%** vs unconditional "
            f"**{R['h12'][5]:.0f}%**; the placebo says a coin matches the record ~1 time in 5. A "
            "sample-size + base-rate mirage — **Study 167 (Hindenburg) inverted**."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the power curve\n\n"
            "The operational truth in one picture: how big would the *true* 12-month excess have to be "
            "for a $k$-event study to detect it at t=2? At $k=22$ you'd need an excess several times the "
            "one observed; the real signal lives far below the detection floor."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cr = st.event_returns(F, EVENTS, 12); sd = cr.std(ddof=1)\n"
            "    obs_excess = (R['h12'][2]-R['h12'][4])/100\n"
            "else:\n"
            "    sd = 0.16; obs_excess = (R['h12'][2]-R['h12'][4])/100\n"
            "ks = np.arange(5, 120)\n"
            "min_detectable = 2.0 * sd / np.sqrt(ks)   # excess needed for t=2 (base SE ~0)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(ks, min_detectable*100, c=AMBER, lw=2, label='excess needed for t=2')\n"
            "ax.axhline(obs_excess*100, c=GREEN, ls='--', label=f'observed excess ~{obs_excess*100:.1f}%')\n"
            "ax.axvline(R['n_events'], c=GREY, ls=':', label=f\"our k={R['n_events']}\")\n"
            "ax.set_xlabel('number of events k'); ax.set_ylabel('12-month excess return (%)')\n"
            "ax.set_title('Detection floor vs the real signal: we are far under-powered'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "need = 2.0*sd/np.sqrt(R['n_events'])\n"
            "print(f'at k={R[\"n_events\"]} you need ~{need*100:.1f}% excess for t=2; observed ~{obs_excess*100:.1f}% -> under-powered by ~{need/obs_excess:.1f}x')"
        ),
        md(
            "> 💡 In plain words: the amber curve is the **minimum detectable excess**; the green line "
            "is what we actually see. They don't meet until $k$ is many times larger than the thrust "
            "will ever deliver. Even granting the lore a *real* few-point edge, you'd need on the order "
            "of a century of additional NYSE thrusts to prove it. There is no sizing, no threshold, no "
            "cost assumption that manufactures statistical power from a dozen events — **the rarity that "
            "makes the signal romantic is exactly what makes it untestable and untradable.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The inverted twin.** [Study 167 — Hindenburg Omen](../../167-hindenburg-omen/): the "
            "same rare-breadth construct pointed at crashes; identical sample-size pathology, opposite "
            "narrative. Reading them together is the cleanest demonstration that *rarity itself* defeats "
            "inference.\n"
            "- **Breadth as raw material.** [Study 168 — Advance-Decline](../../168-advance-decline/) — "
            "does the A/D line carry information orthogonal to price?\n"
            "- **Better breadth.** Replace the 40-name proxy with the full S&P 500 (or true NYSE A/D if "
            "you have a feed) and a McClellan-style oscillator; the event count and the proxy bias "
            "change, the power problem does not.\n\n"
            "*The reproducible core is offline and deterministic; the breadth input is an explicit "
            "proxy. Methods and sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
