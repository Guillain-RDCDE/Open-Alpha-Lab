"""Generate the two narrative notebooks for Study 391 (CEO-Turnover).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily
closes under ../_cache/ (each event ticker + SPY) and otherwise quote the frozen headline
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily closes,
# hardcoded ~25-event CEO-change table, as-of 2026-06-22; 23 events with price history,
# 11 forced / 12 planned; market-model CAR, SPY benchmark).
R = dict(
    asof="2026-06-22", n_table=26, n_used=23, n_forced=11, n_planned=12,
    fingerprint="2b48ba1b75f8",
    # canonical CAR[0,+2] buckets: (n, mean_pct, win_pct, t_vs0)
    forced=(11, 1.02, 45, 0.63),
    planned=(12, -0.45, 42, -0.43),
    allb=(23, 0.25, 43, 0.27),
    diff_pp=1.47, diff_t=0.76, forced_placebo_p=0.246,
    # the un-tradable day-0 window [0,0]: forced mean, t, diff_t, placebo_p
    day0=(2.05, 1.95, 2.43, 0.007),
    # tradable lag1 [+1,+3]: forced mean, t, diff_t, net@10bps
    lag1=(-0.81, -0.70, -0.47, -0.91),
    # robustness windows: (label, forced_mean, forced_t, diff_pp, diff_t)
    robust=[("[0,0]", 2.05, 1.95, 2.70, 2.43), ("[0,+2]", 1.02, 0.63, 1.47, 0.76),
            ("[-1,+1]", 2.44, 1.33, 3.04, 1.48), ("[0,+4]", 1.52, 0.73, 2.22, 0.87)],
    # synthetic control: (planted_bps, forced_mean, forced_t, planned_mean, diff_t)
    syn=[(0.0, -0.95, -1.51, -0.24, -0.64), (500.0, 4.05, 6.41, -0.24, 3.87)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Good_or_bad%3F: Busted](https://img.shields.io/badge/Good_or_bad%3F-Busted-8b949e?style=flat-square)\n\n"
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

from ceo_turnover import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES, EVENTS = data.load_real()
    PANEL = st.car_panel(PRICES, EVENTS)            # canonical CAR[0,+2]
else:
    PRICES = EVENTS = PANEL = None
print("real price cache present:", HAVE_REAL,
      "| events with data:", (0 if PANEL is None else len(PANEL)))
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell
# can quote it whether or not the cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Is firing the CEO good news or bad news for the stock? 👔\n"
            "### The oldest argument on the trading floor, settled with a stopwatch\n\n"
            + BADGES +
            "When a board pushes out its CEO, two camps shout past each other. **\"Buy it\"** — the "
            "underperformer is gone, the activists won, here comes the relief rally. **\"Sell it\"** — "
            "leadership chaos, uncertainty, who's steering the ship now? Both sound obvious. They "
            "can't both be right.\n\n"
            "So we did the boring thing: take ~25 real, dated large-cap CEO changes — tagged "
            "**forced** (shown the door) or **planned** (orderly hand-off) — and measure exactly what "
            "the stock did around each one, *after subtracting out whatever the whole market did that "
            "day*. The answer is a little surprising, and a lot humbling.\n\n"
            "> 📓 **Plain-language layer.** Want the market-model math, the *t*-stats and the placebo "
            "test? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Real CEO-turnover databases aren't free, so we **hardcode a "
            "transparent table** of well-known changes (and label each forced/planned ourselves — a "
            "judgement call at the edges, which we flag). Two delisted names drop out. Every chart is "
            "drawn by the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does a *forced* ouster move the stock? | **Yes — on day one.** Forced-CEO stocks jump "
            f"about **+{R['day0'][0]:.1f}%** the day the news lands (a real, significant pop). |\n"
            "| Great — can I trade that? | **No.** That jump is the price *instantly* re-rating on the "
            "news. By the time you can buy (the next day), it's already in the price — and what follows "
            f"is **{R['lag1'][0]:+.1f}%** of nothing. |\n"
            "| Is *planned* succession different? | **Barely.** A planned hand-off is a shrug "
            f"(**{R['planned'][1]:+.1f}%**). Forced minus planned over a holdable window is "
            f"**+{R['diff_pp']:.1f} points** — but that gap is **statistically indistinguishable from "
            "luck** with so few events. |\n"
            "| So… good news or bad news? | **Neither, reliably.** The only dependable move is the "
            "day-one jump you can't catch. Everything you *could* trade is noise of either sign. |\n\n"
            "> Firing the CEO is real news — the market reacts in the first minutes. But \"react\" "
            "isn't \"tradable,\" and ~a dozen examples per camp can't tell good from bad."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When the board fires an underperforming CEO, the stock pops — buy the relief.\"*  \n"
            "> *\"When a CEO is forced out, it's chaos — sell the uncertainty.\"*\n\n"
            "Both are repeated as floor wisdom, often about the *same* event. The seduction is that a "
            "CEO change is a **clean, dated catalyst**: you know the day, you know the direction "
            "(someone got fired), so surely there's a trade. We'll take that seriously and measure the "
            "**abnormal return** — the stock's move *minus the market's* — around each announcement, "
            "split by whether the exit was forced or planned."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two different questions hide inside \"does firing the CEO move the stock,\" and only one of "
            "them is tradable. (1) *Does the price react to the news?* — almost certainly yes; markets "
            "reprice on information. (2) *Is there a move left **for you** after the news is public?* — "
            "that's the only one that pays, and it requires a **drift** in the days *after* the "
            "announcement, not just a jump *on* it. A signal that fully fires in the first instant is "
            "real economics and a useless trade. And to tell forced from planned, you need enough "
            "events to see through the noise — which, for famous CEO firings, you simply don't have."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We hardcode **~25 notable CEO changes** (as-of {R['asof']}; **{R['n_used']}** have clean "
            f"price history — **{R['n_forced']}** forced, **{R['n_planned']}** planned) and run a "
            "textbook **event study**:\n\n"
            "1. **Subtract the market.** For each stock, fit a simple line — *how this stock normally "
            "moves with the S&P* — over a calm window **before** the announcement. The **abnormal "
            "return** is whatever the stock did beyond that.\n"
            "2. **Add up the event days.** Cumulate the abnormal return over a short window around the "
            "announcement (the **CAR**). Do it for the forced camp and the planned camp.\n"
            "3. **Stress the luck.** Draw the same handful of *random* (stock, date) windows thousands "
            "of times and ask how often chance looks as good. With a dozen events per camp, that's the "
            "honest test."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, every event on one chart.** Each bar is one CEO change's abnormal return over the "
            "3-day window — green if the exit was forced, grey if planned. Notice the *spread*: a few "
            "big winners, a few big losers, no clean pattern."
        ),
        code(
            "if HAVE_REAL:\n"
            "    p = PANEL.sort_values('car')\n"
            "    cars = p['car'].values*100; cols = [GREEN if f else GREY for f in p['forced']]\n"
            "    labs = [f\"{t} {d.year}\" for t,d in zip(p['ticker'], p['announce_date'])]\n"
            "else:\n"
            "    cars = np.array([-8.2,-5.2,-4.8,-3.9,-3.5,-3.1,-2.9,-1.2,-0.4,-0.3,-0.1,-0.1,0.7,1.0,1.1,1.4,1.4,3.0,3.9,4.1,7.4,9.8,10.2])\n"
            "    cols = [GREY]*len(cars); labs = ['']*len(cars)\n"
            "fig, ax = plt.subplots(figsize=(9.6, 5.4))\n"
            "y = np.arange(len(cars))\n"
            "ax.barh(y, cars, color=cols)\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=7)\n"
            "ax.set_xlabel('abnormal return over [0,+2] days (%)')\n"
            "from matplotlib.patches import Patch\n"
            "ax.legend(handles=[Patch(color=GREEN,label='forced out'), Patch(color=GREY,label='planned')], loc='lower right')\n"
            "ax.set_title('Every CEO change: abnormal return around the announcement')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'{len(cars)} events; spread from {cars.min():.1f}% to {cars.max():.1f}% — no clean sign')"
        ),
        md(
            f"The cloud is the story: outcomes run from **−8%** to **+10%** in *both* camps. The forced "
            f"average ({R['forced'][1]:+.1f}%) is a touch positive, the planned average "
            f"({R['planned'][1]:+.1f}%) a touch negative — but look how few dots there are, and how far "
            "they scatter. Now the key move: *when* does the reaction actually happen?"
        ),
        md(
            "**Where the move hides — day one vs the days after.** Here's the forced camp's abnormal "
            "return measured two ways: just the **announcement day** (you can't trade this — you only "
            "see it at the close) vs the **holdable** 3-day window."
        ),
        code(
            "if HAVE_REAL:\n"
            "    p00 = st.car_panel(PRICES, EVENTS, window=(0,0))\n"
            "    f00 = p00.loc[p00['forced'],'car'].mean()*100\n"
            "    f02 = PANEL.loc[PANEL['forced'],'car'].mean()*100\n"
            "    plag = st.car_panel(PRICES, EVENTS, window=(0,2), lag=1)\n"
            "    flag = plag.loc[plag['forced'],'car'].mean()*100\n"
            "else:\n"
            "    f00, f02, flag = R['day0'][0], R['forced'][1], R['lag1'][0]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "names = ['day 0 only\\n(you CANT trade)', 'hold [0,+2]\\n(includes day 0)', 'enter +1 day\\n(what you CAN do)']\n"
            "vals = [f00, f02, flag]; cols = [AMBER, GREEN, RED]\n"
            "ax.bar(names, vals, color=cols)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('forced-CEO abnormal return (%)')\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_title('The whole move is the day-one jump — and it is gone by the time you can act')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'day0 {f00:+.1f}%  ->  hold-from-next-day {flag:+.1f}%  (the catchable part is nothing)')"
        ),
        md(
            f"There it is. The forced-CEO stock pops **+{R['day0'][0]:.1f}%** *on the announcement "
            "day* — a genuine, instant re-rating. But that's the price already adjusting to the news. "
            f"Enter the **next** day, like a real trader must, and the move ahead of you is "
            f"**{R['lag1'][0]:+.1f}%** — noise. The good news got priced while you were reading the "
            "headline."
        ),
        md(
            "**Could a dozen random windows look like the forced camp?** The honest small-sample test: "
            "draw 11 *random* (stock, date) windows over and over, and see where the actual forced "
            "average lands in the cloud of luck (canonical 3-day window)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    f = PANEL.loc[PANEL['forced'],'car'].to_numpy()\n"
            "    null = st.placebo_car_dist(PRICES, data.TICKERS, k=len(f), n_draws=6000, seed=391)\n"
            "    obs = f.mean(); pval = st.placebo_pvalue(obs, null)\n"
            "else:\n"
            "    rng=np.random.default_rng(391); null = rng.normal(0.0, 0.012, 6000)\n"
            "    obs = R['forced'][1]/100; pval = R['forced_placebo_p']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(null*100, bins=50, color=GREY, alpha=.85, label='11 RANDOM windows (luck)')\n"
            "ax.axvline(obs*100, c=GREEN, lw=2.5, label=f'the actual forced camp ({obs*100:+.1f}%)')\n"
            "ax.set_xlabel('average abnormal return over [0,+2] (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The forced camp sits inside the luck cloud — placebo p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'a random 11-window draw matches/beats the forced camp {pval*100:.0f}% of the time')"
        ),
        md(
            f"The green line — the real forced camp — sits **inside** the grey cloud (placebo "
            f"*p* ≈ **{R['forced_placebo_p']:.2f}**). Over the window you could actually hold, the "
            "forced-CEO average is **what a handful of random dates produces anyway.** The day-one jump "
            "was real; the *tradable* part is not."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** A forced ouster *does* move the stock on day one "
            f"(**+{R['day0'][0]:.1f}%**, real) — but only on the day you can't trade. Over a holdable "
            f"window forced is **+{R['forced'][1]:.1f}%**, planned a shrug, and the gap between them "
            f"(**+{R['diff_pp']:.1f}pp**) is inside the noise. Real-as-a-day-one-reaction, weak-as-a-signal.\n"
            "- **Tradability — Mirage.** The move is the instant the news hits. Enter the next day and "
            f"you get **{R['lag1'][0]:+.1f}%** — worse after costs. Nothing to capture.\n"
            "- **\"Good news or bad news?\" — Busted.** The only reliable move is the un-catchable "
            "day-one jump; the *holdable* drift is statistically zero in **both** directions. The clean "
            "fire-the-CEO trade doesn't exist."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — count the events\n\n"
            "Forget significance for a second. How *many* shots does this give you, and how spread out "
            "are they? Here's the calendar of forced ousters — the punchline is how few there are."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fy = [d.year for d,fo in zip(PANEL['announce_date'], PANEL['forced']) if fo]\n"
            "    py = [d.year for d,fo in zip(PANEL['announce_date'], PANEL['forced']) if not fo]\n"
            "else:\n"
            "    fy=[2010,2011,2016,2017,2017,2018,2019,2019,2019,2021]; py=[2011,2013,2014,2015,2015,2016,2019,2020,2020,2021]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 2.8))\n"
            "ax.eventplot(fy, colors=GREEN, lineoffsets=1, linelengths=.7, label='forced')\n"
            "ax.eventplot(py, colors=GREY, lineoffsets=0, linelengths=.7, label='planned')\n"
            "ax.set_yticks([0,1]); ax.set_yticklabels(['planned','forced']); ax.set_xlim(2009, 2023)\n"
            "ax.set_title(f'{len(fy)} forced + {len(py)} planned events in ~12 years'); ax.set_xlabel('year')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'{len(fy)} forced ousters total — about one a year, scattered. Too few to learn from, too instant to trade.')"
        ),
        md(
            f"About **{R['n_forced']} forced ousters** of famous large-caps in a decade-plus. Even if "
            "each one carried a clean signal, that's not enough events to build a strategy on — and the "
            "signal *isn't* clean, and it fires before you can act. Three strikes: too **few**, too "
            "**noisy**, too **fast**."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **A frequent corporate event for contrast.** [Study 249 — "
            "Index-Inclusion](../249-index-inclusion/) runs the same event-study machinery on S&P "
            "additions — a *frequent* event with a measurable-but-tiny pop. Reading them together "
            "shows what rarity does to inference.\n"
            "- **Another dated catalyst.** [Study 228 — Pre-Earnings-Runup](../228-pre-earnings-runup/) "
            "asks whether a *known, scheduled* event leaves a capturable drift.\n"
            "- **Build your own.** Swap our hardcoded table for a bigger CEO-change list (or change the "
            "forced/planned calls) and re-run — more events tighten the error bars, but the day-one "
            "timing problem won't go away.\n\n"
            "*Think there's a tradable fire-the-CEO edge? Show the forced camp landing **outside** the "
            "luck cloud on a window you could actually **hold from the next day** — then we'll talk.*"
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
            "# CEO-Turnover — a quantitative event-study teardown 🔬\n"
            "### Market-model CARs by bucket · forced−planned Welch *t* + a placebo non-event-window "
            "null · the [0,0]-vs-[0,+2] window split · a 1-day-lag tradable variant + costs · a "
            "synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We separate "
            "the two things \"firing the CEO moves the stock\" fuses: a **real announcement-instant "
            "repricing** from a **(non-existent) post-announcement drift**, and we confront the "
            "forced-vs-planned contrast with the **sample size**. The decisive object is not the sign "
            "of any point estimate (the buckets disagree, as the folklore does) but its **power**: with "
            "~a dozen events per bucket, the forced−planned CAR is statistically indistinguishable from "
            "random windows on the same names.\n\n"
            "> ⚠️ **Data + label note.** True CEO-turnover databases aren't free; we use a hardcoded, "
            "labelled table of ~25 notable large-cap changes (forced/planned is the believers' framing "
            "and subjective at the margin; two delisted names + one pre-IPO event drop out — a mild "
            "survivorship tilt, named on the Signal axis). Real data: yfinance daily closes, each "
            "ticker + SPY. Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `WEAK` | Day-0 forced CAR **+{R['day0'][0]:.1f}%** (placebo "
            f"**p = {R['day0'][3]:.3f}**) is real — but only on the un-holdable [0,0] window. Canonical "
            f"[0,+2]: forced **{R['forced'][1]:+.1f}%** (t = {R['forced'][3]:.2f}), **forced−planned "
            f"{R['diff_pp']:+.1f}pp** at Welch **t = {R['diff_t']:.2f}**, placebo **p = "
            f"{R['forced_placebo_p']:.2f}** — fails t ≥ 2. |\n"
            f"| **Tradability** | `MIRAGE` | Enter **+1 day** (the only executable timing): forced CAR "
            f"**{R['lag1'][0]:+.1f}%** (t = {R['lag1'][1]:.2f}), net **{R['lag1'][3]:+.1f}%** after "
            "10 bps. No post-announcement drift. |\n"
            f"| **Good or bad?** | `BUSTED` | The decidable move is the un-catchable announcement "
            "instant; the holdable drift is **zero in both directions**. Window-mining + small-sample "
            "illusion. |\n\n"
            "> 💡 In plain words: the stock *does* reprice when a CEO is forced out — instantly, on the "
            "news — but \"reprices\" isn't \"drifts,\" and ~a dozen events per bucket can't separate a "
            "forced relief-pop from a planned shrug once you measure something you could actually hold."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "For event $i$ with announcement day $0$, fit the market model "
            "$r_{i,t} = \\alpha_i + \\beta_i\\, r_{m,t} + \\varepsilon_{i,t}$ on a clean estimation "
            "window $[-130,-10]$, then the **abnormal return** is "
            "$AR_{i,t} = r_{i,t} - (\\hat\\alpha_i + \\hat\\beta_i\\, r_{m,t})$ and the **CAR** over "
            "window $[\\tau_1,\\tau_2]$ is $\\mathrm{CAR}_i = \\sum_{t=\\tau_1}^{\\tau_2} AR_{i,t}$.\n\n"
            "- **H₁ (forced ≠ 0).** $\\mathbb{E}[\\mathrm{CAR}\\mid\\text{forced}] \\neq 0$ over a "
            "*holdable* window.\n"
            "- **H₂ (forced ≠ planned).** $\\mathbb{E}[\\mathrm{CAR}\\mid\\text{forced}] \\neq "
            "\\mathbb{E}[\\mathrm{CAR}\\mid\\text{planned}]$ — the believers' core directional claim.\n"
            "- **H₃ (it's deployable).** The bucket difference survives a 1-day execution lag and costs.\n\n"
            "We find **H₁ supported only at $[0,0]$** (the un-tradable instant), **rejected on any "
            "holdable window**; **H₂ not rejected** (forced−planned $t < 1$ over $[0,+2]$); **H₃ "
            "rejected** (lag-1 CAR flips to noise of the wrong sign). The legend is true exactly where "
            "it's unobtainable (the announcement instant) and unproven exactly where it would pay (a "
            "*post-news drift*)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The whole teardown is one comparison of small-sample means judged by their **standard "
            "error**:\n\n"
            "$$t = \\frac{\\bar{\\mathrm{CAR}}_{\\text{forced}} - \\bar{\\mathrm{CAR}}_{\\text{planned}}}"
            "{\\sqrt{\\,s^2_{\\text{f}}/k_{\\text{f}} + s^2_{\\text{p}}/k_{\\text{p}}\\,}},\\qquad "
            "k_{\\text{f}}, k_{\\text{p}} \\approx 11\\text{–}12.$$\n\n"
            "With $k \\approx 12$ and CARs that scatter $\\pm 5$–$10\\%$, $s/\\sqrt{k}$ is large: even a "
            "couple-of-point gap drowns in its own SE. A raw **win-rate** is the wrong lens (it's near "
            "50/50 by construction); the honest instruments are the **Welch t** and a **placebo / "
            "randomization null** — random non-event $(\\text{ticker},\\text{date})$ windows on the same "
            "names. And the *window choice itself* is a degree of freedom: a 1-day $[0,0]$ window can "
            "cross $t=2$ while $[0,+2]$ does not, which is precisely the look-ahead trap — significance "
            "on the un-tradable instant is not a tradable signal."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Event table.** Hardcoded ~25 large-cap CEO changes (ticker, announce date, "
            f"forced/planned); **{R['n_used']}** with clean price history ({R['n_forced']} forced / "
            f"{R['n_planned']} planned), as-of {R['asof']}, fingerprint `{R['fingerprint']}`.\n"
            "- **Market model.** $r = \\alpha + \\beta\\,r_{\\mathrm{SPY}}$ on $[-130,-10]$ (120-day "
            "estimation, 10-day gap so the event can't leak into the fit).\n"
            "- **CAR window.** Canonical $[0,+2]$; robustness over $[0,0]$, $[-1,+1]$, $[0,+4]$.\n"
            "- **Null #1 (Welch t).** Each bucket mean vs 0; forced−planned two-sample.\n"
            "- **Null #2 (placebo).** Random non-event $(\\text{ticker},\\text{date})$ windows on the "
            "same names; $p = \\Pr[|\\text{random mean}| \\ge |\\text{forced mean}|]$.\n"
            "- **Tradable variant.** Enter the close **+1 day** after the announcement (you learn the "
            "news intraday, fill next), hold the window; one-way 10-bps round-trip.\n"
            "- **Positive control.** A deterministic per-event abnormal-return panel with a **plantable "
            "forced-bucket CAR edge**: the engine must recover a planted edge **and** must NOT "
            "manufacture significance when the true edge is 0."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The buckets — small means, big error bars\n\n"
            "Mean CAR$[0,+2]$ per bucket with its $\\pm$ standard error. Forced is mildly positive, "
            "planned mildly negative, both error bars straddle zero."
        ),
        code(
            "if HAVE_REAL:\n"
            "    f = PANEL.loc[PANEL['forced'],'car'].to_numpy(); p = PANEL.loc[~PANEL['forced'],'car'].to_numpy()\n"
            "    means = [f.mean()*100, p.mean()*100]; ses = [f.std(ddof=1)/np.sqrt(len(f))*100, p.std(ddof=1)/np.sqrt(len(p))*100]\n"
            "    tf, tp = st.welch_t(f), st.welch_t(p); dt = st.welch_t(f, p)\n"
            "else:\n"
            "    means=[R['forced'][1], R['planned'][1]]; ses=[1.6, 1.05]; tf,tp,dt=R['forced'][3],R['planned'][3],R['diff_t']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['forced','planned'], means, yerr=ses, capsize=6, color=[GREEN, GREY], width=.55)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('mean CAR [0,+2] (%)')\n"
            "for i,(m,s) in enumerate(zip(means,ses)): ax.annotate(f'{m:+.2f}%\\n(t={[tf,tp][i]:+.2f})',(i,m),ha='center',va='bottom' if m>=0 else 'top')\n"
            "ax.set_title(f'Forced vs planned: difference {means[0]-means[1]:+.2f}pp at Welch t={dt:.2f} (n.s.)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'forced {means[0]:+.2f}% (t={tf:+.2f}) | planned {means[1]:+.2f}% (t={tp:+.2f}) | diff t={dt:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: forced is **{R['forced'][1]:+.1f}%** at t = {R['forced'][3]:.2f}, "
            f"planned **{R['planned'][1]:+.1f}%** at t = {R['planned'][3]:.2f}; the **{R['diff_pp']:+.1f}pp** "
            f"gap is Welch **t = {R['diff_t']:.2f}**. H₁ fails on a holdable window and H₂ fails outright "
            "— a positive whisper for forced, a negative whisper for planned, both inside their error bars."
        ),
        md(
            "### 4b · The decisive split — significance lives only in the un-tradable instant\n\n"
            "Same forced bucket, four windows. The single-day $[0,0]$ — the close-to-close move *on* "
            "the news — crosses $t=2$; every **holdable** window collapses below it. The window that "
            "lights up is the one you can't trade."
        ),
        code(
            "wins = [(0,0),(0,2),(-1,1),(0,4)]; labs=['[0,0]','[0,+2]','[-1,+1]','[0,+4]']\n"
            "if HAVE_REAL:\n"
            "    tt=[]; dd=[]\n"
            "    for w in wins:\n"
            "        pw = st.car_panel(PRICES, EVENTS, window=w)\n"
            "        ff = pw.loc[pw['forced'],'car'].to_numpy(); pp = pw.loc[~pw['forced'],'car'].to_numpy()\n"
            "        tt.append(st.welch_t(ff)); dd.append(st.welch_t(ff, pp))\n"
            "else:\n"
            "    tt=[r[2] for r in R['robust']]; dd=[r[4] for r in R['robust']]\n"
            "x=np.arange(len(wins))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, tt, .4, color=GREEN, label='forced vs 0')\n"
            "ax.bar(x+.2, dd, .4, color=AMBER, label='forced − planned')\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 (significance bar)')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels(labs)\n"
            "ax.set_ylabel('Welch t'); ax.set_title('Only the un-tradable [0,0] window crosses t=2'); ax.legend()\n"
            "ax.annotate('you CANT\\ntrade this', (0, max(tt[0],dd[0])), ha='center', va='bottom', color=RED, fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('forced-vs-0 t by window:', {l: round(t,2) for l,t in zip(labs,tt)})"
        ),
        md(
            f"> 💡 In plain words: at $[0,0]$ the forced−planned t is **{R['day0'][2]:.2f}** (and the "
            f"forced placebo p is **{R['day0'][3]:.3f}**) — a *real* announcement-day repricing. Widen "
            f"to the holdable $[0,+2]$ and it's **{R['diff_t']:.2f}**. Significance that exists only on "
            "the exact print, and vanishes the instant the window is wide enough to hold, is the "
            "textbook signature of an un-tradable information event — not an edge."
        ),
        md(
            "### 4c · The placebo + the tradable lag — neither rescues it\n\n"
            "Left: the forced camp's mean CAR$[0,+2]$ against random non-event windows on the same "
            "names. Right: what you actually get entering **+1 day** (the only executable timing), "
            "gross vs net of 10 bps."
        ),
        code(
            "if HAVE_REAL:\n"
            "    f = PANEL.loc[PANEL['forced'],'car'].to_numpy()\n"
            "    null = st.placebo_car_dist(PRICES, data.TICKERS, k=len(f), n_draws=8000, seed=391)\n"
            "    obs = f.mean(); pval = st.placebo_pvalue(obs, null)\n"
            "    plag = st.car_panel(PRICES, EVENTS, window=(0,2), lag=1)\n"
            "    flag = plag.loc[plag['forced'],'car'].mean()\n"
            "    nc = st.net_of_costs(flag)\n"
            "    g, nn = nc['gross_pct'], nc['net_pct']\n"
            "else:\n"
            "    rng=np.random.default_rng(391); null = rng.normal(0.0, 0.012, 8000)\n"
            "    obs = R['forced'][1]/100; pval = R['forced_placebo_p']\n"
            "    g, nn = R['lag1'][0], R['lag1'][3]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.2))\n"
            "a1.hist(null*100, bins=50, color=GREY, alpha=.85, label='random windows (null)')\n"
            "a1.axvline(obs*100, c=GREEN, lw=2.5, label=f'forced camp {obs*100:+.1f}%')\n"
            "a1.set_xlabel('mean CAR [0,+2] (%)'); a1.set_ylabel('freq')\n"
            "a1.set_title(f'Placebo p = {pval:.2f}'); a1.legend()\n"
            "a2.bar(['gross','net @10bps'], [g, nn], color=[GREEN, RED])\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('forced CAR, enter +1 day (%)')\n"
            "for i,v in enumerate([g,nn]): a2.annotate(f'{v:+.1f}%',(i,v),ha='center',va='top')\n"
            "a2.set_title('Tradable timing: the drift is gone (and negative)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'placebo p={pval:.3f} | enter+1day gross {g:+.1f}% net {nn:+.1f}%')"
        ),
        md(
            f"> 💡 In plain words: the forced camp's holdable mean sits mid-cloud (placebo "
            f"**p ≈ {R['forced_placebo_p']:.2f}**), and the *only executable* version — enter the next "
            f"day — delivers **{R['lag1'][0]:+.1f}%** gross, **{R['lag1'][3]:+.1f}%** net. Cost isn't "
            "the killer; **timing** is. The information is in the price before you can touch it."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "A deterministic per-event panel (14 forced / 12 planned). With a **zero** planted edge the "
            "forced−planned test must stay below $t=2$; with a **+500 bps/event** planted forced edge it "
            "must light up. Both hold — the engine is unbiased *and* this sample size only detects "
            "implausibly large edges."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 500.0):\n"
            "    syn = data.synthetic_events(car_bps=edge, seed=391)\n"
            "    ff, pp = syn['forced_car'], syn['planned_car']\n"
            "    res.append((edge, ff.mean()*100, st.welch_t(ff), pp.mean()*100, st.welch_t(ff, pp)))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = ['planted 0 bps\\n(null)', 'planted +500 bps\\n(huge)']\n"
            "dts = [r[4] for r in res]\n"
            "ax.bar(labels, dts, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2'); ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(dts): ax.annotate(f't={t:+.2f}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "ax.set_ylabel('forced − planned Welch t'); ax.set_title('Control: ~12/bucket detects only a HUGE edge'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,fm,ft,pm,dt in res: print(f'planted {e:+.0f}bps: forced {fm:+.2f}% (t={ft:+.2f}) planned {pm:+.2f}% diff t={dt:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** real edge the control's forced−planned t is "
            f"**{R['syn'][0][4]:.2f}** (below 2 — no false positive); only a **+5%/event** edge — "
            f"enormous — reaches **t = {R['syn'][1][4]:.2f}**. So the machinery is honest, and the "
            f"real-tape forced−planned t of **{R['diff_t']:.2f}** is exactly what an *absent or tiny* "
            "edge looks like through a 12-per-bucket keyhole. The sample size is the verdict."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — a genuine day-0 forced repricing (**+{R['day0'][0]:.1f}%**, placebo "
            f"**p = {R['day0'][3]:.3f}**), but confined to the un-holdable $[0,0]$ window. Holdable "
            f"$[0,+2]$: forced **{R['forced'][1]:+.1f}%** (t = {R['forced'][3]:.2f}), forced−planned "
            f"**{R['diff_pp']:+.1f}pp** at **t = {R['diff_t']:.2f}** / placebo **p = "
            f"{R['forced_placebo_p']:.2f}**. Literature support + a positive-but-insignificant, "
            "window-fragile estimate on a subjective label ⇒ WEAK, not REAL.\n"
            f"- **Tradability `MIRAGE`** — enter +1 day (the only executable timing): forced "
            f"**{R['lag1'][0]:+.1f}%** (t = {R['lag1'][1]:.2f}), net **{R['lag1'][3]:+.1f}%**. The whole "
            "move is the announcement instant; there is no post-news drift to capture.\n"
            "- **Good or bad? `BUSTED`** — the only reliable move is un-catchable; the holdable drift is "
            "**zero in both directions**. A window-mining + small-sample illusion."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the power curve\n\n"
            "The operational truth in one picture: how big would the *true* forced−planned CAR have to "
            "be for a $k$-per-bucket study to detect it at $t=2$? At $k\\approx 12$ you'd need a gap "
            "several times the one observed; the real signal lives far below the detection floor."
        ),
        code(
            "if HAVE_REAL:\n"
            "    f = PANEL.loc[PANEL['forced'],'car'].to_numpy(); p = PANEL.loc[~PANEL['forced'],'car'].to_numpy()\n"
            "    sd = np.sqrt(f.var(ddof=1)/len(f) + p.var(ddof=1)/len(p))   # SE of the difference\n"
            "    obs_diff = (f.mean() - p.mean()); kf = len(f)\n"
            "else:\n"
            "    sd = (R['diff_pp']/100)/max(R['diff_t'],1e-9); obs_diff = R['diff_pp']/100; kf = R['n_forced']\n"
            "ks = np.arange(5, 120)\n"
            "se_at_k = sd * np.sqrt(kf / ks)            # SE shrinks ~1/sqrt(k); scale from observed k\n"
            "min_detectable = 2.0 * se_at_k\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(ks, min_detectable*100, c=AMBER, lw=2, label='forced−planned gap needed for t=2')\n"
            "ax.axhline(abs(obs_diff)*100, c=GREEN, ls='--', label=f'observed gap ~{abs(obs_diff)*100:.1f}pp')\n"
            "ax.axvline(kf, c=GREY, ls=':', label=f'our k≈{kf} per bucket')\n"
            "ax.set_xlabel('events per bucket k'); ax.set_ylabel('forced−planned CAR gap (pp)')\n"
            "ax.set_title('Detection floor vs the real signal: we are far under-powered'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "need = 2.0*sd*100\n"
            "print(f'at k≈{kf} you need ~{need:.1f}pp gap for t=2; observed ~{abs(obs_diff)*100:.1f}pp -> under-powered by ~{need/(abs(obs_diff)*100):.1f}x')"
        ),
        md(
            "> 💡 In plain words: the amber curve is the **minimum detectable gap**; the green line is "
            "what we see. They don't meet until $k$ is many times larger than the famous-CEO-firing "
            "tape will ever deliver. Even granting the lore a *real* couple-of-point gap, you'd need "
            "many decades of additional ousters to prove it — and you *still* couldn't trade it, because "
            "the move is in the price before the next open. **The rarity and instant-repricing that make "
            "the story vivid are exactly what make it untestable and untradable.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The frequent-event mirror.** [Study 249 — Index-Inclusion](../249-index-inclusion/): "
            "the same market-model event study on a *frequent* corporate catalyst (S&P additions) — "
            "measurable-but-tiny print, opposite end of the power spectrum.\n"
            "- **A scheduled catalyst.** [Study 228 — Pre-Earnings-Runup](../228-pre-earnings-runup/) "
            "and [Study 369 — Earnings-Revision-Momentum](../369-earnings-revision-momentum/) — does a "
            "*known* event leave a capturable drift?\n"
            "- **Better data.** Replace the hardcoded table with a full ExecuComp/BoardEx turnover "
            "panel (hundreds of events), add successor-origin (insider/outsider) and prior-performance "
            "controls; the error bars tighten, but the day-one timing problem — the move is priced "
            "before you can act — does not.\n\n"
            "*The reproducible core is offline and deterministic; the event table is an explicit "
            "hardcoded, labelled stand-in. Methods and sources: [`docs/references.md`](../docs/references.md); "
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
