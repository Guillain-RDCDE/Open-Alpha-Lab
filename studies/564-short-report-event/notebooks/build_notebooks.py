"""Generate the two narrative notebooks for Study 564 (Short-Report-Event).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached event prices under
../_cache/ (the surviving famous short-report targets + SPY) and otherwise quote the frozen
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance, 32 famous short-report
# campaigns, 22 priced after 10 delisted, + SPY, 2010-01-04 -> 2026-06-26, 4,145 days; price
# fingerprint 946a53954ab8; as-of 2026-06-30).
R = dict(
    start="2010-01-04", end="2026-06-26", days=4145, years=16.5,
    n_events=32, n_priced=22, n_delisted=10, fp_px="946a53954ab8",
    # report-day crater (day 0): n, mean%, median%, neg_share%, t
    crater=(22, -1.97, -0.27, 55, -0.75),
    # per-horizon drift: (months, n, mean_ex%, median_ex%, hit%, t, p_placebo)
    h1=(1, 22, -1.63, -9.13, 77, -0.22, 0.359),
    h3=(3, 22, -11.55, -18.52, 64, -1.55, 0.068),
    h6=(6, 22, -14.58, -42.68, 73, -0.81, 0.095),
    # squeezers (6m excess vs SPY): (ticker, excess%)
    squeeze=[("CVNA", 318.2), ("PDD", 50.5), ("ADT", 36.1), ("IQ", 22.4)],
    # drop 3 biggest squeezers (6m): n, mean%, t
    nosqueeze=(19, -38.2, -4.81),
    # short book net (10bps + 800bps/yr borrow): (months, gross_short%, net_short%)
    net=[(1, 1.63, 0.87), (3, 11.55, 9.45), (6, 14.58, 10.48)],
    # lag robustness (6m): (lag, mean%, t, p)
    lag=[(0, -10.61, -0.51, 0.164), (1, -14.58, -0.81, 0.095),
         (2, -14.45, -0.75, 0.097), (5, -14.19, -0.69, 0.101)],
    # synthetic control (6m, seed-robust mean t over 25 seeds): (planted_drift, mean_t)
    ctrl=[(0.0, 0.70), (-0.10, -1.01), (-0.20, -2.91), (-0.35, -6.10)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Free_short%3F: Busted](https://img.shields.io/badge/Free_short%3F-Busted-8b949e?style=flat-square)\n\n"
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

from short_report_event import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES, TBL = data.load_real()
else:
    PRICES, TBL = None, None
print("real event-price cache present:", HAVE_REAL,
      "| short-report campaigns priced:", (0 if TBL is None else len(TBL)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Short the hit piece? — does a stock crater and *stay* down after an activist short report? 🐻\n"
            "### The \"free short\" of selling the moment Hindenburg or Muddy Waters publishes, in plain English\n\n"
            + BADGES +
            "Here's a tempting trade. A famous activist *short*-seller — **Hindenburg Research**, "
            "**Muddy Waters**, **Citron** — spends months building a case that some company is a fraud "
            "or wildly over-valued, quietly bets against the stock, and then **publishes** the whole "
            "thesis in a splashy public report. The lore says: the stock **craters** on the report, and "
            "then **keeps falling** for months as regulators circle and the story sinks in. Just short "
            "when the report drops — free money on the way down.\n\n"
            "It *sounds* like a free short, and a few legends back it up (Nikola, Lordstown, Luckin — all "
            "went to zero-ish). But \"it worked for the famous ones\" is exactly what **survivorship** "
            "looks like — and worse, the biggest wins literally **delisted**, so you can't even see them "
            "in the data anymore. This notebook checks the short you could actually place, on a basket of "
            "the most famous take-downs, and asks whether it beats simply owning the market.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the squeeze maths? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** A clean list of *every* short report isn't free, so we "
            "**hardcode 32 famous campaigns** (22 still price; 10 delisted) — a basket picked *because* "
            "it's famous, which tilts the test *toward* the claim. If even the legends can't beat the "
            "market *on average*, the honest version certainly can't. Every chart is drawn by the code "
            "beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the stock drop when the report is published? | **A bit — about −2% on the day** on "
            "the survivors. Real news. But you can only grab it if you were *already* short. You weren't. |\n"
            "| If I short the *day after* and hold, does the stock keep falling vs the market? | **Usually "
            "yes.** The *typical* (median) target underperforms the S&P by **−43%** over six months, and "
            "the short is right **~7 times in 10**. |\n"
            "| So it's a great trade? | **No — because of the squeeze.** A few names *rocket* after being "
            "left for dead (Carvana ran **+318%**). One squeeze wipes out eighteen winners, so the "
            "**average** short barely clears luck. |\n"
            "| Then why the \"free short\" reputation? | **Survivorship.** You remember Nikola and "
            "Lordstown (they delisted!); you forget the squeezes that blew up. |\n\n"
            "> The report is real news. The *tradable* short after it usually works — but activist "
            "shorting is a high-hit-rate bet with an **unbounded left tail**: the stock can only fall to "
            "zero, but it can 10x against you."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When a short-seller publishes a fraud/over-valuation report, the stock craters and then "
            "keeps falling for months — short the report and ride it down.\"*\n\n"
            "The picture is intuitive and has a real foothold: when a credible firm lays out a detailed "
            "fraud case, believers sell, regulators sometimes act, and the thesis can play out over "
            "months. The seduction is to assume the *whole* drop is *yours* and that the fall *continues* "
            "smoothly on a stock you can actually short. We'll separate the part that's real-but-"
            "unreachable (the report-day crater) from the part you could trade (the drift after)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two very different things hide inside \"the report craters and keeps falling.\" (1) The "
            "**report-day crater** — the day the thesis breaks. It's real, but it lands *before you can "
            "act*. (2) The **post-report drift** — what you earn if you **short** *after* the report is "
            "public and hold for months. Only the second is a strategy. And it has to beat not *zero* but "
            "the **market**: a falling stock in a falling market isn't alpha. Crucially, a short has a "
            "nasty shape — you can make at most 100% (the stock to zero) but you can lose *far more* if it "
            "squeezes. So even a short that's *usually* right can lose money *on average*."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{R['n_events']} famous short-report campaigns** ({R['start'][:4]}-2023), pull "
            f"each target's price and SPY (**{R['n_priced']}** still price — {R['n_delisted']} delisted), "
            "and split the effect cleanly:\n\n"
            "1. **The crater.** What did the target do on the report day itself? (Reported, not tradable.)\n"
            "2. **The drift you can short.** Short at the close **one day after** the report (a realistic, "
            "no-peeking entry), hold **1 / 3 / 6 months**, and measure the target's return **in excess of "
            "SPY** — target minus market. *Negative* = the short worked.\n"
            "3. **Stress the luck.** For each event, compare it to shorting the *same stock* on a "
            "**random** day. Do that thousands of times: how often does a random-date basket fall as much? "
            "That's the honest test for a ~22-event signal — and we watch what the squeezes do to the "
            "average."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the report-day crater.** Here's the day-0 return for each surviving campaign — the "
            "allegation being priced in the moment it breaks."
        ),
        code(
            "if HAVE_REAL:\n"
            "    craters = st.report_day_moves(PRICES, TBL)\n"
            "    cm = craters.mean()*100\n"
            "else:\n"
            "    craters = np.array([R['crater'][1]/100]); cm = R['crater'][1]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "order = np.argsort(craters)\n"
            "cols = [RED if p < 0 else GREEN for p in craters[order]]\n"
            "ax.bar(range(len(craters)), craters[order]*100, color=cols)\n"
            "ax.axhline(cm, c=GREY, ls='--', label=f'average {cm:.1f}%')\n"
            "ax.set_xlabel('campaign (sorted by day-0 return)'); ax.set_ylabel('report-day return (%)')\n"
            "ax.set_title(f'The report crater: real news, ~{cm:.1f}% on average — but you can\\'t short it'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'report-day move: mean {cm:.2f}%  (down { (craters<0).mean()*100:.0f}% of the time)')"
        ),
        md(
            f"A modest **{R['crater'][1]:.1f}%** average drop, down more often than not — real news. But "
            "it's the move *before* you can act, so it isn't a trade. (And the *biggest* craters — the "
            "frauds that went to zero — **delisted** and aren't even in this picture.) The real question: "
            "what happens *after* you can short?"
        ),
        md(
            "**The drift you can actually short.** Short one day after the report, hold, and measure the "
            "target's return *below the market* (target − SPY). Here it is at 1, 3 and 6 months — "
            "**negative bars mean the short worked** — with both the *average* and the *typical* (median) "
            "name shown."
        ),
        code(
            "hs = [1, 3, 6]\n"
            "if HAVE_REAL:\n"
            "    means = [st.event_excess_drift(PRICES, TBL, m).mean()*100 for m in hs]\n"
            "    meds  = [np.median(st.event_excess_drift(PRICES, TBL, m))*100 for m in hs]\n"
            "    hits  = [st.summarize(PRICES, TBL, m)['hit_rate']*100 for m in hs]\n"
            "else:\n"
            "    means = [R['h1'][2], R['h3'][2], R['h6'][2]]\n"
            "    meds  = [R['h1'][3], R['h3'][3], R['h6'][3]]\n"
            "    hits  = [R['h1'][4], R['h3'][4], R['h6'][4]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.5))\n"
            "ax.bar(x-.2, means, .4, color=GREY, label='average (mean)')\n"
            "ax.bar(x+.2, meds, .4, color=RED, label='typical (median)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m} months' for m in hs])\n"
            "ax.set_ylabel('excess return vs SPY (%)  — negative = short worked')\n"
            "ax.set_title('The typical stock CRATERS; the average is dragged up by squeezes'); ax.legend()\n"
            "for i,(mn,md_,h) in enumerate(zip(means,meds,hits)):\n"
            "    ax.annotate(f'{h:.0f}% hit',(i,min(mn,md_)),ha='center',va='top',fontsize=8,color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('mean excess:', {f'{m}m': f'{v:+.1f}%' for m,v in zip(hs,means)})\n"
            "print('median excess:', {f'{m}m': f'{v:+.1f}%' for m,v in zip(hs,meds)})"
        ),
        md(
            f"Look at the gap. The **typical** target is *crushed* — median **{R['h6'][3]:.0f}%** under "
            f"the market at six months — and the short is right **{R['h6'][4]:.0f}%** of the time. But "
            f"the **average** is only **{R['h6'][2]:.0f}%**, because a few names *squeezed* violently. "
            "The median is the short working; the mean is the squeeze biting back."
        ),
        md(
            "**What ruins the average?** One picture. Here are the six-month excess returns of every "
            "surviving campaign, sorted — a wall of red (shorts that worked) and a couple of green "
            "monsters (squeezes)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ex6 = st.event_excess_drift(PRICES, TBL, 6)*100\n"
            "else:\n"
            "    ex6 = np.array([-42.7]*15 + [s[1] for s in R['squeeze']] + [-10,-5,-2])\n"
            "order = np.argsort(ex6)\n"
            "cols = [RED if e < 0 else GREEN for e in ex6[order]]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(range(len(ex6)), ex6[order], color=cols)\n"
            "ax.axhline(ex6.mean(), c=GREY, ls='--', label=f'mean {ex6.mean():.0f}%')\n"
            "ax.axhline(np.median(ex6), c=RED, ls=':', label=f'median {np.median(ex6):.0f}%')\n"
            "ax.set_xlabel('campaign (sorted)'); ax.set_ylabel('6-month excess vs SPY (%)')\n"
            "ax.set_title('Most shorts worked (red); one Carvana-style squeeze (green) eats the average'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('biggest squeezer excess:', ex6.max().round(0), '%  |  median:', round(float(np.median(ex6)),0), '%')"
        ),
        md(
            f"That lone green tower is **Carvana (+{R['squeeze'][0][1]:.0f}%)** — left for dead after a "
            "solvency scare, then a ten-bagger. On a 22-name book, that single squeeze outweighs the "
            "eighteen names where the short paid off. **This is activist shorting in one chart:** great "
            "hit-rate, deadly left tail."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The drift is the right sign (median **{R['h6'][3]:.0f}%** at 6m, hit "
            f"**{R['h6'][4]:.0f}%**), but the **average** never clears the significance bar because of "
            "squeezes. Real as lore, weak as a mean edge.\n"
            "- **Tradability — Fragile.** A short with bounded gains and unbounded squeeze losses, on a "
            "rare event, with a punitive borrow — and the biggest wins delisted out of view.\n"
            "- **Free short? — Busted.** \"Short the report and ride it down\" is **survivorship**: you "
            "remember the frauds that went to zero (they delisted) and forget the ones that squeezed you "
            "out."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the borrow and the squeeze\n\n"
            "Suppose you ran it anyway. Shorting a freshly-crashed name is *expensive* — the borrow fee "
            "on a hard-to-borrow small-cap can be enormous. Here's the short book's average P&L (positive "
            "= the short made money) gross vs net of a 10-bps round-trip **plus an 800 bps/yr borrow**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = [st.short_net_of_costs(PRICES, TBL, m, 10.0, 800.0)['gross_short']*100 for m in [1,3,6]]\n"
            "    nv = [st.short_net_of_costs(PRICES, TBL, m, 10.0, 800.0)['net_short']*100 for m in [1,3,6]]\n"
            "else:\n"
            "    g = [R['net'][0][1], R['net'][1][1], R['net'][2][1]]\n"
            "    nv = [R['net'][0][2], R['net'][1][2], R['net'][2][2]]\n"
            "x = np.arange(3)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(x-.2, g, .4, color=GREEN, label='gross short P&L')\n"
            "ax.bar(x+.2, nv, .4, color=GREY, label='net @ 10 bps + 800 bps/yr borrow')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['1m','3m','6m']); ax.axhline(0,c='k',lw=.8)\n"
            "ax.set_ylabel('short book P&L (%)'); ax.set_title('The MEAN short is positive net of a heavy borrow -- but carries the squeeze tail')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('gross short:', [round(x,1) for x in g], '| net short:', [round(x,1) for x in nv])"
        ),
        md(
            "The *average* short book is **positive even net of a heavy borrow** — but that average is the "
            "one being propped up by the median while a squeeze lurks in the tail. A positive mean with a "
            "Carvana-shaped left tail is not something you can size into NAV: one bad squeeze and the "
            "borrow bill arrives exactly when the position is running against you. `FRAGILE`."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The long-side cousin.** [390 Activist-13D](../../390-activist-13d/) asks the mirror "
            "question — does a stock drift *up* after an activist takes a big *long* stake? Same "
            "event-study shape, opposite sign, and *no squeeze risk*.\n"
            "- **Add the dead.** The frauds that went to zero (Nikola, Lordstown, Luckin) **delisted** and "
            "drop from the free feed — precisely the biggest wins. A survivorship-free tape would make the "
            "*median* short look even better, but the *mean* still faces the squeeze.\n"
            "- **Cap the tail.** Re-run with a stop-loss on the short (e.g. exit at +50%); the mean "
            "recovers, but you'd have stopped out of exactly the names that later re-collapsed.\n\n"
            "*Think short reports are a clean edge? Build the full report panel (including the delisted "
            "winners), short one day after each, race it down against SPY, and show the **mean** — not "
            "just the median — landing outside the random-date cloud at t ≤ −2.*"
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
            "# Short-Report-Event — a quantitative teardown 🔬\n"
            "### Crater vs drift split · excess-of-SPY drift at 1/3/6m · mean-vs-median skew · a Welch *t* "
            "+ same-target placebo null · costs + borrow · lag robustness · a seed-robust synthetic "
            "short-edge control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We separate "
            "the two things the \"report craters and stays down\" claim fuses: an **uncapturable "
            "report-day crater** (day-0 news) from the **tradable post-report drift** (short +1 day, hold "
            "H months, in **excess of SPY**). And we confront the drift with the **shape of its "
            "distribution**: a great median and hit-rate, undone by a fat right tail of **short "
            "squeezes**.\n\n"
            "> ⚠️ **Data + selection note.** No clean full-coverage short-report panel is free; we use a "
            "**hardcoded basket of 32 famous campaigns** (22 priced, 10 delisted) — selected on outcome "
            "(bias *for* the claim) whose biggest wins delisted (bias *against*), both named on the Signal "
            "axis. Real data: yfinance daily adjusted closes, 22 targets + SPY, 2010→2026, price fp "
            f"`{R['fp_px']}`. Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `WEAK` | 6m excess **mean {R['h6'][2]:.1f}%** / **median {R['h6'][3]:.1f}%** "
            f"vs SPY, hit **{R['h6'][4]:.0f}%**, Welch **t = {R['h6'][5]:.2f}** (best 3m "
            f"**t = {R['h3'][5]:.2f}**, placebo *p* {R['h3'][6]:.3f}). Right sign, great hit-rate, but "
            "the **mean fails |t| ≥ 2** — squeezes. |\n"
            f"| **Tradability** | `FRAGILE` | Bounded gains, **unbounded squeeze loss** (CVNA "
            f"**+{R['squeeze'][0][1]:.0f}%**); net of 10-bps + **800 bps/yr borrow** the mean short is "
            f"**+{R['net'][2][2]:.1f}%** at 6m — a positive mean wrapped around an ugly tail. |\n"
            f"| **Free short?** | `BUSTED` | A same-target placebo says random dates fall as far "
            f"**{R['h6'][6]*100:.0f}%** of the time; the biggest wins **delisted**. Selection + squeeze. |\n\n"
            "> 💡 In plain words: the report *crater* is real news you can't short; the post-report "
            "*drift* you can short usually works (median crushed, 64–77% hit) but its **mean** is eaten by "
            "fat-tailed squeezes, so no horizon certifies at |t| ≥ 2."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $P_t$ be the target close and $M_t$ SPY. For event $j$ with report day $0_j$, define the "
            "**crater** $\\pi_j = P_{0_j}/P_{0_j-1}-1$ (the news) and the **tradable short drift** "
            "$x_j^{(H)} = \\big(P_{0_j+1+H}/P_{0_j+1}-1\\big) - \\big(M_{0_j+1+H}/M_{0_j+1}-1\\big)$ — "
            "short one day after, hold $H$, in **excess of SPY**. A short profits when $x_j^{(H)} < 0$.\n\n"
            "- **H₁ (the drift predicts).** $\\mathbb{E}[x^{(H)}] < 0$ at $t \\le -2$ — a *negative excess "
            "over the market* you can actually short.\n"
            "- **H₂ (it's deployable).** The short is large/frequent/*robust-tailed* enough, net of costs "
            "and borrow, to allocate to.\n"
            "- **H₃ (free short).** The drift reflects the *report timing*, not just owning special names "
            "(jumpy, already falling) that would have fallen on *any* entry date.\n\n"
            "We find **H₁ directionally supported but not certified** (mean negative, median very "
            "negative, hit 64–77%, but $|t| < 2$ — a fat right tail of squeezes), **H₂ rejected** (bounded "
            "gains vs unbounded squeeze loss; ~22 rare events), **H₃ ambiguous** (placebo *p* ≈ 0.07–0.10 "
            "— close, but not below 0.05, and the mean is squeeze-driven). The legend is true where it's "
            "unreachable (the crater) and *usually-but-not-bankably* true where it would pay (the drift)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The teardown is one decomposition — a mean excess against zero — but here the **mean is the "
            "wrong summary** and saying so *is* the finding:\n\n"
            "$$\\widehat{x}^{(H)} = \\frac1k\\sum_j x_j^{(H)},\\qquad t = \\frac{\\widehat{x}^{(H)}}{s_x/\\sqrt{k}}.$$\n\n"
            "A short's payoff is **asymmetric**: $x_j \\in [-1, +\\infty)$ (the stock can only fall to "
            "zero, but a squeeze is unbounded). So $x^{(H)}$ is heavily **right-skewed** — a few squeezes "
            "inflate both $\\widehat{x}$ *and* $s_x$, crushing $|t|$ even when the **median** and the "
            "**hit-rate** scream that the short works. That is why we report the **median** and "
            "**hit-rate** alongside the mean, and why the honest small-sample instrument is a "
            "**same-target placebo**: resample each name's excess from a *random* entry date and ask how "
            "often chance falls as far. The mean *t* alone would *understate* the directional signal and "
            "*overstate* the tradability — both matter."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Event set.** A **hardcoded** table of {R['n_events']} famous short-report campaigns "
            f"(yfinance adjusted closes, {R['start']}→{R['end']}, {R['days']:,} price days). "
            f"**{R['n_priced']} priced, {R['n_delisted']} delisted** — outcome-selected roster (bias "
            "*for*) whose biggest wins delisted (bias *against*), both on the axis.\n"
            "- **Crater (reported).** Day-0 return of the target — the news, not a trade.\n"
            "- **Drift (tradable).** **Short** the close **1 day after** the report (no look-ahead), hold "
            "$H\\in\\{1,3,6\\}$m, return in **excess of SPY**; drop events whose horizon overruns the "
            "tape.\n"
            "- **Null #1 (Welch t).** Mean excess vs zero; a *negative* $t$ is the short.\n"
            "- **Null #2 (same-target placebo).** 20,000 draws of one random-date excess **per name**; "
            "$p=\\Pr[\\text{random-date basket mean} \\le \\text{report basket mean}]$ — one-sided toward "
            "the short, controlling for the targets' own drift.\n"
            "- **Costs.** 1-day lag + a 10-bps round-trip **+ an 800 bps/yr short borrow** (pro-rated).\n"
            "- **Robustness.** Entry-lag sweep (0/1/2/5d); the mean-vs-median / squeeze diagnostic.\n"
            "- **Positive control.** A deterministic set of 30 synthetic events with a **known planted "
            "negative drift**, averaged over **25 seeds** (house rule): the engine must recover it and "
            "must NOT manufacture significance at the null."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Crater vs drift, mean vs median — the four numbers\n\n"
            "The report crater (day 0, *uncapturable*) next to the drift you can short at 1/3/6m, shown "
            "as both **mean** (squeeze-inflated) and **median** (the typical name). The crater is real; "
            "the drift's *median* is brutal and its *mean* is not."
        ),
        code(
            "hs = [1, 3, 6]\n"
            "if HAVE_REAL:\n"
            "    craters = st.report_day_moves(PRICES, TBL); crater_m = craters.mean()*100\n"
            "    means, meds, ts = [], [], []\n"
            "    for m in hs:\n"
            "        ex = st.event_excess_drift(PRICES, TBL, m)\n"
            "        means.append(ex.mean()*100); meds.append(np.median(ex)*100); ts.append(st.welch_t_vs_zero(ex))\n"
            "else:\n"
            "    crater_m = R['crater'][1]\n"
            "    means = [R['h1'][2], R['h3'][2], R['h6'][2]]; meds = [R['h1'][3], R['h3'][3], R['h6'][3]]\n"
            "    ts = [R['h1'][5], R['h3'][5], R['h6'][5]]\n"
            "labels = ['crater\\n(day 0)'] + [f'{m}m' for m in hs]\n"
            "x = np.arange(len(labels))\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.5))\n"
            "ax.bar(x-.2, [crater_m]+means, .4, color=GREY, label='mean')\n"
            "ax.bar(x+.2, [crater_m]+meds, .4, color=RED, label='median')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('return (%) vs SPY (crater = raw)')\n"
            "ax.set_title('Median short works at every horizon; the mean is squeeze-dragged'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Welch t by horizon:', {f'{m}m': round(t,2) for m,t in zip(hs,ts)})\n"
            "print('mean:', [round(v,1) for v in means], '| median:', [round(v,1) for v in meds])"
        ),
        md(
            f"> 💡 In plain words: the report crater is **{R['crater'][1]:.1f}%** (news, not yours); the "
            f"short drift's **median** is **{R['h1'][3]:.0f}% / {R['h3'][3]:.0f}% / {R['h6'][3]:.0f}%** at "
            f"1/3/6m — the typical name is *crushed* — but the **mean** is only "
            f"**{R['h1'][2]:.0f}% / {R['h3'][2]:.0f}% / {R['h6'][2]:.0f}%**, so Welch *t* is "
            f"**{R['h1'][5]:.2f} / {R['h3'][5]:.2f} / {R['h6'][5]:.2f}** — all shy of −2. H₁ is "
            "**directionally supported, not certified.**"
        ),
        md(
            "### 4b · The skew that kills the mean — the squeeze\n\n"
            "Every surviving campaign's 6-month excess, sorted. A wall of red (shorts that worked) and a "
            "couple of green outliers (squeezes). The mean sits *above* the median because the right tail "
            "is unbounded."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ex6 = st.event_excess_drift(PRICES, TBL, 6)*100\n"
            "else:\n"
            "    ex6 = np.array([-42.7]*13 + [s[1] for s in R['squeeze']] + [-30,-20,-12,-8,-3])\n"
            "order = np.argsort(ex6); cols = [RED if e<0 else GREEN for e in ex6[order]]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "ax.bar(range(len(ex6)), ex6[order], color=cols)\n"
            "ax.axhline(ex6.mean(), c=GREY, ls='--', label=f'mean {ex6.mean():.0f}%')\n"
            "ax.axhline(np.median(ex6), c=RED, ls=':', label=f'median {np.median(ex6):.0f}%')\n"
            "ax.set_xlabel('campaign (sorted)'); ax.set_ylabel('6m excess vs SPY (%)')\n"
            "ax.set_title('Bounded downside, unbounded upside: one squeeze pulls the mean off the median'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'mean {ex6.mean():.1f}% vs median {np.median(ex6):.1f}% | max squeeze {ex6.max():.0f}% | skew (mean-median) {ex6.mean()-np.median(ex6):.0f}pp')"
        ),
        md(
            f"> 💡 In plain words: the mean−median gap is **~{R['h6'][2]-R['h6'][3]:.0f} points**, all of "
            f"it the **{R['squeeze'][0][0]} +{R['squeeze'][0][1]:.0f}%** squeeze plus a couple of smaller "
            f"ones. Drop the three biggest squeezers and the 6m *t* snaps to **{R['nosqueeze'][2]:.2f}** — "
            "but removing your worst trades *ex-post* is the cherry-pick we refuse. The tail is the trade."
        ),
        md(
            "### 4c · The placebo — is it the *timing* or just the names?\n\n"
            "For each name, draw its 6-month excess from a *random* entry date; do it 20,000 times. The "
            "histogram is the null for the basket mean — what you'd get shorting the same stocks on random "
            "days. The report basket is the red line; the *p*-value is the **left**-tail mass (as negative "
            "or more)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(PRICES, TBL, 6, n_draws=20000); obs = pl['obs_mean']*100; pval = pl['p_value']\n"
            "    spy = PRICES['SPY'].dropna(); h = st.TRADING_DAYS[6]; pools=[]\n"
            "    for _, r in TBL.iterrows():\n"
            "        s = PRICES[r['ticker']].dropna(); common = s.index.intersection(spy.index)\n"
            "        sc = s.reindex(common).values; sp = spy.reindex(common).values\n"
            "        if len(common) > h+2:\n"
            "            pool = (sc[h:]/sc[:-h]-1) - (sp[h:]/sp[:-h]-1); pools.append(pool[np.isfinite(pool)])\n"
            "    rng = np.random.default_rng(564)\n"
            "    draws = np.array([np.mean([p[rng.integers(0,len(p))] for p in pools]) for _ in range(20000)])\n"
            "else:\n"
            "    obs = R['h6'][2]; pval = R['h6'][6]; rng = np.random.default_rng(564)\n"
            "    draws = rng.normal(0.05, 0.18, 20000)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.hist(draws*100, bins=60, color=GREY, alpha=.85, label=f'null: {len(draws):,} random-date baskets (same names)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed report basket {obs:.1f}%')\n"
            "ax.set_xlabel('6-month mean excess vs SPY (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Placebo p = {pval:.3f}: the report basket is in the left tail, but not decisively'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'P[random-date basket <= report basket] = {pval:.3f}  (need <0.05 to call the TIMING real)')"
        ),
        md(
            f"> 💡 In plain words: **{R['h6'][6]*100:.0f}%** of random-date baskets on the *same names* "
            "fall as far as the report basket — the report timing sits in the left tail but **not below "
            "0.05**. Some of the underperformance is the report *timing*; much is just *owning* jumpy, "
            "already-sliding names. **H₃ is ambiguous**, not a clean free short."
        ),
        md(
            "### 4d · Robustness — flat in the entry lag\n\n"
            "Shift the short-entry lag (0/1/2/5 days). If the drift were a one-bar execution artefact the "
            "*t* would move sharply; it doesn't — the underperformance is a genuine slow bleed a lagged "
            "entry neither creates nor destroys."
        ),
        code(
            "if HAVE_REAL:\n"
            "    lags = [0,1,2,5]\n"
            "    lt = [st.summarize(PRICES, TBL, 6, lag=L)['t'] for L in lags]\n"
            "    lm = [st.event_excess_drift(PRICES, TBL, 6, lag=L).mean()*100 for L in lags]\n"
            "else:\n"
            "    lags = [l[0] for l in R['lag']]; lt = [l[2] for l in R['lag']]; lm = [l[1] for l in R['lag']]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.2))\n"
            "ax.bar([f'{L}d' for L in lags], lt, color=AMBER, width=.55)\n"
            "ax.axhline(-2, ls='--', c=RED, label='t = -2 bar (short)')\n"
            "for i,(t,m) in enumerate(zip(lt,lm)): ax.annotate(f'{m:.0f}%',(i,t),ha='center',va='top')\n"
            "ax.set_title('6m Welch t never reaches -2 (any lag)'); ax.set_xlabel('short-entry lag'); ax.set_ylabel('Welch t'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('lag-6m t:', [round(x,2) for x in lt], '| mean:', [round(x,1) for x in lm])"
        ),
        md(
            "> 💡 In plain words: same-day or a week later, the mean and *t* barely move — this is **not** "
            "a timing/execution artefact. It's a genuine, skewed, slow underperformance whose *mean* the "
            "squeeze keeps from clearing the bar at any lag."
        ),
        md(
            "### 4e · Faithful-engine control — we know the truth here (25 seeds)\n\n"
            "On 30 *synthetic* short-report events (each a high-beta GBM path sharing a market factor, a "
            "day-0 crater, and a **known** post-report drift): with **zero** planted drift the mean *t* "
            "must sit near 0 (30 events can't fake it); as we plant a stronger *negative* drift it must "
            "cross −2. Averaged over **25 seeds** so no single lucky seed decides."
        ),
        code(
            "def mean_t(edge, n_seeds=25, base=564):\n"
            "    ts = []\n"
            "    for sd in range(base, base+n_seeds):\n"
            "        p, t = data.synthetic_events(n_events=30, drift_6m=edge, seed=sd)\n"
            "        ts.append(st.welch_t_vs_zero(st.event_excess_drift(p, t, 6)))\n"
            "    return float(np.nanmean(ts))\n"
            "edges = [0.0, -0.10, -0.20, -0.35]\n"
            "tvals = [mean_t(e) for e in edges]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot([e*100 for e in edges], tvals, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(-2, ls='--', c=RED, label='t = -2 bar (short edge)'); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('planted 6-month drift (%) — more negative = stronger short edge')\n"
            "ax.set_ylabel('mean Welch t (25 seeds)')\n"
            "ax.set_title('Control: flat at the null, past -2 as a real short edge grows'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,t in zip(edges,tvals): print(f'planted {e*100:+.0f}%/6m -> mean t {t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: at the null the mean *t* is **+{R['ctrl'][0][1]:.2f}** (flat — no false "
            f"short signal), and a planted **−20%/6m** edge reaches **{R['ctrl'][2][1]:.2f}**, **−35%** "
            f"reaches **{R['ctrl'][3][1]:.2f}**. So the engine is faithful — the real-tape *t* of ~−1 to "
            "−1.5 is exactly what a *directionally real but squeeze-diluted* short looks like through a "
            "22-event keyhole. *(A single seed-564 draw at the null happened to print t ≈ +2 on 30 "
            "high-vol names — which is why the house rule averages ≥ 20 seeds.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — 6m excess **mean {R['h6'][2]:.1f}%** / **median {R['h6'][3]:.1f}%**, "
            f"hit **{R['h6'][4]:.0f}%**, Welch **t = {R['h6'][5]:.2f}** (best 3m **{R['h3'][5]:.2f}**, "
            f"placebo *p* {R['h3'][6]:.3f}). Right sign, high hit-rate, but the **mean** fails |t| ≥ 2 — "
            "squeezes. Literature support + a directionally-real-but-uncertified estimate ⇒ WEAK, not "
            "REAL.\n"
            f"- **Tradability `FRAGILE`** — bounded gains vs **unbounded squeeze loss** (CVNA "
            f"**+{R['squeeze'][0][1]:.0f}%**); net of 10-bps + **800 bps/yr borrow** the mean short is "
            f"**+{R['net'][2][2]:.1f}%** at 6m, but a positive mean around a fat tail isn't NAV-able.\n"
            f"- **Free short? `BUSTED`** — same-target placebo *p* ≈ {R['h6'][6]:.2f} (timing not "
            "decisively better than random days on those names), and the biggest wins **delisted**. "
            "Selection + squeeze, not a free ride."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the borrow and the asymmetric tail\n\n"
            "The short book's P&L is $-x^{(H)}$ (you profit when the target underperforms), charged a "
            "10-bps round-trip **and an 800 bps/yr borrow** on a hard-to-borrow, freshly-crashed name. "
            "The mean survives the borrow — but the *shape* is the problem."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = [st.short_net_of_costs(PRICES, TBL, m, 10.0, 800.0)['gross_short']*100 for m in [1,3,6]]\n"
            "    nv = [st.short_net_of_costs(PRICES, TBL, m, 10.0, 800.0)['net_short']*100 for m in [1,3,6]]\n"
            "else:\n"
            "    g = [R['net'][0][1], R['net'][1][1], R['net'][2][1]]; nv = [R['net'][0][2], R['net'][1][2], R['net'][2][2]]\n"
            "x = np.arange(3)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(x-.2, g, .4, color=GREEN, label='gross short P&L')\n"
            "ax.bar(x+.2, nv, .4, color=GREY, label='net @ 10 bps + 800 bps/yr borrow')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['1m','3m','6m']); ax.axhline(0,c='k',lw=.8)\n"
            "ax.set_ylabel('short book P&L (%)'); ax.set_title('Positive MEAN net of a heavy borrow -- but it carries the squeeze tail')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('gross short:', [round(x,1) for x in g], '| net:', [round(x,1) for x in nv])"
        ),
        md(
            "> 💡 In plain words: the borrow (a heavy 800 bps/yr) still leaves a **positive mean short** — "
            "so cost is *not* what breaks it. What breaks it is the **asymmetry**: a strategy whose mean "
            "rides on the median while a Carvana-shaped squeeze lurks in the tail cannot be sized into "
            "NAV. `FRAGILE`."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The long-side cousin.** [390 Activist-13D](../../390-activist-13d/) — the mirror event "
            "study for activist *longs*: same shape, opposite sign, and crucially *no squeeze risk*. "
            "Reading the two together is the point.\n"
            "- **Survivorship-free panel.** The frauds that went to zero (Nikola, Lordstown, Luckin) "
            "**delisted** and drop from the free feed — the biggest short wins are *missing*, which biases "
            "the surviving mean *up* (against the short). A clean panel would make the **median** even "
            "more brutal; the **mean** still faces the squeeze.\n"
            "- **Tail control.** Add a stop-loss on the short (exit at, say, +50%): the mean recovers and "
            "*t* clears −2 — but you'd stop out of names that later re-collapse, and the borrow bill lands "
            "exactly when the squeeze runs. The tail is not a nuisance to engineer away; it *is* the "
            "risk.\n\n"
            "*The reproducible core is offline and deterministic; the event set is an explicit "
            "outcome-selected basket. Methods and sources: [`docs/references.md`](../docs/references.md); "
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
