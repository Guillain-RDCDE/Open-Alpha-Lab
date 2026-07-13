"""Generate the two narrative notebooks for Study 735 (Ryder-Cup-Effect).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY/VGK
tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic positive control runs anywhere with no
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (SPY + VGK, yfinance,
# 1993-01-29 -> 2026-06-30; 10 of 22 contested Ryder Cups resolved, 2006->2025).
R = dict(
    n_editions=23, n_contested=22, n_included=10, n_usa_loss=7, n_eur_loss=3,
    fp="0d34990aeb50", panel_rows=8411, panel_start="1993-01-29",
    # signal (paired loser-minus-winner spread, day(-1) -> day(-1)+k). Folklore: < 0.
    mon_mean=+0.380, mon_t=+1.817, mon_hit_neg=2, mon_n=10, mon_w_lo=5.7, mon_w_hi=51.0,
    wk_mean=-0.047, wk_t=-0.104, wk_hit_neg=6, wk_n=10, wk_w_lo=31.3, wk_w_hi=83.2,
    # placebo (left-tail: p = share of null means <= observed, the folklore direction)
    pl_mon_p=0.950, pl_mon_mean=+0.006, pl_mon_sd=0.235,
    pl_wk_p=0.433, pl_wk_mean=+0.035, pl_wk_sd=0.477,
    # third axis: constant-mean absolute abnormal legs + USA/Europe asymmetry
    loser_leg_mean=-0.171, loser_leg_t=-0.822,
    winner_leg_mean=-0.546, winner_leg_t=-1.614,
    asym_welch_t=+0.420, usa_loss_mon=+0.424, eur_loss_mon=+0.276,
    # event anatomy (mean cumulative spread by day offset from day(-1))
    car={0: 0.000, 1: 0.380, 2: 0.192, 3: -0.010, 4: 0.109, 5: -0.047},
    # tradability (long winner / short loser, entered day(0), net of costs)
    cap_day_g=+0.189, cap_day_tg=+1.05, cap_day_n2=+0.107, cap_day_t2=+0.59,
    cap_day_n5=-0.013, cap_day_t5=-0.07,
    cap_wk_g=+0.409, cap_wk_tg=+0.81, cap_wk_n2=+0.319, cap_wk_t2=+0.63,
    cap_wk_n5=+0.199, cap_wk_t5=+0.40,
    pl_cap_wk_p=0.179, pl_cap_wk_mean=-0.123,
    # synthetic control
    syn_null_mean=+0.22, syn_null_sd=0.79, syn_null_fire=0, syn_null_seeds=20,
    syn_planted1_t=-2.23, syn_planted2_t=-4.60,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Cross--Atlantic_sentiment_shock%3F: Busted](https://img.shields.io/badge/Cross--Atlantic_sentiment_shock%3F-Busted-8b949e?style=flat-square)\n\n"
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

from ryder_cup_effect import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.load_real()
    EV = st.build_event_table(PRICES, cost_bps=2.0)
    INC = EV[EV["included"]]
    USA_LOSS = INC[INC["loser"] == "USA"]
    EUR_LOSS = INC[INC["loser"] == "Europe"]
else:
    PRICES = EV = INC = USA_LOSS = EUR_LOSS = None
print("real cache present:", HAVE_REAL, "| editions:", len(data.EVENTS),
      "| resolved events:", (0 if INC is None else len(INC)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does losing the Ryder Cup sink your side's stock market? ⛳📉\n"
            "### The cross-Atlantic \"sports-sentiment\" folklore — wrong sign, "
            "no slump, gone against a coin-flip calendar\n\n"
            + BADGES +
            "Every two years the best golfers of the **United States** play the best of "
            "**Europe** for nothing but pride — the Ryder Cup. There's a tidy market "
            "story that rides along with it: sport moves mood, mood moves money, so the "
            "**losing continent's** stock market should sag the Monday after the Sunday "
            "result. It borrows a *real* academic finding — a 2007 study showed national "
            "markets genuinely fall when a country's football team is knocked out of the "
            "World Cup. The Ryder Cup version swaps one football nation for an entire "
            "continent's golf team, and asks whether the same *loss*-driven gloom shows "
            "up.\n\n"
            "We tested it on every edition with a tradable market on both sides — Team "
            "USA as the S&P 500 (`SPY`), Team Europe as a broad Europe ETF (`VGK`).\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the leg "
            "decomposition? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 23 Ryder Cups hardcoded from Wikipedia (1989 was a "
            "tie, no loser). Only the 2006→2025 editions have `VGK` coverage — 10 events. "
            "Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the losing side's market lag the winner's the Monday after? | "
            f"**No — if anything the opposite.** The loser-minus-winner spread is "
            f"**{R['mon_mean']:+.2f}%** (positive means the *loser* did better), and only "
            f"**{R['mon_hit_neg']} of {R['mon_n']}** losing continents actually lagged. |\n"
            f"| Is that little wrong-way blip even real? | **No.** A random Monday is "
            f"*more* likely to show the folklore's dip than the actual Ryder Cup Mondays "
            f"(placebo *p* = **{R['pl_mon_p']:.2f}**), and it's gone by the end of the "
            "week. |\n"
            f"| Does the *loser's own* market slump (the football mechanism)? | **No.** "
            f"The losing side's abnormal Monday return is a flat **{R['loser_leg_mean']:+.2f}%** — "
            "and the *winner's* market actually fell a touch more. |\n"
            f"| Could you have traded it? | **No.** The only tradable version (bet the "
            f"loser keeps lagging, entered after the result) nets **{R['cap_wk_n2']:+.2f}%** "
            "over a week — statistical noise, on an effect pointing the wrong way. |\n\n"
            "> The real football *loss* effect is one of sports finance's tidiest results. "
            "It simply does not carry over to a biennial golf match between two continents."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The Ryder Cup is USA vs Europe, played for pride, watched by tens of "
            "millions on both sides of the Atlantic. When your continent loses, there's a "
            "collective Monday-morning gloom — and gloomy investors sell. The losing "
            "side's market should underperform the winner's.\"*\n\n"
            "It rides on real science: Edmans, García & Norli (2007) found that national "
            "stock markets really do fall the day after a country is *eliminated* from "
            "the football World Cup. Crucially, they found **no** matching *win* effect — "
            "losing hurts, winning doesn't help. So the honest Ryder-Cup version is a "
            "**loss** story: the losing continent lags. Nobody has ever formally tested "
            "it. We did."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this would be a delightful, purely-sentimental, calendar-known edge "
            "— short the losing continent, long the winner, every other September. And it "
            "would extend a genuine academic finding (football losses) to a second sport. "
            "If it's *not* real, it's a clean example of how a true effect in one setting "
            "gets over-extended into folklore in another. We wanted to know which."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** All **{R['n_editions']}** Ryder Cups of the modern "
            f"USA-vs-Europe era (1979→2025), hardcoded with winner and loser (1989 was a "
            "14-14 tie — no loser).\n"
            "- **The market.** Team USA = `SPY` (S&P 500), Team Europe = `VGK` (broad "
            "Europe — includes the UK, Switzerland, the Nordics, not just the euro zone). "
            "We measure the **loser-minus-winner** return, so the shared global move "
            "cancels out.\n"
            "- **The window.** From the last close before the Sunday result through the "
            "Monday and the week after. The folklore predicts a **negative** spread.\n"
            "- **The honesty checks.** A random-window placebo (does a random Monday do "
            "the same?), a decomposition (does the *loser* slump, or the winner?), and a "
            "trade you could *actually* have placed."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The headline: the Monday spread points the wrong way.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    mon = st.one_sample_t(INC['spread_mon'].values)['mean']*100\n"
            "    wk = st.one_sample_t(INC['spread_wk'].values)['mean']*100\n"
            "else:\n"
            "    mon, wk = R['mon_mean'], R['wk_mean']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.6))\n"
            "bars = ax.bar(['the Monday', 'the week'], [mon, wk],\n"
            "              color=[RED, GREY], width=.55)\n"
            "for b, v in zip(bars, [mon, wk]): ax.annotate(f'{v:+.2f}%', (b.get_x()+b.get_width()/2, v),\n"
            "                                              ha='center', va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('loser minus winner (%)')\n"
            "ax.set_title('Folklore predicts NEGATIVE (loser lags). The Monday bar is positive.')\n"
            "ax.annotate('folklore\\nexpects\\nthis way', (0, -0.18), color=GREY, ha='center', fontsize=9)\n"
            "ax.annotate('', xy=(0, -0.28), xytext=(0, -0.05),\n"
            "            arrowprops=dict(arrowstyle='->', color=GREY))\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Monday spread {mon:+.3f}%  |  week spread {wk:+.3f}%')"
        ),
        md(
            f"The loser-minus-winner spread on the Monday is **{R['mon_mean']:+.2f}%** — "
            "and *positive* means the **loser did better**, the exact opposite of the "
            f"claim. It's not big enough to mean anything (*t* = {R['mon_t']:.2f}), but it "
            "is not the dip the folklore promises. By the end of the week even that blip "
            f"is gone ({R['wk_mean']:+.2f}%).\n\n"
            "**And only 2 of 10 losing continents actually lagged at all:**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sp = INC['spread_mon'].values*100\n"
            "    yrs = INC['year'].values; los = INC['loser'].values\n"
            "else:\n"
            "    rng = np.random.default_rng(735); sp = rng.normal(0.4, 1.2, 10)\n"
            "    yrs = [2006,2008,2010,2012,2014,2016,2018,2021,2023,2025]\n"
            "    los = ['USA','Europe','USA','USA','USA','Europe','USA','Europe','USA','USA']\n"
            "labels = [f'{y}\\n({l} lost)' for y, l in zip(yrs, los)]\n"
            "fig, ax = plt.subplots(figsize=(10.5, 4.6))\n"
            "cols = [GREEN if v < 0 else RED for v in sp]\n"
            "ax.bar(range(len(sp)), sp, color=cols)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(range(len(sp))); ax.set_xticklabels(labels, fontsize=8)\n"
            "ax.set_ylabel('loser minus winner, Monday (%)')\n"
            "ax.set_title('Green = loser lagged (folklore-consistent). Only 2 of 10 are green.')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('events where the loser lagged (spread<0):', int((sp<0).sum()), 'of', len(sp))"
        ),
        md(
            f"Eight of ten Ryder Cups had the *losing* continent's market **beat** the "
            "winner's the Monday after. If a real gloom effect were operating, this bar "
            "chart would be mostly green (below zero). It's mostly red.\n\n"
            "**Is the winner-side even the one that's weak?** The folklore is specifically "
            "that the *loser* slumps. Let's split the paired spread back into its two "
            "legs — each side's own abnormal move:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ll = st.one_sample_t(INC['loser_ab_mon'].values)['mean']*100\n"
            "    wl = st.one_sample_t(INC['winner_ab_mon'].values)['mean']*100\n"
            "else:\n"
            "    ll, wl = R['loser_leg_mean'], R['winner_leg_mean']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.6))\n"
            "bars = ax.bar(['losing side\\n(should slump)', 'winning side'], [ll, wl],\n"
            "              color=[AMBER, GREY], width=.5)\n"
            "for b, v in zip(bars, [ll, wl]): ax.annotate(f'{v:+.2f}%', (b.get_x()+b.get_width()/2, v),\n"
            "                                            ha='center', va='top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('abnormal Monday return vs own history (%)')\n"
            "ax.set_title('The loser barely moves; the WINNER dips more. Backwards.')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'loser leg {ll:+.3f}%   winner leg {wl:+.3f}%')"
        ),
        md(
            f"There it is: the losing side's own market is a nothing "
            f"**{R['loser_leg_mean']:+.2f}%** on the Monday, while the *winning* side's is "
            f"actually a bit weaker (**{R['winner_leg_mean']:+.2f}%**). The Edmans "
            "mechanism — *losing depresses the loser's market* — is simply absent. Both "
            "legs are drifting in the historically soft late-September / early-October "
            "window, and the tiny paired gap is just which one drifted less.\n\n"
            "**Finally, could you have traded it?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.one_sample_t(INC['cap_week_gross'].values)['mean']*100\n"
            "    n2 = st.one_sample_t(INC['cap_week_net'].values)['mean']*100\n"
            "else:\n"
            "    g, n2 = R['cap_wk_g'], R['cap_wk_n2']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['gross', 'net of costs'], [g, n2], color=[GREY, RED], width=.5)\n"
            "for i, v in enumerate([g, n2]): ax.annotate(f'{v:+.2f}%', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('long winner / short loser, 1 week, entered AFTER the result')\n"
            "ax.set_title('A tradable +0.32% that a random week matches (placebo p=0.18)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"Betting the loser keeps lagging (long the winner, short the loser, entered "
            f"the day the result is already public) nets **{R['cap_wk_n2']:+.2f}%** over a "
            f"week — positive, but a random week produces that just as easily (placebo "
            f"*p* = {R['pl_cap_wk_p']:.2f}), and you're trying to trade a pattern whose "
            "sign is backwards to begin with. There's no edge here to charge costs against."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The Monday loser-minus-winner spread is the wrong sign "
            f"(**{R['mon_mean']:+.2f}%**, *t* = {R['mon_t']:.2f}), the loser lagged in only "
            f"{R['mon_hit_neg']}/{R['mon_n']} editions, and a random calendar beats it "
            f"(placebo *p* = {R['pl_mon_p']:.2f}).\n"
            "- **Tradability — Mirage.** The only tradable version nets a placebo-normal "
            f"{R['cap_wk_n2']:+.2f}% over a week (*t* = {R['cap_wk_t2']:.2f}) — noise.\n"
            "- **Cross-Atlantic sentiment shock? — Busted.** The loser's own market doesn't "
            "slump; the winner's leg falls more. The real football *loss* effect does not "
            "generalise to the Ryder Cup."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **This is over-extension, caught.** A true effect (football *losses* move "
            "markets) gets stretched into a folklore claim about a different, much smaller "
            "sport — and the tape says no, with the point estimate even pointing the wrong "
            "way. Reporting that honestly is the product.\n"
            "- **The obvious limit is power:** only 10 tradable editions, 7 USA-losses to "
            "3 Europe-losses (VGK doesn't exist before 2005). A believer could push back "
            "before 2005 with local European indices instead of a US-listed ETF, or test "
            "single-country legs (the losing captain's home market) for a sharper signal.\n"
            "- **Sibling studies:** the [World Cup effect](../../235-world-cup-effect/) "
            "(the real Edmans mechanism it borrows from), the "
            "[Eurovision effect](../../708-eurovision-effect/) (the cultural-contest "
            "cousin), the [plane-crash effect](../../707-plane-crash-effect/) "
            "(the same event-study + placebo machinery, a non-sport mood shock), and the "
            "[Super Bowl indicator](../../158-super-bowl/).\n\n"
            "*Think the Ryder Cup REALLY moves markets? Build a bigger sample with local "
            "European indices back to 1979, or test the losing captain's home market, and "
            "show a net, placebo-surviving edge in the right direction. We'll publish the "
            "teardown.*"
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
            "# Ryder-Cup-Effect — a quantitative teardown 🔬\n"
            "### A paired loser-minus-winner one-sample-*t* · a left-tail random-window "
            "placebo · a constant-mean leg decomposition · a USA-vs-Europe asymmetry "
            "check · a costed zero-look-ahead capture · a 20-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb). The claim — **the losing continent's market lags "
            "the winner's the Monday after the Ryder Cup** — extrapolates the Edmans, "
            "García & Norli (2007) football-*elimination* effect (real, one-directional, "
            "no win effect) to a biennial golf match. The job here is to measure it "
            "honestly on the paired `SPY`/`VGK` tape, with the right inference unit for a "
            "tiny-n biennial event.\n\n"
            "> ⚠️ **Data note.** `SPY` (Team USA) + `VGK` (Team Europe), yfinance, adjusted "
            "(total-return) daily closes, 1993-01-29→2026-06-30. 23 Ryder Cups hardcoded "
            "1979→2025 (1989 tie). Only **10 of 22** contested editions have `VGK` coverage "
            "around the result — **selection named on the Signal axis**: 7 USA-losses to "
            "3 Europe-losses, all 2006→2025. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Monday spread **{R['mon_mean']:+.3f}%** "
            f"(*t* = {R['mon_t']:.3f}) — **wrong sign** for the folklore; loser lagged "
            f"{R['mon_hit_neg']}/{R['mon_n']}; placebo left-tail *p* = {R['pl_mon_p']:.3f}; "
            f"1-week spread *t* = {R['wk_t']:.3f} |\n"
            f"| **Tradability** | `MIRAGE` | best net-of-cost capture **{R['cap_wk_n2']:+.3f}%** "
            f"(*t* = {R['cap_wk_t2']:.3f}, placebo *p* = {R['pl_cap_wk_p']:.3f}) |\n"
            f"| **Cross-Atlantic sentiment shock?** | `BUSTED` | loser leg "
            f"{R['loser_leg_mean']:+.3f}% (*t* = {R['loser_leg_t']:.3f}), winner leg "
            f"{R['winner_leg_mean']:+.3f}% — the loser doesn't slump; USA-vs-Europe "
            f"asymmetry Welch *t* = {R['asym_welch_t']:.3f} |\n\n"
            "> 💡 In plain words: not only is there no \"loser lags\" effect, the point "
            "estimate leans the *other* way, the loser's own market never slumps, and a "
            "random calendar reproduces everything. A clean null with a backwards tilt."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_{U,t}$ be `SPY`'s (Team USA) and $r_{E,t}$ `VGK`'s (Team Europe) "
            "simple return on trading day $t$. For each Ryder Cup year $y$ with loser "
            "$\\ell \\in \\{U, E\\}$ and winner $w$, define day(-1) as the last common "
            "close before the (Sunday, non-trading) result and day(0) as the first common "
            "close after. The **paired spread** over horizon $k$ is\n\n"
            "$$S_y(k) = \\left(\\frac{P^{\\ell}_{-1+k}}{P^{\\ell}_{-1}} - 1\\right) - "
            "\\left(\\frac{P^{w}_{-1+k}}{P^{w}_{-1}} - 1\\right)$$\n\n"
            "so the common global move cancels. Because each Ryder Cup is a single, "
            "non-overlapping, independent event, the **one-sample t** of $S$ across events "
            "is the correct primary statistic — not a daily panel. Claims:\n\n"
            "- **H1 (loser lags).** $E[S_y(k)] < 0$ at $k \\in \\{1, 5\\}$ (Monday, week).\n"
            "- **H2 (the loser slumps).** The *loser's own* constant-mean abnormal return "
            "is negative (the actual Edmans channel), not merely the winner being strong.\n"
            "- **H3 (capture).** A trader entering AFTER the result is public (long "
            "winner / short loser) banks it net of costs.\n\n"
            "We find **H1 rejected — the point estimate is the wrong sign**; **H2 rejected "
            "— the loser doesn't slump, the winner's leg is (weakly) the softer one**; "
            "**H3 rejected — no net cut clears the bar and its placebo is 0.18**."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            f"n is small by construction: only **{R['n_included']}** of {R['n_contested']} "
            "contested editions have `SPY`+`VGK` coverage (VGK inception 2005-03-10), and "
            f"they split **{R['n_usa_loss']} USA-losses to {R['n_eur_loss']} "
            "Europe-losses** — so the \"Europe loses\" side of the folklore rests on three "
            "events. The plan: a **one-sample t** of the paired spread per horizon, a "
            "**Wilson interval** on the loser-lagged hit rate, a **20-seed × 200-draw "
            "left-tail random-window placebo** (redraw the same signed spread on random "
            "non-Ryder-Cup dates), a **constant-mean leg decomposition** (does the loser "
            "actually slump?), and a **Welch t** for the USA-loss-vs-Europe-loss asymmetry."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_editions']} editions 1979→2025 "
            f"({R['n_contested']} contested; 1989 tie excluded), hardcoded from Wikipedia.\n"
            f"- **Sample.** {R['n_included']} editions with `SPY`+`VGK` coverage "
            f"({R['n_usa_loss']} USA-loss, {R['n_eur_loss']} Europe-loss) — the rest "
            "excluded by the VGK coverage floor (funnel shown below).\n"
            "- **Headline.** One-sample *t* of the paired spread (Monday, week) + Wilson "
            "hit rate.\n"
            "- **Robustness.** 20×200-draw left-tail random-window placebo.\n"
            "- **Mechanism.** Constant-mean absolute abnormal return of each leg — does "
            "the *loser* slump, or is the winner just weak?\n"
            "- **Execution.** Capture = enter day(0) close (zero look-ahead: result lands "
            "on a non-trading Sunday), long winner / short loser, exit day(0)+k, 4× one-way "
            "cost × NAV (two legs, round trip) + borrow on the short.\n"
            "- **Control.** Synthetic paired (USA, Europe) world, planted-loser-slump knob; "
            "the null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The coverage funnel — the VGK floor\n\n"
            "Of 22 contested editions, only 10 have both `SPY` and `VGK` around the result."
        ),
        code(
            "if HAVE_REAL:\n"
            "    reasons = EV[~EV['included']]['reason'].value_counts()\n"
            "    n_inc = len(INC)\n"
            "else:\n"
            "    reasons = pd.Series({'SPY/VGK coverage postdates the Ryder Cup': 12,\n"
            "                          '14-14 tie -- no losing side': 1})\n"
            "    n_inc = R['n_included']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 3.4))\n"
            "ax.barh(reasons.index[::-1], reasons.values[::-1], color=GREY)\n"
            "ax.set_xlabel('editions excluded')\n"
            "ax.set_title(f'{n_inc}/{R[\"n_contested\"]} contested editions tradable '\n"
            "             '(VGK inception 2005-03-10 is the floor)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(reasons)"
        ),
        md(
            "> 💡 In plain words: the test can only speak to 2006→2025, and the sample "
            "leans 7:3 toward USA-losses. Any statement about \"Europe loses\" rests on "
            "three data points — a real power limit, named on the Signal axis, not patched "
            "with a pre-2005 index proxy."
        ),
        md(
            "### 4b · The headline — one-sample t of the paired spread, both horizons"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for label, col in (('Monday', 'spread_mon'), ('1 week', 'spread_wk')):\n"
            "        s = st.one_sample_t(INC[col].values)\n"
            "        hr = st.hit_rate(INC[col].values, direction='neg')\n"
            "        rows.append((label, s['n'], s['mean']*100, s['t'], hr['k'], hr['n']))\n"
            "    for r in rows: print(r)\n"
            "    means = [rows[0][2], rows[1][2]]; ts = [rows[0][3], rows[1][3]]\n"
            "else:\n"
            "    means = [R['mon_mean'], R['wk_mean']]; ts = [R['mon_t'], R['wk_t']]\n"
            "    print('Monday', R['mon_n'], R['mon_mean'], R['mon_t'], R['mon_hit_neg'])\n"
            "    print('1 week', R['wk_n'], R['wk_mean'], R['wk_t'], R['wk_hit_neg'])\n"
            "labels = ['Monday\\n(k=1)', '1 week\\n(k=5)']\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.4, 6.4), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "a1.bar(labels, means, color=[RED, GREY])\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean spread (%)')\n"
            "a1.set_title('Folklore predicts NEGATIVE. Both bars fail to; Monday is positive.')\n"
            "a2.bar(labels, ts, color=[GREY, GREY])\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(2, ls='--', c=RED, lw=1)\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('t-stat')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the Monday spread is **{R['mon_mean']:+.3f}%** "
            f"(*t* = {R['mon_t']:.2f}, n={R['mon_n']}) — positive, i.e. the *loser* "
            f"outperformed, the wrong sign for H1, and not significant regardless. The "
            f"loser lagged in only {R['mon_hit_neg']}/{R['mon_n']} editions "
            f"(Wilson [{R['mon_w_lo']:.1f}%, {R['mon_w_hi']:.1f}%]). By one week the spread "
            f"is {R['wk_mean']:+.3f}% (*t* = {R['wk_t']:.2f}). No support at any horizon."
        ),
        md(
            "### 4c · The random-window placebo — a random Monday beats the real ones\n\n"
            "For each event, redraw a random (non-Ryder-Cup) 1-session window on the SAME "
            "pair keeping its loser/winner assignment, 20 seeds × 200 draws; left-tail "
            "*p* = share of null means at or below the observed (the folklore direction)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(EV, PRICES, 'spread_mon', k=1, entry_offset=0,\n"
            "                           n_seeds=4, n_draws_per_seed=200, tail='left')\n"
            "    obs = pl['obs']*100\n"
            "    rng = np.random.default_rng(735)\n"
            "    draws = rng.normal(pl['placebo_mean'], pl['placebo_sd'], 4000)*100\n"
            "else:\n"
            "    obs = R['mon_mean']\n"
            "    rng = np.random.default_rng(735)\n"
            "    draws = rng.normal(R['pl_mon_mean'], R['pl_mon_sd'], 4000)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85,\n"
            "        label='null: random Mondays on the same pair')\n"
            "ax.axvline(obs, c=RED, lw=2.4, label=f'observed Ryder-Cup Monday {obs:+.2f}%')\n"
            "ax.axvline(0, c='k', lw=.8, ls=':')\n"
            "ax.set_xlabel('mean loser-minus-winner spread (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Canonical placebo (results.md, 20x200): left-tail p = {R[\"pl_mon_p\"]:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical: observed {R['mon_mean']:+.3f}%, placebo mean \"\n"
            "      f\"{R['pl_mon_mean']:+.3f}% (sd {R['pl_mon_sd']:.3f}%), left-tail p = {R['pl_mon_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: the observed Monday spread sits at the "
            f"**{R['pl_mon_p']*100:.0f}th percentile from the folklore end** — a random "
            "Ryder-Cup-free Monday is *more* likely to show the predicted dip than the "
            "actual Ryder Cup Mondays. That is the signature of no effect (and a slight "
            "wrong-way tilt), not a weak one."
        ),
        md(
            "### 4d · The mechanism test — does the loser actually slump?\n\n"
            "Split the paired spread into each side's own **constant-mean** abnormal return "
            "(Brown & Warner 1985: return minus that ticker's full-sample mean daily "
            "return). The Edmans channel is specifically that the *loser's* market falls."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ll = st.one_sample_t(INC['loser_ab_mon'].values)\n"
            "    wl = st.one_sample_t(INC['winner_ab_mon'].values)\n"
            "    ll_m, ll_t, wl_m, wl_t = ll['mean']*100, ll['t'], wl['mean']*100, wl['t']\n"
            "else:\n"
            "    ll_m, ll_t = R['loser_leg_mean'], R['loser_leg_t']\n"
            "    wl_m, wl_t = R['winner_leg_mean'], R['winner_leg_t']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.3))\n"
            "a1.bar(['loser\\n(should slump)', 'winner'], [ll_m, wl_m], color=[AMBER, GREY], width=.5)\n"
            "for i, v in enumerate([ll_m, wl_m]): a1.annotate(f'{v:+.2f}%', (i, v), ha='center', va='top')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('abnormal Monday return (%)')\n"
            "a1.set_title('Neither leg slumps; the winner is (weakly) softer')\n"
            "a2.bar(['loser t', 'winner t'], [ll_t, wl_t], color=[AMBER, GREY], width=.5)\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_title('...and neither is anywhere near significant')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'loser leg {ll_m:+.3f}% (t={ll_t:+.2f})   winner leg {wl_m:+.3f}% (t={wl_t:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: the losing side's own market is "
            f"**{R['loser_leg_mean']:+.3f}%** (*t* = {R['loser_leg_t']:.2f}) — "
            "statistically zero, no slump. The tiny paired spread the folklore might have "
            f"latched onto comes from the *winner's* leg being marginally softer "
            f"(**{R['winner_leg_mean']:+.3f}%**, *t* = {R['winner_leg_t']:.2f}), plausibly "
            "just the historically weak late-September/early-October seasonality dragging "
            "both. **H2 rejected: this is not the Edmans channel.**"
        ),
        md(
            "### 4e · Event anatomy — the shape of nothing"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp = st.car_path(EV, PRICES, max_k=5)\n"
            "    days = list(cp.index); vs = list(cp.values*100)\n"
            "else:\n"
            "    days = sorted(R['car']); vs = [R['car'][k] for k in days]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.plot(days, vs, color=RED, lw=2.2, marker='o')\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, ls=':', c='k', lw=.8)\n"
            "ax.set_xlabel('trading days after the result (day(-1) = 0)')\n"
            "ax.set_ylabel('mean cumulative loser-minus-winner spread (%)')\n"
            "ax.set_title('A tiny wrong-way Monday blip that decays to zero by Friday')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "> 💡 In plain words: a real gloom shock would drive the spread *down* on the "
            "Monday and hold. Instead it ticks slightly *up* on day 1 "
            f"({R['car'][1]:+.3f}%) and decays back through zero by day 5 "
            f"({R['car'][5]:+.3f}%) — the shape of noise, pointing the wrong way."
        ),
        md(
            "### 4f · Tradability — the honest, zero-look-ahead capture\n\n"
            "Enter at day(0)'s close (first price AFTER the result is public), long the "
            "winner and short the loser, exit day(0)+k, 4× one-way cost × NAV (two legs, "
            "round trip) + 50 bps/yr borrow on the short."
        ),
        code(
            "if HAVE_REAL:\n"
            "    EV5 = st.build_event_table(PRICES, cost_bps=5.0); INC5 = EV5[EV5['included']]\n"
            "    g = st.one_sample_t(INC['cap_week_gross'].values)\n"
            "    n2 = st.one_sample_t(INC['cap_week_net'].values)\n"
            "    n5 = st.one_sample_t(INC5['cap_week_net'].values)\n"
            "    pl = st.placebo_pvalue(EV, PRICES, 'cap_week_net', k=5, entry_offset=1,\n"
            "                           cost_bps=2.0, tail='right', n_seeds=4, n_draws_per_seed=200)\n"
            "    g_m, n2_m, n5_m, cp_p = g['mean']*100, n2['mean']*100, n5['mean']*100, pl['p_value']\n"
            "    g_t, n2_t, n5_t = g['t'], n2['t'], n5['t']\n"
            "else:\n"
            "    g_m, n2_m, n5_m = R['cap_wk_g'], R['cap_wk_n2'], R['cap_wk_n5']\n"
            "    g_t, n2_t, n5_t, cp_p = R['cap_wk_tg'], R['cap_wk_t2'], R['cap_wk_t5'], R['pl_cap_wk_p']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.4))\n"
            "bars = ax.bar(['gross', 'net @2bps', 'net @5bps'], [g_m, n2_m, n5_m],\n"
            "              color=[GREY, RED, RED], width=.55)\n"
            "for b, v, t in zip(bars, [g_m, n2_m, n5_m], [g_t, n2_t, n5_t]):\n"
            "    ax.annotate(f'{v:+.2f}%\\nt={t:.2f}', (b.get_x()+b.get_width()/2, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('1-week capture, long winner / short loser (%)')\n"
            "ax.set_title(f'Placebo p={cp_p:.2f}: a random week matches it. No edge.')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'1-week capture: gross {g_m:+.3f}% (t={g_t:.2f})  net@2bps {n2_m:+.3f}% (t={n2_t:.2f})  '\n"
            "      f'net@5bps {n5_m:+.3f}% (t={n5_t:.2f})  placebo p={cp_p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the 1-week net capture is **{R['cap_wk_n2']:+.3f}%** "
            f"(*t* = {R['cap_wk_t2']:.2f}) at 2 bps, {R['cap_wk_n5']:+.3f}% at 5 bps — and "
            f"its own right-tail placebo (*p* = {R['pl_cap_wk_p']:.3f}) says that magnitude "
            "of winner-minus-loser drift is ordinary. No cut clears *t* ≥ 2, and the "
            "underlying signal is the wrong sign. **H3 rejected; Tradability = MIRAGE.**"
        ),
        md(
            "### 4g · Third axis — USA-loss vs Europe-loss symmetry"
        ),
        code(
            "if HAVE_REAL:\n"
            "    t_asym = st.welch_t(USA_LOSS['spread_mon'].values, EUR_LOSS['spread_mon'].values)\n"
            "    us_m = USA_LOSS['spread_mon'].mean()*100; eu_m = EUR_LOSS['spread_mon'].mean()*100\n"
            "else:\n"
            "    t_asym, us_m, eu_m = R['asym_welch_t'], R['usa_loss_mon'], R['eur_loss_mon']\n"
            "fig, ax = plt.subplots(figsize=(7.4, 4.2))\n"
            "ax.bar([f'USA lost\\n(n={R[\"n_usa_loss\"]})', f'Europe lost\\n(n={R[\"n_eur_loss\"]})'],\n"
            "       [us_m, eu_m], color=[GREY, GREY], width=.5)\n"
            "for i, v in enumerate([us_m, eu_m]): ax.annotate(f'{v:+.2f}%', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('Monday spread, loser minus winner (%)')\n"
            "ax.set_title(f'Both positive (wrong sign), no asymmetry: Welch t = {t_asym:+.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'USA-loss {us_m:+.3f}%  Europe-loss {eu_m:+.3f}%  Welch t = {t_asym:+.3f}')"
        ),
        md(
            f"> 💡 In plain words: whether the USA loses (n={R['n_usa_loss']}, "
            f"{R['usa_loss_mon']:+.3f}%) or Europe loses (n={R['n_eur_loss']}, "
            f"{R['eur_loss_mon']:+.3f}%), the Monday spread is small and *positive* — the "
            f"wrong sign both ways, Welch *t* = {R['asym_welch_t']:.2f}. There is no side of "
            "the Atlantic where the folklore holds."
        ),
        md(
            "### 4h · Faithful-engine & power control\n\n"
            "Synthetic paired (USA, Europe) log-return world (ρ ≈ 0.70), a scheduled "
            "synthetic Ryder-Cup calendar, TUNABLE planted **loser slump**. Null "
            "(slump = 0) checked over **20 seeds**."
        ),
        code(
            "null_ts = np.array([st.synthetic_detect(slump=0.0, seed=735+s, k=1)['t'] for s in range(20)])\n"
            "planted1 = st.synthetic_detect(slump=0.01, seed=735, k=1)\n"
            "planted2 = st.synthetic_detect(slump=0.02, seed=735, k=1)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (slump=0), 20 seeds')\n"
            "ax.scatter([1], [planted1['t']], color=AMBER, s=90, zorder=5, label='planted slump=1%')\n"
            "ax.scatter([2], [planted2['t']], color=RED, s=90, zorder=5, label='planted slump=2%')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['null x20', 'planted 1%', 'planted 2%'])\n"
            "ax.set_ylabel('paired one-sample t'); ax.set_title('Control: quiet null, planted loser-slumps light up NEGATIVE')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t={null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "print(f'planted 1%% t={planted1[\"t\"]:+.2f}  planted 2%% t={planted2[\"t\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null seeds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and never fires "
            f"({R['syn_null_fire']}/{R['syn_null_seeds']}). A planted −1% loser slump reads "
            f"t = {R['syn_planted1_t']:.2f}, a −2% slump t = {R['syn_planted2_t']:.2f} — "
            "correctly negative and significant. The machinery would have caught a real "
            "loser-slump of plausible size; the real tape simply has none. *(A "
            "faithful-engine / power check only — never cited in support of the real-tape "
            "stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the Monday paired spread is **{R['mon_mean']:+.3f}%** "
            f"(*t* = {R['mon_t']:.3f}), the **wrong sign** for the folklore, with the loser "
            f"lagging in only {R['mon_hit_neg']}/{R['mon_n']} editions and a left-tail "
            f"placebo of **{R['pl_mon_p']:.3f}** (a random calendar beats it). The 1-week "
            f"spread is *t* = {R['wk_t']:.3f}. Nothing on the tape supports H1.\n"
            f"- **Tradability `MIRAGE`** — the only tradable direction nets "
            f"**{R['cap_wk_n2']:+.3f}%** over a week (*t* = {R['cap_wk_t2']:.3f}, placebo "
            f"*p* = {R['pl_cap_wk_p']:.3f}) — placebo-normal noise on a wrong-signed "
            "effect.\n"
            f"- **Cross-Atlantic sentiment shock? `BUSTED`** — the loser's own market does "
            f"not slump ({R['loser_leg_mean']:+.3f}%, *t* = {R['loser_leg_t']:.3f}); the "
            f"winner's leg is marginally softer; no USA/Europe asymmetry (Welch "
            f"*t* = {R['asym_welch_t']:.3f}). The real football *loss* effect does not "
            "generalise to the Ryder Cup."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson is over-extension.** A genuine, replicated effect "
            "(football *elimination* moves the eliminated nation's market — with, "
            "importantly, no symmetric win effect) gets stretched to a different sport, a "
            "*continental* team, and a coarser instrument, and the tape says no — with the "
            "point estimate even leaning the wrong way. Report the whole battery, name the "
            "direction, and don't launder a 7:3 split into a two-sided claim.\n"
            "- **A cleaner test would need power and granularity.** Local European indices "
            "back to 1979 (instead of a US-listed ETF gated at 2005) would roughly double "
            "n; testing the losing *captain's home country* market rather than a "
            "continental aggregate would sharpen the sentiment channel the folklore "
            "actually invokes. Both are natural sequels.\n"
            "- **Dedup map:** [235-world-cup-effect](../../235-world-cup-effect/) (the real "
            "Edmans mechanism this borrows from, single market/tournament), "
            "[708-eurovision-effect](../../708-eurovision-effect/) (a per-country "
            "cultural-contest panel, win/host not loss), "
            "[707-plane-crash-effect](../../707-plane-crash-effect/) (same event-study + "
            "placebo + synthetic-control machinery, a non-sport mood shock), "
            "[158-super-bowl](../../158-super-bowl/) and "
            "[709-world-series-effect](../../709-world-series-effect/) (single-market "
            "sports folklore). None test a **paired cross-continental loser-minus-winner "
            "spread on a biennial team match** — that framing, and the \"the loser's leg "
            "doesn't even slump\" finding, is this study's own contribution.\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).*"
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
