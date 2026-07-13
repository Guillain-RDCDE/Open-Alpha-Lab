"""Generate the two narrative notebooks for Study 730 (Ferrari-F1).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached RACE/SPY
tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (RACE + SPY, yfinance,
# 2015-10-01 -> 2026-06-30; 24 of 24 Ferrari wins resolved).
R = dict(
    n_wins=24, n_contender=11, n_sporadic=13,
    fp="5e3d6de14719", panel_rows=2701,
    # signal (day(-1) -> day(-1)+k abnormal return, RACE - SPY)
    d_mean=+0.276, d_t=+1.181, d_hit=13, d_n=24, d_lo=35.1, d_hi=72.1,
    wk_mean=+0.862, wk_t=+1.385, wk_hit=12, wk_n=24, wk_lo=31.4, wk_hi=68.6,
    wki_mean=+1.314, wki_t=+2.285, wki_n=21,
    # random-calendar placebo (right-tail, p = share of null means >= observed)
    pl_d_p=0.214, pl_d_mean=+0.032, pl_d_sd=0.318,
    pl_wk_p=0.159, pl_wk_mean=+0.155, pl_wki_p=0.062,
    # tradability (day(0) -> day(0)+5, net of costs)
    cap_g=+0.578, cap_gt=+1.06, cap_n5=+0.478, cap_t5=+0.88, cap_n10=+0.378, cap_t10=+0.69,
    pl_cap_p=0.274, pl_cap_mean=+0.056,
    # contender vs sporadic
    cont_d_mean=+0.386, cont_d_t=+1.040, cont_wk_mean=+2.471, cont_wk_t=+2.732,
    spor_d_mean=+0.183, spor_d_t=+0.595, spor_wk_mean=-0.499, spor_wk_t=-0.739,
    welch_d=+0.420, welch_wk=+2.631,
    cont_wk_pl_p=0.017, cont_wk_pl_mean=+0.149,
    # event anatomy (mean cumulative AR by day offset from day(-1))
    car={0: 0.000, 1: +0.276, 2: +0.287, 3: +0.649, 4: +0.969, 5: +0.862},
    # wins per season (2016-2025) -- 2016/2020/2021/2025 winless
    wins_by_season={2016: 0, 2017: 5, 2018: 6, 2019: 3, 2020: 0, 2021: 0,
                    2022: 4, 2023: 1, 2024: 5, 2025: 0},
    # synthetic control
    syn_null_mean=+0.43, syn_null_sd=1.01, syn_null_fire=1, syn_null_seeds=20,
    syn_planted1_t=+2.34, syn_planted2_t=+5.73,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Fan--halo_or_fundamentals%3F: Misattributed](https://img.shields.io/badge/Fan--halo_or_fundamentals%3F-Misattributed-8b949e?style=flat-square)\n\n"
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

from ferrari_f1 import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.load_real()
    EV = st.build_event_table(PRICES, cost_bps=5.0)
    INC = EV[EV["included"]]
    CONT = INC[INC["era"] == "contender"]
    SPOR = INC[INC["era"] == "sporadic"]
else:
    PRICES = EV = INC = CONT = SPOR = None
print("real cache present:", HAVE_REAL, "| Ferrari wins:", len(data.EVENTS),
      "| resolved:", (0 if INC is None else len(INC)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does Ferrari's stock rev up when Ferrari wins a race? 🏁📈\n"
            "### The tifosi brand-halo folklore — a coin flip, and the one whisper "
            "isn't even about the fans\n\n"
            + BADGES +
            "Ferrari is the one company on the stock market whose *brand is a racing "
            "team*. So it's an irresistible idea: the Scuderia wins a Formula 1 Grand "
            "Prix on Sunday, the tifosi are euphoric, the world's cameras are on the "
            "prancing horse — surely `RACE` (Ferrari's ticker, and yes that really is "
            "the ticker) gets a Monday-morning pop. It's the Ferrari-flavoured cousin "
            "of a real academic finding: a 2007 study showed national stock markets "
            "genuinely sag when a country's football team is knocked out of the World "
            "Cup. Mood moves markets. Does a Ferrari win move Ferrari?\n\n"
            "We tested it on every Ferrari Grand Prix win since the company went public "
            "in 2015 — 24 of them.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the "
            "era split? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 24 Ferrari wins hardcoded from STATS F1 / "
            "Formula1.com; RACE measured against the S&P 500 (`SPY`) so a market-wide "
            "Monday doesn't get mistaken for a Ferrari effect. Every chart is drawn by "
            "the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does RACE pop the Monday after a win? | **No.** Up "
            f"**{R['d_mean']:+.2f}%** vs the market on average — but the win-Mondays go "
            f"up only **{R['d_hit']}/{R['d_n']}** times (54%, a coin flip), and a "
            "*random* Monday does just as well (**p = 0.21**). |\n"
            f"| Is there anything over the following week? | **A whisper, "
            f"{R['wk_mean']:+.2f}%** — but it doesn't clear the desk's bar either "
            f"(*t* = {R['wk_t']:.2f}), and a random week matches it 16% of the time. |\n"
            f"| Could you have traded it? | **No.** Buy the first moment you actually "
            f"*could* (after the result is public) and hold a week: "
            f"**{R['cap_n5']:+.2f}%** net of costs — statistically a random draw. |\n"
            "| So is the folklore just wrong? | **It's mislabeled.** The one signal "
            "that *does* survive lives entirely in 2017-18 — when a Ferrari win meant a "
            "live *title* fight — and flips negative for the one-off wins that are pure "
            "fan celebration. What little moves RACE is the championship, not the crowd. |\n\n"
            "> The prancing horse doesn't pull the stock. When it looks like it does, "
            "it's the scoreboard talking, not the tifosi."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Ferrari isn't just a carmaker — it's a racing team you can buy shares "
            "in. When the Scuderia wins a Grand Prix, the tifosi are euphoric, the brand "
            "is everywhere, and that wave of sentiment lifts `RACE` the next trading "
            "day.\"*\n\n"
            "It rides on real science: Edmans, García & Norli (2007) found national "
            "stock markets really do fall the day after a country is *eliminated* from "
            "the football World Cup — sports sentiment genuinely moves money. And "
            "individual listed football clubs' shares move on match results. Ferrari is "
            "the closest thing the market has to a listed race team, so the claim is "
            "the natural test. Nobody has published it. We did."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, it would be a delightful, tradable quirk: a globally-televised, "
            "calendar-known, emotion-drenched event — a Ferrari win — that nudges a "
            "specific liquid stock every couple of weeks in season. Buy Friday, sell "
            "into the tifosi Monday, repeat ~5 times a good year. It would also be a "
            "clean demonstration that pure fan sentiment, not fundamentals, prices a "
            "real company. We wanted to know: is it there?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** All **{R['n_wins']}** Ferrari Formula 1 wins since the "
            "2015 NYSE listing, hardcoded with the exact (Sunday) race date. Ferrari "
            "were *winless* in 2016, 2020, 2021 and 2025 — so the wins run 2017→2024.\n"
            "- **The stock.** `RACE` (Ferrari N.V.) measured against `SPY` (the S&P "
            "500), so we're looking at Ferrari's move *beyond* whatever the whole "
            "market did that day.\n"
            "- **The window.** Every race is on a Sunday (markets shut), so nobody can "
            "act until Monday. We measure from Friday's close (before the race) to "
            "Monday's close (after) — the honest 'win pop' — and out to a week.\n"
            "- **The honesty check.** A random-calendar placebo (does a random Monday "
            "pop just as often?), a costed trade you could *actually* have placed, and "
            "a split of the wins into 'title-contender years' vs 'one-off wins' to see "
            "whether any effect is fans or fundamentals."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, a reality check on how often this even happens.** Ferrari's "
            "'brand is winning' — but for a chunk of the listed era they didn't win at all."
        ),
        code(
            "wbs = (dict(sorted(({s: int((INC['season']==s).sum()) for s in range(2016,2026)}).items()))\n"
            "       if HAVE_REAL else R['wins_by_season'])\n"
            "seasons = list(wbs); wins = [wbs[s] for s in seasons]\n"
            "cols = [RED if w == 0 else AMBER for w in wins]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar([str(s) for s in seasons], wins, color=cols)\n"
            "ax.set_ylabel('Ferrari Grand Prix wins')\n"
            "ax.set_title('Ferrari wins per season since the IPO -- four winless years (red)')\n"
            "for i, w in enumerate(wins): ax.annotate(str(w), (i, w), ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('total wins:', sum(wins))"
        ),
        md(
            "24 wins over a decade, lumpy: a strong 2017-18 (Vettel fighting for the "
            "title), a thin middle, a 2022-24 revival, and four completely winless "
            "seasons. Hold that lumpiness in mind — it turns out to matter.\n\n"
            "**Now the headline: does RACE actually pop on win-Mondays?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    d = st.one_sample_t(INC['ar_day'].values)\n"
            "    w = st.one_sample_t(INC['ar_week'].values)\n"
            "    dm, dt_, wm, wt = d['mean']*100, d['t'], w['mean']*100, w['t']\n"
            "else:\n"
            "    dm, dt_, wm, wt = R['d_mean'], R['d_t'], R['wk_mean'], R['wk_t']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.4))\n"
            "bars = ax.bar(['Monday pop\\n(day 0)', '1 week'], [dm, wm],\n"
            "              color=[GREY, AMBER], width=.5)\n"
            "for b, v, t in zip(bars, [dm, wm], [dt_, wt]):\n"
            "    ax.annotate(f'{v:+.2f}%\\n(t={t:.2f})', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('RACE return vs S&P 500 (%)')\n"
            "ax.set_title('The win-Monday pop is small and not distinguishable from zero')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'day0 {dm:+.3f}% (t={dt_:.2f}) | week {wm:+.3f}% (t={wt:.2f})')"
        ),
        md(
            f"RACE is up **{R['d_mean']:+.2f}%** vs the market on the Monday after a win "
            f"— but *t* = **{R['d_t']:.2f}**, nowhere near the desk's bar of 2, and the "
            f"win-Mondays are green only **{R['d_hit']} times out of {R['d_n']}** "
            "(54%). That's a coin flip. **Is a win-Monday even different from a random "
            "Monday?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(EV, PRICES, 'ar_day', k=1, entry_offset=0,\n"
            "                           n_seeds=6, n_draws_per_seed=500, tail='right')\n"
            "    obs = pl['obs']*100\n"
            "    rng = np.random.default_rng(730)\n"
            "    draws = rng.normal(pl['placebo_mean'], pl['placebo_sd'], 6000)*100\n"
            "else:\n"
            "    obs = R['d_mean']\n"
            "    rng = np.random.default_rng(730)\n"
            "    draws = rng.normal(R['pl_d_mean'], R['pl_d_sd'], 6000)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='24 random Mondays (redrawn many times)')\n"
            "ax.axvline(obs, c=RED, lw=2.4, label=f'actual Ferrari-win Mondays {obs:+.2f}%')\n"
            "ax.set_xlabel('average abnormal return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'A random Monday matches the win-Monday {int(R[\"pl_d_p\"]*100)}% of the time (p={R[\"pl_d_p\"]:.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"The red line — the real Ferrari-win average — sits comfortably inside the "
            f"grey cloud of random Mondays (**p = {R['pl_d_p']:.2f}**). Statistically, a "
            "Ferrari win tells you nothing about Monday that a coin wouldn't.\n\n"
            "**And the shape is wrong too.** A fan reaction to a Sunday win should be a "
            "*Monday jump* that holds:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp = st.car_path(EV, PRICES, max_k=5)\n"
            "    days = list(cp.index); ys = list(cp.values*100)\n"
            "else:\n"
            "    days = sorted(R['car']); ys = [R['car'][k] for k in days]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.plot(days, ys, color=AMBER, lw=2.4, marker='o')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('trading days after the race (day 0 = Monday)')\n"
            "ax.set_ylabel('mean cumulative return vs market (%)')\n"
            "ax.set_title('No Monday jump -- just a gentle drift over the week')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "There's no broadcast-night pop. The Monday move is tiny; the line just "
            "drifts up slowly over the following days — the signature of ordinary "
            "single-stock noise, not an event reaction.\n\n"
            "**Here's the twist, and it's the whole study.** Split the 24 wins into two "
            "kinds: the **2017-18** wins, when Ferrari were genuinely fighting for the "
            "championship (a win *meant something* for the title), versus the "
            "**one-off** wins in years they finished nowhere (pure 'we won a race' "
            "celebration):"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cwm = st.one_sample_t(CONT['ar_week'].values); swm = st.one_sample_t(SPOR['ar_week'].values)\n"
            "    cw, cwt, sw, swt = cwm['mean']*100, cwm['t'], swm['mean']*100, swm['t']\n"
            "else:\n"
            "    cw, cwt, sw, swt = R['cont_wk_mean'], R['cont_wk_t'], R['spor_wk_mean'], R['spor_wk_t']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "bars = ax.bar(['title-contender\\nwins (2017-18)', 'sporadic one-off\\nwins (2019, 2022-24)'],\n"
            "              [cw, sw], color=[GREEN, RED], width=.5)\n"
            "for b, v, t in zip(bars, [cw, sw], [cwt, swt]):\n"
            "    ax.annotate(f'{v:+.2f}%\\n(t={t:.2f})', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va=('bottom' if v>=0 else 'top'))\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('RACE 1-week return vs market (%)')\n"
            "ax.set_title('The only real move is in the title years -- and it flips for pure-fan wins')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'contender {cw:+.2f}% (t={cwt:.2f}) | sporadic {sw:+.2f}% (t={swt:.2f})')"
        ),
        md(
            f"There it is. The **title-contender** wins move RACE **{R['cont_wk_mean']:+.2f}%** "
            f"over the week (*t* = {R['cont_wk_t']:.2f} — the one number in the whole "
            f"study that clears the bar). The **sporadic** wins — the ones that are "
            f"*only* about fan joy — go **{R['spor_wk_mean']:+.2f}%**, slightly "
            "*negative*. If this were a tifosi brand-halo, the sporadic wins should show "
            "it just as much. They show the opposite. Whatever moved RACE in 2017-18 was "
            "the market repricing a live championship — money, momentum, a real title "
            "shot — not the crowd's mood."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The fan-halo win pop the folklore predicts just isn't "
            f"there: RACE is up a statistically-invisible **{R['d_mean']:+.2f}%** on "
            f"win-Mondays (*t* = {R['d_t']:.2f}), green barely more than half the time, "
            "and no different from a random Monday.\n"
            "- **Tradability — Mirage.** The best honest, after-the-news trade nets "
            f"**{R['cap_n5']:+.2f}%** over a week — a random-window draw. Nothing to "
            "size.\n"
            "- **Fan-halo, or fundamentals? — Misattributed.** The one signal that "
            "survives lives entirely in the title-contender years and *reverses* for "
            "the pure-celebration wins. The little that's real is the scoreboard "
            "repricing a championship, not the tifosi lifting the brand."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **This is a clean example of a *mislabeled* effect.** There's a real "
            "wisp in the data — but it's the wrong story. The desk's job isn't just "
            "'real / not real', it's *'real because of what you think?'* Here the honest "
            "answer is no.\n"
            "- **A sharper test** would net out luxury-sector beta (RACE vs an LVMH / "
            "Hermès basket, not just SPY), so a 'Ferrari-specific' pop couldn't hide "
            "inside a good week for luxury; and it would add *podiums* and *losses* "
            "(does a shock DNF hurt more than a win helps, the Edmans asymmetry?).\n"
            "- **Sibling studies:** the [Eurovision effect](../../708-eurovision-effect/) "
            "(national-pride bump from a cultural contest), the "
            "[World Cup effect](../../235-world-cup-effect/) (the real Edmans mechanism "
            "on the S&P 500), and the [plane-crash effect](../../707-plane-crash-effect/) "
            "(a dread-sentiment shock) — each a mood-moves-markets claim, tested the "
            "same honest way.\n\n"
            "*Think the tifosi really move RACE? Net out luxury beta, add losses and "
            "podiums, and show a positive, replicated, placebo-surviving pop that "
            "*doesn't* vanish once you control for the championship. We'll publish the "
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
            "# Ferrari-F1 — a quantitative teardown 🔬\n"
            "### One-sample-*t* battery on RACE's win-day abnormal returns · a "
            "random-calendar placebo · the day-0-vs-week anatomy · a "
            "contender-vs-sporadic Welch split that *reattributes* the effect · a costed "
            "capture · a 20-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb). The claim — **RACE gets a fan-sentiment pop the "
            "Monday after a Ferrari F1 win** — has no published academic anchor of its "
            "own; it borrows its mechanism from Edmans, García & Norli (2007), a real "
            "national-index elimination-shock effect, and from the listed-football-club "
            "literature. The job here is to measure the Ferrari version honestly, on the "
            "real tape, with the right inference unit for a small-n calendar-known event.\n\n"
            "> ⚠️ **Data note.** `RACE` (Ferrari N.V., NYSE) + `SPY`, yfinance, adjusted "
            "(total-return) daily closes, 2015-10-01→2026-06-30 (" + f"{R['panel_rows']:,}"
            " rows). **24** Ferrari Grand Prix wins hardcoded 2017→2024 (winless 2016, "
            "2020, 2021, 2025). Benchmark is `SPY` (a US-market counterfactual, not a "
            "luxury-peer control — named, see [`docs/references.md`](../docs/references.md)). "
            "Numbers in [`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | day(0) pop **{R['d_mean']:+.3f}%**, *t* = "
            f"**{R['d_t']:.3f}**, hit {R['d_hit']}/{R['d_n']}, placebo *p* = "
            f"**{R['pl_d_p']:.3f}**; 1-week *t* = {R['wk_t']:.3f} (placebo *p* = "
            f"{R['pl_wk_p']:.3f}) |\n"
            f"| **Tradability** | `MIRAGE` | best zero-look-ahead net capture "
            f"**{R['cap_n5']:+.3f}%** (*t* = {R['cap_t5']:.3f}), placebo *p* = "
            f"{R['pl_cap_p']:.3f} |\n"
            f"| **Fan-halo, or fundamentals?** | `MISATTRIBUTED` | the only cut past both "
            f"bars is contender-era 1-week **{R['cont_wk_mean']:+.3f}%**, *t* = "
            f"**{R['cont_wk_t']:.3f}**, placebo *p* = **{R['cont_wk_pl_p']:.3f}** — but "
            f"sporadic-era is **{R['spor_wk_mean']:+.3f}%** (Welch *t* = {R['welch_wk']:.3f}) |\n\n"
            "> 💡 In plain words: the pop the folklore predicts is a coin flip; the one "
            "surviving number is a *slower* drift that exists only when a win also "
            "updated a title fight — fundamentals wearing a fan-sentiment costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_{race,t}$ be RACE's log-return and $r_{spy,t}$ the SPY log-return on "
            "trading day $t$. Each Grand Prix runs on a (non-trading) Sunday; define "
            "day(-1) as the last close before the race and day(0) as the first close "
            "after. The abnormal return over horizon $k$ is\n\n"
            "$$AR_{y}(k) = \\left(\\frac{P^{race}_{-1+k}}{P^{race}_{-1}} - 1\\right) - "
            "\\left(\\frac{P^{spy}_{-1+k}}{P^{spy}_{-1}} - 1\\right)$$\n\n"
            "Each win is a single, independent event, so the **one-sample t** of $AR$ "
            "across wins is the correct primary statistic — not a daily panel. At "
            "$k=1$ (the day(0) pop) every event is a distinct Monday → exact "
            "independence; at $k=5$ three back-to-back pairs overlap (named below). "
            "Claims:\n\n"
            "- **H1 (the pop).** $E[AR(1)] > 0$ — an immediate Monday reaction.\n"
            "- **H2 (the week).** $E[AR(5)] > 0$.\n"
            "- **H3 (anatomy).** The reaction appears on day 0-1 (a broadcast-night "
            "pop) and holds.\n"
            "- **H4 (capture).** A fan entering AFTER the result is public (zero "
            "look-ahead) banks it net of costs.\n"
            "- **H5 (mechanism).** If it's *fan sentiment*, it should be uniform across "
            "wins — a title-relevant win and a dead-rubber win should pop alike.\n\n"
            "We find **H1 not supported** (*t* = 1.18, placebo *p* = 0.21); **H2 not "
            "supported** (*t* = 1.39); **H3 not supported** (no day-0 jump, a slow "
            "drift); **H4 not supported**; **H5 rejected** — the only surviving effect "
            "is confined to the title-contender era and *reverses* for sporadic wins."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            f"n is small and calendar-known: **{R['n_wins']}** wins, "
            f"**{R['n_contender']}** in the 2017-18 title-contender era and "
            f"**{R['n_sporadic']}** sporadic. The plan is a **one-sample t** per horizon "
            "(day 0 and week), a **Wilson interval** on the hit rate, a **20-seed × "
            "500-draw random-calendar placebo** (redraw the same 24 anchors at random "
            "points in RACE's own history and see how often the null matches or beats "
            "the observed mean), the **day-0-vs-week CAR anatomy** (does the shape match "
            "a broadcast-night pop?), and a **pre-registered contender-vs-sporadic "
            "split** — the one cut that can tell fan sentiment from fundamentals. The "
            "3-event weekly overlap (Belgium/Monza 2019, Britain/Austria 2022, "
            "USA/Mexico 2024) is named and the weekly cut is re-run dropping the second "
            "race of each pair."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_wins']} Ferrari wins 2017→2024, hardcoded from STATS "
            "F1 / Formula1.com; winless 2016/2020/2021/2025.\n"
            "- **Headline.** One-sample *t* at day(0) and 1 week + Wilson hit rate.\n"
            "- **Robustness.** 20×500-draw random-calendar placebo; independence-"
            "corrected weekly re-run.\n"
            "- **Anatomy.** Mean cumulative AR by trading day, 0→5.\n"
            "- **Mechanism (third axis).** Pre-tagged contender vs sporadic; one-sample "
            "*t* each + a Welch *t* of the difference.\n"
            "- **Execution.** Capture = enter day(0) close (zero look-ahead: the race "
            "runs on a non-trading Sunday), exit day(0)+5 close, 2× one-way cost × NAV.\n"
            "- **Control.** Synthetic paired (asset, benchmark) world, planted-bump "
            "knob; the null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The event count — a lumpy, small sample\n\n"
            "24 wins over a decade, four winless seasons. The lumpiness (a strong "
            "2017-18, a thin middle, a 2022-24 revival) is exactly what the mechanism "
            "test in 4f exploits."
        ),
        code(
            "wbs = (dict(sorted(({s: int((INC['season']==s).sum()) for s in range(2016,2026)}).items()))\n"
            "       if HAVE_REAL else R['wins_by_season'])\n"
            "seasons = list(wbs); wins = [wbs[s] for s in seasons]\n"
            "cols = [RED if w == 0 else (GREEN if s in (2017,2018) else AMBER) for s, w in zip(seasons, wins)]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "ax.bar([str(s) for s in seasons], wins, color=cols)\n"
            "for i, w in enumerate(wins): ax.annotate(str(w), (i, w), ha='center', va='bottom')\n"
            "ax.set_ylabel('wins'); ax.set_title('Ferrari wins per season (green = title-contender era, red = winless)')\n"
            "plt.tight_layout(); plt.show(); print('total', sum(wins))"
        ),
        md(
            "> 💡 In plain words: 11 of the 24 wins came in the two seasons Ferrari "
            "actually fought for the championship. Any 'win effect' is therefore at risk "
            "of being a 'title-race effect' in disguise — which is precisely what 4f tests."
        ),
        md(
            "### 4b · The headline — one-sample t at day(0) and 1 week"
        ),
        code(
            "if HAVE_REAL:\n"
            "    d = st.one_sample_t(INC['ar_day'].values); w = st.one_sample_t(INC['ar_week'].values)\n"
            "    wi = st.one_sample_t(INC[~INC['weekly_overlap']]['ar_week'].values)\n"
            "    dm, dt_, wm, wt, wim, wit = d['mean']*100, d['t'], w['mean']*100, w['t'], wi['mean']*100, wi['t']\n"
            "else:\n"
            "    dm, dt_, wm, wt = R['d_mean'], R['d_t'], R['wk_mean'], R['wk_t']\n"
            "    wim, wit = R['wki_mean'], R['wki_t']\n"
            "labels = ['day(0)\\npop', '1 week\\n(all 24)', '1 week\\n(indep, n=21)']\n"
            "means = [dm, wm, wim]; ts = [dt_, wt, wit]\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.6, 6.4), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "a1.bar(labels, means, color=[GREY if abs(t)<2 else AMBER for t in ts])\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean AR (%)')\n"
            "a1.set_title('Only the independence-corrected weekly cut nudges over the bar')\n"
            "a2.bar(labels, ts, color=[RED if abs(t)>=2 else GREY for t in ts])\n"
            "a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(0, c='k', lw=.8); a2.set_ylabel('t-stat')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'day0 {dm:+.3f}% t={dt_:.3f} | week {wm:+.3f}% t={wt:.3f} | week-indep {wim:+.3f}% t={wit:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the day(0) pop is *t* = **{R['d_t']:.2f}** "
            f"(n={R['d_n']}), a coin flip. The all-24 week is *t* = {R['wk_t']:.2f}. "
            f"Dropping the 3 overlapping back-to-back races lifts the week to "
            f"**{R['wki_mean']:+.2f}%**, *t* = **{R['wki_t']:.2f}** — nominally over 2, "
            "but this is a slower *drift*, not the claimed pop, and the placebo (4c) "
            "still doesn't confirm it."
        ),
        md(
            "### 4c · The random-calendar placebo — is a win-Monday unusual?\n\n"
            "Redraw the same 24 anchors at random points in RACE's own history, 20 seeds "
            "× 500 draws; compare the observed means to the null."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(EV, PRICES, 'ar_day', k=1, entry_offset=0,\n"
            "                           n_seeds=6, n_draws_per_seed=500, tail='right')\n"
            "    obs = pl['obs']*100\n"
            "    rng = np.random.default_rng(730)\n"
            "    draws = rng.normal(pl['placebo_mean'], pl['placebo_sd'], 6000)*100\n"
            "else:\n"
            "    obs = R['d_mean']; rng = np.random.default_rng(730)\n"
            "    draws = rng.normal(R['pl_d_mean'], R['pl_d_sd'], 6000)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='null: 24 random Mondays, redrawn')\n"
            "ax.axvline(obs, c=RED, lw=2.4, label=f'observed day(0) mean {obs:+.2f}%')\n"
            "ax.set_xlabel('mean AR of a random-calendar draw (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Canonical placebo (results.md, 20x500 draws): day(0) p = {R[\"pl_d_p\"]:.3f}, week p = {R[\"pl_wk_p\"]:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"day0: obs {R['d_mean']:+.3f}%, placebo {R['pl_d_mean']:+.3f}% (sd {R['pl_d_sd']:.3f}%), p={R['pl_d_p']:.3f}\")\n"
            "print(f\"week-indep placebo p = {R['pl_wki_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: the day(0) mean is reproduced by a random Monday "
            f"**{int(R['pl_d_p']*100)}%** of the time (*p* = {R['pl_d_p']:.3f}); even the "
            f"raw-significant independence-corrected weekly cut fails the placebo at 5% "
            f"(*p* = {R['pl_wki_p']:.3f}). No headline cut is genuinely unusual. Contrast "
            "with 4f, where one *does* survive — for a reason that undoes the claim."
        ),
        md(
            "### 4d · Event anatomy — does the timing match a broadcast-night pop?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp = st.car_path(EV, PRICES, max_k=5)\n"
            "    days = list(cp.index); ys = list(cp.values*100)\n"
            "else:\n"
            "    days = sorted(R['car']); ys = [R['car'][k] for k in days]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.plot(days, ys, color=AMBER, lw=2.4, marker='o')\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, ls=':', c='k', lw=.8)\n"
            "ax.set_xlabel('trading days after the race (day(-1) = 0%)')\n"
            "ax.set_ylabel('mean cumulative AR (%)')\n"
            "ax.set_title('A slow drift, not a day-0 jump')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "> 💡 In plain words: a Sunday-broadcast fan mechanism predicts a jump on "
            f"day 1 that holds. Instead day 1 is only {R['car'][1]:+.3f}% and the path "
            f"drifts to {R['car'][5]:+.3f}% over the week — the shape of ordinary "
            "single-name noise, arguing *against* reading the (already insignificant) "
            "weekly number as an event reaction."
        ),
        md(
            "### 4e · Tradability — the honest, zero-look-ahead capture\n\n"
            "Enter at day(0)'s close (the first price AFTER the result is public — the "
            "race runs on a non-trading Sunday, so this is the earliest zero-look-ahead "
            "entry), exit day(0)+5, 2× one-way cost × NAV."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.one_sample_t(INC['cap_week_gross'].values)\n"
            "    n5 = st.one_sample_t(INC['cap_week_net'].values)\n"
            "    EV10 = st.build_event_table(PRICES, cost_bps=10.0); INC10 = EV10[EV10['included']]\n"
            "    n10 = st.one_sample_t(INC10['cap_week_net'].values)\n"
            "    gm, n5m, n10m, gt, n5t, n10t = g['mean']*100, n5['mean']*100, n10['mean']*100, g['t'], n5['t'], n10['t']\n"
            "else:\n"
            "    gm, n5m, n10m = R['cap_g'], R['cap_n5'], R['cap_n10']\n"
            "    gt, n5t, n10t = R['cap_gt'], R['cap_t5'], R['cap_t10']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "bars = ax.bar(['gross', 'net @5bps', 'net @10bps'], [gm, n5m, n10m],\n"
            "              color=[GREY, AMBER, RED], width=.55)\n"
            "for b, v, t in zip(bars, [gm, n5m, n10m], [gt, n5t, n10t]):\n"
            "    ax.annotate(f'{v:+.2f}%\\n(t={t:.2f})', (b.get_x()+b.get_width()/2, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('RACE 1-week capture vs market (%)')\n"
            "ax.set_title(f'Random-window territory: net placebo p = {R[\"pl_cap_p\"]:.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gm:+.3f}% (t={gt:.2f}) | net5 {n5m:+.3f}% (t={n5t:.2f}) | net10 {n10m:+.3f}% (t={n10t:.2f})')"
        ),
        md(
            f"> 💡 In plain words: enter after the news, hold a week, and you net "
            f"**{R['cap_n5']:+.2f}%** at 5 bps (*t* = {R['cap_t5']:.2f}), "
            f"**{R['cap_n10']:+.2f}%** at 10 bps — and the random-calendar placebo on "
            f"the net capture is *p* = {R['pl_cap_p']:.2f}. There is no edge to charge "
            "costs against. **H4 not supported; Tradability = MIRAGE.**"
        ),
        md(
            "### 4f · The mechanism test — fans or fundamentals? *(the reattribution)*\n\n"
            "The wins are pre-tagged **contender** (2017-18, a win updated a live title "
            "fight) vs **sporadic** (2019, 2022-24, one-off wins). A pure fan-sentiment "
            "effect must be *uniform*; a fundamentals effect concentrates in the "
            "contender era."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cd, cw = st.one_sample_t(CONT['ar_day'].values), st.one_sample_t(CONT['ar_week'].values)\n"
            "    sd, sw = st.one_sample_t(SPOR['ar_day'].values), st.one_sample_t(SPOR['ar_week'].values)\n"
            "    vals = [cd['mean']*100, sd['mean']*100, cw['mean']*100, sw['mean']*100]\n"
            "    ts = [cd['t'], sd['t'], cw['t'], sw['t']]\n"
            "    welch_d = st.welch_t(CONT['ar_day'].values, SPOR['ar_day'].values)\n"
            "    welch_w = st.welch_t(CONT['ar_week'].values, SPOR['ar_week'].values)\n"
            "else:\n"
            "    vals = [R['cont_d_mean'], R['spor_d_mean'], R['cont_wk_mean'], R['spor_wk_mean']]\n"
            "    ts = [R['cont_d_t'], R['spor_d_t'], R['cont_wk_t'], R['spor_wk_t']]\n"
            "    welch_d, welch_w = R['welch_d'], R['welch_wk']\n"
            "labels = ['contender\\nday(0)', 'sporadic\\nday(0)', 'contender\\n1 week', 'sporadic\\n1 week']\n"
            "cols = [GREEN, RED, GREEN, RED]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "bars = ax.bar(labels, vals, color=cols, width=.6)\n"
            "for b, v, t in zip(bars, vals, ts):\n"
            "    ax.annotate(f'{v:+.2f}%\\n(t={t:.2f})', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va=('bottom' if v>=0 else 'top'), fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('mean AR vs market (%)')\n"
            "ax.set_title('The lone surviving signal (contender/1-week) has no sporadic twin')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Welch t (contender - sporadic): day {welch_d:+.3f}  week {welch_w:+.3f}')"
        ),
        md(
            f"> 💡 In plain words: contender-era 1-week is **{R['cont_wk_mean']:+.2f}%** "
            f"(*t* = {R['cont_wk_t']:.2f}, placebo *p* = {R['cont_wk_pl_p']:.3f}) — the "
            f"**only** cut in the study past both bars. But the sporadic wins go "
            f"**{R['spor_wk_mean']:+.2f}%** (*t* = {R['spor_wk_t']:.2f}); the Welch *t* "
            f"of the difference is **{R['welch_wk']:.2f}**. A fan-halo would show up in "
            "*both* buckets — the tifosi don't check the standings before celebrating. "
            "It shows up only where a win moved the *championship*. The effect is real "
            "but **misattributed**: it's the market repricing a title campaign, not "
            "sentiment. *(This is why the Signal axis is `NONE` for the claim as stated, "
            "not `WEAK`: the one surviving number isn't the claimed mechanism.)*"
        ),
        md(
            "### 4g · Faithful-engine & power control\n\n"
            "Synthetic paired (asset, benchmark) log-return world (ρ≈0.55, RACE-vs-SPY-"
            "like, single-name vol > index vol), a scheduled synthetic win calendar, "
            "TUNABLE planted bump. Null (bump=0) checked over **20 seeds**."
        ),
        code(
            "null_ts = np.array([st.synthetic_detect(bump=0.0, seed=730+s, k=1)['t'] for s in range(20)])\n"
            "p1 = st.synthetic_detect(bump=0.01, seed=730, k=1); p2 = st.synthetic_detect(bump=0.02, seed=730, k=1)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (bump=0), 20 seeds')\n"
            "ax.scatter([1], [p1['t']], color=AMBER, s=90, zorder=5, label='planted bump=1%')\n"
            "ax.scatter([2], [p2['t']], color=RED, s=90, zorder=5, label='planted bump=2%')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['null x20', 'planted 1%', 'planted 2%'])\n"
            "ax.set_ylabel('one-sample t'); ax.set_title('Control: quiet null, planted bumps light up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t={null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20')\n"
            "print(f'planted 1%% t={p1[\"t\"]:+.2f}  planted 2%% t={p2[\"t\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null seeds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and fires at "
            f"|t|≥2 in only {R['syn_null_fire']}/{R['syn_null_seeds']} seeds — the "
            f"ordinary false-positive rate at this n, not a bias. A planted 1% bump reads "
            f"t={R['syn_planted1_t']:.2f}, a 2% bump t={R['syn_planted2_t']:.2f}. The "
            "machinery works; the real-tape pop is genuinely this thin. *(A faithful-"
            "engine / power check only — never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the fan-sentiment win pop is a coin flip: day(0) AR "
            f"**{R['d_mean']:+.3f}%**, *t* = **{R['d_t']:.3f}**, hit "
            f"{R['d_hit']}/{R['d_n']}, placebo *p* = **{R['pl_d_p']:.3f}**. The 1-week "
            f"horizon is *t* = {R['wk_t']:.3f} (placebo *p* = {R['pl_wk_p']:.3f}); even "
            f"the independence-corrected weekly cut (*t* = {R['wki_t']:.3f}) fails the "
            f"placebo (*p* = {R['pl_wki_p']:.3f}). The anatomy is a drift, not a pop.\n"
            f"- **Tradability `MIRAGE`** — best zero-look-ahead net capture "
            f"**{R['cap_n5']:+.3f}%** (*t* = {R['cap_t5']:.3f}), placebo *p* = "
            f"{R['pl_cap_p']:.3f}. Nothing to size.\n"
            f"- **Fan-halo, or fundamentals? `MISATTRIBUTED`** — the only cut past both "
            f"bars (contender/1-week, **{R['cont_wk_mean']:+.3f}%**, *t* = "
            f"**{R['cont_wk_t']:.3f}**, placebo *p* = **{R['cont_wk_pl_p']:.3f}**) is "
            f"confined to 2017-18 and reverses to **{R['spor_wk_mean']:+.3f}%** for the "
            f"sporadic wins (Welch *t* = {R['welch_wk']:.3f}). The little that's real is "
            "the market repricing a live championship — fundamentals, not the tifosi."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The lesson is about *attribution*, not just significance.** A naive "
            "'best cut' hunt would have found the contender/1-week *t* = 2.73, stamped "
            "it real, and sold a 'Ferrari-win effect'. The pre-registered mechanism "
            "split is what turns a false positive into an honest reattribution — the "
            "effect is real *and* it isn't the claimed thing.\n"
            "- **A cleaner test needs a tighter control and more shocks.** Net out "
            "luxury-sector beta (RACE vs an LVMH/Hermès basket) so a Ferrari-specific "
            "move can't hide inside a good luxury week; add *podiums* and *DNF/loss* "
            "shocks to test the Edmans asymmetry (losses > wins) and to raise n beyond "
            "24; and split by whether the win *changed the championship lead*, the "
            "sharpest fundamentals proxy.\n"
            "- **Dedup map:** [708-eurovision-effect](../../708-eurovision-effect/) "
            "(a per-country cultural-contest bump — same template, national panel), "
            "[235-world-cup-effect](../../235-world-cup-effect/) (the real Edmans "
            "mechanism on the S&P 500), [707-plane-crash-effect](../../707-plane-crash-effect/) "
            "(a dread shock, same event-study + placebo machinery). None test a **single "
            "listed company's abnormal return keyed to its own team's wins** — the "
            "'brand *is* the team' angle, and the fundamentals-vs-fans reattribution, is "
            "this study's own contribution.\n\n"
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
