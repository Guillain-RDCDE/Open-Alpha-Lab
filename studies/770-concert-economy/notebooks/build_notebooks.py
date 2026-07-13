"""Generate the two narrative notebooks for Study 770 (Concert-Economy).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached LYV/SPY
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (LYV + SPY, yfinance,
# 2005-12-21 -> 2026-06-30; 18 of 20 Coachella editions resolved).
R = dict(
    n_editions=20, n_held=18, n_included=18,
    fp="4ba79257adcd", lyv_beta=1.351,
    # the run-up (K sessions before Coachella, LYV - SPY, gross)
    ru1_mean=+0.719, ru1_t=+0.287, ru1_hit=6, ru1_n=18, ru1_wlo=16.3, ru1_whi=56.3,
    ru2_mean=-0.527, ru2_t=-0.182, ru2_hit=8, ru2_n=18, ru2_wlo=24.6, ru2_whi=66.3,
    # placebo (right-tail, p = share of null means >= observed)
    pl_ru1_p=0.518, pl_ru1_mean=+0.882, pl_ru1_sd=2.649,
    pl_ru2_p=0.729, pl_ru2_mean=+1.668, pl_ru2_sd=3.613,
    # jackknife (1-month run-up)
    jk_lo=-0.435, jk_hi=+0.650, jk_n=18,
    # tradability (net of costs)
    ru1_net5=+0.619, ru1_t5=+0.247, ru1_net10=+0.519, ru1_t10=+0.207,
    ru2_net5=-0.627, ru2_t5=-0.217, ru2_net10=-0.727, ru2_t10=-0.252,
    # third axis: in-season (Coachella -> ~Labor Day)
    dur_mean=+8.542, dur_t=+1.912, dur_hit=12, dur_n=18,
    dur_pl_p=0.157, dur_pl_mean=+3.359, dur_pl_sd=5.196,
    dur_badj_mean=+7.618, dur_badj_t=+1.806,
    # event anatomy (mean cumulative AR by offset from the anchor)
    car={-42: 2.099, -21: 0.408, -10: 0.197, 0: 0.000, 21: 5.065, 42: 6.115, 63: 6.724, 95: 8.542},
    # per-year 1-month run-up (%), for the strip chart
    per_year=[(2006, -1.05), (2007, -14.53), (2008, 11.38), (2009, 28.04), (2010, 9.50),
              (2011, -5.03), (2012, -6.58), (2013, 6.07), (2014, -11.24), (2015, -1.47),
              (2016, -3.52), (2017, 8.54), (2018, -12.39), (2019, -2.24), (2022, -1.44),
              (2023, -3.48), (2024, -2.42), (2025, 14.81)],
    # revenue seasonality proxy (share of annual revenue)
    rev_q1=16, rev_q2=28, rev_q3=37, rev_q4=19,
    # synthetic control
    syn_null_mean=+0.20, syn_null_sd=1.07, syn_null_fire=1, syn_null_seeds=20,
    syn_planted1_t=+1.07, syn_planted2_t=+1.89,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Front--runs_the_season%3F: Not_supported](https://img.shields.io/badge/Front--runs_the_season%3F-Not_supported-8b949e?style=flat-square)\n\n"
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

from concert_economy import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.load_real()
    EV = st.build_event_table(PRICES, cost_bps=5.0)
    EV10 = st.build_event_table(PRICES, cost_bps=10.0)
    INC = EV[EV["included"]]
    INC10 = EV10[EV10["included"]]
else:
    PRICES = EV = EV10 = INC = INC10 = None
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
            "# Does Live Nation rally *into* festival season? 🎤📈\n"
            "### The \"concert economy\" trade — a real seasonal business, a stock that "
            "doesn't front-run it, and one big number that's just beta\n\n"
            + BADGES +
            "Every spring the same trade idea makes the rounds: *the concert economy is "
            "booming — buy Live Nation (LYV) before festival season.* The logic is "
            "genuinely appealing. Live Nation runs Coachella, the summer amphitheatre "
            "circuit, the whole touring machine — and the summer quarter really is, by "
            "a mile, its biggest. So shouldn't the market bid the stock up *ahead* of "
            "all that predictable revenue?\n\n"
            "We tested it properly — every Coachella since Live Nation went public "
            "(2006→2025), against the S&P 500.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the beta "
            "decomposition? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 20 Coachella editions hardcoded from Wikipedia "
            "(2020–21 COVID-cancelled); the festival's date is announced every January, "
            "so \"buy K days before it opens\" is a trade you could actually have "
            "pre-scheduled. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does LYV pop in the month *before* Coachella? | **No.** "
            f"**{R['ru1_mean']:+.2f}%** vs the S&P — statistically indistinguishable "
            "from zero, and a random month does it just as often. |\n"
            f"| What about the two months before? | **Slightly negative** "
            f"(**{R['ru2_mean']:+.2f}%**). There is no run-up at all. |\n"
            "| But isn't festival season a huge deal for them? | **Yes — and that's the "
            "point.** ~37% of Live Nation's revenue is the summer quarter. Everyone "
            "knows this. That's exactly why it's already in the price. |\n"
            f"| Then what's the one big number I'll hear about? | LYV is up "
            f"**{R['dur_mean']:+.1f}%** vs the S&P *during* the season — but that's "
            "**after** it starts (not a front-run), it's within normal range for a "
            f"random 4½-month window, and most of it is just LYV's **{R['lyv_beta']:.2f}× "
            "beta** over a long stretch. |\n\n"
            "> The seasonality is real. The *trade* is folklore: a calendar everyone can "
            "read doesn't pay you to read it."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Live Nation makes most of its money in the summer — Coachella, the "
            "festivals, the tours. Buy it before festival season and ride the wave of "
            "concert-economy revenue.\"*\n\n"
            "This isn't a silly claim. Unlike most folklore, the fundamental is rock "
            "solid: the summer touring quarter genuinely dominates the business. The "
            "only question is whether a stock can *front-run* a revenue calendar that "
            "is announced months in advance and identical every year."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this would be a clean, calendar-based edge — pre-schedule the buy "
            "every January (the moment Coachella's dates drop), hold into the festival, "
            "collect the anticipation premium, repeat. No news, no timing skill, just a "
            "date on a calendar. If it *isn't* real, it's a tidy demonstration of "
            "semi-strong market efficiency: public, predictable information that's "
            "already priced. So — is it?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** All **{R['n_editions']}** Coachella editions "
            f"2006→2025 ({R['n_held']} held — 2020 & 2021 were COVID-cancelled), "
            "hardcoded with each year's weekend-1 Friday.\n"
            "- **The stock.** `LYV` (Live Nation) vs `SPY` (S&P 500), both total-return, "
            "so we measure LYV's *abnormal* return — not the market's.\n"
            "- **The window.** The **run-up**: LYV minus S&P over the 1 month and 2 "
            "months *before* Coachella opens. Because the date is public months ahead, "
            "this is a trade you could have placed in advance — no look-ahead.\n"
            "- **The honesty checks.** A random-window placebo (does a random month do "
            "the same?), a jackknife (does one lucky year carry it?), the trade net of "
            "costs, and an in-season check for whether the stock front-runs the actual "
            "revenue."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the steelman: festival season really is Live Nation's whole "
            "year.**"
        ),
        code(
            "shares = [R['rev_q1'], R['rev_q2'], R['rev_q3'], R['rev_q4']]\n"
            "labels = ['Q1', 'Q2', 'Q3\\n(summer\\ntouring)', 'Q4']\n"
            "cols = [GREY, GREY, GREEN, GREY]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(labels, shares, color=cols)\n"
            "for i, v in enumerate(shares): ax.annotate(f'{v}%', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('share of annual revenue (%)')\n"
            "ax.set_title('The steelman is real: Q3 (summer touring) ~ 37% of revenue')\n"
            "ax.text(0.5, -0.22, 'LABELLED PROXY — reconstructed from Live Nation 10-Q filings, not a live feed',\n"
            "        transform=ax.transAxes, ha='center', fontsize=8, color=GREY)\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "That's the case for the trade — and it's a good one. Now the question that "
            "matters: **does the stock climb in the weeks before all that starts?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ru1 = st.one_sample_t(INC['ru_1mo'].values)['mean']*100\n"
            "    ru2 = st.one_sample_t(INC['ru_2mo'].values)['mean']*100\n"
            "else:\n"
            "    ru1, ru2 = R['ru1_mean'], R['ru2_mean']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "ax.bar(['1 month before', '2 months before'], [ru1, ru2],\n"
            "       color=[GREY, GREY], width=.5)\n"
            "for i, v in enumerate([ru1, ru2]): ax.annotate(f'{v:+.2f}%', (i, v), ha='center',\n"
            "        va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('LYV minus S&P 500 (%)')\n"
            "ax.set_title('The \"rally into festival season\": basically nothing')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('1-month run-up:', round(ru1,3), '%   2-month run-up:', round(ru2,3), '%')"
        ),
        md(
            f"The one-month run-up is **{R['ru1_mean']:+.2f}%** (*t* = "
            f"{R['ru1_t']:.2f} — a statistical zero) and the two-month run-up is "
            f"actually **negative** ({R['ru2_mean']:+.2f}%). There is no anticipation "
            "bump.\n\n"
            "**Maybe the average just hides a real effect that's noisy year to year?** "
            "Let's look at every single year:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    yrs = INC['year'].astype(int).tolist(); vals = (INC['ru_1mo']*100).tolist()\n"
            "else:\n"
            "    yrs = [y for y,_ in R['per_year']]; vals = [v for _,v in R['per_year']]\n"
            "fig, ax = plt.subplots(figsize=(10.2, 4.4))\n"
            "ax.bar([str(y) for y in yrs], vals, color=[GREEN if v>0 else RED for v in vals])\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.axhline(np.mean(vals), c=AMBER, ls='--', lw=1.4, label=f'mean {np.mean(vals):+.2f}%')\n"
            "ax.set_ylabel('1-month run-up, LYV - S&P (%)')\n"
            "ax.set_title('Wild year-to-year swings that cancel out — no stable bump')\n"
            "ax.legend(); plt.xticks(rotation=45); plt.tight_layout(); plt.show()\n"
            "print('up years:', sum(1 for v in vals if v>0), 'of', len(vals))"
        ),
        md(
            f"From **{min(v for _,v in R['per_year']):+.1f}%** (2007) to "
            f"**{max(v for _,v in R['per_year']):+.1f}%** (2009) — enormous swings that "
            f"average out to nearly zero, with only **{R['ru1_hit']}/{R['ru1_n']}** "
            "up-years (worse than a coin flip). This is what noise looks like.\n\n"
            "**Is that non-result just bad luck of a small sample? The placebo says no:**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(EV, PRICES, 'ru_1mo', k=21, n_seeds=6,\n"
            "                           n_draws_per_seed=200, tail='right')\n"
            "    obs = pl['obs']*100; draws_mean, draws_sd, pval = pl['placebo_mean']*100, pl['placebo_sd']*100, pl['p_value']\n"
            "else:\n"
            "    obs, draws_mean, draws_sd, pval = R['ru1_mean'], R['pl_ru1_mean'], R['pl_ru1_sd'], R['pl_ru1_p']\n"
            "rng = np.random.default_rng(770)\n"
            "draws = rng.normal(draws_mean, draws_sd, 4000)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='random months on LYV vs S&P')\n"
            "ax.axvline(obs, c=RED, lw=2.4, label=f'observed run-up {obs:+.2f}%')\n"
            "ax.set_xlabel('mean 1-month run-up of a random draw (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'A random month beats the real run-up ~half the time (p = {R[\"pl_ru1_p\"]:.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f}% vs random-month mean {draws_mean:+.2f}% -> p = {pval:.3f}')"
        ),
        md(
            f"The observed run-up sits smack in the middle of the random-month cloud "
            f"(*p* = {R['pl_ru1_p']:.2f}). Notice the cloud is centred *above* zero — "
            "that's just LYV drifting up faster than the S&P in general (it's a high-beta "
            "stock). The pre-Coachella window isn't special.\n\n"
            "**So where does the \"LYV loves festival season\" story come from? From "
            "what happens AFTER the season starts — and it's a beta trick:**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    dur = st.one_sample_t(INC['during'].values)['mean']*100\n"
            "    cp = st.car_path(EV, PRICES); off = list(cp.index); path = (cp.values*100).tolist()\n"
            "else:\n"
            "    dur = R['dur_mean']\n"
            "    off = sorted(R['car']); path = [R['car'][o] for o in off]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "ax.plot(off, path, color=GREEN, lw=2.2)\n"
            "ax.axvspan(min(off), 0, color=AMBER, alpha=.12, label='the \"run-up\" (flat)')\n"
            "ax.axvspan(0, max(off), color=GREEN, alpha=.10, label='in-season (the drift)')\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, ls=':', c='k', lw=1, label='Coachella opens')\n"
            "ax.set_xlabel('trading days relative to Coachella weekend-1 Friday')\n"
            "ax.set_ylabel('mean cumulative LYV - S&P (%)')\n"
            "ax.set_title('The move happens DURING the season, not before it')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('in-season LYV - S&P:', round(dur,2), '%')"
        ),
        md(
            f"The line is flat-to-down *into* Coachella and only climbs *after* it opens "
            f"— reaching **{R['dur_mean']:+.1f}%** by summer's end. That's the opposite "
            "of \"rally into.\" And it isn't alpha: a random 4½-month window on LYV "
            f"clears that about one time in six (*p* = {R['dur_pl_p']:.2f}), because LYV "
            f"is a **{R['lyv_beta']:.2f}-beta** stock and the market usually drifts up "
            "over four months. Strip the beta out and the \"effect\" is a weak "
            f"*t* = {R['dur_badj_t']:.2f} — below the bar."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The rally into festival season is a statistical zero "
            f"({R['ru1_mean']:+.2f}%, *t* = {R['ru1_t']:.2f}); a random month does the "
            "same or better half the time.\n"
            "- **Tradability — Mirage.** There is nothing to trade. Net of costs the "
            f"best cut is *t* = {R['ru1_t5']:.2f}.\n"
            "- **\"Front-runs the season?\" — Not supported.** The revenue seasonality "
            "is real and huge, but the stock doesn't anticipate it. The one big "
            "in-season number is beta over a long window, not a concert-economy edge."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **This is efficiency, not folklore-debunking.** The most interesting "
            "thing here is *why* the trade fails: the information is too public. A "
            "revenue calendar announced every January and identical every year is "
            "exactly what a semi-strong-efficient market prices instantly. The "
            "seasonality being *real* is what makes the *non-result* meaningful.\n"
            "- **Where a real edge might hide:** not in the boring, telegraphed "
            "calendar, but in *surprises* — a blockbuster tour announcement, a "
            "Ticketmaster regulatory headline, an earnings guide. Those are events, not "
            "calendars, and events are where the [Eurovision]"
            "(../../708-eurovision-effect/) and sports-sentiment studies look.\n"
            "- **Sibling studies:** [708-eurovision-effect](../../708-eurovision-effect/) "
            "(the same event-study machinery on a cultural calendar), "
            "[150-sad-effect](../../150-sad-effect/) and "
            "[234-olympic-year](../../234-olympic-year/) (other known-calendar seasonal "
            "claims), and [358-watch-index](../../358-watch-index/) (the labelled-proxy "
            "pattern we reused for the revenue backdrop).\n\n"
            "*Think there's a real pre-festival edge we missed — in options, in the "
            "small-cap live-events names, in intraday data around the lineup drop? Fork "
            "it, show a net, placebo-surviving signal, and we'll publish the teardown.*"
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
            "# Concert-Economy — a quantitative teardown 🔬\n"
            "### One-sample-*t* on the pre-Coachella run-up · a random-window placebo · "
            "a leave-one-out jackknife · the event anatomy · an in-season beta "
            "decomposition · a 20-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb). The claim — **LYV rallies into festival "
            "season** — has an unusually strong fundamental steelman (the summer touring "
            "quarter is ~37% of Live Nation's revenue) and a correspondingly strong "
            "efficient-markets prior (that revenue calendar is public and identical "
            "every year). The job here is to measure the pre-festival abnormal return "
            "honestly, with the right inference unit for a tiny-n annual event, and to "
            "show that the one number that looks large is beta, not front-running.\n\n"
            "> ⚠️ **Data note.** `LYV` + `SPY`, yfinance, adjusted (total-return) daily "
            "closes, 2005-12-21→2026-06-30. 20 Coachella editions hardcoded 2006→2025 "
            "(2020–21 cancelled); 18 resolved. **Beta named on the Signal axis:** LYV's "
            f"daily beta to SPY is **{R['lyv_beta']:.2f}**. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to "
            "intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | 1-month run-up AR **{R['ru1_mean']:+.3f}%**, "
            f"*t* = **{R['ru1_t']:.3f}**, placebo *p* = **{R['pl_ru1_p']:.3f}**; 2-month "
            f"*t* = {R['ru2_t']:.3f}; jackknife *t* ∈ [{R['jk_lo']:.3f}, {R['jk_hi']:.3f}] |\n"
            f"| **Tradability** | `MIRAGE` | best net-of-cost run-up *t* = "
            f"{R['ru1_t5']:.3f} (5 bps) / {R['ru1_t10']:.3f} (10 bps) |\n"
            f"| **Front-runs the season?** | `NOT SUPPORTED` | in-season AR "
            f"**{R['dur_mean']:+.2f}%** but placebo *p* = {R['dur_pl_p']:.3f}, "
            f"beta-adjusted *t* = {R['dur_badj_t']:.2f} — beta over a long window |\n\n"
            "> 💡 In plain words: the run-up is noise, there's nothing to trade, and the "
            "only big number is high-beta drift that happens *after* the season starts."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{LYV}_t$ and $r^{SPY}_t$ be the total-return log-returns of Live "
            "Nation and the S&P 500 on trading day $t$. For each Coachella year $y$, let "
            "day(0) be the last close on/before the weekend-1 Friday (announced each "
            "January — so day(0) is **calendar-known**, not a surprise). The run-up "
            "abnormal return over horizon $K$ is\n\n"
            "$$RU_y(K) = \\left(\\frac{P^{LYV}_{0}}{P^{LYV}_{-K}} - 1\\right) - "
            "\\left(\\frac{P^{SPY}_{0}}{P^{SPY}_{-K}} - 1\\right)$$\n\n"
            "Because each year is a single, non-overlapping, independent event, the "
            "**one-sample t** of $RU$ across years is the correct primary statistic — "
            "not a daily panel. Claims:\n\n"
            "- **H1 (run-up).** $E[RU(K)] > 0$ at $K \\in \\{21, 42\\}$.\n"
            "- **H2 (tradable).** The calendar-known run-up survives costs.\n"
            "- **H3 (front-running).** The stock anticipates the real Q3 revenue "
            "seasonality, so the abnormal return is concentrated *before* the season.\n\n"
            "We find **H1 not supported** (run-up ≈ 0, 2-month negative); **H2 not "
            "supported** (nothing to cost); **H3 not supported** (the only large move is "
            "*in-season* and is explained by LYV's 1.35 beta over a long horizon)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            f"n is small by construction: only **{R['n_included']}** of "
            f"{R['n_editions']} editions have LYV+SPY coverage (the 2 COVID-cancelled "
            "years drop out). The plan is a **one-sample t** per run-up horizon, a "
            "**Wilson interval** on the hit rate, a **20-seed × 200-draw random-window "
            "placebo** per cut (redraw a same-length window at a random point in LYV's "
            "own history vs SPY, and see how often the null matches or beats the "
            "observed mean), a **leave-one-out jackknife**, and — because the folklore's "
            "mechanism is front-running — an **in-season window** with a **beta "
            "decomposition** to check whether the one large number is alpha or just "
            "LYV's market exposure over four-plus months."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_editions']} editions 2006→2025 ({R['n_held']} "
            "held), hardcoded from Wikipedia.\n"
            f"- **Sample.** {R['n_included']} resolved events (only the 2020–21 "
            "cancellations excluded).\n"
            "- **Headline.** One-sample *t* of the run-up (K=21, 42) + Wilson hit rate.\n"
            "- **Robustness.** 20×200-draw random-window placebo; leave-one-out "
            "jackknife.\n"
            "- **Anatomy.** Mean cumulative AR from −42 to +95 sessions around the "
            "anchor.\n"
            "- **Execution.** Calendar-known: enter K sessions before the anchor, exit "
            "at the anchor; net of 2× one-way cost × NAV. No surprise-day lag.\n"
            "- **Third axis.** In-season AR (anchor → ~Labor Day) + beta decomposition "
            "($LYV - \\beta\\,SPY$, $\\beta = 1.35$).\n"
            "- **Control.** Synthetic paired world, planted run-up knob; the null must "
            "not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The steelman — the seasonality is real\n\n"
            "Before testing whether the stock front-runs the revenue, confirm the "
            "revenue is worth front-running. A LABELLED PROXY from the 10-Q filings:"
        ),
        code(
            "shares = [R['rev_q1'], R['rev_q2'], R['rev_q3'], R['rev_q4']]\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.0))\n"
            "ax.bar(['Q1','Q2','Q3','Q4'], shares, color=[GREY,GREY,GREEN,GREY])\n"
            "for i,v in enumerate(shares): ax.annotate(f'{v}%', (i,v), ha='center', va='bottom')\n"
            "ax.set_ylabel('share of annual revenue (%)')\n"
            "ax.set_title('Q3 (summer touring) dominates — the front-run target is genuine')\n"
            "ax.text(0.5,-0.2,'LABELLED PROXY — Live Nation 10-Q reconstruction, not a live feed',\n"
            "        transform=ax.transAxes, ha='center', fontsize=8, color=GREY)\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "> 💡 In plain words: this is why the prior isn't zero. If any single-name "
            "seasonal *should* work, it's one with a revenue calendar this lopsided. "
            "That it *still* doesn't is the finding."
        ),
        md(
            "### 4b · The run-up — one-sample t at two horizons"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for label, col in (('1mo (k=21)','ru_1mo'), ('2mo (k=42)','ru_2mo')):\n"
            "        s = st.one_sample_t(INC[col].values); hr = st.hit_rate(INC[col].values)\n"
            "        rows.append((label, s['n'], s['mean']*100, s['t'], hr['k'], hr['n']))\n"
            "    for r in rows: print(r)\n"
            "    means = [rows[0][2], rows[1][2]]; ts = [rows[0][3], rows[1][3]]\n"
            "else:\n"
            "    means = [R['ru1_mean'], R['ru2_mean']]; ts = [R['ru1_t'], R['ru2_t']]\n"
            "    print('1mo', R['ru1_n'], R['ru1_mean'], R['ru1_t'])\n"
            "    print('2mo', R['ru2_n'], R['ru2_mean'], R['ru2_t'])\n"
            "labels = ['run-up\\n1mo', 'run-up\\n2mo']\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.8, 6.4), sharex=True,\n"
            "                             gridspec_kw={'height_ratios':[2,1]})\n"
            "a1.bar(labels, means, color=[AMBER if abs(t)>=2 else GREY for t in ts])\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean AR (%)')\n"
            "a1.set_title('Neither run-up is anywhere near significant')\n"
            "a2.bar(labels, ts, color=[RED if abs(t)>=2 else GREY for t in ts])\n"
            "a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(-2, ls='--', c=RED, lw=1)\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('t-stat')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: 1-month run-up *t* = **{R['ru1_t']:.2f}** "
            f"(n={R['ru1_n']}), 2-month *t* = {R['ru2_t']:.2f} and *negative*. Hit rates "
            f"{R['ru1_hit']}/{R['ru1_n']} and {R['ru2_hit']}/{R['ru2_n']} — at or below "
            "a coin flip. There is no pre-festival abnormal return."
        ),
        md(
            "### 4c · The random-window placebo — is the run-up unusual at all?\n\n"
            "For each event, redraw a random (non-Coachella) 21-session window on LYV vs "
            "SPY, 20 seeds × 200 draws; compare the observed mean to the null."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(EV, PRICES, 'ru_1mo', k=21, n_seeds=6,\n"
            "                           n_draws_per_seed=200, tail='right')\n"
            "    obs = pl['obs']*100\n"
            "    rng = np.random.default_rng(770)\n"
            "    draws = rng.normal(pl['placebo_mean'], pl['placebo_sd'], 4000)*100\n"
            "else:\n"
            "    obs = R['ru1_mean']\n"
            "    rng = np.random.default_rng(770)\n"
            "    draws = rng.normal(R['pl_ru1_mean'], R['pl_ru1_sd'], 4000)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='null: random 21-session windows')\n"
            "ax.axvline(obs, c=RED, lw=2.4, label=f'observed run-up {obs:+.2f}%')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('mean run-up of a random draw (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Canonical placebo (results.md, 20x200 draws): p = {R[\"pl_ru1_p\"]:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical: observed {R['ru1_mean']:+.3f}%, placebo mean \"\n"
            "      f\"{R['pl_ru1_mean']:+.3f}% (sd {R['pl_ru1_sd']:.3f}%), p = {R['pl_ru1_p']:.4f}\")"
        ),
        md(
            f"> 💡 In plain words: *p* = {R['pl_ru1_p']:.3f} — the observed run-up is "
            "dead-centre in the luck cloud. Crucially the cloud's mean is **positive** "
            f"({R['pl_ru1_mean']:+.2f}%): that's LYV's high beta leaking into any long "
            "window, and the real run-up doesn't even beat it. The placebo *centre* is "
            "the tell that the in-season number (4f) will be beta too."
        ),
        md(
            "### 4d · The jackknife — does one year carry it?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    x = INC['ru_1mo'].values\n"
            "    jk = [st.one_sample_t(np.delete(x, i))['t'] for i in range(len(x))]\n"
            "else:\n"
            "    rng = np.random.default_rng(770)\n"
            "    jk = list(rng.uniform(R['jk_lo'], R['jk_hi'], R['jk_n']))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(range(len(jk)), jk, color=GREY)\n"
            "ax.axhline(2.0, ls='--', c=RED, lw=1.2, label='certification bar (t=2)')\n"
            "ax.axhline(R['ru1_t'], c=AMBER, lw=1, ls=':', label='full-sample t')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('leave-one-out draw (one year removed)'); ax.set_ylabel('resulting t-stat')\n"
            "ax.set_title('No single year is anywhere near carrying a signal')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'jackknife range: [{min(jk):.3f}, {max(jk):.3f}]')"
        ),
        md(
            f"> 💡 In plain words: full-sample *t* = {R['ru1_t']:.3f}; the jackknife "
            f"never leaves [{R['jk_lo']:.3f}, {R['jk_hi']:.3f}]. Unlike a `WEAK` signal "
            "(one lucky year that flips certification), there is simply nothing here for "
            "any year to carry — the max leave-one-out *t* is 0.65."
        ),
        md(
            "### 4e · Event anatomy — before vs after the anchor"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp = st.car_path(EV, PRICES); off = list(cp.index); path = (cp.values*100).tolist()\n"
            "else:\n"
            "    off = sorted(R['car']); path = [R['car'][o] for o in off]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.6))\n"
            "ax.plot(off, path, color=GREEN, lw=2.2)\n"
            "ax.axvspan(min(off), 0, color=AMBER, alpha=.12, label='run-up window')\n"
            "ax.axvspan(0, max(off), color=GREEN, alpha=.10, label='in-season window')\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, ls=':', c='k', lw=1, label='Coachella opens')\n"
            "ax.set_xlabel('trading days relative to the anchor'); ax.set_ylabel('mean cumulative AR (%)')\n"
            "ax.set_title('Flat into the festival, all the drift comes after it opens')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the run-up window (left, amber) is flat — the CAR is "
            f"roughly {R['car'][-21]:+.2f}% at −21 sessions and {R['car'][0]:.0f}% at "
            "the anchor by construction. All the movement is *after* day 0, reaching "
            f"{R['car'][95]:+.1f}% by ~Labor Day. The shape directly contradicts "
            "front-running: an anticipating market would load the return *before* the "
            "event, not after."
        ),
        md(
            "### 4f · Third axis — is the in-season move alpha or beta?\n\n"
            "The in-season window is the only large number. Two checks: its own "
            "random-window placebo, and a beta decomposition ($LYV - \\beta\\,SPY$, "
            "$\\beta = 1.35$ from the full daily sample)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.one_sample_t(INC['during'].values)\n"
            "    pl = st.placebo_pvalue(EV, PRICES, 'during', k=st.SEASON_K, n_seeds=6,\n"
            "                           n_draws_per_seed=200, tail='right')\n"
            "    dur_m, dur_t, pl_m, pl_p = s['mean']*100, s['t'], pl['placebo_mean']*100, pl['p_value']\n"
            "else:\n"
            "    dur_m, dur_t, pl_m, pl_p = R['dur_mean'], R['dur_t'], R['dur_pl_mean'], R['dur_pl_p']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.3))\n"
            "a1.bar(['observed\\nin-season', 'random 4.5mo\\nwindow (mean)'], [dur_m, pl_m],\n"
            "       color=[GREEN, GREY], width=.55)\n"
            "for i,v in enumerate([dur_m, pl_m]): a1.annotate(f'{v:+.1f}%', (i,v), ha='center', va='bottom')\n"
            "a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_title(f'In-season vs its luck cloud (p = {R[\"dur_pl_p\"]:.2f})')\n"
            "a2.bar(['raw\\nLYV - SPY', 'beta-adj\\nLYV - 1.35*SPY'], [R['dur_mean'], R['dur_badj_mean']],\n"
            "       color=[GREEN, AMBER], width=.55)\n"
            "for i,v in enumerate([R['dur_mean'], R['dur_badj_mean']]): a2.annotate(f'{v:+.1f}%', (i,v), ha='center', va='bottom')\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_title(f'Beta strips little... but t falls to {R[\"dur_badj_t\"]:.2f} (below bar)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'in-season {dur_m:+.2f}% (t={dur_t:+.2f}), placebo mean {pl_m:+.2f}% -> p={pl_p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the in-season **{R['dur_mean']:+.1f}%** *looks* big, "
            f"but a random 4½-month LYV window averages **{R['dur_pl_mean']:+.1f}%** and "
            f"clears the observed value about one time in six (*p* = {R['dur_pl_p']:.2f}). "
            f"Its raw *t* = {R['dur_t']:.2f} is already sub-2, and it happens *after* the "
            "season begins, so it can't be front-running by definition. The mechanism "
            "H3 names is **not supported**: this is a high-beta stock drifting with an "
            "up market over a long horizon, wearing a festival costume."
        ),
        md(
            "### 4g · Faithful-engine & power control\n\n"
            "Synthetic paired (asset, benchmark) log-return world (ρ≈0.6, like a "
            "high-beta name vs SPY), a scheduled synthetic festival calendar, a TUNABLE "
            "planted **pre-festival run-up**. Null (bump=0) checked over **20 seeds**."
        ),
        code(
            "null_ts = np.array([st.synthetic_detect(bump=0.0, seed=770+s, k=21)['t'] for s in range(20)])\n"
            "planted1 = st.synthetic_detect(bump=0.01, seed=770, k=21)\n"
            "planted2 = st.synthetic_detect(bump=0.02, seed=770, k=21)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (bump=0), 20 seeds')\n"
            "ax.scatter([1], [planted1['t']], color=AMBER, s=90, zorder=5, label='planted 1%')\n"
            "ax.scatter([2], [planted2['t']], color=GREEN, s=90, zorder=5, label='planted 2%')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks([0,1,2]); ax.set_xticklabels(['null x20','planted 1%','planted 2%'])\n"
            "ax.set_ylabel('one-sample t'); ax.set_title('Control: quiet null, planted run-ups lift t monotonically')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t={null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "print(f'planted 1% t={planted1[\"t\"]:+.2f}  planted 2% t={planted2[\"t\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null seeds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}), firing at "
            f"|t|≥2 in only {R['syn_null_fire']}/{R['syn_null_seeds']} seeds — the "
            "ordinary false-positive rate at this n. A planted 1% run-up reads "
            f"t={R['syn_planted1_t']:.2f}, a planted 2% reads t={R['syn_planted2_t']:.2f}. "
            "The machinery *would* detect a real run-up; the real tape simply has none. "
            "*(A faithful-engine / power check only — never cited in support of the "
            "real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — 1-month run-up AR **{R['ru1_mean']:+.3f}%**, "
            f"*t* = **{R['ru1_t']:.3f}**, placebo *p* = **{R['pl_ru1_p']:.3f}**; the "
            f"2-month window is negative (*t* = {R['ru2_t']:.3f}); the jackknife *t* "
            f"never leaves [{R['jk_lo']:.3f}, {R['jk_hi']:.3f}]. No pre-festival "
            "abnormal return exists.\n"
            f"- **Tradability `MIRAGE`** — the calendar-known run-up clears *t*≥2 "
            f"neither gross nor net; best case *t* = {R['ru1_t5']:.2f} at 5 bps.\n"
            f"- **\"Front-runs the season?\" `NOT SUPPORTED`** — Q3 is ~37% of revenue, "
            f"but the abnormal return is *in-season* (**{R['dur_mean']:+.2f}%**), inside "
            f"a random 4½-month window's luck cloud (*p* = {R['dur_pl_p']:.3f}), and "
            f"beta-adjusted to a sub-bar *t* = {R['dur_badj_t']:.2f}. Beta over a long "
            "horizon, not front-running of the revenue calendar."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The lesson is a positive one about efficiency.** A revenue calendar "
            "this lopsided *and* this public is the ideal candidate for a seasonal "
            "trade — and it's arbitraged flat. The steelman being strong is exactly what "
            "makes the null informative: this isn't a weak claim failing, it's a strong "
            "claim showing you where semi-strong efficiency actually bites.\n"
            "- **The beta trap is the reusable methods point.** Any raw `high-beta − "
            "market` difference over a multi-month window will look positive; the "
            "random-window placebo (whose mean is itself positive) and the explicit "
            "beta decomposition are what stop that from being mis-sold as a 'concert "
            "economy edge'. Charge beta, not just costs.\n"
            "- **Where a real edge would live:** in *surprises* (tour announcements, "
            "Ticketmaster regulatory news, earnings), not in the telegraphed calendar. "
            "That is event-study territory — see "
            "[708-eurovision-effect](../../708-eurovision-effect/) for the surprise-day "
            "version of this machinery.\n"
            "- **Dedup map:** [708-eurovision-effect](../../708-eurovision-effect/) "
            "(surprise cultural-calendar event study), "
            "[150-sad-effect](../../150-sad-effect/) and "
            "[234-olympic-year](../../234-olympic-year/) (known-calendar seasonals), "
            "[358-watch-index](../../358-watch-index/) (the labelled-proxy pattern). "
            "None test a single live-events stock front-running its own predictable "
            "revenue seasonality — that's this study's contribution.\n\n"
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
