"""Generate the two narrative notebooks for Study 309 (OJ-Frost).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached OJ=F
parquet under ../_cache/ if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-05-31).
R = dict(
    asof="2026-05-31",
    oj_start="2001-09-17", oj_end="2026-05-29", oj_n=6198, oj_fp="1b3126f827b9",
    n_freeze_total=12, n_in_tape=4, n_pre_tape=8,
    in_tape="2003, 2009, 2010, 2022",
    # reactive lag=1 event study
    w5_n=4, w5_win=0, w5_mean=-240, w5_t=-4.51, w5_placebo=17, w5_excess=-243,
    w10_mean=-340, w10_t=-1.85, w21_mean=-392, w21_t=-1.54,
    # per-event w=5
    ev_2003=-338, ev_2009=-326, ev_2010=-44, ev_2022=-250,
    # perfect-foresight ceiling
    lag0_mean=-360, lag0_t=-3.83,
    # winter seasonality
    winter_bps=-6.2, other_bps=3.5, winter_diff=-9.7, winter_t=-1.36,
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from oj_frost import data, strategy as st

ASOF = pd.Timestamp("2026-05-31")

def _have_cache():
    return os.path.exists(data._cache_path("OJ=F", data.DEFAULT_CACHE))

HAVE_REAL = _have_cache()

def load_oj():
    f = data.fetch_oj("OJ=F", fetch=False)
    return f[f.index <= ASOF]

print("real OJ=F cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# OJ-Frost 🍊 — does the *Trading Places* trade actually exist?\n"
            "### A hard freeze, a crop wiped out, orange-juice futures to the moon — or just a great movie?\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![The_Trading_Places_trade%3F: Busted](https://img.shields.io/badge/The_Trading_Places_trade%3F-Busted-8b949e?style=flat-square)\n\n"
            "In *Trading Places* (1983) the whole plot turns on one trade: a freeze is coming to "
            "Florida, the orange crop is doomed, frozen-concentrate OJ futures will spike — so corner "
            "the market first and get rich. It's such a clean story that 'buy OJ when it freezes' "
            "became folklore. This notebook asks, in plain English: **if you had actually run that "
            "trade, would you have made money?**\n\n"
            "> 📓 **This is the plain-language layer.** Want the event-study statistics, the "
            "small-sample trap and the placebo control? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is there a freeze pop you can trade? | **No.** Of the 12 big freezes on record, "
            f"**{R['n_pre_tape']} happened before** the OJ futures tape on Yahoo even starts (2001). "
            f"Only **{R['n_in_tape']}** are testable. |\n"
            "| And those four? | **They all *fell*.** On every one of the four freezes in the data, "
            "OJ futures were **down** a week later. |\n"
            "| So the *Trading Places* trade…? | **Busted.** Great cinema, money-losing finance — at "
            "least on everything we can actually measure. |\n"
            "| Is it just winter, then? | **No.** OJ's Dec–Feb 'freeze-risk' season is, if anything, "
            "slightly *worse* than the rest of the year, and not significant. |\n\n"
            "> The legend isn't a lie so much as a **survivor**: it lives on freezes from the 1970s "
            "and 80s that no modern data feed contains. On the freezes we *can* check, buying OJ after "
            "the cold snap was a loser, four for four."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Look, a freeze hits the citrus belt, half the Florida crop is gone, and frozen "
            "orange-juice concentrate gets scarce. The futures have to spike. Get long OJ before "
            "everyone else figures out how bad the damage is — that's the whole game.\"*\n\n"
            "It isn't pure Hollywood, either. The serious version is **Richard Roll's 1984 paper "
            "*Orange Juice and Weather*** — FCOJ futures really do react to Florida freeze forecasts; "
            "OJ is the textbook 'weather commodity.' So the channel is real. The bet folklore makes is "
            "the next step: that you could have **traded** it."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two reasons this one is worth a careful look. First, it's the most *famous* trade in "
            "movie history — if it works, that's a satisfying 'the legend is true.' Second, and more "
            "useful: it's a perfect lesson in **survivorship of a story**. The trades everyone "
            "remembers are the dramatic 1980s freezes — and those are exactly the ones missing from "
            "the data you can actually pull up. When the evidence for an edge sits entirely in the "
            "period you *can't* test, that's a giant red flag, and it shows up again and again."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Simple, honest plan:\n\n"
            "1. **List the freezes.** A hardcoded table of severe Florida citrus-belt freezes (the "
            "ones the lore is built on).\n"
            "2. **Trade each one — but fairly.** You can't act on a freeze you haven't heard about, so "
            "we buy on the *first session after* the cold night and hold for a week (one honest "
            "execution lag — never the freeze-day move itself).\n"
            "3. **Compare to random dates.** Buy OJ for a week starting on thousands of *random* days "
            "too — that's the 'is the freeze window special, or is this just how OJ moves?' baseline.\n"
            "4. **Check the calendar too.** Is OJ just generally strong in winter (Dec–Feb)?\n\n"
            "If buying after a freeze doesn't beat buying on a random Tuesday, there's no trade.\n\n"
            "Tape: Yahoo `OJ=F` continuous front-month OJ futures."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the punchline that ends the movie.** How many of the famous freezes are even "
            "*in* the data you can download today?"
        ),
        code(
            "fz = data.freeze_dates()\n"
            "if HAVE_REAL:\n"
            "    f = load_oj(); start = f.index.min()\n"
            "    in_tape = fz[(fz >= start) & (fz <= f.index.max())]\n"
            "    pre = fz[fz < start]\n"
            "else:\n"
            f"    start = pd.Timestamp('{R['oj_start']}')\n"
            "    in_tape = fz[fz >= start]; pre = fz[fz < start]\n"
            "fig, ax = plt.subplots(figsize=(10, 3.2))\n"
            "ax.scatter(pre, [1]*len(pre), s=120, color=GREY, label=f'before the tape ({len(pre)})', zorder=3)\n"
            "ax.scatter(in_tape, [1]*len(in_tape), s=160, color=RED, marker='X',\n"
            "           label=f'in the tradable tape ({len(in_tape)})', zorder=4)\n"
            "ax.axvline(start, ls='--', c='k', lw=1)\n"
            "ax.annotate('OJ=F tape\\nbegins', (start, 1.25), ha='center', fontsize=9)\n"
            "ax.set_yticks([]); ax.set_ylim(0.5, 1.6); ax.legend(loc='lower right')\n"
            "ax.set_title('Most of the famous freezes predate the data you can trade')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'{len(pre)} freezes before the tape, {len(in_tape)} inside it: {list(in_tape.year)}')"
        ),
        md(
            f"There's the whole study in one chart. **{R['n_pre_tape']} of the {R['n_freeze_total']} "
            "freezes** — including every dramatic 1980s freeze the legend is built on — happened "
            "**before** the OJ futures tape on Yahoo begins in 2001. The trade everyone remembers is "
            "literally untestable with the data a normal person can get."
        ),
        md(
            f"**So what about the {R['n_in_tape']} freezes we *can* check?** Here's what OJ did over "
            "the week *after* each one (buying the first session after the freeze):"
        ),
        code(
            "if HAVE_REAL:\n"
            "    f = load_oj()\n"
            "    led = st.window_returns(f, in_tape, window=5, lag=1)\n"
            "    evs = [d.year for d in led['event_date']]\n"
            "    rets = (led['ret_gross']*1e4).tolist()\n"
            "else:\n"
            f"    evs = [2003, 2009, 2010, 2022]\n"
            f"    rets = [{R['ev_2003']}, {R['ev_2009']}, {R['ev_2010']}, {R['ev_2022']}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "b = ax.bar([str(e) for e in evs], rets, color=[RED if r < 0 else GREEN for r in rets], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('OJ return, week after the freeze (bps)')\n"
            "ax.set_title('The \"freeze pop\"? Every freeze in the data was a week of LOSSES')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Per-freeze 5-day returns (bps):', [round(r) for r in rets])"
        ),
        md(
            f"Four freezes, four losing weeks: **{R['ev_2003']}, {R['ev_2009']}, {R['ev_2010']}, "
            f"{R['ev_2022']} bps**. Not a single pop. The mean is about **{R['w5_mean']} bps** — you'd "
            "have lost ~2.4% a week buying the freeze. The movie trade isn't just weak on this tape; "
            "it points the *wrong way*."
        ),
        md(
            "**Could it just be that OJ is strong in winter generally?** Here's Dec–Feb vs the rest of "
            "the year:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ws = st.winter_seasonality(load_oj())\n"
            "    w, o = ws['winter_mean_bps'], ws['other_mean_bps']\n"
            "else:\n"
            f"    w, o = {R['winter_bps']}, {R['other_bps']}\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "ax.bar(['Winter (Dec-Feb)', 'Rest of year'], [w, o],\n"
            "       color=[RED if w < 0 else GREEN, GREEN if o > 0 else RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean daily return (bps)')\n"
            "ax.set_title('No \"freeze-risk season\" either — winter is if anything weaker')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Winter {w:+.1f} bps/day  vs  rest {o:+.1f} bps/day')"
        ),
        md(
            f"Winter days average **{R['winter_bps']:+.1f} bps** vs **{R['other_bps']:+.1f} bps** the "
            "rest of the year — the supposed freeze-risk season is actually a touch *worse*, and "
            "statistically it's a coin flip. No seasonal trade hiding here either."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Only {R['n_in_tape']} freezes are in the tradable tape, and all four "
            "were losing weeks. Four events can't prove anything — and what little there is points the "
            "wrong way. Winter seasonality is a non-event too.\n"
            "- **Tradability — Mirage.** Nothing to trade: the legendary freezes aren't in the data, "
            "the ones that are go the wrong way, and OJ futures are a tiny, wide-spread market on top "
            "of it.\n"
            "- **The *Trading Places* trade? — Busted.** A wonderful movie premise and, on every "
            "freeze we can measure, a money-loser."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's no edge to charge costs against — but even if there *were*, the news is worse. "
            "The honest trade enters the day **after** the freeze. But maybe the pop happens *on* the "
            "freeze day and you just missed it? So let's give the trade a superpower it can't really "
            "have — perfect foresight, entering on the freeze session itself:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    f = load_oj()\n"
            "    m1 = st.summarize_events(st.window_returns(f, in_tape, window=5, lag=1), 'ret_gross')['mean_bps']\n"
            "    m0 = st.summarize_events(st.window_returns(f, in_tape, window=5, lag=0), 'ret_gross')['mean_bps']\n"
            "else:\n"
            f"    m1, m0 = {R['w5_mean']}, {R['lag0_mean']}\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['Honest (enter day after)', 'Perfect foresight (enter on freeze)'], [m1, m0],\n"
            "       color=[RED, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean 5-day return (bps)')\n"
            "ax.set_title('Even with perfect foresight, the freeze week loses money')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Honest lag: {m1:+.0f} bps   Perfect foresight: {m0:+.0f} bps')"
        ),
        md(
            f"Even cheating — entering right on the freeze session — the week still averages "
            f"**{R['lag0_mean']} bps**. The 'pop' isn't a move you were too slow to catch; on this "
            "tape it simply **isn't there**. There is no version of the trade, fast or slow, fair or "
            "not, that makes money."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The data that built the legend is gone.** The honest fix would be a pre-2001 FCOJ "
            "series (the 1977/1981/1983/1985 freezes). If you have one, fork this and run the *real* "
            "*Trading Places* era — that's the only way to test the legend on its own turf.\n"
            "- **Forecast-anticipation, not freeze-reaction.** Roll's signal is in the *forecast*, "
            "not the freeze. A version keyed to cold-snap *forecasts* (before the night) is a "
            "different, harder experiment.\n"
            "- **Weather folklore in general.** See [Study 281 — El-Nino](../../281-el-nino/) for "
            "another weather-to-markets trade that didn't survive contact with the data.\n\n"
            "*Think the freeze trade is real and we just have the wrong window? Fork this, change the "
            "window or the freeze list, and show OJ popping — on data you can actually pull.*"
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
            "# OJ-Frost — a quantitative teardown 🔬\n"
            "### Real OJ=F tape · freeze event study · the n=4 small-sample trap · placebo control · "
            "perfect-foresight ceiling · winter seasonality · synthetic positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![The_Trading_Places_trade%3F: Busted](https://img.shields.io/badge/The_Trading_Places_trade%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We run an event study around a "
            "hardcoded freeze table on the modern OJ futures tape, pin it against a random-date "
            "placebo, expose the small-sample fragility honestly, and bound the un-tradable ceiling.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo `OJ=F` continuous front-month FCOJ "
            "future, 2001–2026; as-of 2026-05-31 (partial June dropped); the offline core and tests "
            "run on a deterministic synthetic tape. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Only **n = {R['n_in_tape']}** freezes in the tape; 5-day reactive "
            f"mean **{R['w5_mean']} bps** (win-rate {R['w5_win']}%). The HAC *t* = {R['w5_t']} is a "
            f"**degenerate small-sample artefact**, not evidence of an edge. Winter *t* = {R['winter_t']}. |\n"
            f"| **Tradability** | `MIRAGE` | {R['n_pre_tape']} of {R['n_freeze_total']} lore freezes "
            "predate the tape; in-tape freezes go the wrong way; FCOJ is a thin, wide-spread contract. |\n"
            f"| **The *Trading Places* trade?** | `BUSTED` | Perfect-foresight ceiling (lag=0) still "
            f"{R['lag0_mean']} bps (*t* = {R['lag0_t']}). No fill, fast or slow, recovers a pop. |\n\n"
            "> 💡 In plain words: the famous trade is built on freezes the data doesn't contain; the "
            "four we can test all lost money over the following week; and the apparent *t*-stat is just "
            "four numbers that happened to share a sign — not a signal."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $F$ be the set of severe Florida freeze dates and $r_{[1,w]}^{(e)}$ the OJ log-return "
            "over the $w$-session window starting one session **after** freeze $e$ (the reactive, "
            "look-ahead-free entry). The folklore + Roll (1984) channel gives:\n\n"
            "- **H₁ (freeze pop).** $\\mathbb{E}\\big[r_{[1,w]}^{(e)} \\mid e \\in F\\big] > 0$ and "
            "exceeds the same statistic on random non-freeze dates.\n"
            "- **H₂ (it's capturable).** The forward window — the part you could actually trade after "
            "the cold night — is where the pop lives (not only the un-catchable freeze-day move).\n"
            "- **H₃ (winter premium).** Dec–Feb daily returns exceed the rest of the year.\n\n"
            "We find **H₁ rejected** (negative, and uncertifiable at n = 4), **H₂ rejected** (even the "
            "lag=0 ceiling is negative), and **H₃ rejected** (insignificant, wrong sign)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The weather-to-price channel is genuinely documented (Roll 1984; Boudoukh et al. 2007), "
            "so the *interesting* content isn't 'does OJ react to freezes' — it's whether a **listable, "
            "tradable** rule on the **modern** tape banks anything. The decisive fact is a "
            "**survivorship-of-the-narrative** problem: the events that built the belief "
            f"({R['n_pre_tape']}/{R['n_freeze_total']}) are outside any retail data feed, and the "
            "remaining sample is far too small to certify a sign. Naming *why* the legend can't be "
            "tested is the product here, more than the (foregone) verdict."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Events.** Hardcoded `FREEZE_EVENTS` table of severe Florida freezes.\n"
            "- **Entry.** First session at-or-after the freeze, then **lag = 1** (the documented "
            "execution lag — you act no sooner than the session after the cold night); hold $w$ "
            "sessions. A **lag = 0** 'perfect foresight' variant bounds the un-tradable ceiling.\n"
            "- **Control.** A random-date placebo: the same $w$-window mean on thousands of random "
            "entry sessions — nets out OJ's ambient drift and the window length.\n"
            "- **Inference.** Newey-West HAC *t* and a circular block-bootstrap CI — **reported with "
            "the loud caveat that at n = 4 neither is interpretable.**\n"
            "- **Seasonality.** Dec–Feb vs rest-of-year daily mean, HAC *t* on a winter-dummy slope.\n"
            "- **Positive control.** Deterministic synthetic tape with a front-loaded post-freeze "
            "spike — proves the engine *can* detect a freeze edge when one exists.\n\n"
            "Tape: Yahoo `OJ=F`, 2001–2026, as-of 2026-05-31, gross (costs are moot with no edge)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The sample problem — the legend is mostly un-testable\n\n"
            "Before any statistic: how many freezes can the data even see?"
        ),
        code(
            "fz = data.freeze_dates()\n"
            "if HAVE_REAL:\n"
            "    f = load_oj(); start, end = f.index.min(), f.index.max()\n"
            "else:\n"
            f"    start, end = pd.Timestamp('{R['oj_start']}'), pd.Timestamp('{R['oj_end']}')\n"
            "in_tape = fz[(fz >= start) & (fz <= end)]; pre = fz[fz < start]\n"
            "fig, ax = plt.subplots(figsize=(10, 3.0))\n"
            "ax.scatter(pre, [1]*len(pre), s=110, color=GREY, label=f'pre-tape, untestable (n={len(pre)})')\n"
            "ax.scatter(in_tape, [1]*len(in_tape), s=150, color=RED, marker='X', label=f'in tape (n={len(in_tape)})')\n"
            "ax.axvspan(start, end, color=GREEN, alpha=.06)\n"
            "ax.axvline(start, ls='--', c='k', lw=1)\n"
            "ax.set_yticks([]); ax.set_ylim(.6,1.5); ax.legend(loc='upper left', fontsize=9)\n"
            "ax.set_title('The OJ=F tape (shaded) misses every 1977-2001 freeze')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pre-tape: {list(pre.year)}'); print(f'in-tape: {list(in_tape.year)}')"
        ),
        md(
            f"> 💡 In plain words: {R['n_pre_tape']} of {R['n_freeze_total']} freezes are outside the "
            "data. The *Trading Places*-era freezes — the ones that made the legend — are exactly the "
            "ones we can't touch. Any edge measured here rests on **four** events."
        ),
        md(
            "### 4b · The freeze event study vs the placebo — and the n=4 trap\n\n"
            "The reactive (lag=1) window mean, by horizon, against the random-date placebo "
            "distribution. **The headline is the sample size, not the *t*-stat.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    f = load_oj()\n"
            "    rows = []\n"
            "    for w in (5, 10, 21):\n"
            "        led = st.window_returns(f, in_tape, window=w, lag=1)\n"
            "        ctrl = st.random_control_windows(f, n_events=len(led), window=w, n_draws=5000)\n"
            "        s = st.summarize_events(led, 'ret_gross', control_means=ctrl)\n"
            "        rows.append((w, s['n_events'], s['win_rate']*100, s['mean_bps'], s['tstat'], s['placebo_pct']*100))\n"
            "    tbl = pd.DataFrame(rows, columns=['window','n','win%','mean_bps','HAC_t','placebo_pct'])\n"
            "else:\n"
            "    tbl = pd.DataFrame({'window':[5,10,21],'n':[4,4,4],'win%':[0,25,25],\n"
            f"        'mean_bps':[{R['w5_mean']},{R['w10_mean']},{R['w21_mean']}],\n"
            f"        'HAC_t':[{R['w5_t']},{R['w10_t']},{R['w21_t']}],'placebo_pct':[{R['w5_placebo']},17,20]}})\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.bar(tbl['window'].astype(str)+'d', tbl['mean_bps'], color=RED, width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean window return (bps)')\n"
            "ax.set_xlabel('forward window'); ax.set_title('Every horizon: negative. And every cell is n=4.')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tbl.round(2).to_string(index=False))"
        ),
        md(
            f"> 💡 In plain words: the 5-day window's HAC *t* of **{R['w5_t']}** looks decisive, but it "
            "is **four identically-signed numbers** — the block bootstrap collapses to a point and the "
            "asymptotic *t* is meaningless. This is the textbook MacKinlay (1997) small-sample failure. "
            "The honest statement is **no usable signal**, with the footnote that the scraps point the "
            "*wrong* way for the folklore. The freeze window sits around the "
            f"{R['w5_placebo']}th placebo percentile — not even unusual."
        ),
        md(
            "### 4c · The un-tradable ceiling — perfect foresight (lag=0)\n\n"
            "Maybe the reactive lag throws away the pop? Bound it: enter on the freeze session itself "
            "(look-ahead — you don't yet know the crop damage, so this is a *ceiling*, not a strategy)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    f = load_oj()\n"
            "    s1 = st.summarize_events(st.window_returns(f, in_tape, window=5, lag=1), 'ret_gross')\n"
            "    s0 = st.summarize_events(st.window_returns(f, in_tape, window=5, lag=0), 'ret_gross')\n"
            "    vals, ts = [s1['mean_bps'], s0['mean_bps']], [s1['tstat'], s0['tstat']]\n"
            "else:\n"
            f"    vals, ts = [{R['w5_mean']}, {R['lag0_mean']}], [{R['w5_t']}, {R['lag0_t']}]\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "b = ax.bar(['Reactive (lag=1)','Perfect foresight (lag=0)'], vals, color=[RED, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean 5-day return (bps)')\n"
            "for bar_, t_ in zip(b, ts):\n"
            "    ax.annotate(f't={t_:+.2f}*', (bar_.get_x()+bar_.get_width()/2, bar_.get_height()),\n"
            "                ha='center', va='top', color='white', fontweight='bold')\n"
            "ax.set_title('No fill recovers a pop (*t at n=4 is not interpretable)')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'lag=1 {{vals[0]:+.0f}} bps   lag=0 {{vals[1]:+.0f}} bps')"
        ),
        md(
            f"> 💡 In plain words: even cheating with perfect foresight the 5-day window is "
            f"**{R['lag0_mean']} bps**. The pop is not a move you were too slow to capture — it isn't "
            "in the tape at all. H₂ rejected."
        ),
        md(
            "### 4d · Winter seasonality\n\n"
            "The calendar version of the story: is OJ simply strong in the Dec–Feb freeze-risk season?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ws = st.winter_seasonality(load_oj())\n"
            "    w, o, diff, t = ws['winter_mean_bps'], ws['other_mean_bps'], ws['diff_bps'], ws['tstat']\n"
            "else:\n"
            f"    w, o, diff, t = {R['winter_bps']}, {R['other_bps']}, {R['winter_diff']}, {R['winter_t']}\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "ax.bar(['Winter (DJF)','Rest'], [w, o], color=[RED if w<0 else GREEN, GREEN if o>0 else RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean daily return (bps)')\n"
            "ax.set_title(f'Winter premium: {diff:+.1f} bps/day (HAC t={t:+.2f}) - not significant')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'winter {w:+.1f}  rest {o:+.1f}  diff {diff:+.1f} bps/day  t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the winter–rest difference is **{R['winter_diff']} bps/day** at HAC "
            f"*t* = **{R['winter_t']}** — wrong sign, not significant. No seasonal trade. H₃ rejected."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — n = {R['n_in_tape']} in-tape freezes, 5-day reactive mean "
            f"{R['w5_mean']} bps (win {R['w5_win']}%). The HAC *t* = {R['w5_t']} is a degenerate "
            "n = 4 artefact, **not** a real (negative) edge; the freeze window isn't even unusual vs "
            f"placebo ({R['w5_placebo']}th pct). Winter *t* = {R['winter_t']}. Nothing certifiable.\n"
            f"- **Tradability `MIRAGE`** — {R['n_pre_tape']}/{R['n_freeze_total']} lore freezes predate "
            "the tape, the in-tape ones lose, and FCOJ is a thin, wide-spread contract. No edge, no "
            "capacity.\n"
            f"- **The *Trading Places* trade? `BUSTED`** — perfect-foresight ceiling {R['lag0_mean']} "
            "bps; no fill recovers a pop. The legend lives in untradable history."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it?\n\n"
            "There is no positive edge to charge costs against, so the cost sweep is moot — but it's "
            "worth stating *why* even a hypothetical edge would struggle. FCOJ is one of the smallest, "
            "widest-spread futures markets; a realistic round-trip cost on a single-contract retail "
            "fill is large relative to a few-day move, and open interest caps size at a level that "
            "rules out meaningful capital. So the Tradability axis was `MIRAGE` **before** the Signal "
            "axis even reported: a thin contract, a four-event sample, and the famous freezes missing "
            "from the data. The candid bottom line: there is nothing here to size."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the *engine* a faithful freeze detector, or would it have missed a real pop? Plant a "
            "front-loaded post-freeze spike on a synthetic tape and sweep its size — the event study "
            "should light up cleanly when an effect is actually there (and stay quiet at zero)."
        ),
        code(
            "jumps = [0.0, 0.05, 0.10, 0.20, 0.30]\n"
            "tstats, placebos = [], []\n"
            "for j in jumps:\n"
            "    fr, tr = data.synthetic_oj(freeze_jump=j, seed=309)\n"
            "    led = st.window_returns(fr, tr['syn_freezes'], window=5, lag=1)\n"
            "    ctrl = st.random_control_windows(fr, n_events=len(led), window=5, n_draws=1000)\n"
            "    s = st.summarize_events(led, 'ret_gross', control_means=ctrl)\n"
            "    tstats.append(s['tstat']); placebos.append(s['placebo_pct']*100)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.plot(jumps, tstats, 'o-', c=GREEN, lw=2, label='HAC t')\n"
            "ax.axhline(2, ls='--', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted post-freeze spike (total log-move)'); ax.set_ylabel('HAC t-stat')\n"
            "ax.set_title('The engine is a faithful detector: t rises with a real planted pop')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('planted jump -> HAC t:', [round(t,2) for t in tstats])\n"
            "print('placebo percentile:', [round(p) for p in placebos])"
        ),
        md(
            "The engine cleanly detects a planted freeze pop (HAC *t* climbs through 2 as the spike "
            "grows; placebo percentile → 100), and is quiet at zero. So the real-tape null is a "
            "statement about **the market on this tape**, not a broken pipeline: with a 12-event "
            "synthetic the harness banks a real effect, while the real `OJ=F` tape offers only four "
            "freezes and no pop.\n\n"
            "The right next experiment is a **pre-2001 FCOJ series** — the *Trading Places* era — "
            "where the legend was born. Until then, on the data a normal trader can pull, the freeze "
            "trade is a movie, not a market. See [Study 281 — El-Nino](../../281-el-nino/) for the "
            "same fate met by another weather-to-markets story."
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
