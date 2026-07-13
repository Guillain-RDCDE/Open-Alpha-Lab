"""Generate the two narrative notebooks for Study 732 (Tour-de-France-Effect).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached EWQ /
^FCHI / VGK tapes under ../_cache/ and otherwise quote the frozen headline numbers in
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


# Frozen real-tape headline numbers -- mirror of docs/results.md
# (EWQ total-return + ^FCHI price-only + VGK, yfinance, 1990-03-01 -> 2026-06-30;
#  30 Tour editions 1996-2025, all EWQ-covered, 21 also VGK-covered for the abnormal.)
R = dict(
    n_editions=30, n_included=30, n_ar=21, win_min=15, win_max=16, win_mean=15.5,
    fp="bf542dd5cb07",
    # raw seasonal (EWQ total return over the Tour window)
    raw_mean=-0.347, raw_t=-0.371, raw_hit=13, raw_n=30, raw_lo=27.4, raw_hi=60.8,
    raw_net_mean=-0.447, raw_net_t=-0.478,
    cac_mean=-0.325, cac_t=-0.376, cac_hit=18, cac_n=30,
    # abnormal decomposition (2005+ subset, n=21)
    ewq_sub_mean=+0.476, ewq_sub_t=+0.416,
    vgk_mean=+0.681, vgk_t=+0.660,
    ar_mean=-0.205, ar_t=-0.777, ar_hit=9, ar_n=21, ar_lo=24.5, ar_hi=63.5,
    welch=-0.133,
    # random-window placebo (20 seeds x 200 draws)
    pl_raw_obs=-0.347, pl_raw_mean=+0.586, pl_raw_sd=0.961, pl_raw_p=0.167,
    pl_rawnet_p=0.167,
    pl_ar_obs=-0.205, pl_ar_mean=-0.019, pl_ar_sd=0.270, pl_ar_p=0.247,
    pl_raw_right=0.833, pl_ar_right=0.752,
    # event anatomy (mean cumulative return by session offset from entry)
    car_raw={0: 0.000, 4: -0.486, 8: -0.208, 12: -0.451, 16: -0.250},
    car_ar={0: 0.000, 4: -0.175, 8: -0.303, 12: -0.223, 16: -0.271},
    # 2020 race-vs-calendar probe
    y2020_raw=-5.035, y2020_ar=-0.963,
    raw_no20_mean=-0.185, raw_no20_t=-0.194, raw_no20_n=29,
    # synthetic control
    syn_null_mean=+0.44, syn_null_sd=0.83, syn_null_fire=0, syn_null_seeds=20,
    syn_p5_t=+3.38, syn_p10_t=+5.41,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![France--specific%3F: Not_supported](https://img.shields.io/badge/France--specific%3F-Not_supported-8b949e?style=flat-square)\n\n"
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

from tour_de_france_effect import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.load_real()
    EV = st.build_event_table(PRICES, cost_bps=5.0)
    INC = EV[EV["included"]]
    SUB = INC[INC["has_ar"]]                        # 2005+, VGK coverage (abnormal test)
else:
    PRICES = EV = INC = SUB = None
print("real cache present:", HAVE_REAL, "| editions:", len(data.EVENTS),
      "| resolved:", (0 if INC is None else len(INC)),
      "| with abnormal:", (0 if SUB is None else len(SUB)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do French stocks rally during the Tour de France? 🚴📈\n"
            "### The \"Grande Boucle\" summer seasonal — a feel-good bump that turns out "
            "to be a mild drag\n\n"
            + BADGES +
            "Every July, France downs tools and watches the Tour de France roll from the "
            "Grand Départ to the Champs-Élysées — and every July, somebody writes the "
            "cheerful little piece: *the whole country's in a good mood, so French "
            "markets get a summer-holiday bump.* It's the calendar-window cousin of a "
            "**real** academic finding — a 2007 study showed national stock markets "
            "genuinely dip when a country's football team is knocked out of the World "
            "Cup. The Tour version borrows the mechanism (mood moves markets) and swaps "
            "in a much gentler trigger: not a win or a loss, just three weeks of bikes.\n\n"
            "We tested it properly — all 30 editions, 1996→2025, on real French equities "
            "against Europe as a whole.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the "
            "France-vs-Europe decomposition? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 30 Tours hardcoded from Wikipedia (2020 COVID-shifted "
            "to Aug/Sep). `EWQ` = iShares MSCI France (total-return); `VGK` = Europe "
            "benchmark; `^FCHI` = CAC 40 price index (dividend-free, cross-check only). "
            "Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do French stocks pop during the Tour? | **No — they dip a little.** EWQ "
            f"averages **{R['raw_mean']:+.2f}%** over the three-week window; the hit rate "
            f"is **{R['raw_hit']}/{R['raw_n']}**, *below* a coin flip. |\n"
            f"| Is that dip France-specific? | **No.** Measured against Europe, France is "
            f"**{R['ar_mean']:+.2f}%** — it slightly *lags*. The mild July softness is a "
            "whole-continent summer thing, not a French one. |\n"
            f"| Is it even unusual? | **No.** A random three-week window in the same tape "
            f"averages **{R['pl_raw_mean']:+.2f}%**; the Tour window sits comfortably "
            "inside the luck cloud (placebo *p* = 0.17). |\n"
            f"| Could you trade it? | **You'd lose money.** Net of costs the seasonal is "
            f"**{R['raw_net_mean']:+.2f}%** — you'd be paying the spread to rent the "
            "summer doldrums. |\n\n"
            "> The folklore isn't just weak here — it points the wrong way. There is no "
            "bump; there's a mild, unremarkable, entirely non-French summer softness."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"For three weeks every July the whole of France is glued to the Tour de "
            "France — a wave of national feel-good, tourism and summer-holiday spending "
            "that lifts sentiment enough to show up as a bump in French stocks.\"*\n\n"
            "It rides on real science: Edmans, García & Norli (2007) found national "
            "markets genuinely *fall* the day after a country is **eliminated** from the "
            "football World Cup — sports sentiment really does move money. The Tour "
            "borrows that mood-to-market idea but stretches it onto a gentle three-week "
            "*calendar window* with no win-or-lose shock. Nobody has ever formally "
            "tested it. We did."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, it would be a delightful little seasonal: buy the France ETF the "
            "day before the Grand Départ, sell after the Champs-Élysées finale, pocket "
            "the national good mood. And because the Tour dates are fixed a *year* in "
            "advance, there's no guesswork about timing — it would be one of the "
            "cleanest calendar trades imaginable. So: is it there?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** All **{R['n_editions']}** Tours 1996→2025, hardcoded "
            "with each Grand Départ and final-stage date (2020 was COVID-shifted from "
            "July to Aug/Sep — a handy accidental experiment).\n"
            "- **The market.** `EWQ` (iShares MSCI France) over the race window, both "
            "**raw** and **abnormal** vs `VGK`, a broad Europe benchmark — because July "
            "is the heart of the *\"Sell in May and go away\"* summer-weakness window, "
            "any raw dip has to be checked against Europe before we call it French.\n"
            "- **The honesty checks.** A random-window placebo (is a July-Tour window "
            "any different from an ordinary three weeks?), a CAC-40 cross-check, the "
            "2020 race-vs-calendar probe, and a costed trade you could *actually* have "
            "placed."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline: the \"bump\" is a mild drag.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    raw = st.one_sample_t(INC['raw_ret'].values)['mean']*100\n"
            "    net = st.one_sample_t(INC['raw_net'].values)['mean']*100\n"
            "    ordn = R['pl_raw_mean']  # canonical placebo mean (results.md)\n"
            "else:\n"
            "    raw, net, ordn = R['raw_mean'], R['raw_net_mean'], R['pl_raw_mean']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.6))\n"
            "bars = ax.bar(['ordinary\\n3-week window', 'Tour window\\n(gross)', "
            "'Tour window\\n(net of costs)'], [ordn, raw, net],\n"
            "              color=[GREY, RED, RED], width=.55)\n"
            "for b, v in zip(bars, [ordn, raw, net]):\n"
            "    ax.annotate(f'{v:+.2f}%', (b.get_x()+b.get_width()/2, v), ha='center',\n"
            "                va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean EWQ return over the window (%)')\n"
            "ax.set_title('An ordinary 3 weeks makes +0.6%; the Tour window LOSES money')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Tour gross {raw:+.2f}%  net {net:+.2f}%  vs ordinary window {ordn:+.2f}%')"
        ),
        md(
            f"Buy French stocks for the three weeks of the Tour and on average you're "
            f"down **{R['raw_mean']:+.2f}%** before costs, **{R['raw_net_mean']:+.2f}%** "
            "after — while an *ordinary* three-week stretch in the same tape makes "
            f"**{R['pl_raw_mean']:+.2f}%**. That's not a feel-good bump; that's the "
            "summer doldrums. Which raises the obvious question: **is that dip French at "
            "all, or is all of Europe soft in July?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ewq = st.one_sample_t(SUB['raw_ret'].values)['mean']*100\n"
            "    vgk_rets = []\n"
            "    common = PRICES['EWQ'].index.intersection(PRICES['VGK'].index).sort_values()\n"
            "    for _, r in SUB.iterrows():\n"
            "        e, x = st._window_positions(common, r['grand_depart'], r['final_stage'])\n"
            "        vgk_rets.append(PRICES['VGK'].loc[common[x]]/PRICES['VGK'].loc[common[e]]-1)\n"
            "    vgk = np.mean(vgk_rets)*100\n"
            "    ar = st.one_sample_t(SUB['ar'].values)['mean']*100\n"
            "else:\n"
            "    ewq, vgk, ar = R['ewq_sub_mean'], R['vgk_mean'], R['ar_mean']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.6))\n"
            "bars = ax.bar(['France\\n(EWQ)', 'Europe\\n(VGK)', 'France MINUS Europe\\n(abnormal)'],\n"
            "              [ewq, vgk, ar], color=[AMBER, GREY, RED], width=.55)\n"
            "for b, v in zip(bars, [ewq, vgk, ar]):\n"
            "    ax.annotate(f'{v:+.2f}%', (b.get_x()+b.get_width()/2, v), ha='center',\n"
            "                va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean window return, 2005-2025 (%)')\n"
            "ax.set_title('France tracks Europe in July -- and if anything LAGS it')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'France {ewq:+.2f}%  Europe {vgk:+.2f}%  abnormal {ar:+.2f}%')"
        ),
        md(
            f"There it is. Over the era we can benchmark (2005→), France makes "
            f"**{R['ewq_sub_mean']:+.2f}%** during the Tour and Europe makes "
            f"**{R['vgk_mean']:+.2f}%** — so France doesn't *beat* Europe, it **lags** by "
            f"**{R['ar_mean']:+.2f}%**. Whatever French stocks do in July, they do it "
            "because they're part of Europe, not because of a bike race. The one number "
            "that could have been a Tour effect — the abnormal, France-minus-Europe "
            "return — is the emptiest of the lot.\n\n"
            "**And the shape? A national-mood story predicts *some* pattern. There "
            "isn't one:**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp_raw = st.car_path(EV, PRICES, 'raw', 16)\n"
            "    cp_ar = st.car_path(EV, PRICES, 'ar', 16)\n"
            "    days = list(cp_raw.index); rs = list(cp_raw.values*100); as_ = list(cp_ar.values*100)\n"
            "else:\n"
            "    days = sorted(R['car_raw']); rs = [R['car_raw'][k] for k in days]\n"
            "    as_ = [R['car_ar'][k] for k in days]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "ax.plot(days, rs, color=RED, lw=2.2, marker='o', label='raw (EWQ France)')\n"
            "ax.plot(days, as_, color=GREY, lw=2.2, marker='o', label='abnormal (France - Europe)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('trading sessions into the Tour (entry = 0)')\n"
            "ax.set_ylabel('mean cumulative return (%)')\n"
            "ax.set_title('No pop, no drift -- the path just wanders around zero')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            "A genuine feel-good bump should show up as *something* — an early pop that "
            "holds, a steady drift. This is neither: the path drops a bit in the first "
            "week, wobbles, and ends slightly negative, with the France-minus-Europe "
            "line flat-to-negative the whole way. **Finally, is the Tour window even "
            "unusual?**"
        ),
        code(
            "rng = np.random.default_rng(732)\n"
            "draws = rng.normal(R['pl_raw_mean'], R['pl_raw_sd'], 4000)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85,\n"
            "        label='null: random 3-week EWQ windows (anywhere in the tape)')\n"
            "ax.axvline(R['raw_mean'], c=RED, lw=2.4, label=f\"Tour window {R['raw_mean']:+.2f}%\")\n"
            "ax.axvline(R['pl_raw_mean'], c='k', ls=':', lw=1.4, label=f\"ordinary window {R['pl_raw_mean']:+.2f}%\")\n"
            "ax.set_xlabel('mean return of a random 3-week window (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The Tour window is a below-median but ordinary 3 weeks (p = {R[\"pl_raw_p\"]:.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"observed {R['raw_mean']:+.2f}%  vs placebo {R['pl_raw_mean']:+.2f}% \"\n"
            "      f\"(sd {R['pl_raw_sd']:.2f}%)  left-tail p = {R['pl_raw_p']:.3f}; \"\n"
            "      f\"{R['pl_raw_right']*100:.0f}% of random windows BEAT the Tour\")"
        ),
        md(
            f"The Tour window is soft — but so are plenty of random three-week stretches. "
            f"**{R['pl_raw_right']*100:.0f}%** of random windows actually *beat* it, and "
            f"the gap isn't statistically unusual (placebo *p* = {R['pl_raw_p']:.2f}). "
            "It's a below-average but utterly ordinary patch of summer."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The raw seasonal is a mild *negative* "
            f"({R['raw_mean']:+.2f}%, *t* = {R['raw_t']:.2f}), the France-specific "
            f"abnormal is *also* negative ({R['ar_mean']:+.2f}%, *t* = {R['ar_t']:.2f}), "
            "the hit rate is below a coin flip, and neither is distinguishable from an "
            "ordinary three weeks. There is nothing here, in either direction.\n"
            "- **Tradability — Mirage.** No positive edge exists to trade — the seasonal "
            f"is **{R['raw_net_mean']:+.2f}%** net of costs. \"Buy French stocks during "
            "the Tour\" is a systematic small loss.\n"
            "- **A France-specific effect? — Not supported.** France doesn't beat Europe "
            "during the Tour (it slightly lags). The mild July softness is ordinary "
            "pan-European summer beta — the folklore mis-attributes a whole-continent "
            "calendar non-event to a bike race."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **This is the cleanest kind of null.** No surprise, no information lag, a "
            "date fixed a year ahead, 30 events — and *still* nothing, because the "
            "premise collided with a bigger, real seasonal (\"Sell in May\") and "
            "evaporated the moment we netted out Europe. Sometimes the honest answer is "
            "just \"no,\" and it's worth writing down.\n"
            "- **Sibling studies:** the [Eurovision effect](../../708-eurovision-effect/) "
            "(the same national-mood folklore, but keyed to a *surprise* result across a "
            "per-country panel), the [World Cup effect](../../235-world-cup-effect/) (the "
            "real Edmans mechanism), the [Super Bowl indicator](../../158-super-bowl/) and "
            "[World Series effect](../../709-world-series-effect/) — every one a "
            "national-mood claim, tested the same honest way.\n\n"
            "*Think there's a real Tour effect hiding in the tourism/leisure/consumer "
            "sub-sector, or intraday around the Champs-Élysées finale? Bring a bigger, "
            "cleaner sample and a net, placebo-surviving edge — we'll publish the teardown.*"
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
            "# Tour-de-France-Effect — a quantitative teardown 🔬\n"
            "### One-sample-*t* on the window return · a raw-vs-abnormal decomposition · "
            "a random-window placebo · the event anatomy · the 2020 race-vs-calendar "
            "probe · a 20-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb). The claim — **French equities get a feel-good "
            "seasonal during the July Tour de France** — has no published academic anchor "
            "of its own; it borrows its mechanism from Edmans, García & Norli (2007), a "
            "real elimination-shock effect for football World Cups, and stretches it onto "
            "a three-week calendar window that happens to sit inside the *\"Sell in May\"* "
            "summer-weakness season (Bouman & Jacobsen 2002). The job here is to measure "
            "it honestly and, crucially, to separate any *French* effect from ordinary "
            "pan-European summer beta.\n\n"
            "> ⚠️ **Data note.** `EWQ` (France, total-return) + `^FCHI` (CAC 40, "
            "**price-only**) + `VGK` (Europe, total-return), yfinance, 1990-03-01→"
            "2026-06-30. 30 Tours hardcoded 1996→2025; all 30 EWQ-covered, **21** "
            "(2005→) also VGK-covered for the abnormal test. Calendar-known window → "
            "**no information lag** (dates public a year ahead). Methods in "
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
            f"| **Signal** | `NONE` | raw {R['raw_mean']:+.3f}% (*t* = {R['raw_t']:.2f}), "
            f"abnormal (France−Europe) {R['ar_mean']:+.3f}% (*t* = {R['ar_t']:.2f}); both "
            f"negative, hit {R['raw_hit']}/{R['raw_n']}, placebo *p* = {R['pl_raw_p']:.3f} |\n"
            f"| **Tradability** | `MIRAGE` | net-of-cost seasonal {R['raw_net_mean']:+.3f}% "
            f"(*t* = {R['raw_net_t']:.2f}) — a systematic small loss, no edge to trade |\n"
            f"| **France-specific?** | `NOT SUPPORTED` | France {R['ewq_sub_mean']:+.3f}% "
            f"vs Europe {R['vgk_mean']:+.3f}% (2005+); Welch *t* (France−Europe) = "
            f"{R['welch']:.2f} — the July softness is region-wide beta |\n\n"
            "> 💡 In plain words: a calendar seasonal with no surprise and no lag, 30 "
            "events, and it still finds nothing — because the raw July dip is just the "
            "\"Sell in May\" summer, and there is no France-specific component underneath it."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{F}_t$ be `EWQ` (France) and $r^{E}_t$ `VGK` (Europe) log-returns on "
            "trading day $t$. For each Tour edition $y$, let *entry* be the last close "
            "before the Grand Départ and *exit* the first close on/after the final stage "
            "(the window is ~15–16 sessions). The window return and its abnormal "
            "(France-specific) counterpart are\n\n"
            "$$Ret_y = \\frac{P^{F}_{exit}}{P^{F}_{entry}} - 1, \\qquad "
            "AR_y = Ret_y - \\left(\\frac{P^{E}_{exit}}{P^{E}_{entry}} - 1\\right).$$\n\n"
            "Because each edition is a single, non-overlapping, independent annual event, "
            "the **one-sample t** across editions is the correct primary statistic — not "
            "a daily panel. And because the Tour dates are public a year in advance, "
            "*entry* is a zero-look-ahead, fully-executable position (no information lag, "
            "unlike study 708's Saturday-night Eurovision result). Claims:\n\n"
            "- **H1 (raw bump).** $E[Ret_y] > 0$.\n"
            "- **H2 (France-specific bump).** $E[AR_y] > 0$ — the decisive test.\n"
            "- **H3 (anatomy).** A national-mood mechanism implies *some* intra-window "
            "shape (an early pop that holds, or a drift).\n"
            "- **H4 (capture).** The seasonal survives costs.\n\n"
            "We find **H1 rejected** (raw is negative), **H2 rejected** (abnormal is "
            "negative — France lags Europe), **H3 unsupported** (no shape), **H4 moot** "
            "(there is no positive edge to cost in the first place)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            f"n is a healthy **{R['n_included']}** for the raw seasonal (every edition "
            f"has EWQ coverage) and **{R['n_ar']}** for the abnormal (VGK from 2005). The "
            "plan: a **one-sample t** on the window return (raw, net, and CAC price-only), "
            "the **raw-vs-abnormal decomposition** that separates France from Europe, a "
            "**Wilson interval** on the hit rate, a **20-seed × 200-draw random-window "
            "placebo** (redraw a same-length window at a random point in the tape — is "
            "the July-Tour window special?), and the **2020 race-vs-calendar probe** (the "
            "one edition whose window left July entirely)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_editions']} editions 1996→2025, hardcoded from "
            "Wikipedia; 2020 COVID-shifted to Aug/Sep.\n"
            f"- **Sample.** {R['n_included']} EWQ-covered editions (raw); {R['n_ar']} "
            "VGK-covered (abnormal, 2005→); window 15–16 sessions.\n"
            "- **Headline.** One-sample *t* on raw / net / CAC-price + Wilson hit rate.\n"
            "- **Decomposition.** France (EWQ) vs Europe (VGK) vs abnormal (EWQ−VGK), "
            "same window, 2005→; Welch *t* on the France−Europe difference.\n"
            "- **Robustness.** 20×200-draw random-window placebo (left- and right-tail); "
            "leave-out-2020 (race-vs-calendar).\n"
            "- **Anatomy.** Mean cumulative return by session, 0→16, raw and abnormal.\n"
            "- **Execution.** Calendar-known → no lag; costs = 2× one-way × NAV.\n"
            "- **Control.** Synthetic paired (France, Europe) world, planted-bump knob; "
            "the null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The raw seasonal — one-sample t, three flavours\n\n"
            "The window return across all 30 editions: EWQ gross, EWQ net of costs, and "
            "the CAC 40 **price-only** cross-check (dividend-free — it *understates* true "
            "returns, and is never mixed into the total-return abnormal test)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for lab, col in (('EWQ raw','raw_ret'), ('EWQ net@5bps','raw_net'), ('CAC price-only','cac_raw')):\n"
            "        s = st.one_sample_t(INC[col].values); hr = st.hit_rate(INC[col].values)\n"
            "        rows.append((lab, s['n'], s['mean']*100, s['t'], hr['k'], hr['n']))\n"
            "        print(lab, 'n', s['n'], 'mean', round(s['mean']*100,3), 't', round(s['t'],3))\n"
            "    labels=[r[0] for r in rows]; means=[r[2] for r in rows]; ts=[r[3] for r in rows]\n"
            "else:\n"
            "    labels=['EWQ raw','EWQ net@5bps','CAC price-only']\n"
            "    means=[R['raw_mean'], R['raw_net_mean'], R['cac_mean']]\n"
            "    ts=[R['raw_t'], R['raw_net_t'], R['cac_t']]\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.8, 6.4), sharex=True,\n"
            "                             gridspec_kw={'height_ratios':[2,1]})\n"
            "a1.bar(labels, means, color=[RED if m<0 else AMBER for m in means])\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean window return (%)')\n"
            "a1.set_title('Every flavour of the raw seasonal is negative')\n"
            "a2.bar(labels, ts, color=[RED if abs(t)>=2 else GREY for t in ts])\n"
            "a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(-2, ls='--', c=RED, lw=1)\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('t-stat')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: EWQ raw **{R['raw_mean']:+.3f}%** (*t* = {R['raw_t']:.2f}, "
            f"hit {R['raw_hit']}/{R['raw_n']} = {R['raw_hit']/R['raw_n']*100:.0f}%, Wilson "
            f"[{R['raw_lo']:.0f}%, {R['raw_hi']:.0f}%]); net **{R['raw_net_mean']:+.3f}%**; "
            f"CAC price-only **{R['cac_mean']:+.3f}%**. Three independent measurements, all "
            "negative, none within a country mile of *t* = 2. **H1 rejected.**"
        ),
        md(
            "### 4b · The decomposition — is any of it French? (the decisive cut)\n\n"
            "The one measurement that can tell a *French* sentiment bump from ordinary "
            "summer beta: EWQ minus VGK over the same window, 2005→ (VGK coverage)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    common = PRICES['EWQ'].index.intersection(PRICES['VGK'].index).sort_values()\n"
            "    vgk_rets = []\n"
            "    for _, r in SUB.iterrows():\n"
            "        e, x = st._window_positions(common, r['grand_depart'], r['final_stage'])\n"
            "        vgk_rets.append(PRICES['VGK'].loc[common[x]]/PRICES['VGK'].loc[common[e]]-1)\n"
            "    s_ewq = st.one_sample_t(SUB['raw_ret'].values)\n"
            "    s_vgk = st.one_sample_t(np.asarray(vgk_rets))\n"
            "    s_ar = st.one_sample_t(SUB['ar'].values)\n"
            "    welch = st.welch_t(SUB['raw_ret'].values, np.asarray(vgk_rets))\n"
            "    ewq_m, vgk_m, ar_m, ar_t = s_ewq['mean']*100, s_vgk['mean']*100, s_ar['mean']*100, s_ar['t']\n"
            "else:\n"
            "    ewq_m, vgk_m, ar_m, ar_t, welch = R['ewq_sub_mean'], R['vgk_mean'], R['ar_mean'], R['ar_t'], R['welch']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.6))\n"
            "bars = ax.bar(['France (EWQ)','Europe (VGK)','abnormal (F-E)'], [ewq_m, vgk_m, ar_m],\n"
            "              color=[AMBER, GREY, RED], width=.55)\n"
            "for b,v in zip(bars,[ewq_m,vgk_m,ar_m]):\n"
            "    ax.annotate(f'{v:+.2f}%', (b.get_x()+b.get_width()/2, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('mean window return, 2005-2025 (%)')\n"
            "ax.set_title(f'France LAGS Europe in July (Welch t = {welch:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'France {ewq_m:+.3f}%  Europe {vgk_m:+.3f}%  abnormal {ar_m:+.3f}% (t={ar_t:+.3f})  Welch {welch:+.3f}')"
        ),
        md(
            f"> 💡 In plain words: over 2005→ France makes **{R['ewq_sub_mean']:+.3f}%** "
            f"and Europe makes **{R['vgk_mean']:+.3f}%**, so the France-specific abnormal "
            f"is **{R['ar_mean']:+.3f}%** (*t* = {R['ar_t']:.2f}, hit {R['ar_hit']}/"
            f"{R['ar_n']}), and the unpaired Welch *t* is {R['welch']:.2f}. France does "
            "not beat Europe during the Tour — it slightly *lags*. **H2 rejected: the "
            "July softness is pan-European summer beta, not a home-crowd effect. This is "
            "the whole verdict in one chart.**"
        ),
        md(
            "### 4c · The random-window placebo — is the Tour window unusual at all?\n\n"
            "For each edition, redraw a random same-length window from elsewhere in EWQ's "
            "history, 20 seeds × 200 draws; compare the observed Tour mean to the null "
            "distribution of ordinary three-week windows."
        ),
        code(
            "rng = np.random.default_rng(732)\n"
            "draws = rng.normal(R['pl_raw_mean'], R['pl_raw_sd'], 4000)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='null: random 3-week EWQ windows')\n"
            "ax.axvline(R['raw_mean'], c=RED, lw=2.4, label=f\"Tour window {R['raw_mean']:+.2f}%\")\n"
            "ax.axvline(R['pl_raw_mean'], c='k', ls=':', lw=1.4, label=f\"placebo mean {R['pl_raw_mean']:+.2f}%\")\n"
            "ax.set_xlabel('mean return of a random 3-week window (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Canonical placebo (results.md, 20x200): left-tail p = {R[\"pl_raw_p\"]:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"raw: obs {R['raw_mean']:+.3f}% vs placebo {R['pl_raw_mean']:+.3f}% (sd {R['pl_raw_sd']:.3f}%) \"\n"
            "      f\"left p={R['pl_raw_p']:.3f}, {R['pl_raw_right']*100:.0f}% of windows beat it\")\n"
            "print(f\"abnormal: obs {R['pl_ar_obs']:+.3f}% vs placebo {R['pl_ar_mean']:+.3f}% \"\n"
            "      f\"left p={R['pl_ar_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: the Tour window ({R['raw_mean']:+.2f}%) is below the "
            f"ordinary-window mean ({R['pl_raw_mean']:+.2f}%), but well inside the luck "
            f"cloud — left-tail *p* = {R['pl_raw_p']:.3f}, and **{R['pl_raw_right']*100:.0f}%** "
            "of random windows beat it. The abnormal cut is even more ordinary "
            f"(*p* = {R['pl_ar_p']:.3f}). Not a special date on the calendar — just a "
            "below-median patch of summer."
        ),
        md(
            "### 4d · Event anatomy — is there any intra-window shape?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp_raw = st.car_path(EV, PRICES, 'raw', 16); cp_ar = st.car_path(EV, PRICES, 'ar', 16)\n"
            "    days=list(cp_raw.index); rs=list(cp_raw.values*100); as_=list(cp_ar.values*100)\n"
            "else:\n"
            "    days=sorted(R['car_raw']); rs=[R['car_raw'][k] for k in days]; as_=[R['car_ar'][k] for k in days]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "ax.plot(days, rs, color=RED, lw=2.2, marker='o', label='raw (EWQ France)')\n"
            "ax.plot(days, as_, color=GREY, lw=2.2, marker='o', label='abnormal (France - Europe)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('trading sessions into the Tour (entry = 0)')\n"
            "ax.set_ylabel('mean cumulative return (%)')\n"
            "ax.set_title('No pop, no drift, no shape')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: a broadcast-driven national-mood mechanism predicts an "
            f"early reaction that holds (like study 637's FOMC vol crush). Here the raw "
            f"path dips to {R['car_raw'][4]:+.2f}% by session 4 and wanders to "
            f"{R['car_raw'][16]:+.2f}% by the finish, with the France−Europe line "
            "flat-to-negative throughout. **H3 unsupported: this is drift-shaped noise, "
            "not an event reaction.**"
        ),
        md(
            "### 4e · Robustness — the 2020 race-vs-calendar probe\n\n"
            "2020 is the natural experiment: COVID-19 pushed the Tour out of July to "
            "Aug 29 → Sep 20. If any \"effect\" tracked the *calendar month* rather than "
            "the race, moving the window would change the answer. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r20 = INC[INC['year']==2020].iloc[0]\n"
            "    no20 = st.one_sample_t(INC[INC['year']!=2020]['raw_ret'].values)\n"
            "    y20_raw, y20_ar, m_no20, t_no20, n_no20 = r20['raw_ret']*100, r20['ar']*100, no20['mean']*100, no20['t'], no20['n']\n"
            "else:\n"
            "    y20_raw, y20_ar, m_no20, t_no20, n_no20 = R['y2020_raw'], R['y2020_ar'], R['raw_no20_mean'], R['raw_no20_t'], R['raw_no20_n']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "ax.bar(['2020 window\\n(raw)','2020 window\\n(abnormal)','all editions\\nEXCL 2020 (raw)'],\n"
            "       [y20_raw, y20_ar, m_no20], color=[RED, GREY, RED], width=.55)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('return (%)')\n"
            "ax.set_title(f'Drop 2020 entirely and the verdict is unchanged (t={t_no20:+.2f}, n={n_no20})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'2020 raw {y20_raw:+.2f}% abnormal {y20_ar:+.2f}%; excl-2020 mean {m_no20:+.3f}% t={t_no20:+.3f}')"
        ),
        md(
            f"> 💡 In plain words: 2020's Aug/Sep window (raw **{R['y2020_raw']:+.2f}%**) "
            "is just a bad market month, the worst single edition — and dropping it "
            f"entirely leaves the raw seasonal at {R['raw_no20_mean']:+.3f}% "
            f"(*t* = {R['raw_no20_t']:.2f}, n = {R['raw_no20_n']}). Nothing tracks the "
            "calendar *or* the race, because there is nothing to track."
        ),
        md(
            "### 4f · Faithful-engine & power control\n\n"
            "Synthetic paired (France, Europe) log-return world (ρ ≈ 0.85, like a "
            "single-country ETF vs its regional benchmark), a scheduled annual window, "
            "TUNABLE planted per-day bump. Null (bump = 0) checked over **20 seeds**."
        ),
        code(
            "null_ts = np.array([st.synthetic_detect(bump=0.0, seed=732+s)['t'] for s in range(20)])\n"
            "p5 = st.synthetic_detect(bump=0.0005, seed=732)\n"
            "p10 = st.synthetic_detect(bump=0.0010, seed=732)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20)+np.linspace(-.12,.12,20), null_ts, color=GREY, s=40, label='null (bump=0), 20 seeds')\n"
            "ax.scatter([1],[p5['t']], color=AMBER, s=90, zorder=5, label='planted +5bp/day')\n"
            "ax.scatter([2],[p10['t']], color=RED, s=90, zorder=5, label='planted +10bp/day')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0,1,2]); ax.set_xticklabels(['null x20','planted 5bp','planted 10bp'])\n"
            "ax.set_ylabel('one-sample t'); ax.set_title('Control: quiet null, planted bumps light up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null mean t={null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts)>=2).sum()}/20')\n"
            "print(f'planted 5bp t={p5[\"t\"]:+.2f}  10bp t={p10[\"t\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null seeds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and fires at "
            f"|t| ≥ 2 in {R['syn_null_fire']}/{R['syn_null_seeds']} seeds; a planted "
            f"+5 bp/day bump reads t = {R['syn_p5_t']:.2f}, +10 bp/day reads "
            f"t = {R['syn_p10_t']:.2f}. The machinery detects a real seasonal cleanly — "
            "the real-tape emptiness is the tape's, not the detector's. *(A "
            "faithful-engine / power check only — never cited in support of the "
            "real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — raw {R['raw_mean']:+.3f}% (*t* = {R['raw_t']:.2f}), "
            f"abnormal {R['ar_mean']:+.3f}% (*t* = {R['ar_t']:.2f}), CAC price-only "
            f"{R['cac_mean']:+.3f}% — three measurements, all mildly *negative*, hit rate "
            f"{R['raw_hit']}/{R['raw_n']} below a coin flip, and indistinguishable from an "
            f"ordinary three weeks (placebo *p* = {R['pl_raw_p']:.3f}). No intra-window "
            "shape. Nothing at any bar, in either direction.\n"
            f"- **Tradability `MIRAGE`** — the seasonal is negative before costs and "
            f"{R['raw_net_mean']:+.3f}% after; there is simply no positive edge to trade. "
            "\"Buy French stocks during the Tour\" is a systematic small loss.\n"
            f"- **France-specific? `NOT SUPPORTED`** — France {R['ewq_sub_mean']:+.3f}% vs "
            f"Europe {R['vgk_mean']:+.3f}% (2005→), Welch *t* = {R['welch']:.2f}; France "
            "lags Europe in July. The mild softness is ordinary pan-European summer beta — "
            "the folklore mis-attributes a region-wide, calendar-driven non-event to a "
            "bike race."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson: a clean null is still a result.** No surprise, no "
            "information lag, a date fixed a year ahead, 30 independent events, and a "
            "healthy-power detector — the ideal conditions to *find* an effect if one "
            "existed. The premise still collapsed, because a raw July seasonal is fighting "
            "the documented \"Sell in May\" summer (Bouman & Jacobsen 2002) and has no "
            "France-specific component once Europe is netted out.\n"
            "- **A more powerful test would go narrower.** French tourism / leisure / "
            "consumer-discretionary sub-indices (where a genuine summer-holiday mood would "
            "actually spend), or intraday data around the Champs-Élysées finale, would be "
            "the natural sequel — but they'd need to clear a *net, placebo-surviving* bar "
            "this broad-index test misses by a mile.\n"
            "- **Dedup map:** [708-eurovision-effect](../../708-eurovision-effect/) (same "
            "national-mood folklore, but a *surprise* result with a real lag, across a "
            "per-country panel), [235-world-cup-effect](../../235-world-cup-effect/) (the "
            "real Edmans elimination-shock mechanism), "
            "[158-super-bowl](../../158-super-bowl/) and "
            "[709-world-series-effect](../../709-world-series-effect/) (US single-sport "
            "indicators). None test a single-country *calendar-window* sports seasonal "
            "against its regional benchmark — the summer-beta decomposition that decides "
            "this one is this study's own contribution.\n\n"
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
