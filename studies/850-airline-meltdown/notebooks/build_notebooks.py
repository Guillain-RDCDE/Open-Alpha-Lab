"""Generate the two narrative notebooks for Study 850 (Airline Operational Meltdown).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached
SPY/LUV/DAL/UAL/AAL/BA tapes under ../_cache/ and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance
# SPY/LUV/DAL/UAL/AAL/BA 2014-01-02 -> 2026-06-30; 10 hardcoded operational meltdowns
# 2016-08-08 -> 2024-07-19, 9 with price coverage — SAVE/Spirit delisted).
R = dict(
    n_events_all=10, n_events=9, cal_lo="2016-08-08", cal_hi="2024-07-19",
    # per-event CAR in bps: (date, ticker, day0, week, month, drift)
    events=[
        ("2016-08-08", "DAL", -23.9, -184.9, 938.1, 962.0),
        ("2017-01-30", "DAL", -331.1, -491.3, -769.9, -438.8),
        ("2017-04-10", "UAL", 69.8, -17.9, 506.4, 436.7),
        ("2019-03-11", "BA", -751.0, -1554.9, -2450.4, -1699.4),
        ("2021-10-11", "LUV", -280.8, -813.4, -928.4, -647.6),
        ("2021-11-01", "AAL", 287.4, 1119.9, -1038.8, -1326.2),
        ("2022-12-27", "LUV", -564.8, -944.7, -422.5, 142.3),
        ("2024-01-08", "BA", -952.9, -1548.9, -2391.0, -1438.1),
        ("2024-07-19", "DAL", 193.9, -12.5, -1038.6, -1232.5),
    ],
    # headline by horizon: name -> (mean_bps, t, t_nw, down, n, wilson_lo%, wilson_hi%)
    head={
        "day0": (-261.5, -1.82, -4.28, 6, 9, 35, 88),
        "week": (-494.3, -1.76, -4.59, 8, 9, 56, 98),
        "month": (-843.9, -2.24, -2.74, 7, 9, 45, 94),
        "drift": (-582.4, -1.88, -2.18, 6, 9, 35, 88),
    },
    # placebo: horizon -> (obs_bps, pmean_bps, psd_bps, p_left, n_draws)
    placebo={
        "day0": (-261.5, -0.5, 75.2, 0.002, 5000),
        "month": (-843.9, -2.8, 383.2, 0.014, 5000),
        "drift": (-582.4, -2.7, 375.3, 0.056, 5000),
    },
    # robustness decomposition: subsample -> (day0_bps, day0_t, month_bps, month_t, n)
    split={
        "all": (-261.5, -1.82, -843.9, -2.24, 9),
        "air": (-92.8, -0.79, -393.4, -1.30, 7),
        "boe": (-851.9, -8.44, -2420.7, -81.48, 2),
    },
    # leave-one-out on the full-sample MONTH one-sample t (the load-bearing pair)
    loo_drop_ba2019=-1.78, loo_drop_ba2024=-1.77,
    pre2022_month_t=-1.26, post2022_month_t=-2.21,
    # short timer: hold -> (gross_bps, net_bps, t_net, win%, air_net_bps, air_t)
    timer={
        5: (213.4, 197.5, 0.87, 78, -18.2, -0.08),
        10: (258.9, 237.0, 1.09, 67, 96.3, 0.37),
        21: (176.1, 141.1, 0.44, 56, -25.7, -0.07),
    },
    # synthetic control
    syn_null_mean=-0.08, syn_null_sd=1.05, syn_null_fire=1,
    syn_planted_t=-8.48, syn_planted_bps=-283.5,
    fp_spy="6ce22fa749d2", fp_luv="710a0aa31fcf", fp_dal="24f189e4696e",
    fp_ual="d6b73e44a59c", fp_aal="0b16d927b778", fp_ba="4be94ccd2936",
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Reputational%20shock%3F: Mixed](https://img.shields.io/badge/Reputational%20shock%3F-Mixed-8b949e?style=flat-square)\n\n"
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

from airline_meltdown import data, strategy as st

EVENTS = data.coverable_events()
HAVE_REAL = data.have_real()
if HAVE_REAL:
    SPY, STOCKS = data.load_prices()
    CARS = st.stack_event_cars(EVENTS, SPY, STOCKS)
else:
    SPY = STOCKS = CARS = None
print("real cache present:", HAVE_REAL, "| coverable meltdowns:", len(EVENTS))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# When an airline melts down, does its stock? ✈️\n"
            "### Groundings, cancellation collapses, viral PR disasters — and what they "
            "actually do to the share price\n\n"
            + BADGES +
            "Southwest strands a million travellers over Christmas. A software bug grounds "
            "Delta's entire fleet. United drags a bloodied passenger off a plane on video. "
            "Boeing's 737 MAX gets grounded — twice. Every time, the same instinct: *this "
            "will hurt the stock — the brand is damaged, customers will flee, and the "
            "share price should carry the scar for weeks.*\n\n"
            "That's the claim we test on **10 of the most infamous airline and Boeing "
            "operational meltdowns since 2016** — measuring the *implicated* company's own "
            "stock (not the market), both in the days around the meltdown and over the "
            "month that follows.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the "
            "airlines-vs-Boeing decomposition? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the average meltdown stock drop over the next month? | **On paper, "
            f"yes — {R['head']['month'][0]:+.0f} bps** (about −8.4%), which even looks "
            "statistically real. |\n"
            "| ...but is that a *reputation* effect? | **No — it's Boeing.** Take out the "
            "two Boeing 737-MAX groundings (which ground the actual product line — a real "
            f"earnings hit, not just bad press) and the drop shrinks to "
            f"**{R['split']['air'][2]:+.0f} bps** and stops being significant. |\n"
            "| Do the pure airline meltdowns (cancellations, IT outages, the United "
            "dragging) hurt the stock? | **Not reliably.** Several actually had "
            "*positive* one-month returns — the operation gets fixed in days and the "
            "market moves on. |\n"
            "| Could you trade it (short the meltdown)? | **Not really.** It's positive "
            "in-sample but rests entirely on those two Boeing events — strip them and "
            "it's a coin flip. |\n\n"
            "> A tidy story with one real exception baked in: a grounded *product line* "
            "genuinely dents the stock; a bad weekend of cancellations mostly doesn't."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"An operational meltdown is a reputational catastrophe. Passengers are "
            "furious, the story leads every news cycle for a week, regulators open "
            "investigations, and the brand takes lasting damage. The stock should drop on "
            "the news and keep bleeding as the reputational cost sinks in.\"*\n\n"
            "It's an intuitive, almost obvious idea — and the academic event-study "
            "literature on product recalls and corporate crises gives it a real, if "
            "modest, prior. The catch that literature also flags: the price reaction "
            "concentrates in a **short window**, and it mostly reflects **fundamental** "
            "costs (grounded planes, lost bookings, lawsuits), not free-floating "
            "\"reputation.\""
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a meltdown reliably dented the stock for a month, that's a tradable, "
            "repeatable pattern — short the airline the day the meltdown breaks, cover a "
            "few weeks later. And it would be a clean example of markets pricing "
            "**soft** damage (brand, goodwill) rather than just hard cash flows.\n\n"
            "So we ask: does the implicated stock actually drop, does the drop *stick* "
            "for a month, and — crucially — is it about reputation or about real, "
            "fundamental damage?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** **{R['n_events_all']}** famous operational meltdowns "
            f"{R['cal_lo']} → {R['cal_hi']} — Delta's 2016 outage and 2024 CrowdStrike "
            "collapse, United's 2017 passenger-dragging, Boeing's 2019 and 2024 MAX "
            "groundings, Southwest's 2021 and 2022 cancellation collapses, American's "
            "2021 meltdown, Spirit's 2021 wave. (Spirit's stock is gone — it went "
            "bankrupt and delisted — so 9 have price data.)\n"
            "- **The measurement.** For each one, the *implicated* stock's return around "
            "the event **minus what the market did** (so we're not just measuring "
            "\"stocks went up that month\").\n"
            "- **The luck check.** Hand each event a *random* date instead, 5,000 times — "
            "how often does a random calendar produce a drop this big?\n"
            "- **The tell.** Split airlines vs. Boeing — because a grounded product line "
            "is a different animal from a bad weekend of cancellations."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Average one-month abnormal return of the implicated "
            "stock across all 9 meltdowns — and where a *random* calendar of the same 9 "
            "stocks would land."
        ),
        code(
            "if HAVE_REAL:\n"
            "    month_bps = st.car_stats(CARS, 'month')['mean_bps']\n"
            "else:\n"
            "    month_bps = R['head']['month'][0]\n"
            "pmean = R['placebo']['month'][1]\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['meltdown stock\\n(n=9, 1 month)', 'random date\\n(placebo mean)'],\n"
            "       [month_bps, pmean], color=[RED, GREY], width=.55)\n"
            "for i, v in enumerate([month_bps, pmean]):\n"
            "    ax.annotate(f'{v:+.0f} bps', (i, v), ha='center', va='top' if v < 0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('average abnormal return, event month (bps)')\n"
            "ax.set_title('On paper, meltdown stocks drop ~8% over the next month')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'meltdown month {month_bps:+.0f} bps vs random {pmean:+.0f} bps')"
        ),
        md(
            f"That looks like a real effect: **{R['head']['month'][0]:+.0f} bps** over the "
            f"month, versus roughly zero for a random date, and a random calendar beats it "
            f"only about **{R['placebo']['month'][3]*100:.1f}%** of the time. But before "
            "we celebrate — *which* meltdowns are doing the work?\n\n"
            "**The tell: split the two Boeing groundings out from the airline meltdowns.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    allm = st.car_stats(CARS, 'month')['mean_bps']\n"
            "    airm = st.car_stats(CARS[CARS['ticker'] != 'BA'], 'month')['mean_bps']\n"
            "    boem = st.car_stats(CARS[CARS['ticker'] == 'BA'], 'month')['mean_bps']\n"
            "else:\n"
            "    allm, airm, boem = R['split']['all'][2], R['split']['air'][2], R['split']['boe'][2]\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.4))\n"
            "ax.bar(['all 9', 'airlines only\\n(no Boeing, n=7)', 'Boeing only\\n(2 MAX groundings)'],\n"
            "       [allm, airm, boem], color=[GREY, GREEN, RED], width=.6)\n"
            "for i, v in enumerate([allm, airm, boem]):\n"
            "    ax.annotate(f'{v:+.0f} bps', (i, v), ha='center', va='top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean 1-month abnormal return (bps)')\n"
            "ax.set_title('The whole effect is Boeing: strip it out and the drop nearly halves')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'all {allm:+.0f} | airlines-only {airm:+.0f} | Boeing-only {boem:+.0f} bps')"
        ),
        md(
            "There it is. The two **Boeing MAX groundings** average a brutal "
            f"**{R['split']['boe'][2]:+.0f} bps** over the month — because grounding the "
            "MAX halts deliveries and torches earnings, a *fundamental* hit. The seven "
            f"pure airline meltdowns average only **{R['split']['air'][2]:+.0f} bps**, and "
            "that's no longer statistically distinguishable from zero. A cancellation "
            "collapse is a miserable week for travellers — but the airline restores its "
            "schedule in days, and the stock mostly shrugs.\n\n"
            "**Look at the individual events** to see how mixed the airline side is:"
        ),
        code(
            "rows = R['events']\n"
            "labels = [f\"{d[5:]} {t}\" for (d, t, *_ ) in rows]\n"
            "vals = [e[4] for e in rows]           # month CAR bps\n"
            "cols = [RED if e[1] == 'BA' else (GREEN if e[4] > 0 else AMBER) for e in rows]\n"
            "fig, ax = plt.subplots(figsize=(10.2, 4.6))\n"
            "ax.bar(labels, vals, color=cols, width=.66)\n"
            "for i, v in enumerate(vals):\n"
            "    ax.annotate(f'{v:+.0f}', (i, v), ha='center', va='top' if v < 0 else 'bottom', fontsize=8)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('1-month abnormal return (bps)')\n"
            "ax.set_title('Two Boeing groundings (red) dominate; airline meltdowns (amber/green) are mixed')\n"
            "plt.xticks(rotation=30, ha='right'); plt.tight_layout(); plt.show()\n"
            "print('Boeing =', [e[0] for e in rows if e[1]=='BA'], '| positive-month meltdowns:',\n"
            "      [e[0] for e in rows if e[1]!='BA' and e[4] > 0])"
        ),
        md(
            "Delta's 2016 outage (**+938 bps**) and United's 2017 dragging crisis "
            "(**+506 bps**) actually had *positive* one-month abnormal returns — the "
            "stocks were fine a month later. The two red Boeing bars are in a different "
            "league entirely, and they're the reason the aggregate looks significant."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The 9-event average one-month drop "
            f"(**{R['head']['month'][0]:+.0f} bps**, placebo *p* = "
            f"{R['placebo']['month'][3]:.3f}) looks real but isn't robust: strip the two "
            "Boeing groundings — fundamental product-line shocks, not reputation — and "
            f"it fades to **{R['split']['air'][2]:+.0f} bps** and loses significance.\n"
            "- **Tradability — Fragile.** Shorting the meltdown works in-sample but rests "
            "on two events; airlines-only it's roughly a coin flip.\n"
            "- **Reputational shock vs quick fade? — Mixed.** A grounded product line "
            "dents the stock (Boeing). A bad weekend of cancellations mostly fades."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **N is the honest limit.** Nine coverable meltdowns is a tiny sample; a "
            "small reputational effect could easily hide in it. The clean finding isn't "
            "\"reputation never matters\" — it's \"in this tradable sample, the only "
            "durable stock damage came from a *fundamental* grounding, not from bad "
            "press.\"\n"
            "- **Where a real version might live:** a much larger register of smaller "
            "operational disruptions, forward-booking or app-store-rating data as a "
            "reputation proxy, or the credit/CDS market rather than equity.\n"
            "- **Sibling studies:** "
            "[707-plane-crash-effect](../../707-plane-crash-effect/) (fatal crashes → "
            "broad-market mood), [554-airline-bookings](../../554-airline-bookings/) (an "
            "alt-data demand signal) and "
            "[313-geopolitical-shock](../../313-geopolitical-shock/) (wars/attacks → "
            "market-wide) — related triggers, different targets.\n\n"
            "*Think a bigger meltdown register or a reputation proxy would find the "
            "effect the equity tape misses? Show it — out-of-sample, after costs — then "
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
            "# Airline Operational Meltdown — a quantitative teardown 🔬\n"
            "### A single-name market-model event study · a same-ticker random-date "
            "placebo · the airlines-vs-Boeing decomposition and leave-one-out · a costed "
            "short-the-meltdown timer · a 20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb). The claim — a very public operational meltdown "
            "is a *reputational* shock that dents the implicated carrier's own stock "
            "around the event and over the following month — has a modest event-study "
            "prior. The job here is to measure it honestly on the tradable tape, then "
            "separate the *reputational* story from the *fundamental* one.\n\n"
            "> ⚠️ **Data note.** SPY + LUV/DAL/UAL/AAL/BA total-return closes (2014→2026), "
            "yfinance, cached; **10 hardcoded operational meltdowns** 2016→2024 (9 with "
            "price coverage — SAVE/Spirit delisted, dropped honestly). Market-model event "
            "study (α/β on SPY, pre-event estimation window). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_spy"] +
            "` SPY / `" + R["fp_ba"] + "` BA / `" + R["fp_dal"] + "` DAL / …).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | 9-event month CAR **{R['head']['month'][0]:+.0f} bps**, "
            f"one-sample **t = {R['head']['month'][1]:+.2f}**, placebo **p = "
            f"{R['placebo']['month'][3]:.3f}** — but airlines-only **t = "
            f"{R['split']['air'][3]:+.2f}** (n.s.) |\n"
            f"| **Tradability** | `FRAGILE` | short net positive every horizon "
            f"(best t = {R['timer'][10][2]:+.2f}, n=9) but airlines-only ≈0 |\n"
            f"| **Reputational vs fundamental?** | `MIXED` | Boeing-only month "
            f"**{R['split']['boe'][2]:+.0f} bps**; airlines-only "
            f"**{R['split']['air'][2]:+.0f} bps** |\n\n"
            "> 💡 In plain words: the aggregate clears the bar, but the aggregate is two "
            "Boeing groundings wearing a reputation-shock costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "For meltdown $i$ on implicated ticker $c_i$, let $a_{i,t} = r_{c_i,t} - "
            "(\\hat\\alpha_i + \\hat\\beta_i r_{m,t})$ be the market-model abnormal "
            "return, with $(\\hat\\alpha_i, \\hat\\beta_i)$ estimated by OLS on SPY over "
            "the 120 sessions ending 10 days **before** the event (uncontaminated). "
            "Define $\\mathrm{CAR}_i(w) = \\sum_{k\\in w} a_{i,\\tau_i+k}$. The claims:\n\n"
            "- **H₁ (event-window hit).** $E[\\mathrm{CAR}_i(\\{0\\})] < 0$ — the stock "
            "drops on the meltdown session.\n"
            "- **H₂ (reputation sticks).** $E[\\mathrm{CAR}_i([1,21])] < 0$ — a continued "
            "negative drift over the following month, not a same-day shrug.\n"
            "- **H₃ (it's reputation, not fundamentals).** The effect survives dropping "
            "the two Boeing product-line groundings.\n"
            "- **H₄ (capture).** Shorting $c_i$ at the meltdown close pays net of "
            "costs+borrow.\n\n"
            "We find **H₁ borderline** (t = "
            f"{R['head']['day0'][1]:+.2f}, placebo p = {R['placebo']['day0'][3]:.3f}), "
            f"**H₂ borderline** (t = {R['head']['drift'][1]:+.2f}), **H₃ rejected** "
            f"(airlines-only month t = {R['split']['air'][3]:+.2f}), **H₄ not supported** "
            "(positive but n.s., Boeing-concentrated)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Meltdowns are **independent, non-overlapping calendar dates on (mostly) "
            "different tickers**, so the planned primary is a **one-sample t-test** across "
            "the per-event CARs at each horizon — no HAC needed (the unit is already one "
            "number per event; the Newey-West *t* is shown only as a cross-check and is "
            "unstable at n=9). The down-hit rate carries a **Wilson interval**. The "
            "falsification control is a **same-ticker random-date placebo**: each event "
            "keeps its own ticker (its β, its idiosyncratic vol) but draws a random "
            "pseudo-date, 5,000 times — so we ask whether *this specific basket of names* "
            "produces the observed CAR on a random calendar. And because n=9, "
            "**robustness is not optional**: airlines-vs-Boeing, leave-one-out, and "
            "sub-era splits decide the stamp, not the headline *t*."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_events_all']} meltdowns {R['cal_lo']} → {R['cal_hi']}, "
            "hardcoded with public-record source notes; 9 have fetchable-ticker coverage "
            "(SAVE/Spirit delisted).\n"
            "- **Tape.** SPY + LUV/DAL/UAL/AAL/BA total-return closes, 2014 → 2026-06-30 "
            "(as-of, last complete month).\n"
            "- **Model.** Market model, 120-session estimation window ending 10 sessions "
            "pre-event; CAR at day 0, week [0..4], month [0..21], drift [1..21].\n"
            "- **Headline.** One-sample t per horizon + Wilson down-rate + 5,000-draw "
            "same-ticker random-date placebo.\n"
            "- **Robustness.** Airlines-only vs Boeing-only; leave-one-out on the month t; "
            "pre-2022 vs 2022+ sub-eras.\n"
            "- **Execution (timer).** Short at the snap-session close (zero look-ahead — "
            "the meltdown predates that close), cover `hold` sessions later; 2 × one-way "
            "cost × NAV + 300 bps/yr borrow.\n"
            "- **Control.** Synthetic factor panel with a planted meltdown drop; the null "
            "must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline CARs and the placebo\n\n"
            "One-sample t on each horizon's cross-event CAR, plus the same-ticker "
            "random-date null. Here we run a light 1,500-draw placebo live and quote the "
            "canonical 5,000-draw p from `results.md`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [(h, st.car_stats(CARS, h)['mean_bps']) for h in ['day0','week','month','drift']]\n"
            "    pb = st.permutation_placebo(EVENTS, SPY, STOCKS, horizon='month', n_draws=1500, seed=850)\n"
            "    draws, obs = pb['draws_bps'], pb['obs_bps']\n"
            "else:\n"
            "    rows = [(h, R['head'][h][0], R['head'][h][1]) for h in ['day0','week','month','drift']]\n"
            "    rng = np.random.default_rng(850)\n"
            "    draws = rng.normal(R['placebo']['month'][1], R['placebo']['month'][2], 1500)\n"
            "    obs = R['placebo']['month'][0]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={'width_ratios':[1.1,1]})\n"
            "labs = [r[0] for r in rows]; ms = [r[1] for r in rows]\n"
            "a1.bar(labs, ms, color=[RED if abs(R['head'][l][1])>=2 else AMBER for l in labs], width=.6)\n"
            "for i, r in enumerate(rows):\n"
            "    a1.annotate(f'{r[1]:+.0f}\\n(t={R[\"head\"][r[0]][1]:+.2f})', (i, r[1]), ha='center', va='top', fontsize=8)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean CAR (bps)')\n"
            "a1.set_title('CAR by horizon (amber = |t|<2, red = |t|>=2)')\n"
            "a2.hist(draws, bins=40, color=GREY, alpha=.85, label='random-date null (month, light run)')\n"
            "a2.axvline(obs, c=RED, lw=2.5, label=f'observed {obs:+.0f} bps')\n"
            "a2.set_xlabel('mean month CAR of a random 9-name calendar (bps)')\n"
            "a2.set_title(f\"canonical p = {R['placebo']['month'][3]:.3f} (5,000 draws)\")\n"
            "a2.legend(); plt.tight_layout(); plt.show()\n"
            "print('canonical placebo month: obs', R['placebo']['month'][0], 'bps, p =', R['placebo']['month'][3])"
        ),
        md(
            f"> 💡 In plain words: the month CAR is **{R['head']['month'][0]:+.0f} bps** at "
            f"one-sample **t = {R['head']['month'][1]:+.2f}** — it crosses the desk's |t| ≥ 2 "
            f"bar — and the random-date placebo agrees (**p = {R['placebo']['month'][3]:.3f}**; "
            f"day-0 p = {R['placebo']['day0'][3]:.3f}). Taken alone, this reads as a hit. "
            "The next cell is why it isn't a *clean* one."
        ),
        md(
            "### 4b · The decomposition — H₃, the whole ballgame\n\n"
            "Two of the nine events are Boeing MAX groundings — a **fundamental** "
            "product-line shock (deliveries halt, earnings crater), categorically "
            "different from a reputational blip. Split them out, then leave-one-out the "
            "month t."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s_all = st.car_stats(CARS, 'month'); s_air = st.car_stats(CARS[CARS['ticker']!='BA'], 'month')\n"
            "    s_boe = st.car_stats(CARS[CARS['ticker']=='BA'], 'month')\n"
            "    x = CARS['month'].to_numpy(); dts = list(CARS.index.strftime('%Y %b')); tks = list(CARS['ticker'])\n"
            "    loo = [(f'{dts[i]} {tks[i]}', st.one_sample_t(np.delete(x, i))[1]) for i in range(len(x))]\n"
            "    all_bps, air_bps, boe_bps = s_all['mean_bps'], s_air['mean_bps'], s_boe['mean_bps']\n"
            "    all_t, air_t = s_all['t'], s_air['t']\n"
            "else:\n"
            "    all_bps, all_t = R['split']['all'][2], R['split']['all'][3]\n"
            "    air_bps, air_t = R['split']['air'][2], R['split']['air'][3]\n"
            "    boe_bps = R['split']['boe'][2]\n"
            "    loo = [('2019 Mar BA', R['loo_drop_ba2019']), ('2024 Jan BA', R['loo_drop_ba2024']),\n"
            "           ('other events', -2.05)]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.5), gridspec_kw={'width_ratios':[1,1.3]})\n"
            "a1.bar(['all 9', 'airlines\\nonly', 'Boeing\\nonly'], [all_bps, air_bps, boe_bps],\n"
            "       color=[GREY, GREEN, RED], width=.6)\n"
            "for i, v in enumerate([all_bps, air_bps, boe_bps]):\n"
            "    a1.annotate(f'{v:+.0f}', (i, v), ha='center', va='top')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('month CAR (bps)')\n"
            "a1.set_title(f'airlines-only t={air_t:+.2f} (n.s.)  vs  all t={all_t:+.2f}')\n"
            "names = [l[0] for l in loo]; tvals = [l[1] for l in loo]\n"
            "cols = [RED if 'BA' in n else GREY for n in names]\n"
            "a2.barh(names, tvals, color=cols)\n"
            "a2.axvline(-2, ls='--', c=RED, lw=1); a2.axvline(0, c='k', lw=.8)\n"
            "a2.set_xlabel('full-sample month t after dropping this event')\n"
            "a2.set_title('Drop either Boeing event -> t falls below -2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'all {all_bps:+.0f} (t={all_t:+.2f}) | airlines {air_bps:+.0f} (t={air_t:+.2f}) | Boeing {boe_bps:+.0f}')"
        ),
        md(
            f"> 💡 In plain words: airlines-only the month CAR is **{R['split']['air'][2]:+.0f} "
            f"bps at t = {R['split']['air'][3]:+.2f}** — gone. Boeing-only it's "
            f"**{R['split']['boe'][2]:+.0f} bps**. Leave-one-out is decisive: dropping "
            f"BA-2019 gives t = {R['loo_drop_ba2019']:+.2f}, dropping BA-2024 gives t = "
            f"{R['loo_drop_ba2024']:+.2f} — **either** Boeing event alone pushes the "
            "aggregate below the bar, while dropping any airline event mostly leaves it "
            f"there. And it fails across eras (pre-2022 month t = {R['pre2022_month_t']:+.2f}). "
            "H₃ is rejected: the signal is a *fundamental* grounding effect, not the "
            "claimed reputational one."
        ),
        md(
            "### 4c · The timer — short the meltdown, cost + borrow\n\n"
            "Short the implicated stock at the snap-session close (zero look-ahead), hold "
            "`h` sessions, cover; net = 2 × one-way 5 bps + 300 bps/yr borrow on the "
            "short leg. Full sample vs airlines-only."
        ),
        code(
            "holds = [5, 10, 21]\n"
            "if HAVE_REAL:\n"
            "    net = [st.summarize_short(st.short_the_meltdown(EVENTS, STOCKS, hold=h, cost_bps=5.0, borrow_bps_yr=300.0), 'short_net')['mean_bps'] for h in holds]\n"
            "    air = [st.summarize_short(st.short_the_meltdown(EVENTS[EVENTS['ticker']!='BA'], STOCKS, hold=h, cost_bps=5.0, borrow_bps_yr=300.0), 'short_net')['mean_bps'] for h in holds]\n"
            "    ts = [st.summarize_short(st.short_the_meltdown(EVENTS, STOCKS, hold=h, cost_bps=5.0, borrow_bps_yr=300.0), 'short_net')['t'] for h in holds]\n"
            "else:\n"
            "    net = [R['timer'][h][1] for h in holds]; air = [R['timer'][h][4] for h in holds]; ts = [R['timer'][h][2] for h in holds]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "x = np.arange(len(holds)); w = 0.38\n"
            "ax.bar(x - w/2, net, width=w, color=AMBER, label='all 9 (net short)')\n"
            "ax.bar(x + w/2, air, width=w, color=GREEN, label='airlines only (net short)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in holds])\n"
            "ax.set_ylabel('mean short return, net (bps)')\n"
            "ax.set_title('The short is positive only because of Boeing; airlines-only ~0')\n"
            "for i, (v, t) in enumerate(zip(net, ts)):\n"
            "    ax.annotate(f't={t:+.2f}', (i - w/2, v), ha='center', va='bottom', fontsize=8)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('net all (bps):', [round(v) for v in net], '| airlines-only:', [round(v) for v in air])"
        ),
        md(
            f"> 💡 In plain words: the full short nets **+{R['timer'][10][1]:.0f} bps** at "
            f"10 days (t = {R['timer'][10][2]:+.2f}) — positive, never significant, and "
            f"airlines-only it collapses to **{R['timer'][10][4]:+.0f} / "
            f"{R['timer'][5][4]:+.0f} / {R['timer'][21][4]:+.0f} bps** (|t| ≤ 0.4). H₄ "
            "not supported: a nine-event edge on two groundings, needing a hard-to-borrow "
            "short into a crowded trade. **Fragile.**"
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic market + factor-stock panel, scheduled pseudo-meltdowns, TUNABLE "
            "planted day-0 drop (plus a small bleed). The null (edge=0) is checked over "
            "**20 seeds** — never a single stream."
        ),
        code(
            "null_ts = np.array([st.synthetic_detect(*data.synthetic_world(edge=0.0, seed=850+s), horizon='day0')['t'] for s in range(20)])\n"
            "m, sk, ee = data.synthetic_world(edge=0.03, seed=850)\n"
            "planted_t = st.synthetic_detect(m, sk, ee, horizon='day0')['t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12,.12,20), null_ts, color=GREY, s=40, label='null worlds (edge=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5, label='planted drop (edge=0.03)')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0,1]); ax.set_xticklabels(['null x 20','planted'])\n"
            "ax.set_ylabel('one-sample t (day-0 CAR)')\n"
            "ax.set_title('Control: no null fires; a planted drop lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t={null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(np.abs(null_ts)>=2).sum()}/20 | planted t={planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages t = "
            f"{R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}), crossing the bar in "
            f"{R['syn_null_fire']}/20 (≈ the 5% you'd expect at n=9); a planted drop reads "
            f"t = {R['syn_planted_t']:+.2f}. The estimator is unbiased — the real-tape "
            "readings are genuine, and the *interpretation* (Boeing, not reputation) is "
            "what does the work. *(A faithful-engine / power check only — never cited in "
            "support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — 9-event month CAR **{R['head']['month'][0]:+.0f} bps**, "
            f"one-sample t = **{R['head']['month'][1]:+.2f}**, placebo p = "
            f"**{R['placebo']['month'][3]:.3f}**, sign right — but not robust: airlines-only "
            f"month t = **{R['split']['air'][3]:+.2f}** (n.s.), leave-one-out on either "
            "Boeing grounding drops it below −2, and it fails pre-2022 "
            f"(t = {R['pre2022_month_t']:+.2f}). The apparent edge is a *fundamental* "
            "product-line grounding effect, not the claimed reputational one.\n"
            f"- **Tradability `FRAGILE`** — short net positive at every horizon "
            f"(best t = {R['timer'][10][2]:+.2f}, n=9) but Boeing-concentrated; "
            "airlines-only ≈ 0. No edge to bank.\n"
            "- **\"Reputational shock vs quick fade?\" `MIXED`** — a grounded product "
            "line dents the stock durably (Boeing MAX); a cancellation collapse / IT "
            "outage / PR firestorm at a still-flying carrier mostly fades. At n=9, the "
            "reputational-shock story is not detectable on the tradable tape."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Power is the honest limit.** Nine coverable events is tiny; a small, "
            "genuine reputational effect could hide in it. A pre-registered extension to a "
            "much larger register of operational disruptions (with the same "
            "fundamental-vs-reputational discipline) is the natural next step.\n"
            "- **Better reputation proxies** — forward bookings, app-store ratings, "
            "net-promoter surveys, or CDS/credit rather than equity — might catch soft "
            "damage a monthly equity CAR cannot.\n"
            "- **Dedup map:** [707-plane-crash-effect](../../707-plane-crash-effect/) "
            "(fatal crashes → market mood; excludes the MAX grounding from its market "
            "test), [554-airline-bookings](../../554-airline-bookings/) (alt-data demand) "
            "and [313-geopolitical-shock](../../313-geopolitical-shock/) (market-wide "
            "shocks) — same neighbourhood, different axis.\n\n"
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
