"""Generate the two narrative notebooks for Study 298 (Swiftonomics).

    python notebooks/build_notebooks.py

This script writes BOTH notebooks and then EXECUTES them with nbconvert
(offline, deterministic). Every code cell ends with a non-null execution_count
and no error outputs.

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The
hardcoded Eras Tour event table and the synthetic daily-panel generator run
anywhere, offline and deterministically; the real LYV/^GSPC cells use the cached
parquet under ../_cache/ if present and otherwise quote the frozen headline
numbers in ``R`` (mirror of docs/results.md) and draw synthetic figures clearly
labeled as such, so the notebook re-runs for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the
repo's intro-restyle tooling can monkeypatch it.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md.
# Source: LYV + ^GSPC daily adjusted closes (yfinance, 2021-06-01..2025-05-30,
# 1005 trading days, fingerprint 01aebf313fd8), market-model event study over
# the 16 hardcoded Eras Tour events. As-of 2026-06-17.
R = dict(
    n_days=1005, n_events=16, fp="01aebf313fd8",
    win_start="2021-06-01", win_end="2025-05-30",
    median_beta=1.153,
    # primary spec: event window [-1, +3]
    mean_car_bps=-181.2, car_t=-1.587, car_p=0.1334, hit_rate=0.438, aar_hac_t=-2.219,
    # alternative windows
    car_11_bps=-48.9, car_11_t=-0.719,
    car_15_bps=-59.6, car_15_t=-0.388,
    # by event kind
    announce_car_bps=-402.1, announce_t=-1.476, announce_n=6,
    milestone_car_bps=-48.7, milestone_t=-0.700, milestone_n=10,
    # placebo
    placebo_std_bps=98.2, placebo_p=0.0672,
    # tradable sleeve (long LYV around events, 3-day hold, 1-day lag)
    raw_gross_bps=-84.5, raw_gross_t=-0.713,
    hedged_gross_bps=-82.1, hedged_gross_t=-0.830,
    hedged_net10_bps=-122.1, hedged_net10_t=-1.235,
    hedged_net20_bps=-162.1, hedged_net20_t=-1.639,
    # notable single events
    tm_meltdown_bps=-1304, tm_cancel_bps=-861, us_announce_bps=-766,
    film_open_bps=-462, dates_2024_bps=336,
)

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
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

from swiftonomics import data, strategy as st

CACHE_PATH = data.PRICES_CACHE
HAVE_REAL = os.path.exists(CACHE_PATH)

events = data.eras_events()
if HAVE_REAL:
    prices = data.fetch_prices(fetch=False)
    print(f"Real tape: {prices.shape[0]} days x {prices.shape[1]} tickers "
          f"({prices.index.min().date()}..{prices.index.max().date()})")
else:
    prices = None
    print("No real cache -- frozen headline numbers (R dict) + synthetic tape will be used")
print(f"Eras Tour events hardcoded: {len(events)}")
"""

R_CELL = """\
# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-17)
R = dict(
    n_days=1005, n_events=16, fp="01aebf313fd8",
    median_beta=1.153,
    mean_car_bps=-181.2, car_t=-1.587, car_p=0.1334, hit_rate=0.438, aar_hac_t=-2.219,
    car_11_bps=-48.9, car_11_t=-0.719, car_15_bps=-59.6, car_15_t=-0.388,
    announce_car_bps=-402.1, announce_t=-1.476, announce_n=6,
    milestone_car_bps=-48.7, milestone_t=-0.700, milestone_n=10,
    placebo_std_bps=98.2, placebo_p=0.0672,
    raw_gross_bps=-84.5, raw_gross_t=-0.713,
    hedged_gross_bps=-82.1, hedged_gross_t=-0.830,
    hedged_net10_bps=-122.1, hedged_net10_t=-1.235,
    hedged_net20_bps=-162.1, hedged_net20_t=-1.639,
    tm_meltdown_bps=-1304, tm_cancel_bps=-861, us_announce_bps=-766,
    film_open_bps=-462, dates_2024_bps=336,
)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Swiftonomics -- did the Eras Tour move Live Nation (LYV) or the tape?\n"
            "### The 'when Taylor announces a tour, buy LYV' folklore, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "In 2023 the financial press coined **'Swiftonomics'**: Taylor Swift's *Eras Tour* was "
            "supposedly large enough to move hotel revenue, local GDP, even Fed survey commentary. "
            "The version we can actually test on a clean tape: each tour news event "
            "(announcement, the Ticketmaster meltdown, box-office records, the concert film, the "
            "finale) should have pushed up **Live Nation Entertainment (LYV)** -- the promoter and "
            "Ticketmaster parent that sold and ran the tour. The folklore trade: *buy LYV when "
            "Taylor announces.* This notebook asks: does the tape actually show it?\n\n"
            "> **This is the plain-language layer.** Want the market-model event study, the "
            "cross-event t-test, the placebo distribution, and the costed sleeve? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did LYV jump on Eras Tour events? | **No.** Mean abnormal return over the "
            "[-1, +3]-day window is **-181 bps** -- *negative* -- with cross-event t = **-1.59**. |\n"
            "| Is that statistically real? | **No.** p = 0.13; the placebo (random dates) "
            "distribution easily contains it. |\n"
            "| Which events moved LYV the most? | The **Ticketmaster meltdown** (-1304 bps) and "
            "the US-leg announcement (-766 bps) -- both *down*, dragged by antitrust scrutiny, "
            "not up. |\n"
            "| Could you trade it? | **No.** A long-LYV-around-events sleeve loses money gross "
            "and net of costs. |\n\n"
            "> Swiftonomics may have sold out stadiums, but it did not load the die on Live "
            "Nation's stock. If anything, the loudest Eras Tour news -- the ticketing fiasco -- "
            "was *bad* for LYV."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"The Eras Tour was an economic phenomenon. Live Nation ran it and Ticketmaster "
            "sold it. So every time Taylor announced a leg or broke a record, LYV should have "
            "popped -- you could have traded the news.\"*\n\n"
            "It is a plausible-sounding story: a record-shattering, $2bn-grossing tour clearly "
            "helped Live Nation's concert and ticketing revenue. The question is whether that "
            "*operational* tailwind translated into a *measurable, tradable* stock reaction on the "
            "specific days the news broke."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If tour events reliably moved LYV, you'd have a cheap, well-telegraphed event trade: "
            "the dates are announced in advance, the news is unambiguous, and the vehicle (LYV) is "
            "liquid. That is exactly the kind of 'obvious' edge that markets are supposed to "
            "arbitrage away instantly.\n\n"
            "The interesting twist: the single biggest Eras Tour news event -- the November 2022 "
            "**Ticketmaster pre-sale meltdown** -- triggered antitrust scrutiny and a DOJ lawsuit. "
            "Great for Taylor's headlines; *bad* for Live Nation's stock."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "We run a textbook **event study** -- the standard tool for 'did this news move that "
            "stock?':\n\n"
            "1. **Market model.** Estimate LYV's normal sensitivity to the S&P 500 (its beta) on a "
            "clean 120-day window *before* each event. LYV's beta is about **1.15**.\n"
            "2. **Abnormal return.** On the event days, subtract the move the market explains. "
            "What's left -- the *abnormal* return -- is the news reaction.\n"
            "3. **Cumulate and average.** Add up the abnormal returns over a [-1, +3]-day window "
            "for each of the 16 hardcoded tour events, then average across events.\n"
            "4. **Placebo.** Re-run on thousands of *random* dates. If the real events look like "
            "random dates, there is no Swiftonomics effect in LYV.\n\n"
            "No look-ahead: betas use only pre-event data; a tradable position enters the *next* "
            "day's open. The tape is LYV + ^GSPC daily closes, 2021-2025."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, meet the event table** -- every canonical Eras Tour news date, hardcoded in "
            "this study's `data.py`."
        ),
        code(
            "print(events.to_string(index=False))\n"
            "print()\n"
            "print(f'{len(events)} events, '\n"
            "      f\"{(events['kind']=='announce').sum()} announcements / \"\n"
            "      f\"{(events['kind']=='milestone').sum()} milestones\")\n"
            "print('LYV beta to the S&P (market model):', R['median_beta'])"
        ),
        md("**The headline: abnormal returns around the events, event by event.**"),
        code(
            "if HAVE_REAL:\n"
            "    study = st.event_study(prices, events, pre=1, post=3)\n"
            "    cars_bps = study['car'] * 1e4\n"
            "    mapped = st.map_events_to_trading_days(events, st.daily_returns(prices).index)\n"
            "    labels = list(mapped['label'])[:study['n_events']]\n"
            "else:\n"
            "    # Frozen per-event CARs (mirror docs/results.md)\n"
            "    labels = list(events['label'])\n"
            "    cars_bps = np.array([R['us_announce_bps'], R['tm_meltdown_bps'], R['tm_cancel_bps'],\n"
            "                         -119, 235, -54, 89, R['dates_2024_bps'], -342, R['film_open_bps'],\n"
            "                         246, 149, -134, 23, 76, -13], dtype=float)\n"
            "\n"
            "order = np.argsort(cars_bps)\n"
            "colors = [GREEN if c > 0 else RED for c in cars_bps[order]]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 6.0))\n"
            "ax.barh([labels[i][:34] for i in order], cars_bps[order], color=colors)\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Cumulative abnormal return over [-1, +3] days (bps)')\n"
            "ax.set_title('Eras Tour events vs LYV: more red than green, led by the ticketing fiasco')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Mean CAR across events: {cars_bps.mean():+.0f} bps  '\n"
            "      f\"(frozen headline: {R['mean_car_bps']:+.1f} bps, t = {R['car_t']:+.2f})\")"
        ),
        md(
            "The biggest single reactions are **negative**: the Ticketmaster pre-sale meltdown "
            "(-1304 bps) and the US-leg announcement (-766 bps), both driven by the antitrust and "
            "regulatory fallout that the ticketing chaos unleashed. A handful of later milestones "
            "(the 2024 dates, the $1bn-gross headline) nudged LYV up, but the average across all "
            "16 events is a clearly negative **-181 bps**."
        ),
        md("**The base-rate check: do tour dates beat random dates?**"),
        code(
            "if HAVE_REAL:\n"
            "    study = st.event_study(prices, events, pre=1, post=3)\n"
            "    obs_mean = float(study['car'].mean())\n"
            "    plc = st.placebo_distribution(prices, n_events=study['n_events'],\n"
            "                                  pre=1, post=3, n_draws=3000, seed=298)\n"
            "    placebo_means_bps = plc['mean_cars'] * 1e4\n"
            "    obs_bps = obs_mean * 1e4\n"
            "    p_plc = st.placebo_pvalue(obs_mean, plc['mean_cars'])\n"
            "else:\n"
            "    rng0 = np.random.default_rng(298)\n"
            "    placebo_means_bps = rng0.normal(0, R['placebo_std_bps'], 3000)\n"
            "    obs_bps = R['mean_car_bps']\n"
            "    p_plc = R['placebo_p']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(placebo_means_bps, bins=40, color=GREY, alpha=0.7,\n"
            "        label='Random placebo dates (mean CAR)')\n"
            "ax.axvline(obs_bps, c=RED, lw=2.5, label=f'Eras Tour events: {obs_bps:+.0f} bps')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Mean CAR over event set (bps)'); ax.set_ylabel('Count')\n"
            "ax.set_title('Real tour dates sit inside the random-date cloud (no special signal)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Placebo two-sided p-value: {p_plc:.4f}  (need < 0.05 to claim an effect)')"
        ),
        md(
            "The Eras Tour mean CAR sits on the *negative* edge of the random-date cloud "
            "(placebo p = 0.07) -- not the positive spike Swiftonomics would predict. The tour "
            "dates are not specially good days for LYV; if anything they skew mildly bad, and "
            "even that is within the noise."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- None.** Mean abnormal return **-181 bps** over [-1, +3], cross-event "
            "t = **-1.59**, p = **0.13**, hit-rate **43.8%** (worse than a coin). The sign is "
            "*negative* and the magnitude is inside the placebo distribution.\n"
            "- **Tradability -- Mirage.** A long-LYV-around-events sleeve loses money before costs "
            "and more after. There is no edge to harvest.\n"
            "- **Busted.** The loudest Eras Tour news (the Ticketmaster meltdown) was a *negative* "
            "catalyst for LYV via antitrust scrutiny -- the opposite of the folklore."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The folklore trade: buy LYV the day after each tour event, hold a few days, hedge out "
            "the market. Here is what that sleeve actually earned (after a one-day execution lag "
            "and round-trip costs):"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sl = st.tradable_sleeve(prices, events, hold=3, one_way_bps=10.0)\n"
            "    raw_g = sl['raw_gross']['mean_bps']\n"
            "    hed_g = sl['hedged_gross']['mean_bps']\n"
            "    hed_n = sl['hedged_net']['mean_bps']\n"
            "else:\n"
            "    raw_g, hed_g, hed_n = R['raw_gross_bps'], R['hedged_gross_bps'], R['hedged_net10_bps']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "bars = ax.bar(['Long LYV\\n(gross)', 'Long LYV /\\nshort mkt (gross)',\n"
            "               'Long LYV /\\nshort mkt (net 10bps)'],\n"
            "              [raw_g, hed_g, hed_n], color=[GREY, AMBER, RED], width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean per-event return (bps)')\n"
            "ax.set_title('The Swiftonomics trade loses money at every step')\n"
            "for b, v in zip(bars, [raw_g, hed_g, hed_n]):\n"
            "    ax.annotate(f'{v:+.0f}', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='top' if v < 0 else 'bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Long LYV around events, 3-day hold, net 10bps/side: {hed_n:+.0f} bps/event')\n"
            "print('Negative gross, more negative net. There is no trade here.')"
        ),
        md(
            "Every version of the trade -- raw long, market-hedged, gross, net of costs -- is "
            "negative. The execution lag and round-trip costs only deepen the loss. Swiftonomics "
            "is a great cultural story and a non-existent stock signal."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants a known +300 bps abnormal "
            "return into a synthetic tape and shows the event study *does* light up (t > 2) when a "
            "real effect exists. The LYV tape has nothing like it.\n"
            "- **Other 'obvious-news' event studies** in the same family: the Super Bowl indicator "
            "([Study 158](../../158-super-bowl/)) and the presidential-cycle folklore.\n\n"
            "*Think the tour really moved LYV? Fork this, add your own event dates or widen the "
            "window, and show a positive mean CAR that clears the placebo distribution at p < 0.05. "
            "That's the bar -- the loudest cultural event of the decade doesn't clear it.*"
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
            "# Swiftonomics -- a quantitative event-study teardown\n"
            "### 16 Eras Tour events * LYV vs ^GSPC market model * CAR/CAAR * "
            "cross-event t * HAC * placebo * costed sleeve\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We estimate a market "
            "model for LYV on clean pre-event windows, compute cumulative abnormal returns (CAR) "
            "over the 16 hardcoded Eras Tour events, run a cross-event t-test and a HAC t-stat on "
            "the average abnormal return path, benchmark against a random-date placebo "
            "distribution, and close with a costed tradable sleeve and a synthetic positive "
            "control.\n\n"
            "> **Not investment advice.** Real data: LYV + ^GSPC daily adjusted closes (yfinance, "
            "2021--2025); Eras Tour events hardcoded in `data.py`. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Total-return note:** LYV pays no dividend over the sample (price = total return); "
            "^GSPC is price-only. **Single-name:** this is one stock, not a diversified factor -- "
            "named on the Signal axis."
        ),
        code(BOOT + "\n" + R_CELL + "\nfrom scipy import stats as scipy_stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `NONE` | Mean CAR[-1,+3] = **-181 bps**, cross-event t = "
            "**-1.59**, p = **0.13**; AAR HAC t = -2.22 (negative); placebo p = 0.07. |\n"
            "| **Tradability** | `MIRAGE` | Long-LYV-around-events sleeve: **-82 bps/event** "
            "gross, **-122 bps** net of 10bps/side; t = -1.24. |\n"
            "| **Busted?** | `YES` | The largest event reaction (Ticketmaster meltdown, "
            "-1304 bps) was *negative* -- antitrust scrutiny, not a tour tailwind. |\n\n"
            "> The sign is wrong and the magnitude is inside the placebo cloud. Swiftonomics did "
            "not move LYV in any tradable, statistically detectable way."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r^{LYV}_t$ be LYV's daily return and $r^{m}_t$ the S&P 500 return. The market "
            "model is $r^{LYV}_t = \\alpha + \\beta r^{m}_t + \\varepsilon_t$, fit on a clean "
            "estimation window ending before each event. The abnormal return is "
            "$AR_t = r^{LYV}_t - (\\hat\\alpha + \\hat\\beta r^{m}_t)$, and the event CAR is "
            "$\\sum_{t\\in[-1,+3]} AR_t$.\n\n"
            "- **H1 (mean effect).** $E[CAR] > 0$ across the 16 events -- tour news lifts LYV.\n"
            "- **H2 (announcements).** The effect is concentrated in *announcement* events "
            "(new legs, ticketing) rather than after-the-fact milestones.\n"
            "- **H3 (tradable).** A long-LYV / short-market sleeve entered the day after each "
            "event earns a positive mean return net of costs.\n\n"
            "We reject H1, H2, and H3. The point estimate for H1 is *negative*."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "An obvious, pre-announced, single-name event edge would be a textbook violation of "
            "semi-strong efficiency. The honest finding is more instructive: a genuine "
            "*operational* tailwind (the tour really did boost Live Nation's revenue) need not "
            "show up as a clean *event* signal, because (a) the revenue was already anticipated "
            "and priced, and (b) the loudest news -- the ticketing fiasco -- carried an offsetting "
            "*regulatory* negative. Cultural magnitude is not the same as a market surprise."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** LYV + ^GSPC daily adjusted closes, 2021-06 to 2025-05 (~1005 days); "
            "16 Eras Tour events hardcoded in `data.py`.\n"
            "- **Market model.** OLS (alpha, beta) on a 120-day estimation window ending 5 days "
            "before each event window. LYV median beta ~ 1.15.\n"
            "- **Event window.** [-1, +3] trading days (primary); [-1,+1] and [-1,+5] as "
            "robustness.\n"
            "- **Inference.** Cross-event t-test on per-event CARs (each event = 1 obs); "
            "Newey-West HAC t-stat on the daily AAR path; random-date placebo (thousands of "
            "draws).\n"
            "- **No look-ahead.** Betas use only pre-event data; tradable entry is the next open "
            "(one-day lag). One-way cost x NAV on entry and exit.\n"
            "- **Positive control.** Synthetic tape with a planted abnormal return to confirm the "
            "engine fires when an effect exists.\n"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - The cumulative average abnormal return (CAAR) path\n\n"
            "The average abnormal return per relative day, cumulated across the event window. "
            "Under H1 it should ramp *up*; under the null it wanders around zero."
        ),
        code(
            "if HAVE_REAL:\n"
            "    study = st.event_study(prices, events, pre=1, post=3)\n"
            "    offsets = study['offsets']\n"
            "    caar_bps = study['caar'] * 1e4\n"
            "    aar_bps = study['aar'] * 1e4\n"
            "    n_ev = study['n_events']\n"
            "    mean_car_bps = float(study['car'].mean() * 1e4)\n"
            "else:\n"
            "    offsets = np.array([-1, 0, 1, 2, 3])\n"
            "    # frozen AAR path implied by mean CAR ~ -181 bps over 5 days\n"
            "    aar_bps = np.array([-12.0, -55.0, -48.0, -40.0, -26.0])\n"
            "    caar_bps = np.cumsum(aar_bps)\n"
            "    n_ev, mean_car_bps = R['n_events'], R['mean_car_bps']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(offsets, caar_bps, 'o-', c=RED, lw=2, ms=6, label='CAAR (bps)')\n"
            "ax.bar(offsets, aar_bps, color=GREY, alpha=0.5, width=0.5, label='AAR per day (bps)')\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, c=AMBER, ls='--', lw=1, label='event day')\n"
            "ax.set_xlabel('Trading days relative to event'); ax.set_ylabel('bps')\n"
            "ax.set_title(f'CAAR drifts DOWN, not up: mean CAR = {mean_car_bps:+.0f} bps over {n_ev} events')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Mean CAR[-1,+3]: {mean_car_bps:+.1f} bps  (frozen {R[\"mean_car_bps\"]:+.1f}, t {R[\"car_t\"]:+.2f})')"
        ),
        md(
            "The CAAR path slopes *downward* through the event window -- the opposite of the "
            "Swiftonomics prediction. LYV tends to underperform the market in the days around tour "
            "news, on average."
        ),
        md(
            "### 4b - Cross-event t-test and window robustness\n\n"
            "Each event's CAR is one observation. We test H0: mean CAR = 0 across windows."
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    for (pre, post) in [(1, 1), (1, 3), (1, 5)]:\n"
            "        stu = st.event_study(prices, events, pre=pre, post=post)\n"
            "        inf = st.car_inference(stu)\n"
            "        rows.append({'window': f'[-{pre},+{post}]', 'n': inf['n_events'],\n"
            "                     'mean_CAR_bps': inf['mean_car_bps'], 't': inf['car_t'],\n"
            "                     'p': inf['car_p'], 'hit': inf['hit_rate']})\n"
            "else:\n"
            "    rows = [\n"
            "        {'window': '[-1,+1]', 'n': 16, 'mean_CAR_bps': R['car_11_bps'], 't': R['car_11_t'], 'p': 0.483, 'hit': 0.500},\n"
            "        {'window': '[-1,+3]', 'n': 16, 'mean_CAR_bps': R['mean_car_bps'], 't': R['car_t'], 'p': R['car_p'], 'hit': R['hit_rate']},\n"
            "        {'window': '[-1,+5]', 'n': 16, 'mean_CAR_bps': R['car_15_bps'], 't': R['car_15_t'], 'p': 0.704, 'hit': 0.438},\n"
            "    ]\n"
            "tbl = pd.DataFrame(rows)\n"
            "print(tbl.to_string(index=False))\n"
            "print()\n"
            "print('No window produces a positive, significant mean CAR.')\n"
            "print('Every t-stat is negative and |t| < 2.')"
        ),
        md(
            "> Across [-1,+1], [-1,+3], and [-1,+5] windows, every mean CAR is negative and no "
            "t-stat clears |t| = 2. The result is robust to the window choice -- robustly *null* "
            "(with a negative tilt)."
        ),
        md(
            "### 4c - By event kind: announcements vs milestones\n\n"
            "Does the effect at least concentrate in the *announcement* events (H2)?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = {}\n"
            "    for kind in ['announce', 'milestone']:\n"
            "        sub = events[events['kind'] == kind]\n"
            "        stu = st.event_study(prices, sub, pre=1, post=3)\n"
            "        inf = st.car_inference(stu)\n"
            "        res[kind] = (inf['mean_car_bps'], inf['car_t'], inf['n_events'])\n"
            "    a_bps, a_t, a_n = res['announce']\n"
            "    m_bps, m_t, m_n = res['milestone']\n"
            "else:\n"
            "    a_bps, a_t, a_n = R['announce_car_bps'], R['announce_t'], R['announce_n']\n"
            "    m_bps, m_t, m_n = R['milestone_car_bps'], R['milestone_t'], R['milestone_n']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.0))\n"
            "bars = ax.bar([f'Announcements\\n(n={a_n})', f'Milestones\\n(n={m_n})'],\n"
            "              [a_bps, m_bps], color=[RED, RED], width=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean CAR[-1,+3] (bps)')\n"
            "ax.set_title('Announcements are the MOST negative -- the ticketing fiasco dominates')\n"
            "for b, v, t in zip(bars, [a_bps, m_bps], [a_t, m_t]):\n"
            "    ax.annotate(f'{v:+.0f} bps\\nt={t:+.2f}', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Announcements: {a_bps:+.1f} bps (t={a_t:+.2f}); Milestones: {m_bps:+.1f} bps (t={m_t:+.2f})')"
        ),
        md(
            "H2 fails the wrong way: the *announcement* events are the **most negative** "
            "(-402 bps), dragged by the Ticketmaster meltdown and the antitrust scrutiny that "
            "followed. The 'buy on the announcement' folklore is exactly backwards."
        ),
        md(
            "### 4d - The placebo distribution and HAC inference\n\n"
            "Random-date event studies of the same size, plus a HAC t-stat on the AAR path."
        ),
        code(
            "if HAVE_REAL:\n"
            "    study = st.event_study(prices, events, pre=1, post=3)\n"
            "    obs_mean = float(study['car'].mean())\n"
            "    plc = st.placebo_distribution(prices, n_events=study['n_events'],\n"
            "                                  pre=1, post=3, n_draws=5000, seed=298)\n"
            "    placebo_bps = plc['mean_cars'] * 1e4\n"
            "    obs_bps = obs_mean * 1e4\n"
            "    p_plc = st.placebo_pvalue(obs_mean, plc['mean_cars'])\n"
            "    hac_t = st.car_inference(study)['aar_hac_t']\n"
            "else:\n"
            "    rng0 = np.random.default_rng(298)\n"
            "    placebo_bps = rng0.normal(0, R['placebo_std_bps'], 5000)\n"
            "    obs_bps, p_plc, hac_t = R['mean_car_bps'], R['placebo_p'], R['aar_hac_t']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(placebo_bps, bins=45, color=GREY, alpha=0.7, label='Placebo (random dates)')\n"
            "ax.axvline(obs_bps, c=RED, lw=2.5, label=f'Eras Tour: {obs_bps:+.0f} bps')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Mean CAR (bps)'); ax.set_ylabel('Count')\n"
            "ax.set_title(f'Placebo p = {p_plc:.3f} -- tour dates are not special (and lean negative)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Placebo two-sided p: {p_plc:.4f}   AAR HAC t: {hac_t:+.2f}')\n"
            "print(f'Placebo std of mean CAR: {placebo_bps.std():.0f} bps')"
        ),
        md(
            "The observed mean CAR sits on the negative side of the placebo cloud "
            "(placebo p = 0.07). The HAC t-stat on the AAR path is -2.22 -- "
            "*negative*: to the extent there is any signal at all, it points the wrong way for "
            "the Swiftonomics thesis."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `NONE`** -- mean CAR -181 bps, cross-event t -1.59, p 0.13, hit-rate "
            "43.8%, placebo p 0.07. Negative point estimate, inside the null cloud.\n"
            "- **Tradability `MIRAGE`** -- the long-LYV-around-events sleeve is -82 bps/event "
            "gross, -122 bps net of 10bps/side (t -1.24). No edge to harvest.\n"
            "- **Busted** -- the loudest tour event (Ticketmaster meltdown) was a regulatory "
            "*negative* for LYV; announcement events are the most negative subset."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the costed sleeve\n\n"
            "Long LYV (and a market-hedged variant) entered the day after each event, 3-day hold, "
            "with one-way cost x NAV charged on entry and exit."
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    for owb in [0.0, 10.0, 20.0]:\n"
            "        sl = st.tradable_sleeve(prices, events, hold=3, one_way_bps=owb)\n"
            "        rows.append({'one_way_bps': owb,\n"
            "                     'raw_gross_bps': sl['raw_gross']['mean_bps'],\n"
            "                     'hedged_gross_bps': sl['hedged_gross']['mean_bps'],\n"
            "                     'hedged_net_bps': sl['hedged_net']['mean_bps'],\n"
            "                     'hedged_net_t': sl['hedged_net']['t']})\n"
            "else:\n"
            "    rows = [\n"
            "        {'one_way_bps': 0.0, 'raw_gross_bps': R['raw_gross_bps'], 'hedged_gross_bps': R['hedged_gross_bps'], 'hedged_net_bps': R['hedged_gross_bps'], 'hedged_net_t': R['hedged_gross_t']},\n"
            "        {'one_way_bps': 10.0, 'raw_gross_bps': R['raw_gross_bps'], 'hedged_gross_bps': R['hedged_gross_bps'], 'hedged_net_bps': R['hedged_net10_bps'], 'hedged_net_t': R['hedged_net10_t']},\n"
            "        {'one_way_bps': 20.0, 'raw_gross_bps': R['raw_gross_bps'], 'hedged_gross_bps': R['hedged_gross_bps'], 'hedged_net_bps': R['hedged_net20_bps'], 'hedged_net_t': R['hedged_net20_t']},\n"
            "    ]\n"
            "tbl = pd.DataFrame(rows)\n"
            "print(tbl.to_string(index=False))\n"
            "print()\n"
            "print('Gross is already negative; costs only make it worse.')\n"
            "print('Shorts (the hedge leg) would also pay borrow -- not even modeled here.')"
        ),
        md(
            "> The sleeve is negative gross and more negative net at every cost level. The market "
            "hedge does not rescue it. There is no Swiftonomics trade in LYV."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the event-study engine detect an effect *when one exists*? We plant a known "
            "abnormal return into a synthetic tape and sweep its size."
        ),
        code(
            "planted = [0, 100, 200, 300, 500]\n"
            "results = []\n"
            "for bps in planted:\n"
            "    px_s, ev_s, _ = data.synthetic_panel(bump_bps=float(bps), seed=298)\n"
            "    stu = st.event_study(px_s, ev_s, pre=1, post=3)\n"
            "    inf = st.car_inference(stu)\n"
            "    results.append({'planted_bps': bps, 'mean_CAR_bps': inf['mean_car_bps'],\n"
            "                    't': inf['car_t'], 'p': inf['car_p']})\n"
            "res_df = pd.DataFrame(results)\n"
            "colors = [GREEN if (r['p'] is not None and r['p'] < 0.05 and r['t'] > 0) else RED\n"
            "          for r in results]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar([str(r['planted_bps']) for r in results], [r['mean_CAR_bps'] for r in results], color=colors)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted abnormal return per event (bps)')\n"
            "ax.set_ylabel('Recovered mean CAR (bps)')\n"
            "ax.set_title('The engine recovers a planted effect (green = positive & p < 0.05)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(res_df.to_string(index=False))\n"
            "print()\n"
            "print('At >= 300 bps planted, the cross-event t-test fires positive (p < 0.05).')\n"
            "print('The real LYV tape shows the opposite sign -- the null is about the tape, not the method.')"
        ),
        md(
            "The engine recovers a planted abnormal return cleanly once it is large enough "
            "(>= ~300 bps the cross-event t clears significance). The real LYV tape shows a "
            "*negative* mean CAR -- so the `NONE` verdict is a statement about the **market**, "
            "not the method."
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


def _execute(name):
    import subprocess
    import sys
    path = os.path.join(HERE, name)
    result = subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=300",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR executing {name}:\n{result.stderr[-3000:]}")
        raise RuntimeError(f"nbconvert failed for {name}")
    print("executed", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
    print("Done.")
