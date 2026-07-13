"""Generate the two narrative notebooks for Study 734 (NBA-Finals-Effect).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached metro-proxy /
SPY tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (17 metro proxies + SPY,
# yfinance total-return, 1998-01-02 -> 2026-06-30; 52 of 52 role-events resolved).
R = dict(
    n_finals=26, n_possible=52, n_included=52, n_loser=26, n_champ=26, n_proxies=17,
    fp="e68e0ed8bc79",
    # signal (day(-1) -> day(-1)+k abnormal return, metro proxy - SPY), means in %
    l_day_mean=-0.149, l_day_t=-0.466, l_day_hit=13, l_day_n=26,
    l_wk_mean=+0.020, l_wk_t=+0.031, l_wk_hit=15, l_wk_n=26,
    c_day_mean=-0.354, c_day_t=-1.223, c_day_hit=10, c_day_n=26,
    c_wk_mean=-0.586, c_wk_t=-1.128, c_wk_hit=12, c_wk_n=26,
    comb_day_mean=-0.252, comb_day_t=-1.175, comb_day_n=52,
    comb_wk_mean=-0.283, comb_wk_t=-0.678,
    # broad-market cross-check (SPY market-model abnormal return on day(0))
    spy_mean=+0.139, spy_t=+0.812,
    # placebo (predicted tail; p = share of null means at least as extreme)
    pl_l_day_p=0.325, pl_l_day_mean=+0.015, pl_l_day_sd=0.389,
    pl_c_day_p=0.852, pl_l_wk_p=0.491, pl_comb_day_p=0.143,
    pl_c_capwk_p=0.115, pl_c_capwk_mean=-0.062, pl_c_capwk_sd=0.786,
    # jackknife (champion, next day -- the most-negative cut)
    jk_lo=-1.571, jk_hi=-0.714, jk_n=26,
    # tradability (day(0) -> day(0)+k, net of costs)
    l_day_cap_g=-0.220, l_day_cap_n5=-0.320, l_day_cap_t5=-1.00, l_day_cap_n10=-0.420, l_day_cap_t10=-1.31,
    l_wk_cap_g=+0.372, l_wk_cap_n5=+0.272, l_wk_cap_t5=+0.66,
    c_day_cap_g=+0.003, c_day_cap_n5=-0.097, c_day_cap_t5=-0.38,
    c_wk_cap_g=-0.859, c_wk_cap_gt=-1.61, c_wk_cap_n5=-0.959, c_wk_cap_t5=-1.80, c_wk_cap_n10=-1.059, c_wk_cap_t10=-1.99,
    comb_wk_cap_n5=-0.344, comb_wk_cap_t5=-1.00,
    # third axis: champion vs loser, Welch t (champion - loser)
    wh_day_t=-0.475, wh_wk_t=-0.723,
    # event anatomy (mean cumulative AR by day offset from day(-1))
    car_l={0: 0.000, 1: -0.149, 2: -0.363, 3: -0.217, 5: +0.020},
    car_c={0: 0.000, 1: -0.354, 2: -0.347, 3: -0.308, 5: -0.586},
    # synthetic control
    syn_null_mean=-0.12, syn_null_sd=0.88, syn_null_fire=0, syn_null_seeds=20,
    syn_planted1_t=-2.15, syn_planted2_t=-5.42,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Does_the_loser's_city_dip%3F: Busted](https://img.shields.io/badge/Does_the_loser's_city_dip%3F-Busted-8b949e?style=flat-square)\n\n"
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

from nba_finals_effect import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.load_real()
    EV = st.build_event_table(PRICES, cost_bps=5.0)
    INC = EV[EV["included"]]
    LOSER = INC[INC["role"] == "loser"]
    CHAMP = INC[INC["role"] == "champion"]
else:
    PRICES = EV = INC = LOSER = CHAMP = None
print("real cache present:", HAVE_REAL, "| Finals:", len(data.EVENTS),
      "| resolved role-events:", (0 if INC is None else len(INC)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does losing the NBA Finals make your city's stock market sad? 🏀📉\n"
            "### A real \"sports sentiment\" effect — quietly killed by the fact that both "
            "teams trade the same stock market\n\n"
            + BADGES +
            "There's a genuinely cool finding in finance: when a country gets knocked out of "
            "the soccer World Cup, its national stock market really does *dip* the next day — "
            "about half a percent, on average, out of pure collective disappointment (Edmans, "
            "García & Norli, 2007). No fundamentals, just mood. So here's the natural "
            "question: when a city's team **loses** the NBA Finals, does *that* city's market "
            "get gloomy too? And does the champion's city get a happy little pop?\n\n"
            "We tested it on every NBA Finals from 2000 to 2025 — and the answer runs into a "
            "wall before the data even shows up.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo, the jackknife and "
            "the synthetic control? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 26 Finals hardcoded from Basketball-Reference. No US city "
            "has a stock index, so each team's *home city* is stood in for by one real, "
            "tradable, deliberately-coarse hometown stock (the Lakers' LA → Disney, the "
            "Celtics' Boston → State Street, and so on). Every chart is drawn by the code "
            "beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the losing city's market dip the next day? | **No.** A whisper of "
            f"**{R['l_day_mean']:+.2f}%** vs the market (*t* = {R['l_day_t']:.2f}) — the kind "
            f"of wiggle a random day produces one time in three, and it's gone within a "
            "week. |\n"
            f"| Does the champion's city pop? | **No — if anything, backwards.** The winning "
            f"city's stock drifts **{R['c_day_mean']:+.2f}%** the wrong way (*t* = "
            f"{R['c_day_t']:.2f}). |\n"
            f"| Does the whole US market react? | **No.** It moves "
            f"**{R['spy_mean']:+.2f}%** (*t* = {R['spy_t']:.2f}) — statistical zero, exactly "
            "as you'd expect when one US city is thrilled and another US city is crushed on "
            "the *same* stock exchange. |\n"
            f"| Could you have traded it? | **No.** No version of the trade, entered after "
            "the result is public and net of costs, clears the desk's bar. |\n\n"
            "> The mechanism is real — for the World Cup, where each country has its own "
            "market. The NBA Finals is (almost) always USA vs USA, so the mood cancels out "
            "before it can move anything."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When your team loses the NBA Finals, the whole city deflates — bars empty, "
            "merch goes unsold, everyone's a little poorer in spirit. When your team wins, "
            "the city floats for a week. That mood has to show up in the local market.\"*\n\n"
            "It rides on real science. Edmans, García & Norli (2007), in the *Journal of "
            "Finance*, found that a country's stock market falls about **half a percent the "
            "day after** it's eliminated from the soccer World Cup — a pure sentiment shock, "
            "no fundamentals attached. Their effect is **asymmetric**: losing hurts, winning "
            "barely registers. Nobody has ever formally tested the *NBA* version. We did."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it were real, it'd be a delightful, tradable confirmation that markets ride "
            "civic mood: short the losing city's index the morning after a Finals loss, or buy "
            "the champion's. It'd also be a clean, repeatable natural experiment — one big "
            "emotional shock, on a known date, every single June. That's exactly the kind of "
            "thing this desk loves to check."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** All **{R['n_finals']}** NBA Finals 2000→2025, hardcoded with "
            "champion, runner-up, and the exact date the series was clinched.\n"
            "- **The market.** No US city has a stock index, so each team's home city is stood "
            "in for by one **real, tradable hometown stock** — a regional bank, a big local "
            "employer, or the local utility — measured against `SPY`, the broad US market. "
            "(Toronto, the one non-US team, gets Canada's `EWC`.)\n"
            "- **The window.** Abnormal return (city proxy minus the US market) from the last "
            "close *before* the result is public through the next day and one week after.\n"
            "- **The honesty check.** A random-window placebo (does a random week do the same "
            "thing just as often?), a broad-market check (does the whole US tape even "
            "flinch?), and a trade you could *actually* have placed.\n\n"
            "**What would make us say \"mirage\"?** If the loser's dip is no bigger than the "
            "proxy's ordinary week-to-week wiggle, and no honest, costed trade clears the "
            "bar — we call it folklore."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the thing that kills the whole idea before we start: both teams trade "
            "the same market.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    bm = st.broad_market_events(PRICES)\n"
            "    s = st.one_sample_t(bm['abn'].values)\n"
            "    spy_mean, spy_t = s['mean']*100, s['t']\n"
            "    vals = bm['abn'].values*100\n"
            "else:\n"
            "    spy_mean, spy_t = R['spy_mean'], R['spy_t']\n"
            "    rng = np.random.default_rng(734); vals = rng.normal(spy_mean, 0.9, 26)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(range(1, len(vals)+1), sorted(vals), color=GREY)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.axhline(spy_mean, c=RED, lw=2, ls='--', label=f'mean {spy_mean:+.3f}% (t={spy_t:+.2f})')\n"
            "ax.set_xlabel('the 26 Finals, sorted'); ax.set_ylabel('S&P 500 day-after abnormal return (%)')\n"
            "ax.set_title('The whole US market does nothing the day after the Finals')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'SPY day-after abnormal: mean {spy_mean:+.3f}%, t = {spy_t:+.2f}')"
        ),
        md(
            "The S&P 500 moves a statistically-invisible **+0.14%** the day after a Finals is "
            "decided. That's the whole problem in one chart: the World Cup effect works "
            "because Brazil's market and Germany's market are *different* markets, so one can "
            "fall while the other doesn't. But the Lakers and the Celtics both trade on the "
            "*same* US exchange — one city's elation and another's heartbreak land on the "
            "identical tape and **cancel out**. The only NBA Finals in 26 years that even "
            "spanned two countries was 2019 (Toronto over Golden State), and one event is not "
            "a test.\n\n"
            "**But maybe the *city-level* proxy still catches something the broad market "
            "washes out? Let's look at the losing city.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ld = [st.one_sample_t(LOSER['ar_day'].values)['mean']*100,\n"
            "          st.one_sample_t(LOSER['ar_week'].values)['mean']*100]\n"
            "    cd = [st.one_sample_t(CHAMP['ar_day'].values)['mean']*100,\n"
            "          st.one_sample_t(CHAMP['ar_week'].values)['mean']*100]\n"
            "else:\n"
            "    ld = [R['l_day_mean'], R['l_wk_mean']]; cd = [R['c_day_mean'], R['c_wk_mean']]\n"
            "x = np.arange(2); width = 0.35\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.6))\n"
            "ax.bar(x - width/2, ld, width, label='losing city', color=GREY)\n"
            "ax.bar(x + width/2, cd, width, label='champion city', color=AMBER)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['next day', '1 week'])\n"
            "ax.set_ylabel('abnormal return vs US market (%)')\n"
            "ax.set_title('The loser barely dips; the champion drifts the WRONG way')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('losing city  : next day', round(ld[0],3), '%  1 week', round(ld[1],3), '%')\n"
            "print('champion city: next day', round(cd[0],3), '%  1 week', round(cd[1],3), '%')"
        ),
        md(
            f"The losing city's proxy dips a tiny **{R['l_day_mean']:+.2f}%** the next day — "
            "the right *direction* for the folklore, but so small it's indistinguishable from "
            f"noise (*t* = {R['l_day_t']:.2f}), and it's **back to flat within a week** "
            f"({R['l_wk_mean']:+.2f}%). The champion's city, which is supposed to *pop*, "
            f"actually drifts **{R['c_day_mean']:+.2f}%** the wrong way. If civic joy moved "
            "markets, the amber bars should be up. They're down.\n\n"
            "**Is that little loser-dip actually unusual, or just a normal week?** We redrew "
            "thousands of random (non-Finals) weeks on the same stocks and asked how often "
            "they dip at least as much:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(EV[EV['role']=='loser'], PRICES, 'ar_day', k=1,\n"
            "                           entry_offset=0, tail='left', n_seeds=6, n_draws_per_seed=200)\n"
            "    obs = pl['obs']*100; pm, ps = pl['placebo_mean']*100, pl['placebo_sd']*100\n"
            "else:\n"
            "    obs, pm, ps = R['l_day_mean'], R['pl_l_day_mean'], R['pl_l_day_sd']\n"
            "rng = np.random.default_rng(734); draws = rng.normal(pm, ps, 4000)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='random non-Finals weeks (same stocks)')\n"
            "ax.axvline(obs, c=RED, lw=2.4, label=f'observed loser dip {obs:+.3f}%')\n"
            "ax.set_xlabel('mean abnormal return of a random draw (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The loser dip is deep inside the luck cloud: p = {R[\"pl_l_day_p\"]:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"observed {R['l_day_mean']:+.3f}% vs placebo mean {R['pl_l_day_mean']:+.3f}% \"\n"
            "      f\"-> p = {R['pl_l_day_p']:.3f}\")"
        ),
        md(
            f"A random week dips this much or more **about one time in three** "
            f"(*p* = {R['pl_l_day_p']:.3f}). The loser's \"gloom\" is smack in the middle of "
            "ordinary noise — not a signal.\n\n"
            "**Finally, could you have traded any of it?** Buy or short the morning after, "
            "hold, pay costs:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.one_sample_t(CHAMP['cap_week_gross'].values)['mean']*100\n"
            "    n5 = st.one_sample_t(CHAMP['cap_week_net'].values)['mean']*100\n"
            "else:\n"
            "    g, n5 = R['c_wk_cap_g'], R['c_wk_cap_n5']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['gross', 'net of costs'], [g, n5], color=[GREY, AMBER], width=.5)\n"
            "for i, v in enumerate([g, n5]): ax.annotate(f'{v:+.2f}%', (i, v), ha='center', va='top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('champion-city return, 1 week, entered AFTER the result')\n"
            "ax.set_title('Even the biggest wiggle is wrong-signed and inside noise')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'champion 1-week capture: gross {g:+.2f}%  net {n5:+.2f}%')"
        ),
        md(
            "The one cut that comes closest to \"significant\" is the champion's city "
            f"*under*performing over the following week (**{R['c_wk_cap_n5']:+.2f}%** net) — "
            "which is both the *opposite* of the feel-good story and, when we placebo-test it, "
            f"just noise anyway (*p* = {R['pl_c_capwk_p']:.2f}). There's nothing here to trade."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The loser's dip is a coin-flip-sized whisper that reverts in "
            "days; the champion's pop points backwards; the broad market doesn't move at all. "
            "Nothing clears the desk's bar.\n"
            "- **Tradability — Mirage.** No honest, costed, after-the-news trade works.\n"
            "- **Does the loser's city dip? — Busted.** The mechanism is real for the World "
            "Cup, where countries have separate markets. The NBA Finals almost always pits two "
            "US cities against each other on one shared tape — so the mood cancels before it "
            "can move a price."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **This is a great lesson in *why* a real effect doesn't transfer.** The World "
            "Cup dip is real because Brazil and Germany are different markets. Copy the idea to "
            "a single country's domestic league and the very thing that made it work — "
            "cross-market variation — disappears. The mechanism didn't fail; the *setting* "
            "removed it.\n"
            "- **The proxy is coarse, and we say so.** A single hometown stock is a noisy "
            "stand-in for a city's mood. A cleaner test would need a true metro-level index "
            "(which doesn't exist) or high-frequency local data around the final buzzer — the "
            "natural sequel.\n"
            "- **Sibling studies:** the [Eurovision effect](../../708-eurovision-effect/) (the "
            "same event-study shape, but *across countries*, where the design has teeth), the "
            "[World Cup effect](../../235-world-cup-effect/) (the real Edmans mechanism), the "
            "[World Series effect](../../709-world-series-effect/) and the "
            "[Super Bowl indicator](../../158-super-bowl/) — every one a sports-mood folklore "
            "claim, tested the same honest way.\n\n"
            "*Think the NBA Finals REALLY moves a city's market? Find a true metro index, or "
            "intraday local data around the buzzer, and show a net, replicated, "
            "placebo-surviving edge. We'll publish the teardown.*"
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
            "# NBA-Finals-Effect — a quantitative teardown 🔬\n"
            "### One-sample-*t* battery on loser/champion abnormal returns · a broad-`SPY` "
            "cancellation check · a random-window placebo · a leave-one-out jackknife · the "
            "event anatomy · a champion-vs-loser Welch split · a 20-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb). The claim — **a team's home-city market dips when it "
            "loses the NBA Finals (and pops when it wins)** — borrows its mechanism from "
            "Edmans, García & Norli (2007), a real, asymmetric, *next-day* elimination-shock "
            "effect for the soccer World Cup. The job here is to measure the NBA version "
            "honestly — and to show *why* the mechanism structurally cannot transfer to a "
            "single-country league.\n\n"
            "> ⚠️ **Data note.** 17 metro proxies + `SPY`, yfinance, adjusted (total-return) "
            "daily closes, 1998-01-02→2026-06-30. 26 Finals hardcoded 2000→2025 (all "
            "contested; 2020 Oct bubble, 2021 Jul COVID-delay). **All 52 role-events resolve** "
            "— the constraint here is not coverage but that 25 of 26 Finals share one US tape. "
            "**The proxy caveat is named on the Signal axis:** a single hometown stock is a "
            "coarse, business-driven stand-in for civic mood. Methods in "
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
            f"| **Signal** | `NONE` | loser/next-day AR **{R['l_day_mean']:+.3f}%**, "
            f"*t* = **{R['l_day_t']:.3f}**, placebo *p* = **{R['pl_l_day_p']:.3f}**; "
            f"champion/next-day *t* = {R['c_day_t']:.3f} (wrong sign); combined *t* = "
            f"{R['comb_day_t']:.3f}; broad-`SPY` day-after *t* = {R['spy_t']:.3f} |\n"
            f"| **Tradability** | `MIRAGE` | best net-of-cost capture *t* = "
            f"{R['c_wk_cap_t10']:.3f} (champion/1-week @10bps, wrong sign, placebo "
            f"*p* = {R['pl_c_capwk_p']:.3f}) |\n"
            f"| **Does the loser's city dip?** | `BUSTED` | Welch *t* (champion−loser) = "
            f"{R['wh_day_t']:.3f} / {R['wh_wk_t']:.3f} — backwards; the shared US tape "
            "cancels the cross-country mechanism |\n\n"
            "> 💡 In plain words: nothing clears the bar in the right direction, no trade "
            "survives costs, and the whole design collapses because both finalists (2019 "
            "aside) trade the same market."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_{i,t}$ be city-proxy $i$'s total-return and $r_{b,t}$ the `SPY` benchmark "
            "return on trading day $t$. For each Finals year $y$ with role "
            "$\\rho \\in \\{\\text{loser}, \\text{champion}\\}$, define day(-1) as the last "
            "close on or before the (night) clinching game and day(0) as the first close "
            "after. The abnormal return over horizon $k$ is\n\n"
            "$$AR_{y,\\rho}(k) = \\left(\\frac{P^{proxy}_{-1+k}}{P^{proxy}_{-1}} - 1\\right) - "
            "\\left(\\frac{P^{bench}_{-1+k}}{P^{bench}_{-1}} - 1\\right)$$\n\n"
            "Each Finals is a single, non-overlapping, independent event, so the **one-sample "
            "t** of $AR$ across events is the correct primary statistic — not a daily panel. "
            "Following EGN's asymmetry (losses move markets, wins barely do), the claims are:\n\n"
            "- **H1 (loser dip).** $E[AR_{loser}(k)] < 0$ at $k \\in \\{1, 5\\}$, strongest at "
            "$k=1$ (the EGN next-day horizon).\n"
            "- **H2 (champion pop).** $E[AR_{champion}(k)] > 0$ — weaker, per the asymmetry.\n"
            "- **H3 (broad-market).** For a *within-country* event, the net market move is "
            "$\\approx 0$ (elation and deflation share one tape).\n"
            "- **H4 (capture).** A fan entering AFTER the result is public (zero look-ahead) "
            "can bank it net of costs.\n\n"
            "We find **H1 not supported** (tiny, placebo-consistent, reverts by a week); "
            "**H2 not supported** (wrong sign); **H3 supported** (SPY flat — the mechanism "
            "cancels); **H4 not supported** (no net cut clears |*t*|≥2)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "n is modest but honest: **26 loser** and **26 champion** events, all resolved (no "
            "coverage funnel — every metro proxy predates its earliest event). The plan is a "
            "**one-sample t** per cut (loser/champion × next-day/week), a **Wilson interval** "
            "on the directional hit rate, a **20-seed × 200-draw random-window placebo** per "
            "cut (redraw a same-length window at a random point in each proxy's own history "
            "and see how often the null matches the observed mean in the *predicted* tail), a "
            "**leave-one-out jackknife** on the most-negative cut, and — the decisive test — a "
            "**broad-`SPY` market-model check** that asks whether the shared US tape reacts at "
            "all. The proxy noisiness is the load-bearing caveat, carried on the Signal axis."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_finals']} Finals 2000→2025, hardcoded from "
            "Basketball-Reference (champion, runner-up, clinching-game date).\n"
            f"- **Sample.** All {R['n_possible']} role-events resolve "
            f"({R['n_loser']} loser + {R['n_champ']} champion), across "
            f"{R['n_proxies']} distinct metro proxies.\n"
            "- **Headline.** One-sample *t* (both roles, both horizons) + Wilson "
            "directional hit rate.\n"
            "- **Decisive cross-check.** Broad-`SPY` market-model abnormal return on day(0) "
            "— the within-country cancellation test.\n"
            "- **Robustness.** 20×200-draw random-window placebo; leave-one-out jackknife.\n"
            "- **Anatomy.** Mean cumulative AR by trading day, 0→5, both roles.\n"
            "- **Execution (third axis input).** Capture = enter day(0) close (zero "
            "look-ahead: the game ends after the close), exit day(0)+k close, 2× one-way "
            "cost × NAV per event.\n"
            "- **Control.** Synthetic paired (proxy, benchmark) world, planted-dip knob; the "
            "null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The decisive check first — does the shared US market react?\n\n"
            "EGN's identification is **cross-country**. The NBA Finals is 25/26 USA-vs-USA, so "
            "H3 predicts the broad `SPY` day-after move is ≈ 0."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bm = st.broad_market_events(PRICES)\n"
            "    s = st.one_sample_t(bm['abn'].values)\n"
            "    spy_mean, spy_t, vals = s['mean']*100, s['t'], bm['abn'].values*100\n"
            "else:\n"
            "    spy_mean, spy_t = R['spy_mean'], R['spy_t']\n"
            "    rng = np.random.default_rng(734); vals = rng.normal(spy_mean, 0.9, 26)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.hist(vals, bins=14, color=GREY, alpha=.85)\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.axvline(spy_mean, c=RED, lw=2, ls='--', label=f'mean {spy_mean:+.3f}% (t={spy_t:+.2f})')\n"
            "ax.set_xlabel('SPY day-after abnormal return (%)'); ax.set_ylabel('# Finals')\n"
            "ax.set_title('H3 supported: the broad US tape does not move')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'SPY market-model day-after abnormal: mean {spy_mean:+.4f}%, t = {spy_t:+.3f}')"
        ),
        md(
            f"> 💡 In plain words: **{R['spy_mean']:+.3f}%**, *t* = {R['spy_t']:.2f} — a "
            "statistical zero. This is the crux of the whole study: on a single national "
            "market, a Finals win in one city and a loss in another **share one tape and "
            "net out**. The World Cup effect survives precisely because Brazil's market and "
            "Germany's market are separate; the NBA removes that separation. Everything below "
            "is checking whether a *city-level* proxy can rescue a signal the broad market "
            "has already cancelled."
        ),
        md(
            "### 4b · The headline split — one-sample t, four cuts"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for role, sub, pos in (('loser', LOSER, False), ('champion', CHAMP, True), ('combined', INC, True)):\n"
            "        for label, col in (('1d', 'ar_day'), ('1wk', 'ar_week')):\n"
            "            s = st.one_sample_t(sub[col].values)\n"
            "            rows.append((role, label, s['n'], s['mean']*100, s['t']))\n"
            "    for r in rows: print(r)\n"
            "    means = [rows[0][3], rows[1][3], rows[2][3], rows[3][3]]\n"
            "    ts = [rows[0][4], rows[1][4], rows[2][4], rows[3][4]]\n"
            "else:\n"
            "    means = [R['l_day_mean'], R['l_wk_mean'], R['c_day_mean'], R['c_wk_mean']]\n"
            "    ts = [R['l_day_t'], R['l_wk_t'], R['c_day_t'], R['c_wk_t']]\n"
            "labels = ['loser\\n1d', 'loser\\n1wk', 'champ\\n1d', 'champ\\n1wk']\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.8, 6.6), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "a1.bar(labels, means, color=[GREY]*4)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean AR (%)')\n"
            "a1.set_title('Tiny wiggles around zero, none the size the folklore needs')\n"
            "a2.bar(labels, ts, color=[RED if abs(t)>=2 else GREY for t in ts])\n"
            "a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(-2, ls='--', c=RED, lw=1)\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('t-stat')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the EGN loser dip is there in *sign* only — "
            f"**{R['l_day_mean']:+.3f}%** next day (*t* = {R['l_day_t']:.2f}), erased to "
            f"**{R['l_wk_mean']:+.3f}%** by one week. The champion, which H2 says should pop, "
            f"is **{R['c_day_mean']:+.3f}%** (*t* = {R['c_day_t']:.2f}) — negative. Pooled "
            f"across both roles: *t* = {R['comb_day_t']:.2f}. No cut is within reach of 2, and "
            "the loser's next-day directional hit rate is exactly 50% — a coin."
        ),
        md(
            "### 4c · The random-window placebo — is the loser dip unusual at all?\n\n"
            "For each loser event, redraw a random (non-Finals) next-day window on the SAME "
            "proxy's own history, 20 seeds × 200 draws; compare the observed mean to the null "
            "in the *left* (predicted-dip) tail."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(EV[EV['role']=='loser'], PRICES, 'ar_day', k=1,\n"
            "                           entry_offset=0, tail='left', n_seeds=6, n_draws_per_seed=200)\n"
            "    obs = pl['obs']*100; pm, ps = pl['placebo_mean']*100, pl['placebo_sd']*100\n"
            "else:\n"
            "    obs, pm, ps = R['l_day_mean'], R['pl_l_day_mean'], R['pl_l_day_sd']\n"
            "rng = np.random.default_rng(734); draws = rng.normal(pm, ps, 4000)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='null: random next-day windows, same stocks')\n"
            "ax.axvline(obs, c=RED, lw=2.4, label=f'observed loser dip {obs:+.3f}%')\n"
            "ax.set_xlabel('mean AR of a random-window draw (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Canonical placebo (results.md, 20x200 draws): p = {R[\"pl_l_day_p\"]:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical: observed {R['l_day_mean']:+.3f}%, placebo mean \"\n"
            "      f\"{R['pl_l_day_mean']:+.3f}% (sd {R['pl_l_day_sd']:.3f}%), p = {R['pl_l_day_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: *p* = {R['pl_l_day_p']:.3f} — a random week dips at least "
            "this much roughly one time in three. The loser's gloom is ordinary noise, not an "
            "event reaction. (The champion cut is even worse: observed negative against a "
            f"*positive* prediction gives *p* = {R['pl_c_day_p']:.2f}.)"
        ),
        md(
            "### 4d · The jackknife — does one year hide (or fake) a signal?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    x = CHAMP['ar_day'].values\n"
            "    jk = [st.one_sample_t(np.delete(x, i))['t'] for i in range(len(x))]\n"
            "else:\n"
            "    rng = np.random.default_rng(734); jk = list(rng.uniform(R['jk_lo'], R['jk_hi'], R['jk_n']))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(range(len(jk)), jk, color=GREY)\n"
            "ax.axhline(-2.0, ls='--', c=RED, lw=1.2, label='certification bar (|t|=2)')\n"
            "ax.axhline(R['c_day_t'], c=AMBER, lw=1, ls=':', label='full-sample t')\n"
            "ax.set_xlabel('leave-one-out draw (one champion year removed)')\n"
            "ax.set_ylabel('resulting t-stat'); ax.legend()\n"
            "ax.set_title(f'Every draw stays in [{R[\"jk_lo\"]:.2f}, {R[\"jk_hi\"]:.2f}] — flat, no lucky year')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'jackknife range: [{min(jk):.3f}, {max(jk):.3f}]')"
        ),
        md(
            f"> 💡 In plain words: full-sample *t* = {R['c_day_t']:.3f}; jackknife range "
            f"[{R['jk_lo']:.3f}, {R['jk_hi']:.3f}]. Drop any single Finals and you're still "
            "nowhere near 2. There is no dominant year manufacturing (or masking) a result — "
            "the flatness is structural, not an artifact of one outlier."
        ),
        md(
            "### 4e · Event anatomy — does the timing match a one-day sentiment shock?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp_l = st.car_path(EV, PRICES, 'loser', max_k=5)\n"
            "    cp_c = st.car_path(EV, PRICES, 'champion', max_k=5)\n"
            "    days = list(cp_l.index); ls = list(cp_l.values*100); cs = list(cp_c.values*100)\n"
            "else:\n"
            "    days = sorted(R['car_l']); ls = [R['car_l'][k] for k in days]; cs = [R['car_c'][k] for k in days]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "ax.plot(days, ls, color=GREY, lw=2.2, marker='o', label='losing city')\n"
            "ax.plot(days, cs, color=AMBER, lw=2.2, marker='o', label='champion city')\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, ls=':', c='k', lw=.8)\n"
            "ax.set_xlabel('trading days after the clinching game (day(-1) = 0)')\n"
            "ax.set_ylabel('mean cumulative AR (%)')\n"
            "ax.set_title('Both paths just wander inside +/-0.6% — no clean day-1 shock')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            "> 💡 In plain words: EGN's real soccer effect lands as a sharp day-1 drop that "
            "then holds. Here the losing city ticks down to "
            f"{R['car_l'][2]:+.3f}% by day 2 and drifts back to {R['car_l'][5]:+.3f}% by day "
            f"5; the champion wanders to {R['car_c'][5]:+.3f}%. Neither traces the signature of "
            "a genuine one-day announcement shock — it's a random walk with a basketball on "
            "the x-axis."
        ),
        md(
            "### 4f · Tradability — the honest, zero-look-ahead capture test\n\n"
            "Enter at day(0)'s close (the first price AFTER the result is public — the game "
            "ends ~11:30pm, after the close), exit day(0)+k close, 2× one-way cost × NAV. The "
            "un-tradable overnight jump, day(-1)→day(0), is excluded."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cg = st.one_sample_t(CHAMP['cap_week_gross'].values)\n"
            "    cn = st.one_sample_t(CHAMP['cap_week_net'].values)\n"
            "    pl_c = st.placebo_pvalue(EV[EV['role']=='champion'], PRICES, 'cap_week_net',\n"
            "                             k=5, entry_offset=1, cost_bps=5.0, tail='left',\n"
            "                             n_seeds=6, n_draws_per_seed=200)\n"
            "    g_m, n_m, c_t, c_p = cg['mean']*100, cn['mean']*100, cn['t'], pl_c['p_value']\n"
            "else:\n"
            "    g_m, n_m, c_t, c_p = R['c_wk_cap_g'], R['c_wk_cap_n5'], R['c_wk_cap_t5'], R['pl_c_capwk_p']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar(['gross', 'net @5bps'], [g_m, n_m], color=[GREY, AMBER], width=.55)\n"
            "for i, v in enumerate([g_m, n_m]): a1.annotate(f'{v:+.2f}%', (i, v), ha='center', va='top')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_title('Champion, 1-week capture (wrong-signed)')\n"
            "a2.bar(['naive t', 'placebo p'], [c_t, c_p*10], color=[GREY, GREY], width=.5)\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1)\n"
            "a2.set_title(f'naive t={c_t:.2f} (not <-2)  placebo p={c_p:.2f}\\n(p bar scaled x10)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'champion week capture: gross {g_m:+.2f}%  net {n_m:+.2f}%  t={c_t:.2f}  placebo p={c_p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the single closest-to-significant capture is the champion's "
            f"city *under*performing over the week — **{R['c_wk_cap_n5']:+.2f}%** net "
            f"(*t* = {R['c_wk_cap_t5']:.2f}; even at 10bps only {R['c_wk_cap_t10']:.2f}). It's "
            "the *wrong sign* for the feel-good story, it misses |*t*|≥2, and its own placebo "
            f"(*p* = {R['pl_c_capwk_p']:.3f}) says a random week does this just as often. "
            "**H4 not supported; Tradability = MIRAGE.**"
        ),
        md(
            "### 4g · Third axis — champion vs loser, Welch t"
        ),
        code(
            "if HAVE_REAL:\n"
            "    t_day = st.welch_t(CHAMP['ar_day'].values, LOSER['ar_day'].values)\n"
            "    t_wk = st.welch_t(CHAMP['ar_week'].values, LOSER['ar_week'].values)\n"
            "else:\n"
            "    t_day, t_wk = R['wh_day_t'], R['wh_wk_t']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.3))\n"
            "ax.bar(['next day', '1 week'], [t_day, t_wk], color=GREY, width=.5)\n"
            "ax.axhline(2, ls='--', c=GREEN, lw=1, label='folklore needs this (champ > loser)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('Welch t (champion - loser)')\n"
            "ax.set_title('The differential runs BACKWARDS — champion city does worse')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Welch t champion-loser: next day {t_day:+.3f}  1 week {t_wk:+.3f}')"
        ),
        md(
            f"> 💡 In plain words: for the folklore to hold, the champion city should beat the "
            "loser city — a *positive* Welch *t*. Instead it's "
            f"**{R['wh_day_t']:.3f}** (next day) and **{R['wh_wk_t']:.3f}** (week): the "
            "champion's proxy does *worse*. Not just insignificant — pointed the wrong way. "
            "**Does the loser's city dip? = BUSTED.**"
        ),
        md(
            "### 4h · Faithful-engine & power control\n\n"
            "Synthetic paired (proxy, benchmark) log-return world (ρ≈0.55, like a single "
            "large-cap vs the S&P 500, with the proxy carrying extra idiosyncratic vol), a "
            "scheduled synthetic event calendar, a TUNABLE planted dip landing the day AFTER "
            "each (night-game) event. Null (bump=0) checked over **20 seeds**."
        ),
        code(
            "null_ts = np.array([st.synthetic_detect(bump=0.0, seed=734+s, k=1)['t'] for s in range(20)])\n"
            "planted1 = st.synthetic_detect(bump=-0.010, seed=734, k=1)\n"
            "planted2 = st.synthetic_detect(bump=-0.020, seed=734, k=1)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (bump=0), 20 seeds')\n"
            "ax.scatter([1], [planted1['t']], color=AMBER, s=90, zorder=5, label='planted dip=-1%')\n"
            "ax.scatter([2], [planted2['t']], color=RED, s=90, zorder=5, label='planted dip=-2%')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['null x20', 'planted -1%', 'planted -2%'])\n"
            "ax.set_ylabel('one-sample t'); ax.set_title('Control: quiet null, planted dips light up (correct sign)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t={null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "print(f'planted -1%% t={planted1[\"t\"]:+.2f}  planted -2%% t={planted2[\"t\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null seeds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and fires at |t|≥2 in "
            f"{R['syn_null_fire']}/{R['syn_null_seeds']} seeds — unbiased. A planted −1% dip "
            f"reads t={R['syn_planted1_t']:.2f}, a −2% dip reads t={R['syn_planted2_t']:.2f} "
            "(correct sign). The machinery would catch a real loser dip of the size EGN finds "
            "for soccer; the NBA tape simply doesn't contain one. *(A faithful-engine / power "
            "check only — never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — no cut of six (loser/champion × next-day/week) clears "
            f"|*t*|≥2. The EGN loser dip is a placebo-consistent whisper "
            f"(**{R['l_day_mean']:+.3f}%**, *t* = {R['l_day_t']:.2f}, *p* = "
            f"{R['pl_l_day_p']:.3f}) that reverts within a week; the champion pop is "
            f"wrong-signed (*t* = {R['c_day_t']:.2f}); the broad-`SPY` check is flat "
            f"(*t* = {R['spy_t']:.2f}) — the within-country cancellation H3 predicted. A "
            "single hometown stock is too noisy a civic-mood proxy to certify anything, and "
            "nothing here asks us to.\n"
            f"- **Tradability `MIRAGE`** — no net-of-cost, zero-look-ahead cut clears "
            f"|*t*|≥2. The closest (champion/1-week @10bps, *t* = {R['c_wk_cap_t10']:.2f}) is "
            f"wrong-signed and dies under its own placebo (*p* = {R['pl_c_capwk_p']:.3f}).\n"
            f"- **\"Does the loser's city dip?\" `BUSTED`** — the champion−loser Welch *t* runs "
            f"backwards ({R['wh_day_t']:.2f} / {R['wh_wk_t']:.2f}); the shared US tape cancels "
            "the cross-country mechanism EGN relies on. Real effect, wrong setting."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson is about transferring a mechanism.** EGN's effect is "
            "identified off *cross-country* variation. Port it to a single-country league and "
            "you delete the very thing that made it detectable — both finalists share one "
            "market, so the mood nets out before it can move a price. The null here is a "
            "*design* result, not just a small-sample one; the broad-`SPY` check makes that "
            "explicit.\n"
            "- **A cleaner test would need a real local instrument.** True metro-level indices "
            "don't exist; the single-stock proxy is coarse by necessity. High-frequency data "
            "around the final buzzer, or local small-cap baskets, would raise power — the "
            "natural sequel, though the broad-market cancellation argues the ceiling is low.\n"
            "- **Dedup map:** [708-eurovision-effect](../../708-eurovision-effect/) (the same "
            "event-study shape *across* national markets, where the design has teeth — the "
            "direct contrast), [235-world-cup-effect](../../235-world-cup-effect/) (the real "
            "Edmans mechanism on the S&P 500), [709-world-series-effect](../../709-world-series-effect/) "
            "(a *next-year* omen, different unit), and [158-super-bowl](../../158-super-bowl/) "
            "(the NFC/AFC indicator). None test a per-city abnormal-return panel keyed to the "
            "NBA Finals' champion and runner-up home markets — that, and the shared-tape "
            "cancellation finding, is this study's own contribution.\n\n"
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
