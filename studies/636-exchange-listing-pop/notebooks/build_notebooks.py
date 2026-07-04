"""Generate the two narrative notebooks for Study 636 (Exchange-Listing-Pop).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached Binance
close panel under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic control runs anywhere with no network.
Heavy pieces (the 2,000-draw placebo, the 20-seed sweep) are NOT re-run in the
notebooks — canonical numbers are quoted from ``R``; in-notebook placebos use a small
draw count purely for the picture.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (Binance klines,
# 129 Coinbase listing events 2019-2026, panel 2017-08-17 -> 2026-07-02).
R = dict(
    start="2017-08-17", end="2026-07-02", years=8.9, days=3242, n_coins=134,
    n_events=129, n_clusters=95, n_pre=56, n_post=73,
    # (label, a, b, mean CAR log %, simple %, cluster t)
    windows=[
        ("early run-up [-20..-6]", -20, -6, 2.40, 2.43, 1.25),
        ("run-up [-5..-1]", -5, -1, 10.23, 10.77, 5.91),
        ("listing day [0]", 0, 0, 0.77, 0.77, 0.82),
        ("THE POP [-5..0]", -5, 0, 11.00, 11.63, 5.74),
        ("post [+1..+5]", 1, 5, -6.14, -5.96, -3.93),
        ("THE FADE [+1..+30]", 1, 30, -17.45, -16.02, -5.73),
    ],
    pop=dict(car=11.00, t=5.74, t_month=5.30, med=7.11, share_pos=76,
             obs_event=14.60, welch=6.92, p="< 0.0005", n_pooled=188947),
    fade=dict(car=-17.45, t=-5.73, t_month=-4.96, med=-17.97, share_pos=19,
              obs_event=-16.79, welch=-4.38, p="< 0.0005", n_pooled=185676),
    era=dict(pre_pop=17.17, pre_pop_t=5.52, post_pop=7.06, post_pop_t=3.06,
             pre_fade=-16.98, pre_fade_t=-3.95, post_fade=-17.76, post_fade_t=-4.23,
             diff_welch=2.61, n_pre_cl=37, n_post_cl=58),
    # follower LONG (hold, cost bps, gross %, t, net %, t, hit %)
    long_tr=[(5, 10, -4.58, -2.59, -5.03, -2.84, 19),
             (5, 25, -4.58, -2.59, -5.63, -3.17, 17),
             (5, 50, -4.58, -2.59, -6.63, -3.74, 14),
             (30, 10, -11.97, -3.92, -12.61, -4.13, 18),
             (30, 25, -11.97, -3.92, -13.21, -4.33, 18),
             (30, 50, -11.97, -3.92, -14.21, -4.66, 18)],
    # fade SHORT at +30d (cost bps, gross %, t, net %, t, hit %) at 10%/yr borrow
    short_tr=[(10, 11.97, 3.92, 10.74, 3.52, 79),
              (25, 11.97, 3.92, 10.14, 3.32, 76),
              (50, 11.97, 3.92, 9.14, 3.00, 75)],
    # borrow sensitivity (borrow %/yr, net %, t, hit %) at 25 bps, +30d
    borrow=[(10, 10.14, 3.32, 76), (50, 6.86, 2.25, 74), (100, 2.75, 0.90, 67)],
    # announcement gaps (coin, announced, trading, gap days)
    gaps=[("MATIC", "2021-03-09", "2021-03-11", 2), ("ADA", "2021-03-16", "2021-03-18", 2),
          ("DOGE", "2021-06-01", "2021-06-03", 2), ("SOL", "2021-05-20", "2021-06-17", 28)],
    # synthetic control (pop %, fade %, car pop %, t, car fade %, t)
    syn=[(0.0, 0.0, 0.41, 0.21, 2.89, 0.70), (15.0, 10.0, 15.41, 7.71, -7.11, -1.73)],
    fingerprint="f6a046325f17",
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Gone_before_you_can_act%3F: Confirmed](https://img.shields.io/badge/Gone_before_you_can_act%3F-Confirmed-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from exchange_listing_pop import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PX = data.load_real()
    EV = data.events_frame()
    EV = EV[EV["coin"].isin(PX.columns)].reset_index(drop=True)
    ABN = st.abnormal_returns(PX)
else:
    PX = EV = ABN = None
print("real cache present:", HAVE_REAL,
      "| events:", (0 if EV is None else len(EV)),
      "| panel:", (None if PX is None else PX.shape))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a Coinbase listing really make a coin pop — and then give it all back? 🚀\n"
            "### The \"Coinbase effect\", measured on 129 real listings — in plain English\n\n"
            + BADGES +
            "Crypto folklore says that when Coinbase — the biggest regulated US exchange — announces "
            "it will list a coin, the coin **jumps double digits** on the news… and then **bleeds it "
            "all back** over the following weeks. It even has a name: the **Coinbase effect**. There "
            "was a real US criminal case about people front-running these announcements (the first "
            "crypto insider-trading conviction), so somebody clearly believed the pop was worth "
            "cheating for.\n\n"
            "We measured it properly: **129 Coinbase listings, 2019–2026**, with prices from Binance — "
            "where these coins already trade long before Coinbase adds them.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, placebos and cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Honesty note up front.** Returns are measured **relative to Bitcoin** (so a "
            "market-wide crypto rally doesn't get credited to the listing), and the event dates come "
            "from Coinbase's **own API** (the first day each coin actually traded there), not from "
            "hand-picked headlines. Every chart is drawn by the code beside it."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the pop exist? | **Yes, emphatically.** On average a coin gains **~+11% vs Bitcoin** "
            "in the week into its Coinbase listing — across 129 events, far too consistently to be luck "
            "(odds well under 1 in 2,000). |\n"
            "| Does it give it back? | **Yes — and then some.** Over the next 30 days the average coin "
            "loses **~−17% vs Bitcoin**. The hangover is *bigger* than the party. |\n"
            "| Can *you* catch the pop? | **No.** Almost all of it happens **between the announcement "
            "and the first trade** — before you can click buy. The listing day itself adds less than "
            "+1%. Buying at the first possible close then holding a month lost **−12.6% net** on "
            "average. |\n"
            "| So is there *any* trade? | Only the uncomfortable one: **shorting** the freshly listed "
            "coin — about **+10% net per event** on paper — *if* you can actually borrow a hot new "
            "altcoin, which is exactly where borrowing is scarce and expensive. |\n\n"
            "> The folklore is **true as a description** and a **trap as an instruction**. The pop is "
            "real; it just isn't *yours*."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Getting listed on Coinbase is a golden ticket: the coin pops double digits on the "
            "announcement — then gives it all back.\"*\n\n"
            "The mechanism is plausible: a Coinbase listing (a) opens the coin to a huge pool of US "
            "buyers who couldn't easily hold it before, and (b) works as a **stamp of approval** "
            "(Coinbase reviews assets before listing). Both fire **when the news drops** — not when "
            "trading starts. Messari's research desk quantified the pop in 2020–21; the DOJ's "
            "*U.S. v. Wahi* case (2022) proved insiders traded on it; Coinbase answered with a public "
            "**listing roadmap** (April 2022) meant to kill the information edge. Our question: what's "
            "left, and for whom?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the pop were catchable, it would be one of the easiest trades in crypto: watch the "
            "Coinbase blog, buy, wait, sell. If instead it's **priced before you can act**, then the "
            "folklore — endlessly repeated as a buying strategy — is actually a **wealth transfer** "
            "from the people who buy the news to the people who front-ran it. And if the *give-back* "
            "is real too, buying the news isn't just late — it's buying the top of a mountain."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **Events.** {R['n_events']} Coinbase listings 2019–2026. Day 0 = the first day the "
            "coin's USD book traded on Coinbase Exchange, straight from **Coinbase's own API** — no "
            "hand-curated dates. A price-agreement check between venues kills ticker mix-ups.\n"
            f"- **Prices.** Binance daily closes ({R['years']:.0f} years, {R['n_coins']} coins + BTC) — "
            "the coins trade there *months or years before* Coinbase lists them, so we can see the "
            "before, the during and the after.\n"
            "- **Measure.** Each coin's return **minus Bitcoin's** (so \"crypto went up that week\" "
            "doesn't count), averaged across events, with same-day listings grouped (Coinbase often "
            "announces several coins in one post).\n"
            "- **Then:** compare the days *before* the listing, the listing day, and the month after."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — one chart tells the whole story\n\n"
            "Average cumulative return vs Bitcoin from 20 days before the Coinbase listing to 30 days "
            "after, across all events. Watch where the mountain peaks."
        ),
        code(
            "if HAVE_REAL:\n"
            "    M = st.event_matrix(ABN, EV, -20, 30)\n"
            "    path = M.mean(axis=1).cumsum() * 100\n"
            "else:\n"
            "    path = None\n"
            "fig, ax = plt.subplots(figsize=(9.6, 5.0))\n"
            "if path is not None:\n"
            "    ax.plot(path.index, path.values, lw=2.5, color=GREEN)\n"
            "    ax.fill_between(path.index, 0, path.values, alpha=.12, color=GREEN)\n"
            "ax.axvline(0, ls='--', c=RED, lw=1.5, label='listing day on Coinbase (day 0)')\n"
            "ax.axhline(0, c=GREY, lw=1)\n"
            "ax.set_xlabel('days around the Coinbase listing'); ax.set_ylabel('avg cumulative return vs BTC (%)')\n"
            "ax.set_title('The Coinbase effect: straight up into the listing - straight down after')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "if path is not None:\n"
            "    print(f'peak around day 0: {path.loc[0]:+.1f}%   day +30: {path.loc[30]:+.1f}%')"
        ),
        md(
            "There's the legend, drawn by the tape. Measured from 20 days out, the average coin "
            "climbs ~+20% vs Bitcoin into its listing — with the headline chunk, "
            f"**+{R['pop']['car']:.0f}%**, packed into the final five days — then gives "
            f"**~{R['fade']['car']:.0f} points back** over the month after. The mountain peaks "
            "**almost exactly at day 0** — the first day you could actually buy on Coinbase."
        ),
        md(
            "**Zoom in: when exactly does the pop happen?** Split the event into the run-up, the "
            "listing day itself, and the aftermath."
        ),
        code(
            "if HAVE_REAL:\n"
            "    segs = [st.window_summary(ABN, EV, a, b) for a, b in [(-5, -1), (0, 0), (1, 5), (1, 30)]]\n"
            "    vals = [s['mean_car'] * 100 for s in segs]\n"
            "else:\n"
            "    vals = [R['windows'][1][3], R['windows'][2][3], R['windows'][4][3], R['windows'][5][3]]\n"
            "labels = ['5 days BEFORE\\n(announcement window)', 'listing day\\nitself', '5 days\\nafter', '30 days\\nafter']\n"
            "colors = [GREEN, AMBER, RED, RED]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "ax.bar(labels, vals, color=colors, width=.6)\n"
            "for i, v in enumerate(vals): ax.annotate(f'{v:+.1f}%', (i, v), ha='center',\n"
            "        va='bottom' if v > 0 else 'top')\n"
            "ax.axhline(0, c=GREY, lw=1)\n"
            "ax.set_ylabel('avg return vs BTC (%)')\n"
            "ax.set_title('The pop happens BEFORE the listing - the first tradable day is the top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('run-up / day0 / +5d / +30d:', [f'{v:+.1f}%' for v in vals])"
        ),
        md(
            f"> The green bar — **+{R['windows'][1][3]:.1f}%** — happens in the few days between the "
            f"announcement and the first trade (2 days apart in the old regime: DOGE was announced "
            f"June 1, 2021 and tradable June 3). The **listing day itself adds just "
            f"+{R['windows'][2][3]:.1f}%** — statistically nothing. By the time *you* can buy, the "
            "move has already been made by people who acted on the announcement — or before it."
        ),
        md(
            "**So what happens if you buy anyway?** The only version of the trade available to a "
            "normal person: buy at the close of the listing day, hold, measured against Bitcoin, "
            "after realistic fees."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [st.trade_summary(st.follower_trade(PX, EV, hold=h, cost_bps=25.0, hedged=True))\n"
            "            for h in (5, 30)]\n"
            "    vals = [r['net_pct'] for r in rows]\n"
            "    sh = st.trade_summary(st.follower_trade(PX, EV, hold=30, cost_bps=25.0, hedged=True, side='short'))\n"
            "    short_net = sh['net_pct']\n"
            "else:\n"
            "    vals = [R['long_tr'][1][4], R['long_tr'][4][4]]; short_net = R['short_tr'][1][3]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "bars = ['BUY the listing,\\nsell +5 days', 'BUY the listing,\\nsell +30 days', 'SHORT the listing,\\ncover +30 days']\n"
            "vv = [vals[0], vals[1], short_net]\n"
            "ax.bar(bars, vv, color=[RED, RED, AMBER], width=.55)\n"
            "for i, v in enumerate(vv): ax.annotate(f'{v:+.1f}%', (i, v), ha='center', va='bottom' if v > 0 else 'top')\n"
            "ax.axhline(0, c=GREY, lw=1)\n"
            "ax.set_ylabel('avg NET return per event (%)')\n"
            "ax.set_title('Buying the news loses money - the only paper edge is the fade (short)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net per event:', [f'{v:+.1f}%' for v in vv], '(25 bps/leg, shorts pay borrow)')"
        ),
        md(
            f"Buying the listing and holding a month lost **{R['long_tr'][4][4]:.1f}% net per event** "
            f"on average — and lost in **{100-R['long_tr'][4][6]:.0f}%** of cases. The only trade the "
            f"tape actually pays is the **fade**: short the freshly listed coin, hedge with Bitcoin, "
            f"collect **+{R['short_tr'][1][3]:.1f}% net** on paper. The catch: you must **borrow** a "
            "hot, just-listed altcoin to short it — precisely the coins that are hard and expensive "
            "to borrow. At 100%/yr borrow the edge is gone. Real, but only for whoever has the "
            "borrow — which is probably not you."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** The pop (**+{R['pop']['car']:.0f}%** vs BTC) and the give-back "
            f"(**{R['fade']['car']:.0f}%**) are both on the tape at overwhelming significance across "
            f"{R['n_events']} listings. The folklore describes reality.\n"
            f"- **Tradability — Fragile.** The folklore's *instruction* (buy the news) loses "
            f"**{R['long_tr'][4][4]:.1f}% net** per event. The mirror trade (short it) makes "
            f"**+{R['short_tr'][1][3]:.1f}%** on paper but lives or dies on borrowing a hot new "
            "altcoin.\n"
            f"- **\"Gone before you can act?\" — Confirmed.** {R['windows'][1][3]:+.1f} pp of the pop "
            f"happens before the first tradable day; the listing day adds +{R['windows'][2][3]:.1f}%. "
            f"And since Coinbase's 2022 transparency roadmap, the pop has **halved** "
            f"(+{R['era']['pre_pop']:.0f}% → +{R['era']['post_pop']:.0f}%)."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The equity ancestor.** The same story existed in stocks — the S&P 500 inclusion pop "
            "([study 249](../../249-index-inclusion/)) — and died once everyone front-ran the "
            "front-runners. Crypto is speed-running the same decay: the pop already halved after "
            "Coinbase's 2022 roadmap policy.\n"
            "- **Why does the fade exceed the pop?** New listings arrive at peak attention. Attention "
            "fades, early holders take profit into the new demand, and the coin drifts back — the "
            "same \"buy the rumor, sell the news\" asymmetry as IPOs and index inclusions.\n"
            "- **The insider chapter is real law now.** *U.S. v. Wahi* (2022): a Coinbase employee "
            "tipped listings to relatives — the first crypto insider-trading conviction. The pop was "
            "literally worth going to prison for. That tells you where in the timeline the money is.\n\n"
            "*Think you can borrow freshly-listed alts at scale and harvest the fade? Show the borrow "
            "quotes — then we'll talk.*"
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
            "# The Exchange-Listing-Pop — a quantitative teardown 🔬\n"
            "### BTC-adjusted clustered CARs on 129 Coinbase listings · random-date placebo · "
            "month-cluster robustness · the 2022 roadmap natural experiment · cost × borrow sweep "
            "on the fade · a synthetic faithful-engine control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "\"Coinbase effect\" is folklore with a criminal case attached (*U.S. v. Wahi*, 2022) — "
            "so the job is to measure it honestly: venue-API event dates, market-adjusted CARs, "
            "cluster-aware inference, and the only executable version of the trade.\n\n"
            "> ⚠️ **Data note.** Day 0 = first Coinbase Exchange **USD-product** daily candle "
            "(Coinbase's own API, hardcoded table with a per-row venue price-agreement check). "
            "Prices: Binance USDT klines (the coins trade there ≥ 60 days before every event); "
            "abnormal = coin log return − BTC log return. **Survivorship named:** only "
            "Coinbase-served candle histories enter; five 2018–20 USDC-quoted books (BAT, ZEC, CVC, "
            "DNT, MANA) are excluded as mis-datable. Numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fingerprint"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` | Pop CAR[−5..0] **+{R['pop']['car']:.2f}%** (cluster "
            f"*t* = **{R['pop']['t']:+.2f}**, month-cluster *t* = {R['pop']['t_month']:+.2f}, placebo "
            f"*p* {R['pop']['p']}); fade CAR[+1..+30] **{R['fade']['car']:.2f}%** "
            f"(*t* = **{R['fade']['t']:+.2f}**). |\n"
            f"| **Tradability** | `FRAGILE` | Follower long: **{R['long_tr'][4][4]:.2f}% net** "
            f"(*t* = {R['long_tr'][4][5]:+.2f}). Fade short: **+{R['short_tr'][1][3]:.2f}% net** "
            f"(*t* = {R['short_tr'][1][4]:+.2f}) at 10%/yr borrow — **+{R['borrow'][2][1]:.2f}%** "
            f"(*t* = {R['borrow'][2][2]:+.2f}) at 100%/yr. Borrow-bound, short-only. |\n"
            f"| **Gone before you can act?** | `CONFIRMED` | Run-up [−5..−1] "
            f"**+{R['windows'][1][3]:.2f}%** (*t* = {R['windows'][1][5]:+.2f}) vs listing day "
            f"**+{R['windows'][2][3]:.2f}%** (*t* = {R['windows'][2][5]:+.2f}); roadmap era halved "
            f"the pop (Welch *t* = {R['era']['diff_welch']:+.2f}). |\n\n"
            "> 💡 In plain words: the legend is true, the hangover is bigger than the party, and the "
            "party is over before the doors open."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_{i,t}$ be coin $i$'s daily log return and $m_t$ BTC's. The abnormal return is "
            "$a_{i,t} = r_{i,t} - m_t$ (market model, $\\beta = 1$ — the standard crypto event-study "
            "adjustment) and the CAR over event offsets $[a..b]$ around listing day $\\tau_i$ is\n\n"
            "$$\\mathrm{CAR}_i(a,b) = \\sum_{k=a}^{b} a_{i,\\tau_i+k}.$$\n\n"
            "- **H₁ (pop).** $\\overline{\\mathrm{CAR}}(-5,0) > 0$ — the coin outruns BTC into the "
            "listing.\n"
            "- **H₂ (give-back).** $\\overline{\\mathrm{CAR}}(+1,+30) < 0$ — and gives it back after.\n"
            "- **H₃ (tradability).** A follower entering at the day-0 close (ONE lag) captures "
            "something net of costs — long the pop's continuation, or short the fade.\n\n"
            "Same-day listings (Coinbase announces several assets per post) are **one news draw**: "
            "the primary unit is the *listing-day cluster* (95 clusters for 129 events)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Three honesty problems sit on top of a naive event average: **(a) market timing** — "
            "2021 listings landed in a bull market, so raw returns overstate the event (hence BTC-"
            "adjustment); **(b) cross-sectional clustering** — multi-asset announcements are "
            "correlated draws (hence day-clusters, and a month-cluster robustness); **(c) the "
            "counterfactual** — maybe these coins outrun BTC around *any* date (hence a random-date "
            "placebo on the same coins). Tradability then has exactly one honest version: the "
            "**day-0 close** is the first fill a non-insider can get; every window before that is "
            "someone else's money."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Events.** {R['n_events']} listings 2019–2026 (hardcoded, per-row venue-ratio check; "
            f"{R['n_clusters']} same-day clusters; {R['n_pre']} events pre-roadmap / {R['n_post']} "
            "after 2022-04-28).\n"
            f"- **Tape.** Binance USDT daily klines, {R['days']:,} days × {R['n_coins']} coins + BTC, "
            f"{R['start']} → {R['end']} ({R['years']} yrs).\n"
            "- **Windows.** [−20..−6], [−5..−1], [0], [−5..0] (pop), [+1..+5], [+1..+30] (fade).\n"
            "- **Primary null.** One-sample *t* on per-cluster CARs; month-cluster and hit-rate "
            "robustness; Welch *t* + empirical *p* vs a same-coin random-date placebo (2,000 draws, "
            "20-seed stable — canonical numbers from `docs/results.md`).\n"
            "- **Costs.** 10/25/50 bps one-way per leg; shorts pay borrow 10%/yr (alt) / 3%/yr (BTC), "
            "sensitivity to 100%/yr.\n"
            "- **Control.** Deterministic synthetic world with plantable pop (−2..0) and fade "
            "(+1..+20); the null must stay quiet."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The event-time CAR path — the whole story in one curve\n\n"
            "Mean cumulative abnormal return by event day, with the per-event spread behind it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    M = st.event_matrix(ABN, EV, -20, 30)\n"
            "    paths = M.cumsum() * 100\n"
            "    mean_path = paths.mean(axis=1)\n"
            "    lo, hi = paths.quantile(.25, axis=1), paths.quantile(.75, axis=1)\n"
            "    fig, ax = plt.subplots(figsize=(9.8, 5.2))\n"
            "    ax.fill_between(paths.index, lo, hi, alpha=.18, color=GREY, label='interquartile range (events)')\n"
            "    ax.plot(mean_path.index, mean_path.values, lw=2.6, color=GREEN, label='mean CAR')\n"
            "    ax.axvline(0, ls='--', c=RED, lw=1.5, label='listing day (day 0)')\n"
            "    ax.axhline(0, c=GREY, lw=1)\n"
            "    ax.set_xlabel('event day'); ax.set_ylabel('cumulative abnormal return vs BTC (%)')\n"
            "    ax.set_title('129 Coinbase listings: the mountain peaks at the first tradable day')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'mean CAR at day 0: {mean_path.loc[0]:+.2f}%  at +30: {mean_path.loc[30]:+.2f}%  (n={M.shape[1]} events)')\n"
            "else:\n"
            "    print('cache missing - see R for canonical numbers'); print(R['windows'])"
        ),
        md(
            "### 4b · Windows, clustered *t*, placebo\n\n"
            "Cluster-level one-sample *t* per window; the placebo pits the event CARs against the "
            "same coins on random non-event dates (≥ 90 days away). The in-notebook placebo uses "
            "**300 draws for the picture**; canonical inference (2,000 draws, 20 seeds) is quoted "
            "from `docs/results.md`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    print('window                     mean CAR   cluster t')\n"
            "    for lab, a, b, *_ in R['windows']:\n"
            "        s = st.window_summary(ABN, EV, a, b)\n"
            "        print(f\"  {lab:<24} {s['mean_car']*100:+8.2f}%   {s['t']:+6.2f}   (n={s['n_clusters']} clusters)\")\n"
            "    pl = st.placebo_pvalue(ABN, EV, -5, 0, n_draws=300, seed=636, side='greater')\n"
            "    fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "    ax.hist(pl['draws'] * 100, bins=40, color=GREY, alpha=.85,\n"
            "            label='placebo: mean CAR of random pseudo-listing calendars (300 draws)')\n"
            "    ax.axvline(pl['obs'] * 100, c=GREEN, lw=2.5, label=f\"observed pop {pl['obs']*100:+.1f}%\")\n"
            "    ax.set_xlabel('mean CAR[-5..0] (%)'); ax.set_ylabel('frequency')\n"
            "    ax.set_title('The pop sits far outside the luck cloud')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f\"placebo (300 draws): p = {pl['p']:.4f}; canonical 2,000-draw, 20-seed: p {R['pop']['p']}\")\n"
            "else:\n"
            "    print(R['pop'], R['fade'])"
        ),
        md(
            f"> 💡 In plain words: random weeks on these same coins produce a mean \"pop\" near "
            f"**zero**; the observed **+{R['pop']['obs_event']:.1f}%** (event-level) is beyond every "
            f"placebo draw (canonical Welch *t* = {R['pop']['welch']:+.2f} vs "
            f"{R['pop']['n_pooled']:,} placebo windows, *p* {R['pop']['p']}; fade Welch "
            f"*t* = {R['fade']['welch']:+.2f}, *p* {R['fade']['p']}). Robustness: month-clustered "
            f"*t* = {R['pop']['t_month']:+.2f} / {R['fade']['t_month']:+.2f}, median cluster pop "
            f"+{R['pop']['med']:.1f}%, {R['pop']['share_pos']}% of clusters positive — not an "
            "outlier, not a 2021 artefact."
        ),
        md(
            "### 4c · The 2022 roadmap — a natural experiment on information leakage\n\n"
            "2022-04-28: after the DOJ front-running case, Coinbase began pre-announcing listings on "
            "a public roadmap (weeks of notice instead of 2 days). If the pop is announcement-driven "
            "information, front-loading the news should **flatten and spread** it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ev_pre = EV[EV['day0'] < data.ROADMAP_ERA]; ev_post = EV[EV['day0'] >= data.ROADMAP_ERA]\n"
            "    pp = st.window_summary(ABN, ev_pre, -5, 0); qq = st.window_summary(ABN, ev_post, -5, 0)\n"
            "    fp = st.window_summary(ABN, ev_pre, 1, 30); fq = st.window_summary(ABN, ev_post, 1, 30)\n"
            "    vals = [pp['mean_car']*100, qq['mean_car']*100, fp['mean_car']*100, fq['mean_car']*100]\n"
            "else:\n"
            "    vals = [R['era']['pre_pop'], R['era']['post_pop'], R['era']['pre_fade'], R['era']['post_fade']]\n"
            "labels = ['POP\\npre-roadmap', 'POP\\nroadmap era', 'FADE\\npre-roadmap', 'FADE\\nroadmap era']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "ax.bar(labels, vals, color=[GREEN, AMBER, RED, RED], width=.6)\n"
            "for i, v in enumerate(vals): ax.annotate(f'{v:+.1f}%', (i, v), ha='center', va='bottom' if v > 0 else 'top')\n"
            "ax.axhline(0, c=GREY, lw=1); ax.set_ylabel('mean CAR (%)')\n"
            "ax.set_title('Transparency halved the pop - and left the fade untouched')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('pop pre/post:', f\"{vals[0]:+.1f}% / {vals[1]:+.1f}%\", ' fade pre/post:', f\"{vals[2]:+.1f}% / {vals[3]:+.1f}%\")"
        ),
        md(
            f"> 💡 In plain words: pre-roadmap pop **+{R['era']['pre_pop']:.2f}%** "
            f"(*t* = {R['era']['pre_pop_t']:+.2f}) vs roadmap-era **+{R['era']['post_pop']:.2f}%** "
            f"(*t* = {R['era']['post_pop_t']:+.2f}) — difference Welch *t* = "
            f"**{R['era']['diff_welch']:+.2f}**. Publishing the news weeks early let it leak into a "
            f"long dribble ([−20..−6] is only +{R['windows'][0][3]:.1f}%, *t* = "
            f"{R['windows'][0][5]:.2f} — spread thin, it stops being measurable). The **fade didn't "
            f"budge** ({R['era']['pre_fade']:+.2f}% → {R['era']['post_fade']:+.2f}%): attention decay "
            "doesn't care about disclosure policy. Textbook McLean-Pontiff, in fast-forward."
        ),
        md(
            "### 4d · Tradability — the one executable version, costed\n\n"
            "Enter at the **day-0 close** (one lag), exit at +5/+30; hedged with BTC; one-way costs "
            "per leg; shorts pay borrow. The long rides the folklore; the short fades it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    print('LONG the listing (hedged short-BTC):')\n"
            "    for h in (5, 30):\n"
            "        for cb in (10., 25., 50.):\n"
            "            s = st.trade_summary(st.follower_trade(PX, EV, hold=h, cost_bps=cb, hedged=True))\n"
            "            print(f'  +{h:>2}d {cb:>4.0f}bps: net {s[\"net_pct\"]:+7.2f}%  t={s[\"t_net\"]:+5.2f}  hit {s[\"hit\"]*100:.0f}%')\n"
            "    print('SHORT the listing / long BTC (borrow sweep, 25 bps, +30d):')\n"
            "    for br in (0.10, 0.50, 1.00):\n"
            "        s = st.trade_summary(st.follower_trade(PX, EV, hold=30, cost_bps=25., hedged=True, side='short', borrow_alt=br))\n"
            "        print(f'  borrow {br*100:>4.0f}%/yr: net {s[\"net_pct\"]:+7.2f}%  t={s[\"t_net\"]:+5.2f}  hit {s[\"hit\"]*100:.0f}%')\n"
            "borrows = [b[0] for b in R['borrow']]; nets = [b[1] for b in R['borrow']]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.bar([f'{b}%/yr' for b in borrows], nets, color=[GREEN, AMBER, RED], width=.5)\n"
            "for i, v in enumerate(nets): ax.annotate(f'{v:+.1f}%', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c=GREY, lw=1)\n"
            "ax.set_xlabel('annualised borrow cost on the freshly-listed alt')\n"
            "ax.set_ylabel('net return per event (%)')\n"
            "ax.set_title('The fade pays until the borrow desk takes it away')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the *buy-the-news* trade loses **{R['long_tr'][4][4]:.1f}% net** "
            f"per event (*t* = {R['long_tr'][4][5]:+.2f}, hit rate {R['long_tr'][4][6]}%) — it is not "
            f"merely dead, it is a donation. The **fade** nets **+{R['short_tr'][1][3]:.1f}%** "
            f"(*t* = {R['short_tr'][1][4]:+.2f}, hit {R['short_tr'][1][5]}%) at 10%/yr borrow and "
            f"survives 50%/yr (+{R['borrow'][1][1]:.1f}%, *t* = {R['borrow'][1][2]:+.2f}) — but at "
            f"100%/yr it's noise (+{R['borrow'][2][1]:.1f}%, *t* = {R['borrow'][2][2]:+.2f}). "
            "Fresh listings are exactly where spot borrow is scarce, rationed and repriced against "
            "you — the constraint is the borrow desk, not the signal. Hence **FRAGILE**."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Deterministic synthetic world, staggered pseudo-listings, a plantable pop and fade. "
            "The null must stay quiet; the planted effect must be recovered at its true size."
        ),
        code(
            "res = []\n"
            "for pop, fade in [(0.0, 0.0), (0.15, 0.10)]:\n"
            "    spx, sev = data.synthetic_world(pop=pop, fade=fade, seed=636)\n"
            "    sabn = st.abnormal_returns(spx)\n"
            "    sp = st.window_summary(sabn, sev, -5, 0); sf = st.window_summary(sabn, sev, 1, 30)\n"
            "    res.append((pop, sp['mean_car']*100, sp['t'], sf['mean_car']*100, sf['t']))\n"
            "    print(f'planted pop={pop*100:5.1f}% fade={fade*100:5.1f}%: '\n"
            "          f\"CAR[-5..0]={sp['mean_car']*100:+6.2f}% (t{sp['t']:+6.2f})  \"\n"
            "          f\"CAR[+1..+30]={sf['mean_car']*100:+6.2f}% (t{sf['t']:+6.2f})\")\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "labels = ['null\\n(pop 0%)', 'planted\\n(pop +15%)']\n"
            "ax.bar(labels, [r[2] for r in res], color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i, r in enumerate(res): ax.annotate(f't={r[2]:.2f}', (i, r[2]), ha='center', va='bottom')\n"
            "ax.set_ylabel('cluster t on CAR[-5..0]')\n"
            "ax.set_title('Control: the null stays quiet; the planted pop lights up at true size')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: with nothing planted the machinery reads "
            f"*t* = {R['syn'][0][3]:+.2f} (it cannot conjure a pop from noise); a planted +15% pop is "
            f"recovered at +{R['syn'][1][2]:.2f}% (*t* = {R['syn'][1][3]:+.2f}). The real-tape "
            f"*t* = {R['pop']['t']:+.2f} is therefore the genuine article. *(Machinery proof only — "
            "never cited in support of the stamps.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — pop CAR[−5..0] **+{R['pop']['car']:.2f}%** (cluster "
            f"*t* = {R['pop']['t']:+.2f}, month-cluster {R['pop']['t_month']:+.2f}, placebo *p* "
            f"{R['pop']['p']} across 20 seeds) and fade CAR[+1..+30] **{R['fade']['car']:.2f}%** "
            f"(*t* = {R['fade']['t']:+.2f}, month-cluster {R['fade']['t_month']:+.2f}) on "
            f"{R['n_events']} events / {R['n_clusters']} clusters. Both halves of the folklore clear "
            "the bar. Survivorship named.\n"
            f"- **Tradability `FRAGILE`** — the follower long is **{R['long_tr'][4][4]:.2f}% net** "
            f"(*t* = {R['long_tr'][4][5]:+.2f}); the fade short is **+{R['short_tr'][1][3]:.2f}% net** "
            f"(*t* = {R['short_tr'][1][4]:+.2f}) at 10%/yr borrow, +{R['borrow'][1][1]:.2f}% at "
            f"50%/yr, **+{R['borrow'][2][1]:.2f}% (t = {R['borrow'][2][2]:+.2f})** at 100%/yr. "
            "Short-only, borrow-bound, per-event capacity tiny. Not INVESTABLE.\n"
            f"- **Gone before you can act? `CONFIRMED`** — [−5..−1] carries "
            f"+{R['windows'][1][3]:.2f} pp (*t* = {R['windows'][1][5]:+.2f}) vs the listing day's "
            f"+{R['windows'][2][3]:.2f}% (*t* = {R['windows'][2][5]:+.2f}); documented announcement→"
            "trade gaps are 2 days (Pro era) to weeks (roadmap era); the roadmap halved the pop "
            f"(Welch *t* = {R['era']['diff_welch']:+.2f})."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The decay is the lesson.** Announcement effects migrate up the information chain: "
            "blog-post era → 2-day pop at the announcement; roadmap era → half the pop, leaked over "
            "weeks; *Wahi* era → some of it traded before any public post. The equity ancestor "
            "([249-index-inclusion](../../249-index-inclusion/)) completed the same arc in ~20 years; "
            "crypto did it in three.\n"
            "- **The fade is an attention story, not a listing story.** It survives the policy change "
            "because it's driven by post-listing attention decay and early-holder distribution — "
            "which suggests it generalises to other attention spikes (new perp listings, trending "
            "pairs) and will persist until borrow supply on fresh listings industrialises.\n"
            "- **Refinements that would sharpen it:** per-event beta estimation (we fix β = 1), "
            "volume-conditioned fades, and an announcement-timestamped intraday version — the daily "
            "tape necessarily smears the first minutes after the blog post.\n\n"
            "*The reproducible core is offline and deterministic; canonical numbers live in "
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
