"""Generate the two narrative notebooks for Study 634 (US-Leads-the-World).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY +
overseas-index bars under ../_cache/ and otherwise quote the frozen headline numbers in
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SPY total-return +
# 4 price-only cash indices, 1997-01-02 -> 2026-06-30, 7,415-7,417 aligned pairs/market).
R = dict(
    start="1997-01-02", end="2026-06-30", years=29.5, asof="2026-06-30",
    fingerprint="c6ad53f6345c",
    # signal: ticker -> (label, cc slope, HAC t, R2 %, n)
    cc={"^N225": ("Nikkei 225 (Tokyo)", 0.561, 19.59, 20.0, 7415),
        "^GDAXI": ("DAX (Frankfurt)", 0.222, 9.21, 3.5, 7417),
        "^FTSE": ("FTSE 100 (London)", 0.230, 10.50, 6.2, 7417),
        "^AXJO": ("ASX 200 (Sydney)", 0.387, 11.18, 23.2, 7417)},
    basket=(0.350, 14.85, 19.5, 7417),
    placebo=(0.0070, 0.75, 50),                     # mean |slope|, mean |t|, seeds
    # mechanism: ticker -> (gap slope, gap t, gap R2%, oc slope, oc t, oc R2%, gap share %, live %)
    mech={"^N225": (0.368, 21.15, 38.2, 0.193, 6.66, 4.1, 66, 100),
          "^GDAXI": (0.158, 6.94, 8.6, 0.066, 4.11, 0.4, 71, 100),
          "^FTSE": (0.139, 7.56, 23.2, 0.147, 5.05, 2.9, 60, 12),
          "^AXJO": (0.194, 14.67, 27.0, 0.251, 16.53, 16.5, 50, 59)},
    # decay by era, GLOBAL-4: (label, slope, t, n)
    eras=[("1997-2003", 0.336, 17.56, 1760), ("2004-2010", 0.410, 15.88, 1763),
          ("2011-2017", 0.429, 10.99, 1761), ("2018-2026", 0.266, 4.49, 2133)],
    # |US| size quintiles, GLOBAL-4: (label, mean |US| %, slope, t)
    size=[("Q1", 0.09, 0.322, 1.83), ("Q2", 0.29, 0.259, 4.37), ("Q3", 0.57, 0.407, 10.46),
          ("Q4", 0.97, 0.380, 15.57), ("Q5", 2.18, 0.343, 12.51)],
    # pre/post 2010: ticker -> (pre slope, pre t, post slope, post t, t_diff)
    prepost={"^N225": (0.544, 16.2, 0.584, 11.5, -0.66),
             "^GDAXI": (0.247, 9.0, 0.190, 4.7, 1.16),
             "^FTSE": (0.289, 13.5, 0.154, 4.7, 3.42),
             "^AXJO": (0.411, 21.3, 0.356, 5.0, 0.74),
             "GLOBAL4": (0.373, 21.4, 0.321, 7.1, 1.06)},
    # feasible open-entry trade: ticker -> (gross bps, HAC t, n, net@5, net@10)
    feasible={"^N225": (9.5, 7.22, 7386, -0.5, -10.5),
              "^GDAXI": (5.0, 3.58, 7381, -5.0, -15.0),
              "^AXJO": (22.8, 16.40, 4392, 12.8, 2.8)},
    # phantom close-to-close backtest: ticker -> (gross bps, HAC t)
    phantom={"^N225": (48.2, 27.59), "^GDAXI": (20.9, 12.76),
             "^FTSE": (18.7, 14.06), "^AXJO": (35.3, 27.22)},
    # synthetic: (planted beta, cc slope, cc t) for SYN_A
    syn=[(0.0, 0.003, 0.25), (0.35, 0.354, 25.82)],
    stale={"^FTSE": 88, "^AXJO": 41},
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Survives_algo_era%3F: Confirmed](https://img.shields.io/badge/Survives_algo_era%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from us_leads_the_world import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    BARS = data.load_real()
    PAIRS = data.build_all_pairs(BARS)
    BK = data.basket_pairs(PAIRS)
else:
    BARS = PAIRS = BK = None
print("real tape cached:", HAVE_REAL,
      "| aligned pairs:", (0 if PAIRS is None else {k: len(v) for k, v in PAIRS.items()}))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# When America sneezes, does the world really catch a cold? 🌏\n"
            "### Today's Wall Street close predicts tomorrow's Tokyo — hugely, reliably, for 30 years. "
            "And you still can't make a dime on it.\n\n"
            + BADGES +
            "Here's the setup. New York closes at 4pm. **Three hours later, Tokyo opens.** Then Sydney "
            "trades, then Frankfurt and London the next morning — all *before* New York opens again. So "
            "every overseas market starts its day already knowing exactly what the US just did.\n\n"
            "The old traders' line — *\"America sneezes, the world catches a cold\"* — claims those "
            "markets don't just know it, they **follow** it: a green day in New York means a green "
            "morning in Tokyo. Unlike most folklore on this desk, this one has three decades of academic "
            "paper behind it. We put it on the tape: 29.5 years, four countries, and an honest attempt "
            "to actually **trade** it.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the gap decomposition and the cost "
            "math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Data note up front.** SPY is total-return; the four overseas indices are price-only "
            "levels. Yahoo's index *opens* are stale on many days (we flag and exclude them wherever an "
            "open matters). Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does tomorrow's Tokyo follow today's New York? | **Massively.** A +1% US day moves the "
            "next Tokyo session about **+0.56%** on average — and the same holds (smaller) in Sydney, "
            "Frankfurt and London. This is one of the strongest, cleanest effects on our whole bench. |\n"
            "| Hasn't 30 years of arbitrage killed it? | **No.** The 2018–2026 slice is as decisive as "
            "the 1990s. It *can't* be arbitraged away — and understanding why is the whole story. |\n"
            "| So… free money? | **None.** Two-thirds of the move happens **at the opening print** — a "
            "price that exists before any order can fill. What's left after the open is thinner than "
            "the cost of trading it. |\n"
            "| Can't I just buy the Japan ETF (EWJ) in New York? | **That's the trap.** EWJ trades "
            "*during US hours* — by your first chance to buy it, it has already moved with the US "
            "market, same day. The time zone that creates the prediction is what the wrapper removes. |\n\n"
            "> A rare verdict: the folklore is **completely true** and **completely untradable** at the "
            "same time."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The US is the world's market leader. Tokyo, Sydney, Frankfurt and London open after "
            "New York closes — and they open the way New York closed. Yesterday's S&P is this morning's "
            "Nikkei.\"*\n\n"
            "This isn't just folklore — it's **Hamao, Masulis & Ng (1990)**, **Becker, Finnerty & Gupta "
            "(1990)** and, in its modern form, **Rapach, Strauss & Zhou (2013, Journal of Finance)**: "
            "lagged US returns predict essentially every other country's market, and almost nothing "
            "predicts the US back. The mechanism is brutally simple: **the trading day itself is "
            "staggered around the globe**, so US news is always \"yesterday's news\" that the rest of "
            "the world hasn't traded yet.\n\n"
            "The interesting question was never *whether* the correlation exists — it's whether the "
            "**time-zone spillover survives honestly measured, and whether anyone can eat it**."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If this is real, every morning briefing that starts with \"US stocks rose overnight, "
            "expect Asian shares to open higher\" is *statistically justified* — not filler. And if it "
            "were tradable, it would be the simplest strategy on Earth: read the US close, buy Tokyo "
            "three hours later.\n\n"
            "The catch we're hunting: **where exactly** does the follow-through happen? If it lands in "
            "the *opening price itself*, then the market has priced the US news before you can click — "
            "prediction without profit. That single distinction (the **gap** vs the **rest of the "
            "day**) decides whether this is an edge or just a fact of geography."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{R['years']:.0f} years** of daily bars ({R['start']} → {R['end']}): **SPY** for "
            "the US, and the four big overseas indices — **Nikkei 225** (Tokyo), **ASX 200** (Sydney), "
            "**DAX** (Frankfurt), **FTSE 100** (London). For each US day *t* we find each market's "
            "**first session strictly after** that date (Tokyo opens ~7pm ET the same evening; Europe "
            "the next morning) and ask three things:\n\n"
            "1. **Follow-through.** Does the next overseas session move with the US day? (A slope: "
            "overseas-% per US-%.)\n"
            "2. **Where it lands.** Split each overseas day into the **opening gap** (last close → "
            "open, printed before anyone can trade) and the **open→close** part (the only piece a "
            "trader can own).\n"
            "3. **The only real trade.** Follow the US sign at the next overseas *open*, sell at its "
            "close, pay costs — and see what's left."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the follow-through.** How much does each market's next session move per 1% of "
            "US day?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    vals = {tk: st.ols_hac(p['us'], p['cc'])['slope'] for tk, p in PAIRS.items()}\n"
            "else:\n"
            "    vals = {tk: v[1] for tk, v in R['cc'].items()}\n"
            "labels = [R['cc'][tk][0].split(' (')[1][:-1] + '\\n' + R['cc'][tk][0].split(' (')[0] for tk in vals]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.6))\n"
            "ax.bar(labels, [v*100 for v in vals.values()], color=GREEN, width=.6)\n"
            "for i, v in enumerate(vals.values()):\n"
            "    ax.annotate(f'+{v*100:.0f} bps', (i, v*100), ha='center', va='bottom', fontweight='bold')\n"
            "ax.set_ylabel('next-session move per +1% US day (bps)')\n"
            "ax.set_title('America sneezes: tomorrow\\'s session, per 1% of today\\'s US move')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({k: f'+{v:.3f}' for k, v in vals.items()})"
        ),
        md(
            f"A +1% US day is followed, on average, by **+{R['cc']['^N225'][1]*100:.0f} bps in Tokyo**, "
            f"+{R['cc']['^AXJO'][1]*100:.0f} in Sydney, +{R['cc']['^FTSE'][1]*100:.0f} in London and "
            f"+{R['cc']['^GDAXI'][1]*100:.0f} in Frankfurt. The quants notebook shows these are about as "
            "far from luck as market statistics ever get (odds-of-fluke *t*-scores of **9 to 20**; the "
            "desk's bar is 2). And it hasn't faded: the 2018–2026 slice alone still clears the bar "
            f"easily (t = {R['eras'][3][2]:.1f}).\n\n"
            "So the folklore is *true*. Now the important chart."
        ),
        md(
            "**Where does the move happen?** Each overseas day = the **opening gap** (printed before "
            "anyone can trade) + the **rest of the day** (what a trader can actually own). If the gap "
            "eats the spillover, the prediction is worthless."
        ),
        code(
            "tks = ['^N225', '^GDAXI']   # the two markets with honestly recorded opens\n"
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for tk in tks:\n"
            "        s = st.market_summary(PAIRS[tk])\n"
            "        rows.append((s['gap']['slope'], s['oc']['slope']))\n"
            "else:\n"
            "    rows = [(R['mech'][tk][0], R['mech'][tk][3]) for tk in tks]\n"
            "x = np.arange(len(tks))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.6))\n"
            "ax.bar(x-.18, [r[0]*100 for r in rows], .34, color=GREY, label='opening gap (untradable print)')\n"
            "ax.bar(x+.18, [r[1]*100 for r in rows], .34, color=GREEN, label='open->close (what you can own)')\n"
            "for i, r in enumerate(rows):\n"
            "    ax.annotate(f'{r[0]*100:+.0f}', (i-.18, r[0]*100), ha='center', va='bottom')\n"
            "    ax.annotate(f'{r[1]*100:+.0f}', (i+.18, r[1]*100), ha='center', va='bottom')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['Tokyo (Nikkei)', 'Frankfurt (DAX)'])\n"
            "ax.set_ylabel('bps of next session per +1% US day')\n"
            "ax.set_title('The spillover is delivered AT THE OPEN - before you can trade')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"There's the catch. In Tokyo, **{R['mech']['^N225'][6]}%** of the follow-through is in the "
            "opening print itself (the gap tracks the US day so tightly that knowing yesterday's S&P "
            f"explains **{R['mech']['^N225'][2]:.0f}%** of the gap's variance — the Nikkei open is "
            "almost a *formula* applied to the US close). In Frankfurt it's "
            f"**{R['mech']['^GDAXI'][6]}%**. The market you're allowed to touch — open to close — keeps "
            "only a sliver."
        ),
        md(
            "**The only trade you can actually do:** know the US close (4pm ET), then buy (or short) "
            "the overseas market **at its open**, sell at its close, every day, and pay costs."
        ),
        code(
            "tks = ['^N225', '^GDAXI']\n"
            "if HAVE_REAL:\n"
            "    g5 = [st.feasible_trade(PAIRS[tk], 5.0) for tk in tks]\n"
            "    gross = [r['gross_bps'] for r in g5]; net5 = [r['net_bps'] for r in g5]\n"
            "else:\n"
            "    gross = [R['feasible'][tk][0] for tk in tks]; net5 = [R['feasible'][tk][3] for tk in tks]\n"
            "x = np.arange(len(tks))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.5))\n"
            "ax.bar(x-.18, gross, .34, color=AMBER, label='gross (before costs)')\n"
            "ax.bar(x+.18, net5, .34, color=RED, label='net of 5 bps each way')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i in range(len(tks)):\n"
            "    ax.annotate(f'{gross[i]:+.1f}', (i-.18, gross[i]), ha='center', va='bottom')\n"
            "    ax.annotate(f'{net5[i]:+.1f}', (i+.18, min(net5[i],0)), ha='center', va='top')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['Tokyo (Nikkei)', 'Frankfurt (DAX)'])\n"
            "ax.set_ylabel('bps per day')\n"
            "ax.set_title('Follow the US sign at the next open: dead at 5 bps each way')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('gross bps/day:', [round(v,1) for v in gross], ' net@5bps:', [round(v,1) for v in net5])"
        ),
        md(
            f"Gross, there *is* a residue after the open (**+{R['feasible']['^N225'][0]:.1f} bps/day** "
            "in Tokyo — markets don't absorb the US news at the bell with 100% precision). But it's a "
            "daily-turnover trade: at a modest **5 bps each way** it nets "
            f"**{R['feasible']['^N225'][3]:+.1f} bps/day** in Tokyo and "
            f"**{R['feasible']['^GDAXI'][3]:+.1f}** in Frankfurt. Nothing left.\n\n"
            "And the two escape hatches are locked too:\n"
            "- **The Sydney line that *looks* profitable** in our table is an accounting ghost — the ASX "
            "\"open\" is a 10-minute staggered auction, so the printed open still contains yesterday's "
            "prices for stocks that haven't opened yet. The \"profit\" is repricing you could never buy.\n"
            "- **The US-listed wrappers (EWJ for Japan, EWG for Germany)** trade during *US* hours. By "
            "the time you can touch them, they've already moved with the US market — same day, no "
            "prediction left."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** Tomorrow's overseas session follows today's US close with slopes of "
            f"+0.22 to +0.56 (Tokyo the strongest), *t*-scores of 9–20, stable for {R['years']:.0f} "
            "years. As real as market statistics get.\n"
            "- **Tradability — Mirage.** Two-thirds of the move is in the opening print you cannot "
            "trade; the open-to-close residue dies at 5 bps each way; the accessible wrappers reprice "
            "contemporaneously. Prediction without a vehicle.\n"
            "- **Survives the algo era? — Confirmed.** Post-2010 is as strong as pre-2010 (Tokyo got "
            "*stronger*). It persists precisely because there's no P&L in it — geography isn't an "
            "inefficiency."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Why doesn't it decay like other anomalies?** Published *tradable* edges get eaten "
            "(McLean & Pontiff 2016). This one is *information flow*: the profit-shaped part (the gap) "
            "is captured by the opening auction itself. What's published can't decay if nobody can "
            "trade it.\n"
            "- **The one that did decay is the exception that proves it:** London — the market whose "
            "session most overlaps New York's, where index futures arbitrage is densest — halved its "
            "slope after 2010. Overlap creates arbitrage; time-zone separation protects the pattern.\n"
            "- **Build your own.** Swap in Korea, India or Brazil; or measure the gap slope against US "
            "*futures* moves during foreign hours. The engine (`us_leads_the_world/`) aligns any pair "
            "of calendars.\n\n"
            "*Think you've found the after-the-open residue net of real costs — index futures at 1 bp, "
            "borrow paid on the short nights? Show the net line clearing zero with an honest open. "
            "Then we'll talk.*"
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
            "# US-Leads-the-World — a quantitative teardown 🔬\n"
            "### Per-market predictive HAC slopes · overnight-gap vs open→close decomposition with "
            "stale-open forensics · era decay + |move|-size conditioning · feasible-vs-phantom trade "
            "accounting · a pre/post-2010 difference test · a planted-beta synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "cross-timezone spillover (Hamao-Masulis-Ng 1990; Rapach-Strauss-Zhou 2013) is a documented "
            "effect that *should* be real — so the job is an honest measurement of (a) how strong, "
            "(b) **where in the session** it lands, and (c) whether one basis point of it is "
            "capturable.\n\n"
            "> ⚠️ **Data note.** SPY total-return adjusted; ^N225/^GDAXI/^FTSE/^AXJO **price-only** "
            "index levels. Yahoo's daily index **opens are stale** (open = previous close) on "
            f"**{R['stale']['^FTSE']}%** of ^FTSE days and **{R['stale']['^AXJO']}%** of ^AXJO days — "
            "every open-based leg runs on live-open days only, with the share quoted. Alignment gives "
            "exactly **one execution lag** by construction (the predictor is public before the target "
            "session opens). Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (as-of " + R['asof'] + ", fingerprint `"
            + R['fingerprint'] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` | Next-session slopes: Tokyo **+{R['cc']['^N225'][1]:.3f}** (HAC "
            f"**t = +{R['cc']['^N225'][2]:.2f}**), Sydney +{R['cc']['^AXJO'][1]:.3f} "
            f"(+{R['cc']['^AXJO'][2]:.2f}), London +{R['cc']['^FTSE'][1]:.3f} "
            f"(+{R['cc']['^FTSE'][2]:.2f}), Frankfurt +{R['cc']['^GDAXI'][1]:.3f} "
            f"(+{R['cc']['^GDAXI'][2]:.2f}); GLOBAL-4 **+{R['basket'][0]:.3f}** "
            f"(**t = +{R['basket'][1]:.2f}**, R² = {R['basket'][2]:.1f}%). 50-seed shuffle placebo "
            f"mean \\|t\\| = {R['placebo'][1]:.2f}. |\n"
            f"| **Tradability** | `MIRAGE` | Gap delivers 50–71% of the slope before the first "
            f"tradable print; the feasible open-entry trade nets **{R['feasible']['^N225'][3]:+.1f}** "
            f"(Tokyo) and **{R['feasible']['^GDAXI'][3]:+.1f}** (Frankfurt) bps/day at 5 bps one-way; "
            "the ^AXJO 'survivor' rests on a non-tradable staggered-auction open; EWJ/EWG reprice "
            "contemporaneously in US hours. |\n"
            f"| **Survives the algo era?** | `CONFIRMED` | GLOBAL-4 post-2010 slope "
            f"**+{R['prepost']['GLOBAL4'][2]:.3f}** (t = +{R['prepost']['GLOBAL4'][3]:.1f}), pre/post "
            f"t_diff = +{R['prepost']['GLOBAL4'][4]:.2f} (ns); Tokyo rose. Only London decayed "
            f"(t_diff = +{R['prepost']['^FTSE'][4]:.2f}). |\n\n"
            "> 💡 In plain words: the strongest *t* on the bench, attached to zero capturable dollars — "
            "geography is a prediction machine, not a P&L machine."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{US}_t$ be SPY's day-$t$ close-to-close return (public 16:00 ET) and $r^{F}_{t+}$ "
            "the return of foreign market $F$'s **first session strictly after** date $t$ (Tokyo opens "
            "~19:00 ET day $t$; Frankfurt/London ~03:00–04:00 ET day $t{+}1$). The spillover "
            "regression is\n\n"
            "$$r^{F}_{t+} = \\alpha + \\beta\\, r^{US}_t + \\varepsilon_{t+},$$\n\n"
            "with Newey-West (HAC) inference. Decompose $r^{F}_{t+}$ into the **overnight gap** "
            "$g_{t+}$ (prev close → open) and the **intraday leg** $o_{t+}$ (open → close):\n\n"
            "- **H₁ (spillover).** $\\beta > 0$, robustly, per market and pooled.\n"
            "- **H₂ (mechanism).** The gap carries most of $\\beta$ — the open impounds the US day.\n"
            "- **H₃ (tradability).** $\\mathbb{E}[\\mathrm{sign}(r^{US}_t)\\, o_{t+}]$ survives costs "
            "(the only executable version — the gap prints before any order can fill).\n"
            "- **H₄ (persistence).** $\\beta_{\\text{post-2010}}$ is not significantly below "
            "$\\beta_{\\text{pre}}$.\n\n"
            "We find **H₁ decisively supported** (t = 9–20), **H₂ confirmed** (gap share 50–71%, gap "
            "R² up to 38%), **H₃ rejected** (net ≤ 0 at 5 bps one-way on honestly-opened markets), "
            "**H₄ confirmed** (only London decayed). Distinct from "
            "[379-etf-lead-lag](../../379-etf-lead-lag/README.md): that was *same-timezone* intra-US "
            "lead-lag (None) — here the sessions genuinely do not overlap, so a predictive link can "
            "and does exist."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The **time-zone alignment is the entire identification**. Because Tokyo's session lies "
            "strictly inside the window between two US closes, $r^{US}_t$ is in the information set of "
            "every trade of session $t{+}$ — the regression is predictive by construction, with no "
            "overlap contamination on the Asian legs (European sessions overlap the *next* US morning; "
            "the predictor is still strictly prior, only the error term picks up contemporaneous US "
            "noise, which HAC absorbs).\n\n"
            "The gap/intraday split then decides everything tradable. A slope living in the gap is "
            "**an opening-auction repricing** — the textbook reason a documented, published, "
            "35-year-old predictability can persist: there is no counterfactual P&L for arbitrageurs "
            "to compete over. The desk's rule: *a real signal is not an edge until a vehicle exists*."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** yfinance daily Open+Close: SPY + ^N225/^GDAXI/^FTSE/^AXJO, {R['start']} → "
            f"{R['end']} ({R['years']} yrs), 7,415–7,417 aligned pairs/market. SPY total-return; "
            "indices price-only (labeled).\n"
            "- **Alignment.** US day $t$ → first foreign session strictly after date $t$ "
            "(`searchsorted`, ≤ 7 calendar days). One execution lag by construction.\n"
            "- **Inference.** OLS slope with Newey-West SE (Bartlett, `floor(4(n/100)^{2/9})` lags) on "
            "cc / gap / oc legs; HAC mean-t on strategy returns; pre/post difference t with SEs in "
            "quadrature.\n"
            "- **Stale-open guard.** A day's open is *live* iff it differs from the prior close "
            "(rel. tol. 1e-6). Gap/oc/trade legs use live-open days only "
            f"(^FTSE {R['stale']['^FTSE']}% stale, ^AXJO {R['stale']['^AXJO']}%).\n"
            "- **Placebo.** Shuffle the US predictor, **50 seeds** (house rule ≥ 20), report mean "
            "|slope| and |t|.\n"
            "- **Costs.** Feasible trade = sign(US) × next oc, one round trip/day, 5/10 bps one-way.\n"
            "- **Positive control.** Synthetic world with planted gap-beta ∈ {0, 0.35}: the pipeline "
            "must recover exactly the plant, locate it in the gap, and find nothing at 0."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The spillover slopes, per market + basket\n\n"
            "Close-to-close next session on US day $t$; HAC t against the desk's t = 2 bar. The "
            "GLOBAL-4 line is the equal-weight basket of the four next-session returns."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = {tk: st.ols_hac(p['us'], p['cc']) for tk, p in PAIRS.items()}\n"
            "    rb = st.ols_hac(BK['us'], BK['basket'])\n"
            "    rows = [(tk, r['slope'], r['t']) for tk, r in res.items()] + [('GLOBAL-4', rb['slope'], rb['t'])]\n"
            "else:\n"
            "    rows = [(tk, v[1], v[2]) for tk, v in R['cc'].items()] + [('GLOBAL-4', R['basket'][0], R['basket'][1])]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))\n"
            "names = [r[0] for r in rows]\n"
            "a1.bar(names, [r[1] for r in rows], color=[GREEN]*4+[AMBER], width=.6)\n"
            "for i, r in enumerate(rows): a1.annotate(f'{r[1]:+.3f}', (i, r[1]), ha='center', va='bottom', fontsize=9)\n"
            "a1.set_ylabel('slope (next session per 1.0 of US day)'); a1.set_title('Spillover slope')\n"
            "a1.tick_params(axis='x', rotation=30)\n"
            "a2.bar(names, [r[2] for r in rows], color=[GREEN]*4+[AMBER], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i, r in enumerate(rows): a2.annotate(f'{r[2]:.1f}', (i, r[2]), ha='center', va='bottom', fontsize=9)\n"
            "a2.set_ylabel('Newey-West HAC t'); a2.set_title('...and its HAC t'); a2.legend()\n"
            "a2.tick_params(axis='x', rotation=30)\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in rows: print(f'{r[0]:9s} slope={r[1]:+.3f}  HAC t={r[2]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: every market clears the t = 2 bar by a factor of **4 to 10** — Tokyo "
            f"at **t = +{R['cc']['^N225'][2]:.1f}** with R² = {R['cc']['^N225'][3]:.0f}% is as strong as "
            "daily-frequency return prediction gets. The 50-seed shuffle placebo (next cell of "
            f"`verify.py`) sits at mean |slope| = {R['placebo'][0]:.4f}, |t| = {R['placebo'][1]:.2f} — "
            "the *time link* carries everything."
        ),
        md(
            "### 4b · Mechanism — the gap eats the slope\n\n"
            "Per market, the cc slope split into its gap and open→close components (live-open days "
            "only; the stale share is on the chart). The gap regression's R² is the punchline."
        ),
        code(
            "tks = list(R['mech'].keys())\n"
            "if HAVE_REAL:\n"
            "    mech = {}\n"
            "    for tk in tks:\n"
            "        s = st.market_summary(PAIRS[tk])\n"
            "        mech[tk] = (s['gap']['slope'], s['oc']['slope'], s['live_share']*100, s['gap']['r2']*100)\n"
            "else:\n"
            "    mech = {tk: (v[0], v[3], v[7], v[2]) for tk, v in R['mech'].items()}\n"
            "x = np.arange(len(tks))\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.8))\n"
            "ax.bar(x-.18, [mech[t][0] for t in tks], .34, color=GREY, label='overnight gap (untradable print)')\n"
            "ax.bar(x+.18, [mech[t][1] for t in tks], .34, color=GREEN, label='open->close (ownable leg)')\n"
            "for i, t in enumerate(tks):\n"
            "    ax.annotate(f'{mech[t][0]:+.2f}', (i-.18, mech[t][0]), ha='center', va='bottom', fontsize=9)\n"
            "    ax.annotate(f'{mech[t][1]:+.2f}', (i+.18, mech[t][1]), ha='center', va='bottom', fontsize=9)\n"
            "    ax.annotate(f'live opens {mech[t][2]:.0f}%', (i, -0.04), ha='center', fontsize=8, color=RED)\n"
            "ax.set_xticks(x); ax.set_xticklabels(tks); ax.set_ylim(-0.08, 0.45)\n"
            "ax.set_ylabel('slope component'); ax.legend()\n"
            "ax.set_title('Where the spillover lands: mostly in the opening gap')\n"
            "plt.tight_layout(); plt.show()\n"
            "for t in tks: print(f'{t:7s} gap={mech[t][0]:+.3f} (R2 {mech[t][3]:.0f}%)  oc={mech[t][1]:+.3f}  live={mech[t][2]:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: Tokyo's opening gap alone has slope +{R['mech']['^N225'][0]:.3f} with "
            f"**R² = {R['mech']['^N225'][2]:.0f}%** — yesterday's S&P *is* this morning's Nikkei open, "
            f"almost mechanically. The ownable open→close legs keep slopes of only "
            f"+{R['mech']['^GDAXI'][3]:.3f}–+{R['mech']['^N225'][3]:.3f} (R² 0.4–4%) on honestly-opened "
            "markets. **Forensic note:** ^AXJO's fat oc slope "
            f"(+{R['mech']['^AXJO'][3]:.3f}, R² {R['mech']['^AXJO'][5]:.1f}%) is the staggered-auction "
            "artefact — the printed ASX open still contains unopened constituents at stale prices, so "
            "part of the 'intraday' leg is pre-open repricing. ^FTSE's split rests on 12% of days and "
            "is decorative."
        ),
        md(
            "### 4c · Decay and conditioning\n\n"
            "GLOBAL-4 slope by era, and by |US-move| quintile — does the spillover fade, and does it "
            "need a big sneeze?"
        ),
        code(
            "eras = [('1997-2003','1997','2003'),('2004-2010','2004','2010'),\n"
            "        ('2011-2017','2011','2017'),('2018-2026','2018','2026')]\n"
            "if HAVE_REAL:\n"
            "    bk_p = BK.rename(columns={'basket':'cc'})\n"
            "    er = [(r['label'], r['slope'], r['t']) for r in st.era_slopes(bk_p, eras)]\n"
            "    sz = [(r['label'], r['mean_abs_us_pct'], r['slope'], r['t']) for r in st.size_conditional(BK)]\n"
            "else:\n"
            "    er = [(e[0], e[1], e[2]) for e in R['eras']]\n"
            "    sz = R['size']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))\n"
            "a1.bar([e[0] for e in er], [e[1] for e in er], color=AMBER, width=.6)\n"
            "for i, e in enumerate(er): a1.annotate(f'{e[1]:+.2f}\\nt={e[2]:.1f}', (i, e[1]), ha='center', va='bottom', fontsize=9)\n"
            "a1.set_ylim(0, .55); a1.set_title('GLOBAL-4 slope by era'); a1.tick_params(axis='x', rotation=20)\n"
            "a2.bar([f'{s[0]}\\n({s[1]:.2f}%)' for s in sz], [s[2] for s in sz], color=GREEN, width=.6)\n"
            "for i, s in enumerate(sz): a2.annotate(f'{s[2]:+.2f}\\nt={s[3]:.1f}', (i, s[2]), ha='center', va='bottom', fontsize=9)\n"
            "a2.set_ylim(0, .55); a2.set_title('GLOBAL-4 slope by |US move| quintile (mean |US|)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('eras:', [(e[0], round(e[1],3), round(e[2],2)) for e in er])\n"
            "print('size:', [(s[0], round(s[2],3), round(s[3],2)) for s in sz])"
        ),
        md(
            f"> 💡 In plain words: no monotone decay — the slope wanders {R['eras'][0][1]:.2f} → "
            f"{R['eras'][2][1]:.2f} → {R['eras'][3][1]:.2f} and the weakest era still posts "
            f"**t = {R['eras'][3][2]:.2f}**. And it is **proportional**: the slope is flat "
            f"(~{min(s[2] for s in R['size']):.2f}–{max(s[2] for s in R['size']):.2f}) across move "
            "sizes — the world reprices whatever the US did, big or small (only the tiniest-move "
            "quintile lacks the signal-to-noise to certify it)."
        ),
        md(
            "### 4d · Tradability — phantom vs feasible, then costs\n\n"
            "The **phantom** backtest (sign(US) held close-to-close — requires entering at a foreign "
            "close that prints *before* the US close exists) vs the **feasible** one (enter at the "
            "next open, own only open→close), gross and net."
        ),
        code(
            "tks = ['^N225', '^GDAXI', '^AXJO']\n"
            "if HAVE_REAL:\n"
            "    ph = [st.phantom_trade(PAIRS[tk])['gross_bps'] for tk in tks]\n"
            "    fs = [st.feasible_trade(PAIRS[tk], 5.0) for tk in tks]\n"
            "    gr = [r['gross_bps'] for r in fs]; n5 = [r['net_bps'] for r in fs]\n"
            "else:\n"
            "    ph = [R['phantom'][tk][0] for tk in tks]\n"
            "    gr = [R['feasible'][tk][0] for tk in tks]; n5 = [R['feasible'][tk][3] for tk in tks]\n"
            "x = np.arange(len(tks))\n"
            "fig, ax = plt.subplots(figsize=(9.8, 4.8))\n"
            "ax.bar(x-.28, ph, .26, color=GREY, label='PHANTOM: close-to-close (needs a time machine)')\n"
            "ax.bar(x, gr, .26, color=AMBER, label='feasible: open-entry, gross')\n"
            "ax.bar(x+.28, n5, .26, color=RED, label='feasible, net @5 bps one-way')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i in range(len(tks)):\n"
            "    ax.annotate(f'{ph[i]:+.0f}', (i-.28, ph[i]), ha='center', va='bottom', fontsize=9)\n"
            "    ax.annotate(f'{gr[i]:+.1f}', (i, gr[i]), ha='center', va='bottom', fontsize=9)\n"
            "    ax.annotate(f'{n5[i]:+.1f}', (i+.28, min(n5[i],0)), ha='center', va='top', fontsize=9)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['Tokyo ^N225','Frankfurt ^GDAXI','Sydney ^AXJO*'])\n"
            "ax.set_ylabel('bps per day'); ax.legend(fontsize=9)\n"
            "ax.set_title('The paper edge is a time machine; the feasible edge dies at 5 bps')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('* ^AXJO net line rests on a staggered-auction open that is not a tradable price')"
        ),
        md(
            f"> 💡 In plain words: the naive backtest books **+{R['phantom']['^N225'][0]:.0f} bps/day** "
            "in Tokyo (t = +27.6!) — by trading at 01:00 ET on information released at 16:00 ET. The "
            f"executable version keeps **+{R['feasible']['^N225'][0]:.1f} bps/day gross** (a real "
            f"open-auction underreaction residue, HAC t = +{R['feasible']['^N225'][1]:.2f}) and "
            f"**{R['feasible']['^N225'][3]:+.1f} bps/day net** at 5 bps one-way — dead. The ^AXJO "
            f"'survivor' (+{R['feasible']['^AXJO'][3]:.1f} net) is credited with pre-open repricing "
            "inside a 10-minute staggered auction — not a fillable price. And the wrappers (EWJ/EWG) "
            "trade in US hours, where the spillover is contemporaneous: **no vehicle, no edge**."
        ),
        md(
            "### 4e · Third axis — pre/post-2010, with a t on the difference\n\n"
            "Slopes either side of 2010-01-01 (the algo/HFT era divide), HAC SEs combined in "
            "quadrature for the difference test."
        ),
        code(
            "tks = ['^N225', '^GDAXI', '^FTSE', '^AXJO']\n"
            "if HAVE_REAL:\n"
            "    pp = {tk: st.prepost(PAIRS[tk]) for tk in tks}\n"
            "    rows = [(tk, pp[tk]['pre']['slope'], pp[tk]['post']['slope'], pp[tk]['t_diff']) for tk in tks]\n"
            "else:\n"
            "    rows = [(tk, R['prepost'][tk][0], R['prepost'][tk][2], R['prepost'][tk][4]) for tk in tks]\n"
            "x = np.arange(len(rows))\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.6))\n"
            "ax.bar(x-.18, [r[1] for r in rows], .34, color=GREY, label='pre-2010')\n"
            "ax.bar(x+.18, [r[2] for r in rows], .34, color=GREEN, label='post-2010')\n"
            "for i, r in enumerate(rows):\n"
            "    sig = ' *' if abs(r[3]) > 2 else ''\n"
            "    ax.annotate(f't_diff={r[3]:+.2f}{sig}', (i, max(r[1], r[2])+.015), ha='center', fontsize=9)\n"
            "ax.set_xticks(x); ax.set_xticklabels(tks); ax.set_ylabel('cc slope')\n"
            "ax.set_title('The algo era did not kill it - only London decayed significantly')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for r in rows: print(f'{r[0]:7s} pre={r[1]:+.3f} post={r[2]:+.3f} t_diff={r[3]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: three of four markets show **no significant decay** — Tokyo's slope "
            f"*rose* ({R['prepost']['^N225'][0]:+.3f} → {R['prepost']['^N225'][2]:+.3f}). The basket "
            f"holds +{R['prepost']['GLOBAL4'][2]:.3f} at t = +{R['prepost']['GLOBAL4'][3]:.1f} post-2010 "
            f"(t_diff = +{R['prepost']['GLOBAL4'][4]:.2f}, ns). The one decayer is **London** "
            f"(t_diff = +{R['prepost']['^FTSE'][4]:.2f}) — the session that most overlaps New York, "
            "where futures arbitrage can actually reach the pattern. Time-zone separation is the moat. "
            "**CONFIRMED.**"
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "A deterministic synthetic world where US day-t feeds the next foreign **gap** with a "
            "planted beta (and the intraday leg is clean noise): the pipeline must recover the plant, "
            "locate it in the gap, and find nothing at beta = 0."
        ),
        code(
            "res = []\n"
            "for beta in (0.0, 0.35):\n"
            "    syn = data.synthetic_world(beta=beta, seed=634)\n"
            "    p = syn['SYN_A']\n"
            "    s = st.market_summary(p)\n"
            "    res.append((beta, s['cc']['slope'], s['cc']['t'], s['gap']['slope'], s['oc']['slope']))\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "labels = [f'planted beta = {r[0]:.2f}' for r in res]\n"
            "ax.bar(np.arange(2)-.15, [r[1] for r in res], .3, color=[GREY, GREEN], label='recovered cc slope')\n"
            "ax.bar(np.arange(2)+.15, [r[3] for r in res], .3, color=[GREY, AMBER], label='recovered gap slope')\n"
            "ax.axhline(0.35, ls='--', c=RED, label='the plant (0.35)')\n"
            "for i, r in enumerate(res): ax.annotate(f'{r[1]:+.3f}\\n(t={r[2]:+.1f})', (i-.15, r[1]), ha='center', va='bottom', fontsize=9)\n"
            "ax.set_xticks(np.arange(2)); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('recovered slope'); ax.legend(fontsize=9)\n"
            "ax.set_title('Control: finds exactly the plant, in the right leg, and nothing at zero')\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in res: print(f'beta={r[0]:.2f}: cc={r[1]:+.3f} (t={r[2]:+.2f})  gap={r[3]:+.3f}  oc={r[4]:+.3f}')"
        ),
        md(
            f"> 💡 In plain words: at beta = 0 the detector reads {R['syn'][0][1]:+.3f} "
            f"(t = {R['syn'][0][2]:+.2f}) — it cannot manufacture a spillover; at beta = 0.35 it reads "
            f"{R['syn'][1][1]:+.3f} (t = +{R['syn'][1][2]:.1f}), pinned to the gap leg with the intraday "
            "leg flat. The machinery is faithful, so the real-tape slopes are the genuine article. *(A "
            "faithful-engine / power check only — never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — per-market next-session slopes +{R['cc']['^GDAXI'][1]:.3f} to "
            f"+{R['cc']['^N225'][1]:.3f} with HAC t = +{R['cc']['^GDAXI'][2]:.1f} to "
            f"+{R['cc']['^N225'][2]:.1f}; GLOBAL-4 **+{R['basket'][0]:.3f}** at "
            f"**t = +{R['basket'][1]:.2f}** (R² {R['basket'][2]:.1f}%); shuffle placebo |t| ≈ "
            f"{R['placebo'][1]:.2f}; flat across move sizes; every era clears the bar. No survivorship "
            "issue of consequence (four continuously computed national indices).\n"
            f"- **Tradability `MIRAGE`** — 50–71% of the slope is an opening-print repricing; the "
            f"feasible open-entry trade nets {R['feasible']['^N225'][3]:+.1f} (Tokyo) / "
            f"{R['feasible']['^GDAXI'][3]:+.1f} (Frankfurt) bps/day at 5 bps one-way; the ^AXJO "
            "survivor line is a staggered-auction accounting ghost; EWJ/EWG reprice contemporaneously "
            "in US hours. Prediction without a vehicle.\n"
            f"- **Survives the algo era? `CONFIRMED`** — post-2010 GLOBAL-4 slope "
            f"+{R['prepost']['GLOBAL4'][2]:.3f} (t = +{R['prepost']['GLOBAL4'][3]:.1f}), difference "
            f"insignificant; Tokyo strengthened; only London (the overlapping session) decayed "
            f"(t_diff = +{R['prepost']['^FTSE'][4]:.2f})."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The McLean-Pontiff exemption.** Published anomalies decay because arbitrage capital "
            "attacks them. This spillover is *information flow priced at an auction* — the tradable "
            "residue is already ~zero, so publication has nothing to destroy. The London decay is the "
            "exception that proves it: where sessions overlap, futures arbitrage reaches the pattern.\n"
            "- **The professional version exists — at a different cost point.** Nikkei futures on CME "
            "trade nearly 24h at ~1 bp; the open-auction underreaction residue "
            f"(+{R['feasible']['^N225'][0]:.1f} bps/day gross, t = +{R['feasible']['^N225'][1]:.1f}) is "
            "exactly the kind of crumb HFT market-makers harvest. That is capacity-bound plumbing, not "
            "a retail strategy — and on the honest index opens it is *already* thinner than any retail "
            "cost stack.\n"
            "- **Extensions.** Add Korea/India/Brazil; regress the foreign *gap* on US futures moves "
            "during foreign hours (splitting 'US day' from 'overnight world news'); or test volatility "
            "spillover (Hamao-Masulis-Ng's second result) with a GARCH layer.\n\n"
            "*The reproducible core is offline and deterministic; alignment gives exactly one execution "
            "lag by construction. Methods and sources: [`docs/references.md`](../docs/references.md); "
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
