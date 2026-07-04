"""Generate the two narrative notebooks for Study 631 (Restatement Shock, Item 4.02).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached EDGAR event
list + yfinance prices under ../_cache/ and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md). The synthetic machinery control runs anywhere, no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (EDGAR FTS Item-4.02 8-Ks +
# yfinance daily adjusted closes vs SPY, as-of 2026-06-30, fingerprint 9b979967dc4c).
R = dict(
    n_fts=1498, n_ticker=520, n_usable=359, start="2004-09", end="2026-06",
    ar0_mean=-2.32, ar0_median=-0.66, ar0_t=-3.17, ar0_t_wins=-4.55, ar0_t_hac=-2.27,
    car_mean=-15.84, car_median=-5.86, car_t=-4.01, car_t_wins=-5.06, car_t_hac=-4.41,
    n_months=162, neg_share=0.624,
    # penny floors: (floor $, n, car_pct, t, t_hac)
    floors=[(1, 289, -13.41, -4.79, -4.20), (5, 228, -12.97, -4.05, -3.54)],
    # chronic-decay placebo (the verdict-maker)
    n_pairs=324, plc_event=-17.49, plc_event_t=-4.07, plc_plc=-13.13, plc_plc_t=-4.27,
    diff=-4.36, diff_t=-0.85, diff_t_wins=-0.93, diff_t_hac=-0.63,
    # horizon: (days, n, car_pct, t, t_hac)
    horizon=[(5, 367, -0.75, -0.92, -1.17), (21, 365, -6.27, -3.36, -2.70),
             (42, 361, -8.54, -2.79, -2.86), (63, 359, -15.84, -4.01, -4.41),
             (126, 352, -23.90, -5.37, -4.47)],
    # era: (label, n, ann_pct, ann_t_hac, car_pct, car_t, car_t_hac)
    era=[("2004-2012", 92, -0.40, -0.48, -2.81, -0.64, -0.28),
         ("2013-2026", 267, -2.98, -2.28, -20.33, -4.02, -5.13)],
    split_musd=0.18,
    small=(180, -21.80, -3.32, -3.60), large=(179, -9.85, -2.27, -1.56),  # (n,car,t,t_hac)
    welch_diff=-1.52,
    # overlay: (cost_bps, borrow_pct, gross, drag, net, net_t, net_t_hac)
    overlay=[(10, 3, 15.84, 0.97, 14.87, 3.77, 4.09),
             (20, 10, 15.84, 2.92, 12.92, 3.27, 3.45),
             (50, 25, 15.84, 7.27, 8.57, 2.17, 2.03)],
    # syn three worlds: (label, drift_pct, chronic_pct, raw_car, raw_t_hac, diff, diff_t)
    syn=[("null", 0, 0, 0.83, 1.48, 0.92, 0.71),
         ("true underreaction", -8, 0, -7.17, -8.80, -7.08, -5.42),
         ("chronic bleed only", 0, -40, -9.17, -11.37, 0.92, 0.71)],
    fingerprint="9b979967dc4c",
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Small-cap story?: Mixed](https://img.shields.io/badge/Small--cap_story%3F-Mixed-8b949e?style=flat-square)\n\n"
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

from restatement_shock import data, strategy as st

HAVE_REAL = data.have_real()
AS_OF = "2026-06-30"
if HAVE_REAL:
    EV, CLOSE, DVOL, SPY = data.load_real()
    SPY = SPY[SPY.index <= AS_OF]; CLOSE = CLOSE[CLOSE.index <= AS_OF]
    DVOL = DVOL[DVOL.index <= AS_OF]
    TAB = st.build_event_table(EV, CLOSE, DVOL, SPY)
else:
    EV = CLOSE = DVOL = SPY = TAB = None
print("real cache present:", HAVE_REAL,
      "| usable events:", (0 if TAB is None else len(TAB)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# \"Do not rely on our financials\" — what happens after a company confesses? 💣\n"
            "### The Item 4.02 restatement shock, in plain English\n\n"
            + BADGES +
            "There is a special kind of SEC filing no CEO wants to sign. It's called an "
            "**Item 4.02 8-K**, and it says, in regulatory language: *\"our previously published "
            "financial statements are wrong — **do not rely on them**.\"* Not \"we missed "
            "earnings.\" Not \"guidance is soft.\" The **numbers themselves were false** and will "
            "be restated.\n\n"
            "The folk claim: the market — slow, distracted, in denial — can't absorb an accounting "
            "bomb in one day. The stock drops on the news, then **keeps bleeding for months** as "
            "the real damage dribbles out. If that's true, you could **short the confession** and "
            "collect the drift.\n\n"
            "We pulled ~1,500 real \"do not rely\" filings from the SEC's archive (2004→2026) and "
            "measured what actually happens — the day of the bomb, and the three months after. "
            "The answer has a twist that most tellings of this story miss.\n\n"
            "> 📓 Want the *t*-stats, the HAC clustering and the borrow math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **One bias, named up front.** Our price source only knows companies that still "
            "exist. Firms that went **bankrupt** after confessing — the worst endings — drop out "
            "(65% of filings don't even map to a live ticker). Whatever bleeding we measure is an "
            "**understatement**."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the stock drop on the confession? | **Yes, instantly** — about "
            f"**{R['ar0_mean']:+.1f}%** vs the market in two days, statistically certified. |\n"
            "| Does it keep bleeding for months? | **The stock does — but not because of the "
            "confession.** Here's the twist: the *same stocks* were bleeding at almost the same "
            "rate in a random window a year *before* they confessed. They're melting ice cubes; "
            "the 8-K is a symptom, not the cause of the melt. |\n"
            "| Can you get rich shorting confessions? | **No.** The paper profit is really "
            "\"short tiny melting stocks\" — a trade you can't actually put on (no shares to "
            "borrow, ~$0.2M/day of liquidity). The *event* itself adds nothing bankable. |\n"
            "| Is it a small-cap thing? | Directionally yes (the smaller half bleeds twice as "
            "hard), but the halves aren't statistically separable. |\n"
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Accounting scandals aren't one-day events. The 8-K drop is just the first cut; "
            "the market underreacts, and the stock grinds lower for months as auditors, lawyers "
            "and short-sellers dig out the real damage.\"*\n\n"
            "This is the forensic-accounting cousin of a famous anomaly (post-earnings-"
            "announcement drift): markets are supposedly **slow to absorb bad fundamental news** — "
            "and an Item 4.02 is the *purest* bad fundamental news there is: the company itself "
            "declaring its own books unreliable.\n\n"
            "Our desk sibling [229-beneish-m-score](../../229-beneish-m-score/) tries to "
            "**predict** manipulation before anyone admits it. This study is the other end of the "
            "pipeline: the **confession itself**, and what the tape does after it."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the drift is real and event-driven, it's a shortable edge — and an indictment of "
            "market efficiency at its most embarrassing point (public, mandatory, unambiguous bad "
            "news). If it isn't, the lesson flips twice: first, even accounting bombs get priced "
            "**fast**; second — and this is the trap this study exists to expose — a stock can "
            "fall for months after an event **without the event causing any of it**, if the kind "
            "of company that has such events is the kind that falls all the time."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"1. **Collect confessions.** EDGAR full-text search for 8-Ks whose item list contains "
            f"**4.02** — **{R['n_fts']:,}** filings, {R['start']} → {R['end']}, sampled evenly "
            "across time.\n"
            f"2. **Match to the tape.** {R['n_ticker']} map to a ticker; **{R['n_usable']}** have "
            "enough price history (the dead drop out — named bias).\n"
            "3. **Measure two windows vs the market (SPY).** The **bomb**: filing day + next day. "
            "The **drift**: from the close of the day after the filing (when you could actually "
            "trade) to 63 trading days later — about 3 months.\n"
            "4. **Count honestly.** Confessions cluster in waves, so we average events month by "
            "month before testing.\n"
            "5. **The twist control.** For every event, we measure the *same stock* over a "
            "same-length window **one year earlier** — no confession anywhere in sight. If the "
            "stock bleeds there too, the \"drift\" isn't the event's doing."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown — let's actually look\n\n**First, the bomb itself.**"),
        code(
            "if HAVE_REAL:\n"
            "    ar0 = TAB['ar0'].to_numpy()*100\n"
            "else:\n"
            "    ar0 = np.random.default_rng(631).normal(R['ar0_mean'], 12, R['n_usable'])\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.hist(np.clip(ar0, -60, 40), bins=60, color=RED, alpha=.85)\n"
            "ax.axvline(0, c=GREY, lw=1)\n"
            "ax.axvline(np.mean(ar0), c='k', ls='--', lw=2, label=f'mean {np.mean(ar0):+.1f}%')\n"
            "ax.set_xlabel('return vs market, filing day + next day (%)')\n"
            "ax.set_ylabel('number of confessions')\n"
            "ax.set_title('The bomb: what \\'do not rely on our financials\\' does on the day')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'mean {np.mean(ar0):+.2f}%  median {np.median(ar0):+.2f}%  n={len(ar0)}')"
        ),
        md(
            f"The average confession costs about **{R['ar0_mean']:+.1f}%** versus the market in "
            f"two days (median {R['ar0_median']:+.1f}% — many 4.02s are technical bookkeeping "
            "fixes, a few are catastrophes). Real, instant, certified.\n\n"
            "**Now the months after.** We start the clock at the close of the day *after* the "
            "filing and follow the stock vs the market."
        ),
        code(
            "if HAVE_REAL:\n"
            "    hz = st.horizon_curve(EV, CLOSE, DVOL, SPY)\n"
            "    days = hz['days'].tolist(); cars = hz['car_pct'].tolist()\n"
            "else:\n"
            "    days = [h[0] for h in R['horizon']]; cars = [h[2] for h in R['horizon']]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "ax.plot([0]+days, [0]+cars, marker='o', color=RED, lw=2)\n"
            "ax.axhline(0, c=GREY, lw=1)\n"
            "ax.fill_between([0]+days, [0]+cars, 0, color=RED, alpha=.12)\n"
            "ax.set_xlabel('trading days after entry (close of day +1)')\n"
            "ax.set_ylabel('average slide vs market (%)')\n"
            "ax.set_title('After the bomb: the average confessed stock keeps sliding...')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('mean drift by horizon (%):', dict(zip([int(d) for d in days], [round(c,2) for c in cars])))"
        ),
        md(
            f"Over the next **3 months** the average confessed stock slides another "
            f"**{R['car_mean']:+.1f}%** vs the market — and it *looks* statistically solid. Case "
            "closed, market inefficient, short everything?\n\n"
            "**Not so fast. Here's the twist.** What kind of company files a \"do not rely\"? "
            "Overwhelmingly: tiny, struggling, cash-burning microcaps (median trading volume in "
            "our panel: **$0.18M a day**). Stocks like that bleed *all the time*, confession or "
            "no confession. So we ran the control: the **same stocks**, a same-length window, "
            "**one year before** the event."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.chronic_decay_placebo(EV, CLOSE, DVOL, SPY)\n"
            "    vals = [pl['event_pct'], pl['placebo_pct'], pl['diff_pct']]\n"
            "    dt = pl['diff_t']\n"
            "else:\n"
            "    vals = [R['plc_event'], R['plc_plc'], R['diff']]; dt = R['diff_t']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "labs = ['3 months AFTER\\nthe confession', 'a random window\\n1 YEAR BEFORE it',\n"
            "        'the difference\\n(what the event adds)']\n"
            "ax.bar(labs, vals, color=[RED, GREY, AMBER], width=.55)\n"
            "for i, v in enumerate(vals):\n"
            "    ax.annotate(f'{v:+.1f}%', (i, v), ha='center', va='top', fontweight='bold')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('return vs market (%)')\n"
            "ax.set_title('The twist: these stocks were melting anyway')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'event {vals[0]:+.2f}%  placebo {vals[1]:+.2f}%  difference {vals[2]:+.2f}% (t={dt:+.2f})')"
        ),
        md(
            f"There it is. After the confession: **{R['plc_event']:+.1f}%**. In a random no-news "
            f"window a year earlier: **{R['plc_plc']:+.1f}%**. The confession's own contribution "
            f"is only **{R['diff']:+.1f}%** — and statistically indistinguishable from zero "
            "(*t* ≈ −0.9, quants notebook). These are **melting ice cubes**: the 8-K doesn't "
            "start the melt, it's a thermometer reading taken partway through it.\n\n"
            "**And the \"trade\"?** Shorting each confession for 3 months, hedged, does show a "
            "paper profit — but it's the melting-ice-cube exposure, not the event, and the names "
            "are unshortable in practice: sub-$1 prices, no shares to locate, ~$0.2M/day of "
            "volume, borrow fees that can exceed the drift. The quants notebook prices it at "
            "three cost levels; even where the model says \"profit,\" the locate desk says \"no.\""
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **The bomb is real** — {R['ar0_mean']:+.1f}% vs market in two days, certified. "
            "Markets price confessions *fast*.\n"
            f"- **The months-of-drift story is mostly an illusion of composition** — confessed "
            f"stocks fall another {R['car_mean']:+.1f}% in 3 months, but the same stocks fell "
            f"{R['plc_plc']:+.1f}% in windows with no news at all. The event-specific part "
            f"({R['diff']:+.1f}%) certifies nothing.\n"
            "- **Tradability — Mirage.** What survives costs on paper is a generic "
            "short-dying-microcaps trade you can't actually execute at size.\n"
            "- **And remember the named bias:** the bankrupt are missing from our tape, so the "
            "true post-confession carnage is *worse* than measured — but proving the "
            "*underreaction* claim needs the event to add something beyond the melt, and on this "
            "tape it doesn't."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Get the dead.** A survivorship-free tape (CRSP with delisting returns) would "
            "restore the bankrupt tail — the one place the event-specific drift could still be "
            "hiding.\n"
            "- **Severity matters.** A 4.02 triggered by fraud is a different animal from one "
            "triggered by a lease-accounting technicality (or the 2021 SPAC-warrant wave). "
            "Reading the 8-K text, not just the item number, is the natural refinement.\n"
            "- **The pipeline view.** Pair with [229-beneish-m-score](../../229-beneish-m-score/): "
            "the M-Score flags cooked books *before* the confession; this study shows the "
            "confession itself is priced in two days.\n\n"
            "*Think three months of drift on hard-to-borrow microcaps is a business? Price the "
            "locate first — then we'll talk.*"
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
            "# Restatement Shock (Item 4.02) — a quantitative teardown 🔬\n"
            "### Market-adjusted event study on EDGAR \"do not rely\" 8-Ks · month-clustered HAC "
            "inference · a chronic-decay paired placebo · horizon, era & penny-floor robustness "
            "· size split · borrow-aware shorting costs\n\n"
            + BADGES +
            "Deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim: the market **underreacts** to Item 4.02 non-reliance 8-Ks — the confession "
            "keeps hurting for months. The raw drift *screams*; the study's job is to ask whether "
            "it is **event-specific** — and the answer flips the verdict.\n\n"
            "> ⚠️ **Deads-missing bias, named on the Signal axis.** Tickers come from the SEC's "
            "*current* CIK→ticker map and yfinance carries no delisted history: 65% of filings "
            "never reach the tape, and post-confession **bankruptcies drop out**. Every negative "
            "number below is an **understatement** of the true damage; the bias is worst for old "
            "events (visible in the era split).\n"
            ">\n"
            "> 💡 The `💡 In plain words` notes translate each block back to intuition. Numbers: "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fingerprint"] + "`)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | **Real on the bomb**: AR[0,+1] = **{R['ar0_mean']:+.2f}%** "
            f"(HAC t = **{R['ar0_t_hac']:+.2f}**, winsorized t = {R['ar0_t_wins']:+.2f}). **Not "
            f"certified on the underreaction drift**: raw CAR[+2,+64] = **{R['car_mean']:+.2f}%** "
            f"(HAC t = {R['car_t_hac']:+.2f}) collapses to **{R['diff']:+.2f}%** at paired "
            f"t = **{R['diff_t']:+.2f}** against the same-stock chronic-decay placebo. |\n"
            f"| **Tradability** | `MIRAGE` | The modeled short nets {R['overlay'][1][4]:+.2f}%/event "
            f"at 20 bps + 10%/yr borrow — but the edge is chronic microcap decay, **not the "
            f"event** (event-specific t = {R['diff_t']:+.2f}), on names with median "
            f"**${R['split_musd']:.2f}M/day** dollar volume where locates are scarce-to-"
            "nonexistent. |\n"
            f"| **Small-cap story?** | `MIXED` | Small half **{R['small'][1]:+.2f}%** (HAC "
            f"t = {R['small'][3]:+.2f}) vs large half {R['large'][1]:+.2f}% (HAC "
            f"t = {R['large'][3]:+.2f}); Welch t (diff) = **{R['welch_diff']:+.2f}** — the "
            "limits-to-arbitrage direction, not separable. |\n\n"
            "> 💡 In plain words: the bomb is instant and certified. The famous drift is real "
            "*bleeding* but not real *underreaction* — the confessing stocks bleed the same way "
            "in windows with no confession at all."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $t_0$ be the first trading day on/after the 8-K filing date, $r$ log returns, "
            "$m$ the SPY log return. Define\n\n"
            "$$AR_{[0,+1]} = \\sum_{s=t_0}^{t_0+1}(r_s - m_s), \\qquad "
            "CAR_{[+2,+64]} = \\sum_{s=t_0+2}^{t_0+64}(r_s - m_s).$$\n\n"
            "- **H₁ (the bomb).** $\\mathbb{E}[AR_{[0,+1]}] < 0$, large and significant.\n"
            "- **H₂ (raw drift).** $\\mathbb{E}[CAR_{[+2,+64]}] < 0$ with an overlap-honest "
            "t ≥ 2.\n"
            "- **H₂′ (the actual claim — event-specific drift).** The drift must exceed the same "
            "stock's *unconditional* decay: with $CAR^{plc}$ measured on a same-length window "
            "entered 252 trading days earlier, $\\mathbb{E}[CAR - CAR^{plc}] < 0$ at t ≥ 2. "
            "Underreaction means the *event* moves the future — not that fallers keep falling.\n"
            "- **H₃ (limits to arbitrage).** The drift concentrates in the small half "
            "(pre-event dollar volume).\n\n"
            "Entry at the **close of day +1** is the single, documented execution lag."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two inference traps this study exists to dodge\n\n"
            "**(a) Overlap.** Restatement waves cluster (SOX era, the 2021 SPAC-warrant wave), "
            "and 63-day windows of same-wave events overlap almost fully — a per-event t "
            "over ~360 windows overstates independence. Headline statistic: collapse to "
            "**calendar-month means**, then **Newey-West (3-lag) HAC t** across months.\n\n"
            "**(b) Composition.** The 4.02 population is chronically-bleeding microcaps. "
            "Conditioning on the event also conditions on *being that kind of stock*. The "
            "**paired chronic-decay placebo** (same stock, same window length, one year earlier) "
            "is the only test here that can attribute the drift to the *event*. Microcap tails "
            "are violent, so winsorized (1%/99%) and penny-floor variants ride along."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Events.** EDGAR FTS, 8-Ks with `items` ∋ `4.02`, sampled per quarter "
            f"{R['start']} → {R['end']} with deterministic quarter-stratified thinning: "
            f"**{R['n_fts']:,}** filings → **{R['n_ticker']}** with a ticker → "
            f"**{R['n_usable']}** usable (60d pre-history + complete 63d post window).\n"
            "- **Benchmark.** SPY total-return; abnormal = stock log return − SPY log return.\n"
            "- **Windows.** Announcement [0,+1]; drift [+2,+64], entry close of +1 (one lag).\n"
            "- **Inference.** Per-event t (shown, refused) → winsorized t → **month-clustered "
            "NW-HAC t (headline)**; penny floors $1/$5; horizons 5→126d; era split "
            "2004-2012 / 2013-2026.\n"
            "- **Attribution.** Paired chronic-decay placebo, lag 252d — the verdict-maker.\n"
            "- **Third axis.** Median split on pre-event median dollar volume ([−65,−6]); "
            "Welch t.\n"
            "- **Costs.** Short at close +1, cover +64, SPY-hedged; one-way bps × 2 legs + "
            "1 bp × 2 on SPY + borrow × 63/252. Shorts pay borrow.\n"
            "- **Machinery.** Three synthetic worlds (null / true underreaction / chronic "
            "bleed): the pipeline must tell drift from decay."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md("### 4a · The bomb — announcement window [0,+1]"),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize(TAB)\n"
            "    ar0 = TAB['ar0'].to_numpy()*100\n"
            "else:\n"
            "    s = dict(ar0_mean_pct=R['ar0_mean'], ar0_median_pct=R['ar0_median'],\n"
            "             ar0_t=R['ar0_t'], ar0_t_wins=R['ar0_t_wins'], ar0_t_hac=R['ar0_t_hac'])\n"
            "    ar0 = np.random.default_rng(631).normal(R['ar0_mean'], 12, R['n_usable'])\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.hist(np.clip(ar0, -60, 40), bins=60, color=RED, alpha=.85)\n"
            "ax.axvline(np.mean(ar0), c='k', ls='--', lw=2, label=f'mean {np.mean(ar0):+.1f}%')\n"
            "ax.axvline(0, c=GREY, lw=1)\n"
            "ax.set_xlabel('AR[0,+1] vs SPY (%)'); ax.set_ylabel('events')\n"
            "ax.set_title(f\"Announcement: mean {s['ar0_mean_pct']:+.2f}% | per-event t={s['ar0_t']:+.1f} | HAC t={s['ar0_t_hac']:+.2f}\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"AR0 mean {s['ar0_mean_pct']:+.2f}%  median {s['ar0_median_pct']:+.2f}%  \"\n"
            "      f\"t={s['ar0_t']:+.2f}  wins t={s['ar0_t_wins']:+.2f}  HAC t={s['ar0_t_hac']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: the confession costs **{R['ar0_mean']:+.1f}%** vs the market "
            f"in two days (median {R['ar0_median']:+.1f}% — most 4.02s are technical; a minority "
            f"are catastrophic). Winsorized t = **{R['ar0_t_wins']:+.2f}**, month-clustered HAC "
            f"t = **{R['ar0_t_hac']:+.2f}** — H₁ **certified**. This half of the claim is just "
            "market efficiency working."
        ),
        md(
            "### 4b · The raw drift — loud, robust… and not yet an answer\n\n"
            "Horizon curve plus the headline 63-day window, with winsorized and penny-floor "
            "variants. Everything here passes; the question it *cannot* answer is attribution."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize(TAB)\n"
            "    hz = st.horizon_curve(EV, CLOSE, DVOL, SPY)\n"
            "    H = list(zip(hz['days'], hz['n'], hz['car_pct'], hz['t'], hz['t_hac']))\n"
            "    FL = []\n"
            "    for f in (1.0, 5.0):\n"
            "        sub = TAB[TAB['p_in'] >= f]\n"
            "        FL.append((f, len(sub), sub['car'].mean()*100,\n"
            "                   st.ttest_vs_zero(sub['car'].to_numpy()),\n"
            "                   st.clustered_hac_t(sub)['t_hac']))\n"
            "    core = (s['car_mean_pct'], s['car_median_pct'], s['car_t'], s['car_t_wins'], s['car_t_hac'])\n"
            "else:\n"
            "    H = R['horizon']; FL = R['floors']\n"
            "    core = (R['car_mean'], R['car_median'], R['car_t'], R['car_t_wins'], R['car_t_hac'])\n"
            "days = [h[0] for h in H]; cars = [h[2] for h in H]; thac = [h[4] for h in H]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))\n"
            "a1.plot([0]+days, [0]+cars, marker='o', color=RED, lw=2)\n"
            "a1.axhline(0, c=GREY, lw=1); a1.fill_between([0]+days, [0]+cars, 0, color=RED, alpha=.12)\n"
            "a1.set_xlabel('trading days after entry'); a1.set_ylabel('mean CAR vs SPY (%)')\n"
            "a1.set_title('Raw drift by horizon')\n"
            "a2.bar([str(int(d)) for d in days], thac, color=[AMBER if abs(t)<2 else RED for t in thac], width=.6)\n"
            "a2.axhline(-2, ls='--', c='k', label='|t|=2 bar'); a2.axhline(0, c=GREY, lw=1)\n"
            "for i,t in enumerate(thac): a2.annotate(f'{t:+.2f}',(i,t),ha='center',va='top' if t<0 else 'bottom')\n"
            "a2.set_xlabel('horizon (trading days)'); a2.set_ylabel('month-clustered HAC t')\n"
            "a2.set_title('...and its clustered t'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'63d: CAR {core[0]:+.2f}% (median {core[1]:+.2f}%)  t={core[2]:+.2f}  '\n"
            "      f'wins t={core[3]:+.2f}  HAC t={core[4]:+.2f}')\n"
            "for f in FL: print(f'entry >= ${f[0]:.0f}: n={int(f[1])}  CAR {f[2]:+.2f}%  t={f[3]:+.2f}  HAC t={f[4]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the raw 3-month drift is **{R['car_mean']:+.2f}%** with a "
            f"month-clustered HAC t of **{R['car_t_hac']:+.2f}** — it clears the bar, survives "
            f"winsorization ({R['car_t_wins']:+.2f}), penny floors (≥$5: {R['floors'][1][2]:+.2f}%, "
            f"HAC {R['floors'][1][4]:+.2f}) and grows with horizon. A naive study stops here and "
            "stamps REAL. The next cell is why we don't."
        ),
        md(
            "### 4c · The chronic-decay placebo — the verdict-maker\n\n"
            "H₂′: same stock, same 63-day window length, entered **252 trading days before** the "
            "event. If the \"drift\" is the event's doing, the placebo should be ≈ 0 and the "
            "paired difference should carry the whole effect."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.chronic_decay_placebo(EV, CLOSE, DVOL, SPY)\n"
            "else:\n"
            "    pl = dict(n_pairs=R['n_pairs'], event_pct=R['plc_event'], event_t=R['plc_event_t'],\n"
            "              placebo_pct=R['plc_plc'], placebo_t=R['plc_plc_t'], diff_pct=R['diff'],\n"
            "              diff_t=R['diff_t'], diff_t_wins=R['diff_t_wins'], diff_t_hac=R['diff_t_hac'])\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "labs = ['event window\\n[+2,+64]', 'placebo window\\n(1y earlier, no event)',\n"
            "        'paired difference\\n(event-specific)']\n"
            "vals = [pl['event_pct'], pl['placebo_pct'], pl['diff_pct']]\n"
            "ts   = [pl['event_t'], pl['placebo_t'], pl['diff_t']]\n"
            "ax.bar(labs, vals, color=[RED, GREY, AMBER], width=.55)\n"
            "for i,(v,t) in enumerate(zip(vals,ts)):\n"
            "    ax.annotate(f'{v:+.1f}%\\nt={t:+.2f}', (i, v), ha='center', va='top', fontweight='bold')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('market-adjusted CAR (%)')\n"
            "ax.set_title(f'Chronic decay absorbs the drift (n={pl[\"n_pairs\"]} pairs)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"event {pl['event_pct']:+.2f}% (t={pl['event_t']:+.2f}) | placebo \"\n"
            "      f\"{pl['placebo_pct']:+.2f}% (t={pl['placebo_t']:+.2f}) | diff {pl['diff_pct']:+.2f}% \"\n"
            "      f\"(t={pl['diff_t']:+.2f}, wins {pl['diff_t_wins']:+.2f}, HAC {pl['diff_t_hac']:+.2f})\")"
        ),
        md(
            f"> 💡 In plain words: the placebo windows — no confession anywhere near them — bleed "
            f"**{R['plc_plc']:+.2f}%** (t = {R['plc_plc_t']:+.2f}), almost as much as the event "
            f"windows ({R['plc_event']:+.2f}%). The event-specific increment is "
            f"**{R['diff']:+.2f}%** at paired t = **{R['diff_t']:+.2f}** (winsorized "
            f"{R['diff_t_wins']:+.2f}, HAC {R['diff_t_hac']:+.2f}). Right sign, nowhere near the "
            "bar. **H₂′ fails** — the drift is composition (melting ice cubes), not underreaction. "
            "This single test flips the study's verdict from REAL to MIXED."
        ),
        md(
            "### 4d · Era split — where the survivor-mapping bias shows its face\n\n"
            "The CIK→ticker map is current: a 2005 filer still mapped today is a 20-year "
            "survivor, so the early era keeps only its *winners*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    E = []\n"
            "    for lab, m in (('2004-2012', TAB['date'] <= '2012-12-31'),\n"
            "                   ('2013-2026', TAB['date'] > '2012-12-31')):\n"
            "        se = st.summarize(TAB[m])\n"
            "        E.append((lab, se['n_events'], se['ar0_mean_pct'], se['ar0_t_hac'],\n"
            "                  se['car_mean_pct'], se['car_t'], se['car_t_hac']))\n"
            "else:\n"
            "    E = R['era']\n"
            "labs = [e[0] for e in E]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(labs, [e[2] for e in E], color=RED, width=.5)\n"
            "for i,e in enumerate(E): a1.annotate(f'{e[2]:+.1f}%\\nHAC t={e[3]:+.2f}',(i,e[2]),ha='center',va='top')\n"
            "a1.set_title('Announcement bomb by era'); a1.set_ylabel('AR[0,+1] (%)')\n"
            "a2.bar(labs, [e[4] for e in E], color=AMBER, width=.5)\n"
            "for i,e in enumerate(E): a2.annotate(f'{e[4]:+.1f}%\\nHAC t={e[6]:+.2f}',(i,e[4]),ha='center',\n"
            "    va='top' if e[4]<0 else 'bottom')\n"
            "a2.set_title('3-month raw drift by era'); a2.set_ylabel('CAR[+2,+64] (%)')\n"
            "for a in (a1,a2): a.axhline(0, c=GREY, lw=1)\n"
            "plt.tight_layout(); plt.show()\n"
            "for e in E: print(f'{e[0]}: n={e[1]}  ann {e[2]:+.2f}% (HAC t={e[3]:+.2f})  drift {e[4]:+.2f}% (t={e[5]:+.2f}, HAC t={e[6]:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: the early era is flat (drift {R['era'][0][4]:+.1f}%, HAC "
            f"t = {R['era'][0][6]:+.2f}) — exactly where only 20-year survivors remain on the "
            f"tape — and the modern era carries everything ({R['era'][1][4]:+.1f}%, HAC "
            f"t = {R['era'][1][6]:+.2f}, SPAC-era microcaps). Consistent with the named mapping "
            "bias plus chronic decay; inconsistent with a stable underreaction premium."
        ),
        md(
            "### 4e · Third axis — limits to arbitrage: is it a small-cap story?\n\n"
            "Median split on pre-event median **dollar volume** ([−65,−6]; only the "
            "cross-sectional order is used). Shleifer-Vishny: mispricing persists where "
            "arbitrage capital won't go."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sp = st.size_split(TAB)\n"
            "    sm = (sp['n_small'], sp['small_car_pct'], sp['small_t'], sp['small_t_hac'])\n"
            "    lg = (sp['n_large'], sp['large_car_pct'], sp['large_t'], sp['large_t_hac'])\n"
            "    wd = sp['welch_t_diff']; med = sp['median_dvol_musd']\n"
            "else:\n"
            "    sm, lg, wd, med = R['small'], R['large'], R['welch_diff'], R['split_musd']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['SMALL half\\n(illiquid)','LARGE half\\n(liquid)'], [sm[1], lg[1]],\n"
            "       color=[RED, GREY], width=.5)\n"
            "for i,(v,t) in enumerate([(sm[1],sm[3]),(lg[1],lg[3])]):\n"
            "    ax.annotate(f'{v:+.2f}%\\nHAC t={t:+.2f}',(i,v),ha='center',va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c=GREY, lw=1)\n"
            "ax.set_ylabel('3-month raw drift CAR (%)')\n"
            "ax.set_title(f'Drift by pre-event dollar volume (split ${med:.2f}M/day) | Welch t diff = {wd:+.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'small: n={sm[0]} CAR {sm[1]:+.2f}% t={sm[2]:+.2f} HAC {sm[3]:+.2f} | '\n"
            "      f'large: n={lg[0]} CAR {lg[1]:+.2f}% t={lg[2]:+.2f} HAC {lg[3]:+.2f} | Welch diff {wd:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: **{R['small'][1]:+.2f}%** in the illiquid half vs "
            f"**{R['large'][1]:+.2f}%** in the liquid half — the direction Shleifer-Vishny "
            f"predicts — but Welch t = **{R['welch_diff']:+.2f}**: not separable. And note the "
            f"whole panel is microcap (the *median* split point is ${R['split_musd']:.2f}M/day); "
            "\"large\" here means *less tiny*. **MIXED** — and the placebo already showed the "
            "drift isn't event-specific in either half's favor."
        ),
        md(
            "### 4f · Tradability — the short pays its bills first\n\n"
            "Short at close +1, cover at +64 (the market-adjusted CAR *is* the SPY-hedged P&L); "
            "one-way costs × 2 legs on the stock + 1 bp × 2 on SPY + borrow × 63/252."
        ),
        code(
            "if HAVE_REAL:\n"
            "    OV = []\n"
            "    for cb, bw in ((10.0,3.0),(20.0,10.0),(50.0,25.0)):\n"
            "        o = st.short_overlay(TAB, cost_bps=cb, borrow_ann_pct=bw)\n"
            "        OV.append((cb, bw, o['gross_pct'], o['drag_pct'], o['net_pct'],\n"
            "                   o['net_t'], o['net_t_hac']))\n"
            "else:\n"
            "    OV = R['overlay']\n"
            "labels = [f'{int(o[0])}bps\\n{int(o[1])}%/yr borrow' for o in OV]\n"
            "x = np.arange(len(OV))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(x-.18, [o[2] for o in OV], .34, color=GREY, label='gross (raw drift)')\n"
            "ax.bar(x+.18, [o[4] for o in OV], .34, color=AMBER, label='net of costs + borrow')\n"
            "for i,o in enumerate(OV): ax.annotate(f'{o[4]:+.1f}%\\nt={o[5]:+.2f}',(i+.18,o[4]),\n"
            "    ha='center', va='bottom', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('per-event 3-month return (%)')\n"
            "ax.set_title('Modeled short survives on paper — but the edge is not the event')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for o in OV: print(f'cost={o[0]:.0f}bps borrow={o[1]:.0f}%/yr: gross {o[2]:+.2f}% '\n"
            "                   f'drag {o[3]:.2f}% net {o[4]:+.2f}%  t={o[5]:+.2f}  HAC t={o[6]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the *model* says even punitive terms leave "
            f"{R['overlay'][2][4]:+.2f}%/event. So why MIRAGE? Three reasons. **(1)** The edge "
            f"is not the claimed one — the event adds {R['diff']:+.2f}% (t = {R['diff_t']:+.2f}); "
            "what you'd really be harvesting is generic melting-microcap exposure. **(2)** "
            f"Access: median pre-event dollar volume is ${R['split_musd']:.2f}M/day, many names "
            "are sub-$1 with no locate at any fee — the modeled borrow grid is generous "
            "fiction for half the panel. **(3)** Capacity ≈ nil; a buy-in forces you out "
            "exactly when the melt accelerates. Paper edge, unclaimable — **MIRAGE**."
        ),
        md(
            "### 4g · Machinery control — three worlds where we know the truth\n\n"
            "Null (nothing planted), TRUE underreaction (−8% planted post-event drift), and "
            "the melting-ice-cube confound (−40%/yr chronic bleed on event stocks, **zero** "
            "event effect). The pipeline must tell drift from decay. *(Machinery proof only — "
            "never market evidence.)*"
        ),
        code(
            "res = []\n"
            "for lab, drift, chronic in (('null', 0.0, 0.0), ('true underreaction', -0.08, 0.0),\n"
            "                            ('chronic bleed only', 0.0, -0.40)):\n"
            "    ev_s, cl_s, dv_s, spy_s = data.synthetic_market(drift=drift, chronic=chronic, seed=631)\n"
            "    tab_s = st.build_event_table(ev_s, cl_s, dv_s, spy_s)\n"
            "    s_s = st.summarize(tab_s)\n"
            "    pl_s = st.chronic_decay_placebo(ev_s, cl_s, dv_s, spy_s)\n"
            "    res.append((lab, s_s['car_mean_pct'], s_s['car_t_hac'], pl_s['diff_pct'], pl_s['diff_t']))\n"
            "x = np.arange(len(res))\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.5))\n"
            "ax.bar(x-.18, [r[2] for r in res], .34, color=GREY, label='raw drift HAC t')\n"
            "ax.bar(x+.18, [r[4] for r in res], .34, color=AMBER, label='event-specific paired t')\n"
            "ax.axhline(-2, ls='--', c=RED, label='|t|=2 bar'); ax.axhline(2, ls='--', c=RED)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels([r[0] for r in res])\n"
            "ax.set_ylabel('t-statistic')\n"
            "ax.set_title('Raw drift false-fires on chronic bleed; the paired placebo does not')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for r in res: print(f'{r[0]:<20}: raw {r[1]:+.2f}% (HAC t={r[2]:+.2f}) | '\n"
            "                    f'event-specific {r[3]:+.2f}% (paired t={r[4]:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: in the chronic-bleed world the raw detector screams (HAC "
            f"t = {R['syn'][2][4]:+.2f}) with **zero** true event effect — a false conviction — "
            f"while the paired placebo stays quiet (t = {R['syn'][2][6]:+.2f}); given a TRUE "
            f"planted underreaction both fire (paired t = {R['syn'][1][6]:+.2f}); in the null, "
            "neither. The machinery separates drift from decay — and the real tape matches the "
            "**chronic-bleed** world, not the underreaction one. *(Never cited as market "
            "evidence.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** (split spelled out): **Real on the bomb** — AR[0,+1] = "
            f"{R['ar0_mean']:+.2f}%, HAC t = {R['ar0_t_hac']:+.2f}, winsorized "
            f"{R['ar0_t_wins']:+.2f}. **Not certified on the underreaction drift** — raw "
            f"{R['car_mean']:+.2f}% (HAC t = {R['car_t_hac']:+.2f}) is real *bleeding*, but the "
            f"event-specific part is {R['diff']:+.2f}% at paired t = {R['diff_t']:+.2f}: the "
            "confessing stocks bleed the same in no-event windows. **Deads-missing bias named**: "
            "the true damage is worse than this survivor-tilted tape can show, but a stamp needs "
            "the tape we have.\n"
            f"- **Tradability `MIRAGE`** — the modeled net (+{R['overlay'][1][4]:.2f}%/event at "
            "realistic terms) is chronic-decay exposure, not the event; unshortable sub-"
            f"${R['split_musd']:.2f}M/day scandal names; capacity ≈ nil.\n"
            f"- **Small-cap story? `MIXED`** — {R['small'][1]:+.2f}% (small) vs "
            f"{R['large'][1]:+.2f}% (large), right direction, Welch t = {R['welch_diff']:+.2f} — "
            "not separable."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Get the dead.** The one upgrade that could change the verdict: a survivorship-"
            "free tape (CRSP + delisting returns). The direction of the bias is *known* — it "
            "hides the bankrupt tail where a genuine event-specific drift could live.\n"
            "- **Severity conditioning.** 4.02(a) vs 4.02(b) (management vs auditor), fraud "
            "language, restatement magnitude — Palmrose et al. find severity is *the* "
            "cross-sectional driver; the 2021 SPAC-warrant wave is a natural low-severity "
            "stratum.\n"
            "- **Matched controls.** The placebo here is the stock's own past; a "
            "characteristics-matched control panel (same size/price/momentum, no 4.02) is the "
            "next rigor level — it also handles time-varying decay rates.\n"
            "- **The pipeline view.** [229-beneish-m-score](../../229-beneish-m-score/) predicts "
            "the confession; this study shows the confession itself is priced in two days — so "
            "the alpha, if any, lives *before* the 8-K, not after it.\n\n"
            "*Reproducible core offline & deterministic; one execution lag (close of day +1); "
            "shorts pay borrow. Sources: [`docs/references.md`](../docs/references.md); frozen "
            "numbers: [`docs/results.md`](../docs/results.md).*"
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
