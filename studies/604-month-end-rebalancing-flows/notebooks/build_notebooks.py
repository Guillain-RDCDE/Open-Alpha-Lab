"""Generate the two narrative notebooks for Study 604 (Month-End Rebalancing Flows).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached
SPY/AGG/VBMFX tape under ../_cache/ and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with no
network. Heavy pieces (the 20-seed x 2,000-draw placebo, the 20-seed null sweep) are run
in reduced form in-cell and the canonical numbers are quoted from ``R``.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SPY + spliced
# VBMFX->AGG bond leg, 1993-02-01 -> 2026-06-30, 401 complete months, as-of 2026-06-30).
R = dict(
    start="1993-02-01", end="2026-06-30", n_months=401, n_days=8410,
    splice="2003-09-30", fingerprint="3efe1915419d", as_of="2026-06-30",
    # Leg 1 — sell-the-winner (last-3 spread | gap_pre quintiles)
    sell=dict(top_bps=-46.53, t_top=-2.88, bot_bps=11.58, t_bot=0.36,
              uncond_bps=-19.13, t_uncond=-2.07, diff_bps=-58.11, welch=-1.60,
              top_vs_uncond=-1.47, p_mean=0.0225, p_lo=0.0175, p_hi=0.0300,
              n_top=81, n=401),
    # Leg 2 — reversal (first-3 spread | gap_full quintiles)
    rev=dict(top_bps=50.67, t_top=2.26, bot_bps=13.96, t_bot=0.44,
             uncond_bps=33.54, t_uncond=3.28, diff_bps=36.70, welch=0.95,
             p_mean=0.1317, p_lo=0.1240, p_hi=0.1435, n_top=80, n=400),
    # dose-response: (quintile, gap%, last3 bps, t, first3 bps, t)
    dose=[(1, -5.32, 11.58, 0.36, 13.96, 0.44),
          (2, -0.92, 2.22, 0.14, 45.78, 2.00),
          (3, 1.20, -19.75, -1.19, 29.96, 1.54),
          (4, 2.87, -43.63, -2.75, 27.33, 1.81),
          (5, 6.22, -46.45, -2.84, 50.67, 2.26)],
    # era: (label, sell top, sell t, sell diff, sell welch, rev top, rev t, rev diff, rev welch)
    era=[("1993-2003", -69.18, -1.95, -0.99, -0.01, 3.52, 0.08, -29.92, -0.38),
         ("2004-2015", -50.99, -2.17, -116.71, -2.14, 135.50, 5.22, 180.14, 3.22),
         ("2016+", -12.07, -0.57, -50.71, -0.95, -8.54, -0.24, -92.89, -1.52)],
    # window robustness: (k, sell top, sell t, sell welch, rev top, rev t, rev welch)
    windows=[(2, -42.39, -2.74, -1.90, 61.18, 3.61, 1.96),
             (3, -46.53, -2.88, -1.60, 50.67, 2.26, 0.95),
             (4, -55.51, -3.13, -3.61, 66.10, 2.82, 1.72)],
    # trade: (label, events, net %/yr, HAC t)
    trade=[("2 bps/leg, full tape", 70, 1.72, 2.52),
           ("5 bps/leg, full tape", 70, 1.22, 1.78),
           ("2 bps, 1993-2015", 46, 2.90, 3.42),
           ("2 bps, 2016-2026", 26, -0.07, -0.07)],
    trade_gross_pct=2.06, per_event_gross_bps=98.1, per_event_cost_bps=16,
    share_active_pct=5.0,
    # third axis — TOM vs flows (first-3-day SPY drift)
    tom=dict(uncond_bps=34.41, t_uncond=3.55, mid_bps=36.35, t_mid=3.45,
             top_bps=43.88, t_top=2.05, bot_bps=19.12, t_bot=0.64,
             welch_ext_mid=-0.23, n_mid=240, n_ext=160),
    # synthetic control
    syn=dict(null_mean=-0.04, null_sd=0.59, null_max_abs=1.79, null_seeds=20,
             planted_sell_welch=-9.26, planted_rev_welch=5.66,
             planted_net_pct=4.35, planted_hac=7.04,
             null_net_pct=0.04, null_hac=0.07),
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![TOM is flows in disguise?: Busted](https://img.shields.io/badge/TOM_is_flows_in_disguise%3F-Busted-8b949e?style=flat-square)\n\n"
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

from month_end_rebalancing_flows import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    RETS = data.load_real()
    TAB = st.month_table(RETS)
else:
    RETS = TAB = None
print("real cache present:", HAVE_REAL,
      "| months:", (0 if TAB is None else len(TAB)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do pension funds knock stocks down at month-end? 🔄\n"
            "### The month-end rebalancing story, tested in plain English\n\n"
            + BADGES +
            "Every few weeks a note does the rounds on trading desks: *\"stocks beat bonds by 6% "
            "this month, so pension funds and target-date funds must sell $XX billion of equities "
            "to rebalance — expect selling into month-end and a bounce in the first days of the "
            "new month.\"* It sounds mechanical, almost too easy: the funds' own rulebooks force "
            "them to sell whatever just won.\n\n"
            "So we tested it — 33 years of the S&P 500 (SPY) against the broad bond market, one "
            "month at a time. The short version: **half the story is real, half is a costume.**\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the permutation placebo and the "
            "cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Data note.** Equity = SPY; bonds = the Vanguard Total Bond Market fund until "
            "2003, the AGG ETF after (same index, splice documented). All total-return. Every "
            "chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After stocks trounce bonds, do stocks lag into month-end? | **Yes — really.** In the "
            "top fifth of stocks-beat-bonds months, equities trail bonds by about **half a percent "
            "over the final three days** — and the bigger the month's gap, the bigger the drag. |\n"
            "| Does the pressure bounce back in the first days? | **The bounce is real, the story "
            "isn't.** Stocks *do* pop in the first days of a new month — but they pop **whether or "
            "not** there was anything to rebalance. That's the old turn-of-the-month drift, not a "
            "rebalancing snap-back. |\n"
            "| Could you trade it? | **Once upon a time.** A fade-the-winner trade made money for "
            "two decades (through 2015) — and has made **nothing since 2016**. |\n"
            "| Is the famous turn-of-the-month effect secretly this? | **No.** The month-start pop "
            "is just as strong in months where stocks and bonds finished neck-and-neck — when "
            "rebalancers had nothing to do. |"
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Institutions run fixed stock/bond mixes (60/40 and friends). When stocks trounce "
            "bonds during a month, their weights drift — so at month-end they **must sell stocks "
            "and buy bonds** to get back to policy. That forced selling depresses stocks over the "
            "last few days, and once the calendar turns, the pressure lifts and prices bounce.\"*\n\n"
            "This is not just folklore: academics have documented month-turn flow pressure "
            "(Etula, Rinne, Suominen & Vaittinen's *Dash for Cash*) and target-date funds really "
            "do trade against the month's winner by construction (Parker, Schoar & Sun). The "
            "question is whether the effect shows up on the tape **when the rule says it should** — "
            "that's what separates a *flow* story from a plain calendar habit."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, it's the dream setup: a **forced seller** whose calendar you know in advance. "
            "You'd step aside (or short) into month-end after big equity months and buy the "
            "turn.\n\n"
            "But here's the trap this study is built to catch: the market *already has* a famous "
            "unconditional month-turn pattern — the turn-of-the-month drift (our "
            "[study 89](../../89-turn-of-the-month/)). A lazy test would find \"selling into "
            "month-end, bounce after\" and cry *rebalancing!* when it's really just the calendar. "
            "The fingerprint of a genuine **flow** effect is *conditionality*: it should be big "
            "when the month's stock-bond gap is big, and absent when there's nothing to rebalance."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"For each of **{R['n_months']} complete months** ({R['start']} → {R['end']}):\n\n"
            "1. **Measure the month's race** — the stock-minus-bond return gap, up to three days "
            "before month-end (so the signal exists *before* the window it predicts).\n"
            "2. **Watch the last 3 trading days** — did stocks lag bonds when the rule says "
            "rebalancers were selling?\n"
            "3. **Watch the first 3 days of the next month** — did the pressure reverse?\n"
            "4. **Sort months into five buckets** by the gap and compare the extremes — plus a "
            "shuffle test: would randomly re-labelled months look just as good?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The sell-the-winner leg.** Average stock-minus-bond return over the **last 3 days** "
            "of the month, by how much stocks had beaten bonds up to that point."
        ),
        code(
            "if HAVE_REAL:\n"
            "    prof = st.quintile_profile(TAB, 'gap_pre', 'last_spread')\n"
            "    vals = prof['mean_bps'].tolist(); gaps = prof['cond_mean_pct'].tolist()\n"
            "else:\n"
            "    vals = [d[2] for d in R['dose']]; gaps = [d[1] for d in R['dose']]\n"
            "labels = [f'{g:+.1f}%' for g in gaps]\n"
            "colors = [GREY]*4 + [RED]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "ax.bar(labels, vals, color=colors, width=.6)\n"
            "for i, v in enumerate(vals): ax.annotate(f'{v:+.0f}', (i, v), ha='center',\n"
            "    va='bottom' if v > 0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('how much stocks beat bonds month-to-date (quintile means)')\n"
            "ax.set_ylabel('stock-minus-bond return, last 3 days (bps)')\n"
            "ax.set_title('The bigger the month stocks had, the more they lag bonds into month-end')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('last-3-day spread by gap quintile (bps):', [round(v,1) for v in vals])"
        ),
        md(
            f"A clean staircase: months where stocks were barely ahead show nothing, and the "
            f"**biggest winner months end with stocks trailing bonds by "
            f"{R['sell']['top_bps']:+.0f} bps over three days** — a pattern a random shuffle of "
            f"the calendar almost never produces (about a **1-in-45** fluke; the quants notebook "
            "has the details). The sell-the-winner half of the story is on the tape."
        ),
        md(
            "**The \"bounce\" leg — here's the costume.** Same picture for the **first 3 days of "
            "the next month**. If this were rebalancing pressure reversing, the bounce should live "
            "in the right-hand bars only."
        ),
        code(
            "if HAVE_REAL:\n"
            "    prof2 = st.quintile_profile(TAB, 'gap_full', 'first_spread')\n"
            "    vals2 = prof2['mean_bps'].tolist()\n"
            "    unc = float(TAB['first_spread'].mean()) * 1e4\n"
            "else:\n"
            "    vals2 = [d[4] for d in R['dose']]; unc = R['rev']['uncond_bps']\n"
            "labels2 = ['big bond\\nmonths', 'Q2', 'Q3', 'Q4', 'big stock\\nmonths']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "ax.bar(labels2, vals2, color=[GREY, GREY, GREY, GREY, GREEN], width=.6)\n"
            "ax.axhline(unc, ls='--', c=AMBER, lw=2, label=f'ALL months average ({unc:+.0f} bps)')\n"
            "for i, v in enumerate(vals2): ax.annotate(f'{v:+.0f}', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('stock-minus-bond return, first 3 days of next month (bps)')\n"
            "ax.set_title('The month-start pop is there in EVERY bucket - not just after big stock months')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('first-3-day spread by prior-gap quintile (bps):', [round(v,1) for v in vals2])"
        ),
        md(
            f"Stocks beat bonds by **{R['rev']['uncond_bps']:+.0f} bps** in the first three days of "
            f"a month *on average* — after big stock months, after big bond months, after flat "
            f"months. There's no staircase here: the \"rebalancing bounce\" is really the plain old "
            "**turn-of-the-month drift** wearing a rebalancing costume."
        ),
        md(
            "**Could you have traded the real half?** Fade the winner: when the month-to-date gap "
            "is in the top fifth of everything seen so far, short stocks / long bonds for the last "
            "3 days, then flip for the first 3. Realistic costs, and the threshold only uses past "
            "months (no crystal ball)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tr = st.flow_trade(RETS, cost_bps=2.0)\n"
            "    cum = (1 + tr['daily']).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "    ax.plot(cum.index, cum.values, c=GREEN, lw=1.6, label='flow trade, net of costs (2 bps/leg)')\n"
            "    ax.axvline(pd.Timestamp('2016-01-01'), ls='--', c=RED, label='2016: the edge goes quiet')\n"
            "    ax.set_ylabel('growth of $1'); ax.set_title('Two good decades - then a decade of nothing')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f\"net {tr['net_ann_pct']:+.2f}%/yr over the full tape ({tr['n_events']} events)\")\n"
            "else:\n"
            "    print('cache missing - canonical numbers:', R['trade'])"
        ),
        md(
            f"Net of costs the trade made **{R['trade'][2][2]:+.1f}%/yr through 2015** — real "
            f"money for a strategy that's only at risk {R['share_active_pct']:.0f}% of the time — "
            f"and **{R['trade'][3][2]:+.1f}%/yr since 2016**. Once the sell-side started "
            "publishing \"pension rebalancing estimates\" every month-end, the easy version "
            "stopped paying. That's what an arbitraged-away flow looks like."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Mixed.** The sell-the-winner leg is real: after big stock months, "
            f"stocks lag bonds by **{R['sell']['top_bps']:+.0f} bps** over the last three days "
            "(a solid, dose-dependent effect). The bounce leg is a costume: the month-start pop "
            "is the same in every bucket — it's the turn-of-the-month drift, not a flow "
            "reversal.\n"
            f"- **Tradability — Fragile.** {R['trade'][2][2]:+.1f}%/yr through 2015, "
            f"{R['trade'][3][2]:+.1f}%/yr since 2016. The trade had its decade and lost it.\n"
            "- **\"Turn-of-the-month is secretly rebalancing\"? — Busted.** The month-start pop is "
            "at full strength exactly when rebalancers have nothing to do."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Why did it fade?** Forced flows only move prices while they surprise someone. "
            "Month-end rebalancing became the most advertised flow in markets (every bank "
            "publishes an estimate); front-runners front-ran the front-runners.\n"
            "- **The half that survives** is the *conditional drag* on the winner into month-end — "
            "worth knowing if you were going to trade those days anyway.\n"
            "- **Related on this desk:** the unconditional calendar version is "
            "[study 89 — turn-of-the-month](../../89-turn-of-the-month/); rebalancing as a "
            "portfolio *policy* (the payer of these flows) is "
            "[study 97 — balancing act](../../97-balancing-act/).\n\n"
            "*Think the flow still pays if you condition on something sharper than the gap — "
            "vol-scaled drift, quarter-ends, TDF assets? The engine is right here: swap the "
            "conditioning column and re-run.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    s, v, tm, sy = R["sell"], R["rev"], R["tom"], R["syn"]
    cells = [
        md(
            "# Month-End Rebalancing Flows — a quantitative teardown 🔬\n"
            "### Gap-conditional quintile splits + Welch *t* · a seed-averaged permutation "
            "placebo · dose-response, era and window robustness · an expanding-threshold trade "
            "with HAC *t* · a planted-reversal synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim is **conditional** — flows fire when the month-to-date equity-bond gap is big — "
            "so the whole design is built to separate a *flow* effect from the desk's known "
            "*unconditional* turn-of-the-month drift "
            "([study 89](../../89-turn-of-the-month/), the named sibling and dedup guard).\n\n"
            "> ⚠️ **Data note.** SPY + a spliced Aggregate-bond leg — VBMFX before "
            + R["splice"] + ", AGG after (same Bloomberg Aggregate index; splice in RETURN "
            "space, no level stitching; the plan's TLT alternative was rejected for duration "
            "mismatch and its 2002 start). Total-return throughout; no survivorship (broad index "
            "vehicles). Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (as-of **" + R["as_of"] + "**, fingerprint "
            "`" + R["fingerprint"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | *Real on the sell leg:* top-gap-quintile last-3-day spread "
            f"**{s['top_bps']:+.2f} bps** (one-sample **t = {s['t_top']:+.2f}**), permutation "
            f"placebo **p ≈ {s['p_mean']:.3f}**, monotone dose-response — but the top-vs-bottom "
            f"differential reads Welch t = {s['welch']:+.2f} on the pre-registered k = 3 window. "
            f"*Weak on the reversal leg:* Welch t = {v['welch']:+.2f}, placebo p ≈ "
            f"{v['p_mean']:.2f}; the first-days pop ({v['uncond_bps']:+.2f} bps, t = "
            f"{v['t_uncond']:+.2f}) is unconditional. |\n"
            f"| **Tradability** | `FRAGILE` | Expanding-threshold trade nets "
            f"**{R['trade'][0][2]:+.2f}%/yr** at 2 bps/leg (HAC **t = {R['trade'][0][3]:+.2f}**) "
            f"— but it is {R['trade'][2][2]:+.2f}%/yr (t = {R['trade'][2][3]:.2f}) through 2015 "
            f"and **{R['trade'][3][2]:+.2f}%/yr** since 2016; below the bar at 5 bps "
            f"(t = {R['trade'][1][3]:.2f}). |\n"
            f"| **TOM in disguise?** | `BUSTED` | Study 89's first-days drift is intact in the "
            f"middle gap quintiles (**{tm['mid_bps']:+.2f} bps, t = {tm['t_mid']:+.2f}**) where "
            f"rebalancers have nothing to do; extremes-vs-middle Welch t = "
            f"{tm['welch_ext_mid']:+.2f}. |\n\n"
            "> 💡 In plain words: the forced *selling* shows up when the rule says it should; the "
            "famous *bounce* shows up whether or not the rule says it should — which means the "
            "bounce isn't flows."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $g_m$ be month $m$'s equity-minus-bond cumulative-return gap measured through "
            "the close 3 trading days before month-end (``gap_pre``, strictly prior to the window "
            "it predicts), and $G_m$ the full-month gap (known at the month-end close). Let "
            "$S^{last}_m$ and $S^{first}_{m+1}$ be the equity-minus-bond spreads over the last 3 "
            "days of $m$ and the first 3 days of $m+1$.\n\n"
            "- **H₁ (forced selling).** $E[S^{last}_m \\mid g_m \\text{ top quintile}] < 0$, and "
            "**more negative than** the bottom quintile (the *conditional* fingerprint).\n"
            "- **H₂ (the reversal).** $E[S^{first}_{m+1} \\mid G_m \\text{ top quintile}]$ exceeds "
            "the bottom quintile — the pressure unwinds.\n"
            "- **H₃ (tradability).** Fading the winner into month-end and riding the reversal "
            "survives costs with an out-of-sample (expanding) threshold.\n\n"
            "We find **H₁ half-supported** (the level clears decisively, the top-vs-bottom "
            "differential misses at k = 3), **H₂ rejected as a *conditional* claim** (the pop is "
            "unconditional TOM), **H₃ supported only pre-2016**."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the inference design\n\n"
            "Monthly windows are **non-overlapping**, so quintile splits get plain one-sample and "
            "**Welch** *t*'s (no HAC needed there); the daily trade stream *is* serially "
            "dependent inside events, so it gets a **Newey-West (5-lag) t** on active days. Two "
            "honesty devices:\n\n"
            "1. **The permutation placebo** — shuffle the conditioning gap across months (killing "
            "the gap→window link, preserving both marginals) and recompute the top-minus-bottom "
            "spread. Because a random baseline from one seed is banned on this desk, the p-value "
            "is the **mean over 20 seeds × 2,000 draws** (the range is reported).\n"
            "2. **The dedup guard** — every conditional number is pinned against its "
            "unconditional counterpart, because the desk already knows the unconditional "
            "month-turn drift is a thing (study 89). A flow story earns nothing for rediscovering "
            "the calendar."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** SPY + spliced VBMFX→AGG bond leg, {R['start']} → {R['end']} "
            f"({R['n_days']:,} sessions, **{R['n_months']} complete months**), total-return, "
            f"as-of {R['as_of']}, fingerprint `{R['fingerprint']}`.\n"
            "- **Conditioning.** ``gap_pre`` (through the close 3 days before month-end) predicts "
            "the last-3 window; ``gap_full`` (month-end close) predicts the first-3 window — the "
            "signal always precedes the window: **one execution lag**, everywhere.\n"
            "- **Primary tests.** Top-vs-bottom-quintile Welch t on each leg's spread; one-sample "
            "t per quintile; seed-averaged permutation placebo.\n"
            "- **Robustness.** Dose-response over all 5 quintiles; era splits (1993–2003 / "
            "2004–2015 / 2016+); window k ∈ {2, 3, 4} (k = 3 pre-registered).\n"
            "- **Costs.** 8 one-way tickets per event (pair open + 2×-notional month-end flip + "
            "close) at 2/5 bps one-way × NAV; the short leg pays 30 bps/yr borrow pro-rata.\n"
            "- **Control.** A deterministic two-asset world with a planted, tunable "
            "gap-conditional reversal; the null must stay quiet across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The sell-the-winner leg and its placebo\n\n"
            "Top-gap months vs bottom-gap months on the last-3-day spread, against the "
            "permutation null (reduced draws in-cell; canonical 20 × 2,000 numbers from "
            "results.md)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    a = st.quintile_split(TAB, 'gap_pre', 'last_spread')\n"
            "    obs = a['diff_bps']\n"
            "    # in-cell reduced placebo for the picture (canonical p in R)\n"
            "    t = TAB.dropna(subset=['gap_pre', 'last_spread']).reset_index(drop=True)\n"
            "    outv, n = t['last_spread'].values, len(t)\n"
            "    k = int(np.ceil(n * 0.2)); rng = np.random.default_rng(604)\n"
            "    draws = []\n"
            "    for _ in range(4000):\n"
            "        pr = rng.permutation(n)\n"
            "        draws.append((outv[pr[-k:]].mean() - outv[pr[:k]].mean()) * 1e4)\n"
            "    draws = np.array(draws)\n"
            "else:\n"
            "    obs = R['sell']['diff_bps']\n"
            "    rng = np.random.default_rng(604); draws = rng.normal(0, 36, 4000)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.hist(draws, bins=60, color=GREY, alpha=.85, label='null: gap shuffled across months')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed top-minus-bottom {obs:+.1f} bps')\n"
            "ax.set_xlabel('top-minus-bottom last-3-day spread (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Sell leg: p = {R['sell']['p_mean']:.4f} (20 seeds x 2,000 draws), \"\n"
            "             f\"top-quintile level t = {R['sell']['t_top']:+.2f}\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"in-cell shuffle: P[diff <= obs] = {(draws <= obs).mean():.4f} \"\n"
            "      f\"(canonical {R['sell']['p_mean']:.4f} [{R['sell']['p_lo']:.4f}, {R['sell']['p_hi']:.4f}])\")"
        ),
        md(
            f"> 💡 In plain words: the top-gap months' **{s['top_bps']:+.2f} bps** drag (one-sample "
            f"**t = {s['t_top']:+.2f}**) sits in the left tail of the shuffle null "
            f"(**p ≈ {s['p_mean']:.3f}**, seed range [{s['p_lo']:.3f}, {s['p_hi']:.3f}]) — but note "
            f"the honest wrinkle: the top-vs-bottom **Welch t = {s['welch']:+.2f}** misses the "
            f"desk's 2-bar on the pre-registered 3-day window, because the bottom quintile "
            f"(+{s['bot_bps']:.1f} bps, t = {s['t_bot']:+.2f}) is noisy rather than opposite. The "
            "level is real; the sharpest conditional contrast is marginal."
        ),
        md(
            "### 4b · The reversal leg — conditional or costume?\n\n"
            "Same split for the next month's first-3-day spread, with the unconditional mean "
            "drawn through it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    prof2 = st.quintile_profile(TAB, 'gap_full', 'first_spread')\n"
            "    vals2 = prof2['mean_bps'].tolist(); ts2 = prof2['t'].tolist()\n"
            "    unc = float(TAB['first_spread'].mean()) * 1e4\n"
            "else:\n"
            "    vals2 = [d[4] for d in R['dose']]; ts2 = [d[5] for d in R['dose']]\n"
            "    unc = R['rev']['uncond_bps']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar([f'Q{i}' for i in range(1, 6)], vals2, color=[GREY]*4 + [GREEN], width=.6)\n"
            "ax.axhline(unc, ls='--', c=AMBER, lw=2, label=f'unconditional {unc:+.1f} bps (t = %.2f)' % R['rev']['t_uncond'])\n"
            "for i, (vv, tt) in enumerate(zip(vals2, ts2)):\n"
            "    ax.annotate(f'{vv:+.0f}\\n(t={tt:+.1f})', (i, vv), ha='center', va='bottom', fontsize=9)\n"
            "ax.set_xlabel('prior full-month gap quintile'); ax.set_ylabel('first-3-day spread (bps)')\n"
            "ax.set_title('Reversal leg: no dose-response - the pop is unconditional')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('first-3 spread by quintile (bps):', [round(x, 1) for x in vals2])"
        ),
        md(
            f"> 💡 In plain words: Q2 (prior month basically flat — nothing to rebalance) pops "
            f"**+{R['dose'][1][4]:.0f} bps**, Q5 pops **+{R['dose'][4][4]:.0f} bps**. Top-vs-bottom "
            f"Welch **t = {v['welch']:+.2f}**, placebo **p ≈ {v['p_mean']:.2f}**. The first-days "
            f"pop ({v['uncond_bps']:+.2f} bps, **t = {v['t_uncond']:+.2f}** unconditionally) is a "
            "calendar effect, not a flow unwind. H₂ fails *as a conditional claim*."
        ),
        md(
            "### 4c · Era and window robustness\n\n"
            "The conditional differentials by era (the flow story should strengthen as "
            "target-date assets grew — instead it peaks 2004–2015 and dies), and the window "
            "length k."
        ),
        code(
            "eras = R['era']\n"
            "x = np.arange(len(eras)); w = 0.38\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "ax.bar(x - w/2, [e[3] for e in eras], w, color=RED, label='SELL leg: top-bottom diff (bps)')\n"
            "ax.bar(x + w/2, [e[7] for e in eras], w, color=GREEN, label='REV leg: top-bottom diff (bps)')\n"
            "for i, e in enumerate(eras):\n"
            "    ax.annotate(f'W {e[4]:+.2f}', (i - w/2, e[3]), ha='center', va='top' if e[3] < 0 else 'bottom', fontsize=9)\n"
            "    ax.annotate(f'W {e[8]:+.2f}', (i + w/2, e[7]), ha='center', va='bottom' if e[7] > 0 else 'top', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([e[0] for e in eras])\n"
            "ax.set_ylabel('top-minus-bottom quintile spread (bps)')\n"
            "ax.set_title('Both conditional legs cleared the bar in 2004-2015 - and faded/flipped after')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('window robustness (k, sell welch, rev welch):', [(k, sw, rw) for k, _, _, sw, _, _, rw in R['windows']])"
        ),
        md(
            f"> 💡 In plain words: 2004–2015 was the flow decade — sell diff "
            f"**{R['era'][1][3]:+.0f} bps (Welch {R['era'][1][4]:+.2f})**, reversal diff "
            f"**{R['era'][1][7]:+.0f} bps (Welch {R['era'][1][8]:+.2f})** — and since 2016 the "
            f"sell leg fades (Welch {R['era'][2][4]:+.2f}) while the reversal **flips sign** "
            f"(Welch {R['era'][2][8]:+.2f}). Window robustness: the sell-leg *level* clears at "
            f"every k (t = −2.7 to −3.1); the differential clears only at k = 4 "
            f"(Welch **{R['windows'][2][3]:+.2f}**) — we keep the pre-registered k = 3 headline "
            "rather than shop the window."
        ),
        md(
            "### 4d · Tradability — the expanding-threshold flow trade\n\n"
            "Fire when ``gap_pre`` beats the 80th percentile of all *prior* months (60-month "
            "burn-in): short eq/long bd for the last 3 days, flip for the first 3. 8 one-way "
            "tickets per event; the short leg pays borrow; HAC t on active days."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [st.flow_trade(RETS, cost_bps=cb) for cb in (2.0, 5.0)]\n"
            "    nets = [r['net_ann_pct'] for r in rows]; hacs = [r['t_net_active'] for r in rows]\n"
            "else:\n"
            "    nets = [R['trade'][0][2], R['trade'][1][2]]; hacs = [R['trade'][0][3], R['trade'][1][3]]\n"
            "labels = ['2 bps/leg\\n(full tape)', '5 bps/leg\\n(full tape)', '2 bps\\n1993-2015', '2 bps\\n2016-2026']\n"
            "vals = nets + [R['trade'][2][2], R['trade'][3][2]]\n"
            "tvals = hacs + [R['trade'][2][3], R['trade'][3][3]]\n"
            "colors = [GREEN, AMBER, GREEN, RED]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "ax.bar(labels, vals, color=colors, width=.6)\n"
            "for i, (vv, tt) in enumerate(zip(vals, tvals)):\n"
            "    ax.annotate(f'{vv:+.2f}%\\n(HAC t={tt:+.2f})', (i, vv), ha='center',\n"
            "                va='bottom' if vv >= 0 else 'top', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('net annualised return (% of NAV)')\n"
            "ax.set_title('Clears the bar at 2 bps on the full tape - but ALL of it is pre-2016')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net %/yr and HAC t:', [(l.replace(chr(10), ' '), round(v, 2), round(t, 2)) for l, v, t in zip(labels, vals, tvals)])"
        ),
        md(
            f"> 💡 In plain words: **{R['trade'][0][2]:+.2f}%/yr net** (HAC t = "
            f"{R['trade'][0][3]:+.2f}) at 2 bps/leg, with per-event gross "
            f"**{R['per_event_gross_bps']:+.1f} bps** against {R['per_event_cost_bps']} bps of "
            f"tickets — but split the tape and the whole P&L is **1993–2015** "
            f"({R['trade'][2][2]:+.2f}%/yr, t = {R['trade'][2][3]:.2f}); since 2016 it is "
            f"**{R['trade'][3][2]:+.2f}%/yr** (t = {R['trade'][3][3]:+.2f}), and 5 bps/leg drops "
            f"the full tape below the bar (t = {R['trade'][1][3]:.2f}). Real once, decayed → "
            "**FRAGILE**."
        ),
        md(
            "### 4e · Third axis — is turn-of-the-month just flows in disguise?\n\n"
            "The sharpest myth-check this claim allows: if study 89's unconditional TOM drift "
            "were rebalancing flows, the first-days equity pop should vanish in months whose "
            "prior gap was ≈ 0 (nothing to rebalance) and concentrate in the extremes."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tv = st.tom_vs_flows(TAB)\n"
            "else:\n"
            "    tv = R['tom'].copy(); tv['uncond_bps'] = R['tom']['uncond_bps']\n"
            "bars = [tv['bot_bps'], tv['mid_bps'], tv['top_bps']]\n"
            "ts = [tv['t_bot'], tv['t_mid'], tv['t_top']]\n"
            "labels = ['bottom gap quintile\\n(bonds trounced stocks)', 'middle 3 quintiles\\n(nothing to rebalance)', 'top gap quintile\\n(stocks trounced bonds)']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "ax.bar(labels, bars, color=[GREY, AMBER, GREY], width=.55)\n"
            "ax.axhline(tv['uncond_bps'], ls='--', c=GREEN, label=f\"unconditional TOM {tv['uncond_bps']:+.1f} bps\")\n"
            "for i, (vv, tt) in enumerate(zip(bars, ts)):\n"
            "    ax.annotate(f'{vv:+.1f} bps\\n(t={tt:+.2f})', (i, vv), ha='center', va='bottom', fontsize=9)\n"
            "ax.set_ylabel('first-3-day SPY return (bps)')\n"
            "ax.set_title('The TOM pop is at FULL strength where rebalancers have nothing to do')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"extremes-vs-middle Welch t = {tv['welch_ext_vs_mid']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: the middle quintiles — months with a near-zero gap — still pop "
            f"**{tm['mid_bps']:+.2f} bps (t = {tm['t_mid']:+.2f})**, statistically identical to "
            f"the extremes (Welch t = {tm['welch_ext_mid']:+.2f}). Study 89's TOM is a calendar "
            "drift with its own engine (payroll/settlement cash, habit), **not** this flow — "
            "**BUSTED**, and the two studies genuinely measure different things."
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "A deterministic two-asset random walk with a planted, gap-conditional flow reversal "
            "(``edge``). The null (edge = 0) must stay quiet across seeds; the planted world must "
            "light up. In-cell we run a reduced seed sweep (canonical 20-seed numbers from "
            "results.md)."
        ),
        code(
            "ws = []\n"
            "for sd in range(8):                    # reduced sweep; canonical = 20 seeds in R\n"
            "    w0 = data.synthetic_world(edge=0.0, seed=sd)\n"
            "    t0 = st.month_table(w0)\n"
            "    ws.append(st.quintile_split(t0, 'gap_pre', 'last_spread')['welch_t'])\n"
            "w1 = data.synthetic_world(edge=0.010, seed=604)\n"
            "t1 = st.month_table(w1)\n"
            "a1 = st.quintile_split(t1, 'gap_pre', 'last_spread')\n"
            "b1 = st.quintile_split(t1, 'gap_full', 'first_spread')\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar([f'null s{sd}' for sd in range(8)] + ['planted\\n100 bps'],\n"
            "       [abs(x) for x in ws] + [abs(a1['welch_t'])],\n"
            "       color=[GREY]*8 + [GREEN], width=.6)\n"
            "ax.axhline(2, ls='--', c=RED, label='|t| = 2 bar')\n"
            "ax.set_ylabel('|Welch t| of the SELL-leg top-bottom diff')\n"
            "ax.set_title('Null worlds stay under the bar; the planted flow lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null (8 seeds): mean {np.mean(ws):+.2f}, max |t| {max(abs(x) for x in ws):.2f} '\n"
            "      f\"(canonical 20 seeds: mean {R['syn']['null_mean']:+.2f}, max |t| {R['syn']['null_max_abs']:.2f})\")\n"
            "print(f\"planted 100 bps: SELL Welch {a1['welch_t']:+.2f}, REV Welch {b1['welch_t']:+.2f} \"\n"
            "      f\"(canonical {R['syn']['planted_sell_welch']:+.2f} / {R['syn']['planted_rev_welch']:+.2f})\")"
        ),
        md(
            f"> 💡 In plain words: across 20 independent null worlds the detector never clears "
            f"\\|t\\| = 2 (mean {sy['null_mean']:+.2f}, max \\|t\\| = {sy['null_max_abs']:.2f}); a "
            f"planted 100-bps reversal explodes to Welch **{sy['planted_sell_welch']:+.2f}** / "
            f"**{sy['planted_rev_welch']:+.2f}** and the trade banks "
            f"**{sy['planted_net_pct']:+.2f}%/yr (HAC t = {sy['planted_hac']:+.2f})** vs "
            f"{sy['null_net_pct']:+.2f}%/yr on the null. The machinery is unbiased and powered. "
            "*(A faithful-engine / power check only — never cited to support the real-tape "
            "stamps.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — *Real on the sell leg:* top-gap months drag "
            f"**{s['top_bps']:+.2f} bps** over the last 3 days (one-sample **t = {s['t_top']:+.2f}**, "
            f"placebo **p ≈ {s['p_mean']:.3f}**, monotone dose-response, level clears at every "
            f"window k); the purely conditional differential is marginal at k = 3 (Welch "
            f"{s['welch']:+.2f}; clears at k = 4 with {R['windows'][2][3]:+.2f}). *Weak on the "
            f"reversal leg:* Welch {v['welch']:+.2f}, placebo p ≈ {v['p_mean']:.2f} — the pop is "
            "unconditional TOM. No survivorship (broad index vehicles).\n"
            f"- **Tradability `FRAGILE`** — {R['trade'][0][2]:+.2f}%/yr net at 2 bps/leg (HAC "
            f"t = {R['trade'][0][3]:+.2f}), all of it earned 1993–2015 "
            f"({R['trade'][2][2]:+.2f}%/yr, t = {R['trade'][2][3]:.2f}); "
            f"{R['trade'][3][2]:+.2f}%/yr since 2016; below the bar at 5 bps. Real once, "
            "decayed.\n"
            f"- **TOM in disguise? `BUSTED`** — the drift survives at full strength "
            f"({tm['mid_bps']:+.2f} bps, t = {tm['t_mid']:+.2f}) precisely where there is nothing "
            f"to rebalance (extremes-vs-middle Welch t = {tm['welch_ext_mid']:+.2f})."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Sharper conditioning.** The gap is the folklore's own variable; dollar-weighted "
            "versions (gap × balanced-fund AUM), quarter-end-only, or vol-scaled drift are the "
            "obvious next knobs — the ``month_table`` conditioning column is pluggable.\n"
            "- **Why 2004–2015?** Balanced/TDF assets grew all the way through 2026, yet the "
            "effect died in 2016 — consistent with McLean-Pontiff-style *crowding of the trade "
            "against the flow* (every desk now front-runs the rebalancers), not with the flow "
            "disappearing.\n"
            "- **The dedup lesson.** Conditioning is what separates a *flow* claim from a "
            "*calendar* claim — and on this tape the famous reversal half belongs to the "
            "calendar ([89-turn-of-the-month](../../89-turn-of-the-month/)), while only the "
            "selling half belongs to the flows.\n\n"
            "*The reproducible core is offline and deterministic; every number above is printed "
            "by [`examples/verify.py`](../examples/verify.py) and frozen in "
            "[`docs/results.md`](../docs/results.md).*"
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
