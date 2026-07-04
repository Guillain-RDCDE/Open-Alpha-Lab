"""Generate the two narrative notebooks for Study 602 (Macro-Announcement-Day Premium).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY/TLT
closes under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic positive control runs anywhere with no network.
Heavy draws are reduced in-notebook (canonical numbers are quoted from ``R``).
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


# Frozen real-tape headline numbers — mirror of docs/results.md (SPY+TLT yfinance,
# 1997-01-02 -> 2026-06-30, hardcoded FOMC/CPI/NFP calendar). As-of 2026-06-30.
R = dict(
    start="1997-01-02", end="2026-06-30", years=29.5, n_days=7419,
    n_fomc=236, n_cpi=353, n_nfp=353, n_adays=923, frac_days=12.4,
    n_overlap=19, n_mapped=10,
    a_mean_bps=10.63, rest_mean_bps=3.69, diff_bps=6.94, welch_t=1.57,
    a_t0=2.55, share_return=32.1, p_placebo=0.0563,
    # per type: (name, n, A-mean bps, diff vs pure rest bps, welch t)
    types=[("FOMC", 236, 22.20, 18.52, 2.25),
           ("CPI", 353, 4.55, 0.87, 0.12),
           ("NFP", 353, 13.24, 9.56, 1.38)],
    # decades: (label, A-mean, rest, diff, welch t, n_A)
    decades=[("1997-2006", 13.92, 2.46, 11.46, 1.51, 315),
             ("2007-2016", 14.05, 2.01, 12.04, 1.56, 312),
             ("2017-2026", 3.53, 6.75, -3.22, -0.42, 296)],
    tlt=dict(a=-0.56, rest=2.18, diff=-2.74, t=-0.68, n=749, days=6016),
    # overlay: (one-way bps, gross bps/A-day, net bps/A-day, net t0, net ann %, bh ann %)
    overlay=[(1.0, 10.63, 8.63, 2.07, 2.48, 10.05),
             (2.0, 10.63, 6.63, 1.59, 1.84, 10.05),
             (5.0, 10.63, 0.63, 0.15, -0.05, 10.05)],
    exfomc=dict(mean=6.65, diff=2.97, t=0.58, n=687, p=0.3175),
    fomc_leg=dict(mean=22.20, diff=18.52, t=2.25, n=236),
    # FOMC engine by decade: (label, diff bps, welch t, n)
    fomc_decades=[("1997-2006", 14.28, 1.21, 80),
                  ("2007-2016", 45.29, 2.90, 80),
                  ("2017-2026", -6.09, -0.41, 76)],
    # synthetic sweep (100 seeds): (edge bps/day, mean diff, mean t, share |t|>=2 %)
    syn=[(0.0, 0.45, 0.10, 4), (20.0, 20.45, 4.72, 100)],
    fingerprint="e42015f7ee3e",
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![All_just_FOMC%3F: Confirmed](https://img.shields.io/badge/All_just_FOMC%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from macro_announcement_premium import data, strategy as st

START, ASOF = "1997-01-01", "2026-06-30"
HAVE_REAL = data.have_real()
if HAVE_REAL:
    PX = data.load_real()
    PX = PX[(PX.index >= "1996-12-01") & (PX.index <= ASOF)]
    RET = st.daily_returns(PX["SPY"].dropna())
    RET = RET[RET.index >= START]
    MASKS = data.announcement_masks(RET.index)
else:
    PX = RET = MASKS = None
print("real cache present:", HAVE_REAL,
      "| daily returns:", (0 if RET is None else len(RET)),
      "| A-days:", (0 if MASKS is None else int(MASKS["ANY"].sum())))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Is the stock market's whole reward earned on a few macro-news days? 📅\n"
            "### The Savor-Wilson announcement-day premium, retried on the modern tape — in plain English\n\n"
            + BADGES +
            "Every month, on days everyone knows about **months in advance**, the US government "
            "publishes its biggest economic numbers: the **CPI** (inflation), the **jobs report** "
            "(\"NFP\"), and the Fed's **FOMC decision**. A famous 2013 academic paper (Savor & Wilson) "
            "claimed something remarkable: **most of the stock market's entire long-run reward is "
            "earned on precisely those days** — about one day in eight.\n\n"
            "If that's true, it sounds like the cheat code of the century: be invested on ~31 "
            "scheduled days a year, sit in cash the rest, sleep well. We rebuilt the whole thing on "
            "29.5 years of real prices with the actual, dated release calendar. Short version: the "
            "silhouette of the effect is there — but the honest version doesn't survive the audit, "
            "and the trade definitely doesn't.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** Real data: SPY (and TLT) total-return closes, 1997→2026, against "
            "a hardcoded calendar of **actual** FOMC/CPI/NFP release dates (sources in "
            "[`data.py`](../macro_announcement_premium/data.py)). Every chart is drawn by the code "
            "beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do announcement days earn more? | **On average yes** — about **+11 bps** vs **+4 bps** "
            "on ordinary days, and one-eighth of the days carried a **third** of the market's "
            "cumulative return. But that gap is **not statistically solid** (it could plausibly be "
            "luck — the quants notebook puts it right at the edge, *p* ≈ 0.06). |\n"
            "| Is it CPI? Jobs day? The Fed? | **It's the Fed. Full stop.** Take the FOMC days out "
            "and the remaining CPI/jobs days earn basically the same as any other day. |\n"
            "| Is it still alive? | **Not since 2017.** The gap was healthy 1997-2016 and has been "
            "slightly *negative* for the last decade. |\n"
            "| Can I get rich holding stocks only on those days? | **No.** Even at near-zero trading "
            "costs the \"announcement days only\" strategy makes ~2.5%/yr while just holding the "
            "index made ~10%/yr — and at normal retail costs it makes **nothing**. |\n\n"
            "> The honest verdict: a **Weak** signal (real-looking silhouette, below the "
            "significance bar, all of it Fed days, faded since 2017) and a **Mirage** to trade."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Stocks earn their risk premium when investors bear the scariest risk — and the "
            "scariest, most concentrated risk arrives on scheduled macro-announcement mornings. So "
            "the equity premium should cluster on CPI, employment and FOMC days.\"*\n\n"
            "That's **Savor & Wilson (2013)**, published in a top journal on 1958-2009 data: about "
            "**+11.4 bps** on announcement days vs **+1.1 bps** on other days. The pitch is elegant "
            "because the calendar is **public months ahead** — no crystal ball needed. Our question: "
            "does it hold on the modern, tradable tape (SPY, 1997→2026), and can you eat it?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If most of the equity premium really lands on ~31 pre-known days a year, two big things "
            "follow. For **theory**: risk premia are compensation for *event* risk, not calendar "
            "time — you get paid for holding through the scary mornings. For **practice**: an "
            "investor could hold stocks an eighth of the time and collect most of the reward — the "
            "ultimate low-effort timing strategy.\n\n"
            "Both hinge on the premium being (a) statistically real, (b) spread across the macro "
            "calendar rather than one event type, and (c) big enough to survive trading costs. We "
            "test all three."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We hardcode the **actual release calendar** — {R['n_fomc']} FOMC decision days (Fed "
            f"historical calendars) + {R['n_cpi']} CPI and {R['n_nfp']} jobs-report release dates "
            "(from the BLS's own archive, cross-checked against its official historical-release-"
            "dates table; shutdown-delayed releases and holiday quirks included). That gives "
            f"**{R['n_adays']} announcement sessions** out of {R['n_days']:,} trading days "
            f"({R['frac_days']:.1f}%). Then:\n\n"
            "1. **Compare.** Average SPY return on announcement days vs all other days.\n"
            "2. **Stress it.** Could 923 random days look this special? (20,000 random calendars.)\n"
            "3. **Split it.** By type (FOMC / CPI / NFP) and by decade.\n"
            "4. **Trade it.** Hold SPY *only* on announcement days, entering the night before "
            "(the schedule is public), and charge real costs."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline gap.** Average daily return on announcement days vs the rest."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.event_vs_rest(RET.values, MASKS['ANY'])\n"
            "    am, rm, sh = s['ev_mean_bps'], s['base_mean_bps'], s['share_of_total']*100\n"
            "else:\n"
            "    am, rm, sh = R['a_mean_bps'], R['rest_mean_bps'], R['share_return']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.4))\n"
            "a1.bar(['announcement\\ndays', 'all other\\ndays'], [am, rm], color=[AMBER, GREY], width=.6)\n"
            "for i, v in enumerate([am, rm]): a1.annotate(f'{v:+.1f} bps', (i, v), ha='center', va='bottom')\n"
            "a1.set_ylabel('average return (bps/day)'); a1.set_title('Announcement days do earn more...')\n"
            "a2.bar(['share of\\ntrading days', 'share of\\ntotal return'], [R['frac_days'], sh], color=[GREY, AMBER], width=.6)\n"
            "for i, v in enumerate([R['frac_days'], sh]): a2.annotate(f'{v:.1f}%', (i, v), ha='center', va='bottom')\n"
            "a2.set_ylabel('%'); a2.set_title('...and 1/8 of the days carry 1/3 of the return')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'A-days {am:+.1f} bps/day vs others {rm:+.1f}; A-days = {R[\"frac_days\"]:.1f}% of days, {sh:.1f}% of return')"
        ),
        md(
            f"Looks impressive: **{R['a_mean_bps']:+.1f} bps** on announcement days vs "
            f"**{R['rest_mean_bps']:+.1f} bps** otherwise, and **{R['frac_days']:.1f}%** of the days "
            f"carried **{R['share_return']:.1f}%** of the return. But \"looks impressive\" is where "
            f"most market legends live. The statistical test says this gap is **Welch *t* = "
            f"{R['welch_t']:.2f}** — *below* the desk's bar of 2, with about a **1-in-18** chance "
            "(*p* ≈ 0.056) that 923 random days would look this good. Suggestive, not proven.\n\n"
            "**Now the twist — split it by announcement type.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pure = ~MASKS['ANY']\n"
            "    rows = [(nm, st.event_vs_rest(RET.values, MASKS[nm], base=pure)) for nm in ('FOMC','CPI','NFP')]\n"
            "    labels = [r[0] for r in rows]; diffs = [r[1]['diff_bps'] for r in rows]; ts = [r[1]['welch_t'] for r in rows]\n"
            "else:\n"
            "    labels = [t[0] for t in R['types']]; diffs = [t[3] for t in R['types']]; ts = [t[4] for t in R['types']]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "cols = [GREEN if t >= 2 else GREY for t in ts]\n"
            "ax.bar(labels, diffs, color=cols, width=.55)\n"
            "for i, (d, t) in enumerate(zip(diffs, ts)):\n"
            "    ax.annotate(f'{d:+.1f} bps\\n(t={t:.2f})', (i, max(d, 0)), ha='center', va='bottom')\n"
            "ax.set_ylabel('extra return vs ordinary days (bps/day)')\n"
            "ax.set_title('The whole premium is the Fed: only FOMC days clear the bar')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('diff vs ordinary days:', {l: f'{d:+.1f} bps (t={t:.2f})' for l, d, t in zip(labels, diffs, ts)})"
        ),
        md(
            f"There's the reveal. **FOMC days: {R['types'][0][3]:+.1f} bps** over an ordinary day "
            f"(*t* = {R['types'][0][4]:.2f} — clears the bar). **CPI days: "
            f"{R['types'][1][3]:+.1f} bps** (*t* = {R['types'][1][4]:.2f} — nothing). NFP days lean "
            f"positive but don't clear. The \"macro announcement premium\" is, on this tape, **the "
            "Fed-day premium wearing a bigger calendar**.\n\n"
            "**Is it at least alive?** Same test, by decade."
        ),
        code(
            "if HAVE_REAL:\n"
            "    splits = [('1997-2006','1997-01-01','2006-12-31'), ('2007-2016','2007-01-01','2016-12-31'),\n"
            "              ('2017-2026','2017-01-01','2026-06-30')]\n"
            "    dec = st.subperiod_stats(RET, MASKS['ANY'], splits)\n"
            "    labels = [d['label'] for d in dec]; diffs = [d['diff_bps'] for d in dec]\n"
            "else:\n"
            "    labels = [d[0] for d in R['decades']]; diffs = [d[3] for d in R['decades']]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(labels, diffs, color=[AMBER, AMBER, RED], width=.55)\n"
            "for i, v in enumerate(diffs): ax.annotate(f'{v:+.1f}', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('A-day premium (bps/day)'); ax.set_title('Healthy for two decades - gone (negative) since 2017')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('A-day premium by decade:', dict(zip(labels, [f'{v:+.1f} bps/day' for v in diffs])))"
        ),
        md(
            f"The premium lived in 1997-2016 (**{R['decades'][0][3]:+.1f}** and "
            f"**{R['decades'][1][3]:+.1f} bps/day**) and has been **{R['decades'][2][3]:+.1f} "
            "bps/day since 2017** — announcement days have earned *less* than ordinary days for "
            "nearly a decade.\n\n"
            "**Finally — the money question.** Hold SPY only on announcement days (you know the "
            "schedule in advance: buy the close before, sell the close after), vs just staying "
            "invested."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [st.overlay_stats(RET, MASKS['ANY'], cost_bps=cb) for cb in (1.0, 2.0, 5.0)]\n"
            "    net_ann = [r['net_ann_pct'] for r in rows]; bh = rows[0]['bh_ann_pct']\n"
            "else:\n"
            "    net_ann = [o[4] for o in R['overlay']]; bh = R['overlay'][0][5]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.bar(['1 bp', '2 bps', '5 bps'], net_ann, color=RED, width=.55, label='A-days-only strategy (net)')\n"
            "ax.axhline(bh, ls='--', c=GREEN, label=f'just hold SPY ({bh:.1f}%/yr)')\n"
            "for i, v in enumerate(net_ann): ax.annotate(f'{v:+.1f}%', (i, v), ha='center', va='bottom')\n"
            "ax.set_xlabel('one-way trading cost'); ax.set_ylabel('annualised return (%)')\n"
            "ax.set_title('The timing strategy is a mirage: it never comes close to buy & hold')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('A-days-only net ann:', [f'{v:+.2f}%' for v in net_ann], ' vs buy & hold', f'{bh:.2f}%')"
        ),
        md(
            f"Even at an institutional **1 bp** per trade the strategy nets "
            f"**{R['overlay'][0][4]:+.1f}%/yr vs {R['overlay'][0][5]:.1f}%/yr** for doing nothing, "
            f"and at a realistic **5 bps** it makes **{R['overlay'][2][4]:+.1f}%/yr** — nothing. "
            "You'd be trading ~31 round trips a year to give up three-quarters of the market's "
            "return. *(We don't even credit T-bill interest on the cash days — adding it wouldn't "
            "change the picture at retail costs.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** Announcement days earn more on average "
            f"(**{R['diff_bps']:+.1f} bps/day**) and carry an outsized share of the return, but the "
            f"gap sits **below the significance bar** (*t* = {R['welch_t']:.2f}, *p* ≈ 0.06), is "
            "entirely a **Fed-day** effect, and has been absent since 2017.\n"
            "- **Tradability — Mirage.** The announcement-days-only strategy loses to buy & hold at "
            "**every** cost level — there is nothing to deploy.\n"
            "- **\"Is it all just FOMC?\" — Confirmed.** Strip the Fed days out and CPI/jobs days "
            f"earn **{R['exfomc']['diff']:+.1f} bps/day** over ordinary days (*t* = "
            f"{R['exfomc']['t']:.2f}) — statistically nothing."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Why did the textbook version look stronger?** Savor-Wilson's sample is 1958-2009 — "
            "the era before the modern always-on news cycle, and one that ends right at the "
            "2008-2009 Fed-day fireworks. Post-publication decay is the oldest story in the "
            "anomaly literature.\n"
            "- **The Fed-day corner is its own rabbit hole** — the desk has torn it down three ways: "
            "[pre-FOMC drift](../../517-pre-fomc-drift/README.md) (real, decayed post-2012), "
            "[Fed drift folklore](../../67-fed-drift/README.md), and the "
            "[FOMC cycle](../../135-fomc-cycle/README.md).\n"
            "- **Build your own.** Swap SPY for small caps or high-beta names (Savor-Wilson predict "
            "a steeper announcement-day risk-return line), or add PPI/GDP release days — the "
            "calendar machinery in `data.py` makes that a ten-line change.\n\n"
            "*Think the premium hides in higher-beta assets or intraday windows? The calendar is "
            "hardcoded and sourced — show us a leg that clears t = 2 out of sample, then we'll talk.*"
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
            "# The Macro-Announcement-Day Premium — a quantitative teardown 🔬\n"
            "### Pooled Welch *t* + a same-density random-calendar placebo · per-type and per-decade "
            "splits · the TLT check · costs × turnover on the A-day overlay · the ex-FOMC third axis "
            "· a 100-seed synthetic null/power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "Savor & Wilson (2013, JFQA) report ~11.4 bps on scheduled macro-announcement days vs "
            "~1.1 bps otherwise (1958-2009). We retry the **pooled** claim — CPI + NFP + FOMC — on "
            "the modern tradable tape with an actual, source-documented release calendar.\n\n"
            "> ⚠️ **Data note.** SPY + TLT total-return closes (yfinance), 1997-01 → 2026-06; "
            f"hardcoded calendar of {R['n_fomc']} FOMC + {R['n_cpi']} CPI + {R['n_nfp']} NFP actual "
            "release dates (BLS archive index cross-checked against the official "
            "`histreleasedates.pdf`; construction in "
            "[`data.py`](../macro_announcement_premium/data.py)). SPY/TLT are survivorship-clean "
            "index vehicles. Numbers in [`docs/results.md`](../docs/results.md) (fingerprint `"
            + R['fingerprint'] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition.\n"
            ">\n"
            "> **Dedup map:** this is the *pooled* macro-day claim. The FOMC-only corner is "
            "[517-pre-fomc-drift](../../517-pre-fomc-drift/README.md) / "
            "[67-fed-drift](../../67-fed-drift/README.md); the cycle-week pattern is "
            "[135-fomc-cycle](../../135-fomc-cycle/README.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Pooled A-day premium **{R['diff_bps']:+.2f} bps/day** at Welch "
            f"**t = {R['welch_t']:.2f}**, placebo **p = {R['p_placebo']:.4f}** — sub-bar. Only the "
            f"FOMC leg clears (t = {R['types'][0][4]:.2f}); CPI t = {R['types'][1][4]:.2f}, NFP "
            f"t = {R['types'][2][4]:.2f}; 2017-2026 diff **{R['decades'][2][3]:+.2f}** bps/day. |\n"
            f"| **Tradability** | `MIRAGE` | A-days-only overlay nets **{R['overlay'][0][4]:+.2f}%/yr "
            f"at 1 bp** vs **{R['overlay'][0][5]:.2f}%** buy & hold; **{R['overlay'][2][4]:+.2f}%/yr "
            f"at 5 bps**. {R['overlay'][0][0]:.0f}-bp net t₀ = {R['overlay'][0][3]:.2f} is the best "
            "case and still forfeits ~3/4 of the market return. |\n"
            f"| **All just FOMC?** | `CONFIRMED` | Ex-FOMC A-days: **{R['exfomc']['diff']:+.2f} "
            f"bps/day**, t = **{R['exfomc']['t']:.2f}**, placebo p = {R['exfomc']['p']:.4f} — a "
            f"statistical zero vs the FOMC leg's {R['fomc_leg']['diff']:+.2f} bps/day "
            f"(t = {R['fomc_leg']['t']:.2f}). |\n\n"
            "> 💡 In plain words: the announcement-day premium shows its silhouette, but on the "
            "modern tape it never clears the bar, it's all Fed days, it faded in 2017, and the "
            "trade version loses to doing nothing."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $a_t\\in\\{0,1\\}$ tag scheduled announcement sessions (CPI, NFP or FOMC — the "
            "union), all knowable ex-ante from published schedules. Let $r_t$ be SPY's total-return "
            "close-to-close daily return. Savor-Wilson:\n\n"
            "$$\\mathbb{E}[r_t\\,|\\,a_t=1]\\;\\gg\\;\\mathbb{E}[r_t\\,|\\,a_t=0],$$\n\n"
            "with the announcement-day mean carrying *most* of the unconditional equity premium — "
            "compensation for bearing concentrated macro-news risk (Ai-Bansal give the preference "
            "foundations).\n\n"
            "- **H₁ (pooled premium).** The A-minus-rest gap is positive and clears Welch t ≥ 2.\n"
            "- **H₂ (breadth).** The premium is spread across CPI/NFP/FOMC — not one event type.\n"
            "- **H₃ (deployability).** An A-days-only overlay survives costs and competes with "
            "buy & hold.\n\n"
            "We find **H₁ rejected at the bar** (t = 1.57, p = 0.056), **H₂ rejected** (only FOMC "
            "clears; ex-FOMC t = 0.58), **H₃ rejected** (loses to buy & hold at every cost level)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the honesty rails\n\n"
            "- **Welch t** for the group split: A-days are meaningfully more volatile, so the "
            "unequal-variance statistic is the right one; daily close-to-close returns carry "
            "negligible serial correlation at this horizon (the placebo is the sharper null "
            "anyway).\n"
            "- **Random-calendar placebo:** keep the *count* of A-days (923), relocate them "
            "uniformly, 20,000 draws — the exact answer to \"could any same-density calendar look "
            "this special?\".\n"
            "- **One execution lag, documented:** the schedule is public in advance → enter the "
            "close *before* the A-day, exit the A-day close. One round trip per A-day, one-way "
            "costs × NAV on both legs, long-only, no borrow. Raw total-return arithmetic (no T-bill "
            "credit on idle cash — an omission that penalises the overlay, never flatters it).\n"
            "- **Holiday releases** map to the next session (10 cases); 19 sessions carry two "
            "announcement types and count once in the union."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** SPY daily total-return closes, {R['start']} → {R['end']} "
            f"({R['n_days']:,} returns, {R['years']:.1f} yrs); TLT from 2002-08.\n"
            f"- **Calendar.** {R['n_fomc']} FOMC decision days (Fed historical calendars, scheduled "
            f"only) + {R['n_cpi']} CPI + {R['n_nfp']} NFP actual release dates (BLS archive index, "
            "cross-checked 19/19 against `histreleasedates.pdf`; shutdown gaps kept as-is). "
            f"**{R['n_adays']}** distinct A-sessions = {R['frac_days']:.1f}% of days.\n"
            "- **Primary test.** Pooled A-vs-rest Welch t + 20,000-draw placebo.\n"
            "- **Splits.** Per type (each vs pure non-A days) and per decade.\n"
            "- **Overlay.** Prior-close entry / A-day-close exit, 1 / 2 / 5 bps one-way.\n"
            "- **Third axis.** The pooled test with FOMC sessions removed.\n"
            "- **Control.** Synthetic worlds with a planted A-day edge, **100 seeds** (desk rule: "
            "≥ 20): the null must not light up, the planted edge must."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The pooled premium and its placebo\n\n"
            "The observed A-minus-rest gap against 20,000 same-density random calendars. (The "
            "notebook redraws 4,000 for the chart; the canonical p uses the full 20,000 — quoted "
            "from `docs/results.md`.)"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(RET.values, MASKS['ANY'], n_draws=4000)\n"
            "    draws = pl['draws']*1e4; obs = pl['obs']*1e4\n"
            "else:\n"
            "    rng = np.random.default_rng(602); draws = rng.normal(0.0, 4.4, 4000); obs = R['diff_bps']\n"
            "pv, wt = R['p_placebo'], R['welch_t']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.hist(draws, bins=60, color=GREY, alpha=.85, label='null: random same-density calendars')\n"
            "ax.axvline(obs, c=AMBER, lw=2.5, label=f'observed {obs:+.1f} bps/day')\n"
            "ax.axvline(np.quantile(draws, 0.95), c=RED, ls='--', lw=1.5, label='95th percentile of luck')\n"
            "ax.set_xlabel('A-minus-rest premium (bps/day)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Inside the luck cloud: Welch t = {wt:.2f}, placebo p = {pv:.4f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f} bps/day  Welch t={wt:.2f}  canonical placebo p={pv:.4f} (20,000 draws)')"
        ),
        md(
            f"> 💡 In plain words: the amber line sits at the **edge** of the luck cloud, not "
            f"outside it — {R['p_placebo']*100:.1f}% of random calendars beat the observed "
            f"**{R['diff_bps']:+.2f} bps/day**, and Welch *t* = {R['welch_t']:.2f} < 2. Under the "
            "house law (*REAL is earned by the tape*), a sub-2 *t* with three decades of published "
            "support reads **WEAK** — \"the literature says real; this tape alone can't certify "
            "it.\""
        ),
        md(
            "### 4b · Per-type split — where does the premium live?\n\n"
            "Each announcement type against the **pure non-announcement baseline** "
            "(6,496 days carrying no release of any type)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pure = ~MASKS['ANY']\n"
            "    rows = [(nm, st.event_vs_rest(RET.values, MASKS[nm], base=pure)) for nm in ('FOMC','CPI','NFP')]\n"
            "    tab = [(nm, s['n_ev'], s['ev_mean_bps'], s['diff_bps'], s['welch_t']) for nm, s in rows]\n"
            "else:\n"
            "    tab = R['types']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.4))\n"
            "labels = [t[0] for t in tab]\n"
            "a1.bar(labels, [t[3] for t in tab], color=[GREEN if t[4] >= 2 else GREY for t in tab], width=.55)\n"
            "for i, t in enumerate(tab): a1.annotate(f'{t[3]:+.1f}', (i, max(t[3], 0)), ha='center', va='bottom')\n"
            "a1.set_ylabel('diff vs pure non-A days (bps/day)'); a1.set_title('Premium by type')\n"
            "a2.bar(labels, [t[4] for t in tab], color=[GREEN if t[4] >= 2 else GREY for t in tab], width=.55)\n"
            "a2.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i, t in enumerate(tab): a2.annotate(f'{t[4]:.2f}', (i, max(t[4], 0)), ha='center', va='bottom')\n"
            "a2.set_ylabel('Welch t'); a2.set_title('Only FOMC clears the bar'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for t in tab: print(f'{t[0]:<5} n={t[1]:>3}  mean {t[2]:+7.2f}  diff {t[3]:+7.2f} bps/day  Welch t={t[4]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: **FOMC days earn {R['types'][0][3]:+.1f} bps over an ordinary day "
            f"(t = {R['types'][0][4]:.2f})** — that's the entire effect. CPI days are "
            f"indistinguishable from noise (t = {R['types'][1][4]:.2f}); NFP days lean positive "
            f"(t = {R['types'][2][4]:.2f}) but don't clear. Savor-Wilson's breadth claim — macro "
            "news generally, not just the Fed — fails on this tape."
        ),
        md(
            "### 4c · Decades + the TLT check\n\n"
            "Sub-period stability for the pooled premium, and the same calendar applied to long "
            "Treasuries (Savor-Wilson found announcement-day premia in bonds too)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    splits = [('1997-2006','1997-01-01','2006-12-31'), ('2007-2016','2007-01-01','2016-12-31'),\n"
            "              ('2017-2026','2017-01-01','2026-06-30')]\n"
            "    dec = [(d['label'], d['diff_bps'], d['welch_t']) for d in st.subperiod_stats(RET, MASKS['ANY'], splits)]\n"
            "    tlt_r = st.daily_returns(PX['TLT'].dropna()); tlt_r = tlt_r[tlt_r.index >= '2002-08-01']\n"
            "    tm = data.announcement_masks(tlt_r.index)\n"
            "    ts = st.event_vs_rest(tlt_r.values, tm['ANY'])\n"
            "    tlt_diff, tlt_t = ts['diff_bps'], ts['welch_t']\n"
            "else:\n"
            "    dec = [(d[0], d[3], d[4]) for d in R['decades']]\n"
            "    tlt_diff, tlt_t = R['tlt']['diff'], R['tlt']['t']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "labels = [d[0] for d in dec] + ['TLT\\n(2002-)']\n"
            "vals = [d[1] for d in dec] + [tlt_diff]\n"
            "tvals = [d[2] for d in dec] + [tlt_t]\n"
            "ax.bar(labels, vals, color=[AMBER, AMBER, RED, GREY], width=.55)\n"
            "for i, (v, t) in enumerate(zip(vals, tvals)):\n"
            "    ax.annotate(f'{v:+.1f}\\n(t={t:.2f})', (i, max(v, 0)), ha='center', va='bottom', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('A-day premium (bps/day)')\n"
            "ax.set_title('Alive 1997-2016, negative since 2017; nothing in Treasuries')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('decades:', [(d[0], f'{d[1]:+.2f} bps (t={d[2]:.2f})') for d in dec])\n"
            "print(f'TLT: diff {tlt_diff:+.2f} bps/day (Welch t={tlt_t:.2f})')"
        ),
        md(
            f"> 💡 In plain words: the pooled gap was ~+11-12 bps/day for two decades (each *t* ≈ "
            f"1.5, never individually decisive) and has been **{R['decades'][2][3]:+.1f} bps/day "
            f"since 2017**. And the Treasury side of Savor-Wilson does not replicate on TLT "
            f"(**{R['tlt']['diff']:+.2f} bps/day**, *t* = {R['tlt']['t']:.2f}). No sub-period rescue, "
            "no cross-asset rescue."
        ),
        md(
            "### 4d · Costs — the A-days-only overlay\n\n"
            "Enter the prior close, exit the A-day close, ~31 round trips/yr, 12.4% time in market. "
            "Gross/net labeled; raw total-return; no T-bill credit on cash (conservative)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [st.overlay_stats(RET, MASKS['ANY'], cost_bps=cb) for cb in (1.0, 2.0, 5.0)]\n"
            "    tab = [(r['cost_bps'], r['gross_bps_per_aday'], r['net_bps_per_aday'], r['net_t0'],\n"
            "            r['net_ann_pct'], r['bh_ann_pct']) for r in rows]\n"
            "else:\n"
            "    tab = R['overlay']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "x = np.arange(3)\n"
            "ax.bar(x - .18, [t[4] for t in tab], .36, color=RED, label='A-days-only overlay (net ann)')\n"
            "ax.bar(x + .18, [t[5] for t in tab], .36, color=GREY, label='buy & hold')\n"
            "for i, t in enumerate(tab):\n"
            "    ax.annotate(f'{t[4]:+.1f}%', (i - .18, max(t[4], 0)), ha='center', va='bottom', fontsize=9)\n"
            "    ax.annotate(f'{t[5]:.1f}%', (i + .18, t[5]), ha='center', va='bottom', fontsize=9)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['1 bp', '2 bps', '5 bps'])\n"
            "ax.set_xlabel('one-way cost'); ax.set_ylabel('annualised return (%)')\n"
            "ax.set_title('Never in the race: the overlay forfeits ~3/4 of the market return'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for t in tab: print(f'{t[0]:>4.1f} bps/leg: gross {t[1]:+.2f} -> net {t[2]:+.2f} bps/A-day '\n"
            "                    f'(net t0={t[3]:+.2f}) | net ann {t[4]:+.2f}% vs B&H {t[5]:.2f}%')"
        ),
        md(
            f"> 💡 In plain words: best case (1 bp/leg, institutional) the overlay nets "
            f"**{R['overlay'][0][4]:+.2f}%/yr** — against **{R['overlay'][0][5]:.2f}%/yr** for "
            f"holding SPY and doing nothing. At 5 bps the net A-day take is "
            f"**{R['overlay'][2][2]:+.2f} bps** (t₀ = {R['overlay'][2][3]:.2f}): dead. **MIRAGE** — "
            "there is no deployable version of this calendar."
        ),
        md(
            "### 4e · Third axis — remove the FOMC sessions\n\n"
            "The sharpest myth-check: is the *pooled* framing anything more than the FOMC-day "
            "premium the desk already tore down in [517](../../517-pre-fomc-drift/README.md) / "
            "[67](../../67-fed-drift/README.md)?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pure = ~MASKS['ANY']; exf = MASKS['ANY'] & ~MASKS['FOMC']\n"
            "    s_ex = st.event_vs_rest(RET.values, exf, base=pure)\n"
            "    s_fo = st.event_vs_rest(RET.values, MASKS['FOMC'], base=pure)\n"
            "    pair = [('CPI/NFP only\\n(no FOMC)', s_ex['diff_bps'], s_ex['welch_t']),\n"
            "            ('FOMC days', s_fo['diff_bps'], s_fo['welch_t'])]\n"
            "else:\n"
            "    pair = [('CPI/NFP only\\n(no FOMC)', R['exfomc']['diff'], R['exfomc']['t']),\n"
            "            ('FOMC days', R['fomc_leg']['diff'], R['fomc_leg']['t'])]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.4))\n"
            "a1.bar([p[0] for p in pair], [p[1] for p in pair], color=[GREY, GREEN], width=.5)\n"
            "for i, p in enumerate(pair): a1.annotate(f'{p[1]:+.1f}', (i, max(p[1], 0)), ha='center', va='bottom')\n"
            "a1.set_ylabel('diff vs pure non-A days (bps/day)'); a1.set_title('Premium: all of it is the Fed')\n"
            "a2.bar([p[0] for p in pair], [p[2] for p in pair], color=[GREY, GREEN], width=.5)\n"
            "a2.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i, p in enumerate(pair): a2.annotate(f't={p[2]:.2f}', (i, max(p[2], 0)), ha='center', va='bottom')\n"
            "a2.set_ylabel('Welch t'); a2.set_title('Strip FOMC out -> statistical zero'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for p in pair: print(f'{p[0].replace(chr(10), \" \"):<22} diff {p[1]:+7.2f} bps/day  Welch t={p[2]:+.2f}')\n"
            "print('FOMC engine by decade:', [(d[0], f'{d[1]:+.1f} bps (t={d[2]:.2f})') for d in R['fomc_decades']])"
        ),
        md(
            f"> 💡 In plain words: **CONFIRMED — it's all FOMC.** Ex-FOMC announcement days earn "
            f"**{R['exfomc']['diff']:+.2f} bps/day** over ordinary days (*t* = "
            f"{R['exfomc']['t']:.2f}, placebo *p* = {R['exfomc']['p']:.4f}); the FOMC leg carries "
            f"**{R['fomc_leg']['diff']:+.2f} bps/day** (*t* = {R['fomc_leg']['t']:.2f}). And even "
            f"the FOMC engine is a 2007-2016 story (*t* = {R['fomc_decades'][1][2]:.2f}) that reads "
            f"*t* = {R['fomc_decades'][2][2]:.2f} since 2017 — consistent with the decayed pre-FOMC "
            "drift in study 517."
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic daily worlds with a **planted** A-day edge, seed-averaged (the notebook runs "
            "20 seeds for speed; the canonical 100-seed numbers are quoted from `docs/results.md`). "
            "The null must not light up; the planted edge must."
        ),
        code(
            "rows = []\n"
            "for edge in (0.0, 0.0020):\n"
            "    sw = st.synthetic_sweep(edge=edge, n_seeds=20)\n"
            "    rows.append((sw['edge_bps'], sw['mean_diff_bps'], sw['mean_t']))\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "labels = [f'planted\\n{r[0]:+.0f} bps/day' for r in rows]\n"
            "ax.bar(labels, [r[2] for r in rows], color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i, r in enumerate(rows): ax.annotate(f'mean t={r[2]:.2f}', (i, max(r[2], 0)), ha='center', va='bottom')\n"
            "ax.set_ylabel('mean Welch t (20 seeds)')\n"
            "ax.set_title('Control: zero edge -> t~0; planted edge -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in rows: print(f'planted {r[0]:+6.1f} bps/day: mean diff {r[1]:+7.2f}  mean t {r[2]:+6.2f}')\n"
            "print('canonical (100 seeds):', [(s[0], f'mean t={s[2]:.2f}, share |t|>=2: {s[3]}%') for s in R['syn']])"
        ),
        md(
            f"> 💡 In plain words: with no planted premium the detector reads mean *t* = "
            f"{R['syn'][0][2]:.2f} and a {R['syn'][0][3]}% false-positive rate (nominal); a planted "
            f"+20 bps/day edge is recovered at mean *t* = {R['syn'][1][2]:.2f} in "
            f"{R['syn'][1][3]}/100 seeds. The machinery is unbiased and well-powered — the real-tape "
            f"*t* = {R['welch_t']:.2f} is a genuine sub-bar reading, not a power failure. *(A "
            "machinery proof only — never cited to support a stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — pooled A-day premium **{R['diff_bps']:+.2f} bps/day** at Welch "
            f"**t = {R['welch_t']:.2f}** (placebo p = {R['p_placebo']:.4f}) on {R['years']:.1f} "
            f"years of SPY: below the t ≥ 2 bar. Only FOMC clears (t = {R['types'][0][4]:.2f}); "
            f"CPI/NFP carry nothing; the pooled effect reads {R['decades'][2][3]:+.2f} bps/day "
            "since 2017; TLT shows nothing. The literature (1958-2009) says real; this tape can't "
            "certify the pooled claim.\n"
            f"- **Tradability `MIRAGE`** — the A-days-only overlay nets {R['overlay'][0][4]:+.2f}%/yr "
            f"at 1 bp vs {R['overlay'][0][5]:.2f}% buy & hold and {R['overlay'][2][4]:+.2f}%/yr at "
            "5 bps. Nothing deployable survives.\n"
            f"- **All just FOMC? `CONFIRMED`** — ex-FOMC t = {R['exfomc']['t']:.2f} (placebo "
            f"p = {R['exfomc']['p']:.2f}) vs FOMC t = {R['fomc_leg']['t']:.2f}. The pooled "
            "macro-announcement premium is the FOMC-day premium wearing a bigger calendar — and "
            "that engine faded after 2017."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Sample matters.** Savor-Wilson end in 2009; the 2007-2016 window is where our FOMC "
            "leg peaks (t = 2.90). A result born in one regime and sold as a law of markets is the "
            "anomaly literature's oldest failure mode (McLean-Pontiff).\n"
            "- **The beta channel is untested here.** Savor-Wilson's deeper claim — the *security "
            "market line steepens* on A-days — needs a cross-section, not one index; a beta-sorted "
            "panel on announcement days is the natural extension.\n"
            "- **Intraday would sharpen it.** The FOMC statement lands at 14:00 ET; close-to-close "
            "dilutes the event window (study 517 splits overnight/intraday around the FOMC and "
            "finds the drift decayed there too).\n\n"
            "*The reproducible core is offline and deterministic; the calendar is actual release "
            "dates, hardcoded and sourced in `data.py`. Methods and sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
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
