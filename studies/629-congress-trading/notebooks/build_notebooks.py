"""Generate the two narrative notebooks for Study 629 (Congress Trading).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached Senate Stock
Watcher events + price panel under ../_cache/ and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md). The synthetic control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (Senate Stock Watcher PTRs,
# 2,776 disclosed stock purchases 2015-01 -> 2021-03, 26 senators; yfinance total-return
# closes 2013-06 -> 2022-12 + SPY).
R = dict(
    n_events=2776, n_senators=26, disc_start="2015-01-05", disc_end="2021-03-09",
    n_R=1990, n_D=786, perdue_share=39.3, famous_n=1212, famous_share=43.7,
    lag_median=19, lag_mean=26.3,
    n_tickers=438, n_dropped=1, dropped="ITC", n_used=437,
    n_covered=2221, coverage=80.0,
    # event study (color): (horizon td, mean abn %, naive t, n)
    bhar=[(63, -0.14, -0.46, 2221), (126, 1.07, 2.33, 2221), (252, 4.87, 6.35, 2221)],
    # calendar-time (PRIMARY): (hold, alpha %/yr, NW t, days, avg names)
    caltime=[(63, -0.01, -0.00, 1617, 48), (126, 3.72, 1.11, 1680, 80),
             (252, 2.50, 0.89, 1806, 123)],
    # placebo (20 seeds, hold=126)
    plc_alpha=3.31, plc_alpha_sd=0.84, plc_t=1.79, plc_t_sd=0.43,
    real_minus_plc=0.41, plc_z=0.49,
    # costs at hold=126: (one-way bps, net alpha %/yr, NW t)
    costs=[(0.0, 3.72, 1.11), (5.0, 3.52, 1.05), (10.0, 3.32, 0.99)],
    # splits (h=126): (label, grp %, n_grp, rest %, n_rest, diff pp, welch t)
    splits126=[("R vs D", 1.12, 1553, 0.95, 668, 0.17, 0.18),
               ("big (>=$15k) vs small", 1.52, 585, 0.91, 1636, 0.60, 0.56),
               ("famous vs rest", 0.59, 840, 1.36, 1381, -0.77, -0.79)],
    famous252=(9.61, 1.99, 7.61, 4.57),      # grp %, rest %, diff pp, welch t at h=252
    # third axis, calendar-time hold=126: (label, alpha %/yr, NW t)
    famous_ct=(0.09, 0.02), boring_ct=(4.66, 1.85), exPerdue_ct=(4.75, 1.87),
    # synthetic: null over 20 seeds + planted
    syn_null_t=0.03, syn_null_sd=0.97, syn_null_max=2.10,
    syn_edge_alpha=27.37, syn_edge_t=12.25,
    fingerprint="884027dc84c9", asof="2026-07-03",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Famous_names_beat%3F: Busted](https://img.shields.io/badge/Famous_names_beat%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from congress_trading import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PX, SPY, EV = data.load_real()
    ABN = st.event_abnormal(PX, SPY, EV)
else:
    PX = SPY = EV = ABN = None
print("real cache present:", HAVE_REAL,
      "| events:", (0 if EV is None else len(EV)),
      "| tickers:", (0 if PX is None else PX.shape[1]))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do senators' stock trades beat the market? 🏛️\n"
            "### The congress-trading meme vs the disclosed tape — in plain English\n\n"
            + BADGES +
            "Everyone *knows* this one. A famous 2004 academic study (Ziobrowski et al.) found US "
            "senators' stock purchases beat the market by about **1% a month** in the 1990s — \"senators "
            "trade like corporate insiders.\" Twitter accounts track congressional trades to millions of "
            "followers; apps sell copy-trading of politicians' portfolios.\n\n"
            "There's just one catch nobody mentions: you can only copy a trade **once it's disclosed** — "
            "and by then it's about **three weeks old**. So we tested the only version of the claim that "
            "exists for you: buy what a senator bought, the day the disclosure becomes public, and see if "
            "it beats the S&P 500.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the placebo and the clustering "
            "autopsy? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Data notes up front.** The tape is the **Senate Stock Watcher** archive — 2,776 "
            "disclosed stock purchases by 26 senators, **2015–March 2021** (the scraper stopped there). "
            "One senator (David Perdue) is **39%** of it. About 20% of events drop because their ticker "
            "no longer exists on Yahoo (**survivorship** — a bias *in favour* of the claim). Every chart "
            "is drawn by the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do senators' disclosed buys beat the market? | **No.** Copy-trading every disclosed "
            "purchase earns roughly **0 to +3.7% a year** over the S&P — statistically "
            "indistinguishable from zero. |\n"
            "| But I've seen \"+5% in 12 months\" headlines? | That number is real — and it's an "
            "**illusion**: thousands of overlapping trades re-counting **one** market rally (2020–21). "
            "Handle the overlap honestly and it collapses. |\n"
            "| Is the little drift that remains *their* skill? | **No.** Buying the **same stocks on "
            "random dates** earns the same thing (+3.3%/yr). The drift is *which stocks they own*, not "
            "*when they trade*. |\n"
            "| Are the scandal names (Perdue, Loeffler, Inhofe) the alpha? | **No — busted.** Their "
            "copy-portfolio earned **+0.09%/yr** over SPY. The famous names are exactly where the "
            "nothing is. |\n\n"
            "> The 1990s paper may well have been right *for the 1990s, at the senators' own trade "
            "dates*. But the thing you can actually do — copy disclosures — has no edge on the modern "
            "tape. Modern academic work (Belmont et al. 2022) finds the same."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Senators know things. Committee hearings, briefings, lobbyists. Their personal stock "
            "trades beat the market — an academic study proved it. Follow the disclosures and ride "
            "along.\"*\n\n"
            "The study is real: **Ziobrowski et al. (2004)**, senators' purchases 1993–1998, ~**85 bps a "
            "month** of outperformance. Since the **STOCK Act (2012)** every trade must be disclosed "
            "within 30–45 days, which is what makes the copy-trading industry — and this test — "
            "possible.\n\n"
            f"On our tape the median gap between a senator's trade and its public disclosure is "
            f"**{R['lag_median']} days**. That's the product you're being sold: a three-week-old fax."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the meme is right, it's the cheapest edge in finance — a free, public, legally mandated "
            "feed of insider-grade trades. If it's wrong, a lot of people are paying for apps that sell "
            "them stale noise. And there's a civic-sized question underneath: the meme is *the* argument "
            "in the congressional-stock-ban debate. Whether the disclosed tape actually carries alpha is "
            "worth measuring, not assuming."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take every Senate **P**eriodic **T**ransaction **R**eport in the Senate Stock Watcher "
            f"archive — **{R['n_events']:,} stock purchases** by **{R['n_senators']} senators**, "
            f"disclosures {R['disc_start']} → {R['disc_end']} — and for each one we:\n\n"
            "1. **Buy at the first market close after the disclosure** (the only date a member of the "
            "public could act).\n"
            "2. **Measure the return vs the S&P 500** (SPY, dividends included on both sides) over the "
            "next 3, 6 and 12 months.\n"
            "3. **Test it two ways** — the naive way (average all trades: *looks* great) and the honest "
            "way (a day-by-day portfolio that can't double-count overlapping trades), plus a placebo: "
            "the **same stocks bought on random dates**. If the placebo earns the same, the disclosure "
            "date carries no information."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the chart the meme shows you.** Average abnormal return (vs SPY) after each "
            "disclosed purchase:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = st.pooled_table(ABN)\n"
            "    means = [r['mean_pct'] for r in rows]\n"
            "else:\n"
            "    means = [b[1] for b in R['bhar']]\n"
            "labels = ['3 months', '6 months', '12 months']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(labels, means, color=[GREY, AMBER, GREEN], width=.55)\n"
            "for i, v in enumerate(means): ax.annotate(f'{v:+.2f}%', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('avg return vs SPY after a disclosed purchase (%)')\n"
            "ax.set_title('The chart the meme shows you: +4.9% over the market in 12 months!')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('pooled mean abnormal (3/6/12 mo):', [f'{v:+.2f}%' for v in means])"
        ),
        md(
            f"**+{R['bhar'][2][1]:.2f}% over the market in a year**, across {R['n_covered']:,} trades — "
            "that *looks* like Ziobrowski lives. Here's the problem: those 12-month windows **overlap "
            "massively**. One senator filing 300 purchases in 2019–20 gets the same 2020–21 rebound "
            "counted 300 times. That's not 2,221 independent bets — it's a handful of market episodes "
            "wearing 2,221 costumes.\n\n"
            "**The honest version** builds one day-by-day portfolio: every day, hold whatever was "
            "disclosed-bought in the last N months, equal-weighted, and compare that single daily series "
            "to SPY. No double-counting possible."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ct = [st.calendar_time(PX, SPY, EV, hold=h) for h in (63, 126, 252)]\n"
            "    alphas = [r['alpha_ann_pct'] for r in ct]; tvals = [r['nw_t'] for r in ct]\n"
            "else:\n"
            "    alphas = [c[1] for c in R['caltime']]; tvals = [c[2] for c in R['caltime']]\n"
            "labels = ['hold 3 mo', 'hold 6 mo', 'hold 12 mo']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar(labels, alphas, color=AMBER, width=.55)\n"
            "for i, v in enumerate(alphas): a1.annotate(f'{v:+.1f}%/yr', (i, v), ha='center', va='bottom')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('alpha vs SPY (%/yr)')\n"
            "a1.set_title('Honest portfolio: the alpha...')\n"
            "a2.bar(labels, tvals, color=GREY, width=.55)\n"
            "a2.axhline(2, ls='--', c=RED, label='significance bar (t=2)')\n"
            "for i, v in enumerate(tvals): a2.annotate(f't={v:.2f}', (i, v), ha='center', va='bottom')\n"
            "a2.set_ylabel('Newey-West t'); a2.set_ylim(-0.4, 2.6); a2.legend()\n"
            "a2.set_title('...and how sure we are (not at all)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('calendar-time:', [f'{a:+.2f}%/yr (t={t:.2f})' for a, t in zip(alphas, tvals)])"
        ),
        md(
            f"The +4.9% headline collapses to **+{R['caltime'][2][1]:.1f}%/yr at t = "
            f"{R['caltime'][2][2]:.2f}** — statistical noise. The best horizon manages "
            f"**+{R['caltime'][1][1]:.1f}%/yr at t = {R['caltime'][1][2]:.2f}**, still far below the "
            "bar.\n\n"
            "**And even that drift isn't timing skill.** The killer test: buy the **same stocks** on "
            "**random dates** instead of the disclosure dates (20 different random draws):"
        ),
        code(
            "real_alpha = R['caltime'][1][1]\n"
            "if HAVE_REAL:\n"
            "    pl = st.placebo_random_dates(PX, SPY, EV, hold=126, n_seeds=20)\n"
            "    plc, plc_sd = pl['mean_alpha_ann_pct'], pl['sd_alpha_ann_pct']\n"
            "    real_alpha = st.calendar_time(PX, SPY, EV, hold=126)['alpha_ann_pct']\n"
            "else:\n"
            "    plc, plc_sd = R['plc_alpha'], R['plc_alpha_sd']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "bars = ax.bar(['senators\\' actual timing', 'SAME stocks,\\nRANDOM dates (20 draws)'],\n"
            "              [real_alpha, plc], color=[GREEN, GREY], width=.55,\n"
            "              yerr=[0, plc_sd], capsize=6)\n"
            "for i, v in enumerate([real_alpha, plc]): ax.annotate(f'{v:+.2f}%/yr', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('alpha vs SPY (%/yr)')\n"
            "ax.set_title('The placebo: random timing earns the same -> the dates carry no information')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real {real_alpha:+.2f}%/yr vs placebo {plc:+.2f}%/yr (sd {plc_sd:.2f})')"
        ),
        md(
            f"Random dates earn **+{R['plc_alpha']:.2f}%/yr**; the senators' actual dates earn "
            f"**+{R['caltime'][1][1]:.2f}%/yr**. Difference: **+{R['real_minus_plc']:.2f}%/yr** — "
            "well inside the noise. Whatever tiny drift exists comes from *which stocks senators happen "
            "to own* (an equal-weight basket of their names beat the cap-weighted S&P in 2015–22 "
            "regardless of timing), not from any information in the trades.\n\n"
            "**Last hope: the famous names.** Surely Perdue, Loeffler, Inhofe — the senators the DOJ "
            "actually investigated in 2020 — are where the edge hides?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    fam = st.calendar_time(PX, SPY, EV[EV['famous']], hold=126)\n"
            "    bor = st.calendar_time(PX, SPY, EV[~EV['famous']], hold=126)\n"
            "    fa, ba = fam['alpha_ann_pct'], bor['alpha_ann_pct']\n"
            "else:\n"
            "    fa, ba = R['famous_ct'][0], R['boring_ct'][0]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['FAMOUS names\\n(Perdue, Loeffler, Inhofe)', 'the BORING 23 others'],\n"
            "       [fa, ba], color=[RED, GREY], width=.55)\n"
            "for i, v in enumerate([fa, ba]): ax.annotate(f'{v:+.2f}%/yr', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('honest portfolio alpha vs SPY (%/yr)')\n"
            "ax.set_title('The meme names earned... nothing at all')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'famous {fa:+.2f}%/yr vs boring {ba:+.2f}%/yr (neither is significant)')"
        ),
        md(
            f"The famous names: **+{R['famous_ct'][0]:.2f}%/yr** — as close to exactly zero as markets "
            f"get. The boring rest: +{R['boring_ct'][0]:.2f}%/yr (still not significant, t = "
            f"{R['boring_ct'][1]:.2f}). The meme has it exactly backwards: what little drift exists "
            "lives with the senators nobody tweets about."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The copy-trade earns 0 to +3.7%/yr over SPY with *t* ≤ "
            f"{R['caltime'][1][2]:.2f} — noise — and the random-dates placebo shows the dates carry no "
            "information anyway. The famous +4.9%/12-mo pooled number is overlap double-counting.\n"
            "- **Tradability — Mirage.** You'd be trading 19-day-old news with zero measurable edge, on "
            "a tape that's *survivor-tilted in the claim's favour* and still says no.\n"
            "- **\"The famous names are the alpha\"? — Busted.** Perdue/Loeffler/Inhofe's copy-portfolio: "
            f"+{R['famous_ct'][0]:.2f}%/yr. The scandal is political, not statistical.\n\n"
            "> Ziobrowski measured senators at their **own trade dates** in the 1990s. We measured what "
            "**you** can do — and the modern disclosed tape agrees with the modern papers: nothing there."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **This is the Senate, 2015–2021.** The archive stops in March 2021 — the Tuberville era "
            "and the House (Pelosi, the biggest meme of all) are out of scope by data, not by choice. "
            "House PTRs exist (House Stock Watcher, Quiver) if you want to extend the test.\n"
            "- **Purchases only.** Sales are harder to interpret (liquidity, divorce, diversification). "
            "Richard Burr's infamous February 2020 dump is a *sale* story — a different study.\n"
            "- **Survivorship works for the meme here.** A fifth of the disclosed buys point at tickers "
            "that later died or were absorbed; they're missing from Yahoo and from our panel. Adding "
            "them back would likely make the copy-trade look *worse*.\n"
            "- **The sibling study.** [Corporate insiders (Form 4)](../../263-insider-buying/) — people "
            "trading their *own* company — is the version of this claim with actual legal teeth and a "
            "much longer literature.\n\n"
            "*Think the edge is in the House, in options, or in committee-matched trades? The engine is "
            "three files — swap the event table and show us a calendar-time t over 2.*"
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
            "# Congress Trading — a quantitative teardown 🔬\n"
            "### Disclosure-date event study vs calendar-time NW alpha · a 20-seed random-dates "
            "placebo (timing vs stock list) · party / size / famous Welch splits · the famous-names "
            "clustering autopsy · costs · a synthetic faithful-engine control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "Ziobrowski et al. (2004) found senators' purchases beat the market by ~85 bps/mo "
            "(1993–98, **transaction-date** entry). The only *replicable* version is the "
            "**disclosure-date** trade — and the modern tape (Belmont et al. 2022 agree) says it "
            "carries nothing.\n\n"
            "> ⚠️ **Data + survivorship note.** Senate Stock Watcher PTR archive: "
            f"**{R['n_events']:,}** stock purchases, disclosures {R['disc_start']} → {R['disc_end']} "
            f"(hard stop — the scraper died), {R['n_senators']} senators, Perdue = "
            f"{R['perdue_share']:.1f}% of events. {R['coverage']:.0f}% of events resolve on Yahoo with "
            "a full 12-mo window; the missing ~20% are delisted/renamed tickers — a **survivor tilt "
            "that favours the claim**. One recycled ticker (ITC) dropped by an integrity guard. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fingerprint"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Calendar-time alpha **{R['caltime'][0][1]:+.2f} / "
            f"{R['caltime'][1][1]:+.2f} / {R['caltime'][2][1]:+.2f} %/yr** at 3/6/12-mo holds, "
            f"**NW t = {R['caltime'][0][2]:+.2f} / {R['caltime'][1][2]:+.2f} / "
            f"{R['caltime'][2][2]:+.2f}** — never near 2; placebo shows the drift is the stock list "
            f"(real − placebo = **{R['real_minus_plc']:+.2f}%/yr**, {R['plc_z']:+.2f}σ). |\n"
            f"| **Tradability** | `MIRAGE` | Median **{R['lag_median']}-day** disclosure lag; net alpha "
            f"{R['costs'][2][1]:+.2f}%/yr at 10 bps (t = {R['costs'][2][2]:.2f}); survivor-tilted panel "
            "*helps* and it still fails. |\n"
            f"| **Famous names beat?** | `BUSTED` | Famous calendar-time alpha "
            f"**{R['famous_ct'][0]:+.2f}%/yr (t = {R['famous_ct'][1]:.2f})** vs boring "
            f"{R['boring_ct'][0]:+.2f}%/yr (t = {R['boring_ct'][1]:.2f}); the pooled famous premium "
            f"(+{R['famous252'][2]:.1f} pp at 12 mo, Welch t = {R['famous252'][3]:.1f}) is an overlap "
            "artifact. |\n\n"
            "> 💡 In plain words: measured at the only date you can trade, senators' disclosed buys are "
            "noise; the +4.9%/12-mo pooled headline is double-counting; and the meme names are where "
            "the zero is."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let event $e$ be a disclosed purchase (senator $s$, ticker $k$, disclosure date $d_e$). "
            "Entry is the first close $t_0(e) > d_e$ (exactly one execution lag). The BHAR at horizon "
            "$h$ is\n\n"
            "$$\\mathrm{abn}_h(e) = \\frac{P_{k,t_0+h}}{P_{k,t_0}} - \\frac{M_{t_0+h}}{M_{t_0}},$$\n\n"
            "with $M$ = SPY, both total-return.\n\n"
            "- **H₁ (the modern Ziobrowski).** Disclosed purchases carry positive abnormal return: a "
            "calendar-time portfolio of them has $\\alpha > 0$ with a robust $t \\ge 2$.\n"
            "- **H₂ (timing, not list).** The alpha comes from *when* they trade, not just *which* "
            "stocks they hold — i.e. it must beat a same-tickers random-dates placebo.\n"
            "- **H₃ (the meme).** The famous names (DOJ-probe subset) out-earn the boring aggregate.\n\n"
            "We find **H₁ rejected** (max NW t = 1.11), **H₂ rejected** (+0.41%/yr vs placebo, +0.49σ), "
            "**H₃ busted** (famous alpha ≈ 0; the pooled contrast is a clustering artifact)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two statistical traps this dataset sets\n\n"
            "**Trap 1: overlapping BHARs.** 2,221 12-month windows drawn from ~6 years of tape are not "
            "2,221 independent observations — especially when one senator (Perdue, 39% of events) files "
            "hundreds of buys within months of each other. The pooled naive t treats every window as "
            "fresh evidence; Fama (1998) is the classic warning, and the **calendar-time portfolio** "
            "(one daily excess-return series, HAC t) is the standard fix — the desk's verdict "
            "statistic.\n\n"
            "**Trap 2: list vs timing.** An equal-weight basket of senator-held names can beat "
            "cap-weighted SPY for reasons that have nothing to do with disclosure dates (size tilt, "
            "2020–21 breadth). The **random-dates placebo** (same tickers, random disclosure days, "
            "≥ 20 seeds per house rule) prices the list; only the *excess over placebo* is timing "
            "information."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Events.** {R['n_events']:,} PTR stock purchases (Senate Stock Watcher per-day "
            f"reports; the filename carries the disclosure date), {R['disc_start']} → {R['disc_end']}, "
            f"{R['n_senators']} senators, R {R['n_R']:,} / D {R['n_D']:,}.\n"
            f"- **Prices.** yfinance total-return closes, {R['n_tickers']} resolvable tickers + SPY; "
            f"{R['n_dropped']} recycled ticker dropped ({R['dropped']}); coverage "
            f"{R['n_covered']:,}/{R['n_events']:,} = {R['coverage']:.0f}% (survivorship named).\n"
            "- **Execution.** ONE lag: entry at the first close strictly after disclosure. Long-only; "
            "no borrow.\n"
            "- **Color.** Pooled BHAR means at 63/126/252 td with naive t (flagged as overstated).\n"
            "- **Primary.** Calendar-time EW portfolio (hold = 63/126/252 td), daily excess vs SPY, "
            "**Newey-West t (10 lags)**.\n"
            "- **Placebo.** Same tickers, uniform-random disclosure dates, 20 seeds; report mean ± sd.\n"
            "- **Splits.** Welch t: party, trade size (≥ $15k), famous (Perdue/Loeffler/Inhofe — the "
            "2020 DOJ-probe names with purchases on this tape).\n"
            "- **Costs.** 0/5/10 bps one-way × 2 legs amortised over the hold.\n"
            "- **Control.** Synthetic world with planted post-disclosure drift; null over 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The overlap illusion — pooled BHAR vs calendar-time\n\n"
            "The pooled per-event view says +4.87% at 12 months (naive t = 6.35). The calendar-time "
            "engine — which cannot double-count a market episode — says t = 0.89. Same tape, same "
            "trades: the entire 'significance' was pseudo-replication."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bh = [(r['h'], r['mean_pct'], r['naive_t']) for r in st.pooled_table(ABN)]\n"
            "    ct = [(h, *[st.calendar_time(PX, SPY, EV, hold=h)[k] for k in ('alpha_ann_pct', 'nw_t')])\n"
            "          for h in (63, 126, 252)]\n"
            "else:\n"
            "    bh = [(b[0], b[1], b[2]) for b in R['bhar']]\n"
            "    ct = [(c[0], c[1], c[2]) for c in R['caltime']]\n"
            "x = np.arange(3); w = .38\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.5))\n"
            "ax.bar(x - w/2, [b[2] for b in bh], w, color=RED, label='pooled BHAR naive t (overlap-blind)')\n"
            "ax.bar(x + w/2, [c[2] for c in ct], w, color=GREEN, label='calendar-time Newey-West t (honest)')\n"
            "ax.axhline(2, ls='--', c=GREY, label='t = 2 bar')\n"
            "for i, b in enumerate(bh): ax.annotate(f'{b[2]:.2f}', (i - w/2, b[2]), ha='center', va='bottom')\n"
            "for i, c in enumerate(ct): ax.annotate(f'{c[2]:.2f}', (i + w/2, max(c[2], 0)), ha='center', va='bottom')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['3 mo', '6 mo', '12 mo'])\n"
            "ax.set_ylabel('t-statistic'); ax.legend()\n"
            "ax.set_title('Same trades, same tape: the significance was the overlap, not the senators')\n"
            "plt.tight_layout(); plt.show()\n"
            "for (h, m, t), (_, a, nt) in zip(bh, ct):\n"
            "    print(f'h={h:>3}: pooled {m:+.2f}% (naive t={t:+.2f})  |  calendar-time {a:+.2f}%/yr (NW t={nt:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: the red bars are what you get if you pretend 2,221 overlapping "
            f"windows are independent — *t* = {R['bhar'][2][2]:.2f} at 12 months, a publishable "
            f"'anomaly'. The green bars are the same information without double-counting: "
            f"*t* = {R['caltime'][2][2]:.2f}. The gap between red and green **is** the meme."
        ),
        md(
            "### 4b · Timing vs stock list — the 20-seed random-dates placebo\n\n"
            "Same tickers, same event count, disclosure dates redrawn uniformly on the tape "
            "(20 seeds). If the senators' *dates* matter, the real alpha must sit outside the "
            "placebo cloud."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_random_dates(PX, SPY, EV, hold=126, n_seeds=20)\n"
            "    plc, plc_sd = pl['mean_alpha_ann_pct'], pl['sd_alpha_ann_pct']\n"
            "    real = st.calendar_time(PX, SPY, EV, hold=126)['alpha_ann_pct']\n"
            "else:\n"
            "    plc, plc_sd, real = R['plc_alpha'], R['plc_alpha_sd'], R['caltime'][1][1]\n"
            "z = (real - plc) / plc_sd\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "xs = np.linspace(plc - 4*plc_sd, plc + 4*plc_sd, 300)\n"
            "ax.plot(xs, np.exp(-((xs - plc)/plc_sd)**2 / 2), c=GREY, label='placebo alpha (20 random-date seeds)')\n"
            "ax.axvline(plc, c=GREY, ls=':')\n"
            "ax.axvline(real, c=GREEN, lw=2.5, label=f'real disclosure dates ({real:+.2f}%/yr)')\n"
            "ax.set_xlabel('calendar-time alpha vs SPY (%/yr), hold = 126 td'); ax.set_yticks([])\n"
            "ax.set_title(f'Real timing sits {z:+.2f} sd inside the random-timing cloud')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'real {real:+.2f}%/yr | placebo {plc:+.2f} +/- {plc_sd:.2f}%/yr | z = {z:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: buying the senators' stocks on **random days** earns "
            f"**+{R['plc_alpha']:.2f}%/yr** over SPY (their names, equal-weighted, simply beat the "
            f"cap-weighted index over 2015–22). Their **actual** dates earn "
            f"**+{R['caltime'][1][1]:.2f}%/yr** — {R['plc_z']:+.2f}σ from the placebo mean. The "
            "disclosure dates carry **zero** incremental information. H₂ dead."
        ),
        md(
            "### 4c · Splits — party, size (Welch t; color)\n\n"
            "The folklore sub-claims: one party trades better; bigger trades mean stronger conviction."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for label, mask in [('R vs D', ABN['party'] == 'R'),\n"
            "                        ('big (>=$15k) vs small', ~ABN['amount'].str.startswith('$1,001')),\n"
            "                        ('famous vs rest', ABN['famous'])]:\n"
            "        r = st.split_welch(ABN, mask, h=126)\n"
            "        rows.append((label, r['diff_pct'], r['welch_t']))\n"
            "else:\n"
            "    rows = [(s[0], s[5], s[6]) for s in R['splits126']]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.2))\n"
            "ax.barh([r[0] for r in rows], [r[2] for r in rows], color=GREY, height=.5)\n"
            "ax.axvline(2, ls='--', c=RED); ax.axvline(-2, ls='--', c=RED)\n"
            "for i, r in enumerate(rows): ax.annotate(f'  t={r[2]:+.2f} ({r[1]:+.2f} pp)', (r[2], i), va='center')\n"
            "ax.set_xlim(-3, 3); ax.set_xlabel('Welch t of the 6-mo abnormal-return difference')\n"
            "ax.set_title('No split rescues the claim at 6 months')\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in rows: print(f'{r[0]:24s} diff {r[1]:+.2f} pp  Welch t = {r[2]:+.2f}')"
        ),
        md(
            "> 💡 In plain words: Republicans vs Democrats: nothing (t = +0.18). Big trades vs small: "
            "nothing (t = +0.56). Famous vs rest at 6 months: nothing (t = −0.79 — the famous names "
            "are *behind*). Every folklore split is flat where the windows overlap least."
        ),
        md(
            "### 4d · The famous-names autopsy (the third axis)\n\n"
            f"At 12 months the pooled split *explodes*: famous **+{R['famous252'][0]:.2f}%** vs boring "
            f"+{R['famous252'][1]:.2f}%, Welch **t = +{R['famous252'][3]:.2f}**. A real-tape t ≥ 2! "
            "Except: Welch assumes independent observations, and the famous subset is 840 windows from "
            "three senators — mostly Perdue's clustered 2019–20 buys, each re-counting the same 2020–21 "
            "rebound (the same split is **negative** at 6 months). The calendar-time engine — which "
            "fixes exactly this — renders the verdict:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    fam = st.calendar_time(PX, SPY, EV[EV['famous']], hold=126)\n"
            "    bor = st.calendar_time(PX, SPY, EV[~EV['famous']], hold=126)\n"
            "    exP = st.calendar_time(PX, SPY, EV[EV['senator'] != 'David A Perdue , Jr'], hold=126)\n"
            "    trio = [('FAMOUS (Perdue/Loeffler/Inhofe)', fam['alpha_ann_pct'], fam['nw_t']),\n"
            "            ('BORING (23 others)', bor['alpha_ann_pct'], bor['nw_t']),\n"
            "            ('ex-Perdue aggregate', exP['alpha_ann_pct'], exP['nw_t'])]\n"
            "else:\n"
            "    trio = [('FAMOUS (Perdue/Loeffler/Inhofe)', *R['famous_ct']),\n"
            "            ('BORING (23 others)', *R['boring_ct']),\n"
            "            ('ex-Perdue aggregate', *R['exPerdue_ct'])]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "cols = [RED, GREY, GREY]\n"
            "ax.bar([t[0].replace(' (', '\\n(') for t in trio], [t[1] for t in trio], color=cols, width=.55)\n"
            "for i, t in enumerate(trio): ax.annotate(f'{t[1]:+.2f}%/yr\\n(t={t[2]:.2f})', (i, t[1]), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('calendar-time alpha vs SPY (%/yr), hold=126')\n"
            "ax.set_title('Overlap handled: the famous names are exactly zero')\n"
            "plt.tight_layout(); plt.show()\n"
            "for t in trio: print(f'{t[0]:34s} alpha {t[1]:+.2f}%/yr  NW t = {t[2]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the meme's star exhibit (famous pooled Welch t = "
            f"{R['famous252'][3]:.1f}) is the textbook failure mode this desk exists to catch — a "
            f"significant-looking t built on non-independent observations. Handled honestly, the famous "
            f"names earn **{R['famous_ct'][0]:+.2f}%/yr (t = {R['famous_ct'][1]:.2f})**. "
            "**BUSTED**, with the artifact dissected rather than just asserted."
        ),
        md(
            "### 4e · Costs (formality) and the faithful-engine control\n\n"
            "Costs barely move a 126-td-hold portfolio (2 legs amortised over six months) — the problem "
            "was never the friction. The synthetic control proves the machinery: a null world (no "
            "planted drift, 20 seeds) must stay flat; a planted +6 bps/day post-disclosure drift must "
            "light up."
        ),
        code(
            "print('costs at hold=126 (frozen from results.md):')\n"
            "for cb, na, t in R['costs']: print(f'  {cb:>4.1f} bps one-way: net alpha {na:+.2f}%/yr  NW t = {t:+.2f}')\n"
            "ts = []\n"
            "for s in range(20):\n"
            "    spx, sspy, sev = data.synthetic_world(edge_daily=0.0, seed=629 + s)\n"
            "    ts.append(st.calendar_time(spx, sspy, sev, hold=126)['nw_t'])\n"
            "spx, sspy, sev = data.synthetic_world(edge_daily=0.0006, seed=629)\n"
            "rp = st.calendar_time(spx, sspy, sev, hold=126)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(range(20), ts, c=GREY, label='null worlds (edge=0), 20 seeds')\n"
            "ax.axhline(rp['nw_t'], c=GREEN, lw=2, label=f'planted +6 bps/day: t = {rp[\"nw_t\"]:.1f}')\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 bar'); ax.axhline(-2, ls='--', c=RED)\n"
            "ax.set_xlabel('seed'); ax.set_ylabel('calendar-time NW t')\n"
            "ax.set_title(f'Null: mean t = {np.mean(ts):+.2f} (sd {np.std(ts):.2f}) -> the engine cannot fake it; planted edge -> lights up')\n"
            "ax.legend(loc='center right'); plt.tight_layout(); plt.show()\n"
            "print(f'null 20 seeds: mean t {np.mean(ts):+.2f}, sd {np.std(ts):.2f}, max|t| {np.max(np.abs(ts)):.2f}')\n"
            "print(f'planted edge: alpha {rp[\"alpha_ann_pct\"]:+.2f}%/yr, NW t {rp[\"nw_t\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: in 20 senator-free worlds the engine reads **t ≈ 0** (mean "
            f"{R['syn_null_t']:+.2f}, sd {R['syn_null_sd']:.2f}); plant a real post-disclosure drift and "
            f"it screams (**t = {R['syn_edge_t']:.1f}**). So when it reads t ≤ 1.11 on the real senators, "
            "that's the tape talking, not a blind instrument. *(Machinery proof only — never cited as "
            "market evidence.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — calendar-time alpha {R['caltime'][0][1]:+.2f} / "
            f"{R['caltime'][1][1]:+.2f} / {R['caltime'][2][1]:+.2f} %/yr (3/6/12-mo holds) with "
            f"**NW t = {R['caltime'][0][2]:+.2f} / {R['caltime'][1][2]:+.2f} / "
            f"{R['caltime'][2][2]:+.2f}**; the random-dates placebo absorbs the entire drift "
            f"({R['real_minus_plc']:+.2f}%/yr, {R['plc_z']:+.2f}σ). The pooled 12-mo +4.87% (naive "
            f"t = 6.35) is overlap pseudo-replication. Survivor-tilted panel (~20% of events lost with "
            "dead tickers) *favours* the claim and it still fails. Matches Eggers-Hainmueller (2013) "
            "and Belmont et al. (2022); Ziobrowski's 1990s transaction-date alpha does not survive to "
            "the disclosed era.\n"
            f"- **Tradability `MIRAGE`** — median {R['lag_median']}-day staleness, zero measurable "
            f"timing information, net {R['costs'][2][1]:+.2f}%/yr at 10 bps (t = "
            f"{R['costs'][2][2]:.2f}). Copy-trading apps sell the stock list, not an edge.\n"
            f"- **Famous names beat? `BUSTED`** — famous calendar-time alpha "
            f"**{R['famous_ct'][0]:+.2f}%/yr (t = {R['famous_ct'][1]:.2f})** vs boring "
            f"{R['boring_ct'][0]:+.2f}%/yr (t = {R['boring_ct'][1]:.2f}); the meme's Welch t = "
            f"{R['famous252'][3]:.1f} is a clustering artifact of one senator's overlapping buys."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Scope is data-bound.** Senate only, purchases only, 2015 → March 2021 (the scraper's "
            "death). The House tape (Pelosi-era memes), options PTRs, and 2021+ Senate filings are open "
            "extensions — House Stock Watcher / Quiver / Capitol Trades carry them.\n"
            "- **Committee-matched trades** (senator buys a stock overseen by their committee) is the "
            "sharpest remaining hypothesis — it needs a committee-assignment table joined to tickers' "
            "sectors; the event engine here takes any extra event column.\n"
            "- **Sales and the Burr trade.** February 2020's infamous dumps are *sales*; a symmetric "
            "study needs short-window CARs around the transaction (not disclosure) date and a very "
            "careful selection story.\n"
            "- **The structural sibling** is [263-insider-buying](../../263-insider-buying/) — Form 4 "
            "corporate insiders, where the law, the information set and the literature are all "
            "different (and stronger).\n\n"
            "*The reproducible core is offline and deterministic; the verdict statistic is the "
            "calendar-time Newey-West t at the disclosure-date entry, with a 20-seed random-dates "
            "placebo separating timing from stock list. Methods and sources: "
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
