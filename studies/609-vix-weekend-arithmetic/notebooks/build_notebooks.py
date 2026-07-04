"""Generate the two narrative notebooks for Study 609 (VIX Weekend Arithmetic).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached ^VIX/VIXY closes
under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network. Notebook cells
keep the placebo at 4,000 draws for speed; the canonical 20,000-draw p is quoted from ``R``.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance ^VIX 1990-01-02 ->
# 2026-06-30 [9,190 daily changes] + VIXY 2011-01-04 -> 2026-06-30, as-of 2026-06-30).
R = dict(
    start="1990-01-03", end="2026-06-30", n_days=9190,
    # weekday table: (weekday, mean %, t, n)
    table=[("Mon", 1.805, 10.25, 1733), ("Tue", -0.158, -1.05, 1886),
           ("Wed", -0.564, -3.82, 1883), ("Thu", 0.013, 0.09, 1849),
           ("Fri", -0.977, -6.02, 1839)],
    mon_mean=1.805, fri_mean=-0.977, spread=2.783,
    welch_t=11.61, hac_t=11.19, hac_mon_t=10.23, hac_fri_t=-3.96,
    p_placebo="< 1/20,000",
    # arithmetic race: (weekday, model f=0, observed, model at fitted f)
    race=[("Mon", 4.77, 1.81, 1.53), ("Tue", 0.00, -0.16, 0.00),
          ("Wed", 0.00, -0.56, 0.00), ("Thu", -2.33, 0.01, -0.76),
          ("Fri", -2.44, -0.98, -0.77)],
    f_implied=0.597, fit_rmse=0.46,
    # decades: (label, mon, fri, spread, welch t)
    decades=[("1990-1999", 2.345, -1.120, 3.464, 8.77),
             ("2000-2009", 1.856, -0.593, 2.449, 6.63),
             ("2010-2019", 1.565, -1.217, 2.781, 5.20),
             ("2020-2026", 1.247, -0.981, 2.229, 3.34)],
    # calendar gaps
    gap_post=2.067, gap_pre=-0.997, gap_mid=-0.379, n_post=1904, n_pre=1904,
    gap_t_post_pre=13.21, gap_t_post_mid=12.83,
    # third axis (VIXY 2011+)
    vixy_mon=-0.324, vixy_mon_t=-1.72, vixy_rest=-0.136, vixy_welch=-0.92,
    vix_mon_matched=1.626, n_weekends=728, vixy_years=15.5, trades_yr=47,
    # harvest: (one-way bps, gross %/yr, net %/yr)
    harvest=[(2.0, -15.2, -17.1), (5.0, -15.2, -19.9), (10.0, -15.2, -24.6)],
    # synthetic: (planted f, spread, welch t, nw t, recovered f)
    syn=[(1.0, -0.111, -0.63, -0.64, 1.000), (0.3, 4.342, 24.49, 25.09, 0.308)],
    fp_vix="63115cff08dc", fp_vixy="829eafeefcc4", asof="2026-06-30",
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![ETP_harvest%3F: Busted](https://img.shields.io/badge/ETP_harvest%3F-Busted-8b949e?style=flat-square)\n\n"
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
WD = ["Mon", "Tue", "Wed", "Thu", "Fri"]

from vix_weekend_arithmetic import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    VIX = data.load_vix()
    VIXY = data.load_vixy()
    D = st.dlog_pct(VIX)
else:
    VIX = VIXY = D = None
print("real ^VIX/VIXY cache present:", HAVE_REAL,
      "| daily changes:", (0 if D is None else len(D)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The fear gauge takes weekends off — by design 📅\n"
            "### The VIX weekend seesaw: a *guaranteed* calendar pattern hiding in the index's own formula\n\n"
            + BADGES +
            "Here's a pattern you can set your watch by. The VIX — Wall Street's \"fear gauge\" — "
            "tends to **fall on Fridays** and **jump on Mondays**. Not because the world gets safer "
            "before the weekend and scarier after it, but because of **arithmetic**: the VIX measures "
            "expected market wobble over the next **30 calendar days**, and a weekend is two calendar "
            "days when almost nothing can wobble (the market is closed). When the weekend slides *into* "
            "that 30-day window on Thursday/Friday, the number gets marked down; when it slides *out* "
            "over the weekend, Monday's print pops back up.\n\n"
            "One warning before we start: the folk version of this story often gets the direction "
            "**backwards** (\"the VIX drifts up into Friday and drops on Monday\"). The arithmetic — "
            "and 36 years of tape — say the exact opposite.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the day-count model and the cost math? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is there really a weekend seesaw in the VIX? | **Yes — a huge one.** Mondays average "
            "**+1.8%**, Fridays **−1.0%**, every decade since 1990. The odds of this being luck are "
            "less than 1 in 20,000. |\n"
            "| Which way does it go? | **Down into Friday, up on Monday** — exactly what the formula's "
            "calendar arithmetic predicts, and the *opposite* of the popular \"up into Friday\" "
            "retelling. |\n"
            "| Is it *pure* arithmetic? | **Mostly.** The tape shows about **40%** of the "
            "full-arithmetic size — option markets price a weekend day at roughly **60%** of a normal "
            "trading day's risk, not zero. |\n"
            "| Can you trade it? | **No.** Everything you can actually buy (VIX futures, ETPs like "
            "VXX/VIXY) already *knows* the calendar. The literal \"hold over the weekend\" trade "
            "**loses ~20%/yr** after costs. |\n\n"
            "> A guaranteed pattern that everyone can see, in an index nobody can buy — the market's "
            "politest way of saying *no free lunch*."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The VIX has a day-of-week drift baked into its own formula — pure calendar-variance "
            "arithmetic, a guaranteed pattern hiding in plain sight.\"*\n\n"
            "The mechanism is real math, not folklore. The VIX is (the square root of) the market's "
            "expected S&P-500 variance over the next **30 calendar days**, annualized on a calendar "
            "clock. But variance mostly happens while the market **trades**. Count the days in the "
            "window:\n\n"
            "- quoted on **Monday**: the next 30 days hold **22 trading + 8 weekend** days,\n"
            "- quoted on **Friday**: they hold **20 trading + 10 weekend** days.\n\n"
            "Same formula, different fuel. If weekends carried zero risk, Friday's VIX would print "
            "about **5% lower** than Monday's for the *same* underlying fear — so the index would "
            "mechanically sag into every weekend and snap back after it."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Three things ride on this. **First**, anyone reading the VIX as a pure fear thermometer "
            "is reading a thermometer that dips every Friday for non-fear reasons — Monday's \"vol "
            "spike\" headlines are partly the calendar talking. **Second**, if the pattern is this "
            "guaranteed, the obvious question is whether you can *harvest* it — buy vol Friday, sell "
            "it Monday. **Third**, the size of the seesaw quietly measures something deep: how much "
            "the options market thinks a weekend is worth, in risk terms."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **every** ^VIX close since 1990 — **{R['n_days']:,}** daily changes through "
            f"{R['end']} — and simply sort them by the day of the week they land on. Then we:\n\n"
            "1. **Measure the seesaw.** Average change by weekday, with honest significance tests.\n"
            "2. **Race it against the arithmetic.** The day-count model has one free knob — how much "
            "of a trading day's risk a weekend day carries. We fit it and see how much of the pattern "
            "the formula explains.\n"
            "3. **Try to cash it.** Buy a real, tradable VIX product (VIXY) at Friday's close, sell at "
            "Monday's close, every weekend for 15 years, minus costs."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The day-of-week table.** Average daily change of the VIX by weekday, 1990–2026."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tbl = st.weekday_table(D)\n"
            "    means = [r['mean_pct'] for r in tbl]\n"
            "else:\n"
            "    means = [r[1] for r in R['table']]\n"
            "colors = [GREEN if m > 0 else RED for m in means]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.6))\n"
            "ax.bar(WD, means, color=colors, width=.6)\n"
            "for i, v in enumerate(means):\n"
            "    ax.annotate(f'{v:+.2f}%', (i, v), ha='center', va='bottom' if v > 0 else 'top')\n"
            "ax.axhline(0, c=GREY, lw=1)\n"
            "ax.set_ylabel('average daily change of the VIX (%)')\n"
            "ax.set_title('Down into the weekend, +1.8% pop on Monday — 36 years of ^VIX')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('weekday means (%):', [round(m, 3) for m in means])"
        ),
        md(
            f"There's the seesaw — and the **sign**. Mondays average **+{R['mon_mean']:.1f}%**, "
            f"Fridays **{R['fri_mean']:.1f}%**: the VIX falls *into* the weekend and pops *after* it. "
            f"The Monday-minus-Friday gap is **{R['spread']:+.1f}% per day** — for scale, that's a "
            "bigger day-of-week gap than almost any calendar effect on any tape this desk has run. "
            "The quants notebook shows it's about **11 standard errors** from zero, in **every** "
            "decade, and that it follows holiday weekends too (it's the *market closure*, not the "
            "word \"Monday\")."
        ),
        md(
            "**Is it the arithmetic?** The day-count model has one knob: how much of a trading day's "
            "variance a weekend day carries (call it *f*). If *f* = 0 (weekends are risk-free), the "
            "model predicts a **+4.8%** Monday pop. If *f* = 1 (weekends are full trading days), it "
            "predicts **nothing**. Fit *f* to the tape and compare shapes."
        ),
        code(
            "if HAVE_REAL:\n"
            "    imp = st.implied_weekend_fraction(D)\n"
            "    obs = [imp['obs'][k] for k in range(5)]\n"
            "    full = [imp['model_full_arithmetic'][k] for k in range(5)]\n"
            "    fit = [imp['model_at_fit'][k] for k in range(5)]\n"
            "    f_hat = imp['f']\n"
            "else:\n"
            "    obs = [r[2] for r in R['race']]; full = [r[1] for r in R['race']]\n"
            "    fit = [r[3] for r in R['race']]; f_hat = R['f_implied']\n"
            "x = np.arange(5); w = 0.27\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.8))\n"
            "ax.bar(x - w, full, w, color=GREY, alpha=.55, label='full arithmetic (weekends = zero risk)')\n"
            "ax.bar(x,     obs,  w, color=GREEN, label='observed 1990-2026')\n"
            "ax.bar(x + w, fit,  w, color=AMBER, label=f'model at fitted f = {f_hat:.2f}')\n"
            "ax.set_xticks(x); ax.set_xticklabels(WD); ax.axhline(0, c=GREY, lw=1)\n"
            "ax.set_ylabel('daily change of the VIX (%)')\n"
            "ax.set_title('The tape shows the arithmetic shape, at ~40% of the full size')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'implied weekend fraction f = {f_hat:.3f} '\n"
            "      f'(a weekend day is priced at ~{f_hat*100:.0f}% of a trading day)')"
        ),
        md(
            f"The tape (green) has exactly the model's shape — sag late in the week, pop on Monday — "
            f"but at about **40%** of the zero-risk-weekend size (grey). The best-fit knob says the "
            f"market prices a weekend day at **~{R['f_implied']*100:.0f}%** of a trading day's risk "
            "(*f* ≈ 0.6). Which makes sense: markets are closed on weekends, but wars, elections and "
            "banking crises are not. The seesaw is the arithmetic **times** the market's partial "
            "weekend discount."
        ),
        md(
            "**Now try to cash it.** The VIX itself is a formula — you can't buy it. The closest "
            "tradable thing is a VIX-futures ETP (we use VIXY, 2011+). If the Monday pop leaks into "
            "the product, buying Friday's close and selling Monday's close should print money."
        ),
        code(
            "if HAVE_REAL:\n"
            "    v = st.vixy_weekend(VIXY, VIX, cost_bps=5.0)\n"
            "    pair = [v['vix_mon_mean_pct'], v['mon_mean_pct']]\n"
            "else:\n"
            "    pair = [R['vix_mon_matched'], R['vixy_mon']]\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.5))\n"
            "ax.bar(['^VIX index\\n(the formula)', 'VIXY\\n(what you can buy)'], pair,\n"
            "       color=[GREY, RED], width=.55)\n"
            "for i, v_ in enumerate(pair):\n"
            "    ax.annotate(f'{v_:+.2f}%', (i, v_), ha='center', va='bottom' if v_ > 0 else 'top')\n"
            "ax.axhline(0, c=GREY, lw=1)\n"
            "ax.set_ylabel('average Monday (over-the-weekend) change, 2011-2026 (%)')\n"
            "ax.set_title('The index pops on Monday. The thing you can buy... loses money.')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'index Monday mean {pair[0]:+.2f}%  vs tradable VIXY Monday mean {pair[1]:+.2f}%')"
        ),
        md(
            f"And there the dream dies. Over the same 728 weekends the **index** jumped "
            f"**+{R['vix_mon_matched']:.2f}%** on the average Monday — but **VIXY**, the thing you can "
            f"actually buy, *lost* **{R['vixy_mon']:.2f}%**. Why? VIX futures are bets on where the "
            "index will be at expiry, priced by people who own calendars: Friday's futures price "
            "already includes Monday's mechanical pop. What's left is the product's ordinary running "
            f"cost. The literal weekend-hold strategy loses **~{abs(R['harvest'][1][2]):.0f}%/yr** "
            "after costs."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** The weekend seesaw is enormous ({R['spread']:+.1f}%/day "
            "Monday-vs-Friday, ~11 standard errors, every decade, holidays included) and its "
            "direction is exactly what the formula's calendar arithmetic predicts — **down into "
            "Friday, up on Monday**. (The popular \"up-into-Friday\" version is backwards.)\n"
            "- **Tradability — Mirage.** The index isn't tradable, and the tradable products are "
            "priced by people who can count weekends: the harvest trade loses ~20%/yr net.\n"
            "- **\"Can a VIX ETP harvest the Monday pop?\" — Busted.** 728 weekends, 15.5 years: the "
            "ETP's weekend return is *negative*. The pattern lives in the formula, not in any price "
            "you can touch."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Read the VIX with weekday glasses.** A Friday VIX of 18 and a Monday VIX of 18.5 can "
            "be the *same* fear level — the difference is the calendar. Monday \"vol spike\" stories "
            "deserve a 1.8% haircut before you believe them.\n"
            "- **The weekend is worth ~60% of a trading day.** That fitted *f* is a live estimate of "
            "how the options market prices non-trading time — academics found the same thing in "
            "option prices directly (Jones & Shemesh 2018).\n"
            "- **Siblings on the desk.** The *equity* weekend effect ([90-weekend](../../90-weekend/)) "
            "is a different (and mostly dead) claim; the VIX-ETP bleed the harvest trade drowned in is "
            "measured in [375-vxx-roll-decay](../../375-vxx-roll-decay/).\n\n"
            "*Think you've found a corner where the weekend arithmetic leaks into something tradable — "
            "weekly options, VIX futures near expiry? Show the net P&L after the roll and we'll talk.*"
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
            "# VIX Weekend Arithmetic — a quantitative teardown 🔬\n"
            "### Day-of-week table with HAC inference · label-shuffle placebo · the variance-day-count "
            "model race and the implied weekend fraction · decade + calendar-gap robustness · the VIXY "
            "leak test with costs · a planted-*f* synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The claim "
            "is *mechanical*: the VIX's 30-**calendar**-day window vs trading-day variance accrual "
            "forces a day-of-week drift into the index. Mechanical claims still owe the tape a number "
            "— so we measure the seesaw, race it against the model's predicted magnitude, and test the "
            "only version of it anyone could trade.\n\n"
            "> ⚠️ **Data note.** ^VIX daily closes 1990-01-02 → 2026-06-30 (9,190 changes; a single "
            "continuously published index — no survivorship on the Signal axis) + VIXY total-return "
            "closes 2011+ for the third axis (the surviving continuous-tape VXX-equivalent, named "
            "here). Offline core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R['fp_vix'] + "` / `"
            + R['fp_vixy'] + "`, as-of " + R['asof'] + ").\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` | Mon **+{R['mon_mean']:.3f}%/day** vs Fri "
            f"**{R['fri_mean']:.3f}%/day**; spread **+{R['spread']:.3f}%/day**, Welch "
            f"**t = {R['welch_t']:.2f}**, Newey-West(10) **t = {R['hac_t']:.2f}**, placebo "
            f"**p {R['p_placebo']}**; t ≥ 3.3 in every decade; calendar-gap cut t = "
            f"{R['gap_t_post_pre']:.2f}. Direction = the arithmetic's (folk version is "
            "sign-flipped). |\n"
            f"| **Tradability** | `MIRAGE` | Index untradable; VIXY weekend mean "
            f"**{R['vixy_mon']:.3f}%** (wrong direction) vs index **+{R['vix_mon_matched']:.3f}%**; "
            f"harvest = **{R['harvest'][1][1]:.1f}%/yr gross**, **{R['harvest'][1][2]:.1f}%/yr net** "
            "at 5 bps. |\n"
            f"| **ETP harvest?** | `BUSTED` | {R['n_weekends']} weekends / {R['vixy_years']:.1f} yrs: "
            f"Monday-vs-rest Welch **t = {R['vixy_welch']:.2f}**, negative mean. |\n\n"
            "> 💡 In plain words: the seesaw is as real and as mechanical as calendar effects get — "
            "and precisely because it is mechanical and public, no tradable instrument pays it."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "VIX² is the (strike-integrated) risk-neutral expected variance of the S&P 500 over the "
            "next 30 **calendar** days, annualized in calendar time. Suppose variance accrues at rate "
            "$\\sigma^2$ per **trading** day and $f\\sigma^2$ per weekend day. The window quoted on "
            "weekday $d$ holds $N_T(d)$ trading and $N_W(d)$ weekend days — (22, 8) Mon–Wed, (21, 9) "
            "Thu, **(20, 10) Fri** — so\n\n"
            "$$\\mathrm{VIX}(d) \\;\\propto\\; \\sqrt{N_{\\mathrm{eff}}(d)/30},\\qquad "
            "N_{\\mathrm{eff}}(d) = N_T(d) + f\\,N_W(d),$$\n\n"
            "and the pure-arithmetic close-to-close change landing on weekday $d$ is "
            "$50\\,\\ln\\!\\big(N_{\\mathrm{eff}}(d)/N_{\\mathrm{eff}}(d\\!-\\!1)\\big)$ percent.\n\n"
            "- **H₁ (the seesaw exists).** Mean Δln VIX by weekday shows the model's pattern: down "
            "into Thu/Fri, up on Monday. *(Note the folk version quotes the sign backwards — the "
            "model itself is unambiguous.)*\n"
            "- **H₂ (it's the arithmetic).** The magnitude matches the model for some economically "
            "sensible $f\\in[0,1]$, stable across decades, and follows *calendar gaps* (holidays), "
            "not weekday labels.\n"
            "- **H₃ (it's harvestable).** A tradable VIX claim (VIXY) inherits the Monday pop net of "
            "carry and costs.\n\n"
            "We find **H₁ decisively supported** (HAC t ≈ 11), **H₂ supported at ~40% amplitude** "
            "(implied f ≈ 0.6 — a partial weekend discount, consistent with Jones-Shemesh option "
            "pricing), **H₃ rejected outright** (the ETP's weekend mean is *negative*)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "If H₁–H₂ hold, every day-of-week reading of the VIX (Monday \"spike\" headlines, "
            "Friday \"complacency\" pieces, weekday-conditioned vol signals) needs a mechanical "
            "correction of order **1–2%/day** — enormous by calendar-anomaly standards. If H₃ held, "
            "there'd be a ~47-trades/yr harvest; the desk's prior (from "
            "[375-vxx-roll-decay](../../375-vxx-roll-decay/)) is that futures anticipate the formula "
            "and the ETP's roll bleed (~−15%/yr) swamps whatever is left. The inference bar: "
            "autocorrelation-robust **t ≥ 2 on the real tape** for the Signal stamp; the synthetic "
            "control is machinery proof only, never market evidence."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** ^VIX closes {R['start']} → {R['end']} ({R['n_days']:,} daily changes); unit "
            "= Δln VIX in % (the level ranges 9→83, so log changes).\n"
            "- **Day-of-week table.** Mean by the weekday the close lands on; one-sample t per "
            "weekday (within-weekday observations are ~1 week apart).\n"
            "- **Headline contrast.** Monday vs Friday: Welch t on the groups **and** a "
            "Newey-West(10) t on the Mon−Fri contrast from the dummy regression "
            "$d_t = a + b\\,\\mathbf{1}_{Mon} + c\\,\\mathbf{1}_{Fri} + e_t$ (Δln VIX is serially "
            "correlated; HAC is the robust statistic).\n"
            "- **Placebo.** 20,000 seeded reshuffles of the Mon/Fri tags; p = P[shuffled spread ≥ "
            "observed].\n"
            "- **The race.** Fit the single parameter f by least squares over the 5 (demeaned) "
            "weekday means; report the implied weekend fraction and the fit residual.\n"
            "- **Robustness.** By decade; and by *calendar gap* (post-gap = previous close ≥ 3 "
            "calendar days back; pre-gap = next close ≥ 3 days ahead — catches holiday weekends; the "
            "exchange calendar is public and known in advance).\n"
            "- **Third axis.** VIXY total-return closes 2011+: Monday(-over-weekend) return vs other "
            "days (Welch t) + the literal Friday-close→Monday-close hold, one round trip per weekend "
            "at 2/5/10 bps one-way. Entry at Friday's close on a calendar known in advance = the one "
            "documented execution lag.\n"
            "- **Positive control.** Synthetic log-AR(1) vol quoted through the arithmetic with a "
            "planted f: the null (f = 1) must stay silent; a planted f = 0.3 must be recovered."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The day-of-week table and the HAC contrast\n\n"
            "Mean Δln VIX by weekday with one-sample t's, then the Monday-minus-Friday contrast with "
            "Welch and Newey-West(10) statistics, and the label-shuffle placebo (4,000 draws in this "
            "cell for speed; the canonical 20,000-draw run in `verify.py` gives the quoted p)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tbl = st.weekday_table(D)\n"
            "    for r in tbl:\n"
            "        print(f\"  {r['weekday']}: {r['mean_pct']:+.3f}%/day  t = {r['t']:+.2f}  (n={r['n']:,})\")\n"
            "    c = st.mon_fri_contrast(D, lags=10)\n"
            "    pl = st.placebo_spread(D, n_draws=4000, seed=609)\n"
            "    print(f\"\\n  Mon-Fri spread {c['spread']:+.3f}%/day  Welch t={c['welch_t']:+.2f}  \"\n"
            "          f\"NW(10) t={c['hac_t']:+.2f}  placebo p={pl['p_value']:.4f} (4k draws)\")\n"
            "    means = [r['mean_pct'] for r in tbl]; tvals = [r['t'] for r in tbl]\n"
            "else:\n"
            "    means = [r[1] for r in R['table']]; tvals = [r[2] for r in R['table']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.4))\n"
            "cols = [GREEN if m > 0 else RED for m in means]\n"
            "a1.bar(WD, means, color=cols, width=.6); a1.axhline(0, c=GREY, lw=1)\n"
            "a1.set_ylabel('mean dln VIX (%/day)'); a1.set_title('The seesaw')\n"
            "a2.bar(WD, tvals, color=cols, width=.6); a2.axhline(0, c=GREY, lw=1)\n"
            "a2.axhline(2, ls='--', c=GREY); a2.axhline(-2, ls='--', c=GREY, label='|t| = 2')\n"
            "a2.set_ylabel('one-sample t'); a2.set_title('...and its significance'); a2.legend()\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: Monday **+{R['mon_mean']:.2f}%** (t = +{R['table'][0][2]:.1f}) and "
            f"Friday **{R['fri_mean']:.2f}%** (t = {R['table'][4][2]:.1f}) are both individually "
            f"massive; the Mon−Fri contrast is **+{R['spread']:.2f}%/day** at Welch "
            f"**t = {R['welch_t']:.1f}** / NW(10) **t = {R['hac_t']:.1f}** with placebo "
            f"**p {R['p_placebo']}**. The desk bar is t ≥ 2; this clears it five times over. Note "
            "Wednesday also prints negative (−0.56%) — the tape spreads the mark-down across the "
            "back half of the week a bit differently than the stylized model (see 4b)."
        ),
        md(
            "### 4b · The arithmetic race — magnitude vs the day-count model\n\n"
            "The model's one free parameter is the weekend variance fraction f. Fit it by least "
            "squares over the five weekday means and compare the three profiles: full arithmetic "
            "(f = 0), the tape, and the model at the fitted f."
        ),
        code(
            "if HAVE_REAL:\n"
            "    imp = st.implied_weekend_fraction(D)\n"
            "    obs = [imp['obs'][k] for k in range(5)]\n"
            "    full = [imp['model_full_arithmetic'][k] for k in range(5)]\n"
            "    fit = [imp['model_at_fit'][k] for k in range(5)]\n"
            "    f_hat, rmse = imp['f'], imp['rmse_pct']\n"
            "else:\n"
            "    obs = [r[2] for r in R['race']]; full = [r[1] for r in R['race']]\n"
            "    fit = [r[3] for r in R['race']]; f_hat, rmse = R['f_implied'], R['fit_rmse']\n"
            "x = np.arange(5); w = 0.27\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.8))\n"
            "ax.bar(x - w, full, w, color=GREY, alpha=.55, label='model, f = 0 (full arithmetic)')\n"
            "ax.bar(x,     obs,  w, color=GREEN, label='observed')\n"
            "ax.bar(x + w, fit,  w, color=AMBER, label=f'model, fitted f = {f_hat:.3f}')\n"
            "ax.set_xticks(x); ax.set_xticklabels(WD); ax.axhline(0, c=GREY, lw=1)\n"
            "ax.set_ylabel('dln VIX (%/day)')\n"
            "ax.set_title(f'Day-count model vs tape: implied weekend fraction f = {f_hat:.3f} (RMSE {rmse:.2f}%)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'implied f = {f_hat:.3f}   fit RMSE = {rmse:.2f}%/day')"
        ),
        md(
            f"> 💡 In plain words: the tape runs the model's shape at **~40%** of the zero-weekend-"
            f"variance amplitude — the fitted **f = {R['f_implied']:.3f}** says option markets price "
            "a weekend day at **~60%** of a trading day's variance. Two honest caveats: (i) realized "
            "weekend variance is far *lower* than 60% of a trading day (French-Roll), so the market "
            "under-discounts weekends — exactly Jones-Shemesh's option-mispricing result; (ii) the "
            f"fit is imperfect (RMSE {R['fit_rmse']:.2f}%/day): the tape puts more mark-down on "
            "Wednesday and less on Thursday than the stylized week. The seesaw is arithmetic × a "
            "behavioral weekend discount, not a clean law."
        ),
        md(
            "### 4c · Robustness — decades and calendar gaps\n\n"
            "A mechanical effect must show up in every regime and must follow *market closures*, not "
            "the weekday label. Decade-by-decade Monday-vs-Friday, then the gap cut (post-gap = "
            "previous close ≥ 3 calendar days back; pre-gap = next close ≥ 3 ahead — holiday "
            "weekends included)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    dec = st.by_decade(D)\n"
            "    labs = [r['decade'] for r in dec]; sprd = [r['spread'] for r in dec]\n"
            "    wt = [r['welch_t'] for r in dec]\n"
            "    g = st.gap_table(D)\n"
            "    gaps = [g['post_mean'], g['mid_mean'], g['pre_mean']]\n"
            "    print(f\"gap cut: post {g['post_mean']:+.3f}% (n={g['n_post']:,})  mid \"\n"
            "          f\"{g['mid_mean']:+.3f}%  pre {g['pre_mean']:+.3f}% (n={g['n_pre']:,})  \"\n"
            "          f\"post-vs-pre Welch t = {g['welch_t_post_vs_pre']:+.2f}\")\n"
            "else:\n"
            "    labs = [r[0] for r in R['decades']]; sprd = [r[3] for r in R['decades']]\n"
            "    wt = [r[4] for r in R['decades']]\n"
            "    gaps = [R['gap_post'], R['gap_mid'], R['gap_pre']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.4))\n"
            "a1.bar(labs, sprd, color=GREEN, width=.6)\n"
            "for i, (s, t) in enumerate(zip(sprd, wt)):\n"
            "    a1.annotate(f'{s:+.2f}%\\nt={t:.1f}', (i, s), ha='center', va='bottom', fontsize=9)\n"
            "a1.set_ylabel('Mon - Fri spread (%/day)'); a1.set_ylim(0, 4.3)\n"
            "a1.set_title('Every decade clears t = 2')\n"
            "a2.bar(['post-gap\\n(after closure)', 'mid-week', 'pre-gap\\n(before closure)'], gaps,\n"
            "       color=[GREEN, GREY, RED], width=.6)\n"
            "for i, v in enumerate(gaps):\n"
            "    a2.annotate(f'{v:+.2f}%', (i, v), ha='center', va='bottom' if v > 0 else 'top')\n"
            "a2.axhline(0, c=GREY, lw=1)\n"
            "a2.set_ylabel('mean dln VIX (%/day)'); a2.set_title('It follows market closures, not the label')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the spread runs **+{R['decades'][0][3]:.1f}% → "
            f"+{R['decades'][3][3]:.1f}%** across the four decades (t = {R['decades'][0][4]:.1f} → "
            f"{R['decades'][3][4]:.1f}) — a mild narrowing, never a disappearance. And cut by "
            f"*calendar gap* instead of weekday, post-closure days print **+{R['gap_post']:.2f}%** vs "
            f"**{R['gap_pre']:.2f}%** pre-closure (Welch t = {R['gap_t_post_pre']:.1f}) — holiday "
            "weekends behave exactly like ordinary ones. That is the mechanism's signature: it's the "
            "closure entering and leaving the 30-day window."
        ),
        md(
            "### 4d · The leak test — can VIXY harvest the pop? (third axis)\n\n"
            "The index is a formula; the tradable claim is a short-term VIX-futures ETP. Buy Friday's "
            "close, sell Monday's close, ~47 weekends/yr, one round trip per weekend at 2/5/10 bps "
            "one-way. If the futures anticipate the arithmetic, the ETP's weekend return should show "
            "nothing but its usual roll bleed."
        ),
        code(
            "if HAVE_REAL:\n"
            "    v = st.vixy_weekend(VIXY, VIX, cost_bps=5.0)\n"
            "    idx_mon, etp_mon = v['vix_mon_mean_pct'], v['mon_mean_pct']\n"
            "    net = [st.vixy_weekend(VIXY, VIX, cost_bps=cb)['net_ann_pct'] for cb in (2.0, 5.0, 10.0)]\n"
            "    gross = v['gross_ann_pct']\n"
            "    print(f\"VIXY Monday mean {etp_mon:+.3f}% (t={v['mon_t']:+.2f})  vs other days \"\n"
            "          f\"{v['rest_mean_pct']:+.3f}%   Welch t = {v['welch_t']:+.2f}\")\n"
            "else:\n"
            "    idx_mon, etp_mon = R['vix_mon_matched'], R['vixy_mon']\n"
            "    net = [h[2] for h in R['harvest']]; gross = R['harvest'][0][1]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.4))\n"
            "a1.bar(['^VIX index', 'VIXY (tradable)'], [idx_mon, etp_mon], color=[GREY, RED], width=.5)\n"
            "for i, v_ in enumerate([idx_mon, etp_mon]):\n"
            "    a1.annotate(f'{v_:+.2f}%', (i, v_), ha='center', va='bottom' if v_ > 0 else 'top')\n"
            "a1.axhline(0, c=GREY, lw=1); a1.set_ylabel('mean Monday change (%)')\n"
            "a1.set_title('The pop does not leak into the vehicle')\n"
            "a2.bar(['2 bps', '5 bps', '10 bps'], net, color=RED, width=.5, label='weekend hold, net')\n"
            "a2.axhline(gross, ls='--', c=GREY, label=f'gross ({gross:.1f}%/yr)')\n"
            "for i, v_ in enumerate(net): a2.annotate(f'{v_:.1f}%', (i, v_), ha='center', va='top')\n"
            "a2.axhline(0, c=GREY, lw=1); a2.set_xlabel('one-way cost')\n"
            "a2.set_ylabel('annualised P&L (%/yr)'); a2.set_title('The harvest: deeply negative at any cost')\n"
            "a2.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: over the matched 2011–2026 window the **index** pops "
            f"**+{R['vix_mon_matched']:.2f}%** on Mondays while **VIXY loses {R['vixy_mon']:.2f}%** — "
            f"the *wrong direction*, and statistically just its ordinary decay (Monday-vs-rest Welch "
            f"t = {R['vixy_welch']:.2f}). The literal harvest bleeds **{R['harvest'][0][1]:.1f}%/yr "
            f"gross** and **{R['harvest'][1][2]:.1f}%/yr net** at 5 bps. Futures price the forward "
            "VIX; a deterministic dip in the spot formula is already in Friday's futures price. "
            "**BUSTED** — and the Tradability stamp is a clean **MIRAGE**."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic log-AR(1) \"true vol\" quoted through the day-count arithmetic with a planted "
            "weekend fraction. f = 1.0 is the null (weekends count fully — no weekday pattern): the "
            "machinery must stay silent. f = 0.3 plants a big seesaw: the machinery must light up "
            "*and* recover f."
        ),
        code(
            "res = []\n"
            "for f in (1.0, 0.3):\n"
            "    syn = data.synthetic_tape(f=f, seed=609)\n"
            "    ds = st.dlog_pct(syn)\n"
            "    cs = st.mon_fri_contrast(ds, lags=10)\n"
            "    im = st.implied_weekend_fraction(ds)\n"
            "    res.append((f, cs['spread'], cs['welch_t'], cs['hac_t'], im['f']))\n"
            "    print(f'planted f={f:.1f}: spread {cs[\"spread\"]:+.3f}%/day  Welch t={cs[\"welch_t\"]:+.2f}  '\n"
            "          f'NW t={cs[\"hac_t\"]:+.2f}  recovered f={im[\"f\"]:.3f}')\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "labels = [f'planted f = {r[0]:.1f}\\n(null)' if r[0] == 1.0 else f'planted f = {r[0]:.1f}' for r in res]\n"
            "ax.bar(labels, [r[3] for r in res], color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i, r in enumerate(res):\n"
            "    ax.annotate(f't={r[3]:.2f}\\nf_hat={r[4]:.3f}', (i, max(r[3], 0)), ha='center', va='bottom')\n"
            "ax.set_ylabel('NW t of the Mon-Fri spread')\n"
            "ax.set_title('Control: null stays silent, planted f recovered')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: with weekends counted fully (the null) the estimator reports "
            f"t = {R['syn'][0][3]:.2f} and f = {R['syn'][0][4]:.3f} — no manufactured significance; "
            f"with a planted f = 0.3 it reports t = {R['syn'][1][3]:.2f} and recovers "
            f"f = {R['syn'][1][4]:.3f}. The machinery is unbiased in both directions, so the real-tape "
            f"t ≈ {R['hac_t']:.0f} and f ≈ {R['f_implied']:.2f} are genuine measurements. *(A "
            "faithful-engine / power check only — never cited to support the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — Mon **+{R['mon_mean']:.3f}%/day**, Fri **{R['fri_mean']:.3f}%/day**, "
            f"spread **+{R['spread']:.3f}%/day** at Welch **t = {R['welch_t']:.2f}** / NW(10) "
            f"**t = {R['hac_t']:.2f}**, placebo **p {R['p_placebo']}**, t ≥ 3.3 in every decade, "
            f"calendar-gap signature t = {R['gap_t_post_pre']:.1f}. Direction exactly as the "
            "arithmetic predicts (the folk \"up-into-Friday\" version is sign-flipped). Magnitude = "
            f"the arithmetic × a partial weekend discount (implied f = {R['f_implied']:.3f}).\n"
            f"- **Tradability `MIRAGE`** — the index is a formula; the tradable claim (VIXY) shows "
            f"**{R['vixy_mon']:.3f}%** on the average weekend vs the index's "
            f"**+{R['vix_mon_matched']:.3f}%**, and the literal harvest loses "
            f"**{R['harvest'][1][2]:.1f}%/yr net** at 5 bps. Fully anticipated by construction.\n"
            f"- **ETP harvest? `BUSTED`** — {R['n_weekends']} weekends, {R['vixy_years']:.1f} years, "
            f"Welch t = {R['vixy_welch']:.2f}, wrong sign. The seesaw never leaves the formula."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The weekday correction matters for research.** Any signal conditioned on VIX changes "
            "(vol-spike triggers, risk-on/off filters) inherits a ±1–2%/day mechanical weekday tilt; "
            "de-seasonalize by weekday (or by calendar gap) before interpreting.\n"
            "- **f is a live behavioral quantity.** Realized weekend variance is ~10–20% of a trading "
            "day (French-Roll); the options market prices ~60%. That 40-point wedge is the "
            "Jones-Shemesh mispricing, visible in an index formula — a nice teaching bridge from "
            "microstructure to macro folklore.\n"
            "- **Where a real trade *might* hide** is not the spot formula but instruments with "
            "imperfect calendar handling — deep-weekly options into 3-day holiday weekends, or VIX "
            "futures in their final hours. Both are cost- and capacity-hostile; the desk's priors "
            "sit in [605-vix-settlement-day](../../605-vix-settlement-day/) and "
            "[111-vix-term-structure](../../111-vix-term-structure/).\n\n"
            "*The reproducible core is offline and deterministic; methods and sources in "
            "[`docs/references.md`](../docs/references.md); frozen numbers in "
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
