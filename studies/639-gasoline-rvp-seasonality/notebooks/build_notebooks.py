"""Generate the two narrative notebooks for Study 639 (Gasoline RVP Seasonality).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily closes
under ../_cache/ (sliced to the frozen as-of) and otherwise quote the frozen headline numbers
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance RB=F/CL=F/UGA/^IRX,
# 2005-01-03 -> 2026-06-30 as-of, 257 monthly excess obs, fingerprint ef905dc22f3c).
R = dict(
    start="2005-01-03", end="2026-06-30", asof="2026-06-30", n_daily=5412, n_excess=257,
    # signal (spliced RB - CL, spot proxy), Feb-Apr per-year panel, n=21 years
    win_mean=3.77, rest_mean=-1.22, gap=4.98, welch_years=6.20,
    win_sum=11.30, t_win=5.10, hit=90.5, n_years=21, welch_pooled=4.23,
    n_pooled_w=66, n_pooled_r=191,
    # per-calendar-month (mean %/mo, t, n)
    months=[("Jan", 0.57, 0.31, 21), ("Feb", 1.34, 0.88, 22), ("Mar", 8.62, 5.34, 22),
            ("Apr", 1.72, 0.95, 22), ("May", -0.39, -0.26, 22), ("Jun", -1.54, -1.21, 22),
            ("Jul", 0.71, 0.59, 21), ("Aug", -0.50, -0.19, 21), ("Sep", -6.88, -3.12, 21),
            ("Oct", -2.82, -1.67, 21), ("Nov", 0.10, 0.09, 21), ("Dec", 0.79, 0.73, 21)],
    sep_mean=-6.88, sep_t=-3.12, sep_hit=85.7,
    # robustness: (label, cum %, t, welch, hit%)
    robust=[("Mar only", 8.78, 5.22, 5.61, 90), ("Mar-Apr", 10.14, 6.00, 6.80, 90),
            ("Feb-Apr", 11.30, 5.10, 6.20, 90), ("Feb-May", 10.91, 5.22, 6.38, 90)],
    halves=[("2005-2015", 9.60, 2.77, 11), ("2016-2025", 13.18, 4.81, 10)],
    # third axis (holder UGA vs spliced RB, per-year paired, n=18)
    roll_chain=16.08, roll_holder=5.68, roll_gap=-10.40, roll_t=-8.06, roll_neg=100,
    roll_n=18, sep_chain=-7.96, sep_holder=-1.58, sep_gap=6.38, sep_gap_t=4.95,
    # tradability
    inv_win_sum=-0.18, inv_t=-0.07, inv_hit=47.1, inv_n=17, inv_welch=-0.74,
    inv_sep=-0.02, inv_sep_t=-0.01,
    overlay=[(5.0, 9.45, 9.35, 1.48), (10.0, 9.45, 9.25, 1.46), (20.0, 9.45, 9.05, 1.43)],
    overlay_xcash=8.91, overlay_xcash_t=1.41, overlay_hit=72.2, overlay_n=18,
    # synthetic control (mean t over 20 seeds)
    syn_null=(-0.22, -0.24, 0.16, 0.03, 0.16),      # welch, cum, sep_t, gap, gap_t
    syn_plant=(4.61, 8.76, -2.76, -5.97, -42.35),
    fingerprint="ef905dc22f3c",
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Curve_prices_it%3F: Confirmed](https://img.shields.io/badge/Curve_prices_it%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from gasoline_rvp_seasonality import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PX = data.load_real()
    MRET = data.monthly_returns(PX)
    EX = st.excess_series(MRET)          # log(1+RB) - log(1+CL), spliced chains = spot proxy
else:
    PX = MRET = EX = None
print("real cache present:", HAVE_REAL,
      "| monthly excess obs:", (0 if EX is None else len(EX)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The seasonal with a law behind it ⛽\n"
            "### Every spring, US gasoline gets pricier *by federal statute* — you can see it on the chart. So why can't you buy it?\n\n"
            + BADGES +
            "Most market seasonals are folklore — \"sell in May,\" Santa rallies, harvest lows. This one "
            "is different: it has a **date written in the Federal Register**. By **May 1** every year, US "
            "refineries and terminals must ship **summer-blend gasoline** — a low-evaporation recipe "
            "(low \"RVP\") that can't use cheap butane and costs genuinely more to make. After "
            "**September 15**, the cheap winter blend becomes legal again.\n\n"
            "So gasoline *should* get expensive relative to crude oil every spring and cheap again every "
            "fall — on schedule, by law. We check the tape. Spoiler: **the seasonal is enormous and lands "
            "exactly on the statute's dates**… and the second half of this story is the best free lesson "
            "in futures markets you'll get this year.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the paired roll-gap test and the cost "
            "math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **One honest caveat up front.** Our gasoline \"price\" series (`RB=F`) is a chain of "
            "futures contracts spliced together — it tracks the **pump-level price seasonal** perfectly, "
            "but nobody earns its roll-date jumps. That distinction turns out to be the entire story."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does gasoline really outrun crude every spring? | **Yes, massively.** Feb–Apr, gasoline "
            f"beats crude by about **+{R['win_sum']:.0f}% per window**, positive in **19 of 21 years**. "
            "The odds of that being luck are microscopic. |\n"
            "| Does it land on the legal dates? | **Exactly.** Of all 12 months, only **March** (run-up "
            "to the May-1 deadline) and **September** (switch-back) pass the strictest statistical bar — "
            "the two months the statute names. |\n"
            "| Can you buy it? | **No.** The futures curve has read the law too: summer contracts "
            "already cost more all winter. Anyone actually *holding* gasoline futures pays the seasonal "
            f"back at every roll — **{R['roll_gap']:.1f}% per spring**, in **18 out of 18 years**. |\n"
            "| What's left after the roll? | **A coin flip.** The investable version earned "
            f"**{R['inv_win_sum']:.2f}%** per window over 17 years — statistically indistinguishable "
            "from zero. |\n\n"
            "> A seasonal can be 100% real at the pump and 0% available in your brokerage account. "
            "This is the cleanest example we know."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Summer gasoline is a legally different, more expensive product, and the switch happens "
            "on fixed dates (summer blend by May 1, winter blend legal again after Sept 15). So the "
            "gasoline-minus-crude spread must rally into spring and dump in September — every year, "
            "by law.\"*\n\n"
            "The mechanism is real chemistry and real regulation: winter gasoline is padded with cheap "
            "butane, which evaporates too easily in summer heat (smog), so the EPA caps summer "
            "volatility (Reid Vapor Pressure). Refiners must drain the winter recipe, retool, and ship "
            "the pricier summer one — all against a **hard deadline**. Unlike \"sell in May,\" this "
            "calendar has a *cause you can cite*: 40 CFR § 1090.215."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a price move is (a) big, (b) annual, and (c) scheduled by statute, it looks like the "
            "easiest trade in the world — just be long gasoline (against crude, to cancel the oil price "
            "itself) from February to April. If that worked, it would be free money printed by the "
            "Federal Register.\n\n"
            "The catch every commodity trader learns once: you can't hold \"gasoline.\" You hold "
            "**futures contracts**, and the sellers of those contracts have also read the law. The "
            "question isn't whether the seasonal exists — it's **who pockets it**."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{R['n_years']} years** of daily prices (2005→2026): RBOB gasoline futures "
            "(`RB=F`), WTI crude (`CL=F`) as the control leg, and the gasoline ETF **UGA** as the "
            "\"real holder.\" Then:\n\n"
            "1. **Build the spread.** Each month, gasoline's return *minus* crude's return — so a "
            "general oil rally (or crash) cancels out and only gasoline-specific moves remain.\n"
            "2. **Check the calendar.** Average that spread month by month across ~21 years. If the law "
            "matters, Feb–Apr should be positive and September negative — and *only* those.\n"
            "3. **Follow the money.** Compare the spliced price chain (what the *pump* sees) with UGA "
            "(what a *holder* earns, paying every roll). The difference is who keeps the seasonal."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The calendar, month by month.** Average gasoline-minus-crude monthly excess across all "
            "years. The statute predicts: up into spring, down in September."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mt = st.month_table(EX)\n"
            "    vals = mt['mean_pct'].tolist()\n"
            "else:\n"
            "    vals = [m[1] for m in R['months']]\n"
            "names = [m[0] for m in R['months']]\n"
            "cols = [GREEN if n in ('Feb','Mar','Apr') else (RED if n=='Sep' else GREY) for n in names]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.6))\n"
            "ax.bar(names, vals, color=cols, width=.65)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i, v in enumerate(vals):\n"
            "    if abs(v) > 2.5: ax.annotate(f'{v:+.1f}%', (i, v), ha='center', va='bottom' if v>0 else 'top')\n"
            "ax.set_ylabel('gasoline minus crude (%/month, avg across years)')\n"
            "ax.set_title('The RVP statute, drawn by the market: up into May 1, down after Sept 15')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({n: round(v,2) for n, v in zip(names, vals)})"
        ),
        md(
            f"That is a law showing up in prices. **March: +{R['months'][2][1]:.1f}%/month** on average "
            f"(refiners scramble ahead of the May-1 deadline), **September: {R['sep_mean']:.1f}%/month** "
            "(cheap winter blend comes back). Every other month is noise around zero. The spring window "
            f"(Feb–Apr) cumulates to **+{R['win_sum']:.1f}%** and was positive in **19 of 21 years**."
        ),
        md(
            "**Now the trap.** The green line below is the spliced price chain — the seasonal the pump "
            "sees. The grey line is **UGA**, a real fund that holds front-month gasoline futures and "
            "rolls them every month. Same windows, same years. Watch the gap."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rg = st.roll_gap_stats(MRET, months=data.RUNUP_MONTHS)\n"
            "    chain, holder, gap = rg['chain_pct'], rg['holder_pct'], rg['gap_pct']\n"
            "else:\n"
            "    chain, holder, gap = R['roll_chain'], R['roll_holder'], R['roll_gap']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.6))\n"
            "bars = ax.bar(['price chain\\n(what the pump sees)', 'UGA holder\\n(what you can buy)',\n"
            "               'the roll gap\\n(what the curve eats)'], [chain, holder, gap],\n"
            "              color=[GREEN, GREY, RED], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for b, v in zip(bars, [chain, holder, gap]):\n"
            "    ax.annotate(f'{v:+.1f}%', (b.get_x()+b.get_width()/2, v), ha='center',\n"
            "                va='bottom' if v>0 else 'top')\n"
            "ax.set_ylabel('avg Feb-Apr window return (%, 2009-2026)')\n"
            "ax.set_title('The spring seasonal exists - and the futures roll takes it back')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'chain {chain:+.2f}%  holder {holder:+.2f}%  roll gap {gap:+.2f}% per window')"
        ),
        md(
            f"The price chain gains **+{R['roll_chain']:.1f}%** per spring window; the actual holder "
            f"keeps **+{R['roll_holder']:.1f}%** (which is mostly just oil going up in spring). The "
            f"missing **{R['roll_gap']:.1f} points** is the **roll**: each month the fund must sell its "
            "expiring contract and buy the next one — and all winter long, the next (more-summer-y) "
            "contract *already costs more*, because the sellers know about May 1 too. That happened in "
            f"**{R['roll_n']} out of {R['roll_n']} years**. No exceptions, ever, in our sample.\n\n"
            "And the mirror image: in September the holder *doesn't* lose the full crash — the curve "
            f"already discounted it (holder {R['sep_holder']:+.1f}% vs chain {R['sep_chain']:+.1f}%). "
            "The curve gives back on the way down what it took on the way up. It prices the **statute**, "
            "both directions."
        ),
        md(
            "**So what's actually tradable?** Long UGA / short crude, February through April — the "
            "honest, buyable version of \"own the spring gasoline seasonal.\""
        ),
        code(
            "if HAVE_REAL:\n"
            "    exi = st.excess_series(MRET, long='UGA', short='CL=F')\n"
            "    pan = st.per_year_panel(exi, data.RUNUP_MONTHS)['win_sum'] * 100\n"
            "    yrs, vals = pan.index.tolist(), pan.tolist()\n"
            "else:\n"
            "    yrs, vals = list(range(2010, 2027)), [R['inv_win_sum']] * 17\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "ax.bar([str(y) for y in yrs], vals, color=[GREEN if v > 0 else RED for v in vals], width=.65)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('investable spring crack, per year (%/window)')\n"
            "ax.set_title(f'The buyable version: avg {np.mean(vals):+.2f}% per window - a coin flip')\n"
            "plt.xticks(rotation=45); plt.tight_layout(); plt.show()\n"
            "print(f'mean {np.mean(vals):+.2f}%/window, positive {100*np.mean([v>0 for v in vals]):.0f}% of years')"
        ),
        md(
            f"Average: **{R['inv_win_sum']:+.2f}% per window**, positive **{R['inv_hit']:.0f}%** of "
            "years — heads or tails. Twenty-one years of a thundering, statute-dated, 90%-hit-rate "
            "seasonal at the price level… and the version you can actually put in an account earns "
            "**nothing**. The sellers of May futures charged you the summer premium in January."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** The RVP seasonal is one of the most statistically solid calendar "
            f"effects on the desk's whole bench: **+{R['win_sum']:.1f}%** per Feb–Apr window "
            f"(*t* ≈ {R['welch_years']:.1f}), and only the statute's own two months survive the "
            "strictest test.\n"
            f"- **Tradability — Mirage.** The futures curve pre-prices the law. Holders pay the "
            f"seasonal back at the rolls ({R['roll_gap']:.1f}%/window, {R['roll_n']}/{R['roll_n']} "
            f"years); the investable crack earns {R['inv_win_sum']:+.2f}%/window. There is nothing to "
            "buy.\n"
            "- **\"Does the curve already price it?\" — Confirmed.** Both directions: it charges you "
            "the summer premium in spring rolls and refunds the winter discount in fall."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **This is THE lesson about commodity seasonals.** Spot seasonality ≠ futures returns. "
            "Any seasonal that everyone can date (harvests, heating oil in winter, natural gas "
            "shoulder months) lives in the *forward curve*, not in the holder's P&L. Before trading "
            "any commodity seasonal, ask: *is the curve shape already doing this?*\n"
            "- **Contrast with a calendar that IS capturable:** "
            "[516-dividend-month-premium](../../516-dividend-month-premium/README.md) — also a "
            "schedule known in advance, but there no arbitrageur can fully pre-price it away.\n"
            "- **Named siblings:** [226-crude-seasonality](../../226-crude-seasonality/README.md) "
            "(crude *outright* by month) and [306-crack-spread](../../306-crack-spread/README.md) "
            "(the crack *level* as a stock-timing signal). This study is the spread's *calendar*.\n\n"
            "*Think you've found the commodity seasonal the curve missed? Show the holder-vs-chain "
            "gap first — then we'll talk.*"
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
            "# Gasoline RVP Seasonality — a quantitative teardown 🔬\n"
            "### Per-year Welch panel on the RB−CL excess · Bonferroni month table · window/halves "
            "robustness · the paired holder-vs-splice roll-gap test · costs on the investable legs · "
            "a 20-seed planted-seasonal control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The claim "
            "is a **statute** (40 CFR 1090.215: summer-RVP fuel at refineries/terminals by May 1, winter "
            "blend legal after Sept 15), so the test design is a *dated-calendar* test on the "
            "gasoline-minus-crude spread — and a second, sharper test of whether the futures curve "
            "pre-prices the date.\n\n"
            "> ⚠️ **Construction note (load-bearing).** Yahoo `RB=F`/`CL=F` are **spliced front-month "
            "chains** — roll jumps included, earned by nobody. The Signal axis reads the splice as a "
            "**spot-price proxy**, labelled as such everywhere; the Tradability and third axes read "
            "`UGA`, the real holder. No survivorship (two futures chains + one live ETF, not a survivor "
            "basket). Real data: yfinance daily closes 2005→2026, as-of **" + R["asof"] + "**, "
            "fingerprint `" + R["fingerprint"] + "` — numbers frozen in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` | Feb–Apr window **+{R['win_sum']:.2f}%** cumulative excess "
            f"(Welch **t = {R['welch_years']:.2f}** across {R['n_years']} yearly pairs, one-sample "
            f"t = {R['t_win']:.2f}, hit {R['hit']:.1f}%); only Mar (t = +{R['months'][2][2]:.2f}) and "
            f"Sep (t = {R['sep_t']:.2f}) survive Bonferroni ×12 — the statute's own months. |\n"
            f"| **Tradability** | `MIRAGE` | Investable crack (UGA−CL) Feb–Apr: "
            f"**{R['inv_win_sum']:+.2f}%/window, t = {R['inv_t']:+.2f}** (n = {R['inv_n']}); roll gap "
            f"**{R['roll_gap']:.2f}%/window** (t = {R['roll_t']:.2f}), negative {R['roll_n']}/"
            f"{R['roll_n']} years; long-only UGA overlay t = {R['overlay'][1][3]:.2f} net. |\n"
            f"| **Curve prices it?** | `CONFIRMED` | Paired holder-minus-splice gap "
            f"**{R['roll_gap']:.2f}%** in the run-up (t = **{R['roll_t']:.2f}**) and "
            f"**+{R['sep_gap']:.2f}%** giveback in Sep (t = +{R['sep_gap_t']:.2f}). |\n\n"
            "> 💡 In plain words: the law is on the tape, loudly and on the right dates — and the "
            "futures curve pays itself the whole thing before you arrive."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $x_m = \\log(1+r^{RB}_m) - \\log(1+r^{CL}_m)$ — the monthly log-excess of the RBOB "
            "front chain over the WTI front chain (dollar-neutral, monthly rebalanced; the oil level "
            "nets out). The statute predicts a **dated** pattern:\n\n"
            "- **H₁ (run-up).** $E[x_m] > 0$ for $m \\in$ {Feb, Mar, Apr} — the transition into the "
            "May-1 refinery/terminal deadline.\n"
            "- **H₂ (switch-back).** $E[x_m] < 0$ for Sep — winter blend legal after Sep 15.\n"
            "- **H₃ (curve pricing).** For a real front-month holder $H$ (UGA) vs the spliced chain "
            "$S$: $\\sum_{win}(\\log(1+r^H) - \\log(1+r^S)) < 0$ — the roll pays the seasonal away.\n\n"
            "We find **H₁ and H₂ decisively supported** (Welch t = 6.20 across years; Mar and Sep the "
            "only Bonferroni survivors) and **H₃ decisively supported too** (gap t = −8.06, negative "
            "18/18) — which is exactly why the *tradable* version of H₁ is dead."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The per-year panel is the honest unit: each year contributes **one** window mean and one "
            "rest mean, so 66 pooled window months can't masquerade as 66 independent observations "
            "(monthly serial correlation inside a window year is absorbed into the yearly unit; annual "
            "observations of a monthly-rebalanced spread are essentially serially uncorrelated, so the "
            "plain Welch/one-sample t across ~21 years is the right statistic — no HAC lag choice to "
            "snoop). The month table gets a **Bonferroni ×12** bar because testing 12 months is 12 "
            "tests. The third axis is **paired** (same year, same window, holder minus chain), which "
            "removes the oil-level variance entirely — hence its enormous t.\n\n"
            "Execution lag: the calendar is a **statute known years in advance**; entries use the prior "
            "month-end close. One lag, documented, trivially satisfied."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** yfinance daily closes: `RB=F`, `CL=F`, `UGA`, `^IRX`; {R['n_daily']:,} rows, "
            f"{R['start']} → {R['end']} (as-of; partial month dropped); {R['n_excess']} monthly excess "
            "obs; UGA from 2008-02.\n"
            "- **Signal.** $x_m$ on the spliced chains (spot proxy, labelled). Primary: **Welch t "
            "across per-year pairs** (window mean vs rest mean). Secondary: one-sample t of per-year "
            "window sums; pooled-months Welch; per-month table with Bonferroni ×12 (|t| ≥ ~3.0).\n"
            "- **Robustness.** Window variants {Mar}, {Mar,Apr}, {Feb–Apr}, {Feb–May}; halves split "
            "2005–2015 / 2016–2025.\n"
            "- **Third axis.** Paired per-year holder-minus-splice window gap (UGA vs RB=F), run-up and "
            "September.\n"
            "- **Tradability.** (a) investable crack = UGA−CL excess, same panel machinery; (b) "
            "long-only UGA Feb–Apr overlay, one round trip/yr, **5/10/20 bps one-way × NAV**, plus "
            "net-of-window-T-bill (excess-vs-cash; `^IRX`). Long-only ⇒ no borrow.\n"
            "- **Control.** Synthetic world with planted `amp` (window seasonal, mirrored −amp in Sep) "
            "and planted `roll_drag` on the holder; **mean t over 20 seeds** (no single-seed "
            "baselines); null must stay quiet."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The month table and its Bonferroni bar\n\n"
            "Twelve one-sample t's, so the honest bar is |t| ≥ ~3.0. If the statute is driving, the "
            "survivors should be March and September — and only those."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mt = st.month_table(EX)\n"
            "    tvals = mt['t'].tolist(); means = mt['mean_pct'].tolist()\n"
            "else:\n"
            "    tvals = [m[2] for m in R['months']]; means = [m[1] for m in R['months']]\n"
            "names = [m[0] for m in R['months']]\n"
            "cols = [GREEN if n in ('Feb','Mar','Apr') else (RED if n=='Sep' else GREY) for n in names]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.6))\n"
            "ax.bar(names, tvals, color=cols, width=.65)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for y in (3.0, -3.0): ax.axhline(y, ls='--', c=RED, lw=1)\n"
            "ax.annotate('Bonferroni x12 bar (|t|=3.0)', (0.2, 3.15), color=RED, fontsize=9)\n"
            "for i, (t, v) in enumerate(zip(tvals, means)):\n"
            "    if abs(t) >= 3.0: ax.annotate(f't={t:+.2f}\\n{v:+.1f}%/mo', (i, t), ha='center',\n"
            "                                  va='bottom' if t>0 else 'top', fontsize=9)\n"
            "ax.set_ylabel('one-sample t (per-year panel)')\n"
            "ax.set_title('Only the two statute months survive Bonferroni: March and September')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({n: round(t,2) for n, t in zip(names, tvals)})"
        ),
        md(
            f"> 💡 In plain words: out of twelve months, exactly two clear the multiple-testing bar — "
            f"**March (t = +{R['months'][2][2]:.2f})**, the panic-run into the May-1 deadline, and "
            f"**September (t = {R['sep_t']:.2f})**, the switch-back. That's the statute's fingerprint, "
            "not a data-mined calendar."
        ),
        md(
            "### 4b · The per-year panel — the primary test\n\n"
            "Each dot is one year's cumulative Feb–Apr excess (spot proxy). The Welch t across the "
            "yearly window-vs-rest pairs is the headline statistic."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ws = st.window_stats(EX, months=data.RUNUP_MONTHS)\n"
            "    pan = ws['panel']['win_sum'] * 100\n"
            "    yrs, vals = pan.index.tolist(), pan.tolist()\n"
            "    wt, t1, hit = ws['welch_t_years'], ws['t_win_sum'], ws['hit_rate']*100\n"
            "else:\n"
            "    yrs = list(range(2006, 2027)); vals = [R['win_sum']]*21\n"
            "    wt, t1, hit = R['welch_years'], R['t_win'], R['hit']\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.4))\n"
            "ax.bar([str(y) for y in yrs], vals, color=[GREEN if v>0 else RED for v in vals], width=.65)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.axhline(np.mean(vals), ls='--', c=GREY, label=f'mean {np.mean(vals):+.1f}%/window')\n"
            "ax.set_ylabel('Feb-Apr cumulative excess (%/window)')\n"
            "ax.set_title(f'Per-year panel: Welch t (years) = {wt:+.2f}, one-sample t = {t1:+.2f}, hit {hit:.0f}%')\n"
            "ax.legend(); plt.xticks(rotation=45); plt.tight_layout(); plt.show()\n"
            "print(f'mean window {np.mean(vals):+.2f}%  Welch t {wt:+.2f}  t {t1:+.2f}  hit {hit:.1f}%')"
        ),
        md(
            f"> 💡 In plain words: **+{R['win_sum']:.1f}% per spring window**, positive in 19 of 21 "
            f"years, Welch **t = {R['welch_years']:.2f}** — far past the desk's t ≥ 2 bar. On the spot "
            "proxy, this is about as real as a calendar effect gets."
        ),
        md(
            "### 4c · Robustness — window variants + halves\n\n"
            "A statute-dated effect shouldn't care exactly where we draw the window, and shouldn't "
            "live in one decade."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob = []\n"
            "    for label, mos in (('Mar only', (3,)), ('Mar-Apr', (3,4)), ('Feb-Apr', (2,3,4)), ('Feb-May', (2,3,4,5))):\n"
            "        w2 = st.window_stats(EX, months=mos)\n"
            "        rob.append((label, w2['win_sum_pct'], w2['t_win_sum'], w2['welch_t_years']))\n"
            "    pan = st.window_stats(EX, months=data.RUNUP_MONTHS)['panel']\n"
            "    halves = [('2005-2015', pan.loc[:2015]['win_sum']*100), ('2016-2025', pan.loc[2016:]['win_sum']*100)]\n"
            "    hv = [(lab, s.mean(), st.ttest_vs_zero(s.values/100), len(s)) for lab, s in halves]\n"
            "else:\n"
            "    rob = [(r[0], r[1], r[2], r[3]) for r in R['robust']]\n"
            "    hv = [(h[0], h[1], h[2], h[3]) for h in R['halves']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.3))\n"
            "a1.bar([r[0] for r in rob], [r[3] for r in rob], color=AMBER, width=.55)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i, r in enumerate(rob): a1.annotate(f'{r[1]:+.1f}%\\nt={r[3]:.1f}', (i, r[3]), ha='center', va='bottom', fontsize=9)\n"
            "a1.set_ylabel('Welch t (years)'); a1.set_ylim(0, 8.2); a1.set_title('Window variants'); a1.legend()\n"
            "a2.bar([h[0] for h in hv], [h[2] for h in hv], color=AMBER, width=.45)\n"
            "a2.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i, h in enumerate(hv): a2.annotate(f'{h[1]:+.1f}%\\nt={h[2]:.2f} (n={h[3]})', (i, h[2]), ha='center', va='bottom', fontsize=9)\n"
            "a2.set_ylabel('one-sample t of window sums'); a2.set_ylim(0, 6.2); a2.set_title('Halves split'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('variants:', [(r[0], round(r[1],2), round(r[3],2)) for r in rob])\n"
            "print('halves  :', [(h[0], round(h[1],2), round(h[2],2), h[3]) for h in hv])"
        ),
        md(
            f"> 💡 In plain words: every window variant sits at Welch t ≥ {min(r[3] for r in R['robust']):.1f}, "
            f"and each half clears the bar alone (t = {R['halves'][0][2]:.2f} and "
            f"{R['halves'][1][2]:.2f}). No window-snooping, no single-regime artefact."
        ),
        md(
            "### 4d · The third axis — the paired roll-gap test\n\n"
            "Per year, same window: cumulative log return of the **holder** (UGA) minus the **spliced "
            "chain** (RB=F). If the curve pre-prices the statute, this gap is negative in the run-up "
            "and positive in September. Paired, so the oil level cancels exactly."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rg = st.roll_gap_stats(MRET, months=data.RUNUP_MONTHS)\n"
            "    pan = rg['panel']['gap_sum'] * 100\n"
            "    yrs, vals = pan.index.tolist(), pan.tolist()\n"
            "    gt = rg['t']\n"
            "else:\n"
            "    yrs = list(range(2009, 2027)); vals = [R['roll_gap']]*18; gt = R['roll_t']\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "ax.bar([str(y) for y in yrs], vals, color=RED, width=.65)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.axhline(np.mean(vals), ls='--', c=GREY, label=f'mean {np.mean(vals):+.1f}%/window')\n"
            "ax.set_ylabel('holder minus spliced chain, Feb-Apr (%/window)')\n"
            "ax.set_title(f'The roll gap: negative in {sum(v<0 for v in vals)}/{len(vals)} years, paired t = {gt:+.2f}')\n"
            "ax.legend(); plt.xticks(rotation=45); plt.tight_layout(); plt.show()\n"
            "print(f'mean gap {np.mean(vals):+.2f}%/window  paired t {gt:+.2f}  negative {100*np.mean([v<0 for v in vals]):.0f}% of years')"
        ),
        md(
            f"> 💡 In plain words: **every single spring since UGA exists** ({R['roll_n']}/"
            f"{R['roll_n']} years), holding the front month cost about **{-R['roll_gap']:.0f} points** "
            f"of the {R['roll_chain']:.0f}-point window gain — paired **t = {R['roll_t']:.2f}**. And in "
            f"September the sign flips (**+{R['sep_gap']:.2f}%**, t = +{R['sep_gap_t']:.2f}): the curve "
            "refunds the switch-back it already discounted. The seasonal is a property of the *curve "
            "shape*, not of holding gasoline. (The gap also carries UGA's ~0.75%/yr fee and T-bill "
            "collateral interest — second-order beside a 10-point window gap.)"
        ),
        md(
            "### 4e · Tradability — the investable legs, with costs\n\n"
            "Two honest vehicles: the dollar-neutral **investable crack** (long UGA / short CL, "
            "monthly rebalanced) and the **long-only UGA window overlay** (one round trip/yr, one-way "
            "bps × NAV, then minus the window T-bill for the excess-vs-cash race)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    exi = st.excess_series(MRET, long='UGA', short='CL=F')\n"
            "    wsi = st.window_stats(exi, months=data.RUNUP_MONTHS)\n"
            "    inv = (wsi['win_sum_pct'], wsi['t_win_sum'])\n"
            "    ov = [st.overlay_stats(MRET, cost_bps=cb) for cb in (5.0, 10.0, 20.0)]\n"
            "    oc = st.overlay_excess_cash(MRET, data.tbill_monthly(PX), cost_bps=10.0)\n"
            "    bars = [inv[1]] + [o['t_net'] for o in ov] + [oc['t']]\n"
            "    labs = ['investable crack\\n(UGA-CL)'] + [f'UGA overlay\\nnet {o[\"cost_bps\"]:.0f} bps' for o in ov] + ['UGA overlay\\n10bps + T-bill']\n"
            "else:\n"
            "    inv = (R['inv_win_sum'], R['inv_t'])\n"
            "    bars = [R['inv_t']] + [o[3] for o in R['overlay']] + [R['overlay_xcash_t']]\n"
            "    labs = ['investable crack\\n(UGA-CL)', 'UGA overlay\\nnet 5 bps', 'UGA overlay\\nnet 10 bps', 'UGA overlay\\nnet 20 bps', 'UGA overlay\\n10bps + T-bill']\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.4))\n"
            "ax.bar(labs, bars, color=[RED, AMBER, AMBER, AMBER, AMBER], width=.6)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i, t in enumerate(bars): ax.annotate(f't={t:+.2f}', (i, t), ha='center', va='bottom' if t>0 else 'top')\n"
            "ax.set_ylabel('one-sample t (per-year net panel)'); ax.set_ylim(-1.2, 3.0)\n"
            "ax.set_title('Nothing investable clears the bar')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'investable crack: {inv[0]:+.2f}%/window t={inv[1]:+.2f}; overlay ts:', [round(b,2) for b in bars[1:]])"
        ),
        md(
            f"> 💡 In plain words: the dollar-neutral version is a coin flip "
            f"(**{R['inv_win_sum']:+.2f}%/window, t = {R['inv_t']:+.2f}**, hit {R['inv_hit']:.0f}%). "
            f"The long-only window is +{R['overlay'][1][1]:.1f}% gross on average but t ≈ 1.4 — it's "
            "unhedged energy beta with 2008/2020-sized wrecks in the panel, and it fails the bar at "
            "every cost level (costs barely matter: one round trip a year). **MIRAGE** is the only "
            "honest stamp."
        ),
        md(
            "### 4f · Faithful-engine control — we know the truth here\n\n"
            "Synthetic world with a planted window seasonal (`amp`, mirrored −amp in Sep) and a planted "
            "holder drag (`roll_drag`). Mean t over 20 seeds; the null must stay quiet on every "
            "detector. *(Machinery proof only — never cited as market evidence.)*"
        ),
        code(
            "rows = []\n"
            "for amp, drag in ((0.0, 0.0), (0.03, 0.02)):\n"
            "    wt, st9, gt = [], [], []\n"
            "    for s in range(20):\n"
            "        m = data.synthetic_world(amp=amp, roll_drag=drag, seed=639 + s)\n"
            "        exs = st.excess_series(m)\n"
            "        wt.append(st.window_stats(exs, months=data.RUNUP_MONTHS)['welch_t_years'])\n"
            "        st9.append(st.single_month_stats(exs, month=data.SWITCHBACK_MONTH)['t'])\n"
            "        gt.append(st.roll_gap_stats(m, months=data.RUNUP_MONTHS)['t'])\n"
            "    rows.append((amp, drag, np.mean(wt), np.mean(st9), np.mean(gt)))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "x = np.arange(2); w = 0.25\n"
            "ax.bar(x - w, [r[2] for r in rows], w, color=GREEN, label='window Welch t')\n"
            "ax.bar(x, [r[3] for r in rows], w, color=GREY, label='Sep t')\n"
            "ax.bar(x + w, [np.clip(r[4], -12, 12) for r in rows], w, color=RED, label='roll-gap t (clipped)')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['null (amp=0, drag=0)', 'planted (amp=3%, drag=2%)'])\n"
            "ax.set_ylabel('mean t over 20 seeds')\n"
            "ax.set_title('Null stays quiet on every detector; planted world lights them all up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for r in rows: print(f'amp={r[0]*100:.0f}% drag={r[1]*100:.0f}%: window t={r[2]:+.2f}  Sep t={r[3]:+.2f}  roll-gap t={r[4]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with nothing planted, every detector reads ≈0 "
            f"(window {R['syn_null'][0]:+.2f}, Sep {R['syn_null'][2]:+.2f}, gap {R['syn_null'][4]:+.2f}); "
            f"plant a 3%/mo seasonal + 2%/mo holder drag and they light up "
            f"(window {R['syn_plant'][0]:+.2f}, Sep {R['syn_plant'][2]:+.2f}, gap {R['syn_plant'][4]:.1f}). "
            "The machinery measures what it claims to."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — Feb–Apr window **+{R['win_sum']:.2f}%** cumulative excess, Welch "
            f"**t = {R['welch_years']:.2f}** across {R['n_years']} yearly pairs (one-sample "
            f"t = {R['t_win']:.2f}, hit {R['hit']:.1f}%); Mar (+{R['months'][2][1]:.2f}%/mo, "
            f"t = +{R['months'][2][2]:.2f}) and Sep ({R['sep_mean']:.2f}%/mo, t = {R['sep_t']:.2f}) are "
            "the only Bonferroni-×12 survivors — the statute's own two months; both halves independently "
            "clear t ≥ 2. Spot-proxy label on the splice, said out loud.\n"
            f"- **Tradability `MIRAGE`** — investable crack {R['inv_win_sum']:+.2f}%/window at "
            f"t = {R['inv_t']:+.2f}; the roll gap hands the seasonal to the curve "
            f"({R['roll_gap']:.2f}%/window, t = {R['roll_t']:.2f}, negative {R['roll_n']}/{R['roll_n']} "
            f"years); long-only UGA overlay t ≈ {R['overlay'][1][3]:.2f} net of costs and cash. "
            "Nothing clears the bar.\n"
            f"- **Curve prices it? `CONFIRMED`** — the paired gap is the statute's mirror in the curve: "
            f"{R['roll_gap']:.2f}% charged in the spring rolls, +{R['sep_gap']:.2f}% refunded in "
            "September. A seasonal you can see — priced so you can't buy it."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general law:** for storable seasonal commodities, deterministic seasonals migrate "
            "into the forward curve (Borovkova & Geman 2006). The tradable residual is whatever the "
            "curve *mis*-prices — weather surprises, refinery outages — none of it calendar-dated.\n"
            "- **Fixed-maturity variants:** a fixed summer contract (e.g., buy the May RB crack in "
            "January and hold, no rolls) tests whether the *curve level* itself drifts seasonally — a "
            "different, subtler claim; yfinance's chain data can't cleanly separate it, a CME-data "
            "follow-up could.\n"
            "- **Named siblings:** [226-crude-seasonality](../../226-crude-seasonality/README.md) (WTI "
            "outright by month — weak, regime-dependent), [306-crack-spread](../../306-crack-spread/"
            "README.md) (the crack *level* as a refiner-stock timer — none). This study is the third "
            "axis of that triangle: the *dated calendar* on the *spread*, and who pockets it.\n\n"
            "*The reproducible core is offline and deterministic; the signal is the statute window on "
            "the RB−CL excess, the myth-check is the paired UGA-vs-splice roll gap. Methods and "
            "sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
