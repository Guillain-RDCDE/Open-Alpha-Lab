"""Generate the two narrative notebooks for Study 649 (Gold Seasonality).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached GLD/^IRX tapes
under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance GLD/^IRX
# 2004-11-18 -> 2026-06-30, 259 month-end log returns).
R = dict(
    start="2004-11-18", end="2026-06-30", n_months=259,
    bonferroni_crit=2.89,
    # month table: month -> (mean_pct, n, t_naive, t_hac)
    month={
        1: (+3.47, 22, +3.22, +3.77), 2: (+1.21, 22, +1.19, +1.52),
        3: (-0.08, 22, -0.08, -0.08), 4: (+1.57, 22, +1.66, +1.98),
        5: (-0.54, 22, -0.63, -0.55), 6: (-0.85, 22, -0.72, -0.99),
        7: (+1.15, 21, +1.23, +1.65), 8: (+1.47, 21, +1.49, +1.35),
        9: (+0.03, 21, +0.02, +0.02), 10: (+0.64, 21, +0.56, +0.74),
        11: (+0.94, 21, +0.75, +0.62), 12: (+0.71, 22, +0.72, +0.65),
    },
    n_bonferroni_survive=1, best_month=1, best_mean=+3.47, best_t=+3.77,
    # September vs rest
    sep_mean=+0.03, sep_rest_mean=+0.88, sep_n=21, sep_rest_n=238,
    sep_spread=-0.85, sep_t=-0.64, sep_ci_lo=-3.23, sep_ci_hi=+1.60,
    # Summer vs rest
    sum_mean=+0.29, sum_rest_mean=+1.07, sum_n=86, sum_rest_n=173,
    sum_spread=-0.79, sum_t=-1.24,
    # Era contrast (September, split 2013-04-01)
    era_split="2013-04-01",
    era_early=+2.49, era_early_n=8, era_early_t=+0.53,
    era_late=-1.48, era_late_n=13, era_late_t=-1.60, era_diff_t=-1.40,
    # Third axis — timer vs buy & hold (excess of cash)
    bh_sharpe=+0.47, bh_cagr=+8.59, bh_vol=17.1, bh_maxdd=-46.7, bh_n=259,
    timer_gross_sharpe=-0.02, timer_gross_cagr=+1.48, timer_vol=5.8, timer_maxdd=-28.6,
    timer_net5_sharpe=-0.04, timer_net5_cagr=+1.38,
    timer_net10_sharpe=-0.05, timer_net10_cagr=+1.28,
    sep_hit=9, sep_hit_n=21, sep_hit_pct=42.9, sep_hit_lo=24.5, sep_hit_hi=63.5,
    # synthetic control
    syn_null_mean=-0.31, syn_null_sd=1.08, syn_null_fire=2, syn_planted_t=+6.58,
    fp_gld="44f6ff1685e4", fp_irx="03b833c2e7e3",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Gold's_best_month%3F: Busted](https://img.shields.io/badge/Gold's_best_month%3F-Busted-8b949e?style=flat-square)\n\n"
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

from gold_seasonality import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    GLD, IRX = data.load_real()
    RET = st.monthly_log_returns(GLD)
    CASH = st.monthly_cash_return(IRX)
else:
    GLD = IRX = RET = CASH = None
print("real cache present:", HAVE_REAL, "| monthly observations:", (0 if RET is None else len(RET)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Is September really gold's best month? 🪙📅\n"
            "### The oldest calendar story on the gold desk — a real physical mechanism, "
            "and no price edge to show for it\n\n"
            + BADGES +
            "Every August you'll read some version of it: *\"gold loves September\"* — Indian "
            "wedding season is starting, Diwali shoppers are stocking up on jewellery months "
            "ahead, and jewellers everywhere restock before the year-end holidays. Meanwhile "
            "summer is supposedly dead — nobody's buying gold rings in July. It's a story with "
            "a real mechanism behind it: Indian households really do buy an enormous amount of "
            "physical gold around the autumn festivals.\n\n"
            "The question this study asks is narrower and more useful: does any of that physical "
            "demand actually show up as a **price edge** in the gold you can buy on a US "
            "exchange? We tested 21+ years of GLD (SPDR Gold Shares) to find out.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the Bonferroni correction and the "
            "bootstrap CI? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Is September really gold's best month? | **No.** Its average monthly return over "
            f"21+ years is **{R['sep_mean']:+.2f}%** — essentially zero, and slightly *below* the "
            f"other 11 months' **{R['sep_rest_mean']:+.2f}%** average. |\n"
            f"| Is summer really a quiet lull? | **Only a little, and not certified.** Summer "
            f"months average **{R['sum_mean']:+.2f}%** — still positive, just a bit softer than "
            "the rest of the year, and not statistically distinguishable from noise. |\n"
            f"| So is *any* month special? | **One is — January**, at "
            f"**{R['best_mean']:+.2f}%/month**, a genuinely large effect. But that's a *different* "
            "story than the one we set out to test, and this study doesn't chase it. |\n"
            f"| Could you trade the September idea anyway? | **You'd be much better off not "
            f"trying.** Owning gold *only* in September earns **{R['timer_net5_cagr']:+.1f}%/yr** "
            f"net of costs — buying and holding gold the whole time earns "
            f"**{R['bh_cagr']:+.1f}%/yr**. |\n\n"
            "> The mechanism is real. The price pattern isn't."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Gold has a calendar. Indian wedding season and pre-Diwali jewellery buying "
            "pull physical demand into September, and Northern-hemisphere jewellers restock "
            "ahead of the holidays — that's why September is gold's best month. Summer, by "
            "contrast, is the dead season: nobody's shopping for gold jewellery in July.\"*\n\n"
            "This isn't idle chart-reading — the World Gold Council tracks Indian gold demand "
            "quarterly, and it genuinely clusters around the spring Akshaya Tritiya festival and "
            "the autumn wedding-and-festival season. The mechanism is textbook supply and demand: "
            "real households, buying real jewellery, on a real calendar."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a September premium were real and reliable, it would be one of the easiest trades "
            "in finance: no options, no leverage, no forecasting skill required — just buy GLD "
            "before September and sell it after, every single year. Financial-media roundups and "
            "seasonality-chart vendors sell exactly this idea every autumn.\n\n"
            "So we ask two things: does September actually outperform, on the real tape, once we "
            "test it properly — and if you built the obvious trade around it, would you actually "
            "come out ahead of just buying gold and holding it?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The tape.** {R['n_months']} months of GLD, {R['start']} → {R['end']} — every "
            "calendar month gold has traded as a US ETF.\n"
            "- **The comparison.** September's average monthly return vs the other 11 months, "
            "and — because a chart-watcher could cherry-pick *any* of the 12 months and call it "
            "\"the\" seasonal — every month gets tested, with a **Bonferroni** correction for "
            "looking at 12 of them at once.\n"
            "- **The trade check.** Buy GLD only in September (the calendar is public years in "
            "advance — no crystal ball needed), hold cash otherwise, pay costs, compare to simply "
            "buying and holding."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the whole calendar.** Average GLD monthly return for every one of the 12 "
            "months."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ms = st.month_stats(RET)\n"
            "    means = [ms.loc[m, 'mean'] * 100 for m in range(1, 13)]\n"
            "else:\n"
            "    means = [R['month'][m][0] for m in range(1, 13)]\n"
            "names = " + repr(MONTH_NAMES) + "\n"
            "fig, ax = plt.subplots(figsize=(10.0, 4.6))\n"
            "cols = [AMBER if m == 9 else (GREY if m != 1 else GREEN) for m in range(1, 13)]\n"
            "ax.bar(names, means, color=cols, width=.62)\n"
            "for i, v in enumerate(means): ax.annotate(f'{v:+.2f}%', (i, v), ha='center',\n"
            "    va='top' if v < 0 else 'bottom', fontsize=8.5)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('average monthly GLD return')\n"
            "ax.set_title(\"September (amber) isn't even close to the real standout (January, green)\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print({n: round(v, 2) for n, v in zip(names, means)})"
        ),
        md(
            f"September (amber) sits at **{R['sep_mean']:+.2f}%** — dead in the middle of the "
            f"pack. The genuine outlier is **January** (green), at **{R['best_mean']:+.2f}%/month** "
            "— a real effect statistically, but a completely different claim (a gold \"January "
            "effect\"), and not one this study set out to test or chase.\n\n"
            "**Head-to-head:** September vs every other month, pooled."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sep = st.month_vs_rest(RET, data.STRONG_MONTHS)\n"
            "    sm, rm = sep['mean'], sep['rest_mean']\n"
            "else:\n"
            "    sm, rm = R['sep_mean']/100, R['sep_rest_mean']/100\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['September\\n(n=21)', 'other 11 months\\n(n=238)'], [sm*100, rm*100],\n"
            "       color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([sm*100, rm*100]): ax.annotate(f'{v:+.2f}%', (i, v),\n"
            "    ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('average monthly return')\n"
            "ax.set_title('September is not gold\\'s best month -- it barely moves the needle')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'September {sm*100:+.2f}%  vs  rest {rm*100:+.2f}%')"
        ),
        md(
            f"September (**{R['sep_mean']:+.2f}%**) is actually a touch *below* the other 11 "
            f"months' average (**{R['sep_rest_mean']:+.2f}%**) — the wrong direction from the "
            "claim. The quants notebook shows this gap is nowhere near statistically real (Welch "
            f"*t* = {R['sep_t']:+.2f}, need ±2 to even start taking it seriously).\n\n"
            "**What about the \"summer lull\"?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    summ = st.month_vs_rest(RET, data.SUMMER_MONTHS)\n"
            "    a, b = summ['mean'], summ['rest_mean']\n"
            "else:\n"
            "    a, b = R['sum_mean']/100, R['sum_rest_mean']/100\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['summer May-Aug\\n(n=86)', 'other 8 months\\n(n=173)'], [a*100, b*100],\n"
            "       color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([a*100, b*100]): ax.annotate(f'{v:+.2f}%', (i, v),\n"
            "    ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('average monthly return')\n"
            "ax.set_title('Summer is softer -- but still positive, and not a certified pattern')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'summer {a*100:+.2f}%  vs  rest {b*100:+.2f}%')"
        ),
        md(
            f"Summer is a little softer (**{R['sum_mean']:+.2f}%** vs **{R['sum_rest_mean']:+.2f}%** "
            "the rest of the year) — the right *direction* for the story, but gold still made "
            "money on average every summer, and the gap isn't statistically real either.\n\n"
            "**Finally, the trade.** If you actually built the \"own gold only in September\" "
            "strategy, how would it have done against simply buying gold once and holding it?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    bh = st.summary(RET, rf=CASH)\n"
            "    timer = st.strong_month_timer(RET, data.STRONG_MONTHS, cash=CASH)\n"
            "    net = st.apply_timer_costs(timer, data.STRONG_MONTHS, cost_bps_one_way=5.0)\n"
            "    tn = st.summary(net, rf=CASH)\n"
            "    bh_cagr, t_cagr = bh['cagr']*100, tn['cagr']*100\n"
            "else:\n"
            "    bh_cagr, t_cagr = R['bh_cagr'], R['timer_net5_cagr']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "ax.bar(['buy & hold\\ngold, always', 'own gold only\\nin September'], [bh_cagr, t_cagr],\n"
            "       color=[GREEN, RED], width=.55)\n"
            "for i, v in enumerate([bh_cagr, t_cagr]): ax.annotate(f'{v:+.1f}%/yr', (i, v),\n"
            "    ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('CAGR, net of costs')\n"
            "ax.set_title('Chasing the September story costs you almost the entire return')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'buy & hold {bh_cagr:+.1f}%/yr   September-only timer (net) {t_cagr:+.1f}%/yr')"
        ),
        md(
            f"Buying gold once and holding it earned **{R['bh_cagr']:+.1f}%/yr**. Trying to be "
            f"clever and own it *only* in September earned **{R['timer_net5_cagr']:+.1f}%/yr net** "
            "— you'd have given up nearly the entire two-decade gold bull market to sit in cash "
            "11 months a year, waiting for a month that, on this tape, wasn't even special."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** September's average monthly return is statistically "
            "indistinguishable from an average month — and points the wrong direction. No month "
            "clears our multiple-testing bar in September's favor; the one that does (January) is "
            "a different story we didn't set out to tell.\n"
            "- **Tradability — Mirage.** A strategy built around the September idea loses almost "
            "all of gold's return relative to simply buying and holding it.\n"
            "- **\"Gold's best month\"? — Busted.** The physical-demand mechanism is real. The "
            "price pattern, on the ETF you can actually buy, is not."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Physical demand ≠ price seasonality.** India's gold buying really does cluster "
            "seasonally — but gold's *price* is set by a much larger, dollar-denominated global "
            "futures/ETF-flow market, which can absorb a predictable regional demand pulse without "
            "leaving a systematic footprint in the return series.\n"
            "- **January was the actual surprise here.** A genuinely large, statistically robust "
            "monthly effect — just not the one we were asked to test. A dedicated gold "
            "\"January effect\" study, done with the same discipline (Bonferroni, bootstrap, "
            "out-of-sample era split), would be the natural next study.\n"
            "- **Sibling studies:** [289-diwali-muhurat](../../289-diwali-muhurat/) tests the same "
            "festival's *equity* omen in India; [69-safe-haven](../../69-safe-haven/) tests gold's "
            "inflation/crash-hedge behavior; neither touches gold's monthly calendar.\n\n"
            "*Think there's a cleaner version of the seasonal story — maybe centered on the actual "
            "Diwali date, or on ETF flows rather than price? Show a net, certifiable edge after "
            "costs, and we'll take a look.*"
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
            "# Gold Seasonality — a quantitative teardown 🔬\n"
            "### A 12-cell Bonferroni-corrected month table · a Welch/HAC split on September and "
            "on summer · a circular block-bootstrap CI · a 2013 era contrast · an excess-of-cash "
            "timer race · a 20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **September is gold's best month, summer is a lull** — has a stated "
            "mechanism (Indian physical demand, World Gold Council *Gold Demand Trends*) and a "
            "century of chart-vendor repetition, but no rigorous, multiple-testing-corrected test "
            "on the record. The job here is to measure it honestly.\n\n"
            "> ⚠️ **Data note.** GLD daily adjusted closes (2004→2026) resampled to month-end log "
            "returns, plus ^IRX (13-week T-bill) for the excess-of-cash timer race — both "
            "yfinance, cached. No survivorship on the Signal axis (a single, continuously-listed "
            "physically-backed ETF, not a panel). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_gld"] +
            "` / `" + R["fp_irx"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | September **{R['sep_mean']:+.2f}%/mo** (n={R['sep_n']}) vs "
            f"rest **{R['sep_rest_mean']:+.2f}%** (n={R['sep_rest_n']}): Welch "
            f"**t = {R['sep_t']:+.2f}**, bootstrap CI **[{R['sep_ci_lo']:+.2f}%, "
            f"{R['sep_ci_hi']:+.2f}%]**; {R['n_bonferroni_survive']}/12 months clear Bonferroni "
            f"(**|t| ≥ {R['bonferroni_crit']:.2f}**), and it isn't September |\n"
            f"| **Tradability** | `MIRAGE` | timer net CAGR **{R['timer_net5_cagr']:+.2f}%/yr** vs "
            f"buy-and-hold **{R['bh_cagr']:+.2f}%/yr**; excess-of-cash Sharpe "
            f"**{R['timer_net5_sharpe']:+.2f}** vs **{R['bh_sharpe']:+.2f}**; hit rate "
            f"{R['sep_hit_pct']:.1f}% |\n"
            f"| **Gold's \"best month\"?** | `BUSTED` | actual best month is January "
            f"(HAC t = {R['best_t']:+.2f}), not September |\n\n"
            "> 💡 In plain words: the physical-demand mechanism is real; the price seasonal is "
            "not — on this tape it isn't even the right sign."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_{m,y}$ be GLD's log return in calendar month $m$ of year $y$. The claim "
            "predicts two things:\n\n"
            "- **H₁ (September premium).** $E[r_{9,y}] \\gg E[r_{m,y} \\mid m \\ne 9]$ — a large, "
            "systematic September outperformance, driven by Indian wedding-season/pre-Diwali "
            "physical demand and year-end jeweller restocking.\n"
            "- **H₂ (summer lull).** $E[r_{m,y} \\mid m \\in \\{5,6,7,8\\}] \\ll E[r_{m,y} \\mid "
            "m \\notin \\{5,...,8\\}]$ — a systematic summer underperformance.\n"
            "- **H₃ (bankability).** A calendar-known "\
            "\"own gold only in September\" timer beats buy-and-hold net of costs.\n\n"
            "We find **H₁ rejected** (wrong sign, *t* = "
            f"{R['sep_t']:+.2f}), **H₂ directionally right but not certified** (*t* = "
            f"{R['sum_t']:+.2f}), **H₃ rejected decisively** (the timer forfeits most of gold's "
            "return)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Twelve calendar months means twelve simultaneous tests if you eyeball a "
            "month-of-year table and pick the best-looking cell — exactly the trap seasonality "
            "roundups fall into every autumn. The primary test is a **Welch t** on September "
            "(and, separately, summer) vs the rest, cross-checked with **Newey-West (HAC)** "
            "one-sample *t*'s per month and a **Bonferroni** bar (α = 0.05/12) for the full "
            "12-cell table. A **circular block-bootstrap** CI (5,000 draws, 12-month blocks, "
            "respecting the annual seasonal structure) gives an honest interval on the September "
            "spread rather than a single point estimate. The 2013 era split is justified "
            "*ex ante* by an external event (the 2013-04-12/15 gold crash), not snooped, and "
            "tested as a **difference**, never eyeballed."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** GLD daily adjusted closes {R['start']} → {R['end']}, resampled to "
            f"{R['n_months']} month-end log returns; ^IRX for the excess-of-cash race. As-of "
            "2026-06-30 (last complete month).\n"
            "- **Headline.** 12-cell month table (naive + HAC *t*), Bonferroni bar for 12 tests; "
            "Welch *t* + bootstrap CI on September vs rest and summer vs rest.\n"
            "- **Cross-check.** Pre/post-2013 era contrast on the September effect, within-era "
            "Welch *t*'s and a Welch *t* of the difference.\n"
            "- **Execution (third axis).** The timer's position is set from the calendar alone — "
            "September is the same slot every year — so no signal-to-trade lag applies; the "
            "monthly return already spans the August close → September close. 2 one-way legs/yr "
            "× cost × NAV, charged only on the active month. Both the timer and buy-and-hold are "
            "raced **excess of ^IRX cash**.\n"
            "- **Control.** Synthetic i.i.d. monthly-return world, planted September-premium / "
            "summer-discount knob; the null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The full 12-cell month table, Bonferroni-corrected\n\n"
            "One-sample naive and Newey-West (HAC) *t*-stats per calendar month. The Bonferroni "
            f"bar for 12 simultaneous tests is **|t| ≥ {R['bonferroni_crit']:.2f}**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ms = st.month_stats(RET)\n"
            "    crit = st.bonferroni_crit_t(12, df=len(RET) - 2)\n"
            "    means = [ms.loc[m, 'mean'] * 100 for m in range(1, 13)]\n"
            "    thac = [ms.loc[m, 'tstat_hac'] for m in range(1, 13)]\n"
            "else:\n"
            "    crit = R['bonferroni_crit']\n"
            "    means = [R['month'][m][0] for m in range(1, 13)]\n"
            "    thac = [R['month'][m][3] for m in range(1, 13)]\n"
            "names = " + repr(MONTH_NAMES) + "\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.6, 6.6), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [3, 2]})\n"
            "cols = [GREEN if abs(t) >= crit else (AMBER if m == 9 else GREY)\n"
            "        for m, t in zip(range(1, 13), thac)]\n"
            "a1.bar(names, means, color=cols, width=.62)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean monthly return (%)')\n"
            "a1.set_title('Only January clears Bonferroni -- September is unremarkable')\n"
            "a2.bar(names, thac, color=cols, width=.62)\n"
            "a2.axhline(crit, ls='--', c=RED, lw=1); a2.axhline(-crit, ls='--', c=RED, lw=1)\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('HAC t (one-sample)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Bonferroni bar: |t| >= {crit:.2f}')\n"
            "print({n: round(t, 2) for n, t in zip(names, thac)})"
        ),
        md(
            f"> 💡 In plain words: **{R['n_bonferroni_survive']}/12** months clear the Bonferroni "
            f"bar — **January** (HAC *t* = {R['best_t']:+.2f}), not September (HAC *t* = "
            f"{R['month'][9][3]:+.2f}). If you tested all 12 months and only report the one that "
            "worked, you're not reporting a seasonal — you're reporting the winner of a 12-way "
            "lottery. Correcting for that lottery leaves September with nothing."
        ),
        md(
            "### 4b · September vs the rest, with an honest interval\n\n"
            "Welch *t* of the group split, plus a circular block-bootstrap 95% CI (5,000 draws, "
            "12-month blocks, respecting the annual seasonal structure) on the spread."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sep = st.month_vs_rest(RET, data.STRONG_MONTHS)\n"
            "    ci = st.spread_bootstrap_ci(RET, data.STRONG_MONTHS)\n"
            "    spread, t_, lo, hi = sep['spread']*100, sep['t'], ci['lo']*100, ci['hi']*100\n"
            "else:\n"
            "    spread, t_, lo, hi = R['sep_spread'], R['sep_t'], R['sep_ci_lo'], R['sep_ci_hi']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.errorbar([0], [spread], yerr=[[spread - lo], [hi - spread]], fmt='o',\n"
            "            color=RED, ecolor=GREY, elinewidth=2.5, capsize=8, markersize=10)\n"
            "ax.axhline(0, c='k', lw=1, ls='--')\n"
            "ax.set_xticks([0]); ax.set_xticklabels(['September minus\\nother-11-months spread'])\n"
            "ax.set_ylabel('spread (%, monthly)')\n"
            "ax.set_title(f'Point estimate {spread:+.2f}% -- CI [{lo:+.2f}%, {hi:+.2f}%] straddles zero')\n"
            "ax.set_xlim(-0.6, 0.6)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'spread {spread:+.2f}%  Welch t = {t_:+.2f}  bootstrap CI [{lo:+.2f}%, {hi:+.2f}%]')"
        ),
        md(
            f"> 💡 In plain words: the point estimate is **negative** ({R['sep_spread']:+.2f}%) — "
            "the opposite sign from the claim — and the 95% interval spans more than three times "
            "the point estimate in either direction. There is no version of \"September is "
            "special\" that survives this interval."
        ),
        md(
            "### 4c · Summer vs the rest — the other half of the claim"
        ),
        code(
            "if HAVE_REAL:\n"
            "    summ = st.month_vs_rest(RET, data.SUMMER_MONTHS)\n"
            "    a, b, t_ = summ['mean'], summ['rest_mean'], summ['t']\n"
            "else:\n"
            "    a, b, t_ = R['sum_mean']/100, R['sum_rest_mean']/100, R['sum_t']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.2))\n"
            "ax.bar(['summer (May-Aug)', 'other 8 months'], [a*100, b*100], color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([a*100, b*100]): ax.annotate(f'{v:+.2f}%', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean monthly return (%)')\n"
            "ax.set_title(f'Directionally a lull (Welch t = {t_:+.2f}) but not certified')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'summer {a*100:+.2f}%  vs  rest {b*100:+.2f}%   Welch t = {t_:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: summer underperforms the rest of the year by "
            f"{abs(R['sum_spread']):.2f} points/month at *t* = {R['sum_t']:+.2f} — directionally "
            "consistent with the lull story, well short of the |t| = 2 bar, and summer is still "
            f"**positive on average** ({R['sum_mean']:+.2f}%/mo), not the dead season implied."
        ),
        md(
            "### 4d · The era contrast — justified split, tested as a difference\n\n"
            f"Split at **{R['era_split']}** (the 2013-04-12/15 gold crash — externally dated, "
            "chosen ex ante, not snooped)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ec = st.era_contrast(RET, data.STRONG_MONTHS, data.ERA_SPLIT)\n"
            "    e, l = ec['early_mean'], ec['late_mean']\n"
            "    et, lt, dt = ec['welch_t_early'], ec['welch_t_late'], ec['welch_t_diff']\n"
            "else:\n"
            "    e, l = R['era_early']/100, R['era_late']/100\n"
            "    et, lt, dt = R['era_early_t'], R['era_late_t'], R['era_diff_t']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "ax.bar(['2004 - 2013-04\\n(n=8)', '2013-04 - 2026\\n(n=13)'], [e*100, l*100],\n"
            "       color=[AMBER, GREY], width=.55)\n"
            "for i, (v, t_) in enumerate([(e, et), (l, lt)]):\n"
            "    ax.annotate(f'{v*100:+.2f}%\\n(within-era t={t_:+.2f})', (i, v*100), ha='center', va='top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('September mean return (%)')\n"
            "ax.set_title(f'Neither era certifies September (diff t = {dt:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'early {e*100:+.2f}% (t={et:+.2f})  late {l*100:+.2f}% (t={lt:+.2f})  diff t = {dt:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: September's point estimate flips from "
            f"{R['era_early']:+.2f}% pre-2013 to {R['era_late']:+.2f}% since — but **neither era "
            f"is individually significant** (*t* = {R['era_early_t']:+.2f} and "
            f"{R['era_late_t']:+.2f}), and the difference itself is not certified "
            f"(*t* = {R['era_diff_t']:+.2f}). There is no honest reading in which September was "
            "once a real effect that later decayed — it was never significant in either era."
        ),
        md(
            "### 4e · The third axis — the honest timer test\n\n"
            "Long GLD only in September (calendar-known rule, zero look-ahead — the monthly "
            "return already spans the August close → September close), cash (^IRX) the other 11 "
            "months; 2 one-way legs/yr × cost × NAV on the active month. Both sides raced "
            "excess-of-cash."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bh = st.summary(RET, rf=CASH)\n"
            "    timer = st.strong_month_timer(RET, data.STRONG_MONTHS, cash=CASH)\n"
            "    sg = st.summary(timer, rf=CASH)\n"
            "    net5 = st.apply_timer_costs(timer, data.STRONG_MONTHS, cost_bps_one_way=5.0)\n"
            "    net10 = st.apply_timer_costs(timer, data.STRONG_MONTHS, cost_bps_one_way=10.0)\n"
            "    s5, s10 = st.summary(net5, rf=CASH), st.summary(net10, rf=CASH)\n"
            "    hr = st.hit_rate(RET, data.STRONG_MONTHS)\n"
            "    bh_s, bh_c = bh['sharpe'], bh['cagr']*100\n"
            "    t_s, t_c = s5['sharpe'], s5['cagr']*100\n"
            "    hit_pct = hr['rate']*100\n"
            "else:\n"
            "    bh_s, bh_c = R['bh_sharpe'], R['bh_cagr']\n"
            "    t_s, t_c = R['timer_net5_sharpe'], R['timer_net5_cagr']\n"
            "    hit_pct = R['sep_hit_pct']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar(['buy & hold', 'timer, net'], [bh_s, t_s], color=[GREEN, RED], width=.55)\n"
            "for i, v in enumerate([bh_s, t_s]): a1.annotate(f'{v:+.2f}', (i, v), ha='center',\n"
            "    va='bottom' if v > 0 else 'top')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('Sharpe, excess of cash')\n"
            "a1.set_title('Excess-of-cash Sharpe')\n"
            "a2.bar(['buy & hold', 'timer, net'], [bh_c, t_c], color=[GREEN, RED], width=.55)\n"
            "for i, v in enumerate([bh_c, t_c]): a2.annotate(f'{v:+.1f}%', (i, v), ha='center', va='bottom')\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('CAGR (%/yr)')\n"
            "a2.set_title('CAGR, net of costs')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'buy&hold Sharpe {bh_s:+.2f} CAGR {bh_c:+.1f}%   timer(net) Sharpe {t_s:+.2f} '\n"
            "      f'CAGR {t_c:+.1f}%   Sept hit rate {hit_pct:.1f}%')"
        ),
        md(
            f"> 💡 In plain words: buy-and-hold earns **{R['bh_cagr']:+.2f}%/yr** at excess-of-cash "
            f"Sharpe **{R['bh_sharpe']:+.2f}**; the September-only timer nets "
            f"**{R['timer_net5_cagr']:+.2f}%/yr** at Sharpe **{R['timer_net5_sharpe']:+.2f}** — "
            "*negative*. The timer's hit rate on its one active month is "
            f"**{R['sep_hit_pct']:.1f}%** (Wilson [{R['sep_hit_lo']:.1f}%, {R['sep_hit_hi']:.1f}%]) "
            "— nominally below a coin flip. Sitting in cash 11 months a year, on an asset that "
            "compounded at high single digits for two decades, forfeits almost the entire return "
            "for a month that isn't even special."
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic i.i.d. monthly gold-return world, TUNABLE planted September premium / "
            "summer discount. The null (seasonal = 0) is checked over **20 seeds** — never a "
            "single stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    df = data.synthetic_world(seasonal=0.0, seed=649 + s_)\n"
            "    null_ts.append(st.synthetic_detect(df, data.STRONG_MONTHS)['t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "df = data.synthetic_world(seasonal=0.03, seed=649)\n"
            "planted_t = st.synthetic_detect(df, data.STRONG_MONTHS)['t']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (seasonal=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5, label='planted September premium')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Welch t (September vs rest)')\n"
            "ax.set_title('Control: the null rarely fires; a planted premium lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and fires roughly at the "
            f"nominal false-positive rate ({R['syn_null_fire']}/20); a planted September premium "
            f"reads t = {R['syn_planted_t']:.2f}. The machinery is unbiased — the real-tape "
            f"t = {R['sep_t']:.2f} is the genuine, honest measurement, not a detector failing to "
            "look. *(A faithful-engine / power check only — never cited in support of the "
            "real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — September's mean monthly return is **{R['sep_mean']:+.2f}%** "
            f"vs **{R['sep_rest_mean']:+.2f}%** for the rest of the year (wrong sign), Welch t = "
            f"**{R['sep_t']:+.2f}**, bootstrap CI **[{R['sep_ci_lo']:+.2f}%, "
            f"{R['sep_ci_hi']:+.2f}%]**. Only **{R['n_bonferroni_survive']}/12** months clear the "
            f"Bonferroni bar (**January**, HAC t = {R['best_t']:+.2f}) — a different, untested "
            f"claim. Summer is directionally softer ({R['sum_t']:+.2f}) but not certified, and "
            f"the pre/post-2013 era contrast (diff t = {R['era_diff_t']:+.2f}) shows September "
            "was never significant in either era.\n"
            f"- **Tradability `MIRAGE`** — the September-only timer nets "
            f"**{R['timer_net5_cagr']:+.2f}%/yr** at Sharpe **{R['timer_net5_sharpe']:+.2f}** vs "
            f"buy-and-hold's **{R['bh_cagr']:+.2f}%/yr** at Sharpe **{R['bh_sharpe']:+.2f}**, with "
            f"a {R['sep_hit_pct']:.1f}% hit rate — below a coin flip.\n"
            "- **\"Gold's best month\"? `BUSTED`** — the mechanism (Indian physical demand) is "
            "real; the price pattern on the tradable ETF is not."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson is multiple testing.** Any commodity, index or FX pair has 12 "
            "calendar months; testing all of them and reporting the best one without correction "
            "manufactures a \"seasonal\" on pure noise roughly 1 time in 20 per cell, and this "
            "study's own table shows exactly that shape (January nominally survives; September, "
            "the one everyone talks about, doesn't).\n"
            "- **A cleaner physical-demand test** would use World Gold Council quarterly demand "
            "data or Indian import statistics directly, rather than a US ETF's price — the "
            "physical flow and the dollar price are two different objects, and this study "
            "deliberately tests the one you can actually trade.\n"
            "- **Dedup map:** [289-diwali-muhurat](../../289-diwali-muhurat/) (the same festival, "
            "Indian *equities*), [69-safe-haven](../../69-safe-haven/) (gold's inflation/crash "
            "behavior), [580-gold-lease-rate](../../580-gold-lease-rate/) (a microstructure "
            "carry signal), [640-gold-overnight](../../640-gold-overnight/) (the daily clock, not "
            "month-of-year), [305-gold-oil-ratio](../../305-gold-oil-ratio/) and "
            "[113-gold-silver-ratio](../../113-gold-silver-ratio/) (cross-asset ratios, no "
            "calendar axis).\n\n"
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
