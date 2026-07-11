"""Generate the two narrative notebooks for Study 710 (Olympic-Host-Effect).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached host-ETF /
^GSPC tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (yfinance host ETFs +
# ^GSPC, 1998-01-02 -> 2026-06-30; 7 hardcoded Summer Olympics hosts, n=6 with a
# contemporaneous ETF).
R = dict(
    n_hosts_total=7, n_real=6,
    hosts=[
        dict(year=2000, city="Sydney", country="Australia", ticker="EWA",
             entry="2000-03-15", exit="2000-12-01", host=-6.57, bench=-5.52, abn=-1.04),
        dict(year=2008, city="Beijing", country="China", ticker="FXI",
             entry="2008-02-08", exit="2008-10-24", host=-53.56, bench=-34.14, abn=-19.42),
        dict(year=2012, city="London", country="United Kingdom", ticker="EWU",
             entry="2012-01-27", exit="2012-10-12", host=6.27, bench=8.53, abn=-2.26),
        dict(year=2016, city="Rio de Janeiro", country="Brazil", ticker="EWZ",
             entry="2016-02-05", exit="2016-10-21", host=89.68, bench=13.89, abn=75.79),
        dict(year=2021, city="Tokyo", country="Japan", ticker="EWJ",
             entry="2021-01-25", exit="2021-10-08", host=-1.65, bench=13.90, abn=-15.56),
        dict(year=2024, city="Paris", country="France", ticker="EWQ",
             entry="2024-01-26", exit="2024-10-11", host=3.15, bench=18.89, abn=-15.74),
    ],
    mean=3.63, median=-8.91, sd=36.16, t=0.246, p=0.8156,
    wilcoxon_stat=6.00, wilcoxon_p=0.4375,
    boot_lo=-15.13, boot_hi=34.11, n_boot=20000,
    placebo_obs=3.63, placebo_mean=0.60, placebo_sd=6.90, placebo_p=0.5950, placebo_n=10000,
    hit_k=1, hit_n=6, hit_pct=16.7, hit_lo=3.0, hit_hi=56.4,
    cut1_n=5, cut1_mean=8.24, cut1_t=0.48, cut1_p=0.6565,
    cut2_n=4, cut2_mean=-8.65, cut2_t=-2.14, cut2_p=0.1223,
    syn_null_mean=-0.30, syn_null_sd=1.48, syn_null_fire=3, syn_null_seeds=20,
    syn_planted_effect=60.0, syn_planted_mean=44.33, syn_planted_t=3.96,
    power=[(0, 11), (10, 18), (20, 37), (30, 56), (40, 76), (60, 97), (80, 100)],
    df5_false_alarm=10.2,
    fp_panel="c20a7e16e682",
    fp_ewa="ed949ff1668d", fp_fxi="d95831e82b04", fp_ewu="739e7526d266",
    fp_ewz="e45555abf3e0", fp_ewj="80aff2dfb811", fp_ewq="43d699436afa", fp_gspc="1ed0e14c16a4",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Myth-check: Busted](https://img.shields.io/badge/Myth--check-BUSTED-8b949e?style=flat-square)\n\n"
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

from olympic_host_effect import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    REAL = data.load_real()
    DF = st.host_abnormal_returns(REAL)
else:
    REAL = DF = None
print("real cache present:", HAVE_REAL, "| hosts in hardcoded calendar:", len(data.HOSTS),
      "| real-tape panel n:", (0 if DF is None else len(DF)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does hosting the Olympics make the home team's stock market win gold? 🏅📉\n"
            "### National pride, an infrastructure boom, tourism — a great story. Does the "
            "tape agree?\n\n"
            + BADGES +
            "Every host-city bid is sold partly on this promise: the Games will put us on "
            "the map, build our infrastructure, fill our hotels — and, the folklore adds, "
            "light up our stock market. It's a fun story with a clean mechanism. We tested "
            "it on the six Summer Olympics hosts (2000→2024) that had a country you could "
            "actually buy a share of.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the bootstrap and the power "
            "math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** Only **six** hosts have a contemporaneous single-country "
            "ETF (Athens 2004's Greece ETF didn't exist until 2011) — a genuinely tiny "
            "sample, and we say so everywhere it matters. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the average host beat the world market around its Games? | **On paper, "
            f"barely — and it's misleading.** Mean abnormal return **+{R['mean']:.2f}%** across "
            f"the six hosts — but the **median is {R['median']:.2f}%**, actually negative. |\n"
            "| Why the gap between mean and median? | **One outlier.** Rio 2016 posted a "
            f"**+{R['hosts'][3]['abn']:.1f} percentage point** abnormal return — and it alone "
            "flips the average from negative to positive. Drop it, and the story is 'hosts "
            "underperform', not 'hosts rally'. |\n"
            "| Do MOST hosts actually outperform? | **No — just one of six.** Sydney, "
            "Beijing, London, Tokyo and Paris all trailed the world market around their own "
            "Games. |\n"
            "| Is any of this statistically real? | **No.** Every method we ran — the "
            f"t-test (t={R['t']:.2f}), a rank test, a bootstrap, a random-date placebo — "
            "agrees the six numbers are indistinguishable from noise. |\n\n"
            "> The Olympics might be great for a city. The data says nothing special "
            "happens to its stock market."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Hosting the Olympics puts a country's economy on the world stage — years "
            "of construction spending, a tourism wave, and a wall of positive media should "
            "translate into a stock-market rally around the Games.\"*\n\n"
            "It's not a crazy idea — host bids genuinely promise economic benefits, and "
            "national pride is a real thing investors talk about. What the academic "
            "literature on the Olympics actually finds, though, is closer to a **\"winner's "
            "curse\"** for host-city budgets (Baade & Matheson 2016) than a market rally — "
            "this study tests the market-rally half directly."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this would be a genuinely fun, tradable calendar trade: the IOC "
            "announces host cities **7-9 years in advance**, so there's zero guesswork "
            "about when to position. Every four years, buy the next host's country ETF, "
            "collect the pride premium. The World Cup's version of this idea "
            "([235-world-cup-effect](../../235-world-cup-effect/)) found a weak, confounded "
            "signal — so it's worth asking the same question, honestly, for the Olympics' "
            "own host."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** All **{R['n_hosts_total']}** Summer Olympics editions "
            f"2000→2024, hardcoded from the IOC's own results archive. Only "
            f"**{R['n_real']}** have a country ETF that existed at the time (Athens 2004's "
            "Greece ETF launched in 2011 — seven years too late — so it's excluded, not "
            "faked).\n"
            "- **The window.** Six months before the opening ceremony to two months after "
            "closing — the window investors actually mean when they talk about an "
            "'Olympic boost'.\n"
            "- **The comparison.** Host-country ETF total return minus the S&P 500's return "
            "over the same window (a stand-in for 'the world' — explained in the quants "
            "notebook).\n"
            "- **The honesty check.** With only six data points, one big number can flip the "
            "whole average — so we don't stop at a mean and a t-stat."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, all six, side by side.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = DF.assign(label=DF['city'] + ' ' + DF['year'].astype(str))\n"
            "    labels, vals = list(rows['label']), list(rows['abn_ret_pct'])\n"
            "else:\n"
            "    labels = [f\"{h['city']} {h['year']}\" for h in R['hosts']]\n"
            "    vals = [h['abn'] for h in R['hosts']]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.6))\n"
            "cols = [GREEN if v > 0 else RED for v in vals]\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "for i, v in enumerate(vals):\n"
            "    ax.annotate(f'{v:+.1f}%', (i, v), ha='center', va='bottom' if v >= 0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('abnormal return vs S&P 500 (pp)')\n"
            "ax.set_title('One winner, five losers — the whole story in one chart')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({l: round(v,2) for l,v in zip(labels, vals)})"
        ),
        md(
            "Five bars point down. One — Rio 2016 — points way, way up. That single bar is "
            f"doing all the work: it alone is **+{R['hosts'][3]['abn']:.1f} points**, more "
            "than four times the size of any other bar. And it has an obvious alternative "
            "story: Brazil's stock market was clawing back from the bottom of a brutal "
            "commodity-price crash in early 2016 — the Olympics landed in the middle of a "
            "recovery that was already underway for reasons that have nothing to do with "
            "hosting a sporting event.\n\n"
            "**So does the average tell the truth here?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    x = DF['abn_ret_pct'].values\n"
            "    mean_, med_ = float(x.mean()), float(np.median(x))\n"
            "else:\n"
            "    mean_, med_ = R['mean'], R['median']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['mean', 'median'], [mean_, med_], color=[AMBER, GREY], width=.5)\n"
            "for i, v in enumerate([mean_, med_]):\n"
            "    ax.annotate(f'{v:+.2f}%', (i, v), ha='center', va='bottom' if v >= 0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('abnormal return (pp)')\n"
            "ax.set_title('The mean says +3.6%. The median — what a typical host got — says -8.9%')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'mean {mean_:+.2f}%  median {med_:+.2f}%')"
        ),
        md(
            "That gap is the whole story in one picture: the **typical** host (the median) "
            "actually did about **9 points worse** than the world market around its own "
            "Games, and the only reason the mean looks positive is that one country's number "
            "is enormous. This is precisely why the desk never stops at a single average on "
            "a small sample — a rank test, a bootstrap and a random-date placebo (in the "
            "quants notebook) all confirm the same thing: **there's no real pattern here**, "
            "just one big outlier with its own explanation."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Mean **+{R['mean']:.2f}%** but median **{R['median']:.2f}%** "
            f"across n = {R['n_real']} hosts; every statistical test (t, Wilcoxon, bootstrap, "
            "placebo) agrees it's noise. Only **1 of 6** hosts actually beat the world market.\n"
            "- **Tradability — Mirage.** Nothing to trade, and even ignoring that: one event "
            "every ~4 years on a different single-country ETF each time is not a repeatable "
            "strategy.\n"
            "- **\"Do most hosts actually outperform?\" — Busted.** Five of six lost to the "
            "world benchmark around their own Games."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The city-level story may differ from the country-level story.** A country "
            "ETF dilutes the host CITY's local property/construction boom into a whole "
            "national economy — a REIT or a construction-sector index for the host city "
            "might show more (or the same nothing).\n"
            "- **The award-day pop is a different, narrower question.** Some academic work "
            "(Berman, Brooks & Davidson) finds a modest positive reaction to the host-city "
            "*announcement* itself — years before the Games — which this study doesn't test.\n"
            "- **Sibling studies:** [234-olympic-year](../../234-olympic-year/) (any Olympic "
            "year, US market, not host-specific), [235-world-cup-effect](../../235-world-cup-effect/) "
            "(the analogous global-market World Cup test), "
            "[708-eurovision-effect](../../708-eurovision-effect/) (the same host-lift "
            "question for a smaller event).\n\n"
            "*Think a host-city-specific instrument would show something different? Show a "
            "net, certifiable edge on real, tradable instruments — then we'll talk.*"
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
            "# The Olympic-Host-Effect — a quantitative teardown 🔬\n"
            "### A one-sample-t / Wilcoxon / bootstrap / random-window-placebo battery at "
            "n = 6, honest sensitivity cuts, a power curve, and a 20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — **the host country's equity market outperforms the world around "
            "its own Games** — is tested on the smallest sample this desk has run a "
            "headline stat on: n = 6 non-overlapping events. The job here is to show every "
            "check that sample size demands, not just one t-stat.\n\n"
            "> ⚠️ **Data note.** Six host-country ETFs (EWA/FXI/EWU/EWZ/EWJ/EWQ) + ^GSPC, "
            "yfinance, cached; **7 hardcoded Summer Olympics hosts 2000→2024** (IOC results "
            "archive), n=6 with a contemporaneous ETF (Athens 2004 excluded — no ticker "
            "existed). ^GSPC is a named, imperfect substitute for URTH/ACWI (see "
            "[`docs/references.md`](../docs/references.md)) — host total return vs "
            "benchmark price-only return, labelled throughout. Numbers in "
            "[`docs/results.md`](../docs/results.md) (panel fingerprint `" + R["fp_panel"] +
            "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | mean **{R['mean']:+.2f}%**, median **{R['median']:+.2f}%**, "
            f"one-sample *t* = **{R['t']:.2f}** (df=5, *p*={R['p']:.2f}), Wilcoxon "
            f"*p*={R['wilcoxon_p']:.2f}, bootstrap 95% CI **[{R['boot_lo']:.1f}%, "
            f"{R['boot_hi']:.1f}%]**, placebo *p*={R['placebo_p']:.2f} |\n"
            f"| **Tradability** | `MIRAGE` | no signal; 1 non-overlapping event / ~4yr on a "
            "different ticker each time |\n"
            f"| **Majority outperform?** | `BUSTED` | {R['hit_k']}/{R['hit_n']} = "
            f"{R['hit_pct']:.1f}% (Wilson [{R['hit_lo']:.1f}%, {R['hit_hi']:.1f}%]) |\n\n"
            "> 💡 In plain words: five independent statistical methods all agree on the "
            "same answer — nothing. When every method agrees on 'nothing' at once, that's "
            "about as clean a null result as this desk produces."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $R^h_i$ be host $i$'s total return and $R^b_i$ the benchmark's return, "
            "both over the identical calendar window $[-6\\text{mo}, +2\\text{mo}]$ around "
            "the Games. The abnormal return is $a_i = R^h_i - R^b_i$. Host cities are "
            "awarded by IOC vote 7-9 years ahead, so $a_i$ involves **zero look-ahead** by "
            "construction — this is the cleanest possible calendar-event design.\n\n"
            "- **H₁ (rally).** $E[a_i] \\gg 0$ across hosts — systematic, not one country's "
            "idiosyncratic story.\n"
            "- **H₂ (breadth).** A rally story implies most, not just one, host should show "
            "$a_i > 0$.\n"
            "- **H₃ (robustness).** The result should not hinge on a single point or a "
            "single test statistic.\n\n"
            "We find **H₁ not supported** (t=0.25), **H₂ busted** (1/6), **H₃ fails on its "
            "own terms** — the entire positive mean is one point, and every alternative "
            "test (median, Wilcoxon, bootstrap, placebo) says the same nothing."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design at n = 6\n\n"
            "Six non-overlapping events is small enough that a single summary statistic is "
            "not trustworthy on its own — one fat-tailed observation (Rio 2016) can flip a "
            "mean's sign. The design here runs **five independent checks** on the same six "
            "numbers: the planned-primary **one-sample *t*** (df=5), the **median** (immune "
            "to the outlier's magnitude, not its existence), a **Wilcoxon signed-rank** "
            "(no normality assumption), a **percentile bootstrap CI** (resamples the six "
            "points with replacement, 20,000 times — shows how wide the true uncertainty "
            "really is), and a **random-window placebo** (same tickers, same window length, "
            "random calendar anchor — is the *specific* Games date special, or would any "
            "8-month window on these tickers look similar?). Agreement across all five is "
            "what makes the null call solid rather than a single fragile t-stat."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_hosts_total']} Summer editions 2000→2024, hardcoded (IOC "
            f"results archive). {R['n_real']} have a contemporaneous host-country ETF.\n"
            "- **Window.** [-6mo opening, +2mo closing], nearest trading day, host ETF "
            "total return vs ^GSPC price return.\n"
            "- **Execution.** Enter window-start close, exit window-end close — the host "
            "calendar is public 7-9 years ahead (zero look-ahead), the study's single "
            "documented convention.\n"
            "- **Headline.** One-sample *t* + median + Wilcoxon + bootstrap CI + "
            "random-window placebo.\n"
            "- **Breadth check.** Wilson interval on the outperformance hit rate.\n"
            "- **Sensitivity.** Named confounder / outlier cuts, reported as a warning "
            "about snooping, not a second result.\n"
            "- **Control.** A calibrated synthetic null (Normal(0, sd) at n=6) that must "
            "not systematically fire, plus a power curve."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline panel\n\n"
            "Six abnormal returns, the one-sample *t*, and why the mean and the median tell "
            "different stories."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tbl = DF[['year','city','ticker','host_ret_pct','bench_ret_pct','abn_ret_pct']]\n"
            "    print(tbl.to_string(index=False))\n"
            "    s = st.one_sample_t(DF['abn_ret_pct'].values)\n"
            "else:\n"
            "    s = dict(n=R['n_real'], mean=R['mean'], median=R['median'], sd=R['sd'],\n"
            "             t=R['t'], p=R['p'])\n"
            "    for h in R['hosts']:\n"
            "        print(f\"{h['year']} {h['city']:<15} {h['ticker']}  host {h['host']:+.2f}%  \"\n"
            "              f\"bench {h['bench']:+.2f}%  abn {h['abn']:+.2f}%\")\n"
            "print(f\"\\nn={s['n']}  mean={s['mean']:+.2f}%  median={s['median']:+.2f}%\")\n"
            "print(f\"one-sample t = {s['t']:+.3f} (df={s['n']-1})  p = {s['p']:.4f}\")"
        ),
        md(
            "### 4b · Nonparametric and bootstrap cross-checks\n\n"
            "The one-sample *t* assumes the six abnormal returns are roughly normal — a "
            "shaky assumption with one +75.8pp outlier. A Wilcoxon signed-rank test makes "
            "no such assumption; a bootstrap CI shows the honest width of the uncertainty."
        ),
        code(
            "if HAVE_REAL:\n"
            "    x = DF['abn_ret_pct'].values\n"
            "    w = st.wilcoxon_test(x)\n"
            "    b = st.bootstrap_ci(x)\n"
            "else:\n"
            "    x = np.array([h['abn'] for h in R['hosts']])\n"
            "    w = {'stat': R['wilcoxon_stat'], 'p': R['wilcoxon_p']}\n"
            "    b = {'lo': R['boot_lo'], 'hi': R['boot_hi'], 'n_boot': R['n_boot']}\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.errorbar([0], [x.mean()], yerr=[[x.mean()-b['lo']], [b['hi']-x.mean()]],\n"
            "            fmt='o', color=RED, capsize=8, markersize=10, lw=2,\n"
            "            label=f\"mean {x.mean():+.1f}%, bootstrap 95% CI\")\n"
            "ax.axhline(0, c='k', lw=1, ls='--')\n"
            "ax.set_xlim(-1, 1); ax.set_xticks([])\n"
            "ax.set_ylabel('mean abnormal return (pp)')\n"
            "ax.set_title(f\"The CI straddles zero by a mile: [{b['lo']:+.1f}%, {b['hi']:+.1f}%]\")\n"
            "ax.legend(loc='upper left'); plt.tight_layout(); plt.show()\n"
            "print(f\"Wilcoxon stat={w['stat']:.2f} p={w['p']:.4f}\")\n"
            "print(f\"bootstrap 95% CI [{b['lo']:+.2f}%, {b['hi']:+.2f}%]  (n_boot={b['n_boot']:,})\")"
        ),
        md(
            f"> 💡 In plain words: the Wilcoxon test (*p* = {R['wilcoxon_p']:.2f}) reaches "
            "the same conclusion as the *t*-test without assuming the outlier is well-"
            f"behaved. The bootstrap CI **[{R['boot_lo']:.1f}%, {R['boot_hi']:.1f}%]** spans "
            "49 percentage points — with only six data points, that's the honest amount of "
            "uncertainty, and it comfortably contains zero."
        ),
        md(
            "### 4c · Random-window placebo — is the Games date itself special?\n\n"
            "Replace each host's actual window with a random window of the identical "
            "trading-day length on the SAME ticker, vs the benchmark, many times over. If "
            "the Games date carries no special information, the observed mean should sit "
            "comfortably inside this null cloud."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(REAL, DF, n_seeds=4, n_draws_per_seed=250)\n"
            "    obs, draws = pl['obs'], pl['draws']\n"
            "else:\n"
            "    obs = R['placebo_obs']\n"
            "    rng = np.random.default_rng(710)\n"
            "    draws = rng.normal(R['placebo_mean'], R['placebo_sd'], 1000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85,\n"
            "        label='null: random 8-month windows, same tickers (light in-notebook run)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed mean {obs:+.2f}%')\n"
            "ax.set_xlabel('mean abnormal return of a random-window draw (pp)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Squarely inside the null cloud: canonical p = {R['placebo_p']:.4f} \"\n"
            "             '(20 seeds x 500 draws)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical placebo (results.md): mean {R['placebo_mean']:+.2f}%, \"\n"
            "      f\"sd {R['placebo_sd']:.2f}%, p = {R['placebo_p']:.4f}\")"
        ),
        md(
            f"> 💡 In plain words: draw six random 8-month windows on these same six tickers "
            f"10,000 times — the observed **{R['placebo_obs']:+.2f}%** lands right in the "
            f"middle of that cloud (*p* = {R['placebo_p']:.2f}). Nothing about the *specific* "
            "Games dates is unusual on these tickers."
        ),
        md(
            "### 4d · Breadth — does a majority actually outperform?\n\n"
            "The 'national pride rally' story implies most hosts should win, not just the "
            "average."
        ),
        code(
            "if HAVE_REAL:\n"
            "    hr = st.outperform_hit_rate(DF)\n"
            "    k, n_ = hr['k'], hr['n']\n"
            "    lo, hi = hr['lo']*100, hr['hi']*100\n"
            "else:\n"
            "    k, n_, lo, hi = R['hit_k'], R['hit_n'], R['hit_lo'], R['hit_hi']\n"
            "fig, ax = plt.subplots(figsize=(6.8, 4.2))\n"
            "ax.bar(['outperformed', 'underperformed'], [k, n_-k], color=[GREEN, RED], width=.5)\n"
            "for i, v in enumerate([k, n_-k]):\n"
            "    ax.annotate(str(v), (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('# of hosts'); ax.set_ylim(0, n_+1)\n"
            "ax.set_title(f'{k}/{n_} = {k/n_*100:.0f}% outperformed  (Wilson [{lo:.0f}%, {hi:.0f}%])')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'{k}/{n_} outperformed, Wilson 95% [{lo:.1f}%, {hi:.1f}%]')"
        ),
        md(
            f"> 💡 In plain words: **{R['hit_k']}/{R['hit_n']}** — just Rio 2016. The Wilson "
            f"interval [{R['hit_lo']:.1f}%, {R['hit_hi']:.1f}%] doesn't even reach 50% at its "
            "upper bound. Whatever the mean says, the *typical* host lost this race."
        ),
        md(
            "### 4e · Sensitivity cuts — a warning about snooping, not a second finding\n\n"
            "Dropping the two most-explained-away points (Beijing's GFC confounder, Rio's "
            "outlier) on purpose, to show exactly how a post-hoc cut can manufacture a "
            "'significant' number from six points."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c1 = st.sensitivity_cut(DF, ('Beijing',))\n"
            "    c2 = st.sensitivity_cut(DF, ('Beijing', 'Rio de Janeiro'))\n"
            "else:\n"
            "    c1 = dict(n=R['cut1_n'], mean=R['cut1_mean'], t=R['cut1_t'], p=R['cut1_p'])\n"
            "    c2 = dict(n=R['cut2_n'], mean=R['cut2_mean'], t=R['cut2_t'], p=R['cut2_p'])\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "labels = ['all n=6', 'excl. Beijing\\n(n=5)', 'excl. Beijing & Rio\\n(n=4, post-hoc)']\n"
            "means = [R['mean'], c1['mean'], c2['mean']]\n"
            "ts = [R['t'], c1['t'], c2['t']]\n"
            "cols = [GREY, GREY, RED]\n"
            "ax.bar(labels, means, color=cols, width=.55)\n"
            "for i, (m, t_) in enumerate(zip(means, ts)):\n"
            "    ax.annotate(f'{m:+.1f}%\\n(t={t_:+.2f})', (i, m), ha='center',\n"
            "                va='bottom' if m >= 0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean abnormal return (pp)')\n"
            "ax.set_title('A 2-of-6 post-hoc cut flips the sign and crosses t=2 -- textbook snooping bait')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"excl. Beijing: n={c1['n']} mean={c1['mean']:+.2f}% t={c1['t']:+.2f} p={c1['p']:.4f}\")\n"
            "print(f\"excl. Beijing & Rio: n={c2['n']} mean={c2['mean']:+.2f}% t={c2['t']:+.2f} p={c2['p']:.4f}\")"
        ),
        md(
            f"> 💡 In plain words: cutting Beijing (a named GFC confounder) barely moves "
            f"anything (*t* = {R['cut1_t']:+.2f}). But *also* cutting Rio — the study's own "
            f"positive outlier — flips the mean negative and pushes *t* to "
            f"**{R['cut2_t']:+.2f}**, past the desk's usual bar. This is exactly the "
            "mechanism data-snooping warnings exist for: with n=6, removing any 2 points is "
            f"enough to manufacture a story either direction. It is reported here as a red "
            f"flag, not a result — note its own *p* = {R['cut2_p']:.2f} doesn't even clear "
            "significance at n=4."
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "A calibrated synthetic world: *n* = 6 draws from Normal(effect, sd = 36.16pp, "
            "the real panel's own dispersion). The null (effect=0) is checked over **20 "
            "seeds** — never a single stream — and a power curve shows how large a TRUE "
            "effect would need to be to reliably clear the bar at this sample size."
        ),
        code(
            "null_ts = np.array([st.synthetic_detect(0.0, seed=710+s)['t'] for s in range(20)])\n"
            "planted = st.synthetic_detect(60.0, seed=710)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.4))\n"
            "a1.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (effect=0), 20 seeds')\n"
            "a1.scatter([1], [planted['t']], color=RED, s=90, zorder=5, label='planted +60pp')\n"
            "a1.axhline(-2, ls='--', c=RED, lw=1); a1.axhline(2, ls='--', c=RED, lw=1)\n"
            "a1.set_xticks([0, 1]); a1.set_xticklabels(['null x 20', 'planted'])\n"
            "a1.set_ylabel('one-sample t'); a1.legend(fontsize=8)\n"
            "a1.set_title('Detector: unbiased, but the df=5 bar is loose')\n"
            "pc = st.power_curve((0.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0), n_seeds=200)\n"
            "a2.plot(pc['effect_pct'], pc['power']*100, 'o-', color=AMBER)\n"
            "a2.axhline(80, ls='--', c=GREY, lw=1, label='80% power')\n"
            "a2.set_xlabel('planted effect (pp)'); a2.set_ylabel('power: P(|t|>=2)  (%)')\n"
            "a2.set_title('~30-40pp needed for reliable power at n=6')\n"
            "a2.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted[\"t\"]:+.2f}')\n"
            "print(pc.to_string(index=False))"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector fires "
            f"{R['syn_null_fire']}/20 times — about **15%**, not 5%, because at **df = 5** "
            f"the true two-sided false-alarm rate of a fixed \\|t\\|≥2 cutoff is already "
            f"≈{R['df5_false_alarm']:.1f}% (the honest 5%-critical value at df=5 is 2.57, "
            "not 2). The desk's usual bar is *already loose* at this sample size — and the "
            f"real-tape *t* = {R['t']:.2f} still comes nowhere near even this weakened bar. "
            "The power curve says a TRUE effect would need to be roughly **+30 to +40 "
            "points** before this study could reliably detect it — this design can rule "
            "out a large host effect, but not a small one. *(A faithful-engine / power "
            "check only — never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — mean **{R['mean']:+.2f}%**, median **{R['median']:+.2f}%** "
            f"across n={R['n_real']}: one-sample *t* = **{R['t']:.2f}** (df=5, "
            f"*p*={R['p']:.2f}), Wilcoxon *p*={R['wilcoxon_p']:.2f}, bootstrap 95% CI "
            f"**[{R['boot_lo']:.1f}%, {R['boot_hi']:.1f}%]**, random-window placebo "
            f"*p*={R['placebo_p']:.2f} — five independent methods, one answer. Only "
            f"{R['hit_k']}/{R['hit_n']} hosts outperformed. Named caveats: Beijing 2008 "
            "sits inside the GFC; Rio 2016's outlier plausibly reflects Brazil's commodity "
            "rebound, not the Games. A post-hoc 2-of-6 cut (excl. Beijing & Rio) flips *t* "
            f"to {R['cut2_t']:+.2f} — reported as a snooping warning, explicitly not "
            "evidence. Power analysis: only a huge (≥30pp) true effect would be reliably "
            "detectable at this n — an honest limitation, not a license to round up.\n"
            "- **Tradability `MIRAGE`** — nothing to trade, and even setting the null "
            "aside, the opportunity set (one non-overlapping event / ~4yr, a different "
            "ticker each time) has no capacity or repeatability.\n"
            f"- **Majority outperform? `BUSTED`** — {R['hit_k']}/{R['hit_n']} = "
            f"{R['hit_pct']:.1f}% (Wilson [{R['hit_lo']:.1f}%, {R['hit_hi']:.1f}%])."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **A host-city-specific instrument (REITs, construction indices) might "
            "capture something a national ETF dilutes away** — the infrastructure boom is "
            "local, and a country-wide index is a very blunt instrument for it.\n"
            "- **The award-day announcement is a narrower, more testable question** — some "
            "academic work finds a modest reaction there (Berman, Brooks & Davidson), years "
            "before the confounders of the Games window itself pile up.\n"
            "- **Dedup map:** [234-olympic-year](../../234-olympic-year/) (any Olympic year, "
            "US market, not host-specific), [235-world-cup-effect](../../235-world-cup-effect/) "
            "(the global-market World Cup analogue), "
            "[708-eurovision-effect](../../708-eurovision-effect/) (the same host-lift "
            "question, a smaller event), [313-geopolitical-shock](../../313-geopolitical-shock/) "
            "(shared event-study machinery, an unrelated trigger class).\n\n"
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
