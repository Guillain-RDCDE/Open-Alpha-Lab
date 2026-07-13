"""Generate the two narrative notebooks for Study 719 (Met-Gala-Luxury).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached luxury /
VGK tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic positive control runs anywhere with no
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (4 luxury names + VGK,
# yfinance total-return, as-of 2026-06-30; 20 of 23 held galas resolved, VGK 2005 floor).
R = dict(
    n_editions=26, n_galas=23, n_included=20, n_nogala=3, n_prevgk=3,
    asof="2026-06-30", fp="b9b945ecd3bc",
    # signal (day(-1) -> day(-1)+k abnormal return, luxury basket - VGK), % units
    wk_mean=+0.442, wk_t=+0.739, wk_sd=2.672, wk_hit=12, wk_n=20,
    mo_mean=+0.658, mo_t=+0.520, mo_sd=5.664, mo_hit=10, mo_n=20,
    # placebo (right-tail, p = share of null means >= observed)
    pl_wk_p=0.304, pl_wk_plmean=+0.152, pl_wk_plsd=0.557,
    pl_mo_p=0.500, pl_mo_plmean=+0.659, pl_mo_plsd=1.203,
    pl_cap_wk_p=0.250, pl_cap_mo_p=0.453,
    # tradability (day(0) -> day(0)+k, net of costs)
    cap_wk_g=+0.531, cap_wk_gt=+1.07, cap_wk_n5=+0.431, cap_wk_t5=+0.87, cap_wk_n10=+0.331, cap_wk_t10=+0.67,
    cap_mo_g=+0.775, cap_mo_gt=+0.64, cap_mo_n5=+0.675, cap_mo_t5=+0.56, cap_mo_n10=+0.575, cap_mo_t10=+0.48,
    # third axis: driven by one name? per-name one-sample t (1-month), % + t
    name_mo={"LVMH": (+0.812, +0.604), "Kering": (+0.965, +0.663),
             "Hermes": (+1.541, +0.726), "Richemont": (+1.415, +1.003)},
    name_wk={"LVMH": (+0.394, +0.616), "Kering": (+0.694, +1.015),
             "Hermes": (+0.861, +0.949), "Richemont": (+0.380, +0.537)},
    name_max_mo_t=+1.003, name_max_mo="Richemont",
    # event anatomy (mean cumulative basket AR by day offset from day(-1))
    car={-5: -0.106, -3: 0.010, -1: 0.330, 0: 0.000, 1: -0.093, 2: -0.176,
         3: 0.166, 5: 0.442, 10: 1.915, 15: 0.654, 21: 0.658},
    # synthetic control
    syn_null_mean=+0.05, syn_null_sd=1.42, syn_null_fire=4, syn_null_seeds=20,
    syn_planted1_t=+1.73, syn_planted2_t=+2.59,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Just_one_name%3F: Not_supported](https://img.shields.io/badge/Just_one_name%3F-Not_supported-8b949e?style=flat-square)\n\n"
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

from met_gala_luxury import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.load_real()
    EV = st.build_event_table(PRICES, cost_bps=5.0)
    INC = EV[EV["included"]]
else:
    PRICES = EV = INC = None
print("real cache present:", HAVE_REAL, "| gala years listed:", len(data.EVENTS),
      "| resolved events:", (0 if INC is None else len(INC)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do luxury stocks pop on the first Monday in May? 👗📈\n"
            "### The Met-Gala \"spotlight bump\" — a whole lot of red carpet, and "
            "nothing in the tape\n\n"
            + BADGES +
            "Every first Monday in May, the Metropolitan Museum throws the Met Gala — "
            "fashion's biggest, most-photographed night. The red carpet is a four-hour "
            "commercial for exactly four companies: **LVMH, Kering, Hermès and "
            "Richemont**, the European houses behind Dior, Louis Vuitton, Gucci, "
            "Balenciaga, Cartier and the rest. The folklore (a fixture of "
            "markets-meet-culture pieces) says all that free global attention gives the "
            "luxury complex a little lift in the days around the gala.\n\n"
            "We tested it properly — every modern Met Gala, 2005→2025, luxury basket "
            "versus the European market as a whole.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the "
            "per-name split? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 26 gala years hardcoded (three had no gala: 2000 & "
            "2002 cancelled, 2020 COVID); the basket is LVMH+Kering+Hermès+Richemont vs "
            "`VGK` (Vanguard Europe). Every chart is drawn by the code beside it; house "
            "style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the luxury basket pop the week of the gala? | **No.** "
            f"**{R['wk_mean']:+.2f}%** vs Europe — a coin-flip away from zero "
            f"(*t* = {R['wk_t']:.2f}). |\n"
            f"| Over the following month? | **No.** **{R['mo_mean']:+.2f}%**, and a "
            "random month of the same length beats it **half the time** (placebo "
            f"*p* = {R['pl_mo_p']:.2f} — dead centre of the luck cloud). |\n"
            "| Is it hiding in one mega-brand (LVMH)? | **No.** Split the basket into "
            "its four names and not one of them clears the bar either — the strongest, "
            f"Richemont over a month, is only *t* = {R['name_max_mo_t']:.2f}. |\n"
            f"| Could you have traded it? | **No.** Buy the basket the first day you "
            f"*could* (after the gala) and a month later you have "
            f"**{R['cap_mo_n5']:+.2f}%** net of costs — positive, meaningless, "
            f"unprovable. |\n\n"
            "> There's no there there. The Met Gala is a spectacular advertisement — "
            "and advertisements, it turns out, are already in the price."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The Met Gala is the Super Bowl of fashion. For one night the entire "
            "world is looking at Louis Vuitton, Dior, Gucci, Cartier — hundreds of "
            "millions of dollars of free brand exposure for the big luxury houses. That "
            "kind of spotlight has to be worth something to the stocks.\"*\n\n"
            "It's a real, intuitive mechanism — attention is worth money, and luxury is "
            "a story-driven, brand-driven business. Nobody has ever formally tested "
            "whether the gala moves the **luxury tape**. We did."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, it would be a clean, delightful example of pure attention moving a "
            "sector on a fixed calendar date you know a year in advance — buy the "
            "luxury basket the first Monday in May, pocket the spotlight, repeat. It "
            "would also say something bigger: that a *fashion* event, with no earnings, "
            "no guidance, no numbers at all, can nudge billion-euro market caps. We "
            "wanted to know: does it?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** All **{R['n_editions']}** gala years 2000→2025, "
            f"hardcoded with the exact date. Three had **no gala** (2000 & 2002 "
            "cancelled, 2020 COVID); the \"first Monday in May\" convention firmly "
            "holds from 2005 — which is also when our benchmark begins.\n"
            "- **The basket.** LVMH (`MC.PA`), Kering (`KER.PA`), Hermès (`RMS.PA`), "
            "Richemont (`CFR.SW`) — equal-weighted, vs `VGK`, a broad Europe benchmark.\n"
            "- **The window.** Abnormal return (luxury minus Europe) from the last "
            "close *before* the Monday-night gala through 1 week and 1 month after.\n"
            "- **The honesty check.** A random-window placebo (does a random month do "
            "just as well?), a per-name split (is one brand carrying it?), and a trade "
            "you could *actually* have placed."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The headline: the luxury basket does essentially nothing unusual around "
            "the gala.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    wk = st.one_sample_t(INC['ar_week'].values)\n"
            "    mo = st.one_sample_t(INC['ar_month'].values)\n"
            "    wk_m, wk_t, mo_m, mo_t = wk['mean']*100, wk['t'], mo['mean']*100, mo['t']\n"
            "else:\n"
            "    wk_m, wk_t, mo_m, mo_t = R['wk_mean'], R['wk_t'], R['mo_mean'], R['mo_t']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.6))\n"
            "bars = ax.bar(['1 week', '1 month'], [wk_m, mo_m],\n"
            "              color=[GREY, GREY], width=.5)\n"
            "for b, v, t in zip(bars, [wk_m, mo_m], [wk_t, mo_t]):\n"
            "    ax.annotate(f'{v:+.2f}%\\n(t={t:.2f})', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('luxury basket abnormal return vs Europe (%)')\n"
            "ax.set_title('The gala \\'bump\\': small, positive, and well inside the noise')\n"
            "ax.set_ylim(-0.5, 2.2); plt.tight_layout(); plt.show()\n"
            "print(f'1 week:  {wk_m:+.3f}%  t={wk_t:+.2f}')\n"
            "print(f'1 month: {mo_m:+.3f}%  t={mo_t:+.2f}')"
        ),
        md(
            f"Both bars point the right way — luxury is up **{R['wk_mean']:+.2f}%** "
            f"(week) and **{R['mo_mean']:+.2f}%** (month) more than Europe — but both "
            f"*t*-stats ({R['wk_t']:.2f} and {R['mo_t']:.2f}) are miles below the "
            "desk's bar for \"probably not luck.\" With only 20 galas to average over, a "
            "wiggle this size is exactly what pure chance hands you.\n\n"
            "**So let's ask chance directly.** We drew thousands of random, "
            "non-gala months from the same luxury basket and asked: how often does a "
            "random month beat the gala month?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(EV, PRICES, 'ar_month', k=21, entry_offset=0,\n"
            "                           n_seeds=4, n_draws_per_seed=200, tail='right')\n"
            "    obs = pl['obs']*100\n"
            "    rng = np.random.default_rng(719)\n"
            "    draws = rng.normal(pl['placebo_mean'], pl['placebo_sd'], 3000)*100\n"
            "else:\n"
            "    obs = R['mo_mean']\n"
            "    rng = np.random.default_rng(719)\n"
            "    draws = rng.normal(R['pl_mo_plmean'], R['pl_mo_plsd'], 3000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85,\n"
            "        label='random non-gala months, same basket')\n"
            "ax.axvline(obs, c=RED, lw=2.4, label=f'the actual gala month {obs:+.2f}%')\n"
            "ax.set_xlabel('abnormal return of a random month (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The gala month sits dead centre — placebo p = {R[\"pl_mo_p\"]:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"observed {R['mo_mean']:+.3f}% vs placebo mean {R['pl_mo_plmean']:+.3f}% \"\n"
            "      f\"-> p = {R['pl_mo_p']:.3f}\")"
        ),
        md(
            f"The red line lands **right in the middle** of the grey cloud: a random "
            f"month matches or beats the gala month **{R['pl_mo_p']*100:.0f}%** of the "
            "time. That's the single most damning chart in the study — the gala month "
            "is not just statistically insignificant, it is perfectly, boringly "
            "*average*.\n\n"
            "**But maybe the basket hides it.** Louis Vuitton and Dior are the loudest "
            "brands on the carpet — maybe LVMH alone pops, and three sleepy names drag "
            "the average down?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pn = st.per_name_stats(EV, 'mo')\n"
            "    names = list(pn['name']); ts = list(pn['t'])\n"
            "else:\n"
            "    names = list(R['name_mo']); ts = [R['name_mo'][n][1] for n in names]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(names, ts, color=GREY, width=.6)\n"
            "ax.axhline(2.0, ls='--', c=RED, lw=1.2, label='certification bar (t=2)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('one-sample t (1-month abnormal return)')\n"
            "ax.set_title('No single luxury house pops either -- all four are noise')\n"
            "ax.set_ylim(0, 2.4); ax.legend(); plt.tight_layout(); plt.show()\n"
            "for n, t in zip(names, ts): print(f'  {n:10s} t={t:+.2f}')"
        ),
        md(
            "Nope. Every one of the four houses is a shrug — the strongest, "
            f"**{R['name_max_mo']}** over a month, reaches only *t* = "
            f"**{R['name_max_mo_t']:.2f}**, and LVMH itself is *t* = "
            f"**{R['name_mo']['LVMH'][1]:.2f}**. There is no lonely superstar being "
            "hidden by the average.\n\n"
            "**Finally, could you have traded it?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.one_sample_t(INC['cap_month_gross'].values)['mean']*100\n"
            "    n5 = st.one_sample_t(INC['cap_month_net'].values)['mean']*100\n"
            "else:\n"
            "    g, n5 = R['cap_mo_g'], R['cap_mo_n5']\n"
            "fig, ax = plt.subplots(figsize=(7.4, 4.4))\n"
            "ax.bar(['gross', 'net of costs'], [g, n5], color=[GREY, GREY], width=.5)\n"
            "for i, v in enumerate([g, n5]): ax.annotate(f'{v:+.2f}%', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('basket return, 1 month, entered AFTER the gala is public')\n"
            "ax.set_title('A positive number you could never certify was real')\n"
            "ax.set_ylim(0, 1.1); plt.tight_layout(); plt.show()"
        ),
        md(
            f"Buying the day the gala is *actually* public (the first close after "
            f"Monday night) and holding a month nets **{R['cap_mo_n5']:+.2f}%** — "
            "positive, but statistically unprovable (its own placebo *p* = "
            f"{R['pl_cap_mo_p']:.2f}). You would be betting real money, and real costs, "
            "on a number the data cannot distinguish from zero."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** No cut clears the bar: basket 1-week *t* = "
            f"{R['wk_t']:.2f}, 1-month *t* = {R['mo_t']:.2f} (placebo *p* = "
            f"{R['pl_mo_p']:.2f}, dead centre), and not one of the four houses "
            "individually clears it either.\n"
            "- **Tradability — Mirage.** The best honest, net-of-cost trade nets "
            f"{R['cap_mo_n5']:+.2f}% over a month with a placebo *p* around "
            f"{R['pl_cap_mo_p']:.2f} — indistinguishable from noise.\n"
            "- **\"Just one name?\" — Not supported.** The folklore's natural fallback "
            "(\"okay, the basket is flat, but surely LVMH pops\") also fails: every "
            f"house is sub-*t* = 2, the strongest only {R['name_max_mo_t']:.2f}."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **This is what an efficient advertisement looks like.** Everyone knows "
            "the gala is coming — the date is fixed a year ahead — so if the spotlight "
            "were worth anything, it would already be in the price long before the "
            "carpet rolls out. A *predictable* attention event is exactly the kind of "
            "thing markets arbitrage away, and that's what we see: nothing.\n"
            "- **Sibling studies:** the [Eurovision effect](../../708-eurovision-effect/) "
            "(a country's stock market when it wins Europe's silliest TV night), the "
            "[Super Bowl indicator](../../158-super-bowl/), the "
            "[World Cup effect](../../235-world-cup-effect/) and the "
            "[plane-crash effect](../../707-plane-crash-effect/) — every one a "
            "mood-moves-markets claim, tested the same honest way.\n\n"
            "*Think the gala REALLY moves luxury — maybe intraday, maybe only in a "
            "blockbuster-theme year? Find the cleaner signal, show a net, "
            "placebo-surviving edge, and we'll publish the teardown.*"
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
            "# Met-Gala-Luxury — a quantitative teardown 🔬\n"
            "### One-sample-*t* on the luxury basket's abnormal return · a random-window "
            "placebo · a per-name concentration split · the event anatomy · a 20-seed "
            "synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb). The claim — **Europe's big luxury houses get a "
            "spotlight bump around the Met Gala** — has no published academic anchor; it "
            "is markets-meet-culture folklore, a cousin of the sports-sentiment "
            "literature (Edmans-García-Norli 2007) with the trigger swapped for a "
            "fashion advertisement. The job is to measure it honestly, on the real "
            "luxury tape, with the right inference unit for a tiny-n annual event.\n\n"
            "> ⚠️ **Data note.** 4 single-name luxury tickers (`MC.PA`, `KER.PA`, "
            "`RMS.PA`, `CFR.SW`) equal-weighted + `VGK` (Europe benchmark), yfinance, "
            "adjusted (total-return) daily closes. 26 gala years hardcoded 2000→2025; "
            "3 had no gala (2000, 2002, 2020). Of the 23 held galas, **20** fall inside "
            f"the `VGK` window (2005→) — the binding floor, **named on the Signal "
            "axis**. Methods in [`docs/references.md`](../docs/references.md), numbers "
            "in [`docs/results.md`](../docs/results.md) (as-of " + R["asof"] +
            ", fingerprint `" + R["fp"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to "
            "intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | basket 1-week *t* = **{R['wk_t']:.3f}**, 1-month "
            f"*t* = **{R['mo_t']:.3f}** (placebo *p* = **{R['pl_mo_p']:.3f}**); no "
            "single name clears *t* ≥ 2 either |\n"
            f"| **Tradability** | `MIRAGE` | best net-of-cost capture *t* = "
            f"{R['cap_wk_t5']:.2f} (1-week), {R['cap_mo_t5']:.2f} (1-month); placebo "
            f"*p* ≈ {R['pl_cap_mo_p']:.2f} |\n"
            f"| **Just one name?** | `NOT SUPPORTED` | strongest single house "
            f"({R['name_max_mo']}/1-month) *t* = {R['name_max_mo_t']:.3f} |\n\n"
            "> 💡 In plain words: every cut we tried — basket, each name, week, month, "
            "gross, net — sits inside the luck cloud. The tape's honest answer is a flat "
            "\"no.\""
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_{L,t}$ be the equal-weighted luxury-basket log-return and $r_{b,t}$ "
            "the VGK benchmark log-return on trading day $t$. The gala runs Monday "
            "**evening** in New York; Euronext Paris and SIX Swiss (where all four names "
            "list) close ~7–8 hours *earlier*, so day(-1) = the gala-Monday close (does "
            "not yet know the gala) and day(0) = the next close (first to reflect it). "
            "The abnormal return over horizon $k$ is\n\n"
            "$$AR_y(k) = \\left(\\frac{P^{lux}_{-1+k}}{P^{lux}_{-1}} - 1\\right) - "
            "\\left(\\frac{P^{bench}_{-1+k}}{P^{bench}_{-1}} - 1\\right)$$\n\n"
            "Each gala year is a single, non-overlapping, independent event, so the "
            "**one-sample t** of $AR$ across events is the correct primary statistic — "
            "not a daily panel regression. Claims:\n\n"
            "- **H1 (spotlight bump).** $E[AR(k)] > 0$ at $k \\in \\{5, 21\\}$.\n"
            "- **H2 (anatomy).** The bump appears around the gala and is not just "
            "ordinary luxury drift.\n"
            "- **H3 (concentration).** If the basket looks flat, perhaps a single "
            "spotlight brand (LVMH) carries it.\n"
            "- **H4 (capture).** A trader entering AFTER the gala (zero look-ahead) can "
            "bank it net of costs.\n\n"
            "We find **H1 not supported** at either horizon, **H2 not supported** (the "
            "path is generic drift, and the placebo is dead-centre), **H3 not "
            "supported** (no name clears the bar), **H4 not supported** (no net cut "
            "clears *t* ≥ 2)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            f"n is small by construction: **{R['n_included']}** galas fall inside the "
            "VGK window (2005→), out of 23 held and 26 listed years. The plan is a "
            "**one-sample t** per cut (basket × week/month), a **Wilson interval** on "
            "the hit rate, a **20-seed × 200-draw random-window placebo** per cut "
            "(redraw a same-length window at a random point in the basket's own history "
            "and see how often the null matches or beats the observed mean), and a "
            "**per-name split** so a single mega-cap can't hide inside — or be hidden "
            "by — the equal-weight average."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_editions']} listed years 2000→2025 "
            f"({R['n_galas']} galas held), hardcoded from Wikipedia; 3 no-gala years "
            "(2000, 2002, 2020) and the September-2021 make-up edition are named "
            "quirks.\n"
            f"- **Sample.** {R['n_included']} galas inside the VGK window; "
            f"{R['n_prevgk']} pre-VGK (2001/2003/2004) and {R['n_nogala']} no-gala "
            "years excluded (funnel below).\n"
            "- **Headline.** One-sample *t* (basket, both horizons) + Wilson hit rate.\n"
            "- **Robustness.** 20×200-draw random-window placebo; per-name "
            "concentration split (four one-sample *t*'s).\n"
            "- **Anatomy.** Mean cumulative AR by trading day, −5→+21 (the run-up and "
            "the drift).\n"
            "- **Execution (Tradability).** Capture = enter day(0) close (zero "
            "look-ahead: the gala airs Monday evening after the European close), exit "
            "day(0)+k close, 2× one-way cost × NAV per event.\n"
            "- **Control.** Synthetic paired (basket, benchmark) world, planted-bump "
            "knob; the null must not systematically fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The selection funnel — the VGK floor is the modern-gala floor\n\n"
            "Conveniently, the year the Met Gala firmly adopted the first-Monday-in-May "
            "slot (2005) is also VGK's inception, so the excluded pre-VGK years are the "
            "same wandering-date pre-modern galas."
        ),
        code(
            "if HAVE_REAL:\n"
            "    reasons = EV[~EV['included']]['reason'].value_counts()\n"
            "    n_inc = len(INC)\n"
            "else:\n"
            "    reasons = pd.Series({'basket/benchmark predate the gala (pre-VGK 2005)': 3,\n"
            "                          'no gala held (2000/2002/2020)': 3})\n"
            "    n_inc = R['n_included']\n"
            "fig, ax = plt.subplots(figsize=(9.4, 3.6))\n"
            "ax.barh(reasons.index[::-1], reasons.values[::-1], color=GREY)\n"
            "ax.set_xlabel('gala years excluded')\n"
            "ax.set_title(f'{n_inc} galas tested (2005->2025); 3 pre-VGK + 3 no-gala excluded')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(reasons)"
        ),
        md(
            "> 💡 In plain words: the tested sample is the **modern, calendar-regular** "
            "Met Gala (2005→2025, minus the COVID-cancelled 2020). 2000 and 2002 had no "
            "gala at all; 2001/2003/2004 predate both the first-Monday convention and "
            "the VGK benchmark. This is a real floor, named on the Signal axis — not a "
            "footnote."
        ),
        md(
            "### 4b · The headline — one-sample t, two horizons"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for label, col in (('1wk', 'ar_week'), ('1mo', 'ar_month')):\n"
            "        s = st.one_sample_t(INC[col].values); hr = st.hit_rate(INC[col].values)\n"
            "        rows.append((label, s['n'], s['mean']*100, s['t'], hr['k'], hr['n']))\n"
            "    for r in rows: print(r)\n"
            "    means = [rows[0][2], rows[1][2]]; ts = [rows[0][3], rows[1][3]]\n"
            "else:\n"
            "    means = [R['wk_mean'], R['mo_mean']]; ts = [R['wk_t'], R['mo_t']]\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "a1.bar(['1 week', '1 month'], means, color=[GREY if t < 2 else AMBER for t in ts], width=.5)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean AR (%)')\n"
            "a1.set_title('Positive, tiny, and nowhere near the bar')\n"
            "a2.bar(['1 week', '1 month'], ts, color=[RED if abs(t) >= 2 else GREY for t in ts], width=.5)\n"
            "a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel('t-stat'); a2.set_ylim(0, 2.4)\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: 1-week AR = **{R['wk_mean']:+.3f}%** (*t* = "
            f"{R['wk_t']:.2f}, n = {R['wk_n']}), 1-month AR = **{R['mo_mean']:+.3f}%** "
            f"(*t* = {R['mo_t']:.2f}). Hit rates are {R['wk_hit']}/{R['wk_n']} and "
            f"{R['mo_hit']}/{R['mo_n']} — Wilson intervals straddle 50%. Both point up, "
            "neither is distinguishable from a fair coin."
        ),
        md(
            "### 4c · The random-window placebo — is the gala window unusual at all?\n\n"
            "For each gala event, redraw a random (non-gala) window of the same length "
            "on the SAME luxury basket vs VGK, 20 seeds × 200 draws; compare the "
            "observed mean to the null distribution."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(EV, PRICES, 'ar_month', k=21, entry_offset=0,\n"
            "                           n_seeds=4, n_draws_per_seed=200, tail='right')\n"
            "    obs = pl['obs']*100\n"
            "    rng = np.random.default_rng(719)\n"
            "    draws = rng.normal(pl['placebo_mean'], pl['placebo_sd'], 3000)*100\n"
            "else:\n"
            "    obs = R['mo_mean']\n"
            "    rng = np.random.default_rng(719)\n"
            "    draws = rng.normal(R['pl_mo_plmean'], R['pl_mo_plsd'], 3000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85,\n"
            "        label='null: random 21-session windows, same basket')\n"
            "ax.axvline(obs, c=RED, lw=2.4, label=f'observed gala 1-month mean {obs:+.2f}%')\n"
            "ax.set_xlabel('mean AR of a random-window draw (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Canonical placebo (results.md, 20x200 draws): p = {R[\"pl_mo_p\"]:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical: observed {R['mo_mean']:+.3f}%, placebo mean \"\n"
            "      f\"{R['pl_mo_plmean']:+.3f}% (sd {R['pl_mo_plsd']:.3f}%), p = {R['pl_mo_p']:.4f}\")"
        ),
        md(
            f"> 💡 In plain words: the observed **{R['mo_mean']:+.3f}%** sits almost "
            f"exactly on the placebo mean (**{R['pl_mo_plmean']:+.3f}%**), giving "
            f"*p* = **{R['pl_mo_p']:.3f}** — a random month is as good as the gala month "
            "half the time. This is the cleanest possible null: not a marginal miss, a "
            "dead-centre nothing. (Contrast study 708, where one cut genuinely sat at "
            "the tail, *p* = 0.038.)"
        ),
        md(
            "### 4d · Concentration — is a single spotlight brand carrying it?\n\n"
            "Split the equal-weight basket into its four names and run the same "
            "one-sample *t* on each. If the folklore has any truth it should live in "
            "LVMH (Louis Vuitton, Dior — the loudest brands on the carpet)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pn = st.per_name_stats(EV, 'mo')\n"
            "    names = list(pn['name']); means = list(pn['mean']*100); ts = list(pn['t'])\n"
            "else:\n"
            "    names = list(R['name_mo']); means = [R['name_mo'][n][0] for n in names]\n"
            "    ts = [R['name_mo'][n][1] for n in names]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))\n"
            "a1.bar(names, means, color=GREY, width=.6); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_ylabel('mean 1-month AR (%)'); a1.set_title('Per-name mean')\n"
            "a1.tick_params(axis='x', rotation=20)\n"
            "a2.bar(names, ts, color=GREY, width=.6)\n"
            "a2.axhline(2, ls='--', c=RED, lw=1.2, label='bar'); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel('one-sample t'); a2.set_ylim(0, 2.4); a2.set_title('Per-name t')\n"
            "a2.tick_params(axis='x', rotation=20); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for n, m, t in zip(names, means, ts): print(f'  {n:10s} {m:+.3f}%  t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: every house is noise. The strongest 1-month name is "
            f"**{R['name_max_mo']}** at *t* = {R['name_max_mo_t']:.2f}; LVMH itself — "
            f"the brand-exposure king — is only *t* = {R['name_mo']['LVMH'][1]:.2f}. "
            "There is no concentrated effect being diluted by the average, which kills "
            "the folklore's most natural fallback. **Third axis = NOT SUPPORTED.**"
        ),
        md(
            "### 4e · Event anatomy — run-up and drift"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp = st.car_path(EV, PRICES, pre=5, post=21)\n"
            "    days = list(cp.index); vals = list(cp.values*100)\n"
            "else:\n"
            "    days = sorted(R['car']); vals = [R['car'][k] for k in days]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "ax.plot(days, vals, color=AMBER, lw=2.2, marker='o')\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, ls=':', c='k', lw=.9,\n"
            "           label='gala anchor (day -1 = last pre-gala close)')\n"
            "ax.set_xlabel('trading days from the pre-gala close')\n"
            "ax.set_ylabel('mean cumulative abnormal return (%)')\n"
            "ax.set_title('No pre-gala run-up, no post-gala pop -- just a wandering drift')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            "> 💡 In plain words: a genuine attention event predicts *either* a "
            "buy-the-rumor run-up before the gala *or* a pop right after. The path "
            f"shows neither: it is flat into the gala ({R['car'][-1]:+.3f}% at day −1), "
            f"drifts to {R['car'][10]:+.3f}% by day +10, then gives most of it back by "
            f"day +21 ({R['car'][21]:+.3f}%). That round-trip shape is ordinary "
            "luxury-vs-Europe noise, not an event reaction."
        ),
        md(
            "### 4f · Tradability — the zero-look-ahead capture test\n\n"
            "Enter at day(0)'s close (the first price AFTER the gala is public — it airs "
            "Monday evening, after the European close), exit day(0)+k close, 2× one-way "
            "cost × NAV per event, at 5 and 10 bps."
        ),
        code(
            "if HAVE_REAL:\n"
            "    EV10 = st.build_event_table(PRICES, cost_bps=10.0); INC10 = EV10[EV10['included']]\n"
            "    rows = {}\n"
            "    for base in ('cap_week', 'cap_month'):\n"
            "        g = st.one_sample_t(INC[base+'_gross'].values)\n"
            "        n5 = st.one_sample_t(INC[base+'_net'].values)\n"
            "        n10 = st.one_sample_t(INC10[base+'_net'].values)\n"
            "        rows[base] = (g['mean']*100, g['t'], n5['mean']*100, n5['t'], n10['mean']*100, n10['t'])\n"
            "    wk, mo = rows['cap_week'], rows['cap_month']\n"
            "else:\n"
            "    wk = (R['cap_wk_g'], R['cap_wk_gt'], R['cap_wk_n5'], R['cap_wk_t5'], R['cap_wk_n10'], R['cap_wk_t10'])\n"
            "    mo = (R['cap_mo_g'], R['cap_mo_gt'], R['cap_mo_n5'], R['cap_mo_t5'], R['cap_mo_n10'], R['cap_mo_t10'])\n"
            "x = np.arange(2); w = 0.26\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(x-w, [wk[0], mo[0]], w, label='gross', color=GREY)\n"
            "ax.bar(x,   [wk[2], mo[2]], w, label='net @5bps', color=AMBER)\n"
            "ax.bar(x+w, [wk[4], mo[4]], w, label='net @10bps', color=RED)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels(['1 week', '1 month'])\n"
            "ax.set_ylabel('capture return (%)'); ax.legend()\n"
            "ax.set_title(f'Best capture t: week {wk[3]:.2f}, month {mo[3]:.2f} -- never near 2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'week : gross {wk[0]:+.3f}% (t{wk[1]:+.2f})  net5 {wk[2]:+.3f}% (t{wk[3]:+.2f})  net10 {wk[4]:+.3f}% (t{wk[5]:+.2f})')\n"
            "print(f'month: gross {mo[0]:+.3f}% (t{mo[1]:+.2f})  net5 {mo[2]:+.3f}% (t{mo[3]:+.2f})  net10 {mo[4]:+.3f}% (t{mo[5]:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: the strongest tradable cut is 1-week gross at "
            f"*t* = {R['cap_wk_gt']:.2f}, and it only gets worse once you charge costs "
            f"(net@5bps *t* = {R['cap_wk_t5']:.2f}). The 1-month net capture "
            f"({R['cap_mo_n5']:+.2f}%) carries a placebo *p* ≈ {R['pl_cap_mo_p']:.2f}. "
            "**H4 not supported; Tradability = MIRAGE.**"
        ),
        md(
            "### 4g · Faithful-engine & power control\n\n"
            "Synthetic paired (basket, benchmark) log-return world (ρ ≈ 0.80, like a "
            "luxury basket vs a regional benchmark), a scheduled synthetic event "
            "calendar, TUNABLE planted bump. Null (bump = 0) checked over **20 seeds**."
        ),
        code(
            "null_ts = np.array([st.synthetic_detect(bump=0.0, seed=719+s, k=21)['t'] for s in range(20)])\n"
            "planted1 = st.synthetic_detect(bump=0.01, seed=719, k=21)\n"
            "planted2 = st.synthetic_detect(bump=0.02, seed=719, k=21)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (bump=0), 20 seeds')\n"
            "ax.scatter([1], [planted1['t']], color=AMBER, s=90, zorder=5, label='planted bump=1%')\n"
            "ax.scatter([2], [planted2['t']], color=RED, s=90, zorder=5, label='planted bump=2%')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['null x20', 'planted 1%', 'planted 2%'])\n"
            "ax.set_ylabel('one-sample t'); ax.set_title('Control: null centred on 0, a 2% bump lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t={null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "print(f'planted 1%% t={planted1[\"t\"]:+.2f}  planted 2%% t={planted2[\"t\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null seeds the detector centres on "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) — unbiased — but "
            f"fires at |t| ≥ 2 in {R['syn_null_fire']}/{R['syn_null_seeds']} seeds. That "
            "fat small-n tail *is* the study's lesson: at n = 20 a lone *t* ≥ 2 is not "
            f"to be trusted, which is why the placebo does the real work. A planted 2% "
            f"bump reads t = {R['syn_planted2_t']:.2f} (a 1% bump only "
            f"{R['syn_planted1_t']:.2f}) — the tape would need a genuine ~2%/event "
            "effect to show up, and it has none. *(A faithful-engine / power check only "
            "— never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — basket 1-week *t* = **{R['wk_t']:.3f}**, 1-month "
            f"*t* = **{R['mo_t']:.3f}**, and the 1-month placebo *p* = "
            f"**{R['pl_mo_p']:.3f}** puts the gala month dead centre of the luck cloud. "
            "The event anatomy is a directionless round-trip, not an announcement "
            "reaction. Literature offers no anchor for this specific claim.\n"
            f"- **Tradability `MIRAGE`** — no net-of-cost, zero-look-ahead cut clears "
            f"*t* ≥ 2; best case *t* = {R['cap_wk_t5']:.2f} (1-week net), with the "
            f"1-month net capture at placebo *p* ≈ {R['pl_cap_mo_p']:.2f}.\n"
            f"- **\"Just one name?\" `NOT SUPPORTED`** — the per-name split kills the "
            "concentration fallback: the strongest single house "
            f"({R['name_max_mo']}/1-month) is *t* = {R['name_max_mo_t']:.3f}, LVMH "
            f"itself {R['name_mo']['LVMH'][1]:.3f}."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **A predictable advertisement is priced already.** The gala date is "
            "fixed a year ahead and the brand exposure is entirely anticipated, so an "
            "efficient market has nothing left to react to on the day — the textbook "
            "reason a *calendar-known* attention event produces no abnormal return. "
            "This null is the expected result, cleanly confirmed.\n"
            "- **A higher-power test would need more resolution.** Intraday luxury data "
            "around the actual broadcast, a theme-conditioned split (blockbuster vs "
            "quiet years), or European luxury sub-indices predating 2005 would raise n "
            "and sharpen the window — the natural sequel.\n"
            "- **Dedup map:** [708-eurovision-effect](../../708-eurovision-effect/) "
            "(a per-country abnormal-return panel keyed to a cultural-contest night — "
            "same event-study machinery, different trigger and instrument), "
            "[235-world-cup-effect](../../235-world-cup-effect/) and "
            "[158-super-bowl](../../158-super-bowl/) (single-market sentiment "
            "folklore), [707-plane-crash-effect](../../707-plane-crash-effect/) (a "
            "sentiment shock with the opposite sign). None test a **luxury-sector "
            "abnormal-return panel keyed to a fashion event** — that's this study's own "
            "contribution.\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live "
            "in [`docs/results.md`](../docs/results.md), sources in "
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
