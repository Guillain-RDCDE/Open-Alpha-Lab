"""Generate the two narrative notebooks for Study 741 (Cicada-Brood).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY
total-return tape under ../_cache/ and otherwise quote the frozen headline numbers in
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SPY total-return
# close 1993-01-29 -> 2026-06-30; 30 hardcoded periodical-cicada emergences, 24 distinct
# emergence years, 1996 -> 2025).
R = dict(
    n_broods=30, cal_lo=1996, cal_hi=2025, n_events=24, span=30, brood_years_in_span=24,
    pool_years=33, spy_rows=8411, K=42, KS=21,
    # headline cuts: (n, raw%, abn%, abn_t, hit, hit_n, hit%, wilson_lo, wilson_hi)
    cut_all=(24, 2.400, 0.385, 0.37, 18, 24, 75, 55, 88),
    cut_17=(23, 2.626, 0.607, 0.57, 18, 23, 78, 58, 90),
    cut_fam=(5, 3.755, 1.706, 1.19, 5, 5, 100, 57, 100),
    baseline_bps=180.1, cicada_mean_bps=240.0, cicada_minus_base_bps=60.0,
    placebo_obs_pct=2.400, placebo_mean_pct=1.802, placebo_sd_pct=0.606,
    placebo_p=0.165, placebo_draws=20000,
    short_abn_pct=0.142, short_abn_t=0.20,
    welch_cicada_pct=2.400, welch_non_pct=0.201, welch_non_n=9, welch_t=0.874,
    # anatomy offset -> (CAR%, t)
    anat={0: (0.304, 1.52), 10: (0.517, 0.92), 21: (0.142, 0.20),
          31: (0.444, 0.45), 42: (0.385, 0.37)},
    # timer leg -> (mean_bps, win%, t_vs0, excess_bps, excess_t)
    timer_gross=(240.0, 75, 2.25, 60.0, 0.56),
    timer_net5=(230.0, 71, 2.15, 50.0, 0.47),
    timer_net10=(220.0, 71, 2.06, 40.0, 0.37),
    syn_null_mean=0.15, syn_null_sd=0.93, syn_null_fire=1,
    syn_planted_t=4.46, syn_planted_pct=4.79,
    fp_spy="a11dcd0a0a72",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Cicada_indicator%3F: Busted](https://img.shields.io/badge/Cicada_indicator%3F-Busted-8b949e?style=flat-square)\n\n"
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

from cicada_brood import data, strategy as st

BROODS = data.brood_table()
EV_YEARS = data.brood_years()
HAVE_REAL = data.have_real()
if HAVE_REAL:
    SPY = data.load_real()
    RET = st.daily_returns(SPY)
    AR = st.abnormal_returns(RET)
    POOL = data.all_years(SPY)
else:
    SPY = RET = AR = POOL = None
print("real cache present:", HAVE_REAL, "| brood rows:", len(BROODS),
      "| distinct emergence years:", len(EV_YEARS))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do the cicadas ring the bell for stocks? 🦗📈\n"
            "### The \"17-year cicada bull\" — a fixed, famous calendar that means "
            "absolutely nothing\n\n"
            + BADGES +
            "Every 13 or 17 years, billions of periodical cicadas claw out of the ground "
            "across the eastern US, sing deafeningly for six weeks, and blanket the news. "
            "It's one of the few genuinely *fixed* events in nature — the year of the next "
            "emergence has been known, to the year, since the last one. So it makes an "
            "irresistible bit of market folklore: two of the biggest emergences (2004 and "
            "2021) landed in rising markets, and *voilà*, a \"cicada indicator.\"\n\n"
            "This study is a **deliberate joke with a serious payoff.** There's no reason "
            "on earth cicadas should move the S&P 500 — so we build the *strongest* "
            "version of the claim, run the exact same rigorous event study we'd run on a "
            "real effect, and watch it come back empty. It's the cleanest way to show what "
            "a genuine null looks like — and how easily a coincidence dresses up as a "
            "signal.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the "
            "beta-vs-alpha timer? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> 🔬 **For the quants:** the event unit is the emergence *year* (independent, "
            "non-overlapping), so the primary is a one-sample *t* across years, not a "
            "daily panel — and the tradability grade is the **excess over the every-spring "
            "baseline** (alpha), never the raw return vs zero (which is just equity beta)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do cicada-emergence springs beat ordinary springs for the S&P 500? | "
            f"**No.** The average cicada spring runs a statistically-nothing "
            f"**+{R['cut_all'][2]:.2f}%** *abnormal* return (*t* = {R['cut_all'][3]:+.2f}); "
            f"a random calendar of springs beats it about **1 time in 6** "
            f"(placebo *p* = {R['placebo_p']:.2f}). |\n"
            f"| But 75% of cicada springs were up! | Sure — and so are most springs. "
            "Stocks drift up; a coin weighted to heads isn't a cicada effect. |\n"
            "| The 5 *famous* broods were 5-for-5 up — isn't that something? | That's "
            "**selection**, the exact trick that mints folklore: pick the 5 emergences "
            "everyone remembers and you've picked 5 up markets after the fact. |\n"
            f"| Could you trade it? | **No.** You could schedule the trade decades ahead "
            "(the one signal with zero look-ahead) and *still* earn only the market's "
            f"ordinary drift — the edge over just-hold-every-spring is "
            f"**+{R['timer_gross'][3]:.0f} bps at *t* = {R['timer_gross'][4]:.2f}**. |\n\n"
            "> The whole point: a fixed, famous, foreseeable natural calendar produces "
            "exactly the noise you'd expect it to. This is what an honest *nothing* looks "
            "like."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Periodical cicadas emerge on a rigid 13- or 17-year clock — you know the "
            "year decades in advance. The big emergences coincided with strong markets "
            "(Brood X in 2004 and 2021, both up years). A savvy almanac-reader could "
            "position for the 'cicada bull' well ahead of time.\"*\n\n"
            "This is folklore in the grand tradition of the **Super Bowl indicator**, the "
            "**January barometer** and **\"Sell in May and Go Away\"** — impressive-looking "
            "calendar coincidences with no mechanism attached. We're not strawmanning it; "
            "we're giving it its best shot: a real, precisely-dated, nationally-famous "
            "calendar, and the full event-study treatment."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a calendar this arbitrary *did* predict returns, it would be a gift — a "
            "signal you could set and forget, no news to watch, no look-ahead to worry "
            "about. It would also be a small scandal for market efficiency: prices moving "
            "on *bugs*. So the stakes are really about **method**: this is the desk's "
            "control experiment. If our apparatus can be talked into seeing a signal in "
            "cicadas, it can be talked into anything — and if it correctly says "
            "\"nothing here,\" that's the reassurance that makes the *real* studies "
            "trustworthy."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** **{R['n_broods']}** mapped periodical-cicada brood "
            f"emergences {R['cal_lo']}–{R['cal_hi']}, from the University of Connecticut "
            f"brood chart — **{R['n_events']} distinct emergence years** (the unit we "
            "test).\n"
            "- **The window.** Cicadas surface in early-mid May and swarm through late "
            "June, so each event is the S&P 500's **May–June** window (first session after "
            "May 1, held ~2 months).\n"
            f"- **The comparison.** That window's *abnormal* return (above the market's "
            "own average drift) — and the same window drawn from **random years**, 20,000 "
            "times, to see if cicada springs stand out.\n"
            "- **The base-rate check.** How many of the last 30 years even *had* a brood? "
            f"(Spoiler: **{R['brood_years_in_span']} of {R['span']}**.)\n"
            "- **The trade check.** Hold the S&P for the cicada spring only — scheduled "
            "years ahead — pay costs, and compare to just holding *every* spring."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Average cicada-spring S&P return vs. what a random "
            "spring looks like (same May–June window, so this isn't just the seasonal "
            "\"Sell in May\" pattern)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    evt = st.build_event_table(SPY, AR, EV_YEARS, k=R['K'])\n"
            "    obs = float(evt['raw_ret'].mean()) * 100\n"
            "    base = st.unconditional_spring_baseline(SPY, POOL, k=R['K']) / 100\n"
            "else:\n"
            "    obs, base = R['placebo_obs_pct'], R['baseline_bps'] / 100\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "ax.bar(['cicada springs\\n(n=24)', 'every spring\\n(all 33 yrs)',\n"
            "        'random 24-spring\\ndraw (placebo)'],\n"
            "       [obs, base, R['placebo_mean_pct']], color=[AMBER, GREY, GREY], width=.6)\n"
            "for i, v in enumerate([obs, base, R['placebo_mean_pct']]):\n"
            "    ax.annotate(f'{v:+.2f}%', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean May-June total return (%)')\n"
            "ax.set_title('Cicada springs sit right on top of ordinary springs')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'cicada {obs:+.2f}%  vs  every-spring baseline {base:+.2f}%  '\n"
            "      f\"(placebo p = {R['placebo_p']:.3f} over {R['placebo_draws']:,} draws)\")"
        ),
        md(
            f"The cicada-spring bar (**+{R['placebo_obs_pct']:.2f}%**) is a whisker above "
            f"the every-spring bar (**+{R['baseline_bps']/100:.2f}%**) — about **+"
            f"{R['cicada_minus_base_bps']:.0f} bps** — but a random draw of 24 springs "
            f"produces a gap that big **{R['placebo_p']*100:.0f}%** of the time. That's not "
            "a signal; that's what noise looks like.\n\n"
            "**Next, the trick that fools everyone: the up-rate.** 75% of cicada springs "
            "were up. Sounds like a lot — until you remember stocks mostly go up, and you "
            "watch what happens when you cherry-pick the *famous* broods."
        ),
        code(
            "labels = ['all notable\\nbroods', '17-year\\nbroods only', 'famous\\nmarquee only']\n"
            "if HAVE_REAL:\n"
            "    cuts = []\n"
            "    for yrs in (EV_YEARS, data.brood_years(cycle=17),\n"
            "                data.brood_years(famous_only=True)):\n"
            "        e = st.build_event_table(SPY, AR, yrs, k=R['K'])\n"
            "        hr = st.hit_rate(e['raw_ret'].to_numpy())\n"
            "        cuts.append((hr['rate'] * 100, hr['n']))\n"
            "else:\n"
            "    cuts = [(R['cut_all'][6], R['cut_all'][0]),\n"
            "            (R['cut_17'][6], R['cut_17'][0]),\n"
            "            (R['cut_fam'][6], R['cut_fam'][0])]\n"
            "rates = [c[0] for c in cuts]; ns = [c[1] for c in cuts]\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.4))\n"
            "cols = [GREY, GREY, RED]\n"
            "ax.bar(labels, rates, color=cols, width=.6)\n"
            "for i, (v, n) in enumerate(zip(rates, ns)):\n"
            "    ax.annotate(f'{v:.0f}% up\\n(n={n})', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(50, ls='--', c='k', lw=1, label='a coin flip')\n"
            "ax.set_ylim(0, 112); ax.set_ylabel('share of springs that were up (%)')\n"
            "ax.set_title('The fewer, more famous broods you keep, the more \\'perfect\\' it looks')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('up-rates:', dict(zip(['all','17yr','famous'], [round(r) for r in rates])))"
        ),
        md(
            f"See the illusion form in real time. All 24 broods: **{R['cut_all'][6]:.0f}% "
            f"up**. Keep only the marquee ones everyone remembers (2004, 2007, 2013, 2021, "
            f"2024): **{R['cut_fam'][6]:.0f}% up — 5 for 5**. That looks like destiny. It's "
            "**selection**: those five happened to fall in up springs, and picking them "
            "*after* the fact is the entire recipe for a market myth. With only 5 events, "
            "even a coin lands 5 heads about 1 time in 30.\n\n"
            "**Finally, the trade.** Cicada years are the one signal you could schedule "
            "decades ahead — zero look-ahead. So does the trade pay?"
        ),
        code(
            "legs = ['gross', 'net (5 bps)', 'net (10 bps)']\n"
            "if HAVE_REAL:\n"
            "    means, excess = [], []\n"
            "    for cost in (0.0, 5.0, 10.0):\n"
            "        lg = st.spring_timer(SPY, EV_YEARS, k=R['K'], cost_bps=cost)\n"
            "        col = 'ret_gross' if cost == 0 else 'ret_net'\n"
            "        means.append(st.summarize_timer(lg, col=col)['mean_bps'])\n"
            "        excess.append(st.excess_over_baseline(SPY, EV_YEARS, POOL,\n"
            "                       k=R['K'], cost_bps=cost)['excess_bps'])\n"
            "    base_bps = st.unconditional_spring_baseline(SPY, POOL, k=R['K'])\n"
            "else:\n"
            "    means = [R['timer_gross'][0], R['timer_net5'][0], R['timer_net10'][0]]\n"
            "    excess = [R['timer_gross'][3], R['timer_net5'][3], R['timer_net10'][3]]\n"
            "    base_bps = R['baseline_bps']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))\n"
            "a1.bar(legs, means, color=AMBER, width=.6)\n"
            "a1.axhline(base_bps, ls='--', c=GREY, lw=1.5,\n"
            "           label=f'just hold EVERY spring ({base_bps:+.0f} bps)')\n"
            "for i, v in enumerate(means):\n"
            "    a1.annotate(f'{v:+.0f}', (i, v), ha='center', va='bottom')\n"
            "a1.set_ylabel('mean cicada-spring return (bps)')\n"
            "a1.set_title('Looks like a fat positive return...'); a1.legend()\n"
            "a2.bar(legs, excess, color=RED, width=.6)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "for i, v in enumerate(excess):\n"
            "    a2.annotate(f'{v:+.0f}', (i, v), ha='center', va='bottom')\n"
            "a2.set_ylabel('EXCESS over every-spring baseline (bps)')\n"
            "a2.set_title('...but the edge over just-holding is ~nothing')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('cicada-spring mean (bps):', [round(v) for v in means])\n"
            "print('excess over baseline (bps):', [round(v) for v in excess])"
        ),
        md(
            f"On the left, the cicada trade earns a chunky-looking **+"
            f"{R['timer_gross'][0]:.0f} bps** — but that's just what holding *any* two "
            "months of stocks earns on average. The honest question is the panel on the "
            f"right: how much does cicada-timing beat *just holding every spring*? "
            f"**+{R['timer_gross'][3]:.0f} bps gross**, shrinking with costs — a rounding "
            "error, not an edge. You waited 17 years for the bugs and the market handed "
            "you the same beta it hands everyone."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Cicada-spring abnormal return **+{R['cut_all'][2]:.2f}%** "
            f"(*t* = {R['cut_all'][3]:+.2f}), placebo *p* = {R['placebo_p']:.2f}. The 75% "
            "up-rate and the 5/5 famous-brood streak are drift and selection, not a "
            "cicada effect.\n"
            f"- **Tradability — Mirage.** A perfectly-foreseeable calendar trade whose edge "
            f"over just-holding-every-spring is **+{R['timer_gross'][3]:.0f} bps "
            f"(*t* = {R['timer_gross'][4]:.2f})** — pure beta.\n"
            f"- **\"Cicada indicator?\" — Busted.** A brood emerges in "
            f"**{R['brood_years_in_span']} of {R['span']}** recent years; \"cicada years\" "
            "are nearly the whole calendar. Exactly the nothing we built the study to find."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **This is the desk's control experiment.** It exists to be *negative* — to "
            "show the same apparatus that grades real anomalies returns an honest zero on "
            "a pattern with no mechanism. If you ever see this rig hand a cicada calendar a "
            "green stamp, something is broken.\n"
            "- **The look-elsewhere lesson generalises.** Any fixed, evocative calendar "
            "(full moons, election years, leap years, World Cups) can be mined for a "
            "5-for-5 streak in a bull market. The fix isn't a cleverer test on the same "
            "data — it's out-of-sample data and a pre-registered window.\n"
            "- **Sibling studies:** [707-plane-crash-effect](../../707-plane-crash-effect/) "
            "and [708-eurovision-effect](../../708-eurovision-effect/) — same machinery "
            "(hardcoded calendar, event study, placebo, costed timer), real sentiment "
            "triggers instead of a pure coincidence. Cicada-Brood is their deliberately "
            "absurd cousin.\n\n"
            "*Think a different fixed calendar hides a real edge? Pre-register the window, "
            "bring out-of-sample years, grade it on excess-over-baseline after costs — "
            "then we'll talk.*"
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
            "# The Cicada-Brood effect — a quantitative teardown 🔬\n"
            "### A one-sample-*t* battery across 24 independent emergence years · a "
            "season-controlled 20-seed random-year placebo · the May–June event anatomy · "
            "a beta-vs-alpha timer (excess over the every-spring baseline) · a 20-seed "
            "synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb). The claim — a fixed-calendar \"cicada indicator\" "
            "in the family of the Super Bowl predictor — has **no proposed mechanism**; "
            "it is on the desk precisely as a spurious-pattern control. The job here is to "
            "run the full apparatus honestly and confirm it returns nothing — including "
            "resisting the two traps (a >50% up-rate and a cherry-picked marquee subset) "
            "that would fool a careless reader.\n\n"
            "> ⚠️ **Data note.** SPY **total-return** close (`auto_adjust=True`), yfinance, "
            f"1993-01-29 → 2026-06-30 ({R['spy_rows']:,} rows, cached); **{R['n_broods']} "
            f"hardcoded periodical-cicada emergences → {R['n_events']} distinct emergence "
            "years** (UConn/Cooley brood chart). SPY is a real tradable instrument — **no "
            "proxy**. No survivorship (whole-market index; fixed astronomical schedule). "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp_spy"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | cicada-spring abnormal CAR **+{R['cut_all'][2]:.2f}%**, "
            f"one-sample **t = {R['cut_all'][3]:+.2f}** (n={R['cut_all'][0]}), random-year "
            f"placebo **p = {R['placebo_p']:.3f}**, Welch vs non-cicada springs "
            f"t = {R['welch_t']:+.2f} |\n"
            f"| **Tradability** | `MIRAGE` | timer excess over every-spring baseline "
            f"**+{R['timer_gross'][3]:.0f} bps gross, t = {R['timer_gross'][4]:.2f}**; the "
            f"+{R['timer_gross'][2]:.2f} t-vs-zero is equity beta, not alpha |\n"
            f"| **Cicada indicator?** | `BUSTED` | a brood emerges in "
            f"**{R['brood_years_in_span']}/{R['span']}** years — the base rate alone kills "
            "it |\n\n"
            "> 💡 In plain words: no mechanism, no signal, no edge — and the two things "
            "that *look* like evidence (a 75% up-rate, a 5/5 famous-brood streak) are drift "
            "and selection."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be SPY's daily total return and $a_t = r_t - \\bar{r}$ its abnormal "
            "return under a constant-mean market model (Brown & Warner 1985). For emergence "
            "year $i$, anchor $\\tau_i$ = first session on/after May 1, window length "
            f"$K = {R['K']}$ sessions. The claims:\n\n"
            "- **H₁ (cicada springs are special).** $E\\big[\\sum_{k=0}^{K} a_{\\tau_i+k}\\big] "
            "> 0$ — a positive *abnormal* spring CAR, systematically across emergence "
            "years.\n"
            "- **H₂ (beats a random spring).** The mean cicada-spring window return exceeds "
            "what a random calendar of the same number of springs produces.\n"
            "- **H₃ (beats a quiet spring).** Cicada springs out-return non-cicada springs "
            "(Welch, same window).\n"
            "- **H₄ (tradable).** A long-SPY cicada-spring overlay beats the unconditional "
            "every-spring baseline net of costs — *excess*, not raw.\n\n"
            f"We find **H₁ not supported** (t = {R['cut_all'][3]:+.2f}), **H₂ not "
            f"supported** (placebo p = {R['placebo_p']:.3f}), **H₃ not supported** "
            f"(Welch t = {R['welch_t']:+.2f}), **H₄ not supported** (excess "
            f"+{R['timer_gross'][3]:.0f} bps, t = {R['timer_gross'][4]:.2f})."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Emergence years are **independent, non-overlapping** (one SPY spring each, "
            "years apart), so the planned primary is a **one-sample *t*** across the "
            f"{R['n_events']} per-year abnormal CARs — **not** a daily-panel regression, "
            "which would treat ~40 autocorrelated daily returns per event as independent "
            "and wildly overstate significance. Two traps are pre-empted by design: (i) the "
            "**up-rate carries a Wilson interval** and is read against the equity base rate, "
            "never against 50%; (ii) the **random-year placebo uses the same May–June "
            "window**, so the seasonal (\"Sell in May\") baseline cancels and the test "
            "isolates *cicada* springs from *random* springs. Tradability is graded on "
            "**excess over the every-spring baseline** (alpha), because a two-month equity "
            "window is positive on average for reasons unrelated to cicadas (beta)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_broods']} mapped emergences {R['cal_lo']}–{R['cal_hi']} "
            f"→ {R['n_events']} distinct years (UConn/Cooley chart), hardcoded.\n"
            f"- **Tape.** SPY total-return close, 1993 → 2026-06-30 (as-of, last complete "
            f"month), {R['spy_rows']:,} rows, fingerprint `{R['fp_spy']}`.\n"
            f"- **Window.** Anchor = first session on/after May 1; K = {R['K']} sessions "
            f"(~2 months) headline, {R['KS']} (~1 month) cross-check.\n"
            "- **Headline.** One-sample *t* on the abnormal CAR + Wilson up-rate + three "
            "cuts (all / 17-year-only / famous-marquee) to expose selection.\n"
            f"- **Placebo.** {R['placebo_draws']:,} random-year draws (20 seeds × 1,000), "
            "same window.\n"
            "- **Contrast.** Welch cicada vs non-cicada springs.\n"
            "- **Execution.** Zero look-ahead by construction (year known since the last "
            "emergence); enter last April close, exit ~end June; 2 × one-way cost × NAV; "
            "long-only; graded on excess over baseline.\n"
            "- **Control.** Synthetic random-walk tape, planted spring bump; null must not "
            "systematically fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline and its season-controlled placebo\n\n"
            "One-sample *t* on the per-year abnormal CAR, and the random-year null. In the "
            "notebook we run a lighter placebo (4 seeds × 500) and quote the canonical "
            f"{R['placebo_draws']:,}-draw *p* from `results.md`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    evt = st.build_event_table(SPY, AR, EV_YEARS, k=R['K'])\n"
            "    s = st.one_sample_t(evt['abn_car'].to_numpy())\n"
            "    obs_raw = float(evt['raw_ret'].mean())\n"
            "    pl = st.random_year_placebo(SPY, POOL, n_events=len(EV_YEARS), k=R['K'],\n"
            "                                 n_seeds=4, n_draws_per_seed=500, base_seed=741)\n"
            "    draws = pl['means']\n"
            "    print(f\"abnormal CAR {s['mean']*100:+.3f}%  one-sample t = {s['t']:+.3f}  \"\n"
            "          f\"(n={s['n']})\")\n"
            "else:\n"
            "    obs_raw = R['placebo_obs_pct'] / 100\n"
            "    rng = np.random.default_rng(741)\n"
            "    draws = rng.normal(R['placebo_mean_pct']/100, R['placebo_sd_pct']/100, 2000)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.hist(draws * 100, bins=45, color=GREY, alpha=.85,\n"
            "        label='null: random calendars of 24 springs (light in-notebook run)')\n"
            "ax.axvline(obs_raw * 100, c=RED, lw=2.5,\n"
            "           label=f'observed cicada-spring mean {obs_raw*100:+.2f}%')\n"
            "ax.axvline(R['baseline_pct'] if 'baseline_pct' in R else R['baseline_bps']/100,\n"
            "           c=GREEN, ls='--', lw=1.5, label=f\"every-spring baseline {R['baseline_bps']/100:+.2f}%\")\n"
            "ax.set_xlabel('mean May-June total return of a random 24-spring calendar (%)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Squarely inside the luck cloud: canonical p = {R['placebo_p']:.3f}\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical placebo (results.md): mean {R['placebo_mean_pct']:+.3f}%, \"\n"
            "      f\"sd {R['placebo_sd_pct']:.3f}%, right-tail p = {R['placebo_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: the observed **+{R['placebo_obs_pct']:.2f}%** sits just "
            f"inside the right shoulder of the null ({R['placebo_mean_pct']:+.2f} ± "
            f"{R['placebo_sd_pct']:.2f}%); **p = {R['placebo_p']:.3f}** — a random 24-spring "
            f"calendar beats it about 1 time in 6. With abnormal-CAR t = "
            f"**{R['cut_all'][3]:+.2f}** and the 1-month window at t = "
            f"**{R['short_abn_t']:+.2f}** (+{R['short_abn_pct']:.2f}%), H₁ and H₂ are dead."
        ),
        md(
            "### 4b · Event anatomy — the May–June CAR path\n\n"
            "Per-offset cumulative abnormal return from the May-1 anchor, each offset's own "
            "one-sample *t*. If cicadas mattered, the path would climb and a *t* would "
            "clear the bar somewhere. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp = st.car_path(SPY, AR, EV_YEARS, k=R['K'])\n"
            "    ks = list(cp.index); cars = list(cp['car'] * 100); ts = list(cp['t'])\n"
            "else:\n"
            "    ks = sorted(R['anat']); cars = [R['anat'][k][0] for k in ks]\n"
            "    ts = [R['anat'][k][1] for k in ks]\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.6, 6.4), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "a1.plot(ks, cars, color=AMBER, lw=2); a1.axhline(0, c='k', lw=.8)\n"
            "a1.fill_between(ks, 0, cars, color=AMBER, alpha=.15)\n"
            "a1.set_ylabel('mean CAR (%)')\n"
            "a1.set_title('Event anatomy: the cicada-spring CAR just wanders around zero')\n"
            "a2.plot(ks, ts, color=GREY, lw=1.5)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(-2, ls='--', c=RED, lw=1)\n"
            "a2.set_ylim(-3, 3); a2.set_ylabel('one-sample t')\n"
            "a2.set_xlabel('sessions after the May-1 anchor')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('CAR% by offset:', {k: round(c, 3) for k, c in zip(ks, cars) if k in (0,10,21,31,42)})\n"
            "print('max |t| along the path:', round(max(abs(t) for t in ts), 2))"
        ),
        md(
            f"> 💡 In plain words: the CAR never gets more than a fraction of a percent from "
            f"flat, and its *t* peaks at a forgettable **+{R['anat'][0][1]:.2f}** on the "
            "very first in-window session, then fades. No offset clears |*t*| ≥ 2. There is "
            "no build-up, no peak, no shape — just a two-month random walk."
        ),
        md(
            "### 4c · The two traps — up-rate base rate & marquee selection\n\n"
            "The results a careless believer would quote: the up-rate, and the "
            "famous-broods-only cut. Both are shown *with* their honest context."
        ),
        code(
            "labels = ['all\\nbroods', '17-year\\nonly', 'famous\\nmarquee']\n"
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for yrs in (EV_YEARS, data.brood_years(cycle=17),\n"
            "                data.brood_years(famous_only=True)):\n"
            "        e = st.build_event_table(SPY, AR, yrs, k=R['K'])\n"
            "        sa = st.one_sample_t(e['abn_car'].to_numpy())\n"
            "        hr = st.hit_rate(e['raw_ret'].to_numpy())\n"
            "        rows.append((hr['rate']*100, hr['lo']*100, hr['hi']*100, sa['t'], hr['n']))\n"
            "    non = [y for y in POOL if y not in set(EV_YEARS)]\n"
            "    wa = st.all_year_windows(SPY, POOL, k=R['K'])\n"
            "    a = np.array([wa[y] for y in EV_YEARS if y in wa.index])\n"
            "    b = np.array([wa[y] for y in non if y in wa.index])\n"
            "    welch = st.welch_t(a, b)\n"
            "else:\n"
            "    rows = [(R['cut_all'][6], R['cut_all'][7], R['cut_all'][8], R['cut_all'][3], R['cut_all'][0]),\n"
            "            (R['cut_17'][6], R['cut_17'][7], R['cut_17'][8], R['cut_17'][3], R['cut_17'][0]),\n"
            "            (R['cut_fam'][6], R['cut_fam'][7], R['cut_fam'][8], R['cut_fam'][3], R['cut_fam'][0])]\n"
            "    welch = R['welch_t']\n"
            "rates = [r[0] for r in rows]; los = [r[0]-r[1] for r in rows]; his = [r[2]-r[0] for r in rows]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))\n"
            "a1.bar(labels, rates, yerr=[los, his], capsize=5, color=[GREY, GREY, RED], width=.6)\n"
            "a1.axhline(50, ls='--', c='k', lw=1, label='coin flip')\n"
            "for i, r in enumerate(rows):\n"
            "    a1.annotate(f'{r[0]:.0f}%\\n(n={r[4]}, t={r[3]:+.2f})', (i, r[0]), ha='center', va='bottom', fontsize=8)\n"
            "a1.set_ylim(0, 125); a1.set_ylabel('up-rate (%, Wilson 95%)')\n"
            "a1.set_title('Selection at work: fewer, famous broods -> \\'perfect\\''); a1.legend()\n"
            "if HAVE_REAL:\n"
            "    a2.hist(b*100, bins=8, color=GREY, alpha=.6, label=f'non-cicada springs (n={len(b)})')\n"
            "    a2.hist(a*100, bins=12, color=AMBER, alpha=.6, label=f'cicada springs (n={len(a)})')\n"
            "    a2.axvline(a.mean()*100, c=AMBER, lw=2); a2.axvline(b.mean()*100, c=GREY, lw=2)\n"
            "else:\n"
            "    a2.text(.5, .5, 'cicada vs non-cicada springs', ha='center', transform=a2.transAxes)\n"
            "a2.set_xlabel('May-June return (%)'); a2.set_ylabel('years')\n"
            "a2.set_title(f'Cicada vs quiet springs: Welch t = {welch:+.2f}'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('up-rates:', [round(r[0]) for r in rows], '| abn-CAR t:', [round(r[3],2) for r in rows])\n"
            "print(f'Welch cicada-vs-non t = {welch:+.3f}')"
        ),
        md(
            f"> 💡 In plain words: the all-broods up-rate is **{R['cut_all'][6]:.0f}%** "
            f"(Wilson [{R['cut_all'][7]:.0f}%, {R['cut_all'][8]:.0f}%]) — wide, and sitting "
            "on an equity base rate that's already north of 50%. Slice to the 5 famous "
            f"broods and it's **{R['cut_fam'][6]:.0f}% up** at t = {R['cut_fam'][3]:+.2f}: "
            "a pure small-sample selection mirage (5 fair coins land 5 heads ~1 in 32). And "
            f"cicada springs don't beat quiet springs — Welch t = **{R['welch_t']:+.2f}**, "
            f"underpowered anyway because only **{R['welch_non_n']}** of the last 33 springs "
            "were cicada-free."
        ),
        md(
            "### 4d · The timer — beta vs alpha\n\n"
            "The tradability trap, made explicit. A one-sample *t* of the raw cicada-spring "
            "return vs **zero** looks significant — but that is just SPY's ordinary "
            "two-month drift. The edge, if any, is the **excess over the every-spring "
            "baseline**."
        ),
        code(
            "legs = ['gross', 'net 5bps', 'net 10bps']\n"
            "if HAVE_REAL:\n"
            "    tvs0, exc, exc_t = [], [], []\n"
            "    for cost in (0.0, 5.0, 10.0):\n"
            "        lg = st.spring_timer(SPY, EV_YEARS, k=R['K'], cost_bps=cost)\n"
            "        col = 'ret_gross' if cost == 0 else 'ret_net'\n"
            "        tvs0.append(st.summarize_timer(lg, col=col)['t'])\n"
            "        ex = st.excess_over_baseline(SPY, EV_YEARS, POOL, k=R['K'], cost_bps=cost)\n"
            "        exc.append(ex['excess_bps']); exc_t.append(ex['t'])\n"
            "else:\n"
            "    tvs0 = [R['timer_gross'][2], R['timer_net5'][2], R['timer_net10'][2]]\n"
            "    exc = [R['timer_gross'][3], R['timer_net5'][3], R['timer_net10'][3]]\n"
            "    exc_t = [R['timer_gross'][4], R['timer_net5'][4], R['timer_net10'][4]]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.5))\n"
            "x = np.arange(len(legs)); w = 0.38\n"
            "ax.bar(x - w/2, tvs0, width=w, color=GREY, label='t vs ZERO (beta: just equity drift)')\n"
            "ax.bar(x + w/2, exc_t, width=w, color=RED, label='t of EXCESS over every-spring baseline (alpha)')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(legs)\n"
            "ax.set_ylabel('one-sample t'); ax.set_ylim(-1, 3)\n"
            "for i, (v, e) in enumerate(zip(tvs0, exc_t)):\n"
            "    ax.annotate(f'{v:+.2f}', (i - w/2, v), ha='center', va='bottom', fontsize=8)\n"
            "    ax.annotate(f'{e:+.2f}', (i + w/2, e), ha='center', va='bottom', fontsize=8)\n"
            "ax.set_title('The whole illusion in one chart: beta clears 2, alpha is ~0')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('t vs zero (beta):', [round(v,2) for v in tvs0])\n"
            "print('excess bps:', [round(v) for v in exc], '| excess t (alpha):', [round(v,2) for v in exc_t])"
        ),
        md(
            f"> 💡 In plain words: the grey bars (t vs zero) clear 2 — and mean nothing, "
            "because holding *any* two-month equity window does that. The red bars (t of the "
            f"excess over just-holding-every-spring) sit at **+{R['timer_gross'][4]:.2f} / "
            f"+{R['timer_net5'][4]:.2f} / +{R['timer_net10'][4]:.2f}** across cost levels — "
            "no alpha. H₄ is not supported. This single chart is the study's whole moral: "
            "grade the *excess*, not the gross."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic random-walk SPY-like tape (~46 years, window vol matched to the real "
            "tape), alternating synthetic \"emergence\" years, a TUNABLE planted spring "
            "bump. The null (bump = 0) is checked over **20 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    close, em = data.synthetic_world(bump=0.0, seed=741 + s_)\n"
            "    null_ts.append(st.synthetic_detect(close, em, k=R['K'])['t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "close, em = data.synthetic_world(bump=0.04, seed=741)\n"
            "planted_t = st.synthetic_detect(close, em, k=R['K'])['t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (bump=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5, label='planted bump = +4.0%')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('one-sample t (abnormal CAR)')\n"
            "ax.set_title('Control: the null averages ~0; a planted bump lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages t = "
            f"{R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and crosses |t| ≥ 2 in "
            f"just {R['syn_null_fire']}/20 seeds — exactly the ~5% false-positive rate a "
            f"correct 5% test *should* show, never a systematic misfire; a planted +4% "
            f"spring bump reads t = {R['syn_planted_t']:.2f}. The machinery is unbiased, so "
            f"the real-tape t = {R['cut_all'][3]:+.2f} is a genuine null. *(A "
            "faithful-engine / power check only — never cited in support of the real-tape "
            "stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — cicada-spring abnormal CAR **+{R['cut_all'][2]:.2f}%**, "
            f"one-sample t = **{R['cut_all'][3]:+.2f}** (n={R['cut_all'][0]}); random-year "
            f"placebo **p = {R['placebo_p']:.3f}**; Welch vs non-cicada springs t = "
            f"**{R['welch_t']:+.2f}**; 1-month window t = {R['short_abn_t']:+.2f}; no "
            "event-anatomy offset clears |t| ≥ 2. The 75% up-rate and the 5/5 famous-brood "
            "cut are base-rate and selection artifacts.\n"
            f"- **Tradability `MIRAGE`** — the timer's +{R['timer_gross'][2]:.2f} t-vs-zero "
            f"is pure equity beta; the excess over the every-spring baseline is "
            f"**+{R['timer_gross'][3]:.0f} bps at t = {R['timer_gross'][4]:.2f}** (gross), "
            f"fading to +{R['timer_net10'][3]:.0f} bps net of 10 bps costs. No alpha to "
            "charge costs against.\n"
            f"- **\"Cicada indicator?\" `BUSTED`** — a brood emerges in "
            f"**{R['brood_years_in_span']} of {R['span']}** recent years, so \"cicada "
            "years\" are nearly the whole calendar. The pattern that survives is a coin flip "
            "on top of drift — exactly what a mechanism-free fixed calendar should produce."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **This study is a negative control by design.** Its value is precisely that "
            "the apparatus returns nothing on a pattern with no mechanism — the reassurance "
            "that the desk's *positive* verdicts elsewhere aren't the same machinery "
            "hallucinating. A green stamp here would be a bug report.\n"
            "- **The generalisable lesson.** Any fixed, evocative calendar can be mined for "
            "a small-n streak in a secular bull market; the two failure modes this notebook "
            "isolates — reading an up-rate against 50% instead of the equity base rate, and "
            "grading a timer on gross instead of excess-over-baseline — are how almanac "
            "indicators get sold. The antidote is out-of-sample data, a pre-registered "
            "window, and alpha-not-beta accounting.\n"
            "- **Dedup map:** [707-plane-crash-effect](../../707-plane-crash-effect/) and "
            "[708-eurovision-effect](../../708-eurovision-effect/) share this exact "
            "apparatus on real sentiment triggers; Cicada-Brood is the null control that "
            "calibrates them.\n\n"
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
