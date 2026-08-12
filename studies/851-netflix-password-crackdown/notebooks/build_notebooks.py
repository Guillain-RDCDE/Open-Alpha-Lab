"""Generate the two narrative notebooks for Study 851 (Netflix Password Crackdown).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached
NFLX/SPY/QQQ tapes under ../_cache/ and otherwise quote the frozen headline numbers in
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance NFLX/SPY/QQQ
# total-return closes 2015-01-02 -> 2026-06-30; 5 hardcoded public crackdown dates
# 2022-04-20 -> 2023-10-19; market-model AR, estimation window 120d ending 10d before
# the event window; event window [-1..+5]).
R = dict(
    n_events=5, cal_lo="2022-04-20", cal_hi="2023-10-19",
    # per-event day-0 abnormal return (vs SPY) and estimated beta
    ev={"2022-04-20": (-34.55, 1.60, "Q1'22 letter first flags paid-sharing (sub miss)"),
        "2022-08-22": (-2.34, 1.66, "LatAm 'add-a-home' test"),
        "2023-05-23": (-0.34, 1.50, "broad US rollout"),
        "2023-07-20": (-7.64, 1.37, "Q2'23 +5.9M subs (rev miss)"),
        "2023-10-19": (17.34, 1.55, "Q3'23 +8.8M subs — the payoff")},
    day0_mean_pct=-5.51, day0_t=-0.657, hit=1, hit_n=5, wilson=(4.0, 62.0),
    boot_lo=-21.7, boot_hi=8.4,
    wincar_mean_pct=-4.01, wincar_t=-0.383,
    postcar_mean_pct=1.39, postcar_t=0.447,
    qqq_day0_mean_pct=-4.79, qqq_day0_t=-0.594,
    placebo_mean_pct=-0.01, placebo_sd_pct=1.07, placebo_p_right=0.998,
    placebo_p_left=0.003, placebo_draws=4000,
    drop_crash_mean_pct=1.75, drop_crash_t=0.324, drop_crash_n=4,
    conf_mean_pct=4.85, conf_t=0.388, conf_n=2,
    # timer: hold -> (gross_bps, net10_bps, t_net, win_pct, uncond_bps)
    timer={1: (-87.7, -107.7, -1.06, 20, 12.9), 3: (65.4, 45.4, 0.19, 60, 39.6),
           5: (-231.9, -251.9, -0.56, 40, 66.6), 10: (62.7, 42.7, 0.11, 40, 134.0),
           21: (264.7, 244.7, 0.34, 60, 277.0)},
    syn_null_mean_t=0.37, syn_null_sd_t=1.06, syn_null_fire=1,
    syn_planted_pct=2.86, syn_planted_t=11.12,
    fp_nflx="5fff2e717a0c", fp_spy="da459bd64b22", fp_qqq="f8f1d5c5542a",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Upside surprise%3F: 1 of 5 events](https://img.shields.io/badge/Upside_surprise%3F-1_of_5_events-8b949e?style=flat-square)\n\n"
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

from nflx_crackdown import data, strategy as st

EVENTS = data.event_table()
HAVE_REAL = data.have_real()
if HAVE_REAL:
    PX = data.load_real()
    R_N = st.daily_returns(PX["NFLX"]); R_S = st.daily_returns(PX["SPY"]); R_Q = st.daily_returns(PX["QQQ"])
else:
    PX = R_N = R_S = R_Q = None
print("real cache present:", HAVE_REAL, "| crackdown dates in table:", len(EVENTS))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The scary Netflix policy that worked 🔐📈\n"
            "### The 2023 \"password crackdown\" was supposed to spike churn. Instead it "
            "became an upside surprise. Was that a *tradable* signal?\n\n"
            + BADGES +
            "For years, Netflix let password sharing slide. Then in 2022–23 it started "
            "**charging for it** — the \"password crackdown\". Wall Street was sure this "
            "would drive furious customers out the door. It didn't: subscribers *grew*, "
            "and by the Q3 2023 earnings the stock jumped ~16% in a day.\n\n"
            "That's a great story. But this desk asks a narrower, colder question: around "
            "the **five public dates** of the crackdown saga, did NFLX actually deliver an "
            "**abnormal return** you could have traded — or is \"the scary policy that "
            "worked\" a *business* fact that never was a *market* signal? With only **five "
            "events**, we already know the honest answer will be humble.\n\n"
            "> 📓 **Want the market-model math, the placebo and the cost sweep?** See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Did NFLX pop on the average crackdown date? | **No.** The average "
            f"event-day *abnormal* return (stripping out the market) is "
            f"**{R['day0_mean_pct']:+.1f}%** — and it's *negative*, dragged down by the "
            "2022 announcement crash. |\n"
            f"| Is that a real signal, or one outlier? | **One outlier.** Drop the single "
            f"2022-04 crash and the other four average **{R['drop_crash_mean_pct']:+.1f}%** "
            "— basically nothing. |\n"
            f"| Was there *any* upside surprise? | **Yes — exactly once.** Only "
            f"**{R['hit']} of {R['hit_n']}** events was an up-day: the Q3'23 confirmation "
            f"(**+{R['ev']['2023-10-19'][0]:.0f}%** abnormal). The policy worked; the "
            "*stock reaction* was a single spike, not a pattern. |\n"
            f"| Could you have traded it? | **No.** Buying NFLX after each event and "
            "holding never reliably beats just owning the stock — the point estimates are "
            "all over the place, none significant. |\n\n"
            "> A famous \"it worked!\" narrative — real as a business outcome, invisible "
            "as a five-event trading edge."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The password crackdown was the policy everyone said would backfire. "
            "Churn would spike, growth would stall. Instead Netflix added millions of "
            "subscribers — the scary policy quietly became the growth story of 2023, and "
            "the stock re-rated as the market realised it.\"*\n\n"
            "This is a real, well-documented corporate turnaround. The **five market-"
            "facing dates** are matters of public record — the Q1'22 shareholder letter "
            "that first flagged paid-sharing (amid the sub miss that crashed the stock), "
            "the 2022-08 LatAm test, the 2023-05 broad US rollout, and the Q2/Q3'23 "
            "letters that confirmed the subscriber gains."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "\"The policy worked\" and \"there was a tradable abnormal return around the "
            "policy\" are **two different claims**. Subscriber growth is a slow business "
            "fact; a stock reaction is a fast, one-day event. We measure the second one "
            "properly — NFLX's return on each event session with the market's move "
            "*removed* (a market model), so we're not just re-discovering \"tech went up "
            "in 2023\"."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** **{R['n_events']}** public crackdown dates, "
            f"{R['cal_lo']} → {R['cal_hi']}, hardcoded from the Netflix letters/newsroom "
            "(earnings print after the close, so the reaction is the next session).\n"
            "- **The yardstick.** NFLX's return on each event session **minus** what a "
            "market model (fit on the prior ~6 months vs SPY) says it *should* have done "
            "— the abnormal return.\n"
            "- **The luck check.** Draw 5 random dates instead, 4,000 times — is the real "
            "five-date average unusual, or just what a random handful of days looks like?\n"
            "- **The trade check.** Buy NFLX at each event close, hold a few sessions, pay "
            "costs, compare to simply holding NFLX."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the five events, one bar each.** Abnormal return on each event "
            "session (market removed). Notice the shape of the story."
        ),
        code(
            "labels = list(R['ev'].keys())\n"
            "if HAVE_REAL:\n"
            "    mat, kept, _ = st.event_car(R_N, R_S, EVENTS['date'], pre=1, post=5, model='market')\n"
            "    vals = [row[1] * 100 for row in mat]\n"
            "    labels = [d.strftime('%Y-%m-%d') for d in kept]\n"
            "else:\n"
            "    vals = [R['ev'][k][0] for k in labels]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "cols = [GREEN if v > 0 else RED for v in vals]\n"
            "ax.bar(range(len(vals)), vals, color=cols, width=.62)\n"
            "for i, v in enumerate(vals):\n"
            "    ax.annotate(f'{v:+.1f}%', (i, v), ha='center', va='bottom' if v > 0 else 'top', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(range(len(vals)))\n"
            "ax.set_xticklabels([l[2:] for l in labels], rotation=0, fontsize=8)\n"
            "ax.set_ylabel('abnormal NFLX return, event session (%)')\n"
            "ax.set_title('One big crash (the 2022 flag), one big pop (the 2023 payoff), noise between')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('event-day abnormal returns (%):', [round(v,1) for v in vals])"
        ),
        md(
            f"The picture *is* the finding: the 2022-04 announcement was a "
            f"**{R['ev']['2022-04-20'][0]:.0f}%** abnormal crash (Netflix flagged the "
            "policy in the same breath as its first subscriber decline in a decade), the "
            f"Q3'23 confirmation was a **+{R['ev']['2023-10-19'][0]:.0f}%** pop, and the "
            "three middle dates were noise. \"The policy worked\" is true — but the market "
            "reaction was **one crash and one pop**, not a repeatable signal.\n\n"
            "**Now average them and ask if it's distinguishable from luck.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    d0 = st.day0_stats(R_N, R_S, EVENTS['date'], model='market')\n"
            "    obs = d0['mean'] * 100\n"
            "    pl = st.placebo_distribution(R_N, R_S, d0['n'], model='market', n_draws=1500, seed=851)\n"
            "    draws = pl * 100\n"
            "else:\n"
            "    obs = R['day0_mean_pct']\n"
            "    rng = np.random.default_rng(851)\n"
            "    draws = rng.normal(R['placebo_mean_pct'], R['placebo_sd_pct'], 1500)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=45, color=GREY, alpha=.85, label='null: 5 random dates (light in-notebook run)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed 5-event mean {obs:+.1f}%')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('mean abnormal NFLX return of a random 5-date calendar (%)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title('The real five-date average sits in the LEFT tail — the wrong side for an upside claim')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"observed {obs:+.2f}%  vs random-calendar mean {R['placebo_mean_pct']:+.2f}% \"\n"
            "      f\"(sd {R['placebo_sd_pct']:.2f}%); canonical left-tail p = {R['placebo_p_left']:.3f}\")"
        ),
        md(
            f"> Because one −35% crash sits in the sample, the *average* of the five is "
            f"actually **negative** ({R['day0_mean_pct']:+.1f}%) and lands in the **left** "
            "tail of the random-calendar null — the opposite direction from the \"upside "
            "surprise\" the story promises. That's not evidence of a *downside* edge "
            "either; it's what happens when one outlier dominates a five-point average. "
            "The honest read is simply: **five events can't tell you much.**\n\n"
            "**Finally, could you have traded it?**"
        ),
        code(
            "holds = sorted(R['timer'])\n"
            "if HAVE_REAL:\n"
            "    net, base = [], []\n"
            "    for h in holds:\n"
            "        n10 = st.summarize_trade(st.buy_the_event(PX['NFLX'], EVENTS['date'], hold=h, cost_bps=10.0), 'ret_net')\n"
            "        net.append(n10['mean_bps'])\n"
            "        base.append(float((PX['NFLX'].shift(-h) / PX['NFLX'] - 1.0).mean() * 1e4))\n"
            "else:\n"
            "    net = [R['timer'][h][1] for h in holds]\n"
            "    base = [R['timer'][h][4] for h in holds]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "x = np.arange(len(holds)); w = 0.38\n"
            "ax.bar(x - w/2, net, width=w, color=RED, label='buy-the-event (net, 10 bps)')\n"
            "ax.bar(x + w/2, base, width=w, color=GREY, label='just hold NFLX (unconditional)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in holds])\n"
            "ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('Buying the event is a coin toss around plain buy-and-hold')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('net buy-the-event (bps):', dict(zip(holds, [round(v) for v in net])))\n"
            "print('unconditional NFLX (bps):', dict(zip(holds, [round(v) for v in base])))"
        ),
        md(
            "At every horizon the \"buy the crackdown event\" trade is indistinguishable "
            "from — often worse than — simply owning NFLX, with win rates from 20% to 60% "
            "on five trades. There is no edge here to bank.\n\n"

            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The average crackdown-date abnormal return is "
            f"**{R['day0_mean_pct']:+.1f}%** (*t* = {R['day0_t']:.2f}); the entire result "
            "is one 2022 crash and one 2023 pop. Only 1 of 5 events was the promised "
            "upside. As a *tradable news signal*, it isn't there.\n"
            "- **Tradability — Mirage.** No horizon beats buy-and-hold; five trades can't "
            "clear costs or significance.\n"
            "- **\"The scary policy that worked?\" — true, but as a business fact, not a "
            "market edge.** Subscribers grew; the stock's reaction was a single spike you "
            "couldn't have known to catch in advance.\n\n"

            "## 6 · Going further 🚪\n\n"
            "- **This is a case study, not a factor.** Five events is far too few for "
            "statistics; the value here is the *anatomy*, not a *t*-stat. Don't read the "
            "negative average as a short signal — read it as \"one outlier, no power.\"\n"
            "- **Where a real version might live:** a *portfolio* of many "
            "\"feared-policy-that-worked\" corporate events across names and years, so the "
            "sample size can actually support inference.\n"
            "- **Sibling studies:** [551-netflix-top10](../../551-netflix-top10/) (a "
            "different NFLX signal — the Top-10 content chart), "
            "[552-app-store-rankings](../../552-app-store-rankings/) (app-download alt-data), "
            "[299-keynote-drift](../../299-keynote-drift/) (a scheduled-announcement event "
            "study) and [622-thematic-etf-curse](../../622-thematic-etf-curse/) "
            "(narrative-driven launches) — the same \"does a story move a tradable price\" "
            "question, different triggers.\n\n"
            "*Think a broad basket of feared-but-successful policy events carries a real "
            "drift? Build it — many names, out of sample, after costs — then we'll talk.*"
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
            "# Netflix Password Crackdown — a quantitative teardown 🔬\n"
            "### A market-model event study on 5 public dates · per-event CARs · a "
            "4,000-draw random-calendar placebo · a leave-one-out robustness cut · a "
            "costed long-only timer · a 20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb). The claim — Netflix's 2023 paid-sharing "
            "(\"password crackdown\"), feared to spike churn, became an upside surprise — "
            "is a real corporate turnaround. The job here is to test whether it shows up "
            "as a **tradable abnormal-return signal** around its five public dates, and to "
            "be explicit that **N = 5 → almost no power** (this is a case study).\n\n"
            "> ⚠️ **Data note.** NFLX + SPY + QQQ total-return closes (2015→2026-06-30), "
            "yfinance, cached; **5 hardcoded public crackdown dates** 2022→2023 (Netflix "
            "letters/newsroom). Market model: OLS α/β on a 120-session estimation window "
            "ending 10 sessions before the event window; event window [−1..+5]; earnings "
            "react next session (one documented lag). Numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_nflx"] +
            "` NFLX / `" + R["fp_spy"] + "` SPY / `" + R["fp_qqq"] + "` QQQ).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | cross-event mean day-0 abnormal return "
            f"**{R['day0_mean_pct']:+.2f}%**, one-sample **t = {R['day0_t']:+.2f}** "
            f"(n=5), up-days {R['hit']}/{R['hit_n']} (Wilson [{R['wilson'][0]:.0f}%, "
            f"{R['wilson'][1]:.0f}%]) |\n"
            f"| **Tradability** | `MIRAGE` | buy-the-event never reliably beats holding "
            f"NFLX; net *t* ∈ [{R['timer'][1][2]:.2f}, {R['timer'][21][2]:.2f}] across "
            "1/3/5/10/21-day holds |\n"
            f"| **Upside surprise?** | `1 of 5` | only Q3'23 (**+{R['ev']['2023-10-19'][0]:.0f}%** "
            f"abnormal) delivered; the mean is dragged negative by the "
            f"**{R['ev']['2022-04-20'][0]:.0f}%** 2022 flag crash |\n\n"
            "> 💡 In plain words: a genuine business turnaround, but the five-event stock "
            "reaction is one crash + one pop + three shrugs — no power, no tradable edge."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be NFLX's daily return and $r^m_t$ the benchmark's. Under a "
            "one-factor market model fit on an estimation window, the *normal* return is "
            "$\\hat\\alpha + \\hat\\beta r^m_t$ and the **abnormal** return is "
            "$a_t = r_t - (\\hat\\alpha + \\hat\\beta r^m_t)$. For each crackdown event "
            "$i$ with reaction session $\\tau_i$:\n\n"
            "- **H₁ (upside surprise).** $E[a_{\\tau_i}] > 0$, systematically across "
            "events.\n"
            "- **H₂ (drift).** The post-event CAR $\\sum_{k=1}^{5} a_{\\tau_i+k}$ is "
            "positive (a re-rating that keeps paying).\n"
            "- **H₃ (capture).** A long-NFLX-after-the-event overlay beats unconditional "
            "buy-and-hold net of costs.\n\n"
            f"We find **H₁ not supported** (mean {R['day0_mean_pct']:+.2f}%, t = "
            f"{R['day0_t']:.2f} — *negative*, driven by one outlier), **H₂ not "
            f"supported** (post-CAR {R['postcar_mean_pct']:+.2f}%, t = "
            f"{R['postcar_t']:.2f}), **H₃ not supported** (no horizon beats hold). The "
            "honest headline is **low power**, not a discovered anti-signal."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The five events are **independent, non-overlapping calendar dates**, so the "
            "planned primary is a **one-sample t** across the per-event abnormal returns "
            "— but with **4 degrees of freedom** that t has famously fat tails (the |t|≥2 "
            "rule of thumb corresponds to p≈0.12 here, not 0.05), which is exactly why we "
            "lean on a **non-parametric random-calendar placebo** (4,000 draws of 5 random "
            "dates) and an **event-bootstrap CI** instead of trusting the t alone. A "
            "**leave-one-out** cut isolates how completely the 2022 crash drives the "
            "result. The **synthetic control** is run on **30** pseudo-events precisely so "
            "the machinery's calibration can be judged free of the small-N fat tails."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_events']} public dates {R['cal_lo']} → {R['cal_hi']}, "
            "hardcoded (earnings react next session — one documented lag).\n"
            "- **Tape.** NFLX + SPY + QQQ total-return closes, 2015 → 2026-06-30 (as-of, "
            "last complete month).\n"
            "- **Abnormal return.** Market model, OLS α/β on 120 sessions ending 10 "
            "sessions before the event window (strictly out-of-sample); SPY headline, QQQ "
            "cross-check.\n"
            "- **Headline.** Cross-event mean day-0 AR + one-sample t + Wilson hit rate + "
            "event-bootstrap CI.\n"
            "- **Anatomy.** CAR path [−1..+5]; whole-window and post-event CARs.\n"
            "- **Placebo.** 4,000 random 5-date calendars, both tails.\n"
            "- **Robustness.** Drop the 2022-04 crash; confirmation-earnings-only subset.\n"
            "- **Timer.** Long NFLX from the event close, hold 1/3/5/10/21 sessions, one "
            "round trip of one-way cost ×2, vs unconditional NFLX baseline.\n"
            "- **Control.** Synthetic one-factor tape, planted one-day jump on 30 "
            "pseudo-events; the null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Per-event abnormal returns and the CAR path\n\n"
            "The event window [−1..+5] abnormal-return path (SPY market model), and each "
            "event's own day-0 abnormal return."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp = st.car_path_stats(R_N, R_S, EVENTS['date'], pre=1, post=5, model='market')\n"
            "    ks = list(cp.index); car = list(cp['car'] * 100)\n"
            "    mat, kept, betas = st.event_car(R_N, R_S, EVENTS['date'], pre=1, post=5, model='market')\n"
            "    ev_vals = [row[1] * 100 for row in mat]\n"
            "    ev_lab = [d.strftime('%y-%m-%d') for d in kept]\n"
            "else:\n"
            "    ks = list(range(-1, 6)); car = None\n"
            "    ev_lab = [k[2:] for k in R['ev']]; ev_vals = [R['ev'][k][0] for k in R['ev']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.4))\n"
            "cols = [GREEN if v > 0 else RED for v in ev_vals]\n"
            "a1.bar(range(len(ev_vals)), ev_vals, color=cols, width=.62)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_xticks(range(len(ev_vals)))\n"
            "a1.set_xticklabels(ev_lab, rotation=45, fontsize=8, ha='right')\n"
            "a1.set_ylabel('day-0 abnormal return (%)')\n"
            "a1.set_title('Per-event: one crash, one pop')\n"
            "if car is not None:\n"
            "    a2.plot(ks, car, marker='o', color=GREY)\n"
            "    a2.axhline(0, c='k', lw=.8); a2.set_xlabel('offset (sessions)')\n"
            "    a2.set_ylabel('mean CAR (%)')\n"
            "    a2.set_title('Mean CAR path [-1..+5] — noisy, near zero, huge error bars')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('day-0 abnormal (%):', [round(v,1) for v in ev_vals])\n"
            "print(f\"whole-window CAR {R['wincar_mean_pct']:+.2f}% (t={R['wincar_t']:.2f}); \"\n"
            "      f\"post-event CAR {R['postcar_mean_pct']:+.2f}% (t={R['postcar_t']:.2f})\")"
        ),
        md(
            f"> 💡 In plain words: the CAR path is a wandering line near zero — the "
            f"whole-window CAR is **{R['wincar_mean_pct']:+.2f}%** (t = {R['wincar_t']:.2f}) "
            f"and the post-event drift **{R['postcar_mean_pct']:+.2f}%** (t = "
            f"{R['postcar_t']:.2f}). Nothing here clears the bar; the two big bars (2022 "
            "crash, 2023 pop) sit on opposite sides and cancel."
        ),
        md(
            "### 4b · The headline mean and its random-calendar placebo\n\n"
            "Cross-event mean day-0 AR vs 4,000 random 5-date calendars. In the notebook "
            "we run a lighter placebo and quote the canonical 4,000-draw p from "
            "`results.md`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    d0 = st.day0_stats(R_N, R_S, EVENTS['date'], model='market')\n"
            "    obs = d0['mean']\n"
            "    blo, bhi = st.block_bootstrap_ci(d0['per_event'], n_boot=4000, seed=851)\n"
            "    pl = st.placebo_distribution(R_N, R_S, d0['n'], model='market', n_draws=1500, seed=851)\n"
            "    draws = pl * 100\n"
            "    print(f\"mean {d0['mean']*100:+.2f}%  t = {d0['t']:+.2f}  \"\n"
            "          f\"bootstrap 95% CI [{blo*100:+.1f}%, {bhi*100:+.1f}%]\")\n"
            "else:\n"
            "    obs = R['day0_mean_pct'] / 100\n"
            "    rng = np.random.default_rng(851)\n"
            "    draws = rng.normal(R['placebo_mean_pct'], R['placebo_sd_pct'], 1500)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.hist(draws, bins=45, color=GREY, alpha=.85, label='null: random 5-date calendars')\n"
            "ax.axvline(obs * 100, c=RED, lw=2.5, label=f'observed {obs*100:+.2f}%')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('mean abnormal NFLX return, random 5-date calendar (%)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Observed sits in the LEFT tail (canonical left-p={R['placebo_p_left']:.3f}) — one outlier, wrong sign for H1\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical placebo: mean {R['placebo_mean_pct']:+.2f}% (sd {R['placebo_sd_pct']:.2f}%), \"\n"
            "      f\"left-p {R['placebo_p_left']:.3f}, right-p {R['placebo_p_right']:.3f}; \"\n"
            "      f\"Wilson hit [{R['wilson'][0]:.0f}%,{R['wilson'][1]:.0f}%]\")"
        ),
        md(
            f"> 💡 In plain words: the observed **{R['day0_mean_pct']:+.2f}%** is more "
            f"*negative* than 99.7% of random five-date calendars (left-p = "
            f"{R['placebo_p_left']:.3f}) — but that is the **2022 crash outlier** talking, "
            f"not a downside edge. The event-bootstrap CI **[{R['boot_lo']:+.1f}%, "
            f"{R['boot_hi']:+.1f}%]** straddles zero by a mile. H₁ (a positive upside "
            "surprise) is not supported; neither is any robust negative."
        ),
        md(
            "### 4c · Robustness — the whole result is one date\n\n"
            "Leave-one-out on the 2022-04 announcement crash, and a confirmation-earnings-"
            "only subset."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sub = EVENTS.loc[EVENTS['date'] != '2022-04-20', 'date']\n"
            "    d0s = st.day0_stats(R_N, R_S, sub, model='market')\n"
            "    conf = EVENTS.loc[EVENTS['date'].isin(pd.to_datetime(['2023-07-20','2023-10-19'])), 'date']\n"
            "    d0c = st.day0_stats(R_N, R_S, conf, model='market')\n"
            "    drop_m, drop_t = d0s['mean']*100, d0s['t']\n"
            "    conf_m, conf_t = d0c['mean']*100, d0c['t']\n"
            "else:\n"
            "    drop_m, drop_t = R['drop_crash_mean_pct'], R['drop_crash_t']\n"
            "    conf_m, conf_t = R['conf_mean_pct'], R['conf_t']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(['all 5', 'drop 2022 crash\\n(n=4)', 'confirmation only\\n(n=2)'],\n"
            "       [R['day0_mean_pct'], drop_m, conf_m], color=[RED, GREY, AMBER], width=.55)\n"
            "for i, v in enumerate([R['day0_mean_pct'], drop_m, conf_m]):\n"
            "    ax.annotate(f'{v:+.1f}%', (i, v), ha='center', va='bottom' if v>0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('mean day-0 abnormal return (%)')\n"
            "ax.set_title('Remove one crash and the negative vanishes — a five-point average is fragile')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'drop-crash (n=4): {drop_m:+.2f}% (t={drop_t:+.2f});  '\n"
            "      f'confirmation-only (n=2): {conf_m:+.2f}% (t={conf_t:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: drop the single 2022-04 crash and the mean flips to "
            f"**{R['drop_crash_mean_pct']:+.2f}%** (t = {R['drop_crash_t']:.2f}) — "
            "essentially zero. The confirmation earnings alone (Q2+Q3'23) average "
            f"**{R['conf_mean_pct']:+.2f}%** (t = {R['conf_t']:.2f}), positive but "
            "utterly powerless on n=2. Every cut says the same thing: too few events."
        ),
        md(
            "### 4d · The timer — an honest \"buy the event\" cost sweep\n\n"
            "Enter NFLX at the event-session close (zero look-ahead — the after-close "
            "print is already public), hold `h` sessions, one round trip of one-way costs "
            "charged twice, long-only, vs the unconditional NFLX `h`-day baseline."
        ),
        code(
            "holds = sorted(R['timer'])\n"
            "if HAVE_REAL:\n"
            "    gross, net10, ts, base = [], [], [], []\n"
            "    for h in holds:\n"
            "        g = st.summarize_trade(st.buy_the_event(PX['NFLX'], EVENTS['date'], hold=h, cost_bps=0.0), 'ret_gross')\n"
            "        n10 = st.summarize_trade(st.buy_the_event(PX['NFLX'], EVENTS['date'], hold=h, cost_bps=10.0), 'ret_net')\n"
            "        gross.append(g['mean_bps']); net10.append(n10['mean_bps']); ts.append(n10['t'])\n"
            "        base.append(float((PX['NFLX'].shift(-h) / PX['NFLX'] - 1.0).mean() * 1e4))\n"
            "else:\n"
            "    gross = [R['timer'][h][0] for h in holds]\n"
            "    net10 = [R['timer'][h][1] for h in holds]\n"
            "    ts = [R['timer'][h][2] for h in holds]\n"
            "    base = [R['timer'][h][4] for h in holds]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "x = np.arange(len(holds)); w = 0.27\n"
            "ax.bar(x - w, gross, width=w, color=GREY, label='gross')\n"
            "ax.bar(x, net10, width=w, color=RED, label='net (10 bps)')\n"
            "ax.bar(x + w, base, width=w, color=GREEN, label='unconditional NFLX')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in holds])\n"
            "ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('No horizon is a reliable win vs plain buy-and-hold (n=5 trades)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('net 10bps (bps):', dict(zip(holds, [round(v) for v in net10])))\n"
            "print('t(net):', dict(zip(holds, [round(t,2) for t in ts])))\n"
            "print('unconditional NFLX (bps):', dict(zip(holds, [round(v) for v in base])))"
        ),
        md(
            f"> 💡 In plain words: the net buy-the-event return swings from "
            f"**{R['timer'][1][1]:+.0f} bps** (1d) to **{R['timer'][21][1]:+.0f} bps** "
            f"(21d) with t-stats in [{R['timer'][1][2]:.2f}, {R['timer'][21][2]:.2f}] — "
            "pure noise on five trades, and never a clean win over just holding NFLX. "
            "H₃ is not supported. **Tradability: Mirage.**"
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic one-factor tape, **30** scheduled pseudo-events, a TUNABLE planted "
            "one-day jump. The null (edge=0) is checked over **20 seeds**; 30 events (not "
            "5) so the small-N fat tails don't muddy the calibration read."
        ),
        code(
            "null_ts = np.array([st.synthetic_detect(*data.synthetic_world(edge=0.0, seed=851+s))['t']\n"
            "                    for s in range(20)])\n"
            "a, m, e = data.synthetic_world(edge=0.03, seed=851)\n"
            "planted = st.synthetic_detect(a, m, e)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40, label='null worlds (edge=0), 20 seeds')\n"
            "ax.scatter([1], [planted['t']], color=GREEN, s=90, zorder=5, label='planted jump = +3%')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('one-sample t (day-0 abnormal return)')\n"
            "ax.set_title('Control: null centres at ~0, a planted jump lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), \"\n"
            "      f\"|t|>=2 in {(np.abs(null_ts)>=2).sum()}/20  |  planted t = {planted['t']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages t = "
            f"{R['syn_null_mean_t']:+.2f} (sd {R['syn_null_sd_t']:.2f}), crossing |t|≥2 "
            f"in {R['syn_null_fire']}/20 — well-calibrated; a planted +3% jump reads t = "
            f"{R['syn_planted_t']:.2f}. The engine is unbiased and powerful *when it has "
            "the events*. The real-tape emptiness is a **sample-size** verdict, not a "
            "broken detector. *(A faithful-engine / power check only — never cited in "
            "support of the real-tape stamp.)*\n\n"

            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — cross-event mean day-0 abnormal return "
            f"**{R['day0_mean_pct']:+.2f}%** (one-sample t = {R['day0_t']:.2f}, n=5), "
            f"up-days {R['hit']}/{R['hit_n']}, whole-window CAR {R['wincar_mean_pct']:+.2f}% "
            f"(t = {R['wincar_t']:.2f}). The negative average is one 2022 crash "
            f"(leave-one-out → {R['drop_crash_mean_pct']:+.2f}%); the promised upside "
            "surprise is a single event (Q3'23). No robust signal, either sign.\n"
            f"- **Tradability `MIRAGE`** — buy-the-event never reliably beats holding "
            f"NFLX; net t ∈ [{R['timer'][1][2]:.2f}, {R['timer'][21][2]:.2f}] over "
            "1–21-day holds.\n"
            "- **\"The scary policy that worked?\" — a business fact, not a market edge.** "
            "Subscribers grew; the tradable abnormal-return signal around the events does "
            "not exist at n=5.\n\n"

            "## 6 · Going further\n\n"
            "- **Power is the whole story.** Five independent events cannot support a "
            "factor claim; the deliverable is an honest anatomy. A pre-registered "
            "*portfolio* of many feared-but-successful policy events (across firms and "
            "years) is the only way to turn this into inference.\n"
            "- **Dedup map:** [551-netflix-top10](../../551-netflix-top10/) (NFLX Top-10 "
            "content signal), [552-app-store-rankings](../../552-app-store-rankings/) "
            "(app-download alt-data), [299-keynote-drift](../../299-keynote-drift/) "
            "(scheduled-announcement drift) and "
            "[622-thematic-etf-curse](../../622-thematic-etf-curse/) (narrative launches) "
            "— same \"does a story move a tradable price\" question, all on this desk's "
            "honesty rails.\n\n"
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
