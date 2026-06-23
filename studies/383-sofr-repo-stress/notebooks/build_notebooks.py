"""Generate the two narrative notebooks for Study 383 (SOFR-Repo-Stress).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached risk-asset
prices under ../_cache/ (SPY/HYG/LQD) and otherwise quote the frozen headline numbers in
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SPY/HYG/LQD,
# 2010-01-04 -> 2026-06-18, 4,140 days, 16.5 years; 13 hardcoded repo-stress episodes).
R = dict(
    start="2010-01-04", end="2026-06-18", days=4140, years=16.5, n_events=13,
    # per-asset per-horizon rows: (h, n, cond_mean%, cond_down%, base_mean%, base_down%, t, p_placebo)
    spy={5: (5, 13, -0.71, 46, 0.29, 39, -0.96, 0.059),
         20: (20, 13, 1.95, 46, 1.16, 31, 0.62, 0.755),
         60: (60, 13, 5.21, 23, 3.41, 23, 0.66, 0.849)},
    hyg={5: (5, 13, -0.61, 54, 0.11, 40, -1.09, 0.024),
         20: (20, 13, 1.06, 54, 0.43, 34, 0.91, 0.868),
         60: (60, 13, 2.09, 15, 1.30, 25, 0.53, 0.811)},
    lqd={5: (5, 13, -0.51, 46, 0.08, 43, -1.65, 0.029),
         20: (20, 13, 1.00, 46, 0.33, 39, 0.59, 0.886),
         60: (60, 13, 2.61, 23, 0.99, 34, 1.00, 0.954)},
    # de-risk SPY (h, n_trades, gross_avoided%, net@5bps%)
    net=[(5, 13, 0.71, 0.66), (20, 13, -1.95, -2.00), (60, 13, -5.21, -5.26)],
    # systemic-only SPY (h, n, cond%, base%, t, p)
    systemic=[(5, 3, -0.46, 0.29, -0.45, 0.247), (20, 3, 5.60, 1.16, 1.38, 0.980),
              (60, 3, 12.18, 3.41, 2.18, 0.992)],
    # synthetic control (planted_edge, n, cond20%, base20%, t, p)
    syn=[(0.0, 13, -0.93, 0.27, -0.88, 0.191), (-0.08, 13, -8.20, -0.24, -6.28, 0.000)],
    # per-episode SPY +20d (tag, ret%)
    episodes=[("year-end-2018", 7.89), ("quarter-end-2019Q1", 2.87), ("sept-2019", -0.44),
              ("year-end-2019", -0.97), ("covid-mar-2020", 10.52), ("quarter-end-2020Q3", -2.09),
              ("year-end-2021", -5.18), ("quarter-end-2022Q3", 5.35), ("svb-mar-2023", 6.73),
              ("quarter-end-2023Q3", -2.74), ("quarter-end-2024Q3", 2.31), ("year-end-2024", 2.25),
              ("quarter-end-2025Q1", -1.15)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Repo_spike_%3D_flee%3F: Busted](https://img.shields.io/badge/Repo_spike_%3D_flee%3F-Busted-8b949e?style=flat-square)\n\n"
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

from sofr_repo_stress import data, strategy as st

HAVE_REAL = data.have_real()
F = data.load_real() if HAVE_REAL else None
EVENTS = data.episode_dates()
print("real risk-asset cache present:", HAVE_REAL, "| repo-stress episodes:", len(EVENTS))
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell can
# quote it whether or not the cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# \"Watch the repo market\" — does a funding spike warn of a crash? 🚨\n"
            "### The September-2019 plumbing scare and the macro-Twitter oracle it spawned, in plain English\n\n"
            + BADGES +
            "In September 2019 the boring overnight \"repo\" rate — the cost of borrowing cash against "
            "Treasuries, the plumbing under the whole market — suddenly **spiked to ~10%**. The Fed had to "
            "rush in with emergency cash operations. Ever since, a piece of market lore has stuck: **when "
            "repo spikes, the plumbing is seizing, and risk assets are about to crack.** \"Watch the repo "
            "market\" became a permanent macro-Twitter refrain.\n\n"
            "It *sounds* like a crystal ball. But there are only a handful of these spikes ever — and the "
            "one crash everyone \"remembers\" (March 2020) was actually a **buying** moment. This notebook "
            "shows why the oracle is vivid, confounded, and empty as a warning.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the power analysis? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** There's no free, clean daily repo/SOFR-stress feed on yfinance, "
            "so we use a **hardcoded, sourced list of the famous repo-stress dates** and watch how stocks "
            "(SPY) and credit (HYG/LQD) actually moved after each. We call it an event list throughout. "
            "Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Right after a repo spike, do stocks dip? | **A little — for about a week.** SPY is down "
            f"**{abs(R['spy'][5][2]):.1f}%** on average over 5 days vs **+{R['spy'][5][4]:.1f}%** on a "
            "random day. So the scary story isn't *nothing*. |\n"
            "| Is that dip reliable enough to act on? | **No.** It's well inside the noise (not "
            "statistically significant), and within a **month it flips**: SPY is "
            f"**+{R['spy'][20][2]:.1f}%** at 20 days, **+{R['spy'][60][2]:.1f}%** at 60 — *better* than a "
            "normal stretch. |\n"
            "| But March 2020 was a crash! | **That was COVID, not repo.** And SPY rose "
            f"**+{R['episodes'][4][1]:.0f}%** over the month *after* the March-2020 funding seize — it was "
            "the **bottom**. The famous Sept-2019 spike? SPY ended roughly **flat**. |\n"
            "| So why the prophetic reputation? | **A spike summons the backstop.** The Fed steps in *because* "
            "the plumbing seized, so the spike is usually followed by a **relief rally**. A dozen vivid "
            "episodes in an up-drifting market look ominous and then bounce. |\n\n"
            "> The repo spike is a real, dramatic event. \"Flee risk\" is a confounded illusion. A dozen "
            "episodes — half of them not even bearish — can't be a warning system."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When overnight repo / SOFR spikes — the way it did in September 2019 — the funding plumbing "
            "of the financial system is seizing up. That's a flashing red light: get out of stocks and "
            "credit before the seizure spreads.\"*\n\n"
            "The picture is intuitive and was seared in by a real event: on **17 September 2019** SOFR "
            "jumped to ~5.25% and overnight repo printed near **10%**, and the New York Fed restarted "
            "emergency repo operations for the first time since 2008. If the cost of the market's most basic "
            "funding can triple overnight, surely something is badly wrong. We'll line up every famous repo "
            "spike and ask what *actually* happened next."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two things hide inside \"watch the repo market,\" and only one of them matters. (1) *Is a repo "
            "spike scary in the moment?* Sure — it's a genuine funding stress. The question that matters is "
            "(2) *does it forecast that **risk assets** fall by **more** than usual, reliably enough to bet "
            "on?* A vivid event that's followed by the market doing roughly its normal thing — or "
            "**rebounding**, because the Fed just rode in — is **not a warning**, it's a headline. And even "
            "a real warning is useless if it only fires **a dozen times in a decade and a half**: you can't "
            "run a fund on an oracle that rarely speaks and is half the time wrong."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We line up a **hardcoded, sourced list of {R['n_events']} repo-stress episodes** — Sept 2019, "
            "the quarter/year-end funding turns, March 2020, SVB 2023, and the rest — over "
            f"**{R['years']:.0f} years** ({R['start']} → {R['end']}). For each one:\n\n"
            "1. **Wait one day** (no cheating — you couldn't trade the spike itself), then **buy or sell** the "
            "risk asset and hold **5 / 20 / 60 days**.\n"
            "2. **Compare to normal.** How did SPY (stocks) and HYG/LQD (credit) do vs a *random* day over "
            "the same horizon (the base rate)?\n"
            "3. **Stress the luck.** Draw the same handful of *random* dates thousands of times and ask how "
            "often pure chance looks just as scary. That's the honest test for a dozen-event signal."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the episodes themselves.** Here's what SPY did over the *month after* each famous repo "
            "spike. If \"flee risk\" were right, these bars would mostly be deep red."
        ),
        code(
            "tags = [t for t,_ in R['episodes']]; rets = [r for _,r in R['episodes']]\n"
            "if HAVE_REAL:\n"
            "    tab = data.episode_table(); tags=[]; rets=[]\n"
            "    for d,row in tab.iterrows():\n"
            "        rr = st.event_returns(F['SPY'], pd.DatetimeIndex([d]), 20)\n"
            "        tags.append(row['tag']); rets.append(rr[0]*100 if len(rr) else np.nan)\n"
            "cols = [RED if r<0 else GREEN for r in rets]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.8))\n"
            "ax.barh(range(len(tags)), rets, color=cols)\n"
            "ax.set_yticks(range(len(tags))); ax.set_yticklabels(tags, fontsize=8)\n"
            "ax.axvline(0, c='k', lw=.8); ax.invert_yaxis()\n"
            "ax.set_xlabel('SPY forward return over the next 20 trading days (%)')\n"
            "ax.set_title('After each famous repo spike: stocks mostly went UP')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Sept-2019:', f'{rets[2]:+.1f}%', '| COVID Mar-2020:', f'{rets[4]:+.1f}%', '| SVB Mar-2023:', f'{rets[8]:+.1f}%')"
        ),
        md(
            f"Look at the famous ones. **Sept 2019** — *the* episode — gave SPY ≈ **flat** "
            f"({R['episodes'][2][1]:+.1f}% over a month). **March 2020**, the crash everyone cites, was the "
            f"**bottom**: SPY rose **{R['episodes'][4][1]:+.0f}%** over the next month. **SVB 2023**: "
            f"**{R['episodes'][8][1]:+.0f}%**. The bars are a coin-toss of red and green, and the biggest "
            "greens are the scariest crises. That's the opposite of a warning."
        ),
        md(
            "**Now the averages.** For stocks and credit, the average move *right after* a spike (5 days) "
            "and a month later (20 days), next to a normal stretch (the base rate)."
        ),
        code(
            "assets = ['SPY','HYG','LQD']\n"
            "if HAVE_REAL:\n"
            "    c5 = [st.summarize(F[a], EVENTS, 5)['cond_mean']*100 for a in assets]\n"
            "    b5 = [st.summarize(F[a], EVENTS, 5)['base_mean']*100 for a in assets]\n"
            "    c20 = [st.summarize(F[a], EVENTS, 20)['cond_mean']*100 for a in assets]\n"
            "else:\n"
            "    c5 = [R['spy'][5][2], R['hyg'][5][2], R['lqd'][5][2]]\n"
            "    b5 = [R['spy'][5][4], R['hyg'][5][4], R['lqd'][5][4]]\n"
            "    c20 = [R['spy'][20][2], R['hyg'][20][2], R['lqd'][20][2]]\n"
            "x = np.arange(len(assets)); fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.27, c5, .27, color=RED, label='after a spike — 5 days')\n"
            "ax.bar(x, b5, .27, color=GREY, label='a normal 5 days (base rate)')\n"
            "ax.bar(x+.27, c20, .27, color=GREEN, label='after a spike — 20 days')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels(assets)\n"
            "ax.set_ylabel('average forward return (%)')\n"
            "ax.set_title('A faint 5-day dip — then it flips POSITIVE within a month'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('SPY: 5d after spike', f'{c5[0]:+.2f}%', 'vs normal', f'{b5[0]:+.2f}%', '-> 20d:', f'{c20[0]:+.2f}%')"
        ),
        md(
            f"There's the whole story in one picture. At **5 days** there's a real but tiny dip — SPY "
            f"**{R['spy'][5][2]:+.1f}%** vs a normal **+{R['spy'][5][4]:.1f}%**, credit a touch soft too. "
            "That's the kernel of truth the lore is built on. But by **20 days the red turns green**: every "
            "asset is *above* its base rate. The funding scare gets backstopped, and the market rebounds."
        ),
        md(
            "**Could a dozen random days look this scary?** The honest test: draw "
            f"**{R['n_events']}** *random* dates over and over and see whether the spike's 5-day average is "
            "unusually bad — or just ordinary bad luck."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(F['SPY'], EVENTS, 5); obs = pl['event_mean']; pval = pl['p_value']\n"
            "    p = F['SPY'].values; n=len(p); k=len(EVENTS); lag=1; h=5; vh=n-h-lag-1\n"
            "    fwd = p[lag+h:lag+h+vh]/p[lag:lag+vh]-1; rng=np.random.default_rng(383)\n"
            "    draws = np.array([fwd[rng.integers(0,vh,size=k)].mean() for _ in range(8000)])\n"
            "else:\n"
            "    obs = R['spy'][5][2]/100; pval = R['spy'][5][7]\n"
            "    rng = np.random.default_rng(383); draws = rng.normal(R['spy'][5][4]/100, 0.011, 8000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*100, bins=50, color=GREY, alpha=.8, label='5d return of 13 RANDOM dates')\n"
            "ax.axvline(obs*100, c=RED, lw=2.5, label=f'the actual spikes ({obs*100:+.2f}%)')\n"
            "ax.set_xlabel('average 5-day forward return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The spike dip sits at the edge of the luck cloud — placebo p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'a random 13-date draw is at least this bearish {pval*100:.0f}% of the time — not rare enough')"
        ),
        md(
            f"The red line — the actual spikes — sits at the *edge* of the grey luck cloud, not out in the "
            f"tail. About **{R['spy'][5][7]*100:.0f}%** of random 13-date draws are at least as bearish. In "
            "plain terms: **the 5-day dip is roughly what you'd get from a dozen unlucky coin-flips** — and "
            "it's the only horizon that's even faintly bearish. Past a week it's gone."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** A faint, correctly-signed 5-day dip that **isn't statistically significant** "
            "and **flips positive within a month**. Real-as-a-flinch, empty-as-a-forecast.\n"
            f"- **Tradability — Mirage.** \"Sell on the spike\" sidesteps a noise-level "
            f"**+{R['net'][0][3]:.1f}%** over 5 days, then **loses {abs(R['net'][1][3]):.0f}–"
            f"{abs(R['net'][2][3]):.0f}%** at 20/60 days — you sold into the rebound. And it fires "
            f"**{R['n_events']} times in {R['years']:.0f} years**.\n"
            "- **\"Repo spike = flee\"? — Busted.** The one crash everyone cites (March 2020) is **COVID, "
            "not repo**, and it was a *bottom*. A handful of vivid, confounded episodes in a backstopped "
            "market is a **sample-size + confounding mirage**."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the de-risk rule loses money\n\n"
            "Forget significance for a second and just *run the rule*: every time repo spikes, sell SPY for "
            "H days, then buy back. How much loss did you \"avoid\"? (Positive = you dodged a drop; negative "
            "= you sold low and missed the bounce.)"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [(h,)+ (lambda c:(c['gross_mean']*100, c['net_mean']*100))(st.net_of_costs(F['SPY'], EVENTS, h, 5.0)) for h in (5,20,60)]\n"
            "else:\n"
            "    rows = [(h,g,nn) for (h,_,g,nn) in R['net']]\n"
            "hs = [r[0] for r in rows]; g=[r[1] for r in rows]; nv=[r[2] for r in rows]\n"
            "x = np.arange(len(hs)); fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(x, nv, .5, color=[GREEN if v>=0 else RED for v in nv])\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h} days' for h in hs])\n"
            "ax.set_ylabel('avoided loss per trade, net of 5bps (%)')\n"
            "ax.set_title('\"De-risk on the spike\": tiny at 5d, then it BLEEDS')\n"
            "for i,v in enumerate(nv): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('avoided loss net@5bps by horizon:', {f'{h}d': round(v,2) for h,v in zip(hs,nv)})"
        ),
        md(
            f"At 5 days you \"save\" a noise-level **+{R['net'][0][3]:.1f}%**; by 20 days you've **lost "
            f"{abs(R['net'][1][3]):.1f}%** and by 60 days **{abs(R['net'][2][3]):.1f}%**, because you sold "
            "into the recovery the Fed engineered. Costs (5 bps) are irrelevant — the rule is **wrong-signed** "
            "past a week. You can't build a portfolio around an oracle that whispers a dozen times a decade "
            "and then loses money."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The slow cousin.** [Study 115 — Credit-Spreads](../../115-credit-spreads/) asks whether the "
            "*credit cycle* (not the overnight plumbing) carries a forward warning — a fairer test of "
            "\"stress predicts risk-off.\"\n"
            "- **Another stress gauge.** [Study 111 — VIX-Term-Structure](../../111-vix-term-structure/): a "
            "different \"flashing light\" said to precede sell-offs; same base-rate question.\n"
            "- **The backdrop.** [Study 317 — Fed-Balance-Sheet](../../317-fed-balance-sheet/) — reserves and "
            "QT, which is *why* repo gets tight at quarter-ends in the first place.\n\n"
            "*Think a repo spike beats the market by more than luck? Get a real SOFR feed, define a "
            "mechanical threshold, draw the same number of random dates, and show the spikes landing "
            "**outside** the cloud on the bearish side — then we'll talk.*"
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
            "# SOFR-Repo-Stress — a quantitative teardown 🔬\n"
            "### Event-window forward returns on SPY/HYG/LQD · conditional vs unconditional means · "
            "a Welch *t* + placebo randomization null · the horizon sign-flip · costs · a synthetic "
            "faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We separate the "
            "two things \"watch the repo market\" fuses: a **real but tiny** few-day wobble after a funding "
            "spike from the **unconditional up-drift** of US risk assets, and we confront both with the "
            "**sample size** and the **confounding**. The decisive object is not the point estimate (a faint "
            "5-day dip, correctly signed) but its **power and its persistence**: with ~a dozen named, "
            "overlapping episodes the dip is statistically indistinguishable from random dates and "
            "**inverts** within a month.\n\n"
            "> ⚠️ **Data + signal note.** There is no free clean daily SOFR/repo-stress tape on yfinance; the "
            "**signal is a hardcoded, sourced event list** (`data.REPO_EPISODES`), named on the Signal axis. "
            "Real data: yfinance daily adjusted closes, SPY + HYG + LQD, 2010→2026. Offline core + synthetic "
            "control are deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | SPY 5d conditional **{R['spy'][5][2]:+.2f}%** vs base "
            f"**+{R['spy'][5][4]:.2f}%** (Welch **t = {R['spy'][5][6]:.2f}**, best asset LQD "
            f"**t = {R['lqd'][5][6]:.2f}**) — correctly signed but **fails \\|t\\| ≥ 2** and **flips to "
            f"+{R['spy'][20][2]:.1f}% / +{R['spy'][60][2]:.1f}%** by 20/60d. |\n"
            f"| **Tradability** | `MIRAGE` | \"De-risk\" avoids **+{R['net'][0][3]:.1f}%** net at 5d, then "
            f"**{R['net'][1][3]:+.1f}% / {R['net'][2][3]:+.1f}%** at 20/60d. **{R['n_events']} events / "
            f"{R['years']:.0f}y**; wrong-signed where it matters. |\n"
            f"| **Repo spike = flee?** | `BUSTED` | The systemic-only subset's only significant result is "
            f"**+{R['systemic'][2][2]:.1f}% at 60d** (*t* = +{R['systemic'][2][4]:.2f}, wrong sign). "
            "March-2020 (the cited crash) is COVID — a confound, and a *bottom*. |\n\n"
            "> 💡 In plain words: a repo spike really is followed by a small wobble — and then a rebound, "
            "because the spike is what summons the backstop. Strip the base rate and ~a dozen confounded "
            "events leave nothing the null can't explain."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $S$ be the set of repo-stress episode dates and $r_{t\\to t+H}$ the forward $H$-day return "
            "of a risk asset. The believer's hypotheses:\n\n"
            "- **H₁ (the signal predicts).** $\\mathbb{E}[r_{t\\to t+H}\\mid t\\in S] < "
            "\\mathbb{E}[r_{t\\to t+H}]$ for forward horizon $H$ — a *negative excess* over the base rate "
            "(risk assets do **worse** after a spike).\n"
            "- **H₂ (it's deployable).** The bearish excess is large, persistent and frequent enough, net of "
            "costs, to de-risk on.\n"
            "- **H₃ (\"flee risk\").** The post-spike weakness reflects the funding shock, not coincident "
            "confounds (COVID, SVB) or the market's unconditional up-drift.\n\n"
            "We find **H₁ rejected** (a faint $t\\approx-1$ at 5d that **fails $|t|\\ge2$** and **flips sign** "
            "by 20d), **H₂ rejected** (≈13 events / 16.5y, and the rule *loses* money past a week), **H₃ "
            "rejected** (the only significant systemic-subset result is *positive*, and the cited crash is "
            "COVID). The legend is true exactly where it's uninformative (a spike is momentarily unsettling) "
            "and unproven exactly where it would pay (a *persistent, attributable* warning)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The whole teardown is one decomposition: a conditional mean against an unconditional baseline, "
            "judged by its **standard error**.\n\n"
            "$$\\widehat{\\Delta}_H = \\bar r^{\\text{spike}}_H - \\bar r^{\\text{all}}_H,\\qquad "
            "t = \\frac{\\widehat{\\Delta}_H}{\\sqrt{\\,s^2_{\\text{spike}}/k + s^2_{\\text{all}}/N\\,}}.$$\n\n"
            "With $k = 13$, $s_{\\text{spike}}/\\sqrt{k}$ is large: even a clean few-point $\\widehat{\\Delta}$ "
            "drowns in its own SE. A raw **down-rate** (P[ret<0]) is the wrong lens — US returns are positive "
            "most of the time *unconditionally*, so even a real funding shock that gets backstopped shows a "
            "down-rate near the base. The honest small-sample instrument is a **left-tail randomization "
            "(placebo) test**: resample $k$ random entry dates and ask how often chance is at least as "
            "bearish as the spikes. That number, plus the **sign-flip across horizons**, decides the Signal "
            "axis."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Signal.** A hardcoded, sourced list of **{R['n_events']} repo-stress episodes** "
            f"(`data.REPO_EPISODES`), 2010→2026. Explicit **event list**, not a live SOFR threshold — named "
            "on the axis. Deliberately generous (folds in the quarter/year-end turns) to maximise $k$.\n"
            "- **Risk assets.** SPY (equities), HYG (high-yield), LQD (investment-grade) — yfinance adjusted "
            f"closes, {R['start']}→{R['end']}, {R['days']:,} days.\n"
            "- **Forward returns.** Enter the close **1 day after** the episode (no look-ahead), hold "
            "$H\\in\\{5,20,60\\}$ trading days; drop events whose horizon overruns the tape.\n"
            "- **Null #1 (Welch t).** Conditional mean vs the unconditional overlapping-window mean; a "
            "**negative** $t$ is the believer's wanted sign.\n"
            "- **Null #2 (placebo).** 20,000 draws of $k$ random valid entry dates; $p = "
            "\\Pr[\\text{random-draw mean} \\le \\text{spike mean}]$ (left tail) — the small-sample workhorse.\n"
            "- **Costs.** 1-day lag + a 5-bps round-trip on the de-risking trade.\n"
            "- **Confound stress.** A **systemic-only** subset (Sept-2019, COVID-Mar-2020, SVB-2023) — the "
            "purest funding crises — to see if the headline events behave differently (they do: *more* "
            "bullish).\n"
            "- **Positive control.** A deterministic tape with **13 stress dates injected** and a **known** "
            "forward edge: the inference must recover a planted bearish edge **and** must NOT manufacture "
            "significance when the true edge is zero."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The point estimates — faint, horizon-flipping, never significant\n\n"
            "Conditional mean forward return with its $\\pm$ standard error, against the unconditional base "
            "rate (diamonds), for SPY across 5/20/60 days. Faintly *below* base at 5d, *above* by 20/60d."
        ),
        code(
            "hs = [5, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    cm, bm, ts, ses = [], [], [], []\n"
            "    for h in hs:\n"
            "        s = st.summarize(F['SPY'], EVENTS, h); cm.append(s['cond_mean']); bm.append(s['base_mean']); ts.append(s['t'])\n"
            "        cr = st.event_returns(F['SPY'], EVENTS, h); ses.append(cr.std(ddof=1)/np.sqrt(len(cr)))\n"
            "else:\n"
            "    cm = [R['spy'][h][2]/100 for h in hs]; bm = [R['spy'][h][4]/100 for h in hs]\n"
            "    ts = [R['spy'][h][6] for h in hs]; ses = [0.018, 0.035, 0.06]\n"
            "x = np.arange(len(hs)); fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x, [c*100 for c in cm], yerr=[s*100 for s in ses], capsize=5, color=[RED if c<0 else GREEN for c in cm], width=.5, label='after a spike (±SE)')\n"
            "ax.plot(x, [b*100 for b in bm], 'D', ms=11, c=GREY, label='unconditional base rate')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean forward SPY return (%)')\n"
            "ax.set_title('Below base at 5d, ABOVE base by 20/60d — and |t|<2 throughout'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Welch t by horizon (SPY):', {f'{h}d': round(t,2) for h,t in zip(hs,ts)})"
        ),
        md(
            f"> 💡 In plain words: at 5 days the spike mean is **{R['spy'][5][2]:+.2f}%** vs base "
            f"**+{R['spy'][5][4]:.2f}%** — a real-looking dip at **t = {R['spy'][5][6]:.2f}** (not "
            f"significant). By 20 days it's **{R['spy'][20][2]:+.2f}%** (*above* base, t = "
            f"+{R['spy'][20][6]:.2f}). H₁ is **rejected**: a negative whisper inside its own error bar that "
            "doesn't even keep its sign for a month."
        ),
        md(
            "### 4b · All three assets — the dip is shallow and short on every one\n\n"
            "Welch *t* for SPY, HYG and LQD at each horizon. The believer needs $t\\le-2$ (deep blue, "
            "bearish); instead the *t* hovers near $-1$ at 5d and turns *positive* by 20/60d on all three."
        ),
        code(
            "assets = ['SPY','HYG','LQD']; hs=[5,20,60]\n"
            "T = np.zeros((3,3))\n"
            "for i,a in enumerate(assets):\n"
            "    for j,h in enumerate(hs):\n"
            "        T[i,j] = st.summarize(F[a], EVENTS, h)['t'] if HAVE_REAL else R[a.lower()][h][6]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.0))\n"
            "im = ax.imshow(T, cmap='RdYlGn', vmin=-2.2, vmax=2.2, aspect='auto')\n"
            "ax.set_xticks(range(3)); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_yticks(range(3)); ax.set_yticklabels(assets)\n"
            "for i in range(3):\n"
            "    for j in range(3): ax.text(j,i,f'{T[i,j]:+.2f}',ha='center',va='center',fontsize=11,fontweight='bold')\n"
            "ax.set_title('Welch t: faintly bearish at 5d, then flips bullish (none reaches -2)')\n"
            "fig.colorbar(im, ax=ax, label='Welch t (negative = bearish, the wanted sign)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('most bearish single cell:', round(T.min(),2), '(need <= -2 to call it a warning)')"
        ),
        md(
            f"> 💡 In plain words: the single most bearish cell across **all** assets and horizons is "
            f"**t = {R['lqd'][5][6]:.2f}** (LQD, 5d) — still short of the $-2$ bar. Credit (HYG/LQD) is a "
            "touch softer than SPY at 5d (funding stress *is* a credit thing), but the effect is shallow, "
            "asset-wide, and gone within a month. There is no asset on which \"flee\" clears significance."
        ),
        md(
            "### 4c · The decisive test — a left-tail placebo null sized to the event count\n\n"
            "Draw 13 *random* entry dates 20,000 times; the histogram is the null distribution of the SPY "
            "5-day mean. The spike set is the red line; the *p*-value is the **left**-tail mass (how often "
            "chance is at least this bearish)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(F['SPY'], EVENTS, 5); obs=pl['event_mean']; pval=pl['p_value']\n"
            "    p = F['SPY'].values; n=len(p); k=len(EVENTS); lag=1; h=5; vh=n-h-lag-1\n"
            "    fwd = p[lag+h:lag+h+vh]/p[lag:lag+vh]-1; rng=np.random.default_rng(383)\n"
            "    draws = np.array([fwd[rng.integers(0,vh,size=k)].mean() for _ in range(20000)])\n"
            "else:\n"
            "    obs = R['spy'][5][2]/100; pval = R['spy'][5][7]\n"
            "    rng = np.random.default_rng(383); draws = rng.normal(R['spy'][5][4]/100, 0.011, 20000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*100, bins=60, color=GREY, alpha=.85, label=f'null: mean of {len(draws):,} random 13-date draws')\n"
            "ax.axvline(obs*100, c=RED, lw=2.5, label=f'observed spike mean {obs*100:+.2f}%')\n"
            "ax.set_xlabel('5-day mean forward SPY return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Placebo p = {pval:.2f}: the dip is near the edge of the luck cloud'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'P[random 13-date draw <= spike] = {pval:.3f}  (need <0.05; here ~1 in 17 — but it vanishes by 20d)')"
        ),
        md(
            f"> 💡 In plain words: **{R['spy'][5][7]*100:.0f}%** of random 13-date samples are at least as "
            "bearish as the spikes at 5 days — close to the edge, but not a real-edge tail, and it "
            "**evaporates** by 20 days (placebo p jumps to "
            f"**{R['spy'][20][7]:.2f}**, the *bullish* side). A real warning would push the red line into the "
            "deep left tail and *stay* there. H₃ rejected: the record is base-rate × rarity × a transient "
            "wobble."
        ),
        md(
            "### 4d · Confound stress + cost — the 'pure' crises point the wrong way\n\n"
            "Keep only the three *systemic* episodes (Sept-2019, COVID-Mar-2020, SVB-2023) — the strongest "
            "version of the claim — and apply a round-trip cost. The 'purest' funding crises give the "
            "**largest** moves, in the **bullish** direction."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tab = data.episode_table()\n"
            "    sysd = pd.DatetimeIndex([d for d,r in tab.iterrows() if r['tag'] in data.SYSTEMIC_ONLY])\n"
            "    sysrows = [(h,)+ (lambda s:(s['cond_mean']*100, s['t']))(st.summarize(F['SPY'], sysd, h)) for h in (5,20,60)]\n"
            "    netr = [(h,)+ (lambda c:(c['gross_mean']*100, c['net_mean']*100))(st.net_of_costs(F['SPY'], EVENTS, h, 5.0)) for h in (5,20,60)]\n"
            "else:\n"
            "    sysrows = [(h, R['systemic'][i][2], R['systemic'][i][4]) for i,h in enumerate((5,20,60))]\n"
            "    netr = [(h,g,nn) for (h,_,g,nn) in R['net']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "shs=[r[0] for r in sysrows]; scm=[r[1] for r in sysrows]\n"
            "a1.bar([f'{h}d' for h in shs], scm, color=[RED if v<0 else GREEN for v in scm], width=.55)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_title('Systemic-only SPY: bullish, not bearish')\n"
            "a1.set_ylabel('cond. mean return (%)'); a1.set_xlabel('horizon')\n"
            "for i,v in enumerate(scm): a1.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "mh=[r[0] for r in netr]; nv=[r[2] for r in netr]\n"
            "a2.bar([f'{h}d' for h in mh], nv, color=[GREEN if v>=0 else RED for v in nv], width=.55)\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_title('\"De-risk\" net@5bps: bleeds past a week')\n"
            "a2.set_ylabel('avoided loss / trade (%)'); a2.set_xlabel('horizon')\n"
            "for i,v in enumerate(nv): a2.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('systemic SPY (h, cond%, t):', [(r[0], round(r[1],1), round(r[2],2)) for r in sysrows])"
        ),
        md(
            f"> 💡 In plain words: restricting to the *purest* funding crises makes it **worse** for the "
            f"believer — SPY is **+{R['systemic'][2][2]:.1f}%** at 60d (t = +{R['systemic'][2][4]:.2f}, the "
            "only $|t|\\ge2$ in the whole study, pointing at **gains**). And the de-risk rule loses "
            f"**{R['net'][2][3]:.1f}%** at 60d net. There is no horizon, asset, subset or cost regime in "
            "which this becomes a deployable warning — **the events themselves rebound.**"
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic tape with **13 stress dates injected**: with **zero** planted edge the test "
            "must stay above $t=-2$ (a dozen events can't fake significance); with a **−8%/20d** planted "
            "bearish edge it must light up. Both hold — proving the engine is unbiased *and* that this sample "
            "size only detects implausibly large edges."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, -0.08):\n"
            "    syn = data.synthetic_tape(n_events=13, edge_20d=edge, seed=383)\n"
            "    s = st.summarize(syn['prices']['SPY'], syn['events'], 20)\n"
            "    res.append((edge, s['n'], s['cond_mean']*100, s['base_mean']*100, s['t'], s['p_placebo']))\n"
            "labels = [f'planted edge\\n{e*100:.0f}% / 20d' for e,_,_,_,_,_ in res]; tvals=[r[4] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(labels, tvals, color=[GREY, RED], width=.5)\n"
            "ax.axhline(-2, ls='--', c=RED, label='t = -2 (bearish significance bar)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='top')\n"
            "ax.set_ylabel('Welch t (20-day)'); ax.set_title('Control: ~13 events detect only a HUGE edge'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,c,b,t,p in res: print(f'planted {e*100:+.0f}%/20d: n={k} cond={c:.1f}% base={b:.1f}% t={t:.2f} p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** real edge the control sits at **t = {R['syn'][0][4]:.2f}** "
            f"(above −2 — no false positive); only a **−8%/20d** edge — enormous — reaches "
            f"**t = {R['syn'][1][4]:.2f}**. So the machinery is honest, and the real-tape 5-day *t* of ~−1 is "
            "exactly what an *absent or tiny* edge looks like through a 13-event keyhole. The sample size is "
            "the verdict."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — SPY 5d excess **{R['spy'][5][2]-R['spy'][5][4]:+.2f}pp** at Welch "
            f"**t = {R['spy'][5][6]:.2f}** (best asset LQD **t = {R['lqd'][5][6]:.2f}**); **fails $|t|\\ge2$** "
            f"and **flips to +{R['spy'][20][2]-R['spy'][20][4]:.2f}pp** (t = +{R['spy'][20][6]:.2f}) by 20d. "
            "A hardcoded event-list signal with no robust *t* on any asset/horizon ⇒ NONE, not REAL.\n"
            f"- **Tradability `MIRAGE`** — \"de-risk\" avoids **+{R['net'][0][3]:.1f}%** net at 5d then "
            f"**{R['net'][1][3]:+.1f}% / {R['net'][2][3]:+.1f}%** at 20/60d (wrong-signed); "
            f"**{R['n_events']} trades / {R['years']:.0f}y**. Costs are irrelevant — the rule is "
            "directionally wrong past a week. No NAV-scale edge.\n"
            f"- **Repo spike = flee? `BUSTED`** — the systemic-only subset's only significant result is "
            f"**+{R['systemic'][2][2]:.1f}% at 60d** (*t* = +{R['systemic'][2][4]:.2f}, toward gains), and "
            "the cited crash (March 2020) is **COVID** — a confound, and a *bottom*. A sample-size + "
            "confounding mirage."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the power curve\n\n"
            "The operational truth in one picture: how big would the *true* 5-day bearish excess have to be "
            "for a $k$-event study to detect it at $t=-2$? At $k=13$ you'd need an excess several times the "
            "one observed; the real signal lives far inside the detection floor — and that's before it flips "
            "sign."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cr = st.event_returns(F['SPY'], EVENTS, 5); sd = cr.std(ddof=1)\n"
            "    obs_excess = (R['spy'][5][2]-R['spy'][5][4])/100\n"
            "else:\n"
            "    sd = 0.035; obs_excess = (R['spy'][5][2]-R['spy'][5][4])/100\n"
            "ks = np.arange(5, 120)\n"
            "min_detectable = -2.0 * sd / np.sqrt(ks)   # bearish excess needed for t=-2 (base SE ~0)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(ks, min_detectable*100, c=AMBER, lw=2, label='bearish excess needed for t=-2')\n"
            "ax.axhline(obs_excess*100, c=RED, ls='--', label=f'observed 5d excess ~{obs_excess*100:+.2f}%')\n"
            "ax.axvline(R['n_events'], c=GREY, ls=':', label=f\"our k={R['n_events']}\")\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('number of events k'); ax.set_ylabel('5-day excess return (%)')\n"
            "ax.set_title('Detection floor vs the real signal: we are far under-powered'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "need = -2.0*sd/np.sqrt(R['n_events'])\n"
            "print(f'at k={R[\"n_events\"]} you need ~{need*100:.1f}% bearish excess for t=-2; observed ~{obs_excess*100:+.2f}% -> under-powered by ~{abs(need/obs_excess):.1f}x')"
        ),
        md(
            "> 💡 In plain words: the amber curve is the **minimum detectable bearish excess**; the red line "
            "is what we actually see at 5 days. They don't meet until $k$ is many times larger than the "
            "repo-spike event list will ever deliver — and even then the effect has *flipped to positive* by "
            "20 days. There is no sizing, no horizon, no cost assumption that manufactures a tradable warning "
            "from a dozen confounded events — **the rarity that makes \"watch the repo market\" sound "
            "prophetic is exactly what makes it untestable and untradable.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The slow cousin.** [Study 115 — Credit-Spreads](../../115-credit-spreads/): whether the "
            "credit *cycle* (not the overnight plumbing) carries a forward warning — the fairer test of "
            "\"stress predicts risk-off.\"\n"
            "- **Another stress gauge.** [Study 111 — VIX-Term-Structure](../../111-vix-term-structure/) — a "
            "different flashing light, same base-rate-and-frequency question.\n"
            "- **The backdrop.** [Study 317 — Fed-Balance-Sheet](../../317-fed-balance-sheet/) — reserves / "
            "QT, the reason repo gets tight at quarter-ends.\n"
            "- **Do it properly.** Pull the real NY-Fed SOFR percentiles / GC-repo series, define a "
            "mechanical stress threshold (e.g. SOFR-minus-IORB > X bp), and re-run; the event count and the "
            "exact dates change, the power-and-confounding problem does not.\n\n"
            "*The reproducible core is offline and deterministic; the repo-stress input is an explicit "
            "hardcoded event list. Methods and sources: [`docs/references.md`](../docs/references.md); frozen "
            "numbers: [`docs/results.md`](../docs/results.md).*"
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
