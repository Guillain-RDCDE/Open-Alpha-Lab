"""Generate the two narrative notebooks for Study 606 (OPEC Announcement Effect).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached CL=F /
BZ=F / USO OHLC frames under ../_cache/ and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md). The synthetic machinery control runs anywhere.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (107 OPEC/OPEC+
# ministerial decision days 2000-2026 x {CL=F, BZ=F, USO}, as-of 2026-06-30).
R = dict(
    as_of="2026-06-30", n_meetings=107, first="2000-03-28", last="2026-06-07",
    fp_cl="1a2103f8c32a", fp_bz="7e9543880619", fp_uso="6976f7deb0d1",
    # vol: (n_ev, abs_ev_bps, abs_base_bps, mult, ci_lo, ci_hi, welch_t, bf_t,
    #       var_ratio, p_placebo, rng_mult, rng_t)
    vol=dict(
        CL=(105, 248.3, 177.1, 1.40, 1.15, 1.67, 3.07, 2.84, 1.23, 0.018, 1.45, 4.80),
        BZ=(68, 242.2, 158.0, 1.53, 1.21, 1.89, 3.01, 2.57, 2.12, 0.000, 1.52, 4.23),
        USO=(73, 242.1, 159.8, 1.52, 1.20, 1.85, 3.17, 3.09, 2.15, 0.000, 1.47, 3.64),
    ),
    # drift: {tape: [(k, cum_bps, hac_t), ...]}
    drift=dict(
        CL=[(0, -11.2, -0.30), (1, -50.3, -0.92), (2, -20.5, -0.16),
            (3, 16.6, 0.51), (5, -215.4, -0.89)],
        BZ=[(0, -0.1, -0.07), (1, -65.2, -1.15), (2, -47.2, -0.70),
            (3, -35.9, -0.44), (5, -50.0, -0.63)],
        USO=[(0, -14.2, -0.36), (1, -57.3, -0.90), (2, -39.3, -0.46),
             (3, -37.7, -0.34), (5, -31.5, -0.16)],
    ),
    # continuation: (n, gross_bps, t, hit_pct, ex2020_bps, ex2020_t, pre16_t,
    #                post16_t, lag_t)
    cont=dict(
        CL=(105, 365.0, 1.40, 61, 67.2, 1.29, 0.95, 1.27, 1.40),
        BZ=(68, 136.2, 1.69, 60, 111.5, 1.62, 0.87, 1.45, 0.58),
        USO=(73, 158.1, 2.20, 56, 70.7, 1.19, 0.68, 2.25, 1.91),
    ),
    cl_net=[(2.0, 361.0), (5.0, 355.0), (10.0, 345.0)],
    # synthetic: (label, mult, welch_t, p, day0_drift_bps, hac_t)
    syn=[("null (x1)", 0.96, -0.51, 0.688, -6.3, -0.22),
         ("vol x2 planted", 1.92, 6.12, 0.000, -12.6, -0.26),
         ("x2 + drift +150", 2.01, 6.33, 0.000, 137.4, 3.22)],
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Vol_doubles%3F: Busted](https://img.shields.io/badge/Vol_doubles%3F-Busted-8b949e?style=flat-square)\n\n"
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

from opec_announcement_effect import data, strategy as st

AS_OF = "2026-06-30"
DATES = data.meeting_dates()
HAVE_REAL = data.have_real()
TAPES = data.load_real(asof=AS_OF) if HAVE_REAL else {}
print("real tapes cached:", HAVE_REAL, "| meetings:", len(DATES),
      f"({DATES.min().date()} -> {DATES.max().date()})")
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"

TK_LABEL = {"CL=F": "CL", "BZ=F": "BZ", "USO": "USO"}


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do OPEC meetings really move oil? 🛢️\n"
            "### The announcement-day folklore — louder, yes; tradable, no — in plain English\n\n"
            + BADGES +
            "Every few months, oil ministers file into a room in Vienna (or a videoconference), "
            "argue about production quotas, and the world's energy desks hold their breath. The "
            "folklore says two things about those days: **volatility doubles**, and the move after "
            "the decision **keeps going** — so you can \"trade the announcement.\"\n\n"
            "We froze the *entire* calendar of OPEC and OPEC+ ministerial decision days since 2000 "
            "— **107 meetings**, from the March 2000 quota raise to the June 2026 OPEC+ meeting — "
            "and put both halves of the folklore against three oil tapes: WTI futures, Brent "
            "futures, and the USO ETF.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the bootstrap CIs and the HAC "
            "regressions? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Are OPEC decision days louder than normal days? | **Yes — really.** About "
            "**1.4–1.5× louder** on all three tapes, and that's statistically solid, not luck. |\n"
            "| Does volatility *double*, like the folklore says? | **No.** ~1.5× is the honest "
            "number; a genuine doubling would have lit our tests up far harder (we checked, by "
            "planting one in a simulator). |\n"
            "| Does oil drift after the decision? | **No.** Up-days and down-days cancel almost "
            "perfectly — at every horizon from the decision day to a week later, the average move "
            "is statistically **zero**. |\n"
            "| Can you \"trade the announcement\" (ride the day-0 direction)? | **Not honestly.** "
            "The rule only looks good if the five wild meetings of 2020 are in the sample. Remove "
            "them and it's noise. |\n\n"
            "> The folklore is **half right**: OPEC days are genuinely louder. The half you could "
            "monetise — the doubling, the drift — isn't there."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"OPEC meeting days move oil. Vol doubles on decision day, and once the decision "
            "is out, the move keeps going — get on it.\"*\n\n"
            "It's plausible folklore. OPEC(+) controls a big slice of world production; its "
            "decisions are scheduled, public, and occasionally shocking (the no-cut of November "
            "2014, the price war of March 2020, the surprise 2-million-barrel cut of October "
            "2022). The academic literature agrees the days are *special* — but is vague on how "
            "special, and quietly split on whether anything is left to trade once the decision "
            "hits the tape."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two different products are being sold in one sentence. **\"Vol doubles\"** is a "
            "claim about *risk* — it tells you what options should cost and how wide your stops "
            "must be. **\"The drift is tradable\"** is a claim about *direction* — free money if "
            "true. A desk that conflates them buys expensive straddles *and* chases moves that "
            "don't continue. We test each on its own."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** All **{R['n_meetings']}** OPEC Conference + OPEC+ ministerial "
            f"decision days {R['first'][:4]}–{R['last'][:4]}, hardcoded from the OPEC "
            "press-release archive (weekend meetings roll to the next trading day).\n"
            "- **Louder?** Compare the size of the day-0 move (and the intraday high-low range) "
            "against ordinary days far from any meeting.\n"
            "- **Drift?** Average the *signed* move from the day before the decision to 1–5 days "
            "after. If announcements start trends, this is where they'd show.\n"
            "- **Tradable?** Follow the day-0 direction for a week, after realistic futures "
            "costs — and check the rule survives without 2020, the wildest year in oil history."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: are decision days louder?** Average |daily move| on OPEC decision days vs "
            "ordinary days, per tape."
        ),
        code(
            "labs = ['WTI (CL=F)', 'Brent (BZ=F)', 'USO ETF']\n"
            "keys = ['CL', 'BZ', 'USO']\n"
            "if HAVE_REAL:\n"
            "    rows = [st.vol_stats(TAPES[tk], DATES, n_boot=200, n_placebo=200)\n"
            "            for tk in ('CL=F', 'BZ=F', 'USO')]\n"
            "    ev = [r['abs_ev_bps'] for r in rows]; ba = [r['abs_base_bps'] for r in rows]\n"
            "else:\n"
            "    ev = [R['vol'][k][1] for k in keys]; ba = [R['vol'][k][2] for k in keys]\n"
            "x = np.arange(3); w = .36\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(x - w/2, ev, w, color=AMBER, label='OPEC decision day')\n"
            "ax.bar(x + w/2, ba, w, color=GREY, label='ordinary day')\n"
            "for i in range(3):\n"
            "    ax.annotate(f'{ev[i]/ba[i]:.2f}x', (i - w/2, ev[i]), ha='center', va='bottom', fontweight='bold')\n"
            "ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel('average |daily move| (bps)')\n"
            "ax.set_title('OPEC decision days ARE louder - by about half, not double')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('vol multiples:', [f'{e/b:.2f}x' for e, b in zip(ev, ba)])"
        ),
        md(
            f"Louder — genuinely. WTI moves **{R['vol']['CL'][1]:.0f} bps** on decision days vs "
            f"**{R['vol']['CL'][2]:.0f} bps** normally (**{R['vol']['CL'][3]:.2f}×**); Brent and "
            f"USO read **{R['vol']['BZ'][3]:.2f}×** and **{R['vol']['USO'][3]:.2f}×**. The quants "
            "notebook shows this clears the significance bar comfortably on all three tapes. But "
            "notice what it *isn't*: **2×**. The folklore's \"vol doubles\" is half again too "
            "generous."
        ),
        md(
            "**Second: is there a drift to ride?** The average *signed* move after the decision, "
            "from the day before to 0–5 days after. Loud days can still average to zero — if cuts "
            "rally and collapses crash, the mean nets out."
        ),
        code(
            "ks = [0, 1, 2, 3, 5]\n"
            "if HAVE_REAL:\n"
            "    drift = {lab: [d['mean_bps'] for d in st.drift_stats(TAPES[tk], DATES)]\n"
            "             for lab, tk in zip(keys, ('CL=F', 'BZ=F', 'USO'))}\n"
            "else:\n"
            "    drift = {k: [row[1] for row in R['drift'][k]] for k in keys}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "for k, c in zip(keys, [AMBER, GREY, GREEN]):\n"
            "    ax.plot(ks, drift[k], 'o-', color=c, label=k)\n"
            "ax.axhline(0, c=RED, lw=1.5)\n"
            "ax.set_xlabel('days after the decision'); ax.set_ylabel('average signed move (bps)')\n"
            "ax.set_title('The post-decision drift: statistically zero, everywhere')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print({k: [round(v, 1) for v in drift[k]] for k in keys})"
        ),
        md(
            "Wiggles around zero, never significant — the quants notebook runs the proper "
            "(autocorrelation-robust) test on all **15** horizon-×-tape combinations and every "
            "single one sits **far below the significance bar**. OPEC days are loud, but they are "
            "loud in *both directions*, and the directions cancel."
        ),
        md(
            "**Third: the \"trade the announcement\" rule.** Go with the day-0 direction, hold "
            "five days. Here's the trick the folklore plays — watch what happens when we remove "
            "the five meetings of 2020 (the price war and the historic 9.7 mb/d cut):"
        ),
        code(
            "if HAVE_REAL:\n"
            "    no20 = DATES[(DATES < '2020-01-01') | (DATES > '2020-12-31')]\n"
            "    g_all = [st.continuation_stats(TAPES[tk], DATES)['gross_bps'] for tk in ('CL=F','BZ=F','USO')]\n"
            "    g_ex  = [st.continuation_stats(TAPES[tk], no20)['gross_bps'] for tk in ('CL=F','BZ=F','USO')]\n"
            "else:\n"
            "    g_all = [R['cont'][k][1] for k in keys]; g_ex = [R['cont'][k][4] for k in keys]\n"
            "x = np.arange(3); w = .36\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(x - w/2, g_all, w, color=AMBER, label='all meetings')\n"
            "ax.bar(x + w/2, g_ex, w, color=GREY, label='without the 5 meetings of 2020')\n"
            "for i in range(3):\n"
            "    ax.annotate(f'{g_all[i]:+.0f}', (i - w/2, g_all[i]), ha='center', va='bottom')\n"
            "    ax.annotate(f'{g_ex[i]:+.0f}', (i + w/2, g_ex[i]), ha='center', va='bottom')\n"
            "ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel('avg gross per trade (bps)')\n"
            "ax.set_title('\"Trade the announcement\": mostly a 2020 story')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('all:', [round(v) for v in g_all], ' ex-2020:', [round(v) for v in g_ex])"
        ),
        md(
            f"On WTI the average \"edge\" falls from **+{R['cont']['CL'][1]:.0f}** to "
            f"**+{R['cont']['CL'][4]:.0f} bps** per trade without 2020 — and it was never "
            "statistically significant to begin with (*t* ≈ 1.4). The one cell that grazes "
            f"significance (USO, *t* = {R['cont']['USO'][2]:.2f}) collapses to "
            f"*t* = {R['cont']['USO'][5]:.2f} ex-2020. Costs aren't the problem — the *signal* "
            "is: a rule that needs five specific meetings from the wildest year in oil history "
            "is not a rule."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Mixed.** *Real on the vol:* decision days run **1.4–1.5× louder** on "
            "all three tapes, comfortably significant. *None on the drift:* the signed "
            "post-decision move is zero at every horizon.\n"
            "- **Tradability — Mirage.** No drift to harvest; the day-0-direction rule is a 2020 "
            "artifact; and the extra vol is exactly when option protection is priced up.\n"
            f"- **\"Vol doubles\"? — Busted.** **~1.5×**, measured with confidence intervals that "
            "exclude 2× on every tape. Louder, yes. Doubled, no."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Why no drift?** OPEC decisions are *scheduled* news — desks pre-position for "
            "weeks, and the surprise is absorbed within the session. What's left is variance, "
            "not direction. Compare [313-geopolitical-shock](../../313-geopolitical-shock/) — "
            "*unscheduled* oil shocks — and [226-crude-seasonality](../../226-crude-seasonality/) "
            "for the calendar-time cousin.\n"
            "- **The vol IS real information.** 1.5× isn't tradable as direction, but it prices "
            "straddles, sizes stops, and warns you off tight-stop positions into a meeting.\n"
            "- **Build your own.** The full 107-meeting table lives in "
            "[`data.py`](../opec_announcement_effect/data.py) with sources — swap in decision "
            "labels (cut/raise/hold) and test the asymmetry the literature reports.\n\n"
            "*Think the announcement is tradable with a finer clock? Show a rule that clears "
            "*t* = 2 on intraday data **without** 2020 in the sample — then we'll talk.*"
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
            "# The OPEC Announcement Effect — a quantitative teardown 🔬\n"
            "### 107 hardcoded ministerial decision days · vol multiple with bootstrap CI + "
            "placebo · Brown-Forsythe + Welch spread tests · HAC drift regressions over 15 "
            "horizon×tape cells · the continuation rule's 2020 decomposition · a planted-effect "
            "machinery control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim splits into a **variance** claim (\"vol doubles\") and a **mean** claim "
            "(\"the drift is tradable\") — we test each with the appropriate inference and let "
            "the tape grade them separately.\n\n"
            "> ⚠️ **Data note.** Event input = the frozen 107-meeting OPEC/OPEC+ table in "
            "[`data.py`](../opec_announcement_effect/data.py) (source-commented; JMMC/subgroup "
            "calls excluded by a pre-registered scope rule — conservative for the vol claim). "
            "Tapes: yfinance daily OHLC `CL=F` / `BZ=F` / `USO`, as-of **" + R["as_of"] + "** "
            "(fingerprints `" + R["fp_cl"] + "` / `" + R["fp_bz"] + "` / `" + R["fp_uso"] + "`). "
            "No survivorship — futures + a live ETF. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | *Real on the vol:* multiple **1.40–1.53×** across tapes, "
            f"Welch *t* = **+3.01 to +3.17**, range *t* to **+4.80**, placebo *p* ≤ 0.018. "
            f"*None on the drift:* all 15 horizon×tape HAC *t* ∈ (−1.2, +0.6). |\n"
            f"| **Tradability** | `MIRAGE` | Day-0-sign continuation: best cell USO "
            f"*t* = {R['cont']['USO'][2]:.2f} → **{R['cont']['USO'][5]:.2f} ex-2020**; CL "
            f"*t* = {R['cont']['CL'][2]:.2f} at every cost level (net +345 bps/event at 10 bps "
            "one-way — never significant). |\n"
            f"| **Vol doubles?** | `BUSTED` | 95% bootstrap CI on the multiple: CL "
            f"**{R['vol']['CL'][4]:.2f}–{R['vol']['CL'][5]:.2f}**, BZ "
            f"{R['vol']['BZ'][4]:.2f}–{R['vol']['BZ'][5]:.2f}, USO "
            f"{R['vol']['USO'][4]:.2f}–{R['vol']['USO'][5]:.2f} — **2.0 outside all three**; "
            "a planted ×2 world reads Welch *t* ≈ 6 in this harness. |\n\n"
            "> 💡 In plain words: the meeting days are genuinely loud (about half again as loud "
            "as normal), but nothing directional survives — and \"doubles\" fails its own number."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be the daily close-to-close return and $E$ the set of day-0 sessions "
            "(first tradable session at-or-after each decision date). The folklore asserts:\n\n"
            "- **H₁ (variance).** $\\mathbb{E}[|r_t| \\mid t \\in E] \\approx 2\\, "
            "\\mathbb{E}[|r_t| \\mid t \\notin E]$ — \"vol doubles.\"\n"
            "- **H₂ (mean).** Post-decision drift: $\\mathbb{E}[\\sum_{j=0}^{k} r_{e+j}] \\neq 0$ "
            "for k = 0..5.\n"
            "- **H₃ (continuation).** $\\mathrm{sign}(r_{e})$ predicts $\\sum_{j=1}^{5} r_{e+j}$ "
            "— the tradable version.\n\n"
            "We find **H₁ half-supported** (elevated ~1.5×, *t* ≥ 3 — but the CI excludes 2.0), "
            "**H₂ rejected** (all |HAC *t*| < 1.2), **H₃ rejected** (never robustly significant; "
            "the one borderline cell is carried by the five 2020 meetings)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the inference each claim needs\n\n"
            "**Variance claims** need spread-robust tests: oil returns are fat-tailed and "
            "vol-clustered, so we use Welch on |r|, a Brown-Forsythe-style test (Welch on "
            "absolute deviations from group medians), a variance ratio, a **block-bootstrap CI "
            "on the multiple** (events i.i.d. — months apart; baseline in circular blocks of 10) "
            "and a seeded 2,000-draw random-calendar placebo.\n\n"
            "**Mean claims** need HAC: event windows overlap their own autocorrelation, so the "
            "Signal-axis statistic is a Newey-West dummy regression of daily returns on the "
            "day-0..+k window (lags k+5), with the per-event one-sample *t* beside it.\n\n"
            "**Tradability** needs an execution convention and a stress test: enter at the day-0 "
            "settle (the decision lands intraday, hours earlier — documented), lagged variant at "
            "close(+1); costs 2/5/10 bps one-way; and the sample re-run **without 2020**."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Events.** {R['n_meetings']} decision days {R['first']} → {R['last']}: every "
            "OPEC Conference 2000-2016, the production-setting consultations (2001/2003/Doha "
            "2006), every ONOMM 2016-2026. Weekend meetings roll to the next session per asset; "
            "CL maps 105, BZ 68, USO 73 (tape starts differ).\n"
            "- **Baseline.** All sessions further than ±5 from any day 0 (the halo keeps "
            "pre-positioning and echo days out of \"normal\").\n"
            "- **Vol.** |r| and intraday range (H−L)/C₋₁, event vs baseline.\n"
            "- **Drift.** Cumulative close(−1)→close(+k), k ∈ {0,1,2,3,5}.\n"
            "- **Continuation.** sign(day-0) × fwd(+1..+5), gross and net.\n"
            "- **Machinery control.** A deterministic world with meetings every ~63 sessions and "
            "a planted vol multiple / drift — the null must stay quiet, the planted folklore "
            "must fire."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The vol multiple, its CI, and the placebo\n\n"
            "Mean |r| on day 0 vs baseline, per tape — with the full-draw bootstrap CI "
            "(5,000 draws) and the 2,000-draw random-calendar placebo."
        ),
        code(
            "keys = ['CL', 'BZ', 'USO']; tks = ['CL=F', 'BZ=F', 'USO']\n"
            "if HAVE_REAL:\n"
            "    V = {k: st.vol_stats(TAPES[tk], DATES, n_boot=5000, n_placebo=2000)\n"
            "         for k, tk in zip(keys, tks)}\n"
            "    rows = [(k, V[k]['n_events'], V[k]['vol_multiple'], V[k]['ci_lo'], V[k]['ci_hi'],\n"
            "             V[k]['welch_t_abs'], V[k]['bf_t'], V[k]['p_placebo']) for k in keys]\n"
            "else:\n"
            "    rows = [(k, R['vol'][k][0], R['vol'][k][3], R['vol'][k][4], R['vol'][k][5],\n"
            "             R['vol'][k][6], R['vol'][k][7], R['vol'][k][9]) for k in keys]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "y = np.arange(3)\n"
            "for i, (k, n, m, lo, hi, w, bf, p) in enumerate(rows):\n"
            "    ax.plot([lo, hi], [i, i], color=GREY, lw=3)\n"
            "    ax.plot(m, i, 'o', color=AMBER, ms=11, zorder=3)\n"
            "    ax.annotate(f'{m:.2f}x  (Welch t={w:+.2f}, p={p:.3f})', (m, i + .17), ha='center')\n"
            "ax.axvline(1.0, color=GREEN, lw=1.5, label='no effect (1.0x)')\n"
            "ax.axvline(2.0, color=RED, lw=1.5, ls='--', label='the folklore (2.0x)')\n"
            "ax.set_yticks(y); ax.set_yticklabels([f'{k} (n={r[1]})' for k, r in zip(keys, rows)])\n"
            "ax.set_xlabel('decision-day vol multiple (mean |r| ratio, 95% bootstrap CI)')\n"
            "ax.set_title('Real - and nowhere near 2x'); ax.legend(loc='lower right')\n"
            "plt.tight_layout(); plt.show()\n"
            "for k, n, m, lo, hi, w, bf, p in rows:\n"
            "    print(f'{k:4s} n={n:3d}  x{m:.2f} [{lo:.2f}-{hi:.2f}]  Welch t={w:+.2f}  BF t={bf:+.2f}  placebo p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the amber dots (the measured multiples, "
            f"**{R['vol']['CL'][3]:.2f}× / {R['vol']['BZ'][3]:.2f}× / {R['vol']['USO'][3]:.2f}×**) "
            "sit far right of 1.0 — the effect is real (*t* ≥ 3, placebo *p* ≤ 0.018 — that's the "
            "Real half of the Signal stamp) — and every grey CI bar **stops short of the red 2.0 "
            "line**. The intraday range agrees: 1.45–1.52× at Welch *t* = +3.64 to +4.80."
        ),
        md(
            "### 4b · Drift — 15 cells of nothing\n\n"
            "The Newey-West dummy-regression *t* for the cumulative drift close(−1)→close(+k), "
            "for every horizon and tape. The claim needs bars outside ±2."
        ),
        code(
            "ks = [0, 1, 2, 3, 5]\n"
            "if HAVE_REAL:\n"
            "    T = {k: [d['hac_t'] for d in st.drift_stats(TAPES[tk], DATES)]\n"
            "         for k, tk in zip(keys, tks)}\n"
            "else:\n"
            "    T = {k: [row[2] for row in R['drift'][k]] for k in keys}\n"
            "x = np.arange(5); w = .26\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "for i, (k, c) in enumerate(zip(keys, [AMBER, GREY, GREEN])):\n"
            "    ax.bar(x + (i - 1) * w, T[k], w, color=c, label=k)\n"
            "ax.axhline(2, ls='--', c=RED); ax.axhline(-2, ls='--', c=RED, label='|t| = 2 bar')\n"
            "ax.set_ylim(-2.6, 2.6); ax.set_xticks(x); ax.set_xticklabels([f'k={k}' for k in ks])\n"
            "ax.set_xlabel('horizon (sessions after day 0)'); ax.set_ylabel('HAC (Newey-West) t')\n"
            "ax.set_title('Signed post-decision drift: every cell inside the bars')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print({k: [round(t, 2) for t in T[k]] for k in keys})"
        ),
        md(
            "> 💡 In plain words: fifteen chances to find a drift, fifteen bars comfortably "
            "inside ±2 (the extremes are −1.15 and +0.51). The loud day 0 is **two-sided** — "
            "cuts rally, collapses crash — and the signed mean nets to zero. This is the None "
            "half of the Signal stamp."
        ),
        md(
            "### 4c · Continuation — the 2020 decomposition\n\n"
            "sign(day-0) held to close(+5), gross per event, full sample vs ex-2020 vs the "
            "sub-periods. One borderline cell, and it doesn't survive."
        ),
        code(
            "if HAVE_REAL:\n"
            "    no20 = DATES[(DATES < '2020-01-01') | (DATES > '2020-12-31')]\n"
            "    C = {k: (st.continuation_stats(TAPES[tk], DATES),\n"
            "             st.continuation_stats(TAPES[tk], no20)) for k, tk in zip(keys, tks)}\n"
            "    t_all = [C[k][0]['t'] for k in keys]; t_ex = [C[k][1]['t'] for k in keys]\n"
            "else:\n"
            "    t_all = [R['cont'][k][2] for k in keys]; t_ex = [R['cont'][k][5] for k in keys]\n"
            "x = np.arange(3); w = .36\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.bar(x - w/2, t_all, w, color=AMBER, label='all meetings')\n"
            "ax.bar(x + w/2, t_ex, w, color=GREY, label='ex-2020 (5 meetings removed)')\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i in range(3):\n"
            "    ax.annotate(f'{t_all[i]:.2f}', (i - w/2, t_all[i]), ha='center', va='bottom')\n"
            "    ax.annotate(f'{t_ex[i]:.2f}', (i + w/2, t_ex[i]), ha='center', va='bottom')\n"
            "ax.set_xticks(x); ax.set_xticklabels(keys); ax.set_ylabel('one-sample t of sign(day0) x fwd(+1..+5)')\n"
            "ax.set_ylim(0, 2.7); ax.set_title('\"Trade the announcement\": the bar is never cleared robustly')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('t all:', [round(t, 2) for t in t_all], '  t ex-2020:', [round(t, 2) for t in t_ex])"
        ),
        md(
            f"> 💡 In plain words: only USO ever grazes the bar (*t* = "
            f"{R['cont']['USO'][2]:.2f}) — and removing the five 2020 meetings drops it to "
            f"**{R['cont']['USO'][5]:.2f}**. Sub-periods: pre-2016 *t* = "
            f"{R['cont']['CL'][6]:.2f}/{R['cont']['BZ'][6]:.2f}/{R['cont']['USO'][6]:.2f}, "
            f"2016+ *t* = {R['cont']['CL'][7]:.2f}/{R['cont']['BZ'][7]:.2f}/"
            f"{R['cont']['USO'][7]:.2f}. The lagged (close(+1)) entry reads *t* = "
            f"{R['cont']['CL'][8]:.2f}/{R['cont']['BZ'][8]:.2f}/{R['cont']['USO'][8]:.2f}. "
            "Hit rates 56–61% — the kind that vanishes with five outliers.\n\n"
            "**Costs are not the story.** CL gross +365.0 bps/event nets "
            "+361.0 / +355.0 / +345.0 at 2/5/10 bps one-way (~4 trades/yr, trivial capacity "
            "for retail) — the *t* never moves. The rule fails on **significance**, not "
            "friction — which is exactly what makes it a Mirage rather than Fragile."
        ),
        md(
            "### 4d · Machinery control — a world where the folklore is TRUE\n\n"
            "Deterministic tape, meetings every ~63 sessions. Null (×1, no drift) must stay "
            "quiet; the folklore planted at full strength (vol ×2) and a planted +150 bps day-0 "
            "drift must light up. *(A machinery proof — never cited in support of a stamp.)*"
        ),
        code(
            "res = []\n"
            "for lab, vm, dr in [('null (x1)', 1.0, 0.0), ('vol x2 planted', 2.0, 0.0),\n"
            "                    ('x2 + drift +150', 2.0, 0.015)]:\n"
            "    px, dd = data.synthetic_world(vol_mult=vm, drift=dr, seed=606)\n"
            "    v = st.vol_stats(px, dd, n_boot=500, n_placebo=300)\n"
            "    d0 = st.drift_stats(px, dd, horizons=(0,))[0]\n"
            "    res.append((lab, v['vol_multiple'], v['welch_t_abs'], d0['hac_t']))\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "labs = [r[0] for r in res]\n"
            "a1.bar(labs, [r[2] for r in res], color=[GREY, GREEN, GREEN], width=.55)\n"
            "a1.axhline(2, ls='--', c=RED); a1.set_title('vol detector (Welch t on |r|)')\n"
            "for i, r in enumerate(res): a1.annotate(f'x{r[1]:.2f}\\nt={r[2]:.1f}', (i, max(r[2], 0)), ha='center', va='bottom', fontsize=9)\n"
            "a2.bar(labs, [r[3] for r in res], color=[GREY, GREY, GREEN], width=.55)\n"
            "a2.axhline(2, ls='--', c=RED); a2.set_title('drift detector (day-0 HAC t)')\n"
            "for i, r in enumerate(res): a2.annotate(f't={r[3]:.2f}', (i, max(r[3], 0)), ha='center', va='bottom', fontsize=9)\n"
            "for a in (a1, a2): a.tick_params(axis='x', labelsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in res: print(f'{r[0]:18s} vol x{r[1]:.2f}  welch t={r[2]:+.2f}  day-0 HAC t={r[3]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: in a world where vol truly doubles, this harness reads "
            f"**×{R['syn'][1][1]:.2f} at Welch *t* ≈ {R['syn'][1][2]:.0f}** — and a planted "
            f"+150 bps drift reads HAC *t* = {R['syn'][2][5]:.2f}. On the null it stays silent "
            f"(×{R['syn'][0][1]:.2f}, *t* = {R['syn'][0][2]:.2f}). So the real tape's ×1.4–1.5 "
            "at *t* ≈ 3 and its 15 dead drift cells are measurements, not blind spots."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — *Real on the vol:* multiple **1.40×/1.53×/1.52×** "
            f"(CL/BZ/USO), Welch *t* = **+3.07/+3.01/+3.17**, BF *t* +2.57 to +3.09, range *t* "
            "to +4.80, placebo *p* ≤ 0.018 — clears *t* ≥ 2 on the real tape. *None on the "
            "drift:* all 15 horizon×tape HAC *t* inside (−1.2, +0.6).\n"
            f"- **Tradability `MIRAGE`** — nothing directional to deploy: drift ≈ 0 at every "
            f"horizon; continuation never robustly clears the bar (best cell "
            f"{R['cont']['USO'][2]:.2f} → {R['cont']['USO'][5]:.2f} ex-2020) and costs are "
            "irrelevant to that failure; the vol premium is precisely when optionality is "
            "already expensive.\n"
            f"- **Vol doubles? `BUSTED`** — 1.4–1.5× with 95% CIs of "
            f"{R['vol']['CL'][4]:.2f}–{R['vol']['CL'][5]:.2f} / "
            f"{R['vol']['BZ'][4]:.2f}–{R['vol']['BZ'][5]:.2f} / "
            f"{R['vol']['USO'][4]:.2f}–{R['vol']['USO'][5]:.2f}: 2.0 excluded everywhere, and "
            "the planted-×2 control proves the harness had the power to see a doubling."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Decision asymmetry.** The literature (Loutia et al. 2016; Demirer-Kutan 2010) "
            "finds cuts ≠ raises ≠ holds. The frozen table carries per-meeting labels — sort "
            "the drift by decision type and the two-sidedness becomes visible instead of "
            "averaged away.\n"
            "- **The excluded events are the sharp ones.** JMMC and V8-subgroup surprises "
            "(3 Apr 2023, the 2025 V8 calls) were excluded by the scope rule; adding them would "
            "likely *raise* the vol multiple — the scope choice is conservative, and the "
            "doubling still fails.\n"
            "- **Options, not futures.** A 1.5× realized-vol day is a straddle question: is the "
            "OPEC-day variance premium over- or under-priced in CL weeklies? That's a different "
            "study — this one only establishes the realized side.\n\n"
            "*The reproducible core is offline and deterministic; siblings: "
            "[313-geopolitical-shock](../../313-geopolitical-shock/) (unscheduled shocks), "
            "[226-crude-seasonality](../../226-crude-seasonality/) (calendar time), "
            "[602-macro-announcement-premium](../../602-macro-announcement-premium/) (the macro "
            "cousin). Methods and sources: [`docs/references.md`](../docs/references.md); frozen "
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
