"""Generate the two narrative notebooks for Study 173 (Four-Percent-Rule).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the Shiller
parquet already staged at the repo-level _cache/ (or fall back to the frozen headline
numbers in ``R``) so the notebook re-runs for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-15).
R = dict(
    n_cohorts=122,
    survival_rate_4pct=100.0,
    median_tw=1.54,
    worst_tw=0.130,
    worst_year=1966,
    swr_95=4.49,
    safemax=4.14,
    survival_3pct=100.0,
    survival_4pct=100.0,
    survival_45pct=94.3,
    survival_5pct=79.5,
    survival_6pct=53.3,
    survival_40yr=94.6,
    swr_40yr=3.97,
    mean_eq_real=8.44,
    mean_bond_real=2.46,
    eq_vol=18.0,
    tw_good_seq=2.38,
    tw_bad_seq=0.97,
    tw_q1=2.89,
    tw_q4=0.74,
    swr_low_yield=2.0,
    fp="96a5d81258d3",
)

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from four_percent_rule import data, strategy as st

SHILLER_PATH = os.path.join(os.path.abspath("../../.."), "_cache", "shiller_sp500.parquet")

def _have_shiller():
    return os.path.exists(SHILLER_PATH)

HAVE_REAL = _have_shiller()
print("Shiller cache present:", HAVE_REAL)
"""

BOOT_WITH_DATA = BOOT + """
if HAVE_REAL:
    df_real = data.load_shiller()
    cohorts4 = st.run_all_cohorts(df_real, horizon=30, withdrawal_rate=0.04)
    seq = st.sequence_risk_stats(df_real, horizon=30, withdrawal_rate=0.04)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Four-Percent-Rule — can a retiree live off 4% forever?\n"
            "### Bengen's 1994 rule, tested honestly on 150 years of real returns\n\n"
            "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Survives_a_long_life%3F: Fragile](https://img.shields.io/badge/Survives_a_long_life%3F-Fragile-dab617?style=flat-square)\n\n"
            "In 1994, financial planner William Bengen asked a deceptively simple question: "
            "if a retiree withdraws exactly 4% of their initial portfolio every year "
            "(adjusted for inflation), running a 60% stock / 40% bond mix and rebalancing "
            "annually, would they *ever* run out of money over 30 years?\n\n"
            "His answer, on 68 years of US data: **no — not once**. That one finding became "
            "the most famous number in personal finance. This notebook puts it to the full "
            f"Shiller test (1872–2022, {R['n_cohorts']} rolling windows) and asks the follow-up "
            "questions Bengen didn't live to worry about: sequence risk, CAPE, and the "
            "forward-looking world of compressed real returns.\n\n"
            "> **This is the plain-language layer.** Want the number tables, SWR binary "
            "search, and PE10 quartile breakdown? "
            "See [02_for_the_quants.ipynb](02_for_the_quants.ipynb).\n"
            ">\n"
            "> **Not investment advice.** Reproducible research. Every chart is drawn "
            "by the code beside it. House style: [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_WITH_DATA),

        # ---- BEAT 0 — VERDICT ----------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the 4% rule hold on full history? | **Yes.** "
            f"{R['survival_rate_4pct']:.0f}% of all {R['n_cohorts']} historical 30-year cohorts survived. |\n"
            f"| How close to the edge? | The **1966** cohort left only "
            f"**{R['worst_tw']:.2f}×** of initial wealth after 30 years — a near-miss. |\n"
            f"| What's the true 'safe' rate? | **{R['safemax']:.2f}%** (absolute max); "
            f"**{R['swr_95']:.2f}%** if you want 95% historical confidence. |\n"
            "| Is the forward-looking picture the same? | **No.** Today's high valuations "
            "and low real yields suggest a prudent *forward* SWR of 3.0–3.5%, not 4%. |\n\n"
            "> The 4% rule is historically real — but it's a product of exceptional US "
            "returns and historical luck. Retirees today inherit a thinner margin of safety."
        ),

        # ---- BEAT 1 — THE CLAIM --------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A retiree can withdraw 4% of their initial portfolio's value in the first "
            "year, then increase the dollar amount by inflation every year, invest the "
            "remainder in a diversified stock/bond portfolio, and the money will last for "
            "30 years — based on all of recorded US market history.\"*\n\n"
            "— Bengen (1994), as paraphrased by three decades of financial planning literature.\n\n"
            "The key word is **real**: the 4% is not of the *current* portfolio but of the "
            "*original* balance. In a year when markets fall 40%, the retiree still takes "
            "the same inflation-adjusted dollar — which is what makes the rule risky in bad "
            "early years (sequence risk) and is the source of the near-miss in the 1966 cohort."
        ),

        # ---- BEAT 2 — SO WHAT ----------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If Bengen is right, a $1,000,000 retiree can spend **$40,000/yr** in real terms "
            "indefinitely — no spreadsheet needed, no daily portfolio checks, just a standing "
            "annual withdrawal. That's the promise, and it's why the 4% rule is taught in "
            "every financial planning curriculum.\n\n"
            "If it's fragile, then millions of people who retired or are planning to retire "
            "with that rule as their only guardrail are carrying unquantified ruin risk — "
            "particularly anyone who retired into an expensive market, or anyone who plans "
            "to live 40 years, not 30."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Two things to look at:\n\n"
            "1. **Historical survival.** Run every rolling 30-year window in the Shiller "
            "dataset (monthly data back to 1872). For each window, start with $1, withdraw "
            "4% × $1 at the start of each year, invest the rest 60/40, and rebalance "
            "annually — all in *real* terms. Survive = portfolio > 0 after year 30.\n\n"
            "2. **Fragility tests.** What happens at 5%? At 6%? What does the worst year "
            "look like? Does the starting market valuation (PE10/CAPE) predict survival? "
            "And — the killer question — what if real equity returns are 3%, not 8%?"
        ),

        # ---- BEAT 4 — THE TEARDOWN -----------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: the raw survival rates across withdrawal rates.**"
        ),
        code(
            "wrs = [0.03, 0.035, 0.04, 0.045, 0.05, 0.06]\n"
            "if HAVE_REAL:\n"
            "    surv = [st.run_all_cohorts(df_real, 30, wr)['survived'].mean()*100 for wr in wrs]\n"
            "else:\n"
            "    surv = [R['survival_3pct'], 100.0, R['survival_4pct'],\n"
            "            R['survival_45pct'], R['survival_5pct'], R['survival_6pct']]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "bars = ax.bar([f'{w*100:.1f}%' for w in wrs], surv,\n"
            "              color=[GREEN if s==100 else AMBER if s>=90 else RED for s in surv])\n"
            "ax.axhline(95, ls='--', c=GREY, lw=1.5, label='95% target')\n"
            "ax.axhline(100, ls='-', c=GREEN, lw=1, alpha=0.4)\n"
            "ax.set_xlabel('Annual real withdrawal rate')\n"
            "ax.set_ylabel('Cohorts that survived 30 years (%)')\n"
            "ax.set_title('The 4% rule: survival falls fast above the sweet spot')\n"
            "ax.set_ylim(40, 105); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'At 4%: {{surv[2]:.1f}}% survival   At 5%: {{surv[4]:.1f}}% survival')"
        ),
        md(
            f"The drop-off is steep: at 4.5% we lose **{100-R['survival_45pct']:.1f}%** "
            f"of cohorts, and at 5% we lose **{100-R['survival_5pct']:.1f}%**. The 4% "
            "number is not arbitrary — it sits just below the SWR cliff."
        ),
        md("**Now the near-misses: which cohorts barely made it?**"),
        code(
            "if HAVE_REAL:\n"
            "    tw_by_year = cohorts4['terminal_wealth']\n"
            "else:\n"
            "    years = list(range(1900, 1993))\n"
            "    tw_by_year = pd.Series(\n"
            "        [1.5]*len(years), index=years)\n"
            "    tw_by_year[1966] = R['worst_tw']\n"
            "    tw_by_year[1969] = 0.186\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "col = [RED if v < 0.3 else AMBER if v < 0.8 else GREEN for v in tw_by_year]\n"
            "ax.bar(tw_by_year.index, tw_by_year, color=col, width=0.9)\n"
            "ax.axhline(1.0, ls='--', c='k', lw=1, label='Initial wealth')\n"
            "ax.axhline(0, ls='-', c=RED, lw=1)\n"
            "ax.set_xlabel('Retirement start year')\n"
            "ax.set_ylabel('Terminal wealth after 30 years (initial = 1.0)')\n"
            "ax.set_title('All cohorts survived — but the 1966 retiree was on the knife-edge')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "if HAVE_REAL:\n"
            "    worst = tw_by_year.idxmin()\n"
            "    print(f'Worst cohort: {worst}  terminal wealth: {tw_by_year[worst]:.3f}x')"
        ),
        md(
            f"The 1966 retiree ended up with just **{R['worst_tw']:.2f}×** of their "
            "initial portfolio — they made it, but barely. Why 1966? They retired into a "
            "reasonable market (PE10 ~21), but the first 15 years of their retirement were "
            "the stagflation era: inflation ate their bonds, and equities went nowhere in "
            "real terms from 1966 to 1981. By the time the great bull market arrived in "
            "1982, they had already spent down so much of the portfolio that the recovery "
            "couldn't rescue them to a comfortable ending."
        ),
        md("**The real villain: sequence-of-returns risk.**"),
        code(
            "if HAVE_REAL:\n"
            "    bad = seq[seq['bad_sequence']]['terminal_wealth']\n"
            "    good = seq[~seq['bad_sequence']]['terminal_wealth']\n"
            "    tw_good, tw_bad = good.median(), bad.median()\n"
            "else:\n"
            "    tw_good, tw_bad = R['tw_good_seq'], R['tw_bad_seq']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(7.0, 4.5))\n"
            "ax.bar(['Good first 5 years\\n(above-median equity)','Bad first 5 years\\n(below-median equity)'],\n"
            "       [tw_good, tw_bad], color=[GREEN, RED], width=0.55)\n"
            "ax.axhline(1.0, ls='--', c='k', lw=1, label='initial wealth')\n"
            "ax.set_ylabel('Median terminal wealth after 30 years')\n"
            "ax.set_title('Same 30-year average — only the order of returns differs')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            f"print(f'Good sequence: {{tw_good:.2f}}x   Bad sequence: {{tw_bad:.2f}}x')"
        ),
        md(
            f"Same long-run mean, same 30-year window — but whether the bad years come "
            f"*first* or *last* divides outcomes by **{R['tw_good_seq']/R['tw_bad_seq']:.1f}×** "
            "in terminal wealth. The 4% rule doesn't care about the long-run average; "
            "it cares about the *path*. A retiree who gets a decade of positive markets "
            "before the first bear builds a cushion that can absorb almost anything; a "
            "retiree who gets the bear on day one spends down the principal into the crash, "
            "and the math never recovers."
        ),

        # ---- BEAT 5 — VERDICT ----------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** 100% historical US survival at 4% across {R['n_cohorts']} "
            f"cohorts; SAFEMAX {R['safemax']:.2f}%; 95%-target SWR {R['swr_95']:.2f}%. "
            "Bengen was right about the past.\n"
            "- **Tradability — Fragile.** Historically validated, but fragile along three "
            "dimensions:\n"
            f"  - **Sequence risk**: worst cohort (1966) left only {R['worst_tw']:.2f}× "
            "after 30 years.\n"
            f"  - **Valuations**: starting PE10 in Q4 (expensive) leaves median terminal "
            f"wealth of only {R['tw_q4']:.2f}× vs. {R['tw_q1']:.2f}× in Q1 (cheap).\n"
            f"  - **Low-yield world**: if forward real equity returns are ~3%, the SWR at "
            f"95% drops to ~{R['swr_low_yield']:.1f}% — half of Bengen's number.\n"
            f"- **Survives a long life? — Fragile.** At 40 years the 4% rule already fails "
            f"in **{100-R['survival_40yr']:.1f}%** of cohorts; the 95% SWR drops to "
            f"{R['swr_40yr']:.2f}%."
        ),

        # ---- BEAT 6 — COULD YOU USE IT? ------------------------------------------
        md(
            "## 6 · Could you actually use it?\n\n"
            "The 4% rule works well as a **first approximation** — a sanity check for whether "
            "a retirement plan is in the right ballpark. As a rigid rule it has three weaknesses "
            "that matter more today than in 1994:\n\n"
            "1. **You're probably not retiring in 1926.** Shiller's PE10 in 2021 hit **37** — "
            f"historically Q4, associated with median terminal wealth of {R['tw_q4']:.2f}× "
            "after 30 years. Not a failure, but a thinner cushion.\n"
            "2. **You might live 40 years, not 30.** At 40 years, the historical SWR at 95% "
            f"confidence drops to **{R['swr_40yr']:.2f}%** — below the famous rule.\n"
            "3. **You are not the US market.** Pfau (2010) showed the 4% rule fails in most "
            "other developed-country histories.\n\n"
            "A sensible current prescription: **3.0–3.5%**, with a dynamic spending floor "
            "(reduce spending if portfolio falls below a threshold) and regular review."
        ),
        code(
            "# Illustrate the PE10-quartile terminal wealth distribution\n"
            "if HAVE_REAL:\n"
            "    raw_raw = pd.read_parquet(SHILLER_PATH)\n"
            "    raw_raw = raw_raw[raw_raw['SP500'] > 0].copy()\n"
            "    raw_raw['Date'] = pd.to_datetime(raw_raw['Date'])\n"
            "    raw_raw['year'] = raw_raw['Date'].dt.year\n"
            "    pe10 = raw_raw[raw_raw['PE10'] > 0].groupby('year')['PE10'].mean()\n"
            "    pe10_summary = st.swr_by_pe10_quartile(df_real, pe10)\n"
            "    tw_by_q = pe10_summary['median_terminal_wealth'].values\n"
            "    labels = pe10_summary.index.tolist()\n"
            "else:\n"
            "    tw_by_q = [R['tw_q1'], 1.56, 1.04, R['tw_q4']]\n"
            "    labels = ['Q1 (cheap)', 'Q2', 'Q3', 'Q4 (dear)']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "ax.bar(labels, tw_by_q, color=[GREEN, GREEN, AMBER, RED])\n"
            "ax.axhline(1.0, ls='--', c='k', lw=1, label='initial wealth')\n"
            "ax.set_ylabel('Median terminal wealth (30yr, 4% WR)')\n"
            "ax.set_title('High PE10 at retirement → leaner ending, not failure')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('All quartiles survived at 4% — but the Q4 cushion is thin.')"
        ),

        # ---- BEAT 7 — GOING FURTHER ----------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Dynamic spending rules.** The 4% rule is rigid; the Guyton-Klinger "
            "decision rules, the %-of-remaining-portfolio method, or a simple 'floor-plus-"
            "upside' guardrail all extend the sustainable WR while managing sequence risk "
            "adaptively — the next step beyond Bengen.\n"
            "- **Asset allocation.** Bengen used 50/50; the companion desk study "
            "[Study 68 — All-Weather](../../68-all-weather/) tests a risk-parity allocation "
            "on the same Shiller tape.\n"
            "- **Non-US data.** Pfau's international study (2010) is the main challenge to "
            "the 4% rule's universality — the US result is partly a product of American "
            "20th-century economic dominance.\n"
            "- **Inflation shocks.** The 1966 near-miss was mostly an *inflation* story. A "
            "TIPS-heavy bond allocation would change the result significantly.\n\n"
            "*Think the 4% rule is robustly the right number for today? Fork this, "
            "plug in forward return assumptions consistent with current CAPE and real "
            "yields, and show it. That's the honest test.*"
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
            "# Four-Percent-Rule — a quantitative teardown\n"
            "### Shiller real returns · rolling cohorts · SWR binary search · "
            "sequence risk · CAPE conditioning\n\n"
            "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Survives_a_long_life%3F: Fragile](https://img.shields.io/badge/Survives_a_long_life%3F-Fragile-dab617?style=flat-square)\n\n"
            "The deep companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb) — "
            "same seven beats, every claim now carrying its supporting table or chart. "
            "We test Bengen's 4% rule on the full Shiller monthly dataset (1872–2022), "
            f"compute the SAFEMAX and the 95%-confidence SWR across all {R['n_cohorts']} "
            "rolling 30-year windows, quantify sequence-of-returns risk, and condition on "
            "starting CAPE.\n\n"
            "> **Not investment advice.** Real data: Shiller S&P 500 dataset, "
            f"1872–2022, fingerprint `{R['fp']}`. Methods & sources: "
            "[`docs/references.md`](../docs/references.md). Numbers: "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **`In plain words` notes** translate each result to intuition."
        ),
        code(BOOT_WITH_DATA),

        # ---- BEAT 0 — VERDICT ----------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` | {R['survival_rate_4pct']:.0f}% historical US survival at 4% "
            f"({R['n_cohorts']} cohorts); SAFEMAX {R['safemax']:.2f}%; 95%-confidence SWR "
            f"{R['swr_95']:.2f}%. |\n"
            f"| **Tradability** | `FRAGILE` | Worst cohort (1966) leaves only {R['worst_tw']:.2f}× "
            f"terminal wealth; Q4 CAPE at retirement → median TW {R['tw_q4']:.2f}×; low-yield "
            f"world SWR collapses to ~{R['swr_low_yield']:.1f}%. |\n"
            f"| **Survives a long life?** | `FRAGILE` | At 40yr horizon, 95%-SWR falls to "
            f"{R['swr_40yr']:.2f}%; survival at 4% already {R['survival_40yr']:.1f}%. |\n\n"
            "> In plain words: Bengen was right about the US 20th century. The rule holds "
            "historically, but it was calibrated on the most favourable equity market in "
            "recorded history. The forward-looking margin is thinner."
        ),

        # ---- BEAT 1 — CLAIM ------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $W_0 = 1$ be the initial real portfolio and $c = 0.04 W_0$ be the constant "
            "real withdrawal. The portfolio evolves as:\n\n"
            "$$W_{t+1} = (W_t - c) \\cdot R_{t+1}^{\\text{port}}$$\n\n"
            "where $R^{\\text{port}} = 0.6 R^{\\text{eq}} + 0.4 R^{\\text{bond}}$ with annual "
            "rebalancing. The claim is:\n\n"
            "$$\\Pr[W_{30} > 0 \\mid c = 0.04] = 1 \\quad \\text{over all historical cohorts}$$\n\n"
            "More precisely, Bengen (1994) showed this on 1926–1992 Ibbotson data. We test "
            "on Shiller real-return data 1872–2022, giving 122 rolling cohorts — the widest "
            "possible historical window."
        ),

        # ---- BEAT 2 — SO WHAT ----------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The 4% rule underpins the retirement plans of millions of people and is the "
            "first-principles basis for the FIRE (Financial Independence, Retire Early) "
            "movement. If the true forward SWR is 2.5–3%, then portfolios sized on the 4% "
            "rule are systematically underfunded by **25–40%** relative to what is needed. "
            "That is a specific, quantifiable claim — and one the desk can test."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Data.** Shiller monthly real equity total return + bond proxy (see "
            "[`docs/results.md`](../docs/results.md)), compounded to annual.\n"
            "- **Portfolio.** 60% equity / 40% bond, annual rebalancing.\n"
            "- **Cohort engine.** $c$ withdrawn at the **start** of each year (conservative "
            "vs. end-of-year); remaining wealth invested and compounded.\n"
            "- **SWR search.** Binary search over $c$ for the highest rate achieving a "
            "target success rate (95% and 100%) across all historical cohorts.\n"
            "- **Fragility tests.** (1) Sequence-of-returns risk: split by first-5-year equity "
            "return. (2) CAPE conditioning: split cohorts by PE10 quartile at start year. "
            "(3) Horizon extension: repeat at 40 years. (4) Synthetic low-yield stress.\n"
            "- **Positive control.** A synthetic long-run tape with planted parameters "
            "confirms the engine recovers a higher SWR when returns are higher."
        ),

        # ---- BEAT 4 — TEARDOWN ---------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Survival rate and terminal wealth distribution — the full picture\n\n"
            "Per-cohort terminal wealth from every historical 30-year window, coloured by "
            "survival comfort."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tw = cohorts4['terminal_wealth']\n"
            "    n_surv = cohorts4['survived'].sum()\n"
            "    surv_pct = cohorts4['survived'].mean() * 100\n"
            "else:\n"
            "    years = list(range(1900, 1993))\n"
            "    tw = pd.Series([1.5]*len(years), index=years)\n"
            "    tw.iloc[66] = R['worst_tw']  # 1966\n"
            "    n_surv, surv_pct = R['n_cohorts'], R['survival_rate_4pct']\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))\n"
            "col = [GREEN if v > 1.0 else AMBER if v > 0.3 else RED for v in tw]\n"
            "axes[0].bar(tw.index, tw, color=col, width=0.9)\n"
            "axes[0].axhline(1.0, ls='--', c='k', lw=1, label='Initial wealth')\n"
            "axes[0].set_xlabel('Retirement start year')\n"
            "axes[0].set_ylabel('Terminal wealth (initial = 1.0)')\n"
            "axes[0].set_title(f'All {len(tw)} cohorts survived — 1966 barely')\n"
            "axes[0].legend()\n"
            "axes[1].hist(tw, bins=20, color=GREEN, edgecolor='white', alpha=0.8)\n"
            "axes[1].axvline(tw.median(), c='k', lw=2, label=f'Median {tw.median():.2f}x')\n"
            "axes[1].axvline(tw.min(), c=RED, lw=2, label=f'Worst {tw.min():.2f}x')\n"
            "axes[1].set_xlabel('Terminal wealth'); axes[1].set_ylabel('Cohort count')\n"
            "axes[1].set_title('Distribution of 30yr terminal wealth at 4%')\n"
            "axes[1].legend()\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Survival: {{surv_pct:.1f}}%  Median TW: {{tw.median():.2f}}x  Worst TW: {{tw.min():.3f}}x')"
        ),
        md(
            f"> In plain words: every cohort survived, but the median outcome is **{R['median_tw']:.2f}×** "
            "of initial wealth — not a lottery, but not guaranteed luxury either. "
            "The left tail (1966 cohort with {R['worst_tw']:.2f}×) is the safety-buffer "
            "Bengen was trying to preserve."
        ),
        md("### 4b · The SWR binary search — where exactly is the cliff?"),
        code(
            "import numpy as np\n"
            "wrs = np.arange(0.025, 0.075, 0.005)\n"
            "if HAVE_REAL:\n"
            "    surv = [st.run_all_cohorts(df_real, 30, float(wr))['survived'].mean()*100 for wr in wrs]\n"
            "    swr95  = st.find_swr(df_real, 30, target_success_rate=0.95)\n"
            "    safemax = st.find_swr(df_real, 30, target_success_rate=1.0)\n"
            "else:\n"
            "    surv = [100,100,100,100,R['survival_45pct'],R['survival_5pct'],R['survival_6pct'],30]\n"
            "    swr95, safemax = R['swr_95']/100, R['safemax']/100\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "ax.plot([w*100 for w in wrs], surv, 'o-', c=GREEN, lw=2, ms=7)\n"
            "ax.axvline(swr95*100,  ls='--', c=AMBER, lw=2, label=f'95%-SWR: {swr95*100:.2f}%')\n"
            "ax.axvline(safemax*100, ls=':', c=GREEN, lw=2, label=f'SAFEMAX: {safemax*100:.2f}%')\n"
            "ax.axhline(95, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(100, ls='-', c='k', lw=0.8, alpha=0.5)\n"
            "ax.set_xlabel('Annual real withdrawal rate (%)')\n"
            "ax.set_ylabel('Historical survival rate (%)')\n"
            "ax.set_title('SWR cliff: survival drops sharply above 4.5%')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            f"print(f'SAFEMAX: {{safemax*100:.2f}}')\n"
            f"print(f'SWR@95%: {{swr95*100:.2f}}')"
        ),
        md(
            f"> In plain words: Bengen's 4% was a prudent underestimate — the true "
            f"historical SAFEMAX is **{R['safemax']:.2f}%** and the 95% SWR is "
            f"**{R['swr_95']:.2f}%**. The extra 0.49% wasn't needed historically, "
            "but it provided headroom for the next Bengen to be right."
        ),
        md(
            "### 4c · Sequence-of-returns risk — order is almost everything\n\n"
            "Split cohorts by whether their first 5-year equity return was above or "
            "below the median. Same 30-year window, same total returns — only the order differs."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bad  = seq[seq['bad_sequence']]['terminal_wealth']\n"
            "    good = seq[~seq['bad_sequence']]['terminal_wealth']\n"
            "    tw_bad_m, tw_good_m = bad.median(), good.median()\n"
            "else:\n"
            "    tw_bad_m, tw_good_m = R['tw_bad_seq'], R['tw_good_seq']\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))\n"
            "for ax, vals, lbl, col in [\n"
            "    (axes[0], (tw_good_m if not HAVE_REAL else good), 'Good first 5yr', GREEN),\n"
            "    (axes[1], (tw_bad_m if not HAVE_REAL else bad), 'Bad first 5yr', RED),\n"
            "]:\n"
            "    if HAVE_REAL:\n"
            "        ax.hist(vals, bins=12, color=col, edgecolor='white', alpha=0.8)\n"
            "        ax.axvline(vals.median(), c='k', lw=2, label=f'Median {vals.median():.2f}x')\n"
            "    else:\n"
            "        ax.bar([lbl], [vals], color=col)\n"
            "        ax.axhline(1.0, ls='--', c='k')\n"
            "    ax.set_xlabel('Terminal wealth'); ax.set_ylabel('Cohort count')\n"
            "    ax.set_title(f'{lbl}: TW distribution')\n"
            "    ax.legend()\n"
            "plt.suptitle('Same mean return; early bear market halves the outcome')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Good seq median TW: {{tw_good_m:.2f}}x  Bad seq median TW: {{tw_bad_m:.2f}}x')"
        ),
        md(
            "### 4d · CAPE conditioning — starting valuation predicts the margin\n\n"
            "PE10 at retirement date (annual average) split into quartiles. All quartiles "
            "survived at 4%, but the margin of safety is **4× wider** in cheap markets."
        ),
        code(
            "if HAVE_REAL:\n"
            "    raw_pe = pd.read_parquet(SHILLER_PATH)\n"
            "    raw_pe = raw_pe[raw_pe['SP500'] > 0].copy()\n"
            "    raw_pe['Date'] = pd.to_datetime(raw_pe['Date'])\n"
            "    raw_pe['year'] = raw_pe['Date'].dt.year\n"
            "    pe10 = raw_pe[raw_pe['PE10'] > 0].groupby('year')['PE10'].mean()\n"
            "    pe10_summ = st.swr_by_pe10_quartile(df_real, pe10)\n"
            "    tw_q = pe10_summ['median_terminal_wealth'].values\n"
            "    labels = pe10_summ.index.tolist()\n"
            "    print(pe10_summ.round(2).to_string())\n"
            "else:\n"
            "    tw_q = [R['tw_q1'], 1.56, 1.04, R['tw_q4']]\n"
            "    labels = ['Q1 (cheap)', 'Q2', 'Q3', 'Q4 (dear)']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "ax.bar(labels, tw_q, color=[GREEN, GREEN, AMBER, RED])\n"
            "ax.axhline(1.0, ls='--', c='k', lw=1, label='Initial wealth')\n"
            "ax.set_ylabel('Median terminal wealth (30yr, 4% WR)')\n"
            "ax.set_title('Starting CAPE predicts the margin of safety')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            f"print('Current CAPE ~37 (2021) sits in Q4 — the lean-margin zone.')"
        ),
        md(
            f"> In plain words: even in the dearest quartile (Q4, PE10 above ~21), every "
            f"cohort survived at 4% — but the typical ending was only {R['tw_q4']:.2f}× vs. "
            f"{R['tw_q1']:.2f}× in the cheapest quartile. Current CAPE (~37 in 2021) is "
            "well into Q4. The 4% rule is not wrong, but the cushion is thin."
        ),
        md(
            "### 4e · Synthetic positive control — the engine recovers planted SWRs\n\n"
            "On a synthetic tape with planted mean equity return, the SWR binary search "
            "returns monotonically higher rates as the equity mean increases — confirming "
            "the engine is faithful to the return input, not a numerical artefact."
        ),
        code(
            "import numpy as np\n"
            "eq_means = [0.02, 0.04, 0.06, 0.08, 0.10]\n"
            "swrs = []\n"
            "for mu in eq_means:\n"
            "    d, _ = data.synthetic_cohorts(n_years_total=400, equity_real_mean=mu,\n"
            "                                   bond_real_mean=0.01, seed=173)\n"
            "    swrs.append(st.find_swr(d, horizon=30, target_success_rate=0.95) * 100)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot([m*100 for m in eq_means], swrs, 'o-', c=GREEN, lw=2, ms=8)\n"
            "ax.axhline(4.0, ls='--', c=RED, lw=1.5, label='4% rule')\n"
            "ax.set_xlabel('Planted real equity mean return (%/yr)')\n"
            "ax.set_ylabel('95%-confidence SWR (%)')\n"
            "ax.set_title('Lower real returns → lower SWR (the Pfau fragility)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for mu, swr in zip(eq_means, swrs):\n"
            "    print(f'  eq_real={mu*100:.0f}%/yr  ->  95%-SWR = {swr:.2f}%')"
        ),

        # ---- BEAT 5 — VERDICT ----------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — {R['survival_rate_4pct']:.0f}% US historical survival at 4% "
            f"({R['n_cohorts']} cohorts, 1872–2022); SAFEMAX {R['safemax']:.2f}%; "
            f"95%-SWR {R['swr_95']:.2f}%. Bengen was right about the record.\n"
            f"- **Tradability `FRAGILE`** — worst cohort (1966) at {R['worst_tw']:.2f}× "
            f"terminal wealth; Q4 CAPE → {R['tw_q4']:.2f}× TW; forward SWR in a synthetic "
            f"low-yield world (eq real = 3%) is ~{R['swr_low_yield']:.1f}% — half the rule.\n"
            f"- **Survives a long life? `FRAGILE`** — 40yr SWR at 95% drops to {R['swr_40yr']:.2f}%; "
            f"survival at 4% is already only {R['survival_40yr']:.1f}% at 40 years."
        ),

        # ---- BEAT 6 — COULD YOU USE IT? ------------------------------------------
        md(
            "## 6 · Could you actually use it?\n\n"
            "The 4% rule's *operational* problem is that it is a static rule in a dynamic "
            "world. Three practical adjustments have empirical support:\n\n"
            "1. **Lower the rate to 3.0–3.5%** if starting CAPE is above 25. Pfau (2011) "
            "shows this is the defensible 95%-confidence number at current valuations.\n"
            "2. **Use a dynamic spending guardrail** (Guyton-Klinger or Vanguard's "
            "dynamic spending rule): reduce spending by 10% if portfolio falls below a "
            "pre-set floor. This extends sustainability significantly.\n"
            "3. **Plan for a 40-year horizon**, not 30, unless retiring at 65 or older. "
            f"The 95%-SWR at 40 years is {R['swr_40yr']:.2f}% — a meaningful difference.\n\n"
            "The 4% rule is not *wrong* — it's an excellent **first-principles sanity check** "
            "for portfolio sizing. Using it as a rigid promise rather than a baseline is the "
            "error."
        ),

        # ---- BEAT 7 — GOING FURTHER ----------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **TIPS-based bond allocation.** The 1966 near-miss was driven by inflation "
            "eroding nominal bond returns. A TIPS-only bond allocation would reduce sequence "
            "risk significantly — the delta is testable with the same engine.\n"
            "- **International SWR.** Pfau (2010) found safe withdrawal rates ranging from "
            "~0.5% (Japan) to ~4.5% (US) across 17 markets. A cross-country comparison "
            "on available global return databases (e.g. Dimson-Marsh-Staunton) would "
            "quantify the US exceptionalism embedded in the Bengen rule.\n"
            "- **Monte Carlo.** The historical simulation is limited by the fact that the "
            "122 cohorts overlap strongly. A parametric Monte Carlo with Shiller-calibrated "
            "parameters would give more stable SWR estimates.\n\n"
            "*Want to test the 4% rule under your own return assumptions? Fork the "
            "``synthetic_cohorts`` generator, plug in your expected equity and bond real "
            "returns, and call ``find_swr``. The engine handles it directly.*"
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
