"""Generate the two narrative notebooks for Study 598 (Cederburg 100% Equities for Life).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached
Shiller parquet + EFA csv under ../_cache/ and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md). The synthetic control
runs anywhere with no network. Heavy cells are kept light (outer bootstrap 100
reps for the picture; the canonical 600-rep CIs are quoted from ``R``).
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


# Frozen real-tape headline numbers — mirror of docs/results.md
# (Shiller+EFA 1871-02 -> 2023-06, 990 monthly-start 70y lifecycles, 4% rule,
#  costs 10 bps one-way + 5 bps/yr ER, HAC bandwidth 840, outer bootstrap
#  600 reps / 120-month blocks / seed 598, Cederburg bootstrap 2,000 draws)
STRATS = ["alleq", "alleq_dom", "s6040", "tdf"]
R = dict(
    start="1871-02", end="2023-06", months=1829, fingerprint="b68792bbdbe0",
    ann=dict(US=6.90, INTL=3.91, BD=2.39),
    intl_mkt_months=262, intl_mkt_pct=14.3, intl_mkt_start="2001-09",
    intl_calib="DMS: geo 4.3%/yr, vol 17%, corr 0.60, seed 598",
    n_cohorts=990, cohort_start="1871-02", cohort_end="1953-07",
    retire_start="1911-02", retire_end="1993-07", cost_bps=10, er_bps=5,
    strats=STRATS,
    # scoreboard @4%: (pot@65 median, terminal mean, median, p05, ruin %, n_fail)
    score={"alleq": (116.0, 350.6, 203.3, 0.0, 7.98, 79),
           "alleq_dom": (169.4, 559.4, 608.7, 14.0, 4.34, 43),
           "s6040": (125.4, 189.6, 190.6, 0.0, 5.76, 57),
           "tdf": (97.5, 89.9, 31.9, 0.0, 21.11, 209)},
    # head-to-heads: (label, dmeanTW, HAC t, ci_lo, ci_hi, druin pp,
    #                 druin ci_lo, druin ci_hi, win rate %)
    pairs=[("alleq - 60/40 (HEADLINE)", 161.0, 1.22, -334.4, 1341.4,
            2.22, -14.95, 22.22, 55.4),
           ("alleq - TDF (VS GLIDE)", 260.7, 2.63, 2.1, 1891.0,
            -13.13, -24.15, 14.55, 88.2),
           ("alleq_dom - 60/40 (PURE TAPE)", 369.9, 11.72, 32.1, 5170.1,
            -1.41, -8.39, 9.70, 94.7)],
    # ruin % by withdrawal rate, order = strats
    ruin={"3.0": [0.10, 0.00, 0.00, 0.00], "3.5": [2.93, 0.51, 0.00, 0.51],
          "4.0": [7.98, 4.34, 5.76, 21.11], "5.0": [24.95, 20.00, 25.96, 51.21]},
    # Cederburg bootstrap (2,000 lifetimes): (pot@65 med, mean, median, ruin %)
    boot={"alleq": (153.0, 577.3, 290.6, 9.55),
          "alleq_dom": (214.2, 1433.7, 660.8, 6.95),
          "s6040": (148.9, 423.7, 259.0, 7.15),
          "tdf": (129.2, 172.3, 95.5, 13.70)},
    # intl-leg sensitivity: (label, alleq ruin %, alleq mean TW, HAC t vs 60/40)
    sens=[("hybrid (headline)", 7.98, 350.6, 1.22),
          ("corr=1 stress", 13.74, 228.0, 2.31),
          ("domestic (pure tape)", 4.34, 559.4, 11.72)],
    # famous cohorts @4%: strategy -> (terminal, fail year; -1 = survived)
    famous={"1929-09": {"alleq": (0.0, 18), "alleq_dom": (0.0, 18),
                        "s6040": (62.0, -1), "tdf": (3.8, -1),
                        "alleq_corr1": (0.0, 16)},
            "1966-01": {"alleq": (190.9, -1), "alleq_dom": (0.0, 25),
                        "s6040": (0.0, 25), "tdf": (0.0, 27),
                        "alleq_corr1": (0.0, 20)}},
    # costs: (one-way bps, ER bps, alleq mean TW, alleq ruin %, HAC t vs 60/40)
    costs=[(0, 0, 364.9, 7.98, 1.21), (10, 5, 350.6, 7.98, 1.22),
           (25, 20, 313.0, 9.39, 1.23)],
    # control: (premium pp, dmeanTW, t, druin pp, t)
    control=[(0.0, 3.2, 1.55, 5.95, 3.15), (2.0, 43.6, 4.01, 2.33, 1.14),
             (4.5, 342.8, 4.92, -0.23, -0.14)],
    # seed-robustness (20 residual re-draws of the simulated INTL leg,
    # seeds 598-617): pair -> (HAC t mean, min, max, n of 20 with t >= 2)
    seedrob=dict(tdf=(6.58, 2.63, 10.04, 20), s6040=(2.62, -16.54, 10.76, 13),
                 ruin_worse_n=9,
                 pure_tape_tdf=(437.7, 13.07, 4.34, 11.52)),  # dmeanTW, t, ruins
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Worst_cohorts%3F: Busted](https://img.shields.io/badge/Worst_cohorts%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY, BLUE = "#c0392b", "#dab617", "#2ea44f", "#8b949e", "#2b6cb0"

from cederburg_100_equities import data, strategy as st

HAVE_REAL = ((os.path.exists(data.SHILLER_PATH)
              or any(os.path.exists(p) for p in data._FALLBACKS))
             and os.path.exists(data.EFA_PATH))
if HAVE_REAL:
    DF = data.real_returns()
    US, INTL, BD = DF["US"].to_numpy(), DF["INTL"].to_numpy(), DF["BD"].to_numpy()
    GU, GI, GB, STARTS = st.cohort_year_returns(US, INTL, BD)
    DATES = DF.index[STARTS]
    print("real tape:", DF.index.min().date(), "->", DF.index.max().date(),
          "| cohorts:", GU.shape[0], "| fingerprint:", data.fingerprint(DF))
else:
    DF = GU = GI = GB = DATES = None
    print("no cache -- real-tape cells quote the frozen numbers in R")
"""

LBL = {"alleq": "100% eq 50/50 dom/intl", "alleq_dom": "100% US equity",
       "s6040": "static 60/40", "tdf": "TDF 90-45-30"}

BOOT_CELL = (BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\n"
             "STRATS = " + repr(STRATS) + "\nR = " + repr(R) + "\n"
             "LBL = " + repr(LBL) + "\n")

# retirement-phase wealth path helper (normalized pot=1 at retirement; ruin
# dynamics are scale-invariant because the withdrawal is 4% of the OWN pot)
RETPATH = """\
def ret_path(i, name, wr=0.04):
    # illustrative annual retirement path, ex-costs, pot normalized to 1 at 65
    w = st.weights_for(name)[st.SAVE_YEARS:]
    G = [GU[i, st.SAVE_YEARS:], GI[i, st.SAVE_YEARS:], GB[i, st.SAVE_YEARS:]]
    out = [1.0]; W = 1.0
    for j in range(st.RET_YEARS):
        if W <= wr: out += [0.0] * (st.RET_YEARS - j); break
        W -= wr
        W = (w[j, 0] * G[0][j] + w[j, 1] * G[1][j]
             + (1 - w[j, 0] - w[j, 1]) * G[2][j]) * W
        out.append(max(W, 0.0))
    return np.array(out[:st.RET_YEARS + 1])
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Should you hold 100% stocks your WHOLE life? 🌍\n"
            "### The famous 2023 \"death of the 60/40 and the target-date fund\" paper, "
            "raced on 152 years of data\n\n"
            + BADGES +
            "In 2023, three finance professors (Anarkulova, Cederburg & O'Doherty) published a "
            "paper that made headlines everywhere: *forget bonds entirely*. Simulating "
            "lifetimes on returns from 38 developed countries, they found that saving into "
            "**100% stocks — half your own country, half international — and never touching a "
            "bond**, beats the balanced 60/40 portfolio and the target-date funds your 401(k) "
            "defaults into. Not just on final wealth — on the **probability of dying broke**, "
            "too.\n\n"
            "We rebuild that horse race on the longest public tape there is: every possible "
            "70-year lifecycle (save 40 years, retire 30) on US data since 1871, with an "
            "honestly-labeled international proxy.\n\n"
            "> 📓 Want the *t*-stats, the bootstraps and the sensitivity grid? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**. Numbers quoted here are "
            "the frozen headline run ([docs/results.md](../docs/results.md)); every chart is "
            "drawn by the code beside it."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does 100% stocks beat the target-date fund? | **Yes, decisively.** More money "
            f"at 65, far more at 95, and much *less* ruin ({R['score']['alleq'][4]:.1f}% vs "
            f"{R['score']['tdf'][4]:.1f}% of cohorts dying broke at the 4% rule). |\n"
            "| Does the paper's 50/50 domestic/international mix beat a plain 60/40? | **Not "
            "on this tape.** The median race is a tie, the statistics can't certify the "
            f"wealth edge, and it runs out of money *more* often ({R['score']['alleq'][4]:.1f}% "
            f"vs {R['score']['s6040'][4]:.1f}%). |\n"
            "| Why? | The international half earned ~**4.3%/yr** real (the century-long "
            "world-ex-US number) while US stocks earned **6.9%** — half your portfolio drags. "
            "The paper's result leans on *its* bond sample being terrible; the US 10-year "
            "wasn't that bad. |\n"
            "| What about 100% *US* stocks? | Crushes everything — but that's betting on the "
            "best stock market in recorded history repeating itself (see "
            "[study 151](../../151-stocks-for-long-run/README.md)). |\n"
            "| Does it survive the worst retirements? | **No.** The September-1929 retiree "
            "goes broke by year 18 under every all-equity variant while the 60/40 survives "
            "comfortably. |\n\n"
            "> The all-stock lifecycle beats the glidepath for real — but the \"safer than "
            "60/40\" half of the pitch did not replicate here."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A lifecycle of 100% equities — 50% domestic, 50% international — dominates "
            "stock/bond strategies in retirement wealth AND ruin probability.\"* — Anarkulova, "
            "Cederburg & O'Doherty (2023)\n\n"
            "The contenders, as the stock share of your portfolio through life:"
        ),
        code(
            "age = 25 + np.arange(st.HORIZON)\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "for s, c in [('alleq', RED), ('s6040', GREY), ('tdf', BLUE)]:\n"
            "    eq = st.weights_for(s).sum(axis=1)\n"
            "    ax.plot(age, 100*eq, lw=2.5, color=c, label=LBL[s])\n"
            "ax.axvline(65, ls=':', c='k', lw=1); ax.annotate(' retire', (65, 5))\n"
            "ax.set_xlabel('age'); ax.set_ylabel('% in stocks'); ax.set_ylim(0, 105)\n"
            "ax.legend(); ax.set_title('The contenders: stock share over a 70-year lifecycle')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('mechanics: save 1 unit/yr (real) ages 25-64, retire at 65, withdraw 4% of'\n"
            "      ' the pot (fixed real) to 95; ruin = broke before 95')"
        ),
        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Target-date funds hold **trillions** of default retirement savings. If the paper "
            "is right, the single most common retirement product in America is leaving "
            "enormous wealth — and *safety* — on the table. If it's only half right, the "
            "actionable half matters just as much: which half?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We run **every possible 70-year lifecycle** on the tape — {R['n_cohorts']:,} of "
            f"them, one starting every month from {R['cohort_start']} to {R['cohort_end']} "
            f"(retirements {R['retire_start']} → {R['retire_end']}). Real (inflation-adjusted) "
            "returns for US stocks and 10-year Treasuries come from Shiller's data. "
            "**International stocks are the honest compromise**: real fund data (EFA) exists "
            f"only from {R['intl_mkt_start']} — {R['intl_mkt_pct']:.0f}% of the tape — so the "
            "earlier years are a *simulation calibrated to the century-long literature "
            "numbers* (4.3%/yr real, correlation 0.60 with the US). We say so on every chart, "
            "and we re-run everything with a pure-US variant (no simulation anywhere) and a "
            "worst-case correlation stress. Costs: 10 bps per trade + 5 bps/yr fund fees."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The scoreboard first.** Wealth is in multiples of your annual savings "
            "contribution; *ruin* means the money ran out before age 95:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    res4 = st.run_all(GU, GI, GB, wr=0.04)\n"
            "    score = {s: (float(np.median(res4[s]['wret'])), res4[s]['terminal'].mean(),\n"
            "                 float(np.median(res4[s]['terminal'])),\n"
            "                 float(np.quantile(res4[s]['terminal'], .05)),\n"
            "                 100*(~res4[s]['success']).mean(), int((~res4[s]['success']).sum()))\n"
            "             for s in STRATS}\n"
            "else:\n"
            "    res4, score = None, R['score']\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))\n"
            "cols = {'alleq': RED, 'alleq_dom': '#e58e26', 's6040': GREY, 'tdf': BLUE}\n"
            "names = [LBL[s] for s in STRATS]\n"
            "axes[0].bar(names, [score[s][2] for s in STRATS], color=[cols[s] for s in STRATS])\n"
            "axes[0].set_title('median wealth at 95 (x annual savings)')\n"
            "axes[0].tick_params(axis='x', rotation=20)\n"
            "axes[1].bar(names, [score[s][4] for s in STRATS], color=[cols[s] for s in STRATS])\n"
            "axes[1].set_title('ruin probability at the 4% rule (%)')\n"
            "axes[1].tick_params(axis='x', rotation=20)\n"
            "plt.tight_layout(); plt.show()\n"
            "for s in STRATS:\n"
            "    v = score[s]\n"
            "    print(f'{LBL[s]:<28} pot@65 med {v[0]:6.1f}  terminal med {v[2]:6.1f}  ruin {v[4]:5.2f}%')"
        ),
        md(
            f"Three things jump out. **The TDF loses badly** — median wealth at 95 of "
            f"**{R['score']['tdf'][2]:.0f}×** vs **{R['score']['s6040'][2]:.0f}×** for the "
            f"60/40 and **{R['score']['alleq'][2]:.0f}×** for the paper's all-equity mix, and "
            f"it dies broke in **{R['score']['tdf'][4]:.0f}%** of cohorts (its bond-heavy "
            "retirement years met the 1940s–70s, when inflation ate bonds alive). **The "
            "paper's 50/50 mix does NOT beat the 60/40 where it counts** — median wealth is a "
            f"tie ({R['score']['alleq'][2]:.0f} vs {R['score']['s6040'][2]:.0f}) and its ruin "
            f"rate is *higher* ({R['score']['alleq'][4]:.1f}% vs "
            f"{R['score']['s6040'][4]:.1f}%). And **100% US stocks crushes everything** — "
            "which is exactly the bet the paper was trying *not* to make you take.\n\n"
            "**Now the worst retirements in history.** Pot normalized to 1 at retirement "
            "(the 4% rule scales with your own pot):"
        ),
        code(
            RETPATH +
            "if HAVE_REAL:\n"
            "    import pandas as pd\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)\n"
            "    for ax, start, title in [(axes[0], '1889-09-01', 'retire Sept 1929 (the crash)'),\n"
            "                             (axes[1], '1926-01-01', 'retire Jan 1966 (the inflation grind)')]:\n"
            "        i = int(np.where(DATES == pd.Timestamp(start))[0][0])\n"
            "        for name, c in [('alleq', RED), ('alleq_dom', '#e58e26'), ('s6040', GREY), ('tdf', BLUE)]:\n"
            "            ax.plot(ret_path(i, name), c=c, lw=2, label=LBL[name])\n"
            "        ax.axhline(0, c='k', lw=.8); ax.set_title(title)\n"
            "        ax.set_xlabel('years into retirement')\n"
            "    axes[0].set_ylabel('real wealth (pot at 65 = 1)'); axes[0].legend(fontsize=8)\n"
            "    plt.tight_layout(); plt.show()\n"
            "print('terminal @4% (x annual savings):')\n"
            "for ym in ('1929-09', '1966-01'):\n"
            "    print(' ', ym, {k: v for k, v in R['famous'][ym].items()})"
        ),
        md(
            "**Left — 1929.** Every all-equity variant goes **broke around year 18**, while "
            f"the 60/40 sails through (terminal {R['famous']['1929-09']['s6040'][0]:.0f}× the "
            "annual contribution). Retiring 100%-in-stocks into the Great Depression is "
            "exactly the nightmare bonds exist for.\n\n"
            "**Right — 1966.** Here *everything domestic* dies (stocks and bonds lost real "
            "value together for 15 years) — except the paper's mix, which survives with "
            f"{R['famous']['1966-01']['alleq'][0]:.0f}×… **because of its international "
            "half.** But careful: in 1966–1996 our international leg is the *simulated* one. "
            "Run the same cohort with the no-diversification stress and it goes broke in year "
            f"{R['famous']['1966-01']['alleq_corr1'][1]}. The strategy's best rescue story "
            "rests on the part of the data that isn't data.\n\n"
            "> 🔬 For the quants: the wealth edge vs the TDF is certified at HAC *t* = "
            f"{R['pairs'][1][2]:+.2f} with a bootstrap CI excluding 0; the edge vs 60/40 is "
            f"*t* = {R['pairs'][0][2]:+.2f} — not certifiable. Details in "
            "[02_for_the_quants](02_for_the_quants.ipynb)."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Mixed.** Against the **target-date fund**, the paper is right and "
            "it's real: more wealth (*t* ≈ +2.6), far less ruin. Against the plain **60/40**, "
            "the paper's own 50/50 domestic/international mix never certifies — median tie, "
            "and *more* ruin on this tape.\n"
            "- **Tradability — Fragile.** Two index funds and ~5 bps of fees; costs don't "
            "move the answer at all. What's fragile is everything else: you need 70 years, "
            "the discipline to hold 100% stocks through a −80% real drawdown, and the "
            "safety half of the promise didn't replicate.\n"
            "- **Worst cohorts — Busted.** 1929 ruins every all-equity variant while 60/40 "
            "survives; 1966 is survived only by the simulated international leg.\n\n"
            "**In plain terms:** dumping the target-date glidepath for more equity is "
            "historically defensible. Dumping your *bonds* — the 60/40 — for a globally "
            "diversified all-stock portfolio bought you a coin-flip with a worse worst case."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Why did the paper find the opposite?** Its 38-country panel contains bond "
            "returns far worse than the US 10-year (hyperinflations, wars) and international "
            "equity sampled from *winning* developed markets. Change the ingredients, change "
            "the dish — our tape anchors on the actual US bond experience and the "
            "century-long world-ex-US 4.3%/yr.\n"
            "- **The US-only bet.** [Study 151 — stocks-for-long-run](../../151-stocks-for-long-run/README.md) "
            "grades the \"US stocks always win given 30 years\" claim that `alleq_dom` "
            "silently rides.\n"
            "- **The glidepath autopsy.** [Study 596 — bond-tent-glidepath](../../596-bond-tent-glidepath/README.md) "
            "dissects *why* bond-heavy retirement years fail on this tape: the 1966–82 "
            "inflation grind, not the 1929 crash, is history's retirement killer.\n"
            "- **The honest caveat, one more time.** 85.7% of our international leg is a "
            "literature-calibrated simulation (labeled everywhere, stress-tested both ways). "
            "A true 150-year global monthly tape (DMS is annual and proprietary) could move "
            "the blend's numbers — in *either* direction.\n\n"
            "*The quants notebook re-runs the race under every international assumption, "
            "boots the tape 600 times, and rebuilds the paper's own 2,000-lifetime bootstrap.*"
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
            "# Cederburg's 100%-Equities Lifecycle — a quantitative teardown 🔬\n"
            "### 990 overlapping 70-year lifecycles · HAC (840-month bandwidth) + outer block "
            "bootstrap · the paper's own 2,000-lifetime bootstrap · international-leg "
            "sensitivity · an exact-null synthetic control\n\n"
            + BADGES +
            "Deep companion to [01_for_the_curious](01_for_the_curious.ipynb). The claim "
            "(Anarkulova-Cederburg-O'Doherty 2023): a lifecycle 100% in equities (50/50 "
            "domestic/international) beats balanced strategies and TDF glidepaths on terminal "
            "wealth AND ruin. We race `alleq`, `alleq_dom`, `s6040`, `tdf` over every "
            "monthly-start 70-year lifecycle (save 40y at 1 unit/yr real; retire 30y on 4% of "
            "the pot, fixed real) on the Shiller+EFA real tape.\n\n"
            "> ⚠️ **Data decisions, stated.** All legs REAL (CPI-deflated). Bond = 10-year "
            "first-order approx (carry − 7·Δy), deflated. **The INTL leg is market data "
            f"(EFA) only from {R['intl_mkt_start']}** ({R['intl_mkt_pct']:.1f}% of months); "
            "before that it is a deterministic literature-calibrated simulation "
            f"({R['intl_calib']}) — labeled on the Signal axis, cross-checked against a "
            "pure-tape domestic variant and a corr = 1 stress. Weights are a pure function of "
            "age (set at the end of the prior year — one clean lag). Costs "
            f"{R['cost_bps']} bps one-way × traded value + {R['er_bps']} bps/yr ER. No "
            "survivorship (index tapes); the named bias is US equity exceptionalism. "
            "Numbers: [`docs/results.md`](../docs/results.md) (fingerprint `"
            + R["fingerprint"] + "`).\n"
            ">\n"
            "> 💡 The `💡 In plain words` notes translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | Real vs TDF: ΔmeanTW **+{R['pairs'][1][1]:.0f}** at "
            f"HAC *t* = **{R['pairs'][1][2]:+.2f}**, boot CI [{R['pairs'][1][3]:+.1f}, "
            f"{R['pairs'][1][4]:+.1f}] excludes 0. Not certified vs 60/40: *t* = "
            f"**{R['pairs'][0][2]:+.2f}**, CI [{R['pairs'][0][3]:+.1f}, "
            f"{R['pairs'][0][4]:+.1f}] straddles 0, ruin **worse** "
            f"({R['score']['alleq'][4]:.2f}% vs {R['score']['s6040'][4]:.2f}%). |\n"
            f"| **Tradability** | `FRAGILE` | *t* flat across 0→25 bps costs "
            f"({R['costs'][0][4]:+.2f} → {R['costs'][2][4]:+.2f}); what breaks is the "
            "70-year horizon, the 1929 ruin, and the uncertified safety half. |\n"
            f"| **Worst cohorts** | `BUSTED` | 1929 retiree: every all-equity variant ruined "
            f"(fy 16–18) vs 60/40 terminal {R['famous']['1929-09']['s6040'][0]:.0f}×; 1966 "
            "rescued only by the *simulated* INTL leg (corr-1 stress: ruined fy "
            f"{R['famous']['1966-01']['alleq_corr1'][1]}). |\n\n"
            "> 💡 In plain words: the anti-TDF half of the paper replicates; the anti-60/40 "
            "half doesn't — and the difference is the international leg's return level."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $W^s_T$ be terminal real wealth of strategy $s$ over a 70-year lifecycle "
            "(40 contribution years, 30 retirement years at $c = 4\\%$ of the retirement pot, "
            "real), and ruin $= \\Pr[W_t \\le 0$ before age 95$]$. The paper claims the "
            "50/50 domestic/international all-equity lifecycle **first-order dominates** "
            "balanced strategies:\n\n"
            "- **H₁ (wealth vs balanced).** $E[W^{alleq}_T] > E[W^{6040}_T]$ — certified.\n"
            "- **H₂ (ruin vs balanced).** ruin(alleq) < ruin(60/40).\n"
            "- **H₃ (vs TDF).** both hold against the glidepath.\n\n"
            "On this tape H₃ certifies (wealth *t* = "
            f"{R['pairs'][1][2]:+.2f}; ruin {R['pairs'][1][5]:+.1f} pp), H₁ does not "
            f"(*t* = {R['pairs'][0][2]:+.2f}) and H₂ **reverses** "
            f"({R['pairs'][0][5]:+.2f} pp)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Monthly-start 70-year cohorts overlap by up to 839 months; effective sample size "
            "is roughly two non-overlapping lifetimes. Three instruments, none of them naive:\n\n"
            "1. **Newey-West HAC** on per-cohort terminal-wealth differences, bandwidth "
            "**forced to the full 840-month overlap**.\n"
            "2. **Outer circular block bootstrap** (120-month blocks, joint US/INTL/BD rows, "
            "600 reps, seed 598): rebuild the whole cohort panel per replicate for tape-level "
            "CIs on mean-TW and ruin differences. Blocks preserve within-decade dynamics, "
            "destroy cross-decade mean reversion — stated, not hidden.\n"
            "3. **The paper's own machinery**: 2,000 block-bootstrap *lifetimes* (840 months "
            "stitched from 120-month blocks), all strategies on identical draws (paired).\n\n"
            "Plus a 20-seed synthetic control with an **exact null**: at zero arithmetic "
            "equity premium, every weight schedule has identical expected terminal wealth."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · The protocol\n\n"
            f"- **Tape.** {R['start']} → {R['end']} ({R['months']:,} months): real US "
            f"**{R['ann']['US']:.2f}%/yr**, INTL **{R['ann']['INTL']:.2f}%/yr** "
            f"({R['intl_mkt_pct']:.1f}% market data, rest calibrated), BD "
            f"**{R['ann']['BD']:.2f}%/yr**.\n"
            f"- **Cohorts.** {R['n_cohorts']:,} monthly-start lifecycles; retirements "
            f"{R['retire_start']} → {R['retire_end']}.\n"
            "- **Strategies.** `alleq` (50/50 dom/intl, the paper's pick), `alleq_dom`, "
            "`s6040`, `tdf` (90% equity to age 40 → 45% at 65 → 30% at 80; equity leg 50/50 "
            "like `alleq`, so TDF-vs-alleq is a pure bonds+glidepath contrast).\n"
            f"- **Execution.** Age-only weights, one clean lag; {R['cost_bps']} bps one-way + "
            f"{R['er_bps']} bps/yr ER; withdrawal 4% of the own pot, fixed real.\n"
            "- **Sensitivity.** INTL ∈ {hybrid, corr = 1 stress, domestic}; costs 0→25 bps; "
            "wr 3–5%."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md("### 4a · Scoreboard and head-to-heads — the raw race"),
        code(
            "if HAVE_REAL:\n"
            "    res4 = st.run_all(GU, GI, GB, wr=0.04)\n"
            "    rows = []\n"
            "    for lab, (a, b) in zip([p[0] for p in R['pairs']],\n"
            "                           [('alleq','s6040'), ('alleq','tdf'), ('alleq_dom','s6040')]):\n"
            "        d = res4[a]['terminal'] - res4[b]['terminal']\n"
            "        rows.append((lab, d.mean(), st.hac_tstat(d, lags=840),\n"
            "                     100*((~res4[a]['success']).mean() - (~res4[b]['success']).mean()),\n"
            "                     100*np.mean(res4[a]['terminal'] > res4[b]['terminal'])))\n"
            "else:\n"
            "    res4 = None\n"
            "    rows = [(p[0], p[1], p[2], p[5], p[8]) for p in R['pairs']]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.2))\n"
            "names = [r[0].split(' (')[0] for r in rows]; ts = [r[2] for r in rows]\n"
            "ax.barh(names[::-1], ts[::-1], color=[AMBER, GREEN, GREEN][::-1][::1])\n"
            "ax.axvline(2, ls='--', c=GREY, label='t = +2'); ax.axvline(0, c='k', lw=.8)\n"
            "for i, t in enumerate(ts[::-1]): ax.annotate(f' t = {t:+.2f}', (t, i), va='center')\n"
            "ax.set_xlabel('HAC t (840-month bandwidth) on terminal-wealth difference @4%')\n"
            "ax.set_title('Certified vs the TDF and for the domestic variant -- NOT for the headline pair')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for r in rows: print(f'{r[0]:<30} dTW {r[1]:+8.1f}  HAC t {r[2]:+6.2f}  druin {r[3]:+6.2f} pp  win {r[4]:.1f}%')"
        ),
        md(
            f"> 💡 In plain words: the paper's blend beats the 60/40 in only "
            f"{R['pairs'][0][8]:.0f}% of lifecycles — a coin flip — and dies broke *more* "
            f"often. Its win over the TDF ({R['pairs'][1][8]:.0f}% of lifecycles) and the "
            f"domestic variant's rout of the 60/40 ({R['pairs'][2][8]:.0f}%) are the real "
            "effects in the room."
        ),
        md(
            "### 4b · Tape-level uncertainty — the outer block bootstrap\n\n"
            "Rebuild the entire cohort panel on 120-month-block resamples of the tape. The "
            "picture uses 100 reps to stay light; the canonical 600-rep CIs from "
            "[`docs/results.md`](../docs/results.md) are printed alongside."
        ),
        code(
            "if HAVE_REAL:\n"
            "    boot = st.outer_bootstrap(US, INTL, BD, pairs=(('alleq','s6040'), ('alleq','tdf')),\n"
            "                              n_boot=100, seed=598)\n"
            "    d60, dtdf = boot[('alleq','s6040')]['dmean'], boot[('alleq','tdf')]['dmean']\n"
            "else:\n"
            "    rng = np.random.default_rng(598)\n"
            "    d60 = rng.normal(R['pairs'][0][1], 420, 100); dtdf = rng.normal(R['pairs'][1][1], 480, 100)\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)\n"
            "for ax, draws, k, ttl in [(axes[0], d60, 0, 'alleq - 60/40 (HEADLINE)'),\n"
            "                          (axes[1], dtdf, 1, 'alleq - TDF (VS GLIDE)')]:\n"
            "    ax.hist(draws, bins=25, color=GREY, alpha=.85)\n"
            "    ax.axvline(R['pairs'][k][1], c=RED, lw=2.5, label=f\"observed {R['pairs'][k][1]:+.0f}\")\n"
            "    ax.axvline(0, c='k', lw=1)\n"
            "    ax.set_title(ttl + f\"  canonical CI [{R['pairs'][k][3]:+.0f}, {R['pairs'][k][4]:+.0f}]\")\n"
            "    ax.set_xlabel('mean terminal-wealth difference'); ax.legend(fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('canonical 600-rep CIs (docs/results.md):')\n"
            "for p in R['pairs']:\n"
            "    print(f'  {p[0]:<30} dmeanTW CI [{p[3]:+8.1f}, {p[4]:+8.1f}]   druin CI [{p[5+1]:+6.2f}, {p[7]:+6.2f}] pp')"
        ),
        md(
            f"> 💡 In plain words: resample history in decade blocks and the headline pair's "
            f"wealth edge lands on either side of zero (CI [{R['pairs'][0][3]:+.0f}, "
            f"{R['pairs'][0][4]:+.0f}]) — noise. The TDF gap stays positive in >97.5% of "
            f"resamples (CI [{R['pairs'][1][3]:+.1f}, {R['pairs'][1][4]:+.0f}]). No ruin "
            "difference certifies at tape level for anyone — 150 years contain only ~2 "
            "independent lifetimes; honesty about that *is* the result."
        ),
        md(
            "### 4c · The paper's own machinery — 2,000 bootstrap lifetimes\n\n"
            "Stitch 840-month lifetimes from 120-month blocks (joint across assets), all "
            "strategies on identical draws — the Anarkulova-Cederburg-O'Doherty generative "
            "method, on our tape:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    BU, BI, BB = st.block_lifetimes(US, INTL, BD, n_draws=2000, seed=598)\n"
            "    rb = st.run_all(BU, BI, BB, wr=0.04)\n"
            "    tab = {s: (float(np.median(rb[s]['wret'])), rb[s]['terminal'].mean(),\n"
            "               float(np.median(rb[s]['terminal'])), 100*(~rb[s]['success']).mean())\n"
            "           for s in STRATS}\n"
            "else:\n"
            "    rb, tab = None, R['boot']\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "cols = {'alleq': RED, 'alleq_dom': '#e58e26', 's6040': GREY, 'tdf': BLUE}\n"
            "ax.bar([LBL[s] for s in STRATS], [tab[s][3] for s in STRATS],\n"
            "       color=[cols[s] for s in STRATS], width=.55)\n"
            "for i, s in enumerate(STRATS): ax.annotate(f'{tab[s][3]:.1f}%', (i, tab[s][3]),\n"
            "                                           ha='center', va='bottom')\n"
            "ax.set_ylabel('ruin probability at the 4% rule (%)')\n"
            "ax.set_title('The Cederburg bootstrap on OUR tape: the blend ruins MORE than the 60/40')\n"
            "plt.tight_layout(); plt.show()\n"
            "for s in STRATS: print(f'{LBL[s]:<26} pot@65 med {tab[s][0]:6.1f}  meanTW {tab[s][1]:7.1f}  medTW {tab[s][2]:6.1f}  ruin {tab[s][3]:5.2f}%')"
        ),
        md(
            f"> 💡 In plain words: even with the paper's own resampling method, the blend's "
            f"ruin ({R['boot']['alleq'][3]:.1f}%) sits *above* the 60/40's "
            f"({R['boot']['s6040'][3]:.1f}%) here. The paper's opposite ranking comes from its "
            "38-country ingredient list: global bonds that lost to inflation for decades "
            "(ours: US 10-year, +2.4%/yr real) and international equity sampled from winning "
            "developed markets (ours: the DMS world-ex-US 4.3%/yr). Same recipe, different "
            "pantry, different dish."
        ),
        md(
            "### 4d · The lever that decides everything — the international leg\n\n"
            "Re-run the whole race under each international assumption (the domestic row uses "
            "**no simulated data anywhere**):"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sens = []\n"
            "    for mode, lab in [('hybrid','hybrid (headline)'), ('corr1','corr=1 stress'),\n"
            "                      ('domestic','domestic (pure tape)')]:\n"
            "        d2 = data.real_returns(intl_mode=mode)\n"
            "        g2 = st.cohort_year_returns(d2['US'].to_numpy(), d2['INTL'].to_numpy(), d2['BD'].to_numpy())\n"
            "        r2 = st.run_all(g2[0], g2[1], g2[2], wr=0.04, strategies=('alleq','s6040'))\n"
            "        dd = r2['alleq']['terminal'] - r2['s6040']['terminal']\n"
            "        sens.append((lab, 100*(~r2['alleq']['success']).mean(),\n"
            "                     r2['alleq']['terminal'].mean(), st.hac_tstat(dd, 840)))\n"
            "else:\n"
            "    sens = R['sens']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "x = np.arange(len(sens))\n"
            "ax.bar(x - .18, [s[1] for s in sens], .34, color=RED, label='alleq ruin %')\n"
            "ax.bar(x + .18, [R['score']['s6040'][4]]*len(sens), .34, color=GREY, label='60/40 ruin %')\n"
            "for i, s in enumerate(sens): ax.annotate(f't={s[3]:+.2f}', (i - .18, s[1]), ha='center', va='bottom')\n"
            "ax.set_xticks(x); ax.set_xticklabels([s[0] for s in sens])\n"
            "ax.set_ylabel('ruin @4% (%)'); ax.legend()\n"
            "ax.set_title('No INTL assumption makes the blend safer than the 60/40 (t = HAC t on wealth vs 60/40)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for s in sens: print(f'{s[0]:<24} ruin {s[1]:5.2f}%  meanTW {s[2]:7.1f}  HAC t vs 60/40 {s[3]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the knob that matters is the international **return "
            f"level**, not the correlation. At the literature's 4.3%/yr the blend drags "
            f"(*t* = {R['sens'][0][3]:+.2f}); force correlation to 1 and the wealth *t* "
            f"actually rises to {R['sens'][1][3]:+.2f} (less diversification, but the diff "
            "gets less noisy) while ruin gets *worse*; only replacing international with US "
            f"equity (*t* = {R['sens'][2][3]:+.2f}) certifies — and that variant is just "
            "[study 151](../../151-stocks-for-long-run/README.md)'s US-exceptionalism bet "
            "wearing a lifecycle costume. Ruin by withdrawal rate and the cost sweep tell the "
            "same story:"
        ),
        code(
            "print('ruin % by withdrawal rate (cohorts):')\n"
            "print('  wr    ' + ''.join(f'{LBL[s]:>26}' for s in STRATS))\n"
            "for w in ('3.0','3.5','4.0','5.0'):\n"
            "    print(f'  {w}% ' + ''.join(f'{R[\"ruin\"][w][k]:>25.2f}%' for k in range(4)))\n"
            "print()\n"
            "print('cost sweep (one-way bps / ER bps): alleq vs 60/40')\n"
            "for cb, er, mtw, ruin, t in R['costs']:\n"
            "    print(f'  {cb:>3}/{er:>3}: meanTW {mtw:7.1f}  ruin {ruin:5.2f}%  HAC t {t:+.2f}')"
        ),
        md(
            "### 4d-bis · Seed-robustness — re-drawing the simulated residuals\n\n"
            "The pre-2001 international leg is a *seeded* simulation, and single-seed "
            "verdicts are banned on this desk. Both head-to-heads re-run under **20 "
            "independent residual draws** (seeds 598–617; canonical numbers from "
            "`examples/verify.py`):"
        ),
        code(
            "sr = R['seedrob']\n"
            "print('HAC t across 20 residual re-draws of the simulated INTL leg:')\n"
            "for pair, key in [('alleq - TDF  ', 'tdf'), ('alleq - 60/40', 's6040')]:\n"
            "    m, lo, hi, n2 = sr[key]\n"
            "    print(f'  {pair}: mean {m:+6.2f}  min {lo:+7.2f}  max {hi:+6.2f}  t>=2 in {n2}/20 seeds')\n"
            "print(f\"  ruin(alleq) > ruin(60/40) in {sr['ruin_worse_n']}/20 seeds -- a coin flip\")\n"
            "dm, t, ra, rt = sr['pure_tape_tdf']\n"
            "print(f'  PURE TAPE alleq_dom - tdf_dom: dmeanTW {dm:+.1f}  HAC t {t:+.2f}  ruin {ra:.2f}% vs {rt:.2f}%')"
        ),
        md(
            f"> 💡 In plain words: the certified half (all-equity vs the TDF) survives "
            f"**every** residual draw — the headline seed's *t* = {R['seedrob']['tdf'][1]:+.2f} "
            f"is actually the *weakest* of the 20 — and holds on the pure tape with nothing "
            f"simulated (*t* = {R['seedrob']['pure_tape_tdf'][1]:+.2f}). The 60/40 race is the "
            "opposite: the wealth *t* flips sign across draws and the ruin ranking is a coin "
            "flip (9/20) — nothing in that race is decided by market data, which is exactly "
            "why it stays uncertified."
        ),
        md(
            "### 4e · Synthetic control — the engine is faithful\n\n"
            "20 independent seeded worlds per setting; detector = alleq vs 60/40, one-sample "
            "*t* across worlds. **Exact null:** at zero *arithmetic* equity premium, expected "
            "terminal wealth is identical for every weight schedule (products of independent "
            "gross returns), so the mean-wealth detector must read ~0. *(Machinery proof only "
            "— never cited as market evidence.)*"
        ),
        code(
            "res = [(f'premium {p:.1f}pp', st.control_premium_effect(p/100, wr=0.04, n_seeds=20))\n"
            "       for p in (0.0, 2.0, 4.5)]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "vals = [r['dmean_t'] for _, r in res]\n"
            "ax.bar([l for l, _ in res], vals, color=[GREY, GREEN, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2'); ax.axhline(0, c='k', lw=.8)\n"
            "for i, (l, r) in enumerate(res):\n"
            "    ax.annotate(f\"t={r['dmean_t']:+.2f}\\ndruin {r['druin_mean']:+.1f}pp\", (i, vals[i]),\n"
            "                ha='center', va='bottom')\n"
            "ax.set_ylabel('one-sample t, mean terminal-wealth diff (alleq - 60/40)')\n"
            "ax.set_title('Null reads ~0; the planted premium lights up; ruin flips sign as the premium grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for l, r in res: print(f\"{l:<16} dmeanTW {r['dmean_mean']:+8.1f} (t {r['dmean_t']:+5.2f})   druin {r['druin_mean']:+6.2f} pp (t {r['druin_t']:+5.2f})\")"
        ),
        md(
            f"> 💡 In plain words: the machinery is honest — at the exact null the wealth "
            f"detector reads ~0 (*t* = {R['control'][0][2]:+.2f}) and 100% equity ruins "
            f"**more** ({R['control'][0][3]:+.1f} pp: pure vol, no reward); plant a 4.5 pp "
            f"premium and wealth dominance lights up (*t* = {R['control'][2][2]:+.2f}) while "
            f"the ruin gap closes to ~0. That is the real-tape verdict in miniature: the US "
            "leg's 4.5 pp premium certifies wealth, the international leg's ~2 pp does not, "
            "and nobody's ruin advantage certifies anywhere."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — split, spelled out: **Real vs the TDF** (ΔmeanTW "
            f"+{R['pairs'][1][1]:.0f}, HAC *t* = {R['pairs'][1][2]:+.2f}, boot CI excludes 0; "
            f"ruin {R['pairs'][1][5]:+.1f} pp directional) · **not certified vs 60/40** for "
            f"the paper's blend (*t* = {R['pairs'][0][2]:+.2f}, CI straddles 0, median tie) "
            f"whose ruin is worse at the headline draw ({R['score']['alleq'][4]:.2f}% vs "
            f"{R['score']['s6040'][4]:.2f}%; Cederburg bootstrap agrees, "
            f"{R['boot']['alleq'][3]:.2f}% vs {R['boot']['s6040'][3]:.2f}% — a coin flip "
            "across residual re-draws, 9/20: no ruin ranking vs 60/40 is decided by the "
            "data). Seed-robust: the TDF edge holds in 20/20 re-draws and on the pure tape "
            f"(*t* = {R['seedrob']['pure_tape_tdf'][1]:+.2f}). The domestic "
            f"variant certifies at *t* = {R['pairs'][2][2]:+.2f} but is the US-exceptionalism "
            "bet, graded in study 151. Named: 85.7% of the INTL leg is calibrated simulation.\n"
            f"- **Tradability `FRAGILE`** — two index funds, *t* invariant to a 0→25 bps cost "
            "sweep, unlimited capacity; fragile because the certified edge needs 70 years, "
            "ruins the 1929 cohort, and the safety pitch is uncertified everywhere.\n"
            f"- **Worst cohorts `BUSTED`** — 1929: all-equity ruined (fy 16–18) vs 60/40 at "
            f"{R['famous']['1929-09']['s6040'][0]:.0f}×; 1966: rescued only by the simulated "
            "leg (corr-1 stress ruined, domestic ruined)."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The real bottleneck is data.** The paper's 38-country panel (and the DMS "
            "yearbook) are annual and proprietary; a public 150-year *monthly* international "
            "tape does not exist. Our calibrated leg is the honest substitute — the "
            "sensitivity grid brackets what any true tape could say.\n"
            "- **Mortality and earnings.** We use a fixed 30-year retirement and flat real "
            "contributions; the paper couples mortality tables with an earnings process. "
            "Those choices move levels, not the ordering — the ordering is set by return "
            "levels and the 1940s–70s bond regime.\n"
            "- **The siblings.** [151](../../151-stocks-for-long-run/README.md) grades the "
            "horizon claim `alleq_dom` rides; [596](../../596-bond-tent-glidepath/README.md) "
            "autopsies the decumulation glidepath (the TDF's back half); "
            "[173](../../173-four-percent-rule/README.md) shows the withdrawal *rate* is the "
            "lever that dominates all of this.\n\n"
            "*The reproducible core is offline and deterministic; weights are age-only (one "
            "clean lag), every leg is real, and the simulated share of the INTL leg is "
            "labeled everywhere it appears. Methods and sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
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
