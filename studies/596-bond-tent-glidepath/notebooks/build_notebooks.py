"""Generate the two narrative notebooks for Study 596 (Bond Tent / Rising Equity Glidepath).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached Shiller
parquet under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic control runs anywhere with no network.
Heavy cells are kept light (bootstrap 200 reps for the picture; the canonical
1,000-rep CIs are quoted from ``R``).
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
# (Shiller 1871-02 -> 2023-06, 1,470 monthly-start 30y cohorts, cost 10 bps,
#  bootstrap = 1,000 reps / 120-month blocks / seed 596, HAC bandwidth 360)
STRATS = ["s6040", "s5050", "s4060", "rising", "declining"]
R = dict(
    start="1871-02", end="2023-06", months=1829, fingerprint="7246f42140ba",
    ann_eq=6.90, ann_bd=2.39, premium=4.52,
    n_cohorts=1470, cohort_start="1871-02", cohort_end="1993-07", cost_bps=10,
    strats=STRATS,
    # success rates (%) by wr, order = strats
    succ={"4.0": [96.33, 95.85, 93.81, 94.01, 95.85],
          "4.5": [88.71, 85.17, 77.82, 79.86, 86.05],
          "5.0": [75.78, 69.80, 59.86, 60.41, 73.95],
          "5.5": [63.40, 57.21, 51.84, 52.65, 62.65]},
    # terminal wealth @4%: (mean, median, p05, n_fail), order = strats
    tw=[(1.880, 1.332, 0.146, 54), (1.555, 0.985, 0.041, 61),
        (1.260, 0.698, 0.000, 91), (1.465, 0.824, 0.000, 88),
        (1.624, 1.115, 0.095, 61)],
    # decomposition: (label, dmeanTW@4, HAC t, ci_lo, ci_hi, dsucc4, dsucc5, ci5_lo, ci5_hi)
    decomp=[("rising - 60/40 (HEADLINE)", -0.4154, -6.11, -1.429, -0.192, -2.31, -15.37, -20.00, 1.36),
            ("rising - 50/50 (SHAPE)",    -0.0904, -1.85, -0.328, -0.024, -1.84,  -9.39, -11.43, 1.23),
            ("rising - declining (TIMING)", -0.1592, -1.53, -0.619, -0.035, -1.84, -13.54, -17.96, 3.13),
            ("50/50 - 60/40 (ALLOCATION)", -0.3250, -5.92, -1.126, -0.128, -0.48,  -5.99,  -8.91, 0.55)],
    safemax=[3.68, 3.61, 3.53, 3.57, 3.62],                      # order = strats
    cape=dict(n_valid=1351, n_q4=338,
              rows={"4.0": [84.0, 82.0, 73.1, 74.3, 82.0],
                    "4.5": [58.6, 50.3, 37.6, 39.6, 51.8],
                    "5.0": [35.8, 30.5, 26.0, 27.2, 31.7]}),
    famous={"1929-09": [0.390, 0.463, 0.466, 1.251, 0.000],
            "1966-01": [0.000, 0.000, 0.000, 0.000, 0.000],
            "1972-12": [0.223, 0.172, 0.110, 0.240, 0.068]},
    rescue=dict(n_fail_6040=54, band_6040="1964-1969", rising_rescue_pct=5.6,
                n_fail_rising=88, band_rising="1961-1969"),
    # robustness: (variant label, succ@4, succ@5, meanTW@4, HAC t vs 60/40)
    robust=[("rising 30-70 (headline)", 94.01, 60.41, 1.465, -6.11),
            ("rising 30-60 (Kitces-Pfau)", 93.61, 58.64, 1.338, -6.75),
            ("rising 30-70 by yr15 (fast)", 95.71, 68.84, 1.752, -2.25),
            ("rising 20-80 (deep tent)", 93.47, 57.21, 1.411, -5.67)],
    costs={"0": (96.33, 94.01, 95.85), "10": (96.33, 94.01, 95.85),
           "25": (96.26, 93.95, 95.71)},
    # synthetic control: (label, dsucc pp, t or None, dp05, dp05_t or None)
    control=[("EXACT NULL: wr=0, iid, prem 4.5pp", 0.00, None, -0.0308, -1.05),
             ("TENT WORLD: wr=4%, iid, prem 0pp", 5.99, 5.65, 0.0, None),
             ("wr=4%, prem 2pp", 4.50, 4.02, 0.0020, 1.00),
             ("wr=4%, prem 4.5pp (historical)", 2.54, 2.92, 0.0289, 1.59),
             ("wr=5%, prem 4.5pp (historical)", -3.86, -3.43, 0.0, None)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Timing_or_less_equity%3F: Busted](https://img.shields.io/badge/Timing_or_less_equity%3F-Busted-8b949e?style=flat-square)\n\n"
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

from bond_tent_glidepath import data, strategy as st

HAVE_REAL = os.path.exists(data.SHILLER_PATH) or any(os.path.exists(p) for p in data._FALLBACKS)
if HAVE_REAL:
    DF = data.real_returns()
    EQ, BD = DF["EQ"].to_numpy(), DF["BD"].to_numpy()
    GE, GB, STARTS = st.cohort_year_returns(EQ, BD)
    DATES = DF.index[STARTS]
    print("real tape:", DF.index.min().date(), "->", DF.index.max().date(),
          "| cohorts:", GE.shape[0], "| fingerprint:", data.fingerprint(DF))
else:
    DF = GE = GB = DATES = None
    print("no Shiller cache -- real-tape cells quote the frozen numbers in R")
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nSTRATS = " + repr(STRATS) + "\nR = " + repr(R) + "\n"

LBL = {"s6040": "static 60/40", "s5050": "static 50/50", "s4060": "static 40/60",
       "rising": "rising tent 30→70", "declining": "declining 70→30"}


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Should you retire with MORE bonds, then buy back the stocks? ⛺\n"
            "### The \"bond tent\" (rising equity glidepath) — a famous retirement recipe, measured on 152 years of data\n\n"
            + BADGES +
            "Here's advice you will hear from serious people (it comes from a real 2014 academic "
            "paper by Wade Pfau and Michael Kitces): the most dangerous moment of your financial "
            "life is the **first decade after you retire**. A market crash then — while you're "
            "selling shares every year to live — can sink you, even if the market recovers later. "
            "That's *sequence-of-returns risk*. Their fix: retire holding **mostly bonds** (say, "
            "only 30% stocks) and then **buy the stocks back gradually** over retirement, up to "
            "60–70%. Plotted over your whole life, your bond holdings look like a **tent** with "
            "its peak on retirement day.\n\n"
            "It sounds airtight. We put it on the full Shiller dataset — every possible 30-year "
            "retirement since 1871 — with the standard 4%-rule withdrawals.\n\n"
            "> 📓 Want the *t*-stats, the bootstrap and the decomposition? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**. Numbers quoted here are the "
            "frozen headline run ([docs/results.md](../docs/results.md)); every chart is drawn by "
            "the code beside it."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the bond tent make a 4%-rule retirement safer? | **No.** Across 1,470 "
            "historical 30-year retirements it *fails more often* than a plain 60/40 "
            f"(**{R['succ']['4.0'][3]:.1f}%** vs **{R['succ']['4.0'][0]:.1f}%** success at 4% "
            "withdrawals — and the gap explodes at 5%). |\n"
            "| Is that just because it holds less stock on average? | **Not only.** Even against "
            "a 50/50 portfolio with the *same* average stock holding, the tent loses. It even "
            "loses to its own **mirror image** (start stock-heavy, glide *down*). |\n"
            "| Does it ever work? | **Yes — in 1929.** Retire into a short, sharp crash where "
            f"bonds hold their value, and the tent shines (**{R['famous']['1929-09'][3]:.2f}×** "
            f"final wealth vs **{R['famous']['1929-09'][0]:.2f}×** for 60/40). That one cohort is "
            "where the folklore comes from. |\n"
            "| So what actually killed retirees historically? | **The 1966–1982 inflation "
            "grind** — fifteen years where stocks *and bonds* lost real value together. A bond "
            "tent holds MORE of the asset that was dying, and buys stocks back *after* missing "
            "the recovery. |\n\n"
            "> The tent insures against the crash everyone remembers — and marches straight into "
            "the disaster that actually did the killing."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Sequence risk peaks at retirement. So hold your **fewest** stocks on retirement "
            "day, and raise the stock share as retirement progresses — a rising equity "
            "glidepath.\"* — Pfau & Kitces (2014)\n\n"
            "The intuition is genuinely clever: if the crash comes **early**, you barely own "
            "stocks; if it comes **late**, you've had decades of growth as a cushion. Let's draw "
            "the contenders."
        ),
        code(
            "yrs = np.arange(30)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.plot(yrs, 100*st.weights_for('rising'), c=BLUE, lw=2.5, label='rising tent 30%\\u219270% (the claim)')\n"
            "ax.plot(yrs, 100*st.weights_for('declining'), c=AMBER, lw=2, ls='--', label='declining 70%\\u219230% (mirror image)')\n"
            "ax.plot(yrs, 100*st.weights_for('s6040'), c=GREY, lw=2, label='static 60/40 (the benchmark)')\n"
            "ax.set_xlabel('years since retirement'); ax.set_ylabel('% in stocks')\n"
            "ax.set_ylim(0, 100); ax.legend(); ax.set_title('The contenders: stock share through a 30-year retirement')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('rising and declining hold the SAME average stock share (50%) -- only the timing differs')"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Target-date funds, retirement books and a thriving corner of the advice industry "
            "sell versions of this. If the tent works, every retiree should tilt bond-heavy at "
            "65. If it doesn't — if it actually *raises* the odds of running out of money — "
            "that's an expensive piece of folklore, and the \"safety\" it sells is the opposite "
            "of what it delivers."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take Shiller's monthly data back to **{R['start']}** — real (inflation-adjusted) "
            "US stock returns and 10-year government bond returns — and run **every possible "
            f"30-year retirement** ({R['n_cohorts']:,} of them, one starting every month from "
            f"{R['cohort_start']} to {R['cohort_end']}). Each retiree follows the classic 4% "
            "rule: withdraw 4% of the starting pot, inflation-adjusted, every year; rebalance "
            "annually to that year's target mix; pay 10 bps on every trade. **Success** = the "
            "money never runs out in 30 years. Then we race the tent against static mixes and "
            "against its own mirror image — same average stock share, reversed timing. That "
            "mirror race is the killer test: if \"stocks late beats stocks early\" is true, the "
            "tent must beat its reflection."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Success rates first.** The share of the 1,470 historical retirements that never "
            "ran out of money:"
        ),
        code(
            "wrs = ['4.0','4.5','5.0','5.5']\n"
            "if HAVE_REAL:\n"
            "    tab = {w: [100*st.simulate(GE, GB, st.weights_for(s), wr=float(w)/100)['success'].mean()\n"
            "               for s in STRATS] for w in wrs}\n"
            "else:\n"
            "    tab = R['succ']\n"
            "x = np.arange(len(wrs)); wdt = 0.16\n"
            "colors = {'s6040': GREY, 's5050': '#b8b8b8', 's4060': '#d8d8d8', 'rising': BLUE, 'declining': AMBER}\n"
            "fig, ax = plt.subplots(figsize=(10.0, 4.8))\n"
            "lbl = {'s6040':'static 60/40','s5050':'static 50/50','s4060':'static 40/60',\n"
            "       'rising':'RISING TENT','declining':'declining mirror'}\n"
            "for k, s in enumerate(STRATS):\n"
            "    ax.bar(x + (k-2)*wdt, [tab[w][k] for w in wrs], wdt, color=colors[s], label=lbl[s])\n"
            "ax.set_xticks(x); ax.set_xticklabels([w + '% withdrawals' for w in wrs])\n"
            "ax.set_ylabel('% of 30-year retirements that survived'); ax.set_ylim(40, 100)\n"
            "ax.set_title('The tent (blue) trails the plain 60/40 at every withdrawal rate')\n"
            "ax.legend(ncol=2, fontsize=9); plt.tight_layout(); plt.show()\n"
            "for w in wrs: print(w+'%:', {s: round(tab[w][k],2) for k, s in enumerate(STRATS)})"
        ),
        md(
            f"The tent survives **{R['succ']['4.0'][3]:.1f}%** of retirements at 4% withdrawals; "
            f"plain 60/40 survives **{R['succ']['4.0'][0]:.1f}%**. Push withdrawals to 5% and the "
            f"gap becomes a chasm: **{R['succ']['5.0'][3]:.1f}% vs {R['succ']['5.0'][0]:.1f}%**. "
            "And look at the amber bars: the tent's **mirror image** — start stock-heavy, glide "
            "down, the exact *opposite* advice — beats it everywhere too. The magic ingredient "
            "(\"stocks late, not early\") points the **wrong way** on real history.\n\n"
            "**But wait — 1929.** The tent's sales pitch is one specific nightmare: retiring "
            "right before a crash. Let's watch the two most famous nightmare retirements in "
            "history, month by month."
        ),
        code(
            "def wealth_path(i, name, wr=0.04):\n"
            "    # illustrative annual path, ex-costs\n"
            "    w = st.weights_for(name); out = [1.0]\n"
            "    W = 1.0 - wr; veq, vbd = W*w[0], W*(1-w[0])\n"
            "    veq, vbd = veq*GE[i,0], vbd*GB[i,0]; out.append(veq+vbd)\n"
            "    for j in range(1, 30):\n"
            "        W = veq + vbd\n"
            "        if W <= wr: out += [0.0]*(30-j); return np.array(out[:31])\n"
            "        W -= wr\n"
            "        veq, vbd = w[j]*W*GE[i,j], (1-w[j])*W*GB[i,j]\n"
            "        out.append(veq+vbd)\n"
            "    return np.array(out)\n"
            "if HAVE_REAL:\n"
            "    import pandas as pd\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)\n"
            "    for ax, ym, title in [(axes[0], '1929-09-01', 'retire Sept 1929 (the crash)'),\n"
            "                          (axes[1], '1966-01-01', 'retire Jan 1966 (the inflation grind)')]:\n"
            "        i = int(np.where(DATES == pd.Timestamp(ym))[0][0])\n"
            "        for name, c, l in [('rising', BLUE, 'rising tent'), ('s6040', GREY, 'static 60/40'),\n"
            "                           ('declining', AMBER, 'declining mirror')]:\n"
            "            ax.plot(wealth_path(i, name), c=c, lw=2, label=l)\n"
            "        ax.axhline(0, c='k', lw=.8); ax.set_title(title); ax.set_xlabel('years into retirement')\n"
            "    axes[0].set_ylabel('real wealth (multiple of starting pot)'); axes[0].legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "print('terminal wealth @4%: 1929-09', dict(zip(STRATS, R['famous']['1929-09'])))\n"
            "print('                    1966-01', dict(zip(STRATS, R['famous']['1966-01'])))"
        ),
        md(
            f"**Left panel — the tent's finest hour.** Retire in September 1929 and the tent ends "
            f"with **{R['famous']['1929-09'][3]:.2f}×** your starting pot while 60/40 limps to "
            f"{R['famous']['1929-09'][0]:.2f}× and the mirror strategy goes **broke**. A short, "
            "deflationary crash, bonds holding value, stocks bought back cheap: everything works "
            "as advertised. This single cohort is the folklore's engine.\n\n"
            "**Right panel — the actual killer.** Retire in January 1966 and *nothing* survives — "
            "but the tent dies no better than anything else, because from 1966 to 1982 inflation "
            "destroyed **bonds too** (our bond returns are inflation-adjusted; that's the point). "
            "The tent held *more* of the dying asset for longer, then raised its stock share "
            "*after* the 1980s recovery had already been missed. Historically the 1966-style "
            "failure vastly outnumbers the 1929-style one: **all 54** retirements that break the "
            f"60/40 at 4% start between {R['rescue']['band_6040'].replace('-', ' and ')} — and "
            f"the tent rescues just **{R['rescue']['rising_rescue_pct']:.1f}%** of them while "
            f"adding failures of its own ({R['rescue']['n_fail_rising']} vs "
            f"{R['tw'][0][3]} for 60/40)."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** On {R['n_cohorts']:,} historical retirements the rising tent "
            "is *less* safe than a plain 60/40 on every measure — success rate, average final "
            "wealth, worst cases. The quants notebook certifies the shortfall statistically "
            "(*t* ≈ −6) and shows it isn't a cost artefact or a shape-detail artefact.\n"
            "- **Tradability — Mirage.** Nothing stops you from implementing a bond tent with "
            "two index funds. It's just that doing so historically **bought you less safety**, "
            "not more.\n"
            "- **\"Is the benefit clever timing, or just holding fewer stocks?\" — Busted: "
            "neither.** Against a 50/50 with the same average stock share the tent still loses, "
            "and it loses to its own mirror image — the timing ingredient points the wrong way "
            "on real history. The tent's genuine benefit exists only in simulated worlds where "
            "stocks barely beat bonds (the quants notebook rebuilds exactly that world and "
            "watches the effect appear)."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **What actually protected retirees?** Boring answers: a lower withdrawal rate "
            "(see [study 173 — the 4% rule](../../173-four-percent-rule/README.md)) and simply "
            "*owning enough stocks throughout*. The 60/40's equity engine out-insured the tent's "
            "bond cushion.\n"
            "- **Why does the literature say it works?** Pfau & Kitces used Monte Carlo "
            "simulations — worlds where returns have no long inflation regimes and (in their "
            "low-return scenarios) stocks barely beat bonds. In *that* laboratory the tent "
            "really does help. History disagreed.\n"
            "- **The accumulation cousin.** The same desk tested \"your age in bonds\" on the "
            "saving phase in [study 172 — hundred-minus-age](../../172-hundred-minus-age/README.md): "
            "same pattern, de-risking that costs more than it protects.\n"
            "- **An honest caveat.** The US tape is the *best* stock market in recorded history — "
            "it flatters stock-heavy rules. And our bonds are inflation-adjusted 10-years, which "
            "the 1940s–70s crushed. Both choices are stated, not hidden; a TIPS-style bond leg "
            "(which didn't exist before 1997) would soften the 1966 story.\n\n"
            "*Think the tent just needs the right shape? The quants notebook races four of them — "
            "the faster a tent exits its bonds, the less it loses.*"
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
            "# The Bond Tent — a quantitative teardown 🔬\n"
            "### 1,470 overlapping 30-year cohorts · HAC (360-month bandwidth) + circular block "
            "bootstrap · the shape-vs-allocation decomposition · the mirror-image timing race · "
            "high-CAPE conditioning · a premium-sweep synthetic control\n\n"
            + BADGES +
            "Deep companion to [01_for_the_curious](01_for_the_curious.ipynb). The claim "
            "(Pfau-Kitces 2014): a **rising equity glidepath** in retirement (30%→70%, the back "
            "half of the \"bond tent\") cuts sequence-of-returns risk vs static allocations. We "
            "test it on the Shiller real-return tape (1871-02 → 2023-06) with 4%-rule real "
            "withdrawals, and decompose the headline gap into **shape** (timing at matched "
            "average equity) and **allocation** (just holding less equity).\n\n"
            "> ⚠️ **Data decisions, stated.** Both legs REAL (CPI-deflated): equity = Shiller "
            "real price + dividend yield; bonds = 10-year first-order approx (carry − 7·Δy), "
            "deflated — harsh on bonds in 1940–80 and that is exactly the regime that matters. "
            "Weights are a deterministic function of years-since-retirement (set at the end of "
            "the prior year — one clean lag; the only \"signal\" is the calendar). Costs 10 bps "
            "one-way × traded value. No survivorship (single index tape); the named bias is US "
            "equity exceptionalism, which flatters the 60/40 benchmark — and is part of the "
            "finding. Numbers: [`docs/results.md`](../docs/results.md) (fingerprint `"
            + R["fingerprint"] + "`).\n"
            ">\n"
            "> 💡 The `💡 In plain words` notes translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Tent trails 60/40 everywhere: success "
            f"**{R['succ']['4.0'][3]:.1f}% vs {R['succ']['4.0'][0]:.1f}%** @4% "
            f"(**{R['succ']['5.0'][3]:.1f}% vs {R['succ']['5.0'][0]:.1f}%** @5%), mean terminal "
            f"wealth **{R['tw'][3][0]:.2f}× vs {R['tw'][0][0]:.2f}×** (HAC *t* = "
            f"**{R['decomp'][0][2]:.2f}**, bootstrap CI [{R['decomp'][0][3]:.2f}, "
            f"{R['decomp'][0][4]:.2f}] excludes 0), SAFEMAX {R['safemax'][3]:.2f}% vs "
            f"{R['safemax'][0]:.2f}%. No positive effect to certify at any threshold. |\n"
            f"| **Tradability** | `MIRAGE` | Implementable with two ETFs — and adopting it cost "
            f"**{R['decomp'][0][1]:.2f}×** terminal wealth and **{-R['decomp'][0][5]:.1f} pp** "
            f"success @4% ({-R['decomp'][0][6]:.1f} pp @5%) vs the simpler static 60/40. |\n"
            f"| **Timing or less equity?** | `BUSTED` | Both channels negative: allocation "
            f"*t* = {R['decomp'][3][2]:.2f}; shape at matched average equity *t* = "
            f"{R['decomp'][1][2]:.2f}; the tent loses the pure-timing mirror race "
            f"({R['decomp'][2][5]:+.1f} pp @4%, {R['decomp'][2][6]:+.1f} pp @5%). |\n\n"
            "> 💡 In plain words: there is no safety benefit to explain — the decomposition's "
            "job turned out to be performing the autopsy."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $w_j$ be the equity weight in retirement year $j$, and define terminal real "
            "wealth $W_{30}$ of a cohort under the 4% rule (withdraw $c=0.04$ real at the start "
            "of each year, rebalance to $w_j$, grow at $w_j R^{EQ}_j + (1-w_j) R^{BD}_j$). "
            "Pfau-Kitces claim that a **rising** path $w_j: 0.30 \\to 0.70$ beats static paths "
            "on *failure risk* — $\\Pr[W_t \\le 0]$ — because sequence risk concentrates early.\n\n"
            "- **H₁ (headline).** rising beats static 60/40 on success rate / tail outcomes.\n"
            "- **H₂ (shape).** rising beats static 50/50 — same 50% *average* equity, so any "
            "edge is pure shape.\n"
            "- **H₃ (timing).** rising beats its mirror declining 70→30 — identical average "
            "weight, reversed order: the cleanest test that *late* equity beats *early* equity.\n\n"
            "All three fail on the tape; H₁'s reversal is certified at HAC *t* ≈ −6."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Monthly-start 30-year cohorts overlap by up to 359 months, so naive t-stats on "
            "cohort panels are fantasy. Two instruments:\n\n"
            "1. **Newey-West HAC** on per-cohort terminal-wealth differences with the bandwidth "
            "**forced to the full 360-month overlap** (the automatic selector would pick ~7 lags "
            "and overstate precision by an order of magnitude).\n"
            "2. **Circular block bootstrap** (120-month blocks, joint EQ/BD rows, 1,000 reps, "
            "seed 596) for distribution statistics (success rates, percentiles). Blocks preserve "
            "within-decade dynamics but destroy cross-decade mean reversion — stated, not "
            "hidden; if anything that *helps* the tent, since long-horizon reversion is the "
            "channel that should reward late equity.\n\n"
            "Success-rate gaps are quoted with bootstrap CIs, never bare."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · The protocol\n\n"
            f"- **Tape.** Shiller monthly, {R['start']} → {R['end']} ({R['months']:,} months), "
            f"real EQ **{R['ann_eq']:.2f}%/yr**, real BD **{R['ann_bd']:.2f}%/yr** (realised "
            f"premium **{R['premium']:.2f} pp**).\n"
            f"- **Cohorts.** {R['n_cohorts']:,} monthly-start 30-year retirements "
            f"({R['cohort_start']} → {R['cohort_end']}); annual real withdrawals at the start of "
            "each year; annual rebalance to the year-*j* target (set at the end of year *j−1*); "
            f"{R['cost_bps']} bps one-way × traded value (withdrawal sales + turnover).\n"
            "- **Strategies.** Static 60/40, 50/50, 40/60; rising 30→70 (avg 50%); declining "
            "70→30 (avg 50%). Robustness: 30→60 (the original Kitces-Pfau span), 30→70 by year "
            "15 (accelerated), 20→80 (deep tent); costs 0/10/25 bps.\n"
            "- **Decomposition identity.** `rising − 60/40 = (rising − 50/50) + (50/50 − 60/40)`.\n"
            "- **Control.** 20 independent seeded synthetic worlds per setting; detector = "
            "rising vs declining; exact null = no withdrawals + i.i.d. (mirror schedules are "
            "exchangeable ⇒ identical terminal-wealth law)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Success rates and terminal wealth — the raw scoreboard"
        ),
        code(
            "wrs = ['4.0','4.5','5.0','5.5']\n"
            "if HAVE_REAL:\n"
            "    tab = {w: [100*st.simulate(GE, GB, st.weights_for(s), wr=float(w)/100)['success'].mean()\n"
            "               for s in STRATS] for w in wrs}\n"
            "    res4 = {s: st.simulate(GE, GB, st.weights_for(s), wr=0.04) for s in STRATS}\n"
            "    tw = [(res4[s]['terminal'].mean(), float(np.median(res4[s]['terminal'])),\n"
            "           float(np.quantile(res4[s]['terminal'], .05)), int((~res4[s]['success']).sum()))\n"
            "          for s in STRATS]\n"
            "else:\n"
            "    tab, tw, res4 = R['succ'], R['tw'], None\n"
            "print('success rates (%):'); print('  wr    ' + ''.join(f'{s:>11}' for s in STRATS))\n"
            "for w in wrs: print(f'  {w}% ' + ''.join(f'{tab[w][k]:>10.2f}%' for k in range(5)))\n"
            "print()\n"
            "print('terminal wealth @4% (multiple of initial):')\n"
            "for k, s in enumerate(STRATS):\n"
            "    print(f'  {s:<10} mean {tw[k][0]:.3f}  median {tw[k][1]:.3f}  p05 {tw[k][2]:.3f}  failures {tw[k][3]}/'\n"
            "          + str(R['n_cohorts']))"
        ),
        md(
            f"> 💡 In plain words: five columns, one loser. The tent has **more failures** "
            f"({R['tw'][3][3]} vs {R['tw'][0][3]}), **less average wealth** "
            f"({R['tw'][3][0]:.2f}× vs {R['tw'][0][0]:.2f}×) and a **worse tail** (5th percentile "
            f"{R['tw'][3][2]:.3f} vs {R['tw'][0][2]:.3f}) than the plain 60/40 it was supposed to "
            "protect you from."
        ),
        md(
            "### 4b · The decomposition — shape vs allocation, plus the mirror race\n\n"
            "`rising − 60/40 = (rising − 50/50) + (50/50 − 60/40)`. HAC *t* (360-month "
            "bandwidth) on per-cohort terminal-wealth differences at 4%; the mirror race "
            "(rising vs declining, identical average weight) isolates pure timing."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pairs = [('rising','s6040'), ('rising','s5050'), ('rising','declining'), ('s5050','s6040')]\n"
            "    labels = [d[0] for d in R['decomp']]\n"
            "    rows = []\n"
            "    res5 = {s: st.simulate(GE, GB, st.weights_for(s), wr=0.05) for s in STRATS}\n"
            "    for (a, b), lab in zip(pairs, labels):\n"
            "        d = res4[a]['terminal'] - res4[b]['terminal']\n"
            "        rows.append((lab, d.mean(), st.hac_tstat(d, lags=360),\n"
            "                     100*(res4[a]['success'].mean()-res4[b]['success'].mean()),\n"
            "                     100*(res5[a]['success'].mean()-res5[b]['success'].mean())))\n"
            "else:\n"
            "    rows = [(d[0], d[1], d[2], d[5], d[6]) for d in R['decomp']]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.6))\n"
            "names = [r[0].split(' (')[0] for r in rows]; ts = [r[2] for r in rows]\n"
            "ax.barh(names[::-1], ts[::-1], color=[GREY, BLUE, BLUE, RED][::-1])\n"
            "ax.axvline(-2, ls='--', c=RED, label='t = -2'); ax.axvline(0, c='k', lw=.8)\n"
            "for i, t in enumerate(ts[::-1]): ax.annotate(f' t = {t:+.2f}', (t, i), va='center',\n"
            "    ha='right' if t < -3 else 'left')\n"
            "ax.set_xlabel('HAC t (360-month bandwidth) on terminal-wealth difference @4%')\n"
            "ax.set_title('Every arrow points the wrong way for the claim'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in rows: print(f'{r[0]:<32} dTW {r[1]:+.4f}  HAC t {r[2]:+.2f}  dsucc@4% {r[3]:+.2f}pp  @5% {r[4]:+.2f}pp')"
        ),
        md(
            f"> 💡 In plain words: the headline gap (rising − 60/40, *t* = {R['decomp'][0][2]:.2f}) "
            "splits into two pieces and **both hurt**. Holding less equity is a certified cost "
            f"(*t* = {R['decomp'][3][2]:.2f}). The shape itself — the only novel ingredient — is "
            f"negative at matched average equity (*t* = {R['decomp'][1][2]:.2f}) and the tent "
            f"loses the mirror race by {-R['decomp'][2][6]:.1f} pp of success at 5% withdrawals. "
            "\"Stocks late beats stocks early\" is simply not what 152 years of US history did."
        ),
        md(
            "### 4c · Block-bootstrap CIs — is the shortfall noise?\n\n"
            "Circular block bootstrap of the joint monthly tape (120-month blocks). The picture "
            "uses 200 reps to stay light; the canonical 1,000-rep CIs from "
            "[`docs/results.md`](../docs/results.md) are printed alongside."
        ),
        code(
            "if HAVE_REAL:\n"
            "    boot = st.block_bootstrap(EQ, BD, pairs=(('rising','s6040'),), n_boot=200, seed=596)\n"
            "    draws = boot[('rising','s6040')]['dmean']\n"
            "else:\n"
            "    rng = np.random.default_rng(596); draws = rng.normal(R['decomp'][0][1], 0.3, 200)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=30, color=GREY, alpha=.85, label='bootstrap draws (200 reps, illustrative)')\n"
            "ax.axvline(R['decomp'][0][1], c=RED, lw=2.5, label=f\"observed {R['decomp'][0][1]:+.3f}x\")\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('mean terminal-wealth difference, rising - 60/40 (x initial, @4%)')\n"
            "ax.set_title(f\"Canonical 95% CI [{R['decomp'][0][3]:+.3f}, {R['decomp'][0][4]:+.3f}] -- zero is outside\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('canonical 1000-rep CIs (docs/results.md):')\n"
            "for d in R['decomp']:\n"
            "    print(f'  {d[0]:<32} dmeanTW CI [{d[3]:+.3f}, {d[4]:+.3f}]   dsucc@5% CI [{d[7]:+.2f}, {d[8]:+.2f}] pp')"
        ),
        md(
            f"> 💡 In plain words: resample history in decade-sized blocks 1,000 times and the "
            f"tent's wealth shortfall vs 60/40 stays negative in >97.5% of resamples "
            f"(CI [{R['decomp'][0][3]:+.2f}, {R['decomp'][0][4]:+.2f}]) — same for the shape and "
            "timing components. The success-rate CIs are wider (failures cluster in one era), "
            "but nowhere is there a *positive* effect to rescue: the claim needs the tent to "
            "**win**, and it never does."
        ),
        md(
            "### 4d · The advertised sweet spot — high-CAPE retirements\n\n"
            "Advocates pitch the tent hardest for retirements at rich valuations. Condition on "
            "top-quartile CAPE (PE10) at the retirement date:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    import pandas as pd\n"
            "    raw = data.load_shiller(); cape = raw['PE10'].reindex(DATES).to_numpy()\n"
            "    valid = cape > 0; q4 = valid & (cape >= np.nanquantile(cape[valid], .75))\n"
            "    res45 = {s: st.simulate(GE, GB, st.weights_for(s), wr=0.045) for s in STRATS}\n"
            "    rows = {w: [100*r[s]['success'][q4].mean() for s in STRATS]\n"
            "            for w, r in [('4.0', res4), ('4.5', res45),\n"
            "                         ('5.0', {s: st.simulate(GE, GB, st.weights_for(s), wr=0.05) for s in STRATS})]}\n"
            "    n_q4 = int(q4.sum())\n"
            "else:\n"
            "    rows, n_q4 = R['cape']['rows'], R['cape']['n_q4']\n"
            "x = np.arange(3); wdt = .3\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(x - wdt/2, [rows[w][0] for w in ('4.0','4.5','5.0')], wdt, color=GREY, label='static 60/40')\n"
            "ax.bar(x + wdt/2, [rows[w][3] for w in ('4.0','4.5','5.0')], wdt, color=BLUE, label='rising tent')\n"
            "for i, w in enumerate(('4.0','4.5','5.0')):\n"
            "    ax.annotate(f'{rows[w][0]:.0f}%', (i-wdt/2, rows[w][0]), ha='center', va='bottom')\n"
            "    ax.annotate(f'{rows[w][3]:.0f}%', (i+wdt/2, rows[w][3]), ha='center', va='bottom')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['4.0%','4.5%','5.0%']); ax.set_xlabel('withdrawal rate')\n"
            "ax.set_ylabel('success rate within top-quartile CAPE cohorts')\n"
            "ax.set_title(f'Even in the advertised sweet spot (n={n_q4} expensive retirements), the tent trails')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for w in ('4.0','4.5','5.0'): print(w+'%:', {s: round(rows[w][k],1) for k, s in enumerate(STRATS)})"
        ),
        md(
            f"> 💡 In plain words: retiring expensive is exactly when the tent should earn its "
            f"keep — and within those {R['cape']['n_q4']} cohorts it trails 60/40 by "
            f"**{R['cape']['rows']['4.5'][0] - R['cape']['rows']['4.5'][3]:.0f} pp** at 4.5% "
            "withdrawals. High CAPE historically preceded low *equity* returns, but the failure "
            "regimes crushed **bonds** as hard — there was no safe tent pole to lean on."
        ),
        md(
            "### 4e · Robustness — every tent shape, every cost level"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob = []\n"
            "    for v, lab in [('rising','rising 30-70 (headline)'), ('rising_3060','rising 30-60 (Kitces-Pfau)'),\n"
            "                   ('rising_fast','rising 30-70 by yr15 (fast)'), ('rising_2080','rising 20-80 (deep tent)')]:\n"
            "        w = st.weights_for(v)\n"
            "        r4 = st.simulate(GE, GB, w, wr=0.04); r5 = st.simulate(GE, GB, w, wr=0.05)\n"
            "        rob.append((lab, 100*r4['success'].mean(), 100*r5['success'].mean(),\n"
            "                    r4['terminal'].mean(), st.hac_tstat(r4['terminal']-res4['s6040']['terminal'], 360)))\n"
            "else:\n"
            "    rob = R['robust']\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "names = [r[0].split(' (')[0]+'\\n('+r[0].split(' (')[1] for r in rob]\n"
            "ax.bar(names, [r[4] for r in rob], color=BLUE, width=.55)\n"
            "ax.axhline(-2, ls='--', c=RED, label='t = -2'); ax.axhline(0, c='k', lw=.8)\n"
            "for i, r in enumerate(rob): ax.annotate(f't={r[4]:+.2f}', (i, r[4]), ha='center', va='bottom')\n"
            "ax.set_ylabel('HAC t vs static 60/40 (terminal wealth @4%)')\n"
            "ax.set_title('Four tent shapes, four losers -- the faster the tent exits bonds, the less it loses')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for r in rob: print(f'{r[0]:<30} succ@4% {r[1]:.2f}%  succ@5% {r[2]:.2f}%  meanTW {r[3]:.3f}  HAC t {r[4]:+.2f}')\n"
            "print('costs (succ@4%, 60/40 | rising | declining):')\n"
            "for cb, v in R['costs'].items(): print(f'  {cb:>3} bps: {v[0]:.2f}% | {v[1]:.2f}% | {v[2]:.2f}%')"
        ),
        md(
            f"> 💡 In plain words: it is not the shape detail. The original Kitces-Pfau span "
            f"(30→60) does *worse* than our headline tent; the deep tent (20→80) does worse; the "
            "only variant that closes the gap is the one that **abandons the tent fastest** "
            f"(30→70 by year 15, *t* = {R['robust'][2][4]:+.2f}). Costs move success by "
            "hundredths of a point — this is not a friction story. SAFEMAX tells the same tale: "
            f"60/40 {R['safemax'][0]:.2f}%, declining {R['safemax'][4]:.2f}%, rising "
            f"{R['safemax'][3]:.2f}%."
        ),
        md(
            "### 4f · Synthetic control — the engine is faithful, and it explains the paper\n\n"
            "20 independent seeded worlds per setting; detector = rising vs declining (same 50% "
            "average equity — pure timing); one-sample *t* across worlds. **Exact null:** no "
            "withdrawals + i.i.d. returns ⇒ a weight path and its mirror give the *same* "
            "terminal-wealth law (products of independent factors are exchangeable), so the "
            "detector must read ~0. Then we sweep the equity premium at a 4% withdrawal rate — "
            "Kitces-Pfau's Monte Carlo laboratory. *(Machinery proof only — never cited as "
            "market evidence.)*"
        ),
        code(
            "settings = [('EXACT NULL\\nwr=0', dict(reversion=0.0, wr=0.0, eq_mu=0.065, bd_mu=0.02)),\n"
            "            ('wr=4%\\nprem 0pp', dict(reversion=0.0, wr=0.04, eq_mu=0.02, bd_mu=0.02)),\n"
            "            ('wr=4%\\nprem 2pp', dict(reversion=0.0, wr=0.04, eq_mu=0.04, bd_mu=0.02)),\n"
            "            ('wr=4%\\nprem 4.5pp', dict(reversion=0.0, wr=0.04, eq_mu=0.065, bd_mu=0.02)),\n"
            "            ('wr=5%\\nprem 4.5pp', dict(reversion=0.0, wr=0.05, eq_mu=0.065, bd_mu=0.02))]\n"
            "res = [(lab, st.control_shape_effect(n_seeds=20, **kw)) for lab, kw in settings]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.5))\n"
            "vals = [r['dsucc_mean'] for _, r in res]\n"
            "cols = [GREY] + [GREEN if v > 0 else RED for v in vals[1:]]\n"
            "ax.bar([l for l, _ in res], vals, color=cols, width=.55)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i, (l, r) in enumerate(res):\n"
            "    t = r['dsucc_t']\n"
            "    ax.annotate('t=n/a' if not np.isfinite(t) else f't={t:+.2f}',\n"
            "                (i, vals[i]), ha='center', va='bottom' if vals[i] >= 0 else 'top')\n"
            "ax.set_ylabel('success-rate gap, rising - declining (pp)')\n"
            "ax.set_title('The tent DOES work in an i.i.d. low-premium world -- and dies at the premium history paid')\n"
            "plt.tight_layout(); plt.show()\n"
            "for l, r in res: print(f\"{l.replace(chr(10),' '):<22} dsucc {r['dsucc_mean']:+.2f} pp (t {r['dsucc_t']:+.2f})  dp05 {r['dp05_mean']:+.4f}\")"
        ),
        md(
            f"> 💡 In plain words: the machinery is honest — at the exact null it finds nothing "
            f"(every world 100% success both ways; Δp05 *t* = {R['control'][0][4]:.2f}), and in "
            f"a zero-premium i.i.d. world at 4% withdrawals the tent's benefit is loud and real "
            f"(**{R['control'][1][1]:+.2f} pp**, *t* = {R['control'][1][2]:+.2f}) — exactly what "
            "Pfau-Kitces found in Monte Carlo. The benefit **shrinks as the equity premium "
            f"grows** ({R['control'][3][1]:+.2f} pp at the historical 4.5 pp premium) and "
            f"**flips sign** at a 5% withdrawal rate ({R['control'][4][1]:+.2f} pp, "
            f"*t* = {R['control'][4][2]:+.2f}). And the *real* tape is worse than i.i.d.: its "
            "killer sequences are decade-plus inflation regimes that crush bonds and stocks "
            "together — a structure Monte Carlo never generates and a bond tent cannot survive. "
            "That is the whole story of the literature-vs-tape disagreement."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the claimed benefit never appears: the tent trails 60/40 on "
            f"success at every withdrawal rate ({R['succ']['4.0'][3]:.1f}% vs "
            f"{R['succ']['4.0'][0]:.1f}% @4%; {R['succ']['5.0'][3]:.1f}% vs "
            f"{R['succ']['5.0'][0]:.1f}% @5%), on mean/median/p05 terminal wealth (HAC *t* = "
            f"{R['decomp'][0][2]:.2f}, bootstrap CI excludes 0) and on SAFEMAX "
            f"({R['safemax'][3]:.2f}% vs {R['safemax'][0]:.2f}%); it loses to its own mirror "
            "image; robust across shapes, costs and the high-CAPE sweet spot. Named caveats: US "
            "equity exceptionalism flatters the equity-heavy benchmark, and the CPI-deflated "
            "10-year bond leg is harsh on bonds — both stated as decisions.\n"
            f"- **Tradability `MIRAGE`** — nothing to deploy: adopting the tent cost "
            f"{R['decomp'][0][1]:.2f}× terminal wealth and {-R['decomp'][0][5]:.1f} pp of "
            "success @4% vs a simpler two-ETF 60/40.\n"
            f"- **Timing or less equity? `BUSTED`** — neither channel exists on the tape: "
            f"allocation is a certified cost (*t* = {R['decomp'][3][2]:.2f}) and shape is "
            f"negative at matched average equity (*t* = {R['decomp'][1][2]:.2f}; mirror race "
            "lost). The tent's genuine habitat is i.i.d. Monte Carlo with a thin equity premium "
            "at a moderate withdrawal rate — the control reproduces Kitces-Pfau there "
            f"({R['control'][1][1]:+.1f} pp, *t* {R['control'][1][2]:+.1f}) — but that world is "
            "not the one the tape recorded."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **International tapes.** Estrada (2016) ran glidepaths across 19 countries and "
            "found the same ordering — rising paths trail static and declining ones historically. "
            "Rebuilding this study on non-US tapes (where equity premia were thinner) should "
            "*shrink* the tent's shortfall without flipping it; the control's premium sweep "
            "predicts exactly how much.\n"
            "- **A TIPS tent.** The tent's fatal regime (1966–82) predates inflation-linked "
            "bonds. A tent pitched on TIPS is a different, untested claim — the 10-year nominal "
            "tent tested here is the one the literature actually recommends against history.\n"
            "- **The siblings.** [Study 172](../../172-hundred-minus-age/README.md) buries the "
            "accumulation-phase cousin (age-in-bonds); [study 173](../../173-four-percent-rule/README.md) "
            "shows the withdrawal *rate* — not the glidepath — is the lever that actually moves "
            "survival. Together the three studies say: pick a sane equity share, keep it, and "
            "watch the withdrawal rate.\n\n"
            "*The reproducible core is offline and deterministic; weights are calendar-only (one "
            "clean lag), both legs are real total returns, and the decomposition is an exact "
            "identity. Methods and sources: [`docs/references.md`](../docs/references.md); frozen "
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
