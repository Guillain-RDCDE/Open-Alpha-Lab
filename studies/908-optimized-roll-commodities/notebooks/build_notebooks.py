"""Generate the two narrative notebooks for Study 908 (Optimized-Roll Commodities).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached tape under
../_cache/ when present and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance total-return,
# excess of BIL, 2010-09 -> 2026-06, 190 months; USCI vs DBC/GSG/DJP; HAC 6 lags; seed 908).
R = dict(
    asof="2026-06-30", fingerprint="16935c3b46d8", n=190,
    # full sample: excess return %/yr, vol %/yr, excess-of-cash Sharpe
    legs={
        "USCI": dict(ex=3.61, vol=14.7, sharpe=0.245, tr=3.95, dd=-64.0, er=1.03),
        "DBC":  dict(ex=2.34, vol=17.2, sharpe=0.136, tr=2.24, dd=-64.8, er=0.85),
        "GSG":  dict(ex=1.14, vol=21.3, sharpe=0.053, tr=0.18, dd=-78.2, er=0.48),
        "DJP":  dict(ex=0.67, vol=16.5, sharpe=0.040, tr=0.66, dd=-69.4, er=0.70),
    },
    # USCI vs bench: sharpe adv, ci_lo, ci_hi, frac<=0, diff %/yr, HAC t
    race={
        "DBC": dict(adv=0.109, lo=-0.130, hi=0.344, frac=0.195, diff=1.26, t=0.63),
        "GSG": dict(adv=0.191, lo=-0.069, hi=0.433, frac=0.075, diff=2.47, t=0.94),
        "DJP": dict(adv=0.204, lo=-0.027, hi=0.437, frac=0.045, diff=2.94, t=1.67),
    },
    # era cut: {bench: {era: (diff %/yr, HAC t, sharpe adv)}}
    eras=["deep-contango 2010-2015", "recovery 2016-2020", "backwardation 2021-2026"],
    era={
        "DBC": [(4.95, 1.49, 0.247), (-7.22, -3.04, -0.480), (5.40, 1.75, 0.438)],
        "GSG": [(7.49, 1.48, 0.321), (-3.32, -0.82, -0.265), (2.86, 0.76, 0.428)],
        "DJP": [(6.91, 2.68, 0.383), (-4.79, -2.13, -0.341), (6.12, 2.01, 0.494)],
    },
    # costed net: {bench: (net sharpe adv, net diff %/yr, HAC t)}
    costed={"DBC": (0.094, 1.06, 0.53), "GSG": (0.175, 2.27, 0.87), "DJP": (0.197, 2.86, 1.63)},
    # PDBC (optimized cousin, 2014-11+) vs GSG
    pdbc=dict(n=139, adv=0.101, lo=-0.051, hi=0.245, diff=1.31, t=0.65),
    # synthetic control: planted %/yr -> (sharpe adv, ci_lo, ci_hi, diff %/yr, HAC t)
    syn=[(0.0, 0.007, -0.079, 0.100, 0.10, 0.14), (3.0, 0.194, 0.109, 0.292, 3.10, 4.27)],
)

BENCH = ["DBC", "GSG", "DJP"]

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
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

from opt_roll import data, strategy as st

BENCH = ["DBC", "GSG", "DJP"]
ERAS = [("deep-contango 2010-2015", "2010-09", "2015-12"),
        ("recovery 2016-2020", "2016-01", "2020-12"),
        ("backwardation 2021-2026", "2021-01", "2026-06")]
HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.load_prices()
    RETS = data.monthly_returns(PRICES, asof=data.AS_OF)
    EX = st.excess_frame(RETS, cash=data.CASH)
    COMMON = st.common_sample(EX, ["USCI"] + BENCH)
else:
    PRICES = RETS = EX = COMMON = None
print("real tape cached:", HAVE_REAL, "| common months:", (0 if COMMON is None else len(COMMON)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    L, rc, er = R["legs"], R["race"], R["era"]
    cells = [
        md(
            "# The commodity fund that dodges the \"contango tax\" 🛢️\n"
            "### Optimized-roll vs front-month — USCI vs DBC, GSG, DJP, in plain English\n\n"
            + BADGES +
            "A commodity index is a stack of **futures**. Futures expire, so every month the fund has "
            "to **roll** — sell the contract about to expire and buy a later one. When later contracts "
            "cost *more* (a shape called **contango**), that roll quietly loses money, over and over. "
            "It's the **contango tax**, and for a naive front-month index it can eat several percent a "
            "year.\n\n"
            "Some funds got clever: instead of always buying the very next contract, an **optimized-roll** "
            "index (like **USCI**) hunts along the curve for the *cheapest* contract to hold — and even "
            "picks the commodities whose curves currently pay you to hold them. The pitch: **same "
            "commodities, less tax, higher return**. We checked it on 16 years of real funds.\n\n"
            "> 📓 **Plain-language layer.** Want the bootstrap CIs and HAC *t*-stats? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Did the optimized fund beat the front-month ones? | **Yes, over the full 16 years** — "
            f"USCI's return-per-unit-risk (Sharpe, above cash) was **+{L['USCI']['sharpe']:.2f}** vs "
            f"**+{L['DBC']['sharpe']:.2f}** (DBC), **+{L['GSG']['sharpe']:.2f}** (GSG) and "
            f"**+{L['DJP']['sharpe']:.2f}** (DJP). Right direction, and a smoother ride. |\n"
            f"| Is it a reliable edge? | **No.** The gap isn't statistically distinguishable from zero, "
            f"and — the killer — it **flipped sign** by regime: USCI *lost* to every benchmark through "
            f"2016-2020 (by {er['DBC'][1][0]:.1f}%/yr vs DBC). It wins in contango eras, loses in "
            "others. |\n"
            "| So is it free money? | **No — it's a bet on the shape of the curve.** When curves are in "
            "contango the optimized rule pays; when the whole complex flips to backwardation or "
            "whipsaws, the clever screen can pick the wrong commodities. |\n"
            "| Is this \"timing the curve\"? | Different. [35-contango](../../35-contango/README.md) "
            "moves *in and out* of the market on a curve signal. Here **both** funds stay fully "
            "invested — we just compare two ways of rolling. |"
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A front-month index rolls up a rising curve and bleeds the contango tax. An "
            "optimized-roll index picks the cheapest contract to hold and the most backwardated "
            "commodities, dodging the tax — so it earns a higher risk-adjusted return, structurally.\"*\n\n"
            "This is real, textbook mechanics: the **roll yield**, not spot price moves, drives most of a "
            "commodity index's long-run return (Gorton-Rouwenhorst 2006, Erb-Harvey 2006). So *how* you "
            "roll genuinely matters. The question is whether the packaged optimized wrappers actually "
            "**deliver** a durable edge you can hold — or just a regime bet."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · One trick first — measure *above cash*\n\n"
            "A commodity index is fully collateralised: it parks cash in **T-bills** and earns that "
            "yield *on top of* the commodity return. In 2023-26 that's ~5%/yr — and it's **the same for "
            "every fund**. If we compared raw returns, a high-interest-rate era would look like a "
            "commodity edge. So we subtract a T-bill fund (**BIL**) from *both* sides. What's left is "
            "only the **spot + roll** difference — exactly the thing the claim is about."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · The teardown — who won?\n\n"
            "Excess-of-cash **Sharpe** (return per unit of risk, above T-bills), 2010-09 → 2026-06."
        ),
        code(
            "funds = ['USCI'] + BENCH\n"
            "if HAVE_REAL:\n"
            "    sh = []\n"
            "    for f in funds:\n"
            "        if f == 'USCI':\n"
            "            sh.append(st.annualized_sharpe(COMMON['USCI'].to_numpy())); continue\n"
            "        sh.append(st.sharpe_race(EX, 'USCI', f)['sharpe_bench'])\n"
            "    sh[0] = st.annualized_sharpe(COMMON['USCI'].to_numpy())\n"
            "else:\n"
            "    sh = [R['legs'][f]['sharpe'] for f in funds]\n"
            "colors = [GREEN, GREY, GREY, GREY]\n"
            "labels = ['USCI\\n(optimized)', 'DBC\\n(semi-opt)', 'GSG\\n(front-month)', 'DJP\\n(front-month)']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "ax.bar(labels, sh, color=colors, width=.6)\n"
            "for i, v in enumerate(sh): ax.annotate(f'{v:+.2f}', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.6); ax.set_ylabel('excess-of-cash Sharpe (2010-2026)')\n"
            "ax.set_title('Optimized roll won the full-sample race - on the right side of a two-regime coin')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('excess-of-cash Sharpe:', {f: round(v,3) for f, v in zip(funds, sh)})"
        ),
        md(
            f"USCI (green) sits clearly above the three front-ish rollers. It also rode gentler: its "
            f"worst drawdown was **{L['USCI']['dd']:.0f}%** vs GSG's brutal **{L['GSG']['dd']:.0f}%**. So "
            "far, the optimized story looks great.\n\n"
            "**But is the edge reliable?** Here's the same race, split into three eras of the commodity "
            "cycle."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = st.era_race(EX, 'USCI', 'DBC', ERAS)\n"
            "    diffs = [r['diff_ann_pct'] for r in rows]\n"
            "else:\n"
            "    diffs = [e[0] for e in R['era']['DBC']]\n"
            "labels = ['deep contango\\n2010-2015', 'recovery\\n2016-2020', 'backwardation\\n2021-2026']\n"
            "colors = [GREEN if d > 0 else RED for d in diffs]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "ax.bar(labels, diffs, color=colors, width=.55)\n"
            "for i, v in enumerate(diffs): ax.annotate(f'{v:+.1f}%/yr', (i, v), ha='center', va='bottom' if v>0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('USCI minus DBC (%/yr)')\n"
            "ax.set_title('The edge FLIPS SIGN by regime - it lost through 2016-2020')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('USCI - DBC by era (%/yr):', [round(d,1) for d in diffs])"
        ),
        md(
            f"There it is. USCI **beat** DBC by ~5%/yr when curves were deep in contango (2010-2015) and "
            f"again in the 2021+ inflation surge — but **lost by {er['DBC'][1][0]:.1f}%/yr through "
            f"2016-2020** (and that loss is statistically real, not noise — see the quants notebook). An "
            "edge that turns significantly *negative* for a whole five-year stretch isn't a durable "
            "premium; it's a **bet on the shape of the futures curve**."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · A sanity check on our own tools\n\n"
            "Before trusting any of this, we feed the machinery a **made-up world where we planted a "
            "real +3%/yr roll edge** — and a twin world with **no edge at all**. The tool must find the "
            "first and stay silent on the second."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.03):\n"
            "    w = data.synthetic_world(roll_edge_annual=edge, seed=908)\n"
            "    d = st.synthetic_detect(w)\n"
            "    res.append((edge*100, d['sharpe_adv'], d['diff_ann_pct'], d['t_diff']))\n"
            "labels = [f'planted {int(e)}%/yr' for e,_,_,_ in res]\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.2))\n"
            "ax.bar(labels, [r[1] for r in res], color=[GREY, GREEN], width=.5)\n"
            "for i,(e,a,df_,t) in enumerate(res): ax.annotate(f'adv {a:+.2f}  (t={t:+.2f})',(i,a),ha='center',va='bottom')\n"
            "ax.set_ylabel('recovered Sharpe advantage'); ax.axhline(0,c='k',lw=.6)\n"
            "ax.set_title('Control: null stays dark, a planted +3%/yr edge lights up')\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,a,df_,t in res: print(f'planted {e:+.0f}%/yr -> Sharpe adv {a:+.3f}, return diff {df_:+.2f}%/yr (t={t:+.2f})')"
        ),
        md(
            f"With **no** planted edge the tool reads a Sharpe advantage of ~0 (*t* = {R['syn'][0][5]:+.2f}); "
            f"with a genuine **+3%/yr** it lights up at *t* = {R['syn'][1][5]:+.2f}. So the machinery would "
            "*happily* certify a real, constant roll edge at this sample size. It doesn't — because on the "
            "real tape the edge isn't constant."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** USCI beat every front-month benchmark on excess-of-cash Sharpe over 16 "
            f"years (**+{L['USCI']['sharpe']:.2f}** vs +{L['DBC']['sharpe']:.2f}/+{L['GSG']['sharpe']:.2f}/"
            f"+{L['DJP']['sharpe']:.2f}) — right direction — but the gap isn't statistically solid and it "
            "**flips sign by regime** (it lost 2016-2020). Directionally sensible, not reliably real.\n"
            "- **Tradability — Mirage.** Costs don't kill it (both are buy-and-hold), and USCI even ended "
            "ahead with a shallower drawdown — but there's no *bankable* roll edge: the advantage could be "
            "zero, the sign depends on the regime, and a chunk of USCI's smoother ride is just its lighter "
            "energy weight (lower risk), not paid roll yield. A regime bet wearing a free-lunch label."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The tax is real; the wrapper edge isn't durable.** Front-month indices really do bleed "
            "roll yield in contango — but the optimized fix only pays *while curves are in contango*.\n"
            "- **Watch the composition.** USCI holds far less energy than GSG, so its lower volatility is "
            "partly a diversification choice, not a roll victory. Compare like with like.\n"
            "- **Cousins on the desk.** [35-contango](../../35-contango/README.md) *times* the curve; "
            "[794-commodity-carry](../../794-commodity-carry/README.md) sorts single commodities on carry; "
            "[661-uso-roll-decay](../../661-uso-roll-decay/README.md) is the roll tax on a single-commodity "
            "fund (USO).\n\n"
            "*Think USCI is simply the better commodity fund? Re-read the 2016-2020 bar — then ask which "
            "regime the next decade will be in.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    L, rc, er, ct = R["legs"], R["race"], R["era"], R["costed"]
    cells = [
        md(
            "# Optimized-Roll Commodities — a quantitative teardown 🔬\n"
            "### Excess-of-cash Sharpe race · paired block-bootstrap advantage CIs · HAC *t* on the return "
            "difference · the era sign-flip · cost & composition-tilt decomposition · a planted-edge "
            "synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The claim is "
            "a structural roll-yield edge: optimized-roll (USCI) should out-earn front-month (GSG, DJP) "
            "and semi-optimized (DBC) on a higher **excess-of-cash Sharpe**. The job is to race them "
            "excess-vs-excess and stress the advantage for significance and era-robustness.\n\n"
            "> ⚠️ **Data note.** yfinance **total-return** closes (already net of each fund's expense "
            "ratio); every leg taken **excess of BIL**; 190 common months (USCI inception 2010-08 gates "
            "the start; PDBC from 2014-11). Numbers in [`docs/results.md`](../docs/results.md) (as-of "
            + R["asof"] + ", fingerprint `" + R["fingerprint"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | USCI excess-of-cash Sharpe **+{L['USCI']['sharpe']:.3f}** vs DBC "
            f"+{L['DBC']['sharpe']:.3f} / GSG +{L['GSG']['sharpe']:.3f} / DJP +{L['DJP']['sharpe']:.3f} "
            f"(right sign) — but every return-difference HAC *t* < 2 (best +{rc['DJP']['t']:.2f} vs DJP), "
            f"every bootstrap Sharpe-adv CI includes 0, and the edge is **significantly negative** in "
            f"2016-2020 (USCI−DBC {er['DBC'][1][0]:.2f}%/yr, *t* = {er['DBC'][1][1]:.2f}). Not era-robust. |\n"
            f"| **Tradability** | `MIRAGE` | Costs are near-irrelevant (buy-and-hold; ER already embedded); "
            f"net Sharpe adv +{ct['DBC'][0]:.3f}/+{ct['GSG'][0]:.3f}/+{ct['DJP'][0]:.3f} — CI still through "
            "0. The raw gap is a **regime bet + a lower-energy/lower-vol composition tilt** "
            f"(USCI vol {L['USCI']['vol']:.1f}% vs GSG {L['GSG']['vol']:.1f}%), not a bankable roll premium. |\n\n"
            "> 💡 In plain words: the optimized wrapper landed on the right side of a two-regime coin over "
            "this tape, but the coin, not a constant premium, is what you'd be buying."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Collateralised commodity total return decomposes as\n\n"
            "$$r^{index}_t \\approx \\underbrace{r^{spot}_t}_{\\text{price}} + "
            "\\underbrace{y^{roll}_t}_{\\text{term structure}} + \\underbrace{r^{f}_t}_{\\text{T-bill collateral}}.$$\n\n"
            "The collateral leg $r^f$ is identical across wrappers, so we net it out by subtracting BIL: "
            "every series below is **excess of cash**, leaving $r^{spot}+y^{roll}$. The roll yield "
            "$y^{roll}$ is negative in contango (front-month bleeds) and the optimized rule aims to make "
            "it less negative (or positive).\n\n"
            "- **H₁ (edge exists).** $\\text{Sharpe}(\\text{USCI}_{ex}) > \\text{Sharpe}(\\text{bench}_{ex})$, "
            "with the paired-bootstrap advantage CI clear of 0 and the monthly return difference "
            "$\\Delta_t = \\text{USCI}_t - \\text{bench}_t$ carrying **HAC *t* ≥ 2**.\n"
            "- **H₂ (durable, not a regime bet).** The advantage holds — same sign — across sub-eras.\n"
            "- **H₃ (bankable).** It survives realistic costs and isn't merely a lower-beta composition "
            "tilt.\n\n"
            "The desk's Real bar needs H₁ **and** H₂; Investable needs H₃ too."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what rides on it\n\n"
            "If H₁-H₂ held, \"just buy the optimized wrapper\" would be a free structural upgrade over a "
            "front-month index — a rare costless Sharpe gain in a notoriously return-poor asset class. If "
            "only H₁ holds (full-sample) but H₂ fails, the wrapper is a **conditional** trade whose payoff "
            "depends on the curve regime you can't forecast — worth understanding, not worth a permanent "
            "allocation on the roll story alone. Inference is **Newey-West HAC** throughout (monthly index "
            "differentials are serially correlated through overlapping roll windows)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · The protocol\n\n"
            "- **Funds.** USCI (optimized), DBC (Optimum Yield, semi-opt), GSG & DJP (front-month), PDBC "
            "(optimized, 2014-11+). yfinance total-return monthly closes, as-of " + R["asof"] + ".\n"
            "- **Excess-of-cash.** Subtract BIL from every leg.\n"
            "- **Race.** Annualised excess-of-cash Sharpe per leg; the advantage SR(USCI)−SR(bench) with a "
            "**paired circular block bootstrap** (block 6 months, 2000 draws, seed 908) 95% CI; HAC mean-*t* "
            "(6 lags) on $\\Delta_t$.\n"
            "- **Eras.** deep-contango 2010-2015 · recovery 2016-2020 · backwardation 2021-2026.\n"
            "- **Costs.** TR already nets ERs; add incremental bid-ask × reconstitution turnover (USCI the "
            "wider spread).\n"
            "- **Control.** `synthetic_world(roll_edge_annual)` plants a tunable constant edge; the null "
            "(0) must not fire."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The full-sample race — right sign, no significance\n\n"
            "Each leg's excess-of-cash return / vol / Sharpe, then the USCI−bench advantage with its "
            "paired-bootstrap 95% CI and the HAC *t* on the monthly difference."
        ),
        code(
            "if HAVE_REAL:\n"
            "    print(f\"{'leg':<6s} {'ex-ret%/yr':>10s} {'vol%':>6s} {'Sharpe':>7s}\")\n"
            "    print(f\"{'USCI':<6s} {COMMON['USCI'].mean()*1200:>+10.2f} {COMMON['USCI'].std(ddof=1)*np.sqrt(12)*100:>6.1f} {st.annualized_sharpe(COMMON['USCI'].to_numpy()):>+7.3f}\")\n"
            "    rows = []\n"
            "    for b in BENCH:\n"
            "        r = st.sharpe_race(EX, 'USCI', b)\n"
            "        print(f'{b:<6s} {r[\"ann_ex_bench\"]:>+10.2f} {r[\"vol_bench\"]:>6.1f} {r[\"sharpe_bench\"]:>+7.3f}')\n"
            "        rows.append((b, r['sharpe_adv'], r['adv_ci_lo'], r['adv_ci_hi'], r['diff_ann_pct'], r['t_diff']))\n"
            "else:\n"
            "    L = R['legs']\n"
            "    print(f\"{'leg':<6s} {'ex-ret%/yr':>10s} {'vol%':>6s} {'Sharpe':>7s}\")\n"
            "    for f in ['USCI'] + BENCH:\n"
            "        print(f\"{f:<6s} {L[f]['ex']:>+10.2f} {L[f]['vol']:>6.1f} {L[f]['sharpe']:>+7.3f}\")\n"
            "    rows = [(b, R['race'][b]['adv'], R['race'][b]['lo'], R['race'][b]['hi'], R['race'][b]['diff'], R['race'][b]['t']) for b in BENCH]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "y = np.arange(len(rows))\n"
            "advs = [r[1] for r in rows]; los = [r[1]-r[2] for r in rows]; his = [r[3]-r[1] for r in rows]\n"
            "ax.errorbar(advs, y, xerr=[los, his], fmt='o', color=GREEN, ecolor=GREY, capsize=4, ms=8)\n"
            "ax.axvline(0, c=RED, lw=1.2, ls='--')\n"
            "ax.set_yticks(y); ax.set_yticklabels([f'USCI - {r[0]}' for r in rows])\n"
            "ax.set_xlabel('excess-of-cash Sharpe advantage (95% bootstrap CI)')\n"
            "ax.set_title('Right sign, but every CI crosses zero'); plt.tight_layout(); plt.show()\n"
            "for b, a, lo, hi, d, t in rows: print(f'USCI - {b:<4s}: Sharpe adv {a:+.3f} CI[{lo:+.3f},{hi:+.3f}]  diff {d:+.2f}%/yr (HAC t={t:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: USCI wins all three head-to-heads on Sharpe (advantage +{rc['DBC']['adv']:.2f} "
            f"to +{rc['DJP']['adv']:.2f}), but **every** confidence interval touches or crosses zero and the "
            f"strongest return-difference *t* is only **+{rc['DJP']['t']:.2f}** (vs the pure front-month DJP). "
            "On this sample the edge is directional, not decisive."
        ),
        md(
            "### 4b · The era sign-flip — the decisive result\n\n"
            "The full-sample win is an average over regimes that disagree. Split it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mat = np.array([[e['diff_ann_pct'] for e in st.era_race(EX,'USCI',b,ERAS)] for b in BENCH])\n"
            "    tmat = np.array([[e['t_diff'] for e in st.era_race(EX,'USCI',b,ERAS)] for b in BENCH])\n"
            "else:\n"
            "    mat = np.array([[e[0] for e in R['era'][b]] for b in BENCH])\n"
            "    tmat = np.array([[e[1] for e in R['era'][b]] for b in BENCH])\n"
            "eras = ['deep contango\\n2010-2015', 'recovery\\n2016-2020', 'backwardation\\n2021-2026']\n"
            "x = np.arange(len(eras)); w = .25\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.8))\n"
            "for i, b in enumerate(BENCH):\n"
            "    ax.bar(x + (i-1)*w, mat[i], w, label=f'USCI - {b}', color=[GREEN,GREY,AMBER][i])\n"
            "ax.axhline(0, c=RED, lw=1.2); ax.set_xticks(x); ax.set_xticklabels(eras)\n"
            "ax.set_ylabel('USCI minus benchmark (%/yr)'); ax.legend()\n"
            "ax.set_title('USCI wins in contango eras, LOSES through 2016-2020 (t up to -3)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for i, b in enumerate(BENCH):\n"
            "    print(f'USCI - {b}:', [f'{mat[i][j]:+.1f}%/yr (t={tmat[i][j]:+.2f})' for j in range(3)])"
        ),
        md(
            f"> 💡 In plain words: across all three benchmarks the pattern is identical — a fat positive "
            f"edge in the deep-contango 2010-2015 and 2021+ backwardation eras, a **significant negative** "
            f"edge through the 2016-2020 recovery (USCI−DBC {er['DBC'][1][0]:.1f}%/yr at *t* = "
            f"{er['DBC'][1][1]:.2f}; USCI−DJP {er['DJP'][1][0]:.1f}%/yr at *t* = {er['DJP'][1][1]:.2f}). "
            "An edge that goes significantly the *wrong* way for a whole era fails H₂ — it is regime-"
            "contingent, not a constant roll premium. **This is why Signal is Weak, not Real.**"
        ),
        md(
            "### 4c · Costs & composition — why it's a Mirage, not a thin-but-Fragile edge\n\n"
            "Total returns already net each ER (USCI 1.03% vs GSG 0.48%). Add an incremental bid-ask "
            "charge on reconstitution turnover (USCI wider). Both legs are buy-and-hold, so friction is "
            "tiny — the advantage barely moves, meaning costs are **not** the blocker."
        ),
        code(
            "if HAVE_REAL:\n"
            "    for b in BENCH:\n"
            "        g = st.sharpe_race(EX,'USCI',b); c = st.costed_race(EX,'USCI',b)\n"
            "        print(f'USCI - {b:<4s}: gross adv {g[\"sharpe_adv\"]:+.3f} -> net {c[\"sharpe_adv_net\"]:+.3f}   '\n"
            "              f'gross diff {g[\"diff_ann_pct\"]:+.2f}%/yr -> net {c[\"diff_ann_pct_net\"]:+.2f}%/yr (t={c[\"t_diff_net\"]:+.2f})')\n"
            "    print()\n"
            "    for f in ['USCI'] + BENCH:\n"
            "        v = COMMON[f].std(ddof=1)*np.sqrt(12)*100 if f=='USCI' else st.sharpe_race(EX,'USCI',f)['vol_bench']\n"
            "        print(f'{f}: vol {v:.1f}%/yr, expense ratio {data.EXPENSE_RATIOS[f]:.2f}%')\n"
            "else:\n"
            "    for b in BENCH:\n"
            "        a, d, t = R['costed'][b]; g = R['race'][b]['adv']\n"
            "        print(f'USCI - {b:<4s}: gross adv {g:+.3f} -> net {a:+.3f}   net diff {d:+.2f}%/yr (t={t:+.2f})')\n"
            "    print(); \n"
            "    for f in ['USCI'] + BENCH: print(f\"{f}: vol {R['legs'][f]['vol']:.1f}%/yr, expense ratio {R['legs'][f]['er']:.2f}%\")"
        ),
        md(
            f"> 💡 In plain words: net of costs the advantage is still +{ct['DBC'][0]:.2f}/+{ct['GSG'][0]:.2f}/"
            f"+{ct['DJP'][0]:.2f} Sharpe — essentially unchanged, so friction isn't what defeats it. What "
            f"does: (i) the sign-flip above, and (ii) **composition** — USCI runs at {L['USCI']['vol']:.1f}% "
            f"vol vs GSG's {L['GSG']['vol']:.1f}% because it holds far less energy, so a large slice of its "
            "Sharpe edge is *lower-beta diversification*, not paid roll yield. A hidden tilt dressed as a "
            "roll premium ⇒ **Mirage**."
        ),
        md(
            "### 4d · Corroborating cousin — PDBC, same story on a shorter tape"
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = st.sharpe_race(EX, 'PDBC', 'GSG')\n"
            "    print(f\"PDBC - GSG (n={r['n']}, {r['start'][:7]}+): Sharpe adv {r['sharpe_adv']:+.3f} \"\n"
            "          f\"CI[{r['adv_ci_lo']:+.3f},{r['adv_ci_hi']:+.3f}]  diff {r['diff_ann_pct']:+.2f}%/yr (HAC t={r['t_diff']:+.2f})\")\n"
            "else:\n"
            "    p = R['pdbc']; print(f\"PDBC - GSG (n={p['n']}): Sharpe adv {p['adv']:+.3f} CI[{p['lo']:+.3f},{p['hi']:+.3f}]  diff {p['diff']:+.2f}%/yr (HAC t={p['t']:+.2f})\")"
        ),
        md(
            f"> 💡 In plain words: the second optimized wrapper (PDBC) also edges front-month GSG — Sharpe "
            f"adv +{R['pdbc']['adv']:.2f} — with a CI through zero and *t* = +{R['pdbc']['t']:.2f}. Same "
            "direction, same lack of robustness. Consistent, not conclusive."
        ),
        md(
            "### 4e · Faithful-engine control — we know the truth here\n\n"
            "Planted constant roll edge vs a null. The estimator must recover the knob and stay dark at 0. "
            "*(Machinery proof — never market evidence.)*"
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.03):\n"
            "    w = data.synthetic_world(roll_edge_annual=edge, seed=908)\n"
            "    d = st.synthetic_detect(w)\n"
            "    res.append((edge*100, d['sharpe_adv'], d['adv_ci_lo'], d['adv_ci_hi'], d['diff_ann_pct'], d['t_diff']))\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.2))\n"
            "y = np.arange(len(res)); advs=[r[1] for r in res]\n"
            "los=[r[1]-r[2] for r in res]; his=[r[3]-r[1] for r in res]\n"
            "ax.errorbar(advs, y, xerr=[los,his], fmt='o', color=GREEN, ecolor=GREY, capsize=4, ms=9)\n"
            "ax.axvline(0, c=RED, lw=1.2, ls='--'); ax.set_yticks(y)\n"
            "ax.set_yticklabels([f'planted {int(r[0])}%/yr' for r in res])\n"
            "ax.set_xlabel('recovered Sharpe advantage (95% CI)')\n"
            "ax.set_title('Null CI straddles 0; planted +3%/yr is recovered clear of 0')\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,a,lo,hi,d,t in res: print(f'planted {e:+.1f}%/yr -> adv {a:+.3f} CI[{lo:+.3f},{hi:+.3f}]  diff {d:+.2f}%/yr (HAC t={t:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: the null reads adv {R['syn'][0][1]:+.3f} (CI through 0, *t* = "
            f"{R['syn'][0][5]:+.2f}); a planted **+3%/yr** comes back at diff {R['syn'][1][4]:+.2f}%/yr, "
            f"*t* = {R['syn'][1][5]:+.2f}, CI clear of 0. The machinery **would** certify a genuine constant "
            "roll edge at this length — so the real-tape verdict is a fact about the tape (regime-flipping), "
            "not a lack of power."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — USCI wins the full-sample excess-of-cash Sharpe race vs all benchmarks "
            f"(+{L['USCI']['sharpe']:.3f} vs +{L['DBC']['sharpe']:.3f}/+{L['GSG']['sharpe']:.3f}/"
            f"+{L['DJP']['sharpe']:.3f}), the right sign, PDBC corroborating — but no return-difference "
            f"clears HAC *t* ≥ 2 (best +{rc['DJP']['t']:.2f}), every bootstrap advantage CI includes 0, and "
            f"the edge is **significantly negative** in 2016-2020 (−{abs(er['DBC'][1][0]):.1f}%/yr vs DBC, "
            f"*t* = {er['DBC'][1][1]:.2f}). Fails the era-robustness leg of the Real bar.\n"
            f"- **Tradability `MIRAGE`** — friction is negligible (buy-and-hold; ERs already embedded) so "
            "net ≈ gross, yet there's no bankable edge: the advantage could be zero, its sign is "
            "regime-contingent, and much of the raw Sharpe gap is a lower-energy/lower-vol **composition "
            "tilt**, not a paid roll premium. The roll-yield 'free lunch' is a contango-regime bet."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Conditional, not constant.** A curve-state-conditioned allocation (hold optimized only "
            "when the aggregate curve is in contango) is the natural follow-up — but that is *timing*, "
            "which is [35-contango](../../35-contango/README.md)'s question, not this one.\n"
            "- **Strip the composition.** Vol-match or energy-match USCI to GSG before racing to isolate "
            "the pure roll component from the diversification tilt.\n"
            "- **Dedup on the desk.** [35-contango](../../35-contango/README.md) times the curve; "
            "[794-commodity-carry](../../794-commodity-carry/README.md) sorts single commodities on carry; "
            "[661-uso-roll-decay](../../661-uso-roll-decay/README.md) is single-commodity roll decay; "
            "[226-crude-seasonality](../../226-crude-seasonality/README.md) a single-commodity calendar "
            "effect. This study races **whole packaged wrappers**, always invested.\n\n"
            "*Frozen numbers: [`docs/results.md`](../docs/results.md); sources: "
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
