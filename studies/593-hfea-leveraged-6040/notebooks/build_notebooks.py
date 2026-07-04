"""Generate the two narrative notebooks for Study 593 (HFEA — UPRO/TMF 55/45).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached
yfinance closes under ../_cache/ and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md). The synthetic control runs anywhere with no
network (seeds reduced in-cell; canonical numbers quoted from ``R``).
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SPY/TLT/
# UPRO/TMF/^IRX, 2002-07-31 -> 2026-06-30, 23.9 yrs, fingerprint b276ef14f488).
R = dict(
    start="2002-07-31", end="2026-06-30", years=23.9, n_months=288, asof="2026-06-30",
    fingerprint="b276ef14f488",
    # synthesis validation: (leg, fee %/yr, corr, te %, nav_ratio)
    cal=[("UPRO (3x SPY)", 2.13, 0.9983, 3.01, 1.0002),
         ("TMF (3x TLT)", 2.19, 0.9967, 3.64, 0.9998)],
    n_synth_days=1739, n_real_days=4278, n_resets=96, reset_turnover_pct=13.5,
    # full-period table: (name, cagr, vol, maxdd, sharpe, wealth multiple or None)
    perf=[("HFEA 55/45", 17.47, 29.6, -70.8, 0.64, 46.8),
          ("SPY", 11.24, 18.8, -55.2, 0.68, 12.7),
          ("60/40 SPY/TLT", 8.99, 10.7, -30.1, 0.75, 7.8),
          ("UPRO leg (3x SPY)", 16.98, 56.4, -95.6, 0.55, None),
          ("TMF leg (3x TLT)", -0.39, 42.6, -92.7, 0.15, None)],
    # races: (bench, gap %/yr, HAC t)
    race_spy=(5.42, 1.31), race_6040=(7.45, 1.91),
    # regime split vs SPY: (window, gap, t)
    reg_pre=("2002-08..2021-12", 11.22, 2.98), reg_post=("2022-01..2026-06", -19.71, -1.59),
    welch_regime=2.23, reg_pre_6040=(11.97, 3.39),
    pre_cagr=(24.28, 0.92, 11.03, 0.72),    # HFEA cagr, sharpe, SPY cagr, sharpe
    post_cagr=(-8.02, -0.09, 12.18, 0.55),
    # 2022 autopsy
    y2022=[("UPRO leg", -56.8), ("TMF leg", -72.6), ("HFEA 55/45", -64.2),
           ("SPY", -18.2), ("60/40", -23.5), ("TLT (1x)", -31.2)],
    corr=dict(pre=-0.30, y2022=0.51, recent=0.32, full=-0.10),
    dd=dict(peak="2021-12-27", trough="2023-10-27", depth=-70.8, now=-32.2),
    y2008=(-27.0, -36.8), y2020=(66.5, 18.3),
    # costs: (one-way bps, net cagr, gap, t, drag bps/yr)
    costs=[(2.0, 17.46, 5.41, 1.31, 1.1), (5.0, 17.44, 5.40, 1.30, 2.7),
           (10.0, 17.41, 5.37, 1.30, 5.4)],
    fee3=(17.19, 5.18, 1.25),               # pessimistic 3% synthesis fee: cagr, gap, t
    real_only=(23.07, 0.81, 15.16, 0.96, 6.61, 1.25),
    # synthetic control: (label, rho, carry %, mean gap, mean t, sd t, share>=2 %)
    syn=[("PLANTED", -0.6, 6, 10.36, 4.90, 0.88, 100),
         ("NULL", 0.6, 0, -3.09, -0.83, 1.03, 0)],
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![2022 falsified it?: Mixed](https://img.shields.io/badge/2022_falsified_it%3F-Mixed-8b949e?style=flat-square)\n\n"
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

from hfea_leveraged_6040 import data as d, strategy as st

HAVE_REAL = d.have_real()
if HAVE_REAL:
    PX = d.load_real()
    TAPE = d.build_tape(PX, asof=d.AS_OF)
    CAL_S = d.calibrate_fee(TAPE["spy"], TAPE["rf"], TAPE["upro"])
    CAL_B = d.calibrate_fee(TAPE["tlt"], TAPE["rf"], TAPE["tmf"])
    LEGS = d.spliced_legs(TAPE, CAL_S["fee_ann"], CAL_B["fee_ann"])
    HFEA = st.rebalanced(LEGS["s3x"], LEGS["b3x"], st.HFEA_W)["ret"]
    SIXTY = st.rebalanced(LEGS["spy"], LEGS["tlt"], st.SIXTY_FORTY_W)["ret"]
    RF = LEGS["rf"]
else:
    PX = TAPE = LEGS = HFEA = SIXTY = RF = None
print("real HFEA cache present:", HAVE_REAL,
      "| days:", (0 if LEGS is None else len(LEGS)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Hedgefundie's Excellent Adventure — can 3x stocks + 3x bonds beat the market forever? 🎢\n"
            "### The most famous retail leverage recipe, run honestly through 24 years — in plain English\n\n"
            + BADGES +
            "In 2019 an anonymous user called **Hedgefundie** posted a recipe on the Bogleheads forum "
            "that became legend: put **55% in UPRO** (a fund that moves 3× the S&P 500 every day) and "
            "**45% in TMF** (3× long-term US Treasuries), rebalance every quarter, and hold for decades. "
            "The pitch: stocks and bonds usually move in **opposite directions**, so each leg insures the "
            "other — letting you run triple leverage without triple ruin. The backtest looked like a "
            "money machine.\n\n"
            "Then **2022** happened: inflation made stocks *and* bonds crash together, and the strategy "
            "lost **64% in one year**. So — was the whole idea a mirage, or did it just take a punch?\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the synthesis validation and the regime "
            "tests? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Every chart is drawn by the code beside it** from cached real market data (yfinance; "
            "the funds are real from 2009, faithfully reconstructed before — checked against the real "
            "funds at 99.8% correlation). House style: [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did HFEA compound faster than the S&P 500? | **In raw dollars, yes — hugely.** $1 in 2002 "
            "became **$46.8** vs **$12.7** in SPY, *even counting* the 2022 disaster. But statistically the "
            "edge is too noisy to certify (the strategy swings ~30% a year), and **all** of it came from "
            "one regime. |\n"
            "| Was the \"bonds insure stocks\" engine real? | **Yes — until it wasn't.** From 2002–2021 "
            "stocks and bonds moved oppositely and HFEA beat SPY by **+11%/yr** (statistically solid). "
            "In 2022 the relationship **flipped** and both legs crashed together. |\n"
            "| Is it a better deal than just SPY, per unit of risk? | **No.** Its Sharpe ratio (return "
            "per unit of risk) is **0.64 vs SPY's 0.68** — you took triple the rollercoaster for *less* "
            "reward per unit of fear. |\n"
            "| Did 2022 kill the thesis? | **It killed the \"law\", not the arithmetic.** The recipe still "
            "out-compounded SPY over the full 24 years — but 2022 proved the insurance leg only works in "
            "the *right regime*, and it's been the wrong regime since. |\n\n"
            "> HFEA is a **regime bet with a leverage amplifier** — spectacular when stock-bond "
            "correlation is negative, catastrophic when it flips. It is not a money machine."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A 55/45 mix of 3x stocks and 3x long bonds, rebalanced quarterly, compounds faster than "
            "the S&P 500 — because the two legs are negatively correlated, the bond leg insures the stock "
            "leg, and rebalancing systematically buys whichever just crashed.\"*\n\n"
            "That's **leveraged diversification** — the retail version of what risk-parity hedge funds do "
            "(the academic case is real: Asness, Frazzini & Pedersen 2012). Our sibling studies "
            "[61](../../61-slow-burn/README.md) and [100](../../100-melting-ice/README.md) tested whether a *single* 3x fund decays to "
            "nothing (it doesn't — but it's ruinous alone). This study asks the next question: does the "
            "**pair** — the *allocation* — actually work?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Tens of thousands of retail investors run some version of this today. If the claim is true, "
            "a patient investor triples the market's compounding with survivable dips. If it's false, "
            "they're holding a leveraged time bomb whose fuse is a single macro variable — the stock-bond "
            "correlation — that nobody controls. 2022 was the live test: bonds were supposed to be the "
            "airbag, and the airbag exploded *with* the car."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **Real funds, extended honestly.** UPRO and TMF exist since mid-2009. Before that we "
            f"reconstruct each leg with the exact daily-leverage arithmetic (3× the index, minus borrow "
            f"costs and fees), **calibrated and checked against the real funds** — 99.8% daily "
            f"correlation. Full tape: **{R['start']} → {R['end']}** ({R['years']:.0f} years).\n"
            "- **The recipe, verbatim.** 55/45, reset at every quarter-end close (earning from the next "
            "day — no time travel), trading costs charged on every reset.\n"
            "- **Fair races.** HFEA vs SPY and vs a classic unlevered 60/40, on total returns, with the "
            "statistical bar the desk applies to everything: a *t* ≥ 2 on the real tape.\n"
            "- **The autopsy.** Then we put 2022 on the table and ask what actually failed."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The mountain and the cliff.** Here's $1 invested in August 2002, on a log scale (every "
            "gridline is 10×)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    navs = {'HFEA 55/45': (1+HFEA).cumprod(), 'SPY': (1+LEGS['spy']).cumprod(),\n"
            "            '60/40 SPY/TLT': (1+SIXTY).cumprod()}\n"
            "    fig, ax = plt.subplots()\n"
            "    for (nm, nav), c in zip(navs.items(), [GREEN, GREY, AMBER]):\n"
            "        ax.plot(nav.index, nav.values, color=c, lw=1.8, label=f'{nm}  (x{nav.iloc[-1]:.1f})')\n"
            "    ax.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2023-10-27'), color=RED, alpha=.12,\n"
            "               label='2022 corr-flip crash')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log scale)'); ax.legend(loc='upper left')\n"
            "    ax.set_title('HFEA out-compounds SPY over 24 years - through a -71% canyon')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print({nm: round(float(nav.iloc[-1]),1) for nm, nav in navs.items()})\n"
            "else:\n"
            "    print('cache missing - canonical multiples:', {p[0]: p[5] for p in R['perf'][:3]})"
        ),
        md(
            f"$1 became **${R['perf'][0][5]:.1f}** in HFEA vs **${R['perf'][1][5]:.1f}** in SPY and "
            f"**${R['perf'][2][5]:.1f}** in 60/40 — *including* the 2022 wreck (red band). And here's the "
            f"quiet magic: the 55/45 **pair** (+{R['perf'][0][1]:.1f}%/yr) out-compounded **both of its "
            f"own ingredients** (3x stocks alone: +{R['perf'][3][1]:.1f}%/yr; 3x bonds alone: "
            f"{R['perf'][4][1]:.1f}%/yr). That's the diversification-plus-rebalancing engine genuinely "
            "working — when the correlation cooperates.\n\n"
            "**But now the price tag.** Same strategy, drawn as *distance below its own record high*:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    fig, ax = plt.subplots()\n"
            "    for r, nm, c in [(HFEA, 'HFEA 55/45', GREEN), (LEGS['spy'], 'SPY', GREY)]:\n"
            "        nav = (1+r).cumprod(); dd = nav/nav.cummax() - 1\n"
            "        ax.fill_between(dd.index, dd.values*100, 0, color=c, alpha=.45, label=nm)\n"
            "    ax.set_ylabel('drawdown from peak (%)'); ax.legend(loc='lower left')\n"
            "    ax.set_title('The cost of leverage: a -71% canyon, still -32% deep in mid-2026')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    dd_state = st.drawdown_state(HFEA)\n"
            "    print(dd_state)\n"
            "else:\n"
            "    print('canonical:', R['dd'])"
        ),
        md(
            f"HFEA peaked on **{R['dd']['peak']}**, fell **{R['dd']['depth']:.0f}%** to its "
            f"**{R['dd']['trough']}** trough, and in June 2026 — four and a half years later — still sits "
            f"**{R['dd']['now']:.0f}%** below that peak while SPY makes new highs. Per unit of risk, HFEA "
            f"was *never* a better deal: Sharpe **{R['perf'][0][4]:.2f}** vs SPY's "
            f"**{R['perf'][1][4]:.2f}** vs 60/40's **{R['perf'][2][4]:.2f}**.\n\n"
            "**What actually broke in 2022?** The insurance. Stocks and bonds had moved in opposite "
            "directions for twenty years; in 2022 inflation flipped the relationship and both 3x legs "
            "crashed *together*:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    names = [n for n,_ in R['y2022']]\n"
            "    vals = [st.year_return(r, 2022) for r in (LEGS['s3x'], LEGS['b3x'], HFEA,\n"
            "                                              LEGS['spy'], SIXTY, LEGS['tlt'])]\n"
            "else:\n"
            "    names = [n for n,_ in R['y2022']]; vals = [v for _,v in R['y2022']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "cols = [RED, RED, RED, GREY, GREY, GREY]\n"
            "ax.bar(names, vals, color=cols, width=.6)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.0f}%', (i,v), ha='center',\n"
            "    va='top' if v<0 else 'bottom')\n"
            "ax.set_ylabel('2022 calendar-year return (%)')\n"
            "ax.set_title('2022: the airbag exploded with the car - both 3x legs crashed together')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dict(zip(names, [round(v,1) for v in vals])))"
        ),
        md(
            f"The bond leg — the *insurance* — lost **{R['y2022'][1][1]:.0f}%**, more than the stock leg "
            f"({R['y2022'][0][1]:.0f}%). The pair lost **{R['y2022'][2][1]:.0f}%** in a year when plain "
            f"SPY lost {R['y2022'][3][1]:.0f}%. The stock-bond correlation, **{R['corr']['pre']:.2f}** "
            f"over 2002–2021, jumped to **+{R['corr']['y2022']:.2f}** in 2022 — and it's still positive "
            f"(+{R['corr']['recent']:.2f} in 2024–26). Compare the years the recipe was built on: in "
            f"**2008** HFEA lost only {R['y2008'][0]:.0f}% while SPY lost {R['y2008'][1]:.0f}% (3x bonds "
            f"*doubled* that year), and in **2020** HFEA made **+{R['y2020'][0]:.0f}%**. When the "
            "correlation is negative, the machine is glorious. When it flips, the machine is a trap."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Mixed.** Over the full 24 years HFEA out-compounded SPY by ~{R['race_spy'][0]:.0f}%/yr "
            "in raw CAGR — but the ride is so violent that statistics can't certify the gap (t ≈ 1.3, "
            "below the desk's bar of 2). Split by the mechanism itself: **before 2022** the edge was "
            f"+{R['reg_pre'][1]:.0f}%/yr and statistically real (t ≈ 3); **since the correlation flip** it's "
            f"{R['reg_post'][1]:.0f}%/yr. Real in one regime, reversed in the other.\n"
            "- **Tradability — Fragile.** The ETFs are liquid and quarterly rebalancing costs almost "
            "nothing — but you're signing up for −71% drawdowns, half-decades under water, *worse* "
            "risk-adjusted returns than plain SPY, and a thesis that lives or dies on a macro variable "
            "that flipped against it in 2022 and hasn't flipped back.\n"
            "- **Did 2022 falsify it? — Mixed.** It falsified the *\"bonds always insure stocks\"* law and "
            "the risk-adjusted case. It did **not** erase the raw compounding arithmetic — even including "
            "the disaster, $1 → $46.8 vs SPY's $12.7. HFEA works *when and only when* its regime holds."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The single-fund story** — is a 3x ETF alone doomed by \"volatility decay\"? Siblings "
            "[61 — slow-burn](../../61-slow-burn/README.md) and [100 — melting-ice](../../100-melting-ice/README.md) measure that "
            "race exactly (spoiler: decay is real maths but path-dependent, and the true killer is the "
            "drawdown).\n"
            "- **The correlation regime itself** — when and why does the stock-bond sign flip? "
            "[579 — equity-bond-corr-flip](../../579-equity-bond-corr-flip/README.md) tests it head-on: it's an "
            "inflation-regime variable, not a constant of nature.\n"
            "- **Try to break it.** Re-run with monthly or annual rebalancing (we did — quarterly wasn't "
            "cherry-picked), a pessimistic borrow fee on the pre-2009 reconstruction, or the real-funds-"
            "only window. The verdict doesn't move.\n\n"
            "*Think the next decade brings back negative correlation and cheap money? That's the actual "
            "bet HFEA makes — say it out loud before you size it.*"
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
            "# HFEA (UPRO/TMF 55/45) — a quantitative teardown 🔬\n"
            "### Per-leg synthesis calibrated & validated on the real funds · HAC log-gap races vs SPY "
            "and 60/40 · a mechanism-based regime split with a Welch difference test · the 2022 "
            "correlation autopsy · cost sweep · a planted/removed diversification-engine control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The claim "
            "(Bogleheads, 2019): **55% UPRO / 45% TMF, quarterly rebalanced, compounds faster than SPY "
            "thanks to leveraged diversification.** Distinct from the desk's single-LETF decay studies "
            "([61](../../61-slow-burn/README.md), [100](../../100-melting-ice/README.md)) — this is the *portfolio allocation* "
            "claim.\n\n"
            "> ⚠️ **Data note.** yfinance daily total-return closes (SPY, TLT, UPRO, TMF) + ^IRX as the "
            "bill/financing leg; real funds from mid-2009, per-leg daily-leverage synthesis before "
            "(calibrated on the overlap, validated at corr 0.998/0.997). Window 2002-07 → 2026-06-30, "
            "gross unless labeled net; Sharpe races excess-vs-excess. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fingerprint"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | Full-period log-CAGR gap vs SPY **+{R['race_spy'][0]:.2f}%/yr** at "
            f"**HAC t = {R['race_spy'][1]:.2f}** (below the bar); pre-2022 **+{R['reg_pre'][1]:.2f}%/yr at "
            f"t = {R['reg_pre'][2]:.2f}**, post-flip **{R['reg_post'][1]:.2f}%/yr**; regime difference "
            f"**Welch t = {R['welch_regime']:.2f}**. Real on the ρ<0 regime, reversed since. |\n"
            f"| **Tradability** | `FRAGILE` | Costs ≤ {R['costs'][2][4]:.0f} bps/yr (not the problem); "
            f"maxDD **{R['perf'][0][3]:.1f}%**, still {R['dd']['now']:.0f}% under water, Sharpe "
            f"**{R['perf'][0][4]:.2f} < {R['perf'][1][4]:.2f}** (SPY), insurance leg regime-dependent. |\n"
            f"| **2022 falsified it?** | `MIXED` | Corr {R['corr']['pre']:+.2f} → +{R['corr']['y2022']:.2f}; "
            f"both legs {R['y2022'][0][1]:.0f}%/{R['y2022'][1][1]:.0f}% in 2022; yet full-tape wealth "
            f"×{R['perf'][0][5]:.1f} vs SPY ×{R['perf'][1][5]:.1f}. A regime exposed, not a law dented. |\n\n"
            "> 💡 In plain words: the engine is real machinery, the full-period edge is statistically "
            "uncertifiable, and the load-bearing assumption is a regime that broke in 2022."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{s3}_t, r^{b3}_t$ be daily returns of the 3x legs and $w = (0.55, 0.45)$. The "
            "portfolio drifts between quarter-ends and resets at each quarter's last close (earning from "
            "$t{+}1$ — the one documented lag):\n\n"
            "$$r^{P}_t = w^{s}_t r^{s3}_t + w^{b}_t r^{b3}_t, \\qquad w_{t^+} = (0.55, 0.45)\\ \\text{at "
            "quarter ends.}$$\n\n"
            "- **H₁ (compounds faster).** $\\mathbb{E}[\\log(1+r^P_m) - \\log(1+r^{SPY}_m)] > 0$, HAC "
            "t ≥ 2 (a geometric claim races in logs).\n"
            "- **H₂ (the engine is diversification).** The edge requires ρ(stocks, bonds) < 0 plus bond "
            "carry — remove either and the pair should lose to the index (tested in the synthetic "
            "control *and* on the 2022 regime).\n"
            "- **H₃ (survivable).** The drawdown/Sharpe profile beats the naked 3x fund and doesn't "
            "surrender the risk-adjusted race to plain SPY.\n\n"
            "We find **H₁ uncertified** on the full tape (t = 1.31) and regime-split (pre-2022 t = 2.98, "
            "post-flip negative), **H₂ confirmed** in both directions, **H₃ half-true** (beats the naked "
            "leg; loses the Sharpe race to SPY)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The inference design has three honesty problems to kill. **(a) Geometric vs arithmetic:** "
            "\"compounds faster\" is about log growth, so the race statistic is the mean monthly "
            "**log**-return gap with Newey-West errors (monthly returns of a 30%-vol levered pair are "
            "serially dependent through vol clustering). **(b) The pre-launch tape:** UPRO/TMF only "
            "exist since 2009 — the 2002–09 extension must be *validated*, not assumed; we calibrate the "
            "all-in fee on the overlap (terminal-NAV match) and require daily corr > 0.99, study-100's "
            "bar. **(c) The regime split must not be snooped:** we split at 2022-01 because that is the "
            "claim's own mechanism failing (the corr flip — an ex-ante, macro-dated marker), and we test "
            "the *difference* with a Welch t, per the desk's inference bar."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** {R['start']} → {R['end']} ({R['years']:.1f}y, {R['n_months']} months): real "
            f"funds {R['n_real_days']:,} days, synthesis {R['n_synth_days']:,} days (pre-launch).\n"
            f"- **Rule.** 55/45, reset at each quarter-end close ({R['n_resets']} resets, avg "
            f"{R['reset_turnover_pct']:.1f}% NAV traded); new weights earn from the next session.\n"
            "- **Races.** vs SPY and vs 60/40 SPY/TLT (same machinery); HAC t on the monthly log gap; "
            "excess-vs-excess Sharpe (monthly, minus compounded ^IRX).\n"
            "- **Costs.** One-way 2/5/10 bps × NAV traded per reset; fund ER + financing already inside "
            "the NAVs (and inside the calibrated synthesis fee).\n"
            "- **Autopsy.** 2022 per-leg damage, the corr flip (monthly SPY/TLT corr by era), drawdown "
            "state at the as-of.\n"
            "- **Control.** Two-asset seeded worlds, 3x legs, 55/45 quarterly: diversification engine "
            "PLANTED (ρ = −0.6, +6% bond carry) vs REMOVED (ρ = +0.6, zero carry), ≥ 20 seeds averaged."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Synthesis calibration & validation (the pre-launch tape has to earn its place)\n\n"
            "Daily identity $r^{3x}_t = 3 r^{idx}_t - 2 r^{bill}_t - f/252$; $f$ calibrated so the "
            "synthetic terminal NAV matches the real fund's over the overlap, then judged on daily "
            "correlation and tracking error."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [('UPRO (3x SPY)', CAL_S), ('TMF (3x TLT)', CAL_B)]\n"
            "    for nm, c in rows:\n"
            "        print(f\"{nm}: fee {c['fee_ann']*100:.2f}%/yr  corr {c['corr']:.4f}  \"\n"
            "              f\"TE {c['te_ann']*100:.2f}%/yr  NAV ratio {c['nav_ratio']:.4f}  ({c['n_days']:,} d)\")\n"
            "    # visual: synthetic vs real UPRO NAV on the overlap\n"
            "    ov = pd.concat({'idx': TAPE['spy'], 'rf': TAPE['rf'], 'real': TAPE['upro']}, axis=1).dropna()\n"
            "    syn = d.synth_letf(ov['idx'], ov['rf'], CAL_S['fee_ann'])\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot((1+ov['real']).cumprod(), color=GREY, lw=2.2, label='real UPRO')\n"
            "    ax.plot((1+syn).cumprod(), color=GREEN, lw=1.0, label='synthetic 3x SPY (calibrated)')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)'); ax.legend()\n"
            "    ax.set_title(f\"Synthesis validation: daily corr {CAL_S['corr']:.4f} - the curves are one line\")\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    for nm, fee, corr, te, nav in R['cal']:\n"
            "        print(f'{nm}: fee {fee:.2f}%/yr  corr {corr:.4f}  TE {te:.2f}%  NAV ratio {nav:.4f}')"
        ),
        md(
            f"> 💡 In plain words: the reconstruction is the real fund to within rounding — corr "
            f"**{R['cal'][0][2]:.4f}** (UPRO) / **{R['cal'][1][2]:.4f}** (TMF), all-in drag "
            f"≈ **{R['cal'][0][1]:.1f}–{R['cal'][1][1]:.1f}%/yr** (ER + swap spread + 2× bills). The "
            "2002–09 extension is earned, not assumed — and a pessimistic 3%/yr fee barely moves the "
            f"headline (gap {R['fee3'][1]:+.2f}%/yr, t = {R['fee3'][2]:.2f})."
        ),
        md(
            "### 4b · The race — full-period performance and the HAC log-gap\n\n"
            "The full table, then the claim's own statistic: the mean monthly log-return gap vs SPY and "
            "vs 60/40, with Newey-West errors."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [('HFEA 55/45', HFEA), ('SPY', LEGS['spy']), ('60/40', SIXTY),\n"
            "            ('UPRO leg', LEGS['s3x']), ('TMF leg', LEGS['b3x'])]\n"
            "    for nm, r in rows:\n"
            "        p = st.perf(r, RF)\n"
            "        print(f\"{nm:12s} CAGR {p['cagr_pct']:+7.2f}%  vol {p['vol_ann_pct']:5.1f}%  \"\n"
            "              f\"maxDD {p['max_dd_pct']:6.1f}%  Sharpe {p['sharpe']:5.2f}\")\n"
            "    r1, r2 = st.race(HFEA, LEGS['spy'], RF), st.race(HFEA, SIXTY, RF)\n"
            "    print(f\"\\nvs SPY  : gap {r1['dlog_ann_pct']:+.2f}%/yr  HAC t = {r1['t_dlog_hac']:+.2f}\")\n"
            "    print(f\"vs 60/40: gap {r2['dlog_ann_pct']:+.2f}%/yr  HAC t = {r2['t_dlog_hac']:+.2f}\")\n"
            "    gaps = [(r1['dlog_ann_pct'], r1['t_dlog_hac']), (r2['dlog_ann_pct'], r2['t_dlog_hac'])]\n"
            "else:\n"
            "    gaps = [R['race_spy'], R['race_6040']]\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "ax.bar(['vs SPY', 'vs 60/40'], [g[1] for g in gaps], color=AMBER, width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i, g in enumerate(gaps): ax.annotate(f'gap {g[0]:+.1f}%/yr\\nt = {g[1]:.2f}', (i, g[1]),\n"
            "    ha='center', va='bottom')\n"
            "ax.set_ylabel('HAC t of monthly log gap'); ax.set_ylim(0, 3.4)\n"
            "ax.set_title('Full period: a big gap the tape cannot certify'); ax.legend()\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: **+{R['race_spy'][0]:.1f}%/yr** of extra compounding sounds enormous — "
            f"but a ~30%-vol strategy needs far more than 24 years to certify it: **HAC t = "
            f"{R['race_spy'][1]:.2f}**, below the bar. And the Sharpe table is the quiet damnation: "
            f"**{R['perf'][0][4]:.2f}** (HFEA) < **{R['perf'][1][4]:.2f}** (SPY) < "
            f"**{R['perf'][2][4]:.2f}** (60/40) — per unit of risk, the leverage bought *nothing*. Note "
            f"also the allocation fact: the pair (+{R['perf'][0][1]:.2f}%) beats both of its own legs "
            f"(+{R['perf'][3][1]:.2f}% / {R['perf'][4][1]:.2f}%) — the rebalancing engine itself is real."
        ),
        md(
            "### 4c · The regime split — the claim's own mechanism, tested\n\n"
            "Split at 2022-01: the macro-dated stock-bond correlation flip (ex-ante marker, not "
            "return-snooped). Per the inference bar, the sub-period contrast carries a Welch t of the "
            "**difference**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mp, mb = st.monthly(HFEA), st.monthly(LEGS['spy'])\n"
            "    dlog = np.log1p(mp) - np.log1p(mb)\n"
            "    pre = dlog[dlog.index <= pd.Period('2021-12','M')].to_numpy()\n"
            "    post = dlog[dlog.index >= pd.Period('2022-01','M')].to_numpy()\n"
            "    hp, hq = st.hac_mean_t(pre), st.hac_mean_t(post)\n"
            "    wt = (pre.mean()-post.mean())/np.sqrt(pre.var(ddof=1)/len(pre)+post.var(ddof=1)/len(post))\n"
            "    reg = [(hp['mean']*12*100, hp['t']), (hq['mean']*12*100, hq['t'])]\n"
            "    print(f'pre-2022 : {reg[0][0]:+.2f}%/yr  t={reg[0][1]:+.2f} (n={len(pre)})')\n"
            "    print(f'post-flip: {reg[1][0]:+.2f}%/yr  t={reg[1][1]:+.2f} (n={len(post)})')\n"
            "    print(f'Welch t of the regime difference = {wt:+.2f}')\n"
            "else:\n"
            "    reg = [R['reg_pre'][1:], R['reg_post'][1:]]; wt = R['welch_regime']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['2002-2021\\n(rho < 0 era)', '2022-2026\\n(rho > 0 era)'], [reg[0][0], reg[1][0]],\n"
            "       color=[GREEN, RED], width=.5)\n"
            "for i, (g, t) in enumerate(reg): ax.annotate(f'{g:+.1f}%/yr\\nHAC t = {t:+.2f}', (i, g),\n"
            "    ha='center', va='bottom' if g > 0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('log-CAGR gap vs SPY (%/yr)')\n"
            "ax.set_title(f'One claim, two regimes (Welch t of difference = {wt:.2f})')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: pre-2022 the claim was **real on the tape** — +{R['reg_pre'][1]:.1f}%/yr "
            f"at HAC t = {R['reg_pre'][2]:.2f} vs SPY (and +{R['reg_pre_6040'][0]:.1f}%/yr at t = "
            f"{R['reg_pre_6040'][1]:.2f} vs 60/40), with HFEA Sharpe {R['pre_cagr'][1]:.2f} *above* SPY's "
            f"{R['pre_cagr'][3]:.2f}. Since the flip it runs {R['reg_post'][1]:+.1f}%/yr (HFEA CAGR "
            f"{R['post_cagr'][0]:+.1f}%, Sharpe {R['post_cagr'][1]:.2f} vs SPY's {R['post_cagr'][3]:.2f}), "
            f"and the two eras are genuinely different (Welch t = {R['welch_regime']:.2f}). The signal is "
            "**regime-conditional** — hence `MIXED`, spelled out."
        ),
        md(
            "### 4d · The 2022 autopsy — the correlation flip in one chart\n\n"
            "Rolling 24-month SPY/TLT correlation (monthly returns). The whole thesis lives below zero."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rc = st.rolling_corr(LEGS, window=24).dropna()\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(rc.index.to_timestamp(), rc.values, color=GREY, lw=1.6)\n"
            "    ax.fill_between(rc.index.to_timestamp(), rc.values, 0, where=(rc.values<0), color=GREEN, alpha=.25,\n"
            "                    label='rho < 0: the insurance works')\n"
            "    ax.fill_between(rc.index.to_timestamp(), rc.values, 0, where=(rc.values>=0), color=RED, alpha=.25,\n"
            "                    label='rho > 0: both legs crash together')\n"
            "    ax.axhline(0, c='k', lw=.8); ax.set_ylabel('rolling 24m SPY/TLT correlation')\n"
            "    ax.set_title('The load-bearing assumption flipped in 2022 and has not flipped back')\n"
            "    ax.legend(loc='upper left'); plt.tight_layout(); plt.show()\n"
            "    print('era correlations:', {'2002-2021': round(st.stock_bond_corr(LEGS, end='2021-12'),2),\n"
            "          '2022': round(st.stock_bond_corr(LEGS, start='2022-01', end='2022-12'),2),\n"
            "          '2024-2026': round(st.stock_bond_corr(LEGS, start='2024-01'),2)})\n"
            "    print('2022 damage:', {nm: round(st.year_return(r, 2022),1) for nm, r in\n"
            "          [('UPRO leg', LEGS['s3x']), ('TMF leg', LEGS['b3x']), ('HFEA', HFEA), ('SPY', LEGS['spy'])]})\n"
            "else:\n"
            "    print('canonical:', R['corr'], R['y2022'])"
        ),
        md(
            f"> 💡 In plain words: monthly stock-bond correlation was **{R['corr']['pre']:.2f}** for two "
            f"decades — exactly the world HFEA was designed in — then hit **+{R['corr']['y2022']:.2f}** in "
            f"2022 and still sits at **+{R['corr']['recent']:.2f}** in 2024–26. In that year the "
            f"*insurance* leg lost {R['y2022'][1][1]:.0f}% — more than the stock leg — and the pair gave "
            f"back {R['y2022'][2][1]:.0f}%. Contrast 2008 (HFEA {R['y2008'][0]:.0f}% vs SPY "
            f"{R['y2008'][1]:.0f}%, the 3x bond leg +110%) and 2020 (+{R['y2020'][0]:.0f}%): same machine, "
            "opposite regime, opposite outcome."
        ),
        md(
            "### 4e · Costs & robustness — quarterly rebalancing is (almost) free\n\n"
            "One-way costs × NAV traded per reset, plus the pessimistic-fee and real-funds-only checks."
        ),
        code(
            "if HAVE_REAL:\n"
            "    for cb in (2.0, 5.0, 10.0):\n"
            "        h = st.rebalanced(LEGS['s3x'], LEGS['b3x'], st.HFEA_W, cost_bps=cb)\n"
            "        p = st.perf(h['ret'], RF); rr = st.race(h['ret'], LEGS['spy'], RF)\n"
            "        print(f\"{cb:4.1f} bps: net CAGR {p['cagr_pct']:+.2f}%  gap {rr['dlog_ann_pct']:+.2f}%/yr  \"\n"
            "              f\"t = {rr['t_dlog_hac']:+.2f}  (drag {h['ann_cost_bps']:.1f} bps/yr)\")\n"
            "    real = LEGS[LEGS['src'] == 'real']\n"
            "    hr = st.rebalanced(real['s3x'], real['b3x'], st.HFEA_W)['ret']\n"
            "    rr = st.race(hr, real['spy'], real['rf'])\n"
            "    pr, ps = st.perf(hr, real['rf']), st.perf(real['spy'], real['rf'])\n"
            "    print(f\"real-only 2009->: gap {rr['dlog_ann_pct']:+.2f}%/yr t = {rr['t_dlog_hac']:+.2f}  \"\n"
            "          f\"Sharpe {pr['sharpe']:.2f} vs SPY {ps['sharpe']:.2f}\")\n"
            "else:\n"
            "    for c in R['costs']: print(c)\n"
            "    print('real-only:', R['real_only'])"
        ),
        md(
            f"> 💡 In plain words: even at 10 bps one-way the annual drag is ~{R['costs'][2][4]:.0f} bps — "
            "costs are a rounding error at quarterly cadence (the funds' internal ~2%/yr financing+ER "
            "drag is already inside the NAVs). Tradability fails on **risk**, not friction: "
            f"−{abs(R['perf'][0][3]):.0f}% drawdown, {abs(R['dd']['now']):.0f}% still unrecovered, and a "
            f"real-funds-only Sharpe of {R['real_only'][1]:.2f} vs SPY's {R['real_only'][3]:.2f}. Hence "
            "`FRAGILE`."
        ),
        md(
            "### 4f · Faithful-engine control — plant the engine, then remove it\n\n"
            "Seeded two-asset worlds, 3x legs, 55/45 quarterly, race vs the 1x stock. PLANTED: ρ = −0.6 "
            "with +6% bond carry (the engine at full strength). NULL: ρ = +0.6, zero carry — the 2022 "
            "world made permanent. Averaged over seeds (canonical run: 20 seeds × 40y in "
            "`examples/verify.py`; this cell runs a lighter 8 × 25y for speed)."
        ),
        code(
            "res = []\n"
            "for nm, rho, carry in [('PLANTED', -0.6, 0.06), ('NULL', +0.6, 0.0)]:\n"
            "    c = st.synthetic_check(rho=rho, bond_mu_exc=carry, n_seeds=8, n_years=25)\n"
            "    res.append((nm, c))\n"
            "    print(f\"{nm:8s} rho={rho:+.1f} carry={carry*100:.0f}%: mean gap {c['mean_gap_ann_pct']:+.2f}%/yr  \"\n"
            "          f\"mean t = {c['mean_t']:+.2f}  share t>=2: {c['share_t_ge_2']*100:.0f}%\")\n"
            "print('canonical (20 seeds x 40y):', R['syn'])\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "ax.bar([r[0] for r in res], [r[1]['mean_t'] for r in res], color=[GREEN, GREY], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i, r in enumerate(res): ax.annotate(f\"t = {r[1]['mean_t']:.2f}\", (i, r[1]['mean_t']),\n"
            "    ha='center', va='bottom')\n"
            "ax.set_ylabel('mean HAC t across seeds')\n"
            "ax.set_title('Engine planted -> detected; engine removed -> nothing'); ax.legend()\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: with the diversification engine planted the harness banks it "
            f"(canonical: mean t = {R['syn'][0][4]:+.2f}, {R['syn'][0][6]:.0f}% of seeds ≥ 2); in the "
            f"permanent-2022 world it refuses to invent one (mean t = {R['syn'][1][4]:+.2f}, "
            f"{R['syn'][1][6]:.0f}%). The machinery is faithful in both directions. *(A machinery proof "
            "only — never cited in support of the Signal stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — full-period gap vs SPY **+{R['race_spy'][0]:.2f}%/yr** at **HAC t = "
            f"{R['race_spy'][1]:.2f}** (uncertifiable); split on the claim's own mechanism: "
            f"**+{R['reg_pre'][1]:.2f}%/yr at t = {R['reg_pre'][2]:.2f}** in the ρ<0 era, "
            f"**{R['reg_post'][1]:.2f}%/yr** since the flip, regime difference **Welch t = "
            f"{R['welch_regime']:.2f}**. Real on one regime, reversed on the other — never a full-tape "
            "certification.\n"
            f"- **Tradability `FRAGILE`** — liquid ETFs, ≤ {R['costs'][2][4]:.0f} bps/yr rebalance drag, "
            f"but **{R['perf'][0][3]:.1f}%** max drawdown, {R['dd']['now']:.0f}% below peak at the as-of, "
            f"Sharpe {R['perf'][0][4]:.2f} < SPY's {R['perf'][1][4]:.2f} < 60/40's {R['perf'][2][4]:.2f}, "
            "and the insurance leg depends on a correlation regime that broke. Not INVESTABLE; not quite "
            f"a Mirage (×{R['perf'][0][5]:.1f} vs ×{R['perf'][1][5]:.1f} raw wealth, disaster included).\n"
            f"- **2022 falsified it? `MIXED`** — the *law* (\"bonds insure stocks\") is falsified: corr "
            f"{R['corr']['pre']:+.2f} → +{R['corr']['y2022']:.2f} (still +{R['corr']['recent']:.2f} in "
            f"2024–26), both legs down {R['y2022'][0][1]:.0f}%/{R['y2022'][1][1]:.0f}% together, "
            "risk-adjusted case gone. The *arithmetic* survives on the full tape. HFEA compounds faster "
            "when and only when its regime holds — a regime bet, not a law."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The financing regime is the other hidden leg.** The 2010s gave ZIRP financing (the "
            "funds' borrow drag ≈ 2×bills + spread ≈ 2%/yr all-in at the calibrated fee); at 2023-era "
            "bill rates the same structure paid >10%/yr to hold. Any HFEA revival case must price that.\n"
            "- **Rebalancing cadence is not the knob.** Monthly (+14.7% CAGR) and annual (+17.0%) "
            "rebalancing bracket the quarterly headline (+17.5%) — the verdict never moves; quarterly "
            "wasn't cherry-picked.\n"
            "- **The regime question is the whole question.** [579 — equity-bond-corr-flip](../../579-equity-bond-corr-flip/README.md) "
            "tests the correlation regime directly; [61](../../61-slow-burn/README.md) / "
            "[100](../../100-melting-ice/README.md) cover the single-fund decay mechanics this study deliberately "
            "does not re-litigate.\n\n"
            "*The reproducible core is offline and deterministic; canonical numbers live in "
            "[`docs/results.md`](../docs/results.md) and are reprinted by "
            "[`examples/verify.py`](../examples/verify.py).*"
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
