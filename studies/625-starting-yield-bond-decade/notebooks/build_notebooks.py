"""Generate the two narrative notebooks for Study 625 (Starting-Yield-Bond-Decade).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the spliced
GS10 cache under ../_cache/ and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md). The synthetic control runs anywhere, no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (Shiller GS10 1871-01→2023-09
# spliced with ^TNX →2026-06; constant-maturity 10y nominal TR roll; as-of 2026-07-03).
R = dict(
    months=1866, start="1871-01", end="2026-06",
    shiller_months=1833, shiller_end="2023-09", tnx_months=33,
    n_overlap=1746, fwd_end="2016-06", fingerprint="36732503ca4f",
    # primary — non-overlapping decades
    n_dec=15, dec_first="1871-01", dec_last="2011-01",
    r2=0.923, slope=1.086, t_slope=12.50,
    intercept_pct=-0.34, t_intercept=-0.72, t_slope_vs1=0.99,
    mae_pp=0.64, resid_pp=0.86,
    # phase sweep (120 anchors)
    ph_r2_min=0.865, ph_r2_med=0.939, ph_r2_max=0.970,
    ph_t_min=8.75, ph_t_med=13.89, ph_n=120,
    # secondary — overlapping HAC
    hac_n=1746, hac_r2=0.916, hac_slope=1.140, hac_t=32.71, hac_lags=120,
    # sub-periods: (first, last, n, r2, slope, t)
    halves=[("1871-01", "1941-01", 8, 0.908, 1.20, 7.69),
            ("1943-10", "2013-10", 8, 0.954, 1.30, 11.17)],
    # 2020 cohort autopsy (PARTIAL decade)
    c_start="2020-12", c_y0=0.93, c_years=5.5, c_realized=-2.21,
    c_dd=-22.9, c_required=4.91, c_end="2026-06",
    # third axis — stocks (1/CAPE vs fwd 10y nominal TR)
    s_n=14, s_first="1881-01", s_last="2011-01",
    s_r2=0.205, s_slope=0.63, s_t=1.76, s_mae=4.57,
    s_hac_r2=0.251, s_hac_t=3.48, s_hac_n=1590,
    # honesty check — nominal yield vs REAL forward return
    rr_n=15, rr_r2=0.239, rr_slope=0.81, rr_t=2.02,
    # tradability arithmetic
    spread_bp=2.0, drag_bp_yr=0.20, cost_ratio=319,
    # synthetic control: (link, mean R2, mean t, share |t|>=2 %)
    syn=[(1.0, 0.900, 11.20, 100), (0.5, 0.729, 6.65, 95), (0.0, 0.102, 0.20, 10)],
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Investable](https://img.shields.io/badge/Tradability-Investable-2ea44f?style=flat-square)\n"
    "![Works_for_stocks%3F: Busted](https://img.shields.io/badge/Works_for_stocks%3F-Busted-8b949e?style=flat-square)\n\n"
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

from starting_yield_bond_decade import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PANEL = data.load_bond_panel()
    DEC = st.decade_table(PANEL, phase=0)
else:
    PANEL = DEC = None
print("real GS10 cache present:", HAVE_REAL,
      "| months:", (0 if PANEL is None else len(PANEL)),
      "| non-overlapping decades:", (0 if DEC is None else len(DEC)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The yield you buy at *is* your bond decade 📜\n"
            "### One number on the day you click \"buy\" tells you the next ten years — in plain English\n\n"
            + BADGES +
            "Here is the strangest-sounding true thing in investing: for a 10-year Treasury "
            "portfolio, **you don't have to forecast anything**. Look at the 10-year yield the day "
            "you buy — say 4% — and that, give or take a fraction of a percent per year, **is what "
            "you will earn per year for the next decade**, whatever the Fed, wars, elections or "
            "recessions do in between. The claim (Jack Bogle wrote it on a napkin in 1991) is that "
            "this one number explains about **90%** of the variation across bond decades.\n\n"
            "It sounds like fortune-telling. It's actually **arithmetic** — and we test it on 155 "
            "years of data, 1871 to 2026.\n\n"
            "> 📓 **Plain-language layer.** Want the regressions, the anchor sweep and the HAC "
            "t-stats? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A note up front.** \"Bond returns\" here = a constant-maturity 10-year US "
            "Treasury portfolio, **nominal** (before inflation), one country's tape. Every chart "
            "is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the starting yield predict the bond decade? | **Yes — almost perfectly.** "
            f"R² = **{R['r2']}** across 15 non-overlapping decades since 1871; the naive rule "
            f"*\"you'll earn the yield you bought at\"* misses by only **{R['mae_pp']} points a "
            f"year** on average. |\n"
            f"| Is it forecasting skill? | **No — it's duration arithmetic.** The best-fit line "
            f"has slope ≈ 1 and intercept ≈ 0: realised decade = starting yield. |\n"
            f"| Can you actually use it? | **Yes.** One Treasury purchase per decade, $0 "
            f"commission, unlimited capacity. But you're locking a **nominal** number — "
            f"inflation is *not* included. |\n"
            f"| Does the same trick work for stocks? | **No.** Same design on the stock "
            f"\"yield\" (1/CAPE): R² = {R['s_r2']} — the fix for stocks is "
            f"[study 120](../../120-excess-cape-yield/README.md). |\n"
        ),

        md(
            "## The money chart\n\n"
            "Each dot below is a **full decade**: the yield you could see on day one "
            "(horizontal) against what a rolling 10-year Treasury portfolio actually returned "
            "per year over the following ten years (vertical). Fifteen decades, no overlap, "
            "1871→2021. If the claim is true, the dots hug the dashed 45° line — *you get what "
            "you paid for*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    x, y = DEC['gs10']*100, DEC['fwd10']*100\n"
            "    fig, ax = plt.subplots(figsize=(7.5, 7))\n"
            "    lim = (0, max(x.max(), y.max())*1.12)\n"
            "    ax.plot(lim, lim, '--', color=GREY, lw=1.5, label='the promise: return = starting yield')\n"
            "    ax.scatter(x, y, s=70, color=GREEN, zorder=3, edgecolor='white', linewidth=1.2)\n"
            "    for d, xi, yi in zip(DEC.index, x, y):\n"
            "        if d.year in (1871, 1921, 1981, 2011):   # selective labels only\n"
            "            ax.annotate(str(d.year) + 's', (xi, yi), textcoords='offset points',\n"
            "                        xytext=(8, -4), fontsize=9, color='dimgray')\n"
            "    ax.set_xlim(lim); ax.set_ylim(lim)\n"
            "    ax.set_xlabel('10-year Treasury yield on day one (%)')\n"
            "    ax.set_ylabel('what the next decade actually paid (%/yr, nominal)')\n"
            "    ax.set_title('15 non-overlapping bond decades, 1871-2021: you get what you paid for')\n"
            "    ax.legend(loc='upper left', frameon=False)\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f\"R2 = {R['r2']}   |   average miss of the naive rule: {R['mae_pp']} pp/yr\")\n"
            "else:\n"
            "    print('cache missing - frozen headline: R2 =', R['r2'], ', MAE =', R['mae_pp'], 'pp/yr')"
        ),
        md(
            f"Fifteen dots, one ruler. R² = **{R['r2']}** — the starting yield explains ~92% of "
            "everything that happened to bond investors across a century and a half that included "
            "two world wars, the Great Depression, the 1970s inflation, and the zero-rate 2010s.\n\n"
            "> 🔬 **For the quants.** Slope t = "
            f"+{R['t_slope']:.1f}, slope statistically indistinguishable from 1, intercept from 0, "
            "and the result survives all 120 possible anchor months (worst R² "
            f"{R['ph_r2_min']}) — see [02_for_the_quants](02_for_the_quants.ipynb)."
        ),

        md(
            "## Why this is arithmetic, not magic\n\n"
            "A bond portfolio that keeps its maturity near 10 years lives on a **see-saw**:\n\n"
            "- If yields **fall** after you buy, your old bonds jump in price (nice!) — but every "
            "coupon and every maturing bond now **reinvests at worse rates**.\n"
            "- If yields **rise**, your bonds get marked down (ouch) — but everything reinvests "
            "at **better rates**.\n\n"
            "Over ~10 years the two effects **cancel almost exactly**, and what's left is… the "
            "yield you started with. Bond mathematicians call it *duration targeting convergence* "
            "(Leibowitz): a duration-D portfolio's return converges to its starting yield after "
            "about 2×D−1 years. The market can delay the payment; it can't change the bill.\n\n"
            "That's why this is the rare \"prediction\" that needs no crystal ball — the number "
            "is printed on the ticket when you board."
        ),

        md(
            "## The 2020 buyer — a prediction of misery, delivered on schedule\n\n"
            f"December 2020: the 10-year yield averaged **{R['c_y0']}%** — the worst starting "
            "hand in 150 years of data. The rule's forecast was brutal and public: *your decade "
            f"will pay about {R['c_y0']}% a year, nominal, before inflation.* Then 2022 happened — "
            "the worst bond crash in modern history. Did it break the rule?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    seg = PANEL['tr'].loc['2020-12-01':]\n"
            "    seg = seg / seg.iloc[0]\n"
            "    yrs = np.arange(len(seg)) / 12\n"
            "    promise = (1 + R['c_y0']/100) ** np.arange(0, 10.01, 1/12)\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(np.arange(len(promise))/12, promise, '--', color=GREY, lw=1.5,\n"
            "            label=f\"the promise: {R['c_y0']}%/yr for 10 years\")\n"
            "    ax.plot(yrs, seg.values, color=RED, lw=2, label='what the 2020 cohort actually has')\n"
            "    ax.annotate('the 2022-23 bond bear\\n(max drawdown -22.9%)',\n"
            "                xy=(2.8, seg.min()), xytext=(4.2, 0.82), fontsize=9, color='dimgray',\n"
            "                arrowprops=dict(arrowstyle='->', color='dimgray'))\n"
            "    ax.set_xlabel('years since December 2020')\n"
            "    ax.set_ylabel('growth of $1 (nominal total return)')\n"
            "    ax.set_title('The 0.93% cohort: a terrible decade was promised - a terrible decade is arriving')\n"
            "    ax.legend(frameon=False); plt.tight_layout(); plt.show()\n"
            "    print(f\"realised so far ({R['c_years']}y): {R['c_realized']}%/yr | \"\n"
            "          f\"needs +{R['c_required']}%/yr over the remaining years to land exactly on the promise\")\n"
            "else:\n"
            "    print('cache missing - frozen: realised', R['c_realized'], '%/yr, max DD', R['c_dd'], '%')"
        ),
        md(
            "Here's the subtle part: the crash is **not a failure of the rule — it's the "
            "mechanism**. The cohort is 5½ years in at **−2.2%/yr**, and to finish exactly on the "
            f"promised 0.93%/yr it needs about **+{R['c_required']}%/yr** for the remaining years "
            "— which is roughly what yields *are* after the crash. The see-saw is pulling the "
            "cohort back to its ticket price. The 2020 lesson isn't \"bonds are unpredictable\"; "
            "it's the opposite: **the misery was printed on the ticket**, and people bought anyway."
        ),

        md(
            "## Does the same trick work for stocks? (No.)\n\n"
            "Stocks have a \"starting yield\" too — the earnings yield (1/CAPE). Run the exact "
            "same experiment: starting earnings yield vs the next decade of stock returns."
        ),
        code(
            "if HAVE_REAL:\n"
            "    stk = data.load_stock_decades()\n"
            "    sdec = st.decade_table(stk, xcol='ey', ycol='fwd10', phase=0)\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)\n"
            "    for ax, tab, xc, ttl, col, r2 in [\n"
            "        (axes[0], DEC, 'gs10', f\"BONDS - R2 = {R['r2']}\", GREEN, R['r2']),\n"
            "        (axes[1], sdec, 'ey', f\"STOCKS - R2 = {R['s_r2']}\", AMBER, R['s_r2'])]:\n"
            "        x, y = tab[xc]*100, tab['fwd10']*100\n"
            "        b, a = np.polyfit(x, y, 1)\n"
            "        xs = np.linspace(x.min(), x.max(), 50)\n"
            "        ax.scatter(x, y, s=60, color=col, edgecolor='white', linewidth=1.2, zorder=3)\n"
            "        ax.plot(xs, a + b*xs, '-', color=GREY, lw=1.5)\n"
            "        ax.set_title(ttl); ax.set_xlabel('starting yield (%)')\n"
            "    axes[0].set_ylabel('next decade (%/yr, nominal)')\n"
            "    fig.suptitle('Same experiment, two assets: arithmetic vs mood', y=1.02)\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f\"average miss of 'return = starting yield': bonds {R['mae_pp']} pp/yr vs stocks {R['s_mae']} pp/yr\")\n"
            "else:\n"
            "    print('cache missing - frozen: bonds R2', R['r2'], 'vs stocks R2', R['s_r2'])"
        ),
        md(
            f"For stocks the dots scatter everywhere: R² = **{R['s_r2']}**, and the naive rule "
            f"misses by **{R['s_mae']} points a year** — seven times the bond error. A bond is a "
            "*contract* (fixed coupons, fixed principal — the yield mechanically becomes your "
            "return); a stock is a *claim on moods and profits*. The repaired stock version — "
            "subtract the bond yield from the earnings yield first — is its own study on this "
            "bench: **[120 — Excess-CAPE-Yield](../../120-excess-cape-yield/README.md)** "
            "(R² 0.70, a statistical pattern, *not* an identity).\n\n"
            "**Busted** on the stock side — the arithmetic belongs to bonds."
        ),

        md(
            "## The fine print (read this before you feel too safe)\n\n"
            f"1. **It's a *nominal* lock.** Rerun everything against inflation-adjusted returns "
            f"and R² collapses from {R['r2']} to **{R['rr_r2']}**. The 1940s and 1970s cohorts "
            "collected every promised coupon and still lost purchasing power. The ticket price "
            "is in dollars, not in groceries.\n"
            "2. **It's beta, not alpha.** Knowing your decade in advance is priceless for "
            "*planning* (retirement ladders, liability matching) — but it beats nobody; everyone "
            "can read the same yield.\n"
            "3. **One country's tape.** The US never defaulted in-sample. A 1913 buyer of "
            "Russian bonds also \"knew his decade.\"\n\n"
            "## Verdict\n\n"
            "| Axis | Stamp |\n|---|---|\n"
            f"| Does the starting yield pin the bond decade? | **REAL** — R² {R['r2']}, slope ≈ 1, "
            "155 years, robust everywhere we poked |\n"
            "| Can you deploy it? | **INVESTABLE** — one Treasury purchase, ~0 cost, unlimited "
            "capacity; but it locks *nominal* dollars only |\n"
            f"| Works for stocks too? | **BUSTED** — R² {R['s_r2']}; see "
            "[study 120](../../120-excess-cape-yield/README.md) |\n\n"
            "*Full stats, anchor sweep, HAC regression and the synthetic control: "
            "[02_for_the_quants.ipynb](02_for_the_quants.ipynb). Reproducible run: "
            "[docs/results.md](../docs/results.md). Not investment advice.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    return nb


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Starting-Yield-Bond-Decade — the quant teardown 📜\n\n"
            + BADGES +
            "**Claim (Bogle 1991; Bogle & Nolan 2015; Leibowitz duration-targeting):** the entry "
            "10-year yield explains ~90% of the variance of the subsequent decade's return of a "
            "constant-maturity Treasury portfolio — an arithmetic identity, not a forecast.\n\n"
            "**Design.** Shiller monthly GS10 1871-01→2023-09 spliced with ^TNX monthly means "
            "→2026-06; constant-maturity 10y **nominal** TR roll (Swinkels-2019-style closed-form "
            "par-bond pricing); signal = month-*t* average yield, entry = end of month *t* (one "
            "documented lag). **Primary unit = the non-overlapping decade** (Valkanov 2003 is why); "
            "overlapping windows are secondary with NW(120) HAC. As-of 2026-07-03, last complete "
            f"month 2026-06, fingerprint `{R['fingerprint']}`."
        ),
        code(BOOT_CELL),

        md("## 0 · Tape and engine"),
        code(
            "if HAVE_REAL:\n"
            "    g = data.load_gs10()\n"
            "    n_sh = int((g['source']=='shiller').sum()); n_tnx = int((g['source']=='tnx').sum())\n"
            "    print(f'{len(g)} months {g.index.min():%Y-%m} -> {g.index.max():%Y-%m} '\n"
            "          f'(shiller {n_sh}, tnx {n_tnx}) | fingerprint {data.fingerprint(g)}')\n"
            "    usable = PANEL.dropna(subset=['fwd10'])\n"
            "    print(f'forward-decade windows: {len(usable)} ({usable.index.min():%Y-%m} -> {usable.index.max():%Y-%m})')\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(g.index, g['gs10']*100, color=GREY, lw=1.2)\n"
            "    tnx_part = g[g['source']=='tnx']\n"
            "    ax.plot(tnx_part.index, tnx_part['gs10']*100, color=RED, lw=1.6, label='^TNX splice (2023-10+)')\n"
            "    ax.set_ylabel('GS10 (%)'); ax.set_title('The yield tape, 1871-2026')\n"
            "    ax.legend(frameon=False); plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('cache missing - frozen tape:', R['months'], 'months', R['start'], '->', R['end'])"
        ),
        md(
            "> 💡 **In plain words.** The return series is *simulated from yields* — each month "
            "we \"buy\" a fresh 10-year par bond and \"sell\" it a month later at the new yield. "
            "Swinkels (2019) validates this construction at ≈0.99 correlation against actual "
            "Treasury index returns. The whole claim is that this simulation's decade average is "
            "pinned by its first input."
        ),

        md("## 1 · Primary — non-overlapping decades (unit = decade)"),
        code(
            "if HAVE_REAL:\n"
            "    d = st.decade_ols(PANEL, phase=0)\n"
            "    print(f\"n = {d['n']} decades ({d['starts'][0]} ... {d['starts'][-1]})\")\n"
            "    print(f\"R2 = {d['r2']:.3f}  slope = {d['slope']:+.3f} (t = {d['t_slope']:+.2f})  \"\n"
            "          f\"intercept = {d['intercept']*100:+.2f}%/yr (t = {d['t_intercept']:+.2f})\")\n"
            "    print(f\"identity: t(slope=1) = {d['t_slope_vs1']:+.2f} | MAE of 'return = yield' = \"\n"
            "          f\"{d['mae_identity']*100:.2f} pp/yr | resid std {d['resid_std']*100:.2f} pp/yr\")\n"
            "    x, y = DEC['gs10']*100, DEC['fwd10']*100\n"
            "    fig, ax = plt.subplots(figsize=(7, 6.5))\n"
            "    lim = (0, max(x.max(), y.max())*1.1)\n"
            "    ax.plot(lim, lim, '--', color=GREY, lw=1.5, label='identity: slope 1, intercept 0')\n"
            "    xs = np.linspace(*lim, 50)\n"
            "    ax.plot(xs, d['intercept']*100 + d['slope']*xs, '-', color=RED, lw=1.5,\n"
            "            label=f\"OLS fit (slope {d['slope']:+.2f})\")\n"
            "    ax.scatter(x, y, s=60, color=GREEN, edgecolor='white', linewidth=1.2, zorder=3)\n"
            "    ax.set_xlim(lim); ax.set_ylim(lim)\n"
            "    ax.set_xlabel('starting GS10 (%)'); ax.set_ylabel('fwd 10y annualised nominal TR (%/yr)')\n"
            "    ax.set_title(f\"Non-overlapping decades: R2 = {d['r2']:.3f}\")\n"
            "    ax.legend(frameon=False, loc='upper left'); plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('frozen: R2', R['r2'], 'slope t', R['t_slope'])"
        ),
        md(
            f"R² = **{R['r2']}** with slope t = **+{R['t_slope']:.1f}** on 15 *independent* "
            "points — no overlap, no HAC needed, the desk's t ≥ 2 bar cleared six times over. "
            f"The identity test cannot reject slope = 1 (t = +{R['t_slope_vs1']:.2f}) or "
            "intercept = 0 — the fitted line *is* \"you get what you paid for.\"\n\n"
            "> 💡 **In plain words.** Fifteen completely separate decades, one dot each. A ruler "
            "through them passes through the origin at 45°. That's not a strategy — it's a "
            "receipt."
        ),

        md("## 2 · The anchor is not doing the work — all 120 phases"),
        code(
            "if HAVE_REAL:\n"
            "    ps = st.phase_sweep(PANEL)\n"
            "    print(f\"R2 min/med/max = {ps['r2_min']:.3f} / {ps['r2_med']:.3f} / {ps['r2_max']:.3f}\"\n"
            "          f\"  | slope-t min/med = {ps['t_min']:+.2f} / {ps['t_med']:+.2f}\"\n"
            "          f\"  ({ps['n_phases']} phases, n = {ps['n_min']}-{ps['n_max']})\")\n"
            "    r2s = [st.decade_ols(PANEL, phase=ph)['r2'] for ph in range(120)]\n"
            "    fig, ax = plt.subplots(figsize=(9.5, 3.8))\n"
            "    ax.bar(range(120), r2s, color=GREEN, width=1.0, edgecolor='white', linewidth=0.3)\n"
            "    ax.axhline(R['r2'], color=GREY, ls='--', lw=1.2)\n"
            "    ax.annotate('headline anchor (phase 0)', xy=(0, R['r2']), xytext=(15, 0.55),\n"
            "                fontsize=9, color='dimgray', arrowprops=dict(arrowstyle='->', color='dimgray'))\n"
            "    ax.set_xlabel('anchor month (phase of the non-overlapping grid)')\n"
            "    ax.set_ylabel('decade R2'); ax.set_ylim(0, 1)\n"
            "    ax.set_title('Decade R2 across all 120 possible anchors')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('frozen: R2 min/med/max', R['ph_r2_min'], R['ph_r2_med'], R['ph_r2_max'])"
        ),
        md(
            f"The *worst* of the 120 anchors gives R² = {R['ph_r2_min']} and slope t = "
            f"+{R['ph_t_min']:.1f}. No cherry-picked phase."
        ),

        md("## 3 · Secondary — overlapping windows with NW(120) HAC"),
        code(
            "if HAVE_REAL:\n"
            "    h = st.overlapping_hac(PANEL, lags=120)\n"
            "    print(f\"n = {h['n']:,} overlapping windows | R2 = {h['r2']:.3f} | \"\n"
            "          f\"slope = {h['slope']:+.3f} | HAC t = {h['hac_t_slope']:+.2f} (lags={h['lags']})\")\n"
            "else:\n"
            "    print('frozen: HAC t', R['hac_t'], 'R2', R['hac_r2'])"
        ),
        md(
            "> 💡 **In plain words.** Using every month as a decade-start creates windows that "
            "share up to 119 of their 120 months — classic pseudo-replication (Valkanov 2003). "
            "The Newey-West correction with 120 lags eats that; the slope still stands at "
            f"t = **+{R['hac_t']:.1f}**. It's secondary; the honest unit is the decade above."
        ),

        md("## 4 · Sub-periods, the 2020 cohort, and the nominal-vs-real collapse"),
        code(
            "if HAVE_REAL:\n"
            "    usable = PANEL.dropna(subset=['fwd10'])\n"
            "    half = usable.index[len(usable)//2]\n"
            "    for name, sub in [('first half ', PANEL.loc[:half]), ('second half', PANEL.loc[half:])]:\n"
            "        s = st.decade_ols(sub, phase=0)\n"
            "        print(f\"{name}: {s['starts'][0]}..{s['starts'][-1]}  n={s['n']}  \"\n"
            "              f\"R2={s['r2']:.3f}  slope={s['slope']:+.2f} (t={s['t_slope']:+.2f})\")\n"
            "    c = st.cohort_autopsy(PANEL, start='2020-12-01')\n"
            "    print(f\"\\n2020 cohort (PARTIAL decade): start {c['start']} at {c['start_yield']*100:.2f}% | \"\n"
            "          f\"{c['years_elapsed']:.1f}y realised {c['realized_ann']*100:+.2f}%/yr | \"\n"
            "          f\"max DD {c['max_drawdown']*100:.1f}% | needs {c['required_rest_ann']*100:+.2f}%/yr to land\")\n"
            "    rp = data.load_bond_panel_real()\n"
            "    dr = st.decade_ols(rp, xcol='gs10', ycol='fwd10_real', phase=0)\n"
            "    print(f\"\\nnominal yield vs REAL fwd decade: n={dr['n']}  R2={dr['r2']:.3f}  \"\n"
            "          f\"slope={dr['slope']:+.2f} (t={dr['t_slope']:+.2f})  <- the lock is nominal only\")\n"
            "else:\n"
            "    print('frozen: halves', R['halves'], '| cohort', R['c_realized'], '| real R2', R['rr_r2'])"
        ),
        md(
            f"Both halves hold on their own (R² {R['halves'][0][3]} and {R['halves'][1][3]}) — "
            "gold standard and fiat alike. The 2020 cohort is the identity *in progress*: "
            f"{R['c_realized']}%/yr realised after the 2022 crash, with the remaining-years "
            f"requirement (+{R['c_required']}%/yr) sitting right where post-crash yields are — "
            "mean reversion toward the ticket price, exactly as duration targeting predicts.\n\n"
            f"**The honesty check:** against CPI-deflated returns R² collapses to "
            f"**{R['rr_r2']}** (slope t = {R['rr_t']:+.2f}). The starting yield pins dollars, "
            "not purchasing power — named on the Tradability axis."
        ),

        md("## 5 · Third axis — the same design on stocks"),
        code(
            "if HAVE_REAL:\n"
            "    stk = data.load_stock_decades()\n"
            "    sd = st.decade_ols(stk, xcol='ey', ycol='fwd10', phase=0)\n"
            "    sh = st.overlapping_hac(stk, xcol='ey', ycol='fwd10', lags=120)\n"
            "    print(f\"stocks (1/CAPE vs fwd 10y nominal TR): n={sd['n']} decades \"\n"
            "          f\"({sd['starts'][0]}..{sd['starts'][-1]})\")\n"
            "    print(f\"  R2 = {sd['r2']:.3f}  slope = {sd['slope']:+.2f} (t = {sd['t_slope']:+.2f})  \"\n"
            "          f\"MAE = {sd['mae_identity']*100:.2f} pp/yr\")\n"
            "    print(f\"  overlapping: R2 = {sh['r2']:.3f}  HAC t = {sh['hac_t_slope']:+.2f} (n={sh['n']:,})\")\n"
            "    print(f\"  vs bonds: R2 {R['r2']} and MAE {R['mae_pp']} pp/yr on the identical design\")\n"
            "else:\n"
            "    print('frozen: stocks R2', R['s_r2'], 't', R['s_t'])"
        ),
        md(
            f"Decade-unit t = **{R['s_t']}** — *below* the bar that bonds clear at "
            f"+{R['t_slope']:.1f}. The overlapping HAC t (+{R['s_hac_t']:.2f}) says there is "
            "*some* long-horizon signal in valuations (that is exactly "
            "[study 120](../../120-excess-cape-yield/README.md)'s territory, where the "
            "bond-adjusted ECY reaches R² 0.70 on excess returns) — but nothing like an identity. "
            "**Busted** as a transfer of *this* claim."
        ),

        md("## 6 · Tradability arithmetic"),
        code(
            "spread_bp = R['spread_bp']\n"
            "drag = spread_bp / 10\n"
            "print(f'one entry per decade: TreasuryDirect $0 commission; ~{spread_bp:.0f} bp one-way spread')\n"
            "print(f'amortised drag = {drag:.2f} bp/yr vs forecast MAE {R[\"mae_pp\"]} pp/yr '\n"
            "      f'(~{R[\"cost_ratio\"]}x larger) | capacity > $25T | long-only, no borrow')"
        ),
        md(
            "> 💡 **In plain words.** There is nothing to \"trade\": you buy one bond (or a "
            "constant-maturity fund) and the arithmetic does the rest. Costs are three orders of "
            "magnitude below the forecast error. What you *cannot* buy with it is protection "
            "from inflation, or any edge over the next person — the yield is public."
        ),

        md(
            "## 7 · Synthetic control — machinery proof (never market evidence)\n\n"
            "A seeded AR(1) yield path drives the **same** pricing engine; `link` blends "
            "mechanical returns with yield-independent noise. Planted world (`link=1`) must light "
            "up; the null (`link=0`) must not. **Nulls are seed-averaged (20 seeds)** — "
            "single-seed nulls are banned on this desk."
        ),
        code(
            "for link in (1.0, 0.5, 0.0):\n"
            "    r2s, ts = [], []\n"
            "    for seed in range(625, 645):\n"
            "        frame, truth = data.synthetic_world(link=link, seed=seed)\n"
            "        s = st.decade_ols(frame, phase=0)\n"
            "        r2s.append(s['r2']); ts.append(s['t_slope'])\n"
            "    r2s, ts = np.array(r2s), np.array(ts)\n"
            "    print(f'link={link:.1f}: mean R2 = {r2s.mean():.3f}  mean t = {ts.mean():+.2f}  '\n"
            "          f'share |t|>=2 = {(np.abs(ts)>=2).mean()*100:.0f}%  (20 seeds, n=14 decades)')"
        ),
        md(
            "The engine banks the planted identity (mean R² 0.90 at `link=1`) and rejects at "
            "≈ the nominal 10% rate in the null — the real-tape R² 0.92 is measurement, not "
            "construction. *(A machinery proof only.)*"
        ),

        md(
            "## Verdict\n\n"
            f"- **Signal — REAL.** R² = {R['r2']}, slope t = +{R['t_slope']:.1f} on 15 "
            f"non-overlapping decades; worst-phase R² {R['ph_r2_min']} / t +{R['ph_t_min']:.1f}; "
            f"HAC t +{R['hac_t']:.1f} on overlapping windows; slope ≈ 1, intercept ≈ 0; both "
            "century-halves hold. Caveat: one country's never-defaulted tape.\n"
            f"- **Tradability — INVESTABLE.** {R['drag_bp_yr']:.1f} bp/yr amortised cost, >$25T "
            "capacity, $0 retail access. You deploy a *known nominal decade* — beta, not alpha; "
            f"the real-return R² is only {R['rr_r2']}.\n"
            f"- **Works for stocks? — BUSTED.** R² {R['s_r2']}, decade t {R['s_t']} < 2, MAE "
            f"{R['s_mae']} pp/yr. The stock-side repair is "
            "[study 120's ECY](../../120-excess-cape-yield/README.md).\n\n"
            "*Reproducible run: [docs/results.md](../docs/results.md) · literature: "
            "[docs/references.md](../docs/references.md) · not investment advice.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    return nb


if __name__ == "__main__":
    for name, builder in [("01_for_the_curious.ipynb", build_curious),
                          ("02_for_the_quants.ipynb", build_quants)]:
        nb = builder()
        path = os.path.join(HERE, name)
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("wrote", path)
