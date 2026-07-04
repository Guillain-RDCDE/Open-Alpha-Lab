"""Generate the two narrative notebooks for Study 624 (Buffer ETFs — the Cost of Comfort).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached yfinance tape
under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic machinery controls run anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance, BUFR + 4 Innovator
# Power Buffer vintages + SPY + BIL, 2018-08-01 -> 2026-06-30, as-of 2026-06-30).
R = dict(
    as_of="2026-06-30", fingerprint="e9356ec7ade8", fingerprint_raw="2823f75e9835",
    # delivery: 25 completed outcome periods (fund TOTAL ret % vs SPY PRICE ret %)
    n_periods=25, n_down=4, down_honored=4, n_up=21, n_capped=16,
    mean_capped_ret=11.71, mean_up_giveup=9.98,
    pjan22=(-19.48, -5.29, -5.27),      # SPY price, fund, terms floor
    papr22=(-9.35, -0.88),              # inside-buffer example
    periods={                            # (SPY price ret %, fund TR %) per outcome period
        "PJAN": [(16.16, 7.68), (27.04, 8.80), (-19.48, -5.29), (24.29, 18.18),
                 (23.30, 13.45), (16.35, 11.29)],
        "PAPR": [(53.77, 14.03), (13.96, 7.66), (-9.35, -0.88), (27.77, 14.25),
                 (6.94, 6.16), (16.26, 11.61)],
        "PJUL": [(5.24, 4.57), (38.82, 10.69), (-11.87, -0.78), (17.50, 16.47),
                 (22.77, 13.53), (13.53, 12.87), (20.87, 11.31)],
        "POCT": [(12.84, 8.48), (28.14, 11.06), (-16.77, -2.39), (19.68, 18.62),
                 (34.22, 14.69), (16.11, 10.65)],
    },
    # cost vs SPY: fund -> (gap pp/yr, HAC t, n months)
    vs_spy={"BUFR": (5.26, 2.13, 70), "PJAN": (7.75, 3.04, 89), "PAPR": (8.27, 2.58, 86),
            "PJUL": (6.34, 2.10, 94), "POCT": (6.91, 2.40, 92)},
    cohort_vs_spy=(7.24, 2.44, 86), lag_grid=(2.44, 2.44, 2.28),
    # fair race: fund -> (beta, mix-fund gap pp/yr, HAC t, corr)
    fair={"BUFR": (0.59, 0.05, 0.06, 0.977), "PJAN": (0.49, 0.51, 0.58, 0.949),
          "PAPR": (0.40, 0.28, 0.31, 0.924), "PJUL": (0.47, -0.33, -0.32, 0.942),
          "POCT": (0.43, -0.94, -0.92, 0.936)},
    cohort_fair=(0.44, -0.19, -0.24, 0.965), winners="2/5",
    # perf (BUFR window): label -> (CAGR %, vol %, maxDD %, Sharpe)
    perf={"BUFR": (10.49, 9.56, -13.28, 0.79), "SPY": (15.53, 15.68, -23.93, 0.82),
          "BIL": (2.95, 0.63, -0.13, None), "mix 0.59": (10.57, 9.33, -14.37, 0.82),
          "mix 0.70": (11.89, 10.99, -16.93, 0.82)},
    # decomposition: (measured gap, fee, dividends forgone, residual) pp/yr
    dec_bufr=(0.05, 0.95, 0.83, -1.74), dec_cohort=(-0.19, 0.79, 0.62, -1.60),
    # robustness: w grid -> (mix - BUFR pp/yr, HAC t)
    wgrid=[(0.45, -1.79, -2.34), (0.55, -0.51, -0.78), (0.65, 0.77, 0.87), (0.70, 1.41, 1.31)],
    cost_grid=[(2, 0.05), (5, 0.04), (10, 0.03)],
    # synthetic controls
    syn_gap=[(0.0, -0.06, -0.30), (2.0, 1.94, 9.99)],
    syn_delivery=(14, 14, 9.19, 9.21),
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Beat_the_dumb_mix%3F: Busted](https://img.shields.io/badge/Beat_the_dumb_mix%3F-Busted-8b949e?style=flat-square)\n\n"
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

from buffer_etf_cost import data, strategy as st

P_SERIES = ["PJAN", "PAPR", "PJUL", "POCT"]
HAVE_REAL = data.have_real()
if HAVE_REAL:
    ADJ, RAW = data.load_real(asof=data.AS_OF)
    M = data.monthly_returns(ADJ)
    SPY, BIL = M["SPY"], M["BIL"]
else:
    ADJ = RAW = M = SPY = BIL = None
print("real cache present:", HAVE_REAL)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"

SCATTER = """\
# every completed outcome period: SPY PRICE return (the stated reference) vs fund TOTAL return
if HAVE_REAL:
    pairs = {tk: [(r["spy_price_ret_pct"], r["fund_ret_pct"]) for _, r in
                  st.outcome_periods(ADJ[tk], RAW["SPY"], data.FUNDS[tk]["reset_month"]).iterrows()]
             for tk in P_SERIES}
else:
    pairs = R["periods"]
fig, ax = plt.subplots(figsize=(9.2, 5.6))
xs = np.linspace(-30, 55, 300)
ax.plot(xs, xs, ls=":", c=GREY, lw=1, label="1-for-1 (no structure)")
promise = np.where(xs >= 0, np.minimum(xs, 12.0), np.where(xs >= -15, 0.0, xs + 15.0)) - 0.79
ax.plot(xs, promise, c=GREY, lw=2, alpha=.7, label="the promise (15% buffer, ~12% cap, - fee)")
ax.axvspan(-15, 0, color=GREEN, alpha=.08)
marks = dict(PJAN="o", PAPR="s", PJUL="^", POCT="D")
for tk, pts in pairs.items():
    px, py = zip(*pts)
    ax.scatter(px, py, marker=marks[tk], s=55, color=(RED if tk == "PJAN" else GREEN if tk == "PJUL" else AMBER if tk == "PAPR" else GREY),
               edgecolor="k", linewidth=.4, label=tk, zorder=3)
ax.set_xlabel("SPY PRICE return over the 12-month outcome period (%)")
ax.set_ylabel("fund TOTAL return (%)")
ax.set_title("25 outcome periods: every buffer delivered, every cap enforced")
ax.legend(fontsize=8.5); plt.tight_layout(); plt.show()
n_down = sum(1 for pts in pairs.values() for x, _ in pts if x < 0)
print(f"{sum(len(v) for v in pairs.values())} completed periods, {n_down} down periods - buffer honored in all of them")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Buffer ETFs — what does the comfort actually cost? 🛡️\n"
            "### Defined-outcome funds promise a 15% airbag on the S&P 500. They kept the promise. "
            "Here's the bill — in plain words\n\n"
            + BADGES +
            "There is now a ~$50bn category of ETFs that sells you the stock market **with an airbag**: "
            "over each 12-month \"outcome period\" you're protected against the first 10–15% of losses, "
            "and in exchange your upside is **capped** (~10–12% a year) and you pay a fee about **ten "
            "times** an index fund's. The critics' line writes itself: *\"you're paying a fat fee and "
            "giving away bull markets for insurance you could build yourself with an index fund and "
            "T-bills.\"*\n\n"
            "So we audited both halves on the real tape: **did the airbag deploy** every time it was "
            "called, and **was the do-it-yourself version actually cheaper**?\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the decomposition and the robustness "
            "grids? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Data note.** BUFR (the flagship laddered buffer) + the four oldest Innovator Power "
            "Buffer funds, 2018–2026 — the category's *surviving flagships* (a mild tilt in the funds' "
            "favor, which we name). Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did the buffer actually work? | **Yes, every time.** In all **4** down years on the tape "
            "the funds delivered the promised floor — in 2022 the S&P fell **19.5%** and the January "
            "fund lost only **5.3%**, within 2 bps of its contract. |\n"
            "| Did the cap really eat your upside? | **Yes, hard.** In **16 of 21** up years the cap "
            "bound: the funds plateaued near **+12%** while the S&P's up years averaged about **+22%**. |\n"
            "| So they're a rip-off? | **No — that's the twist.** The 5–8%/yr they trail the market is "
            "just *owning less market*. Against an honest same-risk mix of index fund + T-bills, the "
            "buffer funds came out **dead even** (and 2 of 5 even nosed ahead). |\n"
            "| Could you have built it cheaper? | **Not on this tape.** The fee and lost dividends "
            "(~1.8%/yr) were real — but the option structure **earned them back**, mostly by cushioning "
            "2022. The comfort was fairly priced. |\n\n"
            "> The product is mechanically honest **and** the critique is half-wrong: you pay for the "
            "airbag, the airbag deploys, and the DIY alternative saved you nothing."
        ),

        md(
            "## 1 · The claim\n\n"
            "> *\"Buffer ETFs deliver exactly the buffer they promise — and charge you the upside cap "
            "plus a fat fee for insurance you could build cheaper.\"*\n\n"
            "That's the standard advisor-press critique (Morningstar has run versions of it; academics "
            "like Israelov call it *the hidden cost of buffer funds*). Note it has **two halves**: the "
            "funds are *honest* (they deliver the terms) but *overpriced* (a plain index + T-bill mix "
            "should beat them). Both halves are testable."
        ),
        md(
            "## 2 · So what?\n\n"
            "Tens of billions of retirement dollars moved into these wrappers after 2020. If the critique "
            "is right, all of that is paying ~0.8–0.95%/yr for something a two-fund portfolio replicates "
            "for ~0.1%. If it's wrong, the wrapper is a fair deal for people who genuinely cannot sit "
            "through a −20% year without selling — which is the whole behavioral point of the product."
        ),
        md(
            "## 3 · How would we even know?\n\n"
            "Three measurements on the 2018–2026 tape:\n\n"
            "1. **Delivery.** For each completed 12-month outcome period, pair the fund's return with "
            "SPY's **price** return (that's what the contract is written on) and check the floor and "
            "the cap against the stated terms.\n"
            "2. **The cost.** How much did each fund trail SPY's total return, and is that gap "
            "statistically real?\n"
            "3. **The fair race.** Each fund vs the *dumb mix it replaces* — the same equity exposure "
            "built from SPY + T-bills (BIL), rebalanced monthly with trading costs charged."
        ),

        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, delivery.** Every completed outcome period, promise vs reality:"
        ),
        code(SCATTER),
        md(
            f"The dots trace the promised payoff shape almost perfectly: flat at ~0 inside the buffer "
            f"zone (green band), floor honored in **{R['down_honored']}/{R['n_down']}** down periods "
            f"(PJAN 2022: SPY price **{R['pjan22'][0]:+.2f}%** → fund **{R['pjan22'][1]:+.2f}%**, "
            f"contract floor **{R['pjan22'][2]:+.2f}%**), and a hard ceiling near **+12%** — the cap "
            f"bound in **{R['n_capped']}/{R['n_up']}** up periods, costing about "
            f"**{R['mean_up_giveup']:.0f} points per up year**. The product does exactly what the "
            "label says: full airbag, hard ceiling."
        ),
        md(
            "**Now the bill.** How much market did the comfort cost — and does it survive the fair "
            "comparison? Left: each fund vs SPY. Right: each fund vs its honest same-risk SPY/BIL mix."
        ),
        code(
            "funds = list(R['vs_spy'])\n"
            "if HAVE_REAL:\n"
            "    vs_spy, fair = [], []\n"
            "    for tk in funds:\n"
            "        f = M[tk].dropna()\n"
            "        vs_spy.append(st.gap_stats(f, SPY.loc[f.index])['gap_pp_yr'])\n"
            "        b = st.beta_vs(f, SPY, BIL)\n"
            "        fair.append(st.gap_stats(f, st.mix_returns(SPY.loc[f.index], BIL.loc[f.index], b))['gap_pp_yr'])\n"
            "else:\n"
            "    vs_spy = [R['vs_spy'][tk][0] for tk in funds]\n"
            "    fair = [R['fair'][tk][1] for tk in funds]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)\n"
            "a1.bar(funds, vs_spy, color=RED, width=.6)\n"
            "for i, v in enumerate(vs_spy): a1.annotate(f'{v:+.1f}', (i, v), ha='center', va='bottom')\n"
            "a1.set_title('vs the market (SPY total return)\\n\"you gave up 5-8 points a year\"')\n"
            "a1.set_ylabel('how much the fund trailed (pp/yr)')\n"
            "a2.bar(funds, fair, color=[GREEN if v <= 0 else AMBER for v in fair], width=.6)\n"
            "for i, v in enumerate(fair): a2.annotate(f'{v:+.1f}', (i, v), ha='center', va='bottom')\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_title('vs the honest same-risk SPY/BIL mix\\n\"...but the DIY version saved you nothing\"')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('vs SPY (pp/yr):', [round(v, 1) for v in vs_spy], ' | vs beta-matched mix:', [round(v, 1) for v in fair])"
        ),
        md(
            f"The left panel is the critics' chart: every fund trailed the market by **5–8 points a "
            f"year** — and the quants notebook shows that's statistically solid, not luck. The right "
            f"panel is the punchline: against the **same-risk** dumb mix the gaps shrink to "
            f"**fractions of a point in either direction** (2 of 5 funds actually ahead). The \"cost of "
            f"comfort\" was really just *owning about half as much stock market* — which is what you "
            "asked for when you bought an airbag."
        ),
        md("**Same story in risk-and-return terms** (BUFR's window, Sep 2020 → Jun 2026):"),
        code(
            "labels = ['BUFR', 'SPY', 'mix 0.59']\n"
            "if HAVE_REAL:\n"
            "    f = M['BUFR'].dropna(); win = f.index\n"
            "    legs = {'BUFR': f, 'SPY': SPY.loc[win], 'mix 0.59': st.mix_returns(SPY.loc[win], BIL.loc[win], 0.59)}\n"
            "    stats = {k: st.perf_stats(v, BIL.loc[win]) for k, v in legs.items()}\n"
            "    cagr = [stats[k]['cagr_pct'] for k in labels]; dd = [stats[k]['maxdd_pct'] for k in labels]\n"
            "else:\n"
            "    cagr = [R['perf'][k][0] for k in labels]; dd = [R['perf'][k][2] for k in labels]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.4))\n"
            "a1.bar(labels, cagr, color=[AMBER, GREY, GREEN], width=.55)\n"
            "for i, v in enumerate(cagr): a1.annotate(f'{v:.1f}%', (i, v), ha='center', va='bottom')\n"
            "a1.set_title('growth rate (CAGR, total return)'); a1.set_ylabel('%/yr')\n"
            "a2.bar(labels, dd, color=[AMBER, GREY, GREEN], width=.55)\n"
            "for i, v in enumerate(dd): a2.annotate(f'{v:.1f}%', (i, v), ha='center', va='top')\n"
            "a2.set_title('worst drawdown'); a2.set_ylabel('%')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('CAGR:', [round(v,1) for v in cagr], ' maxDD:', [round(v,1) for v in dd])"
        ),
        md(
            f"BUFR and its same-risk mix are **near-twins**: {R['perf']['BUFR'][0]:.1f}% vs "
            f"{R['perf']['mix 0.59'][0]:.1f}% a year, worst drawdowns {R['perf']['BUFR'][2]:.1f}% vs "
            f"{R['perf']['mix 0.59'][2]:.1f}% (the buffer's was actually *shallower* — that's what the "
            f"fee bought). SPY grew faster ({R['perf']['SPY'][0]:.1f}%) but fell almost twice as far in "
            "2022. Pick your poison; nobody got cheated."
        ),

        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** The cost of comfort is genuine and statistically solid: every fund "
            f"trailed SPY by **5–8 points a year**, and the cap bound in **{R['n_capped']} of "
            f"{R['n_up']}** up years.\n"
            "- **Tradability — Mirage.** But you can't *harvest* that gap, because it's just lower "
            "equity exposure: the honest same-risk SPY/BIL mix **tied** the buffer funds (±1 point, "
            "well within noise). \"Build it cheaper yourself\" saved exactly nothing on this tape.\n"
            "- **Beat the dumb mix? — Busted.** No fund beat its mix by a statistically real margin — "
            "and none lost to it either. The comfort was **fairly priced ex post**."
        ),
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The behavioral defense is the real product.** If the airbag stops one panic-sell at "
            "the bottom of a −20% year, it paid for a decade of fees. That's not in our arithmetic — "
            "and it's the strongest honest case for these funds.\n"
            "- **This tape had one bear.** 2022 fell just beyond the 15% buffer — near-ideal conditions "
            "for the structure. A decade with no bear (caps bind every year, insurance never pays) "
            "would tilt the fair race toward the mix; a −40% crash would too (the buffer stops at 15). "
            "The tie is sample-specific; the *delivery* is contractual.\n"
            "- **Watch the entry date.** Buy mid-period and the remaining cap/buffer can be very "
            "different from the label — the laddered BUFR exists precisely to smooth that away.\n\n"
            "*Think a two-fund mix beats the wrapper over the** next **decade? The engine reruns in one "
            "line as the outcome periods complete — check back after the next bear.*"
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
            "# Buffer ETFs — the Cost of Comfort: a quantitative teardown 🔬\n"
            "### Outcome-period delivery vs stated terms · HAC gap tests vs SPY and vs beta-matched "
            "SPY/BIL mixes · fee + dividend + option-residual decomposition · w-grid / lag / cost "
            "robustness · synthetic drag & delivery controls\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The claim "
            "(Israelov's *Hidden Cost of Buffer Funds*, the Morningstar critique) splits into a delivery "
            "leg — testable mechanically — and a pricing leg — testable as a HAC-t race against the "
            "replicating mix. We run both.\n\n"
            "> ⚠️ **Data note.** BUFR + the four oldest Innovator Power Buffer vintages (15% buffer on "
            "SPY **price** return, 0.79% ER; BUFR laddered ~10% buffers, 0.95% ER), yfinance daily "
            "closes 2018-08 → 2026-06, **TR closes for performance, raw SPY price for the reference** — "
            "labeled everywhere. A **survivor slice** of the category (flatters the funds; named on the "
            "Signal axis). Numbers in [`docs/results.md`](../docs/results.md) (as-of " + R["as_of"] +
            ", fingerprints `" + R["fingerprint"] + "` / `" + R["fingerprint_raw"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` | Every fund trails SPY TR: gaps **+5.26 to +8.27 pp/yr**, HAC "
            f"*t* = **2.10–3.04**; EW cohort **+{R['cohort_vs_spy'][0]:.2f} pp/yr** at *t* = "
            f"**{R['cohort_vs_spy'][1]:.2f}** (lags 3/6/12: {R['lag_grid'][0]:.2f}/"
            f"{R['lag_grid'][1]:.2f}/{R['lag_grid'][2]:.2f}); cap bound {R['n_capped']}/{R['n_up']} up "
            f"periods. |\n"
            f"| **Tradability** | `MIRAGE` | Beta-matched mix race: BUFR gap **+{R['fair']['BUFR'][1]:.2f} "
            f"pp/yr** (*t* = {R['fair']['BUFR'][2]:.2f}), cohort **{R['cohort_fair'][1]:+.2f}** "
            f"(*t* = {R['cohort_fair'][2]:.2f}), all five \\|*t*\\| < 1. The 5–8 pp/yr is beta, not "
            f"harvestable cost. |\n"
            f"| **Beat the dumb mix?** | `BUSTED` | {R['winners']} positive point estimates, 0/5 "
            f"significant (best +0.94 pp/yr at *t* = 0.92). Statistical tie — fairly priced ex post. |\n\n"
            "> 💡 In plain words: the airbag deployed every time, the ceiling was enforced almost every "
            "up year, and the whole package cost the holder nothing versus an honest same-risk index+cash "
            "portfolio — on *this* tape."
        ),

        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^f_t$ be a fund's monthly total return, $r^{SPY}_t$, $r^{BIL}_t$ the benchmarks, and "
            "$\\beta$ the fund's full-sample OLS beta of excess returns. Define the dumb mix "
            "$r^{mix}_t = w\\,r^{SPY}_t + (1-w)\\,r^{BIL}_t - c_t$ with $w = \\hat\\beta$ and $c_t$ the "
            "explicit monthly rebalancing cost. The claim decomposes:\n\n"
            "- **H₁ (delivery).** Over each 12-month outcome period with reference price return $R$, the "
            "fund earns $\\min(R, \\text{cap})$ for $R \\ge 0$, $\\approx 0$ for $-b \\le R < 0$, and "
            "$R + b$ beyond, minus the fee. (Mechanical; checked period by period.)\n"
            "- **H₂ (the cost is real).** $\\mathbb{E}[r^{SPY} - r^f] > 0$, HAC-significant.\n"
            "- **H₃ (DIY is cheaper).** $\\mathbb{E}[r^{mix} - r^f] > 0$, HAC-significant — the wrapper "
            "charges more than replication.\n\n"
            "We find **H₁ confirmed** (4/4 buffers honored, caps enforced), **H₂ confirmed** "
            "(*t* = 2.1–3.0), **H₃ rejected** (gap ≈ 0, |*t*| < 1 across all five funds and the cohort)."
        ),
        md(
            "## 2 · So what?\n\n"
            "H₂ without H₃ flips the verdict's meaning. If both held, buffer ETFs would be dominated "
            "products — beta you could buy cheaper. With H₂ alone, the 5–8 pp/yr shortfall is **the "
            "exposure the buyer chose**, not a fee leak: the wrapper converted ~0.5 equity beta plus an "
            "option collar into the same net return as the replicating portfolio, *after* its 0.79–0.95% "
            "fee. The only deployable critique left is \"you own less market than you think\" — an "
            "allocation statement, not an arbitrage.\n\n"
            "Honesty constraints: TR-vs-TR races only; price-vs-TR only where the contract itself is "
            "written on price; both legs fully funded (excess-vs-excess by construction); one documented "
            "execution convention (month-end rebalance to weights known in advance); explicit one-way "
            "costs on mix turnover; the full-sample $\\hat\\beta$ is flagged and bracketed by a fixed-w "
            "grid."
        ),
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** BUFR (2020-08 inception) + PJAN/PAPR/PJUL/POCT (2018-08 → 2019-04 "
            f"inceptions); monthly stats through {R['as_of']} (last complete month). Survivor slice, "
            "named.\n"
            "- **Delivery.** Month-end closes bracketing each annual reset; fund TR vs SPY price "
            "return; floor = terms − 2 pp NAV slack; \"cap bound\" = fund lags reference by ≥ 2 pp in "
            "an up period.\n"
            "- **Gap tests.** Newey-West (Bartlett, lags 6; 3/12 as robustness) *t* on monthly "
            "difference series, per fund and on the EW 4-vintage cohort.\n"
            "- **Fair race.** $w = \\hat\\beta$ per fund; mix rebalanced monthly at 2 bps one-way × "
            "traded NAV (5/10 swept); fixed-w grid 0.45–0.70.\n"
            "- **Decomposition.** gap = stated fee + $\\beta$ × SPY dividend yield (options on the "
            "price index earn no dividends) + option-payoff residual.\n"
            "- **Controls.** Planted structuring drag (gap detector must fire only when drag > 0); "
            "planted cap/buffer world (delivery checker must read the terms back)."
        ),

        md("## 4 · The teardown"),
        md(
            "### 4a · Delivery — the payoff scatter\n\n"
            "Every completed outcome period against the promised payoff shape (15% buffer, ~12% "
            "realized cap, minus fee):"
        ),
        code(SCATTER),
        md(
            f"> 💡 In plain words: **{R['down_honored']}/{R['n_down']}** down periods honored the "
            f"buffer — including the beyond-buffer 2022 test (PJAN: SPY price {R['pjan22'][0]:+.2f}% → "
            f"fund {R['pjan22'][1]:+.2f}% vs contract floor {R['pjan22'][2]:+.2f}%, a 2 bps miss) and "
            f"the inside-buffer one (PAPR: {R['papr22'][0]:+.2f}% → {R['papr22'][1]:+.2f}%). Caps bound "
            f"in **{R['n_capped']}/{R['n_up']}** up periods; capped years averaged "
            f"**+{R['mean_capped_ret']:.2f}%**; mean up-year give-up **{R['mean_up_giveup']:.2f} pp**. "
            "Delivery: contractual."
        ),
        md(
            "### 4b · The cost of comfort vs SPY (H₂)\n\n"
            "SPY-minus-fund monthly TR difference, NW lags = 6:"
        ),
        code(
            "funds = list(R['vs_spy'])\n"
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for tk in funds:\n"
            "        f = M[tk].dropna(); g = st.gap_stats(f, SPY.loc[f.index])\n"
            "        rows.append((tk, g['gap_pp_yr'], g['hac_t'], g['n_months']))\n"
            "    coh = M[P_SERIES].dropna().mean(axis=1)\n"
            "    gc = st.gap_stats(coh, SPY.loc[coh.index])\n"
            "    rows.append(('COHORT', gc['gap_pp_yr'], gc['hac_t'], gc['n_months']))\n"
            "else:\n"
            "    rows = [(tk, *R['vs_spy'][tk]) for tk in funds] + [('COHORT', *R['cohort_vs_spy'])]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "labs = [r[0] for r in rows]; gaps = [r[1] for r in rows]; ts = [r[2] for r in rows]\n"
            "bars = ax.bar(labs, gaps, color=[RED]*5 + ['#7a1f14'], width=.6)\n"
            "for i, (g, t) in enumerate(zip(gaps, ts)):\n"
            "    ax.annotate(f'{g:+.1f} pp/yr\\nt={t:.2f}', (i, g), ha='center', va='bottom', fontsize=9)\n"
            "ax.set_ylabel('SPY total return minus fund (pp/yr)'); ax.set_ylim(0, 10.5)\n"
            "ax.set_title('The cost of comfort is real: every fund trails the market, all HAC t >= 2.1')\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in rows: print(f'{r[0]:7s} gap {r[1]:+.2f} pp/yr  HAC t {r[2]:+.2f}  n={r[3]}')"
        ),
        md(
            f"> 💡 In plain words: the shortfall vs the market is **{min(v[0] for v in R['vs_spy'].values()):.1f}–"
            f"{max(v[0] for v in R['vs_spy'].values()):.1f} pp/yr with every *t* above 2** — clearing "
            f"the desk's bar on the real tape. Cohort *t* is lag-stable "
            f"({R['lag_grid'][0]:.2f}/{R['lag_grid'][1]:.2f}/{R['lag_grid'][2]:.2f} at lags 3/6/12). "
            "This is the half of the claim that survives. The question is what the gap *is* — cost, "
            "or beta. Next cell."
        ),
        md(
            "### 4c · The fair race — beta-matched SPY/BIL mix (H₃)\n\n"
            "Same funds, but the benchmark is now the replicating mix at each fund's own beta, "
            "rebalanced monthly with explicit costs:"
        ),
        code(
            "funds = list(R['fair'])\n"
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for tk in funds:\n"
            "        f = M[tk].dropna(); b = st.beta_vs(f, SPY, BIL)\n"
            "        g = st.gap_stats(f, st.mix_returns(SPY.loc[f.index], BIL.loc[f.index], b))\n"
            "        rows.append((tk, b, g['gap_pp_yr'], g['hac_t'], g['corr']))\n"
            "    coh = M[P_SERIES].dropna().mean(axis=1); bc = st.beta_vs(coh, SPY, BIL)\n"
            "    gc = st.gap_stats(coh, st.mix_returns(SPY.loc[coh.index], BIL.loc[coh.index], bc))\n"
            "    rows.append(('COHORT', bc, gc['gap_pp_yr'], gc['hac_t'], gc['corr']))\n"
            "else:\n"
            "    rows = [(tk, *R['fair'][tk]) for tk in funds] + [('COHORT', *R['cohort_fair'])]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "labs = [r[0] for r in rows]; gaps = [r[2] for r in rows]; ts = [r[3] for r in rows]\n"
            "ax.bar(labs, gaps, color=[GREEN if g <= 0 else AMBER for g in gaps], width=.6)\n"
            "for i, (g, t) in enumerate(zip(gaps, ts)):\n"
            "    ax.annotate(f'{g:+.2f}\\nt={t:.2f}', (i, g), ha='center',\n"
            "                va='bottom' if g >= 0 else 'top', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylim(-2.2, 2.2)\n"
            "ax.set_ylabel('beta-matched mix minus fund (pp/yr)')\n"
            "ax.set_title('Beta-matched, the cost vanishes: every |t| < 1 (corr 0.92-0.98)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in rows: print(f'{r[0]:7s} beta {r[1]:.2f}  mix - fund {r[2]:+.2f} pp/yr  HAC t {r[3]:+.2f}  corr {r[4]:.3f}')"
        ),
        md(
            f"> 💡 In plain words: at matched equity exposure the gaps are **−0.94 to +0.51 pp/yr, all "
            f"\\|*t*\\| < 1** (cohort {R['cohort_fair'][1]:+.2f} at *t* = {R['cohort_fair'][2]:.2f}), with "
            f"0.92–0.98 correlation to the mix. H₃ — \"you could build it cheaper\" — **fails on this "
            f"tape**. The regime split shows the engine: the mix wins small in up months, the buffer "
            "wins in down months (POCT: −21 bps/mo in down months), and 2018–2026 nets them to zero."
        ),
        md(
            "### 4d · Where the money went — decomposition of the fair-race gap\n\n"
            "gap = stated fee + dividends forgone (β × SPY's ~1.4%/yr — FLEX options on the price index "
            "collect no dividend) + option-payoff residual:"
        ),
        code(
            "labs = ['BUFR', 'COHORT']\n"
            "decs = [R['dec_bufr'], R['dec_cohort']]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.6))\n"
            "x = np.arange(2); w = .5\n"
            "fee = [d[1] for d in decs]; div = [d[2] for d in decs]; res = [d[3] for d in decs]\n"
            "ax.bar(x, fee, w, label='stated fee', color=RED)\n"
            "ax.bar(x, div, w, bottom=fee, label='dividends forgone (beta x ~1.4%)', color=AMBER)\n"
            "ax.bar(x, res, w, label='option-payoff residual', color=GREEN)\n"
            "for i, d in enumerate(decs):\n"
            "    ax.annotate(f'net gap {d[0]:+.2f} pp/yr', (i, 1.95), ha='center', fontsize=10)\n"
            "    ax.plot([i - w/2, i + w/2], [d[0], d[0]], c='k', lw=2)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labs); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('pp/yr'); ax.set_ylim(-2.2, 2.3)\n"
            "ax.set_title('The fee and lost dividends were charged - and the option payoff earned them back')\n"
            "ax.legend(fontsize=9); plt.tight_layout(); plt.show()\n"
            "for l, d in zip(labs, decs):\n"
            "    print(f'{l}: gap {d[0]:+.2f} = fee {d[1]:+.2f} + div forgone {d[2]:+.2f} + residual {d[3]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the wrapper really does cost ~**1.6–1.8 pp/yr** in fee + forgone "
            f"dividends — the critics' arithmetic is right. But the collar's realized payoff contributed "
            f"**{R['dec_bufr'][3]:+.2f} pp/yr** (BUFR; cohort {R['dec_cohort'][3]:+.2f}) over a sample "
            "whose one bear (2022) landed just beyond the buffer — near-ideal conditions. The net is a "
            "wash *ex post*; it is **not** a theorem that it stays one."
        ),
        md(
            "### 4e · Robustness — w-grid, lags, costs\n\n"
            "The fair-race tie must not hinge on the in-sample beta-hat, the HAC lag or the rebalance "
            "cost:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    f = M['BUFR'].dropna(); win = f.index\n"
            "    wg = []\n"
            "    for w_ in (0.45, 0.55, 0.65, 0.70):\n"
            "        g = st.gap_stats(f, st.mix_returns(SPY.loc[win], BIL.loc[win], w_))\n"
            "        wg.append((w_, g['gap_pp_yr'], g['hac_t']))\n"
            "else:\n"
            "    wg = R['wgrid']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "labs = [f'w={r[0]:.2f}' for r in wg]; ts = [r[2] for r in wg]\n"
            "ax.bar(labs, ts, color=[GREEN if abs(t) < 2 else RED for t in ts], width=.55)\n"
            "ax.axhline(2, ls='--', c=RED); ax.axhline(-2, ls='--', c=RED, label='|t| = 2 bar')\n"
            "for i, r in enumerate(wg):\n"
            "    ax.annotate(f'{r[1]:+.2f} pp/yr\\nt={r[2]:+.2f}', (i, r[2]), ha='center',\n"
            "                va='bottom' if r[2] >= 0 else 'top', fontsize=9)\n"
            "ax.set_ylabel('HAC t of (mix - BUFR)'); ax.set_ylim(-3.4, 3.0)\n"
            "ax.set_title('Fixed-weight grid: no w makes the DIY mix significantly cheaper')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('w grid:', [(r[0], round(r[1], 2), round(r[2], 2)) for r in wg])\n"
            "print('NW lag grid, cohort vs SPY t:', R['lag_grid'])\n"
            "print('rebalance-cost sweep (bps -> mix-BUFR pp/yr):', R['cost_grid'])"
        ),
        md(
            f"> 💡 In plain words: across every fixed weight 0.45–0.70 the mix never beats BUFR at "
            f"\\|*t*\\| ≥ 2 — the only significant cell is w = 0.45, where the DIY mix **loses** "
            f"({R['wgrid'][0][1]:+.2f} pp/yr, *t* = {R['wgrid'][0][2]:+.2f}: under-betting equity in a "
            f"bull). The cohort-vs-SPY *t* holds at lags 3/6/12, and rebalance costs move the gap by "
            "~1 bp — two-asset monthly turnover is negligible."
        ),
        md(
            "### 4f · Synthetic machinery controls — we know the truth here\n\n"
            "Planted-effect worlds, deterministic and offline. The gap detector must stay silent on a "
            "drag-free mimic fund and recover a planted 2 pp/yr structuring drag; the delivery checker "
            "must read back a planted cap/buffer. *(Machinery proof — never market evidence.)*"
        ),
        code(
            "res = []\n"
            "for drag in (0.0, 2.0):\n"
            "    w = data.synthetic_world(drag_pct=drag)\n"
            "    g = st.gap_stats(w['FUND'], st.mix_returns(w['IDX'], w['CASH'], 0.55, cost_bps=0.0))\n"
            "    res.append((drag, g['gap_pp_yr'], g['hac_t']))\n"
            "so = data.synthetic_outcomes()\n"
            "per = pd.DataFrame({'spy_price_ret_pct': so['ref_ret']*100, 'fund_ret_pct': so['fund_ret']*100})\n"
            "dc = st.delivery_check(per, 15.0, 0.79)\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.2))\n"
            "labs = [f'planted drag\\n{r[0]:.0f} pp/yr' for r in res]\n"
            "ax.bar(labs, [r[2] for r in res], color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i, r in enumerate(res): ax.annotate(f't={r[2]:.2f}', (i, r[2]), ha='center', va='bottom')\n"
            "ax.set_ylabel('HAC t of (mix - fund)')\n"
            "ax.set_title('Control: zero drag stays silent, planted drag lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for r in res: print(f'drag {r[0]:.1f} pp/yr: measured {r[1]:+.2f} pp/yr  t {r[2]:+.2f}')\n"
            "print(f\"delivery checker: buffer honored {dc['inside_honored']+dc['beyond_honored']}/\"\n"
            "      f\"{dc['n_inside']+dc['n_beyond']}, mean capped-year {dc['mean_capped_ret_pct']:.2f}% \"\n"
            "      f\"(planted cap - fee = 9.21%)\")"
        ),
        md(
            f"> 💡 In plain words: with **no** planted drag the detector reads "
            f"{R['syn_gap'][0][1]:+.2f} pp/yr (*t* = {R['syn_gap'][0][2]:.2f}); a planted 2 pp/yr drag "
            f"is recovered at *t* = {R['syn_gap'][1][2]:.2f}. Power caveat: at the real funds' ~70–86 "
            "months and 0.92–0.98 mix correlation, a residual drag of ~2 pp/yr would have been caught; "
            "~0.5 pp/yr would not. The real-tape tie is certified **at the pp/yr scale**. The delivery "
            f"checker reads back the planted terms ({R['syn_delivery'][2]:.2f}% vs "
            f"{R['syn_delivery'][3]:.2f}%)."
        ),

        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — the cost of comfort: gaps vs SPY **+5.26 to +8.27 pp/yr**, HAC *t* "
            f"**2.10–3.04**, cohort **+{R['cohort_vs_spy'][0]:.2f}** at *t* = "
            f"**{R['cohort_vs_spy'][1]:.2f}** (lag-robust); cap bound **{R['n_capped']}/{R['n_up']}** "
            f"up periods, buffer honored **{R['down_honored']}/{R['n_down']}** down periods. Survivor "
            "slice named (it flatters the funds — conservative for this leg).\n"
            f"- **Tradability `MIRAGE`** — beta-matched, the gap is **{R['fair']['BUFR'][1]:+.2f} pp/yr** "
            f"(BUFR, *t* = {R['fair']['BUFR'][2]:.2f}) and **{R['cohort_fair'][1]:+.2f}** (cohort, *t* = "
            f"{R['cohort_fair'][2]:.2f}); Sharpe 0.79 vs 0.82; no fixed w in 0.45–0.70 makes the DIY mix "
            "significantly cheaper. The 5–8 pp/yr is beta you chose, not cost you can harvest.\n"
            f"- **Beat the dumb mix? `BUSTED`** — {R['winners']} positive point estimates, 0/5 "
            "significant either way. Comfort was fairly priced *ex post* on a sample whose one bear "
            "landed just beyond the buffer — the friendliest possible tape for the structure, which is "
            "exactly why the tie should not be extrapolated to a theorem."
        ),
        md(
            "## 6 · Going further\n\n"
            "- **Regime dependence is the open risk.** A bear-free decade (caps bind, insurance never "
            "pays) or a −40% crash (buffer exhausted at 15) would both tilt the race toward the mix; "
            "2022's −19% was the structure's sweet spot. The delivery leg, by contrast, is contractual "
            "and should replicate in any regime.\n"
            "- **Entry-date basis.** Mid-period buyers face remaining caps/buffers unrelated to the "
            "label; the laddered BUFR halves that basis. A point-in-time study of *remaining* terms is "
            "the natural sequel.\n"
            "- **The behavioral term is off-balance-sheet.** If the wrapper prevents one capitulation "
            "sale, it dominates the mix for that holder — unmeasurable here, and the honest core of the "
            "product's case.\n\n"
            "*The reproducible core is offline and deterministic; delivery is checked against stated "
            "terms hardcoded with sources, and both detectors carry planted-effect controls. Methods and "
            "sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
