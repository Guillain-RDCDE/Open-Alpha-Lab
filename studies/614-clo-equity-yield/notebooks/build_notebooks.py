"""Generate the two narrative notebooks for Study 614 (CLO Equity Yield).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached double tape
(total-return + price-only) under ../_cache/ and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md). The synthetic control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance double tape,
# ECC 2014-11-30 -> 2026-06-30 (140 months), OXLC 2011-02 -> 2026-06 (185), as-of 2026-06-30).
R = dict(
    start="2014-11-30", end="2026-06-30", n_months=140, years=11.7, as_of="2026-06-30",
    fingerprint="366d9cf7d356",
    # per-fund: (n_months, dist bps/mo, dist %/yr, dist HAC t, TR CAGR %, PX CAGR %, $100 TR, $100 PX)
    decomp={
        "ECC":  (140, 133.30, 17.22, 15.33, 1.75, -13.46, 122.5, 18.5),
        "OXLC": (185, 147.34, 19.19, 15.35, 2.52, -14.38, 146.7,  9.1),
    },
    # perf on the joint ECC window: (CAGR %, vol %, Sharpe excess, maxDD %)
    perf={"ECC": (1.75, 24.63, 0.14, -59.37), "OXLC": (0.64, 32.29, 0.14, -61.08),
          "HYG": (4.16, 7.21, 0.34, -15.25), "SPY": (13.78, 14.97, 0.82, -23.93),
          "BIL": (1.88, 0.57, None, -0.16)},
    # excess over bills, own windows: (bps/mo, HAC t)
    excess={"ECC": (28.47, 0.43), "OXLC": (47.51, 0.81)},
    # HEADLINE spread vs HYG, own windows: (bps/mo, %/yr, HAC t, n, fund CAGR %, HYG CAGR %)
    spread={"ECC": (7.87, 0.95, 0.13, 140, 1.75, 4.16), "OXLC": (17.78, 2.15, 0.34, 185, 2.52, 4.80)},
    # race: (alpha bps/mo, alpha %/yr, t_alpha, b_HYG, t_HYG, b_SPY, t_SPY, R2,
    #        fund CAGR %, bench CAGR %, fund DD %, bench DD %, t_spread)
    race={
        "ECC":  (-18.84, -2.24, -0.33, 1.42, 1.63, 0.18, 0.80, 0.25, 1.75,  6.98, -59.4, -25.5, -0.34),
        "OXLC": (-33.55, -3.95, -0.66, 1.24, 2.81, 0.41, 2.59, 0.24, 2.52, 10.60, -61.1, -28.2, -0.71),
    },
    # robustness across NW lags, own windows: {name: [(lags, spread t, alpha t), ...]}
    robust_lags={"ECC": [(3, 0.14, -0.36), (6, 0.13, -0.33), (12, 0.13, -0.30)],
                 "OXLC": [(3, 0.36, -0.67), (6, 0.34, -0.66), (12, 0.33, -0.65)]},
    # subperiods (joint ECC-window frame): {label: {name: (dist %/yr, dist t, TR %, PX %, spread bps, spread t, n)}}
    sub={"2016-01+": {"ECC": (17.67, 14.9, 2.44, -13.19, 6.53, 0.10, 126),
                      "OXLC": (21.17, 14.0, 3.06, -15.25, 30.75, 0.43, 126)},
         "2020-07+": {"ECC": (19.86, 11.1, 7.98, -10.23, 38.29, 0.43, 72),
                      "OXLC": (21.04, 9.7, 5.30, -13.25, 32.98, 0.34, 72)}},
    # third axis: (dist %/yr, px %/yr, financed-by-price %, kept TR %/yr)
    roc={"ECC": (17.22, -13.46, 78.1, 1.75), "OXLC": (19.19, -14.38, 74.9, 2.52)},
    # crises: (label, ECC, OXLC, HYG, SPY) peak-to-trough TR drawdown %
    crises=[("Credit crunch 2015-16", -29.7, -57.1, -13.4, -13.0),
            ("Q4 2018",               -25.6, -23.4,  -6.4, -19.3),
            ("COVID 2020",            -66.1, -73.8, -22.0, -33.7),
            ("Rate shock 2022",       -19.1, -36.8, -15.5, -24.5)],
    # synthetic: (planted carry bps, planted alpha bps, dist bps, dist t, alpha bps, alpha t)
    syn=[(0.0, 0.0, 0.85, 0.33, 33.40, 0.99),
         (120.0, -150.0, 120.85, 47.36, -116.60, -3.44)],
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Return_OF_capital%3F: Confirmed](https://img.shields.io/badge/Return_OF_capital%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from clo_equity_yield import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    TR, PX = data.load_real()
    MTR, MPX = st.aligned_monthly(TR, PX, ["ECC", "OXLC", "HYG", "SPY", "BIL"])
else:
    TR = PX = MTR = MPX = None
print("real CLO-fund cache present:", HAVE_REAL,
      "| months (ECC window):", (0 if MTR is None else len(MTR)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    d = R["decomp"]
    roc = R["roc"]
    cells = [
        md(
            "# The 15% CLO-equity machine — real income, or your own money handed back? 🏭\n"
            "### ECC and OXLC, taken apart in plain English\n\n"
            + BADGES +
            "Sort every listed fund in America by yield and two names sit near the top, year after "
            "year: **Oxford Lane (OXLC)** and **Eagle Point (ECC)**, the CLO-equity funds, paying "
            "**15, 17, 19 percent**. The pitch: a CLO's equity tranche collects the leftover interest "
            "of a whole portfolio of corporate loans after the safer investors are paid — a fire hose "
            "of cash — and these funds point the hose at your brokerage account, *monthly*.\n\n"
            "Here's the strange thing: **the fire hose is real.** We measure it below at 17-19% a "
            "year, statistically one of the most solid numbers on this desk. And yet a decade-plus of "
            "harvesting it compounded to roughly **T-bill money**, and the share price of OXLC lost "
            "**91% of your capital** along the way. This notebook is about how both things can be "
            "true at once.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the benchmark regression and the "
            "robustness grid? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> 🔬 **For the quants:** there is no index fund for CLO equity — the listed category "
            "essentially *is* these two survivors, so the panel is survivor-tilted by construction "
            "(named on the Signal axis; the honest reading is that the category's full economics are, "
            "if anything, worse). Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Is the ~15% distribution real? | **Yes — it's actually bigger.** ECC paid out "
            f"**{d['ECC'][2]:.1f}%/yr** and OXLC **{d['OXLC'][2]:.1f}%/yr**, measured on the tape, "
            "for 11.7 and 15.4 years. This part of the pitch is honest. |\n"
            f"| Did the income make you rich? | **No.** $100 in ECC with every distribution "
            f"reinvested became **${d['ECC'][6]:.0f}** in 11.7 years (~{d['ECC'][4]:.1f}%/yr — about "
            "T-bills); OXLC did similarly on its own window. Plain HYG beat both. |\n"
            "| So where did the 15-19% come from? | Largely **out of the share price**: ECC's price "
            f"leg eroded **{d['ECC'][5]:.0f}%/yr**, OXLC's **{d['OXLC'][5]:.0f}%/yr**. $100 of OXLC "
            f"*capital* became **${d['OXLC'][7]:.0f}**. About **three quarters** of the payout was "
            "your own money coming back. |\n"
            "| Was it at least safe on the way? | **The opposite.** CLO equity is the *first-loss* "
            "slice: −66% and −74% in eight weeks in 2020, −57% for OXLC in the 2015-16 credit "
            "crunch — always 2-3× worse than plain high-yield bonds. |\n\n"
            "> The CLO-equity coupon is a genuine cash stream — the biggest in the desk's "
            "packaged-carry family — strapped to a wrapper that consumes principal and concentrates "
            "crashes. Real coupon, mirage vehicle."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A CLO takes a few hundred corporate loans, funds them with cheap AAA-to-BB debt, "
            "and whoever holds the* equity tranche *keeps everything left over — historically 15-25% "
            "cash-on-cash a year. ECC and OXLC hold portfolios of these equity tranches and pass the "
            "cash to you as monthly distributions. Yes the price swings; the* income *is real.\"*\n\n"
            "This is the top shelf of the income-investing universe — the highest headline yields a "
            "regular brokerage account can buy. And unlike most desk folklore, the *mechanism* is "
            "real: CLO equity genuinely throws off huge cash flows (the loans pay floating-rate "
            "interest; the CLO's debt costs less; the equity keeps the spread). The question is what "
            "happens to your *capital* while the cash arrives."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the pitch were right, retirees could quintuple their income overnight. If it's wrong "
            "the way we suspect, the 15-19% is largely **a return OF capital dressed as a return ON "
            "capital** — the classic closed-end-fund illusion, at maximum volume. This is the fifth "
            "asset the desk has caught in the same costume: bank-loan funds "
            "([340](../../340-bank-loans/README.md)), pipelines ([341](../../341-mlp-pipelines/README.md)), "
            "BDCs ([342](../../342-bdc-yield/README.md)) and mortgage REITs "
            "([611](../../611-mreit-carry/README.md)) all ran the same play. The CLO flavour is the "
            "most extreme: the coupon is the residual of a **first-loss tranche** of a ~10× levered "
            "loan pool, and the fund adds *its own* leverage and fees on top."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "One trick does all the work: we download every price series **twice** — once including "
            "reinvested distributions (**total return**: what you actually got) and once as the raw "
            "share price (**price-only**: what happened to your capital). Subtract them month by "
            "month and the difference is exactly the **distribution stream** — the 15% the pitch "
            "sells.\n\n"
            "Then three questions:\n"
            "1. **Is the payout real?** Average the distribution component; test it hard.\n"
            "2. **What did it cost?** Look at what the price leg did while the cash flowed.\n"
            "3. **Could you have done better with one boring click?** Compare the funds' *total* "
            "return to plain **HYG** — the ordinary high-yield bond ETF sitting in the same "
            "brokerage account (the quants notebook also runs a full benchmark regression)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the whole story in one picture:** $100 in ECC, 2014→2026 — with distributions "
            "reinvested (what you got) vs the share price alone (what happened to your capital), "
            "against boring HYG and boring SPY."
        ),
        code(
            "if HAVE_REAL:\n"
            "    w = {c: (1.0 + MTR[c]).cumprod() * 100 for c in ['ECC','HYG','SPY']}\n"
            "    w_px = (1.0 + MPX['ECC']).cumprod() * 100\n"
            "    fig, ax = plt.subplots(figsize=(9.8, 5.2))\n"
            "    ax.plot(w['SPY'], c=GREY, lw=1.6, label=f\"SPY total return  (${w['SPY'].iloc[-1]:.0f})\")\n"
            "    ax.plot(w['HYG'], c=AMBER, lw=1.6, label=f\"HYG high-yield bonds  (${w['HYG'].iloc[-1]:.0f})\")\n"
            "    ax.plot(w['ECC'], c=GREEN, lw=2.0, label=f\"ECC total return — distributions reinvested  (${w['ECC'].iloc[-1]:.0f})\")\n"
            "    ax.plot(w_px, c=RED, lw=2.0, label=f\"ECC price only — your capital  (${w_px.iloc[-1]:.0f})\")\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $100 (log scale)')\n"
            "    ax.set_title('11.7 years of harvesting a 17%/yr coupon: $100 -> $122 (T-bill money)')\n"
            "    ax.legend(loc='lower left', fontsize=9)\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f\"$100 -> ECC TR ${w['ECC'].iloc[-1]:.1f} | ECC price-only ${w_px.iloc[-1]:.1f} | \"\n"
            "          f\"HYG ${w['HYG'].iloc[-1]:.1f} | SPY ${w['SPY'].iloc[-1]:.1f}\")\n"
            "else:\n"
            "    print('cache missing - frozen numbers:', '$100 -> ECC TR $%.1f, price-only $%.1f'\n"
            "          % (R['decomp']['ECC'][6], R['decomp']['ECC'][7]))"
        ),
        md(
            f"The green line is the *best case* — every distribution reinvested, 11.7 years of "
            f"17%/yr coupons — and it ends at **${d['ECC'][6]:.0f}**, behind the boring high-yield "
            f"ETF that pays a third of the yield. The red line is your actual capital: "
            f"**${d['ECC'][7]:.0f}** left of $100. On OXLC's longer tape the red line ends at "
            f"**${d['OXLC'][7]:.0f}**.\n\n"
            "**Now the decomposition per fund.** Green = the distribution stream (real, harvested); "
            "red = what the share price did; amber = the total you actually kept."
        ),
        code(
            "names = ['ECC', 'OXLC']\n"
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for nm in names:\n"
            "        mtr, mpx = st.aligned_monthly(TR, PX, [nm, 'HYG', 'SPY', 'BIL'])\n"
            "        dd = st.decompose(mtr, mpx, nm)\n"
            "        rows.append((dd['dist_ann_pct'], dd['px_cagr_pct'], dd['tr_cagr_pct']))\n"
            "else:\n"
            "    rows = [(R['decomp'][nm][2], R['decomp'][nm][5], R['decomp'][nm][4]) for nm in names]\n"
            "x = np.arange(len(names)); wdt = 0.27\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.8))\n"
            "ax.bar(x - wdt, [r[0] for r in rows], wdt, color=GREEN, label='distribution stream (%/yr) — the coupon')\n"
            "ax.bar(x,       [r[1] for r in rows], wdt, color=RED, label='price-only CAGR (%/yr) — the erosion')\n"
            "ax.bar(x + wdt, [r[2] for r in rows], wdt, color=AMBER, label='total-return CAGR (%/yr) — what you kept')\n"
            "for i, r in enumerate(rows):\n"
            "    ax.annotate(f'{r[0]:+.1f}', (i - wdt, r[0]), ha='center', va='bottom', fontsize=9)\n"
            "    ax.annotate(f'{r[1]:+.1f}', (i, r[1]), ha='center', va='top', fontsize=9)\n"
            "    ax.annotate(f'{r[2]:+.1f}', (i + wdt, r[2]), ha='center', va='bottom', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['ECC (2014->)', 'OXLC (2011->)'])\n"
            "ax.set_ylabel('% per year'); ax.legend(fontsize=9)\n"
            "ax.set_title('The coupon is real - and the share price pays for most of it')\n"
            "plt.tight_layout(); plt.show()\n"
            "for nm, r in zip(names, rows):\n"
            "    print(f'{nm}: distribution {r[0]:+.2f}%/yr  price {r[1]:+.2f}%/yr  total {r[2]:+.2f}%/yr')"
        ),
        md(
            f"Both funds pay a huge, real coupon (**+{d['ECC'][2]:.1f}%** and **+{d['OXLC'][2]:.1f}%** "
            f"a year) and both funds' *prices* erode at **−13 to −14%/yr**. Do the division: "
            f"**{roc['ECC'][2]:.0f}%** of ECC's payout and **{roc['OXLC'][2]:.0f}%** of OXLC's was "
            "offset one-for-one by capital shrinkage. That is the literal definition of being handed "
            "your own money back — and the funds' own SEC-mandated 19(a) notices say the same thing "
            "in accounting language.\n\n"
            "**And the risk while you held it?** Here is the \"income sleeve\" in the four storms of "
            "its life, next to plain high-yield bonds and the stock market:"
        ),
        code(
            "labels = [c[0] for c in R['crises']]\n"
            "if HAVE_REAL:\n"
            "    ct = st.crisis_table(TR, ['ECC', 'OXLC', 'HYG', 'SPY'])\n"
            "    ecc, oxlc = ct['ECC'].tolist(), ct['OXLC'].tolist()\n"
            "    hyg, spy = ct['HYG'].tolist(), ct['SPY'].tolist()\n"
            "else:\n"
            "    ecc = [c[1] for c in R['crises']]; oxlc = [c[2] for c in R['crises']]\n"
            "    hyg = [c[3] for c in R['crises']]; spy = [c[4] for c in R['crises']]\n"
            "x = np.arange(len(labels)); wdt = .2\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.8))\n"
            "ax.bar(x - 1.5*wdt, ecc, wdt, color=RED, label='ECC')\n"
            "ax.bar(x - .5*wdt, oxlc, wdt, color='#e07b5f', label='OXLC')\n"
            "ax.bar(x + .5*wdt, hyg, wdt, color=AMBER, label='HYG (plain high yield)')\n"
            "ax.bar(x + 1.5*wdt, spy, wdt, color=GREY, label='SPY (stocks)')\n"
            "for i in range(len(labels)):\n"
            "    ax.annotate(f'{oxlc[i]:.0f}%', (i - .5*wdt, oxlc[i]), ha='center', va='top', fontsize=9)\n"
            "ax.set_xticks(x); ax.set_xticklabels([l.replace(' 2', '\\n2') for l in labels], fontsize=9)\n"
            "ax.set_ylabel('peak-to-trough drawdown (%)')\n"
            "ax.set_title('First-loss means first to fall: 2-3x the drawdown of plain high yield, every time')\n"
            "ax.legend(fontsize=9); plt.tight_layout(); plt.show()\n"
            "print('OXLC drawdowns:', [f'{v:.1f}%' for v in oxlc])"
        ),
        md(
            "**−66% and −74% in eight weeks** in March 2020, −57% for OXLC in the 2015-16 credit "
            "crunch, −37% in 2022 — always a multiple of what plain high-yield bonds did. Why always? "
            "Because CLO equity *is* the loss-absorber: it exists so that the AAA buyers above it "
            "never lose money. When loan defaults are repriced, the equity tranche's value is the "
            "first thing crossed out — and the funds hold it with a second layer of leverage on top."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** The 15% machine pays — more than advertised: **{d['ECC'][2]:.1f}%** "
            f"and **{d['OXLC'][2]:.1f}%/yr** of hard cash, measured over 11.7 and 15.4 years "
            "(statistically bulletproof). The pitch's central number is honest.\n"
            f"- **Tradability — Mirage.** Harvesting that coupon compounded to **T-bill money** "
            f"(+{d['ECC'][4]:.1f}% and +{d['OXLC'][4]:.1f}%/yr) at −60% drawdowns — behind one-click "
            "HYG at 3-4× the risk, and 5-8 points a year behind a passive copy of the funds' own "
            "risk. The double-digit fee-and-leverage drag inside the wrappers ate the difference.\n"
            f"- **Return OF capital? — Confirmed.** **{roc['ECC'][2]:.0f}%** (ECC) and "
            f"**{roc['OXLC'][2]:.0f}%** (OXLC) of the payout was matched one-for-one by price "
            f"erosion; $100 of capital → **${d['ECC'][7]:.0f}** and **${d['OXLC'][7]:.0f}**. Spend "
            "the distribution and you are eating your principal — with a first-loss credit book "
            "attached."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The family pattern.** Bank-loan funds ([340](../../340-bank-loans/README.md)), "
            "pipelines ([341](../../341-mlp-pipelines/README.md)), BDCs "
            "([342](../../342-bdc-yield/README.md)), mortgage REITs "
            "([611](../../611-mreit-carry/README.md)), and now CLO equity: package a levered risk "
            "premium, label the payout \"income,\" and let the NAV quietly pay for it. When you see "
            "a double-digit yield, the first question is never *\"is it real?\"* — it usually is — "
            "but *\"what is it liquidating to pay me?\"*\n"
            "- **The tell you can check yourself.** Every closed-end fund must file 19(a) notices "
            "saying how much of each distribution is income vs **return of capital**. ECC's and "
            "OXLC's have flagged return-of-capital repeatedly. The chart above is just that notice, "
            "drawn on the tape.\n"
            "- **The honest version of the trade.** Institutional CLO *equity held to maturity* has "
            "earned real (if dispersed) returns — the Cordell-Roberts-Schwert paper in "
            "[docs/references.md](../docs/references.md) is the reference. What retail buys through "
            "a 10%+-cost listed wrapper is a different animal.\n\n"
            "*Think the post-2022 floating-rate regime changes the answer? The engine is in "
            "[`clo_equity_yield/`](../clo_equity_yield/); re-run `examples/verify.py` on any window "
            "— the 2020-07+ subperiod is already in the robustness grid, and the price leg still "
            "erodes −10 to −13%/yr there.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    d = R["decomp"]
    sp = R["spread"]
    rc = R["race"]
    roc = R["roc"]
    cells = [
        md(
            "# CLO Equity Yield — a quantitative teardown 🔬\n"
            "### Total-return vs price-only distribution decomposition (HAC *t*) · TR spread vs HYG "
            "· NW-HAC alpha vs a credit/equity benchmark · lag & subperiod robustness · crisis "
            "autopsies · a synthetic faithful-engine control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — *\"ECC/OXLC pay ~15% of real CLO-equity cash flow\"* — is unusual for this desk "
            "in that its central number is **true** (and understated); the study's job is to measure "
            "what the payout costs, whether any *surplus* exists over the one-click alternative, and "
            "how much of the distribution is a return OF capital.\n\n"
            "> ⚠️ **Data + survivorship note.** Double yfinance tape (auto-adjusted total-return AND "
            "raw split-adjusted price-only; OXLC's 2025-09 1-for-5 reverse split handled on both "
            f"legs) for ECC / OXLC / HYG / SPY / BIL; ECC window {R['start']} → {R['end']} "
            f"({R['n_months']} months), OXLC from 2011-02 (185), as-of {R['as_of']}. **There is no "
            "index fund for CLO equity — the listed category is these two survivors**, so the panel "
            "is survivor-tilted by construction (the category's full economics are, if anything, "
            "worse). Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `REAL` | Distribution component **+{d['ECC'][1]:.2f} bps/mo "
            f"(+{d['ECC'][2]:.2f}%/yr)** on ECC at **HAC t = +{d['ECC'][3]:.2f}** (OXLC "
            f"+{d['OXLC'][2]:.2f}%/yr, t = +{d['OXLC'][3]:.2f}); the surplus over plain HYG is "
            f"**zero** (spread HAC t = +{sp['ECC'][2]:.2f} / +{sp['OXLC'][2]:.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | TR CAGR **+{d['ECC'][4]:.2f}% / +{d['OXLC'][4]:.2f}%/yr** "
            f"(excess over bills t = +{R['excess']['ECC'][1]:.2f} / +{R['excess']['OXLC'][1]:.2f} — "
            f"cash), maxDD **{R['perf']['ECC'][3]:.0f}% / {R['perf']['OXLC'][3]:.0f}%**, alpha vs "
            f"HYG+SPY **{rc['ECC'][1]:.1f}% / {rc['OXLC'][1]:.1f}%/yr** (t = {rc['ECC'][2]:.2f} / "
            f"{rc['OXLC'][2]:.2f}), and the beta-matched copy compounded 5-8 pp/yr faster at half "
            "the drawdown. |\n"
            f"| **Return OF capital?** | `CONFIRMED` | **{roc['ECC'][2]:.1f}%** (ECC) / "
            f"**{roc['OXLC'][2]:.1f}%** (OXLC) of the payout stream offset one-for-one by price "
            f"erosion; $100 of capital → **${d['ECC'][7]:.1f} / ${d['OXLC'][7]:.1f}** price-only. |\n\n"
            "> 💡 In plain words: the coupon is real and enormous; the wrapper converts it into "
            "T-bill money at first-loss-credit risk. Real carry, mirage package — the fifth (and "
            "most extreme) member of the family: [340](../../340-bank-loans/README.md), "
            "[341](../../341-mlp-pipelines/README.md), [342](../../342-bdc-yield/README.md), "
            "[611](../../611-mreit-carry/README.md)."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{TR}_{t}$ and $r^{PX}_{t}$ be a fund's monthly total-return and price-only "
            "returns. The **distribution component** (the coupon the pitch sells) is\n\n"
            "$$c_t = r^{TR}_t - r^{PX}_t \\ge 0,$$\n\n"
            "and the claim decomposes into:\n\n"
            "- **H₁ (the payout exists).** $\\bar{c} \\approx$ 15%/yr, significant under a "
            "serial-correlation-robust test.\n"
            "- **H₂ (the payout buys a surplus).** The fund's *total* return beats the one-click "
            "credit alternative: the TR spread vs HYG has a positive HAC $t$, and the NW-HAC "
            "$\\alpha \\ge 0$ in $r^{fund}_t - r^f_t = \\alpha + \\beta_{HYG}(r^{HYG}_t - r^f_t) + "
            "\\beta_{SPY}(r^{SPY}_t - r^f_t) + \\varepsilon_t$.\n"
            "- **H₃ (return ON capital).** The income is spendable without consuming capital: "
            "price-only CAGR $\\approx 0$.\n\n"
            f"We find **H₁ confirmed** (t ≈ 15.3 on both funds, payout 17-19%/yr), **H₂ rejected "
            f"as zero** (spread vs HYG t = +{sp['ECC'][2]:.2f} / +{sp['OXLC'][2]:.2f}; α t = "
            f"{rc['ECC'][2]:.2f} / {rc['OXLC'][2]:.2f}), **H₃ busted** (price legs "
            f"{d['ECC'][5]:.1f}% / {d['OXLC'][5]:.1f}%/yr — ~75-78% of the payout is capital)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "A CLO-equity CEF is double leverage on first-loss credit: the equity tranche is the "
            "residual of a ~10× levered loan pool (it absorbs the first defaults), and the fund adds "
            "its own preferred/note leverage plus management-and-incentive fees on *gross* assets — "
            "total expense ratios run high-single to low-double-digit percent of NAV in the funds' "
            "own reports. If H₂ held, retail would have cheap access to the institutional CLO-equity "
            "premium documented by Cordell-Roberts-Schwert (2023) — genuinely interesting. If only "
            "H₁ holds, the sector is the loan-securitization flavour of the desk's packaged-carry "
            "family: a real coupon financed by NAV erosion, with board-declared distributions "
            "(Rule 19a-1 return-of-capital notices are the accounting tell) doing the marketing. "
            "The decisive statistics are the **HAC t of the TR spread vs HYG** and the **NW-HAC "
            "alpha** — never the yield."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** yfinance daily, downloaded twice (`auto_adjust=True/False`); monthly "
            f"resample, partial months dropped; ECC window {R['start']} → {R['end']} "
            f"({R['n_months']} months), OXLC 2011-02 → (185). As-of {R['as_of']}, fingerprint "
            f"`{R['fingerprint']}`.\n"
            "- **Payout.** $c_t = r^{TR}_t - r^{PX}_t$; mean tested with Newey-West HAC t "
            "(6 lags; 3/12 as robustness) — payout streams are serially correlated.\n"
            "- **Headline.** TR spread vs HYG (both legs total return, same account, no timing), "
            "HAC t, each fund on its own full window.\n"
            "- **Benchmark.** NW-HAC OLS of excess returns on HYG + SPY excess returns "
            "(excess-on-excess, so α is a risk-adjusted spread). Full-sample betas — a "
            "risk-decomposition benchmarking choice, stated openly; nothing is a timed rule.\n"
            "- **Execution/costs.** Passive buy-and-hold decomposition: entry at the first "
            "month-end close (the one documented lag — there is no timing signal). Two one-way "
            "trades / 11.7 yrs at ~10 bps ≈ 1.7 bps/yr; the funds' 9-13%/yr fee+leverage drag is "
            "inside the tape.\n"
            "- **Crises.** Fixed documented windows (2015-16 crunch, Q4-2018, COVID, 2022); "
            "peak-to-trough TR drawdowns on the daily tape.\n"
            "- **Positive control.** Synthetic world with planted `carry` and `alpha` knobs; the "
            "null (0, 0) must stay silent."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The distribution decomposition — H₁\n\n"
            "Monthly distribution component per fund, with its HAC *t*; against it, the price-only "
            "CAGR it rode on."
        ),
        code(
            "names = ['ECC', 'OXLC']\n"
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for nm in names:\n"
            "        mtr, mpx = st.aligned_monthly(TR, PX, [nm, 'HYG', 'SPY', 'BIL'])\n"
            "        dd = st.decompose(mtr, mpx, nm)\n"
            "        rows.append((nm, dd['dist_bps_mo'], dd['dist_ann_pct'], dd['dist_hac_t'],\n"
            "                     dd['tr_cagr_pct'], dd['px_cagr_pct'], dd['n_months']))\n"
            "else:\n"
            "    rows = [(nm, R['decomp'][nm][1], R['decomp'][nm][2], R['decomp'][nm][3],\n"
            "             R['decomp'][nm][4], R['decomp'][nm][5], R['decomp'][nm][0]) for nm in names]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.4))\n"
            "a1.bar(names, [r[3] for r in rows], color=GREEN, width=.5)\n"
            "a1.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i, r in enumerate(rows): a1.annotate(f't={r[3]:.1f}', (i, r[3]), ha='center', va='bottom')\n"
            "a1.set_ylabel('HAC t of the distribution component')\n"
            "a1.set_title('H1: the payout is real (t = 15.3 on both)')\n"
            "a1.legend()\n"
            "a2.bar(np.arange(2) - .18, [r[2] for r in rows], .36, color=GREEN, label='distribution %/yr')\n"
            "a2.bar(np.arange(2) + .18, [r[5] for r in rows], .36, color=RED, label='price-only CAGR %/yr')\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_xticks(np.arange(2)); a2.set_xticklabels(names)\n"
            "a2.set_ylabel('% per year'); a2.set_title('...and the NAV pays for ~3/4 of it'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in rows:\n"
            "    print(f'{r[0]:>4}: dist {r[1]:+.2f} bps/mo = {r[2]:+.2f}%/yr (HAC t {r[3]:+.2f}) | '\n"
            "          f'TR {r[4]:+.2f}%/yr  PX {r[5]:+.2f}%/yr  (n={r[6]})')"
        ),
        md(
            f"> 💡 In plain words: the coupon is as real as statistics gets — ECC paid "
            f"**+{d['ECC'][2]:.2f}%/yr** at **t = {d['ECC'][3]:.1f}**, OXLC **+{d['OXLC'][2]:.2f}%/yr** "
            f"at **t = {d['OXLC'][3]:.1f}** — and both price legs eroded **−13 to −14%/yr**. The "
            f"machine pays its coupon mostly out of its own NAV: the financed-by-price share is "
            f"**{roc['ECC'][2]:.1f}%** (ECC) and **{roc['OXLC'][2]:.1f}%** (OXLC) — H₃'s answer, "
            "measured (the third-axis table in [`docs/results.md`](../docs/results.md))."
        ),
        md(
            "### 4b · The headline test — TR spread vs HYG, and the NW-HAC alpha (H₂)\n\n"
            "Left: the HAC *t* of each fund's monthly total-return spread over plain HYG (own full "
            "windows). Right: the beta-matched HYG+SPY copy of ECC, compounding past it."
        ),
        code(
            "names = ['ECC', 'OXLC']\n"
            "if HAVE_REAL:\n"
            "    srows, arows = [], []\n"
            "    for nm in names:\n"
            "        mtr, _ = st.aligned_monthly(TR, PX, [nm, 'HYG', 'SPY', 'BIL'])\n"
            "        s = st.tr_spread_vs(mtr, nm)\n"
            "        br = st.benchmark_race(mtr, nm)\n"
            "        srows.append((nm, s['spread_bps_mo'], s['spread_t']))\n"
            "        arows.append((nm, br['alpha_bps_mo'], br['t_alpha'], br['beta_HYG'], br['beta_SPY'],\n"
            "                      br['name_cagr_pct'], br['bench_cagr_pct']))\n"
            "    mtr_e, _ = st.aligned_monthly(TR, PX, ['ECC', 'HYG', 'SPY', 'BIL'])\n"
            "    br_e = st.benchmark_race(mtr_e, 'ECC')\n"
            "    w_e = (1 + mtr_e['ECC']).cumprod() * 100\n"
            "    w_b = (1 + br_e['bench_series']).cumprod() * 100\n"
            "else:\n"
            "    srows = [(nm, R['spread'][nm][0], R['spread'][nm][2]) for nm in names]\n"
            "    arows = [(nm, R['race'][nm][0], R['race'][nm][2], R['race'][nm][3], R['race'][nm][5],\n"
            "              R['race'][nm][8], R['race'][nm][9]) for nm in names]\n"
            "    w_e = w_b = None\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.5))\n"
            "a1.bar([r[0] for r in srows], [r[2] for r in srows], color=GREY, width=.5)\n"
            "a1.axhline(2, ls='--', c=RED, label='t = 2 bar'); a1.axhline(0, c='k', lw=.8)\n"
            "for i, r in enumerate(srows):\n"
            "    a1.annotate(f'{r[1]:+.0f} bps\\nt={r[2]:.2f}', (i, r[2]), ha='center', va='bottom', fontsize=9)\n"
            "a1.set_ylim(-0.5, 2.6)\n"
            "a1.set_ylabel('HAC t of TR spread vs HYG')\n"
            "a1.set_title('The 15-19%-payer surplus over 5%-yield HYG: statistically zero')\n"
            "a1.legend()\n"
            "if w_e is not None:\n"
            "    a2.plot(w_b, c=GREY, lw=1.8, label=f'beta-matched HYG+SPY copy  (${w_b.iloc[-1]:.0f})')\n"
            "    a2.plot(w_e, c=RED, lw=1.8, label=f'ECC total return  (${w_e.iloc[-1]:.0f})')\n"
            "    a2.set_yscale('log'); a2.legend(fontsize=9)\n"
            "    a2.set_title('The DIY copy of ECC\\'s own betas, growing past it')\n"
            "    a2.set_ylabel('growth of $100 (log)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in srows: print(f'{r[0]:>4}: spread vs HYG {r[1]:+.2f} bps/mo  HAC t = {r[2]:+.2f}')\n"
            "for r in arows:\n"
            "    print(f'{r[0]:>4}: alpha {r[1]:+7.2f} bps/mo (t {r[2]:+.2f})  b_HYG {r[3]:+.2f}  '\n"
            "          f'b_SPY {r[4]:+.2f}  | CAGR {r[5]:+.2f}%/yr vs bench {r[6]:+.2f}%/yr')"
        ),
        md(
            f"> 💡 In plain words: the headline test lands at **t = +{sp['ECC'][2]:.2f}** (ECC) and "
            f"**+{sp['OXLC'][2]:.2f}** (OXLC) — after a decade-plus, the 15-19% machines produced "
            "**no detectable surplus** over the boring credit ETF, while running 3-4× its risk "
            f"(and on compounding they finished *behind* it: +{sp['ECC'][4]:.2f}% vs "
            f"+{sp['ECC'][5]:.2f}%/yr for ECC). The factor regression says the same: α = "
            f"**{rc['ECC'][0]:.1f} / {rc['OXLC'][0]:.1f} bps/mo** (t = {rc['ECC'][2]:.2f} / "
            f"{rc['OXLC'][2]:.2f}) with R² of only ~0.25 — three quarters of the variance is wrapper "
            "noise (premium/discount swings, leverage resets) you are paid nothing to hold. Note the "
            "honest asymmetry: the deficit is *not* statistically significant either — these funds "
            "are so volatile that even −2 to −4%/yr can't clear a t-bar. Zero premium is the finding."
        ),
        md(
            "### 4c · Robustness — lags and subperiods\n\n"
            "The zero spread / zero alpha must not be a HAC-lag choice, and must survive dropping "
            "the 2015 entry and isolating the post-COVID 'golden age of CLO equity' regime."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for nm in ['ECC', 'OXLC']:\n"
            "        mtr, _ = st.aligned_monthly(TR, PX, [nm, 'HYG', 'SPY', 'BIL'])\n"
            "        for lags in (3, 6, 12):\n"
            "            s = st.tr_spread_vs(mtr, nm, lags=lags)\n"
            "            cp = st.carry_premium(mtr, nm, lags=lags)\n"
            "            rows.append((f'{nm} lags={lags}', s['spread_t'], cp['t_alpha']))\n"
            "else:\n"
            "    rows = [(f'{nm} lags={l}', ts, ta) for nm in ['ECC', 'OXLC']\n"
            "            for l, ts, ta in R['robust_lags'][nm]]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.2))\n"
            "x = np.arange(len(rows))\n"
            "ax.bar(x - .18, [r[1] for r in rows], .36, color=GREY, label='TR spread vs HYG (t)')\n"
            "ax.bar(x + .18, [r[2] for r in rows], .36, color=RED, label='alpha vs HYG+SPY (t)')\n"
            "ax.axhline(2, ls='--', c='k', lw=1, label='|t| = 2'); ax.axhline(-2, ls='--', c='k', lw=1)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=8, rotation=20)\n"
            "ax.set_ylim(-2.6, 2.6); ax.set_ylabel('HAC t'); ax.legend(fontsize=9)\n"
            "ax.set_title('Nothing clears the bar in any direction, at any lag')\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in rows: print(f'{r[0]:<14}: spread t = {r[1]:+.2f} | alpha t = {r[2]:+.2f}')\n"
            "print()\n"
            "if HAVE_REAL:\n"
            "    for lab, a in [('2016-01+', '2016-01-01'), ('2020-07+', '2020-07-01')]:\n"
            "        sub_tr, sub_px = MTR.loc[a:], MPX.loc[a:]\n"
            "        for nm in ['ECC', 'OXLC']:\n"
            "            dd = st.decompose(sub_tr, sub_px, nm)\n"
            "            s = st.tr_spread_vs(sub_tr, nm)\n"
            "            print(f'{lab} {nm:>4}: dist {dd[\"dist_ann_pct\"]:+.2f}%/yr (t {dd[\"dist_hac_t\"]:+.1f}) | '\n"
            "                  f'TR {dd[\"tr_cagr_pct\"]:+.2f}%  PX {dd[\"px_cagr_pct\"]:+.2f}%  | '\n"
            "                  f'spread vs HYG t = {s[\"spread_t\"]:+.2f} (n={s[\"n\"]})')\n"
            "else:\n"
            "    for lab, sub in R['sub'].items():\n"
            "        for nm, v in sub.items():\n"
            "            print(f'{lab} {nm:>4}: dist {v[0]:+.2f}%/yr (t {v[1]:+.1f}) | TR {v[2]:+.2f}%  '\n"
            "                  f'PX {v[3]:+.2f}%  | spread vs HYG t = {v[5]:+.2f} (n={v[6]})')"
        ),
        md(
            f"> 💡 In plain words: lags 3/6/12 move nothing (spread t stays at ~+0.1/+0.3, alpha t "
            f"at ~−0.3/−0.7). Ex the 2015 entry, same picture. And in the post-COVID floating-rate "
            f"regime — the sector's best-case marketing story — the payout hit ~20-21%/yr, TR "
            f"improved to +{R['sub']['2020-07+']['ECC'][2]:.1f}% / "
            f"+{R['sub']['2020-07+']['OXLC'][2]:.1f}%/yr… and the price legs *still* eroded "
            f"{R['sub']['2020-07+']['ECC'][3]:.1f}% / {R['sub']['2020-07+']['OXLC'][3]:.1f}%/yr with "
            "the HYG spread still at t ≈ 0.4. The erosion is chronic, not one bad regime."
        ),
        md(
            "### 4d · Crisis autopsies — first-loss means first to fall\n\n"
            "Peak-to-trough total-return drawdowns inside four fixed, documented windows. A genuine "
            "income sleeve should look like HYG's column. It looks like levered first-loss credit."
        ),
        code(
            "cols = ['ECC', 'OXLC', 'HYG', 'SPY']\n"
            "if HAVE_REAL:\n"
            "    ct = st.crisis_table(TR, cols)\n"
            "else:\n"
            "    ct = pd.DataFrame([c[1:] for c in R['crises']], columns=cols,\n"
            "                      index=[c[0] for c in R['crises']])\n"
            "print(ct.round(1).to_string())\n"
            "fig, ax = plt.subplots(figsize=(9.8, 4.6))\n"
            "x = np.arange(len(ct)); wdt = .19\n"
            "shades = [RED, '#e07b5f', AMBER, GREY]\n"
            "for k, (c, col) in enumerate(zip(cols, shades)):\n"
            "    ax.bar(x + (k - 1.5) * wdt, ct[c].values, wdt, color=col, label=c)\n"
            "ax.set_xticks(x); ax.set_xticklabels([i.replace(' 2', '\\n2') for i in ct.index], fontsize=9)\n"
            "ax.set_ylabel('drawdown (%)'); ax.legend(fontsize=9, ncol=4)\n"
            "ax.set_title('Every stress: 2-3x the drawdown of plain high yield')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "> 💡 In plain words: −66%/−74% in eight weeks in 2020 while HYG fell 22%; −57% for OXLC "
            "in the 2015-16 loan-spread blowout that cost HYG 13%. The equity tranche exists to "
            "absorb the pool's first losses so the AAAs never take one — in a repricing it is crossed "
            "out first, and the funds hold it with fund-level leverage on top. The coupon is crash "
            "insurance premium collected in reverse: it pays until the month you needed it not to."
        ),
        md(
            "### 4e · Faithful-engine control — we know the truth here\n\n"
            "Deterministic factor world with planted `carry` and `alpha` knobs (seed 614). The null "
            "must stay silent; planted values must light up."
        ),
        code(
            "res = []\n"
            "for carry, alpha in [(0.0, 0.0), (0.012, -0.015)]:\n"
            "    ttr, tpx = data.synthetic_world(carry=carry, alpha=alpha, seed=614)\n"
            "    smtr, smpx = st.aligned_monthly(ttr, tpx, ['CLOE', 'HYG', 'SPY', 'BIL'])\n"
            "    sd = st.decompose(smtr, smpx, 'CLOE')\n"
            "    scp = st.carry_premium(smtr, 'CLOE')\n"
            "    res.append((carry * 1e4, alpha * 1e4, sd['dist_bps_mo'], sd['dist_hac_t'],\n"
            "                scp['alpha_bps_mo'], scp['t_alpha']))\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))\n"
            "labels = [f'planted carry\\n{r[0]:.0f} bps' for r in res]\n"
            "a1.bar(labels, [r[3] for r in res], color=[GREY, GREEN], width=.5)\n"
            "a1.axhline(2, ls='--', c=RED); a1.set_title('payout detector: silent at 0, lights up at 120 bps')\n"
            "for i, r in enumerate(res): a1.annotate(f't={r[3]:.1f}', (i, r[3]), ha='center', va='bottom')\n"
            "labels2 = [f'planted alpha\\n{r[1]:.0f} bps' for r in res]\n"
            "a2.bar(labels2, [r[5] for r in res], color=[GREY, RED], width=.5)\n"
            "a2.axhline(-2, ls='--', c='k'); a2.set_title('alpha test: silent at 0, recovers -150 bps')\n"
            "for i, r in enumerate(res): a2.annotate(f't={r[5]:.2f}', (i, r[5]), ha='center',\n"
            "                                        va='top' if r[5] < 0 else 'bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in res:\n"
            "    print(f'planted carry={r[0]:+6.1f} alpha={r[1]:+6.1f} bps/mo -> '\n"
            "          f'dist {r[2]:+.2f} (t {r[3]:+.2f}) | alpha {r[4]:+.2f} (t {r[5]:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: under the (0, 0) null the detectors read t = "
            f"{R['syn'][0][3]:+.2f} and {R['syn'][0][5]:+.2f} — no manufactured significance; the "
            f"planted +120 bps payout and −150 bps alpha are recovered at t = {R['syn'][1][3]:.1f} "
            f"and {R['syn'][1][5]:.2f}, betas 1.45/0.41 vs planted 1.60/0.45. *(A machinery proof "
            "only — never cited in support of the real-tape stamps.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — the payout exists and exceeds the pitch: ECC's distribution "
            f"component is **+{d['ECC'][1]:.2f} bps/mo (+{d['ECC'][2]:.2f}%/yr) at HAC "
            f"t = +{d['ECC'][3]:.2f}**, OXLC's **+{d['OXLC'][2]:.2f}%/yr at t = +{d['OXLC'][3]:.2f}** "
            f"— and the erosion financing it is on the same tape (price legs {d['ECC'][5]:.2f}% / "
            f"{d['OXLC'][5]:.2f}%/yr). What fails is the surplus: spread vs HYG t = "
            f"+{sp['ECC'][2]:.2f} / +{sp['OXLC'][2]:.2f}. Survivorship named: no index fund exists; "
            "the category is its two survivors.\n"
            f"- **Tradability `MIRAGE`** — 11.7-15.4 years of harvest left TR CAGR at "
            f"**+{d['ECC'][4]:.2f}% / +{d['OXLC'][4]:.2f}%/yr** (excess over bills t = "
            f"+{R['excess']['ECC'][1]:.2f} / +{R['excess']['OXLC'][1]:.2f}: cash, statistically) at "
            f"25-32% vol and **{R['perf']['ECC'][3]:.0f}% / {R['perf']['OXLC'][3]:.0f}%** drawdowns, "
            f"with no alpha vs a passive HYG+SPY mix whose copy compounded to "
            f"+{rc['ECC'][9]:.2f}% / +{rc['OXLC'][9]:.2f}%/yr at half the drawdown. Trading costs "
            "are irrelevant (~1.7 bps/yr); the 9-13%/yr wrapper drag is the mechanism.\n"
            f"- **Return OF capital? `CONFIRMED`** — **{roc['ECC'][2]:.1f}%** (ECC) and "
            f"**{roc['OXLC'][2]:.1f}%** (OXLC) of the distribution stream was offset one-for-one by "
            f"price erosion; $100 of capital → **${d['ECC'][7]:.1f} / ${d['OXLC'][7]:.1f}**. The "
            "market-price mirror of the funds' own 19(a) notices."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The packaged-carry family.** [340 — Bank-Loans](../../340-bank-loans/README.md), "
            "[341 — MLP-Pipelines](../../341-mlp-pipelines/README.md), "
            "[342 — BDC-Yield](../../342-bdc-yield/README.md) and "
            "[611 — mREIT-Carry](../../611-mreit-carry/README.md) are the named siblings — same "
            "Real × Mirage pathology. This is the family's limit case: the coupon is the residual "
            "of a **first-loss tranche** (not a margin, not a toll), double-levered by the wrapper, "
            "with the biggest measured yield (17-19%/yr) and the starkest capital-consumption share "
            "(~75-78%).\n"
            "- **What would change the verdict.** A NAV-based tape (the funds publish monthly NAV "
            "estimates) would split the price erosion into NAV erosion vs discount widening; "
            "institutional CLO-equity *held to maturity* (Cordell-Roberts-Schwert 2023) shows "
            "positive but dispersed economics — the gap between that and these tapes is the wrapper "
            "cost, and a cheap CLO-equity vehicle would be a different study.\n"
            "- **The accounting cross-check.** Rule 19a-1 notices per distribution (both funds' "
            "sites) flag the return-of-capital share directly; our returns-arithmetic split is the "
            "market-price version and lands in the same place.\n\n"
            "*The reproducible core is offline and deterministic; numbers frozen in "
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
