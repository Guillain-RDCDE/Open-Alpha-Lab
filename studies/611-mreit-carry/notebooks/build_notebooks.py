"""Generate the two narrative notebooks for Study 611 (mREIT Carry).

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
# REM/NLY 2007-06-30 -> 2026-06-30 (229 months), AGNC 2008-06 -> (217), as-of 2026-06-30).
R = dict(
    start="2007-06-30", end="2026-06-30", n_months=229, years=19.1, as_of="2026-06-30",
    fingerprint="3b2ec2fc7dd3",
    # per-name: (n_months, div bps/mo, div %/yr, div HAC t, TR CAGR %, PX CAGR %, $100 TR, $100 PX)
    decomp={
        "REM":  (229,  84.69, 10.65, 18.09,  -1.13, -10.96,  80.5, 10.9),
        "NLY":  (229, 103.04, 13.09, 23.01,   7.52,  -5.18, 399.0, 36.2),
        "AGNC": (217, 121.46, 15.59, 17.80,  11.80,  -3.22, 751.4, 55.3),
    },
    # perf on the REM window: (CAGR %, vol %, Sharpe excess, maxDD %)
    perf={"REM": (-1.13, 24.17, 0.04, -67.44), "IEF": (3.32, 6.70, 0.32, -23.15),
          "SPY": (10.68, 15.58, 0.64, -50.78), "BIL": (1.36, 0.56, None, -0.42)},
    rem_excess_bps=7.78, rem_excess_t=0.18,
    # race: (alpha bps/mo, alpha %/yr, t_alpha, b_IEF, t_IEF, b_SPY, t_SPY, R2,
    #        name CAGR %, bench CAGR %, name DD %, bench DD %, t_spread)
    race={
        "REM":  (-87.62, -10.02, -2.50, 0.42, 1.91, 1.05, 6.80, 0.46, -1.13, 12.07, -67.4, -49.7, -3.01),
        "NLY":  ( -8.11,  -0.97, -0.22, 1.05, 4.06, 0.78, 4.34, 0.31,  7.52, 11.03, -45.8, -33.4, -0.27),
        "AGNC": ( 18.83,   2.28,  0.43, 1.20, 4.41, 0.82, 5.03, 0.32, 11.80, 12.03, -48.1, -36.0,  0.51),
    },
    robust_lags=[(3, -87.62, -2.32), (6, -87.62, -2.50), (12, -87.62, -2.78)],
    exgfc=dict(alpha_bps=-76.48, t=-1.93, div_ann=10.33, div_t=17.0, tr_cagr=4.28, px_cagr=-5.78),
    # crises: (label, REM, NLY, AGNC, SPY, IEF) peak-to-trough TR drawdown %
    crises=[("GFC 2007-09",        -74.7, -40.9, -31.0, -55.2,  -6.2),
            ("Taper tantrum 2013", -24.8, -34.6, -36.9,  -5.6,  -9.1),
            ("COVID 2020",         -68.5, -60.1, -51.8, -33.7,  -4.7),
            ("Rate shock 2022",    -40.3, -47.8, -48.2, -24.5, -17.4)],
    # synthetic: (planted carry bps, planted alpha bps, div bps, div t, alpha bps, alpha t)
    syn=[(0.0, 0.0, 1.49, 0.71, -6.45, -0.35),
         (100.0, -70.0, 101.49, 48.21, -76.45, -4.15)],
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Free_lunch%3F: Busted](https://img.shields.io/badge/Free_lunch%3F-Busted-8b949e?style=flat-square)\n\n"
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

from mreit_carry import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    TR, PX = data.load_real()
    MTR, MPX = st.aligned_monthly(TR, PX, ["REM", "IEF", "SPY", "BIL"])
else:
    TR = PX = MTR = MPX = None
print("real mREIT cache present:", HAVE_REAL,
      "| months:", (0 if MTR is None else len(MTR)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    d = R["decomp"]
    cells = [
        md(
            "# The 10-14% mortgage-REIT dividend — real income, or your own money handed back? 🏚️\n"
            "### The mREIT carry machine, taken apart in plain English\n\n"
            + BADGES +
            "Open any income-investing screen and there they are: mortgage REITs, yielding **10, 12, "
            "14 percent**. Annaly. AGNC. The sector ETF, REM. The pitch is seductive: these companies "
            "borrow cheap, buy government-backed mortgage bonds, and pass the levered interest spread "
            "to you as a fat dividend. *\"Yes, the price bounces around — but the income is real.\"*\n\n"
            "Here's the strange thing: **both halves of that sentence are true.** The income *is* real — "
            "we measure it below at 10-16% a year, and it's one of the most statistically solid numbers "
            "on this desk. And yet holding the sector fund for nineteen years turned $100 into **$80.50**. "
            "This notebook is about how both things can be true at once.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the benchmark regression and the "
            "robustness grid? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> 🔬 **For the quants:** the headline leg is **REM**, an index *fund* — every mREIT blow-up "
            "is inside its tape, so the sector number is not survivor-biased. The single names (NLY, "
            "AGNC) are the two biggest *survivors* and are quoted as colour. Every chart is drawn by the "
            "code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Is the 10-14% dividend real? | **Yes.** The sector fund actually paid out "
            f"**{d['REM'][2]:.1f}% a year** in dividends for 19 straight years — measured, not "
            "advertised. The flagships paid 13-16%. This part of the pitch is honest. |\n"
            f"| Did the income make you rich? | **No.** $100 in the sector fund in 2007, every dividend "
            f"reinvested, became **${d['REM'][6]:.0f}** by 2026 — *less than T-bills* — because the "
            f"share price itself eroded **{d['REM'][5]:.0f}% a year**. |\n"
            "| So where did the coupon come from? | Largely **out of the price**. The fund handed you "
            f"~10.65%/yr with one hand while the NAV leaked ~11%/yr out the other. $100 of *price* "
            f"became **${d['REM'][7]:.0f}**. |\n"
            "| Was it at least safe on the way? | **The opposite.** It crashed **harder than the stock "
            "market** in 2008 (−75%), 2013, 2020 (−68% in eight weeks) *and* 2022. |\n\n"
            "> The mortgage-REIT coupon is a genuine *carry trade* — real cash, harvested monthly — "
            "strapped to a vehicle that consumes principal and blows up exactly when everything else "
            "does. Real carry, mirage vehicle."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Mortgage REITs pay 10-14%. They borrow at short rates, buy government-guaranteed "
            "mortgage bonds, lever the spread 6-8×, and are legally required to pay ~90% of the income "
            "out to you. The price is volatile, sure — but the dividend stream is real money, and if "
            "you harvest it, the carry compounds.\"*\n\n"
            "This is the backbone of a thousand \"retire on dividends\" model portfolios. And unlike "
            "most desk folklore, the *mechanism* is real: an agency mREIT genuinely is a levered "
            "bond-carry book, and the coupon genuinely lands in your brokerage account. The question "
            "is what happens to the *rest* of your money while it does."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the pitch were right, a retiree could replace bonds with mREITs and quadruple the "
            "income. If it's wrong the way we suspect, the 10-14% is partly **your own capital handed "
            "back with a yield label on it** — and the \"bond substitute\" carries *stock-crash* risk "
            "concentrated in the worst weeks of the century. This is the third asset the desk has "
            "caught in the same costume: pipelines ([Study 341](../../341-mlp-pipelines/README.md)) "
            "and private-credit BDCs ([Study 342](../../342-bdc-yield/README.md)) ran the same play. "
            "The mortgage flavour is the purest, because here the income really *is* a carry trade — "
            "which means it's short liquidity, and liquidity is what vanishes in a crisis."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "One trick does all the work: we download every price series **twice** — once including "
            "reinvested dividends (**total return**: what you actually got) and once as the raw share "
            "price (**price-only**: what happened to your capital). Subtract them month by month and "
            "the difference is exactly the **dividend stream** — the carry the pitch sells.\n\n"
            "Then three questions:\n"
            "1. **Is the carry real?** Average the dividend component; test it hard.\n"
            "2. **What did it cost?** Look at what the price leg did while the coupon flowed.\n"
            "3. **Could you have done better with a bare-hands copy?** Compare the fund to a simple "
            "mix of Treasury bonds + stocks with the *same* measured risk (the quants notebook runs "
            "that regression). If the packaged carry loses to the DIY version, the package is the "
            "problem."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the whole story in one picture:** $100 in the sector fund, 2007→2026 — with "
            "dividends reinvested (what you got) vs the share price alone (what happened to your "
            "capital), against boring stocks and boring Treasuries."
        ),
        code(
            "if HAVE_REAL:\n"
            "    w = {c: (1.0 + MTR[c]).cumprod() * 100 for c in ['REM','SPY','IEF']}\n"
            "    w_px = (1.0 + MPX['REM']).cumprod() * 100\n"
            "    fig, ax = plt.subplots(figsize=(9.8, 5.2))\n"
            "    ax.plot(w['SPY'], c=GREY, lw=1.6, label=f\"SPY total return  (${w['SPY'].iloc[-1]:.0f})\")\n"
            "    ax.plot(w['IEF'], c=AMBER, lw=1.6, label=f\"IEF 7-10y Treasuries  (${w['IEF'].iloc[-1]:.0f})\")\n"
            "    ax.plot(w['REM'], c=GREEN, lw=2.0, label=f\"REM total return — dividends reinvested  (${w['REM'].iloc[-1]:.0f})\")\n"
            "    ax.plot(w_px, c=RED, lw=2.0, label=f\"REM price only — your capital  (${w_px.iloc[-1]:.0f})\")\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $100 (log scale)')\n"
            "    ax.set_title('Nineteen years of harvesting a 10.65%/yr coupon: $100 -> $80.50')\n"
            "    ax.legend(loc='lower left', fontsize=9)\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f\"$100 -> REM TR ${w['REM'].iloc[-1]:.1f} | REM price-only ${w_px.iloc[-1]:.1f} | \"\n"
            "          f\"SPY ${w['SPY'].iloc[-1]:.1f} | IEF ${w['IEF'].iloc[-1]:.1f}\")\n"
            "else:\n"
            "    print('cache missing - frozen numbers:', '$100 -> REM TR $%.1f, price-only $%.1f'\n"
            "          % (R['decomp']['REM'][6], R['decomp']['REM'][7]))"
        ),
        md(
            f"The green line is the *best case* — every dividend reinvested, 19 years of 10.65%/yr "
            f"coupons — and it ends at **${d['REM'][6]:.0f}**, below T-bills. The red line is your "
            f"actual capital: **${d['REM'][7]:.0f}** left of $100. The coupon was real; it was also, "
            "to a first approximation, **your own principal coming back to you** on a levered, "
            "crash-prone conveyor belt."
        ),
        md(
            "**Now the decomposition per name.** Green = the dividend stream (real, harvested); red = "
            "what the share price did; amber = the total you actually kept."
        ),
        code(
            "names = ['REM', 'NLY', 'AGNC']\n"
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for nm in names:\n"
            "        mtr, mpx = st.aligned_monthly(TR, PX, [nm, 'IEF', 'SPY', 'BIL'])\n"
            "        dd = st.decompose(mtr, mpx, nm)\n"
            "        rows.append((dd['div_ann_pct'], dd['px_cagr_pct'], dd['tr_cagr_pct']))\n"
            "else:\n"
            "    rows = [(R['decomp'][nm][2], R['decomp'][nm][5], R['decomp'][nm][4]) for nm in names]\n"
            "x = np.arange(len(names)); wdt = 0.27\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.8))\n"
            "ax.bar(x - wdt, [r[0] for r in rows], wdt, color=GREEN, label='dividend stream (%/yr) — the carry')\n"
            "ax.bar(x,       [r[1] for r in rows], wdt, color=RED, label='price-only CAGR (%/yr) — the erosion')\n"
            "ax.bar(x + wdt, [r[2] for r in rows], wdt, color=AMBER, label='total-return CAGR (%/yr) — what you kept')\n"
            "for i, r in enumerate(rows):\n"
            "    ax.annotate(f'{r[0]:+.1f}', (i - wdt, r[0]), ha='center', va='bottom', fontsize=9)\n"
            "    ax.annotate(f'{r[1]:+.1f}', (i, r[1]), ha='center', va='top', fontsize=9)\n"
            "    ax.annotate(f'{r[2]:+.1f}', (i + wdt, r[2]), ha='center', va='bottom', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['REM (the sector)', 'NLY', 'AGNC'])\n"
            "ax.set_ylabel('% per year'); ax.legend(fontsize=9)\n"
            "ax.set_title('The coupon is real - and the price leg pays for it')\n"
            "plt.tight_layout(); plt.show()\n"
            "for nm, r in zip(names, rows):\n"
            "    print(f'{nm}: dividend {r[0]:+.2f}%/yr  price {r[1]:+.2f}%/yr  total {r[2]:+.2f}%/yr')"
        ),
        md(
            f"Every name pays a double-digit real coupon (**+{d['REM'][2]:.1f}%** to "
            f"**+{d['AGNC'][2]:.1f}%** a year) and every name's *price* erodes (−3 to −11%/yr). The "
            "sector fund — the only number free of survivor luck — nets out **negative**. The two "
            "survivors netted positive precisely because they're the two that survived; the graveyard "
            "(Thornburg, the 2020 margin-call cohort…) is inside REM's red bar.\n\n"
            "**And the \"real drawdowns\" half of the pitch?** Understated, if anything. Here is the "
            "\"income sleeve\" in the four storms of its life, next to the stock market:"
        ),
        code(
            "labels = [c[0] for c in R['crises']]\n"
            "if HAVE_REAL:\n"
            "    ct = st.crisis_table(TR, ['REM', 'SPY', 'IEF'])\n"
            "    rem, spy, ief = ct['REM'].tolist(), ct['SPY'].tolist(), ct['IEF'].tolist()\n"
            "else:\n"
            "    rem = [c[1] for c in R['crises']]; spy = [c[4] for c in R['crises']]; ief = [c[5] for c in R['crises']]\n"
            "x = np.arange(len(labels)); wdt = .27\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.8))\n"
            "ax.bar(x - wdt, rem, wdt, color=RED, label='REM (the \"income sleeve\")')\n"
            "ax.bar(x,       spy, wdt, color=GREY, label='SPY (stocks)')\n"
            "ax.bar(x + wdt, ief, wdt, color=AMBER, label='IEF (Treasuries)')\n"
            "for i in range(len(labels)):\n"
            "    ax.annotate(f'{rem[i]:.0f}%', (i - wdt, rem[i]), ha='center', va='top', fontsize=9)\n"
            "ax.set_xticks(x); ax.set_xticklabels([l.replace(' 2', '\\n2') for l in labels], fontsize=9)\n"
            "ax.set_ylabel('peak-to-trough drawdown (%)')\n"
            "ax.set_title('Four storms: the 12% \"bond substitute\" fell harder than stocks every time')\n"
            "ax.legend(fontsize=9); plt.tight_layout(); plt.show()\n"
            "print('REM drawdowns:', [f'{v:.1f}%' for v in rem])"
        ),
        md(
            "**−74.7%** in the GFC (worse than banks' index-level fall), **−68.5% in eight weeks** in "
            "March 2020, **−40%** in 2022 — and in 2013 it managed −25% while the stock market barely "
            "noticed. Why always? Because the carry is funded with overnight repo: when markets panic, "
            "the *lenders* call the loans, and the fund sells its bonds into the crash. The dividend "
            "machine is short exactly the thing that disappears in a crisis — liquidity."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** The carry exists, spectacularly: **{d['REM'][2]:.1f}%/yr** of hard "
            "dividend cash on the sector fund for 19 years (statistically one of the most solid numbers "
            "on this desk), 13-16% on the flagships. The pitch's numbers are honest.\n"
            f"- **Tradability — Mirage.** Harvesting that coupon left you with **${d['REM'][6]:.0f}** "
            f"per $100 — below T-bills — at 24% volatility and a −67% drawdown, and ~10%/yr behind a "
            "simple DIY Treasuries+stocks mix with the same measured risk. A worse bond, a worse stock, "
            "a worse carry trade.\n"
            f"- **\"10-14% free lunch\"? — Busted.** The coupon was financed by the price leg "
            f"(**{d['REM'][5]:.0f}%/yr**; $100 of capital → **${d['REM'][7]:.0f}**). Spend the dividend "
            "and you are eating your principal — with a margin call attached."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The family pattern.** Pipelines ([341](../../341-mlp-pipelines/README.md)), BDCs "
            "([342](../../342-bdc-yield/README.md)), and now mortgage REITs: package a levered risk "
            "premium, label the payout \"income,\" and let the NAV quietly pay for it. When you see a "
            "double-digit yield, the first question is never *\"is it real?\"* — it usually is — but "
            "*\"what is it liquidating to pay me?\"*\n"
            "- **The survivor illusion.** NLY and AGNC netting positive is exactly what a graveyard "
            "looks like from above: judge sectors by their index fund, not by the names still on the "
            "screen.\n"
            "- **The honest version of the trade.** If you want levered bond carry, the quants notebook "
            "shows what a duration-matched Treasuries position actually earned — more, with a third of "
            "the drawdown. The carry wasn't the mistake; the *wrapper* was.\n\n"
            "*Think the coupon can outrun the erosion — maybe post-2022, with fatter MBS spreads? The "
            "engine is in [`mreit_carry/`](../mreit_carry/); re-run `examples/verify.py` on any window "
            "and watch the alpha, not the yield.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    d = R["decomp"]
    rc = R["race"]
    cells = [
        md(
            "# mREIT Carry — a quantitative teardown 🔬\n"
            "### Total-return vs price-only carry decomposition (HAC *t*) · NW-HAC alpha vs a "
            "duration-matched levered-IEF benchmark · lag & subperiod robustness · crisis autopsies · "
            "a synthetic faithful-engine control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The claim "
            "— *\"mREITs pay 10-14% of real leveraged-MBS carry\"* — is unusual for this desk in that "
            "its central number is **true**; the study's job is to measure what the carry costs and "
            "what it loses to its own passive benchmark.\n\n"
            "> ⚠️ **Data + survivorship note.** Double yfinance tape (auto-adjusted total-return AND "
            "raw split-adjusted price-only) for REM / NLY / AGNC / IEF / SPY / BIL, REM window "
            f"{R['start']} → {R['end']} ({R['n_months']} months, GFC included), as-of {R['as_of']}. "
            "**REM is an index fund — the headline leg is not survivor-biased**; NLY/AGNC are the two "
            "biggest survivors, quoted as colour. Offline core + synthetic control are deterministic. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fingerprint"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` | Dividend component **+{d['REM'][1]:.2f} bps/mo "
            f"(+{d['REM'][2]:.2f}%/yr)** on REM at **HAC t = +{d['REM'][3]:.2f}** (NLY t = "
            f"{d['NLY'][3]:.2f}, AGNC t = {d['AGNC'][3]:.2f}); the financing erosion is equally real "
            f"— carry premium **{rc['REM'][0]:.2f} bps/mo at HAC t = {rc['REM'][2]:.2f}**. |\n"
            f"| **Tradability** | `MIRAGE` | TR CAGR **{d['REM'][4]:.2f}%/yr** (excess over bills "
            f"t = +{R['rem_excess_t']:.2f} — cash), maxDD **{R['perf']['REM'][3]:.1f}%**, and "
            f"**{rc['REM'][1]:.2f}%/yr (t = {rc['REM'][2]:.2f})** behind the beta-matched levered "
            "IEF+SPY benchmark. |\n"
            f"| **Free lunch?** | `BUSTED` | Price-only CAGR **{d['REM'][5]:.2f}%/yr** — $100 of "
            f"capital → **${d['REM'][7]:.1f}**. The coupon is NAV liquidation plus crash risk "
            "(β_SPY = 1.05). |\n\n"
            "> 💡 In plain words: the coupon is real and huge; the vehicle converts it into less than "
            "T-bills at equity-crash risk. Real carry, mirage package — the third sibling of "
            "[341](../../341-mlp-pipelines/README.md) and [342](../../342-bdc-yield/README.md)."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{TR}_{t}$ and $r^{PX}_{t}$ be a name's monthly total-return and price-only "
            "returns. The **dividend component** (the carry the pitch sells) is\n\n"
            "$$c_t = r^{TR}_t - r^{PX}_t \\ge 0,$$\n\n"
            "and the claim decomposes into:\n\n"
            "- **H₁ (the carry exists).** $\\bar{c} \\approx$ 10-14%/yr, significant under a "
            "serial-correlation-robust test.\n"
            "- **H₂ (the carry is kept).** Total return compounds it: TR CAGR is meaningfully "
            "positive and the vehicle beats a passive duration-matched mix — i.e. NW-HAC $\\alpha "
            "\\ge 0$ in $r^{mREIT}_t - r^f_t = \\alpha + \\beta_{IEF}(r^{IEF}_t - r^f_t) + "
            "\\beta_{SPY}(r^{SPY}_t - r^f_t) + \\varepsilon_t$.\n"
            "- **H₃ (free lunch).** The income is spendable without consuming capital: price-only "
            "CAGR $\\approx 0$.\n\n"
            f"We find **H₁ confirmed** (t = 17.8-23.0), **H₂ rejected** (REM α = {rc['REM'][0]:.1f} "
            f"bps/mo at t = {rc['REM'][2]:.2f}; TR below bills), **H₃ busted** (price leg "
            f"{d['REM'][5]:.1f}%/yr)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "An agency mREIT is a levered repo carry book: long MBS duration and (negative) "
            "convexity, short overnight liquidity, distributing the levered net interest margin. If "
            "H₂ held, the package would deliver a retail-accessible carry premium unavailable in "
            "plain ETFs — genuinely interesting. If only H₁ holds, the sector is the mortgage flavour "
            "of the desk's packaged-carry family (MLPs, BDCs): a real coupon financed by NAV erosion, "
            "with the true factor exposure hiding under the income label. The decisive statistic is "
            "the **NW-HAC alpha against the investor's realistic alternative** — levered IEF plus "
            "equity beta, financed at bills — not the yield, and not raw CAGR."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** yfinance daily, downloaded twice (`auto_adjust=True/False`); monthly "
            f"resample, partial months dropped; REM/NLY window {R['start']} → {R['end']} "
            f"({R['n_months']} months), AGNC from 2008-06 (217). As-of {R['as_of']}, fingerprint "
            f"`{R['fingerprint']}`.\n"
            "- **Carry.** $c_t = r^{TR}_t - r^{PX}_t$; mean tested with Newey-West HAC t "
            "(6 lags; 3/12 as robustness) — payout streams are serially correlated.\n"
            "- **Benchmark.** NW-HAC OLS of excess returns on IEF + SPY excess returns "
            "(excess-on-excess, so α is a risk-adjusted spread). Full-sample betas — a "
            "risk-decomposition benchmarking choice, stated openly; nothing is a timed rule.\n"
            "- **Execution/costs.** Passive buy-and-hold decomposition: entry at the first month-end "
            "close (the one documented lag — there is no timing signal). Two one-way trades / 19.1 "
            "yrs ≈ 0.5 bps/yr; REM's 0.48% ER is inside the tape.\n"
            "- **Crises.** Fixed documented windows (GFC, taper 2013, COVID, 2022); peak-to-trough TR "
            "drawdowns on the daily tape.\n"
            "- **Positive control.** Synthetic world with planted `carry` and `alpha` knobs; the null "
            "(0, 0) must stay silent."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The carry decomposition — H₁\n\n"
            "Monthly dividend component per name, with its HAC *t*; against it, the price-only CAGR "
            "it rode on."
        ),
        code(
            "names = ['REM', 'NLY', 'AGNC']\n"
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for nm in names:\n"
            "        mtr, mpx = st.aligned_monthly(TR, PX, [nm, 'IEF', 'SPY', 'BIL'])\n"
            "        dd = st.decompose(mtr, mpx, nm)\n"
            "        rows.append((nm, dd['div_bps_mo'], dd['div_ann_pct'], dd['div_hac_t'],\n"
            "                     dd['tr_cagr_pct'], dd['px_cagr_pct'], dd['n_months']))\n"
            "else:\n"
            "    rows = [(nm, R['decomp'][nm][1], R['decomp'][nm][2], R['decomp'][nm][3],\n"
            "             R['decomp'][nm][4], R['decomp'][nm][5], R['decomp'][nm][0]) for nm in names]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.4))\n"
            "a1.bar(names, [r[3] for r in rows], color=GREEN, width=.55)\n"
            "a1.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i, r in enumerate(rows): a1.annotate(f't={r[3]:.1f}', (i, r[3]), ha='center', va='bottom')\n"
            "a1.set_ylabel('HAC t of the dividend component'); a1.set_title('H1: the carry is real (t = 18-23)')\n"
            "a1.legend()\n"
            "a2.bar(np.arange(3) - .18, [r[2] for r in rows], .36, color=GREEN, label='dividend %/yr')\n"
            "a2.bar(np.arange(3) + .18, [r[5] for r in rows], .36, color=RED, label='price-only CAGR %/yr')\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_xticks(np.arange(3)); a2.set_xticklabels(names)\n"
            "a2.set_ylabel('% per year'); a2.set_title('...and the NAV pays for it'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in rows:\n"
            "    print(f'{r[0]:>4}: div {r[1]:+.2f} bps/mo = {r[2]:+.2f}%/yr (HAC t {r[3]:+.2f}) | '\n"
            "          f'TR {r[4]:+.2f}%/yr  PX {r[5]:+.2f}%/yr  (n={r[6]})')"
        ),
        md(
            f"> 💡 In plain words: the coupon is as real as statistics gets — REM paid "
            f"**+{d['REM'][2]:.2f}%/yr** of hard dividend cash at **t = {d['REM'][3]:.1f}** — and "
            f"every name's price leg is negative. The sector fund's price fell "
            f"**{d['REM'][5]:.2f}%/yr**: the machine pays its coupon substantially out of its own NAV."
        ),
        md(
            "### 4b · The decisive test — NW-HAC alpha vs the duration-matched benchmark (H₂)\n\n"
            "Each name's monthly excess return regressed on IEF and SPY excess returns (NW, 6 lags). "
            "The benchmark portfolio (cash + β·factors) is what a DIY investor could hold instead."
        ),
        code(
            "names = ['REM', 'NLY', 'AGNC']\n"
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for nm in names:\n"
            "        mtr, _ = st.aligned_monthly(TR, PX, [nm, 'IEF', 'SPY', 'BIL'])\n"
            "        br = st.benchmark_race(mtr, nm)\n"
            "        rows.append((nm, br['alpha_bps_mo'], br['t_alpha'], br['beta_IEF'], br['beta_SPY'],\n"
            "                     br['name_cagr_pct'], br['bench_cagr_pct']))\n"
            "    mtr_rem, _ = st.aligned_monthly(TR, PX, ['REM', 'IEF', 'SPY', 'BIL'])\n"
            "    br_rem = st.benchmark_race(mtr_rem, 'REM')\n"
            "    w_rem = (1 + mtr_rem['REM']).cumprod() * 100\n"
            "    w_ben = (1 + br_rem['bench_series']).cumprod() * 100\n"
            "else:\n"
            "    rows = [(nm, R['race'][nm][0], R['race'][nm][2], R['race'][nm][3], R['race'][nm][5],\n"
            "             R['race'][nm][8], R['race'][nm][9]) for nm in names]\n"
            "    w_rem = w_ben = None\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.5))\n"
            "cols = [RED if r[1] < 0 else GREY for r in rows]\n"
            "a1.bar(names, [r[1] for r in rows], color=cols, width=.55)\n"
            "a1.axhline(0, c='k', lw=.8)\n"
            "for i, r in enumerate(rows):\n"
            "    a1.annotate(f'{r[1]:+.0f} bps\\nt={r[2]:.2f}', (i, r[1]), ha='center',\n"
            "                va='top' if r[1] < 0 else 'bottom', fontsize=9)\n"
            "a1.set_ylabel('NW-HAC alpha (bps/month)')\n"
            "a1.set_title('Carry premium vs levered IEF+SPY: none - REM significantly negative')\n"
            "if w_rem is not None:\n"
            "    a2.plot(w_ben, c=GREY, lw=1.8, label=f'beta-matched benchmark  (${w_ben.iloc[-1]:.0f})')\n"
            "    a2.plot(w_rem, c=RED, lw=1.8, label=f'REM total return  (${w_rem.iloc[-1]:.0f})')\n"
            "    a2.set_yscale('log'); a2.legend(fontsize=9)\n"
            "    a2.set_title('The DIY copy of REM\\'s own betas, growing past it')\n"
            "    a2.set_ylabel('growth of $100 (log)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in rows:\n"
            "    print(f'{r[0]:>4}: alpha {r[1]:+7.2f} bps/mo (t {r[2]:+.2f})  b_IEF {r[3]:+.2f}  '\n"
            "          f'b_SPY {r[4]:+.2f}  | CAGR {r[5]:+.2f}%/yr vs bench {r[6]:+.2f}%/yr')"
        ),
        md(
            f"> 💡 In plain words: two punchlines. **The costume comes off** — REM's equity beta is "
            f"**{rc['REM'][5]:.2f}** (t = {rc['REM'][6]:.1f}) against a duration beta of only "
            f"{rc['REM'][3]:.2f}: the \"bond-like\" vehicle is stock risk. And **the package destroys "
            f"value**: α = **{rc['REM'][0]:.1f} bps/mo ({rc['REM'][1]:.1f}%/yr) at HAC "
            f"t = {rc['REM'][2]:.2f}** — the passive copy of REM's own betas turned $100 into ~$880 "
            f"while REM made $80. Even the surviving flagships add nothing (t = {rc['NLY'][2]:.2f}, "
            f"{rc['AGNC'][2]:+.2f}) while carrying 12-14 pp deeper drawdowns."
        ),
        md(
            "### 4c · Robustness — lags and the ex-GFC subperiod\n\n"
            "The negative alpha must not be a HAC-lag choice or a single-crisis artefact."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for lags in (3, 6, 12):\n"
            "        cp = st.carry_premium(MTR, 'REM', lags=lags)\n"
            "        rows.append((f'NW lags={lags}', cp['alpha_bps_mo'], cp['t_alpha']))\n"
            "    cp = st.carry_premium(MTR.loc['2010-01-01':], 'REM')\n"
            "    rows.append(('ex-GFC (2010+)', cp['alpha_bps_mo'], cp['t_alpha']))\n"
            "else:\n"
            "    rows = [(f'NW lags={l}', a, t) for l, a, t in R['robust_lags']]\n"
            "    rows.append(('ex-GFC (2010+)', R['exgfc']['alpha_bps'], R['exgfc']['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.2))\n"
            "ax.bar([r[0] for r in rows], [r[2] for r in rows], color=RED, width=.55)\n"
            "ax.axhline(-2, ls='--', c='k', lw=1, label='t = -2')\n"
            "for i, r in enumerate(rows): ax.annotate(f't={r[2]:.2f}', (i, r[2]), ha='center', va='top')\n"
            "ax.set_ylabel('HAC t of REM alpha'); ax.legend()\n"
            "ax.set_title('The negative carry premium survives lag choices; ex-GFC it is -76 bps/mo (t = -1.9)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in rows: print(f'{r[0]:<16}: alpha {r[1]:+.2f} bps/mo  t = {r[2]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: lags 3/6/12 give t = {R['robust_lags'][0][2]:.2f} / "
            f"{R['robust_lags'][1][2]:.2f} / {R['robust_lags'][2][2]:.2f}. Dropping the GFC entirely "
            f"still leaves α = {R['exgfc']['alpha_bps']:.1f} bps/mo (t = {R['exgfc']['t']:.2f}, "
            f"marginal on the shorter tape) with the dividend intact (+{R['exgfc']['div_ann']:.1f}%/yr, "
            f"t = {R['exgfc']['div_t']:.0f}) and the price still eroding "
            f"({R['exgfc']['px_cagr']:.1f}%/yr). The erosion is chronic, not one bad crisis."
        ),
        md(
            "### 4d · Crisis autopsies — the carry's tail\n\n"
            "Peak-to-trough total-return drawdowns inside four fixed, documented windows. A genuine "
            "bond substitute should look like IEF's column. It looks like leverage."
        ),
        code(
            "cols = ['REM', 'NLY', 'AGNC', 'SPY', 'IEF']\n"
            "if HAVE_REAL:\n"
            "    ct = st.crisis_table(TR, cols)\n"
            "else:\n"
            "    ct = pd.DataFrame([c[1:] for c in R['crises']], columns=cols,\n"
            "                      index=[c[0] for c in R['crises']])\n"
            "print(ct.round(1).to_string())\n"
            "fig, ax = plt.subplots(figsize=(9.8, 4.6))\n"
            "x = np.arange(len(ct)); wdt = .16\n"
            "shades = [RED, '#e07b5f', AMBER, GREY, GREEN]\n"
            "for k, (c, col) in enumerate(zip(cols, shades)):\n"
            "    ax.bar(x + (k - 2) * wdt, ct[c].values, wdt, color=col, label=c)\n"
            "ax.set_xticks(x); ax.set_xticklabels([i.replace(' 2', '\\n2') for i in ct.index], fontsize=9)\n"
            "ax.set_ylabel('drawdown (%)'); ax.legend(fontsize=9, ncol=5)\n"
            "ax.set_title('Every stress: the income sleeve falls MORE than equities while duration rallies')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "> 💡 In plain words: in 2008 and 2020 the mREIT complex fell **more than the stock "
            "market** while the Treasuries it supposedly monetises went **up** — repo funding "
            "vanished and the books were forced sellers (Gorton-Metrick run-on-repo; the 2020 margin "
            "spiral). 2013 is the negative-convexity signature: −25 to −37% on a rates *tantrum* that "
            "cost SPY 5.6%. The carry is short liquidity and short volatility — it pays until the one "
            "month you needed it not to. *(AGNC's GFC cell starts at its 2008-05 listing.)*"
        ),
        md(
            "### 4e · Faithful-engine control — we know the truth here\n\n"
            "Deterministic three-factor world with planted `carry` and `alpha` knobs (seed 611). The "
            "null must stay silent; planted values must be recovered."
        ),
        code(
            "res = []\n"
            "for carry, alpha in [(0.0, 0.0), (0.010, -0.0070)]:\n"
            "    ttr, tpx = data.synthetic_world(carry=carry, alpha=alpha, seed=611)\n"
            "    smtr, smpx = st.aligned_monthly(ttr, tpx, ['MREIT', 'IEF', 'SPY', 'BIL'])\n"
            "    sd = st.decompose(smtr, smpx, 'MREIT')\n"
            "    scp = st.carry_premium(smtr, 'MREIT')\n"
            "    res.append((carry * 1e4, alpha * 1e4, sd['div_bps_mo'], sd['div_hac_t'],\n"
            "                scp['alpha_bps_mo'], scp['t_alpha']))\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))\n"
            "labels = [f'planted carry\\n{r[0]:.0f} bps' for r in res]\n"
            "a1.bar(labels, [r[3] for r in res], color=[GREY, GREEN], width=.5)\n"
            "a1.axhline(2, ls='--', c=RED); a1.set_title('carry detector: silent at 0, lights up at 100 bps')\n"
            "for i, r in enumerate(res): a1.annotate(f't={r[3]:.1f}', (i, r[3]), ha='center', va='bottom')\n"
            "labels2 = [f'planted alpha\\n{r[1]:.0f} bps' for r in res]\n"
            "a2.bar(labels2, [r[5] for r in res], color=[GREY, RED], width=.5)\n"
            "a2.axhline(-2, ls='--', c='k'); a2.set_title('alpha test: silent at 0, recovers -70 bps')\n"
            "for i, r in enumerate(res): a2.annotate(f't={r[5]:.2f}', (i, r[5]), ha='center',\n"
            "                                        va='top' if r[5] < 0 else 'bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in res:\n"
            "    print(f'planted carry={r[0]:+6.1f} alpha={r[1]:+6.1f} bps/mo -> '\n"
            "          f'div {r[2]:+.2f} (t {r[3]:+.2f}) | alpha {r[4]:+.2f} (t {r[5]:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: under the (0, 0) null the detectors read t = "
            f"{R['syn'][0][3]:+.2f} and {R['syn'][0][5]:+.2f} — no manufactured significance; the "
            f"planted +100 bps carry and −70 bps alpha are recovered at t = {R['syn'][1][3]:.1f} and "
            f"{R['syn'][1][5]:.2f}, betas 2.64/0.56 vs planted 2.50/0.55. *(A machinery proof only — "
            "never cited in support of the real-tape stamps.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — the leveraged-MBS carry exists and is huge: REM's dividend "
            f"component is **+{d['REM'][1]:.2f} bps/mo (+{d['REM'][2]:.2f}%/yr) at HAC "
            f"t = +{d['REM'][3]:.2f}** (NLY +{d['NLY'][2]:.2f}%/yr t = {d['NLY'][3]:.1f}; AGNC "
            f"+{d['AGNC'][2]:.2f}%/yr t = {d['AGNC'][3]:.1f}), and the erosion financing it clears "
            f"the bar too (α t = {rc['REM'][2]:.2f}, robust across lags). The headline leg is an "
            "index fund — not survivor-biased.\n"
            f"- **Tradability `MIRAGE`** — 19.1 years of harvest left TR CAGR at "
            f"**{d['REM'][4]:.2f}%/yr** (excess over bills t = +{R['rem_excess_t']:.2f}: cash, "
            f"statistically) at {R['perf']['REM'][1]:.0f}% vol and a {R['perf']['REM'][3]:.0f}% "
            f"drawdown, losing **{rc['REM'][1]:.2f}%/yr (t = {rc['REM'][2]:.2f})** to a passive "
            "levered IEF+SPY mix. Costs are irrelevant (~0.5 bps/yr); the wrapper is the problem.\n"
            f"- **Free lunch? `BUSTED`** — the coupon is financed by the price leg "
            f"(**{d['REM'][5]:.2f}%/yr**; $100 → ${d['REM'][7]:.1f} price-only) and carries "
            f"β_SPY = {rc['REM'][5]:.2f} plus a repo margin call. Spending the dividend is spending "
            "principal."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The packaged-carry family.** [341 — MLP-Pipelines](../../341-mlp-pipelines/README.md) "
            "(energy beta under a toll-road label) and [342 — BDC-Yield](../../342-bdc-yield/README.md) "
            "(levered private credit under a senior-loan label) are the named siblings — same Real × "
            "Mirage pathology, three different assets. The mortgage flavour is the purest carry trade "
            "of the three: levered repo NIM, short liquidity, short convexity (Koijen et al. 2018; "
            "Gorton-Metrick 2012; Hanson 2014).\n"
            "- **What would change the verdict.** A post-2022 regime with wide MBS spreads and a "
            "steep curve could make H₂ pass on a fresh window; the engine re-runs on any slice — "
            "watch the NW alpha vs the levered-IEF benchmark, never the yield.\n"
            "- **Book equity vs market price.** Our NAV erosion is measured on market prices; "
            "book-value-per-share erosion (from filings) is the natural cross-check and gives the "
            "same sign — the payout has exceeded the economic earn for most of the sample.\n\n"
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
