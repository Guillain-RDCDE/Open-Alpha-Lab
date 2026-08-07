"""Generate the two narrative notebooks for Study 823 (Variance-Risk-Premium Return Predictor).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from
the frozen ``R`` dict (mirroring docs/results.md); the live cells run only the fast
synthetic positive control, so execution is quick and network-free.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily SPY + ^VIX,
# 1993-03-31 -> 2026-06-30, 400 month-ends; VRP = (VIX/100)^2/12 - trailing-21d realized
# variance; forward SPY return regressed on VRP_t; SPY fingerprint b61b772dba46).
R = dict(
    fp="b61b772dba46", n_days=8411, n_months=400, start="1993-03-31", end="2026-06-30",
    vrp_mean_var=0.00083, vrp_volpts=10.0,
    # headline by horizon: (label, slope, NW t, R2 %, tercile spread %)
    horiz=[("1m", 0.41, 0.27, 0.12, 0.68), ("3m", 2.12, 0.75, 1.07, 2.33),
           ("6m", 1.45, 0.51, 0.23, 2.20), ("12m", -1.53, -0.65, 0.12, 0.57)],
    h3_slope=2.12, h3_t=0.75, h3_r2=1.07, h3_tols=2.06,
    # two-era cut, 3-month horizon
    early_slope=7.30, early_t=5.93, early_r2=8.65, early_n=199,
    late_slope=-1.84, late_t=-1.28, late_r2=1.27, late_n=195,
    early1_t=3.33, late1_t=-0.72,
    # placebo (3m)
    placebo_obs=2.12, placebo_mean=-0.03, placebo_sd=1.06, placebo_p=0.024,
    # timer (1m, 5 bps)
    timer_sharpe=0.55, bh_sharpe=0.69, switches_yr=5.0, inv_frac=0.51,
    timer_spread=-0.33, timer_t=-2.21,
    # synthetic control (20 seeds, 3m): (edge, mean NW t)
    ctrl=[(0.0, 0.28), (2.0, 1.53), (4.0, 2.76), (6.0, 3.95), (10.0, 6.19)],
    null_t=0.28, null_fire=0, planted_edge=6.0, planted_t=3.95, planted_r2=6.6, planted_fire=20,
)


BOOT = (
    "import sys, os\n"
    'sys.path.insert(0, os.path.abspath(".."))          # the study package\n'
    'sys.path.insert(0, os.path.abspath("../../.."))    # repo root\n'
    "%matplotlib inline\n"
    "import numpy as np, pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    'plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,\n'
    '                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})\n'
    'RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"\n'
    "\n"
    "from vrp_predictor import data, strategy as st\n"
    "\n"
    "# Frozen real-tape headline (mirror of docs/results.md) — the notebook runs fully offline.\n"
    "R = " + repr(R) + "\n"
    "print('Study 823 — frozen real-tape headline loaded:', R['n_months'], 'month-ends,',\n"
    "      R['start'], '->', R['end'], '| SPY fingerprint', R['fp'])\n"
)


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Variance Risk Premium — does 'the price of fear' tell you *when* the market "
            "will pay? 🌩️\n"
            "### The gap between the volatility options charge and the volatility that actually "
            "shows up, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![In--sample_vs_out--of--sample: Decayed](https://img.shields.io/badge/In--sample_vs_out--of--sample-Decayed-8b949e?style=flat-square)\n\n"
            "Every day the options market quotes a price for *future* volatility — that's the **VIX**, "
            "the 'fear gauge.' And every day the market actually *delivers* some amount of volatility — "
            "what really happened. The **variance risk premium** is the gap between the two: how much "
            "extra investors pay for volatility insurance above what volatility turns out to cost. On "
            "average it's positive — insurance has a markup.\n\n"
            "The famous idea we test (Bollerslev, Tauchen & Zhou, 2009): when that markup is **fat**, "
            "investors are scared and risk-averse — and a scared market goes on to pay a **higher** "
            "return over the next few months. So a fat variance risk premium should *predict* higher "
            "forward stock returns, most strongly about a **quarter** ahead.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does a fat premium precede higher returns? | **It used to.** The *shape* the theory "
            f"predicts is exactly here — positive, peaking about a **quarter** ahead — and back in "
            f"**1993–2009** it was a strong signal (a *t*-stat of **+{R['early_t']}**, the bar is 2). |\n"
            f"| Does it still work? | **No.** On the full 1993–2026 tape the signal fades to a *t* of "
            f"just **+{R['h3_t']}**, and after **2010** it actually goes *backwards* (*t* "
            f"{R['late_t']}). A famous edge that quietly faded once everyone knew about it. |\n"
            f"| Could you trade it? | **No.** A timer that owns stocks only when the premium is fat "
            f"**loses** to just buying and holding (Sharpe {R['timer_sharpe']} vs {R['bh_sharpe']}). |\n\n"
            "> The variance risk premium isn't a myth — it was a real predictor in its day, and it "
            "still points the right way. But its punch has decayed to a whisper, and the whisper turned "
            "negative in the last fifteen years."
        ),

        # ---- 1 THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"The difference between implied and realized variance — the variance risk premium — "
            "predicts aggregate stock market returns, with the explanatory power highest at the "
            "quarterly horizon.\"* — Bollerslev, Tauchen & Zhou (2009).\n\n"
            "The intuition: the variance risk premium is what you pay for **protection against "
            "turbulence**. When it's fat, the market is unusually risk-averse — and a risk-averse "
            "market is *cheap*, so it earns more going forward. It's the equity premium, read off the "
            "options market instead of off valuations."
        ),

        # ---- 2 SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If it held, you'd have a forward-looking market timer built from *option* prices — a "
            "cleaner, faster read on the equity premium than slow valuation ratios (P/E, "
            "price-to-dividend), which barely move month to month. It would also tie the stock market's "
            "return directly to the **price of volatility risk**. That's why the paper was such a big "
            "deal. The question is whether it *survived* — out of its original sample, and out of its "
            "original decade."
        ),

        # ---- 3 HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Measure the premium.** Implied variance = the VIX, squared and put on a monthly "
            "footing. Realized variance = how much the S&P actually bounced around over the last month "
            "(21 trading days). The premium is implied **minus** realized.\n"
            "2. **Line it up with the future.** For each month, note the premium, then look at how SPY "
            "did over the *next* 1 and 3 months.\n"
            "3. **Fit the line.** Does a fatter premium reliably precede a higher forward return? "
            "Measure the slope and — crucially — its *honest* significance.\n\n"
            "*Timing:* the premium is known at month-end; we hold over the following months. No peeking."
        ),

        # ---- 4 THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**How much of next quarter's return does the premium explain, across horizons?** The "
            "theory says the R² should *hump* and peak around 3 months."
        ),
        code(
            "horiz = R['horiz']\n"
            "labs = [h[0] for h in horiz]; r2 = [h[3] for h in horiz]; ts = [h[2] for h in horiz]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "cols = [GREEN if h[1] > 0 else RED for h in horiz]\n"
            "ax.bar(labs, r2, color=cols, width=.55)\n"
            "for x,(rr,tt) in enumerate(zip(r2, ts)):\n"
            "    ax.annotate(f't={tt:+.2f}', (x, rr), ha='center', va='bottom', fontsize=9, color=GREY)\n"
            "ax.set_ylabel('predictive R² (%)'); ax.set_xlabel('forward horizon')\n"
            "ax.set_title('The R² humps and peaks at the quarter — exactly the BTZ shape')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('R² by horizon (%):', [h[3] for h in horiz], '| NW t:', [h[2] for h in horiz])"
        ),
        md(
            f"The **shape is right**: positive, peaking at the 3-month horizon (R² "
            f"**{R['h3_r2']}%**). But look at the *t*-stats printed on the bars — even at the peak the "
            f"honest (Newey-West) *t* is only **+{R['h3_t']}**, well under the bar of 2. The pattern is "
            "there; the significance, on the full tape, is not."
        ),
        md(
            "**Why so weak — was it ever strong?** Split the tape in two: the early years (which cover "
            "the paper's own 1990s–2000s sample) versus everything since 2010."
        ),
        code(
            "eras = ['1993-2009\\n(BTZ era)', '2010-2026']\n"
            f"tvals = [{R['early_t']}, {R['late_t']}]\n"
            f"r2v = [{R['early_r2']}, {R['late_r2']}]\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.3))\n"
            "cols = [GREEN if t>0 else RED for t in tvals]\n"
            "ax.bar(eras, tvals, color=cols, width=.5)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = +2 bar'); ax.axhline(0, c='k', lw=1)\n"
            "for x,(t,rr) in enumerate(zip(tvals, r2v)):\n"
            "    ax.annotate(f'R²={rr}%', (x, t), ha='center', va='bottom' if t>=0 else 'top', color=GREY)\n"
            "ax.set_ylabel('predictive t-stat (3-month)')\n"
            "ax.set_title('Strong in the original era (t≈6), gone and backwards since 2010')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('early t', R['early_t'], '(R²', str(R['early_r2'])+'%)  |  late t', R['late_t'], '(R²', str(R['late_r2'])+'%)')"
        ),
        md(
            f"There it is. In **1993–2009** the premium was a genuinely strong predictor — *t* = "
            f"**+{R['early_t']}**, explaining **{R['early_r2']}%** of the next quarter's return. In "
            f"**2010–2026** it **decays and flips negative** (*t* {R['late_t']}). This is the desk's "
            "most common story: an edge that was real, got published, got crowded, and faded."
        ),

        # ---- 5 VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** Right sign, right horizon (R² peaks at the quarter), and genuinely "
            f"strong in its original era (*t* +{R['early_t']}) — but the full-tape *t* is just "
            f"+{R['h3_t']} and it inverts after 2010.\n"
            "- **Tradability — Mirage.** A premium-timer loses to buy-and-hold (next notebook).\n"
            "- **In-sample vs out-of-sample — Decayed.** A ~6-*t* edge collapsed to nothing."
        ),

        # ---- 6 COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            f"No. A timer that owns SPY only when the premium is fat sits in cash about half the time "
            f"and ends up with a *worse* Sharpe ({R['timer_sharpe']}) than simply buying and holding "
            f"({R['bh_sharpe']}), giving up **{R['timer_spread']}%/month** on average while trading "
            f"{R['switches_yr']:.0f}× a year. The next notebook does the cost maths."
        ),

        # ---- 7 GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Does the premium even exist?** [130 Vol-Risk-Premium](../../130-vol-risk-premium/) "
            "asks whether implied beats realized *on average* (it does, ~10 vol-points here) — the "
            "*level*, not the *timing*.\n"
            "- **Other VIX signals.** [111 VIX-Term-Structure](../../111-vix-term-structure/) (the "
            "curve shape) and [3 Fear-Gauge](../../3-fear-gauge/) (the raw level as a dip-buy).\n\n"
            "*Think the predictor is alive and we just measured it crudely? Fork this, swap in a "
            "model-implied expected-variance (Carr-Wu variance swaps), and show the out-of-sample "
            "3-month *t* clearing 2 after 2010.*"
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
            "# The Variance Risk Premium as a return predictor — a quantitative teardown 🔬\n"
            "### monthly IV−RV · predictive regression · Newey-West slope *t* · OLS-vs-HAC · "
            "quarterly-peaking R² · two-era decay · block-bootstrap placebo · timer costs · synthetic control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![In--sample_vs_out--of--sample: Decayed](https://img.shields.io/badge/In--sample_vs_out--of--sample-Decayed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We build "
            "the monthly variance risk premium `VRP = (VIX/100)²/12 − RV₂₁`, regress the forward 1- and "
            "3-month SPY return on it, and test the Bollerslev-Tauchen-Zhou predictive claim with an "
            "honest HAC standard error.\n\n"
            "> ⚠️ **Not investment advice.** Real data: yfinance daily SPY + `^VIX`, "
            f"{R['n_days']:,} days → {R['n_months']} month-ends ({R['start']} → {R['end']}, SPY fp "
            f"`{R['fp']}`); the offline core and tests run on a deterministic synthetic world. Methods "
            "in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- 0 VERDICT ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | 3-month predictive slope **+{R['h3_slope']}**, **HAC *t* "
            f"+{R['h3_t']}** (OLS *t* +{R['h3_tols']} — the HAC deflates it), R² {R['h3_r2']}% peaking "
            f"at the quarter. Strong in 1993–2009 (*t* +{R['early_t']}), inverts 2010–2026 (*t* "
            f"{R['late_t']}). |\n"
            f"| **Tradability** | `MIRAGE` | VRP timer Sharpe **{R['timer_sharpe']}** vs buy-and-hold "
            f"**{R['bh_sharpe']}**; mean return **{R['timer_spread']}%/mo** (NW *t* {R['timer_t']}). |\n"
            f"| **In-sample vs out-of-sample** | `DECAYED` | early *t* +{R['early_t']} → late *t* "
            f"{R['late_t']} (3m); +{R['early1_t']} → {R['late1_t']} (1m). |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), the BTZ *shape* is "
            "present, but on the full modern tape the predictive slope is faint under an honest HAC "
            "standard error and has decayed to nothing since 2010."
        ),

        # ---- 1 THE CLAIM ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\text{VRP}_t = \\text{IV}_t - \\text{RV}_t$ and $r_{t\\to t+h}$ the forward "
            "$h$-month SPY excess return.\n\n"
            "- **H₁ (predictability).** In $r_{t\\to t+h} = \\alpha + \\beta\\,\\text{VRP}_t + "
            "\\varepsilon$, $\\beta > 0$ at HAC *t* ≥ 2.\n"
            "- **H₂ (horizon shape).** The predictive R² is hump-shaped and peaks near $h = 3$ months.\n"
            "- **H₃ (stability).** The slope's sign and significance are stable across sub-samples.\n"
            "- **H₄ (tradable).** A VRP-conditioned timer beats buy-and-hold net of costs.\n\n"
            f"We find **H₂ confirmed** (R² peaks at 3m, {R['h3_r2']}%), **H₁ rejected on the full tape** "
            f"(HAC *t* +{R['h3_t']}), **H₃ rejected** (strong 1993–2009, inverted 2010–2026), and **H₄ "
            "rejected** (the timer loses to buy-and-hold)."
        ),

        # ---- 2 SO WHAT ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is a **forward-looking equity-premium proxy read off option prices**. If VRP "
            "predicts returns, the price of variance risk *is* the price of the equity premium — a "
            "faster, cleaner signal than sticky valuation ratios. The desk already separated the "
            "premium's *existence* ([130](../../130-vol-risk-premium/)) from its *shape* "
            "([111](../../111-vix-term-structure/)) and its *level* as fear "
            "([3](../../3-fear-gauge/)); here we isolate its **time-series predictive power**."
        ),

        # ---- 3 PROTOCOL ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** $\\text{IV}_t = (\\text{VIX}_t/100)^2/12$; $\\text{RV}_t = \\sum_{21}$ daily "
            "squared log returns; $\\text{VRP}_t = \\text{IV}_t - \\text{RV}_t$, known at month-end $t$.\n"
            "- **Target.** $r_{t\\to t+h}$ = SPY log return over the next $h$ months (risk-free proxied "
            "at 0 — named on the Signal axis; a near-constant bill moves $\\alpha$, not $\\beta$).\n"
            "- **Inference.** OLS slope with a **Newey-West** (HAC, Bartlett) standard error, lags "
            "$=\\lceil 1.5h\\rceil$ — overlapping multi-month returns are serially correlated, so the "
            "homoskedastic *t* overstates significance. A **block-bootstrap placebo** (6-month rotation) "
            "cross-checks.\n"
            "- **Robustness.** Horizons 1/3/6/12m; a two-era split at 2010.\n"
            "- **Frictions.** A long/flat timer (own SPY when VRP > its expanding median, else cash), "
            "one-way cost × NAV per switch.\n"
            "- **Positive control.** A deterministic synthetic world with a planted VRP→return relation "
            "(`edge > 0`) and a null, averaged over 20 seeds (house rule).\n\n"
            "Timing: VRP known at close of month-end $t$; the position is held $t\\to t+h$. That one-lag "
            "convention is the single documented execution rule — the VIX print and the RV window are "
            "both causal, so no future data enters the signal."
        ),

        # ---- 4 TEARDOWN ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The predictive regression — H₁ and the OLS-vs-HAC gap\n\n"
            "Forward 3-month SPY return on $\\text{VRP}_t$. The homoskedastic OLS *t* looks significant; "
            "the honest HAC *t* does not — the whole story of overlapping-return inference."
        ),
        code(
            "labs = ['OLS t\\n(naive)', 'Newey-West t\\n(honest)']\n"
            f"vals = [{R['h3_tols']}, {R['h3_t']}]\n"
            "fig, ax = plt.subplots(figsize=(7.2, 4.3))\n"
            "ax.bar(labs, vals, color=[GREY, AMBER], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, lw=1, label='t = +2 bar'); ax.axhline(0, c='k', lw=1)\n"
            f"ax.set_ylabel('t-stat on the 3-month slope (+{R['h3_slope']})')\n"
            "ax.set_title('The overlap correction deflates the naive t below the bar')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('3m slope', R['h3_slope'], '| OLS t', R['h3_tols'], '| HAC t', R['h3_t'], '| R²', str(R['h3_r2'])+'%')"
        ),
        md(
            f"> 💡 In plain words: H₁ **rejected on the full tape.** The slope is positive "
            f"(+{R['h3_slope']}) and the naive OLS *t* (+{R['h3_tols']}) would 'pass' — but overlapping "
            f"3-month returns are autocorrelated, and the Newey-West correction pulls the *t* down to "
            f"+{R['h3_t']}. The naive test was fooling itself."
        ),
        md(
            "### 4b · Horizon sweep — H₂ (does the R² hump at the quarter?)\n\n"
            "Slope, HAC *t* and R² across 1/3/6/12-month forward horizons."
        ),
        code(
            "horiz = R['horiz']\n"
            "labs = [h[0] for h in horiz]; r2 = [h[3] for h in horiz]; ts = [h[2] for h in horiz]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "cols = [GREEN if h[1] > 0 else RED for h in horiz]\n"
            "a1.bar(labs, r2, color=cols, width=.55); a1.set_ylabel('R² (%)')\n"
            "a1.set_title('R² peaks at 3m (the BTZ hump)')\n"
            "a2.bar(labs, ts, color=cols, width=.55); a2.axhline(2, ls='--', c=GREY, lw=1)\n"
            "a2.axhline(0, c='k', lw=1); a2.set_ylabel('Newey-West t')\n"
            "a2.set_title('...but the HAC t never clears 2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('R² %:', r2, '| HAC t:', ts)"
        ),
        md(
            "> 💡 In plain words: H₂ **confirmed** — the R² is hump-shaped and peaks at the 3-month "
            "horizon, precisely the Bollerslev-Tauchen-Zhou signature. The mechanism is real; the "
            "full-sample *significance* is what's missing."
        ),
        md(
            "### 4c · Two-era cut — H₃ (is the edge stable?)\n\n"
            "The 3-month predictive regression in 1993–2009 (the paper's era) vs 2010–2026."
        ),
        code(
            "eras = ['1993-2009 (BTZ era)', '2010-2026']\n"
            f"tvals = [{R['early_t']}, {R['late_t']}]; slopes = [{R['early_slope']}, {R['late_slope']}]\n"
            f"r2v = [{R['early_r2']}, {R['late_r2']}]\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "cols = [GREEN if t>0 else RED for t in tvals]\n"
            "ax.bar(range(2), tvals, color=cols, width=.5)\n"
            "ax.set_xticks(range(2)); ax.set_xticklabels(eras)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = +2 bar'); ax.axhline(0, c='k', lw=1)\n"
            "for x,(t,s,rr) in enumerate(zip(tvals, slopes, r2v)):\n"
            "    ax.annotate(f'slope {s:+.1f}\\nR² {rr}%', (x, t), ha='center',\n"
            "                va='bottom' if t>=0 else 'top', fontsize=9, color=GREY)\n"
            "ax.set_ylabel('3-month predictive t-stat')\n"
            "ax.set_title('A ~6-t edge in the original era decays and inverts after 2010')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('early: slope', R['early_slope'], 't', R['early_t'], 'R²', str(R['early_r2'])+'% (n='+str(R['early_n'])+')  |',\n"
            "      'late: slope', R['late_slope'], 't', R['late_t'], 'R²', str(R['late_r2'])+'% (n='+str(R['late_n'])+')')"
        ),
        md(
            f"> 💡 In plain words: H₃ **rejected.** The predictor was strong and highly significant in "
            f"**1993–2009** (*t* +{R['early_t']}, R² {R['early_r2']}%) — a window that brackets BTZ's "
            f"own 1990–2007 sample — then **decayed and inverted** in **2010–2026** (*t* {R['late_t']}). "
            "A post-publication fade, not a stable signal."
        ),
        md(
            "### 4d · Placebo — a lucky alignment?\n\n"
            "Block-rotate the forward returns against VRP (6-month blocks, 2,000 draws)."
        ),
        code(
            f"obs, pmean, psd, p = {R['placebo_obs']}, {R['placebo_mean']}, {R['placebo_sd']}, {R['placebo_p']}\n"
            "draws = np.random.default_rng(0).normal(pmean, psd, 4000)\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "ax.hist(draws, bins=45, color=GREY, alpha=.55)\n"
            "ax.axvline(obs, c=AMBER, lw=2.2, label=f'observed slope {obs:+.2f}')\n"
            "ax.axvline(pmean, c='k', lw=1, ls='--', label=f'null mean {pmean:+.2f}')\n"
            "ax.set_xlabel('predictive slope under block rotation'); ax.set_ylabel('count')\n"
            "ax.set_title('Naive rotation puts it ~2σ right (p=%s) — but the HAC t is only +%s' % (p, R['h3_t']))\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f} vs null {pmean:+.2f} (sd {psd}) -> right-tail p {p}')"
        ),
        md(
            f"> 💡 In plain words: the block rotation places the slope ~2σ right (*p* {R['placebo_p']}) — "
            "but it holds VRP **fixed**, so it misses the fat-tailed leverage of a few crisis-month VRP "
            f"spikes that the **HAC** *t* (+{R['h3_t']}) captures. We stamp on the HAC and the era split, "
            "not the naive rotation — both say the full-tape edge isn't robust."
        ),

        # ---- 5 VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — H₂ confirmed (R² peaks at the quarter, {R['h3_r2']}%), H₁ rejected "
            f"on the full tape (slope +{R['h3_slope']}, HAC *t* +{R['h3_t']} vs OLS +{R['h3_tols']}), "
            f"H₃ rejected (early *t* +{R['early_t']} → late {R['late_t']}).\n"
            f"- **Tradability `MIRAGE`** — timer Sharpe {R['timer_sharpe']} vs buy-and-hold "
            f"{R['bh_sharpe']}; mean return {R['timer_spread']}%/mo (NW *t* {R['timer_t']}).\n"
            f"- **In-sample vs out-of-sample `DECAYED`** — a ~6-*t* edge collapsed to an insignificant, "
            "wrong-signed −1.28."
        ),

        # ---- 6 TRADE IT ----
        md(
            "## 6 · Could you trade it? — the timer costs\n\n"
            "Own SPY when VRP > its expanding median, else cash; 5 bps one-way per switch. Compare "
            "Sharpe **and** mean return to buy-and-hold SPY."
        ),
        code(
            f"ts, bs, sw = {R['timer_sharpe']}, {R['bh_sharpe']}, {R['switches_yr']}\n"
            f"sp, spt = {R['timer_spread']}, {R['timer_t']}\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))\n"
            "a1.bar(['timer\\n(net)','buy & hold'], [ts, bs], color=[AMBER, GREY], width=.5)\n"
            "a1.set_ylabel('annualised Sharpe'); a1.axhline(0, c='k', lw=1)\n"
            "a1.set_title(f'Sharpe: {ts} vs {bs} (timer loses)')\n"
            "a2.bar(['timer - buy&hold'], [sp], color=(RED if sp<0 else GREEN), width=.4)\n"
            "a2.axhline(0, c='k', lw=1); a2.set_ylabel('mean-return spread (%/month)')\n"
            "a2.set_title(f'Mean return: {sp}%/mo (NW t {spt})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'timer Sharpe {ts} | buy&hold {bs} | switches/yr {sw} | mean spread {sp}%/mo (t {spt})')"
        ),
        md(
            f"> 💡 In plain words: the timer sits in cash ~{int(R['inv_frac']*100)}% of the time invested "
            f"and still **loses** to the index — a lower Sharpe ({R['timer_sharpe']} vs {R['bh_sharpe']}) "
            f"and a significantly negative mean-return spread ({R['timer_spread']}%/mo, NW *t* "
            f"{R['timer_t']}). `MIRAGE`."
        ),

        # ---- 7 SYNTHETIC CONTROL ----
        md(
            "## 7 · Going further — the synthetic positive control (LIVE)\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant a real VRP→return "
            "relation (`edge > 0`) of increasing strength and watch the predictive HAC *t* rise — "
            "averaged over 20 seeds so no single lucky seed can fake it. This cell runs **live** "
            "(deterministic, no network); everything above is the frozen real tape."
        ),
        code(
            "edges = [c[0] for c in R['ctrl']]\n"
            "res = [st.synthetic_mean_t(data, edge=e, n_seeds=20, horizon=3) for e in edges]\n"
            "ts = [r['mean_t'] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(edges, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = +2 bar'); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted edge (VRP -> forward-return loading)')\n"
            "ax.set_ylabel('mean predictive HAC t (20 seeds, 3m)')\n"
            "ax.set_title('The harness banks a planted edge — flat at the null, past +2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for e, r in zip(edges, res):\n"
            "    print(f'edge {e:+.1f} -> mean slope {r[\"mean_slope\"]:+.2f}, mean HAC t {r[\"mean_t\"]:+.2f}, fires {int(r[\"fire_frac\"]*20)}/20')"
        ),
        md(
            f"The mean HAC *t* is ≈ 0 at the null (fires **{R['null_fire']}/20**, no false positive) and "
            f"climbs far past +2 as the planted edge grows (at `edge=6`, *t* ≈ +{R['planted_t']}, fires "
            f"**{R['planted_fire']}/20**) — so the engine is a faithful, unbiased detector. The weak "
            "full-tape real-tape result and the sharp 1993-2009 → 2010-2026 decay are therefore genuine "
            "features of the market, not a broken harness. For whether the premium even *exists*, see "
            "[130 Vol-Risk-Premium](../../130-vol-risk-premium/)."
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
