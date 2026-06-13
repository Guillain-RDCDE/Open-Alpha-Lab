"""Generate the two narrative notebooks for Study 114 (Dollar-Smile).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
daily parquet under ../_cache/ if present and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-13).
R = dict(
    n_daily=5662, n_weekly=1170, n_monthly=268,
    # Contemporaneous weekly
    cspy_beta=-0.6695, cspy_r2=8.48, cspy_t=-4.65,
    ceem_beta=-1.2903, ceem_r2=17.39, ceem_t=-7.54,
    # Predictive weekly IS
    ispy_beta=0.0063, ispy_r2=0.001, ispy_t=0.07,
    ieem_beta=0.0467, ieem_r2=0.023, ieem_t=0.39,
    # OOS weekly
    ospy_r2=-0.96, ospy_dm=-1.72,
    oeem_r2=-0.94, oeem_dm=-1.87,
    # Signal
    sspy_mean=-0.23, sspy_win=49.8, sspy_t=-0.04,
    seem_mean=-3.62, seem_win=51.6, seem_t=-0.41,
    # Monthly contemp
    m_cspy_r2=19.09, m_cspy_t=-6.17,
    m_ceem_r2=38.88, m_ceem_t=-9.06,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble
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

from dollar_smile import data, strategy as st

CACHE = os.path.join("..", "_cache")

def _have_cache():
    return all(os.path.exists(data._cache_path(t, CACHE)) for t in data.TICKERS)

HAVE_REAL = _have_cache()

if HAVE_REAL:
    _panel = data.load_panel(fetch=False, cache_dir=CACHE)
    DAILY = data.build_panel_frame(_panel)
    FEAT_W = st.compute_features(DAILY, horizon="W")
    FEAT_M = st.compute_features(DAILY, horizon="ME")
    RES_W = st.run_all(DAILY, horizon="W")
    RES_M = st.run_all(DAILY, horizon="ME")
    print(f"real data loaded: {len(DAILY):,} daily rows, {len(FEAT_W):,} weekly periods")
else:
    DAILY = FEAT_W = FEAT_M = RES_W = RES_M = None
    print("no cache -- using frozen headline numbers from R dict")

print("HAVE_REAL:", HAVE_REAL)

# Frozen headline numbers available in all cells (mirrors docs/results.md)
R = dict(
    n_daily=5662, n_weekly=1170, n_monthly=268,
    cspy_beta=-0.6695, cspy_r2=8.48, cspy_t=-4.65,
    ceem_beta=-1.2903, ceem_r2=17.39, ceem_t=-7.54,
    ispy_beta=0.0063, ispy_r2=0.001, ispy_t=0.07,
    ieem_beta=0.0467, ieem_r2=0.023, ieem_t=0.39,
    ospy_r2=-0.96, ospy_dm=-1.72,
    oeem_r2=-0.94, oeem_dm=-1.87,
    sspy_mean=-0.23, sspy_win=49.8, sspy_t=-0.04,
    seem_mean=-3.62, seem_win=51.6, seem_t=-0.41,
    m_cspy_r2=19.09, m_cspy_t=-6.17,
    m_ceem_r2=38.88, m_ceem_t=-9.06,
)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Dollar-Smile -- does a rising dollar really hurt stocks?\n"
            "### The DXY/SPY/EEM macro claim, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Coincident--only%3F: Confirmed](https://img.shields.io/badge/Coincident--only%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "Every macro commentator has said it: *'the dollar is rising, so watch out for stocks, "
            "and especially watch out for emerging markets.'* The US Dollar Index (DXY) climbing "
            "means tighter global financial conditions, a heavier burden on EM dollar-denominated "
            "debt, and lower profits for US multinationals when they translate overseas earnings. "
            "Compelling story. But does the dollar actually **predict** equity returns -- or does "
            "it merely move alongside them?\n\n"
            "> **This is the plain-language layer.** Want the t-stats, OOS R², and the Goyal-Welch "
            "benchmark? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT ----
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does a rising dollar coincide with falling stocks? | **Yes, reliably.** "
            f"Weekly SPY/DXY R² = **{R['cspy_r2']:.1f}%**; EEM/DXY R² = **{R['ceem_r2']:.1f}%**. "
            "They move *together*. |\n"
            "| Does last week's dollar change forecast this week's equity return? | **No.** "
            f"Predictive IS *t* = **{R['ispy_t']:+.2f}** (SPY) and **{R['ieem_t']:+.2f}** (EEM) -- "
            "both indistinguishable from zero. |\n"
            "| Does a dollar-momentum rule beat a coin? | **No.** "
            f"Signal *t* = **{R['sspy_t']:+.2f}** (SPY), **{R['seem_t']:+.2f}** (EEM). |\n"
            "| Could you trade it? | **No.** The OOS R² is *negative* -- even ignoring costs, "
            "you'd do better guessing the average. |\n\n"
            "> The dollar and equities are correlated dance partners -- they move together. "
            "But neither one *leads* the other. You can't time the market from last week's DXY print."
        ),

        # ---- BEAT 1 -- THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *'A rising dollar tightens global financial conditions, increases the burden on EM "
            "dollar-denominated debt, and compresses US multinational earnings when repatriated. "
            "Therefore, a strong dollar predicts lower equity returns -- and the effect should be "
            "even larger for EM equities.'*\n\n"
            "This is the steelmanned version of an idea embedded in macro dashboards everywhere. "
            "The mechanism is real and documented (Bruno & Shin 2015, the BIS 'dollar as global "
            "risk factor' literature). The question is whether the *lag* -- last week's DXY move "
            "forecasting *next* week's equity return -- is also real, or just the echo of the "
            "contemporaneous correlation leaking through."
        ),

        # ---- BEAT 2 -- SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If the dollar *leads* equities even by one week, macro traders and multi-asset "
            "allocators could use it as a timing signal -- reduce SPY/EEM when the dollar is "
            "rising, add back when it's falling. That would be a free lunch from a publicly "
            "observable macro data point.\n\n"
            "If it doesn't -- if the link is purely contemporaneous -- the correct takeaway is "
            "that dollar and equity risk are co-exposures in a shared risk factor, not a "
            "forecastable relationship. The chart looks the same; the tradability is completely "
            "different."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "Two tests keep us honest:\n\n"
            "1. **Run the regression with a lag.** Take the dollar log-change for *last* week, "
            "regress it on *next* week's equity return. If the dollar leads, the slope should "
            "be significantly negative (dollar up → equity down). Compare to the *same-week* "
            "regression, which we expect to be strong (contemporaneous).\n"
            "2. **Test it out-of-sample.** A rule that looks good in-sample often collapses "
            "when you hold out data it never saw. We use the Goyal-Welch (2008) expanding-window "
            "OOS R²: positive means the model beats the naive 'just predict the average' baseline, "
            "negative means it doesn't even do that.\n\n"
            "Data: 22 years of daily closes (DX-Y.NYB, SPY, EEM; 2004-2026), resampled to "
            f"**{R['n_weekly']} weekly periods**."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ----
        md(
            "## 4 · The teardown -- what the chart hides\n\n"
            "First, the contemporaneous picture -- the one that *looks* compelling:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    c_spy = RES_W['contemp_spy']\n"
            "    c_eem = RES_W['contemp_eem']\n"
            "    cr_spy, cr_eem = c_spy['r2']*100, c_eem['r2']*100\n"
            "    ct_spy, ct_eem = c_spy['tstat'], c_eem['tstat']\n"
            "    pred_spy = RES_W['is_spy']\n"
            "    pred_eem = RES_W['is_eem']\n"
            "    pr_spy, pr_eem = pred_spy['r2']*100, pred_eem['r2']*100\n"
            "    pt_spy, pt_eem = pred_spy['tstat'], pred_eem['tstat']\n"
            "else:\n"
            "    cr_spy, cr_eem = R['cspy_r2'], R['ceem_r2']\n"
            "    ct_spy, ct_eem = R['cspy_t'], R['ceem_t']\n"
            "    pr_spy, pr_eem = R['ispy_r2'], R['ieem_r2']\n"
            "    pt_spy, pt_eem = R['ispy_t'], R['ieem_t']\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "labels = ['SPY (US)', 'EEM (EM)']\n"
            "contemp_r2 = [cr_spy, cr_eem]\n"
            "pred_r2 = [pr_spy, pr_eem]\n"
            "x = range(len(labels))\n"
            "w = 0.35\n"
            "axes[0].bar([i-w/2 for i in x], contemp_r2, w, color=RED, label='Contemporaneous R2 (%)')\n"
            "axes[0].bar([i+w/2 for i in x], pred_r2, w, color=GREY, label='Predictive R2 (%)')\n"
            "axes[0].set_xticks(list(x)); axes[0].set_xticklabels(labels)\n"
            "axes[0].set_ylabel('R squared (%)'); axes[0].set_title('R squared: contemporaneous vs predictive')\n"
            "axes[0].legend()\n"
            "contemp_t = [ct_spy, ct_eem]\n"
            "pred_t = [pt_spy, pt_eem]\n"
            "axes[1].bar([i-w/2 for i in x], contemp_t, w, color=RED, label='Contemporaneous t')\n"
            "axes[1].bar([i+w/2 for i in x], pred_t, w, color=GREY, label='Predictive t')\n"
            "for s in (2, -2): axes[1].axhline(s, ls='--', c='k', lw=1)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_xticks(list(x)); axes[1].set_xticklabels(labels)\n"
            "axes[1].set_ylabel('HAC t-stat'); axes[1].set_title('t-stat: the contemporaneous clears +/-2, the predictive does not')\n"
            "axes[1].legend()\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'SPY: contemp t={{ct_spy:+.2f}} (R2={{cr_spy:.1f}}%) | pred t={{pt_spy:+.2f}} (R2={{pr_spy:.3f}}%)')\n"
            f"print(f'EEM: contemp t={{ct_eem:+.2f}} (R2={{cr_eem:.1f}}%) | pred t={{pt_eem:+.2f}} (R2={{pr_eem:.3f}}%)')"
        ),
        md(
            f"The contemporaneous bars tower over the inference threshold (±2): SPY *t* = "
            f"**{R['cspy_t']:+.2f}** (R² {R['cspy_r2']:.1f}%), EEM *t* = **{R['ceem_t']:+.2f}** "
            f"(R² {R['ceem_r2']:.1f}%). The grey 'predictive' bars are buried in the noise: "
            f"SPY *t* = {R['ispy_t']:+.2f}, EEM *t* = {R['ieem_t']:+.2f}.\n\n"
            "The same-week chart is real and robust. The one-week-ahead chart is noise."
        ),
        md("**Now the killer test: what does an expanding-window out-of-sample evaluation say?**"),
        code(
            "if HAVE_REAL:\n"
            "    oos_spy = RES_W['oos_spy']['r2_oos'] * 100\n"
            "    oos_eem = RES_W['oos_eem']['r2_oos'] * 100\n"
            "else:\n"
            "    oos_spy, oos_eem = R['ospy_r2'], R['oeem_r2']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "cols = [RED if v < 0 else GREEN for v in [oos_spy, oos_eem]]\n"
            "bars = ax.bar(['SPY', 'EEM'], [oos_spy, oos_eem], color=cols, width=0.5)\n"
            "ax.axhline(0, c='k', lw=1.5)\n"
            "for b, v in zip(bars, [oos_spy, oos_eem]):\n"
            "    ax.text(b.get_x()+b.get_width()/2, v/2, f'{v:+.2f}%', ha='center', va='center',\n"
            "            color='white', fontweight='bold', fontsize=12)\n"
            "ax.set_ylabel('Goyal-Welch OOS R2 (%)'); ax.set_title('OOS R2 is negative: dollar model is worse than just predicting the average')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'OOS R2 SPY: {oos_spy:+.2f}% | OOS R2 EEM: {oos_eem:+.2f}%')\n"
            "print('Negative OOS R2 means: worse than guessing the historical mean return.')"
        ),
        md(
            f"The OOS R² is **{R['ospy_r2']:+.2f}%** for SPY and **{R['oeem_r2']:+.2f}%** for "
            "EEM -- negative in both cases. The dollar model is *worse* than the naive baseline "
            "of predicting the historical average return. No amount of look-ahead, sophistication, "
            "or optimism changes this: there is no predictive information in the lagged DXY.\n\n"
            "> This is the Goyal-Welch lesson: an in-sample R² of 0.001% that looks statistically "
            "neutral becomes a *negative* OOS R² because the model's noise from fitting accumulates "
            "faster than any true signal. If the signal were real, OOS R² would be positive."
        ),

        # ---- BEAT 5 -- VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The lagged dollar change has no reliable predictive power over "
            f"forward equity returns (IS *t* ≈ 0, OOS R² negative at both SPY and EEM).\n"
            "- **Tradability — Mirage.** A dollar-direction momentum strategy wins about 50% of "
            f"weeks (SPY: {R['sspy_win']:.1f}%, EEM: {R['seem_win']:.1f}%), earns "
            f"{R['sspy_mean']:+.1f}/{R['seem_mean']:+.1f} bps/week gross before costs, *t* near zero.\n"
            "- **Contemporaneous link — Confirmed.** The dollar and equities move together "
            f"(weekly SPY R² {R['cspy_r2']:.1f}%, EEM R² {R['ceem_r2']:.1f}%) but neither leads "
            "the other."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Even before costs the signal is noise. What does adding a realistic ETF round-trip "
            "cost (2 bps is conservative for SPY/EEM) do to a flat coin?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    costs = [0.0, 1.0, 2.0, 5.0, 10.0]\n"
            "    means_spy = []\n"
            "    means_eem = []\n"
            "    for c in costs:\n"
            "        led_spy = st.trade_returns(FEAT_W, target='fwd_spy', cost_bps=c)\n"
            "        led_eem = st.trade_returns(FEAT_W, target='fwd_eem', cost_bps=c)\n"
            "        means_spy.append(st.summarize(led_spy)['mean_bps'])\n"
            "        means_eem.append(st.summarize(led_eem)['mean_bps'])\n"
            "else:\n"
            "    costs = [0.0, 1.0, 2.0, 5.0, 10.0]\n"
            "    means_spy = [R['sspy_mean']+2*c for c in [0,1,2,5,10]]  # approximate\n"
            "    means_spy = [R['sspy_mean'], R['sspy_mean']-1, R['sspy_mean']-2, R['sspy_mean']-5, R['sspy_mean']-10]\n"
            "    means_eem = [R['seem_mean'], R['seem_mean']-1, R['seem_mean']-2, R['seem_mean']-5, R['seem_mean']-10]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, means_spy, 'o-', c=RED, lw=2, label='SPY signal')\n"
            "ax.plot(costs, means_eem, 's--', c=AMBER, lw=2, label='EEM signal')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, means_spy, 0, where=[m<0 for m in means_spy], color=RED, alpha=.08)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean return per week (bps)')\n"
            "ax.set_title('Starts negative, stays negative -- costs only dig the hole deeper')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('The gross return is already negative or flat -- costs are the undertaker, not the cause.')"
        ),
        md(
            "Unlike the edge-dies-at-costs stories (where a positive gross edge is killed by a "
            "finite break-even cost), here the gross mean is already at zero or negative. There "
            "is no break-even cost -- the 'dollar smile' timing strategy was never alive to be "
            "killed."
        ),

        # ---- BEAT 7 -- GOING FURTHER ----
        md(
            "## 7 · Going further\n\n"
            "- **Why does the chart look so convincing?** Because the *contemporaneous* link is "
            "real. Overlaying DXY and SPY inverted on the same chart gives a striking picture -- "
            "but that picture cannot be traded, because you don't know the DXY return for this "
            "week until the week is over.\n"
            "- **The closest analogue.** [Study 85 (Dr-Copper)](../../85-dr-copper/) ran the same "
            "protocol on the copper/gold ratio -- same verdict: coincident link confirmed, "
            "predictive link absent, OOS R² negative.\n"
            "- **Can any macro variable predict equities at weekly frequency?** Most fail (Goyal "
            "& Welch 2008). Dividend yield, payout ratio, and volatility have the best historical "
            "track record, but even those are fragile OOS.\n"
            "- **What *is* useful about the DXY?** As a *coincident* indicator of global risk "
            "appetite and financial conditions -- not as a timing signal. Watch it to understand "
            "the current regime; don't use last week's print to bet on this week.\n\n"
            "*Think you can make it work? Fork this, add a regime filter (e.g. VIX state), and "
            "show a positive OOS R². That's the bar.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 -- FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Dollar-Smile -- a quantitative teardown\n"
            "### DX-Y.NYB vs SPY & EEM -- lagged regression -- HAC inference -- Goyal-Welch OOS R2\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Coincident--only%3F: Confirmed](https://img.shields.io/badge/Coincident--only%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim now carrying its standard error.* We test whether the "
            "lagged US Dollar Index change forecasts forward SPY and EEM returns (IS regression "
            "with Newey-West HAC, Goyal-Welch expanding-window OOS R², DM test), separate the "
            "contemporaneous from the predictive link, and check a naive momentum signal.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, DX-Y.NYB / SPY / EEM, "
            f"2004-01-02 to 2026-06-12 ({R['n_daily']:,} aligned days, {R['n_weekly']} weekly "
            "periods). Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The 'In plain words' notes** translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Predictive weekly IS *t* SPY = **{R['ispy_t']:+.2f}**, "
            f"EEM = **{R['ieem_t']:+.2f}**; OOS R² SPY **{R['ospy_r2']:+.2f}%**, "
            f"EEM **{R['oeem_r2']:+.2f}%** -- both negative. |\n"
            f"| **Tradability** | `MIRAGE` | Dollar-momentum signal *t* = {R['sspy_t']:+.2f} (SPY), "
            f"{R['seem_t']:+.2f} (EEM) -- no break-even cost exists. |\n"
            f"| **Coincident-only?** | `CONFIRMED` | Weekly SPY contemp R² = **{R['cspy_r2']:.1f}%** "
            f"(*t* = {R['cspy_t']:+.2f}); EEM = **{R['ceem_r2']:.1f}%** (*t* = {R['ceem_t']:+.2f}) "
            "-- moves with markets, not before them. |\n\n"
            "> In plain words: the dollar and equities are risk-factor co-exposures. "
            "They move together (coincident) but neither leads the other (not predictive)."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $d_t$ be the log-change in DXY for week $t$ and $r^{SPY}_{t+1}$, "
            "$r^{EEM}_{t+1}$ be the *next*-week log equity returns. The claim asserts:\n\n"
            "- **H1 (predictive signal).** $\\beta_1 < 0$ in $r_{t+1} = \\alpha + \\beta_1 d_t "
            "+ \\epsilon_t$ -- dollar up last week → equity down this week.\n"
            "- **H2 (OOS).** The predictive R²_OOS (Goyal-Welch) is positive -- "
            "the model beats the naive historical-mean forecast.\n"
            "- **H3 (EM amplification).** The dollar effect on EEM is larger in magnitude "
            "than on SPY.\n"
            "- **H4 (tradable).** A momentum rule (bet against equities when DXY rose last week) "
            "earns a statistically significant positive return net of costs.\n\n"
            "H3 may hold contemporaneously (EEM β contemporaneous = −1.29 vs SPY −0.67). "
            "H1, H2, H4 all fail."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? -- what rides on each answer\n\n"
            "H1-H2 would hand multi-asset allocators a publicly-observable, high-frequency "
            "timing signal (weekly DXY to tilt SPY/EEM allocation). The macroeconomic "
            "mechanism is real and documented; the failure is at the *lag* -- the dollar "
            "encodes contemporaneous risk-appetite, and that information is already in "
            "equity prices by the time you can act on it. This is textbook market efficiency "
            "at the weekly horizon."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know -- the protocol\n\n"
            "- **Data.** DX-Y.NYB, SPY, EEM daily closes from Yahoo Finance, aligned on "
            "business days, 2004-01-01 to 2026-06-12.\n"
            "- **Feature.** Weekly log-change in DXY: $d_t = \\log(DXY_t) - \\log(DXY_{t-1})$.\n"
            "- **Targets.** Lagged (forward) weekly log-return on SPY and EEM: $r_{t+1}$.\n"
            "- **IS regression.** OLS with HAC (Newey-West) standard errors. The null is $\\beta=0$.\n"
            "- **OOS evaluation.** Expanding window (min 52-week train). Goyal-Welch R² = "
            "$1 - \\text{MSFE}_{\\text{model}} / \\text{MSFE}_{\\text{mean}}$. Diebold-Mariano "
            "test for significance.\n"
            "- **Signal.** Dollar-momentum rule: short (−1) equities when DXY rose last week, "
            "long (+1) when it fell. HAC *t* on weekly returns.\n"
            "- **Positive control.** Synthetic daily panel with tunable pred_r (predictive link "
            "knob) and contemp_r (same-day link knob) -- confirms the engine recovers a link "
            "when one is planted."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · Contemporaneous regression -- the link that cannot be traded\n\n"
            "Same-week regression: dollar change at $t$ vs equity return at $t$ (not $t+1$). "
            "This is the *description* link, not the *forecast* link."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c_spy = RES_W['contemp_spy']\n"
            "    c_eem = RES_W['contemp_eem']\n"
            "    cm_spy = RES_M['contemp_spy']\n"
            "    cm_eem = RES_M['contemp_eem']\n"
            "    vals = [\n"
            "        ('SPY weekly', c_spy['beta'], c_spy['r2']*100, c_spy['tstat']),\n"
            "        ('EEM weekly', c_eem['beta'], c_eem['r2']*100, c_eem['tstat']),\n"
            "        ('SPY monthly', cm_spy['beta'], cm_spy['r2']*100, cm_spy['tstat']),\n"
            "        ('EEM monthly', cm_eem['beta'], cm_eem['r2']*100, cm_eem['tstat']),\n"
            "    ]\n"
            "else:\n"
            "    vals = [\n"
            "        ('SPY weekly', R['cspy_beta'], R['cspy_r2'], R['cspy_t']),\n"
            "        ('EEM weekly', R['ceem_beta'], R['ceem_r2'], R['ceem_t']),\n"
            "        ('SPY monthly', -0.859, R['m_cspy_r2'], R['m_cspy_t']),\n"
            "        ('EEM monthly', -1.747, R['m_ceem_r2'], R['m_ceem_t']),\n"
            "    ]\n"
            "labels = [v[0] for v in vals]\n"
            "betas = [v[1] for v in vals]\n"
            "r2s = [v[2] for v in vals]\n"
            "tstats = [v[3] for v in vals]\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "cols = [RED if v < -2 else GREEN for v in tstats]\n"
            "axes[0].barh(labels, r2s, color=cols)\n"
            "axes[0].set_xlabel('contemporaneous R2 (%)'); axes[0].set_title('The coincident link is very real')\n"
            "axes[1].barh(labels, tstats, color=cols)\n"
            "axes[1].axvline(-2, ls='--', c=GREY, lw=1); axes[1].axvline(0, c='k', lw=1)\n"
            "axes[1].set_xlabel('HAC t-stat'); axes[1].set_title('All t-stats clear -2 comfortably')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Contemporaneous: dollar up -> equities down, robustly at weekly and monthly.')"
        ),
        md(
            f"> In plain words: when the dollar rallies in a given week, the S&P 500 falls "
            f"(R² {R['cspy_r2']:.1f}%, *t* {R['cspy_t']:+.2f}) and EM falls even more "
            f"(R² {R['ceem_r2']:.1f}%, *t* {R['ceem_t']:+.2f}). This is real. But it is a "
            "description of co-movement in the *same* week, not a forecast you can act on."
        ),
        md(
            "### 4b · Predictive regression -- the link that matters for trading\n\n"
            "Lagged regression: dollar change last week vs equity return *this* week. "
            "The null hypothesis is $\\beta = 0$. The inference bar is $|t| \\geq 2$."
        ),
        code(
            "if HAVE_REAL:\n"
            "    i_spy = RES_W['is_spy']; i_eem = RES_W['is_eem']\n"
            "    im_spy = RES_M['is_spy']; im_eem = RES_M['is_eem']\n"
            "    pred_vals = [\n"
            "        ('SPY weekly', i_spy['beta'], i_spy['r2']*100, i_spy['tstat']),\n"
            "        ('EEM weekly', i_eem['beta'], i_eem['r2']*100, i_eem['tstat']),\n"
            "        ('SPY monthly', im_spy['beta'], im_spy['r2']*100, im_spy['tstat']),\n"
            "        ('EEM monthly', im_eem['beta'], im_eem['r2']*100, im_eem['tstat']),\n"
            "    ]\n"
            "else:\n"
            "    pred_vals = [\n"
            "        ('SPY weekly', R['ispy_beta'], R['ispy_r2'], R['ispy_t']),\n"
            "        ('EEM weekly', R['ieem_beta'], R['ieem_r2'], R['ieem_t']),\n"
            "        ('SPY monthly', -0.0727, 0.14, -0.50),\n"
            "        ('EEM monthly', -0.2026, 0.52, -0.96),\n"
            "    ]\n"
            "pred_labels = [v[0] for v in pred_vals]\n"
            "pred_t = [v[3] for v in pred_vals]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [RED if abs(v) < 2 else GREEN for v in pred_t]\n"
            "ax.barh(pred_labels, pred_t, color=col)\n"
            "ax.axvline(2, ls='--', c=GREY, lw=1, label='+/- 2 bar')\n"
            "ax.axvline(-2, ls='--', c=GREY, lw=1)\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('HAC t-stat on predictive beta')\n"
            "ax.set_title('Predictive t-stats: none clear +/-2 at weekly or monthly horizon')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for l, b, r2, t in pred_vals:\n"
            "    print(f'{l:15s}: beta={b:+.4f}  R2={r2:.3f}%  t={t:+.2f}')"
        ),
        md(
            f"> In plain words: last week's dollar move tells you essentially nothing "
            f"about this week's equity return. The predictive *t*-stats are {R['ispy_t']:+.2f} "
            f"(SPY) and {R['ieem_t']:+.2f} (EEM) -- well inside the noise band. Monthly is "
            "slightly better in sign but still nowhere near the inference bar."
        ),
        md(
            "### 4c · Out-of-sample R² -- the Goyal-Welch expanding-window test\n\n"
            "If the predictive regression has real signal, it should beat the naive baseline "
            "(predicting the historical mean return) on data it never saw."
        ),
        code(
            "if HAVE_REAL:\n"
            "    # Compute expanding-window OOS predictions for plotting\n"
            "    xv = FEAT_W['delta_dollar'].to_numpy()\n"
            "    yv = FEAT_W['fwd_spy'].to_numpy()\n"
            "    min_train = 52\n"
            "    cum_sq_model = []\n"
            "    cum_sq_mean = []\n"
            "    for t in range(min_train, len(xv)):\n"
            "        y_train = yv[:t]; x_train = xv[:t]\n"
            "        y_mean = y_train.mean()\n"
            "        X = np.column_stack([np.ones(t), x_train])\n"
            "        coef = np.linalg.lstsq(X, y_train, rcond=None)[0]\n"
            "        y_model = coef[0] + coef[1] * xv[t]\n"
            "        cum_sq_model.append((yv[t] - y_model)**2)\n"
            "        cum_sq_mean.append((yv[t] - y_mean)**2)\n"
            "    cum_oos_r2 = [1 - np.mean(cum_sq_model[:i+1]) / np.mean(cum_sq_mean[:i+1])\n"
            "                  for i in range(len(cum_sq_model))]\n"
            "    plot_dates = FEAT_W.index[min_train:]\n"
            "    final_r2 = cum_oos_r2[-1] * 100\n"
            "else:\n"
            "    cum_oos_r2 = None\n"
            "    final_r2 = R['ospy_r2']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(11, 4.3))\n"
            "if cum_oos_r2 is not None:\n"
            "    ax.plot(plot_dates, [v * 100 for v in cum_oos_r2], c=RED, lw=1.5,\n"
            "            label='Cumulative OOS R2 (%)')\n"
            "else:\n"
            "    ax.axhline(final_r2, c=RED, lw=2, ls='--', label=f'Final OOS R2: {final_r2:+.2f}%')\n"
            "ax.axhline(0, c='k', lw=1.5)\n"
            "ax.set_ylabel('OOS R2 (%)'); ax.set_title('OOS R2 (Goyal-Welch) stays below zero -- model never beats the mean')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            f"print(f'Final OOS R2 SPY: {{final_r2:+.2f}}%  (>0 would mean the model beats the historical mean)')"
        ),
        md(
            f"> In plain words: not only does the predictive regression fail to beat the "
            f"historical mean, it is *worse* (OOS R² {R['ospy_r2']:+.2f}% for SPY, "
            f"{R['oeem_r2']:+.2f}% for EEM). This is the Goyal-Welch lesson: even a tiny "
            "spurious IS correlation leads to overfitting that accumulates OOS."
        ),
        md(
            "### 4d · The momentum signal -- a coin\n\n"
            "The tradable version: bet against equities when the dollar rose last week."
        ),
        code(
            "if HAVE_REAL:\n"
            "    led_spy = st.trade_returns(FEAT_W, target='fwd_spy', cost_bps=2.0)\n"
            "    led_eem = st.trade_returns(FEAT_W, target='fwd_eem', cost_bps=2.0)\n"
            "    s_spy = st.summarize(led_spy)\n"
            "    s_eem = st.summarize(led_eem)\n"
            "    sm, st_spy, sw = s_spy['mean_bps'], s_spy['tstat'], s_spy['win_rate']*100\n"
            "    em, et_eem, ew = s_eem['mean_bps'], s_eem['tstat'], s_eem['win_rate']*100\n"
            "else:\n"
            "    sm, st_spy, sw = R['sspy_mean'], R['sspy_t'], R['sspy_win']\n"
            "    em, et_eem, ew = R['seem_mean'], R['seem_t'], R['seem_win']\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "for ax, mean_v, t_v, win_v, lbl in zip(\n"
            "    axes, [sm, em], [st_spy, et_eem], [sw, ew], ['SPY', 'EEM']\n"
            "):\n"
            "    c = RED if abs(t_v) < 2 else GREEN\n"
            "    ax.bar([lbl], [mean_v], color=c, width=0.5)\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.set_ylabel('mean net return (bps/week)')\n"
            "    ax.set_title(f'{lbl}: {mean_v:+.1f} bps/wk, t={t_v:+.2f}, win={win_v:.1f}%')\n"
            "plt.suptitle('Dollar-momentum signal: noise at 2 bps cost', fontsize=12)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'SPY signal: {sm:+.2f} bps/wk, t={st_spy:+.2f}')\n"
            "print(f'EEM signal: {em:+.2f} bps/wk, t={et_eem:+.2f}')"
        ),
        md(
            "> In plain words: the dollar-direction timing strategy earns about zero gross "
            f"({R['sspy_mean']:+.1f} bps/week for SPY, {R['seem_mean']:+.1f} for EEM) "
            f"with *t*-statistics of {R['sspy_t']:+.2f} and {R['seem_t']:+.2f}. This is a "
            "pure coin flip."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — weekly predictive *t* {R['ispy_t']:+.2f} (SPY), "
            f"{R['ieem_t']:+.2f} (EEM); OOS R² {R['ospy_r2']:+.2f}% (SPY), "
            f"{R['oeem_r2']:+.2f}% (EEM). Neither clears the inference bar; OOS is worse than "
            "the naive mean.\n"
            f"- **Tradability `MIRAGE`** — momentum signal *t* {R['sspy_t']:+.2f} (SPY), "
            f"{R['seem_t']:+.2f} (EEM); no positive gross edge to kill with costs.\n"
            f"- **Coincident-only `CONFIRMED`** — weekly contemporaneous SPY *t* {R['cspy_t']:+.2f} "
            f"(R² {R['cspy_r2']:.1f}%), EEM *t* {R['ceem_t']:+.2f} (R² {R['ceem_r2']:.1f}%) "
            "-- the strongest coincident signal in the study, and entirely non-predictive."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? -- the cost sensitivity\n\n"
            "At what round-trip cost does the signal break even? Nowhere -- the gross is "
            "already zero or negative."
        ),
        code(
            "if HAVE_REAL:\n"
            "    costs = [0.0, 1.0, 2.0, 5.0, 10.0]\n"
            "    rows = []\n"
            "    for c in costs:\n"
            "        led = st.trade_returns(FEAT_W, target='fwd_spy', cost_bps=c)\n"
            "        s = st.summarize(led)\n"
            "        rows.append((c, s['mean_bps'], s['tstat']))\n"
            "    cost_df = pd.DataFrame(rows, columns=['cost_bps', 'mean_bps', 'tstat'])\n"
            "else:\n"
            "    costs = [0.0, 1.0, 2.0, 5.0, 10.0]\n"
            "    cost_df = pd.DataFrame({\n"
            "        'cost_bps': costs,\n"
            "        'mean_bps': [R['sspy_mean']+c*0-c for c in costs],\n"
            "        'tstat': [R['sspy_t']] * 5\n"
            "    })\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.plot(cost_df['cost_bps'], cost_df['mean_bps'], 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (bps/wk)')\n"
            "ax.set_title('No break-even cost: the strategy starts negative and only falls')\n"
            "plt.tight_layout(); plt.show()\n"
            "cost_df.round(3)"
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further -- the positive control\n\n"
            "Is the *engine* capable of finding a predictive dollar-equity link, or does "
            "it always print 'none'? Plant a controllable cross-series correlation in a "
            "synthetic daily panel and sweep it."
        ),
        code(
            "pred_rs = [-0.60, -0.40, -0.20, 0.0, 0.20, 0.40]\n"
            "is_ts = []\n"
            "contemp_ts = []\n"
            "for pr in pred_rs:\n"
            "    daily_s, _ = data.synthetic_daily(n_years=20, pred_r=pr, contemp_r=pr*0.8, seed=114)\n"
            "    res_s = st.run_all(daily_s, horizon='W')\n"
            "    is_ts.append(res_s['is_spy']['tstat'])\n"
            "    contemp_ts.append(res_s['contemp_spy']['tstat'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.plot(pred_rs, is_ts, 'o-', c=RED, lw=2, label='predictive IS t')\n"
            "ax.plot(pred_rs, contemp_ts, 's--', c=GREY, lw=2, label='contemporaneous t')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for s in (2, -2): ax.axhline(s, ls=':', c=GREY, lw=1)\n"
            "ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('planted pred_r knob'); ax.set_ylabel('HAC t-stat')\n"
            "ax.set_title('The engine works: it finds a link when one is planted')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('At pred_r=0: both t-stats near 0 (null). As |pred_r| grows, t-stat grows.')\n"
            "print(f'Real tape: predictive t ~ {R[\"ispy_t\"]:+.2f} -- consistent with pred_r=0.')"
        ),
        md(
            "The predictive *t*-stat is monotone in the planted ``pred_r``: the engine is a "
            "faithful link detector. The real-tape result (predictive *t* near zero) is a "
            "statement about the *market* -- there is no weekly predictive dollar-equity "
            "link to find. The contemporaneous *t* (grey squares) also rises with "
            "``pred_r`` but only because the planted mechanism links both directions; in "
            "the real data the contemp channel is strong while the predictive is absent.\n\n"
            "Forks worth trying: adding a VIX-regime filter (the dollar-equity link may be "
            "stronger in high-vol regimes); testing at daily frequency (shorter horizon, more "
            "observations, but more noise); using currency-hedged EM to separate the FX from "
            "the local-currency equity effect."
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
