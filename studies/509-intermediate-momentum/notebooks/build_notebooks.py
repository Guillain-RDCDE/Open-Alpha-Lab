"""Generate the two narrative notebooks for Study 509 (Intermediate-Momentum).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
control runs anywhere, offline and deterministic; the real-panel cells use the cached
yfinance parquets under ../_cache/ if present and otherwise quote the frozen headline
numbers in ``R`` (a mirror of docs/results.md, as-of 2026-06-26).
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


# Frozen real-panel headline numbers -- mirror of docs/results.md (as-of 2026-06-26).
R = dict(
    n_tickers=46,
    n_days=4780,
    fp_prices="e21fc3ab636a", fp_spy="11cdeebb16d8",
    period_start="2007-01-03", period_end="2025-12-31",
    # intermediate long-short (t-12..t-7)
    int_n=215, int_mean=-0.31, int_vol=11.46, int_sharpe=-0.027, int_t=-0.121,
    int_hit=48.8, int_dd=-48.4, int_turn=28.7, int_net=-1.50, int_net_t=-0.59,
    int_placebo=0.565, int_real_mo=-0.026,
    # recent long-short (t-6..t-1)
    rec_n=221, rec_mean=0.25, rec_vol=12.17, rec_sharpe=0.021, rec_t=0.090,
    rec_hit=54.3, rec_dd=-52.9, rec_turn=28.6, rec_net=-0.94, rec_net_t=-0.34,
    rec_placebo=0.400, rec_real_mo=0.021,
    # synthetic control: planted-premium -> (int_mean, int_t, rec_mean, rec_t)
    control={
        0.00: (0.66, 0.27, -2.90, -1.30),
        0.06: (7.74, 3.37, -3.15, -1.33),
        0.12: (14.29, 6.26, -4.15, -1.78),
    },
    # year-by-year intermediate long-short (gross %)
    annual={
        2008: 15.1, 2009: -14.2, 2010: 12.7, 2011: 18.4, 2012: 4.1,
        2013: -8.0, 2014: -5.3, 2015: 6.6, 2016: -6.3, 2017: 14.4,
        2018: -15.4, 2019: -5.4, 2020: -14.7, 2021: -12.4, 2022: -5.7,
        2023: -0.1, 2024: 10.8, 2025: -1.4,
    },
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from intermediate_momentum import data, strategy as st

def _have_cache():
    study = os.path.abspath("..")
    return (os.path.exists(os.path.join(study, "_cache", "im_prices.parquet"))
            and os.path.exists(os.path.join(study, "_cache", "im_spy.parquet")))

HAVE_REAL = _have_cache()
print("yfinance cache present:", HAVE_REAL)

if HAVE_REAL:
    prices, spy = data.fetch_panel()
    inter  = st.momentum_longshort(prices, window="intermediate", min_stocks=18)
    recent = st.momentum_longshort(prices, window="recent", min_stocks=18)
    s_int  = st.summary(inter["ls_ret"])
    s_rec  = st.summary(recent["ls_ret"])
    print(f"INT mean {s_int['mean']*100:+.2f}%/yr t={s_int['tstat']:+.3f} | "
          f"REC mean {s_rec['mean']*100:+.2f}%/yr t={s_rec['tstat']:+.3f}")
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Intermediate-Momentum -- does momentum live in the *intermediate* past?\n"
            "### Novy-Marx (2012): the t-12..t-7 window vs the t-6..t-1 window, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Intermediate%20wins%3F: Busted](https://img.shields.io/badge/Intermediate%20wins%3F-Busted-8b949e?style=flat-square)\n\n"
            "Cross-sectional momentum -- buy past winners, sell past losers -- usually ranks "
            "stocks on their return over the last twelve months (skipping the most recent one). "
            "In 2012, Robert Novy-Marx asked a sharp question: *which part* of that window does "
            "the work? He found the premium comes from the **intermediate** past -- roughly "
            "twelve to seven months ago -- and **not** the recent past (the months just before, "
            "which we implement as t-6..t-1). He called it momentum's 'echo'.\n\n"
            "We test that decomposition on a large-cap survivor basket using yfinance daily "
            "prices: build one long-short ranked on the intermediate window, one on the recent "
            "window, and see which (if either) carries the drift.\n\n"
            "> **This is the plain-language layer.** Want the formation windows, HAC *t*, the "
            "placebo and the cost ledger? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT ----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the intermediate window carry a drift? | **No, not here.** "
            f"Long-short **{R['int_mean']:+.2f}%/yr**, HAC *t* = **{R['int_t']:+.2f}** "
            "-- inside its placebo distribution. |\n"
            "| Does the recent window? | **Also no.** "
            f"**{R['rec_mean']:+.2f}%/yr**, *t* = **{R['rec_t']:+.2f}**. |\n"
            "| Does intermediate beat recent (Novy-Marx's point)? | **No** -- the intermediate "
            "sort is, if anything, **marginally worse**. The echo is **Busted** on this basket. |\n"
            "| Is it tradable? | **Mirage.** Both sorts go **negative net** of costs+borrow "
            f"({R['int_net']:+.2f}% / {R['rec_net']:+.2f}%/yr). |\n\n"
            "> Novy-Marx (2012) documented the intermediate effect on a broad CRSP cross-section. "
            "On 46 large-cap survivors it does not reproduce -- both windows are flat."
        ),

        # ---- BEAT 1 -- THE CLAIM --------------------------------------------
        md(
            "## 1 -- The claim\n\n"
            "> *\"Momentum is not really momentum. The return that predicts next month's "
            "cross-section is the return from twelve to seven months ago -- the intermediate "
            "horizon -- not the recent six months.\"*\n\n"
            "-- Novy-Marx (2012), *Journal of Financial Economics*\n\n"
            "Standard momentum ranks stocks on their past 12-month return (skipping month t-1, "
            "where short-term reversal lives). Novy-Marx split that window in two and showed the "
            "predictive power sat almost entirely in the older, intermediate slice. That is "
            "awkward for behavioural 'underreaction' stories, which would expect the *recent* "
            "news to drive the drift."
        ),

        # ---- BEAT 2 -- SO WHAT ----------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "If true, the result reshapes how you build a momentum signal (rank on the "
            "intermediate window, ignore the recent one) and it constrains the theories that "
            "can explain momentum at all. It became one of the most-cited momentum refinements "
            "of the 2010s.\n\n"
            "Our questions:\n"
            "1. **Does either window carry a drift on a realistic large-cap panel?**\n"
            "2. **Does the intermediate window beat the recent one, as claimed?**\n"
            "3. **Does anything survive costs?**"
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ----------------------------------------
        md(
            "## 3 -- How would we even know?\n\n"
            "Three disciplines keep us honest:\n\n"
            "1. **One execution lag.** The signal is computed at month-end; we enter one "
            "trading day later. No look-ahead, no same-bar fills.\n"
            "2. **A label-shuffle placebo.** We randomly permute which stocks are 'winners' vs "
            "'losers' and re-run -- if the real labels carry no information, the real mean sits "
            "inside the shuffle distribution.\n"
            "3. **Name the survivorship bias.** The basket is current large-caps projected "
            "backwards. Past losers that delisted -- the natural short leg -- are absent, so any "
            "premium here is an upper bound."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -----------------------------------------
        md(
            "## 4 -- The teardown\n\n"
            "**First, does the engine work when the effect is planted?** We build a synthetic "
            "world where a momentum premium exists *only* in the intermediate window."
        ),
        code(
            "premiums = [0.0, 0.06, 0.12]\n"
            "rows = []\n"
            "for prem in premiums:\n"
            "    ps, sp, truth = data.synthetic_panel(mom_premium=prem, seed=509)\n"
            "    ri = st.momentum_longshort(ps, window='intermediate', min_stocks=10)\n"
            "    rr = st.momentum_longshort(ps, window='recent', min_stocks=10)\n"
            "    si = st.summary(ri['ls_ret']); sr = st.summary(rr['ls_ret'])\n"
            "    rows.append({'premium': prem,\n"
            "                 'int_mean_%': si['mean']*100, 'int_t': si['tstat'],\n"
            "                 'rec_mean_%': sr['mean']*100, 'rec_t': sr['tstat']})\n"
            "ctrl = pd.DataFrame(rows)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "x = np.arange(len(premiums)); w = 0.38\n"
            "ax.bar(x - w/2, ctrl['int_mean_%'], w, color=AMBER, label='intermediate sort')\n"
            "ax.bar(x + w/2, ctrl['rec_mean_%'], w, color=GREY, label='recent sort')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels([str(p) for p in premiums])\n"
            "ax.set_xlabel('planted intermediate-window premium')\n"
            "ax.set_ylabel('long-short mean (%/yr)')\n"
            "ax.set_title('Synthetic control: only the intermediate sort lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),
        md(
            "The engine is faithful: it finds the planted drift in the **intermediate** sort "
            "(growing with the premium) and leaves the **recent** sort flat. So the verdict on "
            "the real tape reflects the market, not a broken detector."
        ),
        md("**Now the honest test on the real yfinance panel.**"),
        code(
            "if HAVE_REAL:\n"
            "    int_m, int_t = s_int['mean']*100, s_int['tstat']\n"
            "    rec_m, rec_t = s_rec['mean']*100, s_rec['tstat']\n"
            "else:\n"
            "    int_m, int_t = R['int_mean'], R['int_t']\n"
            "    rec_m, rec_t = R['rec_mean'], R['rec_t']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.5))\n"
            "labels = ['Intermediate\\n(t-12..t-7)', 'Recent\\n(t-6..t-1)']\n"
            "vals = [int_m, rec_m]\n"
            "cols = [RED if abs(t) < 2 else GREEN for t in [int_t, rec_t]]\n"
            "bars = ax.bar(labels, vals, color=cols, width=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b, t in zip(bars, [int_t, rec_t]):\n"
            "    ax.text(b.get_x()+b.get_width()/2, b.get_height(),\n"
            "            f't={t:+.2f}', ha='center',\n"
            "            va='bottom' if b.get_height() >= 0 else 'top', fontsize=10)\n"
            "ax.set_ylabel('long-short mean return (%/yr, gross)')\n"
            "ax.set_title('Real panel: neither window carries a drift')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Intermediate: {int_m:+.2f}%/yr (t={int_t:+.2f})')\n"
            "print(f'Recent:       {rec_m:+.2f}%/yr (t={rec_t:+.2f})')"
        ),
        md(
            f"On the real basket the intermediate long-short earns **{R['int_mean']:+.2f}%/yr** "
            f"(HAC *t* = **{R['int_t']:+.2f}**) and the recent one **{R['rec_mean']:+.2f}%/yr** "
            f"(*t* = **{R['rec_t']:+.2f}**) -- both indistinguishable from zero, and the "
            "intermediate sort is marginally the **worse** of the two. Novy-Marx's ordering "
            "does not reproduce here."
        ),

        # ---- BEAT 5 -- VERDICT ----------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- NONE.** Intermediate *t* = {R['int_t']:+.2f}, recent "
            f"*t* = {R['rec_t']:+.2f}; both inside their placebo distributions "
            f"(p = {R['int_placebo']:.2f} / {R['rec_placebo']:.2f}). Neither clears |t| >= 2.\n"
            f"- **Tradability -- MIRAGE.** Net of costs+borrow, both go negative "
            f"({R['int_net']:+.2f}% / {R['rec_net']:+.2f}%/yr).\n"
            "- **Intermediate wins? -- BUSTED (here).** The intermediate sort does not beat the "
            "recent one; the horizon decomposition dissolves on a large-cap survivor basket."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 -- Could you actually trade it?\n\n"
            "Even setting the flat signal aside, the obstacles are familiar:\n\n"
            "1. **Turnover.** Monthly rank changes drive ~29% one-sided turnover; costs alone "
            "push an already-flat sort negative.\n"
            "2. **Short borrow.** The short leg (past losers) accrues borrow fees, and the "
            "deepest losers are the hardest to borrow.\n"
            "3. **Survivorship.** The very names that would have made the short leg pay -- past "
            "losers that delisted -- are not in a survivor basket. The real-world short leg is "
            "harder and costlier than the backtest shows."
        ),

        # ---- BEAT 7 -- GOING FURTHER ----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Broader universe.** Novy-Marx used the full CRSP cross-section; the echo is "
            "strongest in smaller names. A Russell 3000 pull (with delisted names) would be the "
            "fair test.\n"
            "- **International.** Goyal & Wahal (2015) find the echo is largely a US "
            "phenomenon -- weak or absent abroad.\n"
            "- **[Study 24 -- Stampede](../../24-stampede/)**: plain 12-1 momentum -- the parent "
            "effect this study decomposes.\n"
            "- **[Study 237 -- Residual-Momentum](../../237-residual-momentum/)**: another "
            "attempt to find a cleaner momentum signal.\n\n"
            "*Think the intermediate echo survives in a broad, delisting-complete universe net "
            "of costs? Fork this, expand the basket, add delisted names, and show t > 2 on the "
            "intermediate sort. That is the bar.*"
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
            "# Intermediate-Momentum -- a quantitative teardown\n"
            "### yfinance panel * formation-window sort * HAC inference * label-shuffle placebo\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Intermediate%20wins%3F: Busted](https://img.shields.io/badge/Intermediate%20wins%3F-Busted-8b949e?style=flat-square)\n\n"
            "The quantitative companion to the "
            "[notebook for the curious](01_for_the_curious.ipynb) -- same seven beats, every "
            "claim carrying its standard error. We test Novy-Marx (2012): two monthly "
            "equal-weight dollar-neutral tercile long-shorts, one ranked on the intermediate "
            "window (t-12..t-7), one on the recent window (t-6..t-1), one execution lag, costs "
            "+ borrow, HAC *t*, and a label-shuffle placebo.\n\n"
            f"> **Not investment advice.** Real data: yfinance daily adjusted-close, "
            f"{R['n_tickers']} large-caps, {R['period_start']}--{R['period_end']}, "
            "as-of 2026-06-26 (fp `e21fc3ab636a`). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship bias is named:** universe = current large-caps projected "
            "backwards; all results are upper-bound estimates."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT ----------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Intermediate **{R['int_mean']:+.2f}%/yr** "
            f"(HAC *t* = **{R['int_t']:+.3f}**), recent **{R['rec_mean']:+.2f}%/yr** "
            f"(*t* = **{R['rec_t']:+.3f}**); placebo p = {R['int_placebo']:.2f}/"
            f"{R['rec_placebo']:.2f}. |\n"
            f"| **Tradability** | `MIRAGE` | Net of costs+borrow: intermediate "
            f"**{R['int_net']:+.2f}%/yr**, recent **{R['rec_net']:+.2f}%/yr** "
            f"(~{R['int_turn']:.0f}% turnover). |\n"
            "| **Intermediate wins?** | `BUSTED` | The intermediate sort does **not** beat the "
            "recent one here -- it is marginally worse. |\n\n"
            "> The Novy-Marx (2012) echo is documented on a broad CRSP cross-section. On 46 "
            "large-cap survivors it does not reproduce."
        ),

        # ---- BEAT 1 -- THE CLAIM --------------------------------------------
        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $r^{\\text{int}}_{i,t}$ = stock $i$'s cumulative return over months "
            "$[t-12, t-7)$ and $r^{\\text{rec}}_{i,t}$ over $[t-6, t-1)$. Novy-Marx (2012) "
            "asserts:\n\n"
            "- **H1 (intermediate carries the drift).** A long-short on $r^{\\text{int}}$ "
            "earns a positive premium: $\\alpha^{\\text{int}} > 0$.\n"
            "- **H2 (recent does not).** A long-short on $r^{\\text{rec}}$ earns little: "
            "$\\alpha^{\\text{rec}} \\approx 0$, contaminated by short-term reversal.\n"
            "- **H3 (ordering).** $\\alpha^{\\text{int}} > \\alpha^{\\text{rec}}$.\n\n"
            "On this basket we **reject H1** (intermediate t = -0.12), the recent sort is also "
            "flat (consistent with the *spirit* of H2 but not as 'recent < intermediate'), and "
            "we **reject H3** -- the intermediate sort is marginally the worse of the two."
        ),

        # ---- BEAT 2 -- SO WHAT ----------------------------------------------
        md(
            "## 2 -- So what? -- the economic stakes\n\n"
            "The horizon decomposition is one of the most-cited momentum refinements of the "
            "2010s and shapes how systematic desks build the signal. If it does not survive on "
            "tradable large-caps net of costs, the practical edge it implies is far smaller "
            "than the broad-universe academic result suggests."
        ),

        # ---- BEAT 3 -- PROTOCOL ---------------------------------------------
        md(
            "## 3 -- The protocol\n\n"
            "- **Signal.** At each month-end, rank the cross-section on the chosen "
            "formation-window return (intermediate [t-12,t-7) or recent [t-6,t-1)).\n"
            "- **Legs.** Long top tercile, short bottom tercile, equal-weight, dollar-neutral.\n"
            "- **Execution lag.** Enter one trading day after the signal; hold one calendar "
            "month.\n"
            "- **Costs.** 5 bps per leg-side on turnover (enter + exit, two legs) + 50 bps "
            "annual borrow on the short leg.\n"
            "- **Inference.** Newey-West HAC *t* on the monthly long-short; a 200-shuffle "
            "label-shuffle placebo.\n"
            "- **Universe caveat.** 46 large-cap survivors (yfinance); survivorship-biased."
        ),

        # ---- BEAT 4 -- TEARDOWN ---------------------------------------------
        md("## 4 -- The teardown"),
        md(
            "### 4a -- Positive control: the engine is a faithful detector\n\n"
            "Plant the premium only in the intermediate window and sweep its size. The "
            "intermediate long-short's *t* should rise monotonically; the recent one should "
            "stay flat."
        ),
        code(
            "premiums = [0.0, 0.03, 0.06, 0.09, 0.12]\n"
            "int_means, int_ts, rec_means = [], [], []\n"
            "for prem in premiums:\n"
            "    ps, sp, _ = data.synthetic_panel(mom_premium=prem, seed=509)\n"
            "    ri = st.momentum_longshort(ps, window='intermediate', min_stocks=10)\n"
            "    rr = st.momentum_longshort(ps, window='recent', min_stocks=10)\n"
            "    si = st.summary(ri['ls_ret']); sr = st.summary(rr['ls_ret'])\n"
            "    int_means.append(si['mean']*100); int_ts.append(si['tstat'])\n"
            "    rec_means.append(sr['mean']*100)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(premiums, int_means, 'o-', c=AMBER, lw=2, label='intermediate mean')\n"
            "ax.plot(premiums, rec_means, 's--', c=GREY, lw=2, label='recent mean')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for p, m, t in zip(premiums, int_means, int_ts):\n"
            "    ax.text(p, m, f't={t:+.1f}', ha='center', va='bottom', fontsize=8)\n"
            "ax.set_xlabel('planted intermediate-window premium')\n"
            "ax.set_ylabel('long-short mean (%/yr)')\n"
            "ax.set_title('Positive control: intermediate t is monotone in the planted premium')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            "### 4b -- Real panel: the two long-shorts side by side"
        ),
        code(
            "if HAVE_REAL:\n"
            "    int_m, int_t, int_s = s_int['mean']*100, s_int['tstat'], s_int['sharpe']\n"
            "    rec_m, rec_t, rec_s = s_rec['mean']*100, s_rec['tstat'], s_rec['sharpe']\n"
            "    eq_int = (1 + inter['ls_ret']).cumprod()\n"
            "    eq_rec = (1 + recent['ls_ret']).cumprod()\n"
            "else:\n"
            "    int_m, int_t, int_s = R['int_mean'], R['int_t'], R['int_sharpe']\n"
            "    rec_m, rec_t, rec_s = R['rec_mean'], R['rec_t'], R['rec_sharpe']\n"
            "    eq_int = eq_rec = None\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "labels = ['Intermediate', 'Recent']\n"
            "vals = [int_m, rec_m]\n"
            "cols = [RED if abs(t) < 2 else GREEN for t in [int_t, rec_t]]\n"
            "bars = axes[0].bar(labels, vals, color=cols, width=0.5)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "for b, t in zip(bars, [int_t, rec_t]):\n"
            "    axes[0].text(b.get_x()+b.get_width()/2, b.get_height(), f't={t:+.2f}',\n"
            "                 ha='center', va='bottom' if b.get_height()>=0 else 'top')\n"
            "axes[0].set_ylabel('mean (%/yr, gross)')\n"
            "axes[0].set_title('Neither window clears |t|>=2')\n"
            "if eq_int is not None:\n"
            "    axes[1].plot(eq_int.index, eq_int.values, c=AMBER, lw=2, label='intermediate')\n"
            "    axes[1].plot(eq_rec.index, eq_rec.values, c=GREY, lw=2, ls='--', label='recent')\n"
            "    axes[1].axhline(1, c='k', lw=1, ls=':')\n"
            "    axes[1].set_ylabel('cumulative wealth (long-short)')\n"
            "    axes[1].set_title('Equity curves: signless drift, deep drawdowns')\n"
            "    axes[1].legend()\n"
            "else:\n"
            "    axes[1].axis('off'); axes[1].text(0.5,0.5,'cache absent', ha='center')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Intermediate: {int_m:+.2f}%/yr | Sharpe {int_s:+.3f} | HAC t {int_t:+.3f}')\n"
            "print(f'Recent:       {rec_m:+.2f}%/yr | Sharpe {rec_s:+.3f} | HAC t {rec_t:+.3f}')"
        ),
        md(
            f"> Intermediate **{R['int_mean']:+.2f}%/yr** (t = {R['int_t']:+.3f}), recent "
            f"**{R['rec_mean']:+.2f}%/yr** (t = {R['rec_t']:+.3f}). Both flat; the equity "
            "curves wander with ~48--53% max drawdowns and no net drift."
        ),
        md(
            "### 4c -- The label-shuffle placebo\n\n"
            "Permute which stocks are winners vs losers at every rebalance (200 shuffles) and "
            "rebuild. Where does the real mean fall in the shuffle distribution?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    plac_int = st.placebo_pvalue(prices, window='intermediate',\n"
            "                                 n_shuffles=200, min_stocks=18)\n"
            "    plac_rec = st.placebo_pvalue(prices, window='recent',\n"
            "                                 n_shuffles=200, min_stocks=18)\n"
            "    pi, pr = plac_int['p_value'], plac_rec['p_value']\n"
            "    ri_mo = plac_int['real_mean']*100; rr_mo = plac_rec['real_mean']*100\n"
            "else:\n"
            "    pi, pr = R['int_placebo'], R['rec_placebo']\n"
            "    ri_mo, rr_mo = R['int_real_mo'], R['rec_real_mo']\n"
            "print(f'Intermediate: real {ri_mo:+.3f}%/mo | placebo p = {pi:.3f}')\n"
            "print(f'Recent:       real {rr_mo:+.3f}%/mo | placebo p = {pr:.3f}')\n"
            "print('Both real sorts sit inside their shuffle distributions -> no real signal.')"
        ),
        md(
            "### 4d -- Net of costs"
        ),
        code(
            "if HAVE_REAL:\n"
            "    net_int = st.apply_costs(inter); net_rec = st.apply_costs(recent)\n"
            "    si_n = st.summary(net_int); sr_n = st.summary(net_rec)\n"
            "    gi, gr = s_int['mean']*100, s_rec['mean']*100\n"
            "    ni, nr = si_n['mean']*100, sr_n['mean']*100\n"
            "else:\n"
            "    gi, gr = R['int_mean'], R['rec_mean']\n"
            "    ni, nr = R['int_net'], R['rec_net']\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "x = np.arange(2); w = 0.38\n"
            "ax.bar(x - w/2, [gi, gr], w, color=GREY, label='gross')\n"
            "ax.bar(x + w/2, [ni, nr], w, color=RED, label='net of costs+borrow')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['Intermediate', 'Recent'])\n"
            "ax.set_ylabel('mean (%/yr)')\n"
            "ax.set_title('Costs push both flat sorts negative')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Intermediate: gross {gi:+.2f}%/yr -> net {ni:+.2f}%/yr')\n"
            "print(f'Recent:       gross {gr:+.2f}%/yr -> net {nr:+.2f}%/yr')"
        ),
        md(
            "### 4e -- Year-by-year intermediate long-short"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ic = inter.copy(); ic['year'] = ic.index.year\n"
            "    annual = ic.groupby('year')['ls_ret'].apply(lambda x: float((1+x).prod()-1))*100\n"
            "else:\n"
            "    annual = pd.Series(R['annual'])\n"
            "fig, ax = plt.subplots(figsize=(11, 4.0))\n"
            "col = [GREEN if v > 0 else RED for v in annual.values]\n"
            "ax.bar(annual.index.astype(int), annual.values, color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('year'); ax.set_ylabel('intermediate long-short (%/yr)')\n"
            "ax.set_title('A signless ride: strong crisis years, weak post-2017 drought')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(annual.round(1).to_string())"
        ),

        # ---- BEAT 5 -- VERDICT ----------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `NONE`** -- intermediate HAC *t* = {R['int_t']:+.3f}, recent "
            f"*t* = {R['rec_t']:+.3f}; both inside placebo (p = {R['int_placebo']:.2f}/"
            f"{R['rec_placebo']:.2f}). Novy-Marx (2012) documents the echo on a broad CRSP "
            "universe; it does not survive on 46 large-cap survivors.\n"
            f"- **Tradability `MIRAGE`** -- net of costs+borrow both go negative "
            f"({R['int_net']:+.2f}% / {R['rec_net']:+.2f}%/yr) at ~{R['int_turn']:.0f}% "
            "turnover. Nothing to monetise.\n"
            "- **Intermediate wins? `BUSTED`** -- the intermediate sort is marginally **worse** "
            "than the recent one; the horizon ordering reverses on this basket."
        ),

        # ---- BEAT 6 -- TRADABILITY ------------------------------------------
        md(
            "## 6 -- Could you trade it?\n\n"
            "1. **Flat signal.** There is no gross drift to harvest in the first place.\n"
            "2. **Turnover + borrow.** ~29% monthly turnover and short borrow turn a flat sort "
            "into a small bleed.\n"
            "3. **Survivorship.** The deepest past losers -- the short leg's would-be engine -- "
            "delisted and are absent; the real short leg is harder and dearer.\n"
            "4. **Crowding + decay.** A heavily-published momentum refinement is a textbook "
            "McLean-Pontiff (2016) post-publication decay candidate."
        ),

        # ---- BEAT 7 -- GOING FURTHER ----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Broad, delisting-complete universe.** Novy-Marx used all of CRSP; the echo is "
            "strongest in smaller, riskier names absent from a large-cap basket.\n"
            "- **International.** Goyal & Wahal (2015): the echo is largely US-specific.\n"
            "- **[Study 24 -- Stampede](../../24-stampede/)**: plain 12-1 momentum -- the parent "
            "effect.\n"
            "- **[Study 237 -- Residual-Momentum](../../237-residual-momentum/)**: a different "
            "cleaner-momentum attempt on the same infrastructure.\n\n"
            "*Think the intermediate echo survives in a broad, delisting-complete universe net "
            "of costs? Fork this, expand the basket with delisted names, and show t > 2 on the "
            "intermediate sort. That is the bar.*"
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
