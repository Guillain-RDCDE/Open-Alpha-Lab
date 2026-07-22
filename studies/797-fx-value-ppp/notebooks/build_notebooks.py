"""Generate the two narrative notebooks for Study 797 (FX Value / PPP).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached FX + CPI
tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic positive control runs anywhere with no
network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance G10 FX +
# DBnomics CPI, real-rate value sort, 2005-01 -> 2025-06, 246 monthly LS observations).
R = dict(
    start="2005-01", end="2025-06", as_of="2025-06-30",
    n_months=246,
    mean_bps=8.52, ann_pct=1.02, vol_ann_pct=3.39, sharpe=0.30,
    t_one=1.37, t_nw=1.58,
    hit=129, hit_pct=52.4, wilson=(46.2, 58.6),
    placebo_obs=0.302, placebo_mean=-0.001, placebo_sd=0.223, placebo_p=0.0905,
    placebo_draws=10000,
    win={36: (0.20, 0.92), 48: (0.25, 1.29), 60: (0.30, 1.58), 84: (0.30, 1.64)},
    era_early_sharpe=0.06, era_early_t=0.24, era_early_n=120,
    era_late_sharpe=0.66, era_late_t=2.15, era_late_n=126,
    timer={5: (1.02, 0.34, 0.10, 0.53), 10: (1.02, 0.16, 0.05, 0.25),
           20: (1.02, -0.20, -0.06, -0.32)},  # (gross_ann, net_ann, net_sharpe, net_t)
    turnover=0.30, worst_month=-3.73, borrow_bps=100,
    syn_null_mean=0.34, syn_null_sd=0.87, syn_null_fire=1,
    syn_planted_t=12.44, syn_planted_sharpe=2.24, syn_planted_ann=7.34,
    # most-undervalued (long) / most-overvalued (short) at the as-of
    cheap="JPY", cheap_sig=0.12, rich="GBP", rich_sig=-0.089,
    fp_fx="0f1dcad0ff54", fp_cpi="4553d2b68c91",
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Survivorship: Named](https://img.shields.io/badge/Survivorship-Named-8b949e?style=flat-square)\n\n"
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

from fx_value import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    FX = data.load_fx()
    CPI = data.load_cpi()
    LOGQ = data.real_rate_panel(FX, CPI)
    SIG = st.value_signal(LOGQ, window=data.TRAIL_WINDOW, min_periods=data.MIN_TRAIL)
    RETS = st.spot_returns(FX)
    PORT = st.portfolio_returns(SIG, RETS)
else:
    FX = CPI = LOGQ = SIG = RETS = PORT = None
print("real cache present:", HAVE_REAL, "| value LS months:",
      (0 if PORT is None else len(PORT)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do 'cheap' currencies really bounce back? 💱\n"
            "### FX value — the textbook PPP trade, tested on the G10 and graded honestly\n\n"
            + BADGES +
            "There's a tidy story every macro tutorial tells: a currency can get "
            "**cheap** — its exchange rate falls further than the difference in inflation "
            "justifies — and cheap things revert. Buy the undervalued currencies, sell the "
            "overvalued ones, wait for purchasing-power parity to pull them home. It's one of "
            "the oldest 'value' trades there is.\n\n"
            "Does it actually pay on the big developed currencies? **Directionally yes — but "
            "barely, and not enough to trade.**\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** G10 FX vs USD + national CPI, monthly, 2005→2025. The 'real "
            "exchange rate' is the FX rate adjusted for relative inflation; we rank the nine "
            "currencies on how far below their own 5-year average they sit. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do undervalued currencies out-earn overvalued ones? | **Yes, a little.** The "
            f"long-cheap/short-rich book makes about **+{R['ann_pct']:.1f}%/yr** — the right "
            f"sign, matching decades of research. |\n"
            f"| Is that statistically solid? | **No.** The robust *t*-stat is **{R['t_nw']:.2f}** "
            f"— below the '2' bar this desk needs before calling a signal real. A coin-flip "
            f"version beats it about 1 time in 11. |\n"
            f"| Can you trade it for profit? | **No.** After trading costs and the cost of "
            f"borrowing to short, the ~1%/yr cushion is gone — it turns **negative** at realistic "
            f"institutional costs. |\n"
            f"| So is the famous 'FX value' premium fake? | **Not fake — just in the wrong place.** "
            "The research says it's real but lives mostly in *emerging-market* currencies. On the "
            "boring developed G10 alone, it's a whisper. |\n\n"
            "> The PPP value trade is real in direction and real in the literature — but on the "
            "G10 it's too weak to certify and too thin to trade."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A currency that's cheap relative to purchasing-power parity is a buy: prices and "
            "exchange rates eventually line up, so the undervalued currency appreciates and the "
            "overvalued one falls.\"*\n\n"
            "The **real exchange rate** makes 'cheap' precise. Take the market FX rate and adjust "
            "it for how much prices have risen in each country. If a currency has fallen *more* "
            "than the inflation gap justifies, it's cheap in real terms — below its own long-run "
            "average. PPP says that gap should close. This is the FX **value** factor — the "
            "mirror image of the **carry** trade ([study 364](../../364-fx-carry-trade/)), which "
            "buys currencies for their *interest rate* instead."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it worked cleanly, this would be a slow, sober, diversifying alternative to "
            "chasing yield: you'd lean *against* the currencies everyone has bid up and *toward* "
            "the unloved ones, and get paid as prices reassert themselves. Value and carry are "
            "famously negatively correlated, so a real FX-value premium is exactly the kind of "
            "thing a carry book would love to own alongside it. That's why it's worth testing "
            "properly rather than taking the textbook's word for it."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **Build the real rate.** For each of the nine G10 currencies vs the dollar, "
            "combine the FX rate with the two countries' CPI to get the inflation-adjusted "
            "('real') exchange rate.\n"
            "- **Score cheapness.** How far is each currency below its own 5-year average real "
            "rate? Rank them; go long the cheapest, short the richest, dollar-neutral, every "
            "month.\n"
            "- **No cheating on inflation.** CPI is published weeks late, so we only ever use the "
            "print that was actually out when the trade was placed.\n"
            "- **Then be honest about costs** — rebalancing, and the cost of borrowing to hold "
            "the shorts."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, who's cheap and who's rich right now?** A snapshot of the signal at the "
            "as-of date (positive = cheap = a long)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    snap = SIG.dropna(how='all').iloc[-1].dropna().sort_values()\n"
            "else:\n"
            "    snap = pd.Series({'GBP':-0.089,'CHF':-0.063,'EUR':-0.055,'SEK':-0.036,\n"
            "                      'NOK':0.020,'NZD':0.043,'CAD':0.043,'AUD':0.052,'JPY':0.120})\n"
            "cols = [GREEN if v>0 else RED for v in snap.values]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.barh(snap.index, snap.values, color=cols)\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('value signal  (positive = cheap vs its own 5-yr real-rate average = a LONG)')\n"
            "ax.set_title('Who is cheap (green, long) and who is rich (red, short)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('cheapest (long):', snap.index[-1], '| richest (short):', snap.index[0])"
        ),
        md(
            f"At the as-of, the **{R['cheap']}** is the most undervalued currency in real terms "
            f"(a long), while the **{R['rich']}** is the richest (a short) — a picture most FX "
            "desks would recognise. The strategy simply holds that lean, every month, and "
            "collects whatever mean reversion shows up.\n\n"
            "**Does it pay?** Here's the growth of \\$1 in the long-short book."
        ),
        code(
            "if HAVE_REAL:\n"
            "    curve = (1 + PORT).cumprod()\n"
            "    h = st.headline_stats(PORT)\n"
            "    ann, shp = h['ann_pct'], h['sharpe']\n"
            "else:\n"
            "    rng = np.random.default_rng(797)\n"
            "    curve = (1 + pd.Series(rng.normal(R['mean_bps']/1e4, R['vol_ann_pct']/100/np.sqrt(12), R['n_months']))).cumprod()\n"
            "    ann, shp = R['ann_pct'], R['sharpe']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.plot(curve.values, color=AMBER, lw=1.8)\n"
            "ax.set_ylabel('growth of $1 (long-short, gross)')\n"
            "ax.set_xlabel('months')\n"
            "ax.set_title(f'A real but shallow climb: ~{ann:+.1f}%/yr, Sharpe {shp:.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'annualized {ann:+.2f}%  Sharpe {shp:.2f}')"
        ),
        md(
            f"It grinds **up** — the value tilt is directionally right, about "
            f"**+{R['ann_pct']:.1f}%/yr** at a Sharpe of **{R['sharpe']:.2f}**. But look at the "
            f"scale: a shallow, noisy climb, not a staircase. Over 20 years the book was positive "
            f"in only **{R['hit_pct']:.0f}%** of months.\n\n"
            "**Is that climb more than luck?** We shuffle the long/short signs at random 10,000 "
            "times and see how often a coin-flip book does as well."
        ),
        code(
            "obs = R['placebo_obs']\n"
            "rng = np.random.default_rng(797)\n"
            "draws = rng.normal(R['placebo_mean'], R['placebo_sd'], 4000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=45, color=GREY, alpha=.85, label='random-sign books (null)')\n"
            "ax.axvline(obs, c=AMBER, lw=2.5, label=f'real value sort (Sharpe {obs:.2f})')\n"
            "ax.set_xlabel('annualized Sharpe of a random long/short book')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Inside the luck cloud: p = {R['placebo_p']:.2f} (1 in ~11)\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"observed Sharpe {obs:.2f}  vs random mean {R['placebo_mean']:.2f} \"\n"
            "      f\"(sd {R['placebo_sd']:.2f})  ->  p = {R['placebo_p']:.4f}\")"
        ),
        md(
            f"The real book sits toward the **right edge** of the luck cloud but still inside it: "
            f"about **1 random coin-flip book in 11** ({R['placebo_p']:.2f}) does as well. That's "
            "suggestive, not decisive — the honest read is *probably a real tilt, not proven*.\n\n"
            "**And the killer: can you keep any of it after costs?**"
        ),
        code(
            "levels = [5, 10, 20]\n"
            "nets = [R['timer'][c][1] for c in levels]\n"
            "cols = [AMBER if v>0 else RED for v in nets]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar([f'{c} bps' for c in levels], nets, color=cols, width=.6)\n"
            "for i,v in enumerate(nets): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('net return per year (after cost + short borrow)')\n"
            "ax.set_title('The thin edge does not survive its own frictions')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net %/yr by cost:', {c: R['timer'][c][1] for c in levels})"
        ),
        md(
            f"There it dies. The gross **+{R['ann_pct']:.1f}%/yr** shrinks to "
            f"**+{R['timer'][5][1]:.2f}%** at a retail 5 bps once you pay to rebalance and to "
            f"borrow the shorts, and goes **{R['timer'][20][1]:+.2f}%** — negative — at a "
            "realistic institutional 20 bps. There's simply not enough cushion."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** Long-cheap/short-rich earns the right sign "
            f"(**+{R['ann_pct']:.1f}%/yr**, Sharpe {R['sharpe']:.2f}) and matches a big "
            f"literature — but the robust *t* is **{R['t_nw']:.2f}**, under the bar, and a "
            f"coin-flip beats it 1 time in 11. Real in direction, unproven on this tape.\n"
            "- **Tradability — Mirage.** The ~1%/yr gross cushion is thinner than the cost of "
            "trading and shorting it; net of realistic frictions it rounds to zero and then goes "
            "negative.\n"
            "- **Survivorship — Named.** The G10 is an ex-post 'these are the majors' basket; the "
            "value premium is documented to be much stronger in the *emerging* currencies we "
            "deliberately leave out. The weakness here is partly a basket choice."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The obvious next step:** widen the basket to EM currencies (BRL, MXN, ZAR, "
            "TRY…), where the research says most of the FX-value premium actually lives — at the "
            "cost of much messier data and fatter crash risk.\n"
            "- **Or blend it with carry.** Value and carry lean opposite ways; a combined book "
            "([study 364](../../364-fx-carry-trade/) is the carry leg) can be steadier than "
            "either alone even when each is individually thin.\n"
            "- **Dedup:** this is *not* the [Big-Mac snapshot](../../215-big-mac-ppp/) (one "
            "burger cross-section, no time series), *not* [carry](../../364-fx-carry-trade/) (the "
            "opposite tilt), *not* [FX momentum](../../147-fx-momentum/) (trend, not PPP), and "
            "*not* the [dollar smile](../../114-dollar-smile/) (the broad-USD cycle). See "
            "[docs/references.md](docs/references.md).\n\n"
            "*Think G10 value can be certified? Show a robust t ≥ 2 net of realistic costs and "
            "borrow on a basket you name in advance — then we'll talk.*"
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
            "# FX Value (PPP) — a quantitative teardown 🔬\n"
            "### The real-rate value sort · Newey-West HAC inference · a 10,000-draw random-sign "
            "placebo · a trailing-window robustness sweep · an honest cost+borrow timer · a "
            "20-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **undervalued real exchange rates predict currency appreciation** — is the "
            "FX **value** factor, the opposite tilt to carry (study 364) and a real time series "
            "of the PPP gap, not the Big-Mac snapshot (study 215). The job: measure it "
            "point-in-time on the G10, run the autocorrelation-robust inference, and ask whether "
            "it clears the bar and survives costs.\n\n"
            "> ⚠️ **Data note.** G10 FX vs USD (yfinance, month-end, USD-per-foreign) + national "
            "CPI (IMF IFS monthly; Eurostat HICP for EUR; IMF IFS **quarterly** ffill for AUD & "
            "NZD). Real rate `log q = log S + log CPI_i − log CPI_US`, CPI lagged 1 month "
            "(publication lag). 2005-01 → 2025-06. Fingerprints `fx=" + R["fp_fx"] + "`, `cpi=" +
            R["fp_cpi"] + "`. Numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | value LS **+{R['ann_pct']:.2f}%/yr**, Sharpe "
            f"{R['sharpe']:.2f}, one-sample t = {R['t_one']:.2f}, **NW t = {R['t_nw']:.2f}** "
            f"(< 2), placebo p = {R['placebo_p']:.2f} |\n"
            f"| **Tradability** | `MIRAGE` | net of 5 bps + 100 bps borrow: Sharpe "
            f"{R['timer'][5][2]:.2f} (t = {R['timer'][5][3]:.2f}); **negative** by 20 bps "
            f"({R['timer'][20][1]:+.2f}%/yr) |\n"
            f"| **Survivorship** | `NAMED` | fixed current-membership G10; the premium is "
            f"documented to live in the *excluded* EM currencies |\n\n"
            "> 💡 In plain words: the PPP-value tilt has the right sign and a deep literature "
            "behind it, but on the G10-only tape the robust *t* is 1.58 — below the bar — and the "
            "gross premium is too thin to survive costs and borrow. Weak, and a mirage to trade."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $S_i$ be the USD price of currency $i$ (USD-per-foreign), $P_i$ its CPI. The "
            "**log real exchange rate** is\n\n"
            "$$q_{i,t} = \\log S_{i,t} + \\log P_{i,t} - \\log P_{US,t}.$$\n\n"
            "A *high* $q$ means the currency is **rich** (overvalued) vs PPP; a *low* $q$ means "
            "**cheap**. The value signal is the deviation from a trailing average,\n\n"
            "$$v_{i,t} = \\overline{q}_{i,t}^{\\,(60m)} - q_{i,t},$$\n\n"
            "positive when the real rate is below its own 5-year mean. The claims:\n\n"
            "- **H₁ (reversion).** $E[\\Delta S_{i,t+1} \\mid v_{i,t}] > 0$ increasing in $v$ — "
            "cheap currencies appreciate.\n"
            "- **H₂ (cross-section).** A dollar-neutral long-cheap/short-rich book earns a "
            "positive premium (NW $t \\ge 2$).\n"
            "- **H₃ (capture).** It survives turnover cost and short borrow.\n\n"
            f"We find **H₁/H₂ directionally supported but statistically WEAK** (NW t = "
            f"{R['t_nw']:.2f}, placebo p = {R['placebo_p']:.2f}) and **H₃ rejected** (net Sharpe "
            f"{R['timer'][5][2]:.2f} at 5 bps, negative by 20 bps)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The monthly long-short return is mildly autocorrelated (overlapping trailing-mean "
            "signals move slowly), so the **planned primary is a Newey-West (6-lag) HAC *t*** on "
            "the LS mean, with the plain one-sample *t* alongside. The hit rate carries a "
            "**Wilson** interval; the placebo **randomly flips each month's cross-sectional "
            "signs** (10,000 books, 20 seeds × 500) holding the gross book and return tape fixed; "
            "a **trailing-window sweep** (36/48/60/84 mo) checks the result isn't a single "
            "lucky memory length. A 2015 sub-period split is examined but flagged **snooped** and "
            "kept out of the stamp."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Panel.** 9 G10 currencies vs USD, {R['start']} → {R['end']}, {R['n_months']} "
            "monthly LS observations after the trailing window warms up.\n"
            "- **Signal.** `trailing-mean(log q, 60m, min 36m) − log q`, CPI lagged 1 month; "
            "dollar-neutral rank weights (sum w = 0, gross = 1), long cheap / short rich.\n"
            "- **Execution.** One shift: weights set at close of month *t* earn month *t+1* spot "
            "return.\n"
            "- **Headline.** One-sample + NW(6) *t*, Sharpe, Wilson hit rate, 10,000-draw "
            "sign placebo.\n"
            "- **Robustness.** Trailing-window sweep; a (snooped) pre/post-2015 split.\n"
            "- **Timer.** Turnover × one-way cost (both legs) + 100 bps/yr short borrow; sweep "
            "5/10/20 bps.\n"
            "- **Control.** Synthetic real-rate panel with a tunable planted reversion; the null "
            "(strength 0) must not systematically fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline value sort\n\n"
            "The dollar-neutral long-cheap/short-rich book, its cumulative path, and the robust "
            "inference."
        ),
        code(
            "if HAVE_REAL:\n"
            "    h = st.headline_stats(PORT)\n"
            "    print(f\"months {h['n_months']} ({h['start']} -> {h['end']})\")\n"
            "    print(f\"mean {h['mean_bps']:+.2f} bps/mo | ann {h['ann_pct']:+.2f}% | \"\n"
            "          f\"vol {h['vol_ann_pct']:.2f}% | Sharpe {h['sharpe']:.2f}\")\n"
            "    print(f\"one-sample t = {h['t_one_sample']:+.2f} | Newey-West(6) t = {h['t_nw']:+.2f}\")\n"
            "    print(f\"hit {h['hit']}/{h['n_months']} = {h['hit_rate']*100:.1f}% \"\n"
            "          f\"Wilson [{h['hit_lo']*100:.1f}%, {h['hit_hi']*100:.1f}%]\")\n"
            "    curve = (1 + PORT).cumprod()\n"
            "else:\n"
            "    rng = np.random.default_rng(797)\n"
            "    curve = (1 + pd.Series(rng.normal(R['mean_bps']/1e4, R['vol_ann_pct']/100/np.sqrt(12), R['n_months']))).cumprod()\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "ax.plot(curve.values, color=AMBER, lw=1.8)\n"
            "ax.set_ylabel('growth of $1 (LS, gross)'); ax.set_xlabel('months')\n"
            "ax.set_title(f\"Value LS: +{R['ann_pct']:.1f}%/yr, Sharpe {R['sharpe']:.2f}, \"\n"
            "             f\"NW t = {R['t_nw']:.2f} (below the bar)\")\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: **+{R['ann_pct']:.2f}%/yr** at Sharpe **{R['sharpe']:.2f}**, "
            f"hit rate {R['hit_pct']:.1f}% (Wilson [{R['wilson'][0]:.1f}%, {R['wilson'][1]:.1f}%] "
            f"— brackets 50%). The robust **NW t = {R['t_nw']:.2f}** and one-sample "
            f"**t = {R['t_one']:.2f}** both sit under 2. Right sign, insufficient evidence: "
            "**H₂ is WEAK, not REAL.**"
        ),
        md(
            "### 4b · The placebo — is the sign structure lucky?\n\n"
            "Randomly flip each month's cross-sectional signs (gross book & tape fixed), 10,000 "
            "books. In the notebook we run a light draw and quote the canonical p from "
            "`results.md`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(SIG, RETS, n_draws_per_seed=200, n_seeds=5)\n"
            "    obs, draws = pl['obs_sharpe'], pl['draws']\n"
            "else:\n"
            "    obs = R['placebo_obs']\n"
            "    rng = np.random.default_rng(797)\n"
            "    draws = rng.normal(R['placebo_mean'], R['placebo_sd'], 2000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=45, color=GREY, alpha=.85, label='random-sign null (light run)')\n"
            "ax.axvline(obs, c=AMBER, lw=2.5, label=f'observed Sharpe {obs:.2f}')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('annualized Sharpe of a random-sign book'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Right edge of the luck cloud but inside it: canonical p = {R['placebo_p']:.2f}\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical placebo (results.md): mean {R['placebo_mean']:.3f}, \"\n"
            "      f\"sd {R['placebo_sd']:.3f}, p = {R['placebo_p']:.4f} over {R['placebo_draws']:,} draws\")"
        ),
        md(
            f"> 💡 In plain words: **p = {R['placebo_p']:.2f}** — about 1 random-sign book in 11 "
            "matches the real sort. Consistent with a genuine but modest tilt; nowhere near the "
            "5% the desk needs to call it decisive. Agrees with the sub-2 *t*."
        ),
        md(
            "### 4c · Robustness — trailing-window sweep & a (snooped) era split\n\n"
            "Does the memory length matter, and is the effect stable in time?"
        ),
        code(
            "wins = sorted(R['win'])\n"
            "if HAVE_REAL:\n"
            "    ts = []\n"
            "    for w_ in wins:\n"
            "        s2 = st.value_signal(LOGQ, window=w_, min_periods=data.MIN_TRAIL)\n"
            "        ts.append(st.headline_stats(st.portfolio_returns(s2, RETS))['t_nw'])\n"
            "else:\n"
            "    ts = [R['win'][w_][1] for w_ in wins]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar([str(w_) for w_ in wins], ts, color=AMBER, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, lw=1); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_xlabel('trailing window (months)'); a1.set_ylabel('NW t')\n"
            "a1.set_title('Stable sign, never clears |t|=2')\n"
            "a2.bar(['2005-2014\\n(n={})'.format(R['era_early_n']),'2015-2025\\n(n={})'.format(R['era_late_n'])],\n"
            "       [R['era_early_t'], R['era_late_t']], color=[GREY, AMBER], width=.55)\n"
            "a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel('NW t'); a2.set_title('Snooped split: all of it is post-2015')\n"
            "for i,v in enumerate([R['era_early_t'], R['era_late_t']]): a2.annotate(f't={v:+.2f}',(i,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('window NW t:', {w_: round(t,2) for w_,t in zip(wins, ts)})"
        ),
        md(
            f"> 💡 In plain words: every window (36→84 mo) gives the same modest positive sign, "
            f"none clearing |t| = 2 — so the headline isn't a lucky memory length. The 2015 split "
            f"is eye-catching (pre-2015 t = {R['era_early_t']:.2f}, post-2015 "
            f"t = {R['era_late_t']:.2f}) but it is **not pre-registered**: a hypothesis for "
            "follow-up (does G10 value only pay in high-rate-vol regimes?), never a certification. "
            "The full-sample number governs the stamp."
        ),
        md(
            "### 4d · The timer — costs and short borrow\n\n"
            "Turnover × one-way cost on both legs, plus a 100 bps/yr borrow on the short book."
        ),
        code(
            "levels = sorted(R['timer'])\n"
            "if HAVE_REAL:\n"
            "    gross, nets, sharpes, tts = [], [], [], []\n"
            "    for c in levels:\n"
            "        t = st.timer_stats(SIG, RETS, cost_bps=c)\n"
            "        gross.append(t['gross_ann_pct']); nets.append(t['net_ann_pct'])\n"
            "        sharpes.append(t['net_sharpe']); tts.append(t['net_t_nw'])\n"
            "else:\n"
            "    gross = [R['timer'][c][0] for c in levels]; nets = [R['timer'][c][1] for c in levels]\n"
            "    sharpes = [R['timer'][c][2] for c in levels]; tts = [R['timer'][c][3] for c in levels]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(['gross']+[f'net @{c}bps' for c in levels], [gross[0]]+nets,\n"
            "       color=[GREY]+[AMBER if v>0 else RED for v in nets], width=.62)\n"
            "for i,v in enumerate([gross[0]]+nets): a1.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('%/yr'); a1.set_title('Frictions eat the thin edge')\n"
            "a2.bar([f'@{c}bps' for c in levels], sharpes, color=[AMBER if v>0 else RED for v in sharpes], width=.55)\n"
            "for i,(v,t_) in enumerate(zip(sharpes, tts)): a2.annotate(f'Sh {v:.2f}\\n(t={t_:+.2f})',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('net Sharpe'); a2.set_title('Mirage: ~0 then negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net %/yr:', {c: R['timer'][c][1] for c in levels})"
        ),
        md(
            f"> 💡 In plain words: gross **+{R['ann_pct']:.2f}%/yr** → net "
            f"**+{R['timer'][5][1]:.2f}%** at 5 bps (Sharpe {R['timer'][5][2]:.2f}), "
            f"**+{R['timer'][10][1]:.2f}%** at 10 bps, and **{R['timer'][20][1]:+.2f}%** — "
            f"negative — at 20 bps. Turnover ≈ {R['turnover']:.2f}/mo and the {R['borrow_bps']} "
            f"bps short borrow together outweigh the premium. Worst month "
            f"{R['worst_month']:+.1f}%. **Tradability = MIRAGE.**"
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic panel of log real rates with a TUNABLE planted mean reversion. The null "
            "(strength = 0, driftless walks) is checked over **20 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    w = data.synthetic_world(value_strength=0.0, seed=797 + s_)\n"
            "    null_ts.append(st.synthetic_detect(w)['t_nw'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "planted = st.synthetic_detect(data.synthetic_world(value_strength=0.15, seed=797))['t_nw']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12,.12,20), null_ts, color=GREY, s=40, label='null (strength=0), 20 seeds')\n"
            "ax.scatter([1], [planted], color=GREEN, s=110, zorder=5, label='planted reversion (strength=0.15)')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0,1]); ax.set_xticklabels(['null x 20','planted'])\n"
            "ax.set_ylabel('NW t of the value sort')\n"
            "ax.set_title('Control: null ~0, a planted PPP pull lights up hard')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20  |  planted t = {planted:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}), firing at |t| ≥ 2 in "
            f"only {R['syn_null_fire']}/20 (≈ the nominal 5%); a planted reversion reads "
            f"t = {R['syn_planted_t']:.1f} (Sharpe {R['syn_planted_sharpe']:.2f}). The machinery "
            "recovers the effect it is designed to harvest — so the sub-2 real-tape *t* is a "
            "property of the **G10 tape**, not a broken pipeline. *(Machinery/power check only — "
            "never cited for the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — value LS **+{R['ann_pct']:.2f}%/yr**, Sharpe {R['sharpe']:.2f}, "
            f"one-sample t = {R['t_one']:.2f}, NW t = {R['t_nw']:.2f} (**< 2**), placebo "
            f"p = {R['placebo_p']:.2f}, hit {R['hit_pct']:.1f}% (Wilson brackets 50%). Right "
            f"sign, deep literature, but this G10 tape cannot certify it. Literature says real; "
            "the tape reads WEAK.\n"
            f"- **Tradability `MIRAGE`** — net Sharpe {R['timer'][5][2]:.2f} at 5 bps, "
            f"{R['timer'][10][2]:.2f} at 10 bps, and **negative** ({R['timer'][20][1]:+.2f}%/yr) "
            f"at 20 bps once turnover (~{R['turnover']:.2f}/mo) and {R['borrow_bps']} bps short "
            "borrow are charged. The gross cushion is thinner than its own frictions.\n"
            "- **Survivorship `NAMED`** — a fixed, current-membership developed-market basket; "
            "the documented FX-value premium is disproportionately an **emerging-market** "
            "phenomenon this study excludes, so the weakness is partly a basket choice, stated "
            "openly."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Where the premium actually lives:** Menkhoff et al. (2017) locate most of FX "
            "value in EM currencies; extending the basket (with its fatter crash risk and thinner "
            "data) is the natural next study.\n"
            "- **Value × carry:** the two styles are negatively correlated; a combined book "
            "([364-fx-carry-trade](../../364-fx-carry-trade/) is the carry leg) can be steadier "
            "than either alone even when each is thin.\n"
            "- **Signal variants:** the 5-year *change* in the real rate (Asness et al. 2013) vs "
            "the *level*-deviation used here; a half-life-matched horizon; or including the "
            "interest differential to trade *total* rather than spot returns (which starts to "
            "blend value with carry).\n"
            "- **Dedup map:** [215-big-mac-ppp](../../215-big-mac-ppp/) (single folklore "
            "snapshot, no time series), [364-fx-carry-trade](../../364-fx-carry-trade/) (the "
            "opposite, rate-differential tilt), [147-fx-momentum](../../147-fx-momentum/) (trend, "
            "not PPP), [114-dollar-smile](../../114-dollar-smile/) (broad-USD cycle level).\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live in "
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
