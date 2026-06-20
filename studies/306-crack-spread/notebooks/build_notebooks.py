"""Generate the two narrative notebooks for Study 306 (Crack-Spread).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached daily
parquet under ../_cache/ if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader. Synthetic cells
are clearly bannered as synthetic; real numbers live only in ``R``.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-17,
# tape through 2026-05-29, fingerprint 40b398a38f3b).
R = dict(
    n_days=3549, fp="40b398a38f3b", crack_mean=22.6, crack_min=2.4, crack_max=68.8,
    # predictive vs coincident regression
    pred_slope=-0.00024, pred_t=-0.51, pred_r2=0.0001, pred_n=3547,
    coin_slope=0.00311, coin_t=2.06, coin_r2=0.0205, coin_n=3548,
    pred_pre_t=1.72, pred_post_t=-1.24,
    # timers vs buy-and-hold (gross, active = timer - B&H)
    bh_ann=20.3, bh_sharpe=0.54,
    lvl_ann=18.0, lvl_sharpe=0.83, lvl_active=-1.92, lvl_active_t=-0.25, lvl_pim=45, lvl_sw=193,
    mom20_ann=20.0, mom20_sharpe=0.76, mom20_active=-0.27, mom20_active_t=-0.04, mom20_pim=50, mom20_sw=370,
    mom1_ann=22.4, mom1_sharpe=0.82, mom1_active=1.74, mom1_active_t=0.27, mom1_pim=51, mom1_sw=1751,
    # cost sweep (20-day momentum timer, active)
    cost0=-0.27, cost0_t=-0.04, cost2=-0.80, cost2_t=-0.13,
    cost5=-1.57, cost5_t=-0.26, cost10=-2.86, cost10_t=-0.47,
    # bootstrap CIs
    mom20_ci_lo=0.19, mom20_ci_hi=1.38, bh_ci_lo=-0.02, bh_ci_hi=1.13,
    # synthetic positive control
    syn_null_pred_t=-1.28, syn_null_active=1.9, syn_null_active_t=0.28,
    syn_lead_pred_t=17.40, syn_lead_active=110.5, syn_lead_active_t=9.92,
)


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

from crack_spread import data, strategy as st

AS_OF_CUTOFF = "2026-06-01"   # drop the partial June 2026 month

def _have_cache():
    return os.path.exists(data._cache_path("CL=F", data.DEFAULT_CACHE))

HAVE_REAL = _have_cache()

def real_tape():
    f = data.load_real(fetch=False)
    return f[f.index < AS_OF_CUTOFF]

print("real crack/refiner cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Crack-Spread — can a refiner's margin predict its stock?\n"
            "### A classic energy-desk trade, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Coincident_%E2%89%A0_predictive%3F: Confirmed](https://img.shields.io/badge/Coincident_%E2%89%A0_predictive%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The **crack spread** is a refinery's profit in one number: the value of the gasoline "
            "and heating oil it makes, minus the crude it buys. Energy traders love the idea that a "
            "**fat or rising crack spread predicts refiner stocks** — Valero, Marathon, Phillips 66 "
            "— so you can *time* them off the margin. This notebook asks the only two questions that "
            "matter: *does the crack actually forecast the stock, and could you make money trying?*\n\n"
            "> **This is the plain-language layer.** Want the t-stats, the regressions and the "
            "cost math? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the crack move with refiner stocks? | **Yes — but only *today*.** Same-day, the "
            f"crack and the basket are clearly linked (*t* = +{R['coin_t']:.2f}). That's expected: "
            "the crack *is* the margin. |\n"
            "| Does it *predict* tomorrow? | **No.** Yesterday's crack change has essentially zero "
            f"power over today's return (*t* = {R['pred_t']:.2f}, R² ≈ 0.0001) — pure noise. |\n"
            "| Can you time refiners off it? | **No.** No crack-timer beats just buying and holding "
            f"the basket; the best one adds +{R['mom1_active']:.1f}%/yr gross (*t* = "
            f"+{R['mom1_active_t']:.2f}) and turns negative the moment you pay costs. |\n"
            "| So what is it? | A perfect mirror of refiner profits **right now** — which is exactly "
            "why it can't see the future. A mirror is not a crystal ball. |\n\n"
            "> The crack spread is genuinely useful — for *hedging* a refiner's margin, not for "
            "*timing* its stock. The same-day link fools people into thinking they've found a signal."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When refining margins are fat, refiners print money — so a wide or rising crack "
            "spread is a green light to be long Valero, Marathon and Phillips 66.\"*\n\n"
            "The recipe is the **3-2-1 crack**: take 3 barrels of crude, and the products are worth "
            "roughly 2 barrels of gasoline plus 1 of heating oil. In dollars per barrel:\n\n"
            "$$\\text{crack} = \\frac{2\\times(\\text{gasoline}\\times 42) + (\\text{heating oil}\\times 42)}{3} - \\text{crude}$$\n\n"
            "When that number is high, the refiner's margin is high. The bet is that you can read "
            "the margin *today* and trade the stock *tomorrow*."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it worked, it would be a clean, intuitive sector-timing tool — a public, free number "
            "(crude and product futures trade all day) that front-runs an entire industry. That's "
            "exactly the kind of 'too obvious to be true' edge worth checking. And the *way* it "
            "fails is instructive: it's the single most common mistake in factor research — "
            "confusing a thing that moves **with** prices for a thing that moves them **first**."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Two honest checks:\n\n"
            "1. **The lag test.** Line up *yesterday's* change in the crack against *today's* refiner "
            "return. If the crack leads, this lagged link is real and strong. If it's only a "
            "*same-day* mirror, the lagged link vanishes.\n"
            "2. **Race a timer against doing nothing.** Build the obvious rules — be long when the "
            "crack is above its trailing median, or when it's been rising — and compare them to just "
            "holding the basket. Crucially, compare the *active* return (timer minus buy-and-hold), "
            "because a strategy that sits in cash half the time looks calmer without being better.\n\n"
            "Tape: the 3-2-1 crack from crude/gasoline/heating-oil futures vs an equal-weight "
            "VLO+MPC+PSX basket, daily since 2012."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: today vs tomorrow.** Here's how strongly the crack change lines up with the "
            "refiner return on the *same day*, versus how well *yesterday's* crack predicts *today*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    f = real_tape()\n"
            "    pred = st.predictive_regression(f); coin = st.contemporaneous_regression(f)\n"
            "    coin_t, pred_t = coin['tstat'], pred['tstat']\n"
            "    tape = 'REAL tape'\n"
            "else:\n"
            f"    coin_t, pred_t = {R['coin_t']}, {R['pred_t']}\n"
            "    tape = 'frozen real numbers (docs/results.md)'\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "b = ax.bar(['Same day\\n(coincident)', 'Yesterday → today\\n(predictive)'],\n"
            "           [coin_t, pred_t], color=[GREY, RED], width=.5)\n"
            "ax.axhline(2, ls='--', c='k', lw=1, label='|t| = 2 bar')\n"
            "ax.axhline(-2, ls='--', c='k', lw=1); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('HAC t-stat of the crack→stock link'); ax.legend()\n"
            "ax.set_title('The crack mirrors refiners today — but cannot predict tomorrow')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'[{tape}]  same-day t={coin_t:+.2f}   predictive t={pred_t:+.2f}')"
        ),
        md(
            f"There's the whole story in one chart. The **same-day** link clears the bar "
            f"(*t* = +{R['coin_t']:.2f}) — refiners obviously track their own margin. But the "
            f"**predictive** link is dead (*t* = {R['pred_t']:.2f}): yesterday's crack tells you "
            "nothing about today's stock. The signal everyone 'sees' is the mirror, not a forecast."
        ),
        md(
            "**Now: does a crack-timer beat just holding the basket?** Two natural rules — long when "
            "the crack is above its trailing median, and long when it's been rising — vs buy-and-hold."
        ),
        code(
            "if HAVE_REAL:\n"
            "    f = real_tape()\n"
            "    lvl = st.summarize_timer(f, st.level_signal(f['crack']))\n"
            "    mom = st.summarize_timer(f, st.momentum_signal(f['crack'], 20))\n"
            "    vals = [lvl['active_ann'], mom['active_ann']]\n"
            "    ts = [lvl['active_t'], mom['active_t']]\n"
            "else:\n"
            f"    vals = [{R['lvl_active']}, {R['mom20_active']}]\n"
            f"    ts = [{R['lvl_active_t']}, {R['mom20_active_t']}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "b = ax.bar(['Level timer', 'Momentum timer'], vals,\n"
            "           color=[RED if v <= 0 else AMBER for v in vals], width=.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('active return vs buy-and-hold (%/yr)')\n"
            "ax.set_title('No crack-timer adds anything over just holding the basket')\n"
            "for bar_, t_ in zip(b, ts):\n"
            "    ax.annotate(f't={t_:+.2f}', (bar_.get_x()+bar_.get_width()/2,\n"
            "        bar_.get_height()), ha='center', va='bottom' if bar_.get_height()>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Level active {vals[0]:+.2f}%/yr (t={ts[0]:+.2f})   Momentum active {vals[1]:+.2f}%/yr (t={ts[1]:+.2f})')"
        ),
        md(
            f"Both timers *underperform* buy-and-hold on the active return "
            f"(level {R['lvl_active']:+.1f}%/yr, momentum {R['mom20_active']:+.1f}%/yr), and neither "
            "is statistically distinguishable from zero. A 'rising margins' rule gives you nothing "
            "you didn't already have by simply owning the refiners."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Yesterday's crack change doesn't forecast today's refiner return "
            f"(*t* = {R['pred_t']:.2f}, R² ≈ 0.0001). The only real link is *same-day*, which you "
            "can't trade.\n"
            "- **Tradability — Mirage.** No crack-timer beats buy-and-hold; the best gross edge "
            f"(+{R['mom1_active']:.1f}%/yr) is noise and dies at the first cost.\n"
            "- **Coincident ≠ predictive — Confirmed.** The crack *is* the margin in real time, so "
            "it mirrors the stock. Mistaking that mirror for a crystal ball is the entire illusion."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's no gross edge to protect, so costs only make it worse. Here's the momentum "
            "timer's active return as you charge realistic round-trip costs:"
        ),
        code(
            "costs = [0.0, 2.0, 5.0, 10.0]\n"
            "if HAVE_REAL:\n"
            "    f = real_tape(); pos = st.momentum_signal(f['crack'], 20)\n"
            "    net = [st.summarize_timer(f, pos, cost_bps=c)['active_ann'] for c in costs]\n"
            "else:\n"
            f"    net = [{R['cost0']}, {R['cost2']}, {R['cost5']}, {R['cost10']}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net, 0, color=RED, alpha=.12)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('active return vs B&H (%/yr)')\n"
            "ax.set_title('Already at zero gross — costs just dig the hole deeper')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Active return: {net[0]:+.2f}%/yr gross -> {net[-1]:+.2f}%/yr at 10 bps')"
        ),
        md(
            f"Even gross, the timer is at {R['cost0']:+.1f}%/yr active; by 10 bps it's "
            f"{R['cost10']:+.1f}%/yr. Unlike a real-but-fragile signal, there's nothing here to "
            "salvage — the crack-timer is a mirage from the first dollar."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The same trap, elsewhere.** Any 'X moves with Y, so X predicts Y' story deserves "
            "the lag test we ran here. See [Study 226 — Crude-Seasonality](../../226-crude-seasonality/) "
            "and [Study 132 — Yield-Curve-Steepener](../../132-yield-curve-steepener/) for the same "
            "discipline applied to other commodity/curve claims.\n"
            "- **Could a *weekly* crack lead?** We tested daily. Some argue refining margins matter "
            "at a slower frequency; a fork could test weekly EIA crack data against weekly returns "
            "(beware of far fewer observations).\n"
            "- **The hedging angle.** The crack's real job is to *hedge* a refiner's margin, not "
            "time the stock. That's a genuine use — just not the one being marketed.\n\n"
            "*Think there's a slower-moving lead we missed? Fork this, change the horizon, and show "
            "a lagged crack→refiner link that clears |t| = 2 out of sample.*"
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
            "# Crack-Spread — a quantitative teardown\n"
            "### Real daily tape · predictive vs coincident regression · excess-vs-excess timer "
            "race · cost sweep · synthetic lead-lag control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Coincident_%E2%89%A0_predictive%3F: Confirmed](https://img.shields.io/badge/Coincident_%E2%89%A0_predictive%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We isolate the predictive "
            "content of the 3-2-1 crack for an equal-weight refiner basket, race two crack-timers "
            "against always-long buy-and-hold on the **active** return, and confirm with a synthetic "
            "lead-lag control that our null is the market's, not the harness's.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily futures + equities, 2012-04 to "
            "2026-05 (partial June 2026 dropped); as-of 2026-06-17; the offline core and tests run "
            "on a deterministic synthetic tape. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Lagged crack→basket slope HAC *t* = **{R['pred_t']:.2f}** "
            f"(R² ≈ {R['pred_r2']:.4f}), unstable across halves (pre +{R['pred_pre_t']:.2f}, post "
            f"{R['pred_post_t']:.2f}). Only the *same-day* relation is significant "
            f"(*t* = +{R['coin_t']:.2f}, R² = {R['coin_r2']:.3f}) — unforecastable. |\n"
            f"| **Tradability** | `MIRAGE` | No timer beats B&H on active return: level "
            f"{R['lvl_active']:+.2f}%/yr (*t* {R['lvl_active_t']:+.2f}), 20-day momentum "
            f"{R['mom20_active']:+.2f}%/yr (*t* {R['mom20_active_t']:+.2f}); negative at any cost. "
            f"Timer Sharpe CI [{R['mom20_ci_lo']:.2f}, {R['mom20_ci_hi']:.2f}] inside B&H's. |\n"
            f"| **Coincident ≠ predictive?** | `CONFIRMED` | Same-day *t* = +{R['coin_t']:.2f} vs "
            f"predictive *t* = {R['pred_t']:.2f}: a textbook zero-lag confound. |\n\n"
            "> In plain words: the crack spread is the refiner's margin expressed in market prices, "
            "so it co-moves with the equity contemporaneously and forecasts nothing — and a "
            "part-time-in-cash crack-timer's flattering Sharpe is cash drag, not alpha."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $c_t$ be the 3-2-1 crack level and $r_t$ the basket's daily log return. The "
            "tradable hypothesis is **predictive**:\n\n"
            "$$r_t = \\alpha + \\beta\\,\\Delta c_{t-1} + \\varepsilon_t, \\qquad H_1: \\beta \\neq 0$$\n\n"
            "- **H₁ (predictive signal).** Yesterday's crack change forecasts today's return.\n"
            "- **H₂ (tradable timer).** A crack-conditioned timer beats always-long B&H on the "
            "*active* return (timer − B&H), net of costs.\n"
            "- **H₃ (it's not just the mirror).** The lagged effect is distinct from the trivially "
            "large *contemporaneous* slope $\\Delta c_t \\to r_t$.\n\n"
            "We find **H₁ rejected**, **H₂ rejected**, and **H₃ is the whole point**: only the "
            "contemporaneous slope is real, and it is not tradable."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "A public, intraday-observable margin that *led* a whole sub-sector would be a genuine, "
            "scalable timing tool. The interesting content is not the binary verdict (markets are "
            "efficient in liquid large-caps — few will be shocked) but the **mechanism**: the crack "
            "is, by accounting identity, the revenue-minus-input line of the business, so a same-day "
            "correlation is guaranteed and uninformative for timing. Naming that confound precisely "
            "is worth more than the stamp."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** OLS of $r_t$ on standardised $\\Delta c_{t-1}$ (predictive) and "
            "$\\Delta c_t$ (coincident foil); **Newey-West HAC** *t* on each slope.\n"
            "- **Timers.** `level` (crack > trailing-252 median) and `momentum` (crack rose over "
            "the lookback); 0/1 positions, **one execution lag** (signal at close $t$ → return "
            "$t{+}1$), costs **one-way × NAV** on each switch.\n"
            "- **Race.** Excess-vs-excess: the **active return** (timer − buy-and-hold) and its "
            "HAC *t*; a part-time-in-cash timer's raw Sharpe is not a fair comparison.\n"
            "- **Inference.** Circular block-bootstrap (21-day) Sharpe CIs; pre/post split.\n"
            "- **Positive control.** Deterministic synthetic tape with a tunable `lead` — the crack "
            "co-moves same-day regardless, and *predicts* only when a lead is planted.\n\n"
            "Tape: 3-2-1 crack (RB=F, HO=F, CL=F) vs equal-weight TR basket (VLO, MPC, PSX), daily "
            "2012–2026."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Predictive vs coincident — the zero-lag confound\n\n"
            "The same regressor (the crack change) at lag 1 (predictive) and lag 0 (coincident)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    f = real_tape()\n"
            "    pred = st.predictive_regression(f); coin = st.contemporaneous_regression(f)\n"
            "    rows = [('Predictive  (Δcrack_{t-1})', pred['slope'], pred['tstat'], pred['r2']),\n"
            "            ('Coincident  (Δcrack_t)',     coin['slope'], coin['tstat'], coin['r2'])]\n"
            "else:\n"
            f"    rows = [('Predictive  (Δcrack_{{t-1}})', {R['pred_slope']}, {R['pred_t']}, {R['pred_r2']}),\n"
            f"            ('Coincident  (Δcrack_t)',     {R['coin_slope']}, {R['coin_t']}, {R['coin_r2']})]\n"
            "dfr = pd.DataFrame(rows, columns=['relation','slope','t','r2'])\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "a1.bar(dfr['relation'], dfr['t'], color=[RED, GREY], width=.5)\n"
            "a1.axhline(2, ls='--', c='k'); a1.axhline(-2, ls='--', c='k'); a1.axhline(0, c='k', lw=1)\n"
            "a1.set_ylabel('HAC t-stat'); a1.set_title('Only the same-day link clears |t|=2')\n"
            "a1.tick_params(axis='x', labelsize=8)\n"
            "a2.bar(dfr['relation'], dfr['r2']*100, color=[RED, GREY], width=.5)\n"
            "a2.set_ylabel('R^2 (%)'); a2.set_title('...and it explains barely 2% of daily variance')\n"
            "a2.tick_params(axis='x', labelsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dfr.round(5).to_string(index=False))"
        ),
        md(
            f"> 💡 In plain words: the predictive slope's HAC *t* is **{R['pred_t']:.2f}** — economic "
            f"and statistical zero. The coincident slope is significant (*t* = +{R['coin_t']:.2f}) "
            "but tiny in R² and untradeable. **H₁ rejected, H₃ confirmed**: the 'signal' is the "
            f"mirror. The predictive slope also flips sign across halves (pre +{R['pred_pre_t']:.2f}, "
            f"post {R['pred_post_t']:.2f}) — the signature of noise, not a decaying real effect."
        ),
        md(
            "### 4b · The timer race — excess vs excess\n\n"
            "Two crack-timers vs always-long buy-and-hold, judged on the **active return** so the "
            "cash-drag does not flatter the verdict."
        ),
        code(
            "if HAVE_REAL:\n"
            "    f = real_tape()\n"
            "    lvl = st.summarize_timer(f, st.level_signal(f['crack']))\n"
            "    mom = st.summarize_timer(f, st.momentum_signal(f['crack'], 20))\n"
            "    bh_sh = lvl['bh_sharpe']\n"
            "    rows = [('Level timer', lvl['timer_sharpe'], lvl['active_ann'], lvl['active_t'], lvl['pct_in_market']),\n"
            "            ('Momentum (20d)', mom['timer_sharpe'], mom['active_ann'], mom['active_t'], mom['pct_in_market'])]\n"
            "else:\n"
            f"    bh_sh = {R['bh_sharpe']}\n"
            f"    rows = [('Level timer', {R['lvl_sharpe']}, {R['lvl_active']}, {R['lvl_active_t']}, {R['lvl_pim']}),\n"
            f"            ('Momentum (20d)', {R['mom20_sharpe']}, {R['mom20_active']}, {R['mom20_active_t']}, {R['mom20_pim']})]\n"
            "dft = pd.DataFrame(rows, columns=['timer','timer_sharpe','active_ann','active_t','pim'])\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "a1.bar(list(dft['timer'])+['Buy & hold'], list(dft['timer_sharpe'])+[bh_sh],\n"
            "       color=[AMBER, AMBER, GREEN], width=.55)\n"
            "a1.set_ylabel('annualised Sharpe'); a1.set_title('Timers look calmer (higher Sharpe)...')\n"
            "a1.tick_params(axis='x', labelsize=8)\n"
            "a2.bar(dft['timer'], dft['active_ann'], color=RED, width=.45)\n"
            "a2.axhline(0, c='k', lw=1); a2.set_ylabel('active return vs B&H (%/yr)')\n"
            "a2.set_title('...but add ZERO active return'); a2.tick_params(axis='x', labelsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dft.round(2).to_string(index=False))"
        ),
        md(
            f"> 💡 In plain words: the timers post a *higher* Sharpe than B&H "
            f"({R['lvl_sharpe']:.2f}/{R['mom20_sharpe']:.2f} vs {R['bh_sharpe']:.2f}) purely because "
            f"they sit in cash ~{R['mom20_pim']}% of the time — lower vol, not more return. On the "
            "honest metric, the **active return is ~0 and insignificant** "
            f"({R['mom20_active']:+.2f}%/yr, *t* = {R['mom20_active_t']:+.2f}). **H₂ rejected.** The "
            "raw-Sharpe-vs-excess-Sharpe trap is exactly what METHODOLOGY warns about."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — lagged crack→basket HAC *t* = {R['pred_t']:.2f}, "
            f"R² ≈ {R['pred_r2']:.4f}, sign-flipping across halves. The only real link is "
            f"contemporaneous (*t* = +{R['coin_t']:.2f}), which cannot be traded.\n"
            f"- **Tradability `MIRAGE`** — no timer beats B&H on active return (best gross "
            f"+{R['mom1_active']:.2f}%/yr at *t* = +{R['mom1_active_t']:.2f}); negative at any cost; "
            f"timer Sharpe CI [{R['mom20_ci_lo']:.2f}, {R['mom20_ci_hi']:.2f}] sits inside B&H's "
            f"[{R['bh_ci_lo']:.2f}, {R['bh_ci_hi']:.2f}].\n"
            "- **Coincident ≠ predictive `CONFIRMED`** — the crack is the margin in real time; the "
            "same-day correlation is an accounting identity, not a forecast."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — cost sweep\n\n"
            "There is no gross edge to defend, so the cost sweep only confirms the obvious."
        ),
        code(
            "costs = [0.0, 2.0, 5.0, 10.0]\n"
            "if HAVE_REAL:\n"
            "    f = real_tape(); pos = st.momentum_signal(f['crack'], 20)\n"
            "    act = [st.summarize_timer(f, pos, cost_bps=c)['active_ann'] for c in costs]\n"
            "    tt = [st.summarize_timer(f, pos, cost_bps=c)['active_t'] for c in costs]\n"
            "else:\n"
            f"    act = [{R['cost0']}, {R['cost2']}, {R['cost5']}, {R['cost10']}]\n"
            f"    tt = [{R['cost0_t']}, {R['cost2_t']}, {R['cost5_t']}, {R['cost10_t']}]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, act, 'o-', c=RED, lw=2, label='active return (%/yr)')\n"
            "ax2 = ax.twinx(); ax2.plot(costs, tt, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(2, ls=':', c=GREY); ax2.axhline(-2, ls=':', c=GREY)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('active return (%/yr)', color=RED)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('20-day crack-timer: negative gross, more negative net')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'At 10 bps: active {R['cost10']:+.2f}%/yr, t={R['cost10_t']:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: contrast the desk's *fragile* signals, which start positive gross "
            f"and erode under costs. Here the active return is already {R['cost0']:+.2f}%/yr gross "
            "and only worsens — a mirage from the first dollar, not an edge that costs ate."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the *engine* a faithful lead-lag detector, or does it always print null? Plant a "
            "true predictive lead and confirm both the regression and the timer light up — while a "
            "zero-lead tape still shows the same-day mirror but no predictability."
        ),
        code(
            "rows = []\n"
            "for lead, lab in [(0.0, 'lead = 0 (coincident only — the null)'),\n"
            "                  (0.4, 'lead = 0.4 (genuine predictive lead)')]:\n"
            "    f, _ = data.synthetic_tape(n_days=4000, lead=lead, seed=306)\n"
            "    pr = st.predictive_regression(f)\n"
            "    s = st.summarize_timer(f, st.momentum_signal(f['crack'], 1))\n"
            "    rows.append((lab, pr['tstat'], s['active_ann'], s['active_t']))\n"
            "dfs = pd.DataFrame(rows, columns=['synthetic tape','pred_t','timer_active','active_t'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "x = np.arange(len(dfs)); w = .35\n"
            "ax.bar(x-w/2, dfs['pred_t'], w, color=GREEN, label='predictive HAC t')\n"
            "ax.bar(x+w/2, dfs['active_t'], w, color=AMBER, label='timer active HAC t')\n"
            "ax.axhline(2, ls='--', c='k'); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels(dfs['synthetic tape'], fontsize=8)\n"
            "ax.set_ylabel('HAC t-stat'); ax.legend()\n"
            "ax.set_title('SYNTHETIC control: the engine detects a lead only when one is planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('[SYNTHETIC tape — not the market]')\n"
            "print(dfs.round(2).to_string(index=False))"
        ),
        md(
            f"> 💡 In plain words (synthetic, not the market): at lead = 0 the predictive *t* is "
            f"~{R['syn_null_pred_t']:.1f} and the timer adds nothing "
            f"(*t* = +{R['syn_null_active_t']:.2f}) — exactly the real-tape picture. Plant a real "
            f"lead and the predictive *t* jumps to +{R['syn_lead_pred_t']:.1f} and the timer to "
            f"*t* = +{R['syn_lead_active_t']:.2f}. The harness is faithful, so the real-tape null is "
            "a fact about **the market**: the crack is a coincident mirror, not a leading indicator, "
            "of refiner equity.\n\n"
            "For the same excess-vs-excess discipline on other commodity/curve claims, see "
            "[Study 226 — Crude-Seasonality](../../226-crude-seasonality/) and "
            "[Study 132 — Yield-Curve-Steepener](../../132-yield-curve-steepener/)."
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
