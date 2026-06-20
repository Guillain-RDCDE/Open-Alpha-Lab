"""Generate the two narrative notebooks for Study 326 (Stablecoin-Depeg).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures (positive control, n=2 wall) run anywhere, offline and deterministic; the real-tape
cells use the cached BTC/ETH daily parquets under ../_cache/ if present and otherwise quote
the frozen headline numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs
for any reader, online or off.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-05-31).
R = dict(
    ust_car=-15.4, ust_fwd=-5.1,
    usdc_car=32.9, usdc_fwd=29.1,
    mean_car=8.7, mean_fwd=12.0,
    p_value=0.78, p_fwd=0.88,
    ci_lo=-15.4, ci_hi=32.9,
    n4_mean=5.6, n4_t=0.64, n4_p=0.69,
    rule_excess0=-5.1, rule_excess10=-5.5, rule_excess30=-6.3,
    # synthetic positive control
    syn0_car=2.2, syn0_t=0.60, syn5_car=-7.7, syn5_t=-2.36,
    syn10_car=-16.9, syn10_t=-5.71, syn20_car=-33.5, syn20_t=-13.82,
    asof="2026-05-31",
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

from stablecoin_depeg import data, strategy as st

PRE, POST = 5, 10
EVENTS = data.depeg_events(major_only=True)
HAVE_REAL = data.have_real_cache()

def basket():
    b = data.real_basket()
    return b[b.index <= pd.Timestamp("2026-05-31")]   # drop the in-progress month

print("real BTC/ETH cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Stablecoin-Depeg — is breaking the buck a sell signal? 🪙\n"
            "### Two famous depegs, BTC & ETH, and a heuristic that only works if you cherry-pick\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Depeg_%3D_sell_signal%3F: Busted](https://img.shields.io/badge/Depeg_%3D_sell_signal%3F-Busted-8b949e?style=flat-square)\n\n"
            "When a stablecoin loses its peg, the crypto-Twitter reflex is *sell everything* — a "
            "broken dollar-coin must mean systemic stress and contagion. There are really only two "
            "undisputed, market-wide examples to test it on: the **UST/Luna** death spiral of "
            "May 2022, and the **USDC** break to 88 cents over the Silicon Valley Bank weekend in "
            "March 2023. This notebook asks the simplest possible question: in the days around each "
            "depeg, did Bitcoin and Ethereum *fall*?\n\n"
            "> **This is the plain-language layer.** Want the event-study windows, the synthetic "
            "control and the n = 2 inference wall? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did UST's depeg crash crypto? | **Yes.** The basket fell "
            f"**{R['ust_car']:.0f}%** in the window — an endogenous crypto collapse dragged "
            "everything down. |\n"
            "| Did USDC's depeg crash crypto? | **No — the opposite.** The basket **rose "
            f"+{R['usdc_car']:.0f}%**: USDC broke because a *bank* failed, and money fled the "
            "banking system *into* crypto. |\n"
            "| So is a depeg a sell signal? | **No.** Two events, opposite directions; the average "
            f"move is **+{R['mean_car']:.0f}%** (the *wrong* sign), and a random week is at least "
            f"as bad {R['p_value']*100:.0f}% of the time. |\n"
            "| Could you trade it? | **No.** Selling on each depeg *lost* "
            f"**{R['rule_excess0']:.0f}%** versus just holding — you'd have missed the USDC rebound. |\n\n"
            "> A depeg tells you a coin broke. It does **not** tell you *why* — and the 'why' is the "
            "only thing that moves the market. Same headline, opposite outcomes."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A major stablecoin losing its peg is a flashing red light for the whole crypto "
            "market. It means liquidity is drying up, collateral is in doubt, and contagion is "
            "coming. When a stablecoin breaks the buck — get out.\"*\n\n"
            "It's a reasonable-sounding story, and it has a poster child: **UST/Luna** in May 2022, "
            "whose algorithmic death spiral vaporised ~$40bn and is widely blamed for the cascade "
            "that took down Three Arrows, Celsius and Voyager. If that's your only example, 'depeg "
            "= sell' looks like gospel."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a depeg really were a clean sell signal, it would be one of the few *dated, public, "
            "unambiguous* triggers in all of crypto — no indicator to compute, no parameter to "
            "tune, just 'the coin broke, hit sell.' That's rare and valuable. But it would also say "
            "something deep about how crypto works: that the market treats *any* stablecoin stress "
            "as systemic. The interesting part isn't the binary yes/no — it's discovering that the "
            "**same event type can mean opposite things** depending on its cause."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "An **event study**: line up the days around each depeg and look at what BTC + ETH "
            "actually did.\n\n"
            "1. **Centre on the depeg day** and measure the cumulative move from 5 days before to "
            "10 days after.\n"
            "2. **Compare to random weeks.** Crypto is volatile — a −15% fortnight isn't rare. So we "
            "draw thousands of *random* non-event windows from the same tape and ask: is a depeg "
            "window actually unusual?\n"
            "3. **Try to trade it.** Go to cash for 10 days after each depeg and see if you beat "
            "just holding.\n\n"
            "**What would make us say 'mirage'?** If the two depegs don't both fall, if the average "
            "looks like a random week, and if selling on the news doesn't pay. (Spoiler: all three.)"
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "Here are the two events, side by side — the cumulative move of the BTC+ETH basket in "
            "the [-5, +10] day window around each depeg:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = basket()\n"
            "    w = st.event_windows(b, EVENTS, pre=PRE, post=POST)\n"
            "    cars = st.event_cars(w, from_day=0, pre=PRE)*100\n"
            "    paths = [(1+w[c]).cumprod().to_numpy()*100-100 for c in w.columns]\n"
            "else:\n"
            f"    cars = np.array([{R['ust_car']}, {R['usdc_car']}])\n"
            "    paths = None\n"
            "labels = ['UST/Luna\\n(May 2022)', 'USDC / SVB\\n(Mar 2023)']\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "cols = [RED if c < 0 else GREEN for c in cars]\n"
            "bars_ = ax.bar(labels, cars, color=cols, width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('cumulative move, [-5,+10] days (%)')\n"
            "ax.set_title('Two depegs, opposite directions')\n"
            "for bar_, c in zip(bars_, cars):\n"
            "    ax.annotate(f'{c:+.0f}%', (bar_.get_x()+bar_.get_width()/2,\n"
            "        c + (2 if c>0 else -3)), ha='center', va='bottom' if c>0 else 'top', fontweight='bold')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'UST {cars[0]:+.1f}%   USDC {cars[1]:+.1f}%   mean {cars.mean():+.1f}%')"
        ),
        md(
            f"There's the whole study in one chart. UST's depeg **did** crash the market "
            f"(**{R['ust_car']:.0f}%**) — that's the example everyone remembers. But the USDC depeg, "
            f"six months later, came with a **+{R['usdc_car']:.0f}%** *rally*. Why? USDC didn't break "
            "because crypto was sick — it broke because **Silicon Valley Bank** failed and $3.3bn of "
            "USDC reserves were stuck there. The banking panic that followed sent money *out* of "
            "banks and *into* crypto. Same headline ('stablecoin loses peg'), opposite cause, "
            "opposite outcome."
        ),
        md(
            "**Is a depeg window even unusual?** Crypto routinely moves double digits in a "
            "fortnight. Let's drop the two real windows onto the distribution of *random* "
            "two-event windows from the same years:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = basket()\n"
            "    null = st.null_car_distribution(b, EVENTS, pre=PRE, post=POST, from_day=0, n_draws=5000)\n"
            "    obs = st.event_cars(st.event_windows(b, EVENTS, pre=PRE, post=POST), from_day=0, pre=PRE).mean()\n"
            "else:\n"
            "    sb, sev, _ = data.synthetic_tape(n_events=2, shock=0.0, seed=1)\n"
            "    null = st.null_car_distribution(sb, sev, n_draws=5000)\n"
            f"    obs = {R['mean_car']/100}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(null*100, bins=60, color=GREY, alpha=.6)\n"
            "ax.axvline(obs*100, c=RED, lw=2.5, label=f'real depegs: mean {obs*100:+.0f}%')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('mean cumulative move of 2 windows (%)'); ax.set_ylabel('random draws')\n"
            "ax.set_title('A depeg window is an utterly ordinary window'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'A random pair of windows is at least as low as {obs*100:+.0f}% about '\n"
            f"      f'{R['p_value']*100:.0f}% of the time — not unusual at all.')"
        ),
        md(
            f"The real depeg windows (red line, average **+{R['mean_car']:.0f}%**) sit right in the "
            f"fat middle of the random pile — a random pair of weeks is at least as bad "
            f"**{R['p_value']*100:.0f}%** of the time. There is no 'depeg effect' to see."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Two events, opposite signs (UST {R['ust_car']:.0f}%, USDC "
            f"+{R['usdc_car']:.0f}%); the average is **positive** and looks like a random week "
            f"(p = {R['p_value']:.2f}). And with only **two** real events, you couldn't compute a "
            "trustworthy statistic even if they agreed.\n"
            "- **Tradability — Mirage.** Selling on the depeg *lost* "
            f"{R['rule_excess0']:.0f}% vs holding — you'd have sold the USDC bottom.\n"
            "- **Depeg = sell signal? — Busted.** The market reaction depends entirely on *why* the "
            "coin broke, and the word 'depeg' doesn't carry that."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Suppose you'd run the rule anyway: on each depeg, sell the basket and sit in cash for "
            "10 days, then buy back. Here's what that did versus simply holding (we act one day "
            "*after* the depeg, since you only know it happened intraday):"
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = basket()\n"
            "    excess = []\n"
            "    for cost in (0.0, 10.0, 30.0):\n"
            "        bk = st.sell_on_depeg(b, EVENTS, hold=10, lag=1, cost_bps=cost)\n"
            "        excess.append((bk['strat_ret']-bk['ret']).sum()*100)\n"
            "else:\n"
            f"    excess = [{R['rule_excess0']}, {R['rule_excess10']}, {R['rule_excess30']}]\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "ax.bar(['0 bps','10 bps','30 bps'], excess, color=RED, width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('cumulative excess vs buy & hold (%)')\n"
            "ax.set_xlabel('round-trip cost'); ax.set_title('Selling on the depeg LOSES money — even free')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Even with zero costs the rule trails buy-and-hold by {excess[0]:+.1f}%.')"
        ),
        md(
            f"Negative before the first cent of cost (**{R['rule_excess0']:.0f}%**), and worse as "
            "you add fees. The rule's one job — get you out before the crash — backfires, because "
            "the window it most wants to dodge (the USDC week) is the one that *rallied*. This isn't "
            "a costs story; there's simply no edge underneath."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The cause is the signal, not the symptom.** The real lesson is that an *endogenous* "
            "depeg (crypto-native, like UST) and an *exogenous* one (TradFi spillover, like "
            "USDC/SVB) are different events wearing the same word. A genuinely predictive study "
            "would classify the cause — but that's a judgement call you can only make *after* the "
            "fact.\n"
            "- **The other stablecoin thesis.** [Study 295 — Stablecoin-Supply](../../295-stablecoin-supply/) "
            "tests whether stablecoin *supply growth* ('dry powder') forecasts BTC — a different "
            "input and a different (also-unconvincing) answer.\n"
            "- **More events?** As the stablecoin sector grows there will be more depegs; fork this "
            "and add them. But beware: you'll be tempted to keep only the scary ones.\n\n"
            "*Think there's a depeg edge if you condition on the cause? Fork this, add a "
            "cause-classifier, and show it out-of-sample.*"
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
            "# Stablecoin-Depeg — a quantitative teardown 🔬\n"
            "### Event study · cumulative abnormal return · synthetic control · the n = 2 wall · "
            "one-lag tradable rule · synthetic positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Depeg_%3D_sell_signal%3F: Busted](https://img.shields.io/badge/Depeg_%3D_sell_signal%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error (where one even exists).* We "
            "run an event study on a BTC+ETH basket around the two market-wide stablecoin depegs, "
            "benchmark the cumulative abnormal return against a synthetic-control null, and confront "
            "the inference floor: **n = 2.**\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo BTC-USD / ETH-USD daily, equal-weight; "
            f"as-of {R['asof']} (in-progress month dropped); the offline core and tests run on a "
            "deterministic synthetic tape. Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | UST CAR {R['ust_car']:.1f}%, USDC CAR +{R['usdc_car']:.1f}% "
            f"(opposite signs); mean +{R['mean_car']:.1f}% (wrong sign), synthetic-control "
            f"p = **{R['p_value']:.2f}**; n = 2 ⇒ HAC *t* **undefined**. |\n"
            f"| **Tradability** | `MIRAGE` | 'sell on depeg' trails B&H by {R['rule_excess0']:.1f}% "
            "gross (misses the USDC rebound); worse with costs. |\n"
            f"| **Depeg = sell signal?** | `BUSTED` | reaction sign is set by the *cause* "
            "(endogenous UST crash vs exogenous USDC/SVB rally), not by the depeg itself. |\n\n"
            "> 💡 In plain words: the thesis survives only on a sample of one (UST). The second real "
            "data point flips its sign, the synthetic control says neither window is unusual, and "
            "two observations can't certify anything regardless."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let the basket's cumulative abnormal return over the window $[-5,+10]$ around depeg "
            "$i$ be $\\mathrm{CAR}_i$ (abnormal = relative to the basket's own mean daily return "
            "outside event windows — the constant-mean event-study benchmark; crypto has no clean "
            "factor model). The believers' hypotheses:\n\n"
            "- **H₁ (sell signal).** $\\mathbb{E}[\\mathrm{CAR}_i] < 0$ — depegs are followed by "
            "abnormal *losses*.\n"
            "- **H₂ (unusual).** The depeg windows sit in the left tail of the distribution of "
            "random non-event windows.\n"
            "- **H₃ (tradable).** Exiting after a depeg beats buy-and-hold, net of costs.\n\n"
            "We find **H₁ rejected** (mean CAR is *positive*; the two events disagree on sign), "
            "**H₂ rejected** (p = "
            f"{R['p_value']:.2f}), and **H₃ rejected** ({R['rule_excess0']:.1f}% vs B&H). The "
            "binding meta-constraint is **n = 2**: no robust statistic is definable."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "A dated, public, parameter-free trigger would be a rare prize in crypto. But the "
            "quantitative stakes here are mostly *negative knowledge*: demonstrating that a salient "
            "event type carries **no directional information** once you stop conditioning on the "
            "outcome. The mechanism — endogenous vs exogenous cause — is the transferable lesson, "
            "and it generalises to every 'X happened, the market must do Y' reflex."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Window.** Daily basket return aligned on relative day, $[-5,+10]$; event-day-"
            "inclusive CAR and forward CAR (from $t{+}1$, the tradable part).\n"
            "- **Benchmark.** Constant-mean abnormal return; synthetic-control null = mean CAR of "
            "*two* randomly-placed non-event windows (like-for-like count), events excluded with a "
            "±20-day buffer.\n"
            "- **Inference.** HAC/Newey-West *t* on the per-event CARs (returns NaN below n = 3 by "
            "design); circular block-bootstrap CI; empirical left-tail p-value vs the null.\n"
            "- **Execution.** One documented lag: act at the close of $t{+}1$ (depeg confirmed "
            "intraday on $t$); costs one-way × NAV, two legs per round-trip; race excess-of-cash "
            "vs excess-of-cash.\n"
            "- **Positive control.** Deterministic synthetic tape with a tunable planted depeg "
            "shock — proves the engine detects a real effect and reads ~null otherwise.\n\n"
            "Basket: equal-weight BTC-USD + ETH-USD daily (Yahoo, 2017→)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The two event paths — opposite signs\n\n"
            "The cumulative basket path through each window. One dives, one climbs."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = basket(); w = st.event_windows(b, EVENTS, pre=PRE, post=POST)\n"
            "    paths = {c: (1+w[c]).cumprod()*100-100 for c in w.columns}\n"
            "    cars = st.event_cars(w, from_day=0, pre=PRE)*100\n"
            "else:\n"
            "    sb, sev, _ = data.synthetic_tape(n_events=2, shock=-0.10, seed=2)\n"
            "    w = st.event_windows(sb, sev, pre=PRE, post=POST)\n"
            "    paths = {c: (1+w[c]).cumprod()*100-100 for c in w.columns}\n"
            f"    cars = np.array([{R['ust_car']}, {R['usdc_car']}])\n"
            "labels = ['UST/Luna (May 2022)', 'USDC / SVB (Mar 2023)']\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "for (c, p), lab, col in zip(paths.items(), labels, [RED, GREEN]):\n"
            "    ax.plot(p.index, p.values, lw=2.2, color=col, label=lab)\n"
            "ax.axvline(0, ls='--', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('trading days from depeg'); ax.set_ylabel('cumulative basket return (%)')\n"
            "ax.set_title('Cumulative abnormal path around each depeg'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'UST full CAR {cars[0]:+.1f}%   USDC full CAR {cars[1]:+.1f}%   mean {cars.mean():+.1f}%')"
        ),
        md(
            f"> 💡 In plain words: UST's window ends at **{R['ust_car']:.0f}%**, USDC's at "
            f"**+{R['usdc_car']:.0f}%**. The mean (+{R['mean_car']:.0f}%) is the *wrong sign* for a "
            "sell signal, and it's an average of two numbers that don't even agree on direction."
        ),
        md(
            "### 4b · The synthetic control — is a depeg window unusual?\n\n"
            "The null: the mean CAR of *two* randomly-placed non-event windows. Where do the real "
            "depegs fall?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = basket()\n"
            "    null = st.null_car_distribution(b, EVENTS, pre=PRE, post=POST, from_day=0, n_draws=5000)\n"
            "    res = st.summarize_event_study(b, EVENTS, from_day=0, pre=PRE, post=POST)\n"
            "    obs, pval = res['mean_car']*100, res['p_value']\n"
            "else:\n"
            "    sb, sev, _ = data.synthetic_tape(n_events=2, shock=0.0, seed=3)\n"
            "    null = st.null_car_distribution(sb, sev, n_draws=5000)\n"
            f"    obs, pval = {R['mean_car']}, {R['p_value']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(null*100, bins=60, color=GREY, alpha=.6)\n"
            "q5 = np.percentile(null*100, 5)\n"
            "ax.axvline(q5, ls=':', c=AMBER, lw=1.5, label='5th pct of null')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed mean {obs:+.0f}% (p={pval:.2f})')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('mean CAR of 2 windows (%)'); ax.set_ylabel('random draws')\n"
            "ax.set_title('The depeg windows sit in the fat middle of the null'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Empirical left-tail p-value: {pval:.3f}  (a real sell signal would be << 0.05)')"
        ),
        md(
            f"> 💡 In plain words: a random pair of windows is at least as low as the observed mean "
            f"**{R['p_value']*100:.0f}%** of the time. The depeg windows are not in the tail — they "
            "are typical. **H₂ rejected.**"
        ),
        md(
            "### 4c · The n = 2 wall and the near-misses\n\n"
            "Even setting sign aside: two observations cannot produce a robust statistic. The HAC "
            "*t* is **undefined** at n = 2, and the block-bootstrap CI just spans the two CARs. "
            "Adding the two *near-miss* depegs we excluded from the headline (USDT Oct-2022, "
            "FTX-week USDT) gives n = 4 — and still nothing."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = basket()\n"
            "    r2 = st.summarize_event_study(b, EVENTS, from_day=0, pre=PRE, post=POST)\n"
            "    ev4 = data.depeg_events(major_only=False)\n"
            "    r4 = st.summarize_event_study(b, ev4, from_day=0, pre=PRE, post=POST)\n"
            "    rows = [('n=2 (major)', r2['n_events'], r2['mean_car']*100, r2['tstat'], r2['p_value']),\n"
            "            ('n=4 (+near-miss)', r4['n_events'], r4['mean_car']*100, r4['tstat'], r4['p_value'])]\n"
            "else:\n"
            f"    rows = [('n=2 (major)', 2, {R['mean_car']}, float('nan'), {R['p_value']}),\n"
            f"            ('n=4 (+near-miss)', 4, {R['n4_mean']}, {R['n4_t']}, {R['n4_p']})]\n"
            "dfn = pd.DataFrame(rows, columns=['sample','n','mean_CAR_%','HAC_t','p_value'])\n"
            "print(dfn.to_string(index=False))\n"
            "print('\\nHAC t at n=2 is NaN by design — two points cannot certify anything.')"
        ),
        md(
            f"> 💡 In plain words: at n = 4 the mean is still **positive** (+{R['n4_mean']:.0f}%) with "
            f"HAC *t* = **+{R['n4_t']:.2f}** — nowhere near the |t| ≥ 2 bar, and the wrong sign. "
            "There is no depeg sell-signal at any honest sample size we can assemble."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — UST {R['ust_car']:.1f}% vs USDC +{R['usdc_car']:.1f}% (opposite "
            f"signs); mean +{R['mean_car']:.1f}% (wrong sign), synthetic-control p = {R['p_value']:.2f}; "
            f"n = 4 gives HAC *t* = +{R['n4_t']:.2f}. n = 2 makes a robust *t* undefined regardless.\n"
            f"- **Tradability `MIRAGE`** — 'sell on depeg' trails B&H by {R['rule_excess0']:.1f}% "
            "gross and worse with costs; no edge to charge costs against.\n"
            "- **Depeg = sell signal? `BUSTED`** — the reaction's sign is set by the cause "
            "(endogenous crypto collapse vs exogenous TradFi spillover), which 'depeg' does not encode."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the two-trade record and costs\n\n"
            "One documented lag (act at the close of $t{+}1$), cash for 10 days, costs one-way × "
            "NAV. Excess return *of cash* vs the basket's excess *of cash*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = basket(); excess = []\n"
            "    for cost in (0.0, 10.0, 30.0):\n"
            "        bk = st.sell_on_depeg(b, EVENTS, hold=10, lag=1, cost_bps=cost)\n"
            "        excess.append((bk['strat_ret']-bk['ret']).sum()*100)\n"
            "else:\n"
            f"    excess = [{R['rule_excess0']}, {R['rule_excess10']}, {R['rule_excess30']}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot([0,10,30], excess, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.fill_between([0,10,30], excess, 0, color=RED, alpha=.12)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('cum. excess vs B&H (%)')\n"
            "ax.set_title('Negative before the first basis point — a mirage, not a cost story')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Gross excess {{excess[0]:+.1f}}%  ->  at 30bps {{excess[2]:+.1f}}%')"
        ),
        md(
            f"> 💡 In plain words: the rule is **{R['rule_excess0']:.0f}%** behind buy-and-hold even "
            "free, because its one mandatory exit (the USDC week) coincided with a rally. Costs only "
            "make a hole deeper. Unlike a real-but-fragile signal, there is nothing here to salvage."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the *engine* a faithful detector, or is it just always reading 'nothing'? Plant a "
            "depeg crash of varying size on a synthetic tape (with enough events to clear the bar) "
            "and confirm it recovers exactly what was planted."
        ),
        code(
            "shocks = [0.0, -0.025, -0.05, -0.10, -0.15, -0.20]\n"
            "rows = []\n"
            "for sh in shocks:\n"
            "    sb, sev, _ = data.synthetic_tape(n_events=10, shock=sh, seed=326)\n"
            "    r = st.summarize_event_study(sb, sev, from_day=0)\n"
            "    rows.append((sh, r['mean_car']*100, r['tstat'], r['p_value']))\n"
            "dfp = pd.DataFrame(rows, columns=['planted_shock','mean_CAR_%','HAC_t','p_value'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(dfp['planted_shock']*100, dfp['mean_CAR_%'], 'o-', c=GREEN, lw=2, label='recovered CAR')\n"
            "ax.plot(dfp['planted_shock']*100, dfp['planted_shock']*100, ':', c=GREY, lw=1.5, label='planted (day-0)')\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('planted day-0 shock (%)'); ax.set_ylabel('recovered mean CAR (%)')\n"
            "ax.set_title('The engine is a faithful depeg-shock detector'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dfp.round(3).to_string(index=False))"
        ),
        md(
            f"The engine reads ~null at shock 0 (CAR +{R['syn0_car']:.1f}%, *t* +{R['syn0_t']:.2f}) "
            f"and recovers a planted crash cleanly (at −10%: CAR {R['syn10_car']:.1f}%, *t* "
            f"**{R['syn10_t']:.2f}**). So the engine *would* catch a real depeg sell-off — the real "
            "tape's 'nothing' is a statement about the **market**, not a broken harness. *(A "
            "synthetic *t* is a machinery proof; it never backs a Signal stamp.)*\n\n"
            "For the *other* stablecoin thesis — supply growth as 'dry powder' — see "
            "[Study 295 — Stablecoin-Supply](../../295-stablecoin-supply/); for another crypto "
            "event study, [Study 291 — Doge-Tweets](../../291-doge-tweets/)."
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
