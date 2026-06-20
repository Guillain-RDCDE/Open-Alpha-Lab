"""Generate the two narrative notebooks for Study 347 (Look-Ahead-Bias).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells read the cached SPY
price parquet under ../_cache/ if present and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-05-29).
R = dict(
    tape="SPY", window="2005-01 → 2026-05", days=5385, fp="b20030e8a296",
    # momentum signal on real SPY (gross)
    peek_sharpe=4.87, peek_t=28.07, peek_cagr=97.7,
    lag1_sharpe=-0.14, lag1_t=-0.77, lag1_cagr=-3.0,
    inflation=5.01,
    # net 5bps/side
    peek_sharpe_5=4.69, peek_t_5=27.01, lag1_sharpe_5=-0.33, lag1_t_5=-1.84,
    # synthetic controls (reversion signal, lagged)
    ctrl_null_lag1=-0.03, ctrl_null_lag1_t=-0.11, ctrl_null_peek=-6.36,
    ctrl_edge_lag1=3.17, ctrl_edge_lag1_t=12.82, ctrl_edge_peek=-6.57, edge=0.002,
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

from look_ahead_bias import data, strategy as st

def _have_cache():
    return os.path.exists(data._cache_path(data.DEFAULT_CACHE, "SPY"))

HAVE_REAL = _have_cache()

def real_tape():
    s = data.load_real("SPY", fetch=False)
    # drop the in-progress final month (as-of = last full month).
    asof = pd.Timestamp.today().normalize().to_period("M").to_timestamp() - pd.Timedelta(days=1)
    return s[s.index <= asof]

print("real SPY cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Look-Ahead-Bias — free alpha from peeking one bar ahead 🔮\n"
            "### The most expensive missing line of code in quant finance: `.shift(1)`\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![A_one--bar_peek_inflates_the_backtest%3F: Confirmed](https://img.shields.io/badge/A_one--bar_peek_inflates_the_backtest%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "Imagine a backtest that lets you *act on today's closing price using today's "
            "closing price* — buy at the open knowing how the day ends. Sounds absurd, but "
            "it's a one-character bug away in every backtest: forget to push your signal "
            "forward one bar, and a nothing strategy turns into a Sharpe-5 superstar. This "
            "notebook plants that bug on purpose and measures exactly how much free alpha it "
            "invents.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the bootstrap and "
            "the sign-flip proof? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below "
            "is drawn by the code beside it. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| How much does a one-bar peek help? | **It manufactures a Sharpe of ~5 from "
            f"nothing.** On real SPY a momentum rule with *no* real edge jumps to Sharpe "
            f"**{R['peek_sharpe']:.1f}** the instant it earns its own bar. |\n"
            "| Is any of it real? | **No.** Trade the same signal honestly (one bar later) "
            f"and the Sharpe is **{R['lag1_sharpe']:.2f}** — a coin flip. The 'edge' was the "
            "signal correlating with the data that defined it. |\n"
            "| Can't trading costs catch it? | **No — and that's the scary part.** The fake "
            "Sharpe barely moves with costs, because a peek isn't an edge you trade, it's an "
            "accounting tautology. |\n"
            "| What's the fix? | **One line.** Push every signal forward one bar before it "
            "earns a return — the desk's *one-execution-lag* rule. |\n\n"
            "> A look-ahead backtest looks *beautiful and robust* — high Sharpe, cheap to "
            "trade. Both are illusions. That's exactly why it fools people."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "Nobody *claims* look-ahead bias is good — they ship it by accident. The claim "
            "we're steelmanning is the seductive backtest it produces: *\"I built a simple "
            "momentum rule and it has a Sharpe near 5 — it must be a real edge, look how "
            "stable it is even after costs.\"* This is backtesting **sin #1** in López de "
            "Prado's *Advances in Financial Machine Learning* and Deutsche Bank's *Seven "
            "Sins of Quantitative Investing*: the single most common way a paper Sharpe turns "
            "out to be fiction. The strong version of the trap is that the result looks "
            "*more* convincing than a real edge — higher Sharpe, more significant, "
            "cost-proof — so it survives exactly the checks people trust."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "A real edge and a look-ahead artefact are worth opposite amounts: one is a "
            "business, the other loses you money the day you go live. Worse, the artefact is "
            "*persuasive* — it sails through cost sweeps and significance tests. If you can't "
            "tell them apart at the desk, you will fund the mirage and starve the edge. The "
            "stakes here aren't one strategy; they're whether your whole research process can "
            "be trusted. The fix costs one line of code — but only if you know to look for it."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We run **one** ordinary signal — a 20-day momentum rule (lean long when the "
            "price is above its recent average) — two ways on the *same* prices:\n\n"
            "1. **peek** — the position decided at today's close earns *today's* return (the "
            "bar that produced the signal). The bug.\n"
            "2. **lag1** — the position decided at today's close earns *tomorrow's* return. "
            "The only version a human can actually trade.\n\n"
            "Then we look at three tapes: a **random walk** (no edge exists — anything the "
            "peek finds is pure bias), a **planted-edge** synthetic market (to prove our "
            "machine *can* find a real, fairly-tradable edge), and **real SPY**. If the peek "
            "lights up where the honest lag is dark, we've caught the bias red-handed."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the cleanest case: a coin-flip market.** We build a pure random walk — "
            "by construction there is *nothing to predict*. Watch what the peek does to the "
            "same momentum rule the honest lag finds worthless:"
        ),
        code(
            "p, _ = data.synthetic_prices(n_days=3000, edge=0.0, seed=347)\n"
            "res = st.compare(p, signal='momentum', cost_bps=0.0)\n"
            "eq_peek = (1+res['peek']).cumprod()\n"
            "eq_lag1 = (1+res['lag1']).cumprod()\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "ax.plot(eq_peek.values, c=RED, lw=2, label=f\"peek (look-ahead)  Sharpe {res['summary_peek']['sharpe']:.1f}\")\n"
            "ax.plot(eq_lag1.values, c=GREEN, lw=2, label=f\"lag1 (honest)     Sharpe {res['summary_lag1']['sharpe']:.2f}\")\n"
            "ax.set_yscale('log'); ax.set_xlabel('trading day'); ax.set_ylabel('equity (log)')\n"
            "ax.set_title('Same signal, same random-walk tape — one peeks, one does not')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"peek Sharpe {res['summary_peek']['sharpe']:.2f} | lag1 Sharpe {res['summary_lag1']['sharpe']:.2f} | inflation {res['sharpe_inflation']:.2f}\")"
        ),
        md(
            "On a market with **zero** predictability, the honest rule earns nothing (a flat "
            "line, Sharpe ~0) while the peek rockets up a smooth exponential. Every cent of "
            "that curve is the look-ahead bias. It isn't finding a pattern — it's letting the "
            "signal grade its own homework."
        ),
        md(
            "**Now the real tape.** Twenty years of SPY. Does the same trick work on real "
            "prices?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = real_tape()\n"
            "    rr = st.compare(s, signal='momentum', cost_bps=0.0)\n"
            "    sharpes = [rr['summary_peek']['sharpe'], rr['summary_lag1']['sharpe']]\n"
            "else:\n"
            f"    sharpes = [{R['peek_sharpe']}, {R['lag1_sharpe']}]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "b = ax.bar(['peek\\n(look-ahead)', 'lag1\\n(honest)'], sharpes, color=[RED, GREEN], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(2, ls=':', c=GREY, label='Sharpe = 2 (a great strategy)')\n"
            "ax.set_ylabel('annualised Sharpe'); ax.legend()\n"
            "ax.set_title('Real SPY (2005–2026): the peek invents a Sharpe ~5; honest = nothing')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'peek Sharpe {sharpes[0]:.2f} vs honest {sharpes[1]:.2f}')"
        ),
        md(
            f"On real SPY the peek prints Sharpe **{R['peek_sharpe']:.1f}** — better than "
            "almost any real strategy ever published — while the honest, tradable version of "
            f"the *exact same rule* sits at **{R['lag1_sharpe']:.2f}**, indistinguishable "
            "from zero. There is no momentum edge here; there is only the bias."
        ),
        md(
            "**\"But surely trading costs kill it?\"** This is the trap. Costs kill *real* "
            "thin edges. A look-ahead artefact isn't an edge — so costs barely scratch it:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    costs = [0.0, 2.0, 5.0, 10.0]\n"
            "    peek_s = [st.compare(s, signal='momentum', cost_bps=c)['summary_peek']['sharpe'] for c in costs]\n"
            "    lag1_s = [st.compare(s, signal='momentum', cost_bps=c)['summary_lag1']['sharpe'] for c in costs]\n"
            "else:\n"
            "    costs = [0.0, 2.0, 5.0, 10.0]\n"
            f"    peek_s = [{R['peek_sharpe']}, 4.80, {R['peek_sharpe_5']}, 4.50]\n"
            f"    lag1_s = [{R['lag1_sharpe']}, -0.21, {R['lag1_sharpe_5']}, -0.53]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.plot(costs, peek_s, 'o-', c=RED, lw=2, label='peek (look-ahead)')\n"
            "ax.plot(costs, lag1_s, 's-', c=GREEN, lw=2, label='lag1 (honest)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('trading cost (bps per side)'); ax.set_ylabel('annualised Sharpe')\n"
            "ax.set_title('Costs cannot erase a peek — because it was never an edge'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('peek Sharpe by cost:', [f'{x:.2f}' for x in peek_s])"
        ),
        md(
            "The honest line wanders around zero; the peek line stays sky-high across the "
            "whole cost sweep. **The very 'robustness' that makes a look-ahead backtest "
            "convincing is the proof it's fake** — a real edge bleeds out under costs; a "
            "tautology doesn't."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Traded honestly, the rule has no edge "
            f"(Sharpe {R['lag1_sharpe']:.2f}). The whole headline was an artefact.\n"
            "- **Tradability — Mirage.** The Sharpe-5 'strategy' needs you to fill on a close "
            "that hasn't happened. Nobody can trade it, at any price.\n"
            "- **Does a one-bar peek inflate the backtest? — Confirmed.** A zero-edge rule "
            f"becomes Sharpe **{R['peek_sharpe']:.1f}** on real SPY, cost-proof, the moment "
            "the signal earns its own bar. The fix is one `shift`."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No — and that's the entire point. To 'trade' the peek you'd have to know each "
            "day's closing price *before* it prints and act on it at that same close. That's "
            "not a hard execution problem; it's time travel. The honest version of the rule "
            "is fully tradable and earns nothing. So the only real takeaway is defensive: "
            "**before you believe any backtest, find the line that shifts the signal forward "
            "one bar.** If it isn't there, the Sharpe on the page is measuring the future "
            "leaking into the past — not an edge you can keep."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Other peeks.** This is the close-to-close version. Intraday, the same bug "
            "hides as 'enter at the open using the day's high/low', or 'rebalance on "
            "month-end data you only got days later'. Fork this and plant those.\n"
            "- **Restated data.** Fundamentals get *revised*; using the final number instead "
            "of the as-first-reported one is look-ahead too (Sloan 1996). A point-in-time "
            "version of this study is a natural next step.\n"
            "- **The deep version.** The companion "
            "[02_for_the_quants.ipynb](02_for_the_quants.ipynb) proves the bias is "
            "*contemporaneous correlation* (flip the signal, the fake Sharpe flips sign) and "
            "shows the harness banks a real planted edge only with the correct lag.\n\n"
            "*Think your favourite backtest is clean? Fork this, drop your signal in, and run "
            "it peek vs lag1. If the two look different, you've found a bug — or an edge. "
            "Only the lag tells you which.*"
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
            "# Look-Ahead-Bias — a quantitative teardown 🔬\n"
            "### One signal, two lags · HAC *t* + block bootstrap · the sign-flip proof · "
            "synthetic positive & negative controls · cost-immunity\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![A_one--bar_peek_inflates_the_backtest%3F: Confirmed](https://img.shields.io/badge/A_one--bar_peek_inflates_the_backtest%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) "
            "— *same seven beats, every claim now carrying its standard error.* We run one "
            "z-score momentum signal under same-bar (**peek**) and one-bar-lag (**lag1**) "
            "execution and quantify the manufactured Sharpe, then prove it is contemporaneous "
            "correlation (not a forecast) with a sign-flip test and a planted-edge control.\n\n"
            "> ⚠️ **Not investment advice.** Real data: SPY daily total-return closes "
            "2005–2026 (as-of 2026-05-29 — the last full month); the offline core and tests "
            "run on a deterministic synthetic generator. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number (real SPY, as-of 2026-05-29) |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Correctly-lagged momentum: Sharpe {R['lag1_sharpe']:.2f}, "
            f"HAC *t* = **{R['lag1_t']:.2f}** — no edge. |\n"
            f"| **Tradability** | `MIRAGE` | The peek Sharpe {R['peek_sharpe']:.2f} requires "
            "filling on an unprinted close; cost-proof because it isn't an edge. |\n"
            f"| **A one-bar peek inflates the backtest?** | `CONFIRMED` | Peek Sharpe "
            f"**{R['peek_sharpe']:.2f}** (HAC *t* = {R['peek_t']:.1f}); inflation "
            f"+{R['inflation']:.2f} Sharpe; the planted control is bankable only at lag1. |\n\n"
            "> 💡 In plain words: the only difference between the two books is a single "
            "`.shift(1)`. That one line is the difference between Sharpe ~0 and Sharpe ~5."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $z_t$ be the causal 20-day z-score of $\\log P$ (uses closes $\\le t$), the "
            "position $w_t = \\tanh(z_t)$, and the bar return $r_t = P_t/P_{t-1}-1$.\n\n"
            "- **peek (look-ahead):** $\\pi^{peek}_t = w_t \\cdot r_t$. The position formed "
            "*from* $r_t$ (via $z_t$) earns $r_t$.\n"
            "- **lag1 (correct):** $\\pi^{lag1}_t = w_{t-1} \\cdot r_t$ — one shift, applied "
            "once.\n\n"
            "- **H₁ (the trap).** $\\text{Sharpe}(\\pi^{peek}) \\gg 0$ with high HAC *t*.\n"
            "- **H₂ (it's bias, not edge).** $\\text{Sharpe}(\\pi^{lag1}) \\approx 0$ on a "
            "tape with no real structure, so the gap is *pure look-ahead*.\n"
            "- **H₃ (mechanism).** The peek is $\\mathrm{Cov}(w_t, r_t)$ — contemporaneous "
            "correlation. Flip the signal's sign and the peek Sharpe flips sign with "
            "*identical magnitude*; a genuine forecast would not.\n\n"
            "We confirm **H₁, H₂ and H₃** below — on a random walk, a planted-edge control, "
            "and real SPY."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Look-ahead is the failure that makes a backtest look *better* than a real edge: "
            "higher Sharpe, more significant, and — crucially — cost-insensitive, because "
            "$\\mathrm{Cov}(w_t, r_t)$ is not an edge that bid-ask spreads erode. So it "
            "survives exactly the three filters a desk trusts most (Sharpe, *t*-stat, cost "
            "sweep). The desk's defence is structural, not statistical: **shift once, "
            "document the effective lag.** This study measures the size of the error that "
            "rule prevents — and shows why no amount of post-hoc significance testing can "
            "catch it."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** 20-day causal z-score, $w_t=\\tanh(z_t)$ (momentum) or "
            "$-\\tanh(z_t)$ (reversion control). Leverage capped at ±1; turnover finite.\n"
            "- **Conventions.** `peek` ($w_t r_t$) vs `lag1` ($w_{t-1} r_t$) — the *only* "
            "difference between books is the one shift.\n"
            "- **Tapes.** (i) random walk null; (ii) planted lag-tradable reversion edge "
            "(pull on $t{+}1$ depends only on $z_t$ — fair to trade); (iii) real SPY.\n"
            "- **Inference.** Newey-West HAC *t* on the mean daily return; circular "
            "block-bootstrap CI (Politis-Romano). Excess-of-zero (a fully-invested ±1 book).\n"
            "- **Costs.** One-way turnover × NAV, $|w_t-w_{t-1}|\\cdot\\text{bps}$, identical "
            "accounting across conventions.\n"
            "- **Controls.** Negative (null → lag1 ≈ 0) and positive (planted edge → lag1 "
            "significant) — a harness that can't bank a fair edge proves nothing by missing "
            "one.\n"
            "- **Falsifier.** If `peek` ≈ `lag1` everywhere, there is no look-ahead tax and "
            "the study is wrong."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The look-ahead tax on a pure random walk\n\n"
            "No structure exists, so any non-zero `peek` Sharpe is bias by construction. HAC "
            "*t* on both books:"
        ),
        code(
            "p, _ = data.synthetic_prices(n_days=3000, edge=0.0, seed=347)\n"
            "res = st.compare(p, signal='momentum', cost_bps=0.0)\n"
            "tp, tl = res['test_peek'], res['test_lag1']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "b = ax.bar(['peek\\n(look-ahead)','lag1\\n(honest)'],\n"
            "           [res['summary_peek']['sharpe'], res['summary_lag1']['sharpe']],\n"
            "           color=[RED, GREEN], width=.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for bar_, t_ in zip(b, [tp['tstat'], tl['tstat']]):\n"
            "    ax.annotate(f'HAC t={t_:+.1f}', (bar_.get_x()+bar_.get_width()/2, bar_.get_height()),\n"
            "        ha='center', va='bottom' if bar_.get_height()>=0 else 'top')\n"
            "ax.set_ylabel('annualised Sharpe')\n"
            "ax.set_title('Random walk: peek is hugely significant, honest is noise')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"peek t={tp['tstat']:+.2f} Sharpe={res['summary_peek']['sharpe']:.2f} | \"\n"
            "      f\"lag1 t={tl['tstat']:+.2f} Sharpe={res['summary_lag1']['sharpe']:.2f}\")"
        ),
        md(
            "> 💡 In plain words: on a coin-flip market the honest rule is statistically "
            "zero (|*t*| < 1), while the peek clears *t* > 20. Significance testing **cannot** "
            "save you — the artefact is *more* significant than most real edges. **H₂ "
            "confirmed.**"
        ),
        md(
            "### 4b · The mechanism — it's contemporaneous correlation (the sign-flip)\n\n"
            "If the peek were a forecast, flipping the signal direction would destroy it. If "
            "it's just $\\mathrm{Cov}(w_t, r_t)$, flipping the signal flips the Sharpe sign "
            "with *identical magnitude*. Test it:"
        ),
        code(
            "mom = st.compare(p, signal='momentum')\n"
            "rev = st.compare(p, signal='reversion')\n"
            "vals = [mom['summary_peek']['sharpe'], rev['summary_peek']['sharpe'],\n"
            "        mom['summary_lag1']['sharpe'], rev['summary_lag1']['sharpe']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "b = ax.bar(['peek\\nmomentum','peek\\nreversion','lag1\\nmomentum','lag1\\nreversion'],\n"
            "           vals, color=[RED, RED, GREEN, GREEN], width=.6)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised Sharpe')\n"
            "ax.set_title('Flip the signal → the PEEK flips sign, same size (a tautology)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"peek momentum {vals[0]:+.2f}  peek reversion {vals[1]:+.2f}  \"\n"
            "      f\"(sum = {vals[0]+vals[1]:+.4f}, i.e. mirror images)\")"
        ),
        md(
            "> 💡 In plain words: the peek Sharpe is $+6.4$ for momentum and $-6.4$ for "
            "reversion — perfect mirror images summing to zero. That is the fingerprint of "
            "$\\mathrm{Cov}(w_t,r_t)$: the position is built from $r_t$, so it *is* $r_t$ "
            "wearing a hat. No forecast behaves this way. **H₃ confirmed.**"
        ),
        md(
            "### 4c · The positive & negative controls — the harness is honest\n\n"
            "Plant a *fair, lag-tradable* reversion edge (the pull on $t{+}1$ uses only $z_t$). "
            "A trustworthy harness banks it with `lag1` and finds nothing on the null:"
        ),
        code(
            "rows = []\n"
            "for edge, label in [(0.0, 'null (edge=0)'), (0.002, 'edge=0.002')]:\n"
            "    pe, _ = data.synthetic_prices(n_days=3000, edge=edge, seed=347)\n"
            "    rc = st.compare(pe, signal='reversion', cost_bps=0.0)\n"
            "    rows.append((label, rc['summary_lag1']['sharpe'], rc['test_lag1']['tstat'],\n"
            "                 rc['summary_peek']['sharpe'], rc['test_peek']['tstat']))\n"
            "import pandas as pd\n"
            "df = pd.DataFrame(rows, columns=['tape','lag1 Sharpe','lag1 HAC t','peek Sharpe','peek HAC t']).set_index('tape')\n"
            "display(df.round(2))\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "x = np.arange(len(rows)); w = .35\n"
            "ax.bar(x-w/2, [r[1] for r in rows], w, color=GREEN, label='lag1 (honest)')\n"
            "ax.bar(x+w/2, [r[3] for r in rows], w, color=RED, label='peek (look-ahead)')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows])\n"
            "ax.set_ylabel('annualised Sharpe'); ax.legend()\n"
            "ax.set_title('Planted edge: banked ONLY at lag1; the peek is wrong-signed noise')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: on the **null** the lagged reversion rule scores ~0 "
            f"({R['ctrl_null_lag1']:.2f}, *t*={R['ctrl_null_lag1_t']:.2f}) — a clean negative "
            f"control. With a planted edge the lagged rule banks it "
            f"({R['ctrl_edge_lag1']:.2f}, *t*={R['ctrl_edge_lag1_t']:.1f}) — the positive "
            "control passes — while the peek is *negative* (it's just the contemporaneous "
            "correlation of a reversion position, the wrong sign). The machine is honest; "
            "only the lag tells truth from artefact."
        ),
        md(
            "### 4d · The real tape — and the cost-immunity tell\n\n"
            "Real SPY, both conventions, swept over costs. The honest rule is dead; the peek "
            "is alive at every cost level — the tell that it isn't an edge."
        ),
        code(
            "costs = [0.0, 2.0, 5.0, 10.0]\n"
            "if HAVE_REAL:\n"
            "    s = real_tape()\n"
            "    peek_s = [st.compare(s, signal='momentum', cost_bps=c)['summary_peek']['sharpe'] for c in costs]\n"
            "    lag1_s = [st.compare(s, signal='momentum', cost_bps=c)['summary_lag1']['sharpe'] for c in costs]\n"
            "    base = st.compare(s, signal='momentum', cost_bps=0.0)\n"
            "    print(f\"SPY fingerprint {data.fingerprint(s)}  n={len(s)}\")\n"
            "    print(f\"peek HAC t={base['test_peek']['tstat']:.2f}  lag1 HAC t={base['test_lag1']['tstat']:.2f}\")\n"
            "else:\n"
            f"    peek_s = [{R['peek_sharpe']}, 4.80, {R['peek_sharpe_5']}, 4.50]\n"
            f"    lag1_s = [{R['lag1_sharpe']}, -0.21, {R['lag1_sharpe_5']}, -0.53]\n"
            f"    print('OFFLINE — frozen (as-of 2026-05-29, fp {R['fp']}):')\n"
            f"    print('peek Sharpe {R['peek_sharpe']} (t={R['peek_t']}) | lag1 Sharpe {R['lag1_sharpe']} (t={R['lag1_t']})')\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.plot(costs, peek_s, 'o-', c=RED, lw=2, label='peek (look-ahead)')\n"
            "ax.plot(costs, lag1_s, 's-', c=GREEN, lw=2, label='lag1 (honest)')\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(2, ls=':', c=GREY)\n"
            "ax.set_xlabel('cost (bps/side)'); ax.set_ylabel('annualised Sharpe')\n"
            "ax.set_title('Real SPY: the peek is cost-proof (it was never an edge)'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('peek Sharpe by cost:', [f'{x:.2f}' for x in peek_s])"
        ),
        md(
            f"> 💡 In plain words: on real SPY the peek prints Sharpe **{R['peek_sharpe']:.2f}** "
            f"(HAC *t* = {R['peek_t']:.1f}) and barely moves under costs, while the honest "
            f"rule sits at **{R['lag1_sharpe']:.2f}** (*t* = {R['lag1_t']:.2f}). The "
            "cost-immunity is not strength — it is the *signature* of a tautology. **H₁ "
            "confirmed on the real tape.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the correctly-lagged momentum rule earns Sharpe "
            f"{R['lag1_sharpe']:.2f} (HAC *t* = {R['lag1_t']:.2f}) on real SPY. No edge.\n"
            f"- **Tradability `MIRAGE`** — the peek Sharpe {R['peek_sharpe']:.2f} demands a "
            "fill on a close that has not printed; un-executable at any cost, and cost-proof "
            "because it is not an edge.\n"
            f"- **A one-bar peek inflates the backtest? `CONFIRMED`** — inflation "
            f"+{R['inflation']:.2f} Sharpe on real SPY; the sign-flip proves it is "
            "$\\mathrm{Cov}(w_t,r_t)$; the planted control is bankable only at lag1. The "
            "empirical case for the desk's **one-execution-lag rule**."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "Beat 6 usually asks where a real edge dies under costs and capacity. Here it "
            "asks something simpler: the look-ahead book requires acting on $r_t$ at the "
            "close that *generated* $r_t$ — a physical impossibility, not a cost problem. The "
            "honest book ($w_{t-1} r_t$) is fully tradable and earns nothing. So the desk "
            "discipline is the deliverable:\n\n"
            "1. **One shift, applied once.** Signal at close of $t$ ⇒ return of $t{+}1$.\n"
            "2. **Document the *effective* lag** in the docstring (a *second* hidden shift is "
            "as wrong as none — it once made a reversal study read a two-day-old signal).\n"
            "3. **Treat cost-immunity as a red flag,** not a feature, until you've found the "
            "shift in the code."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Intraday & calendar peeks.** Enter-at-open-using-the-day's-range, or "
            "rebalance on month-end accounting available only with a lag — same bug, "
            "different bar. Fork the generator to plant them.\n"
            "- **Restated fundamentals.** Point-in-time vs as-restated data is the "
            "fundamental analogue (Sloan 1996); the sign of an anomaly can invert.\n"
            "- **A snooping × look-ahead grid.** Combine with a White (2000) Reality Check to "
            "show the two biases are independent failures that *stack*.\n\n"
            "*The look-ahead Sharpe is the future leaking into the past. The fix is one "
            "`.shift(1)` — and the discipline to always check it's there. Fork this, drop in "
            "your own signal, and run peek vs lag1: the gap is your bug, measured.*"
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
