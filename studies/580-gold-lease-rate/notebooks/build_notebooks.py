"""Generate the two narrative notebooks for Study 580 (Gold-Lease-Rate).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). **This is a synthetic-only
study** — the LBMA discontinued the daily GOFO benchmark in 2015 and no free continuous lease-rate
tape exists, so ``HAVE_REAL`` is always False. Every cell computes directly from the deterministic,
offline generator in ``gold_lease_rate/data.py`` (seed 580), so the numbers are reproducible for any
reader offline. The dict ``R`` below mirrors docs/results.md as a cross-check.
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-30; synthetic-only).
# NULL world = lead_beta 0 (fp 756d06b1b283); positive control = lead_beta 0.020 (fp e3ed82fc53fe).
R = dict(
    fp_null="756d06b1b283", fp_ctrl="e3ed82fc53fe", n_months=300, n_reg=299,
    # null world (headline)
    null_b=0.0002, null_t=0.08, null_corr=0.005, null_r2=0.000, null_p=0.936,
    null_gross=4.8, null_net=4.7, null_bh=9.2, null_sharpe=0.44, null_switches=34, null_frac=44,
    # positive control (single seed 580)
    ctrl_b=0.0202, ctrl_t=7.73, ctrl_corr=0.409, ctrl_r2=0.167, ctrl_p=0.0005,
    ctrl_net=14.5, ctrl_sharpe=1.20,
    cost_bps=5.0,
    # lag sweep: (lag, null_t, ctrl_t)
    lags=[(0, -0.15, 6.03), (1, 0.08, 7.73), (2, -0.54, 5.60),
          (3, -0.28, 4.66), (6, -0.85, 1.70), (12, -0.44, -0.25)],
    # seed-robust control: (lead_beta, mean t over 25 seeds)
    ctrl_seeds=[(0.000, -0.43), (0.005, 1.52), (0.010, 3.47), (0.020, 7.37), (0.030, 11.27)],
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from gold_lease_rate import data, strategy as st

# SYNTHETIC-ONLY: no real tape exists (LBMA discontinued GOFO 2015-01-30). Everything below runs
# on the deterministic, offline generator (seed 580) and is reproducible for any reader offline.
HAVE_REAL = not data.fetch_series(fetch=False).empty   # always False by design
NULL,  _ = data.synthetic_series(lead_beta=0.0,  lag=1, seed=580)   # the honest headline world
CTRL,  _ = data.synthetic_series(lead_beta=0.02, lag=1, seed=580)   # the planted positive control
print("real lease-rate tape present:", HAVE_REAL, "(synthetic-only study — GOFO discontinued 2015)")
print("null world fp:", data.fingerprint(NULL), "| positive-control fp:", data.fingerprint(CTRL))
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Gold Lease Rate — does the cost to *borrow* gold predict its price? 🪙\n"
            "### An obscure bullion-market number, and whether it can time gold — in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "Gold isn't just bought and sold — it's also **lent**. Banks and refiners borrow physical "
            "bullion, and the price they pay is the **gold lease rate**. When metal is scarce and "
            "everyone wants it *now*, that borrow cost spikes. Bullion-market folklore says those "
            "spikes are a **crystal ball**: a jump in the cost to borrow gold today supposedly "
            "foreshadows a gold *rally* tomorrow.\n\n"
            "It's the kind of clever, obscure signal that sounds like an edge. So we test it. But "
            "there's a catch you'll meet in a moment — **the data was switched off in 2015** — so we "
            "have to build a synthetic world instead, and be honest about what that does and doesn't "
            "prove.\n\n"
            "> 📓 **This is the plain-language layer.** Want the regressions, the placebo test and the "
            "cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Can we test this on *real* gold-lease data? | **No.** The one clean public source — the "
            "LBMA's daily **GOFO** benchmark — was **discontinued on 2015-01-30**. No free, continuous "
            "lease-rate history exists. |\n"
            "| So what did we do? | Built a **synthetic world** where we can *turn the signal on or "
            "off* and check the engine catches it. |\n"
            f"| When the signal is OFF (the null), does the engine cry wolf? | **No.** The lead-lag "
            f"comes out flat (slope-*t* **+{R['null_t']}**, a shuffle test *p* = {R['null_p']}). Good — "
            "no false alarm. |\n"
            f"| When the signal is ON, does the engine catch it? | **Yes** — slope-*t* jumps to "
            f"**+{R['ctrl_t']}**, and it holds up across 25 different random worlds. |\n"
            "| So could you trade the folklore? | **No** — there's nothing to trade (no lease-rate "
            "product), and even the trading rule *loses to just holding gold* in the null world. |\n\n"
            "> The honest verdict is **`WEAK × MIRAGE`**: a well-built engine with **no real data to "
            "point it at**. We can prove the machinery works — we can't prove the folklore does."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"A spike in the gold lease rate — the cost to borrow bullion — foreshadows a rise in "
            "the gold price.\"* — commodity-microstructure folklore.\n\n"
            "The story: the lease rate is roughly `LIBOR − GOFO`, where **GOFO** was the rate to swap "
            "gold for dollars. When physical metal is tight (a scramble, *backwardation*), the lease "
            "rate spikes — and, the folklore says, that tightness spills into the spot price soon "
            "after. A borrow-cost signal that *leads* the price would be a genuine edge, because "
            "almost nobody watches it."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If it were true, an obscure plumbing rate would time one of the world's most-watched "
            "assets — a clean, uncrowded signal. The desk has looked at gold through **ratios** "
            "([113 gold-silver](../../113-gold-silver-ratio/), [305 gold-oil](../../305-gold-oil-ratio/), "
            "[388 lumber-gold](../../388-lumber-gold-ratio/)) and through the "
            "[miners' equity beta](../../208-gold-miners/). This one is different: a **microstructure "
            "carry/borrow** signal, the gold cousin of a *convenience yield*."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "Simple, in principle:\n\n"
            "1. Each month, note the **lease rate** (standardised, so 'high' and 'low' are comparable "
            "over time).\n"
            "2. Look at gold's return over the **following** month.\n"
            "3. Does a high lease rate this month go with a higher gold return next month? Fit a line; "
            "check the slope.\n\n"
            "**The catch:** there's no free lease-rate history to run this on. GOFO ended in 2015. So "
            "we build a **synthetic** monthly world where we control the truth — a lease rate that "
            "mean-reverts and occasionally spikes (like the real one), and a knob that decides whether "
            "it *actually* predicts gold. With the knob at zero (the *null*), it predicts nothing. "
            "That null is our honest headline."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does the synthetic lease rate look like? A calm baseline with occasional "
            "**spikes** — physical scrambles."
        ),
        code(
            "fig, ax = plt.subplots(figsize=(9.5, 3.8))\n"
            "ax.plot(NULL.index, NULL['lease_rate']*100, color=AMBER, lw=1.3)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('gold lease rate %')\n"
            "ax.set_title('Synthetic gold lease rate — a calm baseline with physical-scramble spikes')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"lease rate: mean {NULL['lease_rate'].mean()*100:.2f}%, max {NULL['lease_rate'].max()*100:.2f}%\")"
        ),
        md(
            "**Now the test.** In the *null* world (signal OFF), does a high lease rate this month "
            "foreshadow a higher gold return next month? Plot next-month gold return against this "
            "month's lease-rate level."
        ),
        code(
            "reg = st.predictive_regression(NULL, lag=1)\n"
            "x = NULL['lease_z'].to_numpy()[:-1]; y = NULL['gold_ret'].to_numpy()[1:]*100\n"
            "fig, ax = plt.subplots(figsize=(8, 4.6))\n"
            "ax.scatter(x, y, s=16, color=GREY, alpha=.7)\n"
            "xs = np.linspace(x.min(), x.max(), 50)\n"
            "ax.plot(xs, (reg['b']*xs + (y.mean()/100 - reg['b']*x.mean()))*100, c=RED, lw=2)\n"
            "ax.set_xlabel('this month lease rate (standardised)'); ax.set_ylabel('NEXT month gold return %')\n"
            "ax.set_title(f\"Null world: flat line — slope-t {reg['t']:+.2f}, corr {reg['corr']:+.3f}\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"null world: slope-t {reg['t']:+.2f}, correlation {reg['corr']:+.3f}, R2 {reg['r2']:.3f}\")"
        ),
        md(
            f"A **flat cloud** — no relationship. Slope-*t* **+{R['null_t']}**, correlation "
            f"**+{R['null_corr']}**. That's exactly what 'no signal' looks like, and it's the honest "
            "headline: with no real data, the default synthetic world carries no lead-lag."
        ),
        md(
            "**But is the engine just blind?** Flip the knob ON (plant a real lead-lag) and re-run. If "
            "the engine is any good, it should now light up."
        ),
        code(
            "r_null = st.predictive_regression(NULL, lag=1)['t']\n"
            "r_ctrl = st.predictive_regression(CTRL, lag=1)['t']\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['signal OFF\\n(null)','signal ON\\n(planted)'], [r_null, r_ctrl], color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, lw=1, label='t = 2 bar'); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('predictive slope-t'); ax.legend()\n"
            "ax.set_title('Engine check: flat when OFF, lights up when ON')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'signal OFF t {r_null:+.2f}  |  signal ON t {r_ctrl:+.2f}')"
        ),
        md(
            f"When the lead-lag is really there, the engine catches it (slope-*t* **+{R['ctrl_t']}**, "
            f"well past the *t* = 2 bar). So the flat null result is a statement about the **absence of "
            "a planted signal**, not a broken detector."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** There is **no real lease-rate tape** (GOFO ended 2015), so the axis "
            "is capped at *Weak* — we cannot earn *Real* (that needs a robust result on real data). "
            "The null world is flat; the engine is faithful.\n"
            "- **Tradability — Mirage.** No lease-rate product to trade, the rate is quoted in arrears "
            "with a wide spread, and the trading rule *underperforms just holding gold* in the null "
            "world."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Even pretending the signal were real, you couldn't: there's no gold-lease ETF, the rate "
            "is published in arrears with a wide bid/ask, and the operational rule — *hold gold when "
            "the lease rate is high, else sit in cash* — spends more than half its life in cash and "
            "so **gives up gold's drift**. In the null world it earns about **+4.7%/yr** versus "
            "**+9.2%/yr** for simply buying and holding gold. Mirage."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The data is the whole problem.** If you can source a real, continuous gold-lease-rate "
            "series (reconstructed GOFO, a data vendor's implied lease rate), drop it into the "
            "study's `_cache/` and re-run — the engine is ready and waiting.\n"
            "- **Neighbours.** Gold through ratios: [113 gold-silver](../../113-gold-silver-ratio/), "
            "[305 gold-oil](../../305-gold-oil-ratio/); through miners: "
            "[208 gold-miners](../../208-gold-miners/).\n\n"
            "*Think the lease rate really does lead gold? Find the tape, feed it in, and show the "
            "slope clearing t = 2 the right way.*"
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
            "# The Gold Lease Rate — a quantitative teardown 🔬\n"
            "### Lagged predictive regression · label-shuffle placebo · costed long-flat rule vs buy-&-hold · lag sweep · seed-robust synthetic control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether the lagged "
            "gold lease rate predicts forward gold returns.\n\n"
            "> ⚠️ **Not investment advice.** **Synthetic-only study** — the LBMA discontinued the daily "
            "GOFO benchmark on 2015-01-30 and no free continuous lease-rate tape exists, so there is "
            "**no real tape** and the SIGNAL axis is **capped at WEAK**. All cells run on the "
            f"deterministic offline generator (seed 580; null fp `{R['fp_null']}`, control fp "
            f"`{R['fp_ctrl']}`). Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | **No real tape** (GOFO ended 2015) → capped at WEAK. Null world "
            f"slope-*t* **+{R['null_t']}** (corr **+{R['null_corr']}**, R² **{R['null_r2']:.3f}**, "
            f"placebo *p* **{R['null_p']}**); planted control slope-*t* **+{R['ctrl_t']}** proves the "
            "engine is faithful. |\n"
            f"| **Tradability** | `MIRAGE` | No lease-rate product; long-flat rule earns "
            f"**+{R['null_net']}%/yr net** vs **+{R['null_bh']}%/yr** buy-&-hold on the null world "
            "(loses to passive gold). |\n\n"
            "> 💡 In plain words: the engine works (the control proves it), but with no real data to "
            "point it at, the honest cap is `WEAK × MIRAGE`."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $L_t$ be the standardised gold lease rate at month $t$ and $r^{gold}_{t}$ the gold "
            "return.\n\n"
            "- **H₁ (lead-lag).** In $r^{gold}_{t} = a + b\\,L_{t-1} + e_t$, the slope $b > 0$ at "
            "*t* ≥ 2 (a high lease rate leads a higher gold return).\n"
            "- **H₂ (specificity).** The relation peaks at the true lead and is not a lucky lag.\n"
            "- **H₃ (tradable).** A long-flat rule on the lagged signal beats buy-&-hold gross **and** "
            "net.\n"
            "- **H₄ (data).** A *real*, continuous lease-rate tape exists to test H₁ on.\n\n"
            "We find **H₄ fails outright** (GOFO discontinued 2015 → no free tape), so H₁ is tested "
            "only on a synthetic world: **null → flat**, **planted → recovered**. H₃ **fails** even on "
            "the null (the rule loses to passive gold). The engine is honest; the data is missing."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The lease rate is the gold analogue of a **convenience yield / carry**, and carry is a "
            "documented commodity predictor (Gorton-Rouwenhorst 2006; Koijen-Moskowitz-Pedersen-Vrugt "
            "2018). So a *time-series* lease-rate → gold lead-lag is a priori plausible — which is "
            "exactly why it needs a placebo and a real tape before it earns a stamp. The binding "
            "constraint here is **data availability**, the same limitation the desk names on its other "
            "synthetic-only studies ([273 lego](../../273-lego-returns/), "
            "[275 whisky-cask](../../275-whisky-cask/), [276 sneaker-resale](../../276-sneaker-resale/))."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Predictor.** Standardise the lease rate: $L_t = z(\\text{lease}_t)$.\n"
            "- **Regression.** OLS $r^{gold}_{t} = a + b\\,L_{t-\\text{lag}} + e_t$; the sign & *t* of "
            "$b$ are the claim.\n"
            "- **Placebo.** Shuffle $L$ against forward returns 2000× and read the |t| tail.\n"
            "- **Trade.** Long gold when $L_{t-\\text{lag}} > 0$ else cash; one-way cost × NAV on each "
            "switch; compare to buy-&-hold.\n"
            "- **Robustness.** Sweep the lag ∈ {0,1,2,3,6,12}.\n"
            "- **Positive control.** Plant $b>0$ (`lead_beta`) of rising strength; average the *t* over "
            "**25 seeds** (house rule); confirm it's flat at the null.\n\n"
            "Timing (the single documented lag): the predictor is the lease-*z* observed at the close "
            "of month $t-\\text{lag}$ — already public — and the position is entered at that close and "
            "held one month. No contemporaneous or future data enters the signal."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The predictive regression — H₁\n\n"
            "Null world (headline) beside the planted positive control, each with its placebo *p*."
        ),
        code(
            "reg_n = st.predictive_regression(NULL, lag=1); p_n = st.placebo_pvalue(NULL, lag=1, n_perm=2000, seed=580)\n"
            "reg_c = st.predictive_regression(CTRL, lag=1); p_c = st.placebo_pvalue(CTRL, lag=1, n_perm=2000, seed=580)\n"
            "tab = pd.DataFrame({\n"
            "    'null (lead_beta 0)':   [reg_n['b'], reg_n['t'], reg_n['corr'], reg_n['r2'], p_n],\n"
            "    'control (lead_beta .02)':[reg_c['b'], reg_c['t'], reg_c['corr'], reg_c['r2'], p_c],\n"
            "}, index=['slope b','slope t','corr','R2','placebo p'])\n"
            "print(tab.round(4).to_string())\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['null','control'], [reg_n['t'], reg_c['t']], color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, lw=1, label='t = 2'); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('predictive slope-t'); ax.legend()\n"
            "ax.set_title('H1: flat at the null, clears t=2 when the lead-lag is planted')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: on the null world H₁ is **flat** (slope-*t* **+{R['null_t']}**, "
            f"placebo *p* **{R['null_p']}**) — no lead-lag. Plant one and it snaps to *t* "
            f"**+{R['ctrl_t']}** (placebo *p* **{R['ctrl_p']}**). The engine is a faithful detector; "
            "there is simply no *real* world to run it on."
        ),
        md(
            "### 4b · Specificity — H₂ (the lag sweep)\n\n"
            "Re-fit at several lags. A real, correctly-specified lead-lag should peak at the true lag "
            "and decay; noise should stay near zero everywhere."
        ),
        code(
            "sweep_n = st.lag_sweep(NULL); sweep_c = st.lag_sweep(CTRL)\n"
            "lags = sweep_n.index.tolist()\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(lags, sweep_n['t'], 'o-', c=GREY, lw=2, label='null')\n"
            "ax.plot(lags, sweep_c['t'], 's-', c=GREEN, lw=2, label='control (true lag=1)')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('lag (months)'); ax.set_ylabel('predictive slope-t'); ax.legend()\n"
            "ax.set_title('Lag sweep: null flat everywhere; control peaks at the true lag')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('null t by lag:   ', [round(v,2) for v in sweep_n['t']])\n"
            "print('control t by lag:', [round(v,2) for v in sweep_c['t']])"
        ),
        md(
            "> 💡 In plain words: H₂ behaves correctly — the null scatters around zero at every lag "
            "(no lucky lag manufactures significance), while the planted control peaks at lag 1 (its "
            "true lead) and decays. That decay pattern is the fingerprint of a genuine lead-lag."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — H₄ fails (no real tape; GOFO discontinued 2015), so the axis is "
            f"capped at WEAK. Null flat (*t* +{R['null_t']}), control faithful (*t* +{R['ctrl_t']}).\n"
            f"- **Tradability `MIRAGE`** — no product; the long-flat rule loses to passive gold on the "
            f"null (**+{R['null_net']}%** vs **+{R['null_bh']}%/yr**)."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the long-flat rule vs buy-&-hold\n\n"
            "The operational form of the folklore: long gold when the lagged lease-*z* is high, else "
            "cash, with a one-way cost on each switch. Compare to simply holding gold."
        ),
        code(
            "bt_n = st.long_flat_backtest(NULL, lag=1, z_threshold=0.0, cost_bps=5.0)\n"
            "bt_c = st.long_flat_backtest(CTRL, lag=1, z_threshold=0.0, cost_bps=5.0)\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "x = np.arange(2); w = .26\n"
            "ax.bar(x-w, [bt_n['ann_net']*100, bt_c['ann_net']*100], width=w, color=AMBER, label='rule (net)')\n"
            "ax.bar(x,    [bt_n['ann_bh']*100,  bt_c['ann_bh']*100],  width=w, color=GREY,  label='buy & hold gold')\n"
            "ax.set_xticks(x-w/2); ax.set_xticklabels(['null world','control world'])\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('annualised return %'); ax.legend()\n"
            "ax.set_title('Long-flat rule vs buy-&-hold: loses to gold at the null, wins only when signal is real')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"null:    rule net {bt_n['ann_net']*100:+.1f}%/yr (Sharpe {bt_n['sharpe_net']:.2f}) vs buy-hold {bt_n['ann_bh']*100:+.1f}%/yr, {bt_n['switches']} switches, {bt_n['frac_long']*100:.0f}% long\")\n"
            "print(f\"control: rule net {bt_c['ann_net']*100:+.1f}%/yr (Sharpe {bt_c['sharpe_net']:.2f}) vs buy-hold {bt_c['ann_bh']*100:+.1f}%/yr\")"
        ),
        md(
            f"> 💡 In plain words: on the null world the rule earns **+{R['null_net']}%/yr net** "
            f"(Sharpe {R['null_sharpe']}) versus **+{R['null_bh']}%/yr** buy-&-hold — it *loses* to "
            f"passive gold by sitting in cash {R['null_frac']}% of the time. Only when the signal is "
            f"genuinely planted (control) does it win (**+{R['ctrl_net']}%/yr**, Sharpe "
            f"{R['ctrl_sharpe']}). No real signal ⇒ `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control (seed-robust)\n\n"
            "Is the engine a faithful detector, or does it print noise? Plant a lead-lag "
            "(`lead_beta > 0`) of rising strength and watch the mean predictive *t* — averaged over "
            "**25 seeds** so no single lucky seed can fake it."
        ),
        code(
            "betas = [0.0, 0.005, 0.010, 0.020, 0.030]\n"
            "ts = [st.synthetic_mean_t(data, lead_beta=b, lag=1, n_seeds=25) for b in betas]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(betas, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=RED, lw=1, label='t = 2 bar'); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('planted lead_beta (stronger lead-lag →)')\n"
            "ax.set_ylabel('mean predictive t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted lead-lag — flat at the null, past +2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for b, t in zip(betas, ts): print(f'lead_beta {b:.3f} -> mean t {t:+.2f}')"
        ),
        md(
            "The mean *t* is ≈ 0 at the null and rises monotonically past +2 as the planted lead-lag "
            "grows — so the engine is faithful. The study's inability to certify the folklore is "
            "therefore a **data** limitation (GOFO discontinued 2015), not a broken detector: "
            "`WEAK × MIRAGE`. Feed a real lease-rate tape into `_cache/` and this same machinery would "
            "adjudicate the claim. Neighbours: [113 gold-silver](../../113-gold-silver-ratio/), "
            "[305 gold-oil](../../305-gold-oil-ratio/), [208 gold-miners](../../208-gold-miners/)."
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
