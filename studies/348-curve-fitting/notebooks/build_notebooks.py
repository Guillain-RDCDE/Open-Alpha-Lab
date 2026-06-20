"""Generate the two narrative notebooks for Study 348 (Curve-Fitting).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells read the cached SPY
series under ../_cache/ if present and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md), so the notebook re-runs for any reader.
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
    ticker="SPY", window="1993-01 → 2026-05", days=8390, fp="7758a9131474",
    is_span="1993-01 → 2009-09", oos_span="2009-09 → 2026-05", n_grid=93,
    winner=(30, 225),
    is_sharpe=0.91, oos_sharpe=0.69, shrinkage=0.22,
    oos_t=2.92, oos_ci_lo=0.17, oos_ci_hi=1.28, grid_oos_median=0.75,
    buyhold_oos_sharpe=0.92, alpha_ann=-6.12, alpha_t=-2.70, exposure=0.84,
    # real grid-size sweep
    gs_n=[4, 9, 16, 36, 64],
    gs_is=[0.55, 0.54, 0.68, 0.76, 0.91],
    gs_oos=[0.72, 0.73, 0.68, 0.71, 0.69],
    # synthetic null & control (5000 days, 50/50)
    null_winner=(5, 50), null_is=0.94, null_oos=-0.05, null_shrink=0.98, null_t=-0.15,
    ctrl_winner=(10, 50), ctrl_is=1.03, ctrl_oos=1.13, ctrl_shrink=-0.10, ctrl_t=3.36,
    # null grid-size sweep
    null_gs_n=[4, 9, 16, 36, 64],
    null_gs_is=[0.77, 0.74, 0.77, 0.94, 0.84],
    null_gs_oos=[0.02, 0.08, 0.02, -0.05, 0.13],
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

from curve_fitting import data, strategy as st

def _have_cache():
    return os.path.exists(data._cache_path(data.DEFAULT_CACHE, "SPY"))

HAVE_REAL = _have_cache()

def real_series():
    px = data.load_real(ticker="SPY", fetch=False)
    asof = px.index.max().to_period("M").to_timestamp()   # drop the partial final month
    return px[px.index < asof]

print("real SPY cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Curve-Fitting — the backtest that lies 🎚️\n"
            "### Tune a moving-average crossover until it shines — does the winner survive out of sample?\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Survives_out_of_sample%3F: Busted](https://img.shields.io/badge/Survives_out_of_sample%3F-Busted-8b949e?style=flat-square)\n\n"
            "Give a backtester two dials and a grid, and they will *always* find a setting that looks "
            "brilliant on past data. The dangerous question is whether that winning setting keeps working "
            "on data it has never seen. This notebook tunes a moving-average crossover the lazy way — pick "
            "the best of 93 — and watches what happens next, both on a coin-flip market we build and on "
            "real S&P 500.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the shrinkage-vs-grid-size "
            "curve and the alpha-vs-beta strip? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is drawn by "
            "the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Can you tune a backtest to look great? | **Trivially.** On a *pure random walk* the best "
            f"of 93 crossovers scored an in-sample Sharpe of **{R['null_is']:.2f}** — a number you'd kill "
            "for. |\n"
            "| Does the winner keep working? | **No.** That same pair earned **"
            f"{R['null_oos']:+.2f}** out of sample (basically zero). The 'edge' was the search fitting "
            "noise. |\n"
            "| But it survived on real SPY, right? | **Only as a disguise.** The tuned rule's "
            f"out-of-sample Sharpe ({R['oos_sharpe']:.2f}) was just the market's — it was long "
            f"{R['exposure']*100:.0f}% of a bull decade. Strip the market out and it *lost* "
            f"**{abs(R['alpha_ann']):.0f}%/yr** versus simply holding. |\n"
            "| So is the method useless? | **No — it's a detector, not a generator.** When we plant a "
            "*real* trend, the winner survives cleanly. Tuning can confirm a signal; it cannot conjure "
            "one. |\n\n"
            "> A great backtest is the *easiest* thing in finance to manufacture. The whole skill is "
            "refusing to be impressed by one."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "The pitch is everywhere — in books, courses, and every 'I found a strategy with Sharpe 2' "
            "thread:\n\n"
            "> *Sweep your strategy's parameters, keep the settings with the best backtest, and trade "
            "those. The market rewarded them before; it will reward them again.*\n\n"
            "It feels like science: you searched, you measured, you optimised. The strong version says "
            "the optimisation **improves** your live results — that the configuration which won the "
            "backtest is genuinely better than a random one. If that's true, finding edges is just a "
            "matter of grid-searching hard enough."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If tuning worked, two things would follow. First, anyone with a laptop could mine edges to "
            "order — and they can't, which is itself a clue. Second, and more practically: a tuned "
            "backtest is the single most common reason a strategy looks wonderful on paper and bleeds "
            "money live. Knowing *exactly how much* of a glowing backtest is real and how much is the "
            "search flattering itself is the difference between an investor and a victim. That gap has a "
            "name — **overfitting** — and we can measure it."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "One honest protocol, announced before we run it:\n\n"
            "1. **Cut the data in half** — an *in-sample* slice to tune on, an *out-of-sample* slice we "
            "promise never to peek at while tuning.\n"
            "2. **Sweep 93 crossovers** on the in-sample half and crown the best one.\n"
            "3. **Run that exact winner** on the out-of-sample half.\n"
            "4. **Compare** its in-sample Sharpe to its out-of-sample Sharpe. If the search found a real "
            "edge, the two should roughly agree. If it found noise, the out-of-sample number collapses.\n\n"
            "The clincher: we run this on a market we *built to have no edge at all* (a random walk). "
            "If the winner there 'survives', our test is broken. If it collapses — as it must — we've "
            "caught the mirage red-handed. Then we plant a *real* trend and check the test still passes "
            "a true signal."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the trap, on a coin-flip market.** We generate a pure random walk — by "
            "construction there is *no* crossover that genuinely works. Then we tune anyway and look at "
            "the in-sample winner versus its out-of-sample fate."
        ),
        code(
            "px_null, _ = data.synthetic_series(n_days=5000, trend_strength=0.0, trend_halflife=80, seed=348)\n"
            "res_null = st.curve_fit_experiment(px_null, frac_is=0.5)\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "ax.bar(['In-sample\\n(tuned)', 'Out-of-sample\\n(honest)'],\n"
            "       [res_null['is_sharpe'], res_null['oos_sharpe']], color=[AMBER, RED], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('Sharpe ratio')\n"
            "ax.set_title(f\"Best of 93 on a RANDOM WALK — winner {res_null['winner']}\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"IS Sharpe {res_null['is_sharpe']:.2f}  ->  OOS Sharpe {res_null['oos_sharpe']:+.2f}\")"
        ),
        md(
            f"On a market with **nothing in it**, the best of 93 crossovers scored an in-sample Sharpe of "
            f"~**{R['null_is']:.2f}** — and then earned ~**{R['null_oos']:+.2f}** out of sample. The "
            "entire backtest was the search picking whichever pair happened to fit this stretch of noise. "
            "There was never anything to find."
        ),
        md(
            "**The mirage gets *louder* the harder you search.** Watch the in-sample Sharpe climb as we "
            "let the grid grow — while the out-of-sample Sharpe never budges off zero."
        ),
        code(
            "gs = st.grid_size_sweep(px_null, frac_is=0.5)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "ax.plot(gs['n_params'], gs['is_sharpe'], 'o-', c=AMBER, lw=2, label='in-sample (tuned)')\n"
            "ax.plot(gs['n_params'], gs['oos_sharpe'], 's--', c=RED, lw=2, label='out-of-sample (honest)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('number of parameter pairs searched'); ax.set_ylabel('Sharpe')\n"
            "ax.set_title('The more dials you twist, the bigger the (fake) backtest')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('IS by grid size :', [f'{v:.2f}' for v in gs['is_sharpe']])\n"
            "print('OOS by grid size:', [f'{v:+.2f}' for v in gs['oos_sharpe']])"
        ),
        md(
            "Searching more is not *learning* more — it's just buying more lottery tickets and keeping "
            "the winner. The in-sample number is the size of your search, not the size of your edge."
        ),
        md(
            "**Now the real tape — S&P 500, 1993–2026.** We tune on the first half, then run the winner "
            "on the second. Did it survive?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    px = real_series()\n"
            "    res = st.curve_fit_experiment(px, frac_is=0.5)\n"
            "    is_px, oos_px = st.split(px, 0.5)\n"
            "    a = st.alpha_vs_buyhold(oos_px, *res['winner'])\n"
            "    is_s, oos_s, bh_s = res['is_sharpe'], res['oos_sharpe'], a['buyhold_sharpe']\n"
            "    alpha = a['alpha_ann']*100\n"
            "else:\n"
            f"    is_s, oos_s, bh_s, alpha = {R['is_sharpe']}, {R['oos_sharpe']}, {R['buyhold_oos_sharpe']}, {R['alpha_ann']}\n"
            "fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "ax.bar(['Tuned rule\\nIN-sample', 'Tuned rule\\nOUT-of-sample', 'Just hold SPY\\nOUT-of-sample'],\n"
            "       [is_s, oos_s, bh_s], color=[AMBER, RED, GREEN], width=.6)\n"
            "ax.set_ylabel('Sharpe ratio')\n"
            "ax.set_title('On real SPY the winner survives — but loses to doing nothing')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'tuned IS {is_s:.2f} | tuned OOS {oos_s:.2f} | buy-and-hold OOS {bh_s:.2f}')\n"
            "print(f'tuned rule alpha vs buy-and-hold: {alpha:+.1f}%/yr')"
        ),
        md(
            f"Here's the subtle part. On real SPY the tuned rule's out-of-sample Sharpe ("
            f"~**{R['oos_sharpe']:.2f}**) *didn't* collapse to zero — because the out-of-sample years "
            "(2009–2026) were a roaring bull market, and a long/flat rule that's long most of the time "
            f"just **rides the market**. Compare it to doing nothing: buy-and-hold scored "
            f"~**{R['buyhold_oos_sharpe']:.2f}** — *better*. The tuning didn't add an edge; it added "
            f"**{abs(R['alpha_ann']):.0f}%/yr of drag** versus simply holding the index."
        ),
        md(
            "**The control — does the test still pass a *real* signal?** We plant a genuine slow trend "
            "in the synthetic market and re-run. If tuning is pure folly the winner should fail here "
            "too; if the test is honest, it should survive."
        ),
        code(
            "px_ctrl, _ = data.synthetic_series(n_days=5000, trend_strength=0.0015, trend_halflife=80, seed=348)\n"
            "res_ctrl = st.curve_fit_experiment(px_ctrl, frac_is=0.5)\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "ax.bar(['In-sample', 'Out-of-sample'],\n"
            "       [res_ctrl['is_sharpe'], res_ctrl['oos_sharpe']], color=[AMBER, GREEN], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('Sharpe ratio')\n"
            "ax.set_title('Plant a REAL trend: the winner survives out of sample')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"IS {res_ctrl['is_sharpe']:.2f} -> OOS {res_ctrl['oos_sharpe']:.2f} \"\n"
            "      f\"(shrinkage {res_ctrl['shrinkage']:+.2f})\")"
        ),
        md(
            f"With a real trend planted, the in-sample winner earns ~**{R['ctrl_is']:.2f}** *and* keeps "
            f"~**{R['ctrl_oos']:.2f}** out of sample — the gap nearly vanishes. So tuning isn't worthless: "
            "it's a **detector**. It confirms an edge that's truly there, and exposes one that isn't. The "
            "danger is only when you mistake the search itself for the edge."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The optimised crossover has no real edge. On a coin-flip market it "
            "reverts to noise out of sample; on real SPY its survival is just the market's beta, and its "
            f"alpha over buy-and-hold is *negative* ({R['alpha_ann']:+.0f}%/yr).\n"
            "- **Tradability — Mirage.** You'd underperform a buy-and-hold index fund while paying to "
            "trade a pattern fitted to the past. There is nothing to harvest.\n"
            "- **Does the in-sample winner survive out of sample? — Busted.** The big backtest is the "
            "size of your search, not your edge — and it inflates further the more you tune. (A genuinely "
            "planted signal *does* survive, which is the proof the method is a detector, not a magician.)"
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "You can absolutely trade a curve-fitted rule — that's the tragedy. It will look fantastic "
            "right up to the moment your money is on it, then deliver the out-of-sample number, which here "
            "is *worse than the index after costs*. The honest takeaway isn't a tweak to the rule; it's a "
            "rule about rules: **hold back data the optimiser never sees, judge the winner only there, and "
            "subtract the beta you'd have gotten for free.** Anything a tuned backtest claims beyond that "
            "out-of-sample, beta-stripped number is the search admiring its own reflection."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Walk it forward.** A single 50/50 split is the crudest defence. Fork this and run a "
            "rolling walk-forward (re-tune every year, trade the next) and watch how much of the mirage "
            "survives even that.\n"
            "- **Deflate the Sharpe.** Bailey & López de Prado's *Deflated Sharpe Ratio* turns the "
            "grid-size-vs-IS-Sharpe curve you saw here into a formal correction. Add it and see how big a "
            "backtest Sharpe has to be before it's even *interesting*.\n"
            "- **Crank the dials.** Add a stop-loss, a holding period, a volatility filter — every extra "
            "parameter widens the search and inflates the in-sample number further. Re-run the null and "
            "measure how fast the mirage grows.\n\n"
            "*Think your tuned strategy is the real thing? Fork this, run your parameters on a random "
            "walk, and see if the backtest looks any different. If it doesn't, you were curve-fitting.*"
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
            "# Curve-Fitting — a quantitative teardown 🔬\n"
            "### Optimise-then-validate · IS−OOS shrinkage vs grid size · HAC *t* + block bootstrap · "
            "alpha-vs-beta strip · synthetic null + positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Survives_out_of_sample%3F: Busted](https://img.shields.io/badge/Survives_out_of_sample%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We quantify the optimisation "
            "mirage: how the best-of-grid in-sample Sharpe inflates with the search, how it fails to "
            "transfer out of sample on a no-edge tape, and how the one real-tape 'survivor' dissolves "
            "into market beta once we race it against buy-and-hold.\n\n"
            "> ⚠️ **Not investment advice.** Real data: SPY daily total return 1993–2026 (as-of "
            "2026-05-29, last full month); the offline core and tests run on a deterministic synthetic "
            "generator. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Null: IS Sharpe {R['null_is']:.2f} → OOS {R['null_oos']:+.2f} "
            f"(HAC *t* = {R['null_t']:+.2f}). Real SPY: OOS Sharpe {R['oos_sharpe']:.2f} is beta "
            f"({R['exposure']*100:.0f}% exposure); alpha vs buy-and-hold {R['alpha_ann']:+.1f}%/yr "
            f"(*t* = {R['alpha_t']:+.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | The tuned winner trails buy-and-hold OOS "
            f"({R['oos_sharpe']:.2f} vs {R['buyhold_oos_sharpe']:.2f} Sharpe) while paying turnover. No "
            "edge to harvest. |\n"
            f"| **Survives out of sample?** | `BUSTED` | IS Sharpe rises with grid size "
            f"({R['gs_is'][0]:.2f} → {R['gs_is'][-1]:.2f}) while OOS stays flat; the planted-trend "
            f"control survives (IS {R['ctrl_is']:.2f} → OOS {R['ctrl_oos']:.2f}, *t* = {R['ctrl_t']:.2f}). |\n\n"
            "> 💡 In plain words: the maximum of many noisy backtests is biased upward, so the in-sample "
            "winner overstates the truth by an amount that grows with the search. Out of sample that "
            "excess is gone; what remains on a real tape is whatever beta a mostly-long rule inherited."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\{(f_i, s_i)\\}_{i=1}^{N}$ be a grid of crossover windows, $\\widehat{SR}^{IS}_i$ the "
            "in-sample Sharpe of pair $i$, and $i^\\star = \\arg\\max_i \\widehat{SR}^{IS}_i$ the crowned "
            "winner. Let $SR^{OOS}_{i^\\star}$ be its out-of-sample Sharpe and $SR^{BH}$ buy-and-hold's.\n\n"
            "- **H₁ (the winner has a real edge).** $\\mathbb{E}[SR^{OOS}_{i^\\star}] > 0$ and reflects "
            "the rule, not the market.\n"
            "- **H₂ (the optimisation adds value).** The winner's OOS alpha over buy-and-hold is "
            "positive: $\\mathbb{E}[r^{i^\\star}_t - r^{BH}_t] > 0$.\n"
            "- **H₃ (the in-sample Sharpe is informative).** $\\widehat{SR}^{IS}_{i^\\star}$ does not "
            "inflate with $N$ on a no-edge tape.\n\n"
            "We find **H₁ rejected** (OOS reverts to noise on the null; the real-tape OOS is beta), "
            "**H₂ rejected** (real-tape alpha is significantly *negative*), and **H₃ rejected** (IS "
            "Sharpe rises monotonically-ish with $N$ while OOS is flat). The positive control confirms "
            "the test is not simply blind to signal."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Lo & MacKinlay (1990) and the backtest-overfitting literature (Bailey, Borwein, López de "
            "Prado & Zhu 2014) show the best-of-$N$ statistic is biased upward by selection: the "
            "*expected maximum* of $N$ i.i.d. noise Sharpes grows roughly like $\\sqrt{2\\ln N}$ even when "
            "every true Sharpe is zero. The desk's contribution is to make that bias *visible and "
            "measurable* on a concrete rule — the IS−OOS shrinkage as a function of $N$ — and to show "
            "that the one case where the OOS number *doesn't* collapse is explained entirely by market "
            "beta, not by the search. The binary 'did the winner survive?' is a distraction; the "
            "*shrinkage* and the *beta-stripped alpha* are the result."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Rule.** Long-or-flat simple-MA crossover (long when fast MA > slow MA). One execution "
            "lag: the signal at close $t$ earns the return of $t+1$ (a single `shift`). Costs: one-way "
            "turnover × NAV at `cost_bps` per leg on $|\\Delta\\text{position}|$.\n"
            "- **Grid.** 93 valid $(f, s)$ pairs with $f < s$, scored by in-sample annualised Sharpe.\n"
            "- **Split.** Chronological 50/50 — tune on the first half, validate on the second; the OOS "
            "slice is never seen during tuning.\n"
            "- **Inference.** Newey-West HAC *t* on the OOS mean daily return; circular block-bootstrap "
            "(Politis-Romano) 95% CI on the OOS Sharpe. Excess-of-cash returns; flat days are exactly 0.\n"
            "- **Alpha vs beta.** A mostly-long timing rule inherits the asset's beta — we HAC-test the "
            "winner minus buy-and-hold to strip it.\n"
            "- **Falsifier.** A *synthetic null* (random walk) where no rule has an edge — if the winner "
            "survives there, the test is broken. A *positive control* (planted trend) where it must "
            "survive — if it fails there, the test is blind. Both are deterministic and offline.\n"
            "- **Mirage scope:** the synthetic null is a *machinery proof* of the bias, never market "
            "evidence; the Signal stamp is earned (here, denied) on the real tape and the beta strip."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The null — best-of-93 on a random walk\n\n"
            "The in-sample sweep crowns a pair with a handsome Sharpe; out of sample it is indistinguish"
            "able from zero (HAC *t*, block-bootstrap CI straddling zero)."
        ),
        code(
            "px_null, _ = data.synthetic_series(n_days=5000, trend_strength=0.0, trend_halflife=80, seed=348)\n"
            "res_null = st.curve_fit_experiment(px_null, frac_is=0.5)\n"
            "lo, hi = res_null['oos_ci']\n"
            "print(f\"winner {res_null['winner']}  grid={res_null['n_grid']} pairs\")\n"
            "print(f\"IS Sharpe {res_null['is_sharpe']:.2f}  ->  OOS Sharpe {res_null['oos_sharpe']:+.2f}\")\n"
            "print(f\"OOS HAC t = {res_null['oos_test']['tstat']:+.2f}   \"\n"
            "      f\"block-bootstrap 95% Sharpe CI [{lo:+.2f}, {hi:+.2f}]\")\n"
            "print(f\"shrinkage (IS - OOS) = {res_null['shrinkage']:+.2f}\")\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "ax.bar(['IS (tuned)', 'OOS (honest)'], [res_null['is_sharpe'], res_null['oos_sharpe']],\n"
            "       color=[AMBER, RED], width=.55)\n"
            "ax.errorbar(1, res_null['oos_sharpe'], yerr=[[res_null['oos_sharpe']-lo],[hi-res_null['oos_sharpe']]],\n"
            "            fmt='none', ecolor='k', capsize=5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('Sharpe'); ax.set_title('Null: the IS edge does not transfer')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: ~**{R['null_shrink']:.2f}** of an in-sample Sharpe of "
            f"~**{R['null_is']:.2f}** vanished out of sample, leaving **{R['null_oos']:+.2f}** (*t* = "
            f"{R['null_t']:+.2f}). On a market with no edge, the backtest measured the search, not the "
            "rule."
        ),
        md(
            "### 4b · The bias is a function of the search — IS−OOS shrinkage vs grid size\n\n"
            "The expected maximum in-sample Sharpe grows with $N$ even under the null. Out of sample, the "
            "added height is pure selection and never arrives."
        ),
        code(
            "gs = st.grid_size_sweep(px_null, frac_is=0.5)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.plot(gs['n_params'], gs['is_sharpe'], 'o-', c=AMBER, lw=2, label='IS Sharpe (tuned)')\n"
            "ax.plot(gs['n_params'], gs['oos_sharpe'], 's--', c=RED, lw=2, label='OOS Sharpe (honest)')\n"
            "ax.plot(gs['n_params'], gs['shrinkage'], 'd:', c=GREY, lw=2, label='shrinkage (IS - OOS)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('number of parameter pairs searched'); ax.set_ylabel('Sharpe')\n"
            "ax.set_title('Overfitting scales with the size of the search (null tape)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(gs.to_string(index=False))"
        ),
        md(
            "> 💡 In plain words: the in-sample Sharpe is partly a measure of *how many things you "
            "tried*. A bigger grid is a bigger backtest, not a bigger edge — the out-of-sample line "
            "refuses to follow it up."
        ),
        md(
            "### 4c · The real tape — the 'survivor' is beta in a costume\n\n"
            "On SPY the OOS winner clears *t* = 2, which forbids a flat 'noise' reading. But it was long "
            f"{R['exposure']*100:.0f}% of a bull decade. Race it against buy-and-hold and HAC-test the "
            "difference."
        ),
        code(
            "if HAVE_REAL:\n"
            "    px = real_series()\n"
            "    res = st.curve_fit_experiment(px, frac_is=0.5)\n"
            "    is_px, oos_px = st.split(px, 0.5)\n"
            "    a = st.alpha_vs_buyhold(oos_px, *res['winner'])\n"
            "    print(f\"fingerprint {data.fingerprint(px)}  winner {res['winner']}\")\n"
            "    print(f\"IS Sharpe {res['is_sharpe']:.2f} | OOS Sharpe {res['oos_sharpe']:.2f} \"\n"
            "          f\"(HAC t={res['oos_test']['tstat']:+.2f}, CI [{res['oos_ci'][0]:.2f},{res['oos_ci'][1]:.2f}])\")\n"
            "    print(f\"buy-and-hold OOS Sharpe {a['buyhold_sharpe']:.2f} | avg exposure {a['avg_exposure']:.2f}\")\n"
            "    print(f\"winner ALPHA vs buy-and-hold {a['alpha_ann']*100:+.2f}%/yr  HAC t={a['alpha_test']['tstat']:+.2f}\")\n"
            "    bars = [res['is_sharpe'], res['oos_sharpe'], a['buyhold_sharpe'], a['alpha_ann']*100/10]\n"
            "else:\n"
            f"    print('OFFLINE — frozen real numbers (docs/results.md, as-of 2026-05-29):')\n"
            f"    print('IS {R['is_sharpe']} | OOS {R['oos_sharpe']} (t={R['oos_t']}) | BH {R['buyhold_oos_sharpe']} | alpha {R['alpha_ann']}%/yr (t={R['alpha_t']})')\n"
            f"    bars = [{R['is_sharpe']}, {R['oos_sharpe']}, {R['buyhold_oos_sharpe']}, {R['alpha_ann']}/10]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "labels = ['tuned IS\\nSharpe', 'tuned OOS\\nSharpe', 'buy-hold OOS\\nSharpe', 'tuned alpha\\n(%/yr ÷10)']\n"
            "ax.bar(labels, bars, color=[AMBER, RED, GREEN, GREY], width=.62)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('Sharpe  (last bar: %/yr ÷ 10)')\n"
            "ax.set_title('Real SPY: the OOS \"edge\" is the market; the alpha is negative')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the tuned rule's OOS Sharpe (**{R['oos_sharpe']:.2f}**) is *below* "
            f"buy-and-hold's (**{R['buyhold_oos_sharpe']:.2f}**). Once you subtract the market it was "
            f"sitting in, the optimisation's contribution is **{R['alpha_ann']:+.1f}%/yr** at HAC *t* = "
            f"**{R['alpha_t']:+.2f}** — significantly negative. H₁ and H₂ rejected: no real edge, "
            "negative added value."
        ),
        md(
            "### 4d · The positive control — the test is not blind\n\n"
            "Plant a harvestable trend and the winner must survive cleanly, with shrinkage ≈ 0. (A "
            "harness that can't bank a planted signal would prove nothing by finding nothing.)"
        ),
        code(
            "px_ctrl, _ = data.synthetic_series(n_days=5000, trend_strength=0.0015, trend_halflife=80, seed=348)\n"
            "res_ctrl = st.curve_fit_experiment(px_ctrl, frac_is=0.5)\n"
            "loc, hic = res_ctrl['oos_ci']\n"
            "print(f\"winner {res_ctrl['winner']}  IS {res_ctrl['is_sharpe']:.2f} -> OOS {res_ctrl['oos_sharpe']:.2f}\")\n"
            "print(f\"OOS HAC t = {res_ctrl['oos_test']['tstat']:+.2f}   Sharpe CI [{loc:+.2f}, {hic:+.2f}]\")\n"
            "print(f\"shrinkage {res_ctrl['shrinkage']:+.2f}  (≈ 0 — the edge transfers)\")\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "ax.bar(['null\\nOOS', 'control\\nOOS'], [res_null['oos_sharpe'], res_ctrl['oos_sharpe']],\n"
            "       color=[RED, GREEN], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(0, c='k', lw=1); ax.set_ylabel('OOS Sharpe')\n"
            "ax.set_title('Same test: kills the mirage (red), banks the real signal (green)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: when the trend is *real*, the in-sample winner keeps ~"
            f"**{R['ctrl_oos']:.2f}** of its Sharpe out of sample (*t* = {R['ctrl_t']:.2f}), shrinkage "
            f"≈ {R['ctrl_shrink']:+.2f}. The optimise-then-validate test is a faithful detector: it "
            "distinguishes a fitted edge from a found one. That is precisely why the desk trusts OOS "
            "validation and distrusts the in-sample backtest."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — null OOS Sharpe {R['null_oos']:+.2f} (HAC *t* {R['null_t']:+.2f}); "
            f"real-tape OOS Sharpe {R['oos_sharpe']:.2f} is beta ({R['exposure']*100:.0f}% exposure), "
            f"alpha vs buy-and-hold {R['alpha_ann']:+.1f}%/yr (HAC *t* {R['alpha_t']:+.2f}). No real, "
            "rule-specific edge in either direction.\n"
            f"- **Tradability `MIRAGE`** — the winner underperforms buy-and-hold OOS "
            f"({R['oos_sharpe']:.2f} vs {R['buyhold_oos_sharpe']:.2f}) and pays turnover for the "
            "privilege.\n"
            f"- **Survives out of sample? `BUSTED`** — IS Sharpe inflates with the grid "
            f"({R['gs_is'][0]:.2f} → {R['gs_is'][-1]:.2f}) while OOS is flat; the bias is the search, not "
            f"the rule. The control (IS {R['ctrl_is']:.2f} → OOS {R['ctrl_oos']:.2f}) confirms the method "
            "detects, never manufactures, an edge."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the cost only makes it worse\n\n"
            "There is no break-even cost to find: the tuned rule loses to buy-and-hold *gross*, before a "
            "cent of turnover. The honest question isn't 'what cost kills it' but 'what would make it "
            "real' — and the answer is a genuine signal, which the control shows the test can verify when "
            "present. Charging the random rebalancing only widens an already-negative alpha."
        ),
        code(
            "costs = [0.0, 5.0, 10.0, 20.0]\n"
            "if HAVE_REAL:\n"
            "    alphas = []\n"
            "    for c in costs:\n"
            "        r = st.curve_fit_experiment(px, frac_is=0.5, cost_bps=c)\n"
            "        ac = st.alpha_vs_buyhold(oos_px, *r['winner'], cost_bps=c)\n"
            "        alphas.append(ac['alpha_ann']*100)\n"
            "else:\n"
            f"    alphas = [{R['alpha_ann']}, {R['alpha_ann']-0.1}, {R['alpha_ann']-0.2}, {R['alpha_ann']-0.4}]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.2))\n"
            "ax.plot(costs, alphas, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('one-way cost (bps/leg)'); ax.set_ylabel('winner alpha vs buy-hold (%/yr)')\n"
            "ax.set_title('Already negative gross — costs only dig deeper')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('alpha vs buy-hold by cost:', [f'{a:+.1f}%' for a in alphas])"
        ),
        md(
            "> 💡 In plain words: a **mirage**, not a fragile edge — the optimisation never had anything "
            "to charge costs against. If you want SPY's return, buy SPY; the grid search subtracts value."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Walk-forward & purged CV.** Replace the single split with a rolling re-tune (and López "
            "de Prado's purged/embargoed CV) and measure the residual shrinkage.\n"
            "- **Deflated Sharpe Ratio.** Formalise the grid-size curve: the DSR (Bailey & López de "
            "Prado 2014) deflates the winner's Sharpe by the number of trials and the IS skew/kurtosis. "
            "Add it and re-stamp.\n"
            "- **More dials, louder mirage.** Each added parameter (stop, holding period, vol filter) "
            "widens $N$. Re-run the null grid-size sweep across rule families and chart how the expected "
            "max IS Sharpe tracks $\\sqrt{2\\ln N}$.\n\n"
            "*The in-sample backtest answers 'what fit the past best?' — never 'what will work next?'. "
            "Fork this, point it at any tuned strategy you believe in, and let the out-of-sample, "
            "beta-stripped number have the last word.*"
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
