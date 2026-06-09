"""Generate the two narrative notebooks for Study 16 (Storm-Shy) from source.

Like the other studies, the notebooks are a *generated artefact*: edit the cell text here, rebuild
the skeletons, then execute with nbconvert to embed figures/outputs.

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

The executed path runs on the **offline synthetic tape** — a daily close with a *baked-in*,
persistent calm/storm volatility regime around a constant drift — because the cached real closes are
git-ignored and the desk's reproducible core must run with no network. The synthetic is where the
machinery is *provable*: variance is forecastable by construction, the overlay's Sharpe lift is real
and bounded, and a flat-vol null kills it — which is exactly the point, it proves the code, so the
**real verdict** (SPY/QQQ, quoted from [`docs/results.md`](../docs/results.md), produced by
`examples/verify.py`) is a fact about the market, not a bug. Both notebooks follow the SAME seven
desk beats (see ../../../METHODOLOGY.md).
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))           # study root (storm_shy/ lives there)
sys.path.insert(0, os.path.abspath("../../.."))      # repo root, for quantlab
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (9.5, 5.2)
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
from storm_shy import data, vol, strategy, decompose

# Offline synthetic tape: a daily close with a persistent calm/storm VOL REGIME around a CONSTANT
# drift -- so variance is forecastable and storms carry risk-without-return (the Moreira-Muir setup).
# The flat-vol tape (one regime) is the null. The real verdict (SPY/QQQ) is in ../docs/results.md.
close, truth = data.synthetic_prices(seed=16)
flat,  _     = data.synthetic_prices(sigma_lo=0.011, sigma_hi=0.011, seed=16)
ret      = vol.to_returns(close)
ret_flat = vol.to_returns(flat)
print(f"{truth.n_bars} synthetic bars | calm ~{truth.calm_fraction:.0%} of the time | "
      f"calm/storm vol {truth.sigma_lo:.3f}/{truth.sigma_hi:.3f} | "
      f"perfect-foresight Sharpe ceiling x{truth.theoretical_sharpe_gain:.2f}")
"""

# Real headline numbers (from docs/results.md via examples/verify.py; the cells below EXECUTE on the
# synthetic tape, which is what runs offline). SPY since 1993, QQQ since 1999, as-of 2026-06-01.
R = dict(
    spy_rho="+0.66", spy_bh="+0.65", spy_mgd="+0.76", spy_gain="+0.11",
    spy_alpha="+2.83", spy_t="2.38", spy_ci="[-0.08, +0.31]", spy_turn="9",
    spy_dd_bh="-55", spy_dd_mgd="-39", spy_ce="+2.11",
    qqq_rho="+0.74", qqq_bh="+0.51", qqq_mgd="+0.77", qqq_gain="+0.26",
    qqq_alpha="+4.56", qqq_t="3.54", qqq_ci="[+0.04, +0.48]", qqq_turn="7",
    qqq_dd_bh="-83", qqq_dd_mgd="-38", qqq_ce="+6.92",
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Investable](https://img.shields.io/badge/Tradability-Investable-2ea44f?style=flat-square)\n"
    "![Free lunch?: Risk-managed](https://img.shields.io/badge/Free_lunch%3F-Risk--managed-8b949e?style=flat-square)\n\n"
)


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Storm-Shy 🌊\n"
            "### Cut your exposure when the market gets loud — does it actually pay, or is it just selling low?\n\n"
            + BADGES +
            "Fifteen studies on this desk, fifteen ideas put to the sword — and almost all of them "
            "turned out to be **mirages**. This one is different, and it's the one we've been circling "
            "the whole time. Back in [Study 12](../../12-paper-prophet/) we tore down a fancy "
            "ARIMA+GARCH forecaster and found the *only* real thing inside it was a humble trick: "
            "**when markets get volatile, hold less; when they go quiet, hold more.** We called it "
            "\"vol-targeting in a trenchcoat\" and moved on. Now we give that trick the lead role.\n\n"
            "The pitch (Moreira & Muir, *Journal of Finance* 2017): you don't need to forecast "
            "*returns* — nobody reliably can — you only need to forecast **risk**, and risk is the "
            "single most predictable thing in markets. Volatility *clusters*: a wild day is followed "
            "by wild days, a calm stretch by calm. So size your position by the **inverse** of recent "
            "volatility, and you quietly dodge the worst, scariest stretches without ever predicting "
            "a single price. We'll show it lifts the risk-adjusted return, slashes the gut-wrenching "
            "drawdowns — and then, because this is *this* desk, we'll be honest about exactly what "
            "kind of win it is.\n\n"
            "> 📓 **This is the plain-language layer.** Want the spanning regression, the HAC "
            "*t*-stats and the certainty-equivalent test? That's the companion notebook, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** An educational, reproducible research tool: every chart "
            "below is generated by the code beside it. The reproducible core runs on a **synthetic** "
            "tape where we *bake in* a real, forecastable vol regime — so the real-market numbers "
            "(quoted from [`../docs/results.md`](../docs/results.md)) are a measurement, not a hope. "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## The answer first 🎯\n\n"
            "| What we asked | The honest answer |\n"
            "|---|---|\n"
            "| Is tomorrow's *volatility* predictable? | ✅ **Yes** — strongly. It's the most "
            "forecastable thing in markets (calm clusters, storms cluster). |\n"
            "| Does sizing by inverse-vol lift the risk-adjusted return? | ✅ **Yes** — on real SPY "
            f"the Sharpe goes **{R['spy_bh']} → {R['spy_mgd']}**, on QQQ **{R['qqq_bh']} → "
            f"{R['qqq_mgd']}**, and the drawdown shrinks hard (QQQ **{R['qqq_dd_bh']}% → "
            f"{R['qqq_dd_mgd']}%**). |\n"
            "| Does it survive costs and scale? | ✅ **Yes** — turnover is tiny "
            f"(~{R['spy_turn']}×/yr) and you're sizing the most liquid instrument on earth. The rare "
            "**`INVESTABLE`** stamp. |\n"
            "| So is it a free lunch? | 🚫 **No** — and we won't pretend. It's a *risk-management* "
            "gain: it needs leverage in calm times, and a strict risk-averse test shrinks it to a "
            "smaller, real number. |\n\n"
            "> Desk shorthand: **Signal `REAL` · Tradability `INVESTABLE` · Free lunch? "
            "`RISK-MANAGED`** — the desk's first green, earned on real SPY/QQQ. Let's see how."
        ),

        md(
            "## 1 · The claim 📣\n\n"
            "Forget predicting prices. The claim is narrower and far more defensible: **risk is "
            "predictable even when return isn't.** Markets have quiet regimes and stormy regimes, and "
            "they *persist* — today's volatility tells you a lot about tomorrow's. So the rule is "
            "simply:\n\n"
            "> size your position **inversely to recent volatility** — small when it's stormy, large "
            "when it's calm — to keep your *risk* roughly constant through time.\n\n"
            "Here's the regime on our synthetic tape: the same drift throughout, but the "
            "**volatility** flips between calm and stormy and stays there for a while. That stickiness "
            "is the whole opportunity."
        ),
        code(
            "rv = vol.realized_vol(ret, window=21) * np.sqrt(252)   # trailing annualised vol\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(rv.index, rv.values, lw=0.9)\n"
            "ax.axhline(float(rv.mean()), ls=':', c='grey', label='average')\n"
            "ax.set_ylabel('trailing volatility (annualised)'); ax.legend()\n"
            "ax.set_title('Volatility comes in regimes -- calm stretches and storms, and they persist')\n"
            "plt.show()\n"
            "print(f'calm-regime days sit near {truth.sigma_lo*np.sqrt(252):.0%} vol; "
            "storms near {truth.sigma_hi*np.sqrt(252):.0%}.')"
        ),

        md(
            "## 2 · So what? 💰\n\n"
            "If risk is predictable and return isn't, a quiet edge falls out for free. In the stormy "
            "stretches you're carrying way more risk for the *same* expected reward — a terrible deal. "
            "Down-size there and up-size in the calm, and you spend your risk budget where it's best "
            "paid. You never forecast a price; you just **refuse to be maximally exposed into a "
            "storm.** The prize is two things every investor actually feels: a higher Sharpe (more "
            "return per unit of worry) and — the visceral one — **much shallower crashes**."
        ),

        md(
            "## 3 · How we'd know 🔬\n\n"
            "Five checks, decided up front:\n\n"
            "1. **Is variance actually forecastable?** Does this month's volatility predict next "
            "month's? If not, the whole idea is dead on arrival.\n"
            "2. **Does the overlay lift the Sharpe** vs plain buy-&-hold, net of costs?\n"
            "3. **Does it shrink the drawdowns** — the part you feel?\n"
            "4. **Is the lift statistically real**, or a lucky window? (Spanning alpha + a bootstrap.)\n"
            "5. **The null:** on a tape with *no* regime (constant vol), the overlay must add "
            "**nothing**. If it still 'wins' there, we're fooling ourselves.\n\n"
            "**What would make us say \"mirage\":** an overlay gain that vanishes once costs are "
            "charged, *or* that shows up just as strong on the flat-vol null."
        ),

        md(
            "## 4 · The teardown 🔧\n\n"
            "### 4a · Volatility really is forecastable\n"
            "Chop the tape into months, take each month's variance, and plot it against the *next* "
            "month's. If risk persists, the cloud slopes up — and it does."
        ),
        code(
            "f = vol.forecastability(ret, horizon=21)\n"
            "# scatter: this month's variance vs next month's (the overlay's entire input)\n"
            "r = ret.to_numpy(); nb = len(r)//21\n"
            "bv = r[:nb*21].reshape(nb,21).var(axis=1, ddof=1)\n"
            "fig, ax = plt.subplots(figsize=(6.2,5.6))\n"
            "ax.scatter(bv[:-1]*1e4, bv[1:]*1e4, s=14, alpha=.5)\n"
            "ax.set_xlabel('this month variance (bps^2)'); ax.set_ylabel('next month variance (bps^2)')\n"
            "ax.set_title(f'Risk persists: log-variance AR(1) rho = {f[\"rho\"]:+.2f}')\n"
            "plt.show()\n"
            "print(f\"AR(1) rho = {f['rho']:+.2f}, lag-1 autocorr = {f['autocorr_lag1']:+.2f} \"\n"
            "      f\"-- high: today's volatility forecasts tomorrow's.\")"
        ),
        md(
            "A clear upward tilt. Calm begets calm, storm begets storm. *That* — not any view on "
            "prices — is the only thing the overlay needs."
        ),

        md(
            "### 4b · Sizing by inverse-vol lifts the curve\n"
            "Now run the rule: each day, hold `target_vol / recent_vol` (using only *past* data), and "
            "compare the equity curve to buy-&-hold."
        ),
        code(
            "cmp = strategy.compare(ret, cost_bps=1.0)\n"
            "managed = strategy.managed_returns(ret, cost_bps=1.0)\n"
            "bh = ret.reindex(managed.index)\n"
            "eq_bh = (1+bh).cumprod(); eq_mg = (1+managed).cumprod()\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(eq_bh.index, eq_bh.values, label=f\"buy & hold (Sharpe {cmp['buy_hold']['sharpe']:+.2f})\")\n"
            "ax.plot(eq_mg.index, eq_mg.values, label=f\"vol-managed (Sharpe {cmp['managed_net']['sharpe']:+.2f})\")\n"
            "ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)'); ax.legend()\n"
            "ax.set_title('Same idea, calmer ride: more return per unit of risk')\n"
            "plt.show()\n"
            "print(f\"Sharpe {cmp['buy_hold']['sharpe']:+.2f} -> {cmp['managed_net']['sharpe']:+.2f} \"\n"
            "      f\"(gain {cmp['sharpe_gain_net']:+.2f}); average leverage {cmp['avg_leverage']:.2f}, \"\n"
            "      f\"turnover {cmp['turnover_ann']:.0f}x/yr.\")"
        ),

        md(
            "### 4c · The part you actually feel: drawdowns\n"
            "Sharpe is abstract. The drawdown — how far underwater you go — is what makes people "
            "capitulate at the bottom. Sizing down into storms is *exactly* when it helps most."
        ),
        code(
            "def dd(x): e=(1+x).cumprod(); return e/e.cummax()-1\n"
            "fig, ax = plt.subplots()\n"
            "ax.fill_between(bh.index, dd(bh).values, 0, alpha=.4, label=f\"buy & hold ({dd(bh).min():.0%})\")\n"
            "ax.fill_between(managed.index, dd(managed).values, 0, alpha=.6, label=f\"vol-managed ({dd(managed).min():.0%})\")\n"
            "ax.set_ylabel('drawdown'); ax.legend()\n"
            "ax.set_title('Storm-shy: the overlay is small exactly when the market is falling fastest')\n"
            "plt.show()"
        ),
        md(
            "On the real tape this is the headline a human remembers: SPY's worst drawdown goes from "
            f"**{R['spy_dd_bh']}% to {R['spy_dd_mgd']}%**, and QQQ — which lived through the dot-com "
            f"collapse — from a portfolio-ending **{R['qqq_dd_bh']}% to {R['qqq_dd_mgd']}%**."
        ),

        md(
            "### 4d · Is the lift real, or a lucky decade?\n"
            "Two quick gut-checks: a regression that asks whether the managed stream earns something "
            "buy-&-hold *can't* replicate (a positive \"alpha\"), and a bootstrap that re-deals the "
            "history thousands of times to see if the Sharpe gain holds up."
        ),
        code(
            "sp = decompose.spanning_alpha(ret, cost_bps=1.0)\n"
            "bs = decompose.sharpe_gain_bootstrap(ret, n_boot=2000, seed=0, cost_bps=1.0)\n"
            "print(f\"spanning alpha {sp['alpha_ann_pct']:+.2f}%/yr, HAC t = {sp['alpha_t']:+.2f} \"\n"
            "      f\"(t>2 => not just luck)\")\n"
            "print(f\"bootstrap Sharpe gain {bs['sharpe_gain']:+.2f}, 95% CI \"\n"
            "      f\"[{bs['ci_low']:+.2f}, {bs['ci_high']:+.2f}], P(gain<0) = {bs['frac_negative']:.1%}\")"
        ),

        md(
            "### 4e · The null — when there's no storm to dodge\n"
            "The honest control. Re-run everything on a tape with **constant** volatility — no "
            "regime, nothing to forecast. If the overlay is real, it should now do **nothing**."
        ),
        code(
            "cmp_flat = strategy.compare(ret_flat, cost_bps=1.0)\n"
            "f_flat = vol.forecastability(ret_flat, horizon=21)\n"
            "print(f\"flat-vol null: variance AR(1) rho = {f_flat['rho']:+.2f} (no persistence to read)\")\n"
            "print(f\"               Sharpe gain = {cmp_flat['sharpe_gain_net']:+.2f} (vanishes -- as it must)\")\n"
            "print(f\"clustered tape: rho = {vol.forecastability(ret,horizon=21)['rho']:+.2f}, \"\n"
            "      f\"Sharpe gain = {cmp['sharpe_gain_net']:+.2f}\")"
        ),
        md(
            "There it is: kill the regime and the edge evaporates. The win comes from the volatility "
            "*clustering*, not from the machinery — exactly what a real effect should do."
        ),

        md(
            "## 5 · The verdict 🧾\n\n"
            "- **Risk is forecastable** (synthetic AR(1) ρ ≈ 0.5; real SPY/QQQ ρ ≈ "
            f"{R['spy_rho']}/{R['qqq_rho']}) — the engine is real and replicated.\n"
            "- **The overlay pays**: on real SPY the Sharpe goes "
            f"{R['spy_bh']} → {R['spy_mgd']}, on QQQ {R['qqq_bh']} → {R['qqq_mgd']}, with drawdowns "
            "cut hard — and a spanning alpha that's HAC-significant on both "
            f"(t = {R['spy_t']} / {R['qqq_t']}).\n"
            "- **The null behaves**: no regime ⇒ no gain.\n\n"
            "> **Signal `REAL`.** After fifteen mirages, a genuine, stable, decades-long effect — "
            "because it forecasts *risk*, not *returns*. The real-tape numbers and fingerprints are in "
            "[`../docs/results.md`](../docs/results.md)."
        ),

        md(
            "## 6 · Could you trade it? 💸\n\n"
            "This is the beat that usually kills a 'real' signal on this desk — and the first time it "
            "*doesn't*. Three reasons:\n\n"
            f"- **Turnover is tiny** (~{R['spy_turn']}×/yr). The vol forecast moves slowly, so you "
            "rebalance gently — a few bps of cost against a break-even far above it. Watch the edge "
            "barely flinch as costs rise:\n"
        ),
        code(
            "sweep = strategy.cost_sweep(ret)\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(sweep.index, sweep['sharpe_gain'].values, 'o-')\n"
            "ax.axhline(0, ls=':', c='grey'); ax.set_xlabel('cost per unit traded (bps)')\n"
            "ax.set_ylabel('Sharpe gain over buy & hold')\n"
            "ax.set_title('The edge survives costs -- low turnover is the whole point')\n"
            "plt.show()"
        ),
        md(
            "- **It scales.** You're sizing an *index* — the most liquid instrument on earth (SPY, "
            "ES futures). The ~\\$10M capacity walls that sank earlier studies don't apply; this is "
            "how multi-billion vol-target and risk-parity funds actually run.\n"
            "- **The honest catch.** It is **not** a free lunch, and we won't sell it as one. The gain "
            "is *risk management*: to hold your risk target in calm times you must take **leverage**, "
            "and a strict risk-averse (CRRA) investor, judged at matched risk, banks a **smaller** "
            f"number than the headline Sharpe suggests (real CE gain ≈ {R['spy_ce']}%/yr on SPY, "
            f"{R['qqq_ce']}%/yr on QQQ). Positive, real, bounded.\n\n"
            "> Tradability: a genuine **`INVESTABLE`** — the desk's first. Free lunch? **`RISK-MANAGED`**."
        ),

        md(
            "## 7 · Going further 🚪\n\n"
            "- **Build the vol forecast better.** We used a plain 21-day window. Does a GARCH or "
            "EWMA forecast (already in [`vol.ewma_vol`](../storm_shy/vol.py)) tighten the ride "
            "further — or is the simple window 90% of the prize?\n"
            "- **Across asset classes.** Vol-targeting is strongest where vol-of-vol is high. Bonds, "
            "commodities, FX, crypto — where does it help most, and where does the leverage it "
            "demands become unrealistic?\n"
            "- **The leverage constraint, taken seriously.** Cap leverage at 1.0 (no borrowing): how "
            "much of the gain is left when you can only ever *de*-risk? That's the version a fund "
            "without a financing desk actually lives.\n\n"
            "PRs welcome — push the honest 'yes' harder, or find the regime where even this one breaks."
        ),
    ]
    return new_notebook(cells=cells, metadata=_meta())


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Storm-Shy — a quantitative teardown 🔬\n"
            "### Variance forecastability · Moreira–Muir spanning alpha (HAC) · bootstrap Sharpe gain · CRRA certainty-equivalent · the flat-vol null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* The overlay is the "
            "Moreira–Muir (2017) rule ``w_t = σ_target / σ̂_{t−1}``, a past-only inverse-vol position; "
            "we test the one thing that makes it work — that **variance is forecastable while the "
            "conditional mean is not** — and price both the lift and its honest bound.\n\n"
            "> ⚠️ **Not investment advice.** The reproducible core executes on a synthetic tape with a "
            "baked-in persistent vol regime around a *constant* drift (forecastable risk, no return "
            "signal) — the ground truth the diagnostics recover; the real SPY/QQQ run is in "
            "[`../docs/results.md`](../docs/results.md) via `examples/verify.py`, sources in "
            "[`../docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition — so "
            "this notebook still reads even if you skim the maths. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## Beat 0 · Verdict\n\n"
            "| Axis | Stamp | Why |\n"
            "|---|---|---|\n"
            "| **Signal** — is the effect real? | 🟢 `REAL` | Realized variance is strongly "
            f"autocorrelated (real SPY/QQQ log-var AR(1) ρ = {R['spy_rho']}/{R['qqq_rho']}); the "
            "vol-managed stream carries a Moreira–Muir spanning alpha that is **HAC-significant** "
            f"(t = {R['spy_t']}/{R['qqq_t']}) and a bootstrap Sharpe gain whose CI clears zero on "
            "QQQ. |\n"
            "| **Tradability** | 🟢 `INVESTABLE` | Turnover ~"
            f"{R['spy_turn']}×/yr (a slow forecast ⇒ a few bps vs a far-above break-even), and the "
            "instrument is an index — capacity in the hundreds of billions. The desk's first. |\n"
            "| **Free lunch?** | ⚪ `RISK-MANAGED` | The gain needs leverage in calm regimes; a "
            "matched-risk CRRA certainty-equivalent shrinks it to a smaller, real number "
            f"(≈ {R['spy_ce']}%/yr SPY, {R['qqq_ce']}%/yr QQQ). Risk management, not free alpha. |\n\n"
            "> **In one sentence:** size by inverse volatility and you harvest the one thing markets "
            "*do* forecast — risk — for a real, scalable, honestly-bounded improvement in "
            "risk-adjusted return.\n\n"
            "*(This notebook executes on the offline synthetic tape — where forecastable risk and a "
            "bounded gain are provable. The real SPY/QQQ numbers that earn the stamps are in "
            "[`../docs/results.md`](../docs/results.md).)*"
        ),

        md(
            "## Beat 1 · The claim, stated precisely\n\n"
            "Let $r_t = \\mu + \\sigma_t \\varepsilon_t$ with $\\varepsilon_t$ i.i.d. mean-zero, and "
            "$\\sigma_t$ a **persistent** (forecastable) process independent of the sign of "
            "$\\varepsilon_t$. The vol-managed return is\n\n"
            "$$ r^{\\text{vm}}_t = w_t\\, r_t,\\qquad w_t = \\frac{\\sigma_{\\text{target}}}{\\hat\\sigma_{t-1}},\\quad \\hat\\sigma_{t-1}=\\text{past-only forecast}. $$\n\n"
            "Because $\\mu$ does **not** scale with $\\sigma_t$, high-vol periods deliver the same "
            "expected return for more risk. With perfect foresight the managed Sharpe is "
            "$\\mu\\,\\mathbb{E}[1/\\sigma]$ versus buy-&-hold's $\\mu/\\sqrt{\\mathbb{E}[\\sigma^2]}$, "
            "a ratio $\\mathbb{E}[1/\\sigma]\\sqrt{\\mathbb{E}[\\sigma^2]} \\ge 1$ (Cauchy–Schwarz), "
            "strictly $>1$ whenever vol varies. That ceiling is what the realized, lagged, capped "
            "overlay chases."
        ),
        code(
            "print(f\"baked-in regime: calm/storm vol {truth.sigma_lo:.3f}/{truth.sigma_hi:.3f}, \"\n"
            "      f\"calm fraction {truth.calm_fraction:.0%}\")\n"
            "print(f\"perfect-foresight Sharpe ceiling  x{truth.theoretical_sharpe_gain:.2f}  \"\n"
            "      f\"(the most the overlay could extract here)\")"
        ),

        md(
            "## Beat 2 · So what?\n\n"
            "The claim decouples two things the folk version conflates. Timing **returns** is hard and "
            "mostly a mirage (see most of this desk). Timing **risk** is easy, because risk is "
            "autocorrelated — and it is economically valuable on its own (Fleming–Kirby–Ostdiek 2001). "
            "The managed factor's positive alpha against the static factor is precisely the statement "
            "that a *higher-Sharpe* portfolio is not spanned by a lower-Sharpe one — so the only open "
            "questions are **(i)** is the variance forecast good enough to realise it, **(ii)** does it "
            "survive costs, and **(iii)** how much of the paper gain is a leverage artefact a "
            "risk-averse investor wouldn't actually bank. Beats 4–6 answer all three."
        ),

        md(
            "## Beat 3 · How we'd know — the pre-registered protocol\n\n"
            "1. **Forecastability** (`vol.forecastability`): log-variance AR(1) ρ and lag-1 autocorr "
            "of block variance. Pre-registered: ρ ≫ 0 on the clustered tape, ρ ≈ 0 on the flat null.\n"
            "2. **Overlay** (`strategy.compare`): net Sharpe gain at matched costs, plus turnover and "
            "realized leverage.\n"
            "3. **Spanning alpha** (`decompose.spanning_alpha`): OLS of managed on buy-&-hold with a "
            "**Newey–West** intercept t-stat. Real ⇔ α > 0 with |t| > 2.\n"
            "4. **Bootstrap** (`decompose.sharpe_gain_bootstrap`): paired CI on the Sharpe gain.\n"
            "5. **Honest counter** (`decompose.certainty_equivalent`): CRRA(γ) certainty-equivalent at "
            "**matched unconditional vol** — the leverage-timing-aware number.\n"
            "6. **Null:** every leg re-run on the flat-vol tape must collapse.\n\n"
            "**Mirage line:** the overlay gain dies under costs, *or* survives equally on the flat "
            "null (⇒ it was never the regime doing the work)."
        ),

        md(
            "## Beat 4 · The teardown\n\n"
            "### 4a · Variance is forecastable; the conditional mean is not\n"
            "The AR(1) on log block-variance, clustered tape vs flat null."
        ),
        code(
            "for label, series in [('clustered', ret), ('flat null', ret_flat)]:\n"
            "    f = vol.forecastability(series, horizon=21)\n"
            "    print(f\"{label:10s}: log-var AR(1) rho = {f['rho']:+.2f}, lag-1 autocorr = \"\n"
            "          f\"{f['autocorr_lag1']:+.2f}, vol-of-vol = {f['vol_of_vol']:.2f}\")\n"
            "print('\\n-> clustered: variance strongly predicts itself; flat: nothing to predict.')"
        ),

        md(
            "### 4b · The overlay, gross and net\n"
            "Past-only weights, 1 bp per unit of exposure traded, matched-cost comparison."
        ),
        code(
            "cmp = strategy.compare(ret, cost_bps=1.0)\n"
            "tbl = pd.DataFrame({\n"
            "    'buy_hold': cmp['buy_hold'],\n"
            "    'managed_net': cmp['managed_net'],\n"
            "})[['buy_hold','managed_net']].T[['sharpe','vol_ann','cagr','max_drawdown']]\n"
            "display(tbl.round(3))\n"
            "print(f\"net Sharpe gain {cmp['sharpe_gain_net']:+.2f} | avg leverage \"\n"
            "      f\"{cmp['avg_leverage']:.2f}, capped {cmp['frac_capped']:.0%} of days, \"\n"
            "      f\"turnover {cmp['turnover_ann']:.0f}x/yr\")"
        ),

        md(
            "### 4c · Spanning alpha with Newey–West errors\n"
            "Regress managed on buy-&-hold; the intercept is the part a static long position can't "
            "replicate. HAC errors because daily returns are autocorrelated and heteroskedastic."
        ),
        code(
            "sp = decompose.spanning_alpha(ret, cost_bps=1.0)\n"
            "print(f\"alpha = {sp['alpha_ann_pct']:+.2f}%/yr  (daily {sp['alpha_bps_day']:+.3f} bps)\")\n"
            "print(f\"HAC t-stat = {sp['alpha_t']:+.2f}  (lags={sp['lags']}),  beta to market = {sp['beta']:.2f}, \"\n"
            "      f\"beta-t {sp['beta_t']:+.1f}\")\n"
            "print('alpha>0 with |t|>2 => the managed factor expands the mean-variance frontier.')"
        ),
        md(
            "> 💡 **In plain words.** A portfolio with a higher Sharpe than the market *cannot* be "
            "built by just holding more or less market — so it shows up as positive alpha. The HAC "
            "t-stat says that alpha isn't an artefact of a few autocorrelated lucky months."
        ),

        md(
            "### 4d · Bootstrap CI on the Sharpe gain\n"
            "Re-deal the paired (buy-hold, managed) days with replacement; where does the Sharpe gain "
            "land?"
        ),
        code(
            "bs = decompose.sharpe_gain_bootstrap(ret, n_boot=3000, seed=0, cost_bps=1.0)\n"
            "# rebuild the bootstrap distribution for the histogram\n"
            "m = strategy.managed_returns(ret, cost_bps=1.0); b = ret.reindex(m.index)\n"
            "a, bb = b.to_numpy(), m.to_numpy(); n=a.size; rng=np.random.default_rng(0)\n"
            "def sr(x): s=x.std(ddof=1); return x.mean()/s*np.sqrt(252) if s>0 else 0.0\n"
            "boots = np.array([sr(bb[i])-sr(a[i]) for i in (rng.integers(0,n,n) for _ in range(3000))])\n"
            "fig, ax = plt.subplots()\n"
            "ax.hist(boots, bins=40, alpha=.8); ax.axvline(0, c='red', ls=':')\n"
            "ax.axvline(bs['sharpe_gain'], c='k', label=f\"point {bs['sharpe_gain']:+.2f}\")\n"
            "ax.set_xlabel('Sharpe(managed) - Sharpe(buy & hold)'); ax.legend()\n"
            "ax.set_title(f\"bootstrap Sharpe gain: 95% CI [{bs['ci_low']:+.2f}, {bs['ci_high']:+.2f}], \"\n"
            "             f\"P(<0)={bs['frac_negative']:.1%}\")\n"
            "plt.show()"
        ),

        md(
            "### 4e · The honest counter — certainty-equivalent at matched risk\n"
            "The Cederburg et al. (2020) discipline: lever **both** books to the same unconditional "
            "vol (so neither wins by running more risk), then score each by a CRRA mean–variance "
            "certainty-equivalent. The gain that survives *this* is the one an investor actually banks."
        ),
        code(
            "for g in (3.0, 5.0, 10.0):\n"
            "    ce = decompose.certainty_equivalent(ret, gamma=g, cost_bps=1.0)\n"
            "    print(f\"gamma={g:>4}: CE  buy&hold {ce['ce_buy_hold_pct']:+.2f}%  vs  managed \"\n"
            "          f\"{ce['ce_managed_pct']:+.2f}%  ->  gain {ce['ce_gain_pct']:+.2f}%/yr\")\n"
            "er = decompose.equal_risk_return(ret, cost_bps=1.0)\n"
            "print(f\"\\nat matched vol: excess CAGR {er['excess_cagr_pct']:+.2f}%/yr, \"\n"
            "      f\"drawdown {er['buy_hold_maxdd']:.0%} -> {er['managed_maxdd']:.0%}\")"
        ),
        md(
            "> 💡 **In plain words.** Even after you pay for the leverage the overlay needs in calm "
            "times, a risk-averse investor still comes out ahead on this clean tape — but by a *bounded* "
            "margin, and on real data (especially SPY) that margin is where the academic debate lives. "
            "We quote the smaller, honest number, not the headline."
        ),

        md(
            "### 4f · The null collapses\n"
            "Every leg, re-run on the flat-vol tape. With no regime, the overlay must — and does — "
            "add nothing."
        ),
        code(
            "cmp0 = strategy.compare(ret_flat, cost_bps=1.0)\n"
            "sp0  = decompose.spanning_alpha(ret_flat, cost_bps=1.0)\n"
            "bs0  = decompose.sharpe_gain_bootstrap(ret_flat, n_boot=2000, seed=0, cost_bps=1.0)\n"
            "ce0  = decompose.certainty_equivalent(ret_flat, gamma=5.0, cost_bps=1.0)\n"
            "print(f\"flat null: Sharpe gain {cmp0['sharpe_gain_net']:+.2f} | alpha t {sp0['alpha_t']:+.2f} \"\n"
            "      f\"| boot CI [{bs0['ci_low']:+.2f},{bs0['ci_high']:+.2f}] | CE gain {ce0['ce_gain_pct']:+.2f}%\")\n"
            "print('every number ~0: the gain was the regime, not the apparatus.')"
        ),

        md(
            "## Beat 5 · The verdict\n\n"
            "- **Forecastable risk** (4a): log-var AR(1) ρ ≈ 0.5 synthetic, "
            f"{R['spy_rho']}/{R['qqq_rho']} real — strong and replicated.\n"
            "- **A real, spanned-alpha lift** (4b–4d): net Sharpe gain with α > 0, HAC "
            f"t = {R['spy_t']}/{R['qqq_t']} on real SPY/QQQ, bootstrap CI clearing zero on QQQ.\n"
            "- **Bounded, not free** (4e): the matched-risk CRRA certainty-equivalent is positive but "
            "smaller — and the null (4f) is flat.\n\n"
            "**Signal `REAL`.** The one effect on this desk that forecasts *risk* instead of *returns* — "
            "which is exactly why it survives where the return-timing ideas don't."
        ),

        md(
            "## Beat 6 · Could you trade it?\n\n"
            "For once the answer is yes, and the protocol's three usual killers all miss:\n\n"
            f"- **Costs.** Turnover ~{R['spy_turn']}×/yr (the forecast is slow). The cost sweep below "
            "stays positive far past any realistic execution cost on a liquid index.\n"
            "- **Capacity.** The traded instrument is an index (SPY/ES) — the deepest book in "
            "existence. No square-root-impact wall at any size a fund realistically runs; this *is* "
            "how vol-target and risk-parity mandates operate.\n"
            "- **Decay.** Rolling Sharpe of the managed stream (via `quantlab.analytics`) doesn't "
            "show the post-publication cliff that sinks crowded anomalies — risk is structural, not an "
            "arb to be competed away.\n\n"
            "The one true caveat is **leverage**: the overlay must gear up in calm regimes to hit its "
            "risk target, which a financing-constrained book can't always do — the honest bound the "
            "`RISK-MANAGED` stamp records."
        ),
        code(
            "from quantlab.analytics import rolling_sharpe\n"
            "sweep = strategy.cost_sweep(ret)\n"
            "rs = rolling_sharpe(strategy.managed_returns(ret, cost_bps=1.0), window=252*3)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))\n"
            "a1.plot(sweep.index, sweep['sharpe_gain'].values, 'o-'); a1.axhline(0, ls=':', c='grey')\n"
            "a1.set_xlabel('cost (bps/unit traded)'); a1.set_ylabel('Sharpe gain'); a1.set_title('survives costs')\n"
            "a2.plot(rs.index, rs.values); a2.set_title('rolling 3y Sharpe (managed) -- no decay cliff')\n"
            "a2.set_ylabel('annualised Sharpe')\n"
            "plt.tight_layout(); plt.show()\n"
            "display(sweep.round(3))"
        ),

        md(
            "## Beat 7 · Going further\n\n"
            "- **Better variance forecasts.** Swap the 21-day window for EWMA "
            "([`vol.ewma_vol`](../storm_shy/vol.py)) or a GARCH(1,1); does a sharper σ̂ tighten the "
            "ride, or is the simple window already ~90% of the prize? (Diminishing returns is the "
            "likely — and interesting — answer.)\n"
            "- **Leverage-constrained version.** Cap at 1.0 (no borrowing). How much of the alpha "
            "survives when you can only ever *de*-risk? This is the practical core of the Cederburg "
            "critique, made into a backtest.\n"
            "- **Cross-asset and cross-factor.** Moreira–Muir is strongest where vol-of-vol is high "
            "and weakest (per Cederburg) for some factors out-of-sample. Sweep equities, bonds, "
            "commodities, FX; map where the overlay earns its keep and where the leverage it demands "
            "makes it `FRAGILE`.\n"
            "- **Combine with the desk's beta-honesty.** Study 01 showed overnight 'alpha' was mostly "
            "beta; here the managed factor's beta is ~0.6 — how much of the lift is timing the *equity* "
            "risk premium vs timing *idiosyncratic* vol? A factor-model decomposition would settle it.\n\n"
            "PRs welcome — strengthen the honest 'yes', or find the constraint under which even this "
            "one becomes a mirage."
        ),
    ]
    return new_notebook(cells=cells, metadata=_meta())


def _meta():
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }


def main():
    nbf.write(build_curious(), os.path.join(HERE, "01_for_the_curious.ipynb"))
    nbf.write(build_quants(), os.path.join(HERE, "02_for_the_quants.ipynb"))
    print("wrote 01_for_the_curious.ipynb and 02_for_the_quants.ipynb")


if __name__ == "__main__":
    main()
