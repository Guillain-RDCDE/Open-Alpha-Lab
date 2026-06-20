"""Build the two narrative notebooks for Study 339 (Convertible-Bonds).

    python build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        01_for_the_curious.ipynb 02_for_the_quants.ipynb

Both notebooks execute OFFLINE: each data cell tries the real tape (cached CWB/SPY/AGG/SHY)
and falls back to the deterministic synthetic tape, bannering which it ran. The real headline
numbers live in ONE dict `R` below, mirroring docs/results.md — synthetic cells never wear a
real-tape banner.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-05-31).
# CWB (SPDR Bloomberg Convertible Securities) vs SPY / AGG / SHY, 2009-04-16 → 2026-05-29.
# ---------------------------------------------------------------------------
R = dict(
    asof="2026-05-31", start="2009-04-16", end="2026-05-29", rows=4307, years=17.09,
    panel_fp="0ab3e0356db0",
    fp_cwb="1ba5397da707", fp_spy="32ea3bc507fb", fp_agg="e27a0b52d246", fp_shy="515f22fb975e",
    # convexity regression: r_CWB = a + b*r_SPY + g*up2 ; gamma is the convexity coefficient
    gamma=-1.897, t_gamma=-2.96, beta=0.658, r2=0.692,
    gamma_ci_lo=-3.484, gamma_ci_hi=-0.873, gamma_frac_pos=0.0,
    # up/down capture vs SPY
    up_capture=0.664, down_capture=0.645, asymmetry=0.019,
    # the matched-blend race (excess-of-cash Sharpe, 2 bps/rebal)
    cwb_cagr=12.0, cwb_vol=13.1, cwb_sharpe=0.817, cwb_mdd=-32.1,
    blend_cagr=11.0, blend_vol=10.9, blend_sharpe=0.879, blend_mdd=-22.0,
    spy_cagr=15.6, spy_vol=17.2, spy_sharpe=0.838, spy_mdd=-33.7,
    agg_cagr=2.8, agg_vol=4.7, agg_sharpe=0.378,
    w_stock=0.63,
    sharpe_diff=-0.062, sharpe_diff_lo=-0.308, sharpe_diff_hi=0.185, sharpe_frac=0.30,
    daily_diff_t=0.71,
    # the cushion failure — March 2020 (worst SPY month)
    mar20_spy=-12.5, mar20_cwb=-13.1, mar20_blend=-7.7,
    # subsamples (both halves negative gamma)
    g_early=-1.847, t_early=-1.56, g_late=-2.322, t_late=-3.40,
    er=0.40,  # CWB expense ratio, %/yr
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
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

from convertible_bonds import data, strategy as st


def load_tape():
    \"\"\"Real CWB/SPY/AGG/SHY frame if reachable, else the deterministic synthetic tape.\"\"\"
    try:
        frame = data.load_real(end="2026-05-31")
        if len(frame) > 1000 and {"CWB", "SPY", "AGG", "SHY"}.issubset(frame.columns):
            return frame, "real"
    except Exception as e:
        print("real load failed -> synthetic:", type(e).__name__)
    frame, _ = data.synthetic_convex(n_days=4000, gamma=0.6, seed=339)
    return frame, "synthetic"


FRAME, TAPE = load_tape()
# Column map: real tape uses tickers; synthetic uses IDX/BND/CVT.
if TAPE == "real":
    CVT, IDX, BND, CASH = "CWB", "SPY", "AGG", "SHY"
    BANNER = "REAL tape — CWB / SPY / AGG / SHY, total return"
else:
    CVT, IDX, BND, CASH = "CVT", "IDX", "BND", None
    BANNER = "SYNTHETIC tape (offline fallback — a PLANTED-convex control, NOT real CWB)"
RETS = st.to_returns(FRAME)
RF = RETS[CASH] if CASH else None
print("tape in this run:", BANNER)
print("rows:", len(FRAME), " range:", FRAME.index[0].date(), "->", FRAME.index[-1].date())
"""


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
            "# Convertible-Bonds — equity upside with a bond floor? 🪢\n"
            "### Do convertibles really catch the rally but cushion the crash?\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Equity_upside_bond_downside%3F: Busted](https://img.shields.io/badge/Equity_upside_bond_downside%3F-Busted-8b949e?style=flat-square)\n\n"
            "Convertible bonds are sold as the best of both worlds: a corporate bond with a "
            "built-in stock option, so you *convert* and ride the rally if the company soars, "
            "but you collect the coupon and lean on the bond floor if it sinks. The picture is "
            "a hockey-stick — a **convex** payoff. This notebook asks whether the most popular "
            "convertibles ETF (**CWB**) actually delivers that curve, or whether it's just a "
            "plain stock/bond mix wearing a fancier name.\n\n"
            "> 📓 **This is the plain-language layer.** Want the regressions, the bootstrap "
            "confidence intervals and the capacity math? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below "
            "is drawn by the code beside it. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first 🎯\n\n"
            f"*(Headline numbers are the **real** CWB/SPY/AGG/SHY tape, {R['start']} → "
            f"{R['end']}, from [docs/results.md](../docs/results.md). The executed cells "
            "below run on whichever tape this machine has — they banner which.)*\n\n"
            "| Question | Plain answer |\n"
            "|---|---|\n"
            "| Is the payoff actually *convex* (a hockey-stick)? | **No.** The curvature we "
            f"measure has the **wrong sign** (slightly *concave*), and it's significant "
            f"(*t* = {R['t_gamma']}) — the opposite of the brochure. |\n"
            "| Does it catch more of the rally than the fall? | **Barely.** It keeps "
            f"~{R['up_capture']:.0%} of up days and ~{R['down_capture']:.0%} of down days — "
            "almost symmetric, not the promised asymmetry. |\n"
            "| Can you replicate it with a plain stock/bond mix? | **Yes — and cheaper.** A "
            f"~{R['w_stock']:.0%} stocks / {1-R['w_stock']:.0%} bonds blend matched its "
            "risk and *edged it* on Sharpe, at a fraction of the fee. |\n\n"
            "> So a convertibles ETF behaves like **a stock/bond blend with an extra fee**. "
            "The convexity that justifies the wrapper does not show up in the tape."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "The convertibles pitch, at full strength:\n\n"
            "> *A convertible bond gives you **equity upside with bond downside**. If the "
            "stock takes off, your bond converts into shares and you ride it up. If the stock "
            "tanks, you still hold a bond paying a coupon, so a **bond floor** catches you. "
            "It's a one-way bet — heads you win like a stock, tails you lose like a bond.*\n\n"
            "Drawn as a chart, that's a **convex** payoff: a curve that bends *up*, capturing "
            "more of the gains than the losses. Convexity is the whole product."
        ),

        # ---- BEAT 2 — SO WHAT? -----------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If convexity is real and free, convertibles are a genuine free lunch — more "
            "upside per unit of downside than any plain mix of the same stock and bond. "
            "Issuers love them (cheaper coupons), and a whole fund category is sold on the "
            "curve. But there's a catch built into the idea: an *option* is never free. The "
            "convexity is *paid for* — in a lower coupon, in credit risk, in a fee. The "
            "question is whether what you get back is curvature you couldn't have bought more "
            "cheaply by just holding some stock and some bonds yourself."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Two pictures settle it:\n\n"
            "1. **Plot every day:** the stock market's move across, the convertible's move "
            "up. A *convex* payoff bends upward — a smile. A plain mix is a straight line. "
            "We literally fit the curve and read its bend.\n"
            "2. **Build the plain mix it claims to beat:** some stocks, some bonds, dialled to "
            "the convertible's risk. If the convertible can't out-curve *and* out-earn that "
            "blend, the wrapper added nothing.\n\n"
            "**What would make us say 'mirage':** if the fitted curve doesn't bend up (no "
            "convexity), if up-capture isn't clearly above down-capture, and if a cheap "
            "stock/bond blend keeps pace. (It's all three.)"
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "### Does the payoff bend upward?\n\n"
            "Each dot is one day: the stock market across, the convertible up. If the payoff "
            "were convex, a fitted curve would **bend up** — steeper on the right (rallies) "
            "than the left (selloffs). Let's fit it and see which way it bends."
        ),
        code(
            "reg = st.convexity_regression(RETS, CVT, IDX)\n"
            "print(BANNER)\n"
            "print(f'fitted payoff:  r_{CVT} = a + b*r_{IDX} + gamma*up^2')\n"
            "print(f'  slope  b     = {reg[\"beta\"]:.3f}')\n"
            "print(f'  curvature g  = {reg[\"gamma\"]:.3f}   (>0 = convex smile, <0 = concave frown)')\n"
            "print(f'  robust t(g)  = {reg[\"t_gamma\"]:.2f}')\n\n"
            "x = RETS[IDX].to_numpy(); y = RETS[CVT].to_numpy()\n"
            "grid = np.linspace(x.min(), x.max(), 200)\n"
            "fit = reg['alpha'] + reg['beta']*grid + reg['gamma']*np.maximum(grid,0)**2\n"
            "fig, ax = plt.subplots()\n"
            "ax.scatter(x*100, y*100, s=4, alpha=.15, color=GREY)\n"
            "ax.plot(grid*100, fit*100, color=(GREEN if reg['gamma']>0 else RED), lw=2.5,\n"
            "        label=f'fitted payoff (g={reg[\"gamma\"]:.2f})')\n"
            "ax.plot(grid*100, (reg['alpha']+reg['beta']*grid)*100, '--', color='k', lw=1,\n"
            "        label='straight-line mix')\n"
            "ax.axhline(0, color='k', lw=.5); ax.axvline(0, color='k', lw=.5)\n"
            "ax.set_xlabel('stock market daily move (%)'); ax.set_ylabel(f'{CVT} daily move (%)')\n"
            "ax.set_title(f'Does the convertible payoff bend up? — {BANNER}'); ax.legend()\n"
            "plt.show()"
        ),
        md(
            "> 🔬 **For the quants:** the bend is the coefficient on `up² = max(r,0)²`. On the "
            "**real** CWB tape it comes out *negative* "
            f"(g = {R['gamma']}, *t* = {R['t_gamma']}) — a slight **frown**, the opposite of "
            "the advertised smile. The synthetic fallback plants a real smile so you can see "
            "what convexity *would* look like."
        ),

        md(
            "### Does it keep more of the rally than the fall?\n\n"
            "The other way to read convexity: on days the market rose, how much did the "
            "convertible capture — versus on days it fell? Convexity means **up-capture well "
            "above down-capture**."
        ),
        code(
            "cap = st.capture_ratios(RETS, CVT, IDX)\n"
            "print(f'up-capture   : {cap[\"up_capture\"]:.3f}')\n"
            "print(f'down-capture : {cap[\"down_capture\"]:.3f}')\n"
            "print(f'asymmetry    : {cap[\"asymmetry\"]:+.3f}  (convexity wants this clearly > 0)')\n\n"
            "fig, ax = plt.subplots(figsize=(6,4.5))\n"
            "ax.bar(['up-capture','down-capture'], [cap['up_capture'], cap['down_capture']],\n"
            "       color=[GREEN, RED])\n"
            "ax.set_ylabel(f'{CVT} capture of the market'); ax.set_ylim(0, 1.1)\n"
            "ax.set_title(f'Asymmetry = {cap[\"asymmetry\"]:+.3f}  ({BANNER})')\n"
            "plt.show()"
        ),

        md(
            "### Can a plain stock/bond blend keep up?\n\n"
            "Now the head-to-head: build a simple stocks+bonds mix dialled to the "
            "convertible's equity exposure, rebalanced yearly, and race the growth of $1."
        ),
        code(
            "beta = st.beta_to(RETS, CVT, IDX)\n"
            "w = float(np.clip(beta, 0, 1))\n"
            "blend = st.matched_blend(RETS, IDX, BND, w_stock=w, rebalance='annual', cost_bps=2.0)\n"
            "eq_cvt = (1+RETS[CVT]).cumprod(); eq_bl = (1+blend).cumprod(); eq_idx = (1+RETS[IDX]).cumprod()\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(eq_cvt, color=AMBER, lw=2, label=f'{CVT} (convertibles)')\n"
            "ax.plot(eq_bl, color=GREEN, lw=2, label=f'matched blend ({w:.0%} {IDX}/{1-w:.0%} {BND})')\n"
            "ax.plot(eq_idx, color=GREY, lw=1, label=f'{IDX} (100% stocks)')\n"
            "ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "ax.set_title(f'Convertibles vs a plain matched blend — {BANNER}'); ax.legend()\n"
            "plt.show()\n"
            "for nm, s in [(CVT, RETS[CVT]), ('blend', blend), (IDX, RETS[IDX])]:\n"
            "    d = st.stats(s, rf=RF)\n"
            "    print(f'{nm:8s} CAGR={d[\"cagr\"]:+.1%}  vol={d[\"vol\"]:.1%}  Sharpe(xs)={d[\"sharpe\"]:.3f}  maxDD={d[\"max_dd\"]:.1%}')"
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Convexity — not there.** The fitted payoff doesn't bend up; on the real CWB "
            f"tape it bends slightly *down* (*t* = {R['t_gamma']}). Up-capture "
            f"({R['up_capture']:.0%}) barely beats down-capture ({R['down_capture']:.0%}). "
            "→ **Signal: None.**\n"
            "- **A plain blend matched it — cheaper.** A "
            f"{R['w_stock']:.0%}/{1-R['w_stock']:.0%} stock/bond mix edged CWB on Sharpe "
            f"({R['blend_sharpe']} vs {R['cwb_sharpe']}) at a smaller drawdown, with no "
            f"{R['er']:.2f}%/yr fee. → **Tradability: Mirage.**\n"
            "- **The bond floor failed when it mattered.** In the worst month (March 2020) "
            f"CWB fell **{R['mar20_cwb']}%** — *more* than the stock market "
            f"({R['mar20_spy']}%) and far more than the blend ({R['mar20_blend']}%). → the "
            "*'equity upside, bond downside'* story is **Busted**."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT? ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "You absolutely *can* buy CWB — it's a liquid ETF. The problem isn't access, it's "
            f"that you're paying ~{R['er']:.2f}%/yr (plus the convertibles' embedded option "
            "cost and credit risk) for a payoff you could **replicate with two of the "
            "cheapest funds on earth** (a broad stock ETF + a broad bond ETF) at near-zero "
            "fee — and the replica matched the risk-adjusted return. There's no curvature to "
            "pay up for, so the fee is dead weight."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Single-name converts vs the ETF.** A diversified ETF blends balanced, "
            "busted and equity-like converts into mush; an individual deep-in-the-money "
            "convertible *is* near-linear-to-equity, while an out-of-the-money one is "
            "near-pure-bond. Does convexity live at the single-name level even if the basket "
            "washes it out?\n"
            "- **Other convertibles funds (ICVT, FCVT).** Same test, different wrappers.\n"
            "- **Regime conditioning.** Does the curve bend up only in low-vol bull regimes?\n\n"
            "Fork the notebook, swap the ticker in `data.load_real`, and re-run."
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
            "# Convertible-Bonds — a quantitative teardown 🔬\n"
            "### Quadratic convexity regression · HAC inference · block bootstrap · matched-blend race · capacity\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Equity_upside_bond_downside%3F: Busted](https://img.shields.io/badge/Equity_upside_bond_downside%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) "
            "— *same seven beats, every claim now carrying its standard error.* We test the "
            "convertibles convexity claim as a **sign-and-significance** question about the "
            "quadratic coefficient of the payoff, and as a **replication** question against a "
            "beta-matched stock/bond blend.\n\n"
            "> ⚠️ **Not investment advice.** Real tape: CWB / SPY / AGG / SHY total return "
            "(`yfinance auto_adjust=True`), cache-first; offline fallback is a planted-convex "
            "synthetic control (machinery proof only). Methods + sources in "
            "[`docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition. "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\nfrom convertible_bonds import data, strategy as st\n"),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            f"*Real CWB/SPY/AGG/SHY tape, {R['start']} → {R['end']} "
            f"({R['rows']:,} days, panel fp `{R['panel_fp']}`), from "
            "[docs/results.md](../docs/results.md).*\n\n"
            "| Axis | Stamp | The decisive number |\n"
            "|---|---|---|\n"
            f"| **Signal** — is the payoff convex? | **None** | Quadratic gamma = "
            f"**{R['gamma']}** (HAC *t* = **{R['t_gamma']}**) — *negative*, the wrong sign; "
            f"bootstrap CI **[{R['gamma_ci_lo']}, {R['gamma_ci_hi']}]** wholly **below** 0. |\n"
            f"| **Tradability** — better than a plain blend? | **Mirage** | A "
            f"{R['w_stock']:.0%}/{1-R['w_stock']:.0%} {('SPY/AGG' if True else '')} blend "
            f"matched the beta and **edged** CWB on excess Sharpe ({R['blend_sharpe']} vs "
            f"{R['cwb_sharpe']}); the diff is insignificant (*t* = {R['daily_diff_t']}). |\n"
            f"| **Equity upside, bond downside?** | **Busted** | Up-capture {R['up_capture']} "
            f"≈ down-capture {R['down_capture']} (asym {R['asymmetry']:+}); in Mar-2020 CWB "
            f"fell {R['mar20_cwb']}% vs SPY {R['mar20_spy']}% — no floor. |\n\n"
            "> 💡 **In plain words:** the curve that *defines* a convertible — bend up, catch "
            "the rally, cushion the fall — is simply not in the CWB tape. What's left is a "
            "stock/bond blend you can buy for less."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Three testable hypotheses behind *'equity upside, bond downside'*:\n\n"
            "- **H₁ (convexity):** the convertible's return is a *convex* function of the "
            "equity index — the coefficient on `up² = max(r_idx, 0)²` is **> 0** with a "
            "robust *t* ≥ 2.\n"
            "- **H₂ (asymmetry):** up-capture is materially **above** down-capture.\n"
            "- **H₃ (irreplaceable):** a beta-matched linear stock/bond blend **cannot** "
            "match the convertible's risk-adjusted return.\n\n"
            "Convexity is real here only if **all three** hold. We reject if gamma isn't "
            "positive-significant, capture is roughly symmetric, and the blend keeps pace."
        ),

        # ---- BEAT 2 — SO WHAT? -----------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Convexity is the *only* thing a convertible offers that a stock and a bond, held "
            "separately, do not — a non-linear, asymmetric payoff. If H₁–H₃ hold, the ETF is "
            "a cheap way to rent gamma. If they fail, the category is a **linear stock/bond "
            "exposure with an option premium and a fund fee bolted on** — negative expected "
            "value versus the do-it-yourself blend, dressed as a free lunch."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "1. **Decompose** the payoff: OLS of `r_cvt` on `[1, r_idx, up²]`. Gamma is the "
            "convexity; we recover it by Frisch-Waugh-Lovell and cross-check.\n"
            "2. **Robust inference:** a Newey-West (HAC) *t* on gamma (returns are "
            "autocorrelated and heteroskedastic), plus a **circular block bootstrap** CI on "
            "gamma that preserves volatility clustering.\n"
            "3. **Asymmetry:** up-/down-capture vs the index.\n"
            "4. **Alpha vs beta / replication:** a beta-matched annual-rebalance stock/bond "
            "blend, raced on **excess-of-cash** Sharpe (SHY proxy) with a HAC *t* and block "
            "bootstrap on the Sharpe *difference*.\n"
            "5. **Capacity / cost:** the replica is two of the cheapest ETFs alive; CWB "
            f"charges ~{R['er']:.2f}%/yr — the break-even the convexity would have to beat.\n"
            "6. **Verdict** — the stamps with their numbers.\n\n"
            "> 💡 **In plain words:** we don't just check *if* the convertible went up with "
            "stocks — we check whether it went up *faster than proportionally*, which is the "
            "actual mathematical content of 'convexity'."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown\n\n"
            "### 4.1 The convexity regression (H₁)\n\n"
            "`r_cvt = α + β·r_idx + γ·up²`. γ is the bend; we want γ > 0, robust *t* ≥ 2."
        ),
        code(
            "reg = st.convexity_regression(RETS, CVT, IDX)\n"
            "print(BANNER)\n"
            "print(f'beta (linear)   : {reg[\"beta\"]:.4f}')\n"
            "print(f'gamma (convex.) : {reg[\"gamma\"]:.4f}   FWL check {reg[\"gamma_fwl\"]:.4f}')\n"
            "print(f'HAC t(gamma)    : {reg[\"t_gamma\"]:.2f}')\n"
            "print(f'R^2             : {reg[\"r2\"]:.3f}')\n"
            "verdict = 'CONVEX (smile)' if reg['gamma']>0 and reg['t_gamma']>2 else \\\n"
            "          ('CONCAVE (frown)' if reg['gamma']<0 and reg['t_gamma']<-2 else 'flat / noisy')\n"
            "print('payoff shape    :', verdict)"
        ),
        md(
            "> 💡 **In plain words:** on the real CWB tape γ comes out **negative and "
            "significant** — the payoff bends *down*, not up. That is the single cleanest "
            "rejection of the convexity thesis: not 'we found nothing', but 'we found the "
            "**opposite**'. (The synthetic fallback plants γ > 0 so the harness can prove it "
            "*detects* convexity when it's really there — a machinery check, never evidence.)"
        ),

        md("### 4.2 Bootstrap CI on gamma\n\nDoes the bend's sign survive resampling?"),
        code(
            "bg = st.bootstrap_gamma(RETS, CVT, IDX, block=21, n_boot=1000, seed=339)\n"
            "print(f'gamma point : {bg[\"point\"]:+.3f}')\n"
            "print(f'95% CI      : [{bg[\"ci95\"][0]:+.3f}, {bg[\"ci95\"][1]:+.3f}]')\n"
            "print(f'P(gamma>0)  : {bg[\"frac_pos\"]:.2%}')"
        ),

        md(
            "### 4.3 Capture asymmetry (H₂)\n\n"
            "Convexity should show as up-capture well above down-capture."
        ),
        code(
            "cap = st.capture_ratios(RETS, CVT, IDX)\n"
            "print(f'up-capture   : {cap[\"up_capture\"]:.3f}  (n={cap[\"n_up\"]})')\n"
            "print(f'down-capture : {cap[\"down_capture\"]:.3f}  (n={cap[\"n_dn\"]})')\n"
            "print(f'asymmetry    : {cap[\"asymmetry\"]:+.3f}')"
        ),

        md(
            "### 4.4 The replication race (H₃)\n\n"
            "Beta-match a stock/bond blend to the convertible, race on **excess-of-cash** "
            "Sharpe, then bootstrap the Sharpe *difference* (CWB − blend)."
        ),
        code(
            "beta = st.beta_to(RETS, CVT, IDX); w = float(np.clip(beta, 0, 1))\n"
            "blend = st.matched_blend(RETS, IDX, BND, w_stock=w, rebalance='annual', cost_bps=2.0)\n"
            "rows = []\n"
            "for nm, s in [(CVT, RETS[CVT]), ('matched blend', blend), (IDX, RETS[IDX]), (BND, RETS[BND])]:\n"
            "    d = st.stats(s, rf=RF)\n"
            "    rows.append((nm, d['cagr'], d['vol'], d['sharpe'], d['max_dd']))\n"
            "tbl = pd.DataFrame(rows, columns=['arm','CAGR','vol','Sharpe_xs','maxDD']).set_index('arm')\n"
            "print(f'matched weight w_stock = {w:.2f} (= CWB beta to {IDX})')\n"
            "tbl"
        ),
        code(
            "bs = st.bootstrap_sharpe_diff(RETS[CVT], blend, rf=RF, block=21, n_boot=1000, seed=339)\n"
            "dr = (RETS[CVT] - blend).dropna()\n"
            "print(f'Sharpe diff (CWB - blend): {bs[\"point\"]:+.3f}')\n"
            "print(f'95% CI                  : [{bs[\"ci95\"][0]:+.3f}, {bs[\"ci95\"][1]:+.3f}]')\n"
            "print(f'CWB wins the race in    : {bs[\"frac_a_wins\"]:.0%} of resamples')\n"
            "print(f'HAC t on daily (CWB-blend) return diff: {st.hac_tstat(dr.to_numpy()):.2f}')"
        ),
        md(
            "> 💡 **In plain words:** the matched blend's Sharpe sits *above* CWB's and the "
            "difference is statistically a coin flip — so CWB is, at best, indistinguishable "
            "from the cheap DIY mix, and on the point estimate a touch worse. H₃ fails."
        ),

        # ---- BEAT 5 — VERDICT ------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "| Hypothesis | Test | Real CWB tape | Holds? |\n"
            "|---|---|---|---|\n"
            f"| H₁ convexity | γ > 0, HAC *t* ≥ 2 | γ = **{R['gamma']}**, *t* = "
            f"**{R['t_gamma']}** (CI [{R['gamma_ci_lo']}, {R['gamma_ci_hi']}]) | ❌ "
            "(*wrong sign*) |\n"
            f"| H₂ asymmetry | up-cap ≫ down-cap | {R['up_capture']} vs {R['down_capture']} "
            f"(asym {R['asymmetry']:+}) | ❌ |\n"
            f"| H₃ irreplaceable | blend can't match | blend Sharpe {R['blend_sharpe']} ≥ CWB "
            f"{R['cwb_sharpe']}, diff *t* = {R['daily_diff_t']} | ❌ |\n\n"
            "All three fail. **Signal None · Tradability Mirage · 'Equity upside, bond "
            "downside' Busted.** Robust across halves: γ stays negative in "
            f"2009–2015 ({R['g_early']}, *t* {R['t_early']}) and 2016–2026 "
            f"({R['g_late']}, *t* {R['t_late']})."
        ),

        # ---- BEAT 6 — TRADE IT? ----------------------------------------------
        md(
            "## 6 · Could you trade it?\n\n"
            "There's nothing to *trade* — the question is whether to *own* CWB over the DIY "
            f"replica. CWB charges ~{R['er']:.2f}%/yr; the replica is a broad-stock ETF + a "
            "broad-bond ETF at single-digit-bp fees. For CWB to be worth the wrapper it would "
            f"need to deliver convexity worth more than ~{R['er']:.2f}%/yr — but the measured "
            "convexity is *negative*, so the fee buys you a *worse* shape than the blend. "
            "Capacity isn't the constraint (CWB is liquid); **economics** is."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Single-name decomposition.** An ETF averages over balanced / busted / "
            "equity-like converts; fit γ on individual issues bucketed by moneyness — "
            "convexity may live where the basket washes it out.\n"
            "- **Conditional γ.** Roll the quadratic regression; does the bend turn positive "
            "in calm bull regimes and negative in crashes (i.e. *negative* gamma exactly "
            "when you wanted the floor)?\n"
            "- **Credit leg.** Decompose CWB into equity + rates + credit factors; how much "
            "of the 'bond floor' is just HY credit beta (which sells off *with* equities)?\n"
            "- **Other wrappers:** ICVT, FCVT — same harness, swap the ticker.\n\n"
            "Fork it, change the ticker in `data.load_real`, re-run."
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
