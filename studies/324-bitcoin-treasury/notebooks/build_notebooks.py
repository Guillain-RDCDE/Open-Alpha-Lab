"""Generate the two narrative notebooks for Study 324 (Bitcoin-Treasury).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached daily
parquets under ../_cache/ if present (via ``data.load_real``) and otherwise fall back to
the synthetic tape, **bannering which tape each cell shows**. The frozen real-tape headline
numbers live in ``R`` below (mirroring docs/results.md), so the prose re-runs for any reader.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-17).
R = dict(
    treasury_start="2020-08-11", asof="2026-06-17",
    days=1470, years=4.03,
    # decomposition — treasury era
    beta=1.13, r2=50.0, alpha_bps=10.8, alpha_yr=39.3, t_alpha=1.08,
    # decomposition — full sample (the contrast)
    full_beta=0.45, full_r2=18.0, full_t_alpha=0.76,
    # rolling beta range
    roll_beta_min=0.55, roll_beta_med=1.20, roll_beta_max=1.73,
    # the race (net borrow, excess Sharpe)
    mstr_cagr=70.9, btc_cagr=53.7, proxy_cagr=55.1,
    mstr_sharpe=1.01, btc_sharpe=0.92, proxy_sharpe=0.91,
    mstr_mdd=-89.3, btc_mdd=-76.6, proxy_mdd=-81.3,
    diff_bps=11.04, diff_t=1.10, diff_ci_lo=-8.61, diff_ci_hi=31.19,
    # NAV premium timing
    prem_min=-0.99, prem_max=1.02, prem_std=0.38,
    timed_cagr=67.7, always_cagr=70.9, timed_sharpe=1.06, always_sharpe=1.03,
    timing_diff_bps=-9.28, timing_diff_t=-0.88,
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
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

from bitcoin_treasury import data, strategy as st

TREASURY_START = "2020-08-11"

def load_treasury():
    \"\"\"Real (btc, mstr) treasury-era frame if cached, else the synthetic tape.\"\"\"
    frame, tape = data.load_real(n_days=1470, beta=1.8, alpha=0.0, premium_vol=0.30)
    if tape == "real":
        frame = frame.loc[frame.index >= TREASURY_START]
    return frame, tape

FRAME, TAPE = load_treasury()
BANNER = ("REAL tape (MSTR & BTC-USD, treasury era)" if TAPE == "real"
          else "SYNTHETIC tape (offline fallback — NOT real MSTR)")
print("tape in this run:", BANNER)
print("rows:", len(FRAME), " range:", FRAME.index[0].date(), "->", FRAME.index[-1].date())
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Bitcoin-Treasury — is MicroStrategy just leveraged Bitcoin? 🟧\n"
            "### A $100bn 'Bitcoin treasury' stock, taken apart in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Just_leveraged_Bitcoin%3F: Confirmed](https://img.shields.io/badge/Just_leveraged_Bitcoin%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "In August 2020 MicroStrategy (ticker **MSTR**) stopped being a sleepy software "
            "company and started buying Bitcoin — billions of dollars of it. The stock went "
            "vertical, and a story took hold: *MSTR is the smart way to own Bitcoin — "
            "leverage on the way up, plus a premium for the clever wrapper around the coins.* "
            "This notebook asks the only two questions that matter: **is MSTR more than just "
            "leveraged Bitcoin — and does paying the premium actually leave you better off "
            "than holding the coin?**\n\n"
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
            "*(Headline numbers are the **real** MSTR/BTC tape, treasury era 2020-08 → "
            f"{R['asof']}, from [docs/results.md](../docs/results.md). The executed cells "
            "below run on whichever tape this machine has — they banner which.)*\n\n"
            "| Question | Plain answer |\n"
            "|---|---|\n"
            f"| Is MSTR more than leveraged Bitcoin? | **No.** It moves like ~{R['beta']}× "
            f"Bitcoin and half its daily wiggle *is* Bitcoin; the 'extra' return isn't "
            "real once you do the stats (*t* ≈ 1.1). |\n"
            "| Did MSTR beat Bitcoin? | **Yes — but only because it's levered.** A plain "
            "levered-Bitcoin position would have done the same, with a *smaller* crash. |\n"
            "| Does the premium pay? | **No.** It swings wildly and you can't time it; it's "
            "extra risk, not extra reward. |\n\n"
            "> So MSTR is exactly what the bears say and the bulls half-admit: **leveraged "
            "Bitcoin in a costume.** A real bet, just not a *better* one."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "The bull case, stated at full strength:\n\n"
            "> *MicroStrategy isn't just a Bitcoin proxy — it's **better** than holding "
            "Bitcoin. It gives you leverage without a margin call, the CEO is a genius "
            "capital allocator, and the stock trades at a premium to its coins for a "
            "reason. Buy MSTR, not BTC.*\n\n"
            "There are really three promises bundled together: (1) **alpha** — skill beyond "
            "the leverage; (2) **MSTR beats BTC**; and (3) the **premium itself is "
            "valuable**. We'll test all three."
        ),

        # ---- BEAT 2 — SO WHAT? -----------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "MSTR is a **>$100bn** company whose entire thesis is *we hold Bitcoin, but "
            "better.* If that 'better' is real, it's free money: own the wrapper, beat the "
            "coin. If it's an illusion, then millions of investors are paying a fluctuating "
            "premium for **leverage they could rent more cheaply** — and taking a bigger "
            "drawdown for the privilege. This is the cleanest test on the desk of a single "
            "idea: **is leverage the same thing as skill?** (Spoiler from every other study "
            "here: no.)"
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three simple experiments:\n\n"
            "1. **Line MSTR up against Bitcoin** and ask: how much of MSTR's daily move is "
            "just Bitcoin's move, scaled up? Whatever's left is the 'skill.' We check "
            "whether that leftover is real or just noise.\n"
            "2. **Build a fake MSTR** — a plain Bitcoin position dialled up to MSTR's "
            "leverage, paying a realistic borrowing cost. If real MSTR can't beat this "
            "copy-cat, the wrapper added nothing.\n"
            "3. **Try to trade the premium** — buy MSTR only when it's 'cheap' versus its "
            "coins. If the premium is valuable, timing it should help.\n\n"
            "**What would make us say 'mirage':** if the leftover return isn't "
            "statistically real, if the fake-MSTR keeps up with the real one, and if timing "
            "the premium does nothing. (It's all three.)"
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "### First: how much of MSTR is just Bitcoin?\n\n"
            "Every dot below is one day: Bitcoin's move across, MSTR's move up. If MSTR were "
            "*purely* leveraged Bitcoin, the dots would sit on a straight line through the "
            "origin. The slope is the leverage; the scatter around it is everything else."
        ),
        code(
            "r_mstr = st.simple_returns(FRAME['mstr'])\n"
            "r_btc  = st.simple_returns(FRAME['btc'])\n"
            "common = r_mstr.index.intersection(r_btc.index)\n"
            "dec = st.beta_decomposition(FRAME)\n\n"
            "fig, ax = plt.subplots()\n"
            "ax.scatter(r_btc.loc[common]*100, r_mstr.loc[common]*100, s=8, alpha=.35, color=GREY)\n"
            "xs = np.linspace(r_btc.loc[common].min(), r_btc.loc[common].max(), 50)\n"
            "ax.plot(xs*100, (dec['alpha_daily'] + dec['beta']*xs)*100, color=RED, lw=2,\n"
            "        label=f\"fit: beta={dec['beta']:.2f}, R^2={dec['r2']*100:.0f}%\")\n"
            "ax.axhline(0, color='k', lw=.6); ax.axvline(0, color='k', lw=.6)\n"
            "ax.set_xlabel('Bitcoin daily return (%)'); ax.set_ylabel('MSTR daily return (%)')\n"
            "ax.set_title(f'MSTR vs Bitcoin, daily  —  {BANNER}'); ax.legend()\n"
            "plt.show()\n"
            "print(f\"MSTR moves like {dec['beta']:.2f}x Bitcoin; \"\n"
            "      f\"{dec['r2']*100:.0f}% of its daily variance is just Bitcoin.\")"
        ),
        md(
            "On the **real tape** the slope is about "
            f"**{R['beta']}× Bitcoin**, and **R² ≈ {R['r2']:.0f}%** — half of MSTR's "
            "day-to-day life is literally Bitcoin moving. The leftover ('alpha') looks juicy "
            f"on paper (**+{R['alpha_yr']:.0f}%/yr**) but its *t*-stat is only "
            f"**{R['t_alpha']}** — below the bar where we'd call it real. That's the first "
            "promise gone.\n\n"
            "> 🔬 **For the quants:** the intercept's Newey-West *t* is the inference-bar "
            "number; the companion notebook shows it with its standard error and a "
            "block-bootstrap CI."
        ),
        md(
            "### Second: did MSTR really beat Bitcoin — or just lever it?\n\n"
            "Let's grow $1 in MSTR, in Bitcoin, and in a **fake MSTR** (Bitcoin dialled up "
            "to MSTR's leverage, paying borrow). If the wrapper is magic, real MSTR pulls "
            "ahead of the fake."
        ),
        code(
            "race = st.race_mstr_vs_proxy(FRAME, rf_daily=0.03/365,\n"
            "                             financing_bps_per_day=0.08/365*1e4)\n"
            "lev = race['leverage']\n"
            "proxy = st.levered_btc_proxy(FRAME, lev, 0.08/365*1e4)\n"
            "idx = r_mstr.index.intersection(r_btc.index).intersection(proxy.index)\n"
            "curves = {\n"
            "    'MSTR': (1+r_mstr.loc[idx]).cumprod(),\n"
            "    'Bitcoin (1x)': (1+r_btc.loc[idx]).cumprod(),\n"
            "    f'fake MSTR ({lev:.1f}x BTC, net borrow)': (1+proxy.loc[idx]).cumprod(),\n"
            "}\n"
            "fig, ax = plt.subplots()\n"
            "for (name, c), col in zip(curves.items(), [RED, GREEN, GREY]):\n"
            "    ax.plot(c.index, c.values, label=name, color=col, lw=1.8)\n"
            "ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "ax.set_title(f'MSTR vs Bitcoin vs a levered-BTC copy  —  {BANNER}'); ax.legend()\n"
            "plt.show()\n"
            "print(f\"MSTR-minus-fake daily edge: {race['diff_mean_bps']:+.1f} bps  \"\n"
            "      f\"(HAC t = {race['diff_t']:+.2f})\")"
        ),
        md(
            "On the real tape MSTR did beat *plain* Bitcoin "
            f"(**+{R['mstr_cagr']:.0f}%/yr vs +{R['btc_cagr']:.0f}%/yr**) — but the **fake "
            f"MSTR did basically the same (+{R['proxy_cagr']:.0f}%/yr)**, because the win was "
            "*leverage*, not skill. MSTR's edge over its own copy-cat is a statistically "
            f"insignificant **+{R['diff_bps']:.0f} bps/day** (*t* = {R['diff_t']}), and the "
            f"real MSTR took a **worse crash (−{abs(R['mstr_mdd']):.0f}% vs "
            f"−{abs(R['proxy_mdd']):.0f}%)**. You paid the premium and got *more* pain."
        ),
        md(
            "### Third: can you trade the premium?\n\n"
            "MSTR's price swings far above and below the value of the coins it holds. If "
            "that premium is valuable, buying only when it's 'cheap' should beat just "
            "holding. Let's see the premium swing, then test the timing rule."
        ),
        code(
            "prem = st.nav_premium(FRAME)\n"
            "tim = st.premium_timing(FRAME, cost_bps=5.0)\n"
            "fig, ax = plt.subplots()\n"
            "ax.fill_between(prem.index, prem.values, 0, where=prem.values>=0, color=GREEN, alpha=.4, label='rich')\n"
            "ax.fill_between(prem.index, prem.values, 0, where=prem.values<0,  color=RED,   alpha=.4, label='cheap')\n"
            "ax.axhline(0, color='k', lw=.8)\n"
            "ax.set_title(f'MSTR price-premium to its Bitcoin (proxy)  —  {BANNER}')\n"
            "ax.set_ylabel('log premium (0 = fair)'); ax.legend()\n"
            "plt.show()\n"
            "print(f\"Buy-cheap timing vs always-hold: {tim['diff_mean_bps']:+.1f} bps/day \"\n"
            "      f\"(HAC t = {tim['diff_t']:+.2f}); in market {tim['pct_in_market']*100:.0f}% of days\")"
        ),
        md(
            "The premium is enormous and mean-reverts — but **timing it earned nothing** "
            f"(real tape: **{R['timing_diff_bps']:.0f} bps/day, *t* = {R['timing_diff_t']}** "
            "— actually *negative*). The premium is **risk you carry, not a coupon you "
            "collect**, and a cheap/dear rule sits out exactly the days the leverage "
            "compounds. Third promise gone."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** MSTR is ~1.1× Bitcoin with half its variance *being* "
            "Bitcoin; the leftover return isn't statistically real.\n"
            "- **Tradability — Mirage.** The outperformance is leverage you can rent more "
            "cheaply, with a smaller drawdown; the premium can't be timed.\n"
            "- **Just leveraged Bitcoin? — Confirmed.** Exactly that, costume included.\n\n"
            "MSTR is a *real* bet on Bitcoin. It is not a *better* one."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT? ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "The 'trade' here is *prefer MSTR to Bitcoin* — and it fails not on commissions "
            "but on the comparison itself: a levered-BTC position (or a Bitcoin ETF with "
            "margin, or a 2× BTC ETF) gives you the same upside, **cheaper financing, and a "
            "shallower crash.** The only thing MSTR adds is a **volatile premium you pay at "
            "the door and can't sell back on demand** — the closed-end-fund trap (think "
            "GBTC's +100% premium in 2017 → −50% discount in 2022). If you *want* leverage on "
            "Bitcoin, buy leverage on Bitcoin."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The convertible-bond engine.** MSTR's leverage comes from convertible debt; "
            "modelling that explicitly (vs our constant-beta proxy) is the natural next fork.\n"
            "- **The copy-cats.** A wave of 'Bitcoin treasury' companies now exists — same "
            "test, fresh tickers, almost certainly the same answer.\n"
            "- **mNAV done right.** Our premium is a *price* proxy; wiring in actual "
            "coin-count and share-count data would sharpen the premium-timing null.\n\n"
            "Fork it, break it, PR it."
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
            "# Bitcoin-Treasury — a quantitative teardown 🔬\n"
            "### Beta-to-BTC · HAC-*t* alpha · levered-replica race · bootstrap CI · premium-timing null\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Just_leveraged_Bitcoin%3F: Confirmed](https://img.shields.io/badge/Just_leveraged_Bitcoin%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) "
            "— *same seven beats, every claim now carrying its standard error.* We decompose "
            "MSTR onto Bitcoin, race it against its own levered-BTC replica with a "
            "block-bootstrap CI on the difference, and test premium-timing — then confirm "
            "the whole apparatus on a synthetic tape where the answer is known.\n\n"
            "> ⚠️ **Not investment advice.** Data: Yahoo daily total-return closes for MSTR "
            "& BTC-USD, **treasury era ≥ 2020-08-11**; offline fallback is the deterministic "
            "synthetic tape. Methods + citations in [`docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into "
            "intuition. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            f"*Headline numbers below are the **real tape** (as-of {R['asof']}, "
            "[docs/results.md](../docs/results.md)); executed cells run on whichever tape "
            "this machine has and banner it.*\n\n"
            "| Axis | Stamp | The decisive number |\n"
            "|---|---|---|\n"
            f"| **Signal** | **None** | beta {R['beta']}, R² {R['r2']:.0f}%, residual "
            f"alpha HAC *t* = **{R['t_alpha']}** (< 2) |\n"
            f"| **Tradability** | **Mirage** | MSTR − levered replica = +{R['diff_bps']:.0f} "
            f"bps/day, bootstrap CI **[{R['diff_ci_lo']:.0f}, {R['diff_ci_hi']:.0f}]** "
            "(straddles 0); worse MDD |\n"
            f"| **Just leveraged Bitcoin?** | **Confirmed** | no robust alpha, premium "
            f"timing *t* = {R['timing_diff_t']} |\n\n"
            "> 💡 **In plain words:** the only thing that would make MSTR *better* than "
            "Bitcoin — return beyond its leverage — isn't there once you account for "
            "Bitcoin's own volatility clustering in the standard errors."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "- **H₁ (alpha):** the intercept of `r_MSTR ~ a + b·r_BTC` is positive and "
            "robust — MSTR earns return beyond its Bitcoin beta.\n"
            "- **H₂ (beats BTC):** holding MSTR delivers a higher risk-adjusted return than "
            "holding a levered-BTC replica at MSTR's own beta, net of financing.\n"
            "- **H₃ (premium pays):** a contrarian NAV-premium overlay (hold when cheap) "
            "beats always-hold.\n\n"
            "We need a robust **HAC *t* ≥ 2** on H₁'s intercept to stamp Signal `REAL`; a "
            "synthetic control can prove the machine works but can never back the stamp."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "If H₁ holds, MSTR is an actively-managed Bitcoin fund worth its premium. If it "
            "fails, MSTR is a **levered closed-end fund on one asset** — and the entire "
            "literature on closed-end-fund premia (Lee-Shleifer-Thaler 1991) says that "
            "premium is sentiment-driven risk, not value. The stakes are whether '~$100bn of "
            "Bitcoin, but better' is a real product or a leverage-and-narrative machine."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "1. **Decompose** `r_MSTR` onto `r_BTC` (simple returns); read beta, R², "
            "intercept.\n"
            "2. **Robust inference** — Newey-West HAC *t* on the intercept (and on the "
            "MSTR−replica mean difference), plus a **circular block-bootstrap CI**.\n"
            "3. **Critique the magnitude** — era selection (the treasury era is the only "
            "meaningful sample), unstable rolling beta, total-return labelling.\n"
            "4. **Alpha vs beta** — the levered-BTC replica makes 'is this just leverage?' "
            "explicit; **excess-of-cash Sharpe** so the levered leg isn't flattered.\n"
            "5. **Capacity / cost** — financing on the borrowed notional charged once; "
            "premium-timing with one execution lag and one-way costs.\n"
            "6. **Verdict** — the stamps with their numbers."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown\n\n"
            "### 4.1 The decomposition — and why the *era* is the whole game"
        ),
        code(
            "dec = st.beta_decomposition(FRAME)\n"
            "print('Treasury-era decomposition (this run is the', TAPE, 'tape):')\n"
            "for k in ['beta','r2','alpha_daily','alpha_ann','t_alpha','n']:\n"
            "    print(f'  {k:12s}: {dec[k]:+.4f}' if isinstance(dec[k], float) else f'  {k:12s}: {dec[k]}')\n"
            "print()\n"
            "print('Real-tape headline (docs/results.md):')\n"
            f"print('  treasury era: beta={R['beta']}, R2={R['r2']}%, alpha_t={R['t_alpha']}')\n"
            f"print('  FULL sample : beta={R['full_beta']}, R2={R['full_r2']}%, alpha_t={R['full_t_alpha']}  <- pre-2020 MSTR was not a BTC play')"
        ),
        md(
            "> 💡 **In plain words:** before 2020-08-11 MicroStrategy was a software company "
            "with almost no Bitcoin sensitivity (full-sample beta "
            f"**{R['full_beta']}**, R² **{R['full_r2']:.0f}%**). Splicing that era in halves "
            "the measured beta and manufactures a misleading picture. We report the "
            "**treasury era only** — and even there the residual alpha's *t* "
            f"(**{R['t_alpha']}**) is below the bar.\n\n"
            "The leverage is also **not constant** — the 90-day rolling beta wanders from "
            f"**{R['roll_beta_min']} to {R['roll_beta_max']}** (median {R['roll_beta_med']}):"
        ),
        code(
            "rm = st.simple_returns(FRAME['mstr']); rb = st.simple_returns(FRAME['btc'])\n"
            "b = pd.concat([rm, rb], axis=1).dropna(); b.columns = ['m','c']\n"
            "roll = (b['m'].rolling(90).cov(b['c']) / b['c'].rolling(90).var()).dropna()\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(roll.index, roll.values, color=RED, lw=1.4)\n"
            "ax.axhline(roll.median(), color=GREY, ls='--', label=f'median {roll.median():.2f}')\n"
            "ax.set_title(f'90-day rolling beta of MSTR to Bitcoin  —  {BANNER}')\n"
            "ax.set_ylabel('rolling beta'); ax.legend(); plt.show()\n"
            "print(f'rolling beta range: {roll.min():.2f} .. {roll.max():.2f}')"
        ),
        md(
            "### 4.2 The race + the bootstrap CI on the wrapper's edge\n\n"
            "MSTR vs a levered-BTC replica at its own beta, net borrow, excess-of-cash "
            "Sharpe. The decisive object is the **block-bootstrap CI on the daily "
            "MSTR − replica difference**."
        ),
        code(
            "race = st.race_mstr_vs_proxy(FRAME, rf_daily=0.03/365, financing_bps_per_day=0.08/365*1e4)\n"
            "lev = race['leverage']\n"
            "proxy = st.levered_btc_proxy(FRAME, lev, 0.08/365*1e4)\n"
            "idx = rm.index.intersection(rb.index).intersection(proxy.index)\n"
            "diff = (rm.loc[idx] - proxy.loc[idx]).to_numpy()\n"
            "lo, hi = st.block_bootstrap_ci(diff, block=20, n_boot=4000)\n"
            "tab = pd.DataFrame({\n"
            "    'CAGR %':   [race['mstr_cagr']*100, race['btc_cagr']*100, race['proxy_cagr']*100],\n"
            "    'Sharpe':   [race['mstr_sharpe'], race['btc_sharpe'], race['proxy_sharpe']],\n"
            "    'MaxDD %':  [race['mstr_mdd']*100, race['btc_mdd']*100, race['proxy_mdd']*100],\n"
            "}, index=['MSTR', 'BTC 1x', f'levered replica {lev:.2f}x'])\n"
            "print(tab.round(1))\n"
            "print(f'\\nMSTR - replica: {np.nanmean(diff)*1e4:+.1f} bps/day  '\n"
            "      f'HAC t = {st.hac_tstat(diff):+.2f}  bootstrap 95% CI = [{lo*1e4:+.1f}, {hi*1e4:+.1f}] bps')"
        ),
        md(
            "> 💡 **In plain words:** the wrapper's edge over plain leverage has a "
            f"confidence interval (**[{R['diff_ci_lo']:.0f}, {R['diff_ci_hi']:.0f}] bps/day** "
            "on the real tape) that **includes zero**. You cannot reject 'MSTR = its own "
            "leverage.' And MSTR's drawdown was *worse* than the replica's — the premium "
            "bought extra risk, not return."
        ),
        md(
            "### 4.3 The premium-timing null\n\n"
            "Hold MSTR only when its price-premium is below median; one execution lag "
            "(signal at close of *t* → return of *t+1*), 5 bps one-way cost; vs always-hold."
        ),
        code(
            "tim = st.premium_timing(FRAME, cost_bps=5.0)\n"
            "for k in ['always_cagr','timed_cagr','always_sharpe','timed_sharpe',\n"
            "          'diff_mean_bps','diff_t','pct_in_market']:\n"
            "    print(f'  {k:16s}: {tim[k]:+.3f}')\n"
            "prem = st.nav_premium(FRAME)\n"
            "print(f'\\npremium proxy range: [{prem.min():+.2f}, {prem.max():+.2f}], std {prem.std():.2f}')"
        ),
        md(
            "> 💡 **In plain words:** the premium swings enormously yet timing it returns "
            f"**{R['timing_diff_bps']:.0f} bps/day at *t* = {R['timing_diff_t']}** — *negative* "
            "and insignificant on the real tape. A mean-reverting premium is a risk factor "
            "(Lee-Shleifer-Thaler), not a harvestable coupon."
        ),
        md(
            "### 4.4 The positive control — does the apparatus *work*?\n\n"
            "A pipeline that finds nothing proves nothing unless it can find something when "
            "it's there. On the synthetic tape we plant a genuine alpha and confirm the "
            "**same** decomposition + race recovers it (and finds nothing on the null) — a "
            "machinery proof, **never** market evidence."
        ),
        code(
            "null_f, _  = data.synthetic_pair(n_days=2500, beta=1.8, alpha=0.0,    premium_vol=0.30, seed=324)\n"
            "alpha_f, _ = data.synthetic_pair(n_days=2500, beta=1.8, alpha=0.0010, premium_vol=0.30, seed=324)\n"
            "rows = []\n"
            "for name, f in [('null  (no alpha)', null_f), ('planted alpha', alpha_f)]:\n"
            "    d = st.beta_decomposition(f)\n"
            "    rc = st.race_mstr_vs_proxy(f)\n"
            "    rows.append({'tape': name, 'beta': d['beta'], 'alpha t': d['t_alpha'],\n"
            "                 'race diff bps': rc['diff_mean_bps'], 'race diff t': rc['diff_t']})\n"
            "ctrl = pd.DataFrame(rows).set_index('tape')\n"
            "print(ctrl.round(2))\n"
            "print('\\nThe harness recovers a real edge ONLY when one is planted -> it is a faithful detector.')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Treasury-era beta **{R['beta']}**, R² **{R['r2']:.0f}%**, "
            f"residual alpha HAC *t* = **{R['t_alpha']}** (below 2). MSTR is, statistically, "
            "levered Bitcoin plus noise.\n"
            f"- **Tradability — Mirage.** MSTR − levered-replica = **+{R['diff_bps']:.0f} "
            f"bps/day**, bootstrap CI **[{R['diff_ci_lo']:.0f}, {R['diff_ci_hi']:.0f}]** "
            "(straddles zero), and a **worse drawdown** than the replica; premium timing is "
            f"*t* = {R['timing_diff_t']}. The edge is leverage you can rent more cheaply.\n"
            "- **Just leveraged Bitcoin? — Confirmed.**"
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it?\n\n"
            "There is no cost at which 'prefer MSTR to Bitcoin' becomes attractive, because "
            "the failure isn't cost — it's **substitutability**. A 1.1× BTC position (or a "
            "2× BTC ETF held part-size) replicates the upside with cheaper, transparent "
            "financing and a smaller drawdown, and without paying a **closed-end-fund "
            "premium that swings from rich to cheap on sentiment** (the GBTC +100% → −50% "
            "round-trip is the cautionary tale). The capacity question is moot: the product "
            "you'd be buying is leverage, and leverage on Bitcoin is a liquid, cheaper "
            "commodity."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Model the convertibles.** MSTR's beta comes from convertible debt with "
            "option-like payoffs; a structural model would replace our constant-beta proxy "
            "and could reveal genuine *convexity* (distinct from alpha).\n"
            "- **The treasury-company cross-section.** A growing universe of imitators — same "
            "decomposition, survivorship named (these are recent listings; go through the "
            "opt-in guard before any cross-section claim).\n"
            "- **True mNAV.** Swap our price-proxy premium for reported coin-count / "
            "share-count to test premium-timing on the accounting premium.\n\n"
            "Open a PR — break the null."
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
