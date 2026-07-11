"""Generate the two narrative notebooks for Study 658 (Put-Write-Premium).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached PUTW/SPY/BIL
panel under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance PUTW/SPY/BIL,
# 2016-02-24 -> 2026-06-30, 2,600 daily returns; window bound by PUTW's inception).
R = dict(
    start="2016-02-24", end="2026-06-30", n=2600, n_rows=2601,
    putw_excess_ann=6.78, putw_excess_t=2.00,
    spy_excess_ann=14.26, spy_excess_t=2.90,
    gap_ann=-7.48, gap_t=-3.21,
    sharpe_putw=0.53, sharpe_spy=0.80, sharpe_gap=-0.27,
    boot_lo=-0.48, boot_hi=-0.00, boot_win_pct=2.4,
    alpha_ann=-1.78, alpha_t=-0.95, beta=0.600, beta_t=10.26,
    cb_normal=0.551, cb_normal_t=14.05, cb_extra=0.382, cb_extra_t=2.34,
    cb_total=0.932, n_crash_days=33,
    volm_putw=-8.2, volm_spy=-10.1, covid_putw=-28.4, covid_spy=-33.7,
    mdd_putw=-28.4, mdd_spy=-33.7,
    n_months=124, n_up=89, n_dn=35, cap_up=59.3, cap_dn=65.8,
    worst_m_putw=-13.1, worst_m_spy=-12.5, worst_m_date="2020-03",
    vol_putw=12.7, vol_spy=17.8,
    skew_putw=-1.80, skew_spy=-0.32,
    worst_d_putw=-10.51, worst_d_spy=-10.94,
    syn_null_t_mean=-0.53, syn_null_t_sd=0.98, syn_null_fire=1,
    syn_planted_alpha=4.69, syn_planted_t=4.25,
    fp="3a3191a10828",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Truncated_beta%3F: Confirmed](https://img.shields.io/badge/Truncated_beta%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from put_write_premium import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PX = data.load_real()
    RET = st.daily_returns(PX)
    EX = st.excess(RET, cash_col=data.CASH)
else:
    PX = RET = EX = None
print("real cache present:", HAVE_REAL, "| joint trading days:", (0 if PX is None else len(PX)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# \"Get paid for insurance you sell yourself\" — does put-writing actually beat "
            "just owning stocks? 🎯\n"
            "### The CBOE PUT-write pitch, tested on the one fund that actually does it\n\n"
            + BADGES +
            "Every option-income newsletter has some version of this trade: sell a put on the "
            "S&P 500 every month, collect the premium, and if the market falls you just end up "
            "buying stocks at a discount anyway. \"You get paid whether the market goes up, down, "
            "or sideways.\" The academic version has a real name — the **variance risk premium** "
            "— and a real, tradable wrapper: **PUTW**, an ETF that has done exactly this, live, "
            "since February 2016.\n\n"
            "So: does it work? Not \"is the premium real in theory\" — does the actual fund, over "
            "its actual decade, actually beat just buying the index?\n\n"
            "> 📓 **Plain-language layer.** Want the CAPM regression, the crash-beta interaction "
            "and the bootstrap? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** PUTW's *entire* live history (2016-02-24 → 2026-06-30) against "
            "SPY, with BIL (T-bills) as cash. We don't borrow the untradeable 1986→ CBOE PUT "
            "index's longer, friendlier sample — that would credit a return nobody could have "
            "banked. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Did PUTW beat SPY, risk-adjusted, since 2016? | **No.** Sharpe **{R['sharpe_putw']:.2f} "
            f"vs {R['sharpe_spy']:.2f}** — and the gap is statistically real (a bootstrap "
            f"confidence interval that stays negative). |\n"
            f"| Is any of its return something *beyond* just holding less S&P 500? | **No.** "
            f"Once you control for the fact that it moves about **60%** as much as the market, "
            "there's nothing left over — no measurable extra \"premium.\" |\n"
            "| Does the lower risk protect you in a crash? | **Only sometimes — and it fades "
            "exactly when you'd need it.** In an ordinary week PUTW moves about 55% as much as "
            "SPY; on the worst days in the whole sample that number jumps to **93%** — almost "
            "full market risk, right when a real diversifier should be doing its job. |\n"
            f"| So what did selling puts for a decade actually get you? | A smaller, less "
            f"stable slice of the S&P 500 — one whose single worst month "
            f"(**{R['worst_m_putw']:+.1f}%**, March 2020) was actually *worse* than the S&P's own "
            f"worst month (**{R['worst_m_spy']:+.1f}%**), despite carrying less average risk. |\n\n"
            "> The premium exists on paper. The fund that's supposed to bank it didn't."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Options are systematically overpriced for the risk they carry — sellers of "
            "insurance get compensated more than the insurance is actuarially 'worth.' Sell puts "
            "on the S&P every month, collect that premium, forever, and you'll do at least as "
            "well as buy-and-hold with a smoother ride.\"*\n\n"
            "It's not a strawman: the CBOE has run an actual index on exactly this rule since "
            "1986, and over that long sample it more or less matched the S&P's return at roughly "
            "two-thirds the volatility — a genuinely better historical Sharpe ratio. That's the "
            "steelman. The question is whether the fund you can actually *buy* today delivers "
            "the same thing."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If put-writing really banks a distinct risk-adjusted premium, it's one of the "
            "cleanest \"free lunches\" in finance — a way to earn more per unit of risk than the "
            "market itself pays, just by selling options instead of buying stock. It would be a "
            "genuine alternative to a plain S&P 500 index fund, not just a repackaging of the "
            "same market exposure with extra steps and an expense ratio."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The vehicle.** PUTW's entire live history, **{R['start']} → {R['end']}** "
            f"({R['n']:,} trading days) — not the untradeable index, which has a much longer and "
            "friendlier sample nobody could have actually captured.\n"
            "- **The comparison.** Sharpe ratio (return per unit of risk, both measured against "
            "T-bills) against SPY, with a statistical confidence interval on the gap — not just "
            "a single number that could flip in a friendlier slice of history.\n"
            "- **The \"just beta\" check.** A regression that asks: once you account for the fact "
            "that PUTW simply moves *less* than the market, is there anything left over?\n"
            "- **The crash check.** Does the 'moves less than the market' behavior hold up on the "
            "worst days, or does it quietly disappear exactly when it would matter?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the scoreboard.** Ten years of writing puts vs ten years of just holding "
            "the index."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sh_p = st.sharpe_ann(EX['PUTW'].to_numpy())\n"
            "    sh_s = st.sharpe_ann(EX['SPY'].to_numpy())\n"
            "else:\n"
            "    sh_p, sh_s = R['sharpe_putw'], R['sharpe_spy']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.4))\n"
            "ax.bar(['PUTW\\n(put-writer)', 'SPY\\n(buy & hold)'], [sh_p, sh_s], color=[RED, GREY], width=.55)\n"
            "for i, v in enumerate([sh_p, sh_s]): ax.annotate(f'{v:.2f}', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('Sharpe ratio (excess of T-bills)')\n"
            "ax.set_title('The pitch says put-writing wins on a risk-adjusted basis. It didn\\'t.')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Sharpe PUTW {sh_p:.2f}  vs SPY {sh_s:.2f}')"
        ),
        md(
            f"PUTW's Sharpe (**{R['sharpe_putw']:.2f}**) trails SPY's (**{R['sharpe_spy']:.2f}**) "
            f"over the fund's entire life. A statistical test (in the quants notebook) confirms "
            "this isn't just noise — a bootstrap on the gap stays negative essentially every "
            "time you resample the decade.\n\n"
            "**Next, the \"is it just beta\" check.** PUTW moves less than the market — is that "
            "the whole story?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cb = st.capm_alpha_beta(EX['PUTW'].to_numpy(), EX['SPY'].to_numpy())\n"
            "    beta, alpha = cb['beta'], cb['alpha_ann_pct']\n"
            "else:\n"
            "    beta, alpha = R['beta'], R['alpha_ann']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['market beta\\n(explained by SPY)', 'leftover \"alpha\"\\n(unexplained)'],\n"
            "       [beta, alpha/100], color=[GREY, RED], width=.55)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('beta (left bar)  /  alpha as a fraction (right bar)')\n"
            "ax.set_title('~60% is just market beta. What\\'s left over is statistically nothing.')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'beta {beta:.2f}  alpha {alpha:+.2f}%/yr')"
        ),
        md(
            f"PUTW moves about **{R['beta']:.0%}** as much as SPY on an average day — that's a "
            "real, strongly-measured number. What's left over after accounting for that beta is "
            f"an \"alpha\" of **{R['alpha_ann']:+.2f}%/yr** that is **statistically "
            "indistinguishable from zero** — the point estimate is even slightly negative. "
            "There is no detectable premium beyond holding a smaller amount of stock.\n\n"
            "**Finally, the crash check.** Does the 'lower risk' actually protect you when it "
            "counts?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cx = st.crash_beta_interaction(EX['PUTW'].to_numpy(), EX['SPY'].to_numpy(),\n"
            "                                    RET['SPY'].to_numpy(), threshold=data.CRASH_DAY_THRESHOLD)\n"
            "    b_norm, b_crash = cx['beta_normal'], cx['crash_beta_total']\n"
            "else:\n"
            "    b_norm, b_crash = R['cb_normal'], R['cb_total']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.4))\n"
            "ax.bar(['an ordinary day', 'the worst days\\n(SPY down 3%+)'], [b_norm, b_crash],\n"
            "       color=[GREY, RED], width=.55)\n"
            "for i, v in enumerate([b_norm, b_crash]): ax.annotate(f'{v:.0%}', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(1.0, ls='--', c='k', lw=1, label='same risk as the market (beta = 1)')\n"
            "ax.set_ylabel('how much PUTW moves per 1% SPY move')\n"
            "ax.set_title('The \"lower risk\" fades away exactly on the worst days')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'normal-day beta {b_norm:.2f}  crash-day beta {b_crash:.2f}')"
        ),
        md(
            f"On an ordinary day, PUTW moves about **{R['cb_normal']:.0%}** as much as the S&P — "
            "the \"you carry less risk\" story looks true. But on the worst days in the whole "
            f"decade (the S&P down 3% or more), that number jumps to **{R['cb_total']:.0%}** — "
            "almost full market risk, precisely when a real diversifier is supposed to prove its "
            "worth. And the clincher: despite averaging less volatility than the S&P, PUTW's "
            f"single worst month (**{R['worst_m_putw']:+.1f}%**, March 2020) was *worse* than the "
            f"S&P's own worst month (**{R['worst_m_spy']:+.1f}%**). That's not a smoother ride — "
            "it's a ride that's smaller most of the time and just as rough, or rougher, on the "
            "days that matter most."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Over its entire live history, PUTW trailed SPY on both raw "
            "excess return and Sharpe, and any \"premium\" over cash it did earn disappears once "
            "you control for the market exposure it's carrying.\n"
            "- **Tradability — Mirage.** There's no edge to actually deploy — it's a smaller, "
            "less stable version of the index it's compared against, with the diversification "
            "benefit disappearing right when a crash hits.\n"
            "- **\"Is it just truncated equity beta?\" — Confirmed.** ~60% market beta explains "
            "the average day; nothing measurable is left over; and even that beta isn't stable "
            "— it grows toward full market risk exactly in the tail."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The variance risk premium is real — just not free money.** Sibling study "
            "[130-vol-risk-premium](../../130-vol-risk-premium/) shows the raw premium in the "
            "options market is one of the most statistically solid facts in finance. The lesson "
            "here isn't that the premium is fake — it's that *packaging* it into a monthly-roll "
            "ETF, over one particular decade, didn't hand you anything beyond a smaller equity "
            "position.\n"
            "- **Sibling studies:** [62-premium-seller](../../62-premium-seller/) and "
            "[337-covered-call-etf](../../337-covered-call-etf/) (the covered-call side of the "
            "same family) and [354-the-wheel](../../354-the-wheel/) (both legs, model-priced) — "
            "all land on the same honest conclusion: selling options against the S&P 500 is "
            "short volatility in a costume, not a free upgrade to your Sharpe ratio.\n\n"
            "*Think a longer sample, a different strike, or a smarter roll schedule changes the "
            "answer? Show a positive, certifiable CAPM alpha on real, tradable prices — after "
            "the fund's own costs — and we'll take another look.*"
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
            "# The Put-Write-Premium — a quantitative teardown 🔬\n"
            "### CAPM alpha/beta on PUTW's real tape · a crash-conditional beta interaction · a "
            "block-bootstrapped Sharpe race · monthly capture and tail decomposition · a 20-seed "
            "synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — **systematic put-writing harvests the variance risk premium and beats "
            "buy-and-hold risk-adjusted** — has a genuine, well-documented mechanism (implied vol "
            "runs richer than realized vol on average) and a real academic anchor (the CBOE PUT "
            "index's long-run Sharpe advantage over 1986→2015). The job here is to test the "
            "*fund* that actually implements it, on its own live tape, and ask the only question "
            "that pays: is any of the outperformance beyond beta, and does it survive a crash?\n\n"
            "> ⚠️ **Data note.** PUTW / SPY / BIL daily auto-adjusted closes, "
            f"{R['start']} → {R['end']} ({R['n']:,} daily returns), yfinance, cached. Window is "
            "bound by PUTW's own inception — not the longer, untradeable CBOE PUT index. No "
            "survivorship (both funds are live, currently-listed, measured over their full "
            "window). Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | excess gap vs SPY **{R['gap_ann']:+.2f}%/yr** "
            f"(HAC *t* = {R['gap_t']:+.2f}); Sharpe **{R['sharpe_putw']:.2f} vs "
            f"{R['sharpe_spy']:.2f}**, bootstrap 95% CI **[{R['boot_lo']:+.2f}, {R['boot_hi']:+.2f}]**; "
            f"CAPM alpha **{R['alpha_ann']:+.2f}%/yr at *t* = {R['alpha_t']:+.2f}** |\n"
            f"| **Tradability** | `MIRAGE` | beta widens from **{R['cb_normal']:.2f}** (normal) "
            f"to **{R['cb_total']:.2f}** (crash days, *t* = {R['cb_extra_t']:+.2f}); worst month "
            f"**{R['worst_m_putw']:+.1f}%** beats SPY's own worst month ({R['worst_m_spy']:+.1f}%) |\n"
            f"| **Truncated beta?** | `CONFIRMED` | beta **{R['beta']:.3f}** (*t* = {R['beta_t']:+.2f}) "
            f"explains the average day; alpha statistically zero |\n\n"
            "> 💡 In plain words: the fund is a smaller, less stable slice of SPY's own market "
            "risk — not a distinct risk-adjusted edge."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{ex}_{putw,t} = r_{putw,t} - r_{bil,t}$ and $r^{ex}_{spy,t} = r_{spy,t} - "
            "r_{bil,t}$ be daily excess-of-cash returns. The claims:\n\n"
            "- **H₁ (premium exists).** $E[r^{ex}_{putw,t}] > 0$ — PUTW earns something over "
            "cash.\n"
            "- **H₂ (beats buy-and-hold, risk-adjusted).** "
            "$\\text{Sharpe}(r^{ex}_{putw}) \\ge \\text{Sharpe}(r^{ex}_{spy})$.\n"
            "- **H₃ (genuine alpha, not just beta).** In "
            "$r^{ex}_{putw,t} = \\alpha + \\beta \\, r^{ex}_{spy,t} + \\epsilon_t$, "
            "$\\alpha$ is significantly positive.\n"
            "- **H₄ (stable protection).** $\\beta$ does not widen materially on the market's "
            "worst days.\n\n"
            f"We find **H₁ marginal** (excess return {R['putw_excess_ann']:+.2f}%/yr, HAC "
            f"*t* = {R['putw_excess_t']:+.2f} — right at the desk's own bar, not comfortably "
            f"above it), **H₂ rejected** (Sharpe {R['sharpe_putw']:.2f} < {R['sharpe_spy']:.2f}, "
            f"bootstrap CI excludes zero on the losing side), **H₃ rejected** "
            f"(alpha *t* = {R['alpha_t']:+.2f}), **H₄ rejected** (crash-beta interaction "
            f"*t* = {R['cb_extra_t']:+.2f})."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Every excess-of-cash number here is measured against **BIL** (the T-bill ETF), the "
            "same convention as sibling study 655-ivy-portfolio, so no Sharpe race compares a "
            "raw return to an excess return. The CAPM regression is the honest version of "
            "\"is the premium real\" — a **Newey-West (HAC, 10 daily lags)** *t* on both the "
            "intercept (alpha) and the slope (beta); the crash-conditional beta adds an "
            "**interaction term** on an *ex-ante*, non-snooped threshold (SPY daily return "
            "≤ −3%) rather than a full-sample quantile that would risk fitting the tail it's "
            "meant to test. The Sharpe race between PUTW and SPY runs a **circular block "
            "bootstrap** (block = 21 trading days ≈ one option roll) on the *difference*, not a "
            "single point estimate."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** PUTW / SPY / BIL daily auto-adjusted closes, {R['start']} → {R['end']} "
            f"({R['n']:,} returns) — PUTW's own inception is the binding constraint, not the "
            "longer CBOE PUT index. As-of 2026-06-30 (last complete month).\n"
            "- **Headline.** Excess-of-cash HAC *t* for each leg + the paired excess-return gap "
            "+ Sharpe with a bootstrap CI on the difference.\n"
            "- **Alpha vs beta.** CAPM regression, HAC-robust, both coefficients.\n"
            "- **Crash conditioning.** An interaction regression: normal-day beta vs an extra "
            "crash-day slope on an ex-ante threshold.\n"
            "- **Cross-checks.** Named crash-window drawdowns, monthly up/down capture, daily "
            "skew and worst-day/worst-month comparisons.\n"
            "- **Execution.** Zero look-ahead by construction — realized returns of an "
            "already-listed fund, nothing timed.\n"
            "- **Control.** A synthetic Black-Scholes cash-secured-put engine with a tunable "
            "VRP knob; the alpha detector must not fire on a fairly-priced null across 20 "
            "seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline split — excess-of-cash return and the Sharpe race\n\n"
            "Both legs' excess-of-cash return (HAC *t*), the paired gap, and a circular block "
            "bootstrap on the Sharpe difference."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ex_p, ex_s = EX['PUTW'].to_numpy(), EX['SPY'].to_numpy()\n"
            "    rp = st.excess_return_stats(ex_p); rs = st.excess_return_stats(ex_s)\n"
            "    diff = st.hac_mean_t(ex_p - ex_s, lags=10)\n"
            "    sh_p, sh_s = st.sharpe_ann(ex_p), st.sharpe_ann(ex_s)\n"
            "    bs = st.bootstrap_sharpe_diff(ex_p, ex_s)\n"
            "    print(f\"PUTW-BIL {rp['ann_pct']:+.2f}%/yr (t={rp['t']:+.2f})   \"\n"
            "          f\"SPY-BIL {rs['ann_pct']:+.2f}%/yr (t={rs['t']:+.2f})\")\n"
            "    print(f\"gap {diff['mean']*252*100:+.2f}%/yr  t={diff['t']:+.2f}\")\n"
            "    print(f\"Sharpe {sh_p:.2f} vs {sh_s:.2f}   bootstrap CI [{bs['ci_lo']:+.2f}, {bs['ci_hi']:+.2f}]\")\n"
            "    lo, hi, win = bs['ci_lo'], bs['ci_hi'], bs['frac_putw_wins']*100\n"
            "else:\n"
            "    sh_p, sh_s, lo, hi, win = R['sharpe_putw'], R['sharpe_spy'], R['boot_lo'], R['boot_hi'], R['boot_win_pct']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['PUTW', 'SPY'], [sh_p, sh_s], color=[RED, GREY], width=.55)\n"
            "for i, v in enumerate([sh_p, sh_s]): ax.annotate(f'{v:.2f}', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('Sharpe (excess of cash)')\n"
            "ax.set_title(f'Sharpe gap {sh_p-sh_s:+.2f}, bootstrap CI [{lo:+.2f}, {hi:+.2f}] '\n"
            "             f'-- PUTW wins {win:.1f}% of resamples')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: PUTW earns {R['putw_excess_ann']:+.2f}%/yr over cash "
            f"(HAC *t* = {R['putw_excess_t']:+.2f} — right at, not above, the desk's *t* ≥ 2 "
            f"bar), SPY earns {R['spy_excess_ann']:+.2f}%/yr (*t* = {R['spy_excess_t']:+.2f}). "
            f"The gap ({R['gap_ann']:+.2f}%/yr, *t* = {R['gap_t']:+.2f}) and the Sharpe bootstrap "
            f"(CI [{R['boot_lo']:+.2f}, {R['boot_hi']:+.2f}], PUTW wins only "
            f"{R['boot_win_pct']:.1f}% of draws) both point the same way: on this fund's entire "
            "live tape, H₂ is rejected."
        ),
        md(
            "### 4b · Alpha vs beta — the CAPM split\n\n"
            "$r^{ex}_{putw,t} = \\alpha + \\beta \\, r^{ex}_{spy,t} + \\epsilon_t$, HAC (10 lags)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cb = st.capm_alpha_beta(EX['PUTW'].to_numpy(), EX['SPY'].to_numpy())\n"
            "    alpha, beta, ta, tb = cb['alpha_ann_pct'], cb['beta'], cb['t_alpha'], cb['t_beta']\n"
            "else:\n"
            "    alpha, beta, ta, tb = R['alpha_ann'], R['beta'], R['alpha_t'], R['beta_t']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar(['beta'], [beta], color=GREY, width=.5)\n"
            "a1.axhline(1.0, ls='--', c='k', lw=1)\n"
            "a1.set_title(f'beta = {beta:.3f}  (t = {tb:+.2f})'); a1.set_ylim(0, 1.1)\n"
            "a2.bar(['alpha (ann. %)'], [alpha], color=[RED if abs(ta) < 2 else GREEN], width=.5)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_title(f'alpha = {alpha:+.2f}%/yr  (t = {ta:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'alpha {alpha:+.2f}%/yr (t={ta:+.2f})   beta {beta:.3f} (t={tb:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: beta = **{R['beta']:.3f}** at *t* = {R['beta_t']:+.2f} — "
            "PUTW's average-day behavior is almost entirely explained by holding about 60% of "
            f"SPY's market exposure. Alpha = **{R['alpha_ann']:+.2f}%/yr** at "
            f"*t* = {R['alpha_t']:+.2f} — statistically zero, point estimate even negative. H₃ "
            "is rejected: there is no measurable premium beyond beta on this tape."
        ),
        md(
            "### 4c · Does the lower beta survive a crash?\n\n"
            "$r^{ex}_{putw,t} = \\alpha + \\beta_0 r^{ex}_{spy,t} + \\gamma D_t + "
            "\\delta (D_t \\cdot r^{ex}_{spy,t}) + \\epsilon_t$, "
            "$D_t = \\mathbf{1}\\{r_{spy,t} \\le -3\\%\\}$ — an ex-ante threshold, not a "
            "full-sample quantile."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cx = st.crash_beta_interaction(EX['PUTW'].to_numpy(), EX['SPY'].to_numpy(),\n"
            "                                    RET['SPY'].to_numpy(), threshold=data.CRASH_DAY_THRESHOLD)\n"
            "    bn, be, bt, ncrash = cx['beta_normal'], cx['crash_beta_extra'], cx['t_crash_beta_extra'], cx['n_crash_days']\n"
            "else:\n"
            "    bn, be, bt, ncrash = R['cb_normal'], R['cb_extra'], R['cb_extra_t'], R['n_crash_days']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['normal-day beta', 'crash-day beta\\n(normal + extra)'], [bn, bn + be],\n"
            "       color=[GREY, RED], width=.55)\n"
            "for i, v in enumerate([bn, bn + be]): ax.annotate(f'{v:.3f}', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(1.0, ls='--', c='k', lw=1, label='full equity beta')\n"
            "ax.set_ylabel('beta to SPY'); ax.legend()\n"
            "ax.set_title(f'Extra crash-day beta = {be:+.3f} (t = {bt:+.2f}), n={ncrash} crash days')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: normal-day beta = **{R['cb_normal']:.3f}** "
            f"(*t* = {R['cb_normal_t']:+.2f}); on the **{R['n_crash_days']}** days SPY fell 3%+, "
            f"beta gains an extra **+{R['cb_extra']:.3f}** (*t* = {R['cb_extra_t']:+.2f}, "
            f"significant), pushing the crash-day beta to **≈{R['cb_total']:.3f}** — essentially "
            "full equity exposure. H₄ is rejected: the 'lower risk' is not a stable property, "
            "it is a normal-times property that fades exactly when a diversifier is supposed to "
            "earn its keep."
        ),
        md(
            "### 4d · Named crash windows, monthly capture, and the tail signature\n\n"
            "Cross-checks: drawdowns in the two named vol shocks, monthly up/down capture "
            "(Morningstar-style mean-return ratio), and the worst single day/month each way."
        ),
        code(
            "if HAVE_REAL:\n"
            "    dd = {name: (st.window_drawdown(RET['PUTW'], s, e)*100, st.window_drawdown(RET['SPY'], s, e)*100)\n"
            "          for name, (s, e) in data.CRASH_WINDOWS.items()}\n"
            "    cap = st.monthly_capture(PX)\n"
            "    up_c, dn_c = cap['up_capture']*100, cap['dn_capture']*100\n"
            "    wm_p, wm_s = cap['worst_month_putw']*100, cap['worst_month_spy']*100\n"
            "else:\n"
            "    dd = {'volmageddon_2018': (R['volm_putw'], R['volm_spy']),\n"
            "          'covid_2020': (R['covid_putw'], R['covid_spy'])}\n"
            "    up_c, dn_c = R['cap_up'], R['cap_dn']\n"
            "    wm_p, wm_s = R['worst_m_putw'], R['worst_m_spy']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "names = list(dd.keys())\n"
            "p_vals = [dd[n][0] for n in names]; s_vals = [dd[n][1] for n in names]\n"
            "x = np.arange(len(names)); w = 0.35\n"
            "a1.bar(x - w/2, p_vals, w, color=RED, label='PUTW')\n"
            "a1.bar(x + w/2, s_vals, w, color=GREY, label='SPY')\n"
            "a1.set_xticks(x); a1.set_xticklabels(['Volmageddon\\n2018', 'COVID\\n2020'])\n"
            "a1.set_ylabel('drawdown (%)'); a1.legend(); a1.set_title('Named crash windows')\n"
            "a2.bar(['up capture', 'down capture'], [up_c, dn_c], color=[AMBER, RED], width=.55)\n"
            "a2.axhline(100, ls='--', c='k', lw=1)\n"
            "for i, v in enumerate([up_c, dn_c]): a2.annotate(f'{v:.1f}%', (i, v), ha='center', va='bottom')\n"
            "a2.set_ylabel('% of SPY\\'s mean monthly move captured')\n"
            "a2.set_title('Keeps 59% of up months, gives back 66% of down months')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'worst month: PUTW {wm_p:+.1f}%  vs SPY {wm_s:+.1f}%')"
        ),
        md(
            f"> 💡 In plain words: PUTW looks protective in both named windows "
            f"(**{R['volm_putw']:+.1f}%** vs SPY {R['volm_spy']:+.1f}% in 2018; "
            f"**{R['covid_putw']:+.1f}%** vs {R['covid_spy']:+.1f}% in 2020) — but that is "
            "exactly what a beta-0.6 asset does in any broad drawdown, and does not contradict "
            f"4c. The capture asymmetry ({R['cap_up']:.1f}% up / {R['cap_dn']:.1f}% down) is the "
            "textbook short-vol shape, and the sharpest single number: despite {:.0f}% less "
            "annualized volatility than SPY, PUTW's single worst month "
            f"(**{R['worst_m_putw']:+.1f}%**, {R['worst_m_date']}) was *worse* than SPY's own "
            f"worst month (**{R['worst_m_spy']:+.1f}%**). A lower-vol asset with a fatter left "
            "tail than its own benchmark is the fingerprint of a truncated-upside payoff, not a "
            "genuine risk reducer.".format(100 * (1 - R["vol_putw"] / R["vol_spy"]))
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Deterministic monthly cash-secured-ATM-put engine (Black-Scholes premium, "
            "`data.bs_atm_put`), options priced at `realized_vol × (1 + harvest)`. "
            "`harvest = 0` embeds **no** volatility risk premium; the CAPM-alpha detector must "
            "not manufacture significance there. Null checked over **20 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    w = data.synthetic_world(harvest=0.0, seed=658 + s_)\n"
            "    null_ts.append(st.synthetic_detect(w)['t_alpha'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "w = data.synthetic_world(harvest=0.25, seed=658)\n"
            "sy = st.synthetic_detect(w)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (harvest=0), 20 seeds')\n"
            "ax.scatter([1], [sy['t_alpha']], color=RED, s=90, zorder=5,\n"
            "           label='planted harvest = +0.25')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('t (CAPM alpha)')\n"
            "ax.set_title('Control: the null rarely fires; a planted premium lights up clean')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {sy[\"t_alpha\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"*t* = {R['syn_null_t_mean']:+.2f} (sd {R['syn_null_t_sd']:.2f}) and fires in only "
            f"**{R['syn_null_fire']}/20** seeds — about the nominal 5% false-positive rate, "
            "correctly calibrated, not over- or under-powered. A planted premium "
            f"(alpha {R['syn_planted_alpha']:+.2f}%/yr) reads *t* = {R['syn_planted_t']:+.2f} — "
            "clean detection. The machinery is unbiased; the real tape's insignificant alpha "
            "(*t* = −0.95) is the genuine article, not a blind spot in the test. *(A "
            "faithful-engine / power check only — never cited in support of the real-tape "
            "stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — excess-of-cash return is marginal at best "
            f"({R['putw_excess_ann']:+.2f}%/yr, *t* = {R['putw_excess_t']:+.2f}); the actual "
            f"claim (beats buy-and-hold risk-adjusted) is rejected decisively: gap vs SPY "
            f"**{R['gap_ann']:+.2f}%/yr** at *t* = **{R['gap_t']:+.2f}**, Sharpe "
            f"**{R['sharpe_putw']:.2f} vs {R['sharpe_spy']:.2f}** with a bootstrap CI "
            f"[{R['boot_lo']:+.2f}, {R['boot_hi']:+.2f}], CAPM alpha statistically zero "
            f"(*t* = {R['alpha_t']:+.2f}).\n"
            f"- **Tradability `MIRAGE`** — beta widens from **{R['cb_normal']:.2f}** to "
            f"**{R['cb_total']:.2f}** (*t* = {R['cb_extra_t']:+.2f}) on the worst days, and the "
            f"worst single month (**{R['worst_m_putw']:+.1f}%**) beat SPY's own worst month "
            f"(**{R['worst_m_spy']:+.1f}%**) despite {R['vol_putw']:.1f}% vs {R['vol_spy']:.1f}% "
            "annualized vol; no residual edge survives the fund's own cost structure because "
            "there was never one to begin with.\n"
            f"- **Truncated beta? `CONFIRMED`** — beta {R['beta']:.3f} (*t* = {R['beta_t']:+.2f}) "
            "explains the average day, alpha is statistically zero, and the beta itself is "
            "unstable — converging toward full equity exposure precisely on the days a "
            "diversifier is supposed to prove its worth."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The premium is real; the packaging is the problem.** "
            "[130-vol-risk-premium](../../130-vol-risk-premium/) shows implied vol systematically "
            "exceeds realized vol (HAC *t* = +22.9) — this study shows a real monthly-roll ETF "
            "did not translate that into alpha beyond beta over its own decade. The gap between "
            "\"the premium exists\" and \"a fund banks it net of beta and costs\" is the whole "
            "story.\n"
            "- **The crash-conditional-beta test is the general lesson.** Any short-vol wrapper "
            "— covered calls, the Wheel, tail-hedge sellers — should be tested for whether its "
            "'lower risk' is stable or regime-dependent; a single full-sample beta hides exactly "
            "the failure mode that matters.\n"
            "- **Dedup map:** [62-premium-seller](../../62-premium-seller/) and "
            "[337-covered-call-etf](../../337-covered-call-etf/) (the call side), "
            "[354-the-wheel](../../354-the-wheel/) (both legs, model-priced), "
            "[130-vol-risk-premium](../../130-vol-risk-premium/) (the raw premium, no fund), "
            "[617-crash-insurance-cost](../../617-crash-insurance-cost/) (the buyer's mirror "
            "image) — none of them run the CAPM-alpha-vs-beta and crash-interaction test this "
            "study runs on PUTW's own live tape.\n\n"
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
