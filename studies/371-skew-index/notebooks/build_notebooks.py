"""Generate the two narrative notebooks for Study 371 (CBOE SKEW Index).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SKEW/VIX/SPY tape
under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SKEW/VIX/SPY,
# 1993-01-29 -> 2026-06-18, 8,330 joint days, 33.4 years).
R = dict(
    start="1993-01-29", end="2026-06-18", days=8330, years=33.4,
    skew_mean=123.7, vix_mean=19.5, corr=-0.19,
    # VIX-controlled HAC regression: (months, n, beta_skew%, t_skew, beta_vix%, t_vix)
    reg1=(1, 8308, 0.057, 0.35, 0.427, 1.54),
    reg3=(3, 8266, 0.247, 0.49, 0.890, 1.13),
    # decile bucket: (months, n_hi, hi_mean%, rest_mean%, hi_win%, rest_win%, welch_t, placebo_p)
    dec1=(1, 831, 1.02, 0.97, 69, 65, 0.37, 0.619),
    dec3=(3, 827, 2.37, 2.94, 71, 72, -2.33, 0.026),
    # block bootstrap: (months, obs_diff%, block_se%, block_t, episodes)
    blk1=(1, 0.05, 0.38, 0.13, 23),
    blk3=(3, -0.57, 1.13, -0.50, 13),
    # tail warning: (dd%, hi_rate%, base_rate%, placebo_p)
    tail10=(-10, 1.7, 4.4, 1.000),
    tail5=(-5, 13.7, 17.3, 0.998),
    # fade sleeve
    sleeve_cagr=9.04, sleeve_sharpe=0.57, bh_cagr=10.94, bh_sharpe=0.65, pct_cash=10,
    # synthetic control: (edge, n, beta_skew%, t_skew)
    syn=[(0.0, 7978, 0.109, 0.42), (0.0005, 7978, -0.764, -2.85)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Warns_before_crashes%3F: Busted](https://img.shields.io/badge/Warns_before_crashes%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from skew_index import data, strategy as st

HAVE_REAL = data.have_real()
F = data.load_real() if HAVE_REAL else None
print("real SKEW/VIX/SPY cache present:", HAVE_REAL,
      "| joint days:", (0 if F is None else len(F)))
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell can
# quote it whether or not the cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The \"black-swan index\" — does SKEW warn you before crashes? 🪝\n"
            "### CBOE's tail-risk gauge, the one the VIX supposedly can't give you, in plain English\n\n"
            + BADGES +
            "There's a number traders whisper about when they want to sound scary: the **CBOE SKEW "
            "index**. The pitch is irresistible — the VIX only measures *ordinary* fear, but SKEW reads "
            "how much people are paying for **crash insurance**: deep out-of-the-money puts that only pay "
            "off if the market falls off a cliff. So when SKEW spikes, the story goes, **the smart money "
            "is bracing for a black swan** — a warning the VIX is blind to.\n\n"
            "A gauge that lights up *before* crashes would be priceless. This notebook asks, with 33 "
            "years of data, the only question that matters: **does a high SKEW actually precede trouble — "
            "and does it tell you anything the VIX doesn't already?**\n\n"
            "> 📓 **Plain-language layer.** Want the regressions, the HAC *t*-stats and the block "
            "bootstrap? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After a high SKEW, does the market fall more often? | **No — it falls *less*.** Big "
            "drawdowns are **rarer** after a high SKEW (1.7% vs 4.4% normally). The \"warning\" fires "
            "before calm. |\n"
            "| Does SKEW tell you anything VIX doesn't? | **No.** Once you account for the VIX, SKEW's "
            "own predictive power is statistically **zero** (and points the *wrong* way). |\n"
            "| Can you trade it — de-risk when SKEW is high? | **It loses money.** Sitting out the "
            "highest-SKEW days **lowers** both your return and your risk-adjusted return. |\n"
            "| Then why the scary reputation? | **It prices the crash everyone's *worried* about** — "
            "which is exactly the crash that usually doesn't come. Fear priced ≠ fear realised. |\n\n"
            "> SKEW is a real, fascinating measurement of how much the option market *fears* a tail. "
            "It just isn't a forecast of one."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The VIX measures everyday volatility — fear of a normal-sized move. SKEW measures the "
            "price of the **tail**: how richly the market pays for deep out-of-the-money puts that only "
            "matter in a crash. So a high SKEW means insiders are loading up on crash protection — heed "
            "the warning.\"*\n\n"
            "CBOE built SKEW from out-of-the-money S&P 500 option prices and scaled it so **100 = a "
            "normal, symmetric distribution**; readings of 130–145 mean the option market is pricing a "
            "*fatter left tail*. The seduction is the framing: a dedicated, real-time **black-swan "
            "detector**. We'll line it up next to the VIX and the actual future and see whether the "
            "detector detects anything."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two very different claims hide inside \"SKEW warns you.\" (1) *Does a high SKEW precede "
            "weaker markets / more crashes?* And (2) *does it do so beyond what the VIX already tells "
            "you?* The second is the real test: SKEW and VIX are both squeezed from the same option "
            "prices, so if SKEW only repeats VIX's message it's noise dressed as insight. And even a "
            "genuine warning is worthless if acting on it — going to cash when SKEW is high — **costs** "
            "you return rather than saving you from drawdowns."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{R['days']:,}** days of daily SKEW, VIX and SPY closes "
            f"({R['start']} → {R['end']}, **{R['years']:.0f} years**) and do three honest things:\n\n"
            "1. **Look forward.** After a high SKEW (top 10% of days), what did SPY do over the next "
            "**1 and 3 months** — and how often did a real **crash** (a −5% / −10% drawdown) follow, "
            "versus a normal day?\n"
            "2. **Subtract the VIX.** Put SKEW *and* VIX in the same regression and ask whether SKEW "
            "adds anything once VIX is accounted for.\n"
            "3. **Count the real evidence.** High-SKEW days come in a handful of *clusters*, not "
            "hundreds of independent bets — so we count the **distinct episodes** and bootstrap honestly."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, here's SKEW next to the VIX.** Notice they don't move together — SKEW often climbs "
            "when markets are *calm* (low VIX). That's the first clue: SKEW is pricing a fear that isn't "
            "showing up in realised volatility."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fig, ax = plt.subplots(2, 1, figsize=(9.4, 5.2), sharex=True)\n"
            "    ax[0].plot(F.index, F['skew'], c=RED, lw=.8); ax[0].set_ylabel('SKEW')\n"
            "    ax[0].axhline(F['skew'].quantile(.90), ls='--', c=GREY, label='top-decile line')\n"
            "    ax[0].set_title('CBOE SKEW (top) vs VIX (bottom) — they often move APART'); ax[0].legend(loc='upper left')\n"
            "    ax[1].plot(F.index, F['vix'], c=AMBER, lw=.8); ax[1].set_ylabel('VIX'); ax[1].set_xlabel('year')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('corr(SKEW, VIX) =', round(F['skew'].corr(F['vix']), 2), '-> barely related')\n"
            "else:\n"
            "    print('no cache — see docs/results.md: corr(SKEW,VIX)=', R['corr'])"
        ),
        md(
            f"The correlation is just **{R['corr']:+.2f}** — SKEW and VIX are nearly *unrelated*, even "
            "mildly opposite. So SKEW isn't simply \"VIX for the tail.\" Good — that means it *could*, in "
            "principle, carry its own warning. Does it?"
        ),
        md(
            "**Does a crash follow a high SKEW?** Here's the punchline chart. After a high-SKEW day, how "
            "often does a real drawdown hit over the next month — versus a random day?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    t10 = st.tail_warning(F, 1, dd=-0.10); t5 = st.tail_warning(F, 1, dd=-0.05)\n"
            "    hi = [t5['hi_tail_rate']*100, t10['hi_tail_rate']*100]\n"
            "    base = [t5['base_tail_rate']*100, t10['base_tail_rate']*100]\n"
            "else:\n"
            "    hi = [R['tail5'][1], R['tail10'][1]]; base = [R['tail5'][2], R['tail10'][2]]\n"
            "labels = ['drawdown\\nworse than -5%', 'drawdown\\nworse than -10%']\n"
            "x = np.arange(2)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.bar(x-.2, hi, .4, color=RED, label='AFTER a high SKEW')\n"
            "ax.bar(x+.2, base, .4, color=GREY, label='any normal day')\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('% of months with this drawdown')\n"
            "ax.set_title('The crash gauge fires before CALM: fewer tails follow a high SKEW')\n"
            "for i,(h,b) in enumerate(zip(hi,base)):\n"
            "    ax.annotate(f'{h:.1f}%',(i-.2,h),ha='center',va='bottom')\n"
            "    ax.annotate(f'{b:.1f}%',(i+.2,b),ha='center',va='bottom')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'-10% drawdowns: {hi[1]:.1f}% after high SKEW vs {base[1]:.1f}% normally -> LOWER, not higher')"
        ),
        md(
            f"There it is. A −10% month happens **{R['tail10'][1]:.1f}%** of the time after a high SKEW "
            f"— versus **{R['tail10'][2]:.1f}%** on a normal day. The supposed crash warning is followed "
            "by **fewer** crashes. SKEW prices the tail the market is *afraid* of, and the market is "
            "usually afraid at the wrong times."
        ),
        md(
            "**But wait — isn't a high-SKEW day followed by weaker returns?** A little, at 3 months. The "
            "trap is that those high-SKEW days aren't hundreds of independent bets — they bunch into a "
            "*handful of episodes*. Count the real, distinct episodes and the \"signal\" evaporates."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b1 = st.block_bootstrap_decile(F, 1); b3 = st.block_bootstrap_decile(F, 3)\n"
            "    naive_t = [st.decile_summary(F,1)['t'], st.decile_summary(F,3)['t']]\n"
            "    honest_t = [b1['block_t'], b3['block_t']]; eps = [b1['n_episodes'], b3['n_episodes']]\n"
            "else:\n"
            "    naive_t = [R['dec1'][6], R['dec3'][6]]; honest_t = [R['blk1'][3], R['blk3'][3]]\n"
            "    eps = [R['blk1'][4], R['blk3'][4]]\n"
            "x = np.arange(2)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(x-.2, [abs(t) for t in naive_t], .4, color=GREY, label='naive t (pretends every day is independent)')\n"
            "ax.bar(x+.2, [abs(t) for t in honest_t], .4, color=RED, label='honest t (counts distinct episodes)')\n"
            "ax.axhline(2, ls='--', c='k', label='significance bar (|t|=2)')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['1 month','3 months']); ax.set_ylabel('|t-statistic|')\n"
            "ax.set_title('The \"significant\" bucket is a counting illusion'); ax.legend()\n"
            "for i,(n,h,e) in enumerate(zip(naive_t,honest_t,eps)):\n"
            "    ax.annotate(f'~{e} episodes',(i+.2,abs(h)),ha='center',va='bottom',fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'3-month: naive |t|={abs(naive_t[1]):.2f} -> honest |t|={abs(honest_t[1]):.2f} on only ~{eps[1]} real episodes')"
        ),
        md(
            f"The grey bars look impressive because they treat **800+ overlapping days** as 800 separate "
            f"experiments. But there are only about **{R['blk3'][4]}** genuinely distinct high-SKEW "
            f"episodes in 33 years. Count them honestly and the *t* collapses to "
            f"**{R['blk3'][3]:+.1f}** — pure noise. The \"weaker 3-month returns\" were a counting "
            "illusion, not a forecast."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Once you control for the VIX, SKEW's own predictive power is "
            "statistically zero — and the lone \"significant\" bucket is a counting artifact (~13 real "
            "episodes).\n"
            "- **Tradability — Mirage.** De-risking on high SKEW **loses** to buy-and-hold "
            f"(**{R['sleeve_cagr']:.1f}%** vs **{R['bh_cagr']:.1f}%** a year), because high-SKEW days "
            "aren't followed by losses.\n"
            "- **Warns before crashes? — Busted.** Big drawdowns are *rarer* after a high SKEW "
            f"(**{R['tail10'][1]:.1f}%** vs **{R['tail10'][2]:.1f}%**). The black-swan gauge fires "
            "before calm."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the de-risk trade\n\n"
            "Forget significance and just *do* the obvious trade: go to cash whenever SKEW is in its top "
            "decile (the \"warning\"), hold SPY otherwise. If SKEW saw crashes coming, this should beat "
            "buy-and-hold by dodging the worst months. Here's what actually happens."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fs = st.fade_sleeve(F)\n"
            "    s_cagr, s_sh, b_cagr, b_sh = fs['sleeve_cagr']*100, fs['sleeve_sharpe'], fs['bh_cagr']*100, fs['bh_sharpe']\n"
            "else:\n"
            "    s_cagr, s_sh, b_cagr, b_sh = R['sleeve_cagr'], R['sleeve_sharpe'], R['bh_cagr'], R['bh_sharpe']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.2))\n"
            "a1.bar(['fade high\\nSKEW', 'buy &\\nhold SPY'], [s_cagr, b_cagr], color=[RED, GREEN], width=.55)\n"
            "a1.set_ylabel('annual return (%)'); a1.set_title('De-risking on SKEW LOWERS return')\n"
            "for i,v in enumerate([s_cagr, b_cagr]): a1.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "a2.bar(['fade high\\nSKEW', 'buy &\\nhold SPY'], [s_sh, b_sh], color=[RED, GREEN], width=.55)\n"
            "a2.set_ylabel('Sharpe ratio'); a2.set_title('...and LOWERS risk-adjusted return too')\n"
            "for i,v in enumerate([s_sh, b_sh]): a2.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'fade sleeve: {s_cagr:.1f}% CAGR / Sharpe {s_sh:.2f}  vs  buy-hold {b_cagr:.1f}% / {b_sh:.2f}')"
        ),
        md(
            f"The \"warning\" trade **underperforms** on both counts — **{R['sleeve_cagr']:.1f}%** vs "
            f"**{R['bh_cagr']:.1f}%** a year, Sharpe **{R['sleeve_sharpe']:.2f}** vs "
            f"**{R['bh_sharpe']:.2f}**. Every day you spent in cash \"for safety\" was, on average, a "
            "perfectly good day to be invested. The crash gauge cost you money by crying wolf."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The honest reading of SKEW.** It's a genuine, well-built measure of how much the option "
            "market is *paying* for tail protection. That's a sentiment / risk-appetite gauge — not a "
            "forecast. The crash everyone insures against is usually the one that doesn't come.\n"
            "- **Build your own test.** Swap the −10% drawdown threshold, the decile cutoff, or the "
            "horizon — the result is sticky: high SKEW does not precede more tails, and never beats VIX "
            "in the regression.\n"
            "- **The deeper lesson.** *Implied* tail risk (what the market fears) and *realised* tail "
            "risk (what happens) are different objects. SKEW measures the first beautifully and the "
            "second not at all.\n\n"
            "*Think SKEW beats VIX as a crash warning? Put both in one regression, show SKEW's "
            "coefficient clearing |t| = 2 with HAC errors, and show the de-risk trade beating "
            "buy-and-hold — then we'll talk.*"
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
            "# CBOE SKEW — a quantitative teardown 🔬\n"
            "### Forward returns on SKEW *controlling for VIX* (HAC) · the bucket *t* vs its block-bootstrap "
            "correction · a tail-warning test vs base rate · a costed fade sleeve · a planted-edge control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The believers' "
            "claim about SKEW is **incremental**: it warns when VIX is calm. So the decisive object is "
            "**SKEW's coefficient in a forward-return regression that already controls for VIX**, read with "
            "**HAC (Newey-West)** standard errors — overlapping forward-return windows are heavily "
            "autocorrelated, so a naive *t* is a pseudo-replication trap.\n\n"
            "> ⚠️ **Data note.** Real data: yfinance daily closes for `^SKEW`, `^VIX`, `SPY`, joint window "
            f"{R['start']}→{R['end']} ({R['days']:,} days). SPY is **price-only** (no dividends; labelled on "
            "the Tradability axis). Offline core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | VIX-controlled HAC **t(SKEW) = +{R['reg1'][3]:.2f}** (1m) / "
            f"**+{R['reg3'][3]:.2f}** (3m) — far below 2, *positive*-signed. The 3m bucket Welch "
            f"**t = {R['dec3'][6]:.2f}** is pseudo-replication: ~{R['blk3'][4]} episodes, block-bootstrap "
            f"**t = {R['blk3'][3]:.2f}**. |\n"
            f"| **Tradability** | `MIRAGE` | Fade-high-SKEW sleeve **{R['sleeve_cagr']:.1f}%** CAGR / "
            f"Sharpe **{R['sleeve_sharpe']:.2f}** vs buy-hold **{R['bh_cagr']:.1f}%** / "
            f"**{R['bh_sharpe']:.2f}** — strictly worse. |\n"
            f"| **Warns before crashes?** | `BUSTED` | −10% forward drawdown rate **{R['tail10'][1]:.1f}%** "
            f"after high SKEW vs **{R['tail10'][2]:.1f}%** base (placebo p ≈ {R['tail10'][3]:.2f}). The tail "
            "gauge precedes *fewer* tails. |\n\n"
            "> 💡 In plain words: SKEW and VIX are both risk-neutral moments of the same option surface; "
            "once VIX is in the regression SKEW adds nothing, and the implied tail it prices is the tail "
            "the market *fears*, which is empirically the tail that mostly doesn't arrive."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $S_t$ = SKEW, $V_t$ = VIX, and $r_{t\\to t+H}$ the forward $H$-month SPY return entered "
            "one day after $t$. The hypotheses:\n\n"
            "- **H₁ (incremental signal).** In $r_{t\\to t+H} = \\alpha + \\beta_S \\tilde S_t + "
            "\\beta_V \\tilde V_t + \\varepsilon$ (predictors standardised), $\\beta_S < 0$ with a robust "
            "$|t| \\ge 2$ — SKEW predicts weaker forward returns **beyond VIX**.\n"
            "- **H₂ (tail warning).** $\\Pr[\\text{drawdown} < d \\mid S_t \\text{ high}] > "
            "\\Pr[\\text{drawdown} < d]$ — a high SKEW raises the forward tail-event probability.\n"
            "- **H₃ (deployable).** A rule that de-risks on high SKEW beats buy-and-hold net of costs.\n\n"
            "We find **H₁ rejected** ($t_S \\approx +0.4$, wrong sign), **H₂ rejected** (the tail rate "
            "*falls*), **H₃ rejected** (the sleeve loses). The lone artifact — a 3m bucket Welch "
            f"$t={R['dec3'][6]:.2f}$ — dies under a block bootstrap that respects the ~{R['blk3'][4]} "
            "distinct episodes."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The crux is **collinearity vs orthogonality**. SKEW and VIX are both extracted from the OTM "
            "S&P 500 option surface, so the only economically interesting part of SKEW is its component "
            "**orthogonal to VIX**. The right instrument is therefore the multivariate slope:\n\n"
            "$$\\hat\\beta_S = \\frac{\\mathrm{Cov}(r,\\ \\tilde S \\mid \\tilde V)}"
            "{\\mathrm{Var}(\\tilde S \\mid \\tilde V)},\\qquad t_S = \\hat\\beta_S / "
            "\\mathrm{se}_{\\mathrm{HAC}}(\\hat\\beta_S).$$\n\n"
            "Overlapping daily forward windows make $\\varepsilon$ strongly autocorrelated, so OLS "
            "standard errors are badly understated; **Newey-West** (lags = horizon) is mandatory. A "
            "**bucket** comparison (top-decile SKEW vs rest) is even more dangerous: its naive Welch *t* "
            "treats 800+ overlapping days as independent, while the *effective* sample is the number of "
            "**distinct high-SKEW episodes** — which a **block bootstrap** exposes."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** Daily `^SKEW`, `^VIX`, `SPY` closes, joint window {R['start']}→{R['end']} "
            f"({R['days']:,} days, {R['years']:.0f}y). SKEW mean **{R['skew_mean']}**, VIX mean "
            f"**{R['vix_mean']}**, corr **{R['corr']:+.2f}**.\n"
            "- **Forward returns.** Enter the close **1 day after** the signal (no look-ahead); horizons "
            "$H \\in \\{1, 3\\}$ months.\n"
            "- **Decisive test (Signal).** OLS of forward return on standardised SKEW *and* VIX; "
            "**HAC (Newey-West)** *t* with lags = horizon. Read $t_S$.\n"
            "- **Bucket + correction.** Top-decile SKEW vs rest (naive Welch *t* + placebo), then a "
            "**circular block bootstrap** (block = horizon) and an **episode count** for the honest *t*.\n"
            "- **Tail warning (myth axis).** Forward 1-month worst-drawdown ≤ {−5%, −10%}; conditional "
            "tail rate vs base rate, placebo *p*.\n"
            "- **Tradability.** A long/flat sleeve to cash while SKEW is top-decile (1-day lag, 2-bps "
            "one-way × turnover) vs buy-and-hold SPY, on the same tape.\n"
            "- **Positive control.** A deterministic (SKEW, VIX, SPY) panel with a **planted-edge knob** "
            "in the VIX-orthogonal part of SKEW: $t_S$ must stay quiet at edge 0 and light up when on."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The decisive regression — SKEW controlling for VIX\n\n"
            "Forward-return slope on standardised SKEW *and* VIX, with HAC (Newey-West) *t*-stats. "
            "$t_S$ is SKEW's *incremental* coefficient."
        ),
        code(
            "hs = [1, 3]\n"
            "if HAVE_REAL:\n"
            "    ts_skew = [st.regress_forward(F,m)['t_skew'] for m in hs]\n"
            "    ts_vix  = [st.regress_forward(F,m)['t_vix']  for m in hs]\n"
            "else:\n"
            "    ts_skew = [R['reg1'][3], R['reg3'][3]]; ts_vix = [R['reg1'][5], R['reg3'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(x-.2, ts_skew, .4, color=RED, label='t(SKEW) — controlling for VIX')\n"
            "ax.bar(x+.2, ts_vix, .4, color=AMBER, label='t(VIX)')\n"
            "ax.axhline(2, ls='--', c='k', label='|t|=2'); ax.axhline(-2, ls='--', c='k')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m}-month' for m in hs]); ax.set_ylabel('HAC t-statistic')\n"
            "ax.set_ylim(-2.4, 2.4); ax.set_title('SKEW adds nothing beyond VIX (and points the wrong way)')\n"
            "for i,t in enumerate(ts_skew): ax.annotate(f'{t:+.2f}',(i-.2,t),ha='center',va='bottom')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('t(SKEW | VIX):', {f'{m}m': round(t,2) for m,t in zip(hs,ts_skew)})"
        ),
        md(
            f"> 💡 In plain words: SKEW's own *t* is **+{R['reg1'][3]:.2f}** (1m) / **+{R['reg3'][3]:.2f}** "
            "(3m) — not just insignificant but the *wrong sign* for a crash warning (higher SKEW → "
            "marginally *higher* forward returns). Even VIX barely registers at these horizons. **H₁ "
            "rejected:** there is no information in SKEW orthogonal to VIX."
        ),
        md(
            "### 4b · The bucket *t* and why it's a mirage\n\n"
            "The top-decile-SKEW bucket shows a 3-month Welch *t* near −2.3 — tempting. But those days "
            "cluster into ~a dozen episodes. The block bootstrap (block = horizon), respecting the "
            "autocorrelation, gives the honest *t*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    naive = [st.decile_summary(F,m)['t'] for m in hs]\n"
            "    bb = [st.block_bootstrap_decile(F,m) for m in hs]\n"
            "    honest = [b['block_t'] for b in bb]; eps = [b['n_episodes'] for b in bb]\n"
            "else:\n"
            "    naive = [R['dec1'][6], R['dec3'][6]]; honest = [R['blk1'][3], R['blk3'][3]]; eps = [R['blk1'][4], R['blk3'][4]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(x-.2, naive, .4, color=GREY, label='naive Welch t (overlapping days as independent)')\n"
            "ax.bar(x+.2, honest, .4, color=RED, label='block-bootstrap t (episode-aware)')\n"
            "ax.axhline(-2, ls='--', c='k', label='|t|=2'); ax.axhline(2, ls='--', c='k'); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m}-month' for m in hs]); ax.set_ylabel('t-statistic (hi - rest)')\n"
            "ax.set_title('A pseudo-replication illusion: -2.3 -> -0.5 once episodes are counted')\n"
            "for i,(h,e) in enumerate(zip(honest,eps)): ax.annotate(f'~{e} episodes',(i+.2,h),ha='center',va='top',fontsize=9)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('3m: naive t =', round(naive[1],2), '-> block-bootstrap t =', round(honest[1],2), f'on ~{eps[1]} episodes')"
        ),
        md(
            f"> 💡 In plain words: the naive 3m *t* of **{R['dec3'][6]:.2f}** assumes "
            f"**{R['dec3'][1]}** independent observations; there are really ~**{R['blk3'][4]}** distinct "
            f"high-SKEW episodes, and the episode-aware *t* is **{R['blk3'][3]:.2f}**. The \"significant\" "
            "bucket was an artifact of counting overlapping windows as independent — the single most "
            "common way a volatility-folklore signal fakes significance."
        ),
        md(
            "### 4c · The tail-warning claim — the myth axis\n\n"
            "SKEW's entire reputation is *tail* insurance. So directly: after a high SKEW, is a forward "
            "drawdown more likely than usual? Conditional tail rate vs base rate, with a placebo null."
        ),
        code(
            "if HAVE_REAL:\n"
            "    t5 = st.tail_warning(F,1,dd=-0.05); t10 = st.tail_warning(F,1,dd=-0.10)\n"
            "    hi = [t5['hi_tail_rate']*100, t10['hi_tail_rate']*100]\n"
            "    base = [t5['base_tail_rate']*100, t10['base_tail_rate']*100]\n"
            "    ps = [t5['p_placebo'], t10['p_placebo']]\n"
            "else:\n"
            "    hi = [R['tail5'][1], R['tail10'][1]]; base = [R['tail5'][2], R['tail10'][2]]; ps = [R['tail5'][3], R['tail10'][3]]\n"
            "x = np.arange(2)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(x-.2, hi, .4, color=RED, label='after high SKEW')\n"
            "ax.bar(x+.2, base, .4, color=GREY, label='unconditional base rate')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['1m drawdown < -5%','1m drawdown < -10%'])\n"
            "ax.set_ylabel('forward tail-event rate (%)')\n"
            "ax.set_title('Conditional tail rate is BELOW base rate (placebo p ~ 1.0)')\n"
            "for i,(h,b) in enumerate(zip(hi,base)):\n"
            "    ax.annotate(f'{h:.1f}%',(i-.2,h),ha='center',va='bottom')\n"
            "    ax.annotate(f'{b:.1f}%',(i+.2,b),ha='center',va='bottom')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('placebo p (P[random draw >= conditional rate]):', {'-5%':round(ps[0],2), '-10%':round(ps[1],2)})"
        ),
        md(
            f"> 💡 In plain words: the conditional −10% rate is **{R['tail10'][1]:.1f}%** vs a base "
            f"**{R['tail10'][2]:.1f}%**; the placebo *p* ≈ **{R['tail10'][3]:.2f}** says a random draw "
            "essentially *always* shows more tail risk than the high-SKEW set. **H₂ rejected, with the "
            "sign inverted:** high SKEW marks priced-but-unrealised fear. The gauge fires before calm."
        ),
        md(
            "### 4d · Faithful-engine & planted-edge control — we know the truth here\n\n"
            "A deterministic (SKEW, VIX, SPY) panel with corr(SKEW, VIX) ≈ 0.70 and a knob that plants a "
            "**VIX-orthogonal** predictive edge in SKEW. With the knob **off** the VIX-controlled HAC *t* "
            "must stay below 2 (no false positive); with a real edge it must light up — proving the "
            "engine is unbiased and the real-tape null is genuine."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.0005):\n"
            "    syn = data.synthetic_panel(edge=edge, seed=371)\n"
            "    r = st.regress_forward(syn, 1); res.append((edge, r['t_skew']))\n"
            "labels = [f'planted edge\\n{e}' for e,_ in res]; tvals = [t for _,t in res]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(-2, ls='--', c=RED, label='|t|=2 (significance bar)'); ax.axhline(2, ls='--', c=RED); ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "ax.set_ylabel('HAC t(SKEW | VIX), 1-month'); ax.set_title('Control: knob OFF stays quiet, knob ON lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,t in res: print(f'planted edge={e}: t(SKEW|VIX)={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted edge the control sits at "
            f"**t = +{R['syn'][0][3]:.2f}** (no false positive); a genuine VIX-orthogonal edge drives "
            f"**t = {R['syn'][1][3]:.2f}** with the correct negative sign. The machinery detects a real "
            f"SKEW signal when one exists — so the real-tape **t = +{R['reg1'][3]:.2f}** is what an "
            "*absent* incremental signal looks like, not a power failure."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — VIX-controlled HAC **t(SKEW) = +{R['reg1'][3]:.2f}** (1m) / "
            f"**+{R['reg3'][3]:.2f}** (3m), below 2 and wrong-signed. The 3m bucket Welch "
            f"**t = {R['dec3'][6]:.2f}** is pseudo-replication (≈{R['blk3'][4]} episodes → block-bootstrap "
            f"**t = {R['blk3'][3]:.2f}**). No information beyond VIX ⇒ NONE, not WEAK.\n"
            f"- **Tradability `MIRAGE`** — fade-high-SKEW sleeve **{R['sleeve_cagr']:.1f}%** CAGR / Sharpe "
            f"**{R['sleeve_sharpe']:.2f}** vs buy-hold **{R['bh_cagr']:.1f}%** / **{R['bh_sharpe']:.2f}**. "
            "De-risking on the \"warning\" strictly *lowers* return and risk-adjusted return.\n"
            f"- **Warns before crashes? `BUSTED`** — −10% forward drawdown rate **{R['tail10'][1]:.1f}%** "
            f"after high SKEW vs **{R['tail10'][2]:.1f}%** base (placebo p ≈ {R['tail10'][3]:.2f}). The "
            "black-swan gauge prices fear that mostly doesn't realise; it fires before calm."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the cost of crying wolf\n\n"
            "The operational truth: the fade sleeve sits in cash ~10% of days and *gives up* the return "
            "of those days for nothing. Here is the equity curve of the \"warning\" trade against simply "
            "holding SPY."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ret = F['spy'].pct_change().fillna(0.0)\n"
            "    thr = F['skew'].quantile(0.90)\n"
            "    in_mkt = (F['skew'] < thr).shift(1).fillna(True).astype(float)\n"
            "    pos_change = in_mkt.diff().abs().fillna(0.0)\n"
            "    sleeve = (in_mkt*ret - pos_change*(2/1e4))\n"
            "    eq_sleeve = (1+sleeve).cumprod(); eq_bh = (1+ret).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "    ax.plot(eq_bh.index, eq_bh.values, c=GREEN, lw=1.4, label='buy & hold SPY')\n"
            "    ax.plot(eq_sleeve.index, eq_sleeve.values, c=RED, lw=1.4, label='fade high SKEW (to cash)')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)'); ax.set_xlabel('year')\n"
            "    ax.set_title('Crying wolf costs money: the \"warning\" trade trails buy-and-hold'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'final wealth — buy-hold: {eq_bh.iloc[-1]:.1f}x   fade sleeve: {eq_sleeve.iloc[-1]:.1f}x')\n"
            "else:\n"
            "    print('no cache — frozen: sleeve', R['sleeve_cagr'], '% vs buy-hold', R['bh_cagr'], '%')"
        ),
        md(
            "> 💡 In plain words: the red line trails the green one throughout — there is no episode where "
            "sitting out high-SKEW days rescued the portfolio, because high-SKEW days simply aren't "
            "where the losses concentrate. The entire economic content of \"de-risk when SKEW is high\" is "
            "**a small, persistent drag from missing good days**. There is no threshold, horizon, or cost "
            "regime that turns priced-but-unrealised tail fear into an edge."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Implied vs realised tails.** SKEW is a clean *risk-neutral* third moment — a measure of "
            "what the option market **charges** for the tail, i.e. tail *fear*. Realised tail risk is a "
            "different object; the variance-risk-premium literature (Bakshi-Kapadia-Madan; Bollerslev-"
            "Tauchen-Zhou) is the right lens for *why* priced fear and realised outcomes diverge.\n"
            "- **Orthogonalised SKEW.** Regress SKEW on VIX first and test only the residual — it is "
            "exactly the multivariate slope here, and it is null. There is no hidden signal in the "
            "VIX-orthogonal component.\n"
            "- **Cross-section vs index.** Option-implied skewness *does* show up in the cross-section of "
            "single stocks (Conrad-Dittmar-Ghysels); that is not the CBOE index-timing claim tested here, "
            "and conflating the two is the usual source of the SKEW myth.\n\n"
            "*The reproducible core is offline and deterministic. Methods and sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: [`docs/results.md`](../docs/results.md).*"
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
