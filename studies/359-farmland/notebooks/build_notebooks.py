"""Generate the two narrative notebooks for Study 359 (Farmland).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. The listed-proxy cells read the cached month-end
LAND/FPI/SPY prices under ../_cache/ (yfinance auto-adjusted) and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md). The hardcoded NCREIF/CPI series and the
synthetic appraisal-smoothing control run anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance LAND/FPI/SPY month-end,
# 2014-02 → 2026-06; hardcoded cited NCREIF farmland annual series 1992-2023). As-of 2026-06-22.
R = dict(
    win="2014-02 → 2026-06", n_land=149, n_fpi=146,
    land_ret=4.9, land_vol=30.0, land_beta=0.91, land_tbeta=5.19, land_corr=0.44,
    land_alpha=-8.2, land_talpha=-1.03, land_sh_g=0.16, land_sh_n=0.16,
    fpi_ret=5.5, fpi_vol=29.3, fpi_beta=0.69, fpi_tbeta=4.11, fpi_corr=0.35,
    fpi_alpha=-4.3, fpi_talpha=-0.59, fpi_sh_g=0.19, fpi_sh_n=0.18,
    spy_ret=14.4, spy_vol=14.6, spy_sh=0.98,
    nc_years="1992-2023", nc_n=32, nc_mean=11.0, nc_vol=6.6, nc_rho1=0.63,
    nc_real=8.4, nc_inflb=0.66, nc_infl_t=0.83, nc_corr_cpi=0.14, nc_corr_spx=-0.13,
    spx_mean=11.6, spx_vol=17.7, spx_inflb=-3.98, spx_infl_t=-2.69, spx_corr_cpi=-0.33,
    unsm_vol=13.5, vol_ratio=0.49,
    syn=[(0.10, 0.053, 0.900, 0.32, 0.03), (0.20, 0.111, 0.800, 0.32, 0.06),
         (0.40, 0.250, 0.600, 0.32, 0.13)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Smooth %26 uncorrelated%3F: Busted](https://img.shields.io/badge/Smooth_%26_uncorrelated%3F-Busted-c0392b?style=flat-square)\n\n"
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

from farmland import data, strategy as st

HAVE_REAL = data.have_real()
REAL = data.load_real() if HAVE_REAL else None
ANN = data.ncreif_annual()   # hardcoded cited public series — always present
print("listed-proxy price cache present:", HAVE_REAL,
      "| NCREIF annual rows:", len(ANN))
"""

# The frozen headline dict is embedded into the notebook's first code cell so every
# downstream cell can quote it whether or not the cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Farmland: the billionaires' secret inflation hedge? 🌾\n"
            "### The smoothest line on the chart — and why it's an optical illusion, in plain English\n\n"
            + BADGES +
            "You've heard the pitch. *Farmland is the best inflation hedge. It barely moves. It's "
            "uncorrelated with stocks. That's why Bill Gates is the biggest private farmland owner in "
            "America.* And the famous **NCREIF Farmland Index** seems to prove it: a calm, steady line "
            "that drifts up ~11%/year while stocks lurch around.\n\n"
            "Two things are true at once here, and the trick is telling them apart. Farmland really is "
            "a solid real asset. But the *calmness* — the low volatility, the near-zero correlation — "
            "is mostly an **accounting illusion**, and the only farmland you can actually buy looks "
            "nothing like that smooth line.\n\n"
            "> 📓 **Plain-language layer.** Want the regressions, the HAC *t*-stats and the un-smoothing "
            "algebra? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** Every chart below is drawn by the code beside it; house "
            "style in [METHODOLOGY.md](../../../METHODOLOGY.md). The NCREIF numbers are a cited public "
            "series; the listed REITs are real yfinance data."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is farmland a real asset that beats inflation? | **Yes.** The index earned "
            f"**~{R['nc_real']:.0f}%/year above inflation** over 30+ years. As a real-economy store of "
            "value, it's genuine. |\n"
            "| Is it the *smooth, uncorrelated* thing the chart shows? | **No — that's the illusion.** "
            f"The index is **appraisal-based** (valued by surveyors, not trades). Un-smooth it and its "
            f"volatility **doubles** ({R['nc_vol']:.1f}% → {R['unsm_vol']:.1f}%). |\n"
            "| Is it the *best inflation hedge*? | **Mildly, and unproven.** Its inflation link is "
            f"**positive but not statistically significant** (t = {R['nc_infl_t']:.2f} over "
            f"{R['nc_n']} years). It beats stocks at it, but the proof is weak. |\n"
            "| Can *I* buy that smooth line? | **No.** The only retail vehicles are two farmland REITs "
            f"(LAND, FPI). They carry **~30% volatility** and move *with* the stock market "
            f"(beta ≈ {R['land_beta']:.1f}). |\n\n"
            "> Real asset, real return — but the calm is painted on, and what you can actually hold is "
            "a leveraged, jumpy REIT."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Farmland is the ultimate inflation hedge: it compounds steadily, it barely "
            "fluctuates, and it's uncorrelated with the stock market — a diversifier that protects you "
            "when everything else falls. The smart money (Bill Gates owns ~270,000 acres) has known "
            "this for decades.\"*\n\n"
            "The evidence cited is almost always the **NCREIF Farmland Index** — institutional US "
            "farmland, total return (rent + land appreciation), published quarterly since the early "
            "1990s. On a chart it's the smoothest asset you'll ever see."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Three different claims are bundled together, and they don't stand or fall together. "
            "(1) *Does farmland earn a real return?* If yes, it's a legitimate store of value. "
            "(2) *Is it genuinely low-risk and uncorrelated?* Only if the smooth index reflects real "
            "prices — not surveyor estimates that lag reality. (3) *Can you actually own that profile?* "
            "Only if a tradable vehicle delivers it. We can check all three — and the answers split: "
            "**yes, no, and no.**"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The smooth index.** A cited public series of NCREIF farmland annual returns "
            f"({R['nc_years']}), next to CPI inflation and the S&P 500.\n"
            "- **The un-smoothing test.** An appraisal index is a *moving average* of true values, so "
            "it looks calm and lagged. There's a standard way to reverse that (the Geltner method) and "
            "see the volatility the survey hides.\n"
            "- **The tradable proxies.** The two listed farmland REITs — **LAND** (Gladstone Land) and "
            "**FPI** (Farmland Partners) — from real market data, compared to **SPY**. This is what a "
            "normal investor can actually buy."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the famous smooth chart.** Here is the appraisal index next to the stock market, "
            "year by year. One of these lines is calm; the other is a rollercoaster."
        ),
        code(
            "yrs = ANN.index.values\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "ax.plot(yrs, ANN['farmland']*100, 'o-', c=GREEN, lw=2, label='NCREIF farmland (appraised)')\n"
            "ax.plot(yrs, ANN['spx']*100, 's-', c=GREY, lw=1.4, alpha=.8, label='S&P 500')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('annual total return (%)'); ax.set_xlabel('year')\n"
            "ax.set_title('Farmland looks impossibly smooth next to stocks'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"farmland vol: {ANN['farmland'].std()*100:.1f}%   S&P vol: {ANN['spx'].std()*100:.1f}%\")"
        ),
        md(
            f"Farmland's annual swings are tiny (**{R['nc_vol']:.1f}%** vol) next to the S&P's "
            f"(**{R['spx_vol']:.1f}%**), and the two barely move together (correlation "
            f"**{R['nc_corr_spx']:+.2f}**). Case closed? Not quite — because *who decides farmland's "
            "price each year?* A surveyor, not a market."
        ),
        md(
            "**The catch: appraisal smoothing.** Nobody trades a cornfield every day, so its "
            "\"return\" is an appraiser's opinion that only nudges toward reality each year. That "
            "**moving-average** effect mechanically shrinks the ups and downs. The tell is "
            "*autocorrelation*: a real tradable asset's good and bad years are roughly random, but the "
            "appraisal index's years are strongly linked to the year before. Reverse the smoothing and "
            "the hidden volatility reappears:"
        ),
        code(
            "rep = ANN['farmland'].values\n"
            "rho1 = st.autocorr(rep, 1)\n"
            "sm = st.smoothing_summary(rep)\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "ax.bar(['as reported\\n(appraised)', 'un-smoothed\\n(implied true)'],\n"
            "       [sm['vol_reported']*100, sm['vol_unsmoothed']*100], color=[GREEN, RED], width=.55)\n"
            "for i,v in enumerate([sm['vol_reported']*100, sm['vol_unsmoothed']*100]):\n"
            "    ax.annotate(f'{v:.1f}%', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('annual volatility (%)')\n"
            "ax.set_title(f'Un-smoothing doubles the risk (year-to-year link rho1 = {rho1:.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'reported vol {sm[\"vol_reported\"]*100:.1f}%  ->  un-smoothed {sm[\"vol_unsmoothed\"]*100:.1f}%  (the survey hid ~{(1-sm[\"vol_ratio\"])*100:.0f}%)')"
        ),
        md(
            f"The reported {R['nc_vol']:.1f}% volatility is really more like **{R['unsm_vol']:.1f}%** "
            "once you undo the appraisal lag — the smoothness is paint, not steel. **And the only "
            "farmland you can buy proves it.** Here is what the tradable REITs actually did:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets = REAL['returns']\n"
            "    vols = [rets[c].std()*np.sqrt(12)*100 for c in ['LAND','FPI','SPY']]\n"
            "    betas = [st.market_beta(rets['LAND'], rets['SPY'])['beta'],\n"
            "             st.market_beta(rets['FPI'],  rets['SPY'])['beta'], 1.0]\n"
            "else:\n"
            "    vols  = [R['land_vol'], R['fpi_vol'], R['spy_vol']]\n"
            "    betas = [R['land_beta'], R['fpi_beta'], 1.0]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.2))\n"
            "labs = ['LAND', 'FPI', 'SPY']; cols = [RED, RED, GREY]\n"
            "a1.bar(labs, vols, color=cols, width=.6); a1.set_title('Annual volatility (%)')\n"
            "for i,v in enumerate(vols): a1.annotate(f'{v:.0f}%',(i,v),ha='center',va='bottom')\n"
            "a2.bar(labs, betas, color=cols, width=.6); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_title('Market beta (move-with-stocks)')\n"
            "for i,v in enumerate(betas): a2.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('tradable farmland: ~30% vol, beta ~0.7-0.9 — the opposite of calm & uncorrelated')"
        ),
        md(
            f"There it is. The buyable farmland (LAND, FPI) runs at **~30% volatility** — *double* the "
            f"S&P — and moves *with* the market (beta ≈ {R['land_beta']:.1f}). The serene 6.6%-vol "
            "diversifier exists only in the appraiser's spreadsheet."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** Farmland earns a real return (~{R['nc_real']:.0f}%/yr real) and "
            f"leans the right way on inflation — but the inflation link isn't statistically solid "
            f"(t = {R['nc_infl_t']:.2f}), and the diversification is mostly an appraisal artifact.\n"
            "- **Tradability — Fragile.** You *can* buy LAND/FPI, but they're small, leveraged, "
            "~30%-vol REITs (Sharpe ~0.2 vs the market's ~1.0) — not the smooth index.\n"
            "- **Smooth & uncorrelated? — Busted.** The calm is appraisal smoothing; un-smoothed, "
            "farmland's volatility doubles and its market link reappears."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — what you'd really own\n\n"
            "Suppose you're sold on farmland and buy the only retail vehicles. How does a dollar in "
            "LAND/FPI compare with a dollar in the S&P, after the bumps?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets = REAL['common']\n"
            "    growth = (1 + rets[['LAND','FPI','SPY']]).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "    for c, col in zip(['LAND','FPI','SPY'], [GREEN, AMBER, GREY]):\n"
            "        ax.plot(growth.index, growth[c], lw=1.8, label=c, c=col)\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "    ax.set_title('Buyable farmland vs the S&P — jumpier, and behind'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print({c: round(float(growth[c].iloc[-1]),2) for c in ['LAND','FPI','SPY']})\n"
            "else:\n"
            "    sh = {'LAND': R['land_sh_n'], 'FPI': R['fpi_sh_n'], 'SPY': R['spy_sh']}\n"
            "    fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "    ax.bar(list(sh), list(sh.values()), color=[GREEN, AMBER, GREY], width=.6)\n"
            "    for i,(k,v) in enumerate(sh.items()): ax.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "    ax.set_ylabel('net Sharpe ratio'); ax.set_title('Risk-adjusted return: farmland REITs vs S&P')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('net Sharpe — LAND %.2f  FPI %.2f  SPY %.2f' % (R['land_sh_n'], R['fpi_sh_n'], R['spy_sh']))"
        ),
        md(
            f"Over the listed era LAND/FPI returned ~5%/yr at ~30% vol — a **net Sharpe ~0.2** against "
            f"the S&P's **~{R['spy_sh']:.1f}** — while taking you on a much wilder ride. As a "
            "diversifier in a sleeve it has a place; as *the* inflation-proof calm compounder of the "
            "pitch, the tradable version doesn't deliver."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The same illusion elsewhere.** Every appraisal-based private-asset index — commercial "
            "real estate, private equity, infrastructure — wears the same smooth mask. The 0.63 "
            "year-to-year link is the giveaway; un-smooth before you trust a Sharpe or a correlation.\n"
            "- **The measurement-artifact cousin.** [Study 120 — Excess-CAPE-Yield](../../120-excess-cape-yield/) "
            "is the same shape: a real signal that turns into a mirage once you account for how it's "
            "measured.\n"
            "- **Do it yourself.** Swap in the live NCREIF quarterly series (members-only) and re-run "
            "the un-smoothing; the doubling of volatility is robust to the exact smoothing parameter.\n\n"
            "*Think farmland is your low-vol inflation shield? Plot the un-smoothed series next to your "
            "REIT holding's drawdowns — and decide whether you're buying the asset or the appraisal.*"
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
            "# Farmland — a quantitative teardown 🔬\n"
            "### CAPM beta of the listed proxies (HAC t) · the inflation regression vs CPI · "
            "Geltner appraisal un-smoothing · a closed-form synthetic smoothing control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We "
            "separate three claims the folklore fuses: a **real** real-asset return (genuine), a "
            "**weak/unproven** inflation-hedge loading, and a **busted** \"low-vol, uncorrelated\" "
            "property that is an *appraisal-smoothing artifact*. The decisive objects: the CAPM beta of "
            "the tradable REITs, the CPI loading of the index, and the un-smoothing identity that "
            "restores the hidden volatility and beta.\n\n"
            "> ⚠️ **Not investment advice.** Real data: yfinance auto-adjusted month-end LAND/FPI/SPY "
            "(2014-02 → 2026-06). The NCREIF farmland / CPI / S&P annual series is a **hardcoded cited "
            "public** dataset (1992-2023) — a transparent proxy for the paywalled quarterly index, "
            "labelled as such. Synthetic control is deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | NCREIF inflation-beta **+{R['nc_inflb']:.2f}** but "
            f"**t = {R['nc_infl_t']:.2f}** (n={R['nc_n']}) — positive, *insignificant*; real return "
            f"+{R['nc_real']:.1f}%/yr; tradable proxies are equity-like (β = {R['land_beta']:.2f}, "
            f"t = {R['land_tbeta']:.1f}). Real-economy plausible, not a robust t≥2 tradable signal. |\n"
            f"| **Tradability** | `FRAGILE` | LAND/FPI net Sharpe **{R['land_sh_n']:.2f}/{R['fpi_sh_n']:.2f}** "
            f"vs SPY **{R['spy_sh']:.2f}**, ann vol **~{R['land_vol']:.0f}%**; small, leveraged, "
            "illiquid REITs. Buyable, poor risk-adjusted, not the smooth index. |\n"
            f"| **Smooth & uncorrelated?** | `BUSTED` | Appraisal ρ₁ = **{R['nc_rho1']:.2f}**; "
            f"un-smoothing lifts vol {R['nc_vol']:.1f}% → **{R['unsm_vol']:.1f}%**. Synthetic control: a "
            "true 0.32 beta is reported as 0.03–0.13 — the \"uncorrelated\" calm is mechanical. |\n\n"
            "> 💡 In plain words: farmland really earns a real return, but its reputation as a calm, "
            "uncorrelated inflation shield is an accounting illusion — and the only thing you can hold "
            "is a jumpy, leveraged REIT."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{F}_t$ be farmland's total return, $\\pi_t$ realised CPI inflation, $r^{M}_t$ the "
            "market return. The pitch is three hypotheses:\n\n"
            "- **H₁ (real inflation hedge).** In $r^{F}_t = \\alpha + \\beta_\\pi \\pi_t + \\varepsilon_t$, "
            "$\\beta_\\pi > 0$ and significant.\n"
            "- **H₂ (low-beta diversifier).** In the CAPM $r^{F}_t - r^f = \\alpha + \\beta_M (r^M_t - r^f) + u_t$, "
            "$\\beta_M \\approx 0$ on the *tradable* asset.\n"
            "- **H₃ (the smooth index is the true risk).** The reported volatility and $\\beta_M$ are "
            "the volatility and beta you actually bear.\n\n"
            f"We find **H₁ weakly accepted** (sign right, $t = {R['nc_infl_t']:.2f}$), **H₂ rejected** "
            f"on the tradable proxies ($\\beta_M = {R['fpi_beta']:.2f}$–${R['land_beta']:.2f}$, large "
            "$t$), and **H₃ rejected**: the smooth index is an appraisal moving-average whose "
            "volatility and beta are shrunk by a known factor."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "An appraisal-based index reports $r^{app}_t = a\\,r^{true}_t + (1-a)\\,r^{app}_{t-1}$ "
            "(surveyors anchor on last period). For iid true returns this MA(∞) implies, in the limit,\n\n"
            "$$\\frac{\\mathrm{Var}(r^{app})}{\\mathrm{Var}(r^{true})} = \\frac{a}{2-a},\\qquad "
            "\\rho_1(r^{app}) = 1-a,\\qquad \\beta^{app}_M \\approx a\\,\\beta^{true}_M.$$\n\n"
            "So a small smoothing parameter $a$ **manufactures** low volatility, high autocorrelation, "
            "and a near-zero beta — *exactly* the three properties the pitch celebrates. The "
            "diagnostics ($\\rho_1$) reveal $a$, and the un-smoothing inverts the filter. The signal "
            "you can actually trade lives in the listed REITs, where no appraiser intervenes."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Listed proxies.** yfinance auto-adjusted month-end closes, LAND ({R['n_land']} mo), "
            f"FPI ({R['n_fpi']} mo), SPY; monthly simple total returns. CAPM beta via OLS with "
            "**Newey-West HAC** (6 lags) — robust to the autocorrelation overlapping/illiquid returns "
            "inject.\n"
            f"- **Inflation regression.** NCREIF farmland annual TR on CPI ({R['nc_years']}, "
            f"n={R['nc_n']}), HAC (2 lags); the S&P run as a contrast. The Signal axis is the *sign and "
            "significance* of $\\beta_\\pi$.\n"
            "- **Un-smoothing.** Geltner first-order reversal "
            "$r^{true}_t = (r^{app}_t - \\rho\\,r^{app}_{t-1})/(1-\\rho)$ with $\\rho=\\hat\\rho_1$; "
            "report the volatility lift.\n"
            "- **Positive control.** A deterministic smoothing generator with known $a,\\beta$: the "
            "measured variance ratio, $\\rho_1$ and appraised beta must hit the closed forms "
            "$a/(2-a)$, $1-a$, $a\\beta$.\n"
            "- **Costs (named).** One-way 15 bps on NAV for the thin REITs; leverage and illiquidity "
            "flagged as caveats, not modelled as alpha. Survivorship: only the two surviving REITs are "
            "in the proxy sample."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The tradable proxies are high-beta, not diversifiers\n\n"
            "CAPM beta of LAND and FPI on SPY, with HAC standard errors. A real low-beta diversifier "
            "would sit near zero; these don't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets = REAL['returns']\n"
            "    res = {c: st.market_beta(rets[c], rets['SPY'], lags=6) for c in ['LAND','FPI']}\n"
            "    betas = [res['LAND']['beta'], res['FPI']['beta']]\n"
            "    ses   = [res['LAND']['se_beta'], res['FPI']['se_beta']]\n"
            "    ts    = [res['LAND']['t_beta'], res['FPI']['t_beta']]\n"
            "else:\n"
            "    betas = [R['land_beta'], R['fpi_beta']]\n"
            "    ts    = [R['land_tbeta'], R['fpi_tbeta']]\n"
            "    ses   = [b/t for b,t in zip(betas, ts)]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(['LAND','FPI'], betas, yerr=[1.96*s for s in ses], capsize=5, color=RED, width=.5)\n"
            "ax.axhline(0, c='k', lw=.8, label='a true diversifier (beta=0)')\n"
            "ax.axhline(1, ls='--', c=GREY, label='the market (beta=1)')\n"
            "ax.set_ylabel('CAPM market beta'); ax.set_ylim(-.2, 1.2)\n"
            "ax.set_title('Listed farmland: beta 0.7-0.9, not zero (HAC 95% CI)'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for c,b,t in zip(['LAND','FPI'], betas, ts): print(f'{c}: beta={b:.2f}  HAC t={t:.2f}')"
        ),
        md(
            f"> 💡 In plain words: LAND's beta is **{R['land_beta']:.2f}** (HAC t = {R['land_tbeta']:.1f}), "
            f"FPI's **{R['fpi_beta']:.2f}** (t = {R['fpi_tbeta']:.1f}) — both decisively *not* zero. The "
            "only farmland you can hold trades like a high-vol equity. **H₂ rejected** on the tape."
        ),
        md(
            "### 4b · The inflation hedge is real-but-weak\n\n"
            "NCREIF farmland annual return regressed on CPI (the empirical content of \"inflation "
            "hedge\"), with the S&P 500 as a contrast. Positive slope = hedges; significance = whether "
            "we can trust it."
        ),
        code(
            "rf = st.inflation_beta(ANN, 'farmland', lags=2)\n"
            "rs = st.inflation_beta(ANN, 'spx', lags=2)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "x = ANN['cpi'].values*100\n"
            "ax.scatter(x, ANN['farmland'].values*100, c=GREEN, label='farmland', zorder=3)\n"
            "xs = np.linspace(x.min(), x.max(), 50)\n"
            "ax.plot(xs, (rf['alpha'] + rf['beta']*xs/100)*100, c=GREEN, lw=2)\n"
            "ax.set_xlabel('CPI inflation (%)'); ax.set_ylabel('farmland annual return (%)')\n"
            "ax.set_title(f\"Inflation beta +{rf['beta']:.2f} (t={rf['t_beta']:.2f}) — positive but not significant\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"farmland: infl-beta={rf['beta']:+.2f} (t={rf['t_beta']:.2f})  corr_CPI={rf['corr_cpi']:+.2f}  mean real={rf['mean_real']*100:.1f}%\")\n"
            "print(f\"S&P 500 : infl-beta={rs['beta']:+.2f} (t={rs['t_beta']:.2f})  corr_CPI={rs['corr_cpi']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: farmland's inflation loading is **+{R['nc_inflb']:.2f}** but "
            f"**t = {R['nc_infl_t']:.2f}** — the right sign, not significant on {R['nc_n']} years. "
            f"Equities load *negatively* (−{abs(R['spx_inflb']):.1f}, t = {R['spx_infl_t']:.1f}), so "
            "farmland is the better inflation companion — but \"best inflation hedge\" overstates what "
            "the data can prove. **H₁ weakly accepted → Signal WEAK** (no robust t≥2)."
        ),
        md(
            "### 4c · The smoothness is an appraisal artifact — un-smoothing\n\n"
            "The reported ρ₁ reveals the smoothing; reversing the filter restores the volatility a "
            "transaction-based index would show."
        ),
        code(
            "rep = ANN['farmland'].values\n"
            "sm = st.smoothing_summary(rep)\n"
            "un = st.unsmooth(rep)\n"
            "yrs = ANN.index.values\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.plot(yrs, rep*100, 'o-', c=GREEN, lw=2, label=f\"reported (vol {sm['vol_reported']*100:.1f}%, rho1 {sm['rho1']:.2f})\")\n"
            "ax.plot(yrs[1:], un*100, 's--', c=RED, lw=1.6, label=f\"un-smoothed (vol {sm['vol_unsmoothed']*100:.1f}%)\")\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('annual return (%)'); ax.set_xlabel('year')\n"
            "ax.set_title('Geltner un-smoothing reveals the hidden volatility'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"rho1={sm['rho1']:.2f}  reported vol={sm['vol_reported']*100:.1f}%  un-smoothed={sm['vol_unsmoothed']*100:.1f}%  ratio={sm['vol_ratio']:.2f}\")"
        ),
        md(
            f"> 💡 In plain words: ρ₁ = **{R['nc_rho1']:.2f}** is far too high for a freely-traded asset "
            f"— it's the appraisal moving-average. Inverting it lifts volatility {R['nc_vol']:.1f}% → "
            f"**{R['unsm_vol']:.1f}%** (the survey hides ~{(1-R['vol_ratio'])*100:.0f}%). **H₃ rejected.**"
        ),
        md(
            "### 4d · Faithful-engine control — measured shrink = closed form\n\n"
            "On a deterministic smoothed tape with known $a$ and a true beta of 0.30, the measured "
            "variance ratio, ρ₁ and appraised beta must equal $a/(2-a)$, $1-a$, $\\approx a\\beta$:"
        ),
        code(
            "rows = []\n"
            "for a in (0.10, 0.20, 0.40):\n"
            "    df = data.synthetic_smoothing(n=4000, a=a, beta=0.30, seed=359)\n"
            "    vr = np.var(df['app'],ddof=1)/np.var(df['true'],ddof=1)\n"
            "    rho = st.autocorr(df['app'].values,1)\n"
            "    ba = st.ols_hac(df['app'].values, df['mkt'].values, lags=4)['beta']\n"
            "    rows.append((a, vr, st.expected_var_ratio(a), rho, st.expected_rho1(a), ba))\n"
            "a_=[r[0] for r in rows]; x=np.arange(len(a_))\n"
            "fig, (p1,p2) = plt.subplots(1,2,figsize=(9.8,4.2))\n"
            "p1.bar(x-.18,[r[1] for r in rows],.36,color=GREEN,label='measured')\n"
            "p1.bar(x+.18,[r[2] for r in rows],.36,color=GREY,label='exact a/(2-a)')\n"
            "p1.set_xticks(x); p1.set_xticklabels([f'a={a}' for a in a_]); p1.set_title('Variance ratio'); p1.legend()\n"
            "p2.bar(x-.18,[r[3] for r in rows],.36,color=GREEN,label='measured')\n"
            "p2.bar(x+.18,[r[4] for r in rows],.36,color=GREY,label='exact 1-a')\n"
            "p2.set_xticks(x); p2.set_xticklabels([f'a={a}' for a in a_]); p2.set_title('Lag-1 autocorrelation'); p2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for a,vr,vre,rho,rhoe,ba in rows: print(f'a={a}: var-ratio {vr:.3f} (exact {vre:.3f})  rho1 {rho:.3f} (exact {rhoe:.3f})  true beta 0.30 -> appraised {ba:.2f}')"
        ),
        md(
            "> 💡 In plain words: the engine reproduces the closed-form shrink at every $a$ — the "
            "\"low volatility, near-zero beta\" of an appraisal index is a *mechanical* consequence of "
            "smoothing, not a property of farmland. A true 0.30 beta is reported as 0.03–0.13."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — inflation-beta +{R['nc_inflb']:.2f} (t = {R['nc_infl_t']:.2f}, "
            f"n = {R['nc_n']}): right sign, *not significant*; real return +{R['nc_real']:.1f}%/yr is "
            "genuine, but the tradable proxies are equity-like (β = 0.7–0.9, large t). Real-economy "
            "plausible, no robust t≥2 tradable signal.\n"
            f"- **Tradability `FRAGILE`** — LAND/FPI net Sharpe {R['land_sh_n']:.2f}/{R['fpi_sh_n']:.2f} "
            f"vs SPY {R['spy_sh']:.2f}, ~{R['land_vol']:.0f}% vol; small, leveraged, illiquid. Buyable, "
            "poor, and nothing like the smooth index.\n"
            f"- **Smooth & uncorrelated? `BUSTED`** — ρ₁ = {R['nc_rho1']:.2f}; un-smoothing lifts vol "
            f"{R['nc_vol']:.1f}% → {R['unsm_vol']:.1f}%; the control shows a true 0.30 beta reported as "
            "~0.05. The calm is measurement, not the asset."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the risk-adjusted reality\n\n"
            "Net-of-cost Sharpe and volatility of the holdable proxies vs the market — the honest "
            "risk-adjusted comparison for a real-money allocation."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets = REAL['returns']\n"
            "    rows = {c: st.net_sharpe(rets[c], cost_oneway=0.0015) for c in ['LAND','FPI']}\n"
            "    rows['SPY'] = st.net_sharpe(rets['SPY'], cost_oneway=0.0001)\n"
            "    sh = {c: rows[c]['sharpe_net'] for c in ['LAND','FPI','SPY']}\n"
            "    vol = {c: rows[c]['ann_vol']*100 for c in ['LAND','FPI','SPY']}\n"
            "else:\n"
            "    sh = {'LAND': R['land_sh_n'], 'FPI': R['fpi_sh_n'], 'SPY': R['spy_sh']}\n"
            "    vol = {'LAND': R['land_vol'], 'FPI': R['fpi_vol'], 'SPY': R['spy_vol']}\n"
            "fig, (a1,a2) = plt.subplots(1,2,figsize=(9.8,4.2))\n"
            "ks = list(sh); cols=[RED,RED,GREY]\n"
            "a1.bar(ks,[sh[k] for k in ks],color=cols,width=.6); a1.set_title('Net Sharpe (after 15bps)')\n"
            "for i,k in enumerate(ks): a1.annotate(f'{sh[k]:.2f}',(i,sh[k]),ha='center',va='bottom')\n"
            "a2.bar(ks,[vol[k] for k in ks],color=cols,width=.6); a2.set_title('Annual volatility (%)')\n"
            "for i,k in enumerate(ks): a2.annotate(f'{vol[k]:.0f}%',(i,vol[k]),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net Sharpe:', {k: round(sh[k],2) for k in ks}, ' vol%:', {k: round(vol[k],0) for k in ks})"
        ),
        md(
            f"> 💡 In plain words: net Sharpe ~{R['land_sh_n']:.2f} on the farmland REITs against "
            f"~{R['spy_sh']:.2f} on the S&P, at double the volatility. There is no sizing that turns the "
            "tradable vehicle into the smooth index — the appraisal calm simply isn't for sale. A small "
            "diversifying sleeve is defensible; \"the best inflation hedge\" is not what the tape shows."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Generalised un-smoothing.** Replace the first-order reversal with the Fisher-Geltner "
            "multi-lag filter or the Getmansky-Lo-Makarov MA(q) estimator; the volatility-doubling is "
            "robust to the order. Re-estimate on the live NCREIF quarterly series.\n"
            "- **Inflation decomposition.** Split CPI into expected vs unexpected (Fama-Schwert) and "
            "test farmland's loading on each; the hedge story is usually about *unexpected* inflation, "
            "where power is even lower.\n"
            "- **The artifact family.** Every appraisal/marked-to-model private index "
            "([commercial RE, PE, infra]) shares the ρ₁ fingerprint; the desk's "
            "[Study 120](../../120-excess-cape-yield/) is the equity-valuation cousin of "
            "\"real signal, mirage in the measurement.\"\n\n"
            "*The reproducible core is offline and deterministic; the listed-proxy cells fall back to "
            "the frozen `R` numbers when the cache is absent. Methods and sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
            "[`docs/results.md`](../docs/results.md).*"
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
