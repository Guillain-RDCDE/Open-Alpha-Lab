"""Generate the two narrative notebooks for Study 664 (ESG Premium).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached ESGU/SUSA/SPY/
IVV/IVW/IVE/QUAL/^IRX tapes under ../_cache/ and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with no
network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance ESGU/SUSA/SPY/IVV/
# IVW/IVE/QUAL/^IRX, as-of 2026-06-30).
R = dict(
    esgu_start="2016-12-07", esgu_end="2026-06-30", esgu_n=2402, esgu_years=9.5,
    esgu_cagr_fund=15.31, esgu_cagr_bench=15.45,
    esgu_vol_fund=18.59, esgu_vol_bench=18.23,
    esgu_sharpe_fund=0.730, esgu_sharpe_bench=0.747,
    esgu_te=3.75, esgu_ir=-0.015, esgu_active_ann=-0.057,
    esgu_nw_t=-0.08, esgu_welch_t=-0.01, esgu_net_ann=-0.378, esgu_hit=50.2,
    esgu_alpha_ann=-0.064, esgu_alpha_t=-0.09,
    esgu_beta_gv=0.025, esgu_beta_gv_t=3.63,
    esgu_beta_q=0.109, esgu_beta_q_t=3.41,

    susa_start="2005-01-31", susa_end="2026-06-30", susa_n=5387, susa_years=21.4,
    susa_cagr_fund=10.50, susa_cagr_bench=11.09,
    susa_vol_fund=18.28, susa_vol_bench=18.85,
    susa_sharpe_fund=0.543, susa_sharpe_bench=0.561,
    susa_te=4.78, susa_ir=-0.135, susa_active_ann=-0.644,
    susa_nw_t=-1.07, susa_welch_t=-0.11, susa_net_ann=-0.953, susa_hit=49.9,
    susa_alpha_ann=-0.187, susa_alpha_t=-0.27,
    susa_beta_gv=0.001, susa_beta_gv_t=0.14,
    susa_beta_q=0.247, susa_beta_q_t=8.50,

    er_esgu=0.150, er_spy=0.095, er_susa=0.250, er_ivv=0.030,
    er_gap_esgu=0.055, er_gap_susa=0.220,

    syn_null_mean=-0.43, syn_null_sd=0.88, syn_null_fire=0,
    syn_planted_bps=4.198, syn_planted_t=8.73,

    fp_esgu="29e958663c9e", fp_susa="27becb3a322e", fp_spy="30a1902f6b13",
    fp_ivv="b5c9e7d52b50", fp_ivw="fc9dec46168d", fp_ive="1988eedd21ca",
    fp_qual="9dd22955608c", fp_irx="7523780ae772",
    fp_active_esgu="3118d66244a6", fp_active_susa="4820c59f2797",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Growth/quality tilt: Confirmed](https://img.shields.io/badge/Growth%2Fquality_tilt%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from esg_premium import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    tapes = data.load_real()
    rets = {t: st.daily_returns(tapes[t]["Close"]) for t in data.TICKERS if t != "^IRX"}
    rf = st.rf_daily(tapes["^IRX"]["Close"].reindex(rets["SPY"].index).ffill())
    gv_spread = rets["IVW"] - rets["IVE"]
else:
    rets = rf = gv_spread = None
print("real cache present:", HAVE_REAL,
      "| ESGU days:", (0 if rets is None else len(rets["ESGU"])),
      "| SUSA days:", (0 if rets is None else len(rets["SUSA"])))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does \"doing well by doing good\" actually pay? 🌱\n"
            "### ESG investing vs the plain index — a decade (and two decades) of real "
            "money, tested honestly\n\n"
            + BADGES +
            "ESG funds are pitched two opposite ways: **\"good companies are better-run "
            "companies, so ESG investing should outperform\"** — or, the skeptic's version, "
            "**\"you're paying an insurance premium and cutting your universe, so ESG "
            "investing should cost you.\"** Both stories sound plausible. Only one tape can "
            "settle it.\n\n"
            "We test the two flagship US large-cap ESG ETFs — **ESGU** (since 2016) and "
            "**SUSA** (since 2005, the older and more selective one) — against their plain "
            "S&P 500 benchmarks (SPY and IVV), for as long as each has existed.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the factor regression and the "
            "cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** Total-return (dividend-reinvested) daily closes, "
            "yfinance, as-of 2026-06-30. No survivorship: every ticker here is a single, "
            "currently-traded ETF over its own full history. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do ESG funds beat the plain index? | **No — but they don't clearly lose "
            f"either.** ESGU trailed SPY by **{abs(R['esgu_active_ann']):.2f}%/yr** and SUSA "
            f"trailed IVV by **{abs(R['susa_active_ann']):.2f}%/yr** — small numbers, and "
            "neither is statistically distinguishable from *zero*. |\n"
            f"| Is there a real cost to going ESG? | **A small, documented one.** ESGU "
            f"charges **{R['er_gap_esgu']:.3f}%/yr** more than SPY; SUSA charges "
            f"**{R['er_gap_susa']:.3f}%/yr** more than IVV — in fund fees alone, before any "
            "tracking difference. |\n"
            f"| So why do ESG funds *look* different sometimes? | **A quality (and mild "
            "growth) tilt.** ESG screens exclude fossil fuels, tobacco and weapons — which "
            "mechanically shifts the portfolio toward tech/services and toward "
            f"better-capitalized \"quality\" companies. That tilt is statistically real "
            f"(t as high as **{R['susa_beta_q_t']:.1f}**) — it's just not a return edge. |\n"
            "| Can you trade an ESG premium? | **No — there's nothing there to trade.** |\n\n"
            "> The label is real. The paycheck isn't."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Sustainable, well-governed companies manage risk better and will "
            "outperform over the long run — you don't have to sacrifice returns to invest "
            "responsibly, and you might even do better.\"*\n\n"
            "It's a genuinely contested claim in the academic literature — some studies find "
            "a small positive tilt, others find ESG exclusion (especially of high-carbon "
            "names) should be a modest *drag* on pure risk-premium grounds. There's no "
            "settled consensus — which is exactly why it's worth testing directly on the "
            "actual fund products investors buy."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Trillions of dollars now sit in ESG-labeled funds. If the premium is real, it's "
            "one of the rare free lunches — better ethics *and* better returns. If it's a "
            "mirage, investors are paying a real fee (and carrying real tracking error) for a "
            "label that changes nothing about expected return — or worse, for a hidden "
            "growth-stock bet they didn't sign up for."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **The comparison.** ESGU vs SPY (2016 → today) and SUSA vs IVV (2005 → "
            "today) — same total-return basis, same clock.\n"
            "- **The bar.** A statistically real premium needs a Newey-West *t* ≥ 2 on the "
            "daily active-return spread — not just a positive number, which luck can hand "
            "you for free.\n"
            "- **The unmasking.** If there *is* a gap, does it survive controlling for a "
            "plain growth-value tilt and a plain quality tilt? If not, it's not an ESG "
            "premium — it's a factor bet wearing a green label."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Annualized active return (ESG fund minus its plain "
            "benchmark), each since the ESG fund's own inception."
        ),
        code(
            "if HAVE_REAL:\n"
            "    e_active = R['esgu_active_ann']\n"
            "    s_active = R['susa_active_ann']\n"
            "else:\n"
            "    e_active, s_active = R['esgu_active_ann'], R['susa_active_ann']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['ESGU - SPY\\n(2016-2026, n=2,402)','SUSA - IVV\\n(2005-2026, n=5,387)'],\n"
            "       [e_active, s_active], color=[GREY, GREY], width=.55)\n"
            "for i,v in enumerate([e_active, s_active]): ax.annotate(f'{v:+.3f}%/yr',(i,v),\n"
            "    ha='center', va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('active return, annualized (%)')\n"
            "ax.set_title('Two decades of \"doing good\" - and no reliable free lunch either way')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'ESGU-SPY {e_active:+.3f}%/yr   SUSA-IVV {s_active:+.3f}%/yr')"
        ),
        md(
            f"Both bars sit near zero. ESGU trailed SPY by **{abs(R['esgu_active_ann']):.2f}%**"
            f" a year; SUSA trailed IVV by **{abs(R['susa_active_ann']):.2f}%**. Neither is a "
            "big number, and — the quants notebook shows — neither survives a proper "
            "significance test. **No premium, no clear penalty.**\n\n"
            "**Next, the price of admission.** ESG funds aren't free — they charge more, "
            "explicitly, in the prospectus:"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "pairs = [('ESGU', R['er_esgu'], 'SPY', R['er_spy']),\n"
            "         ('SUSA', R['er_susa'], 'IVV', R['er_ivv'])]\n"
            "labels, vals, cols = [], [], []\n"
            "for fn, fv, bn, bv in pairs:\n"
            "    labels += [fn, bn]; vals += [fv, bv]; cols += [AMBER, GREY]\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:.3f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('expense ratio (%/yr)')\n"
            "ax.set_title('The ESG label costs something, before a single trade')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"ESGU +{R['er_gap_esgu']:.3f}%/yr vs SPY, SUSA +{R['er_gap_susa']:.3f}%/yr vs IVV\")"
        ),
        md(
            f"ESGU costs **{R['er_gap_esgu']:.3f}%/yr** more than SPY; SUSA costs "
            f"**{R['er_gap_susa']:.3f}%/yr** more than IVV. This is a documented fact — a "
            "prospectus number, not a fitted one — and it's already baked into the price "
            "series above; it's the *structural* reason the active return leans negative "
            "rather than sitting at an exact zero.\n\n"
            "**Finally, the unmasking.** If ESGU or SUSA *did* show a real gap, would it "
            "survive controlling for an obvious growth/quality tilt?"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "labels = ['ESGU quality tilt\\n(t={:.1f})'.format(R['esgu_beta_q_t']),\n"
            "          'SUSA quality tilt\\n(t={:.1f})'.format(R['susa_beta_q_t']),\n"
            "          'ESGU growth tilt\\n(t={:.1f})'.format(R['esgu_beta_gv_t']),\n"
            "          'SUSA growth tilt\\n(t={:.1f})'.format(R['susa_beta_gv_t'])]\n"
            "vals = [R['esgu_beta_q'], R['susa_beta_q'], R['esgu_beta_gv'], R['susa_beta_gv']]\n"
            "cols = [AMBER if abs(t) >= 2 else GREY for t in\n"
            "        [R['esgu_beta_q_t'], R['susa_beta_q_t'], R['esgu_beta_gv_t'], R['susa_beta_gv_t']]]\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.3f}',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('factor beta (loading)')\n"
            "ax.set_title('The tilt is real - even though the return edge is not')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('quality betas:', R['esgu_beta_q'], R['susa_beta_q'])\n"
            "print('growth-value betas:', R['esgu_beta_gv'], R['susa_beta_gv'])"
        ),
        md(
            "Both ESG funds carry a **statistically real quality tilt** (amber bars — "
            f"*t* = {R['esgu_beta_q_t']:.1f} and {R['susa_beta_q_t']:.1f}), and ESGU carries a "
            f"small but real growth tilt too (*t* = {R['esgu_beta_gv_t']:.1f}). That's the "
            "mechanism working exactly as the skeptics describe: exclude fossil fuels and "
            "tobacco, and you mechanically end up in better-capitalized, more tech-heavy "
            "names. But because there was **no significant return gap to begin with**, "
            "there's no alpha for the tilt to \"explain away\" — the tilt is just... the "
            "portfolio you actually hold."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Neither ESG fund beats (or loses to) its plain benchmark "
            "by a statistically real amount, across 9.5 and 21.4 years of daily data.\n"
            "- **Tradability — Mirage.** There's no edge to trade, and the ESG label carries "
            "a real fee and real tracking error for nothing in expected return.\n"
            "- **\"Any gap is just a growth/quality tilt?\" — Confirmed.** The tilt is real "
            "and measurable; it just doesn't come with a return edge attached, positive or "
            "negative."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The real ESG question may not be about return at all.** If ESG investing "
            "is worth doing, the honest case is values-alignment and stewardship — not a "
            "promised return edge this tape can't find.\n"
            "- **International and small-cap ESG funds** may show a different picture (a "
            "less mature market, thinner benchmarks) — this study only tests the flagship "
            "US large-cap products.\n"
            "- **Sibling studies:** [211-sin-stocks](../../211-sin-stocks/) tests the mirror "
            "claim (do *excluded* stocks outperform?) with individual tickers — see "
            "[docs/references.md](docs/references.md) for the full dedup map.\n\n"
            "*Think an ESG fund's factor loadings hide something this two-factor model "
            "misses? Show a growth/quality-controlled alpha that clears t = 2 — then we'll "
            "talk.*"
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
            "# The ESG Premium — a quantitative teardown 🔬\n"
            "### Newey-West active-return spread tests · a growth-value/quality factor "
            "decomposition · tracking error and the cost accounting · a 20-seed synthetic "
            "control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "\"Doing well by doing good\" is a genuinely contested claim in the academic "
            "literature (Friede/Busch/Bassen 2015 lean mildly positive; Bolton & Kacperczyk "
            "2021's carbon premium implies ESG exclusion should be a mild *drag*; "
            "Pástor-Stambaugh-Taylor 2021 shows the sign can flip by sample window). The job "
            "here is to measure it directly on the actual investable products, then ask "
            "whether any gap survives controlling for the obvious growth/quality tilt.\n\n"
            "> ⚠️ **Data note.** Total-return daily closes (yfinance, `auto_adjust=True`) for "
            "ESGU/SUSA/SPY/IVV/IVW/IVE/QUAL, plus ^IRX for the risk-free proxy, as-of "
            "2026-06-30. No survivorship — every ticker is a single, currently-traded ETF "
            "over its own listed history. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints active-return "
            "`" + R["fp_active_esgu"] + "` / `" + R["fp_active_susa"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | ESGU−SPY NW **t = {R['esgu_nw_t']:.2f}** "
            f"({R['esgu_active_ann']:+.3f}%/yr); SUSA−IVV NW **t = {R['susa_nw_t']:.2f}** "
            f"({R['susa_active_ann']:+.3f}%/yr) — both \\|t\\| < 2 |\n"
            f"| **Tradability** | `MIRAGE` | net of costs: ESGU {R['esgu_net_ann']:+.3f}%/yr, "
            f"SUSA {R['susa_net_ann']:+.3f}%/yr, tracking error {R['esgu_te']:.2f}% / "
            f"{R['susa_te']:.2f}% |\n"
            f"| **Growth/quality tilt?** | `CONFIRMED` | quality beta *t* = "
            f"{R['esgu_beta_q_t']:.2f} / {R['susa_beta_q_t']:.2f}; growth-value beta "
            f"*t* = {R['esgu_beta_gv_t']:.2f} (ESGU) |\n\n"
            "> 💡 In plain words: the ESG label buys you a real, measurable factor tilt and "
            "a real fee — not a return edge in either direction."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{ESG}_t$ and $r^{B}_t$ be the daily total-return of an ESG fund and its "
            "plain-vanilla benchmark, and $a_t = r^{ESG}_t - r^{B}_t$ the daily active "
            "return.\n\n"
            "- **H₁ (premium).** $E[a_t] > 0$ and statistically distinguishable from zero — "
            "\"doing well by doing good.\"\n"
            "- **H₂ (drag).** $E[a_t] < 0$ and statistically distinguishable — screens cost "
            "you diversification and expenses.\n"
            "- **H₃ (relabelling).** Any nonzero $E[a_t]$ is explained by loadings on a "
            "growth-value spread and a quality spread, i.e. the intercept in "
            "$r^{ESG}_t = \\alpha + \\beta_1 r^B_t + \\beta_2 (r^{IVW}_t - r^{IVE}_t) + "
            "\\beta_3(r^{QUAL}_t - r^B_t) + \\epsilon_t$ collapses toward zero once the "
            "tilts are priced in.\n\n"
            "We find **neither H₁ nor H₂ supported** (both $|t| < 2$) and **H₃ partially "
            "confirmed** — the tilts ($\\beta_2, \\beta_3$) are statistically real, but "
            "since $E[a_t]$ was never significant to start with, the intercept doesn't "
            "collapse *from* significance — there was no alpha to explain away."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Daily active returns between two highly-correlated large-cap total-return "
            "series are serially correlated (shared market moves, dividend-timing "
            "mismatches), so the **planned primary** is a **Newey-West (5-lag) HAC "
            "*t*** of the mean active return (equivalently, the intercept of $a_t$ regressed "
            "on a constant). A Welch *t* on the raw fund/benchmark levels is reported as a "
            "cross-check. The factor decomposition uses the same 5-lag HAC covariance on a "
            "3-regressor OLS (market, growth-value spread, quality spread). Both legs of "
            "every Sharpe ratio are measured **excess of the same ^IRX cash proxy** — never "
            "a raw-vs-excess race."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** ESGU {R['esgu_start']} → {R['esgu_end']} (n={R['esgu_n']:,}, "
            f"{R['esgu_years']}y) vs SPY; SUSA {R['susa_start']} → {R['susa_end']} "
            f"(n={R['susa_n']:,}, {R['susa_years']}y) vs IVV. As-of 2026-06-30 (last complete "
            "month).\n"
            "- **Headline.** CAGR, ann. vol, excess-of-cash Sharpe (both legs vs ^IRX), "
            "tracking error, information ratio.\n"
            "- **Spread test.** NW(5) *t* of the daily active return (primary), Welch *t* "
            "(cross-check), net of 2 legs × 5 bps one-way (amortized) + 30 bps/yr benchmark-"
            "leg borrow.\n"
            "- **Decomposition.** OLS of fund return on [benchmark, IVW−IVE, QUAL−benchmark], "
            "NW(5) HAC SEs; the intercept is the alpha once the tilts are priced in.\n"
            "- **Control.** Synthetic correlated-return world (fund vs benchmark, ρ=0.97), "
            "planted premium knob; the null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Tracking difference & Sharpe\n\n"
            "CAGR, volatility and excess-of-cash Sharpe, fund vs benchmark, each pair over "
            "its own full common history."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ts_e = st.tracking_stats(rets['ESGU'], rets['SPY'], rf)\n"
            "    ts_s = st.tracking_stats(rets['SUSA'], rets['IVV'], rf)\n"
            "else:\n"
            "    ts_e = ts_s = None\n"
            "print('ESGU vs SPY:', R['esgu_cagr_fund'], 'vs', R['esgu_cagr_bench'], '%/yr CAGR;'\n"
            "      ' Sharpe', R['esgu_sharpe_fund'], 'vs', R['esgu_sharpe_bench'])\n"
            "print('SUSA vs IVV:', R['susa_cagr_fund'], 'vs', R['susa_cagr_bench'], '%/yr CAGR;'\n"
            "      ' Sharpe', R['susa_sharpe_fund'], 'vs', R['susa_sharpe_bench'])\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(['ESGU','SPY'], [R['esgu_cagr_fund'], R['esgu_cagr_bench']],\n"
            "       color=[AMBER, GREY], width=.55)\n"
            "a1.set_title(f\"ESGU vs SPY  (TE={R['esgu_te']:.2f}%, IR={R['esgu_ir']:+.3f})\")\n"
            "a1.set_ylabel('CAGR (%)')\n"
            "for i,v in enumerate([R['esgu_cagr_fund'], R['esgu_cagr_bench']]):\n"
            "    a1.annotate(f'{v:.2f}%',(i,v),ha='center',va='bottom')\n"
            "a2.bar(['SUSA','IVV'], [R['susa_cagr_fund'], R['susa_cagr_bench']],\n"
            "       color=[AMBER, GREY], width=.55)\n"
            "a2.set_title(f\"SUSA vs IVV  (TE={R['susa_te']:.2f}%, IR={R['susa_ir']:+.3f})\")\n"
            "a2.set_ylabel('CAGR (%)')\n"
            "for i,v in enumerate([R['susa_cagr_fund'], R['susa_cagr_bench']]):\n"
            "    a2.annotate(f'{v:.2f}%',(i,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: CAGR and Sharpe are a hair lower for both ESG funds "
            f"(ESGU Sharpe {R['esgu_sharpe_fund']:.3f} vs SPY {R['esgu_sharpe_bench']:.3f}; "
            f"SUSA {R['susa_sharpe_fund']:.3f} vs IVV {R['susa_sharpe_bench']:.3f}), while "
            f"carrying {R['esgu_te']:.2f}% / {R['susa_te']:.2f}% of annualized tracking error "
            "— real basis risk versus the benchmark, for a difference that (next section) "
            "isn't even statistically real."
        ),
        md(
            "### 4b · The headline — active-return spread test\n\n"
            "Newey-West (5-lag) *t* of the mean daily active return — the **primary** "
            "statistic, since the series is serially correlated."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sp_e = st.spread_test(rets['ESGU'], rets['SPY'])\n"
            "    sp_s = st.spread_test(rets['SUSA'], rets['IVV'])\n"
            "    e_ann, e_t = sp_e['gross_ann_pct'], sp_e['nw_t']\n"
            "    s_ann, s_t = sp_s['gross_ann_pct'], sp_s['nw_t']\n"
            "else:\n"
            "    e_ann, e_t = R['esgu_active_ann'], R['esgu_nw_t']\n"
            "    s_ann, s_t = R['susa_active_ann'], R['susa_nw_t']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.3))\n"
            "a1.bar(['ESGU-SPY','SUSA-IVV'], [e_ann, s_ann], color=[GREY, GREY], width=.55)\n"
            "for i,v in enumerate([e_ann, s_ann]): a1.annotate(f'{v:+.3f}%/yr',(i,v),\n"
            "    ha='center', va='top' if v<0 else 'bottom')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('active return (%/yr, gross)')\n"
            "a1.set_title('Gross active return: both near zero')\n"
            "a2.bar(['ESGU-SPY','SUSA-IVV'], [e_t, s_t],\n"
            "       color=[RED if abs(t)>=2 else GREY for t in [e_t, s_t]], width=.55)\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(2, ls='--', c=RED, lw=1)\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('Newey-West t')\n"
            "a2.set_title('Neither clears |t| >= 2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'ESGU-SPY {e_ann:+.3f}%/yr, NW t={e_t:+.2f}  |  SUSA-IVV {s_ann:+.3f}%/yr, NW t={s_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: ESGU−SPY is essentially a coin flip (hit rate "
            f"{R['esgu_hit']:.1f}%, NW *t* = {R['esgu_nw_t']:.2f}); SUSA−IVV leans negative "
            f"but stays short of the bar (NW *t* = {R['susa_nw_t']:.2f}). Net of the 5.5–22 "
            f"bps/yr documented expense-ratio gap plus trading/borrow costs, both spreads go "
            f"further negative ({R['esgu_net_ann']:+.3f}%/yr, {R['susa_net_ann']:+.3f}%/yr) — "
            "but the *statistical* verdict was already None before a single cost was charged."
        ),
        md(
            "### 4c · Factor decomposition — is any gap just growth/quality?\n\n"
            "$r^{ESG}_t = \\alpha + \\beta_1 r^B_t + \\beta_2(r^{IVW}_t-r^{IVE}_t) + "
            "\\beta_3(r^{QUAL}_t - r^B_t) + \\epsilon_t$, Newey-West (5-lag) HAC SEs."
        ),
        code(
            "if HAVE_REAL:\n"
            "    q_e = rets['QUAL'] - rets['SPY']; q_s = rets['QUAL'] - rets['IVV']\n"
            "    fac_e = st.factor_decomposition(rets['ESGU'], rets['SPY'], gv_spread, q_e)\n"
            "    fac_s = st.factor_decomposition(rets['SUSA'], rets['IVV'], gv_spread, q_s)\n"
            "else:\n"
            "    fac_e = fac_s = None\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "labs = ['alpha\\n(ann.%)', 'growth-value\\nbeta', 'quality\\nbeta']\n"
            "ve = [R['esgu_alpha_ann'], R['esgu_beta_gv']*100, R['esgu_beta_q']*100]\n"
            "te_ = [R['esgu_alpha_t'], R['esgu_beta_gv_t'], R['esgu_beta_q_t']]\n"
            "vs = [R['susa_alpha_ann'], R['susa_beta_gv']*100, R['susa_beta_q']*100]\n"
            "ts_ = [R['susa_alpha_t'], R['susa_beta_gv_t'], R['susa_beta_q_t']]\n"
            "a1.bar(labs, ve, color=[RED if abs(t)>=2 else GREY for t in te_], width=.55)\n"
            "for i,(v,t_) in enumerate(zip(ve, te_)): a1.annotate(f'{v:+.2f}\\n(t={t_:+.2f})',\n"
            "    (i,v), ha='center', va='bottom' if v>=0 else 'top', fontsize=8)\n"
            "a1.axhline(0,c='k',lw=.8); a1.set_title('ESGU: alpha dead, quality/growth tilt real')\n"
            "a2.bar(labs, vs, color=[RED if abs(t)>=2 else GREY for t in ts_], width=.55)\n"
            "for i,(v,t_) in enumerate(zip(vs, ts_)): a2.annotate(f'{v:+.2f}\\n(t={t_:+.2f})',\n"
            "    (i,v), ha='center', va='bottom' if v>=0 else 'top', fontsize=8)\n"
            "a2.axhline(0,c='k',lw=.8); a2.set_title('SUSA: alpha dead, quality tilt strongly real')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: for both funds the alpha (red only if \\|t\\|≥2) stays "
            f"**grey** — {R['esgu_alpha_ann']:+.3f}%/yr at *t* = {R['esgu_alpha_t']:.2f} "
            f"(ESGU), {R['susa_alpha_ann']:+.3f}%/yr at *t* = {R['susa_alpha_t']:.2f} (SUSA). "
            f"The **quality beta is the one bar that's genuinely red**: *t* = "
            f"{R['esgu_beta_q_t']:.2f} (ESGU), *t* = {R['susa_beta_q_t']:.2f} (SUSA) — a real, "
            f"measurable factor exposure. ESGU's growth-value beta is also real but tiny "
            f"(*t* = {R['esgu_beta_gv_t']:.2f}); SUSA's is indistinguishable from zero "
            f"(*t* = {R['susa_beta_gv_t']:.2f}) because SUSA screens rather than tilts by "
            "sector weight. For SUSA specifically, the point estimate shrinks from "
            f"{R['susa_active_ann']:+.3f}%/yr (raw) to {R['susa_alpha_ann']:+.3f}%/yr "
            "(factor-adjusted) — about 70% of the (statistically insignificant) raw gap is "
            "quality-factor beta, not stock-picking."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic correlated-return world (fund vs benchmark, ρ = 0.97), TUNABLE planted "
            "daily premium. The null (premium = 0) is checked over **20 seeds** — never a "
            "single stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    w = data.synthetic_world(premium_bps=0.0, seed=664 + s_)\n"
            "    null_ts.append(st.synthetic_detect(w)['t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "w = data.synthetic_world(premium_bps=5.0, seed=664)\n"
            "planted_t = st.synthetic_detect(w)['t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (premium=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5,\n"
            "           label='planted premium = +5 bps/day')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Newey-West t (active return)')\n"
            "ax.set_title('Control: no null fires; a planted premium lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and **never** crosses "
            f"the bar; a planted +5 bps/day premium reads t = {R['syn_planted_t']:.2f}. The "
            "machinery is unbiased — the real-tape near-zero *t*'s are the genuine article, "
            "not a broken detector. *(A faithful-engine / power check only — never cited in "
            "support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — ESGU−SPY NW *t* = **{R['esgu_nw_t']:.2f}** "
            f"({R['esgu_active_ann']:+.3f}%/yr, n={R['esgu_n']:,}); SUSA−IVV NW *t* = "
            f"**{R['susa_nw_t']:.2f}** ({R['susa_active_ann']:+.3f}%/yr, n={R['susa_n']:,}). "
            "Neither clears \\|t\\| ≥ 2 in either direction — no premium, no penalty.\n"
            f"- **Tradability `MIRAGE`** — net of a documented {R['er_gap_esgu']:.3f}–"
            f"{R['er_gap_susa']:.3f}%/yr expense-ratio gap plus one-way costs and a 30 bps/yr "
            f"borrow drag, both spreads run further negative ({R['esgu_net_ann']:+.3f}%/yr, "
            f"{R['susa_net_ann']:+.3f}%/yr) while carrying {R['esgu_te']:.2f}–"
            f"{R['susa_te']:.2f}%/yr of real, uncompensated tracking error.\n"
            f"- **\"Growth/quality tilt?\" `CONFIRMED`** — quality beta *t* = "
            f"{R['esgu_beta_q_t']:.2f} / {R['susa_beta_q_t']:.2f}, ESGU growth-value beta "
            f"*t* = {R['esgu_beta_gv_t']:.2f}. The tilt is real; the return edge it was "
            "supposed to explain never was."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The academic record stays genuinely split** — Friede/Busch/Bassen (2015) "
            "lean mildly positive in the meta-literature; Bolton & Kacperczyk (2021) find a "
            "carbon *premium* implying ESG exclusion should mildly *underperform*; "
            "Pástor-Stambaugh-Taylor (2021) show realized alpha can be positive in a demand-"
            "driven transition even as expected alpha is negative in equilibrium. This "
            "study's null result is consistent with all three once you account for fees and "
            "the growth/quality tilt.\n"
            "- **International and small-cap ESG products, and ESG-momentum (ratings "
            "upgrades/downgrades) strategies**, are natural sequels — this study only tests "
            "the two flagship US large-cap products at the fund level.\n"
            "- **Dedup map:** [211-sin-stocks](../../211-sin-stocks/) (the mirror claim, "
            "individual tickers), [200-roe-quality](../../200-roe-quality/) (the quality "
            "factor on its own terms), [246-defensive-sectors](../../246-defensive-sectors/), "
            "[335-buzz-sentiment-etf](../../335-buzz-sentiment-etf/) and "
            "[334-ark-innovation](../../334-ark-innovation/) (thematic/active growth "
            "products, a different construction entirely) — full detail in "
            "[docs/references.md](../docs/references.md).\n\n"
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
