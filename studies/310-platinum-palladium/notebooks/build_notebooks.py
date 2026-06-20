"""Generate the two narrative notebooks for Study 310 (Platinum-Palladium).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached daily
parquets under ../_cache/ if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader, online or off.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-05-31).
R = dict(
    # Data
    n_pl=4124, n_pa=4107, date_start="2010-01-04", date_end="2026-05-29",
    fp_pl="136a4cd1f31c", fp_pa="a53cb60090ad",
    ratio_mean=1.336, ratio_min=0.31, ratio_max=3.705,
    # Structural tests
    adf_p=0.0110, half_life=474.7, coint_p=0.7010,
    # Primary strategy (enter |z|>1.5, exit |z|<0.5, window=252)
    n_trades=26, avg_hold=87,
    strat_win=0.731, strat_mean=-101.1, strat_t=-0.32, strat_skew=-2.23,
    rand_win=0.500, rand_mean=-117.5, rand_t=-0.41,
    boot_lo=-704.7, boot_hi=430.9,
    # Cost sweep (net)
    net0=-101.1, net0_t=-0.32,
    net5=-106.1, net5_t=-0.34,
    net10=-111.1, net10_t=-0.36,
    net20=-121.1, net20_t=-0.39,
    # Buy and hold
    pl_total=23.7, pl_ann=1.5,
    pa_total=117.6, pa_ann=7.4,
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

from platinum_palladium import data, strategy as st

PL_TICKER = data.PLATINUM_TICKER   # PL=F
PA_TICKER = data.PALLADIUM_TICKER  # PA=F
AS_OF = pd.Timestamp("2026-05-31")  # drop partial June 2026

def _have_cache():
    return (os.path.exists(data._cache_path(PL_TICKER, data.DEFAULT_CACHE)) and
            os.path.exists(data._cache_path(PA_TICKER, data.DEFAULT_CACHE)))

def load_pair():
    \"\"\"Cache-first real tape, trimmed to AS_OF; None when offline.\"\"\"
    try:
        out = data.load_real(fetch=False)
        if out is None:
            return None
        plat, pall = out
        return plat[plat.index <= AS_OF], pall[pall.index <= AS_OF]
    except Exception:
        return None

HAVE_REAL = _have_cache()
print("real daily cache present:", HAVE_REAL)
"""

# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Platinum-Palladium -- does the autocatalyst-metals ratio snap back?\n"
            "### Two catalytic-converter metals, one famous ratio trade -- tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Ratio_mean--reverts%3F: Mixed](https://img.shields.io/badge/Ratio_mean--reverts%3F-Mixed-dab617?style=flat-square)\n\n"
            "Platinum and palladium are the two great *catalytic-converter* metals: both scrub "
            "pollution out of car exhaust, and for a while they were partly interchangeable. So "
            "the old trade says: when palladium gets expensive relative to platinum, sell "
            "palladium and buy platinum, because the ratio 'always comes back'. This notebook "
            "asks: **is the snap-back real, or did a one-off demand shock blow the trade up?**\n\n"
            "> **This is the plain-language layer.** Want the t-stats, the unit-root tests, and "
            "the cointegration analysis? That's **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn by "
            "the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            f"| Question | Answer |\n|---|---|\n"
            f"| Does the platinum/palladium ratio snap back? | **Not tradably.** Over 2010-2026 "
            f"it swung from {R['ratio_min']:.2f} to {R['ratio_max']:.2f} -- palladium went from "
            f"half platinum's price to over *three times* it -- driven by a one-off car-emissions "
            f"demand boom, not a restoring spring. |\n"
            f"| Did the reversion trade make money? | **No -- it lost.** The z-score strategy "
            f"earned **{R['strat_mean']:.0f} bps/trade** (negative!) at HAC *t* = {R['strat_t']:.2f}. "
            f"It even *won* {R['strat_win']*100:.0f}% of trades -- but the rare losers were "
            f"catastrophic. |\n"
            f"| Is the ratio 'stationary'? | **Technically yes, uselessly so.** A textbook unit-root "
            f"test rejects a random walk (p = {R['adf_p']:.3f}), but the reversion half-life is "
            f"**{R['half_life']:.0f} trading days (~22 months)** -- a geological timescale. |\n"
            f"| Could you trade it? | **No.** ~1.6 signals/year, 4-month holds, negative expectancy. "
            f"Meanwhile just holding palladium returned ~{R['pa_ann']:.0f}%/yr. |\n\n"
            "> The lesson: a high win-rate (73%!) can hide a *losing* strategy when the rare losses "
            "are huge -- and 'both metals are catalysts' was not the stable anchor the trade assumed."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim\n\n"
            "> *'Platinum and palladium are both platinum-group metals used in catalytic "
            "converters, and carmakers can substitute one for the other. So their price ratio "
            "is anchored: when palladium gets rich relative to platinum, short palladium and buy "
            "platinum -- the ratio always reverts to its historical mean, and the patient "
            "spread trader profits from the snap-back.'*\n\n"
            "It is a clean, intuitive relative-value story: two near-substitutes whose *relative* "
            "price should mean-revert even as both wander. The bet is that extremes in the ratio "
            "are temporary dislocations you can fade."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "If the claim held, this would be a low-turnover, market-neutral commodity spread -- "
            "long one metal, short the other -- with a built-in fundamental anchor (substitution). "
            "That's a desirable, diversifying return stream. The critical question is whether the "
            "'ratio always reverts' assertion is a statistical fact or a story that survived only "
            "until a genuine regime change rolled over it."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 -- How would we even know?\n\n"
            "Two questions before we look at returns:\n\n"
            "1. **Does the ratio actually mean-revert?** We test whether the PL/PA ratio is "
            "stationary (ADF test), whether platinum and palladium prices are cointegrated "
            "(Engle-Granger), and how fast any reversion is (OU half-life). If the half-life is "
            "two years, 'snap-back' is the wrong word.\n"
            "2. **Did the trade pay?** We run a z-score spread (enter at |z| > 1.5, exit at "
            "|z| < 0.5) and compare it to a **random-direction control** on the same dates, plus "
            "a buy-and-hold of each metal. PL=F and PA=F daily futures, 2010-2026, from Yahoo."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 -- The teardown -- let's actually look\n\n"
            "**First, the ratio itself.** Here is platinum / palladium over time -- watch it invert:"
        ),
        code(
            "pair = load_pair() if HAVE_REAL else None\n"
            "if pair is not None:\n"
            "    plat, pall = pair\n"
            "    ratio = st.compute_ratio(plat, pall)\n"
            "    z_ratio = st.zscore(ratio, 252)\n"
            "    adf_p = st.adf_pvalue(ratio); hl = st.half_life(ratio)\n"
            "    banner = 'REAL TAPE (PL=F / PA=F, Yahoo, 2010-2026)'\n"
            "else:\n"
            "    ratio = None; z_ratio = None\n"
            f"    adf_p = {R['adf_p']}; hl = {R['half_life']}\n"
            "    banner = 'OFFLINE -- frozen headline numbers from docs/results.md'\n"
            "print(banner)\n"
            "\n"
            "if ratio is not None:\n"
            "    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 7), sharex=True)\n"
            "    ax1.plot(ratio.index, ratio.values, c=AMBER, lw=1.1)\n"
            "    ax1.axhline(1.0, c=RED, ls='--', lw=1, label='ratio = 1 (equal price)')\n"
            "    ax1.axhline(ratio.mean(), c=GREY, ls=':', lw=1, label=f'mean {ratio.mean():.2f}')\n"
            "    ax1.set_ylabel('PL / PA ratio'); ax1.legend()\n"
            "    ax1.set_title('Platinum/palladium ratio -- palladium overtook platinum 2018-2022')\n"
            "    zc = z_ratio.dropna()\n"
            "    ax2.plot(zc.index, zc.values, c=GREY, lw=0.8)\n"
            "    ax2.axhline(1.5, c=RED, ls='--', lw=1); ax2.axhline(-1.5, c=GREEN, ls='--', lw=1)\n"
            "    ax2.axhline(0, c='k', lw=0.8)\n"
            "    ax2.set_ylabel('252-day rolling z-score'); ax2.set_title('Z-score of the ratio')\n"
            "    plt.tight_layout(); plt.show()\n"
            "print(f'ADF p-value: {adf_p:.4f}   OU half-life: {hl:.0f} trading days (~{hl/21:.0f} months)')"
        ),
        md(
            f"The ratio ran from **{R['ratio_min']:.2f}** (palladium ~3x platinum) to "
            f"**{R['ratio_max']:.2f}** (platinum ~3.7x palladium) -- a colossal swing. The ADF "
            f"test *does* reject a random walk (p = {R['adf_p']:.3f}), because over 16 years the "
            "ratio wandered far and partly came back. But the estimated reversion half-life is "
            f"**{R['half_life']:.0f} trading days (~22 months)**. That is not a snap-back; that's a "
            "slow structural drift. The 2018-2022 inversion was a one-off demand shock (tightening "
            "petrol-engine emissions rules), not a deviation from a spring."
        ),
        md("**Now: did fading the extremes actually make money?**"),
        code(
            "if pair is not None:\n"
            "    entries = st.zscore_entries(ratio, window=252, enter_z=1.5, exit_z=0.5)\n"
            "    ledger = st.run_trades(plat, pall, entries, cost_bps=0.0)\n"
            "    rand_led = st.run_trades(plat, pall, st.random_entries(entries, seed=310), cost_bps=0.0)\n"
            "    s = st.summarize(ledger, 'ret_gross'); r = st.summarize(rand_led, 'ret_gross')\n"
            "    strat_m, strat_w, strat_t = s['mean_bps'], s['win_rate']*100, s['tstat']\n"
            "    rand_m, rand_w, rand_t = r['mean_bps'], r['win_rate']*100, r['tstat']\n"
            "    n_tr = s['n_trades']\n"
            "else:\n"
            f"    strat_m, strat_w, strat_t = {R['strat_mean']}, {R['strat_win']*100:.1f}, {R['strat_t']}\n"
            f"    rand_m, rand_w, rand_t = {R['rand_mean']}, {R['rand_win']*100:.1f}, {R['rand_t']}\n"
            f"    n_tr = {R['n_trades']}\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars = ax.bar(['Z-score reversion\\n(fade the extreme)', 'Random direction\\n(a coin)'],\n"
            "              [strat_m, rand_m], color=[RED, GREY], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('gross profit per trade (bps)')\n"
            "ax.set_title(f'Strategy vs random: {n_tr} trades over 16 years (both LOSE)')\n"
            "for b, (w, t) in zip(bars, [(strat_w, strat_t), (rand_w, rand_t)]):\n"
            "    ax.annotate(f'win {w:.0f}%\\nt={t:+.2f}',\n"
            "                (b.get_x()+b.get_width()/2, min(b.get_height(), 0) - 12),\n"
            "                ha='center', va='top', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Strategy: {strat_m:+.0f} bps/trade, win {strat_w:.0f}%, t={strat_t:+.2f}  ({n_tr} trades)')"
        ),
        md(
            f"Look at the trap: the strategy **won {R['strat_win']*100:.0f}% of its trades** -- a "
            f"number a vendor backtest would put in bold. Yet its mean is **{R['strat_mean']:.0f} "
            f"bps/trade -- negative** -- and the HAC *t* is {R['strat_t']:.2f}, on the wrong side of "
            "zero. The handful of losing trades (the ones that bet on reversion right as the ratio "
            "kept inverting) were catastrophic. High win-rate, losing strategy. The random control "
            "lost too -- there was simply no edge to harvest."
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- NONE.** {R['strat_mean']:.0f} bps/trade, HAC *t* = {R['strat_t']:.2f} "
            f"on {R['n_trades']} trades -- negative and nowhere near the |t| >= 2 bar.\n"
            f"- **Tradability -- MIRAGE.** A {R['strat_win']*100:.0f}% win-rate hides a negative "
            "expectancy (skew -2.2). You'd have lost money running this.\n"
            f"- **Ratio mean-reverts? -- MIXED.** The unit-root test says 'stationary' "
            f"(p = {R['adf_p']:.3f}), but a {R['half_life']:.0f}-day half-life, a failed "
            f"cointegration test (p = {R['coint_p']:.2f}), and a one-off regime change mean "
            "there's nothing tradable here."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 -- Could you actually trade it?\n\n"
            "No -- and unusually, costs are not even the reason. With ~1.6 trades/year the cost "
            "drag is tiny; the gross trade is *already* a loser. Watch the cost sweep stay flat "
            "and negative:"
        ),
        code(
            "costs = [0, 5, 10, 20]\n"
            "if pair is not None:\n"
            "    net_means = [st.summarize(st.run_trades(plat, pall, entries, cost_bps=c),\n"
            "                              'ret_net')['mean_bps'] for c in costs]\n"
            "else:\n"
            f"    net_means = [{R['net0']}, {R['net5']}, {R['net10']}, {R['net20']}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net_means, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (bps/trade)')\n"
            "ax.set_title('Costs are not the killer -- the gross trade already loses')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('Buy and hold instead: platinum ~{R['pl_ann']:.0f}%%/yr, palladium ~{R['pa_ann']:.0f}%%/yr.')"
        ),
        md(
            f"The net mean stays negative across the whole cost range. And the opportunity cost is "
            f"brutal: simply *holding* palladium returned ~{R['pa_ann']:.0f}%/yr over 2010-2026 -- "
            "the very divergence the spread trade kept betting against."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **The positive control.** The companion notebook shows a synthetic pair with "
            "*planted* mean-reversion: the same engine harvests a large, significant edge there. "
            "The real-tape failure is a fact about platinum & palladium, not the method.\n"
            "- **Regime-aware reversion.** A trade that detected the 2018 regime break (e.g. a "
            "structural-break test, or a rolling cointegration check) might have stood aside -- "
            "fork it and try.\n"
            "- **Sibling ratio studies.** [Study 113 -- Gold-Silver-Ratio](../../113-gold-silver-ratio/) "
            "runs the same machinery on gold/silver; "
            "[Study 305 -- Gold-Oil-Ratio](../../305-gold-oil-ratio/) and "
            "[Study 306 -- Crack-Spread](../../306-crack-spread/) are other commodity spreads.\n\n"
            "*Think the ratio really mean-reverts? Fork this, add a structural-break filter or "
            "test a pre-2010 window, and show a spread that clears |t| >= 2 out of sample. That's the bar.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 -- FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Platinum-Palladium -- a quantitative teardown\n"
            "### Real futures tape * ADF + Engle-Granger * OU half-life * z-score spread * "
            "random control * HAC + block-bootstrap inference\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Ratio_mean--reverts%3F: Mixed](https://img.shields.io/badge/Ratio_mean--reverts%3F-Mixed-dab617?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We test whether the "
            "platinum/palladium ratio is stationary, whether the two metals are cointegrated, and "
            "whether a z-score spread earns a statistically robust edge over a random-direction "
            "control on 16 years of daily futures data.\n\n"
            f"> **Not investment advice.** Real data: PL=F and PA=F daily futures "
            f"{R['date_start']} to {R['date_end']} (n = {R['n_pl']:,} / {R['n_pa']:,} bars); offline "
            "core runs on a deterministic synthetic OU tape. Methods & sources in "
            "[`docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> **The `in plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Gross **{R['strat_mean']:.0f} bps/trade**, HAC "
            f"*t* = **{R['strat_t']:.2f}** on n = **{R['n_trades']} trades**; bootstrap 95% CI "
            f"[{R['boot_lo']:.0f}, {R['boot_hi']:.0f}] bps straddles zero. |\n"
            f"| **Tradability** | `MIRAGE` | win-rate {R['strat_win']*100:.0f}% but skew "
            f"{R['strat_skew']:.1f} -- a negative-expectancy, fat-left-tail profile. Costs only worsen it. |\n"
            f"| **Ratio mean-reverts?** | `MIXED` | ADF p = {R['adf_p']:.3f} (rejects unit root) "
            f"*but* OU half-life = {R['half_life']:.0f} days and Engle-Granger p = {R['coint_p']:.2f} "
            "(not cointegrated) -- stationary-on-paper, untradable-in-fact. |\n\n"
            "> *In plain words:* the only 'positive' test (ADF) is a full-sample artefact of a "
            "ratio that swung wildly and partly came back; everything that matters for trading -- "
            "speed, cointegration, the actual P&L -- says there is no edge."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $R_t = P^{\\text{PL}}_t / P^{\\text{PA}}_t$ be the platinum/palladium ratio. "
            "The claim asserts:\n\n"
            "- **H1 (stationarity).** $R_t$ is I(0): finite unconditional variance, returns to a "
            "long-run mean. Equivalently $\\log P^{\\text{PL}}_t - \\beta \\log P^{\\text{PA}}_t$ "
            "is stationary for some $\\beta$ (cointegration), justified by catalyst substitution.\n"
            "- **H2 (signal).** The conditional expected return of the spread "
            "$d_t \\cdot (r^{\\text{PA}}_{t+1} - r^{\\text{PL}}_{t+1})$ is positive when "
            "$d_t = \\text{sign}(z_t)$, measured against a random-direction control.\n"
            "- **H3 (tradable).** Any edge survives realistic round-trip futures costs.\n\n"
            "We find: H1 is *ambiguous* (ADF rejects a unit root, but the half-life is 22 months "
            "and the pair is not cointegrated); H2 is **rejected with the point estimate on the "
            "wrong side of zero**; H3 is moot."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "The platinum/palladium spread is a textbook 'fundamental pairs trade': two "
            "near-substitute autocatalyst metals whose relative price *should* be anchored. The "
            "interesting finding is *not* that it merely fails to clear a significance bar -- it is "
            "that the trade has a **negative gross mean** despite a 73% win-rate, because a "
            "substitution-driven *regime change* (palladium's 2018-2022 boom) is exactly the kind "
            "of event a naive ratio-reversion bet is structurally short."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 -- How we'd know -- the protocol\n\n"
            "- **Data.** PL=F and PA=F daily auto-adjusted closes, 2010-01-04 to 2026-05-29 "
            "(Yahoo). The PL/PA *ratio* largely cancels common front-month roll/level artefacts.\n"
            "- **Structural tests.** (a) ADF on the ratio (autolag AIC); (b) Engle-Granger "
            "cointegration on log(PL) vs log(PA); (c) OU half-life via AR(1) OLS.\n"
            "- **Entry.** Rolling 252-day z-score; enter the spread at |z| > 1.5, exit at |z| < 0.5.\n"
            "- **Control.** Identical entry/exit dates, direction drawn i.i.d. +/-1 (seeded).\n"
            "- **Inference.** Newey-West HAC t-stat *and* a circular block-bootstrap 95% CI on the "
            "mean per-trade return.\n"
            "- **Positive control.** Synthetic OU pair with tunable reversion speed, confirming the "
            "engine recovers a large edge when one is planted."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 -- The teardown"),
        md(
            "### 4a -- Structural tests: is there a mean-reverting ratio to exploit?\n\n"
            "The three tests disagree in a telling way -- and the disagreement is the point:"
        ),
        code(
            "pair = load_pair() if HAVE_REAL else None\n"
            "if pair is not None:\n"
            "    plat, pall = pair\n"
            "    ratio = st.compute_ratio(plat, pall)\n"
            "    adf_p = st.adf_pvalue(ratio); hl = st.half_life(ratio)\n"
            "    ci_p = st.engle_granger_pvalue(plat['close'], pall['close'])\n"
            "    print('REAL TAPE')\n"
            "else:\n"
            f"    adf_p={R['adf_p']}; hl={R['half_life']}; ci_p={R['coint_p']}\n"
            "    print('OFFLINE -- frozen headline numbers')\n"
            "\n"
            "print(f'ADF p-value on PL/PA ratio:          {adf_p:.4f}  (H0: unit root; p<0.05 = reject = stationary)')\n"
            "print(f'Ornstein-Uhlenbeck half-life:        {hl:.1f} trading days (~{hl/21:.0f} months)')\n"
            "print(f'Engle-Granger cointegration p-value: {ci_p:.4f}  (H0: no cointegration)')"
        ),
        md(
            f"> *In plain words:* the ADF *rejects* a unit root (p = {R['adf_p']:.3f}) -- so a "
            "textbook would call the ratio 'stationary'. But that is the signature of a series "
            "that wandered to an extreme and partly mean-reverted *once* over 16 years. The "
            f"half-life of {R['half_life']:.0f} days (~22 months) is far too slow to trade, and the "
            f"Engle-Granger test finds **no cointegration** (p = {R['coint_p']:.2f}) -- there is no "
            "stable long-run linear relationship pinning the two prices together. Stationary on "
            "paper, untradable in fact."
        ),
        md(
            "### 4b -- The z-score spread vs a random-direction control\n\n"
            "Run the strategy on the real tape and put it next to a coin-flip on identical entries:"
        ),
        code(
            "if pair is not None:\n"
            "    entries = st.zscore_entries(ratio, window=252, enter_z=1.5, exit_z=0.5)\n"
            "    ledger = st.run_trades(plat, pall, entries, cost_bps=0.0)\n"
            "    rand_led = st.run_trades(plat, pall, st.random_entries(entries, seed=310), cost_bps=0.0)\n"
            "    s = st.summarize(ledger, 'ret_gross'); r = st.summarize(rand_led, 'ret_gross')\n"
            "    lo, hi = st.block_bootstrap_ci(ledger, 'ret_gross')\n"
            "    sm, sw, st_v, sk = s['mean_bps'], s['win_rate']*100, s['tstat'], s['skew']\n"
            "    rm, rw, rt_v = r['mean_bps'], r['win_rate']*100, r['tstat']\n"
            "    n_tr = s['n_trades']; avg_h = ledger['trading_days'].mean()\n"
            "else:\n"
            f"    sm,sw,st_v,sk = {R['strat_mean']},{R['strat_win']*100:.1f},{R['strat_t']},{R['strat_skew']}\n"
            f"    rm,rw,rt_v = {R['rand_mean']},{R['rand_win']*100:.1f},{R['rand_t']}\n"
            f"    lo,hi = {R['boot_lo']},{R['boot_hi']}; n_tr={R['n_trades']}; avg_h={R['avg_hold']}\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "x=[0,1]; ys=[sm,rm]; ts=[st_v,rt_v]; ws=[sw,rw]\n"
            "ax.bar(x, ys, color=[RED, GREY], width=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for xi,yi,ti,wi in zip(x,ys,ts,ws):\n"
            "    ax.text(xi, yi-25, f't={ti:+.2f}\\nwin {wi:.0f}%', ha='center', va='top', fontsize=9)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'Z-score spread\\n(n={n_tr})','Random direction'])\n"
            "ax.set_ylabel('gross profit per trade (bps)')\n"
            "ax.set_title(f'z-score spread vs coin (n={n_tr}, avg hold ~{avg_h:.0f}d): both negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Strategy: {sm:+.1f} bps, t={st_v:+.2f}, win {sw:.0f}%, skew {sk:+.2f}')\n"
            "print(f'Bootstrap 95%% CI on the mean: [{lo:+.0f}, {hi:+.0f}] bps')"
        ),
        md(
            f"> *In plain words:* the spread earns a **negative** mean ({R['strat_mean']:.0f} bps), "
            f"HAC *t* = {R['strat_t']:.2f}, with a {R['strat_win']*100:.0f}% win-rate and skew "
            f"{R['strat_skew']:.1f}. The block-bootstrap CI is [{R['boot_lo']:.0f}, {R['boot_hi']:.0f}] "
            "bps -- it spans zero by hundreds of basis points. The high win-rate is the classic "
            "'collect-pennies' shape; the rare losers (reversion bets during the regime inversion) "
            "are the steamroller. There is no edge, in either direction the data can pin down."
        ),
        md(
            "### 4c -- Positive control: the engine works when reversion is real\n\n"
            "A synthetic pair with a *planted* OU mean-reversion speed confirms the z-score engine "
            "finds edges when they exist:"
        ),
        code(
            "speeds = [0.0, 0.05, 0.10, 0.15, 0.20]\n"
            "strat_ts, rand_ts = [], []\n"
            "for sp in speeds:\n"
            "    pl, pa, _ = data.synthetic_daily(n_days=3000, mean_rev_speed=sp, seed=310)\n"
            "    rat = st.compute_ratio(pl, pa)\n"
            "    ent = st.zscore_entries(rat, window=252, enter_z=1.5, exit_z=0.5)\n"
            "    if len(ent) < 2:\n"
            "        strat_ts.append(float('nan')); rand_ts.append(float('nan')); continue\n"
            "    led = st.run_trades(pl, pa, ent, cost_bps=0.0)\n"
            "    rnd = st.run_trades(pl, pa, st.random_entries(ent, seed=42), cost_bps=0.0)\n"
            "    strat_ts.append(st.summarize(led, 'ret_gross')['tstat'])\n"
            "    rand_ts.append(st.summarize(rnd, 'ret_gross')['tstat'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(speeds, strat_ts, 'o-', c=GREEN, lw=2, label='z-score spread')\n"
            "ax.plot(speeds, rand_ts, 's--', c=GREY, lw=1.5, label='random control')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1, label='|t|=2 bar'); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.set_xlabel('planted OU mean-reversion speed'); ax.set_ylabel('HAC t-stat')\n"
            "ax.set_title('The engine is faithful: t-stat explodes once reversion is planted')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('speed=0 (random walk): too few entries / t~0.')\n"
            "print('speed>=0.05: the spread clears the bar with huge t-stats.')"
        ),
        md(
            "> *In plain words:* the machine is not broken. When the ratio truly mean-reverts "
            "(speeds >= 0.05) the spread earns enormous, significant t-stats while the random "
            f"control stays near zero. The real-tape result ({R['strat_mean']:.0f} bps, "
            f"t = {R['strat_t']:.2f}) is therefore a statement about platinum and palladium: the "
            "real ratio's 'reversion' was a one-off regime round-trip, not a tradable restoring force."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `NONE`** -- gross {R['strat_mean']:.0f} bps/trade, HAC *t* {R['strat_t']:.2f} "
            f"on {R['n_trades']} trades; bootstrap CI [{R['boot_lo']:.0f}, {R['boot_hi']:.0f}] bps. "
            "Negative point estimate, nowhere near the bar.\n"
            f"- **Tradability `MIRAGE`** -- win-rate {R['strat_win']*100:.0f}%, skew {R['strat_skew']:.1f}; "
            "negative expectancy that costs only deepen.\n"
            f"- **Ratio mean-reverts? `MIXED`** -- ADF p = {R['adf_p']:.3f} (rejects unit root) but "
            f"half-life {R['half_life']:.0f} days and Engle-Granger p = {R['coint_p']:.2f}. Stationary "
            "on paper, no tradable reversion in fact."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 -- Could you trade it? -- a losing gross trade, before costs\n\n"
            "Unusually, costs are not the executioner -- the *gross* trade already loses. The cost "
            "sweep just deepens the hole, and the t-stat never approaches the bar:"
        ),
        code(
            "costs = [0, 5, 10, 20]\n"
            "if pair is not None:\n"
            "    net_m = [st.summarize(st.run_trades(plat, pall, entries, cost_bps=c),'ret_net')['mean_bps'] for c in costs]\n"
            "    net_t = [st.summarize(st.run_trades(plat, pall, entries, cost_bps=c),'ret_net')['tstat'] for c in costs]\n"
            "else:\n"
            f"    net_m = [{R['net0']}, {R['net5']}, {R['net10']}, {R['net20']}]\n"
            f"    net_t = [{R['net0_t']}, {R['net5_t']}, {R['net10_t']}, {R['net20_t']}]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, net_m, 'o-', c=RED, lw=2, label='net mean (bps/trade)')\n"
            "ax2 = ax.twinx(); ax2.plot(costs, net_t, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(2, ls=':', c=GREY); ax2.axhline(-2, ls=':', c=GREY)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (bps/trade)', color=RED)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('Costs deepen a hole that already exists at zero cost')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('Buy-and-hold over 2010-2026: PL=F ~{R['pl_ann']:.0f}%%/yr, PA=F ~{R['pa_ann']:.0f}%%/yr.')"
        ),
        md(
            "> *In plain words:* most failed strategies die at the cost line. This one was dead at "
            "the gross line -- the reversion premise was simply wrong for this pair over this "
            f"window. And the opportunity cost is steep: palladium alone returned ~{R['pa_ann']:.0f}%/yr, "
            "the exact divergence the spread kept fading."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 -- Going further -- extensions and related desk work\n\n"
            "- **Structural-break awareness.** A Chow / Bai-Perron break test or rolling "
            "cointegration check could flag the 2018 regime change and stand the trade down. The "
            "naive ratio-reversion bet is structurally short exactly that event.\n"
            "- **Pre-2010 / spot data.** A longer window (spot PL/PA back to the 1990s) would test "
            "whether the 'always reverts' folklore was ever true or just pre-Dieselgate calm.\n"
            "- **[Study 113 -- Gold-Silver-Ratio](../../113-gold-silver-ratio/)**: the same engine "
            "on gold/silver -- a sibling Weak/Fragile result for comparison.\n"
            "- **[Study 305 -- Gold-Oil-Ratio](../../305-gold-oil-ratio/)** and "
            "**[Study 306 -- Crack-Spread](../../306-crack-spread/)**: other commodity ratio/spread "
            "signals in the Carry-curves-commodities family.\n\n"
            "*Fork this, add a structural-break filter or a pre-2010 window, and show a PGM spread "
            "that clears |t| >= 2 out of sample. That's the bar.*"
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
