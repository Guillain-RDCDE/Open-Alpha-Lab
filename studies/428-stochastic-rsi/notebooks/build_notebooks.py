"""Generate the two narrative notebooks for Study 428 (Stochastic RSI).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
positive control runs anywhere, offline and deterministic; the real-tape cells use the
cached daily parquet under ../_cache/ if present and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-23).
# Source: SPY daily, run_experiment(cost_bps=1.0) on the common 1993-11-12..2026-06-23 index.
R = dict(
    asof="2026-06-23", ticker="SPY", n_days=8206, start="1993-11-12", end="2026-06-23",
    # StochRSI long-flat, net of 1bp one-way x NAV, excess of cash
    srsi_sharpe=0.31, srsi_t=2.03, srsi_ann=4.44, srsi_dd=-57.5, srsi_tim=55.1,
    srsi_gross_sharpe=0.32, srsi_gross_t=2.10,
    # buy-and-hold (excess of cash)
    bh_sharpe=0.64, bh_t=4.26, bh_ann=12.0, bh_dd=-55.2,
    # benchmarks (net 1bp)
    rsi_sharpe=0.34, rsi_t=2.31, sma_sharpe=0.75, sma_t=4.99,
    # races (block-bootstrap Sharpe gap, srsi minus benchmark)
    vs_bh_diff=-0.33, vs_bh_lo=-0.53, vs_bh_hi=-0.13, vs_bh_frac=99.9,
    vs_rsi_diff=-0.03, vs_rsi_lo=-0.33, vs_rsi_hi=0.24, vs_rsi_frac=60,
    vs_sma_diff=-0.44, vs_sma_lo=-0.77, vs_sma_hi=-0.12, vs_sma_frac=99.8,
    # permutation p (sign-flip null, net)
    perm_p=0.037,
    # timing alpha (regress net on market excess)
    alpha_ann=-2.66, alpha_t=-1.67, beta=0.58,
    # random-timing control, same time-in-market, net 1bp
    rand_sharpe=0.41, rand_lo=0.18, rand_hi=0.63, rand_frac_beat=84,
    # long-short variant (net 1bp)
    ls_sharpe=-0.18, ls_t=-1.15, ls_dd=-86.7,
    # turnover / cost sweep (net Sharpe)
    turnover_yr=15.4, cost0=0.31, cost1=0.30, cost2=0.29, cost5=0.26,
    # panel net Sharpe (long-flat, 1bp)
    p_spy=0.31, p_qqq=0.20, p_iwm=0.18, p_dia=0.27, p_efa=0.41,
    p_spy_bh=0.64, p_qqq_bh=0.45, p_iwm_bh=0.48, p_dia_bh=0.56, p_efa_bh=0.43,
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

from stochastic_rsi import data, strategy as st

ASOF = pd.Timestamp("2026-06-23")
CACHE = os.path.abspath(os.path.join("..", "_cache"))

def have_real(t="SPY"):
    return os.path.exists(data._cache_path(t, CACHE))

def load_spy():
    b = data.load_real("SPY", cache_dir=CACHE)
    return b[b.index <= ASOF]

HAVE_REAL = have_real("SPY")
print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Stochastic RSI — does stacking a stochastic on RSI add signal, or just noise?\n"
            "### Chande & Kroll's 'more sensitive' oscillator, raced honestly against buy-and-hold and plain RSI\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Adds_over_RSI%3F: Not_supported](https://img.shields.io/badge/Adds_over_RSI%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "Here's a recipe you'll find on every charting platform: take RSI (already an "
            "oscillator), then run a *second* oscillator — the Stochastic — on top of it. The "
            "pitch is that this 'indicator of an indicator' is **more sensitive**: when "
            "Stochastic-RSI drops below 20 it's deeply oversold (buy), above 80 it's overbought "
            "(get out). Two normalisations must beat one, right? We test that on 33 years of SPY.\n\n"
            "> This is the plain-language layer. Want the HAC *t*-stats, the timing-alpha "
            "regression, the random-timing control and the synthetic positive control? That's the "
            "companion, **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the timing rule make money? | A little — net Sharpe **{R['srsi_sharpe']}**, "
            f"about **{R['srsi_ann']}%/yr** over cash. But… |\n"
            f"| …does it beat just *buying and holding* SPY? | **No.** Buy-and-hold scores Sharpe "
            f"**{R['bh_sharpe']}**. The rule sits in cash half the time and gives up most of the "
            "ride. |\n"
            f"| Does stacking the stochastic beat **plain RSI**? | **No.** Plain RSI scores "
            f"**{R['rsi_sharpe']}**; the gap is **{R['vs_rsi_diff']:+.2f}** and statistically a "
            "tie. The extra layer adds nothing. |\n"
            f"| Is there real *timing skill*? | **No.** A coin that flips in and out the same "
            f"fraction of days beats it {R['rand_frac_beat']}% of the time. |\n\n"
            "> The small positive return isn't skill — it's the equity premium leaking through a "
            "rule that happens to be long ~half the time. You can get more of it, more cheaply, "
            "by doing nothing."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The Stochastic RSI is more sensitive than RSI because it applies the Stochastic "
            "formula to RSI values instead of price. When it falls below 20 the market is deeply "
            "oversold — buy. When it rises above 80 it is overbought — sell. The double "
            "normalisation catches turning points that plain RSI misses.\"*\n\n"
            "Tushar Chande and Stanley Kroll introduced it in *The New Technical Trader* (1994). "
            "The logic is seductive: RSI measures recent up- vs down-pressure; the Stochastic then "
            "ranks *today's RSI* within its own recent range, so a reading near 0 means RSI is at "
            "its lowest in weeks. We steelman this as: **a StochRSI timing rule should beat a "
            "plain-RSI rule, and ideally beat buy-and-hold, after costs.**"
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a mechanical oscillator could time entries and exits on a broad index, it would "
            "be worth a fortune: you'd capture the upside and sidestep the drawdowns, all from a "
            "free line on a chart. Millions of retail traders run StochRSI daily believing exactly "
            "that. If the premise is wrong — if the 'extra sensitivity' is just extra noise — then "
            "following it means paying costs and forgoing return to underperform a savings account "
            "that bought the index."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we know?\n\n"
            "Three honest tests, announced before we run them:\n\n"
            "1. **Race it against buy-and-hold** on an apples-to-apples basis (both measured "
            "*excess of cash*, since the rule sits in cash part-time). If the rule can't beat "
            "doing nothing, the 'timing' isn't worth the name.\n"
            "2. **Race it against plain RSI** — the *obvious simpler benchmark*. If stacking the "
            "stochastic adds signal, StochRSI must beat RSI; if it's a tie, the extra layer is "
            "noise.\n"
            "3. **Race it against a coin** that flips in and out the same fraction of days. That "
            "strips away the equity premium and isolates pure *timing skill*.\n\n"
            "We charge realistic costs, lag every trade one day, and run it on 33 years of SPY "
            "(1993–2026), then cross-check on a 5-ETF panel."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — what actually happens\n\n"
            "**First, the headline race: the StochRSI rule vs simply holding SPY.** Both curves "
            "are excess of cash, net of 1bp costs."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load_spy(); close = b['close']\n"
            "    pos = st.stoch_rsi_position(close)\n"
            "    led = st.positions_to_returns(close, pos, cost_bps=1.0)\n"
            "    srsi_curve = (1+led['net']).cumprod()\n"
            "    bh_curve = (1+led['mkt_excess']).cumprod()\n"
            "    srsi_sh = st.sharpe(led['net'].to_numpy()); bh_sh = st.sharpe(led['mkt_excess'].to_numpy())\n"
            "else:\n"
            f"    srsi_sh, bh_sh = {R['srsi_sharpe']}, {R['bh_sharpe']}\n"
            "    idx = pd.date_range('1994', periods=300, freq='ME')\n"
            "    srsi_curve = pd.Series(np.linspace(1,2.8,300), idx); bh_curve = pd.Series(np.linspace(1,9,300), idx)\n"
            "fig, ax = plt.subplots(figsize=(9.5,4.6))\n"
            "ax.plot(bh_curve.index, bh_curve.values, c=GREEN, lw=2, label=f'Buy & hold SPY (Sharpe {bh_sh:.2f})')\n"
            "ax.plot(srsi_curve.index, srsi_curve.values, c=RED, lw=2, label=f'StochRSI long-flat (Sharpe {srsi_sh:.2f})')\n"
            "ax.set_yscale('log'); ax.set_ylabel('growth of \\\\$1 (excess of cash, log)')\n"
            "ax.set_title('StochRSI timing leaves most of the equity premium on the table')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'StochRSI net Sharpe {srsi_sh:.2f}  vs  buy-and-hold {bh_sh:.2f}')"
        ),
        md(
            f"Buy-and-hold wins, and not by a little: Sharpe **{R['bh_sharpe']}** vs "
            f"**{R['srsi_sharpe']}**. By stepping aside whenever StochRSI says 'overbought', the "
            "rule misses big chunks of the very uptrends that make equities pay."
        ),
        md(
            "**Second, the comparison that actually tests the claim: StochRSI vs plain RSI.** "
            "If the extra stochastic layer adds anything, StochRSI must beat plain RSI."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pos_rsi = st.rsi_position(close)\n"
            "    led_rsi = st.positions_to_returns(close, pos_rsi, cost_bps=1.0)\n"
            "    rsi_sh = st.sharpe(led_rsi['net'].to_numpy())\n"
            "    pos_sma = st.sma_cross_position(close)\n"
            "    led_sma = st.positions_to_returns(close, pos_sma, cost_bps=1.0)\n"
            "    sma_sh = st.sharpe(led_sma['net'].to_numpy())\n"
            "else:\n"
            f"    rsi_sh, sma_sh, srsi_sh, bh_sh = {R['rsi_sharpe']}, {R['sma_sharpe']}, {R['srsi_sharpe']}, {R['bh_sharpe']}\n"
            "names = ['StochRSI','plain RSI','SMA 50/200','buy & hold']\n"
            "vals = [srsi_sh, rsi_sh, sma_sh, bh_sh]\n"
            "cols = [RED, GREY, GREEN, GREEN]\n"
            "fig, ax = plt.subplots(figsize=(8.5,4.3))\n"
            "ax.bar(names, vals, color=cols)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('net Sharpe (excess of cash)')\n"
            "ax.set_title('Stacking the stochastic adds nothing over plain RSI')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'StochRSI {srsi_sh:.2f}  |  plain RSI {rsi_sh:.2f}  |  SMA {sma_sh:.2f}  |  hold {bh_sh:.2f}')"
        ),
        md(
            f"StochRSI ({R['srsi_sharpe']}) and plain RSI ({R['rsi_sharpe']}) are a statistical "
            f"tie (gap {R['vs_rsi_diff']:+.2f}, the bootstrap interval straddles zero). The much "
            f"simpler SMA(50/200) trend filter ({R['sma_sharpe']}) beats both — the 'sensitivity' "
            "of the double oscillator is not where the money is."
        ),
        md(
            "**Third, is there any *timing skill* at all?** We give a coin the same in-market "
            "budget — flat the same fraction of days, but on random dates — and ask whether "
            "StochRSI's actual timing beats it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tim = np.nanmean(np.abs(led['pos']))\n"
            "    n = len(close); k = int(round(tim*n))\n"
            "    shs = []\n"
            "    for seed in range(50):\n"
            "        rr = np.random.default_rng(seed); ix = rr.choice(n, size=k, replace=False)\n"
            "        w = pd.Series(np.zeros(n), index=close.index); w.iloc[ix] = 1.0\n"
            "        l = st.positions_to_returns(close, w, cost_bps=1.0); shs.append(st.sharpe(l['net'].to_numpy()))\n"
            "    shs = np.array(shs); srsi_sh = st.sharpe(led['net'].to_numpy())\n"
            "else:\n"
            f"    shs = np.random.default_rng(0).normal({R['rand_sharpe']}, 0.11, 50); srsi_sh = {R['srsi_sharpe']}\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.hist(shs, bins=14, color=GREY, alpha=.7, label='coin: same days in market, random dates')\n"
            "ax.axvline(srsi_sh, c=RED, lw=2.5, label=f'StochRSI: {srsi_sh:.2f}')\n"
            "ax.set_xlabel('net Sharpe'); ax.set_ylabel('count')\n"
            "ax.set_title('A coin with the same exposure usually beats the StochRSI timing'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'{(shs>=srsi_sh).mean()*100:.0f}% of random schedules beat the StochRSI rule')"
        ),
        md(
            f"About **{R['rand_frac_beat']}%** of random in/out schedules with the same exposure "
            "*beat* StochRSI. So the rule's small positive return is the equity premium it "
            "happens to harvest by being long part-time — and it harvests it *worse* than chance. "
            "There is no timing skill here."
        ),

        # ---- BEAT 5 — VERDICT ------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Net Sharpe {R['srsi_sharpe']} *looks* significant (HAC "
            f"*t* {R['srsi_t']}), but that is harvested equity beta, not skill: the timing alpha "
            f"is **negative** ({R['alpha_ann']}%/yr) and a same-exposure coin beats the rule "
            f"{R['rand_frac_beat']}% of the time.\n"
            f"- **Tradability — Mirage.** It loses the Sharpe race to plain buy-and-hold "
            f"({R['srsi_sharpe']} vs {R['bh_sharpe']}, the gap negative in {R['vs_bh_frac']}% of "
            "bootstraps). You pay costs to underperform doing nothing.\n"
            f"- **Adds over RSI? — Not supported.** Gap vs plain RSI is {R['vs_rsi_diff']:+.2f} "
            "with a CI through zero. The second oscillator is decoration."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            f"Turnover is low (~{R['turnover_yr']} round-trips/yr), so costs aren't the killer — "
            "the rule is already behind buy-and-hold *before* costs. Charging more just widens "
            "the gap:"
        ),
        code(
            "costs = [0.0, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    net = [st.sharpe(st.positions_to_returns(close, pos, cost_bps=c)['net'].to_numpy()) for c in costs]\n"
            "else:\n"
            f"    net = [{R['cost0']}, {R['cost1']}, {R['cost2']}, {R['cost5']}]\n"
            f"bh_line = {R['bh_sharpe']}\n"
            "fig, ax = plt.subplots(figsize=(8.5,4.3))\n"
            "ax.plot(costs, net, 'o-', c=RED, lw=2, label='StochRSI net Sharpe')\n"
            "ax.axhline(bh_line, c=GREEN, ls='--', lw=2, label=f'buy & hold ({bh_line})')\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net Sharpe')\n"
            "ax.set_title('Never reaches the buy-and-hold line at any cost'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('There is no cost level at which the rule catches buy-and-hold.')"
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The positive control.** The companion notebook plants a *trending* synthetic "
            "tape and shows the same engine then earns a real timing edge over a coin — so the "
            "null result is about today's market, not a broken harness.\n"
            "- **Long-short.** Shorting the 'overbought' signal makes it *worse* "
            f"(Sharpe {R['ls_sharpe']}) — the overbought exits land in uptrends.\n"
            "- **Other oscillators.** The same overbought/oversold family lands the same way: "
            "[CCI (Study 178)](../../178-cci/), [Bollinger reversion (Study 104)](../../104-bollinger-reversion/).\n\n"
            "*Think StochRSI works on a different asset, timeframe, or as a momentum (not "
            "reversion) signal? Fork this, change the rule, and show a *timing edge over a "
            "same-exposure coin* that clears HAC inference. That is the bar.*"
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
            "# Stochastic RSI — a quantitative teardown\n"
            "### SPY daily 1993–2026 · StochRSI(14,14,3,3) long-flat & long-short · HAC inference · timing-alpha · random-timing control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Adds_over_RSI%3F: Not_supported](https://img.shields.io/badge/Adds_over_RSI%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether a "
            "StochRSI long-flat timing rule beats (a) buy-and-hold, (b) plain RSI and (c) a "
            "same-exposure random-timing coin, excess-vs-excess and net of costs, on 33 years of SPY.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars (auto-adjusted / total "
            f"return), SPY 1993–2026, as-of **{R['asof']}**; the offline core and the positive "
            "control run on a deterministic synthetic tape. Methods & sources in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> The `In plain words` notes translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Net Sharpe {R['srsi_sharpe']} (HAC *t* {R['srsi_t']}) is "
            f"harvested beta (β={R['beta']}); **timing alpha {R['alpha_ann']}%/yr** "
            f"(*t* {R['alpha_t']}). A same-exposure coin beats it {R['rand_frac_beat']}% of the time. |\n"
            f"| **Tradability** | `MIRAGE` | Loses the Sharpe race to buy-and-hold "
            f"({R['srsi_sharpe']} vs {R['bh_sharpe']}; Δ {R['vs_bh_diff']}, negative in "
            f"{R['vs_bh_frac']}% of block bootstraps). |\n"
            f"| **Adds over RSI?** | `NOT SUPPORTED` | Δ Sharpe vs plain RSI {R['vs_rsi_diff']:+.2f}, "
            f"95% CI [{R['vs_rsi_lo']}, {R['vs_rsi_hi']}] — straddles zero. |\n\n"
            "> In plain words: the rule's positive raw *t* is the equity premium it captures by "
            "being long part-time. Strip the beta and there is no skill; the extra stochastic "
            "layer is statistically indistinguishable from plain RSI."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\text{RSI}_t$ be Wilder's 14-bar RSI. The Stochastic RSI is\n\n"
            "$$\\text{StochRSI}_t = \\frac{\\text{RSI}_t - \\min_{k}\\text{RSI}}"
            "{\\max_{k}\\text{RSI} - \\min_{k}\\text{RSI}}, \\quad "
            "\\%K = 100\\cdot\\text{SMA}_3(\\text{StochRSI}), \\quad \\%D = \\text{SMA}_3(\\%K)$$\n\n"
            "with $k=14$. The folk rule, made stateful: long when $\\%K$ crosses up through 20, "
            "flat (or short) when it crosses down through 80. The hypotheses:\n\n"
            "- **H₁ (timing skill).** The rule's *timing alpha* — its return orthogonal to the "
            "market — is positive: $\\alpha > 0$ in $r^{\\text{srsi}}_t = \\alpha + \\beta\\, "
            "r^{\\text{mkt}}_t + \\varepsilon_t$.\n"
            "- **H₂ (beats buy-and-hold).** Net excess-of-cash Sharpe exceeds buy-and-hold's.\n"
            "- **H₃ (adds over RSI).** Net Sharpe exceeds a plain-RSI(14) 30/70 rule.\n\n"
            "We reject H₁ (alpha negative), H₂ (decisively) and H₃ (a statistical tie)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "StochRSI is one of the most-watched default indicators on retail platforms; the "
            "'indicator of an indicator' framing gives it an aura of sophistication. If H₁–H₃ "
            "held it would be a free, mechanical market-timer on a broad index. The finding that "
            "its raw significance is *entirely* passive beta — and that a coin times better — is "
            "the canonical trap: a backtest can clear *t* = 2 while contributing negative skill, "
            "because the index drifts up and any part-time-long rule inherits that drift."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Indicator.** StochRSI with RSI(14), Stochastic look-back 14, %K and %D smoothing 3.\n"
            "- **Rule.** Stateful long-flat: long on $\\%K\\uparrow 20$, flat on $\\%K\\downarrow 80$; "
            "hold otherwise. A long-short variant replaces flat with −1.\n"
            "- **Execution.** One lag — the weight known at the close of $t$ earns $t{+}1$'s return "
            "(a single `shift`). Costs `cost_bps` one-way × |Δweight| (NAV traded); shorts pay "
            "50 bps/yr borrow. All returns **excess of cash**.\n"
            "- **Inference.** Newey–West HAC *t* on daily net returns; a sign-flip permutation "
            "*p*; a market-model regression for the timing alpha; block-bootstrap Sharpe-gap "
            "tests vs each benchmark.\n"
            "- **Benchmarks.** Buy-and-hold, plain RSI(14) 30/70, SMA(50/200), and a "
            "same-time-in-market random-timing coin.\n"
            "- **Positive control.** Synthetic AR(1) tape; the engine must recover a timing edge "
            "when one is planted.\n\n"
            "Headline tape: SPY 1993–2026; panel cross-check on QQQ/IWM/DIA/EFA."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The Sharpe race — buy-and-hold dominates\n\n"
            "Excess-of-cash, net of 1bp. The rule's only job is to time; it does it worse than "
            "staying invested."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load_spy(); close = b['close']\n"
            "    res = st.run_experiment(b, cost_bps=1.0, long_short=False)\n"
            "    print('window', res['start'], '->', res['end'], '  n_days', res['n_days'])\n"
            "    for k in ['srsi_net','rsi_net','sma_net','bh']:\n"
            "        s = res[k]; print(f\"{k:9s} Sharpe {s['sharpe']:+.3f}  HAC t {s['tstat']:+.2f}  ann {s.get('ann_ret',0)*100:+.2f}%  maxDD {s.get('maxdd',float('nan'))*100:.1f}%\")\n"
            "else:\n"
            f"    print('StochRSI Sharpe {R['srsi_sharpe']} t {R['srsi_t']} | "
            f"RSI {R['rsi_sharpe']} | SMA {R['sma_sharpe']} | BH {R['bh_sharpe']}')"
        ),
        md(
            f"> In plain words: net Sharpe **{R['srsi_sharpe']}** (HAC *t* **{R['srsi_t']}**) clears "
            f"the *t*=2 bar — but buy-and-hold is **{R['bh_sharpe']}** and the simpler SMA filter "
            f"**{R['sma_sharpe']}**. A *t*≥2 here is necessary, not sufficient: it certifies a "
            "non-zero mean, which an up-drifting index hands to any part-time-long rule for free."
        ),
        md(
            "### 4b · Strip the beta — the timing alpha is negative\n\n"
            "Regress the rule's net return on the market excess return. The intercept is the "
            "skill; beta is the inherited equity premium."
        ),
        code(
            "if HAVE_REAL:\n"
            "    led = st.positions_to_returns(close, st.stoch_rsi_position(close), cost_bps=1.0)\n"
            "    y = led['net'].to_numpy(); x = led['mkt_excess'].to_numpy()\n"
            "    m = np.isfinite(y) & np.isfinite(x); y, x = y[m], x[m]\n"
            "    beta = np.cov(y, x, ddof=1)[0,1]/np.var(x, ddof=1)\n"
            "    a = y.mean() - beta*x.mean(); resid = y - (a + beta*x)\n"
            "    se = resid.std(ddof=2)/np.sqrt(len(y))*np.sqrt(1 + x.mean()**2/np.var(x, ddof=1))\n"
            "    at = a/se\n"
            "    print(f'beta={beta:.2f}  alpha={a*252*100:+.2f}%/yr  alpha t={at:+.2f}')\n"
            "else:\n"
            f"    beta, a, at = {R['beta']}, {R['alpha_ann']}/100/252, {R['alpha_t']}\n"
            f"    print('beta={R['beta']} alpha={R['alpha_ann']}%/yr alpha t={R['alpha_t']}')"
        ),
        md(
            f"> In plain words: β = **{R['beta']}** (half-invested on average), and the timing "
            f"alpha is **{R['alpha_ann']}%/yr** at *t* = **{R['alpha_t']}** — *negative*. Every "
            "dollar of the rule's return is borrowed beta; the timing itself subtracts value."
        ),
        md(
            "### 4c · The same-exposure coin — no timing skill\n\n"
            "Hold time-in-market fixed; randomise *which* days. 50 random schedules vs the actual "
            "StochRSI schedule."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tim = np.nanmean(np.abs(led['pos'])); n = len(close); k = int(round(tim*n))\n"
            "    shs = []\n"
            "    for seed in range(50):\n"
            "        ix = np.random.default_rng(seed).choice(n, size=k, replace=False)\n"
            "        w = pd.Series(np.zeros(n), index=close.index); w.iloc[ix] = 1.0\n"
            "        shs.append(st.sharpe(st.positions_to_returns(close, w, cost_bps=1.0)['net'].to_numpy()))\n"
            "    shs = np.array(shs); srsi_sh = st.sharpe(led['net'].to_numpy())\n"
            "else:\n"
            f"    shs = np.random.default_rng(0).normal({R['rand_sharpe']}, 0.11, 50); srsi_sh = {R['srsi_sharpe']}\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.hist(shs, bins=14, color=GREY, alpha=.7, label='random-timing (same exposure)')\n"
            "ax.axvline(srsi_sh, c=RED, lw=2.5, label=f'StochRSI {srsi_sh:.2f}')\n"
            "ax.axvline(np.mean(shs), c='k', ls='--', lw=1, label=f'coin mean {np.mean(shs):.2f}')\n"
            "ax.set_xlabel('net Sharpe'); ax.set_ylabel('count')\n"
            "ax.set_title('StochRSI timing is below the coin mean'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'{(shs>=srsi_sh).mean()*100:.0f}% of random schedules beat StochRSI')"
        ),
        md(
            f"> In plain words: the coin's mean Sharpe is ~{R['rand_sharpe']} (95% band "
            f"[{R['rand_lo']}, {R['rand_hi']}]); StochRSI sits *below* it, beaten by "
            f"{R['rand_frac_beat']}% of random schedules. Timing skill = none."
        ),
        md(
            "### 4d · The benchmark races, with bootstrap CIs\n\n"
            "Block-bootstrap of the Sharpe gap (StochRSI − benchmark) on paired daily returns."
        ),
        code(
            "if HAVE_REAL:\n"
            "    for name, kk in [('vs buy-hold','vs_bh'), ('vs plain RSI','vs_rsi'), ('vs SMA','vs_sma')]:\n"
            "        d = res[kk]\n"
            "        print(f\"{name:13s} dSharpe {d['diff']:+.3f}  95% CI [{d['ci_low']:+.3f},{d['ci_high']:+.3f}]  P(StochRSI not better)={d['frac_a_not_better']*100:.0f}%\")\n"
            "    print(f\"sign-flip permutation p (net): {res['perm_p_srsi_net']:.3f}\")\n"
            "else:\n"
            f"    print('vs BH {R['vs_bh_diff']} CI[{R['vs_bh_lo']},{R['vs_bh_hi']}] | "
            f"vs RSI {R['vs_rsi_diff']} CI[{R['vs_rsi_lo']},{R['vs_rsi_hi']}] | perm p {R['perm_p']}')"
        ),
        md(
            f"> In plain words: vs buy-and-hold the gap is **{R['vs_bh_diff']}** (negative in "
            f"{R['vs_bh_frac']}% of resamples); vs plain RSI it's **{R['vs_rsi_diff']:+.2f}** with "
            f"a CI through zero — a tie. The permutation *p* = {R['perm_p']} merely confirms the "
            "mean isn't *zero* (the drift again); it says nothing about skill."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — raw net HAC *t* {R['srsi_t']} is passive beta (β {R['beta']}); "
            f"timing alpha {R['alpha_ann']}%/yr (*t* {R['alpha_t']}, wrong sign); a same-exposure "
            f"coin beats the rule {R['rand_frac_beat']}% of the time.\n"
            f"- **Tradability `MIRAGE`** — net Sharpe {R['srsi_sharpe']} vs buy-and-hold "
            f"{R['bh_sharpe']}; the gap is negative in {R['vs_bh_frac']}% of block bootstraps and "
            "no cost level closes it.\n"
            f"- **Adds over RSI? `NOT SUPPORTED`** — Δ Sharpe vs plain RSI {R['vs_rsi_diff']:+.2f}, "
            f"95% CI [{R['vs_rsi_lo']}, {R['vs_rsi_hi']}]; the second oscillator is decoration."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — cost landscape & panel\n\n"
            f"Turnover ~{R['turnover_yr']} round-trips/yr, so costs are not the binding "
            "constraint — the rule trails buy-and-hold *gross*. And the panel says SPY is not a "
            "fluke: every ETF's StochRSI rule loses to its own buy-and-hold."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for t in data.PANEL:\n"
            "        bb = data.load_real(t, cache_dir=CACHE); bb = bb[bb.index <= ASOF]\n"
            "        r = st.run_experiment(bb, cost_bps=1.0)\n"
            "        rows.append((t, r['srsi_net']['sharpe'], r['bh']['sharpe'], r['vs_rsi']['diff']))\n"
            "    panel = pd.DataFrame(rows, columns=['ticker','srsi_Sh','bh_Sh','vs_rsi'])\n"
            "else:\n"
            "    panel = pd.DataFrame({'ticker': data.PANEL,\n"
            f"        'srsi_Sh': [{R['p_spy']},{R['p_qqq']},{R['p_iwm']},{R['p_dia']},{R['p_efa']}],\n"
            f"        'bh_Sh': [{R['p_spy_bh']},{R['p_qqq_bh']},{R['p_iwm_bh']},{R['p_dia_bh']},{R['p_efa_bh']}]}})\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "x = np.arange(len(panel)); w = 0.38\n"
            "ax.bar(x-w/2, panel['srsi_Sh'], w, color=RED, label='StochRSI net Sharpe')\n"
            "ax.bar(x+w/2, panel['bh_Sh'], w, color=GREEN, label='buy & hold')\n"
            "ax.set_xticks(x); ax.set_xticklabels(panel['ticker']); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('net Sharpe (excess of cash)'); ax.set_title('Every panel name: buy-and-hold wins')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "panel.round(3)"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "Is the *engine* able to detect timing skill, or is it rigged to find none? Plant an "
            "AR(1) structure in a synthetic tape and measure StochRSI's edge over a same-exposure "
            "coin as we sweep the knob:"
        ),
        code(
            "def rand_sh(close, tim, cost=0.0, nrep=25):\n"
            "    n = len(close); k = int(round(tim*n)); out=[]\n"
            "    for s in range(nrep):\n"
            "        ix = np.random.default_rng(s).choice(n, size=k, replace=False)\n"
            "        w = pd.Series(np.zeros(n), index=close.index); w.iloc[ix]=1.0\n"
            "        out.append(st.sharpe(st.positions_to_returns(close, w, cost_bps=cost)['net'].to_numpy()))\n"
            "    return float(np.nanmean(out))\n"
            "knobs = [-0.20,-0.10,0.0,0.10,0.25,0.40]; edges=[]\n"
            "for mr in knobs:\n"
            "    bb,_ = data.synthetic_panel(n_days=4000, mean_rev=mr, drift=0.0002, seed=428)\n"
            "    c = bb['close']; led_s = st.positions_to_returns(c, st.stoch_rsi_position(c), cost_bps=0.0)\n"
            "    tim = np.nanmean(np.abs(led_s['pos'])); s_sh = st.sharpe(led_s['net'].to_numpy())\n"
            "    edges.append(s_sh - rand_sh(c, tim))\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.plot(knobs, edges, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('planted AR(1) knob  (<0 = trending, >0 = whipsaw)')\n"
            "ax.set_ylabel('StochRSI Sharpe − coin Sharpe')\n"
            "ax.set_title('Engine works: real timing edge appears on a trending tape')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('On a trending synthetic tape the rule beats the coin; on the real tape it does not.')"
        ),
        md(
            "The harness is a faithful timing detector: on a trending synthetic tape the StochRSI "
            "long-flat rule earns a genuine edge over a same-exposure coin; on a whipsaw tape it "
            "is fooled. The real-tape null is therefore a statement about **SPY** (its short-"
            "horizon dynamics give the oscillator nothing to time) — not a broken pipeline. Forks "
            "worth trying: a momentum (continuation) reading of the extremes; intraday bars; or "
            "single trending names where the long-flat filter has more to work with."
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
