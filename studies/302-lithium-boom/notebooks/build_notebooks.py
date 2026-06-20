"""Generate the two narrative notebooks for Study 302 (Lithium-Boom).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached daily
parquet under ../_cache/ if present and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md), so the notebook re-runs for any reader. Synthetic-only cells
say so in their title; real numbers live in one place (``R``).
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
    asof="2026-06-17",
    # LIT headline (200d trend overlay vs B&H, 10bps/flip, rf=1.5%)
    lit_bh_cagr=7.5, lit_bh_sharpe=0.34, lit_bh_dd=-66, lit_bh_t=1.36,
    lit_g_cagr=10.4, lit_g_sharpe=0.55, lit_g_dd=-39, lit_g_t=2.24,
    lit_n_cagr=9.7, lit_n_sharpe=0.51, lit_n_dd=-41, lit_n_t=2.09,
    lit_ci_lo=0.02, lit_ci_hi=0.99, bh_ci_lo=-0.17, bh_ci_hi=0.85,
    lit_flips_yr=6.5, lit_expo=43,
    # Honest pin — (Trend - B&H) difference
    lit_diff_ann=-0.33, lit_diff_t=-0.06,
    alb_diff_ann=-5.50, alb_diff_t=-0.76, alb_sharpe=0.43,
    sqm_diff_ann=-4.13, sqm_diff_t=-0.55, sqm_sharpe=0.39,
    # Random-timing control percentiles
    lit_rand_mean=-0.45, alb_rand_mean=-0.09, sqm_rand_mean=-0.14, rand_pctile=100,
    # Opportunity cost
    spy_sharpe=0.77,
    # Synthetic positive control (trend_strength sweep)
    syn_rows=[(0.00, 0.40, -86, 0.38, -61, -0.02, -0.69),
              (0.05, 0.41, -87, 0.51, -58, +0.10, -0.20),
              (0.10, 0.42, -88, 0.51, -60, +0.09, -0.27),
              (0.15, 0.43, -89, 0.55, -57, +0.12, -0.16)],
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

from lithium_boom import data, strategy as st

ASOF = "2026-06-17"
RF = 0.015
RF_D = RF / st.TRADING_DAYS_PER_YEAR
THEME = data.THEME_TICKERS  # LIT, ALB, SQM

def _have_cache():
    return all(os.path.exists(data._cache_path(t, data.DEFAULT_CACHE))
               for t in data.ALL_TICKERS)

HAVE_REAL = _have_cache()

def load(t):
    b = data.fetch_daily(t, fetch=False)
    return b[b.index <= ASOF]

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Lithium-Boom — did riding the battery-metals boom pay, or just whipsaw you? 🔋\n"
            "### A trend-follower on the lithium theme (LIT), tested honestly, in plain English\n\n"
            "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Did_riding_the_boom_pay%3F: Misattributed](https://img.shields.io/badge/Did_riding_the_boom_pay%3F-Misattributed-8b949e?style=flat-square)\n\n"
            "Around 2021 the pitch was everywhere: lithium is *the new oil*, the battery-metals "
            "**super-cycle** is a structural multi-year bull market, and a simple trend-follower "
            "riding the **LIT** ETF would have caught the boom *and* stepped aside before the bust. "
            "Then the metal roughly tripled and gave most of it back. This notebook asks the only "
            "two questions that matter: *did the trend rule carry real information — and did it "
            "actually pay you, or just hand you a smaller scare?*\n\n"
            "> **This is the plain-language layer.** Want the t-stats, the apples-to-apples race "
            "and the bootstrap CI? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did the trend rule know *something*? | **Yes.** Its timing crushes a coin that "
            f"trades just as often (the trend sits at the **{R['rand_pctile']}th percentile** of "
            "random timing). The signal is real. |\n"
            "| Did it ride the boom for *return*? | **No.** Versus simply holding LIT, the overlay "
            f"added **~0%/yr** (difference too small to measure). What it really did was **halve "
            f"the drawdown** ({R['lit_bh_dd']}% → {R['lit_n_dd']}%). |\n"
            "| So would you be richer? | **Barely — and not the way you were sold.** You got a "
            "calmer ride, not a bigger pile. And the whole battery-metals sleeve still earned "
            f"**worse** risk-adjusted returns than just owning the S&P 500. |\n"
            "| So what is it? | A real *crash-dodging* tool dressed up as a *boom-riding* one — "
            "useful for sleep, useless as the get-rich story the cycle was marketed as. |\n\n"
            "> The trend rule isn't fake — it genuinely times the theme. But '*ride the boom*' is "
            "the wrong frame: it paid you in a smaller drawdown, not in the runaway return the "
            "super-cycle pitch promised."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Electrification is structural: lithium and battery materials are in a multi-year "
            "super-cycle. Own the theme through **LIT** (or the miners), follow the trend, and "
            "you'll ride the boom — and because you follow the trend, you'll be out before the "
            "bust.\"*\n\n"
            "— the 2021–22 battery-metals super-cycle narrative (Global X fund literature; "
            "Goldman / Benchmark Mineral Intelligence cycle notes)\n\n"
            "The mechanical version we test is the oldest trend rule there is (Faber 2007): **hold "
            "the asset when its price is above its 200-day moving average; otherwise sit in cash.** "
            "It rests on something real — *time-series momentum* (Moskowitz-Ooi-Pedersen 2012): an "
            "asset's own trend tends to persist. The question is what that buys you on a single, "
            "wild commodity theme."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two things make this worth a careful look. First, LIT is a textbook **boom-bust**: it "
            "roughly tripled into 2021–22 then fell ~65%. If trend-following *ever* earns its keep, "
            "it should be here — a long, smooth ride up and a long, grinding ride down. Second, the "
            "pitch quietly blends two very different promises: '*you'll ride the boom*' (a **return** "
            "claim) and '*you'll dodge the bust*' (a **risk** claim). Those are not the same thing, "
            "and almost every 'trend timing' write-up you'll read leans on the first while only the "
            "second is actually true. Untangling them is the lesson that transfers."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three checks keep us honest:\n\n"
            "1. **Race it against just holding the thing.** A timing overlay is only interesting if "
            "it beats buy-and-hold. Because it sits in cash part-time, we compare them *fairly* "
            "(both measured against cash).\n"
            "2. **Race it against a coin that trades as often.** Place the same number of in/out "
            "moves on *random* days. If the trend rule beats that, the *timing* — not just being in "
            "cash sometimes — carried information.\n"
            "3. **Separate 'calmer' from 'richer'.** Measure the drawdown *and* the return-vs-hold "
            "difference, side by side, so a smaller scare can't masquerade as a bigger pile.\n\n"
            "Tape: LIT total-return daily bars from the fund's 2010 launch (plus miners ALB, SQM, "
            f"and SPY as the benchmark). As-of {R['asof']}. Costs: 10 bps one-way on every flip."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the picture everyone remembers — the boom and the bust, and where the trend "
            "rule was in vs out:**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load('LIT'); sig = st.trend_signal(b, 200)\n"
            "    pos = sig.shift(1).fillna(0.0)\n"
            "    fig, ax = plt.subplots(figsize=(10, 5))\n"
            "    ax.plot(b.index, b['close'], c=GREY, lw=1.2, label='LIT (total-return)')\n"
            "    ax.fill_between(b.index, b['close'].min(), b['close'].max(),\n"
            "                    where=pos.to_numpy() > 0.5, color=GREEN, alpha=.12,\n"
            "                    step='post', label='trend rule IN')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('price (log)'); ax.legend(loc='upper left')\n"
            "    ax.set_title('LIT: the battery-metals boom and bust — green = trend rule in the market')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    b, _ = data.synthetic_theme(n_days=4000, trend_strength=0.10, seed=302)\n"
            "    sig = st.trend_signal(b, 200); pos = sig.shift(1).fillna(0.0)\n"
            "    fig, ax = plt.subplots(figsize=(10, 5))\n"
            "    ax.plot(b.index, b['close'], c=GREY, lw=1.2, label='SYNTHETIC boom-bust tape')\n"
            "    ax.fill_between(b.index, b['close'].min(), b['close'].max(),\n"
            "                    where=pos.to_numpy() > 0.5, color=GREEN, alpha=.12,\n"
            "                    step='post', label='trend rule IN')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('price (log)'); ax.legend(loc='upper left')\n"
            "    ax.set_title('SYNTHETIC TAPE (offline) — boom-bust caricature; real LIT shape is the same')\n"
            "    plt.tight_layout(); plt.show()\n"
            "print('Banner:', 'REAL LIT tape' if HAVE_REAL else 'SYNTHETIC fallback (offline)')"
        ),
        md(
            "The rule is in for the boom and steps out across the long bust — exactly the behaviour "
            "the pitch advertises. So far, so good. Now the two questions that decide whether it "
            "*paid*."
        ),
        md(
            "**Question one: did the timing beat a coin that trades just as often?** Replace the "
            "trend signal with random in/out days of the same average exposure:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for t in THEME:\n"
            "        bb = load(t); ss = st.trend_signal(bb, 200)\n"
            "        bt = st.backtest(bb, ss, cost_bps=10.0, rf_annual=RF)\n"
            "        expo = bt['pos'].mean()\n"
            "        tr = st.sharpe(bt['strat_net'].to_numpy(), rf_daily=RF_D)\n"
            "        rnd = [st.sharpe(st.backtest(bb, st.random_timing_signal(len(bb), expo, seed=s),\n"
            "               cost_bps=10.0, rf_annual=RF)['strat_net'].to_numpy(), rf_daily=RF_D)\n"
            "               for s in range(150)]\n"
            "        rows.append((t, tr, np.nanmean(rnd)))\n"
            "else:\n"
            f"    rows = [('LIT', {R['lit_n_sharpe']}, {R['lit_rand_mean']}),\n"
            f"            ('ALB', {R['alb_sharpe']}, {R['alb_rand_mean']}),\n"
            f"            ('SQM', {R['sqm_sharpe']}, {R['sqm_rand_mean']})]\n"
            "labels = [r[0] for r in rows]; trs = [r[1] for r in rows]; rnds = [r[2] for r in rows]\n"
            "x = np.arange(len(labels)); w = 0.35\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "ax.bar(x - w/2, trs, w, color=GREEN, label='trend timing')\n"
            "ax.bar(x + w/2, rnds, w, color=GREY, label='random timing (same exposure)')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('net Sharpe (excess of cash)'); ax.legend()\n"
            "ax.set_title('Trend timing beats a coin that trades as often — the signal is real')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Random timing is a disaster (you buy dips that keep dipping); trend is far above it.')"
        ),
        md(
            f"Random timing **destroys** you (mean Sharpe ~{R['lit_rand_mean']} on LIT) — you keep "
            "buying dips that keep dipping. The trend rule sits at the very top of that "
            f"distribution (the **{R['rand_pctile']}th percentile**). So the *when* genuinely "
            "carried information. **The signal is real.**"
        ),
        md(
            "**Question two — the catch: was that information worth any extra *return* over just "
            "holding LIT?** Here's the difference between the overlay and buy-and-hold, and the "
            "drawdown each suffered:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load('LIT'); sig = st.trend_signal(b, 200)\n"
            "    bt = st.backtest(b, sig, cost_bps=10.0, rf_annual=RF)\n"
            "    diff_ann = np.nanmean(bt['strat_net'].to_numpy() - bt['asset_ret'].to_numpy())*252*100\n"
            "    bh_dd = st.max_drawdown(bt['asset_ret'].to_numpy())*100\n"
            "    tr_dd = st.max_drawdown(bt['strat_net'].to_numpy())*100\n"
            "else:\n"
            f"    diff_ann, bh_dd, tr_dd = {R['lit_diff_ann']}, {R['lit_bh_dd']}, {R['lit_n_dd']}\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar(['Trend − Buy&Hold\\n(extra return)'], [diff_ann],\n"
            "       color=(GREEN if diff_ann > 0 else RED), width=.5)\n"
            "a1.axhline(0, c='k', lw=1); a1.set_ylabel('annualised %/yr')\n"
            "a1.set_ylim(-3, 3); a1.set_title(f'Extra return vs just holding: {diff_ann:+.1f}%/yr ≈ ZERO')\n"
            "a2.bar(['Buy & hold', 'Trend overlay'], [bh_dd, tr_dd], color=[RED, GREEN], width=.5)\n"
            "a2.axhline(0, c='k', lw=1); a2.set_ylabel('worst drawdown %')\n"
            "a2.set_title(f'But the drawdown halved: {bh_dd:.0f}% → {tr_dd:.0f}%')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Extra return vs hold: {diff_ann:+.2f}%/yr   Drawdown: {bh_dd:.0f}% -> {tr_dd:.0f}%')"
        ),
        md(
            f"There it is. Versus simply holding LIT, the overlay added **~{R['lit_diff_ann']:+.1f}%"
            f"/yr** — statistically **zero**. What it really did was cut the worst drawdown roughly "
            f"in half (**{R['lit_bh_dd']}% → {R['lit_n_dd']}%**). You didn't get *richer* from the "
            "boom; you got a *calmer ride* through the bust. That's a real and useful thing — but "
            "it is **not** the boom-riding return the super-cycle story sold."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** The trend timing beats a same-exposure coin at the "
            f"{R['rand_pctile']}th percentile on all three battery names, and LIT's in-market "
            f"return clears the bar (HAC *t* = +{R['lit_n_t']:.2f} net). The *when* knew something.\n"
            f"- **Tradability — Fragile.** Net Sharpe ~{R['lit_n_sharpe']:.1f}, a single thin "
            f"thematic sleeve — and it earns *worse* risk-adjusted returns than just holding the "
            f"S&P 500 ({R['spy_sharpe']:.2f}).\n"
            "- **Did riding the boom pay? — Misattributed.** The extra return over buy-and-hold is "
            "essentially zero. The benefit was a halved drawdown — *crash-dodging*, not "
            "*boom-riding*."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Costs aren't the killer here — the rule flips only a handful of times a year. The "
            "killer is the comparison it loses: **just holding the S&P 500.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    lit = st.sharpe(st.backtest(load('LIT'), st.trend_signal(load('LIT'),200),\n"
            "                    cost_bps=10.0, rf_annual=RF)['strat_net'].to_numpy(), rf_daily=RF_D)\n"
            "    spy = st.sharpe(load('SPY')['close'].pct_change().fillna(0).to_numpy(), rf_daily=RF_D)\n"
            "else:\n"
            f"    lit, spy = {R['lit_n_sharpe']}, {R['spy_sharpe']}\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "b = ax.bar(['LIT trend overlay\\n(the clever thing)', 'Just hold SPY\\n(the boring thing)'],\n"
            "           [lit, spy], color=[AMBER, GREEN], width=.5)\n"
            "ax.set_ylabel('Sharpe (excess of cash)'); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_title('All that timing — and the index still won')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'LIT trend overlay Sharpe {lit:.2f}  vs  SPY buy-and-hold {spy:.2f}')"
        ),
        md(
            f"The clever single-theme overlay lands at ~{R['lit_n_sharpe']:.1f}; doing nothing but "
            f"holding SPY scored ~{R['spy_sharpe']:.2f}. You took on concentrated battery-metals "
            "risk, traded actively, and ended up with a *worse* risk-adjusted result than the "
            "couch portfolio. That's the definition of **fragile**: real, but not worth the bother."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The parent rule, on broad indices.** [Study 116 — Faber-Timing](../../116-faber-timing/) "
            "tests the same 200-day SMA long-flat rule on the index itself — same finding family: "
            "great for drawdowns, thin on return.\n"
            "- **Where trend *does* (barely) earn its keep.** [Study 20 — Freight-Train](../../20-freight-train/) "
            "and [Study 31 — Trade-Winds](../../31-trade-winds/) run trend across *many* markets — "
            "diversification is the only thing that rescues a thin per-sleeve Sharpe. Lithium-Boom "
            "is the one-sleeve case, and shows exactly why one sleeve is fragile.\n"
            "- **Fork it.** Try a faster window, or add the trend sleeve to a 60/40 — does the "
            "drawdown-dodging help a *portfolio* even if it doesn't beat the asset? That's the "
            "honest use of a crash-dodging tool.\n\n"
            "*Think the boom 'paid'? Fork this, and show me the extra return over just holding the "
            "thing — measured against cash, after costs. The drawdown chart is real; the return "
            "chart is the one the pitch never prints.*"
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
            "# Lithium-Boom — a quantitative teardown 🔬\n"
            "### Real total-return tape · one-lag trend overlay · excess-vs-excess race · "
            "random-timing control · block-bootstrap CI · capacity\n\n"
            "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Did_riding_the_boom_pay%3F: Misattributed](https://img.shields.io/badge/Did_riding_the_boom_pay%3F-Misattributed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether a 200-day "
            "trend overlay on the battery-metals theme (LIT, ALB, SQM) carries real timing "
            "information, whether it beats buy-and-hold on a fair excess-of-cash race, and whether "
            "the boom-riding *return* the pitch promises actually shows up — separating it from the "
            "drawdown reduction that a long/flat rule delivers mechanically.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, `auto_adjust=True` "
            f"total-return, LIT from its 2010 launch; as-of {R['asof']}; the offline core and tests "
            "run on a deterministic synthetic boom-bust tape. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` | LIT trend net HAC *t* = **+{R['lit_n_t']:.2f}** (gross "
            f"+{R['lit_g_t']:.2f}); trend timing beats a same-exposure random control at the "
            f"**{R['rand_pctile']}th percentile** on LIT, ALB, SQM. |\n"
            f"| **Tradability** | `FRAGILE` | Net excess-Sharpe ~**{R['lit_n_sharpe']:.2f}**, "
            f"bootstrap 95% CI **[{R['lit_ci_lo']:.2f}, {R['lit_ci_hi']:.2f}]** barely clears zero; "
            f"trails SPY buy-and-hold (**{R['spy_sharpe']:.2f}**). |\n"
            f"| **Did riding the boom pay?** | `MISATTRIBUTED` | (Trend − B&H) daily return "
            f"difference HAC *t* = **{R['lit_diff_t']:+.2f}** (≈0); drawdown {R['lit_bh_dd']}% → "
            f"{R['lit_n_dd']}%. Risk reduction, not return. |\n\n"
            "> 💡 In plain words: the trend rule genuinely *times* the theme (it beats a coin), but "
            "what it delivers is a smaller drawdown, not the extra return the super-cycle story "
            "sold — and the whole sleeve loses to plain SPY anyway."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $P_t$ be the close and $s_t = \\mathbf{1}[P_t > \\mathrm{SMA}_{200}(P)_t]$ the "
            "trend signal; the position held over $t{+}1$ is $s_t$ (one lag). Let $r^{\\text{ov}}$ "
            "be the overlay's net daily return (asset when in, cash when out) and $r^{\\text{bh}}$ "
            "buy-and-hold.\n\n"
            "- **H₁ (signal).** $\\mathbb{E}[r^{\\text{ov}}] > 0$ at HAC $t \\ge 2$.\n"
            "- **H₂ (timing skill).** $\\text{Sharpe}(r^{\\text{ov}})$ exceeds a random-timing "
            "control of identical average exposure.\n"
            "- **H₃ (boom-riding return).** $\\mathbb{E}[r^{\\text{ov}} - r^{\\text{bh}}] > 0$ — "
            "the overlay *adds return* over holding the asset.\n"
            "- **H₄ (tradable sleeve).** Its excess-of-cash Sharpe rivals the broad index (SPY).\n\n"
            "We find **H₁ and H₂ supported**, **H₃ rejected** (the difference is ≈0; the benefit is "
            "a halved drawdown), and **H₄ rejected** (it trails SPY)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Time-series momentum is textbook (Moskowitz-Ooi-Pedersen 2012) and the 200-day rule is "
            "Faber (2007), whose own headline was *risk reduction, not excess return* — a nuance the "
            "super-cycle marketing dropped. The interesting content here isn't whether trend exists "
            "(it does) but the **misattribution**: a long/flat overlay cuts drawdowns *mechanically*, "
            "whether or not the underlying has exploitable persistence, so 'it dodged the crash' is "
            "guaranteed and proves nothing about skill. Naming that — and showing the return "
            "difference is zero while the drawdown halves — is worth more than the binary verdict."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** $P_t > \\mathrm{SMA}_{200}$; hold over $t{+}1$; one `shift(1)`, no "
            "second lag.\n"
            "- **Costs.** 10 bps **one-way on NAV** per flip (turnover = one-way × NAV).\n"
            "- **Racing.** Overlay sits in cash part-time → race **excess-of-cash vs "
            "excess-of-cash** (rf = 1.5%/yr); test the daily *difference* (Trend − B&H) directly.\n"
            "- **Controls.** (i) buy-and-hold the same asset; (ii) a random-timing path of the same "
            "average exposure; (iii) SPY as the opportunity-cost benchmark.\n"
            "- **Inference.** Newey-West HAC *t*; circular block-bootstrap 95% CI for the Sharpe "
            "(preserves vol clustering); the difference test for H₃.\n"
            "- **Positive control.** Deterministic synthetic boom-bust tape with tunable AR(1) "
            "persistence — isolates 'trend skill' from 'long/flat cuts drawdowns mechanically'.\n\n"
            f"Basket: LIT, ALB, SQM, benchmark SPY. As-of {R['asof']}."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline — overlay vs buy-and-hold, with inference\n\n"
            "LIT, 200-day trend overlay, net of 10 bps/flip, against simply holding LIT."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load('LIT'); sig = st.trend_signal(b, 200)\n"
            "    bt = st.backtest(b, sig, cost_bps=10.0, rf_annual=RF)\n"
            "    bh = st.summarize(bt['asset_ret'].to_numpy(), rf_daily=RF_D)\n"
            "    g = st.summarize(bt['strat_gross'].to_numpy(), rf_daily=RF_D)\n"
            "    nz = st.summarize(bt['strat_net'].to_numpy(), rf_daily=RF_D)\n"
            "    ci = st.block_bootstrap_sharpe_ci(bt['strat_net'].to_numpy(), rf_daily=RF_D)\n"
            "    bh_ci = st.block_bootstrap_sharpe_ci(bt['asset_ret'].to_numpy(), rf_daily=RF_D)\n"
            "    tab = pd.DataFrame([\n"
            "        ['B&H', bh['cagr']*100, bh['sharpe'], bh['max_dd']*100, bh['hac_t']],\n"
            "        ['Trend gross', g['cagr']*100, g['sharpe'], g['max_dd']*100, g['hac_t']],\n"
            "        ['Trend net', nz['cagr']*100, nz['sharpe'], nz['max_dd']*100, nz['hac_t']]],\n"
            "        columns=['book','CAGR%','exSharpe','maxDD%','HAC t'])\n"
            "else:\n"
            "    tab = pd.DataFrame([\n"
            f"        ['B&H', {R['lit_bh_cagr']}, {R['lit_bh_sharpe']}, {R['lit_bh_dd']}, {R['lit_bh_t']}],\n"
            f"        ['Trend gross', {R['lit_g_cagr']}, {R['lit_g_sharpe']}, {R['lit_g_dd']}, {R['lit_g_t']}],\n"
            f"        ['Trend net', {R['lit_n_cagr']}, {R['lit_n_sharpe']}, {R['lit_n_dd']}, {R['lit_n_t']}]],\n"
            "        columns=['book','CAGR%','exSharpe','maxDD%','HAC t'])\n"
            f"    ci = ({R['lit_ci_lo']}, {R['lit_ci_hi']}); bh_ci = ({R['bh_ci_lo']}, {R['bh_ci_hi']})\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "col = [GREY, AMBER, GREEN]\n"
            "ax.bar(tab['book'], tab['exSharpe'], color=col, width=.55)\n"
            "ax.errorbar(['Trend net'], [tab['exSharpe'].iloc[2]],\n"
            "            yerr=[[tab['exSharpe'].iloc[2]-ci[0]],[ci[1]-tab['exSharpe'].iloc[2]]],\n"
            "            fmt='none', ecolor='k', capsize=6, lw=1.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('Sharpe (excess of cash)')\n"
            "ax.set_title('Net Sharpe rises vs hold — but the CI nearly touches zero')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(2).to_string(index=False))\n"
            "print(f'Trend net Sharpe 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]  |  B&H CI [{bh_ci[0]:.2f}, {bh_ci[1]:.2f}]')"
        ),
        md(
            f"> 💡 In plain words: the net overlay clears the inference bar on its own returns (HAC "
            f"*t* = +{R['lit_n_t']:.2f}) — H₁ holds. But the Sharpe *improvement* over hold is "
            f"shaky: the bootstrap CI **[{R['lit_ci_lo']:.2f}, {R['lit_ci_hi']:.2f}]** barely clears "
            f"zero and overlaps B&H's **[{R['bh_ci_lo']:.2f}, {R['bh_ci_hi']:.2f}]** almost "
            "entirely. The drawdown, meanwhile, roughly halves."
        ),
        md(
            "### 4b · Does the *timing* beat a coin? (H₂)\n\n"
            "Replace the trend signal with a random 0/1 path of the *same* average exposure — "
            "isolating whether the *when* mattered, not just being in cash part-time."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in THEME:\n"
            "        bb = load(t); ss = st.trend_signal(bb, 200)\n"
            "        bt = st.backtest(bb, ss, cost_bps=10.0, rf_annual=RF); expo = bt['pos'].mean()\n"
            "        tr = st.sharpe(bt['strat_net'].to_numpy(), rf_daily=RF_D)\n"
            "        rnd = np.array([st.sharpe(st.backtest(bb, st.random_timing_signal(len(bb), expo, seed=s),\n"
            "               cost_bps=10.0, rf_annual=RF)['strat_net'].to_numpy(), rf_daily=RF_D)\n"
            "               for s in range(200)])\n"
            "        recs.append((t, tr, float(np.nanmean(rnd)), float((rnd < tr).mean()*100)))\n"
            "    ctrl = pd.DataFrame(recs, columns=['ticker','trend_Sh','rand_mean_Sh','pctile'])\n"
            "else:\n"
            "    ctrl = pd.DataFrame({'ticker':['LIT','ALB','SQM'],\n"
            f"        'trend_Sh':[{R['lit_n_sharpe']},{R['alb_sharpe']},{R['sqm_sharpe']}],\n"
            f"        'rand_mean_Sh':[{R['lit_rand_mean']},{R['alb_rand_mean']},{R['sqm_rand_mean']}],\n"
            f"        'pctile':[{R['rand_pctile']},{R['rand_pctile']},{R['rand_pctile']}]}})\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "x = np.arange(len(ctrl)); w = 0.35\n"
            "ax.bar(x - w/2, ctrl['trend_Sh'], w, color=GREEN, label='trend timing')\n"
            "ax.bar(x + w/2, ctrl['rand_mean_Sh'], w, color=GREY, label='random timing (mean)')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels(ctrl['ticker'])\n"
            "ax.set_ylabel('net Sharpe (excess of cash)'); ax.legend()\n"
            "ax.set_title('Trend timing sits at the 100th percentile of random timing — H₂ holds')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(2).to_string(index=False))"
        ),
        md(
            f"> 💡 In plain words: random timing averages ~{R['lit_rand_mean']} (you buy falling "
            f"knives); the trend rule is at the **{R['rand_pctile']}th percentile** on every name. "
            "The signal's information is real — and it is overwhelmingly about *getting out before "
            "the bust*, which is exactly where the drawdown reduction comes from."
        ),
        md(
            "### 4c · The decisive test — boom-riding *return*? (H₃)\n\n"
            "Test the daily difference (Trend − B&H) directly: a positive, significant mean would "
            "mean the overlay *adds return*; a zero mean means it only reshapes risk."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for t in THEME:\n"
            "        bb = load(t); ss = st.trend_signal(bb, 200)\n"
            "        bt = st.backtest(bb, ss, cost_bps=10.0, rf_annual=RF)\n"
            "        diff = bt['strat_net'].to_numpy() - bt['asset_ret'].to_numpy()\n"
            "        rows.append((t, np.nanmean(diff)*252*100, st.hac_tstat(diff)))\n"
            "    dft = pd.DataFrame(rows, columns=['ticker','diff_ann%','diff_HAC_t'])\n"
            "else:\n"
            "    dft = pd.DataFrame({'ticker':['LIT','ALB','SQM'],\n"
            f"        'diff_ann%':[{R['lit_diff_ann']},{R['alb_diff_ann']},{R['sqm_diff_ann']}],\n"
            f"        'diff_HAC_t':[{R['lit_diff_t']},{R['alb_diff_t']},{R['sqm_diff_t']}]}})\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "col = [GREEN if t > 2 else (RED if t < -2 else GREY) for t in dft['diff_HAC_t']]\n"
            "ax.bar(dft['ticker'], dft['diff_HAC_t'], color=col)\n"
            "ax.axhline(2, ls='--', c=GREY); ax.axhline(-2, ls='--', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('HAC t of (Trend − B&H) daily return'); ax.set_ylim(-3, 3)\n"
            "ax.set_title('Extra return over just holding the asset: indistinguishable from zero — H₃ rejected')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dft.round(2).to_string(index=False))"
        ),
        md(
            f"> 💡 In plain words: on LIT the difference is **{R['lit_diff_ann']:+.1f}%/yr** at HAC "
            f"*t* = **{R['lit_diff_t']:+.2f}** — zero. On the miners it's *negative*. The overlay "
            "does not add return; it swaps return for a smaller drawdown. **H₃ rejected** — the "
            "verdict's third axis."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — LIT trend net HAC *t* = +{R['lit_n_t']:.2f} (gross "
            f"+{R['lit_g_t']:.2f}); trend timing at the {R['rand_pctile']}th percentile of a "
            "same-exposure random control on all three names. H₁, H₂ hold.\n"
            f"- **Tradability `FRAGILE`** — net excess-Sharpe ~{R['lit_n_sharpe']:.2f}, bootstrap CI "
            f"[{R['lit_ci_lo']:.2f}, {R['lit_ci_hi']:.2f}] barely above zero; a single thin theme "
            f"sleeve trailing SPY ({R['spy_sharpe']:.2f}). H₄ rejected.\n"
            f"- **Did riding the boom pay? `MISATTRIBUTED`** — (Trend − B&H) HAC *t* = "
            f"{R['lit_diff_t']:+.2f}; drawdown {R['lit_bh_dd']}% → {R['lit_n_dd']}%. Crash-dodging, "
            "not boom-riding. H₃ rejected."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — cost sweep and the opportunity cost\n\n"
            "Costs are a rounding error (few flips); the binding constraint is the benchmark it "
            "loses to."
        ),
        code(
            "costs = [0.0, 5.0, 10.0, 25.0, 50.0]\n"
            "if HAVE_REAL:\n"
            "    b = load('LIT'); sig = st.trend_signal(b, 200)\n"
            "    sh = [st.sharpe(st.backtest(b, sig, cost_bps=c, rf_annual=RF)['strat_net'].to_numpy(),\n"
            "                    rf_daily=RF_D) for c in costs]\n"
            "    spy = st.sharpe(load('SPY')['close'].pct_change().fillna(0).to_numpy(), rf_daily=RF_D)\n"
            "else:\n"
            f"    sh = [0.55, 0.53, {R['lit_n_sharpe']}, 0.46, 0.36]; spy = {R['spy_sharpe']}\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.plot(costs, sh, 'o-', c=AMBER, lw=2, label='LIT trend net Sharpe')\n"
            "ax.axhline(spy, ls='--', c=GREEN, lw=2, label=f'SPY buy-and-hold ({spy:.2f})')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xlabel('round-trip-equivalent cost (bps/flip)')\n"
            "ax.set_ylabel('Sharpe (excess of cash)'); ax.legend()\n"
            "ax.set_title('Costs barely bend the line — but it sits below SPY the whole way')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Even at 0 bps the overlay ({sh[0]:.2f}) trails SPY ({spy:.2f}).')"
        ),
        md(
            f"> 💡 In plain words: unlike the desk's pure mirages, this doesn't die at the first "
            f"dollar of cost — it's genuinely *fragile*, not fake. But across the whole cost range "
            f"it stays below the SPY line ({R['spy_sharpe']:.2f}). A real edge that still loses to "
            "the couch portfolio: structurally unable to justify the concentration."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the *engine* a faithful trend detector, or does a long/flat rule always look good on "
            "a boom-bust? Plant a tunable AR(1) persistence on the *same* boom-bust envelope and "
            "watch the Sharpe edge appear only when there is real persistence — while the drawdown "
            "drops in **every** case (including the pure random walk)."
        ),
        code(
            "rows = []\n"
            "for ts in [0.0, 0.05, 0.10, 0.15]:\n"
            "    b, _ = data.synthetic_theme(n_days=4000, trend_strength=ts, seed=302)\n"
            "    sig = st.trend_signal(b, 200); bt = st.backtest(b, sig, cost_bps=10.0)\n"
            "    bh_sh = st.sharpe(bt['asset_ret'].to_numpy()); tr_sh = st.sharpe(bt['strat_net'].to_numpy())\n"
            "    bh_dd = st.max_drawdown(bt['asset_ret'].to_numpy())*100\n"
            "    tr_dd = st.max_drawdown(bt['strat_net'].to_numpy())*100\n"
            "    rows.append((ts, bh_sh, tr_sh, tr_sh - bh_sh, bh_dd, tr_dd))\n"
            "syn = pd.DataFrame(rows, columns=['trend_strength','BH_Sh','Trend_Sh','edge','BH_DD%','Trend_DD%'])\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.plot(syn['trend_strength'], syn['edge'], 'o-', c=GREEN, lw=2)\n"
            "a1.axhline(0, c='k', lw=1); a1.set_xlabel('planted AR(1) persistence')\n"
            "a1.set_ylabel('Sharpe edge over B&H'); a1.set_title('Sharpe edge appears only with real trend')\n"
            "a2.plot(syn['trend_strength'], syn['BH_DD%'], 'o-', c=RED, lw=2, label='B&H')\n"
            "a2.plot(syn['trend_strength'], syn['Trend_DD%'], 'o-', c=GREEN, lw=2, label='Trend')\n"
            "a2.set_xlabel('planted AR(1) persistence'); a2.set_ylabel('max drawdown %'); a2.legend()\n"
            "a2.set_title('...but the drawdown drops even at zero persistence')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(syn.round(2).to_string(index=False))"
        ),
        md(
            "The Sharpe edge over buy-and-hold is ~0 at the random walk and grows with planted "
            "persistence — so the engine is a faithful detector, and the real-tape Sharpe edge is a "
            "statement about the theme having *some* persistence. But the drawdown reduction is "
            "present at **zero** persistence too: it is a mechanical property of being out of the "
            "market during declines, not evidence of skill. That single chart is the whole "
            "misattribution.\n\n"
            "For the same rule on broad indices see [Study 116 — Faber-Timing](../../116-faber-timing/); "
            "for trend across many markets (where diversification rescues a thin sleeve), "
            "[Study 20 — Freight-Train](../../20-freight-train/) and "
            "[Study 31 — Trade-Winds](../../31-trade-winds/)."
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
